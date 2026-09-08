"""Intensity-based demultiplexing (channel-cycle) QC.

ASoVi assigns every frame to a channel *positionally*
(``frame i -> channels_name[i % cycle_len]``).  The PHASE of that cycle is not
fixed, though: ``demux_start_offset`` / ``channels_slip`` and the demux editor's
``demux_correction.json`` sidecar rotate it, and ``build_demux_correction`` (in
this module) is what preprocess actually asks for every frame's channel
(``DemuxCorrection.channel_at``) — the preview calls the same function, so what
the editor shows is what the run assigns.
Nothing in the .tif/.dcimg/.sifx inputs confirms that assignment — unlike the
Churchland WidefieldImager, ASoVi has no LED trigger / DAQ line to recover the
phase from.  So a single dropped frame silently phase-shifts every later frame
and swaps donor/source for the rest of the recording.  The only existing guard
(``preprocess`` warns when a *non-last* file's frame count is not a multiple of
``cycle_len``) sees file-boundary shifts but is blind to a drop *inside* a file.

When the cycled channels differ in brightness (blue vs violet excitation, or
different fluorophores) the phase leaves an intensity fingerprint: the per-frame
mean intensity oscillates with period ``cycle_len``, and a phase slip appears as
a shift in that oscillation.  This module detects it from per-frame mean
intensities alone — no trigger, no trial concept.

The **QC half** of this module is detection only: ``analyze_demux`` never
changes the demux — it reports a per-recording QC (the ASoVi analogue of
WidefieldImager's per-trial ``falseAlign`` flag) for the user to act on, and when
the channels are *not* intensity-separable it reports ``inconclusive`` rather
than raising a false alarm.  The **correction half** (``DemuxCorrection`` /
``build_demux_correction`` / ``suggest_correction``) applies that human's
decision: preprocess calls ``build_demux_correction`` before it allocates the
per-channel memmaps, and every frame is demuxed through it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d


@dataclass
class DemuxQC:
    """Result of :func:`analyze_demux`."""

    cycle_len: int
    n_frames: int
    conclusive: bool  # channels intensity-separable AND enough frames to judge
    separability: float  # eta^2 in [0, 1]: intensity variance explained by phase
    dominant_period: int | None  # strongest intensity period in frames (None if inconclusive)
    period_ok: bool  # dominant_period compatible with cycle_len
    phase_means: list[float]  # detrended folded mean per phase (len == cycle_len)
    phase_stds: list[float]
    per_file_offset: list[int]  # best channel shift per input file vs the start (0 = aligned)
    slip_detected: bool
    slip_frame: int | None  # approx global frame where the phase first shifts
    slip_offset: int | None = None  # channel shift (mod resolvable period) after the slip
    messages: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """No problem found (inconclusive counts as ``ok`` — no evidence of a fault)."""
        return not self.slip_detected and self.period_ok


def _folded(residual: np.ndarray, cycle_len: int, start: int = 0):
    """Mean/std of ``residual`` grouped by global phase ``(start + i) % cycle_len``."""
    n = residual.size
    phase = (np.arange(n) + start) % cycle_len
    means = np.zeros(cycle_len)
    stds = np.zeros(cycle_len)
    for p in range(cycle_len):
        sel = residual[phase == p]
        if sel.size:
            means[p] = float(sel.mean())
            stds[p] = float(sel.std())
    return means, stds


def _fold_by_phase(residual: np.ndarray, phase: np.ndarray, cycle_len: int):
    """Mean/std of ``residual`` grouped by an ARBITRARY per-frame phase.

    :func:`_folded` assumes the positional cycle; a corrected demux assigns
    phases that jump at file boundaries (``DemuxCorrection.phases``), so the
    corrected fingerprint has to be folded against that array instead.
    """
    means = np.zeros(cycle_len)
    stds = np.zeros(cycle_len)
    for p in range(cycle_len):
        sel = residual[phase == p]
        if sel.size:
            means[p] = float(sel.mean())
            stds[p] = float(sel.std())
    return means, stds


def _eta2_by_phase(residual: np.ndarray, phase: np.ndarray, cycle_len: int) -> float:
    """Chance-corrected η² of a fold (same measure ``analyze_demux`` reports).

    How much of the residual's variance the phase grouping explains: ~0 when the
    channels are indistinguishable (or mis-assigned), ~1 when each phase sits at
    its own intensity.  Used to put a number on "did the slip correction help".
    """
    n = residual.size
    ss_total = float(np.sum(residual**2))
    if n < 2 or ss_total <= 1e-12:
        return 0.0
    means, _ = _fold_by_phase(residual, phase, cycle_len)
    counts = np.array([int((phase == p).sum()) for p in range(cycle_len)])
    eta_raw = float(np.sum(counts * means**2)) / ss_total
    floor = (cycle_len - 1) / max(n - 1, 1)  # η² of grouping pure noise into c_len bins
    return max(0.0, (eta_raw - floor) / (1.0 - floor)) if floor < 1.0 else 0.0


def _best_shift(pattern: np.ndarray, ref: np.ndarray) -> tuple[int, float]:
    """Circular shift ``d`` in ``[0, C)`` that best aligns ``pattern`` to ``ref``.

    Returns ``(d, corr)`` where ``roll(pattern, d)`` matches ``ref`` best.
    """
    c_len = ref.size
    ref_c = ref - ref.mean()
    ref_norm = float(np.linalg.norm(ref_c))
    best_d, best_c = 0, -np.inf
    for d in range(c_len):
        p = np.roll(pattern, d)
        p_c = p - p.mean()
        denom = float(np.linalg.norm(p_c)) * ref_norm
        corr = float(p_c @ ref_c / denom) if denom > 1e-12 else 0.0
        if corr > best_c:
            best_c, best_d = corr, d
    return best_d, best_c


def analyze_demux(
    means,
    cycle_len: int,
    file_starts=None,
    *,
    sep_threshold: float = 0.05,
    min_cycles: int = 4,
) -> DemuxQC:
    """Check the positional channel cycle against per-frame mean intensities.

    Parameters
    ----------
    means : sequence of float
        Per-frame mean intensity, one value per *processed* frame, in
        acquisition order.
    cycle_len : int
        Configured channel-cycle length (``len(channels_name)``).
    file_starts : sequence of int, optional
        Global (processed-frame) index of each input file's first frame, so a
        slip can be attributed to a file boundary.  Defaults to ``[0]``.
    sep_threshold : float
        Minimum eta^2 for the channels to count as intensity-separable (below
        this the phase is not verifiable from intensity → ``conclusive=False``).
    min_cycles : int
        Minimum number of full cycles required to judge (else inconclusive).
    """
    means = np.asarray(means, dtype=np.float64).ravel()
    n = means.size
    c_len = int(cycle_len)
    if file_starts is None:
        file_starts = [0]
    file_starts = [int(s) for s in file_starts]

    def _inconclusive(msg: str) -> DemuxQC:
        return DemuxQC(
            cycle_len=c_len,
            n_frames=n,
            conclusive=False,
            separability=0.0,
            dominant_period=None,
            period_ok=True,
            phase_means=[float(means.mean())] * max(c_len, 1) if n else [0.0] * max(c_len, 1),
            phase_stds=[float(means.std())] * max(c_len, 1) if n else [0.0] * max(c_len, 1),
            per_file_offset=[0] * len(file_starts),
            slip_detected=False,
            slip_frame=None,
            messages=[msg],
        )

    if c_len <= 1:
        return _inconclusive("cycle_len=1: single channel, no demux to verify")
    if n < max(2 * c_len, min_cycles * c_len):
        return _inconclusive(
            f"only {n} frames (< {min_cycles} cycles of {c_len}): demux QC inconclusive"
        )

    # Detrend slow bleaching drift with a moving average over exactly two cycles,
    # so the per-phase oscillation cancels in the average and the residual keeps
    # only the cycle_len fingerprint.
    w = min(2 * c_len, n)
    trend = uniform_filter1d(means, size=w, mode="nearest")
    resid = means - trend

    # Sliding-window local folds. Separability is measured LOCALLY (mean per-window
    # eta^2), not from the global fold: a mid-recording phase slip cancels the
    # global fingerprint (the bright and dim halves average out) yet leaves every
    # coherent window's fingerprint intact — a global measure would call the very
    # recordings we care about "inconclusive".
    win = min(n, max(w, 8 * c_len, n // 8))  # enough samples/phase to beat the chance eta^2
    step = max(c_len, win // 2)
    # Reference fingerprint anchored on the recording start (frame 0 == the
    # assumed-correct phase); each segment is matched against it, so a non-zero
    # best shift means that segment drifted from the initial channel phase.
    ref, _ = _folded(resid[:win], c_len, start=0)

    win_centers: list[int] = []
    win_offsets: list[int] = []
    local_etas: list[float] = []
    for s in range(0, n - c_len + 1, step):
        e = min(s + win, n)
        seg = resid[s:e]
        if seg.size < 2 * c_len:
            continue
        fm, _ = _folded(seg, c_len, start=s)
        phase = (np.arange(seg.size) + s) % c_len
        cnts = np.array([(phase == p).sum() for p in range(c_len)])
        ss_total = float(np.sum(seg**2))
        eta_raw = float(np.sum(cnts * fm**2)) / ss_total if ss_total > 1e-12 else 0.0
        # Subtract the chance eta^2 of grouping pure noise into c_len bins so that
        # equal-brightness channels read as ~0 rather than a spurious floor.
        floor = (c_len - 1) / max(seg.size - 1, 1)
        adj = max(0.0, (eta_raw - floor) / (1.0 - floor)) if floor < 1.0 else 0.0
        local_etas.append(adj)
        d, _corr = _best_shift(fm, ref)
        win_centers.append((s + e) // 2)
        win_offsets.append(int((c_len - d) % c_len))

    separability = float(np.mean(local_etas)) if local_etas else 0.0
    conclusive = separability >= sep_threshold
    # Display fingerprint = the coherent start fold (the global fold cancels on a slip).
    phase_means, phase_stds = _folded(resid[:win], c_len, start=0)

    if not conclusive:
        qc = _inconclusive(
            f"channels not intensity-separable (eta^2={separability:.3f} < {sep_threshold:g}): "
            "demux phase not verifiable from intensity — rely on Quick Preview / channel config"
        )
        qc.separability = separability
        qc.phase_means = [float(v) for v in phase_means]
        qc.phase_stds = [float(v) for v in phase_stds]
        return qc

    # Dominant intensity period via short-lag autocorrelation (slip-invariant:
    # a phase shift leaves the period unchanged, only the phase flips).
    resid_c = resid - resid.mean()
    max_lag = min(n // 2, c_len * 3 + 1)
    ac = np.array([float(resid_c[: n - lag] @ resid_c[lag:]) for lag in range(max_lag)])
    ac0 = ac[0] if ac[0] > 1e-12 else 1.0
    ac = ac / ac0
    dominant_period = int(np.argmax(ac[1:max_lag]) + 1) if max_lag > 1 else None
    period_ok = dominant_period is None or (
        c_len % dominant_period == 0 or dominant_period % c_len == 0
    )

    # The intensity fingerprint only resolves the phase up to its OWN period: if a
    # sub-period divides cycle_len (e.g. GCaMP/jRGECO/GCaMP/jRGECO reads as a
    # period-2 brightness pattern) then shifts differing by a multiple of that
    # period are indistinguishable, and the degenerate best-shift would otherwise
    # flip between equivalent values and fake a slip. Resolve — and report —
    # offsets modulo the fingerprint period.
    resolvable_period = (
        dominant_period
        if (dominant_period and 1 <= dominant_period < c_len and c_len % dominant_period == 0)
        else c_len
    )
    win_offsets = [o % resolvable_period for o in win_offsets]

    # Per-file channel offset (mod the resolvable period) relative to the start.
    file_bounds = list(file_starts) + [n]
    per_file_offset: list[int] = []
    for i in range(len(file_starts)):
        s, e = file_bounds[i], file_bounds[i + 1]
        if e - s < c_len:
            per_file_offset.append(0)
            continue
        fm, _ = _folded(resid[s:e], c_len, start=s)
        d, _corr = _best_shift(fm, ref)
        per_file_offset.append(int((c_len - d) % c_len) % resolvable_period)

    # A slip = the windowed offset departs from the initial (assumed-correct)
    # phase and *stays* departed (persistence guards against a single noisy window).
    slip_detected = False
    slip_frame: int | None = None
    slip_offset: int | None = None
    if win_offsets:
        base_offset = win_offsets[0]
        for k in range(len(win_offsets)):
            if win_offsets[k] != base_offset and (
                k + 1 >= len(win_offsets) or win_offsets[k + 1] != base_offset
            ):
                slip_detected = True
                slip_frame = win_centers[k]
                slip_offset = (win_offsets[k] - base_offset) % resolvable_period
                break

    messages: list[str] = []
    if resolvable_period < c_len:
        messages.append(
            f"intensity fingerprint period {resolvable_period} < cycle_len {c_len} "
            f"(channels share brightness): phase resolved only modulo {resolvable_period}; "
            f"a slip by a multiple of {resolvable_period} is not detectable from intensity"
        )
    if not period_ok:
        messages.append(
            f"intensity period {dominant_period} incompatible with cycle_len {c_len}: "
            "verify channels_name length"
        )
    if slip_detected:
        messages.append(
            f"phase slip near frame {slip_frame}: frames were likely dropped → channel "
            "assignment shifts downstream. Verify frame alignment / split files per recording."
        )
    if period_ok and not slip_detected:
        messages.append(
            f"demux phase consistent (eta^2={separability:.3f}, period={dominant_period})"
        )

    return DemuxQC(
        cycle_len=c_len,
        n_frames=n,
        conclusive=True,
        separability=separability,
        dominant_period=dominant_period,
        period_ok=period_ok,
        phase_means=[float(v) for v in phase_means],
        phase_stds=[float(v) for v in phase_stds],
        per_file_offset=per_file_offset,
        slip_detected=slip_detected,
        slip_frame=slip_frame,
        slip_offset=slip_offset,
        messages=messages,
    )


def plot_demux_qc(
    qc: DemuxQC, means, file_starts=None, *, channel_labels=None, correction=None,
):
    """QC figure — 1x2 (width ratio 4:1): intensity trace (left, wide) + folded
    per-phase fingerprint (right, narrow) as bar + error-bar + per-frame dot plot.

    ``correction`` (a non-identity :class:`DemuxCorrection`, e.g. from
    ``channels_slip``) overlays the fingerprint the run will actually demux to,
    beside the positional one.  Both are then folded GLOBALLY: a per-file slip is
    exactly what a global fold cancels out, so the uncorrected bars collapse while
    the corrected bars stand up — the visual (and, via η², numeric) proof that the
    declared slip is the right one.  Without a correction the panel is unchanged
    (QC's own start-window fold).

    Returns a Matplotlib ``Figure``; the caller owns saving / closing it.
    """
    import matplotlib.pyplot as plt

    means = np.asarray(means, dtype=np.float64).ravel()
    n = means.size
    if file_starts is None:
        file_starts = [0]
    c_len = qc.cycle_len
    corr = (
        correction
        if correction is not None and not correction.is_identity()
        and int(correction.cycle_len) == c_len
        else None
    )

    fig, (ax_t, ax_p) = plt.subplots(
        1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [4, 1]}
    )

    # --- left (wide): per-frame mean intensity (subsampled for long recordings) ---
    stride = max(1, n // 4000)
    ax_t.plot(np.arange(0, n, stride), means[::stride], lw=0.6, color="#3070b0")
    for s in list(file_starts)[1:]:
        ax_t.axvline(s, color="0.6", lw=0.6, ls=":")
    if corr is not None:
        # shade the frame ranges the correction rotates — with channels_slip these
        # are exactly the files declared slipped, so it doubles as a check that the
        # slip list lines up with the file boundaries.
        for i, (s, e, d) in enumerate(corr.edits):
            if int(d) % c_len == 0:
                continue
            ax_t.axvspan(
                int(s), n if (e is None or int(e) < 0) else int(e),
                color="#f0a848", alpha=0.16, zorder=0,
                label="demux correction" if i == 0 else None,
            )
    if qc.slip_detected and qc.slip_frame is not None:
        ax_t.axvline(qc.slip_frame, color="#c0392b", lw=1.4, ls="--", label="phase slip")
    if ax_t.get_legend_handles_labels()[0]:
        ax_t.legend(loc="upper right", fontsize=8, frameon=False)
    ax_t.set_xlabel("frame")
    ax_t.set_ylabel("mean intensity")
    ax_t.set_title("Per-frame mean intensity")

    # --- right (narrow): folded per-phase fingerprint = bar + error-bar + dots ---
    xp = np.arange(c_len)
    resid = None
    if n >= 2 and c_len >= 1:
        w = max(1, min(2 * c_len, n))
        resid = means - uniform_filter1d(means, size=w, mode="nearest")

    def _dots(centers, phase, color, cap_scale=1.0):
        if resid is None:
            return
        rng = np.random.default_rng(0)  # deterministic jitter
        per_phase_cap = max(1, int(400 * cap_scale) // max(c_len, 1))
        for p in range(c_len):
            vals = resid[phase == p]
            if vals.size > per_phase_cap:
                vals = vals[:: max(1, vals.size // per_phase_cap)]
            jitter = (rng.random(vals.size) - 0.5) * (0.36 if corr is not None else 0.5)
            ax_p.scatter(centers[p] + jitter, vals, s=5, color=color, alpha=0.28,
                         linewidths=0, zorder=3)

    if corr is None:
        ax_p.bar(xp, qc.phase_means, yerr=qc.phase_stds, color="#4c9f70",
                 alpha=0.8, capsize=3, zorder=1)
        _dots(xp, np.arange(n) % c_len, "#20303a")
        panel_title = "Folded fingerprint"
    else:
        ph_pos = np.arange(n) % c_len
        ph_cor = corr.phases(n)
        m_pos, s_pos = _fold_by_phase(resid, ph_pos, c_len)
        m_cor, s_cor = _fold_by_phase(resid, ph_cor, c_len)
        eta_pos = _eta2_by_phase(resid, ph_pos, c_len)
        eta_cor = _eta2_by_phase(resid, ph_cor, c_len)
        ax_p.bar(xp - 0.2, m_pos, width=0.38, yerr=s_pos, color="#9aa5ad",
                 alpha=0.85, capsize=2, zorder=1,
                 label=f"positional (η²={eta_pos:.2f})")
        ax_p.bar(xp + 0.2, m_cor, width=0.38, yerr=s_cor, color="#4c9f70",
                 alpha=0.9, capsize=2, zorder=1,
                 label=f"corrected (η²={eta_cor:.2f})")
        _dots(xp - 0.2, ph_pos, "#5a646b", cap_scale=0.5)
        _dots(xp + 0.2, ph_cor, "#20303a", cap_scale=0.5)
        ax_p.legend(fontsize=6, frameon=False, loc="best")
        panel_title = "Folded fingerprint (global)"

    ax_p.axhline(0.0, color="0.5", lw=0.6)
    labels = channel_labels if channel_labels and len(channel_labels) == c_len else [
        f"phase {p}" for p in range(c_len)
    ]
    ax_p.set_xticks(xp)
    ax_p.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax_p.set_ylabel("detrended mean")
    ax_p.set_title(panel_title, fontsize=9)

    verdict = (
        "SLIP DETECTED"
        if qc.slip_detected
        else ("period mismatch" if not qc.period_ok else ("consistent" if qc.conclusive else "inconclusive"))
    )
    title = (
        f"Demux QC — {verdict}  (eta²={qc.separability:.2f}, "
        f"period={qc.dominant_period}, offsets/file={qc.per_file_offset})"
    )
    if corr is not None:
        title += (
            f"\ncorrection applied: start_offset={corr.start_offset}, edits={corr.edits}"
        )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90 if corr is not None else 0.94))
    return fig


# --------------------------------------------------------------------------- #
# Correction model (produced by the demux editor, consumed by preprocess)
# --------------------------------------------------------------------------- #

DEMUX_CORRECTION_FILENAME = "demux_correction.json"


@dataclass
class DemuxCorrection:
    """A demux phase correction: a global rotation plus ranged phase edits.

    The corrected channel of processed frame ``g`` is
    ``channels_name[(g + start_offset + Σ delta for (start, end, delta) in edits
    if start <= g < end) % cycle_len]``.  ``start_offset`` fixes a wrong frame-0
    phase; each ``(start, end, delta)`` edit advances the phase by ``delta`` for
    frames in ``[start, end)`` — ``end`` of ``None`` (or ``< 0``) means "to the
    end of the recording".  A permanent dropped-frame slip is a single
    ``[start, -1, delta]``; a bounded mis-demuxed stretch is ``[start, end, delta]``;
    overlapping ranges sum.  :func:`analyze_demux` *detects*; this is the
    *correction* the editor produces and preprocess consumes.  An identity
    correction leaves the positional demux unchanged.
    """

    cycle_len: int
    start_offset: int = 0
    edits: list[tuple[int, int | None, int]] = field(default_factory=list)

    @staticmethod
    def _covers(g: int, start: int, end: "int | None") -> bool:
        return start <= g and (end is None or int(end) < 0 or g < end)

    def offset_at(self, g: int) -> int:
        return self.start_offset + sum(
            d for s, e, d in self.edits if self._covers(g, s, e)
        )

    def channel_at(self, g: int) -> int:
        return int((g + self.offset_at(g)) % self.cycle_len)

    def phases(self, n: int) -> np.ndarray:
        """Corrected phase (channel index) for frames ``0..n-1`` (vectorized)."""
        n = int(n)
        off = np.full(n, int(self.start_offset), dtype=np.int64)
        for s, e, d in self.edits:
            s = max(int(s), 0)
            e = n if (e is None or int(e) < 0) else min(int(e), n)
            if s < e:
                off[s:e] += int(d)
        return (np.arange(n, dtype=np.int64) + off) % int(self.cycle_len)

    def is_identity(self) -> bool:
        c = int(self.cycle_len)
        return self.start_offset % c == 0 and all(int(d) % c == 0 for _, _, d in self.edits)

    def to_dict(self) -> dict:
        return {
            "cycle_len": int(self.cycle_len),
            "start_offset": int(self.start_offset),
            "edits": [
                [int(s), (None if (e is None or int(e) < 0) else int(e)), int(d)]
                for s, e, d in self.edits
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DemuxCorrection":
        edits: list[tuple[int, int | None, int]] = []
        for item in d.get("edits", []):
            if len(item) == 3:  # (start, end, delta)
                s, e, dd = item
                edits.append((int(s), (None if e is None else int(e)), int(dd)))
            elif len(item) == 2:  # legacy (frame, delta) == a slip to the end
                f, dd = item
                edits.append((int(f), None, int(dd)))
        return cls(
            cycle_len=int(d["cycle_len"]),
            start_offset=int(d.get("start_offset", 0)),
            edits=edits,
        )


def demux_correction_path(output_dir) -> Path:
    return Path(output_dir) / DEMUX_CORRECTION_FILENAME


def save_demux_correction(output_dir, correction: DemuxCorrection) -> Path:
    p = demux_correction_path(output_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(correction.to_dict(), indent=2), encoding="utf-8")
    return p


def load_demux_correction(output_dir, *, cycle_len: int | None = None) -> "DemuxCorrection | None":
    """Load ``demux_correction.json`` if present, else ``None``.

    If ``cycle_len`` is given and disagrees with the saved file, the file is
    ignored (the channel config changed since the correction was made).
    """
    p = demux_correction_path(output_dir)
    if not p.exists():
        return None
    try:
        corr = DemuxCorrection.from_dict(json.loads(p.read_text(encoding="utf-8")))
    except (ValueError, KeyError, OSError, TypeError):
        return None
    if cycle_len is not None and corr.cycle_len != int(cycle_len):
        return None
    return corr


def file_slip_edits(
    channels_slip, frame_counts, cycle_len: int
) -> list[tuple[int, int | None, int]]:
    """``config.channels_slip`` (per input file) → ``DemuxCorrection`` edits.

    ``channels_slip[i]`` is the phase the *i*-th input file's channel cycle
    starts at, relative to the others: ``[0, 1, 0]`` says the second file's
    frames are one step into the cycle (its first frame is ``channels_name[1]``,
    not ``channels_name[0]``).  Each non-zero entry becomes one edit spanning
    that file's global frame range, so it composes with ``start_offset`` and
    with any editor-made edits.  Slips past the end of ``frame_counts`` (more
    entries than files) are ignored.
    """
    edits: list[tuple[int, int | None, int]] = []
    if not channels_slip or int(cycle_len) <= 1:
        return edits
    start = 0
    for i, n in enumerate(frame_counts):
        end = start + int(n)
        delta = int(channels_slip[i]) % int(cycle_len) if i < len(channels_slip) else 0
        if delta:
            edits.append((start, end, delta))
        start = end
    return edits


def build_demux_correction(config, output_dir, frame_counts) -> DemuxCorrection:
    """The demux correction preprocess applies, from config + on-disk sidecar.

    Precedence (as documented on ``demux_start_offset``): a ``demux_correction.json``
    written by the demux editor wins outright; otherwise the correction is the
    config's global ``demux_start_offset`` plus the per-file ``channels_slip``
    edits.  Previews build the correction the same way so what they show is what
    preprocess will do.
    """
    cyc = int(config.cycle_len)
    corr = load_demux_correction(output_dir, cycle_len=cyc)
    if corr is not None:
        return corr
    return DemuxCorrection(
        cycle_len=cyc,
        start_offset=int(config.demux_start_offset),
        edits=file_slip_edits(config.channels_slip, frame_counts, cyc),
    )


def suggest_correction(qc: DemuxQC, *, start_offset: int = 0) -> DemuxCorrection:
    """Pre-populate a correction from a QC result (a detected slip → one edit).

    The edit undoes the detected post-slip channel shift.  For ``cycle_len`` > 2
    the shift is only resolvable modulo the intensity fingerprint period, so the
    user should confirm it against the live corrected fingerprint.
    """
    edits: list[tuple[int, int | None, int]] = []
    if qc.slip_detected and qc.slip_frame is not None and qc.slip_offset:
        delta = (qc.cycle_len - qc.slip_offset) % qc.cycle_len
        if delta:
            edits.append((int(qc.slip_frame), None, int(delta)))  # a slip persists to the end
    return DemuxCorrection(cycle_len=qc.cycle_len, start_offset=int(start_offset), edits=edits)


def export_corrected_tiffs(config, correction: "DemuxCorrection", out_dir, *, progress=None):
    """De-interleave the raw input frames into one TIFF per channel using the
    corrected assignment (``correction.channel_at``).

    The correction only changes which channel each frame belongs to, so this
    writes one stack per channel holding exactly the frames assigned to it under
    the current correction — a direct way to confirm each channel is homogeneous.
    Streams frame-by-frame (bounded memory).  ``progress(done, total)`` is called
    periodically.  Returns the written paths (empty channels are dropped).
    """
    import tifffile

    from .io import find_input_files, get_frame_count, iter_frames, resolve_exp_stem

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = find_input_files(Path(config.input_dir), config.input_format, config.input_order)
    exp = resolve_exp_stem(
        config.input_dir, config.exp_name, config.input_format, config.input_order
    )
    cyc = int(correction.cycle_len)
    names = config.channels_name
    paths = [out / f"{exp}_corrCh{c}_{names[c]}.tif" for c in range(cyc)]
    writers = [tifffile.TiffWriter(p, bigtiff=True) for p in paths]
    counts = [0] * cyc
    try:
        total = sum(get_frame_count(f) for f in files)
    except Exception:  # noqa: BLE001
        total = 0
    cap = config.max_frames
    g = 0
    try:
        for f in files:
            stop = False
            for frame in iter_frames(f):
                c = correction.channel_at(g)
                writers[c].write(np.ascontiguousarray(frame),
                                 photometric="minisblack", contiguous=True)
                counts[c] += 1
                g += 1
                if progress is not None and g % 50 == 0:
                    progress(g, total)
                if cap is not None and g >= cap:
                    stop = True
                    break
            if stop:
                break
    finally:
        for w in writers:
            w.close()
    if progress is not None:
        progress(g, total)
    kept = []
    for c, p in enumerate(paths):
        if counts[c] == 0:
            p.unlink(missing_ok=True)  # a channel that received no frames
        else:
            kept.append(p)
    return kept


def plot_correction_preview(
    means,
    cycle_len: int,
    correction: "DemuxCorrection",
    file_starts=None,
    channel_labels=None,
):
    """Editor preview of a candidate correction.

    Top: the intensity trace with file boundaries (grey) and edit frames (red).
    Bottom: the folded per-phase fingerprint under the CORRECTED assignment,
    computed over the first third vs the last third of the recording — a good
    correction makes the two match with strong contrast; an unfixed slip makes
    them disagree (or cancels the fold).
    """
    import matplotlib.pyplot as plt

    means = np.asarray(means, dtype=np.float64).ravel()
    n = means.size
    c_len = int(cycle_len)
    phases = correction.phases(n)

    w = max(1, min(2 * c_len, n))
    resid = means - uniform_filter1d(means, size=w, mode="nearest")
    third = max(n // 3, c_len)

    def fold(sl: slice) -> list[float]:
        r, p = resid[sl], phases[sl]
        return [float(r[p == k].mean()) if np.any(p == k) else 0.0 for k in range(c_len)]

    first, last = fold(slice(0, third)), fold(slice(max(n - third, 0), n))

    fig, (ax_t, ax_p) = plt.subplots(2, 1, figsize=(8.0, 5.2))
    stride = max(1, n // 4000)
    ax_t.plot(np.arange(0, n, stride), means[::stride], lw=0.6, color="#3070b0")
    for s in list(file_starts or [])[1:]:
        ax_t.axvline(s, color="0.6", lw=0.6, ls=":")
    # edits are (start, end, delta) -- unpacking them as pairs raised ValueError,
    # and the editor swallowed it into a status line, so the preview NEVER worked.
    for s, e, d in correction.edits:
        if int(d) % c_len == 0:
            continue
        ax_t.axvline(s, color="#c0392b", lw=1.3, ls="--")
        ax_t.axvspan(s, n if e is None or e < 0 else e, color="#c0392b", alpha=0.08, lw=0)
    ax_t.set_xlabel("frame")
    ax_t.set_ylabel("mean intensity")
    ax_t.set_title("Intensity trace — file boundaries (grey), edits (red)")

    x = np.arange(c_len)
    ax_p.bar(x - 0.2, first, 0.4, label="first third", color="#4c9f70")
    ax_p.bar(x + 0.2, last, 0.4, label="last third", color="#b07cc6")
    ax_p.axhline(0.0, color="0.5", lw=0.6)
    ax_p.legend(fontsize=8, frameon=False)
    labels = channel_labels if channel_labels and len(channel_labels) == c_len else [
        f"ch{p}" for p in range(c_len)
    ]
    ax_p.set_xticks(x)
    ax_p.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax_p.set_ylabel("detrended mean")
    ax_p.set_title("Corrected fingerprint: first vs last third (should match)")
    fig.tight_layout()
    return fig
