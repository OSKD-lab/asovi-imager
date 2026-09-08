"""ROI correlation maps (network on atlas + correlation matrix).

Factors the inline correlation block of ``run_pipeline_full.ipynb`` (the
final cell) into reusable, headless functions: ``compute_roi_correlations``
loads the ``roiSignals_{name}_*`` payloads written by ROI extraction and
returns Pearson correlation matrices; the ``plot_*`` helpers render them and
return matplotlib Figures (never ``plt.show``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .io import resolve_exp_stem
from .pca import load_payload

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from .atlas import ACCFv3
    from .config import PipelineConfig


@dataclass
class CorrResult:
    """Correlation result for one ``channels_name`` group.

    Attributes
    ----------
    name : channels_name group.
    C : (n_rois, n_rois) Pearson correlation matrix.
    roi_names : ROI names (rows/cols of ``C``).
    mode : ``"dff"`` or ``"raw"`` — which signal the correlation was computed from.
    """

    name: str
    C: np.ndarray
    roi_names: list[str]
    mode: str
    method: str = "raw"


CORR_METHODS = ("raw", "gsr", "partial")
_METHOD_LABEL = {"raw": "Pearson", "gsr": "GSR", "partial": "partial"}


def _regress_out_global(F: np.ndarray) -> np.ndarray:
    """Global Signal Regression: remove the across-ROI mean signal from each ROI.

    ``g(t)`` is the mean over ROIs (the brain-wide common mode that dominates
    widefield GCaMP); each ROI has its least-squares projection onto ``g``
    subtracted.  Row mean/scale is irrelevant to the downstream correlation, so
    the returned residuals are centred only.
    """
    Fc = F - F.mean(axis=1, keepdims=True)
    g = Fc.mean(axis=0)  # (T,) common component, already centred
    denom = float(g @ g)
    if denom <= 1e-12:
        return Fc
    beta = (Fc @ g) / denom  # (n_rois,)
    return Fc - np.outer(beta, g)


def _partial_correlation(F: np.ndarray) -> np.ndarray:
    """Partial correlation via the (shrinkage) precision matrix.

    Correlation between two ROIs with the linear contribution of *all* other
    ROIs removed — the standard fMRI "direct" functional-connectivity estimate.
    Ledoit-Wolf shrinkage keeps the inverse covariance well-conditioned when the
    number of time points is not >> the number of ROIs.

    Known property (deliberate, not an oversight): the shrinkage target is
    ``mu * I`` in the ROIs' RAW units, so this estimate is NOT scale-invariant --
    rescaling one ROI's amplitude changes its partial correlations. Standardizing
    the rows before inverting would make it scale-free (and closer to the exact
    partial correlation), but it would also move every previously published
    ``partial`` number, so the behaviour is kept as-is.
    """
    from sklearn.covariance import LedoitWolf

    # samples = time points, features = ROIs
    cov = LedoitWolf().fit(F.T).covariance_
    prec = np.linalg.pinv(cov)
    d = np.sqrt(np.diag(prec))
    denom = np.outer(d, d)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = -prec / denom
    C[~np.isfinite(C)] = 0.0
    np.fill_diagonal(C, 1.0)
    return C


def _correlation_matrix(F: np.ndarray, method: str) -> np.ndarray:
    """(n_rois, T) signal → (n_rois, n_rois) correlation matrix for ``method``."""
    if method not in CORR_METHODS:
        raise ValueError(f"corr_method must be one of {CORR_METHODS}, got {method!r}")
    if method == "gsr":
        F = _regress_out_global(F)
    elif method == "partial" and F.shape[0] >= 2 and F.shape[1] >= 2:
        return _partial_correlation(F)
    # A single ROI (an edited rois.csv with one row) makes corrcoef return a 0-d
    # array, and everything downstream -- the heatmap, the network, the CSV --
    # died on it. One ROI correlates with itself: that is a 1x1 matrix.
    return np.atleast_2d(np.corrcoef(F))


def _owning_group(remainder: str, all_names: list[str]) -> str | None:
    """Which configured group a ``roiSignals_<remainder>`` file belongs to.

    Group names may contain underscores, so ``roiSignals_GCaMP_GFAP_run01``
    could read as group ``GCaMP`` (stem ``GFAP_run01``) or group ``GCaMP_GFAP``
    (stem ``run01``).  The longest configured name that the remainder starts
    with (``name`` or ``name_…``) is the true owner.
    """
    cands = [
        g for g in all_names
        if remainder == g or remainder.startswith(g + "_")
    ]
    return max(cands, key=len) if cands else None


def _resolve_roi_signal_file(
    output_dir: Path, name: str, stem: str, ext: str, all_names: list[str]
) -> Path | None:
    """Locate the ``roiSignals`` file for ``name``, avoiding prefix collisions.

    Prefers the exact ``roiSignals_{name}_{stem}.{ext}`` (as written by ROI
    extraction); otherwise falls back to the newest glob match that actually
    belongs to ``name`` (not a longer sibling group).
    """
    exact = output_dir / f"roiSignals_{name}_{stem}.{ext}"
    if exact.exists():
        return exact

    suffix = f".{ext}"
    owned: list[Path] = []
    for p in output_dir.glob(f"roiSignals_{name}_*{suffix}"):
        remainder = p.name[len("roiSignals_"):-len(suffix)]
        if _owning_group(remainder, all_names) == name:
            owned.append(p)
    if not owned:
        return None
    return max(owned, key=lambda p: p.stat().st_mtime)


def compute_roi_correlations(
    config: "PipelineConfig",
    output_dir: Path,
    *,
    method: str | None = None,
) -> dict[str, CorrResult]:
    """Compute ROI correlation matrices for every ``channels_name`` group.

    For each group, loads the first matching ``roiSignals_{name}_*`` payload,
    prefers ``F_dff`` over ``F_raw``, and correlates the ROI signals.  ``method``
    (default ``config.corr_method``) selects how the brain-wide common signal is
    handled: ``"raw"`` (plain Pearson), ``"gsr"`` (global-signal-regressed), or
    ``"partial"`` (partial correlation).  Groups with no payload or signal are
    skipped.
    """
    output_dir = Path(output_dir)
    ext = config.output_format
    all_names = list(config.channel_groups())
    stem = resolve_exp_stem(
        config.input_dir, config.exp_name, config.input_format, config.input_order
    )
    method = method or config.corr_method
    results: dict[str, CorrResult] = {}

    for name in all_names:
        src = _resolve_roi_signal_file(output_dir, name, stem, ext, all_names)
        if src is None:
            continue
        payload = load_payload(src)
        mode = (
            "dff" if "F_dff" in payload
            else ("raw" if "F_raw" in payload else None)
        )
        if mode is None:
            continue

        F = np.asarray(payload[f"F_{mode}"], dtype=float)  # (n_rois, T)
        # Correlate the window the ROI plots show, not the whole timeline. The
        # first seconds are unstable illumination (start_initial_frames) -- a
        # transient shared by every pixel, which reads as brain-wide correlation:
        # independent ROIs came out at r = +0.37 on it. The plot already drops it;
        # the number the plot is captioned with has to drop it too.
        start, end = config.roi_plot_window(F.shape[1])
        F = F[:, start:end]
        # .mat char arrays are fixed-width, so a round-trip pads names with
        # trailing spaces ('VISp_R   '); strip so they match atlas ROI names.
        roi_names = [str(r).strip() for r in payload["roi_names"]]
        C = _correlation_matrix(F, method)
        results[name] = CorrResult(
            name=name, C=C, roi_names=roi_names, mode=mode, method=method
        )

    return results


def plot_correlation_matrix(corr: CorrResult) -> "Figure":
    """Render the correlation matrix as an RdBu_r heatmap. Returns the Figure.

    The trivial diagonal (self-correlation = 1) is masked to NaN *for display
    only* (``corr.C`` is left untouched) and the colour scale is set from the
    off-diagonal range, so structure stays visible even when off-diagonal
    values are small (e.g. partial correlation ~ ±0.05).
    """
    import matplotlib.pyplot as plt

    n = len(corr.roi_names)
    disp = np.array(corr.C, dtype=float)
    np.fill_diagonal(disp, np.nan)  # display-only; keeps corr.C valid

    off = disp[np.isfinite(disp)]
    vmax = float(np.abs(off).max()) if off.size else 1.0
    vmax = min(max(vmax, 1e-3), 1.0)  # symmetric, 0-centred scale

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("lightgray")  # the masked diagonal

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(disp, cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.roi_names, rotation=90, fontsize=6)
    ax.set_yticklabels(corr.roi_names, fontsize=6)
    ax.set_title(
        f"{corr.name} — corr matrix ({corr.mode}, "
        f"{_METHOD_LABEL.get(corr.method, corr.method)}, "
        f"scale ±{vmax:.2f})"
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def resolve_network_threshold(
    C: np.ndarray,
    threshold: "float | str" = "auto",
    edge_density: float = 0.2,
) -> float:
    """Turn a threshold spec into an absolute ``|r|`` cutoff.

    A numeric ``threshold`` is returned as-is (fixed absolute cutoff).
    ``"auto"`` uses *proportional (density) thresholding* — the standard,
    method-agnostic graph choice: keep the strongest ``edge_density`` fraction
    of off-diagonal edges by taking the ``(1 - edge_density)`` quantile of
    ``|r|``.  This adapts to each method's scale, so it works whether the
    matrix is raw (near 1), GSR, or partial (near 0).
    """
    if threshold != "auto":
        return float(threshold)
    off = np.abs(C[~np.eye(len(C), dtype=bool)])
    off = off[np.isfinite(off)]
    if off.size == 0:
        return 0.0
    density = min(max(float(edge_density), 1e-3), 1.0)
    return float(np.quantile(off, 1.0 - density))


def plot_correlation_network(
    corr: CorrResult,
    atlas: "ACCFv3 | None",
    *,
    threshold: "float | str" = "auto",
    edge_density: float = 0.2,
    positions: "dict | None" = None,
) -> "Figure":
    """Render the ROI correlation network overlaid on the atlas. Returns Figure.

    ``atlas=None`` (source-space ROIs, no registration) draws the network on a
    plain background instead of the atlas boundaries; every node's position must
    then come from ``positions`` (there are no atlas default coordinates to fall
    back on).

    Edges with ``|r| >= threshold`` are drawn between ROI point positions;
    red = positive, blue = negative, with width/alpha scaled by ``|r|``.
    ``threshold="auto"`` (default) picks the cutoff from the data so the top
    ``edge_density`` fraction of edges is shown (see
    :func:`resolve_network_threshold`).  ``positions`` maps ROI name -> (row,
    col); used for custom/edited ROIs whose names are not in the atlas.
    """
    import matplotlib.pyplot as plt

    auto = threshold == "auto"
    thr = resolve_network_threshold(corr.C, threshold, edge_density)

    if atlas is not None:
        roi_rc = np.asarray(atlas.point_rois, dtype=float)
        roi_lookup = {str(n).strip(): i for i, n in enumerate(atlas.point_roi_names)}
    else:
        roi_rc = np.empty((0, 2), dtype=float)
        roi_lookup = {}
    pos = {str(k).strip(): v for k, v in (positions or {}).items()}

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    fig, ax = plt.subplots(figsize=(7.6, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    if atlas is not None:
        atlas.draw_boundaries(ax, color=(0.6, 0.6, 0.6), linewidth=0.8)
    else:
        ax.invert_yaxis()  # image row convention (row 0 at top), no atlas to set it
        ax.set_aspect("equal")

    def _coord(name):
        n = name.strip()
        if n in pos:
            return pos[n]  # (row, col) from edited ROIs
        if n in roi_lookup:
            return roi_rc[roi_lookup[n]]
        return (np.nan, np.nan)

    coords = np.array([_coord(r) for r in corr.roi_names])
    n = len(corr.roi_names)
    # Collect the edges above threshold first, so width/alpha can be scaled by
    # their *relative* strength.  Absolute |r| would make partial-correlation
    # edges (|r| ~ 0.04) invisible; relative scaling renders every method well.
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            r = corr.C[i, j]
            if not np.isfinite(r) or abs(r) < thr:
                continue
            (y0, x0), (y1, x1) = coords[i], coords[j]
            if not (np.isfinite(x0) and np.isfinite(x1)):
                continue
            edges.append((abs(r), r, x0, y0, x1, y1))

    rmax = max((e[0] for e in edges), default=1.0)
    span = max(rmax - thr, 1e-9)
    vmax = max(rmax, 1e-3)
    cmap = plt.get_cmap("RdBu_r")
    norm = Normalize(vmin=-vmax, vmax=vmax)
    for ar, r, x0, y0, x1, y1 in edges:
        w = (ar - thr) / span  # 0 (at cutoff) .. 1 (strongest shown)
        ax.plot(
            [x0, x1], [y0, y1],
            color=cmap(norm(r)),
            alpha=0.6 + 0.4 * w,
            linewidth=0.6 + 2.4 * w, zorder=2,
        )
    ax.scatter(
        coords[:, 1], coords[:, 0],
        s=40, c="white", edgecolors="k", linewidths=0.7, zorder=3,
    )
    for (y, x), lbl in zip(coords, corr.roi_names):
        if np.isfinite(x):
            ax.text(x + 3, y, lbl, fontsize=6, color="k", zorder=4)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="correlation r")
    ax.set_xlim(0, atlas.shape_hw[1])
    ax.set_ylim(atlas.shape_hw[0], 0)
    ax.set_aspect("equal")
    ax.axis("off")
    cut = (
        f"|r| >= {thr:.2f} (top {edge_density * 100:.0f}%)"
        if auto else f"|r| >= {thr:.2f}"
    )
    ax.set_title(
        f"{corr.name} — network ({cut}, {corr.mode}, "
        f"{_METHOD_LABEL.get(corr.method, corr.method)})"
    )
    fig.tight_layout()
    return fig
