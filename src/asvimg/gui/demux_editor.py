"""Standalone GUI to inspect and correct a mis-demultiplexed channel cycle.

Separate from the pipeline dashboard.  ASoVi assigns each frame to a channel
positionally (``frame i -> channels_name[i % cycle_len]``); a wrong starting
phase or a dropped frame silently mis-assigns channels.  ``asvimg.demux_qc``
*detects* this from the per-frame mean-intensity fingerprint; this editor lets
the user *correct* it — set a global phase offset and/or mark phase-slip edits —
and writes ``demux_correction.json`` which preprocess then applies.

Run it standalone::

    uv run python -m asvimg.gui.demux_editor [config.yaml | config_dir | input_dir]

It reads the per-frame means preprocess persisted (``demux_means.npz``) for an
instant QC; if those are absent it reads the input frames itself.  The frame
inspector reads a small window of the raw frames on demand around a suspect
index so a slip can be pinned to the exact frame.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np

from asvimg import (
    DemuxCorrection,
    PipelineConfig,
    analyze_demux,
    default_output_dir,
    export_corrected_tiffs,
    file_slip_edits,
    find_input_files,
    get_frame_count,
    iter_frames_with_metadata,
    load_config,
    load_demux_correction,
    load_frames_by_indices,
    plot_correction_preview,
    save_demux_correction,
    suggest_correction,
)

from .figures import figure_to_rgba

_WIN = "dmx_primary"
_TEXREG = "dmx_texreg"
_STATUS = "dmx_status"
_OFFSET = "dmx_start_offset"
_EDITS = "dmx_edits_group"
_PREVIEW = "dmx_preview_panel"
_PREVIEW_TEX = "dmx_preview_tex"
_PREVIEW_IMG = "dmx_preview_img"
_INSPECT_CENTER = "dmx_inspect_center"
_INSPECT_STRIP = "dmx_inspect_strip"
_INSPECT_RADIUS = 4  # frames shown on each side of the suspect frame

# per-channel text colours so the assigned channel visibly changes at a slip
_CH_COLORS = [
    (110, 200, 110), (100, 160, 235), (235, 160, 90), (205, 120, 205),
    (210, 205, 100), (120, 205, 205), (235, 130, 130), (160, 160, 240),
]


# --------------------------------------------------------------------------- #
# Data loading (no dearpygui)
# --------------------------------------------------------------------------- #

def _load_means(config: PipelineConfig, output_dir: Path):
    """(means, file_starts): prefer the npz preprocess saved, else read frames."""
    npz = Path(output_dir) / "demux_means.npz"
    if npz.exists():
        try:
            z = np.load(npz)
            return z["means"].astype(np.float64), [int(s) for s in z["file_starts"]]
        except (OSError, KeyError, ValueError):
            pass
    means: list[float] = []
    file_starts: list[int] = []
    g = 0
    try:
        files = find_input_files(Path(config.input_dir), config.input_format, config.input_order)
    except FileNotFoundError:
        return np.asarray(means, dtype=np.float64), file_starts
    cap = config.max_frames
    for f in files:
        file_starts.append(g)
        stop = False
        for frame, _meta in iter_frames_with_metadata(f, dcimg_backend=config.dcimg_backend):
            means.append(float(frame.mean()))
            g += 1
            if cap is not None and g >= cap:
                stop = True
                break
        if stop:
            break
    return np.asarray(means, dtype=np.float64), file_starts


def _frame_file_bounds(config: PipelineConfig):
    """``[(global_start, path, count), ...]`` for mapping a global frame index
    to its input file + local index."""
    try:
        files = find_input_files(Path(config.input_dir), config.input_format, config.input_order)
    except FileNotFoundError:
        return []
    bounds = []
    acc = 0
    for f in files:
        try:
            c = int(get_frame_count(f))
        except Exception:  # noqa: BLE001 — a bad file must not break the inspector
            c = 0
        bounds.append((acc, f, c))
        acc += c
    return bounds


def _load_frames_around(bounds, center: int, radius: int):
    """Return ``[(global_index, frame2d|None), ...]`` for the window around ``center``."""
    if not bounds:
        return []
    total = bounds[-1][0] + bounds[-1][2]
    lo = max(0, center - radius)
    hi = min(total - 1, center + radius)
    by_file: dict[int, list[tuple[int, int]]] = {}
    for g in range(lo, hi + 1):
        for i, (start, _path, count) in enumerate(bounds):
            if count and start <= g < start + count:
                by_file.setdefault(i, []).append((g, g - start))
                break
    out: list[tuple[int, np.ndarray | None]] = []
    for i, pairs in by_file.items():
        path = bounds[i][1]
        try:
            imgs = load_frames_by_indices(path, [lc for _g, lc in pairs])
        except Exception:  # noqa: BLE001 — reading must not crash the editor
            imgs = [None] * len(pairs)
        out.extend((g, img) for (g, _lc), img in zip(pairs, imgs))
    out.sort(key=lambda t: t[0])
    return out


def _thumb_rgba(img: np.ndarray, size: int = 90) -> tuple[int, int, np.ndarray]:
    """Downsample a 2D frame to a small normalized RGBA thumbnail for dpg."""
    h, w = img.shape[:2]
    step = max(1, max(h, w) // size)
    small = img[::step, ::step]
    vmin, vmax = float(small.min()), float(small.max())
    g = (small - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(small)
    rgba = np.ones((*g.shape, 4), dtype=np.float32)
    rgba[:, :, 0] = rgba[:, :, 1] = rgba[:, :, 2] = g.astype(np.float32)
    return small.shape[1], small.shape[0], rgba.reshape(-1)


# --------------------------------------------------------------------------- #
# Editor
# --------------------------------------------------------------------------- #

class DemuxEditor:
    """Build (:meth:`build`) + run (:meth:`run`) the demux-correction window.

    ``build`` only creates widgets in the current dpg context (display-independent,
    smoke-testable); ``run`` owns the standalone context and render loop and
    returns the saved :class:`DemuxCorrection` (or ``None``).
    """

    def __init__(self, config: PipelineConfig, output_dir: str | Path | None = None) -> None:
        self.config = config
        self.out_dir = Path(output_dir) if output_dir else default_output_dir(
            Path(config.input_dir), config.output_format
        )
        self.cycle_len = config.cycle_len
        self.labels = [
            f"Ch{i} {config.channels_name[i]}·{config.channels_prop[i]}"
            for i in range(self.cycle_len)
        ]
        self.means, self.file_starts = _load_means(config, self.out_dir)
        self.n = int(self.means.size)

        existing = load_demux_correction(self.out_dir, cycle_len=self.cycle_len)
        if existing is not None:
            self.start_offset0 = existing.start_offset
            # store end as an int (-1 == "to the end") for the integer inputs
            self.edits = [[int(s), (-1 if e is None else int(e)), int(d)]
                          for s, e, d in existing.edits]
        else:
            self.start_offset0 = int(config.demux_start_offset)
            # No sidecar yet: seed the edits from the config's per-file
            # channels_slip.  A saved sidecar overrides channels_slip entirely,
            # so starting empty here would silently drop the declared slips.
            bounds = [*self.file_starts, self.n]
            counts = [bounds[i + 1] - bounds[i] for i in range(len(self.file_starts))]
            self.edits = [
                [int(s), int(e), int(d)]
                for s, e, d in file_slip_edits(
                    config.channels_slip, counts, self.cycle_len
                )
            ]

        self.result: DemuxCorrection | None = None
        self.should_close = False
        self._edit_rows: list[tuple[str, str, str, str]] = []  # (start, end, delta, group)
        self._row_seq = 0
        self._frame_bounds = None  # lazily built
        self._inspect_tex: list[str] = []
        self._inspect_shown: list[tuple[int, str, str]] = []  # (global, idx_tag, ch_tag)
        self._export_thread: threading.Thread | None = None
        self._export_status: str | None = None

    # ---- correction state ------------------------------------------------

    def _current_correction(self) -> DemuxCorrection:
        import dearpygui.dearpygui as dpg

        so = int(dpg.get_value(_OFFSET)) if dpg.does_item_exist(_OFFSET) else self.start_offset0
        self._sync_edits_from_widgets()
        return DemuxCorrection(
            self.cycle_len, start_offset=so % self.cycle_len,
            edits=[(int(s), (None if int(e) < 0 else int(e)), int(d)) for s, e, d in self.edits],
        )

    def _sync_edits_from_widgets(self) -> None:
        import dearpygui.dearpygui as dpg

        if not self._edit_rows:
            return
        rows = []
        for s_tag, e_tag, d_tag, grp in self._edit_rows:
            if dpg.does_item_exist(grp):
                rows.append([int(dpg.get_value(s_tag)), int(dpg.get_value(e_tag)),
                             int(dpg.get_value(d_tag))])
        self.edits = rows

    def _rebuild_edit_table(self) -> None:
        import dearpygui.dearpygui as dpg

        dpg.delete_item(_EDITS, children_only=True)
        self._edit_rows.clear()
        for s, e, d in self.edits:
            rid = self._row_seq
            self._row_seq += 1
            grp = f"dmx_row_{rid}"
            s_tag, e_tag, d_tag = f"dmx_s_{rid}", f"dmx_e_{rid}", f"dmx_d_{rid}"
            with dpg.group(horizontal=True, tag=grp, parent=_EDITS):
                dpg.add_text("start")
                dpg.add_input_int(tag=s_tag, default_value=int(s), width=80,
                                  min_value=0, min_clamped=True, step=0)
                dpg.add_text("end")
                dpg.add_input_int(tag=e_tag, default_value=int(e), width=80, step=0)
                dpg.add_text("+ph")
                dpg.add_input_int(tag=d_tag, default_value=int(d), width=60, step=0)
                dpg.add_button(label="inspect", user_data=s_tag, callback=self._on_inspect_edit,
                               width=64)
                dpg.add_button(label="x", user_data=grp, callback=self._on_remove_row, width=24)
            self._edit_rows.append((s_tag, e_tag, d_tag, grp))

    # ---- callbacks -------------------------------------------------------

    def _on_remove_row(self, sender, app_data, user_data) -> None:
        self._sync_edits_from_widgets()
        idx = next((i for i, (_s, _e, _d, g) in enumerate(self._edit_rows) if g == user_data), None)
        if idx is not None and idx < len(self.edits):
            self.edits.pop(idx)
        self._rebuild_edit_table()
        self._update_preview()

    def _on_add_edit(self, sender=None, app_data=None) -> None:
        self._sync_edits_from_widgets()
        self.edits.append([0, -1, 1])  # start 0, end -1 (to the end), +1 phase
        self._rebuild_edit_table()

    def _on_inspect_edit(self, sender, app_data, user_data) -> None:
        import dearpygui.dearpygui as dpg

        if dpg.does_item_exist(user_data):
            self._show_neighborhood(int(dpg.get_value(user_data)))

    def _on_analyze(self, sender=None, app_data=None) -> None:
        import dearpygui.dearpygui as dpg

        if not self.n:
            dpg.set_value(_STATUS, "No per-frame means (run preprocess first, or set input_dir).")
            return
        qc = analyze_demux(self.means, self.cycle_len, self.file_starts or [0])
        verdict = (
            "SLIP" if qc.slip_detected
            else ("period mismatch" if not qc.period_ok
                  else ("consistent" if qc.conclusive else "inconclusive"))
        )
        note = qc.messages[0] if qc.messages else ""
        dpg.set_value(
            _STATUS,
            f"QC: {verdict} | eta^2={qc.separability:.2f} period={qc.dominant_period} "
            f"slip_frame={qc.slip_frame} offset={qc.slip_offset}  —  {note}",
        )
        return qc

    def _on_suggest(self, sender=None, app_data=None) -> None:
        import dearpygui.dearpygui as dpg

        if not self.n:
            return
        qc = analyze_demux(self.means, self.cycle_len, self.file_starts or [0])
        corr = suggest_correction(qc, start_offset=int(dpg.get_value(_OFFSET)))
        self.edits = [[int(s), (-1 if e is None else int(e)), int(d)] for s, e, d in corr.edits]
        self._rebuild_edit_table()
        self._update_preview()
        self._on_analyze()
        if qc.slip_frame is not None:  # jump the inspector to the suspect frame
            self._show_neighborhood(int(qc.slip_frame))

    def _update_preview(self, sender=None, app_data=None) -> None:
        import dearpygui.dearpygui as dpg

        if not self.n:
            return
        corr = self._current_correction()
        try:
            fig = plot_correction_preview(
                self.means, self.cycle_len, corr, self.file_starts, self.labels
            )
            w, h, data = figure_to_rgba(fig)
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception as exc:  # noqa: BLE001 — preview must never crash the editor
            dpg.set_value(_STATUS, f"preview failed: {exc}")
            return
        if dpg.does_item_exist(_PREVIEW_TEX):
            for t in (_PREVIEW_IMG, _PREVIEW_TEX):
                if dpg.does_item_exist(t):
                    dpg.delete_item(t)
        dpg.add_static_texture(width=w, height=h, default_value=data,
                               tag=_PREVIEW_TEX, parent=_TEXREG)
        dpg.add_image(_PREVIEW_TEX, tag=_PREVIEW_IMG, parent=_PREVIEW)
        self._refresh_inspect_labels(corr)  # channel labels follow the correction (no re-read)

    # ---- frame inspector -------------------------------------------------

    def _on_inspect(self, sender=None, app_data=None) -> None:
        import dearpygui.dearpygui as dpg

        self._show_neighborhood(int(dpg.get_value(_INSPECT_CENTER)))

    def _on_add_edit_here(self, sender=None, app_data=None) -> None:
        import dearpygui.dearpygui as dpg

        center = int(dpg.get_value(_INSPECT_CENTER))
        self._sync_edits_from_widgets()
        self.edits = [e for e in self.edits if e[0] != center]  # replace an edit at the same start
        self.edits.append([center, -1, 1])  # a slip to the end (edit the 'end' to bound it)
        self.edits.sort(key=lambda e: e[0])
        self._rebuild_edit_table()
        self._update_preview()

    def _show_neighborhood(self, center: int, radius: int = _INSPECT_RADIUS) -> None:
        import dearpygui.dearpygui as dpg

        if not dpg.does_item_exist(_INSPECT_STRIP):
            return
        center = max(0, int(center))
        dpg.set_value(_INSPECT_CENTER, center)
        dpg.delete_item(_INSPECT_STRIP, children_only=True)
        for t in self._inspect_tex:
            if dpg.does_item_exist(t):
                dpg.delete_item(t)
        self._inspect_tex.clear()
        self._inspect_shown.clear()

        if self._frame_bounds is None:
            self._frame_bounds = _frame_file_bounds(self.config)
        frames = _load_frames_around(self._frame_bounds, center, radius)
        if not frames:
            dpg.add_text("input frames unavailable — the raw data must be present to inspect.",
                         parent=_INSPECT_STRIP)
            return

        corr = self._current_correction()
        for g, img in frames:
            with dpg.group(parent=_INSPECT_STRIP):
                if img is not None:
                    tw, th, data = _thumb_rgba(img)
                    tag = f"dmx_insp_tex_{g}"
                    if dpg.does_item_exist(tag):
                        dpg.delete_item(tag)
                    dpg.add_static_texture(width=tw, height=th, default_value=data,
                                           tag=tag, parent=_TEXREG)
                    self._inspect_tex.append(tag)
                    dpg.add_image(tag, width=90, height=int(90 * th / tw))
                else:
                    dpg.add_text("(unreadable)")
                idx_tag, ch_tag = f"dmx_insp_i_{g}", f"dmx_insp_c_{g}"
                is_center = g == center
                dpg.add_text(f"{'» ' if is_center else ''}f{g}", tag=idx_tag,
                             color=(240, 90, 90) if is_center else (200, 200, 200))
                ch = corr.channel_at(g)
                dpg.add_text(self.config.channels_name[ch], tag=ch_tag,
                             color=_CH_COLORS[ch % len(_CH_COLORS)])
                self._inspect_shown.append((g, idx_tag, ch_tag))

    def _refresh_inspect_labels(self, corr: DemuxCorrection) -> None:
        import dearpygui.dearpygui as dpg

        for g, _idx_tag, ch_tag in self._inspect_shown:
            if dpg.does_item_exist(ch_tag):
                ch = corr.channel_at(g)
                dpg.set_value(ch_tag, self.config.channels_name[ch])
                dpg.configure_item(ch_tag, color=_CH_COLORS[ch % len(_CH_COLORS)])

    # ---- save / cancel ---------------------------------------------------

    def _on_save(self, sender=None, app_data=None) -> None:
        self.result = self._current_correction()
        save_demux_correction(self.out_dir, self.result)
        self.should_close = True

    def _on_cancel(self, sender=None, app_data=None) -> None:
        self.result = None
        self.should_close = True

    def _on_export(self, sender=None, app_data=None) -> None:
        """De-interleave the raw frames into per-channel TIFFs under the current
        correction, on a background thread so the UI stays responsive."""
        import dearpygui.dearpygui as dpg

        if self._export_thread is not None and self._export_thread.is_alive():
            return
        corr = self._current_correction()
        dst = self.out_dir / "demux_corrected"
        self._export_status = "exporting corrected TIFFs..."
        dpg.set_value(_STATUS, self._export_status)

        def _work():
            try:
                paths = export_corrected_tiffs(
                    self.config, corr, dst,
                    progress=lambda d, t: setattr(self, "_export_status", f"exporting {d}/{t}..."),
                )
                self._export_status = f"exported {len(paths)} channel TIFF(s) -> {dst}"
            except Exception as exc:  # noqa: BLE001 — surface the failure in the status line
                self._export_status = f"export failed: {exc}"

        self._export_thread = threading.Thread(target=_work, daemon=True)
        self._export_thread.start()

    # ---- build / run -----------------------------------------------------

    def build(self) -> None:
        """Create the window in the current dpg context (no render loop)."""
        import dearpygui.dearpygui as dpg

        for tag in (_WIN, _TEXREG):
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        dpg.add_texture_registry(tag=_TEXREG)

        with dpg.window(tag=_WIN, label="Demux Correction Editor"):
            with dpg.group(horizontal=True):
                dpg.add_button(label="Analyze", callback=self._on_analyze)
                dpg.add_button(label="Suggest from slip", callback=self._on_suggest)
                dpg.add_spacer(width=20)
                dpg.add_text("Start offset:")
                dpg.add_input_int(tag=_OFFSET, default_value=self.start_offset0 % self.cycle_len,
                                  min_value=0, max_value=self.cycle_len - 1,
                                  min_clamped=True, max_clamped=True, width=90, step=1,
                                  callback=self._update_preview)
                dpg.add_spacer(width=20)
                dpg.add_button(label="Update Preview", callback=self._update_preview)
                dpg.add_button(label="Export corrected TIFF", callback=self._on_export)
                dpg.add_button(label="Save", callback=self._on_save, width=90)
                dpg.add_button(label="Cancel", callback=self._on_cancel, width=90)

            dpg.add_text(
                f"input: {self.config.input_dir}   |   channels: {', '.join(self.labels)}"
                f"   |   frames: {self.n}"
            )
            dpg.add_text("", tag=_STATUS)
            dpg.add_separator()

            with dpg.group(horizontal=True):
                with dpg.child_window(width=430, autosize_y=True):
                    dpg.add_text("Phase-slip edits  (range [start, end); end = -1 → to the end)")
                    dpg.add_group(tag=_EDITS)
                    dpg.add_button(label="+ Add edit", callback=self._on_add_edit)
                    dpg.add_separator()
                    dpg.add_text(
                        "Run preprocess once so the QC can flag a slip, then\n"
                        "'Suggest from slip' pre-fills an edit; use the frame\n"
                        "inspector below to pin the exact frame, then adjust\n"
                        "+phase until the first- and last-third bars match.",
                        wrap=360,
                    )
                with dpg.child_window(autosize_x=True, autosize_y=True):
                    dpg.add_group(tag=_PREVIEW)

            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text("Inspect around frame:")
                dpg.add_input_int(tag=_INSPECT_CENTER, default_value=0, width=110, step=1,
                                  min_value=0, min_clamped=True)
                dpg.add_text(f"(±{_INSPECT_RADIUS})")
                dpg.add_button(label="Show", callback=self._on_inspect)
                dpg.add_button(label="+ edit here", callback=self._on_add_edit_here)
            dpg.add_text("The assigned channel (coloured) should change at a real slip:")
            with dpg.group(horizontal=True, tag=_INSPECT_STRIP):
                pass

        self._rebuild_edit_table()
        if self.n:
            self._update_preview()
            qc = self._on_analyze()
            start = int(qc.slip_frame) if (qc is not None and qc.slip_frame is not None) else 0
            self._show_neighborhood(start)
        else:
            dpg.set_value(_STATUS, "No per-frame means found. Run preprocess first "
                                   "(it saves demux_means.npz) or set input_dir.")
            self._show_neighborhood(0)

    def run(self, *, title: str = "Demux Correction Editor") -> DemuxCorrection | None:
        """Standalone: own context + render loop. Returns the saved correction."""
        import dearpygui.dearpygui as dpg

        dpg.create_context()
        dpg.create_viewport(title=title, width=1400, height=880)
        self.build()
        dpg.set_primary_window(_WIN, True)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        last_status = None
        while dpg.is_dearpygui_running() and not self.should_close:
            if self._export_status is not None and self._export_status != last_status:
                dpg.set_value(_STATUS, self._export_status)  # background export progress
                last_status = self._export_status
            dpg.render_dearpygui_frame()
        dpg.destroy_context()
        return self.result


def open_demux_editor(
    config: PipelineConfig,
    output_dir: str | Path | None = None,
    *,
    title: str = "Demux Correction Editor",
) -> DemuxCorrection | None:
    """Open the standalone editor; returns the saved DemuxCorrection or None."""
    return DemuxEditor(config, output_dir).run(title=title)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def _load_config_arg(arg: str | None) -> PipelineConfig:
    """Resolve the CLI arg into a PipelineConfig.

    Accepts a YAML file, a config dir (db.yaml/ops.yaml), or a bare input dir.
    """
    if arg is None:
        return PipelineConfig()
    p = Path(arg)
    if p.is_dir() and not (p / "ops.yaml").exists() and not (p / "db.yaml").exists():
        return PipelineConfig(input_dir=str(p))  # a bare data directory
    return load_config(p)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    config = _load_config_arg(argv[0] if argv else None)
    corr = open_demux_editor(config)
    if corr is None:
        print("[demux-editor] cancelled — no correction saved.")
    elif corr.is_identity():
        print("[demux-editor] saved identity correction (positional demux unchanged).")
    else:
        print(f"[demux-editor] saved: start_offset={corr.start_offset}, edits={corr.edits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
