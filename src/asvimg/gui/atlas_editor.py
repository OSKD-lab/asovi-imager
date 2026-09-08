"""Standalone GUI to build a 2D top-view CCF atlas, with adjustable tilt.

Separate from the pipeline dashboard (own Dear PyGui context + render loop; never
imported by ``app.py``).  Load the Allen CCF volumes (``.npy`` by-index or raw
``.nrrd``), tilt the top-down projection about the AP / ML / DV axes (about the
volume centre) with a live preview, then write an ``ACCFv3``-readable atlas.

Run it standalone::

    uv run python -m asvimg.gui.atlas_editor \
        [annotation.npy|.nrrd] [template.npy|.nrrd] [structure_tree.csv]

See :mod:`asvimg.atlas_build`.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from ..atlas_build import DEFAULT_GENERATED_ATLAS

_DEF_OUT = str(DEFAULT_GENERATED_ATLAS)

_T = "atl_"
_WIN = _T + "win"
_TEXREG = _T + "texreg"
_PTEX = _T + "ptex"
_PIMG = _T + "pimg"
_STATUS = _T + "status"
_LOG = _T + "log"
_PROG = _T + "prog"


class AtlasEditor:
    """Build (:meth:`build`) + run (:meth:`run`) the atlas builder window."""

    def __init__(self, annotation: str | None = None, template: str | None = None,
                 structure_tree: str | None = None, out_path: str = _DEF_OUT,
                 *, preview_downsample: int = 4) -> None:
        from .. import ccf_data

        vols = ccf_data.find_ccf_volumes()
        tgt = ccf_data.resolve_download_dir()
        self.annotation = annotation or (str(vols[0]) if vols else str(tgt / ccf_data.ANNOTATION_NAME))
        self.template = template or (str(vols[1]) if vols else str(tgt / ccf_data.TEMPLATE_NAME))
        self.structure_tree = structure_tree or str(
            ccf_data.find_structure_tree() or (Path("resources/allenCCF") / ccf_data.STRUCTURE_TREE_NAME))
        self.out_path = out_path
        self.pd = preview_downsample
        self._dl_thread: threading.Thread | None = None
        self._dl_prog: tuple[int, int] | None = None

        self._st = None
        self._av_ds = None
        self._tv_ds = None
        self._loaded = False

        self._dirty = False
        self._preview = None          # (region_ids, n_regions) awaiting upload
        self._lock = threading.Lock()
        self._pending: list[str] = []
        self._status_pending: str | None = None
        self._save_thread: threading.Thread | None = None
        self.last_saved: Path | None = None
        self.should_close = False

    # ---- data (no dpg) ---------------------------------------------------

    def load_data(self) -> None:
        """Load the structure tree + a downsampled volume pair for fast preview."""
        import numpy as np

        from .. import atlas_build as ab

        self._st = ab.load_structure_tree(self.structure_tree)
        d = self.pd
        av = ab.load_ccf_volume(self.annotation, self._st)
        self._av_ds = np.asarray(av[::d, ::d, ::d])
        tv = ab.load_ccf_volume(self.template)
        self._tv_ds = np.asarray(tv[::d, ::d, ::d])
        self._loaded = True

    def _params(self) -> dict:
        import dearpygui.dearpygui as dpg

        def g(tag, default):
            return dpg.get_value(_T + tag) if dpg.does_item_exist(_T + tag) else default

        from ..atlas_build import _AP_CROP_DEFAULT, _ML_CROP_DEFAULT

        return dict(
            tilt_ap_deg=float(g("s_ap", 0.0)), tilt_ml_deg=float(g("s_ml", 0.0)),
            tilt_dv_deg=float(g("s_dv", 0.0)),
            out_hw=(int(g("i_h", 285)), int(g("i_w", 285))),
            min_pixels=int(g("i_min", 15)),
            include_olfactory=bool(g("c_olf", False)),
            include_cerebellum=bool(g("c_cb", False)),
            ap_crop=(float(g("cr_ap0", _AP_CROP_DEFAULT[0])), float(g("cr_ap1", _AP_CROP_DEFAULT[1]))),
            ml_crop=(float(g("cr_ml0", _ML_CROP_DEFAULT[0])), float(g("cr_ml1", _ML_CROP_DEFAULT[1]))),
        )

    def compute_preview(self, p: dict):
        """Tilt the cached downsampled volume, project, and label (no dpg)."""
        import numpy as np

        from .. import atlas_build as ab

        mem = self._st.surface_mask(include_olfactory=p["include_olfactory"],
                                    include_cerebellum=p["include_cerebellum"])
        v = ab.tilt_volume(self._av_ds, tilt_ap_deg=p["tilt_ap_deg"],
                           tilt_ml_deg=p["tilt_ml_deg"], tilt_dv_deg=p["tilt_dv_deg"])
        td = ab.top_down_index(v, mem)
        frame, _ = ab._to_atlas_frame(
            td, p["out_hw"],
            ap_crop=p.get("ap_crop", ab._AP_CROP_DEFAULT),
            ml_crop=p.get("ml_crop", ab._ML_CROP_DEFAULT))
        base = self._st.base
        bm = np.array([base[x] if x < len(base) else "" for x in frame.ravel()],
                      dtype=object).reshape(frame.shape)
        ids, names = ab.assign_ids(bm, min_pixels=p["min_pixels"])
        return ids, len(names) - 1

    # ---- render-loop helpers (main thread) -------------------------------

    def _upload_preview(self, ids, n_reg: int) -> None:
        import dearpygui.dearpygui as dpg
        import numpy as np
        from matplotlib import colormaps

        H, W = ids.shape
        norm = ids.astype(float) / max(int(ids.max()), 1)
        rgba = colormaps["tab20b"](norm).astype(np.float32)
        rgba[ids <= 1] = (0.11, 0.11, 0.18, 1.0)          # background
        # burn the default point ROIs (fixed frame) as red dots for reference
        try:
            from ..atlas import DEFAULT_POINT_ROIS
            sr, sc = H / 285.0, W / 285.0
            for r, c in DEFAULT_POINT_ROIS:
                for cc in (int(c * sc), W - 1 - int(c * sc)):
                    rr = int(r * sr)
                    if 0 <= rr < H and 0 <= cc < W:
                        rgba[max(0, rr - 1):rr + 2, max(0, cc - 1):cc + 2] = (1, 0, 0, 1)
        except Exception:  # noqa: BLE001
            pass

        for t in (_PIMG, _PTEX):
            if dpg.does_item_exist(t):
                dpg.delete_item(t)
        dpg.add_static_texture(width=W, height=H, default_value=rgba.reshape(-1),
                               tag=_PTEX, parent=_TEXREG)
        dpg.add_image(_PTEX, tag=_PIMG, parent=_T + "preview_panel", width=380,
                      height=int(380 * H / W))
        dpg.set_value(_STATUS, f"preview: {n_reg} regions  (tilt ap/ml/dv = "
                               f"{self._params()['tilt_ap_deg']:.0f}/"
                               f"{self._params()['tilt_ml_deg']:.0f}/"
                               f"{self._params()['tilt_dv_deg']:.0f}°)")

    def _drain(self) -> None:
        import dearpygui.dearpygui as dpg

        # recompute preview when parameters changed and the mouse is released
        if self._dirty and self._loaded and not dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
            self._dirty = False
            try:
                ids, n = self.compute_preview(self._params())
                self._upload_preview(ids, n)
            except Exception as exc:  # noqa: BLE001
                dpg.set_value(_STATUS, f"preview failed: {exc}")

        with self._lock:
            lines, self._pending = self._pending, []
            st, self._status_pending = self._status_pending, None
        if lines and dpg.does_item_exist(_LOG):
            cur = dpg.get_value(_LOG)
            dpg.set_value(_LOG, (cur + "\n" + "\n".join(lines)).strip() if cur else "\n".join(lines))
        if st is not None and dpg.does_item_exist(_STATUS):
            dpg.set_value(_STATUS, st)
        if self._dl_prog is not None and dpg.does_item_exist(_PROG):
            done, total = self._dl_prog
            dpg.configure_item(_PROG, default_value=(done / total if total else 0.0),
                               overlay=f"{done // 10**6}/{total // 10**6} MB")

    # ---- callbacks -------------------------------------------------------

    def _cb_param(self, *_a) -> None:
        self._dirty = True

    def _cb_load(self, *_a) -> None:
        import dearpygui.dearpygui as dpg

        # pick up path edits
        for tag, attr in [("p_annot", "annotation"), ("p_tmpl", "template"),
                          ("p_st", "structure_tree"), ("p_out", "out_path")]:
            if dpg.does_item_exist(_T + tag):
                setattr(self, attr, dpg.get_value(_T + tag))
        if not (Path(self.annotation).exists() and Path(self.template).exists()):
            dpg.set_value(_STATUS, "CCF volumes not found — click 'Download CCF' or set the paths above")
            return
        if not Path(self.structure_tree).exists():
            dpg.set_value(_STATUS, f"structure tree not found: {self.structure_tree}")
            return
        dpg.set_value(_STATUS, "loading volumes ...")
        try:
            self.load_data()
            dpg.set_value(_STATUS, f"loaded (downsample x{self.pd}); drag the tilt sliders")
            self._dirty = True
        except Exception as exc:  # noqa: BLE001
            dpg.set_value(_STATUS, f"load failed: {exc}")

    def _cb_download(self, *_a) -> None:
        import dearpygui.dearpygui as dpg

        if self._dl_thread is not None and self._dl_thread.is_alive():
            return
        from .. import ccf_data

        annot = dpg.get_value(_T + "p_annot") if dpg.does_item_exist(_T + "p_annot") else self.annotation
        target = Path(annot).parent if annot else ccf_data.resolve_download_dir()
        self._status(f"downloading CCF volumes (~4.8 GB) into {target} ... (leave the window open)")

        def _work():
            try:
                def prog(label, done, total):
                    self._dl_prog = (done, total)
                    with self._lock:
                        self._status_pending = (f"downloading {label}: {100 * done / max(total, 1):.0f}%  "
                                                 f"({done // 10**6}/{total // 10**6} MB)")

                ccf_data.download_ccf_volumes(target, reporter=prog)
                self._dl_prog = None
                self.annotation = str(target / ccf_data.ANNOTATION_NAME)
                self.template = str(target / ccf_data.TEMPLATE_NAME)
                self._log(f"CCF volumes ready in {target}")
                self._status(f"downloaded to {target} — click Load")
            except Exception as exc:  # noqa: BLE001
                self._dl_prog = None
                self._log(f"download ERROR: {exc}")
                self._status(f"download failed: {exc}")

        self._dl_thread = threading.Thread(target=_work, daemon=True)
        self._dl_thread.start()

    def _cb_reset_tilt(self, *_a) -> None:
        import dearpygui.dearpygui as dpg

        from ..atlas_build import _AP_CROP_DEFAULT, _ML_CROP_DEFAULT

        for t in ("s_ap", "s_ml", "s_dv"):
            if dpg.does_item_exist(_T + t):
                dpg.set_value(_T + t, 0.0)
        for t, v in (("cr_ap0", _AP_CROP_DEFAULT[0]), ("cr_ap1", _AP_CROP_DEFAULT[1]),
                     ("cr_ml0", _ML_CROP_DEFAULT[0]), ("cr_ml1", _ML_CROP_DEFAULT[1])):
            if dpg.does_item_exist(_T + t):
                dpg.set_value(_T + t, v)
        self._dirty = True

    def _cb_save(self, *_a) -> None:
        import dearpygui.dearpygui as dpg

        if self._save_thread is not None and self._save_thread.is_alive():
            return
        p = self._params()
        out = dpg.get_value(_T + "p_out") if dpg.does_item_exist(_T + "p_out") else self.out_path
        annot, tmpl, st = self.annotation, self.template, self.structure_tree
        self._status("building full-resolution atlas ... (this takes a bit)")

        def _work():
            try:
                from .. import atlas_build as ab

                arr = ab.build_atlas(annot, tmpl, st, out_hw=p["out_hw"],
                                     min_pixels=p["min_pixels"],
                                     include_olfactory=p["include_olfactory"],
                                     include_cerebellum=p["include_cerebellum"],
                                     tilt_ap_deg=p["tilt_ap_deg"], tilt_ml_deg=p["tilt_ml_deg"],
                                     tilt_dv_deg=p["tilt_dv_deg"],
                                     ap_crop=p["ap_crop"], ml_crop=p["ml_crop"])
                path = ab.save_atlas_mat(out, arr)
                self.last_saved = path
                self._log(f"wrote {path} ({len(arr.region_names) - 1} regions)")
                self._status(f"saved: {path}")
            except Exception as exc:  # noqa: BLE001
                self._log(f"ERROR: {exc}")
                self._status(f"save failed: {exc}")

        self._save_thread = threading.Thread(target=_work, daemon=True)
        self._save_thread.start()

    def _status(self, msg: str) -> None:
        with self._lock:
            self._status_pending = msg

    def _log(self, msg: str) -> None:
        with self._lock:
            self._pending.append(msg)

    def _cb_close(self, *_a) -> None:
        self.should_close = True

    # ---- build / run -----------------------------------------------------

    def build(self) -> None:
        import dearpygui.dearpygui as dpg

        for t in (_WIN, _TEXREG):
            if dpg.does_item_exist(t):
                dpg.delete_item(t)
        dpg.add_texture_registry(tag=_TEXREG)

        with dpg.window(tag=_WIN, label="Atlas Builder (tilt)"):
            dpg.add_text("Allen CCF volumes (.npy by-index or .nrrd):")
            dpg.add_input_text(tag=_T + "p_annot", default_value=self.annotation, width=680, label="annotation")
            dpg.add_input_text(tag=_T + "p_tmpl", default_value=self.template, width=680, label="template")
            dpg.add_input_text(tag=_T + "p_st", default_value=self.structure_tree, width=680, label="structure tree")
            dpg.add_input_text(tag=_T + "p_out", default_value=self.out_path, width=680, label="output atlas (.h5)")
            with dpg.group(horizontal=True):
                dpg.add_button(label="Load", callback=self._cb_load)
                dpg.add_button(label="Download CCF (~4.8GB)", callback=self._cb_download)
                dpg.add_button(label="Reset tilt+crop", callback=self._cb_reset_tilt)
                dpg.add_button(label="Save atlas (full-res)", callback=self._cb_save)
                dpg.add_button(label="Close", callback=self._cb_close)
            dpg.add_separator()

            with dpg.group(horizontal=True):
                with dpg.child_window(width=430, autosize_y=True):
                    dpg.add_text("Tilt about the centre (degrees):")
                    dpg.add_slider_float(tag=_T + "s_ap", label="AP axis (roll)", default_value=0.0,
                                         min_value=-30, max_value=30, callback=self._cb_param)
                    dpg.add_slider_float(tag=_T + "s_ml", label="ML axis (pitch)", default_value=0.0,
                                         min_value=-30, max_value=30, callback=self._cb_param)
                    dpg.add_slider_float(tag=_T + "s_dv", label="DV axis (yaw)", default_value=0.0,
                                         min_value=-30, max_value=30, callback=self._cb_param)
                    dpg.add_separator()
                    dpg.add_input_int(tag=_T + "i_h", label="out height", default_value=285, min_value=16,
                                      min_clamped=True, step=0, callback=self._cb_param, width=120)
                    dpg.add_input_int(tag=_T + "i_w", label="out width", default_value=285, min_value=16,
                                      min_clamped=True, step=0, callback=self._cb_param, width=120)
                    dpg.add_input_int(tag=_T + "i_min", label="min pixels / area", default_value=15,
                                      min_value=0, min_clamped=True, step=0, callback=self._cb_param, width=120)
                    dpg.add_checkbox(tag=_T + "c_olf", label="include olfactory bulb", callback=self._cb_param)
                    dpg.add_checkbox(tag=_T + "c_cb", label="include cerebellum", callback=self._cb_param)
                    dpg.add_separator()
                    from ..atlas_build import _AP_CROP_DEFAULT, _ML_CROP_DEFAULT
                    dpg.add_text("Crop window (fraction of the volume; default = shipped frame):")
                    dpg.add_slider_float(tag=_T + "cr_ap0", label="AP crop lo", default_value=_AP_CROP_DEFAULT[0],
                                         min_value=0.0, max_value=1.0, callback=self._cb_param)
                    dpg.add_slider_float(tag=_T + "cr_ap1", label="AP crop hi", default_value=_AP_CROP_DEFAULT[1],
                                         min_value=0.0, max_value=1.0, callback=self._cb_param)
                    dpg.add_slider_float(tag=_T + "cr_ml0", label="ML crop lo", default_value=_ML_CROP_DEFAULT[0],
                                         min_value=0.0, max_value=1.0, callback=self._cb_param)
                    dpg.add_slider_float(tag=_T + "cr_ml1", label="ML crop hi", default_value=_ML_CROP_DEFAULT[1],
                                         min_value=0.0, max_value=1.0, callback=self._cb_param)
                    dpg.add_separator()
                    dpg.add_text("Preview uses a downsampled volume; Save rebuilds\n"
                                 "at full resolution. Red dots = default point ROIs\n"
                                 "(fixed frame — watch how the tilt moves cortex\n"
                                 "under them).", wrap=400)
                with dpg.child_window(autosize_x=True, autosize_y=True):
                    dpg.add_group(tag=_T + "preview_panel")

            dpg.add_separator()
            dpg.add_progress_bar(tag=_PROG, default_value=0.0, width=-1, overlay="")
            dpg.add_text("load the volumes to begin", tag=_STATUS)
            dpg.add_input_text(tag=_LOG, multiline=True, readonly=True, width=-1, height=90, default_value="")

    def run(self, *, title: str = "Atlas Builder") -> Path | None:
        import dearpygui.dearpygui as dpg

        dpg.create_context()
        dpg.create_viewport(title=title, width=1250, height=900)
        self.build()
        dpg.set_primary_window(_WIN, True)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        while dpg.is_dearpygui_running() and not self.should_close:
            self._drain()
            dpg.render_dearpygui_frame()
        dpg.destroy_context()
        return self.last_saved


def open_atlas_editor(annotation: str | None = None, template: str | None = None,
                      structure_tree: str | None = None, out_path: str = _DEF_OUT) -> Path | None:
    return AtlasEditor(annotation, template, structure_tree, out_path).run()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    a = (argv + [None, None, None, _DEF_OUT])[:4]
    saved = open_atlas_editor(*a)
    print(f"[atlas-editor] wrote {saved}" if saved else "[atlas-editor] closed; nothing saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
