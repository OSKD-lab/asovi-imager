"""In-process seed-correlation-map panel (functional-connectivity aid).

Opened from the dashboard after annotation.  Pick a dF/F group + seed ROI,
**Compute Map** runs a vectorised seed correlation in Allen-atlas space (with a
temporal ``skip`` stride for speed), saves it to ``<asi>/corrMap/*.npy`` +
``*.png`` and shows it.  Previously computed maps are selectable from a list.
**Add for Annot.** inverse-warps the shown map back to source (registered)
coordinates and drops it into ``map_for_annot`` so it can be used as a
control-point-selection aid.

Like the ROI editor, this creates a window in the dashboard's own dpg context
(no second context); its callbacks run on the main thread.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_WIN = "corrmap_window"
_DRAW = "corrmap_draw"
_TEXREG = "corrmap_texreg"
_TEX = "corrmap_tex"
_STATUS = "corrmap_status"
_SEED = "corrmap_seed"
_METHOD = "corrmap_method"
_GROUP = "corrmap_group"
_SKIP = "corrmap_skip"
_SMOOTH = "corrmap_smooth"
_LIST = "corrmap_list"
_SCALE = 1.6  # atlas px -> drawlist px


class CorrMapPanel:
    def __init__(self, config, atlas, output_dir) -> None:
        self.config = config
        self.atlas = atlas
        self.output_dir = Path(output_dir)
        self.corr_dir = self.output_dir / "corrMap"
        self._tform = None
        self._source_shape = None
        self._cur_map = None   # atlas-space (Ha, Wa) map currently shown
        self._cur_name = None
        self._rois: dict = {}  # configured ROI name -> Roi (rois.csv / atlas defaults)

    # --- open -------------------------------------------------------------

    def _groups(self) -> list[str]:
        try:
            return list(self.config.channel_groups().keys())
        except Exception:  # noqa: BLE001
            return list(dict.fromkeys(self.config.channels_name or []))

    def open(self) -> None:
        import dearpygui.dearpygui as dpg

        from asvimg import (
            compute_transform,
            load_marks,
            resolve_rois,
        )

        marks = self.output_dir / "marks.mat"
        if not marks.exists():
            raise FileNotFoundError("no marks.mat — run annotation first")
        src, ref = load_marks(marks)
        self._tform = compute_transform(
            src, ref, allow_reflection=self.config.annotation_allow_reflection
        )
        # Seed from the *configured* ROI set (edited rois.csv, else atlas defaults).
        self._rois = {r.name: r for r in resolve_rois(self.output_dir, self.atlas)[0]}

        for tag in (_WIN, _TEXREG):
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)

        h, w = self.atlas.shape_hw
        blank = np.zeros((h, w, 4), np.float32)
        blank[:, :, 3] = 1.0
        with dpg.texture_registry(tag=_TEXREG):
            dpg.add_dynamic_texture(
                width=w, height=h, default_value=blank.ravel(), tag=_TEX
            )

        groups = self._groups()
        seeds = list(self._rois.keys())
        with dpg.window(
            label="Correlation Map",
            tag=_WIN,
            width=int(w * _SCALE) + 380,
            height=int(h * _SCALE) + 150,
            on_close=self._on_close,
        ):
            with dpg.group(horizontal=True):
                dpg.add_text("Group:")
                dpg.add_combo(
                    groups, default_value=(groups[0] if groups else ""),
                    tag=_GROUP, width=110,
                )
                dpg.add_text("Seed ROI:")
                dpg.add_combo(
                    seeds, default_value=(seeds[0] if seeds else ""),
                    tag=_SEED, width=150,
                )
                dpg.add_text("Method:")
                dpg.add_combo(["raw", "gsr"], default_value="raw", tag=_METHOD, width=70)
                dpg.add_text("Skip:")
                dpg.add_input_int(
                    default_value=10, min_value=1, min_clamped=True,
                    tag=_SKIP, width=70, step=0,
                )
                dpg.add_text("Smooth:")
                dpg.add_input_float(
                    default_value=2.0, min_value=0.0, min_clamped=True,
                    tag=_SMOOTH, width=80, step=0, format="%.1f",
                )
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text(
                        "Spatial Gaussian sigma (atlas px) applied to the map to\n"
                        "reduce per-pixel speckle. 0 = off."
                    )
                dpg.add_button(label="Compute Map", callback=self._compute)
            with dpg.group(horizontal=True):
                dpg.add_text("Computed:")
                dpg.add_combo(
                    self._list_maps(), tag=_LIST, width=240, callback=self._on_select
                )
                dpg.add_button(label="Add for Annot.", callback=self._add_for_annot)
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text(
                        "Inverse-warp the shown map to source coordinates and\n"
                        "save it into map_for_annot for control-point selection."
                    )
                dpg.add_text("", tag=_STATUS)
            dpg.add_drawlist(width=int(w * _SCALE), height=int(h * _SCALE), tag=_DRAW)

        self._redraw()

    def _on_close(self) -> None:
        import dearpygui.dearpygui as dpg

        for tag in (_WIN, _TEXREG):
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)

    # --- data -------------------------------------------------------------

    def _list_maps(self) -> list[str]:
        if not self.corr_dir.is_dir():
            return []
        return sorted(p.stem for p in self.corr_dir.glob("*.npy"))

    def _roi_mask(self, roi) -> np.ndarray:
        """Circular atlas-space seed mask at a configured ROI (x=col, y=row)."""
        ha, wa = self.atlas.shape_hw
        rr, cc = np.mgrid[:ha, :wa]
        radius = max(float(roi.size) / 2.0, 0.5)
        return ((rr - roi.y) ** 2 + (cc - roi.x) ** 2) <= radius**2

    def _set_status(self, msg: str) -> None:
        import dearpygui.dearpygui as dpg

        if dpg.does_item_exist(_STATUS):
            dpg.set_value(_STATUS, msg)

    def _compute(self, *_a) -> None:
        import dearpygui.dearpygui as dpg

        from asvimg.io import load_dff
        from asvimg.seedmap import compute_seed_corr_map, save_corr_map

        group = dpg.get_value(_GROUP)
        seed = dpg.get_value(_SEED)
        method = dpg.get_value(_METHOD)
        skip = max(1, int(dpg.get_value(_SKIP)))
        smooth = max(0.0, float(dpg.get_value(_SMOOTH)))
        if not group or not seed:
            self._set_status("pick a group and a seed ROI")
            return
        roi = self._rois.get(seed)
        if roi is None:
            self._set_status(f"unknown seed ROI: {seed}")
            return
        self._set_status(f"computing {group}/{seed} ...")
        try:
            # Through the seam (_load_name_dff_stack), NOT load_dff: this panel
            # used to read dff_{name}.npy directly, so it would have kept serving
            # un-denoised data under ica_denoise, and it died outright on
            # donner-less groups (which have no dff file of their own... they do
            # now, but the seam is still the only thing that knows about ICA).
            # Subsample time on the memmap so only T/skip frames come off disk.
            from asvimg.annotation import _load_name_dff_stack

            groups = self.config.channel_groups()
            stack, note = _load_name_dff_stack(
                group, groups[group], self.config, self.output_dir
            )
            if stack is None:
                self._set_status(f"{note}")
                return
            sub = np.ascontiguousarray(
                np.asarray(stack[..., ::skip], dtype=np.float32)
            )  # (H, W, T/skip)
            del stack  # release the memmap handle (Windows) before compute
            cmap = compute_seed_corr_map(
                sub, self._tform, self.atlas,
                seed_mask=self._roi_mask(roi), method=method, skip_frames=1,
                smooth_sigma=smooth,
            )
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"compute failed: {exc}")
            return
        name = f"{group}_{seed}" + ("_gsr" if method == "gsr" else "")
        save_corr_map(self.corr_dir, name, cmap)
        self._cur_map, self._cur_name = cmap, name
        dpg.configure_item(_LIST, items=self._list_maps())
        dpg.set_value(_LIST, name)
        self._show(cmap)
        self._set_status(f"saved {name} -> corrMap")

    def _on_select(self, sender, app_data) -> None:
        name = str(app_data)
        p = self.corr_dir / f"{name}.npy"
        if not p.exists():
            self._set_status(f"not found: {name}")
            return
        cmap = np.load(p)
        if np.shape(cmap) != tuple(self.atlas.shape_hw):
            self._set_status(f"{name}: shape {np.shape(cmap)} != atlas — skipped")
            return
        self._cur_map, self._cur_name = cmap, name
        self._show(cmap)
        self._set_status(f"showing {name}")

    def _add_for_annot(self, *_a) -> None:
        from asvimg import load_atlas_source_mean
        from asvimg.seedmap import unwarp_to_source

        if self._cur_map is None:
            self._set_status("compute or select a map first")
            return
        try:
            if self._source_shape is None:
                src = load_atlas_source_mean(self.config, self.output_dir)
                self._source_shape = np.asarray(src).shape[:2]
            un = unwarp_to_source(self._cur_map, self._tform, self._source_shape)
            map_dir = self.output_dir / "map_for_annot"
            map_dir.mkdir(parents=True, exist_ok=True)
            np.save(map_dir / f"corr_{self._cur_name}.npy", un)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"add failed: {exc}")
            return
        self._set_status(f"added corr_{self._cur_name} -> map_for_annot")

    # --- display ----------------------------------------------------------

    @staticmethod
    def _map_rgba(cmap: np.ndarray) -> np.ndarray:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        from asvimg.seedmap import corr_vmax

        vmax = corr_vmax(cmap)  # auto symmetric range (percentile of |corr|)
        sm = ScalarMappable(norm=Normalize(-vmax, vmax), cmap="RdBu_r")
        rgb = sm.to_rgba(np.nan_to_num(np.asarray(cmap, float), nan=0.0))[:, :, :3]
        h, w = rgb.shape[:2]
        return np.dstack(
            [rgb.astype(np.float32), np.ones((h, w), np.float32)]
        ).ravel()

    def _show(self, cmap: np.ndarray) -> None:
        import dearpygui.dearpygui as dpg

        if np.shape(cmap) != tuple(self.atlas.shape_hw):  # guard the fixed texture
            self._set_status(
                f"map shape {np.shape(cmap)} != atlas {tuple(self.atlas.shape_hw)}"
            )
            return
        if dpg.does_item_exist(_TEX):
            dpg.set_value(_TEX, self._map_rgba(cmap))
        self._redraw()

    def _redraw(self) -> None:
        import dearpygui.dearpygui as dpg

        if not dpg.does_item_exist(_DRAW):
            return
        dpg.delete_item(_DRAW, children_only=True)
        h, w = self.atlas.shape_hw
        dpg.draw_image(_TEX, (0, 0), (w * _SCALE, h * _SCALE), parent=_DRAW)
        s = self.atlas.scale * _SCALE
        for group in self.atlas.boundaries[1:]:  # skip whole-brain outline
            for contour in group:
                c = np.asarray(contour, dtype=float)
                pts = [(c[i, 1] * s, c[i, 0] * s) for i in range(len(c))]
                if len(pts) >= 2:
                    dpg.draw_polyline(pts, color=(0, 0, 0, 90), parent=_DRAW)


def open_panel(config, atlas, output_dir) -> CorrMapPanel:
    panel = CorrMapPanel(config, atlas, output_dir)
    panel.open()
    return panel
