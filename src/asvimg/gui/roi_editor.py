"""In-process ROI editor window (atlas image + editable ROI table).

Opened from the dashboard after annotation.  ROIs live in atlas (standard-brain)
pixel space; the user can move/resize/remove them, add new ones by clicking the
atlas, mirror one to the contralateral side, and save to ``rois.csv`` which ROI
extraction then uses instead of the atlas defaults.

The background image is selectable (atlas / warped channel means / PCA-ICA maps,
warped into atlas space with the annotation transform), region borders can be
toggled, and ROI names can be shown/hidden.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from asvimg import (
    ROIS_FILENAME,
    Roi,
    contralateral,
    resolve_rois,
    save_rois,
)

_WIN = "roi_editor_window"
_DRAW = "roi_editor_draw"
_TABLE = "roi_editor_table"
_TEXREG = "roi_editor_texreg"
_TEX = "roi_editor_tex"
_HANDLERS = "roi_editor_handlers"
_STATUS = "roi_editor_status"
_SCALE = 1.6  # atlas px -> drawlist px


def _png_idx(p: Path) -> int:
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else 0


class RoiEditor:
    def __init__(self, atlas, output_dir, on_saved=None) -> None:
        self.atlas = atlas
        self.output_dir = Path(output_dir)
        self.on_saved = on_saved
        self.rois: list[Roi] = list(resolve_rois(output_dir, atlas)[0])
        self._ids: list[int] = list(range(len(self.rois)))
        self._next_id = len(self.rois)
        self._show_names = True
        self._show_borders = True
        self._add_mode = False
        self._cursor = None  # (x, y) atlas coords for the crosshair
        self._tex = _TEX
        self._tform = None
        self._overlay_raw: dict = {}   # label -> ("gray", 2D) | ("rgb", png Path)
        self._overlay_order: list[str] = ["Atlas"]
        self._bg_cache: dict = {}
        self._red_theme = None

    # --- open / close -----------------------------------------------------

    def open(self) -> None:
        import dearpygui.dearpygui as dpg

        for tag in (_WIN, _TEXREG, _HANDLERS):
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)

        h, w = self.atlas.shape_hw
        self._build_overlay_index()
        self._register_texture(dpg, w, h)
        self._red_theme = self._make_red_text_theme(dpg)

        with dpg.window(label="Edit ROIs", tag=_WIN,
                        width=int((w * _SCALE + 380) * 1.4),
                        height=int(h * _SCALE) + 120, on_close=self._on_close):
            with dpg.group(horizontal=True):
                self._btn_add = dpg.add_button(label="Add ROI (click atlas)",
                                               callback=self._toggle_add)
                dpg.add_checkbox(label="Show names", default_value=True,
                                 callback=self._toggle_names)
                dpg.add_checkbox(label="Show borders", default_value=True,
                                 callback=self._toggle_borders)
                dpg.add_text("Overlay:")
                dpg.add_combo(self._overlay_order, default_value="Atlas", width=150,
                              callback=self._on_overlay)
                dpg.add_button(label="Reset to atlas defaults",
                               callback=self._reset_defaults)
                dpg.add_button(label="Save rois.csv", callback=self._save)
                dpg.add_text("", tag=_STATUS)
            with dpg.group(horizontal=True):
                dpg.add_drawlist(width=int(w * _SCALE), height=int(h * _SCALE),
                                 tag=_DRAW)
                # ROI table — fills the remaining window width; only Name stretches.
                with dpg.child_window(width=-1, autosize_y=True):
                    with dpg.table(tag=_TABLE, header_row=True,
                                   policy=dpg.mvTable_SizingFixedFit,
                                   resizable=True, scrollY=True,
                                   height=int(h * _SCALE)):
                        dpg.add_table_column(label="Name", width_stretch=True,
                                             init_width_or_weight=1.0)
                        dpg.add_table_column(label="X", width_fixed=True,
                                             init_width_or_weight=70)
                        dpg.add_table_column(label="Y", width_fixed=True,
                                             init_width_or_weight=70)
                        dpg.add_table_column(label="Size", width_fixed=True,
                                             init_width_or_weight=60)
                        dpg.add_table_column(label="UI", width_fixed=True,
                                             init_width_or_weight=175)

        with dpg.handler_registry(tag=_HANDLERS):
            dpg.add_mouse_click_handler(callback=self._on_click)
            dpg.add_mouse_move_handler(callback=self._on_move)

        self._rebuild_table()
        self._redraw()

    @staticmethod
    def _make_red_text_theme(dpg):
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (230, 80, 80),
                                    category=dpg.mvThemeCat_Core)
        return theme

    def _register_texture(self, dpg, w, h) -> None:
        rgba = self._rgba(self._load_bg("Atlas"))
        with dpg.texture_registry(tag=_TEXREG):
            dpg.add_dynamic_texture(width=w, height=h, default_value=rgba, tag=_TEX)

    def _on_close(self) -> None:
        import dearpygui.dearpygui as dpg

        for tag in (_WIN, _TEXREG, _HANDLERS):
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)

    # --- background images ------------------------------------------------

    def _build_overlay_index(self) -> None:
        """List selectable overlays: atlas, warped channel means, PCA/ICA maps."""
        self._overlay_order = ["Atlas"]
        self._overlay_raw = {}
        self._tform = None
        marks = self.output_dir / "marks.mat"
        if not marks.exists():
            return
        try:
            from asvimg import compute_transform, load_marks

            src, ref = load_marks(marks)
            self._tform = compute_transform(src, ref)
        except Exception:  # noqa: BLE001
            self._tform = None
            return

        from asvimg import (
            full_channel_mean,
            read_reg_meta,
            reg_meta_path,
        )

        if reg_meta_path(self.output_dir).exists():
            try:
                meta = read_reg_meta(self.output_dir)
                mean_keys = sorted(
                    k for k in meta if str(k).startswith("meanImageCh")
                )
                for k in mean_keys:
                    ch_idx = int(str(k).replace("meanImageCh", ""))
                    label = f"{k} (warped)"
                    self._overlay_raw[label] = (
                        "gray",
                        np.asarray(
                            full_channel_mean(self.output_dir, ch_idx), dtype=float
                        ),
                    )
                    self._overlay_order.append(label)
            except Exception:  # noqa: BLE001
                pass
        for sub, prefix in (("pca_images", "PC"), ("ica_images", "IC")):
            d = self.output_dir / sub
            if d.is_dir():
                # per-group subdirs (flat in older folders); the label needs the
                # group, or two groups' IC1 overwrite each other in the dict
                for png in sorted(d.glob(f"**/{prefix}*.png"), key=_png_idx):
                    label = (
                        png.stem if png.parent == d
                        else f"{png.parent.name}/{png.stem}"
                    )
                    self._overlay_raw[label] = ("rgb", png)
                    self._overlay_order.append(label)

    def _load_bg(self, label: str) -> np.ndarray:
        """(H, W, 3) float [0,1] atlas-space image for the given overlay."""
        if label in self._bg_cache:
            return self._bg_cache[label]
        if label == "Atlas" or label not in self._overlay_raw:
            rgb = np.asarray(self.atlas.image_rgb, dtype=np.float32)
            if rgb.max() > 1.5:
                rgb = rgb / 255.0
            out = np.clip(rgb[:, :, :3], 0.0, 1.0)
        else:
            from asvimg import warp_image

            kind, data = self._overlay_raw[label]
            if kind == "gray":
                w = warp_image(data, self._tform, self.atlas.shape_hw)
                pos = w[w > 0]
                lo, hi = np.percentile(pos, [2, 98]) if pos.size else (0.0, 1.0)
                g = np.clip((w - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
                out = np.dstack([g, g, g]).astype(np.float32)
            else:  # rgb PNG (source space) -> warp each channel
                from PIL import Image

                img = np.asarray(Image.open(data).convert("RGB"), dtype=float) / 255.0
                chans = [warp_image(img[:, :, c], self._tform, self.atlas.shape_hw)
                         for c in range(3)]
                out = np.clip(np.dstack(chans), 0.0, 1.0).astype(np.float32)
        self._bg_cache[label] = out
        return out

    @staticmethod
    def _rgba(rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        return np.dstack([rgb[:, :, :3].astype(np.float32),
                          np.ones((h, w), np.float32)]).ravel()

    def _on_overlay(self, sender, app_data) -> None:
        import dearpygui.dearpygui as dpg

        try:
            rgb = self._load_bg(str(app_data))
        except Exception as exc:  # noqa: BLE001
            if dpg.does_item_exist(_STATUS):
                dpg.set_value(_STATUS, f"overlay failed: {exc}")
            return
        dpg.set_value(_TEX, self._rgba(rgb))
        self._redraw()

    # --- atlas drawing ----------------------------------------------------

    def _redraw(self) -> None:
        import dearpygui.dearpygui as dpg

        if not dpg.does_item_exist(_DRAW):
            return
        dpg.delete_item(_DRAW, children_only=True)
        h, w = self.atlas.shape_hw
        dpg.draw_image(self._tex, (0, 0), (w * _SCALE, h * _SCALE), parent=_DRAW)
        if self._show_borders:
            s = self.atlas.scale * _SCALE
            for group in self.atlas.boundaries[1:]:  # skip whole-brain outline
                for contour in group:
                    c = np.asarray(contour, dtype=float)
                    pts = [(c[i, 1] * s, c[i, 0] * s) for i in range(len(c))]
                    if len(pts) >= 2:
                        dpg.draw_polyline(pts, color=(255, 255, 255, 110), parent=_DRAW)
        for roi in self.rois:
            cx, cy = roi.x * _SCALE, roi.y * _SCALE
            r = max(roi.size / 2.0 * _SCALE, 2.0)
            dpg.draw_circle((cx, cy), r, color=(255, 60, 60, 255),
                            fill=(255, 60, 60, 60), parent=_DRAW)
            if self._show_names:
                dpg.draw_text((cx + r + 2, cy - 7), roi.name, size=13,
                              color=(255, 255, 255, 230), parent=_DRAW)
        if self._add_mode and self._cursor is not None:
            x, y = self._cursor[0] * _SCALE, self._cursor[1] * _SCALE
            dpg.draw_line((x, 0), (x, h * _SCALE), color=(80, 220, 120, 200), parent=_DRAW)
            dpg.draw_line((0, y), (w * _SCALE, y), color=(80, 220, 120, 200), parent=_DRAW)

    # --- table ------------------------------------------------------------

    def _rebuild_table(self) -> None:
        import dearpygui.dearpygui as dpg

        if dpg.does_item_exist(_TABLE):
            dpg.delete_item(_TABLE, children_only=True, slot=1)  # rows only
        for rid, roi in zip(self._ids, self.rois):
            with dpg.table_row(parent=_TABLE):
                dpg.add_input_text(default_value=roi.name, width=-1,
                                   user_data=(rid, "name"), callback=self._edit)
                dpg.add_input_int(default_value=roi.x, width=60, step=0,
                                  user_data=(rid, "x"), callback=self._edit)
                dpg.add_input_int(default_value=roi.y, width=60, step=0,
                                  user_data=(rid, "y"), callback=self._edit)
                dpg.add_input_int(default_value=roi.size, width=55, step=0,
                                  user_data=(rid, "size"), callback=self._edit)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Add-contra", user_data=rid,
                                   callback=self._add_contra)
                    rm = dpg.add_button(label="Remove", user_data=rid,
                                        callback=self._remove)
                    dpg.bind_item_theme(rm, self._red_theme)  # red text

    # --- callbacks --------------------------------------------------------

    def _idx(self, rid) -> int:
        return self._ids.index(rid)

    def _edit(self, sender, app_data, user_data) -> None:
        rid, field = user_data
        roi = self.rois[self._idx(rid)]
        if field == "name":
            roi.name = str(app_data).strip()
        else:
            setattr(roi, field, int(app_data))
        self._redraw()

    def _remove(self, sender, app_data, user_data) -> None:
        i = self._idx(user_data)
        del self.rois[i]
        del self._ids[i]
        self._rebuild_table()
        self._redraw()

    def _add_contra(self, sender, app_data, user_data) -> None:
        roi = self.rois[self._idx(user_data)]
        self._append(contralateral(roi, self.atlas.shape_hw[1]))

    def _append(self, roi: Roi) -> None:
        self.rois.append(roi)
        self._ids.append(self._next_id)
        self._next_id += 1
        self._rebuild_table()
        self._redraw()

    def _toggle_add(self, sender, app_data) -> None:
        import dearpygui.dearpygui as dpg

        self._add_mode = not self._add_mode
        dpg.configure_item(
            self._btn_add,
            label="Adding... (click atlas)" if self._add_mode else "Add ROI (click atlas)",
        )
        self._redraw()

    def _toggle_names(self, sender, app_data) -> None:
        self._show_names = bool(app_data)
        self._redraw()

    def _toggle_borders(self, sender, app_data) -> None:
        self._show_borders = bool(app_data)
        self._redraw()

    def _reset_defaults(self, sender=None, app_data=None) -> None:
        from asvimg import default_rois

        self.rois = list(default_rois(self.atlas))
        self._ids = list(range(len(self.rois)))
        self._next_id = len(self.rois)
        self._rebuild_table()
        self._redraw()

    def _save(self, sender=None, app_data=None) -> None:
        import dearpygui.dearpygui as dpg

        path = self.output_dir / ROIS_FILENAME
        save_rois(path, self.rois)
        if dpg.does_item_exist(_STATUS):
            dpg.set_value(_STATUS, f"saved {len(self.rois)} ROIs -> {ROIS_FILENAME}")
        if self.on_saved is not None:
            self.on_saved(path, len(self.rois))

    # --- atlas mouse interaction -----------------------------------------

    def _mouse_atlas_pos(self):
        import dearpygui.dearpygui as dpg

        if not dpg.is_item_hovered(_DRAW):
            return None
        mx, my = dpg.get_drawing_mouse_pos()
        h, w = self.atlas.shape_hw
        x = int(round(mx / _SCALE))
        y = int(round(my / _SCALE))
        if 0 <= x < w and 0 <= y < h:
            return x, y
        return None

    def _on_move(self, sender, app_data) -> None:
        if not self._add_mode:
            return
        pos = self._mouse_atlas_pos()
        if pos != self._cursor:
            self._cursor = pos
            self._redraw()

    def _on_click(self, sender, app_data) -> None:
        if not self._add_mode:
            return
        pos = self._mouse_atlas_pos()
        if pos is None:
            return
        from asvimg.rois import DEFAULT_ROI_SIZE

        n = sum(1 for r in self.rois if r.name.startswith("ROI"))
        self._append(Roi(f"ROI{n + 1}", pos[0], pos[1], DEFAULT_ROI_SIZE))


def open_editor(atlas, output_dir, on_saved=None) -> RoiEditor:
    ed = RoiEditor(atlas, output_dir, on_saved=on_saved)
    ed.open()
    return ed
