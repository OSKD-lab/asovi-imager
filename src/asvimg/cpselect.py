"""Interactive control point selection GUI using Dear PyGui.

Provides a ``cpselect``-style interface for selecting corresponding point
pairs between a source image and a reference (atlas) image, with live
warp preview.
"""

from __future__ import annotations

from enum import Enum, auto

import numpy as np

_DEFAULT_LANDMARKS: dict[str, tuple[float, float]] = {
    "Front Edge": (55.0, 142.5),
    "RSC Edge": (212.5, 142.5),
}


class _State(Enum):
    IDLE = auto()
    PLACING_SOURCE = auto()
    PLACING_REFERENCE = auto()


def _normalize_rgba(image: np.ndarray) -> np.ndarray:
    """Convert any image array to (H, W, 4) float32 in [0, 1] for DPG."""
    img = np.asarray(image, dtype=np.float64)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    # Normalize RGB to 0-1 BEFORE adding alpha
    if img.ndim == 3 and img.shape[2] in (3, 4):
        rgb = img[:, :, :3]
        vmin, vmax = rgb.min(), rgb.max()
        if vmax > vmin:
            rgb = (rgb - vmin) / (vmax - vmin)
        else:
            rgb = np.zeros_like(rgb)
        alpha = np.ones((*img.shape[:2], 1), dtype=np.float64)
        img = np.concatenate([rgb, alpha], axis=-1)

    return img.astype(np.float32)


def cpselect_gui(
    source_img: np.ndarray | list[np.ndarray],
    reference_img: np.ndarray,
    initial_source_pts: np.ndarray | None = None,
    initial_ref_pts: np.ndarray | None = None,
    n_default_pairs: int = 2,
    boundaries: list[list[np.ndarray]] | None = None,
    boundary_scale: float = 0.25,
    ref_landmarks: dict[str, tuple[float, float]] | None = _DEFAULT_LANDMARKS,
    source_labels: list[str] | None = None,
    title: str = "Control Point Selection",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Open a cpselect-style GUI for control point selection with warp preview.

    Parameters
    ----------
    source_img : 2D/3D array, or list of arrays (same H, W).
        When a list is given, prev/next buttons switch between images.
        All images must have the same spatial dimensions.
    reference_img : 2D grayscale or (H, W, 3) RGB array.
    initial_source_pts : (N, 2) pre-existing source points as (row, col).
    initial_ref_pts : (N, 2) pre-existing reference points as (row, col).
    n_default_pairs : Number of point pairs to place on startup (default 2).
    boundaries : Atlas region boundaries from ``AtlasData.boundaries``.
    boundary_scale : Scale factor for boundary coordinates (default 1/4).
    ref_landmarks : Preset reference landmark buttons.
    source_labels : Display names for each source image (optional).
    title : Window title.

    Returns
    -------
    (source_pts, ref_pts) each (N, 2) as (row, col), or None if cancelled.
    """
    import dearpygui.dearpygui as dpg

    from .annotation import compute_transform, warp_image

    # Normalize source images to a list
    if isinstance(source_img, np.ndarray):
        source_imgs = [source_img]
    else:
        source_imgs = list(source_img)

    if source_labels is None:
        if len(source_imgs) == 1:
            source_labels = ["Source"]
        else:
            source_labels = [
                f"Source {i + 1}/{len(source_imgs)}" for i in range(len(source_imgs))
            ]

    # Validate all source images have the same spatial dimensions
    h0, w0 = source_imgs[0].shape[:2]
    for i, img in enumerate(source_imgs[1:], 1):
        if img.shape[0] != h0 or img.shape[1] != w0:
            raise ValueError(
                f"Source image {i} has shape {img.shape[:2]}, "
                f"expected ({h0}, {w0}). All source images must have the same H, W."
            )

    # Pre-compute RGBA textures and grayscale for all source images
    src_rgbas = [_normalize_rgba(img) for img in source_imgs]
    src_grays = []
    for img in source_imgs:
        if img.ndim == 2:
            src_grays.append(img.astype(np.float64))
        else:
            src_grays.append(np.mean(img[:, :, :3].astype(np.float64), axis=2))

    ref_rgba = _normalize_rgba(reference_img)

    src_h, src_w = src_rgbas[0].shape[:2]
    ref_h, ref_w = ref_rgba.shape[:2]

    # Current source image index (mutable list for closure access)
    src_idx = [0]

    # State
    source_pts: list[tuple[float, float]] = []  # (row, col)
    ref_pts: list[tuple[float, float]] = []
    state = _State.IDLE
    current_pair_idx = 0
    selected_idx: int | None = None
    result: list[tuple[np.ndarray, np.ndarray] | None] = [None]

    if initial_source_pts is not None and initial_ref_pts is not None:
        for r, c in initial_source_pts:
            source_pts.append((float(r), float(c)))
        for r, c in initial_ref_pts:
            ref_pts.append((float(r), float(c)))

    # ---- Dear PyGui setup ----
    dpg.create_context()
    dpg.create_viewport(title=title, width=1600, height=700)

    # Create initial preview (blank)
    preview_data = np.zeros((ref_h, ref_w, 4), dtype=np.float32)
    preview_data[:, :, 3] = 1.0

    # Register textures
    with dpg.texture_registry():
        src_tex = dpg.add_raw_texture(
            src_w, src_h, src_rgbas[0].flatten(), format=dpg.mvFormat_Float_rgba
        )
        ref_tex = dpg.add_raw_texture(
            ref_w, ref_h, ref_rgba.flatten(), format=dpg.mvFormat_Float_rgba
        )
        preview_tex = dpg.add_raw_texture(
            ref_w, ref_h, preview_data.flatten(), format=dpg.mvFormat_Float_rgba
        )

    # Theme for boundary lines (magenta)
    with dpg.theme(tag="_border_theme"):
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(
                dpg.mvPlotCol_Line, (255, 0, 255, 180), category=dpg.mvThemeCat_Plots
            )

    def _switch_source(delta: int):
        """Switch to prev/next source image."""
        new_idx = (src_idx[0] + delta) % len(source_imgs)
        src_idx[0] = new_idx
        dpg.set_value(src_tex, src_rgbas[new_idx].flatten())
        if dpg.does_item_exist("src_label"):
            dpg.set_value("src_label", source_labels[new_idx])
        _update_preview()

    def _on_src_prev(sender, app_data):
        _switch_source(-1)

    def _on_src_next(sender, app_data):
        _switch_source(1)

    def _update_preview():
        """Compute warp and update preview texture and boundaries."""
        n = min(len(source_pts), len(ref_pts))
        if n < 2:
            blank = np.zeros((ref_h, ref_w, 4), dtype=np.float32)
            blank[:, :, 3] = 1.0
            dpg.set_value(preview_tex, blank.flatten())
            _draw_preview_boundaries()
            return

        try:
            src_arr = np.array(source_pts[:n])
            ref_arr = np.array(ref_pts[:n])
            tform = compute_transform(src_arr, ref_arr)
            warped = warp_image(src_grays[src_idx[0]], tform, (ref_h, ref_w))

            warped_rgba = _normalize_rgba(warped)
            dpg.set_value(preview_tex, warped_rgba.flatten())
        except Exception:
            pass
        _draw_preview_boundaries()

    # Track boundary line series tags for cleanup
    _boundary_series_tags: list[str] = []

    _ref_boundary_tags: list[str] = []

    def _draw_preview_boundaries():
        """Draw or clear boundaries on preview plot using line_series."""
        # Remove existing boundary series
        for tag in _boundary_series_tags:
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        _boundary_series_tags.clear()

        if (
            boundaries is None
            or not dpg.does_item_exist("border_checkbox")
            or not dpg.get_value("border_checkbox")
        ):
            return

        for i in range(1, len(boundaries)):
            for j, contour in enumerate(boundaries[i]):
                pts = contour * boundary_scale  # (M, 2) as (row, col)
                xs = pts[:, 1].tolist()  # col
                ys = (ref_h - pts[:, 0]).tolist()  # row → flipped for plot
                tag = f"_border_{i}_{j}"
                dpg.add_line_series(
                    xs,
                    ys,
                    parent="preview_y",
                    tag=tag,
                )
                dpg.bind_item_theme(tag, "_border_theme")
                _boundary_series_tags.append(tag)

    def _draw_ref_boundaries():
        """Draw or clear boundaries on reference plot."""
        for tag in _ref_boundary_tags:
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        _ref_boundary_tags.clear()

        if (
            boundaries is None
            or not dpg.does_item_exist("border_checkbox")
            or not dpg.get_value("border_checkbox")
        ):
            return

        for i in range(1, len(boundaries)):
            for j, contour in enumerate(boundaries[i]):
                pts = contour * boundary_scale
                xs = pts[:, 1].tolist()
                ys = (ref_h - pts[:, 0]).tolist()
                tag = f"_refborder_{i}_{j}"
                dpg.add_line_series(
                    xs,
                    ys,
                    parent="ref_y",
                    tag=tag,
                )
                dpg.bind_item_theme(tag, "_border_theme")
                _ref_boundary_tags.append(tag)

    def _on_border_toggle(sender, app_data):
        _draw_preview_boundaries()
        _draw_ref_boundaries()

    def _status_text() -> str:
        if state == _State.PLACING_SOURCE:
            return f"Click SOURCE image to place point {current_pair_idx + 1}"
        elif state == _State.PLACING_REFERENCE:
            return f"Click REFERENCE image to place point {current_pair_idx + 1}"
        else:
            n = min(len(source_pts), len(ref_pts))
            return f"{n} point pair(s) placed. [+ Add Pair] or [Finish] to close."

    def _redraw_markers():
        # Clear and redraw source markers
        dpg.delete_item("src_draw", children_only=True)
        for i, (r, c) in enumerate(source_pts):
            px, py = _image_to_plot_src(r, c)
            color = (255, 255, 0) if i == selected_idx else (0, 255, 0)
            radius = 6
            dpg.draw_circle(
                (px, py), radius, color=color, thickness=2, parent="src_draw"
            )
            dpg.draw_text(
                (px + 8, py - 8),
                str(i + 1),
                color=color,
                size=16,
                parent="src_draw",
            )

        # Clear and redraw reference markers
        dpg.delete_item("ref_draw", children_only=True)
        for i, (r, c) in enumerate(ref_pts):
            px, py = _image_to_plot_ref(r, c)
            color = (255, 255, 0) if i == selected_idx else (0, 255, 0)
            radius = 6
            dpg.draw_circle(
                (px, py), radius, color=color, thickness=2, parent="ref_draw"
            )
            dpg.draw_text(
                (px + 8, py - 8),
                str(i + 1),
                color=color,
                size=16,
                parent="ref_draw",
            )

        dpg.set_value("status_text", _status_text())

    # DPG plot Y is inverted vs image row: image_row = img_h - plot_y
    def _plot_to_image_src(plot_x: float, plot_y: float) -> tuple[float, float]:
        return (src_h - plot_y, plot_x)  # (row, col)

    def _image_to_plot_src(row: float, col: float) -> tuple[float, float]:
        return (col, src_h - row)  # (plot_x, plot_y)

    def _plot_to_image_ref(plot_x: float, plot_y: float) -> tuple[float, float]:
        return (ref_h - plot_y, plot_x)  # (row, col)

    def _image_to_plot_ref(row: float, col: float) -> tuple[float, float]:
        return (col, ref_h - row)  # (plot_x, plot_y)

    def _on_src_click(sender, app_data):
        nonlocal state, current_pair_idx
        if state != _State.PLACING_SOURCE:
            return
        mouse_pos = dpg.get_plot_mouse_pos()
        row, col = _plot_to_image_src(mouse_pos[0], mouse_pos[1])
        if 0 <= col < src_w and 0 <= row < src_h:
            if current_pair_idx < len(source_pts):
                source_pts[current_pair_idx] = (row, col)
            else:
                source_pts.append((row, col))
            state = _State.PLACING_REFERENCE
            _redraw_markers()

    def _on_ref_click(sender, app_data):
        nonlocal state, current_pair_idx
        if state != _State.PLACING_REFERENCE:
            return
        mouse_pos = dpg.get_plot_mouse_pos()
        row, col = _plot_to_image_ref(mouse_pos[0], mouse_pos[1])
        if 0 <= col < ref_w and 0 <= row < ref_h:
            if current_pair_idx < len(ref_pts):
                ref_pts[current_pair_idx] = (row, col)
            else:
                ref_pts.append((row, col))
            current_pair_idx += 1
            if current_pair_idx < n_default_pairs:
                state = _State.PLACING_SOURCE
            else:
                state = _State.IDLE
            _redraw_markers()
            _update_preview()

    def _on_add_pair(sender, app_data):
        nonlocal state, current_pair_idx
        current_pair_idx = min(len(source_pts), len(ref_pts))
        state = _State.PLACING_SOURCE
        _redraw_markers()

    def _on_delete(sender, app_data):
        nonlocal selected_idx, state
        if selected_idx is not None and selected_idx < len(source_pts):
            source_pts.pop(selected_idx)
            if selected_idx < len(ref_pts):
                ref_pts.pop(selected_idx)
            selected_idx = None
            state = _State.IDLE
            _redraw_markers()
            _update_preview()

    def _set_ref_point(row: float, col: float):
        """Set reference point for current pair (used by landmarks and Apply)."""
        nonlocal state, current_pair_idx
        if state != _State.PLACING_REFERENCE:
            return
        if current_pair_idx < len(ref_pts):
            ref_pts[current_pair_idx] = (row, col)
        else:
            ref_pts.append((row, col))
        current_pair_idx += 1
        if current_pair_idx < n_default_pairs:
            state = _State.PLACING_SOURCE
        else:
            state = _State.IDLE
        _redraw_markers()
        _update_preview()

    def _on_landmark(sender, app_data, user_data):
        """Callback for landmark preset buttons."""
        row, col = user_data
        _set_ref_point(row, col)

    def _on_apply_coords(sender, app_data):
        """Apply manually entered row, col as reference point."""
        try:
            row = float(dpg.get_value("input_row"))
            col = float(dpg.get_value("input_col"))
            _set_ref_point(row, col)
        except (ValueError, TypeError):
            pass

    _should_close = [False]

    def _on_finish(sender, app_data):
        n = min(len(source_pts), len(ref_pts))
        if n >= 2:
            result[0] = (
                np.array(source_pts[:n]),
                np.array(ref_pts[:n]),
            )
        _should_close[0] = True

    def _on_cancel(sender, app_data):
        result[0] = None
        _should_close[0] = True

    # Build UI
    with dpg.window(tag="primary", label=title):
        # Toolbar row 1
        with dpg.group(horizontal=True):
            dpg.add_button(label="+ Add Pair", callback=_on_add_pair)
            dpg.add_button(label="Delete Selected", callback=_on_delete)
            dpg.add_spacer(width=30)
            dpg.add_button(label="Finish", callback=_on_finish, width=80)
            dpg.add_button(label="Cancel", callback=_on_cancel, width=80)
            if boundaries is not None:
                dpg.add_spacer(width=20)
                dpg.add_checkbox(
                    label="Show Borders",
                    tag="border_checkbox",
                    default_value=False,
                    callback=_on_border_toggle,
                )
            if len(source_imgs) > 1:
                dpg.add_spacer(width=20)
                dpg.add_button(label="<", callback=_on_src_prev, width=30)
                dpg.add_text(source_labels[0], tag="src_label")
                dpg.add_button(label=">", callback=_on_src_next, width=30)

        # Toolbar row 2: landmark presets + manual coordinate input
        with dpg.group(horizontal=True):
            dpg.add_text("Ref presets:")
            if ref_landmarks:
                for name, (r, c) in ref_landmarks.items():
                    dpg.add_button(
                        label=name,
                        callback=_on_landmark,
                        user_data=(r, c),
                    )
            dpg.add_spacer(width=20)
            dpg.add_text("row:")
            dpg.add_input_float(tag="input_row", width=70, default_value=0.0, step=0)
            dpg.add_text("col:")
            dpg.add_input_float(tag="input_col", width=70, default_value=0.0, step=0)
            dpg.add_button(label="Apply", callback=_on_apply_coords)

        dpg.add_text(_status_text(), tag="status_text")
        dpg.add_separator()

        # Three plots side by side
        plot_w = 500
        with dpg.group(horizontal=True):
            # Source image plot
            with dpg.plot(
                label="Source",
                tag="src_plot",
                width=plot_w,
                height=-1,
                equal_aspects=True,
            ):
                dpg.add_plot_axis(dpg.mvXAxis, tag="src_x", no_tick_labels=True)
                with dpg.plot_axis(dpg.mvYAxis, tag="src_y", no_tick_labels=True):
                    dpg.add_image_series(src_tex, [0, 0], [src_w, src_h], tag="src_img")
                dpg.set_axis_limits("src_x", 0, src_w)
                dpg.set_axis_limits("src_y", src_h, 0)
                dpg.set_axis_limits_auto("src_x")
                dpg.set_axis_limits_auto("src_y")

            # Reference image plot
            with dpg.plot(
                label="Reference (Atlas)",
                tag="ref_plot",
                width=plot_w,
                height=-1,
                equal_aspects=True,
            ):
                dpg.add_plot_axis(dpg.mvXAxis, tag="ref_x", no_tick_labels=True)
                with dpg.plot_axis(dpg.mvYAxis, tag="ref_y", no_tick_labels=True):
                    dpg.add_image_series(ref_tex, [0, 0], [ref_w, ref_h], tag="ref_img")
                dpg.set_axis_limits("ref_x", 0, ref_w)
                dpg.set_axis_limits("ref_y", ref_h, 0)
                dpg.set_axis_limits_auto("ref_x")
                dpg.set_axis_limits_auto("ref_y")

            # Warp preview plot (same axis setup as source/reference)
            with dpg.plot(
                label="Warp Preview",
                tag="preview_plot",
                width=plot_w,
                height=-1,
                equal_aspects=True,
            ):
                dpg.add_plot_axis(dpg.mvXAxis, tag="preview_x", no_tick_labels=True)
                with dpg.plot_axis(dpg.mvYAxis, tag="preview_y", no_tick_labels=True):
                    dpg.add_image_series(
                        preview_tex, [0, 0], [ref_w, ref_h], tag="preview_img"
                    )
                dpg.set_axis_limits("preview_x", 0, ref_w)
                dpg.set_axis_limits("preview_y", ref_h, 0)
                dpg.set_axis_limits_auto("preview_x")
                dpg.set_axis_limits_auto("preview_y")

    # Draw layers for markers and guide line
    with dpg.draw_node(parent="src_plot", tag="src_draw"):
        pass
    with dpg.draw_node(parent="src_plot", tag="src_guide"):
        pass
    with dpg.draw_node(parent="ref_plot", tag="ref_draw"):
        pass

    # Click handlers
    def _on_mouse_click(sender, app_data):
        nonlocal state, selected_idx
        button = app_data  # 0=left, 1=right, 2=middle

        if button != 0:
            return

        if state == _State.PLACING_SOURCE and dpg.is_item_hovered("src_plot"):
            _on_src_click(sender, app_data)
        elif state == _State.PLACING_REFERENCE and dpg.is_item_hovered("ref_plot"):
            _on_ref_click(sender, app_data)
        elif state == _State.IDLE:
            # Check if clicking near an existing point to select it
            if dpg.is_item_hovered("src_plot"):
                mouse_pos = dpg.get_plot_mouse_pos()
                row, col = _plot_to_image_src(mouse_pos[0], mouse_pos[1])
                selected_idx = _find_nearest(source_pts, row, col, threshold=15)
                _redraw_markers()
            elif dpg.is_item_hovered("ref_plot"):
                mouse_pos = dpg.get_plot_mouse_pos()
                row, col = _plot_to_image_ref(mouse_pos[0], mouse_pos[1])
                selected_idx = _find_nearest(ref_pts, row, col, threshold=15)
                _redraw_markers()

    def _on_key_press(sender, app_data):
        """Arrow keys nudge the selected source point by 1 pixel."""
        if selected_idx is None or selected_idx >= len(source_pts):
            return
        row, col = source_pts[selected_idx]
        key = app_data
        if key == dpg.mvKey_Up:
            row -= 1
        elif key == dpg.mvKey_Down:
            row += 1
        elif key == dpg.mvKey_Left:
            col -= 1
        elif key == dpg.mvKey_Right:
            col += 1
        else:
            return
        source_pts[selected_idx] = (row, col)
        _redraw_markers()
        _update_preview()

    def _on_mouse_move(sender, app_data):
        """Draw guide line from last placed source point to cursor while placing next."""
        dpg.delete_item("src_guide", children_only=True)
        if state != _State.PLACING_SOURCE or len(source_pts) == 0:
            return
        # Draw line from the previous source point to the cursor
        prev_r, prev_c = source_pts[-1]
        mouse_pos = dpg.get_plot_mouse_pos()
        px, py = _image_to_plot_src(prev_r, prev_c)
        cur_row, cur_col = _plot_to_image_src(mouse_pos[0], mouse_pos[1])
        cx, cy = _image_to_plot_src(cur_row, cur_col)
        dpg.draw_line(
            (px, py),
            (cx, cy),
            color=(255, 128, 200, 150),
            thickness=1,
            parent="src_guide",
        )

    with dpg.handler_registry():
        dpg.add_mouse_click_handler(callback=_on_mouse_click)
        dpg.add_key_press_handler(callback=_on_key_press)
        dpg.add_mouse_move_handler(callback=_on_mouse_move)

    # Start placing if needed
    if len(source_pts) < n_default_pairs:
        state = _State.PLACING_SOURCE
        current_pair_idx = len(source_pts)
    _redraw_markers()

    # Draw reference boundaries on startup
    _draw_ref_boundaries()

    # If initial points provided, show preview immediately
    if len(source_pts) >= 2 and len(ref_pts) >= 2:
        _update_preview()

    dpg.set_primary_window("primary", True)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    # Manual render loop — flag-based shutdown for Jupyter compatibility
    while dpg.is_dearpygui_running() and not _should_close[0]:
        dpg.render_dearpygui_frame()

    dpg.destroy_context()

    return result[0]


def _find_nearest(
    pts: list[tuple[float, float]],
    row: float,
    col: float,
    threshold: float,
) -> int | None:
    """Find the nearest point within threshold distance."""
    best_idx = None
    best_dist = threshold
    for i, (r, c) in enumerate(pts):
        d = ((r - row) ** 2 + (c - col) ** 2) ** 0.5
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx
