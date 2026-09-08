"""Dear PyGui dialog for ICA component exclusion selection."""

from __future__ import annotations

import numpy as np


def ica_select_gui(
    ica_result,
    *,
    title: str = "ICA Component Selection - Click to EXCLUDE",
    preselected: list[int] | None = None,
) -> list[int] | None:
    """Open a Dear PyGui window to select IC components for exclusion.

    Displays the IC spatial maps in a scrollable grid. Click a component to
    toggle its exclusion (red frame = excluded). The maps are rendered with the
    same symmetric ``RdBu_r`` range used elsewhere (``±`` 99th-percentile of
    ``|value|``, robust to outlier pixels).

    Like :func:`asvimg.cpselect.cpselect_gui`, this creates and destroys its own
    dearpygui context, so it must run in a process that has no other live
    context (the dashboard launches it in a child process via
    ``gui/subproc_ica.py``).

    Parameters
    ----------
    ica_result : IcaResult from compute_ica.
    title : window title.

    Returns
    -------
    Sorted list of excluded component indices (0-based), or None if cancelled.
    """
    import dearpygui.dearpygui as dpg
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    spatial_maps = np.asarray(ica_result.spatial)
    n_ica = spatial_maps.shape[0]
    if n_ica == 0:
        return []

    # --- Render each IC spatial map to an RGBA thumbnail texture ---
    rgbas: list[np.ndarray] = []
    for i in range(n_ica):
        spatial = spatial_maps[i]
        vmax = float(np.percentile(np.abs(spatial), 99)) or 1.0  # robust to outliers
        norm = Normalize(vmin=-vmax, vmax=vmax)
        sm = ScalarMappable(norm=norm, cmap="RdBu_r")
        rgbas.append(sm.to_rgba(spatial).astype(np.float32))  # (H, W, 4) in [0, 1]
    tex_h, tex_w = rgbas[0].shape[:2]

    # Grid geometry — keep the map's aspect ratio within a `thumb`-px box.
    thumb = 110
    if tex_w >= tex_h:
        disp_w = thumb
        disp_h = max(1, round(thumb * tex_h / tex_w))
    else:
        disp_h = thumb
        disp_w = max(1, round(thumb * tex_w / tex_h))
    n_cols = min(5, n_ica)  # wrap at 5 components per row
    n_rows = -(-n_ica // n_cols)  # ceil division

    # Re-opening the picker starts from the choice already recorded for this
    # group, so a second pass is a review, not a re-do.
    excluded: set[int] = {
        int(i) for i in (preselected or []) if 0 <= int(i) < n_ica
    }
    result: list[list[int] | None] = [None]
    should_close = [False]

    def _fmt_status() -> str:
        if excluded:
            nums = ", ".join(str(i + 1) for i in sorted(excluded))
            return f"Excluded ICs ({len(excluded)}): {nums}"
        return "Excluded ICs: (none)"

    # ---- Dear PyGui setup ----
    dpg.create_context()
    dpg.create_viewport(
        title=title,
        width=min(n_cols * (thumb + 16) + 60, 1500),
        height=min(2 * (n_rows * (thumb + 42) + 170), 1900),  # 2x taller
    )

    with dpg.texture_registry():
        # add_static_texture is always RGBA float (no `format` kwarg).
        tex_ids = [
            dpg.add_static_texture(tex_w, tex_h, rgba.flatten())
            for rgba in rgbas
        ]

    # Frame themes bound per-thumbnail: gray = kept, red = excluded. The button
    # frame padding is filled with the theme colour, so it reads as a border.
    def _frame_theme(rgb: tuple[int, int, int]) -> int:
        hov = tuple(min(255, c + 40) for c in rgb) + (255,)
        act = tuple(min(255, c + 60) for c in rgb) + (255,)
        with dpg.theme() as th:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Button, rgb + (255,))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, hov)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, act)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 4, 4)
        return th

    incl_theme = _frame_theme((70, 70, 70))
    excl_theme = _frame_theme((200, 40, 40))

    def _on_toggle(sender, app_data, user_data):
        idx = user_data
        if idx in excluded:
            excluded.discard(idx)
            dpg.bind_item_theme(sender, incl_theme)
        else:
            excluded.add(idx)
            dpg.bind_item_theme(sender, excl_theme)
        dpg.set_value("status_text", _fmt_status())

    def _on_clear(sender, app_data):
        for idx in list(excluded):
            dpg.bind_item_theme(f"icbtn_{idx}", incl_theme)
        excluded.clear()
        dpg.set_value("status_text", _fmt_status())

    def _on_apply(sender, app_data):
        result[0] = sorted(excluded)
        should_close[0] = True

    def _on_cancel(sender, app_data):
        result[0] = None
        should_close[0] = True

    with dpg.window(tag="primary", label=title):
        with dpg.group(horizontal=True):
            dpg.add_button(label="Apply", callback=_on_apply, width=90)
            dpg.add_button(label="Cancel", callback=_on_cancel, width=90)
            dpg.add_button(label="Clear all", callback=_on_clear, width=90)
            dpg.add_spacer(width=20)
            dpg.add_text(
                "Click a component to toggle EXCLUDE (red frame = excluded)."
            )
        dpg.add_text(_fmt_status(), tag="status_text")
        dpg.add_separator()

        with dpg.child_window(width=-1, height=-1, border=False):
            for r in range(n_rows):
                with dpg.group(horizontal=True):
                    for c in range(n_cols):
                        idx = r * n_cols + c
                        if idx >= n_ica:
                            break
                        with dpg.group():
                            btn = dpg.add_image_button(
                                tex_ids[idx],
                                width=disp_w,
                                height=disp_h,
                                callback=_on_toggle,
                                user_data=idx,
                                tag=f"icbtn_{idx}",
                            )
                            dpg.bind_item_theme(
                                btn, excl_theme if idx in excluded else incl_theme
                            )
                            dpg.add_text(f"IC {idx + 1}")

    dpg.set_primary_window("primary", True)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    # Manual render loop — flag-based shutdown (matches cpselect_gui).
    while dpg.is_dearpygui_running() and not should_close[0]:
        dpg.render_dearpygui_frame()

    dpg.destroy_context()
    return result[0]
