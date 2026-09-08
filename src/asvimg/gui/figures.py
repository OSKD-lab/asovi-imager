"""Embed matplotlib figures into dearpygui as textures (replaces plt.show)."""

from __future__ import annotations

import numpy as np


def figure_to_rgba(fig) -> tuple[int, int, np.ndarray]:
    """Render a matplotlib Figure to ``(width, height, flat float32 RGBA)``."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.float32) / 255.0
    return w, h, rgba.reshape(-1)


class FigurePanels:
    """Manages one collapsing panel per figure key.

    Each figure gets its own ``collapsing_header`` (like the config sections)
    so it can be folded/unfolded independently.  Re-adding the same key
    replaces the previous figure (texture + panel are deleted and rebuilt so
    size changes are handled).  All dpg calls must run on the main/render
    thread.

    Tags are derived from a per-instance integer id (not from the key text), so
    keys that would sanitise to the same string never collide.  Textures use
    ``add_static_texture`` (which *copies* the pixel data into dpg) so the
    transient numpy buffer need not be retained for the texture's lifetime.
    """

    def __init__(self, container: str | int, texture_registry: str | int) -> None:
        self._container = container
        self._registry = texture_registry
        self._items: dict[str, tuple[str, str]] = {}  # key -> (panel_tag, tex_tag)
        self._open: dict[str, bool] = {}
        self._next_id = 0

    def add(self, key: str, fig) -> None:
        """Rasterise + embed a Figure (single-thread convenience path)."""
        w, h, data = figure_to_rgba(fig)
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:
            pass
        self.add_rgba(key, w, h, data)

    def add_rgba(self, key: str, w: int, h: int, data) -> None:
        """Embed pre-rasterised RGBA (main/render thread only)."""
        import dearpygui.dearpygui as dpg

        # Preserve the previous open/closed state when a figure is refreshed.
        default_open = self._open.get(key, True)
        if key in self._items:
            old_panel, old_tex = self._items[key]
            if dpg.does_item_exist(old_panel):
                default_open = dpg.get_value(old_panel)
            for t in (old_panel, old_tex):
                if dpg.does_item_exist(t):
                    dpg.delete_item(t)

        idx = self._next_id
        self._next_id += 1
        panel_tag = f"figpanel_{idx}"
        tex_tag = f"figtex_{idx}"

        # static_texture copies `data` into dpg, so the numpy buffer is safe to
        # drop after this call (unlike raw_texture, which is zero-copy).
        dpg.add_static_texture(
            width=w, height=h, default_value=data,
            tag=tex_tag, parent=self._registry,
        )
        with dpg.collapsing_header(
            label=key, parent=self._container, tag=panel_tag,
            default_open=default_open,
        ):
            dpg.add_image(tex_tag)
        self._items[key] = (panel_tag, tex_tag)
        self._open[key] = default_open

    def clear(self) -> None:
        import dearpygui.dearpygui as dpg

        for panel_tag, tex_tag in self._items.values():
            for t in (panel_tag, tex_tag):
                if dpg.does_item_exist(t):
                    dpg.delete_item(t)
        self._items.clear()
        self._open.clear()
