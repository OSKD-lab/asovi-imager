"""In-process movie preview: play every exported dF/F movie side by side.

Opened from the dashboard (main dpg context, like the ROI editor).  All movies
share one timeline and play simultaneously; a slider scrubs, an auto-play
button toggles playback, and radio buttons pick a x0.25 .. x4 speed.  The
window advances via :meth:`MoviePreview.tick`, which the dashboard render loop
calls once per frame.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

_WIN = "movie_preview_win"
_TEXREG = "movie_preview_texreg"
_SLIDER = "movie_preview_slider"
_PLAY = "movie_preview_play"
_STATUS = "movie_preview_status"
_ROW_THEME = "movie_preview_rowtheme"

# label -> playback-speed multiplier (relative to each movie's encoded fps)
_SPEEDS = {"x0.25": 0.25, "x0.5": 0.5, "x1": 1.0, "x2": 2.0, "x4": 4.0}

_MOVIE_EXTS = ("*.avi", "*.mp4", "*.mov", "*.mkv")

_DISP_H = 320          # target display height per movie (px)
_MAX_ROW_W = 1500      # shrink display height if the row would exceed this


def find_movies(output_dir) -> list[Path]:
    """Every exported movie under ``<output_dir>/../movies`` (sorted by name)."""
    d = Path(output_dir) / "movies"
    if not d.is_dir():
        return []
    out: list[Path] = []
    for pat in _MOVIE_EXTS:
        out += d.glob(pat)
    return sorted(set(out))


class MoviePreview:
    def __init__(self, movie_paths: list[Path], base_fps: float | None = None) -> None:
        self.paths = [Path(p) for p in movie_paths]
        self.frames: list[list[np.ndarray]] = []   # per movie: RGBA uint8 frames
        self.sizes: list[tuple[int, int]] = []      # per movie: (w, h) native
        self.disp: list[tuple[int, int]] = []        # per movie: (w, h) displayed
        self.tex_tags: list[str] = []
        # x1 = this many frames/sec.  Callers pass the realtime acquisition rate
        # (fps / cycle_len) so x1 is realtime regardless of save_movie_speed,
        # which the encoded AVI fps already bakes in; else fall back to the
        # encoded fps.
        self._base_override = base_fps
        self.base_fps = 20.0
        self.total = 1
        self.speed = 1.0
        self.playing = False
        self.pos = 0.0            # float frame index on the shared timeline
        self._shown = -1
        self._last: float | None = None

    # --- load -------------------------------------------------------------

    def _load(self) -> None:
        import cv2

        fps_vals: list[float] = []
        for p in self.paths:
            cap = cv2.VideoCapture(str(p))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 0:
                fps_vals.append(float(fps))
            frames: list[np.ndarray] = []
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                rgba = np.empty((h, w, 4), dtype=np.uint8)
                rgba[:, :, :3] = rgb
                rgba[:, :, 3] = 255
                frames.append(rgba)
            cap.release()
            if not frames:  # unreadable movie -> 1 black frame so the layout holds
                frames = [np.zeros((_DISP_H, _DISP_H, 4), dtype=np.uint8)]
            self.frames.append(frames)
            h, w = frames[0].shape[:2]
            self.sizes.append((w, h))

        if self._base_override and self._base_override > 0:
            self.base_fps = float(self._base_override)
        else:
            self.base_fps = fps_vals[0] if fps_vals else 20.0
        self.total = max(len(f) for f in self.frames)
        self._compute_display_sizes()

    def _compute_display_sizes(self) -> None:
        # scale every movie to a common display height, shrinking if the row of
        # movies would be wider than _MAX_ROW_W (kept side by side, no gaps).
        disp_h = float(_DISP_H)
        row_w = sum(w * disp_h / h for (w, h) in self.sizes)
        if row_w > _MAX_ROW_W:
            disp_h *= _MAX_ROW_W / row_w
        self.disp = [
            (max(1, int(round(w * disp_h / h))), max(1, int(round(disp_h))))
            for (w, h) in self.sizes
        ]

    # --- open / close -----------------------------------------------------

    def open(self) -> "MoviePreview":
        import dearpygui.dearpygui as dpg

        for tag in (_WIN, _TEXREG, _ROW_THEME):
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)

        self._load()

        with dpg.texture_registry(tag=_TEXREG):
            for i, (w, h) in enumerate(self.sizes):
                tag = f"movie_tex_{i}"
                dpg.add_dynamic_texture(
                    width=w, height=h, default_value=self._flat(i, 0), tag=tag
                )
                self.tex_tags.append(tag)

        # zero item spacing so the movies sit flush against each other
        with dpg.theme(tag=_ROW_THEME):
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 0, 0)

        row_w = sum(w for (w, _) in self.disp)
        row_h = max(h for (_, h) in self.disp)
        with dpg.window(
            label="Movie preview", tag=_WIN,
            width=min(_MAX_ROW_W + 40, row_w + 40), height=row_h + 152,
            on_close=self.close,
        ):
            with dpg.group(horizontal=True) as row:
                for i, tag in enumerate(self.tex_tags):
                    w, h = self.disp[i]
                    dpg.add_image(tag, width=w, height=h)
            dpg.bind_item_theme(row, _ROW_THEME)

            dpg.add_slider_int(
                tag=_SLIDER, width=-1, min_value=0,
                max_value=max(0, self.total - 1), default_value=0,
                callback=self._on_slider,
            )
            with dpg.group(horizontal=True):
                dpg.add_button(label="> Play", tag=_PLAY, callback=self._toggle_play)
                dpg.add_text("  Speed:")
                dpg.add_radio_button(
                    list(_SPEEDS.keys()), default_value="x1", horizontal=True,
                    callback=self._on_speed,
                )
            dpg.add_text("", tag=_STATUS)  # own line (below the Play/Speed controls)

        self._render_current(force=True)
        self._update_status()
        return self

    def is_open(self) -> bool:
        import dearpygui.dearpygui as dpg

        return dpg.does_item_exist(_WIN)

    def close(self, *args) -> None:
        import dearpygui.dearpygui as dpg

        self.playing = False
        for tag in (_WIN, _TEXREG, _ROW_THEME):
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)

    # --- frame data -------------------------------------------------------

    def _flat(self, movie_idx: int, frame_idx: int) -> np.ndarray:
        frames = self.frames[movie_idx]
        f = frames[min(frame_idx, len(frames) - 1)]
        return (f.astype(np.float32) / 255.0).ravel()

    def _render_current(self, force: bool = False) -> None:
        import dearpygui.dearpygui as dpg

        cur = int(self.pos) % self.total if self.total else 0
        if not force and cur == self._shown:
            return
        self._shown = cur
        for i, tag in enumerate(self.tex_tags):
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, self._flat(i, cur))
        if dpg.does_item_exist(_SLIDER):
            dpg.set_value(_SLIDER, cur)

    # --- callbacks --------------------------------------------------------

    def _toggle_play(self, *args) -> None:
        import dearpygui.dearpygui as dpg

        self.playing = not self.playing
        self._last = time.perf_counter() if self.playing else None
        if dpg.does_item_exist(_PLAY):
            dpg.set_item_label(_PLAY, "|| Pause" if self.playing else "> Play")

    def _on_speed(self, sender, value) -> None:
        self.speed = _SPEEDS.get(value, 1.0)
        self._update_status()

    def _on_slider(self, sender, value) -> None:
        self.pos = float(value)
        self._render_current(force=True)

    def _update_status(self) -> None:
        import dearpygui.dearpygui as dpg

        if dpg.does_item_exist(_STATUS):
            dpg.set_value(
                _STATUS,
                f"{len(self.paths)} movie(s), {self.total} frames  "
                f"x1 = {self.base_fps:.1f} fps  |  now x{self.speed:g} "
                f"({self.base_fps * self.speed:.1f} fps)",
            )

    # --- per-frame tick (driven by the dashboard render loop) -------------

    def tick(self) -> None:
        if not self.is_open():
            return
        if self.playing and self.total > 1:
            now = time.perf_counter()
            if self._last is not None:
                self.pos += (now - self._last) * self.base_fps * self.speed
                if self.pos >= self.total:
                    self.pos %= self.total  # loop
            self._last = now
            self._render_current()


def open_preview(movie_paths: list[Path], base_fps: float | None = None) -> MoviePreview:
    return MoviePreview(movie_paths, base_fps=base_fps).open()
