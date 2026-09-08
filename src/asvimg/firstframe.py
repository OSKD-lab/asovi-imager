"""Interactive first-frame (BL/UV) detection dialog using tkinter."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import load_frames_by_indices


def _normalize_u8(arr: np.ndarray) -> np.ndarray:
    """Normalize a 2D array to uint8 for display."""
    arr = arr.astype(np.float64)
    mn, mx = arr.min(), arr.max()
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)


def detect_first_frame_gui(
    input_file: Path, n_frames: int = 12, thumbnail_width: int = 160
) -> bool:
    """Show first N frames and ask the user whether frame 0 is BL.

    Returns True if the first frame is Blue (BL), False otherwise.
    Raises RuntimeError if tkinter is not available.
    """
    try:
        import tkinter as tk
        from PIL import Image, ImageTk
    except ImportError as e:
        raise RuntimeError(
            "tkinter or Pillow not available. "
            "Set start_frame_bl manually in config."
        ) from e

    indices = list(range(min(n_frames, 100)))
    frames = load_frames_by_indices(input_file, indices)
    n_loaded = len(frames)

    result: list[bool] = []

    root = tk.Tk()
    root.title("First Frame Detection — Is frame 0 Blue (BL)?")

    # Build thumbnail grid
    cols = min(6, n_loaded)
    rows = (n_loaded + cols - 1) // cols
    tk_images: list[ImageTk.PhotoImage] = []

    frame_grid = tk.Frame(root)
    frame_grid.pack(padx=8, pady=8)

    for idx, arr in enumerate(frames):
        u8 = _normalize_u8(arr)
        h, w = u8.shape[:2]
        scale = thumbnail_width / max(w, 1)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = Image.fromarray(u8, mode="L").resize(
            (new_w, new_h), Image.NEAREST
        )
        tk_img = ImageTk.PhotoImage(img)
        tk_images.append(tk_img)

        r, c = divmod(idx, cols)
        cell = tk.Frame(frame_grid)
        cell.grid(row=r, column=c, padx=4, pady=4)
        tk.Label(cell, image=tk_img).pack()
        tk.Label(cell, text=f"frame {idx}").pack()

    # Buttons
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=12)

    def on_yes() -> None:
        result.append(True)
        root.destroy()

    def on_no() -> None:
        result.append(False)
        root.destroy()

    tk.Label(
        btn_frame, text="Is the 1st frame Blue (BL)?", font=("", 14)
    ).pack(pady=(0, 8))
    tk.Button(
        btn_frame, text="Yes — BL first", command=on_yes,
        width=16, height=2,
    ).pack(side=tk.LEFT, padx=8)
    tk.Button(
        btn_frame, text="No — UV first", command=on_no,
        width=16, height=2,
    ).pack(side=tk.LEFT, padx=8)

    root.protocol("WM_DELETE_WINDOW", on_yes)  # default to BL if closed
    root.mainloop()

    return result[0] if result else True
