import configparser
import threading
from pathlib import Path

import dearpygui.dearpygui as dpg
import numpy as np
import tifffile

# ---------------------------------------------------------------------------
# sifx / dat readers
# ---------------------------------------------------------------------------

def _ordered_dat_files(path) -> int:
    """Sort key for Andor spool .dat files.

    Andor writes the sequence number with its digits *reversed* in the
    filename (e.g. sequence 10 → ``0100000000spool.dat``).  Reversing
    the digit string back recovers the true acquisition order.

    Adapted from ``sif_parser.utils.ordered_dat_files``.
    """
    numeric = "".join(filter(str.isdigit, Path(path).name))
    return int(numeric[::-1])


def read_info_from_sifx(path: str) -> dict:
    sifx = Path(path)
    spool_dir = sifx.parent

    ini_path = spool_dir / "acquisitionmetadata.ini"
    if not ini_path.exists():
        raise FileNotFoundError(f"acquisitionmetadata.ini not found in {spool_dir}")

    cfg = configparser.ConfigParser()
    cfg.read(ini_path, encoding="utf-8-sig")

    width = int(cfg["data"]["AOIWidth"])
    height = int(cfg["data"]["AOIHeight"])
    stride = int(cfg["data"]["AOIStride"])
    pixel_encoding = cfg["data"]["PixelEncoding"]
    image_size_bytes = int(cfg["data"]["ImageSizeBytes"])
    images_per_file = int(cfg["multiimage"]["ImagesPerFile"])
    
    dat_files = sorted(spool_dir.glob("*spool.dat"), key=_ordered_dat_files)

    return {
        "width": width,
        "height": height,
        "stride": stride,
        "pixel_encoding": pixel_encoding,
        "image_size_bytes": image_size_bytes,
        "images_per_file": images_per_file,
        "dat_files": [str(f) for f in dat_files],
        "total_frames": len(dat_files) * images_per_file,
        "sifx_path": str(sifx),
        "ini_path": str(ini_path),
    }

_PIXEL_DTYPES = {
    "Mono8": np.uint8,
    "Mono16": np.uint16,
    "Mono32": np.uint32,
}

def _read_single_frame(info: dict, global_idx: int) -> np.ndarray:
    """Read a single frame by flat index (file-sequential order)."""
    w = info["width"]
    h = info["height"]
    stride = info["stride"]
    img_bytes = info["image_size_bytes"]
    ipf = info["images_per_file"]
    dtype = _PIXEL_DTYPES[info["pixel_encoding"]]
    bpp = dtype().itemsize
    pixels_per_row = stride // bpp

    file_idx = global_idx // ipf
    local_idx = global_idx % ipf
    offset = local_idx * img_bytes
    raw = np.fromfile(info["dat_files"][file_idx], dtype=dtype,
                      count=pixels_per_row * h, offset=offset)
    return raw.reshape(h, pixels_per_row)[:, :w]

def _read_channel_frame(info: dict, ch: int, t: int, n_ch: int) -> np.ndarray:
    """Read frame for channel `ch` at timepoint `t`.

    Dat files are grouped by n_ch: each group of n_ch consecutive files
    corresponds to one block of timepoints. Within a group, file[ch]
    contains images_per_file frames for that channel.
    """
    ipf = info["images_per_file"]
    file_group = t // ipf
    file_idx = file_group * n_ch + ch
    local_frame = t % ipf
    return _read_single_frame(info, file_idx * ipf + local_frame)

def read_images_from_sifx(path: str, start_idx=0, end_idx=None) -> np.ndarray:
    info = read_info_from_sifx(path)
    total = info["total_frames"]
    if end_idx is None:
        end_idx = total
    end_idx = min(end_idx, total)
    frames = [_read_single_frame(info, i) for i in range(start_idx, end_idx)]
    return np.stack(frames)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _to_rgba(frame: np.ndarray) -> np.ndarray:
    f = frame.astype(np.float32)
    lo, hi = f.min(), f.max()
    f = (f - lo) / (hi - lo + 1e-8)
    rgba = np.zeros((*f.shape, 4), dtype=np.float32)
    rgba[:, :, 0] = rgba[:, :, 1] = rgba[:, :, 2] = f
    rgba[:, :, 3] = 1.0
    return rgba

# ---------------------------------------------------------------------------
# app state
# ---------------------------------------------------------------------------

_state: dict = {}

def _goto_step(step: int):
    for s in range(1, 5):
        dpg.configure_item(f"step_{s}", show=(s == step))

# ---------------------------------------------------------------------------
# Step 1: File selection
# ---------------------------------------------------------------------------

def _on_file_selected(sender, app_data):
    selections = app_data.get("selections", {})
    if not selections:
        return
    filepath = list(selections.values())[0]
    try:
        info = read_info_from_sifx(filepath)
    except Exception as e:
        dpg.set_value("info_text", f"Error: {e}")
        return
    _state["info"] = info
    dpg.set_value("info_text",
        f"File: {info['sifx_path']}\n"
        f"Size: {info['width']} x {info['height']}\n"
        f"Pixel: {info['pixel_encoding']}\n"
        f"Total frames: {info['total_frames']}\n"
        f"Dat files: {len(info['dat_files'])}")
    dpg.configure_item("btn_next_1", enabled=True)

def _on_next_1():
    _goto_step(2)

# ---------------------------------------------------------------------------
# Step 2: Channel config
# ---------------------------------------------------------------------------

def _on_back_2():
    _goto_step(1)

def _on_next_2():
    n_ch = dpg.get_value("input_n_ch")
    if n_ch < 1 or n_ch > 4:
        return
    _state["n_ch"] = n_ch
    _setup_preview()
    _goto_step(3)

# ---------------------------------------------------------------------------
# Step 3: Preview
# ---------------------------------------------------------------------------

def _setup_preview():
    info = _state["info"]
    n_ch = _state["n_ch"]
    h, w = info["height"], info["width"]
    ipf = info["images_per_file"]
    n_file_groups = len(info["dat_files"]) // n_ch
    n_timepoints = n_file_groups * ipf

    # clean up old textures, images, labels
    for tag in _state.get("tex_tags", []):
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
    for tag in _state.get("preview_items", []):
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)

    # create textures and labeled images
    blank = np.zeros((h, w, 4), dtype=np.float32)
    blank[:, :, 3] = 1.0
    tex_tags = []
    preview_items = []
    for ch in range(n_ch):
        tex_tag = f"tex_ch{ch}"
        grp_tag = f"grp_ch{ch}"
        lbl_tag = f"lbl_ch{ch}"
        img_tag = f"img_ch{ch}"
        if dpg.does_item_exist(tex_tag):
            dpg.delete_item(tex_tag)
        dpg.add_raw_texture(w, h, blank.flatten(),
                            format=dpg.mvFormat_Float_rgba,
                            tag=tex_tag, parent="tex_reg")
        with dpg.group(tag=grp_tag, parent="preview_row"):
            dpg.add_text(f"Ch{ch + 1}", tag=lbl_tag)
            dpg.add_image(tex_tag, tag=img_tag, width=w // 2, height=h // 2)
        tex_tags.append(tex_tag)
        preview_items.extend([grp_tag, lbl_tag, img_tag])

    _state["tex_tags"] = tex_tags
    _state["preview_items"] = preview_items
    _state["n_timepoints"] = n_timepoints

    dpg.configure_item("slider_t", max_value=max(n_timepoints - 1, 0))
    dpg.set_value("slider_t", 0)
    _update_preview(0)

def _on_slider_change(sender, app_data):
    _update_preview(app_data)

def _update_preview(t: int):
    info = _state["info"]
    n_ch = _state["n_ch"]
    for ch in range(n_ch):
        frame = _read_channel_frame(info, ch, t, n_ch)
        rgba = _to_rgba(frame)
        dpg.set_value(f"tex_ch{ch}", rgba.flatten())
    dpg.set_value("preview_label",
                  f"Timepoint {t} / {_state['n_timepoints'] - 1}")

def _on_slider_prev():
    t = max(dpg.get_value("slider_t") - 1, 0)
    dpg.set_value("slider_t", t)
    _update_preview(t)

def _on_slider_next():
    t = min(dpg.get_value("slider_t") + 1, _state.get("n_timepoints", 1) - 1)
    dpg.set_value("slider_t", t)
    _update_preview(t)

def _on_back_3():
    _goto_step(2)

def _on_next_3():
    info = _state["info"]
    default_out = str(Path(info["sifx_path"]).parent / "_converted")
    dpg.set_value("output_dir", default_out)
    _goto_step(4)

# ---------------------------------------------------------------------------
# Step 4: Convert
# ---------------------------------------------------------------------------

def _on_back_4():
    _goto_step(3)

def _on_convert():
    dpg.configure_item("btn_convert", enabled=False)
    dpg.set_value("convert_status", "Converting...")
    dpg.set_value("progress_bar", 0.0)
    t = threading.Thread(target=_convert_worker, daemon=True)
    t.start()

def _convert_worker():
    info = _state["info"]
    n_ch = _state["n_ch"]
    out_dir = Path(dpg.get_value("output_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ipf = info["images_per_file"]
    n_file_groups = len(info["dat_files"]) // n_ch
    n_timepoints = n_file_groups * ipf
    dtype = _PIXEL_DTYPES[info["pixel_encoding"]]

    for ch in range(n_ch):
        out_path = out_dir / f"ch{ch + 1}.tiff"
        frames = []
        for t in range(n_timepoints):
            frame = _read_channel_frame(info, ch, t, n_ch)
            frames.append(frame)
            progress = (ch * n_timepoints + t + 1) / (n_ch * n_timepoints)
            dpg.set_value("progress_bar", progress)
        stack = np.stack(frames).astype(dtype)
        tifffile.imwrite(str(out_path), stack, bigtiff=True)

    dpg.set_value("convert_status",
                  f"Done! Saved {n_ch} files to {out_dir}")
    dpg.configure_item("btn_convert", enabled=True)

# ---------------------------------------------------------------------------
# GUI layout
# ---------------------------------------------------------------------------

dpg.create_context()
dpg.create_viewport(title="sifx Converter", width=900, height=600)

with dpg.texture_registry(tag="tex_reg"):
    pass

with dpg.file_dialog(callback=_on_file_selected, show=False,
                     tag="file_dlg", width=700, height=400):
    dpg.add_file_extension(".sifx", color=(0, 255, 0, 255))
    dpg.add_file_extension(".*")

with dpg.window(label="sifx Converter", tag="primary"):

    # --- Step 1: File Selection ---
    with dpg.group(tag="step_1", show=True):
        dpg.add_text("Step 1: Select .sifx file")
        dpg.add_separator()
        dpg.add_button(label="Browse...",
                       callback=lambda: dpg.show_item("file_dlg"))
        dpg.add_text("", tag="info_text")
        dpg.add_separator()
        dpg.add_button(label="Next >>", tag="btn_next_1",
                       callback=_on_next_1, enabled=False)

    # --- Step 2: Channel Config ---
    with dpg.group(tag="step_2", show=False):
        dpg.add_text("Step 2: Number of channels")
        dpg.add_separator()
        dpg.add_text("Dat files are grouped by N files:\n"
                     "  Group[i] file 0 = Ch1, file 1 = Ch2, ...\n"
                     "  Each file has ImagesPerFile timepoints\n"
                     "  (N = number of channels)")
        dpg.add_input_int(label="Channels", tag="input_n_ch",
                          default_value=1, min_value=1, max_value=4,
                          min_clamped=True, max_clamped=True)
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_button(label="<< Back", callback=_on_back_2)
            dpg.add_button(label="Next >>", callback=_on_next_2)

    # --- Step 3: Preview ---
    with dpg.group(tag="step_3", show=False):
        dpg.add_text("Step 3: Preview")
        dpg.add_separator()
        dpg.add_text("", tag="preview_label")
        with dpg.group(horizontal=True):
            dpg.add_button(label="-", callback=_on_slider_prev, width=30)
            dpg.add_slider_int(label="Timepoint", tag="slider_t",
                               default_value=0, min_value=0, max_value=0,
                               callback=_on_slider_change, width=340)
            dpg.add_button(label="+", callback=_on_slider_next, width=30)
        with dpg.group(horizontal=True, tag="preview_row"):
            pass
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_button(label="<< Back", callback=_on_back_3)
            dpg.add_button(label="Next >>", callback=_on_next_3)

    # --- Step 4: Convert ---
    with dpg.group(tag="step_4", show=False):
        dpg.add_text("Step 4: Convert to BigTIFF")
        dpg.add_separator()
        dpg.add_input_text(label="Output dir", tag="output_dir")
        dpg.add_button(label="Convert", tag="btn_convert",
                       callback=_on_convert)
        dpg.add_progress_bar(tag="progress_bar", default_value=0.0,
                             width=400)
        dpg.add_text("", tag="convert_status")
        dpg.add_separator()
        dpg.add_button(label="<< Back", callback=_on_back_4)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("primary", True)
dpg.start_dearpygui()
dpg.destroy_context()
