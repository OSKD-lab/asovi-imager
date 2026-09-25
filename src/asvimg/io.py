"""I/O utilities for ASoVi imaging pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import h5py
import imageio.v3 as iio
import numpy as np
import tifffile
from scipy.io import loadmat, savemat

from .dcimg import get_dcimg_file_class
from .nd2rec import Nd2Recording
from .sifx import SIFXFile

SUPPORTED_OUTPUT_FORMATS = {"mat", "npy", "h5"}
# Single-file stack extensions. An Ito even/odd HDF5 recording is instead a
# *folder* of .h5 parts (see h5rec.H5Recording), dispatched on ``path.is_dir()``.
SUPPORTED_INPUT_EXTENSIONS = (".tif", ".tiff", ".dcimg", ".sifx", ".nd2", ".h5")

# Read orders find_input_files can impose on a multi-file recording. The order
# is the concatenation order of the whole timeline, so it decides where every
# frame lands (and, through channels_slip, which channel it is called).
#   "natural" — natsort by name: rec_2 before rec_10. The default; matches how
#               acquisition software names spooled parts, and also what Windows
#               Explorer shows (it sorts numerically too, via StrCmpLogicalW).
#   "name"    — plain lexicographic by name: rec_10 before rec_2. This is the
#               byte-order sort MATLAB `dir` / Python `sorted()` give, i.e. what
#               a legacy script that globbed and sorted would have read.
#   "mtime"   — oldest modification time first. For camera output this is when
#               the file was *written*, i.e. acquisition order.
#   "ctime"   — oldest creation time first. CAUTION: on Windows a file copied or
#               restored from a backup gets a *new* creation time, so on a copied
#               dataset this is the copy order (often shuffled), not acquisition
#               order. Check the timestamps in "Quick Preview (All)" before
#               trusting it; "mtime" survives a copy, ctime usually does not.
_INPUT_ORDERS: tuple[str, ...] = ("natural", "name", "mtime", "ctime")


def _order_files(files: list[Path], input_order: str) -> list[Path]:
    """Sort ``files`` by ``input_order`` (see :data:`_INPUT_ORDERS`).

    Time orders tie-break in *natural* name order, so a batch the filesystem
    reports one identical timestamp for (a whole directory copied at once, or a
    coarse clock) is still deterministic and degrades to exactly the default
    order rather than to a lexicographic one.
    """
    if input_order not in _INPUT_ORDERS:
        raise ValueError(
            f"input_order must be one of {_INPUT_ORDERS}, got {input_order!r}"
        )
    if input_order == "name":
        return sorted(files, key=lambda p: p.name)

    from natsort import natsort_keygen

    nkey = natsort_keygen()
    if input_order in ("mtime", "ctime"):
        attr = "st_mtime" if input_order == "mtime" else "st_ctime"
        # stat() runs once per file (a key function, not a comparator) — it is a
        # network round-trip per file on a share, so this matters.
        return sorted(files, key=lambda p: (getattr(p.stat(), attr), nkey(p.name)))
    return sorted(files, key=lambda p: nkey(p.name))


def _detect_input_formats(
    input_dir: Path, input_order: str = "natural"
) -> dict[str, list[Path]]:
    """Map each input format present in ``input_dir`` to its files.

    Keys are ``"tif"`` (.tif/.tiff), ``"dcimg"``, ``"sifx"``, ``"nd2"``, ``"h5"`` (an Ito
    even/odd recording, represented by the folder itself). Files within a format
    are sorted by ``input_order`` (default natural order: ``file_2`` before
    ``file_10``).
    """
    present: dict[str, list[Path]] = {}
    tif = [*input_dir.glob("*.tif"), *input_dir.glob("*.tiff")]
    if tif:
        present["tif"] = _order_files(tif, input_order)
    for fmt, pattern in (("dcimg", "*.dcimg"), ("sifx", "*.sifx"), ("nd2", "*.nd2")):
        hit = _order_files(list(input_dir.glob(pattern)), input_order)
        if hit:
            present[fmt] = hit

    from .h5rec import is_ito_h5_recording

    if is_ito_h5_recording(input_dir):
        # A recording is a folder of parts merged into one source-index-ordered
        # stream, so it is a single entry (the folder); read via H5Recording.
        present["h5"] = [Path(input_dir)]
    return present


def find_input_files(
    input_dir: Path, input_format: str = "auto", input_order: str = "natural"
) -> list[Path]:
    """Supported imaging inputs in ``input_dir``, in the configured read order.

    ``input_format`` selects the format to read: ``"auto"`` detects the single
    format present and **raises when a folder mixes formats** (so the pipeline
    never silently reads one of several); ``"tif"``/``"dcimg"``/``"sifx"``/``"nd2"``/``"h5"``
    force that one.

    ``input_order`` is the order multi-file recordings of one format are
    concatenated in — ``"natural"`` (default; matches how acquisition software
    names spooled/segmented files), ``"name"``, ``"mtime"`` or ``"ctime"``. See
    :data:`_INPUT_ORDERS`. This is the single chokepoint every caller goes
    through, so the GUI preview and the run always agree on the order.
    """
    input_dir = Path(input_dir)
    present = _detect_input_formats(input_dir, input_order)

    if input_format != "auto":
        files = present.get(input_format)
        if not files:
            have = ", ".join(sorted(present)) or "none"
            raise FileNotFoundError(
                f"input_format={input_format!r} but no {input_format} input found "
                f"in {input_dir} (formats present: {have})"
            )
        return files

    if not present:
        raise FileNotFoundError(
            f"No supported files found in {input_dir} "
            f"(expected .tif/.tiff/.dcimg/.sifx/.nd2 or an even/odd .h5 recording folder)"
        )
    if len(present) > 1:
        raise ValueError(
            f"{input_dir} contains multiple input formats "
            f"({', '.join(sorted(present))}); set input_format to one of them to choose."
        )
    (files,) = present.values()
    return files


def resolve_exp_stem(
    input_dir,
    exp_name: str | None = None,
    input_format: str = "auto",
    input_order: str = "natural",
) -> str:
    """Experiment stem used in ``roiSignals_{name}_{stem}`` filenames.

    Matches the notebook's ``_exp_stem`` derivation: the stem of the first
    input file, falling back to ``exp_name`` (or ``"output"``) when the input
    directory has no readable imaging files (e.g. when pointing at an
    output_dir copied from another machine, or when the format is ambiguous —
    the real error then surfaces when the stage actually reads the input).

    Note that ``input_order`` therefore changes the stem: whichever file sorts
    first names it. ``exp_name`` does **not** pin this — it is only the fallback
    for a directory with no readable input, so re-ordering renames
    ``roiSignals_*`` even when ``exp_name`` is set.
    """
    try:
        files = find_input_files(Path(input_dir), input_format, input_order)
    except (FileNotFoundError, ValueError):
        files = []
    if files:
        return files[0].stem
    return exp_name or "output"


def input_file_times(path: Path) -> tuple[float, float]:
    """``(ctime, mtime)`` for an input file or recording folder, ``(0, 0)`` if
    it cannot be stat'd. Both are surfaced in the preview so a bad ``ctime``
    (copied/restored data) is visible before a run, not after one."""
    try:
        st = Path(path).stat()
    except OSError:
        return (0.0, 0.0)
    return (st.st_ctime, st.st_mtime)


def get_frame_count(path: Path) -> int:
    """Return frame count for a supported image stack."""
    if path.is_dir():  # Ito even/odd HDF5 recording folder
        from .h5rec import H5Recording

        with H5Recording(path) as rec:
            return rec.nfrms

    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        with tifffile.TiffFile(path) as tif:
            return len(tif.pages)

    if suffix == ".dcimg":
        dcimg_cls = get_dcimg_file_class()
        with dcimg_cls(path) as dcimg:
            return int(dcimg.nfrms)

    if suffix == ".sifx":
        with SIFXFile(path) as sifx:
            return sifx.effective_nfrms

    if suffix == ".nd2":
        with Nd2Recording(path) as rec:
            return rec.nfrms

    raise ValueError(f"Unsupported input extension: {path.suffix}")


def iter_frames(path: Path) -> Iterator[np.ndarray]:
    """Yield frames from a supported image stack."""
    if path.is_dir():  # Ito even/odd HDF5 recording folder
        from .h5rec import H5Recording

        with H5Recording(path) as rec:
            for idx in range(rec.nfrms):
                yield rec.frame(idx, copy=False)
        return

    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        with tifffile.TiffFile(path) as tif:
            for page in tif.pages:
                yield page.asarray()
        return

    if suffix == ".dcimg":
        dcimg_cls = get_dcimg_file_class()
        with dcimg_cls(path) as dcimg:
            for idx in range(int(dcimg.nfrms)):
                yield np.asarray(dcimg.frame(idx, copy=False))
        return

    if suffix == ".sifx":
        with SIFXFile(path) as sifx:
            for idx in range(sifx.effective_nfrms):
                yield sifx.frame(idx, copy=False)
        return

    if suffix == ".nd2":
        with Nd2Recording(path) as rec:
            yield from rec.iter_frames()
        return

    raise ValueError(f"Unsupported input extension: {path.suffix}")


def iter_frames_with_metadata(
    path: Path,
    dcimg_backend: str = "auto",
) -> Iterator[tuple[np.ndarray, dict[str, Any]]]:
    """Yield (frame, metadata) for a supported image stack.

    TIFF/TIFF metadata includes frame index only (timestamp is unavailable).
    DCIMG metadata includes timestamps and stamps when available.
    """
    if path.is_dir():  # Ito even/odd HDF5 recording folder
        from .h5rec import H5Recording

        with H5Recording(path) as rec:
            for idx in range(rec.nfrms):
                yield rec.frame_with_metadata(idx, copy=False)
        return

    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        with tifffile.TiffFile(path) as tif:
            for idx, page in enumerate(tif.pages):
                frame = page.asarray()
                yield (
                    frame,
                    {
                        "frame_index": int(idx),
                        "timestamp_sec": None,
                        "timestamp_microsec": None,
                        "framestamp": None,
                        "camerastamp": None,
                        "image_height": int(frame.shape[0]),
                        "image_width": int(frame.shape[1]),
                        "dtype": str(frame.dtype),
                        "bits_per_pixel": int(frame.dtype.itemsize * 8),
                    },
                )
        return

    if suffix == ".dcimg":
        dcimg_cls = get_dcimg_file_class()
        with dcimg_cls(path, backend=dcimg_backend) as dcimg:
            for idx in range(int(dcimg.nfrms)):
                frame, meta = dcimg.frame_with_metadata(idx, copy=False)
                meta = {
                    **meta,
                    "image_height": int(frame.shape[0]),
                    "image_width": int(frame.shape[1]),
                    "dtype": str(frame.dtype),
                    "bits_per_pixel": int(frame.dtype.itemsize * 8),
                }
                yield np.asarray(frame), meta
        return

    if suffix == ".sifx":
        with SIFXFile(path) as sifx:
            for idx in range(sifx.effective_nfrms):
                frame, meta = sifx.frame_with_metadata(idx, copy=False)
                yield np.asarray(frame), meta
        return

    if suffix == ".nd2":
        with Nd2Recording(path) as rec:
            for idx in range(rec.nfrms):
                yield rec.frame_with_metadata(idx)
        return

    raise ValueError(f"Unsupported input extension: {path.suffix}")


def load_frames_by_indices(path: Path, indices: list[int]) -> list[np.ndarray]:
    """Load specific frame indices from a supported image stack."""
    if not indices:
        return []

    if path.is_dir():  # Ito even/odd HDF5 recording folder
        from .h5rec import H5Recording

        with H5Recording(path) as rec:
            return [rec.frame(int(idx), copy=True).astype(np.float64) for idx in indices]

    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        with tifffile.TiffFile(path) as tif:
            return [tif.pages[idx].asarray().astype(np.float64) for idx in indices]

    if suffix == ".dcimg":
        dcimg_cls = get_dcimg_file_class()
        with dcimg_cls(path) as dcimg:
            return [
                dcimg.frame(int(idx), copy=False).astype(np.float64) for idx in indices
            ]

    if suffix == ".sifx":
        with SIFXFile(path) as sifx:
            return [
                sifx.frame(int(idx), copy=True).astype(np.float64) for idx in indices
            ]

    if suffix == ".nd2":
        with Nd2Recording(path) as rec:
            return [rec.frame(int(idx)).astype(np.float64) for idx in indices]

    raise ValueError(f"Unsupported input extension: {path.suffix}")


def load_dcimg_metadata(
    path: Path,
    frame_indices: list[int] | None = None,
    backend: str = "auto",
) -> dict[str, Any]:
    """Load DCIMG metadata.

    Parameters
    ----------
    path : Path
        Target .dcimg file path.
    frame_indices : list[int] | None
        Frame indices to fetch per-frame metadata for.
        If None, returns summary only.
    backend : str
        "auto" (default), "sdk", or "native".
    """
    dcimg_cls = get_dcimg_file_class()
    with dcimg_cls(path, backend=backend) as dcimg:
        out: dict[str, Any] = {
            "path": str(path),
            "absolute_path": str(path.resolve()),
            "summary": dcimg.metadata_summary(),
            "frames": [],
        }

        if frame_indices:
            for idx in frame_indices:
                out["frames"].append(dcimg.frame_metadata(int(idx)))

        return out


def load_input_metadata(
    path: Path,
    frame_indices: list[int] | None = None,
    backend: str = "auto",
) -> dict[str, Any]:
    """Load metadata from a supported input file.

    For TIFF/TIFF returns basic shape/count metadata.
    For DCIMG returns detailed summary and optional per-frame metadata.
    """
    if path.is_dir():  # Ito even/odd HDF5 recording folder
        from .h5rec import H5Recording

        with H5Recording(path) as rec:
            out: dict[str, Any] = {
                "path": str(path),
                "absolute_path": str(path.resolve()),
                "summary": rec.metadata_summary(),
                "frames": [],
            }
            if frame_indices:
                for idx in frame_indices:
                    out["frames"].append(rec.frame_metadata(int(idx)))
            return out

    suffix = path.suffix.lower()

    if suffix in (".tif", ".tiff"):
        with tifffile.TiffFile(path) as tif:
            first = tif.pages[0].asarray()
            return {
                "path": str(path),
                "absolute_path": str(path.resolve()),
                "summary": {
                    "backend": "tiff",
                    "nfrms": len(tif.pages),
                    "height": int(first.shape[0]),
                    "width": int(first.shape[1]),
                    "dtype": str(first.dtype),
                    "bits_per_pixel": int(first.dtype.itemsize * 8),
                },
                "frames": [],
            }

    if suffix == ".dcimg":
        return load_dcimg_metadata(path, frame_indices=frame_indices, backend=backend)

    if suffix == ".sifx":
        with SIFXFile(path) as sifx:
            out: dict[str, Any] = {
                "path": str(path),
                "absolute_path": str(path.resolve()),
                "summary": sifx.metadata_summary(),
                "frames": [],
            }
            if frame_indices:
                for idx in frame_indices:
                    out["frames"].append(sifx.frame_metadata(int(idx)))
            return out

    if suffix == ".nd2":
        with Nd2Recording(path) as rec:
            out = {
                "path": str(path),
                "absolute_path": str(path.resolve()),
                "summary": rec.metadata_summary(),
                "frames": [],
            }
            if frame_indices:
                for idx in frame_indices:
                    out["frames"].append(rec.frame_metadata(int(idx)))
            return out

    raise ValueError(f"Unsupported input extension: {path.suffix}")


def load_template_mat(path: Path) -> np.ndarray:
    """Load template image from .mat or HDF5 .mat file."""
    try:
        data = loadmat(path)
        if "proc_template" in data:
            return np.asarray(data["proc_template"])
    except NotImplementedError:
        pass

    with h5py.File(path, "r") as h5f:
        if "proc_template" in h5f:
            return np.asarray(h5f["proc_template"]).T
        keys = list(h5f.keys())
        if not keys:
            raise ValueError(f"No datasets found in {path}")
        return np.asarray(h5f[keys[0]]).T


def load_template_image(path: Path) -> np.ndarray:
    """Load a 2D template image from a file, dispatching on extension.

    Supports .mat (``proc_template`` key), .npy, .tif/.tiff, and any format
    handled by ``imageio`` (png/jpg/bmp/...). Multi-channel images are
    reduced to grayscale by averaging across channels.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Template image not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".mat":
        img = load_template_mat(path)
    elif suffix == ".npy":
        img = np.load(path)
    elif suffix in (".tif", ".tiff"):
        img = tifffile.imread(path)
    else:
        img = np.asarray(iio.imread(path))

    img = np.asarray(img)
    if img.ndim == 3:
        # (H, W, C) colour image → grayscale by channel mean
        img = img.mean(axis=-1)
    if img.ndim != 2:
        raise ValueError(
            f"Template image must be 2D (H, W); got shape {img.shape} from {path}"
        )
    return img.astype(np.float64, copy=False)


def write_h5_payload(group: h5py.Group, payload: dict[str, Any]) -> None:
    """Write a dict of arrays/scalars to an HDF5 group.

    Strings need saying out loud: HDF5 has no fixed-width unicode type, so a
    numpy ``<U`` array (``roi_names``, ``channels_name``) raises "no conversion
    path" and takes the whole stage down. They are written as variable-length
    UTF-8, which is what h5py hands back as bytes on read.
    """
    for key, value in payload.items():
        arr = value if isinstance(value, np.ndarray) else None
        if arr is None and isinstance(value, (list, tuple)) and value:
            arr = np.asarray(value)

        if arr is not None and arr.dtype.kind in "USO":
            group.create_dataset(
                key,
                data=[str(v) for v in np.asarray(arr).ravel().tolist()],
                dtype=h5py.string_dtype(),
            )
        elif arr is not None:
            group.create_dataset(key, data=arr)
        elif isinstance(value, (int, float, bool, np.integer, np.floating, str)):
            group.attrs[key] = value
        else:
            group.create_dataset(key, data=np.asarray(value))


def _with_format_suffix(path: Path, output_format: str) -> Path:
    """Append ``.<fmt>`` without clobbering dots already in the stem.

    ``Path.with_suffix`` replaces everything after the last dot, which corrupts
    names whose stem contains dots (e.g. an OME-TIFF stem ``run.ome``, or a
    ``frameRoiCh_run.ome_01`` batch name).  Callers pass suffix-less paths, so
    we simply append (idempotently).
    """
    suffix = f".{output_format}"
    path = Path(path)
    if path.name.endswith(suffix):
        return path
    return path.parent / (path.name + suffix)


# --------------------------------------------------------------------------
# MATLAB v7.3 (.mat, HDF5-backed) writer
# --------------------------------------------------------------------------
# scipy.io.savemat only writes MATLAB v5, whose per-array byte_count is a
# uint32 -> a hard 4 GiB per-variable limit.  Atlas-warped dF/F stacks can
# exceed that, so oversized payloads are written as MATLAB v7.3 instead: an
# HDF5 file following MATLAB's conventions (reversed dims + MATLAB_class attrs,
# preceded by the 128-byte MAT header inside a 512-byte userblock).  These load
# transparently via MATLAB `load`, with no size limit.

_MAT_V5_ELEMENT_LIMIT = 2**32  # uint32 byte_count ceiling for a v5 array

_NP_TO_MATLAB_CLASS = {
    "float64": "double", "float32": "single",
    "int8": "int8", "int16": "int16", "int32": "int32", "int64": "int64",
    "uint8": "uint8", "uint16": "uint16", "uint32": "uint32", "uint64": "uint64",
}


def _mat_platform_token() -> str:
    import sys

    return {"win32": "PCWIN64", "linux": "GLNXA64", "darwin": "MACI64"}.get(
        sys.platform, "PCWIN64"
    )


def _mat73_header() -> bytes:
    """128-byte MAT-file header MATLAB writes ahead of the HDF5 payload."""
    import time

    desc = (
        f"MATLAB 7.3 MAT-file, Platform: {_mat_platform_token()}, "
        f"Created on: {time.asctime(time.localtime())} HDF5 schema 1.00 ."
    )
    hdr = bytearray(b" " * 116)
    desc_b = desc.encode("latin1")[:116]
    hdr[: len(desc_b)] = desc_b
    hdr += b"\x00" * 8   # bytes 116..123: subsys data offset (unused)
    hdr += b"\x00\x02"   # bytes 124..125: version 0x0200
    hdr += b"IM"         # bytes 126..127: little-endian indicator
    return bytes(hdr)


def _mat73_write_var(h5f: h5py.File, name: str, value: Any) -> None:
    """Write one payload entry as a MATLAB-compatible HDF5 dataset."""
    # Strings -> MATLAB 'char' (uint16 code units, shape reversed from (1, N)).
    if isinstance(value, (str, bytes)):
        s = value.decode("latin1") if isinstance(value, bytes) else value
        codes = np.array([ord(c) for c in s], dtype=np.uint16).reshape(-1, 1)
        dset = h5f.create_dataset(name, data=codes)
        dset.attrs["MATLAB_class"] = np.bytes_(b"char")
        dset.attrs["MATLAB_int_decode"] = np.int32(2)
        return

    arr = np.asarray(value)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8)
        mclass = "logical"
    else:
        mclass = _NP_TO_MATLAB_CLASS.get(arr.dtype.name)
        if mclass is None:  # unsupported dtype -> fall back to double
            arr = arr.astype(np.float64)
            mclass = "double"

    # MATLAB arrays are >= 2-D: 0-D -> (1,1), 1-D -> (1, N) row vector.
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, arr.shape[0])

    rev_shape = arr.shape[::-1]  # HDF5 (C-order) == MATLAB (column-major) reversed
    if arr.ndim == 3 and arr.nbytes > 256 * 1024**2:
        # Stream frame-by-frame so we never hold a full transposed copy in RAM.
        d0, d1, d2 = arr.shape
        dset = h5f.create_dataset(
            name, shape=rev_shape, dtype=arr.dtype, chunks=(1, d1, d0)
        )
        for k in range(d2):
            dset[k, :, :] = arr[:, :, k].T
    else:
        dset = h5f.create_dataset(name, data=np.ascontiguousarray(arr.T))

    dset.attrs["MATLAB_class"] = np.bytes_(mclass.encode("latin1"))
    if mclass == "logical":
        dset.attrs["MATLAB_int_decode"] = np.int32(1)


def _savemat_v73(target: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as a MATLAB v7.3 (.mat) file (no 4 GiB array limit)."""
    with h5py.File(target, "w", userblock_size=512) as h5f:
        for name, value in payload.items():
            _mat73_write_var(h5f, name, value)
    with open(target, "r+b") as fh:  # stamp the MAT header into the userblock
        fh.write(_mat73_header())


def _payload_needs_v73(payload: dict[str, Any]) -> bool:
    """True when any array is too large for the MATLAB v5 (uint32) format."""
    return any(
        isinstance(v, np.ndarray) and v.nbytes >= _MAT_V5_ELEMENT_LIMIT - 4096
        for v in payload.values()
    )


def save_payload(path: Path, payload: dict[str, Any], output_format: str) -> None:
    """Save payload dict in the specified format (mat/npy/h5).

    ``mat`` writes MATLAB v5 via scipy; payloads with an array above the v5
    4 GiB limit are transparently written as MATLAB v7.3 instead (still ``.mat``,
    still loadable via MATLAB ``load``).
    """
    target = _with_format_suffix(path, output_format)
    if output_format == "mat":
        if _payload_needs_v73(payload):
            _savemat_v73(target, payload)
        else:
            savemat(target, payload)
    elif output_format == "npy":
        np.save(target, payload, allow_pickle=True)
    elif output_format == "h5":
        # Write beside the target and rename on success: a half-written .h5 left
        # by a failed save would otherwise be picked up as a valid result by the
        # next stage (correlation reads whatever roiSignals_* it finds).
        tmp = target.with_name(target.name + ".part")
        try:
            with h5py.File(tmp, "w") as h5f:
                write_h5_payload(h5f, payload)
            tmp.replace(target)
        finally:
            tmp.unlink(missing_ok=True)
    else:
        raise ValueError(f"Unsupported format: {output_format}")


# --------------------------------------------------------------------------
# Full-timeline loaders (concatenate ALL preprocess batches)
# --------------------------------------------------------------------------
# preprocess splits a recording into ``frameRoiCh_{exp}_{NN}`` /
# ``frameRoiDf_{name}_{exp}_{NN}`` files, one per ``batch_size`` batch.  These
# helpers stitch every batch back into a single ``(H, W, T_total)`` stack so
# downstream stages (PCA, ROI, movies, annotated exports) see the whole
# recording instead of only the first batch.  They are the single seam a
# future memmap-based layout can re-point without touching any call site.


def _sorted_batch_files(output_dir: Path, pattern: str) -> list[Path]:
    """Batch files matching ``pattern``, ordered by their trailing ``_NN``
    index (natural sort, so ``_2`` precedes ``_10``)."""
    from natsort import natsorted

    return natsorted(Path(output_dir).glob(pattern), key=lambda p: p.name)


def load_full_channel_stack(output_dir, ch_idx: int) -> np.ndarray:
    """Full-timeline registered channel as ``(H, W, T)`` for legacy consumers.

    Reads the internal ``reg_Ch{ch_idx}.npy`` (T, H, W) and presents it as a
    zero-copy ``(H, W, T)`` transpose view (per-frame access stays contiguous).
    """
    return np.moveaxis(load_reg_channel(output_dir, ch_idx), 0, 2)


def load_full_dff_stack(output_dir, name: str) -> np.ndarray:
    """Full-timeline dF/F as ``(H, W, T)`` float64 for legacy consumers.

    Reads the internal ``dff_{name}.npy`` (T, H, W); raises
    ``FileNotFoundError`` when the name has no linear-subtraction output.
    """
    return np.moveaxis(load_dff(output_dir, name), 0, 2).astype(np.float64)


def mean_source_stack(output_dir, indices: list[int]) -> np.ndarray:
    """Average of the given registered channels as ``(H, W, T)`` float64.

    Channels are trimmed to their common minimum frame count first — at a
    mid-cycle recording tail sibling channels can differ by one frame, which
    would otherwise make ``np.mean`` of the ragged list raise.
    """
    stacks = [load_full_channel_stack(output_dir, i) for i in indices]  # (H,W,Ti)
    t = min(s.shape[2] for s in stacks)
    return np.mean([s[..., :t].astype(np.float64) for s in stacks], axis=0)


def full_channel_mean(output_dir, ch_idx: int) -> np.ndarray:
    """Full-recording mean image of channel ``ch_idx`` as ``(H, W)`` float64.

    Uses ``reg_meta['meanImageCh{i}']`` when present, else computes it from the
    ``reg_Ch{i}.npy`` timeline.
    """
    meta_path = reg_meta_path(output_dir)
    if meta_path.exists():
        with np.load(meta_path, allow_pickle=True) as z:
            key = f"meanImageCh{ch_idx}"
            if key in z.files:
                return np.asarray(z[key], dtype=np.float64)
    return load_reg_channel(output_dir, ch_idx).mean(axis=0).astype(np.float64)


# --------------------------------------------------------------------------
# suite2p-style per-channel memmap layout (use_mmap path)
# --------------------------------------------------------------------------
# reg_Ch{i}.npy   (T, H, W) uint16   registered+binned channel, memory-mappable
# dff_{name}.npy  (T, H, W) float32  linear-subtraction dF/F per channels_name group
# reg_meta.npz    proc_template, meanImageCh{i}, channels_name/prop, imageSize,
#                 per-channel frame count (T_per_ch), fps, source/donner indices.
#
# Arrays are (T, H, W) so per-frame writes/reads are contiguous (streaming).
# ``reg_meta['T_per_ch']`` is the authoritative frame count per channel: files
# may be preallocated slightly longer (only on cancellation / early stop), so
# readers slice ``[:T]``.  A complete run allocates the exact count (no waste).


def reg_channel_path(output_dir, ch_idx: int) -> Path:
    return Path(output_dir) / f"reg_Ch{ch_idx}.npy"


def dff_name_path(output_dir, name: str) -> Path:
    return Path(output_dir) / f"dff_{name}.npy"


def reg_meta_path(output_dir) -> Path:
    return Path(output_dir) / "reg_meta.npz"


def open_reg_memmap(path, n_frames: int, hw: tuple[int, int],
                    dtype=np.uint16) -> np.ndarray:
    """Create a preallocated ``(n_frames, H, W)`` .npy backed by a writable
    memmap (``open_memmap`` w+); fill it with ``arr[t0:t1] = chunk``."""
    from numpy.lib.format import open_memmap

    h, w = int(hw[0]), int(hw[1])
    return open_memmap(
        Path(path), mode="w+", dtype=dtype, shape=(int(n_frames), h, w)
    )


def write_reg_meta(output_dir, meta: dict[str, Any]) -> None:
    """Write the reg/dff metadata sidecar (``reg_meta.npz``)."""
    np.savez(reg_meta_path(output_dir), **meta)


def read_reg_meta(output_dir) -> dict[str, Any]:
    """Read ``reg_meta.npz`` back into a plain dict (arrays materialized)."""
    with np.load(reg_meta_path(output_dir), allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _reg_ch_frame_count(output_dir, ch_idx: int) -> int | None:
    """Authoritative frame count for channel ``ch_idx`` from ``reg_meta``.

    Returns None when the sidecar is absent or does not record per-channel T.
    """
    meta_path = reg_meta_path(output_dir)
    if not meta_path.exists():
        return None
    with np.load(meta_path, allow_pickle=True) as z:
        if "T_per_ch" not in z.files:
            return None
        t_per_ch = np.asarray(z["T_per_ch"]).ravel()
    if ch_idx < 0 or ch_idx >= t_per_ch.size:
        return None
    return int(t_per_ch[ch_idx])


def load_reg_channel(output_dir, ch_idx: int, *, mmap: bool = False) -> np.ndarray:
    """Load ``reg_Ch{ch_idx}.npy`` as ``(T, H, W)``, trimmed to the meta count.

    ``mmap=True`` returns a read-only memmap (chunk it to stay memory-bounded);
    ``mmap=False`` returns the array in RAM.
    """
    path = reg_channel_path(output_dir, ch_idx)
    if not path.exists():
        raise FileNotFoundError(f"No reg_Ch{ch_idx}.npy under {output_dir}")
    arr = np.load(path, mmap_mode="r" if mmap else None)
    t = _reg_ch_frame_count(output_dir, ch_idx)
    return arr[:t] if t is not None and t < arr.shape[0] else arr


def load_dff(output_dir, name: str, *, mmap: bool = False) -> np.ndarray:
    """Load ``dff_{name}.npy`` as ``(T, H, W)`` float32/64."""
    path = dff_name_path(output_dir, name)
    if not path.exists():
        raise FileNotFoundError(f"No dff_{name}.npy under {output_dir}")
    return np.load(path, mmap_mode="r" if mmap else None)


def save_stack_tiff(
    path: Path,
    data: np.ndarray,
    *,
    tiff_format: str = "big-tiff",
    compression: bool = False,
) -> list[Path]:
    """Save (H, W, T) uint16 array as TIFF stack.

    Parameters
    ----------
    path : output file path (.tif / .ome.tif).
    data : (H, W, T) uint16 array.
    tiff_format : "big-tiff" or "ome-tiff".
    compression : apply LZW compression.

    Returns
    -------
    List of saved file paths.
    """
    frames = np.ascontiguousarray(np.moveaxis(data, 2, 0))  # (T, H, W)
    compress = "zlib" if compression else None

    if tiff_format == "ome-tiff":
        ome_path = path.with_suffix(".ome.tif")
        tifffile.imwrite(
            str(ome_path),
            frames,
            bigtiff=True,
            ome=True,
            compression=compress,
            metadata={"axes": "TYX"},
        )
        return [ome_path]

    # big-tiff (default)
    tifffile.imwrite(
        str(path),
        frames,
        bigtiff=True,
        compression=compress,
    )
    return [path]


def save_grayscale_png(path: Path, image: np.ndarray) -> None:
    """Save a 2D array as a grayscale PNG (normalized to 0-255)."""
    arr = np.asarray(image, dtype=np.float64)
    mn, mx = np.min(arr), np.max(arr)
    if mx <= mn:
        norm = np.zeros_like(arr, dtype=np.uint8)
    else:
        norm = ((arr - mn) / (mx - mn) * 255.0).astype(np.uint8)
    iio.imwrite(path, norm)


def build_fig_frames(
    channel_images: list[list[np.ndarray]],
) -> np.ndarray:
    """Compose a preview montage from per-batch, per-channel mean images.

    Parameters
    ----------
    channel_images : list of batches, each batch is a list of N channel mean images.

    Returns
    -------
    Montage array: N rows (one per channel) x B columns (one per batch).
    """
    n_batches = len(channel_images)
    n_channels = len(channel_images[0])
    rows = []
    for ch in range(n_channels):
        row = np.hstack([channel_images[b][ch] for b in range(n_batches)])
        rows.append(row)
    return np.vstack(rows)
