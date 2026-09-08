"""Reader for Ito-lab even/odd HDF5 WFCI recordings.

One *recording* is a folder of ``*.h5`` parts written by the Blackfly
acquisition (``format: hdf5_blosc2_even_odd_batches``): the interleaved 2-channel
stream is split into ``*_even_jobNNNN_*.h5`` / ``*_odd_jobNNNN_*.h5`` parts, each
holding a ``frames`` (T, H, W) dataset plus uncompressed ``source_indices`` /
``camera_timestamp_ns`` / ``host_time_ns``.

:class:`H5Recording` merges the parts back into a single stream ordered by
``source_index`` — the original interleaved order — so the pipeline's positional
demux (``frame_index % cycle_len``) assigns even/odd back to their channels with
no downstream change.  It mirrors :class:`~asvimg.sifx.SIFXFile` /
``DcimgFile``: ``nfrms`` / ``frame`` / ``frame_with_metadata`` / ``frame_metadata``
/ ``metadata_summary``, usable as a context manager.

**Codec-agnostic on purpose.**  The frame data may use *any* HDF5 compression
filter (this batch is blosc2/zstd/bitshuffle, but gzip/zstd/lz4/… are equally
valid).  We never decode a codec by hand: importing ``hdf5plugin`` registers the
third-party filters (gzip/shuffle are built into libhdf5) and h5py decodes
whatever filter the file records transparently.  The only codec-aware code here
inspects the dataset's filter pipeline to raise a *legible* error when a required
plugin is missing, instead of h5py's cryptic "can't open directory".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np

try:  # registers blosc2 / zstd / lz4 / bitshuffle / … HDF5 filters with libhdf5
    import hdf5plugin  # noqa: F401

    _HAS_HDF5PLUGIN = True
except Exception:  # pragma: no cover - only when the optional dep is absent
    _HAS_HDF5PLUGIN = False

# Cache big enough to hold several chunks so sequential-within-a-part reads hit
# the chunk cache instead of re-decoding the (16, H, W) chunk for every frame.
_RDCC_NBYTES = 64 * 1024 * 1024
_RDCC_NSLOTS = 1009  # prime, per the h5py chunk-cache guidance


def describe_filters(dataset: h5py.Dataset) -> list[tuple[int, str]]:
    """Return ``[(filter_id, name), …]`` for a dataset's compression pipeline."""
    plist = dataset.id.get_create_plist()
    out: list[tuple[int, str]] = []
    for i in range(plist.get_nfilters()):
        info = plist.get_filter(i)
        fid = int(info[0])
        name = info[3]
        out.append((fid, name.decode() if isinstance(name, bytes) else str(name)))
    return out


def _source_indices(h5: h5py.File) -> np.ndarray:
    """Global (original-stream) frame index for each stored frame of a part.

    Prefers the explicit ``source_indices`` dataset; falls back to
    ``source_start_frame + source_frame_step * arange`` from the part's attrs.
    """
    if "source_indices" in h5:
        return np.asarray(h5["source_indices"], dtype=np.int64)
    n = h5["frames"].shape[0]
    start = int(h5.attrs.get("source_start_frame", 0))
    step = int(h5.attrs.get("source_frame_step", 1))
    return start + step * np.arange(n, dtype=np.int64)


def _collect_parts(folder: Path) -> list[Path]:
    """``*.h5`` files under ``folder`` that carry a ``frames`` dataset, sorted."""
    parts: list[Path] = []
    for path in sorted(folder.glob("*.h5")):
        try:
            with h5py.File(path, "r") as h5:
                if "frames" in h5:
                    parts.append(path)
        except OSError:
            continue
    return parts


def is_ito_h5_recording(folder) -> bool:
    """True when ``folder`` is a directory holding ≥1 ``*.h5`` part with frames."""
    folder = Path(folder)
    if not folder.is_dir():
        return False
    return bool(_collect_parts(folder))


class H5Recording:
    """A folder of even/odd ``*.h5`` parts as one source-index-ordered stream."""

    def __init__(self, folder) -> None:
        self.folder = Path(folder)
        parts = _collect_parts(self.folder)
        if not parts:
            raise FileNotFoundError(
                f"No HDF5 parts with a 'frames' dataset found in {self.folder}"
            )
        self._parts = parts

        # Build the merged (source_index -> (part, local_index)) map. Only the
        # uncompressed source_indices are read here (cheap; no frame decode).
        src_chunks, part_chunks, local_chunks = [], [], []
        for part_idx, path in enumerate(parts):
            with h5py.File(path, "r") as h5:
                si = _source_indices(h5)
                n = int(si.shape[0])
                if part_idx == 0:
                    frames = h5["frames"]
                    self._hw = (int(frames.shape[1]), int(frames.shape[2]))
                    self._dtype = np.dtype(frames.dtype)
                    self._filters = describe_filters(frames)
            src_chunks.append(si)
            part_chunks.append(np.full(n, part_idx, dtype=np.int64))
            local_chunks.append(np.arange(n, dtype=np.int64))

        src = np.concatenate(src_chunks)
        # Stable sort by source_index reconstructs the original interleaved order
        # (even/odd merge). Stable keeps a deterministic tie-break if two parts
        # ever share an index (they don't in practice).
        order = np.argsort(src, kind="stable")
        self._src = src[order]
        self._part = np.concatenate(part_chunks)[order]
        self._local = np.concatenate(local_chunks)[order]

        self._open: dict[int, h5py.File] = {}

    # --- lifecycle --------------------------------------------------------

    def __enter__(self) -> "H5Recording":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        for h in self._open.values():
            try:
                h.close()
            except Exception:
                pass
        self._open.clear()

    # --- geometry ---------------------------------------------------------

    @property
    def nfrms(self) -> int:
        return int(self._src.shape[0])

    @property
    def shape_hw(self) -> tuple[int, int]:
        return self._hw

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    # --- frame access -----------------------------------------------------

    def _handle(self, part_idx: int) -> h5py.File:
        h = self._open.get(part_idx)
        if h is None:
            h = h5py.File(
                self._parts[part_idx], "r",
                rdcc_nbytes=_RDCC_NBYTES, rdcc_nslots=_RDCC_NSLOTS,
            )
            self._open[part_idx] = h
        return h

    def _read_frame(self, idx: int) -> np.ndarray:
        part_idx = int(self._part[idx])
        local = int(self._local[idx])
        dataset = self._handle(part_idx)["frames"]
        try:
            return np.asarray(dataset[local])
        except OSError as e:  # most likely a missing compression-filter plugin
            filters = ", ".join(f"{fid}:{name}" for fid, name in self._filters)
            hint = (
                "" if _HAS_HDF5PLUGIN
                else " — install 'hdf5plugin' to register blosc2/zstd/lz4/… filters"
            )
            raise OSError(
                f"Failed to read compressed HDF5 frame {idx} from "
                f"{self._parts[part_idx].name} (filters=[{filters}]){hint}: {e}"
            ) from e

    def frame(self, index: int, copy: bool = True) -> np.ndarray:
        # h5py hands back a freshly-decoded array, so it is always a copy; the
        # ``copy`` flag exists only for API parity with the sifx/dcimg readers.
        return self._read_frame(int(index))

    def frame_with_metadata(
        self, index: int, copy: bool = True
    ) -> tuple[np.ndarray, dict[str, Any]]:
        index = int(index)
        frame = self._read_frame(index)
        part_idx = int(self._part[index])
        local = int(self._local[index])
        source_index = int(self._src[index])

        cam_ns = host_ns = None
        h5 = self._handle(part_idx)
        if "camera_timestamp_ns" in h5:
            cam_ns = int(np.asarray(h5["camera_timestamp_ns"][local]))
        if "host_time_ns" in h5:
            host_ns = int(np.asarray(h5["host_time_ns"][local]))

        meta: dict[str, Any] = {
            "frame_index": index,
            "timestamp_sec": None if cam_ns is None else cam_ns / 1e9,
            "timestamp_microsec": None if cam_ns is None else cam_ns / 1e3,
            "framestamp": source_index,   # original-stream (interleaved) index
            "camerastamp": cam_ns,
            "image_height": int(frame.shape[0]),
            "image_width": int(frame.shape[1]),
            "dtype": str(frame.dtype),
            "bits_per_pixel": int(frame.dtype.itemsize * 8),
            # h5-specific extras (harmless to downstream, useful for QC)
            "source_index": source_index,
            "camera_timestamp_ns": cam_ns,
            "host_time_ns": host_ns,
        }
        return frame, meta

    def frame_metadata(self, index: int) -> dict[str, Any]:
        _, meta = self.frame_with_metadata(index, copy=False)
        return meta

    def metadata_summary(self) -> dict[str, Any]:
        return {
            "backend": "h5_ito",
            "nfrms": self.nfrms,
            "height": int(self._hw[0]),
            "width": int(self._hw[1]),
            "dtype": str(self._dtype),
            "bits_per_pixel": int(self._dtype.itemsize * 8),
            "n_parts": len(self._parts),
            "source_start": int(self._src[0]),
            "source_end": int(self._src[-1]),
            "compression": [f"{fid}:{name}" for fid, name in self._filters],
        }

    # --- convenience ------------------------------------------------------

    def iter_frames(self) -> Iterator[np.ndarray]:
        for idx in range(self.nfrms):
            yield self._read_frame(idx)
