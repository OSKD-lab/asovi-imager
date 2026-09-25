"""Reader for Nikon NIS-Elements ``.nd2`` recordings.

An ND2 file stores one image per *sequence* (one acquisition step of the time
loop), and a multi-channel acquisition keeps **every channel inside that one
sequence**: ``ND2File.read_frame(seq)`` returns ``(C, Y, X)``.  The pipeline's
demux is positional instead (``frame_index % cycle_len``), so
:class:`Nd2Recording` flattens the ``T x C`` grid into one interleaved stream —
``t0c0, t0c1, …, t1c0, t1c1, …`` — exactly as an alternating-illumination
camera would have written it.  With ``channels_name`` listing the ND2 channels
in their stored order, frame ``t*C + c`` lands on channel ``c`` with no
downstream change.  It mirrors :class:`~asvimg.h5rec.H5Recording` /
:class:`~asvimg.sifx.SIFXFile`: ``nfrms`` / ``frame`` / ``frame_with_metadata`` /
``frame_metadata`` / ``metadata_summary``, usable as a context manager.

Only ``T`` / ``C`` / ``Y`` / ``X`` are accepted.  A Z-stack, multi-point (``P``)
or RGB (``S``) file raises instead of being flattened into the time axis, where
it would silently read as extra channels or frames.

Needs the ``nd2`` extra (``pip install "asovi-imager[nd2]"``); the import is
deferred to :class:`Nd2Recording`, so ``import asvimg`` does not require it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np

# Axes the flattening understands.  Everything else is refused (see module doc).
_ACCEPTED_AXES = frozenset({"T", "C", "Y", "X"})


def _open_nd2(path: Path):
    try:
        import nd2
    except ModuleNotFoundError as exc:  # optional extra
        raise ModuleNotFoundError(
            'reading .nd2 files needs the `nd2` extra: '
            'pip install "asovi-imager[nd2]"'
        ) from exc
    return nd2.ND2File(path)


class Nd2Recording:
    """One ``.nd2`` file as a channel-interleaved frame stream."""

    def __init__(self, path) -> None:
        self.path = Path(path)
        self._f = _open_nd2(self.path)
        try:
            sizes = dict(self._f.sizes)
            extra = sorted(set(sizes) - _ACCEPTED_AXES)
            if extra:
                raise ValueError(
                    f"{self.path.name}: ND2 axes {sizes} include {extra}; only "
                    f"T/C/Y/X recordings can be read (a Z-stack, multi-point or "
                    f"RGB file would be flattened into the time axis). Export a "
                    f"single-position, single-plane T x C series instead."
                )
            self._n_seq = int(sizes.get("T", 1))
            self._n_ch = int(sizes.get("C", 1))
            self._hw = (int(sizes["Y"]), int(sizes["X"]))
            self._dtype = np.dtype(self._f.dtype)
        except Exception:
            self._f.close()
            raise

        # read_frame(seq) hands back all C channels at once; consecutive stream
        # frames share a sequence, so keep the last one instead of re-reading it
        # C times.  Same for the per-sequence metadata (it is parsed per call).
        self._seq_cache: tuple[int, np.ndarray] | None = None
        self._meta_cache: tuple[int, Any] | None = None

    # --- lifecycle --------------------------------------------------------

    def __enter__(self) -> "Nd2Recording":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._seq_cache = None
        self._meta_cache = None
        self._f.close()

    # --- geometry ---------------------------------------------------------

    @property
    def nfrms(self) -> int:
        """Stream length = sequences x channels."""
        return self._n_seq * self._n_ch

    @property
    def n_channels(self) -> int:
        return self._n_ch

    @property
    def n_sequences(self) -> int:
        return self._n_seq

    @property
    def shape_hw(self) -> tuple[int, int]:
        return self._hw

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def channel_names(self) -> list[str]:
        """ND2 channel names in stored order (= stream order within a sequence)."""
        try:
            chans = self._f.metadata.channels or []
            names = [str(c.channel.name) for c in chans]
        except Exception:  # legacy files carry no structured metadata
            names = []
        if len(names) != self._n_ch:
            return [f"C{i}" for i in range(self._n_ch)]
        return names

    # --- frame access -----------------------------------------------------

    def _split(self, index: int) -> tuple[int, int]:
        if not 0 <= index < self.nfrms:
            raise IndexError(f"frame {index} out of range (nfrms={self.nfrms})")
        return divmod(index, self._n_ch)

    def _sequence(self, seq: int) -> np.ndarray:
        if self._seq_cache is None or self._seq_cache[0] != seq:
            self._seq_cache = (seq, self._f.read_frame(seq))
        return self._seq_cache[1]

    def _read_frame(self, index: int) -> np.ndarray:
        seq, ch = self._split(int(index))
        arr = self._sequence(seq)
        plane = arr[ch] if self._n_ch > 1 else arr
        # read_frame is a read-only view into the file's memmap, and a channel
        # plane of it is strided (channels are interleaved per pixel).  Always
        # copy: ascontiguousarray would return a single-channel frame as that
        # same view, which crashes (access violation) once the file is closed.
        return np.array(plane, order="C", copy=True)

    def frame(self, index: int, copy: bool = True) -> np.ndarray:
        # always a copy (see _read_frame); ``copy`` is for API parity only
        return self._read_frame(int(index))

    def _seq_time_sec(self, seq: int, ch: int) -> float | None:
        if self._meta_cache is None or self._meta_cache[0] != seq:
            try:
                meta = self._f.frame_metadata(seq)
            except Exception:
                meta = None
            self._meta_cache = (seq, meta)
        meta = self._meta_cache[1]
        try:  # legacy files give a dict without per-channel timestamps
            return float(meta.channels[ch].time.relativeTimeMs) / 1e3
        except Exception:
            return None

    def frame_with_metadata(
        self, index: int, copy: bool = True
    ) -> tuple[np.ndarray, dict[str, Any]]:
        index = int(index)
        frame = self._read_frame(index)
        seq, ch = self._split(index)
        t_sec = self._seq_time_sec(seq, ch)
        meta: dict[str, Any] = {
            "frame_index": index,
            "timestamp_sec": t_sec,
            "timestamp_microsec": None if t_sec is None else t_sec * 1e6,
            "framestamp": seq,
            "camerastamp": None,
            "image_height": int(frame.shape[0]),
            "image_width": int(frame.shape[1]),
            "dtype": str(frame.dtype),
            "bits_per_pixel": int(frame.dtype.itemsize * 8),
            # nd2-specific extras (harmless to downstream, useful for QC)
            "nd2_sequence": seq,
            "nd2_channel": ch,
        }
        return frame, meta

    def frame_metadata(self, index: int) -> dict[str, Any]:
        _, meta = self.frame_with_metadata(index, copy=False)
        return meta

    def metadata_summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "backend": "nd2",
            "nfrms": self.nfrms,
            "height": int(self._hw[0]),
            "width": int(self._hw[1]),
            "dtype": str(self._dtype),
            "bits_per_pixel": int(self._dtype.itemsize * 8),
            "n_sequences": self._n_seq,
            "n_channels": self._n_ch,
            "channel_names": self.channel_names,
            "sizes": {k: int(v) for k, v in dict(self._f.sizes).items()},
        }
        try:
            attrs = self._f.attributes
            out["bits_significant"] = int(attrs.bitsPerComponentSignificant)
        except Exception:
            pass
        try:
            vs = self._f.voxel_size()
            out["pixel_size_um"] = [float(vs.x), float(vs.y)]
        except Exception:
            pass
        return out

    # --- convenience ------------------------------------------------------

    def iter_frames(self) -> Iterator[np.ndarray]:
        for idx in range(self.nfrms):
            yield self._read_frame(idx)


def nd2_channel_count(path) -> int:
    """Channels per sequence of an ``.nd2`` file (the ``C`` size, 1 if absent)."""
    with Nd2Recording(path) as rec:
        return rec.n_channels
