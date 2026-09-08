"""SIFX (Andor spool) reader.

Reads frames from Andor spool directories.  The entry point is a ``.sifx``
file; the parent directory must also contain ``acquisitionmetadata.ini``
and one or more ``*spool.dat`` files.
"""

from __future__ import annotations

import configparser
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# Pixel encoding -> numpy dtype mapping
_PIXEL_DTYPES: dict[str, np.dtype] = {
    "Mono8": np.dtype(np.uint8),
    "Mono16": np.dtype(np.uint16),
    "Mono32": np.dtype(np.uint32),
}


def _ordered_dat_files(path: Path) -> int:
    """Sort key for Andor spool .dat files.

    Andor writes the sequence number with its digits *reversed* in the
    filename (e.g. sequence 10 -> ``0100000000spool.dat``).  Reversing
    the digit string back recovers the true acquisition order.
    """
    numeric = "".join(filter(str.isdigit, path.name))
    return int(numeric[::-1])


class SIFXFile:
    """SIFX spool reader facade.

    Parameters
    ----------
    sifx_path : str | Path
        Path to the ``.sifx`` file.  The parent directory must contain
        ``acquisitionmetadata.ini`` and ``*spool.dat`` files.
    """

    def __init__(self, sifx_path: str | Path) -> None:
        self.sifx_path = Path(sifx_path)
        self.spool_dir = self.sifx_path.parent

        # Parse acquisitionmetadata.ini
        ini_path = self.spool_dir / "acquisitionmetadata.ini"
        if not ini_path.exists():
            raise FileNotFoundError(
                f"acquisitionmetadata.ini not found in {self.spool_dir}"
            )

        cfg = configparser.ConfigParser()
        cfg.read(ini_path, encoding="utf-8-sig")

        self._width = int(cfg["data"]["AOIWidth"])
        self._height = int(cfg["data"]["AOIHeight"])
        self._stride = int(cfg["data"]["AOIStride"])
        self._pixel_encoding: str = cfg["data"]["PixelEncoding"]
        self._image_size_bytes = int(cfg["data"]["ImageSizeBytes"])
        self._images_per_file = int(cfg["multiimage"]["ImagesPerFile"])

        if self._pixel_encoding not in _PIXEL_DTYPES:
            raise ValueError(
                f"Unsupported pixel encoding: {self._pixel_encoding}. "
                f"Supported: {sorted(_PIXEL_DTYPES)}"
            )

        self._dtype = _PIXEL_DTYPES[self._pixel_encoding]
        self._bpp = self._dtype.itemsize
        self._pixels_per_row = self._stride // self._bpp

        # Discover and sort dat files
        self._dat_files = sorted(
            self.spool_dir.glob("*spool.dat"), key=_ordered_dat_files
        )
        if not self._dat_files:
            raise FileNotFoundError(
                f"No *spool.dat files found in {self.spool_dir}"
            )

        # Count the frames actually ON DISK, not files x ImagesPerFile: an
        # acquisition stopped mid-spool leaves a short last file, and trusting the
        # nominal count made preprocess read past the end of it and die at ~99%.
        self._total_frames = sum(
            int(p.stat().st_size) // self._image_size_bytes for p in self._dat_files
        )

        # Optionally parse .sifx header via sif_parser for rich metadata
        self._sifx_info: dict[str, Any] | None = None
        self._header_nfrms: int | None = None
        try:
            import sif_parser

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, info = sif_parser.np_open(
                    str(self.sifx_path), ignore_corrupt=True
                )
            self._sifx_info = dict(info)
            nof = self._sifx_info.get("NumberOfFrames")
            if nof is not None:
                self._header_nfrms = int(nof)
        except Exception:
            pass

    def __enter__(self) -> SIFXFile:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Release resources (no-op for file-based access)."""

    # -- properties ----------------------------------------------------------

    @property
    def nfrms(self) -> int:
        """Total number of frames across all dat files."""
        return self._total_frames

    @property
    def header_nfrms(self) -> int | None:
        """Frame count from the .sifx header (via sif_parser), or None."""
        return self._header_nfrms

    @property
    def effective_nfrms(self) -> int:
        """Frames actually acquired: excludes trailing spool padding.

        A fully written spool can hold more frames than were acquired -- the
        tail is zero padding.  The header count catches that; the on-disk
        count above catches the opposite failure (an acquisition stopped
        mid-spool leaves a short last file).  ``min`` is right in both
        directions, so the two guards are not redundant.
        """
        if self._header_nfrms is not None:
            return min(self._header_nfrms, self._total_frames)
        return self._total_frames

    @property
    def padding_frames(self) -> int:
        """Trailing padding frames (dat total - header count).

        0 when the header is unavailable or the counts already agree.
        """
        if self._header_nfrms is None:
            return 0
        return max(0, self._total_frames - self._header_nfrms)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    # -- frame access --------------------------------------------------------

    def frame(self, index: int, copy: bool = True) -> np.ndarray:
        """Read a single frame by flat index."""
        if index < 0 or index >= self._total_frames:
            raise IndexError(
                f"Frame index {index} out of range [0, {self._total_frames})"
            )

        file_idx = index // self._images_per_file
        local_idx = index % self._images_per_file
        offset = local_idx * self._image_size_bytes

        want = self._pixels_per_row * self._height
        raw = np.fromfile(
            self._dat_files[file_idx],
            dtype=self._dtype,
            count=want,
            offset=offset,
        )
        if raw.size < want:
            # A truncated spool: say which file, instead of a bare reshape error.
            raise IOError(
                f"Spool file {self._dat_files[file_idx].name} is truncated: frame "
                f"{index} needs {want} samples at offset {offset}, got {raw.size}"
            )
        frame = raw.reshape(self._height, self._pixels_per_row)[:, : self._width]
        return frame.copy() if copy else frame

    def frame_with_metadata(
        self, index: int, copy: bool = True
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Read frame and per-frame metadata."""
        frame = self.frame(index, copy=copy)
        meta: dict[str, Any] = {
            "frame_index": int(index),
            "timestamp_sec": None,
            "timestamp_microsec": None,
            "framestamp": None,
            "camerastamp": None,
            "image_height": int(frame.shape[0]),
            "image_width": int(frame.shape[1]),
            "dtype": str(frame.dtype),
            "bits_per_pixel": int(frame.dtype.itemsize * 8),
        }

        # Enrich with per-frame timestamp from sifx header if available
        if self._sifx_info is not None:
            ts = self._sifx_info.get(f"timestamp_of_{index}")
            if ts is not None:
                meta["timestamp_sec"] = float(ts)

        return frame, meta

    def frame_metadata(self, index: int) -> dict[str, Any]:
        """Return per-frame metadata without copying frame data."""
        _, meta = self.frame_with_metadata(index, copy=False)
        return meta

    def metadata_summary(self) -> dict[str, Any]:
        """Return summary metadata dict."""
        summary: dict[str, Any] = {
            "backend": "sifx",
            "nfrms": self._total_frames,
            "width": self._width,
            "height": self._height,
            "dtype": str(self._dtype),
            "pixel_encoding": self._pixel_encoding,
            "stride": self._stride,
            "image_size_bytes": self._image_size_bytes,
            "images_per_file": self._images_per_file,
            "dat_file_count": len(self._dat_files),
            "sifx_path": str(self.sifx_path),
        }

        # Merge rich metadata from sifx header if available
        if self._sifx_info is not None:
            for key in (
                "DetectorType",
                "ExposureTime",
                "CycleTime",
                "DetectorTemperature",
                "NumberOfFrames",
            ):
                if key in self._sifx_info:
                    summary[key] = self._sifx_info[key]

        return summary
