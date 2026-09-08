"""DCIMG reader with SDK-first backend selection.

Default behavior uses the official DCIMG API when available.
If the SDK runtime is unavailable (for example on macOS/Linux environments
without installed dcimgapi runtime), it falls back to the pure Python parser.
"""

from __future__ import annotations

import ctypes
import platform
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# SDK backend (official API via ctypes)
# ---------------------------------------------------------------------------


class _SdkError(RuntimeError):
    pass


class _DCIMG_IDPARAML:
    NUMBEROF_FRAME = 2
    IMAGE_WIDTH = 9
    IMAGE_HEIGHT = 10
    IMAGE_ROWBYTES = 11
    IMAGE_PIXELTYPE = 12
    FILEFORMAT_VERSION = 21


class _DCIMG_PIXELTYPE:
    MONO8 = 1
    MONO16 = 2
    MONO32 = 4


class _DCIMG_CODEPAGE:
    UTF8 = 65001


class _DCIMG_ERR:
    FILENOTOPENED = -2147481547


class _DCIMG_INIT(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("size", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
        ("guid", ctypes.c_void_p),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.size = ctypes.sizeof(type(self))


class _DCIMG_OPEN(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("size", ctypes.c_int32),
        ("codepage", ctypes.c_int32),
        ("hdcimg", ctypes.c_void_p),
        ("path", ctypes.c_char_p),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.size = ctypes.sizeof(type(self))


class _DCIMG_TIMESTAMP(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("sec", ctypes.c_uint32),
        ("microsec", ctypes.c_int32),
    ]


class _DCIMG_FRAME(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("size", ctypes.c_int32),
        ("iKind", ctypes.c_int32),
        ("option", ctypes.c_int32),
        ("iFrame", ctypes.c_int32),
        ("buf", ctypes.c_void_p),
        ("rowbytes", ctypes.c_int32),
        ("type", ctypes.c_int32),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("left", ctypes.c_int32),
        ("top", ctypes.c_int32),
        ("timestamp", _DCIMG_TIMESTAMP),
        ("framestamp", ctypes.c_int32),
        ("camerastamp", ctypes.c_int32),
        ("conversionfactor_coeff", ctypes.c_double),
        ("conversionfactor_offset", ctypes.c_double),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.size = ctypes.sizeof(type(self))
        self.iKind = 0
        self.option = 0


class _SdkDcimgBackend:
    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self._dll: Any = self._load_dll()
        self._bind_functions()
        self._init_runtime()

        self._hdcimg = ctypes.c_void_p(None)
        self.nfrms = 0
        self.xsize = 0
        self.ysize = 0
        self.rowbytes = 0
        self.pixel_type = 0
        self.file_format_version = 0
        self.dtype = np.uint16

        self.open()

    @staticmethod
    def _load_dll() -> Any:
        system = platform.system().lower()
        candidates: list[str] = []

        if system == "windows":
            candidates = ["dcimgapi.dll"]
            loader = ctypes.WinDLL
        elif system == "linux":
            candidates = ["libdcimgapi.so", "/usr/local/lib/libdcimgapi.so"]
            loader = ctypes.CDLL
        elif system == "darwin":
            candidates = ["libdcimgapi.dylib"]
            loader = ctypes.CDLL
        else:
            raise _SdkError(f"Unsupported OS for SDK backend: {system}")

        last_err: Exception | None = None
        for name in candidates:
            try:
                return loader(name)
            except Exception as exc:  # pragma: no cover
                last_err = exc

        raise _SdkError(f"Failed to load DCIMG SDK runtime: {last_err}")

    def _bind_functions(self) -> None:
        self._dcimg_init = self._dll.dcimg_init
        self._dcimg_init.argtypes = [ctypes.POINTER(_DCIMG_INIT)]

        self._dcimg_open = self._dll.dcimg_openA
        self._dcimg_open.argtypes = [ctypes.POINTER(_DCIMG_OPEN)]

        self._dcimg_getparaml = self._dll.dcimg_getparaml
        self._dcimg_getparaml.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_int32),
        ]

        self._dcimg_copyframe = self._dll.dcimg_copyframe
        self._dcimg_copyframe.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DCIMG_FRAME)]

        self._dcimg_close = self._dll.dcimg_close
        self._dcimg_close.argtypes = [ctypes.c_void_p]

    @staticmethod
    def _is_failed(err: int) -> bool:
        return int(err) < 0

    def _init_runtime(self) -> None:
        init_param = _DCIMG_INIT()
        err = int(self._dcimg_init(ctypes.byref(init_param)))
        if self._is_failed(err):
            raise _SdkError(f"dcimg_init failed: {err}")

    def _get_param(self, param_id: int) -> int:
        if not self._hdcimg:
            raise _SdkError("dcimg handle is not open")
        v = ctypes.c_int32(0)
        err = int(self._dcimg_getparaml(self._hdcimg, param_id, ctypes.byref(v)))
        if self._is_failed(err):
            raise _SdkError(f"dcimg_getparaml failed id={param_id}: {err}")
        return int(v.value)

    def open(self) -> None:
        self.close()

        op = _DCIMG_OPEN()
        op.codepage = _DCIMG_CODEPAGE.UTF8
        op.path = str(self.file_path).encode("utf-8")

        err = int(self._dcimg_open(ctypes.byref(op)))
        if self._is_failed(err):
            raise _SdkError(f"dcimg_open failed: {err}")

        self._hdcimg = op.hdcimg
        self.nfrms = self._get_param(_DCIMG_IDPARAML.NUMBEROF_FRAME)
        self.xsize = self._get_param(_DCIMG_IDPARAML.IMAGE_WIDTH)
        self.ysize = self._get_param(_DCIMG_IDPARAML.IMAGE_HEIGHT)
        self.rowbytes = self._get_param(_DCIMG_IDPARAML.IMAGE_ROWBYTES)
        self.pixel_type = self._get_param(_DCIMG_IDPARAML.IMAGE_PIXELTYPE)
        self.file_format_version = self._get_param(_DCIMG_IDPARAML.FILEFORMAT_VERSION)

        if self.pixel_type == _DCIMG_PIXELTYPE.MONO8:
            self.dtype = np.uint8
        elif self.pixel_type == _DCIMG_PIXELTYPE.MONO16:
            self.dtype = np.uint16
        elif self.pixel_type == _DCIMG_PIXELTYPE.MONO32:
            self.dtype = np.int32
        else:
            raise _SdkError(f"Unsupported pixel type from SDK: {self.pixel_type}")

    def close(self) -> None:
        if self._hdcimg:
            try:
                self._dcimg_close(self._hdcimg)
            except Exception:
                pass
        self._hdcimg = ctypes.c_void_p(None)

    def frame(self, index: int, copy: bool = True) -> np.ndarray:
        frame, _ = self.frame_with_metadata(index, copy=copy)
        return frame

    def frame_with_metadata(
        self, index: int, copy: bool = True
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if index < 0 or index >= self.nfrms:
            raise IndexError(f"Frame index out of range: {index}")

        arr = np.zeros((self.ysize, self.xsize), dtype=self.dtype)

        fr = _DCIMG_FRAME()
        fr.iFrame = int(index)
        fr.buf = arr.ctypes.data_as(ctypes.c_void_p)
        fr.width = self.xsize
        fr.height = self.ysize
        fr.rowbytes = self.rowbytes
        fr.type = self.pixel_type

        err = int(self._dcimg_copyframe(self._hdcimg, ctypes.byref(fr)))
        if self._is_failed(err):
            raise _SdkError(f"dcimg_copyframe failed index={index}: {err}")

        meta = {
            "frame_index": int(index),
            "timestamp_sec": int(fr.timestamp.sec),
            "timestamp_microsec": int(fr.timestamp.microsec),
            "framestamp": int(fr.framestamp),
            "camerastamp": int(fr.camerastamp),
            "conversionfactor_coeff": float(fr.conversionfactor_coeff),
            "conversionfactor_offset": float(fr.conversionfactor_offset),
        }

        if copy:
            arr = arr.copy()
        return arr, meta

    def frame_metadata(self, index: int) -> dict[str, Any]:
        _, meta = self.frame_with_metadata(index, copy=False)
        return meta

    def metadata_summary(self) -> dict[str, Any]:
        return {
            "backend": "sdk",
            "nfrms": int(self.nfrms),
            "width": int(self.xsize),
            "height": int(self.ysize),
            "dtype": str(np.dtype(self.dtype)),
            "pixel_type": int(self.pixel_type),
            "rowbytes": int(self.rowbytes),
            "file_format_version": int(self.file_format_version),
        }


# ---------------------------------------------------------------------------
# Native backend (pure Python parser)
# ---------------------------------------------------------------------------


class _NativeDcimgBackend:
    FILE_HDR_DTYPE = [
        ("file_format", "S8"),
        ("format_version", "<u4"),
        ("skip", "5<u4"),
        ("nsess", "<u4"),
        ("nfrms", "<u4"),
        ("header_size", "<u4"),
        ("skip2", "<u4"),
        ("file_size", "<u8"),
        ("skip3", "2<u4"),
        ("file_size2", "<u8"),
    ]

    SESS_HDR_DTYPE_OLD = [
        ("session_size", "<u8"),
        ("skip1", "6<u4"),
        ("nfrms", "<u4"),
        ("byte_depth", "<u4"),
        ("skip2", "<u4"),
        ("xsize", "<u4"),
        ("bytes_per_row", "<u4"),
        ("ysize", "<u4"),
        ("bytes_per_img", "<u4"),
        ("skip3", "2<u4"),
        ("offset_to_data", "<u4"),
        ("session_data_size", "<u8"),
    ]

    SESS_HDR_DTYPE_NEW = [
        ("session_size", "<u8"),
        ("skip1", "13<u4"),
        ("nfrms", "<u4"),
        ("byte_depth", "<u4"),
        ("skip2", "<u4"),
        ("xsize", "<u4"),
        ("ysize", "<u4"),
        ("bytes_per_row", "<u4"),
        ("bytes_per_img", "<u4"),
        ("skip3", "2<u4"),
        ("offset_to_data", "<u8"),
        ("skip4", "5<u4"),
        ("frame_footer_size", "<u4"),
    ]

    FMT_OLD = 1
    FMT_NEW = 2

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self._mm: np.memmap | None = None
        self._data: np.ndarray | None = None
        self._fmt = self.FMT_NEW
        self._file_format_version = 0

        self.nfrms = 0
        self.xsize = 0
        self.ysize = 0
        self.byte_depth = 0
        self.bytes_per_row = 0
        self.bytes_per_img = 0
        self.dtype = np.uint16

        self._framestamps: np.ndarray | None = None
        self._timestamps_raw: np.ndarray | None = None

        self.open()

    def open(self) -> None:
        self.close()
        self._mm = np.memmap(self.file_path, mode="r")

        file_header = np.ndarray((1,), dtype=self.FILE_HDR_DTYPE, buffer=self._mm)
        if file_header["file_format"][0] != b"DCIMG":
            raise ValueError(f"Invalid DCIMG file: {self.file_path}")

        fmt_version = int(file_header["format_version"][0])
        self._file_format_version = fmt_version
        header_size = int(file_header["header_size"][0])

        if fmt_version == 0x7:
            self._fmt = self.FMT_OLD
            sess_dtype = self.SESS_HDR_DTYPE_OLD
        elif fmt_version in (0x1000000, 0x2000000):
            self._fmt = self.FMT_NEW
            sess_dtype = self.SESS_HDR_DTYPE_NEW
        else:
            raise ValueError(f"Unsupported DCIMG format version: {fmt_version:#x}")

        sess_header = np.ndarray(
            (1,), dtype=sess_dtype, buffer=self._mm, offset=header_size
        )

        self.nfrms = int(sess_header["nfrms"][0])
        self.byte_depth = int(sess_header["byte_depth"][0])
        self.xsize = int(sess_header["xsize"][0])
        self.ysize = int(sess_header["ysize"][0])
        self.bytes_per_row = int(sess_header["bytes_per_row"][0])
        self.bytes_per_img = int(sess_header["bytes_per_img"][0])

        if self.byte_depth == 1:
            self.dtype = np.uint8
        elif self.byte_depth == 2:
            self.dtype = np.uint16
        else:
            raise ValueError(f"Unsupported byte depth: {self.byte_depth}")

        if self.bytes_per_img != self.bytes_per_row * self.ysize:
            raise ValueError("Invalid DCIMG: bytes_per_img mismatch")

        data_offset = header_size + int(sess_header["offset_to_data"][0])

        if self._fmt == self.FMT_OLD:
            strides = (self.bytes_per_img, self.bytes_per_row, self.byte_depth)
        else:
            frame_footer_size = int(sess_header["frame_footer_size"][0])
            padding = self.bytes_per_img - self.xsize * self.ysize * self.byte_depth
            padding //= self.ysize
            strides = (
                self.bytes_per_img + frame_footer_size,
                self.xsize * self.byte_depth + padding,
                self.byte_depth,
            )

        self._data = np.ndarray(
            (self.nfrms, self.ysize, self.xsize),
            dtype=self.dtype,
            buffer=self._mm,
            offset=data_offset,
            strides=strides,
        )

        self._bind_native_metadata(file_header, sess_header, data_offset)

    def _bind_native_metadata(
        self,
        file_header: np.ndarray,
        sess_header: np.ndarray,
        data_offset: int,
    ) -> None:
        self._framestamps = None
        self._timestamps_raw = None

        if self._mm is None:
            return

        if self._fmt == self.FMT_OLD:
            session_footer_offset = int(file_header["header_size"][0]) + int(
                sess_header["session_data_size"][0]
            )
            fs_offset = session_footer_offset + 272
            ts_offset = fs_offset + 4 * self.nfrms

            self._framestamps = np.ndarray(
                (self.nfrms,), dtype=np.uint32, buffer=self._mm, offset=fs_offset
            )
            self._timestamps_raw = np.ndarray(
                (self.nfrms, 2), dtype=np.uint32, buffer=self._mm, offset=ts_offset
            )
            return

        frame_footer_size = int(sess_header["frame_footer_size"][0])
        if frame_footer_size < 12:
            return

        stride = self.bytes_per_img + frame_footer_size
        fs_offset = data_offset + self.bytes_per_img
        ts_offset = fs_offset + 4

        self._framestamps = np.ndarray(
            (self.nfrms,),
            dtype=np.uint32,
            buffer=self._mm,
            offset=fs_offset,
            strides=(stride,),
        )
        self._timestamps_raw = np.ndarray(
            (self.nfrms, 2),
            dtype=np.uint32,
            buffer=self._mm,
            offset=ts_offset,
            strides=(stride, 4),
        )

    def close(self) -> None:
        self._data = None
        self._framestamps = None
        self._timestamps_raw = None
        self._mm = None

    def frame(self, index: int, copy: bool = True) -> np.ndarray:
        if self._data is None:
            raise RuntimeError("DCIMG file is closed")
        arr = self._data[index]
        return np.array(arr, copy=True) if copy else arr

    def frame_with_metadata(
        self, index: int, copy: bool = True
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return self.frame(index, copy=copy), self.frame_metadata(index)

    def frame_metadata(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= self.nfrms:
            raise IndexError(f"Frame index out of range: {index}")

        timestamp_sec: int | None = None
        timestamp_microsec: int | None = None
        framestamp: int | None = None

        if self._timestamps_raw is not None:
            timestamp_sec = int(self._timestamps_raw[index, 0])
            timestamp_microsec = int(self._timestamps_raw[index, 1])
        if self._framestamps is not None:
            framestamp = int(self._framestamps[index])

        return {
            "frame_index": int(index),
            "timestamp_sec": timestamp_sec,
            "timestamp_microsec": timestamp_microsec,
            "framestamp": framestamp,
            "camerastamp": None,
            "conversionfactor_coeff": None,
            "conversionfactor_offset": None,
        }

    def metadata_summary(self) -> dict[str, Any]:
        return {
            "backend": "native",
            "nfrms": int(self.nfrms),
            "width": int(self.xsize),
            "height": int(self.ysize),
            "dtype": str(np.dtype(self.dtype)),
            "pixel_type": int(self.byte_depth),
            "rowbytes": int(self.bytes_per_row),
            "file_format_version": int(self._file_format_version),
        }


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------


class DCIMGFile:
    """DCIMG reader facade.

    Parameters
    ----------
    file_path : str | Path
        DCIMG file path.
    backend : str
        "auto" (default), "sdk", or "native".
        "auto" prefers SDK backend and falls back to native parser.
    """

    def __init__(self, file_path: str | Path, backend: str = "auto"):
        self.file_path = Path(file_path)
        self.backend = backend
        self._impl: _SdkDcimgBackend | _NativeDcimgBackend
        self.backend_name = ""

        if backend not in {"auto", "sdk", "native"}:
            raise ValueError("backend must be one of: auto, sdk, native")

        if backend == "sdk":
            self._impl = _SdkDcimgBackend(self.file_path)
            self.backend_name = "sdk"
        elif backend == "native":
            self._impl = _NativeDcimgBackend(self.file_path)
            self.backend_name = "native"
        else:
            try:
                self._impl = _SdkDcimgBackend(self.file_path)
                self.backend_name = "sdk"
            except Exception:
                self._impl = _NativeDcimgBackend(self.file_path)
                self.backend_name = "native"

    def __enter__(self) -> "DCIMGFile":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def nfrms(self) -> int:
        return int(self._impl.nfrms)

    @property
    def xsize(self) -> int:
        return int(self._impl.xsize)

    @property
    def ysize(self) -> int:
        return int(self._impl.ysize)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._impl.dtype)

    def close(self) -> None:
        self._impl.close()

    def frame(self, index: int, copy: bool = True) -> np.ndarray:
        return self._impl.frame(index, copy=copy)

    def frame_with_metadata(
        self, index: int, copy: bool = True
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return self._impl.frame_with_metadata(index, copy=copy)

    def frame_metadata(self, index: int) -> dict[str, Any]:
        return self._impl.frame_metadata(index)

    def metadata_summary(self) -> dict[str, Any]:
        out = self._impl.metadata_summary()
        out["selected_backend"] = self.backend_name
        return out


def get_dcimg_file_class() -> type[DCIMGFile]:
    """Return DCIMG reader facade class used by asvimg I/O layer."""
    return DCIMGFile
