"""Convert Andor ``.sifx`` spool acquisitions into stacked BigTIFF files.

The spool is streamed frame-by-frame straight into a :class:`tifffile.TiffWriter`,
so even multi-GB acquisitions convert with a small, constant memory footprint
(one frame at a time) instead of stacking everything into RAM first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .cancel import Cancelled, CancellationToken
from .sifx import SIFXFile

__all__ = ["find_sifx", "convert_sifx_to_tiff", "main"]


def find_sifx(folder: str | Path) -> list[Path]:
    """Return the ``.sifx`` files reachable from ``folder``.

    Accepts either a ``.sifx`` file directly, or a directory — searched
    non-recursively first, then recursively as a fallback.
    """
    p = Path(folder)
    if p.is_file() and p.suffix.lower() == ".sifx":
        return [p]
    files = sorted(p.glob("*.sifx"))
    if not files:
        files = sorted(p.rglob("*.sifx"))
    return files


def _cancelled(cancel) -> bool:
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    is_set = getattr(cancel, "is_set", None)
    return bool(is_set()) if callable(is_set) else False


def convert_sifx_to_tiff(
    sifx_path: str | Path,
    out_path: str | Path,
    *,
    compression: bool = False,
    max_frames: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    cancel: CancellationToken | Callable[[], bool] | None = None,
) -> int:
    """Stream every frame of a ``.sifx`` spool into one stacked BigTIFF.

    Parameters
    ----------
    sifx_path : path to the ``.sifx`` file (its folder holds the ``*spool.dat``).
    out_path  : output ``.tif`` path (parent dirs are created).
    compression : zlib-compress the TIFF (slower, smaller) when True.
    max_frames : cap the number of frames written (None -> all).
    progress   : optional ``callback(current, total)`` for UI feedback.
    cancel     : a :class:`CancellationToken` (or ``() -> bool``) to abort;
                 aborting raises :class:`Cancelled` after closing a valid,
                 partial TIFF.

    Returns
    -------
    int : number of frames written.
    """
    import tifffile

    sifx_path = Path(sifx_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    compress = "zlib" if compression else None

    written = 0
    with SIFXFile(sifx_path) as sifx:
        total = sifx.effective_nfrms
        if max_frames is not None:
            total = min(total, int(max_frames))
        with tifffile.TiffWriter(str(out_path), bigtiff=True) as tif:
            for i in range(total):
                if _cancelled(cancel):
                    if progress is not None:
                        progress(written, total)
                    raise Cancelled()
                frame = sifx.frame(i, copy=False)
                # contiguous streaming is only valid without compression
                tif.write(
                    frame,
                    contiguous=(compress is None),
                    compression=compress,
                    photometric="minisblack",
                )
                written += 1
                if progress is not None and (i % 50 == 0 or written == total):
                    progress(written, total)
    return written


def main(argv: list[str] | None = None) -> int:
    """CLI: ``asovi-sifx2tiff <folder-or-sifx> [--out DIR] [--compress] ...``."""
    import argparse

    ap = argparse.ArgumentParser(
        description="Convert Andor .sifx spool(s) into stacked BigTIFF files."
    )
    ap.add_argument("path", help="a .sifx file, or a folder holding one/more spools")
    ap.add_argument("--out", default=None,
                    help="output directory (default: next to each .sifx)")
    ap.add_argument("--compress", action="store_true", help="zlib-compress the TIFF")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap frames written per spool (default: all)")
    args = ap.parse_args(argv)

    sifx_files = find_sifx(args.path)
    if not sifx_files:
        print(f"No .sifx spool found under {args.path}")
        return 1

    out_dir = Path(args.out) if args.out else None
    for sp in sifx_files:
        dest_dir = out_dir if out_dir is not None else sp.parent
        out_path = dest_dir / f"{sp.stem}_stacked.tif"
        print(f"[sifx] {sp} -> {out_path}")

        def _prog(cur, total):
            print(f"\r  {cur}/{total} frames", end="", flush=True)

        n = convert_sifx_to_tiff(
            sp, out_path, compression=args.compress,
            max_frames=args.max_frames, progress=_prog,
        )
        print(f"\n[sifx] wrote {n} frames -> {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
