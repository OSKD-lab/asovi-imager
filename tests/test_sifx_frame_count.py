"""A .sifx spool can hold more frames than were acquired.

Andor writes whole spool files, so a run that stops early leaves zero padding in
the tail of the last one.  ``SIFXFile._total_frames`` counts what is on disk, which
is right for the *opposite* failure (a run killed mid-spool leaves a short file)
but hands the padding to the pipeline as if it were data.  The ``.sifx`` header
knows the real count, so ``effective_nfrms`` takes the smaller of the two and every
sequential reader uses it.
"""
from __future__ import annotations

import numpy as np
import pytest

from asvimg.io import get_frame_count, iter_frames
from asvimg.sifx import SIFXFile

W, H, BPP = 4, 3, 2
IMAGE_SIZE_BYTES = W * BPP * H
IMAGES_PER_FILE = 2

INI = f"""[data]
AOIWidth = {W}
AOIHeight = {H}
AOIStride = {W * BPP}
PixelEncoding = Mono16
ImageSizeBytes = {IMAGE_SIZE_BYTES}

[multiimage]
ImagesPerFile = {IMAGES_PER_FILE}
"""


def _spool(tmp_path, n_files=2):
    """A synthetic spool directory holding ``n_files * IMAGES_PER_FILE`` frames."""
    d = tmp_path / "spool"
    d.mkdir()
    (d / "acquisitionmetadata.ini").write_text(INI, encoding="utf-8")
    for seq in range(n_files):
        # Andor reverses the digits of the sequence number in the filename.
        name = f"{str(seq)[::-1]:0<10}spool.dat"
        (d / name).write_bytes(
            np.arange(IMAGES_PER_FILE * IMAGE_SIZE_BYTES, dtype=np.uint8).tobytes()
        )
    sifx = d / "rec.sifx"
    sifx.write_bytes(b"")
    return sifx


def _header(monkeypatch, nframes):
    """Make the .sifx header parse report ``nframes`` (or fail, when None)."""
    sif_parser = pytest.importorskip("sif_parser")

    def fake_np_open(path, **kwargs):
        if nframes is None:
            raise RuntimeError("unreadable header")
        return None, {"NumberOfFrames": nframes}

    monkeypatch.setattr(sif_parser, "np_open", fake_np_open)


def test_total_frames_is_what_is_on_disk(tmp_path, monkeypatch):
    _header(monkeypatch, None)
    with SIFXFile(_spool(tmp_path)) as sifx:
        assert sifx.nfrms == 4


def test_no_header_falls_back_to_the_on_disk_count(tmp_path, monkeypatch):
    _header(monkeypatch, None)
    with SIFXFile(_spool(tmp_path)) as sifx:
        assert sifx.header_nfrms is None
        assert sifx.effective_nfrms == 4
        assert sifx.padding_frames == 0


def test_padded_spool_is_trimmed_to_the_header_count(tmp_path, monkeypatch):
    """Acquisition stopped at 3 frames; the spool file was written whole."""
    _header(monkeypatch, 3)
    with SIFXFile(_spool(tmp_path)) as sifx:
        assert sifx.header_nfrms == 3
        assert sifx.effective_nfrms == 3
        assert sifx.padding_frames == 1
        # Random access is still the raw view -- only sequential readers trim.
        assert sifx.frame(3).shape == (H, W)


def test_short_spool_is_not_extended_to_the_header_count(tmp_path, monkeypatch):
    """The opposite failure: killed mid-spool, so disk has fewer than the header."""
    _header(monkeypatch, 10)
    with SIFXFile(_spool(tmp_path)) as sifx:
        assert sifx.effective_nfrms == 4
        assert sifx.padding_frames == 0


def test_io_readers_use_the_effective_count(tmp_path, monkeypatch):
    _header(monkeypatch, 3)
    sifx = _spool(tmp_path)
    assert get_frame_count(sifx) == 3
    assert len(list(iter_frames(sifx))) == 3
