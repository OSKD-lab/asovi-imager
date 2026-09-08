"""Ito even/odd HDF5 recording reader (asvimg.h5rec + io.py dispatch).

Fixtures use gzip-compressed frames on purpose: gzip is filter id 1 (built into
libhdf5, no hdf5plugin needed), so a passing read proves the reader is
codec-agnostic and does not special-case blosc2.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from asvimg import io
from asvimg.h5rec import H5Recording, is_ito_h5_recording


def _write_recording(folder, n_per_part=5, h=8, w=10, start=100):
    """Write a 2-part even/odd recording; frame pixels encode the source_index so
    ordering can be checked. Returns the list of source_index values in order."""
    folder.mkdir(parents=True, exist_ok=True)
    order = []
    for grp, s0 in (("even", start), ("odd", start + 1)):
        src = np.arange(s0, s0 + 2 * n_per_part, 2, dtype=np.int64)
        # each frame is filled with its own source_index -> unambiguous identity
        frames = np.stack([np.full((h, w), s, dtype=np.uint16) for s in src])
        with h5py.File(folder / f"rec_{grp}_job0000.h5", "w") as f:
            f.create_dataset(
                "frames", data=frames, compression="gzip", compression_opts=6,
                chunks=(2, h, w),
            )
            f.create_dataset("source_indices", data=src)
            f.create_dataset(
                "camera_timestamp_ns", data=(src * 1000).astype(np.int64)
            )
            f.create_dataset("host_time_ns", data=(src * 1000 + 7).astype(np.int64))
        order.extend(int(s) for s in src)
    return sorted(order)


def test_detection_and_find_input_files(tmp_path):
    folder = tmp_path / "rec"
    _write_recording(folder)
    assert is_ito_h5_recording(folder) is True
    assert is_ito_h5_recording(tmp_path / "nope") is False

    files = io.find_input_files(folder)
    assert files == [folder]                       # one entry = the whole folder
    assert io.resolve_exp_stem(folder) == "rec"    # exp name from the folder


def test_merge_is_interleaved_by_source_index(tmp_path):
    folder = tmp_path / "rec"
    expected = _write_recording(folder, n_per_part=5, start=100)

    with H5Recording(folder) as rec:
        assert rec.nfrms == len(expected)          # 2 parts * 5 frames
        assert rec.shape_hw == (8, 10)
        assert rec.dtype == np.dtype(np.uint16)
        # frames come out in source_index order 100,101,102,... (even/odd merged)
        got = [int(rec.frame(i)[0, 0]) for i in range(rec.nfrms)]
        assert got == expected
        # strictly alternating parity => the positional demux splits it cleanly
        assert got == sorted(got)
        assert [s % 2 for s in got] == [0, 1] * 5


def test_frame_metadata_and_summary(tmp_path):
    folder = tmp_path / "rec"
    _write_recording(folder, start=100)

    with H5Recording(folder) as rec:
        _, meta = rec.frame_with_metadata(1)       # 2nd frame = source_index 101
        assert meta["source_index"] == 101
        assert meta["framestamp"] == 101
        assert meta["camera_timestamp_ns"] == 101 * 1000
        assert meta["timestamp_sec"] == pytest.approx(101 * 1000 / 1e9)
        assert meta["image_height"] == 8 and meta["image_width"] == 10
        assert meta["bits_per_pixel"] == 16

        summary = rec.metadata_summary()
        assert summary["backend"] == "h5_ito"
        assert summary["nfrms"] == 10
        assert summary["n_parts"] == 2
        assert summary["source_start"] == 100 and summary["source_end"] == 109
        # gzip == filter id 1 (deflate); proves reading is codec-agnostic
        assert summary["compression"] == ["1:deflate"]


def test_io_dispatch_matches_direct_reader(tmp_path):
    folder = tmp_path / "rec"
    expected = _write_recording(folder, start=100)

    assert io.get_frame_count(folder) == len(expected)

    frames = list(io.iter_frames(folder))
    assert [int(f[0, 0]) for f in frames] == expected

    idxs = [int(m["source_index"]) for _, m in io.iter_frames_with_metadata(folder)]
    assert idxs == expected

    picked = io.load_frames_by_indices(folder, [0, 1, 9])
    assert [f.dtype for f in picked] == [np.float64] * 3       # loader upcasts
    assert [int(f[0, 0]) for f in picked] == [expected[0], expected[1], expected[9]]

    md = io.load_input_metadata(folder, frame_indices=[0])
    assert md["summary"]["backend"] == "h5_ito"
    assert md["frames"][0]["source_index"] == expected[0]


def test_input_format_auto_errors_on_mixed_formats(tmp_path):
    """auto refuses a folder that mixes formats, rather than silently picking one."""
    import tifffile

    folder = tmp_path / "rec"
    _write_recording(folder, start=100)                 # h5 parts
    tifffile.imwrite(folder / "stray.tif", np.zeros((2, 8, 10), np.uint16))

    with pytest.raises(ValueError, match="multiple input formats"):
        io.find_input_files(folder, "auto")

    # explicit selection resolves the mix either way
    assert io.find_input_files(folder, "tif") == [folder / "stray.tif"]
    assert io.find_input_files(folder, "h5") == [folder]


def test_input_format_explicit_absent_raises(tmp_path):
    folder = tmp_path / "rec"
    _write_recording(folder, start=100)                 # only h5 present
    with pytest.raises(FileNotFoundError, match="no dcimg input"):
        io.find_input_files(folder, "dcimg")


def test_input_format_auto_single_format_ok(tmp_path):
    folder = tmp_path / "rec"
    expected = _write_recording(folder, start=100)
    assert io.find_input_files(folder, "auto") == [folder]
    assert io.get_frame_count(folder) == len(expected)


def test_source_indices_fallback_from_attrs(tmp_path):
    """A part without a source_indices dataset falls back to the start/step attrs."""
    folder = tmp_path / "rec"
    folder.mkdir()
    with h5py.File(folder / "rec_even_job0000.h5", "w") as f:
        frames = np.stack([np.full((4, 4), i, dtype=np.uint16) for i in range(3)])
        d = f.create_dataset("frames", data=frames, compression="gzip", chunks=(1, 4, 4))
        d  # noqa
        f.attrs["source_start_frame"] = 200
        f.attrs["source_frame_step"] = 2
    with H5Recording(folder) as rec:
        idxs = [rec.frame_with_metadata(i)[1]["source_index"] for i in range(rec.nfrms)]
        assert idxs == [200, 202, 204]
