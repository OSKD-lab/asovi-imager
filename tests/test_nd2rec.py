"""Nikon .nd2 reader (asvimg.nd2rec + io.py dispatch).

The ``nd2`` package cannot write ND2 files, so these tests swap in a stand-in
for ``nd2.ND2File`` with the same surface the reader uses: ``sizes`` /
``dtype`` / ``read_frame(seq) -> (C, Y, X)`` / ``frame_metadata`` /
``metadata`` / ``attributes`` / ``voxel_size`` / ``close``.  The (C, Y, X)
per-sequence layout is what the real library returns for a multi-channel file
(checked against the OME sample ``header_test1.nd2``).  Each pixel encodes
``seq * 10 + channel`` so the flattening order can be read straight off it.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from asvimg import io, nd2rec
from asvimg.nd2rec import Nd2Recording


class _FakeND2:
    def __init__(self, sizes, names=None, period_ms=25.0):
        self.sizes = dict(sizes)
        self.dtype = np.dtype(np.uint16)
        self.closed = False
        n_ch = self.sizes.get("C", 1)
        self._names = names or [f"ch{i}" for i in range(n_ch)]
        self._period_ms = period_ms
        self.metadata = SimpleNamespace(
            channels=[SimpleNamespace(channel=SimpleNamespace(name=n)) for n in self._names]
        )
        self.attributes = SimpleNamespace(bitsPerComponentSignificant=12)
        self.reads = 0

    def read_frame(self, seq):
        self.reads += 1
        h, w = self.sizes["Y"], self.sizes["X"]
        n_ch = self.sizes.get("C", 1)
        arr = np.stack(
            [np.full((h, w), seq * 10 + c, dtype=np.uint16) for c in range(n_ch)]
        )
        arr = arr if "C" in self.sizes else arr[0]
        # the real one is a read-only view into the file's memmap
        arr.flags.writeable = False
        return arr

    def frame_metadata(self, seq):
        t = SimpleNamespace(relativeTimeMs=seq * self._period_ms)
        return SimpleNamespace(
            channels=[SimpleNamespace(time=t) for _ in self._names]
        )

    def voxel_size(self):
        return SimpleNamespace(x=6.5, y=6.5, z=1.0)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_nd2(monkeypatch):
    """Route every ``.nd2`` open to a fake; returns a setter for its sizes."""
    state = {"sizes": {"T": 4, "C": 2, "Y": 6, "X": 8}, "names": ["GCaMP", "Iso"]}
    opened: list[_FakeND2] = []

    def _open(path):
        f = _FakeND2(state["sizes"], state["names"])
        opened.append(f)
        return f

    monkeypatch.setattr(nd2rec, "_open_nd2", _open)
    state["opened"] = opened
    return state


def _touch(folder, name="rec.nd2"):
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_bytes(b"")  # detection is by extension; the fake supplies the pixels
    return p


def test_flattens_t_by_c_into_interleaved_stream(tmp_path, fake_nd2):
    p = _touch(tmp_path)
    assert io.get_frame_count(p) == 8  # T=4 x C=2
    values = [int(f[0, 0]) for f in io.iter_frames(p)]
    # t0c0, t0c1, t1c0, t1c1, ... -> frame k is channel k % C
    assert values == [0, 1, 10, 11, 20, 21, 30, 31]
    frames = list(io.iter_frames(p))
    assert frames[0].shape == (6, 8) and frames[0].flags["C_CONTIGUOUS"]


def test_each_sequence_is_read_once(tmp_path, fake_nd2):
    p = _touch(tmp_path)
    list(io.iter_frames(p))
    assert fake_nd2["opened"][-1].reads == 4  # one read_frame per sequence, not per channel
    assert fake_nd2["opened"][-1].closed


def test_random_access_and_metadata(tmp_path, fake_nd2):
    p = _touch(tmp_path)
    got = io.load_frames_by_indices(p, [5, 0, 6])
    assert [int(f[0, 0]) for f in got] == [21, 0, 30]
    assert got[0].dtype == np.float64

    metas = [m for _, m in io.iter_frames_with_metadata(p)]
    assert [m["nd2_sequence"] for m in metas] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [m["nd2_channel"] for m in metas] == [0, 1] * 4
    assert metas[3]["timestamp_sec"] == pytest.approx(0.025)

    summary = io.load_input_metadata(p)["summary"]
    assert summary["backend"] == "nd2"
    assert summary["n_channels"] == 2 and summary["n_sequences"] == 4
    assert summary["channel_names"] == ["GCaMP", "Iso"]


def test_single_channel_file(tmp_path, fake_nd2):
    fake_nd2["sizes"] = {"T": 3, "Y": 4, "X": 5}
    fake_nd2["names"] = ["GCaMP"]
    p = _touch(tmp_path)
    frames = list(io.iter_frames(p))
    assert [int(f[0, 0]) for f in frames] == [0, 10, 20]
    # a single-channel plane is already contiguous: it must still be copied
    # out of the file's view, or it dangles once the file is closed
    assert all(f.flags.writeable and f.flags.owndata for f in frames)


@pytest.mark.parametrize("axis", ["Z", "P", "S"])
def test_rejects_axes_that_would_fold_into_time(tmp_path, fake_nd2, axis):
    fake_nd2["sizes"] = {"T": 2, axis: 3, "C": 2, "Y": 4, "X": 5}
    p = _touch(tmp_path)
    with pytest.raises(ValueError, match="only T/C/Y/X"):
        Nd2Recording(p)
    assert fake_nd2["opened"][-1].closed  # the handle is not leaked


def test_detected_as_its_own_format(tmp_path, fake_nd2):
    folder = tmp_path / "in"
    p = _touch(folder)
    assert io.find_input_files(folder) == [p]
    assert io.find_input_files(folder, "nd2") == [p]
    (folder / "other.tif").write_bytes(b"")
    with pytest.raises(ValueError, match="multiple input formats"):
        io.find_input_files(folder)


def test_missing_extra_names_it(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _no_nd2(name, *a, **k):
        if name == "nd2":
            raise ModuleNotFoundError("No module named 'nd2'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_nd2)
    with pytest.raises(ModuleNotFoundError, match=r"asovi-imager\[nd2\]"):
        Nd2Recording(_touch(tmp_path))


def _run_preprocess(tmp_path, channels_name, channels_prop):
    from asvimg import PipelineConfig, read_reg_meta
    from asvimg.preprocess import PreprocessRunner

    inp = tmp_path / "in"
    _touch(inp)
    out = tmp_path / "out"
    cfg = PipelineConfig(
        input_dir=str(inp), output_dir=str(out),
        channels_name=channels_name, channels_prop=channels_prop,
        do_registration=False, linear_subt=False, binning=1,
        output_format="npy", output_metadata_yaml=False, demux_qc=False,
    )
    PreprocessRunner(cfg).run()
    return out, read_reg_meta(out)


def test_preprocess_demuxes_nd2_channels(tmp_path, fake_nd2):
    out, meta = _run_preprocess(tmp_path, ["GCaMP", "Iso"], ["source", "donner"])
    ch0 = io.load_reg_channel(out, 0)
    ch1 = io.load_reg_channel(out, 1)
    # every ND2 channel lands in its own reg_Ch, in time order
    assert [int(f[0, 0]) for f in ch0] == [0, 10, 20, 30]
    assert [int(f[0, 0]) for f in ch1] == [1, 11, 21, 31]


def test_preprocess_refuses_cycle_not_matching_nd2_channels(tmp_path, fake_nd2):
    fake_nd2["sizes"] = {"T": 4, "C": 3, "Y": 6, "X": 8}
    fake_nd2["names"] = ["a", "b", "c"]
    with pytest.raises(ValueError, match="multiple of 3"):
        _run_preprocess(tmp_path, ["A", "B"], ["source", "donner"])
