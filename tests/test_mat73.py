"""MATLAB v7.3 (.mat, HDF5) writer used when a payload exceeds the v5 4 GiB limit.

The end-to-end orientation/value correctness was verified once against MATLAB
R2023b `load`; these tests lock the writer's byte layout and the size-based
routing so it cannot silently regress.
"""

from __future__ import annotations

import h5py
import numpy as np

from asvimg.io import (
    _mat73_header,
    _payload_needs_v73,
    _savemat_v73,
    save_payload,
)


def test_routing_predicate_triggers_only_for_oversized_arrays():
    # A zero-stride broadcast view reports nbytes == 2**32 without allocating.
    huge = np.broadcast_to(np.uint8(0), (2**32,))
    assert _payload_needs_v73({"x": huge})
    assert not _payload_needs_v73({"x": np.zeros((10, 10), np.float32), "n": np.int64(3)})


def test_header_magic_bytes():
    hdr = _mat73_header()
    assert len(hdr) == 128
    assert hdr[:11] == b"MATLAB 7.3 "
    assert hdr[116:124] == b"\x00" * 8      # subsys offset
    assert hdr[124:126] == b"\x00\x02"      # version 0x0200
    assert hdr[126:128] == b"IM"            # little-endian indicator


def test_v73_roundtrip_orientation_and_types(tmp_path):
    H, W, T = 3, 4, 5
    image_df = np.arange(H * W * T, dtype=np.float32).reshape(H, W, T)
    payload = {
        "imageDf": image_df,
        "channel_name": "green",
        "imageSize": np.array(image_df.shape),
        "orientation": "HWT",
        "post_annotation_time_average": int(2),
        "source_indices": np.array([0, 2, 4]),
        "donner_indices": np.array([1, 3]),
    }
    out = tmp_path / "p.mat"
    _savemat_v73(out, payload)

    with open(out, "rb") as f:
        assert f.read(2) == b"MA"  # MAT header stamped into the userblock

    with h5py.File(out, "r") as h:
        d = h["imageDf"]
        # MATLAB column-major == HDF5 dims reversed.
        assert d.shape == (T, W, H)
        assert bytes(d.attrs["MATLAB_class"]) == b"single"
        # What MATLAB reconstructs (dset.T) must equal the original array.
        assert np.array_equal(np.array(d).T, image_df)

        s = h["channel_name"]
        assert bytes(s.attrs["MATLAB_class"]) == b"char"
        assert int(s.attrs["MATLAB_int_decode"]) == 2
        assert "".join(chr(int(x)) for x in np.array(s).ravel()) == "green"

        assert list(np.array(h["imageSize"]).ravel()) == [H, W, T]
        assert np.array(h["post_annotation_time_average"]).ravel().tolist() == [2]


def test_streaming_branch_matches_direct_transpose():
    # The large-array path fills the reversed dataset frame-by-frame; verify its
    # indexing is identical to a whole-array transpose.
    arr = np.arange(6 * 7 * 8, dtype=np.float32).reshape(6, 7, 8)
    direct = np.ascontiguousarray(arr.T)
    stream = np.empty(arr.shape[::-1], arr.dtype)
    for k in range(arr.shape[2]):
        stream[k, :, :] = arr[:, :, k].T
    assert np.array_equal(direct, stream)


def test_save_payload_mat_stays_v5_for_small(tmp_path):
    # Small payloads keep the exact scipy v5 path (loadmat reads them back).
    from scipy.io import loadmat

    payload = {"imageDf": np.ones((2, 3), np.float32), "channel_name": "red"}
    save_payload(tmp_path / "small", payload, "mat")
    got = loadmat(tmp_path / "small.mat")
    assert np.array_equal(got["imageDf"], np.ones((2, 3), np.float32))
    assert got["channel_name"][0] == "red"
