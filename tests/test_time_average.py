"""Baseline tests for the memory-bounded ``annotation._time_average``.

The row-strip / in-place rewrite (Fix ①: the whole-folder export OOM'd on a
full float64 ``cumsum`` of the atlas-warped stack) must be numerically identical
to the pre-strip implementation, which is baked in here as ``_reference``.
Half-pixel / edge errors only surface on a delta and a linear
ramp, so both are exercised alongside random data, and the strip budget is
shrunk to force the multi-strip and partial-last-strip paths.
"""

from __future__ import annotations

import numpy as np
import pytest

from asvimg import annotation


def _reference(stack: np.ndarray, half_window: int) -> np.ndarray:
    """The pre-strip _time_average, verbatim, as the baseline to match."""
    if half_window <= 0:
        return stack
    T = stack.shape[-1]
    if T == 0:
        return stack
    csum = np.cumsum(stack.astype(np.float64, copy=False), axis=-1)
    out = np.empty(stack.shape, dtype=np.float64)
    for t in range(T):
        lo = max(0, t - half_window)
        hi = min(T, t + half_window + 1)
        upper = csum[..., hi - 1]
        lower = csum[..., lo - 1] if lo > 0 else 0.0
        out[..., t] = (upper - lower) / (hi - lo)
    return out.astype(stack.dtype, copy=False)


@pytest.mark.parametrize("N", [0, 1, 2, 5])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_matches_reference_random(N, dtype):
    x = np.random.default_rng(0).standard_normal((7, 5, 40)).astype(dtype)
    ref = _reference(x.copy(), N)
    got = annotation._time_average(x.copy(), N)
    atol = 1e-5 if dtype is np.float32 else 1e-9
    np.testing.assert_allclose(got, ref, rtol=0, atol=atol)


@pytest.mark.parametrize("N", [0, 1, 3, 7, 200])
def test_matches_reference_delta_and_ramp_many_strips(N, monkeypatch):
    # strip budget = 1 byte -> strip_h clamps to 1 -> one strip per row.
    monkeypatch.setattr(annotation, "_POST_STRIP_BYTES", 1)
    T = 50
    ramp = np.broadcast_to(np.arange(T, dtype=np.float64), (4, 3, T)).copy()
    delta = np.zeros((4, 3, T), dtype=np.float64)
    delta[..., T // 2] = 1.0
    for x in (ramp, delta):
        ref = _reference(x.copy(), N)
        got = annotation._time_average(x.copy(), N)
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-9)


def test_partial_last_strip(monkeypatch):
    # row_bytes = 3*20*8 = 480; budget 960 -> strip_h = 2 over H = 7
    # -> strips of 2, 2, 2, 1 (partial last).
    monkeypatch.setattr(annotation, "_POST_STRIP_BYTES", 3 * 20 * 8 * 2)
    x = np.random.default_rng(1).standard_normal((7, 3, 20)).astype(np.float64)
    ref = _reference(x.copy(), 2)
    got = annotation._time_average(x.copy(), 2)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-9)


def test_in_place_and_returns_same_object():
    x = np.random.default_rng(2).standard_normal((5, 4, 30)).astype(np.float32)
    ref = _reference(x.copy(), 2)
    out = annotation._time_average(x, 2)  # x mutated in place
    assert out is x
    np.testing.assert_allclose(x, ref, rtol=0, atol=1e-5)


def test_matches_reference_on_thw_memmap_view(tmp_path, monkeypatch):
    """Mirror the real export site exactly: a (H, W, T) view of an on-disk
    (T, H, W) float32 memmap (what ``warped_stack_on_disk`` yields), edited in
    place, then re-read from disk to confirm the strided write-back landed.
    """
    from numpy.lib.format import open_memmap

    monkeypatch.setattr(annotation, "_POST_STRIP_BYTES", 6 * 40 * 8 * 2)  # force multi-strip
    T, H, W = 40, 6, 5
    data_thw = np.random.default_rng(3).standard_normal((T, H, W)).astype(np.float32)
    path = tmp_path / "warp.npy"
    mm = open_memmap(path, mode="w+", dtype=np.float32, shape=(T, H, W))
    mm[:] = data_thw
    mm.flush()

    view = np.moveaxis(mm, 0, 2)  # (H, W, T) non-contiguous view, exactly as in export
    ref = _reference(np.moveaxis(data_thw, 0, 2).copy(), 3)
    got = annotation._time_average(view, 3)  # in place on the memmap view
    assert got is view
    np.testing.assert_allclose(np.asarray(got), ref, rtol=0, atol=1e-5)

    mm.flush()
    reread = np.moveaxis(open_memmap(path, mode="r"), 0, 2)
    np.testing.assert_allclose(np.asarray(reread), ref, rtol=0, atol=1e-5)


def test_noop_paths_return_input_untouched():
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    assert annotation._time_average(x, 0) is x
    np.testing.assert_array_equal(x, np.arange(24, dtype=np.float32).reshape(2, 3, 4))
    empty = np.zeros((2, 3, 0), dtype=np.float32)
    assert annotation._time_average(empty, 3) is empty
