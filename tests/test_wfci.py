"""Tests for the WFCI dF/F pre-regression conditioning (detrend + high-pass)."""

import numpy as np

from asvimg.wfci import (
    _exp_detrend,
    _rolling_percentile_baseline,
    wfci_corrected_df_vectorized,
)


def _drift(trace):
    """Slow-drift magnitude = |mean(first third) - mean(last third)|."""
    n = len(trace)
    k = max(1, n // 3)
    return abs(float(trace[:k].mean()) - float(trace[-k:].mean()))


def test_exp_detrend_removes_bleach():
    t = np.arange(600.0)
    osc = 40.0 * np.sin(2 * np.pi * t / 25.0)
    x = (800.0 * np.exp(-0.002 * t) + 500.0 + osc)[None, None, :] * np.ones((2, 3, 1))
    xd = _exp_detrend(x)
    trace0 = x.mean((0, 1))
    trace1 = xd.mean((0, 1))
    assert _drift(trace1) < 0.15 * _drift(trace0)  # bleach trend gone
    assert np.std(trace1) < np.std(trace0)          # dominant slow variance removed


def test_rolling_highpass_removes_slow_drift():
    t = np.arange(600.0)
    osc = 30.0 * np.sin(2 * np.pi * t / 12.0)
    y = (500.0 + 250.0 * np.exp(-0.004 * t) + osc)[None, None, :] * np.ones((2, 3, 1))
    base = _rolling_percentile_baseline(y, fps=10.0, cutoff_sec=20.0, rank=50.0)
    hp = y / base
    raw_drift = _drift((y / y.mean()).mean((0, 1)))  # normalized raw drift for comparison
    assert _drift(hp.mean((0, 1))) < 0.2 * raw_drift  # high-pass strongly flattens the drift
    assert np.all(np.isfinite(hp))


def test_wfci_default_is_backward_compatible():
    rng = np.random.default_rng(0)
    src = 500.0 + rng.normal(size=(3, 4, 200))
    donor = 400.0 + rng.normal(size=(3, 4, 200))
    foi = np.arange(20, 195)
    t = np.arange(200.0)
    a, _ = wfci_corrected_df_vectorized(src, donor, foi, src_times=t, donor_times=t)
    b, _ = wfci_corrected_df_vectorized(
        src, donor, foi, src_times=t, donor_times=t,
        detrend=False, highpass=False,  # explicit off == default
    )
    np.testing.assert_array_equal(a, b)


def test_wfci_detrend_reduces_dff_drift():
    # src carries an exponential bleach NOT shared with the donor, so the plain
    # donor regression leaves it as slow drift in dF/F; the detrend should remove it.
    n = 500
    t = np.arange(n, dtype=np.float64)
    rng = np.random.default_rng(1)
    donor = (400.0 + 30.0 * np.sin(2 * np.pi * t / 15.0))[None, None, :] * np.ones((3, 3, 1))
    donor = donor + rng.normal(scale=1.0, size=donor.shape)
    bleach = 900.0 * np.exp(-0.003 * t)
    src = (bleach + 500.0)[None, None, :] + 0.7 * (donor - donor.mean(-1, keepdims=True))
    src = src + rng.normal(scale=1.0, size=src.shape)
    foi = np.arange(40, n - 5)

    plain, _ = wfci_corrected_df_vectorized(
        src, donor, foi, src_times=t, donor_times=t, fps_channel=10.0,
    )
    dtr, _ = wfci_corrected_df_vectorized(
        src, donor, foi, src_times=t, donor_times=t, fps_channel=10.0, detrend=True,
    )
    assert _drift(dtr.mean((0, 1))) < 0.5 * _drift(plain.mean((0, 1)))


def test_wfci_highpass_reduces_dff_drift():
    n = 500
    t = np.arange(n, dtype=np.float64)
    rng = np.random.default_rng(2)
    donor = (400.0 + 30.0 * np.sin(2 * np.pi * t / 15.0))[None, None, :] * np.ones((3, 3, 1))
    donor = donor + rng.normal(scale=1.0, size=donor.shape)
    drift = 400.0 * np.exp(-0.004 * t)
    src = (drift + 600.0)[None, None, :] + 0.7 * (donor - donor.mean(-1, keepdims=True))
    src = src + rng.normal(scale=1.0, size=src.shape)
    foi = np.arange(40, n - 5)

    plain, _ = wfci_corrected_df_vectorized(
        src, donor, foi, src_times=t, donor_times=t, fps_channel=10.0,
    )
    hp, _ = wfci_corrected_df_vectorized(
        src, donor, foi, src_times=t, donor_times=t, fps_channel=10.0,
        highpass=True, highpass_cutoff_sec=15.0, highpass_rank=50.0,
    )
    assert _drift(hp.mean((0, 1))) < 0.5 * _drift(plain.mean((0, 1)))


def test_hemovar_variance_explained():
    n = 400
    t = np.arange(n, dtype=np.float64)
    rng = np.random.default_rng(3)
    donor = (400.0 + 50.0 * np.sin(2 * np.pi * t / 10.0))[None, None, :] * np.ones((2, 2, 1))
    donor = donor + rng.normal(scale=1.0, size=donor.shape)
    src = np.empty((2, 2, n))
    src[0, 0] = 2.0 * donor[0, 0] + 100 + rng.normal(scale=1.0, size=n)   # ~fully donor-driven
    src[0, 1] = 1.5 * donor[0, 1] + 50 + rng.normal(scale=25.0, size=n)   # moderate
    src[1, 0] = 300 + rng.normal(scale=30.0, size=n)                      # independent
    src[1, 1] = 300 + rng.normal(scale=30.0, size=n)                      # independent
    foi = np.arange(10, n - 5)

    _dff, _base, hv = wfci_corrected_df_vectorized(
        src, donor, foi, src_times=t, donor_times=t, return_hemovar=True,
    )
    assert hv.shape == (2, 2)
    assert 0.0 <= hv.min() and hv.max() <= 1.0
    assert hv[0, 0] > 0.9                          # strongly donor-explained pixel
    assert hv[1, 0] < 0.2 and hv[1, 1] < 0.2       # independent pixels ~ 0
    assert hv[0, 1] > hv[1, 0]                      # moderate > independent


def test_plot_hemovar_map_returns_figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from asvimg.wfci import plot_hemovar_map

    hv = np.clip(np.random.default_rng(0).random((16, 20)), 0, 1)
    fig = plot_hemovar_map(hv, "GCaMP")
    assert fig is not None and len(fig.axes) >= 1
    plt.close(fig)


def test_time_window_is_a_view_for_contiguous_foi():
    """foi is arange(start, end) in every pipeline path; fancy-indexing it copied
    the whole strip (~250 MB x2 per strip, ~730 ms). A slice is a free view."""
    from asvimg.wfci import _time_window

    arr = np.arange(2 * 3 * 10, dtype=np.float64).reshape(2, 3, 10)

    foi = np.arange(2, 8)
    win = _time_window(arr, foi)
    assert win.base is not None                      # a view, not a copy
    assert np.array_equal(win, arr[..., foi])        # ...of exactly the same values
    assert np.shares_memory(win, arr)

    # non-contiguous / unordered foi must still work (falls back to fancy indexing)
    for odd in (np.arange(0, 10, 2), np.array([7, 1, 4]), np.array([], dtype=int)):
        assert np.array_equal(_time_window(arr, odd), arr[..., odd])


def test_vectorized_reports_row_progress():
    H, W, T = 96, 8, 40
    rng = np.random.default_rng(0)
    src = rng.normal(500, 10, size=(H, W, T))
    donor = rng.normal(300, 10, size=(H, W, T))

    seen = []
    wfci_corrected_df_vectorized(
        src, donor, np.arange(T), strip_h=32,
        progress=lambda done, total: seen.append((done, total)),
    )
    # one call per row-strip, monotonic, ending on the last row
    assert seen == [(32, H), (64, H), (96, H)]


def test_preprocess_reports_dff_progress(tmp_path):
    """dF/F progress must span every group and finish at 100% — a bar that
    stalls at 50% on a 2-group recording is worse than none."""
    import tifffile

    from asvimg import PipelineConfig
    from asvimg.preprocess import PreprocessRunner

    T, H, W = 80, 32, 24
    rng = np.random.default_rng(0)
    stack = rng.normal(600, 20, size=(T, H, W)).astype(np.uint16)
    inp = tmp_path / "in"
    inp.mkdir()
    tifffile.imwrite(inp / "rec.tif", stack)

    events = []

    class Rep:
        def on_log(self, *a, **k):
            pass

        def on_stage(self, *a, **k):
            pass

        def on_figure(self, *a, **k):
            pass

        def on_progress(self, stage, current, total, *, message=""):
            if "dF/F" in message:
                events.append((current, total, message))

    cfg = PipelineConfig(
        input_dir=str(inp), output_dir=str(tmp_path / "out"),
        channels_name=["BL", "BL", "RD", "RD"],  # two donner-backed groups
        channels_prop=["source", "donner", "source", "donner"],
        do_registration=False, binning=1, output_format="npy",
        output_metadata_yaml=False, demux_qc=False, hemovar_qc=False,
    )
    PreprocessRunner(cfg).run(reporter=Rep())

    assert events, "the dF/F stage emitted no progress at all"
    current = [e[0] for e in events]
    total = events[0][1]
    assert total == 2 * H  # rows summed over both groups
    assert current == sorted(current)  # monotonic
    assert current[-1] == total  # reaches 100%
    assert any("BL" in e[2] for e in events) and any("RD" in e[2] for e in events)


def test_preprocess_writes_hemovar_map(tmp_path):
    import tifffile

    from asvimg import PipelineConfig
    from asvimg.preprocess import PreprocessRunner

    n, h, w = 40, 8, 10
    rng = np.random.default_rng(0)
    stack = np.empty((n, h, w), np.uint16)
    donor2d = 300.0 + rng.normal(scale=10.0, size=(n // 2, h, w))
    for i in range(n):
        if i % 2 == 0:  # source: correlated with the paired donor frame
            stack[i] = np.clip(
                500.0 + 0.8 * (donor2d[i // 2] - 300.0) + rng.normal(scale=5.0, size=(h, w)),
                0, 65535,
            )
        else:  # donner
            stack[i] = np.clip(donor2d[i // 2], 0, 65535)
    inp = tmp_path / "in"
    inp.mkdir()
    out = tmp_path / "out"
    tifffile.imwrite(inp / "recS1.tif", stack)
    cfg = PipelineConfig(
        input_dir=str(inp), output_dir=str(out),
        channels_name=["BL", "BL"], channels_prop=["source", "donner"],
        do_registration=False, binning=1, output_format="npy",
        output_metadata_yaml=False, demux_qc=False,
    )
    PreprocessRunner(cfg).run()
    hv_path = out / "hemovar_BL.npy"
    assert hv_path.exists()
    hv = np.load(hv_path)
    assert hv.shape == (h, w)
    assert 0.0 <= float(hv.min()) and float(hv.max()) <= 1.0
