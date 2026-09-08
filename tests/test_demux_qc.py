"""Tests for intensity-based demux (channel-cycle) QC."""

import matplotlib

matplotlib.use("Agg")  # headless: never open a window

import numpy as np

from asvimg.demux_qc import (
    DemuxCorrection,
    DemuxQC,
    analyze_demux,
    build_demux_correction,
    file_slip_edits,
    load_demux_correction,
    plot_demux_qc,
    save_demux_correction,
    suggest_correction,
)


def _make_means(n, cycle_len, amps, *, drift=0.0, noise=0.0, seed=0, slip_at=None, slip_by=1):
    """Synthetic per-frame mean intensities for a bright/dim channel cycle.

    ``slip_at`` models dropped frames: from that index on, every frame's phase
    shifts by ``slip_by`` (the exact silent failure the QC must catch).
    """
    rng = np.random.default_rng(seed)
    means = np.empty(n, dtype=np.float64)
    for g in range(n):
        phase = g % cycle_len
        if slip_at is not None and g >= slip_at:
            phase = (g + slip_by) % cycle_len
        means[g] = amps[phase] + drift * g + noise * rng.standard_normal()
    return means


def test_clean_alternating_is_consistent():
    means = _make_means(600, 2, [100.0, 60.0], drift=-0.03, noise=2.0, seed=1)
    qc = analyze_demux(means, 2)
    assert isinstance(qc, DemuxQC)
    assert qc.conclusive
    assert qc.separability > 0.3
    assert qc.dominant_period == 2
    assert qc.period_ok
    assert not qc.slip_detected
    assert qc.ok


def test_phase_slip_detected():
    means = _make_means(600, 2, [100.0, 60.0], drift=-0.02, noise=2.0, seed=2, slip_at=300)
    qc = analyze_demux(means, 2)
    assert qc.conclusive
    assert qc.slip_detected
    assert qc.slip_frame is not None
    assert 220 <= qc.slip_frame <= 380  # localized near the injected drop
    assert not qc.ok


def test_equal_brightness_is_inconclusive_no_false_alarm():
    means = _make_means(600, 2, [80.0, 80.0], noise=5.0, seed=3)
    qc = analyze_demux(means, 2)
    assert not qc.conclusive
    assert not qc.slip_detected  # must NOT raise a false alarm when unverifiable
    assert qc.separability < 0.05


def test_single_channel_skipped():
    means = _make_means(200, 1, [90.0], noise=3.0, seed=4)
    qc = analyze_demux(means, 1)
    assert not qc.conclusive
    assert not qc.slip_detected
    assert any("single channel" in m for m in qc.messages)


def test_cross_file_slip_via_file_starts():
    # Two files; the second (starting at 300) is shifted by one channel.
    means = _make_means(600, 2, [100.0, 60.0], noise=2.0, seed=5, slip_at=300)
    qc = analyze_demux(means, 2, file_starts=[0, 300])
    assert qc.conclusive
    assert len(qc.per_file_offset) == 2
    assert qc.per_file_offset[0] == 0  # first file aligned to the recording start
    assert qc.per_file_offset[1] == 1  # second file shifted by one channel
    assert qc.slip_detected


def test_short_recording_is_inconclusive():
    means = _make_means(6, 2, [100.0, 60.0], seed=6)
    qc = analyze_demux(means, 2)
    assert not qc.conclusive
    assert not qc.slip_detected


def test_four_channel_cycle_consistent():
    means = _make_means(800, 4, [100.0, 70.0, 55.0, 40.0], drift=-0.01, noise=2.0, seed=7)
    qc = analyze_demux(means, 4)
    assert qc.conclusive
    assert qc.period_ok  # dominant period compatible with cycle_len=4
    assert not qc.slip_detected


def test_three_channel_slip_detected():
    means = _make_means(600, 3, [100.0, 70.0, 45.0], noise=2.0, seed=9, slip_at=300)
    qc = analyze_demux(means, 3)
    assert qc.conclusive
    assert qc.dominant_period == 3
    assert qc.slip_detected


def test_four_channel_distinct_detects_two_frame_slip():
    # All four brightnesses differ -> the full phase (shifts 1..3) is resolvable.
    means = _make_means(600, 4, [100.0, 75.0, 55.0, 40.0], noise=2.0, seed=10, slip_at=300, slip_by=2)
    qc = analyze_demux(means, 4)
    assert qc.conclusive
    assert qc.slip_detected  # a 2-frame drop is visible when channels are all distinct


def test_aliased_subperiod_no_false_positive():
    # GCaMP/jRGECO/GCaMP/jRGECO reads as a period-2 brightness pattern: a clean
    # recording must NOT be flagged (the degenerate best-shift used to fake a slip).
    means = _make_means(600, 4, [100.0, 60.0, 100.0, 60.0], noise=2.0, seed=11)
    qc = analyze_demux(means, 4)
    assert qc.conclusive
    assert not qc.slip_detected
    assert qc.dominant_period == 2  # sub-period detected
    # and the limitation is surfaced to the user
    assert any("resolved only modulo 2" in m for m in qc.messages)


def test_aliased_subperiod_unit_slip_detected():
    # A single-frame drop is still visible in the period-2 fingerprint.
    means = _make_means(600, 4, [100.0, 60.0, 100.0, 60.0], noise=2.0, seed=12, slip_at=300, slip_by=1)
    qc = analyze_demux(means, 4)
    assert qc.conclusive
    assert qc.slip_detected


def test_plot_returns_figure():
    means = _make_means(400, 2, [100.0, 60.0], noise=2.0, seed=8, slip_at=200)
    qc = analyze_demux(means, 2)
    fig = plot_demux_qc(qc, means, [0], channel_labels=["Ch0 BL·source", "Ch1 BL·donner"])
    assert fig is not None
    assert len(fig.axes) == 2
    # 1x2 layout, width ratio ~4:1 (left trace wide, right fold narrow)
    w0, w1 = (fig.axes[0].get_position().width, fig.axes[1].get_position().width)
    assert 3.0 < w0 / w1 < 5.0
    # the fold panel carries the dot plot (a scatter PathCollection)
    assert any(len(a.collections) > 0 for a in fig.axes)
    import matplotlib.pyplot as plt

    plt.close(fig)


# --------------------------------------------------------------------------- #
# Correction model
# --------------------------------------------------------------------------- #


def test_correction_phases_channel_and_identity():
    c = DemuxCorrection(4)
    assert c.is_identity()
    np.testing.assert_array_equal(c.phases(8), np.arange(8) % 4)  # identity == positional

    c2 = DemuxCorrection(4, start_offset=1, edits=[(4, None, 1)])  # to-end slip + offset
    assert not c2.is_identity()
    exp = np.array([(g + 1 + (1 if g >= 4 else 0)) % 4 for g in range(8)])
    np.testing.assert_array_equal(c2.phases(8), exp)
    assert c2.channel_at(0) == 1
    assert c2.channel_at(4) == (4 + 2) % 4
    # a rotation by a full cycle is still identity
    assert DemuxCorrection(4, start_offset=0, edits=[(3, None, 4)]).is_identity()


def test_correction_bounded_range_edit():
    # a bounded swap [3, 6): only those frames are phase-shifted, then it reverts
    c = DemuxCorrection(2, edits=[(3, 6, 1)])
    ph = c.phases(9)
    exp = np.array([(g % 2) if not (3 <= g < 6) else ((g + 1) % 2) for g in range(9)])
    np.testing.assert_array_equal(ph, exp)
    assert c.channel_at(2) == 2 % 2          # before the range: positional
    assert c.channel_at(5) == (5 + 1) % 2    # inside the range: shifted
    assert c.channel_at(6) == 6 % 2          # after the range: back to positional


def test_file_slip_edits_maps_per_file_slips_to_ranges():
    # 3 files of 10 frames; only the 2nd is one step into the cycle
    edits = file_slip_edits([0, 1, 0], [10, 10, 10], 2)
    assert edits == [(10, 20, 1)]  # exactly the 2nd file's global frame range

    c = DemuxCorrection(2, edits=edits)
    assert c.channel_at(0) == 0    # file 1: positional
    assert c.channel_at(10) == 1   # file 2: first frame is channels_name[1]
    assert c.channel_at(11) == 0
    assert c.channel_at(20) == 0   # file 3: back to positional

    # slips are taken mod cycle_len, and a full-cycle slip is a no-op
    assert file_slip_edits([0, 2, 0], [4, 4, 4], 2) == []
    assert file_slip_edits([0, 3], [4, 4], 2) == [(4, 8, 1)]
    # no slips / single channel / extra entries past the last file -> nothing
    assert file_slip_edits(None, [4, 4], 2) == []
    assert file_slip_edits([0, 1], [4, 4], 1) == []
    assert file_slip_edits([0, 1, 1], [4], 2) == []


def test_build_demux_correction_config_vs_sidecar(tmp_path):
    from asvimg import PipelineConfig

    cfg = PipelineConfig(
        input_dir=str(tmp_path),
        channels_name=["G", "R"],
        channels_prop=["source", "donner"],
        channels_slip=[0, 1],
        demux_start_offset=1,
    )
    corr = build_demux_correction(cfg, tmp_path, [6, 6])
    assert corr.start_offset == 1 and corr.edits == [(6, 12, 1)]  # offset + slip compose
    assert corr.channel_at(0) == 1
    assert corr.channel_at(6) == (6 + 1 + 1) % 2

    # a demux-editor sidecar wins outright over the config fields
    save_demux_correction(tmp_path, DemuxCorrection(2, start_offset=0, edits=[(3, 5, 1)]))
    corr2 = build_demux_correction(cfg, tmp_path, [6, 6])
    assert corr2.start_offset == 0 and corr2.edits == [(3, 5, 1)]


def test_correction_save_load_roundtrip(tmp_path):
    c = DemuxCorrection(4, start_offset=2, edits=[(100, None, 1), (250, 300, 3)])
    save_demux_correction(tmp_path, c)
    got = load_demux_correction(tmp_path)
    assert got is not None
    assert got.cycle_len == 4 and got.start_offset == 2
    assert got.edits == [(100, None, 1), (250, 300, 3)]  # to-end + bounded preserved
    assert load_demux_correction(tmp_path, cycle_len=2) is None  # cycle mismatch -> ignored
    assert load_demux_correction(tmp_path / "nope") is None  # absent -> None


def test_correction_loads_legacy_two_tuple(tmp_path):
    import json

    from asvimg.demux_qc import demux_correction_path

    # legacy format stored edits as (frame, delta) == a slip that runs to the end
    demux_correction_path(tmp_path).write_text(
        json.dumps({"cycle_len": 2, "start_offset": 0, "edits": [[120, 1]]}),
        encoding="utf-8",
    )
    got = load_demux_correction(tmp_path)
    assert got is not None
    assert got.edits == [(120, None, 1)]


def test_suggest_correction_undoes_slip():
    means = _make_means(600, 4, [100.0, 75.0, 55.0, 40.0], noise=2.0, seed=20, slip_at=300, slip_by=2)
    qc = analyze_demux(means, 4)
    assert qc.slip_detected
    corr = suggest_correction(qc)
    assert corr.edits and corr.edits[0][2] == 2  # edit delta (3rd elem) == injected slip_by
    assert corr.edits[0][1] is None  # a slip persists to the end
    # applying it realigns the channel identity across the slip
    phases = corr.phases(len(means))
    pre = np.argmax([means[:300][phases[:300] == p].mean() for p in range(4)])
    post = np.argmax([means[300:][phases[300:] == p].mean() for p in range(4)])
    assert pre == post


def test_qc_figure_overlays_the_correction():
    """With channels_slip in force the QC panel must show BOTH folds: the
    positional one (collapsed by the slip) and the corrected one (clean)."""
    import matplotlib.pyplot as plt

    from asvimg.demux_qc import _eta2_by_phase, _fold_by_phase

    # 3 files x 200 frames, bright/dim alternating; the 2nd file starts on 'dim'
    c_len = 2
    means = []
    for f, start in enumerate((0, 1, 0)):
        for i in range(200):
            means.append(100.0 if (i + start) % 2 == 0 else 30.0)
    means = np.asarray(means) + np.random.default_rng(0).normal(0, 1.0, 600)
    starts = [0, 200, 400]

    qc = analyze_demux(means, c_len, starts)
    corr = DemuxCorrection(c_len, edits=file_slip_edits([0, 1, 0], [200, 200, 200], c_len))

    # the numeric claim behind the overlay: globally, the positional fold is
    # washed out by the slip while the corrected fold separates the channels
    resid = means - np.convolve(means, np.ones(4) / 4, mode="same")
    eta_pos = _eta2_by_phase(resid, np.arange(600) % c_len, c_len)
    eta_cor = _eta2_by_phase(resid, corr.phases(600), c_len)
    assert eta_cor > 0.8
    assert eta_pos < 0.3
    # and the corrected fold puts the bright and dim channels on opposite signs
    m_cor, _ = _fold_by_phase(resid, corr.phases(600), c_len)
    assert m_cor[0] > 0 > m_cor[1]

    fig = plot_demux_qc(qc, means, starts, channel_labels=["G", "R"], correction=corr)
    ax_p = fig.axes[1]
    legend = [t.get_text() for t in ax_p.get_legend().get_texts()]
    assert any("positional" in t for t in legend) and any("corrected" in t for t in legend)
    assert len(ax_p.patches) == 2 * c_len  # both bar groups drawn
    plt.close(fig)

    # an identity correction must leave the figure exactly as it was (one group)
    fig2 = plot_demux_qc(qc, means, starts, correction=DemuxCorrection(c_len))
    assert len(fig2.axes[1].patches) == c_len
    assert fig2.axes[1].get_legend() is None
    plt.close(fig2)


def test_export_corrected_tiffs(tmp_path):
    import tifffile

    from asvimg import (
        DemuxCorrection,
        PipelineConfig,
        export_corrected_tiffs,
    )

    n, h, w = 40, 6, 8
    stack = np.empty((n, h, w), np.uint16)
    for i in range(n):
        stack[i] = 100 if i % 2 == 0 else 30  # even bright, odd dim
    inp = tmp_path / "in"
    inp.mkdir()
    tifffile.imwrite(inp / "recS1.tif", stack)
    cfg = PipelineConfig(input_dir=str(inp), channels_name=["A", "B"],
                         channels_prop=["source", "donner"])
    dst = tmp_path / "corr"
    # start_offset 1 -> Ch0 collects the odd (dim=30) frames
    paths = export_corrected_tiffs(cfg, DemuxCorrection(2, start_offset=1), dst)
    assert len(paths) == 2
    ch0 = tifffile.imread(paths[0])
    assert ch0.shape[0] == 20  # half the frames routed to Ch0
    assert ch0.mean() < 50  # and they are the dim frames (offset 1)


# --------------------------------------------------------------------------- #
# Preprocess integration
# --------------------------------------------------------------------------- #


def _two_level_stack(n=40, h=12, w=16):
    stack = np.empty((n, h, w), dtype=np.uint16)
    for i in range(n):
        stack[i] = 100 if i % 2 == 0 else 30  # even = bright, odd = dim
    return stack


def _run_preprocess(out, inp, stack, **over):
    import tifffile

    from asvimg import PipelineConfig, read_reg_meta
    from asvimg.preprocess import PreprocessRunner

    inp.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(inp / "recS1.tif", stack)
    cfg = PipelineConfig(
        input_dir=str(inp), output_dir=str(out),
        channels_name=["A", "B"], channels_prop=["source", "donner"],
        do_registration=False, linear_subt=False, binning=1,
        output_format="npy", output_metadata_yaml=False, demux_qc=False, **over,
    )
    PreprocessRunner(cfg).run()
    return read_reg_meta(out)


def test_preprocess_applies_config_start_offset(tmp_path):
    stack = _two_level_stack()
    m0 = _run_preprocess(tmp_path / "o0", tmp_path / "in0", stack, demux_start_offset=0)
    m1 = _run_preprocess(tmp_path / "o1", tmp_path / "in1", stack, demux_start_offset=1)
    # offset 0: Ch0 = even (bright); offset 1: Ch0 = odd (dim) -> the means swap.
    assert m0["meanImageCh0"].mean() > 50 > m1["meanImageCh0"].mean()


def _run_preprocess_two_files(out, inp, stacks, **over):
    import tifffile

    from asvimg import PipelineConfig, read_reg_meta
    from asvimg.preprocess import PreprocessRunner

    inp.mkdir(parents=True, exist_ok=True)
    for i, stack in enumerate(stacks, start=1):
        tifffile.imwrite(inp / f"rec_{i}.tif", stack)
    cfg = PipelineConfig(
        input_dir=str(inp), output_dir=str(out),
        channels_name=["A", "B"], channels_prop=["source", "donner"],
        do_registration=False, linear_subt=False, binning=1,
        output_format="npy", output_metadata_yaml=False, demux_qc=False, **over,
    )
    PreprocessRunner(cfg).run()
    return read_reg_meta(out)


def test_preprocess_applies_channels_slip_per_file(tmp_path):
    first = _two_level_stack(n=20)                 # starts bright (A, B, A, B, ...)
    second = np.roll(_two_level_stack(n=20), 1, 0)  # starts dim: this file is 1 step in

    # Undeclared, the 2nd file's dim frames land in Ch0 -> both channels are mixed.
    mixed = _run_preprocess_two_files(tmp_path / "o0", tmp_path / "in0", [first, second])
    assert 50 < mixed["meanImageCh0"].mean() < 90

    # channels_slip=[0, 1] declares the slip; every bright frame goes back to Ch0.
    m = _run_preprocess_two_files(
        tmp_path / "o1", tmp_path / "in1", [first, second], channels_slip=[0, 1]
    )
    assert m["meanImageCh0"].mean() > 90  # A = bright only
    assert m["meanImageCh1"].mean() < 40  # B = dim only


def test_preprocess_sidecar_overrides_config(tmp_path):
    stack = _two_level_stack()
    out = tmp_path / "out"
    out.mkdir()
    save_demux_correction(out, DemuxCorrection(2, start_offset=1))
    # config says offset 0, but the sidecar (offset 1) must win and survive delete=True
    m = _run_preprocess(out, tmp_path / "in", stack, delete=True, demux_start_offset=0)
    assert m["meanImageCh0"].mean() < 50  # Ch0 got the dim (odd) frames per the sidecar
