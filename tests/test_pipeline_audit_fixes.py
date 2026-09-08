"""Defects a full-pipeline audit reproduced by running the code, pinned here.

Each test is the failure that was actually observed, not a paraphrase of it:
a saturated pixel stored as near-black, a mirrored recording registered against
an unmirrored template, one dead pixel turning the dF/F into NaN, and a QC log
line taking the whole run down on a Japanese console.
"""

import numpy as np
import pytest
import tifffile

from asvimg import PipelineConfig, load_reg_channel
from asvimg.wfci import _rolling_percentile_baseline, _wfci_strip
from asvimg.preprocess import PreprocessRunner

H = W = 64


def _write_tif(inp, stack):
    inp.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(inp / "rec.tif", stack.astype(np.uint16))


def _config(inp, out, **over):
    """Overrides go through the constructor, so __post_init__ actually validates
    them -- setattr afterwards would skip every cross-field rule."""
    base = dict(
        input_dir=str(inp), output_dir=str(out), output_format="npy", exp_name="AUD",
        channels_name=["G", "G"], channels_prop=["source", "donner"],
        fps=20, binning=2, demux_qc=False, hemovar_qc=False,
        output_metadata_yaml=False, save_figures="none",
    )
    return PipelineConfig(**{**base, **over})


class TestSaturationDoesNotWrap:
    """A subpixel Fourier shift rings and overshoots the input max. Casting that
    to uint16 with a plain C cast WRAPS: the brightest pixel in the frame comes
    back near-black, poisons meanImage -> annotation, and shows up as a -10000%
    dF/F spike. MATLAB's uint16() saturates; the port has to as well."""

    def test_a_saturated_glint_stays_bright_through_registration(self, tmp_path):
        rng = np.random.default_rng(0)
        stack = np.full((8, H, W), 6000, np.float64) + rng.normal(0, 30, (8, H, W))
        stack[:, 20:40, 20:40] = 65535        # a saturated glint
        for t in range(8):                    # sub-pixel jitter -> real registration work
            stack[t] = np.roll(stack[t], (t % 3) - 1, axis=0)
        _write_tif(tmp_path / "in", stack)

        PreprocessRunner(_config(tmp_path / "in", tmp_path / "out")).run()
        reg = load_reg_channel(tmp_path / "out", 0)      # (T, H, W)

        box = reg[:, 12:18, 12:18]                       # inside the glint (binning=2)
        assert box.min() > 60000, f"saturated pixel wrapped to {box.min()}"
        assert reg.max() == 65535


class TestFlipRegistersAgainstAFlippedTemplate:
    """With flip=True the frames are mirrored on read. Registering them against an
    UNMIRRORED template finds the mirror offset instead of the motion and applies
    it as a circular shift, dragging the brain back across the frame (and wrapping
    it around the edge)."""

    def test_the_mirrored_brain_stays_where_the_mirror_put_it(self, tmp_path):
        rng = np.random.default_rng(1)
        stack = np.full((8, H, W), 500, np.float64) + rng.normal(0, 5, (8, H, W))
        stack[:, 24:40, 44:56] = 9000          # an asymmetric "brain", right of centre
        _write_tif(tmp_path / "in", stack)
        out = tmp_path / "out"

        PreprocessRunner(_config(tmp_path / "in", out, flip=True)).run()
        reg = load_reg_channel(out, 0)                       # (T, H, W)

        # np.fliplr puts a blob at cols 44..56 (of 64) at cols 8..20; binning=2 -> 4..10
        brightest = int(np.argmax(reg[0].mean(axis=0)))
        assert brightest < W // 4, (
            f"the mirrored blob was dragged to column {brightest} of {reg.shape[2]} "
            "-- the template was not flipped with the frames"
        )
        # ...and the blob did not wrap around the edge into the (empty) far side
        assert reg[0, :, -6:].max() < 3000


class TestDffIsFiniteAtDeadPixels:
    def test_a_zero_baseline_pixel_is_zero_dff_not_nan(self):
        """One dead pixel (0 in both channels) made the regression degenerate, the
        baseline 0, and the dF/F NaN -- which PCA then died on, several stages
        away, with an opaque LinAlgError."""
        rng = np.random.default_rng(2)
        src = rng.normal(1000, 20, (4, 5, 60))
        don = rng.normal(800, 15, (4, 5, 60))
        src[1, 2] = 0.0                       # dead in both channels
        don[1, 2] = 0.0
        foi = np.arange(60)

        df, base, _hemo = _wfci_strip(src, don, foi, 5.0)

        assert np.isfinite(df).all()
        assert base[1, 2] == 0.0
        np.testing.assert_array_equal(df[1, 2], 0.0)

    def test_a_stuck_donner_pixel_keeps_its_dff_scale(self):
        """A donner with no variance leaves the slope undefined. Falling back to
        a=b=0 leaves y = src + mean(src), i.e. a dF/F silently halved at exactly
        the pixels that are already suspect. Fall back to the source's own mean."""
        rng = np.random.default_rng(3)
        t = np.arange(200)
        signal = 1000.0 + 50.0 * np.sin(t / 9)
        src = np.broadcast_to(signal, (1, 2, 200)).copy()
        src += rng.normal(0, 1, src.shape)
        don = np.full((1, 2, 200), 700.0)     # stuck: zero variance
        don[0, 1] += rng.normal(0, 10, 200)   # ...except this one, a normal pixel

        df, _base, _h = _wfci_strip(src, don, np.arange(200), 5.0)

        stuck, normal = df[0, 0], df[0, 1]
        assert abs(stuck.std() / normal.std() - 1.0) < 0.15   # not ~0.5


class TestHighPassOnAVeryShortRecording:
    def test_fewer_frames_than_one_bin_does_not_give_a_nan_baseline(self):
        """T <= round(fps) decimates to a single sample, and interpolating through
        one point extrapolates to NaN -- a 100%-NaN dF/F, saved without a word."""
        x = np.abs(np.random.default_rng(4).normal(1000, 10, (3, 4, 8)))
        base = _rolling_percentile_baseline(x, fps=20.0, cutoff_sec=15.0, rank=8.0)
        assert base.shape == x.shape
        assert np.isfinite(base).all()
        assert (base > 0).all()


class TestAnInterruptedRunIsNotDone:
    """Stop flushes what was read so far, so the outputs are a valid PREFIX of the
    recording. Reporting DONE let the GUI stamp a fresh staleness signature over it
    -- a 40-of-200-frame run recorded as a complete, up-to-date run of that config
    -- and registration_cache="cached" then accepted the prefix as the cache."""

    def _run_and_stop_at(self, tmp_path, n):
        from asvimg import CancellationToken, StageId
        from asvimg.runner import PipelineSession

        rng = np.random.default_rng(5)
        _write_tif(tmp_path / "in", rng.integers(500, 3000, (200, 32, 32)))
        cfg = _config(tmp_path / "in", tmp_path / "out", binning=1,
                      do_registration=False, registration_batch_size=4)

        token = CancellationToken()
        events = []

        class StopAt:
            def on_stage(self, stage, status, *, message="", elapsed=None):
                events.append((str(stage), status))

            def on_progress(self, stage, current, total, *, message=""):
                if current >= n:
                    token.cancel()

            def on_log(self, *a, **k):
                pass

            def on_figure(self, *a, **k):
                pass

        session = PipelineSession(cfg, reporter=StopAt(), cancel=token)
        stats = session.run_preprocess()
        status = [st for stg, st in events if stg == str(StageId.PREPROCESS)][-1]
        return stats, status

    def test_cancelled_preprocess_reports_skipped_not_done(self, tmp_path):
        from asvimg import StageStatus

        stats, status = self._run_and_stop_at(tmp_path, 40)
        assert stats.cancelled
        assert 0 < stats.total_frames < 200
        assert status == StageStatus.SKIPPED

    def test_a_partial_run_is_not_served_as_the_registration_cache(self, tmp_path):
        stats, _ = self._run_and_stop_at(tmp_path, 40)
        partial_T = stats.total_frames

        # the same config again, this time asking for the cache
        cfg = _config(tmp_path / "in", tmp_path / "out", binning=1,
                      do_registration=False, registration_cache="cached")
        again = PreprocessRunner(cfg).run()
        assert not again.cached, "a cancelled run's prefix was reused as the cache"
        assert again.total_frames > partial_T


class TestConfigRejectsWhatUsedToCrashLater:
    def test_a_two_element_filter_window_is_rejected_up_front(self, tmp_path):
        """It was accepted, and blew up unpacking (fx, fy, ft) AFTER the entire
        registration pass had run."""
        with pytest.raises(ValueError, match="filter_xyt"):
            _config(tmp_path / "in", tmp_path / "out", filter_xyt=[3, 3])

    def test_usfac_zero_is_rejected(self, tmp_path):
        """It was accepted and divided by zero inside the registration kernel."""
        with pytest.raises(ValueError, match="usfac"):
            _config(tmp_path / "in", tmp_path / "out", usfac=0)


class TestStalenessSeesEveryFieldThatChangesTheDff:
    def test_the_dff_knobs_are_in_the_preprocess_signature(self):
        """detrend and the rolling high-pass change every value in dff_{name}.npy,
        but were in no stage's signature: the GUI kept reporting preprocess (and
        everything downstream) as fresh, and rewrote ops.yaml to claim the dF/F had
        been detrended when it had not."""
        from asvimg.gui.app import _STAGE_PARAMS

        for name in (
            "detrend",
            "baseline_percentile_highpass",
            "baseline_percentile_highpass_sec",
            "baseline_percentile_highpass_rank",
            "demux_start_offset",
        ):
            assert name in _STAGE_PARAMS["preprocess"], name


class TestOutputFormatsThatWereNeverExercised:
    def test_h5_payloads_can_hold_the_roi_names(self, tmp_path):
        """output_format="h5" is one of three choices in the GUI, and it CRASHED:
        HDF5 has no fixed-width unicode type, so roi_names (a <U array) raised
        "no conversion path" out of the ROI stage -- leaving a truncated .h5 that
        the correlation stage then tried to read."""
        from asvimg import load_payload, save_payload

        payload = {
            "F_dff": np.zeros((3, 10), np.float32),
            "roi_names": np.array(["MOp_L", "SSp_R", "VISp_L"]),
            "n_pixels": np.array([10, 20, 30]),
            "fps": 20.0,
            "ica_denoise_mode": "subtract",
        }
        save_payload(tmp_path / "roiSignals_G_X", payload, "h5")
        got = load_payload(tmp_path / "roiSignals_G_X.h5")

        assert list(got["roi_names"]) == ["MOp_L", "SSp_R", "VISp_L"]
        np.testing.assert_array_equal(got["F_dff"], payload["F_dff"])
        assert float(got["fps"]) == 20.0

    def test_a_failed_h5_save_leaves_no_file_to_mistake_for_a_result(self, tmp_path):
        from asvimg import save_payload

        target = tmp_path / "roiSignals_G_X"
        with pytest.raises(Exception):
            save_payload(target, {"bad": object()}, "h5")
        assert not (tmp_path / "roiSignals_G_X.h5").exists()


class TestCorrelationWithOneRoi:
    def test_a_single_roi_does_not_crash_the_stage(self):
        """An edited rois.csv with one row made np.corrcoef return a 0-d array and
        took the heatmap, the network and the CSV down with it."""
        from asvimg.correlation import _correlation_matrix

        F = np.random.default_rng(6).normal(size=(1, 50))
        for method in ("raw", "gsr", "partial"):
            C = _correlation_matrix(F, method)
            assert C.shape == (1, 1)


class TestCorrelationUsesTheWindowThePlotShows:
    def test_the_illumination_transient_does_not_correlate_everything(self, tmp_path):
        """The first seconds are unstable illumination -- a transient shared by
        every pixel, which reads as brain-wide correlation. The ROI figure already
        dropped it (start_initial_frames); the correlation did not, so INDEPENDENT
        ROIs came out strongly positively correlated."""
        from asvimg import PipelineConfig, save_payload
        from asvimg.correlation import compute_roi_correlations

        rng = np.random.default_rng(7)
        T, start = 400, 40
        F = rng.normal(0, 1, (4, T))            # four INDEPENDENT ROIs
        F[:, :start] += np.linspace(8, 0, start)  # ...plus a shared switch-on ramp

        out = tmp_path / "out"
        out.mkdir(parents=True)
        save_payload(
            out / "roiSignals_G_AUD",
            {"F_dff": F, "roi_names": np.array(["a", "b", "c", "d"]),
             "n_pixels": np.array([9, 9, 9, 9]), "fps": 20.0},
            "npy",
        )
        cfg = _config(tmp_path / "in", out, exp_name="AUD", fps=40, binning=1,
                      start_initial_frames=start, ignore_last_frames=0)

        C = compute_roi_correlations(cfg, out)["G"].C
        off = C[~np.eye(4, dtype=bool)]
        assert abs(off).max() < 0.2, f"transient still leaking into r: {off}"


class TestContralateralMirror:
    def test_mirroring_twice_returns_the_original_roi(self):
        """`width - col` is the wrong axis: a 285-px atlas is symmetric about
        col 142.0 = (w-1)/2, so every mirrored _L ROI sat one pixel too lateral --
        and mirroring back did not land where it started."""
        from asvimg.rois import Roi, contralateral

        roi = Roi("MOp_R", x=100, y=60, size=9)
        back = contralateral(contralateral(roi, 285), 285)
        assert (back.x, back.y, back.name) == (roi.x, roi.y, roi.name)

    def test_the_axis_is_the_image_centre(self):
        from asvimg.rois import Roi, contralateral

        centre = (285 - 1) / 2          # 142.0
        roi = Roi("X_R", x=120, y=10, size=5)
        mirrored = contralateral(roi, 285)
        assert (roi.x + mirrored.x) / 2 == centre


class TestDemuxCorrectionSizesTheChannels:
    def test_a_phase_slip_does_not_drop_the_tail_frames(self, tmp_path):
        """The per-channel memmaps were sized from the POSITIONAL cycle while the
        frames were assigned by the correction. A correction that gives one channel
        more frames than its positional share had that channel's tail silently
        dropped (`if off < T_per_ch[ch]`), and reg_meta then reported the wrong
        count."""
        inp = tmp_path / "in"
        inp.mkdir(parents=True)
        rng = np.random.default_rng(8)
        # two files; the second one's cycle starts one step in (channels_slip)
        tifffile.imwrite(inp / "a.tif", rng.integers(500, 3000, (10, 16, 16)).astype(np.uint16))
        tifffile.imwrite(inp / "b.tif", rng.integers(500, 3000, (11, 16, 16)).astype(np.uint16))

        cfg = _config(inp, tmp_path / "out", binning=1, do_registration=False,
                      channels_slip=[0, 1])
        PreprocessRunner(cfg).run()

        reg0 = load_reg_channel(tmp_path / "out", 0)
        reg1 = load_reg_channel(tmp_path / "out", 1)
        assert reg0.shape[0] + reg1.shape[0] == 21     # every frame landed somewhere
        for reg in (reg0, reg1):
            assert reg[-1].max() > 0                    # ...and no tail frame is a zero husk


class TestQcLogsSurviveACp932Console:
    def test_no_preprocess_log_string_needs_a_utf8_console(self):
        """hemovar QC is on by default and logged 'R²'; on a Japanese Windows
        console (cp932) print() raised UnicodeEncodeError and killed the run."""
        import re
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / "src" / "asvimg" / "preprocess.py").read_text(
            encoding="utf-8"
        )
        for lit in re.findall(r"_log\(\s*(f?[\"'].*?[\"'])\s*[,)]", src, re.S):
            try:
                lit.encode("cp932")
            except UnicodeEncodeError:  # pragma: no cover - the assertion is the point
                pytest.fail(f"non-cp932 character in a preprocess log string: {lit!r}")
