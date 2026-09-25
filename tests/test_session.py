"""Tests for the headless PipelineSession runner and its factored stages.

Stages 2-5 are exercised against the real atlas using fabricated on-disk
preprocess outputs (no imaging data required).  Stage 1's new reporter /
cancellation hooks are exercised with a tiny synthetic TIFF.
"""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from asvimg.config import BUNDLED_ATLAS
from asvimg import (
    CancellationToken,
    PipelineConfig,
    RecordingReporter,
    dff_name_path,
    load_reg_channel,
    reg_channel_path,
    save_payload,
    write_reg_meta,
)
from asvimg.preprocess import PreprocessRunner
from asvimg.runner import (
    FakeIcaProvider,
    FakeMarksProvider,
    PipelineSession,
)

_ATLAS = BUNDLED_ATLAS


def _fabricate_preprocess_outputs(out: Path, *, H=24, W=28, T=20, seed=0) -> None:
    """Write the redesigned intermediate: reg_Ch{i}.npy (T,H,W) for a 4-ch
    cycle, dff_GCaMP.npy (donner-backed group), and the reg_meta.npz sidecar."""
    rng = np.random.default_rng(seed)
    channels_name = ["GCaMP", "jRGECO", "GCaMP", "jRGECO"]
    channels_prop = ["donner", "source", "source", "source"]
    meta = {
        "proc_template": np.zeros((H, W), dtype=np.float64),
        "imageSize": np.array([H, W]),
        "channels_name": np.array(channels_name),
        "channels_prop": np.array(channels_prop),
        "T_per_ch": np.array([T] * 4, dtype=np.int64),
        "fps": np.array(40),
    }
    for i in range(4):
        arr = rng.integers(100, 4000, size=(T, H, W)).astype(np.uint16)
        np.save(reg_channel_path(out, i), arr)
        meta[f"meanImageCh{i}"] = arr.mean(axis=0)
    write_reg_meta(out, meta)
    # GCaMP has a donner partner -> dff_GCaMP.npy (T, H, W) float32
    dff = (rng.standard_normal((T, H, W)) * 0.02).astype(np.float32)
    np.save(dff_name_path(out, "GCaMP"), dff)


def _make_4ch_config(inp: Path, out: Path) -> PipelineConfig:
    return PipelineConfig(
        input_dir=str(inp),
        output_dir=str(out),
        output_format="npy",
        exp_name="TEST",
        channels_name=["GCaMP", "jRGECO", "GCaMP", "jRGECO"],
        channels_prop=["donner", "source", "source", "source"],
        fps=40,
        ch_for_annotation=0,
        pca_n_components=5,
        annotation=False,
        roi_signal="Both",
        save_roi_signals="csv",
        save_annotated_each_ch=True,
        save_annotated_dF_mat=True,
        save_movie=False,
    )


class TestGroupDffGolden(unittest.TestCase):
    """The definition of 'this group's dF/F' — pinned against an independent
    re-implementation, not against the code under test.

    Everything downstream (ROI signals, dfWarped, movies, correlation) is built
    on it, so the ICA work must not move these numbers.  The fixture's jRGECO is
    the interesting case: NO donner and TWO sources — the配置 that exposed the
    PCA/ICA source bug.
    """

    def _expected_donnerless_dff(self, out: Path, cfg, src_indices) -> np.ndarray:
        """mean of the source channels -> per-pixel percentile baseline -> dF/F."""
        stacks = [
            np.moveaxis(np.load(reg_channel_path(out, i)), 0, 2).astype(np.float64)
            for i in src_indices
        ]
        avg = np.mean(stacks, axis=0)  # (H, W, T)
        n = cfg.resolved_start_initial_frames
        for_base = avg[..., n:] if 0 < n < avg.shape[-1] else avg
        base = np.percentile(for_base, cfg.baseline_percentile, axis=2, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(base > 0, (avg - base) / base, 0.0)

    def test_donnerless_group_is_source_mean_then_percentile_baseline(self) -> None:
        from asvimg.annotation import _load_name_dff_stack

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            out.mkdir()
            _fabricate_preprocess_outputs(out)
            cfg = _make_4ch_config(Path(td) / "in", out)
            group = cfg.channel_groups()["jRGECO"]
            self.assertEqual(group["donner_indices"], [])
            self.assertEqual(group["source_indices"], [1, 3])  # two sources, one name

            got, note = _load_name_dff_stack("jRGECO", group, cfg, out)
            want = self._expected_donnerless_dff(out, cfg, [1, 3])
            self.assertIn("baseline", note)
            np.testing.assert_allclose(np.asarray(got, dtype=np.float64), want,
                                       rtol=0, atol=1e-6)

    def test_donner_group_dff_is_the_file_on_disk(self) -> None:
        from asvimg.annotation import _load_name_dff_stack

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            out.mkdir()
            _fabricate_preprocess_outputs(out)
            cfg = _make_4ch_config(Path(td) / "in", out)

            got, note = _load_name_dff_stack("GCaMP", cfg.channel_groups()["GCaMP"], cfg, out)
            # copy out and drop the view: it is memmap-backed, and Windows will
            # not remove the temp dir while the mapping is open
            got = np.array(got)
            want = np.moveaxis(np.load(dff_name_path(out, "GCaMP")), 0, 2)
            self.assertIn("linear subt", note)
            np.testing.assert_array_equal(got, want)

    def test_roi_signals_come_from_that_stack(self) -> None:
        """ROI dF/F == ROI means of the group stack. Pins the whole chain without
        needing the atlas: the adjoint is checked elsewhere, here we only pin that
        ROI reads the SAME dF/F the definition above produces."""
        from asvimg.annotation import _load_name_dff_stack
        from asvimg.extraction import extract_signals_from_source
        from skimage.transform import SimilarityTransform

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            out.mkdir()
            _fabricate_preprocess_outputs(out)
            cfg = _make_4ch_config(Path(td) / "in", out)
            tform = SimilarityTransform(scale=1.2, rotation=0.1, translation=(2, 1))

            for name in ("GCaMP", "jRGECO"):
                stack, _ = _load_name_dff_stack(name, cfg.channel_groups()[name], cfg, out)
                h, w, t = stack.shape
                masks = np.zeros((2, 30, 34), bool)
                masks[0, 8:14, 8:14] = True
                masks[1, 15:20, 18:24] = True
                got = extract_signals_from_source(
                    lambda t0, t1, _s=stack: np.moveaxis(np.asarray(_s[..., t0:t1]), 2, 0),
                    t, (h, w), masks, ["a", "b"], transform=tform,
                ).F
                self.assertEqual(got.shape, (2, t))
                self.assertTrue(np.isfinite(got).all())
                self.assertGreater(np.abs(got).max(), 0)  # the ROIs see signal


class TestIcaOrderingIsSmoothingInvariant(unittest.TestCase):
    """A saved "exclude IC3" must keep meaning the same component.

    pca_smooth_sigma low-passes the temporal scores for the PLOTS. Ordering the
    ICs by the variance of those smoothed scores made the IC numbering depend on
    a display knob: identical maps, permuted numbers. Any persisted exclusion
    would then silently point at a different component.
    """

    def _decompose(self, X, sigma, K=6):
        from asvimg.ica import compute_ica
        from asvimg.pca import compute_pca

        pca = compute_pca(X, n_components=K, smooth_sigma=sigma)
        return compute_ica(pca, n_ica_components=K, random_state=0, max_iter=800)

    def _data(self, H=16, W=18, T=200):
        rng = np.random.default_rng(0)
        t = np.arange(T)
        X = sum(
            rng.normal(size=(H, W))[:, :, None] * np.sin(t / (3 + 4 * i))[None, None, :]
            for i in range(4)
        )
        return X + rng.normal(0, 0.3, size=(H, W, T))

    def test_ic_numbering_does_not_depend_on_smooth_sigma(self) -> None:
        X = self._data()
        a = self._decompose(X, 0.0)
        b = self._decompose(X, 5.0)  # the config default
        K = a.n_components
        corr = np.abs(
            np.corrcoef(a.spatial.reshape(K, -1), b.spatial.reshape(K, -1))[:K, K:]
        )
        # same maps...
        np.testing.assert_allclose(corr.max(axis=1), np.ones(K), atol=1e-6)
        # ...under the same numbers
        np.testing.assert_array_equal(corr.argmax(axis=1), np.arange(K))

    def test_scores_still_project_the_data(self) -> None:
        """SCORE == pinv(ModeICA) @ (X - mean), even at the default smoothing.

        This is what lets the denoising be applied to the FULL timeline: the
        basis is spatial, so the loadings can be recomputed at every frame. With
        the smoothed scores fed into ICA this identity did not hold."""
        X = self._data()
        for sigma in (0.0, 5.0):
            r = self._decompose(X, sigma)
            K = r.n_components
            mode = r.spatial.reshape(K, -1).T                       # (P, K)
            centered = X.reshape(mode.shape[0], -1) - r.mean_image.reshape(-1, 1)
            np.testing.assert_allclose(
                r.temporal, np.linalg.pinv(mode) @ centered, rtol=0, atol=1e-8
            )


class TestPcaSourceIsTheGroupDff(unittest.TestCase):
    """PCA/ICA must analyse the SAME dF/F as ROI and the exports.

    It used to address the stack by channel index and fall back to the RAW
    registered channel whenever the name had no donner — so the components the
    user was shown, and asked to exclude, were the components of a different
    array (raw fluorescence, ~1e3 counts) than the one everything else analysed
    (dF/F, ~1e-2).
    """

    def test_donnerless_group_pca_reads_dff_not_raw(self) -> None:
        from asvimg.annotation import _load_name_dff_stack
        from asvimg.pca import load_annotation_source, load_group_source

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            out.mkdir()
            _fabricate_preprocess_outputs(out)
            cfg = _make_4ch_config(Path(td) / "in", out)
            cfg.ch_for_annotation = 1          # jRGECO: NO donner, TWO sources
            cfg.pca_skip_frames = 1

            want, _ = _load_name_dff_stack("jRGECO", cfg.channel_groups()["jRGECO"], cfg, out)
            want = np.asarray(want, dtype=np.float64)

            got, key, name = load_annotation_source(cfg, out)
            self.assertEqual(name, "jRGECO")
            self.assertEqual(key, "imageDf")               # not imageCh1
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)

            # and it is dF/F, not raw counts (which is what the bug produced)
            self.assertLess(np.abs(got).max(), 100.0)
            raw = np.load(reg_channel_path(out, 1))
            self.assertGreater(raw.mean(), 100.0)          # the raw stack it used to read

            # every group is addressable, not just ch_for_annotation's
            for group in ("GCaMP", "jRGECO"):
                stack, note = load_group_source(cfg, out, group)
                self.assertEqual(stack.shape[:2], want.shape[:2])
                self.assertIn("dF/F", note)

    def test_pca_skip_frames_still_strides(self) -> None:
        from asvimg.pca import load_group_source

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            out.mkdir()
            _fabricate_preprocess_outputs(out, T=20)
            cfg = _make_4ch_config(Path(td) / "in", out)
            cfg.pca_skip_frames = 4
            stack, _ = load_group_source(cfg, out, "GCaMP")
            self.assertEqual(stack.shape[2], len(range(0, 20, 4)))


@unittest.skipUnless(_ATLAS.exists(), f"atlas not found at {_ATLAS}")
class TestSessionStages2to5(unittest.TestCase):
    def test_full_downstream(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            inp = tmp / "input"
            inp.mkdir()
            out = tmp / "out"
            out.mkdir()
            _fabricate_preprocess_outputs(out)
            cfg = _make_4ch_config(inp, out)

            reporter = RecordingReporter()
            session = PipelineSession(
                cfg,
                reporter=reporter,
                marks_provider=FakeMarksProvider(
                    [[5, 5], [5, 22], [18, 14]],
                    [[100, 100], [100, 180], [200, 140]],
                ),
                ica_provider=FakeIcaProvider([]),
            )

            session.run_pca()
            session.run_ica()
            tform = session.run_annotation()
            self.assertIsNotNone(tform)
            session.run_export()
            roi = session.run_roi()
            corr = session.run_correlation()

            produced = {p.name for p in out.iterdir()}
            self.assertTrue(
                any(p.startswith("roiSignals_GCaMP") and p.endswith(".npy")
                    for p in produced)
            )
            self.assertTrue(
                any(p.startswith("roiSignals_jRGECO") and p.endswith(".npy")
                    for p in produced)
            )
            self.assertTrue(
                any("dfWarped_GCaMP" in p for p in produced)
            )
            self.assertTrue(any(p.endswith(".csv") for p in produced))

            # GCaMP has a donner → both raw + dff; jRGECO → raw + p5-baseline dff
            self.assertIn("F_raw", roi["GCaMP"])
            self.assertIn("F_dff", roi["GCaMP"])
            self.assertEqual(roi["GCaMP"]["F_dff"].shape[1], 20)

            cG = corr["GCaMP"]
            self.assertEqual(cG.C.shape[0], cG.C.shape[1])
            self.assertEqual(cG.C.shape[0], len(cG.roi_names))
            self.assertEqual(cG.mode, "dff")

            # Every stage reported done, no failures.
            done = {(s.stage, str(s.status)) for s in reporter.stages}
            for stage in ("pca", "ica", "annotation", "roi", "correlation", "export"):
                self.assertIn((stage, "done"), done)

    def test_downstream_spans_full_timeline(self) -> None:
        """ROI signals must span the whole recording: F_dff/F_raw length equals
        the reg_Ch/dff frame count (donner + no-donner paths)."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            inp = tmp / "input"
            inp.mkdir()
            out = tmp / "out"
            out.mkdir()
            total_T = 33
            _fabricate_preprocess_outputs(out, T=total_T)
            cfg = _make_4ch_config(inp, out)

            session = PipelineSession(
                cfg,
                marks_provider=FakeMarksProvider(
                    [[5, 5], [5, 22], [18, 14]],
                    [[100, 100], [100, 180], [200, 140]],
                ),
                ica_provider=FakeIcaProvider([]),
            )
            session.run_pca()
            session.run_ica()
            self.assertIsNotNone(session.run_annotation())
            roi = session.run_roi()

            # GCaMP: donner path (dff_GCaMP.npy); jRGECO: no-donner p5-baseline
            # path (reg_Ch source channels). Both must be full length.
            self.assertEqual(roi["GCaMP"]["F_dff"].shape[1], total_T)
            self.assertEqual(roi["GCaMP"]["F_raw"].shape[1], total_T)
            self.assertEqual(roi["jRGECO"]["F_dff"].shape[1], total_T)

    def test_pca_skip_frames_subsamples_fit_input(self) -> None:
        """pca_skip_frames strides the PCA/ICA fitting input (fit-only)."""
        import math

        from asvimg import load_annotation_source

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            out = tmp / "out"
            out.mkdir()
            _fabricate_preprocess_outputs(out, T=40)
            cfg = _make_4ch_config(tmp / "in", out)
            cfg.pca_skip_frames = 5
            image_df, _key, name = load_annotation_source(cfg, out)
            self.assertEqual(name, "GCaMP")  # ch_for_annotation=0, has donner
            self.assertEqual(image_df.shape[2], math.ceil(40 / 5))  # 8 frames

    def test_ica_runs_on_pca_skip_subsample(self) -> None:
        """ICA (like PCA) fits on the pca_skip_frames-thinned frame count:
        load_annotation_source strides once and PCA + ICA share it."""
        import math

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            out = tmp / "out"
            out.mkdir()
            T = 40
            _fabricate_preprocess_outputs(out, T=T)
            cfg = _make_4ch_config(tmp / "in", out)
            cfg.pca_skip_frames = 5
            cfg.skip_ica = False
            cfg.pca_n_components = 3
            session = PipelineSession(cfg, ica_provider=FakeIcaProvider([0]))
            session.run_pca()
            # run_ica -> compute_ica(pca_result) + reconstruct(image_df); reconstruct
            # raises if the ICA and image_df frame counts disagree.
            session.run_ica()

            t_sub = math.ceil(T / 5)  # 8
            self.assertEqual(session.state.image_df.shape[2], t_sub)
            self.assertEqual(session.state.pca_result.temporal.shape[1], t_sub)
            self.assertEqual(session.state.denoised_df.shape[2], t_sub)

    def test_save_roi_signals_both_writes_csv_and_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            inp = tmp / "input"
            inp.mkdir()
            out = tmp / "out"
            out.mkdir()
            _fabricate_preprocess_outputs(out)
            cfg = _make_4ch_config(inp, out)
            cfg.save_roi_signals = "both"

            session = PipelineSession(
                cfg,
                marks_provider=FakeMarksProvider(
                    [[5, 5], [5, 22], [18, 14]],
                    [[100, 100], [100, 180], [200, 140]],
                ),
            )
            session.run_annotation()
            session.run_roi()

            csvs = sorted(out.glob("roiSignals_*.csv"))
            pkls = sorted(out.glob("roiSignals_*.pkl"))
            self.assertTrue(csvs)
            self.assertEqual(len(csvs), len(pkls))  # both formats, same count

    def test_no_marks_skips_downstream(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            inp = tmp / "input"
            inp.mkdir()
            out = tmp / "out"
            out.mkdir()
            _fabricate_preprocess_outputs(out)
            cfg = _make_4ch_config(inp, out)
            cfg.annotation = False  # ConfigMarksProvider → (None, None)

            session = PipelineSession(cfg)  # headless defaults
            session.run_pca()
            session.run_ica()
            tform = session.run_annotation()
            self.assertIsNone(tform)
            self.assertEqual(session.run_roi(), {})
            self.assertEqual(session.run_correlation(), {})


class TestReviewFixes(unittest.TestCase):
    """Regression tests for issues found in the adversarial review."""

    def test_save_payload_preserves_dotted_stem(self) -> None:
        # OME-TIFF stems contain a dot ('run.ome'); with_suffix would truncate.
        from asvimg import load_payload, save_payload

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            stem = out / "roiSignals_GCaMP_run.ome_01"
            save_payload(stem, {"v": 7}, "npy")
            written = list(out.glob("*.npy"))
            self.assertEqual(len(written), 1)
            # full stem retained, single .npy suffix
            self.assertEqual(written[0].name, "roiSignals_GCaMP_run.ome_01.npy")
            self.assertEqual(load_payload(written[0])["v"], 7)

    def test_correlation_owning_group_disambiguation(self) -> None:
        from asvimg.correlation import (
            _owning_group,
            _resolve_roi_signal_file,
        )

        names = ["GCaMP", "jRGECO", "GCaMP_GFAP"]
        self.assertEqual(_owning_group("GCaMP_GFAP_run01", names), "GCaMP_GFAP")
        self.assertEqual(_owning_group("GCaMP_run01", names), "GCaMP")
        self.assertIsNone(_owning_group("Other_run01", names))

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            (out / "roiSignals_GCaMP_run01.npy").touch()
            (out / "roiSignals_GCaMP_GFAP_run01.npy").touch()
            # GCaMP must NOT pick the GCaMP_GFAP sibling
            picked = _resolve_roi_signal_file(out, "GCaMP", "run01", "npy", names)
            self.assertEqual(picked.name, "roiSignals_GCaMP_run01.npy")
            picked2 = _resolve_roi_signal_file(out, "GCaMP_GFAP", "run01", "npy", names)
            self.assertEqual(picked2.name, "roiSignals_GCaMP_GFAP_run01.npy")

    def test_run_ica_autoruns_pca_in_fresh_session(self) -> None:
        # GUI runs each stage button in a fresh session; clicking ICA without
        # PCA first must still work (PCA reads only on-disk outputs).
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            out = tmp / "out"
            out.mkdir()
            _fabricate_preprocess_outputs(out)
            cfg = _make_4ch_config(tmp / "in", out)

            rep = RecordingReporter()
            session = PipelineSession(cfg, reporter=rep, ica_provider=FakeIcaProvider([]))
            session.run_ica()  # no run_pca() beforehand

            statuses = {(s.stage, str(s.status)) for s in rep.stages}
            self.assertIn(("pca", "done"), statuses)
            self.assertIn(("ica", "done"), statuses)
            self.assertIsNotNone(session.state.pca_result)

    @unittest.skipUnless(_ATLAS.exists(), f"atlas not found at {_ATLAS}")
    def test_annotation_raw_only_without_pca_ica(self) -> None:
        # No PCA/ICA data on disk or in memory -> cpselect gets the raw frame
        # only, and annotation must NOT run PCA just to produce maps.
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            out = tmp / "out"
            out.mkdir()
            _fabricate_preprocess_outputs(out)
            cfg = _make_4ch_config(tmp / "in", out)

            recorded: dict = {}

            class RecordingMarks:
                def request(self, config, output_dir, atlas, source_img, *,
                            extra_imgs=None, extra_labels=None):
                    recorded["imgs"] = extra_imgs
                    return None, None

            rep = RecordingReporter()
            session = PipelineSession(cfg, reporter=rep, marks_provider=RecordingMarks())
            session.run_annotation()

            statuses = {(s.stage, str(s.status)) for s in rep.stages}
            self.assertNotIn(("pca", "done"), statuses)  # PCA not forced
            self.assertIsNone(recorded.get("imgs"))      # raw frame only

    @unittest.skipUnless(_ATLAS.exists(), f"atlas not found at {_ATLAS}")
    def test_annotation_loads_pca_and_ica_maps_from_disk(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            out = tmp / "out"
            out.mkdir()
            _fabricate_preprocess_outputs(out)
            cfg = _make_4ch_config(tmp / "in", out)

            # PCA/ICA maps left on disk by a prior run (under output_dir)
            (out / "pca_images").mkdir(parents=True)
            (out / "ica_images").mkdir(parents=True)
            rng = np.random.default_rng(0)
            for i in (1, 2):
                Image.fromarray(rng.integers(0, 255, (10, 10, 3), np.uint8)).save(
                    out / "pca_images" / f"PC{i}.png"
                )
            Image.fromarray(rng.integers(0, 255, (10, 10, 3), np.uint8)).save(
                out / "ica_images" / "IC1.png"
            )

            recorded: dict = {}

            class RecordingMarks:
                def request(self, config, output_dir, atlas, source_img, *,
                            extra_imgs=None, extra_labels=None):
                    recorded["imgs"] = extra_imgs
                    recorded["labels"] = extra_labels
                    return None, None

            session = PipelineSession(cfg, marks_provider=RecordingMarks())
            session.run_annotation()

            self.assertEqual(len(recorded["imgs"]), 3)  # PC1, PC2, IC1
            self.assertEqual(recorded["labels"], ["PC1", "PC2", "IC1"])

    @unittest.skipUnless(_ATLAS.exists(), f"atlas not found at {_ATLAS}")
    def test_export_rehydrates_transform_from_disk(self) -> None:
        # GUI runs each stage button in a *fresh* PipelineSession, so the
        # in-memory transform from run_annotation is gone by the time Export is
        # pressed.  Export/ROI must recover it from the saved marks.mat instead
        # of skipping with "no transform".
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            inp = tmp / "input"
            inp.mkdir()
            out = tmp / "out"
            out.mkdir()
            _fabricate_preprocess_outputs(out)
            cfg = _make_4ch_config(inp, out)

            # Session A: annotation only — writes marks.mat, then is discarded.
            session_a = PipelineSession(
                cfg,
                marks_provider=FakeMarksProvider(
                    [[5, 5], [5, 22], [18, 14]],
                    [[100, 100], [100, 180], [200, 140]],
                ),
            )
            self.assertIsNotNone(session_a.run_annotation())
            self.assertTrue((out / "marks.mat").exists())
            del session_a

            # Session B: fresh, no marks_provider — export/roi must NOT skip.
            reporter = RecordingReporter()
            session_b = PipelineSession(cfg, reporter=reporter)
            self.assertIsNone(session_b.state.tform)  # nothing in memory yet
            session_b.run_export()
            roi = session_b.run_roi()

            self.assertIsNotNone(session_b.state.tform)  # rehydrated from disk
            self.assertIn("GCaMP", roi)
            statuses = {(s.stage, str(s.status)) for s in reporter.stages}
            self.assertIn(("export", "done"), statuses)
            self.assertIn(("roi", "done"), statuses)
            self.assertNotIn(("export", "skipped"), statuses)

    @unittest.skipUnless(_ATLAS.exists(), f"atlas not found at {_ATLAS}")
    def test_skipped_downstream_reports_skipped_not_done(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            inp = tmp / "input"
            inp.mkdir()
            out = tmp / "out"
            out.mkdir()
            _fabricate_preprocess_outputs(out)
            cfg = _make_4ch_config(inp, out)
            cfg.annotation = False  # → no marks → downstream skipped

            reporter = RecordingReporter()
            session = PipelineSession(cfg, reporter=reporter)
            session.run_pca()
            session.run_ica()
            session.run_annotation()
            session.run_roi()
            session.run_correlation()

            statuses = {(s.stage, str(s.status)) for s in reporter.stages}
            self.assertIn(("annotation", "skipped"), statuses)
            self.assertIn(("roi", "skipped"), statuses)
            self.assertIn(("correlation", "skipped"), statuses)
            self.assertNotIn(("roi", "done"), statuses)


class TestMultiFile(unittest.TestCase):
    def test_find_input_files_natural_order(self) -> None:
        from asvimg import find_input_files

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            for i in (1, 2, 9, 10, 11, 100):
                (d / f"rec_{i}.tif").touch()
            names = [p.name for p in find_input_files(d)]
            self.assertEqual(
                names,
                ["rec_1.tif", "rec_2.tif", "rec_9.tif", "rec_10.tif",
                 "rec_11.tif", "rec_100.tif"],
            )

    def test_find_input_files_name_order(self) -> None:
        """'name' is the plain lexicographic sort a file browser shows —
        rec_10 before rec_2 — as opposed to natural order."""
        from asvimg import find_input_files

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            for i in (1, 2, 9, 10, 11, 100):
                (d / f"rec_{i}.tif").touch()
            names = [p.name for p in find_input_files(d, "auto", "name")]
            self.assertEqual(
                names,
                ["rec_1.tif", "rec_10.tif", "rec_100.tif", "rec_11.tif",
                 "rec_2.tif", "rec_9.tif"],
            )

    def test_find_input_files_time_orders(self) -> None:
        """mtime order is by modification time, independent of the name — the
        case that matters for a dataset whose alphabetical order does not match
        the acquisition sequence."""
        import os

        from asvimg import find_input_files

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            # Written in an order that is neither natural nor lexicographic.
            for offset, name in ((300, "b_rec_1.tif"), (100, "c_rec_2.tif"),
                                 (200, "a_rec_3.tif")):
                p = d / name
                p.touch()
                os.utime(p, (offset, offset))

            self.assertEqual(
                [p.name for p in find_input_files(d, "auto", "mtime")],
                ["c_rec_2.tif", "a_rec_3.tif", "b_rec_1.tif"],
            )
            # ...and the other orders are unaffected by those timestamps.
            self.assertEqual(
                [p.name for p in find_input_files(d, "auto", "natural")],
                ["a_rec_3.tif", "b_rec_1.tif", "c_rec_2.tif"],
            )

    def test_find_input_files_time_order_ties_break_on_name(self) -> None:
        """Equal timestamps (a whole batch copied at once, or a coarse
        filesystem clock) must still give a deterministic order."""
        import os

        from asvimg import find_input_files

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            for i in (3, 1, 10, 2):
                p = d / f"rec_{i}.tif"
                p.touch()
                os.utime(p, (500, 500))
            # ...and the tie-break is natural, not lexicographic: an all-tied
            # batch must degrade to exactly the default order, not to rec_10
            # before rec_2.  Needs a cross-decade name to be meaningful — with
            # only rec_1..rec_3 the two orders coincide and the test proves
            # nothing.
            self.assertEqual(
                [p.name for p in find_input_files(d, "auto", "mtime")],
                ["rec_1.tif", "rec_2.tif", "rec_3.tif", "rec_10.tif"],
            )
            self.assertEqual(
                [p.name for p in find_input_files(d, "auto", "mtime")],
                [p.name for p in find_input_files(d, "auto", "natural")],
            )

    def test_time_order_survives_coarse_timestamp_granularity(self) -> None:
        """Partial ties (exFAT/FAT report mtime to 2 s, so parts written a
        second apart collapse pairwise) must keep the real time order for every
        untied file and fall back to natural order only within a bucket."""
        import os

        from asvimg import find_input_files

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            for i in range(1, 13):
                p = d / f"rec_{i}.tif"
                p.touch()
                bucket = 1000 + (i // 2) * 2
                os.utime(p, (bucket, bucket))
            self.assertEqual(
                [p.name for p in find_input_files(d, "auto", "mtime")],
                [f"rec_{i}.tif" for i in range(1, 13)],
            )

    def test_find_input_files_default_order_is_natural(self) -> None:
        """The default must stay natural order: it is what every existing
        output was produced with."""
        from asvimg import find_input_files
        from asvimg.config import PipelineConfig

        self.assertEqual(PipelineConfig().input_order, "natural")
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            for i in (1, 2, 10):
                (d / f"rec_{i}.tif").touch()
            self.assertEqual(
                [p.name for p in find_input_files(d)],
                [p.name for p in find_input_files(d, "auto", "natural")],
            )

    def test_invalid_input_order_rejected(self) -> None:
        from asvimg import find_input_files
        from asvimg.config import PipelineConfig

        with self.assertRaises(ValueError):
            PipelineConfig(input_order="creation")
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "rec_1.tif").touch()
            with self.assertRaises(ValueError):
                find_input_files(d, "auto", "creation")

    def test_input_order_changes_exp_stem(self) -> None:
        """input_order picks which file is 'first', so it also names the
        experiment — the surprising-but-intended side effect."""
        import os

        from asvimg import resolve_exp_stem

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            for offset, name in ((900, "a_first_by_name.tif"),
                                 (100, "z_first_by_time.tif")):
                p = d / name
                p.touch()
                os.utime(p, (offset, offset))
            self.assertEqual(resolve_exp_stem(d, None, "auto", "natural"),
                             "a_first_by_name")
            self.assertEqual(resolve_exp_stem(d, None, "auto", "mtime"),
                             "z_first_by_time")

    def test_channel_misalignment_warning_on_non_multiple_file(self) -> None:
        import tifffile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            inp = tmp / "input"
            inp.mkdir()
            out = tmp / "out"
            out.mkdir()
            rng = np.random.default_rng(2)
            # cycle_len=2; first (non-last) file has an ODD frame count -> warn.
            # photometric='minisblack' forces a grayscale page stack (else
            # tifffile would treat a small leading axis as RGB samples).
            tifffile.imwrite(
                inp / "seg_1.tif",
                rng.integers(200, 3000, size=(5, 12, 12)).astype(np.uint16),
                photometric="minisblack",
            )
            tifffile.imwrite(
                inp / "seg_2.tif",
                rng.integers(200, 3000, size=(2, 12, 12)).astype(np.uint16),
                photometric="minisblack",
            )
            cfg = PipelineConfig(
                input_dir=str(inp), output_dir=str(out), output_format="npy",
                exp_name="MF", channels_name=["BL", "BL"],
                channels_prop=["source", "donner"], do_registration=True,
                usfac=2, binning=1, fps=20, template_stride=2,
            )
            rep = RecordingReporter()
            PreprocessRunner(cfg).run(reporter=rep)
            warned = [l.text for l in rep.logs if "not a multiple" in l.text]
            self.assertTrue(warned, msg=[l.text for l in rep.logs])
            self.assertIn("seg_1.tif", warned[0])


class TestPreprocessHooks(unittest.TestCase):
    def _make_config(self, inp: Path, out: Path) -> PipelineConfig:
        return PipelineConfig(
            input_dir=str(inp),
            output_dir=str(out),
            output_format="npy",
            exp_name="S1",
            channels_name=["BL", "BL"],
            channels_prop=["source", "donner"],
            do_registration=True,
            usfac=5,
            binning=2,
            linear_subt=True,
            fps=20,
            template_stride=4,
        )

    def _write_tiff(self, inp: Path, T=24, H=16, W=20) -> None:
        import tifffile

        stack = np.random.default_rng(1).integers(
            200, 3000, size=(T, H, W)
        ).astype(np.uint16)
        tifffile.imwrite(inp / "timelapseS1_MM.tif", stack)

    def test_reporter_receives_per_frame_progress(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            inp = tmp / "input"
            inp.mkdir()
            out = tmp / "out"
            out.mkdir()
            self._write_tiff(inp)
            rep = RecordingReporter()
            stats = PreprocessRunner(self._make_config(inp, out)).run(reporter=rep)

            self.assertEqual(stats.total_frames, 24)
            # preprocess reports two phases with different units: registration in
            # frames, then dF/F in image rows. Each must reach its own total.
            reg = [p for p in rep.progress if "dF/F" not in p.message]
            dff = [p for p in rep.progress if "dF/F" in p.message]
            self.assertGreaterEqual(len(reg), 24)
            self.assertEqual(reg[-1].current, 24)  # every frame registered
            self.assertTrue(dff)
            self.assertEqual(dff[-1].current, dff[-1].total)  # dF/F reached 100%
            self.assertTrue(sorted(out.glob("reg_Ch*.npy")))
            self.assertTrue(sorted(out.glob("dff_*.npy")))
            # logs were routed to the reporter, not stdout
            self.assertTrue(any("Creating template" in l.text for l in rep.logs))

    def test_filter_xyt_rewrites_reg_channels(self) -> None:
        """filter_xyt smooths each reg_Ch in place, reopening the .npy with
        mode='w+'.  On Windows, truncating a file that still has an open memory
        mapping raises OSError EINVAL, so the reg-write memmaps must be closed
        before this pass — regression guard for that."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            inp = tmp / "input"
            inp.mkdir()
            out = tmp / "out"
            out.mkdir()
            self._write_tiff(inp, T=24, H=16, W=20)
            cfg = self._make_config(inp, out)
            cfg.filter_xyt = [3, 3, 3]
            PreprocessRunner(cfg).run()  # must not raise on the w+ reopen
            for ch in (0, 1):
                p = out / f"reg_Ch{ch}.npy"
                self.assertTrue(p.exists())
                self.assertEqual(np.load(p).shape[0], 12)  # 24 frames / 2 ch

    def test_do_registration_false_skips_registration(self) -> None:
        """do_registration=False must NOT call register_single (only the log
        label changed before); the run still produces reg_Ch outputs."""
        from unittest import mock

        from asvimg import DftRegistrator

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            inp = tmp / "input"
            inp.mkdir()
            out = tmp / "out"
            out.mkdir()
            self._write_tiff(inp, T=8)
            cfg = self._make_config(inp, out)
            cfg.do_registration = False
            with mock.patch.object(DftRegistrator, "register_batch") as m:
                PreprocessRunner(cfg).run()
            m.assert_not_called()
            self.assertTrue((out / "reg_Ch0.npy").exists())

    def test_use_mmap_dff_matches_in_ram(self) -> None:
        """The row-strip streamed dF/F (use_mmap=True) must produce byte-for-
        byte the same dff_{name}.npy as the in-RAM path (use_mmap=False)."""
        from asvimg import dff_name_path

        def _run(use_mmap: bool) -> np.ndarray:
            tmp = Path(tempfile.mkdtemp())
            inp = tmp / "input"
            inp.mkdir()
            out = tmp / "out"
            out.mkdir()
            self._write_tiff(inp, T=24)  # BL/BL -> 12 frames/channel
            cfg = self._make_config(inp, out)
            cfg.use_mmap = use_mmap
            PreprocessRunner(cfg).run()
            return np.load(dff_name_path(out, "BL"))

        np.testing.assert_array_equal(_run(False), _run(True))

    def test_registration_batch_size_invariant(self) -> None:
        """registration_batch_size changes only batching, not the result:
        batch=1 and batch>1 must produce identical reg_Ch output."""
        def _run(bs: int) -> np.ndarray:
            tmp = Path(tempfile.mkdtemp())
            inp = tmp / "input"
            inp.mkdir()
            out = tmp / "out"
            out.mkdir()
            self._write_tiff(inp, T=24)
            cfg = self._make_config(inp, out)
            cfg.registration_batch_size = bs
            PreprocessRunner(cfg).run()
            return load_reg_channel(out, 0)

        np.testing.assert_array_equal(_run(1), _run(7))

    def test_continuous_reg_ch_per_channel(self) -> None:
        """Each channel is one continuous (T,H,W) reg_Ch file (+ dff) under
        either dF/F read strategy."""
        for use_mmap in (False, True):
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                inp = tmp / "input"
                inp.mkdir()
                out = tmp / "out"
                out.mkdir()
                self._write_tiff(inp, T=24)  # cycle_len=2 -> 12 frames/channel
                cfg = self._make_config(inp, out)
                cfg.use_mmap = use_mmap
                PreprocessRunner(cfg).run()

                self.assertTrue((out / "reg_Ch0.npy").exists())
                self.assertTrue((out / "reg_Ch1.npy").exists())
                self.assertEqual(load_reg_channel(out, 0).shape[0], 12, msg=f"use_mmap={use_mmap}")
                self.assertTrue(sorted(out.glob("dff_*.npy")))

    def test_template_cached_under_output_not_input(self) -> None:
        """The auto-template is written under output_dir, not the raw source dir."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            inp = tmp / "input"
            inp.mkdir()
            out = tmp / "out"
            out.mkdir()
            self._write_tiff(inp, T=12)
            PreprocessRunner(self._make_config(inp, out)).run()
            self.assertTrue((out / "templateImage.mat").exists())
            self.assertFalse((inp / "templateImage.mat").exists())

    def test_registration_cache_skips_when_output_exists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            inp = tmp / "input"
            inp.mkdir()
            out = tmp / "out"
            out.mkdir()
            self._write_tiff(inp)
            cfg = self._make_config(inp, out)
            cfg.registration_cache = "cached"

            # first run: no cache yet -> computes and writes reg_Ch*.npy
            s1 = PreprocessRunner(cfg).run()
            self.assertFalse(s1.cached)
            self.assertTrue(sorted(out.glob("reg_Ch*.npy")))

            # second run: output exists -> skip registration entirely
            rep = RecordingReporter()
            s2 = PreprocessRunner(cfg).run(reporter=rep)
            self.assertTrue(s2.cached)
            self.assertFalse(
                any("Creating template" in l.text for l in rep.logs)
            )

    def test_cached_run_still_writes_registered_tiffs(self) -> None:
        """The registered TIFFs are written from reg_Ch*.npy -- which is exactly
        what a cached run reuses -- so ticking save_registered_each_ch must not
        need a re-registration. Raw TIFFs still do (the frames are only seen
        while the input is read), and say so."""
        import tifffile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            inp, out = tmp / "input", tmp / "out"
            inp.mkdir()
            out.mkdir()
            self._write_tiff(inp)
            cfg = self._make_config(inp, out)
            cfg.registration_cache = "cached"
            PreprocessRunner(cfg).run()
            self.assertFalse((out / "tiffs").exists())

            cfg.save_registered_each_ch = True
            cfg.save_raw_each_ch = True
            rep = RecordingReporter()
            s2 = PreprocessRunner(cfg).run(reporter=rep)
            self.assertTrue(s2.cached)
            tifs = sorted((out / "tiffs").glob("reg_*/*.tif"))
            self.assertEqual(len(tifs), 2)  # one per channel of the cycle
            for ch, path in enumerate(tifs):
                np.testing.assert_array_equal(
                    tifffile.imread(path), np.asarray(load_reg_channel(out, ch))
                )
            self.assertFalse(list((out / "tiffs").glob("raw_*")))
            self.assertTrue(any("save_raw_each_ch is IGNORED" in l.text for l in rep.logs))

    def test_cancellation_stops_partway(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            inp = tmp / "input"
            inp.mkdir()
            out = tmp / "out"
            out.mkdir()
            self._write_tiff(inp)
            token = CancellationToken()

            class CancelAt:
                def on_stage(self, *a, **k):
                    pass

                def on_progress(self, stage, current, total, *, message=""):
                    if current >= 8:
                        token.cancel()

                def on_log(self, *a, **k):
                    pass

                def on_figure(self, *a, **k):
                    pass

            cfg = self._make_config(inp, out)
            cfg.registration_batch_size = 4  # flush often so progress fires mid-run
            stats = PreprocessRunner(cfg).run(reporter=CancelAt(), cancel=token)
            self.assertGreater(stats.total_frames, 0)
            self.assertLess(stats.total_frames, 24)


class TestTransformReflection(unittest.TestCase):
    """The SVD (Umeyama) transform and its reflection constraint."""

    @staticmethod
    def _make_pairs(scale, rot_deg, reflect=False, seed=0):
        rng = np.random.default_rng(seed)
        src_xy = rng.uniform(20, 260, size=(6, 2))
        a = np.radians(rot_deg)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        ref_xy = scale * (src_xy @ R.T) + np.array([12.0, -7.0])
        if reflect:
            ref_xy[:, 0] = -ref_xy[:, 0]
        # compute_transform takes (row, col); our arrays are (x, y) = (col, row)
        return src_xy[:, ::-1], ref_xy[:, ::-1], src_xy, ref_xy

    def test_nonreflective_recovers_scale_and_rotation(self) -> None:
        from asvimg import compute_transform
        from skimage.transform import SimilarityTransform

        src, ref, src_xy, ref_xy = self._make_pairs(1.5, 30.0)
        t = compute_transform(src, ref)
        self.assertIsInstance(t, SimilarityTransform)
        self.assertAlmostEqual(t.scale, 1.5, places=4)
        self.assertAlmostEqual(np.degrees(t.rotation), 30.0, places=3)
        resid = np.sqrt(((t(src_xy) - ref_xy) ** 2).sum(1).mean())
        self.assertLess(resid, 1e-6)

    def test_reflected_data_needs_allow_reflection(self) -> None:
        from asvimg import compute_transform
        from skimage.transform import AffineTransform

        src, ref, src_xy, ref_xy = self._make_pairs(1.5, 30.0, reflect=True)

        # Default (non-reflective) cannot fit a mirror -> large residual.
        t_nr = compute_transform(src, ref)
        resid_nr = np.sqrt(((t_nr(src_xy) - ref_xy) ** 2).sum(1).mean())
        self.assertGreater(resid_nr, 10.0)
        self.assertGreaterEqual(np.linalg.det(np.asarray(t_nr.params)[:2, :2]), 0)

        # Allowing reflection recovers it exactly, with det < 0.
        t_r = compute_transform(src, ref, allow_reflection=True)
        self.assertIsInstance(t_r, AffineTransform)
        resid_r = np.sqrt(((t_r(src_xy) - ref_xy) ** 2).sum(1).mean())
        self.assertLess(resid_r, 1e-6)
        self.assertLess(np.linalg.det(np.asarray(t_r.params)[:2, :2]), 0)

    @unittest.skipUnless(_ATLAS.exists(), f"atlas not found at {_ATLAS}")
    def test_transform_overlay_has_four_panels(self) -> None:
        from asvimg import compute_transform
        from asvimg.atlas import ACCFv3
        from asvimg.runner.session import PipelineSession

        atlas = ACCFv3.from_mat(str(_ATLAS))
        src, ref, *_ = self._make_pairs(1.2, 3.0)
        tform = compute_transform(src, ref)
        source_img = np.random.default_rng(0).random((40, 44))
        session = PipelineSession(
            PipelineConfig(channels_name=["G", "R"], channels_prop=["source", "donner"])
        )
        fig = session._plot_transform_overlay(source_img, tform, atlas)
        self.assertEqual(len(fig.axes), 4)  # 1 source, 2 warped, 3 +border, 4 src+border

    def test_describe_tform_flags_reflection(self) -> None:
        from asvimg import compute_transform
        from asvimg.runner.session import PipelineSession

        src, ref, *_ = self._make_pairs(1.5, 30.0, reflect=True)
        t_r = compute_transform(src, ref, allow_reflection=True)
        self.assertIn("REFLECTED", PipelineSession._describe_tform(t_r, 6))
        t_nr = compute_transform(src, ref)
        self.assertNotIn("REFLECTED", PipelineSession._describe_tform(t_nr, 6))


class TestCorrelationMethods(unittest.TestCase):
    """raw / gsr / partial correlation options for the ROI network."""

    @staticmethod
    def _signals_with_global(n_roi=8, T=400, seed=0):
        rng = np.random.default_rng(seed)
        g = rng.standard_normal(T) * 3.0  # strong brain-wide common signal
        c1, c2 = rng.standard_normal(T), rng.standard_normal(T)  # two clusters
        F = np.empty((n_roi, T))
        for i in range(n_roi):
            F[i] = g + (c1 if i < n_roi // 2 else c2) * 0.8 + rng.standard_normal(T) * 0.5
        return F

    def _offdiag(self, C):
        return C[~np.eye(len(C), dtype=bool)]

    def test_gsr_and_partial_break_global_saturation(self) -> None:
        from asvimg.correlation import _correlation_matrix

        F = self._signals_with_global()
        raw = self._offdiag(_correlation_matrix(F, "raw"))
        gsr = self._offdiag(_correlation_matrix(F, "gsr"))
        par = self._offdiag(_correlation_matrix(F, "partial"))

        # raw is saturated toward 1; both alternatives collapse the common mode.
        self.assertGreater(raw.mean(), 0.8)
        self.assertLess(abs(gsr.mean()), 0.4)
        self.assertLess(abs(par.mean()), 0.4)
        # gsr still exposes within-cluster structure (some strong positives)
        self.assertGreater(gsr.max(), 0.3)

    def test_auto_threshold_adapts_to_method_scale(self) -> None:
        from asvimg.correlation import (
            _correlation_matrix,
            resolve_network_threshold,
        )

        F = self._signals_with_global(n_roi=10, T=400)
        density = 0.2

        def kept_fraction(C):
            thr = resolve_network_threshold(C, "auto", density)
            off = np.abs(C[~np.eye(len(C), dtype=bool)])
            return (off >= thr).mean(), thr

        frac_raw, thr_raw = kept_fraction(_correlation_matrix(F, "raw"))
        frac_par, thr_par = kept_fraction(_correlation_matrix(F, "partial"))
        # ~top 20% kept regardless of method, at very different absolute cutoffs
        self.assertAlmostEqual(frac_raw, density, delta=0.08)
        self.assertAlmostEqual(frac_par, density, delta=0.08)
        self.assertGreater(thr_raw, thr_par)  # raw cutoff near 1, partial tiny
        # a numeric threshold is passed through unchanged
        self.assertEqual(resolve_network_threshold(F, 0.42), 0.42)

    @unittest.skipUnless(_ATLAS.exists(), f"atlas not found at {_ATLAS}")
    def test_network_places_custom_roi_positions(self) -> None:
        from asvimg.atlas import ACCFv3
        from asvimg.correlation import (
            CorrResult,
            plot_correlation_network,
        )

        atlas = ACCFv3.from_mat(str(_ATLAS))
        C = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
        corr = CorrResult(name="G", C=C, roi_names=["MyA", "MyB", "MyC"],
                          mode="dff", method="raw")

        def node_offsets(fig):
            scat = [c for c in fig.axes[0].collections][-1]
            return np.asarray(scat.get_offsets())

        # custom names are not in the atlas -> without positions the nodes are NaN
        f0 = plot_correlation_network(corr, atlas, threshold=0.0)
        self.assertFalse(np.isfinite(node_offsets(f0)).all())
        # with edited-ROI positions they are placed
        pos = {"MyA": (100, 80), "MyB": (120, 200), "MyC": (80, 140)}
        f1 = plot_correlation_network(corr, atlas, threshold=0.0, positions=pos)
        self.assertTrue(np.isfinite(node_offsets(f1)).all())

    def test_mat_padded_roi_names_are_stripped(self) -> None:
        # .mat char arrays are fixed-width, so a round-trip pads ROI names with
        # trailing spaces; compute must strip them or the network plot can't
        # match them to atlas ROIs (only the max-width names would line up).
        from asvimg import PipelineConfig, save_payload
        from asvimg.correlation import compute_roi_correlations

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            F = self._signals_with_global(n_roi=3, T=120)
            save_payload(
                out / "roiSignals_GCaMP_TEST",
                {"F_dff": F, "roi_names": ["VISp_R   ", "MOs-m_L  ", "SSp-bfd_L"]},
                "npy",
            )
            cfg = PipelineConfig(
                input_dir=str(out / "in"), output_dir=str(out), output_format="npy",
                exp_name="TEST", channels_name=["GCaMP", "GCaMP"],
                channels_prop=["source", "donner"],
            )
            res = compute_roi_correlations(cfg, out)
            self.assertEqual(res["GCaMP"].roi_names, ["VISp_R", "MOs-m_L", "SSp-bfd_L"])

    def test_invalid_method_raises(self) -> None:
        from asvimg.correlation import _correlation_matrix

        with self.assertRaises(ValueError):
            _correlation_matrix(np.zeros((3, 10)), "bogus")

    def test_compute_respects_config_method_and_records_it(self) -> None:
        from asvimg import PipelineConfig, save_payload
        from asvimg.correlation import compute_roi_correlations

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            F = self._signals_with_global(n_roi=6, T=300)
            save_payload(
                out / "roiSignals_GCaMP_TEST",
                {"F_dff": F, "roi_names": [f"R{i}" for i in range(6)]},
                "npy",
            )
            cfg = PipelineConfig(
                input_dir=str(out / "in"),
                output_dir=str(out),
                output_format="npy",
                exp_name="TEST",
                channels_name=["GCaMP", "GCaMP"],
                channels_prop=["source", "donner"],
                corr_method="gsr",
            )
            res = compute_roi_correlations(cfg, out)
            self.assertIn("GCaMP", res)
            self.assertEqual(res["GCaMP"].method, "gsr")
            # explicit override wins over config
            res_raw = compute_roi_correlations(cfg, out, method="raw")
            self.assertEqual(res_raw["GCaMP"].method, "raw")
            off_gsr = self._offdiag(res["GCaMP"].C).mean()
            off_raw = self._offdiag(res_raw["GCaMP"].C).mean()
            self.assertLess(off_gsr, off_raw)


class TestRoiPerRoiProgress(unittest.TestCase):
    def test_extract_signals_reports_each_roi(self) -> None:
        from asvimg import StageId
        from asvimg.extraction import extract_signals

        rng = np.random.default_rng(0)
        H, W, T, N = 8, 9, 12, 5
        xyt = rng.standard_normal((H, W, T))
        masks = np.zeros((N, H, W), dtype=bool)
        for i in range(N):  # one distinct pixel per ROI so masks are non-empty
            masks[i, i, i] = True
        names = [f"R{i}" for i in range(N)]

        rep = RecordingReporter()
        res = extract_signals(
            xyt, masks, names, reporter=rep, stage=StageId.ROI,
            progress_label="G/raw",
        )
        roi_prog = [p for p in rep.progress if str(p.stage) == "roi"]
        # one event per ROI, counting up to N, carrying the ROI name
        self.assertEqual(len(roi_prog), N)
        self.assertEqual([p.current for p in roi_prog], list(range(1, N + 1)))
        self.assertTrue(all(p.total == N for p in roi_prog))
        self.assertIn("R0", roi_prog[0].message)
        self.assertIn("G/raw", roi_prog[0].message)

        # per-ROI result must equal the fast vectorised path exactly
        res_vec = extract_signals(xyt, masks, names)
        self.assertTrue(np.allclose(res.F, res_vec.F))


class TestRoiAdjoint(unittest.TestCase):
    """The ROI stage reads signals straight out of the source: warping is linear
    and an ROI mean is linear, so the two collapse into one weight map per ROI.
    These pin that it really is the SAME number the warp-then-average path gave."""

    H, W, HA, WA, T = 40, 48, 70, 80, 12

    def _stack(self):
        return np.random.default_rng(3).normal(0, 2, size=(self.H, self.W, self.T))

    def _disks(self, centers, r=5):
        yy, xx = np.mgrid[0:self.HA, 0:self.WA]
        return np.stack([(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r for cy, cx in centers])

    def _both_paths(self, stack, masks, tform):
        from asvimg.extraction import (
            extract_signals,
            extract_signals_from_source,
        )
        from asvimg.annotation import warp_stack

        names = [f"R{i}" for i in range(len(masks))]
        old = extract_signals(
            warp_stack(stack, tform, (self.HA, self.WA), dtype=np.float64), masks, names
        ).F
        new = extract_signals_from_source(
            lambda t0, t1: np.moveaxis(stack[..., t0:t1], 2, 0),
            stack.shape[2], (self.H, self.W), masks, names,
            transform=tform, chunk=5,  # chunked, to catch a time-blocking bug
        ).F
        return old, new

    def _edge_centers(self, tform):
        """ROI centres on the source footprint's edge in atlas space: each disk is
        partly outside the FOV, where a wrong adjoint (dropping vs zero-blending
        out-of-bounds neighbours) would show up."""
        corners = np.array(
            [[0, 0], [self.W - 1, 0], [self.W - 1, self.H - 1], [0, self.H - 1]], float
        )
        a = np.asarray(tform(corners))  # (x, y) in atlas
        mids = [(a[i] + a[(i + 1) % 4]) / 2 for i in range(4)]
        return [(float(p[1]), float(p[0])) for p in mids] + [
            (float(a[:, 1].mean()), float(a[:, 0].mean()))  # one fully inside
        ]

    def test_matches_warp_then_average(self) -> None:
        from skimage.transform import SimilarityTransform

        tform = SimilarityTransform(scale=1.4, rotation=0.2, translation=(8, -5))
        old, new = self._both_paths(self._stack(), self._disks(self._edge_centers(tform)), tform)
        self.assertTrue(np.abs(old).max() > 0.01)  # the ROIs actually see signal
        np.testing.assert_allclose(new, old, rtol=0, atol=1e-9)

    def test_matches_under_reflection(self) -> None:
        from skimage.transform import AffineTransform

        tform = AffineTransform(
            matrix=np.array([[-1.2, 0.2, 60.0], [0.3, 1.15, -4.0], [0, 0, 1.0]])
        )
        self.assertLess(np.linalg.det(np.asarray(tform.params)[:2, :2]), 0)
        old, new = self._both_paths(self._stack(), self._disks(self._edge_centers(tform)), tform)
        np.testing.assert_allclose(new, old, rtol=0, atol=1e-9)

    def test_nans_are_zeroed_like_warp_stack(self) -> None:
        from skimage.transform import SimilarityTransform

        tform = SimilarityTransform(scale=1.3, rotation=0.1, translation=(5, 5))
        stack = self._stack()
        stack[5:9, 10:16, :] = np.nan
        old, new = self._both_paths(stack, self._disks([(35, 40), (20, 25)]), tform)
        self.assertTrue(np.isfinite(new).all())
        np.testing.assert_allclose(new, old, rtol=0, atol=1e-9)

    def test_delta_pixel_lands_where_skimage_puts_it(self) -> None:
        """A half-pixel coordinate-convention slip would hide behind smooth noise;
        a single bright pixel exposes it."""
        from skimage.transform import SimilarityTransform

        delta = np.zeros((self.H, self.W, 1))
        delta[20, 24, 0] = 1000.0
        tform = SimilarityTransform(scale=1.0, rotation=0.0, translation=(0, 0))
        old, new = self._both_paths(delta, self._disks([(20, 24)], r=1), tform)
        self.assertGreater(old.ravel()[0], 0)
        np.testing.assert_allclose(new, old, rtol=0, atol=1e-9)


class TestWarpKernel(unittest.TestCase):
    """warp_stack's kernel is torch grid_sample (float64), not skimage's
    per-frame warp. These pin it to the ORIGINAL skimage kernel — defined here
    verbatim, so the module's own float32 default cannot mask a difference —
    across every way a resampling swap is known to go quietly wrong."""

    H, W, HA, WA = 40, 48, 70, 80

    def _baseline(self, stack, tform):
        """The kernel as it shipped before grid_sample: per-frame skimage, float64."""
        from skimage.transform import warp

        data = np.array(stack, dtype=np.float64)
        data[np.isnan(data)] = 0.0
        out = np.zeros((self.HA, self.WA, data.shape[2]), dtype=np.float64)
        for i in range(data.shape[2]):
            out[:, :, i] = warp(
                data[:, :, i], tform.inverse, output_shape=(self.HA, self.WA),
                preserve_range=True, order=1,
            )
        return out

    def _both(self, stack, tform):
        from asvimg.annotation import warp_stack

        return (
            self._baseline(stack, tform),
            warp_stack(stack, tform, (self.HA, self.WA), dtype=np.float64),
        )

    def test_matches_skimage_including_reflection(self) -> None:
        from skimage.transform import AffineTransform, SimilarityTransform

        stack = np.random.default_rng(0).normal(0, 2, size=(self.H, self.W, 6))
        reflected = AffineTransform(
            matrix=np.array([[-1.2, 0.2, 60.0], [0.3, 1.15, -4.0], [0, 0, 1.0]])
        )
        self.assertLess(np.linalg.det(np.asarray(reflected.params)[:2, :2]), 0)
        for tform in (
            SimilarityTransform(scale=1.5, rotation=0.2, translation=(6, -4)),
            SimilarityTransform(scale=0.7, rotation=-0.4, translation=(-5, 9)),
            reflected,
        ):
            ref, got = self._both(stack, tform)
            np.testing.assert_allclose(got, ref, rtol=0, atol=1e-9)

    def test_delta_impulse_lands_on_the_same_pixel(self) -> None:
        """A half-pixel convention slip (align_corners paired with the wrong
        normalization) still produces a plausible-looking warp; a single bright
        pixel is what exposes it."""
        from skimage.transform import SimilarityTransform

        delta = np.zeros((self.H, self.W, 1))
        delta[20, 24, 0] = 1000.0
        ref, got = self._both(delta, SimilarityTransform(
            scale=1.5, rotation=0.2, translation=(6, -4)))
        self.assertEqual(
            np.unravel_index(np.argmax(got[:, :, 0]), (self.HA, self.WA)),
            np.unravel_index(np.argmax(ref[:, :, 0]), (self.HA, self.WA)),
        )
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-9)

    def test_linear_ramp_has_no_half_pixel_offset(self) -> None:
        """On a ramp, a half-pixel slip shows up as a ~0.5 mean signed offset —
        the residue a max-abs check on smooth noise can hide."""
        from skimage.transform import SimilarityTransform

        ramp = np.tile(np.arange(self.W, dtype=np.float64), (self.H, 1))[:, :, None]
        ref, got = self._both(ramp, SimilarityTransform(
            scale=1.5, rotation=0.2, translation=(6, -4)))
        inside = ref[:, :, 0] > 0
        offset = float(np.mean(got[:, :, 0][inside] - ref[:, :, 0][inside]))
        self.assertLess(abs(offset), 1e-9)

    def test_nan_is_zeroed_before_warping(self) -> None:
        from skimage.transform import SimilarityTransform

        stack = np.random.default_rng(1).normal(0, 2, size=(self.H, self.W, 3))
        stack[5:9, 10:16, :] = np.nan
        ref, got = self._both(stack, SimilarityTransform(
            scale=1.3, rotation=0.1, translation=(4, 4)))
        self.assertTrue(np.isfinite(got).all())
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-9)

    def test_strictly_positive_uint16_where_skimage_could_clip(self) -> None:
        """skimage's warp clips the output to the input's value range. It is
        inert here in both regimes -- when pure-cval pixels exist it widens the
        range to include cval, and when the source covers the atlas every output
        is a convex combination of in-range values -- but that is a property of
        the geometry, not of order=1. Pin it: an input with NO zeros is exactly
        where a clip would bite, and it must not."""
        from skimage.transform import SimilarityTransform

        u16 = np.random.default_rng(2).integers(
            700, 900, size=(self.H, self.W, 2)
        ).astype(np.uint16)
        for tform in (
            SimilarityTransform(scale=1.2, rotation=0.1, translation=(5, 5)),  # zeros exist
            SimilarityTransform(scale=3.0, rotation=0.05, translation=(-8, -6)),  # atlas covered
        ):
            ref, got = self._both(u16, tform)
            np.testing.assert_allclose(got, ref, rtol=0, atol=1e-6)


class TestWarpStreaming(unittest.TestCase):
    """warp_stack streams in time-chunks and can write into a caller-owned
    destination, so the export never holds the (H_atlas, W_atlas, T) stack."""

    def test_chunking_does_not_change_the_result(self) -> None:
        from skimage.transform import SimilarityTransform

        from asvimg.annotation import warp_stack

        stack = np.random.default_rng(0).normal(0, 2, size=(20, 24, 17))
        tform = SimilarityTransform(scale=1.3, rotation=0.2, translation=(3, -2))
        whole = warp_stack(stack, tform, (30, 32), dtype=np.float64, chunk=10**6)
        for chunk in (1, 4, 17):
            got = warp_stack(stack, tform, (30, 32), dtype=np.float64, chunk=chunk)
            np.testing.assert_array_equal(got, whole)  # bit-identical, any chunking

    def test_writes_into_a_caller_supplied_out(self) -> None:
        from skimage.transform import SimilarityTransform

        from asvimg.annotation import warp_stack

        stack = (np.random.default_rng(1).random((16, 18, 5)) * 1000).astype(np.uint16)
        tform = SimilarityTransform(scale=1.1, rotation=0.05, translation=(1, 1))
        ref = warp_stack(stack, tform, (20, 22), dtype=np.float64)

        out = np.zeros((20, 22, 5), dtype=np.uint16)  # cast-on-write, as TIFF export does
        got = warp_stack(stack, tform, (20, 22), out=out)
        self.assertIs(got, out)
        np.testing.assert_array_equal(out, ref.astype(np.uint16))

    def test_warped_stack_on_disk_roundtrip_and_cleanup(self) -> None:
        from skimage.transform import SimilarityTransform

        from asvimg.annotation import warp_stack, warped_stack_on_disk

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "_warptmp_x.npy"
            stack = np.random.default_rng(2).normal(0, 2, size=(12, 14, 9))
            tform = SimilarityTransform(scale=1.2, rotation=0.1, translation=(2, 0))
            ref = warp_stack(stack, tform, (18, 20), dtype=np.float32)

            with warped_stack_on_disk(stack, tform, (18, 20), path) as warped:
                self.assertTrue(path.exists())
                self.assertEqual(warped.shape, (18, 20, 9))
                np.testing.assert_allclose(np.asarray(warped), ref, rtol=0, atol=0)
                # the (T,H,W) memmap comes back C-contiguous, so save_stack_tiff's
                # moveaxis+ascontiguousarray writes it without materializing a copy
                thw = np.moveaxis(warped, 2, 0)
                self.assertTrue(thw.flags["C_CONTIGUOUS"])
                self.assertTrue(np.shares_memory(np.ascontiguousarray(thw), thw))

            self.assertFalse(path.exists())  # temp file removed on exit


class TestCompletedStages(unittest.TestCase):
    def test_detects_stage_artifacts_on_disk(self) -> None:
        from asvimg import StageId
        from asvimg.runner import completed_stages

        with tempfile.TemporaryDirectory() as td:
            asi = Path(td) / "asi"
            npy = asi / "npy"
            npy.mkdir(parents=True)

            # nothing yet
            none = completed_stages(npy)
            self.assertFalse(any(none.values()))

            # reg_Ch*.npy are PREallocated at the start of a run, so they only say
            # preprocess started. A run that died before writing reg_meta.npz must
            # not read as Done -- the GUI would stamp it as a fresh baseline.
            (npy / "reg_Ch0.npy").touch()
            self.assertFalse(completed_stages(npy)[StageId.PREPROCESS])

            # lay down artifacts for a subset of stages
            (npy / "reg_meta.npz").touch()                  # preprocess (written last)
            (npy / "pca_images").mkdir()
            (npy / "pca_images" / "PC1.png").touch()        # pca
            (npy / "marks.mat").touch()                     # annotation
            (npy / "roiSignals_G_x.npy").touch()            # roi

            done = completed_stages(npy)
            self.assertTrue(done[StageId.PREPROCESS])
            self.assertTrue(done[StageId.PCA])
            self.assertTrue(done[StageId.ANNOTATION])
            self.assertTrue(done[StageId.ROI])
            self.assertFalse(done[StageId.ICA])           # no ica_images
            self.assertFalse(done[StageId.CORRELATION])   # never persisted
            self.assertFalse(done[StageId.EXPORT])

    def test_stage_artifacts_returns_paths_with_mtime(self) -> None:
        from asvimg import StageId
        from asvimg.runner import stage_artifacts

        with tempfile.TemporaryDirectory() as td:
            npy = Path(td) / "asi" / "npy"
            npy.mkdir(parents=True)
            (npy / "reg_Ch0.npy").touch()
            (npy / "reg_meta.npz").touch()
            arts = stage_artifacts(npy)
            self.assertIsNotNone(arts[StageId.PREPROCESS])
            self.assertTrue(arts[StageId.PREPROCESS].exists())  # has a real mtime
            self.assertIsNone(arts[StageId.ICA])


class TestSaveFigures(unittest.TestCase):
    def _run_pca(self, save_figures):
        tmp = Path(tempfile.mkdtemp())
        out = tmp / "out"
        out.mkdir()
        _fabricate_preprocess_outputs(out)
        cfg = _make_4ch_config(tmp / "in", out)
        cfg.save_figures = save_figures
        # NullReporter (headless): figures are still built + saved when enabled
        PipelineSession(cfg).run_pca()
        return out / "figures"

    def test_png_pdf_saved_headless(self) -> None:
        figs = self._run_pca("png+pdf")
        pngs = list(figs.glob("*.png"))
        pdfs = list(figs.glob("*.pdf"))
        self.assertTrue(pngs)
        self.assertEqual(len(pngs), len(pdfs))

    def test_png_only(self) -> None:
        figs = self._run_pca("png")
        self.assertTrue(list(figs.glob("*.png")))
        self.assertEqual(list(figs.glob("*.pdf")), [])

    def test_none_writes_nothing(self) -> None:
        figs = self._run_pca("none")
        self.assertFalse(figs.exists())


class TestExportProgress(unittest.TestCase):
    def test_warp_stack_reports_frame_progress(self) -> None:
        from skimage.transform import SimilarityTransform

        from asvimg.annotation import warp_stack

        calls: list = []
        stack = np.zeros((8, 8, 50), dtype=float)
        warp_stack(stack, SimilarityTransform(), (8, 8),
                   on_frame=lambda d, t: calls.append((d, t)))
        self.assertTrue(calls)
        self.assertEqual(calls[-1], (50, 50))  # completion reported

    @unittest.skipUnless(_ATLAS.exists(), f"atlas not found at {_ATLAS}")
    def test_run_export_reports_progress(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            out = tmp / "out"
            out.mkdir()
            _fabricate_preprocess_outputs(out)
            cfg = _make_4ch_config(tmp / "in", out)
            cfg.save_annotated_dF_mat = True  # export does real warping

            rep = RecordingReporter()
            session = PipelineSession(
                cfg, reporter=rep,
                marks_provider=FakeMarksProvider(
                    [[5, 5], [5, 22], [18, 14]],
                    [[100, 100], [100, 180], [200, 140]],
                ),
            )
            session.run_annotation()
            session.run_export()
            exp = [p for p in rep.progress if str(p.stage) == "export"]
            self.assertTrue(exp)  # progress bar now moves during export
            # export status messages reach the log window (reporter), not stdout
            exp_logs = [l for l in rep.logs if str(l.stage) == "export"]
            self.assertTrue(exp_logs)


class TestRois(unittest.TestCase):
    def test_contralateral_and_masks_and_io(self) -> None:
        from asvimg import (
            Roi,
            build_masks,
            contralateral,
            load_rois,
            save_rois,
        )

        r = Roi("VISp_R", 200, 100, 8)
        c = contralateral(r, width=285)
        self.assertEqual(c.name, "VISp_L")
        # the symmetry axis of a 285-px image is (285-1)/2 = 142.0, not 142.5
        self.assertEqual(c.x, (285 - 1) - 200)
        # bigger size -> more pixels
        masks, names = build_masks([Roi("a", 50, 50, 12), Roi("b", 60, 60, 4)], (100, 100))
        self.assertEqual(names, ["a", "b"])
        self.assertGreater(masks[0].sum(), masks[1].sum())
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "rois.csv"
            save_rois(p, [r, c])
            back = load_rois(p)
            self.assertEqual([x.name for x in back], ["VISp_R", "VISp_L"])

    @unittest.skipUnless(_ATLAS.exists(), f"atlas not found at {_ATLAS}")
    def test_run_roi_uses_custom_rois_csv(self) -> None:
        from asvimg import Roi, load_payload, save_rois

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            out = tmp / "out"
            out.mkdir()
            _fabricate_preprocess_outputs(out)
            cfg = _make_4ch_config(tmp / "in", out)
            save_rois(out / "rois.csv", [Roi("L", 100, 140, 12), Roi("R", 185, 140, 6)])

            session = PipelineSession(
                cfg,
                marks_provider=FakeMarksProvider(
                    [[5, 5], [5, 22], [18, 14]],
                    [[100, 100], [100, 180], [200, 140]],
                ),
            )
            session.run_annotation()
            session.run_roi()
            payload = load_payload(next(out.glob("roiSignals_*.npy")))
            names = [str(n).strip() for n in payload["roi_names"]]
            self.assertEqual(names, ["L", "R"])              # custom ROIs used
            self.assertGreater(payload["n_pixels"][0], payload["n_pixels"][1])  # per-ROI size

    @unittest.skipUnless(_ATLAS.exists(), f"atlas not found at {_ATLAS}")
    def test_run_roi_reports_progress_and_warps_nothing(self) -> None:
        """ROI signals come from the source via the adjoint — the stage must
        report progress and must NEVER build a warped stack (16.6 GB at T=6000)."""
        from asvimg import annotation

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            out = tmp / "out"
            out.mkdir()
            _fabricate_preprocess_outputs(out)
            cfg = _make_4ch_config(tmp / "in", out)

            rep = RecordingReporter()
            session = PipelineSession(
                cfg,
                reporter=rep,
                marks_provider=FakeMarksProvider(
                    [[5, 5], [5, 22], [18, 14]],
                    [[100, 100], [100, 180], [200, 140]],
                ),
            )
            session.run_annotation()

            calls = []
            real_warp_stack = annotation.warp_stack
            annotation.warp_stack = lambda *a, **k: calls.append(a) or real_warp_stack(*a, **k)
            try:
                session.run_roi()
            finally:
                annotation.warp_stack = real_warp_stack

            self.assertEqual(calls, [], "the ROI stage warped a stack")
            roi_prog = [p for p in rep.progress if "ROI signals" in p.message]
            self.assertTrue(roi_prog, "the ROI stage emitted no progress")
            self.assertEqual(roi_prog[-1].current, roi_prog[-1].total)  # reaches 100%
            self.assertTrue(any("raw" in p.message for p in roi_prog))
            self.assertTrue(any("dff" in p.message for p in roi_prog))



class TestPlotStyle(unittest.TestCase):
    def test_apply_style_sets_rcparams(self) -> None:
        import matplotlib.pyplot as plt

        from asvimg import apply_style

        apply_style(force=True)
        self.assertEqual(plt.rcParams["pdf.fonttype"], 42)
        self.assertFalse(plt.rcParams["axes.spines.top"])
        self.assertFalse(plt.rcParams["axes.spines.right"])
        self.assertTrue(plt.rcParams["axes.grid"])
        self.assertEqual(plt.rcParams["axes.grid.axis"], "y")

    def test_session_applies_style(self) -> None:
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        from asvimg import plotstyle

        plotstyle._APPLIED = False  # let the session's apply_style() run
        mpl.rcParams["pdf.fonttype"] = 3  # clobber, then construct a session
        with tempfile.TemporaryDirectory() as td:
            PipelineSession(_make_4ch_config(Path(td) / "in", Path(td)))
        self.assertEqual(plt.rcParams["pdf.fonttype"], 42)


class TestFrameWindow(unittest.TestCase):
    """start_initial_frames / ignore_last_frames and the ROI-plot window."""

    def test_resolved_start_and_window(self) -> None:
        cfg = PipelineConfig(
            fps=20, channels_name=["G", "R"], channels_prop=["source", "donner"],
            start_initial_frames=None, ignore_last_frames=3,
        )
        self.assertEqual(cfg.resolved_start_initial_frames, 40)  # 4*20/2
        self.assertEqual(cfg.roi_plot_window(100), (40, 97))
        # crossed ends fall back to the full range
        self.assertEqual(cfg.roi_plot_window(30), (0, 30))

    def test_explicit_start_overrides_default(self) -> None:
        cfg = PipelineConfig(
            fps=20, channels_name=["G", "R"], channels_prop=["source", "donner"],
            start_initial_frames=5, ignore_last_frames=0,
        )
        self.assertEqual(cfg.resolved_start_initial_frames, 5)
        self.assertEqual(cfg.roi_plot_window(50), (5, 50))

    def test_legacy_ignore_initial_frames_still_loads(self) -> None:
        import warnings

        from asvimg.config import load_config, save_config

        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            base = PipelineConfig(
                channels_name=["G", "R"], channels_prop=["source", "donner"], fps=20
            )
            save_config(base, p)
            # rewrite ops.yaml with the OLD field name
            ops = (p / "ops.yaml").read_text().replace(
                "start_initial_frames: null", "ignore_initial_frames: 7"
            )
            (p / "ops.yaml").write_text(ops)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                cfg = load_config(p)
            self.assertEqual(cfg.start_initial_frames, 7)
            self.assertTrue(any("renamed" in str(x.message).lower() for x in w))


class TestQuickPreview(unittest.TestCase):
    def test_preview_emits_labelled_frame_grid(self) -> None:
        import tifffile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            inp = tmp / "input"
            inp.mkdir()
            stack = np.random.default_rng(2).integers(
                200, 3000, size=(12, 16, 20)
            ).astype(np.uint16)
            tifffile.imwrite(inp / "recS1.tif", stack)

            cfg = PipelineConfig(
                input_dir=str(inp),
                output_dir=str(tmp / "out"),
                output_format="npy",
                channels_name=["G", "R", "G", "R"],
                channels_prop=["source", "source", "donner", "source"],
                fps=20,
            )
            rep = RecordingReporter()
            PipelineSession(cfg, reporter=rep).preview_input(n_frames=12)

            figs = [f for f in rep.figures if f.key == "input_preview"]
            self.assertEqual(len(figs), 1)
            # the log names the channel cycle so G/R is visible
            self.assertTrue(any("first 12 frames" in l.text for l in rep.logs))

    def test_preview_lists_input_files_in_read_order(self) -> None:
        import tifffile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            inp = tmp / "input"
            inp.mkdir()
            rng = np.random.default_rng(3)
            for i, n in ((1, 12), (2, 8), (10, 4)):
                tifffile.imwrite(
                    inp / f"rec_{i}.tif",
                    rng.integers(200, 3000, size=(n, 16, 20)).astype(np.uint16),
                )

            cfg = PipelineConfig(
                input_dir=str(inp),
                output_dir=str(tmp / "out"),
                output_format="npy",
                channels_name=["G", "R"],
                channels_prop=["source", "source"],
            )
            rep = RecordingReporter()
            PipelineSession(cfg, reporter=rep).preview_input(n_frames=4)

            listed = [
                l.text for l in rep.logs
                if re.search(r"\d+\.\s.*rec_\d+\.tif", l.text)
            ]
            names = [re.search(r"(rec_\d+\.tif)", t).group(1) for t in listed]
            # natural order: rec_2 before rec_10, matching the concat read order
            self.assertEqual(names, ["rec_1.tif", "rec_2.tif", "rec_10.tif"])
            # each listed file carries a human-readable size, plus a total line
            self.assertTrue(all(re.search(r"\d+(\.\d+)?\s*(B|KB|MB|GB)", t) for t in listed))
            self.assertTrue(any("total" in l.text for l in rep.logs))
            # the order in force is named, and both timestamps are shown so a
            # wrong input_order is caught here rather than after a long run
            self.assertTrue(any("input_order='natural'" in l.text for l in rep.logs))
            self.assertTrue(
                all(len(re.findall(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", t)) == 2
                    for t in listed)
            )

    def test_preview_marks_the_sort_key_column_for_every_order(self) -> None:
        """The preview's whole job is "is this the order I meant?", so the
        column actually sorted on must be marked whichever order is in force —
        including the name-based ones, where the sort key is the filename."""
        import os

        import tifffile

        from asvimg import find_input_files
        from asvimg.runner.session import StageId

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            inp = tmp / "input"
            inp.mkdir()
            rng = np.random.default_rng(7)
            for name, when in (("b_rec.tif", 300), ("a_rec.tif", 100)):
                tifffile.imwrite(
                    inp / name, rng.integers(1, 99, size=(4, 8, 8)).astype(np.uint16)
                )
                os.utime(inp / name, (when, when))

            for order, marked in (("natural", "name"), ("name", "name"),
                                  ("mtime", "modified"), ("ctime", "created")):
                cfg = PipelineConfig(
                    input_dir=str(inp), output_dir=str(tmp / f"out_{order}"),
                    output_format="npy", input_order=order,
                    channels_name=["G", "R"], channels_prop=["source", "source"],
                )
                rep = RecordingReporter()
                sess = PipelineSession(cfg, reporter=rep)
                sess._log_input_files(
                    StageId.PREPROCESS, find_input_files(inp, "auto", order)
                )
                header = next(l.text for l in rep.logs if "created" in l.text)
                self.assertIn("<-", header, f"{order}: no sort-key marker")
                # the marker sits on the column that order really sorted on
                self.assertLess(
                    header.index(marked), header.index("<-"),
                    f"{order}: marker is not on the {marked} column ({header!r})",
                )

    def test_preview_survives_unrepresentable_timestamps(self) -> None:
        """A preview must never be the thing that fails: some archive/restore
        paths and network shares report pre-1970 or absurd times, which
        datetime.fromtimestamp rejects outright on Windows."""
        from asvimg.runner.session import _fmt_time

        for bad in (-1.0, -86400.0, 159089797835.0, 1e20):
            self.assertEqual(_fmt_time(bad), "?")
        self.assertEqual(_fmt_time(0), "-")
        self.assertRegex(_fmt_time(1_000_000.0), r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d$")

    @staticmethod
    def _write_files(inp: Path, counts: tuple[int, ...]) -> None:
        import tifffile

        inp.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(4)
        for i, n in enumerate(counts, start=1):
            tifffile.imwrite(
                inp / f"rec_{i}.tif",
                rng.integers(200, 3000, size=(n, 16, 20)).astype(np.uint16),
            )

    def test_preview_all_covers_every_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            inp = tmp / "input"
            self._write_files(inp, (8, 8, 8))

            cfg = PipelineConfig(
                input_dir=str(inp),
                output_dir=str(tmp / "out"),
                output_format="npy",
                channels_name=["G", "R"],
                channels_prop=["source", "donner"],
            )
            rep = RecordingReporter()
            PipelineSession(cfg, reporter=rep).preview_input_all(n_frames=2)

            figs = [f for f in rep.figures if f.key == "input_preview_all"]
            self.assertEqual(len(figs), 1)
            self.assertTrue(any("each of 3 file(s)" in l.text for l in rep.logs))
            # nothing to correct -> no demux-correction line
            self.assertFalse(any("demux correction" in l.text for l in rep.logs))

    def test_preview_all_labels_use_channels_slip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            inp = tmp / "input"
            self._write_files(inp, (8, 8, 8))

            cfg = PipelineConfig(
                input_dir=str(inp),
                output_dir=str(tmp / "out"),
                output_format="npy",
                channels_name=["G", "R"],
                channels_prop=["source", "donner"],
                channels_slip=[0, 1, 0],  # the 2nd file's cycle starts 1 step in
            )
            from asvimg import StageId

            rep = RecordingReporter()
            session = PipelineSession(cfg, reporter=rep)
            session.preview_input_all(n_frames=2)

            self.assertEqual(len([f for f in rep.figures if f.key == "input_preview_all"]), 1)
            # the slip is reported, scoped to the 2nd file's global frame range
            self.assertTrue(
                any("demux correction" in l.text and "(8, 16, 1)" in l.text
                    for l in rep.logs)
            )
            # and the labels follow it: file 2's first frame is R, not G
            corr = session._demux_correction(StageId.PREPROCESS, [8, 8, 8])
            self.assertIn("Ch1 R-donner", session._channel_label(corr, 8))
            self.assertIn("Ch0 G-source", session._channel_label(corr, 16))


class TestCustomAnnotationMaps(unittest.TestCase):
    """map_for_annot custom cpselect maps (png / npy / tif → grayscale, resized)."""

    def _session(self, out: Path) -> PipelineSession:
        cfg = PipelineConfig(
            input_dir=str(out.parent), output_dir=str(out), exp_name="S1"
        )
        return PipelineSession(cfg, reporter=RecordingReporter())

    def test_load_gray_map_rgb_npy_and_stack(self) -> None:
        import tifffile

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            # RGB npy (H, W, 3) -> luminance 2-D
            rgb = np.zeros((6, 8, 3), dtype=np.uint8)
            rgb[..., 0] = 255  # pure red
            np.save(d / "rgb.npy", rgb)
            g = PipelineSession._load_gray_map(d / "rgb.npy")
            self.assertEqual(g.shape, (6, 8))
            self.assertTrue(np.allclose(g, 255 * 0.2989))  # red luminance
            # multi-page tif (N, H, W) -> mean projection 2-D
            stack = np.stack([np.full((6, 8), k, np.float32) for k in (0, 2, 4)])
            tifffile.imwrite(d / "stack.tif", stack)
            gs = PipelineSession._load_gray_map(d / "stack.tif")
            self.assertEqual(gs.shape, (6, 8))
            self.assertTrue(np.allclose(gs, 2.0))  # mean of 0, 2, 4

    def test_custom_maps_grayscale_resized_and_aspect_warned(self) -> None:
        import tifffile
        from PIL import Image

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            out = tmp / "asi" / "npy"
            out.mkdir(parents=True)
            map_dir = out / "map_for_annot"
            map_dir.mkdir(parents=True)
            source_img = np.zeros((16, 20), dtype=np.float64)  # H x W
            Image.fromarray(
                np.random.default_rng(0).integers(0, 255, (16, 20, 3), np.uint8)
            ).save(map_dir / "a_rgb.png")
            np.save(map_dir / "b.npy", np.random.rand(16, 20).astype(np.float32))
            tifffile.imwrite(map_dir / "c.tif", np.random.rand(16, 20).astype(np.float32))
            np.save(map_dir / "d_wrong.npy", np.random.rand(10, 40).astype(np.float32))

            sess = self._session(out)
            imgs, labels = sess._load_custom_annotation_maps(source_img)

            self.assertEqual(labels, ["a_rgb", "b", "c", "d_wrong"])
            self.assertEqual(len(imgs), 4)
            for im in imgs:
                self.assertEqual(im.ndim, 2)          # grayscale
                self.assertEqual(im.shape, (16, 20))  # resized to source
            self.assertTrue(
                any(
                    "WARNING" in l.text and "d_wrong" in l.text
                    for l in sess.reporter.logs
                )
            )

    def test_load_gray_map_la_and_palette_png(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            # grayscale + alpha (LA) PNG -> alpha dropped, luminance kept
            lum = np.full((5, 7), 120, np.uint8)
            Image.fromarray(lum).convert("LA").save(d / "la.png")
            g = PipelineSession._load_gray_map(d / "la.png")
            self.assertEqual(g.shape, (5, 7))
            self.assertTrue(np.allclose(g, 120))
            # palette PNG of a solid red -> resolved to luminance, not raw index 0
            red = np.zeros((5, 7, 3), np.uint8)
            red[..., 0] = 255
            Image.fromarray(red).convert("P").save(d / "pal.png")
            gp = PipelineSession._load_gray_map(d / "pal.png")
            self.assertEqual(gp.shape, (5, 7))
            self.assertTrue(np.allclose(gp, 255 * 0.2989, atol=3))  # red luminance

    def test_custom_maps_skip_degenerate_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            out = tmp / "asi" / "npy"
            out.mkdir(parents=True)
            map_dir = out / "map_for_annot"
            map_dir.mkdir(parents=True)
            np.save(map_dir / "a_valid.npy", np.random.rand(16, 20).astype(np.float32))
            np.save(map_dir / "z_bad.npy", np.zeros((0, 5), np.float32))  # degenerate
            sess = self._session(out)
            imgs, labels = sess._load_custom_annotation_maps(np.zeros((16, 20)))
            # the bad file is skipped (not fatal), the good one still loads
            self.assertEqual(labels, ["a_valid"])
            self.assertEqual(imgs[0].shape, (16, 20))
            self.assertTrue(
                any("skipped" in l.text and "z_bad" in l.text for l in sess.reporter.logs)
            )

    def test_custom_maps_empty_creates_folder(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            out = tmp / "asi" / "npy"
            out.mkdir(parents=True)
            sess = self._session(out)
            imgs, labels = sess._load_custom_annotation_maps(np.zeros((8, 8)))
            self.assertEqual((imgs, labels), ([], []))
            self.assertTrue((out / "map_for_annot").is_dir())


class TestSeedCorrMap(unittest.TestCase):
    """Seed-based atlas-space correlation maps (asvimg.seedmap)."""

    class _FakeAtlas:
        def __init__(self, h, w) -> None:
            self.shape_hw = (h, w)
            self.brain_mask = np.ones((h, w), bool)
            self.point_roi_names = ["seedA"]
            self._c = (h // 2, w // 2)

        def get_point_roi_mask(self, roi, diameter=5):
            rr, cc = np.mgrid[: self.shape_hw[0], : self.shape_hw[1]]
            pr, pc = self._c
            return ((rr - pr) ** 2 + (cc - pc) ** 2) <= (diameter / 2.0) ** 2

    def _data(self):
        from skimage.transform import SimilarityTransform

        h, w, T = 20, 24, 60
        rng = np.random.default_rng(0)
        s = rng.standard_normal(T)
        dff = rng.standard_normal((h, w, T)) * 0.01
        atlas = self._FakeAtlas(h, w)
        dff[atlas.get_point_roi_mask("seedA", diameter=5)] = s  # seed -> series s
        dff[0, 0, :] = s   # perfectly correlated (+1)
        dff[0, 1, :] = -s  # anti-correlated (-1)
        return dff, SimilarityTransform(), atlas  # identity transform (src == atlas)

    def test_raw_corr_extremes(self) -> None:
        from asvimg.seedmap import compute_seed_corr_map

        dff, tform, atlas = self._data()
        cmap = compute_seed_corr_map(
            dff, tform, atlas, "seedA", method="raw", skip_frames=1
        )
        self.assertEqual(cmap.shape, atlas.shape_hw)
        self.assertAlmostEqual(float(cmap[0, 0]), 1.0, places=4)
        self.assertAlmostEqual(float(cmap[0, 1]), -1.0, places=4)
        self.assertTrue(np.all(np.abs(cmap) <= 1.0 + 1e-5))

    def test_seed_mask_path_and_shape_check(self) -> None:
        from asvimg.seedmap import compute_seed_corr_map

        dff, tform, atlas = self._data()
        mask = atlas.get_point_roi_mask("seedA", diameter=5)  # (h, w) bool
        cmap = compute_seed_corr_map(
            dff, tform, atlas, seed_mask=mask, skip_frames=1
        )
        self.assertAlmostEqual(float(cmap[0, 0]), 1.0, places=4)
        with self.assertRaises(ValueError):  # mask must match atlas shape
            compute_seed_corr_map(
                dff, tform, atlas, seed_mask=np.ones((3, 3), bool), skip_frames=1
            )

    def test_gsr_runs_and_bounded(self) -> None:
        from asvimg.seedmap import compute_seed_corr_map

        dff, tform, atlas = self._data()
        cmap = compute_seed_corr_map(
            dff, tform, atlas, "seedA", method="gsr", skip_frames=2
        )
        self.assertEqual(cmap.shape, atlas.shape_hw)
        self.assertTrue(np.all(np.abs(np.nan_to_num(cmap)) <= 1.0 + 1e-5))

    def test_unwarp_is_inverse_of_warp(self) -> None:
        # A non-identity transform verifies unwarp_to_source really goes
        # atlas -> source (not the same direction as warp_image).
        from skimage.transform import SimilarityTransform

        from asvimg.annotation import warp_image
        from asvimg.seedmap import unwarp_to_source

        tform = SimilarityTransform(scale=1.0, translation=(3, -2))
        src = np.zeros((30, 40), np.float64)
        src[10:20, 15:25] = 1.0  # a bright block
        warped = warp_image(src, tform, (32, 44))          # source -> atlas
        back = unwarp_to_source(warped, tform, src.shape)  # atlas -> source

        def _centroid(a):
            ys, xs = np.nonzero(a > 0.5)
            return ys.mean(), xs.mean()

        cy0, cx0 = _centroid(src)
        cy1, cx1 = _centroid(back)
        self.assertLess(abs(cy1 - cy0), 1.5)  # block returns to its origin
        self.assertLess(abs(cx1 - cx0), 1.5)

    def test_smoothing_reduces_speckle_and_vmax(self) -> None:
        from asvimg.seedmap import compute_seed_corr_map, corr_vmax

        dff, tform, atlas = self._data()
        rough = compute_seed_corr_map(
            dff, tform, atlas, "seedA", skip_frames=1, smooth_sigma=0.0
        )
        smooth = compute_seed_corr_map(
            dff, tform, atlas, "seedA", skip_frames=1, smooth_sigma=3.0
        )

        def _rough(m):
            g = np.gradient(np.nan_to_num(m))
            return float(np.mean(np.abs(g[0])) + np.mean(np.abs(g[1])))

        self.assertLess(_rough(smooth), _rough(rough))  # smoothing calms speckle
        self.assertEqual(smooth.shape, atlas.shape_hw)
        self.assertGreater(corr_vmax(rough), 0.0)  # positive auto half-range

    def test_save_and_unwarp_roundtrip(self) -> None:
        from asvimg.seedmap import (
            compute_seed_corr_map,
            save_corr_map,
            unwarp_to_source,
        )

        dff, tform, atlas = self._data()
        cmap = compute_seed_corr_map(dff, tform, atlas, "seedA", skip_frames=1)
        with tempfile.TemporaryDirectory() as td:
            npy, png = save_corr_map(Path(td), "seedA_map", cmap)
            self.assertTrue(npy.exists() and png.exists())
            self.assertEqual(np.load(npy).shape, atlas.shape_hw)
        un = unwarp_to_source(cmap, tform, atlas.shape_hw)  # identity -> same grid
        self.assertEqual(un.shape, atlas.shape_hw)
        self.assertFalse(np.isnan(un).any())


if __name__ == "__main__":
    unittest.main()
