"""roi_space="source": ROI signals without atlas registration.

When the ROIs are already in the recording's own coordinates there is no warp to
do — the mask IS the weight map — so ROI signals (and correlation) can be read
with no transform at all. These tests pin: the numbers equal a plain masked mean,
the stage runs with annotation skipped, and correlation follows.
"""

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from asvimg import (
    PipelineConfig,
    Roi,
    SOURCE_ROIS_FILENAME,
    load_payload,
    reg_channel_path,
    save_rois,
    write_reg_meta,
)
from asvimg.extraction import extract_signals_from_source

H, W, T = 20, 24, 80


def _fixture(out: Path):
    """One donner-backed group with a known dF/F, so a source ROI's mean is
    computable by hand."""
    rng = np.random.default_rng(0)
    t = np.arange(T)
    dff = (rng.normal(size=(H, W))[:, :, None] * np.sin(t / 11)[None, None, :]
           + rng.normal(0, 0.05, size=(H, W, T)))
    meta = {
        "proc_template": np.zeros((H, W)),
        "imageSize": np.array([H, W]),
        "channels_name": np.array(["BL", "BL"]),
        "channels_prop": np.array(["source", "donner"]),
        "T_per_ch": np.array([T, T], dtype=np.int64),
        "fps": np.array(20),
    }
    for i in range(2):
        arr = rng.integers(400, 3000, size=(T, H, W)).astype(np.uint16)
        np.save(reg_channel_path(out, i), arr)
        meta[f"meanImageCh{i}"] = arr.mean(axis=0)
    write_reg_meta(out, meta)
    np.save(out / "dff_BL.npy", np.moveaxis(dff, 2, 0).astype(np.float32))
    return dff


def _config(out: Path, **over) -> PipelineConfig:
    cfg = PipelineConfig(
        input_dir=str(out.parent / "in"), output_dir=str(out), output_format="npy",
        exp_name="SRC", channels_name=["BL", "BL"], channels_prop=["source", "donner"],
        fps=20, roi_signal="dff", roi_space="source", annotation=False,
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture()
def out():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "out"
        d.mkdir(parents=True)
        yield d


def test_transform_none_is_a_plain_masked_mean():
    """The kernel with transform=None must equal averaging the masked pixels —
    no warp, no atlas."""
    rng = np.random.default_rng(1)
    stack = rng.normal(size=(H, W, T))
    masks = np.zeros((2, H, W), bool)
    masks[0, 3:7, 4:9] = True          # a box
    masks[1, 12:16, 15:20] = True

    res = extract_signals_from_source(
        lambda t0, t1: np.moveaxis(stack[..., t0:t1], 2, 0),
        T, (H, W), masks, ["a", "b"], transform=None,
    )
    for i in range(2):
        want = stack[masks[i]].mean(axis=0)          # (T,) direct masked mean
        np.testing.assert_allclose(res.F[i], want, rtol=0, atol=1e-12)


def test_source_masks_must_match_the_source_shape():
    stack = np.zeros((H, W, T))
    bad = np.zeros((1, H + 1, W), bool)              # wrong shape
    with pytest.raises(ValueError, match="source shape"):
        extract_signals_from_source(
            lambda t0, t1: np.moveaxis(stack[..., t0:t1], 2, 0),
            T, (H, W), bad, ["x"], transform=None,
        )


def test_roi_stage_runs_with_annotation_skipped(out):
    """The whole point: annotation=False (no marks, no transform), yet ROI signals
    come out — because the ROIs are already in source space."""
    dff = _fixture(out)
    save_rois(out / SOURCE_ROIS_FILENAME, [
        Roi("left", x=6, y=5, size=5),
        Roi("right", x=17, y=13, size=5),
    ])
    cfg = _config(out)

    from asvimg.runner import PipelineSession

    s = PipelineSession(cfg)          # no marks provider, annotation=False
    s.run_preprocess() if False else None  # dff already on disk
    payloads = s.run_roi()

    assert set(payloads) == {"BL"}
    F = np.asarray(load_payload(next(out.glob("roiSignals_BL_*.npy")))["F_dff"])
    assert F.shape == (2, T)

    # the "left" ROI signal equals the dF/F averaged over that disc, by hand
    from asvimg.rois import build_masks
    masks, _ = build_masks(
        [Roi("left", x=6, y=5, size=5), Roi("right", x=17, y=13, size=5)], (H, W)
    )
    want = dff[masks[0]].mean(axis=0)
    np.testing.assert_allclose(F[0], want.astype(np.float32), rtol=0, atol=1e-4)


def test_no_source_rois_is_a_clean_skip(out):
    """roi_space='source' with no rois_source.csv extracts nothing (rather than
    inventing atlas regions), and does not raise."""
    _fixture(out)
    from asvimg.runner import PipelineSession

    payloads = PipelineSession(_config(out)).run_roi()
    assert payloads == {}
    assert not list(out.glob("roiSignals_*"))


def test_correlation_runs_in_source_mode(out):
    """Correlation must follow source-space ROI signals with no atlas."""
    _fixture(out)
    save_rois(out / SOURCE_ROIS_FILENAME, [
        Roi("a", x=6, y=5, size=5),
        Roi("b", x=17, y=13, size=5),
        Roi("c", x=10, y=10, size=5),
    ])
    cfg = _config(out, save_figures="none")
    from asvimg.runner import PipelineSession

    s = PipelineSession(cfg)
    s.run_roi()
    corrs = s.run_correlation()
    assert "BL" in corrs
    assert corrs["BL"].C.shape == (3, 3)


def test_atlas_mode_still_needs_a_transform(out):
    """The default (atlas) path is unchanged: no transform -> ROI is skipped."""
    _fixture(out)
    cfg = _config(out, roi_space="atlas")
    from asvimg.runner import PipelineSession

    payloads = PipelineSession(cfg).run_roi()
    assert payloads == {}
