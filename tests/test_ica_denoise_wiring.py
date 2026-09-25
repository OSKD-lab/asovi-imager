"""ica_denoise: the exclusion reaches the saved artifacts — or nothing changes.

The contract this feature lives or dies by:
  * default ("off")  -> every output is byte-identical to before it existed
  * "subtract" with nothing excluded -> still identical (the applicator is exact)
  * "subtract" with an artifact IC    -> the artifact is gone from ROI signals,
                                         and every payload SAYS it is denoised
  * a missing or stale denoised stack -> RAISES. It never quietly serves the plain
                                         dF/F under a config that says otherwise.
"""

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from skimage.transform import SimilarityTransform

from conftest import TEST_ATLAS_OR_DEFAULT
from asvimg import PipelineConfig, load_payload, reg_channel_path, write_reg_meta
from asvimg.annotation import _load_name_dff_stack
from asvimg.extraction import extract_signals_from_source
from asvimg.ica_state import denoised_dff_path, save_exclusions
from asvimg.runner import FakeIcaProvider, PipelineSession

H, W, T = 14, 16, 120


def _fixture(out: Path):
    """One donner-backed group whose dF/F carries a stripe-shaped artifact."""
    rng = np.random.default_rng(0)
    t = np.arange(T)
    dff = (rng.normal(size=(H, W))[:, :, None] * np.sin(t / 17)[None, None, :]
           + rng.normal(0, 0.05, size=(H, W, T)))
    art = np.zeros((H, W))
    art[:, 2:5] = 1.0
    dff = dff + art[:, :, None] * (2.0 * np.sin(t / 7))[None, None, :]

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
    return art


def _config(out: Path, **over) -> PipelineConfig:
    cfg = PipelineConfig(
        input_dir=str(out.parent / "in"), output_dir=str(out), output_format="npy",
        exp_name="TEST", channels_name=["BL", "BL"], channels_prop=["source", "donner"],
        fps=20, ch_for_annotation=0, pca_n_components=4, ica_n_components=4,
        pca_skip_frames=1, annotation=False,
        # "gui" = always ask the (injected) picker. With "cache" a recorded answer
        # is replayed and the provider is never consulted.
        ica_exclusion="gui",
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _roi_signals(cfg, out):
    stack, note = _load_name_dff_stack("BL", cfg.channel_groups()["BL"], cfg, out)
    masks = np.zeros((2, 20, 22), bool)
    masks[0, 4:9, 4:9] = True
    masks[1, 10:15, 12:18] = True
    F = extract_signals_from_source(
        lambda t0, t1, _s=stack: np.moveaxis(np.asarray(_s[..., t0:t1]), 2, 0),
        stack.shape[2], stack.shape[:2], masks, ["a", "b"],
        transform=SimilarityTransform(scale=1.2, rotation=0.1, translation=(1, 1)),
    ).F
    del stack
    return F, note


@pytest.fixture()
def out():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "out"
        d.mkdir(parents=True)
        _fixture(d)
        yield d


def _artifact_corrs(out, cfg):
    """|corr| of every IC map with the stripe, in IC order."""
    s = PipelineSession(cfg, ica_provider=FakeIcaProvider([]))
    s.run_pca()
    s.run_ica()
    ica = s.state.ica_results["BL"]
    art = np.zeros((H, W))
    art[:, 2:5] = 1.0
    maps = ica.spatial.reshape(ica.n_components, -1)
    return np.array([abs(np.corrcoef(m, art.ravel())[0, 1]) for m in maps])


def _artifact_ic(out, cfg):
    return int(np.argmax(_artifact_corrs(out, cfg)))


def _artifact_ics(out, cfg, rel=0.5):
    """Every IC that carries the stripe, not only the strongest one.

    How many components the artifact lands in is a property of the
    decomposition, not of the wiring under test, and it differs with the BLAS
    FastICA runs on: one component here, two on the CI runner. A user looking
    at the IC maps ticks all of the ones that show the artifact, so the test
    does too -- otherwise it is really asserting that ICA separated the
    stripe perfectly, which is not this file's contract.
    """
    c = _artifact_corrs(out, cfg)
    return [int(i) for i in np.flatnonzero(c >= rel * c.max())]


def test_off_is_the_default_and_reads_the_plain_dff(out):
    cfg = _config(out)
    assert cfg.ica_denoise == "off"
    _, note = _roi_signals(cfg, out)
    assert "ICA" not in note
    assert not denoised_dff_path(out, "BL").exists()


def test_subtract_with_nothing_excluded_changes_nothing(out):
    """Turning the switch on cannot, by itself, move a number."""
    plain, _ = _roi_signals(_config(out), out)

    cfg = _config(out, ica_denoise="subtract")
    PipelineSession(cfg, ica_provider=FakeIcaProvider([])).run_ica()
    got, note = _roi_signals(cfg, out)

    assert "ICA-denoised" in note
    np.testing.assert_allclose(got, plain, rtol=0, atol=1e-5)


def test_subtract_removes_the_artifact_from_the_roi_signals(out):
    cfg0 = _config(out)
    art_ics = _artifact_ics(out, cfg0)
    plain, _ = _roi_signals(cfg0, out)

    cfg = _config(out, ica_denoise="subtract")
    PipelineSession(cfg, ica_provider=FakeIcaProvider(art_ics)).run_ica()
    got, _ = _roi_signals(cfg, out)

    t = np.arange(T)
    course = 2.0 * np.sin(t / 7)

    def carried(F):  # how much of the artifact's time course the ROI trace carries
        v = F[0] - F[0].mean()
        return abs(float(np.dot(v, course - course.mean())
                         / np.dot(course - course.mean(), course - course.mean())))

    assert carried(got) < 0.2 * carried(plain), (
        f"excluded ICs {art_ics}: {carried(plain):.3f} -> {carried(got):.3f}"
    )


def test_provenance_is_stamped_on_the_saved_payloads(out):
    cfg = _config(out, ica_denoise="subtract", annotation=False)
    art_ic = _artifact_ic(out, _config(out))
    s = PipelineSession(cfg, ica_provider=FakeIcaProvider([art_ic]))
    s.run_ica()

    # ROI payload (the atlas is not needed: extract via the adjoint on masks)
    from asvimg.extraction import extract_and_save_roi_signals
    from asvimg.atlas import load_atlas

    atlas_path = TEST_ATLAS_OR_DEFAULT
    if not atlas_path.exists():
        pytest.skip("atlas not available")
    atlas = load_atlas(atlas_path)
    extract_and_save_roi_signals(
        cfg, out, SimilarityTransform(scale=1.0), atlas,
    )
    payload = load_payload(next(out.glob("roiSignals_BL_*.npy")))
    assert int(payload["ica_denoised"]) == 1
    assert str(payload["ica_denoise_mode"]).strip() == "subtract"
    # 1-based, like the IC{i}.png the user clicked and the sidecar's "IC3"
    assert list(np.asarray(payload["ica_excluded_ic"]).ravel()) == [art_ic + 1]


def test_a_missing_basis_raises_instead_of_serving_plain_dff(out):
    """The one failure mode that must never be silent."""
    cfg = _config(out, ica_denoise="subtract")
    with pytest.raises(FileNotFoundError, match="run the ICA stage"):
        _roi_signals(cfg, out)


def test_a_stale_basis_raises(out):
    cfg = _config(out, ica_denoise="subtract")
    PipelineSession(cfg, ica_provider=FakeIcaProvider([1])).run_ica()
    _roi_signals(cfg, out)  # builds ica/dff_BL.npy

    # the dF/F is rewritten underneath: the IC numbers no longer mean the same thing
    np.save(out / "dff_BL.npy", np.zeros((T, H, W), dtype=np.float32))
    with pytest.raises(RuntimeError, match="stale"):
        _roi_signals(cfg, out)


def test_the_denoised_stack_is_rebuilt_when_the_choice_changes(out):
    cfg = _config(out, ica_denoise="subtract")
    PipelineSession(cfg, ica_provider=FakeIcaProvider([0])).run_ica()
    first, _ = _roi_signals(cfg, out)

    # a different exclusion, same basis -> the stack must be rebuilt, not reused
    save_exclusions(out, {"BL": [1]})
    second, _ = _roi_signals(cfg, out)
    assert not np.allclose(first, second)
