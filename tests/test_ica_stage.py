"""The ICA stage: one decomposition per channel group, and a decision that survives.

PCA/ICA used to run on a single channel. They now run on every group with a
source, and the two things that come out of the stage are persisted separately:
the BASIS (derived, re-computable) and the human DECISION (not re-computable).
"""

import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from asvimg import (
    PipelineConfig,
    basis_is_current,
    load_exclusions,
    reg_channel_path,
    write_reg_meta,
)
from asvimg.ica_state import basis_path, exclusion_path, save_exclusions
from asvimg.runner import FakeIcaProvider, PipelineSession
from asvimg.runner.providers import ConfigIcaProvider


def _fixture(out: Path, H=14, W=16, T=24, seed=0):
    """4-ch cycle: GCaMP (donner-backed) + jRGECO (source-only, TWO sources)."""
    rng = np.random.default_rng(seed)
    names = ["GCaMP", "jRGECO", "GCaMP", "jRGECO"]
    props = ["donner", "source", "source", "source"]
    meta = {
        "proc_template": np.zeros((H, W)),
        "imageSize": np.array([H, W]),
        "channels_name": np.array(names),
        "channels_prop": np.array(props),
        "T_per_ch": np.array([T] * 4, dtype=np.int64),
        "fps": np.array(40),
    }
    for i in range(4):
        arr = rng.integers(400, 3000, size=(T, H, W)).astype(np.uint16)
        np.save(reg_channel_path(out, i), arr)
        meta[f"meanImageCh{i}"] = arr.mean(axis=0)
    write_reg_meta(out, meta)
    np.save(out / "dff_GCaMP.npy", (rng.standard_normal((T, H, W)) * 0.02).astype(np.float32))
    np.save(out / "dff_jRGECO.npy", (rng.standard_normal((T, H, W)) * 0.02).astype(np.float32))


def _config(out: Path) -> PipelineConfig:
    return PipelineConfig(
        input_dir=str(out.parent / "in"), output_dir=str(out), output_format="npy",
        exp_name="TEST",
        channels_name=["GCaMP", "jRGECO", "GCaMP", "jRGECO"],
        channels_prop=["donner", "source", "source", "source"],
        fps=40, ch_for_annotation=0,
        pca_n_components=4, ica_n_components=4, pca_skip_frames=1,
        annotation=False,
    )


@pytest.fixture()
def out():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "out"
        d.mkdir(parents=True)
        _fixture(d)
        yield d


def test_every_group_gets_its_own_decomposition(out):
    fake = FakeIcaProvider({"GCaMP": [1], "jRGECO": []})
    s = PipelineSession(_config(out), ica_provider=fake)
    s.run_pca()
    s.run_ica()

    assert sorted(s.state.pca_results) == ["GCaMP", "jRGECO"]
    assert sorted(s.state.ica_results) == ["GCaMP", "jRGECO"]
    # the picker was opened once per group, and told which of how many
    assert [c[0] for c in fake.calls] == ["GCaMP", "jRGECO"]
    assert [c[2] for c in fake.calls] == [2, 2]

    # maps live under per-group subdirs, so two groups' IC1 cannot collide
    assert (out / "pca_images" / "GCaMP" / "PC1.png").exists()
    assert (out / "pca_images" / "jRGECO" / "PC1.png").exists()
    assert (out / "ica_images" / "GCaMP" / "IC1.png").exists()
    assert (out / "ica_images" / "jRGECO" / "IC1.png").exists()


def test_basis_and_decision_are_persisted_separately(out):
    cfg = _config(out)
    s = PipelineSession(cfg, ica_provider=FakeIcaProvider({"GCaMP": [1, 2], "jRGECO": []}))
    s.run_pca()
    s.run_ica()

    for name in ("GCaMP", "jRGECO"):
        assert basis_path(out, name).exists()
        ok, why = basis_is_current(out, name, cfg)
        assert ok, why

    # the decision: labels, not bare ints (the maps on screen are IC1-based)
    payload = json.loads(exclusion_path(out).read_text(encoding="utf-8"))
    assert payload["groups"]["GCaMP"]["excluded"] == ["IC2", "IC3"]
    assert payload["groups"]["jRGECO"]["excluded"] == []      # reviewed, nothing flagged
    assert load_exclusions(out) == {"GCaMP": [1, 2], "jRGECO": []}


def test_a_stale_basis_is_detected(out):
    cfg = _config(out)
    s = PipelineSession(cfg, ica_provider=FakeIcaProvider([]))
    s.run_pca()
    s.run_ica()
    assert basis_is_current(out, "GCaMP", cfg)[0]

    # a parameter that changes the decomposition -> the IC numbers mean something else
    cfg2 = _config(out)
    cfg2.pca_n_components = 3
    ok, why = basis_is_current(out, "GCaMP", cfg2)
    assert not ok and "pca_n_components" in why

    # the dF/F itself was rewritten (preprocess re-ran)
    np.save(out / "dff_GCaMP.npy", np.zeros((24, 14, 16), dtype=np.float32))
    ok, why = basis_is_current(out, "GCaMP", cfg)
    assert not ok and "changed" in why


def test_a_cancelled_picker_keeps_the_previous_choice(out):
    """None is not []. A crashed or closed IC window must not be recorded as
    'reviewed, nothing to exclude' — that would stamp un-denoised data as clean."""
    cfg = _config(out)
    cfg.ica_exclusion = "gui"          # "gui" always asks; "cache" would replay
    save_exclusions(out, {"GCaMP": [3], "jRGECO": [0]})

    s = PipelineSession(cfg, ica_provider=FakeIcaProvider({"GCaMP": None, "jRGECO": [2]}))
    s.run_pca()
    s.run_ica()

    assert load_exclusions(out) == {"GCaMP": [3], "jRGECO": [2]}


def test_cache_replays_instead_of_re_asking(out):
    """With a choice on record, ica_exclusion="cache" must NOT re-open the picker —
    the same rule annotation="cache" follows for marks.mat."""
    cfg = _config(out)                  # ica_exclusion defaults to "cache"
    save_exclusions(out, {"GCaMP": [3], "jRGECO": []})

    fake = FakeIcaProvider({"GCaMP": [0]})
    s = PipelineSession(cfg, ica_provider=fake)
    s.run_ica()

    assert fake.calls == []                              # never asked
    assert s.state.excluded_by_group["GCaMP"] == [3]     # replayed the record


def test_a_group_the_config_does_not_answer_for_is_not_overwritten(out):
    """A dict that names only one group has said nothing about the other. Recording
    [] for it would destroy a human's judgement that cannot be re-derived."""
    cfg = _config(out)
    save_exclusions(out, {"GCaMP": [1], "jRGECO": [2]})
    cfg.ica_exclusion = {"GCaMP": [0]}   # says nothing about jRGECO

    PipelineSession(cfg).run_ica()

    assert load_exclusions(out) == {"GCaMP": [0], "jRGECO": [2]}


def test_headless_replays_the_recorded_decision(out):
    """The GUI records a choice; a headless re-run reproduces it without a window."""
    cfg = _config(out)
    PipelineSession(cfg, ica_provider=FakeIcaProvider({"GCaMP": [2], "jRGECO": []})).run_ica()
    assert load_exclusions(out)["GCaMP"] == [2]

    # ica_exclusion defaults to "cache" -> ConfigIcaProvider replays the sidecar
    s2 = PipelineSession(cfg)
    assert isinstance(s2.ica_provider, ConfigIcaProvider)
    s2.run_ica()
    assert s2.state.excluded_by_group["GCaMP"] == [2]
    assert s2.state.excluded_ics == [2]


def test_inline_exclusion_in_the_config(out):
    cfg = _config(out)
    cfg.ica_exclusion = {"GCaMP": [0, 3]}
    s = PipelineSession(cfg)
    s.run_ica()
    assert s.state.excluded_by_group["GCaMP"] == [0, 3]
    # jRGECO was not answered for -> no record is written for it (see the test
    # above); what gets APPLIED for it is still nothing.
    from asvimg.ica_state import resolve_exclusion

    assert resolve_exclusion(cfg, out, "jRGECO") == []


def test_gui_mode_is_rejected_headlessly(out):
    cfg = _config(out)
    cfg.ica_exclusion = "gui"
    with pytest.raises(RuntimeError, match="needs a display"):
        PipelineSession(cfg).run_ica()


# --------------------------------------------------------------------------
# Regressions found by an adversarial review of this feature
# --------------------------------------------------------------------------


def test_preprocess_does_not_destroy_the_decision(out):
    """THE bug: preprocess delete=True (the default) used to remove
    ica_exclusion.json, so Run All / asovi-run destroyed the human's judgement in
    stage 1 and the ICA stage then re-recorded it as "reviewed, flagged nothing"."""
    import tifffile

    from asvimg.preprocess import PreprocessRunner

    save_exclusions(out, {"GCaMP": [2]})

    inp = out.parent / "in"
    inp.mkdir(exist_ok=True)
    tifffile.imwrite(
        inp / "rec.tif",
        np.random.default_rng(0).integers(400, 3000, size=(16, 8, 10)).astype(np.uint16),
    )
    cfg = PipelineConfig(
        input_dir=str(inp), output_dir=str(out), output_format="npy",
        channels_name=["GCaMP", "GCaMP"], channels_prop=["source", "donner"],
        do_registration=False, binning=1, delete=True,
        output_metadata_yaml=False, demux_qc=False, hemovar_qc=False,
    )
    PreprocessRunner(cfg).run()

    assert exclusion_path(out).exists()               # the decision survives...
    assert load_exclusions(out) == {"GCaMP": [2]}
    assert not basis_path(out, "GCaMP").exists()      # ...the derived basis does not


def test_a_corrupt_decision_file_raises(out):
    """Reading a corrupt record as "exclude nothing" would hand back un-denoised
    numbers while every payload claims they are denoised."""
    exclusion_path(out).write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        load_exclusions(out)


def test_the_fingerprint_survives_a_copied_output_folder(out):
    """An output dir copied to another machine keeps its bytes, not its mtimes. An
    mtime-keyed fingerprint would call every basis stale — i.e. refuse to run."""
    import shutil

    from asvimg.ica_state import source_fingerprint

    cfg = _config(out)
    PipelineSession(cfg, ica_provider=FakeIcaProvider([])).run_ica()
    before = source_fingerprint(out, "GCaMP")

    copied = out.parent / "copy"
    shutil.copytree(out, copied)
    for p in copied.rglob("*"):          # simulate the timestamps a copy/restore gives
        if p.is_file():
            import os
            os.utime(p, (1_000_000, 1_000_000))

    assert source_fingerprint(copied, "GCaMP") == before
    ok, why = basis_is_current(copied, "GCaMP", cfg)
    assert ok, why


def test_a_display_knob_does_not_invalidate_the_basis(out):
    """pca_smooth_sigma only smooths the plotted traces (the ICA reads
    temporal_raw), so it must not force ROI/export to be rebuilt."""
    cfg = _config(out)
    PipelineSession(cfg, ica_provider=FakeIcaProvider([])).run_ica()

    cfg2 = _config(out)
    cfg2.pca_smooth_sigma = 12.0
    ok, why = basis_is_current(out, "GCaMP", cfg2)
    assert ok, why
