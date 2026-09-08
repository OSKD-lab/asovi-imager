"""A headless run must reproduce an interactive one, bit for bit.

Two decisions in this pipeline cannot be re-derived — where the atlas control
points go (``marks.mat``) and which components are artifacts
(``ica_exclusion.json``).  Everything else is a function of the inputs.  So the
contract is: keep those two files, throw away every derived artifact, run with no
windows at all, and get the same numbers.

That is what makes ``asovi-run`` usable for a batch of animals after reviewing one
of them by hand, and what makes a published result reproducible from what is on
disk.
"""

import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import tifffile

from asvimg.config import BUNDLED_ATLAS
from asvimg import PipelineConfig, load_payload
from asvimg.ica_state import load_exclusions, source_fingerprint
from asvimg.runner import FakeIcaProvider, FakeMarksProvider, PipelineSession

_ATLAS = BUNDLED_ATLAS


def _make_input(inp: Path, T=160, H=48, W=52):
    """2-ch (source/donner) recording whose source carries a stripe artifact."""
    rng = np.random.default_rng(0)
    t = np.arange(T // 2)
    art = np.zeros((H, W))
    art[:, 6:12] = 1.0
    stack = np.empty((T, H, W), np.uint16)
    for i in range(T):
        img = (1500.0 if i % 2 == 0 else 900.0) + rng.normal(0, 12, (H, W))
        if i % 2 == 0:
            img = img + 60.0 * art * np.sin(t[i // 2] / 9)
        stack[i] = np.clip(img, 0, 65535)
    tifffile.imwrite(inp / "rec.tif", stack)


def _config(inp: Path, out: Path, **over) -> PipelineConfig:
    cfg = PipelineConfig(
        input_dir=str(inp), output_dir=str(out), output_format="npy", exp_name="REPRO",
        channels_name=["BL", "BL"], channels_prop=["source", "donner"],
        fps=20, ch_for_annotation=0,
        do_registration=False, binning=1, demux_qc=False, hemovar_qc=False,
        output_metadata_yaml=False,
        pca_n_components=6, ica_n_components=6, pca_skip_frames=2,
        annotation="gui", ica_exclusion="gui",   # session A: a human decides
        ica_denoise="subtract",                  # ...and it must reach the outputs
        save_annotated_dF_mat=True, save_movie=False, save_figures="none",
        delete=True,
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


@pytest.mark.skipif(not _ATLAS.exists(), reason="atlas not available")
def test_headless_reproduces_the_interactive_session():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        inp = tmp / "in"
        inp.mkdir()
        _make_input(inp)
        out = tmp / "out"

        # --- A: the interactive session (cpselect + the IC picker) ---------
        a = PipelineSession(
            _config(inp, out),
            marks_provider=FakeMarksProvider(
                [[8, 8], [8, 40], [38, 24]], [[110, 110], [110, 190], [210, 150]]
            ),
            ica_provider=FakeIcaProvider([1]),   # the human flags IC2
        )
        a.run_preprocess()
        a.run_pca()
        a.run_ica()
        a.run_annotation()
        a.run_roi()
        a.run_export()

        F_a = np.asarray(load_payload(next(out.glob("roiSignals_BL_*.npy")))["F_dff"])
        warped_a = np.asarray(load_payload(next(out.glob("*dfWarped_BL*.npy")))["imageDf"])
        fp_a = source_fingerprint(out, "BL")
        assert load_exclusions(out) == {"BL": [1]}

        # --- throw away everything that is derivable ----------------------
        keep = {"marks.mat", "ica_exclusion.json", "db.yaml", "ops.yaml"}
        for p in sorted(out.iterdir()):
            if p.name in keep:
                continue
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink()

        # --- B: no windows, no providers ----------------------------------
        b = PipelineSession(_config(inp, out, annotation="cache", ica_exclusion="cache"))
        b.run_all()

        F_b = np.asarray(load_payload(next(out.glob("roiSignals_BL_*.npy")))["F_dff"])
        warped_b = np.asarray(load_payload(next(out.glob("*dfWarped_BL*.npy")))["imageDf"])

        assert source_fingerprint(out, "BL") == fp_a   # preprocess re-derived the same dF/F
        assert load_exclusions(out) == {"BL": [1]}     # the human's choice was replayed
        np.testing.assert_array_equal(F_b, F_a)        # ...and it reached the numbers
        np.testing.assert_array_equal(warped_b, warped_a)
