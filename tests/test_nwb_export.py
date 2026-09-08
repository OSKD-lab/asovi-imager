"""asovi-nwb export: build an NWB from processed outputs, DANDI-clean.

Mirrors test_source_space_roi's fixture (a source-space group with a known dF/F);
runs the real ROI stage to produce roiSignals, then packages everything with
write_nwb and asserts the file reads back and nwbinspector --config dandi returns
zero CRITICAL messages (the gate DANDI upload enforces).
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
    Roi,
    SOURCE_ROIS_FILENAME,
    reg_channel_path,
    save_rois,
    write_reg_meta,
)
from asvimg import nwb_meta, nwb_export

pytest.importorskip("pynwb")
pytest.importorskip("nwbinspector")
from pynwb import NWBHDF5IO  # noqa: E402

H, W, T = 20, 24, 80


def _fixture(out: Path):
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


def _good_meta() -> nwb_meta.NwbMetadata:
    m = nwb_meta.NwbMetadata()
    m.session.session_description = "WFCI smoke"
    m.session.session_start_time = "2026-07-18T10:30:00+09:00"
    m.session.experimenter = ["Takeuchi, Ryosuke"]
    m.subject.subject_id = "mouse-02"
    m.subject.species = "Mus musculus"
    m.subject.sex = "M"
    m.subject.age = "P90D"
    m.channels["BL"] = nwb_meta.ChannelMeta(
        excitation_lambda=470.0, emission_lambda=525.0, indicator="GCaMP6s")
    m.pixel_size_um = 20.0
    return m


def _run_roi(out: Path, rois):
    save_rois(out / SOURCE_ROIS_FILENAME, rois)
    from asvimg.runner import PipelineSession

    PipelineSession(_config(out)).run_roi()


def _criticals(path: Path):
    import nwbinspector as ni
    from nwbinspector import inspect_nwbfile

    try:
        msgs = list(inspect_nwbfile(nwbfile_path=str(path), config=ni.load_config("dandi")))
    except Exception:
        msgs = list(inspect_nwbfile(nwbfile_path=str(path)))
    return [f"{m.check_function_name}: {m.message}"
            for m in msgs if "CRITICAL" in str(m.importance)]


@pytest.fixture()
def out():
    # mkdtemp + tolerant rmtree: on Windows h5py/nwbinspector can hold the .nwb
    # open past the test, and TemporaryDirectory's strict cleanup would error.
    import shutil

    td = tempfile.mkdtemp()
    d = Path(td) / "out"
    d.mkdir(parents=True)
    try:
        yield d
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_meta_sidecar_roundtrip_and_prefill(out):
    _fixture(out)
    m = nwb_meta.prefill(_config(out), out)
    assert "BL" in m.channels            # one channel row per source group
    assert m.session.session_id          # seeded from the experiment stem

    m.subject.subject_id = "m1"
    assert nwb_meta.save_nwb_meta(out, m).exists()
    back = nwb_meta.load_nwb_meta(out)
    assert back.subject.subject_id == "m1"
    assert isinstance(back.options, nwb_meta.ExportOptions)

    probs = nwb_meta.validate(back)
    assert any("session_start_time" in p for p in probs)   # still-missing DANDI field flagged


def test_write_source_roi_is_dandi_clean(out):
    _fixture(out)
    _run_roi(out, [Roi("left", x=6, y=5, size=5), Roi("right", x=17, y=13, size=5)])
    m = _good_meta()
    m.options = nwb_meta.ExportOptions(
        include_dff=True, include_reference_images=True, include_roi=True,
        include_warped=False, include_ica=False, include_correlation=False)

    path = nwb_export.write_nwb(_config(out), out, m)
    assert path.exists()

    with NWBHDF5IO(str(path), "r") as io:
        f = io.read()
        oph = f.processing["ophys"]
        assert "OnePhotonSeries_dff_BL" in oph.data_interfaces
        assert tuple(oph["OnePhotonSeries_dff_BL"].data.shape) == (T, H, W)
        assert "ImageSegmentation" in oph.data_interfaces
        assert "DfOverF" in oph.data_interfaces
        assert "reference_images" in oph.data_interfaces
        assert f.subject.subject_id == "mouse-02"
        assert f.subject.sex == "M"

    crit = _criticals(path)
    assert crit == [], f"nwbinspector CRITICAL: {crit}"


def test_write_correlation_and_ica(out):
    _fixture(out)
    _run_roi(out, [Roi("a", x=6, y=5, size=5), Roi("b", x=17, y=13, size=5),
                   Roi("c", x=10, y=10, size=5)])

    from asvimg.ica_state import basis_path

    n_ic = 4
    bp = basis_path(out, "BL")
    bp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        bp,
        spatial=np.random.rand(n_ic, H, W).astype("float32"),
        temporal=np.random.rand(n_ic, T).astype("float32"),
        mean_image=np.random.rand(H, W),
        temporal_variance=np.linspace(1.0, 0.1, n_ic),
        frame_indices=np.arange(T, dtype=np.int64),
        n_components=np.int64(n_ic),
        params=json.dumps({"pca_n_components": n_ic}),
        source_fingerprint="test",
    )

    m = _good_meta()
    m.options = nwb_meta.ExportOptions(
        include_dff=True, include_roi=True, include_reference_images=False,
        include_correlation=True, include_ica=True)

    path = nwb_export.write_nwb(_config(out), out, m, out_path=out / "corr.nwb")
    with NWBHDF5IO(str(path), "r") as io:
        f = io.read()
        assert "roi_correlation_BL" in f.scratch
        assert tuple(f.scratch["roi_correlation_BL"].data.shape) == (3, 3)
        assert "RoiResponseSeries_ICA_BL" in f.processing["ophys"].data_interfaces

    crit = _criticals(path)
    assert crit == [], f"nwbinspector CRITICAL: {crit}"


def test_overwrite_guard(out):
    _fixture(out)
    m = _good_meta()
    m.options = nwb_meta.ExportOptions(
        include_dff=True, include_roi=False, include_reference_images=False)
    p1 = nwb_export.write_nwb(_config(out), out, m, out_path=out / "x.nwb")
    with pytest.raises(FileExistsError):
        nwb_export.write_nwb(_config(out), out, m, out_path=out / "x.nwb")
    m.options.overwrite = True
    p2 = nwb_export.write_nwb(_config(out), out, m, out_path=out / "x.nwb")
    assert p1 == p2 and p2.exists()
