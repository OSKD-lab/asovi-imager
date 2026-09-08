"""End-to-end: run the REAL pipeline (preprocess -> roi) on a synthetic recording,
then export NWB and validate it.

This closes the one seam the unit tests fabricate: preprocess.  Every artifact the
NWB writer consumes here (reg_meta.npz, dff_BL.npy, roiSignals_*) is produced by
the actual pipeline stages, not hand-written.

(_sampleData01 — the maintainer's real recordings — is gitignored and absent in
this environment, so the *input frames* are synthetic; the pipeline that turns
them into outputs is the real one.)
"""

import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
pytest.importorskip("pynwb")
pytest.importorskip("nwbinspector")

from asvimg import PipelineConfig, Roi, SOURCE_ROIS_FILENAME, save_rois
from asvimg import nwb_meta, nwb_export
from asvimg.io import read_reg_meta
from pynwb import NWBHDF5IO


def _make_recording(inp: Path, n_cycles: int = 120, h: int = 64, w: int = 80) -> None:
    """A 2-channel (source/donner) interleaved TIFF: a blob with slow 'activity'."""
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:h, 0:w]
    blob = np.exp(-(((xx - w / 2) / 18) ** 2 + ((yy - h / 2) / 14) ** 2))
    frames = np.empty((2 * n_cycles, h, w), np.uint16)
    gain = 1.0 + 0.15 * np.sin(np.arange(n_cycles) / 7.0)
    for k in range(n_cycles):
        s = 1500 * blob * gain[k] + 300 + rng.normal(0, 10, (h, w))
        d = 1400 * blob + 300 + rng.normal(0, 10, (h, w))
        frames[2 * k] = np.clip(s, 0, 65535).astype(np.uint16)      # source
        frames[2 * k + 1] = np.clip(d, 0, 65535).astype(np.uint16)  # donner
    tifffile.imwrite(inp / "rec_timelapseE2E_MM.tif", frames)


@pytest.fixture()
def workdir():
    td = tempfile.mkdtemp()
    try:
        yield Path(td)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_pipeline_to_nwb_end_to_end(workdir):
    inp = workdir / "in"
    inp.mkdir()
    _make_recording(inp)
    out = workdir / "asi" / "npy"
    out.mkdir(parents=True)

    cfg = PipelineConfig(
        input_dir=str(inp), output_dir=str(out), output_format="npy",
        channels_name=["BL", "BL"], channels_prop=["source", "donner"],
        fps=20, binning=2, roi_space="source", roi_signal="Both",
        annotation=False, demux_qc=False, hemovar_qc=False, save_figures="none",
    )

    from asvimg.runner import PipelineSession

    PipelineSession(cfg).run_preprocess()  # REAL preprocess -> reg_Ch/dff/reg_meta
    assert (out / "reg_meta.npz").exists()
    assert (out / "dff_BL.npy").exists()

    H, W = (int(x) for x in read_reg_meta(out)["imageSize"])
    save_rois(out / SOURCE_ROIS_FILENAME, [
        Roi("center", x=W // 2, y=H // 2, size=6),
        Roi("corner", x=6, y=6, size=6),
    ])
    PipelineSession(cfg).run_roi()  # REAL roi -> roiSignals_BL_E2E.npy (fresh session)
    assert list(out.glob("roiSignals_BL_*.npy"))

    meta = nwb_meta.NwbMetadata()
    meta.session.session_description = "E2E synthetic WFCI"
    meta.session.session_start_time = "2026-07-18T10:00:00+09:00"
    meta.subject.subject_id = "e2e-01"
    meta.subject.sex = "M"
    meta.subject.age = "P60D"
    meta.channels["BL"] = nwb_meta.ChannelMeta(
        excitation_lambda=470.0, emission_lambda=525.0, indicator="GCaMP6s")
    meta.pixel_size_um = 20.0
    meta.options = nwb_meta.ExportOptions(
        include_dff=True, include_roi=True, include_reference_images=True,
        include_correlation=True, include_warped=False, include_ica=False)

    path = nwb_export.write_nwb(cfg, out, meta)
    assert path.exists()
    assert path.suffix == ".nwb" and path.parent == out  # default name is <exp-stem>.nwb in output_dir

    with NWBHDF5IO(str(path), "r") as io:
        f = io.read()
        oph = f.processing["ophys"]
        assert "OnePhotonSeries_dff_BL" in oph.data_interfaces
        assert "DfOverF" in oph.data_interfaces       # F_dff
        assert "Fluorescence" in oph.data_interfaces  # F_raw (roi_signal=Both)
        assert "reference_images" in oph.data_interfaces
        assert tuple(oph["OnePhotonSeries_dff_BL"].data.shape)[1:] == (H, W)
        assert "roi_correlation_BL" in f.scratch
        assert f.subject.subject_id == "e2e-01"

    import nwbinspector as ni
    from nwbinspector import inspect_nwbfile

    msgs = list(inspect_nwbfile(nwbfile_path=str(path), config=ni.load_config("dandi")))
    crit = [f"{m.check_function_name}: {m.message}"
            for m in msgs if "CRITICAL" in str(m.importance)]
    assert crit == [], f"nwbinspector CRITICAL: {crit}"
