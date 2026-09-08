"""Display-independent smoke test for the standalone NWB builder GUI.

Builds the editor in a headless dpg context (like test_gui_smoke's demux test),
drives the callbacks (validate / save sidecar / threaded write / inspector), and
asserts the produced file is DANDI-clean.
"""

import shutil
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
from asvimg.config import save_config

pytest.importorskip("pynwb")
pytest.importorskip("nwbinspector")

H, W, T = 20, 24, 80


def _dpg_ok() -> bool:
    try:
        import dearpygui.dearpygui as dpg

        dpg.create_context()
        dpg.destroy_context()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _dpg_ok(), reason="dearpygui unavailable (headless)")


def _fixture(out: Path):
    rng = np.random.default_rng(0)
    t = np.arange(T)
    dff = (rng.normal(size=(H, W))[:, :, None] * np.sin(t / 11)[None, None, :]
           + rng.normal(0, 0.05, size=(H, W, T)))
    meta = {
        "proc_template": np.zeros((H, W)), "imageSize": np.array([H, W]),
        "channels_name": np.array(["BL", "BL"]), "channels_prop": np.array(["source", "donner"]),
        "T_per_ch": np.array([T, T], dtype=np.int64), "fps": np.array(20),
    }
    for i in range(2):
        arr = rng.integers(400, 3000, size=(T, H, W)).astype(np.uint16)
        np.save(reg_channel_path(out, i), arr)
        meta[f"meanImageCh{i}"] = arr.mean(axis=0)
    write_reg_meta(out, meta)
    np.save(out / "dff_BL.npy", np.moveaxis(dff, 2, 0).astype(np.float32))


def _config(out: Path) -> PipelineConfig:
    return PipelineConfig(
        input_dir=str(out.parent / "in"), output_dir=str(out), output_format="npy",
        exp_name="SRC", channels_name=["BL", "BL"], channels_prop=["source", "donner"],
        fps=20, roi_signal="dff", roi_space="source", annotation=False)


def _run_roi(out: Path, rois):
    save_rois(out / SOURCE_ROIS_FILENAME, rois)
    from asvimg.runner import PipelineSession

    PipelineSession(_config(out)).run_roi()


@pytest.fixture()
def out():
    td = tempfile.mkdtemp()
    d = Path(td) / "out"
    d.mkdir(parents=True)
    try:
        yield d
    finally:
        shutil.rmtree(td, ignore_errors=True)


@pytest.fixture()
def dpg_ctx():
    import dearpygui.dearpygui as dpg

    dpg.create_context()
    dpg.create_viewport()
    dpg.setup_dearpygui()
    try:
        yield dpg
    finally:
        dpg.destroy_context()


def test_editor_build_validate_save_write_inspect(out, dpg_ctx):
    dpg = dpg_ctx
    _fixture(out)
    save_config(_config(out), out)  # a realistic processed folder (db.yaml/ops.yaml)
    _run_roi(out, [Roi("left", x=6, y=5, size=5), Roi("right", x=17, y=13, size=5)])

    from asvimg.gui.nwb_editor import NwbEditor, _WIN, _STATUS, _LOG, _PROGRESS

    ed = NwbEditor(_config(out), out)
    ed.build()
    for tag in (_WIN, _STATUS, _LOG, _PROGRESS):
        assert dpg.does_item_exist(tag), tag
    assert "BL" in ed._channel_names
    assert dpg.does_item_exist(ed._ft("ch_BL_emission"))

    # empty Subject -> validate flags DANDI-critical problems
    ed._cb_validate()
    assert "critical" in dpg.get_value(_STATUS).lower()

    # fill the required DANDI fields + channel optics, then validate is clean
    dpg.set_value(ed._ft("f_subject_id"), "mouse-02")
    dpg.set_value(ed._ft("f_sex"), "M")
    dpg.set_value(ed._ft("f_age"), "P90D")
    dpg.set_value(ed._ft("ch_BL_excitation"), "470")
    dpg.set_value(ed._ft("ch_BL_emission"), "525")
    ed._cb_validate()
    assert "valid" in dpg.get_value(_STATUS).lower()

    # save sidecar, then collect_meta round-trips the widget values
    ed._cb_save_meta()
    assert (out / "nwb_metadata.yaml").exists()
    m = ed.collect_meta()
    assert m.subject.subject_id == "mouse-02"
    assert m.channels["BL"].emission_lambda == 525.0

    # threaded write, then drain onto the (headless) widgets
    ed._cb_write()
    assert ed._write_thread is not None
    ed._write_thread.join(timeout=180)
    ed._drain()
    assert ed.last_written is not None and ed.last_written.exists()

    # inspector button -> DANDI-clean
    ed._cb_inspect()
    ed._inspect_thread.join(timeout=120)
    ed._drain()
    assert "CRITICAL" in dpg.get_value(_STATUS)

    import nwbinspector as ni
    from nwbinspector import inspect_nwbfile

    msgs = list(inspect_nwbfile(nwbfile_path=str(ed.last_written), config=ni.load_config("dandi")))
    crit = [f"{m.check_function_name}: {m.message}"
            for m in msgs if "CRITICAL" in str(m.importance)]
    assert crit == [], f"nwbinspector CRITICAL: {crit}"


def test_editor_reload_rebuilds_window(out, dpg_ctx):
    dpg = dpg_ctx
    _fixture(out)
    save_config(_config(out), out)

    from asvimg.gui.nwb_editor import NwbEditor, _WIN

    ed = NwbEditor(_config(out), out)
    ed.build()
    assert dpg.does_item_exist(_WIN)
    ed._cb_reload()  # _load_dir -> reload config+meta -> rebuild
    assert dpg.does_item_exist(_WIN)
    assert "BL" in ed._channel_names
