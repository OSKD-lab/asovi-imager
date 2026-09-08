
"""Display-independent smoke tests for the GUI layer.

Covers everything verifiable without a real window: lazy dearpygui isolation,
IPC round-trip, the config form (built in a headless dpg context), the bridge
event queue, and the interactive subprocess plumbing via a mock child.  The
actual cpselect / ICA windows and the live render loop require a display and
are out of scope here.
"""

from __future__ import annotations

from asvimg.config import BUNDLED_ATLAS

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestGuiLazyImport(unittest.TestCase):
    def test_importing_gui_does_not_import_dearpygui(self) -> None:
        # Fresh subprocess so we measure a clean import.
        from asvimg.gui import ipc

        code = (
            "import sys; import asvimg.gui, asvimg.gui.app, "
            "asvimg.gui.bridge, asvimg.gui.config_form; "
            "print('dpg' if 'dearpygui.dearpygui' in sys.modules else 'clean')"
        )
        p = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ipc.project_root()), env=ipc.child_env(),
            capture_output=True, text=True,
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("clean", p.stdout)


class TestIpc(unittest.TestCase):
    def test_roundtrip_and_project_root(self) -> None:
        import numpy as np

        from asvimg.gui import ipc

        d = ipc.new_session_dir()
        try:
            p = d / "x.pkl"
            ipc.dump({"a": np.arange(3), "b": [np.zeros((2, 2))]}, p)
            back = ipc.load(p)
            self.assertEqual(list(back["a"]), [0, 1, 2])
        finally:
            import shutil

            shutil.rmtree(d, ignore_errors=True)
        self.assertTrue((ipc.project_root() / "pyproject.toml").exists())


class TestBridgeQueue(unittest.TestCase):
    def test_reporter_enqueue_drain_order(self) -> None:
        from asvimg.gui.bridge import GuiBridge

        bridge = GuiBridge()
        bridge.on_log("x", "line1")
        bridge.on_progress("pca", 2, 5)
        bridge.on_stage("pca", "done", elapsed=1.0)
        drained: list = []
        bridge.drain(drained.append)
        self.assertEqual([e[0] for e in drained], ["log", "progress", "stage"])

    def test_run_child_happy_path(self) -> None:
        from asvimg.gui.bridge import GuiBridge

        bridge = GuiBridge()
        with tempfile.TemporaryDirectory() as td:
            mock = Path(td) / "mock_fast_child.py"
            mock.write_text(
                "import sys\n"
                "from asvimg.gui import ipc\n"
                "req = ipc.load(sys.argv[1])\n"
                "ipc.dump({'echo': req}, sys.argv[2])\n"
            )
            old = os.environ.get("PYTHONPATH", "")
            os.environ["PYTHONPATH"] = td + (os.pathsep + old if old else "")
            try:
                result = bridge._run_child("mock_fast_child", {"k": [1, 2, 3]})
            finally:
                if old:
                    os.environ["PYTHONPATH"] = old
                else:
                    os.environ.pop("PYTHONPATH", None)
            self.assertEqual(result, {"echo": {"k": [1, 2, 3]}})

    def test_subproc_entry_arg_guard(self) -> None:
        from asvimg.gui import ipc

        for mod in (
            "asvimg.gui.subproc_cpselect",
            "asvimg.gui.subproc_ica",
        ):
            p = subprocess.run(
                [sys.executable, "-m", mod],
                cwd=str(ipc.project_root()), env=ipc.child_env(),
                capture_output=True, text=True,
            )
            self.assertEqual(p.returncode, 2, (mod, p.stderr))


def _dpg_available() -> bool:
    try:
        import dearpygui.dearpygui as dpg

        dpg.create_context()
        dpg.destroy_context()
        return True
    except Exception:
        return False


@unittest.skipUnless(_dpg_available(), "dearpygui context unavailable (headless)")
class TestConfigForm(unittest.TestCase):
    def test_collect_validate_load_roundtrip(self) -> None:
        import dearpygui.dearpygui as dpg

        from asvimg import PipelineConfig
        from asvimg.gui.config_form import ConfigForm

        dpg.create_context()
        try:
            with dpg.window(tag="w"):
                form = ConfigForm(
                    PipelineConfig(
                        channels_name=["GCaMP", "jRGECO", "GCaMP", "jRGECO"],
                        channels_prop=["donner", "source", "source", "source"],
                        fps=40, binning=4, annotation="gui",
                        filter_xyt=[1, 1, 1], save_movie_vminmax=(-2.0, 4.0),
                    )
                )
                form.build(parent="w")

            collected = form.collect()
            self.assertEqual(collected["channels_name"],
                             ["GCaMP", "jRGECO", "GCaMP", "jRGECO"])
            self.assertEqual(collected["annotation"], "gui")
            self.assertEqual(collected["filter_xyt"], [1, 1, 1])
            self.assertEqual(tuple(collected["save_movie_vminmax"]), (-2.0, 4.0))
            self.assertEqual(form.to_config().cycle_len, 4)

            dpg.set_value("cfgw_baseline_percentile", 200.0)
            ok, err = form.validate()
            self.assertFalse(ok)
            self.assertIn("baseline_percentile", err)
            dpg.set_value("cfgw_baseline_percentile", 5.0)

            form.load(PipelineConfig(binning=8, exp_name="ZZ",
                                     save_roi_signals="csv", annotation=False))
            c3 = form.collect()
            self.assertEqual(c3["binning"], 8)
            self.assertEqual(c3["exp_name"], "ZZ")
            self.assertEqual(c3["save_roi_signals"], "csv")
            self.assertIs(c3["annotation"], False)
        finally:
            dpg.destroy_context()

    def test_inline_annotation_coords_preserved(self) -> None:
        # Loading a coordinate-pair annotation and reading it back must NOT
        # downgrade it to the literal "cache".
        import dearpygui.dearpygui as dpg

        from asvimg import PipelineConfig
        from asvimg.gui.config_form import ConfigForm

        coords = [[[5, 5], [5, 22]], [[100, 100], [100, 180]]]
        dpg.create_context()
        try:
            with dpg.window(tag="w"):
                form = ConfigForm(PipelineConfig(annotation=coords))
                form.build(parent="w")
            ann = form.collect()["annotation"]
            self.assertEqual(list(ann[0]), coords[0])
            self.assertEqual(list(ann[1]), coords[1])
            # to_config() must accept it as a real coordinate-pair annotation
            self.assertEqual(len(form.to_config().annotation), 2)
        finally:
            dpg.destroy_context()

    def test_a_field_with_no_widget_keeps_the_users_value(self) -> None:
        """collect() reads the WIDGETS; every Run then writes the result back over
        the user's ops.yaml. So a field the form does not render must be carried
        through, not silently reset to its default -- that would destroy the value
        on disk. (demux_start_offset was exactly this: no widget, and resetting it
        to 0 swaps source and donner.)"""
        import dearpygui.dearpygui as dpg

        from asvimg import PipelineConfig
        from asvimg.gui.config_form import ConfigForm

        dpg.create_context()
        try:
            cfg = PipelineConfig(demux_start_offset=1, corr_edge_density=0.35)
            form = ConfigForm(cfg)          # NOT built: no widget exists at all
            got = form.collect()
            self.assertEqual(got["demux_start_offset"], 1)
            self.assertEqual(got["corr_edge_density"], 0.35)
        finally:
            dpg.destroy_context()

    def test_inline_ica_exclusion_dict_survives_a_load(self) -> None:
        """The IC exclusion is one of the two decisions only a human can make.
        Load-ing a config that carries it inline must not downgrade it to the
        literal "cache" -- the next Run would write that back to ops.yaml."""
        import dearpygui.dearpygui as dpg

        from asvimg import PipelineConfig
        from asvimg.gui.config_form import ConfigForm

        dpg.create_context()
        try:
            with dpg.window(tag="w"):
                form = ConfigForm(PipelineConfig())
                form.build(parent="w")

            form.load(PipelineConfig(channels_name=["BL", "BL"],
                                     channels_prop=["source", "donner"],
                                     ica_exclusion={"BL": [2, 7]}))
            self.assertEqual(form.collect()["ica_exclusion"], {"BL": [2, 7]})
            self.assertEqual(form.to_config().ica_exclusion, {"BL": [2, 7]})
        finally:
            dpg.destroy_context()

    def test_float_fields_keep_the_value_the_user_typed(self) -> None:
        """add_input_float is backed by a C float: 0.35 came back as
        0.3499999940395355 and got written to ops.yaml on every Run."""
        import dearpygui.dearpygui as dpg

        from asvimg import PipelineConfig
        from asvimg.gui.config_form import ConfigForm

        dpg.create_context()
        try:
            with dpg.window(tag="w"):
                form = ConfigForm(PipelineConfig(corr_edge_density=0.35,
                                                 corr_network_threshold=0.45))
                form.build(parent="w")
            got = form.collect()
            self.assertEqual(got["corr_edge_density"], 0.35)
            self.assertEqual(got["corr_network_threshold"], 0.45)
        finally:
            dpg.destroy_context()

    def test_corr_map_panel_opens_and_lists(self) -> None:
        import dearpygui.dearpygui as dpg
        import numpy as np

        from asvimg import (
            ROIS_FILENAME,
            PipelineConfig,
            Roi,
            save_marks,
            save_rois,
        )
        from asvimg.atlas import ACCFv3
        from asvimg.gui.corr_map_panel import CorrMapPanel

        _ATLAS = BUNDLED_ATLAS
        if not _ATLAS.exists():
            self.skipTest("atlas not found")

        dpg.create_context()
        dpg.create_viewport()
        dpg.setup_dearpygui()
        try:
            atlas = ACCFv3.from_mat(str(_ATLAS))
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                out = tmp / "asi" / "npy"
                out.mkdir(parents=True)
                src = np.array([[10.0, 10.0], [10.0, 50.0], [50.0, 10.0]])
                ref = np.array([[20.0, 20.0], [20.0, 60.0], [60.0, 20.0]])
                save_marks(out / "marks.mat", src, ref)
                # a configured ROI set (rois.csv) — the seed list must use THIS,
                # not the atlas defaults
                save_rois(out / ROIS_FILENAME, [Roi("MYSEED", 30, 40, 6)])
                corr_dir = out / "corrMap"
                corr_dir.mkdir(parents=True)
                ha, wa = atlas.shape_hw
                np.save(corr_dir / "GCaMP_seed.npy", np.zeros((ha, wa), np.float32))
                np.save(corr_dir / "wrongshape.npy", np.zeros((3, 3), np.float32))
                cfg = PipelineConfig(
                    input_dir=str(tmp), output_dir=str(out), exp_name="S1"
                )
                panel = CorrMapPanel(cfg, atlas, out)
                panel.open()
                self.assertTrue(dpg.does_item_exist("corrmap_window"))
                self.assertTrue(dpg.does_item_exist("corrmap_tex"))
                # seed dropdown reflects the configured rois.csv
                self.assertIn("MYSEED", panel._rois)
                self.assertIn(
                    "MYSEED", dpg.get_item_configuration("corrmap_seed")["items"]
                )
                self.assertIn("GCaMP_seed", panel._list_maps())
                panel._on_select(None, "GCaMP_seed")  # display w/o recompute
                self.assertEqual(panel._cur_name, "GCaMP_seed")
                # a wrong-shaped map is guarded (no crash, selection unchanged)
                panel._on_select(None, "wrongshape")
                self.assertEqual(panel._cur_name, "GCaMP_seed")
        finally:
            dpg.destroy_context()

    def test_roi_editor_add_contra_remove_save(self) -> None:
        import dearpygui.dearpygui as dpg

        from asvimg import Roi, load_rois
        from asvimg.atlas import ACCFv3
        from asvimg.gui.roi_editor import RoiEditor

        _ATLAS = BUNDLED_ATLAS
        if not _ATLAS.exists():
            self.skipTest("atlas not found")

        dpg.create_context()
        dpg.create_viewport()
        dpg.setup_dearpygui()
        try:
            atlas = ACCFv3.from_mat(str(_ATLAS))
            with tempfile.TemporaryDirectory() as td:
                out = Path(td)
                ed = RoiEditor(atlas, out)
                ed.open()
                self.assertTrue(dpg.does_item_exist("roi_editor_window"))
                n0 = len(ed.rois)
                ed._add_contra(None, None, ed._ids[0])
                self.assertEqual(len(ed.rois), n0 + 1)
                ed._remove(None, None, ed._ids[0])
                self.assertEqual(len(ed.rois), n0)
                ed._save()
                self.assertTrue((out / "rois.csv").exists())
                self.assertEqual(len(load_rois(out / "rois.csv")), n0)
                ed._on_close()
                self.assertFalse(dpg.does_item_exist("roi_editor_window"))
        finally:
            dpg.destroy_context()

    def test_demux_editor_build_inspect_suggest_save(self) -> None:
        import dearpygui.dearpygui as dpg
        import numpy as np
        import tifffile

        from asvimg import PipelineConfig, load_demux_correction
        from asvimg.gui.demux_editor import _OFFSET, _WIN, DemuxEditor

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "asi" / "npy"
            out.mkdir(parents=True)
            inp = Path(td) / "in"
            inp.mkdir()
            # a slipped 2-ch fingerprint (dropped frame at 120); means match the frames
            rng = np.random.default_rng(0)
            n, h, w = 240, 8, 10
            stack = np.empty((n, h, w), np.uint16)
            means = np.empty(n, np.float32)
            for g in range(n):
                ph = g % 2 if g < 120 else (g + 1) % 2
                stack[g] = np.clip((100 if ph == 0 else 60)
                                   + rng.standard_normal((h, w)) * 2, 0, 65535)
                means[g] = float(stack[g].mean())
            tifffile.imwrite(inp / "recS1.tif", stack)
            np.savez(out / "demux_means.npz", means=means,
                     file_starts=np.array([0], np.int64))
            cfg = PipelineConfig(
                input_dir=str(inp), output_dir=str(out),
                channels_name=["BL", "BL"], channels_prop=["source", "donner"],
            )
            dpg.create_context()
            dpg.create_viewport()
            dpg.setup_dearpygui()
            try:
                ed = DemuxEditor(cfg, out)
                self.assertEqual(ed.n, n)
                ed.build()
                self.assertTrue(dpg.does_item_exist(_WIN))
                self.assertTrue(dpg.does_item_exist(_OFFSET))
                self.assertTrue(dpg.does_item_exist("dmx_preview_tex"))  # preview rendered

                # frame inspector: a window of ±4 real frames around a suspect index
                ed._show_neighborhood(120)
                shown = [g for g, _i, _c in ed._inspect_shown]
                self.assertEqual(shown, list(range(116, 125)))  # 120 ±4
                self.assertTrue(dpg.does_item_exist("dmx_insp_tex_120"))  # thumbnail read

                ed._on_suggest()  # pre-fill an edit from the detected slip
                self.assertTrue(ed.edits)
                # a suggested slip is a ranged edit [start, -1(=to end), delta]
                self.assertEqual(len(ed.edits[0]), 3)
                self.assertEqual(ed.edits[0][1], -1)
                k = len(ed.edits)
                ed._on_add_edit()
                self.assertEqual(len(ed.edits), k + 1)

                # export the corrected per-channel TIFFs (background thread)
                ed._on_export()
                self.assertIsNotNone(ed._export_thread)
                ed._export_thread.join(timeout=60)
                tiffs = sorted((out / "demux_corrected").glob("*.tif"))
                self.assertEqual(len(tiffs), 2)  # one stack per channel

                ed._on_save()
                self.assertTrue(ed.should_close)
                corr = load_demux_correction(out, cycle_len=2)
                self.assertIsNotNone(corr)
                self.assertGreaterEqual(len(corr.edits), 1)
            finally:
                dpg.destroy_context()

    def test_movie_preview_open_play_close(self) -> None:
        import dearpygui.dearpygui as dpg
        import numpy as np

        try:
            import cv2
        except Exception:  # noqa: BLE001
            self.skipTest("cv2 unavailable")

        from asvimg.gui import movie_preview as mv

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "asi" / "npy"
            out.mkdir(parents=True)
            mdir = out / "movies"
            mdir.mkdir()

            def _make(path, w, h, n):
                vw = cv2.VideoWriter(
                    str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (w, h)
                )
                for i in range(n):
                    vw.write(np.full((h, w, 3), (i * 8) % 255, np.uint8))
                vw.release()

            _make(mdir / "e_BL_dF.avi", 120, 100, 24)
            _make(mdir / "e_GC_dF.avi", 90, 90, 12)  # shorter → clamps

            found = mv.find_movies(out)
            self.assertEqual([p.name for p in found], ["e_BL_dF.avi", "e_GC_dF.avi"])

            dpg.create_context()
            dpg.create_viewport()
            dpg.setup_dearpygui()
            try:
                p = mv.open_preview(found)
                self.assertTrue(p.is_open())
                self.assertEqual(p.total, 24)  # timeline = longest movie
                self.assertEqual(len(p.tex_tags), 2)

                p._toggle_play()
                self.assertTrue(p.playing)
                p._last = 0.0
                p.pos = 0.0
                import time

                p._last = time.perf_counter() - 0.5  # pretend 0.5 s elapsed
                p.tick()
                self.assertEqual(p._shown, 10)  # 0.5 s * 20 fps * x1

                p._on_speed(None, "x4")
                self.assertEqual(p.speed, 4.0)
                p._on_slider(None, 7)
                self.assertEqual(p._shown, 7)

                p.close()
                self.assertFalse(p.is_open())
            finally:
                dpg.destroy_context()

    def test_dashboard_builds(self) -> None:
        import dearpygui.dearpygui as dpg

        from asvimg.gui.app import DashboardApp

        dpg.create_context()
        try:
            app = DashboardApp()
            app._build_ui()
            for tag in ("primary_window", "log_text", "progress_bar",
                        "status_line", "stagestat_preprocess",
                        "stagestat_correlation"):
                self.assertTrue(dpg.does_item_exist(tag), tag)
            app._handle(("log", "pca", "hello", "info"))
            self.assertIn("hello", dpg.get_value("log_text"))
        finally:
            dpg.destroy_context()

    def test_stage_table_state_and_last_run(self) -> None:
        import dearpygui.dearpygui as dpg

        from asvimg.gui.app import DashboardApp

        dpg.create_context()
        try:
            app = DashboardApp()
            app._build_ui()
            self.assertTrue(dpg.does_item_exist("stage_table"))
            # running -> done updates State + stamps a dated Last-run
            app._handle(("stage", "pca", "running", "", None))
            self.assertEqual(dpg.get_value(app._stage_tag("pca")), "Running")
            app._handle(("stage", "pca", "done", "", 1.4))
            self.assertEqual(dpg.get_value(app._stage_tag("pca")), "Done")
            self.assertIn("/", dpg.get_value(app._stage_last_tag("pca")))  # has date
            app._handle(("stage", "ica", "failed", "boom", 2.0))
            self.assertEqual(dpg.get_value(app._stage_tag("ica")), "Error")
        finally:
            dpg.destroy_context()

    def test_loading_null_output_dir_stays_none(self) -> None:
        # text_dir fields must show empty (not the string "None") when the
        # loaded value is null, else to_config() would return "None".
        import dearpygui.dearpygui as dpg

        from asvimg import PipelineConfig
        from asvimg.config import save_config
        from asvimg.gui.app import DashboardApp
        from asvimg.gui.config_form import _tag

        dpg.create_context()
        try:
            app = DashboardApp()
            app._build_ui()
            with tempfile.TemporaryDirectory() as td:
                cfg = PipelineConfig(
                    input_dir="/x", output_dir=None,
                    channels_name=["BL", "BL"], channels_prop=["source", "donner"],
                )
                save_config(cfg, td)
                app.form.load_from_path(td)
                self.assertEqual(dpg.get_value(_tag("output_dir")), "")
                self.assertIsNone(app.form.to_config().output_dir)
        finally:
            dpg.destroy_context()

    def test_input_dir_folder_browse(self) -> None:
        import dearpygui.dearpygui as dpg

        from asvimg.gui import native_dialog
        from asvimg.gui.app import DashboardApp
        from asvimg.gui.config_form import _tag

        dpg.create_context()
        orig = native_dialog.open_folder
        native_dialog.open_folder = lambda **k: "/picked/input"
        try:
            app = DashboardApp()
            app._build_ui()
            app.form._cb_browse_dir(None, None, _tag("input_dir"))
            self.assertEqual(dpg.get_value(_tag("input_dir")), "/picked/input")
            self.assertEqual(app.form.to_config().input_dir, "/picked/input")
        finally:
            native_dialog.open_folder = orig
            dpg.destroy_context()

    def test_open_input_output_folder_buttons(self) -> None:
        import dearpygui.dearpygui as dpg

        from asvimg.gui import native_dialog
        from asvimg.gui.app import DashboardApp
        from asvimg.gui.config_form import _tag

        dpg.create_context()
        orig = native_dialog.open_in_file_manager
        opened: list = []
        native_dialog.open_in_file_manager = lambda p: (opened.append(str(p)), True)[1]
        try:
            app = DashboardApp()
            app._build_ui()
            with tempfile.TemporaryDirectory() as td:
                inp = Path(td) / "raw"
                inp.mkdir()
                dpg.set_value(_tag("input_dir"), str(inp))
                dpg.set_value(_tag("output_format"), "npy")
                i, o = app._configured_dirs()
                self.assertEqual(i, inp)
                self.assertEqual(o, inp / "asi" / "npy")
                app._cb_open_input()
                self.assertIn(str(inp), opened)
        finally:
            native_dialog.open_in_file_manager = orig
            dpg.destroy_context()

    def test_load_standard_preset_sets_ops_keeps_db(self) -> None:
        import dearpygui.dearpygui as dpg

        from asvimg.gui.app import (
            DashboardApp,
            _PRESET_COMBO,
            _STD_PREFIX,
            _STD_PRESET_DIR,
        )
        from asvimg.gui.config_form import _tag

        self.assertTrue((_STD_PRESET_DIR / "ops_2ch.yaml").exists())
        dpg.create_context()
        try:
            app = DashboardApp()
            app._build_ui()
            dpg.set_value(_tag("input_dir"), "/my/input")
            dpg.set_value(_tag("output_format"), "mat")
            dpg.set_value(_PRESET_COMBO, f"{_STD_PREFIX}ops_2ch")  # standard preset
            app._cb_load_preset()
            c = app.form.to_config()
            self.assertEqual(c.channels_name, ["BL", "BL"])  # ops from preset
            self.assertEqual(c.fps, 40)
            self.assertTrue(c.skip_ica)
            self.assertEqual(c.input_dir, "/my/input")       # db untouched
            self.assertEqual(c.output_format, "mat")
        finally:
            dpg.destroy_context()

    def test_save_as_preset_writes_user_yaml_and_reloads(self) -> None:
        import dearpygui.dearpygui as dpg

        import asvimg.gui.app as app_mod
        from asvimg.gui.app import DashboardApp, _PRESET_COMBO, _PRESET_NAME
        from asvimg.gui.config_form import _tag

        dpg.create_context()
        orig_dir = app_mod._USER_PRESET_DIR
        try:
            with tempfile.TemporaryDirectory() as td:
                app_mod._USER_PRESET_DIR = Path(td)  # user presets -> temp
                app = DashboardApp()
                app._build_ui()
                dpg.set_value(_tag("fps"), 123)
                dpg.set_value(_PRESET_NAME, "myset")
                app._cb_save_preset()
                self.assertTrue((Path(td) / "myset.yaml").exists())
                # dropdown refreshed to include the new user preset, and selected it
                items = dpg.get_item_configuration(_PRESET_COMBO)["items"]
                self.assertIn("myset", items)
                self.assertEqual(dpg.get_value(_PRESET_COMBO), "myset")
                # reloading the preset roundtrips the saved ops value
                dpg.set_value(_tag("fps"), 40)
                app._cb_load_preset()
                self.assertEqual(app.form.to_config().fps, 123)
        finally:
            app_mod._USER_PRESET_DIR = orig_dir
            dpg.destroy_context()

    def test_save_as_preset_prompts_before_overwrite(self) -> None:
        import yaml
        import dearpygui.dearpygui as dpg

        import asvimg.gui.app as app_mod
        from asvimg.gui.app import DashboardApp, _PRESET_NAME
        from asvimg.gui.config_form import _tag

        dpg.create_context()
        orig_dir = app_mod._USER_PRESET_DIR
        try:
            with tempfile.TemporaryDirectory() as td:
                app_mod._USER_PRESET_DIR = Path(td)
                app = DashboardApp()
                app._build_ui()
                target = Path(td) / "dup.yaml"
                # first save creates the file (no prompt for a new name)
                dpg.set_value(_tag("fps"), 111)
                dpg.set_value(_PRESET_NAME, "dup")
                app._cb_save_preset()
                self.assertTrue(target.exists())
                self.assertFalse(dpg.does_item_exist("_preset_overwrite_modal"))
                # saving the same name again prompts; the file is NOT yet changed
                dpg.set_value(_tag("fps"), 222)
                app._cb_save_preset()
                self.assertTrue(dpg.does_item_exist("_preset_overwrite_modal"))
                self.assertEqual(yaml.safe_load(target.read_text())["fps"], 111)
                # confirming overwrite (what the modal's button does) writes it
                app._write_preset("dup", target)
                self.assertEqual(yaml.safe_load(target.read_text())["fps"], 222)
        finally:
            app_mod._USER_PRESET_DIR = orig_dir
            dpg.destroy_context()

    def test_save_as_preset_refuses_standard_name(self) -> None:
        import dearpygui.dearpygui as dpg

        import asvimg.gui.app as app_mod
        from asvimg.gui.app import DashboardApp, _PRESET_NAME, _STATUS
        from asvimg.gui.config_form import _tag

        dpg.create_context()
        orig_dir = app_mod._USER_PRESET_DIR
        try:
            with tempfile.TemporaryDirectory() as td:
                app_mod._USER_PRESET_DIR = Path(td)
                app = DashboardApp()
                app._build_ui()
                dpg.set_value(_tag("fps"), 99)
                dpg.set_value(_PRESET_NAME, "ops_2ch")  # a standard preset name
                app._cb_save_preset()
                # nothing written, no modal, and a clear read-only message
                self.assertFalse((Path(td) / "ops_2ch.yaml").exists())
                self.assertFalse(dpg.does_item_exist("_preset_overwrite_modal"))
                self.assertIn("standard preset", dpg.get_value(_STATUS))
        finally:
            app_mod._USER_PRESET_DIR = orig_dir
            dpg.destroy_context()

    def test_make_export_dir_button(self) -> None:
        import dearpygui.dearpygui as dpg

        from asvimg.gui.app import DashboardApp
        from asvimg.gui.config_form import _tag

        dpg.create_context()
        try:
            app = DashboardApp()
            app._build_ui()
            with tempfile.TemporaryDirectory() as td:
                inp = Path(td) / "raw"
                inp.mkdir()
                dpg.set_value(_tag("input_dir"), str(inp))
                dpg.set_value(_tag("output_format"), "npy")
                app._cb_make_export_dir()  # output_dir blank -> <input>/asi/npy
                exp = inp / "asi" / "npy"
                self.assertTrue((exp / "ops.yaml").exists())
                self.assertTrue((exp / "db.yaml").exists())
                self.assertEqual(dpg.get_value("config_path_input"), str(exp))
        finally:
            dpg.destroy_context()

    def test_run_syncs_ops_and_config_path(self) -> None:
        import dearpygui.dearpygui as dpg

        from asvimg.gui.app import DashboardApp
        from asvimg.gui.config_form import _tag

        dpg.create_context()
        try:
            app = DashboardApp()
            app._build_ui()
            with tempfile.TemporaryDirectory() as td:
                out = Path(td) / "asi" / "npy"
                dpg.set_value(_tag("output_dir"), str(out))
                app._sync_config_to_output(app.form.to_config())
                # output folder created with the config, path field repointed
                self.assertTrue((out / "ops.yaml").exists())
                self.assertTrue((out / "db.yaml").exists())
                self.assertEqual(dpg.get_value("config_path_input"), str(out))
                # a param change re-writes ops on the next run
                dpg.set_value(_tag("save_roi_signals"), "both")
                app._sync_config_to_output(app.form.to_config())
                import yaml

                ops = yaml.safe_load((out / "ops.yaml").read_text())
                self.assertEqual(ops.get("save_roi_signals"), "both")
        finally:
            dpg.destroy_context()


class TestNativeDialog(unittest.TestCase):
    """The OS-native file picker helper (subprocess mocked — no real dialog)."""

    def _run_with(self, fake):
        from asvimg.gui import native_dialog as nd

        orig = nd.subprocess.run
        nd.subprocess.run = fake
        try:
            return nd.open_file(extensions=("yaml", "yml"))
        finally:
            nd.subprocess.run = orig

    def test_returns_selected_path(self) -> None:
        class R:
            returncode = 0
            stdout = "/data/asi/npy/ops.yaml\n"
            stderr = ""

        self.assertEqual(self._run_with(lambda *a, **k: R()), "/data/asi/npy/ops.yaml")

    def test_cancel_returns_none(self) -> None:
        class R:
            returncode = 1
            stdout = ""
            stderr = "User canceled"

        self.assertIsNone(self._run_with(lambda *a, **k: R()))

    def test_missing_helper_returns_none(self) -> None:
        def boom(*a, **k):
            raise FileNotFoundError()

        self.assertIsNone(self._run_with(boom))


if __name__ == "__main__":
    unittest.main()
