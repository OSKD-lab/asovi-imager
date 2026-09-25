"""ASoVi pipeline dashboard — the single long-lived dearpygui application.

Owns one dpg context for its whole lifetime (never torn down).  Heavy work
runs on a daemon worker thread driving a :class:`PipelineSession`; the
:class:`GuiBridge` is the session's reporter + interactive providers and feeds
a thread-safe event queue that the render loop drains each frame.  The two
interactive GUIs (cpselect, ICA) run in child processes, so they never create
a second context in this process.
"""

from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from asvimg import (
    Cancelled,
    PipelineConfig,
    StageId,
    default_output_dir,
)
from asvimg.runner import PipelineSession

from .bridge import GuiBridge
from .config_form import ConfigForm
from .figures import FigurePanels

_STAGES = [
    ("run_preprocess", StageId.PREPROCESS),
    ("run_pca", StageId.PCA),
    ("run_ica", StageId.ICA),
    ("run_annotation", StageId.ANNOTATION),
    ("run_roi", StageId.ROI),
    ("run_correlation", StageId.CORRELATION),
    ("run_export", StageId.EXPORT),
]

_PRIMARY = "primary_window"
_TEXREG = "fig_texture_registry"
_FIGPANELS = "fig_panels"
_LOG = "log_text"
_PROGRESS = "progress_bar"
_STATUS = "status_line"
_CFG_PATH = "config_path_input"
_STAGE_TABLE = "stage_table"
_PRESET_COMBO = "preset_combo"
_PRESET_NAME = "preset_name_input"
# Standard (bundled, read-only) presets ship in the repo; user presets live in
# the home dir and are the only ones "Save as preset" may write / overwrite.
_STD_PRESET_DIR = Path(__file__).resolve().parent.parent / "presets"
_USER_PRESET_DIR = Path.home() / ".asovi" / "presets"
_STD_PREFIX = "[std] "  # dropdown marker for standard (protected) presets
_FACTORY_PRESET = "(factory defaults)"  # dropdown entry -> built-in defaults
# Last session's processing settings, auto-saved on exit and recalled by
# "Read last config".
_LAST_OPS = Path.home() / ".asovi" / "last_ops.yaml"

# Word shown in the State column per status.
_STATE_WORD = {
    "pending": "Pending",
    "running": "Running",
    "done": "Done",
    "failed": "Error",
    "skipped": "Skipped",
    "stale": "Stale",
}
_TERMINAL_STATUS = ("done", "failed", "skipped")

# Quick-preview session methods: QC views, not pipeline stages — they neither
# write ops.yaml nor re-baseline staleness detection.
_PREVIEW_METHODS = frozenset({"preview_input", "preview_input_all", "qc_frame_slip"})


# Row colour per status — Error stands out red, Done green, so a finished stage
# clearly reads as good vs bad.
_STATUS_COLOR = {
    "pending": (180, 180, 180),
    "running": (230, 200, 90),  # amber
    "done": (110, 200, 110),  # green
    "failed": (230, 80, 80),  # red
    "skipped": (150, 150, 150),
    "stale": (150, 150, 150),
}
_PARAM_CHANGED = (210, 180, 90)  # amber: was Done but inputs changed since

# Config fields each stage directly depends on, and its upstream stages. A stage
# is stale (Done -> "Done (* param-changed)") when any of these — or an upstream
# stage's inputs — changed since it last ran. "__rois__" = the edited rois.csv +
# rois_source.csv (see _rois_key).
_STAGE_PARAMS = {
    "preprocess": [
        "input_dir",
        "input_format",
        # order = concatenation order of the timeline: changing it changes every
        # frame index downstream, so preprocess must go stale.
        "input_order",
        "output_dir",
        "output_format",
        "exp_name",
        "dcimg_backend",
        "channels_name",
        "channels_prop",
        "channels_slip",
        "demux_start_offset",
        "demux_qc",
        "do_registration",
        "registration_cache",
        "usfac",
        "linear_subt",
        "baseline_percentile",
        # every one of these changes every value in dff_{name}.npy
        "detrend",
        "baseline_percentile_highpass",
        "baseline_percentile_highpass_sec",
        "baseline_percentile_highpass_rank",
        "hemovar_qc",
        "binning",
        "flip",
        "delete",
        "fps",
        "use_mmap",
        "registration_batch_size",
        "template",
        "template_stride",
        "template_ch",
        "filter_xyt",
        "filter_xyt_kind",
        "max_frames",
        "start_initial_frames",
        "ignore_last_frames",
        "output_metadata_yaml",
        # the per-channel TIFFs (see _STAGE_OUTPUT_PARAMS: they re-flag
        # preprocess only, not everything downstream of it)
        "save_raw_each_ch",
        "save_registered_each_ch",
        "tiff_format",
        "tiff_compression",
    ],
    "pca": [
        "pca_n_components",
        "pca_smooth_sigma",
        "pca_skip_frames",
        "ch_for_annotation",
    ],
    "ica": [
        "skip_ica",
        "ica_n_components",
        "ica_max_iter",
        "ica_random_state",
        "ica_exclusion",
        # NOT "__ica_sel__": the ICA stage WRITES ica_exclusion.json during its own
        # run, and the signature is snapshotted before the worker starts — hashing
        # it here would leave ICA permanently showing "param-changed".
    ],
    "annotation": [
        "annotation",
        "annotation_allow_reflection",
        "annotation_atlas_path",
        "ch_for_annotation",
    ],
    "roi": [
        "roi_signal",
        "roi_space",
        "save_roi_signals",
        "post_annotation_time_average",
        "post_annotation_filter_xyt",
        "post_annotation_filter_kind",
        "__rois__",  # content of rois.csv AND rois_source.csv (see _rois_key)
        # denoising reaches the ROI signals, so re-picking ICs makes them stale.
        # "__ica_sel__" = the content of ica_exclusion.json (like "__rois__").
        "ica_denoise",
        "__ica_sel__",
    ],
    "correlation": [
        "corr_method",
        "corr_auto_threshold",
        "corr_edge_density",
        "corr_network_threshold",
        "roi_space",
        "ica_denoise",
        "__ica_sel__",
    ],
    "export": [
        "save_annotated_each_ch",
        "save_annotated_dF_mat",
        "save_annotated_dF_dtype",
        "save_movie",
        "save_movie_speed",
        "save_movie_codec",
        "save_movie_cmap",
        "save_movie_vminmax",
        "save_movie_merge_chs",
        "tiff_format",
        "tiff_compression",
        "post_annotation_time_average",
        "post_annotation_filter_xyt",
        "post_annotation_filter_kind",
        "export_orientation",
        "ica_denoise",
        "__ica_sel__",
    ],
}
# Params that only decide which extra files a stage writes, never the values
# the next stage reads: they make the stage itself stale but are left out of the
# signature its downstream stages inherit, so ticking "save TIFF" re-flags
# preprocess alone instead of every stage after it.
_STAGE_OUTPUT_PARAMS = {
    "preprocess": {"save_raw_each_ch", "save_registered_each_ch",
                   "tiff_format", "tiff_compression"},
}
_STAGE_UPSTREAM = {
    "preprocess": [],
    "pca": ["preprocess"],
    "ica": ["pca"],
    "annotation": ["preprocess"],
    "roi": ["preprocess", "annotation"],
    "correlation": ["roi"],
    "export": ["preprocess", "annotation"],
}


def _hashable(v):
    return tuple(_hashable(x) for x in v) if isinstance(v, (list, tuple)) else v


class DashboardApp:
    def __init__(self, config_path: str | None = None) -> None:
        self.form = ConfigForm()
        self.bridge = GuiBridge()
        self.figure_panels: FigurePanels | None = None
        self._movie_preview = None
        self.worker: threading.Thread | None = None
        self._log_lines: list[str] = []
        self._initial_config_path = config_path
        self._run_sig: dict = {}  # stage -> input signature at its last run
        self._run_cfg_dict: dict = {}  # form config captured at the current run
        self._run_rois_key: str = ""
        self._stale_tick = 0

    # --- worker management ------------------------------------------------

    def _busy(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def _start(self, method_names: list[str]) -> None:
        if self._busy():
            self._set_status("Busy - a run is already in progress.")
            return
        try:
            config = self.form.to_config()
        except (ValueError, TypeError) as exc:
            self._set_status(f"Invalid config: {exc}")
            return

        # Persist the exact params being run to <output>/ops.yaml and point the
        # config-path field at that folder, so it always reflects the latest
        # outputs.  Quick Preview is not a real run, so it is skipped.
        if not set(method_names) <= _PREVIEW_METHODS:
            self._sync_config_to_output(config)
            self._snapshot_run_inputs()  # baseline for staleness detection

        self._reset_stage_status(method_names)
        self.bridge.cancel.reset()
        self._set_status("Running...")
        self.worker = threading.Thread(
            target=self._run, args=(config, method_names), daemon=True
        )
        self.worker.start()

    def _start_task(self, target, *args, tag: str = "task") -> None:
        """Run an arbitrary (non-pipeline) background job on the worker thread.

        Reuses the same busy-guard / cancel token / worker_done event as the
        pipeline runner, so a utility like the sifx converter shares the UI's
        progress bar, log pane and Stop button without a PipelineSession."""
        if self._busy():
            self._set_status("Busy - a run is already in progress.")
            return
        self.bridge.cancel.reset()
        self._set_status("Running...")
        self.worker = threading.Thread(
            target=self._run_task, args=(target, args, tag), daemon=True
        )
        self.worker.start()

    def _run_task(self, target, args, tag: str) -> None:
        """Worker thread for a background job: reports only via the bridge."""
        try:
            target(*args)
        except Cancelled:
            self.bridge.on_log(tag, f"[{tag}] cancelled by user.")
        except Exception as exc:  # noqa: BLE001 — surface to the log pane
            self.bridge.on_log(tag, f"[{tag}] ERROR: {exc}", level="error")
            self.bridge.on_log(tag, traceback.format_exc(), level="error")
        finally:
            self.bridge.events.put(("worker_done", None))

    def _sync_config_to_output(self, config: PipelineConfig):
        """Write db.yaml/ops.yaml to the run's output dir and point the config
        path field there (creating the mat/npy folder if needed). Returns the dir."""
        import dearpygui.dearpygui as dpg

        if config.output_dir:
            out_dir = Path(config.output_dir)
        elif config.input_dir:
            out_dir = default_output_dir(Path(config.input_dir), config.output_format)
        else:
            return None
        try:
            self.form.save_to(out_dir)  # save_config() creates the folder
            dpg.set_value(_CFG_PATH, str(out_dir))
            return out_dir
        except Exception as exc:  # noqa: BLE001 — non-fatal, just log it
            self._append_log(f"!! [gui] could not update ops.yaml: {exc}")
            return None

    def _section_footer(self, section: str) -> None:
        """Extra widgets appended at the end of a config section."""
        import dearpygui.dearpygui as dpg

        if section == "Data":
            dpg.add_separator()
            with dpg.group(horizontal=True):
                mk_btn = dpg.add_button(
                    label="Make export dir", callback=self._cb_make_export_dir
                )
                sifx_btn = dpg.add_button(
                    label="sifx converter", callback=self._cb_sifx_convert
                )
            with dpg.tooltip(mk_btn):
                dpg.add_text(
                    "Create the output dir (output_dir, else <input_dir>/asi/<format>),\n"
                    "write db.yaml/ops.yaml there, and point the config path field at it."
                )
            with dpg.tooltip(sifx_btn):
                dpg.add_text(
                    "Pick a folder holding an Andor .sifx spool and convert every\n"
                    "frame into a stacked BigTIFF (<sifx-name>_stacked.tif, written\n"
                    "next to the .sifx)."
                )

    def _cb_make_export_dir(self) -> None:
        try:
            config = self.form.to_config()
        except (ValueError, TypeError) as exc:
            self._set_status(f"Invalid config: {exc}")
            return
        out = self._sync_config_to_output(config)
        if out is not None:
            self._set_status(f"Made export dir + db/ops.yaml -> {out}")
        else:
            self._set_status("Set input_dir or output_dir to make an export dir.")

    def _cb_sifx_convert(self) -> None:
        """Pick a folder with an Andor .sifx spool and convert it to a stacked TIFF."""
        from . import native_dialog

        if self._busy():
            self._set_status("Busy - a run is already in progress.")
            return
        inp = self._configured_dirs()[0]
        folder = native_dialog.open_folder(
            title="Select a folder containing a .sifx spool",
            default_dir=str(inp) if inp else None,
        )
        if not folder:
            return
        from asvimg.sifx_convert import find_sifx

        sifx_files = find_sifx(folder)
        if not sifx_files:
            self._set_status(f"No .sifx spool found under {folder}")
            return
        self._set_status(f"Converting {len(sifx_files)} .sifx -> stacked TIFF...")
        self._start_task(self._sifx_worker, sifx_files, tag="sifx")

    def _sifx_worker(self, sifx_files) -> None:
        from asvimg.sifx_convert import convert_sifx_to_tiff

        for sp in sifx_files:
            out = sp.parent / f"{sp.stem}_stacked.tif"
            self.bridge.on_log("sifx", f"[sifx] {sp.name} -> {out.name}")

            def _prog(cur, total, _name=sp.stem):
                self.bridge.on_progress("sifx", cur, total, message=_name)

            n = convert_sifx_to_tiff(sp, out, progress=_prog, cancel=self.bridge.cancel)
            self.bridge.on_log("sifx", f"[sifx] wrote {n} frames -> {out}")

    def _configured_dirs(self):
        """(input_dir, resolved output dir) from the current form, or (None, None)."""
        try:
            cfg = self.form.to_config()
        except (ValueError, TypeError):
            return None, None
        inp = Path(cfg.input_dir) if cfg.input_dir else None
        if cfg.output_dir:
            out = Path(cfg.output_dir)
        elif cfg.input_dir:
            out = default_output_dir(Path(cfg.input_dir), cfg.output_format)
        else:
            out = None
        return inp, out

    def _open_dir(self, path, label: str) -> None:
        from . import native_dialog

        if path is None:
            self._set_status(f"No {label} directory configured.")
            return
        if native_dialog.open_in_file_manager(path):
            self._set_status(f"Opened {label} folder: {path}")
        else:
            self._set_status(f"{label.capitalize()} folder not found: {path}")

    def _cb_open_input(self) -> None:
        self._open_dir(self._configured_dirs()[0], "input")

    def _cb_open_output(self) -> None:
        self._open_dir(self._configured_dirs()[1], "output")

    def _cb_movie_preview(self) -> None:
        from . import movie_preview

        _, out = self._configured_dirs()
        if out is None:
            self._set_status("Set input_dir or output_dir first.")
            return
        movies = movie_preview.find_movies(out)
        if not movies:
            self._set_status("No movies found — run export with save_movie enabled.")
            return
        # x1 = realtime: the encoded AVI fps already includes save_movie_speed,
        # so drive playback from the acquisition per-channel rate instead.
        base_fps = None
        try:
            cfg = self.form.to_config()
            base_fps = cfg.fps / max(1, cfg.cycle_len)
        except Exception:  # noqa: BLE001 — fall back to the encoded fps
            base_fps = None
        try:
            self._movie_preview = movie_preview.open_preview(movies, base_fps=base_fps)
            self._set_status(f"Movie preview: {len(movies)} movie(s).")
        except Exception as exc:  # noqa: BLE001 — surface to the status line
            self._set_status(f"Movie preview failed: {exc}")

    def _update_movie_btn(self) -> None:
        """Enable the Movie-preview button only when export produced movies."""
        import dearpygui.dearpygui as dpg

        btn = getattr(self, "_movie_btn", None)
        if btn is None or not dpg.does_item_exist(btn):
            return
        from . import movie_preview

        _, out = self._configured_dirs()
        available = bool(out) and bool(movie_preview.find_movies(out))
        dpg.configure_item(btn, enabled=available)

    def _set_edit_rois_enabled(self, enabled: bool) -> None:
        import dearpygui.dearpygui as dpg

        btn = getattr(self, "_edit_rois_btn", None)
        if btn is not None and dpg.does_item_exist(btn):
            dpg.configure_item(btn, enabled=enabled)

    def _cb_edit_rois(self) -> None:
        from asvimg.atlas import ACCFv3
        from asvimg.config import resolve_atlas_path

        from . import roi_editor

        try:
            cfg = self.form.to_config()
        except (ValueError, TypeError) as exc:
            self._set_status(f"Invalid config: {exc}")
            return
        _, out = self._configured_dirs()
        if out is None:
            self._set_status("Set input_dir or output_dir first.")
            return
        try:
            atlas = ACCFv3.from_mat(resolve_atlas_path(cfg.annotation_atlas_path))
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Atlas load failed: {exc}")
            return
        out.mkdir(parents=True, exist_ok=True)
        self._roi_editor = roi_editor.open_editor(
            atlas, out, on_saved=self._on_rois_saved
        )

    def _on_rois_saved(self, path, n) -> None:
        self._set_status(f"Saved {n} ROIs -> {path}")
        self._refresh_stale()  # roi/correlation now stale if they were Done

    def _cb_corr_map(self) -> None:
        """Open the seed-correlation-map panel (needs annotation done)."""
        from asvimg.atlas import ACCFv3
        from asvimg.config import resolve_atlas_path

        from . import corr_map_panel

        try:
            cfg = self.form.to_config()
        except (ValueError, TypeError) as exc:
            self._set_status(f"Invalid config: {exc}")
            return
        _, out = self._configured_dirs()
        if out is None:
            self._set_status("Set input_dir or output_dir first.")
            return
        if not (out / "marks.mat").exists():
            self._set_status("Run annotation first (no marks.mat).")
            return
        try:
            atlas = ACCFv3.from_mat(resolve_atlas_path(cfg.annotation_atlas_path))
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Atlas load failed: {exc}")
            return
        try:
            self._corr_panel = corr_map_panel.open_panel(cfg, atlas, out)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Corr Map failed: {exc}")

    def _run(self, config: PipelineConfig, method_names: list[str]) -> None:
        """Worker thread: never touches dpg directly (only the bridge)."""
        session = PipelineSession(
            config,
            reporter=self.bridge,
            marks_provider=self.bridge.marks_provider,
            ica_provider=self.bridge.ica_provider,
            cancel=self.bridge.cancel,
        )
        try:
            for name in method_names:
                getattr(session, name)()
        except Cancelled:
            self.bridge.on_log("run", "[run] cancelled by user.")
        except Exception as exc:  # noqa: BLE001 — surface to the log pane
            self.bridge.on_log("run", f"[run] ERROR: {exc}", level="error")
            self.bridge.on_log("run", traceback.format_exc(), level="error")
        finally:
            self.bridge.events.put(("worker_done", None))

    # --- button callbacks (main thread) -----------------------------------

    def _cb_run_all(self) -> None:
        self._start(["run_all"])

    def _cb_run_stage(self, _sender, _app, method_name) -> None:
        self._start([method_name])

    def _cb_show_result(self, _sender, _app, method_name) -> None:
        """Load the stage's previously-saved figures from disk (no re-run)."""
        import numpy as np

        stage = next((str(s) for m, s in _STAGES if m == method_name), None)
        _, out = self._configured_dirs()
        if stage is None or out is None or not Path(out).is_dir():
            self._set_status("Set/create the output dir first.")
            return
        out = Path(out)

        def _num(p):
            return int("".join(filter(str.isdigit, p.stem)) or 0)

        imgs: list = []
        figdir = out / "figures"
        if figdir.is_dir():
            imgs += [(p.stem, p) for p in sorted(figdir.glob(f"{stage}_*.png"))]
        if stage == "preprocess":
            for nm in ("fig_template.png", "figFrames.png"):
                if (out / nm).exists():
                    imgs.append((Path(nm).stem, out / nm))
        elif stage in ("pca", "ica"):
            # Per-group subdirs (flat in older folders). The panel key must carry
            # the group or the second group's maps overwrite the first's.
            d = out / ("pca_images" if stage == "pca" else "ica_images")
            pat = "**/PC*.png" if stage == "pca" else "**/IC*.png"
            if d.is_dir():
                imgs += [
                    (p.stem if p.parent == d else f"{p.parent.name}/{p.stem}", p)
                    for p in sorted(d.glob(pat), key=_num)
                ]

        if not imgs:
            self._set_status(
                f"No saved {stage} figures — enable save_figures or run the stage."
            )
            return
        from PIL import Image

        shown = 0
        for key, p in imgs:
            try:
                im = Image.open(p).convert("RGBA")
                w, h = im.size
                data = (np.asarray(im, dtype=np.float32) / 255.0).ravel()
                if self.figure_panels is not None:
                    self.figure_panels.add_rgba(f"{stage}:{key}", w, h, data)
                shown += 1
            except Exception:  # noqa: BLE001
                pass
        self._set_status(f"Shown {shown} saved {stage} figure(s) from disk.")

    def _cb_preview(self) -> None:
        # Quick QC of the first frames — not a pipeline stage, reuses the worker.
        self._start(["preview_input"])

    def _cb_preview_all(self) -> None:
        # Same, but the head of every input file (one row per file) — shows a
        # per-file channel-cycle slip that the first file alone cannot.
        self._start(["preview_input_all"])

    def _cb_qc_frame_slip(self) -> None:
        # Image-content demux-slip QC: embed every file's head frames in 2D and
        # flag files whose source/donner assignment lands in the wrong content
        # cluster — catches slips the intensity demux QC misses on channels that
        # differ in content but not brightness.
        self._start(["qc_frame_slip"])

    def _cb_stop(self) -> None:
        if self._busy():
            self.bridge.request_cancel()
            self._set_status("Stopping...")

    def _cb_browse(self) -> None:
        import dearpygui.dearpygui as dpg

        from . import native_dialog

        current = dpg.get_value(_CFG_PATH).strip()
        default_dir = None
        if current:
            p = Path(current)
            default_dir = str(p if p.is_dir() else p.parent)
        path = native_dialog.open_file(
            title="Select ops.yaml / db.yaml",
            extensions=("yaml", "yml"),
            default_dir=default_dir,
        )
        if path:
            dpg.set_value(_CFG_PATH, path)
            self._set_status(f"Selected: {path}  (press Load to apply)")
        else:
            self._set_status("File selection cancelled.")

    def _cb_load(self) -> None:
        import dearpygui.dearpygui as dpg

        if self._busy():
            self._set_status("Cannot load a config while a run is in progress.")
            return
        path = dpg.get_value(_CFG_PATH).strip()
        if not path:
            self._set_status("Enter a config path (file or dir with db/ops.yaml).")
            return
        try:
            self.form.load_from_path(path)
            self._reset_run_ui()  # new config -> clear stale figures/status/log
            done = self._mark_completed_stages(self._resolve_output_dir(path))
            suffix = (
                f"  (already done: {', '.join(done)})"
                if done
                else "  (figures & status reset)"
            )
            self._set_status(f"Loaded config: {path}{suffix}")
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Load failed: {exc}")

    def _resolve_output_dir(self, path: str):
        """The processed asi/ folder a config was loaded from (for stage detection)."""
        p = Path(path)
        if p.is_dir():
            return p
        if p.suffix.lower() in (".yaml", ".yml"):
            return p.parent
        try:
            cfg = self.form.to_config()
        except Exception:  # noqa: BLE001
            return None
        if cfg.output_dir:
            return Path(cfg.output_dir)
        try:
            return default_output_dir(Path(cfg.input_dir), cfg.output_format)
        except Exception:  # noqa: BLE001
            return None

    def _mark_completed_stages(self, out_dir) -> list[str]:
        """Mark stages Done + Last run (artifact mtime) when outputs already exist."""
        if out_dir is None or not Path(out_dir).is_dir():
            return []
        import json
        from datetime import datetime

        from asvimg.runner import stage_artifacts

        # signatures the stages last *ran* with (persisted at run time); compare
        # to the current config so a project saved-but-not-rerun shows stale.
        runs: dict = {}
        try:
            rp = Path(out_dir) / ".stage_runs.json"
            if rp.exists():
                runs = json.loads(rp.read_text())
        except Exception:  # noqa: BLE001
            runs = {}
        try:
            cfg = self.form.collect()
        except Exception:  # noqa: BLE001
            cfg = {}
        rk = self._rois_key()

        done: list[str] = []
        for stage, artifact in stage_artifacts(out_dir).items():
            if artifact is None:
                continue
            s = str(stage)
            self._set_stage(s, "done", None)
            when = datetime.fromtimestamp(artifact.stat().st_mtime).strftime(
                "%Y/%m/%d %H:%M"
            )
            self._set_last_run(s, when)
            cur = self._stage_signature(cfg, rk, s)
            ran_hash = runs.get(s)
            if ran_hash is not None and ran_hash != self._sig_hash(cur):
                # ran with different params than the loaded config -> stale
                self._run_sig[s] = ("__param_changed__",)
            else:
                self._run_sig[s] = cur  # matches (or unknown) -> fresh baseline
            done.append(s)
        self._set_edit_rois_enabled("annotation" in done)
        self._refresh_stale()  # render "(* param-changed)" for mismatched stages
        return done

    def _reset_run_ui(self) -> None:
        """Clear everything tied to the previous config: figures, per-stage
        status, log, and the progress bar."""
        import dearpygui.dearpygui as dpg

        if self.figure_panels is not None:
            self.figure_panels.clear()
        self._set_edit_rois_enabled(False)
        self._reset_stage_status(["run_all"])  # every stage -> pending
        self._log_lines = []
        if dpg.does_item_exist(_LOG):
            dpg.set_value(_LOG, "")
        if dpg.does_item_exist(_PROGRESS):
            dpg.set_value(_PROGRESS, 0.0)
            dpg.configure_item(_PROGRESS, overlay="")

    @staticmethod
    def _preset_names(directory: Path) -> list[str]:
        """Sorted ``*.yaml`` / ``*.yml`` stems in ``directory`` (empty if none)."""
        if not directory.exists():
            return []
        return sorted(
            {p.stem for ext in ("*.yaml", "*.yml") for p in directory.glob(ext)}
        )

    def _list_presets(self) -> list[str]:
        """Dropdown entries: factory defaults, then standard (read-only) presets
        marked with ``_STD_PREFIX``, then the user's own presets."""
        std = [f"{_STD_PREFIX}{n}" for n in self._preset_names(_STD_PRESET_DIR)]
        user = self._preset_names(_USER_PRESET_DIR)
        return [_FACTORY_PRESET, *std, *user]

    @staticmethod
    def _resolve_preset(label: str) -> tuple[str, Path | None]:
        """Map a dropdown label to ``(kind, path)``; kind is
        ``'factory'`` | ``'std'`` | ``'user'`` (path is None for factory)."""
        if not label or label == _FACTORY_PRESET:
            return "factory", None
        if label.startswith(_STD_PREFIX):
            base, name, kind = _STD_PRESET_DIR, label[len(_STD_PREFIX) :], "std"
        else:
            base, name, kind = _USER_PRESET_DIR, label, "user"
        path = base / f"{name}.yaml"
        if not path.exists():
            path = base / f"{name}.yml"
        return kind, path

    def _refresh_preset_combo(self, select: str | None = None) -> None:
        import dearpygui.dearpygui as dpg

        if dpg.does_item_exist(_PRESET_COMBO):
            dpg.configure_item(_PRESET_COMBO, items=self._list_presets())
            if select is not None:
                dpg.set_value(_PRESET_COMBO, select)

    def _cb_load_preset(self) -> None:
        """Apply the preset selected in the dropdown.  '(factory defaults)'
        resets to the built-in PipelineConfig values.  db fields are kept."""
        import dearpygui.dearpygui as dpg

        label = (dpg.get_value(_PRESET_COMBO) or "").strip()
        kind, path = self._resolve_preset(label)
        try:
            if kind == "factory":
                self.form.restore_ops_defaults()
                self._set_status("Restored factory processing settings.")
                return
            if path is None or not path.exists():
                self._set_status(f"Preset not found: {label}")
                return
            self.form.apply_ops_preset(path)
            self._set_status(f"Loaded preset <- {path.name}")
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Load preset failed: {exc}")

    def _cb_save_preset(self) -> None:
        """Save the current settings as a named ``.yaml`` preset in the *user*
        presets folder.  Standard (bundled) presets are read-only and cannot be
        overwritten; prompts before overwriting an existing user preset.  db
        fields are not stored."""
        import dearpygui.dearpygui as dpg

        raw = (dpg.get_value(_PRESET_NAME) or "").strip()
        if not raw:
            self._set_status("Enter a preset name before saving.")
            return
        name = Path(raw).stem  # strip any path / extension -> safe file stem
        if not name or name == _FACTORY_PRESET or name.startswith("["):
            self._set_status(f"Invalid preset name: {raw!r}")
            return
        # Standard presets are read-only: refuse a name that collides with one.
        if name in self._preset_names(_STD_PRESET_DIR):
            self._set_status(
                f"'{name}' is a standard preset (read-only) — choose another name."
            )
            return
        # Validate the config up front so we never prompt-to-overwrite for a
        # config that cannot be saved anyway.
        try:
            self.form.to_config()
        except (ValueError, TypeError) as exc:
            self._set_status(f"Cannot save preset (invalid config): {exc}")
            return
        target = _USER_PRESET_DIR / f"{name}.yaml"
        if target.exists():
            self._confirm_overwrite_preset(name, target)
        else:
            self._write_preset(name, target)

    def _write_preset(self, name: str, target: Path) -> None:
        """Write the current ops to ``target`` (a user preset) and refresh."""
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            self.form.save_ops(target)
        except (ValueError, TypeError) as exc:
            self._set_status(f"Cannot save preset (invalid config): {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Save preset failed: {exc}")
            return
        self._refresh_preset_combo(select=name)
        self._set_status(f"Saved preset -> {name}.yaml")

    def _confirm_overwrite_preset(self, name: str, target: Path) -> None:
        """Modal confirm before overwriting an existing preset file."""
        import dearpygui.dearpygui as dpg

        tag = "_preset_overwrite_modal"
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)

        def _overwrite():
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
            self._write_preset(name, target)

        def _cancel():
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
            self._set_status("Save preset cancelled (kept existing file).")

        w, h = 380, 130
        try:
            vw = dpg.get_viewport_client_width()
            vh = dpg.get_viewport_client_height()
            pos = [max(0, (vw - w) // 2), max(0, (vh - h) // 2)]
        except Exception:  # noqa: BLE001 — fall back to a fixed position
            pos = [560, 380]
        with dpg.window(
            label="Overwrite preset?",
            tag=tag,
            modal=True,
            no_close=True,
            no_resize=True,
            width=w,
            height=h,
            pos=pos,
        ):
            dpg.add_text(f"Preset '{name}.yaml' already exists.")
            dpg.add_text("Overwrite it?")
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(label="Overwrite", width=120, callback=_overwrite)
                dpg.add_button(label="Cancel", width=120, callback=_cancel)

    def _cb_save(self) -> None:
        """Save db.yaml + ops.yaml back to the loaded config location (the path
        in the text box); fall back to the output dir when it is empty."""
        import dearpygui.dearpygui as dpg

        try:
            config = self.form.to_config()
        except (ValueError, TypeError) as exc:
            self._set_status(f"Cannot save invalid config: {exc}")
            return
        cfg_path = dpg.get_value(_CFG_PATH).strip()
        if cfg_path:
            p = Path(cfg_path)
            # a file (db.yaml / ops.yaml / legacy .yaml) -> its folder; else a dir
            out = p.parent if p.suffix.lower() in (".yaml", ".yml") else p
        else:
            out = (
                Path(config.output_dir)
                if config.output_dir
                else default_output_dir(Path(config.input_dir), config.output_format)
            )
        try:
            out.mkdir(parents=True, exist_ok=True)
            self.form.save_to(out)
            self._set_status(f"Saved db.yaml + ops.yaml -> {out}")
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Save failed: {exc}")

    def _cb_read_last(self) -> None:
        """Recall the processing settings from the previous session
        (~/.asovi/last_ops.yaml, auto-saved on exit). db fields are kept."""
        if not _LAST_OPS.exists():
            self._set_status(f"No last-session config saved yet ({_LAST_OPS}).")
            return
        try:
            self.form.apply_ops_preset(_LAST_OPS)
            self._set_status(f"Loaded last-session processing settings <- {_LAST_OPS}")
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Read last config failed: {exc}")

    def _save_last_ops(self) -> None:
        """Persist the current processing settings so 'Read last config' can
        recall them next session. Best-effort; never blocks shutdown."""
        try:
            self.form.save_ops(_LAST_OPS)
        except Exception:  # noqa: BLE001 — invalid config / IO: skip silently on exit
            pass

    # --- event handling (main thread, per frame) --------------------------

    def _safe_handle(self, ev: tuple) -> None:
        """Handle one event, never letting a bad event tear down the loop."""
        try:
            self._handle(ev)
        except Exception as exc:  # noqa: BLE001 — keep the render loop alive
            try:
                self._append_log(f"!! [gui] event handler error: {exc!r}")
            except Exception:
                pass

    def _handle(self, ev: tuple) -> None:
        import dearpygui.dearpygui as dpg

        kind = ev[0]
        if kind == "stage":
            _, stage, status, message, elapsed = ev
            self._set_stage(stage, status, elapsed)
            if status in _TERMINAL_STATUS:  # stamp when it finished this session
                from datetime import datetime

                when = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                if elapsed is not None:
                    when += f" ({elapsed:.1f}s)"
                self._set_last_run(stage, when)
            if status == "done":
                self._stamp_stage_signature(stage)  # fresh baseline for this stage
                self._persist_stage_sig(stage)  # + across sessions
            if stage == "annotation" and status == "done":
                self._set_edit_rois_enabled(True)
            if message:
                self._append_log(f"[{stage}] {status}: {message}")
        elif kind == "progress":
            _, stage, current, total, message = ev
            frac = (current / total) if total else 0.0
            dpg.set_value(_PROGRESS, max(0.0, min(1.0, frac)))
            overlay = f"{stage} {current}/{total}" + (f" {message}" if message else "")
            dpg.configure_item(_PROGRESS, overlay=overlay)
        elif kind == "log":
            _, stage, text, level = ev
            prefix = "!! " if level == "error" else ""
            self._append_log(
                prefix + (text if text.startswith("[") else f"[{stage}] {text}")
            )
        elif kind == "figure":
            _, stage, key, w, h, data = ev
            if self.figure_panels is not None:
                self.figure_panels.add_rgba(f"{stage}:{key}", w, h, data)
        elif kind == "worker_done":
            self._set_status("Idle." if not self.bridge.cancel.is_set() else "Stopped.")
            self._update_movie_btn()  # export may have just written movies

    def _append_log(self, line: str) -> None:
        import dearpygui.dearpygui as dpg

        self._log_lines.append(line)
        if len(self._log_lines) > 500:
            self._log_lines = self._log_lines[-500:]
        dpg.set_value(_LOG, "\n".join(self._log_lines))

    def _set_status(self, text: str) -> None:
        import dearpygui.dearpygui as dpg

        if dpg.does_item_exist(_STATUS):
            dpg.set_value(_STATUS, text)

    def _stage_tag(self, stage: str) -> str:
        return f"stagestat_{stage}"

    def _stage_last_tag(self, stage: str) -> str:
        return f"stagelast_{stage}"

    def _set_stage(self, stage: str, status: str, elapsed) -> None:
        import dearpygui.dearpygui as dpg

        tag = self._stage_tag(stage)
        if not dpg.does_item_exist(tag):
            return
        dpg.set_value(tag, _STATE_WORD.get(status, "Pending"))
        dpg.configure_item(
            tag, color=_STATUS_COLOR.get(status, _STATUS_COLOR["pending"])
        )

    def _set_last_run(self, stage: str, text: str) -> None:
        import dearpygui.dearpygui as dpg

        tag = self._stage_last_tag(stage)
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, text)

    # --- staleness (Done -> "Done (* param-changed)") --------------------

    def _rois_key(self) -> str:
        # Both ROI files feed staleness: rois.csv (atlas space) and
        # rois_source.csv (source space). Either edit should re-flag ROI /
        # correlation.
        _, out = self._configured_dirs()
        if out is None:
            return ""
        parts = []
        for fn in ("rois.csv", "rois_source.csv"):
            p = Path(out) / fn
            try:
                parts.append(p.read_text(encoding="utf-8") if p.exists() else "")
            except Exception:  # noqa: BLE001
                parts.append("")
        return "\x1e".join(parts)

    def _ica_sel_key(self) -> str:
        """Content of ica_exclusion.json — the IC choice, which downstream stages
        depend on ONLY while denoising is on (see _stage_signature)."""
        _, out = self._configured_dirs()
        if out is None:
            return ""
        p = Path(out) / "ica_exclusion.json"
        try:
            return p.read_text(encoding="utf-8") if p.exists() else ""
        except Exception:  # noqa: BLE001
            return ""

    def _stage_signature(self, cfg: dict, rois_key: str, stage: str,
                         *, as_upstream: bool = False):
        denoising = str(cfg.get("ica_denoise", "off")) != "off"
        output_only = _STAGE_OUTPUT_PARAMS.get(stage, set()) if as_upstream else set()
        parts = []
        for f in _STAGE_PARAMS.get(stage, []):
            if f in output_only:
                continue
            if f == "__rois__":
                parts.append(("__rois__", rois_key))
            elif f == "__ica_sel__":
                # Only while denoising: otherwise re-picking ICs (a QC action that
                # changes no output) would mark ROI/export/correlation stale for
                # everyone who never asked for denoising.
                parts.append(("__ica_sel__", self._ica_sel_key() if denoising else ""))
            else:
                parts.append((f, _hashable(cfg.get(f))))
        for up in _STAGE_UPSTREAM.get(stage, []):
            parts.append(("^" + up, self._stage_signature(cfg, rois_key, up, as_upstream=True)))
        # Denoised outputs are built from the ICA basis, so they also depend on the
        # ICA stage — but only then, or an unrelated ica_random_state tweak would
        # mark ROI stale for a user who never turned denoising on.
        if denoising and stage in ("roi", "correlation", "export"):
            parts.append(("^ica", self._stage_signature(cfg, rois_key, "ica", as_upstream=True)))
        return tuple(parts)

    def _snapshot_run_inputs(self) -> None:
        """Capture the config/rois that a run is about to use (for staleness)."""
        try:
            self._run_cfg_dict = self.form.collect()
        except Exception:  # noqa: BLE001
            self._run_cfg_dict = {}
        self._run_rois_key = self._rois_key()

    def _stamp_stage_signature(self, stage: str) -> None:
        self._run_sig[stage] = self._stage_signature(
            self._run_cfg_dict, self._run_rois_key, stage
        )

    @staticmethod
    def _sig_hash(sig) -> str:
        import hashlib

        return hashlib.sha1(repr(sig).encode("utf-8")).hexdigest()

    def _persist_stage_sig(self, stage: str) -> None:
        """Record the signature a stage ran with, so a later reload can tell
        whether the on-disk result still matches the (possibly edited) config."""
        import json

        _, out = self._configured_dirs()
        if out is None:
            return
        p = Path(out) / ".stage_runs.json"
        try:
            data = json.loads(p.read_text()) if p.exists() else {}
        except Exception:  # noqa: BLE001
            data = {}
        data[stage] = self._sig_hash(
            self._stage_signature(self._run_cfg_dict, self._run_rois_key, stage)
        )
        try:
            p.write_text(json.dumps(data))
        except Exception:  # noqa: BLE001
            pass

    def _refresh_stale(self) -> None:
        """Mark any Done stage whose inputs changed since it ran as stale."""
        import dearpygui.dearpygui as dpg

        try:
            cfg = self.form.collect()
        except Exception:  # noqa: BLE001
            return
        rk = self._rois_key()
        for _method, stage in _STAGES:
            s = str(stage)
            tag = self._stage_tag(s)
            if not dpg.does_item_exist(tag):
                continue
            val = dpg.get_value(tag)
            if not val.startswith("Done"):
                continue
            snap = self._run_sig.get(s)
            stale = snap is not None and snap != self._stage_signature(cfg, rk, s)
            want = "Done (* param-changed)" if stale else "Done"
            if val != want:
                dpg.set_value(tag, want)
                dpg.configure_item(
                    tag, color=_PARAM_CHANGED if stale else _STATUS_COLOR["done"]
                )

    def _reset_stage_status(self, method_names: list[str]) -> None:
        import dearpygui.dearpygui as dpg

        run_all = "run_all" in method_names
        for method, stage in _STAGES:
            tag = self._stage_tag(str(stage))
            if not dpg.does_item_exist(tag):
                continue
            if run_all or method in method_names:
                dpg.set_value(tag, _STATE_WORD["pending"])
                dpg.configure_item(tag, color=_STATUS_COLOR["pending"])
                self._set_last_run(str(stage), "-")

    # --- UI construction --------------------------------------------------

    @staticmethod
    def _config_header_theme():
        """Tint the section headers (Data, Channels, ...) a light blue.

        Bound to the config pane so it cascades only to *its* collapsing
        headers, leaving the figure panels on the default dark theme.
        """
        import dearpygui.dearpygui as dpg

        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvCollapsingHeader):
                dpg.add_theme_color(
                    dpg.mvThemeCol_Header, (38, 78, 110), category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_HeaderHovered,
                    (58, 108, 145),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_HeaderActive,
                    (48, 96, 130),
                    category=dpg.mvThemeCat_Core,
                )
        return theme

    @staticmethod
    def _orange_header_theme():
        """Tint the Presets collapsing header orange so it stands apart from
        the (blue) config sections."""
        import dearpygui.dearpygui as dpg

        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvCollapsingHeader):
                dpg.add_theme_color(
                    dpg.mvThemeCol_Header, (170, 90, 25), category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_HeaderHovered,
                    (205, 120, 40),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_HeaderActive,
                    (190, 105, 30),
                    category=dpg.mvThemeCat_Core,
                )
        return theme

    @staticmethod
    def _magenta_button_theme():
        """Magenta styling for the Run All button so it stands out."""
        import dearpygui.dearpygui as dpg

        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(
                    dpg.mvThemeCol_Button, (170, 30, 150), category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonHovered,
                    (205, 55, 180),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonActive,
                    (150, 20, 130),
                    category=dpg.mvThemeCat_Core,
                )
        return theme

    @staticmethod
    def _progress_theme():
        """Medium-grey fill for the progress bar (~50% of the earlier tint)."""
        import dearpygui.dearpygui as dpg

        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(
                    dpg.mvThemeCol_PlotHistogram,
                    (112, 112, 112),
                    category=dpg.mvThemeCat_Core,
                )
        return theme

    def _build_ui(self) -> None:
        import dearpygui.dearpygui as dpg

        dpg.add_texture_registry(tag=_TEXREG)

        with dpg.window(tag=_PRIMARY):
            with dpg.menu_bar():
                with dpg.menu(label="File"):
                    dpg.add_menu_item(label="Load config", callback=self._cb_load)
                    dpg.add_menu_item(label="Save config", callback=self._cb_save)
                with dpg.menu(label="Run"):
                    dpg.add_menu_item(label="Run all", callback=self._cb_run_all)
                    dpg.add_menu_item(label="Stop", callback=self._cb_stop)

            dpg.add_text("Idle.", tag=_STATUS)

            with dpg.group(horizontal=True):
                # Left: config editor (section headers tinted light blue)
                with dpg.child_window(width=540, autosize_y=True) as cfg_cw:
                    dpg.add_text("Configuration  (* = reproducibility/db field)")
                    dpg.add_separator()
                    with dpg.collapsing_header(
                        label="Presets (ops.yaml)", default_open=True
                    ) as presets_hdr:
                        dpg.add_input_text(
                            tag=_CFG_PATH,
                            width=420,
                            default_value=self._initial_config_path or "",
                            hint="config .yaml or dir with db.yaml/ops.yaml — Load, or '...' to browse",
                        )
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="...", callback=self._cb_browse)
                            with dpg.tooltip(dpg.last_item()):
                                dpg.add_text(
                                    "Browse for ops.yaml / db.yaml (system file dialog)"
                                )
                            dpg.add_button(label="Load", callback=self._cb_load)
                            dpg.add_button(label="Save", callback=self._cb_save)
                            dpg.add_separator()
                        with dpg.group(horizontal=True):
                            dpg.add_combo(
                                self._list_presets(),
                                tag=_PRESET_COMBO,
                                width=220,
                                default_value=_FACTORY_PRESET,
                            )
                            dpg.add_button(
                                label="Load preset", callback=self._cb_load_preset
                            )
                            with dpg.tooltip(dpg.last_item()):
                                dpg.add_text(
                                    "Apply the selected preset's processing settings.\n"
                                    "'(factory defaults)' resets to the built-in values;\n"
                                    "'[std] ...' are bundled read-only presets, the rest\n"
                                    "are your own. db (input/output) fields are kept."
                                )
                        with dpg.group(horizontal=True):
                            dpg.add_input_text(
                                tag=_PRESET_NAME, width=220, hint="new preset name"
                            )
                            dpg.add_button(
                                label="Save as preset", callback=self._cb_save_preset
                            )
                            with dpg.tooltip(dpg.last_item()):
                                dpg.add_text(
                                    "Save the current settings as a named .yaml preset\n"
                                    "in your user presets folder (~/.asovi/presets).\n"
                                    "Standard '[std] ...' presets are read-only. db\n"
                                    "fields are not stored."
                                )
                        dpg.add_button(
                            label="Read last config", callback=self._cb_read_last
                        )
                        with dpg.tooltip(dpg.last_item()):
                            dpg.add_text(
                                "Recall the processing settings from your "
                                "previous session\n(~/.asovi/last_ops.yaml, "
                                "auto-saved on exit). db fields are kept."
                            )
                    dpg.bind_item_theme(presets_hdr, self._orange_header_theme())
                    dpg.add_separator()
                    self.form.build(parent=cfg_cw, section_footer=self._section_footer)
                dpg.bind_item_theme(cfg_cw, self._config_header_theme())

                # Right: stages + figures + log
                with dpg.child_window(autosize_x=True, autosize_y=True):
                    with dpg.group(horizontal=True):
                        dpg.add_text("Stages")
                        run_all_btn = dpg.add_button(
                            label="Run All", callback=self._cb_run_all
                        )
                        dpg.bind_item_theme(run_all_btn, self._magenta_button_theme())
                        dpg.add_button(label="Stop", callback=self._cb_stop)
                        dpg.add_button(
                            label="Quick Preview",
                            callback=self._cb_preview,
                        )
                        with dpg.tooltip(dpg.last_item()):
                            dpg.add_text(
                                "Show the first 12 frames of the FIRST input file\n"
                                "(4x3), labelled by channel - check G/R mapping."
                            )
                        dpg.add_button(
                            label="Quick Preview (All)",
                            callback=self._cb_preview_all,
                        )
                        with dpg.tooltip(dpg.last_item()):
                            dpg.add_text(
                                "Show the first 4 frames of EVERY input file\n"
                                "(one row per file), labelled by channel - spot a\n"
                                "file whose cycle starts on the wrong channel,\n"
                                "then fix it with channels_slip."
                            )
                        dpg.add_button(
                            label="QC-frame slip",
                            callback=self._cb_qc_frame_slip,
                        )
                        with dpg.tooltip(dpg.last_item()):
                            dpg.add_text(
                                "Embed the first ~60 frames of EVERY file in 2D by\n"
                                "image CONTENT and flag files whose source/donner\n"
                                "assignment lands in the wrong cluster. Catches a\n"
                                "phase slip the intensity demux QC misses when the\n"
                                "channels differ in content but not brightness\n"
                                "(per-file verdicts go to the log pane)."
                            )
                    with dpg.table(
                        tag=_STAGE_TABLE,
                        header_row=True,
                        policy=dpg.mvTable_SizingStretchProp,
                        borders_innerH=True,
                        borders_outerH=True,
                        borders_innerV=True,
                        borders_outerV=True,
                    ):
                        dpg.add_table_column(label="State")
                        dpg.add_table_column(label="Process")
                        dpg.add_table_column(label="Last run")
                        dpg.add_table_column(label="UI")
                        for method, stage in _STAGES:
                            with dpg.table_row():
                                dpg.add_text(
                                    "Pending",
                                    tag=self._stage_tag(str(stage)),
                                    color=_STATUS_COLOR["pending"],
                                )
                                dpg.add_text(stage.value)
                                dpg.add_text("-", tag=self._stage_last_tag(str(stage)))
                                with dpg.group(horizontal=True):
                                    dpg.add_button(
                                        label="Show",
                                        user_data=method,
                                        callback=self._cb_show_result,
                                    )
                                    with dpg.tooltip(dpg.last_item()):
                                        dpg.add_text(
                                            "Show the previous result "
                                            "figures from disk (no re-run)."
                                        )
                                    dpg.add_button(
                                        label="Run",
                                        user_data=method,
                                        callback=self._cb_run_stage,
                                    )
                    progress = dpg.add_progress_bar(
                        tag=_PROGRESS, default_value=0.0, width=-1, overlay=""
                    )
                    dpg.bind_item_theme(progress, self._progress_theme())
                    dpg.add_separator()
                    dpg.add_text("Utilities")
                    with dpg.group(horizontal=True):
                        dpg.add_button(
                            label="Open input folder", callback=self._cb_open_input
                        )
                        dpg.add_button(
                            label="Open output folder", callback=self._cb_open_output
                        )
                        self._edit_rois_btn = dpg.add_button(
                            label="Edit ROIs",
                            callback=self._cb_edit_rois,
                            enabled=False,
                        )
                        with dpg.tooltip(self._edit_rois_btn):
                            dpg.add_text(
                                "Edit atlas ROIs on the standard brain\n"
                                "(enabled after annotation)."
                            )
                        dpg.add_button(
                            label="Seed-based Corr. Maps", callback=self._cb_corr_map
                        )
                        with dpg.tooltip(dpg.last_item()):
                            dpg.add_text(
                                "Seed-ROI correlation maps in atlas space; save to\n"
                                "asi/corrMap and optionally add to map_for_annot\n"
                                "(needs annotation done)."
                            )
                        self._movie_btn = dpg.add_button(
                            label="Preview Movie",
                            callback=self._cb_movie_preview,
                            enabled=False,
                        )
                        with dpg.tooltip(self._movie_btn):
                            dpg.add_text(
                                "Play all exported dF/F movies side by side "
                                "(enabled once export has written movies)."
                            )
                    dpg.add_separator()
                    dpg.add_text("Figures")
                    dpg.add_group(tag=_FIGPANELS)
                    dpg.add_separator()
                    dpg.add_text("Log")
                    dpg.add_input_text(
                        tag=_LOG,
                        multiline=True,
                        readonly=True,
                        width=-1,
                        height=-1,
                        default_value="",
                    )

        self.figure_panels = FigurePanels(_FIGPANELS, _TEXREG)

    # --- run --------------------------------------------------------------

    def run(self) -> None:
        import dearpygui.dearpygui as dpg

        dpg.create_context()
        self._build_ui()
        if self._initial_config_path:
            try:
                self.form.load_from_path(self._initial_config_path)
            except Exception:
                pass
        icon = Path(__file__).resolve().parent / "assets" / "asovi_icon.ico"
        dpg.create_viewport(title="ASoVi Imager - Pipeline", width=1500, height=900)
        if icon.exists():
            try:  # icon is cosmetic; never let it block startup
                dpg.set_viewport_small_icon(str(icon))
                dpg.set_viewport_large_icon(str(icon))
            except Exception:  # noqa: BLE001
                pass
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window(_PRIMARY, True)
        # Frame-rate cap. vsync (viewport default) SHOULD pace the loop, but it is
        # not always honoured — over remote desktop, some GPU drivers, software
        # rendering — and then the loop spins uncapped, pegs a CPU core, and Windows
        # shows the busy/"app starting" cursor the whole session. Sleeping the
        # remainder of each frame's budget yields the core. render_dearpygui_frame()
        # pumps the OS message queue every call, so a <=33 ms sleep never makes the
        # window "not responding". If vsync IS honoured the frame already ate the
        # budget and the sleep is ~0, so this is safe either way. 30 fps idle / 60
        # fps while a run or a movie preview needs smooth updates.
        busy_dt, idle_dt = 1.0 / 60.0, 1.0 / 30.0
        try:
            while dpg.is_dearpygui_running():
                frame_t0 = time.perf_counter()
                self.bridge.drain(self._safe_handle)
                if self._movie_preview is not None:
                    self._movie_preview.tick()  # advance any open movie preview
                self._stale_tick += 1
                if self._stale_tick >= 20 and not self._busy():
                    self._stale_tick = 0
                    try:
                        self._refresh_stale()  # flag Done stages whose inputs changed
                        self._update_movie_btn()  # enable once export wrote movies
                    except Exception:  # noqa: BLE001
                        pass
                dpg.render_dearpygui_frame()
                target = busy_dt if (self._busy() or self._movie_preview is not None) else idle_dt
                remaining = target - (time.perf_counter() - frame_t0)
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            if self._busy():
                self.bridge.request_cancel()
            self._save_last_ops()  # remember this session for "Read last config"
            dpg.destroy_context()


def run_app(config_path: str | None = None) -> None:
    DashboardApp(config_path).run()
