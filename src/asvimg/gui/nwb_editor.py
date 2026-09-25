"""Standalone GUI to build an NWB file from a processed output folder (asovi-nwb).

Separate from the pipeline dashboard (it owns its own Dear PyGui context and
render loop, and is never imported by ``app.py`` — one dpg context per process).
It collects the metadata NWB / DANDI require but the pipeline never captured
(Subject, session, device, per-channel optics, physical pixel size), persists it
to the ``nwb_metadata.yaml`` sidecar, and calls :func:`asvimg.nwb_export.write_nwb`
to package the processed artifacts.

Run it standalone::

    uv run python -m asvimg.gui.nwb_editor [output_dir | config_dir | config.yaml]

Point it at a processed ``asi/<fmt>`` folder (the one holding ``reg_meta.npz`` +
``db.yaml`` / ``ops.yaml``); use *Browse* to repoint at another.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from asvimg import (
    PipelineConfig,
    default_output_dir,
    load_config,
)
from asvimg import nwb_meta as nm
from asvimg.nwb_meta import (
    ChannelMeta,
    DeviceMeta,
    ExportOptions,
    NwbMetadata,
    SessionMeta,
    SubjectMeta,
)

_T = "nwbed_"
_WIN = _T + "win"
_STATUS = _T + "status"
_LOG = _T + "log"
_PROGRESS = _T + "progress"
_SUMMARY = _T + "summary"
_PATH = _T + "path"


def _fmt(x) -> str:
    return "" if x is None else str(x)


def _pf(s):
    """Parse a float, or None for blank/garbage (so 'unset' round-trips)."""
    s = str(s).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _split(s) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


class _GuiReporter:
    """ProgressReporter that only mutates plain fields (thread-safe); the render
    loop drains them onto widgets (dpg must be touched on the main thread)."""

    def __init__(self, editor: "NwbEditor") -> None:
        self.e = editor

    def on_stage(self, stage, status, **_k) -> None:
        with self.e._lock:
            self.e._pending.append(f"[{status}] {stage}")

    def on_progress(self, stage, current, total, **_k) -> None:
        with self.e._lock:
            self.e._progress = (int(current), int(total))

    def on_log(self, stage, text, **_k) -> None:
        with self.e._lock:
            self.e._pending.append(str(text))

    def on_figure(self, *_a, **_k) -> None:
        pass


class NwbEditor:
    """Build (:meth:`build`) + run (:meth:`run`) the NWB-builder window.

    ``build`` only creates widgets in the current dpg context (smoke-testable);
    ``run`` owns the standalone context + render loop and returns the written path.
    """

    def __init__(self, config: PipelineConfig, output_dir: str | Path | None = None) -> None:
        self.config = config
        self.out_dir = Path(output_dir) if output_dir else (
            Path(config.output_dir) if config.output_dir
            else default_output_dir(Path(config.input_dir), config.output_format)
        )
        self._reload_meta()

        self._lock = threading.Lock()
        self._pending: list[str] = []
        self._progress: tuple[int, int] | None = None
        self._status_pending: str | None = None
        self._write_thread: threading.Thread | None = None
        self._inspect_thread: threading.Thread | None = None
        self.last_written: Path | None = None
        self.should_close = False
        self._reporter = _GuiReporter(self)

    # ---- data (no dpg) ---------------------------------------------------

    def _reload_meta(self) -> None:
        self.summary = nm.detect_artifacts(self.config, self.out_dir)
        self.meta = nm.prefill(self.config, self.out_dir)
        self._channel_names = [
            n for n, g in self.summary["groups"].items() if g["has_source"]
        ]

    def _summary_text(self) -> str:
        s = self.summary
        gpar = []
        for name, g in s["groups"].items():
            flags = "".join([
                "d" if g["dff"] else "-",
                "r" if g["roi_signals"] else "-",
                "w" if g["dfWarped"] else "-",
                "i" if g["basis"] else "-",
            ])
            gpar.append(f"{name}[{flags}]")
        return (
            f"detected: fps={s['fps']} cycle={s['cycle_len']} "
            f"H×W={s['H']}×{s['W']} binning={s['binning']} | "
            f"groups(d=dff,r=roi,w=warped,i=ica): {', '.join(gpar) or '(none)'} | "
            f"marks:{'Y' if s['marks'] else '-'} rois:{'Y' if s['rois_csv'] else '-'} "
            f"rois_src:{'Y' if s['rois_source'] else '-'} "
            f"atlas:{'Y' if s['atlas_available'] else '-'} roi_space={s['roi_space']}"
        )

    # ---- widget access ---------------------------------------------------

    def _ft(self, key: str) -> str:
        return _T + key

    def _gv(self, key: str, default=""):
        import dearpygui.dearpygui as dpg

        tag = self._ft(key)
        return dpg.get_value(tag) if dpg.does_item_exist(tag) else default

    def _go(self, name: str, default):
        return self._gv("opt_" + name, default)

    def collect_meta(self) -> NwbMetadata:
        g = self._gv
        session = SessionMeta(
            session_description=g("f_session_description"),
            session_id=g("f_session_id"),
            session_start_time=g("f_session_start_time"),
            experimenter=_split(g("f_experimenter")),
            lab=g("f_lab"),
            institution=g("f_institution"),
            experiment_description=g("f_experiment_description"),
            keywords=_split(g("f_keywords")),
            related_publications=_split(g("f_related_publications")),
        )
        subject = SubjectMeta(
            subject_id=g("f_subject_id"), species=g("f_species"),
            sex=g("f_sex", "U"), age=g("f_age"), date_of_birth=g("f_date_of_birth"),
            genotype=g("f_genotype"), strain=g("f_strain"), description=g("f_description"),
        )
        device = DeviceMeta(
            name=g("f_dev_name"), description=g("f_dev_description"),
            manufacturer=g("f_dev_manufacturer"), model=g("f_dev_model"),
        )
        channels = {
            name: ChannelMeta(
                excitation_lambda=_pf(g(f"ch_{name}_excitation")),
                emission_lambda=_pf(g(f"ch_{name}_emission")),
                indicator=g(f"ch_{name}_indicator"),
                location=g(f"ch_{name}_location", "dorsal cortex"),
                exposure_time=_pf(g(f"ch_{name}_exposure")),
            )
            for name in self._channel_names
        }
        options = ExportOptions(
            include_dff=bool(self._go("include_dff", True)),
            include_warped=bool(self._go("include_warped", False)),
            include_reference_images=bool(self._go("include_reference_images", True)),
            include_roi=bool(self._go("include_roi", True)),
            include_ica=bool(self._go("include_ica", False)),
            include_correlation=bool(self._go("include_correlation", False)),
            honor_ica_denoise=bool(self._go("honor_ica_denoise", True)),
            compression=("gzip" if self._go("compression_on", True) else "none"),
            compression_opts=int(self._go("compression_opts", 4)),
            shuffle=bool(self._go("shuffle", True)),
            overwrite=bool(self._go("overwrite", False)),
            output_filename=g("opt_output_filename"),
        )
        return NwbMetadata(session, subject, device, channels, _pf(g("f_pixel_size_um")), options)

    # ---- build -----------------------------------------------------------

    def _text(self, key: str, label: str, value, *, width: int = 380,
              multiline: bool = False, height: int = 0) -> None:
        import dearpygui.dearpygui as dpg

        dpg.add_input_text(
            label=label, tag=self._ft(key), default_value=_fmt(value),
            width=width, multiline=multiline, height=(height or 60) if multiline else 0,
        )

    def build(self) -> None:
        import dearpygui.dearpygui as dpg

        if dpg.does_item_exist(_WIN):
            dpg.delete_item(_WIN)

        with dpg.window(tag=_WIN, label="NWB Builder"):
            with dpg.group(horizontal=True):
                dpg.add_text("Output folder:")
                dpg.add_input_text(tag=_PATH, default_value=str(self.out_dir),
                                   width=460, readonly=True)
                dpg.add_button(label="Browse", callback=self._cb_browse)
                dpg.add_button(label="Reload", callback=self._cb_reload)
                dpg.add_button(label="Open folder", callback=self._cb_open_folder)
            dpg.add_text(self._summary_text(), tag=_SUMMARY, wrap=980)
            dpg.add_separator()

            with dpg.child_window(height=-150):
                s, sub, dev, o = (self.meta.session, self.meta.subject,
                                  self.meta.device, self.meta.options)

                with dpg.collapsing_header(label="Session", default_open=True):
                    self._text("f_session_description", "session_description", s.session_description, width=520)
                    self._text("f_session_id", "session_id", s.session_id, width=300)
                    self._text("f_session_start_time", "session_start_time (ISO-8601, tz-aware)",
                               s.session_start_time or "2026-01-01T00:00:00+09:00", width=320)
                    self._text("f_experimenter", "experimenter (comma-separated, 'Last, First')",
                               ", ".join(s.experimenter), width=420)
                    self._text("f_lab", "lab", s.lab, width=260)
                    self._text("f_institution", "institution", s.institution, width=320)
                    self._text("f_experiment_description", "experiment_description",
                               s.experiment_description, width=520, multiline=True)
                    self._text("f_keywords", "keywords (comma-separated)", ", ".join(s.keywords), width=420)
                    self._text("f_related_publications", "related_publications (DOIs, comma-separated)",
                               ", ".join(s.related_publications), width=420)

                with dpg.collapsing_header(label="Subject", default_open=True):
                    self._text("f_subject_id", "subject_id", sub.subject_id, width=240)
                    self._text("f_species", "species (Latin binomial)", sub.species, width=260)
                    dpg.add_combo(("M", "F", "O", "U"), label="sex", tag=self._ft("f_sex"),
                                  default_value=sub.sex or "U", width=80)
                    self._text("f_age", "age (ISO-8601 duration, e.g. P90D)", sub.age, width=200)
                    self._text("f_date_of_birth", "date_of_birth (ISO-8601, optional)",
                               sub.date_of_birth, width=260)
                    self._text("f_genotype", "genotype", sub.genotype, width=360)
                    self._text("f_strain", "strain", sub.strain, width=240)
                    self._text("f_description", "description", sub.description, width=360)

                with dpg.collapsing_header(label="Device"):
                    self._text("f_dev_name", "device name", dev.name, width=260)
                    self._text("f_dev_description", "description", dev.description, width=420)
                    self._text("f_dev_manufacturer", "manufacturer", dev.manufacturer, width=260)
                    self._text("f_dev_model", "model", dev.model, width=260)

                with dpg.collapsing_header(label="Optical channels", default_open=True):
                    dpg.add_text("One row per source-bearing group (channels_name). "
                                 "Blank λ leaves it unset.", wrap=900)
                    for name in self._channel_names:
                        ch = self.meta.channels.get(name) or ChannelMeta()
                        dpg.add_separator()
                        dpg.add_text(f"Group: {name}")
                        self._text(f"ch_{name}_excitation", "excitation λ (nm)",
                                   _fmt(ch.excitation_lambda), width=120)
                        self._text(f"ch_{name}_emission", "emission λ (nm)",
                                   _fmt(ch.emission_lambda), width=120)
                        self._text(f"ch_{name}_indicator", "indicator (e.g. GCaMP6s)",
                                   ch.indicator, width=240)
                        self._text(f"ch_{name}_location", "location", ch.location or "dorsal cortex", width=240)
                        self._text(f"ch_{name}_exposure", "exposure_time (s)",
                                   _fmt(ch.exposure_time), width=120)
                    dpg.add_separator()
                    self._text("f_pixel_size_um", "pixel_size_um (binned pixel pitch)",
                               _fmt(self.meta.pixel_size_um), width=140)

                with dpg.collapsing_header(label="Payloads & options", default_open=True):
                    for name, label, val in [
                        ("include_dff", "dF/F (source)", o.include_dff),
                        ("include_roi", "ROI signals + segmentation", o.include_roi),
                        ("include_warped", "atlas-warped dF/F (needs marks.mat + atlas)", o.include_warped),
                        ("include_reference_images", "reference images (mean + template)", o.include_reference_images),
                        ("include_ica", "ICA components", o.include_ica),
                        ("include_correlation", "correlation matrices", o.include_correlation),
                        ("honor_ica_denoise", "honor ica_denoise (read denoised dF/F)", o.honor_ica_denoise),
                        ("compression_on", "gzip compression", o.compression != "none"),
                        ("shuffle", "shuffle filter", o.shuffle),
                        ("overwrite", "overwrite existing file", o.overwrite),
                    ]:
                        dpg.add_checkbox(label=label, tag=self._ft("opt_" + name), default_value=bool(val))
                    dpg.add_input_int(label="compression level (gzip 0-9)",
                                      tag=self._ft("opt_compression_opts"),
                                      default_value=int(o.compression_opts),
                                      min_value=0, max_value=9, min_clamped=True,
                                      max_clamped=True, width=120, step=1)
                    self._text("opt_output_filename", "output filename (blank = <exp>.nwb)",
                               o.output_filename, width=260)

            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(label="Load metadata", callback=self._cb_load_meta)
                dpg.add_button(label="Save metadata", callback=self._cb_save_meta)
                dpg.add_button(label="Validate", callback=self._cb_validate)
                dpg.add_button(label="Write NWB", callback=self._cb_write, width=110)
                dpg.add_button(label="Run nwbinspector", callback=self._cb_inspect)
                dpg.add_button(label="Close", callback=self._cb_close, width=80)
            dpg.add_progress_bar(tag=_PROGRESS, default_value=0.0, width=-1, overlay="")
            dpg.add_text("", tag=_STATUS)
            dpg.add_input_text(tag=_LOG, multiline=True, readonly=True, width=-1,
                               height=130, default_value="")

    # ---- render-loop drain (main thread) --------------------------------

    def _drain(self) -> None:
        import dearpygui.dearpygui as dpg

        with self._lock:
            lines, self._pending = self._pending, []
            prog, self._progress = self._progress, None
            st, self._status_pending = self._status_pending, None
        if lines and dpg.does_item_exist(_LOG):
            cur = dpg.get_value(_LOG)
            joined = "\n".join(lines)
            dpg.set_value(_LOG, (cur + "\n" + joined).strip() if cur else joined)
        if prog is not None and dpg.does_item_exist(_PROGRESS):
            done, total = prog
            dpg.configure_item(_PROGRESS, default_value=(done / total if total else 0.0),
                               overlay=f"{done}/{total}")
        if st is not None and dpg.does_item_exist(_STATUS):
            dpg.set_value(_STATUS, st)

    def _status(self, msg: str) -> None:
        with self._lock:
            self._status_pending = msg

    def _log(self, msg: str) -> None:
        with self._lock:
            self._pending.append(msg)

    # ---- callbacks -------------------------------------------------------

    def _cb_browse(self, *_a) -> None:
        from . import native_dialog

        picked = native_dialog.open_folder(
            title="Select a processed output folder (asi/...)", default_dir=str(self.out_dir))
        if picked:
            self._load_dir(picked)

    def _load_dir(self, folder: str | Path) -> None:
        import dearpygui.dearpygui as dpg

        folder = Path(folder)
        try:
            cfg = load_config(folder)
            cfg.output_dir = str(folder)
            self.config = cfg
        except Exception:  # noqa: BLE001 — a bare folder without db/ops: keep config, repoint
            self.config.output_dir = str(folder)
        self.out_dir = folder
        self._reload_meta()
        self.build()
        if dpg.is_dearpygui_running():
            dpg.set_primary_window(_WIN, True)
        self._status(f"loaded {folder}")
        self._drain()

    def _cb_reload(self, *_a) -> None:
        self._load_dir(self.out_dir)

    def _cb_open_folder(self, *_a) -> None:
        from . import native_dialog

        native_dialog.open_in_file_manager(self.out_dir)

    def _cb_load_meta(self, *_a) -> None:
        saved = nm.load_nwb_meta(self.out_dir)
        if saved is None:
            self._status("no nwb_metadata.yaml to load")
        else:
            self.meta = saved
            self.build()
            self._status("loaded nwb_metadata.yaml")
        self._drain()

    def _cb_save_meta(self, *_a) -> None:
        p = nm.save_nwb_meta(self.out_dir, self.collect_meta())
        self._status(f"saved {p.name}")
        self._drain()

    def _cb_validate(self, *_a) -> None:
        probs = nm.validate(self.collect_meta())
        dandi = [p for p in probs if p.startswith("[DANDI]")]
        self._log("--- validate ---")
        for p in probs:
            self._log("  " + p)
        self._status(
            "valid — no DANDI-critical problems" if not dandi
            else f"{len(dandi)} DANDI-critical problem(s) (see log)")
        self._drain()

    def _cb_write(self, *_a) -> None:
        if self._write_thread is not None and self._write_thread.is_alive():
            return
        meta = self.collect_meta()
        nm.save_nwb_meta(self.out_dir, meta)  # persist what we are about to write
        dandi = [p for p in nm.validate(meta) if p.startswith("[DANDI]")]
        for p in dandi:
            self._log("WARN " + p)
        self._status("writing NWB...")

        def _work():
            try:
                from asvimg.nwb_export import write_nwb

                path = write_nwb(self.config, self.out_dir, meta, reporter=self._reporter)
                self.last_written = path
                self._status(f"written: {path}")
            except Exception as exc:  # noqa: BLE001 — surface in the status/log, never crash
                self._log(f"ERROR: {exc}")
                self._status(f"write failed: {exc}")

        self._write_thread = threading.Thread(target=_work, daemon=True)
        self._write_thread.start()
        self._drain()

    def _cb_inspect(self, *_a) -> None:
        if self._inspect_thread is not None and self._inspect_thread.is_alive():
            return
        if not self.last_written or not Path(self.last_written).exists():
            self._status("write an NWB first")
            self._drain()
            return
        path = str(self.last_written)

        def _work():
            try:
                try:
                    import nwbinspector as ni
                    from nwbinspector import inspect_nwbfile
                except ModuleNotFoundError as exc:  # optional extra
                    raise ModuleNotFoundError(
                        'validating NWB needs the `nwb` extra: '
                        'uv pip install "asovi-imager[nwb]" (or pip)'
                    ) from exc

                try:
                    msgs = list(inspect_nwbfile(nwbfile_path=path, config=ni.load_config("dandi")))
                except Exception:  # noqa: BLE001
                    msgs = list(inspect_nwbfile(nwbfile_path=path))
                crit = [m for m in msgs if "CRITICAL" in str(m.importance)]
                self._log(f"[nwbinspector] {len(crit)} CRITICAL / {len(msgs)} messages")
                for m in crit:
                    self._log(f"  CRITICAL {m.check_function_name}: {m.message}")
                self._status(f"nwbinspector: {len(crit)} CRITICAL "
                             f"({'DANDI-clean' if not crit else 'blocks upload'})")
            except Exception as exc:  # noqa: BLE001
                self._status(f"nwbinspector failed: {exc}")

        self._inspect_thread = threading.Thread(target=_work, daemon=True)
        self._inspect_thread.start()
        self._drain()

    def _cb_close(self, *_a) -> None:
        self.should_close = True

    # ---- run -------------------------------------------------------------

    def run(self, *, title: str = "NWB Builder") -> Path | None:
        import dearpygui.dearpygui as dpg

        dpg.create_context()
        dpg.create_viewport(title=title, width=1180, height=920)
        self.build()
        dpg.set_primary_window(_WIN, True)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        while dpg.is_dearpygui_running() and not self.should_close:
            self._drain()
            dpg.render_dearpygui_frame()
        dpg.destroy_context()
        return self.last_written


def open_nwb_editor(config: PipelineConfig, output_dir: str | Path | None = None,
                    *, title: str = "NWB Builder") -> Path | None:
    return NwbEditor(config, output_dir).run(title=title)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def _resolve(arg: str | None):
    """(config, output_dir) from a CLI arg: an output/config dir, a YAML, or None."""
    if arg is None:
        return PipelineConfig(), None
    p = Path(arg)
    if p.is_dir():
        if (p / "ops.yaml").exists() or (p / "db.yaml").exists():
            cfg = load_config(p)
            cfg.output_dir = str(p)
            return cfg, p
        return PipelineConfig(output_dir=str(p)), p  # a bare processed folder
    cfg = load_config(p)  # a YAML file
    return cfg, (Path(cfg.output_dir) if cfg.output_dir else None)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    config, out = _resolve(argv[0] if argv else None)
    written = open_nwb_editor(config, out)
    print(f"[nwb-editor] wrote {written}" if written else "[nwb-editor] closed; nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
