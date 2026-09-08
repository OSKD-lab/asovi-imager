"""``PipelineSession`` — headless orchestrator for the full ASoVi pipeline.

Replays ``run_pipeline_full.ipynb`` as explicit, individually-callable stage
methods over the existing ``asvimg.*`` API.  Every stage reports start / done
/ failure, fine-grained progress, log lines, and matplotlib figures through a
:class:`~asvimg.events.ProgressReporter`, and cooperates with
a :class:`~asvimg.cancel.CancellationToken`.

Parity note — with ``ica_denoise="off"`` (the default) the ICA stage is QC only,
exactly as the MATLAB notebook is: it decomposes, records which components a human
called artifacts, and every *saved* artifact is still derived from the plain
linear-subtraction dF/F via ``_load_name_dff_stack``.  The IC exclusion does not
reach those files.

With ``ica_denoise="subtract"`` it does: ``roiSignals_*``, ``{exp}_dfWarped_*``,
the movies and (through the ROI signals) the correlations are computed from
``ica/dff_{name}.npy`` = dF/F minus the excluded components, and each payload
carries ``ica_denoised`` / ``ica_denoise_mode`` / ``ica_excluded_ic`` (1-based IC
numbers) so a file can say what it is.  Excluding nothing is the exact identity,
so the switch alone cannot change a value.

The annotated per-channel TIFFs are unaffected either way: they export the raw
registered channels (uint16), not dF/F, so there is nothing to denoise in them.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from asvimg import (
    ACCFv3,
    Cancelled,
    CancellationToken,
    apply_style,
    NullReporter,
    PcaResult,
    PipelineConfig,
    PreprocessStats,
    ProgressReporter,
    StageId,
    StageStatus,
    compute_ica_from_config,
    compute_pca_from_config,
    compute_roi_correlations,
    compute_transform,
    default_output_dir,
    extract_and_save_roi_signals,
    load_annotation_source,
    load_atlas_source_mean,
    load_marks,
    plot_correlation_matrix,
    plot_correlation_network,
    plot_pca_spatial_grid,
    plot_pca_temporal,
    plot_roi_signal_grid,
    plot_scree,
    reconstruct,
    reg_meta_path,
    resolve_exp_stem,
    save_annotated_dff_payloads,
    save_annotated_tiffs,
    save_ica_source_images,
    save_pca_source_images,
    save_warped_movies,
)
from asvimg.config import resolve_atlas_path
from asvimg.preprocess import PreprocessRunner

from .providers import (
    ConfigIcaProvider,
    ConfigMarksProvider,
    IcaExclusionProvider,
    MarksProvider,
    NoExclusionIcaProvider,
)

if TYPE_CHECKING:
    from skimage.transform import SimilarityTransform


def _fmt_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _fmt_time(ts: float) -> str:
    """``YYYY-MM-DD HH:MM:SS`` local time, or ``-`` when it is unusable.

    A preview must never be the thing that fails: archive extractors, tape/FTP
    restores and some SMB shares hand back pre-1970 or absurd timestamps, and
    ``fromtimestamp`` raises ``OSError`` on those (Windows especially). Those
    files still list — just without a time.
    """
    if not ts:
        return "-"
    import datetime as _dt

    try:
        return _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except (OSError, OverflowError, ValueError):
        return "?"


def _input_size_bytes(path: Path) -> int:
    """On-disk size of an input recording.

    A ``.sifx`` is only a small index — the pixels live in the sibling
    ``*spool.dat`` files, so report those instead.
    """
    try:
        if path.suffix.lower() == ".sifx":
            return sum(p.stat().st_size for p in path.parent.glob("*spool.dat"))
        return path.stat().st_size
    except OSError:
        return 0


def stage_artifacts(output_dir) -> "dict[StageId, Path | None]":
    """The newest on-disk artifact for each stage under ``output_dir`` (or None).

    Lets a front-end show, when an ``ops.yaml`` is loaded from a processed
    ``asi/`` folder, what has already run and *when* (via the file mtime).
    Correlation writes no persistent artifact, so it is always None.
    """
    out = Path(output_dir)
    parent = out  # auxiliary outputs (pca_images/ica_images/movies/tiffs) live under output_dir

    def newest(paths):
        ps = [p for p in paths if p.exists()]
        return max(ps, key=lambda p: p.stat().st_mtime) if ps else None

    def dir_newest(sub: str, pat: str):
        d = parent / sub
        return newest(list(d.glob(pat))) if d.is_dir() else None

    def fig_newest(pat: str):
        d = out / "figures"
        return newest(list(d.glob(pat))) if d.is_dir() else None

    marks = out / "marks.mat"
    export_cands = list(out.glob("*dfWarped_*"))
    for d in (parent / "tiffs", parent / "movies"):
        if d.is_dir():
            export_cands.append(d)

    # reg_Ch*.npy are PREallocated (mode="w+") at the top of the run, so their
    # existence says preprocess STARTED. reg_meta.npz is written at the end, and
    # every reader needs it anyway -- so it is what "preprocess finished" means.
    meta = reg_meta_path(out)

    return {
        StageId.PREPROCESS: meta if meta.exists() else None,
        # "**/" matches BOTH the per-group subdirs written now and the flat
        # layout older output folders have, so an old folder still reads as done.
        StageId.PCA: dir_newest("pca_images", "**/PC*.png"),
        StageId.ICA: dir_newest("ica_images", "**/IC*.png"),
        StageId.ANNOTATION: marks if marks.exists() else None,
        StageId.ROI: newest(list(out.glob("roiSignals_*"))),
        # correlation writes no data file; detect it from a saved figure if any.
        StageId.CORRELATION: fig_newest("correlation_*"),
        StageId.EXPORT: newest(export_cands),
    }


def completed_stages(output_dir) -> dict[StageId, bool]:
    """Which pipeline stages already have artifacts on disk (see stage_artifacts)."""
    return {k: v is not None for k, v in stage_artifacts(output_dir).items()}


class _StageCtl:
    """Handle yielded by :meth:`PipelineSession._stage`; lets a stage body mark
    itself SKIPPED (vs the default DONE) on a normal early return."""

    def __init__(self) -> None:
        self.skipped = False
        self.message = ""

    def skip(self, message: str = "") -> None:
        self.skipped = True
        self.message = message


@dataclass
class SessionState:
    """Mutable artifacts accumulated across stages."""

    output_dir: Path
    exp_stem: str
    atlas: "ACCFv3 | None" = None
    # PCA/ICA run per channels_name group. The bare fields below mirror the
    # ch_for_annotation group, which is the one cpselect offers maps from.
    pca_results: dict = field(default_factory=dict)
    ica_results: dict = field(default_factory=dict)
    excluded_by_group: dict = field(default_factory=dict)
    image_df: np.ndarray | None = None
    data_key: str | None = None
    ch_name: str | None = None
    pca_result: "PcaResult | None" = None
    pca_source_imgs: list[np.ndarray] | None = None
    pca_source_labels: list[str] | None = None
    ica_source_imgs: list[np.ndarray] | None = None
    ica_source_labels: list[str] | None = None
    source_img: np.ndarray | None = None
    denoised_df: np.ndarray | None = None
    excluded_ics: list[int] = field(default_factory=list)
    src_pts: np.ndarray | None = None
    ref_pts: np.ndarray | None = None
    tform: "SimilarityTransform | None" = None
    preprocess_stats: "PreprocessStats | None" = None
    roi_payloads: dict | None = None
    correlations: dict | None = None


class PipelineSession:
    """Drive the seven-stage pipeline headlessly.

    Stages run (``run_all``) in the order
    ``preprocess → pca → ica → annotation → roi → correlation → export``.
    Export runs last because it only needs the atlas transform, whereas
    correlation consumes the ROI signals written by the ROI stage.

    Parameters
    ----------
    config : PipelineConfig
    reporter : event sink (default: discards events).
    marks_provider : control-point source (default: resolve from config).
    ica_provider : IC-exclusion source (default: exclude nothing).
    cancel : cooperative cancellation token (optional).
    """

    def __init__(
        self,
        config: PipelineConfig,
        *,
        reporter: ProgressReporter | None = None,
        marks_provider: MarksProvider | None = None,
        ica_provider: IcaExclusionProvider | None = None,
        cancel: CancellationToken | None = None,
    ) -> None:
        self.config = config
        self.reporter: ProgressReporter = reporter or NullReporter()
        self.marks_provider: MarksProvider = marks_provider or ConfigMarksProvider()
        self._ica_provider_override = ica_provider
        self.cancel = cancel
        self._wants_figures = not isinstance(self.reporter, NullReporter)
        apply_style()  # consistent figure styling for every plot_* helper

        input_dir = Path(config.input_dir)
        output_dir = (
            Path(config.output_dir)
            if config.output_dir
            else default_output_dir(input_dir, config.output_format)
        )
        self.state = SessionState(
            output_dir=output_dir,
            exp_stem=resolve_exp_stem(
                config.input_dir, config.exp_name, config.input_format, config.input_order
            ),
        )
        # Headless default: resolve from config.ica_exclusion, which replays the
        # ica_exclusion.json a GUI session recorded ("cache"). It needs output_dir,
        # so it is built here rather than in the signature default.
        self.ica_provider: IcaExclusionProvider = (
            self._ica_provider_override or ConfigIcaProvider(config, output_dir)
        )

    # --- internals --------------------------------------------------------

    @contextmanager
    def _stage(self, stage: StageId):
        ctl = _StageCtl()
        t0 = time.perf_counter()
        try:
            # Cancel check INSIDE the try so a cancellation requested between
            # stages reports SKIPPED rather than leaving the stage event-less.
            if self.cancel is not None:
                self.cancel.raise_if_set()
            self.reporter.on_stage(stage, StageStatus.RUNNING)
            yield ctl
        except Cancelled:
            self.reporter.on_stage(
                stage, StageStatus.SKIPPED, message="cancelled",
                elapsed=time.perf_counter() - t0,
            )
            raise
        except Exception as exc:  # noqa: BLE001 — report then re-raise
            self.reporter.on_stage(
                stage, StageStatus.FAILED, message=str(exc),
                elapsed=time.perf_counter() - t0,
            )
            raise
        else:
            status = StageStatus.SKIPPED if ctl.skipped else StageStatus.DONE
            self.reporter.on_stage(
                stage, status, message=ctl.message,
                elapsed=time.perf_counter() - t0,
            )

    def _log(self, stage: StageId, msg: str) -> None:
        self.reporter.on_log(stage, msg)

    def _emit_figure(self, stage: StageId, key: str, factory) -> None:
        """Build + emit a figure when a consumer wants it and/or it is saved."""
        save = self.config.save_figures != "none"
        if not self._wants_figures and not save:
            return
        fig = factory()
        if save:
            self._save_figure(stage, key, fig)
        if self._wants_figures:
            self.reporter.on_figure(stage, key, fig)  # reporter closes it
        else:
            import matplotlib.pyplot as plt

            plt.close(fig)

    def _save_figure(self, stage: StageId, key: str, fig) -> None:
        """Write an emitted figure to ``<output>/figures/`` as PNG (+PDF)."""
        fig_dir = self.state.output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        stem = fig_dir / f"{stage.value}_{key}"
        try:
            fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
            if self.config.save_figures == "png+pdf":
                fig.savefig(f"{stem}.pdf", bbox_inches="tight")
            self._log(
                stage,
                f"[fig] saved {stem.name}.png"
                + (" + .pdf" if self.config.save_figures == "png+pdf" else ""),
            )
        except Exception as exc:  # noqa: BLE001 — a save failure must not abort the stage
            self._log(stage, f"[fig] save failed for {key}: {exc}")

    def _require_atlas(self) -> "ACCFv3":
        if self.state.atlas is None:
            self.state.atlas = ACCFv3.from_mat(
                resolve_atlas_path(self.config.annotation_atlas_path)
            )
        return self.state.atlas

    @staticmethod
    def _load_map_pngs(directory: Path, prefix: str) -> tuple[list, list]:
        """Load ``{prefix}{i}.png`` spatial maps from disk as RGB arrays."""
        if not directory.is_dir():
            return [], []
        import re

        def _idx(p: Path) -> int:
            m = re.search(r"(\d+)", p.stem)
            return int(m.group(1)) if m else 0

        # Maps live under pca_images/{group}/ (flat in older folders). Without
        # the group in the label, two groups' IC1 collide and cpselect shows one.
        files = sorted(directory.glob(f"**/{prefix}*.png"), key=_idx)
        if not files:
            return [], []
        from PIL import Image

        imgs = [np.asarray(Image.open(p).convert("RGB")) for p in files]
        labels = [
            p.stem if p.parent == directory else f"{p.parent.name}/{p.stem}"
            for p in files
        ]
        return imgs, labels

    def _annotation_extra_maps(self) -> tuple[list, list]:
        """PCA + ICA spatial maps to offer in cpselect (memory first, else disk).

        Never triggers a PCA/ICA computation: only already-available maps are
        returned, so annotation on data without PCA/ICA shows the raw frame only.
        """
        out_parent = self.state.output_dir
        imgs: list = []
        labels: list = []
        for mem_imgs, mem_labels, subdir, prefix in (
            (self.state.pca_source_imgs, self.state.pca_source_labels, "pca_images", "PC"),
            (self.state.ica_source_imgs, self.state.ica_source_labels, "ica_images", "IC"),
        ):
            if mem_imgs:
                imgs += list(mem_imgs)
                labels += list(mem_labels or [f"{prefix}{i + 1}" for i in range(len(mem_imgs))])
            else:
                disk_imgs, disk_labels = self._load_map_pngs(out_parent / subdir, prefix)
                imgs += disk_imgs
                labels += disk_labels
        return imgs, labels

    @staticmethod
    def _load_gray_map(path: Path) -> np.ndarray:
        """Load a ``.png`` / ``.npy`` / ``.tif(f)`` map as a 2-D float grayscale
        array.  RGB(A) -> luminance (alpha dropped), L+A -> drop alpha, a
        multi-page stack (colour or grayscale) is mean-projected, and a palette
        PNG is resolved to real colours first.  Raises if the result is not a
        non-empty 2-D array."""
        ext = path.suffix.lower()
        if ext == ".npy":
            arr = np.load(path)
        elif ext in (".tif", ".tiff"):
            import tifffile

            arr = tifffile.imread(path)
        else:  # .png / other PIL-readable
            from PIL import Image

            im = Image.open(path)
            if im.mode in ("P", "PA"):  # resolve palette indices to real colours
                im = im.convert("RGBA")
            arr = np.asarray(im)
        arr = np.squeeze(np.asarray(arr))
        # Colour stack (N, H, W, C): average frames, then collapse channels.
        if arr.ndim == 4 and arr.shape[-1] in (2, 3, 4):
            arr = arr.mean(axis=0)
        if arr.ndim == 3:
            c = arr.shape[-1]
            if c == 2:  # grayscale + alpha -> drop alpha
                arr = arr[..., 0]
            elif c in (3, 4):  # RGB(A) -> luminance (drop alpha)
                arr = arr[..., :3].astype(np.float64) @ [0.2989, 0.5870, 0.1140]
            else:  # page-first stack (N, H, W) -> mean projection
                arr = arr.mean(axis=0)
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim != 2 or arr.size == 0:
            raise ValueError(f"expected a non-empty 2-D image, got shape {arr.shape}")
        return arr

    def _load_custom_annotation_maps(self, source_img) -> tuple[list, list]:
        """User-supplied cpselect maps from ``<asi>/map_for_annot``.

        Every ``.png`` / ``.npy`` / ``.tif`` / ``.tiff`` there is loaded, converted
        to grayscale, and resized to the source frame so it shares cpselect's
        coordinate space.  A differing aspect ratio is warned (the resize then
        distorts the map, so picked points may not correspond spatially)."""
        map_dir = self.state.output_dir / "map_for_annot"
        try:
            map_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._log(StageId.ANNOTATION, f"[annot] could not create {map_dir}: {exc}")
            return [], []
        exts = (".png", ".npy", ".tif", ".tiff")
        files = sorted(
            p for p in map_dir.iterdir() if p.is_file() and p.suffix.lower() in exts
        )
        if not files:
            self._log(
                StageId.ANNOTATION,
                f"[annot] map_for_annot is empty ({map_dir}); drop .png / .npy / "
                ".tif files there to add custom annotation maps.",
            )
            return [], []
        import cv2

        src_h, src_w = source_img.shape[:2]
        src_ratio = (src_w / src_h) if src_h else 0.0
        imgs: list = []
        labels: list = []
        for p in files:
            # The whole per-file path (load, aspect check, resize) is guarded so
            # a single malformed/degenerate file is skipped, not fatal.
            try:
                gray = self._load_gray_map(p)
                h, w = gray.shape[:2]
                ratio = (w / h) if h else 0.0
                if src_ratio and abs(ratio - src_ratio) / src_ratio > 0.02:
                    self._log(
                        StageId.ANNOTATION,
                        f"[annot] WARNING {p.name}: aspect {w}x{h} differs from "
                        f"source {src_w}x{src_h}; resized to source, so picked "
                        "points may not correspond spatially.",
                    )
                resized = cv2.resize(
                    gray.astype(np.float32),
                    (src_w, src_h),
                    interpolation=cv2.INTER_AREA,
                )
            except Exception as exc:  # noqa: BLE001 — one bad file must not abort
                self._log(StageId.ANNOTATION, f"[annot] skipped {p.name}: {exc}")
                continue
            imgs.append(resized)
            labels.append(p.stem)
        if imgs:
            self._log(
                StageId.ANNOTATION,
                f"[annot] loaded {len(imgs)} custom map(s) from map_for_annot",
            )
        return imgs, labels

    def _ensure_tform(self, stage: StageId) -> "SimilarityTransform | None":
        """Return the atlas transform, rehydrating from disk if needed.

        ``run_annotation`` keeps the transform in ``state.tform`` only for the
        lifetime of one session.  In the GUI each stage button runs in a fresh
        session, so export / ROI would otherwise see ``None`` even though
        annotation already saved ``marks.mat``.  When in-memory state is empty
        we recompute the transform from the saved control points, exactly as
        annotation did — keeping every downstream stage independently runnable.
        """
        if self.state.tform is not None:
            return self.state.tform
        marks_path = self.state.output_dir / "marks.mat"
        if not marks_path.exists():
            return None
        src_pts, ref_pts = load_marks(marks_path)
        tform = compute_transform(
            src_pts, ref_pts,
            allow_reflection=self.config.annotation_allow_reflection,
        )
        self.state.src_pts = src_pts
        self.state.ref_pts = ref_pts
        self.state.tform = tform
        self._log(stage, f"[{stage.value}] loaded transform from {marks_path.name}")
        return tform

    # --- stages -----------------------------------------------------------

    def run_preprocess(self) -> PreprocessStats:
        """Stage 1 — registration + WFCI linear subtraction."""
        with self._stage(StageId.PREPROCESS) as ctl:
            runner = PreprocessRunner(self.config)
            stats = runner.run(reporter=self.reporter, cancel=self.cancel)
            self.state.output_dir = Path(stats.output_dir)
            self.state.preprocess_stats = stats
            if stats.cancelled:
                # Stop flushes what was read so far, so the outputs are a valid
                # PREFIX of the recording. Reporting DONE would let the GUI stamp
                # a fresh staleness signature over it and let every later stage
                # analyse a fraction of the data as if it were the whole thing.
                ctl.skip(f"cancelled after {stats.total_frames} frames")
            return stats

    def analysis_groups(self) -> list[str]:
        """The channels_name groups PCA/ICA run on — every group with a source.

        Each is a different fluorophore (or a different acquisition of one), so
        each gets its own decomposition: the components of GCaMP say nothing
        about the artifacts in jRGECO.
        """
        return [
            name for name, g in self.config.channel_groups().items()
            if g["source_indices"]
        ]

    def run_pca(self) -> "PcaResult":
        """Stage 2 — PCA on each group's dF/F.

        Returns the ``ch_for_annotation`` group's result (the one whose maps
        cpselect offers); every group's is in ``state.pca_results``.
        """
        with self._stage(StageId.PCA):
            from asvimg.pca import annotation_group_name, load_group_source

            annot = annotation_group_name(self.config)
            fps_channel = self.config.fps / self.config.cycle_len
            names = self.analysis_groups()

            for name in names:
                if self.cancel is not None:
                    self.cancel.raise_if_set()
                image_df, note = load_group_source(
                    self.config, self.state.output_dir, name
                )
                self._log(
                    StageId.PCA,
                    f"[pca] {name}: {note} shape={image_df.shape}"
                    + ("  (annotation source)" if name == annot else ""),
                )
                pca = compute_pca_from_config(image_df, self.config)
                self.state.pca_results[name] = pca

                # Figure keys and PNG dirs are namespaced per group — bare keys
                # would have every group overwrite the previous one's maps.
                self._emit_figure(
                    StageId.PCA, f"spatial_{name}",
                    lambda p=pca: plot_pca_spatial_grid(p),
                )
                self._emit_figure(
                    StageId.PCA, f"temporal_{name}",
                    lambda p=pca: plot_pca_temporal(p, fps_channel),
                )
                self._emit_figure(StageId.PCA, f"scree_{name}", lambda p=pca: plot_scree(p))

                pca_img_dir = self.state.output_dir / "pca_images" / name
                rgb_list, labels = save_pca_source_images(pca, pca_img_dir)
                if name == annot:
                    # cpselect offers the annotation group's maps as extra targets
                    self.state.image_df = image_df
                    self.state.data_key = "imageDf"
                    self.state.ch_name = name
                    self.state.pca_result = pca
                    self.state.pca_source_imgs = rgb_list
                    self.state.pca_source_labels = labels

            self.state.source_img = load_atlas_source_mean(
                self.config, self.state.output_dir
            )
            if self.state.pca_result is None:  # ch_for_annotation had no source
                raise RuntimeError(
                    f"ch_for_annotation points at {annot!r}, which has no source "
                    f"channel; PCA ran on {names}"
                )
            return self.state.pca_result

    def run_ica(self) -> list[int]:
        """Stage 3 — ICA per group: decompose, ask which components are artifacts.

        The decomposition goes to ``ica/{name}/basis.npz`` and the human decision
        to ``ica_exclusion.json``, so a later session (or a headless re-run) can
        replay both without re-opening the picker.

        Whether the exclusion reaches the saved outputs is a separate switch
        (``ica_denoise``); by itself this stage still changes nothing on disk
        except those two records.

        Returns the ``ch_for_annotation`` group's exclusion; every group's is in
        ``state.excluded_by_group``.
        """
        from asvimg.ica_state import save_exclusions, save_ica_basis
        from asvimg.pca import annotation_group_name

        if self.config.skip_ica:
            with self._stage(StageId.ICA) as st:
                self._log(StageId.ICA, "[ica] skipped (skip_ica).")
                st.skip("skip_ica")
                self.state.excluded_ics = []
                return []

        # The GUI runs each stage button in a fresh session, so PCA's in-memory
        # result is gone when ICA is clicked next.  PCA reads only on-disk
        # preprocess outputs, so re-run it here to satisfy the dependency
        # (keeps every stage independently runnable, like _ensure_tform).
        if not self.state.pca_results:
            self.run_pca()

        with self._stage(StageId.ICA):
            from asvimg.ica_state import load_exclusions
            from asvimg.pca import load_group_source

            annot = annotation_group_name(self.config)
            names = [n for n in self.analysis_groups() if n in self.state.pca_results]
            saved = load_exclusions(self.state.output_dir)
            decided: dict[str, list[int]] = dict(saved)  # keep groups we do not touch

            for i, name in enumerate(names):
                if self.cancel is not None:
                    self.cancel.raise_if_set()  # per group: Stop must not open N pickers

                pca = self.state.pca_results[name]
                ica_result = compute_ica_from_config(pca, self.config)
                self.state.ica_results[name] = ica_result
                save_ica_basis(self.state.output_dir, name, ica_result, self.config)
                self._log(
                    StageId.ICA,
                    f"[ica] {name}: {ica_result.spatial.shape[0]} components "
                    f"-> ica/{name}/basis.npz",
                )

                ica_img_dir = self.state.output_dir / "ica_images" / name
                ica_imgs, ica_labels = save_ica_source_images(ica_result, ica_img_dir)

                image_df = self.state.image_df if name == annot else None
                if image_df is None:
                    image_df, _ = load_group_source(
                        self.config, self.state.output_dir, name
                    )

                # Whether to ASK at all is the config's call, mirroring
                # needs_cpselect: "gui" always asks; "cache" asks only for a group
                # with nothing recorded (else it replays); an inline dict / False
                # never asks. Without this the GUI re-opened a picker for every
                # group on every run, however the config was set.
                mode = self.config.ica_exclusion
                if mode == "gui" or (mode == "cache" and name not in saved):
                    picked = self.ica_provider.request(
                        name, ica_result, image_df,
                        index=i, total=len(names),
                        preselected=list(saved.get(name, [])),
                    )
                else:
                    from asvimg.ica_state import resolve_exclusion

                    picked = (
                        resolve_exclusion(self.config, self.state.output_dir, name)
                        if mode not in (False, None) and not isinstance(mode, dict)
                        else (
                            [int(x) for x in mode[name]]
                            if isinstance(mode, dict) and name in mode
                            else None  # no answer for this group; keep the record
                        )
                    )

                if picked is None:
                    # Cancelled / the picker died. NOT the same as "exclude
                    # nothing": keep whatever was recorded and move on, or a
                    # crashed window would silently mark the group as reviewed.
                    self._log(
                        StageId.ICA,
                        f"[ica] {name}: selection cancelled — keeping "
                        f"{saved.get(name, [])}",
                    )
                else:
                    decided[name] = [int(x) for x in picked]
                    self._log(
                        StageId.ICA,
                        f"[ica] {name}: excluded ICs {decided[name]}"
                        if decided[name] else f"[ica] {name}: no components excluded",
                    )
                # Persist after EVERY group: a Stop during group 2 of 3 must not
                # throw away the choice already made for group 1.
                save_exclusions(self.state.output_dir, decided)
                if self.cancel is not None:
                    # ...and a Stop while the LAST picker was open must not let the
                    # stage report DONE.
                    self.cancel.raise_if_set()

                excluded = decided.get(name, [])
                if name == annot:
                    self.state.ica_source_imgs = ica_imgs
                    self.state.ica_source_labels = ica_labels
                    self.state.excluded_ics = list(excluded)
                    # QC picture of what excluding these would do (rank-k, on the
                    # fitted frames) — NOT what apply_ica_denoise writes.
                    self.state.denoised_df = (
                        reconstruct(ica_result, image_df, exclude=excluded)
                        if excluded else image_df
                    )
                    if excluded:
                        self._emit_figure(
                            StageId.ICA, f"denoise_diff_{name}",
                            lambda o=image_df, d=self.state.denoised_df,
                            e=list(excluded): self._ica_diff_figure(o, d, e),
                        )

            self.state.excluded_by_group = decided
            save_exclusions(self.state.output_dir, decided)
            return list(self.state.excluded_ics)

    @staticmethod
    def _ica_diff_figure(original, denoised, excluded):
        import matplotlib.pyplot as plt

        idx = original.shape[2] // 2
        vmin, vmax = np.percentile(original[:, :, idx], [2, 98])
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(original[:, :, idx], cmap="magma", vmin=vmin, vmax=vmax)
        axes[0].set_title("Original dF/F")
        axes[0].axis("off")
        axes[1].imshow(denoised[:, :, idx], cmap="magma", vmin=vmin, vmax=vmax)
        axes[1].set_title(f"Denoised (excluded: {excluded})")
        axes[1].axis("off")
        fig.suptitle("ICA Denoising: Before vs After", fontsize=14)
        fig.tight_layout()
        return fig

    @staticmethod
    def _describe_tform(tform, n_pairs: int) -> str:
        """One-line summary of an atlas transform, flagging any reflection."""
        m = np.asarray(tform.params)[:2, :2]
        det = float(np.linalg.det(m))
        scale = float(np.sqrt(abs(det)))
        rotation = float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))
        flip = "  [REFLECTED - left-right flipped]" if det < 0 else ""
        return (
            f"[annot] scale={scale:.4f}, rotation={rotation:.2f} deg, "
            f"{n_pairs} pairs{flip}"
        )

    @staticmethod
    def _draw_atlas_borders_in_source(ax, atlas, tform, *, color="red", lw=0.5):
        """Draw the atlas region borders warped into the source image's own
        coordinates (via the inverse transform)."""
        minv = np.linalg.inv(np.asarray(tform.params, dtype=float))
        scale = atlas.scale
        for group in atlas.boundaries[1:]:  # skip index 0 = whole-brain outline
            for contour in group:
                c = np.asarray(contour, dtype=float)
                x = c[:, 1] * scale  # atlas 285-space (col, row)
                y = c[:, 0] * scale
                pts = np.column_stack([x, y, np.ones_like(x)]) @ minv.T
                ax.plot(pts[:, 0], pts[:, 1], "-", color=color, linewidth=lw)

    def _plot_transform_overlay(self, source_img, tform, atlas):
        """4-panel visual check of the annotation transform:
        1 source (pre-transform), 2 warped to atlas, 3 warped + atlas border,
        4 source + atlas border warped back into source space."""
        import matplotlib.pyplot as plt

        from asvimg import warp_image

        src = np.asarray(source_img, dtype=np.float64)
        warped = warp_image(src, tform, atlas.shape_hw)

        def _vmm(img):
            pos = img[img > 0]
            return np.percentile(pos, [2, 98]) if pos.size else (0.0, 1.0)

        fig, axes = plt.subplots(2, 2, figsize=(7.3, 7.3))  # 2/3 of the former 11x11
        a1, a2, a3, a4 = axes.ravel()

        vmin, vmax = _vmm(src)
        a1.imshow(src, cmap="gray", vmin=vmin, vmax=vmax)
        a1.set_title("1: source (pre-transform)", fontsize=11)

        wvmin, wvmax = _vmm(warped)
        a2.imshow(warped, cmap="gray", vmin=wvmin, vmax=wvmax)
        a2.set_title("2: transformed -> atlas", fontsize=11)

        a3.imshow(warped, cmap="gray", vmin=wvmin, vmax=wvmax)
        atlas.draw_boundaries(a3, color="red", linewidth=0.5)
        a3.set_xlim(0, atlas.shape_hw[1])
        a3.set_ylim(atlas.shape_hw[0], 0)
        a3.set_title("3: transformed + atlas border", fontsize=11)

        a4.imshow(src, cmap="gray", vmin=vmin, vmax=vmax)
        self._draw_atlas_borders_in_source(a4, atlas, tform)
        a4.set_xlim(0, src.shape[1])
        a4.set_ylim(src.shape[0], 0)
        a4.set_title("4: source + transformed border", fontsize=11)

        for ax in (a1, a2, a3, a4):
            ax.axis("off")
        det = float(np.linalg.det(np.asarray(tform.params)[:2, :2]))
        tag = "REFLECTED (LR-flipped)" if det < 0 else "non-reflective"
        fig.suptitle(f"Annotation transform  ({tag})", fontsize=13)
        fig.tight_layout()
        return fig

    def preview_input(self, n_frames: int = 12, ncols: int = 4):
        """Quick QC: render the first ``n_frames`` raw input frames in a grid,
        each labelled with its cycling channel name — the fastest way to see
        which channel is which (e.g. G vs R) before committing to a run."""
        from asvimg import find_input_files, get_frame_count, iter_frames

        stage = StageId.PREPROCESS
        try:
            files = find_input_files(
                Path(self.config.input_dir),
                self.config.input_format,
                self.config.input_order,
            )
        except Exception as exc:  # noqa: BLE001 — surface to the log pane
            self._log(stage, f"[preview] no input files: {exc}")
            return

        self._log_input_files(stage, files)
        # Only the first file is read, so its frame count is all the correction
        # needs (later files' slip edits start beyond every frame shown here).
        corr = self._demux_correction(stage, [get_frame_count(files[0])])

        frames: list[np.ndarray] = []
        labels: list[str] = []
        gen = iter_frames(files[0])
        try:
            for i in range(n_frames):
                frames.append(np.asarray(next(gen)))
                labels.append(self._channel_label(corr, i))
        except StopIteration:
            pass
        finally:
            close = getattr(gen, "close", None)
            if close:
                close()

        if not frames:
            self._log(stage, "[preview] input produced no frames.")
            return
        self._log(stage, f"[preview] {files[0].name}: first {len(frames)} frames")
        self._emit_figure(
            stage, "input_preview",
            lambda: self._plot_frame_grid(frames, labels, ncols),
        )

    def preview_input_all(self, n_frames: int = 4):
        """Quick QC across files: the first ``n_frames`` frames of *every* input
        file, one row per file.

        Where :meth:`preview_input` only shows the head of the first file, this
        shows the head of each of them — the view that makes a per-file phase
        slip visible (a file whose cycle starts on the wrong channel), and that
        confirms a ``channels_slip`` entry actually fixes it.  Labels use the
        channel preprocess will assign (demux offset + slips), not the bare
        positional cycle.
        """
        from asvimg import find_input_files, get_frame_count, iter_frames

        stage = StageId.PREPROCESS
        try:
            files = find_input_files(
                Path(self.config.input_dir),
                self.config.input_format,
                self.config.input_order,
            )
        except Exception as exc:  # noqa: BLE001 — surface to the log pane
            self._log(stage, f"[preview] no input files: {exc}")
            return

        self._log_input_files(stage, files)
        frame_counts = [get_frame_count(f) for f in files]
        corr = self._demux_correction(stage, frame_counts)

        from asvimg import input_file_times

        order = self.config.input_order
        rows: list[tuple[str, list[tuple[np.ndarray, str]]]] = []
        start = 0  # global index of this file's first frame
        for i, (path, count) in enumerate(zip(files, frame_counts)):
            items: list[tuple[np.ndarray, str]] = []
            gen = iter_frames(path)
            try:
                for j in range(n_frames):
                    img = np.asarray(next(gen))
                    items.append((img, self._channel_label(corr, start + j)))
            except StopIteration:
                pass
            finally:
                close = getattr(gen, "close", None)
                if close:
                    close()
            if items:
                # The row label carries the sort key itself, so the figure alone
                # answers "is this the order I meant?" without the log pane.
                ctime, mtime = input_file_times(path)
                stamp = {"ctime": f"created {_fmt_time(ctime)}",
                         "mtime": f"modified {_fmt_time(mtime)}"}.get(order, "")
                label = f"#{i + 1}  {path.name}\n({count} frames)"
                rows.append((f"{label}\n{stamp}" if stamp else label, items))
            start += count

        if not rows:
            self._log(stage, "[preview] input produced no frames.")
            return
        self._log(
            stage,
            f"[preview] first {n_frames} frame(s) of each of {len(rows)} file(s), "
            f"in read order (input_order={order!r})",
        )
        self._emit_figure(
            stage, "input_preview_all",
            lambda: self._plot_file_head_grid(rows, n_frames, order),
        )

    def qc_frame_slip(self, n_frames: int = 60, downsample: int = 8, method: str = "pca"):
        """QC across files by image CONTENT: embed the first ``n_frames`` frames
        of every input file in 2D and check each file's frames land in the
        channel cluster its demux assignment predicts.

        Where :meth:`preview_input_all` shows the head frames as thumbnails for
        the eye, this reduces them (spatial decimate + content-normalise) and
        clusters them, so it catches a phase slip the *intensity* demux QC misses
        when the two channels have near-equal brightness but different content
        (FT-053: source/donner means differ <0.2%, vasculature distinct). See
        :mod:`asvimg.qc_frameslip`.
        """
        from asvimg import find_input_files, get_frame_count, iter_frames
        from asvimg.qc_frameslip import (
            embed_2d, per_file_consistency, plot_slip_scatter,
        )

        stage = StageId.PREPROCESS
        try:
            files = find_input_files(
                Path(self.config.input_dir),
                self.config.input_format,
                self.config.input_order,
            )
        except Exception as exc:  # noqa: BLE001 — surface to the log pane
            self._log(stage, f"[qc-slip] no input files: {exc}")
            return

        self._log_input_files(stage, files)
        frame_counts = [get_frame_count(f) for f in files]
        corr = self._demux_correction(stage, frame_counts)
        props = self.config.channels_prop or []

        def _clabel(g: int) -> str:
            c = corr.channel_at(g)
            return props[c] if c < len(props) else f"Ch{c}"

        self._log(
            stage,
            f"[qc-slip] reading first {n_frames} frame(s) of {len(files)} file(s) "
            f"(decimate x{downsample}); this streams from disk, can take a minute...",
        )
        imgs: list[np.ndarray] = []
        chans: list[str] = []
        fidx: list[int] = []
        start = 0
        for k, (path, count) in enumerate(zip(files, frame_counts)):
            gen = iter_frames(path)
            try:
                for j in range(n_frames):
                    img = np.asarray(next(gen), dtype=np.float32)[::downsample, ::downsample]
                    imgs.append(img)
                    chans.append(_clabel(start + j))
                    fidx.append(k)
            except StopIteration:
                pass
            finally:
                close = getattr(gen, "close", None)
                if close:
                    close()
            start += count

        if len(imgs) < 2 * max(1, len(set(chans))):
            self._log(stage, "[qc-slip] too few frames read to QC.")
            return
        # crop all to the common (min) geometry, then flatten (guards against a
        # file with a different frame size).
        h = min(im.shape[0] for im in imgs)
        w = min(im.shape[1] for im in imgs)
        X = np.stack([im[:h, :w].ravel() for im in imgs])

        emb, used = embed_2d(X, method)
        lines, flagged = per_file_consistency(emb, chans, fidx)
        self._log(stage, f"[qc-slip] embedding={used}, {len(X)} frames, {len(files)} files")
        for ln in lines:
            self._log(stage, f"[qc-slip]   {ln}")
        if flagged:
            bad = ", ".join(f"#{f + 1} {files[f].name}" for f in sorted(flagged))
            self._log(stage, f"[qc-slip] POSSIBLE SLIP in file(s): {bad}")
        else:
            self._log(stage, "[qc-slip] all files channel-consistent (no slip found).")

        fnames = [f.name for f in files]
        self._emit_figure(
            stage, "qc_frame_slip",
            lambda: plot_slip_scatter(emb, chans, fidx, fnames, used, flagged),
        )

    def _demux_correction(self, stage: StageId, frame_counts: list[int]):
        """The demux correction preprocess would apply, logged when non-identity."""
        from asvimg import build_demux_correction

        corr = build_demux_correction(self.config, self.state.output_dir, frame_counts)
        if not corr.is_identity():
            self._log(
                stage,
                f"[preview] demux correction: start_offset={corr.start_offset}, "
                f"edits={corr.edits}",
            )
        return corr

    def _channel_label(self, corr, g: int) -> str:
        """``frame {g} — Ch{c} {name}-{prop}`` for global frame index ``g``."""
        names = self.config.channels_name or []
        props = self.config.channels_prop or []
        c = corr.channel_at(g)
        ch = names[c] if c < len(names) else "?"
        prop = props[c] if c < len(props) else "?"
        return f"frame {g} — Ch{c} {ch}-{prop}"

    def _log_input_files(self, stage: StageId, files: list[Path]) -> None:
        """List the input files in the order preprocessing will read them.

        Both timestamps are printed for every file, not just the one being
        sorted on, because that is what lets a wrong ``input_order`` be caught
        here instead of after a multi-hour run: data copied from a backup or a
        network share carries a *new* creation time, so a ctime sort can be
        badly shuffled while the mtime column is still cleanly monotonic (or
        vice versa).  Read the column matching the arrow and check it ascends.
        """
        from asvimg import input_file_times

        order = self.config.input_order
        total = 0
        self._log(
            stage,
            f"[preview] input files ({len(files)}), in read order "
            f"(input_order={order!r}):",
        )
        # Mark the column the order actually sorted on. "natural"/"name" sort on
        # the filename, so the marker goes on the name column for those — the
        # header always points at the real sort key, whatever the order is.
        c_key, m_key = ("<-" if order == "ctime" else "  "), ("<-" if order == "mtime" else "  ")
        n_key = " <-" if order in ("natural", "name") else ""
        self._log(
            stage,
            f"[preview]        {'created':19s} {c_key} {'modified':19s} {m_key} "
            f"{'size':>9s}  name{n_key}",
        )
        for i, path in enumerate(files):
            size = _input_size_bytes(path)
            total += size
            ctime, mtime = input_file_times(path)
            self._log(
                stage,
                f"[preview]   {i + 1:>3}. {_fmt_time(ctime):19s} {c_key} "
                f"{_fmt_time(mtime):19s} {m_key} {_fmt_bytes(size):>9s}  {path.name}",
            )
        self._log(stage, f"[preview] total {_fmt_bytes(total)}")
        if order == "ctime":
            self._log(
                stage,
                "[preview] note: 'created' is the filesystem creation time — on data "
                "copied or restored from a backup it is the COPY time, not the "
                "acquisition time. If the created column above is not in acquisition "
                "order, use input_order='mtime'.",
            )

    @staticmethod
    def _plot_frame_grid(frames, labels, ncols):
        import matplotlib.pyplot as plt

        n = len(frames)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax, img, lab in zip(axes, frames, labels):
            vmin, vmax = np.percentile(img, [2, 98])
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(lab, fontsize=9)
            ax.axis("off")
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle("Input preview — first frames by channel", fontsize=13)
        fig.tight_layout()
        return fig

    @staticmethod
    def _plot_file_head_grid(rows, ncols, order: str = "natural"):
        """One row per input file: its first ``ncols`` frames, channel-labelled.

        Rows appear top-to-bottom in the read order, so the figure doubles as
        the order check: ``order`` is named in the title and each row is
        numbered, making a mis-ordered concatenation visible at a glance.

        Frames keep a (tick-less) axes frame rather than ``axis("off")`` so the
        y-label can name the file the row came from.
        """
        import matplotlib.pyplot as plt

        nrows = len(rows)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3 * ncols, 3.3 * nrows), squeeze=False
        )
        for r, (file_label, items) in enumerate(rows):
            for c in range(ncols):
                ax = axes[r][c]
                ax.set_xticks([])
                ax.set_yticks([])
                if c >= len(items):
                    ax.axis("off")
                    continue
                img, label = items[c]
                vmin, vmax = np.percentile(img, [2, 98])
                ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
                ax.set_title(label, fontsize=9)
                if c == 0:
                    ax.set_ylabel(file_label, fontsize=8)
        fig.suptitle(
            "Input preview — first frames of every file\n"
            f"rows are the read order (input_order={order!r}); "
            "frames are concatenated top to bottom",
            fontsize=12,
        )
        fig.tight_layout()
        return fig

    def run_annotation(self) -> "SimilarityTransform | None":
        """Stage 4 — atlas control points → similarity transform."""
        with self._stage(StageId.ANNOTATION) as st:
            atlas = self._require_atlas()
            source_img = self.state.source_img
            if source_img is None:
                source_img = load_atlas_source_mean(self.config, self.state.output_dir)
                self.state.source_img = source_img

            # cpselect flips through the raw frame plus whatever PCA / ICA maps
            # already exist (in memory this session, or saved on disk by a prior
            # run).  PCA/ICA are NOT run just for annotation — if neither exists
            # the user picks control points on the raw frame alone.
            extra_imgs, extra_labels = self._annotation_extra_maps()
            # Custom maps (map_for_annot) only matter when cpselect actually
            # opens; loading them also creates the drop folder on first use.
            from asvimg import needs_cpselect

            if needs_cpselect(self.config, self.state.output_dir):
                custom_imgs, custom_labels = self._load_custom_annotation_maps(source_img)
                extra_imgs = list(extra_imgs) + custom_imgs
                extra_labels = list(extra_labels) + custom_labels
            if extra_imgs:
                self._log(
                    StageId.ANNOTATION,
                    f"[annot] cpselect: raw frame + {len(extra_imgs)} extra map(s)",
                )

            src_pts, ref_pts = self.marks_provider.request(
                self.config,
                self.state.output_dir,
                atlas,
                source_img,
                extra_imgs=extra_imgs or None,
                extra_labels=extra_labels or None,
            )
            self.state.src_pts = src_pts
            self.state.ref_pts = ref_pts

            if src_pts is None or ref_pts is None:
                self.state.tform = None
                self._log(
                    StageId.ANNOTATION,
                    "[annot] no marks - warp/export/ROI will be skipped.",
                )
                st.skip("no marks")
                return None

            tform = compute_transform(
                src_pts, ref_pts,
                allow_reflection=self.config.annotation_allow_reflection,
            )
            self.state.tform = tform
            self._log(StageId.ANNOTATION, self._describe_tform(tform, len(src_pts)))
            self._emit_figure(
                StageId.ANNOTATION, "transform_overlay",
                lambda: self._plot_transform_overlay(source_img, tform, atlas),
            )
            return tform

    def run_export(self) -> None:
        """Stage 7 — annotated TIFFs, warped dF/F payloads, movies."""
        with self._stage(StageId.EXPORT) as st:
            tform = self._ensure_tform(StageId.EXPORT)
            if tform is None:
                self._log(StageId.EXPORT, "[export] no transform - skipped.")
                st.skip("no transform")
                return
            atlas = self._require_atlas()
            out = self.state.output_dir
            kw = dict(reporter=self.reporter, cancel=self.cancel, stage=StageId.EXPORT)
            save_annotated_tiffs(self.config, out, tform, atlas, **kw)
            save_annotated_dff_payloads(self.config, out, tform, atlas, **kw)
            save_warped_movies(self.config, out, tform, atlas, **kw)

    def run_roi(self) -> dict:
        """Stage 5 — ROI signal extraction."""
        with self._stage(StageId.ROI) as st:
            # Source-space ROIs need no atlas registration: the ROIs are already
            # in the recording's own coords, so ROI signals can be read even when
            # annotation was skipped (no transform).
            source_mode = self.config.roi_space == "source"
            if source_mode:
                tform, atlas = None, None
            else:
                tform = self._ensure_tform(StageId.ROI)
                if tform is None:
                    self._log(StageId.ROI, "[roi] no transform - skipped.")
                    st.skip("no transform")
                    return {}
                atlas = self._require_atlas()
            payloads = extract_and_save_roi_signals(
                self.config,
                self.state.output_dir,
                tform,
                atlas,
                reporter=self.reporter,
                cancel=self.cancel,
            )
            self.state.roi_payloads = payloads

            fps_channel = self.config.fps / self.config.cycle_len
            for name, payload in payloads.items():
                F = payload.get("F_dff", payload.get("F_raw"))
                n_t = int(np.asarray(F).shape[1]) if F is not None else 0
                window = self.config.roi_plot_window(n_t) if n_t else None
                self._emit_figure(
                    StageId.ROI, f"signals_{name}",
                    lambda p=payload, n=name, w=window: plot_roi_signal_grid(
                        p, n, fps_channel, window=w
                    ),
                )
            return payloads

    def run_correlation(self) -> dict:
        """Stage 6 — ROI correlation network + matrix."""
        with self._stage(StageId.CORRELATION) as st:
            corrs = compute_roi_correlations(self.config, self.state.output_dir)
            self.state.correlations = corrs
            if not corrs:
                self._log(StageId.CORRELATION, "[corr] no roiSignals files - skipped.")
                st.skip("no roiSignals")
                return {}
            net_threshold = (
                "auto" if self.config.corr_auto_threshold
                else self.config.corr_network_threshold
            )
            # node positions: source-space ROIs come from rois_source.csv and the
            # network is drawn on a plain background (no atlas); atlas-space ROIs
            # come from rois.csv (edited) or the atlas defaults, on the atlas.
            if self.config.roi_space == "source":
                from asvimg import resolve_source_rois

                atlas = None
                rois = resolve_source_rois(self.state.output_dir)
            else:
                from asvimg import resolve_rois

                atlas = self._require_atlas()
                rois, _ = resolve_rois(self.state.output_dir, atlas)
            roi_positions = {r.name: (r.y, r.x) for r in rois}
            for name, corr in corrs.items():
                self._emit_figure(
                    StageId.CORRELATION, f"network_{name}",
                    lambda c=corr: plot_correlation_network(
                        c, atlas,
                        threshold=net_threshold,
                        edge_density=self.config.corr_edge_density,
                        positions=roi_positions,
                    ),
                )
                self._emit_figure(
                    StageId.CORRELATION, f"matrix_{name}",
                    lambda c=corr: plot_correlation_matrix(c),
                )
            return corrs

    # --- orchestration ----------------------------------------------------

    def run_all(self) -> SessionState:
        """Run all stages in order.

        ROI / correlation / export all need the atlas transform, so they are
        skipped when annotation produced no transform.  Export (saving warped
        TIFFs / dF/F / movies) runs last, after the ROI analysis.
        """
        self.run_preprocess()
        self.run_pca()
        self.run_ica()
        self.run_annotation()
        if self.state.tform is not None:
            self.run_roi()
            self.run_correlation()
            self.run_export()
        else:
            self._log(
                StageId.ANNOTATION,
                "[run_all] no transform - ROI/correlation/export skipped.",
            )
        return self.state
