"""Pipeline configuration and helpers."""

from __future__ import annotations

import re
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

# Read (concatenation) order for a multi-file recording. Owned by io (the module
# that applies it) and re-used here so the validation cannot drift from it.
from .io import _INPUT_ORDERS

_DEFAULT_CHANNELS_NAME = ["BL", "BL"]
_DEFAULT_CHANNELS_PROP = ["source", "donner"]

_CHANNEL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


_DB_FIELDS: frozenset[str] = frozenset({
    "input_dir",
    "input_format",
    "input_order",
    "output_dir",
    "output_format",
    "exp_name",
    "output_metadata_yaml",
    "dcimg_backend",
})

# Input file formats find_input_files can select. "auto" detects the single
# format present and errors when a folder mixes formats; the rest force one.
_INPUT_FORMATS: tuple[str, ...] = ("auto", "tif", "dcimg", "sifx", "nd2", "h5")

# Storage dtype for the atlas-warped dF/F (dfWarped) payload.
_ANNOTATED_DF_DTYPES: tuple[str, ...] = ("float32", "float16")


@dataclass
class PipelineConfig:
    input_dir: str = "Analysis/_sampleData01"
    input_format: str = "auto"  # which input format to read: "auto" (detect; error if a folder mixes formats) | "tif" (.tif/.tiff) | "dcimg" | "sifx" | "nd2" (Nikon; T x C) | "h5" (Ito even/odd folder)
    input_order: str = "natural"  # read (concatenation) order of a multi-file recording: "natural" (natsort: rec_2 before rec_10; default) | "name" (plain lexicographic: rec_10 before rec_2) | "mtime" (oldest modification time first = acquisition order) | "ctime" (oldest creation time first — on copied/restored data this is the COPY order, not acquisition; verify in Quick Preview (All), which prints both timestamps). Also decides exp_name/exp_stem inference (first file) and the order channels_slip entries refer to
    output_dir: str | None = None
    channels_name: list[str] | None = None  # e.g. ["GCaMP", "GCaMP"] or ["GCaMP", "jRGECO", "GCaMP", "jRGECO"]
    channels_prop: list[str] | None = None  # e.g. ["source", "donner"] or ["donner", "source", "source", "source"]
    channels_slip: list[int] | None = None  # per-input-file demux phase offset (one entry per input file, read order); e.g. [0, 1, 0] = the 2nd file's channel cycle starts 1 frame into the cycle. None/[] → every file follows the positional demux
    do_registration: bool = True
    registration_cache: str = "force"  # "force": always (re)run preprocess; "cached": skip registration when a COMPLETE previous run's reg_Ch*.npy are in output_dir (a cancelled/crashed run's partial memmaps are detected via reg_meta.npz and re-run)
    linear_subt: bool = True
    baseline_percentile: float = 5.0  # per-pixel percentile used for dF/F baseline (0-100)
    detrend: bool = False  # exponential (photobleaching) detrend of src & donner before the WFCI regression (MATLAB flag_ExpoSub; log-linear a*exp(b*t) fit, subtracted per pixel)
    baseline_percentile_highpass: bool = False  # rolling-percentile ratiometric high-pass of src & donner before the WFCI regression (David Whitney baselinePercentileFilter; MATLAB flag_BaselineFilter). Independent of detrend; when both on, detrend runs first
    baseline_percentile_highpass_sec: float = 120.0  # rolling-percentile window (seconds) for the high-pass baseline
    baseline_percentile_highpass_rank: float = 50.0  # percentile rank (0-100) for the rolling high-pass baseline (David Whitney default 50 = median)
    hemovar_qc: bool = True  # per-group hemo variance-explained (R^2) QC map: fraction of source variance the donner regression explains; saves hemovar_{name}.npy + a figure (WidefieldImager hemoVar analogue)
    start_initial_frames: int | None = None  # initial frames skipped from regression FOI, baseline percentile & ROI plots; None → 4 * (fps / cycle_len)
    ignore_last_frames: int = 0  # trailing frames excluded from regression FOI & ROI plots (0 → keep all)
    binning: int = 2
    flip: bool = False
    delete: bool = True
    fps: int = 20
    use_mmap: bool = False  # dF/F read strategy (output identical either way). False → load each registered channel fully into RAM (fastest; short recordings / big machines). True → read reg_Ch via memmap and stream dF/F in row-strips (bounded peak memory for long recordings).
    registration_batch_size: int = 500  # frames per registration batch (read → register_batch → bin → write); bounds registration read memory. (formerly batch_size)
    usfac: int = 50  # DFT registration subpixel precision (1/usfac px). ~free vs usfac=10 after vectorization, ~5x better alignment
    template: str | None = None  # path to an existing template image (.mat/.npy/.tif/.png/...); None → auto-create via template_stride
    template_stride: int = 100
    template_ch: int = 0  # cycle channel index used when auto-building the registration template
    output_format: str = "mat"
    exp_name: str | None = None
    max_frames: int | None = None
    output_metadata_yaml: bool = True
    dcimg_backend: str = "auto"
    filter_xyt: list[int] | None = None  # 3D filter window [x, y, t], e.g. [3, 3, 3]
    filter_xyt_kind: str = "mean"  # filter_xyt kind: "mean" | "median" | "gaussian"
    demux_qc: bool = True  # intensity-based demux phase QC: check the positional channel cycle against per-frame mean intensity and flag phase slips (detection only; never changes demux)
    demux_start_offset: int = 0  # global demux phase rotation (0..cycle_len-1); a demux_correction.json (from the demux editor) overrides this when present
    annotation: object = "cache"  # "cache" | "gui" | ([src], [ref]) | False
    annotation_allow_reflection: bool = False  # allow a mirror (left-right flip) in the atlas transform; needs non-midline control points to take effect
    ch_for_annotation: int = 0  # channel index (0..cycle_len-1) used as source for annotation/PCA
    roi_signal: str = "Both"  # "Both" | "Raw" | "dff" — which signal(s) to extract per ROI
    roi_space: str = "atlas"  # "atlas": ROIs are Allen-atlas coords, pulled into source space by the annotation transform (needs annotation done) | "source": ROIs are already in the recording's own (binned reg/dff) coords, read directly — no atlas registration needed. Source ROIs come from rois_source.csv (there are no built-in defaults for a raw recording)
    save_roi_signals: str | None = "csv"  # None/False | "csv" | "pickle" | "both" — export ROI time-series as DataFrame (rows=time, cols=ROI)
    corr_method: str = "raw"  # "raw" | "gsr" | "partial" — ROI correlation: plain Pearson, global-signal-regressed, or partial (shrinkage precision)
    corr_auto_threshold: bool = True  # network edges: auto (proportional/density) threshold vs the fixed corr_network_threshold
    corr_edge_density: float = 0.2  # when auto: fraction of strongest edges to keep (0.2 = top 20%)
    corr_network_threshold: float = 0.3  # fixed |r| edge cutoff used when corr_auto_threshold is False
    save_movie: bool = False
    save_raw_each_ch: bool = False  # save raw (pre-processing, input dtype) channels as TIFF
    save_registered_each_ch: bool = False  # save registered+binned uint16 channels as TIFF
    save_annotated_each_ch: bool = False  # save atlas-warped channels as TIFF
    save_annotated_dF_mat: bool = False  # save atlas-warped dF/F per channels_name as mat/npy/h5 (output_format)
    save_annotated_dF_dtype: str = "float32"  # dtype of the dfWarped payload: "float32" | "float16" (halves npy/h5; .mat can't store float16 — scipy upcasts to float64 — so mat stays float32 with a warning)
    export_orientation: str = "HWT"  # axis order for final matrix exports (dfWarped, consolidated frameRoiDf/Ch): "HWT" (H,W,T; MATLAB/legacy parity) | "THW" (T,H,W). Internal reg_Ch/dff .npy are always (T,H,W).
    post_annotation_time_average: int = 0  # half-window N for temporal moving average after warp; 0 disables, 1 = ±1 frame (3-frame mean)
    post_annotation_filter_xyt: list[int] | None = None  # post-warp 3D filter window [x, y, t] on the atlas-warped stack (blank -> none)
    post_annotation_filter_kind: str = "mean"  # post-annotation filter kind: "mean" | "median" | "gaussian"
    tiff_format: str = "big-tiff"  # "big-tiff" | "ome-tiff"
    tiff_compression: bool = False  # zlib compression for TIFFs
    save_figures: str = "none"  # "none" | "png" | "png+pdf" — save emitted QC figures to <output>/figures/

    # --- PCA / ICA (section 02) ---
    pca_n_components: int = 10
    pca_smooth_sigma: float = 5.0  # Gaussian sigma (frames) for temporal smoothing of PCA components
    pca_skip_frames: int = 10  # temporal stride for PCA/ICA fitting input (memory/speed; 1 = every frame). Fit-only; final dF/F is full-length.
    skip_ica: bool = False  # skip ICA denoising entirely (ICA stage becomes a no-op)
    ica_n_components: int | None = None  # None → same as pca_n_components
    ica_max_iter: int = 1000
    ica_random_state: int = 0
    ica_exclusion: object = "cache"  # which ICs are artifacts: "cache" (replay ica_exclusion.json) | "gui" (pick them) | {"GCaMP": [2, 7]} (0-based, inline) | False (exclude nothing). Mirrors `annotation`: the interactive choice is recorded to ica_exclusion.json so a headless re-run reproduces it
    ica_denoise: str = "off"  # "off" (default; the exclusion is QC only, every saved artifact comes from the plain dF/F — MATLAB/notebook parity) | "subtract" (write ica/dff_{name}.npy = dF/F minus the excluded components, and let ROI / dfWarped / movies / correlation read THAT). Excluding nothing is the exact identity, so turning this on cannot change a value by itself

    # --- Atlas ---
    annotation_atlas_path: str = ""  # "" = config.USER_ATLAS, the per-user atlas built by `asovi-atlas --download`; otherwise a path to a .mat / .h5 atlas

    # --- Movie display ---
    save_movie_speed: float = 2.0  # realtime ×N (output fps = fps_channel * save_movie_speed)
    save_movie_codec: str = "MJPG"  # movie codec: "MJPG" (.avi) | "DIB (RAW)" (uncompressed .avi) | "mp4V" (.mp4). A raw 4-char FourCC is also accepted (written to .avi)
    save_movie_vminmax: tuple[float, float] = (-2.0, 10.0)  # dF/F [%] display (vmin, vmax)
    save_movie_cmap: str = "magma"  # movie LUT / colormap: "magma" | "turbo" | "gray" | "viridis"
    save_movie_merge_chs: bool = False  # when >1 source group, concatenate their movies horizontally into one {exp}_merged_dF file (off = one movie per group)

    def __post_init__(self) -> None:
        if self.input_format not in _INPUT_FORMATS:
            raise ValueError(
                f"input_format must be one of {_INPUT_FORMATS}, got {self.input_format!r}"
            )
        if self.input_order not in _INPUT_ORDERS:
            raise ValueError(
                f"input_order must be one of {_INPUT_ORDERS}, got {self.input_order!r}"
            )
        if self.save_annotated_dF_dtype not in _ANNOTATED_DF_DTYPES:
            raise ValueError(
                f"save_annotated_dF_dtype must be one of {_ANNOTATED_DF_DTYPES}, "
                f"got {self.save_annotated_dF_dtype!r}"
            )
        if self.channels_name is None:
            self.channels_name = list(_DEFAULT_CHANNELS_NAME)
        if self.channels_prop is None:
            self.channels_prop = list(_DEFAULT_CHANNELS_PROP)
        if len(self.channels_name) != len(self.channels_prop):
            raise ValueError(
                f"channels_name (len={len(self.channels_name)}) and "
                f"channels_prop (len={len(self.channels_prop)}) must have the same length"
            )
        for name in self.channels_name:
            if not isinstance(name, str) or not _CHANNEL_NAME_RE.match(name):
                raise ValueError(
                    f"channels_name entries must match {_CHANNEL_NAME_RE.pattern} "
                    f"(letters/digits/underscore/hyphen, non-empty); got {name!r}"
                )
        # Accept short forms: 's' -> 'source', 'd' -> 'donner' (case-insensitive).
        _prop_alias = {"s": "source", "d": "donner", "source": "source", "donner": "donner"}
        normalized_prop = []
        for prop in self.channels_prop:
            key = prop.strip().lower() if isinstance(prop, str) else prop
            if key not in _prop_alias:
                raise ValueError(
                    "channels_prop values must be 'donner'/'source' (or 's'/'d'), "
                    f"got {prop!r}"
                )
            normalized_prop.append(_prop_alias[key])
        self.channels_prop = normalized_prop
        if self.channels_slip is not None:
            slips = []
            for slip in self.channels_slip:
                try:
                    slips.append(int(slip))
                except (TypeError, ValueError):
                    raise ValueError(
                        "channels_slip entries must be integers (one per input "
                        f"file, in read order); got {slip!r}"
                    ) from None
            # [] means "no slip anywhere" — normalize so downstream code only
            # has to test `if config.channels_slip`.
            self.channels_slip = slips or None
        if not 0 <= self.ch_for_annotation < len(self.channels_name):
            raise ValueError(
                f"ch_for_annotation must be in [0, {len(self.channels_name)}), "
                f"got {self.ch_for_annotation}"
            )
        if not 0 <= self.template_ch < len(self.channels_name):
            raise ValueError(
                f"template_ch must be in [0, {len(self.channels_name)}), "
                f"got {self.template_ch}"
            )
        if not 0 <= self.demux_start_offset < len(self.channels_name):
            raise ValueError(
                f"demux_start_offset must be in [0, {len(self.channels_name)}), "
                f"got {self.demux_start_offset}"
            )
        if self.roi_signal not in ("Both", "Raw", "dff"):
            raise ValueError(
                f"roi_signal must be one of 'Both', 'Raw', 'dff'; "
                f"got {self.roi_signal!r}"
            )
        if self.roi_space not in ("atlas", "source"):
            raise ValueError(
                f"roi_space must be 'atlas' or 'source'; got {self.roi_space!r}"
            )
        if self.save_roi_signals is False:
            self.save_roi_signals = None
        elif isinstance(self.save_roi_signals, str):
            normalized = self.save_roi_signals.lower()
            if normalized not in ("csv", "pickle", "both"):
                raise ValueError(
                    f"save_roi_signals must be None/False/'csv'/'pickle'/'both'; "
                    f"got {self.save_roi_signals!r}"
                )
            self.save_roi_signals = normalized
        elif self.save_roi_signals is not None:
            raise ValueError(
                f"save_roi_signals must be None/False/'csv'/'pickle'/'both'; "
                f"got {self.save_roi_signals!r}"
            )
        if self.pca_n_components < 1:
            raise ValueError(
                f"pca_n_components must be >= 1, got {self.pca_n_components}"
            )
        if self.pca_smooth_sigma < 0:
            raise ValueError(
                f"pca_smooth_sigma must be >= 0, got {self.pca_smooth_sigma}"
            )
        if self.pca_skip_frames < 1:
            raise ValueError(
                f"pca_skip_frames must be >= 1, got {self.pca_skip_frames}"
            )
        if self.registration_batch_size < 1:
            raise ValueError(
                f"registration_batch_size must be >= 1, got {self.registration_batch_size}"
            )
        if self.ica_n_components is not None and self.ica_n_components < 1:
            raise ValueError(
                f"ica_n_components must be >= 1 or None, got {self.ica_n_components}"
            )
        if isinstance(self.ica_exclusion, dict):
            names = set(self.channels_name)
            for key, ics in self.ica_exclusion.items():
                if key not in names:
                    raise ValueError(
                        f"ica_exclusion names group {key!r}, which is not in "
                        f"channels_name {sorted(names)}"
                    )
                if not all(isinstance(i, (int, np.integer)) and i >= 0 for i in ics):
                    raise ValueError(
                        f"ica_exclusion[{key!r}] must be 0-based IC indices, got {ics!r}"
                    )
        elif self.ica_exclusion not in ("cache", "gui", False, None):
            raise ValueError(
                "ica_exclusion must be 'cache' / 'gui' / False / a "
                f"{{group: [ic, ...]}} dict; got {self.ica_exclusion!r}"
            )
        if self.ica_denoise not in ("off", "subtract"):
            raise ValueError(
                f"ica_denoise must be 'off' or 'subtract', got {self.ica_denoise!r}"
            )
        if self.ica_denoise != "off" and self.skip_ica:
            raise ValueError(
                "ica_denoise='subtract' needs the ICA stage, but skip_ica is on: "
                "there would be no basis to subtract anything with."
            )
        if not 0.0 <= float(self.baseline_percentile) <= 100.0:
            raise ValueError(
                f"baseline_percentile must be in [0, 100], "
                f"got {self.baseline_percentile}"
            )
        if not 0.0 <= float(self.baseline_percentile_highpass_rank) <= 100.0:
            raise ValueError(
                f"baseline_percentile_highpass_rank must be in [0, 100], "
                f"got {self.baseline_percentile_highpass_rank}"
            )
        if float(self.baseline_percentile_highpass_sec) <= 0.0:
            raise ValueError(
                f"baseline_percentile_highpass_sec must be > 0, "
                f"got {self.baseline_percentile_highpass_sec}"
            )
        if self.start_initial_frames is not None and self.start_initial_frames < 0:
            raise ValueError(
                f"start_initial_frames must be >= 0 or None, "
                f"got {self.start_initial_frames}"
            )
        if self.ignore_last_frames < 0:
            raise ValueError(
                f"ignore_last_frames must be >= 0, got {self.ignore_last_frames}"
            )
        if self.save_figures not in ("none", "png", "png+pdf"):
            raise ValueError(
                "save_figures must be 'none' / 'png' / 'png+pdf', "
                f"got {self.save_figures!r}"
            )
        if self.export_orientation not in ("HWT", "THW"):
            raise ValueError(
                "export_orientation must be 'HWT' / 'THW', "
                f"got {self.export_orientation!r}"
            )
        if self.registration_cache not in ("force", "cached"):
            raise ValueError(
                "registration_cache must be 'force' / 'cached', "
                f"got {self.registration_cache!r}"
            )
        if self.post_annotation_time_average < 0:
            raise ValueError(
                f"post_annotation_time_average must be >= 0, "
                f"got {self.post_annotation_time_average}"
            )
        from .filters import FILTER_KINDS

        for _field in ("filter_xyt_kind", "post_annotation_filter_kind"):
            if getattr(self, _field) not in FILTER_KINDS:
                raise ValueError(
                    f"{_field} must be one of {FILTER_KINDS}, "
                    f"got {getattr(self, _field)!r}"
                )
        # A 2-element window is accepted by the GUI's free-text list field and then
        # unpacked as (fx, fy, ft) deep inside preprocess -- i.e. AFTER the whole
        # registration pass. Reject it here, where the message is still actionable.
        for _field in ("filter_xyt", "post_annotation_filter_xyt"):
            win = getattr(self, _field)
            if win is None:
                continue
            if len(win) != 3 or any(int(v) < 1 for v in win):
                raise ValueError(
                    f"{_field} must be None or exactly 3 positive ints (fx, fy, ft), "
                    f"got {win!r}"
                )
        # Ranges the widgets declare but nothing enforced: usfac=0 divided by zero
        # inside the registration kernel.
        if self.usfac < 1:
            raise ValueError(f"usfac must be >= 1, got {self.usfac}")
        if self.binning < 0:
            raise ValueError(f"binning must be >= 0 (0 = no binning), got {self.binning}")
        if float(self.fps) <= 0:
            raise ValueError(f"fps must be > 0, got {self.fps}")
        if self.ica_max_iter < 1:
            raise ValueError(
                f"ica_max_iter must be >= 1, got {self.ica_max_iter}"
            )
        if (
            not isinstance(self.save_movie_vminmax, (tuple, list))
            or len(self.save_movie_vminmax) != 2
        ):
            raise ValueError(
                f"save_movie_vminmax must be a (vmin, vmax) pair, "
                f"got {self.save_movie_vminmax!r}"
            )
        self.save_movie_vminmax = (
            float(self.save_movie_vminmax[0]),
            float(self.save_movie_vminmax[1]),
        )
        _allowed_cmaps = ("magma", "turbo", "gray", "viridis")
        if self.save_movie_cmap not in _allowed_cmaps:
            raise ValueError(
                f"save_movie_cmap must be one of {_allowed_cmaps}, "
                f"got {self.save_movie_cmap!r}"
            )

    @property
    def cycle_len(self) -> int:
        """Number of channels in one cycle."""
        return len(self.channels_name)

    def channel_token(self, ch_idx: int) -> str:
        """Canonical per-channel token used across all output names.

        ``Ch{i}_{name}-{prop}`` (e.g. ``Ch0_BL-source``).  Both the output
        folder name and the file-name prefix are built from this, so every
        per-channel TIFF (raw/registered/annotated) and the quick-preview
        labels line up on the same channel / role identifier.
        """
        return f"Ch{ch_idx}_{self.channels_name[ch_idx]}-{self.channels_prop[ch_idx]}"

    @property
    def resolved_start_initial_frames(self) -> int:
        """Initial frames to skip for regression FOI / baseline percentile.

        Defaults to ``floor(4 * fps / cycle_len)`` (≈ 4 s of unstable illumination
        in the per-channel timeline) when ``start_initial_frames`` is None.
        """
        if self.start_initial_frames is not None:
            return int(self.start_initial_frames)
        return int(4 * self.fps // max(1, self.cycle_len))

    def roi_plot_window(self, n_frames: int) -> tuple[int, int]:
        """The [start, end) per-channel frame window ROI plots are limited to.

        Drops the initial unstable frames (``resolved_start_initial_frames``)
        and the trailing ``ignore_last_frames``.  Falls back to the full range
        if the two ends would cross.
        """
        start = min(self.resolved_start_initial_frames, max(n_frames - 1, 0))
        end = n_frames - int(self.ignore_last_frames)
        if end <= start:
            return 0, n_frames
        return start, end

    def channel_groups(self) -> dict[str, dict[str, list[int]]]:
        """Group channel indices by name and property.

        Returns dict keyed by unique channel name, each value is:
          {"donner_indices": [...], "source_indices": [...]}
        where indices are positions within the cycle (0..N-1).
        """
        groups: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: {"donner_indices": [], "source_indices": []}
        )
        for i, (name, prop) in enumerate(
            zip(self.channels_name, self.channels_prop)
        ):
            groups[name][f"{prop}_indices"].append(i)
        return dict(groups)

    def has_donner(self, name: str) -> bool:
        """Whether the given channel name has at least one donner channel."""
        groups = self.channel_groups()
        return name in groups and len(groups[name]["donner_indices"]) > 0


@dataclass
class PreprocessStats:
    output_format: str
    output_dir: str
    total_seconds: float = 0.0
    template_seconds: float = 0.0
    align_seconds: float = 0.0
    linear_subt_seconds: float = 0.0
    save_seconds: float = 0.0
    files_processed: int = 0
    total_frames: int = 0
    channel_frame_counts: dict[str, int] = field(default_factory=dict)
    batches_saved: int = 0
    cached: bool = False  # True when preprocess was skipped, reusing existing outputs
    # True when the user pressed Stop: the outputs are a valid PREFIX of the
    # recording, not the recording. Callers must not report the stage as Done,
    # and a later registration_cache="cached" run must not reuse it.
    cancelled: bool = False
    # --- demux (channel-cycle) QC summary; see asvimg.demux_qc ---
    demux_qc_conclusive: bool = False  # channels intensity-separable & enough frames to judge
    demux_separability: float = 0.0  # eta^2: intensity variance explained by channel phase
    demux_dominant_period: int | None = None  # strongest intensity period (frames)
    demux_slip_detected: bool = False  # a phase slip (dropped-frame channel shift) was found


LEGACY_CONFIG_NAME = "pipeline01_config.yaml"


def load_cli_config(config_path: str | Path | None) -> tuple["PipelineConfig", str]:
    """Config for a CLI run, plus one line saying where it came from.

    ``--config`` has to be optional.  An installed copy has no checkout to find a
    YAML in, so ``asovi-run --input-dir DIR`` must work on its own; requiring a
    file that only exists in the repository made both CLIs die with
    ``FileNotFoundError: pipeline01_config.yaml`` for every pip user.

    A ``pipeline01_config.yaml`` in the working directory is still picked up when
    no ``--config`` is given -- that is what running from a checkout has always
    done -- but the caller is told, instead of it happening invisibly.
    """
    if config_path:
        return load_config(Path(config_path)), f"config: {config_path}"
    legacy = Path(LEGACY_CONFIG_NAME)
    if legacy.is_file():
        return load_config(legacy), f"config: ./{LEGACY_CONFIG_NAME} (found in the working directory)"
    return PipelineConfig(), "config: built-in defaults (no --config given)"


# The atlas is built on the machine that uses it, into this project's user-state
# directory, by `asovi-atlas --download`.  It is NOT shipped inside the package:
# it is derived from the Allen Mouse Brain Common Coordinate Framework, whose
# terms are not the MIT terms this code carries, so redistributing it inside an
# MIT wheel would have misstated what a user is allowed to do with it.  Building
# it locally also means the user gets the *correct* region names, which the
# MATLAB-era file that used to ship here did not have.
USER_ATLAS = Path.home() / ".asovi" / "atlas" / "wfciAnnotationData_generated.h5"

ATLAS_SETUP_HINT = (
    "Build it once with:\n"
    "    uv run asovi-atlas --download        (or: asovi-atlas --download)\n"
    "It downloads the Allen CCF volumes (~4.8 GB, figshare 25365829, CC BY 4.0)\n"
    "and the structure tree, then writes the atlas to the path above. Set\n"
    "`annotation_atlas_path` if you keep it somewhere else."
)


def resolve_atlas_path(annotation_atlas_path: str | Path | None) -> Path:
    """The atlas file to open for a config value.

    Empty (the default) means :data:`USER_ATLAS`, the per-user atlas that
    ``asovi-atlas`` builds.  Keeping the *config* value empty rather than an
    absolute path is deliberate: the GUI folds it into the annotation stage
    signature, so a machine-specific path would mark the stage stale merely for
    opening the same output folder on another machine.

    This resolves a path; it does not check that the file is there.  The loader
    (:func:`asvimg.atlas.load_atlas`) is what reports a missing atlas, so a
    stage signature can be computed without an atlas on disk.
    """
    if annotation_atlas_path is None or str(annotation_atlas_path) == "":
        return USER_ATLAS
    return Path(annotation_atlas_path)


def load_config(config_path: Path) -> PipelineConfig:
    """Load pipeline config from YAML, merging with defaults.

    ``config_path`` can be:
    - A single YAML file (legacy ``pipeline01_config.yaml``)
    - A directory containing ``db.yaml`` and/or ``ops.yaml``
    - Either ``db.yaml`` or ``ops.yaml`` directly (sibling file is auto-loaded)
    """
    config_path = Path(config_path)
    if config_path.is_dir():
        raw = _load_split_yamls(config_path)
    elif config_path.name in ("db.yaml", "ops.yaml"):
        raw = _load_split_yamls(config_path.parent)
    else:
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    _strip_deprecated(raw)
    _apply_renames(raw)
    base = PipelineConfig()
    payload = asdict(base)
    payload.update(raw)
    return PipelineConfig(**payload)


def _load_split_yamls(directory: Path) -> dict:
    raw: dict = {}
    for name in ("db.yaml", "ops.yaml"):
        p = directory / name
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            raw.update(data)
    return raw


_DEPRECATED_FIELDS = {
    "start_frame_bl": "Use channels_name/channels_prop instead.",
    "bl_only": "Use channels_name/channels_prop instead.",
    "save_each_ch": "Use save_raw_each_ch and/or save_registered_each_ch instead.",
    "atlas_path": "Renamed to annotation_atlas_path.",
    "movie_speed_factor": "Renamed to save_movie_speed.",
    "movie_vmin_dff": "Use save_movie_vminmax[0] instead.",
    "movie_vmax_dff": "Use save_movie_vminmax[1] instead.",
}


_RENAMED_FIELDS: dict[str, str] = {
    "flag_linear_subt": "linear_subt",
    "flag_binning": "binning",
    "flag_flip": "flip",
    "flag_delete": "delete",
    "ignore_initial_frames": "start_initial_frames",
    "batch_size": "registration_batch_size",  # revived as the registration batch chunk
    # pca_n_skip's original meaning (temporal stride for PCA input) is restored
    # by pca_skip_frames. Rename (not deprecate) so old YAML values carry over;
    # _apply_renames runs after _strip_deprecated, so it must NOT be in the
    # deprecated set above.
    "pca_n_skip": "pca_skip_frames",
}


def _apply_renames(raw: dict) -> None:
    for old, new in _RENAMED_FIELDS.items():
        if old in raw:
            warnings.warn(
                f"Config field '{old}' has been renamed to '{new}'.",
                DeprecationWarning,
                stacklevel=2,
            )
            raw.setdefault(new, raw[old])
            del raw[old]


def _strip_deprecated(raw: dict) -> None:
    for deprecated, hint in _DEPRECATED_FIELDS.items():
        if deprecated in raw:
            warnings.warn(
                f"Config field '{deprecated}' is deprecated and ignored. {hint}",
                DeprecationWarning,
                stacklevel=2,
            )
            del raw[deprecated]


def save_config(config: PipelineConfig, output_dir: str | Path) -> None:
    """Save pipeline config as separate db.yaml and ops.yaml for reproducibility."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = asdict(config)
    db_data = {k: v for k, v in data.items() if k in _DB_FIELDS}
    ops_data = {k: v for k, v in data.items() if k not in _DB_FIELDS}
    with (output_dir / "db.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(db_data, f, sort_keys=False, allow_unicode=True)
    with (output_dir / "ops.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(ops_data, f, sort_keys=False, allow_unicode=True)


def default_output_dir(input_dir: Path, output_format: str) -> Path:
    return input_dir / "asi" / output_format


def infer_exp_name(file_name: str) -> str | None:
    match = re.search(r"timelapse(.*?)_MM", file_name)
    if match:
        return match.group(1)

    stem = Path(file_name).stem
    return stem if stem else None


# ---------------------------------------------------------------------------
# Deprecated aliases (kept for backward compatibility with older YAMLs and
# notebooks). Instantiating them emits a DeprecationWarning.
# ---------------------------------------------------------------------------


class Pipeline01Config(PipelineConfig):
    """Deprecated alias for :class:`PipelineConfig`."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Pipeline01Config is renamed to PipelineConfig",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class Pipeline01RunStats(PreprocessStats):
    """Deprecated alias for :class:`PreprocessStats`."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Pipeline01RunStats is renamed to PreprocessStats",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
