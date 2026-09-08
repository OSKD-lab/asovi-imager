"""Declarative widget specs for every editable PipelineConfig field.

Single source of truth driving the auto-generated config form: each field maps
to a widget kind, a section, choices/range, and a tooltip.  ``is_db`` is taken
from ``PipelineConfig``'s ``_DB_FIELDS`` so the form can group the six
reproducibility-critical fields first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from asvimg.config import _DB_FIELDS


@dataclass
class FieldSpec:
    name: str
    kind: str  # bool|int|float|text|combo|str_list|int_list_or_none|float_pair|
    #            int_or_none|annotation_mode
    section: str
    choices: list[str] = field(default_factory=list)
    min: float | None = None
    max: float | None = None
    tooltip: str = ""
    group: str = ""  # optional sub-category shown as a separator within a section
    label: str = ""  # display label override (falls back to `name`)

    @property
    def is_db(self) -> bool:
        return self.name in _DB_FIELDS


# Order here is the display order within each section.
FIELD_SPECS: list[FieldSpec] = [
    # --- Data (db) ---
    FieldSpec("input_dir", "text_dir", "Data", tooltip="Input dir (tif/dcimg/sifx) — '...' opens a folder picker"),
    FieldSpec("output_dir", "text_dir", "Data", tooltip="Output dir (blank -> input_dir/asi/<format>) — '...' opens a folder picker"),
    FieldSpec("input_format", "combo", "Data", choices=["auto", "tif", "dcimg", "sifx", "h5"],
              tooltip="Which input format to read from input_dir. 'auto' detects the one format present and errors if the folder mixes formats; pick tif (.tif/.tiff) / dcimg / sifx / h5 (Ito even/odd folder) to force one and resolve a mix."),
    FieldSpec("input_order", "combo", "Data", choices=["natural", "name", "mtime", "ctime"],
              tooltip="Order the input files are read and concatenated in (one continuous timeline). "
                      "natural = natsort by name (rec_2 before rec_10; default, matches spooled-file naming and what Windows Explorer shows). "
                      "name = plain lexicographic by name (rec_10 before rec_2) — the byte-order sort MATLAB dir / Python sorted() give. "
                      "mtime = oldest modification time first — for camera output this is acquisition order, and it survives copying. "
                      "ctime = oldest creation time first — WARNING: a copied or backup-restored dataset gets NEW creation times, so this is often the copy order, not acquisition order. "
                      "Use 'Quick Preview (All)': it lists every file in the resulting read order with both timestamps, so you can confirm before running. "
                      "Changing this also changes which file is 'first' (exp_name inference) and the order channels_slip entries apply in."),
    FieldSpec("output_format", "combo", "Data", choices=["mat", "npy", "h5"]),
    FieldSpec("exp_name", "text", "Data", tooltip="Experiment name (blank -> infer from filename)"),
    FieldSpec("dcimg_backend", "combo", "Data", choices=["auto", "sdk", "native"]),
    FieldSpec("use_mmap", "bool", "Data",
              tooltip="dF/F read strategy (output identical either way). off = load each registered channel fully into RAM (fastest; short recordings / big machines). on = read reg_Ch via memmap and stream dF/F in row-strips (bounded peak memory for long recordings)."),

    # --- Channels ---
    FieldSpec("channels_name", "str_list", "Channels", tooltip="Cycle channel names, comma-separated"),
    FieldSpec("channels_prop", "str_list", "Channels", tooltip="'source' or 'donner' per channel, comma-separated"),
    FieldSpec("channels_slip", "int_list_or_none", "Channels",
              tooltip="Per-input-file channel-cycle phase, one entry per input file in read order (blank = none). e.g. 0,1,0 = the 2nd file's cycle starts 1 step in (its first frame is channels_name[1]). Use 'Quick Preview (All)' to see each file's head and verify."),
    FieldSpec("demux_qc", "bool", "Channels", tooltip="Intensity-based demux QC: verify the positional channel cycle against per-frame mean intensity and flag phase slips (detection only; never changes demux)"),
    FieldSpec("demux_start_offset", "int", "Channels", min=0,
              tooltip="Global channel-cycle phase: the first frame of the recording is channels_name[demux_start_offset]. Rotates the whole assignment (a demux_correction.json sidecar, if present, wins over this)."),
    FieldSpec("ch_for_annotation", "int", "Channels", min=0, label="ch_for_annotation (0-based)", tooltip="Channel index (0-based) used for PCA/annotation source"),
    FieldSpec("template_ch", "int", "Channels", min=0, label="template_ch (0-based)", tooltip="Cycle channel index (0-based) for auto-template"),

    # --- Preprocess (field order follows the processing order:
    #     registration -> binning -> filter -> dF/F(detrend -> high-pass -> subtraction)) ---
    # 1) Registration (flip is applied per-frame first, then DFT registration)
    FieldSpec("flip", "bool", "Preprocess", group="1. Registration", tooltip="Left-right flip (applied per-frame before registration)"),
    FieldSpec("do_registration", "bool", "Preprocess", group="1. Registration", tooltip="DFT registration (motion correction)"),
    FieldSpec("registration_cache", "combo", "Preprocess", group="1. Registration", choices=["force", "cached"],
              tooltip="force: always (re)run preprocess; cached: skip when reg_Ch*.npy already exist in the output dir"),
    FieldSpec("registration_batch_size", "int", "Preprocess", group="1. Registration", min=1,
              tooltip="Frames per registration batch (read -> register_batch -> bin -> write); bounds registration read memory."),
    FieldSpec("usfac", "int", "Preprocess", group="1. Registration", min=1, tooltip="DFT upsampling factor (higher = finer, slower)"),
    FieldSpec("template", "text", "Preprocess", group="1. Registration", tooltip="Template image path (blank -> auto-create)"),
    FieldSpec("template_stride", "int", "Preprocess", group="1. Registration", min=1, tooltip="Frame interval for template averaging"),
    # 2) Binning + spatiotemporal filter (after registration, over the full timeline)
    FieldSpec("binning", "int", "Preprocess", group="2. Binning + filter", min=0, tooltip="Spatial binning factor (2 = 2x2), applied after registration"),
    FieldSpec("filter_xyt", "int_list_or_none", "Preprocess", group="2. Binning + filter", tooltip="Post-reg 3D filter [x,y,t] e.g. 1,1,1 over the full timeline (blank -> none)"),
    FieldSpec("filter_xyt_kind", "combo", "Preprocess", group="2. Binning + filter", choices=["mean", "median", "gaussian"], tooltip="filter_xyt kind: mean (box) / median / gaussian"),
    # 3) dF/F — linear subtraction (pre-steps run in this order: detrend -> high-pass -> regression -> baseline)
    FieldSpec("linear_subt", "bool", "Preprocess", group="3. dF/F (linear subtraction)", tooltip="WFCI linear subtraction (dF/F)"),
    FieldSpec("detrend", "bool", "Preprocess", group="3. dF/F (linear subtraction)",
              tooltip="Exponential (photobleaching) detrend of src & donner before the WFCI regression (MATLAB flag_ExpoSub)"),
    FieldSpec("baseline_percentile_highpass", "bool", "Preprocess", group="3. dF/F (linear subtraction)",
              label="high-pass (rolling-percentile)",
              tooltip="Rolling-percentile ratiometric high-pass of src & donner before the regression (David Whitney; MATLAB flag_BaselineFilter). When detrend is also on, detrend runs first"),
    FieldSpec("baseline_percentile_highpass_sec", "float", "Preprocess", group="3. dF/F (linear subtraction)", min=0,
              label="high-pass window (s)", tooltip="Rolling-percentile window (seconds) for the high-pass baseline"),
    FieldSpec("baseline_percentile_highpass_rank", "float", "Preprocess", group="3. dF/F (linear subtraction)", min=0, max=100,
              label="high-pass percentile", tooltip="Percentile rank (0-100) for the rolling high-pass baseline (50 = median)"),
    FieldSpec("baseline_percentile", "float", "Preprocess", group="3. dF/F (linear subtraction)", min=0, max=100, tooltip="Per-pixel percentile for the (static) dF/F baseline"),
    FieldSpec("start_initial_frames", "int_or_none", "Preprocess", group="3. dF/F (linear subtraction)", min=0, tooltip="Initial frames skipped from regression/baseline & ROI plots (blank -> auto)"),
    FieldSpec("ignore_last_frames", "int", "Preprocess", group="3. dF/F (linear subtraction)", min=0, tooltip="Trailing frames excluded from regression & ROI plots (0 = keep all)"),
    FieldSpec("hemovar_qc", "bool", "Preprocess", group="3. dF/F (linear subtraction)",
              label="hemo-var QC map",
              tooltip="Save a per-group hemo variance-explained (R²) map: fraction of source variance the donner regression removes. Writes hemovar_{name}.npy + a figure"),
    # 4) General
    FieldSpec("fps", "int", "Preprocess", group="4. General", min=1, tooltip="Camera fps (all channels combined)"),
    FieldSpec("max_frames", "int_or_none", "Preprocess", group="4. General", min=1, tooltip="Frame limit for quick tests (blank -> all)"),
    FieldSpec("delete", "bool", "Preprocess", group="4. General", label="overwrite (delete previous outputs)", tooltip="Overwrite: delete previous outputs in the output dir before running preprocess"),

    # --- Annotation ---
    FieldSpec("annotation", "annotation_mode", "Annotation", choices=["cache", "gui", "False"],
              tooltip="cache = use saved marks; gui = pick points; False = skip"),
    FieldSpec("annotation_allow_reflection", "bool", "Annotation", tooltip="Allow a mirror (left-right flip) in the atlas transform; needs non-midline control points"),
    FieldSpec("annotation_atlas_path", "text", "Annotation", tooltip="Atlas annotation .mat path"),
    FieldSpec("post_annotation_time_average", "int", "Annotation", min=0, tooltip="Half-window for post-warp temporal averaging (0 = off)"),
    FieldSpec("post_annotation_filter_xyt", "int_list_or_none", "Annotation", tooltip="Post-warp 3D filter [x,y,t] on the atlas-warped stack (blank -> none)"),
    FieldSpec("post_annotation_filter_kind", "combo", "Annotation", choices=["mean", "median", "gaussian"], tooltip="Post-annotation filter kind: mean (box) / median / gaussian"),

    # --- PCA / ICA ---
    FieldSpec("pca_n_components", "int", "PCA/ICA", min=1),
    FieldSpec("pca_smooth_sigma", "float", "PCA/ICA", min=0, tooltip="Temporal smoothing sigma (frames)"),
    FieldSpec("pca_skip_frames", "int", "PCA/ICA", min=1,
              tooltip="Temporal stride for PCA/ICA fitting input (memory/speed; 1 = every frame). Fit-only; final dF/F stays full-length."),
    FieldSpec("skip_ica", "bool", "PCA/ICA", label="skip_ica (skip ICA denoising)", tooltip="Skip the ICA denoising stage entirely"),
    FieldSpec("ica_n_components", "int_or_none", "PCA/ICA", min=1, tooltip="Blank -> same as PCA"),
    FieldSpec("ica_max_iter", "int", "PCA/ICA", min=1),
    FieldSpec("ica_random_state", "int", "PCA/ICA"),
    FieldSpec("ica_exclusion", "ica_mode", "PCA/ICA", choices=["cache", "gui", "False"],
              tooltip="Which ICs are artifacts: cache = replay the choice recorded in ica_exclusion.json; gui = pick them (one window per channel group); False = exclude nothing"),
    FieldSpec("ica_denoise", "combo", "PCA/ICA", choices=["off", "subtract"],
              tooltip="off (default) = the IC exclusion is QC only; every saved artifact comes from the plain dF/F (MATLAB/notebook parity). subtract = ROI signals, dfWarped, movies and correlation are computed from dF/F with the excluded components subtracted. Excluding nothing is the exact identity, so turning this on cannot change a value by itself."),

    # --- ROI ---
    FieldSpec("roi_signal", "combo", "ROI", choices=["Both", "Raw", "dff"]),
    FieldSpec("roi_space", "combo", "ROI", choices=["atlas", "source"],
              tooltip="atlas: ROIs are Allen-atlas coords, pulled into the recording via the annotation transform (needs annotation done). source: ROIs are already in the recording's own coords (rois_source.csv) and read directly — no atlas registration, so ROI/correlation run even with annotation=False"),
    FieldSpec("corr_method", "combo", "ROI", choices=["raw", "gsr", "partial"],
              tooltip="ROI correlation: raw Pearson (near 1 from global signal) / gsr = global-signal-regressed / partial = partial correlation"),
    FieldSpec("corr_auto_threshold", "bool", "ROI", tooltip="Network edges: auto (proportional/density) threshold that adapts to each method's scale"),
    FieldSpec("corr_edge_density", "float", "ROI", min=0, max=1, tooltip="When auto: fraction of strongest edges to keep (0.2 = top 20%)"),
    FieldSpec("corr_network_threshold", "float", "ROI", min=0, max=1, tooltip="Fixed |r| edge cutoff used when auto threshold is off"),

    # --- Outputs — every "save file" flag, grouped by category ---
    # Frames (registered / warped stacks as TIFF or payload)
    FieldSpec("save_raw_each_ch", "bool", "Outputs", group="Frames", tooltip="Save raw (pre-processing) channels as TIFF"),
    FieldSpec("save_registered_each_ch", "bool", "Outputs", group="Frames", tooltip="Save registered+binned channels as TIFF"),
    FieldSpec("save_annotated_each_ch", "bool", "Outputs", group="Frames", tooltip="Save atlas-warped channels as TIFF"),
    FieldSpec("save_annotated_dF_mat", "bool", "Outputs", group="Frames", tooltip="Save atlas-warped dF/F per group (mat/npy/h5)"),
    FieldSpec("save_annotated_dF_dtype", "combo", "Outputs", group="Frames", choices=["float32", "float16"],
              label="save_annotated_dF_dtype",
              tooltip="dtype of the saved atlas-warped dF/F payload. float16 halves npy/h5 file size; .mat can't store float16 (scipy upcasts to float64), so a mat export stays float32 with a warning."),
    FieldSpec("export_orientation", "combo", "Outputs", group="Frames", choices=["HWT", "THW"],
              tooltip="Axis order for final matrix exports (dfWarped, consolidated frameRoiDf/Ch). HWT = H,W,T (MATLAB / legacy parity); THW = T,H,W. Internal reg_Ch/dff .npy stay T,H,W."),
    FieldSpec("tiff_format", "combo", "Outputs", group="Frames", choices=["big-tiff", "ome-tiff"]),
    FieldSpec("tiff_compression", "bool", "Outputs", group="Frames"),
    # Movie
    FieldSpec("save_movie", "bool", "Outputs", group="Movie", tooltip="Save atlas-warped dF/F movies"),
    FieldSpec("save_movie_speed", "float", "Outputs", group="Movie", min=0, tooltip="Realtime xN output fps"),
    FieldSpec("save_movie_codec", "combo", "Outputs", group="Movie", choices=["MJPG", "DIB (RAW)", "mp4V"],
              tooltip="Movie codec: MJPG (.avi) / DIB (RAW) = uncompressed .avi / mp4V (.mp4). Uncompressed files are large; some players can't play raw AVI."),
    FieldSpec("save_movie_vminmax", "float_pair", "Outputs", group="Movie", tooltip="dF/F display (vmin, vmax) in %"),
    FieldSpec("save_movie_cmap", "combo", "Outputs", group="Movie", choices=["magma", "turbo", "gray", "viridis"],
              label="save_movie_cmap (LUT)", tooltip="Movie LUT / colormap applied to the dF/F movie"),
    FieldSpec("save_movie_merge_chs", "bool", "Outputs", group="Movie", label="merge_Chs",
              tooltip="When there is more than one source channel group, concatenate their movies horizontally into a single {exp}_merged_dF movie. Off = one movie per group."),
    # ROI
    FieldSpec("save_roi_signals", "combo", "Outputs", group="ROI", choices=["None", "csv", "pickle", "both"],
              tooltip="Export ROI time-series as a DataFrame (csv / pickle / both)"),
    # Figures
    FieldSpec("save_figures", "combo", "Outputs", group="Figures", choices=["none", "png", "png+pdf"],
              tooltip="Save emitted QC figures to <output>/figures/ as PNG (png+pdf also writes editable PDFs)"),
    # Configs
    FieldSpec("output_metadata_yaml", "bool", "Outputs", group="Configs", tooltip="Save input metadata YAML"),
]

SECTIONS: list[str] = []
for _s in (  # follows the pipeline stage order (pca/ica run before annotation)
    "Data", "Channels", "Preprocess", "PCA/ICA", "Annotation", "ROI", "Outputs",
):
    if _s not in SECTIONS:
        SECTIONS.append(_s)


def specs_by_section() -> dict[str, list[FieldSpec]]:
    out: dict[str, list[FieldSpec]] = {s: [] for s in SECTIONS}
    for spec in FIELD_SPECS:
        out.setdefault(spec.section, []).append(spec)
    return out
