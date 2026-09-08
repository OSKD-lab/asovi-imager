# ASoVi_imager

Wide-field cortical imaging (WFCI) analyzer — Python pipeline for registration, linear subtraction (dF/F), PCA/ICA denoising, Allen CCFv3 atlas registration, and ROI signal extraction.

![GUI-img](docs/imgs/fig1.png)


> **Inspired by [suite2p](https://github.com/MouseLand/suite2p) (MouseLand).**
> The preprocessing design — batched registration, a per-channel contiguous
> memmap intermediate (`reg_Ch{i}.npy`, `(T,H,W)`), and batched reference FFT —
> follows suite2p. See [Inspiration & references](#inspiration--references) for
> the full list of algorithms, papers, and repositories this project builds on.

## Documentation

Three readers, three docs. This README is the reference for **what the knobs do and
what comes out**; everything else hangs off it.

| If you are… | Read |
| --- | --- |
| a scientist who just wants to run it | [**はじめかた (日本語ユーザーガイド)**](docs/getting_started_jp.md) — install, GUI, stage by stage, troubleshooting |
| setting up / configuring a run | this README — [entry points](#entry-points), [configuration](#configuration), [output layout](#output-file-layout) |
| about to change the code, or asking **"why is it like this?"** | the module docstrings and **`git log`** — see below. Benchmarks, rejected alternatives and deliberate behaviour changes are recorded in the commit messages; there is no hand-maintained changelog, because a second timeline only drifts from the first |

### Reading the history

Design decisions live in the commit messages, not in a doc. They record what was
measured, what was rejected and why, and which behaviours changed on purpose —
which is exactly what a doc stops telling you six months later.

```bash
git log --oneline -20                            # what has been happening
git log -p -- src/asvimg/wfci.py     # why this file looks like this
git log -S warp_weight_maps --oneline            # when a symbol was introduced (= the decision)
git log -G "dtype=np.float32" --oneline          # when a line matching a pattern changed
git log --grep=demux --oneline                   # everything about one subsystem
git blame src/asvimg/annotation.py   # then `git show <sha>` on the line you doubt
```

Deliberate numeric changes — a payload dtype, a rounding difference, an artifact
that is no longer bit-identical — are always spelled out in the commit that makes
them. Search the **messages** for those (`git log --grep=dtype`,
`git log --grep=float32`): `-S <name>` only finds commits that added or removed
that string, so it will miss a change to a symbol that already existed.

## Environment

- Python 3.13+, managed with [`uv`](https://github.com/astral-sh/uv)
- PyTorch is pinned to the CPU build in `pyproject.toml` — it runs the registration
  FFTs *and* the atlas-warp kernel (`grid_sample`)

**Install it** (import name `asvimg`, distribution name `asovi-imager`):

```bash
pip install git+https://github.com/OSKD-lab/asovi-imager
```

**Or work on it** — `uv sync` installs the dependencies *and* the project itself,
so the `asovi-*` commands and `import asvimg` both work from a checkout:

```bash
uv sync
uv run python -m pytest tests -q     # run from the repository root
```

> **The sample data is not in this repository.** `Analysis/_sampleData*/` is
> gitignored (it holds the maintainer's recordings), yet it is still the default
> `input_dir` in `pipeline01_config.yaml` and in `PipelineConfig`.
> Every `_sampleData*` path below is a placeholder — point `--input-dir` (or
> `input_dir:` in your YAML) at your own recordings, or the first command you run
> fails with `FileNotFoundError: No supported files found in …`.

## Entry points

| Entry point | Covers | When to use |
| --- | --- | --- |
| `asovi-gui` — `uv run python -m asvimg.gui` | Full pipeline via a Dear PyGui dashboard | **Recommended.** Run stage by stage, pick atlas control points, inspect figures |
| `asovi-run` — `uv run python -m asvimg.run_pipeline` | Full pipeline headless (preprocess → PCA/ICA → annotation → ROI → correlation → export) | Batch / scripted full runs |
| `asovi-preprocess` — `uv run python -m asvimg.preprocess` | Preprocess only (registration → linear subt → binning → save) | Batch / unattended preprocessing |
| `asovi-demux` — `uv run python -m asvimg.gui.demux_editor` | Standalone GUI to inspect + fix a mis-demultiplexed channel cycle | When the demux QC flags a phase slip / wrong starting phase |
| `asovi-nwb` — `uv run python -m asvimg.gui.nwb_editor [output_dir]` | Standalone GUI to package a processed folder into an **NWB** file (Neurodata Without Borders): enter Subject / session / device / per-channel metadata, pick payloads, write `<exp>.nwb`, run `nwbinspector` | When you want to share/archive a processed recording (e.g. DANDI upload) |
| `asovi-atlas` — `uv run python -m asvimg.atlas_build --out <atlas.h5>` | Build a top-view Allen-CCF cortical atlas (`ACCFv3`-readable) from the 3D annotation volume — isocortex dorsal projection, correct region labels, L/R split, any resolution, optional `--tilt-ap/--tilt-ml/--tilt-dv`, `--ap-crop/--ml-crop`, olfactory/cerebellum | When you need to (re)generate `annotation_atlas_path`. On first run it offers to download the CCF volumes (figshare 25365829, ~4.8 GB) into `~/.asovi/atlas/` (`--download` / `$ASOVI_ATLAS_DIR` / `--atlas-dir`) |
| `asovi-atlas-gui` — `uv run python -m asvimg.gui.atlas_editor` | GUI for the atlas builder: load the CCF volumes, **tilt** the top-view about the AP/ML/DV axes with a live preview (fixed point ROIs overlaid), then save | When you want to interactively pick the tilt that matches your imaging window |
| `asovi-sifx2tiff` — `uv run python -m asvimg.sifx_convert <path>` | Convert Andor `.sifx` spool(s) to stacked BigTIFF (`--out`, `--compress`, `--max-frames`) | When you want the spools as plain TIFFs. Not a pipeline step — `.sifx` is read directly |
| [notebook](notebooks/run_pipeline_full.ipynb) | The same steps, cell by cell | Exploratory work |

Both spellings work once the project is installed: the `asovi-*` console scripts, or the `python -m asvimg.…` equivalents shown beside them.

## GUI dashboard

```bash
uv run python -m asvimg.gui            # or: asovi-gui
```

The dashboard (`src/asvimg/gui/`) drives the headless `PipelineSession`:

- **Config pane** — every `db`/`ops` field, grouped into collapsing sections (`*` marks reproducibility/db fields). `...` opens the OS-native file dialog to locate an `ops.yaml`; **Load** reads a config (and marks stages already done on disk); **Save** writes `db.yaml`/`ops.yaml`.
- **Presets** — the dropdown applies **ops-only** YAML (`db` fields are never touched, so the input/output stay put): `(factory defaults)` resets every processing field to `PipelineConfig()`; `[std] …` are the read-only ones bundled in [`src/asvimg/presets/`](src/asvimg/presets); anything else is yours, saved by **Save as preset** into `~/.asovi/presets/`. **Read last config** restores the previous session's processing fields from `~/.asovi/last_ops.yaml` (auto-written on exit). A preset is an **overlay**: a key it does not carry keeps its current value rather than reverting to the default — load `(factory defaults)` first if you want a clean slate.
- **Stages** — `preprocess → pca → ica → annotation → roi → correlation → export`, each runnable on its own (they rehydrate what they need from disk) or all at once via **Run All**. Status reads `[ ] / [In Progress] / [Done] / [Error]` (errors in red). **Quick Preview** shows the first 12 frames of the *first* input file, labelled by the channel preprocess will assign them; **Quick Preview (All)** shows the first frames of *every* input file, one row per file — the view that exposes a per-file phase slip (fix it with `channels_slip`). Both log the input files with their sizes in read order.
- **The two interactive stages** — `ica` opens the IC picker (one window per channel group; tick the maps that look like vessels, breathing, or a light leak) and `annotation` opens cpselect. Each choice is written to disk (`ica_exclusion.json`, `marks.mat`) as soon as it is made, so the next run replays it instead of asking again — see *Reproducibility* below.
- **Figures** — each emitted plot is a collapsible panel; set `save_figures` to also write them to `<output>/figures/`.
- **Outputs pane** — three tools that run outside the stage sequence:
  - **Edit ROIs** — add / move / resize / mirror the atlas point ROIs, then **Save rois.csv**. Once `<output>/rois.csv` exists it **replaces the atlas defaults** for ROI extraction, correlation and the seed maps. It is a human decision, like `marks.mat` — keep it (see *Reproducibility*).
  - **Seed-based Corr. Maps** (needs annotation done) — a seed-ROI correlation map in atlas space → `<output>/corrMap/`. **Add for Annot.** inverse-warps the shown map back into source coordinates and drops it in `map_for_annot/`, so you can pick control points on it.
  - **Preview Movie** — play the warped dF/F movies without leaving the app.
- **Stop** cancels a run cooperatively. A cancelled preprocess reports **SKIPPED**, not Done, and stamps `partial` into `reg_meta.npz`: its outputs are a valid *prefix* of the recording, and calling that Done would let every later stage analyse a fraction of the data as if it were the whole thing.
- On any run the current params are written to `<output>/db.yaml` + `<output>/ops.yaml` and the config-path field is pointed at that folder.

## Full pipeline CLI

```bash
uv run python -m asvimg.run_pipeline --config pipeline01_config.yaml   # or: asovi-run
```

Runs `preprocess → pca → ica → annotation → roi → correlation → export` with no
windows: the interactive stages resolve from the config (`annotation: "cache"` /
`ica_exclusion: "cache"` replay `marks.mat` / `ica_exclusion.json`).

| Flag | Meaning |
| --- | --- |
| `--config` | YAML config file **or a directory holding `db.yaml`/`ops.yaml`** (e.g. a processed `asi/npy/` folder) |
| `--input-dir` / `--output-dir` / `--exp-name` / `--output-format` / `--max-frames` | The same overrides as the preprocess CLI |
| `--stages` | Comma-separated subset instead of all, e.g. `--stages roi,correlation` (order respected) |
| `--no-figures` | Do not save QC figures. **Without it `asovi-run` writes `<output>/figures/` regardless of `save_figures`** |

```bash
# redo just the ROI + correlation analysis against an edited rois.csv
uv run python -m asvimg.run_pipeline --config out/asi/npy --stages roi,correlation
```

## Preprocess CLI

```bash
uv run python -m asvimg.preprocess --config pipeline01_config.yaml   # or: asovi-preprocess
```

Common overrides:

```bash
uv run python -m asvimg.preprocess \
    --config pipeline01_config.yaml \
    --input-dir /path/to/your/recordings \
    --output-dir /path/to/your/recordings \
    --output-format mat \
    --max-frames 500
```

| Flag | Meaning |
| --- | --- |
| `--config` | YAML config file, or a directory holding `db.yaml`/`ops.yaml` |
| `--input-dir` | Directory containing `.tif` / `.tiff` / `.dcimg` / `.sifx` files, or an even/odd `.h5` recording folder |
| `--output-dir` | Output directory (default: `<input_dir>/asi/<output_format>`) |
| `--output-format` | `mat` / `npy` / `h5` |
| `--exp-name` | Override experiment name (default: inferred from first input filename) |
| `--max-frames` | Hard cap on processed frames (useful for smoke tests) |
| `--dcimg-backend` | `auto` / `sdk` / `native` (auto prefers the official DCIMG runtime, falls back to the pure-Python parser) |
| `--no-metadata-yaml` | Skip writing `input_metadata.yaml` |
| `--print-stats` | Dump timing/frame stats as YAML after the run |

Sample full flag list: `uv run python -m asvimg.preprocess --help`.

### The bundled atlas

The atlas that ships in the wheel is the lab's original MATLAB one. Its geometry,
region boundaries, midline landmarks and **point ROIs** are correct — including the
left/right split, which comes from 15 hardcoded right-hemisphere coordinates
mirrored across the midline (30 ROIs, `VISp_R` … `MOs-al_L`), not from the file.

Its region *names* are not correct: the ID → acronym mapping is scrambled, and the
`_R` / `_L` suffixes are fictional because the ID map is bilateral — one ID spans
both hemispheres. So `ACCFv3.get_mask()` and `ACCFv3.region_names` **raise** on it
rather than return a plausible-looking wrong answer. Nothing in the pipeline uses
them; ROI signals come from the point ROIs.

If you need per-hemisphere *area* masks, build an atlas with `asovi-atlas` — it
splits each area at the midline and carries correct names — and point
`annotation_atlas_path` at the result.

## Supported inputs

- `.tif` / `.tiff` — including OME-TIFF from µManager (`timelapse..._MM` filenames auto-populate `exp_name`)
- `.dcimg` — Hamamatsu DCIMG (SDK runtime preferred, native parser fallback)
- `.sifx` — Andor spool directories. Frames read is the smaller of two counts, which guard opposite failures of a run that stopped early: the header's `NumberOfFrames` excludes the zero padding a spool file keeps when it was written whole, and the count taken from the `*spool.dat` file sizes excludes frames a nominal `ImagesPerFile` would claim past the end of a short last file. Random access (`SIFXFile.frame`) still spans every frame on disk
- Ito even/odd HDF5 recording — point `--input-dir` at the folder of `*_even_*.h5` / `*_odd_*.h5` parts. The parts are merged back into one stream ordered by `source_index` (the original interleaved order), so even/odd land on their channels via the usual positional demux. Frames are read as stored (any binning was applied at acquisition; the pipeline's `binning` still applies on top). Compression is read via HDF5 filters — blosc2/zstd/lz4 (needs `hdf5plugin`, a dependency) or gzip (built in) all work; no codec is special-cased.

A single `--input-dir` may contain a mix of the file-based formats (an HDF5 recording is instead a whole folder). If a folder holds **more than one** format, `input_format="auto"` refuses it — set `input_format` explicitly to pick one.

### Input read order

Every file of the chosen format in `input_dir` is concatenated into **one continuous timeline**, so their order decides where each frame lands. `input_order` picks it:

| value | order | when to use |
|---|---|---|
| `"natural"` | natsort by name — `rec_2` before `rec_10` | **default.** Matches how acquisition software names spooled/segmented parts of one recording — and what Windows Explorer shows (it sorts numerically too) |
| `"name"` | plain lexicographic by name — `rec_10` before `rec_2` | to reproduce the byte-order sort MATLAB `dir` / Python `sorted()` give, e.g. to match a legacy script |
| `"mtime"` | oldest **modification** time first | when the filenames do not sort into acquisition order. For camera output the write time *is* the acquisition time, and it survives copying |
| `"ctime"` | oldest **creation** time first | rarely — see the warning below |

Time orders tie-break on the filename, so a batch sharing one timestamp is still deterministic.

> **`ctime` is usually not acquisition order.** On Windows a file copied to another disk, or restored from a backup, gets a **new** creation time — the copy time. On a dataset copied to a network share the `ctime` order is typically shuffled beyond recognition while `mtime` still holds the original acquisition sequence. Prefer `mtime`, and confirm with the preview below.

**Check the order before you run it.** `Quick Preview (All)` (`preview_input_all`) lists every file in the resulting read order with **both** timestamps, and renders the head of each file as one row per file, top to bottom in read order. The header marks the column actually sorted on with `<-` (for `mtime`/`ctime` every row carries it too; for `natural`/`name` the sort key is the filename, so the mark sits on the name column). Read down the marked column and confirm it runs in the order you expect — that is the whole point of the view. Both timestamps are printed whichever order is selected, so you can also see at a glance whether a *different* order would have been the better choice.

Two things follow the read order and change with it:

- the **first** file names the experiment. Setting `exp_name` pins the `{exp_name}_*` outputs, but the `roiSignals_*` stem comes from that first file either way (`exp_name` does not rename it — see the `exp_name` row below), so re-ordering renames `roiSignals_*`
- `channels_slip` entries are per input file **in read order**, so re-ordering re-targets them

### Accepted HDF5 recording (`H5Recording`)

The reader accepts any set of `.h5` parts in the folder that meet this contract:

| Item | Required? | Notes |
|---|---|---|
| `frames` dataset, shape `(T, H, W)` | **required** | the only mandatory dataset; read frame-by-frame |
| `source_indices` dataset (`int64`, len `T`) | required for ordering | else falls back to attrs `source_start_frame` + `source_frame_step`; absent both → degenerate `0,1,2,…` |
| compression filter on `frames` | any | gzip built in; blosc2/zstd/lz4/bitshuffle via `hdf5plugin` — decoded transparently, none special-cased |
| `camera_timestamp_ns` / `host_time_ns` (len `T`) | optional | surfaced as per-frame metadata (timestamps) |
| same `(H, W, dtype)` across all parts | required | taken from the first part; mismatched parts are **not** validated and break downstream |

A part **without** a `frames` dataset is ignored. Filenames (the `even`/`odd`/range tokens) and the `.h5.yml` sidecar are **not** read — ordering comes only from the in-file `source_index`, so parts may be renamed freely.

## Configuration

All settings live in a single `PipelineConfig` dataclass (see [`src/asvimg/config.py`](src/asvimg/config.py)). The YAML keys map 1:1 to the field names, and are split into `db.yaml` (reproducibility) + `ops.yaml` (processing) when saved.

A key you leave out takes the dataclass default; the tables below name a default only where the behaviour hinges on it. The authoritative list (the bundled presets are *overlays* and do not carry every key):

```bash
uv run python -c "import dataclasses as d; from asvimg import PipelineConfig as C; c=C(); [print(f'{f.name}: {getattr(c, f.name)!r}') for f in d.fields(C)]"
```

### Data (`db.yaml`)

| key | meaning |
| --- | --- |
| `input_dir` | Directory holding the `.tif` / `.tiff` / `.dcimg` / `.sifx` inputs (or an even/odd `.h5` recording folder). The packaged default points at a sample folder that is **not** in the repository — set this |
| `input_format` | Which input format to read: `"auto"` (default — detect the one format present; **error** if the folder mixes formats) / `"tif"` (.tif/.tiff) / `"dcimg"` / `"sifx"` / `"h5"` (Ito even/odd folder). Set it explicitly to disambiguate a mixed folder |
| `input_order` | Order the input files are read and concatenated in — see [Input read order](#input-read-order): `"natural"` (default) / `"name"` / `"mtime"` / `"ctime"` |
| `output_dir` | Where everything is written (`null` → `<input_dir>/asi/<output_format>`) |
| `exp_name` | Experiment name used in the output filenames (`null` → inferred from the first input filename; `timelapse..._MM` names auto-populate it). Note it does **not** rename `roiSignals_*` — see [Output file layout](#output-file-layout) |
| `dcimg_backend` | DCIMG reader: `"auto"` (default; official runtime, falling back to the pure-Python parser) / `"sdk"` / `"native"` |

`output_format` and `output_metadata_yaml` are the other two `db.yaml` fields — see [Outputs](#outputs).

### Channels

| key | meaning |
| --- | --- |
| `channels_name` | Cycle of channel names (e.g. `["GCaMP", "jRGECO", "GCaMP", "jRGECO"]`). Frame `i` → channel `i % len(channels_name)`. |
| `channels_prop` | Per-channel role — `"source"` or `"donner"` (`"s"` / `"d"` also accepted). Same-name source/donner pairs auto-pair into linear-subtraction dF/F. Multiple sources with the same name are averaged into one signal. A name with **no** donner still gets its own `dff_{name}.npy` from preprocess, computed against a per-pixel `baseline_percentile` baseline (default = 5th percentile, skipping `start_initial_frames`) instead of the donner regression — so every group has a dF/F that PCA/ICA, ROI and export can read. |
| `ch_for_annotation` | Channel index (0-based into the cycle) selecting the **group** whose PCA/ICA maps cpselect offers. It does not restrict the decomposition: PCA/ICA run on *every* `channels_name` group that has a source (one IC picker window per group). |
| `demux_qc` | Intensity-based demux QC (default on). Verifies the *positional* channel cycle (`frame i → channels_name[i % cycle_len]`) against the per-frame mean-intensity fingerprint and flags a **phase slip** (a dropped frame that silently shifts every later channel — the failure the cross-file cycle-multiple warning cannot see *inside* a file). Detection only — it never changes the demux. Works for any `cycle_len ≥ 2`; when several cycled channels share the same brightness (e.g. `GCaMP,jRGECO,GCaMP,jRGECO` reads as a period-2 pattern) the phase is only resolvable modulo that sub-period — a slip by a multiple of it is undetectable from intensity, and the QC says so. Emits a `demux_qc` figure and `demux_*` fields in the preprocess stats; reports `inconclusive` (no false alarm) when the channels are not intensity-separable at all. |
| `demux_start_offset` | Global demux phase rotation (`0..cycle_len-1`): frame 0 is assigned `channels_name[demux_start_offset]`. The simple "fix the starting phase" knob. A `demux_correction.json` written by the demux editor (`asovi-demux`) **overrides** this when present and can additionally encode mid-recording phase-slip edits. |
| `channels_slip` | Per-**input-file** phase, one entry per file in read order (blank = none). `[0, 1, 0]` says the 2nd file's channel cycle starts one step in — its first frame is `channels_name[1]` — which is what a few dropped frames at the previous file's tail look like. Composes with `demux_start_offset`; a demux-editor sidecar still overrides both. Use **Quick Preview (All)** to see each file's head and confirm it. |

Worked examples — the pair must describe the acquisition exactly:

| Acquisition | `channels_name` | `channels_prop` |
| --- | --- | --- |
| Alternating blue / violet (the classic WFCI ratiometric pair) | `["BL", "BL"]` | `["source", "donner"]` |
| Single channel, no hemodynamic reference | `["BL"]` | `["source"]` |
| 4-colour excitation | `["GCaMP", "jRGECO", "GCaMP", "jRGECO"]` | `["donner", "source", "source", "source"]` |

### Preprocessing

| key | meaning |
| --- | --- |
| `do_registration` | DFT motion correction |
| `registration_cache` | `"force"` (always run) / `"cached"` (skip preprocess and reuse the existing `reg_Ch*.npy`). `"cached"` only accepts a **completed** run: the `reg_Ch` memmaps are preallocated at the start, so the test is `reg_meta.npz` (written last, and marked `partial` when the run was cancelled) — a cancelled or crashed run is re-registered instead of being analysed as a prefix. A cached run skips the rest of preprocess too: `save_raw_each_ch` / `save_registered_each_ch` are ignored (per-frame TIFFs are only written while registration runs), and a changed dF/F knob has no effect until you go back to `"force"` |
| `registration_batch_size` | Frames per registration batch (read → register → bin → write); bounds the registration working set. suite2p-style batching |
| `usfac` | DFT registration upsampling factor — subpixel precision `1/usfac` px (must be ≥ 1) |
| `template` | Path to an existing registration template (`.mat` / `.npy` / `.tif` / `.png` …; a relative path resolves against `input_dir`). It must have the **raw** (pre-binning) frame shape, and it is mirrored with the frames when `flip` is on. `null` → auto-build one and cache it as `<output_dir>/templateImage.mat` |
| `template_stride` / `template_ch` | Auto-template only: average channel `template_ch` (0-based in the cycle) taken from the **last** input file, one frame every `template_stride` **cycles** (= `template_stride × len(channels_name)` raw frames) |
| `linear_subt` | WFCI linear subtraction (dF/F). **This flag is what writes `dff_{name}.npy`** — by donner regression for a paired group, against the `baseline_percentile` baseline for a donner-less one. With `false` preprocess writes no dF/F file at all: a **donner-backed** group then has no PCA/ICA source and the PCA stage aborts, while its ROI dF/F, `dfWarped` and movies are skipped. A donner-less group still works (the read seam recomputes the percentile dF/F in RAM). Turn it off only for a registration-only run |
| `detrend` | Exponential (photobleaching) detrend of source & donner **before** the WFCI regression (ports MATLAB `flag_ExpoSub`; per-pixel log-linear `a·exp(b·t)` fit, subtracted). Off by default |
| `baseline_percentile_highpass` | Rolling-percentile ratiometric **high-pass** of source & donner before the regression (David Whitney `baselinePercentileFilter`; ports MATLAB `flag_BaselineFilter`). Independent of `detrend`; when both on, detrend runs first. Off by default |
| `baseline_percentile_highpass_sec` | Rolling-percentile window (seconds) for the high-pass baseline (default 120) |
| `baseline_percentile_highpass_rank` | Percentile rank (0–100) for the rolling high-pass baseline (default 50 = median) |
| `baseline_percentile` | Per-pixel percentile for the (static) dF/F baseline |
| `hemovar_qc` | Per-group **hemo variance-explained (R²) map** (default on): the fraction of source variance the donner regression removes per pixel (WidefieldImager `hemoVar` analogue). Saves `hemovar_{name}.npy` + a QC figure. **Donner-backed groups only** — a group with no donner has no regression to score |
| `binning` | Spatial binning factor (e.g. `2` → 2×2; `0` = none) |
| `flip` | Left/right mirror, applied per frame on read — and to the registration template, so the two stay in one coordinate system |
| `delete` | Delete previous outputs before preprocess: `reg_Ch*.npy`, `dff_*.npy`, `hemovar_*.npy`, `reg_meta.npz`, the cached auto-template `templateImage.mat` (so a changed `template_ch` / `template_stride` takes effect), this experiment's TIFFs, and the derived `ica/`, `pca_images/`, `ica_images/`. The human decisions — `marks.mat`, `ica_exclusion.json`, `rois.csv`, `demux_correction.json` — are **never** deleted, so **Run All** cannot destroy them in preprocess before the stage that replays them |
| `fps` | Camera fps (across all channels, not per-channel; must be > 0) |
| `use_mmap` | dF/F read strategy (output identical either way): `false` = load each registered channel fully into RAM (fastest; short recordings / big machines); `true` = read `reg_Ch` via memmap and stream dF/F in row-strips (bounded peak memory for long recordings) |
| `start_initial_frames` | Initial frames skipped from the regression FOI, the baseline, the ROI plots **and the ROI correlations** (`null` → ~4 s auto). These frames are unstable illumination — a transient every pixel shares, which otherwise reads as brain-wide correlation |
| `ignore_last_frames` | Trailing frames excluded from the regression FOI, the ROI plots **and the ROI correlations** (`0` = keep all) — the same window as `start_initial_frames` |
| `max_frames` | Frame cap (`null` = all frames) |
| `filter_xyt` | `[x, y, t]` 3D filter window applied over the **full registered timeline** (no batch-boundary seams); every downstream artifact inherits it. `null` disables. Must be exactly 3 positive ints — a 2-element window is rejected when the config is built, not hours later inside preprocess |
| `filter_xyt_kind` | Kernel for `filter_xyt`: `"mean"` (box/uniform, the default) / `"median"` / `"gaussian"` (the window maps to `sigma = (size − 1) / 2`) |

> Older YAMLs using `flag_linear_subt` / `flag_binning` / `flag_flip` / `flag_delete` / `ignore_initial_frames` still load (auto-renamed with a warning). `batch_size` (→ `registration_batch_size`) and `pca_n_skip` (→ `pca_skip_frames`) are also auto-renamed on load.

### Downstream analysis (PCA/ICA, annotation, ROI, correlation)

| key | meaning |
| --- | --- |
| `annotation` | `"cache"` (load saved `marks.mat`) / `"gui"` (cpselect) / `(src_pts, ref_pts)` / `false` (skip) |
| `annotation_allow_reflection` | Allow a mirror (left-right flip) in the atlas transform. Needs non-midline control points to have any effect |
| `ch_for_annotation` | Channel index (0-based) whose **group** provides the maps cpselect offers. PCA/ICA themselves run on *every* group with a source — one fluorophore's components say nothing about another's |
| `annotation_atlas_path` | Atlas file. **Empty (the default) = the atlas bundled with the package** (`asvimg/data/wfciAnnotationData.mat`), so an installed copy works with no setup; otherwise a path to a `.mat` / `.h5`, e.g. one built by `asovi-atlas`. Keep it empty (not an absolute path) unless you mean to override — the GUI folds this value into the annotation stage signature, so a machine-specific path marks the stage stale on another machine |
| `pca_n_components`, `pca_smooth_sigma` | PCA component count, and the Gaussian sigma (frames) used to smooth the **plotted** temporal traces. The smoothing is display-only: the decomposition, and therefore the IC numbering, does not depend on it |
| `pca_skip_frames` | Temporal stride for the PCA/ICA fitting input (memory/speed; `1` = every frame). Fit-only — the ICA basis is purely spatial, so it is applied to the full-length timeline |
| `ica_n_components`, `ica_max_iter`, `ica_random_state` | FastICA parameters |
| `skip_ica` | Skip the ICA stage entirely (it becomes a no-op). Cannot be combined with `ica_denoise: "subtract"` — there would be no basis to subtract with, and the config refuses to load |
| `ica_exclusion` | Which ICs are artifacts: `"cache"` (default — replay the choice recorded in `ica_exclusion.json`) / `"gui"` (pick them; one window per channel group) / `{"GCaMP": [2, 7]}` (0-based, inline) / `false` (none). Mirrors `annotation`: an interactive choice is written to `ica_exclusion.json`, so a headless re-run reproduces it with no window |
| `ica_denoise` | `"off"` (default) — the IC exclusion is **QC only**; every saved artifact comes from the plain dF/F (MATLAB / notebook parity). `"subtract"` — ROI signals, `dfWarped`, movies and correlation are computed from dF/F **minus the excluded components** (`ica/dff_{name}.npy`). Excluding nothing is the exact identity, so turning it on cannot by itself change a value. Every `roiSignals_*` and `*_dfWarped_*` payload carries `ica_denoised` / `ica_denoise_mode` / `ica_excluded_ic` regardless of the setting; `ica_excluded_ic` is **1-based** (matching `IC{i}.png`, the picker and `ica_exclusion.json`), unlike the inline `ica_exclusion` dict above, which is 0-based |
| `roi_signal` | `"Both"` / `"Raw"` / `"dff"` — which signal(s) to extract per ROI. `F_raw` never passes through the denoising: it is read straight from the registered channels |
| `roi_space` | `"atlas"` (default) — ROIs are Allen-atlas coordinates (the atlas defaults, or an edited `rois.csv`), pulled back into the recording by the annotation transform, so the ROI stage needs annotation done. `"source"` — ROIs are already in the recording's own (binned reg/dff) coordinates, read directly with **no atlas registration**: ROI signals and correlation then run even with `annotation: false`. Source ROIs come from `rois_source.csv` (there are no built-in defaults for a raw recording — an empty/absent file means the ROI stage extracts nothing) |
| `corr_method` | ROI correlation: `"raw"` (Pearson) / `"gsr"` (global-signal regressed) / `"partial"` (shrinkage precision — note it is computed in the ROIs' raw units, so it is not scale-invariant). All three use the `start_initial_frames` / `ignore_last_frames` window, the same one the ROI plots show |
| `corr_auto_threshold`, `corr_edge_density`, `corr_network_threshold` | Network edges: auto (keep top `edge_density` fraction) vs a fixed `|r|` cutoff |
| `save_movie_speed`, `save_movie_codec`, `save_movie_vminmax`, `save_movie_cmap`, `save_movie_merge_chs` | Warped dF/F movie display: realtime ×N (output fps = per-channel fps × speed); codec `"MJPG"` (.avi) / `"DIB (RAW)"` (uncompressed .avi) / `"mp4V"` (.mp4) — a raw 4-char FourCC is also accepted; `(vmin, vmax)` in dF/F %; the LUT `"magma"` (default) / `"turbo"` / `"gray"` / `"viridis"`; and `save_movie_merge_chs` — when >1 source group, concatenate their movies horizontally into a single `{exp}_merged_dF` (off = one per group) |

### Post-warp processing (applies to every atlas-warped export)

These run on the warped stack, so they shape the annotated TIFFs, `{exp_name}_dfWarped_*` **and** the movies — but **not** `dff_{name}.npy` or the ROI signals (which are extracted via the warp adjoint from the *source*, and never build a warped stack).

| key | meaning |
| --- | --- |
| `post_annotation_time_average` | Half-window `N` for a centered temporal moving average after the warp (`0` = off; `1` = ±1 frame = 3-frame mean). Recorded in the exported payload |
| `post_annotation_filter_xyt` | `[x, y, t]` 3D filter window applied to the atlas-warped stack (`null` = off). Distinct from `filter_xyt`, which runs pre-warp on the registered timeline. Must be exactly 3 positive ints |
| `post_annotation_filter_kind` | Kernel for `post_annotation_filter_xyt`: `"mean"` (default) / `"median"` / `"gaussian"` |

> Either post-annotation step needs every frame of a pixel at once, so enabling one makes the export hold the warped atlas stack in RAM (~2 GB at T=6000) instead of streaming it from disk. Both are off by default.

### Outputs

In the GUI these are all under the **Outputs** pane (Frames / Movie / ROI / Figures / Configs).

| key | meaning |
| --- | --- |
| `output_format` | `mat` / `npy` / `h5` |
| `save_raw_each_ch` | Save per-channel pre-registration TIFF stacks (input dtype) |
| `save_registered_each_ch` | Save per-channel registered + binned TIFFs (uint16) |
| `save_annotated_each_ch` | Save per-channel atlas-warped TIFFs (uint16) |
| `save_annotated_dF_mat` | Save atlas-warped dF/F per group as `{exp}_dfWarped_{name}_01` (mat/npy/h5). The warped payload is **float32** (`single` in MATLAB) — the sources it comes from are uint16 / float32, so float64 only doubled the file |
| `save_annotated_dF_dtype` | dtype of that dfWarped payload: `"float32"` (default) / `"float16"`. float16 halves the npy/h5 file size; `.mat` cannot hold float16 (scipy upcasts it to float64), so a mat export stays float32 and logs a warning |
| `export_orientation` | Axis order of the final matrix exports (dfWarped): `"HWT"` (H,W,T; MATLAB/legacy parity) / `"THW"` (T,H,W). Internal `reg_Ch`/`dff` `.npy` are always (T,H,W) |
| `save_roi_signals` | Export ROI time series as a DataFrame: `null` / `"csv"` (default) / `"pickle"` / `"both"` |
| `save_movie` | Write `{exp}_{name}_dF.avi` |
| `save_figures` | Save emitted QC figures to `<output>/figures/`: `"none"` / `"png"` / `"png+pdf"` (editable PDF). **Applies to the GUI and the notebook only** — `asovi-run` writes `figures/` regardless, unless you pass `--no-figures` |
| `output_metadata_yaml` | Write `input_metadata.yaml` (per-frame channel/timestamp metadata) |
| `tiff_format` | `"big-tiff"` / `"ome-tiff"` |
| `tiff_compression` | zlib compression for TIFFs |

## Output file layout

All files land under `<output_dir>` (default `<input>/asi/<format>`). `{ext}` is
`output_format` (`mat` / `npy` / `h5`) and applies to the two payload exports —
`{exp_name}_dfWarped_*` and `roiSignals_*`. The intermediate `reg_Ch` / `dff`
files are always `.npy` memmaps regardless of `output_format`. The `[flag]` note
marks files written only when that config flag is on.

```text
<output_dir>/                              # e.g. asi/npy
  reg_Ch{i}.npy                            # preprocess — registered+binned channel i, (T,H,W) uint16 memmap
  dff_{name}.npy                           # preprocess — dF/F per channels_name group, (T,H,W) float32  [linear_subt]
                                           #   With a donner: global (whole-recording) linear subtraction.
                                           #   Without one: a percentile-baseline dF/F, so every group still
                                           #   has a source PCA/ICA can analyse. linear_subt=false writes
                                           #   NEITHER: a donner-backed group then has no dF/F at all
  hemovar_{name}.npy                       # preprocess — hemo variance-explained (R²) map (H,W) float32,
                                           #   donner-backed groups only                    [hemovar_qc]
  reg_meta.npz                             # preprocess — proc_template, meanImageCh{i}, channels, T_per_ch,
                                           #   fps, and `partial` (1 = the run was CANCELLED; the stacks are
                                           #   a valid PREFIX of the recording, not the whole of it). Written
                                           #   LAST: its presence is what "preprocess finished" means, and
                                           #   T_per_ch — not the file length — is the authoritative count
  templateImage.mat                        # preprocess — the AUTO-BUILT registration template (proc_template),
                                           #   cached and reused by later runs; removed by `delete` so a
                                           #   template_ch/template_stride edit takes effect  [template unset]
  {exp_name}_dfWarped_{name}_01.{ext}      # export — dF/F warped to Allen atlas (POST-warp),
                                           #   orientation = export_orientation           [save_annotated_dF_mat]
  roiSignals_{name}_{stem}.{ext}           # roi — per-ROI time series: F_raw, F_dff, roi_names, n_pixels, fps,
                                           #   roi_signal_mode, ica_denoised, ica_denoise_mode, ica_excluded_ic
  roiSignals_{name}_{stem}_{raw,dff}.csv   # roi — same as a DataFrame        [save_roi_signals: csv|both]
  roiSignals_{name}_{stem}_{raw,dff}.pkl   # roi — same, pickled DataFrame    [save_roi_signals: pickle|both]
  rois.csv                                 # ROI editor (GUI) — the edited atlas point-ROI set (name,x,y,size).
                                           #   When present it REPLACES the atlas defaults for ROI extraction,
                                           #   correlation and the seed maps                 [roi_space: atlas]
  rois_source.csv                          # ROIs in the recording's own (binned reg/dff) coords (name,x,y,size).
                                           #   Used when roi_space="source" — ROI signals with NO atlas warp
                                           #   (no built-in defaults; absent => nothing extracted)  [roi_space: source]
  ica/{name}/basis.npz                     # ica — the spatial basis + the source fingerprint it was fit on
  ica/dff_{name}.npy                       # ica — dF/F minus the excluded ICs, (T,H,W) float32   [ica_denoise]
  ica/dff_{name}.json                      # ica — what that stack was built from (exclusion, basis id, source
                                           #   fingerprint, params). Missing or mismatched → stack is rebuilt
  ica_exclusion.json                       # ica — which ICs a human flagged, per group ("IC3"; 1-based)
  marks.mat                                # annotation — cpselect control points (src/ref)
  nwb_metadata.yaml                        # NWB builder (asovi-nwb) — Subject/session/device/per-channel optics
                                           #   metadata NWB/DANDI need but the pipeline never captured. Human-only
                                           #   sidecar (like marks.mat/rois.csv); the GUI is its sole writer
  {exp_name}.nwb                           # NWB builder (asovi-nwb) — the packaged NWB file (dF/F, ROI signals +
                                           #   segmentation, atlas-warped dF/F, reference images, ICA, correlation;
                                           #   payloads chosen in the GUI). Validated with `nwbinspector --config dandi`
  db.yaml, ops.yaml                        # effective config (reproducibility)
  .stage_runs.json                         # GUI — the config signature each stage last ran with (staleness)
  input_metadata.yaml                      # per-frame channel/timestamp metadata          [output_metadata_yaml]
  fig_template.png, figFrames.png          # preprocess QC previews
  fig_demux_qc.png                         # preprocess — demux (channel-cycle) phase QC   [demux_qc, no reporter]
  fig_hemovar_{name}.png                   # preprocess — hemo R² QC map                   [hemovar_qc, no reporter]
  demux_means.npz                          # preprocess — per-frame mean intensity (feeds the demux editor)
  demux_correction.json                    # demux editor — phase correction preprocess applies  [asovi-demux]
  demux_corrected/{stem}_corrCh{i}_{name}.tif  # demux editor — raw frames de-interleaved by the correction
                                           #   [Export corrected TIFF]
  figures/{stage}_{key}.png[/.pdf]         # emitted QC figures. asovi-run writes these BY DEFAULT (suppress
                                           #   with --no-figures); the GUI / notebook only when save_figures
                                           #   is png / png+pdf (.pdf only for png+pdf)
  pca_images/{name}/PC{i}.png              # pca — spatial maps per group, 1-based (ch_for_annotation's group
                                           #   is the one cpselect offers)
  ica_images/{name}/IC{i}.png              # ica — spatial maps per group, 1-based (the IC picker shows these)
  corrMap/{group}_{seed}[_gsr].npy/.png    # seed correlation maps (Seed-based Corr. Maps panel)
  map_for_annot/*.png,*.npy,*.tif          # custom cpselect maps you drop in (+ corr_*.npy from Add-for-Annot)
  tiffs/{raw,reg,annot}_Ch{i}_{name}-{prop}/{exp_name}_{raw,reg,annot}_Ch{i}_{name}-{prop}.tif
                                           #   per-channel TIFF stacks                      [save_*_each_ch]
  movies/{exp_name}_{name}_dF.avi          # warped dF/F movies                            [save_movie]
```

Internally the `reg_Ch`/`dff` arrays are stored `(T, H, W)` (contiguous
per-frame I/O, suite2p-style); the `io` loaders present them as `(H, W, T)` to
downstream code via a zero-copy transpose, so existing consumers are unchanged.

**`{exp_name}` vs `{stem}`** — `{exp_name}` is the `exp_name` config field
(`"output"` when unset). `{stem}` is the stem of the **first input file**, and it
**ignores `exp_name`** whenever the input directory still holds readable
recordings (`io.resolve_exp_stem`). So `--exp-name` renames `*_dfWarped_*`,
`tiffs/` and `movies/`, but **not** `roiSignals_*` or `demux_corrected/`. To keep
several animals' ROI files apart, give each its own `output_dir`.

**Pre- vs post-warp dF/F** — `dff_{name}.npy` is in the individual's imaging
coordinates (that recording only); `{exp_name}_dfWarped_{name}` is the same signal
resampled into the common Allen CCF space for cross-animal / atlas-region
analysis. ROI signals are in atlas space either way, but the ROI stage **does not
warp**: warping is linear and an ROI mean is linear, so the atlas ROI masks are
pulled back into source space once (`annotation.warp_weight_maps`, the exact
adjoint of the warp) and every frame's ROI signals fall out of one GEMM against
the source. Identical numbers to warping first (to ~1e-14), without ever building
the warped stack.

**Which stage writes what** — the GUI marks a stage as done when its artifact is
already on disk, so loading an `ops.yaml` from a processed `asi/` folder shows
what has run (`runner.stage_artifacts`):

| Stage | Detected by |
| --- | --- |
| preprocess | `reg_meta.npz` — **not** `reg_Ch*.npy`, which are *preallocated* at the start of the run and so only mean it started; `reg_meta.npz` is written last and means it finished |
| pca | `pca_images/**/PC*.png` |
| ica | `ica_images/**/IC*.png` (not `ica/{name}/basis.npz`) |
| annotation | `marks.mat` |
| roi | `roiSignals_*` |
| correlation | `figures/correlation_*` — the one stage that writes no data file, so it is detected from its saved figure. After an `asovi-run` that is always present; in the GUI under the default `save_figures: "none"` nothing lands on disk, so a reloaded folder reads correlation as not-run no matter how often it ran. Set `save_figures: "png"` if you want the GUI to remember it |
| export | `*dfWarped_*` / `tiffs/` / `movies/` |

**Reproducibility — the decisions a human makes** — everything in `<output_dir>`
is a function of the inputs and `ops.yaml`, *except* what a human decided by hand.
Those decisions cannot be re-derived, so they are the files a re-run must replay,
and none of them is removed by `delete`:

| File | The decision | Replayed by |
| --- | --- | --- |
| `marks.mat` | where the atlas control points go (cpselect) | `annotation: "cache"` (default) |
| `ica_exclusion.json` | which components are artifacts (the IC picker) | `ica_exclusion: "cache"` (default) |
| `rois.csv` | the edited atlas ROI set (the ROI editor) — **replaces** the atlas defaults for ROI extraction, correlation and the seed maps | its mere presence in `<output_dir>` |
| `rois_source.csv` | source-space ROIs (`roi_space: "source"`) — the ROI set for a recording processed without atlas registration | its mere presence in `<output_dir>` |
| `demux_correction.json` | the channel-cycle phase fix (`asovi-demux`) — **overrides** `demux_start_offset` / `channels_slip` | its mere presence in `<output_dir>` |
| `nwb_metadata.yaml` | the Subject / session / device / per-channel metadata for NWB export (`asovi-nwb`) — none of it exists elsewhere in the pipeline | re-loaded by the NWB builder from `<output_dir>` |

Only the first two exist unless you opened the ROI or demux editor. So: review one
animal in the GUI, keep those files, and `asovi-run` reproduces that session with
no windows at all — bit for bit, and for the rest of the batch.
`tests/test_headless_repro.py` holds that contract for `marks.mat`
+ `ica_exclusion.json`.

## Notebook workflow

[`run_pipeline_full.ipynb`](notebooks/run_pipeline_full.ipynb) is
a third front-end onto the same `runner.PipelineSession` — one cell per stage, so
you can stop, look at a figure, and carry on:

```python
from asvimg.gui.bridge import GuiBridge
from asvimg.runner import PipelineSession, stage_artifacts

bridge = GuiBridge()          # runs cpselect / the IC picker in a SEPARATE process
session = PipelineSession(
    config, reporter=NotebookReporter(),   # defined in the notebook
    marks_provider=bridge.marks_provider,
    ica_provider=bridge.ica_provider,
    cancel=bridge.cancel,
)

stats      = session.run_preprocess()    # Stage 1
pca_result = session.run_pca()           # Stage 2
excluded   = session.run_ica()           # Stage 3  (opens the IC picker)
tform      = session.run_annotation()    # Stage 4  (opens cpselect)
roi_out    = session.run_roi()           # Stage 5
corr_out   = session.run_correlation()   # Stage 6
             session.run_export()        # Stage 7
# ...or session.run_all() for the lot; stage_artifacts(out) shows what is on disk.
```

`GuiBridge` is what makes the interactive stages safe here: a Dear PyGui context
is one-per-process, so it launches them as subprocesses. With `annotation: "cache"` /
`ica_exclusion: "cache"` no window opens and the saved `marks.mat` /
`ica_exclusion.json` are replayed. ROI editing is *not* available in the notebook
(no dpg context): edit ROIs in the dashboard, or edit `rois.csv` by hand, then
re-run from Stage 5.

## DCIMG metadata helper

```python
from pathlib import Path
from asvimg import load_dcimg_metadata

meta = load_dcimg_metadata(
    Path("/path/to/your/recording.dcimg"),
    frame_indices=[0, 100],
)
print(meta["summary"])
print(meta["frames"])
```

## Benchmark

```bash
uv run python tests/benchmark_validate_pipeline01.py \
    --input-file /path/to/your/recording.ome.tif
```

Writes `benchmark_report.json` beside the input (`<input>/asi/`). Defaults to the first 500 frames; override with `--max-frames`.

## Inspiration & references

This pipeline is a Python re-implementation of an in-house MATLAB WFCI workflow
and is **strongly inspired by [suite2p](https://github.com/MouseLand/suite2p)
(MouseLand)**. In particular, preprocessing adopts suite2p's batched
registration, a per-channel contiguous memmap intermediate (`reg_Ch{i}.npy`,
`(T,H,W)` uint16), and batched-FFT reference matching. The head-to-head
comparison of the DFT implementations and the integration notes are in the
commit history (`git log -- src/asvimg/registration`).

The WFCI-specific design — interleaved blue/violet (source/donner) acquisition,
per-pixel hemodynamic-style linear subtraction, and the intensity-fingerprint
demux QC — follows the Churchland lab
**[WidefieldImager](https://github.com/churchlandlab/WidefieldImager)** (Musall,
Kaufman et al. 2019).

> **`resources/` is not part of this repository** — it is gitignored. The
> read-only reference checkouts of suite2p 0.14.6 and WidefieldImager, and the
> side-by-side comparison write-up, live there on the maintainer's machine only.
> Fetch them yourself if you want to follow the head-to-head sections; nothing in
> the pipeline imports from `resources/`.

### Algorithms & sources, by stage

**Registration (motion correction)**
- suite2p — Pachitariu M, Stringer C, Dipoppa M, Schröder S, Rossi LF, Dalgleish H, Carandini M, Harris KD. *Suite2p: beyond 10,000 neurons with standard two-photon microscopy.* bioRxiv (2017). doi:[10.1101/061507](https://doi.org/10.1101/061507) · [github.com/MouseLand/suite2p](https://github.com/MouseLand/suite2p)
- Subpixel DFT registration — Guizar-Sicairos M, Thurman ST, Fienup JR. *Efficient subpixel image registration algorithms.* Optics Letters 33(2):156–158 (2008). doi:[10.1364/OL.33.000156](https://doi.org/10.1364/OL.33.000156)

**dF/F (WFCI linear subtraction)**
- Per-pixel linear regression of the reference (“donner”) channel out of the source channel, followed by dF/F normalization (ratiometric hemodynamic-style correction), ported from the original MATLAB pipeline.
- WidefieldImager (Churchland lab) — Musall S, Kaufman MT, Juavinett AL, Gluf S, Churchland AK. *Single-trial neural dynamics are dominated by richly varied movements.* Nature Neuroscience 22:1677–1686 (2019). doi:[10.1038/s41593-019-0502-4](https://doi.org/10.1038/s41593-019-0502-4) · [github.com/churchlandlab/WidefieldImager](https://github.com/churchlandlab/WidefieldImager)

**Demux QC & correction (channel-cycle phase)**
- Intensity-fingerprint verification + correction of the positional channel demux (`asvimg.demux_qc`, `asovi-demux`), adapted from WidefieldImager's evidence-based blue/violet phase recovery (LED-trigger + dark-frame) to ASoVi's trigger-free N-channel model.

**PCA / ICA denoising**
- FastICA — Hyvärinen A, Oja E. *Independent component analysis: algorithms and applications.* Neural Networks 13(4–5):411–430 (2000). doi:[10.1016/S0893-6080(00)00026-5](https://doi.org/10.1016/S0893-6080(00)00026-5)
- scikit-learn (PCA, FastICA, covariance) — Pedregosa F, et al. *Scikit-learn: Machine Learning in Python.* JMLR 12:2825–2830 (2011). [scikit-learn.org](https://scikit-learn.org)

**Atlas registration**
- Allen Mouse Brain Common Coordinate Framework (CCFv3) — Wang Q, et al. *The Allen Mouse Brain Common Coordinate Framework: A 3D Reference Atlas.* Cell 181(4):936–953 (2020). doi:[10.1016/j.cell.2020.04.007](https://doi.org/10.1016/j.cell.2020.04.007) · [atlas.brain-map.org](https://atlas.brain-map.org/)
- scikit-image (similarity transform; the reference bilinear `warp`) — van der Walt S, et al. *scikit-image: image processing in Python.* PeerJ 2:e453 (2014). doi:[10.7717/peerj.453](https://doi.org/10.7717/peerj.453). The stack warp itself runs on torch `grid_sample` (float64), which reproduces that kernel to ~1e-13 while batching whole time-chunks; skimage stays as the reference and the fallback.
- Control-point selection UI inspired by MATLAB's Control Point Selection Tool (`cpselect`).

**ROI correlation**
- Partial correlation via Ledoit-Wolf shrinkage — Ledoit O, Wolf M. *A well-conditioned estimator for large-dimensional covariance matrices.* J. Multivariate Analysis 88(2):365–411 (2004). doi:[10.1016/S0047-259X(03)00096-4](https://doi.org/10.1016/S0047-259X(03)00096-4) (via scikit-learn `LedoitWolf`).

**Core libraries**
- PyTorch (batched CPU FFT) — Paszke A, et al. *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS (2019). [pytorch.org](https://pytorch.org)
- NumPy — Harris CR, et al. Nature 585:357–362 (2020) · SciPy — Virtanen P, et al. Nature Methods 17:261–272 (2020)
- Dear PyGui (dashboard/UI) — [github.com/hoffstadt/DearPyGui](https://github.com/hoffstadt/DearPyGui)

All third-party libraries are used under their respective licenses.

Two parts of this code are derived from work by others, and keep its terms: the
DFT registration in `src/asvimg/registration/` is a port of Manuel Guizar-Sicairos'
`dftregistration.m` (File Exchange 18401, BSD), and the rolling-percentile baseline
in `src/asvimg/wfci.py` follows David Whitney's `baselinePercentileFilter.m`
(MPFI). The bundled atlas is derived from the Allen CCFv3. Origins, terms and what
is derived from what are recorded in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
