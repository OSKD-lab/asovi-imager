"""ROI signal extraction — the atlas ROIs are pulled back into SOURCE space.

Warping is linear and an ROI mean is linear, so the pipeline never warps a stack
to get ROI signals: :func:`extract_signals_from_source` (what the ROI stage uses)
multiplies the source frames by the adjoint weight maps built by
``annotation.warp_weight_maps``.  :func:`extract_signals` stays for stacks that
are ALREADY in atlas space (seed maps) and as the reference the adjoint is
checked against (``tests/test_session.py::TestRoiAdjoint``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from skimage.transform import SimilarityTransform

    from .atlas import ACCFv3
    from .cancel import CancellationToken
    from .config import PipelineConfig
    from .events import ProgressReporter


_ROI_SIGNAL_MODES: dict[str, tuple[str, ...]] = {
    "Both": ("raw", "dff"),
    "Raw": ("raw",),
    "dff": ("dff",),
}


@dataclass
class ExtractionResult:
    """Result of ROI signal extraction.

    Attributes
    ----------
    F : (N, T) float64 — mean signal per ROI per frame.
    roi_names : list of ROI names.
    n_pixels : (N,) int — pixel count per ROI mask.
    masks : (N, H, W) bool — the masks used for extraction.
    """

    F: np.ndarray
    roi_names: list[str]
    n_pixels: np.ndarray
    masks: np.ndarray


def _resolve_masks(masks, names: list[str] | None, diameter: int):
    """(N, H, W) bool masks + names from an ndarray or an ``ACCFv3`` atlas."""
    from .atlas import ACCFv3

    if isinstance(masks, ACCFv3):
        mask_arr = masks.get_point_roi_mask(diameter=diameter)
        if names is None:
            names = list(masks.point_roi_names)
    elif isinstance(masks, np.ndarray):
        mask_arr = masks[np.newaxis] if masks.ndim == 2 else masks
    else:
        raise TypeError(f"masks must be ndarray or ACCFv3, got {type(masks)}")

    if names is None:
        names = [f"ROI_{i}" for i in range(mask_arr.shape[0])]
    return mask_arr, list(names)


def extract_signals(
    xyt: np.ndarray,
    masks,
    names: list[str] | None = None,
    *,
    diameter: int = 8,
    reporter: "ProgressReporter | None" = None,
    cancel: "CancellationToken | None" = None,
    stage=None,
    progress_label: str = "",
) -> ExtractionResult:
    """Extract mean time series from ROI masks of an ALREADY-WARPED stack.

    The pipeline itself no longer warps to get ROI signals — see
    :func:`extract_signals_from_source`, which reaches the same numbers without
    ever building the warped stack.  This stays for atlas-space stacks you
    already hold (and as the reference the adjoint is checked against).

    Parameters
    ----------
    xyt : (H, W, T) array — warped dF/F stack.
    masks : one of:
        - (H, W) bool — single ROI mask
        - (N, H, W) bool — multiple ROI masks
        - ``ACCFv3`` instance — uses ``get_point_roi_mask(diameter=diameter)``
    names : ROI names. Inferred from ACCFv3 or auto-generated if None.
    diameter : point ROI diameter (only used when masks is ACCFv3).
    reporter, cancel, stage, progress_label : when a ``reporter`` and ``stage``
        are given, ROIs are extracted one at a time so each ROI ticks the
        progress bar (message ``"{progress_label}: {roi}"``); otherwise the fast
        vectorised path is used.

    Returns
    -------
    ExtractionResult with F (N, T), names, pixel counts, and masks.
    """
    mask_arr, names = _resolve_masks(masks, names, diameter)
    n_rois = mask_arr.shape[0]

    h, w, t = xyt.shape
    n_pixels = mask_arr.sum(axis=(1, 2))  # (N,)

    # Extract: mean of masked pixels per frame
    # Reshape for vectorized extraction: (N, H*W) @ (H*W, T)
    flat_masks = mask_arr.reshape(n_rois, -1).astype(np.float64)  # (N, H*W)
    flat_xyt = xyt.reshape(-1, t).astype(np.float64)  # (H*W, T)

    # Normalize by pixel count (avoid div-by-zero)
    safe_n = np.maximum(n_pixels, 1).astype(np.float64)

    if reporter is not None and stage is not None:
        # Per-ROI so each one advances the progress bar; same result as the
        # vectorised path, just reported ROI by ROI.
        F = np.empty((n_rois, t), dtype=np.float64)
        for i in range(n_rois):
            if cancel is not None:
                cancel.raise_if_set()
            label = f"{progress_label}: {names[i]}" if progress_label else names[i]
            reporter.on_progress(stage, i + 1, n_rois, message=label)
            F[i] = (flat_masks[i] @ flat_xyt) / safe_n[i]
    else:
        F = (flat_masks @ flat_xyt) / safe_n[:, np.newaxis]  # (N, T)

    return ExtractionResult(
        F=F,
        roi_names=list(names),
        n_pixels=n_pixels.astype(int),
        masks=mask_arr,
    )


def extract_signals_from_source(
    read_block,
    n_frames: int,
    source_hw: tuple[int, int],
    masks,
    names: list[str] | None = None,
    *,
    transform: "SimilarityTransform | None",
    diameter: int = 8,
    chunk: int = 512,
    reporter: "ProgressReporter | None" = None,
    cancel: "CancellationToken | None" = None,
    stage=None,
    progress_label: str = "",
) -> ExtractionResult:
    """ROI signals straight out of the SOURCE stack — no warped stack, ever.

    Two ways the ROI masks reach source space:

    - ``transform`` given (atlas ROIs): warping is linear and an ROI mean is
      linear, so the two compose into one fixed weight map per ROI in source
      space (:func:`warp_weight_maps`).  Identical numbers to ``warp_stack`` →
      :func:`extract_signals` (verified to ~1e-14 relative), without ever
      materializing the (H_atlas, W_atlas, T) warped stack — 3.9 GB at T=6000.
    - ``transform=None`` (source ROIs): the masks are ALREADY in source space
      (``roi_space="source"``, no atlas registration), so they ARE the weight
      maps — a plain masked mean, no warp involved.  ``masks`` must then be shaped
      ``(N, *source_hw)``.

    Either way the signals are a GEMM against the source frames.  Only the source
    pixels that any ROI actually samples are touched (gathering those columns
    first keeps the float64 working set at ~25 MB per chunk instead of ~1.4 GB),
    so the stage is disk-bound.

    ``read_block(t0, t1) -> (k, H, W)`` supplies the source frames; it is called
    in ``chunk``-sized time slices so a memmap streams instead of loading whole.
    """
    mask_arr, names = _resolve_masks(masks, names, diameter)
    n_rois = mask_arr.shape[0]
    n_pixels = mask_arr.sum(axis=(1, 2))
    h, w = int(source_hw[0]), int(source_hw[1])

    if transform is None:
        # Source-space ROIs: the mask IS the weight map, no warp.
        if mask_arr.shape[1:] != (h, w):
            raise ValueError(
                f"roi_space='source' needs masks in source shape {(h, w)}, "
                f"got {mask_arr.shape[1:]}"
            )
        weights = mask_arr.reshape(n_rois, h * w).astype(np.float64)
    else:
        from .annotation import warp_weight_maps

        weights = warp_weight_maps(mask_arr, transform, (h, w))  # (N, H*W)
    weights /= np.maximum(n_pixels, 1).astype(np.float64)[:, np.newaxis]
    # Columns no ROI samples contribute nothing; dropping them is exact.
    cols = np.flatnonzero(weights.any(axis=0))
    weights = np.ascontiguousarray(weights[:, cols])  # (N, n_cols)

    F = np.zeros((n_rois, int(n_frames)), dtype=np.float64)
    if cols.size == 0:  # every ROI maps outside the source FOV
        return ExtractionResult(
            F=F, roi_names=list(names), n_pixels=n_pixels.astype(int), masks=mask_arr,
        )

    for t0 in range(0, int(n_frames), max(1, int(chunk))):
        if cancel is not None:
            cancel.raise_if_set()
        t1 = min(t0 + max(1, int(chunk)), int(n_frames))
        # Gather in the source's own dtype, THEN widen: casting the whole block
        # to float64 first would allocate ~50x more than the ROIs ever read.
        flat = np.asarray(read_block(t0, t1)).reshape(t1 - t0, h * w)
        block = np.asarray(flat[:, cols], dtype=np.float64)  # (k, n_cols)
        np.nan_to_num(block, copy=False)  # warp_stack zeroed NaNs; match it
        F[:, t0:t1] = weights @ block.T
        if reporter is not None and stage is not None:
            label = f"{progress_label}: ROI signals" if progress_label else "ROI signals"
            reporter.on_progress(stage, t1, int(n_frames), message=label)

    return ExtractionResult(
        F=F,
        roi_names=list(names),
        n_pixels=n_pixels.astype(int),
        masks=mask_arr,
    )


def extract_and_save_roi_signals(
    config: "PipelineConfig",
    output_dir: Path,
    tform: "SimilarityTransform | None",
    atlas: "ACCFv3 | None",
    *,
    diameter: int = 8,
    reporter: "ProgressReporter | None" = None,
    cancel: "CancellationToken | None" = None,
) -> dict[str, dict[str, Any]]:
    """Extract ROI time-series for every ``channels_name`` group and save them.

    Port of ``run_pipeline_full.ipynb``'s ROI-extraction cell.  For each group:

    - ``"raw"`` mode → average the source channels (uint16-clipped), take the
      atlas point-ROI means → ``F_raw``.
    - ``"dff"`` mode → the group's dF/F stack (linear-subt when a donner exists,
      else p-th percentile baseline), same ROIs → ``F_dff``.

    Neither warps the data: the atlas ROI masks are pulled back into source space
    once (:func:`extract_signals_from_source`), which is exact and skips building
    a warped stack (3.9 GB in float64 at T=6000, on the 285x285 atlas) — and skips
    the 16.6 GB float64 copy of the 540x640 source that feeding one would need.
    The source itself is streamed from the reg/dff memmaps in time-chunks, so peak
    RAM no longer grows with T.

    Which modes run is controlled by ``config.roi_signal`` (``"Both"`` /
    ``"Raw"`` / ``"dff"``).  Results are written to
    ``roiSignals_{name}_{exp_stem}.{output_format}``, and — when
    ``config.save_roi_signals`` is set — also as a per-mode CSV/pickle
    DataFrame (rows = time [s], cols = ROI).

    Returns
    -------
    dict mapping ``channels_name`` → the saved payload dict (with ``F_raw`` /
    ``F_dff`` / ``roi_names`` / ``n_pixels`` / ``fps`` / ``roi_signal_mode``).
    """
    from .annotation import _load_name_dff_stack
    from .events import StageId
    from .ica_state import resolve_exclusion
    from .io import (
        load_reg_channel,
        reg_channel_path,
        resolve_exp_stem,
        save_payload,
    )

    output_dir = Path(output_dir)
    signal_modes = _ROI_SIGNAL_MODES[config.roi_signal]
    exp_stem = resolve_exp_stem(
        config.input_dir, config.exp_name, config.input_format, config.input_order
    )
    fps_channel = config.fps / config.cycle_len
    channel_groups = config.channel_groups()
    has_reg = reg_channel_path(output_dir, 0).exists()
    uint16_max = np.iinfo(np.uint16).max

    def _log(msg: str) -> None:
        if reporter is not None:
            reporter.on_log(StageId.ROI, msg)
        else:
            print(msg)

    # Two ROI spaces (config.roi_space):
    #   "source" — ROIs are in the recording's own coords (rois_source.csv); no
    #     atlas transform, masks built per-group at the source hw, transform=None.
    #   "atlas"  — edited rois.csv (per-ROI size) override the atlas defaults,
    #     pulled into source space by the annotation transform.
    from .rois import build_masks, resolve_rois, resolve_source_rois

    source_mode = config.roi_space == "source"
    if source_mode:
        source_rois = resolve_source_rois(output_dir)
        if not source_rois:
            _log(
                "[roi] roi_space='source' but no rois_source.csv - nothing to "
                "extract (define source-space ROIs first)"
            )
            return {}
        roi_tform = None  # masks are already in source space
        _log(f"[roi] using {len(source_rois)} source-space ROIs from rois_source.csv")
    else:
        roi_tform = tform
        rois, is_custom = resolve_rois(output_dir, atlas)
        if is_custom:
            roi_target, roi_target_names = build_masks(rois, atlas.shape_hw)
            _log(f"[roi] using {len(rois)} custom ROIs from rois.csv")
        else:
            roi_target, roi_target_names = atlas, None

    def _roi_masks(hw):
        """The ROI masks/target for a branch of shape ``hw``."""
        if source_mode:
            return build_masks(source_rois, hw)
        return roi_target, roi_target_names

    results: dict[str, dict[str, Any]] = {}
    names = list(channel_groups)
    for gi, name in enumerate(names):
        if cancel is not None:
            cancel.raise_if_set()

        group = channel_groups[name]
        src_indices = group["source_indices"]
        if not src_indices:
            _log(f"[roi] {name}: no source channels - skipped")
            continue

        payload_out: dict[str, Any] = {}
        signal_labels: dict[str, str] = {}

        # --- Raw: the average source channel (uint16-clipped), streamed ---
        if "raw" in signal_modes:
            if not has_reg:
                _log(f"[roi] {name}: no reg_Ch file - raw skipped")
            else:
                mms = [
                    load_reg_channel(output_dir, i, mmap=True) for i in src_indices
                ]  # (T, H, W) uint16, read-only
                n_t = min(int(m.shape[0]) for m in mms)
                src_hw = (int(mms[0].shape[1]), int(mms[0].shape[2]))

                def _read_raw(t0, t1, _mms=mms):
                    # Same values the old path warped: channel-average, clipped
                    # and truncated to uint16 BEFORE the geometry step.
                    blk = np.mean(
                        [np.asarray(m[t0:t1], dtype=np.float64) for m in _mms], axis=0
                    )
                    return np.clip(blk, 0, uint16_max).astype(np.uint16)

                _raw_masks, _raw_names = _roi_masks(src_hw)
                res_raw = extract_signals_from_source(
                    _read_raw, n_t, src_hw, _raw_masks, _raw_names,
                    transform=roi_tform, diameter=diameter,
                    reporter=reporter, cancel=cancel, stage=StageId.ROI,
                    progress_label=f"{name}/raw",
                )
                for m in mms:
                    handle = getattr(m, "_mmap", None)
                    if handle is not None:
                        handle.close()
                payload_out["F_raw"] = res_raw.F.astype(np.float32)
                payload_out["roi_names"] = res_raw.roi_names
                payload_out["n_pixels"] = res_raw.n_pixels
                signal_labels["raw"] = (
                    "Raw (uint16, source ROIs)" if source_mode
                    else "Raw (uint16, warped)"
                )

        # --- dF/F: shared helper with movie export ---
        if "dff" in signal_modes:
            dff_data, dff_note = _load_name_dff_stack(
                name, group, config, output_dir
            )
            if dff_data is None:
                _log(f"[roi] {name}: {dff_note} - dff skipped")
            else:
                # (H, W, T) — memmap-backed for the donner path, so slicing time
                # here reads only the frames of the current chunk.
                dff_hw = (int(dff_data.shape[0]), int(dff_data.shape[1]))
                n_t = int(dff_data.shape[2])

                def _read_dff(t0, t1, _d=dff_data):
                    return np.moveaxis(np.asarray(_d[..., t0:t1]), 2, 0)  # (k, H, W)

                _dff_masks, _dff_names = _roi_masks(dff_hw)
                res_dff = extract_signals_from_source(
                    _read_dff, n_t, dff_hw, _dff_masks, _dff_names,
                    transform=roi_tform, diameter=diameter,
                    reporter=reporter, cancel=cancel, stage=StageId.ROI,
                    progress_label=f"{name}/dff",
                )
                payload_out["F_dff"] = res_dff.F.astype(np.float32)
                payload_out.setdefault("roi_names", res_dff.roi_names)
                payload_out.setdefault("n_pixels", res_dff.n_pixels)
                signal_labels["dff"] = dff_note

        if not signal_labels:
            continue

        payload_out["fps"] = float(fps_channel)
        payload_out["roi_signal_mode"] = config.roi_signal
        # Provenance: these files are written to a FIXED name and overwritten in
        # place, so without it a denoised roiSignals_* is indistinguishable from a
        # plain one. savemat-safe types only (no None, no dict).
        #
        # It describes F_dff only: F_raw is read straight from the reg_Ch memmaps
        # and never passes the seam, so a Raw-only payload is not denoised however
        # ica_denoise is set. And the IC numbers are 1-BASED, like every artifact a
        # human ever sees (IC3.png, the picker, ica_exclusion.json) — a 0-based
        # array here would have a scientist excluding the wrong component by hand.
        _mode = getattr(config, "ica_denoise", "off")
        _denoised = _mode != "off" and "dff" in signal_labels
        payload_out["ica_denoised"] = int(_denoised)
        payload_out["ica_denoise_mode"] = str(_mode if _denoised else "off")
        payload_out["ica_excluded_ic"] = np.array(
            [i + 1 for i in resolve_exclusion(config, output_dir, name)]
            if _denoised else [],
            dtype=np.int64,
        )

        out_path = output_dir / f"roiSignals_{name}_{exp_stem}"
        save_payload(out_path, payload_out, config.output_format)
        _log(
            f"[roi] {name}: saved {list(signal_labels)} "
            f"-> {out_path.name}.{config.output_format}"
        )

        # --- DataFrame export (rows = time [s], cols = ROI names) ---
        if config.save_roi_signals:
            import pandas as pd

            fmt = config.save_roi_signals
            write_csv = fmt in ("csv", "both")
            write_pkl = fmt in ("pickle", "both")
            roi_names = [str(r) for r in payload_out["roi_names"]]
            for mode in signal_labels:
                F = np.asarray(payload_out[f"F_{mode}"])  # (n_rois, T)
                t_sec = np.arange(F.shape[1]) / fps_channel
                df = pd.DataFrame(
                    F.T, index=pd.Index(t_sec, name="time_s"), columns=roi_names
                )
                # Build by concatenation (not with_suffix): a dotted stem would
                # otherwise eat the ``_raw``/``_dff`` discriminator and silently
                # overwrite one export with the other.
                if write_csv:
                    p = out_path.parent / f"{out_path.name}_{mode}.csv"
                    df.to_csv(p)
                    _log(f"[roi] {name} {mode}: DataFrame {df.shape} -> {p.name}")
                if write_pkl:
                    p = out_path.parent / f"{out_path.name}_{mode}.pkl"
                    df.to_pickle(p)
                    _log(f"[roi] {name} {mode}: DataFrame {df.shape} -> {p.name}")

        results[name] = payload_out

    if reporter is not None:
        reporter.on_progress(StageId.ROI, len(names), len(names))
    return results


def plot_roi_signal_grid(
    payload: dict[str, Any],
    name: str,
    fps_channel: float,
    window: tuple[int, int] | None = None,
) -> "Figure":
    """Plot per-ROI signal traces for one group's saved payload. Returns Figure.

    Mirrors the notebook's ROI plot: one column per available mode
    (``F_raw`` / ``F_dff``), one row per ROI.  ``window`` is a ``(start, end)``
    per-channel frame range; when given, only that slice is plotted (used to
    drop the unstable initial / trailing frames).
    """
    import matplotlib.pyplot as plt

    modes = [m for m in ("raw", "dff") if f"F_{m}" in payload]
    roi_names = [str(r) for r in payload["roi_names"]]
    n_rois = len(roi_names)
    n_plots = max(1, len(modes))

    fig, axes = plt.subplots(
        n_rois, n_plots,
        figsize=(7 * n_plots, 1.2 * n_rois),
        sharex=True, squeeze=False,
    )
    for col, mode in enumerate(modes):
        F = np.asarray(payload[f"F_{mode}"])
        n_t = F.shape[1]
        s, e = (0, n_t)
        if window is not None:
            s, e = max(0, window[0]), min(n_t, window[1])
            if e <= s:
                s, e = 0, n_t
        t_sec = np.arange(s, e) / fps_channel
        scale = 100.0 if mode == "dff" else 1.0
        ylabel_unit = "%" if mode == "dff" else "a.u."
        for i, roi in enumerate(roi_names):
            ax = axes[i, col]
            ax.plot(t_sec, F[i, s:e] * scale, linewidth=0.5)
            if col == 0:
                ax.set_ylabel(roi, fontsize=7, rotation=0, labelpad=50, va="center")
            if e - s > 1:
                ax.set_xlim(t_sec[0], t_sec[-1])
            ax.tick_params(labelsize=6)
        axes[0, col].set_title(f"{mode} [{ylabel_unit}]", fontsize=9)
        axes[-1, col].set_xlabel("Time (s)")
    fig.suptitle(f"ROI signals — {name}", fontsize=12)
    fig.tight_layout()
    return fig
