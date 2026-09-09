"""Write an NWB file from the processed pipeline outputs (the ``asovi-nwb`` tool).

Pure library — no Dear PyGui.  The standalone GUI (``gui/nwb_editor.py``) collects
the metadata (see :mod:`nwb_meta`) and calls :func:`write_nwb`; the export is a
read-only consumer of ``output_dir`` and never touches the pipeline / config.

Key facts baked in here:

* one-photon widefield → ``OnePhotonSeries`` (never ``TwoPhotonSeries``);
* on-disk ``.npy`` are (T,H,W); the dF/F seam ``_load_name_dff_stack`` hands out an
  (H,W,T) view, so we stream it back to (T,H,W) a chunk at a time — no full transpose;
* dF/F is read through the ``ica_denoise`` seam so ``ica_denoise="subtract"`` is honored;
* atlas-warped dF/F is produced on the fly (warp adjoint of the source dF/F), the
  affine (marks-derived) is stored as a scratch array, correlation matrices too.

Payloads are gated by :class:`nwb_meta.ExportOptions`; a payload whose inputs are
missing is skipped with a log line (never silently).  Raw ``reg_Ch`` frames are
intentionally not written in this version.
"""

from __future__ import annotations

import contextlib
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

try:
    # hdmf ships with pynwb and cannot be deferred: GenericDataChunkIterator is a
    # base class below, so it has to exist when this module is imported.  Importing
    # `asvimg` does not reach here.
    from hdmf.backends.hdf5.h5_utils import H5DataIO
    from hdmf.data_utils import GenericDataChunkIterator
except ModuleNotFoundError as exc:  # optional extra
    raise ModuleNotFoundError(
        'NWB export needs the `nwb` extra: pip install "asovi-imager[nwb]"'
    ) from exc

from .nwb_meta import ChannelMeta, ExportOptions, NwbMetadata


# --------------------------------------------------------------------------- #
# streaming iterator: (H,W,T) view -> NWB (T,H,W), O(chunk) RAM, no full transpose
# --------------------------------------------------------------------------- #

class _HWTFramesIterator(GenericDataChunkIterator):
    """Feed an (H, W, T) array/view into NWB in (T, H, W) order.

    Each buffer indexes a time-slab and ``moveaxis`` es it to frames-first, so the
    global transpose is never materialized -- doing so would cost a full copy
    of the stack."""

    def __init__(self, view_hwt, chunk_t: int):
        self._view = view_hwt
        h, w, t = view_hwt.shape
        self._maxshape = (int(t), int(h), int(w))
        super().__init__(chunk_shape=(int(chunk_t), int(h), int(w)))

    def _get_data(self, selection):
        t_sel, h_sel, w_sel = selection
        block = self._view[h_sel, w_sel, t_sel]  # (h, w, t)
        return np.moveaxis(np.asarray(block, dtype=np.float32), -1, 0)  # (t, h, w)

    def _get_maxshape(self):
        return self._maxshape

    def _get_dtype(self):
        return np.dtype("float32")


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def _s(x: str | None) -> str | None:
    """Empty/blank string -> None (so NWB leaves the field unset)."""
    if x is None:
        return None
    x = str(x).strip()
    return x or None


def _list(x) -> list | None:
    return list(x) if x else None


def _emit_log(reporter, stage, msg: str) -> None:
    if reporter is not None:
        try:
            reporter.on_log(stage, msg)
        except Exception:
            pass


def _emit_progress(reporter, stage, done: int, total: int, msg: str = "") -> None:
    if reporter is not None:
        try:
            reporter.on_progress(stage, done, total, message=msg)
        except Exception:
            pass


def _parse_start_time(raw: str) -> datetime:
    if not str(raw).strip():
        raise ValueError("session_start_time is required (tz-aware ISO-8601 datetime)")
    dt = datetime.fromisoformat(str(raw).strip())
    if dt.tzinfo is None:
        dt = dt.astimezone()  # localize a naive datetime to the system tz
    return dt


def _grid_spacing(pixel_size_um: float | None):
    if pixel_size_um is None:
        return None, "meters"
    m = float(pixel_size_um) * 1e-6
    return [m, m], "meters"


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #

def write_nwb(
    config,
    output_dir: str | Path,
    meta: NwbMetadata,
    *,
    options: ExportOptions | None = None,
    out_path: str | Path | None = None,
    tform=None,
    atlas=None,
    reporter=None,
    cancel=None,
    stage=None,
) -> Path:
    """Assemble and write an NWB file; return the written path.

    ``tform`` / ``atlas`` are reconstructed from ``marks.mat`` / the atlas mat on
    demand when a selected payload needs them; if they cannot be obtained, the
    atlas-dependent payloads are skipped (logged), not fatal.
    """
    try:
        from pynwb import NWBFile, NWBHDF5IO
        from pynwb.file import Subject
        from pynwb.ophys import (
            OpticalChannel, OnePhotonSeries, ImageSegmentation, RoiResponseSeries,
            Fluorescence, DfOverF,
        )
        from pynwb.base import Images
        from pynwb.image import GrayscaleImage
    except ModuleNotFoundError as exc:  # pynwb is an optional extra
        raise ModuleNotFoundError(
            'writing NWB needs the `nwb` extra: pip install "asovi-imager[nwb]"'
        ) from exc
    from .io import read_reg_meta, resolve_exp_stem

    if stage is None:
        try:
            from .events import StageId

            stage = StageId.EXPORT
        except Exception:
            stage = "nwb"

    out = Path(output_dir)
    opts = options or meta.options
    groups = config.channel_groups()
    fps_channel = float(config.fps) / max(1, config.cycle_len)

    reg = read_reg_meta(out)  # raises FileNotFoundError if preprocess never finished
    hw = np.asarray(reg["imageSize"]).ravel()
    source_hw = (int(hw[0]), int(hw[1]))

    try:
        exp_stem = resolve_exp_stem(
            config.input_dir, config.exp_name, config.input_format, config.input_order
        )
    except Exception:
        exp_stem = config.exp_name or "output"
    if out_path is None:
        fname = _s(opts.output_filename) or f"{exp_stem}.nwb"
        if not fname.lower().endswith(".nwb"):
            fname += ".nwb"
        out_path = out / fname
    out_path = Path(out_path)
    if out_path.exists() and not opts.overwrite:
        raise FileExistsError(f"{out_path} exists (set options.overwrite=True to replace)")

    # ---- lazy providers for the atlas transform / atlas -------------------
    _need_atlas = opts.include_warped or (opts.include_roi and config.roi_space == "atlas")

    def _get_atlas():
        nonlocal atlas
        if atlas is None and _need_atlas:
            try:
                from .atlas import ACCFv3
                from .config import resolve_atlas_path

                atlas = ACCFv3.from_mat(resolve_atlas_path(config.annotation_atlas_path))
            except Exception as exc:  # noqa: BLE001
                _emit_log(reporter, stage, f"[nwb] atlas unavailable ({exc}); atlas payloads skipped")
                atlas = None
        return atlas

    def _get_tform():
        nonlocal tform
        if tform is None and _need_atlas:
            marks = out / "marks.mat"
            if not marks.exists():
                _emit_log(reporter, stage, "[nwb] marks.mat missing; atlas payloads skipped")
                return None
            try:
                from .annotation import load_marks, compute_transform

                src, ref = load_marks(marks)
                tform = compute_transform(
                    src, ref, allow_reflection=config.annotation_allow_reflection
                )
            except Exception as exc:  # noqa: BLE001
                _emit_log(reporter, stage, f"[nwb] transform rebuild failed ({exc}); atlas payloads skipped")
                tform = None
        return tform

    # ---- NWBFile + Subject + Device ---------------------------------------
    nwb = NWBFile(
        session_description=_s(meta.session.session_description) or f"WFCI {exp_stem}",
        identifier=str(uuid4()),
        session_start_time=_parse_start_time(meta.session.session_start_time),
        session_id=_s(meta.session.session_id),
        experimenter=_list(meta.session.experimenter),
        lab=_s(meta.session.lab),
        institution=_s(meta.session.institution),
        experiment_description=_s(meta.session.experiment_description),
        keywords=_list(meta.session.keywords),
        related_publications=_list(meta.session.related_publications),
    )
    sub = meta.subject
    dob = None
    if _s(sub.date_of_birth):
        try:
            dob = _parse_start_time(sub.date_of_birth)
        except ValueError:
            dob = None
    nwb.subject = Subject(
        subject_id=_s(sub.subject_id),
        species=_s(sub.species),
        sex=_s(sub.sex),
        age=_s(sub.age),
        date_of_birth=dob,
        genotype=_s(sub.genotype),
        strain=_s(sub.strain),
        description=_s(sub.description) or _s(sub.subject_id),
    )
    device = nwb.create_device(
        name=_s(meta.device.name) or "Widefield",
        description=_s(meta.device.description),
        manufacturer=_s(meta.device.manufacturer),
        model_name=_s(meta.device.model),
    )

    ophys = nwb.create_processing_module(
        "ophys", "hemodynamic-corrected dF/F, ROI signals and segmentation (asovi-nwb)"
    )
    gs, gs_unit = _grid_spacing(meta.pixel_size_um)

    # ---- reference images (shared across groups) --------------------------
    if opts.include_reference_images:
        imgs = []
        for i in range(config.cycle_len):
            key = f"meanImageCh{i}"
            if key in reg:
                imgs.append(GrayscaleImage(name=key, data=np.asarray(reg[key], dtype="float32")))
        if "proc_template" in reg:
            imgs.append(GrayscaleImage(
                name="registration_template",
                data=np.asarray(reg["proc_template"], dtype="float32"),
            ))
        if imgs:
            ophys.add(Images(name="reference_images",
                             description="per-channel mean images + registration template (source space)",
                             images=imgs))
            _emit_log(reporter, stage, f"[nwb] reference_images: {len(imgs)} image(s)")

    # ---- lazily-created shared ROI/segmentation containers ----------------
    _seg = {"img": None, "dff": None, "raw": None}

    def _ensure_seg():
        if _seg["img"] is None:
            _seg["img"] = ImageSegmentation()
            ophys.add(_seg["img"])
        return _seg["img"]

    def _ensure_dff_container():
        if _seg["dff"] is None:
            _seg["dff"] = DfOverF(name="DfOverF")
            ophys.add(_seg["dff"])
        return _seg["dff"]

    def _ensure_raw_container():
        if _seg["raw"] is None:
            _seg["raw"] = Fluorescence(name="Fluorescence")
            ophys.add(_seg["raw"])
        return _seg["raw"]

    # keep warped memmaps alive until after io.write()
    warp_stack_cm = contextlib.ExitStack()

    source_groups = [n for n, g in groups.items() if g["source_indices"]]
    total = max(1, len(source_groups))
    written: list[str] = []

    with warp_stack_cm:
        for gi, name in enumerate(source_groups):
            if cancel is not None:
                cancel.raise_if_set()
            group = groups[name]
            ch: ChannelMeta = meta.channels.get(name) or ChannelMeta()
            emission = ch.emission_lambda
            if emission is None:
                emission = float("nan")
                _emit_log(reporter, stage, f"[nwb] {name}: emission_lambda unset -> NaN")

            oc = OpticalChannel(
                name=f"OpticalChannel_{name}",
                description=f"emission for {name} ({ch.indicator or 'unknown indicator'})",
                emission_lambda=float(emission),
            )
            ip = nwb.create_imaging_plane(
                name=f"ImagingPlane_{name}",
                optical_channel=oc,
                description=f"{ch.location or 'dorsal cortex'} (source space {source_hw[0]}x{source_hw[1]})",
                device=device,
                excitation_lambda=(None if ch.excitation_lambda is None else float(ch.excitation_lambda)),
                indicator=_s(ch.indicator) or "unknown",
                location=_s(ch.location) or "dorsal cortex",
                imaging_rate=fps_channel,
                grid_spacing=gs,
                grid_spacing_unit=gs_unit,
            )

            # --- dF/F (source) OnePhotonSeries ---
            dff_view = None
            if opts.include_dff:
                dff_view = _add_dff_series(
                    nwb, ophys, ip, name, group, config, out, opts,
                    exposure_time=ch.exposure_time, reporter=reporter, stage=stage,
                )
                if dff_view is not None:
                    written.append(f"dff:{name}")

            # --- atlas-warped dF/F ---
            if opts.include_warped:
                _add_warped_series(
                    nwb, ophys, device, name, ch, config, out, opts, source_hw,
                    dff_view=dff_view, group=group,
                    get_tform=_get_tform, get_atlas=_get_atlas, exit_stack=warp_stack_cm,
                    grid=(gs, gs_unit), reporter=reporter, stage=stage, cancel=cancel,
                )

            # --- ROI signals + segmentation ---
            if opts.include_roi:
                ok = _add_roi(
                    nwb, ophys, ip, name, config, out, source_hw,
                    ensure_seg=_ensure_seg, ensure_dff=_ensure_dff_container,
                    ensure_raw=_ensure_raw_container,
                    get_tform=_get_tform, get_atlas=_get_atlas,
                    reporter=reporter, stage=stage,
                )
                if ok:
                    written.append(f"roi:{name}")

            # --- ICA components ---
            if opts.include_ica:
                ok = _add_ica(
                    ophys, ip, name, out, fps_channel,
                    ensure_seg=_ensure_seg, reporter=reporter, stage=stage,
                )
                if ok:
                    written.append(f"ica:{name}")

            _emit_progress(reporter, stage, gi + 1, total, f"group {name}")

        # --- correlation matrices (per group, recomputed) ---
        if opts.include_correlation:
            _add_correlation(nwb, config, out, reporter=reporter, stage=stage)

        # --- source->CCF affine (global) ---
        tf = _get_tform()
        if tf is not None:
            nwb.add_scratch(
                np.asarray(tf.params, dtype="float64"),
                name="source_to_ccf_affine",
                description="source->Allen-CCF homogeneous transform (3x3, (col,row) order; "
                            "from marks.mat via compute_transform)",
            )
            written.append("affine")

        _emit_log(reporter, stage, f"[nwb] writing {out_path.name} ({', '.join(written) or 'metadata only'})")
        with NWBHDF5IO(str(out_path), "w") as io:
            io.write(nwb)

    _emit_log(reporter, stage, f"[nwb] done -> {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# payload builders
# --------------------------------------------------------------------------- #

def _wrap(data, opts: ExportOptions):
    """H5DataIO with the configured compression, for a streaming iterator."""
    if opts.compression and opts.compression.lower() != "none":
        return H5DataIO(data=data, compression=opts.compression,
                        compression_opts=int(opts.compression_opts), shuffle=bool(opts.shuffle))
    return H5DataIO(data=data)


def _add_dff_series(nwb, ophys, ip, name, group, config, out, opts, *,
                    exposure_time, reporter, stage):
    from pynwb.ophys import OnePhotonSeries
    from .annotation import _load_name_dff_stack

    stack, note = _load_name_dff_stack(
        name, group, config, out, use_ica_denoise=opts.honor_ica_denoise
    )
    if stack is None:
        _emit_log(reporter, stage, f"[nwb] {name}: no dF/F ({note}); skipped")
        return None
    fps_channel = float(config.fps) / max(1, config.cycle_len)
    it = _HWTFramesIterator(stack, opts.chunk_frames_source)
    ops = OnePhotonSeries(
        name=f"OnePhotonSeries_dff_{name}",
        imaging_plane=ip,
        data=_wrap(it, opts),
        unit="n.a.",
        rate=fps_channel,
        exposure_time=(None if exposure_time is None else float(exposure_time)),
        description=f"dF/F (source space); {note}",
    )
    ophys.add(ops)
    _emit_log(reporter, stage, f"[nwb] {name}: dF/F {it._maxshape} ({note})")
    return stack


def _add_warped_series(nwb, ophys, device, name, ch, config, out, opts, source_hw, *,
                       dff_view, group, get_tform, get_atlas, exit_stack, grid,
                       reporter, stage, cancel):
    from pynwb.ophys import OpticalChannel, OnePhotonSeries
    from .annotation import warped_stack_on_disk, _load_name_dff_stack

    tform, atlas = get_tform(), get_atlas()
    if tform is None or atlas is None:
        return None
    stack = dff_view
    if stack is None:
        stack, note = _load_name_dff_stack(name, group, config, out,
                                           use_ica_denoise=opts.honor_ica_denoise)
        if stack is None:
            _emit_log(reporter, stage, f"[nwb] {name}: no dF/F to warp; skipped")
            return None

    def _cb(done, total):
        if cancel is not None:
            cancel.raise_if_set()

    # _warptmp_* so a crashed run's leftover is swept by preprocess `delete`
    tmp = out / f"_warptmp_nwb_{name}.npy"
    view = exit_stack.enter_context(
        warped_stack_on_disk(stack, tform, atlas.shape_hw, tmp, on_frame=_cb)
    )
    oc = OpticalChannel(name=f"OpticalChannel_{name}_ccf",
                        description=f"emission for {name} (CCF space)",
                        emission_lambda=float(ch.emission_lambda if ch.emission_lambda is not None else float("nan")))
    ip_ccf = nwb.create_imaging_plane(
        name=f"ImagingPlane_{name}_ccf",
        optical_channel=oc,
        description=f"Allen CCF space ({atlas.shape_hw[0]}x{atlas.shape_hw[1]})",
        device=device,
        excitation_lambda=(None if ch.excitation_lambda is None else float(ch.excitation_lambda)),
        indicator=(ch.indicator or "unknown"),
        location=(ch.location or "dorsal cortex"),
        imaging_rate=float(config.fps) / max(1, config.cycle_len),
    )
    it = _HWTFramesIterator(view, opts.chunk_frames_atlas)
    ophys.add(OnePhotonSeries(
        name=f"OnePhotonSeries_dfWarped_{name}",
        imaging_plane=ip_ccf,
        data=_wrap(it, opts),
        unit="n.a.",
        rate=float(config.fps) / max(1, config.cycle_len),
        description="atlas-warped dF/F (Allen CCF)",
    ))
    _emit_log(reporter, stage, f"[nwb] {name}: atlas-warped dF/F {it._maxshape}")
    return True


def _resolve_source_masks(config, out, source_hw, get_tform, get_atlas, reporter, stage):
    """(names, image_masks (N,H,W) float32) in SOURCE space, or (None, None)."""
    from .rois import build_masks, resolve_rois, resolve_source_rois

    if config.roi_space == "source":
        rois = resolve_source_rois(out)
        if not rois:
            _emit_log(reporter, stage, "[nwb] no rois_source.csv; ROI skipped")
            return None, None
        masks_bool, names = build_masks(rois, source_hw)
        return names, masks_bool.astype("float32")

    atlas, tform = get_atlas(), get_tform()
    if atlas is None or tform is None:
        _emit_log(reporter, stage, "[nwb] atlas ROI needs atlas+transform; ROI skipped")
        return None, None
    from .annotation import warp_weight_maps

    rois, _custom = resolve_rois(out, atlas)
    atlas_masks, names = build_masks(rois, atlas.shape_hw)
    weights = warp_weight_maps(atlas_masks, tform, source_hw)  # (N, H*W) float64
    masks = weights.reshape(len(names), source_hw[0], source_hw[1]).astype("float32")
    return names, masks


def _add_roi(nwb, ophys, ip, name, config, out, source_hw, *,
             ensure_seg, ensure_dff, ensure_raw, get_tform, get_atlas, reporter, stage):
    from .pca import load_payload

    # only the payload (roiSignals_{name}_{stem}.{fmt}) — not the _dff/_raw.csv sidecars
    payloads = sorted(out.glob(f"roiSignals_{name}_*.{config.output_format}"))
    if not payloads:
        _emit_log(reporter, stage, f"[nwb] {name}: no roiSignals_* payload; ROI skipped")
        return False
    payload = load_payload(payloads[-1])
    F_dff = payload.get("F_dff")
    F_raw = payload.get("F_raw")
    if F_dff is None and F_raw is None:
        _emit_log(reporter, stage, f"[nwb] {name}: roiSignals has neither F_dff nor F_raw; skipped")
        return False

    names, masks = _resolve_source_masks(config, out, source_hw, get_tform, get_atlas, reporter, stage)
    if masks is None:
        return False
    n_roi = masks.shape[0]
    ref = F_dff if F_dff is not None else F_raw
    if np.asarray(ref).shape[0] != n_roi:
        _emit_log(reporter, stage,
                  f"[nwb] {name}: ROI count mismatch (masks {n_roi} vs signals {np.asarray(ref).shape[0]}); skipped")
        return False

    fps_roi = float(np.asarray(payload.get("fps", float(config.fps) / max(1, config.cycle_len))).ravel()[0])
    n_pixels = payload.get("n_pixels")
    roi_names = [str(x).strip() for x in payload.get("roi_names", names)]

    seg = ensure_seg()
    ps = seg.create_plane_segmentation(
        name=f"PlaneSegmentation_{name}",
        description=f"ROI masks in source space ({config.roi_space} ROIs)",
        imaging_plane=ip,
    )
    for k in range(n_roi):
        ps.add_roi(image_mask=masks[k])
    ps.add_column(name="region_name", description="ROI / atlas region label", data=list(roi_names))
    if n_pixels is not None:
        ps.add_column(name="n_pixels", description="mask pixel count",
                      data=[int(v) for v in np.asarray(n_pixels).ravel()])

    if F_dff is not None:
        region = ps.create_roi_table_region(description="all ROIs (dF/F)", region=list(range(n_roi)))
        ensure_dff().create_roi_response_series(
            name=f"RoiResponseSeries_dff_{name}",
            data=np.asarray(F_dff, dtype="float32").T,  # (T, n_roi)
            unit="n.a.", rois=region, rate=fps_roi,
        )
    if F_raw is not None:
        region_r = ps.create_roi_table_region(description="all ROIs (raw)", region=list(range(n_roi)))
        ensure_raw().create_roi_response_series(
            name=f"RoiResponseSeries_raw_{name}",
            data=np.asarray(F_raw, dtype="float32").T,
            unit="a.u.", rois=region_r, rate=fps_roi,
        )
    _emit_log(reporter, stage, f"[nwb] {name}: ROI signals + segmentation ({n_roi} ROIs)")
    return True


def _add_ica(ophys, ip, name, out, fps_channel, *, ensure_seg, reporter, stage):
    from pynwb.ophys import RoiResponseSeries
    from .ica_state import load_ica_basis

    basis = load_ica_basis(out, name)
    if basis is None:
        _emit_log(reporter, stage, f"[nwb] {name}: no ICA basis; skipped")
        return False
    spatial = np.asarray(basis["spatial"], dtype="float32")  # (n_ic, H, W)
    temporal = np.asarray(basis["temporal"], dtype="float32")  # (n_ic, T_used)
    frame_indices = np.asarray(basis.get("frame_indices")).ravel() if "frame_indices" in basis else None
    variance = np.asarray(basis.get("temporal_variance", [])).ravel()
    n_ic = spatial.shape[0]

    seg = ensure_seg()
    ps = seg.create_plane_segmentation(
        name=f"PlaneSegmentation_ICA_{name}",
        description="spatial-ICA components (source space); image_mask holds the spatial map",
        imaging_plane=ip,
    )
    for k in range(n_ic):
        ps.add_roi(image_mask=spatial[k])
    if variance.size == n_ic:
        ps.add_column(name="temporal_variance", description="component variance (descending)",
                      data=[float(v) for v in variance])
    region = ps.create_roi_table_region(description="all ICs", region=list(range(n_ic)))

    kw: dict[str, Any] = {}
    if frame_indices is not None and frame_indices.size == temporal.shape[1]:
        kw["timestamps"] = (frame_indices.astype("float64") / fps_channel)
    else:
        kw["rate"] = fps_channel
    ophys.add(RoiResponseSeries(
        name=f"RoiResponseSeries_ICA_{name}",
        data=temporal.T,  # (T_used, n_ic)
        unit="n.a.", rois=region, description="ICA component time courses", **kw,
    ))
    _emit_log(reporter, stage, f"[nwb] {name}: ICA components ({n_ic})")
    return True


def _add_correlation(nwb, config, out, *, reporter, stage):
    from .correlation import compute_roi_correlations

    try:
        corrs = compute_roi_correlations(config, out)
    except Exception as exc:  # noqa: BLE001
        _emit_log(reporter, stage, f"[nwb] correlation skipped ({exc})")
        return
    for name, res in corrs.items():
        C = np.asarray(res.C, dtype="float32")
        labels = ", ".join(res.roi_names)
        nwb.add_scratch(
            C, name=f"roi_correlation_{name}",
            description=f"ROI {res.mode} correlation ({res.method}); row/col order: [{labels}]",
        )
        _emit_log(reporter, stage, f"[nwb] {name}: correlation matrix {C.shape}")
