"""Geometric transform computation and image warping.

Python equivalent of MATLAB's ``fitgeotrans`` + ``xytCrd``.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
from scipy.io import savemat
from skimage.transform import SimilarityTransform, warp

if TYPE_CHECKING:
    from .atlas import ACCFv3
    from .config import PipelineConfig


def compute_transform(
    source_pts: np.ndarray,
    reference_pts: np.ndarray,
    *,
    allow_reflection: bool = False,
):
    """Least-squares similarity transform from source to reference coordinates.

    Uses the Umeyama (1991) closed form: the optimal scale / rotation /
    translation are recovered from the SVD of the point-set cross-covariance,
    which is the minimum-mean-squared-error estimate for any ``N >= 2`` pairs.

    Reflection is a **constraint on the SVD solution**.  Writing
    ``H = svd(U, D, Vᵀ)`` the raw orthogonal fit is ``R = U Vᵀ``; when
    ``det(U Vᵀ) < 0`` that fit is a *mirror*.  With ``allow_reflection=False``
    (default) we clamp the smallest singular direction (``S = diag(1, ±1)``)
    to force ``det(R) = +1`` — a proper rotation, exactly MATLAB's
    ``fitgeotrans(src, ref, 'nonreflectivesimilarity')``.  With
    ``allow_reflection=True`` the constraint is dropped, so a genuine
    left-right flip can be represented when the control points support one.

    Note
    ----
    Reflection is only *recoverable* when the points are not collinear and not
    symmetric about the mirror axis.  Two points that both sit on the atlas
    midline cannot distinguish a flip regardless of this flag.

    Parameters
    ----------
    source_pts : (N, 2) array — points in source image as (row, col).
    reference_pts : (N, 2) array — corresponding points in atlas as (row, col).
    allow_reflection : if True, permit a mirror (det < 0) solution.

    Returns
    -------
    SimilarityTransform (proper rotation) or, when a reflection is chosen,
    AffineTransform.  Either maps source (col, row) → reference (col, row) and
    can be passed directly to ``skimage.transform.warp``.
    """
    src = np.asarray(source_pts, dtype=np.float64)
    ref = np.asarray(reference_pts, dtype=np.float64)

    if src.shape[0] < 2 or ref.shape[0] < 2:
        raise ValueError("At least 2 point pairs are required.")
    if src.shape != ref.shape:
        raise ValueError("source_pts and reference_pts must have the same shape.")

    # Work in (x, y) = (col, row) for the affine matrix
    src_xy = src[:, ::-1].copy()  # (N, 2) as (x, y)
    ref_xy = ref[:, ::-1].copy()

    n = src_xy.shape[0]
    src_c = src_xy.mean(axis=0)
    ref_c = ref_xy.mean(axis=0)
    src_centered = src_xy - src_c
    ref_centered = ref_xy - ref_c

    # Cross-covariance of the point sets (maps src -> ref).
    cov = (ref_centered.T @ src_centered) / n
    U, D, Vt = np.linalg.svd(cov)

    # Reflection constraint: force a proper rotation unless flips are allowed.
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        if not allow_reflection:
            S[-1, -1] = -1.0  # clamp -> det(R) = +1 (non-reflective)
    R = U @ S @ Vt

    # Umeyama scale: trace(D · S) / variance of the (centered) source points.
    var_src = (src_centered**2).sum() / n
    scale = float((D * np.diag(S)).sum() / max(var_src, 1e-12))

    t = ref_c - scale * (R @ src_c)

    matrix = np.eye(3)
    matrix[:2, :2] = scale * R
    matrix[:2, 2] = t

    if np.linalg.det(R) < 0:
        # A mirror cannot be expressed as a similarity; return the full affine.
        from skimage.transform import AffineTransform

        return AffineTransform(matrix=matrix)
    return SimilarityTransform(matrix=matrix)


def warp_image(
    image: np.ndarray,
    transform: SimilarityTransform,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Warp a single 2D image to the reference coordinate space.

    Parameters
    ----------
    image : (H, W) array.
    transform : SimilarityTransform from ``compute_transform``.
    output_shape : (H_out, W_out) — typically atlas size (285, 285).
    """
    return warp(
        image,
        transform.inverse,
        output_shape=output_shape,
        preserve_range=True,
        order=1,
    )


def _warp_grid(transform, output_shape, source_shape):
    """Sampling grid for ``torch.grid_sample`` — (1, H_out, W_out, 2) float64.

    Built once per stack: for every OUTPUT pixel it holds the SOURCE coordinate
    ``warp`` would pull from, normalized to [-1, 1].

    Two conventions have to line up exactly, and getting either wrong produces a
    half-pixel shift that still looks like a perfectly plausible warp:

    * ``transform.inverse.params`` acts on ``(x=col, y=row, 1)`` — the same
      matrix skimage takes for its homography fast path.
    * ``align_corners=False`` means normalized ``g`` maps to pixel
      ``(g + 1) / 2 * S - 0.5``, so the inverse is ``g = (2x + 1) / S - 1``.
      Pairing it with the ``align_corners=True`` formula is the classic bug.

    Returns None when torch is unavailable, which sends the caller to skimage.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover — torch is a hard dependency
        return None

    h_out, w_out = int(output_shape[0]), int(output_shape[1])
    h_src, w_src = int(source_shape[0]), int(source_shape[1])
    minv = np.asarray(transform.inverse.params, dtype=np.float64)
    cc, rr = np.meshgrid(
        np.arange(w_out, dtype=np.float64), np.arange(h_out, dtype=np.float64)
    )
    x = minv[0, 0] * cc + minv[0, 1] * rr + minv[0, 2]
    y = minv[1, 0] * cc + minv[1, 1] * rr + minv[1, 2]
    w = minv[2, 0] * cc + minv[2, 1] * rr + minv[2, 2]
    if not np.allclose(w, 1.0):  # projective safety net; affine leaves w == 1
        x, y = x / w, y / w
    gx = (2.0 * x + 1.0) / w_src - 1.0
    gy = (2.0 * y + 1.0) / h_src - 1.0
    return torch.from_numpy(np.ascontiguousarray(np.stack([gx, gy], axis=-1)[None]))


def _warp_block(block, transform, output_shape, grid) -> np.ndarray:
    """Warp a (n, H, W) float64 block → (n, H_out, W_out) float64.

    Frames-first, deliberately: the source on disk is (T, H, W), so a chunk of it
    is already contiguous in this order.  Taking (H, W, n) here instead would
    transpose the whole chunk twice — hundreds of MB of pointless copying that
    dwarfed the warp itself (17.3 -> 11.6 ms/frame instead of 17.3 -> 3.4).

    torch's ``grid_sample`` in float64 reproduces skimage's bilinear kernel to
    ~1e-13 (out-of-bounds neighbours contribute 0 either way).  float32 does NOT
    qualify: the grid then resolves [-1, 1] to ~8e-6 px, which shows up as a 1e-4
    error — four orders worse than the float32 the pipeline stores.  Keep the
    compute in float64 and let the caller narrow on write.
    """
    n = block.shape[0]
    ha, wa = int(output_shape[0]), int(output_shape[1])

    if grid is not None:
        import torch
        import torch.nn.functional as F

        # T goes in the BATCH dim: torch's CPU grid_sampler_2d parallelizes over
        # N only — with T in the channel dim it runs single-threaded (measured
        # flat at ~5.2 ms/frame from 1 to 36 threads).  The grid is shared, so
        # expand() it with stride 0 rather than materializing N copies.
        src = torch.from_numpy(block).unsqueeze(1)  # (n, 1, H, W)
        res = F.grid_sample(
            src, grid.expand(n, -1, -1, -1),
            mode="bilinear", padding_mode="zeros", align_corners=False,
        )
        return res[:, 0].numpy()

    out = np.empty((n, ha, wa), dtype=np.float64)
    for i in range(n):
        # clip=False (skimage's default is True) so the fallback is the same
        # LINEAR kernel grid_sample implements.  skimage's clip is inert here
        # either way — when pure-cval pixels exist it widens its range to include
        # cval, and when the source covers the atlas every output is a convex
        # combination of in-range values — but that is a property of the geometry,
        # not of order=1.  Turning it off makes the two kernels identical by
        # construction instead.
        out[i] = warp(
            block[i], transform.inverse, output_shape=(ha, wa),
            preserve_range=True, order=1, clip=False,
        )
    return out


def _zero_nans(block: np.ndarray) -> None:
    """NaN → 0 in place (interpolation cannot carry NaN; each one would otherwise
    smear over a 2x2 output neighbourhood).  ±inf is left alone, which is the
    rule this pipeline has always used.

    Done through torch because numpy's scan is single-threaded and, on a warp
    chunk (540x640 float64), costs more than the warp itself: ~2.1 ms/frame for
    ``np.nan_to_num`` vs ~0.1 through torch.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover — torch is a hard dependency
        block[np.isnan(block)] = 0.0
        return
    t = torch.from_numpy(block)
    t[torch.isnan(t)] = 0.0


def _warp_stream(stack, transform, output_shape, write, *, on_frame=None, chunk=128):
    """Warp (H, W, T) in time-chunks, handing each ``(n, H_out, W_out)`` result
    to ``write(t0, t1, warped)``.

    ``stack`` is read frames-first.  Every stack the pipeline warps is an
    (H, W, T) view of an on-disk (T, H, W) memmap, so ``moveaxis`` back gives a
    contiguous read; only seedmap's in-RAM (H, W, T) array pays a transpose.
    """
    n_frames = int(stack.shape[2])
    grid = _warp_grid(transform, output_shape, (stack.shape[0], stack.shape[1]))
    is_float = np.issubdtype(np.asarray(stack[..., :1]).dtype, np.floating)
    for t0 in range(0, n_frames, max(1, int(chunk))):
        t1 = min(t0 + max(1, int(chunk)), n_frames)
        block = np.ascontiguousarray(
            np.moveaxis(stack[..., t0:t1], 2, 0), dtype=np.float64
        )  # (n, H, W)
        if is_float:
            _zero_nans(block)  # uint16 sources cannot carry NaN — skip the scan
        write(t0, t1, _warp_block(block, transform, output_shape, grid))
        if on_frame is not None:
            on_frame(t1, n_frames)  # per chunk; also where cancellation is seen


def warp_stack(
    stack: np.ndarray,
    transform: SimilarityTransform,
    output_shape: tuple[int, int],
    *,
    on_frame=None,
    out: np.ndarray | None = None,
    dtype=np.float32,
    chunk: int = 128,
) -> np.ndarray:
    """Warp an (H, W, T) image stack.

    Equivalent to MATLAB's ``xytCrd.m``.  NaN values are replaced with 0
    before warping.

    Streams in time-chunks: the source is cast a chunk at a time (never a whole
    float64 copy of the stack) and each warped frame is written straight into
    ``out``.  The SOURCE is ~4.3x the ATLAS area (a 540x640 binned source vs the
    285x285 atlas), so it is the float64 cast of the source, not the warped
    output, that dominates: at T=6000 a whole-stack float64 copy of the input is
    16.6 GB, while the warped stack is 1.95 GB in float32 (3.9 GB in float64).
    Hence the chunked cast, the float32 default and the ``out`` hook.

    The kernel is torch ``grid_sample`` in float64 — several-fold faster than
    skimage's per-frame warp at the REAL geometry (540x640 source -> 285x285
    atlas), matching it to ~1e-13 — falling back to skimage when torch is missing.
    Benchmark it in that direction: a small-source -> large-atlas measurement
    inflates the speedup.

    Parameters
    ----------
    stack : (H, W, T) array (a memmap-backed view is fine — it is read in chunks).
    transform : SimilarityTransform.
    output_shape : (H_out, W_out).
    on_frame : optional ``callable(done, total)`` progress callback, fired once
        per chunk — used to drive a progress bar and to observe cancellation.
    out : optional (H_out, W_out, T) destination.  Its dtype wins over ``dtype``:
        pass a uint16 ``out`` to have frames cast on write (what the TIFF export
        wants).  Prefer :func:`warped_stack_on_disk`, which avoids the scatter
        into this layout entirely.
    dtype : dtype of the allocated output when ``out`` is None.  float32 by
        default — every source is uint16 or float32 dF/F, so float64 doubled the
        memory without adding information.
    chunk : frames per read/cast block (bounds the source-side working set).

    Returns
    -------
    (H_out, W_out, T) warped stack (``out`` itself when it was given).
    """
    n_frames = int(stack.shape[2])
    ha, wa = int(output_shape[0]), int(output_shape[1])
    if out is None:
        out = np.zeros((ha, wa, n_frames), dtype=dtype)

    def _write(t0, t1, warped):  # (n, H_out, W_out) -> (H_out, W_out, n)
        out[:, :, t0:t1] = np.moveaxis(warped, 0, 2)

    _warp_stream(
        stack, transform, output_shape, _write, on_frame=on_frame, chunk=chunk
    )
    return out


@contextmanager
def warped_stack_on_disk(
    stack: np.ndarray,
    transform: SimilarityTransform,
    output_shape: tuple[int, int],
    path,
    *,
    dtype=np.float32,
    on_frame=None,
    chunk: int = 128,
):
    """Warp into a ``(T, H_out, W_out)`` .npy memmap and yield it as an
    ``(H_out, W_out, T)`` view; the file is closed and removed on exit.

    Peak RAM becomes O(chunk) instead of the whole warped stack.  The view is
    laid out so ``np.moveaxis(view, 2, 0)`` hands the C-contiguous memmap back
    unchanged — ``save_stack_tiff`` therefore writes it without materializing a
    copy, and per-frame consumers (the movie writer) read one frame at a time.

    The view is only valid inside the ``with`` block: the mapping is closed and
    the backing file unlinked on exit.
    """
    from .io import open_reg_memmap

    path = Path(path)
    mm = open_reg_memmap(
        path, int(stack.shape[2]), (int(output_shape[0]), int(output_shape[1])),
        dtype=dtype,
    )
    try:
        # Writing through the (H, W, T) view still lands on mm[t0:t1] — the same
        # contiguous span of the file — so a dedicated (T, H, W) write path buys
        # nothing (measured: 3.1-3.4 ms/frame either way).  Only the element
        # order differs, and the warp dominates.
        view = np.moveaxis(mm, 0, 2)  # (H_out, W_out, T); writes land in the file
        warp_stack(
            stack, transform, output_shape,
            out=view, on_frame=on_frame, chunk=chunk,
        )
        mm.flush()
        yield view
    finally:
        handle = getattr(mm, "_mmap", None)
        if handle is not None:
            handle.close()  # Windows: release the mapping before unlinking
        del mm
        path.unlink(missing_ok=True)


def warp_weight_maps(
    masks: np.ndarray,
    transform: SimilarityTransform,
    source_shape: tuple[int, int],
) -> np.ndarray:
    """The ADJOINT of :func:`warp_stack` for a set of atlas-space ROI masks.

    Warping ``W`` is bilinear interpolation — a *linear* operator — and an ROI
    mean is a *linear* functional ``mᵀ``.  So

        mean_R(W·I) = (Wᵀ·m / |R|) · I

    i.e. the ROI signal of every frame can be read straight out of the SOURCE
    image with a fixed weight map ``v = Wᵀ·m``, and the warped stack never has to
    exist.  This builds those weight maps (unnormalized; divide by the ROI pixel
    count as ``extract_signals`` does).

    Exact, not an approximation: dropping out-of-bounds neighbours is the same as
    blending them with skimage's ``cval=0``, and bilinear output is a convex
    combination so ``warp``'s default clip never bites.

    Parameters
    ----------
    masks : (N, H_atlas, W_atlas) bool — atlas-space ROI masks.
    transform : the source→atlas transform (``warp`` pulls via its inverse).
    source_shape : (H, W) of the source images.

    Returns
    -------
    (N, H*W) float64 — per-ROI weight over flattened source pixels.
    """
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[np.newaxis]
    n_rois, ha, wa = masks.shape
    h, w = int(source_shape[0]), int(source_shape[1])

    # For each atlas pixel, where does warp() pull from in the source?
    py, px = np.mgrid[0:ha, 0:wa]
    pts = np.column_stack([px.ravel().astype(np.float64), py.ravel().astype(np.float64)])
    src = np.asarray(transform.inverse(pts))  # (P, 2) as (x, y), like skimage
    sx, sy = src[:, 0], src[:, 1]
    x0 = np.floor(sx).astype(np.int64)
    y0 = np.floor(sy).astype(np.int64)
    fx, fy = sx - x0, sy - y0

    flat_masks = masks.reshape(n_rois, -1)
    out = np.zeros((n_rois, h * w), dtype=np.float64)
    for dy in (0, 1):
        for dx in (0, 1):
            yi, xi = y0 + dy, x0 + dx
            wt = (fy if dy else 1.0 - fy) * (fx if dx else 1.0 - fx)
            inside = (yi >= 0) & (yi < h) & (xi >= 0) & (xi < w) & (wt != 0.0)
            flat_idx = (yi * w + xi)[inside]
            wt_in = wt[inside]
            for r in range(n_rois):
                sel = flat_masks[r][inside]
                if not sel.any():
                    continue
                out[r] += np.bincount(
                    flat_idx[sel], weights=wt_in[sel], minlength=h * w
                )
    return out


# ---------- marks I/O (MATLAB compatible) ----------


def save_marks(
    path: Path,
    source_pts: np.ndarray,
    reference_pts: np.ndarray,
) -> None:
    """Save control points to ``marks.mat`` (MATLAB-compatible).

    The file stores ``marks`` as (N, 2) with (col, row) convention to match
    the MATLAB pipeline's ``marks(:,2:3)`` usage.
    """
    src = np.asarray(source_pts, dtype=np.float64)  # (N, 2) row, col
    ref = np.asarray(reference_pts, dtype=np.float64)

    # MATLAB marks format: (N, 2) with (col, row) — swap columns
    marks_matlab = src[:, ::-1].copy()

    p = Path(path)
    if p.suffix != ".mat":
        p = p.with_suffix(".mat")
    savemat(str(p), {"marks": marks_matlab, "fixedPos": ref[:, ::-1].copy()})


def load_marks(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load control points from ``marks.mat``.

    Returns
    -------
    source_pts : (N, 2) as (row, col).
    reference_pts : (N, 2) as (row, col).
    """
    p = Path(path)
    try:
        from scipy.io import loadmat

        raw = loadmat(str(p))
        marks = np.asarray(raw["marks"], dtype=np.float64)
        fixed = np.asarray(raw["fixedPos"], dtype=np.float64)
    except NotImplementedError:
        with h5py.File(p, "r") as f:
            marks = np.array(f["marks"], dtype=np.float64)
            fixed = np.array(f["fixedPos"], dtype=np.float64)

    # stored as (N, 2) with (col, row) — swap to (row, col)
    return marks[:, ::-1].copy(), fixed[:, ::-1].copy()


# --------------------------------------------------------------------------
# Config-driven annotation pipeline (section 02 helpers)
# --------------------------------------------------------------------------


def load_atlas_source_mean(
    config: "PipelineConfig",
    output_dir: Path,
) -> np.ndarray:
    """Full-recording mean of ``imageCh{ch_for_annotation}`` across every
    ``frameRoiCh_*`` batch, as a float64 (H, W) image."""
    from .io import full_channel_mean

    return full_channel_mean(output_dir, config.ch_for_annotation)


def needs_cpselect(config: "PipelineConfig", output_dir) -> bool:
    """True when annotation must open the interactive point picker.

    That is: ``annotation == "gui"``, **or** ``annotation == "cache"`` with no
    ``marks.mat`` yet — cache falls back to picking points on first use instead
    of silently skipping the warp.  Every other mode (cache-with-marks, a
    ([src],[ref]) tuple, ``False``) resolves non-interactively.
    """
    mode = config.annotation
    if mode == "gui":
        return True
    if mode == "cache" and not (Path(output_dir) / "marks.mat").exists():
        return True
    return False


def resolve_marks(
    config: "PipelineConfig",
    output_dir: Path,
    atlas: "ACCFv3",
    source_img: np.ndarray,
    *,
    extra_imgs: list[np.ndarray] | None = None,
    extra_labels: list[str] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Dispatch on ``config.annotation`` and return (src_pts, ref_pts).

    Modes:
      - ``False``          → ``(None, None)``
      - ``"cache"``        → ``load_marks(marks.mat)`` if present else ``(None, None)``
      - ``"gui"``          → back up existing ``marks.mat``, open ``cpselect_gui``,
        save the new marks; cancelled GUI → ``(None, None)``
      - ``(src, ref)``     → save the provided arrays and return them
    """
    output_dir = Path(output_dir)
    marks_path = output_dir / "marks.mat"

    mode = config.annotation

    if mode is False:
        print("Annotation skipped.")
        return None, None

    if mode == "cache":
        if marks_path.exists():
            src_pts, ref_pts = load_marks(marks_path)
            print(f"Loaded cached marks from {marks_path}")
            print(f"  {len(src_pts)} point pairs")
            return src_pts, ref_pts
        print(f"No cached marks at {marks_path} — skipping annotation.")
        return None, None

    if mode == "gui":
        from .cpselect import cpselect_gui

        if marks_path.exists():
            old_path = marks_path.with_suffix(".mat.old")
            old_path.unlink(missing_ok=True)
            marks_path.rename(old_path)
            print(f"Backed up existing marks → {old_path.name}")

        all_imgs = [source_img] + list(extra_imgs or [])
        default_label = (
            f"Mean Ch{config.ch_for_annotation} "
            f"({config.channels_name[config.ch_for_annotation]})"
        )
        all_labels = [default_label] + list(extra_labels or [])

        result = cpselect_gui(
            all_imgs,
            atlas.image_rgb,
            n_default_pairs=2,
            boundaries=atlas.boundaries,
            source_labels=all_labels,
        )
        if result is None:
            print("GUI cancelled — no marks saved.")
            return None, None

        src_pts, ref_pts = result
        save_marks(marks_path, src_pts, ref_pts)
        print(f"Saved marks to {marks_path}")
        print(f"  {len(src_pts)} point pairs")
        return src_pts, ref_pts

    if isinstance(mode, (list, tuple)) and len(mode) == 2:
        src_pts = np.asarray(mode[0], dtype=np.float64)
        ref_pts = np.asarray(mode[1], dtype=np.float64)
        save_marks(marks_path, src_pts, ref_pts)
        print(f"Direct coords → saved to {marks_path}")
        return src_pts, ref_pts

    raise ValueError(f"Unknown annotation mode: {mode!r}")


# --- internal helper shared by movies + ROI extraction ---


def _load_name_dff_stack(
    name: str,
    group: dict,
    config: "PipelineConfig",
    output_dir: Path,
    *,
    use_ica_denoise: bool = True,
) -> tuple[np.ndarray | None, str]:
    """Return a (H, W, T) dF/F stack for a single channels_name group.

    THE definition of "this group's dF/F", and the only one: PCA/ICA, ROI, the
    exports and the movies all come through here, so they cannot disagree about
    what they are analysing.

    - ``dff_{name}.npy`` when preprocess wrote it — a memmap-backed float32
      (H, W, T) view, so consumers stream it instead of materializing a float64
      copy (3.1 GB at T=6000).  Written for EVERY group with a source: with a
      donner it is the linear-subtraction dF/F, without one the source average
      against a per-pixel percentile baseline.
    - donner absent AND no file (an output dir from before that was written, or
      ``linear_subt=False``) → the same percentile-baseline dF/F, computed in RAM.
    - no source channels or missing files → ``(None, reason)``

    ``use_ica_denoise=False`` returns the plain dF/F even under
    ``ica_denoise="subtract"``.  PCA/ICA need that: they FIT the basis the
    denoising is derived from, so asking the seam for the denoised stack there
    would demand a basis that does not exist yet — and, if it did, would fit the
    next basis on data the last one already cleaned.
    """
    from .io import dff_name_path, load_dff, mean_source_stack

    output_dir = Path(output_dir)
    src_indices = group["source_indices"]
    don_indices = group["donner_indices"]

    if not src_indices:
        return None, f"{name}: no source channels"

    has_donner = len(don_indices) > 0

    if use_ica_denoise and getattr(config, "ica_denoise", "off") != "off":
        # THE seam. Every consumer of a group's dF/F comes through here, so the
        # denoised stack cannot be bypassed by one of them (the GUI's seed-map
        # panel used to read dff_{name}.npy directly and would have kept serving
        # un-denoised data). ensure_denoised_dff RAISES rather than falling back:
        # returning the plain dF/F under a config that says "denoised" is the one
        # failure mode that must never be silent.
        from .ica_state import ensure_denoised_dff, resolve_exclusion

        path = ensure_denoised_dff(config, output_dir, name)
        data = np.moveaxis(np.load(path, mmap_mode="r"), 0, 2)
        excluded = resolve_exclusion(config, output_dir, name)
        return data, f"dF/F (ICA-denoised, excluded {[i + 1 for i in excluded]})"

    if dff_name_path(output_dir, name).exists():
        data = np.moveaxis(load_dff(output_dir, name, mmap=True), 0, 2)
        return data, (
            "dF/F (linear subt)" if has_donner
            else f"dF/F (p{float(config.baseline_percentile):g} baseline)"
        )
    if has_donner:
        return None, f"{name}: no dff_{name}.npy"

    # Fallback for donner-less groups whose dff file was never written.
    try:
        src_avg = mean_source_stack(output_dir, src_indices)
    except FileNotFoundError:
        return None, f"{name}: no reg_Ch*.npy"
    pct = float(config.baseline_percentile)
    N = config.resolved_start_initial_frames
    src_for_base = src_avg[..., N:] if 0 < N < src_avg.shape[-1] else src_avg
    baseline = np.percentile(src_for_base, pct, axis=2, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        data = np.where(baseline > 0, (src_avg - baseline) / baseline, 0.0)
    return data, f"dF/F (p{pct:g} baseline, skip {N} init)"


# Row-strip working-set budget for _time_average (float64 bytes per strip).
# Module-level so tests can shrink it to force the multi-strip / partial-last-
# strip paths on small arrays.
_POST_STRIP_BYTES = 256_000_000


def _time_average(stack: np.ndarray, half_window: int, *, reporter=None, stage=None) -> np.ndarray:
    """Centered moving average along the last axis with edge clipping.

    ``half_window=N`` → window of 2N+1 frames; near edges the window is
    truncated so the output length matches the input.

    Memory-bounded and IN PLACE: ``stack`` is the (H, W, T) atlas-warped scratch
    memmap, and under whole-folder concatenation T reaches ~1.8e5, so a single
    float64 ``cumsum`` over the whole stack is tens of GiB (the OOM this
    replaced — 83.5 GiB at T=138000). Each pixel's moving average depends only
    on its OWN time series, so the leading axis (rows) is processed in strips:
    peak RAM is O(strip * W * T), independent of H, and the averaged strip is
    written straight back into ``stack``. Row striping is EXACT (no halo) — no
    cross-row term exists — and the whole strip's original values are read into
    RAM before write-back, so the in-place write is safe. Strip height is
    budgeted like ``_compute_and_save_dff``.

    NOTE (deferred "Fix ②"): the payload writer downstream still materializes one
    float16 copy of the warped stack (~27 GiB at T=180000); see the comment in
    ``save_annotated_dff_payloads``. This function fixes only the time-average
    leg. Baseline-checked against the pre-strip implementation in
    ``tests/test_time_average.py``.
    """
    if half_window <= 0:
        return stack
    T = int(stack.shape[-1])
    if T == 0:
        return stack
    N = int(half_window)
    H = int(stack.shape[0])
    t = np.arange(T)
    lo = np.maximum(0, t - N)              # (T,) window start per output frame
    hi = np.minimum(T, t + N + 1)          # (T,) window end (exclusive)
    counts = (hi - lo).astype(np.float64)  # (T,) frames averaged per output frame
    up_idx = hi - 1                        # cumsum index for the window's upper edge
    lo_idx = np.maximum(lo - 1, 0)         # cumsum index for the lower edge
    lo_has = lo > 0                        # where the lower cumsum term is not 0
    row_bytes = int(np.prod(stack.shape[1:])) * 8  # float64 bytes per leading index
    strip_h = int(np.clip(_POST_STRIP_BYTES // max(1, row_bytes), 1, H))
    for r0 in range(0, H, strip_h):
        r1 = min(r0 + strip_h, H)
        s = np.asarray(stack[r0:r1], dtype=np.float64)  # (strip, W, T) — the only RAM copy
        csum = np.cumsum(s, axis=-1)
        upper = csum[..., up_idx]
        lower = np.where(lo_has, csum[..., lo_idx], 0.0)
        stack[r0:r1] = ((upper - lower) / counts).astype(stack.dtype)
        if reporter is not None and stage is not None:
            reporter.on_progress(stage, r1, H, message="post: time-average")
    return stack


def _needs_post_annotation(config) -> bool:
    """Whether :func:`_post_annotation_process` would do anything.

    When it wouldn't, the warped stack can stay on disk as the export's final
    dtype instead of being materialized in RAM just to be copied.
    """
    return bool(
        config.post_annotation_time_average
        or getattr(config, "post_annotation_filter_xyt", None)
    )


def _post_annotation_process(config, xyt: np.ndarray, *, reporter=None, stage=None) -> np.ndarray:
    """Post-warp temporal average, then the optional post-annotation 3D filter.

    ``xyt`` is the (H, W, T) atlas-space warped scratch memmap; the ``[x, y, t]``
    filter window maps to axis sizes ``(y, x, t)`` = ``(H, W, T)``.

    The temporal average (:func:`_time_average`) is now memory-bounded and edits
    ``xyt`` in place. The optional 3D filter still materializes a float32 copy
    (``apply_xyt_filter``) when a real window is set, but a bare ``[1,1,1]`` is a
    no-op that returns the input untouched; the whole call is skipped when
    neither is configured (see :func:`_needs_post_annotation`).
    """
    xyt = _time_average(
        xyt, config.post_annotation_time_average, reporter=reporter, stage=stage
    )
    fxyt = getattr(config, "post_annotation_filter_xyt", None)
    if fxyt:
        from .filters import apply_xyt_filter

        fx, fy, ft = fxyt
        xyt = apply_xyt_filter(
            xyt, (fy, fx, ft), config.post_annotation_filter_kind
        )
    return xyt


def _export_log(reporter, stage, msg):
    """Route an export message to the reporter (GUI log) or stdout (CLI)."""
    if reporter is not None and stage is not None:
        reporter.on_log(stage, msg)
    else:
        print(msg)


def _export_frame_cb(reporter, cancel, stage, label):
    """A ``warp_stack`` on_frame callback that reports progress + honours cancel."""
    if reporter is None and cancel is None:
        return None

    def cb(done, total):
        if cancel is not None:
            cancel.raise_if_set()
        if reporter is not None and stage is not None:
            reporter.on_progress(stage, done, total, message=label)

    return cb


def save_annotated_tiffs(
    config: "PipelineConfig",
    output_dir: Path,
    tform: SimilarityTransform,
    atlas: "ACCFv3",
    *,
    reporter=None,
    cancel=None,
    stage=None,
) -> None:
    """Warp every registered channel (``reg_Ch{i}.npy``) into atlas space and
    write ``tiffs/annot_Ch{i}_{name}-{prop}/`` big-tiff stacks (one file per
    channel). No-op when ``config.save_annotated_each_ch`` is False.
    """
    if not config.save_annotated_each_ch:
        _export_log(reporter, stage, "Annotated TIFF saving disabled.")
        return

    from .io import load_full_channel_stack, reg_channel_path, save_stack_tiff

    output_dir = Path(output_dir)
    tiff_base = output_dir / "tiffs"

    if not reg_channel_path(output_dir, 0).exists():
        _export_log(reporter, stage, "No reg_Ch*.npy files - annotated TIFF skipped.")
        return

    exp_name = config.exp_name or "output"

    post = _needs_post_annotation(config)
    for ch_idx in range(config.cycle_len):
        if not reg_channel_path(output_dir, ch_idx).exists():
            continue
        # The uint16 (H,W,T) view goes in as-is; warp_stack casts per chunk, so
        # no float64 copy of the source is ever made.
        ch_raw = load_full_channel_stack(output_dir, ch_idx)  # (H,W,T) uint16 view
        tok = config.channel_token(ch_idx)
        ch_dir = tiff_base / f"annot_{tok}"
        ch_dir.mkdir(parents=True, exist_ok=True)
        # Without post-processing the warp writes uint16 straight into the temp
        # memmap (the cast the old code did afterwards, in RAM), and the TIFF is
        # written from that file — the warped stack never enters RAM.
        with warped_stack_on_disk(
            ch_raw, tform, atlas.shape_hw,
            output_dir / f"_warptmp_Ch{ch_idx}.npy",
            dtype=np.float32 if post else np.uint16,
            on_frame=_export_frame_cb(reporter, cancel, stage, f"annot Ch{ch_idx}"),
        ) as ch_warped:
            data = (
                _post_annotation_process(
                    config, ch_warped, reporter=reporter, stage=stage
                ).astype(np.uint16)
                if post else ch_warped
            )
            n_frames = data.shape[2]
            save_stack_tiff(
                ch_dir / f"{exp_name}_annot_{tok}.tif",
                data,
                tiff_format=config.tiff_format,
                compression=config.tiff_compression,
            )
        _export_log(reporter, stage, f"[export] saved annot_{tok} ({n_frames} frames)")

    _export_log(reporter, stage, f"[export] annotated TIFFs -> {tiff_base}")


def save_annotated_dff_payloads(
    config: "PipelineConfig",
    output_dir: Path,
    tform: SimilarityTransform,
    atlas: "ACCFv3",
    *,
    reporter=None,
    cancel=None,
    stage=None,
) -> None:
    """Save atlas-warped dF/F per ``channels_name`` group as ``{exp}_dfWarped_{name}_*``
    using ``config.output_format`` (mat/npy/h5). No-op when
    ``config.save_annotated_dF_mat`` is False.
    """
    if not config.save_annotated_dF_mat:
        _export_log(reporter, stage, "Annotated dF/F payload saving disabled.")
        return

    from .io import save_payload

    output_dir = Path(output_dir)
    exp_name = config.exp_name or "output"
    ext = config.output_format

    for name, group in config.channel_groups().items():
        data, note = _load_name_dff_stack(name, group, config, output_dir)
        if data is None:
            _export_log(reporter, stage, f"[annot-dF] {note} - skipped")
            continue

        # The payload writer needs the whole stack in one array, so this is the
        # one export that cannot stay on disk — but it is opt-in
        # (save_annotated_dF_mat) and now float32, half of what it used to take.
        #
        # KNOWN LIMITATION — deferred "Fix ②" (2026-07): the astype(float16) +
        # np.ascontiguousarray below still pull ONE full copy of the atlas-warped
        # stack into RAM, O(H*W*T) ~= 27 GiB for a 180k-frame whole-folder run.
        # That breaks the low-memory-PC contract the rest of the pipeline honors
        # (_time_average above was fixed to strip; this leg was not). A streaming
        # writer — chunk the (T,H,W) warp memmap into an open_memmap .npy / a
        # chunked h5 dataset plus a small metadata sidecar — would make it
        # O(chunk); it was deferred because it changes the dfWarped file layout
        # (external MATLAB readers). Until then a whole-folder export needs RAM
        # for one float16 copy of the warped stack, or save_annotated_dF_mat=False.
        with warped_stack_on_disk(
            data, tform, atlas.shape_hw,
            output_dir / f"_warptmp_dff_{name}.npy",
            on_frame=_export_frame_cb(reporter, cancel, stage, f"dff warp {name}"),
        ) as warped:
            xyt_warped = _post_annotation_process(
                config, warped, reporter=reporter, stage=stage
            )

            # Final export orientation is user-selectable: HWT (MATLAB/legacy
            # parity) or THW (matches the internal reg_Ch/dff .npy layout).
            arr = (
                np.moveaxis(xyt_warped, 2, 0)
                if config.export_orientation == "THW" else xyt_warped
            )
            # Optional float16 storage (halves npy/h5). scipy.io.savemat upcasts
            # float16 back to float64, so a .mat would grow, not shrink — keep it
            # float32 there and say so, rather than silently doubling the file.
            if getattr(config, "save_annotated_dF_dtype", "float32") == "float16":
                if ext == "mat":
                    _export_log(
                        reporter, stage,
                        f"[annot-dF] float16 requested but .mat cannot store it "
                        f"(scipy upcasts to float64); keeping float32 for {name}",
                    )
                else:
                    arr = arr.astype(np.float16)
            from .ica_state import resolve_exclusion

            _mode = getattr(config, "ica_denoise", "off")
            payload = {
                "imageDf": np.ascontiguousarray(arr),  # detach from the memmap
                "channel_name": name,
                "imageSize": np.array(arr.shape),
                "orientation": config.export_orientation,
                "post_annotation_time_average": int(config.post_annotation_time_average),
                "source_indices": np.array(group["source_indices"]),
                "donner_indices": np.array(group["donner_indices"]),
                # Provenance — the file is overwritten in place under a fixed name.
                # 1-based IC numbers, matching IC{i}.png / the picker / the sidecar.
                "ica_denoised": int(_mode != "off"),
                "ica_denoise_mode": str(_mode),
                "ica_excluded_ic": np.array(
                    [i + 1 for i in resolve_exclusion(config, output_dir, name)]
                    if _mode != "off" else [],
                    dtype=np.int64,
                ),
            }
            n_frames = xyt_warped.shape[2]
        stem = f"{exp_name}_dfWarped_{name}_01"
        save_payload(output_dir / stem, payload, ext)
        _export_log(
            reporter, stage,
            f"[annot-dF] saved {stem}.{ext} ({n_frames} frames, {note})",
        )


# Movie codec choice -> (cv2 FourCC, container extension). "DIB (RAW)" is
# written as truly uncompressed video (FourCC 0): OpenCV's FFMPEG backend has no
# working 'DIB ' tag, so a literal DIB request silently produced no file.
_MOVIE_CODECS: dict[str, tuple[object, str]] = {
    "MJPG": ("MJPG", "avi"),
    "DIB (RAW)": (0, "avi"),
    "mp4V": ("mp4v", "mp4"),
}


def _resolve_movie_codec(codec: str) -> tuple[int, str]:
    """Return ``(cv2 fourcc int, file extension)`` for a ``save_movie_codec``.

    An unrecognised value is treated as a raw 4-char FourCC in an ``.avi``
    container, so a hand-edited ops.yaml with e.g. ``XVID`` still works.
    """
    import cv2

    if codec in _MOVIE_CODECS:
        tag, ext = _MOVIE_CODECS[codec]
        fourcc = tag if isinstance(tag, int) else cv2.VideoWriter_fourcc(*tag)
        return int(fourcc), ext
    tag = (codec or "MJPG")[:4].ljust(4)
    return int(cv2.VideoWriter_fourcc(*tag)), "avi"


def _render_dff_movie_frame(xyt_warped, i, sm, atlas, name, fps_channel):
    """One RGB ``(H, W, 3)`` uint8 movie frame: colormapped dF/F [%] with the
    atlas brain mask applied, region boundaries drawn, and a ``{name} {t}s``
    label burned in."""
    import cv2

    rgba = sm.to_rgba(
        100 * np.asarray(xyt_warped[:, :, i], dtype=np.float64), bytes=True
    )
    rgb = rgba[:, :, :3].copy()
    rgb[~atlas.brain_mask] = 0
    atlas.draw_boundaries_cv2(rgb)
    t_sec = i / fps_channel
    cv2.putText(
        rgb, f"{name} {t_sec:.1f}s", (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return rgb


def save_warped_movies(
    config: "PipelineConfig",
    output_dir: Path,
    tform: SimilarityTransform,
    atlas: "ACCFv3",
    *,
    raw_stem: str | None = None,
    reporter=None,
    cancel=None,
    stage=None,
) -> None:
    """Write atlas-warped dF/F movies.

    One ``{exp}_{name}_dF`` movie per unique ``channels_name`` group, or — when
    ``config.save_movie_merge_chs`` is set and more than one source group has
    data — a single ``{exp}_merged_dF`` movie with the groups side by side.
    The container/codec follow ``config.save_movie_codec``.

    No-op when ``config.save_movie`` is False. Uses ``_load_name_dff_stack`` so
    donner-absent names still get a p5-baseline dF/F movie.
    """
    if not config.save_movie:
        _export_log(reporter, stage, "Movie saving disabled.")
        return

    import cv2
    from contextlib import ExitStack
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from tqdm.auto import tqdm

    output_dir = Path(output_dir)
    movie_dir = output_dir / "movies"
    movie_dir.mkdir(parents=True, exist_ok=True)

    if raw_stem is None:
        from .io import find_input_files

        _files = find_input_files(Path(config.input_dir), config.input_format, config.input_order)
        raw_stem = _files[0].stem if _files else (config.exp_name or "output")

    fps_channel = config.fps / config.cycle_len
    movie_fps = max(1.0, fps_channel * config.save_movie_speed)
    vmin, vmax = config.save_movie_vminmax
    sm = ScalarMappable(
        norm=Normalize(vmin=vmin, vmax=vmax), cmap=config.save_movie_cmap
    )
    fourcc, ext = _resolve_movie_codec(config.save_movie_codec)
    movie_stem = config.exp_name or raw_stem or "output"

    # Resolve each source group's dF/F once; the warp streams it off disk below.
    pairs: list[tuple[str, np.ndarray, str]] = []
    for name, group in config.channel_groups().items():
        data, note = _load_name_dff_stack(name, group, config, output_dir)
        if data is None:
            _export_log(reporter, stage, f"[movie] {note} - skipped")
            continue
        pairs.append((name, data, note))
    if not pairs:
        return

    # merge_Chs: multiple source groups side by side in one movie. Every group
    # warps into the same atlas grid, so heights match; frame counts can differ
    # at a mid-cycle tail, so the merged clip is trimmed to the shortest.
    if bool(getattr(config, "save_movie_merge_chs", False)) and len(pairs) > 1:
        with ExitStack() as es:
            warped_list: list[tuple[str, np.ndarray]] = []
            for name, data, _note in pairs:
                warped = es.enter_context(warped_stack_on_disk(
                    data, tform, atlas.shape_hw,
                    output_dir / f"_warptmp_movie_{name}.npy",
                    on_frame=_export_frame_cb(reporter, cancel, stage, f"movie warp {name}"),
                ))
                xyt = (
                    _post_annotation_process(config, warped)
                    if _needs_post_annotation(config) else warped
                )
                warped_list.append((name, xyt))

            n_frames = min(int(x.shape[2]) for _, x in warped_list)
            h = int(warped_list[0][1].shape[0])
            w_total = sum(int(x.shape[1]) for _, x in warped_list)
            names = "+".join(nm for nm, _ in warped_list)
            movie_path = movie_dir / f"{movie_stem}_merged_dF.{ext}"
            writer = cv2.VideoWriter(str(movie_path), fourcc, movie_fps, (w_total, h))

            _enc_cb = _export_frame_cb(reporter, cancel, stage, "movie encode merged")
            _enc_step = max(1, n_frames // 100)
            try:
                for i in tqdm(range(n_frames), desc="Movie merged", unit="frame"):
                    if _enc_cb is not None and (i % _enc_step == 0 or i == n_frames - 1):
                        _enc_cb(i + 1, n_frames)
                    rgb = np.hstack([
                        _render_dff_movie_frame(x, i, sm, atlas, nm, fps_channel)
                        for nm, x in warped_list
                    ])
                    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            finally:
                writer.release()  # release the file even if the run is cancelled
        _export_log(
            reporter, stage,
            f"[movie] {movie_path.name}: {n_frames} frames x {len(warped_list)} "
            f"chs ({names}) @ {movie_fps:.0f} fps",
        )
        return

    for name, data, note in pairs:
        # The encoder consumes one frame at a time, so the warped stack streams
        # off disk and never lands in RAM (post-processing, if configured, still
        # needs the whole timeline and materializes it).
        with warped_stack_on_disk(
            data, tform, atlas.shape_hw,
            output_dir / f"_warptmp_movie_{name}.npy",
            on_frame=_export_frame_cb(reporter, cancel, stage, f"movie warp {name}"),
        ) as warped:
            xyt_warped = (
                _post_annotation_process(config, warped)
                if _needs_post_annotation(config) else warped
            )
            n_frames = xyt_warped.shape[2]
            h, w = xyt_warped.shape[:2]
            movie_path = movie_dir / f"{movie_stem}_{name}_dF.{ext}"
            writer = cv2.VideoWriter(str(movie_path), fourcc, movie_fps, (w, h))

            _enc_cb = _export_frame_cb(reporter, cancel, stage, f"movie encode {name}")
            _enc_step = max(1, n_frames // 100)
            try:
                for i in tqdm(range(n_frames), desc=f"Movie {name}", unit="frame"):
                    if _enc_cb is not None and (i % _enc_step == 0 or i == n_frames - 1):
                        _enc_cb(i + 1, n_frames)
                    rgb = _render_dff_movie_frame(xyt_warped, i, sm, atlas, name, fps_channel)
                    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            finally:
                writer.release()  # release the file even if the run is cancelled
        _export_log(
            reporter, stage,
            f"[movie] {movie_path.name}: {n_frames} frames @ {movie_fps:.0f} fps "
            f"({note})",
        )
