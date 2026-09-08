"""Seed-based correlation maps (functional-connectivity aid for annotation).

Compute, in Allen-atlas space, the Pearson correlation of every pixel's dF/F
time series with a seed ROI's mean time series.  Save as ``.npy`` (raw corr) +
``.png`` (RdBu_r), and inverse-warp the map back to the individual's registered
coordinates so it can be dropped into ``map_for_annot`` as an annotation aid.

The heavy step is warping the dF/F stack to atlas space; a temporal stride
(``skip_frames``) subsamples time before warping, and the per-pixel Pearson is
fully vectorised, so a whole map is cheap to compute.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CORR_MAP_METHODS = ("raw", "gsr")


def _regress_out_global(X: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Remove the (mean-centred) global signal ``g`` from each row of ``X``."""
    gc = g - g.mean()
    denom = float(gc @ gc)
    if denom <= 0:
        return X
    beta = ((X - X.mean(axis=1, keepdims=True)) @ gc) / denom  # (N,)
    return X - beta[:, None] * gc[None, :]


def _pearson_rows(X: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Pearson r of every row of ``X`` (N, T) against one series ``s`` (T,)."""
    Xc = X - X.mean(axis=1, keepdims=True)
    sc = s - s.mean()
    den = np.sqrt((Xc**2).sum(axis=1) * float(sc @ sc))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (Xc @ sc) / den
    return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def _smooth_masked(corr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth ``corr`` within its finite (in-brain) region only.

    Normalised convolution: NaN pixels neither contribute to nor receive weight,
    so smoothing does not bleed the zero background into the brain edge.
    """
    from scipy.ndimage import gaussian_filter

    m = np.isfinite(corr).astype(np.float64)
    c0 = np.where(m > 0, corr, 0.0).astype(np.float64)
    num = gaussian_filter(c0, sigma)
    den = gaussian_filter(m, sigma)
    out = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-6)
    out[m == 0] = np.nan
    return out.astype(np.float32)


def compute_seed_corr_map(
    dff: np.ndarray,
    tform,
    atlas,
    seed_roi=None,
    *,
    seed_mask: np.ndarray | None = None,
    method: str = "raw",
    skip_frames: int = 10,
    diameter: int = 5,
    smooth_sigma: float = 0.0,
    on_frame=None,
) -> np.ndarray:
    """Atlas-space seed correlation map from a source-space dF/F stack.

    Parameters
    ----------
    dff : (H, W, T) source (registered) dF/F stack.
    tform : annotation ``SimilarityTransform`` (source -> atlas).
    atlas : ``ACCFv3``.
    seed_roi : point-ROI name/index for the seed (uses ``atlas`` default ROIs);
        ignored when ``seed_mask`` is given.
    seed_mask : optional pre-built (Ha, Wa) bool seed mask in atlas space — use
        this to seed from a *configured* ROI (rois.csv) rather than atlas defaults.
    method : ``"raw"`` (Pearson) or ``"gsr"`` (global-signal regressed).
    skip_frames : temporal stride applied before warping + correlating (>= 1).
    diameter : seed ROI diameter in atlas px (only for ``seed_roi``).
    on_frame : optional ``callable(done, total)`` warp progress callback.

    Returns
    -------
    (Ha, Wa) float32 correlation map in [-1, 1]; NaN outside the brain mask.
    """
    from .annotation import warp_stack

    if method not in CORR_MAP_METHODS:
        raise ValueError(f"method must be one of {CORR_MAP_METHODS}, got {method!r}")

    step = max(1, int(skip_frames))
    sub = np.asarray(dff)[:, :, ::step]
    if sub.shape[2] < 2:
        raise ValueError("need >= 2 (subsampled) frames to correlate")
    warped = warp_stack(sub, tform, atlas.shape_hw, on_frame=on_frame)
    ha, wa, t = warped.shape
    X = warped.reshape(ha * wa, t).astype(np.float64)

    if seed_mask is not None:
        m = np.asarray(seed_mask, bool)
        if m.shape != (ha, wa):
            raise ValueError(f"seed_mask shape {m.shape} != atlas {(ha, wa)}")
        mask_flat = m.ravel()
    elif seed_roi is not None:
        mask_flat = np.asarray(
            atlas.get_point_roi_mask(seed_roi, diameter=diameter), bool
        ).ravel()
    else:
        raise ValueError("provide seed_mask or seed_roi")
    if not mask_flat.any():
        raise ValueError("seed covers no atlas pixels")

    if method == "gsr":
        X = _regress_out_global(X, X.mean(axis=0))
    seed_ts = X[mask_flat].mean(axis=0)

    corr = _pearson_rows(X, seed_ts).reshape(ha, wa).astype(np.float32)
    brain = getattr(atlas, "brain_mask", None)
    if brain is not None and np.shape(brain) == (ha, wa):
        corr[~np.asarray(brain, bool)] = np.nan
    if smooth_sigma and smooth_sigma > 0:
        corr = _smooth_masked(corr, float(smooth_sigma))
    return corr


def corr_vmax(corr_map: np.ndarray, pct: float = 99.0) -> float:
    """Symmetric display half-range: the ``pct``-th percentile of ``|corr|``
    over finite pixels (>= a small floor so a flat map still renders)."""
    finite = np.abs(corr_map[np.isfinite(corr_map)])
    v = float(np.percentile(finite, pct)) if finite.size else 0.0
    return v if v > 1e-6 else 1.0


def save_corr_map(out_dir, name: str, corr_map: np.ndarray) -> tuple[Path, Path]:
    """Write ``<out_dir>/<name>.npy`` (raw corr) + ``.png`` (RdBu_r, auto ±pct)."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / f"{name}.npy"
    png_path = out_dir / f"{name}.png"
    np.save(npy_path, np.asarray(corr_map, dtype=np.float32))
    vmax = corr_vmax(corr_map)
    sm = ScalarMappable(norm=Normalize(vmin=-vmax, vmax=vmax), cmap="RdBu_r")
    rgb = (sm.to_rgba(np.nan_to_num(corr_map, nan=0.0))[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(rgb).save(png_path)
    return npy_path, png_path


def unwarp_to_source(corr_atlas: np.ndarray, tform, source_shape) -> np.ndarray:
    """Inverse-warp an atlas-space map back to source (registered) coords.

    ``warp_image`` maps source -> atlas via ``tform.inverse``; the inverse
    direction (atlas -> source) therefore uses the forward ``tform`` as
    skimage's inverse-map.
    """
    from skimage.transform import warp

    data = np.nan_to_num(np.asarray(corr_atlas, dtype=np.float64), nan=0.0)
    return warp(
        data,
        tform,
        output_shape=tuple(int(s) for s in source_shape[:2]),
        preserve_range=True,
        order=1,
    ).astype(np.float32)
