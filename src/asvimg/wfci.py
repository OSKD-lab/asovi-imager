"""WFCI (Wide-Field Cortical Imaging) correction.

Per-pixel linear regression to remove donner contribution from source signal,
followed by dF/F normalization.
"""

from __future__ import annotations

import numpy as np


def wfci_corrected_df(
    src: np.ndarray,
    donor: np.ndarray,
    foi: np.ndarray,
    *,
    baseline_percentile: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scalar (single-pixel) WFCI correction.

    Parameters
    ----------
    src : (T,) – source timeseries for one pixel
    donor : (T,) – donner timeseries for one pixel
    foi : (F,) – frame-of-interest indices for regression

    Returns
    -------
    d_f, y, y_base, subt – all (T,)
    """
    x = np.arange(donor.shape[0])
    x_interp = np.linspace(0, donor.shape[0] - 1, donor.shape[0] * 2)
    donor_i = np.interp(x_interp, x, donor)
    donor_even = donor_i[1::2]

    foi_safe = foi[(foi >= 0) & (foi < src.shape[0])]
    if foi_safe.size < 2:
        foi_safe = np.arange(src.shape[0])

    design = np.column_stack([donor_even[foi_safe], np.ones(foi_safe.size)])
    coeff, *_ = np.linalg.lstsq(design, src[foi_safe], rcond=None)

    y = src - (donor_even * coeff[0] + coeff[1]) + np.mean(src)
    y_base = np.full_like(y, np.percentile(y, baseline_percentile))
    d_f = -1.0 + y / y_base
    subt = y - y_base + np.mean(y)
    return d_f, y, y_base, subt


def _time_window(arr: np.ndarray, foi: np.ndarray) -> np.ndarray:
    """``arr[..., foi]`` — as a VIEW when ``foi`` is a contiguous ascending run.

    ``foi`` is ``arange(start, end)`` in every pipeline path, and fancy-indexing
    it copies the whole strip: two ~250 MB copies per row-strip, measured at
    ~730 ms — a third of the correction's runtime, spent on nothing.  A slice of
    the same range is a free view.  The arange check is O(T) on a few thousand
    indices, i.e. nothing next to the copy it avoids.
    """
    if foi.size and np.array_equal(foi, np.arange(foi[0], foi[0] + foi.size)):
        return arr[..., int(foi[0]) : int(foi[0]) + foi.size]
    return arr[..., foi]


def _wfci_strip(
    src: np.ndarray,
    donor_interp: np.ndarray,
    foi_safe: np.ndarray,
    baseline_percentile: float,
    baseline_ignore_initial: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized WFCI for a row-strip.

    Parameters
    ----------
    src : (strip_h, W, Ts) float64
    donor_interp : (strip_h, W, Ts) float64 – already interpolated to src times
    foi_safe : (F,) int – frame-of-interest indices
    """
    d_foi = _time_window(donor_interp, foi_safe)
    s_foi = _time_window(src, foi_safe)
    F = foi_safe.size

    sum_d = d_foi.sum(axis=2)
    sum_s = s_foi.sum(axis=2)
    sum_d2 = (d_foi * d_foi).sum(axis=2)
    sum_ds = (d_foi * s_foi).sum(axis=2)
    sum_s2 = (s_foi * s_foi).sum(axis=2)
    del d_foi, s_foi

    det = sum_d2 * F - sum_d * sum_d
    safe = np.abs(det) > 1e-10
    det_safe = np.where(safe, det, 1.0)

    # A degenerate donner (a dead or stuck pixel has no variance over the FOI)
    # leaves the slope undefined.  Fall back to a=0, b=mean(src): y then reduces
    # to the source itself and the pixel degrades to the donner-less dF/F.
    # Dropping the intercept as well would leave y = src + mean(src), i.e. a
    # silently HALVED dF/F at exactly the pixels that are already suspect.
    a = np.where(safe, (F * sum_ds - sum_d * sum_s) / det_safe, 0.0)
    b = np.where(safe, (sum_d2 * sum_s - sum_d * sum_ds) / det_safe, sum_s / F)

    # hemo variance-explained (R^2 of the per-pixel donner->source regression over
    # the FOI window): the fraction of source variance the donner accounts for.
    var_s = sum_s2 * F - sum_s * sum_s
    cov = F * sum_ds - sum_d * sum_s
    denom = det * var_s
    hemovar = np.zeros_like(denom)
    ok = denom > 1e-20  # a flat donner or a flat source explains nothing: R^2 = 0
    np.divide(cov * cov, denom, out=hemovar, where=ok)
    np.clip(hemovar, 0.0, 1.0, out=hemovar)

    mean_src = src.mean(axis=2)
    y = src - (donor_interp * a[:, :, None] + b[:, :, None]) + mean_src[:, :, None]

    N = int(baseline_ignore_initial)
    y_for_base = y[..., N:] if 0 < N < y.shape[-1] else y
    y_base = np.percentile(y_for_base, baseline_percentile, axis=2)
    # Same convention as the donner-less path: a non-positive baseline (a dead
    # pixel, a masked corner) is defined to have zero dF/F.  Dividing by it would
    # put a NaN/inf in dff_{name}.npy, and one NaN pixel takes PCA down with an
    # opaque LinAlgError several stages later.
    base_ok = y_base > 0
    image_df = np.where(
        base_ok[:, :, None],
        -1.0 + y / np.where(base_ok, y_base, 1.0)[:, :, None],
        0.0,
    )
    return image_df, y_base, hemovar


def _interpolate_donor_strip(
    donor_strip: np.ndarray,
    src_times: np.ndarray,
    donor_times: np.ndarray,
) -> np.ndarray:
    """Interpolate donor strip (strip_h, W, Td) to src_times → (strip_h, W, Ts).

    Uses vectorized np.interp per pixel row.
    """
    sh, W, Td = donor_strip.shape
    Ts = src_times.shape[0]
    out = np.empty((sh, W, Ts), dtype=np.float64)
    for r in range(sh):
        for c in range(W):
            out[r, c, :] = np.interp(src_times, donor_times, donor_strip[r, c, :])
    return out


def _exp_detrend(x: np.ndarray) -> np.ndarray:
    """Remove a single-exponential (photobleaching) trend per pixel.

    Port of the MATLAB ``flag_ExpoSub`` path: fit ``a·exp(b·t)`` and subtract it,
    re-centering to the per-pixel mean.  Uses a fast **log-linear** fit
    (OLS of ``log(x)`` on ``t``) as a vectorized approximation of MATLAB's
    nonlinear ``fit(...,'exp1')``.  ``x`` is ``(..., T)`` and must be > 0.
    """
    T = x.shape[-1]
    if T < 3:
        return x
    t = np.arange(T, dtype=np.float64)
    tm = t.mean()
    tc = t - tm
    var_t = float((tc * tc).sum()) or 1.0
    lx = np.log(np.clip(x, 1e-6, None))
    lxm = lx.mean(axis=-1, keepdims=True)
    b = (tc * (lx - lxm)).sum(axis=-1, keepdims=True) / var_t  # (..., 1)
    c = lxm - b * tm
    trend = np.exp(c + b * t)  # a·exp(b·t), broadcast to (..., T)
    return x - trend + x.mean(axis=-1, keepdims=True)


def _rolling_percentile_baseline(
    x: np.ndarray, fps: float, cutoff_sec: float, rank: float
) -> np.ndarray:
    """Slowly-varying baseline F0 per pixel (David Whitney ``baselinePercentileFilter``).

    A rolling-percentile trend over a ``cutoff_sec`` window, lightly smoothed.
    Dividing ``x`` by this yields a ratiometric high-pass (MATLAB
    ``flag_BaselineFilter``).  For speed the trace is decimated to ~1 Hz before
    the percentile filter (as MATLAB does with ``bins``) and linearly upsampled
    back.  ``x`` is ``(..., T)``; returns F0 the same shape (clipped > 0).
    """
    from scipy.interpolate import interp1d
    from scipy.ndimage import percentile_filter, uniform_filter1d

    T = x.shape[-1]
    fps = max(float(fps), 1e-6)
    bins = max(1, int(round(fps)))  # decimate to ~1 Hz (MATLAB uses bins=20 @20fps)
    xd = x[..., ::bins]
    Td = xd.shape[-1]
    fps_d = fps / bins
    win = int(round(cutoff_sec * fps_d))
    win = win + 1 - (win % 2)  # odd
    if Td < 3 or win < 3:
        base_d = np.broadcast_to(
            np.percentile(xd, rank, axis=-1, keepdims=True), xd.shape
        )
    else:
        win = min(win, Td if Td % 2 == 1 else Td - 1)
        size = tuple([1] * (xd.ndim - 1) + [win])
        base_d = percentile_filter(xd, percentile=float(rank), size=size, mode="nearest")
        base_d = uniform_filter1d(base_d, size=win, axis=-1, mode="nearest")
    if Td == T:
        base = base_d
    elif Td < 2:
        # A recording shorter than one decimation bin leaves a single sample, and
        # interp1d through one point extrapolates to NaN -- a 100%-NaN dF/F.
        # One sample IS the baseline: hold it.
        base = np.broadcast_to(base_d[..., :1], x.shape)
    else:
        xp = np.arange(Td, dtype=np.float64) * bins
        base = interp1d(
            xp, base_d, axis=-1, kind="linear", bounds_error=False,
            fill_value="extrapolate",
        )(np.arange(T, dtype=np.float64))
    return np.clip(base, 1e-6, None)


def _preprocess_trace(
    x: np.ndarray, *, detrend: bool, highpass: bool,
    fps_channel: float, cutoff_sec: float, rank: float,
) -> np.ndarray:
    """Optional pre-regression conditioning of a src/donor strip (``(..., T)``).

    Order (both may be on): exponential detrend, then rolling-percentile high-pass.
    """
    if detrend:
        x = _exp_detrend(x)
    if highpass:
        x = x / _rolling_percentile_baseline(x, fps_channel, cutoff_sec, rank)
    return x


def wfci_corrected_df_vectorized(
    src: np.ndarray,
    donor: np.ndarray,
    foi: np.ndarray,
    *,
    src_times: np.ndarray | None = None,
    donor_times: np.ndarray | None = None,
    strip_h: int = 32,
    baseline_percentile: float = 5.0,
    baseline_ignore_initial: int = 0,
    detrend: bool = False,
    highpass: bool = False,
    highpass_cutoff_sec: float = 120.0,
    highpass_rank: float = 50.0,
    fps_channel: float = 1.0,
    return_hemovar: bool = False,
    progress=None,
) -> tuple[np.ndarray, ...]:
    """Vectorized WFCI correction with row-strip memory control.

    Parameters
    ----------
    src : (H, W, Ts) float64 – source channel
    donor : (H, W, Td) float64 – donner channel
    foi : (F,) int – frame-of-interest indices for regression (into src time axis)
    src_times : (Ts,) float64 | None – temporal positions of src frames
    donor_times : (Td,) float64 | None – temporal positions of donor frames
        When both are None, uses legacy 2x interpolation (assumes Ts == Td).
        When provided, donor is interpolated to src_times via np.interp.
    strip_h : int – rows per strip (controls peak memory)
    baseline_percentile : float – per-pixel baseline percentile (0-100)
    baseline_ignore_initial : int – leading frames excluded from the baseline
    detrend : bool – remove an exponential trend from src AND raw donor before the
        regression (MATLAB ``flag_ExpoSub``)
    highpass : bool – rolling-percentile high-pass, applied after detrend to src
        AND raw donor (MATLAB ``flag_BaselineFilter``)
    highpass_cutoff_sec / highpass_rank / fps_channel – its window (s), rank
        (0-100) and the per-channel frame rate the window is expressed in
    return_hemovar : bool – also return the per-pixel R^2 map (makes it a 3-tuple)
    progress : callable(rows_done, H) | None – called after each strip. The strip
        loop is the only place long enough to report from: one call per strip is
        the finest progress this correction can offer without splitting the
        per-pixel regression itself.

    Returns
    -------
    image_df : (H, W, Ts) – corrected dF/F
    baseimage_df : (H, W) – per-pixel ``baseline_percentile``-th percentile baseline
    hemovar_map : (H, W) – ONLY when ``return_hemovar=True``: the fraction of
        source variance the donner regression explains (R^2; the WidefieldImager
        ``hemoVar`` analogue). preprocess unpacks three values when ``hemovar_qc``
        is on.
    """
    H, W, Ts = src.shape
    image_df = np.empty_like(src)
    baseimage_df = np.empty((H, W), dtype=np.float64)
    hemovar_map = np.empty((H, W), dtype=np.float64)

    foi_safe = foi[(foi >= 0) & (foi < Ts)]
    if foi_safe.size < 2:
        foi_safe = np.arange(Ts)

    use_legacy = src_times is None and donor_times is None

    if use_legacy:
        # Legacy 2x interpolation (backward compat with 2ch BL/UV)
        Td = donor.shape[2]
        x_interp_odd = np.linspace(0, Td - 1, Td * 2)[1::2]
        idx_lo = np.floor(x_interp_odd).astype(np.intp)
        np.clip(idx_lo, 0, Td - 2, out=idx_lo)
        frac = x_interp_odd - idx_lo

    do_pre = detrend or highpass

    for r0 in range(0, H, strip_h):
        r1 = min(r0 + strip_h, H)
        src_strip = src[r0:r1]
        donor_strip = donor[r0:r1]

        # Optional pre-regression conditioning (MATLAB flag_ExpoSub / flag_BaselineFilter):
        # detrend then high-pass, applied to source AND raw donor before interpolation.
        if do_pre:
            src_strip = _preprocess_trace(
                src_strip, detrend=detrend, highpass=highpass,
                fps_channel=fps_channel, cutoff_sec=highpass_cutoff_sec, rank=highpass_rank,
            )
            donor_strip = _preprocess_trace(
                donor_strip, detrend=detrend, highpass=highpass,
                fps_channel=fps_channel, cutoff_sec=highpass_cutoff_sec, rank=highpass_rank,
            )

        if use_legacy:
            # Legacy: inline interpolation via index arithmetic
            donor_interp = (
                donor_strip[:, :, idx_lo] * (1.0 - frac)
                + donor_strip[:, :, idx_lo + 1] * frac
            )
        else:
            donor_interp = _interpolate_donor_strip(
                donor_strip, src_times, donor_times
            )

        image_df[r0:r1], baseimage_df[r0:r1], hemovar_map[r0:r1] = _wfci_strip(
            src_strip, donor_interp, foi_safe,
            baseline_percentile, baseline_ignore_initial,
        )
        if progress is not None:
            progress(r1, H)

    if return_hemovar:
        return image_df, baseimage_df, hemovar_map
    return image_df, baseimage_df


def plot_hemovar_map(hemovar, name, *, brain_mask=None):
    """QC figure — per-pixel hemo variance-explained (R²) map for a channel group.

    ``hemovar`` is ``(H, W)`` in [0, 1]: the fraction of source variance the
    donner regression explains (the ASoVi analogue of WidefieldImager's
    ``hemoVar``).  Returns a Matplotlib ``Figure``; caller owns saving/closing.
    """
    import matplotlib.pyplot as plt

    disp = np.asarray(hemovar, dtype=np.float64)
    if brain_mask is not None:
        disp = np.where(np.asarray(brain_mask, dtype=bool), disp, np.nan)
    med = float(np.nanmedian(disp)) if np.isfinite(disp).any() else 0.0

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(disp, cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_title(f"Hemo variance explained (R²) — {name}\nmedian = {med:.2f}")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("fraction of source variance explained by donner")
    fig.tight_layout()
    return fig
