"""Selectable N-D spatiotemporal filters (mean / median / gaussian).

Shared by the preprocess ``filter_xyt`` pass and the post-annotation filter so
both offer the same kinds.  ``size`` is a per-axis window given in the array's
own axis order; a window of 1 (per axis) is a no-op for every kind, and for the
gaussian the window is mapped to ``sigma = (size - 1) / 2`` so the three kinds
have a comparable footprint for the same ``size``.
"""

from __future__ import annotations

import numpy as np

FILTER_KINDS = ("mean", "median", "gaussian")


def apply_xyt_filter(arr: np.ndarray, size, kind: str = "mean") -> np.ndarray:
    """Filter ``arr`` with per-axis window ``size`` using ``kind``.

    kind: ``"mean"`` (uniform/box), ``"median"``, or ``"gaussian"``.  Returns an
    array the same shape as ``arr`` (float32 when filtering actually runs).
    """
    kind = (kind or "mean").lower()
    if kind not in FILTER_KINDS:
        raise ValueError(f"filter kind must be one of {FILTER_KINDS}, got {kind!r}")
    size = tuple(int(max(1, s)) for s in size)
    if all(s <= 1 for s in size):
        return arr  # window of 1 on every axis -> nothing to do

    a = np.asarray(arr, dtype=np.float32)
    if kind == "median":
        from scipy.ndimage import median_filter

        return median_filter(a, size=size)
    if kind == "gaussian":
        from scipy.ndimage import gaussian_filter

        sigma = tuple(max(0.0, (s - 1) / 2.0) for s in size)
        return gaussian_filter(a, sigma=sigma)
    from scipy.ndimage import uniform_filter

    return uniform_filter(a, size=size)
