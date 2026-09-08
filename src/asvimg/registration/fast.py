"""Numba-optimized DFT registration with JIT kernels and caching.

Key optimizations over vanilla NumPy:
1. Numba JIT for dftups kernel construction (fused exp loop)
2. Pre-computed rg00 (reference auto-correlation, same every frame)
3. Cached shift grids (meshgrid computed once, not per-frame)
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

from .numpy import DftRegistrationResult


@nb.njit(cache=True)
def _dftups_kernels(
    nr: int, nc: int,
    nor: int, noc: int,
    usfac: int,
    roff: float, coff: float,
    col_idx: np.ndarray,
    row_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build kernr and kernc matrices for dftups (fused exp, Numba JIT)."""
    kernc = np.empty((nc, noc), dtype=np.complex128)
    phase_c = -2.0 * math.pi / (nc * usfac)
    for i in range(nc):
        for j in range(noc):
            angle = phase_c * col_idx[i] * (j - coff)
            kernc[i, j] = math.cos(angle) + 1j * math.sin(angle)

    kernr = np.empty((nor, nr), dtype=np.complex128)
    phase_r = -2.0 * math.pi / (nr * usfac)
    for i in range(nor):
        for j in range(nr):
            angle = phase_r * (i - roff) * row_idx[j]
            kernr[i, j] = math.cos(angle) + 1j * math.sin(angle)

    return kernr, kernc


def _ifftshift_indices(n: int) -> np.ndarray:
    return np.fft.ifftshift(np.arange(n)) - (n // 2)


def dftups_fast(
    input_ft: np.ndarray,
    nor: int, noc: int,
    usfac: int,
    roff: float, coff: float,
    col_idx: np.ndarray,
    row_idx: np.ndarray,
) -> np.ndarray:
    """dftups with pre-computed index arrays and Numba kernel construction."""
    kernr, kernc = _dftups_kernels(
        input_ft.shape[0], input_ft.shape[1],
        nor, noc, usfac, roff, coff,
        col_idx, row_idx,
    )
    return kernr @ input_ft @ kernc


def _dftregistration_fast(
    buf1ft: np.ndarray,
    buf2ft: np.ndarray,
    usfac: int,
    col_idx: np.ndarray,
    row_idx: np.ndarray,
    rg00_precomputed: complex | None,
) -> tuple[float, float, float, float]:
    """Core registration for usfac > 2, optimized."""
    m, n = buf1ft.shape

    mlarge, nlarge = m * 2, n * 2
    cc = np.zeros((mlarge, nlarge), dtype=np.complex128)
    r0 = m + 1 - (m // 2) - 1
    r1 = m + 1 + ((m - 1) // 2)
    c0 = n + 1 - (n // 2) - 1
    c1 = n + 1 + ((n - 1) // 2)
    cc[r0:r1, c0:c1] = np.fft.fftshift(buf1ft) * np.conj(np.fft.fftshift(buf2ft))
    cc = np.fft.ifft2(np.fft.ifftshift(cc))

    flat_idx = np.argmax(np.abs(cc))
    rloc, cloc = divmod(int(flat_idx), nlarge)
    cc_max = cc[rloc, cloc]

    md2 = mlarge // 2
    nd2 = nlarge // 2
    row_shift = (rloc - mlarge if rloc > md2 else rloc) / 2.0
    col_shift = (cloc - nlarge if cloc > nd2 else cloc) / 2.0

    row_shift = round(row_shift * usfac) / usfac
    col_shift = round(col_shift * usfac) / usfac
    dftshift = math.floor(math.ceil(usfac * 1.5) / 2.0)
    ups_size = int(math.ceil(usfac * 1.5))

    cross = buf2ft * np.conj(buf1ft)
    cc2 = np.conj(
        dftups_fast(
            cross, ups_size, ups_size, usfac,
            dftshift - row_shift * usfac,
            dftshift - col_shift * usfac,
            col_idx, row_idx,
        )
    ) / (md2 * nd2 * usfac * usfac)

    flat_idx2 = np.argmax(np.abs(cc2))
    rloc2, cloc2 = divmod(int(flat_idx2), ups_size)
    cc_max = cc2[rloc2, cloc2]

    rg00 = rg00_precomputed

    rf00 = dftups_fast(
        buf2ft * np.conj(buf2ft), 1, 1, usfac, 0.0, 0.0,
        col_idx, row_idx,
    ).squeeze() / (md2 * nd2 * usfac * usfac)

    row_shift += (rloc2 - dftshift) / usfac
    col_shift += (cloc2 - dftshift) / usfac

    error = float(np.sqrt(np.abs(1.0 - cc_max * np.conj(cc_max) / (rg00 * rf00))).real)
    diffphase = float(np.arctan2(np.imag(cc_max), np.real(cc_max)))

    return error, diffphase, float(row_shift), float(col_shift)


class DftRegistratorNumba:
    """Reusable registrator with pre-computed reference data.

    Usage::

        reg = DftRegistratorNumba(template, usfac=500)
        results = reg.register_frames(frames)  # frames: (H, W, N)
    """

    def __init__(self, reference: np.ndarray, usfac: int = 500):
        self.usfac = usfac
        nr, nc = reference.shape
        self._ref_ft = np.fft.fft2(reference.astype(np.float64))

        self._col_idx = _ifftshift_indices(nc).astype(np.float64)
        self._row_idx = _ifftshift_indices(nr).astype(np.float64)

        md2 = nr
        nd2 = nc
        self._rg00 = complex(
            dftups_fast(
                self._ref_ft * np.conj(self._ref_ft),
                1, 1, usfac, 0.0, 0.0,
                self._col_idx, self._row_idx,
            ).squeeze()
        ) / (md2 * nd2 * usfac * usfac)

        nr_grid = np.fft.ifftshift(np.arange(-nr // 2, math.ceil(nr / 2.0)))
        nc_grid = np.fft.ifftshift(np.arange(-nc // 2, math.ceil(nc / 2.0)))
        self._nc_mesh, self._nr_mesh = np.meshgrid(nc_grid, nr_grid)

        # Warm up Numba JIT
        _dftups_kernels(nr, nc, 1, 1, usfac, 0.0, 0.0, self._col_idx, self._row_idx)

    def register_frames(
        self,
        frames: np.ndarray,
        return_registered: bool = False,
    ) -> tuple[list[DftRegistrationResult], np.ndarray | None]:
        """Register frames (H, W, N) against reference."""
        n = frames.shape[2]
        nr, nc = frames.shape[0], frames.shape[1]
        results: list[DftRegistrationResult] = []
        registered = np.empty_like(frames) if return_registered else None

        for i in range(n):
            frame_ft = np.fft.fft2(frames[:, :, i])
            error, diffphase, row_shift, col_shift = _dftregistration_fast(
                self._ref_ft, frame_ft, self.usfac,
                self._col_idx, self._row_idx,
                self._rg00,
            )
            results.append(DftRegistrationResult(error, diffphase, row_shift, col_shift))

            if return_registered:
                greg = frame_ft * np.exp(
                    1j * 2 * np.pi * (
                        -row_shift * self._nr_mesh / nr
                        - col_shift * self._nc_mesh / nc
                    )
                )
                greg *= np.exp(1j * diffphase)
                registered[:, :, i] = np.abs(np.fft.ifft2(greg))

        return results, registered
