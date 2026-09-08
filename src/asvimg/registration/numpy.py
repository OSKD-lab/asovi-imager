"""NumPy (vanilla) DFT registration — Guizar-Sicairos et al., Opt. Lett. 2008."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DftRegistrationResult:
    error: float
    diffphase: float
    row_shift: float
    col_shift: float


def _ifftshift_indices(n: int) -> np.ndarray:
    return np.fft.ifftshift(np.arange(n)) - (n // 2)


def dftups(
    input_ft: np.ndarray,
    nor: int | None = None,
    noc: int | None = None,
    usfac: int = 1,
    roff: float = 0.0,
    coff: float = 0.0,
) -> np.ndarray:
    nr, nc = input_ft.shape
    if nor is None:
        nor = nr
    if noc is None:
        noc = nc

    kernc = np.exp(
        (-1j * 2 * np.pi / (nc * usfac))
        * np.outer(_ifftshift_indices(nc), (np.arange(noc) - coff))
    )
    kernr = np.exp(
        (-1j * 2 * np.pi / (nr * usfac))
        * np.outer((np.arange(nor) - roff), _ifftshift_indices(nr))
    )
    return kernr @ input_ft @ kernc


def dftregistration(
    buf1ft: np.ndarray,
    buf2ft: np.ndarray,
    usfac: int = 1,
) -> tuple[DftRegistrationResult, np.ndarray]:
    if usfac == 0:
        cc_max = np.sum(buf1ft * np.conj(buf2ft))
        rfzero = np.sum(np.abs(buf1ft) ** 2)
        rgzero = np.sum(np.abs(buf2ft) ** 2)
        error = np.sqrt(np.abs(1.0 - cc_max * np.conj(cc_max) / (rgzero * rfzero)))
        diffphase = np.arctan2(np.imag(cc_max), np.real(cc_max))
        result = DftRegistrationResult(float(error), float(diffphase), 0.0, 0.0)
        return result, buf2ft * np.exp(1j * diffphase)

    m, n = buf1ft.shape

    if usfac == 1:
        cc = np.fft.ifft2(buf1ft * np.conj(buf2ft))
        rloc, cloc = np.unravel_index(np.argmax(np.abs(cc)), cc.shape)
        cc_max = cc[rloc, cloc]
        rfzero = np.sum(np.abs(buf1ft) ** 2) / (m * n)
        rgzero = np.sum(np.abs(buf2ft) ** 2) / (m * n)
        error = np.sqrt(np.abs(1.0 - cc_max * np.conj(cc_max) / (rgzero * rfzero)))
        diffphase = np.arctan2(np.imag(cc_max), np.real(cc_max))

        md2 = m // 2
        nd2 = n // 2
        row_shift = rloc - m if rloc > md2 else rloc
        col_shift = cloc - n if cloc > nd2 else cloc
    else:
        mlarge = m * 2
        nlarge = n * 2
        cc = np.zeros((mlarge, nlarge), dtype=np.complex128)

        r0 = m + 1 - (m // 2) - 1
        r1 = m + 1 + ((m - 1) // 2)
        c0 = n + 1 - (n // 2) - 1
        c1 = n + 1 + ((n - 1) // 2)
        cc[r0:r1, c0:c1] = np.fft.fftshift(buf1ft) * np.conj(np.fft.fftshift(buf2ft))
        cc = np.fft.ifft2(np.fft.ifftshift(cc))

        rloc, cloc = np.unravel_index(np.argmax(np.abs(cc)), cc.shape)
        cc_max = cc[rloc, cloc]

        md2 = cc.shape[0] // 2
        nd2 = cc.shape[1] // 2
        row_shift = (rloc - cc.shape[0] if rloc > md2 else rloc) / 2.0
        col_shift = (cloc - cc.shape[1] if cloc > nd2 else cloc) / 2.0

        if usfac > 2:
            row_shift = np.round(row_shift * usfac) / usfac
            col_shift = np.round(col_shift * usfac) / usfac
            dftshift = int(np.floor(np.ceil(usfac * 1.5) / 2.0))
            cc = np.conj(
                dftups(
                    buf2ft * np.conj(buf1ft),
                    int(np.ceil(usfac * 1.5)),
                    int(np.ceil(usfac * 1.5)),
                    usfac,
                    dftshift - row_shift * usfac,
                    dftshift - col_shift * usfac,
                )
            ) / (md2 * nd2 * usfac * usfac)

            rloc, cloc = np.unravel_index(np.argmax(np.abs(cc)), cc.shape)
            cc_max = cc[rloc, cloc]
            rg00 = dftups(buf1ft * np.conj(buf1ft), 1, 1, usfac) / (
                md2 * nd2 * usfac * usfac
            )
            rf00 = dftups(buf2ft * np.conj(buf2ft), 1, 1, usfac) / (
                md2 * nd2 * usfac * usfac
            )
            rg00 = np.asarray(rg00).squeeze()
            rf00 = np.asarray(rf00).squeeze()
            row_shift = row_shift + (rloc - dftshift) / usfac
            col_shift = col_shift + (cloc - dftshift) / usfac
        else:
            rg00 = np.sum(buf1ft * np.conj(buf1ft)) / (m * n)
            rf00 = np.sum(buf2ft * np.conj(buf2ft)) / (m * n)

        error = np.sqrt(np.abs(1.0 - cc_max * np.conj(cc_max) / (rg00 * rf00)))
        diffphase = np.arctan2(np.imag(cc_max), np.real(cc_max))

        if md2 == 1:
            row_shift = 0.0
        if nd2 == 1:
            col_shift = 0.0

    nr, nc = buf2ft.shape
    nr_grid = np.fft.ifftshift(np.arange(-nr // 2, int(np.ceil(nr / 2.0))))
    nc_grid = np.fft.ifftshift(np.arange(-nc // 2, int(np.ceil(nc / 2.0))))
    nc_mesh, nr_mesh = np.meshgrid(nc_grid, nr_grid)
    greg = buf2ft * np.exp(
        1j * 2 * np.pi * (-row_shift * nr_mesh / nr - col_shift * nc_mesh / nc)
    )
    greg = greg * np.exp(1j * diffphase)

    result = DftRegistrationResult(
        float(error), float(diffphase), float(row_shift), float(col_shift)
    )
    return result, greg


def dft_reconstruct(img_ft: np.ndarray, result: DftRegistrationResult) -> np.ndarray:
    nr, nc = img_ft.shape
    nr_grid = np.fft.ifftshift(np.arange(-nr // 2, int(np.ceil(nr / 2.0))))
    nc_grid = np.fft.ifftshift(np.arange(-nc // 2, int(np.ceil(nc / 2.0))))
    nc_mesh, nr_mesh = np.meshgrid(nc_grid, nr_grid)
    registered = img_ft * np.exp(
        1j
        * 2
        * np.pi
        * (-result.row_shift * nr_mesh / nr - result.col_shift * nc_mesh / nc)
    )
    registered = registered * np.exp(1j * result.diffphase)
    return np.abs(np.fft.ifft2(registered))


def dft_reconstruct3d(
    frames: np.ndarray, results: list[DftRegistrationResult]
) -> np.ndarray:
    out = np.zeros_like(frames, dtype=np.float64)
    for i in range(frames.shape[2]):
        out[:, :, i] = dft_reconstruct(np.fft.fft2(frames[:, :, i]), results[i])
    return out


def dft_register_single(
    reference: np.ndarray, target: np.ndarray, usfac: int = 500
) -> tuple[np.ndarray, DftRegistrationResult]:
    result, greg = dftregistration(
        np.fft.fft2(reference), np.fft.fft2(target), usfac=usfac
    )
    return np.abs(np.fft.ifft2(greg)), result
