"""Correctness tests for the vectorized (complex64) batch DFT registration.

Verifies the batched register path actually aligns frames to the reference and
that the fast complex64 path stays within tolerance of the exact complex128
path (the accepted ~0.01 px trade-off).
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from asvimg.registration.torch import DftRegistrator


def _shift_image(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Exact subpixel shift of a 2D image via an FFT phase ramp."""
    H, W = img.shape
    ft = np.fft.fft2(img)
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.fftfreq(W)[None, :]
    return np.abs(np.fft.ifft2(ft * np.exp(-2j * np.pi * (ky * dy + kx * dx))))


class TestVectorizedRegistration(unittest.TestCase):
    def test_registration_aligns_frames(self) -> None:
        """The registered frames must be far closer to the reference than the
        raw shifted frames (central crop to avoid FFT edge wrap)."""
        rng = np.random.default_rng(0)
        H = W = 128
        tmpl = np.clip(rng.normal(1000, 200, (H, W)), 0, None)
        shifts = [(3.0, -2.0), (1.5, 2.5), (-4.0, 1.0), (0.4, -0.6)]
        frames = np.stack([_shift_image(tmpl, dy, dx) for dy, dx in shifts])

        reg = DftRegistrator(torch.from_numpy(tmpl), usfac=100)
        _res, regd = reg.register_batch(
            torch.from_numpy(frames), return_registered=True
        )
        regd = regd.numpy()

        c = slice(20, -20)
        for i in range(len(frames)):
            before = float(np.mean((frames[i][c, c] - tmpl[c, c]) ** 2))
            after = float(np.mean((regd[i][c, c] - tmpl[c, c]) ** 2))
            self.assertLess(after, before * 0.2, msg=f"frame {i}: {after} vs {before}")

    def test_complex64_matches_complex128(self) -> None:
        """The fast complex64 path stays within ~0.05 px of the exact path."""
        rng = np.random.default_rng(1)
        H = W = 96
        tmpl = np.clip(rng.normal(1000, 200, (H, W)), 0, None)
        frames = np.stack([
            _shift_image(tmpl, rng.uniform(-5, 5), rng.uniform(-5, 5))
            for _ in range(8)
        ])
        reg = DftRegistrator(torch.from_numpy(tmpl), usfac=200)
        r64, _ = reg.register_batch(torch.from_numpy(frames), cdtype=torch.complex64)
        r128, _ = reg.register_batch(torch.from_numpy(frames), cdtype=torch.complex128)
        for a, b in zip(r64, r128):
            self.assertLess(abs(a.row_shift - b.row_shift), 0.05)
            self.assertLess(abs(a.col_shift - b.col_shift), 0.05)

    def test_batch_chunking_invariant(self) -> None:
        """Per-frame results are independent of how frames are chunked."""
        rng = np.random.default_rng(2)
        H = W = 80
        tmpl = np.clip(rng.normal(1000, 200, (H, W)), 0, None)
        frames = np.stack([
            _shift_image(tmpl, rng.uniform(-4, 4), rng.uniform(-4, 4))
            for _ in range(10)
        ])
        reg = DftRegistrator(torch.from_numpy(tmpl), usfac=100)
        big, _ = reg.register_batch(torch.from_numpy(frames), chunk=100)
        small, _ = reg.register_batch(torch.from_numpy(frames), chunk=3)
        for a, b in zip(big, small):
            self.assertEqual(a.row_shift, b.row_shift)
            self.assertEqual(a.col_shift, b.col_shift)


if __name__ == "__main__":
    unittest.main()
