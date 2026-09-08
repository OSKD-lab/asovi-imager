"""Scientific-correctness tests for PCA + spatial ICA (denoising core).

Validates the redesigned decomposition: explained-variance ratios, FastICA
whitening/independence, blind source separation of known spatial sources,
the exact reconstruction identity, component exclusion, and that the requested
component count is honoured.
"""

from __future__ import annotations

import unittest

import numpy as np

from asvimg import compute_ica, compute_pca, reconstruct


def _make_sources(H=30, W=32, T=240, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]

    def gauss(cy, cx, sg):
        g = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sg ** 2)))
        return (g - g.mean()).ravel()

    S = np.stack([gauss(8, 8, 4), gauss(20, 24, 5), gauss(15, 5, 3)], axis=1)  # (P,3)
    t = np.arange(T)
    Tt = np.stack([
        np.sin(2 * np.pi * t / 30),
        np.sign(np.sin(2 * np.pi * t / 17)),
        rng.standard_normal(T),
    ], axis=0).astype(float)
    Tt -= Tt.mean(axis=1, keepdims=True)
    X = 100.0 + S @ Tt + 0.01 * rng.standard_normal((H * W, T))
    return X.reshape(H, W, T), S, Tt


class TestPCA(unittest.TestCase):
    def test_variance_ratio_sums_to_one_full_rank(self) -> None:
        Xim, _, _ = _make_sources()
        T = Xim.shape[2]
        pca = compute_pca(Xim, n_components=T)
        self.assertAlmostEqual(float(pca.variance_ratio.sum()), 1.0, places=5)

    def test_variance_ratio_descending_and_bounded(self) -> None:
        Xim, _, _ = _make_sources()
        pca = compute_pca(Xim, n_components=10)
        vr = pca.variance_ratio
        self.assertLessEqual(float(vr.sum()), 1.0 + 1e-6)
        self.assertTrue(np.all(np.diff(vr) <= 1e-9))
        # 3 sources dominate
        self.assertGreater(float(vr[:3].sum()), 0.99)


class TestSpatialICA(unittest.TestCase):
    def test_n_components_honored(self) -> None:
        Xim, _, _ = _make_sources()
        pca = compute_pca(Xim, n_components=10)
        ica = compute_ica(pca, n_ica_components=3, random_state=0)
        self.assertEqual(ica.spatial.shape[0], 3)
        self.assertEqual(ica.temporal.shape, (3, Xim.shape[2]))

    def test_spatial_ics_are_decorrelated(self) -> None:
        Xim, _, _ = _make_sources()
        pca = compute_pca(Xim, n_components=3)
        ica = compute_ica(pca, n_ica_components=3, random_state=0)
        sp = ica.spatial.reshape(3, -1)
        C = np.corrcoef(sp)
        offdiag = C[~np.eye(3, dtype=bool)]
        self.assertLess(float(np.abs(offdiag).max()), 0.1)

    def test_blind_source_separation(self) -> None:
        Xim, S, _ = _make_sources()
        pca = compute_pca(Xim, n_components=3)
        ica = compute_ica(pca, n_ica_components=3, random_state=0)
        sp = ica.spatial.reshape(3, -1)

        def best_corr(a):
            a = a - a.mean()
            return max(abs(np.corrcoef(a, b - b.mean())[0, 1]) for b in sp)

        for k in range(3):
            self.assertGreater(best_corr(S[:, k]), 0.95)

    def test_reconstruction_identity(self) -> None:
        # smooth_sigma=0 → recon(exclude=[]) reproduces the rank-k input
        Xim, _, _ = _make_sources()
        pca = compute_pca(Xim, n_components=3, smooth_sigma=0.0)
        ica = compute_ica(pca, n_ica_components=3, random_state=0)
        rec = reconstruct(ica, Xim, exclude=[])
        rel = np.linalg.norm(rec - Xim) / np.linalg.norm(Xim)
        self.assertLess(float(rel), 1e-2)

    def test_exclude_removes_only_that_component(self) -> None:
        Xim, S, _ = _make_sources()
        P, T = S.shape[0], Xim.shape[2]
        pca = compute_pca(Xim, n_components=3, smooth_sigma=0.0)
        ica = compute_ica(pca, n_ica_components=3, random_state=0)
        sp = ica.spatial.reshape(3, -1)
        k0 = int(np.argmax([
            abs(np.corrcoef(S[:, 0] - S[:, 0].mean(), sp[i] - sp[i].mean())[0, 1])
            for i in range(3)
        ]))
        rec0 = reconstruct(ica, Xim, exclude=[]).reshape(P, T)
        rec1 = reconstruct(ica, Xim, exclude=[k0]).reshape(P, T)
        removed = rec0 - rec1
        u = S[:, 0] / np.linalg.norm(S[:, 0])
        frac = np.linalg.norm(u @ removed) / max(np.linalg.norm(removed), 1e-12)
        self.assertGreater(float(frac), 0.9)

    def test_exclude_all_returns_mean(self) -> None:
        Xim, _, _ = _make_sources()
        pca = compute_pca(Xim, n_components=3)
        ica = compute_ica(pca, n_ica_components=3, random_state=0)
        rec = reconstruct(ica, Xim, exclude=[0, 1, 2])
        self.assertTrue(np.allclose(rec[:, :, 0], ica.mean_image))

    def test_smoothing_does_not_break_spatial_recovery(self) -> None:
        # temporal smoothing must not affect spatial-IC identification (uses U)
        Xim, S, _ = _make_sources()
        pca = compute_pca(Xim, n_components=3, smooth_sigma=5.0)
        ica = compute_ica(pca, n_ica_components=3, random_state=0)
        sp = ica.spatial.reshape(3, -1)
        rec = reconstruct(ica, Xim, exclude=[])
        self.assertEqual(rec.shape, Xim.shape)
        self.assertTrue(np.all(np.isfinite(rec)))

        def best_corr(a):
            a = a - a.mean()
            return max(abs(np.corrcoef(a, b - b.mean())[0, 1]) for b in sp)

        for k in range(3):
            self.assertGreater(best_corr(S[:, k]), 0.95)


if __name__ == "__main__":
    unittest.main()
