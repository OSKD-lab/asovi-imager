"""Selectable mean/median/gaussian filters + config wiring."""

from __future__ import annotations

import unittest

import numpy as np

from asvimg import PipelineConfig
from asvimg.filters import FILTER_KINDS, apply_xyt_filter


def _cfg(**kw) -> PipelineConfig:
    return PipelineConfig(input_dir="x", output_dir="y", exp_name="S", **kw)


class TestFilters(unittest.TestCase):
    def test_noop_for_window_one(self) -> None:
        a = np.random.default_rng(0).random((4, 5, 6)).astype(np.float32)
        for kind in FILTER_KINDS:
            self.assertTrue(np.allclose(apply_xyt_filter(a, (1, 1, 1), kind), a))

    def test_mean_matches_uniform_filter(self) -> None:
        from scipy.ndimage import uniform_filter

        a = np.random.default_rng(1).random((5, 6, 7)).astype(np.float32)
        self.assertTrue(
            np.allclose(
                apply_xyt_filter(a, (3, 3, 3), "mean"),
                uniform_filter(a, size=(3, 3, 3)),
            )
        )

    def test_median_suppresses_outlier(self) -> None:
        from scipy.ndimage import median_filter

        a = np.ones((5, 5, 5), np.float32)
        a[2, 2, 2] = 100.0  # single spike
        med = apply_xyt_filter(a, (3, 3, 3), "median")
        self.assertTrue(np.allclose(med, median_filter(a, size=(3, 3, 3))))
        self.assertAlmostEqual(float(med[2, 2, 2]), 1.0)  # median kills the spike

    def test_gaussian_spreads_impulse(self) -> None:
        a = np.zeros((7, 7, 7), np.float32)
        a[3, 3, 3] = 1.0
        g = apply_xyt_filter(a, (3, 3, 3), "gaussian")
        self.assertLess(float(g[3, 3, 3]), 1.0)
        self.assertGreater(float(g[3, 3, 3]), 0.0)
        self.assertGreater(float(g[2, 3, 3]), 0.0)  # energy spread to neighbours

    def test_invalid_kind_raises(self) -> None:
        with self.assertRaises(ValueError):
            apply_xyt_filter(np.zeros((3, 3, 3)), (3, 3, 3), "box")


class TestFilterConfig(unittest.TestCase):
    def test_valid_kinds_accepted(self) -> None:
        _cfg(filter_xyt_kind="median", post_annotation_filter_kind="gaussian")

    def test_bad_kinds_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _cfg(filter_xyt_kind="box")
        with self.assertRaises(ValueError):
            _cfg(post_annotation_filter_kind="nope")

    def test_post_annotation_process_applies_spatial_filter(self) -> None:
        from asvimg.annotation import _post_annotation_process

        cfg = _cfg(post_annotation_filter_xyt=[3, 3, 1], post_annotation_filter_kind="mean")
        xyt = np.zeros((7, 7, 4), np.float32)
        xyt[3, 3, :] = 1.0  # a bright column
        out = _post_annotation_process(cfg, xyt)
        self.assertLess(float(out[3, 3, 0]), 1.0)     # 3x3 mean spread the spike
        self.assertGreater(float(out[2, 3, 0]), 0.0)  # neighbour picked up energy


if __name__ == "__main__":
    unittest.main()
