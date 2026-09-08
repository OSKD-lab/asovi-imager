from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

from asvimg import (
    PipelineConfig,
    default_output_dir,
    infer_exp_name,
    save_payload,
    wfci_corrected_df,
    wfci_corrected_df_vectorized,
)
from asvimg.preprocess import PreprocessRunner


class TestPipeline01Helpers(unittest.TestCase):
    def test_default_output_dir(self) -> None:
        root = Path("Analysis/_sampleData01")
        self.assertEqual(
            default_output_dir(root, "mat"),
            root / "asi" / "mat",
        )

    def test_runner_uses_default_output_dir_when_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = PipelineConfig(input_dir=tmpdir, output_dir=None, output_format="h5")
            runner = PreprocessRunner(cfg)
            self.assertEqual(runner.output_dir, Path(tmpdir) / "asi" / "h5")

    def test_infer_exp_name(self) -> None:
        self.assertEqual(
            infer_exp_name("timelapse_Sample_MMStack_Default.ome.tif"), "_Sample"
        )
        self.assertEqual(infer_exp_name("abc.tif"), "abc")

    def test_wfci_corrected_df_shapes(self) -> None:
        src = np.linspace(100, 200, 50)
        donor = np.linspace(50, 150, 50)
        foi = np.arange(10, 40)
        d_f, y, y_base, subt = wfci_corrected_df(src, donor, foi)

        self.assertEqual(d_f.shape, src.shape)
        self.assertEqual(y.shape, src.shape)
        self.assertEqual(y_base.shape, src.shape)
        self.assertEqual(subt.shape, src.shape)

    def test_save_payload_for_all_formats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            payload = {
                "imageBL": np.zeros((2, 2, 2), dtype=np.uint16),
                "flag_LineSub": 1,
                "imageSize": np.array([2, 2, 2]),
            }

            save_payload(out / "test_mat", payload, "mat")
            self.assertTrue((out / "test_mat.mat").exists())
            mat = loadmat(out / "test_mat.mat")
            self.assertIn("imageBL", mat)

            save_payload(out / "test_npy", payload, "npy")
            self.assertTrue((out / "test_npy.npy").exists())
            npy = np.load(out / "test_npy.npy", allow_pickle=True).item()
            self.assertIn("imageBL", npy)

            save_payload(out / "test_h5", payload, "h5")
            self.assertTrue((out / "test_h5.h5").exists())
            with h5py.File(out / "test_h5.h5", "r") as h5f:
                self.assertIn("imageBL", h5f)


    def test_vectorized_matches_scalar(self) -> None:
        """Verify vectorized WFCI matches per-pixel version within float64 tolerance."""
        H, W, T = 4, 5, 100
        rng = np.random.default_rng(42)
        src = rng.uniform(50, 200, (H, W, T))
        donor = rng.uniform(30, 150, (H, W, T))
        foi = np.arange(20, 90)

        # Vectorized
        df_vec, base_vec = wfci_corrected_df_vectorized(src, donor, foi)

        # Per-pixel reference
        df_ref = np.zeros_like(src)
        base_ref = np.zeros((H, W))
        for i in range(H):
            for j in range(W):
                d_f, _, y_base, _ = wfci_corrected_df(
                    src[i, j, :], donor[i, j, :], foi
                )
                df_ref[i, j, :] = d_f
                base_ref[i, j] = y_base[0]

        np.testing.assert_allclose(df_vec, df_ref, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(base_vec, base_ref, rtol=1e-12, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
