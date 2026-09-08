"""Unit tests for the suite2p-style per-channel memmap I/O primitives (P1).

Covers the (T, H, W) round-trip, chunked writes, the meta sidecar, and the
prealloc-overshoot case (files allocated longer than the real frame count are
trimmed to ``reg_meta['T_per_ch']`` on read).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from asvimg import (
    dff_name_path,
    load_dff,
    load_reg_channel,
    mean_source_stack,
    open_reg_memmap,
    read_reg_meta,
    reg_channel_path,
    write_reg_meta,
)


class TestRegMemmap(unittest.TestCase):
    def test_chunked_write_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            T, H, W, chunk = 17, 6, 8, 5
            src = np.random.default_rng(0).integers(
                0, 4096, size=(T, H, W)
            ).astype(np.uint16)

            mm = open_reg_memmap(reg_channel_path(out, 0), T, (H, W))
            for s in range(0, T, chunk):  # running-offset chunked fill
                e = min(s + chunk, T)
                mm[s:e] = src[s:e]
            mm.flush()
            del mm
            write_reg_meta(out, {"T_per_ch": np.array([T])})

            # default (in-RAM) and mmap reads both return the full (T,H,W)
            got = load_reg_channel(out, 0)
            self.assertEqual(got.shape, (T, H, W))
            self.assertEqual(got.dtype, np.uint16)
            np.testing.assert_array_equal(got, src)

            got_mm = load_reg_channel(out, 0, mmap=True)
            self.assertTrue(isinstance(got_mm, np.memmap))
            np.testing.assert_array_equal(np.array(got_mm), src)
            # Windows: release the mmap handle before the tempdir is unlinked.
            got_mm._mmap.close()
            del got_mm

    def test_prealloc_overshoot_trimmed_to_meta(self) -> None:
        """Allocate T=10 but only fill 7 (e.g. cancellation) -> reads return 7."""
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            alloc_T, real_T, H, W = 10, 7, 4, 5
            src = np.random.default_rng(1).integers(
                0, 4096, size=(real_T, H, W)
            ).astype(np.uint16)

            mm = open_reg_memmap(reg_channel_path(out, 2), alloc_T, (H, W))
            mm[:real_T] = src
            mm.flush()
            del mm
            write_reg_meta(out, {"T_per_ch": np.array([0, 0, real_T])})

            got = load_reg_channel(out, 2)
            self.assertEqual(got.shape, (real_T, H, W))
            np.testing.assert_array_equal(got, src)

    def test_no_meta_returns_full_allocation(self) -> None:
        """Absent sidecar -> no trimming (defensive fallback)."""
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            mm = open_reg_memmap(reg_channel_path(out, 0), 5, (3, 3))
            mm[:] = 1
            mm.flush()
            del mm
            self.assertEqual(load_reg_channel(out, 0).shape, (5, 3, 3))

    def test_meta_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            meta = {
                "T_per_ch": np.array([12, 12]),
                "channels_name": np.array(["BL", "BL"]),
                "fps": np.array(20),
                "proc_template": np.zeros((4, 4), dtype=np.float64),
            }
            write_reg_meta(out, meta)
            got = read_reg_meta(out)
            np.testing.assert_array_equal(got["T_per_ch"], [12, 12])
            self.assertEqual(list(got["channels_name"]), ["BL", "BL"])
            self.assertEqual(int(got["fps"]), 20)

    def test_dff_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            T, H, W = 9, 4, 5
            data = (np.random.default_rng(2).standard_normal((T, H, W))
                    * 0.02).astype(np.float32)
            np.save(dff_name_path(out, "GCaMP"), data)
            got = load_dff(out, "GCaMP")
            self.assertEqual(got.shape, (T, H, W))
            self.assertEqual(got.dtype, np.float32)
            np.testing.assert_array_equal(got, data)


class TestMeanSourceStack(unittest.TestCase):
    def test_trims_ragged_channels_to_min_T(self) -> None:
        """Sibling source channels can differ by one frame at the cycle tail;
        mean_source_stack must trim to the common min T rather than crash."""
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            H, W = 4, 5
            np.save(reg_channel_path(out, 0),
                    np.ones((10, H, W), dtype=np.uint16))
            np.save(reg_channel_path(out, 1),
                    np.full((9, H, W), 3, dtype=np.uint16))
            write_reg_meta(out, {"T_per_ch": np.array([10, 9])})

            avg = mean_source_stack(out, [0, 1])  # would raise if ragged
            self.assertEqual(avg.shape, (H, W, 9))  # (H, W, min T)
            np.testing.assert_allclose(avg, 2.0)  # (1 + 3) / 2


if __name__ == "__main__":
    unittest.main()
