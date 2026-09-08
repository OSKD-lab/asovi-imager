"""Benchmark Python (NumPy / PyTorch CPU) dftregistration and compare with MATLAB.

Usage
-----
Step 1 (optional): Run MATLAB benchmark to generate reference .mat
    cd matlab && matlab -batch "benchmark_dft_matlab"

Step 2: Run this script
    uv run python tests/benchmark_dft_compare.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import tifffile
from scipy.io import loadmat

from asvimg.registration import (
    DftRegistrationResult,
    DftRegistrator,
    dftregistration,
)
from asvimg.registration.fast import DftRegistratorNumba

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLE_DIR = Path("Analysis/_sampleData01")
TIF_NAME = "mouse-02-ratio-wOil-_2_MMStack_Default.ome.tif"
USFAC = 500
N_FRAMES = 200
TEMPLATE_STRIDE = 100


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_even_frames(tif_path: Path, n_frames: int) -> tuple[np.ndarray, list[int]]:
    """Return (frames, indices) for even-numbered frames (1-indexed: 2,4,6,...)."""
    with tifffile.TiffFile(tif_path) as tif:
        pages = tif.pages
        total = len(pages)
        indices_0 = list(range(1, total, 2))[:n_frames]
        first = pages[0].asarray()
        rows, cols = first.shape[:2]
        frames = np.empty((rows, cols, len(indices_0)), dtype=np.float64)
        for i, idx in enumerate(indices_0):
            frames[:, :, i] = pages[idx].asarray()[:rows, :cols].astype(np.float64)
    indices_1 = [idx + 1 for idx in indices_0]
    return frames, indices_1


def create_template(tif_path: Path, stride: int) -> np.ndarray:
    """Average every `stride`-th frame to create a template."""
    with tifffile.TiffFile(tif_path) as tif:
        pages = tif.pages
        total = len(pages)
        indices = list(range(0, total, stride))
        acc = np.zeros(pages[0].asarray().shape[:2], dtype=np.float64)
        for idx in indices:
            acc += pages[idx].asarray().astype(np.float64)
    return acc / len(indices)


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def run_numpy_benchmark(
    template: np.ndarray, frames: np.ndarray, usfac: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run NumPy dftregistration per-frame."""
    n = frames.shape[2]
    results = np.empty((n, 4), dtype=np.float64)
    frame_times = np.empty(n, dtype=np.float64)
    template_ft = np.fft.fft2(template)

    total_t0 = time.perf_counter()
    for i in range(n):
        frame_ft = np.fft.fft2(frames[:, :, i])
        t0 = time.perf_counter()
        res, _ = dftregistration(template_ft, frame_ft, usfac)
        frame_times[i] = time.perf_counter() - t0
        results[i] = [res.error, res.diffphase, res.row_shift, res.col_shift]
        if (i + 1) % 50 == 0:
            print(f"    NumPy: {i + 1}/{n} done "
                  f"({time.perf_counter() - total_t0:.2f}s)")
    total_elapsed = time.perf_counter() - total_t0
    return results, frame_times, total_elapsed


def run_numba_benchmark(
    template: np.ndarray, frames: np.ndarray, usfac: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run Numba-optimized dftregistration."""
    n = frames.shape[2]
    results = np.empty((n, 4), dtype=np.float64)
    frame_times = np.empty(n, dtype=np.float64)

    reg = DftRegistratorNumba(template, usfac=usfac)

    # Warm up JIT with first frame
    _ = reg.register_frames(frames[:, :, :1])

    total_t0 = time.perf_counter()
    for i in range(n):
        t0 = time.perf_counter()
        res_list, _ = reg.register_frames(frames[:, :, i:i+1])
        frame_times[i] = time.perf_counter() - t0
        r = res_list[0]
        results[i] = [r.error, r.diffphase, r.row_shift, r.col_shift]
        if (i + 1) % 50 == 0:
            print(f"    Numba: {i + 1}/{n} done "
                  f"({time.perf_counter() - total_t0:.2f}s)")
    total_elapsed = time.perf_counter() - total_t0
    return results, frame_times, total_elapsed


def run_torch_benchmark(
    template: np.ndarray, frames: np.ndarray, usfac: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run PyTorch CPU dftregistration with batched FFT."""
    n = frames.shape[2]
    results = np.empty((n, 4), dtype=np.float64)
    frame_times = np.empty(n, dtype=np.float64)

    # (H, W, N) -> (N, H, W)
    frames_t = torch.from_numpy(np.ascontiguousarray(frames.transpose(2, 0, 1)))
    template_t = torch.from_numpy(template)

    reg = DftRegistrator(template_t, usfac=usfac)

    total_t0 = time.perf_counter()
    # batch FFT warmup included in timing
    res_list, _ = reg.register_batch(frames_t, return_registered=False)
    total_elapsed = time.perf_counter() - total_t0

    for i, res in enumerate(res_list):
        results[i] = [res.error, res.diffphase, res.row_shift, res.col_shift]
        frame_times[i] = total_elapsed / n  # uniform estimate

    return results, frame_times, total_elapsed


# ---------------------------------------------------------------------------
# MATLAB loader
# ---------------------------------------------------------------------------

def _load_mat_auto(mat_path: Path) -> dict:
    """Load .mat file, handling both v5/v7 (scipy) and v7.3/HDF5 (h5py)."""
    try:
        return loadmat(mat_path)
    except NotImplementedError:
        out: dict = {}
        with h5py.File(mat_path, "r") as f:
            for key in f.keys():
                ds = f[key]
                if isinstance(ds, h5py.Dataset):
                    arr = np.asarray(ds)
                    if arr.ndim >= 2:
                        arr = arr.T
                    out[key] = arr
        return out


def load_matlab_results(mat_path: Path) -> dict | None:
    if not mat_path.exists():
        return None
    data = _load_mat_auto(mat_path)
    return {
        "results": np.asarray(data["results"]),
        "frame_times": np.asarray(data["frame_times"]).ravel(),
        "total_elapsed": float(np.asarray(data["total_elapsed"]).ravel()[0]),
        "even_indices": np.asarray(data["even_indices"]).ravel().astype(int),
        "template_img": np.asarray(data["template_img"]),
        "usfac": int(np.asarray(data["usfac"]).ravel()[0]),
        "actual_n": int(np.asarray(data["actual_n"]).ravel()[0]),
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_results(a: np.ndarray, b: np.ndarray) -> dict:
    col_names = ["error", "diffphase", "row_shift", "col_shift"]
    diff = a - b
    stats = {}
    for j, name in enumerate(col_names):
        d = diff[:, j]
        stats[name] = {
            "max_abs_diff": float(np.max(np.abs(d))),
            "mean_abs_diff": float(np.mean(np.abs(d))),
            "rms_diff": float(np.sqrt(np.mean(d ** 2))),
        }
    return stats


def _print_comparison(
    label_a: str, label_b: str,
    time_a: float, time_b: float,
    comparison: dict,
    n: int,
) -> None:
    print(f"\n  {label_a} vs {label_b}  ({n} frames)")
    print(f"  {'-'*50}")
    print(f"  {label_a:>12s}: {time_a:8.3f}s  ({time_a/n*1000:.1f}ms/frame)")
    print(f"  {label_b:>12s}: {time_b:8.3f}s  ({time_b/n*1000:.1f}ms/frame)")
    if time_a > 0 and time_b > 0:
        ratio = time_a / time_b
        if ratio > 1:
            print(f"  => {label_b} is {ratio:.2f}x faster")
        else:
            print(f"  => {label_a} is {1/ratio:.2f}x faster")
    for name, s in comparison.items():
        print(f"    {name:>12s}: max|diff|={s['max_abs_diff']:.2e}  "
              f"rms={s['rms_diff']:.2e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    tif_path = SAMPLE_DIR / TIF_NAME
    mat_path = SAMPLE_DIR / "benchmark_dft_result_matlab.mat"
    report_path = SAMPLE_DIR / "benchmark_dft_compare_report.json"

    if not tif_path.exists():
        raise FileNotFoundError(f"TIFF not found: {tif_path}")

    print("Creating template ...")
    template = create_template(tif_path, TEMPLATE_STRIDE)
    print(f"  Template shape: {template.shape}")

    print(f"Loading {N_FRAMES} even frames ...")
    frames, indices_1 = load_even_frames(tif_path, N_FRAMES)
    n = frames.shape[2]
    print(f"  Loaded {n} frames, indices {indices_1[0]}-{indices_1[-1]}")

    # ---- NumPy ----
    print(f"\n[1/3] NumPy dftregistration (usfac={USFAC}) ...")
    np_results, np_times, np_total = run_numpy_benchmark(template, frames, USFAC)
    print(f"  NumPy total: {np_total:.3f}s  |  mean: {np.mean(np_times)*1000:.1f}ms/frame")

    # ---- Numba ----
    print(f"\n[2/3] Numba-optimized dftregistration (usfac={USFAC}) ...")
    nb_results, nb_times, nb_total = run_numba_benchmark(template, frames, USFAC)
    print(f"  Numba total: {nb_total:.3f}s  |  mean: {np.mean(nb_times)*1000:.1f}ms/frame")

    # ---- PyTorch CPU ----
    print(f"\n[3/3] PyTorch CPU dftregistration (usfac={USFAC}) ...")
    pt_results, pt_times, pt_total = run_torch_benchmark(template, frames, USFAC)
    print(f"  Torch total: {pt_total:.3f}s  |  mean: {pt_total/n*1000:.1f}ms/frame")

    # ---- Build report ----
    report: dict = {
        "n_frames": n,
        "usfac": USFAC,
        "template_stride": TEMPLATE_STRIDE,
        "numpy": {
            "total_seconds": np_total,
            "mean_per_frame": float(np.mean(np_times)),
        },
        "numba": {
            "total_seconds": nb_total,
            "mean_per_frame": float(np.mean(nb_times)),
        },
        "torch_cpu": {
            "total_seconds": pt_total,
            "mean_per_frame": pt_total / n,
        },
    }

    # ---- Print comparisons ----
    print(f"\n{'='*60}")
    print(f"  Benchmark: {n} even frames, usfac={USFAC}, {template.shape}")
    print(f"{'='*60}")

    # NumPy vs Numba
    np_vs_nb = compare_results(np_results, nb_results)
    _print_comparison("NumPy", "Numba", np_total, nb_total, np_vs_nb, n)
    report["numpy_vs_numba"] = np_vs_nb

    # NumPy vs Torch
    np_vs_pt = compare_results(np_results, pt_results)
    _print_comparison("NumPy", "Torch CPU", np_total, pt_total, np_vs_pt, n)
    report["numpy_vs_torch"] = np_vs_pt

    # Numba vs Torch
    nb_vs_pt = compare_results(nb_results, pt_results)
    _print_comparison("Numba", "Torch CPU", nb_total, pt_total, nb_vs_pt, n)
    report["numba_vs_torch"] = nb_vs_pt

    # MATLAB comparison
    mat_data = load_matlab_results(mat_path)
    if mat_data is not None:
        mat_results = mat_data["results"]
        mat_total = mat_data["total_elapsed"]
        n_cmp = min(n, len(mat_results))

        mat_vs_np = compare_results(mat_results[:n_cmp], np_results[:n_cmp])
        _print_comparison("MATLAB", "NumPy", mat_total, np_total, mat_vs_np, n_cmp)

        mat_vs_nb = compare_results(mat_results[:n_cmp], nb_results[:n_cmp])
        _print_comparison("MATLAB", "Numba", mat_total, nb_total, mat_vs_nb, n_cmp)

        mat_vs_pt = compare_results(mat_results[:n_cmp], pt_results[:n_cmp])
        _print_comparison("MATLAB", "Torch CPU", mat_total, pt_total, mat_vs_pt, n_cmp)

        report["matlab"] = {
            "total_seconds": mat_total,
            "mean_per_frame": float(np.mean(mat_data["frame_times"])),
        }
        report["matlab_vs_numpy"] = mat_vs_np
        report["matlab_vs_numba"] = mat_vs_nb
        report["matlab_vs_torch"] = mat_vs_pt
    else:
        print(f"\n  MATLAB .mat not found at {mat_path} - skipping MATLAB comparison")

    print(f"{'='*60}")

    # ---- Save ----
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
