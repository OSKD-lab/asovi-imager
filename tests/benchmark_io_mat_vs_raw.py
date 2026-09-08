from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


def _load_first_mat_pair(
    mat_dir: Path,
) -> tuple[Path, Path, np.ndarray, np.ndarray, np.ndarray]:
    bluv_files = sorted(mat_dir.glob("frameRoiBlUv_*.mat"))
    df_files = sorted(mat_dir.glob("frameRoiDf_*.mat"))
    if not bluv_files or not df_files:
        raise FileNotFoundError(f"No mat outputs found in {mat_dir}")

    bluv = loadmat(bluv_files[0])
    df = loadmat(df_files[0])
    image_bl = np.asarray(bluv["imageBL"])
    image_uv = np.asarray(bluv["imageUV"])
    image_df = np.asarray(df["imageDf"])
    return bluv_files[0], df_files[0], image_bl, image_uv, image_df


def _write_raw_append(
    raw_file: Path, image_bl: np.ndarray, image_uv: np.ndarray, image_df: np.ndarray
) -> float:
    t0 = time.perf_counter()
    with raw_file.open("ab") as f:
        f.write(image_bl.tobytes(order="C"))
        f.write(image_uv.tobytes(order="C"))
        f.write(image_df.tobytes(order="C"))
    return time.perf_counter() - t0


def _write_raw_overwrite(
    raw_file: Path, image_bl: np.ndarray, image_uv: np.ndarray, image_df: np.ndarray
) -> float:
    t0 = time.perf_counter()
    with raw_file.open("wb") as f:
        f.write(image_bl.tobytes(order="C"))
        f.write(image_uv.tobytes(order="C"))
        f.write(image_df.tobytes(order="C"))
    return time.perf_counter() - t0


def run_benchmark(args: argparse.Namespace) -> dict:
    mat_dir = Path(args.mat_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bluv_file, df_file, image_bl, image_uv, image_df = _load_first_mat_pair(mat_dir)

    mat_times: list[float] = []
    raw_times: list[float] = []

    mat_path = out_dir / "io_bench_payload.mat"
    raw_path = out_dir / "io_bench_payload.rawbin"

    if args.raw_mode == "append":
        # Keep file for append benchmarking across runs.
        raw_path.touch(exist_ok=True)

    for _ in range(args.runs):
        t0 = time.perf_counter()
        savemat(
            mat_path,
            {
                "imageBL": image_bl,
                "imageUV": image_uv,
                "imageDf": image_df,
            },
        )
        mat_times.append(time.perf_counter() - t0)

    for _ in range(args.runs):
        if args.raw_mode == "append":
            raw_times.append(_write_raw_append(raw_path, image_bl, image_uv, image_df))
        else:
            raw_times.append(
                _write_raw_overwrite(raw_path, image_bl, image_uv, image_df)
            )

    payload_bytes = image_bl.nbytes + image_uv.nbytes + image_df.nbytes

    report = {
        "source": {
            "mat_dir": str(mat_dir),
            "bluv_file": str(bluv_file),
            "df_file": str(df_file),
        },
        "raw_mode": args.raw_mode,
        "runs": args.runs,
        "payload": {
            "shapeBL": list(image_bl.shape),
            "shapeUV": list(image_uv.shape),
            "shapeDf": list(image_df.shape),
            "dtypeBL": str(image_bl.dtype),
            "dtypeUV": str(image_uv.dtype),
            "dtypeDf": str(image_df.dtype),
            "bytes_per_payload": int(payload_bytes),
        },
        "mat_seconds": {
            "values": mat_times,
            "mean": float(np.mean(mat_times)),
            "min": float(np.min(mat_times)),
            "max": float(np.max(mat_times)),
        },
        "raw_binary_seconds": {
            "values": raw_times,
            "mean": float(np.mean(raw_times)),
            "min": float(np.min(raw_times)),
            "max": float(np.max(raw_times)),
        },
        "speedup": {
            "raw_over_mat_mean": float(np.mean(mat_times) / np.mean(raw_times)),
            "raw_over_mat_min_based": float(np.min(mat_times) / np.min(raw_times)),
        },
        "output_files": {
            "mat_path": str(mat_path),
            "raw_path": str(raw_path),
            "mat_bytes": int(mat_path.stat().st_size) if mat_path.exists() else 0,
            "raw_bytes": int(raw_path.stat().st_size) if raw_path.exists() else 0,
        },
    }
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark .mat vs raw-binary write I/O"
    )
    parser.add_argument(
        "--mat-dir",
        default="Analysis/_sampleData01/asi/mat",
        help="Directory containing frameRoi*.mat outputs",
    )
    parser.add_argument(
        "--out-dir",
        default="Analysis/_sampleData01/asi/benchmark_io_artifacts",
        help="Directory to place benchmark artifacts",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of write runs per format",
    )
    parser.add_argument(
        "--raw-mode",
        choices=["append", "overwrite"],
        default="append",
        help="Raw-binary write mode",
    )
    parser.add_argument(
        "--report-path",
        default="Analysis/_sampleData01/asi/benchmark_io_mat_vs_raw.json",
        help="Report JSON output path",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = run_benchmark(args)

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
