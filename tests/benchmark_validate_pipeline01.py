from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.io import loadmat

from asvimg import SUPPORTED_OUTPUT_FORMATS, load_config
from asvimg.preprocess import PreprocessRunner


def _matlab_h5_to_numpy(ds: h5py.Dataset) -> np.ndarray:
    arr = np.asarray(ds)
    if arr.ndim >= 2:
        arr = np.transpose(arr, axes=tuple(range(arr.ndim - 1, -1, -1)))
    return arr


def load_mat_payload(path: Path) -> dict[str, Any]:
    try:
        raw = loadmat(path)
        return {k: v for k, v in raw.items() if not k.startswith("__")}
    except NotImplementedError:
        out: dict[str, Any] = {}
        with h5py.File(path, "r") as h5f:
            for key in h5f.keys():
                node = h5f[key]
                if isinstance(node, h5py.Dataset):
                    out[key] = _matlab_h5_to_numpy(node)
        return out


def load_payload(path: Path) -> dict[str, Any]:
    if path.suffix == ".mat":
        return load_mat_payload(path)

    if path.suffix == ".npy":
        return np.load(path, allow_pickle=True).item()

    if path.suffix == ".h5":
        out: dict[str, Any] = {}
        with h5py.File(path, "r") as h5f:
            for key, ds in h5f.items():
                out[key] = ds[()]
            for key, value in h5f.attrs.items():
                out[key] = value
        return out

    raise ValueError(f"Unsupported payload extension: {path.suffix}")


def summarize_payload(
    payload_ch: dict[str, Any], payload_df: dict[str, Any] | None
) -> dict[str, Any]:
    image_ch0 = np.asarray(payload_ch["imageCh0"])
    summary: dict[str, Any] = {
        "shape_ch0": list(image_ch0.shape),
        "dtype_ch0": str(image_ch0.dtype),
        "mean_ch0": float(np.mean(image_ch0)),
        "std_ch0": float(np.std(image_ch0)),
    }

    if payload_df is not None:
        image_df = np.asarray(payload_df["imageDf"])
        summary["shape_df"] = list(image_df.shape)
        summary["dtype_df"] = str(image_df.dtype)
        summary["mean_df"] = float(np.mean(image_df))
        summary["std_df"] = float(np.std(image_df))
    else:
        summary["mean_df"] = 0.0

    return summary


def validate_format_output(output_dir: Path, ext: str) -> dict[str, Any]:
    ch_files = sorted(output_dir.glob(f"frameRoiCh_*.{ext}"))
    df_files = sorted(output_dir.glob(f"frameRoiDf_*.{ext}"))

    if not ch_files:
        raise FileNotFoundError(f"Missing frameRoiCh_ files in {output_dir} for .{ext}")

    payload_ch = load_payload(ch_files[0])
    payload_df = load_payload(df_files[0]) if df_files else None
    summary = summarize_payload(payload_ch, payload_df)

    return {
        "output_dir": str(output_dir),
        "ch_file_count": len(ch_files),
        "df_file_count": len(df_files),
        "first_ch_file": ch_files[0].name,
        "first_df_file": df_files[0].name if df_files else None,
        "summary": summary,
    }


def compare_to_reference(
    input_dir: Path, generated_summary: dict[str, Any]
) -> dict[str, Any]:
    # Legacy reference files
    ref_bluv = input_dir / "frameRoiBlUv__01.mat"
    ref_df = input_dir / "frameRoiDf__01.mat"
    if not ref_bluv.exists() or not ref_df.exists():
        return {"reference_found": False}

    ref_bluv_payload = load_payload(ref_bluv)
    ref_df_payload = load_payload(ref_df)

    ref_summary: dict[str, Any] = {}
    if "imageBL" in ref_bluv_payload:
        image_bl = np.asarray(ref_bluv_payload["imageBL"])
        ref_summary["mean_ch0"] = float(np.mean(image_bl))
        ref_summary["shape_ch0"] = list(image_bl.shape)
    if "imageDf" in ref_df_payload:
        image_df = np.asarray(ref_df_payload["imageDf"])
        ref_summary["mean_df"] = float(np.mean(image_df))

    result: dict[str, Any] = {"reference_found": True}
    if "shape_ch0" in ref_summary and "shape_ch0" in generated_summary:
        result["shape_ch0_match"] = generated_summary["shape_ch0"] == ref_summary["shape_ch0"]
    if "mean_ch0" in ref_summary:
        result["mean_ch0_delta"] = float(
            abs(generated_summary["mean_ch0"] - ref_summary["mean_ch0"])
        )
    if "mean_df" in ref_summary:
        result["mean_df_delta"] = float(
            abs(generated_summary["mean_df"] - ref_summary["mean_df"])
        )
    return result


def run_all_formats(args: argparse.Namespace) -> dict[str, Any]:
    input_file = Path(args.input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    config_path = Path(args.config)
    report: dict[str, Any] = {
        "input_file": str(input_file),
        "input_dir": str(input_file.parent),
        "config": str(config_path),
        "max_frames": args.max_frames,
        "formats": {},
    }

    for output_format in sorted(SUPPORTED_OUTPUT_FORMATS):
        config = load_config(config_path)
        config.input_dir = str(input_file.parent)
        config.output_dir = (
            None
            if args.output_root is None
            else str(Path(args.output_root) / output_format)
        )
        config.output_format = output_format
        if args.linear_subt is not None:
            config.linear_subt = args.linear_subt
        config.max_frames = args.max_frames

        runner = PreprocessRunner(config)
        stats = runner.run()
        validation = validate_format_output(runner.output_dir, output_format)

        report["formats"][output_format] = {
            "timing": asdict(stats),
            "validation": validation,
        }

    mat_summary = report["formats"]["mat"]["validation"]["summary"]
    npy_summary = report["formats"]["npy"]["validation"]["summary"]
    h5_summary = report["formats"]["h5"]["validation"]["summary"]

    report["cross_format"] = {
        "mat_vs_npy_shape_ch0": mat_summary["shape_ch0"] == npy_summary["shape_ch0"],
        "mat_vs_h5_shape_ch0": mat_summary["shape_ch0"] == h5_summary["shape_ch0"],
        "mat_vs_npy_mean_ch0_delta": float(
            abs(mat_summary["mean_ch0"] - npy_summary["mean_ch0"])
        ),
        "mat_vs_h5_mean_ch0_delta": float(
            abs(mat_summary["mean_ch0"] - h5_summary["mean_ch0"])
        ),
        "mat_vs_npy_mean_df_delta": float(
            abs(mat_summary["mean_df"] - npy_summary["mean_df"])
        ),
        "mat_vs_h5_mean_df_delta": float(
            abs(mat_summary["mean_df"] - h5_summary["mean_df"])
        ),
    }

    report["reference_comparison"] = compare_to_reference(
        input_file.parent, mat_summary
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pipeline01 for mat/npy/h5 on sample data, measure time, and validate outputs"
    )
    parser.add_argument(
        "--input-file",
        default="Analysis/_sampleData01/mouse-02-ratio-wOil-_2_MMStack_Default.ome.tif",
        help="Target OME-TIFF file",
    )
    parser.add_argument(
        "--config",
        default="pipeline01_config.yaml",
        help="Pipeline YAML config",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional root directory for format outputs. If omitted, defaults to input_dir/asi/<format>",
    )
    parser.add_argument(
        "--linear-subt",
        type=lambda s: s.lower() in {"1", "true", "yes", "y"},
        default=None,
        help="Optional override for linear_subt",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional report json path. Defaults to input_dir/asi/benchmark_report.json",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=500,
        help="Frame limit for benchmark run (default: 500)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = run_all_formats(args)

    input_dir = Path(report["input_dir"])
    default_report_path = input_dir / "asi" / "benchmark_report.json"
    report_path = Path(args.report_path) if args.report_path else default_report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved benchmark report: {report_path}")


if __name__ == "__main__":
    main()
