from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sif_parser
from tqdm import tqdm


def load_first_frames(spool_dir: Path, n_frames: int | None = None) -> np.ndarray:
	"""Load spool data and return the first n_frames as float32 (T, H, W)."""
	# The sample spool may contain fewer binary chunks than header count.
	data, _ = sif_parser.np_spool_open(str(spool_dir), ignore_missing=True)
	frames = np.asarray(data)

	if frames.ndim != 3:
		raise ValueError(f"Expected 3D array (T, H, W), got shape={frames.shape}")
	if n_frames is None:
		n_frames = int(frames.shape[0])
	if frames.shape[0] < n_frames:
		raise ValueError(
			f"Requested {n_frames} frames, but only {frames.shape[0]} are available"
		)

	return frames[:n_frames].astype(np.float32, copy=False)


def pearson_corr_matrix(frames: np.ndarray) -> np.ndarray:
	"""Compute frame-wise Pearson correlation matrix for a stack (T, H, W)."""
	t = frames.shape[0]
	flat = frames.reshape(t, -1)
	return np.corrcoef(flat)


def mean_upper_triangle(mat: np.ndarray) -> float:
	if mat.shape[0] < 2:
		return float("nan")
	i, j = np.triu_indices(mat.shape[0], k=1)
	return float(np.mean(mat[i, j]))


def quantify_mod4_similarity(frames: np.ndarray) -> dict[str, Any]:
	"""Quantify within-channel and between-channel similarity for modulo-4 groups."""
	corr = pearson_corr_matrix(frames)
	groups = {ch: np.arange(ch, frames.shape[0], 4) for ch in range(4)}

	within: dict[str, float] = {}
	for ch, idx in groups.items():
		within[f"ch{ch}"] = mean_upper_triangle(corr[np.ix_(idx, idx)])

	between: dict[str, float] = {}
	for ch_a, ch_b in combinations(range(4), 2):
		idx_a = groups[ch_a]
		idx_b = groups[ch_b]
		block = corr[np.ix_(idx_a, idx_b)]
		between[f"ch{ch_a}_vs_ch{ch_b}"] = float(np.mean(block))

	prototypes = np.stack([frames[groups[ch]].mean(axis=0) for ch in range(4)], axis=0)
	prototype_corr = np.corrcoef(prototypes.reshape(4, -1))

	within_mean = float(np.mean(list(within.values())))
	between_mean = float(np.mean(list(between.values())))

	return {
		"n_frames": int(frames.shape[0]),
		"frame_shape": [int(frames.shape[1]), int(frames.shape[2])],
		"group_indices": {f"ch{ch}": groups[ch].tolist() for ch in range(4)},
		"within_channel_mean_corr": within,
		"between_channel_mean_corr": between,
		"overall_within_mean_corr": within_mean,
		"overall_between_mean_corr": between_mean,
		"separation_score_within_minus_between": within_mean - between_mean,
		"prototype_corr_matrix": prototype_corr.tolist(),
	}


def plot_mod4_grid(frames: np.ndarray, output_png: Path) -> None:
	"""Plot 4x8 tight grid where each row is one modulo-4 channel."""
	rows, cols = 4, 8
	if frames.shape[0] < rows * cols:
		raise ValueError(f"Need at least {rows * cols} frames, got {frames.shape[0]}")

	vmin, vmax = np.percentile(frames[: rows * cols], [1, 99])

	fig, axes = plt.subplots(rows, cols, figsize=(16, 8), dpi=140)
	for r in range(rows):
		for c in range(cols):
			frame_idx = r + 4 * c
			ax = axes[r, c]
			ax.imshow(frames[frame_idx], cmap="gray", vmin=vmin, vmax=vmax)
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_title(f"f{frame_idx}", fontsize=8)
			if c == 0:
				ax.set_ylabel(f"ch{r}", fontsize=10)

	fig.suptitle("First 32 frames arranged by frame_index % 4", fontsize=12)
	fig.tight_layout(pad=0.2, w_pad=0.05, h_pad=0.05)
	output_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_png, bbox_inches="tight")
	plt.close(fig)


def normalize_stack_to_uint8(stack: np.ndarray) -> np.ndarray:
	"""Normalize a (T, H, W) stack to uint8 using global min/max in the stack."""
	arr = np.asarray(stack, dtype=np.float32)
	arr_min = float(np.min(arr))
	arr_max = float(np.max(arr))
	den = max(arr_max - arr_min, 1e-12)
	norm = (arr - arr_min) / den
	return np.clip(norm * 255.0, 0, 255).astype(np.uint8)


def write_gray_avi(frames_u8: np.ndarray, out_path: Path, fps: float) -> None:
	"""Write grayscale uint8 stack (T, H, W) to AVI."""
	if frames_u8.ndim != 3:
		raise ValueError(f"Expected (T, H, W), got {frames_u8.shape}")

	t, h, w = frames_u8.shape
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h), isColor=False)
	if not writer.isOpened():
		raise RuntimeError(f"Failed to open VideoWriter: {out_path}")

	try:
		for i in tqdm(
			range(t),
			desc=f"Writing {out_path.name}",
			unit="frame",
			leave=False,
		):
			writer.write(np.ascontiguousarray(frames_u8[i]))
	finally:
		writer.release()


def save_channel_avis(
	frames: np.ndarray,
	out_dir: Path,
	fps: float,
	min_start_frame: int,
) -> dict[str, dict[str, str]]:
	"""Save two AVI types per channel: normal norm8 and min-subtracted norm8."""
	outputs: dict[str, dict[str, str]] = {}
	for ch in tqdm(range(4), desc="Export channels", unit="ch"):
		idx_full = np.arange(ch, frames.shape[0], 4)
		ch_frames = frames[idx_full]
		if ch_frames.size == 0:
			continue

		normal_u8 = normalize_stack_to_uint8(ch_frames)
		normal_path = out_dir / f"sifx_ch{ch}_norm8.avi"
		write_gray_avi(normal_u8, normal_path, fps=fps)

		ref_mask = idx_full >= min_start_frame
		ref_frames = ch_frames[ref_mask]
		if ref_frames.shape[0] == 0:
			ref_frames = ch_frames
		min_image = np.min(ref_frames, axis=0, keepdims=True)
		min_sub = np.clip(ch_frames - min_image, 0.0, None)
		min_sub_u8 = normalize_stack_to_uint8(min_sub)
		min_sub_path = out_dir / f"sifx_ch{ch}_minsub_norm8.avi"
		write_gray_avi(min_sub_u8, min_sub_path, fps=fps)

		outputs[f"ch{ch}"] = {
			"normal_norm8_avi": str(normal_path),
			"minsub_norm8_avi": str(min_sub_path),
		}

	return outputs


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Load first 32 frames from an Andor spool (.sifx + spool.dat), "
			"plot 4x8 modulo-4 channel grid, and report similarity metrics."
		)
	)
	parser.add_argument(
		"--spool-dir",
		type=Path,
		default=Path("Analysis/SampleData/spool_4c-3"),
		help="Directory that contains .sifx/.ini and spool.dat files",
	)
	parser.add_argument(
		"--n-frames",
		type=int,
		default=32,
		help="Number of initial frames to analyze/export. Use <=0 for all frames (default: 32)",
	)
	parser.add_argument(
		"--all-frames",
		action="store_true",
		help="Use all available frames for analysis/export",
	)
	parser.add_argument(
		"--out-dir",
		type=Path,
		default=None,
		help="Output directory for figure and JSON (default: <spool-dir>/ModifiedData)",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="Display plot window in addition to saving PNG",
	)
	parser.add_argument(
		"--fps",
		type=float,
		default=10.0,
		help="FPS for channel AVI outputs (default: 10.0)",
	)
	parser.add_argument(
		"--min-start-frame",
		type=int,
		default=40,
		help="Use frames >= this global frame index to compute min image for min-subtraction (default: 40)",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	spool_dir = args.spool_dir
	if not spool_dir.exists():
		raise FileNotFoundError(f"Spool directory not found: {spool_dir}")

	out_dir = args.out_dir or (spool_dir / "ModifiedData")
	out_dir.mkdir(parents=True, exist_ok=True)

	all_frames = np.asarray(sif_parser.np_spool_open(str(spool_dir), ignore_missing=True)[0])
	if all_frames.ndim != 3:
		raise ValueError(f"Expected 3D array (T, H, W), got shape={all_frames.shape}")

	if args.all_frames or args.n_frames <= 0:
		target_n = int(all_frames.shape[0])
	else:
		target_n = int(args.n_frames)

	if target_n > int(all_frames.shape[0]):
		raise ValueError(
			f"Requested {target_n} frames, but only {all_frames.shape[0]} are available"
		)

	frames = all_frames[:target_n].astype(np.float32, copy=False)

	png_path = out_dir / f"sifx_first{target_n}_mod4_grid.png"
	if target_n >= 32:
		plot_mod4_grid(frames[:32], png_path)
	else:
		print("Skip 4x8 plot: fewer than 32 frames")
		png_path = out_dir / f"sifx_first{target_n}_mod4_grid_skipped.txt"
		png_path.write_text("Skipped: fewer than 32 frames", encoding="utf-8")

	metrics = quantify_mod4_similarity(frames)
	metrics["spool_dir"] = str(spool_dir)
	metrics["figure_path"] = str(png_path)
	metrics["min_subtraction_reference_start_frame"] = int(args.min_start_frame)
	metrics["channel_avi_outputs"] = save_channel_avis(
		frames,
		out_dir=out_dir,
		fps=args.fps,
		min_start_frame=int(args.min_start_frame),
	)

	json_path = out_dir / f"sifx_first{target_n}_mod4_similarity.json"
	json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

	print(json.dumps(metrics, indent=2))
	print(f"Saved figure: {png_path}")
	print(f"Saved metrics: {json_path}")

	if args.show:
		plt.figure(figsize=(8, 4))
		img = plt.imread(png_path)
		plt.imshow(img)
		plt.axis("off")
		plt.tight_layout()
		plt.show()


if __name__ == "__main__":
	main()
