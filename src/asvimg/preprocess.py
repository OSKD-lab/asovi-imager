"""ASoVi preprocess — registration + WFCI linear subtraction runner and CLI.

Reads TIFF/DCIMG/SIFX frames in an N-channel cyclic pattern (demuxed through
``build_demux_correction``) and runs DFT registration -> binning -> dF/F in
read/process/write chunks.  Writes the intermediates the rest of the pipeline
reads: ``reg_Ch{i}.npy`` (T, H, W) uint16 memmaps, one ``dff_{name}.npy``
(T, H, W) float32 per channels_name group, and the ``reg_meta.npz`` sidecar
(plus optional TIFFs / QC figures).  It writes no mat/npy/h5 payload itself —
``output_format`` only picks the default output dir and the format the downstream
stages save in.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm
from scipy.io import savemat
from skimage.transform import resize

from .filters import apply_xyt_filter

from . import (
    SUPPORTED_OUTPUT_FORMATS,
    CancellationToken,
    DftRegistrationResult,
    DftRegistrator,
    PipelineConfig,
    PreprocessStats,
    ProgressReporter,
    StageId,
    build_fig_frames,
    default_output_dir,
    dff_name_path,
    find_input_files,
    get_frame_count,
    infer_exp_name,
    iter_frames_with_metadata,
    load_config,
    NullReporter,
    build_demux_correction,
    load_frames_by_indices,
    load_reg_channel,
    load_template_image,
    load_template_mat,
    open_reg_memmap,
    reg_channel_path,
    reg_meta_path,
    save_config,
    save_grayscale_png,
    save_stack_tiff,
    wfci_corrected_df_vectorized,
    write_reg_meta,
)


class PreprocessRunner:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.input_dir = Path(config.input_dir)
        self.output_dir = (
            Path(config.output_dir)
            if config.output_dir
            else default_output_dir(self.input_dir, config.output_format)
        )

        if config.output_format not in SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(
                f"output_format must be one of {sorted(SUPPORTED_OUTPUT_FORMATS)}"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._reporter: ProgressReporter | None = None

    def _log(self, msg: str) -> None:
        """Route a ``[preprocess]`` line to the reporter, or print it.

        With no reporter (the ``asovi-preprocess`` CLI default) the message is
        printed exactly as before, keeping CLI output byte-for-byte identical.

        A QC line must never be able to kill the run: a Japanese Windows console
        is cp932, and one stray non-ASCII character in a message would otherwise
        raise UnicodeEncodeError out of ``print`` and abort preprocess.  Keep the
        strings ASCII, and degrade rather than die if one slips through.
        """
        if self._reporter is not None:
            self._reporter.on_log(StageId.PREPROCESS, msg)
            return
        try:
            print(msg)
        except UnicodeEncodeError:
            enc = getattr(sys.stdout, "encoding", None) or "ascii"
            print(msg.encode(enc, "replace").decode(enc, "replace"))

    def run(
        self,
        *,
        reporter: ProgressReporter | None = None,
        cancel: CancellationToken | None = None,
    ) -> PreprocessStats:
        run_t0 = time.perf_counter()
        self._reporter = reporter
        input_files = find_input_files(
            self.input_dir, self.config.input_format, self.config.input_order
        )

        stats = PreprocessStats(
            output_format=self.config.output_format,
            output_dir=str(self.output_dir),
            files_processed=len(input_files),
        )

        exp_name = self.config.exp_name or infer_exp_name(input_files[0].name)
        if not exp_name:
            raise ValueError("Failed to infer exp_name. Set exp_name in YAML or CLI.")

        # registration_cache="cached": reuse existing motion-corrected output
        # (frameRoiCh_*) instead of re-running the expensive registration.
        if self.config.registration_cache == "cached":
            existing = sorted(self.output_dir.glob("reg_Ch*.npy"))
            if existing and self._previous_run_was_partial():
                # reg_Ch*.npy are preallocated at the START of a run, so their mere
                # existence proves nothing. Reusing a cancelled or crashed run's
                # prefix (or its zero-filled allocation) would silently analyse a
                # fraction of the recording -- or nothing at all.
                self._log(
                    "[preprocess] cached: the existing reg_Ch*.npy come from an "
                    "incomplete run (cancelled or crashed); re-running registration."
                )
                existing = []
            if existing:
                self._log(
                    f"[preprocess] cached: reusing {len(existing)} existing "
                    f"reg_Ch*.npy file(s); registration skipped."
                )
                if self.config.save_raw_each_ch or self.config.save_registered_each_ch:
                    self._log(
                        "[preprocess] WARNING: save_raw_each_ch / "
                        "save_registered_each_ch are IGNORED on a cached run: "
                        "per-frame TIFFs are only written while registration "
                        "runs. Set registration_cache='force' to (re)generate them."
                    )
                stats.batches_saved = len(existing)
                stats.cached = True
                stats.total_seconds = time.perf_counter() - run_t0
                return stats
            self._log(
                "[preprocess] cached requested but no existing frameRoiCh found; "
                "running registration."
            )

        if self.config.delete:
            self._delete_previous_outputs(exp_name)

        self._log("[preprocess] Creating template...")
        t_template0 = time.perf_counter()
        proc_template = self._load_or_create_template(input_files)
        if self.config.flip:
            # Frames are mirrored on read (below), so the registration reference
            # must live in the mirrored space too: registering a flipped frame
            # against an unflipped template finds the MIRROR offset, not the
            # motion, and applies it as a circular (wrapping) shift.
            proc_template = np.ascontiguousarray(np.fliplr(proc_template))
        stats.template_seconds += time.perf_counter() - t_template0
        save_grayscale_png(self.output_dir / "fig_template.png", proc_template)

        registrator = DftRegistrator(
            torch.from_numpy(proc_template), usfac=self.config.usfac
        )

        # Count total frames for progress bar
        frame_counts = [get_frame_count(f) for f in input_files]
        total_frames_est = sum(frame_counts)
        if self.config.max_frames is not None:
            total_frames_est = min(total_frames_est, self.config.max_frames)

        cycle_len = self.config.cycle_len
        self._log(
            f"[preprocess] {len(input_files)} file(s), ~{total_frames_est} frames, "
            f"{cycle_len}-ch cycle"
        )
        reg_state = "on" if self.config.do_registration else "off"
        self._log(
            f"[preprocess] Stage: register (usfac={self.config.usfac}, "
            f"reg={reg_state}) -> bin x{self.config.binning} -> "
            f"{'linear-subtract' if self.config.linear_subt else 'no-subtract'} "
            f"-> reg_Ch*.npy ({'mmap' if self.config.use_mmap else 'in-RAM'} dF/F)"
        )

        # Channel-cycle alignment across file boundaries: channel index continues
        # (global_frame_index % cycle_len) across files, so a NON-LAST file whose
        # frame count is not a multiple of cycle_len phase-shifts the channel
        # assignment for every later file → silent donor/source swap. (The last
        # file may end mid-cycle harmlessly.) Warn loudly rather than guess.
        if cycle_len > 1 and len(input_files) > 1:
            for f, n in zip(input_files[:-1], frame_counts[:-1]):
                if n % cycle_len != 0:
                    self._log(
                        f"[preprocess] WARNING: '{f.name}' has {n} frames, not a "
                        f"multiple of cycle_len={cycle_len}; channels in later "
                        f"files shift by {n % cycle_len}. Verify channel ordering "
                        f"/ first-frame alignment, or split files per recording."
                    )

        # Demux correction (the demux editor's sidecar, else the config's global
        # offset + per-file channels_slip). channel_at() is identity for an unset
        # correction, so the positional demux is unchanged unless one exists.
        # It is built HERE, before the allocation, because it decides how many
        # frames each channel gets: sizing the memmaps from the positional cycle
        # while assigning frames by the correction silently drops the tail frames
        # of any channel the correction gives extra frames to.
        demux_corr = build_demux_correction(
            self.config, self.output_dir, frame_counts
        )
        if not demux_corr.is_identity():
            self._log(
                f"[preprocess] demux correction applied: "
                f"start_offset={demux_corr.start_offset}, edits={demux_corr.edits}"
            )

        # --- Preallocated per-channel (T, H, W) uint16 memmaps ---------------
        # Every registered+binned frame is written straight into reg_Ch{i}.npy at
        # a running offset: the
        # recording is one continuous file per channel (a global dF/F baseline
        # follows).  T is estimated from the frame counts; on cancellation /
        # max_frames the real count is recorded in reg_meta and readers slice it.
        reg_hw = self._binned_hw(proc_template.shape)
        T_per_ch = np.bincount(
            demux_corr.phases(total_frames_est), minlength=cycle_len
        ).tolist()
        # The allocation below TRUNCATES any existing reg_Ch{i}.npy to zeros, so
        # the sidecar describing the old data stops being true right here. With
        # delete=False a crash mid-run would otherwise leave last run's reg_meta
        # (and its T_per_ch) next to all-zero frames, and every reader would trim
        # by it and quietly analyse zeros. Drop it now; it is rewritten at the end.
        reg_meta_path(self.output_dir).unlink(missing_ok=True)
        reg_mm = [
            open_reg_memmap(
                reg_channel_path(self.output_dir, ch), T_per_ch[ch], reg_hw
            )
            for ch in range(cycle_len)
        ]
        write_off = [0] * cycle_len
        mean_sum = [np.zeros(reg_hw, dtype=np.float64) for _ in range(cycle_len)]
        # Raw (pre-processing) frames stream into per-channel temp memmaps when
        # requested, so RAM stays bounded on long recordings (old code buffered
        # every raw frame of every channel in a list until the end).
        raw_enabled = self.config.save_raw_each_ch
        raw_mm: list | None = [None] * cycle_len if raw_enabled else None
        raw_off = [0] * cycle_len
        raw_paths = [self.output_dir / f"_rawtmp_Ch{ch}.npy" for ch in range(cycle_len)]

        # Registration batch buffer: accumulate up to registration_batch_size
        # cropped/flipped frames, then register_batch -> bin -> write into the
        # per-channel memmaps. Bounds the registration working set (suite2p-style
        # batching) and shares the FFT reference setup across the batch.
        reg_batch_size = max(1, self.config.registration_batch_size)
        buf_frames: list[np.ndarray] = []
        buf_ch: list[int] = []
        reg_progress = 0  # frames registered so far (drives the progress bar)

        def _advance_progress(n: int = 1) -> None:
            nonlocal reg_progress
            reg_progress += n
            if pbar is not None:
                pbar.update(n)
            elif reporter is not None:
                reporter.on_progress(
                    StageId.PREPROCESS, reg_progress, total_frames_est,
                    message="registering",
                )

        def _report_reading(n_buffered: int) -> None:
            # Activity indicator while a batch is being read (bar does not move).
            msg = f"reading batch ({n_buffered}/{reg_batch_size})"
            if pbar is not None:
                pbar.set_description(f"Reading ({n_buffered}/{reg_batch_size})")
            elif reporter is not None:
                reporter.on_progress(
                    StageId.PREPROCESS, reg_progress, total_frames_est, message=msg
                )

        def _flush_reg_batch() -> None:
            if not buf_frames:
                return
            if pbar is not None:
                pbar.set_description("Registration")
            t0 = time.perf_counter()
            stacked = np.stack(buf_frames).astype(np.float64)  # (N, H, W)
            if self.config.do_registration:
                # on_frame fires per frame INSIDE the batch, so the progress bar
                # advances smoothly during registration instead of jumping per batch.
                _res, reg = registrator.register_batch(
                    torch.from_numpy(stacked), return_registered=True,
                    on_frame=lambda _i, _n: _advance_progress(1),
                    need_results=False,  # results are discarded -> skip error/rf00
                )
                reg_np = reg.numpy()
            else:
                reg_np = stacked  # motion correction off: no shift
                _advance_progress(len(buf_ch))
            for j, ch in enumerate(buf_ch):
                off = write_off[ch]
                if off < T_per_ch[ch]:
                    # A subpixel Fourier shift rings around sharp edges and can
                    # overshoot the input max, so a saturated pixel lands above
                    # 65535. A raw C cast would WRAP it to near-black (and poison
                    # meanImage -> annotation -> dF/F); MATLAB's uint16() saturates.
                    r_u16 = (
                        np.clip(self._resize_image(reg_np[j]), 0, np.iinfo(np.uint16).max)
                        .round()
                        .astype(np.uint16)
                    )
                    reg_mm[ch][off] = r_u16
                    mean_sum[ch] += r_u16
                    write_off[ch] += 1
                    stats.channel_frame_counts[f"Ch{ch}"] += 1
            stats.align_seconds += time.perf_counter() - t0
            buf_frames.clear()
            buf_ch.clear()

        # Initialize per-channel frame counts
        for ch_idx in range(cycle_len):
            stats.channel_frame_counts[f"Ch{ch_idx}"] = 0

        input_metadata: dict[str, Any] = {
            "inputs": [],
            "dcimg_backend": self.config.dcimg_backend,
        }
        global_frame_index = 0
        frame_means: list[float] = []  # per-frame mean intensity (demux phase QC)
        file_starts: list[int] = []  # global index of each file's first processed frame
        # Native tqdm only when no reporter is attached; with a reporter the
        # progress is emitted via on_progress (keeps a GUI/CLI in control of
        # rendering and stdout).
        pbar = (
            None
            if reporter is not None
            else tqdm(total=total_frames_est, desc="Registration", unit="frame")
        )
        if reporter is not None:
            reporter.on_progress(StageId.PREPROCESS, 0, total_frames_est)

        read_verb = "registering" if self.config.do_registration else "reading (reg off)"
        for file_idx, input_file in enumerate(input_files):
            self._log(
                f"[preprocess] Reading + {read_verb} '{input_file.name}' "
                f"(file {file_idx + 1}/{len(input_files)})..."
            )
            stop_requested = False
            file_start_index = global_frame_index
            file_frames: list[dict[str, Any]] = []
            file_image_height: int | None = None
            file_image_width: int | None = None
            file_dtype: str | None = None
            file_bits_per_pixel: int | None = None
            for frame, frame_meta in iter_frames_with_metadata(
                input_file, dcimg_backend=self.config.dcimg_backend
            ):
                if (
                    self.config.max_frames is not None
                    and stats.total_frames >= self.config.max_frames
                ):
                    stop_requested = True
                    break

                # Cooperative cancellation: stop gracefully at a frame boundary,
                # then flush + save whatever was processed so far.
                if cancel is not None and cancel.is_set():
                    self._log(
                        "[preprocess] Cancellation requested - stopping after "
                        "current frame."
                    )
                    stop_requested = True
                    stats.cancelled = True  # the outputs are a prefix, not the run
                    break

                stats.total_frames += 1
                frame_means.append(float(frame.mean()))
                ch_idx = demux_corr.channel_at(global_frame_index)
                ch_label = f"Ch{ch_idx}"

                if file_image_height is None:
                    file_image_height = int(
                        frame_meta.get("image_height", frame.shape[0])
                    )
                    file_image_width = int(
                        frame_meta.get("image_width", frame.shape[1])
                    )
                    file_dtype = str(frame_meta.get("dtype", str(frame.dtype)))
                    file_bits_per_pixel = int(
                        frame_meta.get("bits_per_pixel", frame.dtype.itemsize * 8)
                    )
                file_frames.append(
                    {
                        "global_frame_index": int(global_frame_index),
                        "channel_index": ch_idx,
                        "channel_name": self.config.channels_name[ch_idx],
                        "channel_prop": self.config.channels_prop[ch_idx],
                        **frame_meta,
                    }
                )
                global_frame_index += 1
                # Progress is reported per REGISTERED frame (inside the batch
                # flush) so the bar reflects the slow registration work, not the
                # fast read; see _advance_progress / register_batch(on_frame=...).

                if raw_enabled and raw_off[ch_idx] < T_per_ch[ch_idx]:
                    if raw_mm[ch_idx] is None:
                        raw_mm[ch_idx] = open_reg_memmap(
                            raw_paths[ch_idx], T_per_ch[ch_idx],
                            (frame.shape[0], frame.shape[1]), dtype=frame.dtype,
                        )
                    raw_mm[ch_idx][raw_off[ch_idx]] = frame
                    raw_off[ch_idx] += 1

                frame = frame[: proc_template.shape[0], : proc_template.shape[1]]
                if self.config.flip:
                    frame = np.fliplr(frame)

                # Accumulate into the registration batch; flush (register -> bin
                # -> write) when it reaches registration_batch_size.
                buf_frames.append(np.ascontiguousarray(frame))
                buf_ch.append(ch_idx)
                _report_reading(len(buf_frames))
                if len(buf_frames) >= reg_batch_size:
                    _flush_reg_batch()

            if file_frames:
                file_starts.append(file_start_index)
            input_metadata["inputs"].append(
                {
                    "path": str(input_file),
                    "absolute_path": str(input_file.resolve()),
                    "raw_data_absolute_path": str(input_file.resolve()),
                    "suffix": input_file.suffix.lower(),
                    "file_size_bytes": int(input_file.stat().st_size),
                    "image_height": file_image_height,
                    "image_width": file_image_width,
                    "dtype": file_dtype,
                    "bits_per_pixel": file_bits_per_pixel,
                    "processed_frames": len(file_frames),
                    "frames": file_frames,
                }
            )

            if stop_requested:
                break

        _flush_reg_batch()  # register + write the final partial batch

        if pbar is not None:
            pbar.close()

        # Real per-channel counts (write_off < T_per_ch only on cancellation /
        # max_frames); flush the memmaps and release the write handles.
        real_T = list(write_off)
        for mm in reg_mm:
            mm.flush()
            handle = getattr(mm, "_mmap", None)
            if handle is not None:
                handle.close()  # release the Windows file mapping before any w+ reopen
        del reg_mm
        if raw_mm is not None:
            for mm in raw_mm:
                if mm is not None:
                    mm.flush()
                    handle = getattr(mm, "_mmap", None)
                    if handle is not None:
                        handle.close()
            del raw_mm

        mean_images = [
            (mean_sum[ch] / real_T[ch]) if real_T[ch] > 0 else mean_sum[ch]
            for ch in range(cycle_len)
        ]

        # Optional post-registration 3D smoothing over the full timeline.
        # meanImage must summarize the SMOOTHED reg data, so take the filtered
        # means the smoothing pass returns (else the stored mean is inconsistent
        # with the on-disk reg_Ch and full_channel_mean's recompute fallback).
        if self.config.filter_xyt is not None:
            fx, fy, ft = self.config.filter_xyt
            self._log(
                f"[preprocess] filter_xyt ({self.config.filter_xyt_kind}, full "
                f"timeline): x={fx}, y={fy}, t={ft}"
            )
            filtered_means = self._apply_filter_xyt(real_T, (fx, fy, ft))
            for ch, m in enumerate(filtered_means):
                if m is not None:
                    mean_images[ch] = m

        preview_images = [mean_images]

        self._write_reg_meta(proc_template, mean_images, real_T, partial=stats.cancelled)

        t_save0 = time.perf_counter()
        self._save_channel_tiffs(exp_name, real_T, raw_paths if raw_enabled else None, raw_off)
        stats.save_seconds += time.perf_counter() - t_save0

        if self.config.linear_subt:
            t_ls0 = time.perf_counter()
            self._compute_and_save_dff(real_T)
            stats.linear_subt_seconds += time.perf_counter() - t_ls0

        stats.batches_saved = sum(1 for t in real_T if t > 0)

        self._log("[preprocess] Finalizing: preview, metadata, config...")
        if preview_images:
            fig = build_fig_frames(preview_images)
            save_grayscale_png(self.output_dir / "figFrames.png", fig)

        self._run_demux_qc(stats, frame_means, file_starts, cycle_len, demux_corr)
        if frame_means:
            # Persist the per-frame means so the demux editor can re-run the QC
            # instantly without re-reading every frame.
            np.savez(
                self.output_dir / "demux_means.npz",
                means=np.asarray(frame_means, dtype=np.float32),
                file_starts=np.asarray(file_starts, dtype=np.int64),
            )

        if self.config.output_metadata_yaml:
            self._save_input_metadata_yaml(input_metadata)

        save_config(self.config, self.output_dir)

        stats.total_seconds = time.perf_counter() - run_t0
        self._log(
            f"[preprocess] Complete: {stats.total_frames} frames, "
            f"{stats.batches_saved} batch(es) in {stats.total_seconds:.1f}s "
            f"(align {stats.align_seconds:.1f}s, subtract "
            f"{stats.linear_subt_seconds:.1f}s, save {stats.save_seconds:.1f}s) "
            f"-> {self.output_dir}"
        )
        return stats

    def _binned_hw(self, shape) -> tuple[int, int]:
        """Registered/binned (H, W) for a given full-frame ``shape``.

        Mirrors ``_resize_image``'s rounding so the preallocated memmap matches
        what the resize actually produces."""
        h, w = int(shape[0]), int(shape[1])
        if self.config.binning <= 0:
            return h, w
        scale = 1.0 / float(self.config.binning)
        return max(1, int(round(h * scale))), max(1, int(round(w * scale)))

    def _run_demux_qc(
        self,
        stats: PreprocessStats,
        frame_means: list[float],
        file_starts: list[int],
        cycle_len: int,
        correction: "DemuxCorrection | None" = None,
    ) -> None:
        """Intensity-based channel-cycle QC (detection only; see asvimg.demux_qc).

        Flags a demux phase slip (a dropped frame shifting every later channel
        assignment) from the per-frame mean-intensity fingerprint — the silent
        failure the cross-file cycle-multiple warning cannot see inside a file.
        Never changes the positional demux; records a summary in ``stats`` and
        emits a QC figure.
        """
        if not self.config.demux_qc or cycle_len <= 1 or not frame_means:
            return
        from .demux_qc import analyze_demux, plot_demux_qc

        starts = file_starts or [0]
        qc = analyze_demux(frame_means, cycle_len, starts)
        stats.demux_qc_conclusive = qc.conclusive
        stats.demux_separability = float(qc.separability)
        stats.demux_dominant_period = qc.dominant_period
        stats.demux_slip_detected = qc.slip_detected
        corrected = correction is not None and not correction.is_identity()
        for msg in qc.messages:
            if qc.slip_detected and corrected:
                self._log(f"[preprocess] demux QC: {msg} (handled by demux correction)")
            elif qc.slip_detected or not qc.period_ok:
                self._log(f"[preprocess] demux QC: WARNING: {msg}")
            else:
                self._log(f"[preprocess] demux QC: {msg}")

        # Figure: skip the build for the pure-headless NullReporter (nothing would
        # consume it); otherwise emit to the reporter, or — with no reporter (the
        # asovi-preprocess CLI) — save it beside figFrames.png. Always close it
        # ourselves so a non-saving reporter (NullReporter / --no-figures) cannot
        # orphan it; a double close is a no-op.
        if isinstance(self._reporter, NullReporter):
            return
        labels = [
            f"Ch{i} {self.config.channels_name[i]}·{self.config.channels_prop[i]}"
            for i in range(cycle_len)
        ]
        try:
            fig = plot_demux_qc(
                qc,
                np.asarray(frame_means, dtype=np.float64),
                starts,
                channel_labels=labels,
                # With channels_slip (or an editor sidecar) in force, overlay the
                # fingerprint the run actually demuxes to — QC otherwise reports on
                # a positional assignment the run no longer uses.
                correction=correction,
            )
        except Exception as exc:  # noqa: BLE001 — QC visualization must not abort a run
            self._log(f"[preprocess] demux QC figure failed: {exc}")
            return
        import matplotlib.pyplot as plt

        if self._reporter is not None:
            self._reporter.on_figure(StageId.PREPROCESS, "demux_qc", fig)
        else:
            fig.savefig(self.output_dir / "fig_demux_qc.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    def _apply_filter_xyt(self, real_T: list[int], filt: tuple[int, int, int]) -> None:
        """In-place 3D uniform smoothing of every ``reg_Ch{i}.npy`` over the
        FULL timeline (no batch-boundary artifacts).

        ``filt`` = (fx, fy, ft); reg arrays are (T, H, W) so the smoothing
        window is (ft, fy, fx).  Loads one channel at a time (advanced option),
        rewriting the file trimmed to its real frame count.  Returns the
        post-filter mean image per channel (None for empty channels) so the
        caller can store a meanImage consistent with the smoothed data."""
        fx, fy, ft = filt
        uint16_max = np.iinfo(np.uint16).max
        means: list[np.ndarray | None] = [None] * self.config.cycle_len
        for ch in range(self.config.cycle_len):
            if real_T[ch] <= 0:
                continue
            path = reg_channel_path(self.output_dir, ch)
            arr = np.load(path)[: real_T[ch]].astype(np.float32)
            arr = apply_xyt_filter(arr, (ft, fy, fx), self.config.filter_xyt_kind)
            out = np.clip(arr, 0, uint16_max).round().astype(np.uint16)
            mm = open_reg_memmap(path, out.shape[0], (out.shape[1], out.shape[2]))
            mm[:] = out
            mm.flush()
            handle = getattr(mm, "_mmap", None)
            if handle is not None:
                handle.close()  # close mapping so a later reopen isn't blocked (Windows)
            del mm
            means[ch] = out.mean(axis=0).astype(np.float64)
        return means

    def _write_reg_meta(
        self,
        proc_template: np.ndarray,
        mean_images: list[np.ndarray],
        real_T: list[int],
        *,
        partial: bool = False,
    ) -> None:
        """Write the ``reg_meta.npz`` sidecar (shapes, means, channels, T).

        ``partial`` records that the run stopped early (Stop / max_frames), so a
        later ``registration_cache="cached"`` run does not mistake the prefix for
        the whole recording.
        """
        cfg = self.config
        meta: dict[str, Any] = {
            "proc_template": np.asarray(proc_template, dtype=np.float64),
            "imageSize": np.array(mean_images[0].shape),
            "channels_name": np.array(cfg.channels_name),
            "channels_prop": np.array(cfg.channels_prop),
            "T_per_ch": np.array(real_T, dtype=np.int64),
            "fps": np.array(cfg.fps),
            "flag_LinearSubt": np.array(int(cfg.linear_subt)),
            "flag_Binning": np.array(int(cfg.binning)),
            "flag_Flip": np.array(int(cfg.flip)),
            "partial": np.array(int(partial)),
        }
        for ch in range(cfg.cycle_len):
            meta[f"meanImageCh{ch}"] = np.asarray(mean_images[ch], dtype=np.float64)
        write_reg_meta(self.output_dir, meta)

    def _previous_run_was_partial(self) -> bool:
        """Is the reg_Ch*.npy already on disk the output of a COMPLETE run?

        No reg_meta.npz means the run never reached the end (it is written last),
        so the reg_Ch files are a preallocated husk, not data.
        """
        meta_path = reg_meta_path(self.output_dir)
        if not meta_path.exists():
            return True
        try:
            with np.load(meta_path, allow_pickle=True) as z:
                return "partial" in z.files and bool(int(np.asarray(z["partial"])))
        except (OSError, ValueError, KeyError):
            return True  # unreadable sidecar: treat as incomplete

    def _save_channel_tiffs(
        self,
        exp_name: str,
        real_T: list[int],
        raw_paths: list[Path] | None,
        raw_off: list[int],
    ) -> None:
        """Optional per-channel raw / registered TIFF stacks (one file each).

        Raw frames were streamed to per-channel temp memmaps (``raw_paths``);
        they are read back one channel at a time and the temp files removed."""
        cfg = self.config
        tiff_base = self.output_dir / "tiffs"

        if raw_paths is not None:
            self._log(f"[preprocess] Saving raw per-channel TIFFs -> {tiff_base}")
            for ch in range(cfg.cycle_len):
                if raw_off[ch] <= 0 or not raw_paths[ch].exists():
                    continue
                tok = cfg.channel_token(ch)
                ch_dir = tiff_base / f"raw_{tok}"
                ch_dir.mkdir(parents=True, exist_ok=True)
                raw = np.load(raw_paths[ch])[: raw_off[ch]]  # (T, Hf, Wf)
                save_stack_tiff(
                    ch_dir / f"{exp_name}_raw_{tok}.tif",
                    np.moveaxis(raw, 0, 2),  # (Hf, Wf, T)
                    tiff_format=cfg.tiff_format, compression=cfg.tiff_compression,
                )
                del raw
                raw_paths[ch].unlink(missing_ok=True)  # drop the temp memmap

        if cfg.save_registered_each_ch:
            self._log(f"[preprocess] Saving registered per-channel TIFFs -> {tiff_base}")
            for ch in range(cfg.cycle_len):
                if real_T[ch] <= 0:
                    continue
                tok = cfg.channel_token(ch)
                ch_dir = tiff_base / f"reg_{tok}"
                ch_dir.mkdir(parents=True, exist_ok=True)
                arr = np.moveaxis(load_reg_channel(self.output_dir, ch), 0, 2)  # (H,W,T)
                save_stack_tiff(
                    ch_dir / f"{exp_name}_reg_{tok}.tif", arr,
                    tiff_format=cfg.tiff_format, compression=cfg.tiff_compression,
                )

    def _compute_and_save_dff(self, real_T: list[int]) -> None:
        """Full-timeline dF/F for EVERY channels_name group with a source,
        streamed in ROW-STRIPS so peak memory is bounded, writing
        ``dff_{name}.npy`` (T, H, W) float32 with a GLOBAL baseline.

        A group WITH a donner gets the WFCI linear-subtraction dF/F; a group
        WITHOUT one gets the source averaged against a per-pixel
        ``baseline_percentile`` baseline — written to the same ``dff_{name}.npy``.
        So every consumer (ROI, export, movies, PCA/ICA) reads ONE definition of
        "this group's dF/F" from one place instead of each recomputing it.
        (``annotation._load_name_dff_stack`` can still compute the percentile dF/F
        in RAM, but only as a fallback for an output dir written before this
        method wrote the file.)

        Striping rows (not time) keeps every pixel's whole timeline in the
        strip, so the per-pixel baseline percentile is exact.  ``use_mmap=True``
        reads reg_Ch via memmap (strips come off disk → low peak); False loads
        each channel into RAM.
        """
        cfg = self.config
        cycle_len = cfg.cycle_len
        use_mmap = cfg.use_mmap
        fps_channel = float(cfg.fps) / cycle_len
        if cfg.detrend or cfg.baseline_percentile_highpass:
            self._log(
                f"[preprocess] dF/F pre-steps: detrend={cfg.detrend}, "
                f"high-pass={cfg.baseline_percentile_highpass} "
                f"(cutoff={cfg.baseline_percentile_highpass_sec}s, "
                f"rank={cfg.baseline_percentile_highpass_rank})"
            )

        # Progress is counted in IMAGE ROWS across every group: the correction is
        # streamed row-strip by row-strip and each row costs the same (one
        # regression + baseline percentile per pixel over the whole timeline), so
        # rows are the one unit that advances evenly. Plan the total up front
        # (cheap: only the .npy headers are read) so the bar has a denominator.
        groups = self._dff_plan(real_T)
        rows_total = sum(g[3] for g in groups)
        rows_done = 0
        pbar = (
            tqdm(total=rows_total, desc="dF/F", unit="row")
            if self._reporter is None and rows_total
            else None
        )

        def _report(done: int, message: str) -> None:
            # The bar's unit changes here (frames -> rows), so every message says
            # "dF/F" — otherwise the GUI overlay just looks like the registration
            # bar jumping backwards.
            nonlocal rows_done
            rows_done = done
            if pbar is not None:
                pbar.n = done
                pbar.set_description(f"dF/F {message}")
                pbar.refresh()
            elif self._reporter is not None:
                self._reporter.on_progress(
                    StageId.PREPROCESS, done, rows_total,
                    message=f"dF/F {message}",
                )

        base_skip = cfg.resolved_start_initial_frames
        for name, src_idx, don_idx, _H, n in groups:
            src_mm = [load_reg_channel(self.output_dir, i, mmap=use_mmap) for i in src_idx]
            don_mm = [load_reg_channel(self.output_dir, i, mmap=use_mmap) for i in don_idx]
            H, W = int(src_mm[0].shape[1]), int(src_mm[0].shape[2])
            kind = "donner regression" if don_idx else f"p{cfg.baseline_percentile:g} baseline"
            self._log(
                f"[preprocess] dF/F: {name} (source={src_idx}, donner={don_idx}, "
                f"T={n}, {kind})"
            )
            if len(src_idx) > 1:
                # Same-name sources are ONE signal: they are averaged, which halves
                # the effective sampling rate. Right for replicate frames of one
                # fluorophore; wrong if they are actually different channels.
                # (ASCII only: _log prints to the console, which is cp932 on a
                # Japanese Windows and cannot encode dashes/arrows.)
                self._log(
                    f"[preprocess] NOTE: '{name}' has {len(src_idx)} source channels "
                    f"{src_idx}: they are AVERAGED into one signal. Give distinct "
                    f"channels distinct names if that is not what you want."
                )
            group_row0 = rows_done  # rows finished by the groups before this one

            src_times = np.arange(n, dtype=np.float64) * cycle_len + float(np.mean(src_idx))
            don_times = (
                np.arange(n, dtype=np.float64) * cycle_len + float(np.mean(don_idx))
                if don_idx else None
            )
            foi_start = cfg.resolved_start_initial_frames
            foi_end = max(foi_start + 1, n - cfg.ignore_last_frames)
            foi = np.arange(foi_start, foi_end)
            if foi.size <= 2:
                foi = np.arange(0, n)

            dff_mm = open_reg_memmap(
                dff_name_path(self.output_dir, name), n, (H, W), dtype=np.float32
            )
            # hemo variance-explained needs a donner to regress; there is none here
            want_hemovar = cfg.hemovar_qc and bool(don_idx)
            hemovar_map = np.empty((H, W), dtype=np.float32) if want_hemovar else None
            try:
                # Row-strip height budgeted so a strip's float64 working set
                # (strip * W * n) stays ~<=256 MB regardless of T.
                strip_h = int(np.clip(256_000_000 // max(1, W * n * 8), 1, H))
                for r0 in range(0, H, strip_h):
                    r1 = min(r0 + strip_h, H)

                    def _avg(mms):
                        return np.mean(
                            [
                                np.moveaxis(
                                    np.asarray(m[:n, r0:r1, :], dtype=np.float64), 0, 2
                                )
                                for m in mms
                            ],
                            axis=0,
                        )  # (strip, W, n)

                    # Reading a strip off the reg_Ch memmaps is itself slow enough
                    # to look like a hang (use_mmap streams it from disk), so say
                    # so before the bar sits still.
                    _report(group_row0 + r0, f"{name}: reading rows {r0}-{r1}/{H}")
                    src_strip = _avg(src_mm)

                    if not don_idx:
                        # No donner: there is nothing to regress out, so the group's
                        # dF/F is the source average against a per-pixel percentile
                        # baseline — the same definition every consumer used to
                        # recompute for itself.
                        base_win = (
                            src_strip[..., base_skip:]
                            if 0 < base_skip < n else src_strip
                        )
                        base = np.percentile(
                            base_win, cfg.baseline_percentile, axis=2, keepdims=True
                        )
                        with np.errstate(divide="ignore", invalid="ignore"):
                            image_df = np.where(
                                base > 0, (src_strip - base) / base, 0.0
                            )
                        dff_mm[:, r0:r1, :] = np.moveaxis(image_df, 2, 0).astype(np.float32)
                        _report(group_row0 + r1, f"{name}: rows {r1}/{H}")
                        continue

                    don_strip = _avg(don_mm)
                    result = wfci_corrected_df_vectorized(
                        src_strip, don_strip, foi,
                        src_times=src_times, donor_times=don_times,
                        baseline_percentile=cfg.baseline_percentile,
                        baseline_ignore_initial=cfg.resolved_start_initial_frames,
                        detrend=cfg.detrend,
                        highpass=cfg.baseline_percentile_highpass,
                        highpass_cutoff_sec=cfg.baseline_percentile_highpass_sec,
                        highpass_rank=cfg.baseline_percentile_highpass_rank,
                        fps_channel=fps_channel,
                        return_hemovar=want_hemovar,
                        # wfci sub-strips this strip further (32 rows); each of its
                        # steps advances the global row counter.
                        progress=lambda done, _tot, _r0=r0: _report(
                            group_row0 + _r0 + done, f"{name}: rows {_r0 + done}/{H}"
                        ),
                    )
                    if want_hemovar:
                        image_df, _base, hv = result
                        hemovar_map[r0:r1] = hv.astype(np.float32)
                    else:
                        image_df, _base = result
                    dff_mm[:, r0:r1, :] = np.moveaxis(image_df, 2, 0).astype(np.float32)
                dff_mm.flush()
            finally:
                del dff_mm
                for m in src_mm + don_mm:
                    handle = getattr(m, "_mmap", None)
                    if handle is not None:
                        handle.close()
            _report(group_row0 + H, f"{name}: done")

            if want_hemovar and hemovar_map is not None:
                self._save_hemovar(name, hemovar_map)

        if pbar is not None:
            pbar.close()

    def _dff_plan(self, real_T: list[int]) -> list[tuple[str, list[int], list[int], int, int]]:
        """``(name, src_idx, don_idx, H, T)`` per group with a source, in run order.

        ``don_idx`` is empty for a donner-less group: it still gets a
        ``dff_{name}.npy``, computed against a per-pixel percentile baseline
        instead of the donner regression.  Every consumer (ROI, export, PCA/ICA)
        then reads ONE definition of "this group's dF/F" from one place, instead
        of each recomputing it — which is how PCA ended up analysing raw
        fluorescence while ROI analysed dF/F.

        Lets ``_compute_and_save_dff`` know the total row count before it starts,
        so progress has a denominator.  Reads only the ``.npy`` headers (shape),
        never the pixel data.
        """
        plan: list[tuple[str, list[int], list[int], int, int]] = []
        for name, group in self.config.channel_groups().items():
            src_idx = group["source_indices"]
            don_idx = group["donner_indices"]
            if not src_idx:
                continue
            n = min(real_T[i] for i in (src_idx + don_idx))
            if n <= 0:
                continue
            arr = np.load(reg_channel_path(self.output_dir, src_idx[0]), mmap_mode="r")
            H = int(arr.shape[1])
            del arr  # header read only; drop the mapping straight away
            plan.append((name, src_idx, don_idx, H, n))
        return plan

    def _save_hemovar(self, name: str, hemovar_map: np.ndarray) -> None:
        """Save the per-group hemo variance-explained (R²) QC map + figure.

        ``hemovar_map`` (H, W) in [0, 1] is the fraction of source variance the
        donner regression explains (WidefieldImager ``hemoVar`` analogue).
        """
        np.save(self.output_dir / f"hemovar_{name}.npy", hemovar_map.astype(np.float32))
        med = float(np.median(hemovar_map))
        # ASCII only: a cp932 console raises UnicodeEncodeError on 'R²' and would
        # take the whole (default-on) preprocess down with it.
        self._log(
            f"[preprocess] hemo-var QC: {name} median R^2={med:.3f} -> hemovar_{name}.npy"
        )
        if isinstance(self._reporter, NullReporter):
            return
        from .wfci import plot_hemovar_map

        try:
            fig = plot_hemovar_map(hemovar_map, name)
        except Exception as exc:  # noqa: BLE001 — QC figure must not abort a run
            self._log(f"[preprocess] hemo-var figure failed: {exc}")
            return
        import matplotlib.pyplot as plt

        if self._reporter is not None:
            self._reporter.on_figure(StageId.PREPROCESS, f"hemovar_{name}", fig)
        else:
            fig.savefig(self.output_dir / f"fig_hemovar_{name}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    def _delete_previous_outputs(self, exp_name: str) -> None:
        ext = self.config.output_format
        for pattern in [
            # Current intermediate layout
            "reg_Ch*.npy",
            "dff_*.npy",
            "hemovar_*.npy",
            "reg_meta.npz",
            # The auto-built template is DERIVED (from template_ch / template_stride
            # / the input files) but was cached forever: changing either knob was
            # silently ignored, even with delete=True. A user-supplied `template`
            # is a config path and is never touched.
            "templateImage.mat",
            "_rawtmp_Ch*.npy",  # stale temp raw memmaps from an interrupted run
            "_warptmp_*.npy",  # stale temp warp memmaps from an interrupted export
            # Legacy per-batch payloads
            f"frameRoiCh_{exp_name}_*.{ext}",
            f"frameRoiDf_*_{exp_name}_*.{ext}",
            f"frameRoiBlUv_{exp_name}_*.{ext}",
            f"frameRoiDf_{exp_name}_*.{ext}",
            f"frameRoiBl_{exp_name}_*.{ext}",
        ]:
            for old_file in self.output_dir.glob(pattern):
                old_file.unlink(missing_ok=True)

        # The ICA basis and the PC/IC maps are DERIVED from a dF/F that is about
        # to be rewritten, so they go. ica_exclusion.json does NOT: it is the one
        # artifact here that cannot be re-derived (a human judged those components
        # to be artifacts), and Run All would otherwise destroy it in preprocess
        # before the ICA stage ever sees it. It survives, exactly like marks.mat.
        # The basis it refers to is gone, so the ICA stage must re-fit before
        # anything can be applied -- and it re-opens the picker preselected with
        # this choice, which is a review rather than a re-do.
        import shutil

        for sub in ("ica", "pca_images", "ica_images"):
            d = self.output_dir / sub
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)

        tiff_base = self.output_dir / "tiffs"
        if tiff_base.exists():
            for prefix in ("raw_Ch", "reg_Ch", "annot_Ch"):
                for ch_dir in tiff_base.glob(f"{prefix}*"):
                    if not ch_dir.is_dir():
                        continue
                    for pat in (f"{exp_name}_*.tif", f"{exp_name}_*.ome.tif"):
                        for old_file in ch_dir.glob(pat):
                            old_file.unlink(missing_ok=True)

    def _load_or_create_template(self, input_files: list[Path]) -> np.ndarray:
        if self.config.template is not None:
            return self._load_user_template(input_files)

        # Cache the auto-template alongside the outputs, not in the raw source
        # directory (keeps source dirs pristine / works on read-only inputs).
        # A legacy input_dir/templateImage.mat (e.g. written by the MATLAB
        # pipeline) is still read if present, for backward/tooling compatibility.
        template_out = self.output_dir / "templateImage.mat"
        for template_file in (template_out, self.input_dir / "templateImage.mat"):
            if template_file.exists():
                return load_template_mat(template_file).astype(np.float64)

        last_file = input_files[-1]
        n_frames = get_frame_count(last_file)
        cycle_len = self.config.cycle_len
        template_ch = self.config.template_ch
        # Sample frames belonging to template_ch (one per cycle, every template_stride cycles)
        frame_indices = list(
            range(template_ch, n_frames, self.config.template_stride * cycle_len)
        )
        if not frame_indices:
            frame_indices = [min(template_ch, max(0, n_frames - 1))]

        tmp = load_frames_by_indices(last_file, frame_indices)
        proc_template = np.mean(np.stack(tmp, axis=2), axis=2)

        savemat(template_out, {"proc_template": proc_template})
        return proc_template

    def _load_user_template(self, input_files: list[Path]) -> np.ndarray:
        template_path = Path(self.config.template)
        if not template_path.is_absolute():
            template_path = self.input_dir / template_path
        template = load_template_image(template_path)

        sample_frame = load_frames_by_indices(input_files[0], [0])[0]
        if template.shape != sample_frame.shape:
            raise ValueError(
                f"Template image shape {template.shape} does not match "
                f"input frame shape {sample_frame.shape} "
                f"(from {input_files[0].name}). "
                f"Provide a template with matching dimensions."
            )
        return template

    def _save_input_metadata_yaml(self, payload: dict[str, Any]) -> None:
        out_path = self.output_dir / "input_metadata.yaml"
        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        if self.config.binning <= 0:
            return image.astype(np.float64)

        scale = 1.0 / float(self.config.binning)
        new_h = max(1, int(round(image.shape[0] * scale)))
        new_w = max(1, int(round(image.shape[1] * scale)))
        return resize(
            image,
            (new_h, new_w),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
            mode="reflect",
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_from_args(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    if args.input_dir:
        config.input_dir = args.input_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.exp_name:
        config.exp_name = args.exp_name
    if args.output_format:
        config.output_format = args.output_format
    if args.max_frames is not None:
        config.max_frames = args.max_frames
    if args.dcimg_backend:
        config.dcimg_backend = args.dcimg_backend
    if args.no_metadata_yaml:
        config.output_metadata_yaml = False

    runner = PreprocessRunner(config)
    stats = runner.run()

    if args.print_stats:
        print(yaml.safe_dump(asdict(stats), sort_keys=False, allow_unicode=True))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ASoVi preprocess (registration + WFCI) runner"
    )
    parser.add_argument(
        "--config",
        default="pipeline01_config.yaml",
        help="YAML config path",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Input directory containing tif/tiff/dcimg/sifx files",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--exp-name", default=None, help="Override experiment name")
    parser.add_argument(
        "--output-format",
        choices=sorted(SUPPORTED_OUTPUT_FORMATS),
        default=None,
        help="Output format: mat, npy, or h5",
    )
    parser.add_argument(
        "--print-stats",
        action="store_true",
        help="Print run timing and processing stats as YAML",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame processing limit for quick benchmark runs",
    )
    parser.add_argument(
        "--dcimg-backend",
        choices=["auto", "sdk", "native"],
        default=None,
        help="DCIMG backend selection (default from config: auto)",
    )
    parser.add_argument(
        "--no-metadata-yaml",
        action="store_true",
        help="Disable input metadata YAML output",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_from_args(args)


if __name__ == "__main__":
    main()
