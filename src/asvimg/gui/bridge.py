"""GuiBridge — the only place that knows about threads and subprocesses.

It plays three roles for a running :class:`PipelineSession` (all on the worker
thread):

* **reporter** — every ``on_*`` call converts the payload to a plain tuple and
  enqueues it; the main/render thread drains the queue and mutates dpg.
  Figures are rasterised to RGBA *here* (worker thread) so matplotlib/pyplot is
  only ever touched from one thread.
* **marks provider** / **ica provider** — ``request`` serialises its inputs,
  launches the matching interactive GUI in a child process, and blocks the
  worker (not the render loop) until the child writes its result.

Cancellation flows through a shared ``CancellationToken``; cancelling also
terminates any in-flight interactive child.
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import time
from pathlib import Path

from asvimg import CancellationToken

from . import figures, ipc

_CPSELECT_MODULE = "asvimg.gui.subproc_cpselect"
_ICA_MODULE = "asvimg.gui.subproc_ica"


class GuiBridge:
    def __init__(self) -> None:
        self.events: queue.Queue = queue.Queue()
        self.cancel = CancellationToken()
        self._active_proc: subprocess.Popen | None = None
        self.marks_provider = _MarksAdapter(self)
        self.ica_provider = _IcaAdapter(self)

    # --- ProgressReporter (worker thread → queue) -------------------------

    def on_stage(self, stage, status, *, message="", elapsed=None) -> None:
        self.events.put(("stage", str(stage), str(status), message, elapsed))

    def on_progress(self, stage, current, total, *, message="") -> None:
        self.events.put(("progress", str(stage), int(current), int(total), message))

    def on_log(self, stage, text, *, level="info") -> None:
        self.events.put(("log", str(stage), str(text), level))

    def on_figure(self, stage, key, figure) -> None:
        try:
            w, h, data = figures.figure_to_rgba(figure)
        finally:
            try:
                import matplotlib.pyplot as plt

                plt.close(figure)
            except Exception:
                pass
        self.events.put(("figure", str(stage), key, w, h, data))

    def drain(self, handler) -> None:
        """Pop all pending events (call on the main thread)."""
        while True:
            try:
                ev = self.events.get_nowait()
            except queue.Empty:
                return
            handler(ev)

    def request_cancel(self) -> None:
        self.cancel.cancel()
        proc = self._active_proc
        if proc is not None and proc.poll() is None:
            proc.terminate()

    # --- interactive children (worker thread, blocking) -------------------

    def _cpselect(self, config, output_dir, atlas, source_img, extra_imgs, extra_labels):
        default_label = (
            f"Mean Ch{config.ch_for_annotation} "
            f"({config.channels_name[config.ch_for_annotation]})"
        )
        request = {
            "source_imgs": [source_img] + list(extra_imgs or []),
            "ref_rgb": atlas.image_rgb,
            "boundaries": atlas.boundaries,
            "labels": [default_label] + list(extra_labels or []),
            "n_default_pairs": 2,
            "title": "Control Point Selection",
        }
        result = self._run_child(_CPSELECT_MODULE, request)
        if result is None:
            self.on_log("annotation", "[gui] control-point selection cancelled.")
            return None, None
        src_pts, ref_pts = result

        from asvimg import save_marks

        marks_path = Path(output_dir) / "marks.mat"
        if marks_path.exists():
            old = marks_path.with_suffix(".mat.old")
            old.unlink(missing_ok=True)
            marks_path.rename(old)
        save_marks(marks_path, src_pts, ref_pts)
        self.on_log("annotation", f"[gui] saved {len(src_pts)} marks -> {marks_path.name}")
        return src_pts, ref_pts

    def _ica(self, name, ica_result, image_df, *, index=0, total=1, preselected=None):
        """Open the IC picker for ONE group. Returns None when cancelled.

        None is NOT []: with the exclusion able to reach the saved outputs, a
        cancelled or crashed picker collapsing to "exclude nothing" would stamp
        un-denoised data as reviewed. The session keeps the previous record instead.
        """
        title = (
            f"ICA - {name} ({index + 1}/{total}) - Click to EXCLUDE"
            if total > 1
            else f"ICA - {name} - Click to EXCLUDE"
        )
        result = self._run_child(
            _ICA_MODULE,
            {"ica_result": ica_result, "title": title,
             "preselected": list(preselected or [])},
        )
        if result is None:
            self.on_log("ica", f"[gui] {name}: IC selection cancelled.")
            return None
        return list(result)

    def _run_child(self, module: str, request_obj):
        session_dir = ipc.new_session_dir()
        req_path = session_dir / "req.pkl"
        res_path = session_dir / "res.pkl"
        try:
            ipc.dump(request_obj, req_path)
            cmd = ipc.child_command(module, str(req_path), str(res_path))
            proc = subprocess.Popen(
                cmd, cwd=str(ipc.project_root()), env=ipc.child_env()
            )
            self._active_proc = proc
            while proc.poll() is None:
                if self.cancel.is_set():
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    return None
                time.sleep(0.05)
            # A Stop from the main thread may terminate the child directly,
            # letting the loop exit via poll() before the cancel branch — treat
            # any requested cancellation as a clean cancel, not a child error.
            if self.cancel.is_set():
                return None
            if proc.returncode != 0:
                self.on_log(
                    "annotation",
                    f"[gui] child {module} exited with code {proc.returncode}",
                    level="error",
                )
                return None
            if not res_path.exists():
                return None
            return ipc.load(res_path)
        finally:
            self._active_proc = None
            shutil.rmtree(session_dir, ignore_errors=True)


class _MarksAdapter:
    def __init__(self, bridge: GuiBridge) -> None:
        self._b = bridge

    def request(self, config, output_dir, atlas, source_img, *, extra_imgs=None, extra_labels=None):
        # cpselect (child process) opens for annotation="gui", and also for
        # "cache" when no marks.mat exists yet — cache falls back to picking
        # points on first use instead of silently skipping the warp.  Every
        # other case ("cache" with marks / coords / False) resolves without a
        # window, so annotation runs reflect the previous result.
        from asvimg import needs_cpselect, resolve_marks

        if needs_cpselect(config, output_dir):
            if config.annotation == "cache":
                self._b.on_log(
                    "annotation",
                    "[annot] no cached marks.mat — opening cpselect to pick points.",
                )
            return self._b._cpselect(
                config, output_dir, atlas, source_img, extra_imgs, extra_labels
            )
        self._b.on_log("annotation", f"[annot] annotation={config.annotation!r} "
                                     "- resolving without cpselect")
        return resolve_marks(
            config, output_dir, atlas, source_img,
            extra_imgs=extra_imgs, extra_labels=extra_labels,
        )


class _IcaAdapter:
    def __init__(self, bridge: GuiBridge) -> None:
        self._b = bridge

    def request(self, name, ica_result, image_df, *, index=0, total=1,
                preselected=None):
        return self._b._ica(
            name, ica_result, image_df,
            index=index, total=total, preselected=preselected,
        )
