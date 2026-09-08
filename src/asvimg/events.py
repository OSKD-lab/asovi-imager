"""Progress / logging reporting channel for the pipeline runner.

A :class:`ProgressReporter` is the single sink through which the headless
:class:`~asvimg.runner.session.PipelineSession` (and the
``PreprocessRunner``) emit stage transitions, fine-grained progress, log
lines, and matplotlib figures.  Concrete reporters render those events for a
terminal (:class:`TqdmReporter`), discard them (:class:`NullReporter`),
record them for tests (:class:`RecordingReporter`), or — in the GUI — enqueue
them for the render thread.

The Protocol methods take primitive arguments (not event objects) so the many
call sites stay terse; the small event dataclasses exist for reporters that
want to queue or record a single value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class StageId(str, Enum):
    """Canonical identifiers for the pipeline stages."""

    PREPROCESS = "preprocess"
    PCA = "pca"
    ICA = "ica"
    ANNOTATION = "annotation"
    ROI = "roi"
    CORRELATION = "correlation"
    EXPORT = "export"

    def __str__(self) -> str:  # so f-strings print "preprocess" not "StageId.PREPROCESS"
        return self.value


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
    STALE = "stale"

    def __str__(self) -> str:
        return self.value


# --- Event dataclasses (used by queueing / recording reporters) ----------


@dataclass
class StageEvent:
    stage: str
    status: StageStatus
    message: str = ""
    elapsed: float | None = None


@dataclass
class ProgressEvent:
    stage: str
    current: int
    total: int
    message: str = ""


@dataclass
class LogEvent:
    stage: str
    text: str
    level: str = "info"


@dataclass
class FigureEvent:
    stage: str
    key: str
    figure: "Figure"


@runtime_checkable
class ProgressReporter(Protocol):
    """Sink for pipeline events. All methods must be cheap and non-blocking."""

    def on_stage(
        self, stage: str, status: StageStatus, *, message: str = "",
        elapsed: float | None = None,
    ) -> None: ...

    def on_progress(
        self, stage: str, current: int, total: int, *, message: str = "",
    ) -> None: ...

    def on_log(self, stage: str, text: str, *, level: str = "info") -> None: ...

    def on_figure(self, stage: str, key: str, figure: "Figure") -> None: ...


class NullReporter:
    """Discards every event. Default for the headless core."""

    def on_stage(self, stage, status, *, message="", elapsed=None) -> None:
        return None

    def on_progress(self, stage, current, total, *, message="") -> None:
        return None

    def on_log(self, stage, text, *, level="info") -> None:
        return None

    def on_figure(self, stage, key, figure) -> None:
        return None


class RecordingReporter:
    """Records every event for assertions in tests."""

    def __init__(self) -> None:
        self.stages: list[StageEvent] = []
        self.progress: list[ProgressEvent] = []
        self.logs: list[LogEvent] = []
        self.figures: list[FigureEvent] = []

    def on_stage(self, stage, status, *, message="", elapsed=None) -> None:
        self.stages.append(StageEvent(str(stage), status, message, elapsed))

    def on_progress(self, stage, current, total, *, message="") -> None:
        self.progress.append(ProgressEvent(str(stage), current, total, message))

    def on_log(self, stage, text, *, level="info") -> None:
        self.logs.append(LogEvent(str(stage), text, level))

    def on_figure(self, stage, key, figure) -> None:
        self.figures.append(FigureEvent(str(stage), key, figure))


class CompositeReporter:
    """Fan an event out to several reporters."""

    def __init__(self, reporters: list[ProgressReporter]) -> None:
        self._reporters = list(reporters)

    def on_stage(self, stage, status, *, message="", elapsed=None) -> None:
        for r in self._reporters:
            r.on_stage(stage, status, message=message, elapsed=elapsed)

    def on_progress(self, stage, current, total, *, message="") -> None:
        for r in self._reporters:
            r.on_progress(stage, current, total, message=message)

    def on_log(self, stage, text, *, level="info") -> None:
        for r in self._reporters:
            r.on_log(stage, text, level=level)

    def on_figure(self, stage, key, figure) -> None:
        for r in self._reporters:
            r.on_figure(stage, key, figure)


class TqdmReporter:
    """Terminal reporter: per-stage tqdm bars, printed logs, saved figures.

    Used by the ``asovi-run`` CLI.  ``on_figure`` writes each figure to
    ``figure_dir`` (when given) as ``{stage}_{key}.png`` and closes it.
    """

    def __init__(self, figure_dir=None, *, save_figures: bool = True) -> None:
        from pathlib import Path

        self._figure_dir = Path(figure_dir) if figure_dir is not None else None
        self._save_figures = save_figures
        self._bars: dict[str, Any] = {}

    def on_stage(self, stage, status, *, message="", elapsed=None) -> None:
        key = str(stage)
        bar = self._bars.pop(key, None)
        if bar is not None:
            bar.close()
        tag = str(status).upper()
        suffix = f" ({elapsed:.1f}s)" if elapsed is not None else ""
        extra = f" - {message}" if message else ""
        print(f"[{key}] {tag}{suffix}{extra}", flush=True)

    def on_progress(self, stage, current, total, *, message="") -> None:
        from tqdm.auto import tqdm

        key = str(stage)
        bar = self._bars.get(key)
        if bar is None:
            bar = tqdm(total=total, desc=key, unit="step")
            self._bars[key] = bar
        if total and bar.total != total:
            bar.total = total
            bar.refresh()
        bar.n = current
        if message:
            bar.set_postfix_str(message, refresh=False)
        bar.refresh()

    def on_log(self, stage, text, *, level="info") -> None:
        prefix = "" if str(text).startswith("[") else f"[{stage}] "
        print(f"{prefix}{text}", flush=True)

    def on_figure(self, stage, key, figure) -> None:
        if not self._save_figures or self._figure_dir is None:
            return
        self._figure_dir.mkdir(parents=True, exist_ok=True)
        out = self._figure_dir / f"{stage}_{key}.png"
        try:
            figure.savefig(out, dpi=120, bbox_inches="tight")
            print(f"[{stage}] figure saved: {out.name}", flush=True)
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)
