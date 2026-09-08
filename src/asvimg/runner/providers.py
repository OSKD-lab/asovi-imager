"""Injectable providers for the two interactive pipeline steps.

The headless :class:`~asvimg.runner.session.PipelineSession` never
opens a window.  When it reaches control-point selection or ICA component
exclusion it calls a *provider*, which either resolves the answer
non-interactively (the headless defaults here) or — in the GUI — pops the
corresponding window in a child process and returns the user's choice.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from asvimg import ACCFv3, IcaResult, PipelineConfig


@runtime_checkable
class MarksProvider(Protocol):
    """Supplies atlas control points (source ↔ reference)."""

    def request(
        self,
        config: "PipelineConfig",
        output_dir: Path,
        atlas: "ACCFv3",
        source_img: np.ndarray,
        *,
        extra_imgs: list[np.ndarray] | None = None,
        extra_labels: list[str] | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]: ...


@runtime_checkable
class IcaExclusionProvider(Protocol):
    """Supplies the IC indices to exclude, for ONE channel group.

    ``None`` means *cancelled / no answer* and is NOT the same as ``[]``.  With
    the exclusion able to reach the saved outputs, a crashed or closed IC window
    collapsing to "exclude nothing" would stamp un-denoised data as denoised, so
    the caller keeps whatever was recorded before and skips the group instead.

    ``index`` / ``total`` let an interactive front-end say "group 2 of 3";
    ``preselected`` offers the previously saved choice for that group.
    """

    def request(
        self,
        name: str,
        ica_result: "IcaResult",
        image_df: np.ndarray,
        *,
        index: int = 0,
        total: int = 1,
        preselected: list[int] | None = None,
    ) -> list[int] | None: ...


class ConfigMarksProvider:
    """Headless marks provider — resolves from ``config.annotation``.

    Reuses :func:`asvimg.resolve_marks` for the non-interactive modes
    (``"cache"`` / coordinate tuple / ``False``).  ``annotation == "gui"`` is
    rejected with a clear error because no display is available headlessly;
    a GUI front-end injects an interactive provider for that case.
    """

    def request(
        self,
        config: "PipelineConfig",
        output_dir: Path,
        atlas: "ACCFv3",
        source_img: np.ndarray,
        *,
        extra_imgs: list[np.ndarray] | None = None,
        extra_labels: list[str] | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        from asvimg import resolve_marks

        if config.annotation == "gui":
            raise RuntimeError(
                "annotation='gui' needs an interactive MarksProvider "
                "(run via the GUI, or set annotation to 'cache', a "
                "([src],[ref]) coordinate pair, or False)."
            )
        return resolve_marks(
            config,
            output_dir,
            atlas,
            source_img,
            extra_imgs=extra_imgs,
            extra_labels=extra_labels,
        )


class NoExclusionIcaProvider:
    """Headless ICA provider — never excludes any component."""

    def request(self, name, ica_result, image_df, *, index=0, total=1,
                preselected=None) -> list[int]:
        return []


class ConfigIcaProvider:
    """Headless ICA provider — resolves from ``config.ica_exclusion``.

    - ``"cache"`` (default) → replay ``ica_exclusion.json`` (the choice a GUI
      session recorded).  Nothing recorded for a group → exclude nothing.
    - ``{"GCaMP": [2, 7]}`` → those indices, spelled out in the config.
    - ``False`` → exclude nothing.
    - ``"gui"`` → rejected: no display headlessly (a GUI front-end injects its
      own provider for that).
    """

    def __init__(self, config: "PipelineConfig", output_dir) -> None:
        self.config = config
        self.output_dir = Path(output_dir)

    def request(self, name, ica_result, image_df, *, index=0, total=1,
                preselected=None) -> list[int]:
        mode = self.config.ica_exclusion
        if mode == "gui":
            raise RuntimeError(
                "ica_exclusion='gui' needs a display. Run the GUI, or set it to "
                "'cache' (replay ica_exclusion.json) / False / {group: [ics]}."
            )
        if isinstance(mode, dict):
            # A group the dict does not name has NOT been answered for. Returning
            # [] here would record "reviewed, flagged nothing" over a human's
            # saved choice for that group -- the exact clobber None exists to
            # prevent.
            if name not in mode:
                return None
            return [int(i) for i in mode[name]]
        if mode in (False, None):
            # "exclude nothing on this run" is about what gets APPLIED
            # (resolve_exclusion returns [] for this mode); it is not a judgement
            # about the components, so it must not overwrite one.
            return None
        # "cache"
        from asvimg.ica_state import load_exclusions

        return list(load_exclusions(self.output_dir).get(name, []))


# --- Test doubles ---------------------------------------------------------


class FakeMarksProvider:
    """Returns canned control points (for tests / scripted runs)."""

    def __init__(
        self,
        src_pts: np.ndarray | list,
        ref_pts: np.ndarray | list,
        *,
        save: bool = True,
    ) -> None:
        self.src_pts = np.asarray(src_pts, dtype=np.float64)
        self.ref_pts = np.asarray(ref_pts, dtype=np.float64)
        self.save = save

    def request(
        self, config, output_dir, atlas, source_img, *,
        extra_imgs=None, extra_labels=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.save:
            from asvimg import save_marks

            save_marks(Path(output_dir) / "marks.mat", self.src_pts, self.ref_pts)
        return self.src_pts, self.ref_pts


class FakeIcaProvider:
    """Returns a fixed exclusion (for tests).

    ``exclude`` is either one list (used for every group) or ``{group: [ics]}``.
    ``None`` for a group models a cancelled / crashed selection window.
    """

    def __init__(self, exclude: "list[int] | dict[str, list[int] | None] | None" = None) -> None:
        self.exclude = exclude if isinstance(exclude, dict) else list(exclude or [])
        self.calls: list[tuple[str, int, int]] = []

    def request(self, name, ica_result, image_df, *, index=0, total=1,
                preselected=None) -> "list[int] | None":
        self.calls.append((name, index, total))
        if isinstance(self.exclude, dict):
            if name not in self.exclude:
                return []
            picked = self.exclude[name]
            return None if picked is None else list(picked)
        return list(self.exclude)
