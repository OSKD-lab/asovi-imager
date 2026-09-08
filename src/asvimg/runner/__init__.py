"""Headless orchestration layer for the ASoVi pipeline.

``PipelineSession`` replays the steps of ``run_pipeline_full.ipynb`` as an
explicit, scriptable API built purely on the already-factored ``asvimg.*``
functions plus ``PreprocessRunner``.  It has **no** GUI dependency: the two
interactive steps (control-point selection, ICA component exclusion) are
abstracted behind the ``MarksProvider`` / ``IcaExclusionProvider`` protocols,
whose headless defaults resolve them non-interactively from the config.  A GUI
front-end supplies interactive providers instead.
"""

from __future__ import annotations

from .providers import (
    ConfigMarksProvider,
    FakeIcaProvider,
    FakeMarksProvider,
    IcaExclusionProvider,
    MarksProvider,
    NoExclusionIcaProvider,
)
from .session import (
    PipelineSession,
    SessionState,
    completed_stages,
    stage_artifacts,
)

__all__ = [
    "PipelineSession",
    "SessionState",
    "completed_stages",
    "stage_artifacts",
    "MarksProvider",
    "IcaExclusionProvider",
    "ConfigMarksProvider",
    "NoExclusionIcaProvider",
    "FakeMarksProvider",
    "FakeIcaProvider",
]
