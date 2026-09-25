"""Where the test suite finds an atlas.

Nothing ships one any more: the atlas is derived from the Allen CCF, whose terms
are not this project's MIT terms, so it is built per machine by
``asovi-atlas --download`` rather than redistributed inside the wheel. That
leaves the tests without a guaranteed atlas file, so they ask here and skip when
the answer is None.

Resolution order:

1. ``matlab/wfciAnnotationData.mat`` — the MATLAB-era file, present only in the
   lab's internal checkout. Byte-identical to what used to ship, so tests that
   were written against that geometry keep exercising exactly it.
2. :data:`asvimg.config.USER_ATLAS` — an atlas the developer built. This is what
   a public checkout has once someone has run the setup command.

A generated atlas is not interchangeable with the MATLAB one for every purpose:
it carries correct ``region_names`` where the old file's were scrambled, and its
IDs are split L/R. Tests that depend on the *old* labelling must say so rather
than accept whichever file turned up — see ``test_bundled_atlas.py``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from asvimg.config import USER_ATLAS

REPO_ROOT = Path(__file__).resolve().parents[1]
MATLAB_ATLAS = REPO_ROOT / "matlab" / "wfciAnnotationData.mat"


def _resolve() -> Path | None:
    for p in (MATLAB_ATLAS, USER_ATLAS):
        if p.exists():
            return p
    return None


#: An atlas file to test against, or None when neither is on this machine.
TEST_ATLAS: Path | None = _resolve()

#: A never-None Path, for module-level ``skipUnless(... .exists())`` guards that
#: want to name the path they looked for in the skip message.
TEST_ATLAS_OR_DEFAULT: Path = TEST_ATLAS or USER_ATLAS

SKIP_REASON = (
    f"no atlas on this machine (looked for {MATLAB_ATLAS} and {USER_ATLAS}); "
    f"build one with `asovi-atlas --download`"
)


@pytest.fixture(scope="session")
def atlas_path() -> Path:
    """The atlas file, skipping the test when there is none."""
    if TEST_ATLAS is None:
        pytest.skip(SKIP_REASON)
    return TEST_ATLAS
