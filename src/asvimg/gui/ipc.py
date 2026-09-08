"""Parent ↔ child serialization for the interactive subprocess GUIs.

The dashboard owns a single, long-lived dearpygui context, but cpselect and
the ICA selection dialog each create their *own* dearpygui context.  Since
dearpygui allows only one context per process, those windows are launched in
short-lived child processes; this module is the (small) data contract between
parent and child — pickled request/result files in a per-run temp directory.
"""

from __future__ import annotations

import os
import pickle
import sys
import tempfile
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Directory containing ``pyproject.toml``."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: three levels up (src/asvimg/gui -> repo root).
    return here.parents[3]


@lru_cache(maxsize=1)
def package_parent() -> Path:
    """Directory that *contains* the ``asvimg`` package (``src/`` in a checkout).

    An installed copy needs nothing on PYTHONPATH; a checkout that was never
    installed does, and since the package moved under ``src/`` the repo root is
    no longer the right entry.
    """
    return Path(__file__).resolve().parents[2]


def child_env() -> dict[str, str]:
    """Environment for a child process so ``asvimg`` imports resolve."""
    env = dict(os.environ)
    root = str(package_parent())
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root + (os.pathsep + existing if existing else "")
    return env


def child_command(module: str, *args: str) -> list[str]:
    """Build the ``python -m <module> <args...>`` command for a child GUI."""
    return [sys.executable, "-m", module, *args]


def new_session_dir() -> Path:
    """Allocate a unique temp directory for one parent↔child exchange."""
    return Path(tempfile.mkdtemp(prefix="asovi_gui_ipc_"))


def dump(obj: object, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(path: Path) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)
