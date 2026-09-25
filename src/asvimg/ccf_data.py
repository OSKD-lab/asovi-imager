"""Locate — and, on first run, optionally download — the Allen CCF volumes the
atlas builder needs.

The volumes are large (~4.8 GB total), re-downloadable reference data shared by
every recording — a **per-user cache**, not a per-recording setting, so this lives
outside ``PipelineConfig``.

CCF-directory resolution order (first that has the files wins):
  1. an explicit path (CLI ``--atlas-dir`` / function arg)
  2. env ``ASOVI_ATLAS_DIR``
  3. ``resources/atlas_from_figshare/`` (repo-relative — backward compat)
  4. ``~/.asovi/atlas/``  (the default; the first-run download target)

Nothing is downloaded without consent: the CLI asks ``[y/N]`` on an interactive
TTY (or use ``--download``); a non-interactive run raises with manual steps; the
GUI has a Download button.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

ENV_VAR = "ASOVI_ATLAS_DIR"
DEFAULT_DIR = Path.home() / ".asovi" / "atlas"
_REPO_DIR = Path("resources/atlas_from_figshare")
_ALLENCCF_DIR = Path("resources/allenCCF")

FIGSHARE_ARTICLE = 25365829          # "Modified Allen CCF 2017 for cortex-lab/allenCCF" (CC BY 4.0)
ANNOTATION_NAME = "annotation_volume_10um_by_index.npy"
TEMPLATE_NAME = "template_volume_10um.npy"
STRUCTURE_TREE_NAME = "structure_tree_safe_2017.csv"
APPROX_DOWNLOAD_GB = 4.8

# The structure tree is NOT in the figshare article -- only the two volumes are --
# so it is fetched from the project those volumes were prepared for.
#
# It cannot be swapped for a fresh Allen API query, tempting as that is: the
# annotation volume is stored *by index*, and the index is this file's row order.
# A differently-ordered table of the same structures would silently relabel every
# area in the atlas.
STRUCTURE_TREE_URL = (
    "https://raw.githubusercontent.com/cortex-lab/allenCCF/master/structure_tree_safe_2017.csv"
)


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #

def candidate_dirs(explicit: str | Path | None = None) -> list[Path]:
    dirs: list[Path] = []
    if explicit:
        dirs.append(Path(explicit))
    env = os.environ.get(ENV_VAR)
    if env:
        dirs.append(Path(env))
    dirs.append(_REPO_DIR)
    dirs.append(DEFAULT_DIR)
    return dirs


def _has_volumes(d: Path) -> tuple[Path, Path] | None:
    a, t = d / ANNOTATION_NAME, d / TEMPLATE_NAME
    return (a, t) if a.exists() and t.exists() else None


def find_ccf_volumes(explicit: str | Path | None = None) -> tuple[Path, Path] | None:
    """(annotation, template) from the first candidate dir that has both, else None."""
    for d in candidate_dirs(explicit):
        got = _has_volumes(d)
        if got:
            return got
    return None


def find_structure_tree(explicit: str | Path | None = None) -> Path | None:
    """Locate structure_tree_safe_2017.csv (CCF dirs, then resources/allenCCF)."""
    seen: list[Path] = []
    for d in candidate_dirs(explicit):
        seen.append(d / STRUCTURE_TREE_NAME)
    seen.append(_ALLENCCF_DIR / STRUCTURE_TREE_NAME)
    for p in seen:
        if p.exists():
            return p
    return None


def ensure_structure_tree(explicit: str | Path | None = None, *, download: bool = False,
                          interactive: bool | None = None, reporter=None) -> Path:
    """Return the structure tree, fetching it on first run if permitted.

    Same consent rule as :func:`ensure_ccf_volumes`: ``download=True`` fetches
    without asking, an interactive TTY is prompted, and anything else raises with
    manual steps.  It is ~220 kB, so the prompt mentions the source rather than
    the size."""
    got = find_structure_tree(explicit)
    if got:
        return got
    target = resolve_download_dir(explicit)
    if interactive is None:
        interactive = bool(getattr(sys.stdin, "isatty", lambda: False)()
                           and getattr(sys.stdout, "isatty", lambda: False)())

    do = download
    if not do and interactive:
        ans = input(f"{STRUCTURE_TREE_NAME} not found. Fetch it from cortex-lab/allenCCF\n"
                    f"into  {target} ?  [y/N] ")
        do = ans.strip().lower() in ("y", "yes")
    if not do:
        looked = "\n  ".join(str(d / STRUCTURE_TREE_NAME) for d in candidate_dirs(explicit))
        raise FileNotFoundError(
            f"{STRUCTURE_TREE_NAME} not found. Looked in:\n  " + looked +
            "\nFetch it with:  asovi-atlas --download   (or download it yourself from\n"
            f"{STRUCTURE_TREE_URL} into {target}).")

    target.mkdir(parents=True, exist_ok=True)
    dest = target / STRUCTURE_TREE_NAME
    _download_file(STRUCTURE_TREE_URL, dest, reporter=reporter, label=STRUCTURE_TREE_NAME)
    return dest


def resolve_download_dir(explicit: str | Path | None = None) -> Path:
    """Where to download into: the explicit / env target, else the default.
    (Never the repo ``resources`` dir — that is read-only backward-compat.)"""
    if explicit:
        return Path(explicit)
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env)
    return DEFAULT_DIR


# --------------------------------------------------------------------------- #
# ensure (+ ask / download)
# --------------------------------------------------------------------------- #

def ensure_ccf_volumes(explicit: str | Path | None = None, *, download: bool = False,
                       interactive: bool | None = None, reporter=None) -> tuple[Path, Path]:
    """Return (annotation, template), downloading on first run if permitted.

    ``download=True`` fetches without asking.  Otherwise, on an interactive TTY the
    user is prompted ``[y/N]``; a non-interactive run (or a declined prompt) raises
    ``FileNotFoundError`` with manual instructions.  ``reporter`` is an optional
    ``callable(label, done, total)`` progress hook."""
    got = find_ccf_volumes(explicit)
    if got:
        return got
    target = resolve_download_dir(explicit)
    if interactive is None:
        interactive = bool(getattr(sys.stdin, "isatty", lambda: False)()
                           and getattr(sys.stdout, "isatty", lambda: False)())

    do = download
    if not do and interactive:
        ans = input(
            f"Allen CCF volumes not found. Download ~{APPROX_DOWNLOAD_GB:g} GB from figshare "
            f"{FIGSHARE_ARTICLE}\n(Allen CCF, CC-BY) into  {target} ?  [y/N] ")
        do = ans.strip().lower() in ("y", "yes")
    if not do:
        looked = "\n  ".join(str(d) for d in candidate_dirs(explicit))
        raise FileNotFoundError(
            "Allen CCF volumes not found. Looked in:\n  " + looked +
            f"\nDownload them with:  asovi-atlas --download   (or set {ENV_VAR}, or "
            f"pass --atlas-dir),\nor fetch {ANNOTATION_NAME} + {TEMPLATE_NAME} manually "
            f"from https://doi.org/10.6084/m9.figshare.{FIGSHARE_ARTICLE} into {target}.")

    download_ccf_volumes(target, reporter=reporter)
    got = _has_volumes(target)
    if not got:
        raise RuntimeError("download finished but the volumes are still missing")
    return got


# --------------------------------------------------------------------------- #
# download (figshare)
# --------------------------------------------------------------------------- #

def _figshare_manifest() -> dict[str, dict]:
    url = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE}/files"
    with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310 — fixed https host
        return {f["name"]: f for f in json.load(r)}


def download_ccf_volumes(target_dir: str | Path, *, reporter=None,
                         names: tuple[str, ...] = (ANNOTATION_NAME, TEMPLATE_NAME)) -> Path:
    """Download the named files from the figshare article into ``target_dir``
    (md5-verified, atomic via a ``.part`` temp file; skips ones already present)."""
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest = _figshare_manifest()
    for name in names:
        dest = target / name
        if dest.exists():
            continue
        if name not in manifest:
            raise FileNotFoundError(f"{name!r} is not in figshare article {FIGSHARE_ARTICLE}")
        f = manifest[name]
        _download_file(f["download_url"], dest, expected_md5=f.get("computed_md5"),
                       size=f.get("size"), reporter=reporter, label=name)
    return target


def _download_file(url: str, dest: Path, *, expected_md5: str | None = None,
                   size: int | None = None, reporter=None, label: str = "") -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    h = hashlib.md5()
    done = 0
    with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as fh:  # noqa: S310
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            h.update(chunk)
            done += len(chunk)
            if reporter is not None and size:
                try:
                    reporter(label, done, int(size))
                except Exception:  # noqa: BLE001 — progress must never abort a download
                    pass
    if expected_md5 and h.hexdigest() != expected_md5:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"md5 mismatch for {label}: got {h.hexdigest()}, expected {expected_md5}")
    part.replace(dest)
