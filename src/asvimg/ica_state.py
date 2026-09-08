"""On-disk state for the ICA stage: the decomposition, and the decision.

Two artifacts, deliberately separate — the same split ``marks.mat`` (a decision)
and ``reg_meta.npz`` (a derived product) already use:

``ica/{name}/basis.npz``
    DERIVED. The spatial ICA basis, plus everything needed to know whether it is
    still the basis of the data on disk (a fingerprint of ``dff_{name}.npy`` and
    the parameters that determine the decomposition). Re-derivable at any time by
    re-running the stage; deleted when preprocess deletes its outputs.

``ica_exclusion.json``
    THE DECISION. Which components a human judged to be artifacts. This cannot be
    re-derived, so it is the thing a re-run must replay (like ``marks.mat``) and
    the thing a headless run needs in order to reproduce an interactive session.

IC indices are stored as **labels** (``"IC3"``, 1-based), not bare ints: the maps
the user clicks are written as ``IC{i+1}.png``, so a hand-edited ``3`` would
exclude a different component than the one labelled IC3 on screen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import PipelineConfig
    from .ica import IcaResult

EXCLUSION_FILENAME = "ica_exclusion.json"

# Config fields that determine the decomposition. A basis fitted under different
# values is a different basis, and its IC numbers mean something else.
# NOT pca_smooth_sigma: it only low-passes the plotted traces (the ICA reads
# temporal_raw), so listing it here would make a display knob invalidate the
# basis and force ROI / export / movies to be rebuilt for nothing.
_BASIS_PARAMS = (
    "pca_n_components",
    "pca_skip_frames",
    "ica_n_components",
    "ica_random_state",
    "ica_max_iter",
    "baseline_percentile",
)


def ica_dir(output_dir) -> Path:
    return Path(output_dir) / "ica"


def basis_path(output_dir, name: str) -> Path:
    return ica_dir(output_dir) / name / "basis.npz"


def exclusion_path(output_dir) -> Path:
    return Path(output_dir) / EXCLUSION_FILENAME


def source_fingerprint(output_dir, name: str) -> str:
    """Identity of the dF/F the basis was fitted to: size, shape, and a hash of
    three sampled blocks of the content.

    Content, NOT mtime: an output folder copied to another machine (or restored
    from a backup) keeps its bytes but not its timestamps, and an mtime-keyed
    fingerprint would declare every basis stale — which, with ica_denoise on,
    means "refuse to run" on data that is perfectly fine.  Sampling three 1 MB
    blocks keeps it O(1) on a multi-GB stack while still catching a rewrite.
    """
    import hashlib

    from .io import dff_name_path

    p = dff_name_path(output_dir, name)
    if not p.exists():
        return ""
    size = p.stat().st_size
    shape = tuple(int(x) for x in np.load(p, mmap_mode="r").shape)
    block = 1 << 20
    h = hashlib.sha1(f"{size}:{shape}".encode())
    with p.open("rb") as f:
        for off in (0, max(0, size // 2 - block // 2), max(0, size - block)):
            f.seek(off)
            h.update(f.read(block))
    return f"{size}:{shape}:{h.hexdigest()[:16]}"


def save_ica_basis(output_dir, name: str, ica_result: "IcaResult",
                   config: "PipelineConfig") -> Path:
    """Write ``ica/{name}/basis.npz``. Written on every ICA run, even when nothing
    is excluded — "ICA ran and flagged nothing" must be distinguishable from
    "ICA never ran"."""
    path = basis_path(output_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = {k: getattr(config, k) for k in _BASIS_PARAMS}
    np.savez(
        path,
        spatial=np.asarray(ica_result.spatial, dtype=np.float32),
        mean_image=np.asarray(ica_result.mean_image, dtype=np.float64),
        temporal=np.asarray(ica_result.temporal, dtype=np.float32),
        frame_indices=np.asarray(ica_result.frame_indices, dtype=np.int64),
        temporal_variance=np.asarray(ica_result.temporal_variance, dtype=np.float64),
        n_components=np.int64(ica_result.n_components),
        params=json.dumps({k: (None if v is None else v) for k, v in params.items()}),
        source_fingerprint=source_fingerprint(output_dir, name),
    )
    return path


def load_ica_basis(output_dir, name: str) -> dict | None:
    """The saved basis, or None when it is absent."""
    path = basis_path(output_dir, name)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    out["params"] = json.loads(str(out["params"]))
    out["source_fingerprint"] = str(out["source_fingerprint"])
    return out


def basis_id(output_dir, name: str) -> str:
    """Identity of the saved basis itself — so a denoised stack (and a decision)
    can say WHICH basis it was made against.  Re-fitting rewrites basis.npz, and
    an IC index only means something relative to one particular fit."""
    import hashlib

    p = basis_path(output_dir, name)
    if not p.exists():
        return ""
    return hashlib.sha1(p.read_bytes()).hexdigest()[:16]


def basis_is_current(output_dir, name: str, config: "PipelineConfig") -> tuple[bool, str]:
    """(ok, reason) — is the saved basis still the basis of the dF/F on disk?

    A stale basis is worse than a missing one: its IC numbers still resolve, so a
    saved exclusion would silently remove the wrong components.
    """
    basis = load_ica_basis(output_dir, name)
    if basis is None:
        return False, f"no ica/{name}/basis.npz"
    want = {k: getattr(config, k) for k in _BASIS_PARAMS}
    got = basis["params"]
    changed = [k for k in _BASIS_PARAMS if got.get(k) != want[k]]
    if changed:
        return False, f"basis was fitted with different {', '.join(changed)}"
    fp = source_fingerprint(output_dir, name)
    if not fp:
        return False, f"no dff_{name}.npy to check the basis against"
    if fp != basis["source_fingerprint"]:
        return False, f"dff_{name}.npy changed since the basis was fitted"
    return True, "current"


# --------------------------------------------------------------------------
# The decision: which components are artifacts
# --------------------------------------------------------------------------


def _to_labels(indices) -> list[str]:
    return [f"IC{int(i) + 1}" for i in sorted({int(i) for i in indices})]


def _from_labels(labels) -> list[int]:
    out: list[int] = []
    for lab in labels:
        s = str(lab).strip()
        if s.upper().startswith("IC"):
            s = s[2:]
        out.append(int(s) - 1)
    return sorted({i for i in out if i >= 0})


def save_exclusions(output_dir, excluded: dict[str, list[int]]) -> Path:
    """Write ``ica_exclusion.json``.  Groups absent from ``excluded`` are dropped;
    a group mapped to ``[]`` is recorded as "looked at, flagged nothing"."""
    path = exclusion_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "groups": {
            name: {
                "excluded": _to_labels(ics),
                "basis": f"ica/{name}/basis.npz",
            }
            for name, ics in excluded.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_exclusions(output_dir) -> dict[str, list[int]]:
    """``{group: [0-based IC indices]}`` from the sidecar; ``{}`` when absent."""
    path = exclusion_path(output_dir)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        # NOT {}: silently reading a corrupt decision as "exclude nothing" would
        # hand back un-denoised numbers while every payload says they are denoised.
        raise ValueError(
            f"{path} is unreadable ({exc}). It records which components a human "
            f"judged to be artifacts and cannot be re-derived — restore it, or "
            f"delete it to start the review again."
        ) from None
    groups = payload.get("groups") or {}
    out: dict[str, list[int]] = {}
    for name, entry in groups.items():
        try:
            out[str(name)] = _from_labels(entry.get("excluded") or [])
        except (TypeError, ValueError):
            continue
    return out


def resolve_exclusion(config: "PipelineConfig", output_dir, name: str) -> list[int]:
    """The ICs to exclude for one group, from ``config.ica_exclusion``.

    ``"gui"`` resolves like ``"cache"`` here: by the time anything applies an
    exclusion the picker has already run and recorded its answer.
    """
    mode = config.ica_exclusion
    if isinstance(mode, dict):
        return sorted({int(i) for i in mode.get(name, [])})
    if mode in (False, None):
        return []
    return list(load_exclusions(output_dir).get(name, []))


# --------------------------------------------------------------------------
# The applied product: dF/F with the excluded components subtracted
# --------------------------------------------------------------------------


def denoised_dff_path(output_dir, name: str) -> Path:
    """``ica/dff_{name}.npy`` — a SUBDIRECTORY, never ``dff_{name}_ica.npy``.

    channels_name permits underscores, so a group literally named ``X_ica`` would
    otherwise write the same file as the denoised product of group ``X``.
    ``dff_{name}.npy`` itself is never overwritten: it is the input to the
    projection (denoising the denoised is not a thing) and the only copy of the
    linear-subtraction result.
    """
    return ica_dir(output_dir) / f"dff_{name}.npy"


def _denoised_meta_path(output_dir, name: str) -> Path:
    return ica_dir(output_dir) / f"dff_{name}.json"


def denoised_is_current(output_dir, name: str, config: "PipelineConfig",
                        excluded: list[int]) -> tuple[bool, str]:
    """Is the denoised stack on disk the one this config + decision asks for?"""
    npy = denoised_dff_path(output_dir, name)
    meta = _denoised_meta_path(output_dir, name)
    if not npy.exists() or not meta.exists():
        return False, "not built"
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return False, "unreadable metadata"
    if payload.get("excluded") != _to_labels(excluded):
        return False, "the exclusion changed"
    if payload.get("source_fingerprint") != source_fingerprint(output_dir, name):
        return False, f"dff_{name}.npy changed"
    if payload.get("params") != {k: getattr(config, k) for k in _BASIS_PARAMS}:
        return False, "the decomposition parameters changed"
    if payload.get("basis_id") != basis_id(output_dir, name):
        return False, "it was built from a different ICA basis"
    return True, "current"


def ensure_denoised_dff(config: "PipelineConfig", output_dir, name: str,
                        *, log=None, progress=None) -> Path:
    """Path to ``ica/dff_{name}.npy``, building it if it is missing or stale.

    Raises rather than falling back to the plain dF/F: a silent fallback would
    hand back un-denoised numbers under a config that says they are denoised.
    """
    from .ica import apply_ica_denoise
    from .io import dff_name_path

    output_dir = Path(output_dir)
    excluded = resolve_exclusion(config, output_dir, name)

    ok, why = denoised_is_current(output_dir, name, config, excluded)
    if ok:
        return denoised_dff_path(output_dir, name)

    basis = load_ica_basis(output_dir, name)
    if basis is None:
        raise FileNotFoundError(
            f"ica_denoise={config.ica_denoise!r} but no ica/{name}/basis.npz — "
            f"run the ICA stage first."
        )
    fresh, basis_why = basis_is_current(output_dir, name, config)
    if not fresh:
        raise RuntimeError(
            f"ica/{name}/basis.npz is stale ({basis_why}); its IC numbers no "
            f"longer mean what {EXCLUSION_FILENAME} says. Re-run the ICA stage."
        )

    src = dff_name_path(output_dir, name)
    if not src.exists():
        raise FileNotFoundError(f"no dff_{name}.npy to denoise under {output_dir}")

    # Drop the blessing BEFORE touching the stack: an interrupted rebuild must
    # leave a stack that is obviously not current, not a half-written one that an
    # old metadata file still vouches for.
    _denoised_meta_path(output_dir, name).unlink(missing_ok=True)

    mm = np.load(src, mmap_mode="r")  # (T, H, W)
    n_frames, h, w = (int(x) for x in mm.shape)
    if log is not None:
        log(f"[ica] {name}: applying exclusion {_to_labels(excluded)} ({why}) "
            f"-> ica/dff_{name}.npy")
    try:
        apply_ica_denoise(
            np.asarray(basis["spatial"], dtype=np.float64),
            lambda t0, t1: mm[t0:t1],
            n_frames, (h, w),
            denoised_dff_path(output_dir, name),
            exclude=excluded,
            progress=progress,
        )
    finally:
        handle = getattr(mm, "_mmap", None)
        if handle is not None:
            handle.close()
        del mm

    _denoised_meta_path(output_dir, name).write_text(
        json.dumps(
            {
                "excluded": _to_labels(excluded),
                "source_fingerprint": source_fingerprint(output_dir, name),
                "basis_id": basis_id(output_dir, name),
                "params": {k: getattr(config, k) for k in _BASIS_PARAMS},
                "mode": config.ica_denoise,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return denoised_dff_path(output_dir, name)
