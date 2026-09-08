"""NWB metadata sidecar (``nwb_metadata.yaml``) for the ``asovi-nwb`` tool.

This holds the experiment metadata NWB / DANDI require but the pipeline never
captured — Subject (species/sex/age/genotype), session start time, experimenter /
institution, per-channel optics (excitation/emission/indicator), and the physical
pixel size.  It is a **sidecar in ``output_dir``**, the same "human-only input"
class as ``marks.mat`` / ``rois.csv`` / ``demux_correction.json`` — deliberately
NOT a :class:`PipelineConfig` field (``ops.yaml`` stays a pipeline-knobs file).

The standalone GUI (``gui/nwb_editor.py``) is the only writer; the export library
(:mod:`nwb_export`) is the only reader.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

from .config import resolve_atlas_path

NWB_META_FILENAME = "nwb_metadata.yaml"

_SEX_CHOICES = ("M", "F", "O", "U")


@dataclass
class SessionMeta:
    session_description: str = ""
    session_id: str = ""
    session_start_time: str = ""  # ISO-8601, tz-aware preferred, e.g. 2026-07-18T10:30:00+09:00
    experimenter: list[str] = field(default_factory=list)  # "Last, First"
    lab: str = ""
    institution: str = ""
    experiment_description: str = ""
    keywords: list[str] = field(default_factory=list)
    related_publications: list[str] = field(default_factory=list)  # DOIs


@dataclass
class SubjectMeta:
    subject_id: str = ""
    species: str = "Mus musculus"  # Latin binomial or NCBITaxon URI
    sex: str = "U"  # M | F | O | U
    age: str = ""  # ISO-8601 duration, e.g. P90D (or set date_of_birth)
    date_of_birth: str = ""  # ISO-8601 datetime (alternative to age)
    genotype: str = ""
    strain: str = ""
    description: str = ""


@dataclass
class DeviceMeta:
    name: str = "Widefield"
    description: str = "one-photon widefield imaging system"
    manufacturer: str = ""
    model: str = ""


@dataclass
class ChannelMeta:
    """Optics for the SOURCE (signal) channel of one channels_name group."""

    excitation_lambda: float | None = None  # nm
    emission_lambda: float | None = None  # nm
    indicator: str = ""  # e.g. GCaMP6s
    location: str = "dorsal cortex"
    exposure_time: float | None = None  # s


@dataclass
class ExportOptions:
    include_dff: bool = True
    include_warped: bool = False
    include_reference_images: bool = True
    include_roi: bool = True
    include_ica: bool = False
    include_correlation: bool = False
    honor_ica_denoise: bool = True  # read dF/F through the ica_denoise seam
    compression: str = "gzip"  # "gzip" | "none"
    compression_opts: int = 4
    shuffle: bool = True
    chunk_frames_source: int = 8  # ~10 MB/chunk for a 540x640 f32 frame
    chunk_frames_atlas: int = 32  # ~10 MB/chunk for a 285x285 f32 frame
    overwrite: bool = False
    output_filename: str = ""  # blank -> "{exp}.nwb"


@dataclass
class NwbMetadata:
    session: SessionMeta = field(default_factory=SessionMeta)
    subject: SubjectMeta = field(default_factory=SubjectMeta)
    device: DeviceMeta = field(default_factory=DeviceMeta)
    channels: dict[str, ChannelMeta] = field(default_factory=dict)  # keyed by group name
    pixel_size_um: float | None = None  # binned pixel pitch -> ImagingPlane grid_spacing
    options: ExportOptions = field(default_factory=ExportOptions)


# --------------------------------------------------------------------------- #
# (de)serialization
# --------------------------------------------------------------------------- #

def to_dict(meta: NwbMetadata) -> dict[str, Any]:
    return asdict(meta)


def from_dict(d: dict[str, Any] | None) -> NwbMetadata:
    """Rebuild NwbMetadata from a (possibly partial) dict, tolerating missing keys."""
    d = dict(d or {})
    return NwbMetadata(
        session=SessionMeta(**_pick(d.get("session"), SessionMeta)),
        subject=SubjectMeta(**_pick(d.get("subject"), SubjectMeta)),
        device=DeviceMeta(**_pick(d.get("device"), DeviceMeta)),
        channels={
            name: ChannelMeta(**_pick(ch, ChannelMeta))
            for name, ch in (d.get("channels") or {}).items()
        },
        pixel_size_um=d.get("pixel_size_um"),
        options=ExportOptions(**_pick(d.get("options"), ExportOptions)),
    )


def _pick(raw: dict[str, Any] | None, cls) -> dict[str, Any]:
    """Keep only keys that are fields of ``cls`` (drops stale/unknown keys)."""
    if not raw:
        return {}
    valid = {f.name for f in _fields(cls)}
    return {k: v for k, v in raw.items() if k in valid}


def _fields(cls):
    import dataclasses

    return dataclasses.fields(cls)


def nwb_meta_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / NWB_META_FILENAME


def save_nwb_meta(output_dir: str | Path, meta: NwbMetadata) -> Path:
    p = nwb_meta_path(output_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(to_dict(meta), f, sort_keys=False, allow_unicode=True)
    return p


def load_nwb_meta(output_dir: str | Path) -> NwbMetadata | None:
    """Load the saved sidecar, or None if it does not exist."""
    p = nwb_meta_path(output_dir)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return from_dict(yaml.safe_load(f) or {})


# --------------------------------------------------------------------------- #
# prefill from the processed outputs
# --------------------------------------------------------------------------- #

def detect_artifacts(config, output_dir: str | Path) -> dict[str, Any]:
    """Read-only summary of what is on disk, to drive the GUI header + writer.

    Never raises: a missing/partial folder just yields ``reg_meta=False`` and
    empty per-group flags.
    """
    from .io import read_reg_meta, dff_name_path, reg_meta_path

    out = Path(output_dir)
    fmt = getattr(config, "output_format", "mat")
    groups = config.channel_groups()

    summary: dict[str, Any] = {
        "reg_meta": reg_meta_path(out).exists(),
        "H": None, "W": None, "fps": int(config.fps),
        "cycle_len": int(config.cycle_len), "binning": int(config.binning),
        "marks": (out / "marks.mat").exists(),
        "rois_csv": (out / "rois.csv").exists(),
        "rois_source": (out / "rois_source.csv").exists(),
        "atlas_path": str(resolve_atlas_path(config.annotation_atlas_path)),
        "atlas_available": resolve_atlas_path(config.annotation_atlas_path).exists(),
        "roi_space": config.roi_space,
        "groups": {},
    }
    try:
        reg = read_reg_meta(out)
        hw = reg.get("imageSize")
        if hw is not None:
            summary["H"], summary["W"] = int(hw[0]), int(hw[1])
        if reg.get("fps") is not None:
            summary["fps"] = int(reg["fps"])
    except Exception:
        pass

    for name, g in groups.items():
        summary["groups"][name] = {
            "has_source": bool(g["source_indices"]),
            "has_donner": bool(g["donner_indices"]),
            "dff": dff_name_path(out, name).exists(),
            "dff_denoised": (out / "ica" / f"dff_{name}.npy").exists(),
            "basis": (out / "ica" / name / "basis.npz").exists(),
            "roi_signals": bool(list(out.glob(f"roiSignals_{name}_*"))),
            "dfWarped": bool(list(out.glob(f"*dfWarped_{name}_*"))),
        }
    return summary


def prefill(config, output_dir: str | Path) -> NwbMetadata:
    """Build metadata seeded from config + reg_meta, then overlay any saved sidecar.

    Fresh fields the pipeline never had stay blank (the user fills them); the
    derivable ones (session_id, per-group channel rows) are pre-populated.
    """
    summary = detect_artifacts(config, output_dir)
    meta = NwbMetadata()

    # session id from the experiment name / input stem
    try:
        from .io import resolve_exp_stem

        stem = resolve_exp_stem(
            config.input_dir, config.exp_name, config.input_format, config.input_order
        )
    except Exception:
        stem = config.exp_name or ""
    meta.session.session_id = stem or ""
    meta.session.session_description = f"Wide-field cortical imaging ({stem})" if stem else ""

    # one channel row per source-bearing group
    for name, g in summary["groups"].items():
        if g["has_source"]:
            meta.channels[name] = ChannelMeta()

    # overlay a previously-saved sidecar (user edits win; new groups keep defaults)
    saved = load_nwb_meta(output_dir)
    if saved is not None:
        prefilled_channels = meta.channels
        meta = saved
        for name, ch in prefilled_channels.items():
            meta.channels.setdefault(name, ch)
    return meta


# --------------------------------------------------------------------------- #
# validation (mirrors what nwbinspector --config dandi checks at CRITICAL)
# --------------------------------------------------------------------------- #

def validate(meta: NwbMetadata) -> list[str]:
    """Return human-readable problems.  Entries prefixed ``[DANDI]`` are the ones
    nwbinspector will raise as CRITICAL (blocking upload); the rest are advice."""
    problems: list[str] = []
    s, sub = meta.session, meta.subject

    if not s.session_description.strip():
        problems.append("[DANDI] session_description is empty")
    if not s.session_start_time.strip():
        problems.append("[DANDI] session_start_time is empty (must be a tz-aware ISO-8601 datetime)")
    else:
        from datetime import datetime

        try:
            datetime.fromisoformat(s.session_start_time)
        except ValueError:
            problems.append(f"[DANDI] session_start_time is not ISO-8601: {s.session_start_time!r}")

    if not sub.subject_id.strip():
        problems.append("[DANDI] subject_id is empty")
    if not sub.species.strip():
        problems.append("[DANDI] species is empty (Latin binomial, e.g. 'Mus musculus')")
    if sub.sex not in _SEX_CHOICES:
        problems.append(f"[DANDI] sex must be one of {_SEX_CHOICES}, got {sub.sex!r}")
    if not sub.age.strip() and not sub.date_of_birth.strip():
        problems.append("[DANDI] set age (ISO-8601, e.g. P90D) or date_of_birth")
    elif sub.age.strip() and not sub.age.strip().upper().startswith("P"):
        problems.append(f"age should be an ISO-8601 duration (e.g. P90D), got {sub.age!r}")

    for name, ch in meta.channels.items():
        if ch.emission_lambda is None:
            problems.append(f"channel {name!r}: emission_lambda (nm) is unset")
        if ch.excitation_lambda is None:
            problems.append(f"channel {name!r}: excitation_lambda (nm) is unset")
    if meta.pixel_size_um is None:
        problems.append("pixel_size_um is unset (ImagingPlane grid_spacing will be omitted)")
    return problems
