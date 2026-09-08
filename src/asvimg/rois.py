"""Editable point ROIs in atlas (standard-brain) pixel space.

An ROI is a labelled circle at ``(x=col, y=row)`` with a per-ROI ``size``
(diameter, px) in the 285x285 atlas image.  Defaults come from the atlas'
built-in point ROIs; the GUI editor lets the user add/remove/move/resize them
and save to ``rois.csv``, which ROI extraction then uses instead of the atlas
defaults.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_ROI_SIZE = 8
ROIS_FILENAME = "rois.csv"
# Source-space ROIs (recording's own binned coords), used when roi_space="source".
# A SEPARATE file so it can never be confused with the atlas-space rois.csv: the
# two carry the same columns but mean coordinates in different spaces.
SOURCE_ROIS_FILENAME = "rois_source.csv"


@dataclass
class Roi:
    name: str
    x: int  # column in the atlas image (0..W-1)
    y: int  # row in the atlas image (0..H-1)
    size: int = DEFAULT_ROI_SIZE  # circle diameter in px


def default_rois(atlas, size: int = DEFAULT_ROI_SIZE) -> list[Roi]:
    """The atlas' built-in point ROIs as editable :class:`Roi` rows."""
    rois: list[Roi] = []
    for (r, c), name in zip(atlas.point_rois, atlas.point_roi_names):
        rois.append(Roi(str(name), int(round(c)), int(round(r)), int(size)))
    return rois


def contralateral(roi: Roi, width: int) -> Roi:
    """Mirror an ROI across the vertical midline (col -> (width-1) - col).

    The symmetry axis of a ``width``-wide image is at ``(width-1)/2``, not
    ``width/2``: mirroring by ``width - col`` lands one pixel too lateral, and
    round-tripping it does not return the original ROI.
    """
    if roi.name.endswith("_R"):
        name = roi.name[:-2] + "_L"
    elif roi.name.endswith("_L"):
        name = roi.name[:-2] + "_R"
    else:
        name = roi.name + "_contra"
    return Roi(name, (int(width) - 1) - int(roi.x), int(roi.y), int(roi.size))


def build_masks(rois: list[Roi], shape_hw: tuple[int, int]):
    """(rois, (H,W)) -> ((N,H,W) bool masks, names) — matches atlas circle convention."""
    h, w = shape_hw
    rr, cc = np.mgrid[:h, :w]
    masks = np.zeros((len(rois), h, w), dtype=bool)
    names: list[str] = []
    for i, roi in enumerate(rois):
        radius = max(roi.size / 2.0, 0.5)
        masks[i] = ((rr - roi.y) ** 2 + (cc - roi.x) ** 2) <= radius**2
        names.append(roi.name)
    return masks, names


def save_rois(path: Path, rois: list[Roi]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "x", "y", "size"])
        for roi in rois:
            writer.writerow([roi.name, int(roi.x), int(roi.y), int(roi.size)])


def load_rois(path: Path) -> list[Roi]:
    path = Path(path)
    rois: list[Roi] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rois.append(
                Roi(
                    str(row["name"]).strip(),
                    int(round(float(row["x"]))),
                    int(round(float(row["y"]))),
                    int(round(float(row["size"]))),
                )
            )
    return rois


def resolve_rois(output_dir: Path, atlas) -> tuple[list[Roi], bool]:
    """Load ``rois.csv`` from ``output_dir`` if present, else the atlas defaults.

    Returns ``(rois, is_custom)``.
    """
    p = Path(output_dir) / ROIS_FILENAME
    if p.exists():
        rois = load_rois(p)
        if rois:
            return rois, True
    return default_rois(atlas), False


def resolve_source_rois(output_dir: Path) -> list[Roi]:
    """ROIs in the recording's own (binned reg/dff) pixel space, from
    ``rois_source.csv``.

    Unlike :func:`resolve_rois` there is no default set: a raw recording has no
    standard-brain ROIs, so an empty list means "nothing defined yet" and the
    caller should skip rather than invent regions.
    """
    p = Path(output_dir) / SOURCE_ROIS_FILENAME
    return load_rois(p) if p.exists() else []
