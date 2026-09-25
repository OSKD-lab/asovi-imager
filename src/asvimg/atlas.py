"""Allen Brain Atlas (CCFv3 top-view) — ``ACCFv3`` class.

Usage::

    from asvimg.atlas import ACCFv3
    from asvimg.config import resolve_atlas_path

    atlas = ACCFv3.from_mat(resolve_atlas_path(""))   # the per-user atlas
    atlas.draw_boundaries(ax)

``get_mask`` needs an atlas built by ``asovi-atlas`` -- which is now the default.
The MATLAB-era file that used to ship carries no
trustworthy region names and raises rather than answer from the wrong table.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

# Allen CCFv3 top-view region ID → abbreviation.
# ID 1 = background (non-brain).  IDs 2-35 = cortical areas.
# Left hemisphere regions have no suffix; right hemisphere has "_R".
_REGION_NAMES: dict[int, str] = {
    1: "background",
    2: "SSp-bfd",
    3: "SSp-bfd_R",
    4: "SSp-ll",
    5: "SSp-ll_R",
    6: "VISp",
    7: "VISp_R",
    8: "RSP",
    9: "RSP_R",
    10: "VISal",
    11: "VISal_R",
    12: "VISam",
    13: "VISam_R",
    14: "MOs",
    15: "MOp",
    16: "SSp-ul",
    17: "SSp-ul_R",
    18: "SSp-tr",
    19: "SSp-tr_R",
    20: "SSs",
    21: "SSs_R",
    22: "AUDp",
    23: "AUDp_R",
    24: "VISpm",
    25: "VISpm_R",
    26: "VISrl",
    27: "VISrl_R",
    28: "PTLp",
    29: "PTLp_R",
    30: "TEa",
    31: "TEa_R",
    32: "ACA",
    33: "ACA_R",
    34: "FrA",
    35: "FrA_R",
}

# Inverse: name → ID
_NAME_TO_ID: dict[str, int] = {v: k for k, v in _REGION_NAMES.items()}

# Default point ROIs — right hemisphere coordinates as (row, col) in atlas space.
# Left hemisphere mirrors are generated automatically (col → atlas_width - col).
DEFAULT_POINT_ROI_NAMES: list[str] = [
    "VISp_R",
    "VISpm_R",
    "VISam_R",
    "VISrl_R",
    "VISa_R",
    "RSP-p_R",
    "RSP-a_R",
    "SSp-bfd_R",
    "SSp-ll_R",
    "SSp-ul_R",
    "MOp-m_R",
    "MOs-p_R",
    "MOs-m_R",
    "MOs-am_R",
    "MOs-al_R",
]
DEFAULT_POINT_ROIS: np.ndarray = np.array(
    [
        # (row, col) — converted from the notebook's (col, row) plot coords
        [226, 203],
        [215, 183],
        [192, 185],
        [195, 226],
        [180, 195],
        [215, 160],
        [172, 155],
        [160, 228],
        [140, 190],
        [128, 209],
        [123, 180],
        [125, 150],
        [105, 153],
        [65, 165],
        [75, 190],
    ],
    dtype=np.float64,
)


class ACCFv3:
    """Allen Common Coordinate Framework v3 — top-view cortical atlas.

    Holds the RGB reference image, per-pixel region ID map, region boundary
    contours, and landmark points.  Provides convenience methods for mask
    retrieval, boundary drawing, and region lookup.

    Use the ``from_mat`` classmethod to load from the standard
    ``wfciAnnotationData.mat`` file.
    """

    def __init__(
        self,
        image_rgb: np.ndarray,
        region_ids: np.ndarray,
        boundaries: list[list[np.ndarray]],
        fixed_points: np.ndarray,
        *,
        scale: float = 0.25,
        region_names: dict[int, str] | None = None,
        point_rois: np.ndarray | None = None,
        point_roi_names: list[str] | None = None,
    ):
        self.image_rgb = image_rgb
        self.region_ids = region_ids
        self.boundaries = boundaries
        self._fixed_points = fixed_points
        self._scale = scale
        # Only an atlas file that carries its own names has names we trust.  The
        # legacy MATLAB atlas does not, and the _REGION_NAMES fallback is known
        # wrong (see _REGION_NAMES), so anything name-driven refuses to run on it.
        self._region_names_verified = region_names is not None
        self._region_names = region_names if region_names is not None else _REGION_NAMES
        self._name_to_id = {v: k for k, v in self._region_names.items()}

        # Point ROIs: build both-hemisphere set from defaults or arguments
        if point_rois is not None:
            self.point_rois = np.asarray(point_rois, dtype=np.float64)
            self.point_roi_names = (
                list(point_roi_names)
                if point_roi_names
                else [f"ROI_{i}" for i in range(len(self.point_rois))]
            )
        else:
            self.point_rois, self.point_roi_names = self._build_default_point_rois()

    def _build_default_point_rois(self) -> tuple[np.ndarray, list[str]]:
        """Build both-hemisphere point ROIs from right-hemisphere defaults."""
        w = self.shape_hw[1]
        pts_r = DEFAULT_POINT_ROIS.copy()
        names_r = list(DEFAULT_POINT_ROI_NAMES)

        # Mirror: col → (w-1) - col, name _R → _L. The symmetry axis of a w-wide
        # image sits at (w-1)/2 (= 142.0 for the 285-px atlas), not w/2: mirroring
        # by `w - col` put every _L ROI one pixel too lateral.
        pts_l = pts_r.copy()
        pts_l[:, 1] = (w - 1) - pts_l[:, 1]
        names_l = [n.replace("_R", "_L") for n in names_r]

        pts = np.vstack([pts_r, pts_l])
        names = names_r + names_l
        return pts, names

    # ---- properties ----

    @property
    def shape_hw(self) -> tuple[int, int]:
        """Atlas image size (height, width)."""
        return (self.image_rgb.shape[0], self.image_rgb.shape[1])

    @property
    def scale(self) -> float:
        """Full-resolution → display coordinate scale factor (default 0.25)."""
        return self._scale

    @property
    def brain_mask(self) -> np.ndarray:
        """(H, W) bool — True for brain pixels (region_id != 1)."""
        return self.region_ids != 1

    @property
    def n_regions(self) -> int:
        """Number of unique region IDs (including background)."""
        return len(np.unique(self.region_ids))

    @property
    def region_names_verified(self) -> bool:
        """Whether the atlas file carried its own ID → acronym map."""
        return self._region_names_verified

    @property
    def region_names(self) -> dict[int, str]:
        """Region ID → Allen CCF abbreviation mapping.

        Raises on the legacy MATLAB atlas, whose names are wrong -- see
        :meth:`get_mask` for why and what to do instead.
        """
        self._require_verified_names("region_names")
        return dict(self._region_names)

    def _require_verified_names(self, what: str) -> None:
        if self._region_names_verified:
            return
        raise ValueError(
            f"{what} is not available for this atlas. The legacy MATLAB atlas "
            "(wfciAnnotationData.mat) carries no region names, and the built-in "
            "fallback table is known to be wrong: its ID -> acronym mapping is "
            "scrambled, and its _R/_L suffixes are fictional because the ID map is "
            "bilateral (one ID spans both hemispheres). Geometry, boundaries, the "
            "midline and the point ROIs are correct and unaffected. Build an atlas "
            "with `asovi-atlas` and point annotation_atlas_path at it to get "
            "correct names and a per-hemisphere ID map."
        )

    @property
    def fixed_points(self) -> np.ndarray:
        """(N, 2) landmark points as (row, col)."""
        return self._fixed_points

    # ---- methods ----

    def get_mask(self, region: int | str) -> np.ndarray:
        """Get a boolean mask for a region by ID or name.

        Parameters
        ----------
        region : int or str
            Region ID (e.g. 6) or Allen abbreviation (e.g. ``"VISp"``).

        Returns
        -------
        (H, W) bool array.
        """
        self._require_verified_names("get_mask")
        if isinstance(region, str):
            rid = self._name_to_id.get(region)
            if rid is None:
                raise KeyError(
                    f"Unknown region name: {region!r}. "
                    f"Available: {sorted(self._name_to_id.keys())}"
                )
        else:
            rid = int(region)
        return self.region_ids == rid

    def get_point_roi_mask(
        self,
        roi: int | str | None = None,
        diameter: int = 5,
    ) -> np.ndarray:
        """Get a circular mask for point ROI(s).

        Parameters
        ----------
        roi : int, str, or None
            - int: index into ``point_rois``
            - str: ROI name (e.g. ``"V1_R"``)
            - None: return masks for ALL point ROIs as (N, H, W) bool
        diameter : diameter of the circular ROI in pixels.

        Returns
        -------
        (H, W) bool if a single ROI, or (N, H, W) bool if ``roi=None``.
        """
        h, w = self.shape_hw
        radius = diameter / 2.0
        rr, cc = np.mgrid[:h, :w]

        if roi is None:
            masks = np.zeros((len(self.point_rois), h, w), dtype=bool)
            for i, (pr, pc) in enumerate(self.point_rois):
                masks[i] = ((rr - pr) ** 2 + (cc - pc) ** 2) <= radius**2
            return masks

        if isinstance(roi, str):
            try:
                idx = self.point_roi_names.index(roi)
            except ValueError:
                raise KeyError(
                    f"Unknown point ROI name: {roi!r}. "
                    f"Available: {self.point_roi_names}"
                )
        else:
            idx = int(roi)

        pr, pc = self.point_rois[idx]
        return ((rr - pr) ** 2 + (cc - pc) ** 2) <= radius**2

    def draw_boundaries(
        self,
        ax,
        *,
        color: str | tuple = "magenta",
        linewidth: float = 0.8,
        skip_outline: bool = True,
    ) -> None:
        """Draw region boundaries on a matplotlib Axes.

        Parameters
        ----------
        ax : matplotlib Axes
        color : line color
        linewidth : line width
        skip_outline : if True, skip index 0 (whole-brain outline)
        """
        start = 1 if skip_outline else 0
        for i in range(start, len(self.boundaries)):
            for contour in self.boundaries[i]:
                ax.plot(
                    contour[:, 1] * self._scale,
                    contour[:, 0] * self._scale,
                    "-",
                    color=color,
                    linewidth=linewidth,
                )

    def draw_boundaries_cv2(
        self,
        rgb: np.ndarray,
        *,
        color: tuple[int, int, int] = (80, 80, 80),
        thickness: int = 1,
        skip_outline: bool = True,
    ) -> None:
        """Draw region boundaries on an RGB uint8 image (in-place).

        Parameters
        ----------
        rgb : (H, W, 3) uint8 array — modified in-place.
        color : BGR-order color tuple.
        thickness : line thickness.
        skip_outline : if True, skip index 0 (whole-brain outline).
        """
        import cv2

        start = 1 if skip_outline else 0
        for i in range(start, len(self.boundaries)):
            for contour in self.boundaries[i]:
                pts = (contour * self._scale)[:, ::-1].astype(np.int32)
                cv2.polylines(
                    rgb, [pts], isClosed=False, color=color, thickness=thickness
                )

    # ---- classmethods / loaders ----

    @classmethod
    def from_mat(cls, path: str | Path) -> ACCFv3:
        """Load an atlas file.

        Two layouts are read: the shipped MATLAB ``wfciAnnotationData.mat``
        (HDF5 v7.3), and the HDF5 written by :mod:`asvimg.atlas_build`
        (``asovi-atlas``), which additionally carries ``region_names`` /
        ``scale`` / ``point_rois`` and derives region boundaries on load.

        Parameters
        ----------
        path : path to the atlas file.
        """
        path = Path(path)
        if not path.exists():
            # The atlas is built per machine, not shipped, so "missing" is the
            # ordinary first-run state rather than a broken install.  Say how to
            # fix it here: this is the one place every consumer passes through.
            from .config import ATLAS_SETUP_HINT, USER_ATLAS

            where = "the default location" if path == USER_ATLAS else "annotation_atlas_path"
            raise FileNotFoundError(
                f"No atlas at {path}  ({where}).\n{ATLAS_SETUP_HINT}"
            )
        with h5py.File(path, "r") as f:
            if "asovi_atlas_version" in f:
                return cls._from_generated_hdf5(f)

            img = np.array(f["allenImg_resized"])  # (3, H, W)
            image_rgb = np.transpose(img, (2, 1, 0))  # → (H, W, 3)

            region_ids = np.array(f["allenImgID_resized"]).T

            fp_raw = np.array(f["fixedPos"])  # (3, 2) as (col, row)
            fixed_points = fp_raw[:, ::-1].copy()  # → (row, col)

            boundaries = _parse_boundaries(f, f["B"])

        return cls(
            image_rgb=image_rgb,
            region_ids=region_ids,
            boundaries=boundaries,
            fixed_points=fixed_points,
        )

    @classmethod
    def _from_generated_hdf5(cls, f: h5py.File) -> ACCFv3:
        """Read the atlas_build layout: arrays are stored in the ACCFv3 frame
        directly (region_ids/image_rgb/fixedPos as (row,col)); boundaries are
        derived from ``region_ids`` so they never need serialising."""
        def _strs(key):
            return [s.decode() if isinstance(s, bytes) else str(s)
                    for s in np.array(f[key]).ravel().tolist()]

        region_ids = np.array(f["allenImgID_resized"]).astype(np.uint16)
        image_rgb = np.array(f["allenImg_resized"])            # (H, W, 3)
        fixed_points = np.array(f["fixedPos"]).astype(np.float64)  # (3, 2) (row, col)
        scale = float(np.array(f["scale"])) if "scale" in f else 1.0
        region_names = None
        if "region_ids" in f and "region_acronyms" in f:
            ids = [int(x) for x in np.array(f["region_ids"]).ravel().tolist()]
            region_names = dict(zip(ids, _strs("region_acronyms")))
        point_rois = np.array(f["point_rois"]).astype(np.float64) if "point_rois" in f else None
        point_roi_names = _strs("point_roi_names") if "point_roi_names" in f else None

        return cls(
            image_rgb=image_rgb,
            region_ids=region_ids,
            boundaries=boundaries_from_id_map(region_ids),
            fixed_points=fixed_points,
            scale=scale,
            region_names=region_names,
            point_rois=point_rois,
            point_roi_names=point_roi_names,
        )

    def __repr__(self) -> str:
        return (
            f"ACCFv3(shape={self.shape_hw}, "
            f"n_regions={self.n_regions}, scale={self._scale})"
        )


# ---- backward compatibility ----

AtlasData = ACCFv3  # alias


def load_atlas(path: Path) -> ACCFv3:
    """Load atlas — backward-compatible wrapper for ``ACCFv3.from_mat``."""
    return ACCFv3.from_mat(path)


# ---- internal helpers ----


def boundaries_from_id_map(region_ids: np.ndarray) -> list[list[np.ndarray]]:
    """Derive per-region boundary contours from an integer region-ID map.

    Returns a list-of-lists matching the ``B`` layout ``draw_boundaries``
    expects: index 0 is the whole-brain outline (``region_ids > 1``, skipped by
    default), then one entry per region ID (ascending) holding its contours as
    ``(M, 2)`` (row, col) arrays.  Contours are in atlas-pixel space (draw with
    ``scale = 1.0``)."""
    from skimage.measure import find_contours

    out: list[list[np.ndarray]] = [find_contours((region_ids > 1).astype(float), 0.5)]
    for rid in sorted(int(v) for v in np.unique(region_ids)):
        if rid <= 1:
            continue
        out.append(find_contours((region_ids == rid).astype(float), 0.5))
    return out


def _parse_boundaries(
    root: h5py.File, b_dataset: h5py.Dataset
) -> list[list[np.ndarray]]:
    """Parse MATLAB cell-of-cells boundary structure from HDF5."""
    n_regions = b_dataset.shape[0]
    all_boundaries: list[list[np.ndarray]] = []

    for i in range(n_regions):
        ref = b_dataset[i, 0]
        inner = root[ref]  # (1, n_contours) object refs
        n_contours = inner.shape[1]
        contours: list[np.ndarray] = []
        for j in range(n_contours):
            contour_ref = inner[0, j]
            raw = np.array(root[contour_ref])  # (2, M)
            contours.append(raw.T)  # (M, 2) — row=y, col=x
        all_boundaries.append(contours)

    return all_boundaries
