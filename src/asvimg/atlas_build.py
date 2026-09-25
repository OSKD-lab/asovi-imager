"""Build a top-view Allen-CCF cortical atlas (``ACCFv3``-readable) from the 3D
annotation volume — the generator behind ``asovi-atlas``.

Input (download once, e.g. into ``resources/atlas_from_figshare/``; see
``resources/allenCCF/setup_utils.m`` / figshare 25365829):
  * ``annotation_volume_10um_by_index.npy`` — (AP,DV,ML) uint16, values = 1-based
    row index into the structure tree.
  * ``template_volume_10um.npy``            — same shape, grayscale reference.
  * ``structure_tree_safe_2017.csv``        — index → acronym / colour / ontology.

Method (verified: isocortex top-down projection reproduces the shipped atlas to
brain-outline IoU 0.994):
  1. for each (AP,ML) column take the first **isocortex** voxel going down (DV) —
     its structure = the dorsal-surface area (layer 1);
  2. label each pixel with its Allen acronym (layer suffix stripped);
  3. crop AP to an isotropic window + resample to the requested H×W;
  4. flip to the frame ``ACCFv3`` exposes (so the hardcoded point ROIs still land),
     split each area L/R at the midline, assign IDs + a ``region_names`` map;
  5. project the template the same way for the grayscale reference image;
  6. write an ``ACCFv3``-readable HDF5 (region boundaries are derived on load).

Nothing is shipped: the atlas is built here, on the machine that uses it, into
``config.USER_ATLAS``.  The MATLAB-era ``wfciAnnotationData.mat`` that used to
ride inside the package was itself exactly the Allen isocortex dorsal
parcellation (34 areas, **bilateral**) with mislabelled ``_REGION_NAMES``; what
this module generates is both correctly named and split L/R.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Default crop window (fraction of the input AP / ML extent) reproducing the
# shipped frame: AP [4/1320, 1144/1320] (=[4:1144] at full res), ML full width.
_AP_CROP_DEFAULT = (4 / 1320, 1144 / 1320)
_ML_CROP_DEFAULT = (0.0, 1.0)
_BREGMA_AP = 540            # allenCCFbregma: [AP,DV,LR] = [540,0,570]
_LAYER = re.compile(r"(1|2/3|4|5|6a|6b)$")

# Right-hemisphere = higher column in the ACCFv3 frame (matches DEFAULT_POINT_ROIS).
ATLAS_FORMAT_KEY = "asovi_atlas_version"
ATLAS_FORMAT_VERSION = 1


@dataclass
class AtlasArrays:
    """Everything ``save_atlas_mat`` writes / ``ACCFv3.from_mat`` reads."""
    region_ids: np.ndarray          # (H,W) uint16, 1=background
    image_rgb: np.ndarray           # (H,W,3) float32 in [0,1]
    fixed_points: np.ndarray        # (3,2) float64 (row,col) midline landmarks
    scale: float                    # boundary/display scale (contours are in px -> 1.0)
    region_names: dict              # id -> acronym (e.g. 6:"VISp", 7:"VISp_R")
    point_rois: np.ndarray          # (M,2) float64 (row,col)
    point_roi_names: list           # length M


# --------------------------------------------------------------------------- #
# structure tree
# --------------------------------------------------------------------------- #

# Allen structure IDs whose subtree defines a projectable surface set.
_ID_ISOCORTEX = 315
_ID_OLFACTORY = 698   # OLF (olfactory areas; the dorsal-visible part is the main olfactory bulb)
_ID_CEREBELLUM = 512  # CB


@dataclass
class StructureTree:
    """1-based (index 0 is a pad so ``arr[structure_index]`` indexes directly)."""
    acronym: np.ndarray          # full acronym, e.g. "VISp1"
    base: np.ndarray             # layer-stripped, e.g. "VISp"
    is_isocortex: np.ndarray     # bool
    is_olfactory: np.ndarray     # bool
    is_cerebellum: np.ndarray    # bool
    colour: np.ndarray
    sid: np.ndarray | None = None  # Allen structure id (for raw-.nrrd id -> index)

    def surface_mask(self, *, include_olfactory: bool = False,
                     include_cerebellum: bool = False) -> np.ndarray:
        """Per-index bool: which structures are eligible for the top-down surface.

        Isocortex is always included; the olfactory bulb / cerebellum are opt-in."""
        m = self.is_isocortex.copy()
        if include_olfactory:
            m |= self.is_olfactory
        if include_cerebellum:
            m |= self.is_cerebellum
        return m


def load_structure_tree(csv_path: str | Path) -> StructureTree:
    """Parse structure_tree_safe_2017.csv into a :class:`StructureTree`."""
    acr, colour, sid = [""], ["#000000"], [0]
    isctx, isolf, iscb = [False], [False], [False]
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            acr.append(row["acronym"])
            colour.append("#" + (row.get("color_hex_triplet") or "888888"))
            sid.append(int(row["id"]))
            path = row["structure_id_path"]
            isctx.append(f"/{_ID_ISOCORTEX}/" in path)
            isolf.append(f"/{_ID_OLFACTORY}/" in path)
            iscb.append(f"/{_ID_CEREBELLUM}/" in path)
    acr = np.array(acr, dtype=object)
    base = np.array([_LAYER.sub("", a) if a else "" for a in acr], dtype=object)
    return StructureTree(
        acronym=acr, base=base,
        is_isocortex=np.array(isctx, dtype=bool),
        is_olfactory=np.array(isolf, dtype=bool),
        is_cerebellum=np.array(iscb, dtype=bool),
        colour=np.array(colour, dtype=object),
        sid=np.array(sid, dtype=np.int64),
    )


# --------------------------------------------------------------------------- #
# volume loading  (.npy by-index, or raw .nrrd)
# --------------------------------------------------------------------------- #

def load_ccf_volume(path: str | Path, structure_tree: StructureTree | None = None):
    """Load a CCF volume as `(AP, DV, ML)` of 1-based structure-tree row indices.

    ``.npy`` → the ``annotation_volume_10um_by_index.npy`` (already by-index),
    returned as a read-only memmap.  ``.nrrd`` → read fully; if the values look
    like raw Allen structure **ids** (max exceeds the number of tree rows) they
    are mapped to 1-based row indices via ``structure_tree`` (required)."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".npy":
        return np.load(path, mmap_mode="r")
    if suf == ".nrrd":
        try:
            import nrrd
        except ModuleNotFoundError as exc:  # optional extra
            raise ModuleNotFoundError(
                'reading .nrrd volumes needs the `atlas` extra: '
                'uv pip install "asovi-imager[atlas]" (or pip). '
                '.npy volumes need nothing extra'
            ) from exc

        data, _hdr = nrrd.read(str(path))
        data = np.asarray(data)
        n_rows = len(structure_tree.acronym) if structure_tree is not None else 0
        if structure_tree is not None and structure_tree.sid is not None and int(data.max()) >= n_rows:
            # raw Allen ids -> 1-based row index (background id 0 -> 0)
            id_to_row = np.zeros(int(structure_tree.sid.max()) + 1, dtype=np.uint16)
            for row_idx, aid in enumerate(structure_tree.sid):
                if aid > 0:
                    id_to_row[aid] = row_idx
            data = id_to_row[np.clip(data.astype(np.int64), 0, id_to_row.size - 1)]
        return data.astype(np.uint16)
    raise ValueError(f"unsupported volume format: {path.name} (need .npy or .nrrd)")


# --------------------------------------------------------------------------- #
# tilt  (rotate the volume about its centre before the top-down projection)
# --------------------------------------------------------------------------- #

def tilt_volume(volume, *, tilt_ap_deg: float = 0.0, tilt_ml_deg: float = 0.0,
                tilt_dv_deg: float = 0.0):
    """Rotate an `(AP, DV, ML)` volume about its centre (order=0, shape kept).

    ``tilt_ap_deg`` rotates about the AP axis (roll, in the DV–ML plane),
    ``tilt_ml_deg`` about the ML axis (pitch, in the AP–DV plane), ``tilt_dv_deg``
    about the DV axis (yaw, in the AP–ML plane). Applied AP→ML→DV."""
    from scipy.ndimage import rotate

    v = np.asarray(volume)
    for ang, axes in ((tilt_ap_deg, (1, 2)), (tilt_ml_deg, (0, 1)), (tilt_dv_deg, (0, 2))):
        if ang:
            v = rotate(v, float(ang), axes=axes, order=0, reshape=False, prefilter=False)
    return v


# --------------------------------------------------------------------------- #
# top-down surface projection
# --------------------------------------------------------------------------- #

def top_down_index(annotation_volume, membership: np.ndarray, *, chunk: int = 132) -> np.ndarray:
    """(AP,ML) map of the structure index of the first *member* voxel going down.

    ``membership`` is a per-structure-index bool mask (see ``StructureTree.
    surface_mask``). ``annotation_volume`` is an (AP,DV,ML) array/memmap of 1-based
    indices. Columns with no member → 0. Chunked over AP to bound peak memory."""
    av = annotation_volume
    ap, dv, ml = av.shape
    out = np.zeros((ap, ml), dtype=np.uint16)
    for a0 in range(0, ap, chunk):
        a1 = min(a0 + chunk, ap)
        block = np.asarray(av[a0:a1])
        member = membership[block]                      # (n,DV,ML) bool
        has = member.any(axis=1)
        first = np.argmax(member, axis=1)               # (n,ML)
        ii, kk = np.mgrid[0:a1 - a0, 0:ml]
        v = block[ii, first, kk]
        v[~has] = 0
        out[a0:a1] = v
    return out


def top_down_gray(template_volume, annotation_volume, membership: np.ndarray,
                  *, chunk: int = 132) -> np.ndarray:
    """Grayscale top-down projection of the template at the same surface voxel."""
    tv, av = template_volume, annotation_volume
    ap, dv, ml = tv.shape
    out = np.zeros((ap, ml), dtype=np.float32)
    for a0 in range(0, ap, chunk):
        a1 = min(a0 + chunk, ap)
        member = membership[np.asarray(av[a0:a1])]
        has = member.any(axis=1)
        first = np.argmax(member, axis=1)
        blk = np.asarray(tv[a0:a1]).astype(np.float32)
        ii, kk = np.mgrid[0:a1 - a0, 0:ml]
        g = blk[ii, first, kk]
        g[~has] = 0
        out[a0:a1] = g
    return out


# --------------------------------------------------------------------------- #
# crop + resample to the atlas frame
# --------------------------------------------------------------------------- #

def _resample_labels(arr2d: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resample of a (AP,ML) map to (H,W) — exact striding when
    it divides evenly (reproduces the validated 1140->285 x4 path)."""
    H, W = out_hw
    ap, ml = arr2d.shape
    if ap % H == 0 and ml % W == 0:
        return arr2d[:: ap // H, :: ml // W]
    ri = (np.arange(H) * ap / H).astype(int).clip(0, ap - 1)
    ci = (np.arange(W) * ml / W).astype(int).clip(0, ml - 1)
    return arr2d[np.ix_(ri, ci)]


# Where `asovi-atlas` writes when --out is not given -- and, since nothing ships
# inside the package any more, also where an empty `annotation_atlas_path`
# resolves to.  One definition, in config, so the builder and the reader cannot
# drift apart.
from .config import USER_ATLAS as DEFAULT_GENERATED_ATLAS

def _to_atlas_frame(td: np.ndarray, out_hw: tuple[int, int], *,
                    ap_crop: tuple[float, float] = _AP_CROP_DEFAULT,
                    ml_crop: tuple[float, float] = _ML_CROP_DEFAULT):
    """Crop the (AP,ML) map to the given window and resample to (H,W), then flip
    ML into the ACCFv3 frame.

    ``ap_crop`` / ``ml_crop`` are ``(lo, hi)`` fractions in [0,1] of the input's own
    AP / ML extent (resolution-agnostic — works on the full-res or a downsampled/
    tilted map).  The defaults reproduce the shipped frame.  A symmetric ``ml_crop``
    keeps the midline centred, so the L/R split at ``(W-1)/2`` stays correct."""
    H, W = out_hw
    ap, ml = td.shape
    a0, a1 = int(round(ap_crop[0] * ap)), int(round(ap_crop[1] * ap))
    c0, c1 = int(round(ml_crop[0] * ml)), int(round(ml_crop[1] * ml))
    crop = td[max(0, a0):max(a0 + 1, a1), max(0, c0):max(c0 + 1, c1)]
    res = _resample_labels(crop, (H, W))
    return res[:, ::-1].copy(), 1.0                      # -> ACCFv3 frame (raw matched .T[:,::-1])


# --------------------------------------------------------------------------- #
# labels: acronym map -> WFCI-style IDs, split L/R at the midline
# --------------------------------------------------------------------------- #

def assign_ids(base_acr_map: np.ndarray, *, min_pixels: int = 15):
    """Turn a per-pixel base-acronym string map into (region_ids, region_names).

    Areas are ordered by descending pixel count; each area gets a left ID then a
    right ID (base + '_R'), split at the midline col=(W-1)/2. Tiny areas
    (< ``min_pixels`` total) are dropped to background. Background id = 1."""
    H, W = base_acr_map.shape
    mid = (W - 1) / 2.0
    # count areas
    from collections import Counter
    counts = Counter(a for a in base_acr_map.ravel() if a)
    areas = [a for a, n in counts.most_common() if n >= min_pixels]

    region_ids = np.ones((H, W), dtype=np.uint16)
    region_names: dict[int, str] = {1: "background"}
    cols = np.broadcast_to(np.arange(W), (H, W))
    nid = 2
    for a in areas:
        m = base_acr_map == a
        left = m & (cols < mid)
        right = m & (cols >= mid)
        region_ids[left] = nid
        region_names[nid] = a
        nid += 1
        region_ids[right] = nid
        region_names[nid] = a + "_R"
        nid += 1
    return region_ids, region_names


# --------------------------------------------------------------------------- #
# landmarks + point ROIs, scaled to the output resolution
# --------------------------------------------------------------------------- #

def _fixed_points(out_hw: tuple[int, int]) -> np.ndarray:
    """3 midline landmarks (row,col): bregma / lambda / anterior, from the shipped
    atlas (285^2: rows 128.75/212.5/55, col 142.5), scaled to (H,W)."""
    H, W = out_hw
    sr, sc = H / 285.0, W / 285.0
    col = (W - 1) / 2.0
    return np.array([[128.75 * sr, col], [212.5 * sr, col], [55.0 * sr, col]], dtype=np.float64)


def _point_rois(out_hw: tuple[int, int]):
    """The hardcoded default point ROIs (names preserved), scaled to (H,W)."""
    from .atlas import DEFAULT_POINT_ROIS, DEFAULT_POINT_ROI_NAMES

    H, W = out_hw
    sr, sc = H / 285.0, W / 285.0
    pts_r = DEFAULT_POINT_ROIS.astype(np.float64).copy()
    pts_r[:, 0] *= sr
    pts_r[:, 1] *= sc
    names_r = list(DEFAULT_POINT_ROI_NAMES)
    pts_l = pts_r.copy()
    pts_l[:, 1] = (W - 1) - pts_l[:, 1]
    names_l = [n.replace("_R", "_L") for n in names_r]
    return np.vstack([pts_r, pts_l]), names_r + names_l


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #

def build_atlas(
    annotation_volume_path: str | Path,
    template_volume_path: str | Path,
    structure_tree_path: str | Path,
    *,
    out_hw: tuple[int, int] = (285, 285),
    min_pixels: int = 15,
    include_olfactory: bool = False,
    include_cerebellum: bool = False,
    tilt_ap_deg: float = 0.0,
    tilt_ml_deg: float = 0.0,
    tilt_dv_deg: float = 0.0,
    tilt_downsample: int = 2,
    ap_crop: tuple[float, float] = _AP_CROP_DEFAULT,
    ml_crop: tuple[float, float] = _ML_CROP_DEFAULT,
) -> AtlasArrays:
    """Build a top-view cortical atlas from the CCF volumes.

    Isocortex is always projected; ``include_olfactory`` adds the (anterior) main
    olfactory bulb and ``include_cerebellum`` the (posterior) cerebellum. The
    atlas frame stays isocortex-calibrated (so the default point ROIs still land).

    ``tilt_ap_deg`` / ``tilt_ml_deg`` / ``tilt_dv_deg`` rotate the volume about its
    centre before the top-down projection (roll / pitch / yaw) — for a tilted
    top-view.  When any tilt is set the volume is downsampled by ``tilt_downsample``
    first (rotating the full volume is prohibitive); the output resolution is
    unaffected.  Volumes may be ``.npy`` (by-index) or ``.nrrd``."""
    st = load_structure_tree(structure_tree_path)
    membership = st.surface_mask(include_olfactory=include_olfactory,
                                 include_cerebellum=include_cerebellum)
    base = st.base
    tilted = bool(tilt_ap_deg or tilt_ml_deg or tilt_dv_deg)

    if tilted:
        ds = max(1, int(tilt_downsample))
        tk = dict(tilt_ap_deg=tilt_ap_deg, tilt_ml_deg=tilt_ml_deg, tilt_dv_deg=tilt_dv_deg)
        av = tilt_volume(np.asarray(load_ccf_volume(annotation_volume_path, st)[::ds, ::ds, ::ds]), **tk)
        tv = tilt_volume(np.asarray(load_ccf_volume(template_volume_path)[::ds, ::ds, ::ds]), **tk)
    else:
        av = load_ccf_volume(annotation_volume_path, st)
        tv = load_ccf_volume(template_volume_path)

    crop_kw = dict(ap_crop=ap_crop, ml_crop=ml_crop)
    td = top_down_index(av, membership)                  # (AP,ML) structure index
    frame, scale = _to_atlas_frame(td, out_hw, **crop_kw)  # (H,W) in the ACCFv3 frame

    base_map = np.array(
        [base[v] if v < len(base) else "" for v in frame.ravel()], dtype=object
    ).reshape(frame.shape)
    region_ids, region_names = assign_ids(base_map, min_pixels=min_pixels)

    gray = top_down_gray(tv, av, membership)
    gray_frame, _ = _to_atlas_frame(gray.astype(np.uint16), out_hw, **crop_kw)  # same crop/frame
    g = gray_frame.astype(np.float32)
    g = g / g.max() if g.max() > 0 else g
    image_rgb = np.repeat(g[:, :, None], 3, axis=2).astype(np.float32)

    pr, prn = _point_rois(out_hw)
    return AtlasArrays(
        region_ids=region_ids, image_rgb=image_rgb, fixed_points=_fixed_points(out_hw),
        # boundaries are derived in atlas-pixel space on load, so they draw 1:1
        scale=1.0, region_names=region_names, point_rois=pr, point_roi_names=prn,
    )


# --------------------------------------------------------------------------- #
# writer  (ACCFv3.from_mat reads this new HDF5 layout)
# --------------------------------------------------------------------------- #

def save_atlas_mat(path: str | Path, atlas: AtlasArrays) -> Path:
    """Write an ``ACCFv3``-readable HDF5 atlas (boundaries are derived on load)."""
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = sorted(atlas.region_names)
    with h5py.File(path, "w") as f:
        f.create_dataset(ATLAS_FORMAT_KEY, data=ATLAS_FORMAT_VERSION)
        f.create_dataset("allenImgID_resized", data=atlas.region_ids.astype(np.uint16))
        f.create_dataset("allenImg_resized", data=atlas.image_rgb.astype(np.float32))
        f.create_dataset("fixedPos", data=atlas.fixed_points.astype(np.float64))
        f.create_dataset("scale", data=float(atlas.scale))
        f.create_dataset("region_ids", data=np.array(ids, dtype=np.int64))
        f.create_dataset("region_acronyms",
                         data=np.array([atlas.region_names[i] for i in ids], dtype=object),
                         dtype=h5py.string_dtype())
        f.create_dataset("point_rois", data=np.asarray(atlas.point_rois, dtype=np.float64))
        f.create_dataset("point_roi_names",
                         data=np.array(list(atlas.point_roi_names), dtype=object),
                         dtype=h5py.string_dtype())
    return path


# --------------------------------------------------------------------------- #
# CLI  (asovi-atlas)
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    p = argparse.ArgumentParser(
        prog="asovi-atlas",
        description="Build a top-view Allen-CCF cortical atlas (ACCFv3-readable) "
                    "from the 3D annotation volume.")
    p.add_argument("--annotation", default=None, help="annotation_volume_10um_by_index.npy (default: resolve/download)")
    p.add_argument("--template", default=None, help="template_volume_10um.npy (default: resolve/download)")
    p.add_argument("--structure-tree", default=None, help="structure_tree_safe_2017.csv (default: resolve)")
    p.add_argument("--atlas-dir", default=None,
                   help=f"CCF volume dir (else $ASOVI_ATLAS_DIR, resources/, or ~/.asovi/atlas)")
    p.add_argument("--download", action="store_true",
                   help="download the CCF volumes (~4.8 GB, figshare) without prompting")
    p.add_argument("--out", default=str(DEFAULT_GENERATED_ATLAS),
                   help="output atlas file (HDF5; ACCFv3.from_mat reads it). "
                        f"Default: {DEFAULT_GENERATED_ATLAS}")
    p.add_argument("--hw", default="285,285", help="output height,width (e.g. 285,285)")
    p.add_argument("--min-pixels", type=int, default=15,
                   help="drop areas smaller than this (per hemisphere) to background")
    p.add_argument("--include-olfactory", action="store_true",
                   help="also project the main olfactory bulb (anterior)")
    p.add_argument("--include-cerebellum", action="store_true",
                   help="also project the cerebellum (posterior)")
    p.add_argument("--tilt-ap", type=float, default=0.0, help="tilt about the AP axis (roll, deg)")
    p.add_argument("--tilt-ml", type=float, default=0.0, help="tilt about the ML axis (pitch, deg)")
    p.add_argument("--tilt-dv", type=float, default=0.0, help="tilt about the DV axis (yaw, deg)")
    p.add_argument("--ap-crop", default=None, metavar="LO,HI",
                   help="AP crop window as fractions 0..1 (default reproduces the shipped frame)")
    p.add_argument("--ml-crop", default=None, metavar="LO,HI",
                   help="ML crop window as fractions 0..1 (default full width; keep symmetric)")
    a = p.parse_args(list(sys.argv[1:] if argv is None else argv))

    from . import ccf_data

    def _prog(label, done, total):
        pct = 100 * done / total if total else 0
        print(f"\r[asovi-atlas] downloading {label}: {pct:5.1f}%  "
              f"({done / 1e6:.0f}/{total / 1e6:.0f} MB)", end="", flush=True)
        if done >= total:
            print()

    try:
        if a.annotation and a.template:
            annot, tmpl = Path(a.annotation), Path(a.template)
            if not annot.exists() or not tmpl.exists():
                print(f"[asovi-atlas] volume not found: {annot if not annot.exists() else tmpl}")
                return 2
        else:
            annot, tmpl = ccf_data.ensure_ccf_volumes(a.atlas_dir, download=a.download, reporter=_prog)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[asovi-atlas] {exc}")
        return 2
    # The structure tree is not in the figshare article, so it is fetched
    # separately -- under the same consent rule as the volumes.
    if a.structure_tree:
        st = Path(a.structure_tree)
        if not st.exists():
            print(f"[asovi-atlas] structure tree not found: {st}")
            return 2
    else:
        try:
            st = ccf_data.ensure_structure_tree(a.atlas_dir, download=a.download, reporter=_prog)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"[asovi-atlas] {exc}")
            return 2
    h, w = (int(x) for x in a.hw.split(","))
    crop = {}
    if a.ap_crop:
        crop["ap_crop"] = tuple(float(x) for x in a.ap_crop.split(","))
    if a.ml_crop:
        crop["ml_crop"] = tuple(float(x) for x in a.ml_crop.split(","))
    print(f"[asovi-atlas] building {h}x{w} atlas from {annot} ...")
    arr = build_atlas(annot, tmpl, st,
                      out_hw=(h, w), min_pixels=a.min_pixels,
                      include_olfactory=a.include_olfactory,
                      include_cerebellum=a.include_cerebellum,
                      tilt_ap_deg=a.tilt_ap, tilt_ml_deg=a.tilt_ml, tilt_dv_deg=a.tilt_dv, **crop)
    out = save_atlas_mat(a.out, arr)
    print(f"[asovi-atlas] wrote {out}  ({len(arr.region_names) - 1} regions, "
          f"{int((arr.region_ids > 1).sum())} cortical pixels)")
    print(f"  point it at the pipeline via  annotation_atlas_path: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
