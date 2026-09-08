"""asovi-atlas: build a top-view CCF atlas from the annotation volume.

Unit tests cover the pure logic (L/R split, resampling, point-ROI scaling, and
the save -> ACCFv3.from_mat round-trip) with synthetic data. A volume-gated
integration test builds the real 285^2 atlas and checks it reproduces the
shipped one (brain-outline IoU) with the default point ROIs landing in cortex.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from asvimg.config import BUNDLED_ATLAS
from asvimg import atlas_build as ab
from asvimg.atlas import ACCFv3, boundaries_from_id_map

VOL = Path("resources/atlas_from_figshare/annotation_volume_10um_by_index.npy")
TMPL = Path("resources/atlas_from_figshare/template_volume_10um.npy")
ST = Path("resources/allenCCF/structure_tree_safe_2017.csv")
SHIPPED = BUNDLED_ATLAS
_HAVE_VOL = VOL.exists() and TMPL.exists() and ST.exists()


# ---- pure logic ------------------------------------------------------------

def test_assign_ids_splits_left_right():
    # W=4 -> midline=1.5; VISp on both sides, MOs only left
    m = np.array([
        ["VISp", "VISp", "VISp", "VISp"],
        ["MOs", "MOs", "VISp", "VISp"],
        ["MOs", "MOs", "", ""],
    ], dtype=object)
    ids, names = ab.assign_ids(m, min_pixels=1)
    assert names[1] == "background"
    # VISp is the larger area -> ids 2 (L) / 3 (R)
    assert names[2] == "VISp" and names[3] == "VISp_R"
    assert names[4] == "MOs" and names[5] == "MOs_R"
    # left VISp pixels (col<1.5) got id 2, right got id 3
    assert ids[0, 0] == 2 and ids[0, 3] == 3
    # MOs only on the left -> id 4 present, id 5 (MOs_R) has no pixels
    assert (ids == 4).any() and not (ids == 5).any()


def test_assign_ids_drops_tiny_areas():
    m = np.full((10, 10), "", dtype=object)
    m[0:5, 0:5] = "VISp"
    m[9, 9] = "TEa"                       # 1 px -> dropped at min_pixels=15
    ids, names = ab.assign_ids(m, min_pixels=15)
    assert "TEa" not in names.values()
    assert ids[9, 9] == 1                 # background


def test_resample_labels_uses_exact_striding():
    a = np.arange(64, dtype=np.uint16).reshape(8, 8)
    r = ab._resample_labels(a, (4, 4))
    assert r.shape == (4, 4)
    np.testing.assert_array_equal(r, a[::2, ::2])   # nearest via striding


def test_to_atlas_frame_crop_window():
    td = np.arange(120 * 80, dtype=np.uint16).reshape(120, 80)   # (AP, ML)
    full, _ = ab._to_atlas_frame(td, (20, 20))                   # default crop (shipped frame)
    sub, _ = ab._to_atlas_frame(td, (20, 20), ap_crop=(0.5, 1.0), ml_crop=(0.25, 0.75))
    assert full.shape == (20, 20) and sub.shape == (20, 20)
    assert not np.array_equal(full, sub)                          # different windows -> different content
    # a full-window crop equals passing the explicit (0,1)x(0,1) window
    f2, _ = ab._to_atlas_frame(td, (20, 20), ap_crop=ab._AP_CROP_DEFAULT, ml_crop=(0.0, 1.0))
    np.testing.assert_array_equal(full, f2)


def test_point_rois_preserve_names_and_scale():
    pr285, names285 = ab._point_rois((285, 285))
    assert len(names285) == 30 and pr285.shape == (30, 2)
    assert "VISp_R" in names285 and "VISp_L" in names285   # names preserved
    pr570, names570 = ab._point_rois((570, 570))
    assert names570 == names285
    # doubling the resolution doubles the coordinates
    np.testing.assert_allclose(pr570[0], pr285[0] * 2, rtol=1e-6)


def test_fixed_points_on_midline():
    fp = ab._fixed_points((285, 285))
    assert fp.shape == (3, 2)
    np.testing.assert_allclose(fp[:, 1], 142.0)           # col = (285-1)/2


def test_surface_mask_options():
    st = ab.StructureTree(
        acronym=np.array(["", "VISp1", "MOB1", "CENT"], dtype=object),
        base=np.array(["", "VISp", "MOB", "CENT"], dtype=object),
        is_isocortex=np.array([False, True, False, False]),
        is_olfactory=np.array([False, False, True, False]),
        is_cerebellum=np.array([False, False, False, True]),
        colour=np.array(["#000000"] * 4, dtype=object),
    )
    assert list(st.surface_mask()) == [False, True, False, False]           # isocortex only
    assert list(st.surface_mask(include_olfactory=True)) == [False, True, True, False]
    assert list(st.surface_mask(include_cerebellum=True)) == [False, True, False, True]
    assert list(st.surface_mask(include_olfactory=True, include_cerebellum=True)) \
        == [False, True, True, True]


@pytest.mark.skipif(not ST.exists(), reason="structure tree csv not present")
def test_load_structure_tree_flags():
    st = ab.load_structure_tree(ST)
    assert st.is_isocortex.sum() > 100      # many cortical layers/areas
    assert st.is_olfactory.sum() > 0        # OLF subtree present
    assert st.is_cerebellum.sum() > 0       # CB subtree present


def test_boundaries_from_id_map():
    ids = np.ones((20, 20), dtype=np.uint16)
    ids[4:10, 4:10] = 2
    ids[12:18, 12:18] = 3
    b = boundaries_from_id_map(ids)
    assert len(b) == 3                                    # outline + 2 regions
    assert all(len(c) >= 1 for c in b)                    # each has >=1 contour


def test_save_load_roundtrip():
    H = W = 16
    ids = np.ones((H, W), dtype=np.uint16)
    ids[2:8, 2:7] = 2
    ids[2:8, 9:14] = 3
    ids[10:14, 4:12] = 4
    arr = ab.AtlasArrays(
        region_ids=ids,
        image_rgb=np.repeat(np.linspace(0, 1, H * W).reshape(H, W)[:, :, None], 3, axis=2).astype("float32"),
        fixed_points=ab._fixed_points((H, W)),
        scale=1.0,
        region_names={1: "background", 2: "VISp", 3: "VISp_R", 4: "MOs"},
        point_rois=np.array([[3.0, 4.0], [3.0, 11.0]]),
        point_roi_names=["VISp", "VISp_R"],
    )
    with tempfile.TemporaryDirectory() as td:
        p = ab.save_atlas_mat(Path(td) / "a.h5", arr)
        atlas = ACCFv3.from_mat(p)
        np.testing.assert_array_equal(atlas.region_ids, ids)
        assert atlas.region_names[2] == "VISp" and atlas.region_names[4] == "MOs"
        assert atlas.scale == 1.0
        assert atlas.shape_hw == (H, W)
        assert atlas.point_roi_names == ["VISp", "VISp_R"]
        assert len(atlas.boundaries) == 4                 # outline + 3 regions
        assert atlas.get_mask("VISp").sum() == (ids == 2).sum()


# ---- integration (real CCF volume) ----------------------------------------

@pytest.mark.skipif(not _HAVE_VOL, reason="CCF annotation volume not present")
def test_build_reproduces_shipped_atlas():
    with tempfile.TemporaryDirectory() as td:
        arr = ab.build_atlas(VOL, TMPL, ST, out_hw=(285, 285))
        p = ab.save_atlas_mat(Path(td) / "gen.h5", arr)
        gen = ACCFv3.from_mat(p)

        assert gen.shape_hw == (285, 285)
        # correctly-labelled regions (unlike the shipped _REGION_NAMES)
        assert "VISp" in gen.region_names.values()
        assert "VISp_R" in gen.region_names.values()

        # brain outline matches the shipped atlas
        ship = ACCFv3.from_mat(SHIPPED)
        gb, sb = gen.region_ids > 1, ship.region_ids > 1
        iou = (gb & sb).sum() / (gb | sb).sum()
        assert iou > 0.95, f"brain-outline IoU {iou:.3f}"

        # every default point ROI lands inside cortex (frame is compatible)
        inside = sum(int(gen.region_ids[int(round(r)), int(round(c))]) > 1
                     for r, c in gen.point_rois)
        assert inside == len(gen.point_rois)
