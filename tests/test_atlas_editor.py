"""Display-independent smoke test for the tilt atlas-builder GUI.

Builds the window headlessly and exercises the pure projection/preview path
(compute_preview) with synthetic and, if present, the real CCF volume.
"""

from pathlib import Path

import numpy as np
import pytest

from asvimg import atlas_build as ab

ST = Path("resources/allenCCF/structure_tree_safe_2017.csv")
VOL = Path("resources/atlas_from_figshare/annotation_volume_10um_by_index.npy")
TMPL = Path("resources/atlas_from_figshare/template_volume_10um.npy")


def _dpg_ok() -> bool:
    try:
        import dearpygui.dearpygui as dpg

        dpg.create_context()
        dpg.destroy_context()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _dpg_ok(), reason="dearpygui unavailable (headless)")


def test_editor_builds():
    import dearpygui.dearpygui as dpg

    from asvimg.gui.atlas_editor import AtlasEditor, _WIN, _STATUS, _T

    dpg.create_context()
    try:
        AtlasEditor().build()
        assert dpg.does_item_exist(_WIN)
        for t in ("s_ap", "s_ml", "s_dv", "i_h", "i_w", "c_olf", "c_cb", "p_annot"):
            assert dpg.does_item_exist(_T + t), t
        assert dpg.does_item_exist(_STATUS)
    finally:
        dpg.destroy_context()


@pytest.mark.skipif(not ST.exists(), reason="structure tree csv not present")
def test_compute_preview_synthetic():
    from asvimg.gui.atlas_editor import AtlasEditor

    st = ab.load_structure_tree(ST)
    idx = int(np.where(st.is_isocortex)[0][0])          # a real isocortex index
    av = np.zeros((40, 20, 40), np.uint16)
    av[10:30, 0:5, 8:32] = idx                          # a block at the dorsal surface

    ed = AtlasEditor()
    ed._st, ed._av_ds, ed._tv_ds, ed._loaded = st, av, av.copy(), True
    ids, n = ed.compute_preview(dict(
        tilt_ap_deg=0, tilt_ml_deg=0, tilt_dv_deg=0, out_hw=(60, 60),
        min_pixels=1, include_olfactory=False, include_cerebellum=False))
    assert ids.shape == (60, 60)
    assert n >= 1 and (ids > 1).any()


@pytest.mark.skipif(not (VOL.exists() and TMPL.exists()), reason="CCF volume not present")
def test_real_preview_tilt_changes_projection():
    from asvimg.gui.atlas_editor import AtlasEditor

    ed = AtlasEditor(preview_downsample=8)              # small for speed
    ed.load_data()
    assert ed._av_ds is not None

    base = dict(tilt_ap_deg=0, tilt_dv_deg=0, out_hw=(150, 150), min_pixels=15,
                include_olfactory=False, include_cerebellum=False)
    ids0, _ = ed.compute_preview({**base, "tilt_ml_deg": 0})
    ids1, _ = ed.compute_preview({**base, "tilt_ml_deg": 20})
    assert (ids0 > 1).sum() > 100 and (ids1 > 1).sum() > 100
    assert not np.array_equal(ids0, ids1)               # tilt changes the surface projected
