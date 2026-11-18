"""The control-point picker must draw the atlas overlay in the atlas's own frame.

``cpselect_gui``'s defaults describe the legacy MATLAB atlas (285x285,
boundaries stored at 4x).  An atlas built by ``asovi-atlas`` stores boundaries in
atlas pixels (``scale == 1.0``) at any ``--hw``; a caller that falls back to the
defaults draws its region borders at a quarter of their size.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from asvimg.atlas import ACCFv3
from asvimg.atlas_build import AtlasArrays, _fixed_points, _point_rois, save_atlas_mat
from asvimg.cpselect import atlas_overlay_kwargs


def _generated_atlas(tmp_path, hw):
    h, w = hw
    ids = np.ones(hw, np.uint16)
    ids[h // 4 : 3 * h // 4, w // 8 : w // 2] = 2
    ids[h // 4 : 3 * h // 4, w // 2 : 7 * w // 8] = 3
    pr, prn = _point_rois(hw)
    arrays = AtlasArrays(
        ids, np.zeros((h, w, 3), np.float32), _fixed_points(hw), 1.0,
        {1: "background", 2: "X", 3: "X_R"}, pr, prn,
    )
    return ACCFv3.from_mat(save_atlas_mat(tmp_path / f"atlas_{h}x{w}.h5", arrays))


@pytest.mark.parametrize("hw", [(285, 285), (512, 512), (300, 400)])
def test_overlay_kwargs_follow_the_atlas(tmp_path, hw):
    atlas = _generated_atlas(tmp_path, hw)
    kw = atlas_overlay_kwargs(atlas)
    h, w = hw

    assert kw["boundary_scale"] == atlas.scale == 1.0
    # Drawn borders reach the region's real extent, not a quarter of it.
    drawn = np.vstack([c for g in kw["boundaries"][1:] for c in g]) * kw["boundary_scale"]
    assert drawn[:, 0].max() == pytest.approx(3 * h // 4 - 0.5)
    assert drawn[:, 1].max() == pytest.approx(7 * w // 8 - 0.5)

    # Preset landmarks sit on the midline, at the 285-frame rows scaled to H.
    assert kw["ref_landmarks"]["Front Edge"] == pytest.approx((55.0 * h / 285, w / 2))
    assert kw["ref_landmarks"]["RSC Edge"] == pytest.approx((212.5 * h / 285, w / 2))


def test_resolve_marks_passes_the_atlas_overlay(tmp_path, monkeypatch):
    import asvimg.cpselect as cps
    from asvimg.annotation import resolve_marks

    atlas = _generated_atlas(tmp_path, (512, 512))
    seen = {}

    def fake_gui(*args, **kwargs):
        seen.update(kwargs)
        return None  # cancelled

    monkeypatch.setattr(cps, "cpselect_gui", fake_gui)
    config = SimpleNamespace(annotation="gui", ch_for_annotation=0, channels_name=["g"])
    resolve_marks(config, tmp_path, atlas, np.zeros((8, 8)))

    assert seen["boundary_scale"] == 1.0
    assert seen["ref_landmarks"]["Front Edge"] == pytest.approx((55.0 * 512 / 285, 256.0))


def test_gui_bridge_request_carries_the_atlas_overlay(tmp_path):
    from asvimg.gui.bridge import GuiBridge

    atlas = _generated_atlas(tmp_path, (512, 512))
    bridge = GuiBridge.__new__(GuiBridge)
    captured = {}
    bridge._run_child = lambda module, request: captured.update(request)
    bridge.on_log = lambda *a, **k: None
    config = SimpleNamespace(ch_for_annotation=0, channels_name=["g"])
    bridge._cpselect(config, tmp_path, atlas, np.zeros((8, 8)), None, None)

    assert captured["overlay"]["boundary_scale"] == 1.0
    assert "ref_landmarks" in captured["overlay"]
