"""No atlas ships in the wheel; an empty config means the per-user one.

The atlas is derived from the Allen Mouse Brain Common Coordinate Framework,
whose terms are not the MIT terms this code carries, so it is built on the
machine that uses it (``asovi-atlas --download``) rather than redistributed
inside the package. These tests pin that contract from both ends: nothing is
under ``asvimg/``, and a missing atlas says how to get one instead of failing
with a bare HDF5 error.

The geometry assertions still run against whatever atlas this machine has --
the MATLAB-era file in a lab checkout, or a generated one -- via ``conftest``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from asvimg.atlas import ACCFv3
from asvimg.config import USER_ATLAS, PipelineConfig, resolve_atlas_path
from conftest import TEST_ATLAS, SKIP_REASON


def test_no_atlas_is_packaged():
    """The wheel carries code and presets, not licensed reference data."""
    import asvimg

    pkg = Path(asvimg.__file__).resolve().parent
    stray = [p for p in pkg.rglob("*") if p.suffix.lower() in (".mat", ".h5", ".nrrd")]
    assert stray == [], f"reference data inside the package: {stray}"


def test_empty_config_resolves_to_the_user_atlas():
    assert PipelineConfig().annotation_atlas_path == ""
    assert resolve_atlas_path("") == USER_ATLAS
    assert resolve_atlas_path(None) == USER_ATLAS


def test_the_default_is_not_inside_the_package():
    import asvimg

    pkg = Path(asvimg.__file__).resolve().parent
    assert not USER_ATLAS.is_relative_to(pkg)


def test_an_explicit_path_is_left_alone():
    assert resolve_atlas_path("somewhere/else.h5") == Path("somewhere/else.h5")


def test_a_missing_atlas_says_how_to_build_one(tmp_path):
    """The first-run state is "not built yet", not "broken install"."""
    with pytest.raises(FileNotFoundError) as exc:
        ACCFv3.from_mat(tmp_path / "nothing-here.h5")
    msg = str(exc.value)
    assert "asovi-atlas --download" in msg
    assert "figshare" in msg


def test_resolving_a_path_never_touches_the_disk():
    """Stage signatures call this; it must work with no atlas present."""
    assert resolve_atlas_path("") == USER_ATLAS  # regardless of existence


@pytest.mark.skipif(TEST_ATLAS is None, reason=SKIP_REASON)
class TestWhicheverAtlasThisMachineHas:
    @pytest.fixture(scope="class")
    def atlas(self):
        return ACCFv3.from_mat(TEST_ATLAS)

    def test_point_rois_cover_both_hemispheres(self, atlas):
        """The L/R split ROI extraction actually uses: 15 mirrored pairs from the
        hardcoded coordinates in atlas.py, not from the file."""
        names = list(atlas.point_roi_names)
        right = [n for n in names if n.endswith("_R")]
        left = [n for n in names if n.endswith("_L")]
        assert len(right) == len(left) == 15
        assert len(names) == 30
        w = atlas.shape_hw[1]
        by_name = {n: p for n, p in zip(names, atlas.point_rois)}
        for r in right:
            mirror = by_name[r[:-2] + "_L"]
            assert mirror[0] == by_name[r][0]
            assert mirror[1] == (w - 1) - by_name[r][1]

    def test_geometry_works(self, atlas):
        assert atlas.shape_hw == (285, 285)
        assert atlas.image_rgb.shape[:2] == (285, 285)
        assert len(atlas.boundaries) > 1
        assert atlas.fixed_points.shape[1] == 2

    def test_names_are_answered_or_refused_but_never_guessed(self, atlas):
        """A generated atlas carries its own names and answers. The MATLAB one's
        names are scrambled, so it raises rather than return a wrong mask --
        which is the whole reason the generated one is now the default."""
        if atlas.region_names_verified:
            assert atlas.region_names
            assert atlas.get_mask(next(iter(atlas.region_names.values()))).any()
        else:
            with pytest.raises(ValueError, match="asovi-atlas"):
                atlas.region_names
            with pytest.raises(ValueError, match="asovi-atlas"):
                atlas.get_mask("VISp")
