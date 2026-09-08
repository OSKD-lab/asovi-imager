"""The atlas ships inside the package, and its known-wrong parts refuse to run.

`annotation_atlas_path` used to default to a repository-relative path, so a
pip-installed copy had no atlas at all.  The file now rides along in
``asvimg/data/``; an empty config value means that copy.

What ships is the *legacy* MATLAB atlas.  Its geometry, boundaries, midline and
point ROIs are correct — including the left/right split, which comes from the
hardcoded right-hemisphere coordinates in ``atlas.py`` mirrored across the
midline, not from the file.  Its ID map is bilateral and its region *names* are
wrong, so the name-driven API raises instead of returning a plausible mask.
"""
from __future__ import annotations

import filecmp
from pathlib import Path

import pytest

from asvimg.atlas import ACCFv3
from asvimg.config import BUNDLED_ATLAS, PipelineConfig, resolve_atlas_path

REPO_ROOT = Path(__file__).resolve().parent.parent
MATLAB_COPY = REPO_ROOT / "matlab" / "wfciAnnotationData.mat"


def test_the_bundled_atlas_is_inside_the_package():
    assert BUNDLED_ATLAS.exists()
    # ...and under the package directory, or the wheel would not carry it.
    import asvimg

    assert BUNDLED_ATLAS.is_relative_to(Path(asvimg.__file__).resolve().parent)


def test_empty_config_resolves_to_the_bundled_atlas():
    assert PipelineConfig().annotation_atlas_path == ""
    assert resolve_atlas_path("") == BUNDLED_ATLAS
    assert resolve_atlas_path(None) == BUNDLED_ATLAS


def test_an_explicit_path_is_left_alone():
    assert resolve_atlas_path("somewhere/else.h5") == Path("somewhere/else.h5")


@pytest.mark.skipif(not MATLAB_COPY.exists(), reason="matlab/ is internal-only")
def test_the_packaged_copy_matches_the_matlab_one():
    """Two copies of one static file; this fails the day they drift."""
    assert filecmp.cmp(MATLAB_COPY, BUNDLED_ATLAS, shallow=False)


class TestBundledAtlasContents:
    @pytest.fixture(scope="class")
    def atlas(self):
        return ACCFv3.from_mat(BUNDLED_ATLAS)

    def test_point_rois_cover_both_hemispheres(self, atlas):
        """The L/R split that ROI extraction actually uses."""
        names = list(atlas.point_roi_names)
        right = [n for n in names if n.endswith("_R")]
        left = [n for n in names if n.endswith("_L")]
        assert len(right) == len(left) == 15
        assert len(names) == 30
        # every right ROI has its mirror, at (w-1) - col
        w = atlas.shape_hw[1]
        by_name = {n: p for n, p in zip(names, atlas.point_rois)}
        for r in right:
            mirror = by_name[r[:-2] + "_L"]
            assert mirror[0] == by_name[r][0]
            assert mirror[1] == (w - 1) - by_name[r][1]

    def test_region_names_are_refused(self, atlas):
        """The names are wrong; a wrong mask that logs a warning is still wrong."""
        assert atlas.region_names_verified is False
        with pytest.raises(ValueError, match="asovi-atlas"):
            atlas.region_names
        with pytest.raises(ValueError, match="asovi-atlas"):
            atlas.get_mask("VISp")
        with pytest.raises(ValueError, match="asovi-atlas"):
            atlas.get_mask(6)

    def test_geometry_still_works(self, atlas):
        """Everything the pipeline does use is unaffected."""
        assert atlas.shape_hw == (285, 285)
        assert atlas.image_rgb.shape[:2] == (285, 285)
        assert len(atlas.boundaries) > 1
        assert atlas.fixed_points.shape[1] == 2
