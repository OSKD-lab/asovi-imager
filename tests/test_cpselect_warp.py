"""Test cpselect coordinate system and warp correctness.

Generates test images (atlas original + rotated/scaled version) and
verifies that compute_transform + warp_image correctly recovers the
known transformation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from asvimg.annotation import compute_transform, warp_image
from conftest import TEST_ATLAS_OR_DEFAULT
from asvimg.atlas import load_atlas
from PIL import Image

_HERE = Path(__file__).resolve().parent
ATLAS_PATH = TEST_ATLAS_OR_DEFAULT
TEST_OUTPUT_DIR = _HERE / "_test_outputs"


def _make_test_images():
    """Generate atlas PNG and a rotated+scaled version for testing."""
    atlas = load_atlas(ATLAS_PATH)
    img = atlas.image_rgb  # (285, 285, 3) float64, 0-1

    # Original atlas PNG
    original = (img * 255).clip(0, 255).astype(np.uint8)

    # Known transform: 45 deg rotation, 1.5x scale, centered on image
    angle_deg = 45.0
    scale = 0.75
    angle_rad = np.radians(angle_deg)
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # Build forward transform: maps original (col, row) → transformed (col, row)
    # 1. translate center to origin
    # 2. scale
    # 3. rotate
    # 4. translate back
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Inverse transform (transformed → original) for pixel sampling
    # inv = T_center @ R_inv @ S_inv @ T_neg_center
    def inv_map(out_col, out_row):
        dx = out_col - cx
        dy = out_row - cy
        # inverse rotate
        rx = cos_a * dx + sin_a * dy
        ry = -sin_a * dx + cos_a * dy
        # inverse scale
        rx /= scale
        ry /= scale
        return rx + cx, ry + cy

    # Generate transformed image by sampling
    out_h, out_w = h, w  # same size
    transformed = np.zeros_like(img)
    for r in range(out_h):
        for c in range(out_w):
            src_c, src_r = inv_map(c, r)
            sr, sc = int(round(src_r)), int(round(src_c))
            if 0 <= sr < h and 0 <= sc < w:
                transformed[r, c] = img[sr, sc]

    transformed_u8 = (transformed * 255).clip(0, 255).astype(np.uint8)

    # Save PNGs
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(original).save(TEST_OUTPUT_DIR / "atlas_original.png")
    Image.fromarray(transformed_u8).save(TEST_OUTPUT_DIR / "atlas_rot45_scale1.5.png")

    # Define corresponding point pairs (known ground truth)
    # Points on the original image and their locations in the transformed image
    # Use center and offset points
    test_points_original = np.array(
        [
            [cy - 50, cx],  # upper center
            [cy + 80, cx],  # lower center
            [cy, cx - 60],  # left center
        ]
    )  # (row, col) in original

    # Forward transform these points to get their locations in the transformed image
    test_points_transformed = []
    for row, col in test_points_original:
        dx = col - cx
        dy = row - cy
        # scale then rotate
        sx, sy = dx * scale, dy * scale
        rx = cos_a * sx - sin_a * sy
        ry = sin_a * sx + cos_a * sy
        test_points_transformed.append([ry + cy, rx + cx])
    test_points_transformed = np.array(test_points_transformed)

    return {
        "original": img,
        "transformed": transformed,
        "original_u8": original,
        "transformed_u8": transformed_u8,
        "pts_original": test_points_original,
        "pts_transformed": test_points_transformed,
        "angle_deg": angle_deg,
        "scale": scale,
    }


class TestComputeTransform:
    """Test that compute_transform recovers known transformations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not ATLAS_PATH.exists():
            pytest.skip("Atlas data not available")
        self.data = _make_test_images()

    def test_recovered_scale(self):
        """Scale should match the known 1.5x."""
        tform = compute_transform(
            self.data["pts_transformed"],
            self.data["pts_original"],
        )
        assert abs(tform.scale - 1.0 / self.data["scale"]) < 0.05, (
            f"Expected scale ~{1.0 / self.data['scale']:.4f}, got {tform.scale:.4f}"
        )

    def test_recovered_rotation(self):
        """Rotation should match the known -45 degrees."""
        tform = compute_transform(
            self.data["pts_transformed"],
            self.data["pts_original"],
        )
        rot_deg = np.degrees(tform.rotation)
        expected = -self.data["angle_deg"]
        assert abs(rot_deg - expected) < 2.0, (
            f"Expected rotation ~{expected:.1f} deg, got {rot_deg:.1f} deg"
        )

    def test_point_mapping_accuracy(self):
        """Transformed points should map back to original points."""
        tform = compute_transform(
            self.data["pts_transformed"],
            self.data["pts_original"],
        )
        # Forward: transform maps transformed_pts → original_pts
        src_xy = self.data["pts_transformed"][:, ::-1]  # (col, row)
        mapped_xy = tform(src_xy)
        mapped_rc = mapped_xy[:, ::-1]  # back to (row, col)
        error = np.abs(mapped_rc - self.data["pts_original"]).max()
        assert error < 1.0, f"Max mapping error: {error:.2f} px"


class TestWarpImage:
    """Test that warp_image produces correct output."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not ATLAS_PATH.exists():
            pytest.skip("Atlas data not available")
        self.data = _make_test_images()

    def test_warp_dot_positions(self):
        """Dots placed at transformed positions should warp to original positions."""
        tform = compute_transform(
            self.data["pts_transformed"],
            self.data["pts_original"],
        )

        h, w = self.data["original"].shape[:2]

        # Place dots at transformed point locations
        dot_img = np.zeros((h, w))
        for row, col in self.data["pts_transformed"]:
            r, c = int(round(row)), int(round(col))
            if 0 <= r < h and 0 <= c < w:
                dot_img[r, c] = 1.0

        # Warp should move dots to original positions
        warped = warp_image(dot_img, tform, (h, w))

        for i, (exp_row, exp_col) in enumerate(self.data["pts_original"]):
            er, ec = int(round(exp_row)), int(round(exp_col))
            # Check 5x5 neighborhood for the dot
            region = warped[max(0, er - 2) : er + 3, max(0, ec - 2) : ec + 3]
            assert region.max() > 0.01, (
                f"Dot {i} not found near expected position "
                f"(row={er}, col={ec}), max={region.max():.4f}"
            )

    def test_warp_preserves_shape(self):
        """Output shape should match requested output_shape."""
        tform = compute_transform(
            self.data["pts_transformed"],
            self.data["pts_original"],
        )
        out_shape = (200, 250)
        gray = np.mean(self.data["transformed"], axis=2)
        warped = warp_image(gray, tform, out_shape)
        assert warped.shape == out_shape

    def test_png_saved(self):
        """Test images should be saved as PNG."""
        assert (TEST_OUTPUT_DIR / "atlas_original.png").exists()
        assert (TEST_OUTPUT_DIR / "atlas_rot45_scale1.5.png").exists()


if __name__ == "__main__":
    import sys
    from pathlib import Path as _P

    _src = str(_P(__file__).resolve().parents[1] / "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

    from asvimg.annotation import compute_transform, warp_image  # noqa: F811
    from asvimg.atlas import load_atlas  # noqa: F811

    # Generate test images and run quick check
    data = _make_test_images()
    print(f"Original: {data['original_u8'].shape}")
    print(f"Transformed: {data['transformed_u8'].shape}")
    print(f"Points original (row,col):\n{data['pts_original']}")
    print(f"Points transformed (row,col):\n{data['pts_transformed']}")

    tform = compute_transform(
        data["pts_transformed"],
        data["pts_original"],
    )
    print("\nRecovered transform:")
    print(f"  Scale: {tform.scale:.4f} (expected {1 / data['scale']:.4f})")
    print(
        f"  Rotation: {np.degrees(tform.rotation):.2f} deg (expected {-data['angle_deg']:.1f})"
    )

    # Dot test
    h, w = 285, 285
    dot_img = np.zeros((h, w))
    for row, col in data["pts_transformed"]:
        r, c = int(round(row)), int(round(col))
        if 0 <= r < h and 0 <= c < w:
            dot_img[r, c] = 1.0

    warped = warp_image(dot_img, tform, (h, w))
    print("\nDot test:")
    for i, (er, ec) in enumerate(data["pts_original"]):
        er, ec = int(round(er)), int(round(ec))
        region = warped[max(0, er - 2) : er + 3, max(0, ec - 2) : ec + 3]
        peak = np.unravel_index(warped.argmax(), warped.shape) if i == 0 else None
        print(f"  Dot {i}: expected ({er},{ec}), region max={region.max():.4f}")

    print(
        "\nAll checks passed!"
        if all(
            warped[
                max(0, int(round(r))) - 2 : int(round(r)) + 3,
                max(0, int(round(c))) - 2 : int(round(c)) + 3,
            ].max()
            > 0.01
            for r, c in data["pts_original"]
        )
        else "FAILED"
    )

    # --- GUI test ---
    print("\n" + "=" * 60)
    print("GUI TEST")
    print("=" * 60)
    print("Instructions:")
    print("  Source (left):    rotated+scaled atlas")
    print("  Reference (mid): original atlas")
    print("  Preview (right):  warp result + borders")
    print()
    print("  1. Place 2+ corresponding points on both images")
    print("  2. Check 'Show Borders' to verify alignment")
    print("  3. Click 'Finish' when satisfied")
    print()
    print(
        f"  Expected transform: scale≈{1 / data['scale']:.3f}, rotation≈{-data['angle_deg']:.0f}°"
    )
    print("=" * 60)

    from asvimg.cpselect import cpselect_gui

    atlas = load_atlas(ATLAS_PATH)  # noqa: F811

    ref_landmarks = {
        "Front Edge": (55.0, 142.5),
        "RSC Edge": (212.5, 142.5),
    }

    # Multiple source images: original, transformed, and grayscale
    import numpy as np  # noqa: F811

    gray_u8 = np.mean(data["transformed_u8"].astype(np.float64), axis=2)
    gray_u8 = ((gray_u8 / gray_u8.max()) * 255).astype(np.uint8)

    result = cpselect_gui(
        [data["transformed_u8"], data["original_u8"], gray_u8],
        data["original_u8"],
        n_default_pairs=2,
        boundaries=atlas.boundaries,
        ref_landmarks=ref_landmarks,
        source_labels=["Rotated (RGB)", "Original (RGB)", "Rotated (gray)"],
        title="GUI Test: rot45 + scale1.5",
    )

    if result is None:
        print("\nCancelled.")
    else:
        src_pts, ref_pts = result
        print(f"\nSelected {len(src_pts)} point pairs")
        print(f"  Source pts (row, col):\n{src_pts}")
        print(f"  Ref pts (row, col):\n{ref_pts}")

        tform_gui = compute_transform(src_pts, ref_pts)
        print("\nRecovered from GUI:")
        print(f"  Scale:    {tform_gui.scale:.4f}  (expected {1 / data['scale']:.4f})")
        print(
            f"  Rotation: {np.degrees(tform_gui.rotation):.2f}°  (expected {-data['angle_deg']:.1f}°)"
        )

        scale_ok = abs(tform_gui.scale - 1 / data["scale"]) < 0.1
        rot_ok = abs(np.degrees(tform_gui.rotation) - (-data["angle_deg"])) < 10
        print(f"\n  Scale {'OK' if scale_ok else 'WRONG'}")
        print(f"  Rotation {'OK' if rot_ok else 'WRONG'}")

        if scale_ok and rot_ok:
            print("\n>>> GUI TEST PASSED <<<")
        else:
            print("\n>>> GUI TEST FAILED — coordinate issue detected <<<")
            print("  Check if rotation sign is inverted (DPG Y-axis flip issue)")
