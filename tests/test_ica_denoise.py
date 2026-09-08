"""The applied ICA denoising: subtract the flagged components from the data.

The properties that make this safe to point at a scientific output — and that
``reconstruct`` (the QC path) does NOT have — are pinned here.
"""

import numpy as np
import pytest

from asvimg.ica import (
    apply_ica_denoise,
    compute_ica,
    ic_scores,
    reconstruct,
)
from asvimg.pca import compute_pca

H, W, T, SKIP = 16, 20, 300, 5


def _data(seed=0):
    """dF/F-like: 2 'neural' modes + 1 artifact on a stripe + residual noise.

    The artifact's time course is slower than the strided fit's Nyquist — an
    artifact that only lives above it is invisible to a basis fitted on 1-in-N
    frames, and no amount of excluding can remove it. That is a real limitation
    of fitting on a subsample, not a property of the applicator.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    neural = [(rng.normal(size=(H, W)), np.sin(t / 23)),
              (rng.normal(size=(H, W)), np.sin(t / 31 + 1))]
    art_map = np.zeros((H, W))
    art_map[:, 3:6] = 1.0
    art_course = 3.0 * np.sin(t / 11)                   # stripe-shaped, slow enough
    X = sum(m[:, :, None] * c[None, None, :] for m, c in neural)
    X = X + art_map[:, :, None] * art_course[None, None, :]
    X = X + rng.normal(0, 0.3, size=(H, W, T))          # residual: in NO component
    return X, neural, (art_map, art_course)


def _basis(X):
    """Fit the way the pipeline does: on a pca_skip_frames-strided subsample."""
    pca = compute_pca(X[:, :, ::SKIP], n_components=5, smooth_sigma=5.0)
    return compute_ica(pca, n_ica_components=5, random_state=0, max_iter=5000)


def _reader(X):
    return lambda t0, t1: np.moveaxis(X[..., t0:t1], 2, 0)   # (k, H, W)


def _amp(stack, m, c):
    """Amplitude of (spatial map m x time course c) left in the stack."""
    v = (stack.reshape(H * W, -1) * m.ravel()[:, None]).sum(0)
    return float(np.dot(v - v.mean(), c - c.mean()) / (np.linalg.norm(c - c.mean()) ** 2))


def test_exclude_nothing_is_the_exact_identity(tmp_path):
    """The safety property. Turning denoising on with nothing flagged must not
    move a single value — reconstruct() fails this by ~3 sigma."""
    X, _, _ = _data()
    ica = _basis(X)
    out = apply_ica_denoise(ica.spatial, _reader(X), T, (H, W),
                            tmp_path / "d.npy", exclude=[])
    got = np.moveaxis(np.load(out), 0, 2)
    np.testing.assert_allclose(got, X.astype(np.float32), rtol=0, atol=0)

    # what the QC path would have done instead, on the frames it is defined on
    rank_k = reconstruct(ica, X[:, :, ::SKIP], exclude=[])
    assert np.abs(rank_k - X[:, :, ::SKIP]).max() > 0.5 * X.std()


def test_output_is_full_length_though_the_fit_was_strided(tmp_path):
    X, _, _ = _data()
    ica = _basis(X)
    assert ica.temporal.shape[1] == len(range(0, T, SKIP))     # the fit
    out = apply_ica_denoise(ica.spatial, _reader(X), T, (H, W),
                            tmp_path / "d.npy", exclude=[0])
    assert np.load(out).shape == (T, H, W)                     # the application


def test_removes_the_flagged_component_and_keeps_the_rest(tmp_path):
    X, neural, (art_map, art_course) = _data()
    ica = _basis(X)
    maps = ica.spatial.reshape(ica.n_components, -1)
    art_ic = int(np.argmax([abs(np.corrcoef(m, art_map.ravel())[0, 1]) for m in maps]))

    out = apply_ica_denoise(ica.spatial, _reader(X), T, (H, W),
                            tmp_path / "d.npy", exclude=[art_ic])
    den = np.moveaxis(np.load(out), 0, 2).astype(np.float64)

    before = _amp(X, art_map, art_course)
    after = _amp(den, art_map, art_course)
    assert abs(after) < 0.1 * abs(before)                       # artifact gone

    for m, c in neural:                                         # signal kept
        assert abs(_amp(den, m, c) / _amp(X, m, c) - 1.0) < 0.05

    # the residual noise (in no component at all) survives — a rank-k rebuild
    # would have thrown it away
    resid = den - sum(m[:, :, None] * c[None, None, :] for m, c in neural)
    assert resid.std() > 0.2


def test_per_pixel_mean_is_preserved(tmp_path):
    """Centering on the FULL-timeline mean (not the subsample's) is what keeps the
    denoised stack from picking up a time-invariant DC offset."""
    X, _, _ = _data()
    ica = _basis(X)
    out = apply_ica_denoise(ica.spatial, _reader(X), T, (H, W),
                            tmp_path / "d.npy", exclude=[1])
    den = np.moveaxis(np.load(out), 0, 2)
    np.testing.assert_allclose(den.mean(axis=2), X.mean(axis=2), rtol=0, atol=1e-5)


def test_chunking_does_not_change_the_result(tmp_path):
    X, _, _ = _data()
    ica = _basis(X)
    ref = np.load(apply_ica_denoise(ica.spatial, _reader(X), T, (H, W),
                                    tmp_path / "a.npy", exclude=[0, 2], chunk=T))
    for chunk in (1, 7, 64):
        got = np.load(apply_ica_denoise(ica.spatial, _reader(X), T, (H, W),
                                        tmp_path / f"b{chunk}.npy",
                                        exclude=[0, 2], chunk=chunk))
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-5)


def test_scores_match_the_decomposition_on_the_fitted_frames():
    """The full-timeline loadings must agree with the ICA's own scores where the
    two are defined on the same frames (up to the mean they are centered on)."""
    X, _, _ = _data()
    ica = _basis(X)
    sub = X[:, :, ::SKIP]
    got = ic_scores(ica.spatial, ica.mean_image, _reader(sub), sub.shape[2])
    np.testing.assert_allclose(got, ica.temporal, rtol=0, atol=1e-8)


def test_rejects_out_of_range_components(tmp_path):
    X, _, _ = _data()
    ica = _basis(X)
    with pytest.raises(ValueError):
        apply_ica_denoise(ica.spatial, _reader(X), T, (H, W),
                          tmp_path / "d.npy", exclude=[99])


def test_does_not_modify_the_source(tmp_path):
    """A reader that hands back a float64 VIEW of its source (the in-RAM dF/F
    path does) must not have its array denoised out from under it."""
    X, _, _ = _data()
    ica = _basis(X)
    before = X.copy()
    apply_ica_denoise(ica.spatial, _reader(X), T, (H, W),
                      tmp_path / "d.npy", exclude=[0, 1])
    np.testing.assert_array_equal(X, before)


def test_non_finite_input_is_refused(tmp_path):
    """One NaN sample would spread through the mean into the loadings and come out
    as a 100%-NaN movie — which would then be SAVED and blessed as current."""
    X, _, _ = _data()
    ica = _basis(X)
    X[3, 4, 11] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        apply_ica_denoise(ica.spatial, _reader(X), T, (H, W),
                          tmp_path / "d.npy", exclude=[0])
