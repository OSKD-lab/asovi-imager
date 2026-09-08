"""Tests for the image-content demux-slip QC (asvimg.qc_frameslip).

Synthetic head-frames: two distinct content patterns (source/donner), a per-file
strobe that alternates them, plus a brightness drift confound. The clean case
must flag nothing; a file whose channel *assignment* is swapped (a phase slip)
must be the one flagged -- the regression this QC exists to catch."""
from __future__ import annotations

import numpy as np

from asvimg.qc_frameslip import (
    embed_2d,
    normalize_frames,
    per_file_consistency,
    plot_slip_scatter,
)


def _synthetic(n_files=6, n_per=40, d=64, slip_files=(), seed=0):
    """Head frames of ``n_files`` files. True content alternates source/donner by
    parity; ``slip_files`` have their assigned label swapped vs the true content
    (i.e. the demux is off by one there). A per-frame brightness drift is added
    to make sure the QC survives it. Returns (X, channel_labels, file_idx)."""
    rng = np.random.default_rng(seed)
    pat_src = rng.standard_normal(d)
    pat_don = rng.standard_normal(d)
    X, ch, fi = [], [], []
    for f in range(n_files):
        for j in range(n_per):
            true_src = (j % 2 == 0)
            base = pat_src if true_src else pat_don
            drift = 100.0 + 0.5 * (f * n_per + j)          # slow brightness drift
            img = base * 6.0 + rng.standard_normal(d) * 0.8 + drift
            X.append(img)
            assigned_src = true_src if f not in slip_files else (not true_src)
            ch.append("source" if assigned_src else "donner")
            fi.append(f)
    return np.asarray(X), ch, fi


def test_normalize_removes_brightness_and_amplifies_channel():
    X, ch, fi = _synthetic()
    Xn = normalize_frames(X)
    # per-frame mean ~0 after z-score; per-pixel mean ~0 after image subtraction
    assert np.allclose(Xn.mean(axis=1), 0, atol=1e-9)
    assert np.allclose(Xn.mean(axis=0), 0, atol=1e-9)
    # the two channels now separate along PC1
    src = np.array([c == "source" for c in ch])
    emb, used = embed_2d(X, "pca")
    assert used == "pca"
    sep = abs(emb[src, 0].mean() - emb[~src, 0].mean())
    spread = emb[:, 0].std()
    assert sep > spread  # channel axis dominates PC1


def test_clean_recording_flags_nothing():
    X, ch, fi = _synthetic()
    emb, _ = embed_2d(X, "pca")
    lines, flagged = per_file_consistency(emb, ch, fi)
    assert flagged == set(), lines


def test_injected_single_file_slip_is_caught():
    X, ch, fi = _synthetic(slip_files=(3,))
    emb, _ = embed_2d(X, "pca")
    lines, flagged = per_file_consistency(emb, ch, fi)
    assert flagged == {3}, lines


def test_two_slipped_files_both_caught():
    X, ch, fi = _synthetic(n_files=8, slip_files=(2, 5))
    emb, _ = embed_2d(X, "pca")
    lines, flagged = per_file_consistency(emb, ch, fi)
    assert flagged == {2, 5}, lines


def test_inconclusive_when_channels_identical():
    # both channels share one content pattern -> not separable -> no false slip
    rng = np.random.default_rng(1)
    pat = rng.standard_normal(64)
    X, ch, fi = [], [], []
    for f in range(4):
        for j in range(40):
            X.append(pat * 6 + rng.standard_normal(64) * 0.8 + 100)
            ch.append("source" if j % 2 == 0 else "donner")
            fi.append(f)
    emb, _ = embed_2d(np.asarray(X), "pca")
    lines, flagged = per_file_consistency(emb, ch, fi)
    assert flagged == set()
    assert any("inconclusive" in ln for ln in lines)


def test_plot_returns_figure_and_closes():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X, ch, fi = _synthetic(slip_files=(1,))
    emb, used = embed_2d(X, "pca")
    _, flagged = per_file_consistency(emb, ch, fi)
    fig = plot_slip_scatter(emb, ch, fi, [f"f{i}" for i in range(6)], used, flagged)
    assert fig is not None
    assert len(fig.axes) >= 2
    plt.close(fig)
