"""Image-content demux-slip QC.

The intensity demux QC (:mod:`demux_qc`) reads only per-frame *mean* brightness,
so it goes blind (or false-positive) when the two strobed channels have nearly
equal brightness but different image content -- e.g. FT-053, whose source/donner
means differ <0.2% yet whose vasculature patterns are distinct, and whose slowly
drifting mean fooled the intensity QC into flagging a slip that was not there.

This QC instead embeds the head frames of every input file in 2D by image
CONTENT and checks that each file's frames land in the channel cluster their
demux assignment predicts. A phase slip (a file, or its tail, cycling on the
wrong channel) shows up as that file's assigned-source frames sitting in the
donner cluster.

Pure/testable; the frame reading + figure emission live in
``runner.PipelineSession.qc_frame_slip`` (mirroring ``preview_input_all``)."""
from __future__ import annotations

import numpy as np


def normalize_frames(X: np.ndarray) -> np.ndarray:
    """``(n, d)`` raw flattened frames -> content-normalised, channel-axis-amplified.

    Two confounds swamp the channel signal and must go first:

    * **per-frame brightness / drift** -- z-score each frame (subtract its spatial
      mean, divide by its std). This is what makes the QC robust to the slow
      differential-bleaching drift that defeats the intensity QC.
    * **the static FOV** -- subtract the per-pixel mean across all frames. That
      mean is a ~50/50 blend of the source-mean and donner-mean images, so
      removing it leaves ``+delta`` on source frames and ``-delta`` on donner
      frames: it *amplifies* exactly the channel-discriminative axis (after this,
      PC1 ~= the channel axis).
    """
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=1, keepdims=True)              # per-frame: kill brightness/drift
    s = X.std(axis=1, keepdims=True)
    X = X / np.where(s > 0, s, 1.0)                    # per-frame: kill contrast
    X = X - X.mean(axis=0, keepdims=True)              # per-pixel: kill static FOV -> amplify channel
    return X


def embed_2d(X: np.ndarray, method: str = "pca", *, pca_pre: int = 50,
             random_state: int = 0) -> tuple[np.ndarray, str]:
    """Content-normalise ``(n, d)`` frames and project to ``(n, 2)``.

    ``method``: ``"pca"`` (default -- instant, deterministic; after the
    mean-image subtraction the channel split is essentially linear so PC1/PC2
    separate it), ``"tsne"`` (scikit-learn), ``"umap"`` (only if ``umap-learn``
    is importable), or ``"auto"`` (umap if available, else pca). Returns the
    embedding and the method actually used.
    """
    from sklearn.decomposition import PCA

    Xn = normalize_frames(X)
    n, d = Xn.shape
    k = int(min(pca_pre, n - 1, d))
    scores = (PCA(n_components=max(2, k), random_state=random_state).fit_transform(Xn)
              if k >= 2 else Xn)
    method = (method or "pca").lower()

    if method in ("auto", "umap"):
        try:
            import umap  # type: ignore  # noqa: F401

            emb = umap.UMAP(n_components=2, random_state=random_state).fit_transform(scores)
            return np.asarray(emb), "umap"
        except Exception:
            if method == "umap":
                pass  # umap requested but unavailable -> fall back to pca below

    if method == "tsne":
        from sklearn.manifold import TSNE

        perp = float(min(30, max(5, n // 4)))
        emb = TSNE(n_components=2, random_state=random_state, perplexity=perp,
                   init="pca").fit_transform(scores)
        return np.asarray(emb), "tsne"

    return np.asarray(scores[:, :2]), "pca"


def per_file_consistency(emb: np.ndarray, channel_labels, file_idx, *,
                         random_state: int = 0, thresh: float = 0.8):
    """k-means the embedding (k = #channels) and, per file, check each channel's
    frames sit in the cluster that channel occupies globally.

    A phase-consistent file puts each channel almost entirely in its home
    cluster; a slipped file mixes (its assigned-source frames land in the donner
    cluster). Returns ``(lines, flagged)`` -- human-readable per-file lines and
    the set of file indices that look inconsistent (a channel < ``thresh`` in its
    home cluster). ``flagged`` empty == no slip found.
    """
    emb = np.asarray(emb, dtype=float)
    ch = np.asarray(channel_labels)
    fi = np.asarray(file_idx)
    chans = sorted(set(ch.tolist()))
    if len(chans) < 2 or len(emb) < 2 * len(chans):
        return [f"[inconclusive] need >=2 channels and enough frames "
                f"(channels={chans}, n={len(emb)})"], set()

    from sklearn.cluster import KMeans

    k = len(chans)
    lab = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(emb)
    # each channel's home cluster = the one holding most of its frames, globally
    home = {c: int(np.bincount(lab[ch == c], minlength=k).argmax()) for c in chans}
    if len(set(home.values())) < len(chans):
        return [f"[inconclusive] channels not separable in embedding "
                f"(homes={home}) -- content too similar to QC"], set()

    lines, flagged = [], set()
    for f in sorted(set(fi.tolist())):
        sel = fi == f
        parts, ok = [], True
        for c in chans:
            m = sel & (ch == c)
            if not m.any():
                continue
            counts = np.bincount(lab[m], minlength=k)
            in_home = counts[home[c]] / counts.sum()
            parts.append(f"{c}->cl{int(counts.argmax())} {int(counts.max())}/{int(counts.sum())}")
            if in_home < thresh:
                ok = False
        lines.append(f"file {int(f)}: " + "  ".join(parts) + ("" if ok else "   <== SLIP?"))
        if not ok:
            flagged.add(int(f))
    return lines, flagged


def plot_slip_scatter(emb: np.ndarray, channel_labels, file_idx, file_names,
                      method_used: str, flagged=None):
    """Two-panel scatter of the shared 2D embedding: (left) coloured by the
    demux-ASSIGNED channel -- clean = two separated colours, slip = colours
    intermix; (right) coloured by file index -- tells you *which* file's points
    sit in the wrong cluster. Returns a matplotlib Figure."""
    import matplotlib.pyplot as plt

    emb = np.asarray(emb, dtype=float)
    ch = np.asarray(channel_labels)
    fi = np.asarray(file_idx)
    flagged = set(flagged or ())
    chans = sorted(set(ch.tolist()))
    # source=green (GCaMP), donner=violet (isosbestic) to match the physical
    # channels; any other label falls back to the spare palette.
    known = {"source": "#2ca02c", "donner": "#7e2f8e"}
    spare = ["#ff7f0e", "#1f77b4", "#8c564b", "#e377c2"]
    colors, si = {}, 0
    for c in chans:
        if c in known:
            colors[c] = known[c]
        else:
            colors[c] = spare[si % len(spare)]; si += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    for c in chans:
        m = ch == c
        ax1.scatter(emb[m, 0], emb[m, 1], s=10, c=colors[c], alpha=0.7, label=str(c))
    ax1.set_title("coloured by assigned channel\n(clean = separated, slip = intermixed)")
    ax1.legend(fontsize=8, markerscale=1.5)
    ax1.set_xlabel("dim 1"); ax1.set_ylabel("dim 2")

    sc = ax2.scatter(emb[:, 0], emb[:, 1], s=10, c=fi, cmap="viridis", alpha=0.8)
    fig.colorbar(sc, ax=ax2, label="file index")
    # ring the flagged files so the eye lands on them
    if flagged:
        m = np.isin(fi, list(flagged))
        ax2.scatter(emb[m, 0], emb[m, 1], s=60, facecolors="none",
                    edgecolors="red", linewidths=1.2,
                    label=f"flagged files {sorted(flagged)}")
        ax2.legend(fontsize=8)
    ax2.set_title("coloured by file  (which file is in the wrong cluster?)")
    ax2.set_xlabel("dim 1"); ax2.set_ylabel("dim 2")

    n_files = len(set(fi.tolist()))
    fig.suptitle(f"frame-content demux-slip QC  ({method_used}, {len(emb)} frames, "
                 f"{n_files} files)"
                 + (f"  --  POSSIBLE SLIP: files {sorted(flagged)}" if flagged else "  --  no slip"))
    fig.tight_layout()
    return fig
