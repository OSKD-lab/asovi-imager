"""ICA decomposition and denoising for wide-field calcium imaging."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.decomposition import FastICA

from .pca import PcaResult

if TYPE_CHECKING:
    from .config import PipelineConfig


@dataclass
class IcaResult:
    """Spatial-ICA decomposition result.

    Spatial ICA (Makino et al., Neuron 2017 / Mukamel et al., Neuron 2009):
    ICA is applied to the PCA spatial modes so the components are independent in
    space.  The factorization satisfies, exactly,

        X_centered (rank-k) = ModeICA @ SCORE_ICA

    where ``ModeICA`` are the independent spatial maps and ``SCORE_ICA`` their
    time courses, so a denoised movie is rebuilt by keeping a subset of
    components (see :func:`reconstruct`).

    Attributes
    ----------
    spatial : (n_ic, H, W) — independent spatial maps (ModeICA).
    temporal : (n_ic, T) — component time courses (SCORE_ICA).
    mean_image : (H, W) — per-pixel temporal mean (added back on reconstruction).
    temporal_variance : (n_ic,) — variance of each component's time course
        (components are ordered by this, descending).
    frame_indices : (T,) — frames used for the decomposition.
    shape_hw : (H, W) — spatial dimensions.
    n_components : number of independent components (== retained PCA modes).
    """

    spatial: np.ndarray
    temporal: np.ndarray
    mean_image: np.ndarray
    temporal_variance: np.ndarray
    frame_indices: np.ndarray
    shape_hw: tuple[int, int]
    n_components: int


def compute_ica(
    pca_result: PcaResult,
    n_ica_components: int | None = None,
    *,
    random_state: int = 0,
    max_iter: int = 1000,
) -> IcaResult:
    """Spatial ICA on PCA-reduced data (properly whitened FastICA).

    The PCA is first truncated to ``n_ica_components`` modes (None → all PCA
    modes), so the requested dimensionality is honoured by construction (rather
    than silently ignored).  FastICA is run with ``whiten='unit-variance'`` on
    the spatial modes — i.e. independence is sought over pixels — yielding
    independent spatial maps and their time courses.

    The decomposition preserves ``X_centered(rank-k) = ModeICA @ SCORE_ICA``
    for any invertible unmixing, so :func:`reconstruct` can drop components
    exactly.

    Parameters
    ----------
    pca_result : PcaResult from compute_pca.
    n_ica_components : number of components to keep (None = all PCA modes).
    random_state : seed for FastICA (reproducible).
    max_iter : FastICA iteration cap.
    """
    K_pca = pca_result.spatial.shape[0]
    H, W = pca_result.shape_hw
    P = H * W
    K = K_pca if n_ica_components is None else max(1, min(int(n_ica_components), K_pca))

    # PCA factors (retain top-K): X_c(rank-K) = U @ diag(s) @ Vt
    U = pca_result.spatial[:K].reshape(K, P).T  # (P, K) unit-norm columns (U)
    s = np.asarray(pca_result.singular_values[:K], dtype=np.float64)  # (K,)
    # temporal_raw, NOT temporal: pca_smooth_sigma low-passes the latter for the
    # plots, and feeding that here would (a) break the exact factorization
    # ModeICA @ SCORE_ICA == X_centered(rank-K) and (b) make the IC NUMBERING
    # depend on a display knob — the components come out identical but permuted,
    # so a saved "exclude IC3" would silently point at a different component.
    Vt = np.asarray(pca_result.temporal_raw[:K], dtype=np.float64)  # (K, T)
    score_t = s[:, None] * Vt  # (K, T) = S V^T  (PCA temporal scores)

    # Spatial ICA: unmix the K spatial modes across pixels (FastICA whitens).
    ica = FastICA(
        n_components=K,
        whiten="unit-variance",
        random_state=random_state,
        max_iter=max_iter,
    )
    ica.fit(U)  # samples = pixels (P), features = K modes
    if getattr(ica, "n_iter_", 0) and ica.n_iter_ >= max_iter:
        warnings.warn(
            f"FastICA did not converge in {max_iter} iterations; the recovered "
            f"components may be unreliable (raise ica_max_iter).",
            stacklevel=2,
        )
    B = ica.components_  # (K, K) unmixing (whitening folded in)
    if np.linalg.matrix_rank(B) < K:
        warnings.warn(
            "FastICA unmixing matrix is rank-deficient; the ICA reconstruction "
            "may be inexact.",
            stacklevel=2,
        )

    # Independent spatial maps + their time courses, exact reconstruction basis:
    #   ModeICA = U @ B^T ;  SCORE_ICA = inv(B)^T @ (S V^T)
    #   => ModeICA @ SCORE_ICA = U @ (S V^T) = X_c(rank-K)  [since B^T pinv(B)^T = I]
    # Note: FastICA centers U internally for fitting, but B is applied to the
    # non-centered U here, so a constant DC offset (U.mean(0) @ B^T) is left on
    # each spatial map. This does not affect the exact reconstruction (the offset
    # cancels) nor inter-component decorrelation; the maps are independent in the
    # centered sense.
    mode_ica = U @ B.T  # (P, K)
    score_ica = np.linalg.pinv(B).T @ score_t  # (K, T)

    # Canonicalise sign (peak positive) and order by temporal variance (desc).
    for k in range(K):
        col = mode_ica[:, k]
        if col[np.argmax(np.abs(col))] < 0:
            mode_ica[:, k] *= -1.0
            score_ica[k, :] *= -1.0
    tvar = score_ica.var(axis=1)
    order = np.argsort(tvar)[::-1]
    mode_ica = mode_ica[:, order]
    score_ica = score_ica[order, :]
    tvar = tvar[order]

    return IcaResult(
        spatial=mode_ica.T.reshape(K, H, W),
        temporal=score_ica,
        mean_image=pca_result.mean_image,
        temporal_variance=tvar,
        frame_indices=pca_result.frame_indices,
        shape_hw=(H, W),
        n_components=K,
    )


def reconstruct(
    ica_result: IcaResult,
    image_df: np.ndarray | None = None,
    exclude: list[int] | None = None,
) -> np.ndarray:
    """Rebuild a denoised (H, W, T) movie, dropping the excluded components.

    ``denoised = sum_{k not excluded} ModeICA_k ⊗ SCORE_ICA_k + mean``.

    QC / visualisation only — this is NOT how denoising is applied to the data.
    With ``exclude=[]`` it returns the **rank-k reconstruction**, not the input:
    everything outside the top-k subspace is discarded, so the "no-op" case still
    moves every pixel.  It is also defined only on the frames the decomposition
    was fitted to (``pca_skip_frames``-strided), and raises on a full-length
    stack.  To apply an exclusion to the data, use :func:`apply_ica_denoise`,
    which subtracts only the flagged components from the full timeline.

    Parameters
    ----------
    ica_result : IcaResult from compute_ica.
    image_df : optional (H, W, T); validated against the decomposition's T.
    exclude : IC indices to drop (0-based).
    """
    exclude = list(exclude or [])
    H, W = ica_result.shape_hw
    P = H * W
    K = ica_result.spatial.shape[0]
    T = ica_result.temporal.shape[1]

    if image_df is not None and image_df.shape[2] != T:
        raise ValueError(
            f"image_df frame count ({image_df.shape[2]}) does not match the "
            f"decomposition's T ({T})."
        )

    excl = set(exclude)
    keep = [k for k in range(K) if k not in excl]
    if not keep:
        return np.broadcast_to(
            ica_result.mean_image[:, :, np.newaxis], (H, W, T)
        ).copy()

    mode = ica_result.spatial[keep].reshape(len(keep), P).T  # (P, n_keep)
    score = ica_result.temporal[keep]  # (n_keep, T)
    recon = mode @ score + ica_result.mean_image.reshape(P, 1)  # (P, T)
    return recon.reshape(H, W, T)


# --------------------------------------------------------------------------
# Applying an exclusion to the data (as opposed to picturing it)
# --------------------------------------------------------------------------


def _chunk_frames(n_pixels: int, n_frames: int, budget_bytes: int = 256_000_000) -> int:
    """Frames per pass so the float64 working set stays under ``budget_bytes``."""
    per_frame = max(1, n_pixels * 8 * 3)  # block + centered + output
    return int(np.clip(budget_bytes // per_frame, 1, max(1, n_frames)))


def full_timeline_mean(read_block, n_frames: int, hw: tuple[int, int], *,
                       chunk: int | None = None) -> np.ndarray:
    """(H, W) float64 per-pixel mean over the FULL timeline, streamed.

    Not ``IcaResult.mean_image``: that one is the mean of the ``pca_skip_frames``
    subsample the basis was fitted on.  Centering the full stack on it would give
    the excluded components a non-zero mean loading, so subtracting them would
    shift every pixel by a time-invariant image — a silent DC offset in dF/F that
    ROI traces plot and correlation is blind to.  Centering on the full-timeline
    mean makes the excluded loadings mean-zero, so the denoised stack keeps the
    input's per-pixel mean exactly.
    """
    h, w = int(hw[0]), int(hw[1])
    n_frames = int(n_frames)
    step = chunk or _chunk_frames(h * w, n_frames)
    acc = np.zeros(h * w, dtype=np.float64)
    for t0 in range(0, n_frames, step):
        t1 = min(t0 + step, n_frames)
        acc += np.asarray(read_block(t0, t1), dtype=np.float64).reshape(t1 - t0, -1).sum(0)
    return (acc / max(1, n_frames)).reshape(h, w)


def ic_scores(spatial: np.ndarray, mean_image: np.ndarray, read_block,
              n_frames: int, *, chunk: int | None = None) -> np.ndarray:
    """(K, T) loadings of every IC at EVERY frame — ``pinv(ModeICA) @ (X - mean)``.

    The ICA basis is purely SPATIAL, which is what makes this legitimate: a basis
    fitted on a strided subsample can be evaluated at frames it never saw.
    """
    k = spatial.shape[0]
    mode = spatial.reshape(k, -1).T.astype(np.float64)  # (P, K)
    p = mode.shape[0]
    n_frames = int(n_frames)
    # pinv through the (K, K) Gram — a K x K solve, not an SVD of a (P, K) matrix.
    pinv_mode = np.linalg.pinv(mode.T @ mode) @ mode.T  # (K, P)
    mu = np.asarray(mean_image, dtype=np.float64).reshape(p)

    out = np.empty((k, n_frames), dtype=np.float64)
    step = chunk or _chunk_frames(p, n_frames)
    for t0 in range(0, n_frames, step):
        t1 = min(t0 + step, n_frames)
        block = np.asarray(read_block(t0, t1), dtype=np.float64).reshape(t1 - t0, p)
        out[:, t0:t1] = pinv_mode @ (block - mu).T
    return out


def apply_ica_denoise(
    spatial: np.ndarray,
    read_block,
    n_frames: int,
    hw: tuple[int, int],
    out_path,
    *,
    exclude: list[int],
    chunk: int | None = None,
    progress=None,
) -> "Path":
    """Subtract the excluded ICs from the FULL-LENGTH stack; write (T, H, W) float32.

    ``X_denoised(t) = X(t) - ModeICA[:, excluded] @ a_excluded(t)``, with the
    loadings ``a`` recomputed at every frame (:func:`ic_scores`).

    This is what "apply the denoising" means here, and it is deliberately NOT
    :func:`reconstruct`:

    * ``exclude=[]`` is the **exact identity** — turning the feature on with
      nothing flagged cannot change a single value.  ``reconstruct`` would
      instead hand back a rank-k approximation of the whole movie.
    * everything the flagged components do not explain — including the residual
      the top-k subspace never captured — is **kept**.
    * it is defined on the **full timeline**, not just the frames the basis was
      fitted to.

    ``read_block(t0, t1) -> (k, H, W)`` supplies the source; the stack is read
    twice (mean, then subtract) and never held whole.
    """
    from .io import open_reg_memmap

    h, w = int(hw[0]), int(hw[1])
    p = h * w
    n_frames = int(n_frames)
    k = spatial.shape[0]
    excl = sorted({int(i) for i in (exclude or [])})
    if any(i < 0 or i >= k for i in excl):
        raise ValueError(f"exclude {excl} out of range for {k} components")

    step = chunk or _chunk_frames(p, n_frames)
    mean_full = full_timeline_mean(read_block, n_frames, (h, w), chunk=step)
    # A single non-finite sample would spread through the whole output: the mean
    # carries it into mu, the loadings are a dense (K,P)x(P,k) matmul, and every
    # pixel of every frame comes out NaN — which would then be SAVED and blessed
    # as current. Refuse instead.
    if not np.isfinite(mean_full).all():
        bad = int((~np.isfinite(mean_full)).sum())
        raise ValueError(
            f"the dF/F to denoise has non-finite values in {bad} pixel(s); "
            f"denoising them would poison every pixel of every frame. Fix the "
            f"source (a zero baseline gives inf/NaN) before enabling ica_denoise."
        )

    mm = open_reg_memmap(out_path, n_frames, (h, w), dtype=np.float32)
    try:
        if not excl:  # identity — copy through, still streamed
            for t0 in range(0, n_frames, step):
                t1 = min(t0 + step, n_frames)
                mm[t0:t1] = np.asarray(read_block(t0, t1), dtype=np.float32)
                if progress is not None:
                    progress(t1, n_frames)
            mm.flush()
            return Path(out_path)

        mode = spatial.reshape(k, -1).T.astype(np.float64)      # (P, K)
        pinv_mode = np.linalg.pinv(mode.T @ mode) @ mode.T      # (K, P)
        mu = mean_full.reshape(p)
        mode_e = mode[:, excl]                                  # (P, n_excl)
        pinv_e = pinv_mode[excl]                                # (n_excl, P)

        for t0 in range(0, n_frames, step):
            t1 = min(t0 + step, n_frames)
            # np.array, NOT np.asarray: a reader that hands back a float64 VIEW of
            # its source (the in-RAM dF/F path does exactly that) would otherwise
            # be modified in place by the subtraction below — silently denoising
            # the caller's array.
            block = np.array(read_block(t0, t1), dtype=np.float64).reshape(t1 - t0, p)
            a_e = pinv_e @ (block - mu).T                       # (n_excl, k_frames)
            block -= (mode_e @ a_e).T
            mm[t0:t1] = block.reshape(t1 - t0, h, w).astype(np.float32)
            if progress is not None:
                progress(t1, n_frames)
        mm.flush()
    finally:
        handle = getattr(mm, "_mmap", None)
        if handle is not None:
            handle.close()  # Windows: release the mapping
        del mm
    return Path(out_path)


# --------------------------------------------------------------------------
# Config wrapper + interactive denoising
# --------------------------------------------------------------------------


def compute_ica_from_config(
    pca_result: PcaResult,
    config: "PipelineConfig",
) -> IcaResult:
    """Run ``compute_ica`` with ICA parameters from a PipelineConfig."""
    return compute_ica(
        pca_result,
        n_ica_components=config.ica_n_components,
        random_state=config.ica_random_state,
        max_iter=config.ica_max_iter,
    )


def denoise_interactive(
    image_df: np.ndarray,
    pca_result: PcaResult,
    config: "PipelineConfig",
    *,
    show_diff: bool = True,
) -> tuple[np.ndarray, list[int]]:
    """Compute ICA, open the selection GUI, and reconstruct without the
    excluded ICs.

    - GUI cancel or no components excluded → the original ``image_df`` is
      returned unchanged (and ``excluded`` is ``[]``).
    - When ``show_diff`` is true, a middle-frame before/after comparison
      figure is displayed via ``plt.show()``.

    Returns
    -------
    (denoised_image_df, excluded_ic_indices)
    """
    from .ica_gui import ica_select_gui

    ica_result = compute_ica_from_config(pca_result, config)
    print(f"ICA: {ica_result.spatial.shape[0]} components")
    print(f"  Spatial: {ica_result.spatial.shape}")
    print(f"  Temporal: {ica_result.temporal.shape}")

    selected = ica_select_gui(ica_result)
    if selected is None:
        print("GUI cancelled — no denoising applied.")
        return image_df, []

    excluded = list(selected)
    if not excluded:
        print("No components excluded — using original data.")
        return image_df, []

    print(f"Excluded ICs: {excluded}")
    denoised = reconstruct(ica_result, image_df, exclude=excluded)
    print(f"Denoised: {denoised.shape}")

    if show_diff:
        import matplotlib.pyplot as plt

        frame_idx = image_df.shape[2] // 2
        vmin, vmax = np.percentile(image_df[:, :, frame_idx], [2, 98])
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(
            image_df[:, :, frame_idx], cmap="magma", vmin=vmin, vmax=vmax
        )
        axes[0].set_title("Original dF/F")
        axes[0].axis("off")
        axes[1].imshow(
            denoised[:, :, frame_idx], cmap="magma", vmin=vmin, vmax=vmax
        )
        axes[1].set_title(f"Denoised (excluded: {excluded})")
        axes[1].axis("off")
        fig.suptitle("ICA Denoising: Before vs After", fontsize=14)
        fig.tight_layout()
        plt.show()

    return denoised, excluded


def save_ica_source_images(
    ica_result: "IcaResult",
    out_dir,
) -> tuple[list[np.ndarray], list[str]]:
    """Save ICA spatial maps as RGB PNGs and return cpselect-ready lists.

    Mirrors :func:`asvimg.save_pca_source_images` for the independent spatial
    components: each map is rendered with a symmetric ``RdBu_r`` colormap and
    written to ``out_dir/IC{i}.png``.  The returned ``(rgb_list, labels)`` can
    be handed to the annotation UI so the user can pick control points on the
    ICA maps as well as the raw frame.
    """
    from pathlib import Path

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_list: list[np.ndarray] = []
    labels: list[str] = []
    n = ica_result.spatial.shape[0]
    for i in range(n):
        spatial = ica_result.spatial[i]
        vmax = float(np.percentile(np.abs(spatial), 99)) or 1.0  # robust to outlier pixels
        norm = Normalize(vmin=-vmax, vmax=vmax)
        sm = ScalarMappable(norm=norm, cmap="RdBu_r")
        rgb = (sm.to_rgba(spatial)[:, :, :3] * 255).astype(np.uint8)
        Image.fromarray(rgb).save(out_dir / f"IC{i + 1}.png")
        rgb_list.append(rgb)
        labels.append(f"IC{i + 1}")

    return rgb_list, labels
