"""PCA on a channel group's dF/F (per ``channels_name``).

The input is always the group's dF/F — linear-subtraction where a donner exists,
percentile-baseline where it does not — read through the one seam every consumer
uses (``load_group_source`` -> ``annotation._load_name_dff_stack``), never raw
fluorescence.  Reshapes (H, W, T) → (pixels, T), runs truncated SVD, and returns
spatial maps + temporal components. Temporal components can optionally be Gaussian
smoothed to suppress high-frequency noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

    from .config import PipelineConfig


@dataclass
class PcaResult:
    """PCA decomposition result.

    Attributes
    ----------
    spatial : (n_components, H, W) – spatial maps (principal images)
    temporal : (n_components, T_used) – temporal weights, Gaussian-smoothed when
        ``smooth_sigma > 0``.  A DISPLAY product: smoothing it makes the plotted
        traces readable, but it no longer projects the data.
    temporal_raw : (n_components, T_used) – the unsmoothed weights.  Anything
        that computes with the decomposition (ICA, reconstruction, the component
        ordering) must use these: ordering ICs by the variance of a *smoothed*
        score makes the IC numbering depend on a display parameter, so a saved
        "exclude IC3" would point at a different component when ``smooth_sigma``
        changes.  (Measured: identical maps, permuted numbering.)
    variance_ratio : (n_components,) – explained variance ratio
    singular_values : (n_components,) – singular values
    mean_image : (H, W) – pixel-wise temporal mean (subtracted before PCA)
    frame_indices : (T_used,) – which frames were used
    shape_hw : (H, W) – original spatial dimensions
    """

    spatial: np.ndarray
    temporal: np.ndarray
    variance_ratio: np.ndarray
    singular_values: np.ndarray
    mean_image: np.ndarray
    frame_indices: np.ndarray
    shape_hw: tuple[int, int]
    temporal_raw: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.temporal_raw is None:  # no smoothing was applied
            self.temporal_raw = self.temporal


def load_image_df(path: Path, data_key: str = "imageDf") -> np.ndarray:
    """Load imageDf from a pipeline01 output file (mat/npy/h5).

    Returns (H, W, T) float64.
    """
    payload = load_payload(path)
    return np.asarray(payload[data_key], dtype=np.float64)


def load_payload(path: Path) -> dict[str, Any]:
    """Load payload from mat/npy/h5."""
    suffix = path.suffix.lower()
    if suffix == ".mat":
        from scipy.io import loadmat

        try:
            raw = loadmat(path)
            return {k: v for k, v in raw.items() if not k.startswith("__")}
        except NotImplementedError:
            pass
        import h5py

        out: dict[str, Any] = {}
        with h5py.File(path, "r") as h5f:
            for key in h5f.keys():
                node = h5f[key]
                if hasattr(node, "shape"):
                    arr = np.asarray(node)
                    if arr.ndim >= 2:
                        arr = arr.T
                    out[key] = arr
        return out

    if suffix == ".npy":
        return np.load(path, allow_pickle=True).item()

    if suffix == ".h5":
        import h5py

        out = {}
        with h5py.File(path, "r") as h5f:
            for key, ds in h5f.items():
                val = ds[()]
                # HDF5 stores text as variable-length UTF-8; hand back str, not
                # bytes, so an h5 payload reads like the mat/npy ones.
                if isinstance(val, np.ndarray) and val.dtype.kind in "SO":
                    val = np.array([
                        v.decode() if isinstance(v, bytes) else str(v)
                        for v in val.ravel().tolist()
                    ])
                elif isinstance(val, bytes):
                    val = val.decode()
                out[key] = val
            for key, value in h5f.attrs.items():
                out[key] = value
        return out

    raise ValueError(f"Unsupported format: {suffix}")


def compute_pca(
    image_df: np.ndarray,
    n_components: int = 10,
    smooth_sigma: float = 0.0,
    frame_stride: int = 1,
) -> PcaResult:
    """Run PCA on (H, W, T) activity data.

    Parameters
    ----------
    image_df : (H, W, T) float64 – linear-subtraction output (already
        temporally strided by ``frame_stride`` if the caller subsampled it)
    n_components : int – number of principal components to keep
    smooth_sigma : float – Gaussian sigma (in frames) for temporal smoothing
        of the resulting temporal components. 0 disables smoothing.
    frame_stride : int – stride the input was subsampled by, used only to map
        ``frame_indices`` back to true global frame numbers for plots.

    Returns
    -------
    PcaResult with spatial maps, temporal components, etc.
    """
    H, W, T = image_df.shape
    frame_indices = np.arange(T) * int(frame_stride)
    T_used = T

    # (H, W, T) → (pixels, T)
    data = image_df.reshape(H * W, T_used)

    # Subtract temporal mean per pixel
    mean_image = data.mean(axis=1)
    data_centered = data - mean_image[:, None]

    # Truncated SVD (more efficient than full eig for n_components << min(pixels, T))
    # T_used - 1, not T_used: subtracting the temporal mean costs one degree of
    # freedom, so the data has rank <= T-1 and the last "component" is the zero
    # eigenvector. Its map is pure round-off (~1e-14) and the normalization below
    # would blow it up to unit length and ship it into the ICA basis — which, now
    # that the basis is SUBTRACTED from the saved dF/F, is not just a bad picture.
    # Reachable with the defaults on any recording short enough that
    # pca_skip_frames * pca_n_components >= T.
    n_components = min(n_components, max(1, T_used - 1), H * W)

    if T_used <= H * W:
        # Compute covariance in temporal domain: (T_used, T_used)
        cov_t = data_centered.T @ data_centered / (H * W - 1)
        eigvals, eigvecs = np.linalg.eigh(cov_t)
        # eigh returns ascending order → flip
        idx = np.argsort(eigvals)[::-1][:n_components]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]  # (T_used, n_components)

        # Temporal components
        temporal = eigvecs.T  # (n_components, T_used)

        # Spatial maps via projection
        spatial = data_centered @ eigvecs  # (pixels, n_components)
        # Normalize spatial maps
        norms = np.linalg.norm(spatial, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        spatial = spatial / norms
        spatial = spatial.T.reshape(n_components, H, W)

        singular_values = np.sqrt(np.maximum(eigvals * (H * W - 1), 0.0))
    else:
        # Compute covariance in spatial domain: (pixels, pixels) — rare case
        cov_s = data_centered @ data_centered.T / (T_used - 1)
        eigvals, eigvecs = np.linalg.eigh(cov_s)
        idx = np.argsort(eigvals)[::-1][:n_components]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]  # (pixels, n_components)

        spatial = eigvecs.T.reshape(n_components, H, W)
        temporal = eigvecs.T @ data_centered  # (n_components, T_used)
        # Normalize temporal
        norms = np.linalg.norm(temporal, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        temporal = temporal / norms

        singular_values = np.sqrt(np.maximum(eigvals * (T_used - 1), 0.0))

    # Fraction of total variance per component. ||X_c||_F^2 == sum of ALL
    # singular values^2, so s_k^2 / ||X_c||_F^2 is the explained-variance ratio
    # (sums to 1 over the full spectrum; <=1 for the retained top-k). This is
    # branch-independent and avoids the earlier mismatched (N-1) denominator.
    total_sq = float(np.sum(data_centered**2))
    variance_ratio = (
        singular_values**2 / total_sq
        if total_sq > 0
        else np.zeros_like(singular_values)
    )

    # The smoothed scores are for PLOTS only — keep the projecting ones, or the
    # ICA that consumes them inherits a display parameter (see PcaResult).
    temporal_raw = temporal
    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter1d

        temporal = gaussian_filter1d(temporal, sigma=smooth_sigma, axis=1, mode="nearest")

    return PcaResult(
        spatial=spatial,
        temporal=temporal,
        temporal_raw=temporal_raw,
        variance_ratio=variance_ratio,
        singular_values=singular_values,
        mean_image=mean_image.reshape(H, W),
        frame_indices=frame_indices,
        shape_hw=(H, W),
    )


# --------------------------------------------------------------------------
# Annotation source + config wrapper
# --------------------------------------------------------------------------


def load_group_source(
    config: "PipelineConfig",
    output_dir: Path,
    name: str,
) -> tuple[np.ndarray, str]:
    """The (H, W, T_sub) float64 stack PCA / ICA analyse for one channel group.

    It is the group's dF/F — the same array ROI, the exports and the movies
    analyse (``annotation._load_name_dff_stack``), temporally strided by
    ``pca_skip_frames`` so the fit stays memory-bounded.

    This used to be addressed by *channel index* and fell back to the RAW
    registered channel whenever the name had no donner, so PCA/ICA fitted raw
    fluorescence (~1e3 counts, dominated by vignetting and bleach) while every
    other consumer analysed dF/F (~1e-2).  The components a user was shown, and
    asked to exclude, were not the components of the signal anyone else used.

    Returns ``(image_df, note)``; raises FileNotFoundError when the group has
    nothing to read.
    """
    from .annotation import _load_name_dff_stack

    output_dir = Path(output_dir)
    groups = config.channel_groups()
    if name not in groups:
        raise FileNotFoundError(f"no channel group named {name!r}")

    skip = max(1, int(getattr(config, "pca_skip_frames", 1)))
    # The PLAIN dF/F: this is what the ICA basis is fitted on. Feeding the
    # denoised stack back in would fit a basis on data an earlier basis cleaned.
    data, note = _load_name_dff_stack(
        name, groups[name], config, output_dir, use_ica_denoise=False
    )
    if data is None:
        raise FileNotFoundError(f"No PCA/ICA source under {output_dir}: {note}")

    # data may be a memmap-backed (H, W, T) view; the strided read materializes
    # only every skip-th frame.
    sub = np.moveaxis(
        np.asarray(np.moveaxis(data, 2, 0)[::skip], dtype=np.float64), 0, 2
    )
    mm = getattr(getattr(data, "base", None), "_mmap", None)
    if mm is not None:
        mm.close()  # release the mapping (Windows cannot unlink an open one)
    return sub, note


def annotation_group_name(config: "PipelineConfig") -> str:
    """The channels_name group ``ch_for_annotation`` points into."""
    return config.channels_name[config.ch_for_annotation]


def load_annotation_source(
    config: "PipelineConfig",
    output_dir: Path,
) -> tuple[np.ndarray, str, str]:
    """PCA / ICA source for the ``ch_for_annotation`` group (back-compat shim).

    Returns ``(image_df, data_key, ch_name)``; ``image_df`` is (H, W, T) float64.
    """
    name = annotation_group_name(config)
    image_df, _note = load_group_source(config, output_dir, name)
    return image_df, "imageDf", name


def compute_pca_from_config(
    image_df: np.ndarray,
    config: "PipelineConfig",
) -> PcaResult:
    """Run ``compute_pca`` using PCA parameters from a PipelineConfig.

    ``image_df`` is assumed already temporally strided by ``pca_skip_frames``
    (see ``load_annotation_source``); the stride is forwarded only so
    ``frame_indices`` map back to true global frames.
    """
    return compute_pca(
        image_df,
        n_components=config.pca_n_components,
        smooth_sigma=config.pca_smooth_sigma,
        frame_stride=max(1, int(config.pca_skip_frames)),
    )


# --------------------------------------------------------------------------
# Visualization helpers
# --------------------------------------------------------------------------


def plot_pca_spatial_grid(
    pca_result: PcaResult,
    *,
    cols: int | None = None,
):
    """Plot PCA spatial maps in a grid with per-component colorbars.

    Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    n = pca_result.spatial.shape[0]
    if cols is None:
        cols = n // 2 + n % 2
    rows = int(np.ceil(n / cols))

    # Square-ish cells (maps are ~square) packed tightly to minimise whitespace.
    fig, axes = plt.subplots(rows, cols, figsize=(2.15 * cols, 2.25 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i in range(n):
        spatial = pca_result.spatial[i]
        vmax = float(np.percentile(np.abs(spatial), 99)) or 1.0  # robust to outlier pixels
        im = axes[i].imshow(spatial, cmap="RdBu_r", aspect="equal",
                            vmin=-vmax, vmax=vmax)
        title = f"PC{i + 1} ({pca_result.variance_ratio[i] * 100:.1f}%)"
        axes[i].set_title(title, fontsize=8, pad=2)
        axes[i].axis("off")
        cbar = fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.03,
                            ticks=[-vmax, 0.0, vmax])  # 3 compact ticks
        cbar.ax.tick_params(labelsize=6)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("PCA — Spatial Maps", fontsize=12)
    # w_pad leaves room for the colorbar tick labels so they don't touch the
    # next map; h_pad stays tight (rows are the main whitespace source).
    fig.tight_layout(pad=0.3, h_pad=0.25, w_pad=0.9, rect=(0, 0, 1, 0.97))
    return fig


def plot_pca_temporal(pca_result: PcaResult, fps_channel: float):
    """Overlay all PCA temporal components on a single axis. Returns Figure."""
    import matplotlib.pyplot as plt

    n = pca_result.temporal.shape[0]
    t_sec = pca_result.frame_indices / float(fps_channel)

    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    for i in range(n):
        ax.plot(t_sec, pca_result.temporal[i], linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(t_sec[0], t_sec[-1])
    fig.suptitle("PCA — Temporal Components", fontsize=14)
    fig.tight_layout()
    return fig


def plot_scree(pca_result: PcaResult):
    """Scree plot (variance explained bar chart). Returns Figure."""
    import matplotlib.pyplot as plt

    n = pca_result.variance_ratio.shape[0]
    x = range(1, n + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, pca_result.variance_ratio * 100, color="steelblue")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_xticks(list(x))
    ax.set_title("Scree Plot")
    fig.tight_layout()
    return fig


def save_pca_source_images(
    pca_result: PcaResult,
    out_dir: Path,
    *,
    extra_source_img: np.ndarray | None = None,
    extra_source_label: str | None = None,
) -> tuple[list[np.ndarray], list[str]]:
    """Save PCA spatial maps as RGB PNGs and return cpselect-ready lists.

    Each PC is rendered with a symmetric RdBu_r colormap and written to
    ``out_dir/PC{i}.png``. When ``extra_source_img`` is provided (e.g. the
    mean image of ``ch_for_annotation``), it is prepended to the returned
    lists so cpselect can flip through it alongside the PCA maps.

    Returns
    -------
    (rgb_list, labels)
        Lists aligned 1:1 — each RGB array is uint8 (H, W, 3).
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_list: list[np.ndarray] = []
    labels: list[str] = []

    if extra_source_img is not None:
        img = np.asarray(extra_source_img)
        if img.ndim == 2:
            lo, hi = np.percentile(img, [2, 98])
            rng = max(hi - lo, 1e-12)
            normed = np.clip((img - lo) / rng, 0.0, 1.0)
            gray_u8 = (normed * 255).astype(np.uint8)
            rgb = np.stack([gray_u8] * 3, axis=-1)
        else:
            rgb = img.astype(np.uint8)
        rgb_list.append(rgb)
        labels.append(extra_source_label or "Source")

    n = pca_result.spatial.shape[0]
    for i in range(n):
        spatial = pca_result.spatial[i]
        vmax = float(np.percentile(np.abs(spatial), 99)) or 1.0  # robust to outlier pixels
        norm = Normalize(vmin=-vmax, vmax=vmax)
        sm = ScalarMappable(norm=norm, cmap="RdBu_r")
        rgb = (sm.to_rgba(spatial)[:, :, :3] * 255).astype(np.uint8)

        Image.fromarray(rgb).save(out_dir / f"PC{i + 1}.png")
        rgb_list.append(rgb)
        labels.append(f"PC{i + 1} ({pca_result.variance_ratio[i] * 100:.1f}%)")

    return rgb_list, labels
