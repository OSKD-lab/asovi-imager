"""Central matplotlib / seaborn style for every figure the pipeline produces.

Applied once (process-global rcParams) so all ``plot_*`` helpers share a
consistent look without each having to re-style its axes.  The per-axes
``sns.despine`` / ``ax.grid`` from the requested snippet are expressed as
rcParams here so they apply to every axes automatically.
"""

from __future__ import annotations

_APPLIED = False


def apply_style(force: bool = False) -> None:
    """Set the project's standard figure style (idempotent by default).

    Equivalent to::

        sns.set_style("ticks")
        sns.set_context("notebook", font_scale=1.1)
        plt.rcParams["pdf.fonttype"] = 42
        sns.despine(ax=plt.gca())          # -> spines.top/right off
        plt.gca().grid(axis="y", alpha=0.3)  # -> faint y-grid
    """
    global _APPLIED
    if _APPLIED and not force:
        return

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("ticks")
    sns.set_context("notebook", font_scale=1.1)
    plt.rcParams["pdf.fonttype"] = 42  # editable text in Illustrator/PDF
    # despine (top + right) as a global default
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    # faint horizontal grid, drawn behind the data (seaborn sets axisbelow)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["axes.grid.axis"] = "y"
    plt.rcParams["grid.alpha"] = 0.3

    _APPLIED = True
