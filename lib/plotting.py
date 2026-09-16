"""Shared plotting styles for all experiment scripts."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET_COLORS = {
    "ADRA2B": "#1f77b4",
    "CNR1":   "#ff7f0e",
    "DRD4":   "#2ca02c",
    "MTR1A":  "#d62728",
    "SC6A4":  "#9467bd",
    "SGMR2":  "#8c564b",
    "5HT2A":  "#e377c2",
    "AmpC":   "#7f7f7f",
}

MODEL_COLORS = {
    "DeepEns_concat":   "#1f77b4",
    "BoltzNN_v1_wide":  "#ff7f0e",
    "BetterNN":         "#2ca02c",
}


def setup_style() -> None:
    """Set matplotlib defaults for publication-quality figures."""
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.constrained_layout.use": True,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 150) -> None:
    """Save figure, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def add_oracle_line(
    ax: plt.Axes,
    value: float,
    label: str = "Oracle",
    color: str = "black",
) -> None:
    """Draw a horizontal dashed reference line for the oracle baseline."""
    ax.axhline(value, linestyle="--", linewidth=1.2, color=color, alpha=0.7, label=label)
