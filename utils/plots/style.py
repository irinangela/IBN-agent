"""Shared matplotlib style for consistent figures."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# cost / w1, security / w2, latency / w3
METRIC_COLORS = {
    "cost": "#0072B2",
    "security": "#009E73",
    "latency": "#D55E00",
}

WEIGHT_COLORS = {
    "w1": METRIC_COLORS["cost"],
    "w2": METRIC_COLORS["security"],
    "w3": METRIC_COLORS["latency"],
}

WEIGHT_LABELS = {
    "w1": "w1 (cost)",
    "w2": "w2 (security)",
    "w3": "w3 (latency)",
}

AVG_KEYS = {
    "cost": "avg_cost",
    "latency": "avg_latency",
    "security": "avg_security",
}

NORM_KEYS = {
    "cost": "norm_cost",
    "latency": "norm_lat",
    "security": "norm_sec",
}

AXIS_LABELS = {
    "cost": "avg cost (per app)",
    "latency": "avg latency (ms per app)",
    "security": "avg security (per app)",
    "norm_cost": "norm_cost (heuristic)",
    "norm_lat": "norm_lat (heuristic)",
    "norm_sec": "norm_sec (heuristic)",
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "lines.linewidth": 1.4,
        "lines.markersize": 5,
    })


apply_style()


def save_figure(fig, stem: str) -> list:
    """Write PNG (for UI and preview) and PDF (for the thesis) next to each other."""
    paths = []
    for ext in ("png", "pdf"):
        path = f"{stem}.{ext}"
        fig.savefig(
            path,
            dpi=300 if ext == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
        paths.append(path)
    return paths
