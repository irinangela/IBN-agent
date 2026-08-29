"""Figures for the historical runs stored in the CSV file. Independent of any agent run."""

import os

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

from nodes.weights import clip_and_normalize_weights
from simulation.config import WARM_START_WEIGHT_SETS
from utils.plots.io import attempts_of, valid_failure_free
from utils.plots.style import (
    AXIS_LABELS,
    plt,
    save_figure,
)

DENSE_SIMPLEX_MIN_POINTS = 50
CLIP_NEIGHBOURHOOD = 0.15
DEFAULT_SIMPLEX_BASELINE = (0.35, 0.20, 0.45)


def _simplex_points(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep rows that sit on w1+w2+w3 ≈ 1; x=w1, y=w2, w3"""
    out = frame.copy()
    out["W1"] = pd.to_numeric(out["W1"], errors="coerce")
    out["W2"] = pd.to_numeric(out["W2"], errors="coerce")
    out["W3"] = pd.to_numeric(out["W3"], errors="coerce")
    out = out.dropna(subset=["W1", "W2"])
    weight_sum = out["W1"] + out["W2"] + out["W3"].fillna(1.0 - out["W1"] - out["W2"])
    out = out[(weight_sum - 1.0).abs() < 0.05]
    out = out[out["W1"] + out["W2"] <= 1.05]
    return out

def _draw_simplex_outline(ax) -> None:
    ax.plot([0, 1, 0, 0], [0, 0, 1, 0], color="black", linewidth=0.8)


def _weights_from_attempt(row: dict):
    if not row:
        return None
    w1, w2, w3 = row.get("w1"), row.get("w2"), row.get("w3")
    if w1 is None or w2 is None or w3 is None:
        return None
    return (float(w1), float(w2), float(w3))


def _baseline_from_run(run: dict):
    attempts = attempts_of(run) if run else []
    if not attempts:
        return None
    return _weights_from_attempt(attempts[0])


def _box_boundary_samples(base, delta: float, n_edge: int = 16):
    """Corners and edges of the axis-aligned clip box."""
    lo = [b - delta for b in base]
    hi = [b + delta for b in base]
    samples = [
        (lo[0] if a else hi[0], lo[1] if b else hi[1], lo[2] if c else hi[2])
        for a in (0, 1)
        for b in (0, 1)
        for c in (0, 1)
    ]
    ts = np.linspace(0.0, 1.0, n_edge)
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        for u in (lo[others[0]], hi[others[0]]):
            for v in (lo[others[1]], hi[others[1]]):
                for t in ts:
                    point = [0.0, 0.0, 0.0]
                    point[axis] = lo[axis] + float(t) * (hi[axis] - lo[axis])
                    point[others[0]] = u
                    point[others[1]] = v
                    samples.append(tuple(point))
    return samples


def _clip_neighbourhood_xy(base, delta: float = CLIP_NEIGHBOURHOOD):
    """polygon of clip-then-normalize around a baseline"""
    mapped = []
    for w1, w2, w3 in _box_boundary_samples(base, delta):
        out = clip_and_normalize_weights(w1, w2, w3, base[0], base[1], base[2])
        x, y = out["w1"], out["w2"]
        if x + y > 1.05 or x < -0.02 or y < -0.02:
            continue
        mapped.append((x, y))
    if len(mapped) < 3:
        return np.empty((0, 2))
    pts = np.unique(np.round(np.asarray(mapped, dtype=float), 6), axis=0)
    return _convex_hull_xy(pts)


def _convex_hull_xy(pts: np.ndarray) -> np.ndarray:
    """keeps the clip neighbourhood a single polygon."""
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=float)
    return hull if len(hull) >= 3 else pts

def plot_rugged_front(frame: pd.DataFrame, algorithm: str = "best_fit"):
    """Colored by achieved norms"""
    subset = valid_failure_free(frame)
    if "algorithm" in subset.columns:
        subset = subset[subset["algorithm"] == algorithm]
    subset = _simplex_points(subset)
    if subset.empty:
        return None

    metrics = [
        ("norm_lat", "norm_lat (heuristic)"),
        ("norm_cost", "norm_cost (heuristic)"),
        ("norm_sec", "norm_sec (heuristic)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharex=True, sharey=True)
    dense = len(subset) >= DENSE_SIMPLEX_MIN_POINTS

    for ax, (column, title) in zip(axes, metrics):
        if column not in subset.columns:
            ax.set_visible(False)
            continue
        x = subset["W1"].to_numpy()
        y = subset["W2"].to_numpy()
        z = pd.to_numeric(subset[column], errors="coerce").to_numpy()
        _draw_simplex_outline(ax)
        if dense and len(np.unique(np.round(x, 5))) >= 4:
            import matplotlib.tri as mtri

            triang = mtri.Triangulation(x, y)
            tri_x = x[triang.triangles].mean(axis=1)
            tri_y = y[triang.triangles].mean(axis=1)
            triang.set_mask(tri_x + tri_y > 1.0)
            contour = ax.tricontourf(triang, z, levels=12, cmap="cividis")
            fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
        else:
            scatter = ax.scatter(x, y, c=z, cmap="cividis", s=36, edgecolors="black", linewidths=0.3)
            fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_xlabel("w1 (cost)")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal", adjustable="box")

    axes[0].set_ylabel("w2 (security)")
    fig.suptitle(f"Rugged front ({algorithm}, w3 = 1 − w1 − w2)", fontsize=12, y=1.03)
    return fig


def plot_pareto(frame: pd.DataFrame, algorithm: str = "best_fit", run: dict = None):
    """Historical cost vs latency, colored by security.
    Optional agent path overlay."""

    subset = valid_failure_free(frame)
    if "algorithm" in subset.columns:
        subset = subset[subset["algorithm"] == algorithm]
    if subset.empty:
        return None

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    scatter = ax.scatter(
        subset["avg_cost"],
        subset["avg_latency"],
        c=subset["avg_security"],
        cmap="cividis",
        s=28,
        edgecolors="black",
        linewidths=0.25,
        label="historical",
    )
    fig.colorbar(scatter, ax=ax, label="avg security")

    if run:
        attempts = attempts_of(run)
        xs = [row.get("avg_cost") for row in attempts]
        ys = [row.get("avg_latency") for row in attempts]
        if any(value is not None for value in xs) and any(value is not None for value in ys):
            ax.plot(xs, ys, color="#D55E00", marker="o", linewidth=1.4, label="agent path")
            ax.scatter([xs[-1]], [ys[-1]], color="#D55E00", s=64, zorder=5, marker="*")

    ax.set_xlabel(AXIS_LABELS["cost"])
    ax.set_ylabel(AXIS_LABELS["latency"])
    ax.set_title(f"Approximate Pareto front ({algorithm})")
    ax.legend(loc="best")
    return fig


def plot_weight_simplex(run: dict = None, baseline=None):
    """66-point w1+w2+w3=1 grid with a +-0.15 clip-and-normalize neighbourhood."""
    grid = np.asarray(WARM_START_WEIGHT_SETS, dtype=float)
    if grid.size == 0:
        return None

    base = baseline or _baseline_from_run(run) or DEFAULT_SIMPLEX_BASELINE
    neighbourhood = _clip_neighbourhood_xy(base)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    _draw_simplex_outline(ax)

    inside = np.zeros(len(grid), dtype=bool)
    if len(neighbourhood) >= 3:
        closed = np.vstack([neighbourhood, neighbourhood[0]])
        ax.fill(
            neighbourhood[:, 0],
            neighbourhood[:, 1],
            color="#D55E00",
            alpha=0.22,
            linewidth=0,
            label="reachable after clip ±0.15",
            zorder=1,
        )
        ax.plot(
            closed[:, 0],
            closed[:, 1],
            color="#D55E00",
            linewidth=1.0,
            zorder=2,
        )
        inside = MplPath(neighbourhood).contains_points(grid[:, :2], radius=1e-6)

    ax.scatter(
        grid[:, 0],
        grid[:, 1],
        s=22,
        facecolors="white",
        edgecolors="black",
        linewidths=0.5,
        zorder=3,
        label=f"weight grid ({len(grid)} points, step 0.1)",
    )
    if inside.any():
        ax.scatter(
            grid[inside, 0],
            grid[inside, 1],
            s=28,
            color="#0072B2",
            edgecolors="black",
            linewidths=0.4,
            zorder=4,
            label="grid points inside neighbourhood",
        )

    ax.scatter(
        [base[0]],
        [base[1]],
        s=120,
        marker="*",
        color="black",
        zorder=6,
        label=f"baseline ({base[0]:.2f}, {base[1]:.2f}, {base[2]:.2f})",
    )

    attempts = attempts_of(run) if run else []
    path = [xy for xy in (_weights_from_attempt(row) for row in attempts) if xy]
    if len(path) >= 2:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, color="#D55E00", linewidth=1.4, marker="o", markersize=4, zorder=5, label="agent path")
        ax.scatter([xs[-1]], [ys[-1]], s=64, marker="^", color="#D55E00", zorder=7, label="last attempt")

    ax.text(1.02, -0.04, "w1 = 1\n(cost)", ha="left", va="top", fontsize=8)
    ax.text(-0.02, 1.04, "w2 = 1\n(security)", ha="right", va="bottom", fontsize=8)
    ax.text(-0.04, -0.04, "w3 = 1\n(latency)", ha="right", va="top", fontsize=8)
    ax.set_xlabel("w1 (cost)")
    ax.set_ylabel("w2 (security)")
    ax.set_xlim(-0.12, 1.18)
    ax.set_ylim(-0.14, 1.16)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Weight simplex and clip-and-normalize neighbourhood")
    ax.legend(loc="upper right", fontsize=8)
    return fig


def plot_algorithm_comparison(frame: pd.DataFrame):
    """Paired best_fit vs app_rollout at identical weights."""

    subset = valid_failure_free(frame)
    if subset.empty or "algorithm" not in subset.columns:
        return None

    keys = ["seed", "W1", "W2", "W3"]
    best = subset[subset["algorithm"] == "best_fit"]
    roll = subset[subset["algorithm"] == "app_rollout"]
    merged = best.merge(roll, on=keys, suffixes=("_bf", "_ar"))
    if merged.empty:
        return None

    panels = [
        ("avg_cost", "avg cost"),
        ("avg_latency", "avg latency"),
        ("avg_security", "avg security"),
        ("duration_s", "runtime (s)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.4))
    for ax, (column, title) in zip(axes, panels):
        left = pd.to_numeric(merged[f"{column}_bf"], errors="coerce")
        right = pd.to_numeric(merged[f"{column}_ar"], errors="coerce")
        ax.scatter(left, right, s=22, color="#0072B2", edgecolors="black", linewidths=0.25)
        finite = pd.concat([left, right]).dropna()
        if not finite.empty:
            lo, hi = float(finite.min()), float(finite.max())
            pad = (hi - lo) * 0.05 or 1.0
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("best_fit")
        ax.set_ylabel("app_rollout")
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle("Algorithm comparison at identical weights", fontsize=12, y=1.03)
    return fig


def save_dataset_figures(
    frame: pd.DataFrame,
    out_dir: str,
    algorithm: str = "best_fit",
    run: dict = None,
) -> list:
    os.makedirs(out_dir, exist_ok=True)
    written = []
    jobs = (
        ("rugged-front", lambda: plot_rugged_front(frame, algorithm=algorithm)),
        ("pareto", lambda: plot_pareto(frame, algorithm=algorithm, run=run)),
        ("algorithm-comparison", lambda: plot_algorithm_comparison(frame)),
        ("weight-simplex", lambda: plot_weight_simplex(run=run)),
    )
    for name, plotter in jobs:
        fig = plotter()
        if fig is None:
            continue
        written.extend(save_figure(fig, os.path.join(out_dir, name)))
        plt.close(fig)
    return written
