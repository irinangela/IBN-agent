"""The per-run figures that are used to visualize the search process and the results."""

import os

from utils.plots.io import attempts_of, load_run
from utils.plots.style import (
    AXIS_LABELS,
    AVG_KEYS,
    METRIC_COLORS,
    NORM_KEYS,
    WEIGHT_COLORS,
    WEIGHT_LABELS,
    plt,
    save_figure,
)


def _hard_metrics(intent: dict) -> list:
    """Extract the hard metrics from the intent."""
    metrics = []
    for constraint in (intent or {}).get("hard_constraints") or []:
        metric = (constraint.get("metric") or "").lower()
        if metric and metric not in metrics:
            metrics.append(metric)
    return metrics


def _threshold(intent: dict, metric: str):
    """Extract the threshold for a given metric from the intent."""
    for constraint in (intent or {}).get("hard_constraints") or []:
        if (constraint.get("metric") or "").lower() == metric:
            return constraint.get("threshold")
    return None


def _xs(attempts: list) -> list:
    """Extract the iteration number from the attempts."""
    return [row.get("attempt") or (index + 1) for index, row in enumerate(attempts)]


def _integer_xticks(ax, xs) -> None:
    """Set the integer ticks for the x-axis."""
    ticks = sorted({int(value) for value in xs if value is not None})
    if ticks:
        ax.set_xticks(ticks)


def plot_convergence(run: dict):
    """avg_* vs iteration with SLA threshold lines.
    Norms are plotted on a second column.

    Constraints live in operator units (avg_*), so those are the horizontal threshold lines. 
    norm_* is the heuristic landscape and is plotted without thresholds.
    """
    attempts = attempts_of(run)
    if not attempts:
        return None

    metrics = _hard_metrics(run.get("initial_intent") or {})
    if not metrics:
        metrics = [name for name in ("cost", "latency", "security") if any(
            row.get(AVG_KEYS[name]) is not None for row in attempts
        )]
    if not metrics:
        return None

    has_norms = any(
        row.get(NORM_KEYS[metric]) is not None
        for metric in metrics
        for row in attempts
    )
    ncols = 2 if has_norms else 1
    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=ncols,
        figsize=(6.4 * ncols, 2.6 * len(metrics)),
        sharex=True,
        squeeze=False,
    )
    xs = _xs(attempts)
    original_intent = run.get("initial_intent") or {}
    active_intent = run.get("active_intent") or {}
    events = run.get("relaxation_events") or run.get("relaxations") or []

    for row_index, metric in enumerate(metrics):
        color = METRIC_COLORS[metric]
        avg_ax = axes[row_index][0]
        ys = [row.get(AVG_KEYS[metric]) for row in attempts]
        avg_ax.plot(xs, ys, color=color, marker="o", label="delivered")

        requested = _threshold(original_intent, metric)
        active = _threshold(active_intent, metric)
        if requested is not None:
            avg_ax.axhline(
                float(requested), color="black", linestyle="-", linewidth=1.0,
                label="requested",
            )
        if active is not None and requested is not None and float(active) != float(requested):
            avg_ax.axhline(
                float(active), color="black", linestyle="--", linewidth=1.0,
                label="relaxed",
            )
        for event in events:
            iteration = event.get("iteration")
            if iteration is None or event.get("metric") != metric:
                continue
            avg_ax.axvline(float(iteration), color="#666666", linestyle=":", linewidth=1.0)

        avg_ax.set_ylabel(AXIS_LABELS[metric])
        avg_ax.set_title(metric)
        if row_index == 0:
            avg_ax.legend(loc="best")

        if has_norms:
            norm_ax = axes[row_index][1]
            norm_key = NORM_KEYS[metric]
            norm_ys = [row.get(norm_key) for row in attempts]
            if any(value is not None for value in norm_ys):
                norm_ax.plot(xs, norm_ys, color=color, marker="o")
            norm_ax.set_ylabel(AXIS_LABELS[norm_key])
            norm_ax.set_title(f"{metric} (heuristic)")

    axes[-1][0].set_xlabel("iteration")
    _integer_xticks(axes[-1][0], xs)
    if has_norms:
        axes[-1][1].set_xlabel("iteration")
        _integer_xticks(axes[-1][1], xs)
    fig.suptitle("Convergence trajectory", fontsize=12, y=1.02)
    return fig


def plot_weights(run: dict):
    """Plot the search path through the weight triplet."""
    attempts = attempts_of(run)
    if not attempts:
        return None
    xs = _xs(attempts)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for key in ("w1", "w2", "w3"):
        ys = [row.get(key) for row in attempts]
        if all(value is None for value in ys):
            continue
        ax.plot(xs, ys, color=WEIGHT_COLORS[key], marker="o", label=WEIGHT_LABELS[key])
    ax.set_xlabel("iteration")
    _integer_xticks(ax, xs)
    ax.set_ylabel("weight")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Weight trajectory")
    ax.legend(loc="best")
    return fig


def plot_requested_vs_delivered(run: dict):
    """One bar per hard constraint: requested / delivered / relaxed."""
    constraints = (run.get("initial_intent") or {}).get("hard_constraints") or []
    if not constraints:
        return None

    metrics = _hard_metrics(run.get("initial_intent") or {})
    final = run.get("final_metrics") or {}
    if not final and attempts_of(run):
        final = attempts_of(run)[-1]

    labels = []
    requested_vals = []
    delivered_vals = []
    relaxed_vals = []
    has_relax = False
    for metric in metrics:
        requested = _threshold(run.get("initial_intent") or {}, metric)
        active = _threshold(run.get("active_intent") or {}, metric)
        delivered = final.get(AVG_KEYS[metric])
        if requested is None and delivered is None:
            continue
        labels.append(metric)
        requested_vals.append(float(requested) if requested is not None else 0.0)
        delivered_vals.append(float(delivered) if delivered is not None else 0.0)
        if active is not None and requested is not None and float(active) != float(requested):
            relaxed_vals.append(float(active))
            has_relax = True
        else:
            relaxed_vals.append(None)

    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(max(6.5, 1.8 * len(labels) + 2), 3.8))
    xs = list(range(len(labels)))
    width = 0.25 if has_relax else 0.35
    offsets = (-width, 0.0, width) if has_relax else (-width / 2, width / 2)

    ax.bar(
        [x + offsets[0] for x in xs], requested_vals, width,
        label="requested", color="#4D4D4D",
    )
    ax.bar(
        [x + offsets[1] for x in xs], delivered_vals, width,
        label="delivered", color="#0072B2",
    )
    if has_relax:
        relaxed_x = [x + offsets[2] for x, value in zip(xs, relaxed_vals) if value is not None]
        relaxed_y = [value for value in relaxed_vals if value is not None]
        ax.bar(
            relaxed_x, relaxed_y, width,
            label="relaxed", color="white", edgecolor="black", hatch="///",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("operator units")
    outcome = "success" if run.get("success") else "did not meet SLA"
    ax.set_title(f"Requested vs delivered ({outcome})")
    ax.legend(loc="best")
    return fig


def save_run_figures(json_path: str, out_dir: str = None) -> list:
    """Read one analytics JSON and write the three per-run figures beside it."""
    run = load_run(json_path)
    stem = os.path.splitext(os.path.basename(json_path))[0]
    directory = out_dir or os.path.dirname(os.path.abspath(json_path)) or "."
    os.makedirs(directory, exist_ok=True)

    written = []
    for name, plotter in (
        ("convergence", plot_convergence),
        ("weights", plot_weights),
        ("requested-vs-delivered", plot_requested_vs_delivered),
    ):
        fig = plotter(run)
        if fig is None:
            continue
        written.extend(save_figure(fig, os.path.join(directory, f"{stem}-{name}")))
        plt.close(fig)
    return written
