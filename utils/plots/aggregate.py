"""Batch figures over many analytics.
These can be used to group runs into categories for thesis analysis.
"""

import os

import pandas as pd

from utils.plots.style import plt, save_figure


CATEGORIES = (
    "feasible",
    "infeasible-requiring-relaxation",
    "ambiguous",
)


def categorize_run(run: dict) -> str:
    """Map one analytics JSON onto the three evaluation categories.

    - feasible: history said jointly feasible, run succeeded, no threshold change
    - infeasible-requiring-relaxation: HITL needed, or any threshold was loosened,
      or the search ran to failure
    - ambiguous: independently-feasible metrics that are jointly empty (Pareto
      conflict) and the operator did not actually relax

    Parse failures do not produce a JSON, so they are not included in the categories.
    """
    report = run.get("feasibility_report") or {}
    relaxations = run.get("relaxation_events") or run.get("relaxations") or []
    needs_operator = bool(report.get("needs_operator"))
    success = bool(run.get("success"))
    chosen = report.get("chosen_algorithm") or run.get("algorithm") or "best_fit"
    per_algorithm = report.get("per_algorithm") or {}
    slice_ = per_algorithm.get(chosen) or {}
    jointly = slice_.get("jointly_feasible")
    infeasible_metrics = slice_.get("infeasible_metrics") or []

    if relaxations or (run.get("operator_decision") or {}).get("action") == "relax":
        # The search required relaxation.  
        return "infeasible-requiring-relaxation"
    if jointly and success:
        # The search succeeded on a historically feasible SLA.
        return "feasible"
    if jointly is False and not infeasible_metrics and needs_operator and not success:
        # Independent bounds looked OK but the combination did not.
        return "ambiguous"
    if jointly and not success:
        # Even if search failed on a historically feasible SLA, we still consider it feasible.
        return "feasible"
    if needs_operator or not success:
        # The search required relaxation from HITL or failed.
        return "infeasible-requiring-relaxation"
    return "feasible" # Default case.


def _runs_table(runs: list) -> pd.DataFrame:
    rows = []
    for run in runs:
        tokens = (run.get("token_usage") or {}).get("total_tokens") or 0
        rows.append({
            "category": categorize_run(run),
            "success": bool(run.get("success")),
            "iterations": int(run.get("iterations_done") or 0),
            "max_iterations": int(run.get("max_iterations") or 0),
            "tokens": int(tokens),
            "relaxation_order": list(
                (run.get("initial_intent") or {}).get("relaxation_order") or []
            ),
            "relaxation_events": list(
                run.get("relaxation_events") or run.get("relaxations") or []
            ),
        })
    return pd.DataFrame(rows)


def plot_category_summary(runs: list):
    """Evaluation: success rate, iterations, tokens by intent category."""
    table = _runs_table(runs)
    if table.empty:
        return None

    present = [name for name in CATEGORIES if name in set(table["category"])]
    if not present:
        present = sorted(table["category"].unique())

    success_rate = []
    mean_iters = []
    mean_tokens = []
    for name in present:
        slice_ = table[table["category"] == name]
        success_rate.append(float(slice_["success"].mean()) if len(slice_) else 0.0)
        # As a failure, we count the iteration count surpassing the max iterations.
        iters = slice_.apply(
            lambda row: row["iterations"] if row["success"] else (
                row["max_iterations"] or row["iterations"]
            ),
            axis=1,
        )
        mean_iters.append(float(iters.mean()) if len(slice_) else 0.0)
        mean_tokens.append(float(slice_["tokens"].mean()) if len(slice_) else 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    x = list(range(len(present)))
    short = [name.replace("infeasible-requiring-relaxation", "needs relaxation") for name in present]

    axes[0].bar(x, success_rate, color="#4D4D4D")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("success rate")
    axes[0].set_title("Success rate")

    axes[1].bar(x, mean_iters, color="#0072B2")
    axes[1].set_ylabel("mean iterations")
    axes[1].set_title("Iterations to success")

    axes[2].bar(x, mean_tokens, color="#D55E00")
    axes[2].set_ylabel("mean tokens")
    axes[2].set_title("Token cost")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=20, ha="right")

    fig.suptitle("Aggregate by intent category", fontsize=12, y=1.03)
    return fig


def plot_relaxation_correctness(runs: list):
    """Evaluation: correctness of the relaxation order."""
    table = _runs_table(runs)
    with_events = table[table["relaxation_events"].map(lambda events: bool(events))]
    if with_events.empty:
        return None

    correct = 0
    incorrect = 0
    skipped = 0
    for _, row in with_events.iterrows():
        order = row["relaxation_order"]
        events = row["relaxation_events"]
        if not order:
            skipped += 1
            continue
        first = (events[0] or {}).get("metric")
        if first == order[0]:
            correct += 1
        else:
            incorrect += 1

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    labels = ["matched order[0]", "did not match", "no order declared"]
    values = [correct, incorrect, skipped]
    ax.bar(labels, values, color=["#009E73", "#D55E00", "#7F7F7F"])
    ax.set_ylabel("runs")
    ax.set_title("Relaxation-decision correctness")
    return fig


def plot_success_vs_tokens(runs: list):
    """Evaluation: success rate vs token cost."""
    table = _runs_table(runs)
    if table.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for success, marker, label in ((True, "o", "success"), (False, "x", "failure")):
        slice_ = table[table["success"] == success]
        if slice_.empty:
            continue
        ax.scatter(
            slice_["tokens"], slice_["iterations"],
            marker=marker, label=label, color="#0072B2" if success else "#D55E00",
        )
    ax.set_xlabel("total tokens")
    ax.set_ylabel("iterations")
    ax.set_title("Effort vs outcome")
    ax.legend(loc="best")
    return fig


def save_batch_figures(runs: list, out_dir: str) -> list:
    os.makedirs(out_dir, exist_ok=True)
    written = []
    jobs = (
        ("category-summary", lambda: plot_category_summary(runs)),
        ("relaxation-correctness", lambda: plot_relaxation_correctness(runs)),
        ("success-vs-tokens", lambda: plot_success_vs_tokens(runs)),
    )
    for name, plotter in jobs:
        fig = plotter()
        if fig is None:
            continue
        written.extend(save_figure(fig, os.path.join(out_dir, name)))
        plt.close(fig)
    return written
