import os

import pandas as pd
from typing import Dict, Any

from simulation.config import SEED

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

PATH_VALID_RUNS = os.path.join(PROJECT_ROOT, "simulation", "runs", "valid_runs.csv")

# Loop currently uses best_fit (app.py use_rollout=False). 
ALGORITHM = "best_fit"

try:
    raw_df = pd.read_csv(PATH_VALID_RUNS)
    df = raw_df[(raw_df["algorithm"] == ALGORITHM) & (raw_df["seed"] == SEED)].copy()

    if not df.empty:
        n = df["successful_apps"].replace(0, pd.NA)
        df["avg_cost"] = df["total_cost"] / n
        df["avg_security"] = df["total_security"] / n
        df["avg_latency"] = df["total_latency"] / n

    if df.empty:
        print(f"WARNING: No historical data found for seed {SEED} using {ALGORITHM}.")
except FileNotFoundError as e:
    print(f"CRITICAL PATH ERROR: Could not find CSV at {PATH_VALID_RUNS}")
    raise e


def _constraint_label(constraint: Dict[str, Any]) -> str:
    metric_name = (constraint.get("metric") or "").lower()
    operator = constraint.get("operator", "")
    threshold = constraint.get("threshold")
    return f"{metric_name} {operator} {threshold}"


def _best_observed_lines(corpus_df: pd.DataFrame, parsed_intent: Dict[str, Any], metric_cols: Dict[str, str]) -> list[str]:
    """Best corpus values for the primary objective and each hard-constraint metric."""
    lines = []
    seen = set()

    metrics_to_report = []
    primary = (parsed_intent.get("primary_objective") or "").lower()
    if primary in metric_cols:
        metrics_to_report.append(primary)
    for constraint in parsed_intent.get("hard_constraints", []):
        metric_name = (constraint.get("metric") or "").lower()
        if metric_name in metric_cols:
            metrics_to_report.append(metric_name)

    for metric_name in metrics_to_report:
        if metric_name in seen:
            continue
        seen.add(metric_name)
        col = metric_cols[metric_name]
        if col not in corpus_df.columns or corpus_df.empty:
            continue
        if metric_name == "security":
            best_val = corpus_df[col].max()
            lines.append(
                f"Best observed {metric_name} in the corpus: {best_val:.3f} (higher is better)."
            )
        else:
            best_val = corpus_df[col].min()
            unit = " ms" if metric_name == "latency" else ""
            lines.append(f"Best observed {metric_name} in the corpus: {best_val:.3f}{unit}.")
    return lines


def get_warm_start_context(parsed_intent: Dict[str, Any], top_k: int = 3) -> str:
    """
    Filters the historical dataset based on hard constraints and primary objective,
    returning the best historical weight combinations for a 'Warm Start'.
    Constraints are matched in mean operator units (avg_*), matching the verifier.
    """
    valid_df = df[
        (df["valid"] == True) &
        (df["failures"] == 0) &
        (df["algorithm"] == ALGORITHM)
    ].copy()
    filtered_df = valid_df.copy()

    metric_cols = {
        "cost": "avg_cost",
        "security": "avg_security",
        "latency": "avg_latency",
    }

    applied_constraints = []
    for constraint in parsed_intent.get("hard_constraints", []):
        metric_name = (constraint.get("metric") or "").lower()
        operator = constraint.get("operator")
        threshold = constraint.get("threshold")

        if metric_name in metric_cols:
            applied_constraints.append(_constraint_label(constraint))
            col = metric_cols[metric_name]
            if operator in ["<=", "<"]:
                filtered_df = filtered_df[filtered_df[col] <= threshold]
            elif operator in [">=", ">"]:
                filtered_df = filtered_df[filtered_df[col] >= threshold]

    # Fallback: if filters are too strict, reset so the proposer can still negotiate trade-offs
    used_fallback = False
    if filtered_df.empty:
        used_fallback = True
        filtered_df = valid_df.copy()

    primary = (parsed_intent.get("primary_objective") or "").lower()
    if primary in metric_cols:
        target_col = metric_cols[primary]
        # Security: higher is better
        # Cost/latency: lower is better
        ascending = primary != "security"
        filtered_df = filtered_df.sort_values(by=target_col, ascending=ascending)

    prefix = ""
    if used_fallback:
        constraints_txt = ", ".join(applied_constraints) if applied_constraints else "the active hard constraints"
        prefix = (
            f"WARNING: 0 historical runs satisfied the hard constraints ({constraints_txt}). "
            "Showing the best runs by primary objective instead. "
        )
        observed = _best_observed_lines(valid_df, parsed_intent, metric_cols)
        if observed:
            prefix += " ".join(observed) + " "
        prefix += (
            "Weight search may not close this gap. "
        )

    top_runs = filtered_df.head(top_k)

    if top_runs.empty:
        return prefix + "No matching historical runs found." if prefix else "No matching historical runs found."

    context_str = prefix + "Historical Successful Runs (mean operator units):\n"
    for _, row in top_runs.iterrows():
        w1_val = row.get("W1", row.get("w1", 0.0))
        w2_val = row.get("W2", row.get("w2", 0.0))
        w3_val = row.get("W3", row.get("w3", 0.0))

        context_str += (
            f"- Candidate Weights: w1(cost)={w1_val:.2f}, w2(sec)={w2_val:.2f}, w3(lat)={w3_val:.2f} "
            f"| Results -> Cost: {row['avg_cost']:.3f}, Security: {row['avg_security']:.3f}, "
            f"Latency: {row['avg_latency']:.3f} ms\n"
        )

    return context_str
