import os
from typing import Any, Dict, Optional

import pandas as pd

from simulation.config import SEED

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

PATH_VALID_RUNS = os.path.join(PROJECT_ROOT, "simulation", "runs", "valid_runs.csv")

METRIC_COLS = {
    "cost": "avg_cost",
    "security": "avg_security",
    "latency": "avg_latency",
}
ALGORITHMS = ("best_fit", "app_rollout")
DEFAULT_ALGORITHM = "best_fit"
RELAXATION_MARGIN = 0.05
HIGHER_IS_BETTER = {"security"}

try:
    raw_df = pd.read_csv(PATH_VALID_RUNS)
    df = raw_df.copy()
    if not df.empty:
        n = df["successful_apps"].replace(0, pd.NA)
        df["avg_cost"] = df["total_cost"] / n
        df["avg_security"] = df["total_security"] / n
        df["avg_latency"] = df["total_latency"] / n
    if df.empty:
        print(f"WARNING: No historical data found in {PATH_VALID_RUNS}.")
except FileNotFoundError as e:
    print(f"CRITICAL PATH ERROR: Could not find CSV at {PATH_VALID_RUNS}")
    raise e


def filter_runs(seed: int = SEED, algorithm: str = DEFAULT_ALGORITHM) -> pd.DataFrame:
    """Valid, failure-free rows for one seed and algorithm.
    Seed selects the live topology and workload (no separate variant column).
    """
    if df.empty:
        return df.copy()
    return df[
        (df["valid"] == True)
        & (df["failures"] == 0)
        & (df["seed"] == seed)
        & (df["algorithm"] == algorithm)
    ].copy()


def apply_hard_constraints(
    core_df: pd.DataFrame,
    hard_constraints: list,
) -> pd.DataFrame:
    """Keep rows that satisfy every hard constraint in mean operator units."""

    filtered_df = core_df.copy()
    for constraint in hard_constraints or []:
        metric_name = (constraint.get("metric") or "").lower()
        operator = constraint.get("operator")
        threshold = constraint.get("threshold")
        col = METRIC_COLS.get(metric_name)
        if not col or col not in filtered_df.columns or threshold is None:
            continue
        if operator in ["<=", "<"]:
            filtered_df = filtered_df[filtered_df[col] <= threshold]
        elif operator in [">=", ">"]:
            filtered_df = filtered_df[filtered_df[col] >= threshold]
    return filtered_df


def is_jointly_feasible(core_df: pd.DataFrame, hard_constraints: list) -> bool:
    if core_df is None or core_df.empty:
        return False
    if not hard_constraints:
        return True
    return not apply_hard_constraints(core_df, hard_constraints).empty


def best_known_bounds(core_df: pd.DataFrame) -> Dict[str, float]:
    bounds: Dict[str, float] = {}
    if core_df is None or core_df.empty:
        return bounds
    for metric_name, col in METRIC_COLS.items():
        if col not in core_df.columns:
            continue
        series = core_df[col].dropna()
        if series.empty:
            continue
        if metric_name in HIGHER_IS_BETTER:
            bounds[metric_name] = float(series.max())
        else:
            bounds[metric_name] = float(series.min())
    return bounds


def constraint_violates_bound(constraint: Dict[str, Any], bounds: Dict[str, float]) -> bool:
    """True if the requested threshold is beyond the (current) best-known value."""

    metric_name = (constraint.get("metric") or "").lower()
    operator = constraint.get("operator")
    threshold = constraint.get("threshold")
    best = bounds.get(metric_name)
    if best is None or threshold is None:
        return False
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        return False
    if operator in ("<=", "<"):
        return best > threshold if operator == "<=" else best >= threshold
    if operator in (">=", ">"):
        return best < threshold if operator == ">=" else best <= threshold
    return False


def suggested_relaxation(
    metric: str,
    best_value: float,
    margin: float = RELAXATION_MARGIN,
) -> float:
    """Loosen best-known by a small margin"""
    if metric in HIGHER_IS_BETTER:
        return float(best_value) * (1.0 - margin)
    return float(best_value) * (1.0 + margin)


def _constraint_label(constraint: Dict[str, Any]) -> str:
    metric_name = (constraint.get("metric") or "").lower()
    operator = constraint.get("operator", "")
    threshold = constraint.get("threshold")
    return f"{metric_name} {operator} {threshold}"


def _pick_better(metric: str, values: list[float]) -> Optional[float]:
    if not values:
        return None
    if metric in HIGHER_IS_BETTER:
        return max(values)
    return min(values)


def _better_bound(metric: str, per_algorithm: Dict[str, Any]) -> Optional[float]:
    values = []
    for info in per_algorithm.values():
        bound = (info.get("bounds") or {}).get(metric)
        if bound is not None:
            values.append(float(bound))
    return _pick_better(metric, values)


def _conditional_best(metric: str, others: list, seed: int) -> Optional[float]:
    """Best-known value of metric among runs that satisfy the other constraints."""

    values = []
    for algorithm in ALGORITHMS:
        runs = filter_runs(seed, algorithm)
        filtered = apply_hard_constraints(runs, others)
        bound = best_known_bounds(filtered).get(metric)
        if bound is not None:
            values.append(float(bound))
    return _pick_better(metric, values)


def _best_observed_lines(
    core_df: pd.DataFrame,
    parsed_intent: Dict[str, Any],
) -> list[str]:
    """Best known values for the primary objective and each hard-constraint metric."""
    lines = []
    seen = set()
    bounds = best_known_bounds(core_df)

    metrics_to_report = []
    primary = (parsed_intent.get("primary_objective") or "").lower()
    if primary in METRIC_COLS:
        metrics_to_report.append(primary)
    for constraint in parsed_intent.get("hard_constraints", []):
        metric_name = (constraint.get("metric") or "").lower()
        if metric_name in METRIC_COLS:
            metrics_to_report.append(metric_name)

    for metric_name in metrics_to_report:
        if metric_name in seen:
            continue
        seen.add(metric_name)
        best_val = bounds.get(metric_name)
        if best_val is None:
            continue
        if metric_name in HIGHER_IS_BETTER:
            lines.append(
                f"Best observed {metric_name}: {best_val:.3f} (higher is better)."
            )
        else:
            unit = " ms" if metric_name == "latency" else ""
            lines.append(f"Best observed {metric_name}: {best_val:.3f}{unit}.")
    return lines


def _algorithm_slice(seed: int, algorithm: str, hard_constraints: list) -> Dict[str, Any]:
    runs = filter_runs(seed, algorithm)
    bounds = best_known_bounds(runs)
    infeasible = []
    for constraint in hard_constraints or []:
        metric_name = (constraint.get("metric") or "").lower()
        if metric_name not in METRIC_COLS:
            continue
        if constraint_violates_bound(constraint, bounds):
            infeasible.append(metric_name)
    return {
        "bounds": bounds,
        "jointly_feasible": is_jointly_feasible(runs, hard_constraints),
        "infeasible_metrics": infeasible,
    }


def feasibility_report(parsed_intent: Dict[str, Any], seed: int = SEED) -> Dict[str, Any]:
    """Deterministic bounds and joint feasibility per algorithm for this seed."""

    hard_constraints = parsed_intent.get("hard_constraints") or []
    non_relaxable = {
        m.lower() for m in (parsed_intent.get("non_relaxable_constraints") or [])
    }

    per_algorithm = {
        algorithm: _algorithm_slice(seed, algorithm, hard_constraints)
        for algorithm in ALGORITHMS
    }

    best_fit_ok = per_algorithm["best_fit"]["jointly_feasible"]
    rollout_ok = per_algorithm["app_rollout"]["jointly_feasible"]

    auto_switched_to = None
    needs_operator = False
    if best_fit_ok:
        chosen_algorithm = "best_fit"
    elif rollout_ok:
        chosen_algorithm = "app_rollout"
        auto_switched_to = "app_rollout"
    else:
        chosen_algorithm = "best_fit"
        needs_operator = True

    suggested_relaxations = []
    if needs_operator:
        for index, constraint in enumerate(hard_constraints):
            metric_name = (constraint.get("metric") or "").lower()
            if metric_name not in METRIC_COLS:
                continue
            others = [
                other for i, other in enumerate(hard_constraints) if i != index
            ]
            given = [
                _constraint_label(other)
                for other in others
                if (other.get("metric") or "").lower() in METRIC_COLS
            ]
            best = _conditional_best(metric_name, others, seed)
            if best is None:
                # fall back to the best-known value of the independent metric
                best = _better_bound(metric_name, per_algorithm)
                given = []
            if best is None:
                continue
            if not constraint_violates_bound(constraint, {metric_name: best}):
                continue
            suggested_relaxations.append({
                "metric": metric_name,
                "operator": constraint.get("operator"),
                "requested": constraint.get("threshold"),
                "offered": round(suggested_relaxation(metric_name, best), 3),
                "relaxable": metric_name not in non_relaxable,
                "given": given,
            })

    return {
        "seed": seed,
        "needs_operator": needs_operator,
        "auto_switched_to": auto_switched_to,
        "chosen_algorithm": chosen_algorithm,
        "per_algorithm": per_algorithm,
        "suggested_relaxations": suggested_relaxations,
    }


def get_warm_start_context(
    parsed_intent: Dict[str, Any],
    top_k: int = 3,
    *,
    seed: int = SEED,
    algorithm: str = DEFAULT_ALGORITHM,
) -> str:
    """
    Filters the historical dataset based on hard constraints and primary objective,
    returning the best historical weight combinations for a 'Warm Start'.
    Constraints are matched in mean operator units (avg_*), matching the verifier.
    """
    valid_df = filter_runs(seed, algorithm)
    filtered_df = apply_hard_constraints(valid_df, parsed_intent.get("hard_constraints", []))

    applied_constraints = [
        _constraint_label(constraint)
        for constraint in parsed_intent.get("hard_constraints", [])
        if (constraint.get("metric") or "").lower() in METRIC_COLS
    ]

    used_fallback = False
    if filtered_df.empty:
        used_fallback = True
        filtered_df = valid_df.copy()

    primary = (parsed_intent.get("primary_objective") or "").lower()
    if primary in METRIC_COLS and not filtered_df.empty:
        target_col = METRIC_COLS[primary]
        # Security: higher is better. Cost/latency: lower is better.
        ascending = primary not in HIGHER_IS_BETTER
        filtered_df = filtered_df.sort_values(by=target_col, ascending=ascending)

    prefix = ""
    if used_fallback:
        constraints_txt = (
            ", ".join(applied_constraints)
            if applied_constraints
            else "the active hard constraints"
        )
        prefix = (
            f"WARNING: 0 historical runs satisfied the hard constraints ({constraints_txt}). "
            "Showing the best runs by primary objective instead. "
        )
        observed = _best_observed_lines(valid_df, parsed_intent)
        if observed:
            prefix += " ".join(observed) + " "
        prefix += "Weight search may not close this gap. "

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
