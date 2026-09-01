import sys
import os

import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.config import WARM_START_MODE, WARM_START_SEEDS, SEED
from utils.retriever import (
    apply_hard_constraints,
    average_by_weight,
    best_known_bounds,
    constraint_violates_bound,
    feasibility_report,
    filter_runs,
    get_warm_start_context,
    is_jointly_feasible,
    suggested_relaxation,
)


def _first_metric(context: str, label: str) -> float:
    for line in context.splitlines():
        if f"{label}:" in line:
            chunk = line.split(f"{label}:", 1)[1].split(",")[0]
            return float(chunk.replace("ms", "").strip())
    raise AssertionError(f"{label} not found in:\n{context}")


def test_warm_start_warns_when_no_historical_run_meets_sla():
    """Impossible latency SLA must surface the fallback"""

    context = get_warm_start_context({
        "primary_objective": "latency",
        "hard_constraints": [
            {"metric": "latency", "operator": "<=", "threshold": 1.0}
        ],
    })
    assert context.startswith("WARNING: 0 historical runs satisfied the hard constraints")
    assert "latency <= 1.0" in context
    assert "Best observed latency:" in context
    assert "Historical Successful Runs" in context


def test_warm_start_omits_warning_when_runs_meet_sla():
    context = get_warm_start_context({
        "primary_objective": "latency",
        "hard_constraints": [
            {"metric": "latency", "operator": "<=", "threshold": 10000.0}
        ],
    })
    assert not context.startswith("WARNING:")
    assert "Historical Successful Runs" in context


def test_security_warm_start_uses_highest_security():
    """Primary objective security must sort descending on neighborhood means."""
    runs = filter_runs(algorithm="best_fit")
    best = float(average_by_weight(runs)["avg_security"].max())
    context = get_warm_start_context({
        "primary_objective": "security",
        "hard_constraints": [],
    }, algorithm="best_fit")
    assert not context.startswith("WARNING:")
    assert _first_metric(context, "Security") == pytest.approx(best, abs=0.001)
    assert "seed=" not in context


def test_warm_start_ranks_neighborhood_mean_not_a_single_seed():
    """A lucky seed must not beat a weight that is better on average."""
    runs = filter_runs(algorithm="best_fit")
    averaged = average_by_weight(runs)
    best_mean = float(averaged["avg_cost"].min())
    lucky = float(runs["avg_cost"].min())
    context = get_warm_start_context({
        "primary_objective": "cost",
        "hard_constraints": [],
    }, algorithm="best_fit")
    assert _first_metric(context, "Cost") == pytest.approx(best_mean, abs=0.001)
    if lucky < best_mean - 0.001:
        assert _first_metric(context, "Cost") != pytest.approx(lucky, abs=0.001)
    assert "seed=" not in context


def test_warm_start_algorithm_argument_selects_rollout():
    intent = {"primary_objective": "cost", "hard_constraints": []}
    best_fit_ctx = get_warm_start_context(intent, algorithm="best_fit")
    rollout = filter_runs(algorithm="app_rollout")
    if rollout.empty:
        pytest.skip("warm-start dataset has no app_rollout rows yet")
    rollout_ctx = get_warm_start_context(intent, algorithm="app_rollout")
    assert best_fit_ctx != rollout_ctx
    assert "Historical Successful Runs" in best_fit_ctx
    assert "Historical Successful Runs" in rollout_ctx


def test_filter_runs_is_per_algorithm():
    best_fit = filter_runs(algorithm="best_fit")
    assert not best_fit.empty
    assert set(best_fit["algorithm"].unique()) == {"best_fit"}
    rollout = filter_runs(algorithm="app_rollout")
    if rollout.empty:
        pytest.skip("warm-start dataset has no app_rollout rows yet")
    assert set(rollout["algorithm"].unique()) == {"app_rollout"}


def test_near_neighbor_retrieval_excludes_live_seed():
    """Warm-start must not read the live instance in near_neighbor mode."""
    if WARM_START_MODE != "near_neighbor":
        pytest.skip("WARM_START_MODE is not near_neighbor")
    runs = filter_runs(algorithm="best_fit")
    assert not runs.empty
    assert SEED not in set(runs["seed"].astype(int).unique())
    assert set(runs["seed"].astype(int).unique()).issubset(set(WARM_START_SEEDS))


def test_is_jointly_feasible_with_fixture_frame():
    core = pd.DataFrame({
        "avg_cost": [4.0, 2.0],
        "avg_security": [10.0, 8.0],
        "avg_latency": [900.0, 800.0],
    })
    hard = [{"metric": "cost", "operator": "<=", "threshold": 2.5}]
    assert is_jointly_feasible(core, hard) is True
    assert is_jointly_feasible(core, [
        {"metric": "cost", "operator": "<=", "threshold": 1.0}
    ]) is False
    matched = apply_hard_constraints(core, hard)
    assert len(matched) == 1
    assert float(matched.iloc[0]["avg_cost"]) == 2.0


def test_best_known_bounds_respect_direction():
    core = pd.DataFrame({
        "avg_cost": [4.0, 2.0],
        "avg_security": [10.0, 8.0],
        "avg_latency": [900.0, 800.0],
    })
    bounds = best_known_bounds(core)
    assert bounds["cost"] == 2.0
    assert bounds["latency"] == 800.0
    assert bounds["security"] == 10.0


def test_constraint_violates_bound_and_suggested_margin():
    bounds = {"cost": 3.73, "security": 12.0}
    assert constraint_violates_bound(
        {"metric": "cost", "operator": "<=", "threshold": 2.0}, bounds
    )
    assert not constraint_violates_bound(
        {"metric": "cost", "operator": "<=", "threshold": 4.0}, bounds
    )
    assert constraint_violates_bound(
        {"metric": "security", "operator": ">=", "threshold": 20.0}, bounds
    )
    assert suggested_relaxation("cost", 3.73) == pytest.approx(3.73 * 1.05)
    assert suggested_relaxation("security", 12.0) == pytest.approx(12.0 * 0.95)


def test_feasibility_report_impossible_sla_needs_operator():
    report = feasibility_report({
        "primary_objective": "latency",
        "hard_constraints": [
            {"metric": "latency", "operator": "<=", "threshold": 1.0}
        ],
        "non_relaxable_constraints": [],
    })
    assert report["needs_operator"] is True
    assert report["auto_switched_to"] is None
    assert report["per_algorithm"]["best_fit"]["jointly_feasible"] is False
    assert report["per_algorithm"]["app_rollout"]["jointly_feasible"] is False
    assert "latency" in report["per_algorithm"]["best_fit"]["bounds"]
    assert any(item["metric"] == "latency" for item in report["suggested_relaxations"])
    latency_offer = next(
        item for item in report["suggested_relaxations"] if item["metric"] == "latency"
    )
    assert latency_offer["relaxable"] is True
    assert latency_offer.get("given") == []


def test_feasibility_report_easy_sla_stays_on_best_fit():
    report = feasibility_report({
        "primary_objective": "latency",
        "hard_constraints": [
            {"metric": "latency", "operator": "<=", "threshold": 10000.0}
        ],
    })
    assert report["needs_operator"] is False
    assert report["chosen_algorithm"] == "best_fit"
    assert report["auto_switched_to"] is None
    assert report["per_algorithm"]["best_fit"]["jointly_feasible"] is True


def test_feasibility_report_auto_switches_when_only_rollout_is_jointly_feasible():
    best_fit = filter_runs(algorithm="best_fit")
    rollout = filter_runs(algorithm="app_rollout")
    if best_fit.empty or rollout.empty:
        pytest.skip("Need both algorithms in the warm-start dataset")
    bf_min = float(best_fit["avg_cost"].min())
    ar_min = float(rollout["avg_cost"].min())
    if not (ar_min < bf_min):
        pytest.skip("There is not a cheaper app_rollout cost than best_fit")
    threshold = (ar_min + bf_min) / 2.0
    report = feasibility_report({
        "primary_objective": "cost",
        "hard_constraints": [
            {"metric": "cost", "operator": "<=", "threshold": threshold}
        ],
    })
    assert report["needs_operator"] is False
    assert report["auto_switched_to"] == "app_rollout"
    assert report["chosen_algorithm"] == "app_rollout"
    assert report["per_algorithm"]["best_fit"]["jointly_feasible"] is False
    assert report["per_algorithm"]["app_rollout"]["jointly_feasible"] is True


def _conditional_best(metric: str, others: list) -> float:
    values = []
    for algorithm in ("best_fit", "app_rollout"):
        filtered = apply_hard_constraints(filter_runs(algorithm=algorithm), others)
        bound = best_known_bounds(filtered).get(metric)
        if bound is not None:
            values.append(float(bound))
    assert values, f"No best-known bound for {metric} given {others}"
    if metric == "security":
        return max(values)
    return min(values)


def test_feasibility_report_offers_conditional_relaxations():
    """Independently feasible SLAs that never co-occur still get numeric offers."""

    latency_c = {"metric": "latency", "operator": "<=", "threshold": 900.0}
    cost_c = {"metric": "cost", "operator": "<=", "threshold": 450.0}
    report = feasibility_report({
        "primary_objective": "latency",
        "hard_constraints": [latency_c, cost_c],
        "non_relaxable_constraints": [],
    })
    if not report["needs_operator"]:
        pytest.skip("No best-known bound for latency or cost given the constraints")

    by_metric = {item["metric"]: item for item in report["suggested_relaxations"]}
    assert "latency" in by_metric
    assert "cost" in by_metric

    expected_lat = suggested_relaxation(
        "latency", _conditional_best("latency", [cost_c])
    )
    expected_cost = suggested_relaxation(
        "cost", _conditional_best("cost", [latency_c])
    )
    assert by_metric["latency"]["offered"] == pytest.approx(round(expected_lat, 3))
    assert by_metric["cost"]["offered"] == pytest.approx(round(expected_cost, 3))
    assert by_metric["latency"]["offered"] > 900.0
    assert by_metric["cost"]["offered"] > 450.0
    assert by_metric["latency"]["relaxable"] is True
    assert by_metric["cost"]["relaxable"] is True
    assert any("cost" in label for label in by_metric["latency"]["given"])
    assert any("latency" in label for label in by_metric["cost"]["given"])

    lat_bounds = [
        info["bounds"]["latency"]
        for info in report["per_algorithm"].values()
        if (info.get("bounds") or {}).get("latency") is not None
    ]
    assert lat_bounds
    indep_lat = min(lat_bounds)
    assert indep_lat <= 900.0
    assert by_metric["latency"]["offered"] > indep_lat


def test_feasibility_report_falls_back_when_other_constraints_match_nothing():
    report = feasibility_report({
        "primary_objective": "latency",
        "hard_constraints": [
            {"metric": "latency", "operator": "<=", "threshold": 1.0},
            {"metric": "cost", "operator": "<=", "threshold": 1.0},
        ],
        "non_relaxable_constraints": [],
    })
    assert report["needs_operator"] is True
    by_metric = {item["metric"]: item for item in report["suggested_relaxations"]}
    assert "latency" in by_metric
    assert "cost" in by_metric
    assert by_metric["latency"]["given"] == []
    assert by_metric["cost"]["given"] == []


def test_feasibility_report_marks_non_relaxable_in_pareto_suggestions():
    report = feasibility_report({
        "primary_objective": "latency",
        "hard_constraints": [
            {"metric": "latency", "operator": "<=", "threshold": 900.0},
            {"metric": "cost", "operator": "<=", "threshold": 450.0},
        ],
        "non_relaxable_constraints": ["latency"],
    })
    if not report["needs_operator"]:
        pytest.skip("No best-known bound for latency or cost given the constraints")
    by_metric = {item["metric"]: item for item in report["suggested_relaxations"]}
    assert by_metric["latency"]["relaxable"] is False
    assert by_metric["cost"]["relaxable"] is True
