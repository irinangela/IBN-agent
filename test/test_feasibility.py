import copy
import sys
import os

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.feasibility import apply_operator_decision, feasibility_node
from utils.retriever import filter_runs


def _intent(**overrides):
    intent = {
        "primary_objective": "latency",
        "hard_constraints": [
            {"metric": "latency", "operator": "<=", "threshold": 10000.0}
        ],
        "soft_preferences": [],
        "non_relaxable_constraints": [],
        "relaxation_order": ["latency"],
    }
    intent.update(overrides)
    return intent


def test_feasibility_node_proceeds_on_easy_sla():
    result = feasibility_node({
        "active_parsed_intent": _intent(),
        "operator_decision": {},
    })
    assert result["needs_operator"] is False
    assert result["use_rollout"] is False
    assert result["feasibility_report"]["chosen_algorithm"] == "best_fit"
    assert "Proceeding with best_fit" in result["reasoning"]


def test_feasibility_node_needs_operator_on_impossible_sla():
    result = feasibility_node({
        "active_parsed_intent": _intent(
            hard_constraints=[
                {"metric": "latency", "operator": "<=", "threshold": 1.0}
            ]
        ),
        "operator_decision": {},
    })
    assert result["needs_operator"] is True
    assert result["use_rollout"] is False
    assert result["feasibility_report"]["needs_operator"] is True
    assert "Operator input required" in result["reasoning"]


def test_feasibility_node_auto_switch_sets_use_rollout():
    best_fit = filter_runs(algorithm="best_fit")
    rollout = filter_runs(algorithm="app_rollout")
    if best_fit.empty or rollout.empty:
        pytest.skip("Need both algorithms in the warm-start dataset")
    bf_min = float(best_fit["avg_cost"].min())
    ar_min = float(rollout["avg_cost"].min())
    if not (ar_min < bf_min):
        pytest.skip("There is not a cheaper app_rollout cost than best_fit")
    threshold = (ar_min + bf_min) / 2.0
    result = feasibility_node({
        "active_parsed_intent": _intent(
            primary_objective="cost",
            hard_constraints=[
                {"metric": "cost", "operator": "<=", "threshold": threshold}
            ],
        ),
        "operator_decision": {},
    })
    assert result["needs_operator"] is False
    assert result["use_rollout"] is True
    assert result["feasibility_report"]["auto_switched_to"] == "app_rollout"
    assert "Auto-switched" in result["reasoning"]


def test_apply_operator_decision_relax_updates_active_threshold_only():
    original = _intent(
        hard_constraints=[
            {"metric": "cost", "operator": "<=", "threshold": 2.0}
        ],
        non_relaxable_constraints=[],
    )
    state = {
        "original_parsed_intent": copy.deepcopy(original),
        "active_parsed_intent": copy.deepcopy(original),
        "feasibility_report": {
            "chosen_algorithm": "best_fit",
            "suggested_relaxations": [
                {
                    "metric": "cost",
                    "operator": "<=",
                    "requested": 2.0,
                    "offered": 3.9,
                    "relaxable": True,
                }
            ],
        },
    }
    updated = apply_operator_decision(
        state, {"action": "relax", "thresholds": {"cost": 3.9}}
    )
    assert updated["needs_operator"] is False
    assert updated["active_parsed_intent"]["hard_constraints"][0]["threshold"] == 3.9
    assert state["original_parsed_intent"]["hard_constraints"][0]["threshold"] == 2.0
    assert updated["operator_decision"]["action"] == "relax"


def test_apply_operator_decision_continue_leaves_thresholds():
    intent = _intent(
        hard_constraints=[
            {"metric": "cost", "operator": "<=", "threshold": 2.0}
        ]
    )
    updated = apply_operator_decision(
        {
            "active_parsed_intent": copy.deepcopy(intent),
            "feasibility_report": {"chosen_algorithm": "best_fit"},
        },
        {"action": "continue"},
    )
    assert updated["needs_operator"] is False
    assert updated["active_parsed_intent"]["hard_constraints"][0]["threshold"] == 2.0


def test_apply_operator_decision_skips_non_relaxable_metric():
    intent = _intent(
        hard_constraints=[
            {"metric": "latency", "operator": "<=", "threshold": 1.0}
        ],
        non_relaxable_constraints=["latency"],
    )
    updated = apply_operator_decision(
        {
            "active_parsed_intent": copy.deepcopy(intent),
            "feasibility_report": {
                "chosen_algorithm": "best_fit",
                "suggested_relaxations": [
                    {
                        "metric": "latency",
                        "requested": 1.0,
                        "offered": 800.0,
                        "relaxable": False,
                    }
                ],
            },
        },
        {"action": "relax", "thresholds": {"latency": 800.0}},
    )
    assert updated["active_parsed_intent"]["hard_constraints"][0]["threshold"] == 1.0


def test_feasibility_node_applies_injected_continue_decision():
    result = feasibility_node({
        "active_parsed_intent": _intent(
            hard_constraints=[
                {"metric": "latency", "operator": "<=", "threshold": 1.0}
            ]
        ),
        "operator_decision": {"action": "continue"},
    })
    assert result["needs_operator"] is False
    assert result["operator_decision"]["action"] == "continue"
    assert result["active_parsed_intent"]["hard_constraints"][0]["threshold"] == 1.0
