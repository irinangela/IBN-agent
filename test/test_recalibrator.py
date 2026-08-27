import sys
import os
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes import (
    recalibrator_node,
    select_best_attempt,
    format_best_so_far_hint,
    violated_metrics_from_feedback,
    format_relaxation_eligibility,
    build_attempt_record,
)


def test_select_best_attempt_prefers_lower_latency():
    """The 880ms attempt beats 888ms even when w3 is lower."""
    attempts = [
        {"attempt": 1, "w1": 0.05, "w2": 0.0, "w3": 0.95, "avg_latency": 880.513},
        {"attempt": 2, "w1": 0.0, "w2": 0.0, "w3": 1.0, "avg_latency": 888.513},
    ]
    best = select_best_attempt(attempts, "latency")
    assert best["attempt"] == 1
    assert best["w3"] == 0.95


def test_select_best_attempt_prefers_higher_security():
    attempts = [
        {"attempt": 1, "w1": 0.2, "w2": 0.5, "w3": 0.3, "avg_security": 6.5},
        {"attempt": 2, "w1": 0.1, "w2": 0.7, "w3": 0.2, "avg_security": 12.0},
    ]
    best = select_best_attempt(attempts, "security")
    assert best["attempt"] == 2


def test_select_best_attempt_skips_legacy_string_rows():
    attempts = [
        "Attempted Weights: {'w1': 0.8} | Result: FAILED",
        {"attempt": 2, "w1": 0.05, "w2": 0.0, "w3": 0.95, "avg_latency": 880.5},
    ]
    best = select_best_attempt(attempts, "latency")
    assert best["attempt"] == 2


def test_format_best_so_far_says_reverse_when_worse():
    attempts = [
        {"attempt": 1, "w1": 0.05, "w2": 0.0, "w3": 0.95, "avg_latency": 880.513},
        {"attempt": 2, "w1": 0.0, "w2": 0.0, "w3": 1.0, "avg_latency": 888.513},
    ]
    hint = format_best_so_far_hint(attempts, "latency")
    assert "Attempt 1" in hint
    assert "880.513" in hint
    assert "Reverse toward this point" in hint
    assert "worse" in hint.lower()


def test_violated_metrics_from_feedback():
    feedback = (
        "FAILED CONSTRAINTS:\n"
        "- Violated latency: Achieved 880.513 ms (avg per app), but required <= 800.0."
    )
    assert violated_metrics_from_feedback(feedback) == ["latency"]


def test_relaxation_eligibility_when_only_sla_is_non_relaxable():
    intent = {
        "hard_constraints": [
            {"metric": "latency", "operator": "<=", "threshold": 800.0}
        ],
        "non_relaxable_constraints": ["latency"],
        "relaxation_order": ["cost", "security"],
    }
    note = format_relaxation_eligibility(intent)
    assert "No relaxable hard constraint remains" in note
    assert "relax_metric" in note


def test_relaxation_loosening_not_dropping():
    note = format_relaxation_eligibility({
        "hard_constraints": [
            {"metric": "latency", "operator": "<=", "threshold": 900.0}
        ],
        "non_relaxable_constraints": [],
        "relaxation_order": ["latency"],
    })
    assert "loosened" in note
    assert "not dropped" in note


def test_build_attempt_record_stores_operator_unit_metrics():
    record = build_attempt_record(
        attempt_num=1,
        weights={"w1": 0.05, "w2": 0.0, "w3": 0.95},
        results={"avg_cost": 1332.15, "avg_security": 6.58, "avg_latency": 880.513},
        feedback="FAILED CONSTRAINTS:\n- Violated latency: Achieved 880.513 ms",
    )
    assert record["attempt"] == 1
    assert record["avg_latency"] == 880.513
    assert record["w3"] == 0.95


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Requires Anthropic API Key")
def test_recalibration_minor_violation():
    """A minor violation fine-tunes weights but DOES NOT drop constraints."""

    intent_1 = {
        "primary_objective": "cost",
        "hard_constraints": [
            {"metric": "latency", "operator": "<=", "threshold": 900.0}
        ],
        "relaxation_order": ["latency"]
    }
    
    state_1 = {
        "original_parsed_intent": intent_1,
        "active_parsed_intent": intent_1,
        "current_weights": {"w1": 0.8, "w2": 0.1, "w3": 0.1},
        "simulation_results": {
            "avg_cost": 500.0,
            "avg_security": 10.0,
            "avg_latency": 1200.0,
        },
        "historical_context": "Historical Successful Runs (mean operator units):\n",
        "verifier_feedback": "FAILED CONSTRAINTS:\n- Violated latency: Achieved 1200.000 ms (avg per app), but required <= 900.0.",
        "recalibration_history": [],
        "iteration_count": 1,
        "total_input_tokens": 0, "total_output_tokens": 0
    }
    
    result = recalibrator_node(state_1)

    weights = result["current_weights"]
    assert round(sum(weights.values()), 3) == 1.0

    assert len(result["recalibration_history"]) == 1
    assert result["recalibration_history"][0]["avg_latency"] == 1200.0

    assert len(result["active_parsed_intent"]["hard_constraints"]) == 1
    assert result["active_parsed_intent"]["hard_constraints"][0]["threshold"] == 900.0


# @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Requires Anthropic API Key")
# def test_recalibration_impossible_sla():
#     """An impossible violation triggers autonomous constraint dropping."""

#     intent_2 = {
#         "primary_objective": "cost",
#         "hard_constraints": [
#             {"metric": "latency", "operator": "<=", "threshold": 100.0}
#         ],
#         "relaxation_order": ["latency"]
#     }
    
#     state_2 = {
#         "original_parsed_intent": intent_2,
#         "active_parsed_intent": intent_2,
#         "current_weights": {"w1": 0.5, "w2": 0.1, "w3": 0.4},
#         "simulation_results": {
#             "avg_cost": 500.0,
#             "avg_security": 10.0,
#             "avg_latency": 1200.0,
#         },
#         "historical_context": "WARNING: 0 historical runs satisfied the hard constraints (latency <= 100.0).",
#         "verifier_feedback": "FAILED CONSTRAINTS:\n- Violated latency: Achieved 1200.000 ms (avg per app), but required <= 100.0.",
#         "recalibration_history": [
#             "Attempted Weights: {'w1': 0.8, 'w2': 0.1, 'w3': 0.1} | Result: FAILED"
#         ],
#         "iteration_count": 2,
#         "total_input_tokens": 0, "total_output_tokens": 0
#     }
    
#     result = recalibrator_node(state_2)
    
#     assert len(result["active_parsed_intent"]["hard_constraints"]) == 0
#     assert "[IMPORTANT]: Relaxed" in result["reasoning"]
