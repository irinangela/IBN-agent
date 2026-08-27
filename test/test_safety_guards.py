import sys
import os
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes import (
    _clip_and_normalize_weights,
    apply_constraint_relaxation,
    distinct_failed_weight_count,
    MIN_DISTINCT_FAILED_WEIGHTS,
)


def test_clip_prevents_raw_jump_to_one():
    """Proposed (1,0,0) from a balanced baseline cannot remain w1=1.0 after clip+normalize."""
    weights = _clip_and_normalize_weights(
        w1=1.0, w2=0.0, w3=0.0,
        base_w1=0.33, base_w2=0.33, base_w3=0.34,
    )
    assert weights["w1"] < 1.0
    assert weights["w2"] > 0.0
    assert weights["w3"] > 0.0
    assert abs(sum(weights.values()) - 1.0) < 0.002


def test_clip_allows_small_adjustment():
    weights = _clip_and_normalize_weights(
        w1=0.40, w2=0.30, w3=0.30,
        base_w1=0.33, base_w2=0.33, base_w3=0.34,
    )
    assert abs(weights["w1"] - 0.40) < 0.02
    assert abs(sum(weights.values()) - 1.0) < 0.002


def _latency_intent():
    return {
        "hard_constraints": [
            {"metric": "latency", "operator": "<=", "threshold": 900.0}
        ],
        "non_relaxable_constraints": [],
        "relaxation_order": ["latency"],
    }


def test_relaxation_blocks_non_relaxable():
    intent = {
        "hard_constraints": [
            {"metric": "latency", "operator": "<=", "threshold": 900.0}
        ],
        "non_relaxable_constraints": ["latency"],
        "relaxation_order": ["latency"],
    }
    updated, msg = apply_constraint_relaxation(
        copy.deepcopy(intent), "latency", 1100.0
    )
    assert updated["hard_constraints"][0]["threshold"] == 900.0
    assert "NON-RELAXABLE" in msg


def test_relaxation_blocks_metric_outside_order():
    intent = {
        "hard_constraints": [
            {"metric": "latency", "operator": "<=", "threshold": 900.0},
            {"metric": "security", "operator": ">=", "threshold": 8.0},
        ],
        "non_relaxable_constraints": [],
        "relaxation_order": ["latency"],
    }
    updated, msg = apply_constraint_relaxation(
        copy.deepcopy(intent), "security", 6.0
    )
    assert len(updated["hard_constraints"]) == 2
    assert updated["hard_constraints"][1]["threshold"] == 8.0
    assert "not in relaxation_order" in msg


def test_relaxation_loosens_threshold_instead_of_deleting():
    original = _latency_intent()
    updated, msg = apply_constraint_relaxation(
        copy.deepcopy(original), "latency", 1100.0, original_intent=original
    )
    assert len(updated["hard_constraints"]) == 1
    assert updated["hard_constraints"][0]["metric"] == "latency"
    assert updated["hard_constraints"][0]["threshold"] == 1100.0
    assert "latency" in updated["relaxation_order"]
    assert "relaxed from 900.0 to 1100.0" in msg
    assert "+22%" in msg


def test_relaxation_refuses_tighter_threshold():
    updated, msg = apply_constraint_relaxation(
        copy.deepcopy(_latency_intent()), "latency", 800.0
    )
    assert updated["hard_constraints"][0]["threshold"] == 900.0
    assert "not strictly looser" in msg


def test_relaxation_refuses_missing_threshold():
    updated, msg = apply_constraint_relaxation(
        copy.deepcopy(_latency_intent()), "latency", None
    )
    assert updated["hard_constraints"][0]["threshold"] == 900.0
    assert "did not propose a relaxed_threshold" in msg


def test_relaxation_noop_when_metric_has_no_hard_constraint():
    """Relaxing a preference metric that is not a hard constraint is a no-op."""
    intent = {
        "hard_constraints": [
            {"metric": "latency", "operator": "<=", "threshold": 800.0}
        ],
        "non_relaxable_constraints": ["latency"],
        "relaxation_order": ["cost", "security"],
    }
    updated, msg = apply_constraint_relaxation(
        copy.deepcopy(intent), "cost", 600.0
    )
    assert len(updated["hard_constraints"]) == 1
    assert updated["hard_constraints"][0]["metric"] == "latency"
    assert "SYSTEM OVERRIDE" in msg
    assert "no hard constraint" in msg.lower()


def test_distinct_failed_weight_count_rounds_and_dedupes():
    history = [
        {"w1": 0.333, "w2": 0.333, "w3": 0.334},
        {"w1": 0.3334, "w2": 0.333, "w3": 0.334},
        {"w1": 0.5, "w2": 0.25, "w3": 0.25},
    ]
    assert distinct_failed_weight_count(history) == 2
    assert MIN_DISTINCT_FAILED_WEIGHTS == 3


def test_in_loop_gate_requires_three_distinct_vectors():
    two = [
        {"w1": 0.1, "w2": 0.2, "w3": 0.7},
        {"w1": 0.2, "w2": 0.2, "w3": 0.6},
    ]
    three = two + [{"w1": 0.3, "w2": 0.2, "w3": 0.5}]
    assert distinct_failed_weight_count(two) < MIN_DISTINCT_FAILED_WEIGHTS
    assert distinct_failed_weight_count(three) >= MIN_DISTINCT_FAILED_WEIGHTS


def test_relaxation_loosens_security_by_lowering_floor():
    intent = {
        "hard_constraints": [
            {"metric": "security", "operator": ">=", "threshold": 12.0}
        ],
        "non_relaxable_constraints": [],
        "relaxation_order": ["security"],
    }
    updated, msg = apply_constraint_relaxation(
        copy.deepcopy(intent), "security", 10.0, original_intent=intent
    )
    assert updated["hard_constraints"][0]["threshold"] == 10.0
    assert "relaxed from 12.0 to 10.0" in msg
