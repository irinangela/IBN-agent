import sys
import os
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes import _clip_and_normalize_weights, apply_constraint_relaxation


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


def test_relaxation_blocks_non_relaxable():
    intent = {
        "hard_constraints": [
            {"metric": "latency", "operator": "<=", "threshold": 900.0}
        ],
        "non_relaxable_constraints": ["latency"],
        "relaxation_order": ["latency"],
    }
    updated, msg = apply_constraint_relaxation(copy.deepcopy(intent), "latency")
    assert len(updated["hard_constraints"]) == 1
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
    updated, msg = apply_constraint_relaxation(copy.deepcopy(intent), "security")
    assert len(updated["hard_constraints"]) == 2
    assert "not in relaxation_order" in msg


def test_relaxation_allows_metric_in_order():
    intent = {
        "hard_constraints": [
            {"metric": "latency", "operator": "<=", "threshold": 900.0}
        ],
        "non_relaxable_constraints": [],
        "relaxation_order": ["latency"],
    }
    updated, msg = apply_constraint_relaxation(copy.deepcopy(intent), "latency")
    assert len(updated["hard_constraints"]) == 0
    assert "[IMPORTANT]: Relaxed" in msg
    assert "latency" not in updated["relaxation_order"]


def test_relaxation_noop_when_metric_has_no_hard_constraint():
    """Relaxing a preference metric that is not a hard constraint is a no-op."""
    intent = {
        "hard_constraints": [
            {"metric": "latency", "operator": "<=", "threshold": 800.0}
        ],
        "non_relaxable_constraints": ["latency"],
        "relaxation_order": ["cost", "security"],
    }
    updated, msg = apply_constraint_relaxation(copy.deepcopy(intent), "cost")
    assert len(updated["hard_constraints"]) == 1
    assert updated["hard_constraints"][0]["metric"] == "latency"
    assert "SYSTEM OVERRIDE" in msg
    assert "no hard constraint" in msg.lower()
