import sys
import os
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes import intent_parser_node, get_relaxation_order
from nodes.parser import parse_is_checked, STRUCTURED_PARSE_FAILED_MESSAGE

def test_get_relaxation_order_keeps_only_hard_metrics():
    intent = {
        "hard_constraints": [
            {"metric": "cost", "operator": "<=", "threshold": 500.0}
        ],
        "relaxation_order": ["security", "cost", "latency"],
    }
    updated = get_relaxation_order(intent)
    assert updated["relaxation_order"] == ["cost"]


def test_parse_is_checked_requires_named_metrics():
    guessed = {
        "primary_objective": "latency",
        "hard_constraints": [
            {"metric": "latency", "operator": "<=", "threshold": 1000.0}
        ],
    }
    assert not parse_is_checked(
        guessed,
        "Optimize for low bananas. Average bananas must stay under 1000ms.",
    )
    assert parse_is_checked(
        guessed,
        "Optimize for low latency. Average latency must stay under 1000ms.",
    )


def test_get_relaxation_order_empties_when_no_overlap():
    intent = {
        "hard_constraints": [
            {"metric": "latency", "operator": "<=", "threshold": 1000.0}
        ],
        "relaxation_order": ["security", "cost"],
    }
    updated = get_relaxation_order(intent)
    assert updated["relaxation_order"] == []


def test_parser_structured_failure_asks_for_rewrite_instead_of_crashing(monkeypatch):
    from nodes import parser as parser_mod

    monkeypatch.setattr(
        parser_mod,
        "invoke_structured",
        lambda *args, **kwargs: {"parsed": None, "raw": None},
    )
    monkeypatch.setattr(parser_mod, "structured_usage", lambda response: (1, 1))
    result = intent_parser_node({
        "user_intent": "We need average latency no worse than 521 ms.",
        "total_input_tokens": 0,
        "total_output_tokens": 0,
    })
    assert result["parse_failed"] is True
    assert result["needs_operator"] is True
    assert STRUCTURED_PARSE_FAILED_MESSAGE in result["parse_feedback"]
    assert result["original_parsed_intent"] == {}
    assert result["total_input_tokens"] == 1


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not set. Skipping parser tests.")
def test_parser_basic_intent():
    """Parser extracts objectives and thresholds are expressed in operator units."""

    mock_state = {
        "user_intent": "Minimize latency. Cost must be strictly under 500. If needed, relax the cost constraint.",
        "total_input_tokens": 0,
        "total_output_tokens": 0
    }

    result = intent_parser_node(mock_state)

    orig_intent = result["original_parsed_intent"]
    act_intent = result["active_parsed_intent"]

    assert orig_intent["primary_objective"] == "latency"
    assert len(orig_intent["hard_constraints"]) == 1
    assert orig_intent["hard_constraints"][0]["metric"] == "cost"
    hard_metrics = {hc["metric"] for hc in orig_intent["hard_constraints"]}
    assert all(metric in hard_metrics for metric in orig_intent["relaxation_order"])
    assert "security" not in orig_intent["relaxation_order"]

    assert orig_intent["hard_constraints"][0]["threshold"] == 500.0
    assert act_intent["hard_constraints"][0]["threshold"] == 500.0
    assert act_intent["hard_constraints"][0]["threshold"] == orig_intent["hard_constraints"][0]["threshold"]

    assert result["total_input_tokens"] > 0
