import sys
import os
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes import intent_parser_node

@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not set. Skipping parser tests.")
def test_parser_basic_intent():
    """Parser extracts objectives and  thresholds are expressed in operator units."""

    mock_state = {
        "user_intent": "Minimize latency. Cost must be strictly under 500. If needed, relax security.",
        "total_input_tokens": 0,
        "total_output_tokens": 0
    }
    
    result = intent_parser_node(mock_state)
    
    orig_intent = result["original_parsed_intent"]
    act_intent = result["active_parsed_intent"]
 
    assert orig_intent["primary_objective"] == "latency"
    assert len(orig_intent["hard_constraints"]) == 1
    assert orig_intent["hard_constraints"][0]["metric"] == "cost"
    assert "security" in orig_intent["relaxation_order"]
    

    assert orig_intent["hard_constraints"][0]["threshold"] == 500.0
    assert act_intent["hard_constraints"][0]["threshold"] == 500.0
    assert act_intent["hard_constraints"][0]["threshold"] == orig_intent["hard_constraints"][0]["threshold"]
    
    assert result["total_input_tokens"] > 0
