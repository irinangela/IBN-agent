import sys
import os
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes import weight_proposer_node

@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not set. Skipping proposer tests.")
def test_proposer_weights_sum_to_one():
    """The proposer returns valid weights summing to 1.0."""

    mock_intent = {
        "primary_objective": "latency",
        "hard_constraints": [
            {"metric": "cost", "operator": "<=", "threshold": 500.0}
        ],
        "soft_preferences": [],
        "non_relaxable_constraints": [],
        "relaxation_order": ["security"]
    }
    
    mock_state = {
        "active_parsed_intent": mock_intent,
        "total_input_tokens": 0,
        "total_output_tokens": 0
    }
    
    result = weight_proposer_node(mock_state)
    
    weights = result["current_weights"]
    total_weight = sum(weights.values())

    assert "w1" in weights
    assert "w2" in weights
    assert "w3" in weights
    assert isinstance(result["reasoning"], str)
    assert len(result["historical_context"]) > 0 
    
    assert round(total_weight, 3) == 1.0