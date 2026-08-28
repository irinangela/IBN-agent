import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes import verifier_node

def test_verifier_perfect_success():
    """Valid mean operator-unit metrics pass the thresholds."""
    
    mock_intent = {
        "hard_constraints": [
            {"metric": "latency", "operator": "<=", "threshold": 1000.0},
            {"metric": "cost", "operator": "<=", "threshold": 600.0},
        ]
    }
    
    mock_state = {
        "active_parsed_intent": mock_intent, 
        "simulation_results": {
            "failures": 0,
            "avg_cost": 450.0,
            "avg_security": 12.0,
            "avg_latency": 900.0,
        }
    }

    result = verifier_node(mock_state)  

    assert result["constraints_satisfied"] is True
    assert "SUCCESS" in result["verifier_feedback"]

    
def test_verifier_sla_violation():
    """A mean latency above the operator threshold fails the verifier."""

    mock_intent = {
        "hard_constraints": [
            {"metric": "latency", "operator": "<=", "threshold": 900.0}
        ]
    }
    
    mock_state = {
        "active_parsed_intent": mock_intent, 
        "simulation_results": {
            "failures": 0,
            "avg_cost": 500.0,
            "avg_security": 10.0,
            "avg_latency": 1200.0,
        }
    }
    
    result = verifier_node(mock_state)
    
    assert result["constraints_satisfied"] is False
    assert "Violated latency" in result["verifier_feedback"]


def test_verifier_optimizer_failure():
    """Physical placement failures immediately fail the verifier."""

    mock_intent = {
        "hard_constraints": [
            {"metric": "cost", "operator": "<=", "threshold": 1000.0}
        ]
    }
    
    mock_state = {
        "active_parsed_intent": mock_intent,
        "simulation_results": {
            "failures": 3,
            "avg_cost": 400.0,
            "avg_security": 10.0,
            "avg_latency": 900.0,
        }
    }
    
    result = verifier_node(mock_state)
    
    assert result["constraints_satisfied"] is False
    assert "CRITICAL FAILURE" in result["verifier_feedback"]


def test_verifier_unrecognized_metric():
    """Unknown metrics must surface as violations, not silent success."""

    mock_intent = {
        "hard_constraints": [
            {"metric": "energy", "operator": "<=", "threshold": 10.0}
        ]
    }

    mock_state = {
        "active_parsed_intent": mock_intent,
        "simulation_results": {
            "failures": 0,
            "avg_cost": 400.0,
            "avg_security": 10.0,
            "avg_latency": 900.0,
        }
    }

    result = verifier_node(mock_state)

    assert result["constraints_satisfied"] is False
    assert "UNRECOGNIZED METRIC" in result["verifier_feedback"]


def test_verifier_security_minimum():
    """Security uses >= (higher is better) against avg_security."""

    mock_intent = {
        "hard_constraints": [
            {"metric": "security", "operator": ">=", "threshold": 8.0}
        ]
    }

    fail_state = {
        "active_parsed_intent": mock_intent,
        "simulation_results": {
            "failures": 0,
            "avg_cost": 400.0,
            "avg_security": 5.0,
            "avg_latency": 900.0,
        }
    }
    fail_result = verifier_node(fail_state)
    assert fail_result["constraints_satisfied"] is False
    assert "Violated security" in fail_result["verifier_feedback"]

    pass_state = {
        "active_parsed_intent": mock_intent,
        "simulation_results": {
            "failures": 0,
            "avg_cost": 400.0,
            "avg_security": 12.0,
            "avg_latency": 900.0,
        }
    }
    pass_result = verifier_node(pass_state)
    assert pass_result["constraints_satisfied"] is True

def test_unsupported_equality_operator():
    """Equality operators are not supported."""
    mock_intent = {
        "hard_constraints": [
            {"metric": "cost", "operator": "==", "threshold": 400.0}
        ]
    }
    mock_state = {
        "active_parsed_intent": mock_intent,
        "simulation_results": {
            "failures": 0,
            "avg_cost": 400.0,
            "avg_security": 10.0,
            "avg_latency": 900.0,
        }
    }
    result = verifier_node(mock_state)
    assert result["constraints_satisfied"] is False
    assert "UNRECOGNIZED OPERATOR" in result["verifier_feedback"]

