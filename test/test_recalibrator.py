import json
import sys
import os
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes import recalibrator_node

def run_recalibrator_test(test_name: str, mock_state: dict):
    print(f"\n{'='*70}")
    print(f"RUNNING TEST: {test_name}")
    print(f"{'='*70}")
    
    print("\n[INPUT] ACTIVE INTENT:")
    print(json.dumps(mock_state["active_parsed_intent"], indent=2))
    print(f"\n[INPUT] CURRENT WEIGHTS: {mock_state['current_weights']}")
    print(f"[INPUT] VERIFIER FEEDBACK:\n{mock_state['verifier_feedback']}")
    print(f"[INPUT] HISTORY LENGTH: {len(mock_state.get('recalibration_history', []))}")
    print("-" * 70)
    
    try:
        result_state = recalibrator_node(mock_state)
        
        print("\n[OUTPUT] NEW PROPOSED WEIGHTS:")
        print(json.dumps(result_state.get("current_weights"), indent=2))
        
        print("\n[OUTPUT] REASONING:")
        print(result_state.get("reasoning"))
        
        print("\n[OUTPUT] UPDATED ACTIVE INTENT (Check if constraints were dropped!):")
        print(json.dumps(result_state.get("active_parsed_intent"), indent=2))
        
        print(f"\n[OUTPUT] NEW HISTORY LENGTH: {len(result_state.get('recalibration_history', []))}")
        
    except Exception as e:
        import traceback
        print(f"!!! Error during recalibration !!!")
        print(traceback.format_exc())

if __name__ == "__main__":
    
    # TEST CASE 1: Minor SLA Violation -> Fine Tuning
    intent_1 = {
        "primary_objective": "cost",
        "hard_constraints": [
            {"metric": "latency", "operator": "<=", "threshold": 40.0}
        ],
        "soft_preferences": [],
        "non_relaxable_constraints": [],
        "relaxation_order": ["latency"]
    }
    
    state_1 = {
        "original_parsed_intent": intent_1,
        "active_parsed_intent": intent_1,
        "current_weights": {"w1": 0.8, "w2": 0.1, "w3": 0.1},
        "verifier_feedback": "FAILED CONSTRAINTS:\n- Violated latency: Achieved 45.000, but required <= 40.0.",
        "recalibration_history": [],
        "iteration_count": 1
    }
    run_recalibrator_test("Test Case 1 (Minor Violation, Expect Weight Adjustment)", state_1)

    # TEST CASE 2: Impossible SLA -> Constraint Relaxation
    intent_2 = {
        "primary_objective": "cost",
        "hard_constraints": [
            {"metric": "latency", "operator": "<=", "threshold": 1.0} 
        ],
        "soft_preferences": [],
        "non_relaxable_constraints": [],
        "relaxation_order": ["latency"]
    }
    
    state_2 = {
        "original_parsed_intent": intent_2,
        "active_parsed_intent": intent_2,
        "current_weights": {"w1": 0.5, "w2": 0.1, "w3": 0.4},
        "verifier_feedback": "FAILED CONSTRAINTS:\n- Violated latency: Achieved 35.000, but required <= 1.0.",
        "recalibration_history": [
            "Attempted Weights: {'w1': 0.8, 'w2': 0.1, 'w3': 0.1} | Result: FAILED CONSTRAINTS:\n- Violated latency: Achieved 40.000, but required <= 1.0."
        ],
        "iteration_count": 2
    }
    run_recalibrator_test("Test Case 2 (Impossible SLA, Expect Constraint Drop)", state_2)