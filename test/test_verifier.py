import json
import sys
import os
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes import verifier_node

def run_verifier_test(test_name: str, hard_constraints: list, simulation_results: dict):
    print(f"\n{'='*70}")
    print(f"-- RUNNING TEST: {test_name} --")
    print(f"{'='*70}")
    
    mock_intent = {
        "primary_objective": "cost",
        "hard_constraints": hard_constraints,
        "soft_preferences": [],
        "non_relaxable_constraints": [],
        "relaxation_order": []
    }
    
    print("-- PARSED HARD CONSTRAINTS: --")
    print(json.dumps(hard_constraints, indent=2))
    
    print("\n-- SIMULATION RESULTS (From tools.py): --")
    print(json.dumps(simulation_results, indent=2))
    print("-" * 70)

    mock_state = {
        "user_intent": "Mock natural language intent...",
        "parsed_intent": mock_intent,
        "historical_context": "",
        "current_weights": {"w1": 0.5, "w2": 0.2, "w3": 0.3},
        "reasoning": "Mock reasoning",
        "simulation_results": simulation_results,
        "iteration_count": 1,
        "max_iterations": 5,
        "final_response": "",
        "constraints_satisfied": False,
        "verifier_feedback": ""
    }
    
    try:
        result_state = verifier_node(mock_state)
        
        print("-- VERIFIER OUTPUT: --")
        print(f"Constraints Satisfied: {result_state.get('constraints_satisfied')}")
        print(f"Feedback Message:\n{result_state.get('verifier_feedback')}")
        
    except Exception as e:
        import traceback
        print(f"!!! Error during verification !!!")
        print(traceback.format_exc())

if __name__ == "__main__":
    # TEST CASE 1: Perfect Success
    constraints_1 = [
        {"metric": "latency", "operator": "<=", "threshold": 50.0},
        {"metric": "cost", "operator": "<=", "threshold": 5.0}
    ]
    results_1 = {
        "failures": 0,
        "norm_cost": 3.2,
        "norm_sec": 15.0,
        "norm_lat": 42.5,
        "total_score": 10.5
    }
    run_verifier_test("Test Case 1 (Perfect Run)", constraints_1, results_1)

    # TEST CASE 2: Violated Numeric Constraint --> Latency is too high compared to the threshold.
    constraints_2 = [
        {"metric": "latency", "operator": "<=", "threshold": 40.0}
    ]
    results_2 = {
        "failures": 0,
        "norm_cost": 1.1,
        "norm_sec": 20.0,
        "norm_lat": 85.3, # Violates the 40.0 threshold
        "total_score": 12.0
    }
    run_verifier_test("Test Case 2 (SLA Violation)", constraints_2, results_2)

    # TEST CASE 3: Optimizer Failure --> Even if metrics look okay, 'failures > 0' must trigger a failure.
    constraints_3 = [
        {"metric": "cost", "operator": "<=", "threshold": 10.0}
    ]
    results_3 = {
        "failures": 3, # 3 applications could not be placed
        "norm_cost": 2.5,
        "norm_sec": 10.0,
        "norm_lat": 30.0,
        "total_score": float("inf")
    }
    run_verifier_test("Test Case 3 (Optimizer Placement Failure)", constraints_3, results_3)