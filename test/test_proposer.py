import json
import sys
import os
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes import weight_proposer_node

def run_proposer_test(test_name: str, parsed_intent: Dict[str, Any]):
    print(f"\n{'='*70}")
    print(f"-- RUNNING TEST: {test_name} --")
    print(f"{'='*70}")
    print("-- PARSED INTENT INPUT: --")
    print(json.dumps(parsed_intent, indent=2))
    print("=" * 70)
    
    mock_state = {
        "user_intent": "Mock natural language intent...", 
        "parsed_intent": parsed_intent,
        "historical_context": "",
        "current_weights": {},
        "reasoning": "",
        "simulation_results": {},
        "iteration_count": 0,
        "max_iterations": 5,
        "final_response": ""
    }
    
    try:
        result_state = weight_proposer_node(mock_state)
        
        print("-- RETRIEVED HISTORICAL CONTEXT (From CSVs): --")
        print(result_state.get("historical_context", "No context retrieved."))
        print("=" * 70)
        

        print("-- LLM PROPOSED WEIGHTS: --")
        proposed = result_state.get("current_weights", {})
        print(json.dumps(proposed, indent=2))
        
        total = sum(proposed.values())
        print(f"\n -- Verify Sum: {total:.3f} (Should be 1.0) --")
        print(f"\n -- LLM REASONING: --")
        print(result_state.get("reasoning", "No reasoning provided."))
        
    except Exception as e:
        import traceback
        print(f"!!! Error during weight proposition !!!")
        print(traceback.format_exc())

if __name__ == "__main__":
    # TEST CASE 1: Standard Constraint, primary objective is latency, hard limit on cost.
    intent_1 = {
        "primary_objective": "latency",
        "hard_constraints": [
            {
                "metric": "cost",
                "operator": "<=",
                "threshold": 3.0  
            }
        ],
        "soft_preferences": ["keep security reasonable"],
        "non_relaxable_constraints": [],
        "relaxation_order": ["security"]
    }

    # TEST CASE 2: Security Focus, primary objective is security, hard limit on latency.
    intent_2 = {
        "primary_objective": "security",
        "hard_constraints": [
            {
                "metric": "latency",
                "operator": "<=",
                "threshold": 50.0 
            }
        ],
        "soft_preferences": [],
        "non_relaxable_constraints": [],
        "relaxation_order": ["cost"]
    }

    # TEST CASE 3: The "Infeasible" Trap, primary objective is cost.
    # Ridiculously tight constraints that no history will match so this should trigger the empty dataframe fallback logic.
    intent_3 = {
        "primary_objective": "cost",
        "hard_constraints": [
            {
                "metric": "latency",
                "operator": "<=",
                "threshold": 0.001 
            },
            {
                "metric": "cost",
                "operator": "<=",
                "threshold": 0.001
            }
        ],
        "soft_preferences": [],
        "non_relaxable_constraints": ["latency must be under 0.001"],
        "relaxation_order": ["security"]
    }

    run_proposer_test("Test Case 1 (Latency Primary, Cost Constraint)", intent_1)
    run_proposer_test("Test Case 2 (Security Primary, Latency Constraint)", intent_2)
    run_proposer_test("Test Case 3 (Infeasible Constraints -> Fallback Trigger)", intent_3)