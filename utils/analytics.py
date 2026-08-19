import os
import json
import glob
from datetime import datetime

def save_run_analytics(final_state: dict, execution_time_seconds: float):
    """Extracts metrics from the final state and saves them to a JSON file."""
    save_dir = "results-analytics"
    os.makedirs(save_dir, exist_ok=True)
    
    # How many constraints were relaxed
    original_constraints = len(final_state.get("original_parsed_intent", {}).get("hard_constraints", []))
    active_constraints = len(final_state.get("active_parsed_intent", {}).get("hard_constraints", []))
    constraints_dropped = original_constraints - active_constraints
    
    # Analytics payload
    analytics = {
        "initial_intent": final_state.get("original_parsed_intent", {}),
        "active_intent": final_state.get("active_parsed_intent", {}),
        "timestamp": datetime.now().isoformat(),
        "execution_time_seconds": round(execution_time_seconds, 2),
        "success": final_state.get("constraints_satisfied", False),
        "total_iterations": final_state.get("iteration_count", 0),
        "constraints_dropped": constraints_dropped,
        "token_usage": {
            "input_tokens": final_state.get("total_input_tokens", 0),
            "output_tokens": final_state.get("total_output_tokens", 0),
            "total_tokens": final_state.get("total_input_tokens", 0) + final_state.get("total_output_tokens", 0)
        },
        "final_weights": final_state.get("current_weights", {}),
        "final_metrics": final_state.get("simulation_results", {})
    }
    
    date_str = datetime.now().strftime("%d-%m-%Y")
    existing_files = glob.glob(f"{save_dir}/{date_str}-run-*.json")
    run_number = len(existing_files) + 1
    
    filename = os.path.join(save_dir, f"{date_str}-run-{run_number}.json")
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(analytics, f, indent=4)
        
    return filename