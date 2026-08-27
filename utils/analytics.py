import os
import json
import glob
from datetime import datetime


def _threshold_map(intent: dict) -> dict:
    mapping = {}
    for hc in (intent or {}).get("hard_constraints") or []:
        metric = (hc.get("metric") or "").lower()
        if not metric:
            continue
        mapping[metric] = hc.get("threshold")
    return mapping


def _relaxation_records(original_intent: dict, active_intent: dict) -> list:
    original = _threshold_map(original_intent)
    active = _threshold_map(active_intent)
    records = []
    for metric, new_threshold in active.items():
        old_threshold = original.get(metric)
        if old_threshold is None or new_threshold == old_threshold:
            continue
        try:
            old_f = float(old_threshold)
            new_f = float(new_threshold)
            relative = (new_f - old_f) / abs(old_f) * 100.0 if old_f != 0 else None
        except (TypeError, ValueError):
            relative = None
        records.append({
            "metric": metric,
            "original": old_threshold,
            "active": new_threshold,
            "relative_gap_percent": round(relative, 1) if relative is not None else None,
        })
    return records


def save_run_analytics(final_state: dict, execution_time_seconds: float):
    """Extracts metrics from the final state and saves them to a JSON file."""
    save_dir = "results-analytics"
    os.makedirs(save_dir, exist_ok=True)

    original_intent = final_state.get("original_parsed_intent", {})
    active_intent = final_state.get("active_parsed_intent", {})
    original_constraints = len(original_intent.get("hard_constraints", []))
    active_constraints = len(active_intent.get("hard_constraints", []))
    constraints_dropped = original_constraints - active_constraints

    analytics = {
        "initial_intent": original_intent,
        "active_intent": active_intent,
        "timestamp": datetime.now().isoformat(),
        "execution_time_seconds": round(execution_time_seconds, 2),
        "success": final_state.get("constraints_satisfied", False),
        "total_iterations": final_state.get("iteration_count", 0),
        "constraints_dropped": constraints_dropped,
        "relaxations": _relaxation_records(original_intent, active_intent),
        "algorithm": (
            "app_rollout" if final_state.get("use_rollout") else "best_fit"
        ),
        "feasibility_report": final_state.get("feasibility_report", {}),
        "operator_decision": final_state.get("operator_decision", {}),
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