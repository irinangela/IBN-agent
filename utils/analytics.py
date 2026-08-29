import os
import json
import glob
from datetime import datetime
from typing import Optional

from simulation.config import SEED
from nodes.llm import MODEL_ID, TEMPERATURE

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
            "iteration": None,
        })
    return records


def _attempt_from_final_state(final_state: dict) -> Optional[dict]:
    """Both SUCCESS and max-iter exits skip Recalibrator, so the
    last simulation result is missing. Here we add that to the JSON for plotting.
    """
    results = final_state.get("simulation_results") or {}
    weights = final_state.get("current_weights") or {}
    iteration = int(final_state.get("iteration_count") or 0)
    if iteration <= 0:
        return None
    if not results and not weights:
        return None
    return {
        "attempt": iteration,
        "w1": weights.get("w1"),
        "w2": weights.get("w2"),
        "w3": weights.get("w3"),
        "avg_cost": results.get("avg_cost"),
        "avg_security": results.get("avg_security"),
        "avg_latency": results.get("avg_latency"),
        "norm_cost": results.get("norm_cost"),
        "norm_sec": results.get("norm_sec"),
        "norm_lat": results.get("norm_lat"),
        "feedback": final_state.get("verifier_feedback") or "",
        "active_thresholds": _threshold_map(
            final_state.get("active_parsed_intent") or {}
        ),
    }


def _same_sim(left: dict, right: dict) -> bool:
    keys = ("w1", "w2", "w3", "avg_cost", "avg_latency", "avg_security")
    return all(left.get(key) == right.get(key) for key in keys)


def _populate_attempt(target: dict, source: dict) -> None:
    """Fill onto older history rows."""
    for key in ("norm_cost", "norm_sec", "norm_lat", "active_thresholds", "feedback"):
        if target.get(key) in (None, "", {}) and source.get(key) not in (None, "", {}):
            target[key] = source[key]


def complete_attempt_log(final_state: dict) -> list:
    """Per-iteration series used for plotting."""
    history = []
    for record in final_state.get("recalibration_history") or []:
        if isinstance(record, dict):
            history.append(dict(record))

    last = _attempt_from_final_state(final_state)
    if last is None:
        return history
    if not history:
        return [last]

    previous = history[-1]
    if previous.get("attempt") == last.get("attempt") or _same_sim(previous, last):
        _populate_attempt(previous, last)
        return history
    return history + [last]


def save_run_analytics(final_state: dict, execution_time_seconds: float):
    """Extracts metrics from the final state and saves them to a JSON file."""
    save_dir = "results-analytics"
    os.makedirs(save_dir, exist_ok=True)

    original_intent = final_state.get("original_parsed_intent", {})
    active_intent = final_state.get("active_parsed_intent", {})
    original_constraints = len(original_intent.get("hard_constraints", []))
    active_constraints = len(active_intent.get("hard_constraints", []))
    constraints_dropped = original_constraints - active_constraints
    relaxations = _relaxation_records(original_intent, active_intent)

    analytics = {
        "user_intent": final_state.get("user_intent") or "",
        "initial_intent": original_intent,
        "active_intent": active_intent,
        "timestamp": datetime.now().isoformat(),
        "execution_time_seconds": round(execution_time_seconds, 2),
        "success": final_state.get("constraints_satisfied", False),
        "max_iterations": int(final_state.get("max_iterations") or 0),
        "iterations_done": int(final_state.get("iteration_count") or 0),
        "constraints_dropped": constraints_dropped,
        "relaxations": relaxations,
        "relaxation_events": list(relaxations),
        "attempts": complete_attempt_log(final_state),
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
        "final_metrics": final_state.get("simulation_results", {}),
        "seed": SEED,
        "model": MODEL_ID,
        "temperature": TEMPERATURE,
    }
    
    date_str = datetime.now().strftime("%d-%m-%Y")
    existing_files = glob.glob(f"{save_dir}/{date_str}-run-*.json")
    run_number = len(existing_files) + 1
    filename = os.path.join(save_dir, f"{date_str}-run-{run_number}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(analytics, f, indent=4)
    return filename