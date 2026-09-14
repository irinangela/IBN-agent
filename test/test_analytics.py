import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.analytics import complete_attempt_log, save_run_analytics


def _base_state(**overrides):
    state = {
        "user_intent": "latency under 900ms",
        "original_parsed_intent": {
            "primary_objective": "latency",
            "hard_constraints": [
                {"metric": "latency", "operator": "<=", "threshold": 900.0}
            ],
            "relaxation_order": ["latency"],
        },
        "active_parsed_intent": {
            "primary_objective": "latency",
            "hard_constraints": [
                {"metric": "latency", "operator": "<=", "threshold": 945.0}
            ],
            "relaxation_order": ["latency"],
        },
        "constraints_satisfied": True,
        "max_iterations": 5,
        "iteration_count": 2,
        "use_rollout": False,
        "feasibility_report": {},
        "operator_decision": {},
        "total_input_tokens": 10,
        "total_output_tokens": 5,
        "current_weights": {"w1": 0.3, "w2": 0.2, "w3": 0.5},
        "simulation_results": {
            "avg_cost": 500.0,
            "avg_security": 8.0,
            "avg_latency": 920.0,
            "norm_cost": 9.0,
            "norm_sec": 20.0,
            "norm_lat": 25.0,
        },
        "verifier_feedback": "SUCCESS",
        "recalibration_history": [
            {
                "attempt": 1,
                "w1": 0.2,
                "w2": 0.2,
                "w3": 0.6,
                "avg_cost": 520.0,
                "avg_security": 8.0,
                "avg_latency": 980.0,
                "feedback": "FAILED",
            }
        ],
    }
    state.update(overrides)
    return state


def test_complete_attempt_log_appends_last_sim():
    attempts = complete_attempt_log(_base_state())
    assert len(attempts) == 2
    assert attempts[0]["attempt"] == 1
    assert attempts[1]["attempt"] == 2
    assert attempts[1]["norm_lat"] == 25.0
    assert attempts[1]["avg_latency"] == 920.0
    assert attempts[1]["active_thresholds"]["latency"] == 945.0


def test_complete_attempt_log_does_not_duplicate_same_sim():
    history_row = {
        "attempt": 1,
        "w1": 0.3,
        "w2": 0.2,
        "w3": 0.5,
        "avg_cost": 500.0,
        "avg_security": 8.0,
        "avg_latency": 920.0,
        "feedback": "SUCCESS",
    }
    attempts = complete_attempt_log(_base_state(
        iteration_count=1,
        recalibration_history=[history_row],
    ))
    assert len(attempts) == 1
    assert attempts[0]["norm_lat"] == 25.0


def test_save_run_analytics_writes_attempts_and_user_intent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = save_run_analytics(_base_state(), 1.23)
    assert os.path.isfile(path)
    assert "results-analytics-phase-6" in path.replace("\\", "/")

    import json
    payload = json.loads(open(path, encoding="utf-8").read())
    assert payload["user_intent"] == "latency under 900ms"
    assert len(payload["attempts"]) == 2
    assert payload["relaxations"][0]["metric"] == "latency"
    assert payload["relaxation_events"][0]["iteration"] is None
    assert payload["success"] is True
    assert "cold_start" in payload
    assert payload["warm_start_grid_points"] == 66
    assert payload["warm_start_grid_step"] == 0.1


def test_save_run_analytics_does_not_overwrite_when_ids_have_gaps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_dir = "results-analytics-phase-6"
    os.makedirs(save_dir, exist_ok=True)
    from datetime import datetime
    date_str = datetime.now().strftime("%d-%m-%Y")
    open(os.path.join(save_dir, f"{date_str}-run-1.json"), "w", encoding="utf-8").write("{}")
    open(os.path.join(save_dir, f"{date_str}-run-3.json"), "w", encoding="utf-8").write("{}")
    path = save_run_analytics(_base_state(), 0.5)
    assert path.replace("\\", "/").endswith(f"{date_str}-run-4.json")
    assert os.path.isfile(os.path.join(save_dir, f"{date_str}-run-3.json"))
