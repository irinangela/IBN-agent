from typing import Dict, Any

from state import AgentState
from nodes.schemas import UNIT_LABELS, METRIC_RESULT_KEYS


def verifier_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates the simulation results against the user's hard constraints.
    Compares operator-unit thresholds to mean per-app metrics (avg_*).
    Returns deterministic feedback and a boolean success flag.
    """
    results = state.get("simulation_results", {})
    intent = state.get("active_parsed_intent", {})
    hard_constraints = intent.get("hard_constraints", [])

    violations = []

    failures = results.get("failures", 0)
    if failures > 0:
        violations.append(
            f"CRITICAL FAILURE: The optimizer failed to place {failures} applications. "
            "The weights are likely causing capacity bottlenecks or violating minimum security tiers."
        )

    for hc in hard_constraints:
        metric_name = hc.get("metric", "").lower()
        sim_key = METRIC_RESULT_KEYS.get(metric_name)

        if not sim_key or sim_key not in results:
            violations.append(
                f"UNRECOGNIZED METRIC: The verifier cannot check '{metric_name}' "
                "as it is not a known simulator output."
            )
            continue

        sim_val = results[sim_key]
        threshold = hc["threshold"]
        op = hc["operator"]
        unit = UNIT_LABELS.get(metric_name, "")

        is_violated = False
        if op in ["<=", "<"] and sim_val > threshold:
            is_violated = True
        elif op in [">=", ">"] and sim_val < threshold:
            is_violated = True
        elif op == "==" and sim_val != threshold:
            is_violated = True

        if is_violated:
            violations.append(
                f"Violated {metric_name}: Achieved {sim_val:.3f} {unit}, "
                f"but required {op} {threshold}."
            )

    is_satisfied = len(violations) == 0

    if is_satisfied:
        feedback = (
            "SUCCESS: All active hard constraints and placement requirements "
            "were strictly satisfied."
        )
    else:
        feedback = "FAILED CONSTRAINTS:\n" + "\n".join(f"- {v}" for v in violations)

    return {
        "constraints_satisfied": is_satisfied,
        "verifier_feedback": feedback,
    }
