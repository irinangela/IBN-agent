import copy
from typing import Any, Dict

from state import AgentState
from utils.retriever import ALGORITHMS, feasibility_report


def _format_feasibility_reasoning(report: Dict[str, Any]) -> str:
    seeds = report.get("seeds") or [report.get("seed")]
    mode = report.get("warm_start_mode") or "near_neighbor"
    live = report.get("live_seed")
    lines = [
        f"Feasibility pre-check on Warm-start seeds {seeds} "
        f"(mode={mode}, live seed={live}). \n\n"
    ]
    for algorithm, info in (report.get("per_algorithm") or {}).items():
        bounds = info.get("bounds") or {}
        joint = (
            "jointly feasible"
            if info.get("jointly_feasible")
            else "not jointly feasible"
        )
        bound_txt = ", ".join(
            f"{metric}={value:.3f}" for metric, value in bounds.items()
        ) or "no bounds"
        lines.append(f"{algorithm}: {joint}. Best known: {bound_txt}. \n")
    if report.get("auto_switched_to"):
        lines.append(
            f"\n\n Auto-switched live simulator to {report['auto_switched_to']} "
            "(historically feasible)."
        )
    elif report.get("needs_operator"):
        lines.append(
            "\n\n No historical run meets all hard constraints. Operator input required."
        )
    else:
        lines.append(f"\n\nProceeding with {report.get('chosen_algorithm')}.")
    return " ".join(lines)


def apply_operator_decision(
    state: Dict[str, Any],
    decision: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply a continue or a relax choice. Does not re-parse a new intent."""

    action = (decision or {}).get("action")
    report = state.get("feasibility_report") or {}
    intent = copy.deepcopy(state.get("active_parsed_intent") or {})
    non_relaxable = {
        m.lower() for m in (intent.get("non_relaxable_constraints") or [])
    }
    suggested = {
        item["metric"]: item
        for item in (report.get("suggested_relaxations") or [])
        if item.get("metric")
    }

    if action == "relax":
        thresholds = (decision or {}).get("thresholds") or {}
        for constraint in intent.get("hard_constraints") or []:
            metric = (constraint.get("metric") or "").lower()
            if metric not in thresholds:
                continue
            if metric in non_relaxable:
                continue
            if metric in suggested and not suggested[metric].get("relaxable", True):
                continue
            try:
                constraint["threshold"] = float(thresholds[metric])
            except (TypeError, ValueError):
                continue

    chosen = (decision or {}).get("algorithm") or report.get("chosen_algorithm")
    if chosen not in ALGORITHMS:
        chosen = "best_fit"
    use_rollout = chosen == "app_rollout"
    note = f"Operator chose '{action}'."
    if action == "relax":
        note += " Active constraint thresholds were updated."
        if (decision or {}).get("algorithm") in ALGORITHMS:
            note += f" Live simulator set to {chosen}."

    updates: Dict[str, Any] = {
        "active_parsed_intent": intent,
        "needs_operator": False,
        "use_rollout": use_rollout,
        "operator_decision": decision or {},
        "reasoning": note,
    }
    if action == "relax" and (decision or {}).get("algorithm") in ALGORITHMS:
        report = dict(report)
        report["chosen_algorithm"] = chosen
        updates["feasibility_report"] = report
    return updates


def feasibility_node(state: AgentState) -> Dict[str, Any]:
    """Code-only pre-check: skip search when history proves the SLA infeasible."""

    intent = state.get("active_parsed_intent") or {}
    report = feasibility_report(intent)
    updates: Dict[str, Any] = {
        "feasibility_report": report,
        "use_rollout": report.get("chosen_algorithm") == "app_rollout",
        "needs_operator": bool(report.get("needs_operator")),
        "reasoning": _format_feasibility_reasoning(report),
    }

    decision = state.get("operator_decision") or {}
    if decision.get("action") in ("continue", "relax"):
        merged = {**dict(state), **updates}
        updates.update(apply_operator_decision(merged, decision))

    return updates
