from typing import Dict, Any, Optional
import copy
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from state import AgentState
from nodes.llm import llm
from nodes.schemas import MetricType, METRIC_RESULT_KEYS, HIGHER_IS_BETTER
from nodes.weights import clip_and_normalize_weights


MIN_DISTINCT_FAILED_WEIGHTS = 3

def _matching_constraint(intent: Dict[str, Any], metric: str) -> Optional[Dict[str, Any]]:
    for hc in intent.get("hard_constraints") or []:
        if (hc.get("metric") or "").lower() == metric:
            return hc
    return None


def _is_strictly_looser(operator: str, old_threshold: float, new_threshold: float) -> bool:
    if operator in ("<=", "<"):
        return new_threshold > old_threshold
    if operator in (">=", ">"):
        return new_threshold < old_threshold
    return False


def _gap_message(metric: str, original_threshold: float, new_threshold: float) -> str:
    if original_threshold == 0:
        return (
            f"{metric} constraint relaxed from {original_threshold} to {new_threshold}."
        )
    relative = (new_threshold - original_threshold) / abs(original_threshold) * 100.0
    sign = "+" if relative >= 0 else "-"
    return (
        f"{metric} constraint relaxed from {original_threshold} to {new_threshold}, "
        f"{sign}{relative:.0f}%."
    )


def apply_constraint_relaxation(
    parsed_intent: Dict[str, Any],
    relax_metric: Optional[str],
    relaxed_threshold: Optional[float] = None,
    original_intent: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], str]:
    """
    Code-enforced relaxation: loosen a hard-constraint threshold. Never delete it.
    Metric must be in relaxation_order and not in non_relaxable_constraints.
    """
    if not relax_metric:
        if relaxed_threshold is not None:
            return parsed_intent, (
                "\n[SYSTEM OVERRIDE]: relaxed_threshold was set without relax_metric. "
                "Both fields are required to loosen a constraint."
            )
        return parsed_intent, ""

    metric = relax_metric.lower()
    if relaxed_threshold is None:
        return parsed_intent, (
            f"\n[SYSTEM OVERRIDE]: LLM attempted to relax '{metric}' "
            "but did not propose a relaxed_threshold. Constraint unchanged."
        )

    try:
        new_threshold = float(relaxed_threshold)
    except (TypeError, ValueError):
        return parsed_intent, (
            f"\n[SYSTEM OVERRIDE]: LLM attempted to relax '{metric}' "
            "with a non-numeric relaxed_threshold. Constraint unchanged."
        )

    non_relaxable = [m.lower() for m in parsed_intent.get("non_relaxable_constraints", [])]
    relaxation_order = [m.lower() for m in parsed_intent.get("relaxation_order", [])]

    if metric in non_relaxable:
        return parsed_intent, (
            f"\n[SYSTEM OVERRIDE]: LLM attempted to relax '{metric}', "
            f"but code enforcement blocked it because it is NON-RELAXABLE."
        )

    if metric not in relaxation_order:
        return parsed_intent, (
            f"\n[SYSTEM OVERRIDE]: LLM attempted to relax '{metric}', "
            f"but code enforcement blocked it because it is not in relaxation_order."
        )

    constraint = _matching_constraint(parsed_intent, metric)
    if constraint is None:
        return parsed_intent, (
            f"\n[SYSTEM OVERRIDE]: LLM attempted to relax '{metric}', "
            f"but there is no hard constraint on '{metric}' to loosen. "
            f"Only existing hard_constraints can be relaxed."
        )

    try:
        current_threshold = float(constraint["threshold"])
    except (TypeError, ValueError, KeyError):
        return parsed_intent, (
            f"\n[SYSTEM OVERRIDE]: Active '{metric}' constraint has no numeric threshold."
        )

    operator = constraint.get("operator") or ""
    if not _is_strictly_looser(operator, current_threshold, new_threshold):
        return parsed_intent, (
            f"\n[SYSTEM OVERRIDE]: LLM attempted to relax '{metric}' to {new_threshold}, "
            f"which is not strictly looser than the active threshold {current_threshold} "
            f"({operator}). Constraint unchanged."
        )

    original_hc = _matching_constraint(original_intent or {}, metric)
    try:
        original_threshold = (
            float(original_hc["threshold"]) if original_hc is not None else current_threshold
        )
    except (TypeError, ValueError, KeyError):
        original_threshold = current_threshold

    constraint["threshold"] = new_threshold
    return parsed_intent, (
        f"\n[IMPORTANT]: {_gap_message(metric, original_threshold, new_threshold)}"
    )


def distinct_failed_weight_count(history: list) -> int:
    """Count unique failed (w1, w2, w3) vectors, rounded to 3 decimals."""

    seen = set()
    for rec in history or []:
        if not isinstance(rec, dict):
            continue
        try:
            key = (
                round(float(rec.get("w1")), 3),
                round(float(rec.get("w2")), 3),
                round(float(rec.get("w3")), 3),
            )
        except (TypeError, ValueError):
            continue
        seen.add(key)
    return len(seen)


def violated_metrics_from_feedback(feedback: str) -> list[str]:
    """Extract violated metric names from verifier feedback text."""
    found = []
    text = feedback or ""
    for metric in ("latency", "cost", "security"):
        if f"Violated {metric}" in text:
            found.append(metric)
    return found


def relaxable_hard_metrics(intent: Dict[str, Any]) -> list[str]:
    """Hard-constraint metrics that code is allowed to loosen."""
    non_relaxable = {m.lower() for m in intent.get("non_relaxable_constraints", [])}
    relaxation_order = {m.lower() for m in intent.get("relaxation_order", [])}
    eligible = []
    for hc in intent.get("hard_constraints", []):
        metric = (hc.get("metric") or "").lower()
        if metric and metric not in non_relaxable and metric in relaxation_order:
            if metric not in eligible:
                eligible.append(metric)
    return eligible


def _fmt_weight(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def build_attempt_record(
    attempt_num: int,
    weights: Dict[str, Any],
    results: Dict[str, Any],
    feedback: str,
) -> Dict[str, Any]:
    """Structured history row: weights + mean operator-unit metrics."""
    weights = weights or {}
    results = results or {}
    return {
        "attempt": attempt_num,
        "w1": weights.get("w1"),
        "w2": weights.get("w2"),
        "w3": weights.get("w3"),
        "avg_cost": results.get("avg_cost"),
        "avg_security": results.get("avg_security"),
        "avg_latency": results.get("avg_latency"),
        "feedback": feedback or "",
    }


def format_attempt_record(record: Any) -> str:
    """Render one history row for the recalibrator prompt."""
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        return str(record)

    metrics = []
    for key in ("avg_cost", "avg_security", "avg_latency"):
        val = record.get(key)
        if val is not None:
            try:
                metrics.append(f"{key}={float(val):.3f}")
            except (TypeError, ValueError):
                pass
    metric_str = ", ".join(metrics) if metrics else "metrics unavailable"
    feedback = " ".join((record.get("feedback") or "").split())
    return (
        f"Attempt {record.get('attempt', '?')}: "
        f"w1={_fmt_weight(record.get('w1'))}, "
        f"w2={_fmt_weight(record.get('w2'))}, "
        f"w3={_fmt_weight(record.get('w3'))} "
        f"| {metric_str} | {feedback}"
    )


def format_attempts_for_prompt(attempts: list) -> str:
    if not attempts:
        return "None yet."
    return "\n".join(format_attempt_record(rec) for rec in attempts)


def select_best_attempt(
    attempts: list,
    violated_metric: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Return the attempt with the best value on the violated metric.

    latency/cost: lower is better. security: higher is better.
    """
    if not violated_metric:
        return None
    metric = violated_metric.lower()
    key = METRIC_RESULT_KEYS.get(metric)
    if not key:
        return None

    scored = []
    for index, rec in enumerate(attempts or []):
        if not isinstance(rec, dict):
            continue
        val = rec.get(key)
        if val is None:
            continue
        try:
            scored.append((float(val), index, rec))
        except (TypeError, ValueError):
            continue

    if not scored:
        return None

    higher = metric in HIGHER_IS_BETTER
    scored.sort(key=lambda item: (-item[0] if higher else item[0], item[1]))
    return scored[0][2]


def format_best_so_far_hint(
    attempts: list,
    violated_metric: Optional[str],
) -> str:
    """Code-side ranking so the LLM does not have to notice 880 < 888."""
    if not violated_metric:
        return (
            "No violated metric parsed from verifier feedback; "
            "rank attempts by the failed SLA yourself."
        )

    key = METRIC_RESULT_KEYS.get(violated_metric.lower())
    best = select_best_attempt(attempts, violated_metric)
    if not key or not best:
        return (
            f"Violated metric is '{violated_metric}', "
            "but no numeric attempt history is available yet."
        )

    parts = [
        f"Best so far: Attempt {best.get('attempt')} "
        f"(w1={best.get('w1')}, w2={best.get('w2')}, w3={best.get('w3')}) "
        f"{key}={float(best[key]):.3f}."
    ]

    dict_attempts = [rec for rec in (attempts or []) if isinstance(rec, dict)]
    latest = dict_attempts[-1] if dict_attempts else None
    higher = violated_metric.lower() in HIGHER_IS_BETTER

    if latest is not None and latest is not best:
        latest_val = latest.get(key)
        if latest_val is not None:
            worse = (float(latest_val) < float(best[key])) if higher else (
                float(latest_val) > float(best[key])
            )
            if worse:
                parts.append(
                    f"Last attempt (Attempt {latest.get('attempt')}) was worse "
                    f"({key}={float(latest_val):.3f}). Reverse toward this point."
                )

    if len(dict_attempts) >= 2:
        prev, last = dict_attempts[-2], dict_attempts[-1]
        prev_val, last_val = prev.get(key), last.get(key)
        if prev_val is not None and last_val is not None:
            prev_f, last_f = float(prev_val), float(last_val)
            if (last_f < prev_f) if higher else (last_f > prev_f):
                parts.append(
                    f"Last step made {violated_metric} worse "
                    f"({prev_f:.3f} → {last_f:.3f}); reverse that weight change."
                )
            elif (last_f > prev_f) if higher else (last_f < prev_f):
                parts.append(
                    f"Last step improved {violated_metric} "
                    f"({prev_f:.3f} → {last_f:.3f}) but still failed; "
                    f"continue a small step in the same empirical direction."
                )
            else:
                parts.append(
                    f"Last step did not change {violated_metric}; "
                    "try a different nearby point."
                )

    return " ".join(parts)


def format_relaxation_eligibility(intent: Dict[str, Any]) -> str:
    eligible = relaxable_hard_metrics(intent)
    if eligible:
        return (
            "These hard constraints may be loosened (not dropped) if search is stuck "
            f"and at least {MIN_DISTINCT_FAILED_WEIGHTS} distinct weight vectors have failed: "
            + ", ".join(eligible)
            + ". Prefer the earliest entry in relaxation_order. "
            "Propose relax_metric and a strictly looser relaxed_threshold."
        )
    return (
        "[SYSTEM]: No relaxable hard constraint remains. "
        "Leave relax_metric and relaxed_threshold null. Weight search only."
    )


class RecalibrationOutput(BaseModel):
    reasoning: str = Field(
        description=(
            "Analyze the Verifier's feedback. Explain exactly how "
            "you are adjusting the weights or why a relaxation is necessary."
        ),
    )
    relax_metric: Optional[MetricType] = Field(
        default=None,
        description=(
            "Metric whose hard_constraint threshold may be loosened if stuck. "
            "Must currently exist in active hard_constraints, and in relaxation_order, "
            "but not in non_relaxable. Otherwise null. Do not name a soft-preference metric."
        ),
    )
    relaxed_threshold: Optional[float] = Field(
        default=None,
        description=(
            "New numeric threshold for relax_metric. Must be strictly looser than the "
            "active threshold. Null when relax_metric is null."
        ),
    )
    w1: float = Field(description="New adjusted weight for Cost (w1)")
    w2: float = Field(description="New adjusted weight for Security (w2)")
    w3: float = Field(description="New adjusted weight for Latency (w3)")


try:
    with open("prompts/recalibration_system_prompt.md", "r") as f:
        RECALIBRATOR_SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    RECALIBRATOR_SYSTEM_PROMPT = (
        "You are the Adaptive Search Engine for an Intent-Based Networking optimizer. "
        "Your task is to analyze the Verifier's feedback and propose new weights "
        "(w1, w2, w3) for the next optimization run."
    )


def recalibrator_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyzes negative feedback and proposes new weights.
    Handles constraint relaxation if the problem is infeasible.
    """
    original_intent = state.get("original_parsed_intent", {})
    parsed_intent = copy.deepcopy(state.get("active_parsed_intent", {}))
    current_weights = state.get("current_weights", {})
    feedback = state.get("verifier_feedback", "")
    history = list(state.get("recalibration_history") or [])
    results = state.get("simulation_results") or {}
    rag_context = state.get("historical_context") or "No historical context available."

    current_attempt_record = build_attempt_record(
        attempt_num=len(history) + 1,
        weights=current_weights,
        results=results,
        feedback=feedback,
    )
    updated_history = history + [current_attempt_record]

    violated = violated_metrics_from_feedback(feedback)
    primary_violated = violated[0] if violated else None
    best_so_far = format_best_so_far_hint(updated_history, primary_violated)

    structured_llm = llm.with_structured_output(RecalibrationOutput, include_raw=True)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=RECALIBRATOR_SYSTEM_PROMPT),
        ("user",
         "Original Intent:\n{original}\n\n"
         "Active Intent (Current Constraints):\n{active}\n\n"
         "Historical RAG Context (warm start; weights are non-intuitive — trust outcomes):\n{rag}\n\n"
         "Past Attempts (this run, operator units):\n{history}\n\n"
         "Latest Verifier Feedback:\n{feedback}\n\n"
         "Code hint — empirically best attempt so far on the violated metric:\n{best_so_far}\n\n"
         "Relaxation eligibility:\n{relaxation_eligibility}"),
    ])

    chain = prompt | structured_llm

    response = chain.invoke({
        "original": json.dumps(original_intent, indent=2),
        "active": json.dumps(parsed_intent, indent=2),
        "rag": rag_context,
        "history": format_attempts_for_prompt(updated_history),
        "feedback": feedback,
        "best_so_far": best_so_far,
        "relaxation_eligibility": format_relaxation_eligibility(parsed_intent),
    })

    result = response["parsed"]
    raw_msg = response["raw"]

    usage = getattr(raw_msg, "usage_metadata", {}) or {}
    in_tokens = state.get("total_input_tokens", 0) + usage.get("input_tokens", 0)
    out_tokens = state.get("total_output_tokens", 0) + usage.get("output_tokens", 0)

    relax_metric = getattr(result, "relax_metric", None)
    relaxed_threshold = getattr(result, "relaxed_threshold", None)
    distinct_count = distinct_failed_weight_count(updated_history)

    if relax_metric and distinct_count < MIN_DISTINCT_FAILED_WEIGHTS:
        relaxed_msg = (
            f"\n[SYSTEM OVERRIDE]: Relaxation of '{relax_metric}' refused: "
            f"only {distinct_count} distinct failed weight vector(s); "
            f"need ≥{MIN_DISTINCT_FAILED_WEIGHTS}."
        )
    else:
        parsed_intent, relaxed_msg = apply_constraint_relaxation(
            parsed_intent,
            relax_metric,
            relaxed_threshold,
            original_intent=original_intent,
        )
    if parsed_intent.get("hard_constraints") and not relaxable_hard_metrics(parsed_intent):
        relaxed_msg += (
            "\n[SYSTEM]: No relaxable hard constraint remains. Weight search only."
        )

    prev_w1 = current_weights.get("w1", 0.333)
    prev_w2 = current_weights.get("w2", 0.333)
    prev_w3 = current_weights.get("w3", 0.334)

    weights = clip_and_normalize_weights(
        result.w1, result.w2, result.w3,
        prev_w1, prev_w2, prev_w3,
    )

    new_iteration_count = state.get("iteration_count", 0) + 1

    return {
        "current_weights": weights,
        "reasoning": result.reasoning + relaxed_msg,
        "active_parsed_intent": parsed_intent,
        "iteration_count": new_iteration_count,
        "recalibration_history": updated_history,
        "total_input_tokens": in_tokens,
        "total_output_tokens": out_tokens,
    }
