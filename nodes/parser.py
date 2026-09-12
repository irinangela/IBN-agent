from typing import Dict, Any
import copy
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

from state import AgentState
from nodes.llm import llm, invoke_structured, structured_usage
from nodes.schemas import IntentSchema

KNOWN_METRICS = ("latency", "cost", "security")

UNRESOLVED_PARSE_MESSAGE = (
    "Sorry, I could not map this intent onto the optimizer. "
    "The only supported metrics are latency, cost, and security. "
    "Please rewrite the intent and name the metric you care about so I can understand it and map it onto the optimizer. "
    '(for example: "Optimize for low latency. Average latency must stay under 1000ms.").'
)

STRUCTURED_PARSE_FAILED_MESSAGE = (
    "Sorry, something went wrong and I could not create a valid structured intent from the given input. "
    "Please rewrite the intent or try again."
)


def mentioned_metrics(text: str) -> set[str]:
    lowered = (text or "").lower()
    return {m for m in KNOWN_METRICS if re.search(rf"\b{m}s?\b", lowered)}


def parse_is_checked(intent: Dict[str, Any], user_intent: str) -> bool:
    """True only if all assigned metrics are actually mentioned in the operator text."""
    mentioned = mentioned_metrics(user_intent)
    primary = (intent.get("primary_objective") or "").lower()
    if primary not in mentioned:
        return False
    for hc in intent.get("hard_constraints") or []:
        metric = (hc.get("metric") or "").lower()
        if metric and metric not in mentioned:
            return False
    return True


def get_relaxation_order(intent: Dict[str, Any]) -> Dict[str, Any]:
    """Keep relaxation_order entries that appear in hard_constraints."""
    hard = {
        (hc.get("metric") or "").lower()
        for hc in (intent.get("hard_constraints") or [])
        if hc.get("metric")
    }
    intent["relaxation_order"] = [
        m for m in (intent.get("relaxation_order") or [])
        if (m or "").lower() in hard
    ]
    return intent


try:
    with open("prompts/parser_system_prompt.md", "r") as f:
        PARSER_SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    PARSER_SYSTEM_PROMPT = (
        "You are an expert Intent-Based Networking (IBN) agent. Your task is to "
        "translate a network operator's natural language intent into a highly "
        "structured JSON format for a multi-objective service placement optimizer."
    )


def intent_parser_node(state: AgentState) -> Dict[str, Any]:
    """Parses the natural language intent into a structured JSON dict."""
    structured_llm = llm.with_structured_output(IntentSchema, include_raw=True)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=PARSER_SYSTEM_PROMPT),
        ("user", "{intent}"),
    ])

    chain = prompt | structured_llm
    user_intent = state.get("user_intent", "")
    response = invoke_structured(chain, {"intent": user_intent})
    parsed_result = response.get("parsed")
    in_tok, out_tok = structured_usage(response)
    in_tokens = state.get("total_input_tokens", 0) + in_tok
    out_tokens = state.get("total_output_tokens", 0) + out_tok

    if parsed_result is None:
        return {
            "original_parsed_intent": {},
            "active_parsed_intent": {},
            "parse_failed": True,
            "parse_feedback": STRUCTURED_PARSE_FAILED_MESSAGE,
            "needs_operator": True,
            "total_input_tokens": in_tokens,
            "total_output_tokens": out_tokens,
        }

    original_intent_dict = get_relaxation_order(parsed_result.model_dump())
    active_intent_dict = copy.deepcopy(original_intent_dict)
    checked_metrics = parse_is_checked(original_intent_dict, user_intent)

    return {
        "original_parsed_intent": original_intent_dict,
        "active_parsed_intent": active_intent_dict,
        "parse_failed": not checked_metrics,
        "parse_feedback": "" if checked_metrics else UNRESOLVED_PARSE_MESSAGE,
        "needs_operator": not checked_metrics,
        "total_input_tokens": in_tokens,
        "total_output_tokens": out_tokens,
    }
