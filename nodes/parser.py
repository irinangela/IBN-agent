from typing import Dict, Any
import copy

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

from state import AgentState
from nodes.llm import llm
from nodes.schemas import IntentSchema


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
    response = chain.invoke({"intent": user_intent})

    parsed_result = response["parsed"]
    raw_msg = response["raw"]

    usage = getattr(raw_msg, "usage_metadata", {}) or {}
    in_tokens = state.get("total_input_tokens", 0) + usage.get("input_tokens", 0)
    out_tokens = state.get("total_output_tokens", 0) + usage.get("output_tokens", 0)

    # Thresholds stay in operator units (ms / cost / security score). No norm remapping.
    original_intent_dict = get_relaxation_order(parsed_result.model_dump())
    active_intent_dict = copy.deepcopy(original_intent_dict)

    return {
        "original_parsed_intent": original_intent_dict,
        "active_parsed_intent": active_intent_dict,
        "total_input_tokens": in_tokens,
        "total_output_tokens": out_tokens,
    }
