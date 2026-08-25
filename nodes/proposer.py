from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from state import AgentState
from utils.retriever import get_warm_start_context
from nodes.llm import llm
from nodes.weights import clip_and_normalize_weights


class ProposedWeights(BaseModel):
    best_historical_w1: float = Field(
        description="The EXACT w1 value from the single best historical run provided in the context."
    )
    best_historical_w2: float = Field(
        description="The EXACT w2 value from the single best historical run provided in the context."
    )
    best_historical_w3: float = Field(
        description="The EXACT w3 value from the single best historical run provided in the context."
    )
    reasoning: str = Field(
        description="Explain why you are making micro-adjustments (max +/- 0.15) to the historical baseline to satisfy the intent."
    )
    w1: float = Field(description="The adjusted Weight for Cost (between 0.0 and 1.0)")
    w2: float = Field(description="The adjusted Weight for Security (between 0.0 and 1.0)")
    w3: float = Field(description="The adjusted Weight for Latency (between 0.0 and 1.0)")


try:
    with open("prompts/proposer_system_prompt.md", "r") as f:
        PROPOSER_SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    PROPOSER_SYSTEM_PROMPT = (
        "You are an expert Intent-Based Networking (IBN) agent. Your task is to "
        "propose a set of weights for a multi-objective service placement optimizer "
        "based on the parsed intent and historical successful runs. Ensure that the "
        "weights sum to 1.0 and provide reasoning for your selection."
    )


def weight_proposer_node(state: AgentState) -> Dict[str, Any]:
    parsed_intent = state["active_parsed_intent"]
    historical_context = get_warm_start_context(parsed_intent)

    structured_llm = llm.with_structured_output(ProposedWeights, include_raw=True)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=PROPOSER_SYSTEM_PROMPT),
        ("user", "Parsed Intent:\n{intent}\n\nHistorical Context:\n{context}"),
    ])

    chain = prompt | structured_llm
    response = chain.invoke({
        "intent": str(parsed_intent),
        "context": historical_context,
    })

    result = response["parsed"]
    raw_msg = response["raw"]

    usage = getattr(raw_msg, "usage_metadata", {}) or {}
    in_tokens = state.get("total_input_tokens", 0) + usage.get("input_tokens", 0)
    out_tokens = state.get("total_output_tokens", 0) + usage.get("output_tokens", 0)

    weights = clip_and_normalize_weights(
        result.w1, result.w2, result.w3,
        result.best_historical_w1, result.best_historical_w2, result.best_historical_w3,
    )

    return {
        "historical_context": historical_context,
        "current_weights": weights,
        "reasoning": result.reasoning,
        "total_input_tokens": in_tokens,
        "total_output_tokens": out_tokens,
    }
