from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from state import AgentState
from simulation.config import retrieval_seeds
from utils.retriever import get_warm_start_context
from nodes.llm import llm, invoke_structured, structured_usage
from nodes.weights import SIMPLEX_CENTER, clip_and_normalize_weights

COLD_START_PROPOSER_CONTEXT = (
    "COLD START: retrieval pool is empty. There are no historical runs. "
    "Do not invent a historical row and do not copy the system-prompt example "
    "(w1=1.0 for latency). Code will clip your proposal to ±0.15 of the "
    "simplex center w1=0.333, w2=0.333, w3=0.334."
)


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
    algorithm = "app_rollout" if state.get("use_rollout") else "best_fit"
    cold_start = not retrieval_seeds()
    historical_context = (
        COLD_START_PROPOSER_CONTEXT
        if cold_start
        else get_warm_start_context(parsed_intent, algorithm=algorithm)
    )
    clip_base = SIMPLEX_CENTER if cold_start else None

    structured_llm = llm.with_structured_output(ProposedWeights, include_raw=True)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=PROPOSER_SYSTEM_PROMPT),
        ("user", "Parsed Intent:\n{intent}\n\nHistorical Context:\n{context}"),
    ])

    chain = prompt | structured_llm
    response = invoke_structured(chain, {
        "intent": str(parsed_intent),
        "context": historical_context,
    })
    result = response.get("parsed")
    in_tok, out_tok = structured_usage(response)
    in_tokens = state.get("total_input_tokens", 0) + in_tok
    out_tokens = state.get("total_output_tokens", 0) + out_tok

    if result is None:
        weights = clip_and_normalize_weights(0.333, 0.333, 0.334, 0.333, 0.333, 0.334)
        return {
            "historical_context": historical_context,
            "current_weights": weights,
            "reasoning": "Structured parse failed. Switched to using balanced weights so search can continue.",
            "total_input_tokens": in_tokens,
            "total_output_tokens": out_tokens,
        }

    if clip_base is None:
        clip_base = (
            result.best_historical_w1,
            result.best_historical_w2,
            result.best_historical_w3,
        )
    weights = clip_and_normalize_weights(
        result.w1, result.w2, result.w3, *clip_base,
    )

    return {
        "historical_context": historical_context,
        "current_weights": weights,
        "reasoning": result.reasoning,
        "total_input_tokens": in_tokens,
        "total_output_tokens": out_tokens,
    }
