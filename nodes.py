import os
from dotenv import load_dotenv
from typing import Dict, Any, Literal, Optional
import copy
import json

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from state import AgentState
from tools.tools import ALL_TOOLS

load_dotenv()

llm = ChatAnthropic(        # Initialize the LLM and bind tools to it
    model="claude-haiku-4-5", 
    temperature=0,
    api_key=os.getenv("ANTHROPIC_API_KEY")
)
llm_with_tools = llm.bind_tools(ALL_TOOLS)


### Intent Parsing Node

MetricType = Literal["latency", "cost", "security"]

class HardConstraint(BaseModel):
    metric: str = Field(description="The metric being constrained (e.g., 'latency', 'cost', 'security').")
    operator: str = Field(description="The comparison operator ('<=', '>=', '<', '>', '==').")
    threshold: float = Field(description="The numeric threshold value.")

class IntentSchema(BaseModel):
    primary_objective: MetricType = Field(description="The main metric to minimize or maximize.")
    hard_constraints: list[HardConstraint] = Field(default_factory=list, description="Strict numeric limits.")
    soft_preferences: list[str] = Field(default_factory=list, description="General preferences without strict numeric bounds.")
    non_relaxable_constraints: list[str] = Field(default_factory=list, description="Constraints that must NEVER be violated, even if it means failing the task.")
    relaxation_order: list[str] = Field(default_factory=list, description="The order in which constraints can be sacrificed if a solution is infeasible.")

try:
    with open("prompts/parser_system_prompt.md", "r") as f:
            PARSER_SYSTEM_PROMPT = f.read()
except FileNotFoundError:
        PARSER_SYSTEM_PROMPT = "You are an expert Intent-Based Networking (IBN) agent. Your task is to translate a network operator's natural language intent into a highly structured JSON format for a multi-objective service placement optimizer."


def intent_parser_node(state: AgentState) -> Dict[str, Any]:
    """Parses the natural language intent into a structured JSON dict."""
    
    # We use the base LLM for structured output via Pydantic
    structured_llm = llm.with_structured_output(IntentSchema)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", PARSER_SYSTEM_PROMPT),
        ("user", "{intent}")
    ])
    
    chain = prompt | structured_llm
    
    # Extract the user's intent from the state
    user_intent = state.get("user_intent", "")
    
    # Invoke the chain
    parsed_result = chain.invoke({"intent": user_intent})
    
    # Return the dictionary representation to update the state
    return {
            "original_parsed_intent": parsed_result.model_dump(),
            "active_parsed_intent": parsed_result.model_dump()
        }
