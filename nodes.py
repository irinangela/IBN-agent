import os
from dotenv import load_dotenv
from typing import Dict, Any, Literal, Optional
import copy
import json

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from state import AgentState
from tools.tools import ALL_TOOLS

from utils.retriever import get_warm_start_context

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
        SystemMessage(content=PARSER_SYSTEM_PROMPT),
        ("user", "{intent}")
    ])
    
    chain = prompt | structured_llm
    
    # Extract the user's intent from the state
    user_intent = state.get("user_intent", "")
    
    parsed_result = chain.invoke({"intent": user_intent})
    
    # Return the dictionary representation to update the state
    return {
            "original_parsed_intent": parsed_result.model_dump(),
            "active_parsed_intent": parsed_result.model_dump()
        }


### Weight Proposer Node

class ProposedWeights(BaseModel):
    best_historical_w1: float = Field(description="The EXACT w1 value from the single best historical run provided in the context.")
    best_historical_w2: float = Field(description="The EXACT w2 value from the single best historical run provided in the context.")
    best_historical_w3: float = Field(description="The EXACT w3 value from the single best historical run provided in the context.")
    reasoning: str = Field(description="Explain why you are making micro-adjustments (max +/- 0.15) to the historical baseline to satisfy the intent.")
    w1: float = Field(description="The adjusted Weight for Cost (between 0.0 and 1.0)")
    w2: float = Field(description="The adjusted Weight for Security (between 0.0 and 1.0)")
    w3: float = Field(description="The adjusted Weight for Latency (between 0.0 and 1.0)")

try:
    with open("prompts/proposer_system_prompt.md", "r") as f:
            PROPOSER_SYSTEM_PROMPT = f.read()
except FileNotFoundError:
        PROPOSER_SYSTEM_PROMPT = "You are an expert Intent-Based Networking (IBN) agent. Your task is to propose a set of weights for a multi-objective service placement optimizer based on the parsed intent and historical successful runs. Ensure that the weights sum to 1.0 and provide reasoning for your selection."

def weight_proposer_node(state: AgentState) -> Dict[str, Any]:
    parsed_intent = state["active_parsed_intent"]
    
    # Query the dataset containing historical successful runs to get a warm start context
    historical_context = get_warm_start_context(parsed_intent)
    
    structured_llm = llm.with_structured_output(ProposedWeights)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=PROPOSER_SYSTEM_PROMPT),
        ("user", "Parsed Intent:\n{intent}\n\nHistorical Context:\n{context}")
    ])
    
    chain = prompt | structured_llm
    result = chain.invoke({
        "intent": str(parsed_intent),
        "context": historical_context,
    })

    # Normalization of weights to ensure they sum to 1.0
    total = result.w1 + result.w2 + result.w3
    weights = {
        "w1": round(result.w1 / total, 3) if total > 0 else 0.333,
        "w2": round(result.w2 / total, 3) if total > 0 else 0.333,
        "w3": round(result.w3 / total, 3) if total > 0 else 0.334,
    }
    
    return {
        "historical_context": historical_context,
        "current_weights": weights,
        "reasoning": result.reasoning
    }


### Verifier Node

def verifier_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates the simulation results against the user's hard constraints.
    Returns deterministic feedback and a boolean success flag.
    """
    results = state.get("simulation_results", {})
    intent = state.get("active_parsed_intent", {})
    hard_constraints = intent.get("hard_constraints", [])
    
    metric_map = {
        "latency": "norm_lat",
        "cost": "norm_cost",
        "security": "norm_sec"
    }
    
    violations = []
    
    # Failures > 0 --> invalid run.
    failures = results.get("failures", 0)
    if failures > 0:
        violations.append(f"CRITICAL FAILURE: The optimizer failed to place {failures} applications. The weights are likely causing capacity bottlenecks or violating minimum security tiers.")
        
    # Check the numeric hard constraints
    for hc in hard_constraints:
        metric_name = hc.get("metric", "").lower()
        sim_key = metric_map.get(metric_name)
        
        if not sim_key or sim_key not in results:
            continue
            
        sim_val = results[sim_key]
        threshold = hc["threshold"]
        op = hc["operator"]
        
        is_violated = False
        if op in ["<=", "<"] and sim_val > threshold:
            is_violated = True
        elif op in [">=", ">"] and sim_val < threshold:
            is_violated = True
        elif op == "==" and sim_val != threshold:
            is_violated = True
            
        if is_violated:
            violations.append(f"Violated {metric_name}: Achieved {sim_val:.3f}, but required {op} {threshold}.")
            
    # Structured feedback for the next node
    is_satisfied = len(violations) == 0
    
    if is_satisfied:
        feedback = "SUCCESS: All hard constraints and placement requirements were strictly satisfied."
    else:
        feedback = "FAILED CONSTRAINTS:\n" + "\n".join(f"- {v}" for v in violations)
        
    return {
        "constraints_satisfied": is_satisfied,
        "verifier_feedback": feedback
    }


#### Recalibration Node

class RecalibrationOutput(BaseModel):
    reasoning: str = Field(description="Analyze the Verifier's feedback. Explain exactly how you are adjusting the weights or why a relaxation is necessary.")
    metric_to_relax: Optional[str] = Field(description="If constraints seem impossible to satisfy, output the exact metric name (e.g., 'cost', 'security') to drop based on the relaxation_order. Otherwise, return null/None.")
    w1: float = Field(description="New adjusted weight for Cost (w1)")
    w2: float = Field(description="New adjusted weight for Security (w2)")
    w3: float = Field(description="New adjusted weight for Latency (w3)")

try :
    with open("prompts/recalibration_system_prompt.md", "r") as f:
            RECALIBRATOR_SYSTEM_PROMPT = f.read()  
except FileNotFoundError:
        RECALIBRATOR_SYSTEM_PROMPT = "You are the Adaptive Search Engine for an Intent-Based Networking optimizer. Your task is to analyze the Verifier's feedback and propose new weights (w1, w2, w3) for the next optimization run. If the constraints seem impossible to satisfy, you may suggest relaxing one of the hard constraints based on the provided relaxation_order. Ensure that the weights sum to 1.0 and provide reasoning for your adjustments."  

def recalibrator_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyzes negative feedback and proposes new weights. 
    Handles constraint relaxation if the problem is infeasible.
    """
    original_intent = state.get("original_parsed_intent", {})
    parsed_intent = copy.deepcopy(state.get("active_parsed_intent", {}))
    current_weights = state.get("current_weights", {})
    feedback = state.get("verifier_feedback", "")
    history = state.get("recalibration_history", [])

    current_attempt_record = f"Attempted Weights: {current_weights} | Result: {feedback}"
    updated_history = history + [current_attempt_record]

    structured_llm = llm.with_structured_output(RecalibrationOutput)
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=RECALIBRATOR_SYSTEM_PROMPT),
        ("user", "Original Intent:\n{original}\n\nActive Intent (Current Constraints):\n{active}\n\nPast Attempts History:\n{history}\n\nVerifier Feedback on Latest Attempt:\n{feedback}")
    ])
    
    chain = prompt | structured_llm
    
    result = chain.invoke({
        "original": json.dumps(original_intent, indent=2),
        "active": json.dumps(parsed_intent, indent=2),
        "history": json.dumps(updated_history, indent=2),
        "feedback": feedback
    })
    
    # Constraint Relaxation
    relaxed_msg = ""
    if result.metric_to_relax:
        metric = result.metric_to_relax.lower()
        original_count = len(parsed_intent.get("hard_constraints", []))
        parsed_intent["hard_constraints"] = [
            hc for hc in parsed_intent.get("hard_constraints", []) 
            if hc.get("metric").lower() != metric
        ]
        
        if len(parsed_intent["hard_constraints"]) < original_count:
            relaxed_msg = f"\n[IMPORTANT]: Relaxed the '{metric}' constraint to find a feasible solution."
            if metric in parsed_intent.get("relaxation_order", []):
                parsed_intent["relaxation_order"].remove(metric)

    total = result.w1 + result.w2 + result.w3
    weights = {
        "w1": round(result.w1 / total, 3) if total > 0 else 0.333,
        "w2": round(result.w2 / total, 3) if total > 0 else 0.333,
        "w3": round(result.w3 / total, 3) if total > 0 else 0.334,
    }
    
    new_iteration_count = state.get("iteration_count", 0) + 1
    
    return {
        "current_weights": weights,
        "reasoning": result.reasoning + relaxed_msg,
        "active_parsed_intent": parsed_intent, 
        "iteration_count": new_iteration_count,
        "recalibration_history": updated_history
    }