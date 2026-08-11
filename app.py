import json
from typing import Literal
from langgraph.graph import StateGraph, END

from state import AgentState

from nodes import (
    intent_parser_node, 
    weight_proposer_node, 
    verifier_node, 
    recalibrator_node
)

from tools.tools import run_optimization_simulator

# Simulator: invokes the simulator tool with current weights and returns the results.
def simulator_node(state: AgentState):
    print("\n" + "="*50)
    print(f"ITERATION {state.get('iteration_count', 0)}")
    print("="*50)
    
    weights = state.get("current_weights", {})
    
    # Setting use_rollout=False for now so testing is fast. 
    # Switch to True for higher-quality runs.
    result_str = run_optimization_simulator.invoke({
        "w1": weights.get("w1", 0.33),
        "w2": weights.get("w2", 0.33),
        "w3": weights.get("w3", 0.34),
        "use_rollout": False 
    })
    
    results_dict = json.loads(result_str)
    
    return {"simulation_results": results_dict}


# Conditional Routing Logic
def route_verification(state: AgentState) -> Literal["end", "recalibrate"]:
    """
    Decides whether to finish the execution or loop back for recalibration.
    """
    # If the verifier passed all constraints, we are done!
    if state.get("constraints_satisfied", False):
        print("\n --> TARGET ACHIEVED! Exiting loop.")
        return "end"
    
    # If we hit the maximum iteration limit, we must stop to prevent an infinite loop.
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 5)
    
    if iteration_count >= max_iterations:
        print(f"\n --> MAX ITERATIONS ({max_iterations}) REACHED. Forcing exit.")
        return "end"
    
    # If constraints failed and we have iterations left: Loop back!
    print("\n --> CONSTRAINTS FAILED. Routing to Recalibrator...")
    return "recalibrate"


workflow = StateGraph(AgentState)

workflow.add_node("Parser", intent_parser_node)
workflow.add_node("Proposer", weight_proposer_node)
workflow.add_node("Simulator", simulator_node)
workflow.add_node("Verifier", verifier_node)
workflow.add_node("Recalibrator", recalibrator_node)

workflow.set_entry_point("Parser")
workflow.add_edge("Parser", "Proposer")
workflow.add_edge("Proposer", "Simulator")
workflow.add_edge("Simulator", "Verifier")

workflow.add_conditional_edges(
    "Verifier",
    route_verification,
    {
        "end": END,
        "recalibrate": "Recalibrator"
    }
)

workflow.add_edge("Recalibrator", "Simulator")

app = workflow.compile()