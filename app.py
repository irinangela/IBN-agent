import json
from typing import Literal
from langgraph.graph import StateGraph, END

from state import AgentState

from nodes import (
    intent_parser_node,
    feasibility_node,
    weight_proposer_node,
    verifier_node,
    recalibrator_node,
)

from tools.tools import run_optimization_simulator


def simulator_node(state: AgentState):
    print("\n" + "="*50)
    print(f"ITERATION {state.get('iteration_count', 0)}")
    print("="*50)

    weights = state.get("current_weights", {})

    result_str = run_optimization_simulator.invoke({
        "w1": weights.get("w1", 0.33),
        "w2": weights.get("w2", 0.33),
        "w3": weights.get("w3", 0.34),
        "use_rollout": bool(state.get("use_rollout", False)),
    })

    results_dict = json.loads(result_str)

    return {"simulation_results": results_dict}


def route_verification(state: AgentState) -> Literal["end", "recalibrate"]:
    """
    Decides whether to finish the execution or loop back for recalibration.
    """
    if state.get("constraints_satisfied", False):
        print("\n --> TARGET ACHIEVED! Exiting loop.")
        return "end"

    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 5)

    if iteration_count >= max_iterations:
        print(f"\n --> MAX ITERATIONS ({max_iterations}) REACHED. Forcing exit.")
        return "end"

    print("\n --> CONSTRAINTS FAILED. Routing to Recalibrator...")
    return "recalibrate"


parse_workflow = StateGraph(AgentState)
parse_workflow.add_node("Parser", intent_parser_node)
parse_workflow.add_node("Feasibility", feasibility_node)
parse_workflow.set_entry_point("Parser")
parse_workflow.add_edge("Parser", "Feasibility")
parse_workflow.add_edge("Feasibility", END)
parse_app = parse_workflow.compile()

search_workflow = StateGraph(AgentState)
search_workflow.add_node("Proposer", weight_proposer_node)
search_workflow.add_node("Simulator", simulator_node)
search_workflow.add_node("Verifier", verifier_node)
search_workflow.add_node("Recalibrator", recalibrator_node)
search_workflow.set_entry_point("Proposer")
search_workflow.add_edge("Proposer", "Simulator")
search_workflow.add_edge("Simulator", "Verifier")
search_workflow.add_conditional_edges(
    "Verifier",
    route_verification,
    {
        "end": END,
        "recalibrate": "Recalibrator",
    },
)
search_workflow.add_edge("Recalibrator", "Simulator")
search_app = search_workflow.compile()

# Search loop only. Streamlit runs parse_app first so HITL can pause.
app = search_app
