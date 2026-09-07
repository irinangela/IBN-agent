from typing import Literal
from langgraph.graph import StateGraph, END

from state import AgentState

from nodes import (
    intent_parser_node,
    feasibility_node,
    weight_proposer_node,
    simulator_node,
    verifier_node,
    recalibrator_node,
)


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


def route_after_parser(state: AgentState) -> Literal["feasibility", "end"]:
    if state.get("parse_failed"):
        return "end"
    return "feasibility"


parse_workflow = StateGraph(AgentState)
parse_workflow.add_node("Parser", intent_parser_node)
parse_workflow.add_node("Feasibility", feasibility_node)
parse_workflow.set_entry_point("Parser")
parse_workflow.add_conditional_edges(
    "Parser",
    route_after_parser,
    {"feasibility": "Feasibility", "end": END},
)
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
