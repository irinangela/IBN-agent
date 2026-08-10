from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    user_intent: str
    original_parsed_intent: Dict[str, Any]     # Set once by the Parser, NEVER changed.
    active_parsed_intent: Dict[str, Any]       # Modified by the recalibration node.
    historical_context: str
    current_weights: Dict[str, float]
    simulation_results: Dict[str, Any]
    iteration_count: int
    max_iterations: int
    final_response: str