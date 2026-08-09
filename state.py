from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    user_intent: str
    parsed_intent: Dict[str, Any]
    current_weights: Dict[str, float]
    simulation_results: Dict[str, Any]
    iteration_count: int
    max_iterations: int
    final_response: str