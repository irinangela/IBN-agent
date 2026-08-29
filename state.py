from typing import List, TypedDict, Dict, Any

class AgentState(TypedDict):
    user_intent: str
    original_parsed_intent: Dict[str, Any]     # Set once by the Parser, NEVER changed.
    active_parsed_intent: Dict[str, Any]       # Modified by the recalibration node.
    historical_context: str
    current_weights: Dict[str, float]
    reasoning: str
    simulation_results: Dict[str, Any]
    iteration_count: int
    max_iterations: int
    
    constraints_satisfied: bool
    verifier_feedback: str
    recalibration_history: List[Dict[str, Any]]  # Structured past attempts (weights + avg_* metrics). 

    final_response: str

    total_input_tokens: int
    total_output_tokens: int

    use_rollout: bool
    feasibility_report: Dict[str, Any]
    needs_operator: bool
    operator_decision: Dict[str, Any]
    parse_failed: bool
    parse_feedback: str