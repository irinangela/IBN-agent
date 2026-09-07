from typing import List, TypedDict, Dict, Any

class AgentState(TypedDict):
    user_intent: str
    original_parsed_intent: Dict[str, Any]          # Set once by the Parser, NEVER changed.
    active_parsed_intent: Dict[str, Any]            # Modified by the recalibration node.
    historical_context: str                         # Warm start context for the proposer.

    feasibility_report: Dict[str, Any]              # Report based on historical contexts.
    current_weights: Dict[str, float]               # The simulator gets the weights from the state.
    reasoning: str                                  # LLM reasoning for the next recalibration.
    simulation_results: Dict[str, Any]              # Feedback for the next recalibration.
    constraints_satisfied: bool                     # Default: False. If True, the agent has achieved the user's intent.
    verifier_feedback: str                          # Success or failure message.
    recalibration_history: List[Dict[str, Any]]     # Structured past attempts (weights + avg_* metrics). 

    iteration_count: int                            # Number of iterations completed.
    max_iterations: int                             # Maximum number of iterations allowed.
    total_input_tokens: int                         # Total input tokens used by the agent.
    total_output_tokens: int                        # Total output tokens used by the agent.

    use_rollout: bool                               # Default: False (uses best_fit).
    needs_operator: bool                            # Indicator for HITL 
    operator_decision: Dict[str, Any]               # Decision from the operator.
    parse_failed: bool                              # Default: False. If True, the parse failed.
    parse_feedback: str                             # Warning message if the parse failed.