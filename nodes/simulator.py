import json
from typing import Any, Dict

from state import AgentState
from tools.tools import run_optimization_simulator


def simulator_node(state: AgentState) -> Dict[str, Any]:
    """Run the placement simulator with the current weights from state."""
    iteration_count = int(state.get("iteration_count") or 0) + 1
    print("\n" + "=" * 50)
    print(f"ITERATION {iteration_count}")
    print("=" * 50)

    weights = state.get("current_weights", {})

    result_str = run_optimization_simulator.invoke({
        "w1": weights.get("w1", 0.33),
        "w2": weights.get("w2", 0.33),
        "w3": weights.get("w3", 0.34),
        "use_rollout": bool(state.get("use_rollout", False)),
    })

    results_dict = json.loads(result_str)

    return {
        "simulation_results": results_dict,
        "iteration_count": iteration_count,
    }
