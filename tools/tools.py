import os
import sys
import json
from langchain_core.tools import tool

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

_CACHED_TOPO = None
_CACHED_APPS = None

@tool
def run_optimization_simulator(w1: float, w2: float, w3: float, use_rollout: bool = False) -> str:
    """
    Runs the black-box heuristic optimizer to place microservices on the edge-cloud continuum.
    Pass the proposed mathematical weights: w1 (Cost), w2 (Security), and w3 (Latency).
    Returns a JSON string containing mean operator-unit metrics (avg_cost, avg_sec, avg_latency),
    normalized aggregates (norm_*), totals, and failures.
    """
    global _CACHED_TOPO, _CACHED_APPS
    
    import simulation.config as cfg
    import simulation.generator
    from simulation.main import run_best_fit, run_app_rollout, app_security_score
    import numpy as np
    import random
    
    # Inject the LLM's proposed weights into the config
    cfg.W1 = float(w1)
    cfg.W2 = float(w2)
    cfg.W3 = float(w3)
    sys.modules["config"] = cfg

    # Generate and cache topology/workload (Only runs once per session)
    if _CACHED_TOPO is None or _CACHED_APPS is None:
        print("\n[Tool] Initializing topology and workload cache...")
        np.random.seed(cfg.SEED)
        random.seed(cfg.SEED)
        _CACHED_TOPO = simulation.generator.generate_topology()
        _CACHED_APPS = simulation.generator.generate_workload()
        _CACHED_APPS.sort(key=app_security_score, reverse=True)

    print(f"\n[Tool] Running Simulator -> w1:{w1:.2f}, w2:{w2:.2f}, w3:{w3:.2f} (Rollout: {use_rollout})")
    
    if use_rollout:
        result = run_app_rollout(_CACHED_TOPO, _CACHED_APPS)
    else:
        result = run_best_fit(_CACHED_TOPO, _CACHED_APPS)
        
    summary = result["summary"]
    n = max(int(summary.get("successful_apps", 0)), 1)
    
    output_dict = {
        "failures": summary["failures"],
        "successful_apps": summary["successful_apps"],
        "total_cost": round(summary["total_cost"], 3),
        "total_security": round(summary["total_security"], 3),
        "total_latency": round(summary["total_latency"], 3),
        "avg_cost": round(summary["total_cost"] / n, 3),
        "avg_security": round(summary["total_security"] / n, 3),
        "avg_latency": round(summary["total_latency"] / n, 3),
        "norm_cost": round(summary["norm_cost"], 3),
        "norm_sec": round(summary["norm_sec"], 3),
        "norm_lat": round(summary["norm_lat"], 3),
        "total_score": round(summary["score"], 3)
    }
    
    return json.dumps(output_dict, indent=2)

ALL_TOOLS = [run_optimization_simulator]
