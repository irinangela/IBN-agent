"""
simulator_tools.py

Wraps the Thesis_multi_obj_setup service-placement simulator as a LangChain
tool the agent can call repeatedly with different (w1, w2, w3) weight
vectors -- WITHOUT regenerating the topology/workload each time.

SETUP REQUIRED before this will import/run:
  1. Copy these files from Thesis_multi_obj_setup into this agent's folder
     (or add that project's folder to PYTHONPATH):
         generator.py, structs.py, heuristics.py, greedy_fast.py, config.py
     (rollout_worker.py is only needed if you later add an app_rollout tool)
  2. Fix every line marked  # CHECK  below against the real function
     signatures -- these are my best guess from your README, not verified
     against the actual source of generator.py / greedy_fast.py.

WHY the caching matters:
  The agent is supposed to navigate ONE black-box landscape by varying
  weights (that's the whole "rugged Pareto front" experiment). If we
  regenerated topology+workload on every tool call, each weight vector
  would be evaluated against a *different* random problem, and comparisons
  between calls would be meaningless. So: generate once per agent session,
  cache, and reuse for every run_spp_simulator call in that session.

WHY best_fit (not app_rollout):
  best_fit takes ~0.2s/call. app_rollout takes 30-90+s/call in your own
  results.json data. The agent will call this tool many times per
  conversation (that's the point of Testing Cases A and B) -- app_rollout
  would make a single conversation take many minutes to hours. best_fit
  being fast-but-imperfect is also exactly what produces the ruggedness
  your thesis is studying, so this is the correct tool, not a shortcut.
"""

import threading

from langchain_core.tools import tool

import simulation.config
from simulation.generator import generate_topology, generate_workload
from simulation.greedy_fast import resolve_greedy_allocator


# --- one shared instance per process, generated on first use ---------------
_lock = threading.Lock()
_cached = {"topology": None, "apps": None}


def _get_instance():
    """Generate the topology + workload ONCE per agent process and reuse it
    for every subsequent run_spp_simulator call."""
    with _lock:
        if _cached["topology"] is None:
            _cached["topology"] = generate_topology()           
            _cached["apps"] = generate_workload()                    # CHECK: args?
        return _cached["topology"], _cached["apps"]


def reset_instance():
    """Call this if you ever want a fresh random topology/workload for a
    new experiment run (e.g. between test cases, or between seeds)."""
    with _lock:
        _cached["topology"] = None
        _cached["apps"] = None


@tool
def run_spp_simulator(w1: float, w2: float, w3: float) -> dict:
    """
    Runs the fast greedy service-placement heuristic for the given
    objective weights (w1=cost, w2=security, w3=latency) against the
    current problem instance, and returns the resulting placement quality.

    This implementation:
      - sets the global weight values in the simulation.config module
      - deep-copies the cached apps so we don't mutate the shared instance
      - constructs the allocator with resolve_greedy_allocator(topo)
      - calls allocator.solve(apps_copy)
      - computes aggregate metrics from the solved apps
    """
    import copy
    from simulation import config as simcfg
    from simulation.heuristics import calculate_app_score_static, get_app_normalized_metrics

    topo, apps = _get_instance()

    # Apply requested objective weights into the simulation config module
    simcfg.W1 = float(w1)
    simcfg.W2 = float(w2)
    simcfg.W3 = float(w3)

    # Work on a deep copy so cached apps remain unmodified across tool calls
    apps_copy = copy.deepcopy(apps)

    # Create allocator for this topology and solve for the copied workload
    allocator = resolve_greedy_allocator(topo)
    allocator.solve(apps_copy)

    # Aggregate results
    failures = 0
    norm_metrics = []
    scores = []

    for app in apps_copy:
        # An app is considered failed if any microservice remained unassigned
        incomplete = any(ms.assigned_machine_id == -1 for ms in app.microservices)
        if incomplete:
            failures += 1
        else:
            nm = get_app_normalized_metrics(topo, app)
            if nm is not None:
                norm_metrics.append(nm)

        # score can be inf for infeasible apps
        scores.append(calculate_app_score_static(topo, app))

    successful_apps = len(apps_copy) - failures

    # Compute averages over successfully placed apps / valid normalized metrics
    if norm_metrics:
        avg_cost = float(sum(n[0] for n in norm_metrics) / len(norm_metrics))
        avg_sec = float(sum(n[1] for n in norm_metrics) / len(norm_metrics))
        avg_lat = float(sum(n[2] for n in norm_metrics) / len(norm_metrics))
    else:
        avg_cost = float("nan")
        avg_sec = float("nan")
        avg_lat = float("nan")

    # Overall score: mean of finite per-app scores, or +inf if none finite
    import math
    finite_scores = [s for s in scores if math.isfinite(s)]
    overall_score = float(sum(finite_scores) / len(finite_scores)) if finite_scores else float("inf")

    return {
        "score": overall_score,
        "failures": failures,
        "successful_apps": successful_apps,
        "norm_cost": avg_cost,
        "norm_sec": avg_sec,
        "norm_lat": avg_lat,
    }


ALL_SIM_TOOLS = [run_spp_simulator]