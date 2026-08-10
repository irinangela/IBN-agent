"""
Thesis multi-objective runner.

Runs:
  - Best-fit heuristic (GreedyAllocatorFast when USE_FAST_GREEDY=True)
  - App Rollout (RolloutAllocatorV2 with weight-aware workers)

over a sweep of (W1, W2, W3) weight triplets loaded from `config.py`.
"""

from __future__ import annotations

import concurrent.futures
import copy
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import numpy as np

import config as cfg

# Ensure local modules resolve config from this folder.
sys.modules["config"] = cfg

import generator
from greedy_fast import resolve_greedy_allocator
from heuristics import (
    RolloutAllocatorV2,
    calculate_app_score_static,
    compute_app_global_bounds,
    get_app_normalized_metrics,
    get_app_raw_metrics,
)
from rollout_worker import evaluate_batch_thesis, init_rollout_worker


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def app_security_score(app) -> Tuple[int, int, float, int]:
    """
    Ordering heuristic (same idea as repo's `mainv2.py`):
    place apps with hard/high-security services earlier.
    """
    weighted_demand = 0.0
    has_tier_3 = False
    count_tier_3 = 0
    count_tier_2 = 0

    for ms in app.microservices:
        if ms.min_security == 3:
            has_tier_3 = True
            count_tier_3 += 1
        elif ms.min_security == 2:
            count_tier_2 += 1

        if ms.min_security >= 1:
            effective_cpu = ms.cpu_demand * ms.security_overheads[ms.min_security]
            weighted_demand += ms.min_security * effective_cpu

    return (1 if has_tier_3 else 0, count_tier_3, weighted_demand, count_tier_2)


def _summarize_run(
    algorithm: str,
    topo_run,
    apps_run,
    duration_s: float,
) -> Dict[str, Any]:
    total_score = 0.0
    failures = 0
    total_cost = 0.0
    total_sec = 0.0
    total_lat = 0.0
    norm_cost = 0.0
    norm_sec = 0.0
    norm_lat = 0.0
    successful_apps = 0

    for app in apps_run:
        s = calculate_app_score_static(topo_run, app)
        if s == float("inf"):
            failures += 1
            continue

        successful_apps += 1
        total_score += s
        c, sec, lat = get_app_raw_metrics(topo_run, app)
        total_cost += c
        total_sec += sec
        total_lat += lat

        nc, ns, nl = get_app_normalized_metrics(topo_run, app)
        if nc is None:
            failures += 1
            continue
        norm_cost += nc
        norm_sec += ns
        norm_lat += nl

    return {
        "algorithm": algorithm,
        "duration_s": duration_s,
        "weights": {"W1": cfg.W1, "W2": cfg.W2, "W3": cfg.W3},
        "summary": {
            "score": total_score,
            "failures": failures,
            "successful_apps": successful_apps,
            "total_cost": total_cost,
            "total_security": total_sec,
            "total_latency": total_lat,
            "norm_cost": norm_cost,
            "norm_sec": norm_sec,
            "norm_lat": norm_lat,
        },
    }


def run_best_fit(topo, apps) -> Dict[str, Any]:
    topo_run = copy.deepcopy(topo)
    apps_run = copy.deepcopy(apps)

    allocator = resolve_greedy_allocator(topo_run)
    t0 = time.time()
    allocator.solve(apps_run)
    return _summarize_run("best_fit", topo_run, apps_run, time.time() - t0)


class ThesisRolloutAllocatorV2(RolloutAllocatorV2):
    """RolloutAllocatorV2 with weight-aware worker pool for thesis runs."""

    def __init__(self, topo, w1: float, w2: float, w3: float, max_branching=None):
        self._w1 = float(w1)
        self._w2 = float(w2)
        self._w3 = float(w3)
        super().__init__(topo, max_branching=max_branching)

    def solve(self, apps: List):
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=cfg.MAX_WORKERS,
            initializer=init_rollout_worker,
            initargs=(self._w1, self._w2, self._w3),
        ) as executor:
            n_apps = len(apps)
            for app_idx, app in enumerate(apps):
                app.global_bounds = compute_app_global_bounds(app)
                app.mtls_decisions = [False] * (len(app.microservices) - 1)
                self.rollout_app(app, executor)
                if getattr(cfg, "ROLLOUT_PROGRESS", False):
                    print(f"  [App-Rollout] {app_idx + 1}/{n_apps} apps", flush=True)

    def rollout_app(self, app, executor):
        for i, ms in enumerate(app.microservices):
            candidates = self._get_candidates(app, ms, i)

            if not candidates:
                self.greedy._rollback(app, i)
                return False

            self.total_heuristic_calls += len(candidates)

            chunk_size = max(1, len(candidates) // cfg.MAX_WORKERS)
            chunks = [candidates[k:k + chunk_size] for k in range(0, len(candidates), chunk_size)]

            tasks = [
                (self.topo, app, i, chunk, self._use_fast_greedy, self._w1, self._w2, self._w3)
                for chunk in chunks
            ]

            results = executor.map(evaluate_batch_thesis, tasks)

            best_res = None
            best_s = float("inf")

            for res in results:
                if res and res[3] < best_s:
                    best_s = res[3]
                    best_res = res

            if best_res:
                m, t, use_mtls, _s = best_res
                self.greedy._apply_assignment(ms, m, t, use_mtls, app, i)
            else:
                self.greedy._rollback(app, i)
                return False
        return True


def run_app_rollout(topo, apps) -> Dict[str, Any]:
    topo_run = copy.deepcopy(topo)
    apps_run = copy.deepcopy(apps)

    allocator = ThesisRolloutAllocatorV2(topo_run, cfg.W1, cfg.W2, cfg.W3)
    t0 = time.time()
    allocator.solve(apps_run)
    return _summarize_run("app_rollout", topo_run, apps_run, time.time() - t0)


def main() -> None:
    os.makedirs(THIS_DIR, exist_ok=True)
    results_path = os.path.join(THIS_DIR, "results.json")

    set_seeds(cfg.SEED)
    print("=== Thesis multi-objective setup ===")
    print(f"Seed: {cfg.SEED}")
    print(f"Topology: {cfg.NUM_EDGE} edge + {cfg.NUM_FOG} fog + {cfg.NUM_CLOUD} cloud")
    print(f"Clusters: {cfg.NUM_CLUSTERS}, Apps: {cfg.NUM_APPS}")
    print(f"Weights to sweep: {cfg.WEIGHT_SETS}")

    print("\nGenerating topology + workload...")
    topo = generator.generate_topology()
    apps = generator.generate_workload()
    apps.sort(key=app_security_score, reverse=True)
    print(f"Generated: topo.num_nodes={topo.num_nodes}, apps={len(apps)}")

    all_results: Dict[str, Dict] = {}

    for (w1, w2, w3) in cfg.WEIGHT_SETS:
        cfg.W1 = float(w1)
        cfg.W2 = float(w2)
        cfg.W3 = float(w3)
        sys.modules["config"] = cfg

        print("\n--------------------------------------------")
        print(f"Running for weights: W1={cfg.W1}, W2={cfg.W2}, W3={cfg.W3}")

        weight_key = f"w1={cfg.W1}_w2={cfg.W2}_w3={cfg.W3}"
        run_out: Dict[str, Dict] = {}

        if getattr(cfg, "RUN_BEST_FIT", True):
            set_seeds(cfg.SEED)
            print("Running: best-fit heuristic...")
            t0 = time.time()
            run_out["best_fit"] = run_best_fit(topo, apps)
            run_out["best_fit"]["total_wall_s"] = time.time() - t0

        if getattr(cfg, "RUN_APP_ROLLOUT", True):
            set_seeds(cfg.SEED)
            print("Running: app rollout (RolloutAllocatorV2)...")
            t0 = time.time()
            run_out["app_rollout"] = run_app_rollout(topo, apps)
            run_out["app_rollout"]["total_wall_s"] = time.time() - t0

        all_results[weight_key] = run_out

        for k, v in run_out.items():
            s = v["summary"]["score"]
            fails = v["summary"]["failures"]
            print(f"  {k}: score={s:.4f}, failures={fails}")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nFinished. Results written to: {results_path}")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
