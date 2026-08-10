"""
Rollout worker bootstrap for the thesis setup.

On Windows, ProcessPoolExecutor workers spawn fresh interpreters. They must
receive the thesis config module and the active (W1, W2, W3) weights before
heuristics / greedy_fast import `config`.
"""

from __future__ import annotations

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _ensure_import_paths() -> None:
    if THIS_DIR not in sys.path:
        sys.path.insert(0, THIS_DIR)


def init_rollout_worker(w1: float, w2: float, w3: float) -> None:
    """ProcessPoolExecutor initializer: bind thesis config + run weights."""
    _ensure_import_paths()
    import config as thesis_cfg

    thesis_cfg.W1 = float(w1)
    thesis_cfg.W2 = float(w2)
    thesis_cfg.W3 = float(w3)
    sys.modules["config"] = thesis_cfg


def _apply_run_weights(w1: float, w2: float, w3: float) -> None:
    """Set weights on every module that imported config before scoring."""
    _ensure_import_paths()
    import config as thesis_cfg

    thesis_cfg.W1 = float(w1)
    thesis_cfg.W2 = float(w2)
    thesis_cfg.W3 = float(w3)
    sys.modules["config"] = thesis_cfg

    import heuristics

    heuristics.cfg.W1 = float(w1)
    heuristics.cfg.W2 = float(w2)
    heuristics.cfg.W3 = float(w3)


def evaluate_batch_thesis(args):
    """
    Worker entry point: apply run weights, then delegate to repo evaluate_batch_v2.
    """
    topo_master, app_master, ms_index, candidates, use_fast, w1, w2, w3 = args
    _apply_run_weights(w1, w2, w3)

    from heuristics import evaluate_batch_v2

    return evaluate_batch_v2((topo_master, app_master, ms_index, candidates, use_fast))
