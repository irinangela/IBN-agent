"""Load analytics JSONs and historical CSVs."""

import json
import os

import pandas as pd

from simulation.config import SEED

from utils.plots.style import AVG_KEYS, NORM_KEYS


def load_run(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_runs(directory: str) -> list:
    """Every *-run-*.json in a folder, sorted by filename.
    Figure files are not .json, so they are ignored.
    """
    runs = []
    if not os.path.isdir(directory):
        return runs
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        if "-run-" not in name:
            continue
        runs.append(load_run(os.path.join(directory, name)))
    return runs


def attempts_of(run: dict) -> list:
    """Per-iteration rows. Falls back to the final snapshot for older JSONs."""
    attempts = run.get("attempts") or []
    if attempts:
        return attempts
    weights = run.get("final_weights") or {}
    metrics = run.get("final_metrics") or {}
    if not weights and not metrics:
        return []
    row = {
        "attempt": run.get("iterations_done") or 1,
        "w1": weights.get("w1"),
        "w2": weights.get("w2"),
        "w3": weights.get("w3"),
        "feedback": "",
        "active_thresholds": {},
    }
    row.update({key: metrics.get(key) for key in AVG_KEYS.values()})
    row.update({key: metrics.get(key) for key in NORM_KEYS.values()})
    return [row]


def load_valid_runs(csv_path: str, seed: int = SEED) -> pd.DataFrame:
    """Load the historical runs from the CSV file.
    Calculate the average cost, security, and latency per app.
    Filter the runs by the seed if provided.
    """
    frame = pd.read_csv(csv_path)
    if frame.empty:
        return frame
    n_apps = frame["successful_apps"].replace(0, pd.NA)
    frame = frame.copy()
    frame["avg_cost"] = frame["total_cost"] / n_apps
    frame["avg_security"] = frame["total_security"] / n_apps
    frame["avg_latency"] = frame["total_latency"] / n_apps
    if seed is not None:
        frame = frame[frame["seed"] == seed]
    return frame.reset_index(drop=True)


def valid_failure_free(frame: pd.DataFrame) -> pd.DataFrame:
    """Filter the runs by the valid and failure flags."""
    if frame.empty:
        return frame.copy()
    return frame[(frame["valid"] == True) & (frame["failures"] == 0)].copy()
