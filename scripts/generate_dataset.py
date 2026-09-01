"""Build the near-neighbor warm-start dataset.

Generates topology + workload for each seed under the collapsed config, then
sweeps a weight grid with best_fit and app_rollout.

Default seeds are DATASET_SEEDS: the whole WARM_START_SEEDS pool plus the live
SEED, so every retrieval condition (oracle, K=1, K=2, K=5, full pool) reads from one
CSV instead of needing its own sweep.

Resumes on (seed, weights, algorithm), so interrupting costs at most one run.

Examples:
    python scripts/generate_dataset.py --algorithms best_fit,app_rollout --dry-run
    python scripts/generate_dataset.py --algorithms best_fit,app_rollout
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from typing import List, Sequence, Set, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd

from simulation import config as cfg

sys.modules["config"] = cfg

from simulation import generator
from simulation.main import app_security_score, run_app_rollout, run_best_fit, set_seeds

DEFAULT_OUT = os.path.join(ROOT, "simulation", "runs", "valid_runs.csv")
LEGACY_BACKUP = os.path.join(ROOT, "simulation", "runs", "valid_runs_legacy_wide.csv")

# Rough per-run estimation time.
SECONDS_PER_RUN = {"best_fit": 0.25, "app_rollout": 45.0}

CSV_COLUMNS = [
    "seed",
    "weight_key",
    "W1",
    "W2",
    "W3",
    "algorithm",
    "score",
    "failures",
    "successful_apps",
    "total_cost",
    "total_security",
    "total_latency",
    "norm_cost",
    "norm_sec",
    "norm_lat",
    "duration_s",
    "total_wall_s",
    "valid",
]


def _parse_seeds(raw: str | None) -> Tuple[int, ...]:
    if not raw:
        return tuple(cfg.DATASET_SEEDS)
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _parse_algorithms(raw: str) -> List[str]:
    allowed = {"best_fit", "app_rollout"}
    names = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise SystemExit(f"Unknown algorithm(s): {unknown}. Use best_fit and/or app_rollout.")
    if not names:
        raise SystemExit("Need at least one algorithm.")
    return names


def _weight_key(w1: float, w2: float, w3: float) -> str:
    return f"w1={w1}_w2={w2}_w3={w3}"


def _row_key(seed: int, w1: float, w2: float, w3: float, algorithm: str) -> Tuple:
    return (int(seed), round(float(w1), 10), round(float(w2), 10), round(float(w3), 10), algorithm)


def _atomic_write(frame: pd.DataFrame, path: str) -> None:
    tmp = f"{path}.tmp"
    last_error = None
    for attempt in range(6):
        try:
            frame.to_csv(tmp, index=False)
            os.replace(tmp, path)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.3 * (attempt + 1))
    if last_error:
        raise last_error


def _backup_legacy_if_needed(path: str) -> bool:
    """True when the file is a pre-collapse sweep and must not be resumed."""
    if not os.path.isfile(path):
        return False
    existing = pd.read_csv(path)
    if existing.empty or "seed" not in existing.columns:
        return False
    seeds = set(existing["seed"].astype(int).unique())
    if any(seed >= 200 for seed in seeds):
        return False
    if not os.path.isfile(LEGACY_BACKUP):
        shutil.copy2(path, LEGACY_BACKUP)
        print(f"Backed up previous wide-random CSV to {LEGACY_BACKUP}")
    else:
        print(f"Legacy backup already exists at {LEGACY_BACKUP}; not overwriting it.")
    return True


def _load_done(path: str) -> tuple[pd.DataFrame, Set[Tuple]]:
    if not os.path.isfile(path):
        return pd.DataFrame(columns=CSV_COLUMNS), set()
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame(columns=CSV_COLUMNS), set()
    done = {
        _row_key(row.seed, row.W1, row.W2, row.W3, row.algorithm)
        for row in frame.itertuples(index=False)
    }
    return frame, done


def _result_row(seed: int, w1: float, w2: float, w3: float, result: dict) -> dict:
    summary = result["summary"]
    failures = int(summary["failures"])
    return {
        "seed": int(seed),
        "weight_key": _weight_key(w1, w2, w3),
        "W1": float(w1),
        "W2": float(w2),
        "W3": float(w3),
        "algorithm": result["algorithm"],
        "score": summary["score"],
        "failures": failures,
        "successful_apps": summary["successful_apps"],
        "total_cost": summary["total_cost"],
        "total_security": summary["total_security"],
        "total_latency": summary["total_latency"],
        "norm_cost": summary["norm_cost"],
        "norm_sec": summary["norm_sec"],
        "norm_lat": summary["norm_lat"],
        "duration_s": result["duration_s"],
        "total_wall_s": result.get("total_wall_s", result["duration_s"]),
        "valid": failures == 0,
    }


def _generate_instance(seed: int):
    set_seeds(seed)
    topo = generator.generate_topology()
    apps = generator.generate_workload()
    apps.sort(key=app_security_score, reverse=True)
    return topo, apps


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds (default: DATASET_SEEDS = pool + live)",
    )
    parser.add_argument(
        "--algorithms",
        default="best_fit",
        help="Comma-separated: best_fit, app_rollout (default: best_fit)",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="Output CSV (default: valid_runs.csv)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and the time estimate, then exit without simulating",
    )
    args = parser.parse_args(argv)

    seeds = _parse_seeds(args.seeds)
    algorithms = _parse_algorithms(args.algorithms)
    weights: Tuple[Tuple[float, float, float], ...] = tuple(cfg.WARM_START_WEIGHT_SETS)

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if _backup_legacy_if_needed(out_path):
        existing, done = pd.DataFrame(columns=CSV_COLUMNS), set()
        print("Starting a fresh near-neighbor CSV (legacy rows will not be resumed).")
    else:
        existing, done = _load_done(out_path)

    rows: List[dict] = [] if existing.empty else existing.to_dict("records")

    todo = [
        (seed, algorithm)
        for seed in seeds
        for w1, w2, w3 in weights
        for algorithm in algorithms
        if _row_key(seed, w1, w2, w3, algorithm) not in done
    ]
    estimate_s = sum(SECONDS_PER_RUN.get(algorithm, 1.0) for _, algorithm in todo)

    print(
        f"Instance family: SEED(live)={cfg.SEED}, WARM_START_SEEDS={cfg.WARM_START_SEEDS}, "
        f"WARM_START_MODE={cfg.WARM_START_MODE}, WARM_START_K={cfg.WARM_START_K}"
    )
    print(f"Sweeping seeds={seeds} weights={len(weights)} algorithms={algorithms}")
    print(f"Target rows: {len(seeds) * len(weights) * len(algorithms)}")
    print(f"Already done: {len(done)} | remaining: {len(todo)}")
    print(f"Rough estimate: {estimate_s / 60.0:.0f} min ({estimate_s / 3600.0:.1f} h)")
    print(f"Writing {out_path}")

    if args.dry_run:
        print("\nDry run: nothing simulated.")
        return

    started = time.time()
    completed = 0
    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        topo, apps = _generate_instance(seed)
        for w1, w2, w3 in weights:
            cfg.W1, cfg.W2, cfg.W3 = float(w1), float(w2), float(w3)
            sys.modules["config"] = cfg
            for algorithm in algorithms:
                key = _row_key(seed, w1, w2, w3, algorithm)
                if key in done:
                    continue
                set_seeds(seed)
                t0 = time.time()
                if algorithm == "best_fit":
                    result = run_best_fit(topo, apps)
                else:
                    result = run_app_rollout(topo, apps)
                result["total_wall_s"] = time.time() - t0
                row = _result_row(seed, w1, w2, w3, result)
                rows.append(row)
                done.add(key)
                completed += 1
                flag = "OK" if row["valid"] else "FAIL"
                remaining = len(todo) - completed
                eta_min = (time.time() - started) / completed * remaining / 60.0
                print(
                    f"  [{completed}/{len(todo)}] {algorithm:12} "
                    f"w=({w1:.1f},{w2:.1f},{w3:.1f}) "
                    f"score={row['score']:.4f} failures={row['failures']} {flag} "
                    f"| ETA {eta_min:.0f} min",
                    flush=True,
                )
                _atomic_write(pd.DataFrame(rows, columns=CSV_COLUMNS), out_path)

    frame = pd.DataFrame(rows, columns=CSV_COLUMNS)
    _atomic_write(frame, out_path)
    n_invalid = int((frame["valid"] == False).sum()) if not frame.empty else 0
    print(
        f"\nFinished {len(frame)} rows ({n_invalid} invalid) in "
        f"{(time.time() - started) / 60.0:.0f} min -> {out_path}"
    )
    if n_invalid:
        print("WARNING: some placements failed. Treat those rows as invalid warm-start points.")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
