"""Dataset figures from simulation/runs/valid_runs.csv."""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from simulation.config import SEED
from utils.plots import save_dataset_figures
from utils.plots.io import load_run, load_valid_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default=os.path.join(ROOT, "simulation", "runs", "valid_runs.csv"),
        help="Historical sweep CSV",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Topology/workload seed (default: live seed)")
    parser.add_argument("--algorithm", default="best_fit", help="Algorithm for rugged-front and Pareto panels")
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "results-analytics", "dataset"),
        help="Output directory",
    )
    parser.add_argument("--run", default=None, help="Optional analytics JSON to overlay as an agent path")
    args = parser.parse_args()

    frame = load_valid_runs(args.csv, seed=args.seed)
    run = load_run(args.run) if args.run else None
    written = save_dataset_figures(frame, args.out, algorithm=args.algorithm, run=run)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
