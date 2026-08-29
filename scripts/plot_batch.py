"""Aggregate thesis figures from every analytics JSON in a folder."""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.plots import save_batch_figures
from utils.plots.io import load_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        default=os.path.join(ROOT, "results-analytics"),
        help="Folder of *-run-*.json files",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "results-analytics", "aggregate"),
        help="Output directory",
    )
    args = parser.parse_args()
    runs = load_runs(args.dir)
    if not runs:
        print(f"No run JSONs found in {args.dir}")
        return
    written = save_batch_figures(runs, args.out)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
