"""Generate per-run figures from an analytics JSON."""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.plots import save_run_figures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True, help="Path to a results-analytics run JSON")
    parser.add_argument("--out", default=None, help="Output directory (default: next to the JSON)")
    args = parser.parse_args()
    written = save_run_figures(args.json, out_dir=args.out)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
