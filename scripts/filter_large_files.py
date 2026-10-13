"""Move the largest tiles exceeding a size threshold into a target directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow running this script directly from the repo root as `python scripts/filter_large_files.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sateline_ml.gee import find_max_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Directory containing downloaded tiles")
    parser.add_argument("target", help="Directory where the largest tiles will be moved")
    parser.add_argument("--size-limit", type=int, default=15 * 1024 * 1024, help="Size threshold in bytes")
    args = parser.parse_args()

    dest = find_max_size(args.source, args.size_limit, args.target)
    print(f"✅ Selected tiles moved to {dest}")


if __name__ == "__main__":
    main()
