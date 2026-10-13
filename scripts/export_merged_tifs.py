"""Copy current and legacy merged Sentinel-2 products into a flat directory."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sateline_ml.workflows import export_merged_tifs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_dir", help="Download root with 2merged_tif or legacy merge_<date> directories")
    parser.add_argument("export_dir", help="Destination directory for flattened exports")
    parser.add_argument("--scene-prefix", help="Optional lake ID prefix for matching scene TIFFs to shapefiles")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing exports with different contents")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        products = export_merged_tifs(
            args.root_dir, args.export_dir, scene_prefix=args.scene_prefix, overwrite=args.overwrite
        )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Export failed: {exc}\n")
    print(f"Export complete: {len(products)} scene products in {args.export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
