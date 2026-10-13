"""Batch downloader for Sentinel-2 scenes using Google Earth Engine."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

# Keep the original direct-script invocation working on an uninstalled checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sateline_ml.workflows import DEFAULT_BANDS, download_timeseries, iter_windows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start_date", help="Start date inclusive, e.g. 2022-01-01")
    parser.add_argument("end_date", help="End date exclusive, e.g. 2022-12-31")
    parser.add_argument("roi", nargs=4, type=float, metavar="BOUND", help="MIN_LON MIN_LAT MAX_LON MAX_LAT")
    parser.add_argument("output_root", help="Directory where results will be stored")
    parser.add_argument("--step-days", type=int, default=15, help="Window size in days (default: 15)")
    parser.add_argument("--project", required=True, help="Google Earth Engine project ID")
    parser.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS), help="Bands to download in order")
    parser.add_argument("--max-cloud", type=float, default=5.0, help="Maximum scene cloud percentage")
    parser.add_argument(
        "--size-limit", type=int, default=None,
        help="Deprecated compatibility option; scene products are retained without moving source tiles",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.size_limit is not None:
        logging.warning("--size-limit is deprecated and ignored; retained scenes stay separate")
    try:
        products = download_timeseries(
            start_date=args.start_date,
            end_date=args.end_date,
            roi_bounds=args.roi,
            output_root=args.output_root,
            project=args.project,
            step_days=args.step_days,
            bands=args.bands,
            max_cloud=args.max_cloud,
        )
    except (ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Download failed: {exc}\n")
    print(f"Download complete: {len(products)} scene products under {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
