"""Batch downloader for Sentinel-2 scenes using Google Earth Engine."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Allow running this script directly from the repo root as `python scripts/download_timeseries.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sateline_ml.gee import (
    DownloadResult,
    initialize_ee,
    merge_all_bands,
    process_download,
    find_max_size,
)


def iter_windows(start: str, end: str, step_days: int) -> list[tuple[str, str]]:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    windows = []
    current = start_dt
    while current < end_dt:
        next_dt = min(current + timedelta(days=step_days), end_dt)
        windows.append((current.strftime("%Y-%m-%d"), next_dt.strftime("%Y-%m-%d")))
        current = next_dt
    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start_date", help="Start date inclusive, e.g. 2022-01-01")
    parser.add_argument("end_date", help="End date exclusive, e.g. 2022-12-31")
    parser.add_argument("roi", nargs=4, type=float, metavar="MIN_LON MIN_LAT MAX_LON MAX_LAT", help="Bounding box")
    parser.add_argument("output_root", help="Directory where results will be stored")
    parser.add_argument("--step-days", type=int, default=15, help="Window size in days (default: 15)")
    parser.add_argument("--project", default=None, help="Google Earth Engine project id")
    parser.add_argument(
        "--bands",
        nargs="+",
        default=["B2", "B3", "B4", "B8", "B8A", "B11", "B12"],
        help="Bands to download",
    )
    parser.add_argument("--size-limit", type=int, default=15 * 1024 * 1024, help="Size threshold for tile selection in bytes")
    args = parser.parse_args()

    initialize_ee(args.project)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for start, end in iter_windows(args.start_date, args.end_date, args.step_days):
        window_root = output_root / start
        print(f"\n🚀 Processing window {start} ~ {end}")
        bucket_results = process_download(
            project_root=window_root,
            start_date=start,
            end_date=end,
            roi_bounds=args.roi,
            bands_to_download=args.bands,
        )
        if not bucket_results:
            print("  ⚠️ No imagery kept for this window")
            continue

        source_dir = window_root / "1source_tif"
        move_dir = output_root / f"merge_{start}"
        find_max_size(source_dir, args.size_limit, move_dir)

        merged = merge_all_bands(
            root_dir=move_dir,
            bands=args.bands,
            merged_dir=move_dir / "merged_bands",
            final_path=move_dir / f"Sentinel2_merged_{start}_{end}.tif",
        )
        if merged:
            print(f"🎉 Window finished, merged product at {merged}")
        else:
            print("⚠️ No merged product created for this window")


if __name__ == "__main__":
    main()
