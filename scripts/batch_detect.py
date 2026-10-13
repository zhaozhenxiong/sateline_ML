"""Run the quick screening pipeline over all GeoTIFFs in a directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow running this script directly from the repo root as `python scripts/batch_detect.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sateline_ml.detection import batch_detect_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tif_dir", help="Directory containing GeoTIFF files")
    parser.add_argument("shapefile", help="AOI shapefile path")
    parser.add_argument("output_dir", help="Directory to store CSV/PNG outputs")
    parser.add_argument("--filter-field")
    parser.add_argument("--filter-values", nargs="+")
    parser.add_argument("--filter-query")
    parser.add_argument("--filter-indices", nargs="+", type=int)
    parser.add_argument("--advanced", action="store_true", help="Use PyTorch UNet detector")
    parser.add_argument("--mask-tif", help="Mask raster used in advanced mode")
    parser.add_argument("--floating-thresh", type=float, default=0.6)
    parser.add_argument("--suspended-thresh", type=float, default=0.6)
    parser.add_argument("--ndti-std-factor", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--log-dir", help="Directory to write per-scene log files (advanced mode)")
    parser.add_argument("--debug-dir", help="Directory to write per-scene debug visualisations (advanced mode)")
    parser.add_argument("--plastic-model", help="Path to trained plastic classifier (.pt)")
    parser.add_argument("--plastic-library", help="Plastic spectral library CSV")
    parser.add_argument("--plastic-thresh", type=float, default=0.5)
    args = parser.parse_args()

    kwargs = {
        k: v
        for k, v in {
            "filter_field": args.filter_field,
            "filter_values": args.filter_values,
            "filter_query": args.filter_query,
            "filter_indices": args.filter_indices,
        }.items()
        if v is not None
    }

    if args.advanced:
        detection_kwargs = {
            "mask_tif": args.mask_tif,
            "floating_thresh": args.floating_thresh,
            "suspended_thresh": args.suspended_thresh,
            "ndti_std_factor": args.ndti_std_factor,
            "epochs": args.epochs,
            "log_dir": args.log_dir,
            "debug_dir": args.debug_dir,
            "plastic_model_path": args.plastic_model,
            "plastic_library_csv": args.plastic_library,
            "plastic_prob_thresh": args.plastic_thresh,
        }
        results = batch_detect_directory(
            args.tif_dir,
            args.shapefile,
            args.output_dir,
            mode="advanced",
            detection_kwargs=detection_kwargs,
            **kwargs,
        )
    else:
        results = batch_detect_directory(
            args.tif_dir, args.shapefile, args.output_dir, **kwargs
        )
    print(f"\n🎉 Processed {len(results)} files into {args.output_dir}")


if __name__ == "__main__":
    main()
