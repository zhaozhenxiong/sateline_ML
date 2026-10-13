"""Copy merged Sentinel-2 products into a flat directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_dir", help="Directory containing per-date folders with 2merged_tif subdirectories")
    parser.add_argument("export_dir", help="Destination directory for flattened exports")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    for date_dir in sorted(root_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        merged_dir = date_dir / "2merged_tif"
        if not merged_dir.exists():
            continue
        for tif in merged_dir.glob("*.tif"):
            dst = export_dir / f"{date_dir.name}_{tif.name}"
            shutil.copy2(tif, dst)
            print(f"✅ {tif} -> {dst}")

    print(f"\n🎉 Export complete: {export_dir}")


if __name__ == "__main__":
    main()
