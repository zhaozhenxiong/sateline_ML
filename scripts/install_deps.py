"""
Auto-detect and install Python dependencies used in this repository.

Usage examples:
  - python sateline_ML/scripts/install_deps.py                 # auto-detect and install missing
  - python sateline_ML/scripts/install_deps.py --root sateline_ML --dry-run
  - python sateline_ML/scripts/install_deps.py --skip-gee --torch cpu

Notes:
  - On Windows, installing geopandas/rasterio may take time and require wheels.
    This script attempts a straightforward pip install; if it fails, consult:
    https://www.lfd.uci.edu/~gohlke/pythonlibs/ for prebuilt wheels.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


# Map import names to pip package names when they differ
MODULE_TO_PKG: Dict[str, str] = {
    "cv2": "opencv-python",
    "ee": "earthengine-api",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "PIL": "Pillow",
}

# Some packages are optional by feature area
DEFAULT_GROUPS: Dict[str, List[str]] = {
    "base": [
        "numpy",
        "pandas",
        "tqdm",
        "matplotlib",
        "shapely",
        "requests",
        "opencv-python",
    ],
    "geo": [
        "rasterio",
        "geopandas",
    ],
    "ml": [
        "torch",  # CPU-only by default on Windows when installed via PyPI
    ],
    "gee": [
        "earthengine-api",
    ],
    "heavy_ml": [
        "lightgbm",
        "xgboost",
        "catboost",
        "pydensecrf",
    ],
}

# Packages that are only installed when --with-heavy-ml is passed
HEAVY_ML_PKGS: Set[str] = set(DEFAULT_GROUPS["heavy_ml"])


IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))")

# Standard-library top-level module names to ignore during auto-detection
_STDLIB_MODULES: Set[str] = set(getattr(sys, "stdlib_module_names", []))


def parse_imports(py_file: Path) -> Iterable[str]:
    try:
        for line in py_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = IMPORT_RE.match(line)
            if not m:
                continue
            mod = m.group(1) or m.group(2)
            if not mod:
                continue
            top = mod.split(".")[0]
            if top and top != "sateline_ml" and top not in _STDLIB_MODULES:
                yield top
    except Exception:
        return []


def discover_modules(root: Path) -> Set[str]:
    modules: Set[str] = set()
    for py in root.rglob("*.py"):
        modules.update(parse_imports(py))
    return modules


def resolve_packages(modules: Iterable[str]) -> List[str]:
    pkgs: Set[str] = set()
    for m in modules:
        pkg = MODULE_TO_PKG.get(m, m)
        pkgs.add(pkg)
    return sorted(pkgs)


def missing_packages(packages: Iterable[str]) -> List[str]:
    missing: List[str] = []
    for pkg in packages:
        mod = pkg
        # Inverse mapping for known renames
        for k, v in MODULE_TO_PKG.items():
            if v == pkg:
                mod = k
                break
        try:
            __import__(mod)
        except Exception:
            missing.append(pkg)
    return missing


def pip_install(args: List[str]) -> int:
    cmd = [sys.executable, "-m", "pip", "install"] + args
    print("[pip]", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-install project dependencies")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="Root to scan for imports")
    parser.add_argument("--skip-geo", action="store_true", help="Skip installing heavy geospatial deps (geopandas/rasterio)")
    parser.add_argument("--skip-gee", action="store_true", help="Skip Google Earth Engine client")
    parser.add_argument("--skip-ml", action="store_true", help="Skip ML deps (torch)")
    parser.add_argument("--only-base", action="store_true", help="Install only common/base deps")
    parser.add_argument("--torch", choices=["auto", "cpu", "skip"], default="auto", help="Torch install preference")
    parser.add_argument("--with-heavy-ml", action="store_true", help="Install heavy optional ML packages (lightgbm, xgboost, catboost, pydensecrf)")
    parser.add_argument("--upgrade-pip", action="store_true", help="Upgrade pip before installing")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be installed")
    parser.add_argument("--separate", action="store_true", help="Install packages one-by-one instead of grouped")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[WARN] root not found: {root}")
        return

    # Detect modules from source
    detected_mods = discover_modules(root)
    detected_pkgs = resolve_packages(detected_mods)

    # Start from base group, then add optional groups unless skipped
    wanted: List[str] = []
    wanted += DEFAULT_GROUPS["base"]
    if not args.only_base:
        if not args.skip_geo:
            wanted += DEFAULT_GROUPS["geo"]
        if not args.skip_ml and args.torch != "skip":
            wanted += DEFAULT_GROUPS["ml"]
        if not args.skip_gee:
            wanted += DEFAULT_GROUPS["gee"]
        if args.with_heavy_ml:
            wanted += DEFAULT_GROUPS["heavy_ml"]

    # Also include anything detected from code that isn't already in wanted
    for pkg in detected_pkgs:
        if pkg in wanted:
            continue
        if pkg in HEAVY_ML_PKGS and not args.with_heavy_ml:
            continue
        wanted.append(pkg)

    # Torch handling
    if "torch" in wanted:
        if args.torch == "skip":
            wanted.remove("torch")
        elif args.torch == "cpu":
            # Rely on PyPI CPU wheels on Windows/macOS; leave as 'torch'
            pass
        else:  # auto
            pass

    # Compute what is missing right now
    to_install = missing_packages(wanted)

    print("[INFO] Wanted packages:")
    for p in sorted(set(wanted)):
        print("  -", p)
    print("[INFO] Missing:")
    for p in to_install:
        print("  -", p)
    if not to_install:
        print("[INFO] All dependencies already satisfied.")
        return

    if args.dry_run:
        print("[DRY-RUN] Skipping installation.")
        return

    if args.upgrade_pip:
        subprocess.call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])  # best effort

    # Install in groups or separately
    if args.separate:
        for pkg in to_install:
            rc = pip_install([pkg])
            if rc != 0:
                print(f"[ERROR] Failed to install {pkg} (rc={rc}). You may need a prebuilt wheel on Windows.")
    else:
        rc = pip_install(to_install)
        if rc != 0:
            print(f"[WARN] Group install failed (rc={rc}). Retrying one-by-one...")
            for pkg in to_install:
                rc1 = pip_install([pkg])
                if rc1 != 0:
                    print(f"[ERROR] Failed to install {pkg} (rc={rc1}).")

    print("[DONE] Installation step completed.")


if __name__ == "__main__":
    main()

