"""Install declared feature dependencies without importing native libraries.

The dependency source is pyproject.toml, never a scan of notebooks, attic, or
virtual environments. Run inside a virtual environment on the destination host.
"""

from __future__ import annotations

import argparse
import ast
from importlib import metadata, util
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable

MODULES = {
    "earthengine-api": "ee", "opencv-python-headless": "cv2",
    "scikit-learn": "sklearn", "scikit-image": "skimage",
    "pydensecrf": "pydensecrf",
}
CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def declared_groups(root: Path) -> dict[str, list[str]]:
    path = root / "pyproject.toml"
    source = path.read_text(encoding="utf-8")
    try:
        import tomllib
    except ImportError:  # Python 3.10: parse only this project's string arrays.
        section = re.search(
            r"(?ms)^\[project\.optional-dependencies\]\s*\n(.*?)(?=^\[|\Z)", source
        )
        if section is None:
            raise ValueError("pyproject.toml has no declared dependency groups")
        groups = {
            name: ast.literal_eval(value)
            for name, value in re.findall(
                r"(?ms)^([\w-]+)\s*=\s*(\[.*?\])", section.group(1)
            )
        }
    else:
        groups = tomllib.loads(source)["project"]["optional-dependencies"]
    if not groups or any(
        not isinstance(reqs, list) or not all(isinstance(req, str) for req in reqs)
        for reqs in groups.values()
    ):
        raise ValueError("Dependency groups must contain arrays of requirement strings")
    return groups


def package_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9_.-]*", requirement)
    if match is None:
        raise ValueError(f"Invalid declared requirement: {requirement}")
    return match.group().lower().replace("_", "-")


def selected_requirements(args, groups: dict[str, list[str]]) -> list[str]:
    selected = args.group or ["all"]
    if args.with_heavy_ml:
        selected = [*selected, "stacking"]
    unknown = set(selected) - groups.keys()
    if unknown:
        raise ValueError(f"Unknown group(s): {', '.join(sorted(unknown))}")
    excluded = set()
    if args.skip_geo or args.only_base:
        excluded.update(("rasterio", "geopandas"))
    if args.skip_gee or args.only_base:
        excluded.add("earthengine-api")
    if args.skip_ml or args.only_base or args.torch == "skip":
        excluded.add("torch")
    if args.only_base:
        excluded.update(package_name(req) for req in groups.get("stacking", [])
                        if package_name(req) not in {"numpy", "pandas", "opencv-python-headless"})
    return list(dict.fromkeys(
        req for group in selected for req in groups[group]
        if package_name(req) not in excluded
    ))


def missing_packages(requirements: Iterable[str]) -> list[str]:
    """Report missing metadata/modules without loading native extension code.

    pip still receives all selected constraints, so it handles old versions and
    transitive dependencies even when the top-level package is already present.
    """
    missing = []
    for req in requirements:
        name = package_name(req)
        module = MODULES.get(name, name.replace("-", "_"))
        try:
            metadata.version(name)
            found = util.find_spec(module) is not None
        except (metadata.PackageNotFoundError, ImportError, ValueError):
            found = False
        if not found:
            missing.append(req)
    return missing


def install_commands(args, requirements: list[str]) -> list[list[str]]:
    prefix = [sys.executable, "-m", "pip", "install"]
    commands = []
    if args.upgrade_pip:
        commands.append([*prefix, "--upgrade", "pip"])
    cpu_torch = args.torch == "cpu" and sys.platform.startswith(("linux", "win"))
    torch_reqs = [req for req in requirements if package_name(req) == "torch"]
    remaining = [req for req in requirements if not (cpu_torch and package_name(req) == "torch")]
    if cpu_torch and torch_reqs:
        # Reinstall explicitly: an already present CUDA wheel does not satisfy
        # an explicit request for a CPU build on the destination machine.
        commands.append([*prefix, "--upgrade", "--force-reinstall", *torch_reqs,
                         "--index-url", CPU_INDEX])
    if args.separate:
        commands.extend([*prefix, req] for req in remaining)
    elif remaining:
        commands.append([*prefix, *remaining])
    return commands


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Project containing pyproject.toml (no source scanning)")
    parser.add_argument("--group", action="append", help="Declared extra to install; repeat for multiple (default: all)")
    parser.add_argument("--skip-geo", action="store_true", help="Exclude geopandas and rasterio")
    parser.add_argument("--skip-gee", action="store_true", help="Exclude Earth Engine")
    parser.add_argument("--skip-ml", action="store_true", help="Exclude torch")
    parser.add_argument("--only-base", action="store_true", help="Exclude GEE, rasterio, geopandas, torch, and stacking models")
    parser.add_argument("--torch", choices=("auto", "cpu", "skip"), default="auto")
    parser.add_argument("--with-heavy-ml", action="store_true", help="Include the declared stacking extra; CRF is separate")
    parser.add_argument("--upgrade-pip", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--separate", action="store_true")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 10):
        print("[ERROR] sateline-ml requires Python 3.10 or later.", file=sys.stderr)
        return 2
    try:
        wanted = selected_requirements(args, declared_groups(args.root))
    except (OSError, ValueError, KeyError, SyntaxError) as exc:
        print(f"[ERROR] Cannot read declared dependencies: {exc}", file=sys.stderr)
        return 2
    print("[INFO] Selected requirements:")
    for req in wanted:
        print(f"  - {req}")
    missing = missing_packages(wanted)
    print("[INFO] Missing packages:", ", ".join(missing) or "none")
    if args.torch == "cpu" and sys.platform == "darwin" and any(package_name(r) == "torch" for r in wanted):
        print("[INFO] macOS uses its PyPI torch wheel; select device=cpu in the pipeline config for CPU execution.")
    commands = install_commands(args, wanted)
    for cmd in commands:
        print("[DRY-RUN]" if args.dry_run else "[pip]", shlex.join(cmd), flush=True)
        if not args.dry_run:
            try:
                rc = subprocess.call(cmd)
            except OSError as exc:
                print(f"[ERROR] pip could not start: {exc}", file=sys.stderr)
                return 1
            if rc:
                print(f"[ERROR] Dependency installation failed (exit {rc}).", file=sys.stderr)
                return rc if rc > 0 else 1
    print("[DRY-RUN] No changes made." if args.dry_run else "[DONE] Selected dependency constraints satisfied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
