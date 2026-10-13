"""Build distributable packages through clean local staging.

macOS can synthesize AppleDouble sidecars on removable volumes. Staging only
source files in a temporary directory avoids those files interfering with
setuptools' archive creation and cleanup. Data and model artifacts are never
copied into the staging tree.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT_FILES = (
    "pyproject.toml", "README.md", "LICENSE", "MANIFEST.in", "AGENTS.md",
    "pipeline.example.json", "run_pipeline.py",
)
SOURCE_DIRECTORIES = ("sateline_ml", "scripts", "tests")
EXCLUDED_DIRECTORIES = {"__pycache__", "build", "dist", "attic", ".venv", "venv", "node_modules"}


def stage_source(root: Path, destination: Path) -> None:
    """Copy the explicit code/document manifest without scanning data trees."""
    if not (root / "pyproject.toml").is_file():
        raise FileNotFoundError(f"Project metadata not found: {root / 'pyproject.toml'}")
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ROOT_FILES:
        source = root / filename
        if source.is_file():
            shutil.copy2(source, destination / filename)
    for dirname in SOURCE_DIRECTORIES:
        source_root = root / dirname
        if not source_root.is_dir():
            continue
        for folder, dirs, files in os.walk(source_root):
            dirs[:] = [name for name in dirs if not name.startswith(".") and name not in EXCLUDED_DIRECTORIES]
            for filename in files:
                if filename.startswith(".") or not filename.endswith(".py"):
                    continue
                source = Path(folder) / filename
                target = destination / source.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Source project root")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: ROOT/dist)")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    output = (args.output_dir or root / "dist").expanduser().resolve()
    try:
        with tempfile.TemporaryDirectory(prefix="sateline-build-") as directory:
            temporary = Path(directory)
            staged = temporary / "source"
            artifacts = temporary / "artifacts"
            stage_source(root, staged)
            cmd = [sys.executable, "-m", "build", str(staged), "--outdir", str(artifacts)]
            print("[build]", shlex.join(cmd), flush=True)
            result = subprocess.run(cmd)
            if result.returncode:
                print("[ERROR] Build failed; install the build extra with 'python -m pip install build' and inspect the log.",
                      file=sys.stderr)
                return result.returncode if result.returncode > 0 else 1
            produced = sorted(path for path in artifacts.iterdir()
                              if path.name.endswith((".whl", ".tar.gz")) and not path.name.startswith("."))
            if not any(path.name.endswith(".whl") for path in produced) or not any(
                    path.name.endswith(".tar.gz") for path in produced):
                raise RuntimeError("Build did not produce both a wheel and a source distribution")
            output.mkdir(parents=True, exist_ok=True)
            for artifact in produced:
                target = output / artifact.name
                shutil.copy2(artifact, target)
                print(f"[DONE] {target}")
    except (OSError, RuntimeError) as exc:
        print(f"[ERROR] Could not build distributions: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
