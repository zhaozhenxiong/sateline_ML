"""Compatibility entry point for the installed, configurable pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from sateline_ml.cli import main as cli_main
from sateline_ml.config import write_default_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "pipeline.json")
    parser.add_argument("--yes", "-y", action="store_true", help="Compatibility option; runs are always non-interactive")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--overwrite-exports", action="store_true")
    parser.add_argument("--stage", choices=("all", "detect", "download", "export"), default="all")
    args = parser.parse_args()
    if not args.config.exists() and args.config == parser.get_default("config"):
        write_default_config(args.config)
    argv = ["run", "--config", str(args.config), "--stage", args.stage]
    argv.extend(flag for flag in ("--dry-run", "--resume", "--fail-fast", "--overwrite-exports")
                if getattr(args, flag[2:].replace("-", "_")))
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
