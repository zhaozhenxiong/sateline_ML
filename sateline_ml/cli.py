"""Unified CLI; help, config generation and dry-runs need only Python."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sateline", description="Portable Sentinel-2 floating debris pipeline")
    parser.add_argument("--version", action="store_true", help="Print installed version")
    commands = parser.add_subparsers(dest="command")
    init = commands.add_parser("init", help="Create a portable JSON configuration")
    init.add_argument("path", nargs="?", default="pipeline.json")
    doctor = commands.add_parser("doctor", help="Check dependencies, device and configured resources")
    doctor.add_argument("--config", type=Path)
    doctor.add_argument("--stage", choices=("all", "detect", "download"), default="all")
    doctor.add_argument("--no-import-check", action="store_true", help="Only check installed metadata (fast)")
    run = commands.add_parser("run", help="Run download/export/detection stages")
    run.add_argument("--config", type=Path, default=Path("pipeline.json"))
    run.add_argument("--stage", choices=("all", "download", "export", "detect"), default="all")
    run.add_argument("--dry-run", action="store_true", help="Print plan without writing outputs or contacting GEE")
    run.add_argument("--resume", action="store_true", help="Skip completed scenes with unchanged inputs and parameters")
    run.add_argument("--fail-fast", action="store_true", help="Stop detection after the first failed scene")
    run.add_argument("--overwrite-exports", action="store_true", help="Replace existing exported TIFFs with changed contents")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        from . import __version__
        print(__version__)
        return 0
    if args.command is None:
        parser.print_help()
        return 0
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        from .config import load_config, write_default_config
        if args.command == "init":
            path = write_default_config(args.path)
            config = load_config(path)
            for directory in (config.paths.input_dir, config.paths.shapefile_dir):
                if directory:
                    directory.mkdir(parents=True, exist_ok=True)
            print(f"Created {path}\nEdit the configuration, then run: sateline run --config \"{path}\" --dry-run")
            return 0
        if args.command == "doctor":
            from .doctor import check_environment
            config = load_config(args.config) if args.config else None
            report = check_environment(config, stage=args.stage, check_imports=not args.no_import_check)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["ok"] else 1
        from .pipeline import plan_pipeline, run_pipeline
        config = load_config(args.config)
        if args.dry_run:
            result = plan_pipeline(config, args.stage)
            result["overwrite_exports"] = args.overwrite_exports
        else:
            result = run_pipeline(config, stage=args.stage, resume=args.resume, fail_fast=args.fail_fast,
                                  overwrite_exports=args.overwrite_exports)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result.get("detection", {}).get("failed", 0) else 0
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted. Use --resume to skip completed scenes on the next run.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
