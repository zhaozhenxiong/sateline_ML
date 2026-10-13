"""Sequential orchestration with reproducible manifests and safe resume."""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import DEFAULT_BAND_ORDER, KNOWN_BANDS, REQUIRED_BANDS, PipelineConfig

LOGGER = logging.getLogger(__name__)
ID_PATTERN = re.compile(r"(?P<id>\d+)(?=_)")


@dataclass(frozen=True)
class Scene:
    input_tif: Path
    shapefile: Path | None
    output_dir: Path

    def to_dict(self) -> dict:
        return {"input_tif": str(self.input_tif), "shapefile": str(self.shapefile) if self.shapefile else None,
                "output_dir": str(self.output_dir)}


def extract_id_from_tif(path: Path) -> str | None:
    match = ID_PATTERN.search(path.stem)
    return match.group("id") if match else None


def pick_shapefile_for_id(directory: Path | None, lake_id: str | None) -> Path | None:
    """Prefer exact ID, then ID folder, then a delimited ID; reject ambiguity."""
    if directory is None or lake_id is None or not directory.is_dir():
        return None
    shapefiles = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".shp")
    exact = [p for p in shapefiles if p.stem == lake_id]
    if exact:
        candidates = exact
    elif (directory / lake_id).is_dir():
        candidates = sorted(p for p in (directory / lake_id).iterdir() if p.is_file() and p.suffix.lower() == ".shp")
    else:
        token = re.compile(rf"(?<!\d){re.escape(lake_id)}(?!\d)")
        candidates = [p for p in shapefiles if token.search(p.stem)]
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous shapefiles for lake {lake_id}: {', '.join(str(p) for p in candidates)}")
    return candidates[0] if candidates else None


def discover_scenes(config: PipelineConfig, *, allow_missing: bool = False) -> list[Scene]:
    directory = config.paths.input_dir
    if not directory.is_dir():
        if allow_missing:
            return []
        raise ValueError(f"Input directory does not exist: {directory}")
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"})
    scenes, stems = [], set()
    for path in files:
        if path.stem in {".", "..", ""}:
            raise ValueError(f"Invalid scene filename: {path.name}")
        if path.stem.casefold() in stems:
            raise ValueError(f"Scene output name collision: {path.stem}")
        stems.add(path.stem.casefold())
        shp = pick_shapefile_for_id(config.paths.shapefile_dir, extract_id_from_tif(path))
        if config.detection.require_shapefile and shp is None:
            raise ValueError(f"No matching shapefile for {path.name}; configure shapefile_dir or disable require_shapefile")
        scenes.append(Scene(path, shp, config.paths.output_dir / path.stem))
    return scenes


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def _file_identity(path: Path | None) -> dict | None:
    if path is None:
        return None
    identity = {"path": str(path)}
    if path.is_file():
        stat = path.stat()
        identity.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    else:
        identity["missing"] = True
    return identity


def _fingerprint(scene: Scene, config: PipelineConfig) -> str:
    from . import __version__
    sidecars = []
    if scene.shapefile:
        sidecars = [_file_identity(p) for p in sorted(scene.shapefile.parent.iterdir())
                    if p.stem == scene.shapefile.stem and p.is_file()]
    value = {"pipeline_version": __version__, "input": _file_identity(scene.input_tif), "aoi": sidecars,
             "detection": config.to_dict()["detection"],
             "plastic_library": _file_identity(config.detection.plastic_library_csv),
             "plastic_model": _file_identity(config.detection.plastic_model_path)}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _completed(scene: Scene, fingerprint: str) -> bool:
    try:
        status = json.loads((scene.output_dir / "scene.json").read_text(encoding="utf-8"))
        outputs = status.get("outputs", {})
        return (status.get("status") == "completed" and status.get("fingerprint") == fingerprint
                and all(_file_identity(scene.output_dir / name) == outputs.get(name)
                        for name in ("detections.csv", "preview.png"))
                and all((scene.output_dir / name).stat().st_size > 0 for name in ("detections.csv", "preview.png")))
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def _resolve_bands(path: Path, configured: tuple[str, ...]) -> tuple[str, ...]:
    """Read metadata without modifying source imagery, including read-only datasets."""
    import rasterio
    with rasterio.open(path) as src:
        count, descriptions = src.count, src.descriptions
        if src.crs is None:
            raise ValueError(f"GeoTIFF has no CRS: {path}")
    if descriptions and any(descriptions):
        if not all(isinstance(b, str) and b.strip() for b in descriptions):
            raise ValueError(f"Partial band descriptions in {path}; provide complete band metadata")
        order = tuple(b.strip().upper() for b in descriptions)
    else:
        layouts = (configured,
                   ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"),
                   DEFAULT_BAND_ORDER, ("B2", "B3", "B4", "B8", "B8A", "B11", "B12"))
        order = next((layout for layout in layouts if len(layout) == count), ())
        LOGGER.warning("%s has no band descriptions; using configured/known layout: %s", path.name, order)
    if len(order) != count or len(set(order)) != count or set(order) - KNOWN_BANDS or not REQUIRED_BANDS.issubset(order):
        raise ValueError(f"Unsupported or incomplete band layout in {path}: {order} ({count} bands)")
    return order


def _prepare_runtime() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if sys.platform in ("darwin", "win32"):
        multiprocessing.set_start_method("spawn", force=True)
        import torch.multiprocessing
        torch.multiprocessing.set_start_method("spawn", force=True)


def detect_scene(scene: Scene, config: PipelineConfig) -> int:
    """Run the existing science API; return detection row count."""
    settings = config.detection
    from .inputs import prepare_raster
    input_tif = prepare_raster(scene.input_tif, scene.output_dir / f"{scene.input_tif.stem}.vrt",
                               crs_override=settings.raster_crs_override)
    band_order = _resolve_bands(input_tif, settings.band_order)
    if scene.shapefile is not None:
        from .inputs import validate_aoi
        filters = {key: value for key, value in settings.parameters.items()
                   if key in {"filter_field", "filter_values", "filter_query", "filter_indices"}}
        validate_aoi(input_tif, scene.shapefile, **filters)
    _prepare_runtime()
    from .detection import floating_debris_detection
    kwargs = dict(settings.parameters)
    library, model = settings.plastic_library_csv, settings.plastic_model_path
    if library is not None and model is not None and library.is_file() and model.is_file():
        kwargs.update(plastic_library_csv=library, plastic_model_path=model, plastic_retrain=False)
    elif library is not None or model is not None:
        LOGGER.warning("%s: plastic classification skipped; both library and checkpoint files are required", scene.input_tif.name)
    result = floating_debris_detection(
        input_tif=input_tif, shp_file=scene.shapefile,
        output_csv=scene.output_dir / "detections.csv", preview_png=scene.output_dir / "preview.png",
        debug_dir=scene.output_dir / "debug", log_file=scene.output_dir / "run.log",
        model_out_dir=config.paths.output_dir, band_order=band_order,
        device=None if settings.device == "auto" else settings.device, **kwargs,
    )
    return len(result)


def run_detection(config: PipelineConfig, *, resume: bool = False, fail_fast: bool = False) -> dict:
    scenes = discover_scenes(config)
    if not scenes:
        raise ValueError(f"No GeoTIFF inputs in {config.paths.input_dir}")
    summary = {"started_at": _now(), "config": config.to_dict(), "completed": 0, "skipped": 0, "failed": 0, "scenes": []}
    manifest_path = config.paths.output_dir / "run_manifest.json"
    _write_json(manifest_path, summary)
    for scene in scenes:
        fingerprint = _fingerprint(scene, config)
        record = {**scene.to_dict(), "fingerprint": fingerprint, "started_at": _now()}
        if resume and _completed(scene, fingerprint):
            record["status"] = "skipped"
            summary["skipped"] += 1
            LOGGER.info("Skip completed scene: %s", scene.input_tif.name)
        else:
            scene.output_dir.mkdir(parents=True, exist_ok=True)
            record["status"] = "running"
            _write_json(scene.output_dir / "scene.json", record)
            LOGGER.info("Process %s -> %s", scene.input_tif.name, scene.output_dir)
            try:
                # A rerun must never accept an earlier run's files as new outputs.
                for name in ("detections.csv", "preview.png"):
                    (scene.output_dir / name).unlink(missing_ok=True)
                rows = detect_scene(scene, config)
                for name in ("detections.csv", "preview.png"):
                    if not (scene.output_dir / name).is_file() or not (scene.output_dir / name).stat().st_size:
                        raise RuntimeError(f"Detector did not produce {name}")
                record.update(status="completed", rows=rows, outputs={name: _file_identity(scene.output_dir / name)
                              for name in ("detections.csv", "preview.png")})
                summary["completed"] += 1
            except Exception as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                summary["failed"] += 1
                with (scene.output_dir / "run.log").open("a", encoding="utf-8") as stream:
                    stream.write(f"\nPipeline failure: {record['error']}\n")
                LOGGER.exception("Scene failed: %s", scene.input_tif.name)
            record["finished_at"] = _now()
            _write_json(scene.output_dir / "scene.json", record)
        summary["scenes"].append(record)
        _write_json(manifest_path, summary)
        if record["status"] == "failed" and fail_fast:
            break
    summary["finished_at"] = _now()
    _write_json(manifest_path, summary)
    return summary


def plan_pipeline(config: PipelineConfig, stage: str = "all") -> dict:
    downloading = stage == "download" or (stage == "all" and config.download.enabled)
    detecting = stage == "detect" or (stage == "all" and config.detection.enabled)
    if downloading:
        config.download.validate()
    stages = (["download", "export"] if downloading else ["export"] if stage == "export" else []) + (["detect"] if detecting else [])
    if not stages:
        raise ValueError("No stages enabled in this configuration")
    scenes = discover_scenes(config, allow_missing=True) if detecting else []
    return {"stages": stages, "config": config.to_dict(), "existing_scenes": [s.to_dict() for s in scenes],
            "note": "Downloads may add new scenes; dry-run does not contact Earth Engine or load the detector."}


def run_pipeline(config: PipelineConfig, *, stage: str = "all", resume: bool = False,
                 fail_fast: bool = False, overwrite_exports: bool = False) -> dict:
    plan = plan_pipeline(config, stage)
    result = {"stages": plan["stages"], "downloaded": [], "exported": []}
    if "download" in plan["stages"]:
        from .workflows import download_timeseries
        settings = config.download
        result["downloaded"] = [str(p) for p in download_timeseries(
            start_date=settings.start_date, end_date=settings.end_date, roi_bounds=settings.roi_bounds,
            output_root=settings.output_root, project=settings.project, step_days=settings.step_days,
            bands=settings.bands, max_cloud=settings.max_cloud)]
    if "export" in plan["stages"]:
        from .workflows import export_merged_tifs
        result["exported"] = [str(p) for p in export_merged_tifs(config.download.output_root, config.paths.input_dir,
                                                               scene_prefix=config.download.scene_prefix,
                                                               overwrite=overwrite_exports)]
        if not result["exported"]:
            raise ValueError(f"No merged products found in {config.download.output_root}")
    if "detect" in plan["stages"]:
        result["detection"] = run_detection(config, resume=resume, fail_fast=fail_fast)
    return result
