"""Validated, portable JSON configuration. Paths are relative to the config file."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_BAND_ORDER = ("B2", "B3", "B4", "B5", "B8", "B8A", "B9", "B11", "B12")
KNOWN_BANDS = {"B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"}
REQUIRED_BANDS = {"B2", "B3", "B4", "B8", "B11", "B12"}
DEFAULT_PARAMETERS = {
    "epochs": 5, "repeats": 1, "val_split": 0.25,
    "tile_size": 256, "stride": 128, "max_patches": 1024,
    "min_valid_frac": 0.5, "pos_frac": 0.9, "patch_seed": 42,
    "absorbance_thresh": 0.6, "water_ndwi_thresh": 0.1,
    "floating_thresh": 0.7, "suspended_thresh": 0.7,
    "auto_calibrate_thresholds": False, "stacking_preset": None,
    "stacking_sample_limit": 40000, "stacking_pixel_stride": 3,
    "stacking_max_pixels": 150000, "stacking_cv_folds": 5,
    "strict_mode": True, "unc_margin": 0.1,
    "min_area_float": 10, "min_area_susp": 24,
    "cap_float_pixels": 300000, "cap_susp_pixels": 120000,
}
# Output paths and managed resources are intentionally configured separately.
PARAMETER_NAMES = set(DEFAULT_PARAMETERS) | {
    "water_boost", "min_water_ratio", "max_water_ratio", "erode_iters",
    "absorb_min_ratio", "absorb_max_ratio", "absorb_thr_step", "absorb_diff_start",
    "absorb_diff_max", "absorb_close", "seed_target_float", "seed_target_susp",
    "seed_target_float_ratio", "seed_target_susp_ratio", "seed_target_float_floor",
    "seed_target_susp_floor", "relax_steps", "ndvi_thresh", "fdi_std_factor",
    "ndti_std_factor", "filter_field", "filter_values", "filter_query", "filter_indices",
    "plastic_prob_thresh", "auto_threshold_metric", "auto_threshold_candidates",
    "auto_threshold_max_samples", "auto_threshold_batch_size",
}


def _object(value: Any, name: str, allowed: set[str]) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} keys: {', '.join(sorted(unknown))}")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _path(value: Any, base: Path, name: str, *, optional: bool = False) -> Path | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def validate_bands(value: Any, name: str, *, detection: bool = False) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value or not all(isinstance(b, str) for b in value):
        raise ValueError(f"{name} must be a non-empty band list")
    bands = tuple(b.upper() for b in value)
    if len(set(bands)) != len(bands) or set(bands) - KNOWN_BANDS:
        raise ValueError(f"{name} contains duplicate or unsupported Sentinel-2 bands")
    if detection and not REQUIRED_BANDS.issubset(bands):
        raise ValueError(f"{name} requires bands {', '.join(sorted(REQUIRED_BANDS))}")
    return bands


def validate_parameters(params: dict) -> dict:
    _object(params, "detection.parameters", PARAMETER_NAMES)
    result = {**DEFAULT_PARAMETERS, **params}
    for key in ("epochs", "repeats", "tile_size", "stride", "max_patches",
                "stacking_pixel_stride", "stacking_cv_folds", "auto_threshold_batch_size"):
        if key in result:
            _positive_int(result[key], f"detection.parameters.{key}")
    if result["tile_size"] < 32 or result["tile_size"] % 16:
        raise ValueError("tile_size must be at least 32 and divisible by 16 for UNet")
    if result["stride"] > result["tile_size"]:
        raise ValueError("stride must not exceed tile_size")
    for key in ("min_valid_frac", "pos_frac", "floating_thresh", "suspended_thresh",
                "plastic_prob_thresh", "unc_margin", "min_water_ratio", "max_water_ratio",
                "absorb_min_ratio", "absorb_max_ratio"):
        if key in result:
            val = result[key]
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or not 0 <= val <= 1:
                raise ValueError(f"{key} must be between 0 and 1")
    val = result["val_split"]
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not 0 < val < 1:
        raise ValueError("val_split must be between 0 and 1 (exclusive)")
    for key in ("auto_calibrate_thresholds", "strict_mode", "absorb_close"):
        if key in result:
            _boolean(result[key], key)
    for key in ("stacking_sample_limit", "stacking_max_pixels", "cap_float_pixels", "cap_susp_pixels",
                "auto_threshold_max_samples"):
        if result.get(key) is not None:
            _positive_int(result[key], key)
    for key in ("patch_seed", "erode_iters", "relax_steps", "min_area_float", "min_area_susp",
                "seed_target_float_floor", "seed_target_susp_floor"):
        if result.get(key) is not None:
            val = result[key]
            if isinstance(val, bool) or not isinstance(val, int) or val < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("absorbance_thresh", "water_ndwi_thresh", "ndvi_thresh", "water_boost",
                "fdi_std_factor", "ndti_std_factor", "absorb_thr_step", "absorb_diff_start", "absorb_diff_max",
                "seed_target_float", "seed_target_susp", "seed_target_float_ratio", "seed_target_susp_ratio"):
        if result.get(key) is not None:
            val = result[key]
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{key} must be a finite number")
    for lo, hi in (("min_water_ratio", "max_water_ratio"), ("absorb_min_ratio", "absorb_max_ratio")):
        if lo in result and hi in result and result[lo] > result[hi]:
            raise ValueError(f"{lo} must not exceed {hi}")
    if result["stacking_cv_folds"] < 2:
        raise ValueError("stacking_cv_folds must be at least 2")
    for key in ("filter_field", "filter_query", "stacking_preset", "auto_threshold_metric"):
        if result.get(key) is not None and not isinstance(result[key], str):
            raise ValueError(f"{key} must be a string or null")
    if result.get("stacking_preset") is not None and not result["stacking_preset"].strip():
        raise ValueError("stacking_preset must be a non-empty string or null")
    if "auto_threshold_metric" in result and result["auto_threshold_metric"] not in (
        "f1", "precision", "recall", "youden", "balanced_accuracy"
    ):
        raise ValueError("Unsupported auto_threshold_metric")
    if result.get("auto_threshold_candidates") is not None:
        candidates = result["auto_threshold_candidates"]
        if not isinstance(candidates, list) or not candidates or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
            for v in candidates
        ):
            raise ValueError("auto_threshold_candidates must be a non-empty list of probabilities")
    if result.get("filter_indices") is not None:
        indices = result["filter_indices"]
        if not isinstance(indices, list) or not indices or any(type(v) is not int for v in indices):
            raise ValueError("filter_indices must be a non-empty list of row indices")
    return result


@dataclass
class PathsConfig:
    input_dir: Path
    shapefile_dir: Path | None
    output_dir: Path


@dataclass
class DownloadConfig:
    enabled: bool = False
    project: str | None = None
    start_date: str = "2022-01-01"
    end_date: str = "2022-03-01"
    roi_bounds: tuple[float, ...] = (119.8, 30.9, 120.6, 31.6)
    output_root: Path = Path("gee_sentinel3")
    step_days: int = 15
    bands: tuple[str, ...] = DEFAULT_BAND_ORDER
    scene_prefix: str | None = None
    max_cloud: float = 5.0

    def validate(self) -> None:
        from .workflows import validate_download_options
        validate_download_options(start_date=self.start_date, end_date=self.end_date, roi_bounds=self.roi_bounds,
                                  project=self.project, step_days=self.step_days, bands=self.bands, max_cloud=self.max_cloud)


@dataclass
class DetectionConfig:
    enabled: bool = True
    device: str = "auto"
    raster_crs_override: str | None = None
    band_order: tuple[str, ...] = DEFAULT_BAND_ORDER
    require_shapefile: bool = False
    plastic_library_csv: Path | None = None
    plastic_model_path: Path | None = None
    parameters: dict = field(default_factory=lambda: dict(DEFAULT_PARAMETERS))


@dataclass
class PipelineConfig:
    paths: PathsConfig
    download: DownloadConfig
    detection: DetectionConfig
    source: Path | None = None

    def to_dict(self) -> dict:
        def convert(value):
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [convert(v) for v in value]
            return value
        data = asdict(self)
        data.pop("source")
        return {"schema_version": 1, **convert(data)}


def default_config() -> dict:
    return {
        "schema_version": 1,
        "paths": {"input_dir": "mydata/tif", "shapefile_dir": "mydata/shp", "output_dir": "mydata/analysis_outputs"},
        "download": {"enabled": False, "project": None, "start_date": "2022-01-01", "end_date": "2022-03-01",
                     "roi_bounds": [119.8, 30.9, 120.6, 31.6], "output_root": "gee_sentinel3", "step_days": 15,
                     "bands": list(DEFAULT_BAND_ORDER), "scene_prefix": None, "max_cloud": 5.0},
        "detection": {"enabled": True, "device": "auto", "raster_crs_override": None, "band_order": list(DEFAULT_BAND_ORDER),
                      "require_shapefile": False, "plastic_library_csv": None, "plastic_model_path": None,
                      "parameters": dict(DEFAULT_PARAMETERS)},
    }


def load_config(path: Path | str) -> PipelineConfig:
    path = Path(path).expanduser().resolve()
    def reject_constant(value):
        raise ValueError(f"Invalid JSON numeric constant: {value}")
    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"JSON number is outside the finite range: {value}")
        return parsed
    data = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject_constant, parse_float=finite_float)
    _object(data, "config", {"schema_version", "paths", "download", "detection"})
    if type(data.get("schema_version", 1)) is not int or data.get("schema_version", 1) != 1:
        raise ValueError("Unsupported schema_version; expected 1")
    defaults = default_config()
    p = {**defaults["paths"], **_object(data.get("paths", {}), "paths", set(defaults["paths"]))}
    d = {**defaults["download"], **_object(data.get("download", {}), "download", set(defaults["download"]))}
    c = {**defaults["detection"], **_object(data.get("detection", {}), "detection", set(defaults["detection"]))}
    base = path.parent
    paths = PathsConfig(_path(p["input_dir"], base, "input_dir"), _path(p["shapefile_dir"], base, "shapefile_dir", optional=True),
                        _path(p["output_dir"], base, "output_dir"))
    if paths.output_dir == paths.input_dir or paths.output_dir in paths.input_dir.parents or paths.input_dir in paths.output_dir.parents:
        raise ValueError("input_dir and output_dir must be separate directory trees")
    if not isinstance(d["roi_bounds"], (list, tuple)):
        raise ValueError("download.roi_bounds must be a coordinate list")
    prefix = d["scene_prefix"]
    if prefix is not None and (not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", prefix)):
        raise ValueError("download.scene_prefix must contain only letters, digits, underscores, or hyphens")
    download = DownloadConfig(**{**d, "enabled": _boolean(d["enabled"], "download.enabled"),
                                  "output_root": _path(d["output_root"], base, "download.output_root"),
                                  "roi_bounds": tuple(d["roi_bounds"]), "bands": validate_bands(d["bands"], "download.bands")})
    _positive_int(download.step_days, "download.step_days")
    if download.enabled:
        download.validate()
    if c["device"] not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError("detection.device must be auto, cpu, cuda, or mps")
    override = c["raster_crs_override"]
    if override is not None and (not isinstance(override, str) or not override.strip()):
        raise ValueError("detection.raster_crs_override must be a CRS string or null")
    detection = DetectionConfig(**{**c, "enabled": _boolean(c["enabled"], "detection.enabled"),
                                   "require_shapefile": _boolean(c["require_shapefile"], "require_shapefile"),
                                   "band_order": validate_bands(c["band_order"], "detection.band_order", detection=True),
                                   "plastic_library_csv": _path(c["plastic_library_csv"], base, "plastic_library_csv", optional=True),
                                   "plastic_model_path": _path(c["plastic_model_path"], base, "plastic_model_path", optional=True),
                                   "parameters": validate_parameters(c["parameters"])})
    return PipelineConfig(paths, download, detection, path)


def write_default_config(path: Path | str) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing tuned configurations.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(default_config(), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return path
