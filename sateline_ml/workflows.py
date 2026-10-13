"""Portable scene workflows without geospatial dependencies at import time."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
import hashlib
import logging
import math
from pathlib import Path
import re
import shutil
from typing import Sequence

LOGGER = logging.getLogger(__name__)
DEFAULT_BANDS = ("B2", "B3", "B4", "B5", "B8", "B8A", "B9", "B11", "B12")
SENTINEL2_BANDS = frozenset(
    ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12")
)


def _parse_date(value: str, name: str) -> date:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(f"{name} must use YYYY-MM-DD format")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid YYYY-MM-DD date") from exc


def iter_windows(start: str, end: str, step_days: int = 15) -> list[tuple[str, str]]:
    """Return adjacent windows including start and excluding end."""
    if isinstance(step_days, bool) or not isinstance(step_days, int) or step_days <= 0:
        raise ValueError("step_days must be a positive integer")
    start_dt = _parse_date(start, "start_date")
    end_dt = _parse_date(end, "end_date")
    if start_dt >= end_dt:
        raise ValueError("end_date must be after start_date")
    windows = []
    current = start_dt
    while current < end_dt:
        # Clamp before adding, including close to date.max.
        next_dt = current + timedelta(days=min(step_days, (end_dt - current).days))
        windows.append((current.isoformat(), next_dt.isoformat()))
        current = next_dt
    return windows


def validate_download_options(
    *,
    start_date: str,
    end_date: str,
    roi_bounds: Sequence[float],
    project: str,
    step_days: int = 15,
    bands: Sequence[str] = DEFAULT_BANDS,
    max_cloud: float = 5.0,
) -> tuple[list[tuple[str, str]], tuple[float, ...], tuple[str, ...]]:
    """Validate local options before loading Earth Engine or making I/O."""
    windows = iter_windows(start_date, end_date, step_days)
    if not isinstance(project, str) or not project or any(c.isspace() for c in project):
        raise ValueError("project is required and must be a Google Earth Engine project ID")
    if isinstance(roi_bounds, (str, bytes)):
        raise ValueError("roi_bounds must contain MIN_LON, MIN_LAT, MAX_LON, MAX_LAT")
    try:
        raw_bounds = tuple(roi_bounds)
        if len(raw_bounds) != 4 or any(isinstance(value, bool) for value in raw_bounds):
            raise ValueError
        bounds = tuple(float(value) for value in raw_bounds)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("roi_bounds must contain four finite numbers") from exc
    if not all(math.isfinite(value) for value in bounds):
        raise ValueError("roi_bounds must contain four finite numbers")
    min_lon, min_lat, max_lon, max_lat = bounds
    if not (-180 <= min_lon < max_lon <= 180 and -90 <= min_lat < max_lat <= 90):
        raise ValueError("roi_bounds must be ordered and within longitude/latitude limits")
    if isinstance(bands, (str, bytes)):
        raise ValueError("bands must be a nonempty sequence of Sentinel-2 band names")
    try:
        band_order = tuple(bands)
        if (
            not band_order
            or any(band not in SENTINEL2_BANDS for band in band_order)
            or len(set(band_order)) != len(band_order)
        ):
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError("bands must contain unique Sentinel-2 SR band names") from exc
    try:
        cloud = float(max_cloud)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("max_cloud must be between 0 and 100") from exc
    if isinstance(max_cloud, bool) or not math.isfinite(cloud) or not 0 <= cloud <= 100:
        raise ValueError("max_cloud must be between 0 and 100")
    return windows, bounds, band_order


def download_timeseries(
    *,
    start_date: str,
    end_date: str,
    roi_bounds: Sequence[float],
    output_root: Path | str,
    project: str,
    step_days: int = 15,
    bands: Sequence[str] = DEFAULT_BANDS,
    max_cloud: float = 5.0,
    log_to_file: bool = True,
) -> list[Path]:
    """Download retained scene products, keeping timestamps and buckets separate.

    Authenticate with ``earthengine authenticate`` before use on a new machine.
    Initialization is noninteractive; source tiles are never moved by this helper.
    """
    windows, bounds, band_order = validate_download_options(
        start_date=start_date,
        end_date=end_date,
        roi_bounds=roi_bounds,
        project=project,
        step_days=step_days,
        bands=bands,
        max_cloud=max_cloud,
    )
    if isinstance(output_root, str) and not output_root.strip():
        raise ValueError("output_root must be a nonempty directory path")
    output_root = Path(output_root).expanduser()
    from .gee import initialize_ee, process_download

    initialize_ee(project)
    output_root.mkdir(parents=True, exist_ok=True)
    products: list[Path] = []
    for start, end in windows:
        LOGGER.info("Downloading Sentinel-2 window %s to %s", start, end)
        results = process_download(
            project_root=output_root / start,
            start_date=start,
            end_date=end,
            roi_bounds=bounds,
            bands_to_download=band_order,
            max_cloud=float(max_cloud),
            log_to_file=log_to_file,
        )
        products.extend(Path(results[bucket].path) for bucket in sorted(results))
    return products


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _product_paths(root_dir: Path, export_dir: Path) -> list[Path]:
    products = []
    nested_export = _is_within(export_dir, root_dir)
    for candidate in root_dir.rglob("*"):
        if not candidate.is_file() or candidate.suffix.lower() not in {".tif", ".tiff"}:
            continue
        if nested_export and _is_within(candidate.resolve(), export_dir):
            continue
        current_product = candidate.parent.name == "2merged_tif"
        legacy_product = (
            candidate.parent.name.startswith("merge_")
            and candidate.name.startswith("Sentinel2_merged_")
        )
        if current_product or legacy_product:
            products.append(candidate)
    return sorted(products, key=lambda path: path.relative_to(root_dir).as_posix())


def _export_name(source: Path, root_dir: Path, scene_prefix: str | None) -> str:
    relative = source.relative_to(root_dir)
    parts = [part for part in relative.parts if part != "2merged_tif"]
    name = "_".join(parts)
    if scene_prefix is not None:
        name = f"{scene_prefix}_{name}"
    return name


def _files_equal(source: Path, destination: Path) -> bool:
    if source.stat().st_size != destination.stat().st_size:
        return False
    # Compare actual contents on every run. A cached stat comparison can miss
    # changes on filesystems with coarse timestamps or restored modification dates.
    with source.open("rb") as left, destination.open("rb") as right:
        while True:
            block = left.read(1024 * 1024)
            if block != right.read(1024 * 1024):
                return False
            if not block:
                return True


def export_merged_tifs(
    root_dir: Path | str,
    export_dir: Path | str,
    *,
    scene_prefix: str | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Copy current and legacy final TIFFs to a deterministic flat destination.

    ``scene_prefix`` can be a lake ID for shapefile matching. Flat-name collisions
    receive a stable digest of their source-relative path. Identical exports are
    reused; conflicting contents raise unless ``overwrite=True``. Destinations
    inside the source tree are excluded from discovery on every invocation.
    """
    root_dir = Path(root_dir).expanduser().resolve()
    export_dir = Path(export_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Download root is not a directory: {root_dir}")
    if root_dir == export_dir:
        raise ValueError("export_dir must differ from root_dir")
    if scene_prefix is not None:
        if not isinstance(scene_prefix, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", scene_prefix):
            raise ValueError("scene_prefix must contain only letters, digits, underscores, or hyphens")
    products = _product_paths(root_dir, export_dir)
    groups: dict[str, list[Path]] = defaultdict(list)
    for source in products:
        groups[_export_name(source, root_dir, scene_prefix).casefold()].append(source)
    reserved_names = set(groups)
    export_dir.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    for source in products:
        name = _export_name(source, root_dir, scene_prefix)
        if len(groups[name.casefold()]) > 1:
            digest = hashlib.sha256(source.relative_to(root_dir).as_posix().encode()).hexdigest()
            filename = Path(name)
            length = 12
            name = f"{filename.stem}_{digest[:length]}{filename.suffix}"
            suffix_counter = 0
            while name.casefold() in reserved_names:
                length += 1
                name = f"{filename.stem}_{digest[:length]}{filename.suffix}"
                if length > len(digest):
                    # Guard even the extraordinarily unlikely full digest collision.
                    name = f"{filename.stem}_{digest}_{suffix_counter}{filename.suffix}"
                    suffix_counter += 1
        reserved_names.add(name.casefold())
        destination = export_dir / name
        if destination.exists():
            if source.samefile(destination) or _files_equal(source, destination):
                exported.append(destination)
                continue
            if not overwrite:
                raise FileExistsError(f"Export already exists with different contents: {destination}")
        shutil.copy2(source, destination)
        LOGGER.info("Exported %s to %s", source, destination)
        exported.append(destination)
    return exported
