"""Read-only input checks; geospatial dependencies are imported on demand."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def prepare_raster(
    input_tif: Path | str,
    output_vrt: Path | str,
    *,
    crs_override: str | None = None,
) -> Path:
    """Validate raster coordinates or explicitly assign a CRS in a small VRT.

    The override assigns the coordinate system of the existing pixel grid;
    it does not reproject, resample, or rewrite source imagery.
    """
    import math
    import os
    import tempfile
    import xml.etree.ElementTree as ET
    import rasterio
    from rasterio.crs import CRS
    from rasterio.shutil import copy as copy_raster
    from rasterio.warp import transform_bounds

    source = Path(input_tif).expanduser().resolve()
    override = None
    if crs_override is not None:
        if not isinstance(crs_override, str) or not crs_override.strip():
            raise ValueError("raster_crs_override must be a nonempty CRS string, for example EPSG:32651")
        try:
            override = CRS.from_user_input(crs_override)
        except Exception as exc:
            raise ValueError(f"Invalid raster_crs_override {crs_override!r}: {exc}") from exc
    with rasterio.open(source) as src:
        crs = override or src.crs
        if crs is None:
            raise ValueError(f"GeoTIFF has no CRS: {source}; set detection.raster_crs_override to the verified source CRS")
        if crs.is_geographic:
            bounds = tuple(src.bounds)
            try:
                geographic = bounds if crs == CRS.from_epsg(4326) else transform_bounds(crs, "EPSG:4326", *bounds)
                valid = (all(math.isfinite(value) for value in geographic)
                         and -360 <= geographic[0] <= 360 and -360 <= geographic[2] <= 360
                         and -90 <= geographic[1] <= 90 and -90 <= geographic[3] <= 90)
            except Exception:
                valid = False
            if not valid:
                raise ValueError(
                    f"Raster geographic CRS {crs} is inconsistent with bounds {bounds}: {source}. "
                    "Verify the actual coordinate system, then set detection.raster_crs_override; "
                    "the pipeline will create a VRT and leave the source unchanged."
                )
    if override is None:
        return source

    destination = Path(output_vrt).expanduser().resolve()
    if destination == source or (destination.exists() and destination.samefile(source)):
        raise ValueError("output_vrt must differ from the source raster")
    if destination.suffix.lower() != ".vrt":
        raise ValueError("output_vrt must use the .vrt extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sateline-vrt-", dir=destination.parent) as directory:
        temporary = Path(directory) / "raster.vrt"
        copy_raster(source, temporary, driver="VRT")
        # Only the VRT is opened for metadata updates. GDAL retains its original
        # band sources, grid, descriptions, masks, and nodata values.
        with rasterio.open(temporary, "r+") as vrt:
            vrt.crs = override
        tree = ET.parse(temporary)
        # The VRT moves out of its temporary directory. Absolute source paths
        # prevent that move from changing any source-band references.
        for filename in tree.iter("SourceFilename"):
            if filename.get("relativeToVRT") == "1" and filename.text:
                filename.text = str((temporary.parent / filename.text).resolve())
                filename.set("relativeToVRT", "0")
        tree.write(temporary, encoding="utf-8", xml_declaration=True)
        os.replace(temporary, destination)
    return destination


def validate_aoi(
    input_tif: Path | str,
    shapefile: Path | str,
    *,
    filter_field: str | None = None,
    filter_values: Sequence | None = None,
    filter_query: str | None = None,
    filter_indices: Sequence[int] | None = None,
) -> None:
    """Require filtered AOI geometries to cover at least one source raster pixel.

    A failed alignment must not broaden the requested AOI to the entire scene.
    Bounded windows keep this check independent of full-scene array allocation.
    Neither the raster nor any shapefile sidecar is modified.
    """
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import rasterio
    from rasterio.errors import WindowError
    from rasterio.features import geometry_window, rasterize
    from rasterio.windows import Window

    try:
        frame = gpd.read_file(shapefile)
    except Exception as exc:
        raise ValueError(f"Could not read AOI {shapefile}: {exc}") from exc
    if frame.crs is None:
        raise ValueError(f"AOI has no CRS: {shapefile}")
    try:
        if filter_query:
            frame = frame.query(filter_query)
        if filter_field is not None and filter_values is not None:
            if filter_field not in frame.columns:
                raise ValueError(f"AOI filter field {filter_field!r} is missing")
            values = filter_values if isinstance(filter_values, (list, tuple, set, np.ndarray, pd.Series)) else [filter_values]
            frame = frame[frame[filter_field].isin(list(values))]
        if filter_indices is not None:
            indices = [int(filter_indices)] if isinstance(filter_indices, (int, np.integer)) else list(filter_indices)
            frame = frame.iloc[indices]
        frame = frame[frame.geometry.notnull() & ~frame.geometry.is_empty]
    except Exception as exc:
        raise ValueError(f"AOI filtering failed for {shapefile}: {exc}") from exc
    if frame.empty:
        raise ValueError(f"AOI filtering produced no geometries: {shapefile}")

    with rasterio.open(input_tif) as src:
        if src.crs is None:
            raise ValueError(f"GeoTIFF has no CRS: {input_tif}")
        try:
            aligned = frame.to_crs(src.crs)
        except Exception as exc:
            raise ValueError(f"AOI projection failed for {shapefile}: {exc}") from exc
        geometries = [geometry for geometry in aligned.geometry if geometry is not None and not geometry.is_empty]
        if not geometries:
            raise ValueError(f"AOI projection produced no geometries: {shapefile}")
        try:
            window = geometry_window(src, geometries)
        except WindowError as exc:
            raise ValueError(f"AOI does not intersect GeoTIFF {input_tif}: {shapefile}") from exc
        except Exception as exc:
            raise ValueError(f"Could not align AOI with GeoTIFF {input_tif}: {exc}") from exc
        shapes = [(geometry, 1) for geometry in geometries]
        row_start, col_start = int(window.row_off), int(window.col_off)
        row_stop = row_start + int(window.height)
        col_stop = col_start + int(window.width)
        for row in range(row_start, row_stop, 1024):
            for col in range(col_start, col_stop, 1024):
                tile = Window(col, row, min(1024, col_stop - col), min(1024, row_stop - row))
                mask = rasterize(shapes, out_shape=(int(tile.height), int(tile.width)),
                                 transform=src.window_transform(tile), fill=0, dtype="uint8")
                if np.any(mask):
                    return
    raise ValueError(f"AOI covers no pixel centers in GeoTIFF {input_tif}: {shapefile}")
