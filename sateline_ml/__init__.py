"""sateline_ml package

Utilities for downloading Sentinel-2 imagery from Google Earth Engine and
detecting floating debris from multispectral imagery.
"""

from importlib.metadata import PackageNotFoundError, version

from .gee import (
    DownloadResult,
    initialize_ee,
    process_download,
    find_max_size,
    merge_all_bands,
)
from .detection import (
    floating_debris_detection,
    create_mask_tif_from_shp,
)
try:  # Optional stacking exports
    from .stacking import STACKING_PRESETS, run_stacking_pipeline
except Exception:  # pragma: no cover
    STACKING_PRESETS = {}
    run_stacking_pipeline = None  # type: ignore

try:
    __version__ = version("sateline_ml")
except PackageNotFoundError:  # pragma: no cover - package metadata missing
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "DownloadResult",
    "initialize_ee",
    "process_download",
    "find_max_size",
    "merge_all_bands",
    "floating_debris_detection",
    "create_mask_tif_from_shp",
]

if run_stacking_pipeline is not None:
    __all__.extend(["STACKING_PRESETS", "run_stacking_pipeline"])
