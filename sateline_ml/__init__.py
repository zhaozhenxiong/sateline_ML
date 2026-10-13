"""Sentinel-2 download and debris detection, with lazy optional dependencies.

Importing :mod:`sateline_ml` only loads the Python standard library. Install
``sateline-ml[download]`` or ``sateline-ml[detection]`` for the corresponding API.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sateline-ml")
except PackageNotFoundError:
    __version__ = "0.1.0"

_EXPORTS = {
    "DownloadResult": ("gee", "download"),
    "initialize_ee": ("gee", "download"),
    "process_download": ("gee", "download"),
    "find_max_size": ("gee", "download"),
    "merge_all_bands": ("gee", "download"),
    "floating_debris_detection": ("detection", "detection"),
    "create_mask_tif_from_shp": ("detection", "detection"),
    "STACKING_PRESETS": ("stacking", "stacking"),
    "run_stacking_pipeline": ("stacking", "stacking"),
}
__all__ = ["__version__", *_EXPORTS]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, extra = _EXPORTS[name]
    try:
        value = getattr(import_module(f".{module}", __name__), name)
    except ImportError as exc:
        raise ImportError(
            f"{name} requires the {extra} dependencies. "
            f"Install with: python -m pip install 'sateline-ml[{extra}]'. "
            f"Original import error: {exc}"
        ) from exc
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
