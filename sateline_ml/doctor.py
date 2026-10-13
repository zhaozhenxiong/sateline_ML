"""Read-only diagnostics for a destination host, with isolated import probes.

No authentication, Earth Engine initialization, downloads, or directory creation
happen here. Native package import failures (including process aborts) are
captured in subprocesses so they cannot terminate the CLI itself.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from importlib import metadata, util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

_DOWNLOAD_PACKAGES = {
    "earthengine-api": "ee", "numpy": "numpy", "pandas": "pandas",
    "rasterio": "rasterio", "requests": "requests",
}
_DETECTION_PACKAGES = {
    "numpy": "numpy", "pandas": "pandas", "rasterio": "rasterio",
    "geopandas": "geopandas", "shapely": "shapely", "torch": "torch",
    "opencv-python-headless": "cv2", "matplotlib": "matplotlib", "tqdm": "tqdm",
}


def _value(obj: Any, key: str, default=None):
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


def _probe_import(module: str, *, timeout: float = 30) -> dict:
    """Import a single module in a fresh process, returning JSON-safe evidence."""
    source = (
        "import importlib,json,sys\n"
        "module=importlib.import_module(sys.argv[1])\n"
        "data={}\n"
        "if sys.argv[1]=='torch':\n"
        " data['cuda']=bool(module.cuda.is_available())\n"
        " data['mps']=bool(getattr(module.backends,'mps',None) and module.backends.mps.is_available())\n"
        "print(json.dumps(data))\n"
    )
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    try:
        result = subprocess.run(
            [sys.executable, "-c", source, module], capture_output=True,
            text=True, timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired:
        return {"status": "error", "detail": f"Import timed out after {timeout:g} seconds"}
    except OSError as exc:
        return {"status": "error", "detail": f"Could not launch import check: {exc}"}
    if result.returncode:
        # Bound the diagnostic output; never print credential/config contents.
        detail = (result.stderr or result.stdout).strip()[-3000:]
        if "OMP: Error #15" in detail:
            detail = (
                "Native OpenMP runtime conflict: libomp is loaded more than once. "
                "Use a clean virtual environment and reinstall the selected dependencies."
            )
        return {"status": "error", "detail": f"Import process exited {result.returncode}: {detail}"}
    try:
        data = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        data = {}
    return {"status": "ok", "detail": "Import succeeded in an isolated process", "data": data}


def _package_info(name: str, module: str) -> dict:
    info = {"name": name, "module": module, "version": None, "status": "ok", "detail": "Installed"}
    # Full OpenCV also exports cv2. Report the actual existing distribution,
    # while fresh installations prefer the headless wheel for batch operation.
    candidates = [name]
    if name == "opencv-python-headless":
        candidates += ["opencv-python", "opencv-contrib-python-headless", "opencv-contrib-python"]
    for distribution in candidates:
        try:
            info["version"] = metadata.version(distribution)
            info["distribution"] = distribution
            break
        except metadata.PackageNotFoundError:
            pass
    try:
        present = util.find_spec(module) is not None
    except (ImportError, ValueError):
        present = False
    if not present:
        info.update(status="error", detail=f"Missing import {module!r}")
    elif info["version"] is None:
        info.update(status="warning", detail="Module found, but distribution version is unavailable")
    return info


def _writable_destination(path: Path) -> bool:
    parent = path
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.is_dir() and os.access(parent, os.W_OK | os.X_OK)


def check_environment(config=None, *, stage: str = "detect", check_imports: bool = True) -> dict:
    """Return structured diagnostics for packages, device, paths, and assets.

    ``config`` may be PipelineConfig or an equivalent mapping. ``all`` selects
    only enabled config stages; an explicit stage checks it even when disabled.
    Warnings describe optional features/fallbacks and do not make ``ok`` false.
    Package checks use subprocesses; ``check_imports=False`` reads metadata and
    module presence only, suitable for fast diagnostics in a fresh environment.
    """
    if stage not in {"detect", "download", "all"}:
        raise ValueError("stage must be detect, download, or all")
    detection = _value(config, "detection")
    download = _value(config, "download")
    detect = stage == "detect" or (stage == "all" and _value(detection, "enabled", True))
    downloading = stage == "download" or (stage == "all" and _value(download, "enabled", True))
    packages = {}
    if downloading:
        packages.update(_DOWNLOAD_PACKAGES)
    if detect:
        packages.update(_DETECTION_PACKAGES)
        if _value(_value(detection, "parameters", {}), "stacking_preset"):
            packages["scikit-learn"] = "sklearn"
    report = {
        "ok": True, "stage": stage, "selected_stages": [],
        "python": {"version": platform.python_version(), "executable": sys.executable,
                   "platform": f"{platform.system()} {platform.release()} ({platform.machine()})"},
        "packages": [_package_info(name, module) for name, module in sorted(packages.items())],
        "checks": [], "warnings": [], "errors": [],
    }

    def check(name: str, status: str, detail: str, **evidence):
        report["checks"].append({"name": name, "status": status, "detail": detail, **evidence})

    check("python", "ok" if sys.version_info >= (3, 10) else "error", "Python 3.10 or later is required")
    probes = {}
    if check_imports:
        targets = [item["module"] for item in report["packages"] if item["status"] != "error"]
        # Independent processes avoid mixing different OpenMP runtimes. A module
        # boundary check below separately tests the pipeline's combined imports.
        with ThreadPoolExecutor(max_workers=3) as executor:
            probes = dict(zip(targets, executor.map(_probe_import, targets)))
        for item in report["packages"]:
            probe = probes.get(item["module"])
            if probe:
                item.update({k: probe[k] for k in ("status", "detail")})
        if all(item["status"] != "error" for item in report["packages"]):
            for enabled, module in ((downloading, "gee"), (detect, "detection")):
                if enabled:
                    result = _probe_import(f"sateline_ml.{module}")
                    check(f"import:{module}", result["status"], result["detail"])
    else:
        check("imports", "skipped", "Import checks disabled; presence does not confirm native library compatibility")

    paths = _value(config, "paths")
    if detect:
        report["selected_stages"].append("detect")
        requested = _value(detection, "device", "auto")
        data = probes.get("torch", {}).get("data", {})
        if check_imports and "cuda" in data:
            selected = "cuda" if data["cuda"] else "mps" if data["mps"] else "cpu"
            if requested != "auto":
                selected = requested if requested == "cpu" or data.get(requested, False) else "cpu"
            status = "warning" if requested in {"cuda", "mps"} and selected != requested else "ok"
            check("device", status, f"Requested {requested}; available selection {selected}", requested=requested,
                  selected=selected, available={"cpu": True, **data})
        else:
            check("device", "skipped", "Device availability needs a successful isolated torch import", requested=requested)
        if paths is not None:
            input_dir = Path(_value(paths, "input_dir"))
            check("input_dir", "ok" if input_dir.is_dir() else "warning",
                  f"Input directory {'exists' if input_dir.is_dir() else 'does not exist yet'}: {input_dir}")
            shp_value = _value(paths, "shapefile_dir")
            required = _value(detection, "require_shapefile", False)
            shp_ready = shp_value is not None and Path(shp_value).is_dir()
            check("shapefile_dir", "ok" if shp_ready else "error" if required else "warning",
                  f"AOI directory: {shp_value}; {'required' if required else 'optional'}")
            output_dir = Path(_value(paths, "output_dir"))
            check("output_dir", "ok" if _writable_destination(output_dir) else "error",
                  f"Output destination is {'writable' if _writable_destination(output_dir) else 'not writable'}: {output_dir}")
        library = _value(detection, "plastic_library_csv")
        model = _value(detection, "plastic_model_path")
        if library is None and model is None:
            check("plastic_assets", "skipped", "Plastic classification is disabled; no asset paths configured")
        else:
            available = {"library": library is not None and Path(library).is_file(),
                         "model": model is not None and Path(model).is_file()}
            ready = all(available.values())
            check("plastic_assets", "ok" if ready else "warning",
                  "Plastic classification assets are present" if ready else
                  "Plastic classification will be skipped: both spectral library CSV and model checkpoint must exist",
                  available=available, library=str(library) if library is not None else None,
                  model=str(model) if model is not None else None)
        if _value(_value(detection, "parameters", {}), "stacking_preset"):
            check("stacking", "ok" if probes.get("sklearn", {}).get("status") == "ok" else
                  "skipped" if not check_imports else "error",
                  "Stacking requested; extra models remain optional and may fall back to available estimators")
    if downloading:
        report["selected_stages"].append("download")
        project = _value(download, "project")
        check("gee_project", "ok" if isinstance(project, str) and project.strip() else "error" if config is not None else "warning",
              f"Earth Engine project: {project}" if project else "Set download.project before downloading")
        candidates = [Path.home() / ".config/earthengine/credentials",
                      Path.home() / ".config/gcloud/application_default_credentials.json"]
        configured_credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if configured_credentials:
            candidates.append(Path(configured_credentials).expanduser())
        credentials_found = any(path.is_file() for path in candidates)
        check("gee_credentials", "ok" if credentials_found else "warning",
              "A local credential file exists; validity and project access are not verified" if credentials_found else
              "No standard local credential file found; run earthengine authenticate or configure service credentials before downloading")
        destination = _value(download, "output_root")
        if destination is not None:
            check("download_output", "ok" if _writable_destination(Path(destination)) else "error",
                  f"Download output destination: {destination}")
    if not report["selected_stages"]:
        check("stages", "warning", "Both pipeline stages are disabled")
    for item in [*report["packages"], *report["checks"]]:
        if item["status"] in {"error", "warning"}:
            report["errors" if item["status"] == "error" else "warnings"].append(
                f"{item['name']}: {item['detail']}"
            )
    report["ok"] = not report["errors"]
    return report


doctor = check_environment
