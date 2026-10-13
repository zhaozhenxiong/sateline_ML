"""Floating debris detection utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import logging
import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes, rasterize
from rasterio.warp import transform as rio_transform
from shapely.geometry import Point
from tqdm import tqdm
from typing import Sequence, Union
# ---- Matplotlib: force non-interactive backend BEFORE importing pyplot ----
import matplotlib
matplotlib.use("Agg")  # avoid GUI/blocking, faster in headless/multiprocessing
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from .spectral_tools import S2_BANDS, band_depth
from .water_filter import (robust_thresholds_water,hysteresis_water_mask,erode_interior,region_filter,neighborhood_consistency)
from .train_model import (
    TemporalDataset, train_temporal_from_sequences,
    infer_temporal_latest, infer_temporal_sequence,  # 如需全序列输出
)

try:  # Optional stacking utilities
    from .stacking import (
        STACKING_PRESETS,
        build_tabular_feature_stack,
        postprocess_probabilities,
        run_stacking_pipeline,
    )
except Exception:  # pragma: no cover - optional dependency
    STACKING_PRESETS = {}
    build_tabular_feature_stack = None  # type: ignore
    postprocess_probabilities = None  # type: ignore
    run_stacking_pipeline = None  # type: ignore

try:  # Optional ML components
    from .train_model import (
        FloatingDataset,
        build_plastic_feature_matrix,
        build_plastic_feature_matrix_sparse,
        infer_unet,
        infer_unet_tiled,
        load_plastic_model,
        predict_plastic,
        sample_training_patches,
        train_unet_from_patches,
        train_unet_repeated,
        resolve_device,
        calibrate_probability_thresholds,
        ThresholdCalibrationResult,
        save_calibration_result,
        # plastic MLP pieces
        train_plastic_model,
        save_plastic_model,
        PlasticModelMetadata,
    )
except Exception:  # pragma: no cover - optional dependency
    FloatingDataset = None
    build_plastic_feature_matrix = None
    build_plastic_feature_matrix_sparse = None
    infer_unet = None
    infer_unet_tiled = None
    load_plastic_model = None
    predict_plastic = None
    sample_training_patches = None
    train_unet_from_patches = None
    train_unet_repeated = None
    resolve_device = None
    calibrate_probability_thresholds = None
    ThresholdCalibrationResult = None
    save_calibration_result = None
    train_plastic_model = None
    save_plastic_model = None
    PlasticModelMetadata = None
try:  # torch is optional; required for advanced workflows
    import torch  # type: ignore
except ImportError as exc:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

LOGGER = logging.getLogger(__name__)
LOGGER.propagate = False


@dataclass
class DetectionStats:
    label: str
    area_m2: float
    pixel_count: int


def _require_torch() -> None:
    if _TORCH_IMPORT_ERROR is not None:  # pragma: no cover - torch optional
        raise ImportError(
            "Torch is required for UNet-related operations. Install torch before calling this function."
        ) from _TORCH_IMPORT_ERROR


def filter_gdf(
    gdf: gpd.GeoDataFrame,
    filter_field: Optional[str] = None,
    filter_values: Optional[Sequence] = None,
    filter_query: Optional[str] = None,
    filter_indices: Optional[Sequence[int]] = None,
) -> gpd.GeoDataFrame:
    """Filter a GeoDataFrame by attribute, query string, or row indices."""

    if filter_query:
        gdf = gdf.query(filter_query)
    if filter_field is not None and filter_values is not None:
        if filter_field not in gdf.columns:
            raise KeyError(f"Field {filter_field} not found in shapefile columns: {list(gdf.columns)}")
        if not isinstance(filter_values, (list, tuple, set, np.ndarray, pd.Series)):
            filter_values = [filter_values]
        gdf = gdf[gdf[filter_field].isin(filter_values)]
    if filter_indices is not None:
        if isinstance(filter_indices, (int, np.integer)):
            filter_indices = [int(filter_indices)]
        gdf = gdf.iloc[list(filter_indices)]
    gdf = gdf[gdf.geometry.notnull()]
    if gdf.empty:
        raise ValueError("Shapefile filtering resulted in empty GeoDataFrame.")
    return gdf


def resize_to_reference(arr: np.ndarray, ref_shape: Sequence[int]) -> np.ndarray:
    """Resize an array to match a reference height/width, preserving boolean arrays."""

    is_bool = arr.dtype == bool
    arr_uint8 = arr.astype(np.uint8) if is_bool else arr
    resized = cv2.resize(arr_uint8, (ref_shape[1], ref_shape[0]), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool) if is_bool else resized


def _ensure_latlon(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame expressed in EPSG:4326."""

    if gdf.crs is None:
        raise ValueError("Shapefile CRS is undefined; cannot align with raster.")
    try:
        epsg = gdf.crs.to_epsg()
    except Exception:
        epsg = None
    if epsg == 4326:
        return gdf
    return gdf.to_crs("EPSG:4326")


def _guess_utm_epsg(gdf: gpd.GeoDataFrame) -> Optional[int]:
    """Heuristically guess a UTM EPSG code from the geometry centroid."""

    gdf_latlon = _ensure_latlon(gdf)
    centroid = gdf_latlon.geometry.unary_union.centroid
    if centroid.is_empty:
        return None
    if np.isnan(centroid.x) or np.isnan(centroid.y):
        return None
    zone = int((centroid.x + 180.0) // 6) + 1
    zone = max(1, min(zone, 60))
    epsg = 32600 + zone if centroid.y >= 0 else 32700 + zone
    return epsg


def _fallback_project_shapefile(
    gdf_source: Optional[gpd.GeoDataFrame],
    raster_crs,
    transform,
) -> tuple[Optional[gpd.GeoDataFrame], Optional[str]]:
    """Handle rasters incorrectly labelled as geographic by projecting AOI to a likely UTM CRS."""

    if gdf_source is None or gdf_source.empty or transform is None:
        return None, None
    if raster_crs is None:
        return None, None

    pixel_size = max(abs(transform.a), abs(transform.e))
    is_geographic = getattr(raster_crs, "is_geographic", False)
    if not is_geographic or pixel_size <= 1.0:
        return None, None

    epsg = _guess_utm_epsg(gdf_source)
    if epsg is None:
        return None, None

    try:
        gdf_fallback = gdf_source.to_crs(f"EPSG:{epsg}")
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.error("Failed to project shapefile to fallback EPSG:%s: %s", epsg, exc)
        return None, None

    message = (
        "Shapefile mask was empty after rasterize because the raster reports a geographic CRS (%s) "
        "with large pixel size %.2f. Falling back to EPSG:%s for masking. "
        "Please fix the raster CRS metadata to avoid this heuristic."
    ) % (raster_crs, pixel_size, epsg)
    return gdf_fallback, message


def read_bands(file_path: Path | str, band_indices: Sequence[int]):
    with rasterio.open(file_path) as src:
        bands = []
        for band in tqdm(band_indices, desc="Reading bands"):
            bands.append(src.read(band))
        return np.stack(bands, axis=-1), src.transform, src.meta


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (green - nir) / (green + nir + 1e-6)


def compute_fdi(red: np.ndarray, nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    lam_red, lam_nir, lam_swir = 665, 842, 1610
    baseline = red + (swir - red) * (lam_nir - lam_red) / (lam_swir - lam_red)
    return nir - baseline


def compute_ndti(red: np.ndarray, green: np.ndarray) -> np.ndarray:
    return (red - green) / (red + green + 1e-6)


def compute_sabi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Simple Aquatic Biomass Index variant used as texture cue."""
    return (nir - red) / (nir + red + 1e-6)


def _configure_logger(log_file: Optional[Path | str]) -> Optional[logging.Handler]:
    if not log_file:
        return None
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    handler.setLevel(logging.INFO)
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(handler)
    LOGGER.debug("Log file initialised at %s", log_path)
    return handler


def _ensure_stream_logger() -> None:
    if any(isinstance(h, logging.StreamHandler) for h in LOGGER.handlers):
        return
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    stream_handler.setLevel(logging.INFO)
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(stream_handler)


def _save_debug_image(
    debug_dir: Optional[Path | str],
    filename: str,
    array: Optional[np.ndarray],
    *,
    cmap: str = "viridis",
    title: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
) -> None:
    if not debug_dir or array is None:
        return
    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    data = np.array(array, copy=True)
    if mask is not None:
        mask_arr = np.asarray(mask)
        while mask_arr.ndim > data.ndim:
            mask_arr = np.any(np.isfinite(mask_arr), axis=-1)
        try:
            mask_arr = np.broadcast_to(mask_arr, data.shape, subok=True)
        except ValueError:
            mask_arr = np.ones_like(data, dtype=bool)
        if mask_arr.dtype != bool:
            if np.issubdtype(mask_arr.dtype, np.floating):
                mask_arr = np.isfinite(mask_arr) & (mask_arr != 0)
            else:
                mask_arr = mask_arr.astype(bool)
        data = np.where(mask_arr, data, np.nan)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, shrink=0.8)
    if title:
        ax.set_title(title)
    ax.axis("off")
    out_file = debug_path / filename
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def _save_debug_overlay(
    debug_dir: Optional[Path | str],
    filename: str,
    rgb: Optional[np.ndarray],
    overlay: Optional[np.ndarray],
    *,
    title: str,
    color: str,
    alpha: float = 0.4,
    mask: Optional[np.ndarray] = None,
) -> None:
    if not debug_dir or overlay is None:
        return
    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    if rgb is not None:
        ax.imshow(rgb, interpolation="nearest")
    elif mask is not None:
        ax.imshow(mask, interpolation="nearest")
    overlay_mask = overlay.astype(bool)
    # if mask is not None:
    #     overlay_mask &= mask
    ax.imshow(np.where(overlay_mask, 1.0, np.nan), cmap=mcolors.ListedColormap([color]), alpha=alpha)
    ax.set_title(title)
    ax.axis("off")
    out_file = debug_path / filename
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def _save_debug_hist(
    debug_dir: Optional[Path | str],
    filename: str,
    data: Optional[np.ndarray],
    *,
    title: str,
    mask: Optional[np.ndarray] = None,
    bins: int = 50,
) -> None:
    if not debug_dir or data is None:
        return
    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)
    data_arr = np.asarray(data)
    valid = data_arr.astype(np.float32, copy=False).ravel()
    if mask is not None:
        mask_arr = np.asarray(mask)
        while mask_arr.ndim > data_arr.ndim:
            mask_arr = np.any(np.isfinite(mask_arr), axis=-1)
        try:
            mask_arr = np.broadcast_to(mask_arr, data_arr.shape, subok=True)
        except ValueError:
            mask_arr = np.ones_like(data_arr, dtype=bool)
        if mask_arr.dtype != bool:
            if np.issubdtype(mask_arr.dtype, np.floating):
                mask_arr = np.isfinite(mask_arr) & (mask_arr != 0)
            else:
                mask_arr = mask_arr.astype(bool)
        mask_flat = mask_arr.ravel()
        valid = valid[mask_flat[: valid.size]]
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(valid, bins=bins, color="steelblue", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Probability")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(Path(debug_dir) / filename, dpi=200)
    plt.close(fig)


def _sample_rows(total: int, limit: int = 50000, seed: int = 0) -> np.ndarray:
    if total <= limit:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(total, size=limit, replace=False)
    idx.sort()
    return idx.astype(np.int64)


def _save_plastic_feature_correlations(
    debug_dir: Optional[Path | str],
    feat_names: Sequence[str],
    features: np.ndarray,
    labels: np.ndarray,
    *,
    sample_limit: int = 50000,
) -> None:
    if not debug_dir or features.size == 0:
        return
    debug_path = Path(debug_dir)
    try:
        sel = _sample_rows(features.shape[0], limit=sample_limit, seed=0)
        X = features[sel]
        y = labels[sel]
        corrs = []
        for i in range(X.shape[1]):
            xi = X[:, i]
            if np.allclose(xi.std(), 0.0):
                corrs.append(0.0)
                continue
            corr = np.corrcoef(xi, y)[0, 1]
            if not np.isfinite(corr):
                corr = 0.0
            corrs.append(float(corr))
        corrs = np.asarray(corrs, dtype=np.float32)
        abs_corr = np.abs(corrs)
        top_k = int(min(20, len(feat_names)))
        top_idx = np.argsort(abs_corr)[-top_k:]
        top_idx = top_idx[np.argsort(abs_corr[top_idx])]
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = [feat_names[i] for i in top_idx]
        colors = ["#d95f02" if corrs[i] < 0 else "#1b9e77" for i in top_idx]
        ax.barh(bars, corrs[top_idx], color=colors)
        ax.set_xlabel("Pearson correlation with plastic label")
        ax.set_title("Plastic feature correlations (top)")
        fig.tight_layout()
        debug_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(debug_path / "plastic_feature_correlation.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        LOGGER.warning("Failed to write plastic feature correlation plot: %s", exc)


def _mask_count(mask: np.ndarray) -> int:
    """Return the number of non-zero/True pixels as int."""
    return int(np.count_nonzero(mask))


def _robust_profile_stats(
    samples: np.ndarray,
    *,
    lower_pct: float = 5.0,
    upper_pct: float = 95.0,
) -> tuple[np.ndarray, np.ndarray]:
    if samples.ndim != 2 or samples.size == 0:
        return np.zeros(samples.shape[1] if samples.ndim == 2 else 0, dtype=np.float32), np.zeros(
            samples.shape[1] if samples.ndim == 2 else 0, dtype=np.float32
        )
    lower_pct = float(lower_pct)
    upper_pct = float(upper_pct)
    if not (0.0 <= lower_pct < upper_pct <= 100.0):
        raise ValueError("Invalid percentile bounds for spectral profile statistics")
    percentiles = np.nanpercentile(samples, [lower_pct, upper_pct], axis=0)
    lower = percentiles[0].reshape(1, -1)
    upper = percentiles[1].reshape(1, -1)
    trimmed = samples.astype(np.float32, copy=True)
    trimmed[(trimmed < lower) | (trimmed > upper)] = np.nan
    mean = np.nanmean(trimmed, axis=0)
    std = np.nanstd(trimmed, axis=0)
    counts = np.sum(np.isfinite(trimmed), axis=0).astype(np.float32)
    fallback = counts < np.maximum(10.0, 0.1 * samples.shape[0])
    if np.any(fallback):
        safe_cols = np.isfinite(samples).sum(axis=0) > 0
        fallback &= safe_cols
        if np.any(fallback):
            mean[fallback] = np.nanmean(samples[:, fallback], axis=0)
            std[fallback] = np.nanstd(samples[:, fallback], axis=0)
            counts[fallback] = np.sum(np.isfinite(samples[:, fallback]), axis=0).astype(np.float32)
    counts[counts <= 0] = 1.0
    stderr = std / np.sqrt(counts)
    ci95 = 1.96 * stderr
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    ci95 = np.nan_to_num(ci95, nan=0.0, posinf=0.0, neginf=0.0)
    return mean.astype(np.float32, copy=False), ci95.astype(np.float32, copy=False)


def _save_plastic_spectral_profiles(
    debug_dir: Optional[Path | str],
    feat_names: Sequence[str],
    features: np.ndarray,
    labels: np.ndarray,
    *,
    sample_limit: int = 50000,
) -> None:
    if not debug_dir or features.size == 0:
        return
    rho_indices = [(idx, name[4:]) for idx, name in enumerate(feat_names) if name.startswith("rho_")]
    if not rho_indices:
        return
    debug_path = Path(debug_dir)
    try:
        sel = _sample_rows(features.shape[0], limit=sample_limit, seed=1)
        X = features[sel]
        y = labels[sel]
        pos_mask = y > 0.5
        neg_mask = ~pos_mask
        if not np.any(pos_mask) or not np.any(neg_mask):
            return
        rho_cols = np.array([idx for idx, _ in rho_indices], dtype=np.int64)
        band_names = [name for _, name in rho_indices]
        wavelengths = []
        fallback_step = 1.0
        current = 0.0
        for band in band_names:
            info = S2_BANDS.get(band)
            if info is not None:
                wavelengths.append(info["center"])
            else:
                current += fallback_step
                wavelengths.append(current)
        wavelengths = np.asarray(wavelengths, dtype=np.float32)
        pos_spec = X[pos_mask][:, rho_cols]
        neg_spec = X[neg_mask][:, rho_cols]
        pos_mean, pos_ci = _robust_profile_stats(pos_spec, lower_pct=5.0, upper_pct=95.0)
        neg_mean, neg_ci = _robust_profile_stats(neg_spec, lower_pct=5.0, upper_pct=95.0)

        fig, (ax_main, ax_delta) = plt.subplots(
            2,
            1,
            figsize=(9, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        )
        ax_main.plot(wavelengths, pos_mean, label="Plastic samples", color="#d62728")
        ax_main.fill_between(
            wavelengths,
            pos_mean - pos_ci,
            pos_mean + pos_ci,
            color="#d62728",
            alpha=0.25,
        )
        ax_main.plot(wavelengths, neg_mean, label="Background samples", color="#1f77b4")
        ax_main.fill_between(
            wavelengths,
            neg_mean - neg_ci,
            neg_mean + neg_ci,
            color="#1f77b4",
            alpha=0.25,
        )
        ax_main.set_ylabel("Reflectance")
        ax_main.set_title("Average spectral profiles (candidate samples)")
        ax_main.legend()
        ax_main.grid(alpha=0.2)

        delta = pos_mean - neg_mean
        ax_delta.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--", alpha=0.6)
        ax_delta.plot(wavelengths, delta, color="#2ca02c", linewidth=1.6)
        ax_delta.fill_between(wavelengths, 0.0, delta, where=delta >= 0.0, color="#2ca02c", alpha=0.15)
        ax_delta.fill_between(wavelengths, 0.0, delta, where=delta < 0.0, color="#d62728", alpha=0.08)
        ax_delta.set_xlabel("Wavelength (nm or index)")
        ax_delta.set_ylabel("Δ reflectance")
        ax_delta.grid(alpha=0.2)

        fig.tight_layout()
        debug_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(debug_path / "plastic_spectral_profiles.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        LOGGER.warning("Failed to write plastic spectral profile plot: %s", exc)


def mask_to_vector(mask: np.ndarray, transform) -> gpd.GeoDataFrame:
    results = []
    for geom, value in shapes(mask.astype(np.uint8), transform=transform):
        if value == 1:
            results.append(Point(geom["coordinates"][0][0]))
    return gpd.GeoDataFrame(geometry=results)


def _pick_best_device(preferred: Optional[str] = None) -> str:
    """Return 'cuda' | 'mps' | 'cpu' with CUDA > MPS > CPU priority, honoring 'preferred' hint."""
    pref = (preferred or "").lower() if isinstance(preferred, str) else None
    try:
        if (pref in (None, "cuda") or pref not in ("mps", "cpu")) and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    try:
        mps_ok = bool(getattr(torch.backends, "mps").is_available())
        if (pref in (None, "mps") or pref == "mps") and mps_ok:
            return "mps"
    except Exception:
        pass
    return "cpu"

# -------------------- Vectorized helpers for building DataFrame --------------------

def _coords_from_indices(transform, rows, cols):
    # Affine: x = a*col + b*row + c ; y = d*col + e*row + f
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    xs = a * cols + b * rows + c
    ys = d * cols + e * rows + f
    return xs, ys


def _df_from_mask(label, mask_pred, prob_map,
                  ndvi, fdi, ndti, sabi, A_B11, A_B12,
                  transform, crs, water_mask_shape):
    rows, cols = np.where(mask_pred)
    if rows.size == 0:
        return pd.DataFrame(columns=[
            "row","col","x","y","lon","lat","class","probability",
            "NDVI","FDI","NDTI","SABI","A_B11","A_B12","PlasticProb","PlasticFlag","PlasticType"
        ])
    xs, ys = _coords_from_indices(transform, rows, cols)
    if crs:
        try:
            lons, lats = rio_transform(crs, "EPSG:4326", xs.tolist(), ys.tolist())
            lons = np.asarray(lons, dtype=np.float64)
            lats = np.asarray(lats, dtype=np.float64)
        except Exception:
            lons = np.full_like(xs, np.nan, dtype=np.float64)
            lats = np.full_like(ys, np.nan, dtype=np.float64)
    else:
        lons = np.full_like(xs, np.nan, dtype=np.float64)
        lats = np.full_like(ys, np.nan, dtype=np.float64)
    data = {
        "row": rows.astype(np.int32),
        "col": cols.astype(np.int32),
        "x": xs.astype(np.float64),
        "y": ys.astype(np.float64),
        "lon": lons.astype(np.float64),
        "lat": lats.astype(np.float64),
        "class": np.full(rows.shape, label),
        "probability": prob_map[rows, cols].astype(np.float32),
        "NDVI": ndvi[rows, cols].astype(np.float32),
        "FDI": fdi[rows, cols].astype(np.float32),
        "NDTI": ndti[rows, cols].astype(np.float32),
        "SABI": sabi[rows, cols].astype(np.float32),
        "A_B11": A_B11[rows, cols].astype(np.float32),
        "A_B12": A_B12[rows, cols].astype(np.float32),
        "PlasticProb": np.full(rows.shape, np.nan, dtype=np.float32),
        "PlasticFlag": np.zeros(rows.shape, dtype=bool),
        "PlasticType": np.full(rows.shape, None, dtype=object),
    }
    return pd.DataFrame(data)

def _auto_candidate_caps_by_aoi(water_pixels: int,
                                min_frac: float,
                                max_frac: float) -> tuple[int, int]:
    """
    根据 AOI 水体像元数自适应设置“候选预算”上下限：
    - 最低：至少 1500 像元，或 water_pixels*min_frac（二者取大）
    - 最高：不超过 water_pixels*max_frac，且 <= 300k
    """
    min_cap = max(1500, int(water_pixels * float(min_frac)))
    max_cap = min(300_000, max(min_cap * 5, int(water_pixels * float(max_frac))))
    return min_cap, max_cap


def _iterative_relaxation(seed_mask: np.ndarray,
                          base_mask: np.ndarray,
                          score_map: np.ndarray,
                          ndvi: np.ndarray,
                          fdi: np.ndarray,
                          ndti: np.ndarray,
                          A_B11: np.ndarray,
                          A_B12: np.ndarray,
                          *,
                          target_pixels: int,
                          prefer_high: bool,
                          # 初始阈值
                          fdi_strong: float,
                          fdi_weak: float,
                          ndti_strong: float,
                          ndti_weak: float,
                          ndvi_lo: float,
                          ndvi_hi: float,
                          absorb_thr: float,
                          # 形态约束
                          min_area_px: int,
                          max_elong: float,
                          min_sol: float,
                          # 迭代步长与上限
                          max_steps: int = 6) -> np.ndarray:
    """
    按“目标像元数”进行阶梯式放宽：每一步小幅放宽若干阈值 -> 形态过滤 -> 直到达标或步尽。
    """
    current = seed_mask.copy()
    if np.count_nonzero(current) >= target_pixels:
        return current

    # 放宽序列（按敏感度从温和到激进）
    # 每一轮都会：放宽 NDVI 上界、降低 fdi/ndti 强阈、放松吸收一致性差异
    relax_plan = []
    for step in range(max_steps):
        relax_plan.append({
            "ndvi_hi": ndvi_hi + 0.03 * (step + 1),
            "ndvi_lo": ndvi_lo - 0.03 * (step + 1),
            "fdi": max(fdi_weak, fdi_strong - 0.05 * (step + 1)),
            "ndti": max(ndti_weak, ndti_strong - 0.03 * (step + 1)),
            "absorb": max(0.45, absorb_thr - 0.05 * (step + 1)),      # B11/B12 对数吸收下限
            "abs_diff": 0.25 + 0.04 * (step + 1),                      # |A11-A12| 允许差增大
            "min_area": max(6, int(round(min_area_px * (0.8 ** (step + 1))))),
            "min_sol": max(0.25, min_sol - 0.05 * (step + 1)),
            "max_elong": min(12.0, max_elong + 1.0 * (step + 1)),
        })

    for rp in relax_plan:
        absorb_ok = (A_B11 > rp["absorb"]) & (A_B12 > rp["absorb"]) \
                    & (np.abs(A_B11 - A_B12) < rp["abs_diff"]) & base_mask
        # prefer_high=True -> fdi/ndti 走“>=阈值”的方向
        if prefer_high:
            score_ok = (score_map >= (rp["fdi"] if score_map is fdi else rp["ndti"]))
        else:
            score_ok = (score_map <= (rp["fdi"] if score_map is fdi else rp["ndti"]))

        # 合成弱条件并限定 NDVI 带
        weak = base_mask & absorb_ok & score_ok & (ndvi < rp["ndvi_hi"]) & (ndvi > rp["ndvi_lo"])
        # 邻域一致性（保持你的风格）
        grown = _neigh_consistency(weak, weak, k=3, min_hits=7)
        # 形态过滤
        grown = _region_filter(grown, min_area_px=rp["min_area"], max_elongation=rp["max_elong"], min_solidity=rp["min_sol"])
        current |= grown
        if np.count_nonzero(current) >= target_pixels:
            break

    return current


# -------------------- Visualization (with point throttling) --------------------
def visualize_results(
    rgb_image: np.ndarray,
    floating_map: np.ndarray,
    suspended_map: np.ndarray,
    plastic_map: Optional[np.ndarray],
    class_map: np.ndarray,
    df: pd.DataFrame,
    stats: Dict[str, DetectionStats],
    output_png: Path | str,
    mask: Optional[np.ndarray] = None,
    *,
    max_scatter_points: int = 20000,
) -> None:

    def _coerce_map(name: str, data: Any, fallback_shape: Optional[tuple[int, int]]) -> np.ndarray:
        """Ensure probability maps are float32 with a sensible shape for imshow."""
        if data is None:
            if fallback_shape is None:
                raise ValueError(f"No fallback shape available to materialise {name} map.")
            LOGGER.warning("%s map is missing; filling preview with NaNs.", name.capitalize())
            return np.full(fallback_shape, np.nan, dtype=np.float32)

        arr = np.asarray(data)
        if arr.ndim == 0:
            value = float(arr)
            shape = fallback_shape if fallback_shape is not None else (1, 1)
            LOGGER.warning("%s map is scalar; broadcasting value %.3f for preview.", name.capitalize(), value)
            return np.full(shape, value, dtype=np.float32)

        if arr.dtype == object:
            try:
                arr = arr.astype(np.float32)
            except (TypeError, ValueError):
                shape = fallback_shape if fallback_shape is not None else arr.shape[:2]
                if shape is None:
                    raise
                LOGGER.warning("Unable to coerce %s map to float; filling preview with NaNs.", name)
                return np.full(shape, np.nan, dtype=np.float32)
        else:
            arr = arr.astype(np.float32, copy=False)

        if arr.ndim > 2:
            arr = arr[..., 0]

        if fallback_shape is not None and arr.shape != fallback_shape:
            if arr.size == fallback_shape[0] * fallback_shape[1]:
                LOGGER.warning(
                    "%s map has shape %s; reshaping to %s for preview.", name.capitalize(), arr.shape, fallback_shape
                )
                arr = arr.reshape(fallback_shape)
            else:
                LOGGER.warning(
                    "%s map shape %s does not match expected %s; filling preview with NaNs.",
                    name.capitalize(),
                    arr.shape,
                    fallback_shape,
                )
                return np.full(fallback_shape, np.nan, dtype=np.float32)

        return arr

    mask_array = np.asarray(mask)
    fallback_shape = mask_array.shape[:2] if mask_array.ndim >= 2 else None
    floating_map = _coerce_map("floating", floating_map, fallback_shape)
    floating_shape = floating_map.shape
    suspended_map = _coerce_map("suspended", suspended_map, floating_shape)
    plastic_map = _coerce_map("plastic", plastic_map, floating_shape)
    class_map = np.asarray(class_map)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax_float_prob, ax_susp_prob = axes[0]
    ax_plastic_prob, ax_final = axes[1]

    # ---- 概率图（上排；自适应拉伸）----
    base_background = mask if mask is not None else rgb_image
    if base_background is not None:
        ax_float_prob.imshow(base_background, interpolation="nearest")
    im = ax_float_prob.imshow(floating_map, cmap="magma", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax_float_prob, label="Floating debris prob")
    ax_float_prob.set_title("Floating debris probability"); ax_float_prob.axis("off")
    
    
    if base_background is not None:
        ax_susp_prob.imshow(base_background, interpolation="nearest")
    im = ax_susp_prob.imshow(suspended_map, cmap="magma", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax_susp_prob, label="Suspended matter prob")
    ax_susp_prob.set_title("Suspended matter probability"); ax_susp_prob.axis("off")

    
    if base_background is not None:
        ax_plastic_prob.imshow(base_background, interpolation="nearest")
    im = ax_plastic_prob.imshow(plastic_map, cmap="magma", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax_plastic_prob, label="Plastic prob")
    ax_plastic_prob.set_title("Plastic probability"); ax_plastic_prob.axis("off")

    
    color_lookup = {"plastic&floating": "#ffd166",
                    "plastic&suspended": "#ff8fab"}
    
    
    # ==== 左下：漂浮相关（1=漂浮非塑料，4=塑料∧漂浮）====
    def _solid_cmap(color: str) -> ListedColormap:
        # 单色 colormap，供 imshow 叠加二值掩膜用
        return ListedColormap([color])

    # ==== 右下：最终分类 + 颜色条（保持你原来的 0,1,2,4,5 五类显示）====
    cmap = mcolors.ListedColormap([
        "white",    # 0 background
        "#ffd166",  # 4 plastic&floating
        "#ff8fab",  # 5 plastic&suspended
    ])
    bounds = np.arange(-0.5, 3.5, 1)                   # [-0.5, 0.5, ..., 5.5]
    norm = mcolors.BoundaryNorm(bounds, ncolors=cmap.N)
    if base_background is not None:
        ax_final.imshow(base_background, interpolation="nearest")
    im = ax_final.imshow(np.where(class_map, 1.0, np.nan), cmap=cmap, norm=norm, alpha=0.7)

    shown_ticks = [0, 1, 2]                      # 隐藏 3
    cbar = plt.colorbar(im, ax=ax_final, ticks=shown_ticks)
    cbar.ax.set_yticklabels([
        "background",
        "plastic&floating",
        "plastic&suspended",
    ])
    summary = "\n".join(
        f"{stat.label}: {stat.area_m2:.1f} m² ({stat.pixel_count} px)"
        for stat in stats.values()
    )
    ax_final.set_title(f"Final classification\n{summary}")
    ax_final.axis("off")
    plt.tight_layout()
    plt.savefig(output_png, dpi=400, bbox_inches="tight")
    plt.close(fig)


def compute_stats(label: str, binary_mask: np.ndarray, pixel_area: float = 100.0) -> DetectionStats:
    pixel_count = int(np.sum(binary_mask))
    area = float(pixel_count * pixel_area)
    return DetectionStats(label=label, area_m2=area, pixel_count=pixel_count)

try:
    from .temporal_io import align_and_stack_scenes  # -> 返回 (T,H,W,C), band_order, metas, transform, crs, shp_mask
except Exception:
    align_and_stack_scenes = None


# -------------------- Combined map builder --------------------
def build_combined_map(
    floating_prob, suspended_prob, *,
    water_mask, shp_mask, confident,
    floating_thresh, suspended_thresh,
    plastic_mask_pred=None,
    tie_policy="margin"  # 新增：'margin' | 'float' | 'susp' | 'strict'
):
    valid = (water_mask.astype(bool) & shp_mask.astype(bool) & confident.astype(bool))
    fp = np.where(valid, np.nan_to_num(floating_prob, nan=-np.inf), -np.inf).astype(float)
    sp = np.where(valid, np.nan_to_num(suspended_prob, nan=-np.inf), -np.inf).astype(float)

    # 过阈（含等号）
    fp_ok = fp >= float(floating_thresh)
    sp_ok = sp >= float(suspended_thresh)

    base = np.zeros(fp_ok.shape, dtype=np.uint8)
    base[fp_ok & ~sp_ok] = 1
    base[sp_ok & ~fp_ok] = 2

    # 双过阈：用“超过阈值的边际”比较，抵消尺度差
    both = fp_ok & sp_ok
    if tie_policy == "margin":
        m_fp = (fp - floating_thresh)
        m_sp = (sp - suspended_thresh)
        base[both & (m_fp >= m_sp)] = 1
        base[both & (m_sp >  m_fp)] = 2
    elif tie_policy == "float":
        base[both & (fp >= sp)] = 1
        base[both & (sp >  fp)] = 2
    elif tie_policy == "susp":
        base[both & (sp >= fp)] = 2
        base[both & (fp >  sp)] = 1
    elif tie_policy == "strict":
        base[both & (fp > sp)] = 1
        base[both & (sp > fp)] = 2
        # 相等时保持 0
    else:
        raise ValueError("tie_policy only supports 'margin'|'float'|'susp'|'strict'")

    if plastic_mask_pred is not None:
        plast = (plastic_mask_pred.astype(bool) & valid)
        base[plast & (base == 1)] = 4
        base[plast & (base == 2)] = 5
        # 仍不设置 plastic-only 为 3；如需显示可加:
        # base[plast & (base == 0)] = 3

    base[~valid] = 0
    return base

def floating_debris_detection(
    input_tif: Union[Path, str, Sequence[Path | str]],
    mask_tif: Optional[Path | str] = None,
    *,
    output_csv: Path | str,
    preview_png: Path | str,
    shp_file: Optional[Path | str] = None,
    band_order: Sequence[str] = (
        "B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12",),
        # 旋钮：适度放宽水体，推荐 water_boost=1.0 先试
    water_boost: float = 1.0,           # 0=不放宽，1=轻度放宽，>1 更宽
    min_water_ratio: float = 0.1,      # AOI 内水体最小覆盖率目标（25%）
    max_water_ratio: float = 0.9,     # AOI 内水体最大覆盖率保护（80%）
    erode_iters: int = 0,                # 水体腐蚀次数（0=不腐蚀）
    absorb_min_ratio: float = 0.60,   # 在 water_mask 中，至少多少比例满足吸收门（0.60=60%）
    absorb_max_ratio: float = 0.95,   # 保护上限，避免“全变吸收”
    absorb_thr_step: float = 0.02,  # 每次阈值微调步长（对 A 的阈值）
    absorb_diff_start: float  = 0.25 ,  # |A11-A12| 初始上限
    absorb_diff_max : float   = 0.40 ,  # 允许放松到的最大差值
    absorb_close: bool = True,   # 轻度闭运算“补孔”以连通
    seed_target_float: int | float = 20000,
    seed_target_susp: int | float = 20000,
    seed_target_float_ratio: Optional[float] = 0.05,
    seed_target_susp_ratio: Optional[float] = 0.05,
    seed_target_float_floor: Optional[int] = 800,
    seed_target_susp_floor: Optional[int] = 800,
    relax_steps=4,
    ndvi_thresh: float = 0.1,                 # 兼容旧参数
    absorbance_thresh: float = 0.8,
    fdi_std_factor: float = 1.5,
    ndti_std_factor: float = 1.0,
    water_ndwi_thresh: float = 0.05,
    floating_thresh: float = 0.6,
    suspended_thresh: float = 0.6,
    filter_field: Optional[str] = None,
    filter_values: Optional[Sequence] = None,
    filter_query: Optional[str] = None,
    filter_indices: Optional[Sequence[int]] = None,
    device: Optional[str] = None,
    epochs: int = 5,
    repeats: int = 1,
    val_split: float = 0.2,
    model_out_dir: Optional[Path | str] = "mydata",
    log_file: Optional[Path | str] = None,
    debug_dir: Optional[Path | str] = None,
    plastic_model_path: Optional[Path | str] = None,
    plastic_library_csv: Optional[Path | str] = None,
    plastic_prob_thresh: float = 0.5,
    tile_size: int = 256,
    stride: int = 256,
    max_patches: int = 1024,
    min_valid_frac: float = 0.6,
    pos_frac: float = 0.6,
    patch_seed: int = 42,
    auto_calibrate_thresholds: bool = False,
    auto_threshold_metric: str = "f1",
    auto_threshold_candidates: Optional[Sequence[float]] = None,
    auto_threshold_max_samples: Optional[int] = 128,
    auto_threshold_batch_size: int = 2,
    stacking_preset: Optional[str] = None,
    stacking_sample_limit: Optional[int] = 250_000,
    stacking_pixel_stride: int = 1,
    stacking_max_pixels: Optional[int] = None,
    stacking_cv_folds: int = 5,
    strict_mode: bool = True,
    unc_margin: float = 0.20,
    min_area_float: int = 12,
    min_area_susp: int = 12,
    cap_float_pixels: Optional[int] = 150_000,
    cap_susp_pixels: Optional[int] = 150_000,
) -> pd.DataFrame:
    # ===== 初始化 =====
    _require_torch()
    output_csv = Path(output_csv)
    preview_png = Path(preview_png)
    _ensure_stream_logger()
    handler = _configure_logger(log_file)
    if handler:
        LOGGER.info("Logging to %s", handler.baseFilename)

    import cv2 as _cv2
    import rasterio
    import geopandas as gpd
    from rasterio.features import rasterize

    # === 小工具（稳健分位数等） ===
    def _q(a: np.ndarray, p: float, mask: np.ndarray | None = None) -> float:
        if mask is not None:
            a = a[mask]
        a = a[np.isfinite(a)]
        if a.size == 0:
            return float("nan")
        return float(np.percentile(a, p))

    def _robust_thresholds_water(ndwi, ndvi, fdi, ndti, M) -> dict:
        ndwi_hi = _q(ndwi, 60, M); ndwi_lo = _q(ndwi, 35, M)   # 75/50 → 70/45，水体更宽
        if not np.isfinite(ndwi_hi): ndwi_hi = 0.05
        if not np.isfinite(ndwi_lo): ndwi_lo = 0.0

        f70, f80, f90 = _q(fdi,70, M), _q(fdi,80,M), _q(fdi,90,M)
        n70, n80, n90 = _q(ndti,70,M), _q(ndti,80,M), _q(ndti,90,M)
        f25, n25    = _q(fdi,25,M), _q(ndti,25,M)
        f_iqr = (f80 - f25) if (np.isfinite(f80) and np.isfinite(f25)) else 0.0
        n_iqr = (n80 - n25) if (np.isfinite(n80) and np.isfinite(n25)) else 0.0

        # 原来 strong: max(f90, f75 + 0.5*IQR)
        f_strong = np.nanmax([f90, f80 + 0.35*f_iqr])
        f_weak   = np.nanmax([f80, f70 + 0.15*f_iqr])

        n_strong = np.nanmax([n90, n80 + 0.35*n_iqr])
        n_weak   = np.nanmax([n80, n70 + 0.15*n_iqr])

        # 确保“强”不高于 p92，“弱”不高于 p80（防止太严）
        f_strong = float(min(f_strong, _q(fdi, 92, M) if np.isfinite(_q(fdi,92,M)) else f_strong))
        f_weak   = float(min(f_weak,   _q(fdi, 80, M) if np.isfinite(_q(fdi,80,M)) else f_weak))
        n_strong = float(min(n_strong, _q(ndti,92, M) if np.isfinite(_q(ndti,92,M)) else n_strong))
        n_weak   = float(min(n_weak,   _q(ndti,80, M) if np.isfinite(_q(ndti,80,M)) else n_weak))

        ndvi_lo = _q(ndvi, 4, M); ndvi_hi = _q(ndvi, 30, M)    # 5/30 → 4/35，窗口更宽
        if not np.isfinite(ndvi_lo): ndvi_lo = -1.0
        if not np.isfinite(ndvi_hi): ndvi_hi =  0.15

        return dict(
            ndwi_hi=float(ndwi_hi), ndwi_lo=float(ndwi_lo),
            fdi_strong=float(f_strong), fdi_weak=float(f_weak),
            ndti_strong=float(n_strong), ndti_weak=float(n_weak),
            ndvi_lo=float(ndvi_lo), ndvi_hi=float(ndvi_hi),
        )


    def _hysteresis_water(ndwi: np.ndarray, hi: float, lo: float) -> np.ndarray:
        strong = (ndwi > hi); weak = (ndwi > lo)
        if not np.any(strong):
            return weak.astype(bool)
        # 用强种子引导：膨胀→限制在 weak 内
        kernel = np.ones((3,3), np.uint8)
        grown = _cv2.dilate(strong.astype(np.uint8), kernel, iterations=2).astype(bool) & weak
        return grown

    def _erode(mask: np.ndarray, k: int = 1) -> np.ndarray:
        if k <= 0: return mask.astype(bool)
        kernel = np.ones((3,3), np.uint8)
        return _cv2.erode(mask.astype(np.uint8), kernel, iterations=int(k)).astype(bool)

    def _region_filter(bin_mask: np.ndarray, min_area_px=9, max_elongation=6.0, min_solidity=0.6) -> np.ndarray:
        out = np.zeros_like(bin_mask, dtype=np.uint8)
        num, labels, stats, _ = _cv2.connectedComponentsWithStats(bin_mask.astype(np.uint8), connectivity=8)
        for cid in range(1, num):
            area = int(stats[cid, _cv2.CC_STAT_AREA])
            if area < min_area_px:
                continue
            w = int(stats[cid, _cv2.CC_STAT_WIDTH]); h = int(stats[cid, _cv2.CC_STAT_HEIGHT])
            if w == 0 or h == 0: continue
            elong = max(w, h) / max(1.0, min(w, h))
            if elong > max_elongation: continue
            cc_mask = (labels == cid).astype(np.uint8)
            contours, _ = _cv2.findContours(cc_mask, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            hull = _cv2.convexHull(contours[0])
            hull_area = float(_cv2.contourArea(hull))
            if hull_area <= 0: continue
            solidity = float(area) / hull_area
            if solidity < min_solidity: continue
            out[labels == cid] = 1
        return out.astype(bool)

    def _neigh_consistency(core: np.ndarray, cond: np.ndarray, k: int = 5, min_hits: int = 8) -> np.ndarray:
        kernel = np.ones((k, k), np.uint8)
        hits = _cv2.filter2D(cond.astype(np.uint8), -1, kernel, borderType=_cv2.BORDER_REFLECT)
        ok = hits >= min_hits
        return core & ok

    def _cap_candidates(seed: np.ndarray, score: np.ndarray, max_pixels: Optional[int]) -> np.ndarray:
        if max_pixels is None: return seed
        idx = np.flatnonzero(seed.ravel())
        if idx.size <= max_pixels:
            return seed
        s = score.reshape(-1)[idx]
        order = np.argsort(-s)  # 高分优先
        keep = idx[order[:max_pixels]]
        out = np.zeros_like(seed, dtype=bool); out.reshape(-1)[keep] = True
        return out


    def _gen_seeds_with_relax(core, cond, area_min, elong_max, sol_min,
                            neigh_k=5, neigh_hits=8,
                            relax=False):
        # 第一档
        seed_neigh = _neigh_consistency(core, cond, k=neigh_k, min_hits=neigh_hits)
        def _close(mask, k=3, it=1):
            if k <= 0 or it <= 0: return mask
            kr = np.ones((k,k), np.uint8)
            m = _cv2.dilate(mask.astype(np.uint8), kr, iterations=it)
            m = _cv2.erode(m, kr, iterations=it)
            return (m > 0)

        # 在 _gen_seeds_with_relax 里生成 seed_neigh 后加：
        seed_neigh = _close(seed_neigh, k=3, it=1)
        seed = _region_filter(seed_neigh, min_area_px=area_min, max_elongation=elong_max, min_solidity=sol_min)
        if np.count_nonzero(seed) > 0 or not relax:
            return seed

        # 第二档（更松）
        seed_neigh2 = _neigh_consistency(core, cond, k=3, min_hits=max(3, neigh_hits-3))
        seed2 = _region_filter(seed_neigh2, min_area_px=max(4, area_min//2), max_elongation=elong_max+2.0, min_solidity=max(0.25, sol_min-0.2))
        return seed2
    # --- normalize new knobs to int ---
    DEFAULT_SEED_TARGET = 20000.0
    raw_seed_target_float = seed_target_float
    raw_seed_target_susp = seed_target_susp

    def _as_int(x, name, default):
        import numpy as _np
        if x is None:
            return int(default)
        try:
            if isinstance(x, str):
                x_clean = x.strip()
                if not x_clean:
                    return int(default)
                if x_clean.endswith("%"):
                    x = float(x_clean.rstrip("%")) / 100.0
                else:
                    x = float(x_clean)
            if isinstance(x, (list, tuple, _np.ndarray)):
                if len(x) == 0:
                    return int(default)
                # 有多个数时取最大或第一个，二选一；这里取最大更保险
                return int(max(x))
            if isinstance(x, (_np.floating, float)):
                if not np.isfinite(x):
                    return int(default)
                return int(round(float(x)))
            return int(x)
        except Exception:
            LOGGER.warning("%s provided as %r; fallback to %d", name, x, default)
            return int(default)

    def _sanitize_ratio(value: Optional[float], name: str) -> Optional[float]:
        if value is None:
            return None
        try:
            ratio_val = float(value)
        except Exception:
            LOGGER.warning("%s provided as %r; ignore ratio", name, value)
            return None
        if not np.isfinite(ratio_val) or ratio_val <= 0:
            LOGGER.warning("%s ratio %.4f invalid; ignore", name, ratio_val)
            return None
        if ratio_val >= 1.0:
            LOGGER.warning("%s ratio %.3f >= 1.0; ignore", name, ratio_val)
            return None
        return ratio_val

    def _is_default_seed(value: object) -> bool:
        try:
            return abs(float(value) - DEFAULT_SEED_TARGET) < 1e-6
        except Exception:
            return False

    def _resolve_seed_target(base_target: int,
                              ratio: Optional[float],
                              floor: Optional[int],
                              available: int,
                              label: str) -> int:
        avail = max(0, int(available))
        if avail == 0:
            LOGGER.warning("Seed target resolved (%s): no water pixels available; using 0.", label)
            return 0
        target = max(0, int(base_target))
        detail_parts = [f"base={target}"]
        if ratio is not None:
            ratio_target = int(round(avail * ratio))
            ratio_target = max(0, min(ratio_target, avail))
            target = max(target, ratio_target)
            detail_parts.append(f"ratio={ratio:.4f}")
            detail_parts.append(f"ratio_target={ratio_target}")
        if floor is not None and floor > 0:
            floor_target = min(avail, int(floor))
            target = max(target, floor_target)
            detail_parts.append(f"floor={floor_target}")
        if target > avail:
            detail_parts.append(f"clipped={avail}")
            target = avail
        target = max(1, target)
        LOGGER.info(
            "Seed target resolved (%s): %s -> %s (water pixels=%s)",
            label,
            " | ".join(detail_parts),
            target,
            f"{avail:,}",
        )
        return target
         # 最多回退层数

    seed_target_float_ratio = _sanitize_ratio(seed_target_float_ratio, "seed_target_float_ratio")
    seed_target_susp_ratio = _sanitize_ratio(seed_target_susp_ratio, "seed_target_susp_ratio")
    seed_target_float_floor = None if seed_target_float_floor is None else max(0, int(seed_target_float_floor))
    seed_target_susp_floor = None if seed_target_susp_floor is None else max(0, int(seed_target_susp_floor))

    if seed_target_float_ratio is not None and _is_default_seed(raw_seed_target_float):
        seed_target_float = 0
    if seed_target_susp_ratio is not None and _is_default_seed(raw_seed_target_susp):
        seed_target_susp = 0
    # 严格模式参数微调
    if strict_mode:
        absorbance_thresh = max(absorbance_thresh, 0.8)  # 吸收更严
        unc_margin = max(unc_margin, 0.20)
        min_area_float = max(min_area_float, 16)
        min_area_susp  = max(min_area_susp, 16)

    # === 严格模式微调 ===
    # ----------------- 读取矢量 AOI（保持你原逻辑） -----------------
    gdf: Optional[gpd.GeoDataFrame] = None
    gdf_original: Optional[gpd.GeoDataFrame] = None
    shapefile_alignment_warning: Optional[str] = None
    aoi_geometry = None
    if shp_file is not None:
        with tqdm(total=1, desc="Reading shapefile") as pbar:
            gdf_read = gpd.read_file(shp_file)
            gdf_filtered = filter_gdf(gdf_read, filter_field, filter_values, filter_query, filter_indices)
            gdf_original = gdf_filtered.copy()
            try:
                with rasterio.open(input_tif if isinstance(input_tif, (str, Path)) else input_tif[0]) as src_ref:
                    if src_ref.crs is not None:
                        gdf = gdf_filtered.to_crs(src_ref.crs)
                    else:
                        LOGGER.warning("Input raster has no CRS; using shapefile CRS directly.")
                        gdf = gdf_filtered.copy()
            except Exception as exc:
                LOGGER.warning("Failed to align shapefile CRS with raster CRS: %s", exc)
                gdf = gdf_filtered.copy()
            try:
                aoi_geometry = gdf.geometry.union_all()
            except AttributeError:
                aoi_geometry = gdf.geometry.unary_union
            pbar.update(1)
    else:
        msg = "No shapefile provided; using automatic water masking."
        LOGGER.info(msg); tqdm.write(msg)

    # ----------------- 多期对齐 / 单期读取 -----------------
    is_multi = isinstance(input_tif, (list, tuple))
    band_cube: np.ndarray
    transform = None
    src_crs = None
    shp_mask: np.ndarray

    full_band_order = ("B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12")
    compact_band_order = ("B2","B3","B4","B5","B8","B8A","B9","B11","B12")
    minimal_band_order = ("B2","B3","B4","B8","B8A","B11","B12")

    # 重要：band_order 在后面 require_band 时使用
    if is_multi:
        if align_and_stack_scenes is None:
            raise RuntimeError("align_and_stack_scenes() 未找到。请添加 .temporal_io.align_and_stack_scenes 或改为单期输入。")
        stack, tgt_order, _, transform, src_crs, shp_mask = align_and_stack_scenes(
            input_tifs=input_tif,
            target_band_order=full_band_order,
            resampling="bilinear",
            scale=1e-4,
            shp_file=shp_file,
            aoi_filter=(dict(field=filter_field, values=filter_values)
                        if (filter_field and filter_values) else None),
        )
        band_cube = stack[-1]  # 末帧作为当前影像
        band_order = tuple(tgt_order)
    else:
        with rasterio.open(input_tif) as src_tmp:
            available_count = src_tmp.count
            raw_descriptions = src_tmp.descriptions or []
            descriptions = [desc.strip().upper() for desc in raw_descriptions if isinstance(desc, str)]

        known_bands = set(S2_BANDS.keys())
        if descriptions and all(desc in known_bands for desc in descriptions):
            band_order = tuple(descriptions)
        else:
            candidate_orders = []
            if band_order: candidate_orders.append(tuple(band_order))
            candidate_orders.extend([full_band_order, compact_band_order, minimal_band_order])
            seen = set(); unique_candidates = []
            for order in candidate_orders:
                if order not in seen:
                    unique_candidates.append(order); seen.add(order)
            chosen = None
            for order in unique_candidates:
                if len(order) == available_count:
                    chosen = order; break
            if chosen is None:
                if available_count < len(minimal_band_order):
                    raise ValueError(
                        f"Unsupported band configuration: file has {available_count} bands < {len(minimal_band_order)} (minimal)."
                    )
                base = unique_candidates[0]; chosen = tuple(base[:available_count])
                LOGGER.warning("Unknown layout; assume first %s bands of %s -> %s", available_count, base, chosen)
            band_order = tuple(chosen)

        band_indices = list(range(1, len(band_order) + 1))
        bands_raw, transform, src_meta = read_bands(input_tif, band_indices)
        band_cube = bands_raw.astype(np.float32) * 0.0001
        src_crs = src_meta.get("crs") if isinstance(src_meta, dict) else getattr(src_meta, "get", lambda *_: None)("crs")

    # ----------------- shapefile 栅格化掩膜（多期时已返回） -----------------
    if gdf is not None and not is_multi:
        def _rasterize_from_gdf(gdf_for_mask: gpd.GeoDataFrame) -> np.ndarray:
            return rasterize(
                [(geom, 1) for geom in gdf_for_mask.geometry],
                out_shape=(band_cube.shape[0], band_cube.shape[1]),
                transform=transform,
                fill=0,
                dtype=np.uint8,
            ).astype(bool)
        with tqdm(total=1, desc="Creating shapefile mask") as pbar:
            shp_mask = _rasterize_from_gdf(gdf); pbar.update(1)

        if not np.any(shp_mask) and gdf_original is not None:
            fallback_gdf, alignment_msg = _fallback_project_shapefile(gdf_original, src_crs, transform)
            if fallback_gdf is not None:
                with tqdm(total=1, desc="Reprojecting shapefile for mask") as pbar:
                    shp_mask = _rasterize_from_gdf(fallback_gdf); pbar.update(1)
                if np.any(shp_mask):
                    gdf = fallback_gdf; shapefile_alignment_warning = alignment_msg
                    try: aoi_geometry = gdf.geometry.union_all()
                    except AttributeError: aoi_geometry = gdf.geometry.unary_union
        if not np.any(shp_mask):
            LOGGER.warning("Shapefile mask empty; revert to full-scene mask.")
            shp_mask = np.ones((band_cube.shape[0], band_cube.shape[1]), dtype=bool)
        elif shapefile_alignment_warning:
            LOGGER.warning(shapefile_alignment_warning)
    elif not is_multi:
        shp_mask = np.ones((band_cube.shape[0], band_cube.shape[1]), dtype=bool)

    plastic_enabled = plastic_library_csv is not None and Path(plastic_library_csv).exists()
    if plastic_library_csv is not None and not Path(plastic_library_csv).exists():
        LOGGER.warning("Skipping plastic ML (library CSV missing: %s)", plastic_library_csv)
        plastic_enabled = False
    
    # ----------------- 波段准备 -----------------
    def require_band(name: str) -> np.ndarray:
        bm = {b: i for i, b in enumerate(band_order)}
        if name not in bm:
            raise KeyError(f"Band {name} missing. band_order={band_order}")
        return band_cube[:, :, bm[name]]

    with tqdm(total=1, desc="Processing bands") as pbar:
        B2 = require_band("B2");  B3 = require_band("B3");  B4 = require_band("B4")
        B5 = band_cube[:, :, band_order.index("B5")]  if "B5"  in band_order else None
        B6 = band_cube[:, :, band_order.index("B6")]  if "B6"  in band_order else None
        B7 = band_cube[:, :, band_order.index("B7")]  if "B7"  in band_order else None
        B8 = require_band("B8")
        B8A= band_cube[:, :, band_order.index("B8A")] if "B8A" in band_order else None
        B9 = band_cube[:, :, band_order.index("B9")]  if "B9"  in band_order else None
        B11= require_band("B11"); B12 = require_band("B12")
        pbar.update(1)

    for arr in (B2,B3,B4,B8,B11,B12):
        arr[~shp_mask] = np.nan
    if B5 is not None:  B5[~shp_mask] = np.nan
    if B6 is not None:  B6[~shp_mask] = np.nan
    if B7 is not None:  B7[~shp_mask] = np.nan
    if B8A is not None: B8A[~shp_mask] = np.nan
    if B9 is not None:  B9[~shp_mask]  = np.nan
    

    rgb_preview = np.dstack([
        np.clip(B4 / np.nanmax(B4, where=shp_mask, initial=1), 0, 1),
        np.clip(B3 / np.nanmax(B3, where=shp_mask, initial=1), 0, 1),
        np.clip(B2 / np.nanmax(B2, where=shp_mask, initial=1), 0, 1),
    ])

    # ----------------- NDWI & 更稳健的水体（放宽 + 覆盖率自适应） -----------------
    with tqdm(total=1, desc="Computing NDWI") as pbar:
        ndwi = compute_ndwi(B3, B8)
        # 初步粗水体（自适应分位数稍微降低，65% -> 55%，更宽）
        rough_thr = water_ndwi_thresh
        try:
            prc = float(np.nanpercentile(ndwi[shp_mask], 35))
            if np.isfinite(prc):
                rough_thr = max(min(prc,  0.6), -0.5)
        except Exception:
            pass
        rough_water = (ndwi > rough_thr) & shp_mask
        if not np.any(rough_water):
            # 兜底：回到原阈值
            rough_water = (ndwi > water_ndwi_thresh) & shp_mask

        # 临时指标仅用于阈值估计
        _ndvi_tmp = (B8 - B4) / (B8 + B4 + 1e-6)
        _fdi_tmp  = compute_fdi(B4, B8, B11)
        _ndti_tmp = compute_ndti(B4, B3)
        rob = _robust_thresholds_water(ndwi, _ndvi_tmp, _fdi_tmp, _ndti_tmp, rough_water)

        # —— 基准滞后阈值
        hi, lo = float(rob["ndwi_hi"]), float(rob["ndwi_lo"])

        # —— 轻度放宽（随 water_boost 下调）
        if water_boost and water_boost > 0:
            hi -= 0.01 * water_boost
            lo -= 0.02 * water_boost
            hi = float(np.clip(hi, -1.0, 1.0))
            lo = float(np.clip(lo, -1.0, hi - 1e-3))

        # 初始水体 + 轻/无腐蚀
        water_mask = _hysteresis_water(ndwi, hi, lo) & shp_mask
        water_mask = _erode(water_mask, k=erode_iters)
        if not np.any(water_mask):
            water_mask = rough_water

        # —— 覆盖率自适应：把水体比例推向 [min_water_ratio, max_water_ratio]
        total_aoi = max(1, _mask_count(shp_mask))
        target_min = int(min_water_ratio * total_aoi)
        target_max = int(max_water_ratio * total_aoi)
        step = 0.01
        n_iter = 0
        cov = _mask_count(water_mask)

        # 过少：逐步下调 hi/lo
        while cov < target_min and n_iter < 10:
            hi -= step
            lo -= step * 1.2
            lo = float(np.clip(lo, -1.0, hi - 1e-3))
            water_mask = _hysteresis_water(ndwi, hi, lo) & shp_mask
            water_mask = _erode(water_mask, k=erode_iters)
            cov = _mask_count(water_mask)
            n_iter += 1

        # 仍偏少：轻微膨胀一下再与 AOI 相与（避免越界）
        if cov < target_min:
            kernel = np.ones((3,3), np.uint8)
            grown = _cv2.dilate(water_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
            water_mask = (grown | water_mask) & shp_mask
            cov = _mask_count(water_mask)

        # 过多：上调 lo（只收紧弱门即可）
        n_iter = 0
        while cov > target_max and n_iter < 10:
            lo += step
            if lo >= hi: lo = hi - 1e-3
            water_mask = _hysteresis_water(ndwi, hi, lo) & shp_mask
            water_mask = _erode(water_mask, k=erode_iters)
            cov = _mask_count(water_mask)
            n_iter += 1

        pbar.update(1)

    rough_water_pixels = _mask_count(rough_water)
    water_pixels_initial = _mask_count(water_mask)
    aoi_pixels = max(1, _mask_count(shp_mask))
    water_pct = (water_pixels_initial / aoi_pixels) * 100.0
    LOGGER.info(
        "Water mask stats: rough=%s px, tuned=%s px (%.2f%% of AOI, target %.1f%%–%.1f%%).",
        f"{rough_water_pixels:,}",
        f"{water_pixels_initial:,}",
        water_pct,
        min_water_ratio * 100.0,
        max_water_ratio * 100.0,
    )

    water_pixels_available = max(0, water_pixels_initial)
    seed_target_float = _resolve_seed_target(
        seed_target_float,
        seed_target_float_ratio,
        seed_target_float_floor,
        water_pixels_available,
        "floating",
    )
    seed_target_susp = _resolve_seed_target(
        seed_target_susp,
        seed_target_susp_ratio,
        seed_target_susp_floor,
        water_pixels_available,
        "suspended",
    )

    _save_debug_image(debug_dir, "ndwi_map.png", ndwi,
                    title=f"NDWI (hi={hi:.3f}, lo={lo:.3f})",
                    cmap="coolwarm", mask=shp_mask, vmin=-0.5, vmax=0.5)
    _save_debug_overlay(debug_dir, "water_mask_overlay.png", rgb_preview, water_mask,
                        title="Detected water mask (auto-tuned)", color="cyan", mask=shp_mask)

    # ----------------- 指标计算（用稳健阈值） -----------------
    with tqdm(total=3, desc="Computing spectral indices") as pbar:
        ndwi = compute_ndwi(B3, B8); ndwi[~shp_mask] = np.nan; pbar.update(1)
        ndvi = (B8 - B4) / (B8 + B4 + 1e-6); ndvi[~shp_mask] = np.nan; pbar.update(1)
        fdi  = compute_fdi(B4, B8, B11);      fdi[~shp_mask]  = np.nan; pbar.update(1)
        ndti = compute_ndti(B4, B3);          ndti[~shp_mask] = np.nan; pbar.update(1)
        sabi = compute_sabi(B8, B4);          sabi[~shp_mask] = np.nan; pbar.update(1)
    
    fdi_strong, fdi_weak = rob["fdi_strong"], rob["fdi_weak"]
    ndti_strong, ndti_weak = rob["ndti_strong"], rob["ndti_weak"]
    ndvi_lo_r, ndvi_hi_r = rob["ndvi_lo"], rob["ndvi_hi"]
    LOGGER.info(
        "Robust thresholds: NDWI_hi=%.3f NDWI_lo=%.3f | FDI strong=%.3f weak=%.3f | NDTI strong=%.3f weak=%.3f | NDVI window [%.3f, %.3f].",
        rob["ndwi_hi"],
        rob["ndwi_lo"],
        fdi_strong,
        fdi_weak,
        ndti_strong,
        ndti_weak,
        ndvi_lo_r,
        ndvi_hi_r,
    )

    _save_debug_image(debug_dir, "ndvi_map.png", ndvi, title="NDVI", cmap="RdYlGn", mask=water_mask, vmin=-1, vmax=1)
    _save_debug_image(debug_dir, "fdi_map.png",  fdi,  title=f"FDI (str={fdi_strong:.3f}, weak={fdi_weak:.3f})",  cmap="plasma", mask=water_mask)
    _save_debug_image(debug_dir, "ndti_map.png", ndti, title=f"NDTI (str={ndti_strong:.3f}, weak={ndti_weak:.3f})", cmap="viridis", mask=water_mask)

    
    _close_kernel = np.ones((3,3), np.uint8)
    absorb_gating_disabled = False
    # ----------------- 吸收一致性 -----------------
    with tqdm(total=2, desc="Computing absorbance") as pbar:
    # 用 clip 避免 -log10(0)
        B11_pos = np.clip(B11, 1e-6, None)
        B12_pos = np.clip(B12, 1e-6, None)
        A_B11 = -np.log10(B11_pos)
        A_B12 = -np.log10(B12_pos)
        A_B11[~shp_mask] = np.nan; A_B12[~shp_mask] = np.nan
        pbar.update(1)

        # 以 min(A11,A12) 为强度，差值限制保证一致性
        A_min = np.minimum(A_B11, A_B12)
        A_diff = np.abs(A_B11 - A_B12)

        # 初始阈值：优先用给定 absorbance_thresh；若不可用则用分位数估计
        if np.isfinite(absorbance_thresh) and absorbance_thresh > 0:
            thr = float(absorbance_thresh)
        else:
            vals = A_min[water_mask & np.isfinite(A_min)]
            thr = float(np.nanpercentile(vals, 60)) if vals.size > 0 else 0.60

        diff_lim = float(absorb_diff_start)

        # 初始 mask
        absorb_mask = (A_min > thr) & (A_diff < diff_lim) & shp_mask
        pbar.update(1)

    # 自适应覆盖率微调（无论是否=0都执行，而非只在=0时）
    water_total_pixels = max(1, _mask_count(water_mask))
    cov = _mask_count(absorb_mask & water_mask)
    target_min = int(absorb_min_ratio * water_total_pixels)
    target_max = int(absorb_max_ratio * water_total_pixels)

    # 过少：降低强度阈值 thr，必要时放宽差值上限
    iter_cnt = 0
    while cov < target_min and iter_cnt < 20:
        thr -= absorb_thr_step
        if thr < 0.05: thr = 0.05  # 不要太低
        if cov < target_min * 0.7:  # 明显偏少时，适度放宽差值
            diff_lim = min(absorb_diff_max, diff_lim + 0.02)
        absorb_mask = (A_min > thr) & (A_diff < diff_lim) & shp_mask
        cov = _mask_count(absorb_mask & water_mask)
        iter_cnt += 1

    # 过多：升高 thr（只收紧强度门）
    iter_cnt = 0
    while cov > target_max and iter_cnt < 20:
        thr += absorb_thr_step
        absorb_mask = (A_min > thr) & (A_diff < diff_lim) & shp_mask
        cov = _mask_count(absorb_mask & water_mask)
        iter_cnt += 1

    # 形态学“补孔”，保持在水体内
    if absorb_close:
        m = _cv2.morphologyEx(absorb_mask.astype(np.uint8), _cv2.MORPH_CLOSE, _close_kernel, iterations=1)
        absorb_mask = (m > 0) & water_mask

    absorb_final_pixels = _mask_count(absorb_mask & water_mask)
    absorb_cov_pct = (absorb_final_pixels / water_total_pixels) * 100.0
    LOGGER.info(
        "Absorbance gating (auto): thr=%.2f, |Δ|<=%.2f -> %.2f%% of water (%s/%s px, target %.1f%%–%.1f%%).",
        thr,
        diff_lim,
        absorb_cov_pct,
        f"{absorb_final_pixels:,}",
        f"{water_total_pixels:,}",
        absorb_min_ratio * 100.0,
        absorb_max_ratio * 100.0,
    )

    if absorb_final_pixels == 0:
        LOGGER.warning(
            "Absorbance gating produced zero coverage; disabling gating for this scene.")
        absorb_mask = water_mask & shp_mask
        absorb_gating_disabled = True
        absorb_final_pixels = _mask_count(absorb_mask & water_mask)
        absorb_cov_pct = (absorb_final_pixels / water_total_pixels) * 100.0
    elif absorb_cov_pct < max(1.0, absorb_min_ratio * 25.0):
        LOGGER.warning(
            "Absorbance gating coverage %.2f%% is <25%% of target %.2f%%; relaxing to water mask for seed growth.",
            absorb_cov_pct,
            absorb_min_ratio * 100.0,
        )
        absorb_mask = (absorb_mask | water_mask) & water_mask
        absorb_final_pixels = _mask_count(absorb_mask & water_mask)
        absorb_cov_pct = (absorb_final_pixels / water_total_pixels) * 100.0
        absorb_gating_disabled = True

    LOGGER.info(
        "Absorbance gating final coverage: %.2f%% of water mask (%s/%s px).",
        absorb_cov_pct,
        f"{absorb_final_pixels:,}",
        f"{water_total_pixels:,}",
    )

    _save_debug_image(debug_dir, "absorbance_B11.png", A_B11, title="Absorbance B11", cmap="magma", mask=water_mask)
    _save_debug_image(debug_dir, "absorbance_B12.png", A_B12, title="Absorbance B12", cmap="magma", mask=water_mask)

    # ----------------- 外部 mask_tif（可选） -----------------
    if mask_tif is not None and Path(mask_tif).exists():
        with tqdm(total=1, desc="Reading mask") as pbar:
            with rasterio.open(mask_tif) as msk:
                mask = msk.read(1); pbar.update(1)
        ref_shape = water_mask.shape
        with tqdm(total=1, desc="Resizing mask") as pbar:
            mask = resize_to_reference(mask, ref_shape); mask[~shp_mask] = 0; pbar.update(1)
    else:
        mask = water_mask.astype(np.uint8)
        LOGGER.info("No mask_tif provided; using NDWI-derived water mask for seed generation.")

    # ----------------- SABI / BD_B9 / 派生 -----------------
    sabi = compute_sabi(B8, B4); sabi[~shp_mask] = np.nan
    bd_b9 = None
    if B9 is not None and B8A is not None:
        bd_b9 = band_depth(band_value=B9, left_value=B8A, right_value=B11,
                           left_center=865.0, right_center=1610.0, band_center=945.0)
        bd_b9[~shp_mask] = np.nan

    # ----------------- 简单指数阈值种子生成 -----------------
    def _topk_mask(score: np.ndarray, base_mask: np.ndarray, k: int) -> np.ndarray:
        result = np.zeros_like(base_mask, dtype=bool)
        if k <= 0:
            return result
        flat_idx = np.flatnonzero(base_mask.reshape(-1))
        if flat_idx.size == 0:
            return result
        flat_score = score.reshape(-1)[flat_idx]
        if not np.all(np.isfinite(flat_score)):
            flat_score = np.nan_to_num(flat_score, nan=-np.inf)
        k = int(min(max(k, 0), flat_score.size))
        if k == 0:
            return result
        selected = flat_idx[np.argpartition(flat_score, -k)[-k:]]
        result.reshape(-1)[selected] = True
        return result

    def _percentile_seed(score: np.ndarray, base_mask: np.ndarray, percentile: float, min_pixels: int) -> np.ndarray:
        min_pixels = max(1, int(min_pixels))
        candidate = np.zeros_like(base_mask, dtype=bool)
        perc = float(percentile)
        for _ in range(6):
            thr = _q(score, perc, base_mask)
            if not np.isfinite(thr):
                break
            candidate = base_mask & (score >= thr)
            if int(np.count_nonzero(candidate)) >= min_pixels or perc <= 5.0:
                break
            perc = max(5.0, perc - 5.0)
        if int(np.count_nonzero(candidate)) < min_pixels:
            candidate = _topk_mask(score, base_mask, min_pixels)
        return candidate

    def _clean_seed(seed: np.ndarray, *, min_area: int, elong_max: float, sol_min: float) -> np.ndarray:
        if not np.any(seed):
            return seed
        closed = _cv2.morphologyEx(seed.astype(np.uint8), _cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        filtered = _region_filter(
            closed.astype(bool),
            min_area_px=max(4, int(min_area)),
            max_elongation=float(elong_max),
            min_solidity=float(sol_min),
        )
        return filtered

    float_base = (water_mask & absorb_mask) if np.any(absorb_mask) else water_mask
    if np.any(float_base):
        float_min_pixels = max(1, min(seed_target_float, int(np.count_nonzero(float_base))))
        float_seed = _percentile_seed(fdi, float_base, percentile=85.0, min_pixels=float_min_pixels)
        ndvi_lo_simple = _q(ndvi, 5, float_base)
        ndvi_hi_simple = _q(ndvi, 65, float_base)
        if np.isfinite(ndvi_lo_simple):
            float_seed &= ndvi >= ndvi_lo_simple
        if np.isfinite(ndvi_hi_simple):
            float_seed &= ndvi <= ndvi_hi_simple
        float_seed = _clean_seed(float_seed, min_area=min_area_float, elong_max=8.0, sol_min=0.5)
        float_seed = _cap_candidates(float_seed, fdi, cap_float_pixels)
    else:
        float_seed = np.zeros_like(water_mask, dtype=bool)

    susp_base = water_mask
    if np.any(susp_base):
        susp_min_pixels = max(1, min(seed_target_susp, int(np.count_nonzero(susp_base))))
        susp_seed = _percentile_seed(ndti, susp_base, percentile=85.0, min_pixels=susp_min_pixels)
        ndvi_hi_susp = _q(ndvi, 70, susp_base)
        if np.isfinite(ndvi_hi_susp):
            susp_seed &= ndvi <= ndvi_hi_susp
        susp_seed = _clean_seed(susp_seed, min_area=min_area_susp, elong_max=6.0, sol_min=0.6)
        susp_seed = _cap_candidates(susp_seed, ndti, cap_susp_pixels)
    else:
        susp_seed = np.zeros_like(water_mask, dtype=bool)

    # ----------------- 无水体兜底 -----------------
    water_pixels = int(np.count_nonzero(water_mask))
    if water_pixels == 0:
        warn_msg = "No water pixels detected; aborting UNet training."
        LOGGER.warning(warn_msg); tqdm.write(warn_msg)
        floating_prob = np.zeros_like(water_mask, dtype=np.float32)
        suspended_prob = np.zeros_like(water_mask, dtype=np.float32)
        combined_map = np.zeros_like(water_mask, dtype=np.uint8)
        df = pd.DataFrame(columns=[
            "row","col","x","y","lon","lat","class","probability","NDVI","FDI","NDTI","SABI","A_B11","A_B12","PlasticProb","PlasticFlag",
        ])
        stats = {
            "floating_debris": compute_stats("floating_debris", floating_prob.astype(bool)),
            "suspended_matter": compute_stats("suspended_matter", suspended_prob.astype(bool)),
        }
        rgb = np.dstack([
            np.clip(B4 / np.nanmax(B4, where=shp_mask,  initial=1), 0, 1),
            np.clip(B3 / np.nanmax(B3, where=shp_mask,  initial=1), 0, 1),
            np.clip(B2 / np.nanmax(B2, where=shp_mask,  initial=1), 0, 1),
        ])
        visualize_results(
            rgb_image=rgb, 
            floating_map=floating_prob,
            suspended_map=suspended_prob,
            plastic_map=np.zeros_like(floating_prob), 
            class_map=combined_map,
            df=df, stats=stats, output_png=preview_png)

        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        if handler: handler.flush(); LOGGER.removeHandler(handler); handler.close()
        return df

    # ----------------- 打印伪标签规模 -----------------
    floating_seed_count = int(np.count_nonzero(float_seed))
    suspended_seed_count = int(np.count_nonzero(susp_seed))
    LOGGER.info("AOI pixels: %s | floating seeds: %s | suspended seeds: %s",
                f"{water_pixels:,}", f"{floating_seed_count:,}", f"{suspended_seed_count:,}")
    tqdm.write(f"AOI pixels: {water_pixels:,} | floating seeds: {floating_seed_count:,} | suspended seeds: {suspended_seed_count:,}")

    # ----------------- 训练数据准备（与你原逻辑一致） -----------------
    # ----------------- 训练数据准备（多期特征改动点） -----------------
    derived_indices_map = {
        "NDWI": ndwi, "NDVI": ndvi, "FDI": fdi, "NDTI": ndti, "SABI": sabi, "A_B11": A_B11, "A_B12": A_B12,
    }
    if B5 is not None: derived_indices_map["rho_B5"] = B5
    if B9 is not None: derived_indices_map["rho_B9"] = B9

    # 稳健归一（以水体为统计范围）
    def robust_scale(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        v = arr[mask]
        v = v[np.isfinite(v)]
        if v.size < 100:
            return arr
        med = np.median(v)
        iqr = np.percentile(v, 75) - np.percentile(v, 25)
        scale = iqr if iqr > 1e-6 else (np.std(v) + 1e-6)
        out = (arr - med) / scale
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    # —— 先对末帧所有用于训练的光谱/指数稳健归一（与你原先一致）——
    for arr in (B2, B3, B4, B8, B11, B12):
        arr[:] = robust_scale(arr, water_mask)
    if B5  is not None: B5[:]  = robust_scale(B5,  water_mask)
    if B8A is not None: B8A[:] = robust_scale(B8A, water_mask)
    if B9  is not None: B9[:]  = robust_scale(B9,  water_mask)

    ndwi = robust_scale(ndwi, water_mask)
    ndvi = robust_scale(ndvi, water_mask)
    fdi  = robust_scale(fdi,  water_mask)
    ndti = robust_scale(ndti, water_mask)
    sabi = robust_scale(sabi, water_mask)
    A_B11 = robust_scale(A_B11, water_mask)
    A_B12 = robust_scale(A_B12, water_mask)

    # === 构建 features ===
    if not is_multi:
        # 单期：保持你原始特征顺序
        feature_list: List[np.ndarray] = [B2, B3, B4]
        if B5  is not None: feature_list.append(B5)
        feature_list.extend([B8])
        if B8A is not None: feature_list.append(B8A)
        if B9  is not None: feature_list.append(B9)
        feature_list.extend([B11, B12, ndwi, ndvi, fdi, ndti, sabi, A_B11, A_B12])
        features = np.stack(feature_list, axis=-1).astype(np.float32)
    else:
        # 多期：对所有原始波段做 mean/std/max，再拼末帧指数
        # stack: (T,H,W,C) 已按 full_band_order 统一
        def _robust_scale_cube(cube: np.ndarray) -> np.ndarray:
            out = cube.copy()
            for k in range(out.shape[-1]):
                out[..., k] = robust_scale(out[..., k], water_mask)
            return out

        feats_mean = np.nanmean(stack, axis=0)     # (H,W,C)
        feats_std  = np.nanstd(stack, axis=0)
        feats_max  = np.nanmax(stack, axis=0)
        feats_mean = _robust_scale_cube(feats_mean)
        feats_std  = _robust_scale_cube(feats_std)
        feats_max  = _robust_scale_cube(feats_max)

        features = np.concatenate([feats_mean, feats_std, feats_max], axis=-1).astype(np.float32)
        # 末帧指数 7 项（与单期保持一致）
        tail = np.stack([ndwi, ndvi, ndti, fdi, sabi, A_B11, A_B12], axis=-1).astype(np.float32)
        features = np.concatenate([features, tail], axis=-1)

    features[np.isnan(features)] = 0.0

    label_stack = np.stack([float_seed.astype(np.float32), susp_seed.astype(np.float32)], axis=0)
    if label_stack.ndim == 2:
        label_stack = label_stack[np.newaxis, ...]
    elif label_stack.ndim == 3 and label_stack.shape[0] > 8 and label_stack.shape[-1] <= 8:
        label_stack = np.transpose(label_stack, (2, 0, 1))

    if np.count_nonzero(float_seed) == 0 and np.count_nonzero(susp_seed) == 0:
        warn_msg = "No pseudo-labels available; skip UNet training."
        LOGGER.warning(warn_msg); tqdm.write(warn_msg)
        floating_prob = float_seed.astype(np.float32)
        suspended_prob = susp_seed.astype(np.float32)
        combined_map = np.zeros_like(floating_prob, dtype=np.uint8)
        df = pd.DataFrame(columns=[
            "row","col","x","y","lon","lat","class","probability","NDVI","FDI","NDTI","SABI","A_B11","A_B12","PlasticProb","PlasticFlag","PlasticType",
        ])
        stats = {
            "floating_debris": compute_stats("floating_debris", float_seed),
            "suspended_matter": compute_stats("suspended_matter", susp_seed),
        }
        rgb = np.dstack([
            np.clip(B4 / np.nanmax(B4, where=water_mask, initial=1), 0, 1),
            np.clip(B3 / np.nanmax(B3, where=water_mask, initial=1), 0, 1),
            np.clip(B2 / np.nanmax(B2, where=water_mask, initial=1), 0, 1),
        ])
        visualize_results(rgb_image=rgb, floating_map=floating_prob, suspended_map=suspended_prob,
                          plastic_map=np.zeros_like(floating_prob), class_map=combined_map,
                          df=df, stats=stats, output_png=preview_png)
        
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        if handler: handler.flush(); LOGGER.removeHandler(handler); handler.close()
        return df

    if any(v is None for v in (sample_training_patches, FloatingDataset, train_unet_from_patches, infer_unet)):
        raise RuntimeError("train_model utilities are unavailable; reinstall sateline_ml package")
    if torch is None:
        raise RuntimeError("Torch should be available after _require_torch() call")

    if resolve_device is not None:
        selected_device_obj = resolve_device(device)
    else:
        selected_device_obj = torch.device("cpu")
        requested_obj: Optional[torch.device] = None
        if device is not None:
            try: requested_obj = torch.device(device)
            except Exception: requested_obj = None
            if requested_obj is not None:
                if requested_obj.type == "cuda" and torch.cuda.is_available():
                    selected_device_obj = requested_obj
                elif requested_obj.type == "mps":
                    try:
                        if getattr(torch.backends, "mps").is_available():
                            selected_device_obj = requested_obj
                    except Exception:
                        selected_device_obj = torch.device("cpu")
                elif requested_obj.type == "cpu":
                    selected_device_obj = requested_obj
        if device is None:
            if torch.cuda.is_available():
                selected_device_obj = torch.device("cuda")
            else:
                try:
                    if getattr(torch.backends, "mps").is_available():
                        selected_device_obj = torch.device("mps")
                except Exception:
                    selected_device_obj = torch.device("cpu")

    train_device_arg = str(selected_device_obj)
    inference_device_arg = str(selected_device_obj)
    if device is not None:
        try:
            requested_type = torch.device(device).type
        except Exception:
            LOGGER.warning("Invalid device %s; using %s", device, train_device_arg)
        else:
            if requested_type != selected_device_obj.type:
                LOGGER.warning("Requested device %s unavailable; using %s", device, train_device_arg)
    else:
        if selected_device_obj.type != "cpu":
            LOGGER.info("Auto-selected %s device for UNet", train_device_arg)
    if isinstance(input_tif, (list, tuple)):  # 或者你定义一个新参数 sequences
    # 1) 先把多期影像读成列表，再做与你上面一致的特征工程，得到每期 features_t: (H,W,C)
        features_seq = np.stack(list_of_features_per_time, axis=0)     # (T,H,W,C)

        # 2) 伪标签：通常只监督末帧；也可以给 (T,2,H,W)
        masks_seq = label_stack[None, ...]      # (1,2,H,W) —— 只监督末帧
        valid_seq = water_mask[None, ...]       # (1,H,W)

        ds_temp = TemporalDataset([features_seq], [masks_seq], [valid_seq], return_latest_only=True)

        from .train_model import train_temporal_from_sequences, infer_temporal_latest
        model_t, hist_t, dev_t = train_temporal_from_sequences(
            ds_temp, epochs=epochs, cell_type="gru", hidden_channels=64,
            latest_only=True, return_sequence=False, device=device,
        )
        probs = infer_temporal_latest(model_t, features_seq, device=dev_t)  # (2,H,W)
        floating_prob, suspended_prob = probs[0], probs[1] if probs.shape[0] > 1 else np.zeros_like(probs[0])

    else:
        images_list, masks_list, valid_list = sample_training_patches(
            features, label_stack, water_mask.astype(np.float32),
            tile=tile_size, stride=stride, min_valid_frac=min_valid_frac,
            max_patches=max_patches, pos_frac=pos_frac, seed=patch_seed,
        )
        if len(images_list) == 0:
            warn_msg = "No valid training patches (lower min_valid_frac or tile)."
            LOGGER.warning(warn_msg); tqdm.write(warn_msg)
            floating_prob = float_seed.astype(np.float32)
            suspended_prob = susp_seed.astype(np.float32)
            combined_map = np.zeros_like(floating_prob, dtype=np.uint8)
            df = pd.DataFrame(columns=[
                "row","col","x","y","lon","lat","class","probability","NDVI","FDI","NDTI","SABI","A_B11","A_B12","PlasticProb","PlasticFlag",
            ])
            stats = {
                "floating_debris": compute_stats("floating_debris", float_seed),
                "suspended_matter": compute_stats("suspended_matter", susp_seed),
            }
            rgb = np.dstack([
                np.clip(B4 / np.nanmax(B4, where=water_mask, initial=1), 0, 1),
                np.clip(B3 / np.nanmax(B3, where=water_mask, initial=1), 0, 1),
                np.clip(B2 / np.nanmax(B2, where=water_mask, initial=1), 0, 1),
            ])
            visualize_results(rgb_image=rgb, floating_map=floating_prob, suspended_map=suspended_prob,
                            plastic_map=np.zeros_like(floating_prob), class_map=combined_map,
                            df=df, stats=stats, output_png=preview_png)
            _save_debug_hist(debug_dir, "floating_prob_hist.png", floating_prob, title="Floating probability histogram")
            _save_debug_hist(debug_dir, "suspended_prob_hist.png", suspended_prob, title="Suspended probability histogram")
            _save_debug_image(debug_dir, "floating_prob_map.png", floating_prob, title="Floating probability map", cmap="magma")
            _save_debug_image(debug_dir, "suspended_prob_map.png", suspended_prob, title="Suspended probability map", cmap="viridis")
            df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            if handler: handler.flush(); LOGGER.removeHandler(handler); handler.close()
            return df

        dataset = FloatingDataset(images_list, masks_list, valid_list)
        print(f"[INFO] training patches: total={len(dataset)}")

        scene_id = Path(input_tif).stem
        model_root = Path(model_out_dir) if model_out_dir is not None else Path.cwd()
        per_scene_dir = model_root / "models" / scene_id

        if repeats and repeats > 1 and train_unet_repeated is not None:
            save_dir = per_scene_dir
            model, _, device_obj, best_val, best_path = train_unet_repeated(
                dataset, repeats=repeats, val_split=val_split, seed=patch_seed,
                epochs=epochs, lr=1e-3, device=train_device_arg, batch_size=1, save_best_to=save_dir,
            )
            LOGGER.info("Selected best model with val_loss=%.6f; saved to %s", best_val, save_dir)
        else:
            model, _, device_obj = train_unet_from_patches(
                dataset, epochs=epochs, lr=1e-3, device=train_device_arg, batch_size=1, pos_weight=None,
            )
            best_val = float("nan"); best_path = per_scene_dir / "best_unet.pth"
            try:
                per_scene_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "meta": {"repeats": 1, "epochs": epochs}}, best_path)
            except Exception as exc:
                LOGGER.warning("Could not save single-run weights: %s", exc)

        thresholds_path = Path(output_csv).parent / "thresholds.json"
        calibration_result: Optional[ThresholdCalibrationResult] = None
        if (auto_calibrate_thresholds and calibrate_probability_thresholds is not None and len(dataset) > 0):
            try:
                candidate_array = (np.linspace(0.2, 0.95, 16, dtype=np.float32)
                                if auto_threshold_candidates is None
                                else np.asarray(list(auto_threshold_candidates), dtype=np.float32))
                calibration_result = calibrate_probability_thresholds(
                    model=model, dataset=dataset, device=device_obj,
                    class_names=("floating_debris", "suspended_matter"),
                    candidate_thresholds=candidate_array, metric=auto_threshold_metric,
                    batch_size=auto_threshold_batch_size, max_samples=auto_threshold_max_samples, seed=patch_seed,
                )
                floating_thresh = float(calibration_result.thresholds.get("floating_debris", floating_thresh))
                suspended_thresh = float(calibration_result.thresholds.get("suspended_matter", suspended_thresh))
                save_calibration_result(calibration_result, thresholds_path)
                LOGGER.info("Auto-calibrated thresholds (%s): floating=%.3f, suspended=%.3f",
                            calibration_result.metric_name, floating_thresh, suspended_thresh)
            except Exception as calib_exc:
                calibration_result = None
                LOGGER.warning("Threshold calibration failed: %s", calib_exc)

        if calibration_result is None:
            try:
                thresholds_path.parent.mkdir(parents=True, exist_ok=True)
                baseline_payload = {
                    "metric": "configured",
                    "thresholds": {"floating_debris": float(floating_thresh), "suspended_matter": float(suspended_thresh)},
                    "per_class_metrics": {}, "candidate_thresholds": [], "class_order": ["floating_debris","suspended_matter"],
                }
                thresholds_path.write_text(json.dumps(baseline_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as exc:
                LOGGER.warning("Failed to write baseline threshold summary: %s", exc)

        # 更新 registry & best_overall（与你原逻辑一致）
        try:
            registry_path = model_root / "model_registry.csv"
            record = {
                "scene": scene_id, "weights_path": str(best_path),
                "best_val_loss": float(best_val) if best_val == best_val else None,
                "epochs": int(epochs), "repeats": int(repeats), "val_split": float(val_split),
                "device_train": str(train_device_arg), "tile_size": int(tile_size), "stride": int(stride),
                "max_patches": int(max_patches), "min_valid_frac": float(min_valid_frac), "pos_frac": float(pos_frac),
                "patch_seed": int(patch_seed), "floating_thresh": float(floating_thresh), "suspended_thresh": float(suspended_thresh),
                "auto_calibrated": bool(calibration_result is not None),
                "threshold_metric": (calibration_result.metric_name if calibration_result else (auto_threshold_metric if auto_calibrate_thresholds else None)),
                "stacking_preset": stacking_preset,
            }
            row_df = pd.DataFrame([record])
            if registry_path.exists():
                prev = pd.read_csv(registry_path); combined = pd.concat([prev, row_df], ignore_index=True)
            else:
                combined = row_df
            combined.to_csv(registry_path, index=False, encoding="utf-8-sig")

            valid_rows = combined.dropna(subset=["best_val_loss"]) if "best_val_loss" in combined.columns else pd.DataFrame()
            if not valid_rows.empty:
                idx = valid_rows["best_val_loss"].astype(float).idxmin()
                best_row = valid_rows.loc[idx]
                src_path = Path(str(best_row["weights_path"]))
                if src_path.exists():
                    global_best = model_root / "best_overall.pth"
                    try:
                        import shutil as _shutil
                        _shutil.copy2(src_path, global_best)
                        LOGGER.info("Updated global best => %s (val_loss=%.6f)", global_best, float(best_row["best_val_loss"]))
                    except Exception as cp_exc:
                        LOGGER.warning("Failed to update global best: %s", cp_exc)
        except Exception as reg_exc:
            LOGGER.warning("Model registry update failed: %s", reg_exc)

        # ===== Inference device selection: prefer CUDA -> MPS -> CPU =====
        preferred = inference_device_arg if isinstance(inference_device_arg, str) else train_device_arg
        td = torch.device(_pick_best_device(preferred))
        model = model.to(td)
        inference_device = torch.device(td)
        LOGGER.info("Starting tiled inference on %s", inference_device.type)

        # ===== Memory-aware tiled inference with graceful fallback =====
        try:
            if inference_device.type in ("cuda", "mps"):
                tile_candidates = (1024, 896, 768, 640, 512, 384, 320) if inference_device.type == "cuda" else (768, 640, 512, 448, 384, 320)
                for t in tile_candidates:
                    try:
                        probs = infer_unet_tiled(model, features, device=inference_device, tile=t, overlap=64, use_amp=True, show_progress=True)
                        break
                    except RuntimeError as oom:
                        msg = str(oom).lower()
                        if "out of memory" in msg or "mps" in msg or "resource" in msg:
                            try:
                                if inference_device.type == "cuda": torch.cuda.empty_cache()
                                elif inference_device.type == "mps" and hasattr(torch, "mps"): torch.mps.empty_cache()
                            except Exception:
                                pass
                            continue
                        raise
                else:
                    LOGGER.warning("GPU/MPS OOM for all tile sizes; fallback to CPU")
                    cpu_device = torch.device("cpu")
                    model = model.to(cpu_device)
                    probs = infer_unet_tiled(model, features, device=cpu_device, tile=768, overlap=64, use_amp=False, show_progress=True)
            else:
                probs = infer_unet_tiled(model, features, device=inference_device, tile=768, overlap=64, use_amp=False, show_progress=True)
        except Exception as exc:
            LOGGER.warning("Tiled inference failed (%s); retrying full-image on CPU", exc)
            cpu_device = torch.device("cpu")
            model = model.to(cpu_device)
            probs = infer_unet(model, features, device=cpu_device)

        LOGGER.info("Finished inference")
        floating_prob = probs[0]
        suspended_prob = probs[1] if probs.shape[0] > 1 else np.zeros_like(floating_prob)
    floating_prob[~water_mask] = np.nan
    suspended_prob[~water_mask] = np.nan

    # ============ stacking（保持原逻辑） ============
    stacking_applied = False
    if stacking_preset:
        if run_stacking_pipeline is None or build_tabular_feature_stack is None:
            LOGGER.warning("Stacking utilities unavailable; skipping preset %s", stacking_preset)
        else:
            try:
                stack_stride = max(1, int(stacking_pixel_stride))
                base_mask_flat = np.flatnonzero(water_mask.reshape(-1))
                if stack_stride > 1:
                    rows = base_mask_flat // water_mask.shape[1]
                    cols = base_mask_flat %  water_mask.shape[1]
                    stride_keep = (rows % stack_stride == 0) & (cols % stack_stride == 0)
                    mask_flat = base_mask_flat[stride_keep]
                else:
                    mask_flat = base_mask_flat
                if stacking_max_pixels is not None and mask_flat.size > stacking_max_pixels:
                    before_cap = mask_flat.size
                    rng_limit = np.random.default_rng(patch_seed + 1)
                    sel = np.sort(rng_limit.choice(mask_flat.size, size=int(stacking_max_pixels), replace=False))
                    mask_flat = mask_flat[sel]
                    LOGGER.info("Stacking pixel cap: %d -> %d (max=%d)", before_cap, mask_flat.size, stacking_max_pixels)

                LOGGER.info("Stacking candidate pixels: initial=%d, after stride=%d => %d",
                            base_mask_flat.size, stack_stride, mask_flat.size)

                tabular_X, tab_feature_names, stack_indices = build_tabular_feature_stack(
                    band_cube=band_cube, derived_indices=derived_indices_map,
                    band_names=band_order, water_mask=water_mask,
                    kernel_sizes=(3,5), include_texture=True, flat_indices=mask_flat,
                )
                if tabular_X.shape[0] == 0 or stack_indices.size == 0:
                    LOGGER.warning("Stacking preset %s skipped: no valid pixels after subsampling.", stacking_preset)
                    stacking_applied = False
                    stack_json = output_csv.parent / "stacking.json"
                    if stack_json.exists():
                        try: stack_json.unlink()
                        except Exception: pass
                else:
                    labels_map = {
                        "floating_debris": float_seed.reshape(-1)[stack_indices].astype(np.int8),
                        "suspended_matter": susp_seed.reshape(-1)[stack_indices].astype(np.int8),
                    }
                    precomputed_meta = {
                        "unet_floating": floating_prob.reshape(-1)[stack_indices],
                        "unet_suspended": suspended_prob.reshape(-1)[stack_indices],
                    }
                    if stacking_sample_limit is not None and tabular_X.shape[0] > stacking_sample_limit:
                        rng = np.random.default_rng(patch_seed)
                        train_idx = np.sort(rng.choice(tabular_X.shape[0], size=int(stacking_sample_limit), replace=False))
                    else:
                        train_idx = np.arange(tabular_X.shape[0])

                    fused_maps = run_stacking_pipeline(
                        stacking_preset, tabular_X=tabular_X,
                        labels=labels_map, probabilities={},
                        precomputed_meta=precomputed_meta, mask_flat=train_idx,
                        random_state=patch_seed, feature_names=tab_feature_names, cv_folds=stacking_cv_folds,
                    )

                    preset_cfg = STACKING_PRESETS.get(stacking_preset.lower(), {}) if STACKING_PRESETS else {}
                    if postprocess_probabilities is not None and preset_cfg.get("postprocess"):
                        fused_maps = postprocess_probabilities(fused_maps, rgb=rgb_preview, postprocess_cfg=preset_cfg.get("postprocess"))

                    fused_float = floating_prob.reshape(-1)
                    fused_susp  = suspended_prob.reshape(-1)
                    fused_float[stack_indices] = fused_maps["floating_debris"]
                    fused_susp[stack_indices]  = fused_maps["suspended_matter"]
                    floating_prob = fused_float.reshape(water_mask.shape)
                    suspended_prob = fused_susp.reshape(water_mask.shape)
                    floating_prob[~water_mask] = np.nan
                    suspended_prob[~water_mask] = np.nan
                    stacking_applied = True
                    LOGGER.info("Stacking '%s' applied with %d features (stride=%d, pixels=%d)",
                                stacking_preset, len(tab_feature_names), stack_stride, stack_indices.size)
                    try:
                        stacking_meta = {
                            "preset": stacking_preset,
                            "feature_count": int(len(tab_feature_names)),
                            "sample_count": int(tabular_X.shape[0]),
                            "train_sample_count": int(train_idx.size),
                            "stacking_sample_limit": int(stacking_sample_limit) if stacking_sample_limit else None,
                            "stacking_max_pixels": int(stacking_max_pixels) if stacking_max_pixels else None,
                            "pixel_stride": int(stack_stride),
                            "pixel_count_initial": int(base_mask_flat.size),
                            "pixel_count_after_stride": int(mask_flat.size),
                            "pixel_count_after_mask": int(stack_indices.size),
                            "cv_folds": int(stacking_cv_folds),
                            "stacking_applied": True,
                        }
                        stack_json = output_csv.parent / "stacking.json"
                        stack_json.write_text(json.dumps(stacking_meta, ensure_ascii=False, indent=2), encoding="utf-8")
                    except Exception as meta_exc:
                        LOGGER.warning("Failed to write stacking metadata: %s", meta_exc)
            except Exception as stack_exc:
                LOGGER.warning("Stacking '%s' failed: %s", stacking_preset, stack_exc)
                stacking_applied = False
                stack_json = output_csv.parent / "stacking.json"
                if stack_json.exists():
                    try: stack_json.unlink()
                    except Exception: pass

    # ----------------- 概率→互斥类别（加入不确定性边际） -----------------
    
    plastic_prob_map: Optional[np.ndarray] = None
    plastic_mask_pred: Optional[np.ndarray] = None
    plastic_type_map: Optional[np.ndarray] = None

    # ====== 稀疏“塑料识别”分支（与你原逻辑保持一致，仅略清理打印） ======
    if plastic_enabled:
        try:
            rng = np.random.default_rng(1234)
            cand_pos_mask = (floating_prob > max(floating_thresh, 0.6)) |(suspended_prob > max(suspended_thresh, 0.6))& water_mask
            cand_neg_mask = (water_mask & (~cand_pos_mask))
            max_pos, max_neg = 120_000, 180_000
            pos_idx = np.where(cand_pos_mask.ravel())[0]
            neg_idx = np.where(cand_neg_mask.ravel())[0]
            if pos_idx.size > max_pos: pos_idx = rng.choice(pos_idx, size=max_pos, replace=False)
            if neg_idx.size > max_neg: neg_idx = rng.choice(neg_idx, size=max_neg, replace=False)
            sel_idx = np.concatenate([pos_idx, neg_idx])
            if sel_idx.size == 0:
                raise ValueError("No candidate pixels for plastic training.")
            H, W = water_mask.shape
            sel_rows = (sel_idx // W).astype(np.int64); sel_cols = (sel_idx % W).astype(np.int64)

            X_plastic, feat_names, meta = build_plastic_feature_matrix_sparse(
                band_cube, band_order, plastic_library_csv,
                sel_rows, sel_cols, background_mask=water_mask , bg_sample=200_000,
            )
            y_plastic = np.zeros(sel_idx.size, dtype=np.float32); y_plastic[:len(pos_idx)] = 1.0

            model_plastic, _ = train_plastic_model(
                X_plastic, y_plastic, epochs=400, lr=1e-3, weight_decay=5e-5,
                device=str(inference_device), verbose=False, hidden=(256, 128), dropout=0.25,
            )
            if debug_dir is not None:
                _save_plastic_feature_correlations(debug_dir, feat_names, X_plastic, y_plastic)
                _save_plastic_spectral_profiles(debug_dir, feat_names, X_plastic, y_plastic)

            if plastic_model_path is not None:
                meta_obj = PlasticModelMetadata(input_dim=X_plastic.shape[1], feature_names=tuple(feat_names), band_order=tuple(band_order))
                try:
                    from sklearn.metrics import accuracy_score, log_loss
                    _has_sklearn = True
                except Exception:
                    _has_sklearn = False
                probs_train = predict_plastic(model_plastic, X_plastic, device=str(inference_device)).astype(np.float32)
                y_true = y_plastic.astype(np.int32); y_pred = (probs_train >= float(plastic_prob_thresh)).astype(np.int32)
                if _has_sklearn:
                    acc  = float(accuracy_score(y_true, y_pred))
                    loss = float(log_loss(y_true, np.clip(probs_train, 1e-12, 1-1e-12), labels=[0,1]))
                else:
                    p = np.clip(probs_train, 1e-12, 1-1e-12)
                    acc = float((y_true == y_pred).mean())
                    loss = float(-np.mean(y_true*np.log(p) + (1-y_true)*np.log(1-p)))
                LOGGER.info("[Plastic-ML] Accuracy: %.4f | LogLoss: %.4f", acc, loss)
                try: save_plastic_model(plastic_model_path, model_plastic, meta_obj)
                except Exception as _exc: LOGGER.warning("Could not save plastic model: %s", _exc)

            pred_mask = cand_pos_mask
            pred_idx = np.where(pred_mask.ravel())[0]
            pred_rows = (pred_idx // W).astype(np.int64); pred_cols = (pred_idx % W).astype(np.int64)
            X_pred, _, meta_pred = build_plastic_feature_matrix_sparse(
                band_cube, band_order, plastic_library_csv, pred_rows, pred_cols, background_mask=water_mask, bg_sample=200_000,
            )
            prob_vec = predict_plastic(model_plastic, X_pred, device=str(inference_device))
            plastic_prob_map = np.zeros_like(floating_prob, dtype=np.float32)
            plastic_prob_map.ravel()[pred_idx] = prob_vec.astype(np.float32); plastic_prob_map[~water_mask] = np.nan
            

            try:
                endmembers = (meta_pred or {}).get("endmembers") if meta_pred is not None else None
                sensor_order = (meta_pred or {}).get("sensor_band_order") if meta_pred is not None else None
                if endmembers and sensor_order:
                    type_names = list(endmembers.keys())
                    em_matrix = np.stack([np.asarray(endmembers[name], dtype=np.float32) for name in type_names], axis=0)
                    n_bands = em_matrix.shape[1]
                    rho_part = X_pred[:, :n_bands].astype(np.float32)
                    denom_pix = np.linalg.norm(rho_part, axis=1, keepdims=True) + 1e-12
                    denom_em = np.linalg.norm(em_matrix, axis=1) + 1e-12
                    cos = (rho_part @ em_matrix.T) / (denom_pix * denom_em)
                    cos = np.clip(cos, -1.0, 1.0)
                    angles = np.arccos(cos)
                    best_idx = np.argmin(angles, axis=1)
                    type_labels = np.array(type_names, dtype=object)[best_idx]
                else:
                    type_labels = np.full(pred_idx.shape, None, dtype=object)
                plastic_type_map = np.full_like(floating_prob, None, dtype=object)
                plastic_type_map.reshape(-1)[pred_idx] = type_labels
            except Exception as type_exc:
                LOGGER.warning("Plastic type attribution failed: %s", type_exc)
                plastic_type_map = None

        except Exception as exc:
            LOGGER.warning("Plastic training/init failed: %s", exc)
            plastic_prob_map = None; plastic_mask_pred = None
    # ============================================
    # 不确定性边际
    confident = (np.abs(floating_prob - suspended_prob) >= float(unc_margin))
    floating_mask_pred  = (floating_prob  > floating_thresh)  & shp_mask & confident
    suspended_mask_pred = (suspended_prob > suspended_thresh) & shp_mask & confident
    if plastic_prob_map is not None:
        plastic_mask_pred = (plastic_prob_map > plastic_prob_thresh) & shp_mask & confident
    else:
        plastic_mask_pred = None
    # 胜者判定（严格 > 给悬浮，平手给漂浮的版本已在你原逻辑里给示例；这里沿用严格版）
    susp_wins  = suspended_mask_pred & (suspended_prob >  floating_prob)
    float_wins = floating_mask_pred  & (~suspended_mask_pred | (floating_prob >= suspended_prob))


    combined_map = build_combined_map(
            floating_prob, suspended_prob,
            water_mask=water_mask,
            shp_mask=shp_mask,
            confident=confident,
            floating_thresh=floating_thresh,
            suspended_thresh=suspended_thresh,
            plastic_mask_pred=plastic_mask_pred,  # 或 None
            tie_policy="float",  # 与你现有平手给漂浮一致
        )
    
    
    # ----------------- 构建 DataFrame（沿用你的向量化实现） -----------------
    df_float = _df_from_mask("floating_debris",  floating_mask_pred,  floating_prob,
                             ndvi, fdi, ndti, sabi, A_B11, A_B12, transform, src_crs, water_mask.shape)
    df_susp  = _df_from_mask("suspended_matter", suspended_mask_pred, suspended_prob,
                             ndvi, fdi, ndti, sabi, A_B11, A_B12, transform, src_crs, water_mask.shape)
    frames = [frame for frame in (df_float, df_susp) if not frame.empty]
    if frames:
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.DataFrame(columns=df_float.columns)

    # 回填 plastic 标记到“漂浮”行 + 另存 plastic_ml 行
    if plastic_mask_pred is not None and plastic_prob_map is not None and not df.empty:
        H, W = water_mask.shape
        plast_idx_flat = np.flatnonzero(plastic_mask_pred.reshape(-1))
        if "PlasticFlag" not in df.columns:
            df["PlasticFlag"] = False
        if "PlasticProb" not in df.columns:
            df["PlasticProb"] = np.nan
        if plastic_type_map is not None and "PlasticType" not in df.columns:
            df["PlasticType"] = None

        for type_mask in ("floating_debris", "suspended_matter"):
            # 当前这一类（漂浮 / 悬浮）在 df 里的行
            mask_df = (df["class"].values == type_mask)
            if not mask_df.any():
                continue

            # 取出它们的行列号
            rows = df.loc[mask_df, "row"].to_numpy(dtype=np.int64, copy=False)
            cols = df.loc[mask_df, "col"].to_numpy(dtype=np.int64, copy=False)
            # 安全裁到影像大小里
            np.clip(rows, 0, H - 1, out=rows)
            np.clip(cols, 0, W - 1, out=cols)

            # 这些点在影像上的一维索引
            flat_idx = np.ravel_multi_index((rows, cols), dims=(H, W), mode="clip")

            # 哪些点其实落在 plastic 掩膜上
            hit_local = np.isin(flat_idx, plast_idx_flat)          # shape = (这一类的个数,)
            idx_all = np.where(mask_df)[0]                         # df 里的真实行号（同样大小）

            # 不管命中与否，先把 plastic 概率全写回去（你说“要保留概率”）
            probs = plastic_prob_map[rows, cols].astype(np.float32)
            df.loc[idx_all, "PlasticProb"] = probs

            # 只对“既落在 plastic 掩膜上，又概率够高”的行打 True
            hit_global = idx_all[hit_local]                        # 命中的那几行在 df 里的索引
            if plastic_prob_thresh is not None:
                df.loc[hit_global, "PlasticFlag"] = (
                    probs[hit_local] >= plastic_prob_thresh
                )
            else:
                # 没设阈值就维持原逻辑：命中就 True
                df.loc[hit_global, "PlasticFlag"] = True

            # 类型也可以顺便写回去（写给这类的所有点，让你后面好分析）
            if plastic_type_map is not None:
                df.loc[idx_all, "PlasticType"] = np.asarray(
                    plastic_type_map[rows, cols], dtype=object
                )

    if "PlasticProb" in df.columns:
        df["PlasticProb"] = pd.to_numeric(df["PlasticProb"], errors="coerce").fillna(0.0).astype(np.float32)
    if "PlasticFlag" in df.columns:
        df["PlasticFlag"] = df["PlasticFlag"].astype("boolean").fillna(False)
    # ---- Save CSV & Preview ----
    LOGGER.info("Writing detections CSV to %s (rows=%d)", output_csv, len(df))
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    LOGGER.info("Saved CSV")

    stats = {}
        # "floating_nonplastic":  compute_stats("floating_nonplastic",  (combined_map==1)),
        # "suspended_nonplastic": compute_stats("suspended_nonplastic", (combined_map==2)),
    if plastic_mask_pred is not None:
        # stats["plastic_only"] =           compute_stats("plastic_only",           (combined_map==3))
        stats["plastic_and_floating"] =   compute_stats("plastic&floating",   (combined_map==4))
        stats["plastic_and_suspended"] =  compute_stats("plastic&suspended",  (combined_map==5))
    
    
    inter = water_mask & shp_mask
    r = np.clip(B4 / np.nanmax(B4, where=inter, initial=1), 0, 1)
    g = np.clip(B3 / np.nanmax(B3, where=inter, initial=1), 0, 1)
    b = np.clip(B2 / np.nanmax(B2, where=inter, initial=1), 0, 1)
    rgb = np.dstack([r, g, b])
    alph = np.where(inter, 1.0, 0.0)           # 交集=不透明，其他=透明
    rgba = np.dstack([rgb, alph.astype(float)])
    # 非交集处设为 NaN（或用掩膜数组）
    rgb_masked = np.where(inter[..., None], rgb, np.nan)

    for fa, sa, la, dda in zip(
        [floating_mask_pred, suspended_mask_pred, plastic_mask_pred],
        ["floating", "suspended", "plastic"],
        [floating_thresh, suspended_thresh, plastic_prob_thresh],
        [floating_prob, suspended_prob, plastic_prob_map],
    ):
        try:
            if fa is not None:
                _save_debug_overlay(
                    debug_dir,
                    f"{sa}_detections_overlay.png",
                    rgba,
                    fa,
                    title=f"{sa} detections",
                    color="orange",
                )
                _save_debug_image(
                    debug_dir,
                    f"{sa}_detections_prob_map.png",
                    fa,
                    title=f"{sa} probability (thr={la})",
                    cmap="inferno",
                    mask=inter,
                )
            if dda is not None:
                _save_debug_hist(
                    debug_dir,
                    f"{sa}_detections_prob_hist.png",
                    dda,
                    title=f"{sa} detections probability histogram (thr={la})",
                    mask=inter,
                )
        except Exception as e:
            LOGGER.warning("plot failed: %s", e)

    LOGGER.info("Rendering preview to %s", preview_png)
    visualize_results(
        rgb_image=rgba,
        floating_map=floating_prob,
        suspended_map=suspended_prob,
        plastic_map=plastic_prob_map,
        class_map=combined_map,
        df=df,
        stats=stats,
        mask=rgb_masked,
        output_png=preview_png,
    )
    LOGGER.info("Saved preview")

    if handler:
        handler.flush(); LOGGER.removeHandler(handler); handler.close()
    return df


def create_mask_tif_from_shp(
    shp_file: Path | str,
    reference_tif: Path | str,
    output_mask_tif: Path | str,
    filter_field: Optional[str] = None,
    filter_values: Optional[Sequence] = None,
    filter_query: Optional[str] = None,
    filter_indices: Optional[Sequence[int]] = None,
) -> Path:
    with rasterio.open(reference_tif) as src:
        meta = src.meta.copy()
        transform = src.transform
        out_shape = (src.height, src.width)
        target_crs = src.crs

    gdf = gpd.read_file(shp_file)
    gdf = filter_gdf(gdf, filter_field, filter_values, filter_query, filter_indices)
    gdf_original = gdf.copy()

    if target_crs is not None:
        try:
            gdf = gdf.to_crs(target_crs)
        except Exception as exc:
            LOGGER.warning("Failed to reproject shapefile to raster CRS %s: %s", target_crs, exc)

    mask = rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    if not np.any(mask):
        fallback_gdf, warning_msg = _fallback_project_shapefile(gdf_original, target_crs, transform)
        if fallback_gdf is not None:
            mask = rasterize(
                [(geom, 1) for geom in fallback_gdf.geometry],
                out_shape=out_shape,
                transform=transform,
                fill=0,
                dtype=np.uint8,
            )
            if np.any(mask):
                gdf = fallback_gdf
                LOGGER.warning(warning_msg)
        if not np.any(mask):
            raise ValueError(
                "Shapefile mask is empty after rasterization. The shapefile likely does not overlap the raster "
                "or the raster has incorrect CRS metadata."
            )

    meta.update(dtype=rasterio.uint8, count=1, compress="lzw")
    output_mask_tif = Path(output_mask_tif)
    with rasterio.open(output_mask_tif, "w", **meta) as dst:
        dst.write(mask, 1)

    return output_mask_tif


def batch_detect_directory(
    tif_dir: Path | str,
    shp_file: Path | str,
    output_dir: Path | str,
    *,
    mode: str = "quick",
    detection_kwargs: Optional[Dict] = None,
    **filters,
) -> Dict[str, pd.DataFrame]:
    tif_dir = Path(tif_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, pd.DataFrame] = {}
    allowed_filter_keys = {"filter_field", "filter_values", "filter_query", "filter_indices"}
    filter_args = {k: v for k, v in filters.items() if k in allowed_filter_keys and v is not None}
    detection_kwargs = detection_kwargs or {}

    for tif_path in sorted(tif_dir.glob("*.tif")):
        csv_path = output_dir / f"{tif_path.stem}.csv"
        png_path = output_dir / f"{tif_path.stem}.png"
        if mode == "advanced":
            mask_tif = detection_kwargs.get("mask_tif")
            if not mask_tif:
                raise ValueError("Advanced mode requires 'mask_tif' in detection_kwargs")
            per_scene_kwargs = {k: v for k, v in detection_kwargs.items() if k not in {"mask_tif", "log_dir", "debug_dir"}}
            log_dir = detection_kwargs.get("log_dir")
            if log_dir and "log_file" not in per_scene_kwargs:
                log_path = Path(log_dir) / f"{tif_path.stem}.log"
                per_scene_kwargs["log_file"] = log_path
            debug_root = detection_kwargs.get("debug_dir")
            if debug_root and "debug_dir" not in per_scene_kwargs:
                per_scene_kwargs["debug_dir"] = Path(debug_root) / tif_path.stem
            df = floating_debris_detection(
                input_tif=tif_path,
                mask_tif=mask_tif,
                shp_file=shp_file,
                output_csv=csv_path,
                preview_png=png_path,
                **filter_args,
                **per_scene_kwargs,
            )
        results[tif_path.name] = df
    return results


__all__ = [
    "DetectionStats",
    "filter_gdf",
    "resize_to_reference",
    "read_bands",
    "compute_ndwi",
    "compute_fdi",
    "compute_ndti",
    "compute_sabi",
    "mask_to_vector",
    "visualize_results",
    "compute_stats",
    # "plot_sateline",
    "floating_debris_detection",
    "create_mask_tif_from_shp",
    "batch_detect_directory",
    "build_plastic_feature_matrix_sparse",
]
