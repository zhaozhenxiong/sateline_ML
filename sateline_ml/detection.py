# -*- coding: utf-8 -*-
"""Floating debris detection utilities — optimized & hardened."""

from __future__ import annotations
from .train_model import (
    TemporalDataset,
    train_temporal_from_sequences,
    infer_temporal_latest,   # 如需全序列输出可改 infer_temporal_sequence
    # 单期分支 + 塑料分支（可选）
    FloatingDataset,
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
    # plastic MLP
    train_plastic_model,
    save_plastic_model,
    PlasticModelMetadata,
)
from .water_filter import (
    robust_thresholds_water,
    hysteresis_water_mask,
    erode_interior,
    region_filter,
    neighborhood_consistency,
)
from .spectral_tools import S2_BANDS, band_depth

# ---- Matplotlib: force non-interactive backend BEFORE importing pyplot ----
import matplotlib
matplotlib.use("Agg")  # avoid GUI/blocking, faster in headless/multiprocessing

from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import json
import logging
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize, shapes
from rasterio.warp import transform as rio_transform
from shapely.geometry import Point, shape as shp_shape
from tqdm import tqdm

# ---- Local imports (optional availability is tolerated) ----

try:
    import torch  # type: ignore
    _TORCH_IMPORT_ERROR = None
except Exception as _exc:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_IMPORT_ERROR = _exc

# 可选 stacking
try:
    from .stacking import (
        STACKING_PRESETS,
        build_tabular_feature_stack,
        postprocess_probabilities,
        run_stacking_pipeline,
    )
except Exception:  # pragma: no cover
    STACKING_PRESETS = {}
    build_tabular_feature_stack = None  # type: ignore
    postprocess_probabilities = None  # type: ignore
    run_stacking_pipeline = None  # type: ignore

LOGGER = logging.getLogger(__name__)
LOGGER.propagate = False


# -------------------- Dataclasses --------------------
@dataclass
class DetectionStats:
    label: str
    area_m2: float
    pixel_count: int


# -------------------- Small utils --------------------
def _require_torch() -> None:
    if _TORCH_IMPORT_ERROR is not None or torch is None:
        raise ImportError(
            "Torch is required for UNet-related operations. Install torch before calling this function."
        ) from _TORCH_IMPORT_ERROR


def _ensure_stream_logger() -> None:
    if any(isinstance(h, logging.StreamHandler) for h in LOGGER.handlers):
        return
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"))
    h.setLevel(logging.INFO)
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(h)


def _configure_logger(log_file: Optional[Path | str]) -> Optional[logging.Handler]:
    if not log_file:
        return None
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"))
    handler.setLevel(logging.INFO)
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(handler)
    return handler


def _with_scene_logging(function):
    """Release a scene's log handler after success, early return, or failure."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        _ensure_stream_logger()
        handler = _configure_logger(kwargs.get("log_file"))
        try:
            if handler:
                LOGGER.info("Logging to %s", handler.baseFilename)
            return function(*args, **kwargs)
        finally:
            if handler:
                LOGGER.removeHandler(handler)
                handler.close()

    return wrapped


def _load_compatible_plastic_model(
    path: Optional[Path | str],
    *,
    input_dim: int,
    feature_names: Sequence[str],
    band_order: Sequence[str],
    device: str,
):
    """Load an existing classifier without modifying it or changing feature order."""
    if path is None or not Path(path).is_file():
        raise ValueError("Plastic inference requires an existing plastic_model_path checkpoint file")
    model, metadata = load_plastic_model(path, device=device)
    if metadata.input_dim != input_dim:
        raise ValueError(f"Plastic checkpoint input_dim is {metadata.input_dim}; scene features require {input_dim}")
    if tuple(metadata.feature_names) != tuple(feature_names):
        raise ValueError("Plastic checkpoint feature_names do not match this scene and spectral library")
    if tuple(metadata.band_order) != tuple(band_order):
        raise ValueError("Plastic checkpoint band_order does not match this scene")
    return model


def _pixel_area_from_transform(transform) -> float:
    # area of pixel parallelogram in projected CRS
    try:
        return float(abs(transform.a * transform.e - transform.b * transform.d))
    except Exception:
        return 1.0  # safe fallback


def _robust_scale(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Median/IQR scaling within mask; NaNs->0."""
    v = arr[mask]
    v = v[np.isfinite(v)]
    if v.size < 100:
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    med = np.median(v)
    q75, q25 = np.percentile(v, [75, 25])
    iqr = q75 - q25
    scale = iqr if iqr > 1e-6 else (np.std(v) + 1e-6)
    out = (arr - med) / scale
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _resolve_seed_target(
    target: int | float,
    ratio: Optional[float],
    floor: Optional[int],
    water_pixels: int,
    name: str,
) -> int:
    """Normalize seed size target to an int."""
    if isinstance(target, (int, np.integer)) and target > 0:
        v = int(target)
    elif isinstance(target, (float, np.floating)) and target > 1:
        v = int(target)
    elif ratio is not None and ratio > 0:
        v = int(max(1, ratio * water_pixels))
    else:
        v = min(20000, max(1000, water_pixels // 20))
    if floor is not None:
        v = max(int(floor), v)
    LOGGER.info("[seed] %s target -> %d", name, v)
    return v


def _collect_stat_values(label: str, arr: np.ndarray, mask: np.ndarray, fallback_mask: np.ndarray) -> np.ndarray:
    v = arr[mask & np.isfinite(arr)]
    if v.size == 0:
        v = arr[fallback_mask & np.isfinite(arr)]
    if v.size == 0:
        LOGGER.warning(
            "No valid values for %s; falling back to whole array.", label)
        v = arr[np.isfinite(arr)]
    return v.ravel()


def _percentile_safe(values: np.ndarray, q: float, default: float = 0.0) -> float:
    """Return percentile with NaN-safe fallback."""
    if values.size == 0:
        return float(default)
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return float(default)
    return float(np.percentile(vals, q))


# -------------------- basic indices --------------------
def read_bands(file_path: Path | str, band_indices: Sequence[int]):
    with rasterio.open(file_path) as src:
        bands = [src.read(i) for i in tqdm(band_indices, desc="Reading bands")]
        return np.stack(bands, axis=-1), src.transform, src.meta


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (green - nir) / (green + nir + 1e-6)


def compute_fdi(red: np.ndarray, nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    lam_red, lam_nir, lam_swir = 665.0, 842.0, 1610.0
    baseline = red + (swir - red) * (lam_nir - lam_red) / (lam_swir - lam_red)
    return nir - baseline


def compute_ndti(red: np.ndarray, green: np.ndarray) -> np.ndarray:
    return (red - green) / (red + green + 1e-6)


def compute_sabi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    return (nir - red) / (nir + red + 1e-6)


# -------------------- CRS helpers --------------------
def _ensure_latlon(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError(
            "Shapefile CRS is undefined; cannot align with raster.")
    try:
        if gdf.crs.to_epsg() == 4326:
            return gdf
    except Exception:
        pass
    return gdf.to_crs("EPSG:4326")


def _guess_utm_epsg(gdf: gpd.GeoDataFrame) -> Optional[int]:
    g = _ensure_latlon(gdf)
    c = g.geometry.unary_union.centroid
    if c.is_empty or np.isnan(c.x) or np.isnan(c.y):
        return None
    zone = int((c.x + 180.0) // 6) + 1
    zone = max(1, min(zone, 60))
    return (32600 if c.y >= 0 else 32700) + zone


def _fallback_project_shapefile(
    gdf_source: Optional[gpd.GeoDataFrame],
    raster_crs,
    transform,
) -> tuple[Optional[gpd.GeoDataFrame], Optional[str]]:
    if gdf_source is None or gdf_source.empty or transform is None or raster_crs is None:
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
    except Exception as exc:  # pragma: no cover
        LOGGER.error("Failed to project shapefile to EPSG:%s: %s", epsg, exc)
        return None, None
    message = (
        "Shapefile mask was empty after rasterize because the raster reports a geographic CRS (%s) "
        "with large pixel size %.2f. Falling back to EPSG:%s for masking. "
        "Please fix the raster CRS metadata."
    ) % (raster_crs, pixel_size, epsg)
    return gdf_fallback, message


def _maybe_override_bad_geographic_crs(
    raster_crs,
    transform,
    gdf_reference: Optional[gpd.GeoDataFrame],
) -> tuple[Optional[CRS], Optional[str]]:
    if raster_crs is None or transform is None or gdf_reference is None:
        return raster_crs, None
    try:
        crs_obj = raster_crs if isinstance(
            raster_crs, CRS) else CRS.from_user_input(raster_crs)
    except Exception:
        return raster_crs, None
    try:
        pixel_size = max(abs(transform.a), abs(transform.e))
    except Exception:
        return crs_obj, None
    if not getattr(crs_obj, "is_geographic", False) or pixel_size <= 1.0:
        return crs_obj, None
    epsg = _guess_utm_epsg(gdf_reference)
    if epsg is None:
        return crs_obj, None
    corrected = CRS.from_epsg(epsg)
    if corrected == crs_obj:
        return corrected, None
    message = (
        "Raster reports geographic CRS %s but pixel size is %.2f°. "
        "Override to EPSG:%s derived from AOI centroid; update the GeoTIFF metadata to avoid this fix."
    ) % (crs_obj, pixel_size, epsg)
    return corrected, message


# -------------------- vectorization, debug I/O --------------------
def mask_to_vector(
    mask: np.ndarray,
    transform,
    to_points: bool = False,
) -> gpd.GeoDataFrame:
    """Vectorize a binary mask into polygons (default) or centroids."""
    geoms = []
    for geom, val in shapes(mask.astype(np.uint8), mask=mask.astype(np.uint8), transform=transform):
        if val != 1:
            continue
        poly = shp_shape(geom)
        geoms.append(poly.centroid if to_points else poly)
    return gpd.GeoDataFrame(geometry=geoms, crs=None)


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
    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    data = np.array(array, copy=True)
    if mask is not None:
        m = np.asarray(mask)
        try:
            m = np.broadcast_to(m, data.shape, subok=True)
        except Exception:
            m = np.ones_like(data, dtype=bool)
        if m.dtype != bool:
            m = (np.isfinite(m) & (m != 0)) if np.issubdtype(
                m.dtype, np.floating) else m.astype(bool)
        data = np.where(m, data, np.nan)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, shrink=0.8)
    if title:
        ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(Path(debug_dir) / filename, dpi=200)
    plt.close(fig)


def _save_debug_overlay(
    debug_dir: Optional[Path | str],
    filename: str,
    rgb_or_mask: Optional[np.ndarray],
    overlay: Optional[np.ndarray],
    *,
    title: str,
    color: str,
    alpha: float = 0.4,
) -> None:
    if not debug_dir or overlay is None:
        return
    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    if rgb_or_mask is not None:
        ax.imshow(rgb_or_mask, interpolation="nearest")
    ol = overlay.astype(bool)
    ax.imshow(np.where(ol, 1.0, np.nan),
              cmap=ListedColormap([color]), alpha=alpha)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(Path(debug_dir) / filename, dpi=200)
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
    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    arr = np.asarray(data, dtype=np.float32)
    val = arr.ravel()
    if mask is not None:
        m = np.asarray(mask)
        try:
            m = np.broadcast_to(m, arr.shape, subok=True)
        except Exception:
            m = np.ones_like(arr, dtype=bool)
        if m.dtype != bool:
            m = (np.isfinite(m) & (m != 0)) if np.issubdtype(
                m.dtype, np.floating) else m.astype(bool)
        val = val[m.ravel()[: val.size]]
    val = val[np.isfinite(val)]
    if val.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(val, bins=bins)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(Path(debug_dir) / filename, dpi=200)
    plt.close(fig)


def _sample_rows(total: int, limit: int = 50_000, seed: int = 0) -> np.ndarray:
    """Return sorted random row indices with optional downsampling."""
    if total <= limit:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    sel = rng.choice(total, size=int(limit), replace=False)
    sel.sort()
    return sel.astype(np.int64, copy=False)


def _save_plastic_feature_correlations_pre(
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
        fig.savefig(debug_path / "plastic_feature_correlation-pre.png",
                    dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        LOGGER.warning(
            "Failed to write plastic feature correlation plot: %s", exc)


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
        raise ValueError(
            "Invalid percentile bounds for spectral profile statistics")
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
            counts[fallback] = np.sum(np.isfinite(
                samples[:, fallback]), axis=0).astype(np.float32)
    counts[counts <= 0] = 1.0
    stderr = std / np.sqrt(counts)
    ci95 = 1.96 * stderr
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    ci95 = np.nan_to_num(ci95, nan=0.0, posinf=0.0, neginf=0.0)
    return mean.astype(np.float32, copy=False), ci95.astype(np.float32, copy=False)


def _save_plastic_spectral_profiles_pre(
    debug_dir: Optional[Path | str],
    feat_names: Sequence[str],
    features: np.ndarray,
    labels: np.ndarray,
    *,
    sample_limit: int = 50000,
) -> None:
    if not debug_dir or features.size == 0:
        return
    rho_indices = [(idx, name[4:]) for idx, name in enumerate(
        feat_names) if name.startswith("rho_")]
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
        pos_mean, pos_ci = _robust_profile_stats(
            pos_spec, lower_pct=5.0, upper_pct=95.0)
        neg_mean, neg_ci = _robust_profile_stats(
            neg_spec, lower_pct=5.0, upper_pct=95.0)

        fig, (ax_main, ax_delta) = plt.subplots(
            2,
            1,
            figsize=(9, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        )
        ax_main.plot(wavelengths, pos_mean,
                     label="Plastic samples", color="#d62728")
        ax_main.fill_between(
            wavelengths,
            pos_mean - pos_ci,
            pos_mean + pos_ci,
            color="#d62728",
            alpha=0.25,
        )
        ax_main.plot(wavelengths, neg_mean,
                     label="Background samples", color="#1f77b4")
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
        ax_delta.axhline(0.0, color="#444444", linewidth=1.0,
                         linestyle="--", alpha=0.6)
        ax_delta.plot(wavelengths, delta, color="#2ca02c", linewidth=1.6)
        ax_delta.fill_between(wavelengths, 0.0, delta,
                              where=delta >= 0.0, color="#2ca02c", alpha=0.15)
        ax_delta.fill_between(wavelengths, 0.0, delta,
                              where=delta < 0.0, color="#d62728", alpha=0.08)
        ax_delta.set_xlabel("Wavelength (nm or index)")
        ax_delta.set_ylabel("Δ reflectance")
        ax_delta.grid(alpha=0.2)

        fig.tight_layout()
        debug_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(debug_path / "plastic_spectral_profiles—pre.png",
                    dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        LOGGER.warning(
            "Failed to write plastic spectral profile plot: %s", exc)

# --- Plastic analytics: correlations & spectral profiles --------------------


def _save_plastic_feature_correlations(
    debug_dir: Optional[Path | str],
    X,
    feat_names: Sequence[str],
    filename: str = "plastic_feature_correlations.png",
    *,
    method: str = "pearson",          # "pearson" | "spearman"
    sample_row_limit: int = 12000,    # 行数太大时随机抽样
    annotate: bool = False,           # 小特征数时绘制数值
    vmax: float = 1.0,
) -> Optional[Path]:
    """
    保存特征相关性热图（支持稀疏矩阵，自动抽样以节省内存）。
    """
    if not debug_dir:
        return None
    Path(debug_dir).mkdir(parents=True, exist_ok=True)

    # 取 ndarray（必要时抽样）—— 支持 scipy.sparse
    try:
        n_rows, n_cols = X.shape
    except Exception:
        return None

    rng = np.random.default_rng(202311)
    take_idx = np.arange(n_rows)
    if n_rows > sample_row_limit:
        take_idx = np.sort(rng.choice(
            n_rows, size=sample_row_limit, replace=False))

    # 稀疏 -> 稠密（抽样后再 densify）
    try:
        is_sparse = hasattr(X, "tocsr")
    except Exception:
        is_sparse = False

    if is_sparse:
        Xs = X[take_idx]
        Xd = Xs.toarray()
    else:
        Xd = np.asarray(X[take_idx], dtype=np.float32)

    # 标准化（列向 z-score），防止尺度差异影响
    Xd = np.nan_to_num((Xd - Xd.mean(0, keepdims=True)) /
                       (Xd.std(0, keepdims=True) + 1e-8))

    # 相关系数
    if method.lower() == "spearman":
        # 以秩替代
        Xr = np.apply_along_axis(lambda v: np.argsort(np.argsort(v)), 0, Xd)
        C = np.corrcoef(Xr, rowvar=False)
    else:
        C = np.corrcoef(Xd, rowvar=False)

    C = np.clip(C, -1.0, 1.0)
    n = C.shape[0]
    labels = list(feat_names)
    if len(labels) != n:
        # 名称对不上时生成占位
        labels = [f"f{i}" for i in range(n)]

    # 绘图（不依赖 seaborn）
    fig, ax = plt.subplots(figsize=(max(6, n * 0.35), max(4.5, n * 0.35)))
    im = ax.imshow(C, vmin=-vmax, vmax=vmax,
                   cmap="coolwarm", interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.8, label=f"{method.title()} correlation")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f"Plastic features correlation ({method})")
    ax.grid(False)

    # 小矩阵时标注数值
    if annotate and n <= 20:
        for i in range(n):
            for j in range(n):
                val = C[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                        color="black" if abs(val) < 0.75 else "white")

    fig.tight_layout()
    out_path = Path(debug_dir) / filename
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_plastic_spectral_profiles(
    debug_dir: Optional[Path | str],
    band_cube: np.ndarray,
    band_order: Sequence[str],
    *,
    plastic_rows: Optional[np.ndarray] = None,   # 选定塑料像元的行/列（与上游预测一致）
    plastic_cols: Optional[np.ndarray] = None,
    endmembers: Optional[Dict[str, Sequence[float]]] = None,  # 若提供，则叠加端元曲线
    filename_png: str = "plastic_spectral_profiles.png",
    filename_csv: str = "plastic_spectral_profiles.csv",
    percentiles: Sequence[int] = (25, 50, 75),
    normalize: Optional[str] = "max",   # None | "max" | "l2"
    sample_px_limit: int = 20000,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    保存“塑料候选像元”的光谱曲线（中位线 + 四分位带），并可叠加端元谱。
    同时导出 CSV（波长、端元、统计线）。
    """
    if not debug_dir:
        return (None, None)
    Path(debug_dir).mkdir(parents=True, exist_ok=True)

    # 选择可用波段（与 endmembers 维度尽量一致）
    use_bands = [b for b in band_order if b in S2_BANDS]  # 仅限我们知道中心波长的波段
    wl = np.array([S2_BANDS[b]["center"] for b in use_bands], dtype=np.float32)
    bidx = np.array([band_order.index(b) for b in use_bands], dtype=int)

    H, W, _ = band_cube.shape
    # 采样像元索引
    if plastic_rows is None or plastic_cols is None:
        # 若未传入，则对全图随机采样
        rng = np.random.default_rng(123)
        flat = rng.choice(
            H * W, size=min(sample_px_limit, H * W), replace=False)
        rows = (flat // W).astype(np.int64)
        cols = (flat % W).astype(np.int64)
    else:
        rows = np.asarray(plastic_rows, dtype=np.int64)
        cols = np.asarray(plastic_cols, dtype=np.int64)
        if rows.size > sample_px_limit:
            rng = np.random.default_rng(123)
            take = np.sort(rng.choice(
                rows.size, size=sample_px_limit, replace=False))
            rows, cols = rows[take], cols[take]

    # 提取谱
    spectra = band_cube[rows, cols][:, bidx].astype(np.float32)  # [N, B]
    spectra = np.nan_to_num(spectra, nan=0.0, posinf=0.0, neginf=0.0)

    # 归一化（可选）
    if normalize == "max":
        denom = np.max(spectra, axis=1, keepdims=True) + 1e-8
        spectra = spectra / denom
    elif normalize == "l2":
        denom = np.linalg.norm(spectra, axis=1, keepdims=True) + 1e-8
        spectra = spectra / denom

    # 统计线
    P = np.percentile(spectra, q=list(percentiles), axis=0)  # [K, B]
    pmap = {p: P[i] for i, p in enumerate(percentiles)}
    median = pmap.get(50, np.median(spectra, axis=0))
    p25 = pmap.get(25, np.percentile(spectra, 25, axis=0))
    p75 = pmap.get(75, np.percentile(spectra, 75, axis=0))

    # 绘图
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.fill_between(wl, p25, p75, alpha=0.25, label="IQR (25–75%)")
    ax.plot(wl, median, linewidth=2, label="Median (plastic candidates)")

    # 端元曲线（同样按需要归一化以便可比）
    if endmembers:
        for name, vec in endmembers.items():
            y = np.asarray(vec, dtype=np.float32)
            if y.shape[0] != len(wl):
                # 维度不一致则跳过
                continue
            yy = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            if normalize == "max":
                yy = yy / (yy.max() + 1e-8)
            elif normalize == "l2":
                yy = yy / (np.linalg.norm(yy) + 1e-8)
            ax.plot(wl, yy, linewidth=1.5, alpha=0.9,
                    label=f"Endmember: {name}")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance (normalized)" if normalize else "Reflectance")
    ax.set_title("Plastic spectral profiles")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    png_path = Path(debug_dir) / filename_png
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    # CSV 导出
    rows_out = []
    rows_out.append({"series": "median", **{f"{int(w)}": float(v)
                    for w, v in zip(wl, median)}})
    rows_out.append({"series": "p25",   **{f"{int(w)}": float(v)
                    for w, v in zip(wl, p25)}})
    rows_out.append({"series": "p75",   **{f"{int(w)}": float(v)
                    for w, v in zip(wl, p75)}})
    if endmembers:
        for name, vec in endmembers.items():
            if len(vec) == len(wl):
                rows_out.append({"series": f"endmember::{name}", **
                                {f"{int(w)}": float(v) for w, v in zip(wl, vec)}})
    df_out = pd.DataFrame(rows_out)
    csv_path = Path(debug_dir) / filename_csv
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return (png_path, csv_path)


# -------------------- helpers wrapping original water_filter names --------------------
def _robust_thresholds_water_proxy(ndwi, ndvi, fdi, ndti, rough_water):
    return robust_thresholds_water(ndwi, ndvi, fdi, ndti, rough_water)


def _hysteresis_water(ndwi, hi, lo):
    return hysteresis_water_mask(ndwi, hi, lo)


def _erode(mask: np.ndarray, k: int) -> np.ndarray:
    return erode_interior(mask, k) if k and k > 0 else mask


# -------------------- visualization --------------------
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
    max_scatter_points: int = 20000,  # 保留参数位，后续可加入点抽样可视化
) -> None:
    """Preview: probs + final classes. Final classes支持 0/1/2/4/5。"""
    H, W = class_map.shape
    base_bg = mask if mask is not None else rgb_image

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax_f, ax_s = axes[0]
    ax_p, ax_c = axes[1]

    # 上排概率
    if base_bg is not None:
        ax_f.imshow(base_bg, interpolation="nearest")
    imf = ax_f.imshow(floating_map, cmap="magma", vmin=0, vmax=1)
    plt.colorbar(imf, ax=ax_f, label="Floating prob")
    ax_f.set_title("Floating debris probability")
    ax_f.axis("off")

    if base_bg is not None:
        ax_s.imshow(base_bg, interpolation="nearest")
    ims = ax_s.imshow(suspended_map, cmap="magma", vmin=0, vmax=1)
    plt.colorbar(ims, ax=ax_s, label="Suspended prob")
    ax_s.set_title("Suspended matter probability")
    ax_s.axis("off")

    # 左下：塑料概率（若有）
    if plastic_map is not None:
        if base_bg is not None:
            ax_p.imshow(base_bg, interpolation="nearest")
        imp = ax_p.imshow(plastic_map, cmap="magma", vmin=0, vmax=1)
        plt.colorbar(imp, ax=ax_p, label="Plastic prob")
        ax_p.set_title("Plastic probability")
        ax_p.axis("off")
    else:
        ax_p.axis("off")
        ax_p.set_title("Plastic probability (N/A)")

    # 右下：最终分类（0/1/2/4/5）
    # 建立离散 colormap（保持 6 档，未用的 3 也填一个颜色避免索引错位）
    colors = [
        (0, 0, 0, 0),       # 0: transparent/background
        "#7fd3ff",          # 1: floating (non-plastic)
        "#9be37b",          # 2: suspended (non-plastic)
        "#cccccc",          # 3: (unused placeholder)
        "#ffd166",          # 4: plastic & floating
        "#ff8fab",          # 5: plastic & suspended
    ]
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, 5.5 + 1, 1)  # -0.5..5.5
    norm = BoundaryNorm(bounds, ncolors=cmap.N)

    if base_bg is not None:
        ax_c.imshow(base_bg, interpolation="nearest")
    imc = ax_c.imshow(class_map, cmap=cmap, norm=norm, alpha=0.75)
    ax_c.axis("off")

    # 自定义图例
    legend_patches = [
        (colors[1], "floating"),
        (colors[2], "suspended"),
        (colors[4], "plastic & floating"),
        (colors[5], "plastic & suspended"),
    ]
    from matplotlib.patches import Patch
    ax_c.legend(
        handles=[Patch(facecolor=c, edgecolor="none", label=lab)
                 for c, lab in legend_patches],
        loc="lower left",
        frameon=True,
        fontsize=9,
    )

    # 统计摘要
    summary = "\n".join(
        f"{s.label}: {s.area_m2:.1f} m² ({s.pixel_count} px)" for s in stats.values()
    )
    ax_c.set_title(f"Final classification\n{summary}")

    plt.tight_layout()
    plt.savefig(output_png, dpi=400, bbox_inches="tight")
    plt.close(fig)


# -------------------- Combined map builder --------------------
def build_combined_map(
    floating_prob: np.ndarray,
    suspended_prob: np.ndarray,
    *,
    water_mask: np.ndarray,
    shp_mask: np.ndarray,
    confident: np.ndarray,
    floating_thresh: float,
    suspended_thresh: float,
    plastic_mask_pred: Optional[np.ndarray] = None,
    tie_policy: str = "margin",  # 'margin'|'float'|'susp'|'strict'
) -> np.ndarray:
    valid = (water_mask.astype(bool) & shp_mask.astype(
        bool) & confident.astype(bool))
    fp = np.where(valid, np.nan_to_num(
        floating_prob, nan=-np.inf), -np.inf).astype(float)
    sp = np.where(valid, np.nan_to_num(suspended_prob,
                  nan=-np.inf), -np.inf).astype(float)

    fp_ok = fp >= float(floating_thresh)
    sp_ok = sp >= float(suspended_thresh)

    base = np.zeros(fp_ok.shape, dtype=np.uint8)
    base[fp_ok & ~sp_ok] = 1
    base[sp_ok & ~fp_ok] = 2

    both = fp_ok & sp_ok
    if tie_policy == "margin":
        m_fp = (fp - floating_thresh)
        m_sp = (sp - suspended_thresh)
        base[both & (m_fp >= m_sp)] = 1
        base[both & (m_sp > m_fp)] = 2
    elif tie_policy == "float":
        base[both & (fp >= sp)] = 1
        base[both & (sp > fp)] = 2
    elif tie_policy == "susp":
        base[both & (sp >= fp)] = 2
        base[both & (fp > sp)] = 1
    elif tie_policy == "strict":
        base[both & (fp > sp)] = 1
        base[both & (sp > fp)] = 2
    else:
        raise ValueError(
            "tie_policy only supports 'margin'|'float'|'susp'|'strict'")

    if plastic_mask_pred is not None:
        plast = (plastic_mask_pred.astype(bool) & valid)
        base[plast & (base == 1)] = 4
        base[plast & (base == 2)] = 5

    base[~valid] = 0
    return base


# -------------------- Main pipeline --------------------
@_with_scene_logging
def floating_debris_detection(
    input_tif: Union[Path, str, Sequence[Path | str]],
    mask_tif: Optional[Path | str] = None,
    *,
    output_csv: Path | str,
    preview_png: Path | str,
    shp_file: Optional[Path | str] = None,
    band_order: Sequence[str] = (
        "B2", "B3", "B4", "B5", "B8", "B8A", "B9", "B11", "B12"),
    # 水体掩膜控制
    water_boost: float = 1.0,
    min_water_ratio: float = 0.1,
    max_water_ratio: float = 0.9,
    erode_iters: int = 0,
    # 吸收一致性控制
    absorb_min_ratio: float = 0.60,
    absorb_max_ratio: float = 0.95,
    absorb_thr_step: float = 0.02,
    absorb_diff_start: float = 0.25,
    absorb_diff_max: float = 0.40,
    absorb_close: bool = True,
    # 种子目标
    seed_target_float: int | float = 40000,
    seed_target_susp: int | float = 30000,
    seed_target_float_ratio: Optional[float] = 0.08,
    seed_target_susp_ratio: Optional[float] = 0.08,
    seed_target_float_floor: Optional[int] = 2000,
    seed_target_susp_floor: Optional[int] = 2000,
    relax_steps: int = 4,
    # 指数阈值/概率阈值
    ndvi_thresh: float = 0.1,
    absorbance_thresh: float = 0.8,
    fdi_std_factor: float = 1.5,
    ndti_std_factor: float = 1.0,
    water_ndwi_thresh: float = 0.05,
    floating_thresh: float = 0.6,
    suspended_thresh: float = 0.6,
    # AOI筛选
    filter_field: Optional[str] = None,
    filter_values: Optional[Sequence] = None,
    filter_query: Optional[str] = None,
    filter_indices: Optional[Sequence[int]] = None,
    # 训练/推理
    device: Optional[str] = None,
    epochs: int = 5,
    repeats: int = 1,
    val_split: float = 0.2,
    model_out_dir: Optional[Path | str] = "mydata",
    log_file: Optional[Path | str] = None,
    debug_dir: Optional[Path | str] = None,
    # 塑料分支
    plastic_model_path: Optional[Path | str] = None,
    plastic_library_csv: Optional[Path | str] = None,
    plastic_prob_thresh: float = 0.5,
    plastic_retrain: bool = True,
    # patch采样
    tile_size: int = 256,
    stride: int = 256,
    max_patches: int = 1024,
    min_valid_frac: float = 0.6,
    pos_frac: float = 0.6,
    patch_seed: int = 42,
    # 阈值标定
    auto_calibrate_thresholds: bool = False,
    auto_threshold_metric: str = "f1",
    auto_threshold_candidates: Optional[Sequence[float]] = None,
    auto_threshold_max_samples: Optional[int] = 128,
    auto_threshold_batch_size: int = 2,
    # stacking
    stacking_preset: Optional[str] = None,
    stacking_sample_limit: Optional[int] = 250_000,
    stacking_pixel_stride: int = 1,
    stacking_max_pixels: Optional[int] = None,
    stacking_cv_folds: int = 5,
    # 判别/后处理
    strict_mode: bool = True,
    unc_margin: float = 0.20,
    min_area_float: int = 12,
    min_area_susp: int = 12,
    cap_float_pixels: Optional[int] = 150_000,
    cap_susp_pixels: Optional[int] = 150_000,
) -> pd.DataFrame:
    """Detect debris with per-scene UNet training and optional plastic scoring.

    ``plastic_retrain=True`` preserves the legacy adaptive plastic training;
    ``plastic_model_path`` then names the checkpoint output. Set it to ``False``
    to load a compatible existing checkpoint without training or overwriting it.
    Plastic inference errors are propagated in that read-only mode.
    """
    _require_torch()
    if not plastic_retrain and plastic_library_csv is not None:
        if not Path(plastic_library_csv).is_file():
            raise ValueError("Plastic inference requires an existing plastic_library_csv file")
        if plastic_model_path is None or not Path(plastic_model_path).is_file():
            raise ValueError("Plastic inference requires an existing plastic_model_path checkpoint file")
    output_csv = Path(output_csv)
    preview_png = Path(preview_png)

    # ----------------- 读取 AOI 矢量并投到影像 CRS -----------------
    gdf: Optional[gpd.GeoDataFrame] = None
    gdf_original: Optional[gpd.GeoDataFrame] = None
    shapefile_alignment_warnings: List[str] = []
    if shp_file is not None:
        gdf_read = gpd.read_file(shp_file)
        # attribute/row/query 三种筛选
        if filter_query:
            gdf_read = gdf_read.query(filter_query)
        if filter_field is not None and filter_values is not None:
            if filter_field not in gdf_read.columns:
                raise KeyError(
                    f"Field {filter_field} not found in shapefile columns: {list(gdf_read.columns)}")
            gdf_read = gdf_read[gdf_read[filter_field].isin(list(filter_values if isinstance(
                filter_values, (list, tuple, set, np.ndarray, pd.Series)) else [filter_values]))]
        if filter_indices is not None:
            idxs = [int(filter_indices)] if isinstance(
                filter_indices, (int, np.integer)) else list(filter_indices)
            gdf_read = gdf_read.iloc[idxs]
        gdf_read = gdf_read[gdf_read.geometry.notnull()]
        if gdf_read.empty:
            raise ValueError(
                "Shapefile filtering resulted in empty GeoDataFrame.")
        gdf_original = gdf_read.copy()

    # ----------------- 单期/多期影像读取对齐 -----------------
    is_multi = isinstance(input_tif, (list, tuple))
    transform = None
    src_crs = None
    shp_mask: np.ndarray
    band_cube: np.ndarray

    full_band_order = ("B2", "B3", "B4", "B5", "B6", "B7",
                       "B8", "B8A", "B9", "B11", "B12")
    compact_band_order = ("B2", "B3", "B4", "B5", "B8",
                          "B8A", "B9", "B11", "B12")
    minimal_band_order = ("B2", "B3", "B4", "B8", "B8A", "B11", "B12")

    stack = None
    if is_multi:
        # 需要外部提供 align_and_stack_scenes
        try:
            from .temporal_io import align_and_stack_scenes
        except Exception:
            align_and_stack_scenes = None  # type: ignore
        if align_and_stack_scenes is None:
            raise RuntimeError(
                "align_and_stack_scenes() 未找到。请提供 .temporal_io.align_and_stack_scenes 或改为单期输入。")
        stack, tgt_order, metas, transform, src_crs, shp_mask = align_and_stack_scenes(
            input_tifs=input_tif,
            target_band_order=full_band_order,
            resampling="bilinear",
            scale=1e-4,
            shp_file=shp_file,
            aoi_filter=(dict(field=filter_field, values=filter_values)
                        if (filter_field and filter_values) else None),
        )
        # 末帧作为当前影像
        band_cube = stack[-1]
        band_order = tuple(tgt_order)
    else:
        with rasterio.open(input_tif) as src:
            available_count = src.count
            raw_desc = src.descriptions or []
            descriptions = [d.strip().upper()
                            for d in raw_desc if isinstance(d, str)]
        known = set(S2_BANDS.keys())
        if descriptions and all(d in known for d in descriptions):
            band_order = tuple(descriptions)
        else:
            candidates = []
            if band_order:
                candidates.append(tuple(band_order))
            candidates.extend(
                [full_band_order, compact_band_order, minimal_band_order])
            seen, uniq = set(), []
            for o in candidates:
                if o not in seen:
                    uniq.append(o)
                    seen.add(o)
            chosen = None
            for o in uniq:
                if len(o) == available_count:
                    chosen = o
                    break
            if chosen is None:
                if available_count < len(minimal_band_order):
                    raise ValueError(
                        f"Unsupported band configuration: file has {available_count} bands < {len(minimal_band_order)}")
                chosen = tuple(uniq[0][:available_count])
                LOGGER.warning(
                    "Unknown layout; assume first %s bands of %s -> %s", available_count, uniq[0], chosen)
            band_order = tuple(chosen)
        band_indices = list(range(1, len(band_order)+1))
        bands_raw, transform, src_meta = read_bands(input_tif, band_indices)
        band_cube = bands_raw.astype(np.float32) * 0.0001
        src_crs = src_meta.get("crs") if isinstance(
            src_meta, dict) else getattr(src_meta, "get", lambda *_: None)("crs")

    if gdf_original is not None:
        src_crs, crs_override_msg = _maybe_override_bad_geographic_crs(
            src_crs, transform, gdf_original)
        if crs_override_msg:
            shapefile_alignment_warnings.append(crs_override_msg)

    # AOI 栅格化
    if gdf_original is not None:
        try:
            if src_crs is not None:
                gdf = gdf_original.to_crs(src_crs)
            else:
                gdf = gdf_original.copy()
        except Exception as exc:
            LOGGER.warning(
                "Failed to align shapefile CRS with raster CRS: %s", exc)
            gdf = gdf_original.copy()

        shp_mask = rasterize(
            [(geom, 1) for geom in gdf.geometry],
            out_shape=(band_cube.shape[0], band_cube.shape[1]),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        ).astype(bool)

        if not np.any(shp_mask):
            fallback_gdf, alignment_msg = _fallback_project_shapefile(
                gdf_original, src_crs, transform)
            if fallback_gdf is not None:
                shp_mask = rasterize(
                    [(geom, 1) for geom in fallback_gdf.geometry],
                    out_shape=(band_cube.shape[0], band_cube.shape[1]),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8,
                ).astype(bool)
                if np.any(shp_mask) and alignment_msg:
                    shapefile_alignment_warnings.append(alignment_msg)
            if not np.any(shp_mask):
                LOGGER.warning(
                    "Shapefile mask empty; revert to full-scene mask.")
                shp_mask = np.ones(
                    (band_cube.shape[0], band_cube.shape[1]), dtype=bool)
        for msg in shapefile_alignment_warnings:
            LOGGER.warning(msg)
    else:
        shp_mask = np.ones(
            (band_cube.shape[0], band_cube.shape[1]), dtype=bool)

    # 选取波段
    def require_band(name: str) -> np.ndarray:
        bm = {b: i for i, b in enumerate(band_order)}
        if name not in bm:
            raise KeyError(f"Band {name} missing. band_order={band_order}")
        return band_cube[:, :, bm[name]]

    B2 = require_band("B2")
    B3 = require_band("B3")
    B4 = require_band("B4")
    B8 = require_band("B8")
    B11 = require_band("B11")
    B12 = require_band("B12")
    B5 = band_cube[:, :, band_order.index(
        "B5")] if "B5" in band_order else None
    B6 = band_cube[:, :, band_order.index(
        "B6")] if "B6" in band_order else None
    B7 = band_cube[:, :, band_order.index(
        "B7")] if "B7" in band_order else None
    B8A = band_cube[:, :, band_order.index(
        "B8A")] if "B8A" in band_order else None
    B9 = band_cube[:, :, band_order.index(
        "B9")] if "B9" in band_order else None

    # 掩掉 AOI 外
    for arr in (B2, B3, B4, B8, B11, B12):
        arr[~shp_mask] = np.nan
    for arr in (B5, B6, B7, B8A, B9):
        if arr is not None:
            arr[~shp_mask] = np.nan

    # 预览
    inter = shp_mask
    rgb_preview = np.dstack([
        np.clip(B4 / np.nanmax(B4, where=inter, initial=1), 0, 1),
        np.clip(B3 / np.nanmax(B3, where=inter, initial=1), 0, 1),
        np.clip(B2 / np.nanmax(B2, where=inter, initial=1), 0, 1),
    ])

    # ----------------- 水体 NDWI + 滞后门 + 覆盖率自调 -----------------
    ndwi = compute_ndwi(B3, B8)
    rough_thr = water_ndwi_thresh
    try:
        prc = float(np.nanpercentile(ndwi[shp_mask], 35))
        if np.isfinite(prc):
            rough_thr = max(min(prc, 0.6), -0.5)
    except Exception:
        pass
    rough_water = (ndwi > rough_thr) & shp_mask
    if not np.any(rough_water):
        rough_water = (ndwi > water_ndwi_thresh) & shp_mask

    # 临时指标用于阈值估计
    _ndvi_tmp = (B8 - B4) / (B8 + B4 + 1e-6)
    _fdi_tmp = compute_fdi(B4, B8, B11)
    _ndti_tmp = compute_ndti(B4, B3)
    rob = _robust_thresholds_water_proxy(
        ndwi, _ndvi_tmp, _fdi_tmp, _ndti_tmp, rough_water)

    hi, lo = float(rob["ndwi_hi"]), float(rob["ndwi_lo"])
    if water_boost and water_boost > 0:
        hi = float(np.clip(hi - 0.01 * water_boost, -1.0, 1.0))
        lo = float(np.clip(lo - 0.02 * water_boost, -1.0, hi - 1e-3))

    water_mask = _hysteresis_water(ndwi, hi, lo) & shp_mask
    water_mask = _erode(water_mask, erode_iters)
    if not np.any(water_mask):
        water_mask = rough_water

    total_aoi = max(1, int(np.count_nonzero(shp_mask)))
    target_min = int(min_water_ratio * total_aoi)
    target_max = int(max_water_ratio * total_aoi)
    step = 0.01
    cov = int(np.count_nonzero(water_mask))
    it = 0
    while cov < target_min and it < 10:
        hi -= step
        lo -= step * 1.2
        lo = float(np.clip(lo, -1.0, hi - 1e-3))
        water_mask = _hysteresis_water(ndwi, hi, lo) & shp_mask
        water_mask = _erode(water_mask, erode_iters)
        cov = int(np.count_nonzero(water_mask))
        it += 1
    if cov < target_min:
        kernel = np.ones((3, 3), np.uint8)
        water_mask = (cv2.dilate(water_mask.astype(np.uint8),
                      kernel, iterations=1) > 0) & shp_mask
        cov = int(np.count_nonzero(water_mask))
    it = 0
    while cov > target_max and it < 10:
        lo += step
        lo = min(lo, hi - 1e-3)
        water_mask = _hysteresis_water(ndwi, hi, lo) & shp_mask
        water_mask = _erode(water_mask, erode_iters)
        cov = int(np.count_nonzero(water_mask))
        it += 1

    _save_debug_image(debug_dir, "ndwi_map.png", ndwi,
                      title=f"NDWI (hi={hi:.3f}, lo={lo:.3f})", cmap="coolwarm", mask=shp_mask, vmin=-0.5, vmax=0.5)
    _save_debug_overlay(debug_dir, "water_mask_overlay.png", rgb_preview,
                        water_mask, title="Detected water mask (auto-tuned)", color="cyan")

    # ----------------- 指数（稳健缩放） -----------------
    ndvi = (B8 - B4) / (B8 + B4 + 1e-6)
    ndvi[~shp_mask] = np.nan
    fdi = compute_fdi(B4, B8, B11)
    fdi[~shp_mask] = np.nan
    ndti = compute_ndti(B4, B3)
    ndti[~shp_mask] = np.nan
    sabi = compute_sabi(B8, B4)
    sabi[~shp_mask] = np.nan

    fdi_strong, fdi_weak = float(rob["fdi_strong"]), float(rob["fdi_weak"])
    ndti_strong, ndti_weak = float(rob["ndti_strong"]), float(rob["ndti_weak"])
    ndvi_lo_r, ndvi_hi_r = float(rob["ndvi_lo"]), float(rob["ndvi_hi"])

    _save_debug_image(debug_dir, "ndvi_map.png", ndvi, title="NDVI",
                      cmap="RdYlGn", mask=water_mask, vmin=-1, vmax=1)
    _save_debug_image(debug_dir, "fdi_map.png",  fdi,
                      title=f"FDI (str={fdi_strong:.3f}, weak={fdi_weak:.3f})",  cmap="plasma", mask=water_mask)
    _save_debug_image(debug_dir, "ndti_map.png", ndti,
                      title=f"NDTI (str={ndti_strong:.3f}, weak={ndti_weak:.3f})", cmap="viridis", mask=water_mask)

    # 吸收一致性（B11/B12）
    B11_pos = np.clip(B11, 1e-6, None)
    B12_pos = np.clip(B12, 1e-6, None)
    A_B11 = -np.log10(B11_pos)
    A_B11[~shp_mask] = np.nan
    A_B12 = -np.log10(B12_pos)
    A_B12[~shp_mask] = np.nan
    A_min = np.minimum(A_B11, A_B12)
    A_diff = np.abs(A_B11 - A_B12)

    thr = float(absorbance_thresh) if (np.isfinite(absorbance_thresh) and absorbance_thresh > 0) else (
        float(np.nanpercentile(A_min[water_mask & np.isfinite(
            A_min)], 60)) if np.isfinite(A_min).any() else 0.60
    )
    diff_lim = float(absorb_diff_start)
    absorb_mask = (A_min > thr) & (A_diff < diff_lim) & shp_mask

    water_total = max(1, int(np.count_nonzero(water_mask)))
    cov = int(np.count_nonzero(absorb_mask & water_mask))
    tgt_min = int(absorb_min_ratio * water_total)
    tgt_max = int(absorb_max_ratio * water_total)

    it = 0
    while cov < tgt_min and it < 20:
        thr = max(0.05, thr - absorb_thr_step)
        if cov < tgt_min * 0.7:
            diff_lim = min(absorb_diff_max, diff_lim + 0.02)
        absorb_mask = (A_min > thr) & (A_diff < diff_lim) & shp_mask
        cov = int(np.count_nonzero(absorb_mask & water_mask))
        it += 1
    it = 0
    while cov > tgt_max and it < 20:
        thr += absorb_thr_step
        absorb_mask = (A_min > thr) & (A_diff < diff_lim) & shp_mask
        cov = int(np.count_nonzero(absorb_mask & water_mask))
        it += 1

    if absorb_close:
        k = np.ones((3, 3), np.uint8)
        absorb_mask = (cv2.morphologyEx(absorb_mask.astype(
            np.uint8), cv2.MORPH_CLOSE, k, iterations=1) > 0) & water_mask

    _save_debug_image(debug_dir, "absorbance_B11.png", A_B11,
                      title="Absorbance B11", cmap="magma", mask=water_mask)
    _save_debug_image(debug_dir, "absorbance_B12.png", A_B12,
                      title="Absorbance B12", cmap="magma", mask=water_mask)

    # 外部 mask_tif
    if mask_tif is not None and Path(mask_tif).exists():
        with rasterio.open(mask_tif) as msk:
            ext_mask = msk.read(1)
        ext_mask = cv2.resize(
            ext_mask, (water_mask.shape[1], water_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        ext_mask = (ext_mask > 0).astype(np.uint8)
        ext_mask[~shp_mask] = 0
    else:
        ext_mask = water_mask.astype(np.uint8)
        LOGGER.info(
            "No mask_tif provided; using NDWI-derived water mask for seed generation.")

    # BD_B9
    bd_b9 = None
    if B9 is not None and B8A is not None:
        bd_b9 = band_depth(band_value=B9, left_value=B8A, right_value=B11,
                           left_center=865.0, right_center=1610.0, band_center=945.0)
        bd_b9[~shp_mask] = np.nan

    water_pixels = int(np.count_nonzero(water_mask))

    # ----------------- 简易种子（稳健阈值） -----------------
    # 使用 strong/weak 取决于 strict_mode
    fdi_thresh = float(fdi_strong if strict_mode else fdi_weak)
    ndti_thresh_val = float(ndti_strong if strict_mode else ndti_weak)

    floating_seed = (water_mask & (ext_mask == 1) &
                     (ndvi < ndvi_thresh) & (fdi > fdi_thresh) & absorb_mask)

    red_vals = _collect_stat_values(
        "B4 reflectance", B4, water_mask, fallback_mask=shp_mask)
    green_vals = _collect_stat_values(
        "B3 reflectance", B3, water_mask, fallback_mask=shp_mask)
    ndti_vals = _collect_stat_values(
        "NDTI index", ndti, water_mask, fallback_mask=shp_mask)
    red_p70 = _percentile_safe(red_vals, 70, default=0.0)
    red_p60 = _percentile_safe(red_vals, 60, default=red_p70)
    red_p55 = _percentile_safe(red_vals, 55, default=red_p60)
    red_p50 = _percentile_safe(red_vals, 50, default=red_p55)
    green_p65 = _percentile_safe(green_vals, 65, default=0.0)
    green_p60 = _percentile_safe(green_vals, 60, default=green_p65)
    green_p55 = _percentile_safe(green_vals, 55, default=green_p60)
    green_p50 = _percentile_safe(green_vals, 50, default=green_p55)

    def _make_suspended_mask(ndti_thr: float, red_thr: float, green_thr: float, ndwi_cap: float) -> np.ndarray:
        return (
            water_mask &
            (ndti > ndti_thr) &
            (B4 > red_thr) &
            (B3 > green_thr) &
            (ndwi < ndwi_cap)
        )

    ndwi_cap_primary = 0.50
    suspended_seed = _make_suspended_mask(
        ndti_thresh_val, red_p70, green_p65, ndwi_cap_primary)

    if not np.any(suspended_seed) and water_pixels > 0:
        def _safe_float(val: float, default: float) -> float:
            return float(val) if np.isfinite(val) else float(default)

        ndti_soft = _safe_float(ndti_thresh_val, 0.1)
        ndti_relaxed = min(ndti_soft, _safe_float(ndti_weak, ndti_soft - 0.05))

        def _lower_ndti(val: float, delta: float) -> float:
            return max(-0.5, float(val) - delta)

        fallback_plan = [
            ("ndti_relax", _lower_ndti(ndti_soft, 0.05), red_p70, green_p60, 0.15),
            ("weak_stats", _lower_ndti(ndti_relaxed, 0.0), red_p60, green_p55, 0.20),
            ("wide_ndwi", _lower_ndti(ndti_relaxed, 0.08), red_p55, green_p50, 0.28),
        ]
        min_seed_px = min(5000, max(25, int(0.0001 * water_pixels)))
        for tag, ndti_thr, red_thr, green_thr, ndwi_cap in fallback_plan:
            mask = _make_suspended_mask(ndti_thr, red_thr, green_thr, ndwi_cap)
            cnt = int(np.count_nonzero(mask))
            if cnt >= min_seed_px:
                suspended_seed = mask
                LOGGER.info(
                    "Suspended seeds empty; applied %s fallback (ndti>%.3f, red>%.3f, green>%.3f, ndwi<%.2f) -> %s pixels",
                    tag, ndti_thr, red_thr, green_thr, ndwi_cap, f"{cnt:,}",
                )
                break
        if not np.any(suspended_seed):
            ndti_q85 = _percentile_safe(ndti_vals, 85, default=ndti_relaxed)
            ndti_q70 = _percentile_safe(ndti_vals, 70, default=ndti_q85 - 0.05)
            ndti_id = [
                ("percentile85", ndti_q85, red_p55, green_p50, 0.32),
                ("percentile70", ndti_q70, red_p50, green_p50, 0.38),
            ]
            for tag, ndti_thr, red_thr, green_thr, ndwi_cap in ndti_id:
                mask = _make_suspended_mask(
                    ndti_thr, red_thr, green_thr, ndwi_cap)
                cnt = int(np.count_nonzero(mask))
                if cnt >= max(25, min_seed_px // 4):
                    suspended_seed = mask
                    LOGGER.info(
                        "Suspended seeds fallback (%s) triggered -> %s pixels", tag, f"{cnt:,}")
                    break

        if not np.any(suspended_seed) and ndti_vals.size > 0:
            safe_vals = ndti.copy()
            safe_vals[~water_mask | ~np.isfinite(safe_vals)] = -np.inf
            flat = safe_vals.ravel()
            take = max(100, min_seed_px // 2)
            finite_count = np.count_nonzero(np.isfinite(ndti_vals))
            if finite_count > 0:
                take = min(int(finite_count), take)
            if take > 0:
                idx = np.argpartition(flat, -take)[-take:]
                mask = np.zeros_like(water_mask, dtype=bool)
                mask.reshape(-1)[idx] = True
                cnt = int(np.count_nonzero(mask))
                if cnt > 0:
                    suspended_seed = mask
                    LOGGER.info(
                        "Suspended seeds forced by top-N NDTI fallback -> %s pixels", f"{cnt:,}")

        if not np.any(suspended_seed):
            LOGGER.warning(
                "Suspended seeds remain empty even after relaxing thresholds; suspended branch may underperform.")

    if water_pixels == 0:
        LOGGER.warning("No water pixels detected; aborting UNet training.")
        floating_prob = np.zeros_like(water_mask, dtype=np.float32)
        suspended_prob = np.zeros_like(water_mask, dtype=np.float32)
        combined_map = np.zeros_like(water_mask, dtype=np.uint8)
        df = pd.DataFrame(columns=[
            "row", "col", "x", "y", "lon", "lat", "class", "probability", "NDVI", "FDI", "NDTI", "SABI", "A_B11", "A_B12", "PlasticProb", "PlasticFlag", "PlasticType",
        ])
        pix_area = _pixel_area_from_transform(transform)
        stats = {
            "floating_debris": DetectionStats("floating_debris", 0.0, 0),
            "suspended_matter": DetectionStats("suspended_matter", 0.0, 0),
        }
        visualize_results(rgb_image=rgb_preview, floating_map=floating_prob, suspended_map=suspended_prob,
                          plastic_map=None, class_map=combined_map, df=df, stats=stats, output_png=preview_png, mask=None)
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        return df

    LOGGER.info("AOI pixels: %s | floating seeds: %s | suspended seeds: %s",
                f"{water_pixels:,}", f"{int(np.count_nonzero(floating_seed)):,}", f"{int(np.count_nonzero(suspended_seed)):,}")

    # 稳健归一（以水体为范围）
    for arr in (B2, B3, B4, B8, B11, B12):
        arr[:] = _robust_scale(arr, water_mask)
    for arr in (B5, B6, B7, B8A, B9):
        if arr is not None:
            arr[:] = _robust_scale(arr, water_mask)
    ndwi = _robust_scale(ndwi, water_mask)
    ndvi = _robust_scale(ndvi, water_mask)
    fdi = _robust_scale(fdi,  water_mask)
    ndti = _robust_scale(ndti, water_mask)
    sabi = _robust_scale(sabi, water_mask)
    A_B11 = _robust_scale(A_B11, water_mask)
    A_B12 = _robust_scale(A_B12, water_mask)

    # 特征构建
    feature_list: List[np.ndarray] = [B2, B3, B4]
    if B5 is not None:
        feature_list.append(B5)
    feature_list.append(B8)
    if B8A is not None:
        feature_list.append(B8A)
    if B9 is not None:
        feature_list.append(B9)
    feature_list.extend([B11, B12, ndwi, ndvi, fdi, ndti, sabi, A_B11, A_B12])
    features = np.stack(feature_list, axis=-1).astype(np.float32)
    features[np.isnan(features)] = 0.0

    # 伪标签
    label_stack = np.stack([floating_seed.astype(
        np.float32), suspended_seed.astype(np.float32)], axis=0)
    if label_stack.ndim == 2:
        label_stack = label_stack[np.newaxis, ...]
    elif label_stack.ndim == 3 and label_stack.shape[0] > 8 and label_stack.shape[-1] <= 8:
        label_stack = np.transpose(label_stack, (2, 0, 1))

    if int(np.count_nonzero(floating_seed)) == 0 and int(np.count_nonzero(suspended_seed)) == 0:
        LOGGER.warning("No pseudo-labels available; skip UNet training.")
        floating_prob = floating_seed.astype(np.float32)
        suspended_prob = suspended_seed.astype(np.float32)
    else:
        if any(v is None for v in (sample_training_patches, FloatingDataset, train_unet_from_patches, infer_unet)):
            raise RuntimeError(
                "train_model utilities are unavailable; reinstall sateline_ml package")
        if torch is None:
            raise RuntimeError(
                "Torch should be available after _require_torch() call")

        # 设备选择
        if resolve_device is not None:
            sel_dev = resolve_device(device)
            train_device_arg = str(sel_dev)
            inference_device_arg = str(sel_dev)
        else:
            # 简化：优先 cuda -> mps -> cpu
            if device:
                try:
                    sel = torch.device(device)
                except Exception:
                    sel = torch.device("cuda" if torch.cuda.is_available() else (
                        "mps" if getattr(torch.backends, "mps").is_available() else "cpu"))
            else:
                sel = torch.device("cuda" if torch.cuda.is_available() else (
                    "mps" if getattr(torch.backends, "mps").is_available() else "cpu"))
            train_device_arg = str(sel)
            inference_device_arg = str(sel)
        LOGGER.info("Training device: %s", train_device_arg)

        # 采样 patch
        images_list, masks_list, valid_list = sample_training_patches(
            features, label_stack, water_mask.astype(np.float32),
            tile=tile_size, stride=stride, min_valid_frac=min_valid_frac,
            max_patches=max_patches, pos_frac=pos_frac, seed=patch_seed,
        )
        if len(images_list) == 0:
            LOGGER.warning(
                "No valid training patches (lower min_valid_frac or adjust tile/stride). Falling back to seeds.")
            floating_prob = floating_seed.astype(np.float32)
            suspended_prob = suspended_seed.astype(np.float32)
        else:
            dataset = FloatingDataset(images_list, masks_list, valid_list)
            scene_id = Path(input_tif[0] if is_multi else input_tif).stem
            model_root = Path(
                model_out_dir) if model_out_dir is not None else Path.cwd()
            per_scene_dir = model_root / "models" / scene_id

            if repeats and repeats > 1 and train_unet_repeated is not None:
                save_dir = per_scene_dir
                model, _, device_obj, best_val, best_path = train_unet_repeated(
                    dataset, repeats=repeats, val_split=val_split, seed=patch_seed,
                    epochs=epochs, lr=1e-3, device=train_device_arg, batch_size=1, save_best_to=save_dir,
                )
                LOGGER.info("Selected best model with val_loss=%.6f; saved to %s", float(
                    best_val), save_dir)
            else:
                model, _, device_obj = train_unet_from_patches(
                    dataset, epochs=epochs, lr=1e-3, device=train_device_arg, batch_size=1, pos_weight=None,
                )
                best_val = float("nan")
                best_path = per_scene_dir / "best_unet.pth"
                try:
                    per_scene_dir.mkdir(parents=True, exist_ok=True)
                    torch.save({"state_dict": model.state_dict(), "meta": {
                               "repeats": 1, "epochs": epochs}}, best_path)
                except Exception as exc:
                    LOGGER.warning(
                        "Could not save single-run weights: %s", exc)

            # 概率阈值自动标定（可选）
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
                    floating_thresh = float(calibration_result.thresholds.get(
                        "floating_debris", floating_thresh))
                    suspended_thresh = float(calibration_result.thresholds.get(
                        "suspended_matter", suspended_thresh))
                    save_calibration_result(
                        calibration_result, thresholds_path)
                    LOGGER.info("Auto-calibrated thresholds (%s): floating=%.3f, suspended=%.3f",
                                calibration_result.metric_name, floating_thresh, suspended_thresh)
                except Exception as calib_exc:
                    calibration_result = None
                    LOGGER.warning(
                        "Threshold calibration failed: %s", calib_exc)
            if calibration_result is None:
                try:
                    thresholds_path.parent.mkdir(parents=True, exist_ok=True)
                    payload = {
                        "metric": "configured",
                        "thresholds": {"floating_debris": float(floating_thresh), "suspended_matter": float(suspended_thresh)},
                        "per_class_metrics": {}, "candidate_thresholds": [], "class_order": ["floating_debris", "suspended_matter"],
                    }
                    thresholds_path.write_text(json.dumps(
                        payload, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception as exc:
                    LOGGER.warning(
                        "Failed to write baseline threshold summary: %s", exc)

            # 推理（优先 GPU/MPS，自动降级 CPU 并自适应 tile）
            try:
                model = model.to(torch.device(inference_device_arg))
                dev = torch.device(inference_device_arg)
                if dev.type in ("cuda", "mps"):
                    tile_candidates = (1024, 896, 768, 640, 512, 384, 320) if dev.type == "cuda" else (
                        768, 640, 512, 448, 384, 320)
                    for t in tile_candidates:
                        try:
                            probs = infer_unet_tiled(
                                model, features, device=dev, tile=t, overlap=64, use_amp=True, show_progress=True)
                            break
                        except RuntimeError as oom:
                            msg = str(oom).lower()
                            if "out of memory" in msg or "mps" in msg or "resource" in msg:
                                try:
                                    if dev.type == "cuda":
                                        torch.cuda.empty_cache()
                                    elif dev.type == "mps" and hasattr(torch, "mps"):
                                        torch.mps.empty_cache()  # type: ignore
                                except Exception:
                                    pass
                                continue
                            raise
                    else:
                        LOGGER.warning(
                            "GPU/MPS OOM for all tile sizes; fallback to CPU")
                        cpu = torch.device("cpu")
                        model = model.to(cpu)
                        probs = infer_unet_tiled(
                            model, features, device=cpu, tile=768, overlap=64, use_amp=False, show_progress=True)
                else:
                    probs = infer_unet_tiled(
                        model, features, device=dev, tile=768, overlap=64, use_amp=False, show_progress=True)
            except Exception as exc:
                LOGGER.warning(
                    "Tiled inference failed (%s); retrying full-image on CPU", exc)
                cpu = torch.device("cpu")
                model = model.to(cpu)
                probs = infer_unet(model, features, device=cpu)

            floating_prob = probs[0]
            suspended_prob = probs[1] if probs.shape[0] > 1 else np.zeros_like(
                floating_prob)

    floating_prob[~water_mask] = np.nan
    suspended_prob[~water_mask] = np.nan

    # ------------- stacking（可选） -------------
    if stacking_preset and run_stacking_pipeline is not None and build_tabular_feature_stack is not None:
        try:
            # —— 1) 训练采样像元索引（支持步长与最大像元数）——
            stride_px = max(1, int(stacking_pixel_stride))
            H, W = water_mask.shape
            base_flat = np.flatnonzero(water_mask.reshape(-1))
            if stride_px > 1:
                rows = base_flat // W
                cols = base_flat % W
                keep = (rows % stride_px == 0) & (cols % stride_px == 0)
                mask_flat = base_flat[keep]
            else:
                mask_flat = base_flat

            if stacking_max_pixels is not None and mask_flat.size > int(stacking_max_pixels):
                rng = np.random.default_rng(patch_seed + 1)
                sel = np.sort(rng.choice(mask_flat.size, size=int(
                    stacking_max_pixels), replace=False))
                mask_flat = mask_flat[sel]

            # —— 2) 构建派生指标（供表格特征使用）——
            derived_indices_map = {
                "NDWI": ndwi, "NDVI": ndvi, "FDI": fdi, "NDTI": ndti, "SABI": sabi,
                "A_B11": A_B11, "A_B12": A_B12
            }

            # —— 3) 生成表格特征栈（仅抽样像元）——
            tab_X, tab_feat_names, stack_indices = build_tabular_feature_stack(
                band_cube=band_cube,
                derived_indices=derived_indices_map,
                band_names=band_order,
                water_mask=water_mask,
                kernel_sizes=(3, 5),
                include_texture=True,
                flat_indices=mask_flat,
            )

            if tab_X.shape[0] > 0 and stack_indices.size > 0:
                # —— 4) 监督标签（与 stack_indices 对齐）——
                labels_map = {
                    "floating_debris": floating_seed.reshape(-1)[stack_indices].astype(np.int8),
                    "suspended_matter": suspended_seed.reshape(-1)[stack_indices].astype(np.int8),
                }

                # —— 5) 预计算元特征（例如来自 UNet 的像元概率）——
                precomputed_meta = {
                    "unet_floating":  floating_prob.reshape(-1)[stack_indices],
                    "unet_suspended": suspended_prob.reshape(-1)[stack_indices],
                }

                # —— 6) NEW: 超参调优配置（若已有 stacking_tune_config 变量则用之）——
                tune_cfg = (locals().get("stacking_tune_config") or {
                    "enable": True,            # 开启随机搜索
                    "base_iter": 25,           # 基学习器随机搜索次数
                    "meta_iter": 25,           # 元学习器随机搜索次数
                    "cv": 3,                   # 调参时的 CV 折数
                    "scoring": "average_precision",  # 不平衡更稳；也可用 "roc_auc"
                })

                # —— 7) 跑堆叠（与超参调优版接口同步）——
                fused_maps = run_stacking_pipeline(
                    stacking_preset,
                    tabular_X=tab_X,
                    labels=labels_map,
                    probabilities={},                  # 若有单模概率兜底可放这里
                    precomputed_meta=precomputed_meta,
                    mask_flat=np.arange(tab_X.shape[0]),  # 训练/OOF 使用全部 tab_X 行
                    random_state=patch_seed,
                    feature_names=tab_feat_names,
                    cv_folds=stacking_cv_folds,
                    tune_config=tune_cfg,              # NEW: 传入调优配置
                    # calibrator_override=None,        # 如需覆盖预设的校准器，可显式指定
                )

                # —— 8) 可选 CRF/形态学后处理（按预设自动取）——
                if postprocess_probabilities is not None and STACKING_PRESETS:
                    preset_cfg = STACKING_PRESETS.get(
                        stacking_preset.lower(), {}) or {}
                    if preset_cfg.get("postprocess"):
                        fused_maps = postprocess_probabilities(
                            fused_maps, rgb=rgb_preview, postprocess_cfg=preset_cfg.get(
                                "postprocess")
                        )

                # —— 9) 写回到整幅概率图（未抽样像元保持原值）——
                fused_float = floating_prob.reshape(-1).copy()
                fused_susp = suspended_prob.reshape(-1).copy()
                fused_float[stack_indices] = fused_maps["floating_debris"]
                fused_susp[stack_indices] = fused_maps["suspended_matter"]

                floating_prob = fused_float.reshape(H, W)
                suspended_prob = fused_susp.reshape(H, W)
                floating_prob[~water_mask] = np.nan
                suspended_prob[~water_mask] = np.nan

        except Exception as exc:
            LOGGER.warning("Stacking '%s' failed: %s", stacking_preset, exc)

    # ------------- 不确定性边际 + 胜者策略 -> 类图 -------------
    confident = (np.abs(floating_prob - suspended_prob) >= float(unc_margin))
    floating_mask_pred = (
        floating_prob > floating_thresh) & shp_mask & confident
    suspended_mask_pred = (
        suspended_prob > suspended_thresh) & shp_mask & confident

    # 塑料分支（稀疏抽样）
    plastic_prob_map: Optional[np.ndarray] = None
    plastic_mask_pred: Optional[np.ndarray] = None
    plastic_type_map: Optional[np.ndarray] = None
    plastic_enabled = plastic_library_csv is not None and Path(
        plastic_library_csv).exists()
    if plastic_library_csv is not None and not Path(plastic_library_csv).exists():
        LOGGER.warning(
            "Skipping plastic ML (library CSV missing: %s)", plastic_library_csv)
        plastic_enabled = False

    if plastic_enabled:
        try:
            dev = torch.device(inference_device_arg if 'inference_device_arg' in locals(
            ) else ("cuda" if torch and torch.cuda.is_available() else "cpu"))
            rng = np.random.default_rng(1234)
            cand_pos_mask = ((floating_prob > max(floating_thresh, 0.6)) | (
                suspended_prob > max(suspended_thresh, 0.6))) & water_mask
            cand_neg_mask = water_mask & (~cand_pos_mask)
            max_pos, max_neg = 120_000, 180_000
            pos_idx = np.where(cand_pos_mask.ravel())[0]
            neg_idx = np.where(cand_neg_mask.ravel())[0]
            if pos_idx.size > max_pos:
                pos_idx = rng.choice(pos_idx, size=max_pos, replace=False)
            if neg_idx.size > max_neg:
                neg_idx = rng.choice(neg_idx, size=max_neg, replace=False)
            sel_idx = np.concatenate([pos_idx, neg_idx])
            if sel_idx.size == 0:
                raise ValueError("No candidate pixels for plastic training.")
            H, W = water_mask.shape
            sel_rows = (sel_idx // W).astype(np.int64)
            sel_cols = (sel_idx % W).astype(np.int64)

            X_plastic, feat_names, meta = build_plastic_feature_matrix_sparse(
                band_cube, band_order, plastic_library_csv,
                sel_rows, sel_cols, background_mask=water_mask, bg_sample=200_000,
            )
            y_plastic = np.zeros(sel_idx.size, dtype=np.float32)
            y_plastic[:len(pos_idx)] = 1.0

            if not plastic_retrain:
                model_plastic = _load_compatible_plastic_model(
                    plastic_model_path, input_dim=X_plastic.shape[1],
                    feature_names=feat_names, band_order=band_order, device=str(dev),
                )
                LOGGER.info("Using existing plastic checkpoint: %s", plastic_model_path)
            else:
                model_plastic, _ = train_plastic_model(
                    X_plastic, y_plastic, epochs=400, lr=1e-3, weight_decay=5e-5,
                    device=str(dev), verbose=False, hidden=(256, 128), dropout=0.25,
                )
                if debug_dir is not None:
                    _save_plastic_feature_correlations_pre(
                        debug_dir, feat_names, X_plastic, y_plastic)
                    _save_plastic_spectral_profiles_pre(
                        debug_dir, feat_names, X_plastic, y_plastic)

                if plastic_model_path is not None:
                    meta_obj = PlasticModelMetadata(input_dim=X_plastic.shape[1], feature_names=tuple(
                        feat_names), band_order=tuple(band_order))
                    try:
                        from sklearn.metrics import accuracy_score, log_loss
                        _has_sklearn = True
                    except Exception:
                        _has_sklearn = False
                    probs_train = predict_plastic(
                        model_plastic, X_plastic, device=str(dev)).astype(np.float32)
                    y_true = y_plastic.astype(np.int32)
                    y_pred = (probs_train >= float(
                        plastic_prob_thresh)).astype(np.int32)
                    if _has_sklearn:
                        acc = float(accuracy_score(y_true, y_pred))
                        loss = float(log_loss(y_true, np.clip(
                            probs_train, 1e-12, 1-1e-12), labels=[0, 1]))
                    else:
                        p = np.clip(probs_train, 1e-12, 1-1e-12)
                        acc = float((y_true == y_pred).mean())
                        loss = float(-np.mean(y_true*np.log(p) +
                                     (1-y_true)*np.log(1-p)))
                    LOGGER.info(
                        "[Plastic-ML] Accuracy: %.4f | LogLoss: %.4f", acc, loss)
                    try:
                        save_plastic_model(plastic_model_path,
                                           model_plastic, meta_obj)
                    except Exception as _exc:
                        LOGGER.warning("Could not save plastic model: %s", _exc)
            # 预测全部候选
            pred_idx = np.where(cand_pos_mask.ravel())[0]
            pred_rows = (pred_idx // W).astype(np.int64)
            pred_cols = (pred_idx % W).astype(np.int64)
            X_pred, _, meta_pred = build_plastic_feature_matrix_sparse(
                band_cube, band_order, plastic_library_csv, pred_rows, pred_cols, background_mask=water_mask, bg_sample=200_000,
            )
            prob_vec = predict_plastic(model_plastic, X_pred, device=str(dev))
            plastic_prob_map = np.zeros_like(floating_prob, dtype=np.float32)
            plastic_prob_map.ravel()[pred_idx] = prob_vec.astype(np.float32)
            plastic_prob_map[~water_mask] = np.nan
            plastic_mask_pred = (
                plastic_prob_map > plastic_prob_thresh) & shp_mask & confident
            # 1) 特征相关性
            try:
                _save_plastic_feature_correlations(
                    debug_dir=debug_dir,
                    X=X_pred,
                    feat_names=feat_names,
                    filename="plastic_feature_correlations.png",
                    method="pearson",
                    sample_row_limit=12000,
                    annotate=False
                )
            except Exception as exc:
                LOGGER.warning(
                    "Saving plastic feature correlations failed: %s", exc)

            # 2) 光谱剖面（叠加端元）
            try:
                em = (meta_pred or {}).get(
                    "endmembers") if meta_pred is not None else None
                _save_plastic_spectral_profiles(
                    debug_dir=debug_dir,
                    band_cube=band_cube,
                    band_order=band_order,
                    plastic_rows=pred_rows,
                    plastic_cols=pred_cols,
                    endmembers=em,
                    filename_png="plastic_spectral_profiles.png",
                    filename_csv="plastic_spectral_profiles.csv",
                    percentiles=(25, 50, 75),
                    normalize="max",
                    sample_px_limit=20000
                )
            except Exception as exc:
                LOGGER.warning(
                    "Saving plastic spectral profiles failed: %s", exc)
            # 简易塑料类型归因（与端元余弦角最小）
            try:
                endmembers = (meta_pred or {}).get(
                    "endmembers") if meta_pred is not None else None
                if endmembers:
                    type_names = list(endmembers.keys())
                    em_matrix = np.stack(
                        [np.asarray(endmembers[n], dtype=np.float32) for n in type_names], axis=0)
                    n_bands = em_matrix.shape[1]
                    rho_part = X_pred[:, :n_bands].astype(np.float32)
                    denom_pix = np.linalg.norm(
                        rho_part, axis=1, keepdims=True) + 1e-12
                    denom_em = np.linalg.norm(em_matrix, axis=1) + 1e-12
                    cos = (rho_part @ em_matrix.T) / (denom_pix * denom_em)
                    cos = np.clip(cos, -1.0, 1.0)
                    best_idx = np.argmin(np.arccos(cos), axis=1)
                    type_labels = np.array(type_names, dtype=object)[best_idx]
                else:
                    type_labels = np.full(pred_idx.shape, None, dtype=object)
                plastic_type_map = np.full_like(
                    floating_prob, None, dtype=object)
                plastic_type_map.reshape(-1)[pred_idx] = type_labels
            except Exception as exc:
                LOGGER.warning("Plastic type attribution failed: %s", exc)
                plastic_type_map = None

        except Exception as exc:
            if not plastic_retrain:
                raise RuntimeError(f"Requested plastic checkpoint inference failed: {exc}") from exc
            LOGGER.warning("Plastic branch failed: %s", exc)
            plastic_prob_map = None
            plastic_mask_pred = None
            plastic_type_map = None

    combined_map = build_combined_map(
        floating_prob, suspended_prob,
        water_mask=water_mask, shp_mask=shp_mask, confident=confident,
        floating_thresh=floating_thresh, suspended_thresh=suspended_thresh,
        plastic_mask_pred=plastic_mask_pred, tie_policy="float",
    )

    # ----------------- 结果 DataFrame -----------------
    def _coords_from_indices(transform, rows, cols):
        a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
        xs = a * cols + b * rows + c
        ys = d * cols + e * rows + f
        return xs, ys

    def _df_from_mask(label, mask_pred, prob_map,
                      ndvi, fdi, ndti, sabi, A_B11, A_B12,
                      transform, crs, water_shape):
        rows, cols = np.where(mask_pred)
        if rows.size == 0:
            return pd.DataFrame(columns=[
                "row", "col", "x", "y", "lon", "lat", "class", "probability",
                "NDVI", "FDI", "NDTI", "SABI", "A_B11", "A_B12", "PlasticProb", "PlasticFlag", "PlasticType"
            ])
        xs, ys = _coords_from_indices(transform, rows, cols)
        if crs:
            try:
                lons, lats = rio_transform(
                    crs, "EPSG:4326", xs.tolist(), ys.tolist())
                lons = np.asarray(lons, dtype=np.float64)
                lats = np.asarray(lats, dtype=np.float64)
            except Exception:
                lons = np.full_like(xs, np.nan, dtype=np.float64)
                lats = np.full_like(ys, np.nan, dtype=np.float64)
        else:
            lons = np.full_like(xs, np.nan, dtype=np.float64)
            lats = np.full_like(ys, np.nan, dtype=np.float64)
        return pd.DataFrame({
            "row": rows.astype(np.int32),
            "col": cols.astype(np.int32),
            "x": xs.astype(np.float64),
            "y": ys.astype(np.float64),
            "lon": lons, "lat": lats,
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
        })

    df_float = _df_from_mask("floating_debris",  floating_mask_pred,  floating_prob,
                             ndvi, fdi, ndti, sabi, A_B11, A_B12, transform, src_crs, water_mask.shape)
    df_susp = _df_from_mask("suspended_matter", suspended_mask_pred, suspended_prob,
                            ndvi, fdi, ndti, sabi, A_B11, A_B12, transform, src_crs, water_mask.shape)
    frames = [d for d in (df_float, df_susp) if not d.empty]
    df = pd.concat(frames, ignore_index=True) if frames else df_float

    if plastic_mask_pred is not None and plastic_prob_map is not None and not df.empty:
        H, W = water_mask.shape
        plast_idx_flat = np.flatnonzero(plastic_mask_pred.reshape(-1))
        if "PlasticFlag" not in df.columns:
            df["PlasticFlag"] = False
        if "PlasticProb" not in df.columns:
            df["PlasticProb"] = np.nan
        if plastic_type_map is not None and "PlasticType" not in df.columns:
            df["PlasticType"] = None

        for cls in ("floating_debris", "suspended_matter"):
            mdf = (df["class"].values == cls)
            if not mdf.any():
                continue
            rows = df.loc[mdf, "row"].to_numpy(dtype=np.int64, copy=False)
            cols = df.loc[mdf, "col"].to_numpy(dtype=np.int64, copy=False)
            np.clip(rows, 0, H-1, out=rows)
            np.clip(cols, 0, W-1, out=cols)
            flat_idx = np.ravel_multi_index(
                (rows, cols), dims=(H, W), mode="clip")
            hit_local = np.isin(flat_idx, plast_idx_flat)
            idx_all = np.where(mdf)[0]
            probs = plastic_prob_map[rows, cols].astype(np.float32)
            df.loc[idx_all, "PlasticProb"] = probs
            df.loc[idx_all[hit_local], "PlasticFlag"] = (
                probs[hit_local] >= float(plastic_prob_thresh))
            if plastic_type_map is not None:
                df.loc[idx_all, "PlasticType"] = np.asarray(
                    plastic_type_map[rows, cols], dtype=object)

        df["PlasticProb"] = pd.to_numeric(
            df["PlasticProb"], errors="coerce").fillna(0.0).astype(np.float32)
        df["PlasticFlag"] = df["PlasticFlag"].astype("boolean").fillna(False)

    # 保存 CSV & 预览
    LOGGER.info("Writing detections CSV to %s (rows=%d)", output_csv, len(df))
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    pix_area = _pixel_area_from_transform(transform)
    stats: Dict[str, DetectionStats] = {}
    stats["floating"] = DetectionStats("floating", area_m2=float(int(np.count_nonzero(combined_map == 1)) * pix_area),
                                       pixel_count=int(np.count_nonzero(combined_map == 1)))
    stats["suspended"] = DetectionStats("suspended", area_m2=float(int(np.count_nonzero(combined_map == 2)) * pix_area),
                                        pixel_count=int(np.count_nonzero(combined_map == 2)))
    if plastic_mask_pred is not None:
        stats["plastic&floating"] = DetectionStats("plastic&floating",  float(int(
            np.count_nonzero(combined_map == 4)) * pix_area), int(np.count_nonzero(combined_map == 4)))
        stats["plastic&suspended"] = DetectionStats("plastic&suspended", float(int(
            np.count_nonzero(combined_map == 5)) * pix_area), int(np.count_nonzero(combined_map == 5)))

    rgba = np.dstack([rgb_preview, np.where(inter, 1.0, 0.0)])
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
        mask=np.where(inter[..., None], rgb_preview, np.nan),
        output_png=preview_png,
    )

    return df


# -------------------- Utilities --------------------
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
    if filter_query:
        gdf = gdf.query(filter_query)
    if filter_field is not None and filter_values is not None:
        if filter_field not in gdf.columns:
            raise KeyError(
                f"Field {filter_field} not in shapefile: {list(gdf.columns)}")
        gdf = gdf[gdf[filter_field].isin(list(filter_values if isinstance(
            filter_values, (list, tuple, set, np.ndarray, pd.Series)) else [filter_values]))]
    if filter_indices is not None:
        idxs = [int(filter_indices)] if isinstance(
            filter_indices, (int, np.integer)) else list(filter_indices)
        gdf = gdf.iloc[idxs]
    gdf = gdf[gdf.geometry.notnull()]
    if gdf.empty:
        raise ValueError("Empty GeoDataFrame after filtering.")

    gdf_original = gdf.copy()
    if target_crs is not None:
        try:
            gdf = gdf.to_crs(target_crs)
        except Exception as exc:
            LOGGER.warning(
                "Failed to reproject shapefile to raster CRS %s: %s", target_crs, exc)

    mask = rasterize([(geom, 1) for geom in gdf.geometry],
                     out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8)

    if not np.any(mask):
        fallback_gdf, warning_msg = _fallback_project_shapefile(
            gdf_original, target_crs, transform)
        if fallback_gdf is not None:
            mask = rasterize([(geom, 1) for geom in fallback_gdf.geometry],
                             out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8)
            if np.any(mask):
                gdf = fallback_gdf
                LOGGER.warning(warning_msg)
        if not np.any(mask):
            raise ValueError(
                "Shapefile mask is empty after rasterization (CRS mismatch or non-overlap).")

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
    mode: str = "quick",  # "quick" | "advanced"
    detection_kwargs: Optional[Dict] = None,
    **filters,
) -> Dict[str, pd.DataFrame]:
    tif_dir = Path(tif_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, pd.DataFrame] = {}
    allowed_filter_keys = {"filter_field",
                           "filter_values", "filter_query", "filter_indices"}
    filter_args = {k: v for k, v in filters.items(
    ) if k in allowed_filter_keys and v is not None}
    detection_kwargs = detection_kwargs or {}

    for tif_path in sorted(tif_dir.glob("*.tif")):
        csv_path = output_dir / f"{tif_path.stem}.csv"
        png_path = output_dir / f"{tif_path.stem}.png"
        if mode == "advanced":
            mask_tif = detection_kwargs.get("mask_tif")
            if not mask_tif:
                raise ValueError(
                    "Advanced mode requires 'mask_tif' in detection_kwargs")
            per_scene = {k: v for k, v in detection_kwargs.items() if k not in {
                "mask_tif", "log_dir", "debug_dir"}}
            log_dir = detection_kwargs.get("log_dir")
            if log_dir and "log_file" not in per_scene:
                per_scene["log_file"] = Path(log_dir) / f"{tif_path.stem}.log"
            debug_root = detection_kwargs.get("debug_dir")
            if debug_root and "debug_dir" not in per_scene:
                per_scene["debug_dir"] = Path(debug_root) / tif_path.stem
            df = floating_debris_detection(
                input_tif=tif_path,
                mask_tif=mask_tif,
                shp_file=shp_file,
                output_csv=csv_path,
                preview_png=png_path,
                **filter_args,
                **per_scene,
            )
        else:
            per_scene = {k: v for k, v in detection_kwargs.items() if k not in {
                "log_dir", "debug_dir"}}
            log_dir = detection_kwargs.get("log_dir")
            if log_dir and "log_file" not in per_scene:
                per_scene["log_file"] = Path(log_dir) / f"{tif_path.stem}.log"
            debug_root = detection_kwargs.get("debug_dir")
            if debug_root and "debug_dir" not in per_scene:
                per_scene["debug_dir"] = Path(debug_root) / tif_path.stem
            df = floating_debris_detection(
                input_tif=tif_path,
                shp_file=shp_file,
                output_csv=csv_path,
                preview_png=png_path,
                **filter_args,
                **per_scene,
            )
        results[tif_path.name] = df
    return results


__all__ = [
    "DetectionStats",
    "read_bands",
    "compute_ndwi",
    "compute_fdi",
    "compute_ndti",
    "compute_sabi",
    "mask_to_vector",
    "visualize_results",
    "build_combined_map",
    "floating_debris_detection",
    "create_mask_tif_from_shp",
    "batch_detect_directory",
    "_save_plastic_feature_correlations",
    "_save_plastic_spectral_profiles",
    # optional export
    "build_plastic_feature_matrix_sparse",
]
