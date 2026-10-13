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

# ---- Matplotlib: force non-interactive backend BEFORE importing pyplot ----
import matplotlib
matplotlib.use("Agg")  # avoid GUI/blocking, faster in headless/multiprocessing
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from .spectral_tools import S2_BANDS, band_depth
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
    array: np.ndarray,
    *,
    cmap: str = "viridis",
    title: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
) -> None:
    if not debug_dir:
        return
    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    data = array.copy()
    if mask is not None:
        data = np.where(mask, data, np.nan)
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
    rgb: np.ndarray,
    overlay: np.ndarray,
    *,
    title: str,
    color: str,
    alpha: float = 0.4,
    mask: Optional[np.ndarray] = None,
) -> None:
    if not debug_dir:
        return
    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb)
    overlay_mask = overlay.astype(bool)
    if mask is not None:
        overlay_mask &= mask
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
    data: np.ndarray,
    *,
    title: str,
    mask: Optional[np.ndarray] = None,
    bins: int = 50,
) -> None:
    if not debug_dir:
        return
    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)
    valid = data.copy()
    if mask is not None:
        valid = valid[mask]
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


def _save_plastic_spectral_profiles(
    debug_dir: Optional[Path | str],
    feat_names: Sequence[str],
    features: np.ndarray,
    labels: np.ndarray,
    *,
    sample_limit: int = 20000,
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
        pos_mean = np.nanmean(pos_spec, axis=0)
        neg_mean = np.nanmean(neg_spec, axis=0)
        pos_std = np.nanstd(pos_spec, axis=0)
        neg_std = np.nanstd(neg_spec, axis=0)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(wavelengths, pos_mean, label="Plastic samples", color="#d62728")
        ax.fill_between(wavelengths, pos_mean - pos_std, pos_mean + pos_std, color="#d62728", alpha=0.2)
        ax.plot(wavelengths, neg_mean, label="Background samples", color="#1f77b4")
        ax.fill_between(wavelengths, neg_mean - neg_std, neg_mean + neg_std, color="#1f77b4", alpha=0.2)
        ax.set_xlabel("Wavelength (nm or index)")
        ax.set_ylabel("Reflectance")
        ax.set_title("Average spectral profiles (candidate samples)")
        ax.legend()
        ax.grid(alpha=0.2)
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


# -------------------- Visualization (with point throttling) --------------------
def visualize_results(
    rgb_image: np.ndarray,
    floating_map: np.ndarray,
    suspended_map: np.ndarray,
    plastic_map: np.ndarray,
    class_map: np.ndarray,
    df: pd.DataFrame,
    stats: Dict[str, DetectionStats],
    output_png: Path | str,
    *,
    max_scatter_points: int = 20000,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    ax_float_prob, ax_susp_prob, ax_plastic_prob = axes[0]
    ax_float_overlay, ax_susp_plastic_overlay, ax_final = axes[1]

    # ---- 概率图（上排）----
    im = ax_float_prob.imshow(floating_map, cmap="magma", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax_float_prob, label="Floating debris prob")
    ax_float_prob.set_title("Floating debris probability"); ax_float_prob.axis("off")

    im = ax_susp_prob.imshow(suspended_map, cmap="magma", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax_susp_prob, label="Suspended matter prob")
    ax_susp_prob.set_title("Suspended matter probability"); ax_susp_prob.axis("off")

    im = ax_plastic_prob.imshow(plastic_map, cmap="magma", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax_plastic_prob, label="Plastic prob")
    ax_plastic_prob.set_title("Plastic probability"); ax_plastic_prob.axis("off")

    # ---- 叠加底图 ----
    rgb_disp = rgb_image if rgb_image.ndim == 3 else np.dstack([rgb_image] * 3)
    rgb_disp = np.clip(rgb_disp, 0, 1)

    # ==== 左下：漂浮相关（1=漂浮非塑料，4=塑料∧漂浮）====
    ax_float_overlay.imshow(rgb_disp)
    mask_float_only = (class_map == 1)
    mask_plastic_float = (class_map == 4)
    # 先画非塑料，再画与塑料重叠（后者覆盖更醒目）
    
    # ==== 右下：最终分类（0~5 离散色标）====
    classes = [0, 1, 2, 3, 4, 5]
    labels = [
        "background",
        "floating",
        "suspended",
        "plastic",
        "plastic&floating",
        "plastic&suspended",
    ]
    colors = ["white", "orange", "magenta", "#ff8fab", "#ffd166", "#7b2cbf"]
    boundaries = np.arange(-0.5, 6.5, 1.0)
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)
    ax_float_overlay.imshow(
        np.where(mask_float_only, 1.0, np.nan),
        cmap=mcolors.ListedColormap(["orange"]),
        alpha=0.45,
        vmin=0,
        vmax=1,
    )
    ax_float_overlay.imshow(
        np.where(mask_plastic_float, 1.0, np.nan),
        cmap=mcolors.ListedColormap(["#ffd166"]),
        alpha=0.55,
        vmin=0,
        vmax=1,
    )  # 黄
    # 小样本时叠加散点作为参考（沿用 df['class'] 三类）
    if not df.empty and len(df) <= max_scatter_points:
        color_lookup = {"floating_debris": "orange",
                        "suspended_matter": "cyan",
                        "plastic_ml": "magenta"}
        # 画序：漂浮 → 塑料 → 悬浮（避免遮蔽）
        draw_order = [c for c in ["floating_debris", "plastic_ml", "suspended_matter"]
                      if c in df["class"].unique()]
        for cls in draw_order:
            g = df[df["class"] == cls]
            ax_float_overlay.scatter(g["col"], g["row"], s=3,
                                     c=color_lookup.get(cls, "orange"),
                                     label=cls, zorder=5)
        if draw_order:
            ax_float_overlay.legend(loc="upper right", frameon=True)
    ax_float_overlay.set_title("Floating overlays (non-plastic & plastic∧floating)")
    ax_float_overlay.axis("off")

    # ax_susp_plastic_overlay.imshow(rgb_disp)
    mask_sus_only = (class_map == 2)
    mask_plastic_suspend = (class_map == 5)
    # ==== 中下：悬浮 & 塑料（2=悬浮非塑料，3=纯塑料，5=塑料∧悬浮）====
    ax_susp_plastic_overlay.imshow(rgb_disp)
    ax_susp_plastic_overlay.imshow(
        np.where(mask_sus_only, 1.0, np.nan),
        cmap=mcolors.ListedColormap(["magenta"]),
        alpha=0.50,
        vmin=0,
        vmax=1,
    )
    ax_susp_plastic_overlay.imshow(
        np.where(mask_plastic_suspend, 1.0, np.nan),
        cmap=mcolors.ListedColormap(["#7b2cbf"]),
        alpha=0.55,
        vmin=0,
        vmax=1,
    )
    ax_susp_plastic_overlay.set_title("Suspended & plastic overlays")
    ax_susp_plastic_overlay.axis("off")
    cat = np.zeros_like(class_map, dtype=np.uint8)
    cat[class_map == 4] = 1
    cat[class_map == 5] = 2

    # 把 0 遮掉，这样背景不着色
    cmap = ListedColormap(['white', "#ff8fab", "#ffd166"])   # 0 透明
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], ncolors=3)
    im = ax_final.imshow(cat, cmap=cmap, norm=norm, alpha=0.7)
    cbar = plt.colorbar(im, ax=ax_final, ticks=[0,1,2])
    cbar.ax.set_yticklabels(["background","plastic&floating","plastic&suspended"])
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


def plot_sateline(
    input_tif: Path | str,
    shp_file: Path | str,
    output_csv: Path | str,
    preview_png: Path | str,
    band_order: Sequence[str] = ("B2", "B3", "B4", "B8", "B8A", "B11", "B12"),
    filter_field: Optional[str] = None,
    filter_values: Optional[Sequence] = None,
    filter_query: Optional[str] = None,
    filter_indices: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    band_idx = {band: idx for idx, band in enumerate(band_order)}

    gdf = gpd.read_file(shp_file)
    gdf = filter_gdf(gdf, filter_field, filter_values, filter_query, filter_indices)

    with rasterio.open(input_tif) as src:
        gdf = gdf.to_crs(src.crs)
        try:
            aoi_geometry = gdf.geometry.union_all()
        except AttributeError:
            aoi_geometry = gdf.geometry.unary_union
        transform = src.transform
        src_crs = src.crs
        bands_data = {
            band: src.read(idx + 1).astype("float32") * 0.0001
            for band, idx in band_idx.items()
        }

    B2, B3, B4, B8, B11, B12 = (
        bands_data["B2"],
        bands_data["B3"],
        bands_data["B4"],
        bands_data["B8"],
        bands_data["B11"],
        bands_data["B12"],
    )

    ndwi = (B3 - B8) / (B3 + B8 + 1e-6)
    water_mask = ndwi > 0

    ndvi = (B8 - B4) / (B8 + B4 + 1e-6)
    fdi = B11 - (B4 + B8) / 2.0

    fdi_thresh = np.nanmean(fdi[water_mask]) + 1.5 * np.nanstd(fdi[water_mask])
    ndvi_thresh = 0.1

    B11[B11 <= 0] = np.nan
    B12[B12 <= 0] = np.nan
    A_B11 = -np.log10(B11)
    A_B12 = -np.log10(B12)

    absorbance_thresh = 0.8
    absorb_mask = (A_B11 > absorbance_thresh) & (A_B12 > absorbance_thresh)

    water_plastic_mask = water_mask & (ndvi < ndvi_thresh) & (fdi > fdi_thresh) & absorb_mask

    rows, cols = np.where(water_plastic_mask)
    xs, ys = rasterio.transform.xy(transform, rows, cols)

    records = []
    for r, c, x, y in zip(rows, cols, xs, ys):
        point = Point(x, y)
        if point.within(aoi_geometry):
            if src_crs:
                try:
                    lon, lat = rio_transform(src_crs, "EPSG:4326", [x], [y])
                    lon = float(lon[0])
                    lat = float(lat[0])
                except Exception:
                    lon = float("nan")
                    lat = float("nan")
            else:
                lon = float("nan")
                lat = float("nan")
            records.append(
                {
                    "row": int(r),
                    "col": int(c),
                    "x": float(x),
                    "y": float(y),
                    "lon": lon,
                    "lat": lat,
                    "NDVI": float(ndvi[r, c]),
                    "FDI": float(fdi[r, c]),
                    "A_B11": float(A_B11[r, c]),
                    "A_B12": float(A_B12[r, c]),
                }
            )

    df = pd.DataFrame(records)
    output_csv = Path(output_csv)
    preview_png = Path(preview_png)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    rgb = np.dstack(
        [
            np.clip(B4 / np.nanmax(B4), 0, 1),
            np.clip(B3 / np.nanmax(B3), 0, 1),
            np.clip(B2 / np.nanmax(B2), 0, 1),
        ]
    )

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    if len(records) > 0:
        plt.scatter([rec["col"] for rec in records], [rec["row"] for rec in records], s=5, c="orange")
    plt.title("High Likelihood Plastic in Water (Absorbance + AOI)")
    plt.axis("off")
    plt.savefig(preview_png, dpi=200, bbox_inches="tight")
    plt.close()

    return df


def floating_debris_detection(
    input_tif: Path | str,
    mask_tif: Optional[Path | str] = None,
    *,
    output_csv: Path | str,
    preview_png: Path | str,
    shp_file: Optional[Path | str] = None,
    band_order: Sequence[str] = (
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B11",
        "B12",
    ),
    ndvi_thresh: float = 0.1,
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
) -> pd.DataFrame:
    _require_torch()
    output_csv = Path(output_csv)
    preview_png = Path(preview_png)
    plastic_model_path = Path(plastic_model_path) if plastic_model_path else None
    plastic_library_csv = Path(plastic_library_csv) if plastic_library_csv else None
    _ensure_stream_logger()
    handler = _configure_logger(log_file)
    if handler:
        LOGGER.info("Logging to %s", handler.baseFilename)

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
                with rasterio.open(input_tif) as src_ref:
                    if src_ref.crs is not None:
                        gdf = gdf_filtered.to_crs(src_ref.crs)
                    else:
                        LOGGER.warning("Input raster %s has no CRS; using shapefile CRS directly for masking.", input_tif)
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
        msg = "No shapefile provided; falling back to automatic water masking."
        LOGGER.info(msg)
        tqdm.write(msg)

    full_band_order = ("B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12",)
    compact_band_order = ("B2","B3","B4","B5","B8","B8A","B9","B11","B12")
    minimal_band_order = ("B2", "B3", "B4", "B8",  "B8A", "B11", "B12")

    band_order = tuple(band_order) if band_order else full_band_order

    with rasterio.open(input_tif) as src_tmp:
        available_count = src_tmp.count
        raw_descriptions = src_tmp.descriptions or []
        descriptions = [desc.strip().upper() for desc in raw_descriptions if isinstance(desc, str)]

    known_bands = set(S2_BANDS.keys())
    if descriptions and all(desc in known_bands for desc in descriptions):
        band_order = tuple(descriptions)
    else:
        candidate_orders = []
        if band_order:
            candidate_orders.append(tuple(band_order))
        candidate_orders.extend([full_band_order, compact_band_order, minimal_band_order])
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for order in candidate_orders:
            if order not in seen:
                unique_candidates.append(order)
                seen.add(order)

        chosen = None
        for order in unique_candidates:
            if len(order) == available_count:
                chosen = order
                break

        if chosen is None:
            if available_count < len(minimal_band_order):
                raise ValueError(
                    f"Unsupported band configuration: file has {available_count} bands which is fewer than "
                    f"the minimal required {len(minimal_band_order)} bands ({minimal_band_order})."
                )
            base = unique_candidates[0]
            chosen = tuple(base[:available_count])
            LOGGER.warning(
                "Input has %s bands with unknown layout; assuming first %s bands of %s -> %s",
                available_count,
                available_count,
                base,
                chosen,
            )
        band_order = tuple(chosen)

    band_indices = list(range(1, len(band_order) + 1))
    bands_raw, transform, src_meta = read_bands(input_tif, band_indices)
    band_cube = bands_raw.astype(np.float32) * 0.0001
    src_crs = src_meta.get("crs") if isinstance(src_meta, dict) else getattr(src_meta, "get", lambda *_: None)("crs")
    band_map = {band: idx for idx, band in enumerate(band_order)}

    required_bands = {"B2", "B3", "B4", "B8", "B8A", "B11", "B12"}
    missing_required = required_bands.difference(band_map)
    if missing_required:
        raise ValueError(
            "Input raster is missing required bands for detection. "
            f"Missing bands: {sorted(missing_required)}."
        )

    # Enable plastic only if a library CSV is provided (and exists).
    # 仅当库 CSV 存在时启用塑料训练/推理
    plastic_enabled = plastic_library_csv is not None and Path(plastic_library_csv).exists()
    if plastic_library_csv is not None and not Path(plastic_library_csv).exists():
        LOGGER.warning("Skipping plastic ML inference (library CSV missing: %s)", plastic_library_csv)
        plastic_enabled = False


    def require_band(name: str) -> np.ndarray:
        if name not in band_map:
            raise KeyError(f"Band {name} is required but not present in band_order: {band_order}")
        return band_cube[:, :, band_map[name]]

    if gdf is not None:
        def _rasterize_from_gdf(gdf_for_mask: gpd.GeoDataFrame) -> np.ndarray:
            return rasterize(
                [(geom, 1) for geom in gdf_for_mask.geometry],
                out_shape=(band_cube.shape[0], band_cube.shape[1]),
                transform=transform,
                fill=0,
                dtype=np.uint8,
            ).astype(bool)

        with tqdm(total=1, desc="Creating shapefile mask") as pbar:
            shp_mask = _rasterize_from_gdf(gdf)
            pbar.update(1)

        if not np.any(shp_mask) and gdf_original is not None:
            fallback_gdf, alignment_msg = _fallback_project_shapefile(gdf_original, src_crs, transform)
            if fallback_gdf is not None:
                with tqdm(total=1, desc="Reprojecting shapefile for mask") as pbar:
                    shp_mask = _rasterize_from_gdf(fallback_gdf)
                    pbar.update(1)
                if np.any(shp_mask):
                    gdf = fallback_gdf
                    shapefile_alignment_warning = alignment_msg
                    try:
                        aoi_geometry = gdf.geometry.union_all()
                    except AttributeError:
                        aoi_geometry = gdf.geometry.unary_union

        if not np.any(shp_mask):
            LOGGER.warning("Shapefile mask is empty after rasterize; reverting to full-scene mask.")
            shp_mask = np.ones((band_cube.shape[0], band_cube.shape[1]), dtype=bool)
        elif shapefile_alignment_warning:
            LOGGER.warning(shapefile_alignment_warning)
    else:
        shp_mask = np.ones((band_cube.shape[0], band_cube.shape[1]), dtype=bool)

    with tqdm(total=1, desc="Processing bands") as pbar:
        B2 = require_band("B2")
        B3 = require_band("B3")
        B4 = require_band("B4")
        B5 = band_cube[:, :, band_map["B5"]] if "B5" in band_map else None
        B6 = band_cube[:, :, band_map["B6"]] if "B6" in band_map else None
        B7 = band_cube[:, :, band_map["B7"]] if "B7" in band_map else None
        B8 = require_band("B8")
        B8A = band_cube[:, :, band_map["B8A"]] if "B8A" in band_map else None
        B9 = band_cube[:, :, band_map["B9"]] if "B9" in band_map else None
        B11 = require_band("B11")
        B12 = require_band("B12")
        pbar.update(1)

    band_arrays = [
        ("B2", B2),
        ("B3", B3),
        ("B4", B4),
        ("B5", B5),
        ("B6", B6),
        ("B7", B7),
        ("B8", B8),
        ("B8A", B8A),
        ("B9", B9),
        ("B11", B11),
        ("B12", B12),
    ]
    for _, arr in band_arrays:
        if arr is not None:
            arr[~shp_mask] = np.nan

    rgb_preview = np.dstack(
        [
            np.clip(B4 / np.nanmax(B4, where=shp_mask, initial=1), 0, 1),
            np.clip(B3 / np.nanmax(B3, where=shp_mask, initial=1), 0, 1),
            np.clip(B2 / np.nanmax(B2, where=shp_mask, initial=1), 0, 1),
        ]
    )

    with tqdm(total=1, desc="Computing NDWI") as pbar:
        ndwi = compute_ndwi(B3, B8)
        water_mask = (ndwi > water_ndwi_thresh) & shp_mask
        if not np.any(water_mask):
            valid_ndwi = ndwi[shp_mask]
            adaptive_thresh = np.nanpercentile(valid_ndwi, 75)
            if np.isnan(adaptive_thresh):
                adaptive_thresh = np.nanmedian(valid_ndwi)
            if np.isnan(adaptive_thresh):
                adaptive_thresh = water_ndwi_thresh
            water_mask = (ndwi > adaptive_thresh) & shp_mask
            if not np.any(water_mask):
                water_mask = shp_mask.copy()
            LOGGER.warning(
                "NDWI threshold %.3f produced no water; fallback %.3f (pixels=%d)",
                water_ndwi_thresh,
                adaptive_thresh,
                int(np.count_nonzero(water_mask)),
            )
        pbar.update(1)

    _save_debug_image(
        debug_dir,
        "ndwi_map.png",
        ndwi,
        title=f"NDWI (thr={water_ndwi_thresh})",
        cmap="coolwarm",
        mask=shp_mask,
        vmin=-0.5,
        vmax=0.5,
    )
    _save_debug_overlay(
        debug_dir,
        "water_mask_overlay.png",
        rgb_preview,
        water_mask,
        title="Detected water mask",
        color="cyan",
        mask=shp_mask,
    )

    with tqdm(total=3, desc="Computing spectral indices") as pbar:
        ndvi = (B8 - B4) / (B8 + B4 + 1e-6)
        ndvi[~shp_mask] = np.nan
        pbar.update(1)
        fdi = compute_fdi(B4, B8, B11)
        fdi[~shp_mask] = np.nan
        valid_fdi = fdi[water_mask]
        if valid_fdi.size == 0:
            valid_fdi = fdi[shp_mask]
        fdi_mean = np.nanmean(valid_fdi)
        fdi_std = np.nanstd(valid_fdi)
        if np.isnan(fdi_mean):
            fdi_mean = 0.0
        if np.isnan(fdi_std):
            fdi_std = 0.0
        fdi_thresh = fdi_mean + fdi_std_factor * fdi_std
        pbar.update(1)
        ndti = compute_ndti(B4, B3)
        ndti[~shp_mask] = np.nan
        valid_ndti = ndti[water_mask]
        if valid_ndti.size == 0:
            valid_ndti = ndti[shp_mask]
        ndti_mean = np.nanmean(valid_ndti)
        ndti_std = np.nanstd(valid_ndti)
        if np.isnan(ndti_mean):
            ndti_mean = 0.0
        if np.isnan(ndti_std):
            ndti_std = 0.0
        ndti_thresh_val = ndti_mean + ndti_std_factor * ndti_std
        pbar.update(1)

    _save_debug_image(
        debug_dir,
        "ndvi_map.png",
        ndvi,
        title="NDVI",
        cmap="RdYlGn",
        mask=water_mask,
        vmin=-1,
        vmax=1,
    )
    _save_debug_image(
        debug_dir,
        "fdi_map.png",
        fdi,
        title=f"FDI (thr={fdi_thresh:.3f})",
        cmap="plasma",
        mask=water_mask,
    )
    _save_debug_image(
        debug_dir,
        "ndti_map.png",
        ndti,
        title=f"NDTI (thr={ndti_thresh_val:.3f})",
        cmap="viridis",
        mask=water_mask,
    )

    with tqdm(total=2, desc="Computing absorbance") as pbar:
        B11[B11 <= 0] = np.nan
        B12[B12 <= 0] = np.nan
        A_B11 = -np.log10(B11)
        A_B12 = -np.log10(B12)
        A_B11[~shp_mask] = np.nan
        A_B12[~shp_mask] = np.nan
        absorb_mask = (A_B11 > absorbance_thresh) & (A_B12 > absorbance_thresh) & shp_mask
        pbar.update(2)

    _save_debug_image(
        debug_dir,
        "absorbance_B11.png",
        A_B11,
        title="Absorbance B11",
        cmap="magma",
        mask=water_mask,
    )
    _save_debug_image(
        debug_dir,
        "absorbance_B12.png",
        A_B12,
        title="Absorbance B12",
        cmap="magma",
        mask=water_mask,
    )

    if mask_tif is not None and Path(mask_tif).exists():
        with tqdm(total=1, desc="Reading mask") as pbar:
            with rasterio.open(mask_tif) as msk:
                mask = msk.read(1)
            pbar.update(1)

        ref_shape = water_mask.shape
        with tqdm(total=1, desc="Resizing mask") as pbar:
            mask = resize_to_reference(mask, ref_shape)
            mask[~shp_mask] = 0
            pbar.update(1)
    else:
        mask = water_mask.astype(np.uint8)
        LOGGER.info("No mask_tif provided; using NDWI-derived water mask for seed generation.")

    floating_seed = water_mask & (mask == 1) & (ndvi < ndvi_thresh) & (fdi > fdi_thresh) & absorb_mask

    def _percentile_safe(arr: np.ndarray, mask: np.ndarray, q: float, fallback_mask: np.ndarray) -> float:
        vals = arr[mask]
        if vals.size == 0 or np.all(np.isnan(vals)):
            vals = arr[fallback_mask]
        value = np.nanpercentile(vals, q)
        if np.isnan(value):
            value = float(np.nanmedian(vals))
        if np.isnan(value):
            value = 0.0
        return float(value)

    red_percentile = _percentile_safe(B4, water_mask, 70, shp_mask)
    green_percentile = _percentile_safe(B3, water_mask, 65, shp_mask)
    print(f"Red 70th percentile: {red_percentile:.4f}, \nGreen 65th percentile: {green_percentile:.4f}")
    suspended_seed = (
        water_mask
        & (ndti > ndti_thresh_val)
        & (B4 > red_percentile)
        & (B3 > green_percentile)
        & (ndwi < 0.1)
    )

    _save_debug_overlay(
        debug_dir,
        "floating_seed_overlay.png",
        rgb_preview,
        floating_seed,
        title="Floating seed mask",
        color="orange",
        mask=water_mask,
    )
    _save_debug_overlay(
        debug_dir,
        "suspended_seed_overlay.png",
        rgb_preview,
        suspended_seed,
        title="Suspended seed mask",
        color="yellow",
        mask=water_mask,
    )

    water_pixels = int(np.count_nonzero(water_mask))
    if water_pixels == 0:
        warn_msg = "No water pixels detected; aborting UNet training."
        LOGGER.warning(warn_msg)
        tqdm.write(warn_msg)
        floating_prob = np.zeros_like(water_mask, dtype=np.float32)
        suspended_prob = np.zeros_like(water_mask, dtype=np.float32)
        combined_map = np.zeros_like(water_mask, dtype=np.uint8)
        df = pd.DataFrame(columns=[
            "row",
            "col",
            "x",
            "y",
            "lon",
            "lat",
            "class",
            "probability",
            "NDVI",
            "FDI",
            "NDTI",
            "SABI",
            "A_B11",
            "A_B12",
            "PlasticProb",
            "PlasticFlag",
        ])
        stats = {
            "floating_debris": compute_stats("floating_debris", floating_prob.astype(bool)),
            "suspended_matter": compute_stats("suspended_matter", suspended_prob.astype(bool)),
        }
        rgb = np.dstack(
            [
                np.clip(B4 / np.nanmax(B4, where=shp_mask, initial=1), 0, 1),
                np.clip(B3 / np.nanmax(B3, where=shp_mask, initial=1), 0, 1),
                np.clip(B2 / np.nanmax(B2, where=shp_mask, initial=1), 0, 1),
            ]
        )
        visualize_results(
            rgb_image=rgb,
            floating_map=floating_prob,
            suspended_map=suspended_prob,
            plastic_map=np.zeros_like(floating_prob),
            class_map=combined_map,
            df=df,
            stats=stats,
            output_png=preview_png,
        )
        _save_debug_hist(
            debug_dir,
            "floating_prob_hist.png",
            floating_prob,
            title="Floating probability histogram",
        )
        _save_debug_hist(
            debug_dir,
            "suspended_prob_hist.png",
            suspended_prob,
            title="Suspended probability histogram",
        )
        _save_debug_image(
            debug_dir,
            "floating_prob_map.png",
            floating_prob,
            title="Floating probability map",
            cmap="magma",
        )
        _save_debug_image(
            debug_dir,
            "suspended_prob_map.png",
            suspended_prob,
            title="Suspended probability map",
            cmap="viridis",
        )
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        if handler:
            handler.flush()
            LOGGER.removeHandler(handler)
            handler.close()
        return df

    floating_seed_count = int(np.count_nonzero(floating_seed))
    suspended_seed_count = int(np.count_nonzero(suspended_seed))
    aoi_pixels = water_pixels
    seed_msg = (
        f"AOI pixels: {aoi_pixels:,} | floating seeds: {floating_seed_count:,} | "
        f"suspended seeds: {suspended_seed_count:,}"
    )
    LOGGER.info(seed_msg)
    tqdm.write(seed_msg)

    sabi = compute_sabi(B8, B4)
    sabi[~shp_mask] = np.nan
    bd_b9 = None
    if B9 is not None and B8A is not None:
        bd_b9 = band_depth(
            band_value=B9,
            left_value=B8A,
            right_value=B11,
            left_center=865.0,
            right_center=1610.0,
            band_center=945.0,
        )
        bd_b9[~shp_mask] = np.nan

    derived_indices_map = {
        "NDWI": ndwi,
        "NDVI": ndvi,
        "FDI": fdi,
        "NDTI": ndti,
        "SABI": sabi,
        "A_B11": A_B11,
        "A_B12": A_B12,
    }
    if bd_b9 is not None:
        derived_indices_map["BD_B9"] = bd_b9
    if B5 is not None:
        derived_indices_map["rho_B5"] = B5
    if B9 is not None:
        derived_indices_map["rho_B9"] = B9

    with tqdm(total=1, desc="Preparing tensors") as pbar:
        feature_list = [B2, B3, B4]
        if B5 is not None:
            feature_list.append(B5)
        feature_list.extend([B8])
        if B8A is not None:
            feature_list.append(B8A)
        if B9 is not None:
            feature_list.append(B9)
        feature_list.extend(
            [
                B11,
                B12,
                ndwi,
                ndvi,
                fdi,
                ndti,
                sabi,
                A_B11,
                A_B12,
            ]
        )
        if bd_b9 is not None:
            feature_list.append(bd_b9)
        features = np.stack(feature_list, axis=-1).astype(np.float32)
        features[np.isnan(features)] = 0.0
        label_stack = np.stack(
            [floating_seed.astype(np.float32), suspended_seed.astype(np.float32)], axis=0
        )
        if label_stack.ndim == 2:
            label_stack = label_stack[np.newaxis, ...]
        elif label_stack.ndim == 3 and label_stack.shape[0] > 8 and label_stack.shape[-1] <= 8:
            # Handle masks saved as HWC by forcing channels-first layout
            label_stack = np.transpose(label_stack, (2, 0, 1))
        pbar.update(1)

    if floating_seed_count == 0 and suspended_seed_count == 0:
        warn_msg = "No pseudo-labels available inside shapefile; skipping UNet training."
        LOGGER.warning(warn_msg)
        tqdm.write(warn_msg)
        floating_prob = floating_seed.astype(np.float32)
        suspended_prob = suspended_seed.astype(np.float32)
        combined_map = np.zeros_like(floating_prob, dtype=np.uint8)
        df = pd.DataFrame(columns=[
            "row",
            "col",
            "x",
            "y",
            "lon",
            "lat",
            "class",
            "probability",
            "NDVI",
            "FDI",
            "NDTI",
            "SABI",
            "A_B11",
            "A_B12",
            "PlasticProb",
            "PlasticFlag",
            "PlasticType",
        ])
        stats = {
            "floating_debris": compute_stats("floating_debris", floating_seed),
            "suspended_matter": compute_stats("suspended_matter", suspended_seed),
        }
        rgb = np.dstack(
            [
                np.clip(B4 / np.nanmax(B4, where=water_mask, initial=1), 0, 1),
                np.clip(B3 / np.nanmax(B3, where=water_mask, initial=1), 0, 1),
                np.clip(B2 / np.nanmax(B2, where=water_mask, initial=1), 0, 1),
            ]
        )
        visualize_results(
            rgb_image=rgb,
            floating_map=floating_prob,
            suspended_map=suspended_prob,
            plastic_map=np.zeros_like(floating_prob),
            class_map=combined_map,
            df=df,
            stats=stats,
            output_png=preview_png,
        )
        _save_debug_hist(
            debug_dir,
            "floating_prob_hist.png",
            floating_prob,
            title="Floating probability histogram",
            mask=water_mask,
        )
        _save_debug_hist(
            debug_dir,
            "suspended_prob_hist.png",
            suspended_prob,
            title="Suspended probability histogram",
            mask=water_mask,
        )
        _save_debug_image(
            debug_dir,
            "floating_prob_map.png",
            floating_prob,
            title="Floating probability map",
            cmap="magma",
            mask=water_mask,
        )
        _save_debug_image(
            debug_dir,
            "suspended_prob_map.png",
            suspended_prob,
            title="Suspended probability map",
            cmap="viridis",
            mask=water_mask,
        )
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        if handler:
            handler.flush()
            LOGGER.removeHandler(handler)
            handler.close()
        return df

    # dataset = FloatingDataset([features], [label_stack], [water_mask.astype(np.float32)])
    if any(v is None for v in (sample_training_patches, FloatingDataset, train_unet_from_patches, infer_unet)):
        raise RuntimeError("train_model utilities are unavailable; reinstall sateline_ml package")

    if torch is None:
        raise RuntimeError("Torch should be available after _require_torch() call")

    if resolve_device is not None:
        selected_device_obj = resolve_device(device)
    else:
        # Fallback: prefer CUDA → MPS → CPU manually
        selected_device_obj = torch.device("cpu")
        requested_obj: Optional[torch.device] = None
        if device is not None:
            try:
                requested_obj = torch.device(device)
            except Exception:
                requested_obj = None
            if requested_obj is not None:
                if requested_obj.type == "cuda" and torch.cuda.is_available():
                    selected_device_obj = requested_obj
                elif requested_obj.type == "mps":
                    try:
                        if getattr(torch.backends, "mps").is_available():  # type: ignore[attr-defined]
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
                    if getattr(torch.backends, "mps").is_available():  # type: ignore[attr-defined]
                        selected_device_obj = torch.device("mps")
                except Exception:
                    selected_device_obj = torch.device("cpu")

    train_device_arg = str(selected_device_obj)
    inference_device_arg = str(selected_device_obj)

    if device is not None:
        try:
            requested_type = torch.device(device).type
        except Exception:
            LOGGER.warning("Invalid device specification %s; using %s instead", device, train_device_arg)
        else:
            if requested_type != selected_device_obj.type:
                LOGGER.warning("Requested device %s unavailable; using %s instead", device, train_device_arg)
    else:
        if selected_device_obj.type != "cpu":
            LOGGER.info("Auto-selected %s device for UNet training and inference", train_device_arg)

    images_list, masks_list, valid_list = sample_training_patches(
        features,
        label_stack,
        water_mask.astype(np.float32),
        tile=tile_size,
        stride=stride,
        min_valid_frac=min_valid_frac,
        max_patches=max_patches,
        pos_frac=pos_frac,
        seed=patch_seed,
    )

    if len(images_list) == 0:
        warn_msg = "No valid training patches were found (try lowering min_valid_frac or using smaller tile)."
        LOGGER.warning(warn_msg); tqdm.write(warn_msg)
        floating_prob = floating_seed.astype(np.float32)
        suspended_prob = suspended_seed.astype(np.float32)
        combined_map = np.zeros_like(floating_prob, dtype=np.uint8)
        df = pd.DataFrame(columns=[
            "row",
            "col",
            "x",
            "y",
            "class",
            "probability",
            "NDVI",
            "FDI",
            "NDTI",
            "SABI",
            "A_B11",
            "A_B12",
            "PlasticProb",
            "PlasticFlag",
        ])
        stats = {
            "floating_debris": compute_stats("floating_debris", floating_seed),
            "suspended_matter": compute_stats("suspended_matter", suspended_seed),
        }
        rgb = np.dstack([
            np.clip(B4 / np.nanmax(B4, where=water_mask, initial=1), 0, 1),
            np.clip(B3 / np.nanmax(B3, where=water_mask, initial=1), 0, 1),
            np.clip(B2 / np.nanmax(B2, where=water_mask, initial=1), 0, 1),
        ])
        visualize_results(
            rgb_image=rgb,
            floating_map=floating_prob,
            suspended_map=suspended_prob,
            plastic_map=np.zeros_like(floating_prob),
            class_map=combined_map,
            df=df,
            stats=stats,
            output_png=preview_png,
        )
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        if handler:
            handler.flush(); LOGGER.removeHandler(handler); handler.close()
        return df

    dataset = FloatingDataset(images_list, masks_list, valid_list)
    print(f"[INFO] training patches: total={len(dataset)}")

    scene_id = Path(input_tif).stem
    model_root = Path(model_out_dir) if model_out_dir is not None else Path.cwd()
    per_scene_dir = model_root / "models" / scene_id

    if repeats and repeats > 1 and train_unet_repeated is not None:
        save_dir = per_scene_dir
        model, _, device_obj, best_val, best_path = train_unet_repeated(
            dataset,
            repeats=repeats,
            val_split=val_split,
            seed=patch_seed,
            epochs=epochs,
            lr=1e-3,
            device=train_device_arg,
            batch_size=1,
            save_best_to=save_dir,
        )
        LOGGER.info("Selected best model with val_loss=%.6f; saved to %s", best_val, save_dir)
    else:
        model, _, device_obj = train_unet_from_patches(
            dataset,
            epochs=epochs,
            lr=1e-3,
            device=train_device_arg,
            batch_size=1,
            pos_weight=None,
        )
        # Save single-run model for bookkeeping
        best_val = float("nan")
        best_path = per_scene_dir / "best_unet.pth"
        try:
            per_scene_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "meta": {"repeats": 1, "epochs": epochs}}, best_path)
        except Exception as exc:
            LOGGER.warning("Could not save single-run weights: %s", exc)

    thresholds_path = Path(output_csv).parent / "thresholds.json"
    stacking_applied = False
    calibration_result: Optional[ThresholdCalibrationResult] = None
    if (
        auto_calibrate_thresholds
        and calibrate_probability_thresholds is not None
        and len(dataset) > 0
    ):
        try:
            if auto_threshold_candidates is None:
                candidate_array = np.linspace(0.2, 0.95, 16, dtype=np.float32)
            else:
                candidate_array = np.asarray(list(auto_threshold_candidates), dtype=np.float32)
            calibration_result = calibrate_probability_thresholds(
                model=model,
                dataset=dataset,
                device=device_obj,
                class_names=("floating_debris", "suspended_matter"),
                candidate_thresholds=candidate_array,
                metric=auto_threshold_metric,
                batch_size=auto_threshold_batch_size,
                max_samples=auto_threshold_max_samples,
                seed=patch_seed,
            )
            floating_thresh = float(calibration_result.thresholds.get("floating_debris", floating_thresh))
            suspended_thresh = float(calibration_result.thresholds.get("suspended_matter", suspended_thresh))
            save_calibration_result(calibration_result, thresholds_path)
            LOGGER.info(
                "Auto-calibrated thresholds (metric=%s): floating=%.3f, suspended=%.3f",
                calibration_result.metric_name,
                floating_thresh,
                suspended_thresh,
            )
        except Exception as calib_exc:
            calibration_result = None
            LOGGER.warning("Threshold calibration failed: %s", calib_exc)

    if calibration_result is None:
        try:
            thresholds_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_payload = {
                "metric": "configured",
                "thresholds": {
                    "floating_debris": float(floating_thresh),
                    "suspended_matter": float(suspended_thresh),
                },
                "per_class_metrics": {},
                "candidate_thresholds": [],
                "class_order": ["floating_debris", "suspended_matter"],
            }
            thresholds_path.write_text(json.dumps(baseline_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            LOGGER.warning("Failed to write baseline threshold summary: %s", exc)

    # Update model registry and global best snapshot
    try:
        registry_path = model_root / "model_registry.csv"
        record = {
            "scene": scene_id,
            "weights_path": str(best_path),
            "best_val_loss": float(best_val) if best_val == best_val else None,
            "epochs": int(epochs),
            "repeats": int(repeats),
            "val_split": float(val_split),
            "device_train": str(train_device_arg),
            "tile_size": int(tile_size),
            "stride": int(stride),
            "max_patches": int(max_patches),
            "min_valid_frac": float(min_valid_frac),
            "pos_frac": float(pos_frac),
            "patch_seed": int(patch_seed),
            "floating_thresh": float(floating_thresh),
            "suspended_thresh": float(suspended_thresh),
            "auto_calibrated": bool(calibration_result is not None),
            "threshold_metric": (
                calibration_result.metric_name if calibration_result else (auto_threshold_metric if auto_calibrate_thresholds else None)
            ),
            "stacking_preset": stacking_preset if stacking_applied else None,
        }
        row_df = pd.DataFrame([record])
        if registry_path.exists():
            prev = pd.read_csv(registry_path)
            combined = pd.concat([prev, row_df], ignore_index=True)
        else:
            combined = row_df
        combined.to_csv(registry_path, index=False, encoding="utf-8-sig")

        # Maintain global best_overall.pth by minimum best_val_loss among valid rows
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
                    LOGGER.info("Updated global best model => %s (val_loss=%.6f)", global_best, float(best_row["best_val_loss"]))
                except Exception as cp_exc:
                    LOGGER.warning("Failed to update global best: %s", cp_exc)
    except Exception as reg_exc:
        LOGGER.warning("Model registry update failed: %s", reg_exc)

    # ===== Inference device selection: prefer CUDA -> MPS -> CPU =====
    preferred = None
    if isinstance(inference_device_arg, str):
        preferred = inference_device_arg
    elif isinstance(train_device_arg, str):
        preferred = train_device_arg
    elif getattr(device_obj, "type", None):
        preferred = getattr(device_obj, "type")

    td = torch.device(_pick_best_device(preferred))  # returns "cuda"/"mps"/"cpu" string
    model = model.to(td)
    inference_device = torch.device(td)
    LOGGER.info("Starting tiled inference on %s", inference_device.type)

    # ===== Memory-aware tiled inference with graceful fallback =====
    try:
        if inference_device.type in ("cuda", "mps"):
            # Larger -> smaller tile sizes
            tile_candidates = (1024, 896, 768, 640, 512, 384, 320) if inference_device.type == "cuda" \
                            else (768, 640, 512, 448, 384, 320)
            for t in tile_candidates:
                try:
                    probs = infer_unet_tiled(
                        model,
                        features,
                        device=inference_device,
                        tile=t,
                        overlap=64,
                        use_amp=True,          # AMP on GPU/MPS
                        show_progress=True,
                    )
                    break
                except RuntimeError as oom:
                    msg = str(oom).lower()
                    if "out of memory" in msg or "mps" in msg or "resource" in msg:
                        try:
                            if inference_device.type == "cuda":
                                torch.cuda.empty_cache()
                            elif inference_device.type == "mps" and hasattr(torch, "mps"):
                                torch.mps.empty_cache()
                        except Exception:
                            pass
                        continue
                    raise
            else:
                LOGGER.warning("GPU/MPS OOM for all tile sizes; falling back to CPU for inference")
                cpu_device = torch.device("cpu")
                model = model.to(cpu_device)
                probs = infer_unet_tiled(
                    model,
                    features,
                    device=cpu_device,
                    tile=768,
                    overlap=64,
                    use_amp=False,
                    show_progress=True,
                )
        else:
            probs = infer_unet_tiled(
                model,
                features,
                device=inference_device,
                tile=768,
                overlap=64,
                use_amp=False,
                show_progress=True,
            )

    except Exception as exc:
        LOGGER.warning("Tiled inference failed (%s); retrying full-image on CPU", exc)
        cpu_device = torch.device("cpu")
        model = model.to(cpu_device)
        probs = infer_unet(model, features, device=cpu_device)

    LOGGER.info("Finished inference")
    floating_prob = probs[0]
    suspended_prob = probs[1] if probs.shape[0] > 1 else np.zeros_like(floating_prob)

    floating_prob[~water_mask] = 0.0
    suspended_prob[~water_mask] = 0.0

    if stacking_preset:
        if run_stacking_pipeline is None or build_tabular_feature_stack is None:
            LOGGER.warning("Stacking utilities unavailable; skipping preset %s", stacking_preset)
        else:
            try:
                stack_stride = max(1, int(stacking_pixel_stride))
                base_mask_flat = np.flatnonzero(water_mask.reshape(-1))
                if stack_stride > 1:
                    rows = base_mask_flat // water_mask.shape[1]
                    cols = base_mask_flat % water_mask.shape[1]
                    stride_keep = (rows % stack_stride == 0) & (cols % stack_stride == 0)
                    mask_flat = base_mask_flat[stride_keep]
                else:
                    mask_flat = base_mask_flat

                if stacking_max_pixels is not None and mask_flat.size > stacking_max_pixels:
                    before_cap = mask_flat.size
                    rng_limit = np.random.default_rng(patch_seed + 1)
                    sel = np.sort(
                        rng_limit.choice(mask_flat.size, size=int(stacking_max_pixels), replace=False)
                    )
                    mask_flat = mask_flat[sel]
                    LOGGER.info(
                        "Stacking pixel cap applied: %d -> %d (max=%d)",
                        before_cap,
                        mask_flat.size,
                        stacking_max_pixels,
                    )

                LOGGER.info(
                    "Stacking candidate pixels: initial=%d, after stride=%d => %d",
                    base_mask_flat.size,
                    stack_stride,
                    mask_flat.size,
                )

                tabular_X, tab_feature_names, stack_indices = build_tabular_feature_stack(
                    band_cube=band_cube,
                    derived_indices=derived_indices_map,
                    band_names=band_order,
                    water_mask=water_mask,
                    kernel_sizes=(3, 5),
                    include_texture=True,
                    flat_indices=mask_flat,
                )
                if tabular_X.shape[0] == 0 or stack_indices.size == 0:
                    LOGGER.warning(
                        "Stacking preset %s skipped: no valid pixels after subsampling.", stacking_preset
                    )
                    stacking_applied = False
                    stack_json = output_csv.parent / "stacking.json"
                    if stack_json.exists():
                        try:
                            stack_json.unlink()
                        except Exception:
                            pass
                else:
                    labels_map = {
                        "floating_debris": floating_seed.reshape(-1)[stack_indices].astype(np.int8),
                        "suspended_matter": suspended_seed.reshape(-1)[stack_indices].astype(np.int8),
                    }
                    precomputed_meta = {
                        "unet_floating": floating_prob.reshape(-1)[stack_indices],
                        "unet_suspended": suspended_prob.reshape(-1)[stack_indices],
                    }
                    if stacking_sample_limit is not None and tabular_X.shape[0] > stacking_sample_limit:
                        rng = np.random.default_rng(patch_seed)
                        train_idx = np.sort(
                            rng.choice(tabular_X.shape[0], size=int(stacking_sample_limit), replace=False)
                        )
                    else:
                        train_idx = np.arange(tabular_X.shape[0])

                    fused_maps = run_stacking_pipeline(
                        stacking_preset,
                        tabular_X=tabular_X,
                        labels=labels_map,
                        probabilities={},
                        precomputed_meta=precomputed_meta,
                        mask_flat=train_idx,
                        random_state=patch_seed,
                        feature_names=tab_feature_names,
                        cv_folds=stacking_cv_folds,
                    )

                    preset_cfg = STACKING_PRESETS.get(stacking_preset.lower(), {}) if STACKING_PRESETS else {}
                    if postprocess_probabilities is not None and preset_cfg.get("postprocess"):
                        fused_maps = postprocess_probabilities(
                            fused_maps,
                            rgb=rgb_preview,
                            postprocess_cfg=preset_cfg.get("postprocess"),
                        )

                    fused_float = floating_prob.reshape(-1)
                    fused_susp = suspended_prob.reshape(-1)
                    fused_float[stack_indices] = fused_maps["floating_debris"]
                    fused_susp[stack_indices] = fused_maps["suspended_matter"]
                    floating_prob = fused_float.reshape(water_mask.shape)
                    suspended_prob = fused_susp.reshape(water_mask.shape)
                    floating_prob[~water_mask] = 0.0
                    suspended_prob[~water_mask] = 0.0
                    stacking_applied = True
                    LOGGER.info(
                        "Stacking preset '%s' applied using %d features -> fused probabilities computed "
                        "(stride=%d, pixels=%d)",
                        stacking_preset,
                        len(tab_feature_names),
                        stack_stride,
                        stack_indices.size,
                    )
                    try:
                        preset_record = STACKING_PRESETS.get(stacking_preset.lower(), {}) if STACKING_PRESETS else {}
                        stacking_meta = {
                            "preset": stacking_preset,
                            "base_models": preset_record.get("base", []),
                            "meta_model": preset_record.get("meta"),
                            "calibrator": preset_record.get("calibrator"),
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
                LOGGER.warning("Stacking preset '%s' failed: %s", stacking_preset, stack_exc)
                stacking_applied = False
                stack_json = output_csv.parent / "stacking.json"
                if stack_json.exists():
                    try:
                        stack_json.unlink()
                    except Exception:
                        pass

    # Persist probability rasters separately for floating vs suspended outputs
    try:
        prob_profile = src_meta.copy()
        prob_profile.update(
            driver="GTiff",
            dtype=rasterio.float32,
            count=1,
            nodata=0.0,
        )
        prob_dir = output_csv.parent
        prob_dir.mkdir(parents=True, exist_ok=True)
        # float_prob_path = prob_dir / f"{output_csv.stem}_floating_prob.tif"
        # susp_prob_path = prob_dir / f"{output_csv.stem}_suspended_prob.tif"
        # with rasterio.open(float_prob_path, "w", **prob_profile) as dst:
        #     dst.write(floating_prob.astype(np.float32), 1)
        # with rasterio.open(susp_prob_path, "w", **prob_profile) as dst:
        #     dst.write(suspended_prob.astype(np.float32), 1)
    except Exception as write_exc:
        LOGGER.warning("Failed to write probability rasters: %s", write_exc)

    # ========= 塑料推理（用预训练或刚训练的模型）=========
    # ======== 稀疏塑料训练/推理（只对候选像素）========
    plastic_prob_map: Optional[np.ndarray] = None
    plastic_mask_pred: Optional[np.ndarray] = None
    plastic_type_map: Optional[np.ndarray] = None

    if plastic_enabled:
        try:
            rng = np.random.default_rng(1234)
            # 1) 候选集合：优先已预测为浮物；其次可用概率阈值扩大一些
            cand_pos_mask = (floating_prob > max(floating_thresh, 0.6)) & water_mask
            # 负样本从“水体且非强阳性”中抽
            cand_neg_mask = (water_mask & (~cand_pos_mask))

            # 上限采样规模（可调）
            max_pos = 120_000
            max_neg = 180_000

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
            sel_cols = (sel_idx %  W).astype(np.int64)
            # 2) 稀疏构特征（在“可用波段子空间”内）；背景用 water_mask 抽样估计
            X_plastic, feat_names, meta = build_plastic_feature_matrix_sparse(
                band_cube, band_order, plastic_library_csv,
                sel_rows, sel_cols,
                background_mask=water_mask, bg_sample=200_000,
            )

            # 3) 伪标注：阳性=候选阳性；阴性=候选阴性（可再加更严格条件）
            y_plastic = np.zeros(sel_idx.size, dtype=np.float32)
            y_plastic[:len(pos_idx)] = 1.0

            # 4) 训练小 MLP（稀疏子集，不会卡）
            model_plastic, _ = train_plastic_model(
                X_plastic, y_plastic,
                epochs=200, lr=1e-3, weight_decay=1e-4,
                device=str(inference_device), verbose=False,
                hidden=(128, 64), dropout=0.1,
            )

            if debug_dir is not None:
                _save_plastic_feature_correlations(debug_dir, feat_names, X_plastic, y_plastic)
                _save_plastic_spectral_profiles(debug_dir, feat_names, X_plastic, y_plastic)

            # 可选：保存权重（如果给了路径）
            if plastic_model_path is not None:
                meta_obj = PlasticModelMetadata(
                    input_dim=X_plastic.shape[1],
                    feature_names=tuple(feat_names),
                    band_order=tuple(band_order),
                )

                try:
                    # --- Evaluate plastic MLP on training data (print accuracy & loss) ---
                    from sklearn.metrics import accuracy_score, log_loss
                    _has_sklearn = True
                except Exception:
                    _has_sklearn = False

                probs_train = predict_plastic(model_plastic, X_plastic, device=str(inference_device)).astype(np.float32)
                y_true = y_plastic.astype(np.int32)
                y_pred = (probs_train >= float(plastic_prob_thresh)).astype(np.int32)

                if _has_sklearn:
                    acc  = float(accuracy_score(y_true, y_pred))
                    loss = float(log_loss(y_true, np.clip(probs_train, 1e-12, 1-1e-12), labels=[0, 1]))
                else:
                    # 手动计算二分类交叉熵
                    p = np.clip(probs_train, 1e-12, 1 - 1e-12)
                    acc  = float((y_true == y_pred).mean())
                    loss = float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
                print(f"[Plastic-ML] Accuracy: {acc:.4f} | LogLoss: {loss:.4f}")
                LOGGER.info("[Plastic-ML] Accuracy: %.4f | LogLoss: %.4f", acc, loss)
                try:
                    save_plastic_model(plastic_model_path, model_plastic, meta_obj)
                except Exception as _exc:
                    LOGGER.warning("Could not save plastic model: %s", _exc)
                    
            # 5) 推理同样只在“浮物候选像素”上；结果映射回整图
            pred_mask = cand_pos_mask  # 只在这上面判断“是否塑料”
            pred_idx = np.where(pred_mask.ravel())[0]
            pred_rows = (pred_idx // W).astype(np.int64)
            pred_cols = (pred_idx %  W).astype(np.int64)

            X_pred, _, meta_pred = build_plastic_feature_matrix_sparse(
                band_cube, band_order, plastic_library_csv,
                pred_rows, pred_cols,
                background_mask=water_mask, bg_sample=200_000,
            )
            prob_vec = predict_plastic(model_plastic, X_pred, device=str(inference_device))

            plastic_prob_map = np.zeros_like(floating_prob, dtype=np.float32)
            plastic_prob_map.ravel()[pred_idx] = prob_vec.astype(np.float32)
            plastic_prob_map[~water_mask] = 0.0

            plastic_mask_pred = (plastic_prob_map > plastic_prob_thresh) & pred_mask

            try:
                endmembers = (meta_pred or {}).get("endmembers") if meta_pred is not None else None
                sensor_order = (meta_pred or {}).get("sensor_band_order") if meta_pred is not None else None
                if endmembers and sensor_order:
                    type_names = list(endmembers.keys())
                    em_matrix = np.stack([np.asarray(endmembers[name], dtype=np.float32) for name in type_names], axis=0)
                    n_bands = em_matrix.shape[1]
                    rho_part = X_pred[:, :n_bands].astype(np.float32)
                    # Compute spectral angle efficiently
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

            _save_debug_image(
                debug_dir, "plastic_prob_map.png", plastic_prob_map,
                title=f"Plastic probability (sparse, thr={plastic_prob_thresh})",
                cmap="inferno", mask=water_mask,
            )
            _save_debug_hist(
                debug_dir, "plastic_prob_hist.png", plastic_prob_map,
                title="Plastic probability histogram (sparse)", mask=pred_mask,
            )

        except Exception as exc:
            LOGGER.warning("Plastic training/init failed: %s", exc)
            plastic_prob_map = None
            plastic_mask_pred = None

    floating_mask_pred = (floating_prob > floating_thresh) & shp_mask
    suspended_mask_pred = (suspended_prob > suspended_thresh) & shp_mask
    # if plastic_mask_pred is not None:
    #     plastic_mask_pred &= suspended_mask_pred
    #     floating_mask_pred &= (~suspended_mask_pred) & (True if plastic_mask_pred is None else ~plastic_mask_pred)

    # 互斥构图：先算基础层，再覆盖重叠层
    # —— 胜者判定（建议严格 >，平手给漂浮；如需相反，改注释行）——
    susp_wins  = suspended_mask_pred & (suspended_prob >  floating_prob)   # 悬浮严格更大
    # susp_wins  = suspended_mask_pred & (suspended_prob >= floating_prob) # 若想平手也判给悬浮，用这行

    float_wins = floating_mask_pred  & (~suspended_mask_pred | (floating_prob >= suspended_prob))
    # 若上面用了 >= 给悬浮，则这里用 > 给漂浮，保证互斥：
    # float_wins = floating_mask_pred & (~suspended_mask_pred | (floating_prob > suspended_prob))

    combined_map = np.zeros_like(floating_mask_pred, dtype=np.uint8)

    if plastic_mask_pred is None:
        # 非塑料：1=漂浮胜，2=悬浮胜（两者互斥）
        combined_map[float_wins] = 1
        combined_map[susp_wins]  = 2
    else:
        not_plast = ~plastic_mask_pred

        # --- 基础层（非塑料部分，互斥）---
        combined_map[float_wins & not_plast] = 1
        combined_map[susp_wins  & not_plast] = 2

        # --- 塑料相关（互斥划分）---
        plast_only      = plastic_mask_pred & ~floating_mask_pred & ~suspended_mask_pred
        plast_and_float = plastic_mask_pred & float_wins
        plast_and_susp  = plastic_mask_pred & susp_wins

        combined_map[plast_only]      = 3
        combined_map[plast_and_float] = 4
        combined_map[plast_and_susp]  = 5

    # 非水体清零（确保 water_mask 是 bool）
    combined_map[~water_mask] = 0


    # --------- Vectorized DataFrame construction (no per-pixel Python loops) ---------
    LOGGER.info("Building detections DataFrame (vectorized)")
    df_float = _df_from_mask("floating_debris",  floating_mask_pred,  floating_prob,
                             ndvi, fdi, ndti, sabi, A_B11, A_B12, transform, src_crs, water_mask.shape)
    df_susp  = _df_from_mask("suspended_matter", suspended_mask_pred, suspended_prob,
                             ndvi, fdi, ndti, sabi, A_B11, A_B12, transform, src_crs, water_mask.shape)
    # if plastic_mask_pred is not None and plastic_prob_map is not None:
    #     df_plastic=_df_from_mask("plastic_ml", plastic_mask_pred, plastic_prob_map,
    #                          ndvi, fdi, ndti, sabi, A_B11, A_B12, transform, water_mask.shape)
    #     df = pd.concat([df_float, df_susp, df_plastic], ignore_index=True)
    # else:
    df = pd.concat([df_float, df_susp], ignore_index=True)
    # 回填塑料标记到浮漂类
    # --------- 回填塑料标记到浮漂类（修复形状不一致）---------
    if plastic_mask_pred is not None and plastic_prob_map is not None and not df.empty:
        H, W = water_mask.shape

        # plastic 掩膜的线性索引
        plast_idx_flat = np.flatnonzero(plastic_mask_pred.reshape(-1))

        # df 中“浮漂”行的布尔掩膜（长度 == len(df)）
        float_mask_df = (df["class"].values == "floating_debris")

        # 取出浮漂行的 (row, col) 并做边界裁剪（稳妥起见）
        float_rows = df.loc[float_mask_df, "row"].to_numpy(dtype=np.int64, copy=False)
        float_cols = df.loc[float_mask_df, "col"].to_numpy(dtype=np.int64, copy=False)
        np.clip(float_rows, 0, H - 1, out=float_rows)
        np.clip(float_cols, 0, W - 1, out=float_cols)

        # 浮漂行对应的线性索引
        float_idx_flat = np.ravel_multi_index((float_rows, float_cols),
                                            dims=(H, W), mode="clip")

        # 命中判断（长度 == 浮漂子集行数）
        hit_float = np.isin(float_idx_flat, plast_idx_flat)

        # 把命中结果“回填”为与 df 等长的布尔数组
        full_hit = np.zeros(len(df), dtype=bool)
        full_hit[np.where(float_mask_df)[0]] = hit_float

        # 写入标记与概率
        df.loc[full_hit, "PlasticFlag"] = True
        rr = df.loc[full_hit, "row"].to_numpy(dtype=np.intp, copy=False)
        cc = df.loc[full_hit, "col"].to_numpy(dtype=np.intp, copy=False)
        df.loc[full_hit, "PlasticProb"] = plastic_prob_map[rr, cc].astype(np.float32)
        if plastic_type_map is not None:
            df.loc[full_hit, "PlasticType"] = np.asarray(plastic_type_map[rr, cc], dtype=object)
    # --------------------------------------------------------
    # 另外单独产出 plastic_ml 行（可选）
    if plastic_mask_pred is not None and plastic_prob_map is not None:
        pr, pc = np.where(plastic_mask_pred)
        if pr.size > 0:
            xs, ys = _coords_from_indices(transform, pr, pc)
            if src_crs:
                try:
                    lons, lats = rio_transform(src_crs, "EPSG:4326", xs.tolist(), ys.tolist())
                    lons = np.asarray(lons, dtype=np.float64)
                    lats = np.asarray(lats, dtype=np.float64)
                except Exception:
                    lons = np.full_like(xs, np.nan, dtype=np.float64)
                    lats = np.full_like(ys, np.nan, dtype=np.float64)
            else:
                lons = np.full_like(xs, np.nan, dtype=np.float64)
                lats = np.full_like(ys, np.nan, dtype=np.float64)
            df_plastic = pd.DataFrame({
                "row": pr.astype(np.int32),
                "col": pc.astype(np.int32),
                "x": xs.astype(np.float64),
                "y": ys.astype(np.float64),
                "lon": lons.astype(np.float64),
                "lat": lats.astype(np.float64),
                "class": np.full(pr.shape, "plastic_ml"),
                "probability": plastic_prob_map[pr, pc].astype(np.float32),
                "NDVI": ndvi[pr, pc].astype(np.float32),
                "FDI": fdi[pr, pc].astype(np.float32),
                "NDTI": ndti[pr, pc].astype(np.float32),
                "SABI": sabi[pr, pc].astype(np.float32),
                "A_B11": A_B11[pr, pc].astype(np.float32),
                "A_B12": A_B12[pr, pc].astype(np.float32),
                "PlasticProb": plastic_prob_map[pr, pc].astype(np.float32),
                "PlasticFlag": np.full(pr.shape, True),
                "PlasticType": np.asarray(plastic_type_map[pr, pc], dtype=object) if plastic_type_map is not None else np.full(pr.shape, None, dtype=object),
            })
            df = pd.concat([df, df_plastic], ignore_index=True)

    # ---- Save CSV & Preview with logging ----
    LOGGER.info("Writing detections CSV to %s (rows=%d)", output_csv, len(df))
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    LOGGER.info("Saved CSV")

    stats = {
        "floating_nonplastic": compute_stats("floating_nonplastic", (combined_map==1)),
        "suspended_nonplastic": compute_stats("suspended_nonplastic", (combined_map==2)),
        }
    
    if plastic_mask_pred is not None:
        stats["plastic_only"] =  compute_stats("plastic_only", (combined_map==3))
        stats["plastic_and_floating"] = compute_stats("plastic_and_floating", (combined_map==4))
        stats["plastic_and_suspended"] = compute_stats("plastic_and_suspended", (combined_map==5))
    rgb = np.dstack([
        np.clip(B4 / np.nanmax(B4, where=water_mask, initial=1), 0, 1),
        np.clip(B3 / np.nanmax(B3, where=water_mask, initial=1), 0, 1),
        np.clip(B2 / np.nanmax(B2, where=water_mask, initial=1), 0, 1),
    ])
    for fa,sa,la in zip([floating_mask_pred, suspended_mask_pred, plastic_mask_pred],
                  ["floating", "suspended", "plastic"],
                  [ floating_thresh,  suspended_thresh, plastic_prob_thresh]):
        try:
            _save_debug_overlay(
                debug_dir,
                f"{sa}_detections_overlay.png",
                rgb,
                fa,
                title=f"{sa} detections",
                color="orange",
                mask=water_mask,
            )
            _save_debug_image(
                        debug_dir, f"{sa} detections_prob_map.png", fa,
                        title=f" probability (sparse, thr={la})",
                        cmap="inferno", mask=water_mask,
                    )
        except Exception as e:
            LOGGER.warning("plot failed: %s", e)
            
    LOGGER.info("Rendering preview to %s", preview_png)
    visualize_results(
        rgb_image=rgb,
        floating_map=floating_prob,
        suspended_map=suspended_prob,
        plastic_map=(plastic_prob_map if plastic_prob_map is not None else np.zeros_like(floating_prob)),
        class_map=combined_map,
        df=df,
        stats=stats,
        output_png=preview_png,
    )

    LOGGER.info("Saved preview")

    if handler:
        handler.flush()
        LOGGER.removeHandler(handler)
        handler.close()

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
        else:
            df = plot_sateline(
                input_tif=tif_path,
                shp_file=shp_file,
                output_csv=csv_path,
                preview_png=png_path,
                **filter_args,
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
    "plot_sateline",
    "floating_debris_detection",
    "create_mask_tif_from_shp",
    "batch_detect_directory",
    "build_plastic_feature_matrix_sparse",
]
