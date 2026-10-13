"""Spectral helper functions for Sentinel-2 plastic detection workflows."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import numpy as np

# Sentinel-2 approximate band centres / FWHM (nm). For production use official RSRs.
S2_BANDS: Dict[str, Dict[str, float]] = {
    "B2": {"center": 492.0, "fwhm": 65.0},
    "B3": {"center": 560.0, "fwhm": 35.0},
    "B4": {"center": 665.0, "fwhm": 30.0},
    "B5": {"center": 705.0, "fwhm": 15.0},
    "B6": {"center": 740.0, "fwhm": 15.0},
    "B7": {"center": 783.0, "fwhm": 20.0},
    "B8": {"center": 842.0, "fwhm": 115.0},
    "B8A": {"center": 865.0, "fwhm": 20.0},
    "B9": {"center": 945.0, "fwhm": 20.0},
    "B11": {"center": 1610.0, "fwhm": 90.0},
    "B12": {"center": 2190.0, "fwhm": 180.0},
}


def gaussian_kernel(wavelength_nm: np.ndarray, center: float, fwhm: float) -> np.ndarray:
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    weights = np.exp(-0.5 * ((wavelength_nm - center) / sigma) ** 2)
    return weights / (weights.sum() + 1e-12)


def convolve_to_sensor(
    wavelength_nm: Iterable[float],
    reflectance: Iterable[float],
    bands: Mapping[str, Mapping[str, float]] = S2_BANDS,
) -> Dict[str, float]:
    wl = np.asarray(list(wavelength_nm), dtype=float)
    refl = np.asarray(list(reflectance), dtype=float)
    if wl.size == 0:
        raise ValueError("wavelength_nm is empty")
    if np.isnan(refl).all():
        raise ValueError("reflectance contains only NaNs")
    mask = ~np.isnan(refl)
    refl = np.interp(wl, wl[mask], refl[mask])

    out: Dict[str, float] = {}
    for band, spec in bands.items():
        kernel = gaussian_kernel(wl, spec["center"], spec["fwhm"])
        out[band] = float(np.dot(refl, kernel))
    return out


def spectral_angle_mapper(x: np.ndarray, k: np.ndarray) -> float:
    num = float(np.dot(x, k))
    den = float(np.linalg.norm(x) * np.linalg.norm(k) + 1e-12)
    value = np.clip(num / den, -1.0, 1.0)
    return float(np.arccos(value))


def matched_filter_score(x: np.ndarray, k: np.ndarray, sigma_inv: np.ndarray) -> float:
    num = float(k @ (sigma_inv @ x))
    den = float(k @ (sigma_inv @ k) + 1e-12)
    return num / den


def band_depth(
    band_value: np.ndarray | float,
    left_value: np.ndarray | float,
    right_value: np.ndarray | float,
    left_center: float,
    right_center: float,
    band_center: float,
) -> np.ndarray | float:
    """
    Compute band depth for scalars or numpy arrays; returns same shape as inputs.
    """
    band_arr = np.asarray(band_value, dtype=np.float32)
    left_arr = np.asarray(left_value, dtype=np.float32)
    right_arr = np.asarray(right_value, dtype=np.float32)
    slope = (right_arr - left_arr) / (right_center - left_center + 1e-12)
    continuum = left_arr + slope * (band_center - left_center)
    depth = 1.0 - (band_arr / (continuum + 1e-12))
    if depth.ndim == 0:
        return float(depth)
    return depth.astype(np.float32)
