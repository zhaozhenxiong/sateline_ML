"""Feature engineering for plastic spectral detection."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .spectral_tools import (
    S2_BANDS,
    band_depth,
    convolve_to_sensor,
    matched_filter_score,
    spectral_angle_mapper,
)

def load_plastic_library(csv_path: Path | str) -> Dict[str, Dict[str, np.ndarray]]:
    df = pd.read_csv(csv_path)
    if "wavelength_nm" not in df.columns:
        raise ValueError("Library CSV must contain 'wavelength_nm' column")
    wl = df["wavelength_nm"].to_numpy(dtype=float)
    lib: Dict[str, Dict[str, np.ndarray]] = {}
    for column in df.columns:
        if column == "wavelength_nm":
            continue
        lib[column] = {"wl": wl, "refl": df[column].to_numpy(dtype=float)}
    if not lib:
        raise ValueError("Library CSV contains no material columns")
    return lib

def convolve_library_to_S2(lib: Mapping[str, Mapping[str, Sequence[float]]]) -> tuple[Dict[str, np.ndarray], Sequence[str]]:
    band_order = list(S2_BANDS.keys())
    vectors: Dict[str, np.ndarray] = {}
    for name, spec in lib.items():
        conv = convolve_to_sensor(spec["wl"], spec["refl"], bands=S2_BANDS)
        vectors[name] = np.array([conv[b] for b in band_order], dtype=float)
    return vectors, band_order

def compute_background_cov(samples: np.ndarray) -> np.ndarray:
    cov = np.cov(samples, rowvar=False) + 1e-6 * np.eye(samples.shape[1])
    return np.linalg.inv(cov)

def build_plastic_features(
    spectra: np.ndarray,
    band_order: Sequence[str],
    endmembers: Mapping[str, np.ndarray],
    *,
    sigma_inv: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    按可用波段动态构建塑料判别特征：
      - 始终：SAM_/MF_（与端元库相关）
      - 若有 B9：水汽吸收带深度 BD_B9（以 B8A/B11 为两端连续谱）
      - 互补：多组比值/归一化差异/曲率/吸收强度（在 B2,B3,B4,B8,B8A,B11,B12 任意子集上可用哪算哪）
    """
    X = np.asarray(spectra, dtype=float)
    n = X.shape[0]
    features: Dict[str, np.ndarray] = {}

    # ---------- 端元相关分数（始终可算） ----------
    for name, k in endmembers.items():
        features[f"SAM_{name}"] = np.apply_along_axis(spectral_angle_mapper, 1, X, k)
        if sigma_inv is not None:
            features[f"MF_{name}"] = np.apply_along_axis(matched_filter_score, 1, X, k, sigma_inv)

    # ---------- 波段索引与便捷函数 ----------
    idx = {b: i for i, b in enumerate(band_order)}
    def has(*bs: str) -> bool:
        return all(b in idx for b in bs)
    def vec(b: str) -> np.ndarray:
        return X[:, idx[b]]

    # 计算时避免除零/对数问题
    eps = 1e-9
    def safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a + eps) / (b + eps)
    def safe_nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a - b) / (a + b + eps)
    def safe_absorb(a: np.ndarray) -> np.ndarray:
        # 反射率→吸收（对数），裁掉非正值
        return -np.log10(np.clip(a, eps, None))

    # ---------- 关键吸收/曲率类 ----------
    # 真·水汽带深度（需要 B9）；否则用 NIR-SWIR 曲率/斜率替代
    if has("B9", "B8A", "B11"):
        B9, B8A, B11 = vec("B9"), vec("B8A"), vec("B11")
        features["BD_B9"] = np.array([
            band_depth(b9, b8a, b11, left_center=865.0, right_center=1610.0, band_center=945.0)
            for b9, b8a, b11 in zip(B9, B8A, B11)
        ])
    # NIR–SWIR 曲率（没有 B9 时的稳定替代）：B8A 与 (B8,B11) 的弯曲程度
    if has("B8A", "B8", "B11"):
        features["CURV_B8A_B8_B11"] = vec("B8A") - 0.5 * (vec("B8") + vec("B11"))
    if has("B8A", "B11"):
        features["SLOPE_B8A_B11"] = vec("B11") - vec("B8A")

    # ---------- 吸收强度（SWIR） ----------
    if has("B11"):
        A11 = safe_absorb(vec("B11"))
        features["A_B11"] = A11
    if has("B12"):
        A12 = safe_absorb(vec("B12"))
        features["A_B12"] = A12
    if has("B11", "B12"):
        A11, A12 = safe_absorb(vec("B11")), safe_absorb(vec("B12"))
        features["A_DIFF_B11_B12"] = A11 - A12
        features["A_RATIO_B11_B12"] = safe_ratio(A11, A12)

    # ---------- 比值/归一化差异（按可用波段组合尽量多造） ----------
    pair_list = [
        ("B8A", "B11"), ("B8", "B11"), ("B5", "B11"), ("B4", "B11"),
        ("B4", "B8A"), ("B5", "B8"), ("B3", "B8"), ("B2", "B3"),
        ("B8", "B4"), ("B3", "B4"), ("B5", "B9"), ("B2", "B8"),
        ("B12", "B11"), ("B8A", "B12")
    ]
    for a, b in pair_list:
        if has(a, b):
            va, vb = vec(a), vec(b)
            features[f"R_{a}_{b}"]  = safe_ratio(va, vb)
            features[f"ND_{a}_{b}"] = safe_nd(va, vb)

    # ---------- 若具备 B5/B9，可补充水汽带吸收差 ----------
    if has("B5", "B9"):
        features["ABS_DIFF_B5_B9"] = safe_absorb(vec("B5")) - safe_absorb(vec("B9"))

    # 最终 DataFrame
    return pd.DataFrame(features, index=np.arange(n))
