# ==== NEW: robust stats, hysteresis water, morphology & region filters ====
import numpy as np
import cv2

def _q(a: np.ndarray, p: float, mask: np.ndarray | None = None) -> float:
    """Masked percentile with fallbacks."""
    if mask is not None:
        a = a[mask]
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.percentile(a, p))

def robust_thresholds_water(
    ndwi: np.ndarray,
    ndvi: np.ndarray,
    fdi: np.ndarray,
    ndti: np.ndarray,
    water_mask: np.ndarray | None = None,
) -> dict:
    """
    使用分位数/IQR的稳健阈值（在水体内统计为主），用于候选“强种子/弱扩张”。
    """
    M = water_mask
    # NDWI 双阈（滞后）
    ndwi_hi = _q(ndwi, 75, M)
    ndwi_lo = _q(ndwi, 50, M)
    if not np.isfinite(ndwi_hi): ndwi_hi = 0.1
    if not np.isfinite(ndwi_lo): ndwi_lo = 0.0

    # FDI / NDTI 稳健阈值
    fdi_q75 = _q(fdi, 75, M); fdi_q85 = _q(fdi, 85, M); fdi_q90 = _q(fdi, 90, M)
    ndti_q75 = _q(ndti, 75, M); ndti_q85 = _q(ndti, 85, M); ndti_q90 = _q(ndti, 90, M)
    fdi_q25 = _q(fdi, 25, M); ndti_q25 = _q(ndti, 25, M)
    fdi_iqr = (fdi_q75 - fdi_q25) if (np.isfinite(fdi_q75) and np.isfinite(fdi_q25)) else 0.0
    ndti_iqr = (ndti_q75 - ndti_q25) if (np.isfinite(ndti_q75) and np.isfinite(ndti_q25)) else 0.0

    fdi_strong = np.nanmax([fdi_q90, fdi_q75 + 0.5 * fdi_iqr])
    fdi_weak   = np.nanmax([fdi_q85, fdi_q75 + 0.25 * fdi_iqr])
    ndti_strong = np.nanmax([ndti_q90, ndti_q75 + 0.5 * ndti_iqr])
    ndti_weak   = np.nanmax([ndti_q85, ndti_q75 + 0.25 * ndti_iqr])

    # NDVI 窗口（避免阴影极端 & 抑制藻类）
    ndvi_lo = _q(ndvi, 5, M);  ndvi_hi = _q(ndvi, 30, M)
    if not np.isfinite(ndvi_lo): ndvi_lo = -1.0
    if not np.isfinite(ndvi_hi): ndvi_hi =  0.1

    return dict(
        ndwi_hi=float(ndwi_hi), ndwi_lo=float(ndwi_lo),
        fdi_strong=float(fdi_strong), fdi_weak=float(fdi_weak),
        ndti_strong=float(ndti_strong), ndti_weak=float(ndti_weak),
        ndvi_lo=float(ndvi_lo), ndvi_hi=float(ndvi_hi),
    )

def hysteresis_water_mask(ndwi: np.ndarray, hi: float, lo: float) -> np.ndarray:
    """
    滞后阈值：先取 >hi 的强水体，再在连通性约束下扩张到 >lo。
    """
    strong = (ndwi > hi)
    weak   = (ndwi > lo)
    if not np.any(strong):
        return weak.astype(bool)

    # 用强种子引导在 weak 内做连通域限制
    lab, n = cv2.connectedComponents(strong.astype(np.uint8), connectivity=8)
    out = np.zeros_like(strong, dtype=bool)
    # dilate 强种子两次，容忍边界薄弱点
    kernel = np.ones((3,3), np.uint8)
    grown = cv2.dilate(strong.astype(np.uint8), kernel, iterations=2).astype(bool) & weak
    # 只保留与强种子 8 邻接的 weak 区域
    out = grown
    return out

def erode_interior(mask: np.ndarray, k: int = 1) -> np.ndarray:
    """对水体掩膜做轻度腐蚀，避免岸线混合像元进入候选。"""
    if k <= 0: return mask.astype(bool)
    kernel = np.ones((3,3), np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=int(k)).astype(bool)

def region_filter(
    bin_mask: np.ndarray,
    *,
    min_area_px: int = 9,
    max_elongation: float = 6.0,
    min_solidity: float = 0.6,
) -> np.ndarray:
    """
    连通域+形态过滤：面积、细长度、实心度。
    - elongation = max(w,h)/min(w,h)
    - solidity = area / area(convex_hull)
    """
    out = np.zeros_like(bin_mask, dtype=np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask.astype(np.uint8), connectivity=8)

    for cid in range(1, num):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        w = int(stats[cid, cv2.CC_STAT_WIDTH]); h = int(stats[cid, cv2.CC_STAT_HEIGHT])
        if w == 0 or h == 0:
            continue
        elong = max(w, h) / max(1.0, min(w, h))
        if elong > max_elongation:
            continue
        # 近似 solidity：用轮廓凸包面积
        cc_mask = (labels == cid).astype(np.uint8)
        contours, _ = cv2.findContours(cc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        hull = cv2.convexHull(contours[0])
        hull_area = float(cv2.contourArea(hull))
        if hull_area <= 0:
            continue
        solidity = float(area) / hull_area
        if solidity < min_solidity:
            continue
        out[labels == cid] = 1
    return out.astype(bool)

def neighborhood_consistency(mask_core: np.ndarray, cond_map: np.ndarray, k: int = 5, min_hits: int = 8) -> np.ndarray:
    """
    在核周围 k×k 领域里，要求 cond_map 至少 min_hits 个为 True（局部一致性）。
    """
    pad = k // 2
    kernel = np.ones((k, k), np.uint8)
    hits = cv2.filter2D(cond_map.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_REFLECT)
    ok = hits >= min_hits
    return mask_core & ok