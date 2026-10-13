
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract multiple spectral curves from plot images (auto, no manual clicks).
- Works on typical "Wavelength (nm) vs Reflectance (%)" plots with many colored lines.
- Detects axes automatically, isolates the plot area, separates colored lines via color clustering,
  traces each curve, maps pixel coords to numeric axes, and exports wide CSV.
- Dependencies: numpy, pandas, Pillow (PIL). No OpenCV/Skimage required.
- Supports batch processing of multiple images.

Usage:
    python extract_spectra_from_plot.py image1.png image2.png ...
    # Outputs CSVs next to the images: <stem>_spectra.csv

Notes:
- X-axis assumed to be 1000–2500 nm; Y-axis 0–1 (fraction; multiply by 100 for % if needed).
- The script tries to auto-estimate the axes rectangle; if your figures have unusual margins,
  you can override with --crop "l,t,r,b" as fractions (0-1) of width/height.
"""

import sys
import math
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

# --------------------------- small utilities ---------------------------

def imread(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)

def to_hsv(img_rgb: np.ndarray) -> np.ndarray:
    # RGB [0,255] -> HSV [0..1]
    img = img_rgb.astype(np.float32) / 255.0
    r, g, b = img[...,0], img[...,1], img[...,2]
    mx = np.max(img, axis=-1)
    mn = np.min(img, axis=-1)
    diff = mx - mn + 1e-6
    # Hue
    h = np.zeros_like(mx)
    mask = mx == r
    h[mask] = ((g - b)/diff)[mask] % 6
    mask = mx == g
    h[mask] = ((b - r)/diff)[mask] + 2
    mask = mx == b
    h[mask] = ((r - g)/diff)[mask] + 4
    h = (h / 6.0) % 1.0
    # Saturation
    s = np.where(mx == 0, 0, diff / (mx + 1e-6))
    v = mx
    hsv = np.stack([h,s,v], axis=-1)
    return hsv

def linear_rescale(x, a0, a1, b0, b1):
    # map x in [a0,a1] -> [b0,b1]
    return b0 + (np.clip(x, a0, a1) - a0) * (b1 - b0) / (a1 - a0 + 1e-6)

# ------------------------ simple KMeans (numpy) ------------------------

@dataclass
class KMeansResult:
    centers: np.ndarray  # (k, d)
    labels: np.ndarray   # (n,)

def kmeans_numpy(X: np.ndarray, k: int, max_iter: int = 30, tol: float = 1e-4, seed: int = 42) -> KMeansResult:
    """
    Minimal KMeans (L2) with k-means++ init.
    X: (n, d)
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    # k-means++
    centers = np.empty((k, d), dtype=X.dtype)
    idx0 = rng.integers(0, n)
    centers[0] = X[idx0]
    closest_dist_sq = np.sum((X - centers[0])**2, axis=1)
    for i in range(1, k):
        probs = closest_dist_sq / (closest_dist_sq.sum() + 1e-12)
        idx = rng.choice(n, p=probs)
        centers[i] = X[idx]
        dist_sq = np.sum((X - centers[i])**2, axis=1)
        closest_dist_sq = np.minimum(closest_dist_sq, dist_sq)
    labels = np.zeros(n, dtype=np.int32)
    prev_inertia = np.inf
    for it in range(max_iter):
        # assign
        d2 = np.sum((X[:,None,:] - centers[None,:,:])**2, axis=2)  # (n,k)
        labels = d2.argmin(axis=1)
        # update
        new_centers = np.zeros_like(centers)
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                new_centers[i] = X[mask].mean(axis=0)
            else:
                # reinit empty cluster
                new_centers[i] = X[rng.integers(0, n)]
        centers = new_centers
        inertia = np.sum((X - centers[labels])**2)
        if abs(prev_inertia - inertia) < tol * max(prev_inertia, 1.0):
            break
        prev_inertia = inertia
    return KMeansResult(centers=centers, labels=labels)

# -------------------- axes / plot area auto-detection ------------------

def find_axes_rect(img: np.ndarray) -> Tuple[int,int,int,int]:
    """
    Heuristic: find dark axes lines near left and bottom and the tight plot box.
    Returns (x0, y0, x1, y1) inclusive-exclusive rectangle in pixel coords.
    """
    h, w, _ = img.shape
    gray = img.mean(axis=2)
    # invert brightness: dark lines -> large values after (1 -)
    inv = 1.0 - (gray / 255.0)
    # sum darkness along columns/rows
    col_dark = inv.mean(axis=0)
    row_dark = inv.mean(axis=1)
    # left axis near the left 30%
    search_left = int(0.05*w), int(0.35*w)
    x_left = np.argmax(col_dark[search_left[0]:search_left[1]]) + search_left[0]
    # bottom axis near bottom 35%
    search_bottom = int(0.65*h), int(0.95*h)
    y_bottom = np.argmax(row_dark[search_bottom[0]:search_bottom[1]]) + search_bottom[0]
    # right boundary: where content fades to whitespace in right 20%
    search_right = int(0.70*w), int(0.98*w)
    # find last column before big white margin
    thr = np.percentile(col_dark, 60)
    candidates = np.where(col_dark[search_right[0]:search_right[1]] > thr*0.3)[0]
    if len(candidates) > 0:
        x_right = candidates[-1] + search_right[0]
    else:
        x_right = int(0.95*w)
    # top boundary: similar near top 20%
    search_top = int(0.04*h), int(0.25*h)
    candidates = np.where(row_dark[search_top[0]:search_top[1]] > thr*0.3)[0]
    if len(candidates) > 0:
        y_top = candidates[0] + search_top[0]
    else:
        y_top = int(0.10*h)
    # tighten a bit inside to avoid axes lines
    pad_x = int(0.01*w)
    pad_y = int(0.01*h)
    x0 = max(0, x_left + pad_x)
    y0 = max(0, y_top + pad_y)
    x1 = min(w, x_right - pad_x)
    y1 = min(h, y_bottom - pad_y)
    # ensure valid
    if x1 - x0 < 10 or y1 - y0 < 10:
        # fallback generic crop
        x0 = int(0.12*w); x1 = int(0.94*w)
        y0 = int(0.12*h); y1 = int(0.84*h)
    return x0, y0, x1, y1

# -------------------- curve extraction per image -----------------------

def extract_curves_from_image(path: str, x_min=1000.0, x_max=2500.0, y_min=0.0, y_max=1.0,
                              crop_box_frac: Optional[Tuple[float,float,float,float]] = None,
                              k_colors: int = 64,
                              min_pixels_per_curve: int = 400,
                              wavelength_step: float = 2.0) -> pd.DataFrame:
    """
    Returns a wide DataFrame: columns = ['Wavelength', 'Curve_1', 'Curve_2', ...]
    """
    img = imread(path)
    H, W, _ = img.shape

    # crop to plot area
    if crop_box_frac is None:
        x0, y0, x1, y1 = find_axes_rect(img)
    else:
        l, t, r, b = crop_box_frac
        x0 = int(l * W); x1 = int(r * W)
        y0 = int(t * H); y1 = int(b * H)
    plot = img[y0:y1, x0:x1, :]
    h, w, _ = plot.shape

    # HSV to isolate colored strokes (high saturation) and exclude near-white
    hsv = to_hsv(plot)
    s = hsv[...,1]; v = hsv[...,2]
    # mask colored pixels
    color_mask = (s > 0.35) & (v > 0.25)
    pts = np.column_stack(np.where(color_mask))  # (n,2) row=y, col=x
    if pts.shape[0] < 500:
        raise RuntimeError(f"Too few colored pixels found in {path}.")

    # sample colors at these locations
    colors = plot[pts[:,0], pts[:,1], :].astype(np.float32) / 255.0
    # normalize to emphasize hue
    colors_lab = colors.copy()

    # KMeans to split into color groups (more than curves; later we merge small ones out)
    k = min(k_colors, max(8, int(colors.shape[0] // (h*0.8))))  # heuristic cap
    res = kmeans_numpy(colors_lab, k=k, max_iter=25)
    labels = res.labels

    # Build per-cluster point sets
    curves = []
    for ci in range(k):
        mask = labels == ci
        if not np.any(mask):
            continue
        pts_ci = pts[mask]
        # reject tiny clusters and legend noise by area/uniqueness along x
        if pts_ci.shape[0] < min_pixels_per_curve:
            continue

        # For each x, pick y with maximum saturation (nearest to colored line)
        xs = pts_ci[:,1]
        ys = pts_ci[:,0]
        # group by x column
        order = np.argsort(xs)
        xs = xs[order]; ys = ys[order]

        unique_x, idx_start = np.unique(xs, return_index=True)
        # For columns with too many points, take median y to reduce noise
        y_by_x = []
        for i in range(len(unique_x)):
            x = unique_x[i]
            # range for this x
            j0 = idx_start[i]
            j1 = idx_start[i+1] if i+1 < len(idx_start) else len(xs)
            y_med = int(np.median(ys[j0:j1]))
            y_by_x.append((x, y_med))
        x_arr = np.array([p[0] for p in y_by_x], dtype=np.float32)
        y_arr = np.array([p[1] for p in y_by_x], dtype=np.float32)

        # Map pixel -> axis values
        wl = x_min + (x_arr / (w-1)) * (x_max - x_min)
        # image y increases downwards; reflectance increases upwards in plot
        refl = y_max - (y_arr / (h-1)) * (y_max - y_min)

        # Interpolate onto common wavelength grid
        wl_grid = np.arange(x_min, x_max + 1e-6, wavelength_step, dtype=np.float32)
        # Remove duplicates in wl for interpolation
        order2 = np.argsort(wl)
        wl_s = wl[order2]; rf_s = refl[order2]
        # keep unique wl indices
        u_wl, u_idx = np.unique(wl_s, return_index=True)
        wl_s = wl_s[u_idx]; rf_s = rf_s[u_idx]
        # valid monotonic range
        if wl_s.size < 10:
            continue
        rf_interp = np.interp(wl_grid, wl_s, rf_s, left=np.nan, right=np.nan)
        # simple nan fill by nearest
        mask_nan = np.isnan(rf_interp)
        if np.any(mask_nan):
            # forward fill then backward fill
            not_nan = ~mask_nan
            if np.any(not_nan):
                rf_interp[mask_nan] = np.interp(np.flatnonzero(mask_nan),
                                                np.flatnonzero(not_nan),
                                                rf_interp[not_nan])
        curves.append(rf_interp)

    if len(curves) == 0:
        raise RuntimeError(f"No curves extracted from {path}. Try adjusting --crop or k_colors.")

    # assemble wide table
    wl_grid = np.arange(x_min, x_max + 1e-6, wavelength_step, dtype=np.float32)
    data = {"Wavelength_nm": wl_grid}
    for i, c in enumerate(curves, start=1):
        data[f"Curve_{i}"] = c
    df = pd.DataFrame(data)
    # Optionally drop duplicate curves or near-duplicates
    # Here, a light dedup: drop columns with too-high correlation (>0.999) to a previous
    keep_cols = ["Wavelength_nm"]
    for col in df.columns[1:]:
        vec = df[col].values
        is_dup = False
        for kc in keep_cols[1:]:
            kv = df[kc].values
            # corr with finite values
            mask = np.isfinite(vec) & np.isfinite(kv)
            if mask.sum() > 10:
                ccoef = np.corrcoef(vec[mask], kv[mask])[0,1]
                if np.isfinite(ccoef) and ccoef > 0.999:
                    is_dup = True
                    break
        if not is_dup:
            keep_cols.append(col)
    if len(keep_cols) < df.shape[1]:
        df = df[keep_cols]

    return df

# --------------------------- main / CLI --------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract spectral curves from plot images.")
    parser.add_argument("images", nargs="+", help="Input image files (png/jpg).")
    parser.add_argument("--x-min", type=float, default=1000.0)
    parser.add_argument("--x-max", type=float, default=2500.0)
    parser.add_argument("--y-min", type=float, default=0.0)
    parser.add_argument("--y-max", type=float, default=1.0)
    parser.add_argument("--k-colors", type=int, default=80, help="KMeans color clusters.")
    parser.add_argument("--step", type=float, default=2.0, help="Wavelength step for output grid (nm).")
    parser.add_argument("--min-pixels", type=int, default=400, help="Minimum pixels per curve cluster.")
    parser.add_argument("--min-pixels", type=int, default=400, help="Minimum pixels per curve cluster.")
    parser.add_argument("--min-pixels", type=int, default=400, help="Minimum pixels per curve cluster.")
    parser.add_argument("--crop", type=str, default=None, help='Optional crop fractions "l,t,r,b" (0-1).')
    args = parser.parse_args()

    crop = None
    if args.crop:
        parts = [float(x) for x in args.crop.split(",")]
        if len(parts) != 4:
            raise ValueError("--crop must be 'l,t,r,b'")
        crop = tuple(parts)  # type: ignore

    for img_path in args.images:
        try:
            df = extract_curves_from_image(
            img_path,
                x_min=args.x_min, x_max=args.x_max, y_min=args.y_min, y_max=args.y_max,
                crop_box_frac=crop, k_colors=args.k_colors,
                wavelength_step=args.step, min_pixels_per_curve=args.min_pixels
                )
            out_csv = img_path.rsplit(".",1)[0] + "_spectra.csv"
            df.to_csv(out_csv, index=False)
            print(f"[OK] {img_path} -> {out_csv} ({df.shape[1]-1} curves)")
        except Exception as e:
            print(f"[FAIL] {img_path}: {e}")

if __name__ == "__main__":
    main()
