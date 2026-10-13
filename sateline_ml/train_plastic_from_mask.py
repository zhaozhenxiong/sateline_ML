#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a plastic spectral classifier (MLP) from multiple Sentinel-2 scenes.

Inputs:
- One or more Sentinel-2 stacks (.tif) with standard band descriptions or known band order.
- For each scene, a plastic mask raster (1 = plastic, 0 = non-plastic) in the same grid as the scene.
- A spectral endmember library CSV compatible with your `endmember_features` helpers.
- (Optional) a water mask threshold via NDWI is computed automatically; negatives均从水体内采样。

Outputs:
- A .pth file containing the trained PlasticMLP plus PlasticModelMetadata
  (feature_names, input_dim, band_order) -> 兼容现有 load_plastic_model / predict_plastic。

Notes:
- 设备优先级: CUDA > MPS > CPU（复用 resolve_device）
- 默认按正:负 = 1:3 进行均衡抽样；可用 --neg-ratio 调整
- 默认随机留出 15% 作为验证集，报告 AUC / 准确率 / F1
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Sequence, Optional, Tuple
import argparse
import json
import math

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# 如果你的包是包内脚本运行，请改成相对导入：from sateline_ml.train_model import ...
from sateline_ml.train_model import (
    build_plastic_feature_matrix,
    train_plastic_model,
    save_plastic_model,
    PlasticModelMetadata,
    resolve_device,
)
from sateline_ml.spectral_tools import S2_BANDS




def _read_stack(input_tif: Path) -> Tuple[np.ndarray, Sequence[str], any]:
    """Read a multi-band raster and infer band order."""
    with rasterio.open(input_tif) as src:
        bands = []
        for i in range(1, src.count + 1):
            bands.append(src.read(i).astype("float32") * 0.0001)
        cube = np.stack(bands, axis=-1)  # H W C
        desc = [d.strip().upper() if isinstance(d, str) else "" for d in (src.descriptions or [])]
        known = set(S2_BANDS.keys())
        if desc and all(d in known for d in desc):
            band_order = tuple(desc)
        else:
            full_band_order = ("B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12")
            compact_band_order = ("B2","B3","B4","B5","B8","B8A","B9","B11","B12")
            minimal_band_order = ("B2","B3","B4","B8","B8A","B11","B12")
            candidate_orders = [full_band_order, compact_band_order, minimal_band_order]
            band_order = None
            for order in candidate_orders:
                if len(order) == src.count:
                    band_order = order
                    break
            if band_order is None:
                if src.count < len(minimal_band_order):
                    raise ValueError(f"Unrecognized band layout for {input_tif} (count={src.count})")
                base = candidate_orders[0]
                band_order = tuple(base[:src.count])
                print(
                    f"[WARN] {input_tif.name}: guessed band order {band_order} "
                    f"from canonical layout {base}."
                )
        transform = src.transform
    return cube, band_order, transform


def _resize_like(arr: np.ndarray, out_shape: Tuple[int,int]) -> np.ndarray:
    """Nearest-neighbor resize to match scene size."""
    H, W = out_shape
    if arr.shape == (H,W):
        return arr
    scale_y = arr.shape[0] / H
    scale_x = arr.shape[1] / W
    # 使用 rasterio.reproject 比 cv2 更稳（地理栅格）
    src_profile = rasterio.profiles.DefaultGTiffProfile()
    src = rasterio.io.MemoryFile().open(
        driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=arr.dtype
    )
    try:
        src.write(arr, 1)
        dst = np.zeros((H,W), dtype=arr.dtype)
        rasterio.warp.reproject(
            source=arr,
            destination=dst,
            src_transform=rasterio.transform.Affine(scale_x, 0, 0, 0, scale_y, 0),
            dst_transform=rasterio.transform.Affine(1,0,0,0,1,0),
            src_crs="EPSG:4326",
            dst_crs="EPSG:4326",
            resampling=Resampling.nearest,
        )
        return dst
    finally:
        src.close()


def _compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (green - nir) / (green + nir + 1e-10)


def _make_water_mask(cube: np.ndarray, band_order: Sequence[str], thresh: float = 0.05) -> np.ndarray:
    """Simple NDWI水体掩膜（和检测管线保持一致）"""
    idx = {b:i for i,b in enumerate(band_order)}
    if "B3" not in idx or "B8" not in idx:
        raise ValueError("Need B3 (green) and B8 (NIR) to compute NDWI")
    ndwi = _compute_ndwi(cube[:,:,idx["B3"]], cube[:,:,idx["B8"]])
    return ndwi > thresh


def _sample_balanced_indices(pos_mask: np.ndarray, neg_mask: np.ndarray, neg_ratio: float, seed: int=42):
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(pos_mask.ravel())
    neg_idx = np.flatnonzero(neg_mask.ravel())
    n_pos = len(pos_idx)
    n_neg = min(len(neg_idx), int(math.ceil(n_pos * neg_ratio)))
    if n_pos == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    neg_sel = rng.choice(neg_idx, size=n_neg, replace=False) if n_neg > 0 else np.array([], dtype=np.int64)
    return pos_idx, neg_sel


def _train_val_split(n: int, val_ratio: float=0.15, seed: int=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_ratio)))
    return idx[n_val:], idx[:n_val]


def collect_features_from_scene(
    scene_tif: Path, plastic_mask_tif: Path, library_csv: Path,
    water_ndwi_thresh: float = 0.05, neg_ratio: float = 3.0, seed: int = 42
):
    cube, band_order, _ = _read_stack(scene_tif)
    H, W, _ = cube.shape
    with rasterio.open(plastic_mask_tif) as msk:
        pm = msk.read(1).astype(np.uint8)
    if pm.shape != (H,W):
        pm = _resize_like(pm, (H,W))
    water = _make_water_mask(cube, band_order, thresh=water_ndwi_thresh)

    # 构建特征（使用库 + 背景协方差），仅用于水体内像元
    features_all, feat_names, extras = build_plastic_feature_matrix(
        cube, band_order, library_csv, background_mask=water
    )  # shape: (H*W, D)
    # 采样标签
    pos_mask = (pm == 1) & water
    neg_mask = (pm == 0) & water
    pos_idx, neg_idx = _sample_balanced_indices(pos_mask, neg_mask, neg_ratio=neg_ratio, seed=seed)
    if pos_idx.size == 0:
        return None, None, None  # 无正样本，跳过此场景
    sel = np.concatenate([pos_idx, neg_idx], axis=0)
    y = np.concatenate([np.ones(pos_idx.size, dtype=np.float32), np.zeros(neg_idx.size, dtype=np.float32)], axis=0)
    X = features_all[sel]
    return X, y, (feat_names, band_order)


def main():
    parser = argparse.ArgumentParser(description="Train plastic spectral classifier from multiple scenes.")
    g = parser.add_argument_group("Data")
    g.add_argument("--manifest", type=str, default=None,
                   help="CSV with columns: tif,mask （每行一个场景）")
    g.add_argument("--tifs", type=str, nargs="*", default=None,
                   help="List of scene .tif files (if no manifest)")
    g.add_argument("--masks", type=str, nargs="*", default=None,
                   help="List of plastic mask .tif files matched to --tifs")
    g.add_argument("--library-csv", type=str, required=True, help="Spectral endmember library CSV")
    g.add_argument("--water-thresh", type=float, default=0.05, help="NDWI threshold for water")
    g.add_argument("--neg-ratio", type=float, default=3.0, help="Negative:positive sampling ratio within water")
    g.add_argument("--seed", type=int, default=42)

    h = parser.add_argument_group("Training")
    h.add_argument("--epochs", type=int, default=200)
    h.add_argument("--lr", type=float, default=1e-3)
    h.add_argument("--weight-decay", type=float, default=1e-4)
    h.add_argument("--hidden", type=int, nargs="*", default=[128, 64])
    h.add_argument("--dropout", type=float, default=0.10)
    h.add_argument("--device", type=str, default=None, help="cuda | cuda:0 | mps | cpu")

    o = parser.add_argument_group("Output")
    o.add_argument("--out", type=str, required=True, help="Output model path, e.g., mydata/plastic/best_plastic.pth")
    o.add_argument("--report", type=str, default=None, help="Optional JSON report path")

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    # 读取清单
    if args.manifest:
        dfm = pd.read_csv(args.manifest)
        tifs = [Path(p) for p in dfm["tif"].tolist()]
        masks = [Path(p) for p in dfm["mask"].tolist()]
    else:
        if not args.tifs or not args.masks or len(args.tifs) != len(args.masks):
            raise SystemExit("Provide --manifest CSV, or matching --tifs and --masks lists.")
        tifs = [Path(p) for p in args.tifs]
        masks = [Path(p) for p in args.masks]

    lib_csv = Path(args.library_csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 聚合所有场景的样本
    X_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    feat_names_ref: Optional[Sequence[str]] = None
    band_order_ref: Optional[Sequence[str]] = None

    for tif, msk in zip(tifs, masks):
        print(f"[SCENE] {tif.name} with mask {msk.name}")
        res = collect_features_from_scene(
            tif, msk, lib_csv,
            water_ndwi_thresh=args.water_thresh, neg_ratio=args.neg_ratio, seed=args.seed
        )
        if res[0] is None:
            print("  -> no positive samples in water; skip.")
            continue
        X, y, (feat_names, band_order) = res
        X_all.append(X)
        y_all.append(y)
        if feat_names_ref is None:
            feat_names_ref = feat_names
            band_order_ref = band_order
        else:
            # 安全起见：确保特征维度与顺序一致
            if list(feat_names_ref) != list(feat_names):
                raise RuntimeError("Feature name mismatch across scenes. Ensure same library & pipeline.")
            if list(band_order_ref) != list(band_order):
                print("[WARN] Band order differs; using first scene's order in metadata.")

    if not X_all:
        raise SystemExit("No training data collected. Check masks / water threshold.")

    X = np.vstack(X_all).astype(np.float32)
    y = np.concatenate(y_all).astype(np.float32)
    print(f"[DATA] X shape = {X.shape}, positives = {int(y.sum())}, negatives = {int((1-y).sum())}")

    # 训练/验证划分
    train_idx, val_idx = _train_val_split(len(y), val_ratio=0.15, seed=args.seed)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # 训练
    device_str = args.device
    model, hist = train_plastic_model(
        X_train, y_train,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        device=device_str, verbose=True,
        hidden=tuple(args.hidden), dropout=args.dropout,
    )

    # 验证指标
    from sateline_ml.train_model import predict_plastic  # 延迟导入避免循环
    y_val_prob = predict_plastic(model, X_val, device=device_str)
    y_val_pred = (y_val_prob >= 0.5).astype(np.float32)
    auc = roc_auc_score(y_val, y_val_prob) if len(np.unique(y_val)) > 1 else float("nan")
    acc = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)

    print(f"[VAL] AUC={auc:.4f}  ACC={acc:.4f}  F1={f1:.4f}")

    # 保存模型（含元数据）
    meta = PlasticModelMetadata(
        input_dim=X.shape[1],
        feature_names=feat_names_ref or [],
        band_order=band_order_ref or (),
    )
    save_plastic_model(out_path, model, meta)
    print(f"[SAVE] model -> {out_path}")

    # 可选报告
    if args.report:
        rep = {
            "scenes": [str(p) for p in tifs],
            "samples": {"total": int(len(y)), "pos": int(y.sum()), "neg": int((1-y).sum())},
            "val": {"auc": float(auc), "acc": float(acc), "f1": float(f1)},
            "hyperparams": {
                "epochs": args.epochs, "lr": args.lr, "weight_decay": args.weight_decay,
                "hidden": list(args.hidden), "dropout": args.dropout,
                "neg_ratio": args.neg_ratio, "water_thresh": args.water_thresh,
            },
            "device": device_str or str(resolve_device(None)),
            "metadata": {
                "input_dim": int(X.shape[1]),
                "feature_names_head": list((feat_names_ref or [])[:5]),
                "band_order": list(band_order_ref or []),
            },
        }
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] report -> {args.report}")


if __name__ == "__main__":
    main()
