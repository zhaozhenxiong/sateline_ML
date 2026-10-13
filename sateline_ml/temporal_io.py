from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize

try:
    import geopandas as gpd  # 可选
except Exception:
    gpd = None


def _norm_band_desc(descs: Sequence[str]) -> Tuple[str, ...]:
    # 统一成大写去空格，常见写法 "B08", "Band 8", "Blue", etc. 这里只保留标准码
    out = []
    for d in (descs or []):
        if not isinstance(d, str):
            out.append("")
            continue
        s = d.strip().upper().replace(" ", "")
        # 简单归一：B8/B08 → B8；B8A 保持
        if s.startswith("BAND"):  # "BAND8" -> "B8"
            s = "B" + s[4:]
        if s in {"B02","B2"}: out.append("B2");  continue
        if s in {"B03","B3"}: out.append("B3");  continue
        if s in {"B04","B4"}: out.append("B4");  continue
        if s in {"B05","B5"}: out.append("B5");  continue
        if s in {"B06","B6"}: out.append("B6");  continue
        if s in {"B07","B7"}: out.append("B7");  continue
        if s in {"B08","B8"}: out.append("B8");  continue
        if s in {"B8A","B08A"}: out.append("B8A"); continue
        if s in {"B09","B9"}: out.append("B9");  continue
        if s in {"B11","B111"}: out.append("B11"); continue
        if s in {"B12","B121"}: out.append("B12"); continue
        out.append(s)
    return tuple(out)


def _pick_resampling(name: str | Resampling) -> Resampling:
    if isinstance(name, Resampling):
        return name
    m = str(name).lower()
    if m in {"nearest"}: return Resampling.nearest
    if m in {"bilinear","linear"}: return Resampling.bilinear
    if m in {"cubic"}: return Resampling.cubic
    if m in {"average","avg","mean"}: return Resampling.average
    return Resampling.bilinear


def ensure_band_order(
    cube: np.ndarray,
    current_order: Sequence[str],
    target_order: Sequence[str],
) -> np.ndarray:
    """
    把 (H,W,C_cur) 的波段立方体重排成 target_order；缺失波段用 NaN 填。
    """
    H, W, _ = cube.shape
    cur_map = {b: i for i, b in enumerate(current_order)}
    out = np.full((H, W, len(target_order)), np.nan, dtype=cube.dtype)
    for j, b in enumerate(target_order):
        idx = cur_map.get(b, None)
        if idx is not None:
            out[:, :, j] = cube[:, :, idx]
    return out


def read_and_align_one(
    tif_path: Path | str,
    *,
    dst_shape: Tuple[int, int],
    dst_transform,
    dst_crs,
    bands_keep: Optional[Sequence[int]] = None,
    resampling: Resampling = Resampling.bilinear,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """
    读入单幅影像并重投影/重采样到参考网格。
    返回 (H,W,C) 与该文件的波段描述顺序（对齐后）。
    """
    with rasterio.open(tif_path) as src:
        src_crs = src.crs
        src_transform = src.transform
        descs = _norm_band_desc(src.descriptions or [f"B{i}" for i in range(1, src.count + 1)])

        if bands_keep is None:
            bands_keep = list(range(1, src.count + 1))

        H, W = dst_shape
        out = np.zeros((H, W, len(bands_keep)), dtype=np.float32)

        for bi, b in enumerate(bands_keep):
            band = src.read(b)  # (h,w)
            dst = np.zeros((H, W), dtype=np.float32)
            reproject(
                source=band,
                destination=dst,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=resampling,
                num_threads=2,
            )
            out[:, :, bi] = dst

        # 收缩描述到保留波段
        descs_kept = tuple(descs[b - 1] if 1 <= b <= len(descs) else f"B{b}" for b in bands_keep)
    return out, descs_kept


def align_and_stack_scenes(
    input_tifs: Sequence[Path | str],
    *,
    target_band_order: Sequence[str] = ("B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12"),
    resampling: str | Resampling = "bilinear",
    scale: float = 1e-4,
    shp_file: Optional[Path | str] = None,
    aoi_filter: Optional[Dict] = None,  # {"field": "...", "values": [...]} 可选
) -> Tuple[np.ndarray, Tuple[str, ...], List[Tuple[str, ...]], rasterio.Affine, object, np.ndarray]:
    """
    把多期影像堆叠成 (T,H,W,C)，对齐到首幅影像的网格；返回：
    - stack: (T,H,W,C) 反射率（已乘 scale）
    - target_order: 目标波段顺序（C 的含义）
    - per_scene_orders: 每个文件原本（或可识别）的波段描述顺序
    - transform, crs: 参考网格
    - aoi_mask: (H,W) bool，若提供 shapefile 则为 AOI 栅格化掩膜，否则全 True
    """
    assert len(input_tifs) >= 1, "input_tifs 不能为空"

    # 1) 参考网格取第一幅
    first = str(input_tifs[0])
    with rasterio.open(first) as ref:
        dst_transform = ref.transform
        dst_crs = ref.crs
        dst_shape = (ref.height, ref.width)
        ref_descs = _norm_band_desc(ref.descriptions or [f"B{i}" for i in range(1, ref.count + 1)])
        # 尝试确定“与 target 的交集”对应的 band 索引
        ref_map = {d: i+1 for i, d in enumerate(ref_descs)}
        keep_idx = []
        for b in target_band_order:
            if b in ref_map:
                keep_idx.append(ref_map[b])
        if not keep_idx:
            # 如果首幅无法识别描述，就直接按前 N 个波段
            keep_idx = list(range(1, min(len(target_band_order), ref.count) + 1))

    # 2) AOI 掩膜（可选）
    if shp_file is not None and gpd is not None:
        try:
            gdf = gpd.read_file(shp_file)
            if aoi_filter and "field" in aoi_filter:
                fld = aoi_filter["field"]
                vals = set(aoi_filter.get("values", []))
                if fld in gdf.columns and len(vals) > 0:
                    gdf = gdf[gdf[fld].isin(vals)]
            if dst_crs is not None:
                gdf = gdf.to_crs(dst_crs)
            shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
            aoi_mask = rasterize(
                shapes=shapes,
                out_shape=dst_shape,
                transform=dst_transform,
                fill=0,
                dtype=np.uint8,
            ).astype(bool)
        except Exception:
            # 失败时退化为全 True
            aoi_mask = np.ones(dst_shape, dtype=bool)
    else:
        aoi_mask = np.ones(dst_shape, dtype=bool)

    # 3) 逐幅重采样到参考网格，并重排/填补到 target_band_order
    resamp = _pick_resampling(resampling)
    per_scene_orders: List[Tuple[str, ...]] = []
    cubes: List[np.ndarray] = []
    for tif in input_tifs:
        arr_refgrid, descs = read_and_align_one(
            tif, dst_shape=dst_shape, dst_transform=dst_transform, dst_crs=dst_crs,
            bands_keep=None, resampling=resamp,
        )
        per_scene_orders.append(descs)
        arr_refgrid = arr_refgrid.astype(np.float32) * float(scale)
        # 重排到 target
        arr_target = ensure_band_order(arr_refgrid, descs, target_band_order)
        # 非 AOI 部分置 NaN（保持与后续统计一致）
        arr_target[~aoi_mask, :] = np.nan
        cubes.append(arr_target)

    stack = np.stack(cubes, axis=0)  # (T,H,W,C)
    return stack, tuple(target_band_order), per_scene_orders, dst_transform, dst_crs, aoi_mask