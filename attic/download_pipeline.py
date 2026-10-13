# -*- coding: utf-8 -*-
"""
Sentinel-2 GEE 稳健下载与合并（含断点续跑 / 月度最大影像挑选）
- 单窗口（start_date ~ end_date）内：对影像逐一下载所选波段（可分块），合并为多波段 TIFF
- 断点续跑：每个窗口目录下的 checkpoint.json 持久化 processed / monthly_best
- 稳健性：最小化 getInfo 次数（aggregate_array），分页兜底；requests.Session + Retry；下载后用 rasterio 校验
- 月度最佳：按月保留体积最大的多波段影像并落地单张 TIFF
"""

import os
import sys
import time
import json
import glob
import random
import shutil
from datetime import datetime

import pandas as pd
import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.warp import calculate_default_transform, reproject, Resampling

import math
import ee
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError

from requests.adapters import HTTPAdapter
from urllib3.exceptions import ProtocolError
from urllib3.util.retry import Retry

# for k in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"):
#     os.environ.pop(k, None)
# _no_proxy = "localhost,127.0.0.1,.google.com,.googleapis.com,oauth2.googleapis.com,earthengine.googleapis.com"
# os.environ["NO_PROXY"] = _no_proxy
# os.environ["no_proxy"] = _no_proxy

# ====================== 参数与全局常量 ======================
MAX_RETRIES = 5
TIMEOUT_SEC = 180
MISSING_THRESH = 0.40

def robust_initialize_gee(project: str,
                          retries: int = 5,
                          base_wait: float = 3.0) -> None:
    """Initialize Earth Engine with retries to dodge transient HTTP truncation errors."""
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            ee.Initialize(project=project)
            return
        except (ChunkedEncodingError, ConnectionError, ProtocolError) as exc:
            last_err = exc
            print(f"⚠️ GEE 初始化网络异常（{attempt}/{retries}）：{exc}")
        except Exception as exc:
            last_err = exc
            print(f"⚠️ GEE 初始化失败（{attempt}/{retries}）：{exc}")
        if attempt == retries:
            break
        time.sleep(base_wait * attempt + random.random())
        ee.Reset()
    if last_err is not None:
        raise last_err

# 必要：初始化 GEE（你已有的工程）
robust_initialize_gee('singular-tuner-471501-g3')

# ====================== Tee: 同时写 log 与控制台 ======================
class Tee:
    def __init__(self, file):
        self.file = file
        self.stdout = sys.stdout
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()

# ====================== 工具函数 ======================
def cleanup_path(path):
    """安全删除文件/文件夹"""
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.isfile(path):
            os.remove(path)
    except:
        pass


def split_roi(region: ee.Geometry, nx: int | None = None, ny: int | None = None,
              *, scale_m: float = 10.0,
              target_tile_dim_px: int = 6000,
              max_tile_dim_px: int = 10000,
              max_tile_pixels: int = 80_000_000,
              max_grid: int = 8) -> list[ee.Geometry]:
    """
    将 ROI 自适应切分为 nx*ny 个小块；若 nx/ny 为 None，则根据地理尺寸自动估算。
    - scale_m: 影像导出分辨率（Sentinel-2 一般 10m）
    - target_tile_dim_px: 期望单块像素边长（尽量接近，但不超过 max_tile_dim_px）
    - max_tile_dim_px: 单边最大像素，避免触发 GEE 限制（~10k 常用）
    - max_tile_pixels: 单块最大像素数上限（安全阈值，避免 >1e8）
    - max_grid: 单方向最多切多少块（上限保护）
    """
    # ---- 读取 bbox（WGS84） ----
    coords = region.bounds().coordinates().getInfo()[0]
    minx, miny = coords[0]
    maxx, maxy = coords[2]
    cx_lat = (miny + maxy) * 0.5

    # ---- 度->米 近似换算（足够用于切块估算）----
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(cx_lat))

    width_m  = max(0.0, (maxx - minx) * m_per_deg_lon)
    height_m = max(0.0, (maxy - miny) * m_per_deg_lat)

    # 极小 ROI 直接返回
    if width_m == 0 or height_m == 0:
        return [region]

    # ---- 估算像素尺寸 ----
    px_w = width_m  / scale_m
    px_h = height_m / scale_m

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    if nx is None or ny is None:
        # 先按“目标边长像素”估算块数
        tx = math.ceil(px_w / target_tile_dim_px)
        ty = math.ceil(px_h / target_tile_dim_px)

        # 任何方向都必须至少 1，且不超过最大网格上限
        tx = clamp(tx, 1, max_grid)
        ty = clamp(ty, 1, max_grid)

        # 约束：单块边长 <= max_tile_dim_px
        while math.ceil(px_w / tx) > max_tile_dim_px and tx < max_grid:
            tx += 1
        while math.ceil(px_h / ty) > max_tile_dim_px and ty < max_grid:
            ty += 1

        # 约束：单块像素数 <= max_tile_pixels
        def tile_ok(tx_, ty_):
            tw = math.ceil(px_w / tx_)
            th = math.ceil(px_h / ty_)
            return (tw * th) <= max_tile_pixels and tw <= max_tile_dim_px and th <= max_tile_dim_px

        # 若不满足像素上限，尽量增加切分
        safety_it = 0
        while not tile_ok(tx, ty) and (tx < max_grid or ty < max_grid) and safety_it < 64:
            # 增加长边方向的切分更有效
            if math.ceil(px_w / tx) >= math.ceil(px_h / ty):
                tx = min(tx + 1, max_grid)
            else:
                ty = min(ty + 1, max_grid)
            safety_it += 1
        nx, ny = tx, ty
    else:
        nx = clamp(int(nx), 1, max_grid)
        ny = clamp(int(ny), 1, max_grid)

    # ---- 生成子块几何 ----
    dx = (maxx - minx) / nx
    dy = (maxy - miny) / ny
    tiles: list[ee.Geometry] = []
    for i in range(nx):
        for j in range(ny):
            tiles.append(ee.Geometry.Rectangle([
                minx + i*dx, miny + j*dy,
                minx + (i+1)*dx, miny + (j+1)*dy
            ]))
    return tiles


def calc_missing_ratio_multiband(tif_path):
    """计算缺失比例（全部波段都无效/NoData 的像元比例）"""
    with rasterio.open(tif_path) as src:
        data = src.read()
        nodata = src.nodata
        if nodata is not None:
            invalid = np.all(data == nodata, axis=0)
        else:
            invalid = np.all(data == 0, axis=0)
        return invalid.sum() / invalid.size

def make_requests_session(total=6, backoff=0.5):
    s = requests.Session()
    retry = Retry(
        total=total, connect=total, read=total, status=total,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"])
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s

SESSION = make_requests_session()

def robust_getinfo(fn, *, retries=5, base_wait=2.0, tag="getInfo"):
    """对单次 getInfo 做退避重试"""
    wait = base_wait
    for k in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            if k == retries:
                raise
            print(f"  [{tag} 重试 {k}/{retries}] {e}")
            time.sleep(wait + random.random())
            wait = min(30.0, wait * 1.7)

def robust_aggregate_times(col, retries=5):
    """一次性拉回全部时间戳（毫秒）"""
    return robust_getinfo(lambda: col.aggregate_array("system:time_start").getInfo(),
                          retries=retries, tag="aggregate_array(system:time_start)")

def format_ms_to_datestr(ms):
    return datetime.utcfromtimestamp(ms / 1000).strftime("%Y-%m-%d_%H-%M-%S")

def safe_get_collection_size(col, retries=2, sleep_base=2):
    for k in range(retries + 1):
        try:
            return int(ee.Number(col.size()).getInfo())
        except Exception:
            if k == retries:
                return None
            time.sleep(sleep_base * (k + 1))

def iter_collection_paginated_with_dates(col, page_size=10, start=0):
    """
    逐页迭代 Image；为每个 Image 附带服务端 date_str（减少逐条 getInfo）
    """
    col2 = col.map(lambda im: im.set(
        "date_str", ee.Date(im.get("system:time_start")).format("YYYY-MM-dd_HH-mm-ss")
    ))
    i = int(start)
    while True:
        lst = col2.toList(page_size, i)
        try:
            current = robust_getinfo(lambda: lst.getInfo(), tag="page getInfo")
        except Exception:
            break
        if not current:
            break
        n = len(current)
        for j in range(n):
            props = (current[j].get("properties") or {})
            date_str = props.get("date_str")
            yield ee.Image(lst.get(j)), (i + j), date_str
        i += n

def all_bands_downloaded(folder, bands):
    """检查所有波段是否都已下载（任一 tile 存在即认为已下载）"""
    for b in bands:
        pattern = os.path.join(folder, f"{b}*.tif")
        if len(glob.glob(pattern)) == 0:
            return False
    return True

def robust_download_band(img, band, out_path, region, nx=None, ny=None, session: requests.Session=None):
    """
    稳健下载一个波段；可分块；下载后用 rasterio 打开校验。
    """
    scale = 10
    session = session or SESSION

    def _download_one(band_img, path, geom):
        url = band_img.getDownloadURL({
            "scale": scale,
            "region": geom,
            "format": "GEO_TIFF"
        })
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with session.get(url, stream=True, timeout=TIMEOUT_SEC) as r:
                    if r.status_code != 200:
                        raise RuntimeError(f"HTTP {r.status_code}")
                    with open(path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 512):
                            if chunk:
                                f.write(chunk)
                with rasterio.open(path) as _:
                    pass
                return True
            except Exception as e:
                print(f"    [tile 下载重试 {attempt}/{MAX_RETRIES}] {e}")
                cleanup_path(path)
                time.sleep(2 * attempt + random.random())
        return False
        # 自适应：nx/ny 为 None 时自动估算
    if (nx is None or ny is None) or (nx > 1 or ny > 1):
        tiles = split_roi(region, nx=nx, ny=ny)  # 可能返回 1x1，也可能多块
        if len(tiles) > 1:
            ok_all = True
            for idx, tile in enumerate(tiles):
                tile_path = out_path.replace(".tif", f"_tile{idx}.tif")
                try:
                    band_img = img.select(band).clip(tile)
                    ok = _download_one(band_img, tile_path, tile)
                    if ok:
                        print(f"  ✅ 成功下载 {band} tile {idx} (10m)")
                    else:
                        print(f"  ❌ tile {idx} 多次失败")
                        ok_all = False
                except Exception as e:
                    print(f"  ❌ tile {idx} 异常: {e}")
                    ok_all = False
            return ok_all
    # 若 split_roi 仅返回 1 块，或 nx/ny 均为 1，则走单块逻辑
    band_img = img.select(band).clip(region)
    ok = _download_one(band_img, out_path, region)
    if ok:
        print(f"  ✅ 成功下载 {band} (10m)")
        return True
    print(f"  ❌ 放弃波段 {band}")
    return False

def merge_bands(src_folder, out_path, bands):
    """
    合并单波段 tif (包括分块 tile) 成一个多波段 tif
    """
    band_arrays = []
    profile = None
    # 为了在 profile 未创建前也能引用 transform，先记录一个参考文件的 transform/CRS
    ref_transform = None
    ref_crs = None

    for b in bands:
        pattern = os.path.join(src_folder, f"{b}*.tif")
        tile_files = sorted(glob.glob(pattern))
        if len(tile_files) == 0:
            raise FileNotFoundError(f"未找到 {b} 的 tif 文件！")

        if len(tile_files) == 1:
            with rasterio.open(tile_files[0]) as src:
                data = src.read(1).astype(np.float32)
                if profile is None:
                    profile = src.profile.copy()
                    profile.update({
                        "count": len(bands),
                        "dtype": "float32"
                    })
                    ref_transform = src.transform
                    ref_crs = src.crs
                band_arrays.append(data)
        else:
            src_list = [rasterio.open(fp) for fp in tile_files]
            mosaic, out_trans = rio_merge(src_list)
            for src in src_list:
                src.close()
            mosaic = mosaic.astype(np.float32)
            if profile is None:
                # 以 mosaicked 尺寸为准
                # 注意：此时没有 src_list 可引用 profile，所以用已有 ref 信息兜底
                profile = {
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans,
                    "count": len(bands),
                    "dtype": "float32",
                    "crs": ref_crs if ref_crs is not None else "EPSG:4326"
                }
            else:
                # 若 profile 存在，统一输出大小与 transform，以保证所有波段对齐
                profile.update({
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans
                })
            band_arrays.append(mosaic[0])  # 每个 mosaic 只有一个 band

    with rasterio.open(out_path, "w", **profile) as dst:
        for i, arr in enumerate(band_arrays, start=1):
            dst.write(arr, i)

    print(f"✅ 已合并 {len(bands)} 个波段 -> {out_path}")
    return out_path

def reproject_to_match(src_files, target_crs="EPSG:4326"):
    """将输入文件重投影到统一 CRS"""
    reprojected_files = []
    for fp in src_files:
        with rasterio.open(fp) as src:
            src_crs_str = src.crs.to_string() if src.crs is not None else None
            if src_crs_str != target_crs:
                dst_fp = fp.replace(".tif", "_reproj.tif")
                transform, width, height = calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds
                )
                kwargs = src.meta.copy()
                kwargs.update({"crs": target_crs, "transform": transform,
                               "width": width, "height": height})
                with rasterio.open(dst_fp, "w", **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=target_crs,
                            resampling=Resampling.nearest
                        )
                reprojected_files.append(dst_fp)
            else:
                reprojected_files.append(fp)
    return reprojected_files

def merge_all_bands(root_dir=r"target_file",
                    bands=("B2","B3","B4","B8","B8A","B11","B12"),
                    merged_dir="merged_bands",
                    final_path="Sentinel2_merged.tif"):
    os.makedirs(merged_dir, exist_ok=True)
    band_files = {}
    for band in bands:
        pattern = os.path.join(root_dir, f"{band}*.tif")
        candidates = sorted(glob.glob(pattern))
        tile_files = [fp for fp in candidates if "_tile" in os.path.basename(fp)]
        if tile_files:
            files = tile_files
        else:
            non_reproj = [fp for fp in candidates if not fp.endswith("_reproj.tif")]
            files = non_reproj if non_reproj else candidates
        if not files:
            print(f"⚠️ 没有找到 {band} 文件")
            continue
        print(f"🔹 合并 {band}: {len(files)} 个文件")
        reprojected_files = reproject_to_match(files)
        src_files = [rasterio.open(fp) for fp in reprojected_files]
        mosaic, out_trans = rio_merge(src_files)
        out_meta = src_files[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "count": 1
        })
        out_path = os.path.join(merged_dir, f"{band}.tif")
        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(mosaic[0], 1)
        band_files[band] = out_path
        for src in src_files:
            src.close()

    print("✅ 单波段合并完成")
    if not band_files:
        print(f"⚠️ 没有找到任何波段文件，无法生成多波段文件: {final_path}")
        print("🎉 全部流程完成: None")
        return None

    with rasterio.open(list(band_files.values())[0]) as src0:
        meta = src0.meta.copy()
    meta.update(count=len(band_files), driver="GTiff")
    with rasterio.open(final_path, "w", **meta) as dest:
        for idx, (band, path) in enumerate(band_files.items(), start=1):
            with rasterio.open(path) as src:
                dest.write(src.read(1), idx)
            dest.set_band_description(idx, band)
    print(f"🎉 最终多波段文件已生成: {final_path}")
    return merged_dir

def find_max_size(folder='gee_sentinel2/1source_tif',
                  size_limit=15 * 1024 * 1024,
                  move_dir=r"target_file_10"):
    os.makedirs(move_dir, exist_ok=True)
    large_files, file_dict = [], {}
    for root, dirs, files in os.walk(folder):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                filesize = os.path.getsize(filepath)
            except OSError as e:
                print(f"无法获取文件大小: {filepath}, 错误: {e}")
                continue
            if filesize > size_limit:
                if file not in file_dict or filesize > file_dict[file][1]:
                    file_dict[file] = (filepath, filesize)

    for file, (filepath, filesize) in file_dict.items():
        target_path = os.path.join(move_dir, file)
        if not os.path.exists(target_path):
            shutil.move(filepath, target_path)
        large_files.append((filepath, filesize))

    if large_files:
        print("以下文件大于 15 MB，并已移动 (只保留最大重复文件)：")
        for f, s in large_files:
            print(f"{f} - {s / (1024*1024):.2f} MB")
    else:
        print("没有文件大于 15 MB")
    return move_dir

# ====================== Checkpoint I/O ======================
def _ckpt_path(project_root):
    return os.path.join(project_root, "checkpoint.json")

def load_checkpoint(project_root):
    path = _ckpt_path(project_root)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            d.setdefault("processed", [])
            if "monthly_best" not in d:
                # 向后兼容旧的 checkpoint 结构
                legacy = d.pop("daily_candidates", {})
                d["monthly_best"] = {}
                for meta in legacy.values():
                    ts = meta.get("timestamp")
                    if not ts:
                        continue
                    month_key = ts[:7]
                    d["monthly_best"][month_key] = {
                        "merged_path": meta.get("merged_path"),
                        "valid_ratio": meta.get("valid_ratio"),
                        "timestamp": ts,
                        "size": meta.get("size", 0)
                    }
            d.pop("daily_candidates", None)
            return d
        except Exception:
            pass
    return {"processed": [], "monthly_best": {}}

def save_checkpoint(project_root, ckpt):
    path = _ckpt_path(project_root)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def mark_done(project_root):
    with open(os.path.join(project_root, "_DONE"), "w", encoding="utf-8") as f:
        f.write(datetime.now().isoformat())

def is_done(project_root):
    return os.path.exists(os.path.join(project_root, "_DONE"))

# ====================== 主工作函数：单窗口 ======================
def process_download(project_root="gee_sentinel2",
                     start_date='2024-10-30',
                     end_date='2024-11-10',
                     roi_len=[119.8, 30.9, 120.6, 31.6],
                     bands_to_download=("B2","B3","B4","B8","B8A","B11","B12"),
                     page_size=10,
                     TARGET_MB=2137.0,          # 要保留的“2137 MB”
                     TOL_MB=1,             # 容差（±1 MB）
                     nx=2, ny=2):
    """
    单个时间窗口内的完整流程：筛影像 -> 按波段下载(可分块) -> 合并多波段 -> 月度内保留最大的多波段 TIFF -> 断点续跑
    返回：月度最大 TIFF 的元数据 dict 或 None
    """
    target_month = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m")
    roi = ee.Geometry.Rectangle(roi_len)
    max_cloud = 5
    src_root = os.path.join(project_root, "1source_tif")
    merged_root = os.path.join(project_root, "2merged_tif")
    os.makedirs(src_root, exist_ok=True)
    os.makedirs(merged_root, exist_ok=True)

    log_filename = os.path.join(project_root, f"download_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    log_file = open(log_filename, "w", encoding="utf-8")
    sys.stdout = Tee(log_file)

    print(f"=== 窗口: {start_date} ~ {end_date} ===")
    print(f"ROI: {roi_len} | 波段: {list(bands_to_download)} | 分块: {nx}x{ny}")

    ckpt = load_checkpoint(project_root)
    processed = set(ckpt.get("processed", []))
    monthly_best = ckpt.get("monthly_best", {})

    # 已完成窗口直接跳过
    if is_done(project_root):
        print("✅ 检测到 _DONE，跳过该窗口。")
        best_entry = monthly_best.get(target_month)
        result = dict(best_entry) if isinstance(best_entry, dict) else None
        sys.stdout = sys.__stdout__
        log_file.close()
        return result

    col = (ee.ImageCollection("COPERNICUS/S2_SR")
           .filterDate(start_date, end_date)
           .filterBounds(roi)
           .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
           .sort("system:time_start"))

    count = safe_get_collection_size(col)
    print(f"✅ 影像数量：{count if count is not None else '未知（将分页迭代）'}")

    # —— 优先：aggregate 一次取回全部时间戳 —— #
    date_strs = None
    if count and count > 0:
        try:
            ts_list_ms = robust_aggregate_times(col, retries=5)
            if ts_list_ms and len(ts_list_ms) == count:
                date_strs = [format_ms_to_datestr(ms) for ms in ts_list_ms]
        except Exception as e:
            print(f"⚠️ 聚合时间失败，改用分页方案：{e}")

    def _handle_one(img, date_str):
        nonlocal ckpt, processed, monthly_best
        if date_str in processed:
            print(f"↩️ 已处理过：{date_str}（断点续跑跳过）")
            return

        month_key = date_str[:7]

        img_src_folder = os.path.join(src_root, date_str)
        os.makedirs(img_src_folder, exist_ok=True)

        all_ok = True
        for b in bands_to_download:
            out_band_path = os.path.join(img_src_folder, f"{b}.tif")
            if os.path.exists(out_band_path) and os.path.getsize(out_band_path) > 0:
                continue
            ok = robust_download_band(img, b, out_band_path, roi, nx=nx, ny=ny, session=SESSION)
            if not ok:
                all_ok = False
                break

        if not all_ok or not all_bands_downloaded(img_src_folder, bands_to_download):
            print(f"  ⚠️ 波段不完整，跳过 {date_str}")
            # 记录已处理以避免反复踩坑（也可不记录，看你偏好）
            processed.add(date_str)
            ckpt["processed"] = sorted(processed)
            save_checkpoint(project_root, ckpt)
            cleanup_path(img_src_folder)
            return

        merged_path = os.path.join(merged_root, f"{date_str}.tif")
        try:
            merge_bands(img_src_folder, merged_path, bands_to_download)
        except Exception as e:
            print(f"  ❌ 合并失败：{e}")
            processed.add(date_str)
            ckpt["processed"] = sorted(processed)
            save_checkpoint(project_root, ckpt)
            cleanup_path(img_src_folder)
            cleanup_path(merged_path)
            return

        missing_ratio = calc_missing_ratio_multiband(merged_path)
        valid_ratio = 1 - missing_ratio
        print(f"  有效数据比例 = {valid_ratio:.2%}")

        if missing_ratio > MISSING_THRESH:
            print(f"  ⚠️ 丢弃（缺失 > {MISSING_THRESH:.0%}）：{date_str}")
            cleanup_path(img_src_folder)
            cleanup_path(merged_path)
            processed.add(date_str)
            ckpt["processed"] = sorted(processed)
            save_checkpoint(project_root, ckpt)
            return

        try:
            file_size = os.path.getsize(merged_path)
        except OSError:
            file_size = 0

        # 体积校验：允许 TARGET_MB 或 TOL_MB 为 None；为 None 时不做校验
        def is_TARGET_MB_mb(size_bytes):
            try:
                mb = float(size_bytes) / (1024 * 1024)
            except Exception:
                return False  # size_bytes 异常就判失败
            if TARGET_MB is None:
                return True
            tol = 0.0 if TOL_MB is None else float(TOL_MB)
            try:
                tgt = float(TARGET_MB)
            except Exception:
                return True  # TARGET_MB 非数值：安全起见跳过体积校验
            return abs(mb - tgt) <= tol


        size_mb = file_size / (1024 * 1024) if file_size else 0.0
        keep_entry = True
        old = monthly_best.get(month_key)

        if old:
            old_size = int(old.get("size", 0) or 0)
            old_size_mb = old_size / (1024 * 1024)
            # 1) 若旧记录是“2137 MB”，无条件保留，直接丢弃新候选
            if is_TARGET_MB_mb(old_size):
                print(f"  ↩️ 保留已有 {month_key} 月最佳 {old.get('timestamp')} "
                    f"（≈{old_size_mb:.2f} MB≈2137 MB），丢弃 {date_str}")
                keep_entry = False
            # 2) 否则，只有当“新候选更大”时才更新
            elif file_size and file_size < old_size:
                print(f"  ↪️ 更新 {month_key} 月最佳：{date_str} | {size_mb:.2f} MB "
                    f"(> {old_size_mb:.2f} MB)")
                old_merge = old.get("merged_path")
                if old_merge and old_merge != merged_path and os.path.exists(old_merge):
                    cleanup_path(old_merge)
                old_src = old.get("src_folder")
                if old_src and old_src != img_src_folder and os.path.exists(old_src):
                    cleanup_path(old_src)
                old_final = old.get("final_path")
                # 如需清理最终文件，按需打开：
                # if old_final and os.path.exists(old_final):
                #     cleanup_path(old_final)
            # 3) 新候选不更大 → 保留旧记录
            else:
                print(f"  ↩️ 保留已有 {month_key} 月最佳 {old.get('timestamp')} "
                    f"({old_size_mb:.2f} MB)，丢弃 {date_str}")
                keep_entry = False
        else:
            print(f"  ✅ 记录 {month_key} 月候选：{date_str} | {size_mb:.2f} MB")

        if keep_entry:
            monthly_best[month_key] = {
                "merged_path": merged_path,
                "valid_ratio": float(valid_ratio),
                "timestamp": date_str,
                "size": int(file_size),
                "src_folder": img_src_folder
            }
        else:
            cleanup_path(img_src_folder)
            cleanup_path(merged_path)

        processed.add(date_str)
        ckpt["processed"] = sorted(processed)
        ckpt["monthly_best"] = monthly_best
        save_checkpoint(project_root, ckpt)

    # —— 路径 A：aggregate 成功 —— #
    if date_strs is not None:
        image_list = col.toList(count)
        for i in range(count):
            img = ee.Image(image_list.get(i))
            date_str = date_strs[i]
            _handle_one(img, date_str)
    else:
        # —— 路径 B：分页兜底（每页一次 getInfo，已含 date_str）—— #
        for img, global_idx, date_str in iter_collection_paginated_with_dates(col, page_size=page_size, start=0):
            if date_str is None:
                # 极端兜底：单条再算一次
                try:
                    date_str = robust_getinfo(
                        lambda: ee.Date(img.get("system:time_start")).format("YYYY-MM-dd_HH-mm-ss").getInfo(),
                        tag="single date_str"
                    )
                except Exception as e:
                    print(f"  ❌ 获取日期失败，跳过该影像：{e}")
                    continue
            _handle_one(img, date_str)

    best_entry = monthly_best.get(target_month)
    if best_entry and best_entry.get("merged_path") and os.path.exists(best_entry["merged_path"]):
        final_dir = os.path.join(project_root, "final_best")
        os.makedirs(final_dir, exist_ok=True)
        final_path = os.path.join(final_dir, f"{target_month}_best.tif")
        need_copy = True
        if os.path.exists(final_path):
            try:
                if os.path.samefile(final_path, best_entry["merged_path"]):
                    need_copy = False
                elif os.path.getsize(final_path) == best_entry.get("size", 0):
                    need_copy = False
            except (FileNotFoundError, OSError):
                need_copy = True
        if need_copy:
            shutil.copy2(best_entry["merged_path"], final_path)
        best_entry["final_path"] = final_path
        monthly_best[target_month] = best_entry
        ckpt["monthly_best"] = monthly_best
        save_checkpoint(project_root, ckpt)

        print(f"\n✅ 月度最佳影像: {best_entry['timestamp']} | 大小 {best_entry['size'] / (1024*1024):.2f} MB | 输出 {final_path}")

        summary_rows = [{
            "month": target_month,
            "timestamp": best_entry["timestamp"],
            "valid_ratio": best_entry["valid_ratio"],
            "size_bytes": best_entry["size"],
            "merged_path": best_entry["merged_path"],
            "final_path": best_entry.get("final_path", "")
        }]
        df = pd.DataFrame(summary_rows)
        df.to_csv(os.path.join(project_root, f"monthly_best_{target_month}.csv"), index=False, encoding="utf-8")
    else:
        print("\n⚠️ 本窗口未找到符合条件的影像。")
        best_entry = None

    # 窗口完成标记
    mark_done(project_root)

    # 恢复 stdout
    sys.stdout = sys.__stdout__
    log_file.close()
    result = dict(best_entry) if isinstance(best_entry, dict) else None
    return result

# # ====================== 主程序（多窗口循环 + 窗口级断点） ======================
# if __name__ == "__main__":
#     # 逐月遍历 2022-01 至 2023-01
#     month_range = pd.period_range("2020-01", "2020-05", freq="M")
#     bonds = ["B2", "B3", "B4", "B5", "B8", "B8A", "B9", "B11", "B12"]
#     for month in month_range:
#         start_date = month.to_timestamp(how="start").strftime("%Y-%m-%d")
#         end_date = (month + 1).to_timestamp(how="start").strftime("%Y-%m-%d")
#         month_label = month.strftime("%Y-%m")
#         project_root = f"gee_sentinel3/{month_label}"

#         print(f"\n🚀 开始处理月份: {month_label}")
#         os.makedirs(project_root, exist_ok=True)

#         best_meta = process_download(
#             project_root=project_root,
#             start_date=start_date,
#             end_date=end_date,
#             roi_len=[119.8, 30.9, 120.6, 31.6],
#             bands_to_download=tuple(bonds),
#             page_size=10,
#             TARGET_MB=2137.0,
#             TOL_MB=1.0,
#             nx=None, ny=None
#         )

#         if not best_meta:
#             print(f"⚠️ {month_label} 未找到合格影像")
#             continue

#         final_path = best_meta.get("final_path") or best_meta.get("merged_path")
#         print(f"✅ {month_label} 输出: {final_path}")

#     print("\n🎉 全部月份处理结束")



# ====================== 主程序 mutipile（多窗口循环 + 窗口级断点） ======================
if __name__ == "__main__":
    import math
    import os
    import pandas as pd
    import numpy as np

    # 逐月遍历 2022-01 至 2023-01（含 2023-01，前闭后闭）
    month_range = pd.period_range("2020-07", "2020-09", freq="M")

    # Sentinel-2 波段清单（你原来叫 bonds，保留变量名）
    bonds = ["B2", "B3", "B4", "B5", "B8", "B8A", "B9", "B11", "B12"]

    # 读取湖泊 bbox 信息
    df = pd.read_excel("target_lake_info.xlsx")

    for idx, row in df.iterrows():
        # 读列
        lake_id = row.get("Hylak_id")
        min_lat = row.get("min_lat")
        max_lat = row.get("max_lat")
        min_lon = row.get("min_lon")
        max_lon = row.get("max_lon")
        lake_area=row.get("Lake_area")
        if lake_area > 5000:
            continue
        
        # 过滤空值
        if pd.isna(lake_id) or any(pd.isna(v) for v in [min_lat, max_lat, min_lon, max_lon]):
            print(f"⏭️ 行 {idx}: lake_id/bbox 含 NA，跳过")
            continue
        
        # 校正 bbox
        roi_list = [min_lon, min_lat, max_lon, max_lat]

        for month in month_range:
            start_date = month.to_timestamp(how="start").strftime("%Y-%m-%d")
            # 这里 end_date 取下一个月的月初（GEE 常用半开区间 [start, end)）
            end_date = (month + 1).to_timestamp(how="start").strftime("%Y-%m-%d")
            month_label = month.strftime("%Y-%m")
            project_root = f"gee_nature3/{int(lake_id)}/{month_label}"
            os.makedirs(project_root, exist_ok=True)

            print(f"\n🚀 开始处理: lake={lake_id}, 月份={month_label}, ROI={roi_list}")

            best_meta = process_download(
                project_root=project_root,
                start_date=start_date,
                end_date=end_date,
                roi_len=tuple(roi_list),          # ✅ 用当前湖泊的 bbox
                bands_to_download=tuple(bonds),
                page_size=10,
                TARGET_MB=None,
                TOL_MB=1.0,
                nx=None, ny=None
            )

            if not best_meta:
                print(f"⚠️ {month_label} 未找到合格影像")
                continue

            final_path = best_meta.get("final_path") or best_meta.get("merged_path")
            print(f"✅ {month_label} 输出: {final_path}")

    print("\n🎉 全部月份处理结束")
