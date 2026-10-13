"""Google Earth Engine download utilities for Sentinel-2 imagery."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import ee
import numpy as np
import rasterio
import requests
from rasterio.merge import merge as rio_merge
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.warp import reproject, Resampling
LOGGER = logging.getLogger(__name__)

MAX_RETRIES = 5
TIMEOUT_SEC = 180
MISSING_THRESH = 0.40
BUCKET_MINUTES = 5



import pandas as pd  # 补上




# —— 可选：把 EE 请求的超时阈值调大一些 ——
def _bump_ee_deadline():
    for fn, val in [
        (getattr(ee.data, "setDeadline", None), 300),                 # 旧版 API（秒）
        (getattr(ee.data, "setCloudApiDeadlineSeconds", None), 300),  # 新版 Cloud API（秒）
    ]:
        try:
            if fn is not None:
                fn(val)
        except Exception:
            pass

def safe_get_collection_size(col, retries=2, sleep_base=2):
    import time
    for k in range(retries + 1):
        try:
            return int(ee.Number(col.size()).getInfo())
        except Exception:
            if k == retries:
                return None
            time.sleep(sleep_base * (k + 1))

def iter_collection_paginated(col, page_size=10, start=0):
    """逐页返回 (ee.Image, 全局索引)，每次只做小 getInfo()。"""
    i = int(start)
    while True:
        lst = col.toList(page_size, i)
        try:
            current = lst.getInfo()  # 仅取这一页的轻量描述
        except Exception:
            break
        if not current:
            break
        n = len(current)
        for j in range(n):
            yield ee.Image(lst.get(j)), (i + j)
        i += n
        
        
        
def initialize_ee(project: Optional[str] = None) -> None:
    try:
        ee.Initialize(project=project)
        _bump_ee_deadline()
    except ee.EEException as exc:
        raise RuntimeError("Failed to initialise Google Earth Engine") from exc


@dataclass
class DownloadResult:
    """Container storing information for the best scene within a 5-minute bucket."""

    path: Path
    valid_ratio: float
    timestamp: str


class _CompositeStream:
    """Stream that replicates writes to multiple targets."""

    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> None:
        for stream in self._streams:
            stream.write(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def split_roi(roi: ee.Geometry, nx: int = 2, ny: int = 2) -> List[ee.Geometry]:
    coords = roi.bounds().coordinates().getInfo()[0]
    minx, miny = coords[0]
    maxx, maxy = coords[2]
    dx = (maxx - minx) / nx
    dy = (maxy - miny) / ny
    tiles: List[ee.Geometry] = []
    for i in range(nx):
        for j in range(ny):
            tiles.append(
                ee.Geometry.Rectangle(
                    [minx + i * dx, miny + j * dy, minx + (i + 1) * dx, miny + (j + 1) * dy]
                )
            )
    return tiles


def cleanup_path(path: Path) -> None:
    if path.is_dir():
        for child in path.iterdir():
            cleanup_path(child)
        path.rmdir()
    elif path.exists():
        path.unlink()



S2_SCALE = {
    "B1": 60,   # Sentinel-2 SR: B1 60m
    "B2": 10,
    "B3": 10,
    "B4": 10,
    "B5": 20,
    "B6": 20,
    "B7": 20,
    "B8": 10,
    "B8A": 20,
    "B9": 60,
    "B11": 20,
    "B12": 20,
}

def robust_download_band(img, band, out_path, region, nx=1, ny=1):
    scale = S2_SCALE.get(band, 10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok_any = True
    if nx > 1 or ny > 1:
        ok_any = True
        for idx, tile in enumerate(split_roi(region, nx=nx, ny=ny)):
            tile_path = out_path.with_name(f"{out_path.stem}_tile{idx}{out_path.suffix}")
            try:
                url = img.select(band).clip(tile).getDownloadURL(
                    {"scale": scale, "region": tile, "format": "GEO_TIFF"}
                )
                r = requests.get(url, stream=True, timeout=TIMEOUT_SEC)
                if r.status_code != 200:
                    LOGGER.warning("[DL] %s tile%02d HTTP %s", band, idx, r.status_code)
                    ok_any = False
                    continue
                with open(tile_path, "wb") as fh:
                    for chunk in r.iter_content(8192):
                        if chunk: fh.write(chunk)
                with rasterio.open(tile_path): pass
                LOGGER.info("[DL] %s tile%02d OK (scale=%dm)", band, idx, scale)
            except Exception as e:
                LOGGER.warning("[DL] %s tile%02d FAILED: %s", band, idx, e)
                ok_any = False
        return ok_any

    # 单块下载重试
    for attempt in range(1, MAX_RETRIES+1):
        try:
            url = img.select(band).clip(region).getDownloadURL(
                {"scale": scale, "region": region, "format": "GEO_TIFF"}
            )
            r = requests.get(url, stream=True, timeout=TIMEOUT_SEC)
            if r.status_code == 200:
                with open(out_path, "wb") as fh:
                    for chunk in r.iter_content(8192):
                        if chunk: fh.write(chunk)
                with rasterio.open(out_path): pass
                LOGGER.info("[DL] %s OK (scale=%dm)", band, scale)
                return True
            LOGGER.warning("[DL] %s attempt %d/%d HTTP %s", band, attempt, MAX_RETRIES, r.status_code)
        except Exception as e:
            LOGGER.warning("[DL] %s attempt %d/%d FAILED: %s", band, attempt, MAX_RETRIES, e)
        time.sleep(3 * attempt)
    return False


def _band_paths(folder: Path, band: str) -> list[Path]:
    """Find a band's original files without confusing B8 with B8A.

    Reprojected intermediates are excluded so repeat merges do not mosaic the
    same tile twice.
    """
    pattern = re.compile(rf"{re.escape(band)}(?:_tile\d+)?\.(?:tif|tiff)", re.IGNORECASE)
    return sorted(path for path in folder.iterdir() if path.is_file() and pattern.fullmatch(path.name))


def _validate_band_order(bands: Sequence[str]) -> tuple[str, ...]:
    from .workflows import SENTINEL2_BANDS

    if isinstance(bands, (str, bytes)):
        raise ValueError("bands must be a sequence of unique Sentinel-2 band names")
    try:
        band_order = tuple(bands)
        if not band_order or len(set(band_order)) != len(band_order) or any(b not in SENTINEL2_BANDS for b in band_order):
            raise ValueError
    except (ValueError, TypeError) as exc:
        raise ValueError("bands must be a sequence of unique Sentinel-2 band names") from exc
    return band_order


def all_bands_downloaded(folder: Path, bands: Sequence[str]) -> bool:
    return all(_band_paths(folder, band) for band in bands)



def _read_mosaic(paths: list[Path]):
    """读入一组 tile 并 mosaic，返回 (array, transform, crs)"""
    with contextlib.ExitStack() as stack:
        ds = [stack.enter_context(rasterio.open(p)) for p in paths]
        mosaic, out_trans = rio_merge(ds)       # (1, H, W)
        return mosaic[0].astype(np.float32), out_trans, ds[0].crs

def merge_bands(src_folder: Path, out_path: Path, bands: Sequence[str]) -> Path:
    bands = _validate_band_order(bands)
    src_folder, out_path = Path(src_folder), Path(out_path)
    # 1) 为每个波段找到 tiles
    band_tiles: dict[str, list[Path]] = {}
    for b in bands:
        files = _band_paths(src_folder, b)
        if not files:
            raise FileNotFoundError(f"Band {b} not found under {src_folder}")
        band_tiles[b] = files

    # 2) 选择参考网格：优先 10m 的 B2/B3/B4/B8；否则用列表中第一个
    pref = [x for x in ["B2","B3","B4","B8"] if x in band_tiles]
    ref_band = pref[0] if pref else bands[0]
    ref_arr, ref_transform, ref_crs = _read_mosaic(band_tiles[ref_band])

    height, width = ref_arr.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": len(bands),
        "dtype": "float32",
        "crs": ref_crs,
        "transform": ref_transform,
        "compress": "lzw",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        for idx, b in enumerate(bands, start=1):
            arr, tr, crs = _read_mosaic(band_tiles[b])

            # 3) 重采样到参考网格（连续光谱用双线性）
            dst_arr = np.zeros((height, width), dtype=np.float32)
            reproject(
                source=arr,
                destination=dst_arr,
                src_transform=tr,
                src_crs=crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear,
            )
            dst.write(dst_arr, idx)
            # 4) 写入 band 描述
            dst.set_band_description(idx, b)

    return out_path


def floor_to_bucket_5min(date_str: str) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d_%H-%M-%S")
    floored_minute = (dt.minute // BUCKET_MINUTES) * BUCKET_MINUTES
    dt_bucket = dt.replace(minute=floored_minute, second=0)
    return dt_bucket.strftime("%Y-%m-%d_%H-%M")


def calc_missing_ratio_multiband(tif_path: Path) -> float:
    with rasterio.open(tif_path) as src:
        data = src.read()
        nodata = src.nodata
        if nodata is not None:
            invalid = np.all(data == nodata, axis=0)
        else:
            invalid = np.all(data == 0, axis=0)
    return float(invalid.sum() / invalid.size)

def find_max_size(
    folder: Path | str,
    size_limit: int,
    move_dir: Path | str,
) -> Path:
    folder = Path(folder)
    move_dir = Path(move_dir)
    move_dir.mkdir(parents=True, exist_ok=True)

    file_dict = {}
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError:  # pragma: no cover - filesystem dependent
            continue
        if size > size_limit:
            record = file_dict.get(path.name)
            if record is None or size > record[1]:
                file_dict[path.name] = (path, size)

    for _, (src_path, _) in file_dict.items():
        target_path = move_dir / src_path.name
        if not target_path.exists():
            src_path.replace(target_path)

    return move_dir


def reproject_to_match(files: Sequence[Path | str], target_crs: str) -> List[Path]:
    out_files: List[Path] = []
    for file in files:
        path = Path(file)
        with rasterio.open(path) as src:
            if src.crs and src.crs.to_string() == target_crs:
                out_files.append(path)
                continue
            dst_fp = path.with_name(f"{path.stem}_reproj{path.suffix}")
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update({"crs": target_crs, "transform": transform, "width": width, "height": height})
            with rasterio.open(dst_fp, "w", **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.nearest,
                    )
                    if src.descriptions[i - 1]:
                        dst.set_band_description(i, src.descriptions[i - 1])
            out_files.append(dst_fp)
    return out_files


def merge_all_bands(
    root_dir: Path | str,
    bands: Sequence[str],
    merged_dir: Path | str,
    final_path: Path | str,
    target_crs: str = "EPSG:4326",
) -> Optional[Path]:
    root_dir = Path(root_dir)
    merged_dir = Path(merged_dir)
    final_path = Path(final_path)
    bands = _validate_band_order(bands)

    source_files = {band: _band_paths(root_dir, band) for band in bands}
    if not any(source_files.values()):
        LOGGER.warning("No band files were found under %s", root_dir)
        return None
    missing = [band for band, files in source_files.items() if not files]
    if missing:
        raise FileNotFoundError(f"Missing bands {missing} under {root_dir}")

    merged_dir.mkdir(parents=True, exist_ok=True)

    for band in bands:
        files = source_files[band]
        print(f"🔹 合并 {band}: {len(files)} 个文件")
        
        reproj_files = reproject_to_match(files, target_crs)
        with contextlib.ExitStack() as stack:
            datasets = [stack.enter_context(rasterio.open(fp)) for fp in reproj_files]
            mosaic, out_trans = rio_merge(datasets)
            out_meta = datasets[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "count": 1,
        })
        out_path = merged_dir / f"{band}.tif"
        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(mosaic[0], 1)
            dest.set_band_description(1, band)

    # Resample every band's native grid onto one reference grid and stamp the
    # requested descriptions. Native 10/20/60 m band sizes can differ.
    return merge_bands(merged_dir, final_path, bands)

def process_download(
    project_root: Path | str,
    start_date: str,
    end_date: str,
    roi_bounds: Sequence[float],
    bands_to_download: Sequence[str],
    max_cloud: float = 5.0,
    log_to_file: bool = True,
) -> Dict[str, DownloadResult]:
    bands_to_download = _validate_band_order(bands_to_download)
    project_root = Path(project_root)
    src_root = project_root / "1source_tif"
    merged_root = project_root / "2merged_tif"
    src_root.mkdir(parents=True, exist_ok=True)
    merged_root.mkdir(parents=True, exist_ok=True)

    roi = ee.Geometry.Rectangle(list(roi_bounds))
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        .sort("system:time_start")
    )
    log_file = None
    if log_to_file:
        log_file = open(project_root / "download_log.txt", "w", encoding="utf-8")

    streams = [sys.stdout]
    if log_file:
        streams.append(log_file)

    composite = _CompositeStream(*streams)
    daily_candidates: Dict[str, DownloadResult] = {}

    try:
        with contextlib.redirect_stdout(composite):
            # —— 计数改为“尽量拿到”，拿不到就继续分页遍历 ——
            count = safe_get_collection_size(collection)
            if count is not None:
                print(f"Found ~{count} images in range [{start_date}, {end_date}]. (quick count)")
            else:
                print(f"Found images in range [{start_date}, {end_date}] (skip exact count, paginating).")
            page_size = 10
            seen = 0
            for img, idx in iter_collection_paginated(collection, page_size=page_size):
                seen += 1
                date_str = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd_HH-mm-ss").getInfo()
                scene_index = img.get("system:index").getInfo()
                raw_scene_id = str(scene_index or idx)
                safe_scene_id = re.sub(r"[^A-Za-z0-9_-]", "_", raw_scene_id)[:80]
                scene_digest = hashlib.sha256(raw_scene_id.encode()).hexdigest()[:12]
                scene_id = f"{safe_scene_id}_{scene_digest}"
                scene_name = f"{date_str}_{scene_id}"
                bucket_key = floor_to_bucket_5min(date_str)
                prior = daily_candidates.get(bucket_key)
                if prior is not None and prior.path.name == f"{scene_name}.tif":
                    # A duplicate collection record must never overwrite or
                    # delete the same bucket candidate's retained output.
                    continue
                img_src_folder = src_root / scene_name
                img_src_folder.mkdir(parents=True, exist_ok=True)

                all_ok = True
                for band in bands_to_download:
                    out_band_path = img_src_folder / f"{band}.tif"
                    if out_band_path.exists() and out_band_path.stat().st_size > 0:
                        continue
                    ok = robust_download_band(img, band, out_band_path, roi, nx=2, ny=2)
                    if not ok:
                        all_ok = False
                        break
                # 下载循环后
                missing = [b for b in bands_to_download if not _band_paths(img_src_folder, b)]
                if missing:
                    print(f"  ❌ 缺失波段：{missing}，丢弃 {date_str}")
                    cleanup_path(img_src_folder)
                    continue
                
                if not all_ok or not all_bands_downloaded(img_src_folder, bands_to_download):
                    print(f"  ⚠️ 波段不完整，跳过 {date_str}")
                    continue

                merged_path = merged_root / f"{scene_name}.tif"
                try:
                    merge_bands(img_src_folder, merged_path, bands_to_download)
                except Exception as err:
                    print(f"  ❌ 合并失败：{err}")
                    continue

                missing_ratio = calc_missing_ratio_multiband(merged_path)
                valid_ratio = 1 - missing_ratio
                print(f"  有效数据比例 = {valid_ratio:.2%}")

                if missing_ratio > MISSING_THRESH:
                    print(f"  ⚠️ 丢弃（缺失 > {MISSING_THRESH:.0%}）：{date_str}")
                    cleanup_path(img_src_folder)
                    merged_path.unlink(missing_ok=True)
                    continue

                existing = daily_candidates.get(bucket_key)
                if existing is None or valid_ratio > existing.valid_ratio:
                    if existing:
                        print(f"  ↪️ 替换桶 {bucket_key} 之前的 {existing.timestamp} -> {date_str}")
                        existing.path.unlink(missing_ok=True)
                    daily_candidates[bucket_key] = DownloadResult(merged_path, valid_ratio, date_str)
                    if all_ok:
                        print("  ✅ 所有波段下载完成")
                else:
                    print(f"  ↩️ 保留桶 {bucket_key} 里已有更优影像，丢弃 {date_str}")
                    cleanup_path(img_src_folder)
                    merged_path.unlink(missing_ok=True)

            print(f"\n✅ 分页遍历完成，共处理 {seen} 张")
            print("✅ 最终保留影像：")
            for bucket, result in sorted(daily_candidates.items()):
                print(f"  {bucket} -> {result.timestamp} | 有效比例 {result.valid_ratio:.2%} | 路径 {result.path}")
    finally:
        if log_file:
            log_file.close()

    return daily_candidates



