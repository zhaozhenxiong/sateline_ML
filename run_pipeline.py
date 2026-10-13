"""End-to-end Sentinel-2 floating debris + plastic spectral pipeline."""

from __future__ import annotations
from typing import Optional, Tuple, List, Sequence
import re
import argparse

from pathlib import Path
import os
import sys
import multiprocessing as mp
import torch.multiprocessing as tmp
import numpy as np
import rasterio

from sateline_ml.detection import (
    floating_debris_detection, create_mask_tif_from_shp)

# Env: MPS fallback (effective on macOS; no effect on Windows/NVIDIA)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

# -------------------- Config --------------------
# Optional plastic spectral classifier (kept as the existing saved checkpoint file)
PLASTIC_MODEL = Path('models')
PLASTIC_LIBRARY = Path("spectral_library/plastic_spectra.csv")
PLASTIC_THRESH = 0.9

# Detection parameters
EPOCHS = 20
REPEATS = 3  # repeat training runs and keep the best by val loss
VAL_SPLIT = 0.25
MODEL_OUT_DIR = "mydata"

ABSORBANCE_THRESH = 0.6
WATER_NDWI_THRESH = 0.1
FLOATING_THRESH = 0.7
SUSPENDED_THRESH = 0.7

TILE_SIZE = 256
STRIDE = 128
MAX_PATCHES = 3072
MIN_VALID_FRAC = 0.5
POS_FRAC = 0.9
PATCH_SEED = 42

AUTO_CALIBRATE_THRESHOLDS = True
AUTO_THRESHOLD_METRIC = "balanced_accuracy"
AUTO_THRESHOLD_MAX_SAMPLES = 128
AUTO_THRESHOLD_BATCH_SIZE = 2
AUTO_THRESHOLD_CANDIDATES = np.linspace(0.3, 0.98, 36)
# stacking
STACKING_PRESET = "high_recall_plastic"
# "fast_linear"  # or None
STACKING_SAMPLE_LIMIT = 40_000
STACKING_PIXEL_STRIDE = 3  # subsample stacking candidates on a coarser grid
STACKING_MAX_PIXELS = 150_000
STACKING_CV_FOLDS = 7

# band order used to stamp descriptions if missing
DEFAULT_BAND_ORDER = ("B2", "B3", "B4", "B5", "B8", "B8A", "B9", "B11", "B12")

# ---- Strict candidate controls (NEW) ----
STRICT_MODE = True       # 更严谨候选
UNC_MARGIN = 0.1          # 悬/漂概率差的最小边际 |p_f - p_s|
MIN_AREA_FLOAT = 10          # 漂浮连通域最小像素数
MIN_AREA_SUSP = 24          # 悬浮连通域最小像素数
CAP_FLOAT_PIXELS = 300_000   # 漂浮伪标签上限（None 取消上限）
CAP_SUSP_PIXELS = 120_000   # 悬浮伪标签上限（None 取消上限）


def ensure_band_descriptions(tif_path: Path, band_order: tuple[str, ...] = DEFAULT_BAND_ORDER) -> None:
    """Stamp band descriptions when missing so detection keeps thermal/SWIR channels."""
    with rasterio.open(tif_path, "r+") as dst:
        descriptions = dst.descriptions or ()
        if descriptions and all(desc for desc in descriptions[: dst.count]):
            return
        if dst.count != len(band_order):
            raise ValueError(
                f"{tif_path} has {dst.count} bands but expected {len(band_order)}; "
                "update DEFAULT_BAND_ORDER to match this product."
            )
        for idx, band in enumerate(band_order, start=1):
            dst.set_band_description(idx, band)
    print(f"[INFO] stamped band descriptions on {tif_path.name}")


def pick_device() -> str:
    """Prefer CUDA, then MPS, otherwise CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def run_one(input_tif: Path, shapefile: Path | None) -> None:
    output_dir = Path(f"mydata/analysis_outputs/{input_tif.stem}")
    debug_dir = output_dir / "debug"
    log_file = output_dir / "run.log"
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_band_descriptions(input_tif, DEFAULT_BAND_ORDER)

    mask_tif = None
    if shapefile and shapefile.exists():
        mask_tif = output_dir / "water_mask.tif"
        try:
            if (
                not mask_tif.exists()
                or os.path.getmtime(shapefile) > os.path.getmtime(mask_tif)
                or os.path.getmtime(input_tif) > os.path.getmtime(mask_tif)
            ):
                create_mask_tif_from_shp(
                    shp_file=shapefile,
                    reference_tif=input_tif,
                    output_mask_tif=mask_tif,
                )
        except Exception as exc:
            print(
                f"[WARN] Failed to rasterize shapefile {shapefile.name}: {exc}")
            mask_tif = None

    device = pick_device()
    print(f"[INFO] processing: {input_tif.name} | device={device}")

    kwargs = dict(
        input_tif=input_tif,
        mask_tif=mask_tif,
        shp_file=shapefile if shapefile and shapefile.exists() else None,
        output_csv=output_dir / "detections.csv",
        preview_png=output_dir / "preview.png",
        debug_dir=debug_dir,
        log_file=log_file,

        # training
        epochs=EPOCHS,
        repeats=REPEATS,
        val_split=VAL_SPLIT,
        model_out_dir=MODEL_OUT_DIR,
        max_patches=MAX_PATCHES,
        tile_size=TILE_SIZE,
        stride=STRIDE,
        min_valid_frac=MIN_VALID_FRAC,
        pos_frac=POS_FRAC,
        patch_seed=PATCH_SEED,
        device=device,
        absorbance_thresh=ABSORBANCE_THRESH,
        # base thresholds
        water_ndwi_thresh=WATER_NDWI_THRESH,
        floating_thresh=FLOATING_THRESH,
        suspended_thresh=SUSPENDED_THRESH,

        # plastic
        plastic_library_csv=PLASTIC_LIBRARY,
        plastic_prob_thresh=PLASTIC_THRESH,

        # auto threshold search
        auto_calibrate_thresholds=AUTO_CALIBRATE_THRESHOLDS,
        auto_threshold_metric=AUTO_THRESHOLD_METRIC,
        auto_threshold_max_samples=AUTO_THRESHOLD_MAX_SAMPLES,
        auto_threshold_batch_size=AUTO_THRESHOLD_BATCH_SIZE,
        auto_threshold_candidates=AUTO_THRESHOLD_CANDIDATES,

        # stacking
        stacking_preset=STACKING_PRESET,
        stacking_sample_limit=STACKING_SAMPLE_LIMIT,
        stacking_pixel_stride=STACKING_PIXEL_STRIDE,
        stacking_max_pixels=STACKING_MAX_PIXELS,
        stacking_cv_folds=STACKING_CV_FOLDS,

        # band layout
        band_order=DEFAULT_BAND_ORDER,

        # ---------- NEW: strict candidate controls ----------
        strict_mode=STRICT_MODE,
        unc_margin=UNC_MARGIN,
        min_area_float=MIN_AREA_FLOAT,
        min_area_susp=MIN_AREA_SUSP,
        cap_float_pixels=CAP_FLOAT_PIXELS,
        cap_susp_pixels=CAP_SUSP_PIXELS,
    )

    if PLASTIC_MODEL and PLASTIC_LIBRARY:
        kwargs |= dict(plastic_model_path=PLASTIC_MODEL)

    floating_debris_detection(**kwargs)
    print(f"[OK] {input_tif.name} -> {output_dir}")


# 你已有：
# from your_module import run_one   # 确保 run_one(tif_path: Path, shapefile: Optional[Path])
ID_PAT = re.compile(r'(?P<id>\d{1,})(?=_)')  # 匹配 792_2022-01.tif 里的 792；至少3位


def extract_id_from_tif(p: Path) -> Optional[str]:
    """从 TIF 文件名中提取湖泊 ID（例如 '792_2022-01_best.tif' -> '792'）。"""
    m = ID_PAT.search(p.stem)
    return m.group('id') if m else None


def pick_shapefile_for_id(shp_dir: Path, lake_id: str) -> Optional[Path]:
    """
    在 mydata/shp/ 下为给定 ID 选择一个 shapefile：
    1) 优先精确匹配 {id}.shp
    2) 其次包含 id 的 .shp（如 taihu_792.shp）
    3) 若同名文件夹下有 .shp，则选择其中最近修改的一个
    返回 None 表示没有可用 SHP（将回退纯谱学/水体掩膜管线）
    """
    if not shp_dir.exists():
        return None

    candidates: List[Path] = []

    # 1) 精确匹配 {id}.shp
    exact = shp_dir / f"{lake_id}.shp"
    if exact.exists():
        candidates.append(exact)

    # 2) 其它包含 id 的 shp
    for shp in shp_dir.glob("*.shp"):
        if shp == exact:
            continue
        if lake_id in shp.stem:
            candidates.append(shp)

    # 3) 同名子目录中寻找 shp（例如 mydata/shp/792/xxx.shp）
    subdir = shp_dir / lake_id
    if subdir.exists() and subdir.is_dir():
        sub_shps = list(subdir.glob("*.shp"))
        candidates.extend(sub_shps)

    if not candidates:
        return None

    # 若有多个候选，取最近修改时间的一个
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end Sentinel-2 floating debris + plastic spectral pipeline.")
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Process all TIFs sequentially without prompting.")
    args = parser.parse_args(argv)

    tif_dir = Path("mydata/tif")
    shp_dir = Path("mydata/shp")

    tifs = sorted(tif_dir.glob("*.tif"))
    if not tifs:
        print(f"[WARN] no tif in {tif_dir.resolve()}")
        return

    # 组装 (tif, shapefile_for_same_id_or_None)
    pairs: List[Tuple[Path, Optional[Path]]] = []
    for tif in tifs:
        lake_id = extract_id_from_tif(tif)
        if not lake_id:
            print(f"[WARN] skip (no ID in name): {tif.name}")
            continue
        shp = pick_shapefile_for_id(shp_dir, lake_id)
        if shp is None:
            print(
                f"[INFO] {tif.name}: no matching SHP for ID={lake_id}; will run without shapefile.")
        else:
            print(f"[INFO] {tif.name}: matched SHP -> {shp.name}")
        pairs.append((tif, shp))

    if not pairs:
        print("[WARN] no valid tif/id pairs found.")
        return

    # 交互：是否全部顺序处理
    process_all = args.yes
    if not process_all:
        ans = input("Process all TIFs sequentially? (y/n) ").strip().lower()
        process_all = ans == "y"

    if not process_all:
        # 仅处理第一对，避免把列表传给 run_one（修复你之前的错误）
        tif, shp = pairs[0]
        try:
            run_one(tif, shp)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[ERR] {tif.name}: {e}", file=sys.stderr)
        return

    # 顺序处理全部；单个失败不影响后续
    for tif, shp in pairs:
        try:
            run_one(tif, shp)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[ERR] {tif.name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)   # Python 多进程
    except RuntimeError:
        pass
    try:
        tmp.set_start_method("spawn", force=True)  # PyTorch 多进程（若用到）
    except RuntimeError:
        pass
    main()
