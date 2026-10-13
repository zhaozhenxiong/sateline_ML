# live_preview.py
# 示例：
#   监视单个文件：python live_preview.py "E:\data\sateline.tif" --rgb 4 3 2
#   监视文件夹最新TIF：python live_preview.py "E:\data\mosaic_dir"
#   假彩色预览：python live_preview.py "E:\data\sateline.tif" --falsecolor
#   单波段：python live_preview.py "E:\data\dem.tif" --band 1

import argparse
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
# --- auto backend switch (put this at the very top of tif_viwer.py) ---
import sys, matplotlib
if "matplotlib.pyplot" not in sys.modules:
    for cand in ("QtAgg", "MacOSX", "TkAgg"):
        try:
            if "agg" in matplotlib.get_backend().lower():
                matplotlib.use(cand, force=True)
            break
        except Exception:
            continue
# ----------------------------------------------------------------------
import matplotlib.pyplot as plt
# 可选：若要预览 GeoTIFF，建议安装 rasterio；未装也能看 .npy 或 .png
try:
    import rasterio
    from rasterio.enums import Resampling
    _HAS_RIO = True
except Exception:
    _HAS_RIO = False

def percentile_stretch(arr, mask=None, pmin=2, pmax=98):
    out = np.zeros_like(arr, dtype=np.float32)
    for i in range(arr.shape[0]):
        a = arr[i]
        valid = a[mask] if (mask is not None) else a.ravel()
        if valid.size == 0:
            continue
        lo = np.percentile(valid, pmin)
        hi = np.percentile(valid, pmax)
        hi = hi if hi > lo else (lo + 1e-6)
        out[i] = np.clip((a - lo) / (hi - lo), 0, 1)
    return out

def minmax_stretch(arr, mask=None):
    out = np.zeros_like(arr, dtype=np.float32)
    for i in range(arr.shape[0]):
        a = arr[i]
        valid = a[mask] if (mask is not None) else a.ravel()
        if valid.size == 0:
            continue
        lo = float(np.nanmin(valid))
        hi = float(np.nanmax(valid))
        hi = hi if hi > lo else (lo + 1e-6)
        out[i] = np.clip((a - lo) / (hi - lo), 0, 1)
    return out

def read_downsampled_tif(tif, bands, max_side=2500, stretch="percentile", pmin=2, pmax=98, gamma=1.0):
    """读取 GeoTIFF，最长边压到 max_side，返回 (H,W,3/1) 的可显示数组"""
    if not _HAS_RIO:
        raise RuntimeError("未安装 rasterio，无法读取 GeoTIFF。请先 pip install rasterio。")

    with rasterio.open(tif) as src:
        H, W = src.height, src.width
        scale = max(1, int(np.ceil(max(H, W) / max_side)))
        out_h, out_w = max(1, H // scale), max(1, W // scale)

        arr = src.read(
            indexes=bands,
            out_shape=(len(bands), out_h, out_w),
            resampling=Resampling.bilinear,
        ).astype(np.float32)

        # mask：任一所选波段有效才算有效
        masks = [
            src.read_masks(i, out_shape=(out_h, out_w), resampling=Resampling.nearest) > 0
            for i in bands
        ]
        mask = np.logical_and.reduce(masks) if masks else None

    if stretch == "percentile":
        arr = percentile_stretch(arr, mask=mask, pmin=pmin, pmax=pmax)
    else:
        arr = minmax_stretch(arr, mask=mask)

    if gamma != 1.0:
        arr = np.clip(arr, 0, 1) ** (1.0 / gamma)

    if mask is not None:
        for i in range(arr.shape[0]):
            a = arr[i]
            a[~mask] = np.nan
            arr[i] = a

    if arr.shape[0] == 1:
        return arr[0]  # (H,W)
    else:
        return np.moveaxis(arr[:3], 0, -1)  # (H,W,3)

def read_any(path, rgb=None, band=None, falsecolor=False, **kw):
    """支持：GeoTIFF（需 rasterio）、npy、png/jpg 等"""
    p = Path(path)
    ext = p.suffix.lower()

    if ext in (".tif", ".tiff"):
        if falsecolor:
            bands = [8, 4, 3]  # Sentinel-2 常用假彩
        elif rgb:
            bands = [int(x) for x in rgb]
        elif band:
            bands = [int(band)]
        else:
            # 默认 4-3-2；不够则 3-2-1；再不够就单波段 1
            bands = [4, 3, 2] if _HAS_RIO else [1]

        if not _HAS_RIO:
            raise RuntimeError("要看 GeoTIFF 请安装 rasterio：pip install rasterio")
        return read_downsampled_tif(p, bands=bands, **kw)

    elif ext == ".npy":
        arr = np.load(p)
        # 自动归一化到 [0,1]
        arr = arr.astype(np.float32)
        if arr.ndim == 2:
            mn, mx = np.nanmin(arr), np.nanmax(arr)
            mx = mx if mx > mn else (mn + 1e-6)
            return np.clip((arr - mn) / (mx - mn), 0, 1)
        elif arr.ndim == 3 and arr.shape[2] in (1, 3):
            mn, mx = np.nanmin(arr), np.nanmax(arr)
            mx = mx if mx > mn else (mn + 1e-6)
            return np.clip((arr - mn) / (mx - mn), 0, 1)
        else:
            raise ValueError("npy 需为 (H,W) 或 (H,W,1/3) 形状")
    else:
        # 尝试用 matplotlib 读 png/jpg 等
        img = plt.imread(str(p))
        return img

def get_latest_tif_in_dir(d: Path):
    cands = list(d.glob("*.tif")) + list(d.glob("*.tiff"))
    if not cands:
        return None
    return max(cands, key=lambda x: x.stat().st_mtime)

def live_view(target: Path, poll=0.5, **read_kwargs):
    import time
    import matplotlib.pyplot as plt

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis("off")
    im = None
    last_sig = None  # (path, mtime)

    print("[Hint] 按键：q 退出，r 强制刷新，s 保存PNG")

    def draw(path: Path):
        nonlocal im
        img = read_any(path, **read_kwargs)
        ax.set_title(f"{path.name}  ({time.strftime('%H:%M:%S')})", fontsize=10)
        if im is None:
            im = ax.imshow(img)
        else:
            im.set_data(img)
        fig.canvas.draw_idle()

    def current_target():
        if target.is_file():
            return target if target.exists() else None
        else:
            return get_latest_tif_in_dir(target)

    # ✅ 只注册一次键盘事件
    def on_key(event):
        nonlocal last_sig
        if event.key in ("q", "escape"):
            plt.close(fig)
        elif event.key == "r":
            last_sig = None  # 强制刷新
        elif event.key == "s":
            out = Path("live_snapshot.png")
            fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0)
            print(f"[OK] 已保存：{out.resolve()}")

    fig.canvas.mpl_connect("key_press_event", on_key)

    # 主循环
    while plt.fignum_exists(fig.number):
        cur = current_target()
        if cur is not None:
            sig = (str(cur), cur.stat().st_mtime)
            if sig != last_sig:
                try:
                    draw(cur)
                    last_sig = sig
                except Exception as e:
                    print(f"[WARN] 刷新失败：{e}")
        else:
            ax.set_title("未找到可预览的文件", fontsize=10)
            fig.canvas.draw_idle()

        plt.pause(poll)  # 让 UI 响应 + 轮询
        
        
def main():
    ap = argparse.ArgumentParser(description="用 matplotlib 实时查看（自动刷新）TIF/图片/NPY")
    ap.add_argument("path", help="文件或目录路径（目录下将显示最新的 .tif/.tiff）")
    ap.add_argument("--rgb", nargs=3, type=int, help="RGB 波段，如 --rgb 4 3 2（仅 TIF）")
    ap.add_argument("--band", type=int, help="单波段，如 --band 1（仅 TIF）")
    ap.add_argument("--falsecolor", action="store_true", help="假彩 8-4-3（仅 TIF）")
    ap.add_argument("--stretch", choices=["percentile", "minmax"], default="percentile", help="拉伸")
    ap.add_argument("--pmin", type=float, default=2.0, help="百分位下限")
    ap.add_argument("--pmax", type=float, default=98.0, help="百分位上限")
    ap.add_argument("--gamma", type=float, default=1.0, help="伽马校正（>1 变亮）")
    ap.add_argument("--max-side", type=int, default=2500, help="最长边像素上限（仅 TIF）")
    ap.add_argument("--poll", type=float, default=0.5, help="轮询间隔秒")
    args = ap.parse_args()

    target = Path(args.path)
    if not target.exists():
        raise SystemExit(f"路径不存在：{target}")

    live_view(
        target,
        poll=args.poll,
        rgb=args.rgb,
        band=args.band,
        falsecolor=args.falsecolor,
        stretch=args.stretch,
        pmin=args.pmin,
        pmax=args.pmax,
        gamma=args.gamma,
        max_side=args.max_side,
    )

if __name__ == "__main__":
    main()