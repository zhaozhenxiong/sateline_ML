
import pandas as pd
from sateline_ml.gee import(
    initialize_ee,
    process_download,
    find_max_size,
    merge_all_bands
)
if __name__ == "__main__":
    initialize_ee()  # 别忘了初始化

    date_list = pd.date_range("2022-01-01", "2022-03-31", freq="15D")
    roi = [119.8, 30.9, 120.6, 31.6]
    bands = ["B2", "B3", "B4", "B5", "B8", "B8A", "B9", "B11", "B12"]
    for i in range(len(date_list) - 1):
        start_date = date_list[i].strftime("%Y-%m-%d")
        end_date   = date_list[i + 1].strftime("%Y-%m-%d")
        print(f"\n🚀 开始处理: {start_date} ~ {end_date}")

        kept = process_download(
            project_root=f"gee_sentinel3/{start_date}",
            start_date=start_date,
            end_date=end_date,
            roi_bounds=roi,              # ← 统一参数名
            bands_to_download=bands,
            max_cloud=5.0,
            log_to_file=True,
        )
        if not kept:
            print("✅ 本窗口没有保留影像")
            continue

        filtered_dir = find_max_size(
            folder=f"gee_sentinel3/{start_date}/1source_tif",
            size_limit=15 * 1024 * 1024,
            move_dir=f"gee_merge3/{start_date}"
        )
        print(f"✅ 筛选完成: {filtered_dir}")

        # 第三步：合并波段
        merged_dir = merge_all_bands(
            root_dir=f"gee_merge3/{start_date}",
            bands=bands,
            merged_dir=f"gee_merge3/{start_date}/merged_bands",
            final_path=f"gee_merge3/{start_date}/Sentinel2_merged_{start_date}_{end_date}.tif"
        )
        print(f"🎉 全部流程完成: {merged_dir}")
