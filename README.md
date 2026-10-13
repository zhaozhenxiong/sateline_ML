# sateline-ml

可安装、可迁移的 Sentinel-2 水面漂浮物处理 pipeline。下载、波段合并、场景检测和结果管理都放在 `sateline_ml/` 包内，安装后通过 `sateline` 命令运行。原始 `state line.ipynb` 和 `attic/` 仅供参考。

## 安装

Python 3.10 及以上；本项目的完整验证环境使用 Python 3.12。建议每台机器使用独立虚拟环境，避免系统环境中的 GDAL、OpenMP 和 PyTorch 冲突。

```bash
git clone https://github.com/zhaozhenxiong/sateline_ML.git
cd sateline_ML
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[all]"
sateline --version
```

Windows PowerShell 使用 `py -3.12 -m venv .venv` 和 `.venv\Scripts\Activate.ps1`，后续统一用 `python`。开发时将安装命令改成 `python -m pip install -e ".[all,dev]"`。

| 安装方式 | 能力 |
| --- | --- |
| `python -m pip install .` | 配置生成、命令帮助、执行计划、环境诊断；不安装科学计算依赖 |
| `python -m pip install ".[download]"` | GEE 下载与合并；导出本身只用标准库 |
| `python -m pip install ".[detection]"` | 本地 GeoTIFF 检测与塑料 checkpoint 复用 |
| `python -m pip install ".[all]"` | 下载与检测完整流程 |
| `python -m pip install ".[all,stacking]"` | 额外安装 stacking 可选分类器 |

Linux 仅使用 CPU 时，可先执行 `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`，再安装需要的 extras。macOS 的 PyPI PyTorch 支持 CPU/MPS。CUDA 环境按所需 CUDA 版本安装对应的 PyTorch，然后安装本包。

旧安装器仍可使用，并已改为读取 `pyproject.toml` 的依赖声明，不再扫描虚拟环境和历史实验文件：

```bash
python scripts/install_deps.py --dry-run
python scripts/install_deps.py --skip-gee --torch cpu
```

## 本地检测

安装完成后，可在任意工作目录创建任务，不需要回到源码目录：

```bash
sateline init pipeline.json
sateline doctor --config pipeline.json
sateline run --config pipeline.json --dry-run
sateline run --config pipeline.json --stage detect --resume
```

`init` 创建配置和输入目录，不覆盖已有配置。把影像放入 `mydata/tif/`，湖泊边界放入 `mydata/shp/`。可使用 `792_2022-01.tif` 与 `792.shp`；也支持 `lake_792.shp` 或 `mydata/shp/792/*.shp`。ID 使用数字边界匹配，避免将 `792` 错配给 `1792`；多个候选时明确报错。保留 shapefile 的 `.shx`、`.dbf`、`.prj` 等配套文件。

无湖泊 ID 的 GeoTIFF 也会处理。默认没有匹配边界时使用整景谱学水体掩膜；将 `detection.require_shapefile` 设为 `true` 可以强制每景具备边界。提供的 AOI 必须有 CRS，且筛选和重投影后与影像确实相交，避免错误地退回整景分析。

输入支持 `.tif`、`.tiff` 和大写扩展名。同名场景导致的输出目录冲突会在运行前报错。检测以只读方式打开源影像。若影像缺少 CRS，或声明经纬度 CRS 却使用不可能的坐标范围，会明确报错。确认真实 CRS 后可以填写 `detection.raster_crs_override`；pipeline 会创建同网格、同像元的虚拟栅格 `.vrt` 供处理，不修改或复制大型原始影像。覆盖项作用于任务全部影像，混合 CRS 的数据应拆分任务。

## 配置

参考根目录的 `pipeline.example.json`。**全部相对路径以配置文件所在目录为基准**，与运行命令时的当前目录无关。可以把任务目录完整搬到另一台机器继续使用，也可配置绝对路径指向外部数据盘。

| 配置项 | 含义 |
| --- | --- |
| `paths.input_dir` | 待检测 GeoTIFF 目录 |
| `paths.shapefile_dir` | 边界目录；设为 `null` 禁用匹配 |
| `paths.output_dir` | 每景结果及运行记录目录；必须与输入目录分开 |
| `download.enabled` | `--stage all` 是否执行下载 |
| `download.project` | 当前机器有权限访问的 GEE 项目 ID |
| `download.start_date/end_date` | 开始包含、结束不包含，格式 YYYY-MM-DD |
| `download.roi_bounds` | `[min_lon, min_lat, max_lon, max_lat]` |
| `download.step_days` | 下载时间窗口，默认 15 天，必须大于 0 |
| `download.max_cloud` | 场景云量百分比上限，默认 5，范围 0–100 |
| `download.bands` | 下载波段及输出描述顺序 |
| `download.scene_prefix` | 导出文件名前缀；单湖任务可设为 `"792"` 用于边界匹配 |
| `detection.device` | `auto`、`cpu`、`cuda` 或 `mps` |
| `detection.raster_crs_override` | 默认 `null`；仅在确认源影像 CRS 错标时填写正确 CRS，如 `"EPSG:32651"` |
| `detection.band_order` | 缺少描述时的波段顺序；同时支持已有 7/9/11 波段布局 |
| `detection.parameters` | 训练、采样、阈值、AOI 筛选、stacking 等参数 |
| `detection.plastic_library_csv/plastic_model_path` | 可选塑料光谱库和已有 checkpoint 路径 |

模板使用 256 像素训练 tile、128 stride、最多 1024 patch、5 epoch、1 次训练，stacking 和自动阈值标定默认关闭。原入口的较大训练量可通过参数恢复，例如 `epochs=20`、`repeats=3`、`max_patches=3072`。不要将这些默认值视为已经验证的最佳检测阈值。

GeoTIFF 应包含 Sentinel-2 整数 DN，检测内部按 `1e-4` 转换为反射率；已经转换成浮点反射率的数据不能直接使用相同缩放。默认波段顺序为 `B2,B3,B4,B5,B8,B8A,B9,B11,B12`。下载产品会写入波段描述；已有产品若提供描述，必须完整且包含必需波段，缺少描述时使用配置/已知布局。

检测仍按每景数据生成伪标签并训练 UNet，并非加载统一的预训练 UNet。虽然保留了 CUDA/MPS 的降级和分块推理，预处理仍会将整景波段读入内存；超大影像应先裁剪，或拆成较小场景。

## 下载到检测

新机器先认证：

```bash
earthengine authenticate
```

编辑配置，将 `download.enabled` 设为 `true`，填写 `project`、日期、ROI 和可选湖泊 ID `scene_prefix`，然后运行：

```bash
sateline doctor --config pipeline.json
sateline run --config pipeline.json --dry-run
sateline run --config pipeline.json --stage all --resume
```

阶段顺序为 **下载 → 导出 → 检测**。每个时间窗口保留独立场景的 `2merged_tif` 产品；不会将不同时间的场景拼接成一张影像，也不会移动下载源文件。每个场景中的不同分辨率波段会对齐，并正确区分 `B8` 与 `B8A`。

| 参数 | 行为 |
| --- | --- |
| `--stage all` | 执行配置中启用的阶段 |
| `--stage download` | 下载并将保留产品导出到输入目录 |
| `--stage export` | 仅导出已有 `2merged_tif` 或旧 `merge_*/Sentinel2_merged_*.tif` 产品 |
| `--stage detect` | 仅检测本地影像 |
| `--dry-run` | 输出已解析配置与现有场景计划；不认证、不下载、不加载 detector、不写结果 |
| `--resume` | 跳过仍然有效的已完成检测场景；不跳过 GEE 下载请求 |
| `--fail-fast` | 检测遇到第一景失败后停止；默认记录失败并继续后续场景 |
| `--overwrite-exports` | 显式允许替换内容已变化的导出影像；默认遇到差异时报错 |

导出名称确定且处理重名；已有内容相同的导出文件会复用，内容不同时会报错。需要替换旧导出时，先核对数据，再运行 `sateline run --config pipeline.json --stage export --overwrite-exports`；兼容导出脚本也提供 `--overwrite`。

`doctor` 检查版本、依赖是否可导入、组合模块导入、设备可用性、目录和可选资源。导入检查在隔离子进程中执行，因此本机的原生库冲突会转成诊断结果；`--no-import-check` 可做快速静态检查。GEE 诊断仅检查本地凭据文件是否存在，实际下载时才验证凭据和项目权限。

## 输出、续跑与退出码

```text
mydata/analysis_outputs/
  run_manifest.json          # 解析后的配置、成功/跳过/失败数量与逐景记录
  792_2022-01/
    detections.csv
    preview.png
    run.log
    scene.json               # 场景状态、输入/配置指纹、输出记录
    debug/
  models/                    # 每景 UNet 训练产物
```

只有 detector 成功返回且本次产生有效 CSV/PNG 后才标记完成。续跑对输入路径、大小、纳秒修改时间、shapefile 配套文件、配置、塑料资源和包版本做指纹校验，并检查已记录输出文件；变更、失败、中断或丢失输出都会重新执行。指纹使用文件元数据，若人为恢复相同大小和修改时间，使用不带 `--resume` 的运行强制重算。

成功返回 `0`；存在失败场景返回 `1`；配置、安装依赖、阶段等错误返回 `2`；用户中断返回 `130`。失败详情写入场景日志和 JSON，适合脚本或作业调度器检查。

塑料分类默认关闭。配置光谱 CSV 与现有模型文件且两者都存在时，pipeline 使用模型复用模式：检查输入维度、特征顺序和波段顺序，并保持模型文件只读。任一文件缺失则记录跳过；不匹配的 checkpoint 会使该景失败。复用模型应来自相同的 detector 特征和预处理流程；独立训练脚本的预处理可能不同，仅维度和名称匹配不能证明模型已校准。直接调用旧科学 API 时，`plastic_retrain=True` 仍保留原有逐景适配训练，复用时显式传 `plastic_retrain=False`。

## 迁移与打包

源码无需和数据保持同一目录。迁移时带上任务配置、输入影像、边界配套文件、需要的光谱库与 checkpoint；重新创建虚拟环境并安装软件，在目标机重新进行 GEE 认证。不要复制虚拟环境或把原机器的凭据作为项目数据分发。

构建独立软件包：

```bash
python -m pip install build
python scripts/build_distribution.py
```

构建脚本只复制代码、配置模板和文档到本机临时目录，再将 wheel 和源码包写回 `dist/`；这样也能在 macOS 的外置磁盘上使用，避免 AppleDouble 元数据文件影响打包。可用 `--output-dir` 指定其他输出目录。在本机原生文件系统上，也可直接使用 `python -m build`。

将 `dist/sateline_ml-0.1.0-py3-none-any.whl` 拷贝到目标机后：

```bash
python -m pip install "sateline_ml-0.1.0-py3-none-any.whl[all]"
sateline init pipeline.json
sateline doctor --config pipeline.json
```

wheel 包含代码与命令入口，依赖由目标机器安装；原始影像、模型权重和外部光谱 CSV 不打进 wheel。离线机器需另外准备适配其操作系统/Python/架构的依赖 wheel。

## Python API 与兼容入口

```python
from sateline_ml.config import load_config
from sateline_ml.pipeline import run_pipeline

config = load_config("/path/to/task/pipeline.json")
result = run_pipeline(config, stage="detect", resume=True)
```

`from sateline_ml import initialize_ee, floating_debris_detection` 等旧公开 API 保留为惰性导入。还可使用 `python -m sateline_ml` 代替 `sateline`。旧入口保留：

```bash
python run_pipeline.py --yes --resume
python scripts/download_timeseries.py 2022-01-01 2022-03-01 119.8 30.9 120.6 31.6 gee_sentinel3 --project YOUR_PROJECT_ID
python scripts/export_merged_tifs.py gee_sentinel3 export_tifs_from_merged --scene-prefix 792
python scripts/batch_detect.py export_tifs_from_merged path/to/aoi.shp outputs/
```

`run_pipeline.py` 默认配置放在源码根目录，首次使用会生成 `pipeline.json`；现在无需交互询问，按顺序处理所有场景。生产运行请传入明确的任务配置。

## 验证

```bash
python -m pip install -e ".[all,dev]"
python -m pytest -q
python scripts/build_distribution.py
```

测试覆盖配置路径、参数校验、AOI 匹配、失败恢复、输出记录、下载窗口、导出冲突、真实小型 GeoTIFF 波段合并、塑料 checkpoint 与日志资源释放。测试不进行 GEE 认证或真实下载，也不会跑完整生产训练。没有科学依赖时仅运行标准库测试，相关地理/ML 测试会跳过。
