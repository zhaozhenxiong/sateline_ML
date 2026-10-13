# sateline_ML

sateline_ML reorganises the original Jupyter notebook into a reusable Python codebase for downloading Sentinel-2 imagery from Google Earth Engine and detecting floating debris on water surfaces. The pipeline covers GEE download, multi-band merge, optional plastic spectral classification, and per-scene CSV/PNG outputs.

## Tech stack

- **Python** 3.x
- **Google Earth Engine** Python API (`earthengine-api`)
- **Raster I/O / geoprocessing**: `rasterio`, `geopandas`, `shapely`
- **ML / deep learning**: `torch` (UNet), `numpy`, `pandas`
- **Computer vision / plotting**: `opencv-python`, `matplotlib`, `scikit-image`
- **Utilities**: `requests`, `tqdm`
- **Optional stacking / plastic modules**: `sklearn` (used by stacking/plastic analytics when available)

## Common commands

Install dependencies:

```bash
python scripts/install_deps.py
python scripts/install_deps.py --dry-run
python scripts/install_deps.py --skip-geo --torch cpu
```

Authenticate Google Earth Engine:

```bash
earthengine authenticate
```

Download Sentinel-2 scenes over time windows:

```bash
python scripts/download_timeseries.py \
    2022-01-01 2022-03-01 \
    119.8 30.9 120.6 31.6 \
    gee_sentinel3 --project <ee-project-id>  # replace <ee-project-id> with your GEE project ID
```

Export merged GeoTIFFs to a flat directory:

```bash
python scripts/export_merged_tifs.py gee_sentinel3 export_tifs_from_merged
```

Run quick batch detection:

```bash
python scripts/batch_detect.py export_tifs_from_merged path/to/aoi.shp outputs/
```

Run the end-to-end pipeline:

```bash
python run_pipeline.py
```

## Important directories and files

| Path | Description |
|---|---|
| `README.md` | Project overview and quick-start examples |
| `run_pipeline.py` | End-to-end wrapper: matches TIFs to SHPs, runs detection per scene |
| `scripts/download_timeseries.py` | Batch Sentinel-2 downloader over 15-day windows (configurable) |
| `scripts/batch_detect.py` | Quick or advanced batch detection over a folder of GeoTIFFs |
| `scripts/export_merged_tifs.py` | Copy `2merged_tif/*.tif` into a flat export directory |
| `scripts/install_deps.py` | Auto-detect and install required Python packages |
| `scripts/filter_large_files.py` | Move the largest tiles above a size threshold |
| `sateline_ml/__init__.py` | Package public API exports |
| `sateline_ml/gee.py` | GEE initialisation, download, tiling, and band-merge helpers |
| `sateline_ml/detection.py` | Floating-debris detection, masking, and batch utilities |
| `state line.ipynb` | Original reference notebook |
| `models` | Optional saved plastic classifier checkpoint |
| `spectral_library/plastic_spectra.csv` | Plastic spectral library for plastic-branch classification |
| `mydata/tif/` | Input GeoTIFF directory for `run_pipeline.py` |
| `mydata/shp/` | Input shapefile directory for `run_pipeline.py` |
| `mydata/analysis_outputs/` | Per-scene detection outputs from `run_pipeline.py` |

## Working rules

1. **Dependency install** — Run `python scripts/install_deps.py` first. Use `--dry-run` to preview missing packages, `--skip-geo` to skip `geopandas`/`rasterio`, and `--torch cpu` for CPU-only PyTorch.
2. **GEE authentication** — Before any download, ensure `earthengine authenticate` has been run and a valid project ID is passed to `scripts/download_timeseries.py` via `--project`.
3. **Multiprocessing spawn** — On Windows and macOS the codebase forces `multiprocessing.set_start_method("spawn", force=True)` and `torch.multiprocessing.set_start_method("spawn", force=True)`. Do not change this to `fork`.
4. **Matplotlib backend** — `sateline_ml/detection.py` forces `matplotlib.use("Agg")`. Do not switch to an interactive backend in headless or multiprocessing contexts.
5. **Band order** — The default expected band order in `run_pipeline.py` is `("B2", "B3", "B4", "B5", "B8", "B8A", "B9", "B11", "B12")`. Detection falls back to `full_band_order`, `compact_band_order`, and `minimal_band_order` if band descriptions are missing, but products should have band descriptions stamped.
6. **Code organization** — Reusable logic lives under `sateline_ml/`. One-off scripts live under `scripts/`. Keep the original `state line.ipynb` for reference only; implement production changes in the package.
7. **Large files / .gitignore** — Downloads create `1source_tif/`, `2merged_tif/`, `gee_sentinel3/`, `gee_nature3/`, `mydata/`, `target/`, `target_merge/`, `target_n/`, `lake_01-05/`, `HydroLAKES_polys_v10.gdb/`, `HydroLAKES_polys_v10_shp/`, and model/imagery outputs. Add these directories and large `.tif`/`.csv`/`.pt`/`.pth`/`.gdb` artifacts to `.gitignore`; never commit raw imagery or model weights.
8. **Plastic detection prerequisites** — Plastic classification requires `spectral_library/plastic_spectra.csv` to exist. Optional trained classifier checkpoint is `models`. If either is missing the plastic branch is skipped.
9. **Memory / tile settings** — Inference auto-downgrades from CUDA/MPS to CPU and tries tile sizes `1024, 896, 768, 640, 512, 384, 320` on CUDA and `768, 640, 512, 448, 384, 320` on MPS. Keep `tile_size`, `stride`, `max_patches`, and `stacking_max_pixels` conservative on low-memory machines.
10. **Logging / debug** — `floating_debris_detection` writes per-scene `run.log` and debug PNGs under `mydata/analysis_outputs/<scene_stem>/debug/`. Set `debug_dir` and `log_file` arguments explicitly when calling the API directly.
11. **Input naming conventions** — `run_pipeline.py` extracts a lake ID from TIF filenames using the regex `(?P<id>\d{1,})(?=_)`, matching names like `792_2022-01.tif`. Shapefiles in `mydata/shp/` should be named `{id}.shp`, contain the ID in the filename, or live in a subfolder `mydata/shp/{id}/`.
