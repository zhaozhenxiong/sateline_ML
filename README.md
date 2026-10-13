# sateline_ml Python Project

This package reorganises the original Jupyter notebook into a reusable Python
codebase for downloading Sentinel-2 imagery and detecting floating debris on
water surfaces.

Use Python 3.10 or newer.

## Package layout

```
sateline_ml/
  __init__.py
  gee.py                # Google Earth Engine download helpers
  detection.py          # Feature computation and floating debris detection
scripts/
  download_timeseries.py  # Batch Sentinel-2 downloader
  filter_large_files.py   # Move the largest tiles above a size threshold
  export_merged_tifs.py   # Flatten merged products into a single directory
  batch_detect.py         # Run quick screening for every GeoTIFF in a folder
state line.ipynb           # Original notebook (kept for reference)
attic/                    # Unmaintained historical experiments
```

## Quick start

1. **Clone the repository and install dependencies**
   ```bash
   git clone https://github.com/zhaozhenxiong/sateline_ML.git
   cd sateline_ML
   python3 -m venv ../sateline_ml-venv
   source ../sateline_ml-venv/bin/activate
   python scripts/install_deps.py
   ```
   On Windows, create the environment with `python -m venv ..\sateline_ml-venv`
   and activate it with `..\sateline_ml-venv\Scripts\Activate.ps1` in PowerShell.
   Keep the environment outside the checkout because the installer scans all
   Python files in the project directory.
   The script scans the repository for imports and installs missing packages.
   Use `--dry-run` to preview what would be installed.

2. **Authenticate Google Earth Engine**
   ```bash
   earthengine authenticate
   ```

3. **Download Sentinel-2 scenes**
   ```bash
   python scripts/download_timeseries.py \
       2022-01-01 2022-03-01 \
       119.8 30.9 120.6 31.6 \
       gee_sentinel3 --project <ee-project-id>
   ```
   The script loops over 15-day windows (configurable via `--step-days`),
   downloads imagery, keeps the best tile per 5-minute bucket, and merges the
   tiles into multi-band products.

4. **Export merged GeoTIFFs**
   ```bash
   python scripts/export_merged_tifs.py gee_sentinel3 export_tifs_from_merged
   ```

5. **Detect floating debris**
   ```bash
   python scripts/batch_detect.py export_tifs_from_merged path/to/aoi.shp outputs/
   ```
   The detection utilities honour optional shapefile filters (`--filter-field`,
   `--filter-values`, etc.) and save both CSV and PNG previews per scene.

   To also run the optional plastic-spectral branch, ensure the saved plastic
   classifier (`models`) and the spectral library
   (`spectral_library/plastic_spectra.csv`) are present.

### End-to-end pipeline

Place input GeoTIFFs in `mydata/tif/` and lake boundary shapefiles in
`mydata/shp/`. Use filenames such as `792_2022-01.tif` and `792.shp` so the
pipeline can match each scene to its lake. Keep the accompanying `.shx`, `.dbf`,
and `.prj` shapefile files together.

```bash
python run_pipeline.py
```

Results are written under `mydata/analysis_outputs/<scene_stem>/`, including
per-scene CSV/PNG outputs, `run.log`, and debug images.

### Data and local files

The repository includes source code, reference notebooks, and the small CSV
reference libraries. Raw imagery, local lake datasets, generated results,
model weights (including `models`), virtual environments, and credentials are
excluded by `.gitignore`. Obtain input datasets and model checkpoints separately
before running the corresponding workflows.

### API usage

The main entry points are available directly from the package:

```python
from sateline_ml import (
    initialize_ee,
    process_download,
    merge_all_bands,
    floating_debris_detection,
    create_mask_tif_from_shp,
)
```

See `sateline_ml/gee.py` and `sateline_ml/detection.py` for detailed
documentation and function signatures.
