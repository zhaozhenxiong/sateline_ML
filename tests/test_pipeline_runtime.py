"""Small real GeoTIFF tests for the installable detection pipeline."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from sateline_ml.config import DEFAULT_BAND_ORDER, default_config, load_config
from sateline_ml.pipeline import _resolve_bands, run_detection

HAS_GEO = all(importlib.util.find_spec(name) is not None for name in ("numpy", "rasterio"))
HAS_DETECTION = HAS_GEO and all(importlib.util.find_spec(name) is not None for name in
                              ("torch", "geopandas", "pandas", "cv2", "matplotlib", "tqdm"))


@unittest.skipUnless(HAS_GEO, "Install geospatial dependencies for GeoTIFF validation")
class BandLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def raster(self, count=9, descriptions=None, crs="EPSG:32651"):
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        path = self.root / "scene.tif"
        with rasterio.open(path, "w", driver="GTiff", count=count, height=32, width=32, dtype="uint16",
                           crs=crs, transform=from_origin(400000, 3400000, 10, 10)) as dst:
            dst.write(np.full((count, 32, 32), 1000, dtype="uint16"))
            if descriptions is not None:
                dst.descriptions = descriptions
        return path

    def test_existing_descriptions_and_known_fallbacks_are_read_only(self):
        for count, order in ((9, DEFAULT_BAND_ORDER), (7, ("B2", "B3", "B4", "B8", "B8A", "B11", "B12")),
                             (11, ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"))):
            with self.subTest(count=count):
                path = self.raster(count)
                before = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(_resolve_bands(path, DEFAULT_BAND_ORDER), order)
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
        path = self.raster(descriptions=DEFAULT_BAND_ORDER)
        self.assertEqual(_resolve_bands(path, DEFAULT_BAND_ORDER), DEFAULT_BAND_ORDER)

    def test_partial_unknown_metadata_missing_required_band_and_no_crs_fail(self):
        for count, descriptions, crs in (
            (9, ("B2",) + (None,) * 8, "EPSG:32651"),
            (9, DEFAULT_BAND_ORDER[:-1] + ("UNKNOWN",), "EPSG:32651"),
            (8, None, "EPSG:32651"),
            (9, None, None),
        ):
            with self.subTest(count=count, descriptions=descriptions, crs=crs):
                path = self.raster(count, descriptions, crs)
                with self.assertRaises(ValueError):
                    _resolve_bands(path, DEFAULT_BAND_ORDER)


@unittest.skipUnless(HAS_DETECTION, "Install sateline-ml[detection] for real pipeline smoke test")
class DetectionPipelineSmokeTests(BandLayoutTests):
    def test_zero_water_scene_produces_real_csv_png_and_resumes(self):
        # The no-water path verifies orchestration/plotting without expensive UNet training.
        image = self.raster(descriptions=DEFAULT_BAND_ORDER)
        directory = self.root / "input"
        directory.mkdir()
        image = image.rename(directory / image.name)
        before = hashlib.sha256(image.read_bytes()).hexdigest()
        data = default_config()
        data["paths"] = {"input_dir": "input", "shapefile_dir": None, "output_dir": "output"}
        data["detection"]["device"] = "cpu"
        data["detection"]["parameters"].update(min_water_ratio=0, max_water_ratio=0, tile_size=32, stride=16, max_patches=1, epochs=1)
        config_path = self.root / "pipeline.json"
        config_path.write_text(json.dumps(data))
        config = load_config(config_path)
        result = run_detection(config)
        self.assertEqual((result["completed"], result["failed"]), (1, 0), result)
        output = config.paths.output_dir / "scene"
        self.assertTrue((output / "preview.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIn("probability", (output / "detections.csv").read_text(encoding="utf-8-sig"))
        self.assertTrue((output / "run.log").is_file())
        self.assertEqual(run_detection(config, resume=True)["skipped"], 1)
        self.assertEqual(hashlib.sha256(image.read_bytes()).hexdigest(), before)

    def test_explicit_crs_override_recovers_mislabeled_scene_without_source_changes(self):
        import rasterio
        image = self.raster(descriptions=DEFAULT_BAND_ORDER, crs="EPSG:4326")
        directory = self.root / "input"
        directory.mkdir()
        image = image.rename(directory / image.name)
        before = hashlib.sha256(image.read_bytes()).hexdigest()
        data = default_config()
        data["paths"] = {"input_dir": "input", "shapefile_dir": None, "output_dir": "output"}
        data["detection"]["device"] = "cpu"
        data["detection"]["parameters"].update(min_water_ratio=0, max_water_ratio=0, tile_size=32, stride=16, max_patches=1, epochs=1)
        config_path = self.root / "pipeline.json"
        config_path.write_text(json.dumps(data))
        self.assertEqual(run_detection(load_config(config_path))["failed"], 1)
        data["detection"]["raster_crs_override"] = "EPSG:32651"
        config_path.write_text(json.dumps(data))
        result = run_detection(load_config(config_path), resume=True)
        self.assertEqual((result["completed"], result["failed"]), (1, 0), result)
        vrt = self.root / "output/scene/scene.vrt"
        with rasterio.open(vrt) as src:
            self.assertEqual(src.crs.to_epsg(), 32651)
            self.assertEqual(src.descriptions, DEFAULT_BAND_ORDER)
        self.assertEqual(hashlib.sha256(image.read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()
