"""Resource lifecycle and checkpoint inference tests for the detection extra."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


HAS_DETECTION = all(importlib.util.find_spec(name) is not None for name in (
    "torch", "numpy", "pandas", "rasterio", "geopandas", "cv2", "matplotlib", "tqdm",
))


@unittest.skipUnless(HAS_DETECTION, "Install sateline-ml[detection] to test detector resources")
class DetectionResourcesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sateline_ml import detection
        cls.detection = detection

    def test_exception_closes_file_handler_and_does_not_mix_scene_logs(self):
        detection = self.detection
        detection._ensure_stream_logger()
        baseline = list(detection.LOGGER.handlers)
        created = []
        original_configure = detection._configure_logger

        def configure(path):
            handler = original_configure(path)
            created.append(handler)
            return handler

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(detection, "_configure_logger", side_effect=configure):
                for scene in ("first", "second"):
                    with self.assertRaises(Exception):
                        detection.floating_debris_detection(
                            root / f"missing-{scene}.tif",
                            output_csv=root / f"{scene}.csv",
                            preview_png=root / f"{scene}.png",
                            log_file=root / f"{scene}.log",
                        )
                    self.assertEqual(detection.LOGGER.handlers, baseline)
                    self.assertIsNone(created[-1].stream)
            self.assertNotIn("second.log", (root / "first.log").read_text(encoding="utf-8"))
            self.assertIn("second.log", (root / "second.log").read_text(encoding="utf-8"))

    def test_logging_wrapper_preserves_public_signature(self):
        signature = inspect.signature(self.detection.floating_debris_detection)
        self.assertEqual(signature.parameters["output_csv"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(signature.parameters["plastic_retrain"].default, True)

    def test_checkpoint_feature_mismatch_is_an_error(self):
        from sateline_ml.train_model import PlasticModelMetadata
        detection = self.detection
        model = object()
        metadata = PlasticModelMetadata(3, ("a", "b", "c"), ("B2", "B3", "B4"))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            checkpoint.touch()
            with patch.object(detection, "load_plastic_model", return_value=(model, metadata)):
                args = dict(input_dim=3, feature_names=("a", "b", "c"), band_order=("B2", "B3", "B4"), device="cpu")
                self.assertIs(detection._load_compatible_plastic_model(checkpoint, **args), model)
                for key, incompatible in (("input_dim", 4), ("feature_names", ("c", "b", "a")), ("band_order", ("B4", "B3", "B2"))):
                    with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                        detection._load_compatible_plastic_model(checkpoint, **{**args, key: incompatible})

    def test_existing_checkpoint_inference_never_trains_or_overwrites_it(self):
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        from sateline_ml.train_model import PlasticMLP, PlasticModelMetadata, save_plastic_model

        detection = self.detection
        bands = ("B2", "B3", "B4", "B5", "B8", "B8A", "B9", "B11", "B12")
        names = ("feature_a", "feature_b", "feature_c")
        robust = {"ndwi_hi": 0.2, "ndwi_lo": 0.1, "fdi_strong": -1.0,
                  "fdi_weak": -1.0, "ndti_strong": 1.0, "ndti_weak": 1.0,
                  "ndvi_lo": -1.0, "ndvi_hi": 1.0}

        def features(cube, order, library, rows, cols, **kwargs):
            return np.ones((len(rows), 3), dtype=np.float32), names, {}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "scene.tif"
            values = np.array([500, 2000, 600, 500, 500, 500, 500, 100, 100], dtype=np.uint16)
            with rasterio.open(image, "w", driver="GTiff", count=len(bands), height=32, width=32,
                               dtype="uint16", crs="EPSG:32651", transform=from_origin(400000, 3400000, 10, 10)) as dst:
                dst.write(np.broadcast_to(values[:, None, None], (len(bands), 32, 32)).copy())
                dst.descriptions = bands
            library = root / "library.csv"
            library.write_text("feature construction is mocked in this lifecycle test", encoding="utf-8")
            checkpoint = root / "existing.pth"
            save_plastic_model(checkpoint, PlasticMLP(3, hidden=(8, 4)), PlasticModelMetadata(3, names, bands))
            before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            with patch.object(detection, "_robust_thresholds_water_proxy", return_value=robust), \
                 patch.object(detection, "_hysteresis_water", side_effect=lambda ndwi, hi, lo: np.ones_like(ndwi, dtype=bool)), \
                 patch.object(detection, "sample_training_patches", return_value=([], [], [])), \
                 patch.object(detection, "build_plastic_feature_matrix_sparse", side_effect=features), \
                 patch.object(detection, "train_plastic_model", side_effect=AssertionError("must reuse checkpoint")) as train, \
                 patch.object(detection, "save_plastic_model", side_effect=AssertionError("checkpoint is read only")) as save, \
                 patch.object(detection, "visualize_results"), \
                 patch.object(detection, "_save_plastic_feature_correlations"), \
                 patch.object(detection, "_save_plastic_spectral_profiles"):
                result = detection.floating_debris_detection(
                    image, output_csv=root / "detections.csv", preview_png=root / "preview.png",
                    plastic_library_csv=library, plastic_model_path=checkpoint, plastic_retrain=False,
                    device="cpu", log_file=root / "run.log", ndvi_thresh=1.0,
                )
                self.assertGreater(len(result), 0)
                self.assertTrue((root / "detections.csv").is_file())
                train.assert_not_called()
                save.assert_not_called()
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()
