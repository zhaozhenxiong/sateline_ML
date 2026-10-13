"""Control-plane regressions run without geospatial or ML dependencies."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sateline_ml.cli import main
from sateline_ml.config import default_config, load_config, write_default_config
from sateline_ml.pipeline import discover_scenes, pick_shapefile_for_id, run_detection, run_pipeline


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.path = write_default_config(self.root / "pipeline.json")
        self.config = load_config(self.path)
        self.config.paths.input_dir.mkdir(parents=True)
        self.config.paths.shapefile_dir.mkdir(parents=True)

    def inputs(self, *names):
        for name in names:
            (self.config.paths.input_dir / name).write_bytes(b"test raster")

    @staticmethod
    def detector(scene, config):
        (scene.output_dir / "detections.csv").write_text("row,col\n1,2\n")
        (scene.output_dir / "preview.png").write_bytes(b"test preview")
        return 1

    def test_paths_are_relative_to_config_not_cwd(self):
        original = Path.cwd()
        try:
            os.chdir(self.root.parent)
            config = load_config(self.path)
        finally:
            os.chdir(original)
        self.assertEqual(config.paths.input_dir, self.root / "mydata/tif")

    def test_rejects_invalid_configuration_before_native_imports(self):
        for edit in (
            lambda d: d["detection"]["parameters"].update(stride=257),
            lambda d: d["detection"]["parameters"].update(unknown=True),
            lambda d: d["download"].update(step_days=0),
            lambda d: d["detection"].update(band_order=["B2", "B3"]),
            lambda d: d["paths"].update(output_dir="mydata/tif/results"),
            lambda d: d.update(schema_version=True),
            lambda d: d["detection"]["parameters"].update(absorbance_thresh="invalid"),
            lambda d: d["detection"]["parameters"].update(auto_threshold_candidates=[True]),
            lambda d: d["detection"]["parameters"].update(min_water_ratio=0.8, max_water_ratio=0.2),
            lambda d: d["detection"]["parameters"].update(filter_indices=["0"]),
        ):
            with self.subTest(edit=edit):
                data = default_config()
                edit(data)
                self.path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    load_config(self.path)

    def test_exact_shapefile_wins_and_ids_have_numeric_boundaries(self):
        directory = self.config.paths.shapefile_dir
        (directory / "1792.shp").touch()
        self.assertIsNone(pick_shapefile_for_id(directory, "792"))
        (directory / "lake_792.shp").touch()
        exact = directory / "792.shp"
        exact.touch()
        self.assertEqual(pick_shapefile_for_id(directory, "792"), exact)
        exact.unlink()
        (directory / "other_792.shp").touch()
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            pick_shapefile_for_id(directory, "792")

    def test_extension_handling_no_lake_id_and_duplicate_stem(self):
        self.inputs("scene.TIF", "792_image.tiff")
        self.assertEqual(len(discover_scenes(self.config)), 2)
        self.inputs("scene.tiff")
        with self.assertRaisesRegex(ValueError, "collision"):
            discover_scenes(self.config)

    def test_require_aoi_fails_before_outputs_written(self):
        self.inputs("scene.tif")
        self.config.detection.require_shapefile = True
        with self.assertRaisesRegex(ValueError, "No matching shapefile"):
            run_detection(self.config)
        self.assertFalse(self.config.paths.output_dir.exists())

    def test_resume_skips_only_unchanged_completed_scenes(self):
        self.inputs("792_a.tif")
        shp = self.config.paths.shapefile_dir / "792.shp"
        shp.write_bytes(b"shp")
        dbf = shp.with_suffix(".dbf")
        dbf.write_bytes(b"dbf")
        with patch("sateline_ml.pipeline.detect_scene", side_effect=self.detector) as detector:
            self.assertEqual(run_detection(self.config)["completed"], 1)
            self.assertEqual(run_detection(self.config, resume=True)["skipped"], 1)
            dbf.write_bytes(b"changed attributes")
            self.assertEqual(run_detection(self.config, resume=True)["completed"], 1)
            self.config.detection.parameters["epochs"] = 6
            self.assertEqual(run_detection(self.config, resume=True)["completed"], 1)
            scene = discover_scenes(self.config)[0]
            (scene.output_dir / "preview.png").unlink()
            self.assertEqual(run_detection(self.config, resume=True)["completed"], 1)
            scene.input_tif.write_bytes(b"changed raster")
            self.assertEqual(run_detection(self.config, resume=True)["completed"], 1)
            self.assertEqual(detector.call_count, 5)

    def test_failure_continues_records_error_and_resume_retries(self):
        self.inputs("a.tif", "b.tif")
        def fail_first(scene, config):
            if scene.input_tif.stem == "a":
                raise RuntimeError("bad raster")
            return self.detector(scene, config)
        with patch("sateline_ml.pipeline.detect_scene", side_effect=fail_first):
            result = run_detection(self.config)
        self.assertEqual((result["failed"], result["completed"]), (1, 1))
        record = json.loads((self.config.paths.output_dir / "a/scene.json").read_text())
        self.assertEqual(record["status"], "failed")
        self.assertIn("bad raster", record["error"])
        self.assertIn("bad raster", (self.config.paths.output_dir / "a/run.log").read_text())
        with patch("sateline_ml.pipeline.detect_scene", side_effect=self.detector):
            result = run_detection(self.config, resume=True)
        self.assertEqual((result["failed"], result["completed"], result["skipped"]), (0, 1, 1))

    def test_fail_fast_and_stale_outputs_cannot_mark_success(self):
        self.inputs("a.tif", "b.tif")
        with patch("sateline_ml.pipeline.detect_scene", side_effect=self.detector):
            run_detection(self.config)
        with patch("sateline_ml.pipeline.detect_scene", return_value=0) as detector:
            result = run_detection(self.config, fail_fast=True)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(detector.call_count, 1)

    def test_dry_run_no_native_imports_or_outputs_and_exit_codes(self):
        self.inputs("a.tif")
        with redirect_stdout(io.StringIO()), patch("sateline_ml.pipeline.detect_scene") as detector:
            self.assertEqual(main(["run", "--config", str(self.path), "--dry-run"]), 0)
        detector.assert_not_called()
        self.assertFalse(self.config.paths.output_dir.exists())
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), patch("sateline_ml.pipeline.detect_scene", side_effect=RuntimeError("failed")):
            self.assertEqual(main(["run", "--config", str(self.path)]), 1)

    def test_download_export_detect_stage_order(self):
        self.config.download.enabled = True
        self.config.download.project = "test-project"
        self.inputs("a.tif")
        events = []
        def download(**kwargs):
            events.append("download")
            return [self.root / "merged.tif"]
        def export(*args, **kwargs):
            events.append("export")
            self.assertTrue(kwargs["overwrite"])
            return [self.config.paths.input_dir / "a.tif"]
        def detect(*args, **kwargs):
            events.append("detect")
            return {"failed": 0}
        with patch("sateline_ml.workflows.download_timeseries", side_effect=download), patch("sateline_ml.workflows.export_merged_tifs", side_effect=export), patch("sateline_ml.pipeline.run_detection", side_effect=detect):
            run_pipeline(self.config, overwrite_exports=True)
        self.assertEqual(events, ["download", "export", "detect"])

    def test_config_init_does_not_overwrite(self):
        original = self.path.read_text()
        with self.assertRaises(FileExistsError):
            write_default_config(self.path)
        self.assertEqual(self.path.read_text(), original)

    def test_json_overflow_is_rejected_before_scientific_computation(self):
        self.path.write_text('{"detection":{"parameters":{"ndvi_thresh":1e999}}}')
        with self.assertRaisesRegex(ValueError, "finite"):
            load_config(self.path)


if __name__ == "__main__":
    unittest.main()
