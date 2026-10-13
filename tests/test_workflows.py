"""Workflow regressions; no authentication, network, or model loading required."""

from __future__ import annotations

import importlib.util
import hashlib
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from sateline_ml.workflows import (
    DEFAULT_BANDS,
    download_timeseries,
    export_merged_tifs,
    iter_windows,
    validate_download_options,
)


class DownloadWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.options = dict(
            start_date="2022-01-01", end_date="2022-02-01",
            roi_bounds=(119.8, 30.9, 120.6, 31.6), project="ee-project",
        )

    def test_windows_are_adjacent_and_end_exclusive(self):
        self.assertEqual(
            iter_windows("2022-01-01", "2022-02-01", 15),
            [("2022-01-01", "2022-01-16"), ("2022-01-16", "2022-01-31"), ("2022-01-31", "2022-02-01")],
        )

    def test_invalid_steps_fail_before_loop(self):
        for step in (0, -1, True, 1.5, "15"):
            with self.subTest(step=step), self.assertRaises(ValueError):
                iter_windows("2022-01-01", "2022-02-01", step)

    def test_date_validation(self):
        for start, end in (
            ("2022-01-01", "2022-01-01"), ("2022-02-01", "2022-01-01"),
            ("2022-02-30", "2022-03-01"), ("20220101", "2022-03-01"),
        ):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                iter_windows(start, end)

    def test_validation_has_no_heavy_imports_or_filesystem_effects(self):
        fake = SimpleNamespace(initialize_ee=MagicMock(), process_download=MagicMock())
        for change in (
            {"project": ""}, {"project": None}, {"project": "project with spaces"},
            {"roi_bounds": (181, 0, 182, 1)}, {"roi_bounds": (1, 2, 1, 3)},
            {"roi_bounds": (1, float("nan"), 2, 3)}, {"roi_bounds": "1,2,3,4"},
            {"bands": []}, {"bands": ["B2", "B2"]}, {"bands": ["B10"]},
            {"bands": "B2"}, {"max_cloud": 101}, {"max_cloud": float("nan")},
            {"step_days": 0},
        ):
            with self.subTest(change=change), patch.dict(sys.modules, {"sateline_ml.gee": fake}):
                with self.assertRaises(ValueError):
                    download_timeseries(output_root=self.root / "new", **(self.options | change))
        fake.initialize_ee.assert_not_called()
        fake.process_download.assert_not_called()
        self.assertFalse((self.root / "new").exists())

    def test_valid_options_preserve_requested_band_order(self):
        windows, bounds, bands = validate_download_options(**self.options, bands=["B8", "B8A", "B2"])
        self.assertEqual(bands, ("B8", "B8A", "B2"))
        self.assertEqual(bounds, self.options["roi_bounds"])
        self.assertEqual(len(windows), 3)

    def test_download_returns_retained_scenes_without_cross_scene_merge(self):
        files = [self.root / "scene-a.tif", self.root / "scene-b.tif", self.root / "scene-c.tif"]
        fake = SimpleNamespace(
            initialize_ee=MagicMock(),
            process_download=MagicMock(side_effect=[
                {"later": SimpleNamespace(path=files[1]), "earlier": SimpleNamespace(path=files[0])},
                {}, {"last": SimpleNamespace(path=files[2])},
            ]),
            find_max_size=MagicMock(), merge_all_bands=MagicMock(),
        )
        with patch.dict(sys.modules, {"sateline_ml.gee": fake}):
            result = download_timeseries(output_root=self.root / "output", **self.options)
        self.assertEqual(result, files)
        fake.initialize_ee.assert_called_once_with("ee-project")
        self.assertEqual(fake.process_download.call_count, 3)
        self.assertEqual(fake.process_download.call_args_list[0].kwargs["bands_to_download"], DEFAULT_BANDS)
        self.assertEqual(fake.process_download.call_args_list[0].kwargs["project_root"], self.root / "output" / "2022-01-01")
        fake.find_max_size.assert_not_called()
        fake.merge_all_bands.assert_not_called()

    def test_help_and_workflow_import_work_without_geo_or_torch(self):
        code = (
            "import sys; import sateline_ml.workflows; "
            "assert all(x not in sys.modules for x in ('ee', 'torch', 'rasterio', 'numpy'))"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        for script in ("download_timeseries.py", "export_merged_tifs.py"):
            result = subprocess.run([sys.executable, str(scripts / script), "--help"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


class ExportWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "downloads"
        self.root.mkdir()
        self.dest = Path(self.temp.name) / "flat"

    def write(self, relative, contents=b"image"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        return path

    def test_current_and_legacy_final_products_only(self):
        self.write("2022-01-01/2merged_tif/scene-a.tif", b"a")
        self.write("2022-01-01/2merged_tif/scene-b.TIFF", b"b")
        self.write("merge_2022-02-01/Sentinel2_merged_2022-02-01_2022-02-16.tif", b"legacy")
        self.write("2022-01-01/1source_tif/scene/B2_tile0.tif", b"raw")
        self.write("merge_2022-02-01/merged_bands/B2.tif", b"intermediate")
        result = export_merged_tifs(self.root, self.dest, scene_prefix="792")
        self.assertEqual(len(result), 3)
        self.assertTrue(all(path.name.startswith("792_") for path in result))
        self.assertEqual({path.read_bytes() for path in result}, {b"a", b"b", b"legacy"})
        self.assertTrue((self.root / "2022-01-01/1source_tif/scene/B2_tile0.tif").exists())

    def test_flat_name_collisions_are_deterministic_and_preserve_contents(self):
        self.write("a_b/2merged_tif/c.tif", b"first")
        self.write("a/2merged_tif/b_c.tif", b"second")
        first = export_merged_tifs(self.root, self.dest)
        second = export_merged_tifs(self.root, self.dest)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 2)
        self.assertEqual({path.read_bytes() for path in first}, {b"first", b"second"})
        self.assertEqual(len(list(self.dest.iterdir())), 2)

    def test_case_insensitive_flat_collisions_are_safe(self):
        # Distinct source paths on both case-sensitive and insensitive volumes.
        self.write("A_x/2merged_tif/s.tif", b"first")
        self.write("a/2merged_tif/X_s.TIF", b"second")
        result = export_merged_tifs(self.root, self.dest)
        self.assertEqual(len({path.name.casefold() for path in result}), 2)

    def test_destination_is_not_rediscovered_on_rerun(self):
        self.write("2022-01-01/2merged_tif/scene.tif")
        destination = self.root / "exports" / "2merged_tif"
        first = export_merged_tifs(self.root, destination)
        second = export_merged_tifs(self.root, destination)
        self.assertEqual(first, second)
        self.assertEqual(len(list(destination.iterdir())), 1)

    def test_export_to_parent_does_not_exclude_all_inputs(self):
        self.write("2022-01-01/2merged_tif/scene.tif")
        result = export_merged_tifs(self.root, self.root.parent)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].parent, self.root.parent.resolve())

    def test_changed_input_fails_or_is_explicitly_overwritten(self):
        source = self.write("2022-01-01/2merged_tif/scene.tif", b"aaaa")
        [destination] = export_merged_tifs(self.root, self.dest)
        old_stat = source.stat()
        source.write_bytes(b"bbbb")
        os.utime(source, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
        with self.assertRaises(FileExistsError):
            export_merged_tifs(self.root, self.dest)
        self.assertEqual(destination.read_bytes(), b"aaaa")
        self.assertEqual(export_merged_tifs(self.root, self.dest, overwrite=True), [destination])
        self.assertEqual(destination.read_bytes(), b"bbbb")

    def test_bad_paths_and_prefix_fail(self):
        with self.assertRaises(NotADirectoryError):
            export_merged_tifs(self.root / "missing", self.dest)
        with self.assertRaises(ValueError):
            export_merged_tifs(self.root, self.root)
        with self.assertRaises(ValueError):
            export_merged_tifs(self.root, self.dest, scene_prefix="../792")


class EarthEngineMergeRegressionTests(unittest.TestCase):
    """Exercise scene handling and file selection with dependency stubs."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        stubs = {name: MagicMock() for name in (
            "ee", "numpy", "pandas", "rasterio", "rasterio.merge", "rasterio.warp", "requests",
        )}
        path = Path(__file__).resolve().parents[1] / "sateline_ml" / "gee.py"
        spec = importlib.util.spec_from_file_location("sateline_ml.gee", path)
        self.gee = importlib.util.module_from_spec(spec)
        stubs["sateline_ml.gee"] = self.gee
        self.stub_patch = patch.dict(sys.modules, stubs)
        self.stub_patch.start()
        self.addCleanup(self.stub_patch.stop)
        spec.loader.exec_module(self.gee)

    def test_b8_never_selects_b8a_or_reprojected_intermediates(self):
        for filename in ("B8_tile0.tif", "B8A_tile0.tif", "B8_tile0_reproj.tif"):
            (self.root / filename).touch()
        self.assertEqual([path.name for path in self.gee._band_paths(self.root, "B8")], ["B8_tile0.tif"])
        (self.root / "B8_tile0.tif").unlink()
        self.assertFalse(self.gee.all_bands_downloaded(self.root, ["B8", "B8A"]))

    def test_merge_checks_band_completeness_before_reading_rasters(self):
        (self.root / "B8A_tile0.tif").touch()
        with self.assertRaises(FileNotFoundError):
            self.gee.merge_bands(self.root, self.root / "output.tif", ["B8", "B8A"])
        self.gee.rasterio.open.assert_not_called()
        with self.assertRaises(FileNotFoundError):
            self.gee.merge_all_bands(self.root, ["B8", "B8A"], self.root / "intermediate", self.root / "final.tif")
        self.gee.rasterio.open.assert_not_called()

    def test_merge_bands_preserves_exact_paths_and_descriptions(self):
        for filename in ("B8_tile0.tif", "B8A_tile0.tif"):
            (self.root / filename).touch()
        seen = []

        def read(paths):
            seen.append([path.name for path in paths])
            return SimpleNamespace(shape=(2, 2)), "transform", "EPSG:4326"

        with patch.object(self.gee, "_read_mosaic", side_effect=read):
            self.gee.merge_bands(self.root, self.root / "output.tif", ["B8", "B8A"])
        self.assertEqual(seen, [["B8_tile0.tif"], ["B8_tile0.tif"], ["B8A_tile0.tif"]])
        writer = self.gee.rasterio.open.return_value.__enter__.return_value
        self.assertEqual(writer.set_band_description.call_args_list[0].args, (1, "B8"))
        self.assertEqual(writer.set_band_description.call_args_list[1].args, (2, "B8A"))

    def test_same_timestamp_scene_candidates_use_separate_files(self):
        timestamp = "2022-01-01_10-11-12"
        self.gee.ee.Date.return_value.format.return_value.getInfo.return_value = timestamp
        images = []
        # The first two IDs sanitize to the same text; the digest distinguishes
        # them. The fourth record repeats the retained third scene.
        for scene_id in ("tile/A", "tile_A", "tile/C", "tile/C"):
            image = MagicMock()
            image.get.side_effect = lambda key, sid=scene_id: SimpleNamespace(getInfo=lambda: sid)
            images.append(image)

        def download(image, band, output, roi, **kwargs):
            output.write_bytes(b"band")
            return True

        def merge(folder, destination, bands):
            destination.write_bytes(folder.name.encode())
            return destination

        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch.object(self.gee, "safe_get_collection_size", return_value=4),
            patch.object(self.gee, "iter_collection_paginated", return_value=enumerate_images(images)),
            patch.object(self.gee, "robust_download_band", side_effect=download),
            patch.object(self.gee, "merge_bands", side_effect=merge),
            patch.object(self.gee, "calc_missing_ratio_multiband", side_effect=[0.10, 0.20, 0.05]),
        ):
            results = self.gee.process_download(
                self.root, "2022-01-01", "2022-01-02", (1, 1, 2, 2), ["B8", "B8A"], log_to_file=False,
            )
        self.assertEqual(len(results), 1)
        retained = next(iter(results.values())).path
        expected_id = hashlib.sha256(b"tile/C").hexdigest()[:12]
        self.assertEqual(retained.name, f"{timestamp}_tile_C_{expected_id}.tif")
        self.assertTrue(retained.exists())
        self.assertEqual(list((self.root / "2merged_tif").glob("*.tif")), [retained])
        self.assertTrue(any((self.root / "1source_tif").glob(f"{timestamp}_tile_A_*")))


def enumerate_images(images):
    return [(image, index) for index, image in enumerate(images)]


@unittest.skipUnless(
    all(importlib.util.find_spec(name) is not None for name in ("ee", "rasterio", "numpy", "pandas")),
    "Optional download dependencies are not installed",
)
class RealGeoMergeTests(unittest.TestCase):
    """Tiny, offline rasters verify the actual rasterio merge and metadata."""

    def setUp(self):
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        from sateline_ml import gee

        self.np, self.rasterio, self.gee = np, rasterio, gee
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for filename, value, resolution in (
            ("B8_tile0.tif", 123, 10),
            ("B8A_tile0.tif", 999, 20),
            ("B8_tile0_reproj.tif", 456, 10),
        ):
            with rasterio.open(
                self.root / filename, "w", driver="GTiff", width=2, height=2,
                count=1, dtype="uint16", crs="EPSG:32651", nodata=0,
                transform=from_origin(500000, 3400000, resolution, resolution),
            ) as dataset:
                dataset.write(np.full((2, 2), value, dtype=np.uint16), 1)

    def assert_merged_product(self, path):
        with self.rasterio.open(path) as product:
            self.assertEqual(product.descriptions, ("B8", "B8A"))
            self.assertEqual((product.count, product.height, product.width), (2, 2, 2))
            self.assertEqual(product.crs.to_string(), "EPSG:32651")
            self.assertEqual(product.transform.a, 10)
            self.np.testing.assert_allclose(product.read(1), 123)
            self.np.testing.assert_allclose(product.read(2), 999)

    def test_exact_b8_values_and_native_resolution_alignment(self):
        output = self.gee.merge_bands(self.root, self.root / "merged.tif", ["B8", "B8A"])
        self.assert_merged_product(output)

    def test_legacy_merge_uses_complete_described_common_grid(self):
        output = self.gee.merge_all_bands(
            self.root, ["B8", "B8A"], self.root / "merged_bands", self.root / "final.tif",
            target_crs="EPSG:32651",
        )
        self.assert_merged_product(output)

    def test_missing_band_creates_no_partial_product(self):
        output = self.root / "missing.tif"
        with self.assertRaises(FileNotFoundError):
            self.gee.merge_bands(self.root, output, ["B8", "B8A", "B11"])
        self.assertFalse(output.exists())
        intermediate = self.root / "partial_bands"
        with self.assertRaises(FileNotFoundError):
            self.gee.merge_all_bands(self.root, ["B8", "B8A", "B11"], intermediate, output)
        self.assertFalse(intermediate.exists())
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
