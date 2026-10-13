"""AOI validation must reject alignment errors without changing source files."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


HAS_GEO = all(importlib.util.find_spec(name) is not None for name in ("rasterio", "geopandas", "numpy", "pandas", "shapely"))


@unittest.skipUnless(HAS_GEO, "Install sateline-ml[detection] to test AOI validation")
class AOIValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import geopandas as gpd
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        from shapely.geometry import box
        from sateline_ml.inputs import prepare_raster, validate_aoi
        cls.gpd, cls.np, cls.rasterio, cls.from_origin, cls.box = gpd, np, rasterio, staticmethod(from_origin), staticmethod(box)
        cls.validate_aoi = staticmethod(validate_aoi)
        cls.prepare_raster = staticmethod(prepare_raster)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image = self.root / "scene.tif"
        with self.rasterio.open(self.image, "w", driver="GTiff", height=16, width=16, count=1,
                                dtype="uint8", crs="EPSG:32651", transform=self.from_origin(400000, 3400000, 10, 10)) as dst:
            dst.write(self.np.zeros((1, 16, 16), dtype="uint8"))

    def _shapefile(self, geometries, *, crs="EPSG:32651", name="aoi", attributes=None):
        frame = self.gpd.GeoDataFrame(attributes or {"lake": list(range(len(geometries)))}, geometry=geometries, crs=crs)
        path = self.root / f"{name}.shp"
        frame.to_file(path)
        return path

    def test_accepts_reprojected_aoi_and_keeps_read_only_source_unchanged(self):
        polygon = self.box(400020, 3399880, 400100, 3399960)
        frame = self.gpd.GeoDataFrame({"lake": [7]}, geometry=[polygon], crs="EPSG:32651").to_crs("EPSG:4326")
        shapefile = self.root / "aoi.shp"
        frame.to_file(shapefile)
        before = hashlib.sha256(self.image.read_bytes()).hexdigest()
        self.image.chmod(0o444)
        self.addCleanup(self.image.chmod, 0o644)
        self.validate_aoi(self.image, shapefile)
        self.assertEqual(hashlib.sha256(self.image.read_bytes()).hexdigest(), before)

    def test_missing_crs_is_rejected(self):
        shapefile = self._shapefile([self.box(400020, 3399880, 400100, 3399960)])
        shapefile.with_suffix(".prj").unlink()
        with self.assertRaisesRegex(ValueError, "AOI has no CRS"):
            self.validate_aoi(self.image, shapefile)

    def test_nonintersecting_aoi_is_rejected(self):
        shapefile = self._shapefile([self.box(500000, 3500000, 500100, 3500100)])
        with self.assertRaisesRegex(ValueError, "does not intersect"):
            self.validate_aoi(self.image, shapefile)

    def test_subpixel_aoi_without_pixel_center_is_rejected(self):
        shapefile = self._shapefile([self.box(400001, 3399998, 400002, 3399999)])
        with self.assertRaisesRegex(ValueError, "no pixel centers"):
            self.validate_aoi(self.image, shapefile)

    def test_attribute_query_and_index_filters_match_selected_aoi(self):
        shapefile = self._shapefile([
            self.box(500000, 3500000, 500100, 3500100),
            self.box(400020, 3399880, 400100, 3399960),
        ], attributes={"lake": [1, 2]})
        for filters in ({"filter_field": "lake", "filter_values": [2]}, {"filter_query": "lake == 2"}, {"filter_indices": [1]}):
            with self.subTest(filters=filters):
                self.validate_aoi(self.image, shapefile, **filters)
        with self.assertRaisesRegex(ValueError, "no geometries"):
            self.validate_aoi(self.image, shapefile, filter_field="lake", filter_values=[99])
        with self.assertRaisesRegex(ValueError, "filtering failed"):
            self.validate_aoi(self.image, shapefile, filter_field="missing", filter_values=[2])

    def test_projection_failure_is_reported(self):
        shapefile = self._shapefile([self.box(400020, 3399880, 400100, 3399960)])
        with patch.object(self.gpd.GeoDataFrame, "to_crs", side_effect=RuntimeError("bad projection")), \
             self.assertRaisesRegex(ValueError, "AOI projection failed"):
            self.validate_aoi(self.image, shapefile)

    def test_raster_crs_override_creates_exact_read_only_source_vrt(self):
        wrong = self.root / "wrong-geographic.tif"
        values = self.np.arange(3 * 16 * 16, dtype="uint16").reshape(3, 16, 16)
        transform = self.from_origin(194110, 3500620, 10, 10)
        descriptions = ("B2", "B3", "B4")
        with self.rasterio.open(wrong, "w", driver="GTiff", height=16, width=16, count=3,
                                dtype="uint16", crs="EPSG:4326", transform=transform, nodata=65535) as dst:
            dst.write(values)
            dst.descriptions = descriptions
        wrong.chmod(0o444)
        self.addCleanup(wrong.chmod, 0o644)
        before = hashlib.sha256(wrong.read_bytes()).hexdigest()
        destination = self.root / "output" / "assigned.vrt"
        with self.assertRaisesRegex(ValueError, "detection.raster_crs_override"):
            self.prepare_raster(wrong, destination)
        self.assertFalse(destination.parent.exists())
        result = self.prepare_raster(wrong, destination, crs_override="EPSG:32651")
        self.assertEqual(result, destination.resolve())
        with self.rasterio.open(result) as vrt:
            self.assertEqual(vrt.driver, "VRT")
            self.assertEqual(vrt.crs.to_epsg(), 32651)
            self.assertEqual(vrt.transform, transform)
            self.assertEqual((vrt.height, vrt.width, vrt.count), (16, 16, 3))
            self.assertEqual(vrt.descriptions, descriptions)
            self.assertEqual(vrt.nodata, 65535)
            self.np.testing.assert_array_equal(vrt.read(), values)
        self.assertEqual(hashlib.sha256(wrong.read_bytes()).hexdigest(), before)
        self.assertLess(destination.stat().st_size, 10000)

    def test_raster_without_override_returns_valid_source_without_writes(self):
        destination = self.root / "not-created" / "unused.vrt"
        self.assertEqual(self.prepare_raster(self.image, destination), self.image.resolve())
        self.assertFalse(destination.parent.exists())

    def test_invalid_raster_crs_override_and_source_alias_are_rejected(self):
        destination = self.root / "output" / "assigned.vrt"
        with self.assertRaisesRegex(ValueError, "Invalid raster_crs_override"):
            self.prepare_raster(self.image, destination, crs_override="not-a-crs")
        self.assertFalse(destination.parent.exists())
        with self.assertRaisesRegex(ValueError, "must differ"):
            self.prepare_raster(self.image, self.image, crs_override="EPSG:32651")


if __name__ == "__main__":
    unittest.main()
