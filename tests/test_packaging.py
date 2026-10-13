"""Control-plane and installer contracts, runnable without scientific extras."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sateline_install_deps", ROOT / "scripts/install_deps.py")
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)
BUILDER_SPEC = importlib.util.spec_from_file_location("sateline_build_distribution", ROOT / "scripts/build_distribution.py")
builder = importlib.util.module_from_spec(BUILDER_SPEC)
BUILDER_SPEC.loader.exec_module(builder)


def arguments(**overrides):
    data = dict(group=None, with_heavy_ml=False, skip_geo=False, skip_gee=False,
                skip_ml=False, only_base=False, torch="auto", upgrade_pip=False, separate=False)
    return SimpleNamespace(**{**data, **overrides})


class PackagingTests(unittest.TestCase):
    def test_core_import_and_cli_work_when_optional_imports_are_blocked(self):
        # Test a fresh interpreter so already loaded packages cannot conceal an
        # accidental eager import. These are precisely the feature boundaries.
        source = """
import importlib.abc,sys
class BlockScientificImports(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch','ee','rasterio','geopandas','numpy','pandas','cv2','matplotlib','sklearn'}:
   raise AssertionError('Eager optional import: '+fullname)
sys.meta_path.insert(0,BlockScientificImports())
import sateline_ml
import sateline_ml.cli
import sateline_ml.config
import sateline_ml.doctor
assert sateline_ml.__version__ == '0.1.0'
assert sateline_ml.cli.main(['--version']) == 0
sateline_ml.cli.build_parser().parse_args(['doctor','--no-import-check'])
assert 'sateline_ml.detection' not in sys.modules
assert 'sateline_ml.gee' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-c", source], cwd=ROOT,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_extra_is_feature_union_and_does_not_pull_heavy_stacking(self):
        groups = installer.declared_groups(ROOT)
        self.assertEqual(set(groups["all"]), set(groups["download"]) | set(groups["detection"]))
        names = {installer.package_name(req) for req in groups["all"]}
        self.assertNotIn("lightgbm", names)
        self.assertNotIn("xgboost", names)
        self.assertNotIn("pydensecrf", names)
        self.assertIn("scikit-learn", {installer.package_name(r) for r in groups["stacking"]})

    def test_dependency_reader_supports_python310_without_tomllib(self):
        expected = installer.declared_groups(ROOT)
        with patch.dict(sys.modules, {"tomllib": None}):
            self.assertEqual(installer.declared_groups(ROOT), expected)

    def test_skip_flags_cannot_be_reintroduced_by_source_import_scanning(self):
        groups = installer.declared_groups(ROOT)
        args = arguments(skip_geo=True, skip_gee=True, torch="skip")
        names = {installer.package_name(r) for r in installer.selected_requirements(args, groups)}
        self.assertFalse(names & {"rasterio", "geopandas", "earthengine-api", "torch"})
        self.assertIn("numpy", names)
        base = installer.selected_requirements(arguments(only_base=True, with_heavy_ml=True), groups)
        self.assertFalse({installer.package_name(r) for r in base} &
                         {"torch", "rasterio", "geopandas", "earthengine-api", "scikit-learn", "catboost"})

    def test_cpu_torch_has_its_own_official_index_on_linux(self):
        with patch.object(installer.sys, "platform", "linux"):
            commands = installer.install_commands(arguments(torch="cpu"), ["torch>=2.2", "numpy>=1.24"])
        self.assertEqual(len(commands), 2)
        self.assertIn(installer.CPU_INDEX, commands[0])
        self.assertIn("--force-reinstall", commands[0])
        self.assertNotIn("numpy>=1.24", commands[0])
        self.assertNotIn("--index-url", commands[1])

    def test_installer_failures_propagate_a_nonzero_status(self):
        with patch.object(installer, "missing_packages", return_value=[]), \
             patch.object(installer.subprocess, "call", return_value=7) as pip:
            self.assertEqual(installer.main(["--group", "download", "--separate"]), 7)
        self.assertEqual(pip.call_count, 1)
        with patch.object(installer, "missing_packages", return_value=[]), \
             patch.object(installer.subprocess, "call", return_value=4):
            self.assertEqual(installer.main(["--upgrade-pip"]), 4)
        self.assertEqual(installer.main(["--root", "/nonexistent/sateline-root", "--dry-run"]), 2)

    def test_dry_run_does_not_scan_attic_or_virtual_environments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text((ROOT / "pyproject.toml").read_text(), encoding="utf-8")
            for name in ("attic", ".venv"):
                (root / name).mkdir()
                (root / name / "obsolete.py").write_text("import definitely_not_a_real_dependency", encoding="utf-8")
            with patch.object(installer, "missing_packages", return_value=[]), \
                 patch.object(installer.subprocess, "call") as pip:
                self.assertEqual(installer.main(["--root", str(root), "--dry-run"]), 0)
                pip.assert_not_called()
            self.assertNotIn("definitely_not_a_real_dependency", str(installer.declared_groups(root)))

    def test_build_staging_excludes_sidecars_data_weights_and_virtual_envs(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, staged = base / "source", base / "staged"
            root.mkdir()
            (root / "pyproject.toml").write_text("[project]\nname='test'", encoding="utf-8")
            (root / "pipeline.example.json").write_text("{}", encoding="utf-8")
            for filename in ("sateline_ml/module.py", "scripts/helper.py", "tests/test_helper.py",
                             "sateline_ml/._module.py", "sateline_ml/data.csv", "sateline_ml/model.pth",
                             "sateline_ml/.venv/dependency.py", "sateline_ml/__pycache__/cached.py",
                             "mydata/scene.tif", "attic/obsolete.py"):
                path = root / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("test", encoding="utf-8")
            builder.stage_source(root, staged)
            copied = {path.relative_to(staged).as_posix() for path in staged.rglob("*") if path.is_file()}
            self.assertEqual(copied, {"pyproject.toml", "pipeline.example.json", "sateline_ml/module.py",
                                      "scripts/helper.py", "tests/test_helper.py"})

    def test_build_failure_returns_nonzero_without_publishing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dist"
            with patch.object(builder.subprocess, "run", return_value=SimpleNamespace(returncode=5)):
                self.assertEqual(builder.main(["--root", str(ROOT), "--output-dir", str(output)]), 5)
            self.assertFalse(output.exists())


class DoctorTests(unittest.TestCase):
    def setUp(self):
        from sateline_ml import doctor
        self.doctor = doctor

    @staticmethod
    def present(name, module):
        return {"name": name, "module": module, "version": "test", "status": "ok", "detail": "Installed"}

    def test_all_checks_enabled_stages_but_explicit_stage_checks_disabled_one(self):
        config = {"download": {"enabled": False, "project": None}, "detection": {"enabled": False}}
        with patch.object(self.doctor.subprocess, "run") as run:
            report = self.doctor.check_environment(config, stage="all", check_imports=False)
            self.assertTrue(report["ok"])
            self.assertEqual(report["selected_stages"], [])
            self.assertEqual(report["packages"], [])
            run.assert_not_called()
        with patch.object(self.doctor, "_package_info", side_effect=self.present):
            report = self.doctor.check_environment(config, stage="download", check_imports=False)
        self.assertFalse(report["ok"])
        self.assertEqual(report["selected_stages"], ["download"])
        self.assertTrue(any("gee_project" in message for message in report["errors"]))

    def test_isolated_import_abort_is_reported_and_does_not_escape(self):
        failure = subprocess.CompletedProcess([], -6, stdout="", stderr="OMP: duplicate runtime")
        with patch.object(self.doctor.subprocess, "run", return_value=failure) as run:
            result = self.doctor._probe_import("torch")
        self.assertEqual(result["status"], "error")
        self.assertIn("exited -6", result["detail"])
        self.assertEqual(run.call_args.args[0][-1], "torch")
        self.assertNotIn("KMP_DUPLICATE_LIB_OK", run.call_args.kwargs["env"])

    def test_combined_native_import_failure_is_also_reported(self):
        def probe(module):
            if module == "sateline_ml.detection":
                return {"status": "error", "detail": "Import process exited -6: OpenMP conflict"}
            return {"status": "ok", "detail": "Imported", "data": {"cuda": False, "mps": False}}
        with patch.object(self.doctor, "_package_info", side_effect=self.present), \
             patch.object(self.doctor, "_probe_import", side_effect=probe):
            report = self.doctor.check_environment(stage="detect")
        self.assertFalse(report["ok"])
        self.assertTrue(any("import:detection" in error for error in report["errors"]))

    def test_device_fallback_and_missing_plastic_assets_are_reviewable(self):
        config = {"detection": {"device": "cuda", "plastic_library_csv": "/nonexistent/spectra.csv",
                                "plastic_model_path": None, "parameters": {}}}
        def probe(module):
            return {"status": "ok", "detail": "Imported", "data": {"cuda": False, "mps": False}}
        with patch.object(self.doctor, "_package_info", side_effect=self.present), \
             patch.object(self.doctor, "_probe_import", side_effect=probe):
            report = self.doctor.check_environment(config)
        self.assertTrue(report["ok"])
        checks = {item["name"]: item for item in report["checks"]}
        self.assertEqual(checks["device"]["selected"], "cpu")
        self.assertEqual(checks["device"]["status"], "warning")
        self.assertEqual(checks["plastic_assets"]["available"], {"library": False, "model": False})
        json.dumps(report)  # Diagnostics can always be serialized by the CLI.

    def test_metadata_only_doctor_does_not_create_paths_or_run_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"paths": {"input_dir": root / "inputs", "shapefile_dir": None,
                                "output_dir": root / "results"}, "detection": {"device": "auto"}}
            with patch.object(self.doctor, "_package_info", side_effect=self.present), \
                 patch.object(self.doctor.subprocess, "run") as run:
                report = self.doctor.check_environment(config, check_imports=False)
            run.assert_not_called()
            self.assertFalse((root / "results").exists())
            self.assertFalse((root / "inputs").exists())
            self.assertTrue(report["ok"])


if __name__ == "__main__":
    unittest.main()
