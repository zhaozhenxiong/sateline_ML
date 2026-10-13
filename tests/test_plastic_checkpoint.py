"""Small CPU checkpoint round trips; no training or external data required."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest


HAS_TRAINING = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy", "pandas", "tqdm"))


@unittest.skipUnless(HAS_TRAINING, "Install sateline-ml[detection] to test plastic checkpoints")
class PlasticCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from sateline_ml import train_model
        cls.torch = torch
        cls.training = train_model

    def _round_trip(self, legacy):
        training = self.training
        torch = self.torch
        model = training.PlasticMLP(3, hidden=(256, 128), dropout=0.25).eval()
        metadata = training.PlasticModelMetadata(3, ("a", "b", "c"), ("B2", "B3", "B4"))
        inputs = torch.tensor([[0.1, 0.2, 0.3], [0.6, 0.8, 0.9]])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "plastic.pth"
            if legacy:
                torch.save({"state_dict": model.state_dict(), "metadata": metadata}, checkpoint)
            else:
                training.save_plastic_model(checkpoint, model, metadata)
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                self.assertIsInstance(payload["metadata"], dict)
                self.assertEqual(payload["architecture"]["hidden"], [256, 128])
            before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            loaded, loaded_metadata = training.load_plastic_model(checkpoint, device="cpu")
            self.assertEqual(metadata, loaded_metadata)
            self.assertFalse(loaded.training)
            with torch.no_grad():
                torch.testing.assert_close(model(inputs), loaded(inputs), rtol=0, atol=0)
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), before)

    def test_new_checkpoint_preserves_custom_architecture_and_predictions(self):
        self._round_trip(legacy=False)

    def test_legacy_dataclass_checkpoint_infers_custom_architecture(self):
        self._round_trip(legacy=True)


if __name__ == "__main__":
    unittest.main()
