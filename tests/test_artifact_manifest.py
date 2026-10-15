"""Check both released model contracts and their tracked evidence without inference."""

import hashlib
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "artifact_manifest", ROOT / "scripts/artifact_manifest.py"
)
assert SPEC and SPEC.loader
MANIFEST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MANIFEST)


class ArtifactContractTests(unittest.TestCase):
    """Do not confuse reported metrics, source annotations and an independent test."""

    def test_real_detector_report_and_weight_identity(self):
        """The tracked CarFusion report stays reported and points to the real checkpoint."""
        contract, scores = MANIFEST.validate(ROOT)
        self.assertEqual(contract["evaluation"]["status"], "reported")
        self.assertEqual(scores["OKS-mAP"], 0.5038477111813657)
        self.assertEqual(contract["output"]["keypoints"], 14)
        self.assertEqual(contract["hub"]["revision"], "d277c2d8c9b5ee49d9fed85fe43bdd72b16a0f06")

    def test_synthetic_model_uses_different_split_and_schema(self):
        """The current Hub test cannot become an independent score by relabelling it."""
        contract, _ = MANIFEST.validate(ROOT, "manifests/synthetic-model.json")
        self.assertEqual(contract["output"]["keypoints"], 24)
        self.assertEqual(contract["dataset"]["published_test_training_frames"], 234)
        self.assertEqual(contract["dataset"]["model_splits"], {"train": 2259, "validation": 251})
        crosswalk = ROOT / contract["dataset"]["crosswalk"]
        self.assertEqual(
            hashlib.sha256(crosswalk.read_bytes()).hexdigest(),
            contract["dataset"]["crosswalk_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
