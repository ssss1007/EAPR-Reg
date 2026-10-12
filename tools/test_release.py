"""CPU checks for release portability and data/checkpoint contracts."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from ixi__eapr.checkpoints import load_registration_weights
from ixi__eapr.sam2_runtime import build_sam2_from_paths
from tools import prepare_abdomen
from tools.build_release import ROOT, release_files


class CheckpointTests(unittest.TestCase):
    def test_plan_d_checkpoint_and_guidance_outputs_match(self):
        from ape_plan_d_supervised_record_copy import APE
        from abdomen_eapr import registration  # Sets the bundled voxelmorph import path.
        from voxelmorph.torch.iirp import APEGuidanceHead
        source = APE([24, 10, 6], channels=4).eval()
        target = APEGuidanceHead(channels=4).eval()
        target.load_state_dict(source.state_dict(), strict=True)
        features = [torch.randn(1, 256, 4, 4, 4) for _ in range(3)]
        with torch.no_grad():
            expected, actual = source(features), target(features)
        for expected_levels, actual_levels in zip(expected, actual):
            for expected_level, actual_level in zip(expected_levels, actual_levels):
                self.assertTrue(torch.equal(expected_level, actual_level))

    def test_plain_and_wrapped_weights_preserve_parameters(self):
        model = torch.nn.Linear(2, 3)
        expected = {key: value.clone() for key, value in model.state_dict().items()}
        for state in (expected, {"model_state": expected}):
            target = torch.nn.Linear(2, 3)
            load_registration_weights(target, state)
            self.assertTrue(torch.equal(target.weight, expected["weight"]))

    def test_missing_and_unexpected_parameters_fail(self):
        model = torch.nn.Linear(2, 3)
        for state in ({"weight": model.weight}, {**model.state_dict(), "wrong": torch.zeros(1)}):
            with self.assertRaisesRegex(RuntimeError, "architecture mismatch"):
                load_registration_weights(model, state)

    def test_cached_grids_are_rebuilt(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 3))
        model[0].register_buffer("grid", torch.ones(2))
        state = dict(model.state_dict())
        state["0.grid"] = torch.zeros(7)
        load_registration_weights(model, state)
        self.assertTrue(torch.equal(model[0].grid, torch.ones(2)))


class SAM2PathTests(unittest.TestCase):
    def test_supplied_checkpoint_and_config_reach_builder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "custom.pt"
            checkpoint.write_bytes(b"test fixture, not real weights")
            module = SimpleNamespace(__file__=str(root / "build_sam.py"), build_sam2=Mock())
            with patch("ixi__eapr.sam2_runtime.importlib.import_module", return_value=module):
                build_sam2_from_paths("configs/custom.yaml", checkpoint, device="cpu")
            module.build_sam2.assert_called_once_with(
                "configs/custom.yaml", str(checkpoint.resolve()), device="cpu",
                apply_postprocessing=False)

    def test_absolute_external_config_is_not_silently_substituted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module = SimpleNamespace(__file__=str(root / "sam2" / "build_sam.py"))
            with patch("ixi__eapr.sam2_runtime.importlib.import_module", return_value=module):
                with self.assertRaisesRegex(ValueError, "package-relative"):
                    build_sam2_from_paths(root / "external.yaml", root / "custom.pt")


class DataAndPackageTests(unittest.TestCase):
    def test_three_item_processed_payload_validates(self):
        split = json.loads((ROOT / "protocols/prospective_split_v3.json").read_text())
        shape = (2, 3, 4)
        payload = (np.zeros(shape, np.float32), np.zeros(shape, np.uint8), np.ones(shape, bool))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifests").mkdir()
            for name in ("train", "val", "test"):
                (root / name.title()).mkdir()
                for case in split[name]:
                    (root / name.title() / f"{case}.pkl").write_bytes(pickle.dumps(payload))
                pairs = prepare_abdomen.build_pairs(split[name])
                (root / "manifests" / f"{name}_pairs.json").write_text(json.dumps(pairs))
            with patch.object(prepare_abdomen, "EXPECTED_STORED_SHAPE", shape):
                prepare_abdomen.validate_output(root, split)

    def test_release_has_only_source_and_two_abdomen_training_entries(self):
        names = [path.relative_to(ROOT).as_posix() for path in release_files()]
        self.assertEqual(sorted(name for name in names if name.startswith("abdomen_eapr/train")),
                         ["abdomen_eapr/train_ape_v31.py", "abdomen_eapr/train_registration.py"])
        forbidden = {".pkl", ".pt", ".pth", ".nii", ".zip", ".out", ".log"}
        self.assertFalse(any(Path(name).suffix in forbidden for name in names))
        self.assertFalse(any(name.startswith(("abdomen_ct/", "abdomen_ctct/", "IIRP/")) for name in names))


if __name__ == "__main__":
    unittest.main()
