import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .ape import (
    IXIStyleAPE,
    SemanticAssignmentAPE,
    migrate_v31_checkpoint,
    normalized_assignment_entropy,
)
from .config import SEMANTIC_ASSIGNMENT_MODE, resolve_ape_consistency
from .data import binary_surface_distance_metrics
from .registration import (
    FIVE_SCALE_ARCHITECTURE,
    THREE_LEVEL_ARCHITECTURE,
    CleanRegistrationModel,
    infer_registration_architecture,
    registration_state_dict,
)

WORKSPACE = Path(__file__).resolve().parent.parent


class SurfaceMetricTests(unittest.TestCase):
    def test_identical_binary_masks_have_zero_surface_distance(self):
        mask = np.zeros((8, 8, 8), dtype=np.bool_)
        mask[2:6, 2:6, 2:6] = True
        result = binary_surface_distance_metrics(mask, mask, (2.0, 2.0, 2.0))
        self.assertEqual(result["hd95_mm"], 0.0)
        self.assertEqual(result["assd_mm"], 0.0)
        self.assertFalse(result["penalized"])

    def test_one_voxel_shift_respects_physical_spacing(self):
        fixed = np.zeros((12, 12, 12), dtype=np.bool_)
        warped = np.zeros_like(fixed)
        fixed[3:8, 3:8, 3:8] = True
        warped[4:9, 3:8, 3:8] = True
        result = binary_surface_distance_metrics(warped, fixed, (2.0, 2.0, 2.0))
        self.assertAlmostEqual(result["hd95_mm"], 2.0, places=6)
        self.assertGreater(result["assd_mm"], 0.0)
        self.assertLessEqual(result["assd_mm"], 2.0)

    def test_one_sided_empty_mask_uses_grid_diagonal_penalty(self):
        fixed = np.zeros((4, 5, 6), dtype=np.bool_)
        fixed[1, 1, 1] = True
        warped = np.zeros_like(fixed)
        result = binary_surface_distance_metrics(warped, fixed, (2.0, 2.0, 2.0))
        expected = float(np.linalg.norm(np.array([3, 4, 5]) * 2.0))
        self.assertAlmostEqual(result["hd95_mm"], expected)
        self.assertAlmostEqual(result["assd_mm"], expected)
        self.assertTrue(result["penalized"])


class APEContractTests(unittest.TestCase):
    def _features(self):
        generator = torch.Generator().manual_seed(5)
        return [
            torch.randn((1, 256, 8, 8, 8), generator=generator),
            torch.randn((1, 256, 8, 4, 4), generator=generator),
            torch.randn((1, 256, 8, 4, 4), generator=generator),
        ]

    def test_ixi_assignment_is_categorical_inside_mask(self):
        model = IXIStyleAPE(channels=8)
        body = torch.ones((1, 1, 8, 8, 8))
        _, assignments, _, masks = model(self._features(), body)
        for assignment, mask in zip(assignments, masks):
            mass = assignment.sum(dim=1, keepdim=True)
            self.assertTrue(torch.allclose(mass, mask, atol=1e-5))
            entropy = normalized_assignment_entropy(assignment)
            self.assertTrue(torch.isfinite(entropy).all())
            self.assertGreaterEqual(float(entropy.min()), 0.0)
            self.assertLessEqual(float(entropy.max()), 1.0 + 1e-5)

    def test_semantic_state_layout_loads_local_best_ape(self):
        checkpoint_path = (
            WORKSPACE / "abdomen_ctct" / "abdomen_ape_v31_best_fine_fixed.pth"
        )
        if not checkpoint_path.exists():
            self.skipTest("Local v3.1 checkpoint is unavailable")
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model = SemanticAssignmentAPE(
            checkpoint["k_list"],
            checkpoint["labels_per_stage"],
            checkpoint["temperature"],
            checkpoint["ape_channels"],
            checkpoint.get("initial_slot_bias", -2.0),
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    def test_v31_migration_preserves_projectors_and_prototypes(self):
        source = SemanticAssignmentAPE(channels=8)
        checkpoint = {
            "ape_variant": "hierarchical_calibrated_sigmoid_no_residual_v31",
            "epoch": 3,
            "k_list": source.K,
            "labels_per_stage": source.labels_per_stage,
            "temperature": float(source.current_temp),
            "ape_channels": 8,
            "initial_slot_bias": -2.0,
            "model_state_dict": source.state_dict(),
        }
        categorical, metadata = migrate_v31_checkpoint(checkpoint)
        self.assertEqual(metadata["assignment_mode"], "categorical_softmax")
        for target, semantic, discovery in zip(
            categorical.protos, source.semantic_protos, source.discovery_protos
        ):
            self.assertTrue(
                torch.equal(target, torch.cat([semantic, discovery], dim=0))
            )


class ConsistencyPolicyTests(unittest.TestCase):
    def test_new_runs_default_on(self):
        self.assertTrue(resolve_ape_consistency(None, None, resume_training=True))

    def test_existing_checkpoint_reproduces_recorded_off(self):
        checkpoint = {"use_ape_consistency": False}
        self.assertFalse(
            resolve_ape_consistency(None, checkpoint, resume_training=False)
        )
        self.assertFalse(
            resolve_ape_consistency(None, checkpoint, resume_training=True)
        )

    def test_resume_rejects_objective_change(self):
        with self.assertRaisesRegex(ValueError, "mismatch"):
            resolve_ape_consistency(
                True, {"use_ape_consistency": False}, resume_training=True
            )


class RegistrationLayoutTests(unittest.TestCase):
    class DummyGuidance(nn.Module):
        model_mode = SEMANTIC_ASSIGNMENT_MODE
        ape_variant = "hierarchical_semantic_assignment_v31"
        ape_channels = 8
        ape_registration_status = "test"

        def forward(self, volume, body_mask):
            batch, _, depth, height, width = volume.shape
            shapes = (
                (depth, height // 4, width // 4),
                (depth, height // 8, width // 8),
                (depth, height // 16, width // 16),
            )
            semantic, assignments, projected, masks = [], [], [], []
            for shape in shapes:
                base = torch.nn.functional.interpolate(
                    volume, size=shape, mode="trilinear", align_corners=True
                )
                mask = torch.nn.functional.interpolate(
                    body_mask.float(), size=shape, mode="nearest"
                )
                feature = base.repeat(1, 8, 1, 1, 1) * mask
                semantic.append(feature)
                projected.append(feature)
                assignments.append(torch.cat([0.75 * mask, 0.25 * mask], dim=1))
                masks.append(mask)
            return semantic, assignments, projected, masks

    def test_forward_shape_and_checkpoint_prefixes(self):
        size = (32, 32, 32)
        model = CleanRegistrationModel(
            size,
            2,
            self.DummyGuidance(),
            use_fusion=True,
            registration_architecture=THREE_LEVEL_ARCHITECTURE,
        )
        moving = torch.rand((1, 1, *size))
        fixed = torch.rand((1, 1, *size))
        body = torch.ones_like(moving)
        warped, flow = model(moving, fixed, body, body)
        self.assertEqual(tuple(warped.shape), tuple(moving.shape))
        self.assertEqual(tuple(flow.shape), (1, 3, *size))
        prefixes = {key.split(".")[0] for key in registration_state_dict(model)}
        self.assertIn("ape_eadf", prefixes)
        self.assertIn("ape_decoder3", prefixes)
        self.assertIn("fine_refine_encoder", prefixes)
        self.assertNotIn("ape_depth_l2", prefixes)
        self.assertNotIn("half_refine_encoder", prefixes)

    def test_five_scale_forward_and_state_prefixes(self):
        size = (32, 32, 32)
        model = CleanRegistrationModel(
            size,
            2,
            self.DummyGuidance(),
            use_fusion=True,
            registration_architecture=FIVE_SCALE_ARCHITECTURE,
        )
        moving = torch.rand((1, 1, *size))
        fixed = torch.rand((1, 1, *size))
        body = torch.ones_like(moving)
        warped, flow = model(moving, fixed, body, body)
        self.assertEqual(tuple(warped.shape), tuple(moving.shape))
        self.assertEqual(tuple(flow.shape), (1, 3, *size))
        prefixes = {key.split(".")[0] for key in registration_state_dict(model)}
        self.assertIn("ape_depth_l2", prefixes)
        self.assertIn("half_refine_encoder", prefixes)
        self.assertEqual(
            infer_registration_architecture(
                {"model_state_dict": registration_state_dict(model)}
            ),
            FIVE_SCALE_ARCHITECTURE,
        )


if __name__ == "__main__":
    unittest.main()
