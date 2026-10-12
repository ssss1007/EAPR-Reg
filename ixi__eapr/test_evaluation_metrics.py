import unittest

import numpy as np

try:
    from .evaluation_metrics import binary_surface_distance_metrics
except ImportError:
    from evaluation_metrics import binary_surface_distance_metrics


class SurfaceDistanceTests(unittest.TestCase):
    def test_identical_masks(self):
        mask = np.zeros((8, 8, 8), dtype=np.bool_)
        mask[2:6, 2:6, 2:6] = True
        result = binary_surface_distance_metrics(mask, mask, (1.0, 1.0, 1.0))
        self.assertEqual(result["hd95_mm"], 0.0)
        self.assertEqual(result["assd_mm"], 0.0)

    def test_shift_uses_physical_spacing(self):
        fixed = np.zeros((12, 12, 12), dtype=np.bool_)
        warped = np.zeros_like(fixed)
        fixed[3:8, 3:8, 3:8] = True
        warped[4:9, 3:8, 3:8] = True
        result = binary_surface_distance_metrics(warped, fixed, (2.0, 1.0, 1.0))
        self.assertAlmostEqual(result["hd95_mm"], 2.0, places=6)
        self.assertGreater(result["assd_mm"], 0.0)

    def test_empty_prediction_is_penalized(self):
        fixed = np.zeros((4, 5, 6), dtype=np.bool_)
        fixed[1, 1, 1] = True
        warped = np.zeros_like(fixed)
        result = binary_surface_distance_metrics(warped, fixed, (1.0, 1.0, 1.0))
        expected = float(np.linalg.norm(np.array([3, 4, 5])))
        self.assertAlmostEqual(result["hd95_mm"], expected)
        self.assertAlmostEqual(result["assd_mm"], expected)
        self.assertTrue(result["penalized"])


if __name__ == "__main__":
    unittest.main()
