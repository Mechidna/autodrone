import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_openvins_drift import (  # noqa: E402
    best_position_scale_multiplier,
    compute_window_metrics,
    longest_true_run,
    orientation_error_series,
)


def _yaw_rotation(degrees):
    angle = math.radians(degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(
        (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=float,
    )


class AuditOpenVinsDriftTests(unittest.TestCase):
    def test_best_scale_recovers_inverse_metric_overscale(self):
        truth = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 1.0, 0.0))
        )
        estimate = 1.2 * truth

        multiplier = best_position_scale_multiplier(estimate, truth)

        self.assertAlmostEqual(multiplier, 1.0 / 1.2, places=12)

    def test_orientation_decomposition_separates_heading_from_tilt(self):
        truth = np.tile(np.eye(3), (3, 1, 1))
        estimate = np.stack((_yaw_rotation(0.0), _yaw_rotation(5.0), _yaw_rotation(10.0)))

        errors = orientation_error_series(estimate, truth)

        np.testing.assert_allclose(errors["heading_deg"], (0.0, 5.0, 10.0), atol=1e-10)
        np.testing.assert_allclose(errors["yaw_deg"], (0.0, 5.0, 10.0), atol=1e-10)
        np.testing.assert_allclose(errors["geodesic_deg"], (0.0, 5.0, 10.0), atol=1e-10)
        np.testing.assert_allclose(errors["tilt_deg"], 0.0, atol=1e-10)

    def test_sliding_windows_preserve_scale_and_direction(self):
        timestamp = np.linspace(0.0, 4.0, 41)
        truth = np.column_stack((timestamp, np.zeros((41, 2))))
        estimate = 1.2 * truth
        position_error = np.linalg.norm(estimate - truth, axis=1)
        visual = {
            "accepted_features": np.ones(41),
            "active_tracks": np.full(41, 10.0),
            "slam_features": np.full(41, 5.0),
        }

        windows = compute_window_metrics(
            timestamp,
            estimate,
            truth,
            position_error,
            visual,
            window_s=1.0,
            min_displacement_m=0.5,
        )

        self.assertGreater(len(windows), 20)
        np.testing.assert_allclose(
            [row["estimated_to_truth_scale"] for row in windows],
            1.2,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            [row["direction_error_deg"] for row in windows],
            0.0,
            atol=1e-10,
        )

    def test_longest_true_run_includes_endpoints(self):
        run = longest_true_run(
            np.asarray((False, True, True, False, True, True, True, False))
        )
        self.assertEqual(run, (4, 6))
        self.assertIsNone(longest_true_run(np.zeros(4, dtype=bool)))


if __name__ == "__main__":
    unittest.main()
