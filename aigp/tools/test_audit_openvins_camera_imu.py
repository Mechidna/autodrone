import math
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_openvins_camera_imu import (
    ImuSeries,
    _normalized_bearings,
    camera_to_imu_rotation,
    epipolar_residuals_px,
    integrate_imu_frame_rotation,
)


def project(points):
    points = np.asarray(points, dtype=np.float64)
    return points[:, :2] / points[:, 2:3]


class CameraImuAuditTests(unittest.TestCase):
    def test_camera_to_imu_is_a_proper_rotation(self):
        rotation = camera_to_imu_rotation(20.0)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)
        np.testing.assert_allclose(
            rotation[:, 2],
            [math.cos(math.radians(20.0)), 0.0, -math.sin(math.radians(20.0))],
            atol=1e-12,
        )

    def test_imu_rotation_integration_maps_start_coordinates_to_end(self):
        timestamps = np.linspace(0.0, 1.0, 101)
        angular_velocity = np.tile([0.0, 0.0, math.pi / 2.0], (101, 1))
        imu = ImuSeries(timestamps, angular_velocity)
        integrated = integrate_imu_frame_rotation(imu, 0.0, 1.0)
        expected, _ = cv2.Rodrigues(np.asarray([0.0, 0.0, -math.pi / 2.0]))
        np.testing.assert_allclose(integrated, expected, atol=1e-10)

    def test_epipolar_score_prefers_correct_tilt(self):
        rng = np.random.default_rng(7)
        first_points = np.column_stack(
            (
                rng.uniform(-2.0, 2.0, 200),
                rng.uniform(-1.2, 1.2, 200),
                rng.uniform(6.0, 15.0, 200),
            )
        )
        camera_to_imu = camera_to_imu_rotation(20.0)
        imu_rotation, _ = cv2.Rodrigues(
            np.radians(np.asarray([-2.0, 4.0, -7.0]))
        )
        correct_rotation = camera_to_imu.T @ imu_rotation @ camera_to_imu
        translation = np.asarray([0.12, -0.04, 0.2])
        second_points = (correct_rotation @ first_points.T).T + translation
        first = np.column_stack((project(first_points), np.ones(200)))
        second = np.column_stack((project(second_points), np.ones(200)))

        correct = epipolar_residuals_px(first, second, correct_rotation, 320.0)
        wrong_camera_to_imu = camera_to_imu_rotation(-20.0)
        wrong_rotation = (
            wrong_camera_to_imu.T @ imu_rotation @ wrong_camera_to_imu
        )
        wrong = epipolar_residuals_px(first, second, wrong_rotation, 320.0)

        self.assertLess(float(np.median(correct)), 1e-8)
        self.assertGreater(float(np.median(wrong)), 0.5)

    def test_horizontal_flip_recovers_mirrored_bearings(self):
        pixels = np.asarray([[100.0, 80.0], [319.5, 180.0], [550.0, 300.0]])
        canonical = _normalized_bearings(
            pixels, fx=320.0, fy=320.0, cx=320.0, cy=180.0, mirror_x=False
        )
        mirrored_pixels = pixels.copy()
        mirrored_pixels[:, 0] = 639.0 - mirrored_pixels[:, 0]
        recovered = _normalized_bearings(
            mirrored_pixels,
            fx=320.0,
            fy=320.0,
            cx=319.0,
            cy=180.0,
            mirror_x=True,
        )
        np.testing.assert_allclose(recovered, canonical, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
