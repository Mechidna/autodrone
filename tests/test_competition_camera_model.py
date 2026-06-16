import math
import unittest

import numpy as np

from autonomy_core.core.competition_config import RuntimeCompetitionConfig
from autonomy_core.core.frame_conventions import (
    INTERNAL_BODY_FLU,
    MAVLINK_BODY_FRD,
    MAVLINK_LOCAL_NED,
    OPENCV_CAMERA_OPTICAL,
    body_frd_to_internal_body_flu_rotmat,
    body_frd_point_to_camera_optical,
    camera_translation_body_frd,
    mavlink_body_frd_to_opencv_camera_rotmat,
    official_body_frd_to_camera_rotmat,
    official_camera_matrix,
    official_camera_to_body_frd_rotmat,
    official_camera_to_internal_body_flu_rotmat,
    official_dist_coeffs,
    opencv_camera_to_mavlink_body_frd_rotmat,
    project_body_frd_point_to_pixel,
)


class CompetitionCameraModelTests(unittest.TestCase):
    def test_frame_conventions_are_explicit(self):
        self.assertEqual(MAVLINK_LOCAL_NED.x_axis, "north")
        self.assertEqual(MAVLINK_LOCAL_NED.y_axis, "east")
        self.assertEqual(MAVLINK_LOCAL_NED.z_axis, "down")

        self.assertEqual(MAVLINK_BODY_FRD.x_axis, "forward")
        self.assertEqual(MAVLINK_BODY_FRD.y_axis, "right")
        self.assertEqual(MAVLINK_BODY_FRD.z_axis, "down")

        self.assertEqual(INTERNAL_BODY_FLU.x_axis, "forward")
        self.assertEqual(INTERNAL_BODY_FLU.y_axis, "left")
        self.assertEqual(INTERNAL_BODY_FLU.z_axis, "up")

        self.assertEqual(OPENCV_CAMERA_OPTICAL.x_axis, "right")
        self.assertEqual(OPENCV_CAMERA_OPTICAL.y_axis, "down")
        self.assertEqual(OPENCV_CAMERA_OPTICAL.z_axis, "forward")

    def test_official_intrinsics_distortion_and_translation(self):
        config = RuntimeCompetitionConfig()

        np.testing.assert_allclose(
            official_camera_matrix(config),
            np.array(
                [
                    [320.0, 0.0, 320.0],
                    [0.0, 320.0, 180.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            ),
        )
        np.testing.assert_allclose(official_dist_coeffs(config), np.zeros(5))
        np.testing.assert_allclose(camera_translation_body_frd(config), np.zeros(3))

    def test_level_camera_body_mapping_matches_opencv_optical_axes(self):
        rot_camera_body = mavlink_body_frd_to_opencv_camera_rotmat(0.0)

        np.testing.assert_allclose(
            rot_camera_body @ np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
        np.testing.assert_allclose(
            rot_camera_body @ np.array([0.0, 1.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(
            rot_camera_body @ np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
        )

    def test_official_tilt_rotations_are_inverse_and_orthonormal(self):
        config = RuntimeCompetitionConfig()
        rot_body_camera = official_camera_to_body_frd_rotmat(config)
        rot_camera_body = official_body_frd_to_camera_rotmat(config)

        np.testing.assert_allclose(rot_body_camera @ rot_camera_body, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(rot_body_camera.T, rot_camera_body, atol=1e-12)
        np.testing.assert_allclose(rot_body_camera @ rot_body_camera.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(rot_body_camera)), 1.0, places=12)

    def test_official_camera_to_internal_body_flu_is_proper_rotation(self):
        config = RuntimeCompetitionConfig()
        rot_body_camera = official_camera_to_internal_body_flu_rotmat(config)

        np.testing.assert_allclose(
            rot_body_camera,
            body_frd_to_internal_body_flu_rotmat()
            @ official_camera_to_body_frd_rotmat(config),
            atol=1e-12,
        )
        np.testing.assert_allclose(rot_body_camera @ rot_body_camera.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(rot_body_camera)), 1.0, places=12)

    def test_official_camera_optical_axis_points_up_relative_to_body_forward(self):
        config = RuntimeCompetitionConfig()
        rot_body_camera = official_camera_to_body_frd_rotmat(config)
        optical_axis_in_body = rot_body_camera @ np.array([0.0, 0.0, 1.0])
        expected = np.array(
            [
                math.cos(math.radians(20.0)),
                0.0,
                -math.sin(math.radians(20.0)),
            ],
            dtype=float,
        )

        np.testing.assert_allclose(optical_axis_in_body, expected, atol=1e-12)
        self.assertLess(optical_axis_in_body[2], 0.0)

    def test_point_on_official_optical_axis_projects_to_principal_point(self):
        config = RuntimeCompetitionConfig()
        distance_m = 10.0
        point_body = official_camera_to_body_frd_rotmat(config) @ np.array(
            [0.0, 0.0, distance_m]
        )

        pixel = project_body_frd_point_to_pixel(point_body, config=config)

        np.testing.assert_allclose(pixel, np.array([320.0, 180.0]), atol=1e-12)

    def test_projection_sanity_for_straight_ahead_point_catches_tilt_sign(self):
        config = RuntimeCompetitionConfig()
        point_straight_ahead_body = np.array([10.0, 0.0, 0.0])

        pixel = project_body_frd_point_to_pixel(point_straight_ahead_body, config=config)

        expected_v = 180.0 + 320.0 * math.tan(math.radians(20.0))
        self.assertAlmostEqual(pixel[0], 320.0, places=12)
        self.assertAlmostEqual(pixel[1], expected_v, places=12)
        self.assertGreater(pixel[1], config.camera_cy_px)

        inverted = opencv_camera_to_mavlink_body_frd_rotmat(-config.camera_tilt_up_rad)
        inverted_point_camera = inverted.T @ point_straight_ahead_body
        inverted_v = (
            config.camera_fy_px * inverted_point_camera[1] / inverted_point_camera[2]
            + config.camera_cy_px
        )
        self.assertLess(inverted_v, config.camera_cy_px)

    def test_tilted_phase9b_sample_maps_to_positive_internal_z(self):
        config = RuntimeCompetitionConfig()
        tilted_camera_tvec = np.array(
            [0.08145763507435748, 1.640968327338819, 7.513559788178706],
            dtype=float,
        )

        body_flu = official_camera_to_internal_body_flu_rotmat(config) @ tilted_camera_tvec

        self.assertGreater(body_flu[0], 0.0)
        self.assertGreater(body_flu[2], 0.0)
        np.testing.assert_allclose(
            body_flu,
            np.array([7.621680911294569, -0.08145763507435748, 1.027782967495179]),
            atol=1e-12,
        )

    def test_body_point_conversion_uses_official_tilt_without_translation(self):
        config = RuntimeCompetitionConfig()
        point_body = np.array([10.0, 0.0, 0.0])
        point_camera = body_frd_point_to_camera_optical(point_body, config=config)

        np.testing.assert_allclose(
            point_camera,
            np.array(
                [
                    0.0,
                    10.0 * math.sin(math.radians(20.0)),
                    10.0 * math.cos(math.radians(20.0)),
                ],
                dtype=float,
            ),
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
