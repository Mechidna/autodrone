import cv2
import numpy as np

from autonomy_core.core.competition_config import (
    VADR_TS_002,
    planar_square_object_points_m,
)
from autonomy_core.core.frame_conventions import (
    camera_translation_body_frd,
    official_body_frd_to_camera_rotmat,
    official_camera_matrix,
    official_camera_to_body_frd_rotmat,
    official_dist_coeffs,
)
from autonomy_core.perception.gate_perception_yolo import (
    GatePerception,
    KEYPOINT_LAYOUT_INNER4_OUTER4,
    default_live_pnp_corner_reordering,
    object_points_for_keypoint_layout,
)


def _semantic_yolo_perception_without_model() -> GatePerception:
    perception = GatePerception.__new__(GatePerception)
    perception.gate_size = VADR_TS_002.gate_inner_square_m
    perception.model_points = np.asarray(
        planar_square_object_points_m(VADR_TS_002.gate_inner_square_m),
        dtype=np.float32,
    )
    perception.corners_are_semantic = True
    perception.allow_pnp_corner_reordering = default_live_pnp_corner_reordering(
        perception.corners_are_semantic
    )
    perception.keypoint_layout = "inner4"
    perception.keypoint_count = 4
    return perception


def _semantic_yolo_8k_perception_without_model() -> GatePerception:
    perception = GatePerception.__new__(GatePerception)
    perception.gate_size = VADR_TS_002.gate_inner_square_m
    perception.keypoint_layout = KEYPOINT_LAYOUT_INNER4_OUTER4
    perception.model_points = object_points_for_keypoint_layout(
        KEYPOINT_LAYOUT_INNER4_OUTER4,
        inner_gate_size_m=VADR_TS_002.gate_inner_square_m,
        outer_gate_size_m=VADR_TS_002.gate_outer_square_m,
        gate_depth_m=VADR_TS_002.gate_depth_m,
    )
    perception.keypoint_count = int(perception.model_points.shape[0])
    perception.corners_are_semantic = True
    perception.allow_pnp_corner_reordering = default_live_pnp_corner_reordering(
        perception.corners_are_semantic
    )
    return perception


def _gate_object_to_body_frd_rotmat() -> np.ndarray:
    object_x_body = np.array([0.0, 1.0, 0.0], dtype=float)
    object_y_body = np.array([0.0, 0.0, -1.0], dtype=float)
    object_z_body = np.cross(object_x_body, object_y_body)
    return np.column_stack((object_x_body, object_y_body, object_z_body))


def _project_gate_from_body_center(gate_center_body_frd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    object_points = np.asarray(
        planar_square_object_points_m(VADR_TS_002.gate_inner_square_m),
        dtype=np.float32,
    )
    body_to_camera = official_body_frd_to_camera_rotmat(VADR_TS_002)
    object_to_camera = body_to_camera @ _gate_object_to_body_frd_rotmat()
    rvec, _ = cv2.Rodrigues(object_to_camera)

    gate_center_body_frd = np.asarray(gate_center_body_frd, dtype=float).reshape(3)
    tvec = body_to_camera @ (
        gate_center_body_frd - camera_translation_body_frd(VADR_TS_002)
    )
    image_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec.reshape(3, 1),
        official_camera_matrix(VADR_TS_002),
        official_dist_coeffs(VADR_TS_002),
    )
    return image_points.reshape(4, 2).astype(np.float32), tvec


def test_semantic_yolo_defaults_to_semantic_pnp_order_only():
    perception = _semantic_yolo_perception_without_model()

    assert perception.corners_are_semantic
    assert not perception.allow_pnp_corner_reordering
    assert not default_live_pnp_corner_reordering(corners_are_semantic=True)

    image_points, _ = _project_gate_from_body_center(
        np.array([8.0, 0.0, -1.5], dtype=float)
    )
    _, _, debug = perception.estimate_pose(
        image_points,
        official_camera_matrix(VADR_TS_002),
        official_dist_coeffs(VADR_TS_002),
    )

    assert debug["selected_order"] == "tl_tr_br_bl"
    assert debug["live_candidate_orders_allowed"] == "tl_tr_br_bl"
    assert not debug["allow_pnp_corner_reordering"]


def test_inner4_outer4_object_points_match_mixed_depth_layout():
    object_points = object_points_for_keypoint_layout(KEYPOINT_LAYOUT_INNER4_OUTER4)

    assert object_points.shape == (8, 3)
    np.testing.assert_allclose(
        object_points[:4, :2],
        np.asarray(VADR_TS_002.gate_inner_object_points_m, dtype=float)[:, :2],
    )
    np.testing.assert_allclose(
        object_points[:4, 2],
        np.full(4, 0.5 * VADR_TS_002.gate_depth_m),
    )
    np.testing.assert_allclose(
        np.abs(object_points[4:, :2]),
        np.full((4, 2), 0.5 * VADR_TS_002.gate_outer_square_m),
    )
    np.testing.assert_allclose(
        object_points[4:, 2],
        np.full(4, -0.5 * VADR_TS_002.gate_depth_m),
    )


def test_inner4_outer4_pnp_round_trip_recovers_center_plane_tvec():
    perception = _semantic_yolo_8k_perception_without_model()
    camera_matrix = official_camera_matrix(VADR_TS_002)
    dist_coeffs = official_dist_coeffs(VADR_TS_002)
    rvec = np.array([0.04, -0.08, 0.02], dtype=float).reshape(3, 1)
    expected_tvec = np.array([0.25, -0.12, 8.0], dtype=float)

    image_points, _ = cv2.projectPoints(
        perception.model_points,
        rvec,
        expected_tvec.reshape(3, 1),
        camera_matrix,
        dist_coeffs,
    )
    _R_mat, tvec, debug = perception.estimate_pose(
        image_points.reshape(8, 2).astype(np.float32),
        camera_matrix,
        dist_coeffs,
    )

    np.testing.assert_allclose(np.asarray(tvec).reshape(3), expected_tvec, atol=1e-3)
    assert debug["selected_order"] == "tl_tr_br_bl"
    assert debug["selected_solver"] in ("SQPNP", "ITERATIVE", "EPNP")
    assert np.asarray(debug["ordered_points"]).shape == (8, 2)


def test_inner4_outer4_image_order_permutation_applies_to_both_keypoint_groups():
    perception = _semantic_yolo_8k_perception_without_model()
    points = np.arange(16, dtype=np.float32).reshape(8, 2)

    ordered = perception.apply_pnp_order(points, [1, 2, 3, 0])

    np.testing.assert_array_equal(ordered[:4], points[:4][[1, 2, 3, 0]])
    np.testing.assert_array_equal(ordered[4:], points[4:][[1, 2, 3, 0]])


def test_spec_camera_transform_and_pnp_round_trip_gate_center():
    perception = _semantic_yolo_perception_without_model()
    camera_to_body = official_camera_to_body_frd_rotmat(VADR_TS_002)

    np.testing.assert_allclose(
        camera_to_body @ np.array([0.0, 0.0, 1.0], dtype=float),
        np.array(
            [
                np.cos(VADR_TS_002.camera_tilt_up_rad),
                0.0,
                -np.sin(VADR_TS_002.camera_tilt_up_rad),
            ],
            dtype=float,
        ),
        atol=1e-12,
    )

    gate_centers_body_frd = [
        np.array([8.0, 0.0, -1.5], dtype=float),
        np.array([12.0, 0.8, -1.2], dtype=float),
    ]

    for expected_body_center in gate_centers_body_frd:
        image_points, expected_tvec = _project_gate_from_body_center(expected_body_center)
        _, tvec, debug = perception.estimate_pose(
            image_points,
            official_camera_matrix(VADR_TS_002),
            official_dist_coeffs(VADR_TS_002),
        )

        recovered_tvec = np.asarray(tvec, dtype=float).reshape(3)
        recovered_body_center = (
            camera_translation_body_frd(VADR_TS_002)
            + camera_to_body @ recovered_tvec
        )

        np.testing.assert_allclose(recovered_tvec, expected_tvec, atol=0.05)
        np.testing.assert_allclose(
            recovered_body_center,
            expected_body_center,
            atol=0.05,
            err_msg=f"selected_order={debug['selected_order']}",
        )
