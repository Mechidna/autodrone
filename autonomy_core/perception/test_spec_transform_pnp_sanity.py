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
    default_live_pnp_corner_reordering,
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
