import math
import time
from types import SimpleNamespace

import numpy as np

from autonomy_core.core.competition_config import VADR_TS_002
from autonomy_core.core.frame_conventions import (
    body_frd_to_local_ned_rotmat,
    camera_translation_body_frd,
    local_neu_to_ned,
    official_camera_matrix,
    official_camera_to_body_frd_rotmat,
)
from perception_geometry_audit import PerceptionGeometryAudit


def _audit() -> PerceptionGeometryAudit:
    return PerceptionGeometryAudit(
        SimpleNamespace(
            enabled=True,
            print_period_s=0.0,
            max_prints=10,
            max_match_distance_m=5.0,
            reference_pose_source="both",
            gazebo_rotation_mode="transpose",
            gazebo_optical_mode="physical_minus_y",
            known_gate_positions_neu=((0.0, 8.0, 1.5),),
            gate_right_axis_neu=(1.0, 0.0, 0.0),
            gate_up_axis_neu=(0.0, 0.0, 1.0),
        )
    )


def _exact_detection(audit: PerceptionGeometryAudit) -> tuple[dict, dict]:
    known_gate_neu = np.array([0.0, 8.0, 1.5], dtype=float)
    drone_pos_ned = np.zeros(3, dtype=float)
    drone_rpy_rad = np.array([0.0, 0.0, math.pi / 2.0], dtype=float)
    camera_matrix = official_camera_matrix(VADR_TS_002)
    camera_to_body = official_camera_to_body_frd_rotmat(VADR_TS_002)
    camera_translation = camera_translation_body_frd(VADR_TS_002)
    object_points = np.asarray(VADR_TS_002.gate_inner_object_points_m, dtype=float)

    rot_ned_body = body_frd_to_local_ned_rotmat(*drone_rpy_rad)
    expected_body = rot_ned_body.T @ (local_neu_to_ned(known_gate_neu) - drone_pos_ned)
    expected_camera = camera_to_body.T @ (expected_body - camera_translation)
    expected_keypoints = audit.project_known_gate_keypoints(
        known_gate_neu=known_gate_neu,
        drone_pos_ned=drone_pos_ned,
        drone_rpy_rad=drone_rpy_rad,
        camera_matrix=camera_matrix,
        camera_to_body=camera_to_body,
        camera_translation_body=camera_translation,
        object_points_m=object_points,
    )

    detection = {
        "gate_center_world": known_gate_neu.copy(),
        "gate_center_camera": expected_camera.copy(),
        "gate_center_body_frd": camera_to_body @ expected_camera,
        "yolo_keypoints": np.column_stack((expected_keypoints, np.ones(4))),
        "reprojection_error": 0.0,
    }
    context = {
        "known_gate_neu": known_gate_neu,
        "drone_pos_ned": drone_pos_ned,
        "drone_rpy_rad": drone_rpy_rad,
        "camera_matrix": camera_matrix,
        "camera_to_body": camera_to_body,
        "camera_translation": camera_translation,
        "object_points": object_points,
    }
    return detection, context


def test_geometry_audit_reports_zero_residual_for_exact_spec_projection():
    audit = _audit()
    detection, ctx = _exact_detection(audit)

    result = audit.evaluate_detection(
        detection,
        detection_index=0,
        drone_pos_ned=ctx["drone_pos_ned"],
        drone_rpy_rad=ctx["drone_rpy_rad"],
        camera_matrix=ctx["camera_matrix"],
        camera_to_body=ctx["camera_to_body"],
        camera_translation_body=ctx["camera_translation"],
        object_points_m=ctx["object_points"],
        frame_id=42,
    )

    assert result is not None
    assert result["gate_index"] == 0
    assert result["world_error_norm_m"] < 1e-9
    np.testing.assert_allclose(result["body_error_m"], np.zeros(3), atol=1e-9)
    np.testing.assert_allclose(result["camera_error_m"], np.zeros(3), atol=1e-9)
    np.testing.assert_allclose(result["keypoint_center_error_px"], np.zeros(2), atol=1e-9)
    assert result["keypoint_rmse_px"] < 1e-9


def test_geometry_audit_prints_compact_line(capsys):
    audit = _audit()
    detection, ctx = _exact_detection(audit)

    audit.maybe_print(
        [detection],
        drone_pos_ned=ctx["drone_pos_ned"],
        drone_rpy_rad=ctx["drone_rpy_rad"],
        camera_matrix=ctx["camera_matrix"],
        camera_to_body=ctx["camera_to_body"],
        camera_translation_body=ctx["camera_translation"],
        object_points_m=ctx["object_points"],
        frame_id=42,
    )

    out = capsys.readouterr().out
    assert "[GEOM_AUDIT]" in out
    assert "gate=1" in out
    assert "frame=42" in out
    assert "world_norm=0.00" in out


def test_geometry_audit_gazebo_pose_round_trip():
    audit = _audit()
    known_gate_neu = np.array([0.0, 8.0, 1.5], dtype=float)
    wall_now = time.time()
    gazebo_pose = {
        "gazebo_model_pos_world": np.zeros(3, dtype=float),
        "gazebo_model_quat_world": np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
        "gazebo_camera_pos_world": np.zeros(3, dtype=float),
        "gazebo_camera_quat_world": np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
        "gazebo_pose_ros_stamp_sec": 12,
        "gazebo_pose_ros_stamp_nanosec": 300_000_000,
        "gazebo_pose_wall_time": wall_now - 0.25,
        "gazebo_pose_selection_method": "unit_test",
    }
    gate_camera = np.array([0.0, -1.5, 8.0], dtype=float)

    detection = {
        "gate_center_world": known_gate_neu.copy(),
        "gate_center_camera": gate_camera.copy(),
        "gate_center_body_frd": gate_camera.copy(),
        "yolo_keypoints": np.column_stack((
            audit.project_known_gate_keypoints_gazebo(
                known_gate_neu=known_gate_neu,
                gazebo_pose=gazebo_pose,
                camera_matrix=official_camera_matrix(VADR_TS_002),
                object_points_m=np.asarray(
                    VADR_TS_002.gate_inner_object_points_m,
                    dtype=float,
                ),
            ),
            np.ones(4),
        )),
        "reprojection_error": 0.0,
    }

    result = audit.evaluate_detection(
        detection,
        detection_index=0,
        drone_pos_ned=np.zeros(3, dtype=float),
        drone_rpy_rad=np.zeros(3, dtype=float),
        camera_matrix=official_camera_matrix(VADR_TS_002),
        camera_to_body=np.eye(3),
        camera_translation_body=np.zeros(3, dtype=float),
        object_points_m=np.asarray(VADR_TS_002.gate_inner_object_points_m, dtype=float),
        frame_id=7,
        gazebo_pose=gazebo_pose,
        image_wall_time=wall_now,
        image_ros_stamp_sec=12,
        image_ros_stamp_nanosec=500_000_000,
        attitude_wall_time=wall_now - 0.05,
        position_wall_time=wall_now - 0.10,
    )

    assert result is not None
    np.testing.assert_allclose(
        result["world_neu_from_tvec_gazebo_debug"],
        known_gate_neu,
        atol=1e-9,
    )
    np.testing.assert_allclose(result["expected_camera_gazebo_m"], gate_camera, atol=1e-9)
    assert result["world_error_norm_gazebo_m"] < 1e-9
    assert math.isclose(result["runtime_yaw_used_by_perception_deg"], 0.0, abs_tol=1e-9)
    assert math.isclose(result["gazebo_model_yaw_deg"], 90.0, abs_tol=1e-9)
    assert math.isclose(result["gazebo_camera_yaw_deg"], 90.0, abs_tol=1e-9)
    assert math.isclose(result["yaw_runtime_minus_gazebo_deg"], -90.0, abs_tol=1e-9)
    assert math.isclose(result["image_minus_gazebo_pose_ros_dt_s"], 0.2, abs_tol=1e-9)
    assert math.isclose(result["image_minus_gazebo_pose_wall_dt_s"], 0.25, abs_tol=1e-6)
    assert math.isclose(result["image_minus_runtime_pose_wall_dt_s"], 0.10, abs_tol=1e-6)
