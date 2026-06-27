import math
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
