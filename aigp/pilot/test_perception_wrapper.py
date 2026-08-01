from dataclasses import replace
import math
from types import SimpleNamespace
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from autonomy_core.core.competition_config import VADR_TS_002
from autonomy_core.core.frame_conventions import (
    body_frd_to_local_ned_rotmat,
    official_camera_to_body_frd_rotmat,
)
from perception_wrapper import (
    PITCH_INVERTED_MODE,
    PerceptionWrapper,
    X_MIRROR_YAW_CORRECTED_MODE,
    X_MIRROR_YAW_CORRECTION_DEG,
    _body_frd_yaw_rotmat,
)
from runtime_config import load_runtime_config
from autonomy_core.perception.gate_perception_yolo import (
    KEYPOINT_LAYOUT_INNER4_OUTER4,
    object_points_for_keypoint_layout,
)


def _wrapper(
    transform_mode: str | None = None,
    depth_correction_m: float = 0.0,
    depth_correction_per_m: float = 0.0,
    depth_correction_max_m: float = 0.0,
) -> PerceptionWrapper:
    base_config = load_runtime_config()
    config = replace(
        base_config,
        perception=replace(
            base_config.perception,
            transform_mode=(
                transform_mode
                if transform_mode is not None
                else base_config.perception.transform_mode
            ),
            depth_correction_m=depth_correction_m,
            depth_correction_per_m=depth_correction_per_m,
            depth_correction_max_m=depth_correction_max_m,
        ),
    )
    return PerceptionWrapper(
        config=config,
        gate_perception=SimpleNamespace(
            model_points=np.asarray(
                VADR_TS_002.gate_inner_object_points_m,
                dtype=float,
            )
        )
    )


def test_normalize_detection_preserves_gate_normals():
    wrapper = _wrapper()
    detection = {
        "gate_center_camera": np.array([0.0, 0.0, 10.0], dtype=float),
        "gate_center_world": np.array([1.0, 2.0, 3.0], dtype=float),
        "gate_normal_camera": np.array([0.0, 0.0, 1.0], dtype=float),
        "gate_normal_world": np.array([0.0, 1.0, 0.0], dtype=float),
        "gate_normal_world_ned": np.array([0.0, 1.0, -0.0], dtype=float),
        "confidence": 0.9,
    }

    normalized = wrapper._normalize_detection(detection, 0)

    np.testing.assert_allclose(normalized["gate_normal_camera"], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(
        normalized["gate_normal_body"],
        wrapper.camera_to_body @ np.array([0.0, 0.0, 1.0], dtype=float),
    )
    np.testing.assert_allclose(normalized["gate_normal_body_frd"], normalized["gate_normal_body"])
    np.testing.assert_allclose(normalized["gate_normal_world"], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(normalized["gate_normal_world_ned"], [0.0, 1.0, 0.0])


def test_normalize_detection_preserves_inner4_outer4_keypoints():
    object_points = object_points_for_keypoint_layout(KEYPOINT_LAYOUT_INNER4_OUTER4)
    wrapper = PerceptionWrapper(
        gate_perception=SimpleNamespace(model_points=object_points)
    )
    yolo_keypoints = np.column_stack(
        [
            np.arange(8, dtype=float),
            np.arange(10, 18, dtype=float),
            np.linspace(0.5, 0.9, 8, dtype=float),
        ]
    )
    detection = {
        "gate_center_camera": np.array([0.0, 0.0, 10.0], dtype=float),
        "yolo_keypoints": yolo_keypoints,
        "confidence": 0.9,
    }

    normalized = wrapper._normalize_detection(detection, 0)

    assert normalized["keypoints_px"].shape == (8, 2)
    assert normalized["keypoint_conf"].shape == (8,)
    assert normalized["object_points_m"].shape == (8, 3)
    np.testing.assert_allclose(normalized["keypoints_px"], yolo_keypoints[:, :2])
    np.testing.assert_allclose(normalized["keypoint_conf"], yolo_keypoints[:, 2])


def test_depth_correction_changes_projection_but_preserves_raw_tvec():
    wrapper = _wrapper(
        "competition_official_ned",
        depth_correction_m=1.5,
    )
    gate_camera_raw = np.array([2.0, 3.0, 10.0], dtype=float)
    gate_camera_corrected = np.array([2.0, 3.0, 11.5], dtype=float)
    drone_pos_ned = np.array([1.0, -2.0, 0.5], dtype=float)
    drone_rpy_rad = np.zeros(3, dtype=float)

    projected = wrapper._project_detection_to_world(
        {
            "gate_center_camera": gate_camera_raw,
            "tvec": gate_camera_raw,
        },
        drone_pos_ned=drone_pos_ned,
        drone_rpy_rad=drone_rpy_rad,
    )
    normalized = wrapper._normalize_detection(projected, 0)

    expected_body = wrapper.camera_to_body @ gate_camera_corrected
    expected_world_ned = drone_pos_ned + expected_body
    np.testing.assert_allclose(projected["gate_center_camera"], gate_camera_raw)
    np.testing.assert_allclose(
        projected["gate_center_camera_corrected"],
        gate_camera_corrected,
    )
    np.testing.assert_allclose(projected["gate_center_body_frd"], expected_body)
    np.testing.assert_allclose(
        projected["gate_center_world_ned"],
        expected_world_ned,
    )
    np.testing.assert_allclose(normalized["tvec"], gate_camera_raw)
    np.testing.assert_allclose(
        normalized["tvec_corrected"],
        gate_camera_corrected,
    )
    assert normalized["depth_correction_m"] == 1.5


def test_depth_correction_scales_with_raw_depth_and_clamps():
    wrapper = _wrapper(
        "competition_official_ned",
        depth_correction_per_m=0.02,
        depth_correction_max_m=3.0,
    )

    raw_50m = np.array([2.0, 3.0, 50.0], dtype=float)
    raw_100m = np.array([2.0, 3.0, 100.0], dtype=float)
    raw_200m = np.array([2.0, 3.0, 200.0], dtype=float)

    np.testing.assert_allclose(
        wrapper._camera_with_depth_correction(raw_50m),
        [2.0, 3.0, 51.0],
    )
    np.testing.assert_allclose(
        wrapper._camera_with_depth_correction(raw_100m),
        [2.0, 3.0, 102.0],
    )
    np.testing.assert_allclose(
        wrapper._camera_with_depth_correction(raw_200m),
        [2.0, 3.0, 203.0],
    )

    normalized = wrapper._normalize_detection(
        {
            "gate_center_camera": raw_100m,
        },
        0,
    )
    np.testing.assert_allclose(normalized["tvec"], raw_100m)
    np.testing.assert_allclose(
        normalized["tvec_corrected"],
        [2.0, 3.0, 102.0],
    )
    assert normalized["depth_correction_m"] == 2.0


def test_camera_to_body_respects_x_mirror_transform_mode():
    standard = _wrapper("competition_official_ned")
    mirrored = _wrapper("physical_direct_rad_x_mirror")
    official = official_camera_to_body_frd_rotmat(VADR_TS_002)
    camera_x_mirror = np.diag([-1.0, 1.0, 1.0])

    np.testing.assert_allclose(standard.camera_to_body, official)
    np.testing.assert_allclose(
        mirrored.camera_to_body,
        official @ camera_x_mirror,
    )
    np.testing.assert_allclose(
        mirrored.camera_to_body[:, 0],
        -standard.camera_to_body[:, 0],
    )
    np.testing.assert_allclose(
        mirrored.camera_to_body[:, 1:],
        standard.camera_to_body[:, 1:],
    )


def test_camera_to_body_respects_x_mirror_yaw_corrected_transform_mode():
    corrected = _wrapper(X_MIRROR_YAW_CORRECTED_MODE)
    official = official_camera_to_body_frd_rotmat(VADR_TS_002)
    correction_rad = math.radians(X_MIRROR_YAW_CORRECTION_DEG)
    expected = (
        _body_frd_yaw_rotmat(correction_rad)
        @ official
        @ np.diag([-1.0, 1.0, 1.0])
    )

    np.testing.assert_allclose(corrected.camera_to_body, expected)
    np.testing.assert_allclose(
        corrected.camera_to_body[:, 0],
        [math.sin(correction_rad), -math.cos(correction_rad), 0.0],
    )
    np.testing.assert_allclose(
        corrected.camera_to_body[:, 2],
        [
            math.cos(correction_rad) * math.cos(math.radians(20.0)),
            math.sin(correction_rad) * math.cos(math.radians(20.0)),
            -math.sin(math.radians(20.0)),
        ],
    )
    np.testing.assert_allclose(
        corrected.camera_to_body.T @ corrected.camera_to_body,
        np.eye(3),
        atol=1e-12,
    )
    np.testing.assert_allclose(np.linalg.det(corrected.camera_to_body), -1.0)
    assert corrected.transform_yaw_correction_deg == X_MIRROR_YAW_CORRECTION_DEG


def test_x_mirror_yaw_corrected_rotates_positive_course_lean_negative():
    mirrored = _wrapper("physical_direct_rad_x_mirror")
    corrected = _wrapper(X_MIRROR_YAW_CORRECTED_MODE)
    drone_pos_ned = np.zeros(3, dtype=float)
    drone_rpy_rad = np.array([0.0, 0.0, -math.pi], dtype=float)
    body_to_world = body_frd_to_local_ned_rotmat(*drone_rpy_rad)

    mirrored_gate_world_ned = np.array([-44.13, 2.46, -1.5], dtype=float)
    mirrored_gate_body_frd = body_to_world.T @ mirrored_gate_world_ned
    gate_camera_mirrored = mirrored.camera_to_body.T @ mirrored_gate_body_frd

    projected = corrected._project_detection_to_world(
        {"gate_center_camera": gate_camera_mirrored},
        drone_pos_ned=drone_pos_ned,
        drone_rpy_rad=drone_rpy_rad,
    )

    assert projected["gate_center_world_ned"][1] < 0.0
    np.testing.assert_allclose(
        projected["gate_center_world_ned"][:2],
        [-44.150553, -2.058443],
        atol=1e-6,
    )


def test_x_mirror_keeps_stationary_gate_fixed_while_yaw_changes():
    wrapper = _wrapper("physical_direct_rad_x_mirror")
    expected_camera_to_body = (
        official_camera_to_body_frd_rotmat(VADR_TS_002)
        @ np.diag([-1.0, 1.0, 1.0])
    )
    drone_pos_ned = np.array([1.0, -0.5, 0.2], dtype=float)
    gate_world_ned = np.array([21.0, 1.5, -1.0], dtype=float)

    for yaw_rad in (0.0, 0.25):
        drone_rpy_rad = np.array([0.0, 0.0, yaw_rad], dtype=float)
        body_to_world = body_frd_to_local_ned_rotmat(*drone_rpy_rad)
        gate_body_frd = (
            body_to_world.T @ (gate_world_ned - drone_pos_ned)
            - wrapper.camera_translation_body
        )
        gate_camera_mirrored = expected_camera_to_body.T @ gate_body_frd

        projected = wrapper._project_detection_to_world(
            {"gate_center_camera": gate_camera_mirrored},
            drone_pos_ned=drone_pos_ned,
            drone_rpy_rad=drone_rpy_rad,
        )

        np.testing.assert_allclose(
            projected["gate_center_world_ned"],
            gate_world_ned,
            atol=1e-9,
        )


def test_pitch_inverted_mode_changes_only_perception_pitch():
    standard = _wrapper("physical_direct_rad")
    corrected = _wrapper(PITCH_INVERTED_MODE)
    raw_rpy = np.array([0.12, math.radians(17.8), -3.05], dtype=float)

    np.testing.assert_allclose(standard._perception_rpy(raw_rpy), raw_rpy)
    np.testing.assert_allclose(
        corrected._perception_rpy(raw_rpy),
        [raw_rpy[0], -raw_rpy[1], raw_rpy[2]],
    )
    np.testing.assert_allclose(
        raw_rpy,
        [0.12, math.radians(17.8), -3.05],
    )


def test_pitch_inverted_mode_projects_startup_gate_with_negative_pitch():
    wrapper = _wrapper(PITCH_INVERTED_MODE)
    raw_rpy = np.array(
        [0.0, math.radians(17.8), math.radians(-179.9)],
        dtype=float,
    )
    drone_pos_ned = np.array([-0.01, 0.0, 0.0], dtype=float)
    gate_body_frd = np.array([20.969, 0.389, -7.930], dtype=float)
    gate_camera = wrapper.camera_to_body.T @ gate_body_frd

    projected = wrapper._project_detections_to_world(
        [{"gate_center_camera": gate_camera}],
        drone_pos_ned=drone_pos_ned,
        drone_rpy_rad=raw_rpy,
    )[0]

    np.testing.assert_allclose(
        projected["drone_rpy_rad_mavlink"],
        raw_rpy,
    )
    np.testing.assert_allclose(
        projected["drone_rpy_rad_used"],
        [raw_rpy[0], -raw_rpy[1], raw_rpy[2]],
    )
    np.testing.assert_allclose(
        projected["gate_center_world"],
        [-22.398, -0.428, 1.140],
        atol=2e-3,
    )
