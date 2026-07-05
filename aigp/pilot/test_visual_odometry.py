from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from runtime_config import load_runtime_config
from vehicle_state_estimator import VehicleStateEstimator
from visual_odometry import GateKeypointVisualOdometry


OBJECT_POINTS = np.array(
    [
        [-0.75, 0.75, 0.0],
        [0.75, 0.75, 0.0],
        [0.75, -0.75, 0.0],
        [-0.75, -0.75, 0.0],
    ],
    dtype=float,
)

KEYPOINTS_PX = np.array(
    [
        [290.0, 220.0],
        [350.0, 220.0],
        [350.0, 280.0],
        [290.0, 280.0],
    ],
    dtype=float,
)


def _config(**vo_overrides):
    config = load_runtime_config()
    visual_odometry = replace(
        config.visual_odometry,
        min_keypoint_conf=0.0,
        min_features=4,
        min_dt_s=0.01,
        max_dt_s=2.0,
        max_visual_speed_m_s=20.0,
        max_velocity_innovation_m_s=20.0,
        position_alpha_xy=0.0,
        position_alpha_z=0.0,
        **vo_overrides,
    )
    state_estimation = replace(
        config.state_estimation,
        mode="estimator",
        use_imu_prediction=False,
        use_vision_correction=True,
        vision_correction_source="stable_tracks",
        vision_correction_velocity_alpha_xy=1.0,
        vision_correction_velocity_alpha_z=1.0,
        vision_correction_bias_alpha_xy=0.0,
        vision_correction_bias_alpha_z=0.0,
        vision_correction_max_velocity_delta_m_s=20.0,
    )
    return replace(
        config,
        state_estimation=state_estimation,
        visual_odometry=visual_odometry,
    )


def _perception(tvec, *, timestamp):
    return {
        "frame_id": int(round(timestamp * 100.0)),
        "perception_wall_time": float(timestamp),
        "camera_to_body": np.eye(3, dtype=float),
        "camera_translation_body": np.zeros(3, dtype=float),
        "detections": [
            {
                "keypoints_px": KEYPOINTS_PX.copy(),
                "keypoint_conf": np.ones(4, dtype=float),
                "object_points_m": OBJECT_POINTS.copy(),
                "rvec": np.zeros(3, dtype=float),
                "tvec": np.asarray(tvec, dtype=float),
                "gate_center_camera": np.asarray(tvec, dtype=float),
                "confidence": 1.0,
                "reprojection_error": 0.0,
            }
        ],
    }


def _snapshot(perception):
    return SimpleNamespace(
        roll_rad=0.0,
        pitch_rad=0.0,
        yaw_rad=0.0,
        latest_perception=perception,
        stable_gate_landmarks_neu=[],
    )


def test_gate_keypoint_vo_recovers_metric_velocity_from_corner_motion():
    vo = GateKeypointVisualOdometry(_config())
    first = _perception([10.0, 0.0, 3.0], timestamp=1.0)
    second = _perception([9.0, 0.0, 3.0], timestamp=2.0)

    first_measurement = vo.update(
        first,
        roll_rad=0.0,
        pitch_rad=0.0,
        yaw_rad=0.0,
        estimated_pos_neu=np.zeros(3),
        estimated_vel_neu=np.zeros(3),
        now=1.0,
    )
    second_measurement = vo.update(
        second,
        roll_rad=0.0,
        pitch_rad=0.0,
        yaw_rad=0.0,
        estimated_pos_neu=np.zeros(3),
        estimated_vel_neu=np.zeros(3),
        now=2.0,
    )

    assert not first_measurement.valid
    assert first_measurement.reason == "no_reference"
    assert second_measurement.valid
    np.testing.assert_allclose(second_measurement.delta_neu, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(
        second_measurement.velocity_neu,
        np.array([1.0, 0.0, 0.0]),
    )
    assert second_measurement.feature_count == 4


def test_estimator_uses_temporal_vo_when_stable_landmarks_are_unavailable():
    estimator = VehicleStateEstimator(_config(), mode_override="estimator")
    estimator.initialized = True
    estimator.pos_neu = np.zeros(3, dtype=float)
    estimator.vel_neu = np.zeros(3, dtype=float)

    first = _snapshot(_perception([10.0, 0.0, 3.0], timestamp=1.0))
    second = _snapshot(_perception([9.0, 0.0, 3.0], timestamp=2.0))

    estimator._correct_with_vision(first, now=1.0)
    correction = estimator._correct_with_vision(second, now=2.0)

    assert correction["source"] == "gate_keypoint_vo"
    np.testing.assert_allclose(
        correction["visual_velocity_neu"],
        np.array([1.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(estimator.vel_neu, np.array([1.0, 0.0, 0.0]))
