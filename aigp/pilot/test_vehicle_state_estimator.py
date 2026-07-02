from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from runtime_config import load_runtime_config
from vehicle_state_estimator import VehicleStateEstimator


def _config(**overrides):
    config = load_runtime_config()
    values = {
        "mode": "estimator",
        "use_imu_prediction": False,
        "use_vision_correction": True,
        "vision_correction_source": "stable_tracks",
        "vision_correction_alpha": 1.0,
        "vision_correction_alpha_xy": 1.0,
        "vision_correction_alpha_z": 0.0,
        "vision_correction_max_delta_m": 10.0,
        "vision_correction_max_residual_m": 3.0,
        "vision_correction_min_confidence": 0.0,
        "vision_correction_min_measurements": 2,
        "vision_correction_single_landmark_max_residual_m": 0.35,
        "vision_correction_max_avg_residual_m": 0.80,
    }
    values.update(overrides)
    state_estimation = replace(config.state_estimation, **values)
    return replace(config, state_estimation=state_estimation)


def _snapshot(landmarks, detections):
    return SimpleNamespace(
        roll_rad=0.0,
        pitch_rad=0.0,
        yaw_rad=0.0,
        latest_perception={
            "camera_translation_body": np.zeros(3, dtype=float),
            "detections": detections,
        },
        stable_gate_landmarks_neu=landmarks,
    )


def _detection(body_center):
    return {
        "gate_center_body_frd": np.asarray(body_center, dtype=float),
        "confidence": 1.0,
        "reprojection_error": 0.0,
        "gate_center_camera": np.asarray([0.0, 0.0, 1.0], dtype=float),
    }


def _estimator(pos_neu, config=None):
    estimator = VehicleStateEstimator(
        config or _config(),
        mode_override="estimator",
    )
    estimator.initialized = True
    estimator.pos_neu = np.asarray(pos_neu, dtype=float)
    estimator.vel_neu = np.zeros(3, dtype=float)
    return estimator


def test_rejects_single_landmark_with_large_residual():
    estimator = _estimator([2.0, 0.0, 0.0])
    snapshot = _snapshot(
        [{"track_id": 1, "position_neu": np.array([10.0, 0.0, 0.0])}],
        [_detection([10.0, 0.0, 0.0])],
    )

    correction = estimator._correct_with_vision(snapshot)

    assert correction["source"] == "reject:insufficient_landmark_support"
    assert correction["count"] == 1
    np.testing.assert_allclose(estimator.pos_neu, np.array([2.0, 0.0, 0.0]))


def test_rejects_multiple_landmarks_with_high_average_residual():
    estimator = _estimator([2.0, 0.0, 0.0])
    snapshot = _snapshot(
        [
            {"track_id": 1, "position_neu": np.array([10.0, 0.0, 0.0])},
            {"track_id": 2, "position_neu": np.array([0.0, 10.0, 0.0])},
        ],
        [
            _detection([10.0, 0.0, 0.0]),
            _detection([0.0, 10.0, 0.0]),
        ],
    )

    correction = estimator._correct_with_vision(snapshot)

    assert correction["source"] == "reject:high_avg_residual"
    assert correction["count"] == 2
    np.testing.assert_allclose(estimator.pos_neu, np.array([2.0, 0.0, 0.0]))


def test_allows_single_landmark_with_very_low_residual():
    estimator = _estimator([0.2, 0.0, 0.0])
    snapshot = _snapshot(
        [{"track_id": 1, "position_neu": np.array([10.0, 0.0, 0.0])}],
        [_detection([10.0, 0.0, 0.0])],
    )

    correction = estimator._correct_with_vision(snapshot)

    assert correction["source"] == "stable_track:1"
    assert correction["count"] == 1
    np.testing.assert_allclose(estimator.pos_neu, np.zeros(3), atol=1e-9)


def test_allows_multiple_landmarks_with_low_average_residual():
    estimator = _estimator([0.4, 0.0, 0.0])
    snapshot = _snapshot(
        [
            {"track_id": 1, "position_neu": np.array([10.0, 0.0, 0.0])},
            {"track_id": 2, "position_neu": np.array([0.0, 10.0, 0.0])},
        ],
        [
            _detection([10.0, 0.0, 0.0]),
            _detection([0.0, 10.0, 0.0]),
        ],
    )

    correction = estimator._correct_with_vision(snapshot)

    assert correction["source"] == "stable_track:1+stable_track:2"
    assert correction["count"] == 2
    np.testing.assert_allclose(estimator.pos_neu, np.zeros(3), atol=1e-9)


def test_accepted_visual_positions_correct_velocity():
    config = _config(
        vision_correction_alpha_xy=0.0,
        vision_correction_alpha_z=0.0,
        vision_correction_velocity_alpha_xy=1.0,
        vision_correction_velocity_alpha_z=1.0,
        vision_correction_bias_alpha_xy=0.0,
        vision_correction_bias_alpha_z=0.0,
        vision_correction_min_measurements=1,
        vision_correction_velocity_min_measurements=1,
        vision_correction_single_landmark_max_residual_m=10.0,
        vision_correction_max_avg_residual_m=10.0,
        vision_correction_velocity_max_residual_m=10.0,
        vision_correction_max_velocity_innovation_m_s=10.0,
        vision_correction_max_velocity_delta_m_s=10.0,
    )
    estimator = _estimator([0.0, 0.0, 0.0], config=config)

    first = _snapshot(
        [{"track_id": 1, "position_neu": np.array([10.0, 0.0, 0.0])}],
        [_detection([10.0, 0.0, 0.0])],
    )
    second = _snapshot(
        [{"track_id": 1, "position_neu": np.array([11.0, 0.0, 2.0])}],
        [_detection([10.0, 0.0, 0.0])],
    )

    estimator._correct_with_vision(first, now=1.0)
    correction = estimator._correct_with_vision(second, now=2.0)

    np.testing.assert_allclose(
        correction["visual_velocity_neu"],
        np.array([1.0, 0.0, 2.0]),
    )
    np.testing.assert_allclose(estimator.vel_neu, np.array([1.0, 0.0, 2.0]))


def test_short_visual_intervals_do_not_prevent_velocity_measurement():
    config = _config(
        vision_correction_alpha_xy=0.0,
        vision_correction_alpha_z=0.0,
        vision_correction_velocity_alpha_xy=1.0,
        vision_correction_velocity_alpha_z=1.0,
        vision_correction_bias_alpha_xy=0.0,
        vision_correction_bias_alpha_z=0.0,
        vision_correction_min_velocity_dt_s=0.10,
        vision_correction_max_velocity_dt_s=1.0,
        vision_correction_min_measurements=1,
        vision_correction_velocity_min_measurements=1,
        vision_correction_single_landmark_max_residual_m=10.0,
        vision_correction_max_avg_residual_m=10.0,
        vision_correction_velocity_max_residual_m=10.0,
        vision_correction_max_velocity_delta_m_s=10.0,
    )
    estimator = _estimator([0.0, 0.0, 0.0], config=config)

    first = _snapshot(
        [{"track_id": 1, "position_neu": np.array([10.0, 0.0, 0.0])}],
        [_detection([10.0, 0.0, 0.0])],
    )
    second = _snapshot(
        [{"track_id": 1, "position_neu": np.array([10.05, 0.0, 0.0])}],
        [_detection([10.0, 0.0, 0.0])],
    )
    third = _snapshot(
        [{"track_id": 1, "position_neu": np.array([10.11, 0.0, 0.0])}],
        [_detection([10.0, 0.0, 0.0])],
    )

    estimator._correct_with_vision(first, now=1.00)
    short_interval = estimator._correct_with_vision(second, now=1.05)
    correction = estimator._correct_with_vision(third, now=1.11)

    assert short_interval["visual_velocity_neu"] is None
    np.testing.assert_allclose(
        correction["visual_velocity_neu"],
        np.array([1.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(estimator.vel_neu, np.array([1.0, 0.0, 0.0]))


def test_visual_velocity_residual_updates_accel_bias_with_correct_sign():
    config = _config(
        vision_correction_alpha_xy=0.0,
        vision_correction_alpha_z=0.0,
        vision_correction_velocity_alpha_xy=0.0,
        vision_correction_velocity_alpha_z=0.0,
        vision_correction_bias_alpha_xy=1.0,
        vision_correction_bias_alpha_z=1.0,
        vision_correction_min_measurements=1,
        vision_correction_velocity_min_measurements=1,
        vision_correction_single_landmark_max_residual_m=10.0,
        vision_correction_max_avg_residual_m=10.0,
        vision_correction_velocity_max_residual_m=10.0,
        vision_correction_max_velocity_innovation_m_s=10.0,
        vision_correction_max_velocity_delta_m_s=10.0,
        vision_correction_max_bias_delta_m_s2=10.0,
        vision_correction_max_accel_bias_m_s2=10.0,
    )
    estimator = _estimator([0.0, 0.0, 0.0], config=config)

    first = _snapshot(
        [{"track_id": 1, "position_neu": np.array([10.0, 0.0, 0.0])}],
        [_detection([10.0, 0.0, 0.0])],
    )
    second = _snapshot(
        [{"track_id": 1, "position_neu": np.array([10.0, 0.0, 1.0])}],
        [_detection([10.0, 0.0, 0.0])],
    )

    estimator._correct_with_vision(first, now=1.0)
    estimator._correct_with_vision(second, now=2.0)

    np.testing.assert_allclose(estimator.accel_bias_neu, np.array([0.0, 0.0, -1.0]))


def test_visual_velocity_rejects_low_support_without_blocking_position_correction():
    config = _config(
        vision_correction_alpha_xy=1.0,
        vision_correction_alpha_z=1.0,
        vision_correction_velocity_alpha_xy=1.0,
        vision_correction_velocity_alpha_z=1.0,
        vision_correction_min_measurements=1,
        vision_correction_velocity_min_measurements=2,
        vision_correction_single_landmark_max_residual_m=10.0,
        vision_correction_max_avg_residual_m=10.0,
        vision_correction_max_visual_speed_m_s=10.0,
        vision_correction_max_velocity_innovation_m_s=10.0,
    )
    estimator = _estimator([0.0, 0.0, 0.0], config=config)

    first = _snapshot(
        [{"track_id": 1, "position_neu": np.array([10.0, 0.0, 0.0])}],
        [_detection([10.0, 0.0, 0.0])],
    )
    second = _snapshot(
        [{"track_id": 1, "position_neu": np.array([11.0, 0.0, 0.0])}],
        [_detection([10.0, 0.0, 0.0])],
    )

    estimator._correct_with_vision(first, now=1.0)
    correction = estimator._correct_with_vision(second, now=2.0)

    assert correction["visual_velocity_neu"] is None
    assert correction["visual_velocity_reason"] == "reject:visual_velocity_support"
    np.testing.assert_allclose(estimator.pos_neu, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(estimator.vel_neu, np.zeros(3))


def test_visual_velocity_rejects_high_residual_without_blocking_position_correction():
    config = _config(
        vision_correction_alpha_xy=0.0,
        vision_correction_alpha_z=0.0,
        vision_correction_velocity_alpha_xy=1.0,
        vision_correction_velocity_alpha_z=1.0,
        vision_correction_min_measurements=1,
        vision_correction_velocity_min_measurements=1,
        vision_correction_single_landmark_max_residual_m=10.0,
        vision_correction_max_avg_residual_m=10.0,
        vision_correction_velocity_max_residual_m=0.25,
        vision_correction_max_visual_speed_m_s=10.0,
        vision_correction_max_velocity_innovation_m_s=10.0,
    )
    estimator = _estimator([0.0, 0.0, 0.0], config=config)

    first = _snapshot(
        [{"track_id": 1, "position_neu": np.array([10.0, 0.0, 0.0])}],
        [_detection([10.0, 0.0, 0.0])],
    )
    second = _snapshot(
        [{"track_id": 1, "position_neu": np.array([11.0, 0.0, 0.0])}],
        [_detection([10.0, 0.0, 0.0])],
    )

    estimator._correct_with_vision(first, now=1.0)
    correction = estimator._correct_with_vision(second, now=2.0)

    assert correction["source"] == "stable_track:1"
    assert correction["visual_velocity_neu"] is None
    assert correction["visual_velocity_reason"] == "reject:visual_velocity_residual"
    np.testing.assert_allclose(estimator.vel_neu, np.zeros(3))


def test_visual_velocity_rejects_implausible_speed():
    config = _config(
        vision_correction_alpha_xy=0.0,
        vision_correction_alpha_z=0.0,
        vision_correction_velocity_alpha_xy=1.0,
        vision_correction_velocity_alpha_z=1.0,
        vision_correction_min_measurements=1,
        vision_correction_velocity_min_measurements=1,
        vision_correction_single_landmark_max_residual_m=10.0,
        vision_correction_max_avg_residual_m=10.0,
        vision_correction_max_residual_m=10.0,
        vision_correction_velocity_max_residual_m=10.0,
        vision_correction_max_visual_speed_m_s=2.0,
        vision_correction_max_velocity_innovation_m_s=10.0,
    )
    estimator = _estimator([0.0, 0.0, 0.0], config=config)

    first = _snapshot(
        [{"track_id": 1, "position_neu": np.array([10.0, 0.0, 0.0])}],
        [_detection([10.0, 0.0, 0.0])],
    )
    second = _snapshot(
        [{"track_id": 1, "position_neu": np.array([15.0, 0.0, 0.0])}],
        [_detection([10.0, 0.0, 0.0])],
    )

    estimator._correct_with_vision(first, now=1.0)
    correction = estimator._correct_with_vision(second, now=2.0)

    assert correction["visual_velocity_neu"] is None
    assert correction["visual_velocity_reason"] == "reject:visual_velocity_speed"
    np.testing.assert_allclose(estimator.vel_neu, np.zeros(3))


def test_preserves_gazebo_camera_sim_world_projection():
    estimator = _estimator([0.0, 0.0, 0.0])
    latest_perception = {
        "world_pose_source": "gazebo_camera_sim",
        "world_frame": "gazebo_camera_sim_neu",
        "camera_translation_body": np.zeros(3, dtype=float),
        "detections": [
            {
                "gate_center_world": np.array([1.0, 2.0, 3.0], dtype=float),
                "gate_center_world_ned": np.array([2.0, 1.0, -3.0], dtype=float),
                "gate_center_body_frd": np.array([10.0, 0.0, 0.0], dtype=float),
            }
        ],
    }
    estimate = SimpleNamespace(
        valid=True,
        pos_neu=np.array([100.0, 100.0, 100.0], dtype=float),
        yaw_rad=0.0,
        source="mavlink",
    )
    snapshot = SimpleNamespace(roll_rad=0.0, pitch_rad=0.0, yaw_rad=0.0)

    projected = estimator.project_perception_with_estimated_state(
        latest_perception,
        estimate,
        snapshot,
    )

    assert projected["world_pose_source"] == "gazebo_camera_sim"
    assert projected["world_frame"] == "gazebo_camera_sim_neu"
    np.testing.assert_allclose(
        projected["detections"][0]["gate_center_world"],
        np.array([1.0, 2.0, 3.0]),
    )


def test_gazebo_camera_sim_skips_landmark_vision_correction():
    estimator = _estimator([2.0, 0.0, 0.0])
    snapshot = _snapshot(
        [{"track_id": 1, "position_neu": np.array([10.0, 0.0, 0.0])}],
        [_detection([10.0, 0.0, 0.0])],
    )
    snapshot.latest_perception["world_pose_source"] = "gazebo_camera_sim"

    correction = estimator._correct_with_vision(snapshot)

    assert correction["source"] == "gazebo_camera_sim_world_pose"
    assert correction["count"] == 0
    np.testing.assert_allclose(estimator.pos_neu, np.array([2.0, 0.0, 0.0]))
