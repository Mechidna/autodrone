from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from experimental_gate_vio_alignment import ExperimentalGateVioAlignment
from runtime_config import load_runtime_config
from vehicle_state_estimator import VehicleStateEstimator


def _config(*, min_frames=3, trace=False):
    config = load_runtime_config()
    alignment = replace(
        config.experimental_gate_vio_alignment,
        enabled=True,
        min_consistent_frames=min_frames,
        temporal_window_s=1.0,
        min_confidence=0.5,
        min_depth_m=0.0,
        initial_association_radius_m=6.0,
        association_radius_m=3.0,
        consistency_radius_m=0.25,
        max_initial_offset_m=6.0,
        max_update_innovation_m=1.5,
        correction_alpha=1.0,
        max_step_m=1.0,
        trace=trace,
    )
    gate_source = replace(
        config.gate_source,
        known_gate_positions_neu=((10.0, 0.0, 0.0),),
    )
    state_estimation = replace(
        config.state_estimation,
        mode="estimator",
        use_imu_prediction=False,
        use_vision_correction=True,
        vision_correction_source="stable_tracks",
    )
    return replace(
        config,
        experimental_gate_vio_alignment=alignment,
        gate_source=gate_source,
        state_estimation=state_estimation,
    )


def _snapshot(frame_id, *, body_x=7.0, confidence=1.0, reprojection=0.1):
    timestamp = 10.0 + 0.1 * frame_id
    return SimpleNamespace(
        frame_id=frame_id,
        image_wall_time=timestamp,
        roll_rad=0.0,
        pitch_rad=0.0,
        yaw_rad=0.0,
        latest_perception={
            "frame_id": frame_id,
            "image_wall_time": timestamp,
            "camera_translation_body": np.zeros(3, dtype=float),
            "transform_mode": "standard",
            "detections": [
                {
                    "gate_center_body_frd": np.array(
                        [body_x, 0.0, 0.0],
                        dtype=float,
                    ),
                    "gate_center_camera": np.array(
                        [0.0, 0.0, max(body_x, 0.1)],
                        dtype=float,
                    ),
                    "confidence": confidence,
                    "reprojection_error": reprojection,
                }
            ],
        },
        stable_gate_landmarks_neu=[
            {"track_id": 99, "position_neu": np.array([50.0, 0.0, 0.0])}
        ],
    )


def test_disabled_alignment_is_an_exact_position_noop():
    config = load_runtime_config()
    aligner = ExperimentalGateVioAlignment(config)
    raw = np.array([1.0, 2.0, 3.0])

    result = aligner.update(raw, _snapshot(1), now=10.1)

    assert not result.enabled
    assert not result.initialized
    assert result.reason == "disabled"
    np.testing.assert_allclose(result.aligned_pos_neu, raw)
    np.testing.assert_allclose(result.offset_neu, np.zeros(3))


def test_requires_distinct_consistent_frames_before_initializing():
    aligner = ExperimentalGateVioAlignment(_config(min_frames=3))
    raw = np.array([1.0, 0.0, 0.0])

    first = aligner.update(raw, _snapshot(1), now=10.1)
    duplicate = aligner.update(raw, _snapshot(1), now=10.15)
    second = aligner.update(raw, _snapshot(2), now=10.2)
    third = aligner.update(raw, _snapshot(3), now=10.3)

    assert first.reason == "awaiting_temporal_consistency"
    assert duplicate.reason == "duplicate_frame"
    assert second.support_count == 2
    assert third.accepted
    assert third.reason == "initialized"
    assert third.support_count == 3
    np.testing.assert_allclose(third.offset_neu, np.array([2.0, 0.0, 0.0]))
    np.testing.assert_allclose(third.aligned_pos_neu, np.array([3.0, 0.0, 0.0]))


def test_inconsistent_gate_pose_sequence_cannot_initialize():
    aligner = ExperimentalGateVioAlignment(_config(min_frames=3))
    raw = np.array([1.0, 0.0, 0.0])

    results = [
        aligner.update(raw, _snapshot(1, body_x=7.0), now=10.1),
        aligner.update(raw, _snapshot(2, body_x=9.0), now=10.2),
        aligner.update(raw, _snapshot(3, body_x=5.0), now=10.3),
        aligner.update(raw, _snapshot(4, body_x=8.0), now=10.4),
    ]

    assert not any(result.accepted for result in results)
    assert not aligner.initialized
    np.testing.assert_allclose(aligner.offset_neu, np.zeros(3))


def test_large_post_initialization_jump_is_rejected_without_changing_offset():
    aligner = ExperimentalGateVioAlignment(_config(min_frames=3))
    raw = np.array([1.0, 0.0, 0.0])
    for frame_id in (1, 2, 3):
        initialized = aligner.update(raw, _snapshot(frame_id), now=10.0 + frame_id / 10)
    before = initialized.offset_neu.copy()

    rejected = aligner.update(raw, _snapshot(4, body_x=4.0), now=10.4)

    assert not rejected.accepted
    assert rejected.reason == "no_gated_measurements"
    np.testing.assert_allclose(rejected.offset_neu, before)
    np.testing.assert_allclose(rejected.aligned_pos_neu, raw + before)


def test_estimator_exposes_aligned_output_without_mutating_raw_state():
    estimator = VehicleStateEstimator(_config(min_frames=3), mode_override="estimator")
    estimator.initialized = True
    estimator.pos_neu = np.array([1.0, 0.0, 0.0])
    estimator.vel_neu = np.array([0.2, -0.1, 0.0])

    estimates = [estimator.update(_snapshot(frame_id)) for frame_id in (1, 2, 3)]
    final = estimates[-1]

    np.testing.assert_allclose(estimator.pos_neu, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(final.raw_pos_neu, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(final.pos_neu, np.array([3.0, 0.0, 0.0]))
    np.testing.assert_allclose(final.vel_neu, np.array([0.2, -0.1, 0.0]))
    assert final.gate_vio_alignment_initialized
    assert final.gate_vio_alignment_accepted
    assert final.source == "estimator+experimental_gate_alignment"
    assert final.vision_correction_source != "stable_track:99"
