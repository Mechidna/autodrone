import numpy as np

from autonomy_core.perception.gate_memory import GateMemory


def _memory_for_freeze_test() -> GateMemory:
    memory = GateMemory(
        association_radius=5.0,
        commit_radius=0.2,
        min_confidence_per_hit=0.0,
        commit_hits=3,
        commit_confidence_sum=0.0,
        commit_spread_radius=0.2,
        history_size=20,
        min_hits_for_stable=3,
        max_center_std_for_stable=10.0,
        max_camera_std_for_stable=10.0,
        max_reprojection_error_for_stable=100.0,
        max_outlier_distance=10.0,
        min_observation_time=0.0,
    )
    memory.max_committed_match_distance = 5.0
    return memory


def test_committed_track_keeps_fixed_planning_center_after_later_observations():
    memory = _memory_for_freeze_test()
    initial = np.array([1.0, 2.0, 3.0])

    for idx in range(3):
        memory.add_detection(initial, confidence=1.0, timestamp=float(idx))

    track = memory.get_committed_tracks()[0]
    np.testing.assert_allclose(track.center, initial)
    np.testing.assert_allclose(track.planning_center, initial)

    shifted = np.array([2.0, 3.0, 4.0])
    for idx in range(3, 8):
        result = memory.add_detection(shifted, confidence=1.0, timestamp=float(idx))
        assert result["accepted"] is True

    track = memory.get_committed_tracks()[0]
    np.testing.assert_allclose(track.center, initial)
    np.testing.assert_allclose(track.planning_center, initial)
    assert track.filtered_center_world is not None
    assert np.linalg.norm(track.filtered_center_world - initial) > 0.1
    np.testing.assert_allclose(memory.get_committed_centers()[0], initial)


def test_large_residual_on_stable_committed_track_is_outlier_not_match():
    memory = GateMemory(
        association_radius=1.5,
        commit_radius=0.6,
        new_track_block_radius=4.5,
        min_confidence_per_hit=0.0,
        commit_hits=3,
        commit_confidence_sum=0.0,
        commit_spread_radius=0.2,
        history_size=20,
        min_hits_for_stable=3,
        max_center_std_for_stable=0.6,
        max_camera_std_for_stable=10.0,
        max_reprojection_error_for_stable=100.0,
        max_outlier_distance=0.6,
        min_observation_time=0.0,
    )
    memory.max_committed_match_distance = 0.6
    center = np.array([0.0, 0.0, 1.0])
    for idx in range(3):
        result = memory.add_detection(center, confidence=1.0, timestamp=float(idx))
        assert result["accepted"] is True

    track = memory.get_committed_tracks()[0]
    assert track.is_stable
    assert track.ever_stable
    np.testing.assert_allclose(track.center, center)

    far_same_region = np.array([2.5, 0.0, 1.0])
    result = memory.add_detection(
        far_same_region,
        confidence=1.0,
        timestamp=3.0,
    )

    assert result["accepted"] is False
    assert result["reason"].startswith("committed_track_outlier")
    track = memory.get_committed_tracks()[0]
    assert track.obs_history[-1].is_outlier
    assert track.ever_stable
    np.testing.assert_allclose(track.center, center)
    np.testing.assert_allclose(track.planning_center, center)


def test_uncommitted_candidate_center_still_updates_before_commit():
    memory = GateMemory(
        association_radius=5.0,
        min_confidence_per_hit=0.0,
        commit_hits=99,
        commit_confidence_sum=99.0,
        history_size=20,
        max_outlier_distance=10.0,
    )
    first = np.array([0.0, 0.0, 1.0])
    second = np.array([1.0, 0.0, 1.0])

    memory.add_detection(first, confidence=1.0, timestamp=0.0)
    memory.add_detection(second, confidence=1.0, timestamp=1.0)

    track = memory.tracks[0]
    assert not track.committed
    np.testing.assert_allclose(track.center, np.array([0.5, 0.0, 1.0]))
    assert track.planning_center is None


def test_observation_keeps_keypoint_confidence_summary():
    memory = GateMemory(
        association_radius=5.0,
        min_confidence_per_hit=0.0,
        commit_hits=99,
        commit_confidence_sum=99.0,
    )

    memory.add_detection(
        np.array([0.0, 0.0, 1.0]),
        confidence=1.0,
        timestamp=0.0,
        keypoint_conf_min=0.75,
        keypoint_conf_mean=0.90,
    )

    obs = memory.tracks[0].obs_history[-1]
    assert obs.keypoint_conf_min == 0.75
    assert obs.keypoint_conf_mean == 0.90


def test_quality_rejected_detection_does_not_create_track():
    memory = GateMemory(
        association_radius=5.0,
        min_confidence_per_hit=0.0,
    )

    result = memory.add_detection(
        np.array([0.0, 0.0, 1.0]),
        confidence=1.0,
        timestamp=0.0,
        quality_ok=False,
        quality_reason="keypoint_on_image_border",
    )

    assert result["accepted"] is False
    assert result["reason"] == "quality_rejected:keypoint_on_image_border"
    assert memory.tracks == []


def test_stable_promotion_requires_keypoint_confidence_threshold():
    memory = GateMemory(
        association_radius=5.0,
        commit_radius=0.2,
        min_confidence_per_hit=0.0,
        commit_hits=3,
        commit_confidence_sum=0.0,
        commit_spread_radius=0.2,
        history_size=20,
        min_hits_for_stable=3,
        max_center_std_for_stable=10.0,
        max_camera_std_for_stable=10.0,
        max_reprojection_error_for_stable=100.0,
        min_keypoint_conf_for_stable=0.8,
        max_outlier_distance=10.0,
        min_observation_time=0.0,
    )
    memory.max_committed_match_distance = 5.0

    center = np.array([0.0, 0.0, 1.0])
    for idx in range(3):
        memory.add_detection(
            center,
            confidence=1.0,
            timestamp=float(idx),
            keypoint_conf_min=0.75,
            keypoint_conf_mean=0.90,
        )

    track = memory.get_committed_tracks()[0]
    assert not track.is_stable
    assert track.promotion_blocked_reason == "keypoint_conf_low"

    for idx in range(3, 6):
        memory.add_detection(
            center,
            confidence=1.0,
            timestamp=float(idx),
            keypoint_conf_min=0.85,
            keypoint_conf_mean=0.95,
        )

    track = memory.get_committed_tracks()[0]
    assert track.is_stable
