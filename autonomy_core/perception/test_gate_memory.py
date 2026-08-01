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


def test_known_position_commit_filter_blocks_far_track_without_snapping_near_track():
    memory = GateMemory(
        association_radius=2.0,
        commit_radius=0.2,
        min_confidence_per_hit=0.0,
        commit_hits=3,
        commit_confidence_sum=0.0,
        commit_spread_radius=0.2,
        history_size=10,
        min_hits_for_stable=3,
        max_center_std_for_stable=10.0,
        max_camera_std_for_stable=10.0,
        max_reprojection_error_for_stable=100.0,
        max_outlier_distance=10.0,
        min_observation_time=0.0,
        known_position_commit_filter_enabled=True,
        known_position_commit_radius_m=2.0,
        known_gate_positions_neu=[[0.0, 0.0, 0.0]],
    )

    far_center = np.array([10.0, 0.0, 0.0])
    for idx in range(3):
        memory.add_detection(
            far_center,
            confidence=1.0,
            timestamp=float(idx + 1),
        )

    far_track = memory.tracks[0]
    assert not far_track.committed
    assert far_track.commit_blocked_reason == "outside_known_position_commit_radius"
    assert far_track.known_position_match_index == 0
    assert far_track.known_position_distance_m == 10.0

    near_center = np.array([1.0, 0.0, 0.0])
    for idx in range(3):
        memory.add_detection(
            near_center,
            confidence=1.0,
            timestamp=float(idx + 10),
        )

    committed = memory.get_committed_tracks()
    assert len(committed) == 1
    np.testing.assert_allclose(committed[0].center, near_center)
    np.testing.assert_allclose(committed[0].planning_center, near_center)
    assert committed[0].known_position_match_index == 0
    assert committed[0].known_position_distance_m == 1.0


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


def test_new_track_admission_can_be_suppressed_without_changing_memory():
    memory = GateMemory(
        association_radius=1.0,
        min_confidence_per_hit=0.0,
    )

    result = memory.add_detection(
        np.array([10.0, 0.0, 1.0]),
        confidence=1.0,
        timestamp=1.0,
        allow_new_track=False,
        admission_reason="active_gate_transit",
    )

    assert result["accepted"] is False
    assert result["reason"] == "new_track_suppressed:active_gate_transit"
    assert memory.tracks == []


def test_candidate_updates_without_committing_during_admission_freeze():
    memory = GateMemory(
        association_radius=1.0,
        commit_radius=0.2,
        min_confidence_per_hit=0.0,
        commit_hits=2,
        commit_confidence_sum=0.0,
        commit_spread_radius=0.2,
        min_hits_for_stable=2,
        max_center_std_for_stable=10.0,
        max_camera_std_for_stable=10.0,
        max_reprojection_error_for_stable=100.0,
        min_observation_time=0.0,
    )
    center = np.array([10.0, 0.0, 1.0])
    memory.add_detection(center, confidence=1.0, timestamp=0.0)

    frozen = memory.add_detection(
        center,
        confidence=1.0,
        timestamp=1.0,
        allow_new_track=False,
        allow_candidate_commit=False,
        admission_reason="active_gate_transit",
    )

    track = memory.tracks[0]
    assert frozen["accepted"] is True
    assert frozen["reason"] == "updated_track"
    assert frozen["commit_suppressed"] is True
    assert track.hits == 2
    assert not track.committed

    released = memory.add_detection(
        center,
        confidence=1.0,
        timestamp=2.0,
    )
    assert released["committed_now"] is True
    assert track.committed


def test_committed_track_still_updates_during_admission_freeze():
    memory = _memory_for_freeze_test()
    center = np.array([1.0, 2.0, 3.0])
    for idx in range(3):
        memory.add_detection(center, confidence=1.0, timestamp=float(idx))
    track = memory.get_committed_tracks()[0]
    hits_before = track.hits

    result = memory.add_detection(
        center + np.array([0.1, 0.0, 0.0]),
        confidence=1.0,
        timestamp=3.0,
        allow_new_track=False,
        allow_candidate_commit=False,
        admission_reason="active_gate_transit",
    )

    assert result["accepted"] is True
    assert result["reason"] == "matched_committed_track"
    assert track.hits == hits_before + 1
    assert track.committed


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


def test_planning_lock_freezes_center_while_observations_continue():
    memory = _memory_for_freeze_test()
    initial = np.array([1.0, 2.0, 3.0])
    for idx in range(3):
        result = memory.add_detection(initial, confidence=1.0, timestamp=float(idx))
        assert result["accepted"] is True

    track = memory.get_committed_tracks()[0]
    assert track.is_stable
    hits_before_lock = track.hits
    assert memory.lock_track_for_planning(track.id, reason="prearm_test") is True
    locked_center = track.planning_center.copy()

    shifted = initial + np.array([0.5, 0.0, 0.0])
    for idx in range(3, 7):
        result = memory.add_detection(shifted, confidence=1.0, timestamp=float(idx))
        assert result["accepted"] is True
        assert result["track_id"] == track.id

    track = memory.get_track_by_id(track.id)
    assert track is not None
    assert track.planning_locked
    assert track.planning_lock_reason == "prearm_test"
    assert track.hits == hits_before_lock + 4
    assert track.last_seen_time == 6.0
    np.testing.assert_allclose(track.center, locked_center)
    np.testing.assert_allclose(track.planning_center, locked_center)
    assert track.filtered_center_world is not None
    assert np.linalg.norm(track.filtered_center_world - locked_center) > 0.1


def test_planning_lock_rejects_uncommitted_and_unstable_tracks():
    uncommitted_memory = GateMemory(
        association_radius=5.0,
        min_confidence_per_hit=0.0,
        commit_hits=3,
        commit_confidence_sum=0.0,
    )
    uncommitted_memory.add_detection(
        np.array([0.0, 0.0, 1.0]),
        confidence=1.0,
        timestamp=0.0,
    )
    uncommitted = uncommitted_memory.tracks[0]
    assert not uncommitted.committed
    assert uncommitted_memory.lock_track_for_planning(uncommitted.id) is False
    assert not uncommitted.planning_locked

    unstable_memory = GateMemory(
        association_radius=5.0,
        commit_radius=0.2,
        min_confidence_per_hit=0.0,
        commit_hits=1,
        commit_confidence_sum=0.0,
        commit_spread_radius=0.2,
        min_hits_for_stable=3,
        min_observation_time=0.0,
    )
    unstable_memory.add_detection(
        np.array([5.0, 0.0, 1.0]),
        confidence=1.0,
        timestamp=0.0,
    )
    unstable = unstable_memory.get_committed_tracks()[0]
    assert unstable.committed
    assert not unstable.is_stable
    assert not unstable.ever_stable
    assert unstable_memory.lock_track_for_planning(unstable.id) is False
    assert not unstable.planning_locked


def test_planning_lock_is_idempotent_and_does_not_resnapshot_filter():
    memory = _memory_for_freeze_test()
    initial = np.array([2.0, 3.0, 4.0])
    for idx in range(3):
        memory.add_detection(initial, confidence=1.0, timestamp=float(idx))

    track = memory.get_committed_tracks()[0]
    assert memory.lock_track_for_planning(track.id, reason="first_lock") is True
    locked_center = track.planning_center.copy()

    shifted = initial + np.array([0.4, 0.0, 0.0])
    memory.add_detection(shifted, confidence=1.0, timestamp=3.0)
    assert track.filtered_center_world is not None
    assert np.linalg.norm(track.filtered_center_world - locked_center) > 0.0

    assert memory.lock_track_for_planning(track.id, reason="second_lock") is True
    np.testing.assert_allclose(track.center, locked_center)
    np.testing.assert_allclose(track.planning_center, locked_center)
    assert track.planning_lock_reason == "first_lock"


def test_merge_refuses_when_either_track_is_planning_locked():
    memory = _memory_for_freeze_test()
    first_center = np.array([0.0, 0.0, 1.0])
    second_center = np.array([10.0, 0.0, 1.0])
    for idx in range(3):
        memory.add_detection(first_center, confidence=1.0, timestamp=float(idx))
        memory.add_detection(second_center, confidence=1.0, timestamp=float(idx))

    first, second = memory.get_committed_tracks()
    assert first.is_stable
    assert second.is_stable
    assert memory.lock_track_for_planning(first.id) is True
    track_ids_before = [track.id for track in memory.tracks]

    assert memory.merge_track_into(first.id, second.id) is None
    assert memory.merge_track_into(second.id, first.id) is None
    assert [track.id for track in memory.tracks] == track_ids_before
    np.testing.assert_allclose(first.center, first_center)
    np.testing.assert_allclose(second.center, second_center)


def _tiny_gate_keypoints(
    *,
    center_x: float = 335.5,
    center_y: float = 261.0,
    width: float = 5.0,
    height: float = 6.0,
) -> np.ndarray:
    half_width = 0.5 * width
    half_height = 0.5 * height
    return np.asarray(
        [
            [center_x - half_width, center_y - half_height],
            [center_x + half_width, center_y - half_height],
            [center_x + half_width, center_y + half_height],
            [center_x - half_width, center_y + half_height],
        ],
        dtype=float,
    )


def test_temporal_image_association_keeps_depth_aliases_on_one_candidate():
    memory = GateMemory(
        association_radius=1.5,
        min_confidence_per_hit=0.0,
        commit_hits=99,
        commit_confidence_sum=99.0,
        max_outlier_distance=1.5,
        temporal_image_association_enabled=True,
        temporal_image_association_max_age_s=1.0,
        temporal_image_association_max_center_distance_px=2.5,
        temporal_image_association_max_size_ratio=1.5,
        temporal_image_association_min_depth_m=20.0,
    )
    keypoints = _tiny_gate_keypoints()

    first = memory.add_detection(
        np.array([-97.6, -4.6, -20.6]),
        confidence=1.0,
        timestamp=0.0,
        center_camera=np.array([4.53, 24.26, 95.78]),
        keypoints_px=keypoints,
    )
    assert first["reason"] == "new_track"

    jumped = memory.add_detection(
        np.array([-105.0, -5.0, -22.3]),
        confidence=1.0,
        timestamp=0.1,
        center_camera=np.array([4.90, 26.29, 103.03]),
        keypoints_px=keypoints + np.array([0.25, 0.0]),
    )

    assert len(memory.tracks) == 1
    assert jumped["track_id"] == first["track_id"]
    assert jumped["matched_visually"] is True
    assert jumped["reason"] == "image_track_depth_outlier"

    # Once the later PnP mode is repeated, the uncommitted candidate may
    # switch to that dominant, internally consistent depth without creating a
    # second world landmark.
    repeated = memory.add_detection(
        np.array([-104.8, -5.1, -22.2]),
        confidence=1.0,
        timestamp=0.2,
        center_camera=np.array([4.95, 26.20, 102.92]),
        keypoints_px=keypoints + np.array([0.25, 0.0]),
    )
    assert len(memory.tracks) == 1
    assert repeated["track_id"] == first["track_id"]
    assert repeated["accepted"] is True
    assert memory.tracks[0].center[0] < -104.0


def test_temporal_image_association_does_not_merge_distinct_boxes():
    memory = GateMemory(
        association_radius=1.5,
        min_confidence_per_hit=0.0,
        commit_hits=99,
        commit_confidence_sum=99.0,
        temporal_image_association_enabled=True,
        temporal_image_association_max_age_s=1.0,
        temporal_image_association_max_center_distance_px=2.5,
        temporal_image_association_min_depth_m=20.0,
    )

    memory.add_detection(
        np.array([-97.0, 0.0, 0.0]),
        confidence=1.0,
        timestamp=0.0,
        center_camera=np.array([4.5, 24.0, 96.0]),
        keypoints_px=_tiny_gate_keypoints(center_y=261.0),
    )
    second = memory.add_detection(
        np.array([-105.0, 0.0, 0.0]),
        confidence=1.0,
        timestamp=0.1,
        center_camera=np.array([2.2, 18.5, 100.0]),
        keypoints_px=_tiny_gate_keypoints(center_y=244.0),
    )

    assert second["reason"] == "new_track"
    assert len(memory.tracks) == 2


def test_temporal_image_association_rejects_same_frame_duplicate():
    memory = GateMemory(
        association_radius=1.5,
        min_confidence_per_hit=0.0,
        commit_hits=99,
        commit_confidence_sum=99.0,
        temporal_image_association_enabled=True,
        temporal_image_association_min_depth_m=20.0,
    )
    keypoints = _tiny_gate_keypoints()

    first = memory.add_detection(
        np.array([-97.0, 0.0, 0.0]),
        confidence=1.0,
        timestamp=10.0,
        center_camera=np.array([4.5, 24.0, 96.0]),
        keypoints_px=keypoints,
    )
    duplicate = memory.add_detection(
        np.array([-105.0, 0.0, 0.0]),
        confidence=1.0,
        timestamp=10.0,
        center_camera=np.array([4.9, 26.0, 103.0]),
        keypoints_px=keypoints,
    )

    assert first["reason"] == "new_track"
    assert duplicate["reason"] == "same_frame_image_duplicate"
    assert duplicate["matched_visually"] is True
    assert len(memory.tracks) == 1
    assert memory.tracks[0].hits == 1


def test_tiny_detection_requires_longer_inlier_history_for_stability():
    memory = GateMemory(
        association_radius=5.0,
        commit_radius=0.2,
        min_confidence_per_hit=0.0,
        commit_hits=2,
        commit_confidence_sum=0.0,
        commit_spread_radius=0.2,
        history_size=20,
        min_hits_for_stable=3,
        max_center_std_for_stable=0.5,
        max_camera_std_for_stable=0.5,
        max_reprojection_error_for_stable=1.0,
        max_outlier_distance=0.5,
        min_observation_time=0.0,
        tiny_detection_area_px2=50.0,
        tiny_detection_min_hits_for_stable=6,
        tiny_detection_min_observation_time=0.5,
    )
    center_world = np.array([-104.0, -5.0, -22.0])
    center_camera = np.array([5.0, 26.0, 103.0])
    keypoints = _tiny_gate_keypoints(width=5.0, height=5.0)

    for idx in range(5):
        memory.add_detection(
            center_world,
            confidence=1.0,
            timestamp=0.1 * idx,
            center_camera=center_camera,
            keypoints_px=keypoints,
            reprojection_error=0.2,
        )

    track = memory.tracks[0]
    assert track.committed
    assert not track.is_stable
    assert track.required_hits_for_stable == 6
    assert track.promotion_blocked_reason == "insufficient_hits"

    memory.add_detection(
        center_world,
        confidence=1.0,
        timestamp=0.5,
        center_camera=center_camera,
        keypoints_px=keypoints,
        reprojection_error=0.2,
    )
    assert track.is_stable
    assert track.inlier_count == 6
