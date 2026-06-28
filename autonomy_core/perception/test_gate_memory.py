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
