import time

import numpy as np

from autonomy_core.perception.gate_memory import GateObservation, GateTrack
from autonomy_wrapper import PyAIPilotAutonomyAPI


def _stable_track(
    track_id: int,
    center,
    *,
    hits: int = 10,
    score: float = 0.8,
    committed: bool = True,
    stable: bool = True,
):
    center = np.asarray(center, dtype=float)
    now = time.time()
    track = GateTrack(
        id=int(track_id),
        center=center.copy(),
        confidence_sum=float(hits),
        hits=int(hits),
        first_seen_time=now - 1.0,
        last_seen_time=now,
        committed=bool(committed),
        planning_center=center.copy(),
    )
    track.filtered_center_world = center.copy()
    track.center_world_std = np.zeros(3)
    track.is_stable = bool(stable)
    track.stability_score = float(score)
    track.inlier_count = int(hits)
    track.reprojection_error_median = 0.1
    track.obs_history = [
        GateObservation(
            timestamp=now,
            center_world=center.copy(),
            reprojection_error=0.1,
            confidence=1.0,
            keypoint_conf_min=1.0,
            keypoint_conf_mean=1.0,
            quality_ok=True,
        )
    ]
    return track


def test_gate_pass_advances_inside_existing_horizon():
    api = PyAIPilotAutonomyAPI(use_perception=False, race_gate_count=4)
    api.horizon_continuation_enabled = True
    api.terminal_velocity_enabled = True
    api.terminal_speed_m_s = 0.5
    api.gate_centers_neu = [
        np.array([0.0, 10.0, 1.5]),
        np.array([0.0, 20.0, 1.5]),
        np.array([0.0, 30.0, 1.5]),
    ]
    api.gate_track_ids = [-1, -2, -3]
    api.current_gate_idx = 0

    planned = api._path_plan(
        pos=np.array([0.0, 0.0, 1.5]),
        vel=np.zeros(3),
    )

    assert planned
    assert api.active_plan_mode == "gate_horizon"
    assert api.active_horizon_gate_indices == [0, 1, 2]
    assert np.linalg.norm(api.active_terminal_velocity) > 0.0

    advanced = api._advance_gate_if_needed(np.array([0.0, 10.2, 1.5]))

    assert advanced
    assert api.last_gate_pass_preserved_plan
    assert api.current_gate_idx == 1
    assert api.active_waypoints is not None
    assert api.target_manager.locked
    assert api.active_target_track_id == -2
    np.testing.assert_allclose(api.current_gate_pos, np.array([0.0, 20.0, 1.5]))


def test_duplicate_cluster_center_uses_best_sibling_track():
    api = PyAIPilotAutonomyAPI(use_perception=True, race_gate_count=4)
    stale_active = _stable_track(19, np.array([-0.09, 28.06, 4.08]), hits=8, score=0.2)
    better_sibling = _stable_track(18, np.array([-0.03, 29.64, 4.19]), hits=30, score=0.95)
    api.gate_memory.tracks = [stale_active, better_sibling]

    center, source_track_id, quality = api._best_duplicate_cluster_center(19)

    assert source_track_id == 18
    assert quality["ok"]
    assert set(quality["cluster_ids"]) == {18, 19}
    np.testing.assert_allclose(center, better_sibling.filtered_center_world)


def test_horizon_continue_rejects_target_inside_completed_segment_corridor():
    api = PyAIPilotAutonomyAPI(use_perception=True, race_gate_count=5)
    completed_a = np.array([1.50, 58.46, 10.63])
    completed_b = np.array([-2.02, 93.81, 10.30])
    stale_mid_segment = _stable_track(1, np.array([-1.52, 65.01, 6.99]), hits=20, score=0.9)
    api.gate_memory.tracks = [stale_mid_segment]
    api.completed_track_ids = {17, 30}
    api.completed_gate_positions = [completed_a.copy(), completed_b.copy()]
    api.completed_gate_segments = [(completed_a.copy(), completed_b.copy())]

    valid, reason, target, source_track_id = api._validated_horizon_continue_target(
        track_id=1,
        stored_target=stale_mid_segment.center,
        next_gate_idx=4,
    )

    assert not valid
    assert reason == "completed_segment_corridor"
    assert target is None
    assert source_track_id == 1


def test_ordered_perception_gates_keeps_completed_prefix_and_skips_cleared_suffix():
    api = PyAIPilotAutonomyAPI(use_perception=True, race_gate_count=6)
    completed_tracks = [
        _stable_track(19, np.array([-0.09, 28.06, 4.08])),
        _stable_track(0, np.array([-0.34, 35.22, 5.24])),
        _stable_track(17, np.array([1.50, 58.46, 10.63])),
        _stable_track(30, np.array([-2.02, 93.81, 10.30])),
    ]
    stale_cleared = _stable_track(1, np.array([-1.52, 65.01, 6.99]))
    valid_future = _stable_track(77, np.array([-2.41, 121.44, 11.92]))
    api.gate_memory.tracks = [*completed_tracks, stale_cleared, valid_future]
    api.current_gate_idx = 4
    api.race_order_track_ids = [19, 0, 17, 30, 1, 77]
    api.completed_track_ids = {19, 0, 17, 30}
    api.completed_gate_positions = [
        np.array([-0.09, 28.06, 4.08]),
        np.array([-0.34, 35.22, 5.24]),
        np.array([1.50, 58.46, 10.63]),
        np.array([-2.02, 93.81, 10.30]),
    ]
    api.completed_gate_segments = [
        (api.completed_gate_positions[0], api.completed_gate_positions[1]),
        (api.completed_gate_positions[1], api.completed_gate_positions[2]),
        (api.completed_gate_positions[2], api.completed_gate_positions[3]),
    ]

    committed_by_id = {int(track.id): track for track in api.gate_memory.get_committed_tracks()}
    gates, track_ids = api._ordered_perception_gates(committed_by_id)

    assert track_ids == [19, 0, 17, 30, 77]
    assert track_ids[api.current_gate_idx] == 77
    np.testing.assert_allclose(gates[api.current_gate_idx], valid_future.filtered_center_world)


def test_provisional_next_gate_tracks_fresh_uncommitted_candidate_without_locking():
    api = PyAIPilotAutonomyAPI(use_perception=True, race_gate_count=10)
    api.provisional_next_gate_enabled = True
    api.provisional_next_gate_min_hits = 2
    api.current_gate_idx = 7
    api.completed_track_ids = {141}
    api.completed_gate_positions = [np.array([2.3, 211.6, 18.2])]
    candidate = _stable_track(
        187,
        np.array([-2.7, 250.0, 14.5]),
        hits=3,
        committed=False,
        stable=False,
    )
    api.gate_memory.tracks = [candidate]

    planned = api._path_plan_provisional_next_gate(
        pos=np.array([2.4, 213.0, 18.1]),
        vel=np.array([0.0, 1.0, 0.0]),
    )

    assert planned
    assert api.provisional_target_active
    assert api.active_plan_mode == "provisional_next_gate"
    assert api.active_target_track_id == 187
    assert not api.target_manager.locked
    assert api.current_gate_idx == 7


def test_provisional_next_gate_cannot_advance_gate_index():
    api = PyAIPilotAutonomyAPI(use_perception=True, race_gate_count=10)
    api.provisional_target_active = True
    api.provisional_target_track_id = 187
    api.current_gate_idx = 7
    api.current_gate_pos = np.array([-2.7, 250.0, 14.5])
    api.active_target_track_id = 187
    api.previous_gate_pass_position = np.array([-2.7, 249.0, 14.5])
    api.active_gate_normal = np.array([0.0, 1.0, 0.0])

    advanced = api._advance_gate_if_needed(np.array([-2.7, 250.1, 14.5]))

    assert not advanced
    assert api.current_gate_idx == 7
    assert 187 not in api.completed_track_ids


def test_provisional_next_gate_hands_off_to_normal_race_order_target():
    api = PyAIPilotAutonomyAPI(use_perception=True, race_gate_count=10)
    api.provisional_target_active = True
    api.provisional_target_track_id = 187
    api.provisional_target_gate_idx = 7
    api.provisional_target_center = np.array([-2.7, 250.0, 14.5])
    api.current_gate_idx = 7
    api.current_gate_pos = api.provisional_target_center.copy()
    api.active_target_track_id = 187
    api.active_waypoints = np.vstack(
        [
            np.array([2.4, 213.0, 18.1]),
            api.provisional_target_center,
        ]
    )
    api.planner.total_time = 3.0
    api.active_plan_mode = "provisional_next_gate"
    api._candidate_gate_track_ids = [10, 11, 12, 13, 14, 15, 16, 187]

    gates = [
        np.array([0.0, 10.0, 1.5]),
        np.array([0.0, 20.0, 1.5]),
        np.array([0.0, 30.0, 1.5]),
        np.array([0.0, 40.0, 1.5]),
        np.array([0.0, 50.0, 1.5]),
        np.array([0.0, 60.0, 1.5]),
        np.array([0.0, 70.0, 1.5]),
        np.array([-2.8, 249.5, 14.4]),
    ]

    api._install_gate_centers(gates)

    assert not api.provisional_target_active
    assert api.active_waypoints is None
    assert api.last_planned_gate_idx == -1
    assert api.active_target_track_id == 187
    np.testing.assert_allclose(api.current_gate_pos, gates[7])
