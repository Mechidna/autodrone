import math
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
    track.ever_stable = bool(stable)
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


def test_canonical_gate_pose_record_converts_sdf_pose_to_neu_axes():
    yaw0 = PyAIPilotAutonomyAPI._canonical_gate_pose_record_from_sdf_pose(
        gate_id=0,
        sdf_gate_index=1,
        model_name="racing_gate_1",
        pose_values=(10.0, 2.0, 3.0, 0.0, 0.0, 0.0),
    )

    assert yaw0 is not None
    np.testing.assert_allclose(yaw0["center_neu"], [2.0, 10.0, 4.35])
    np.testing.assert_allclose(yaw0["right_axis_neu"], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(yaw0["up_axis_neu"], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(yaw0["normal_neu"], [0.0, 1.0, 0.0])

    yaw90 = PyAIPilotAutonomyAPI._canonical_gate_pose_record_from_sdf_pose(
        gate_id=1,
        sdf_gate_index=2,
        model_name="racing_gate_2",
        pose_values=(10.0, 2.0, 3.0, 0.0, 0.0, math.pi / 2.0),
    )

    assert yaw90 is not None
    np.testing.assert_allclose(yaw90["center_neu"], [2.0, 10.0, 4.35])
    np.testing.assert_allclose(yaw90["normal_neu"], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(yaw90["right_axis_neu"], [0.0, -1.0, 0.0], atol=1e-12)


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

    assert not advanced
    assert api.current_gate_idx == 0

    advanced = api._advance_gate_if_needed(np.array([0.0, 11.6, 1.5]))

    assert advanced
    assert api.last_gate_pass_preserved_plan
    assert api.current_gate_idx == 1
    assert api.active_waypoints is not None
    assert api.target_manager.locked
    assert api.active_target_track_id == -2
    np.testing.assert_allclose(api.current_gate_pos, np.array([0.0, 20.0, 1.5]))


def test_single_gate_corridor_preserves_exit_segment_after_pass():
    api = PyAIPilotAutonomyAPI(use_perception=False, race_gate_count=2)
    api.horizon_continuation_enabled = True
    api.post_gate_exit_continuation_enabled = True
    api.gate_corridor_enabled = True
    api.gate_corridor_length_m = 3.0
    api.gate_centers_neu = [np.array([0.0, 10.0, 1.5])]
    api.gate_track_ids = [-1]
    api.current_gate_idx = 0

    planned = api._path_plan(
        pos=np.array([0.0, 0.0, 1.5]),
        vel=np.zeros(3),
    )

    assert planned
    assert api.active_plan_mode == "single_gate_corridor"
    assert api.active_horizon_gate_indices == [0]

    advanced = api._advance_gate_if_needed(np.array([0.0, 9.72, 1.5]))

    assert not advanced
    assert api.current_gate_idx == 0
    assert api.active_plan_mode == "single_gate_corridor"

    advanced = api._advance_gate_if_needed(np.array([0.0, 11.6, 1.5]))

    assert advanced
    assert api.last_gate_pass_preserved_plan
    assert api.active_waypoints is not None
    assert api.active_plan_mode == "single_gate_corridor"
    assert api.last_planned_gate_idx == 1
    assert api.post_gate_exit_reason == "waiting_for_future_gate_after_pass"
    assert api._post_gate_exit_active()


def test_reference_motion_yaw_after_center_crossing_does_not_look_back():
    api = PyAIPilotAutonomyAPI(use_perception=False, race_gate_count=2)
    api.yaw_reference_motion_near_gate_enabled = False
    api.gate_corridor_enabled = True
    api.gate_corridor_length_m = 3.0
    api.gate_centers_neu = [np.array([0.0, 10.0, 1.5])]
    api.gate_track_ids = [-1]
    api.current_gate_idx = 0

    assert api._path_plan(pos=np.array([0.0, 0.0, 1.5]), vel=np.zeros(3))

    api.gate_plane_crossed = True
    yaw = api._desired_yaw(
        p_ref=np.array([0.0, 10.5, 1.5]),
        v_ref=np.array([0.0, 1.0, 0.0]),
        a_ref=np.zeros(3),
        pos=np.array([0.0, 10.5, 1.5]),
        current_yaw=0.0,
    )

    assert api.last_yaw_target_source == "reference_motion_after_center_crossing"
    assert math.isclose(yaw, math.pi / 2.0, abs_tol=1e-6)


def test_reference_tau_clamps_forward_when_vehicle_is_ahead_on_path():
    api = PyAIPilotAutonomyAPI(use_perception=False, race_gate_count=2)
    api.gate_corridor_enabled = True
    api.gate_corridor_length_m = 3.0
    api.gate_centers_neu = [np.array([0.0, 10.0, 1.5])]
    api.gate_track_ids = [-1]
    api.current_gate_idx = 0

    assert api._path_plan(pos=np.array([0.0, 0.0, 1.5]), vel=np.zeros(3))

    raw_tau = 0.55 * float(api.planner.total_time)
    vehicle_tau = min(float(api.planner.total_time), raw_tau + 0.75)
    vehicle_pos, _, _ = api.planner.sample(vehicle_tau)

    clamped_tau = api._reference_progress_clamped_tau(raw_tau, vehicle_pos)

    assert clamped_tau > raw_tau
    assert clamped_tau <= vehicle_tau + 0.02
    assert api.reference_tau_reason == "path_progress"
    assert api.reference_path_lag_m > api.reference_progress_clamp_tolerance_m


def test_near_plane_aperture_pass_advances_locked_gate_before_exact_plane():
    api = PyAIPilotAutonomyAPI(use_perception=False, race_gate_count=1)
    target = np.array([0.0, 10.0, 1.5])
    api.near_plane_pass_enabled = True
    api.near_plane_pass_back_tolerance_m = 0.35
    api.near_plane_pass_forward_tolerance_m = 0.15
    api.pass_radius_m = 1.25
    api.gate_pass_lateral_radius_m = 0.75
    api.gate_plane_tolerance_m = 0.05
    api.target_manager.lock_target(
        gate_idx=0,
        track_id=-1,
        center_neu=target,
        reason="test",
    )
    api._sync_target_manager_state()
    api.active_gate_normal = np.array([0.0, 1.0, 0.0])
    api.previous_gate_pass_position = np.array([0.0, 9.45, 1.5])

    advanced = api._advance_gate_if_needed(np.array([0.0, 9.72, 1.5]))

    assert advanced
    assert api.current_gate_idx == 1
    assert -1 in api.completed_track_ids


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


def test_race_order_collapses_duplicate_suffix_tracks():
    api = PyAIPilotAutonomyAPI(use_perception=True, race_gate_count=5)
    completed_a = _stable_track(10, np.array([0.0, 10.0, 1.5]))
    completed_b = _stable_track(11, np.array([0.0, 20.0, 1.5]))
    weaker_duplicate = _stable_track(
        199,
        np.array([-3.86, 274.39, 13.54]),
        hits=12,
        score=0.6,
    )
    better_duplicate = _stable_track(
        187,
        np.array([-3.96, 275.16, 13.49]),
        hits=30,
        score=0.95,
    )
    api.gate_memory.tracks = [
        completed_a,
        completed_b,
        weaker_duplicate,
        better_duplicate,
    ]
    api.current_gate_idx = 2
    api.completed_track_ids = {10, 11}
    api.race_order_track_ids = [10, 11, 199, 187]

    committed_by_id = {int(track.id): track for track in api.gate_memory.get_committed_tracks()}
    api._refresh_perception_race_order(
        stable_tracks=[weaker_duplicate, better_duplicate],
        committed_by_id=committed_by_id,
        current_pos=np.array([0.0, 25.0, 1.5]),
    )

    assert api.race_order_track_ids == [10, 11, 187]
    gates, track_ids = api._ordered_perception_gates(committed_by_id)
    assert track_ids == [10, 11, 187]
    np.testing.assert_allclose(gates[api.current_gate_idx], better_duplicate.filtered_center_world)


def test_race_order_prunes_stale_duplicate_suffix_before_gate_count_cap():
    api = PyAIPilotAutonomyAPI(use_perception=True, race_gate_count=10)
    completed = [
        _stable_track(5, np.array([-0.26, 32.80, 4.78])),
        _stable_track(4, np.array([1.50, 59.30, 10.87])),
        _stable_track(20, np.array([-1.96, 91.64, 10.32])),
        _stable_track(56, np.array([-2.53, 121.77, 11.87])),
        _stable_track(65, np.array([2.47, 150.72, 20.39])),
        _stable_track(115, np.array([-0.72, 181.80, 19.47])),
        _stable_track(127, np.array([2.37, 209.21, 18.29])),
        _stable_track(151, np.array([-0.66, 240.27, 14.16])),
    ]
    active = _stable_track(202, np.array([-3.95, 275.03, 13.49]))
    stale_duplicate = _stable_track(59, np.array([-2.53, 121.77, 11.87]))
    stale_time = time.time() - float(api.gate_memory.stale_time) - 1.0
    stale_duplicate.last_seen_time = stale_time
    stale_duplicate.obs_history[-1].timestamp = stale_time
    final_gate = _stable_track(225, np.array([-0.22, 301.05, 9.16]), hits=40)
    api.gate_memory.tracks = [*completed, active, stale_duplicate, final_gate]
    api.current_gate_idx = 8
    api.active_target_track_id = 202
    api.completed_track_ids = {track.id for track in completed}
    api.race_order_track_ids = [*(track.id for track in completed), 202, 59]

    committed_by_id = {
        int(track.id): track for track in api.gate_memory.get_committed_tracks()
    }
    api._refresh_perception_race_order(
        stable_tracks=api.gate_memory.get_stable_tracks(),
        committed_by_id=committed_by_id,
        current_pos=np.array([-3.0, 266.0, 13.4]),
    )

    assert len(api.race_order_track_ids) == api.race_gate_count
    assert api.race_order_track_ids[api.current_gate_idx] == 202
    assert api.race_order_track_ids[api.current_gate_idx + 1] == 225
    assert 59 not in api.race_order_track_ids
    gates, track_ids = api._ordered_perception_gates(committed_by_id)
    assert track_ids[api.current_gate_idx + 1] == 225
    np.testing.assert_allclose(gates[api.current_gate_idx + 1], final_gate.filtered_center_world)


def test_planning_horizon_skips_duplicate_future_waypoint():
    api = PyAIPilotAutonomyAPI(use_perception=False, race_gate_count=4)
    target = np.array([0.0, 10.0, 1.5])
    api.gate_centers_neu = [
        target.copy(),
        np.array([0.2, 10.3, 1.6]),
        np.array([0.0, 30.0, 1.5]),
    ]
    api.gate_track_ids = [199, 187, 300]
    api.current_gate_idx = 0

    targets, track_ids, gate_indices = api._planning_horizon_targets(
        0,
        target,
        199,
    )

    assert track_ids == [199, 300]
    assert gate_indices == [0, 2]
    np.testing.assert_allclose(targets[0], target)
    np.testing.assert_allclose(targets[1], np.array([0.0, 30.0, 1.5]))


def test_planning_horizon_appends_strong_uncommitted_future_track():
    api = PyAIPilotAutonomyAPI(use_perception=True, race_gate_count=4)
    api.planning_horizon_gates = 3
    api.provisional_next_gate_enabled = True
    api.provisional_next_gate_max_age_s = 1.5
    api.provisional_next_gate_max_lateral_m = 20.0
    target = np.array([0.0, 10.0, 1.5])
    future = _stable_track(
        44,
        np.array([0.0, 25.0, 1.5]),
        hits=3,
        committed=False,
        stable=False,
    )
    api.gate_memory.tracks = [future]
    api.gate_centers_neu = [target.copy()]
    api.gate_track_ids = [10]
    api.current_gate_idx = 0

    targets, track_ids, gate_indices = api._planning_horizon_targets(
        0,
        target,
        10,
        pos=np.array([0.0, 0.0, 1.5]),
        vel=np.array([0.0, 1.0, 0.0]),
    )

    assert track_ids == [10, 44]
    assert gate_indices == [0, 1]
    np.testing.assert_allclose(targets[0], target)
    np.testing.assert_allclose(targets[1], future.filtered_center_world)


def test_path_plan_uses_provisional_future_track_without_stealing_active_target():
    api = PyAIPilotAutonomyAPI(use_perception=True, race_gate_count=4)
    api.planning_horizon_gates = 3
    api.provisional_next_gate_enabled = True
    api.provisional_next_gate_max_age_s = 1.5
    target = np.array([0.0, 10.0, 1.5])
    future = _stable_track(
        44,
        np.array([0.0, 25.0, 1.5]),
        hits=3,
        committed=False,
        stable=False,
    )
    api.gate_memory.tracks = [future]
    api.gate_centers_neu = [target.copy()]
    api.gate_track_ids = [10]
    api.current_gate_idx = 0

    planned = api._path_plan(
        pos=np.array([0.0, 0.0, 1.5]),
        vel=np.zeros(3),
    )

    assert planned
    assert api.active_plan_mode == "gate_horizon"
    assert api.active_horizon_track_ids == [10, 44]
    assert api.active_horizon_gate_indices == [0, 1]
    assert api.active_target_track_id == 10
    assert api.target_manager.locked
    assert api.target_manager.active_target_track_id == 10
    np.testing.assert_allclose(api.current_gate_pos, target)


def test_plan_install_logs_boundary_continuity(capsys):
    api = PyAIPilotAutonomyAPI(use_perception=False, race_gate_count=4)
    api.gate_centers_neu = [
        np.array([0.0, 10.0, 1.5]),
        np.array([2.0, 20.0, 2.0]),
        np.array([-1.0, 30.0, 1.2]),
    ]
    api.gate_track_ids = [-1, -2, -3]
    api.current_gate_idx = 0

    planned = api._path_plan(
        pos=np.array([0.0, 0.0, 1.5]),
        vel=np.zeros(3),
    )

    assert planned
    output = capsys.readouterr().out
    lines = [
        line for line in output.splitlines()
        if line.startswith("plan_boundary_continuity ")
    ]
    assert lines
    first = lines[0]
    assert "boundary_i=1" in first
    assert "left_velocity_neu=" in first
    assert "right_velocity_neu=" in first
    assert "left_acceleration_neu=" in first
    assert "right_acceleration_neu=" in first
    assert "left_jerk_neu=" in first
    assert "right_jerk_neu=" in first
    assert "speed_at_waypoint=" in first
    assert "turn_angle_in_deg=" in first
    assert "turn_angle_out_deg=" in first
    assert "velocity_delta=" in first
    assert "acceleration_delta=" in first
    assert "jerk_delta=" in first


def test_adaptive_passthrough_velocity_uses_rounded_tangent_and_speed_cap():
    api = PyAIPilotAutonomyAPI(use_perception=False, race_gate_count=3)
    api.passthrough_velocity_enabled = True
    api.passthrough_velocity_mode = "adaptive"
    api.passthrough_speed_m_s = 0.5
    api.passthrough_speed_max_m_s = 1.2
    api.passthrough_turn_slowdown = 0.5
    waypoints = np.array(
        [
            [0.0, 0.0, 1.5],
            [0.0, 10.0, 1.5],
            [10.0, 10.0, 1.5],
        ]
    )

    velocities = api._compute_passthrough_waypoint_velocities(waypoints)

    assert velocities is not None
    expected_dir = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0)
    speed = float(np.linalg.norm(velocities[1]))
    np.testing.assert_allclose(velocities[1] / speed, expected_dir)
    np.testing.assert_allclose(speed, 0.9)


def test_fixed_passthrough_velocity_keeps_configured_speed():
    api = PyAIPilotAutonomyAPI(use_perception=False, race_gate_count=3)
    api.passthrough_velocity_enabled = True
    api.passthrough_velocity_mode = "fixed"
    api.passthrough_speed_m_s = 0.5
    api.passthrough_speed_max_m_s = 1.2
    waypoints = np.array(
        [
            [0.0, 0.0, 1.5],
            [0.0, 10.0, 1.5],
            [10.0, 10.0, 1.5],
        ]
    )

    velocities = api._compute_passthrough_waypoint_velocities(waypoints)

    assert velocities is not None
    np.testing.assert_allclose(np.linalg.norm(velocities[1]), 0.5)


def test_gate_corridor_waypoints_insert_enter_center_exit():
    api = PyAIPilotAutonomyAPI(use_perception=False, race_gate_count=1)
    api.gate_corridor_enabled = True
    api.gate_corridor_length_m = 3.0

    waypoints, roles = api._build_gate_corridor_waypoints(
        start=np.array([0.0, 0.0, 1.5]),
        targets=[np.array([0.0, 10.0, 1.5])],
        preferred_normals=[np.array([0.0, 1.0, 0.0])],
    )

    assert roles == ["start", "gate_enter", "gate_center", "gate_exit"]
    np.testing.assert_allclose(
        waypoints,
        np.array(
            [
                [0.0, 0.0, 1.5],
                [0.0, 8.5, 1.5],
                [0.0, 10.0, 1.5],
                [0.0, 11.5, 1.5],
            ]
        ),
    )


def test_gate_corridor_skips_enter_point_that_is_already_behind_reference():
    api = PyAIPilotAutonomyAPI(use_perception=False, race_gate_count=1)
    api.gate_corridor_enabled = True
    api.gate_corridor_length_m = 3.0

    waypoints, roles = api._build_gate_corridor_waypoints(
        start=np.array([0.0, 9.0, 1.5]),
        targets=[np.array([0.0, 10.0, 1.5])],
        preferred_normals=[np.array([0.0, 1.0, 0.0])],
    )

    assert roles == ["start", "gate_center", "gate_exit"]
    np.testing.assert_allclose(
        waypoints,
        np.array(
            [
                [0.0, 9.0, 1.5],
                [0.0, 10.0, 1.5],
                [0.0, 11.5, 1.5],
            ]
        ),
    )


def test_single_gate_path_plan_uses_corridor_and_stops_after_exit():
    api = PyAIPilotAutonomyAPI(use_perception=False, race_gate_count=1)
    api.gate_corridor_enabled = True
    api.gate_corridor_length_m = 3.0
    api.gate_centers_neu = [np.array([0.0, 10.0, 1.5])]
    api.gate_track_ids = [-1]
    api.current_gate_idx = 0

    planned = api._path_plan(
        pos=np.array([0.0, 0.0, 1.5]),
        vel=np.zeros(3),
    )

    assert planned
    assert api.active_plan_mode == "single_gate_corridor"
    np.testing.assert_allclose(
        api.active_waypoints,
        np.array(
            [
                [0.0, 0.0, 1.5],
                [0.0, 8.5, 1.5],
                [0.0, 10.0, 1.5],
                [0.0, 11.5, 1.5],
            ]
        ),
        atol=1e-9,
    )
    np.testing.assert_allclose(api.active_terminal_velocity, np.zeros(3))


def test_longitudinal_active_shift_is_deferred_before_gate_enter():
    api = PyAIPilotAutonomyAPI(use_perception=True, race_gate_count=2)
    api.gate_corridor_enabled = True
    api.gate_corridor_length_m = 3.0
    api.active_target_shift_enabled = True
    api.active_target_shift_required_frames = 1
    api.active_target_shift_threshold_m = 0.45
    api.active_target_shift_defer_longitudinal_enabled = True
    api.active_target_shift_longitudinal_min_m = 0.5
    api.active_target_shift_longitudinal_lateral_max_m = 0.5
    api.gate_centers_neu = [np.array([0.0, 10.0, 1.5])]
    api.gate_track_ids = [10]
    api.current_gate_idx = 0

    assert api._path_plan(pos=np.array([0.0, 0.0, 1.5]), vel=np.zeros(3))
    original_generation = api.active_plan_generation
    original_waypoints = api.active_waypoints.copy()
    api.gate_memory.tracks = [_stable_track(10, np.array([0.0, 12.0, 1.5]))]

    replanned = api._maybe_apply_active_target_shift(
        pos=np.array([0.0, 2.0, 1.5]),
        vel=np.zeros(3),
    )

    assert not replanned
    assert api.active_plan_generation == original_generation
    np.testing.assert_allclose(api.active_waypoints, original_waypoints)
    assert api.deferred_longitudinal_shift_samples == [2.0]
    np.testing.assert_allclose(api.gate_centers_neu[0], np.array([0.0, 10.0, 1.5]))


def test_deferred_longitudinal_shift_moves_only_exit_after_gate_enter():
    api = PyAIPilotAutonomyAPI(use_perception=True, race_gate_count=2)
    api.gate_corridor_enabled = True
    api.gate_corridor_length_m = 3.0
    api.active_target_shift_enabled = True
    api.active_target_shift_required_frames = 1
    api.active_target_shift_threshold_m = 0.45
    api.active_target_shift_defer_longitudinal_enabled = True
    api.active_target_shift_longitudinal_min_m = 0.5
    api.active_target_shift_longitudinal_lateral_max_m = 0.5
    api.active_target_shift_longitudinal_exit_shift_max_m = 3.0
    api.gate_centers_neu = [np.array([0.0, 10.0, 1.5])]
    api.gate_track_ids = [10]
    api.current_gate_idx = 0

    assert api._path_plan(pos=np.array([0.0, 0.0, 1.5]), vel=np.zeros(3))
    api.gate_memory.tracks = [_stable_track(10, np.array([0.0, 12.0, 1.5]))]

    replanned = api._maybe_apply_active_target_shift(
        pos=np.array([0.0, 8.5, 1.5]),
        vel=np.array([0.0, 1.0, 0.0]),
    )

    assert replanned
    assert api.active_plan_mode == "single_gate_corridor_exit_shift"
    assert api.active_waypoint_roles == ["start", "gate_center", "gate_exit_shifted"]
    np.testing.assert_allclose(api.active_waypoints[1], np.array([0.0, 10.0, 1.5]))
    np.testing.assert_allclose(api.active_waypoints[2], np.array([0.0, 13.5, 1.5]))
    np.testing.assert_allclose(api.gate_centers_neu[0], np.array([0.0, 10.0, 1.5]))

    advanced = api._advance_gate_if_needed(np.array([0.0, 9.72, 1.5]))

    assert not advanced
    assert api.current_gate_idx == 0

    advanced = api._advance_gate_if_needed(np.array([0.0, 13.6, 1.5]))

    assert advanced
    assert api.current_gate_idx == 1


class _FakePlanner:
    total_time = 1.0
    times = np.array([1.0])
    segment_starts = np.array([0.0, 1.0])

    def __init__(self, z_at_crossing: float):
        self.z_at_crossing = float(z_at_crossing)

    def sample(self, t: float):
        y = -1.0 + 2.0 * float(t)
        p = np.array([0.0, y, self.z_at_crossing])
        v = np.array([0.0, 2.0, 0.0])
        a = np.zeros(3)
        return p, v, a


class _BacktrackingPlanner:
    total_time = 1.0
    times = np.array([1.0])
    segment_starts = np.array([0.0, 1.0])

    def sample(self, t: float):
        t = float(t)
        if t <= 0.25:
            y = -1.0 + 2.0 * t
            vy = 2.0
        else:
            y = -0.5 - 2.0 * (t - 0.25)
            vy = -2.0
        p = np.array([0.0, y, 0.0])
        v = np.array([0.0, vy, 0.0])
        a = np.zeros(3)
        return p, v, a


class _LoopingPlanner:
    total_time = 4.0
    times = np.array([4.0])
    segment_starts = np.array([0.0, 4.0])
    waypoints = np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])

    def sample(self, t: float):
        t = float(np.clip(t, 0.0, self.total_time))
        omega = math.pi / self.total_time
        x = 5.0 * math.sin(omega * t)
        y = -1.0 + 2.0 * t / self.total_time
        p = np.array([x, y, 0.0])
        v = np.array([5.0 * omega * math.cos(omega * t), 2.0 / self.total_time, 0.0])
        a = np.array([-5.0 * omega * omega * math.sin(omega * t), 0.0, 0.0])
        return p, v, a


class _PolylineUTurnPlanner:
    total_time = 10.0
    times = np.array([5.0, 5.0])
    segment_starts = np.array([0.0, 5.0, 10.0])
    waypoints = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 5.0, 0.0],
        ]
    )

    def sample(self, t: float):
        t = float(np.clip(t, 0.0, self.total_time))
        if t <= 5.0:
            y = 2.0 * t
            vy = 2.0
        else:
            y = 10.0 - (t - 5.0)
            vy = -1.0
        p = np.array([0.0, y, 0.0])
        v = np.array([0.0, vy, 0.0])
        a = np.zeros(3)
        return p, v, a


def test_active_gate_plan_validation_rejects_first_crossing_outside_corridor():
    api = PyAIPilotAutonomyAPI(use_perception=False, race_gate_count=1)

    valid, details = api._validate_active_gate_plan_crossing(
        planner=_FakePlanner(z_at_crossing=2.0),
        target=np.zeros(3),
        normal=np.array([0.0, 1.0, 0.0]),
        plan_mode="gate_horizon",
        gate_idx=0,
        track_id=1,
    )

    assert not valid
    assert details["reason"].startswith("lateral_error_too_large")
    assert details["lateral_error_m"] > api.gate_pass_lateral_radius_m


def test_active_gate_plan_validation_accepts_first_crossing_inside_corridor():
    api = PyAIPilotAutonomyAPI(use_perception=False, race_gate_count=1)

    valid, details = api._validate_active_gate_plan_crossing(
        planner=_FakePlanner(z_at_crossing=0.0),
        target=np.zeros(3),
        normal=np.array([0.0, 1.0, 0.0]),
        plan_mode="gate_horizon",
        gate_idx=0,
        track_id=1,
    )

    assert valid
    assert details["reason"] in {"crossed_gate_plane", "on_gate_plane"}


def test_active_gate_plan_validation_rejects_backward_progress_before_crossing():
    api = PyAIPilotAutonomyAPI(use_perception=False, race_gate_count=1)

    valid, details = api._validate_active_gate_plan_crossing(
        planner=_BacktrackingPlanner(),
        target=np.zeros(3),
        normal=np.array([0.0, 1.0, 0.0]),
        plan_mode="single_gate_exit",
        gate_idx=0,
        track_id=1,
    )

    assert not valid
    assert details["reason"] == "backward_progress_before_crossing"
    assert details["backward_progress_m"] >= 0.5


def test_plan_shape_validation_rejects_loop_far_from_waypoint_chain():
    api = PyAIPilotAutonomyAPI(use_perception=False, race_gate_count=1)

    valid, details = api._validate_active_gate_plan_crossing(
        planner=_LoopingPlanner(),
        target=np.array([0.0, 1.0, 0.0]),
        normal=np.array([0.0, 1.0, 0.0]),
        plan_mode="single_gate",
        gate_idx=0,
        track_id=1,
    )

    assert not valid
    assert details["reason"] == "corridor_deviation_too_large"
    assert details["corridor_m"] > api.plan_validation_max_corridor_m


def test_plan_shape_validation_allows_course_level_u_turn_waypoint_chain():
    api = PyAIPilotAutonomyAPI(use_perception=False, race_gate_count=1)

    valid, details = api._validate_minimum_snap_plan_shape(
        planner=_PolylineUTurnPlanner(),
        plan_mode="gate_horizon",
        gate_idx=0,
        track_id=1,
    )

    assert valid
    assert details["reason"] == "shape_validation_ok"
    assert details["polyline_backtrack_m"] == 0.0


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


def test_ordered_perception_gates_preserves_active_slot_with_last_center():
    api = PyAIPilotAutonomyAPI(use_perception=True, race_gate_count=9)
    completed = [
        _stable_track(10 + idx, np.array([0.0, 20.0 + 20.0 * idx, 2.0]))
        for idx in range(7)
    ]
    active = _stable_track(
        157,
        np.array([-0.57, 242.92, 14.01]),
        hits=20,
        stable=False,
    )
    active.ever_stable = True
    active.filtered_center_world = None
    future = _stable_track(209, np.array([-0.19, 302.06, 9.09]))
    api.gate_memory.tracks = [*completed, active, future]
    api.current_gate_idx = 7
    api.active_target_track_id = 157
    api.last_active_target_center = active.center.copy()
    api.race_order_track_ids = [*(track.id for track in completed), 157, 209]
    api.completed_track_ids = {track.id for track in completed}

    committed_by_id = {
        int(track.id): track for track in api.gate_memory.get_committed_tracks()
    }
    gates, track_ids = api._ordered_perception_gates(committed_by_id)

    assert track_ids[api.current_gate_idx] == 157
    assert track_ids[api.current_gate_idx + 1] == 209
    np.testing.assert_allclose(
        gates[api.current_gate_idx],
        api.last_active_target_center,
    )


def test_track_filtered_center_keeps_ever_stable_track_after_latest_outlier():
    api = PyAIPilotAutonomyAPI(use_perception=True, race_gate_count=10)
    center = np.array([-0.19, 302.06, 9.09])
    track = _stable_track(209, center, hits=12)
    track.is_stable = False
    track.ever_stable = True
    track.obs_history.append(
        GateObservation(
            timestamp=time.time(),
            center_world=np.array([3.0, 303.0, 9.0]),
            reprojection_error=0.1,
            confidence=1.0,
            keypoint_conf_min=1.0,
            keypoint_conf_mean=1.0,
            quality_ok=True,
            is_outlier=True,
        )
    )

    filtered, quality = api._track_filtered_center_for_navigation(track)

    assert quality["ok"]
    assert quality["last_observation_outlier"]
    np.testing.assert_allclose(filtered, center)


def test_closer_plausible_blocker_includes_stable_uncommitted_track():
    api = PyAIPilotAutonomyAPI(use_perception=True, race_gate_count=10)
    api.race_order_front_blocker_enabled = True
    first = _stable_track(
        153,
        np.array([-2.9, 260.9, 14.3]),
        hits=20,
        committed=True,
        stable=False,
    )
    closer = _stable_track(
        161,
        np.array([-0.7, 244.7, 13.8]),
        hits=14,
        committed=False,
        stable=True,
    )
    api.gate_memory.tracks = [first, closer]
    committed_by_id = {153: first}

    blocker_id, details = api._closer_plausible_blocker_for_target(
        153,
        [153],
        np.array([2.3, 211.8, 18.1]),
        committed_by_id,
    )

    assert blocker_id == 161
    assert details["reason"] == "stable_uncommitted_candidate"
    assert (
        details["distance"] + api.race_order_front_blocker_margin_m
        < details["first_dist"]
    )


def test_provisional_next_gate_prefers_retained_stable_uncommitted_candidate():
    api = PyAIPilotAutonomyAPI(use_perception=True, race_gate_count=10)
    api.provisional_next_gate_enabled = True
    api.provisional_next_gate_min_hits = 2
    api.provisional_next_gate_max_age_s = 1.0
    api.provisional_next_gate_max_lateral_m = 20.0
    api.gate_memory.stale_time = 3.0
    api.current_gate_idx = 7
    now = time.time()
    closer = _stable_track(
        161,
        np.array([-0.7, 244.7, 13.8]),
        hits=14,
        committed=False,
        stable=True,
    )
    closer.last_seen_time = now - 2.0
    closer.obs_history[-1].timestamp = closer.last_seen_time
    farther = _stable_track(
        153,
        np.array([-2.9, 260.9, 14.3]),
        hits=28,
        committed=True,
        stable=False,
    )
    api.gate_memory.tracks = [closer, farther]

    track, center, details = api._select_provisional_next_gate_candidate(
        np.array([2.3, 211.8, 18.1]),
        np.array([0.0, 1.0, 0.0]),
    )

    assert track is closer
    np.testing.assert_allclose(center, closer.filtered_center_world)
    assert details["stable_retained"] == 1.0


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
