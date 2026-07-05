"""Gate advancement helper functions used by AutonomyAPI wrappers."""

import time

import numpy as np


def reset_crossing_debug(api):
    api.crossing_true_gate_center = np.full(3, np.nan, dtype=float)
    api.crossing_vehicle_position = np.full(3, np.nan, dtype=float)
    api.crossing_error = np.full(3, np.nan, dtype=float)
    api.crossing_lateral_error_xz = float("nan")


def compute_gate_pass_geometry(api, position, target):
    position = np.asarray(position, dtype=float).reshape(3)
    target = np.asarray(target, dtype=float).reshape(3)

    if api.approach_vector is None or not np.all(np.isfinite(api.approach_vector)):
        api.gate_progress_along_approach = float("nan")
        api.gate_lateral_error = float("nan")
        api.gate_plane_crossed = False
        return False, "missing_approach_vector"

    rel = position - target
    progress = float(np.dot(rel, api.approach_vector))
    lateral_vec = rel - progress * api.approach_vector
    lateral_error = float(np.linalg.norm(lateral_vec))

    previous_progress = api.previous_gate_progress_along_approach
    crossed = previous_progress is not None and previous_progress <= 0.0 <= progress

    api.gate_progress_along_approach = progress
    api.gate_lateral_error = lateral_error
    api.gate_plane_crossed = bool(crossed)

    passed_beyond = progress > api.gate_progress_threshold
    inside_gate_radius = lateral_error < api.gate_pass_radius
    complete = inside_gate_radius and (passed_beyond or crossed)

    api.previous_gate_progress_along_approach = progress

    if complete:
        api.near_gate_but_not_crossed = False
        return True, "crossed_gate_plane" if crossed else "past_gate_center"

    if api.distance_to_active_target <= api.race_progression.pass_radius:
        api.near_gate_but_not_crossed = True
        if not inside_gate_radius:
            return False, f"lateral_error_too_large:{lateral_error:.2f}"
        return False, f"not_past_gate_plane:{progress:.2f}"

    api.near_gate_but_not_crossed = False
    return False, "not_near_gate"


def clear_active_perception_target(api, reason=""):
    """
    Remove the completed/stale perception target from all navigation hooks.

    This is deliberately perception-only. If no valid next gate is
    available, the controller should hold stable attitude instead of
    continuing to track or yaw toward the completed gate.
    """
    if not api.use_perception:
        return

    api.target_clear_reason = reason

    api.active_target_gates = []
    api.active_target_track_ids = []
    api.current_target_idx = 0
    api.current_target_gate = None
    api.current_gate_pos = None
    api.last_valid_target = None
    api.active_waypoints = None
    api.active_times = None
    api.p_ref = None
    api.v_ref = None
    api.a_ref = None
    api.active_target_center = None
    api.active_target_center_at_plan = None
    api.active_target_latest_filtered_center = None
    api.active_target_shift_m = float("nan")
    api.active_target_shift_frames = 0
    api.active_target_shift_replan_triggered = False
    api.pending_active_target_correction = None
    api.approach_start_position = None
    pos = np.array([
        api.telemetry.pos["x"],
        api.telemetry.pos["y"],
        api.telemetry.pos["z"],
    ], dtype=float)
    api.perception_hold_position = pos.copy()
    telemetry_yaw = float(api.telemetry.rpy["yaw"])
    api.perception_hold_yaw = api.get_perception_yaw_hold_reference(telemetry_yaw)
    if reason in ("gate_completed", "completed_target_in_control"):
        api.post_completion_grace_until = time.time() + api.no_target_grace_s
        api.post_completion_grace_active = True
    api.hold_anchor = np.array([pos[0], pos[1], pos[2]], dtype=float)
    api.hold_anchor_source = "completion_altitude"
    api.completed_gate_reference_blocked = True
    api.active_target_source = "cleared"
    api.active_target_track_id = None
    api.active_target_cleared = True
    api.next_valid_target_found = False
    api.target_retained_after_completion = False
    api._reset_pending_suffix_state(f"active_target_cleared:{reason}")
    print(f"[TARGET CLEAR] perception active target cleared reason={reason}")
