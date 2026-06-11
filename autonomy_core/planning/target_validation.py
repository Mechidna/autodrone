"""Target validation helpers used through AutonomyAPI wrappers."""

import time

import numpy as np


def canonical_track_id(api, track_id):
    if track_id is None:
        return None
    track_id = int(track_id)
    while track_id in api.track_id_aliases:
        track_id = int(api.track_id_aliases[track_id])
    return track_id


def validate_perception_gate_center(api, center, current_pos):
    center = np.asarray(center, dtype=float).reshape(3)
    current_pos = np.asarray(current_pos, dtype=float).reshape(3)

    if not np.all(np.isfinite(center)):
        return False, "non_finite_center"

    if center[2] < api.safe_min_target_z:
        return False, f"z_below_safe_min:{center[2]:.2f}"

    if center[2] > api.safe_max_target_z:
        return False, f"z_above_safe_max:{center[2]:.2f}"

    dist = float(np.linalg.norm(center - current_pos))
    if dist > api.max_detection_range:
        return False, f"detection_too_far:{dist:.2f}"

    if api.last_valid_target is not None:
        jump = float(np.linalg.norm(center - api.last_valid_target))
        if jump > api.max_gate_jump:
            return False, f"gate_jump_too_large:{jump:.2f}"

    return True, ""


def is_near_completed_gate(api, center, radius=None):
    center = np.asarray(center, dtype=float).reshape(3)
    radius = api.completed_gate_position_radius if radius is None else float(radius)
    for completed in api.completed_gate_positions_this_cycle:
        if float(np.linalg.norm(center - completed)) < radius:
            return True
    return False


def find_duplicate_committed_track(api, center, track_id=None, radius=None):
    center = np.asarray(center, dtype=float).reshape(3)
    radius = api.gate_memory.commit_radius if radius is None else float(radius)
    for tr in api.gate_memory.get_committed_tracks():
        if track_id is not None and tr.id == track_id:
            continue
        if float(np.linalg.norm(center - tr.center)) < radius:
            return tr
    return None


def validate_planning_target(api, center):
    center = np.asarray(center, dtype=float).reshape(3)
    if not np.all(np.isfinite(center)):
        return False, "non_finite_target"
    if center[2] < api.safe_min_target_z:
        return False, f"target_z_below_safe_min:{center[2]:.2f}"
    if center[2] > api.safe_max_target_z:
        return False, f"target_z_above_safe_max:{center[2]:.2f}"
    return True, ""


def validate_candidate_target(api, center, current_pos, track_id=None):
    """
    Generic perception target validation.

    This deliberately avoids course geometry assumptions. It only rejects
    unsafe/numeric targets, completed landmarks, duplicate landmarks, and
    candidate jumps that are implausible given elapsed time since the last
    completed gate.
    """
    center = np.asarray(center, dtype=float).reshape(3)
    current_pos = np.asarray(current_pos, dtype=float).reshape(3)
    api.candidate_track_id = track_id
    api.candidate_center = center.copy()
    api.candidate_order_score = float("nan")

    valid, reason = api.validate_planning_target(center)
    if not valid:
        api.rejected_wrong_order = True
        return False, reason

    if api.is_near_completed_gate(center):
        api.rejected_completed_this_lap = True
        return False, "completed_this_lap"

    duplicate = api.find_duplicate_committed_track(
        center,
        track_id=track_id,
        radius=api.gate_memory.commit_radius,
    )
    if duplicate is not None:
        api.rejected_duplicate = True
        return False, f"duplicate_committed_track:{duplicate.id}"

    dist_from_vehicle = float(np.linalg.norm(center - current_pos))
    api.candidate_order_score = -dist_from_vehicle
    if dist_from_vehicle > api.max_detection_range:
        api.rejected_wrong_order = True
        return False, f"candidate_too_far_from_vehicle:{dist_from_vehicle:.2f}"

    if (
        api.last_completed_valid_gate_position is not None
        and api.last_completed_valid_gate_time is not None
    ):
        elapsed = max(0.0, time.time() - api.last_completed_valid_gate_time)
        jump = float(np.linalg.norm(center - api.last_completed_valid_gate_position))
        max_jump = api.gate_jump_margin + api.max_plausible_gate_speed * elapsed
        api.candidate_order_score = max_jump - jump
        if jump > max_jump:
            api.rejected_wrong_order = True
            return False, f"kinematic_jump_too_large:{jump:.2f}>{max_jump:.2f}"

    return True, ""
