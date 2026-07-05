"""Pending-suffix planning helpers used by AutonomyAPI.

These functions intentionally operate on the AutonomyAPI facade instance.
That keeps all pending_suffix_* fields and public wrapper methods compatible
while moving the suffix-specific implementation out of the monolith.
"""

from __future__ import annotations

import time

import numpy as np

from autonomy_core.planning.minimum_snap_planner_multi_time_optimized import MultiSegmentMinimumSnapPlanner

def reset_pending_suffix_state(self, rejected_reason=""):
    reason = str(rejected_reason or "")
    self.pending_suffix_planner = None
    self.pending_suffix_track_ids = []
    self.pending_suffix_waypoints = None
    self.pending_suffix_times = None
    self.pending_suffix_splice_track_id = None
    self.pending_suffix_splice_tau = float("nan")
    self.pending_suffix_splice_target_idx = -1
    self.pending_suffix_splice_state = None
    self.pending_suffix_created_reason = ""
    self.pending_suffix_valid = False
    self.pending_suffix_created = False
    self.pending_suffix_rejected_reason = reason
    self.pending_suffix_waypoint_types = []
    self.pending_suffix_cleared_reason = reason

def prepare_pending_suffix_for_future_only_replan(self, replan_reason):
    """
    Build a future-only suffix from the current active-gate crossing state.
    This must not replace the active planner or reset its timing.
    """
    self._reset_pending_suffix_state()
    self.pending_suffix_installed = False
    self.future_only_replan_preserved_active_segment = False
    self.future_only_replan_reason = str(replan_reason or "")
    self.replan_suppressed_reason = ""

    if replan_reason not in (
        "tentative_lookahead_new_candidate",
        "tentative_lookahead_shift",
        "new_committed_or_stable_gate",
    ):
        self.pending_suffix_rejected_reason = "not_future_only_replan"
        return False
    if not self.use_perception:
        self.pending_suffix_rejected_reason = "perception_disabled"
        return False
    if self.planner is None or getattr(self.planner, "coeffs", None) is None:
        self.pending_suffix_rejected_reason = "missing_active_planner"
        return False
    if self.active_times is None or len(self.active_target_gates) == 0:
        self.pending_suffix_rejected_reason = "missing_active_horizon"
        return False
    if not (0 <= self.current_target_idx < len(self.active_target_track_ids)):
        self.pending_suffix_rejected_reason = "invalid_current_target_idx"
        return False

    active_track_id = self.canonical_track_id(
        self.active_target_track_ids[self.current_target_idx]
    )
    if active_track_id is None or active_track_id < 0:
        self.pending_suffix_rejected_reason = "invalid_active_track_id"
        return False

    active_times = np.asarray(self.active_times, dtype=float).reshape(-1)
    if self.current_target_idx >= len(active_times):
        self.pending_suffix_rejected_reason = "missing_active_crossing_time"
        return False
    splice_tau = self.active_target_crossing_tau(self.current_target_idx)
    planner_total = float(getattr(self.planner, "total_time", 0.0))
    if not np.isfinite(splice_tau) or splice_tau < 0.0 or splice_tau > planner_total + 1e-6:
        self.pending_suffix_rejected_reason = "invalid_splice_tau"
        return False

    try:
        p_splice, v_splice, a_splice, j_splice, s_splice = self.planner.sample_full(splice_tau)
    except AttributeError:
        p_splice, v_splice, a_splice = self.planner.sample(splice_tau)
        j_splice = np.zeros(3, dtype=float)
        s_splice = np.zeros(3, dtype=float)
    p_splice = np.asarray(p_splice, dtype=float).reshape(3)
    v_splice = np.asarray(v_splice, dtype=float).reshape(3)
    a_splice = np.asarray(a_splice, dtype=float).reshape(3)
    j_splice = np.asarray(j_splice, dtype=float).reshape(3)
    s_splice = np.asarray(s_splice, dtype=float).reshape(3)
    if not (
        np.all(np.isfinite(p_splice))
        and np.all(np.isfinite(v_splice))
        and np.all(np.isfinite(a_splice))
        and np.all(np.isfinite(j_splice))
    ):
        self.pending_suffix_rejected_reason = "non_finite_splice_state"
        return False

    snapshot = {
        "active_target_gates": [g.copy() for g in self.active_target_gates],
        "active_target_track_ids": list(self.active_target_track_ids),
        "current_target_idx": int(self.current_target_idx),
        "current_gate_pos": None
        if self.current_gate_pos is None
        else np.asarray(self.current_gate_pos, dtype=float).copy(),
        "last_valid_target": None
        if self.last_valid_target is None
        else np.asarray(self.last_valid_target, dtype=float).copy(),
        "active_target_track_id": self.active_target_track_id,
        "active_target_center_at_plan": None
        if self.active_target_center_at_plan is None
        else np.asarray(self.active_target_center_at_plan, dtype=float).copy(),
        "active_target_source": self.active_target_source,
        "active_waypoints": None
        if self.active_waypoints is None
        else np.asarray(self.active_waypoints, dtype=float).copy(),
        "active_times": None
        if self.active_times is None
        else np.asarray(self.active_times, dtype=float).copy(),
        "trajectory_start_time": self.trajectory_start_time,
        "previous_sample_tau_used": self.previous_sample_tau_used,
        "previous_sample_tau_plan_id": self.previous_sample_tau_plan_id,
        "planner": self.planner,
        "active_plan_id": self.active_plan_id,
        "race_cursor": self.race_progression.cursor,
        "race_lap": self.race_progression.lap,
    }

    pos = np.array([
        self.telemetry.pos["x"],
        self.telemetry.pos["y"],
        self.telemetry.pos["z"],
    ], dtype=float)

    try:
        _, target_gates, target_track_ids = self.build_waypoint_horizon(
            pos,
            max_gates_ahead=3,
        )
        target_waypoint_types = list(
            self._planning_target_waypoint_types[:len(target_track_ids)]
        )
    finally:
        self.active_target_gates = [
            g.copy() for g in snapshot["active_target_gates"]
        ]
        self.active_target_track_ids = list(snapshot["active_target_track_ids"])
        self.current_target_idx = int(snapshot["current_target_idx"])
        self.current_gate_pos = (
            None
            if snapshot["current_gate_pos"] is None
            else snapshot["current_gate_pos"].copy()
        )
        self.last_valid_target = (
            None
            if snapshot["last_valid_target"] is None
            else snapshot["last_valid_target"].copy()
        )
        self.active_target_track_id = snapshot["active_target_track_id"]
        self.active_target_center_at_plan = (
            None
            if snapshot["active_target_center_at_plan"] is None
            else snapshot["active_target_center_at_plan"].copy()
        )
        self.active_target_source = snapshot["active_target_source"]
        self.active_waypoints = (
            None
            if snapshot["active_waypoints"] is None
            else snapshot["active_waypoints"].copy()
        )
        self.active_times = (
            None
            if snapshot["active_times"] is None
            else snapshot["active_times"].copy()
        )
        self.trajectory_start_time = snapshot["trajectory_start_time"]
        self.previous_sample_tau_used = snapshot["previous_sample_tau_used"]
        self.previous_sample_tau_plan_id = snapshot["previous_sample_tau_plan_id"]
        self.planner = snapshot["planner"]
        self.active_plan_id = snapshot["active_plan_id"]
        self.race_progression.cursor = snapshot["race_cursor"]
        self.race_progression.lap = snapshot["race_lap"]

    target_track_ids = [
        self.canonical_track_id(tid) for tid in target_track_ids
    ]
    if len(target_track_ids) == 0 or target_track_ids[0] != active_track_id:
        self.pending_suffix_rejected_reason = "active_target_changed"
        return False
    current_active_gate = np.asarray(
        self.active_target_gates[self.current_target_idx], dtype=float
    ).reshape(3)
    proposed_active_gate = np.asarray(target_gates[0], dtype=float).reshape(3)
    if (
        not np.all(np.isfinite(current_active_gate))
        or not np.all(np.isfinite(proposed_active_gate))
        or float(np.linalg.norm(proposed_active_gate - current_active_gate)) > 0.25
    ):
        self.pending_suffix_rejected_reason = "active_target_center_changed"
        return False
    if len(target_gates) < 2:
        self.pending_suffix_rejected_reason = "no_future_suffix_targets"
        return False

    future_gates = [
        np.asarray(gate, dtype=float).reshape(3).copy()
        for gate in target_gates[1:]
    ]
    future_track_ids = list(target_track_ids[1:])
    future_waypoint_types = list(target_waypoint_types[1:])
    if len(future_gates) == 0 or len(future_track_ids) == 0:
        self.pending_suffix_rejected_reason = "empty_future_suffix"
        return False

    suffix_waypoints = np.vstack([p_splice] + future_gates)
    suffix_times = self.allocate_segment_times(
        suffix_waypoints,
        current_vel=v_splice,
        vmax=2.5,
        amax=2.0,
        T_min=1.0,
    )
    waypoint_velocities = self.compute_passthrough_waypoint_velocities(suffix_waypoints)
    suffix_planner = MultiSegmentMinimumSnapPlanner()
    suffix_planner.update(
        waypoints=suffix_waypoints,
        times=suffix_times,
        v_start=v_splice,
        v_end=np.zeros(3, dtype=float),
        a_start=a_splice,
        a_end=np.zeros(3, dtype=float),
        j_start=j_splice,
        j_end=np.zeros(3, dtype=float),
        waypoint_velocities=waypoint_velocities,
    )
    validation_ok, validation_debug = self.validate_minimum_snap_geometry(
        suffix_planner,
        suffix_waypoints,
    )
    if not validation_ok:
        self.pending_suffix_rejected_reason = (
            f"validation_failed:{validation_debug.get('reason', '')}"
        )
        return False

    self.pending_suffix_planner = suffix_planner
    self.pending_suffix_track_ids = list(future_track_ids)
    self.pending_suffix_waypoints = suffix_waypoints.copy()
    self.pending_suffix_times = np.asarray(suffix_times, dtype=float).copy()
    self.pending_suffix_waypoint_types = list(future_waypoint_types)
    self.pending_suffix_splice_track_id = active_track_id
    self.pending_suffix_splice_tau = float(splice_tau)
    self.pending_suffix_splice_target_idx = int(self.current_target_idx)
    self.pending_suffix_splice_state = {
        "tau": splice_tau,
        "p": p_splice.copy(),
        "v": v_splice.copy(),
        "a": a_splice.copy(),
        "j": j_splice.copy(),
        "s": s_splice.copy(),
    }
    self.pending_suffix_created_reason = str(replan_reason)
    self.pending_suffix_valid = True
    self.pending_suffix_created = True
    self.pending_suffix_rejected_reason = ""
    self.pending_suffix_cleared_reason = ""
    self.future_only_replan_preserved_active_segment = True
    self.replan_suppressed_reason = "future_only_pending_suffix_created"
    print(
        "[PENDING SUFFIX] created "
        f"reason={replan_reason} splice_track_id={active_track_id} "
        f"future_track_ids={future_track_ids}"
    )
    return True

def install_pending_suffix_after_completion(self, completed_track_id, next_track_id, pos):
    self.pending_suffix_installed = False
    if not self.pending_suffix_valid or self.pending_suffix_planner is None:
        if not self.pending_suffix_rejected_reason:
            self.pending_suffix_rejected_reason = "no_valid_pending_suffix"
        return False

    completed_id = self.canonical_track_id(completed_track_id)
    next_id = self.canonical_track_id(next_track_id)
    if self.canonical_track_id(self.pending_suffix_splice_track_id) != completed_id:
        self._reset_pending_suffix_state("splice_track_mismatch")
        return False
    if len(self.pending_suffix_track_ids) == 0:
        self.pending_suffix_rejected_reason = "suffix_has_no_targets"
        return False
    if next_id is not None and self.canonical_track_id(self.pending_suffix_track_ids[0]) != next_id:
        self.pending_suffix_rejected_reason = "first_suffix_track_not_next_target"
        return False

    suffix_start = np.asarray(self.pending_suffix_waypoints[0], dtype=float).reshape(3)
    pos = np.asarray(pos, dtype=float).reshape(3)
    if float(np.linalg.norm(pos - suffix_start)) > 2.0:
        self.pending_suffix_rejected_reason = "vehicle_far_from_suffix_start"
        return False

    self.planner = self.pending_suffix_planner
    self.active_waypoints = np.asarray(self.pending_suffix_waypoints, dtype=float).copy()
    self.active_times = np.asarray(self.pending_suffix_times, dtype=float).copy()
    self.active_target_track_ids = list(self.pending_suffix_track_ids)
    self.active_target_gates = [
        np.asarray(g, dtype=float).reshape(3).copy()
        for g in self.active_waypoints[1:]
    ]
    self.current_target_idx = 0
    self.current_gate_pos = self.active_target_gates[0].copy()
    self.last_valid_target = self.current_gate_pos.copy()
    self.active_target_track_id = self.canonical_track_id(
        self.active_target_track_ids[0]
    )
    self.active_target_center_at_plan = self.current_gate_pos.copy()
    self.active_target_source = "pending_suffix_after_completion"
    self.next_valid_target_found = True
    self.next_target_installed_same_cycle = True
    self.target_retained_after_completion = True
    self.active_target_cleared = False
    self.no_active_target = False
    self.completed_gate_reference_blocked = False
    self.skipped_target_clear_after_completion = True
    self.target_clear_reason = ""
    self.post_completion_grace_until = 0.0
    self.post_completion_grace_active = False
    self.post_completion_grace_suppressed = True
    self.planning_horizon_track_ids = list(self.pending_suffix_track_ids)
    self.planning_horizon_waypoint_count = int(len(self.active_waypoints))
    self.planning_horizon_waypoints = ";".join(
        f"{i}:{wp[0]:.2f},{wp[1]:.2f},{wp[2]:.2f}"
        for i, wp in enumerate(self.active_waypoints)
    )
    waypoint_types = list(getattr(self, "pending_suffix_waypoint_types", []))
    if len(waypoint_types) != len(self.active_target_track_ids):
        waypoint_types = ["pending_suffix"] * len(self.active_target_track_ids)
    self.planning_horizon_waypoint_types = " ".join(["start"] + waypoint_types)
    self._planning_target_waypoint_types = list(waypoint_types)
    self.trajectory_start_time = time.time()
    self.previous_sample_tau_used = 0.0
    self.previous_sample_tau_plan_id = None
    self.set_active_perception_target_geometry(self.current_gate_pos, pos)
    self.pending_suffix_installed = True
    self.pending_suffix_rejected_reason = ""
    installed_splice_track_id = completed_id
    installed_splice_tau = float(self.pending_suffix_splice_tau)
    installed_splice_target_idx = int(self.pending_suffix_splice_target_idx)
    self.record_installed_plan_for_export(
        plan_source="pending_suffix_install",
        replan_reason=self.pending_suffix_created_reason,
    )
    self._reset_pending_suffix_state("installed")
    self.pending_suffix_installed = True
    self.pending_suffix_splice_track_id = installed_splice_track_id
    self.pending_suffix_splice_tau = installed_splice_tau
    self.pending_suffix_splice_target_idx = installed_splice_target_idx
    self.pending_suffix_cleared_reason = "installed"
    self.pending_suffix_rejected_reason = ""
    print(
        "[PENDING SUFFIX] installed "
        f"splice_track_id={completed_id} track_ids={self.active_target_track_ids}"
    )
    return True
