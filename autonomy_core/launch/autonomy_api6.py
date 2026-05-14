import numpy as np
import time
import math
from autonomy_core.planning.minimum_snap_planner_multi_time_optimized import MultiSegmentMinimumSnapPlanner
from autonomy_core.launch.get_telemetry import GetTelemetry
from autonomy_core.controller.attitude_controller3 import RPGHighLevelTracker
from autonomy_core.perception.gate_perception import GatePerception
from autonomy_core.perception.gate_perception_node import GatePerceptionNode
from autonomy_core.perception.gate_memory import GateMemory
from autonomy_core.launch.race_progression import RaceProgression
from dataclasses import dataclass


def compute_desired_yaw(v_ref, a_ref, last_yaw, eps=1e-3):
    v_xy = np.array(v_ref[:2], dtype=float)
    a_xy = np.array(a_ref[:2], dtype=float)

    if np.linalg.norm(v_xy) > eps:
        return np.arctan2(v_xy[1], v_xy[0])

    if np.linalg.norm(a_xy) > eps:
        return np.arctan2(a_xy[1], a_xy[0])

    return last_yaw


def wrap_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class State:
    pos: np.ndarray
    vel: np.ndarray
    yaw: float


@dataclass
class Reference:
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    yaw: float
    yaw_rate: float = 0.0


class AutonomyAPI:
    def __init__(self, use_perception=False, race_gate_count=None, race_gate_order=None):
        self.use_perception = use_perception

        self.gate_perception = GatePerception() if use_perception else None
        self.perception_node = GatePerceptionNode(self.gate_perception) if use_perception else None
        self.gate_memory = GateMemory(
            association_radius=2.0,
            commit_radius=2.0,
            new_track_block_radius=4.5,
            min_confidence_per_hit=0.5,
            commit_hits=3,
            commit_confidence_sum=2.0,
            stale_time=5.0,
            alpha=0.35,
        )

        self.current_gate_pos = np.array([0.0, 0.0, 0.0], dtype=float)
        self.p_ref = None
        self.v_ref = None
        self.a_ref = None
        self.ref_yaw = None
        self.current_target_gate = None
        self.gate_confidence = 0.0
        self.planner = MultiSegmentMinimumSnapPlanner()
        self.telemetry = GetTelemetry()

        self.replan_time = 0.0
        self.trajectory_start_time = 0.0
        self.time_elapsed = 0.0
        self.last_control_time = None
        self.error_z = 0.0
        self.last_desired_yaw = 0.0

        self.active_waypoints = None
        self.active_times = None
        self.active_target_gates = []
        self.active_target_track_ids = []
        self.current_target_idx = 0

        # Race progress is a sequence cursor over persistent gate track IDs.
        # Landmark memory stays forever; passing a gate never deletes it.
        self.race_progression = RaceProgression(
            race_order=race_gate_order,
            pass_radius=1.25,
            clear_radius=1.75,
            advance_debounce_s=0.75,
            allow_laps=True,
        )

        # Kept for debug/log compatibility with older runner code.
        self.completed_track_ids_this_cycle = set()
        self.completed_gate_events_this_cycle = 0
        self.race_gate_count = race_gate_count

        self.tracker = RPGHighLevelTracker(
            mass=1.0,
            gravity=9.81,
            kp=(2.5, 2.5, 3.5),
            kv=(2.0, 2.0, 2.6),
            max_tilt_deg=20.0,
            max_acc_xy=2.0,
            max_acc_z_up=2.5,
            max_acc_z_down=2.0,
            thrust_hover=0.74,
            thrust_min=0.60,
            thrust_max=1.0,
        )

        # Ground-truth gates for planner debugging / fallback
        self.gt_gates = [
            np.array([0.0, 8.0, 1.5], dtype=float),
            np.array([0.8, 16.0, 1.5], dtype=float),
            np.array([-0.8, 24.0, 1.5], dtype=float),
            np.array([0.8, 16.0, 1.5], dtype=float),
            np.array([0.0, 8.0, 1.5], dtype=float),
            np.array([0.0, 0.0, 1.5], dtype=float),
        ]
        self.current_gate_idx = 0
        self.last_planned_gate_idx = -1
        self.last_plan_started_at = None
        self.last_plan_finished_at = None
        self.last_plan_duration = 0.0
        self.last_plan_mode = None
        self.last_plan_start_gate_idx = None
        self.last_plan_end_gate_idx = None
        self.safe_min_target_z = 1.0
        self.safe_max_target_z = 3.0
        self.max_detection_range = 25.0
        self.max_gate_jump = 12.0
        self.last_valid_target = None
        self.last_raw_gate_center = None
        self.last_perception_accepted = False
        self.last_perception_rejection_reason = ""
        self.last_target_z_clamped = False
        self.last_perception_replan_trigger = False
        self.completed_gate_positions_this_cycle = []
        self.completed_gate_position_radius = 1.5
        self.distance_to_active_target = float("nan")
        self.gate_completion_triggered = False
        self.completion_reason = ""
        self.completed_gate_position = None
        self.active_gate_idx_before = 0
        self.active_gate_idx_after = 0
        self.race_cursor_before = 0
        self.race_cursor_after = 0
        self.active_target_source = ""
        self.target_rejected_completed = False
        self.last_completed_valid_gate_position = None
        self.last_completed_valid_gate_time = None
        self.rejected_perception_track_ids = set()
        self.perception_single_lap_no_reset = True
        self.max_plausible_gate_speed = 12.0
        self.gate_jump_margin = 12.0
        self.perception_hold_position = None
        self.perception_hold_yaw = 0.0
        self.no_active_target = False
        self.no_target_control_mode = ""
        self.hold_anchor_source = ""
        self.hold_anchor = None
        self.velocity_damping_active = False
        self.completed_gate_reference_blocked = False
        self.p_ref_source = ""
        self.yaw_hold_value = float("nan")
        self.no_target_grace_s = 0.75
        self.post_completion_grace_until = 0.0
        self.post_completion_grace_active = False
        self.no_target_roll_source = ""
        self.no_target_pitch_source = ""
        self.horizontal_hold_disabled_after_completion = False
        self.previous_yaw_cmd = None
        self.previous_yaw_cmd_log = float("nan")
        self.last_yaw_cmd_time = None
        self.max_yaw_rate = np.deg2rad(90.0)
        self.raw_yaw_cmd = float("nan")
        self.yaw_cmd_after_unwrap = float("nan")
        self.yaw_rate_limited = False
        self.track_id = None
        self.merged_into_track_id = None
        self.duplicate_merge_reason = ""
        self.race_order_track_ids = []
        self.race_order_inserted = False
        self.race_order_rejected_reason = ""
        self.landmark_uncertainty = float("nan")
        self.track_observations = 0
        self.completed_unique_gate_count = 0
        self.active_gate_idx_clamped_by_race_gate_count = False
        self.suspected_duplicate_track = False
        self.track_id_aliases = {}
        self.race_accepted_track_ids = []
        self.committed_track_centers_log = ""
        self.pairwise_committed_track_distances = ""
        self.duplicate_radius_used = float("nan")
        self.merge_candidate_pairs = ""
        self.merge_blocked_reason = ""
        self.rejected_track_temporary_vs_permanent = ""
        self.active_target_admission_status = ""
        self.race_order_after_merge = []
        self.last_raw_image_corners = None
        self.last_ordered_image_corners = None
        self.last_pnp_rvec = None
        self.last_pnp_tvec = None
        self.last_gate_center_camera = None
        self.last_gate_center_body = None
        self.last_gate_center_world_debug = None
        self.last_gate_normal_world = None
        self.last_reprojection_error = float("nan")
        self.last_detection_drone_pose = None
        self.image_width = 0
        self.image_height = 0
        self.corner_margin_px = 25.0
        self.min_corner_x = float("nan")
        self.max_corner_x = float("nan")
        self.min_corner_y = float("nan")
        self.max_corner_y = float("nan")
        self.corner_margin_ok = False
        self.clipped_detection_rejected = False
        self.rejected_near_image_edge = False
        self.track_update_innovation = float("nan")
        self.track_update_accepted = False
        self.track_center_before_update = None
        self.track_center_after_update = None
        self.candidate_track_id = None
        self.candidate_center = None
        self.candidate_order_score = float("nan")
        self.rejected_wrong_order = False
        self.rejected_duplicate = False
        self.rejected_completed_this_lap = False
        self.race_cursor_advanced = False
        self.active_gate_idx_advanced = False
        self.completed_landmark_count = 0
        self.lap_reset_triggered = False
        self.active_target_cleared = False
        self.active_target_track_id = None
        self.completed_gate_track_id = None
        self.yaw_target_source = ""
        self.target_retained_after_completion = False
        self.next_valid_target_found = False
        self.valid_candidate_count = 0
        self.active_target_center = None
        self.approach_start_position = None
        self.approach_vector = None
        self.previous_gate_progress_along_approach = None
        self.gate_progress_along_approach = float("nan")
        self.gate_lateral_error = float("nan")
        self.gate_plane_crossed = False
        self.near_gate_but_not_crossed = False
        self.completion_blocked_reason = ""
        self.gate_pass_radius = 0.75
        self.gate_progress_threshold = 0.2

    # -------------------------------------------------------------------------
    # Time allocation helpers
    # -------------------------------------------------------------------------

    def choose_T(self, p0, v0, p1, vmax=2.5, amax=2.0, T_min=1.0):
        dp = p1 - p0
        d = np.linalg.norm(dp)

        if d < 1e-6:
            return T_min

        dir_vec = dp / d
        v_along = np.dot(v0, dir_vec)

        t_acc = vmax / amax
        d_acc = 0.5 * amax * t_acc**2

        if d > 2 * d_acc:
            T_base = 2 * t_acc + (d - 2 * d_acc) / vmax
        else:
            T_base = 2 * np.sqrt(d / amax)

        if v_along < 0:
            T_base += min(abs(v_along) / amax, 2.0)
        else:
            T_base -= min(v_along / (2 * amax), 0.5)

        return max(T_base, T_min)

    def allocate_segment_times(self, waypoints, current_vel, vmax=2.5, amax=2.0, T_min=1.0):
        times = []

        for i in range(len(waypoints) - 1):
            p0 = waypoints[i]
            p1 = waypoints[i + 1]

            if i == 0:
                v0 = current_vel
            else:
                v0 = np.zeros(3, dtype=float)

            T = self.choose_T(p0, v0, p1, vmax=vmax, amax=amax, T_min=T_min)
            times.append(T)

        return np.asarray(times, dtype=float)

    def compute_final_exit_velocity(self, gates, default_speed=2.5):
        if len(gates) >= 2:
            d = gates[-1] - gates[-2]
        else:
            d = np.array([0.0, 0.0, 0.0], dtype=float)

        norm_d = np.linalg.norm(d)
        if norm_d < 1e-6:
            return np.array([0.0, default_speed, 0.0], dtype=float)

        return default_speed * (d / norm_d)

    # -------------------------------------------------------------------------
    # Perception / memory
    # -------------------------------------------------------------------------

    def process_frame(self, frame, camera_matrix, dist_coeffs):
        """
        Legacy convenience method.
        In perception mode, this updates gate memory and returns the latest detected gate center.
        In GT mode, returns the current GT target gate.
        """
        if not self.use_perception:
            return self.get_current_target_gate().copy()

        result = self.update_gate_memory_from_frame(frame, camera_matrix, dist_coeffs)

        if result is None or not result.get("accepted", False):
            return np.zeros(3, dtype=float)

        self.current_gate_pos = np.asarray(result["center"], dtype=float)
        return self.current_gate_pos.copy()

    def update_gate_memory_from_frame(self, frame, camera_matrix, dist_coeffs):
        if not self.use_perception or self.perception_node is None:
            return None

        drone_pos = np.array([
            self.telemetry.pos["x"],
            self.telemetry.pos["y"],
            self.telemetry.pos["z"],
        ], dtype=float)

        drone_rpy_rad = np.array([
            float(self.telemetry.rpy["roll"]),
            float(self.telemetry.rpy["pitch"]),
            float(self.telemetry.rpy["yaw"]),
        ], dtype=float) * np.pi / 180.0

        det = self.perception_node.detect_gate(
            frame=frame,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            drone_pos=drone_pos,
            drone_rpy_rad=drone_rpy_rad,
        )

        now = time.time()

        if det is None:
            self.gate_memory.prune(now)
            self.last_raw_gate_center = None
            self.last_perception_accepted = False
            self.last_perception_rejection_reason = "no_detection"
            return None

        self.gate_confidence = float(det["confidence"])
        raw_center = np.asarray(det["gate_center_world"], dtype=float).reshape(3)
        self.last_raw_gate_center = raw_center.copy()
        self.last_raw_image_corners = det.get("raw_corners", None)
        self.last_ordered_image_corners = det.get("ordered_corners", None)
        self.last_pnp_rvec = det.get("rvec", None)
        self.last_pnp_tvec = det.get("tvec", None)
        self.last_gate_center_camera = det.get("gate_center_camera", None)
        self.last_gate_center_body = det.get("gate_center_body", None)
        self.last_gate_center_world_debug = raw_center.copy()
        self.last_gate_normal_world = det.get("gate_normal_world", None)
        self.last_reprojection_error = float(det.get("reprojection_error", np.nan))
        self.last_detection_drone_pose = np.array([
            drone_pos[0],
            drone_pos[1],
            drone_pos[2],
            drone_rpy_rad[0],
            drone_rpy_rad[1],
            drone_rpy_rad[2],
        ], dtype=float)

        image_valid, image_reason = self.validate_detection_image_bounds(
            det.get("ordered_corners", None),
            frame.shape,
        )
        if not image_valid:
            self.gate_memory.prune(now)
            self.last_perception_accepted = False
            self.last_perception_rejection_reason = image_reason
            print(f"[PERCEPTION REJECT] reason={image_reason} corners={self.last_ordered_image_corners}")
            return {
                "accepted": False,
                "reason": image_reason,
                "track_id": None,
                "committed_now": False,
                "committed": False,
                "center": raw_center,
            }

        valid, reason = self.validate_perception_gate_center(raw_center, drone_pos)
        if not valid:
            self.gate_memory.prune(now)
            self.last_perception_accepted = False
            self.last_perception_rejection_reason = reason
            print(f"[PERCEPTION REJECT] reason={reason} raw_center={raw_center}")
            return {
                "accepted": False,
                "reason": reason,
                "track_id": None,
                "committed_now": False,
                "committed": False,
                "center": raw_center,
            }

        if self.is_near_completed_gate(raw_center, radius=self.gate_memory.new_track_block_radius):
            self.gate_memory.prune(now)
            self.last_perception_accepted = False
            self.last_perception_rejection_reason = "near_completed_landmark_this_lap"
            self.rejected_completed_this_lap = True
            self.target_rejected_completed = True
            print(f"[PERCEPTION REJECT] reason=near_completed_landmark_this_lap raw_center={raw_center}")
            return {
                "accepted": False,
                "reason": "near_completed_landmark_this_lap",
                "track_id": None,
                "committed_now": False,
                "committed": False,
                "center": raw_center,
            }

        result = self.gate_memory.add_detection(
            center=raw_center,
            confidence=det["confidence"],
            timestamp=now,
        )
        self.track_update_innovation = self.gate_memory.last_update_innovation
        self.track_update_accepted = self.gate_memory.last_update_accepted
        self.track_center_before_update = self.gate_memory.last_track_center_before
        self.track_center_after_update = self.gate_memory.last_track_center_after

        self.gate_memory.prune(now)
        merge_event = self.refresh_landmark_merges()
        if merge_event.get("merged", False) and result is not None:
            result_track_id = result.get("track_id")
            if result_track_id == merge_event.get("source_id"):
                result["track_id"] = merge_event.get("target_id")
                result["center"] = self.gate_memory.get_track_by_id(result["track_id"]).center.copy()

        if result is not None:
            self.track_id = result.get("track_id")
            tr = self.gate_memory.get_track_by_id(self.track_id) if self.track_id is not None else None
            if tr is not None:
                self.landmark_uncertainty = self.gate_memory.track_uncertainty(tr.id)
                self.track_observations = tr.hits
                if result.get("committed_now", False) or result.get("committed", False):
                    self.accept_track_into_race_order(tr)
            self.last_perception_accepted = bool(result.get("accepted", False))
            self.last_perception_rejection_reason = "" if self.last_perception_accepted else result.get("reason", "")
            print(
                f"[MEM] reason={result['reason']} "
                f"track_id={result['track_id']} "
                f"committed={result['committed']} "
                f"committed_now={result['committed_now']} "
                f"center={result['center']}"
            )

            committed_tracks = self.gate_memory.get_committed_tracks()
            print("[MEM] committed tracks:")
            for tr in committed_tracks:
                print(f"    id={tr.id}, center={tr.center}, hits={tr.hits}")

        return result

    def validate_detection_image_bounds(self, corners, frame_shape):
        self.image_height = int(frame_shape[0])
        self.image_width = int(frame_shape[1])
        self.min_corner_x = float("nan")
        self.max_corner_x = float("nan")
        self.min_corner_y = float("nan")
        self.max_corner_y = float("nan")
        self.corner_margin_ok = False
        self.clipped_detection_rejected = False
        self.rejected_near_image_edge = False

        if corners is None:
            return False, "missing_ordered_corners"

        pts = np.asarray(corners, dtype=float).reshape(-1, 2)
        if pts.shape[0] != 4 or not np.all(np.isfinite(pts)):
            return False, "invalid_ordered_corners"

        xs = pts[:, 0]
        ys = pts[:, 1]
        self.min_corner_x = float(np.min(xs))
        self.max_corner_x = float(np.max(xs))
        self.min_corner_y = float(np.min(ys))
        self.max_corner_y = float(np.max(ys))

        if (
            self.min_corner_x < 0.0
            or self.max_corner_x > self.image_width - 1
            or self.min_corner_y < 0.0
            or self.max_corner_y > self.image_height - 1
        ):
            self.clipped_detection_rejected = True
            return False, "clipped_detection_rejected"

        margin = float(self.corner_margin_px)
        if (
            self.min_corner_x < margin
            or self.max_corner_x > (self.image_width - 1 - margin)
            or self.min_corner_y < margin
            or self.max_corner_y > (self.image_height - 1 - margin)
        ):
            self.rejected_near_image_edge = True
            return False, "rejected_near_image_edge"

        self.corner_margin_ok = True
        return True, ""

    def validate_perception_gate_center(self, center, current_pos):
        center = np.asarray(center, dtype=float).reshape(3)
        current_pos = np.asarray(current_pos, dtype=float).reshape(3)

        if not np.all(np.isfinite(center)):
            return False, "non_finite_center"

        if center[2] < self.safe_min_target_z:
            return False, f"z_below_safe_min:{center[2]:.2f}"

        if center[2] > self.safe_max_target_z:
            return False, f"z_above_safe_max:{center[2]:.2f}"

        dist = float(np.linalg.norm(center - current_pos))
        if dist > self.max_detection_range:
            return False, f"detection_too_far:{dist:.2f}"

        if self.last_valid_target is not None:
            jump = float(np.linalg.norm(center - self.last_valid_target))
            if jump > self.max_gate_jump:
                return False, f"gate_jump_too_large:{jump:.2f}"

        return True, ""

    def is_near_completed_gate(self, center, radius=None):
        center = np.asarray(center, dtype=float).reshape(3)
        radius = self.completed_gate_position_radius if radius is None else float(radius)
        for completed in self.completed_gate_positions_this_cycle:
            if float(np.linalg.norm(center - completed)) < radius:
                return True
        return False

    def find_duplicate_committed_track(self, center, track_id=None, radius=None):
        center = np.asarray(center, dtype=float).reshape(3)
        radius = self.gate_memory.commit_radius if radius is None else float(radius)
        for tr in self.gate_memory.get_committed_tracks():
            if track_id is not None and tr.id == track_id:
                continue
            if float(np.linalg.norm(center - tr.center)) < radius:
                return tr
        return None

    def validate_planning_target(self, center):
        center = np.asarray(center, dtype=float).reshape(3)
        if not np.all(np.isfinite(center)):
            return False, "non_finite_target"
        if center[2] < self.safe_min_target_z:
            return False, f"target_z_below_safe_min:{center[2]:.2f}"
        if center[2] > self.safe_max_target_z:
            return False, f"target_z_above_safe_max:{center[2]:.2f}"
        return True, ""

    def validate_candidate_target(self, center, current_pos, track_id=None):
        """
        Generic perception target validation.

        This deliberately avoids course geometry assumptions. It only rejects
        unsafe/numeric targets, completed landmarks, duplicate landmarks, and
        candidate jumps that are implausible given elapsed time since the last
        completed gate.
        """
        center = np.asarray(center, dtype=float).reshape(3)
        current_pos = np.asarray(current_pos, dtype=float).reshape(3)
        self.candidate_track_id = track_id
        self.candidate_center = center.copy()
        self.candidate_order_score = float("nan")

        valid, reason = self.validate_planning_target(center)
        if not valid:
            self.rejected_wrong_order = True
            return False, reason

        if self.is_near_completed_gate(center):
            self.rejected_completed_this_lap = True
            return False, "completed_this_lap"

        duplicate = self.find_duplicate_committed_track(
            center,
            track_id=track_id,
            radius=self.gate_memory.commit_radius,
        )
        if duplicate is not None:
            self.rejected_duplicate = True
            return False, f"duplicate_committed_track:{duplicate.id}"

        dist_from_vehicle = float(np.linalg.norm(center - current_pos))
        self.candidate_order_score = -dist_from_vehicle
        if dist_from_vehicle > self.max_detection_range:
            self.rejected_wrong_order = True
            return False, f"candidate_too_far_from_vehicle:{dist_from_vehicle:.2f}"

        if (
            self.last_completed_valid_gate_position is not None
            and self.last_completed_valid_gate_time is not None
        ):
            elapsed = max(0.0, time.time() - self.last_completed_valid_gate_time)
            jump = float(np.linalg.norm(center - self.last_completed_valid_gate_position))
            max_jump = self.gate_jump_margin + self.max_plausible_gate_speed * elapsed
            self.candidate_order_score = max_jump - jump
            if jump > max_jump:
                self.rejected_wrong_order = True
                return False, f"kinematic_jump_too_large:{jump:.2f}>{max_jump:.2f}"

        return True, ""

    def canonical_track_id(self, track_id):
        if track_id is None:
            return None
        track_id = int(track_id)
        while track_id in self.track_id_aliases:
            track_id = int(self.track_id_aliases[track_id])
        return track_id

    def apply_landmark_merge_event(self, merge_event):
        self.duplicate_radius_used = self.gate_memory.duplicate_merge_radius
        self.pairwise_committed_track_distances = ";".join(
            f"{a}-{b}:{d:.2f}" for a, b, d in self.gate_memory.last_pairwise_distances
        )
        self.merge_candidate_pairs = ";".join(
            f"{a}-{b}:{d:.2f}" for a, b, d in self.gate_memory.last_merge_candidate_pairs
        )
        self.merge_blocked_reason = self.gate_memory.last_merge_blocked_reason
        if not merge_event or not merge_event.get("merged", False):
            self.merged_into_track_id = None
            self.duplicate_merge_reason = ""
            self.suspected_duplicate_track = False
            return

        source_id = int(merge_event["source_id"])
        target_id = int(merge_event["target_id"])
        self.track_id_aliases[source_id] = target_id
        self.merged_into_track_id = target_id
        self.duplicate_merge_reason = merge_event.get("reason", "")
        self.suspected_duplicate_track = True

        source_completed = source_id in self.completed_track_ids_this_cycle
        self.completed_track_ids_this_cycle.discard(source_id)
        if source_completed:
            self.completed_track_ids_this_cycle.add(target_id)
        self.race_accepted_track_ids = [
            self.canonical_track_id(tid) for tid in self.race_accepted_track_ids
        ]
        deduped = []
        for tid in self.race_accepted_track_ids:
            if tid is not None and tid not in deduped:
                deduped.append(tid)
        self.race_accepted_track_ids = deduped

        self.race_progression.inferred_order = [
            self.canonical_track_id(tid) for tid in self.race_progression.inferred_order
        ]
        order = []
        for tid in self.race_progression.inferred_order:
            if tid is not None and tid not in order:
                order.append(tid)
        self.race_progression.inferred_order = order
        if self.race_progression.cursor > len(order):
            self.race_progression.cursor = len(order)

        if source_id in self.active_target_track_ids:
            self.clear_active_perception_target(reason="active_target_merged_duplicate")

    def refresh_landmark_merges(self):
        merge_event = self.gate_memory.merge_duplicate_committed_tracks()
        self.apply_landmark_merge_event(merge_event)
        return merge_event

    def accept_track_into_race_order(self, tr):
        self.race_order_inserted = False
        self.race_order_rejected_reason = ""
        if tr is None or not tr.committed:
            self.race_order_rejected_reason = "track_not_committed"
            return False

        track_id = self.canonical_track_id(tr.id)
        self.track_id = track_id
        self.landmark_uncertainty = self.gate_memory.track_uncertainty(track_id)
        canonical_track = self.gate_memory.get_track_by_id(track_id)
        self.track_observations = 0 if canonical_track is None else canonical_track.hits
        self.rejected_track_temporary_vs_permanent = ""

        if track_id in self.race_accepted_track_ids:
            self.active_target_admission_status = "accepted"
            return True

        if self.track_observations < self.gate_memory.commit_hits:
            self.race_order_rejected_reason = "insufficient_observations"
            self.rejected_track_temporary_vs_permanent = "temporary"
            return False

        if (
            np.isfinite(self.landmark_uncertainty)
            and self.landmark_uncertainty > self.gate_memory.duplicate_merge_radius
        ):
            self.race_order_rejected_reason = f"landmark_uncertainty_too_high:{self.landmark_uncertainty:.2f}"
            self.rejected_track_temporary_vs_permanent = "temporary"
            return False

        if self.race_gate_count is not None and len(self.race_accepted_track_ids) >= self.race_gate_count:
            self.race_order_rejected_reason = "race_gate_count_reached"
            self.rejected_track_temporary_vs_permanent = "temporary"
            return False

        if self.is_near_completed_gate(tr.center, radius=self.gate_memory.duplicate_merge_radius):
            self.race_order_rejected_reason = "near_completed_unique_gate"
            self.rejected_track_temporary_vs_permanent = "temporary"
            self.suspected_duplicate_track = True
            return False

        duplicate = None
        for accepted_id in self.race_accepted_track_ids:
            accepted = self.gate_memory.get_track_by_id(accepted_id)
            if accepted is None:
                continue
            dist = float(np.linalg.norm(tr.center - accepted.center))
            if dist < self.gate_memory.duplicate_merge_radius:
                duplicate = accepted
                break
        if duplicate is not None:
            self.gate_memory.merge_track_into(track_id, duplicate.id, reason="race_order_duplicate")
            self.apply_landmark_merge_event(self.gate_memory.last_merge_event)
            self.race_order_rejected_reason = "duplicate_of_accepted_race_gate"
            self.rejected_track_temporary_vs_permanent = "merged"
            self.suspected_duplicate_track = True
            return False

        self.race_accepted_track_ids.append(track_id)
        self.race_progression.sync_committed_tracks([tr])
        self.race_order_inserted = True
        self.active_target_admission_status = "accepted"
        return True

    def refresh_race_order_from_memory(self):
        self.refresh_landmark_merges()
        committed_tracks = self.gate_memory.get_committed_tracks()
        self.committed_track_centers_log = ";".join(
            f"{tr.id}:{tr.center[0]:.2f},{tr.center[1]:.2f},{tr.center[2]:.2f}:h{tr.hits}"
            for tr in committed_tracks
        )
        for tr in committed_tracks:
            self.accept_track_into_race_order(tr)

        valid_ids = {tr.id for tr in committed_tracks}
        order = []
        for tid in self.race_progression.inferred_order:
            tid = self.canonical_track_id(tid)
            if tid is None or tid not in valid_ids:
                continue
            if tid not in self.race_accepted_track_ids:
                continue
            if tid not in order:
                order.append(tid)
        self.race_progression.inferred_order = order
        if self.race_progression.cursor > len(order):
            self.race_progression.cursor = len(order)
        self.race_order_track_ids = self.race_progression.order()
        self.race_order_after_merge = list(self.race_order_track_ids)
        return committed_tracks

    def clamp_reference_altitude(self, p_ref, v_ref, a_ref):
        self.last_target_z_clamped = False
        if p_ref[2] >= self.safe_min_target_z:
            return p_ref, v_ref, a_ref

        p_ref = np.array(p_ref, dtype=float)
        v_ref = np.array(v_ref, dtype=float)
        a_ref = np.array(a_ref, dtype=float)

        p_ref[2] = self.safe_min_target_z
        if v_ref[2] < 0.0:
            v_ref[2] = 0.0
        if a_ref[2] < 0.0:
            a_ref[2] = 0.0

        self.last_target_z_clamped = True
        return p_ref, v_ref, a_ref

    def get_committed_waypoints(self):
        committed = self.gate_memory.get_committed_centers()
        return [np.asarray(p, dtype=float) for p in committed]

    def set_active_perception_target_geometry(self, center, start_position):
        if not self.use_perception:
            return

        center = np.asarray(center, dtype=float).reshape(3)
        start_position = np.asarray(start_position, dtype=float).reshape(3)
        approach = center - start_position
        norm = float(np.linalg.norm(approach))

        self.active_target_center = center.copy()
        self.approach_start_position = start_position.copy()
        self.previous_gate_progress_along_approach = None
        self.gate_progress_along_approach = float("nan")
        self.gate_lateral_error = float("nan")
        self.gate_plane_crossed = False
        self.near_gate_but_not_crossed = False
        self.completion_blocked_reason = ""

        if norm < 1e-6:
            self.approach_vector = None
            return

        self.approach_vector = approach / norm

    def compute_gate_pass_geometry(self, position, target):
        position = np.asarray(position, dtype=float).reshape(3)
        target = np.asarray(target, dtype=float).reshape(3)

        if self.approach_vector is None or not np.all(np.isfinite(self.approach_vector)):
            self.gate_progress_along_approach = float("nan")
            self.gate_lateral_error = float("nan")
            self.gate_plane_crossed = False
            return False, "missing_approach_vector"

        rel = position - target
        progress = float(np.dot(rel, self.approach_vector))
        lateral_vec = rel - progress * self.approach_vector
        lateral_error = float(np.linalg.norm(lateral_vec))

        previous_progress = self.previous_gate_progress_along_approach
        crossed = previous_progress is not None and previous_progress <= 0.0 <= progress

        self.gate_progress_along_approach = progress
        self.gate_lateral_error = lateral_error
        self.gate_plane_crossed = bool(crossed)

        passed_beyond = progress > self.gate_progress_threshold
        inside_gate_radius = lateral_error < self.gate_pass_radius
        complete = inside_gate_radius and (passed_beyond or crossed)

        self.previous_gate_progress_along_approach = progress

        if complete:
            self.near_gate_but_not_crossed = False
            return True, "crossed_gate_plane" if crossed else "past_gate_center"

        if self.distance_to_active_target <= self.race_progression.pass_radius:
            self.near_gate_but_not_crossed = True
            if not inside_gate_radius:
                return False, f"lateral_error_too_large:{lateral_error:.2f}"
            return False, f"not_past_gate_plane:{progress:.2f}"

        self.near_gate_but_not_crossed = False
        return False, "not_near_gate"

    def clear_active_perception_target(self, reason=""):
        """
        Remove the completed/stale perception target from all navigation hooks.

        This is deliberately perception-only. If no valid next gate is
        available, the controller should hold stable attitude instead of
        continuing to track or yaw toward the completed gate.
        """
        if not self.use_perception:
            return

        self.active_target_gates = []
        self.active_target_track_ids = []
        self.current_target_idx = 0
        self.current_target_gate = None
        self.current_gate_pos = None
        self.last_valid_target = None
        self.active_waypoints = None
        self.active_times = None
        self.p_ref = None
        self.v_ref = None
        self.a_ref = None
        self.active_target_center = None
        self.approach_start_position = None
        pos = np.array([
            self.telemetry.pos["x"],
            self.telemetry.pos["y"],
            self.telemetry.pos["z"],
        ], dtype=float)
        self.perception_hold_position = pos.copy()
        telemetry_yaw = float(self.telemetry.rpy["yaw"]) * math.pi / 180.0
        self.perception_hold_yaw = telemetry_yaw if np.isfinite(telemetry_yaw) else 0.0
        if reason in ("gate_completed", "completed_target_in_control"):
            self.post_completion_grace_until = time.time() + self.no_target_grace_s
            self.post_completion_grace_active = True
        self.hold_anchor = np.array([pos[0], pos[1], pos[2]], dtype=float)
        self.hold_anchor_source = "completion_altitude"
        self.completed_gate_reference_blocked = True
        self.active_target_source = "cleared"
        self.active_target_track_id = None
        self.active_target_cleared = True
        self.next_valid_target_found = False
        self.target_retained_after_completion = False
        print(f"[TARGET CLEAR] perception active target cleared reason={reason}")

    # -------------------------------------------------------------------------
    # Target / horizon building
    # -------------------------------------------------------------------------

    def get_current_target_gate(self):
        """
        In GT mode: current GT gate by index.
        In perception mode: current active committed target gate, if available.
        """
        if self.use_perception:
            if 0 <= self.current_target_idx < len(self.active_target_gates):
                return self.active_target_gates[self.current_target_idx].copy()

            waypoints, target_gates, _ = self.build_waypoint_horizon_from_memory(
                current_pos=np.array([
                    self.telemetry.pos["x"],
                    self.telemetry.pos["y"],
                    self.telemetry.pos["z"],
                ], dtype=float),
                max_gates_ahead=3,
            )
            if len(target_gates) > 0:
                return target_gates[0].copy()

            return np.array([0.0, 0.0, 1.5], dtype=float)

        if self.current_gate_idx >= len(self.gt_gates):
            return self.gt_gates[-1].copy()
        return self.gt_gates[self.current_gate_idx].copy()

    def build_waypoint_horizon_from_memory(self, current_pos, max_gates_ahead=3):
        """
        Build planning horizon from explicit race progression.

        No spatial coordinate is used to order gates. A predefined race order
        wins if supplied; otherwise committed track IDs are appended in
        discovery order and the progression cursor advances through that list.

        Returns:
            waypoints: np.ndarray shape (N,3)
            target_gates: list[np.ndarray]
            target_track_ids: list[int]
        """
        committed_tracks = self.refresh_race_order_from_memory()
        self.race_progression.update_clearance(current_pos, self.gate_memory.get_track_by_id)
        self.target_rejected_completed = False
        self.rejected_wrong_order = False
        self.rejected_duplicate = False
        self.rejected_completed_this_lap = False
        self.candidate_track_id = None
        self.candidate_center = None
        self.candidate_order_score = float("nan")
        self.lap_reset_triggered = False
        self.next_valid_target_found = False
        self.valid_candidate_count = 0

        if len(committed_tracks) == 0:
            return np.array([current_pos], dtype=float), [], []

        order = self.race_progression.order()
        target_tracks = []
        selected_track_ids = set()
        first_selected_order_idx = None

        for order_idx in range(self.race_progression.cursor, len(order)):
            if len(target_tracks) >= max_gates_ahead:
                break

            track_id = order[order_idx]
            if track_id in selected_track_ids:
                continue

            tr = self.gate_memory.get_track_by_id(track_id)
            if tr is None or not tr.committed:
                # Sequence integrity matters more than horizon length. With a
                # predefined race order, do not skip an unavailable next gate
                # and accidentally plan to a later gate.
                break
            valid, reason = self.validate_planning_target(tr.center)
            if not valid:
                self.last_perception_accepted = False
                self.last_perception_rejection_reason = reason
                print(f"[TARGET REJECT] reason={reason} track_id={tr.id} center={tr.center}")
                break
            if self.is_near_completed_gate(tr.center):
                self.last_perception_accepted = False
                self.last_perception_rejection_reason = "already_completed_landmark"
                self.target_rejected_completed = True
                self.rejected_completed_this_lap = True
                print(f"[TARGET REJECT] reason=already_completed_landmark track_id={tr.id} center={tr.center}")
                continue
            if track_id not in self.race_accepted_track_ids:
                self.last_perception_accepted = False
                self.last_perception_rejection_reason = "track_not_admitted_to_race"
                self.active_target_admission_status = "rejected"
                print(f"[TARGET REJECT] reason=track_not_admitted_to_race track_id={tr.id} center={tr.center}")
                continue
            valid, reason = self.validate_candidate_target(tr.center, current_pos, track_id=tr.id)
            if not valid:
                self.last_perception_accepted = False
                self.last_perception_rejection_reason = reason
                print(f"[TARGET REJECT] reason={reason} track_id={tr.id} center={tr.center}")
                continue
            target_tracks.append(tr)
            selected_track_ids.add(track_id)
            self.valid_candidate_count += 1
            if first_selected_order_idx is None:
                first_selected_order_idx = order_idx

        if len(target_tracks) == 0:
            return np.array([current_pos], dtype=float), [], []

        if first_selected_order_idx is not None:
            self.race_progression.cursor = first_selected_order_idx

        target_gates = [tr.center.copy() for tr in target_tracks]
        target_track_ids = [tr.id for tr in target_tracks]
        if len(target_gates) > 0:
            self.last_valid_target = target_gates[0].copy()
            self.active_target_source = "memory_track"
            self.active_target_track_id = target_track_ids[0]
            self.next_valid_target_found = True

        return np.vstack([current_pos] + target_gates), target_gates, target_track_ids

    def build_waypoint_horizon_from_gt(self, current_pos, max_gates_ahead=3):
        if not self.use_perception:
            max_gates_ahead = 1

        remaining_gates = self.gt_gates[self.current_gate_idx:self.current_gate_idx + max_gates_ahead]
        if len(remaining_gates) == 0:
            return np.array([current_pos], dtype=float), [], []

        target_gates = [np.asarray(g, dtype=float) for g in remaining_gates]
        target_track_ids = [-1] * len(target_gates)  # no memory-track IDs in GT mode
        return np.vstack([current_pos] + target_gates), target_gates, target_track_ids

    def build_waypoint_horizon(self, current_pos, max_gates_ahead=3):
        """
        Unified horizon builder.
        Returns:
            waypoints: np.ndarray shape (N,3)
            target_gates: list[np.ndarray]
            target_track_ids: list[int]
        """
        if self.use_perception:
            return self.build_waypoint_horizon_from_memory(current_pos, max_gates_ahead=max_gates_ahead)
        return self.build_waypoint_horizon_from_gt(current_pos, max_gates_ahead=max_gates_ahead)

    # -------------------------------------------------------------------------
    # Gate advancement
    # -------------------------------------------------------------------------

    def advance_gate_if_needed(self, threshold=1.25):
        """
        Advance when the drone gets close enough to the current active target.
        Works for both GT mode and perception mode.
        """
        pos = np.array([
            self.telemetry.pos["x"],
            self.telemetry.pos["y"],
            self.telemetry.pos["z"],
        ], dtype=float)

        if self.use_perception:
            self.gate_completion_triggered = False
            self.completion_reason = ""
            self.completed_gate_position = None
            self.no_active_target = len(self.active_target_gates) == 0
            self.velocity_damping_active = False
            self.active_gate_idx_before = self.current_gate_idx
            self.race_cursor_before = self.race_progression.cursor
            self.race_cursor_advanced = False
            self.active_gate_idx_advanced = False
            self.lap_reset_triggered = False
            self.active_target_cleared = False
            self.completed_gate_track_id = None
            self.target_retained_after_completion = False
            self.gate_plane_crossed = False
            self.near_gate_but_not_crossed = False
            self.completion_blocked_reason = ""

            target = None
            track_id = None
            if 0 <= self.current_target_idx < len(self.active_target_gates):
                target = np.asarray(self.active_target_gates[self.current_target_idx], dtype=float).reshape(3)
                if 0 <= self.current_target_idx < len(self.active_target_track_ids):
                    track_id = self.canonical_track_id(self.active_target_track_ids[self.current_target_idx])
                    self.active_target_track_id = track_id
                self.active_target_source = "active_target_gates"
            elif self.last_valid_target is not None:
                target = np.asarray(self.last_valid_target, dtype=float).reshape(3)
                self.active_target_source = "last_valid_target"

            if target is None:
                self.distance_to_active_target = float("nan")
                self.active_gate_idx_after = self.current_gate_idx
                self.race_cursor_after = self.race_progression.cursor
                return False

            self.race_progression.pass_radius = float(threshold)
            self.distance_to_active_target = float(np.linalg.norm(pos - target))

            if self.is_near_completed_gate(target):
                self.completion_reason = "already_completed_target"
                self.completion_blocked_reason = self.completion_reason
                self.active_gate_idx_after = self.current_gate_idx
                self.race_cursor_after = self.race_progression.cursor
                return False

            valid, reason = self.validate_candidate_target(
                target,
                pos,
                track_id=track_id,
            )
            if not valid:
                self.completion_reason = reason
                self.completion_blocked_reason = reason
                self.active_gate_idx_after = self.current_gate_idx
                self.race_cursor_after = self.race_progression.cursor
                print(f"[GATE COMPLETE REJECT] reason={reason} track_id={track_id} target={target}")
                return False

            passed_gate, pass_reason = self.compute_gate_pass_geometry(pos, target)
            if not passed_gate:
                self.completion_reason = ""
                self.completion_blocked_reason = pass_reason
                self.active_gate_idx_after = self.current_gate_idx
                self.race_cursor_after = self.race_progression.cursor
                return False

            order = self.race_progression.order()
            current_sequence_id = None
            if self.race_progression.cursor < len(order):
                current_sequence_id = order[self.race_progression.cursor]
            if track_id is None or track_id < 0:
                self.completion_reason = "missing_track_id"
                self.completion_blocked_reason = self.completion_reason
                self.active_gate_idx_after = self.current_gate_idx
                self.race_cursor_after = self.race_progression.cursor
                return False
            if self.race_gate_count is not None and self.current_gate_idx >= self.race_gate_count:
                self.completion_reason = "race_gate_count_reached"
                self.completion_blocked_reason = self.completion_reason
                self.active_gate_idx_clamped_by_race_gate_count = True
                self.active_gate_idx_after = self.current_gate_idx
                self.race_cursor_after = self.race_progression.cursor
                return False
            if current_sequence_id != track_id:
                self.completion_reason = f"cursor_track_mismatch:{current_sequence_id}!={track_id}"
                self.completion_blocked_reason = self.completion_reason
                self.active_gate_idx_after = self.current_gate_idx
                self.race_cursor_after = self.race_progression.cursor
                print(
                    f"[GATE COMPLETE REJECT] reason={self.completion_reason} "
                    f"target={target}"
                )
                return False

            self.gate_completion_triggered = True
            self.completion_reason = pass_reason
            self.completion_blocked_reason = ""
            self.completed_gate_position = target.copy()
            self.completed_gate_track_id = track_id

            self.completed_track_ids_this_cycle.add(track_id)
            self.completed_gate_positions_this_cycle.append(target.copy())
            self.last_completed_valid_gate_position = target.copy()
            self.last_completed_valid_gate_time = time.time()
            self.race_progression.last_passed_track_id = track_id
            self.race_progression.waiting_for_clear_track_id = track_id
            self.race_progression.last_advance_time = self.last_completed_valid_gate_time
            self.race_progression.cursor += 1
            if len(order) > 0 and self.race_progression.cursor >= len(order):
                if self.perception_single_lap_no_reset:
                    self.race_progression.cursor = len(order)
                else:
                    self.race_progression.cursor = 0
                    self.race_progression.lap += 1
                    self.lap_reset_triggered = True
                    self.completed_track_ids_this_cycle.clear()
                    self.completed_gate_positions_this_cycle.clear()
                    self.completed_gate_events_this_cycle = 0
                    self.last_completed_valid_gate_position = None
                    self.last_completed_valid_gate_time = None
            self.race_cursor_advanced = self.race_progression.cursor != self.race_cursor_before

            print(
                f"Perception mode: completed active target track_id={track_id}; "
                f"distance={self.distance_to_active_target:.3f}, "
                f"cursor {self.race_cursor_before}->{self.race_progression.cursor}, "
                f"active_gate_idx {self.active_gate_idx_before}->{self.current_gate_idx + 1}"
            )

            self.completed_gate_events_this_cycle += 1
            if self.race_gate_count is not None:
                next_gate_idx = min(self.current_gate_idx + 1, self.race_gate_count)
                self.active_gate_idx_clamped_by_race_gate_count = next_gate_idx != self.current_gate_idx + 1
                self.current_gate_idx = next_gate_idx
            else:
                self.current_gate_idx += 1
            self.active_gate_idx_advanced = self.current_gate_idx != self.active_gate_idx_before
            self.completed_unique_gate_count = len(self.completed_track_ids_this_cycle)
            self.completed_landmark_count = len(self.completed_gate_positions_this_cycle)

            # A completed perception target must not remain active while the
            # stack waits for a valid next gate. Replanning will install the
            # next target if one passes validation.
            self.clear_active_perception_target(reason="gate_completed")
            self.active_gate_idx_after = self.current_gate_idx
            self.race_cursor_after = self.race_progression.cursor
            return True

        if self.current_gate_idx >= len(self.gt_gates) - 1:
            return False

        target = self.get_current_target_gate()
        dist = np.linalg.norm(pos - target)

        if dist < threshold:
            self.current_gate_idx += 1
            print(f"Advancing to gate {self.current_gate_idx}: {self.gt_gates[self.current_gate_idx]}")
            return True

        return False

    # -------------------------------------------------------------------------
    # Planning
    # -------------------------------------------------------------------------

    def path_plan(self, gate_xyz=None):
        """
        Multi-segment path plan:
        current position -> next gates in horizon

        In GT mode:
            uses only the current GT gate. This prevents the reference from
            racing ahead to future gates before actual gate completion.
        In perception mode:
            uses committed non-redundant gates from GateMemory,
            excluding only the gates completed in the current cycle.
        """
        plan_start = time.time()
        self.last_target_z_clamped = False
        self.last_plan_started_at = plan_start
        self.last_plan_mode = "gt_single_gate" if not self.use_perception else "perception_horizon"
        self.last_plan_start_gate_idx = self.current_gate_idx if not self.use_perception else None
        self.last_plan_end_gate_idx = None
        self.replan_time = time.time()

        pos = np.array([
            self.telemetry.pos["x"],
            self.telemetry.pos["y"],
            self.telemetry.pos["z"],
        ], dtype=float)

        vel = np.array([
            self.telemetry.vel["vx"],
            self.telemetry.vel["vy"],
            self.telemetry.vel["vz"],
        ], dtype=float)

        waypoints, target_gates, target_track_ids = self.build_waypoint_horizon(
            pos,
            max_gates_ahead=1 if not self.use_perception else 3,
        )
        if self.use_perception:
            target_track_ids = [self.canonical_track_id(tid) for tid in target_track_ids]

        print("=== REPLAN DEBUG ===")
        for tr in self.gate_memory.get_committed_tracks():
            print(f"track_id={tr.id}, center={tr.center}")
        print("active_target_track_ids:", self.active_target_track_ids)
        print("race_order:", self.race_progression.order())
        print("race_cursor:", self.race_progression.cursor)
        print("race_lap:", self.race_progression.lap)
        print("completed_track_ids_this_cycle:", sorted(self.completed_track_ids_this_cycle))
        print("completed_gate_events_this_cycle:", self.completed_gate_events_this_cycle)
        print("race_gate_count:", self.race_gate_count)
        print("telemetry pos:", pos)

        if len(waypoints) < 2 or len(target_gates) == 0:
            print("No valid gates available to plan to.")
            if self.use_perception:
                self.clear_active_perception_target(reason="no_valid_next_gate")
            self.last_plan_finished_at = time.time()
            self.last_plan_duration = self.last_plan_finished_at - plan_start
            return False

        # Keep the currently planned target list stable until next replan
        validated_target_gates = []
        validated_target_track_ids = []
        for gate, track_id in zip(target_gates, target_track_ids):
            if self.use_perception:
                valid, reason = self.validate_planning_target(gate)
                if not valid:
                    self.last_perception_accepted = False
                    self.last_perception_rejection_reason = reason
                    print(f"[PLAN TARGET REJECT] reason={reason} track_id={track_id} gate={gate}")
                    break
            gate = np.asarray(gate, dtype=float).copy()
            if gate[2] < self.safe_min_target_z:
                gate[2] = self.safe_min_target_z
                self.last_target_z_clamped = True
            validated_target_gates.append(gate)
            validated_target_track_ids.append(track_id)

        if self.use_perception and len(validated_target_gates) == 0:
            print("No valid perception target available; clearing existing trajectory.")
            self.clear_active_perception_target(reason="validated_target_empty")
            self.last_plan_finished_at = time.time()
            self.last_plan_duration = self.last_plan_finished_at - plan_start
            return False

        if len(validated_target_gates) > 0:
            target_gates = validated_target_gates
            target_track_ids = validated_target_track_ids
            waypoints = np.vstack([pos] + target_gates)

        self.active_target_gates = [g.copy() for g in target_gates]
        self.active_target_track_ids = list(target_track_ids)
        self.current_target_idx = 0
        if self.use_perception and len(target_gates) > 0:
            self.last_valid_target = target_gates[0].copy()
            self.active_target_track_id = target_track_ids[0] if len(target_track_ids) > 0 else None
            self.next_valid_target_found = True
            self.active_target_cleared = False
            self.set_active_perception_target_geometry(target_gates[0], pos)

        times_init = self.allocate_segment_times(
            waypoints,
            current_vel=vel,
            vmax=2.5,
            amax=2.0,
            T_min=1.0,
        )

        if self.use_perception:
            v_end = self.compute_final_exit_velocity(target_gates, default_speed=2.5)
        else:
            # Preset waypoint mode should stop at the active gate. A nonzero
            # exit velocity is what lets the reference continue past the gate.
            v_end = np.zeros(3, dtype=float)

        print("planning waypoint horizon:")
        for i, wp in enumerate(waypoints):
            print(f"  wp[{i}] = {wp}")
        print("target_track_ids:", self.active_target_track_ids)
        print("race_order:", self.race_progression.order())
        print("race_cursor:", self.race_progression.cursor)
        print("race_lap:", self.race_progression.lap)
        print("completed_track_ids_this_cycle:", sorted(self.completed_track_ids_this_cycle))
        print("completed_gate_events_this_cycle:", self.completed_gate_events_this_cycle)
        print("race_gate_count:", self.race_gate_count)
        print("initial segment times:", times_init)
        print("initial total horizon:", float(np.sum(times_init)))
        print("terminal v_end:", v_end)

        # Fixed-time trajectory generation is a small linear solve. The SciPy
        # outer time optimizer caused second-scale offboard loop stalls in
        # perception mode, so online replans use the deterministic allocation.
        times_opt = times_init
        self.planner.update(
            waypoints=waypoints,
            times=times_opt,
            v_start=vel,
            v_end=v_end,
            a_start=np.zeros(3, dtype=float),
            a_end=np.zeros(3, dtype=float),
            j_start=np.zeros(3, dtype=float),
            j_end=np.zeros(3, dtype=float),
        )

        print("fixed segment times:", times_opt)
        print("fixed total horizon:", float(np.sum(times_opt)))

        self.current_gate_pos = waypoints[1].copy()
        self.active_waypoints = waypoints.copy()
        self.active_times = np.asarray(times_opt, dtype=float).copy()
        self.trajectory_start_time = time.time()

        if not self.use_perception:
            self.last_planned_gate_idx = self.current_gate_idx
            self.last_plan_end_gate_idx = self.current_gate_idx

        total_time = self.planner.total_time
        for tau in np.linspace(0.0, total_time, 8):
            p, v, a = self.planner.sample(tau)
            print("tau =", tau)
            print("p =", p)
            print("v =", v)
            print("a =", a)

        self.last_plan_finished_at = time.time()
        self.last_plan_duration = self.last_plan_finished_at - plan_start
        print("planner duration:", self.last_plan_duration)
        return True

    # -------------------------------------------------------------------------
    # Control
    # -------------------------------------------------------------------------

    def continuous_yaw_command(self, raw_yaw, fallback_yaw):
        now = time.time()
        raw_yaw = float(raw_yaw)
        if not np.isfinite(raw_yaw):
            raw_yaw = float(fallback_yaw)

        self.raw_yaw_cmd = raw_yaw
        self.previous_yaw_cmd_log = (
            float("nan") if self.previous_yaw_cmd is None else self.previous_yaw_cmd
        )

        if self.previous_yaw_cmd is None or not np.isfinite(self.previous_yaw_cmd):
            self.previous_yaw_cmd = raw_yaw
            self.last_yaw_cmd_time = now
            self.yaw_cmd_after_unwrap = raw_yaw
            self.yaw_rate_limited = False
            return raw_yaw

        dt = now - self.last_yaw_cmd_time if self.last_yaw_cmd_time is not None else 0.02
        dt = min(max(dt, 1e-3), 0.2)
        unwrapped = self.previous_yaw_cmd + wrap_pi(raw_yaw - self.previous_yaw_cmd)
        delta = unwrapped - self.previous_yaw_cmd
        max_delta = self.max_yaw_rate * dt

        self.yaw_rate_limited = abs(delta) > max_delta
        if self.yaw_rate_limited:
            delta = float(np.clip(delta, -max_delta, max_delta))

        limited = self.previous_yaw_cmd + delta
        self.previous_yaw_cmd = limited
        self.last_yaw_cmd_time = now
        self.yaw_cmd_after_unwrap = limited
        return limited

    def hold_no_target_control(self, state, current_yaw_rad):
        """
        Perception-only safety fallback for "no valid next gate".

        It publishes finite references and uses the normal tracker for altitude
        feedback and horizontal velocity damping without anchoring XY to the
        completed gate/completion point behind the vehicle.
        """
        if self.perception_hold_position is None:
            self.perception_hold_position = state.pos.copy()
        if not np.isfinite(self.perception_hold_yaw):
            self.perception_hold_yaw = current_yaw_rad

        now = time.time()
        self.post_completion_grace_active = now < self.post_completion_grace_until
        hold_z = float(self.perception_hold_position[2])
        if not np.isfinite(hold_z):
            hold_z = float(state.pos[2])

        self.current_target_gate = None
        self.active_target_track_id = None
        self.active_target_source = "no_active_target"
        self.no_active_target = True
        self.no_target_control_mode = "velocity_damping_altitude_hold"
        self.hold_anchor_source = "current_xy_completion_altitude"
        self.yaw_target_source = "hold_yaw"
        self.yaw_hold_value = self.perception_hold_yaw
        self.completed_gate_reference_blocked = True

        self.p_ref = np.array([state.pos[0], state.pos[1], hold_z], dtype=float)
        if self.post_completion_grace_active:
            self.v_ref = np.array([state.vel[0], state.vel[1], 0.0], dtype=float)
            self.no_target_control_mode = "neutral_attitude_altitude_hold"
            self.p_ref_source = "current_xy_altitude_only_grace"
            self.no_target_roll_source = "neutral_grace"
            self.no_target_pitch_source = "neutral_grace"
            self.horizontal_hold_disabled_after_completion = True
            self.velocity_damping_active = False
        else:
            self.v_ref = np.zeros(3, dtype=float)
            self.p_ref_source = "current_xy_velocity_damping_altitude_hold"
            self.no_target_roll_source = "velocity_damping"
            self.no_target_pitch_source = "velocity_damping"
            self.horizontal_hold_disabled_after_completion = False
            self.velocity_damping_active = True
        self.a_ref = np.zeros(3, dtype=float)
        self.hold_anchor = self.p_ref.copy()
        self.ref_yaw = self.perception_hold_yaw
        self.last_desired_yaw = self.perception_hold_yaw

        ref = Reference(
            pos=self.p_ref.copy(),
            vel=self.v_ref.copy(),
            acc=self.a_ref.copy(),
            yaw=self.continuous_yaw_command(self.perception_hold_yaw, current_yaw_rad),
        )

        roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd, dbg = self.tracker.update(state, ref)
        roll_cmd = -roll_cmd
        pitch_cmd = -pitch_cmd
        if self.post_completion_grace_active:
            roll_cmd = 0.0
            pitch_cmd = 0.0

        print("No active perception target; damping velocity and holding altitude/yaw.")
        print("state pos:", state.pos)
        print("hold ref pos:", ref.pos)
        print("hold yaw_des:", self.perception_hold_yaw)
        print("cmd roll pitch yaw thrust:", roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd)

        return roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd

    def attitude_control(self):
        current_yaw_rad = float(self.telemetry.rpy["yaw"]) * math.pi / 180.0

        pos = np.array([
            self.telemetry.pos["x"],
            self.telemetry.pos["y"],
            self.telemetry.pos["z"]
        ], dtype=float)

        state = State(
            pos=pos,
            vel=np.nan_to_num(np.array([
                self.telemetry.vel["vx"],
                self.telemetry.vel["vy"],
                self.telemetry.vel["vz"]
            ], dtype=float), nan=0.0, posinf=0.0, neginf=0.0),
            yaw=current_yaw_rad,
        )

        # Guard against calling before a plan exists
        if self.active_waypoints is None or self.planner.total_time <= 0.0:
            if self.use_perception:
                return self.hold_no_target_control(state, current_yaw_rad)

            print("No active trajectory; returning hover-ish neutral command.")
            self.current_target_gate = None
            self.yaw_target_source = "current_yaw"
            return 0.0, 0.0, current_yaw_rad, self.tracker.thrust_hover

        self.time_elapsed = time.time() - self.trajectory_start_time
        p_ref, v_ref, a_ref = self.planner.sample(self.time_elapsed)
        if self.use_perception:
            p_ref, v_ref, a_ref = self.clamp_reference_altitude(p_ref, v_ref, a_ref)
            self.no_active_target = False
            self.velocity_damping_active = False
            self.p_ref_source = "trajectory"

        self.p_ref = np.array(p_ref, dtype=float)
        self.v_ref = np.array(v_ref, dtype=float)
        self.a_ref = np.array(a_ref, dtype=float)

        if 0 <= self.current_target_idx < len(self.active_target_gates):
            self.current_target_gate = self.active_target_gates[self.current_target_idx].copy()
            if 0 <= self.current_target_idx < len(self.active_target_track_ids):
                self.active_target_track_id = self.active_target_track_ids[self.current_target_idx]
        elif self.use_perception:
            self.current_target_gate = None
            self.active_target_track_id = None
        else:
            self.current_target_gate = self.current_gate_pos.copy()

        if self.use_perception and self.current_target_gate is not None:
            if self.is_near_completed_gate(self.current_target_gate):
                self.target_retained_after_completion = True
                self.clear_active_perception_target(reason="completed_target_in_control")
                print("Completed target reached control path; switching to hold.")
                return self.hold_no_target_control(state, current_yaw_rad)
            to_target = np.asarray(self.current_target_gate[:2], dtype=float) - state.pos[:2]
            if np.linalg.norm(to_target) > 1e-3:
                desired_yaw = np.arctan2(to_target[1], to_target[0])
                self.yaw_target_source = "active_target"
            else:
                desired_yaw = compute_desired_yaw(v_ref, a_ref, self.last_desired_yaw)
                self.yaw_target_source = "reference_motion"
        else:
            desired_yaw = compute_desired_yaw(v_ref, a_ref, self.last_desired_yaw)
            self.yaw_target_source = "reference_motion"
        desired_yaw = self.continuous_yaw_command(desired_yaw, current_yaw_rad)
        self.ref_yaw = desired_yaw
        self.last_desired_yaw = desired_yaw

        ref = Reference(
            pos=np.array(p_ref, dtype=float),
            vel=np.array(v_ref, dtype=float),
            acc=np.array(a_ref, dtype=float),
            yaw=desired_yaw,
        )

        roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd, dbg = self.tracker.update(state, ref)
        roll_cmd = -roll_cmd
        pitch_cmd = -pitch_cmd

        print("state pos:", state.pos)
        print("ref pos:", ref.pos)
        print("ref vel:", ref.vel)
        print("ref acc:", ref.acc)
        print("yaw_des:", desired_yaw)
        print("cmd roll pitch yaw thrust:", roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd)

        return roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd


if __name__ == "__main__":
    api = AutonomyAPI(use_perception=False, race_gate_count=3)

    api.telemetry.pos = {"x": 0.0, "y": 0.0, "z": 0.0}
    api.telemetry.vel = {"vx": 0.0, "vy": 0.0, "vz": 0.0}
    api.telemetry.rpy = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
    api.last_desired_yaw = 0.0

    # Initial plan
    api.path_plan()

    print("Starting mock trajectory test...")

    while True:
        roll, pitch, yaw, thrust = api.attitude_control()

        # Perfect tracking mock
        if api.planner.total_time > 0.0:
            p, v, _ = api.planner.sample(api.time_elapsed)

            api.telemetry.pos["x"] = float(p[0])
            api.telemetry.pos["y"] = float(p[1])
            api.telemetry.pos["z"] = float(p[2])

            api.telemetry.vel["vx"] = float(v[0])
            api.telemetry.vel["vy"] = float(v[1])
            api.telemetry.vel["vz"] = float(v[2])

        # tracker yaw_cmd is in radians, but telemetry stores degrees
        api.telemetry.rpy["yaw"] = float(np.degrees(yaw))

        print(f"roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}, thrust={thrust:.2f}")

        gate_changed = api.advance_gate_if_needed(threshold=1.25)

        # Replan only when:
        # 1) active target changed, or
        # 2) current trajectory horizon is exhausted
        if gate_changed:
            api.path_plan()
        elif api.planner.total_time > 0.0 and api.time_elapsed >= api.planner.total_time:
            api.path_plan()

        time.sleep(0.02)
