import csv
import time
import math
import numpy as np


class FlightLogger:
    def __init__(self, filename="flight_log.csv"):
        self.log_file = open(filename, "w", newline="")
        self.writer = csv.writer(self.log_file)

        self.writer.writerow([
            "t",

            # Actual drone state
            "px", "py", "pz",
            "vx", "vy", "vz",
            "ax", "ay", "az",

            # Commanded attitude/thrust
            "roll_cmd_deg",
            "pitch_cmd_deg",
            "yaw_cmd_deg",
            "thrust_cmd",

            # Planner reference
            "p_ref_x", "p_ref_y", "p_ref_z",
            "v_ref_x", "v_ref_y", "v_ref_z",
            "a_ref_x", "a_ref_y", "a_ref_z",

            # Target gate / waypoint
            "target_x", "target_y", "target_z",

            # Debug info
            "tracking_err_x",
            "tracking_err_y",
            "tracking_err_z",
            "tracking_err_norm",
            "mode",
            "active_gate_idx",
            "loop_dt",
            "replan_requested",
            "replan_duration",
            "hold_command",
            "stale_command_suppressed",
            "plan_mode",
            "plan_start_gate_idx",
            "plan_end_gate_idx",
            "raw_gate_x",
            "raw_gate_y",
            "raw_gate_z",
            "perception_accepted",
            "perception_rejection_reason",
            "last_valid_target_x",
            "last_valid_target_y",
            "last_valid_target_z",
            "target_z_clamped",
            "perception_replan_trigger",
            "distance_to_active_target",
            "gate_completion_triggered",
            "completion_reason",
            "completed_gate_x",
            "completed_gate_y",
            "completed_gate_z",
            "active_gate_idx_before",
            "active_gate_idx_after",
            "race_cursor_before",
            "race_cursor_after",
            "active_target_source",
            "target_rejected_completed",
            "candidate_track_id",
            "candidate_x",
            "candidate_y",
            "candidate_z",
            "candidate_order_score",
            "rejected_wrong_order",
            "rejected_duplicate",
            "rejected_completed_this_lap",
            "race_cursor_advanced",
            "active_gate_idx_advanced",
            "completed_landmark_count",
            "lap_reset_triggered",
            "active_target_cleared",
            "active_target_track_id",
            "completed_gate_track_id",
            "yaw_target_source",
            "target_retained_after_completion",
            "next_valid_target_found",
            "valid_candidate_count",
            "approach_vector_x",
            "approach_vector_y",
            "approach_vector_z",
            "gate_progress_along_approach",
            "gate_lateral_error",
            "gate_plane_crossed",
            "near_gate_but_not_crossed",
            "completion_blocked_reason",
            "no_active_target",
            "no_target_control_mode",
            "hold_anchor_source",
            "hold_anchor_x",
            "hold_anchor_y",
            "hold_anchor_z",
            "velocity_damping_active",
            "completed_gate_reference_blocked",
            "p_ref_source",
            "yaw_hold_value",
            "telemetry_yaw_deg",
            "previous_yaw_cmd_deg",
            "raw_yaw_cmd_deg",
            "yaw_cmd_after_unwrap_deg",
            "yaw_rate_limited",
            "post_completion_grace_active",
            "no_target_roll_source",
            "no_target_pitch_source",
            "horizontal_hold_disabled_after_completion",
            "track_id",
            "merged_into_track_id",
            "duplicate_merge_reason",
            "race_order_track_ids",
            "race_order_inserted",
            "race_order_rejected_reason",
            "landmark_uncertainty",
            "track_observations",
            "completed_unique_gate_count",
            "active_gate_idx_clamped_by_race_gate_count",
            "suspected_duplicate_track",
            "committed_track_centers",
            "pairwise_committed_track_distances",
            "duplicate_radius_used",
            "merge_candidate_pairs",
            "merge_blocked_reason",
            "rejected_track_temporary_vs_permanent",
            "active_target_admission_status",
            "race_order_after_merge",
            "raw_image_corners",
            "ordered_image_corners",
            "pnp_rvec_x",
            "pnp_rvec_y",
            "pnp_rvec_z",
            "pnp_tvec_x",
            "pnp_tvec_y",
            "pnp_tvec_z",
            "gate_center_camera_x",
            "gate_center_camera_y",
            "gate_center_camera_z",
            "gate_center_body_x",
            "gate_center_body_y",
            "gate_center_body_z",
            "gate_center_world_x",
            "gate_center_world_y",
            "gate_center_world_z",
            "gate_normal_world_x",
            "gate_normal_world_y",
            "gate_normal_world_z",
            "detection_drone_x",
            "detection_drone_y",
            "detection_drone_z",
            "detection_drone_roll_rad",
            "detection_drone_pitch_rad",
            "detection_drone_yaw_rad",
            "reprojection_error",
        ])

        self.t0 = time.time()
        self.last_t = None
        self.last_vel = None

    def _vec3(self, value):
        """
        Converts list/tuple/np.array/dict/None into a 3-value numpy vector.
        Returns [nan, nan, nan] if unavailable.
        """
        if value is None:
            return np.array([np.nan, np.nan, np.nan], dtype=float)

        if isinstance(value, dict):
            # Supports {"x":..., "y":..., "z":...}
            if all(k in value for k in ["x", "y", "z"]):
                return np.array([value["x"], value["y"], value["z"]], dtype=float)

            # Supports {"north":..., "east":..., "up":...}
            if all(k in value for k in ["north", "east", "up"]):
                return np.array([value["north"], value["east"], value["up"]], dtype=float)

            return np.array([np.nan, np.nan, np.nan], dtype=float)

        arr = np.array(value, dtype=float).reshape(-1)

        if arr.size >= 3:
            return arr[:3]

        return np.array([np.nan, np.nan, np.nan], dtype=float)

    def _flat_string(self, value):
        if value is None:
            return ""
        arr = np.asarray(value, dtype=float).reshape(-1)
        return " ".join(f"{x:.3f}" for x in arr)

    def log(
        self,
        telemetry,
        roll_cmd,
        pitch_cmd,
        yaw_cmd,
        thrust_cmd,
        p_ref=None,
        v_ref=None,
        a_ref=None,
        target=None,
        mode=None,
        active_gate_idx=None,
        loop_dt=np.nan,
        replan_requested=False,
        replan_duration=0.0,
        hold_command=False,
        stale_command_suppressed=False,
        plan_mode=None,
        plan_start_gate_idx=None,
        plan_end_gate_idx=None,
        raw_gate=None,
        perception_accepted=False,
        perception_rejection_reason="",
        last_valid_target=None,
        target_z_clamped=False,
        perception_replan_trigger=False,
        distance_to_active_target=np.nan,
        gate_completion_triggered=False,
        completion_reason="",
        completed_gate_position=None,
        active_gate_idx_before=None,
        active_gate_idx_after=None,
        race_cursor_before=None,
        race_cursor_after=None,
        active_target_source="",
        target_rejected_completed=False,
        candidate_track_id=None,
        candidate_center=None,
        candidate_order_score=np.nan,
        rejected_wrong_order=False,
        rejected_duplicate=False,
        rejected_completed_this_lap=False,
        race_cursor_advanced=False,
        active_gate_idx_advanced=False,
        completed_landmark_count=0,
        lap_reset_triggered=False,
        active_target_cleared=False,
        active_target_track_id=None,
        completed_gate_track_id=None,
        yaw_target_source="",
        target_retained_after_completion=False,
        next_valid_target_found=False,
        valid_candidate_count=0,
        approach_vector=None,
        gate_progress_along_approach=np.nan,
        gate_lateral_error=np.nan,
        gate_plane_crossed=False,
        near_gate_but_not_crossed=False,
        completion_blocked_reason="",
        no_active_target=False,
        no_target_control_mode="",
        hold_anchor_source="",
        hold_anchor=None,
        velocity_damping_active=False,
        completed_gate_reference_blocked=False,
        p_ref_source="",
        yaw_hold_value=np.nan,
        telemetry_yaw_deg=np.nan,
        previous_yaw_cmd_deg=np.nan,
        raw_yaw_cmd_deg=np.nan,
        yaw_cmd_after_unwrap_deg=np.nan,
        yaw_rate_limited=False,
        post_completion_grace_active=False,
        no_target_roll_source="",
        no_target_pitch_source="",
        horizontal_hold_disabled_after_completion=False,
        track_id=None,
        merged_into_track_id=None,
        duplicate_merge_reason="",
        race_order_track_ids=None,
        race_order_inserted=False,
        race_order_rejected_reason="",
        landmark_uncertainty=np.nan,
        track_observations=0,
        completed_unique_gate_count=0,
        active_gate_idx_clamped_by_race_gate_count=False,
        suspected_duplicate_track=False,
        committed_track_centers="",
        pairwise_committed_track_distances="",
        duplicate_radius_used=np.nan,
        merge_candidate_pairs="",
        merge_blocked_reason="",
        rejected_track_temporary_vs_permanent="",
        active_target_admission_status="",
        race_order_after_merge=None,
        raw_image_corners=None,
        ordered_image_corners=None,
        pnp_rvec=None,
        pnp_tvec=None,
        gate_center_camera=None,
        gate_center_body=None,
        gate_center_world_debug=None,
        gate_normal_world=None,
        detection_drone_pose=None,
        reprojection_error=np.nan,
    ):
        now = time.time()
        t = now - self.t0

        # Your GetTelemetry seems to store pos/vel as dicts
        pos = self._vec3(getattr(telemetry, "pos", None))
        vel = self._vec3(getattr(telemetry, "vel", None))

        if self.last_t is None or self.last_vel is None:
            acc = np.array([0.0, 0.0, 0.0])
        else:
            dt = now - self.last_t
            acc = (vel - self.last_vel) / dt if dt > 1e-6 else np.zeros(3)

        p_ref = self._vec3(p_ref)
        v_ref = self._vec3(v_ref)
        a_ref = self._vec3(a_ref)
        target = self._vec3(target)
        raw_gate = self._vec3(raw_gate)
        last_valid_target = self._vec3(last_valid_target)
        completed_gate_position = self._vec3(completed_gate_position)
        candidate_center = self._vec3(candidate_center)
        approach_vector = self._vec3(approach_vector)
        hold_anchor = self._vec3(hold_anchor)
        pnp_rvec = self._vec3(pnp_rvec)
        pnp_tvec = self._vec3(pnp_tvec)
        gate_center_camera = self._vec3(gate_center_camera)
        gate_center_body = self._vec3(gate_center_body)
        gate_center_world_debug = self._vec3(gate_center_world_debug)
        gate_normal_world = self._vec3(gate_normal_world)
        detection_pose = np.asarray(detection_drone_pose, dtype=float).reshape(-1) if detection_drone_pose is not None else np.full(6, np.nan)
        if detection_pose.size < 6:
            detection_pose = np.full(6, np.nan)

        err = p_ref - pos
        err_norm = float(np.linalg.norm(err)) if not np.any(np.isnan(err)) else np.nan

        self.writer.writerow([
            t,

            pos[0], pos[1], pos[2],
            vel[0], vel[1], vel[2],
            acc[0], acc[1], acc[2],

            math.degrees(roll_cmd),
            math.degrees(pitch_cmd),
            math.degrees(yaw_cmd),
            thrust_cmd,

            p_ref[0], p_ref[1], p_ref[2],
            v_ref[0], v_ref[1], v_ref[2],
            a_ref[0], a_ref[1], a_ref[2],

            target[0], target[1], target[2],

            err[0], err[1], err[2],
            err_norm,
            mode,
            active_gate_idx,
            loop_dt,
            bool(replan_requested),
            replan_duration,
            bool(hold_command),
            bool(stale_command_suppressed),
            plan_mode,
            plan_start_gate_idx,
            plan_end_gate_idx,
            raw_gate[0],
            raw_gate[1],
            raw_gate[2],
            bool(perception_accepted),
            perception_rejection_reason,
            last_valid_target[0],
            last_valid_target[1],
            last_valid_target[2],
            bool(target_z_clamped),
            bool(perception_replan_trigger),
            distance_to_active_target,
            bool(gate_completion_triggered),
            completion_reason,
            completed_gate_position[0],
            completed_gate_position[1],
            completed_gate_position[2],
            active_gate_idx_before,
            active_gate_idx_after,
            race_cursor_before,
            race_cursor_after,
            active_target_source,
            bool(target_rejected_completed),
            candidate_track_id,
            candidate_center[0],
            candidate_center[1],
            candidate_center[2],
            candidate_order_score,
            bool(rejected_wrong_order),
            bool(rejected_duplicate),
            bool(rejected_completed_this_lap),
            bool(race_cursor_advanced),
            bool(active_gate_idx_advanced),
            completed_landmark_count,
            bool(lap_reset_triggered),
            bool(active_target_cleared),
            active_target_track_id,
            completed_gate_track_id,
            yaw_target_source,
            bool(target_retained_after_completion),
            bool(next_valid_target_found),
            valid_candidate_count,
            approach_vector[0],
            approach_vector[1],
            approach_vector[2],
            gate_progress_along_approach,
            gate_lateral_error,
            bool(gate_plane_crossed),
            bool(near_gate_but_not_crossed),
            completion_blocked_reason,
            bool(no_active_target),
            no_target_control_mode,
            hold_anchor_source,
            hold_anchor[0],
            hold_anchor[1],
            hold_anchor[2],
            bool(velocity_damping_active),
            bool(completed_gate_reference_blocked),
            p_ref_source,
            yaw_hold_value,
            telemetry_yaw_deg,
            previous_yaw_cmd_deg,
            raw_yaw_cmd_deg,
            yaw_cmd_after_unwrap_deg,
            bool(yaw_rate_limited),
            bool(post_completion_grace_active),
            no_target_roll_source,
            no_target_pitch_source,
            bool(horizontal_hold_disabled_after_completion),
            track_id,
            merged_into_track_id,
            duplicate_merge_reason,
            "" if race_order_track_ids is None else " ".join(str(x) for x in race_order_track_ids),
            bool(race_order_inserted),
            race_order_rejected_reason,
            landmark_uncertainty,
            track_observations,
            completed_unique_gate_count,
            bool(active_gate_idx_clamped_by_race_gate_count),
            bool(suspected_duplicate_track),
            committed_track_centers,
            pairwise_committed_track_distances,
            duplicate_radius_used,
            merge_candidate_pairs,
            merge_blocked_reason,
            rejected_track_temporary_vs_permanent,
            active_target_admission_status,
            "" if race_order_after_merge is None else " ".join(str(x) for x in race_order_after_merge),
            self._flat_string(raw_image_corners),
            self._flat_string(ordered_image_corners),
            pnp_rvec[0],
            pnp_rvec[1],
            pnp_rvec[2],
            pnp_tvec[0],
            pnp_tvec[1],
            pnp_tvec[2],
            gate_center_camera[0],
            gate_center_camera[1],
            gate_center_camera[2],
            gate_center_body[0],
            gate_center_body[1],
            gate_center_body[2],
            gate_center_world_debug[0],
            gate_center_world_debug[1],
            gate_center_world_debug[2],
            gate_normal_world[0],
            gate_normal_world[1],
            gate_normal_world[2],
            detection_pose[0],
            detection_pose[1],
            detection_pose[2],
            detection_pose[3],
            detection_pose[4],
            detection_pose[5],
            reprojection_error,
        ])

        self.log_file.flush()

        self.last_t = now
        self.last_vel = vel.copy()

    def close(self):
        self.log_file.close()
