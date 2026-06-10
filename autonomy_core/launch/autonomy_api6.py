import numpy as np
import time
import math
import os
import cv2
import json
import itertools
from autonomy_core.planning.minimum_snap_planner_multi_time_optimized import MultiSegmentMinimumSnapPlanner
from autonomy_core.launch.get_telemetry import GetTelemetry
from autonomy_core.controller.attitude_controller3 import RPGHighLevelTracker
from autonomy_core.perception.gate_perception_yolo import GatePerception
from autonomy_core.perception.gate_perception_node import GatePerceptionNode
from autonomy_core.perception.gate_memory import GateMemory
from autonomy_core.perception.corner_measurement import CornerMeasurement
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
    def __init__(
        self,
        use_perception=False,
        race_gate_count=None,
        race_gate_order=None,
        save_perception_debug_frames=True,
        use_lookahead_gate_filter=True,
    ):
        self.use_perception = use_perception
        self.use_lookahead_gate_filter = bool(use_lookahead_gate_filter)
        self.save_perception_debug_frames = bool(save_perception_debug_frames)
        self.perception_debug_frame_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "debug_frames",
        )
        self.camera_offset_body = np.array([0.12, 0.03, 0.242], dtype=float)

        self.gate_perception = GatePerception(
            gate_size=1.5,
            yolo_model_path="/home/paolo/datasets/gazebo_gate_yolo_pose_ab_runs/partial/weights/best.pt",
            preprocess_mode="raw",
            yolo_conf=0.1,
            yolo_imgsz=640,
            yolo_device=0,
        ) if use_perception else None
        self.perception_node = GatePerceptionNode(self.gate_perception) if use_perception else None
        self.gate_memory = GateMemory(
            association_radius=1.5,
            commit_radius=1.5,
            new_track_block_radius=4.5,
            min_confidence_per_hit=0.2,
            commit_hits=4,
            commit_confidence_sum=1.2,
            stale_time=3.0,
            alpha=0.35,
            use_lookahead_gate_filter=self.use_lookahead_gate_filter,
            min_hits_for_stable=6,
            max_center_std_for_stable=0.45,
            max_camera_std_for_stable=0.45,
            max_reprojection_error_for_stable=5.0,
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
        self.wall_tau = 0.0
        self.previous_sample_tau_used = 0.0
        self.previous_sample_tau_plan_id = None
        self.reference_progress_tau_lead_s = 0.25
        self.vehicle_nearest_tau_on_plan = float("nan")
        self.sample_tau_progress_limited = False
        self.sample_tau_before_progress_limit = float("nan")
        self.sample_tau_after_progress_limit = float("nan")
        self.reference_tau_lead_s = float("nan")
        self.reference_progress_lead_m = float("nan")
        self.reference_virtual_clock_enabled = False
        self.reference_projection_sample_count = 100
        self.continue_previous_trajectory_during_replan = True
        self.max_trajectory_holdover_s = 0.75
        self.hover_on_replan_without_valid_trajectory = True
        self.command_stale_safety_threshold_s = 0.5
        self.replan_in_progress = False
        self.continued_previous_trajectory_during_replan = False
        self.hover_due_to_replan = False
        self.hover_due_to_stale_command = False
        self.hover_due_to_no_valid_trajectory = False
        self.previous_trajectory_valid = False
        self.previous_trajectory_time_remaining = float("nan")
        self.trajectory_expired_s = float("nan")
        self.command_stale_age_s = float("nan")
        self.last_control_time = None
        self.error_z = 0.0
        self.last_desired_yaw = 0.0

        self.active_waypoints = None
        self.active_times = None
        self.active_target_gates = []
        self.active_target_track_ids = []
        self.current_target_idx = 0
        self.active_plan_id = 0
        self.installed_plan_sample_count = 160
        self.installed_plan_export_rows = []
        self.installed_plan_export_plan_id = None
        self.plan_geometric_validation_failed = False
        self.plan_geometric_fallback_used = False
        self.plan_validation_failed_segment_idx = -1
        self.plan_max_backward_progress_m = 0.0
        self.plan_max_overshoot_m = 0.0
        self.plan_negative_progress_velocity_count = 0
        self.plan_validation_failure_reason = ""
        self.plan_z_corridor_failed = False
        self.plan_min_z = float("nan")
        self.plan_max_z = float("nan")
        self.plan_z_undershoot_m = 0.0
        self.plan_z_fallback_reason = ""
        self.plan_z_start_below_safe_min = False

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
            thrust_max=0.85,
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
        self.has_commanded_yaw_reference = False
        self.hover_yaw_hold_reference_used = False
        self.hover_yaw_seed_source = ""
        self.hover_yaw_cmd_before_deg = float("nan")
        self.hover_yaw_cmd_after_deg = float("nan")
        self.hover_yaw_used_telemetry_fallback = False
        self.replan_hover_yaw_continuity_used = False
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
        self.next_track_available_after_completion = False
        self.skipped_target_clear_after_completion = False
        self.next_track_after_completion_id = None
        self.next_target_installed_same_cycle = False
        self.promoted_lookahead_to_active = False
        self.promoted_track_id = None
        self.promoted_track_center = np.full(3, np.nan, dtype=float)
        self.previous_horizon_track_ids = []
        self.previous_horizon_waypoint_types = ""
        self.promotion_blocked_reason = ""
        self.pending_lookahead_handoff = None
        self.promotion_normal_race_order_failed = False
        self.promotion_fallback_previous_horizon_used = False
        self.promotion_fallback_candidate_track_id = None
        self.promotion_fallback_candidate_center = np.full(3, np.nan, dtype=float)
        self.promotion_fallback_rejection_reason = ""
        self.previous_horizon_track_ids_at_completion = []
        self.previous_horizon_waypoint_types_at_completion = ""
        self.promoted_track_source = ""
        self.promotion_candidate_hits = 0
        self.promotion_candidate_inliers = 0
        self.promotion_candidate_outliers = 0
        self.promotion_candidate_camera_std = float("nan")
        self.promotion_candidate_center_std = float("nan")
        self.promotion_candidate_stability_blocker = ""
        self.post_completion_candidate_promoted = False
        self.post_completion_candidate_track_id = None
        self.post_completion_candidate_rejected_reason = ""
        self.race_order_after_post_completion_fallback = []
        self.target_clear_reason = ""
        self.post_completion_grace_suppressed = False
        self.use_passthrough_gate_velocities = False
        self.pass_through_speed = 3
        self.use_planning_lookahead_tracks = True
        self.use_raw_rejected_planning_lookahead = False
        self.planning_lookahead_min_hits = 6
        self.perception_transform_mode = "physical_direct_rad_x_mirror"
        # WARNING: gazebo_truth_sim_only is a simulation-only shortcut and
        # must not be used for real flight.
        self.perception_world_pose_source = "gazebo_truth_sim_only"
        self.perception_world_pose_sources = (
            "mavsdk",
            "gazebo_truth_sim_only",
        )
        self.perception_world_pose_source_used = "mavsdk"
        self.world_from_mavsdk = np.full(3, np.nan, dtype=float)
        self.world_from_gazebo_truth = np.full(3, np.nan, dtype=float)
        self.selected_world_estimate = np.full(3, np.nan, dtype=float)
        self.selected_vs_mavsdk_world_delta = np.full(3, np.nan, dtype=float)
        self.selected_vs_gazebo_world_delta = np.full(3, np.nan, dtype=float)
        self.debug_verbose_overlay = False
        self.image_gazebo_pose_snapshot = None
        self.gt_rmse_px_raw_indexed = float("nan")
        self.gt_rmse_px_ordered = float("nan")
        self.gt_rmse_px_best_permutation = float("nan")
        self.gt_center_err_px = float("nan")
        self.gt_corner_order_warning = False
        self.rmse_gate_idx = None
        self.detection_race_idx_for_rmse = None
        # Diagnostic experiment only: compensate the measured far-range PnP
        # depth bias for detections classified as future/lookahead gates.
        self.use_diagnostic_far_depth_correction = False
        self.perception_transform_modes = (
            "legacy_scaled_yaw",
            "direct_rad",
            "physical_direct_rad",
            "physical_direct_rad_x_mirror",
            "yaw_minus_pi_over_2",
            "pi_over_2_minus_yaw",
            "neg_yaw",
            "neg_yaw_plus_pi_over_2",
            "physical_mavsdk_yaw_aligned",
        )
        self.print_perception_transform_startup()
        print(
            "[LOOKAHEAD POLICY] active_target_shift applies only to the current "
            "active target; non-active detections update GateMemory before activation; "
            "an early lookahead center can later seed the active target; soft_tentative "
            "controls admission/type only and is passed to the same minimum-snap waypoint "
            "solver once included."
        )
        self.use_tentative_lookahead_spline = True
        self.lookahead_min_hits = 3
        self.lookahead_min_confidence_sum = 0.8
        self.lookahead_max_reprojection_error = 8.0
        self.lookahead_max_distance = 25.0
        self.use_terminal_passthrough_extension = True
        self.terminal_passthrough_extension_distance = 4.0
        self.suppress_minor_tentative_lookahead_replans = True
        self.tentative_lookahead_replan_min_shift = 0.75
        self.tentative_lookahead_shift_replan_threshold = 0.5
        self.tentative_lookahead_replan_min_interval_s = 0.5
        self.raw_planning_lookahead_ttl_s = 1.25
        self.raw_planning_lookahead_candidates = []
        self.planning_horizon_track_ids = []
        self.planning_horizon_waypoint_count = 0
        self.planning_horizon_waypoints = ""
        self.planning_horizon_waypoint_types = ""
        self._planning_target_waypoint_types = []
        self.future_track_visible_before_completion = False
        self.future_track_blocked_reason = ""
        self.horizon_build_cursor = 0
        self.horizon_available_order = []
        self.horizon_selected_track_ids = []
        self.horizon_rejected_track_ids = []
        self.horizon_rejection_reason = ""
        self.planning_lookahead_track_ids = []
        self.planning_lookahead_source = ""
        self.planning_lookahead_used = False
        self.tentative_lookahead_used = False
        self.tentative_lookahead_track_ids = []
        self.tentative_lookahead_centers = ""
        self.tentative_lookahead_rejection_reason = ""
        self.yolo_detection_count = 0
        self.yolo_detection_confidences = ""
        self.yolo_detection_bboxes = ""
        self.yolo_detection_keypoints = ""
        self.processed_detection_indices = []
        self.yolo_raw_count = 0
        self.pnp_success_count = 0
        self.world_valid_count = 0
        self.memory_update_count = 0
        self.tentative_track_count = 0
        self.tentative_lookahead_eligible_count = 0
        self.tentative_lookahead_eligible_track_ids = []
        self.tentative_lookahead_replan_requested = False
        self.tentative_lookahead_replan_blocked_reason = ""
        self.append_lookahead_called = False
        self.append_lookahead_input_track_ids = []
        self.append_lookahead_selected_track_ids = []
        self.append_lookahead_selected_centers = ""
        self.append_lookahead_selected_types = ""
        self.lookahead_pipeline_reasons = ""
        self.lookahead_pipeline_debug = "raw=0,pnp=0,valid=0,mem=0,tentative=0,eligible=0,used=0,reasons="
        self.perception_detection_flow = ""
        self.perception_detection_flow_entries = {}
        self.yolo_confidence = ""
        self.quad_area_px2 = ""
        self.old_area_confidence = ""
        self.memory_confidence_used = ""
        self.memory_admission_threshold = ""
        self.memory_admission_passed = ""
        self.pnp_camera_original = ""
        self.pnp_camera_depth_corrected = ""
        self.depth_correction_factor = ""
        self.world_original = ""
        self.world_depth_corrected = ""
        self.planning_track_horizon_debug = ""
        self.planning_cycle_debug = ""
        self.horizon_track_decisions = {}
        self.tentative_lookahead_centers_at_plan = {}
        self.tentative_lookahead_shift_m = float("nan")
        self.tentative_lookahead_shift_track_id = None
        self.tentative_lookahead_shift_replan_triggered = False
        self.post_completion_horizon_has_future = False
        self.terminal_passthrough_extension_used = False
        self.terminal_passthrough_extension_point = np.full(3, np.nan, dtype=float)
        self.current_gate_treated_as_terminal = False
        self.first_segment_terminal_velocity_zero = False
        self.tentative_lookahead_replan_suppressed = False
        self.tentative_lookahead_replan_suppression_reason = ""
        self.horizon_material_change_m = float("nan")
        self.first_segment_min_v_ref_predicted = float("nan")
        self.passthrough_velocity_enabled = False
        self.passthrough_speed_used = float("nan")
        self.waypoint_velocity_log = np.full((3, 3), np.nan, dtype=float)
        self.internal_gate_velocity_nonzero = False
        self.terminal_velocity_mode = ""
        self.replan_reason = ""
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
        self.tentative_track_ids = []
        self.stable_track_ids = []
        self.race_admitted_track_ids = []
        self.current_gate_candidate_track_ids = []
        self.selected_current_track_id = None
        self.rejected_current_track_ids = []
        self.current_selection_rejection_reason = ""
        self.future_lookahead_track_ids = []
        self.race_order_assignment_debug = ""
        self.selected_next_gate_track_id = None
        self.selected_next_gate_stability_score = float("nan")
        self.track_history_len = 0
        self.track_filtered_center = None
        self.track_raw_latest_center = None
        self.track_center_std = None
        self.track_center_std_norm = float("nan")
        self.track_camera_std_norm = float("nan")
        self.track_reprojection_error_mean = float("nan")
        self.track_reprojection_error_median = float("nan")
        self.track_outlier_count = 0
        self.track_inlier_count = 0
        self.track_is_stable = False
        self.track_stability_score = float("nan")
        self.promotion_reason = ""
        self.promotion_blocked_reason = ""
        self.promotion_normal_race_order_failed = False
        self.promotion_fallback_previous_horizon_used = False
        self.promotion_fallback_candidate_track_id = None
        self.promotion_fallback_candidate_center = np.full(3, np.nan, dtype=float)
        self.promotion_fallback_rejection_reason = ""
        self.previous_horizon_track_ids_at_completion = []
        self.previous_horizon_waypoint_types_at_completion = ""
        self.promoted_track_source = ""
        self.promotion_candidate_hits = 0
        self.promotion_candidate_inliers = 0
        self.promotion_candidate_outliers = 0
        self.promotion_candidate_camera_std = float("nan")
        self.promotion_candidate_center_std = float("nan")
        self.promotion_candidate_stability_blocker = ""
        self.post_completion_candidate_promoted = False
        self.post_completion_candidate_track_id = None
        self.post_completion_candidate_rejected_reason = ""
        self.race_order_after_post_completion_fallback = []
        self.selected_target_source = ""
        self.last_raw_image_corners = None
        self.last_ordered_image_corners = None
        self.last_pnp_debug_best_ordered_corners = None
        self.last_reprojected_image_corners = None
        self.last_pnp_rvec = None
        self.last_pnp_tvec = None
        self.last_gate_center_camera = None
        self.last_gate_center_body = None
        self.last_gate_center_world_debug = None
        self.last_gate_normal_camera = None
        self.last_gate_normal_world = None
        self.last_reprojection_error = float("nan")
        self.last_corner_reprojection_error_px = float("nan")
        self.last_pnp_candidate_count = 0
        self.last_chosen_pnp_candidate = None
        self.last_pnp_candidate_0_error = float("nan")
        self.last_pnp_candidate_1_error = float("nan")
        self.last_quad_center_x = float("nan")
        self.last_quad_center_y = float("nan")
        self.last_image_center_x = float("nan")
        self.last_image_center_y = float("nan")
        self.last_quad_center_offset_x = float("nan")
        self.last_quad_center_offset_y = float("nan")
        self.last_quad_width_px = float("nan")
        self.last_quad_height_px = float("nan")
        self.last_quad_aspect_ratio = float("nan")
        self.last_quad_area_px = float("nan")
        self.last_detection_drone_pose = None
        self.last_telemetry_rpy_raw_rad = None
        self.image_stamp_sec = 0
        self.image_stamp_nanosec = 0
        self.last_processed_image_stamp = None
        self.skipped_stale_image = False
        self.skipped_image_stamp = ""
        self.duplicate_image_skipped = False
        self.detection_world_computed_once = False
        self.pose_stamp_used_for_detection = float("nan")
        self.telemetry_stamp_current = float("nan")
        self.image_pose_age_s = float("nan")
        self.image_pose_snapshot_position = np.full(3, np.nan, dtype=float)
        self.image_pose_snapshot_rpy_raw = np.full(3, np.nan, dtype=float)
        self.image_pose_snapshot_rpy_perception = np.full(3, np.nan, dtype=float)
        self.corner_measurements = []
        self.corner_measurements_frame = []
        self.corner_measurement_records_frame = []
        self.corner_measurement_count = 0
        self.corner_measurements_log = ""
        self.image_received_wall_time = float("nan")
        self.image_processed_wall_time = float("nan")
        self.telemetry_position_sample_time = float("nan")
        self.telemetry_attitude_sample_time = float("nan")
        self.image_age_s = float("nan")
        self.attitude_age_s = float("nan")
        self.position_age_s = float("nan")
        self.pose_age_relative_to_image_s = float("nan")
        self.detection_drone_yaw_deg = float("nan")
        self.bearing_to_gate_deg = float("nan")
        self.telemetry_yaw_deg_for_image = float("nan")
        self.yaw_error_deg = float("nan")
        self.predicted_quad_offset_from_yaw_px = float("nan")
        self.yaw_pixel_error_px = float("nan")
        self.yaw_image_consistency_status = ""
        self.gazebo_model_pos_world = np.full(3, np.nan, dtype=float)
        self.gazebo_model_quat_world = np.full(4, np.nan, dtype=float)
        self.gazebo_camera_pos_world = np.full(3, np.nan, dtype=float)
        self.gazebo_camera_quat_world = np.full(4, np.nan, dtype=float)
        self.gazebo_pose_wall_time = float("nan")
        self.gazebo_pose_age_s = float("nan")
        self.gazebo_model_yaw_deg = float("nan")
        self.gazebo_camera_yaw_deg = float("nan")
        self.mavsdk_minus_gazebo_pos = np.full(3, np.nan, dtype=float)
        self.mavsdk_minus_gazebo_yaw_deg = float("nan")
        self.gate_world_mavsdk = np.full(3, np.nan, dtype=float)
        self.gate_world_gazebo = np.full(3, np.nan, dtype=float)
        self.gate_world_mavsdk_error_to_gt = float("nan")
        self.gate_world_gazebo_error_to_gt = float("nan")
        self.required_yaw_deg_from_pnp_to_gt = float("nan")
        self.mavsdk_yaw_minus_required_deg = float("nan")
        self.gazebo_yaw_minus_required_deg = float("nan")
        self.gate_world_uncorrected = np.full(3, np.nan, dtype=float)
        self.gate_world_corrected = np.full(3, np.nan, dtype=float)
        self.perception_yaw_correction_rad = 0.0
        self.perception_yaw_correction_initialized = False
        self.use_dynamic_gazebo_perception_yaw_correction = True
        self.dynamic_gazebo_perception_yaw_correction_rad = float("nan")
        self.active_perception_yaw_correction_rad = 0.0
        self.telemetry_yaw_raw_deg = float("nan")
        self.telemetry_yaw_perception_deg = float("nan")
        self.expected_planner_yaw_from_gazebo_deg = float("nan")
        self.dynamic_expected_planner_yaw_from_gazebo_deg = float("nan")
        self.raw_yaw_minus_dynamic_gazebo_expected_deg = float("nan")
        self.perception_yaw_minus_dynamic_gazebo_expected_deg = float("nan")
        self.gate_world_uncorrected = np.full(3, np.nan, dtype=float)
        self.gate_world_corrected = np.full(3, np.nan, dtype=float)
        self.perception_rpy_debug_frames_remaining = 5
        self.transform_sweep_error_stats = {}
        self.last_transform_source = ""
        self.last_camera_to_body_matrix_used = None
        self.last_body_to_world_method_used = ""
        self.live_vs_physical_direct_delta_m = float("nan")
        self.live_camera_axis_mode = ""
        self.live_camera_axis_det = float("nan")
        self.live_uses_x_mirror = False
        self.live_vs_camera_axis_x_flipped_delta_m = float("nan")
        self.per_gate_debug_summary = {}
        self.debug_expected_gate_idx = -1
        self.live_minus_expected = np.full(3, np.nan, dtype=float)
        self.live_lateral_error_m = float("nan")
        self.filtered_minus_expected = np.full(3, np.nan, dtype=float)
        self.selected_order_vs_axis_mode = ""
        self.last_pnp_size_sweep = {}
        self.last_pnp_formulation_debug = []
        self.last_camera_matrix = None
        self.last_dist_coeffs = None
        self.last_live_solver_name = ""
        self.last_pnp_fallback_reason = ""
        self.pnp_selected_order = ""
        self.pnp_selected_solver = ""
        self.pnp_selected_score = float("nan")
        self.pnp_selected_reprojection_error = float("nan")
        self.pnp_selected_gate_center_camera = None
        self.pnp_selected_reason = ""
        self.pnp_candidate_summary = ""
        self.pnp_candidate_world_summary = ""
        self.pnp_selected_world_score = float("nan")
        self.pnp_selected_world_reason = ""
        self.allow_pnp_corner_reordering = False
        self.pnp_live_candidate_orders_allowed = ""
        self.pnp_debug_best_order = ""
        self.pnp_live_vs_debug_best_order_mismatch = False
        self.pnp_lateral_angle = float("nan")
        self.image_center_offset_normalized = float("nan")
        self.keypoint_polygon_signed_area = float("nan")
        self.keypoint_polygon_winding = ""
        self.keypoint_edge_top = float("nan")
        self.keypoint_edge_right = float("nan")
        self.keypoint_edge_bottom = float("nan")
        self.keypoint_edge_left = float("nan")
        self.keypoint_bbox_center = np.full(2, np.nan, dtype=float)
        self.keypoint_polygon_center = np.full(2, np.nan, dtype=float)
        self.keypoint_bbox_polygon_delta = np.full(2, np.nan, dtype=float)
        self.raw_keypoint_polygon_signed_area = float("nan")
        self.raw_keypoint_polygon_winding = ""
        self.reset_transform_validation_debug()
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
        self.nearest_track_id = None
        self.nearest_track_distance = float("nan")
        self.nearest_track_hits = 0
        self.nearest_track_committed = False
        self.nearest_track_stable = False
        self.association_attempted = False
        self.association_success = False
        self.duplicate_rejection_reason = ""
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
        self.continued_existing_plan_after_completion = False
        self.continued_existing_plan_track_id = None
        self.continued_existing_plan_from_idx = -1
        self.continued_existing_plan_to_idx = -1
        self.gate_completion_replan_required = True
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
        self.pending_suffix_installed = False
        self.pending_suffix_rejected_reason = ""
        self.pending_suffix_waypoint_types = []
        self.pending_suffix_cleared_reason = ""
        self.future_only_replan_preserved_active_segment = False
        self.future_only_replan_reason = ""
        self.replan_suppressed_reason = ""
        self.next_valid_target_found = False
        self.valid_candidate_count = 0
        self.active_target_center = None
        self.active_target_center_at_plan = None
        self.active_target_latest_filtered_center = None
        self.active_target_shift_m = float("nan")
        self.active_target_shift_frames = 0
        self.active_target_shift_replan_triggered = False
        self.active_target_shift_threshold_m = 0.45
        self.active_target_shift_required_frames = 3
        self.active_target_shift_replan_min_interval_s = 0.5
        self.freeze_current_gate_target_near_gate = True
        self.current_gate_freeze_distance = 2.0
        self.current_gate_freeze_progress_margin = 1.5
        self.max_current_gate_target_shift_near_gate = 0.35
        self.active_target_shift_suppressed = False
        self.distance_to_active_target_at_shift = float("nan")
        self.target_shift_xy = float("nan")
        self.target_shift_z = float("nan")
        self.shift_replan_allowed = False
        self.shift_replan_suppressed_reason = ""
        self.active_shift_gt_debug_only = False
        self.near_gate_suppression_overridden = False
        self.near_gate_override_reason = ""
        self.committed_target_error_to_filter = float("nan")
        self.committed_target_xy_error_to_filter = float("nan")
        self.committed_target_z_error_to_filter = float("nan")
        self.committed_target_error_to_GT = float("nan")
        self.latest_filter_error_to_GT = float("nan")
        self.target_update_improvement_m = float("nan")
        self.target_update_alpha_used = float("nan")
        self.target_update_aggressive_correction_used = False
        self.gt_behavior_dependency_used = False
        self.gt_behavior_dependency_reason = ""
        self.terminal_extension_source = ""
        self.post_completion_direction_source = ""
        self.active_shift_gt_debug_only = False
        self.tracker_velocity_input = np.full(3, np.nan, dtype=float)
        self.tracker_velocity_was_sanitized = False
        self.tracker_e_p = np.full(3, np.nan, dtype=float)
        self.tracker_e_v = np.full(3, np.nan, dtype=float)
        self.tracker_a_ref = np.full(3, np.nan, dtype=float)
        self.tracker_a_fb = np.full(3, np.nan, dtype=float)
        self.tracker_a_cmd_raw = np.full(3, np.nan, dtype=float)
        self.tracker_a_cmd_limited = np.full(3, np.nan, dtype=float)
        self.thrust_raw_before_clamp = float("nan")
        self.thrust_cmd_after_clamp = float("nan")
        self.thrust_limited = False
        self.vertical_thrust_after_tilt = float("nan")
        self.hover_thrust = float("nan")
        self.pending_active_target_correction = None
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
        self.target_update_event = False
        self.target_update_previous = np.full(3, np.nan, dtype=float)
        self.target_update_new = np.full(3, np.nan, dtype=float)
        self.target_update_delta_m = float("nan")
        self.target_update_source_track_id = None
        self.target_update_raw_detection_center = np.full(3, np.nan, dtype=float)
        self.target_update_filtered_track_center = np.full(3, np.nan, dtype=float)
        self.target_update_reason = ""
        self.crossing_true_gate_center = np.full(3, np.nan, dtype=float)
        self.crossing_vehicle_position = np.full(3, np.nan, dtype=float)
        self.crossing_error = np.full(3, np.nan, dtype=float)
        self.crossing_lateral_error_xz = float("nan")

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

    def record_installed_plan_for_export(self, plan_source, replan_reason):
        planner = self.planner
        total_time = float(getattr(planner, "total_time", 0.0))
        if planner is None or total_time <= 0.0 or getattr(planner, "coeffs", None) is None:
            return

        self.active_plan_id += 1
        plan_id = int(self.active_plan_id)
        sample_count = max(2, int(self.installed_plan_sample_count))
        active_times = (
            ""
            if self.active_times is None
            else " ".join(
                f"{float(t):.6f}"
                for t in np.asarray(self.active_times, dtype=float).reshape(-1)
            )
        )
        active_track_ids = " ".join(str(x) for x in self.active_target_track_ids)
        waypoint_types = str(self.planning_horizon_waypoint_types or "")

        rows = []
        for tau in np.linspace(0.0, total_time, sample_count):
            tau = float(tau)
            try:
                p, v, a, _, _ = planner.sample_full(tau)
            except AttributeError:
                p, v, a = planner.sample(tau)
            rows.append({
                "plan_id": plan_id,
                "plan_source": str(plan_source or ""),
                "replan_reason": str(replan_reason or ""),
                "tau": tau,
                "x": float(p[0]),
                "y": float(p[1]),
                "z": float(p[2]),
                "vx": float(v[0]),
                "vy": float(v[1]),
                "vz": float(v[2]),
                "ax": float(a[0]),
                "ay": float(a[1]),
                "az": float(a[2]),
                "active_times": active_times,
                "active_target_track_ids": active_track_ids,
                "planning_horizon_waypoint_types": waypoint_types,
            })

        self.installed_plan_export_rows = rows
        self.installed_plan_export_plan_id = plan_id
        print(
            "[PLAN EXPORT] "
            f"plan_id={plan_id} source={plan_source} "
            f"samples={len(rows)} total_time={total_time:.3f}s"
        )

    def reset_plan_geometric_validation_debug(self):
        self.plan_geometric_validation_failed = False
        self.plan_geometric_fallback_used = False
        self.plan_validation_failed_segment_idx = -1
        self.plan_max_backward_progress_m = 0.0
        self.plan_max_overshoot_m = 0.0
        self.plan_negative_progress_velocity_count = 0
        self.plan_validation_failure_reason = ""
        self.plan_z_corridor_failed = False
        self.plan_min_z = float("nan")
        self.plan_max_z = float("nan")
        self.plan_z_undershoot_m = 0.0
        self.plan_z_fallback_reason = ""
        self.plan_z_start_below_safe_min = False

    def validate_minimum_snap_geometry(
        self,
        planner,
        waypoints,
        samples_per_segment=80,
        backward_tolerance_m=0.15,
        overshoot_tolerance_m=0.35,
        negative_velocity_tolerance=4,
        endpoint_margin_fraction=0.08,
        z_corridor_tolerance_m=0.05,
        z_endpoint_undershoot_tolerance_m=0.20,
    ):
        waypoints = np.asarray(waypoints, dtype=float)
        times = np.asarray(getattr(planner, "times", []), dtype=float).reshape(-1)
        segment_starts = np.asarray(
            getattr(planner, "segment_starts", []), dtype=float
        ).reshape(-1)
        if len(waypoints) < 2 or len(times) != len(waypoints) - 1:
            return False, {
                "segment_idx": -1,
                "max_backward_progress_m": 0.0,
                "max_overshoot_m": 0.0,
                "negative_progress_velocity_count": 0,
                "plan_min_z": float("nan"),
                "plan_max_z": float("nan"),
                "z_undershoot_m": 0.0,
                "z_start_below_safe_min": False,
                "reason": "invalid_validation_inputs",
            }

        worst = {
            "segment_idx": -1,
            "max_backward_progress_m": 0.0,
            "max_overshoot_m": 0.0,
            "negative_progress_velocity_count": 0,
            "plan_min_z": float("nan"),
            "plan_max_z": float("nan"),
            "z_undershoot_m": 0.0,
            "z_start_below_safe_min": False,
            "reason": "",
        }
        plan_min_z = float("inf")
        plan_max_z = float("-inf")
        worst_z_undershoot = 0.0

        for segment_idx in range(len(times)):
            p0 = waypoints[segment_idx]
            p1 = waypoints[segment_idx + 1]
            delta = p1 - p0
            segment_length = float(np.linalg.norm(delta))
            if not np.isfinite(segment_length) or segment_length < 1e-6:
                continue
            direction = delta / segment_length
            duration = float(times[segment_idx])
            segment_start = float(segment_starts[segment_idx])
            sample_count = max(3, int(samples_per_segment))
            progress_values = []
            z_values = []
            negative_velocity_count = 0

            for tau in np.linspace(0.0, duration, sample_count):
                p, v, _ = planner.sample(segment_start + float(tau))
                progress = float(np.dot(p - p0, direction))
                progress_values.append(progress)
                z = float(p[2])
                z_values.append(z)
                if np.isfinite(z):
                    plan_min_z = min(plan_min_z, z)
                    plan_max_z = max(plan_max_z, z)
                s_dot = float(np.dot(v, direction))
                normalized_tau = float(tau) / duration if duration > 1e-6 else 1.0
                if (
                    normalized_tau < 1.0 - float(endpoint_margin_fraction)
                    and s_dot < -1e-3
                ):
                    negative_velocity_count += 1

            max_backward = 0.0
            max_seen = progress_values[0]
            for progress in progress_values[1:]:
                max_backward = max(max_backward, max_seen - progress)
                max_seen = max(max_seen, progress)

            min_progress = min(progress_values)
            max_progress = max(progress_values)
            min_z = min(z_values) if z_values else float("nan")
            max_overshoot = max(
                0.0,
                max_progress - segment_length,
                -min_progress,
            )
            z_start = float(p0[2])
            z_end = float(p1[2])
            z_start_below_safe_min = bool(
                np.isfinite(z_start)
                and z_start < float(self.safe_min_target_z)
            )
            if z_start_below_safe_min:
                segment_floor = z_start - float(z_corridor_tolerance_m)
            else:
                segment_floor = max(
                    float(self.safe_min_target_z) - float(z_corridor_tolerance_m),
                    min(z_start, z_end) - float(z_endpoint_undershoot_tolerance_m),
                )
            z_undershoot = max(0.0, segment_floor - min_z) if np.isfinite(min_z) else 0.0
            worst_z_undershoot = max(worst_z_undershoot, z_undershoot)

            failed_reasons = []
            if max_backward > backward_tolerance_m:
                failed_reasons.append("backward_progress")
            if max_overshoot > overshoot_tolerance_m:
                failed_reasons.append("segment_overshoot")
            if negative_velocity_count > int(negative_velocity_tolerance):
                failed_reasons.append("negative_progress_velocity")
            if z_undershoot > 0.0:
                failed_reasons.append("z_corridor")

            if (
                max_backward > worst["max_backward_progress_m"]
                or max_overshoot > worst["max_overshoot_m"]
                or negative_velocity_count > worst["negative_progress_velocity_count"]
                or z_undershoot > worst["z_undershoot_m"]
            ):
                worst = {
                    "segment_idx": int(segment_idx),
                    "max_backward_progress_m": float(max_backward),
                    "max_overshoot_m": float(max_overshoot),
                    "negative_progress_velocity_count": int(negative_velocity_count),
                    "plan_min_z": float(plan_min_z),
                    "plan_max_z": float(plan_max_z),
                    "z_undershoot_m": float(z_undershoot),
                    "z_start_below_safe_min": bool(z_start_below_safe_min),
                    "reason": ",".join(failed_reasons),
                }

            if failed_reasons:
                worst["segment_idx"] = int(segment_idx)
                worst["plan_min_z"] = float(plan_min_z)
                worst["plan_max_z"] = float(plan_max_z)
                worst["z_undershoot_m"] = float(z_undershoot)
                worst["z_start_below_safe_min"] = bool(z_start_below_safe_min)
                worst["reason"] = ",".join(failed_reasons)
                return False, worst

        if not worst["reason"]:
            worst["reason"] = "ok"
        worst["plan_min_z"] = float(plan_min_z) if np.isfinite(plan_min_z) else float("nan")
        worst["plan_max_z"] = float(plan_max_z) if np.isfinite(plan_max_z) else float("nan")
        worst["z_undershoot_m"] = float(worst_z_undershoot)
        worst["z_start_below_safe_min"] = bool(
            np.any(waypoints[:-1, 2] < float(self.safe_min_target_z))
        )
        return True, worst

    def compute_final_exit_velocity(self, gates, default_speed=2.5):
        if len(gates) >= 2:
            d = gates[-1] - gates[-2]
        else:
            d = np.array([0.0, 0.0, 0.0], dtype=float)

        norm_d = np.linalg.norm(d)
        if norm_d < 1e-6:
            return np.array([0.0, default_speed, 0.0], dtype=float)

        return default_speed * (d / norm_d)

    def compute_passthrough_waypoint_velocities(self, waypoints):
        waypoints = np.asarray(waypoints, dtype=float)
        velocities = np.full_like(waypoints, np.nan, dtype=float)
        self.internal_gate_velocity_nonzero = False
        self.terminal_velocity_mode = "zero_final_horizon_endpoint"

        if (
            not self.use_passthrough_gate_velocities
            or len(waypoints) < 3
        ):
            return velocities

        speed = float(self.pass_through_speed)
        if not np.isfinite(speed) or speed <= 0.0:
            return velocities

        for i in range(1, len(waypoints) - 1):
            direction = waypoints[i + 1] - waypoints[i - 1]
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                continue
            velocities[i] = speed * (direction / norm)
            self.internal_gate_velocity_nonzero = True

        return velocities

    def gt_navigation_enabled(self):
        return not self.use_perception

    def is_final_race_gate(self):
        if self.race_gate_count is not None:
            return self.current_gate_idx >= int(self.race_gate_count) - 1
        if self.use_perception:
            return False
        if self.gt_navigation_enabled():
            return self.current_gate_idx >= len(self.gt_gates) - 1
        return False

    def compute_terminal_passthrough_extension(self, current_pos, current_gate):
        current_pos = np.asarray(current_pos, dtype=float).reshape(3)
        current_gate = np.asarray(current_gate, dtype=float).reshape(3)

        direction = None
        self.terminal_extension_source = ""
        next_gate_idx = self.current_gate_idx + 1
        next_gate_exists = (
            self.gt_navigation_enabled()
            and next_gate_idx < len(self.gt_gates)
        )
        if self.race_gate_count is not None and self.gt_navigation_enabled():
            next_gate_exists = next_gate_exists and next_gate_idx < int(self.race_gate_count)
        if next_gate_exists:
            direction = np.asarray(self.gt_gates[next_gate_idx], dtype=float).reshape(3) - current_gate
            self.terminal_extension_source = "gt_next_gate"
            self.gt_behavior_dependency_used = True
            self.gt_behavior_dependency_reason = "terminal_extension_gt_next_gate"
        if direction is None and self.last_gate_normal_world is not None:
            normal = np.asarray(self.last_gate_normal_world, dtype=float).reshape(3)
            if np.all(np.isfinite(normal)):
                direction = normal
                self.terminal_extension_source = "perception_gate_normal"
        if direction is None and len(self.completed_gate_positions_this_cycle) > 0:
            previous = np.asarray(
                self.completed_gate_positions_this_cycle[-1],
                dtype=float,
            ).reshape(3)
            direction = current_gate - previous
            self.terminal_extension_source = "previous_completed_to_current_gate"
        if direction is None and self.approach_vector is not None:
            approach = np.asarray(self.approach_vector, dtype=float).reshape(3)
            if np.all(np.isfinite(approach)):
                direction = approach
                self.terminal_extension_source = "approach_vector"
        if direction is None:
            velocity = np.array([
                self.telemetry.vel["vx"],
                self.telemetry.vel["vy"],
                self.telemetry.vel["vz"],
            ], dtype=float)
            if np.all(np.isfinite(velocity)) and float(np.linalg.norm(velocity)) > 1e-6:
                direction = velocity
                self.terminal_extension_source = "current_velocity"
        if direction is None or not np.all(np.isfinite(direction)):
            direction = current_gate - current_pos
            self.terminal_extension_source = "vehicle_to_current_gate"
        if (
            self.terminal_extension_source != "gt_next_gate"
            and np.all(np.isfinite(direction))
            and float(np.dot(direction, current_gate - current_pos)) < 0.0
        ):
            direction = -direction

        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return None
        return current_gate + float(self.terminal_passthrough_extension_distance) * direction / norm

    def remember_raw_planning_lookahead_candidate(self, center, reason, now=None):
        if not self.use_raw_rejected_planning_lookahead:
            return

        center = np.asarray(center, dtype=float).reshape(3)
        if not np.all(np.isfinite(center)):
            return
        if self.is_near_completed_gate(center, radius=self.gate_memory.new_track_block_radius):
            return

        now = time.time() if now is None else float(now)
        planning_center = center.copy()
        if planning_center[2] < self.safe_min_target_z:
            planning_center[2] = self.safe_min_target_z
        if planning_center[2] > self.safe_max_target_z:
            planning_center[2] = self.safe_max_target_z

        self.raw_planning_lookahead_candidates.append({
            "center": planning_center,
            "raw_center": center.copy(),
            "time": now,
            "reason": str(reason or ""),
        })
        self.prune_raw_planning_lookahead_candidates(now=now)

    def prune_raw_planning_lookahead_candidates(self, now=None):
        now = time.time() if now is None else float(now)
        ttl = float(self.raw_planning_lookahead_ttl_s)
        self.raw_planning_lookahead_candidates = [
            c for c in self.raw_planning_lookahead_candidates
            if now - float(c.get("time", 0.0)) <= ttl
        ]

    def _planning_target_safe_center(self, center):
        center = np.asarray(center, dtype=float).reshape(3)
        if not np.all(np.isfinite(center)):
            return None, "non_finite_target"
        safe = center.copy()
        if safe[2] < self.safe_min_target_z:
            safe[2] = self.safe_min_target_z
        if safe[2] > self.safe_max_target_z:
            safe[2] = self.safe_max_target_z
        return safe, ""

    def _track_reprojection_ok_for_tentative_lookahead(self, tr):
        errors = [
            float(getattr(tr, "reprojection_error_mean", np.nan)),
            float(getattr(tr, "reprojection_error_median", np.nan)),
        ]
        finite_errors = [err for err in errors if np.isfinite(err)]
        if len(finite_errors) == 0:
            return False
        return any(err <= self.lookahead_max_reprojection_error for err in finite_errors)

    def _track_reprojection_rejection_for_tentative_lookahead(self, tr):
        mean = float(getattr(tr, "reprojection_error_mean", np.nan))
        median = float(getattr(tr, "reprojection_error_median", np.nan))
        threshold = float(self.lookahead_max_reprojection_error)
        if np.isfinite(mean) and mean > threshold:
            return "reprojection_error_mean_high"
        if np.isfinite(median) and median > threshold:
            return "reprojection_error_median_high"
        return "tentative_reprojection_error_high"

    def _is_duplicate_of_committed_or_stable_track(self, track_id, center):
        center = np.asarray(center, dtype=float).reshape(3)
        for other in self.gate_memory.tracks:
            other_id = self.canonical_track_id(other.id)
            if other_id == track_id:
                continue
            if not (getattr(other, "committed", False) or getattr(other, "is_stable", False)):
                continue
            other_center = getattr(other, "filtered_center_world", None)
            if other_center is None:
                other_center = getattr(other, "center", None)
            if other_center is None:
                continue
            other_center = np.asarray(other_center, dtype=float).reshape(3)
            if not np.all(np.isfinite(other_center)):
                continue
            if float(np.linalg.norm(center - other_center)) < self.gate_memory.commit_radius:
                return True
        return False

    def _tentative_lookahead_rejection(self, tr, track_id, center, current_pos, selected_ids, existing_points):
        if not self.use_tentative_lookahead_spline:
            return "tentative_lookahead_disabled"
        if track_id is None or track_id in selected_ids:
            return "tentative_duplicate_track_id"
        active_track_id = self.canonical_track_id(getattr(self, "active_target_track_id", None))
        if active_track_id is not None and track_id == active_track_id:
            return "tentative_is_active_target"
        if track_id in self.completed_track_ids_this_cycle:
            return "tentative_completed_this_lap"
        if self.is_near_completed_gate(center):
            return "tentative_completed_gate_nearby"
        if any(float(np.linalg.norm(center - p)) < self.gate_memory.commit_radius for p in existing_points):
            return "tentative_duplicate_selected"
        if self._is_duplicate_of_committed_or_stable_track(track_id, center):
            return "tentative_duplicate_committed_or_stable"
        if int(getattr(tr, "hits", 0)) < self.lookahead_min_hits:
            return "tentative_insufficient_hits"
        if float(getattr(tr, "confidence_sum", 0.0)) < self.lookahead_min_confidence_sum:
            return "tentative_insufficient_confidence"
        if not self._track_reprojection_ok_for_tentative_lookahead(tr):
            return self._track_reprojection_rejection_for_tentative_lookahead(tr)
        if float(np.linalg.norm(center - current_pos)) > self.lookahead_max_distance:
            return "tentative_too_far"
        return ""

    def _append_planning_lookahead_targets(
        self,
        current_pos,
        target_gates,
        target_track_ids,
        target_waypoint_types,
        max_gates_ahead,
        allow_raw_candidates,
    ):
        self.append_lookahead_called = True
        self.append_lookahead_input_track_ids = [
            int(getattr(tr, "id", -1))
            for tr in sorted(self.gate_memory.tracks, key=lambda tr: tr.id)
        ]
        self.append_lookahead_selected_track_ids = []
        self.append_lookahead_selected_centers = ""
        self.append_lookahead_selected_types = ""
        initial_selected_count = len(target_track_ids)
        if not self.use_planning_lookahead_tracks:
            self.tentative_lookahead_rejection_reason = "append_disabled"
            return target_gates, target_track_ids, target_waypoint_types

        remaining = int(max_gates_ahead) - len(target_gates)
        if remaining <= 0:
            self.tentative_lookahead_rejection_reason = "append_no_remaining_slots"
            return target_gates, target_track_ids, target_waypoint_types

        now = time.time()
        self.prune_raw_planning_lookahead_candidates(now=now)
        selected_ids = {tid for tid in target_track_ids if tid is not None and tid >= 0}
        existing_points = [np.asarray(g, dtype=float).reshape(3) for g in target_gates]
        self.planning_lookahead_track_ids = []
        lookahead_sources = []

        all_tracks = sorted(self.gate_memory.tracks, key=lambda tr: tr.id)
        for tr in all_tracks:
            if remaining <= 0:
                break
            track_id = self.canonical_track_id(tr.id)
            if track_id is None or track_id in selected_ids:
                if track_id is not None and track_id in selected_ids:
                    self.horizon_track_decisions.setdefault(
                        track_id, "included:hard_current_or_stable"
                    )
                continue
            if track_id in self.completed_track_ids_this_cycle:
                self.horizon_rejected_track_ids.append(track_id)
                self.horizon_rejection_reason = "completed_this_lap"
                self.horizon_track_decisions[track_id] = "excluded:completed_this_lap"
                continue

            is_hard_lookahead = bool(getattr(tr, "is_stable", False))
            is_committed_unstable = bool(
                getattr(tr, "committed", False)
                and not getattr(tr, "is_stable", False)
            )
            if is_hard_lookahead:
                center_source = getattr(tr, "center", None)
            else:
                center_source = getattr(tr, "filtered_center_world", None)
                if center_source is None:
                    center_source = getattr(tr, "center", None)
            if center_source is None:
                self.horizon_rejected_track_ids.append(track_id)
                self.horizon_rejection_reason = "lookahead_missing_center"
                self.horizon_track_decisions[track_id] = "excluded:lookahead_missing_center"
                continue
            center, reason = self._planning_target_safe_center(center_source)
            if center is None:
                self.horizon_rejected_track_ids.append(track_id)
                self.horizon_rejection_reason = reason
                self.horizon_track_decisions[track_id] = f"excluded:{reason}"
                continue
            waypoint_type = "hard_stable"
            if is_hard_lookahead:
                if tr.hits < self.planning_lookahead_min_hits:
                    self.horizon_rejected_track_ids.append(track_id)
                    self.horizon_rejection_reason = "lookahead_insufficient_hits"
                    self.horizon_track_decisions[track_id] = "excluded:lookahead_insufficient_hits"
                    continue
                if self.is_near_completed_gate(center):
                    self.horizon_rejected_track_ids.append(track_id)
                    self.horizon_rejection_reason = "lookahead_completed_this_lap"
                    self.horizon_track_decisions[track_id] = "excluded:lookahead_completed_this_lap"
                    continue
                if any(float(np.linalg.norm(center - p)) < self.gate_memory.commit_radius for p in existing_points):
                    self.horizon_rejected_track_ids.append(track_id)
                    self.horizon_rejection_reason = "lookahead_duplicate_selected"
                    self.horizon_track_decisions[track_id] = "excluded:lookahead_duplicate_selected"
                    continue
                if float(np.linalg.norm(center - current_pos)) > self.max_detection_range:
                    self.horizon_rejected_track_ids.append(track_id)
                    self.horizon_rejection_reason = "lookahead_too_far"
                    self.horizon_track_decisions[track_id] = "excluded:lookahead_too_far"
                    continue
            else:
                reason = self._tentative_lookahead_rejection(
                    tr=tr,
                    track_id=track_id,
                    center=center,
                    current_pos=current_pos,
                    selected_ids=selected_ids,
                    existing_points=existing_points,
                )
                if reason:
                    self.horizon_rejected_track_ids.append(track_id)
                    self.horizon_rejection_reason = reason
                    self.tentative_lookahead_rejection_reason = reason
                    self.horizon_track_decisions[track_id] = f"excluded:{reason}"
                    continue
                waypoint_type = (
                    "soft_committed_unstable"
                    if is_committed_unstable
                    else "soft_tentative"
                )

            target_gates.append(center.copy())
            target_track_ids.append(track_id)
            target_waypoint_types.append(waypoint_type)
            selected_ids.add(track_id)
            existing_points.append(center.copy())
            self.planning_lookahead_track_ids.append(track_id)
            self.horizon_track_decisions[track_id] = f"included:{waypoint_type}"
            if waypoint_type in ("soft_committed_unstable", "soft_tentative"):
                self.tentative_lookahead_used = True
                self.tentative_lookahead_track_ids.append(track_id)
                lookahead_sources.append(waypoint_type)
            else:
                lookahead_sources.append("hard_stable")
            remaining -= 1

        if allow_raw_candidates and remaining > 0:
            for candidate in self.raw_planning_lookahead_candidates:
                if remaining <= 0:
                    break
                center, reason = self._planning_target_safe_center(candidate["center"])
                if center is None:
                    self.horizon_rejection_reason = reason
                    continue
                if self.is_near_completed_gate(center):
                    self.horizon_rejection_reason = "raw_lookahead_completed_this_lap"
                    continue
                if any(float(np.linalg.norm(center - p)) < self.gate_memory.commit_radius for p in existing_points):
                    self.horizon_rejection_reason = "raw_lookahead_duplicate_selected"
                    continue
                if float(np.linalg.norm(center - current_pos)) > self.max_detection_range:
                    self.horizon_rejection_reason = "raw_lookahead_too_far"
                    continue

                target_gates.append(center.copy())
                target_track_ids.append(-1)
                target_waypoint_types.append("soft_tentative")
                existing_points.append(center.copy())
                lookahead_sources.append("raw_rejected_clamped")
                remaining -= 1

        appended_ids = target_track_ids[initial_selected_count:]
        appended_gates = target_gates[initial_selected_count:]
        appended_types = target_waypoint_types[initial_selected_count:]
        self.append_lookahead_selected_track_ids = list(appended_ids)
        self.append_lookahead_selected_centers = ";".join(
            f"{track_id}:{gate[0]:.2f},{gate[1]:.2f},{gate[2]:.2f}"
            for gate, track_id in zip(appended_gates, appended_ids)
        )
        self.append_lookahead_selected_types = " ".join(str(t) for t in appended_types)
        if (
            int(getattr(self, "tentative_lookahead_eligible_count", 0)) > 0
            and not self.tentative_lookahead_used
            and not self.tentative_lookahead_rejection_reason
        ):
            self.tentative_lookahead_rejection_reason = (
                self.horizon_rejection_reason
                or "eligible_tentative_not_selected"
            )
        self.planning_lookahead_used = len(lookahead_sources) > 0
        self.planning_lookahead_source = " ".join(lookahead_sources)
        return target_gates, target_track_ids, target_waypoint_types

    def finalize_planning_horizon_debug(self):
        selected = {
            self.canonical_track_id(tid)
            for tid in self.active_target_track_ids + self.planning_lookahead_track_ids
            if tid is not None
        }
        parts = []
        for tr in sorted(self.gate_memory.tracks, key=lambda item: item.id):
            tid = self.canonical_track_id(tr.id)
            center = getattr(tr, "filtered_center_world", None)
            if center is None:
                center = getattr(tr, "center", np.full(3, np.nan))
            center = np.asarray(center, dtype=float).reshape(3)
            state = "stable" if getattr(tr, "is_stable", False) else "tentative"
            decision = self.horizon_track_decisions.get(tid)
            if decision is None:
                decision = (
                    "included:selected_horizon"
                    if tid in selected
                    else "excluded:not_selected_by_race_or_lookahead_policy"
                )
            parts.append(
                f"track{tid}:state={state},center={center[0]:.2f}/{center[1]:.2f}/{center[2]:.2f},"
                f"decision={decision}"
            )
        self.planning_track_horizon_debug = ";".join(parts)
        active_center = (
            np.asarray(self.active_target_center, dtype=float).reshape(3)
            if self.active_target_center is not None
            else np.full(3, np.nan)
        )
        self.planning_cycle_debug = (
            f"gate_idx={self.current_gate_idx},active_track={self.active_target_track_id},"
            f"active={active_center[0]:.2f}/{active_center[1]:.2f}/{active_center[2]:.2f},"
            f"lookahead_ids={'/'.join(map(str,self.planning_lookahead_track_ids))},"
            f"lookahead_centers={self.append_lookahead_selected_centers},"
            f"types={self.planning_horizon_waypoint_types},"
            f"tracks={self.planning_track_horizon_debug}"
        )
        print("[PLANNING FLOW] " + self.planning_cycle_debug)

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

    def perception_rpy_for_mode(
        self,
        telemetry_rpy_raw_rad,
        mode,
        apply_yaw_correction=True,
    ):
        """
        Temporary transform-convention switch.

        telemetry_rpy_raw_rad is stored in radians by px4_runner.py. The
        legacy mode intentionally preserves the previous working behavior that
        scaled those radians by pi/180 before the camera-to-world transform.
        """
        roll, pitch, yaw = np.asarray(telemetry_rpy_raw_rad, dtype=float).reshape(3)
        if apply_yaw_correction:
            yaw = self.wrap_pi(yaw + self.active_perception_yaw_correction_rad)
        mode = str(mode or "legacy_scaled_yaw")

        if mode == "legacy_scaled_yaw":
            return np.array([roll, pitch, yaw], dtype=float) * np.pi / 180.0
        if mode in ("direct_rad", "physical_direct_rad", "physical_direct_rad_x_mirror"):
            return np.array([roll, pitch, yaw], dtype=float)
        if mode == "yaw_minus_pi_over_2":
            return np.array([roll, pitch, yaw - (np.pi / 2.0)], dtype=float)
        if mode == "pi_over_2_minus_yaw":
            return np.array([roll, pitch, (np.pi / 2.0) - yaw], dtype=float)
        if mode == "neg_yaw":
            return np.array([roll, pitch, -yaw], dtype=float)
        if mode == "neg_yaw_plus_pi_over_2":
            return np.array([roll, pitch, -yaw + (np.pi / 2.0)], dtype=float)
        if mode == "physical_mavsdk_yaw_aligned":
            return np.array([roll, pitch, yaw - (np.pi / 2.0)], dtype=float)

        print(f"[PERCEPTION TRANSFORM WARN] unknown mode={mode}; using legacy_scaled_yaw")
        return np.array([roll, pitch, yaw], dtype=float) * np.pi / 180.0

    @staticmethod
    def wrap_pi(angle):
        return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi

    def initialize_perception_yaw_correction(self, gazebo_pose=None):
        if self.perception_yaw_correction_initialized or not isinstance(gazebo_pose, dict):
            return False
        try:
            gazebo_quat = np.asarray(
                gazebo_pose["gazebo_model_quat_world"], dtype=float
            ).reshape(4)
            gazebo_yaw_rad = math.radians(
                self._quaternion_xyzw_yaw_deg(gazebo_quat)
            )
            telemetry_yaw_raw = float(self.telemetry.rpy["yaw"])
        except (KeyError, TypeError, ValueError):
            return False
        if not np.isfinite(gazebo_yaw_rad) or not np.isfinite(telemetry_yaw_raw):
            return False

        expected_planner_yaw = self.wrap_pi((np.pi / 2.0) - gazebo_yaw_rad)
        self.perception_yaw_correction_rad = self.wrap_pi(
            expected_planner_yaw - telemetry_yaw_raw
        )
        self.active_perception_yaw_correction_rad = self.perception_yaw_correction_rad
        self.expected_planner_yaw_from_gazebo_deg = math.degrees(
            expected_planner_yaw
        )
        self.perception_yaw_correction_initialized = True
        if hasattr(self.telemetry, "set_perception_yaw_correction"):
            self.telemetry.set_perception_yaw_correction(
                self.perception_yaw_correction_rad
            )
        print(
            "[PERCEPTION YAW CORRECTION] "
            f"raw_deg={math.degrees(telemetry_yaw_raw):.3f} "
            f"gazebo_deg={math.degrees(gazebo_yaw_rad):.3f} "
            f"expected_planner_deg={self.expected_planner_yaw_from_gazebo_deg:.3f} "
            f"correction_deg={math.degrees(self.perception_yaw_correction_rad):.3f}"
        )
        return True

    def update_dynamic_gazebo_perception_yaw_correction(
        self,
        gazebo_pose,
        telemetry_yaw_raw_rad,
    ):
        self.dynamic_gazebo_perception_yaw_correction_rad = float("nan")
        self.dynamic_expected_planner_yaw_from_gazebo_deg = float("nan")
        self.raw_yaw_minus_dynamic_gazebo_expected_deg = float("nan")
        self.perception_yaw_minus_dynamic_gazebo_expected_deg = float("nan")
        self.active_perception_yaw_correction_rad = self.perception_yaw_correction_rad

        if not isinstance(gazebo_pose, dict):
            return
        try:
            gazebo_quat = np.asarray(
                gazebo_pose["gazebo_model_quat_world"], dtype=float
            ).reshape(4)
            gazebo_yaw_deg = self._quaternion_xyzw_yaw_deg(gazebo_quat)
        except (KeyError, TypeError, ValueError):
            return
        if not np.isfinite(gazebo_yaw_deg) or not np.isfinite(telemetry_yaw_raw_rad):
            return

        gazebo_yaw_rad = math.radians(gazebo_yaw_deg)
        expected_yaw = self.wrap_pi((np.pi / 2.0) - gazebo_yaw_rad)
        dynamic_correction = self.wrap_pi(
            expected_yaw - float(telemetry_yaw_raw_rad)
        )
        self.dynamic_expected_planner_yaw_from_gazebo_deg = math.degrees(
            expected_yaw
        )
        self.dynamic_gazebo_perception_yaw_correction_rad = dynamic_correction
        self.raw_yaw_minus_dynamic_gazebo_expected_deg = self._wrap_degrees(
            math.degrees(float(telemetry_yaw_raw_rad) - expected_yaw)
        )
        if self.use_dynamic_gazebo_perception_yaw_correction:
            self.active_perception_yaw_correction_rad = dynamic_correction
        perception_yaw = self.wrap_pi(
            float(telemetry_yaw_raw_rad)
            + self.active_perception_yaw_correction_rad
        )
        self.perception_yaw_minus_dynamic_gazebo_expected_deg = self._wrap_degrees(
            math.degrees(perception_yaw - expected_yaw)
        )

    def body_camera_matrix_for_mode(self, mode, default_matrix=None):
        if mode == "legacy_scaled_yaw":
            return self.legacy_body_camera_matrix()
        if mode in ("physical_direct_rad", "physical_mavsdk_yaw_aligned"):
            return self.physical_body_camera_matrix()
        if mode == "physical_direct_rad_x_mirror":
            return self.physical_x_mirror_body_camera_matrix()
        if default_matrix is not None:
            return np.asarray(default_matrix, dtype=float).reshape(3, 3)
        return np.asarray(self.perception_node.R_body_camera, dtype=float).reshape(3, 3)

    @staticmethod
    def legacy_body_camera_matrix():
        return np.array([
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ], dtype=float)

    @staticmethod
    def physical_x_mirror_body_camera_matrix():
        return np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ], dtype=float)

    def print_perception_transform_startup(self):
        if self.perception_node is None:
            return
        R = self.body_camera_matrix_for_mode(self.perception_transform_mode)
        ex = R @ np.array([1.0, 0.0, 0.0], dtype=float)
        ey = R @ np.array([0.0, 1.0, 0.0], dtype=float)
        ez = R @ np.array([0.0, 0.0, 1.0], dtype=float)
        print("[PERCEPTION TRANSFORM STARTUP] "
              f"perception_transform_mode={self.perception_transform_mode}")
        print("[PERCEPTION TRANSFORM STARTUP] R_body_camera="
              f"{np.array2string(R, precision=3, suppress_small=True)}")
        print("[PERCEPTION TRANSFORM STARTUP] "
              f"det(R_body_camera)={np.linalg.det(R):.3f}")
        print("[PERCEPTION TRANSFORM STARTUP] "
              f"camera +x -> body {ex.tolist()}")
        print("[PERCEPTION TRANSFORM STARTUP] "
              f"camera +y -> body {ey.tolist()}")
        print("[PERCEPTION TRANSFORM STARTUP] "
              f"camera +z -> body {ez.tolist()}")
        if self.perception_transform_mode == "physical_direct_rad_x_mirror":
            print(
                "[PERCEPTION TRANSFORM WARNING] "
                "physical_direct_rad_x_mirror uses a horizontal mirror convention; "
                "this indicates the live camera image or optical x-axis is mirrored "
                "relative to OpenCV assumptions."
            )

    def transform_gate_camera_to_world(self, gate_camera, drone_pos, telemetry_rpy_raw_rad, mode, r_body_camera):
        gate_camera = np.asarray(gate_camera, dtype=float).reshape(3)
        drone_pos = np.asarray(drone_pos, dtype=float).reshape(3)
        r_body_camera = np.asarray(r_body_camera, dtype=float).reshape(3, 3)
        rpy = self.perception_rpy_for_mode(telemetry_rpy_raw_rad, mode)
        roll, pitch, yaw = rpy
        r_wb = self.perception_node._rpy_to_rotmat(float(roll), float(pitch), float(yaw))
        gate_body = r_body_camera @ gate_camera
        world = drone_pos + r_wb @ (self.camera_offset_body + gate_body)
        return world, gate_body, rpy

    @staticmethod
    def _gazebo_model_pose_to_planner(gazebo_pose):
        """Convert Gazebo model truth into the established planner frame."""
        if not isinstance(gazebo_pose, dict):
            return None, None
        try:
            position_gazebo = np.asarray(
                gazebo_pose["gazebo_model_pos_world"], dtype=float
            ).reshape(3)
            rotation_gazebo = AutonomyAPI._quaternion_xyzw_to_rotmat(
                np.asarray(
                    gazebo_pose["gazebo_model_quat_world"], dtype=float
                ).reshape(4)
            )
        except (KeyError, TypeError, ValueError):
            return None, None
        if not (
            np.all(np.isfinite(position_gazebo))
            and np.all(np.isfinite(rotation_gazebo))
        ):
            return None, None

        # Gazebo world x/y map to planner y/x. The body reflection is required
        # to keep the result a proper rotation and yields yaw=90deg-gazebo_yaw.
        world_gazebo_to_planner = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)
        body_planner_to_gazebo = np.diag([1.0, -1.0, 1.0])
        position_planner = world_gazebo_to_planner @ position_gazebo
        rotation_planner_body = (
            world_gazebo_to_planner
            @ rotation_gazebo
            @ body_planner_to_gazebo
        )
        return position_planner, rotation_planner_body

    def transform_gate_camera_to_world_for_pose_source(
        self,
        gate_camera,
        drone_pos,
        telemetry_rpy_raw_rad,
        mode,
        r_body_camera,
        gazebo_pose=None,
    ):
        source = str(self.perception_world_pose_source)
        if source not in self.perception_world_pose_sources:
            raise ValueError(
                "perception_world_pose_source must be one of "
                f"{self.perception_world_pose_sources}, got {source!r}"
            )

        world_mavsdk, gate_body, rpy = self.transform_gate_camera_to_world(
            gate_camera=gate_camera,
            drone_pos=drone_pos,
            telemetry_rpy_raw_rad=telemetry_rpy_raw_rad,
            mode=mode,
            r_body_camera=r_body_camera,
        )
        gazebo_position, gazebo_rotation = self._gazebo_model_pose_to_planner(
            gazebo_pose
        )
        world_gazebo = np.full(3, np.nan, dtype=float)
        if gazebo_position is not None and gazebo_rotation is not None:
            world_gazebo = (
                gazebo_position
                + gazebo_rotation @ (self.camera_offset_body + gate_body)
            )

        selected = world_mavsdk.copy()
        source_used = "mavsdk"
        selected_rotation = self.perception_node._rpy_to_rotmat(*map(float, rpy))
        if source == "gazebo_truth_sim_only":
            if np.all(np.isfinite(world_gazebo)):
                selected = world_gazebo.copy()
                source_used = "gazebo_truth_sim_only"
                selected_rotation = gazebo_rotation
            else:
                source_used = "mavsdk_fallback_gazebo_truth_unavailable"

        return {
            "selected_world": selected,
            "world_from_mavsdk": world_mavsdk,
            "world_from_gazebo_truth": world_gazebo,
            "gate_body": gate_body,
            "rpy_mavsdk": rpy,
            "selected_rotation_world_body": selected_rotation,
            "source_used": source_used,
        }

    def update_perception_world_pose_source_debug(self, transform_result):
        self.perception_world_pose_source_used = str(
            transform_result["source_used"]
        )
        self.world_from_mavsdk = np.asarray(
            transform_result["world_from_mavsdk"], dtype=float
        ).reshape(3).copy()
        self.world_from_gazebo_truth = np.asarray(
            transform_result["world_from_gazebo_truth"], dtype=float
        ).reshape(3).copy()
        self.selected_world_estimate = np.asarray(
            transform_result["selected_world"], dtype=float
        ).reshape(3).copy()
        self.selected_vs_mavsdk_world_delta = (
            self.selected_world_estimate - self.world_from_mavsdk
        )
        self.selected_vs_gazebo_world_delta = (
            self.selected_world_estimate - self.world_from_gazebo_truth
        )

    def apply_diagnostic_far_depth_correction(
        self,
        detections,
        drone_pos,
        telemetry_rpy_raw_rad,
        gazebo_pose=None,
    ):
        """Apply the opt-in simulation diagnostic only to future detections."""
        if not detections:
            return detections

        drone_pos = np.asarray(drone_pos, dtype=float).reshape(3)
        r_body_camera = self.body_camera_matrix_for_mode(self.perception_transform_mode)
        race_order = list(self.race_progression.order())
        active_track_id = self.canonical_track_id(self.active_target_track_id)
        classified = []

        for det in detections:
            camera = np.asarray(
                det.get("gate_center_camera", np.full(3, np.nan)),
                dtype=float,
            ).reshape(3)
            world = np.asarray(
                det.get("gate_center_world", np.full(3, np.nan)),
                dtype=float,
            ).reshape(3)
            det["pnp_camera_original"] = camera.copy()
            det["pnp_camera_depth_corrected"] = camera.copy()
            det["depth_correction_factor"] = 1.0
            det["world_original"] = world.copy()
            det["world_depth_corrected"] = world.copy()
            det["diagnostic_far_depth_is_future"] = False
            det["diagnostic_far_depth_classification"] = "unclassified"

            matched_track = None
            matched_distance = float("inf")
            if np.all(np.isfinite(world)):
                for tr in self.gate_memory.tracks:
                    center = getattr(tr, "filtered_center_world", None)
                    if center is None:
                        center = getattr(tr, "center", None)
                    if center is None:
                        continue
                    distance = float(
                        np.linalg.norm(world - np.asarray(center, dtype=float).reshape(3))
                    )
                    if distance < matched_distance:
                        matched_track = tr
                        matched_distance = distance

            matched_id = (
                self.canonical_track_id(matched_track.id)
                if matched_track is not None
                and matched_distance <= self.gate_memory.association_radius
                else None
            )
            race_idx = race_order.index(matched_id) if matched_id in race_order else None
            is_future = race_idx is not None and race_idx > self.current_gate_idx
            if is_future:
                det["diagnostic_far_depth_classification"] = (
                    f"future_race_track:{matched_id}:race_idx={race_idx}"
                )
            elif matched_id is not None and matched_id == active_track_id:
                det["diagnostic_far_depth_classification"] = f"active_track:{matched_id}"
            classified.append({
                "det": det,
                "world": world,
                "distance": (
                    float(np.linalg.norm(world - drone_pos))
                    if np.all(np.isfinite(world))
                    else float("inf")
                ),
                "is_future": is_future,
                "matched_id": matched_id,
                "race_idx": race_idx,
            })

        # Before tracks have race indices, multiple simultaneous detections are
        # classified conservatively: the nearest valid gate remains current,
        # and only farther detections are eligible for this diagnostic.
        valid = [item for item in classified if np.all(np.isfinite(item["world"]))]
        if len(valid) >= 2:
            nearest = min(valid, key=lambda item: item["distance"])
            for item in valid:
                explicitly_current = bool(
                    item["matched_id"] == active_track_id
                    or item["race_idx"] == self.current_gate_idx
                )
                if item is not nearest and not explicitly_current and item["race_idx"] is None:
                    item["is_future"] = True
                    item["det"]["diagnostic_far_depth_classification"] = (
                        "future_by_multi_detection_distance"
                    )
                elif item is nearest and item["race_idx"] is None:
                    item["det"]["diagnostic_far_depth_classification"] = (
                        "current_by_multi_detection_distance"
                    )

        for item in classified:
            det = item["det"]
            det["diagnostic_far_depth_is_future"] = bool(item["is_future"])
            camera = det["pnp_camera_original"]
            depth = float(camera[2]) if np.all(np.isfinite(camera)) else float("nan")
            factor = 1.0
            if item["is_future"] and np.isfinite(depth) and depth > 0.0:
                if depth > 14.0:
                    factor = 1.0 / 0.915
                elif depth >= 8.0:
                    factor = 1.0 / 0.95

            corrected_camera = camera * factor
            corrected_world = det["world_original"].copy()
            if np.all(np.isfinite(corrected_camera)):
                transform_result = self.transform_gate_camera_to_world_for_pose_source(
                    gate_camera=corrected_camera,
                    drone_pos=drone_pos,
                    telemetry_rpy_raw_rad=telemetry_rpy_raw_rad,
                    mode=self.perception_transform_mode,
                    r_body_camera=r_body_camera,
                    gazebo_pose=gazebo_pose,
                )
                corrected_world = transform_result["selected_world"]
                corrected_body = transform_result["gate_body"]
                det["pnp_camera_depth_corrected"] = corrected_camera.copy()
                det["world_depth_corrected"] = corrected_world.copy()
                det["depth_correction_factor"] = float(factor)
                if self.use_diagnostic_far_depth_correction and factor != 1.0:
                    det["gate_center_camera"] = corrected_camera.copy()
                    det["tvec"] = corrected_camera.copy()
                    det["gate_center_body"] = corrected_body.copy()
                    det["gate_center_cam"] = corrected_body.copy()
                    det["gate_center_world"] = corrected_world.copy()

            print(
                "[DIAGNOSTIC FAR DEPTH] "
                f"enabled={self.use_diagnostic_far_depth_correction} "
                f"det={det.get('detection_index', -1)} "
                f"future={item['is_future']} "
                f"class={det['diagnostic_far_depth_classification']} "
                f"factor={factor:.6f} "
                f"camera_original={camera.tolist()} "
                f"camera_corrected={det['pnp_camera_depth_corrected'].tolist()} "
                f"world_original={det['world_original'].tolist()} "
                f"world_corrected={det['world_depth_corrected'].tolist()}"
            )
        return detections

    @staticmethod
    def _safe_norm(vec):
        vec = np.asarray(vec, dtype=float).reshape(-1)
        return float(np.linalg.norm(vec)) if np.all(np.isfinite(vec)) else float("nan")

    def _track_reference_points_for_candidate_selection(self):
        refs = []
        for tr in getattr(self.gate_memory, "tracks", []):
            points = []
            center = getattr(tr, "filtered_center_world", None)
            if center is None:
                center = getattr(tr, "center", None)
            if center is not None:
                points.append(np.asarray(center, dtype=float).reshape(3))
            for obs in list(getattr(tr, "obs_history", []))[-5:]:
                points.append(np.asarray(obs.center_world, dtype=float).reshape(3))
            finite_points = [p for p in points if np.all(np.isfinite(p))]
            if finite_points:
                refs.append((tr, finite_points))
        return refs

    def _temporal_candidate_distance(self, world, track_refs):
        best_dist = float("nan")
        best_track_id = None
        world = np.asarray(world, dtype=float).reshape(3)
        for tr, points in track_refs:
            for point in points:
                dist = float(np.linalg.norm(world - point))
                if not np.isfinite(best_dist) or dist < best_dist:
                    best_dist = dist
                    best_track_id = getattr(tr, "id", None)
        return best_dist, best_track_id

    def _route_direction_for_candidate_selection(self, current_pos):
        if self.active_target_center is not None:
            target = np.asarray(self.active_target_center, dtype=float).reshape(3)
        elif self.last_valid_target is not None:
            target = np.asarray(self.last_valid_target, dtype=float).reshape(3)
        else:
            target = None
        if target is None:
            return None
        vec = target - np.asarray(current_pos, dtype=float).reshape(3)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return None
        return vec / norm

    def select_pnp_candidate_for_live_geometry(
        self,
        det,
        drone_pos,
        telemetry_rpy_raw_rad,
        expected_gate_world=None,
    ):
        candidates = det.get("pnp_candidates", [])
        if not isinstance(candidates, list) or len(candidates) == 0:
            self.pnp_candidate_world_summary = ""
            return det

        r_body_camera = self.body_camera_matrix_for_mode(self.perception_transform_mode)
        rpy = self.perception_rpy_for_mode(telemetry_rpy_raw_rad, self.perception_transform_mode)
        r_wb = self.perception_node._rpy_to_rotmat(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        current_pos = np.asarray(drone_pos, dtype=float).reshape(3)
        route_dir = self._route_direction_for_candidate_selection(current_pos)
        track_refs = self._track_reference_points_for_candidate_selection()
        expected_gate_world = (
            np.asarray(expected_gate_world, dtype=float).reshape(3)
            if expected_gate_world is not None
            else None
        )

        evaluated = []
        for cand in candidates:
            cam = np.asarray(cand.get("tvec", np.full(3, np.nan)), dtype=float).reshape(3)
            normal_camera = np.asarray(cand.get("normal", np.full(3, np.nan)), dtype=float).reshape(3)
            if not np.all(np.isfinite(cam)):
                continue
            gate_body = r_body_camera @ cam
            world = current_pos + r_wb @ (self.camera_offset_body + gate_body)
            normal_body = r_body_camera @ normal_camera if np.all(np.isfinite(normal_camera)) else np.full(3, np.nan)
            normal_world = r_wb @ normal_body if np.all(np.isfinite(normal_body)) else np.full(3, np.nan)
            if np.all(np.isfinite(normal_world)):
                normal_world = normal_world / (np.linalg.norm(normal_world) + 1e-12)

            reproj = float(cand.get("error", np.nan))
            size_depth_error = abs(float(cand.get("depth_disagreement", np.nan)))
            if not np.isfinite(size_depth_error):
                size_depth_error = abs(float(cand.get("depth", np.nan)) - float(cand.get("size_depth", np.nan)))
            if not np.isfinite(size_depth_error):
                size_depth_error = 0.0

            z_error = abs(float(world[2]) - 1.45) if np.isfinite(world[2]) else 10.0
            dist_vehicle = float(np.linalg.norm(world - current_pos)) if np.all(np.isfinite(world)) else 1e3
            range_penalty = 0.0
            if dist_vehicle < 1.0:
                range_penalty = 1.0 - dist_vehicle
            elif dist_vehicle > self.max_detection_range:
                range_penalty = dist_vehicle - self.max_detection_range

            route_penalty = 0.0
            if route_dir is not None and np.all(np.isfinite(normal_world)):
                # Gate plane normal should usually align with course direction;
                # use abs() because the normal sign is ambiguous for a planar target.
                route_penalty = 1.0 - abs(float(np.dot(normal_world, route_dir)))

            temporal_dist, temporal_track_id = self._temporal_candidate_distance(world, track_refs)
            temporal_penalty = 0.0
            temporal_bonus = 0.0
            if np.isfinite(temporal_dist):
                temporal_penalty = min(temporal_dist, 5.0)
                if temporal_dist <= self.gate_memory.association_radius:
                    temporal_bonus = 2.0

            lateral_penalty = 0.0
            if self.active_target_center is not None:
                active = np.asarray(self.active_target_center, dtype=float).reshape(3)
                if np.all(np.isfinite(active)) and float(np.linalg.norm(world[:2] - active[:2])) > self.max_gate_jump:
                    lateral_penalty = 2.0

            score = (
                -1.20 * z_error
                -0.20 * (reproj if np.isfinite(reproj) else 10.0)
                -0.35 * size_depth_error
                -0.15 * range_penalty
                -0.50 * route_penalty
                -0.80 * temporal_penalty
                -1.00 * lateral_penalty
                + temporal_bonus
            )
            reason = "geometry_score"
            if np.isfinite(temporal_dist) and temporal_dist <= self.gate_memory.association_radius:
                reason = "temporal_consistency"

            gt_error = (
                float(np.linalg.norm(world - expected_gate_world))
                if expected_gate_world is not None and np.all(np.isfinite(expected_gate_world))
                else float("nan")
            )
            evaluated.append({
                "candidate": cand,
                "cam": cam,
                "gate_body": gate_body,
                "world": world,
                "normal_camera": normal_camera,
                "normal_world": normal_world,
                "reproj": reproj,
                "size_depth_error": size_depth_error,
                "z_error": z_error,
                "dist_vehicle": dist_vehicle,
                "temporal_dist": temporal_dist,
                "temporal_track_id": temporal_track_id,
                "score": float(score),
                "reason": reason,
                "gt_error": gt_error,
            })

        if not evaluated:
            self.pnp_candidate_world_summary = ""
            return det

        best = max(evaluated, key=lambda item: item["score"])
        best_cand = best["candidate"]
        current_cam = np.asarray(det.get("gate_center_camera", np.full(3, np.nan)), dtype=float).reshape(3)
        if not np.all(np.isfinite(current_cam)):
            current_cam = best["cam"].copy()
        current_reproj = float(det.get("reprojection_error", np.nan))
        fallback = min(evaluated, key=lambda item: item["reproj"] if np.isfinite(item["reproj"]) else 1e9)
        if best["score"] < -50.0:
            best = fallback
            best_cand = best["candidate"]
            best["reason"] = "lowest_reprojection_fallback"

        det["gate_center_camera"] = best["cam"].copy()
        det["gate_center_body"] = best["gate_body"].copy()
        det["gate_center_cam"] = best["gate_body"].copy()
        det["gate_center_world"] = best["world"].copy()
        det["gate_normal_camera"] = best["normal_camera"].copy()
        det["gate_normal_body"] = r_body_camera @ best["normal_camera"] if np.all(np.isfinite(best["normal_camera"])) else np.full(3, np.nan)
        det["gate_normal_world"] = best["normal_world"].copy()
        det["reprojection_error"] = float(best["reproj"])
        det["corner_reprojection_error_px"] = float(best["reproj"])
        det["tvec"] = best["cam"].copy()
        det["rvec"] = np.asarray(best_cand.get("rvec", np.full(3, np.nan)), dtype=float).reshape(3).copy()
        det["ordered_corners"] = np.asarray(best_cand.get("ordered_points", det.get("ordered_corners")), dtype=float).reshape(4, 2).copy()
        det["reprojected_corners"] = best_cand.get("projected_corners", det.get("reprojected_corners", None))
        det["chosen_candidate"] = int(best_cand.get("index", -1))
        det["live_solver_name"] = f"{best_cand.get('solver', '')}_{best_cand.get('order', '')}"
        det["pnp_selected_order"] = best_cand.get("order", "")
        det["pnp_selected_solver"] = best_cand.get("solver", "")
        det["pnp_selected_score"] = float(best["score"])
        det["pnp_selected_reprojection_error"] = float(best["reproj"])
        det["pnp_selected_gate_center_camera"] = best["cam"].copy()
        det["pnp_selected_reason"] = best["reason"]
        det["pnp_selected_world_score"] = float(best["score"])
        det["pnp_selected_world_reason"] = best["reason"]

        summary_parts = []
        for item in sorted(evaluated, key=lambda x: x["score"], reverse=True)[:18]:
            cand = item["candidate"]
            nw = item["normal_world"]
            summary_parts.append(
                f"{cand.get('order','')}/{cand.get('solver','')}:"
                f"reproj={item['reproj']:.2f},"
                f"cam=({item['cam'][0]:.2f},{item['cam'][1]:.2f},{item['cam'][2]:.2f}),"
                f"world=({item['world'][0]:.2f},{item['world'][1]:.2f},{item['world'][2]:.2f}),"
                f"z={item['world'][2]:.2f},"
                f"normal=({nw[0]:.2f},{nw[1]:.2f},{nw[2]:.2f}),"
                f"score={item['score']:.2f},"
                f"reason={item['reason']},"
                f"gt={item['gt_error']:.2f},"
                f"track={item['temporal_track_id']}:{item['temporal_dist']:.2f}"
            )
        self.pnp_candidate_world_summary = ";".join(summary_parts)
        det["pnp_candidate_world_summary"] = self.pnp_candidate_world_summary
        print(
            "[PNP WORLD SELECT] "
            f"reason={best['reason']} order={det['pnp_selected_order']} "
            f"solver={det['pnp_selected_solver']} score={best['score']:.2f} "
            f"world={best['world'].tolist()} cam={best['cam'].tolist()} "
            f"prev_cam={current_cam.tolist()} prev_reproj={current_reproj:.2f}"
        )
        return det

    @staticmethod
    def physical_body_camera_matrix():
        return np.array([
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ], dtype=float)

    @staticmethod
    def _compact_float_list(values, precision=3):
        out = []
        for value in values or []:
            try:
                out.append(f"{float(value):.{precision}f}")
            except Exception:
                out.append("nan")
        return " ".join(out)

    @staticmethod
    def _compact_nested_points(values, precision=1):
        chunks = []
        for item in values or []:
            arr = np.asarray(item, dtype=float).reshape(-1)
            chunks.append(" ".join(f"{x:.{precision}f}" for x in arr))
        return ";".join(chunks)

    @staticmethod
    def _compact_reason(reason):
        reason = str(reason or "")
        if reason.startswith("reprojection_error_high"):
            return "reprojection_error_high"
        if reason.startswith("z_below_safe_min"):
            return "z_below_safe_min"
        if reason.startswith("z_above_safe_max"):
            return "z_above_safe_max"
        if reason.startswith("detection_too_far") or reason.endswith("too_far"):
            return "too_far"
        if "duplicate" in reason:
            return "duplicate_existing_track"
        if "completed" in reason or "near_completed" in reason:
            return "near_completed_gate"
        if "insufficient_hits" in reason:
            return "insufficient_hits"
        if "insufficient_confidence" in reason:
            return "insufficient_confidence_sum"
        if "reprojection_error_mean" in reason:
            return "reprojection_mean_high"
        if "reprojection_error_median" in reason:
            return "reprojection_median_high"
        if "reprojection" in reason:
            return "reprojection_error_high"
        if reason in ("invalid_gate_center", "non_finite_target", "non_finite_center"):
            return "invalid_world_center"
        return reason

    def reset_lookahead_pipeline_debug(self):
        self.yolo_detection_count = 0
        self.yolo_detection_confidences = ""
        self.yolo_detection_bboxes = ""
        self.yolo_detection_keypoints = ""
        self.processed_detection_indices = []
        self.yolo_raw_count = 0
        self.pnp_success_count = 0
        self.world_valid_count = 0
        self.memory_update_count = 0
        self.tentative_track_count = 0
        self.tentative_lookahead_eligible_count = 0
        self.tentative_lookahead_eligible_track_ids = []
        self.tentative_lookahead_replan_requested = False
        self.tentative_lookahead_replan_blocked_reason = ""
        self.lookahead_pipeline_reasons = ""
        self.lookahead_pipeline_debug = "raw=0,pnp=0,valid=0,mem=0,tentative=0,eligible=0,used=0,reasons="
        self.perception_detection_flow = ""
        self.perception_detection_flow_entries = {}
        self.yolo_confidence = ""
        self.quad_area_px2 = ""
        self.old_area_confidence = ""
        self.memory_confidence_used = ""
        self.memory_admission_threshold = ""
        self.memory_admission_passed = ""
        self.pnp_camera_original = ""
        self.pnp_camera_depth_corrected = ""
        self.depth_correction_factor = ""
        self.world_original = ""
        self.world_depth_corrected = ""

    def initialize_detection_flow_debug(self, yolo):
        for meta in getattr(yolo, "last_yolo_candidate_debug", []) or []:
            idx = int(meta.get("detection_index", len(self.perception_detection_flow_entries)))
            self.perception_detection_flow_entries[idx] = {
                "idx": idx,
                "yolo_confidence": float(meta.get("box_confidence", np.nan)),
                "bbox": np.asarray(
                    meta.get("bbox", np.full(4, np.nan)), dtype=float
                ).reshape(4),
                "keypoints": np.asarray(
                    meta.get("keypoints", np.full((4, 3), np.nan)), dtype=float
                ).reshape(-1, 3),
                "quad_area_px2": float("nan"),
                "old_area_confidence": float("nan"),
                "memory_confidence": float("nan"),
                "memory_admission_threshold": float(self.gate_memory.min_confidence_per_hit),
                "memory_admission_passed": False,
                "pnp": not bool(meta.get("rejection_reason", "")),
                "cam": np.full(3, np.nan),
                "raw": np.full(3, np.nan),
                "corrected": np.full(3, np.nan),
                "pnp_camera_original": np.full(3, np.nan),
                "pnp_camera_depth_corrected": np.full(3, np.nan),
                "depth_correction_factor": 1.0,
                "world_original": np.full(3, np.nan),
                "world_depth_corrected": np.full(3, np.nan),
                "diagnostic_far_depth_is_future": False,
                "diagnostic_far_depth_classification": "",
                "track": None,
                "memory": False,
                "state": "",
                "race_idx": None,
                "role": "rejected" if meta.get("rejection_reason") else "visible_but_unused",
                "reason": meta.get("rejection_reason", ""),
            }

    def update_detection_flow_debug(self, det, result=None, rejection_reason=""):
        idx = int(det.get("detection_index", det.get("processed_detection_index", -1)))
        entry = self.perception_detection_flow_entries.setdefault(idx, {"idx": idx})
        cam = np.asarray(det.get("gate_center_camera", np.full(3, np.nan)), dtype=float).reshape(3)
        corrected = np.asarray(det.get("gate_center_world", np.full(3, np.nan)), dtype=float).reshape(3)
        raw_world = np.full(3, np.nan)
        drone_pos = np.asarray(det.get("drone_pos", np.full(3, np.nan)), dtype=float).reshape(3)
        if np.all(np.isfinite(cam)) and np.all(np.isfinite(drone_pos)) and self.last_telemetry_rpy_raw_rad is not None:
            raw_rpy = self.perception_rpy_for_mode(
                self.last_telemetry_rpy_raw_rad,
                self.perception_transform_mode,
                apply_yaw_correction=False,
            )
            r_wb = self.perception_node._rpy_to_rotmat(*map(float, raw_rpy))
            raw_world = drone_pos + r_wb @ (
                self.camera_offset_body
                + self.body_camera_matrix_for_mode(self.perception_transform_mode) @ cam
            )
        track_id = result.get("track_id") if result is not None else None
        tr = self.gate_memory.get_track_by_id(track_id) if track_id is not None else None
        race_order = self.race_progression.order()
        race_idx = race_order.index(track_id) if track_id in race_order else None
        active_id = self.canonical_track_id(self.active_target_track_id)
        lookahead_ids = {
            self.canonical_track_id(tid)
            for tid in self.planning_lookahead_track_ids + self.tentative_lookahead_track_ids
        }
        role = "rejected" if rejection_reason else "visible_but_unused"
        if track_id is not None and self.canonical_track_id(track_id) == active_id:
            role = "active_current_target"
        elif track_id is not None and self.canonical_track_id(track_id) in lookahead_ids:
            role = "tentative_lookahead_target"
        entry.update({
            "yolo_confidence": float(
                det.get("yolo_confidence", entry.get("yolo_confidence", np.nan))
            ),
            "bbox": np.asarray(
                det.get("yolo_bbox", entry.get("bbox", np.full(4, np.nan))),
                dtype=float,
            ).reshape(4),
            "keypoints": np.asarray(
                det.get(
                    "yolo_keypoints",
                    entry.get("keypoints", np.full((4, 3), np.nan)),
                ),
                dtype=float,
            ).reshape(-1, 3),
            "quad_area_px2": float(det.get("quad_area_px2", np.nan)),
            "old_area_confidence": float(
                det.get("old_area_confidence", det.get("quad_area_confidence", np.nan))
            ),
            "memory_confidence": float(
                det.get("memory_confidence", det.get("yolo_confidence", np.nan))
            ),
            "memory_admission_threshold": float(self.gate_memory.min_confidence_per_hit),
            "memory_admission_passed": bool(
                float(det.get("memory_confidence", det.get("yolo_confidence", np.nan)))
                >= self.gate_memory.min_confidence_per_hit
            ),
            "pnp": np.all(np.isfinite(cam)),
            "cam": cam,
            "raw": raw_world,
            "corrected": corrected,
            "pnp_camera_original": np.asarray(
                det.get("pnp_camera_original", cam), dtype=float
            ).reshape(3),
            "pnp_camera_depth_corrected": np.asarray(
                det.get("pnp_camera_depth_corrected", cam), dtype=float
            ).reshape(3),
            "depth_correction_factor": float(det.get("depth_correction_factor", 1.0)),
            "world_original": np.asarray(
                det.get("world_original", corrected), dtype=float
            ).reshape(3),
            "world_depth_corrected": np.asarray(
                det.get("world_depth_corrected", corrected), dtype=float
            ).reshape(3),
            "diagnostic_far_depth_is_future": bool(
                det.get("diagnostic_far_depth_is_future", False)
            ),
            "diagnostic_far_depth_classification": det.get(
                "diagnostic_far_depth_classification", ""
            ),
            "track": track_id,
            "memory": bool(result is not None and result.get("accepted", False)),
            "state": (
                "stable" if tr is not None and getattr(tr, "is_stable", False)
                else "tentative" if tr is not None else ""
            ),
            "race_idx": race_idx,
            "role": role,
            "reason": rejection_reason or (result.get("reason", "") if result else ""),
        })

    @staticmethod
    def _ordered_image_quad(points):
        points = np.asarray(points, dtype=float).reshape(4, 2)
        by_y = points[np.argsort(points[:, 1])]
        top = by_y[:2][np.argsort(by_y[:2, 0])]
        bottom = by_y[2:][np.argsort(by_y[2:, 0])]
        return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=float)

    def _corner_measurement_race_index(self, track_id, world_center):
        canonical_id = self.canonical_track_id(track_id)
        order = self.race_progression.order()
        if canonical_id in order:
            return int(order.index(canonical_id))

        track = (
            self.gate_memory.get_track_by_id(canonical_id)
            if canonical_id is not None
            else None
        )
        track_race_index = getattr(track, "race_order_index", None)
        if track_race_index is not None:
            return int(track_race_index)

        if len(self.gt_gates) == 0:
            return None
        center = np.asarray(world_center, dtype=float).reshape(3)
        if not np.all(np.isfinite(center)):
            return None
        distances = [
            float(np.linalg.norm(center - np.asarray(gate, dtype=float).reshape(3)))
            for gate in self.gt_gates
        ]
        return int(np.argmin(distances)) if min(distances) <= 4.0 else None

    def _project_debug_gt_gate_corners(
        self,
        race_index,
        camera_pose_world,
        camera_matrix,
        dist_coeffs,
    ):
        if race_index is None or not (0 <= int(race_index) < len(self.gt_gates)):
            return None, None
        gate_center = np.asarray(self.gt_gates[int(race_index)], dtype=float).reshape(3)
        half = 0.75
        corners_world = np.array([
            gate_center + [-half, 0.0, half],
            gate_center + [half, 0.0, half],
            gate_center + [half, 0.0, -half],
            gate_center + [-half, 0.0, -half],
        ], dtype=float)
        camera_pose_world = np.asarray(camera_pose_world, dtype=float).reshape(4, 4)
        r_wc = camera_pose_world[:3, :3]
        p_wc = camera_pose_world[:3, 3]
        corners_camera = (r_wc.T @ (corners_world - p_wc).T).T
        if not np.all(corners_camera[:, 2] > 0.0):
            return gate_center, None
        projected, _ = cv2.projectPoints(
            corners_camera.reshape(-1, 1, 3),
            np.zeros((3, 1), dtype=float),
            np.zeros((3, 1), dtype=float),
            np.asarray(camera_matrix, dtype=float).reshape(3, 3),
            np.asarray(dist_coeffs, dtype=float).reshape(-1, 1),
        )
        return gate_center, self._ordered_image_quad(projected.reshape(4, 2))

    @staticmethod
    def _json_safe_number(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    @classmethod
    def _json_safe_array(cls, value):
        array = np.asarray(value, dtype=float)
        return [
            cls._json_safe_number(item)
            for item in array.reshape(-1)
        ]

    def record_corner_measurement(
        self,
        det,
        result,
        camera_matrix,
        dist_coeffs,
    ):
        if result is None:
            return None

        raw_keypoints = np.asarray(
            det.get("yolo_keypoints", np.full((4, 3), np.nan)),
            dtype=float,
        ).reshape(-1, 3)
        if raw_keypoints.shape[0] < 4:
            return None
        keypoints_px = raw_keypoints[:4, :2].copy()
        keypoint_conf = raw_keypoints[:4, 2].copy()
        visibility = (
            np.isfinite(keypoints_px).all(axis=1)
            & np.isfinite(keypoint_conf)
            & (keypoint_conf > 0.0)
            & (keypoints_px[:, 0] >= 0.0)
            & (keypoints_px[:, 0] < float(self.image_width))
            & (keypoints_px[:, 1] >= 0.0)
            & (keypoints_px[:, 1] < float(self.image_height))
        )

        rpy = np.asarray(
            self.image_pose_snapshot_rpy_perception, dtype=float
        ).reshape(3)
        r_wb = self.perception_node._rpy_to_rotmat(*map(float, rpy))
        r_body_camera = self.body_camera_matrix_for_mode(
            self.perception_transform_mode
        )
        camera_pose_world = np.eye(4, dtype=float)
        camera_pose_world[:3, :3] = r_wb @ r_body_camera
        camera_pose_world[:3, 3] = (
            self.image_pose_snapshot_position
            + r_wb @ self.camera_offset_body
        )

        associated_track_id = result.get("track_id")
        race_index = self._corner_measurement_race_index(
            associated_track_id,
            det.get("gate_center_world", np.full(3, np.nan)),
        )
        gt_center, projected_gt = self._project_debug_gt_gate_corners(
            race_index,
            camera_pose_world,
            camera_matrix,
            dist_coeffs,
        )
        residuals = (
            keypoints_px - projected_gt
            if projected_gt is not None
            else np.full((4, 2), np.nan, dtype=float)
        )
        track = (
            self.gate_memory.get_track_by_id(associated_track_id)
            if associated_track_id is not None
            else None
        )
        filtered_center = (
            np.asarray(track.filtered_center_world, dtype=float).reshape(3)
            if track is not None
            and getattr(track, "filtered_center_world", None) is not None
            else np.full(3, np.nan, dtype=float)
        )

        measurement = CornerMeasurement(
            image_stamp=(self.image_stamp_sec, self.image_stamp_nanosec),
            camera_pose_world=camera_pose_world,
            keypoints_px=keypoints_px,
            keypoint_conf=keypoint_conf,
            visibility=visibility,
            associated_track_id=associated_track_id,
            race_index_candidate=race_index,
        )
        self.corner_measurements.append(measurement)
        self.corner_measurements_frame.append(measurement)

        bbox = np.asarray(
            det.get("yolo_bbox", np.full(4, np.nan)), dtype=float
        ).reshape(4)
        record = {
            "image_stamp": (
                f"{self.image_stamp_sec}.{self.image_stamp_nanosec:09d}"
            ),
            "vehicle_pose_snapshot": {
                "position": self._json_safe_array(self.image_pose_snapshot_position),
                "rpy_raw_rad": self._json_safe_array(self.image_pose_snapshot_rpy_raw),
                "rpy_perception_rad": self._json_safe_array(
                    self.image_pose_snapshot_rpy_perception
                ),
                "pose_stamp": self._json_safe_number(
                    self.pose_stamp_used_for_detection
                ),
            },
            "camera_pose_world": self._json_safe_array(camera_pose_world),
            "camera_intrinsics": self._json_safe_array(camera_matrix),
            "dist_coeffs": self._json_safe_array(dist_coeffs),
            "detection_index": int(
                det.get("detection_index", det.get("processed_detection_index", -1))
            ),
            "associated_track_id": associated_track_id,
            "race_index_candidate": race_index,
            "expected_gate_index": race_index,
            "memory_accepted": bool(result.get("accepted", False)),
            "memory_result_reason": str(result.get("reason", "")),
            "keypoints_px": self._json_safe_array(keypoints_px),
            "keypoint_conf": self._json_safe_array(keypoint_conf),
            "visibility": [bool(value) for value in visibility],
            "bbox_xyxy": self._json_safe_array(bbox),
            "bbox_confidence": self._json_safe_number(
                det.get("yolo_box_confidence", det.get("yolo_confidence", np.nan))
            ),
            "pnp_world_estimate": self._json_safe_array(
                det.get("gate_center_world", np.full(3, np.nan))
            ),
            "gate_memory_filtered_estimate": self._json_safe_array(filtered_center),
            "gt_gate_center": (
                self._json_safe_array(gt_center) if gt_center is not None else None
            ),
            "projected_gt_corners_px": (
                self._json_safe_array(projected_gt)
                if projected_gt is not None
                else None
            ),
            "corner_residuals_px": (
                self._json_safe_array(residuals)
                if projected_gt is not None
                else None
            ),
        }
        self.corner_measurement_records_frame.append(record)
        self.corner_measurement_count = len(self.corner_measurements_frame)
        self.corner_measurements_log = json.dumps(
            self.corner_measurement_records_frame,
            separators=(",", ":"),
            allow_nan=False,
        )
        return measurement

    def finalize_detection_flow_debug(self):
        parts = []
        yolo_confidence = []
        quad_area_px2 = []
        old_area_confidence = []
        memory_confidence_used = []
        memory_admission_threshold = []
        memory_admission_passed = []
        pnp_camera_original = []
        pnp_camera_depth_corrected = []
        depth_correction_factor = []
        world_original = []
        world_depth_corrected = []
        for idx in sorted(self.perception_detection_flow_entries):
            e = self.perception_detection_flow_entries[idx]
            vec = lambda v: "/".join(f"{x:.2f}" for x in np.asarray(v, dtype=float).reshape(3))
            yolo_confidence.append(f"det{idx}:{e.get('yolo_confidence', np.nan):.3f}")
            quad_area_px2.append(f"det{idx}:{e.get('quad_area_px2', np.nan):.1f}")
            old_area_confidence.append(f"det{idx}:{e.get('old_area_confidence', np.nan):.3f}")
            memory_confidence_used.append(f"det{idx}:{e.get('memory_confidence', np.nan):.3f}")
            memory_admission_threshold.append(
                f"det{idx}:{e.get('memory_admission_threshold', np.nan):.3f}"
            )
            memory_admission_passed.append(
                f"det{idx}:{int(bool(e.get('memory_admission_passed', False)))}"
            )
            pnp_camera_original.append(
                f"det{idx}:{vec(e.get('pnp_camera_original', np.full(3,np.nan)))}"
            )
            pnp_camera_depth_corrected.append(
                f"det{idx}:{vec(e.get('pnp_camera_depth_corrected', np.full(3,np.nan)))}"
            )
            depth_correction_factor.append(
                f"det{idx}:{e.get('depth_correction_factor', 1.0):.6f}"
            )
            world_original.append(
                f"det{idx}:{vec(e.get('world_original', np.full(3,np.nan)))}"
            )
            world_depth_corrected.append(
                f"det{idx}:{vec(e.get('world_depth_corrected', np.full(3,np.nan)))}"
            )
            parts.append(
                f"det{idx}:yolo={e.get('yolo_confidence', np.nan):.3f},"
                f"area_px2={e.get('quad_area_px2', np.nan):.1f},"
                f"old_area_conf={e.get('old_area_confidence', np.nan):.3f},"
                f"mem_conf={e.get('memory_confidence', np.nan):.3f},"
                f"mem_pass={int(bool(e.get('memory_admission_passed', False)))},"
                f"pnp={int(bool(e.get('pnp')))},"
                f"cam={vec(e.get('cam', np.full(3,np.nan)))},"
                f"raw={vec(e.get('raw', np.full(3,np.nan)))},"
                f"corrected={vec(e.get('corrected', np.full(3,np.nan)))},"
                f"depth_diag_future={int(bool(e.get('diagnostic_far_depth_is_future', False)))},"
                f"depth_diag_class={e.get('diagnostic_far_depth_classification','')},"
                f"depth_factor={e.get('depth_correction_factor', 1.0):.6f},"
                f"track={e.get('track')},mem={int(bool(e.get('memory')))},"
                f"state={e.get('state','')},race={e.get('race_idx')},"
                f"role={e.get('role','')},reason={e.get('reason','')}"
            )
        self.perception_detection_flow = ";".join(parts)
        self.yolo_confidence = ";".join(yolo_confidence)
        self.quad_area_px2 = ";".join(quad_area_px2)
        self.old_area_confidence = ";".join(old_area_confidence)
        self.memory_confidence_used = ";".join(memory_confidence_used)
        self.memory_admission_threshold = ";".join(memory_admission_threshold)
        self.memory_admission_passed = ";".join(memory_admission_passed)
        self.pnp_camera_original = ";".join(pnp_camera_original)
        self.pnp_camera_depth_corrected = ";".join(pnp_camera_depth_corrected)
        self.depth_correction_factor = ";".join(depth_correction_factor)
        self.world_original = ";".join(world_original)
        self.world_depth_corrected = ";".join(world_depth_corrected)
        if parts:
            print("[DETECTION FLOW] " + self.perception_detection_flow)

    def add_lookahead_pipeline_reason(self, label, reason):
        compact = self._compact_reason(reason)
        if not compact:
            return
        entry = f"{label}:{compact}"
        parts = [p for p in str(self.lookahead_pipeline_reasons or "").split(";") if p]
        parts.append(entry)
        self.lookahead_pipeline_reasons = ";".join(parts[-12:])
        self.update_lookahead_pipeline_debug()

    def update_lookahead_pipeline_debug(self):
        self.tentative_track_count = len(getattr(self, "tentative_track_ids", []) or [])
        eligible = 0
        eligible_track_ids = []
        current_pos = np.array([
            self.telemetry.pos["x"],
            self.telemetry.pos["y"],
            self.telemetry.pos["z"],
        ], dtype=float)
        selected_ids = set()
        existing_points = []
        for tr in getattr(self.gate_memory, "tracks", []):
            track_id = self.canonical_track_id(getattr(tr, "id", None))
            if track_id is None:
                continue
            if bool(getattr(tr, "is_stable", False)):
                continue
            center = getattr(tr, "filtered_center_world", None)
            if center is None:
                center = getattr(tr, "center", None)
            if center is None:
                continue
            center, safe_reason = self._planning_target_safe_center(center)
            if center is None:
                compact = self._compact_reason(safe_reason)
                if compact:
                    parts = [p for p in str(self.lookahead_pipeline_reasons or "").split(";") if p]
                    parts.append(f"track{track_id}:{compact}")
                    self.lookahead_pipeline_reasons = ";".join(parts[-12:])
                continue
            reason = self._tentative_lookahead_rejection(
                tr=tr,
                track_id=track_id,
                center=center,
                current_pos=current_pos,
                selected_ids=selected_ids,
                existing_points=existing_points,
            )
            if reason:
                compact = self._compact_reason(reason)
                if compact:
                    parts = [p for p in str(self.lookahead_pipeline_reasons or "").split(";") if p]
                    parts.append(f"track{track_id}:{compact}")
                    self.lookahead_pipeline_reasons = ";".join(parts[-12:])
            else:
                eligible += 1
                eligible_track_ids.append(track_id)
        self.tentative_lookahead_eligible_count = eligible
        self.tentative_lookahead_eligible_track_ids = eligible_track_ids
        self.lookahead_pipeline_debug = (
            f"raw={int(self.yolo_raw_count)},"
            f"pnp={int(self.pnp_success_count)},"
            f"valid={int(self.world_valid_count)},"
            f"mem={int(self.memory_update_count)},"
            f"tentative={int(self.tentative_track_count)},"
            f"eligible={int(self.tentative_lookahead_eligible_count)},"
            f"used={int(bool(self.tentative_lookahead_used))},"
            f"reasons={self.lookahead_pipeline_reasons}"
        )

    def check_tentative_lookahead_new_candidate_replan(self, now=None):
        now = time.time() if now is None else float(now)
        self.tentative_lookahead_replan_requested = False
        self.tentative_lookahead_replan_blocked_reason = ""
        self.tentative_lookahead_replan_suppressed = False
        self.tentative_lookahead_replan_suppression_reason = ""
        self.horizon_material_change_m = float("nan")

        if not self.use_perception:
            self.tentative_lookahead_replan_blocked_reason = "perception_disabled"
            return False
        if self.gate_completion_triggered:
            self.tentative_lookahead_replan_blocked_reason = "gate_completed_same_cycle"
            return False
        active_id = self.canonical_track_id(getattr(self, "active_target_track_id", None))
        if active_id is None:
            self.tentative_lookahead_replan_blocked_reason = "no_active_target"
            return False
        eligible_ids = [
            self.canonical_track_id(tid)
            for tid in getattr(self, "tentative_lookahead_eligible_track_ids", [])
        ]
        eligible_ids = [tid for tid in eligible_ids if tid is not None]
        if len(eligible_ids) == 0:
            self.tentative_lookahead_replan_blocked_reason = "no_eligible_tentative_track"
            return False
        future_ids = [tid for tid in eligible_ids if tid != active_id]
        if len(future_ids) == 0:
            self.tentative_lookahead_replan_blocked_reason = "eligible_is_active_target"
            return False

        horizon_ids = {
            self.canonical_track_id(tid)
            for tid in self.active_target_track_ids[1:]
            if tid is not None and int(tid) >= 0
        }
        horizon_future_centers = [
            np.asarray(center, dtype=float).reshape(3)
            for center in self.active_target_gates[1:]
            if np.all(np.isfinite(np.asarray(center, dtype=float).reshape(3)))
        ]
        horizon_has_future = len(self.active_target_gates) >= 2
        horizon_stale = bool(
            self.active_waypoints is None
            or self.active_times is None
            or (
                self.planner.total_time > 0.0
                and self.time_elapsed >= self.planner.total_time
            )
        )
        current_target_valid = False
        if len(self.active_target_gates) > 0:
            current_target_valid, _ = self.validate_planning_target(
                self.active_target_gates[0]
            )
        horizon_invalid = not current_target_valid
        race_order = {
            self.canonical_track_id(tid)
            for tid in self.race_progression.order()
        }
        new_race_order_ids = [
            tid for tid in future_ids
            if tid in race_order and tid not in horizon_ids
        ]

        material_changes = []
        for track_id in future_ids:
            tr = self.gate_memory.get_track_by_id(track_id)
            if tr is None:
                continue
            center = getattr(tr, "filtered_center_world", None)
            if center is None:
                center = getattr(tr, "center", None)
            if center is None:
                continue
            center = np.asarray(center, dtype=float).reshape(3)
            if not np.all(np.isfinite(center)):
                continue
            if track_id in horizon_ids:
                idx = next(
                    (
                        i for i, tid in enumerate(self.active_target_track_ids[1:], start=1)
                        if self.canonical_track_id(tid) == track_id
                    ),
                    None,
                )
                if idx is not None:
                    material_changes.append(
                        float(np.linalg.norm(center - self.active_target_gates[idx]))
                    )
            elif horizon_future_centers:
                material_changes.append(
                    min(float(np.linalg.norm(center - point)) for point in horizon_future_centers)
                )
            else:
                material_changes.append(float("inf"))
        if material_changes:
            self.horizon_material_change_m = max(material_changes)

        if (
            self.suppress_minor_tentative_lookahead_replans
            and horizon_has_future
            and not horizon_stale
            and not horizon_invalid
            and not new_race_order_ids
            and np.isfinite(self.horizon_material_change_m)
            and self.horizon_material_change_m <= self.tentative_lookahead_replan_min_shift
        ):
            self.tentative_lookahead_replan_suppressed = True
            self.tentative_lookahead_replan_suppression_reason = (
                "internal_passthrough_horizon_minor_lookahead_change"
            )
            self.tentative_lookahead_replan_blocked_reason = (
                self.tentative_lookahead_replan_suppression_reason
            )
            self.tentative_lookahead_shift_m = self.horizon_material_change_m
            print(
                "[TENTATIVE LOOKAHEAD] replan suppressed "
                f"change={self.horizon_material_change_m:.3f}m "
                f"threshold={self.tentative_lookahead_replan_min_shift:.3f}m "
                f"reason={self.tentative_lookahead_replan_suppression_reason}"
            )
            return False
        dt = now - float(getattr(self, "replan_time", 0.0))
        if dt <= 0.5:
            self.tentative_lookahead_replan_blocked_reason = f"replan_throttled:{dt:.2f}"
            return False

        self.tentative_lookahead_replan_requested = True
        self.tentative_lookahead_replan_blocked_reason = ""
        return True

    def update_gate_memory_from_frame(
        self,
        frame,
        camera_matrix,
        dist_coeffs,
        image_stamp_sec=0,
        image_stamp_nanosec=0,
        image_received_wall_time=np.nan,
        image_pose_snapshot=None,
        gazebo_pose=None,
    ):
        if not self.use_perception or self.perception_node is None:
            return None

        self.reset_yaw_image_consistency_debug()
        self.reset_gazebo_pose_comparison_debug()
        self.initialize_perception_yaw_correction(gazebo_pose)
        self.reset_pnp_selection_debug()
        self.reset_association_debug_log_fields()
        self.reset_lookahead_pipeline_debug()
        self.corner_measurements_frame = []
        self.corner_measurement_records_frame = []
        self.corner_measurement_count = 0
        self.corner_measurements_log = ""
        self.perception_world_pose_source_used = str(
            self.perception_world_pose_source
        )
        self.world_from_mavsdk = np.full(3, np.nan, dtype=float)
        self.world_from_gazebo_truth = np.full(3, np.nan, dtype=float)
        self.selected_world_estimate = np.full(3, np.nan, dtype=float)
        self.selected_vs_mavsdk_world_delta = np.full(3, np.nan, dtype=float)
        self.selected_vs_gazebo_world_delta = np.full(3, np.nan, dtype=float)

        process_wall_time = time.time()
        self.image_stamp_sec = int(image_stamp_sec)
        self.image_stamp_nanosec = int(image_stamp_nanosec)
        self.image_received_wall_time = float(image_received_wall_time)
        self.image_processed_wall_time = process_wall_time
        current_position_sample_time = float(
            getattr(self.telemetry, "position_sample_time", np.nan)
        )
        current_attitude_sample_time = float(
            getattr(self.telemetry, "attitude_sample_time", np.nan)
        )
        current_stamps = [
            stamp
            for stamp in (current_position_sample_time, current_attitude_sample_time)
            if np.isfinite(stamp)
        ]
        self.telemetry_stamp_current = (
            max(current_stamps) if current_stamps else float("nan")
        )
        if self.image_stamp_sec != 0 or self.image_stamp_nanosec != 0:
            image_key = ("ros", self.image_stamp_sec, self.image_stamp_nanosec)
        elif np.isfinite(self.image_received_wall_time):
            image_key = ("wall", self.image_received_wall_time)
        else:
            image_key = None
        self.skipped_stale_image = (
            image_key is not None and image_key == self.last_processed_image_stamp
        )
        self.duplicate_image_skipped = self.skipped_stale_image
        self.skipped_image_stamp = (
            f"{self.image_stamp_sec}.{self.image_stamp_nanosec:09d}"
            if self.skipped_stale_image
            else ""
        )
        self.detection_world_computed_once = False
        if self.skipped_stale_image:
            print(
                "[PERCEPTION FRAME] duplicate image skipped "
                f"stamp={self.skipped_image_stamp}"
            )
            return None
        if image_key is not None:
            self.last_processed_image_stamp = image_key

        snapshot = image_pose_snapshot if isinstance(image_pose_snapshot, dict) else {}
        image_gazebo_pose = snapshot.get("gazebo_pose")
        if not isinstance(image_gazebo_pose, dict):
            image_gazebo_pose = gazebo_pose
        self.image_gazebo_pose_snapshot = image_gazebo_pose
        snapshot_position = np.asarray(
            snapshot.get(
                "position",
                [
                    self.telemetry.pos["x"],
                    self.telemetry.pos["y"],
                    self.telemetry.pos["z"],
                ],
            ),
            dtype=float,
        ).reshape(3)
        snapshot_rpy_raw = np.asarray(
            snapshot.get(
                "rpy_raw_rad",
                [
                    self.telemetry.rpy["roll"],
                    self.telemetry.rpy["pitch"],
                    self.telemetry.rpy["yaw"],
                ],
            ),
            dtype=float,
        ).reshape(3)
        self.telemetry_position_sample_time = float(
            snapshot.get("position_sample_time", current_position_sample_time)
        )
        self.telemetry_attitude_sample_time = float(
            snapshot.get("attitude_sample_time", current_attitude_sample_time)
        )
        pose_stamps = [
            stamp
            for stamp in (
                self.telemetry_position_sample_time,
                self.telemetry_attitude_sample_time,
            )
            if np.isfinite(stamp)
        ]
        self.pose_stamp_used_for_detection = (
            min(pose_stamps) if pose_stamps else float("nan")
        )
        self.image_pose_age_s = (
            self.image_received_wall_time - self.pose_stamp_used_for_detection
            if np.isfinite(self.image_received_wall_time)
            and np.isfinite(self.pose_stamp_used_for_detection)
            else float("nan")
        )
        self.image_pose_snapshot_position = snapshot_position.copy()
        self.image_pose_snapshot_rpy_raw = snapshot_rpy_raw.copy()
        if np.isfinite(self.image_received_wall_time):
            self.image_age_s = process_wall_time - self.image_received_wall_time
        if np.isfinite(self.telemetry_attitude_sample_time):
            self.attitude_age_s = process_wall_time - self.telemetry_attitude_sample_time
        if np.isfinite(self.telemetry_position_sample_time):
            self.position_age_s = process_wall_time - self.telemetry_position_sample_time
        if (
            np.isfinite(self.image_received_wall_time)
            and np.isfinite(self.telemetry_attitude_sample_time)
            and np.isfinite(self.telemetry_position_sample_time)
        ):
            pose_sample_time = min(
                self.telemetry_attitude_sample_time,
                self.telemetry_position_sample_time,
            )
            self.pose_age_relative_to_image_s = (
                self.image_received_wall_time - pose_sample_time
            )

        drone_pos = snapshot_position
        telemetry_rpy_raw_rad = snapshot_rpy_raw
        self.last_telemetry_rpy_raw_rad = telemetry_rpy_raw_rad.copy()
        self.telemetry_yaw_raw_deg = math.degrees(float(telemetry_rpy_raw_rad[2]))
        self.update_dynamic_gazebo_perception_yaw_correction(
            gazebo_pose=image_gazebo_pose,
            telemetry_yaw_raw_rad=float(telemetry_rpy_raw_rad[2]),
        )
        drone_rpy_for_perception = self.perception_rpy_for_mode(
            telemetry_rpy_raw_rad,
            self.perception_transform_mode,
        )
        self.image_pose_snapshot_rpy_perception = np.asarray(
            drone_rpy_for_perception, dtype=float
        ).reshape(3).copy()
        self.telemetry_yaw_perception_deg = math.degrees(
            float(drone_rpy_for_perception[2])
        )
        if hasattr(self.telemetry, "yaw_rad_raw"):
            self.telemetry.yaw_rad_raw = float(telemetry_rpy_raw_rad[2])
            self.telemetry.yaw_rad_perception = float(drone_rpy_for_perception[2])
            self.telemetry.perception_yaw_correction_rad = (
                self.active_perception_yaw_correction_rad
            )
        self.last_detection_drone_pose = np.array([
            drone_pos[0],
            drone_pos[1],
            drone_pos[2],
            drone_rpy_for_perception[0],
            drone_rpy_for_perception[1],
            drone_rpy_for_perception[2],
        ], dtype=float)
        self.detection_drone_yaw_deg = math.degrees(float(drone_rpy_for_perception[2]))
        self.capture_gazebo_pose_debug(
            gazebo_pose=image_gazebo_pose,
            mavsdk_pos=drone_pos,
            mavsdk_yaw_rad=float(telemetry_rpy_raw_rad[2]),
            process_wall_time=process_wall_time,
        )

        if self.perception_rpy_debug_frames_remaining > 0:
            print(
                "[PERCEPTION RPY DEBUG] "
                f"mode={self.perception_transform_mode} "
                f"telemetry_rpy_raw_rad={telemetry_rpy_raw_rad.tolist()} "
                f"telemetry_yaw_deg={math.degrees(float(self.telemetry.rpy['yaw'])):.2f} "
                f"drone_rpy_for_perception={drone_rpy_for_perception.tolist()} "
                f"drone_rpy_for_perception_yaw_deg={math.degrees(float(drone_rpy_for_perception[2])):.2f}"
            )
            self.perception_rpy_debug_frames_remaining -= 1

        detections = self.perception_node.detect_gates(
            frame=frame,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            drone_pos=drone_pos,
            drone_rpy_rad=drone_rpy_for_perception,
        )
        self.detection_world_computed_once = True
        node_debug = getattr(self.perception_node, "last_pipeline_debug", {})
        yolo = getattr(self.perception_node, "gate_perception", None)
        self.initialize_detection_flow_debug(yolo)
        self.yolo_raw_count = int(node_debug.get("yolo_raw_count", 0))
        self.yolo_detection_count = int(getattr(yolo, "last_yolo_detection_count", self.yolo_raw_count))
        self.pnp_success_count = int(node_debug.get("pnp_success_count", 0))
        self.world_valid_count = int(node_debug.get("world_valid_count", len(detections)))
        self.processed_detection_indices = list(node_debug.get("processed_detection_indices", []))
        self.yolo_detection_confidences = self._compact_float_list(
            getattr(yolo, "last_yolo_detection_confidences", []),
            precision=3,
        )
        self.yolo_detection_bboxes = self._compact_nested_points(
            getattr(yolo, "last_yolo_detection_bboxes", []),
            precision=1,
        )
        self.yolo_detection_keypoints = self._compact_nested_points(
            getattr(yolo, "last_yolo_detection_keypoints", []),
            precision=1,
        )
        for meta in getattr(yolo, "last_yolo_candidate_debug", []):
            reason = meta.get("rejection_reason", "")
            if reason:
                self.add_lookahead_pipeline_reason(
                    f"det{int(meta.get('detection_index', -1))}",
                    reason,
                )

        now = time.time()

        if len(detections) == 0:
            if self.yolo_raw_count > 0 and self.pnp_success_count == 0:
                self.add_lookahead_pipeline_reason("det*", "no_pnp_solution")
            self.gate_memory.prune(now)
            self.last_raw_gate_center = None
            self.last_perception_accepted = False
            self.last_perception_rejection_reason = "no_detection"
            self.reset_transform_validation_debug()
            self.update_quad_debug(None, frame.shape)
            self.update_gate_filter_summary_logs()
            self.update_lookahead_pipeline_debug()
            self.finalize_detection_flow_debug()
            return None

        expected_gate_world_for_debug = None
        if len(self.gt_gates) > 0:
            gate_idx = int(np.clip(self.current_gate_idx, 0, len(self.gt_gates) - 1))
            expected_gate_world_for_debug = np.asarray(self.gt_gates[gate_idx], dtype=float).reshape(3)
        detections = [
            self.select_pnp_candidate_for_live_geometry(
                det=live_det,
                drone_pos=drone_pos,
                telemetry_rpy_raw_rad=telemetry_rpy_raw_rad,
                expected_gate_world=expected_gate_world_for_debug,
            )
            for live_det in detections
        ]

        if self.perception_transform_mode in (
            "physical_direct_rad",
            "physical_direct_rad_x_mirror",
            "physical_mavsdk_yaw_aligned",
            "legacy_scaled_yaw",
        ):
            r_body_camera = self.body_camera_matrix_for_mode(self.perception_transform_mode)
            rpy = self.perception_rpy_for_mode(telemetry_rpy_raw_rad, self.perception_transform_mode)
            r_wb = self.perception_node._rpy_to_rotmat(float(rpy[0]), float(rpy[1]), float(rpy[2]))
            for live_det in detections:
                gate_camera = np.asarray(
                    live_det.get("gate_center_camera", np.full(3, np.nan)),
                    dtype=float,
                ).reshape(3)
                if not np.all(np.isfinite(gate_camera)):
                    continue
                transform_result = self.transform_gate_camera_to_world_for_pose_source(
                    gate_camera=gate_camera,
                    drone_pos=drone_pos,
                    telemetry_rpy_raw_rad=telemetry_rpy_raw_rad,
                    mode=self.perception_transform_mode,
                    r_body_camera=r_body_camera,
                    gazebo_pose=image_gazebo_pose,
                )
                gate_world = transform_result["selected_world"]
                gate_body = transform_result["gate_body"]
                live_det["gate_center_body"] = gate_body.copy()
                live_det["gate_center_cam"] = gate_body.copy()
                live_det["gate_center_world"] = gate_world.copy()
                live_det["world_from_mavsdk"] = transform_result[
                    "world_from_mavsdk"
                ].copy()
                live_det["world_from_gazebo_truth"] = transform_result[
                    "world_from_gazebo_truth"
                ].copy()
                live_det["selected_world_estimate"] = gate_world.copy()
                live_det["perception_world_pose_source_used"] = transform_result[
                    "source_used"
                ]
                live_det["camera_to_body_matrix_used"] = r_body_camera.copy()
                live_det["body_to_world_method_used"] = self.perception_transform_mode
                gate_normal_camera = np.asarray(
                    live_det.get("gate_normal_camera", np.full(3, np.nan)),
                    dtype=float,
                ).reshape(3)
                if np.all(np.isfinite(gate_normal_camera)):
                    gate_normal_body = r_body_camera @ gate_normal_camera
                    live_det["gate_normal_body"] = gate_normal_body.copy()
                    live_det["gate_normal_world"] = (
                        transform_result["selected_rotation_world_body"]
                        @ gate_normal_body
                    )

        detections = self.apply_diagnostic_far_depth_correction(
            detections=detections,
            drone_pos=drone_pos,
            telemetry_rpy_raw_rad=telemetry_rpy_raw_rad,
            gazebo_pose=image_gazebo_pose,
        )
        det = detections[0]

        if self.perception_transform_mode in (
            "physical_direct_rad",
            "physical_direct_rad_x_mirror",
            "physical_mavsdk_yaw_aligned",
            "legacy_scaled_yaw",
        ):
            gate_camera = np.asarray(det.get("gate_center_camera", np.full(3, np.nan)), dtype=float).reshape(3)
            if np.all(np.isfinite(gate_camera)):
                r_body_camera = self.body_camera_matrix_for_mode(self.perception_transform_mode)
                transform_result = self.transform_gate_camera_to_world_for_pose_source(
                    gate_camera=gate_camera,
                    drone_pos=drone_pos,
                    telemetry_rpy_raw_rad=telemetry_rpy_raw_rad,
                    mode=self.perception_transform_mode,
                    r_body_camera=r_body_camera,
                    gazebo_pose=image_gazebo_pose,
                )
                gate_world = transform_result["selected_world"]
                gate_body = transform_result["gate_body"]
                gate_normal_camera = np.asarray(
                    det.get("gate_normal_camera", np.full(3, np.nan)),
                    dtype=float,
                ).reshape(3)
                det["gate_center_body"] = gate_body.copy()
                det["gate_center_cam"] = gate_body.copy()
                det["gate_center_world"] = gate_world.copy()
                det["world_from_mavsdk"] = transform_result[
                    "world_from_mavsdk"
                ].copy()
                det["world_from_gazebo_truth"] = transform_result[
                    "world_from_gazebo_truth"
                ].copy()
                det["selected_world_estimate"] = gate_world.copy()
                det["perception_world_pose_source_used"] = transform_result[
                    "source_used"
                ]
                det["camera_to_body_matrix_used"] = r_body_camera.copy()
                det["body_to_world_method_used"] = self.perception_transform_mode
                self.update_perception_world_pose_source_debug(transform_result)
                if np.all(np.isfinite(gate_normal_camera)):
                    gate_normal_body = r_body_camera @ gate_normal_camera
                    det["gate_normal_body"] = gate_normal_body.copy()
                    det["gate_normal_world"] = (
                        transform_result["selected_rotation_world_body"]
                        @ gate_normal_body
                    )

        self.gate_confidence = float(det["confidence"])
        raw_center = np.asarray(det["gate_center_world"], dtype=float).reshape(3)
        reproj = float(det.get("reprojection_error", np.nan))
        if np.isfinite(reproj) and reproj > 6.0:
            self.gate_memory.prune(now)
            self.last_perception_accepted = False
            self.last_perception_rejection_reason = f"reprojection_error_high:{reproj:.2f}"
            self.add_lookahead_pipeline_reason(
                f"det{int(det.get('detection_index', 0))}",
                self.last_perception_rejection_reason,
            )
            if len(detections) > 1:
                self.add_supplemental_gate_detections(
                    detections[1:],
                    frame_shape=frame.shape,
                    timestamp=now,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                )
            self.update_gate_filter_summary_logs()
            self.update_lookahead_pipeline_debug()
            self.update_detection_flow_debug(
                det, rejection_reason=self.last_perception_rejection_reason
            )
            self.finalize_detection_flow_debug()
            return {
                "accepted": False,
                "reason": self.last_perception_rejection_reason,
                "track_id": None,
                "committed_now": False,
                "committed": False,
                "center": raw_center,
            }
        self.last_raw_gate_center = raw_center.copy()
        self.last_raw_image_corners = det.get("raw_corners", None)
        self.last_ordered_image_corners = det.get("ordered_corners", None)
        self.last_pnp_debug_best_ordered_corners = det.get("pnp_debug_best_ordered_corners", None)
        self.last_reprojected_image_corners = det.get("reprojected_corners", None)
        self.last_pnp_rvec = det.get("rvec", None)
        self.last_pnp_tvec = det.get("tvec", None)
        self.last_gate_center_camera = det.get("gate_center_camera", None)
        self.last_gate_center_body = det.get("gate_center_body", None)
        self.last_gate_center_world_debug = raw_center.copy()
        self.last_gate_normal_camera = det.get("gate_normal_camera", None)
        self.last_gate_normal_world = det.get("gate_normal_world", None)
        self.last_reprojection_error = float(det.get("reprojection_error", np.nan))
        self.last_corner_reprojection_error_px = float(det.get("corner_reprojection_error_px", np.nan))
        pnp_candidates = det.get("pnp_candidates", [])
        self.last_pnp_candidate_count = len(pnp_candidates) if pnp_candidates is not None else 0
        self.last_chosen_pnp_candidate = det.get("chosen_candidate", None)
        self.last_pnp_candidate_0_error = self.pnp_candidate_error(pnp_candidates, 0)
        self.last_pnp_candidate_1_error = self.pnp_candidate_error(pnp_candidates, 1)
        self.last_transform_source = det.get("transform_source", "")
        self.last_camera_to_body_matrix_used = det.get("camera_to_body_matrix_used", None)
        self.last_body_to_world_method_used = det.get("body_to_world_method_used", "")
        self.last_pnp_size_sweep = det.get("gate_size_sweep", {})
        self.last_pnp_formulation_debug = det.get("pnp_formulation_debug", [])
        self.last_camera_matrix = np.asarray(camera_matrix, dtype=float).copy()
        self.last_dist_coeffs = np.asarray(dist_coeffs, dtype=float).copy()
        self.last_live_solver_name = det.get("live_solver_name", "")
        self.last_pnp_fallback_reason = det.get("pnp_fallback_reason", "")
        self.pnp_selected_order = det.get("pnp_selected_order", "")
        self.pnp_selected_solver = det.get("pnp_selected_solver", "")
        self.pnp_selected_score = float(det.get("pnp_selected_score", np.nan))
        self.pnp_selected_reprojection_error = float(det.get("pnp_selected_reprojection_error", np.nan))
        self.pnp_selected_gate_center_camera = det.get("pnp_selected_gate_center_camera", None)
        self.pnp_selected_reason = det.get("pnp_selected_reason", "")
        self.pnp_candidate_summary = det.get("pnp_candidate_summary", "")
        self.pnp_candidate_world_summary = det.get("pnp_candidate_world_summary", "")
        self.pnp_selected_world_score = float(det.get("pnp_selected_world_score", np.nan))
        self.pnp_selected_world_reason = det.get("pnp_selected_world_reason", "")
        self.allow_pnp_corner_reordering = bool(det.get("allow_pnp_corner_reordering", False))
        self.pnp_live_candidate_orders_allowed = det.get("pnp_live_candidate_orders_allowed", "")
        self.pnp_debug_best_order = det.get("pnp_debug_best_order", "")
        self.pnp_live_vs_debug_best_order_mismatch = bool(
            det.get("pnp_live_vs_debug_best_order_mismatch", False)
        )
        self.pnp_lateral_angle = float(det.get("pnp_lateral_angle", np.nan))
        self.image_center_offset_normalized = float(det.get("image_center_offset_normalized", np.nan))
        self.keypoint_polygon_signed_area = float(det.get("keypoint_polygon_signed_area", np.nan))
        self.keypoint_polygon_winding = det.get("keypoint_polygon_winding", "")
        self.keypoint_edge_top = float(det.get("keypoint_edge_top", np.nan))
        self.keypoint_edge_right = float(det.get("keypoint_edge_right", np.nan))
        self.keypoint_edge_bottom = float(det.get("keypoint_edge_bottom", np.nan))
        self.keypoint_edge_left = float(det.get("keypoint_edge_left", np.nan))
        self.keypoint_bbox_center = det.get("keypoint_bbox_center", np.full(2, np.nan))
        self.keypoint_polygon_center = det.get("keypoint_polygon_center", np.full(2, np.nan))
        self.keypoint_bbox_polygon_delta = det.get("keypoint_bbox_polygon_delta", np.full(2, np.nan))
        self.raw_keypoint_polygon_signed_area = float(
            det.get("raw_keypoint_polygon_signed_area", np.nan)
        )
        self.raw_keypoint_polygon_winding = det.get("raw_keypoint_polygon_winding", "")
        self.update_quad_debug(self.last_ordered_image_corners, frame.shape)
        self.compute_yaw_image_consistency_debug(
            drone_pos=drone_pos,
            telemetry_yaw_rad=float(drone_rpy_for_perception[2]),
            camera_matrix=camera_matrix,
        )
        self.compute_gazebo_pose_gate_comparison_debug(
            pnp_camera=self.last_gate_center_camera,
            mavsdk_pos=drone_pos,
            mavsdk_rpy_raw=telemetry_rpy_raw_rad,
            perception_rpy=drone_rpy_for_perception,
        )

        image_valid, image_reason = self.validate_detection_image_bounds(
            det.get("ordered_corners", None),
            frame.shape,
        )
        if not image_valid:
            self.gate_memory.prune(now)
            self.last_perception_accepted = False
            self.last_perception_rejection_reason = image_reason
            self.add_lookahead_pipeline_reason(
                f"det{int(det.get('detection_index', 0))}",
                image_reason,
            )
            self.reset_transform_validation_debug()
            print(f"[PERCEPTION REJECT] reason={image_reason} corners={self.last_ordered_image_corners}")
            if len(detections) > 1:
                self.add_supplemental_gate_detections(
                    detections[1:],
                    frame_shape=frame.shape,
                    timestamp=now,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                )
            self.update_gate_filter_summary_logs()
            self.update_lookahead_pipeline_debug()
            self.update_detection_flow_debug(det, rejection_reason=image_reason)
            self.finalize_detection_flow_debug()
            self.save_perception_debug_frame(
                frame=frame,
                timestamp=now,
                track_id=None,
                accepted=False,
                rejection_reason=image_reason,
            )
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
            self.add_lookahead_pipeline_reason(
                f"det{int(det.get('detection_index', 0))}",
                reason,
            )
            if reason.startswith("z_below_safe_min"):
                self.future_track_visible_before_completion = True
                self.future_track_blocked_reason = reason
                self.remember_raw_planning_lookahead_candidate(raw_center, reason, now=now)
            self.gate_memory.prune(now)
            self.last_perception_accepted = False
            self.last_perception_rejection_reason = reason
            self.reset_transform_validation_debug()
            print(f"[PERCEPTION REJECT] reason={reason} raw_center={raw_center}")
            if len(detections) > 1:
                self.add_supplemental_gate_detections(
                    detections[1:],
                    frame_shape=frame.shape,
                    timestamp=now,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                )
            self.update_gate_filter_summary_logs()
            self.update_lookahead_pipeline_debug()
            self.update_detection_flow_debug(det, rejection_reason=reason)
            self.finalize_detection_flow_debug()
            self.save_perception_debug_frame(
                frame=frame,
                timestamp=now,
                track_id=None,
                accepted=False,
                rejection_reason=reason,
            )
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
            self.add_lookahead_pipeline_reason(
                f"det{int(det.get('detection_index', 0))}",
                self.last_perception_rejection_reason,
            )
            self.rejected_completed_this_lap = True
            self.target_rejected_completed = True
            self.reset_transform_validation_debug()
            print(f"[PERCEPTION REJECT] reason=near_completed_landmark_this_lap raw_center={raw_center}")
            if len(detections) > 1:
                self.add_supplemental_gate_detections(
                    detections[1:],
                    frame_shape=frame.shape,
                    timestamp=now,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                )
            self.update_gate_filter_summary_logs()
            self.update_lookahead_pipeline_debug()
            self.update_detection_flow_debug(
                det, rejection_reason="near_completed_landmark_this_lap"
            )
            self.finalize_detection_flow_debug()
            self.save_perception_debug_frame(
                frame=frame,
                timestamp=now,
                track_id=None,
                accepted=False,
                rejection_reason="near_completed_landmark_this_lap",
            )
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
            confidence=det["memory_confidence"],
            timestamp=now,
            center_camera=det.get("gate_center_camera", None),
            reprojection_error=det.get("reprojection_error", np.nan),
            solver_name=det.get("live_solver_name", ""),
            active_gate_idx=self.current_gate_idx,
        )
        if result is not None:
            self.memory_update_count += 1
        self.track_update_innovation = self.gate_memory.last_update_innovation
        self.track_update_accepted = self.gate_memory.last_update_accepted
        self.track_center_before_update = self.gate_memory.last_track_center_before
        self.track_center_after_update = self.gate_memory.last_track_center_after
        self.update_association_debug_log_fields()

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
                self.update_track_filter_log_fields(tr)
                if (
                    result.get("committed_now", False)
                    or result.get("stable_now", False)
                    or (result.get("committed", False) and not self.use_lookahead_gate_filter)
                ):
                    self.accept_track_into_race_order(tr)
            self.update_gate_filter_summary_logs()
            self.last_perception_accepted = bool(result.get("accepted", False))
            self.last_perception_rejection_reason = "" if self.last_perception_accepted else result.get("reason", "")
            if not self.last_perception_accepted:
                self.add_lookahead_pipeline_reason(
                    f"track{result.get('track_id')}",
                    self.last_perception_rejection_reason,
                )
            if self.last_perception_accepted:
                self.compute_transform_validation_debug(drone_pos, drone_rpy_for_perception)
            else:
                self.reset_transform_validation_debug()
            print(
                f"[MEM] reason={result['reason']} "
                f"track_id={result['track_id']} "
                f"committed={result['committed']} "
                f"committed_now={result['committed_now']} "
                f"stable={result.get('stable', False)} "
                f"stable_now={result.get('stable_now', False)} "
                f"center={result['center']}"
            )

            committed_tracks = self.gate_memory.get_committed_tracks()
            print("[MEM] committed tracks:")
            for tr in committed_tracks:
                print(
                    f"    id={tr.id}, center={tr.center}, hits={tr.hits}, "
                    f"stable={tr.is_stable}, score={tr.stability_score:.2f}"
            )
        self.record_corner_measurement(
            det=det,
            result=result,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
        self.update_detection_flow_debug(
            det,
            result=result,
            rejection_reason="" if result is not None else "memory_update_failed",
        )

        supplemental_result = self.add_supplemental_gate_detections(
            detections[1:],
            frame_shape=frame.shape,
            timestamp=now,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
        if supplemental_result is not None and (
            supplemental_result.get("stable_now", False)
            or (
                supplemental_result.get("committed_now", False)
                and not self.use_lookahead_gate_filter
            )
        ):
            result = supplemental_result

        self.update_gate_filter_summary_logs()
        self.update_lookahead_pipeline_debug()
        self.finalize_detection_flow_debug()
        if result is not None:
            self.save_perception_debug_frame(
                frame=frame,
                timestamp=now,
                track_id=result.get("track_id"),
                accepted=bool(result.get("accepted", False)),
                rejection_reason=(
                    "" if result.get("accepted", False)
                    else result.get("reason", "")
                ),
            )
        return result

    def add_supplemental_gate_detections(
        self,
        detections,
        frame_shape,
        timestamp,
        camera_matrix,
        dist_coeffs,
    ):
        selected_result = None
        for det in detections:
            det_label = f"det{int(det.get('detection_index', det.get('processed_detection_index', -1)))}"
            raw_center = np.asarray(det["gate_center_world"], dtype=float).reshape(3)

            image_valid, image_reason = self.validate_detection_image_bounds(
                det.get("ordered_corners", None),
                frame_shape,
            )
            if not image_valid:
                self.add_lookahead_pipeline_reason(det_label, image_reason)
                self.update_detection_flow_debug(det, rejection_reason=image_reason)
                continue

            valid, valid_reason = self.validate_perception_gate_center(
                raw_center,
                np.asarray(det.get("drone_pos", np.zeros(3)), dtype=float).reshape(3),
            )
            if not valid:
                self.add_lookahead_pipeline_reason(det_label, valid_reason)
                self.update_detection_flow_debug(det, rejection_reason=valid_reason)
                continue

            if self.is_near_completed_gate(raw_center, radius=self.gate_memory.new_track_block_radius):
                self.add_lookahead_pipeline_reason(det_label, "near_completed_gate")
                self.update_detection_flow_debug(det, rejection_reason="near_completed_gate")
                continue

            result = self.gate_memory.add_detection(
                center=raw_center,
                confidence=det["memory_confidence"],
                timestamp=timestamp,
                center_camera=det.get("gate_center_camera", None),
                reprojection_error=det.get("reprojection_error", np.nan),
                solver_name=det.get("live_solver_name", ""),
                active_gate_idx=self.current_gate_idx,
            )
            if result is not None:
                self.memory_update_count += 1
            self.update_association_debug_log_fields()
            tr = self.gate_memory.get_track_by_id(result.get("track_id")) if result is not None else None
            if tr is not None:
                self.track_id = tr.id
                self.landmark_uncertainty = self.gate_memory.track_uncertainty(tr.id)
                self.track_observations = tr.hits
                self.update_track_filter_log_fields(tr)
                if (
                    result.get("committed_now", False)
                    or result.get("stable_now", False)
                    or (result.get("committed", False) and not self.use_lookahead_gate_filter)
                ):
                    self.accept_track_into_race_order(tr)

            self.record_corner_measurement(
                det=det,
                result=result,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
            )

            if result is not None:
                selected_result = result
                print(
                    f"[MEM] supplemental reason={result['reason']} "
                    f"track_id={result['track_id']} "
                    f"committed={result['committed']} "
                    f"stable={result.get('stable', False)}"
                )
                if result.get("stable_now", False):
                    self.update_detection_flow_debug(det, result=result)
                    break
            self.update_detection_flow_debug(
                det,
                result=result,
                rejection_reason="" if result is not None else "memory_update_failed",
            )

        self.update_gate_filter_summary_logs()
        return selected_result

    def update_track_filter_log_fields(self, tr):
        self.track_history_len = len(getattr(tr, "obs_history", []))
        if self.track_history_len > 0:
            self.track_raw_latest_center = tr.obs_history[-1].center_world.copy()
        else:
            self.track_raw_latest_center = None
        self.track_filtered_center = (
            tr.filtered_center_world.copy()
            if getattr(tr, "filtered_center_world", None) is not None
            else tr.center.copy()
        )
        self.track_center_std = np.asarray(getattr(tr, "center_world_std", np.full(3, np.nan)), dtype=float)
        self.track_center_std_norm = (
            float(np.linalg.norm(self.track_center_std))
            if np.all(np.isfinite(self.track_center_std))
            else float("nan")
        )
        camera_std = np.asarray(getattr(tr, "center_camera_std", np.full(3, np.nan)), dtype=float)
        self.track_camera_std_norm = (
            float(np.linalg.norm(camera_std))
            if np.all(np.isfinite(camera_std))
            else float("nan")
        )
        self.track_reprojection_error_mean = float(getattr(tr, "reprojection_error_mean", np.nan))
        self.track_reprojection_error_median = float(getattr(tr, "reprojection_error_median", np.nan))
        self.track_outlier_count = int(getattr(tr, "outlier_count", 0))
        self.track_inlier_count = int(getattr(tr, "inlier_count", 0))
        self.track_is_stable = bool(getattr(tr, "is_stable", False))
        self.track_stability_score = float(getattr(tr, "stability_score", np.nan))
        self.promotion_blocked_reason = getattr(tr, "promotion_blocked_reason", "")
        self.promotion_reason = "stable" if self.track_is_stable else ""

    def update_association_debug_log_fields(self):
        self.nearest_track_id = self.gate_memory.nearest_track_id
        self.nearest_track_distance = float(self.gate_memory.nearest_track_distance)
        self.nearest_track_hits = int(self.gate_memory.nearest_track_hits)
        self.nearest_track_committed = bool(self.gate_memory.nearest_track_committed)
        self.nearest_track_stable = bool(self.gate_memory.nearest_track_stable)
        self.association_attempted = bool(self.gate_memory.association_attempted)
        self.association_success = bool(self.gate_memory.association_success)
        self.duplicate_rejection_reason = str(self.gate_memory.duplicate_rejection_reason or "")

    def reset_association_debug_log_fields(self):
        self.nearest_track_id = None
        self.nearest_track_distance = float("nan")
        self.nearest_track_hits = 0
        self.nearest_track_committed = False
        self.nearest_track_stable = False
        self.association_attempted = False
        self.association_success = False
        self.duplicate_rejection_reason = ""

    def update_gate_filter_summary_logs(self):
        self.tentative_track_ids = self.gate_memory.tentative_track_ids()
        self.stable_track_ids = self.gate_memory.stable_track_ids()
        self.race_admitted_track_ids = list(self.race_accepted_track_ids)

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

    def update_quad_debug(self, corners, frame_shape):
        if frame_shape is not None:
            self.last_image_center_x = 0.5 * float(frame_shape[1] - 1)
            self.last_image_center_y = 0.5 * float(frame_shape[0] - 1)
        else:
            self.last_image_center_x = float("nan")
            self.last_image_center_y = float("nan")

        self.last_quad_center_x = float("nan")
        self.last_quad_center_y = float("nan")
        self.last_quad_center_offset_x = float("nan")
        self.last_quad_center_offset_y = float("nan")
        self.last_quad_width_px = float("nan")
        self.last_quad_height_px = float("nan")
        self.last_quad_aspect_ratio = float("nan")
        self.last_quad_area_px = float("nan")

        if corners is None:
            return

        pts = np.asarray(corners, dtype=float).reshape(-1, 2)
        if pts.shape[0] != 4 or not np.all(np.isfinite(pts)):
            return

        center = np.mean(pts, axis=0)
        self.last_quad_center_x = float(center[0])
        self.last_quad_center_y = float(center[1])
        self.last_quad_center_offset_x = self.last_quad_center_x - self.last_image_center_x
        self.last_quad_center_offset_y = self.last_quad_center_y - self.last_image_center_y

        top_width = float(np.linalg.norm(pts[1] - pts[0]))
        bottom_width = float(np.linalg.norm(pts[2] - pts[3]))
        right_height = float(np.linalg.norm(pts[2] - pts[1]))
        left_height = float(np.linalg.norm(pts[3] - pts[0]))
        self.last_quad_width_px = 0.5 * (top_width + bottom_width)
        self.last_quad_height_px = 0.5 * (right_height + left_height)
        if self.last_quad_height_px > 1e-6:
            self.last_quad_aspect_ratio = self.last_quad_width_px / self.last_quad_height_px
        self.last_quad_area_px = float(abs(
            0.5 * (
                np.dot(pts[:, 0], np.roll(pts[:, 1], -1))
                - np.dot(pts[:, 1], np.roll(pts[:, 0], -1))
            )
        ))

    def reset_yaw_image_consistency_debug(self):
        self.bearing_to_gate_deg = float("nan")
        self.telemetry_yaw_deg_for_image = float("nan")
        self.yaw_error_deg = float("nan")
        self.predicted_quad_offset_from_yaw_px = float("nan")
        self.yaw_pixel_error_px = float("nan")
        self.yaw_image_consistency_status = ""

    def compute_yaw_image_consistency_debug(
        self,
        drone_pos,
        telemetry_yaw_rad,
        camera_matrix,
    ):
        if len(self.gt_gates) == 0:
            return
        try:
            gate_idx = int(np.clip(self.current_gate_idx, 0, len(self.gt_gates) - 1))
            gate = np.asarray(self.gt_gates[gate_idx], dtype=float).reshape(3)
            pos = np.asarray(drone_pos, dtype=float).reshape(3)
            fx = float(np.asarray(camera_matrix, dtype=float).reshape(3, 3)[0, 0])
            actual_offset = float(self.last_quad_center_offset_x)
        except Exception:
            return
        if not (
            np.all(np.isfinite(gate[:2]))
            and np.all(np.isfinite(pos[:2]))
            and np.isfinite(telemetry_yaw_rad)
            and np.isfinite(fx)
            and np.isfinite(actual_offset)
        ):
            return

        bearing_rad = math.atan2(float(gate[1] - pos[1]), float(gate[0] - pos[0]))
        yaw_error_rad = math.atan2(
            math.sin(float(telemetry_yaw_rad) - bearing_rad),
            math.cos(float(telemetry_yaw_rad) - bearing_rad),
        )
        predicted_offset = fx * math.tan(-yaw_error_rad)

        self.bearing_to_gate_deg = math.degrees(bearing_rad)
        self.telemetry_yaw_deg_for_image = math.degrees(float(telemetry_yaw_rad))
        self.yaw_error_deg = math.degrees(yaw_error_rad)
        self.predicted_quad_offset_from_yaw_px = predicted_offset
        self.yaw_pixel_error_px = predicted_offset - actual_offset
        self.yaw_image_consistency_status = (
            "YAW_IMAGE_INCONSISTENT"
            if abs(self.yaw_pixel_error_px) > 30.0
            else "consistent"
        )
        if self.yaw_image_consistency_status == "YAW_IMAGE_INCONSISTENT":
            print(
                "[YAW_IMAGE_INCONSISTENT] "
                f"gate_idx={gate_idx} "
                f"image_stamp={self.image_stamp_sec}.{self.image_stamp_nanosec:09d} "
                f"image_age_s={self.image_age_s:.3f} "
                f"attitude_age_s={self.attitude_age_s:.3f} "
                f"position_age_s={self.position_age_s:.3f} "
                f"pose_vs_image_s={self.pose_age_relative_to_image_s:.3f} "
                f"yaw_deg={self.telemetry_yaw_deg_for_image:.2f} "
                f"bearing_deg={self.bearing_to_gate_deg:.2f} "
                f"yaw_error_deg={self.yaw_error_deg:.2f} "
                f"predicted_px={predicted_offset:.1f} "
                f"actual_px={actual_offset:.1f} "
                f"error_px={self.yaw_pixel_error_px:.1f}"
            )

    @staticmethod
    def _wrap_degrees(angle_deg):
        return (float(angle_deg) + 180.0) % 360.0 - 180.0

    @staticmethod
    def _quaternion_xyzw_to_rotmat(quat):
        x, y, z, w = np.asarray(quat, dtype=float).reshape(4)
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1e-12:
            return np.full((3, 3), np.nan, dtype=float)
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        return np.array([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ], dtype=float)

    @classmethod
    def _quaternion_xyzw_yaw_deg(cls, quat):
        rot = cls._quaternion_xyzw_to_rotmat(quat)
        if not np.all(np.isfinite(rot)):
            return float("nan")
        return math.degrees(math.atan2(float(rot[1, 0]), float(rot[0, 0])))

    def reset_gazebo_pose_comparison_debug(self):
        self.gazebo_model_pos_world = np.full(3, np.nan, dtype=float)
        self.gazebo_model_quat_world = np.full(4, np.nan, dtype=float)
        self.gazebo_camera_pos_world = np.full(3, np.nan, dtype=float)
        self.gazebo_camera_quat_world = np.full(4, np.nan, dtype=float)
        self.gazebo_pose_wall_time = float("nan")
        self.gazebo_pose_age_s = float("nan")
        self.gazebo_model_yaw_deg = float("nan")
        self.gazebo_camera_yaw_deg = float("nan")
        self.mavsdk_minus_gazebo_pos = np.full(3, np.nan, dtype=float)
        self.mavsdk_minus_gazebo_yaw_deg = float("nan")
        self.gate_world_mavsdk = np.full(3, np.nan, dtype=float)
        self.gate_world_gazebo = np.full(3, np.nan, dtype=float)
        self.gate_world_mavsdk_error_to_gt = float("nan")
        self.gate_world_gazebo_error_to_gt = float("nan")
        self.required_yaw_deg_from_pnp_to_gt = float("nan")
        self.mavsdk_yaw_minus_required_deg = float("nan")
        self.gazebo_yaw_minus_required_deg = float("nan")

    def capture_gazebo_pose_debug(
        self,
        gazebo_pose,
        mavsdk_pos,
        mavsdk_yaw_rad,
        process_wall_time,
    ):
        if not isinstance(gazebo_pose, dict):
            return
        try:
            self.gazebo_model_pos_world = np.asarray(
                gazebo_pose["gazebo_model_pos_world"], dtype=float
            ).reshape(3)
            self.gazebo_model_quat_world = np.asarray(
                gazebo_pose["gazebo_model_quat_world"], dtype=float
            ).reshape(4)
            self.gazebo_camera_pos_world = np.asarray(
                gazebo_pose["gazebo_camera_pos_world"], dtype=float
            ).reshape(3)
            self.gazebo_camera_quat_world = np.asarray(
                gazebo_pose["gazebo_camera_quat_world"], dtype=float
            ).reshape(4)
            self.gazebo_pose_wall_time = float(gazebo_pose["gazebo_pose_wall_time"])
        except (KeyError, TypeError, ValueError):
            self.reset_gazebo_pose_comparison_debug()
            return

        self.gazebo_pose_age_s = float(process_wall_time) - self.gazebo_pose_wall_time
        self.gazebo_model_yaw_deg = self._quaternion_xyzw_yaw_deg(
            self.gazebo_model_quat_world
        )
        self.gazebo_camera_yaw_deg = self._quaternion_xyzw_yaw_deg(
            self.gazebo_camera_quat_world
        )
        gazebo_model_pos_planner = np.array([
            self.gazebo_model_pos_world[1],
            self.gazebo_model_pos_world[0],
            self.gazebo_model_pos_world[2],
        ], dtype=float)
        self.mavsdk_minus_gazebo_pos = (
            np.asarray(mavsdk_pos, dtype=float).reshape(3)
            - gazebo_model_pos_planner
        )
        self.mavsdk_minus_gazebo_yaw_deg = self._wrap_degrees(
            math.degrees(float(mavsdk_yaw_rad)) - self.gazebo_model_yaw_deg
        )

    def compute_gazebo_pose_gate_comparison_debug(
        self,
        pnp_camera,
        mavsdk_pos,
        mavsdk_rpy_raw,
        perception_rpy,
    ):
        if pnp_camera is None or len(self.gt_gates) == 0:
            return
        pnp_camera = np.asarray(pnp_camera, dtype=float).reshape(3)
        mavsdk_pos = np.asarray(mavsdk_pos, dtype=float).reshape(3)
        mavsdk_rpy_raw = np.asarray(mavsdk_rpy_raw, dtype=float).reshape(3)
        perception_rpy = np.asarray(perception_rpy, dtype=float).reshape(3)
        if not np.all(np.isfinite(pnp_camera)):
            return

        gate_idx = int(np.clip(self.current_gate_idx, 0, len(self.gt_gates) - 1))
        gt_gate = np.asarray(self.gt_gates[gate_idx], dtype=float).reshape(3)
        r_body_camera = self.body_camera_matrix_for_mode(self.perception_transform_mode)
        body_vec = self.camera_offset_body + r_body_camera @ pnp_camera
        r_wb_uncorrected = self.perception_node._rpy_to_rotmat(
            float(mavsdk_rpy_raw[0]),
            float(mavsdk_rpy_raw[1]),
            float(mavsdk_rpy_raw[2]),
        )
        r_wb_corrected = self.perception_node._rpy_to_rotmat(
            float(perception_rpy[0]),
            float(perception_rpy[1]),
            float(perception_rpy[2]),
        )
        self.gate_world_uncorrected = mavsdk_pos + r_wb_uncorrected @ body_vec
        self.gate_world_corrected = mavsdk_pos + r_wb_corrected @ body_vec
        self.gate_world_mavsdk = self.gate_world_uncorrected.copy()
        self.gate_world_mavsdk_error_to_gt = float(
            np.linalg.norm(self.gate_world_mavsdk - gt_gate)
        )

        r_wb_gazebo = self._quaternion_xyzw_to_rotmat(self.gazebo_model_quat_world)
        if (
            np.all(np.isfinite(self.gazebo_model_pos_world))
            and np.all(np.isfinite(r_wb_gazebo))
        ):
            self.gate_world_gazebo = (
                self.gazebo_model_pos_world + r_wb_gazebo @ body_vec
            )
            self.gate_world_gazebo_error_to_gt = float(
                np.linalg.norm(self.gate_world_gazebo - gt_gate)
            )

        # For Rz(yaw) @ Ry(pitch) @ Rx(roll), solve the yaw that aligns
        # the pitch/roll-adjusted body vector with the GT horizontal bearing.
        r_no_yaw = self.perception_node._rpy_to_rotmat(
            float(mavsdk_rpy_raw[0]),
            float(mavsdk_rpy_raw[1]),
            0.0,
        )
        leveled_body_vec = r_no_yaw @ body_vec
        desired_delta = gt_gate - mavsdk_pos
        if (
            np.linalg.norm(leveled_body_vec[:2]) > 1e-9
            and np.linalg.norm(desired_delta[:2]) > 1e-9
        ):
            required_yaw = math.atan2(
                float(desired_delta[1]), float(desired_delta[0])
            ) - math.atan2(
                float(leveled_body_vec[1]), float(leveled_body_vec[0])
            )
            self.required_yaw_deg_from_pnp_to_gt = self._wrap_degrees(
                math.degrees(required_yaw)
            )
            self.mavsdk_yaw_minus_required_deg = self._wrap_degrees(
                math.degrees(float(mavsdk_rpy_raw[2]))
                - self.required_yaw_deg_from_pnp_to_gt
            )
            if np.isfinite(self.gazebo_model_yaw_deg):
                self.gazebo_yaw_minus_required_deg = self._wrap_degrees(
                    self.gazebo_model_yaw_deg
                    - self.required_yaw_deg_from_pnp_to_gt
                )

    @staticmethod
    def pnp_candidate_error(candidates, index):
        if candidates is None or len(candidates) <= index:
            return float("nan")
        try:
            return float(candidates[index].get("error", np.nan))
        except Exception:
            return float("nan")

    def reset_transform_validation_debug(self):
        nan3 = np.full(3, np.nan, dtype=float)
        self.expected_gate_cam = nan3.copy()
        self.expected_gate_cam_live_axis = nan3.copy()
        self.expected_gate_cam_old_axis = nan3.copy()
        self.expected_camera_axis_mode = ""
        self.expected_uses_live_axis_convention = False
        self.expected_gate_projected_center = np.full(2, np.nan, dtype=float)
        self.expected_vs_quad_center_error_px = float("nan")
        self.pnp_camera = nan3.copy()
        self.pnp_gate_cam = nan3.copy()
        self.camera_error = nan3.copy()
        self.camera_error_norm = float("nan")
        self.size_depth = float("nan")
        self.size_depth_from_width = float("nan")
        self.size_depth_from_height = float("nan")
        self.pnp_depth_minus_size_depth = float("nan")
        self.expected_gate_body = nan3.copy()
        self.pnp_gate_body = nan3.copy()
        self.body_error = nan3.copy()
        self.expected_gate_world = nan3.copy()
        self.pnp_gate_world = nan3.copy()
        self.world_error = nan3.copy()
        self.world_error_norm = float("nan")
        self.pnp_size_190_cam = nan3.copy()
        self.pnp_size_200_cam = nan3.copy()
        self.pnp_size_210_cam = nan3.copy()
        self.pnp_size_190_world = nan3.copy()
        self.pnp_size_200_world = nan3.copy()
        self.pnp_size_210_world = nan3.copy()
        self.pnp_size_190_reproj_error = float("nan")
        self.pnp_size_200_reproj_error = float("nan")
        self.pnp_size_210_reproj_error = float("nan")
        self.pnp_size_190_gt_error = float("nan")
        self.pnp_size_200_gt_error = float("nan")
        self.pnp_size_210_gt_error = float("nan")
        self.pnp_solver_used = "SOLVEPNP_ITERATIVE"
        self.live_solver_name = ""
        self.live_solver_world = nan3.copy()
        self.live_solver_reproj_error = float("nan")
        self.ippe_world_error_gt = float("nan")
        self.iterative_world_error_gt = float("nan")
        self.pnp_fallback_reason = ""
        self.pnp_best_debug_solver = ""
        self.pnp_best_debug_order = ""
        self.pnp_current_world_error_gt = float("nan")
        self.pnp_best_world_error_gt = float("nan")
        self.pnp_current_cam = nan3.copy()
        self.pnp_best_cam = nan3.copy()
        self.pnp_current_world = nan3.copy()
        self.pnp_best_world = nan3.copy()
        self.pnp_current_reproj_error = float("nan")
        self.pnp_best_reproj_error = float("nan")
        self.pnp_candidate0_world = nan3.copy()
        self.pnp_candidate1_world = nan3.copy()
        self.pnp_candidate0_error = float("nan")
        self.pnp_candidate1_error = float("nan")
        self.pnp_candidate0_projected_corners = None
        self.pnp_candidate1_projected_corners = None
        self.pnp_gt_projected_center = np.full(2, np.nan, dtype=float)
        self.pnp_gt_projected_quad_center_error_px = float("nan")
        self.transform_sweep_best_mode = ""
        self.transform_sweep_best_error = float("nan")
        self.transform_sweep_legacy_error = float("nan")
        self.transform_sweep_direct_rad_error = float("nan")
        self.transform_sweep_pi_over_2_minus_yaw_error = float("nan")
        self.transform_sweep_yaw_minus_pi_over_2_error = float("nan")
        self.transform_sweep_neg_yaw_error = float("nan")
        self.transform_sweep_neg_yaw_plus_pi_over_2_error = float("nan")
        self.transform_sweep_legacy_world = nan3.copy()
        self.transform_sweep_direct_rad_world = nan3.copy()
        self.transform_sweep_yaw_minus_pi_over_2_world = nan3.copy()
        self.transform_sweep_pi_over_2_minus_yaw_world = nan3.copy()
        self.transform_sweep_physical_direct_rad_world = nan3.copy()
        self.transform_sweep_best_world = nan3.copy()
        self.live_vs_physical_direct_delta_m = float("nan")
        self.camera_axis_sweep_flu_x_flipped_world = nan3.copy()
        self.live_camera_axis_mode = ""
        self.live_camera_axis_det = float("nan")
        self.live_uses_x_mirror = False
        self.live_vs_camera_axis_x_flipped_delta_m = float("nan")
        self.camera_axis_sweep_best_mode = ""
        self.camera_axis_sweep_best_error = float("nan")
        self.camera_axis_sweep_best_world = nan3.copy()
        self.camera_axis_sweep_flu_error = float("nan")
        self.camera_axis_sweep_flu_x_flipped_error = float("nan")
        self.camera_axis_sweep_frd_error = float("nan")
        self.camera_axis_sweep_old_default_error = float("nan")
        self.sign_match_pnp_vs_image = False
        self.sign_match_expected_vs_image = False
        self.debug_expected_gate_idx = -1
        self.live_minus_expected = nan3.copy()
        self.live_lateral_error_m = float("nan")
        self.filtered_minus_expected = nan3.copy()
        self.selected_order_vs_axis_mode = ""

    def compute_size_depth_debug(self):
        self.size_depth = float("nan")
        self.size_depth_from_width = float("nan")
        self.size_depth_from_height = float("nan")
        self.pnp_depth_minus_size_depth = float("nan")

        corners = self.last_ordered_image_corners
        if corners is None or self.last_camera_matrix is None:
            return
        try:
            pts = np.asarray(corners, dtype=float).reshape(4, 2)
            k = np.asarray(self.last_camera_matrix, dtype=float).reshape(3, 3)
        except Exception:
            return
        if not np.all(np.isfinite(pts)) or not np.all(np.isfinite(k)):
            return

        fx = float(k[0, 0])
        fy = float(k[1, 1])
        width_px = 0.5 * (
            float(np.linalg.norm(pts[1] - pts[0]))
            + float(np.linalg.norm(pts[2] - pts[3]))
        )
        height_px = 0.5 * (
            float(np.linalg.norm(pts[3] - pts[0]))
            + float(np.linalg.norm(pts[2] - pts[1]))
        )

        gate_size = float(getattr(self.gate_perception, "gate_size", 1.5))
        depths = []
        if width_px > 1.0 and np.isfinite(fx):
            self.size_depth_from_width = fx * gate_size / width_px
            depths.append(self.size_depth_from_width)
        if height_px > 1.0 and np.isfinite(fy):
            self.size_depth_from_height = fy * gate_size / height_px
            depths.append(self.size_depth_from_height)
        if len(depths) > 0:
            self.size_depth = float(np.mean(depths))

        pnp_camera = self._vec3_for_debug(self.last_gate_center_camera)
        if np.isfinite(self.size_depth) and np.all(np.isfinite(pnp_camera)):
            self.pnp_depth_minus_size_depth = float(pnp_camera[2] - self.size_depth)

    def reset_pnp_selection_debug(self):
        self.pnp_selected_order = ""
        self.pnp_selected_solver = ""
        self.pnp_selected_score = float("nan")
        self.pnp_selected_reprojection_error = float("nan")
        self.pnp_selected_gate_center_camera = np.full(3, np.nan, dtype=float)
        self.pnp_selected_reason = ""
        self.pnp_candidate_summary = ""
        self.pnp_candidate_world_summary = ""
        self.pnp_selected_world_score = float("nan")
        self.pnp_selected_world_reason = ""
        self.allow_pnp_corner_reordering = False
        self.pnp_live_candidate_orders_allowed = ""
        self.pnp_debug_best_order = ""
        self.pnp_live_vs_debug_best_order_mismatch = False
        self.pnp_lateral_angle = float("nan")
        self.image_center_offset_normalized = float("nan")
        self.keypoint_polygon_signed_area = float("nan")
        self.keypoint_polygon_winding = ""
        self.keypoint_edge_top = float("nan")
        self.keypoint_edge_right = float("nan")
        self.keypoint_edge_bottom = float("nan")
        self.keypoint_edge_left = float("nan")
        self.keypoint_bbox_center = np.full(2, np.nan, dtype=float)
        self.keypoint_polygon_center = np.full(2, np.nan, dtype=float)
        self.keypoint_bbox_polygon_delta = np.full(2, np.nan, dtype=float)
        self.raw_keypoint_polygon_signed_area = float("nan")
        self.raw_keypoint_polygon_winding = ""

    def compute_transform_sweep_debug(self, drone_pos, expected_gate_world, gate_idx=None):
        if self.last_telemetry_rpy_raw_rad is None:
            return
        if self.last_gate_center_camera is None:
            return

        gate_camera = self._vec3_for_debug(self.last_gate_center_camera)
        if not np.all(np.isfinite(gate_camera)):
            return

        drone_pos = np.asarray(drone_pos, dtype=float).reshape(3)
        expected_gate_world = np.asarray(expected_gate_world, dtype=float).reshape(3)
        telemetry_rpy_raw_rad = np.asarray(self.last_telemetry_rpy_raw_rad, dtype=float).reshape(3)

        matrices = (
            ("current", np.asarray(self.perception_node.R_body_camera, dtype=float).reshape(3, 3)),
            ("physical", self.physical_body_camera_matrix()),
        )

        best_error = float("inf")
        best_mode = ""
        best_world = np.full(3, np.nan, dtype=float)
        current_matrix_errors = {}
        current_matrix_worlds = {}
        physical_matrix_worlds = {}
        print("[TRANSFORM SWEEP] comparing camera-to-world conventions")
        for matrix_name, r_body_camera in matrices:
            for mode in self.perception_transform_modes:
                world, _, rpy = self.transform_gate_camera_to_world(
                    gate_camera=gate_camera,
                    drone_pos=drone_pos,
                    telemetry_rpy_raw_rad=telemetry_rpy_raw_rad,
                    mode=mode,
                    r_body_camera=r_body_camera,
                )
                yaw = rpy[2]
                error = float(np.linalg.norm(world - expected_gate_world))
                yaw_deg = math.degrees(float(yaw))
                print(
                    "[TRANSFORM SWEEP] "
                    f"matrix={matrix_name} mode={mode} "
                    f"world={world.tolist()} error={error:.3f} yaw_used_deg={yaw_deg:.2f}"
                )
                if matrix_name == "current":
                    current_matrix_errors[mode] = error
                    current_matrix_worlds[mode] = world.copy()
                if matrix_name == "physical":
                    physical_matrix_worlds[mode] = world.copy()
                if error < best_error:
                    best_error = error
                    best_mode = f"{matrix_name}:{mode}"
                    best_world = world.copy()

        self.transform_sweep_best_mode = best_mode
        self.transform_sweep_best_error = best_error if np.isfinite(best_error) else float("nan")
        self.transform_sweep_best_world = best_world.copy()
        self.transform_sweep_legacy_error = current_matrix_errors.get("legacy_scaled_yaw", float("nan"))
        self.transform_sweep_direct_rad_error = current_matrix_errors.get("direct_rad", float("nan"))
        self.transform_sweep_pi_over_2_minus_yaw_error = current_matrix_errors.get("pi_over_2_minus_yaw", float("nan"))
        self.transform_sweep_yaw_minus_pi_over_2_error = current_matrix_errors.get("yaw_minus_pi_over_2", float("nan"))
        self.transform_sweep_neg_yaw_error = current_matrix_errors.get("neg_yaw", float("nan"))
        self.transform_sweep_neg_yaw_plus_pi_over_2_error = current_matrix_errors.get("neg_yaw_plus_pi_over_2", float("nan"))
        self.transform_sweep_legacy_world = current_matrix_worlds.get("legacy_scaled_yaw", np.full(3, np.nan)).copy()
        self.transform_sweep_direct_rad_world = current_matrix_worlds.get("direct_rad", np.full(3, np.nan)).copy()
        self.transform_sweep_yaw_minus_pi_over_2_world = current_matrix_worlds.get("yaw_minus_pi_over_2", np.full(3, np.nan)).copy()
        self.transform_sweep_pi_over_2_minus_yaw_world = current_matrix_worlds.get("pi_over_2_minus_yaw", np.full(3, np.nan)).copy()
        self.transform_sweep_physical_direct_rad_world = physical_matrix_worlds.get("direct_rad", np.full(3, np.nan)).copy()
        if np.all(np.isfinite(self.pnp_gate_world)) and np.all(np.isfinite(self.transform_sweep_physical_direct_rad_world)):
            self.live_vs_physical_direct_delta_m = float(
                np.linalg.norm(self.pnp_gate_world - self.transform_sweep_physical_direct_rad_world)
            )
            if self.perception_transform_mode == "physical_direct_rad":
                print(
                    "[PERCEPTION TRANSFORM CHECK] "
                    f"live_vs_physical_direct_delta_m={self.live_vs_physical_direct_delta_m:.9f}"
                )
        else:
            self.live_vs_physical_direct_delta_m = float("nan")

        if gate_idx is not None:
            gate_stats = self.transform_sweep_error_stats.setdefault(int(gate_idx), {})
            for mode, error in current_matrix_errors.items():
                stats = gate_stats.setdefault(f"current:{mode}", [0.0, 0])
                stats[0] += float(error)
                stats[1] += 1
            for mode, world in physical_matrix_worlds.items():
                error = float(np.linalg.norm(world - expected_gate_world))
                stats = gate_stats.setdefault(f"physical:{mode}", [0.0, 0])
                stats[0] += error
                stats[1] += 1
            avg_best_mode = ""
            avg_best_error = float("inf")
            for mode, (error_sum, count) in gate_stats.items():
                if count <= 0:
                    continue
                avg_error = error_sum / count
                if avg_error < avg_best_error:
                    avg_best_error = avg_error
                    avg_best_mode = mode
            if avg_best_mode:
                print(
                    "[TRANSFORM SWEEP SUMMARY] "
                    f"gate_idx={gate_idx} avg_best={avg_best_mode} "
                    f"avg_error={avg_best_error:.3f} samples={gate_stats[avg_best_mode][1]}"
                )

        print(
            "[TRANSFORM SUMMARY] "
            f"selected_live_mode={self.perception_transform_mode} "
            f"legacy_world={self.transform_sweep_legacy_world.tolist()} "
            f"yaw_minus_pi_over_2_world={self.transform_sweep_yaw_minus_pi_over_2_world.tolist()} "
            f"physical_direct_rad_world={self.transform_sweep_physical_direct_rad_world.tolist()} "
            f"selected_world={self.pnp_gate_world.tolist()} "
            f"gt_error={self.world_error_norm:.3f} "
            f"best={self.transform_sweep_best_mode}:{self.transform_sweep_best_error:.3f}"
        )

    def camera_axis_sweep_matrices(self):
        return (
            (
                "opencv_to_body_flu_current",
                np.array([
                    [0.0, 0.0, 1.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                ], dtype=float),
            ),
            (
                "opencv_to_body_flu_x_flipped",
                np.array([
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                ], dtype=float),
            ),
            (
                "opencv_to_body_frd",
                np.array([
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ], dtype=float),
            ),
            (
                "old_live_default",
                np.array([
                    [-1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, -1.0, 0.0],
                ], dtype=float),
            ),
        )

    @staticmethod
    def sign_matches(a, b):
        if not np.isfinite(a) or not np.isfinite(b):
            return False
        if abs(a) < 1e-9 or abs(b) < 1e-9:
            return False
        return bool(np.sign(a) == np.sign(b))

    def compute_camera_axis_sweep_debug(self, drone_pos, expected_gate_world):
        gate_camera = self._vec3_for_debug(self.last_gate_center_camera)
        if not np.all(np.isfinite(gate_camera)):
            return
        if self.last_telemetry_rpy_raw_rad is None:
            return

        drone_pos = np.asarray(drone_pos, dtype=float).reshape(3)
        expected_gate_world = np.asarray(expected_gate_world, dtype=float).reshape(3)
        telemetry_rpy_raw_rad = np.asarray(self.last_telemetry_rpy_raw_rad, dtype=float).reshape(3)

        errors = {}
        worlds = {}
        best_mode = ""
        best_error = float("inf")
        best_world = np.full(3, np.nan, dtype=float)
        print("[CAMERA AXIS SWEEP] direct telemetry RPY camera matrix comparison")
        for mode_name, r_body_camera in self.camera_axis_sweep_matrices():
            world, _, _ = self.transform_gate_camera_to_world(
                gate_camera=gate_camera,
                drone_pos=drone_pos,
                telemetry_rpy_raw_rad=telemetry_rpy_raw_rad,
                mode="direct_rad",
                r_body_camera=r_body_camera,
            )
            error = float(np.linalg.norm(world - expected_gate_world))
            det = float(np.linalg.det(r_body_camera))
            errors[mode_name] = error
            worlds[mode_name] = world.copy()
            print(
                "[CAMERA AXIS SWEEP] "
                f"mode={mode_name} det={det:.3f} "
                f"world={world.tolist()} gt_error={error:.3f}"
            )
            if error < best_error:
                best_error = error
                best_mode = mode_name
                best_world = world.copy()

        self.camera_axis_sweep_best_mode = best_mode
        self.camera_axis_sweep_best_error = best_error if np.isfinite(best_error) else float("nan")
        self.camera_axis_sweep_best_world = best_world.copy()
        self.camera_axis_sweep_flu_x_flipped_world = worlds.get(
            "opencv_to_body_flu_x_flipped",
            np.full(3, np.nan, dtype=float),
        ).copy()
        self.camera_axis_sweep_flu_error = errors.get("opencv_to_body_flu_current", float("nan"))
        self.camera_axis_sweep_flu_x_flipped_error = errors.get("opencv_to_body_flu_x_flipped", float("nan"))
        self.camera_axis_sweep_frd_error = errors.get("opencv_to_body_frd", float("nan"))
        self.camera_axis_sweep_old_default_error = errors.get("old_live_default", float("nan"))

        live_matrix = self.body_camera_matrix_for_mode(self.perception_transform_mode)
        self.live_camera_axis_mode = self.perception_transform_mode
        self.live_camera_axis_det = float(np.linalg.det(live_matrix))
        self.live_uses_x_mirror = self.perception_transform_mode == "physical_direct_rad_x_mirror"
        if self.live_uses_x_mirror and np.all(np.isfinite(self.pnp_gate_world)) and np.all(
            np.isfinite(self.camera_axis_sweep_flu_x_flipped_world)
        ):
            self.live_vs_camera_axis_x_flipped_delta_m = float(
                np.linalg.norm(self.pnp_gate_world - self.camera_axis_sweep_flu_x_flipped_world)
            )
            print(
                "[CAMERA AXIS CHECK] "
                f"live_vs_camera_axis_x_flipped_delta_m="
                f"{self.live_vs_camera_axis_x_flipped_delta_m:.9f}"
            )
            if self.live_vs_camera_axis_x_flipped_delta_m > 1e-6:
                print(
                    "[CAMERA AXIS WARN] "
                    "physical_direct_rad_x_mirror live world does not exactly match "
                    "opencv_to_body_flu_x_flipped sweep"
                )
        else:
            self.live_vs_camera_axis_x_flipped_delta_m = float("nan")

        pnp_camera_x = float(self.pnp_gate_cam[0]) if np.all(np.isfinite(self.pnp_gate_cam)) else float("nan")
        expected_camera_x = (
            float(self.expected_gate_cam[0])
            if np.all(np.isfinite(self.expected_gate_cam))
            else float("nan")
        )
        image_offset_x = float(self.last_quad_center_offset_x)
        self.sign_match_pnp_vs_image = self.sign_matches(pnp_camera_x, image_offset_x)
        self.sign_match_expected_vs_image = self.sign_matches(expected_camera_x, image_offset_x)
        print(
            "[CAMERA AXIS SIGN] "
            f"quad_center_offset_x={image_offset_x:.2f} "
            f"pnp_camera_x={pnp_camera_x:.3f} "
            f"expected_gate_cam_x={expected_camera_x:.3f} "
            f"pnp_vs_image={self.sign_match_pnp_vs_image} "
            f"expected_vs_image={self.sign_match_expected_vs_image}"
        )

    def update_per_gate_debug_summary(self, gate_idx):
        if gate_idx is None or int(gate_idx) < 0:
            return

        gate_idx = int(gate_idx)
        raw_world = self._vec3_for_debug(self.last_gate_center_world_debug)
        filtered = self._vec3_for_debug(self.track_filtered_center)
        expected = self._vec3_for_debug(self.expected_gate_world)
        live_error = raw_world - expected
        filtered_error = filtered - expected

        self.debug_expected_gate_idx = gate_idx
        self.live_minus_expected = live_error.copy()
        self.live_lateral_error_m = (
            float(np.linalg.norm(live_error[:2]))
            if np.all(np.isfinite(live_error[:2]))
            else float("nan")
        )
        self.filtered_minus_expected = filtered_error.copy()
        self.selected_order_vs_axis_mode = (
            f"{self.pnp_selected_order or 'none'}|{self.camera_axis_sweep_best_mode or 'none'}"
        )

        stats = self.per_gate_debug_summary.setdefault(
            gate_idx,
            {
                "count": 0,
                "raw_sum": np.zeros(3, dtype=float),
                "raw_count": 0,
                "filtered_sum": np.zeros(3, dtype=float),
                "filtered_count": 0,
                "error_sum": 0.0,
                "error_count": 0,
                "xyz_error_sum": np.zeros(3, dtype=float),
                "xyz_error_count": 0,
                "axis_mode_counts": {},
                "pnp_order_counts": {},
                "solver_counts": {},
            },
        )
        stats["count"] += 1

        if np.all(np.isfinite(raw_world)):
            stats["raw_sum"] += raw_world
            stats["raw_count"] += 1
        if np.all(np.isfinite(filtered)):
            stats["filtered_sum"] += filtered
            stats["filtered_count"] += 1
        if np.all(np.isfinite(live_error)):
            stats["error_sum"] += float(np.linalg.norm(live_error))
            stats["error_count"] += 1
            stats["xyz_error_sum"] += live_error
            stats["xyz_error_count"] += 1

        axis_mode = self.camera_axis_sweep_best_mode or "none"
        order = self.pnp_selected_order or "none"
        solver = self.pnp_selected_solver or self.live_solver_name or "none"
        stats["axis_mode_counts"][axis_mode] = stats["axis_mode_counts"].get(axis_mode, 0) + 1
        stats["pnp_order_counts"][order] = stats["pnp_order_counts"].get(order, 0) + 1
        stats["solver_counts"][solver] = stats["solver_counts"].get(solver, 0) + 1

        raw_mean = (
            stats["raw_sum"] / stats["raw_count"]
            if stats["raw_count"] > 0
            else np.full(3, np.nan, dtype=float)
        )
        filtered_mean = (
            stats["filtered_sum"] / stats["filtered_count"]
            if stats["filtered_count"] > 0
            else np.full(3, np.nan, dtype=float)
        )
        mean_error = (
            stats["error_sum"] / stats["error_count"]
            if stats["error_count"] > 0
            else float("nan")
        )
        mean_xyz_error = (
            stats["xyz_error_sum"] / stats["xyz_error_count"]
            if stats["xyz_error_count"] > 0
            else np.full(3, np.nan, dtype=float)
        )
        print(
            "[PER-GATE DEBUG SUMMARY] "
            f"gate_idx={gate_idx} count={stats['count']} "
            f"mean_raw=({raw_mean[0]:.2f},{raw_mean[1]:.2f},{raw_mean[2]:.2f}) "
            f"mean_filtered=({filtered_mean[0]:.2f},{filtered_mean[1]:.2f},{filtered_mean[2]:.2f}) "
            f"mean_world_error={mean_error:.2f} "
            f"mean_xyz_error=({mean_xyz_error[0]:.2f},{mean_xyz_error[1]:.2f},{mean_xyz_error[2]:.2f}) "
            f"axis_counts={stats['axis_mode_counts']} "
            f"order_counts={stats['pnp_order_counts']} "
            f"solver_counts={stats['solver_counts']}"
        )

    def compute_transform_validation_debug(self, drone_pos, drone_rpy_rad):
        """
        Debug-only comparison against the known simulator GT gate. This does
        not feed back into control, target selection, or perception memory.
        """
        self.reset_transform_validation_debug()

        if self.perception_node is None or len(self.gt_gates) == 0:
            return

        gate_idx = int(np.clip(self.current_gate_idx, 0, len(self.gt_gates) - 1))
        gt_gate_world = np.asarray(self.gt_gates[gate_idx], dtype=float).reshape(3)
        drone_pos = np.asarray(drone_pos, dtype=float).reshape(3)
        roll, pitch, yaw = np.asarray(drone_rpy_rad, dtype=float).reshape(3)

        R_wb = self.perception_node._rpy_to_rotmat(float(roll), float(pitch), float(yaw))
        R_bw = R_wb.T
        R_body_camera_old = np.asarray(self.perception_node.R_body_camera, dtype=float).reshape(3, 3)
        R_body_camera_live = self.body_camera_matrix_for_mode(
            self.perception_transform_mode,
            default_matrix=R_body_camera_old,
        )
        R_camera_body_old = R_body_camera_old.T
        R_camera_body_live = R_body_camera_live.T

        expected_body_from_camera = (
            R_bw @ (gt_gate_world - drone_pos)
        ) - self.camera_offset_body
        expected_camera_old = R_camera_body_old @ expected_body_from_camera
        expected_camera_live = R_camera_body_live @ expected_body_from_camera

        self.expected_gate_cam_old_axis = expected_camera_old.copy()
        self.expected_gate_cam_live_axis = expected_camera_live.copy()
        self.expected_gate_cam = expected_camera_live.copy()
        self.expected_camera_axis_mode = self.perception_transform_mode
        self.expected_uses_live_axis_convention = True
        self.expected_gate_body = expected_body_from_camera.copy()
        self.expected_gate_world = gt_gate_world.copy()

        self.pnp_gate_cam = self._vec3_for_debug(self.last_gate_center_camera)
        self.pnp_camera = self.pnp_gate_cam.copy()
        self.pnp_gate_body = self._vec3_for_debug(self.last_gate_center_body)
        self.pnp_gate_world = self._vec3_for_debug(self.last_gate_center_world_debug)

        self.camera_error = self.pnp_gate_cam - self.expected_gate_cam
        self.body_error = self.pnp_gate_body - self.expected_gate_body
        self.world_error = self.pnp_gate_world - self.expected_gate_world
        self.camera_error_norm = float(np.linalg.norm(self.camera_error))
        self.world_error_norm = float(np.linalg.norm(self.world_error))
        self.compute_size_depth_debug()
        self.update_gate_size_sweep_debug(self.expected_gate_world)
        self.update_pnp_formulation_debug(self.expected_gate_world, self.expected_gate_cam)
        self.update_gt_projected_center_debug(self.expected_gate_cam)
        self.compute_transform_sweep_debug(drone_pos, self.expected_gate_world, gate_idx=gate_idx)
        self.compute_camera_axis_sweep_debug(drone_pos, self.expected_gate_world)
        self.update_per_gate_debug_summary(gate_idx)

        print(
            "[TRANSFORM VALIDATION] "
            f"gt_gate_idx={gate_idx} "
            f"expected_axis_mode={self.expected_camera_axis_mode} "
            f"expected_uses_live_axis={self.expected_uses_live_axis_convention} "
            f"expected_cam_old_axis={self.expected_gate_cam_old_axis} "
            f"expected_cam={self.expected_gate_cam} "
            f"pnp_cam={self.pnp_gate_cam} "
            f"camera_error={self.camera_error} "
            f"camera_error_norm={self.camera_error_norm:.3f} "
            f"size_depth={self.size_depth:.3f} "
            f"pnp_depth_minus_size_depth={self.pnp_depth_minus_size_depth:.3f} "
            f"world_error_norm={self.world_error_norm:.3f}"
        )

    def update_pnp_formulation_debug(self, expected_gate_world, expected_gate_cam):
        expected_gate_world = np.asarray(expected_gate_world, dtype=float).reshape(3)
        expected_gate_cam = np.asarray(expected_gate_cam, dtype=float).reshape(3)
        self.pnp_current_cam = self.pnp_gate_cam.copy()
        self.pnp_current_world = self.pnp_gate_world.copy()
        self.pnp_current_reproj_error = self.last_reprojection_error
        self.pnp_current_world_error_gt = self.world_error_norm
        self.live_solver_name = self.last_live_solver_name or "SOLVEPNP_ITERATIVE"
        self.pnp_solver_used = self.live_solver_name
        self.live_solver_world = self.pnp_current_world.copy()
        self.live_solver_reproj_error = self.pnp_current_reproj_error
        self.pnp_fallback_reason = self.last_pnp_fallback_reason

        current_entry = None
        best_entry = None
        best_world_error = float("inf")

        entries = self.last_pnp_formulation_debug
        if not isinstance(entries, list):
            entries = []

        for entry in entries:
            solver = entry.get("solver", "")
            order = entry.get("order", "")
            world = self._vec3_for_debug(entry.get("world", None))
            if not np.all(np.isfinite(world)):
                continue
            world_error = float(np.linalg.norm(world - expected_gate_world))
            cam = self._vec3_for_debug(entry.get("camera", None))
            camera_error = float(np.linalg.norm(cam - expected_gate_cam)) if np.all(np.isfinite(cam)) else float("nan")
            entry["world_error_to_expected_gt"] = world_error
            entry["camera_error_to_expected_gt_camera"] = camera_error
            if solver == "IPPE_SQUARE" and order == "tl_tr_br_bl" and current_entry is None:
                current_entry = entry
            if solver == "IPPE_SQUARE" and order == "tl_tr_br_bl":
                self.ippe_world_error_gt = world_error
            if solver == "ITERATIVE" and order == "tl_tr_br_bl":
                self.iterative_world_error_gt = world_error
            if world_error < best_world_error:
                best_world_error = world_error
                best_entry = entry

        if current_entry is not None:
            candidates = current_entry.get("candidates", [])
            if len(candidates) > 0:
                self.pnp_candidate0_world = self._vec3_for_debug(candidates[0].get("world", None))
                self.pnp_candidate0_error = float(candidates[0].get("error", np.nan))
                self.pnp_candidate0_projected_corners = candidates[0].get("projected_corners", None)
            if len(candidates) > 1:
                self.pnp_candidate1_world = self._vec3_for_debug(candidates[1].get("world", None))
                self.pnp_candidate1_error = float(candidates[1].get("error", np.nan))
                self.pnp_candidate1_projected_corners = candidates[1].get("projected_corners", None)

        if best_entry is not None:
            self.pnp_best_debug_solver = str(best_entry.get("solver", ""))
            self.pnp_best_debug_order = str(best_entry.get("order", ""))
            self.pnp_best_cam = self._vec3_for_debug(best_entry.get("camera", None))
            self.pnp_best_world = self._vec3_for_debug(best_entry.get("world", None))
            self.pnp_best_reproj_error = float(best_entry.get("reprojection_error", np.nan))
            self.pnp_best_world_error_gt = float(best_world_error)

    def update_gt_projected_center_debug(self, expected_gate_cam):
        self.pnp_gt_projected_center = np.full(2, np.nan, dtype=float)
        self.pnp_gt_projected_quad_center_error_px = float("nan")
        self.expected_gate_projected_center = np.full(2, np.nan, dtype=float)
        self.expected_vs_quad_center_error_px = float("nan")
        if self.last_camera_matrix is None:
            return
        expected_gate_cam = np.asarray(expected_gate_cam, dtype=float).reshape(3)
        if not np.all(np.isfinite(expected_gate_cam)):
            return
        try:
            projected, _ = cv2.projectPoints(
                np.zeros((1, 3), dtype=np.float32),
                np.zeros(3, dtype=float),
                expected_gate_cam.reshape(3, 1),
                self.last_camera_matrix,
                self.last_dist_coeffs,
            )
            pixel = projected.reshape(2)
        except Exception:
            return
        self.expected_gate_projected_center = pixel.astype(float)
        self.pnp_gt_projected_center = self.expected_gate_projected_center.copy()
        if np.isfinite(self.last_quad_center_x) and np.isfinite(self.last_quad_center_y):
            quad = np.array([self.last_quad_center_x, self.last_quad_center_y], dtype=float)
            self.expected_vs_quad_center_error_px = float(np.linalg.norm(pixel - quad))
            self.pnp_gt_projected_quad_center_error_px = self.expected_vs_quad_center_error_px

    def update_gate_size_sweep_debug(self, expected_gate_world):
        expected_gate_world = np.asarray(expected_gate_world, dtype=float).reshape(3)
        for key, size_label in (("190", "190"), ("200", "200"), ("210", "210")):
            entry = self.last_pnp_size_sweep.get(key, {}) if isinstance(self.last_pnp_size_sweep, dict) else {}
            cam = self._vec3_for_debug(entry.get("camera", None))
            world = self._vec3_for_debug(entry.get("world", None))
            setattr(self, f"pnp_size_{size_label}_cam", cam.copy())
            setattr(self, f"pnp_size_{size_label}_world", world.copy())
            setattr(
                self,
                f"pnp_size_{size_label}_reproj_error",
                float(entry.get("reprojection_error", np.nan)),
            )
            if np.all(np.isfinite(world)) and np.all(np.isfinite(expected_gate_world)):
                gt_error = float(np.linalg.norm(world - expected_gate_world))
            else:
                gt_error = float("nan")
            setattr(self, f"pnp_size_{size_label}_gt_error", gt_error)

    def _debug_camera_pose_world(self):
        r_body_camera = self.body_camera_matrix_for_mode(
            self.perception_transform_mode
        )
        if self.perception_world_pose_source_used == "gazebo_truth_sim_only":
            position, rotation = self._gazebo_model_pose_to_planner(
                self.image_gazebo_pose_snapshot
            )
            if position is not None and rotation is not None:
                return (
                    position + rotation @ self.camera_offset_body,
                    rotation @ r_body_camera,
                )

        rpy = np.asarray(
            self.image_pose_snapshot_rpy_perception, dtype=float
        ).reshape(3)
        position = np.asarray(
            self.image_pose_snapshot_position, dtype=float
        ).reshape(3)
        if not np.all(np.isfinite(rpy)) or not np.all(np.isfinite(position)):
            return None, None
        rotation = self.perception_node._rpy_to_rotmat(*map(float, rpy))
        return (
            position + rotation @ self.camera_offset_body,
            rotation @ r_body_camera,
        )

    def _project_debug_gate_center(self, center_world):
        center = self._vec3_for_debug(center_world)
        camera_matrix = getattr(self, "last_camera_matrix", None)
        if camera_matrix is None or not np.all(np.isfinite(center)):
            return None, None
        camera_position, camera_rotation = self._debug_camera_pose_world()
        if camera_position is None or camera_rotation is None:
            return None, None

        half = 0.75
        corners_world = np.array([
            center + [-half, 0.0, half],
            center + [half, 0.0, half],
            center + [half, 0.0, -half],
            center + [-half, 0.0, -half],
        ], dtype=float)
        points_world = np.vstack([corners_world, center])
        points_camera = (
            camera_rotation.T @ (points_world - camera_position).T
        ).T
        if not np.all(points_camera[:, 2] > 1e-6):
            return None, None
        projected, _ = cv2.projectPoints(
            points_camera.reshape(-1, 1, 3),
            np.zeros((3, 1), dtype=float),
            np.zeros((3, 1), dtype=float),
            np.asarray(camera_matrix, dtype=float).reshape(3, 3),
            np.asarray(
                getattr(self, "last_dist_coeffs", np.zeros((5, 1))),
                dtype=float,
            ).reshape(-1, 1),
        )
        projected = projected.reshape(-1, 2)
        return projected[:4], projected[4]

    @staticmethod
    def _compact_overlay_reason(reason, max_len=34):
        text = str(reason or "none").replace(" ", "_")
        return text if len(text) <= max_len else text[:max_len - 1] + "~"

    def _compute_gt_corner_rmse_debug(
        self,
        yolo_corners,
        projected_gt_corners,
        detection_race_idx,
        rmse_gate_idx,
    ):
        metrics = {
            "raw_indexed": float("nan"),
            "ordered": float("nan"),
            "best_permutation": float("nan"),
            "center_error": float("nan"),
            "order_warning": False,
            "gate_index_match": False,
        }
        if detection_race_idx is None or rmse_gate_idx is None:
            return metrics
        try:
            detection_race_idx = int(detection_race_idx)
            rmse_gate_idx = int(rmse_gate_idx)
        except (TypeError, ValueError):
            return metrics
        metrics["gate_index_match"] = detection_race_idx == rmse_gate_idx
        if not metrics["gate_index_match"]:
            return metrics

        yolo = np.asarray(yolo_corners, dtype=float).reshape(4, 2)
        gt_raw = np.asarray(projected_gt_corners, dtype=float).reshape(4, 2)
        if not np.all(np.isfinite(yolo)) or not np.all(np.isfinite(gt_raw)):
            return metrics

        yolo_ordered = self._ordered_image_quad(yolo)
        gt_ordered = self._ordered_image_quad(gt_raw)
        metrics["raw_indexed"] = float(
            np.sqrt(np.mean((yolo - gt_raw) ** 2))
        )
        metrics["ordered"] = float(
            np.sqrt(np.mean((yolo_ordered - gt_ordered) ** 2))
        )
        metrics["best_permutation"] = min(
            float(np.sqrt(np.mean((yolo - gt_raw[list(permutation)]) ** 2)))
            for permutation in itertools.permutations(range(4))
        )
        metrics["center_error"] = float(
            np.linalg.norm(np.mean(yolo, axis=0) - np.mean(gt_raw, axis=0))
        )
        best = metrics["best_permutation"]
        raw = metrics["raw_indexed"]
        metrics["order_warning"] = bool(
            raw > best + 10.0 and raw > 1.5 * max(best, 1.0)
        )
        return metrics

    def _debug_detection_style(self, entry):
        role = str(entry.get("role", ""))
        track_id = entry.get("track")
        canonical_track = self.canonical_track_id(track_id)
        active_id = self.canonical_track_id(self.active_target_track_id)
        race_idx = entry.get("race_idx")
        accepted = bool(entry.get("memory", False))
        if role == "rejected" or entry.get("reason") and not accepted:
            return "REJECTED", (0, 0, 255)
        if track_id is not None and canonical_track == active_id:
            return "ACTIVE", (0, 255, 0)
        if (
            accepted
            and entry.get("state") == "stable"
            and race_idx is not None
            and int(race_idx) > self.current_gate_idx
        ):
            return "LOOKAHEAD", (255, 255, 0)
        if role == "tentative_lookahead_target" and entry.get("state") == "stable":
            return "LOOKAHEAD", (255, 255, 0)
        return "IGNORED", (0, 255, 255)

    def _draw_all_detection_overlays(self, canvas):
        active_entry = None
        for idx in sorted(self.perception_detection_flow_entries):
            entry = self.perception_detection_flow_entries[idx]
            style, color = self._debug_detection_style(entry)
            keypoints = np.asarray(
                entry.get("keypoints", np.full((4, 3), np.nan)), dtype=float
            ).reshape(-1, 3)
            points = keypoints[:4, :2] if keypoints.shape[0] >= 4 else None
            if points is not None:
                self._draw_debug_corners(
                    canvas,
                    points,
                    color=color,
                    prefix=f"Y{idx}.",
                    connect=True,
                )
                finite = points[np.all(np.isfinite(points), axis=1)]
            else:
                finite = np.empty((0, 2), dtype=float)

            bbox = np.asarray(
                entry.get("bbox", np.full(4, np.nan)), dtype=float
            ).reshape(4)
            if np.all(np.isfinite(bbox)):
                p0 = (int(round(bbox[0])), int(round(bbox[1])))
                p1 = (int(round(bbox[2])), int(round(bbox[3])))
                cv2.rectangle(canvas, p0, p1, color, 2)
                label_origin = (p0[0], max(14, p0[1] - 5))
            elif len(finite):
                label_origin = (
                    int(np.min(finite[:, 0])),
                    max(14, int(np.min(finite[:, 1])) - 5),
                )
            else:
                continue

            center_world = np.asarray(
                entry.get("corrected", np.full(3, np.nan)), dtype=float
            ).reshape(3)
            race_idx = entry.get("race_idx")
            error_norm = float("nan")
            if (
                race_idx is not None
                and 0 <= int(race_idx) < len(self.gt_gates)
                and np.all(np.isfinite(center_world))
            ):
                error_norm = float(
                    np.linalg.norm(
                        center_world
                        - np.asarray(self.gt_gates[int(race_idx)], dtype=float)
                    )
                )
            track_text = "none" if entry.get("track") is None else entry["track"]
            race_text = "?" if race_idx is None else race_idx
            error_text = "nan" if not np.isfinite(error_norm) else f"{error_norm:.2f}"
            reason = self._compact_overlay_reason(entry.get("reason", ""))
            if style == "ACTIVE":
                label = f"ACTIVE t={track_text} r={race_text} e={error_text}"
                active_entry = entry
            elif style == "LOOKAHEAD":
                label = f"LH t={track_text} r={race_text} e={error_text}"
            elif style == "IGNORED":
                label = f"IGN t={track_text} r={race_text} {reason}"
            else:
                label = f"REJ {reason}"
            cv2.putText(
                canvas,
                label,
                label_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

            if len(finite):
                center = np.mean(finite, axis=0)
                cv2.drawMarker(
                    canvas,
                    tuple(np.rint(center).astype(int)),
                    color,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=12,
                    thickness=1,
                )

            if race_idx is not None and 0 <= int(race_idx) < len(self.gt_gates):
                gt_corners, _ = self._project_debug_gate_center(
                    self.gt_gates[int(race_idx)]
                )
                self._draw_debug_corners(
                    canvas,
                    gt_corners,
                    color=(255, 255, 255),
                    prefix=f"G{int(race_idx)}.",
                    connect=True,
                    cross=True,
                )

        return active_entry

    def _draw_planning_target_projections(self, canvas):
        active_corners, active_center = self._project_debug_gate_center(
            self.active_target_center
        )
        self._draw_debug_corners(
            canvas,
            active_corners,
            color=(255, 0, 255),
            prefix="",
            connect=True,
            cross=True,
        )
        if active_center is not None:
            cv2.drawMarker(
                canvas,
                tuple(np.rint(active_center).astype(int)),
                (255, 0, 255),
                markerType=cv2.MARKER_DIAMOND,
                markerSize=16,
                thickness=2,
            )

        lookahead_ids = {
            self.canonical_track_id(track_id)
            for track_id in (
                list(self.planning_lookahead_track_ids)
                + list(self.tentative_lookahead_track_ids)
            )
        }
        for track_id in lookahead_ids:
            track = self.gate_memory.get_track_by_id(track_id)
            if track is None:
                continue
            center = getattr(track, "filtered_center_world", None)
            if center is None:
                center = getattr(track, "center", None)
            corners, projected_center = self._project_debug_gate_center(center)
            color = (255, 255, 0) if getattr(track, "is_stable", False) else (0, 255, 255)
            self._draw_debug_corners(
                canvas,
                corners,
                color=color,
                prefix="",
                connect=True,
                cross=True,
            )
            if projected_center is not None:
                cv2.drawMarker(
                    canvas,
                    tuple(np.rint(projected_center).astype(int)),
                    color,
                    markerType=cv2.MARKER_DIAMOND,
                    markerSize=12,
                    thickness=1,
                )

    def save_perception_debug_frame(
        self,
        frame,
        timestamp,
        track_id=None,
        accepted=False,
        rejection_reason="",
    ):
        if not self.save_perception_debug_frames or frame is None:
            return None

        try:
            os.makedirs(self.perception_debug_frame_dir, exist_ok=True)
        except Exception as exc:
            print(f"[DEBUG FRAME] could not create debug dir: {exc}")
            return None

        if len(frame.shape) == 2:
            canvas = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            canvas = frame.copy()

        raw = self._corners_for_debug(self.last_raw_image_corners)
        ordered = self._corners_for_debug(self.last_ordered_image_corners)
        debug_best = self._corners_for_debug(self.last_pnp_debug_best_ordered_corners)
        reprojected = self._corners_for_debug(self.last_reprojected_image_corners)
        candidate0_projected = self._corners_for_debug(self.pnp_candidate0_projected_corners)
        candidate1_projected = self._corners_for_debug(self.pnp_candidate1_projected_corners)

        if self.debug_verbose_overlay:
            self._draw_debug_corners(canvas, raw, color=(0, 0, 255), prefix="yolo", connect=False)
            self._draw_debug_corners(canvas, ordered, color=(0, 255, 0), prefix="sel", connect=True)
            self._draw_debug_corners(canvas, debug_best, color=(255, 128, 0), prefix="best", connect=True)
            self._draw_debug_corners(canvas, reprojected, color=(255, 0, 0), prefix="rep", connect=True, cross=True)
            self._draw_debug_corners(canvas, candidate0_projected, color=(255, 255, 0), prefix="c0", connect=True, cross=True)
            self._draw_debug_corners(canvas, candidate1_projected, color=(255, 0, 255), prefix="c1", connect=True, cross=True)

        active_entry = self._draw_all_detection_overlays(canvas)
        self._draw_planning_target_projections(canvas)

        gate_world = self._vec3_for_debug(self.last_gate_center_world_debug)
        gate_camera = self._vec3_for_debug(self.last_gate_center_camera)
        expected_camera = self._vec3_for_debug(self.expected_gate_cam)
        pnp_tvec = self._vec3_for_debug(self.last_pnp_tvec)
        filtered = self._vec3_for_debug(self.track_filtered_center)
        expected_world = self._vec3_for_debug(self.expected_gate_world)
        axis_best_world = self._vec3_for_debug(self.camera_axis_sweep_best_world)
        live_error = self._vec3_for_debug(self.live_minus_expected)
        mavsdk_pos = self._vec3_for_debug(self.image_pose_snapshot_position)
        gazebo_pos = self._vec3_for_debug(self.gazebo_model_pos_world)
        track_label = "none" if track_id is None else str(track_id)
        active_status = (
            f"active_track={self.active_target_track_id}"
            if self.active_target_track_id is not None
            else "active_track=none"
        )
        status = "accepted" if accepted else f"rejected:{rejection_reason}"
        verbose_overlay_lines = [
            f"{status}",
            f"track_id={track_label} conf={self.gate_confidence:.3f} reproj={self.last_reprojection_error:.2f}px",
            f"lookahead={'on' if self.use_lookahead_gate_filter else 'off'} tentative={self.tentative_track_ids} stable={self.stable_track_ids} admitted={self.race_accepted_track_ids}",
            f"drone_mavsdk=({mavsdk_pos[0]:.2f},{mavsdk_pos[1]:.2f},{mavsdk_pos[2]:.2f})",
            f"drone_gazebo=({gazebo_pos[0]:.2f},{gazebo_pos[1]:.2f},{gazebo_pos[2]:.2f})",
            f"filter hits={self.track_observations} hist={self.track_history_len} stable={self.track_is_stable} score={self.track_stability_score:.2f} block={self.promotion_blocked_reason or 'none'}",
            f"filtered=({filtered[0]:.2f},{filtered[1]:.2f},{filtered[2]:.2f}) std={self.track_center_std_norm:.2f}",
            f"live={self.live_solver_name} fallback={self.pnp_fallback_reason or 'none'}",
            f"pnp_order sel={self.pnp_selected_order} best={self.pnp_debug_best_order} reorder={self.allow_pnp_corner_reordering}",
            f"poly area={self.keypoint_polygon_signed_area:.1f} {self.keypoint_polygon_winding} edges=({self.keypoint_edge_top:.1f},{self.keypoint_edge_right:.1f},{self.keypoint_edge_bottom:.1f},{self.keypoint_edge_left:.1f})",
            f"bbox-poly=({self.keypoint_bbox_polygon_delta[0]:.1f},{self.keypoint_bbox_polygon_delta[1]:.1f}) lat={self.pnp_lateral_angle:.3f} img={self.image_center_offset_normalized:.3f}",
            f"world=({gate_world[0]:.2f},{gate_world[1]:.2f},{gate_world[2]:.2f})",
            f"axis_best={self.camera_axis_sweep_best_mode or 'none'} ({axis_best_world[0]:.2f},{axis_best_world[1]:.2f},{axis_best_world[2]:.2f})",
            f"expected_gate[{self.debug_expected_gate_idx}]=({expected_world[0]:.2f},{expected_world[1]:.2f},{expected_world[2]:.2f})",
            f"live-expected=({live_error[0]:.2f},{live_error[1]:.2f},{live_error[2]:.2f}) lat={self.live_lateral_error_m:.2f}",
            f"curr_gt_err={self.pnp_current_world_error_gt:.2f} best={self.pnp_best_debug_solver}/{self.pnp_best_debug_order}:{self.pnp_best_world_error_gt:.2f}",
            f"best_world=({self.pnp_best_world[0]:.2f},{self.pnp_best_world[1]:.2f},{self.pnp_best_world[2]:.2f}) reproj={self.pnp_best_reproj_error:.2f}",
            f"pnp_camera=({gate_camera[0]:.2f},{gate_camera[1]:.2f},{gate_camera[2]:.2f})",
            f"expected_camera=({expected_camera[0]:.2f},{expected_camera[1]:.2f},{expected_camera[2]:.2f})",
            f"expected_axis={self.expected_camera_axis_mode or 'none'} live_axis={self.live_camera_axis_mode or 'none'} uses_live={self.expected_uses_live_axis_convention}",
            f"expected_old_axis=({self.expected_gate_cam_old_axis[0]:.2f},{self.expected_gate_cam_old_axis[1]:.2f},{self.expected_gate_cam_old_axis[2]:.2f})",
            f"expected_proj=({self.expected_gate_projected_center[0]:.1f},{self.expected_gate_projected_center[1]:.1f}) quad=({self.last_quad_center_x:.1f},{self.last_quad_center_y:.1f}) err={self.expected_vs_quad_center_error_px:.1f}px",
            f"size_depth={self.size_depth:.2f} width={self.size_depth_from_width:.2f} height={self.size_depth_from_height:.2f} dz={self.pnp_depth_minus_size_depth:.2f}",
            f"camera_error=({self.camera_error[0]:.2f},{self.camera_error[1]:.2f},{self.camera_error[2]:.2f}) norm={self.camera_error_norm:.2f}",
            f"tvec=({pnp_tvec[0]:.2f},{pnp_tvec[1]:.2f},{pnp_tvec[2]:.2f})",
            f"{active_status} t={timestamp:.3f}",
        ]
        decision_entry = active_entry
        if decision_entry is None and track_id is not None:
            decision_entry = next(
                (
                    entry
                    for entry in self.perception_detection_flow_entries.values()
                    if entry.get("track") == track_id
                ),
                None,
            )
        if decision_entry is None and self.perception_detection_flow_entries:
            decision_entry = self.perception_detection_flow_entries[
                sorted(self.perception_detection_flow_entries)[0]
            ]

        if decision_entry is None:
            decision_role = "rejected" if not accepted else "active"
            decision_track = track_label
            decision_race = "?"
            decision_reason = rejection_reason or self.replan_reason or "none"
            geometry_points = ordered
        else:
            style, _ = self._debug_detection_style(decision_entry)
            decision_role = {
                "ACTIVE": "active",
                "LOOKAHEAD": "lookahead",
                "IGNORED": "lookahead_ignored",
                "REJECTED": "rejected",
            }[style]
            decision_track = (
                "none"
                if decision_entry.get("track") is None
                else str(decision_entry.get("track"))
            )
            decision_race = (
                "?"
                if decision_entry.get("race_idx") is None
                else str(decision_entry.get("race_idx"))
            )
            decision_reason = (
                decision_entry.get("reason")
                or self.tentative_lookahead_replan_blocked_reason
                or self.promotion_blocked_reason
                or self.replan_reason
                or "none"
            )
            keypoints = np.asarray(
                decision_entry.get("keypoints", np.full((4, 3), np.nan)),
                dtype=float,
            ).reshape(-1, 3)
            geometry_points = keypoints[:4, :2] if keypoints.shape[0] >= 4 else ordered

        span_w = span_h = quad_area = float("nan")
        gt_metrics = {
            "raw_indexed": float("nan"),
            "ordered": float("nan"),
            "best_permutation": float("nan"),
            "center_error": float("nan"),
            "order_warning": False,
            "gate_index_match": False,
        }
        detection_race_idx = (
            decision_entry.get("race_idx")
            if decision_entry is not None
            else None
        )
        debug_gt_idx = getattr(self, "debug_expected_gate_idx", None)
        try:
            debug_gt_idx = int(debug_gt_idx)
        except (TypeError, ValueError, OverflowError):
            debug_gt_idx = None
        rmse_gate_idx = (
            debug_gt_idx
            if debug_gt_idx is not None
            and 0 <= debug_gt_idx < len(self.gt_gates)
            else detection_race_idx
        )
        if geometry_points is not None:
            points = np.asarray(geometry_points, dtype=float).reshape(-1, 2)
            if points.shape[0] >= 4 and np.all(np.isfinite(points[:4])):
                points = points[:4]
                span_w = 0.5 * (
                    np.linalg.norm(points[1] - points[0])
                    + np.linalg.norm(points[2] - points[3])
                )
                span_h = 0.5 * (
                    np.linalg.norm(points[3] - points[0])
                    + np.linalg.norm(points[2] - points[1])
                )
                quad_area = abs(float(cv2.contourArea(points.astype(np.float32))))
                if rmse_gate_idx is not None and 0 <= int(rmse_gate_idx) < len(self.gt_gates):
                    gt_corners, _ = self._project_debug_gate_center(
                        self.gt_gates[int(rmse_gate_idx)]
                    )
                    if gt_corners is not None:
                        gt_metrics = self._compute_gt_corner_rmse_debug(
                            yolo_corners=points,
                            projected_gt_corners=gt_corners,
                            detection_race_idx=detection_race_idx,
                            rmse_gate_idx=rmse_gate_idx,
                        )

        self.gt_rmse_px_raw_indexed = gt_metrics["raw_indexed"]
        self.gt_rmse_px_ordered = gt_metrics["ordered"]
        self.gt_rmse_px_best_permutation = gt_metrics["best_permutation"]
        self.gt_center_err_px = gt_metrics["center_error"]
        self.gt_corner_order_warning = gt_metrics["order_warning"]
        self.rmse_gate_idx = rmse_gate_idx
        self.detection_race_idx_for_rmse = detection_race_idx

        compact_lines = [
            "DECISION: "
            f"role={decision_role} track={decision_track} race={decision_race} "
            f"source={self.perception_world_pose_source_used} "
            f"reason={self._compact_overlay_reason(decision_reason)}",
            "GEOM: "
            f"span_w_px={span_w:.1f} span_h_px={span_h:.1f} "
            f"quad_area={quad_area:.0f}",
        ]
        if gt_metrics["gate_index_match"]:
            compact_lines.append(
                "GT: "
                f"gt_rmse_px_ordered={gt_metrics['ordered']:.1f} "
                f"best={gt_metrics['best_permutation']:.1f} "
                f"raw={gt_metrics['raw_indexed']:.1f} "
                f"center={gt_metrics['center_error']:.1f} "
                f"order_warn={int(gt_metrics['order_warning'])}"
            )
        else:
            compact_lines.append(
                "GT: rmse=N/A "
                f"rmse_gate_idx={rmse_gate_idx} "
                f"detection_race_idx={detection_race_idx}"
            )
        verbose_overlay_lines.append(
            "gt_corner_rmse "
            f"raw_indexed={gt_metrics['raw_indexed']:.1f} "
            f"ordered={gt_metrics['ordered']:.1f} "
            f"best_permutation={gt_metrics['best_permutation']:.1f} "
            f"center={gt_metrics['center_error']:.1f} "
            f"warning={gt_metrics['order_warning']} "
            f"rmse_gate_idx={rmse_gate_idx} "
            f"detection_race_idx={detection_race_idx}"
        )
        if self.target_update_event:
            shift = (
                np.asarray(self.target_update_new, dtype=float).reshape(3)
                - np.asarray(self.target_update_previous, dtype=float).reshape(3)
            )
            compact_lines.append(
                "TARGET_SHIFT: "
                f"dx={shift[0]:+.2f} dy={shift[1]:+.2f} dz={shift[2]:+.2f} "
                f"norm={np.linalg.norm(shift):.2f}"
            )
        overlay_lines = (
            verbose_overlay_lines if self.debug_verbose_overlay else compact_lines
        )
        self._draw_debug_text_block(canvas, overlay_lines)

        safe_track = "none" if track_id is None else str(track_id)
        filename = f"frame_{timestamp:.3f}_track_{safe_track}.png"
        filename = filename.replace(".", "p", 1)
        path = os.path.join(self.perception_debug_frame_dir, filename)
        ok = cv2.imwrite(path, canvas)
        if not ok:
            print(f"[DEBUG FRAME] cv2.imwrite failed: {path}")
            return None
        print(f"[DEBUG FRAME] saved {path}")
        return path

    @staticmethod
    def _corners_for_debug(corners):
        if corners is None:
            return None
        try:
            pts = np.asarray(corners, dtype=float).reshape(-1, 2)
        except Exception:
            return None
        if pts.shape[0] == 0:
            return None
        return pts

    @staticmethod
    def _vec3_for_debug(value):
        if value is None:
            return np.full(3, np.nan, dtype=float)
        try:
            arr = np.asarray(value, dtype=float).reshape(-1)
        except Exception:
            return np.full(3, np.nan, dtype=float)
        if arr.size < 3:
            return np.full(3, np.nan, dtype=float)
        return arr[:3]

    @staticmethod
    def _draw_debug_corners(canvas, corners, color, prefix, connect=False, cross=False):
        if corners is None:
            return
        pts = []
        for i, pt in enumerate(corners[:4]):
            if not np.all(np.isfinite(pt)):
                continue
            xy = (int(round(pt[0])), int(round(pt[1])))
            pts.append(xy)
            if cross:
                cv2.drawMarker(
                    canvas,
                    xy,
                    color,
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=12,
                    thickness=2,
                )
            else:
                cv2.circle(canvas, xy, 4, color, -1)
            if prefix:
                cv2.putText(
                    canvas,
                    f"{prefix}{i}",
                    (xy[0] + 5, xy[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        if connect and len(pts) >= 2:
            cv2.polylines(
                canvas,
                [np.asarray(pts, dtype=np.int32)],
                isClosed=len(pts) >= 4,
                color=color,
                thickness=2,
            )

    @staticmethod
    def _draw_debug_text_block(canvas, lines):
        line_h = 15
        width = min(520, max(220, canvas.shape[1] - 20))
        height = line_h * len(lines) + 8
        x = 8
        y_top = max(0, canvas.shape[0] - height - 8)
        y0 = y_top + 14
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, y_top), (width, y_top + height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0.0, canvas)
        for i, text in enumerate(lines):
            y = y0 + i * line_h
            cv2.putText(
                canvas,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

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

        if self.use_lookahead_gate_filter and not getattr(tr, "is_stable", False):
            self.race_order_rejected_reason = getattr(
                tr,
                "promotion_blocked_reason",
                "track_not_stable",
            ) or "track_not_stable"
            self.rejected_track_temporary_vs_permanent = "temporary"
            self.active_target_admission_status = "pending_stability"
            print(
                f"TRACK {tr.id} blocked from promotion: "
                f"reason={self.race_order_rejected_reason}"
            )
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
        self.race_order_inserted = True
        self.active_target_admission_status = "accepted"
        self.race_admitted_track_ids = list(self.race_accepted_track_ids)
        print(
            f"TRACK {track_id} admitted to race candidate pool; "
            "sequence index assigned by geometric progress"
        )
        return True

    def assign_race_order_from_progress(self, committed_tracks):
        """Assign uncompleted tracks without letting a farther gate take the current slot."""
        current_pos = np.array([
            self.telemetry.pos["x"],
            self.telemetry.pos["y"],
            self.telemetry.pos["z"],
        ], dtype=float)
        previous_idx = int(self.current_gate_idx)
        candidates = []
        rejected = []

        for tr in committed_tracks:
            track_id = self.canonical_track_id(tr.id)
            center_source = getattr(tr, "filtered_center_world", None)
            if center_source is None:
                center_source = getattr(tr, "center", None)
            if track_id is None or center_source is None:
                continue
            center = np.asarray(center_source, dtype=float).reshape(3)
            valid, reason = self.validate_planning_target(center)
            if not valid:
                rejected.append((track_id, reason))
                continue
            if track_id in self.completed_track_ids_this_cycle or self.is_near_completed_gate(center):
                rejected.append((track_id, "completed_gate"))
                continue
            candidates.append({
                "track": tr,
                "track_id": track_id,
                "center": center,
                "distance": float(np.linalg.norm(center - current_pos)),
                "progress": float("nan"),
            })

        candidates.sort(key=lambda item: (item["distance"], item["track_id"]))
        self.current_gate_candidate_track_ids = [item["track_id"] for item in candidates]
        self.selected_current_track_id = candidates[0]["track_id"] if candidates else None
        self.rejected_current_track_ids = [
            item["track_id"] for item in candidates[1:]
        ] + [track_id for track_id, _ in rejected]

        if len(candidates) >= 2:
            nearest = candidates[0]["center"]
            farthest = max(candidates[1:], key=lambda item: item["distance"])["center"]
            course_direction = farthest - nearest
            norm = float(np.linalg.norm(course_direction))
            if norm > 1e-6:
                course_direction /= norm
                if float(np.dot(nearest - current_pos, course_direction)) < 0.0:
                    course_direction *= -1.0
            else:
                course_direction = (nearest - current_pos) / max(
                    float(np.linalg.norm(nearest - current_pos)), 1e-6
                )
        elif len(candidates) == 1:
            course_direction = candidates[0]["center"] - current_pos
            course_direction /= max(float(np.linalg.norm(course_direction)), 1e-6)
        else:
            course_direction = np.array([0.0, 1.0, 0.0], dtype=float)

        for item in candidates:
            item["progress"] = float(np.dot(item["center"] - current_pos, course_direction))

        if candidates:
            current = candidates[0]
            future = sorted(
                candidates[1:],
                key=lambda item: (item["progress"], item["distance"], item["track_id"]),
            )
            assigned = [current] + future
        else:
            assigned = []

        assigned_index = {
            item["track_id"]: previous_idx + rank
            for rank, item in enumerate(assigned)
        }
        self.future_lookahead_track_ids = [
            item["track_id"] for item in assigned[1:]
        ]

        existing_order = [
            self.canonical_track_id(track_id)
            for track_id in self.race_progression.inferred_order
        ]
        completed_prefix = []
        for track_id in existing_order[:previous_idx]:
            if track_id is not None and track_id not in completed_prefix:
                completed_prefix.append(track_id)

        contiguous_order = list(completed_prefix)
        missing_preceding = False
        for item in assigned:
            track_id = item["track_id"]
            if missing_preceding or track_id not in self.race_accepted_track_ids:
                missing_preceding = True
                continue
            contiguous_order.append(track_id)

        self.race_progression.inferred_order = contiguous_order
        self.race_progression.cursor = min(
            max(int(self.race_progression.cursor), previous_idx),
            len(contiguous_order),
        )

        rejection_parts = []
        for item in assigned[1:]:
            rejection_parts.append(
                f"track{item['track_id']}:farther_than_current_track{self.selected_current_track_id}"
            )
        rejection_parts.extend(f"track{track_id}:{reason}" for track_id, reason in rejected)
        self.current_selection_rejection_reason = ";".join(rejection_parts)

        debug_parts = []
        for rank, item in enumerate(assigned):
            track_id = item["track_id"]
            race_idx = assigned_index[track_id]
            if rank == 0:
                reason = "selected_nearest_valid_current_candidate"
            elif track_id in self.race_accepted_track_ids:
                reason = "future_gate_progress_order"
            else:
                reason = "future_gate_pending_race_admission"
            if track_id in self.race_accepted_track_ids and track_id not in contiguous_order:
                reason += "|withheld_until_preceding_gate_admitted"
            center = item["center"]
            debug_parts.append(
                f"track{track_id}:center={center[0]:.2f}/{center[1]:.2f}/{center[2]:.2f},"
                f"dist={item['distance']:.2f},progress={item['progress']:.2f},"
                f"score={-item['progress']:.2f},prev_active={previous_idx},"
                f"assigned={race_idx},reason={reason}"
            )
        for track_id, reason in rejected:
            debug_parts.append(
                f"track{track_id}:prev_active={previous_idx},assigned=None,reason=rejected:{reason}"
            )
        self.race_order_assignment_debug = ";".join(debug_parts)
        if debug_parts:
            print("[RACE ORDER ASSIGNMENT] " + self.race_order_assignment_debug)

    def refresh_race_order_from_memory(self):
        self.refresh_landmark_merges()
        committed_tracks = self.gate_memory.get_committed_tracks()
        stable_tracks = self.gate_memory.get_stable_tracks()
        self.update_gate_filter_summary_logs()
        self.committed_track_centers_log = ";".join(
            f"{tr.id}:{tr.center[0]:.2f},{tr.center[1]:.2f},{tr.center[2]:.2f}:h{tr.hits}"
            for tr in committed_tracks
        )
        tracks_for_admission = stable_tracks if self.use_lookahead_gate_filter else committed_tracks
        for tr in tracks_for_admission:
            self.accept_track_into_race_order(tr)

        self.assign_race_order_from_progress(committed_tracks)
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

    def nearest_tau_on_active_plan_xy(self, position):
        planner = self.planner
        total_time = float(getattr(planner, "total_time", 0.0))
        if planner is None or total_time <= 0.0:
            return float("nan"), float("nan")

        position = np.asarray(position, dtype=float).reshape(3)
        if not np.all(np.isfinite(position)):
            return float("nan"), float("nan")

        sample_count = max(2, int(self.reference_projection_sample_count))
        best_tau = float("nan")
        best_distance = float("inf")
        best_point = None
        for tau in np.linspace(0.0, total_time, sample_count):
            p, _, _ = planner.sample(float(tau))
            p = np.asarray(p, dtype=float).reshape(3)
            if not np.all(np.isfinite(p)):
                continue
            distance = float(np.linalg.norm(p[:2] - position[:2]))
            if distance < best_distance:
                best_distance = distance
                best_tau = float(tau)
                best_point = p

        if best_point is None:
            return float("nan"), float("nan")

        progress_lead_m = float("nan")
        if np.isfinite(self.sample_tau_after_progress_limit):
            sample_p, _, _ = planner.sample(float(self.sample_tau_after_progress_limit))
            sample_p = np.asarray(sample_p, dtype=float).reshape(3)
            progress_lead_m = float(np.linalg.norm(sample_p[:2] - best_point[:2]))

        return best_tau, progress_lead_m

    def state_position_array(self, state):
        pos = getattr(state, "pos", None)
        if isinstance(pos, dict):
            return np.array([pos["x"], pos["y"], pos["z"]], dtype=float)
        return np.asarray(pos, dtype=float).reshape(3)

    def compute_reference_sample_tau(self, wall_tau, state):
        planner_total_time = float(getattr(self.planner, "total_time", 0.0))
        wall_tau = max(0.0, float(wall_tau))
        sample_tau = min(wall_tau, planner_total_time)

        self.wall_tau = wall_tau
        self.vehicle_nearest_tau_on_plan = float("nan")
        self.sample_tau_progress_limited = False
        self.sample_tau_before_progress_limit = sample_tau
        self.sample_tau_after_progress_limit = sample_tau
        self.reference_tau_lead_s = float("nan")
        self.reference_progress_lead_m = float("nan")
        self.reference_virtual_clock_enabled = False

        if self.previous_sample_tau_plan_id != self.active_plan_id:
            self.previous_sample_tau_used = 0.0
            self.previous_sample_tau_plan_id = self.active_plan_id

        race_complete = (
            self.race_gate_count is not None
            and self.current_gate_idx >= int(self.race_gate_count)
        )
        should_limit = (
            self.use_perception
            and self.planner is not None
            and planner_total_time > 0.0
            and len(self.active_target_gates) > 0
            and not race_complete
        )
        if should_limit:
            position = self.state_position_array(state)
            vehicle_tau, _ = self.nearest_tau_on_active_plan_xy(position)
            self.vehicle_nearest_tau_on_plan = vehicle_tau
            if np.isfinite(vehicle_tau):
                self.reference_virtual_clock_enabled = True
                allowed_tau = min(
                    planner_total_time,
                    vehicle_tau + float(self.reference_progress_tau_lead_s),
                )
                sample_tau = min(sample_tau, allowed_tau)
                self.sample_tau_progress_limited = (
                    self.sample_tau_before_progress_limit - sample_tau > 1e-6
                )

        sample_tau = max(float(self.previous_sample_tau_used), float(sample_tau))
        sample_tau = min(sample_tau, planner_total_time)
        self.previous_sample_tau_used = sample_tau
        self.sample_tau_after_progress_limit = sample_tau
        self.reference_tau_lead_s = (
            sample_tau - self.vehicle_nearest_tau_on_plan
            if np.isfinite(self.vehicle_nearest_tau_on_plan)
            else float("nan")
        )
        if self.reference_virtual_clock_enabled:
            _, progress_lead_m = self.nearest_tau_on_active_plan_xy(
                self.state_position_array(state)
            )
            self.reference_progress_lead_m = progress_lead_m
        return sample_tau

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

    def reset_target_update_event_debug(self):
        self.target_update_event = False
        self.target_update_previous = np.full(3, np.nan, dtype=float)
        self.target_update_new = np.full(3, np.nan, dtype=float)
        self.target_update_delta_m = float("nan")
        self.target_update_source_track_id = None
        self.target_update_raw_detection_center = np.full(3, np.nan, dtype=float)
        self.target_update_filtered_track_center = np.full(3, np.nan, dtype=float)
        self.target_update_reason = ""
        self.active_target_shift_suppressed = False
        self.distance_to_active_target_at_shift = float("nan")
        self.target_shift_xy = float("nan")
        self.target_shift_z = float("nan")
        self.shift_replan_allowed = False
        self.shift_replan_suppressed_reason = ""

    def record_target_update_debug(self, previous, new, track_id, reason):
        new = np.asarray(new, dtype=float).reshape(3)
        previous = (
            np.asarray(previous, dtype=float).reshape(3)
            if previous is not None
            else np.full(3, np.nan, dtype=float)
        )
        delta = (
            float(np.linalg.norm(new - previous))
            if np.all(np.isfinite(previous))
            else float("inf")
        )
        if np.isfinite(delta) and delta <= 0.10:
            return

        canonical_id = self.canonical_track_id(track_id)
        raw = np.full(3, np.nan, dtype=float)
        filtered = np.full(3, np.nan, dtype=float)
        tr = self.gate_memory.get_track_by_id(canonical_id) if canonical_id is not None else None
        if tr is not None:
            filtered_source = getattr(tr, "filtered_center_world", None)
            if filtered_source is None:
                filtered_source = getattr(tr, "center", None)
            if filtered_source is not None:
                filtered = np.asarray(filtered_source, dtype=float).reshape(3)
            history = list(getattr(tr, "obs_history", []))
            if history:
                raw = np.asarray(history[-1].center_world, dtype=float).reshape(3)

        self.target_update_event = True
        self.target_update_previous = previous.copy()
        self.target_update_new = new.copy()
        self.target_update_delta_m = delta
        self.target_update_source_track_id = canonical_id
        self.target_update_raw_detection_center = raw.copy()
        self.target_update_filtered_track_center = filtered.copy()
        self.target_update_reason = str(reason or "")
        print(
            "[TARGET UPDATE] "
            f"reason={self.target_update_reason} track_id={canonical_id} "
            f"delta={delta:.3f} previous={previous.tolist()} new={new.tolist()} "
            f"raw={raw.tolist()} filtered={filtered.tolist()}"
        )

    def _record_promotion_candidate_stability(self, tr):
        if tr is None:
            return
        self.promotion_candidate_hits = int(getattr(tr, "hits", 0))
        self.promotion_candidate_inliers = int(getattr(tr, "inlier_count", 0))
        self.promotion_candidate_outliers = int(getattr(tr, "outlier_count", 0))
        camera_std = np.asarray(
            getattr(tr, "center_camera_std", np.full(3, np.nan)), dtype=float
        ).reshape(3)
        center_std = np.asarray(
            getattr(tr, "center_world_std", np.full(3, np.nan)), dtype=float
        ).reshape(3)
        self.promotion_candidate_camera_std = (
            float(np.linalg.norm(camera_std))
            if np.all(np.isfinite(camera_std))
            else float("nan")
        )
        self.promotion_candidate_center_std = (
            float(np.linalg.norm(center_std))
            if np.all(np.isfinite(center_std))
            else float("nan")
        )
        self.promotion_candidate_stability_blocker = str(
            getattr(tr, "promotion_blocked_reason", "") or ""
        )

    def _try_previous_horizon_promotion_fallback(
        self,
        vehicle_pos,
        completed_track_id,
        completed_center,
    ):
        self.promotion_fallback_previous_horizon_used = False
        self.promotion_fallback_rejection_reason = ""

        handoff = self.pending_lookahead_handoff
        if handoff is None:
            self.promotion_fallback_rejection_reason = "previous_horizon_has_no_future_track"
            return False

        track_id = self.canonical_track_id(handoff.get("track_id"))
        center = np.asarray(
            handoff.get("center", np.full(3, np.nan)), dtype=float
        ).reshape(3)
        waypoint_type = str(handoff.get("waypoint_type", "") or "")
        self.promotion_fallback_candidate_track_id = track_id
        self.promotion_fallback_candidate_center = center.copy()

        tr = self.gate_memory.get_track_by_id(track_id) if track_id is not None else None
        self._record_promotion_candidate_stability(tr)

        def reject(reason):
            self.promotion_fallback_rejection_reason = str(reason)
            self.promotion_blocked_reason = str(reason)
            print(
                "[LOOKAHEAD HANDOFF FALLBACK] rejected "
                f"track_id={track_id} center={center.tolist()} reason={reason}"
            )
            return False

        if track_id is None or track_id < 0:
            return reject("fallback_invalid_track_id")
        if track_id == self.canonical_track_id(completed_track_id):
            return reject("fallback_is_completed_track")
        if tr is None:
            return reject("fallback_track_missing")
        if not bool(getattr(tr, "committed", False)):
            return reject("fallback_track_not_committed")
        if waypoint_type not in ("soft_committed_unstable", "soft_tentative", "hard_stable"):
            return reject(f"fallback_invalid_previous_waypoint_type:{waypoint_type}")
        if int(getattr(tr, "hits", 0)) < int(self.planning_lookahead_min_hits):
            return reject(
                f"fallback_insufficient_hits:{int(getattr(tr, 'hits', 0))}"
                f"<{int(self.planning_lookahead_min_hits)}"
            )
        valid, reason = self.validate_planning_target(center)
        if not valid:
            return reject(f"fallback_invalid_target:{reason}")
        if self.is_near_completed_gate(center):
            return reject("fallback_near_completed_gate")

        vehicle_pos = np.asarray(vehicle_pos, dtype=float).reshape(3)
        completed_center = np.asarray(completed_center, dtype=float).reshape(3)
        previous_gate_idx = self.current_gate_idx - 1
        expected_next = None
        race_direction = None
        if (
            self.gt_navigation_enabled()
            and
            0 <= previous_gate_idx < len(self.gt_gates)
            and 0 <= self.current_gate_idx < len(self.gt_gates)
        ):
            previous_gt = np.asarray(self.gt_gates[previous_gate_idx], dtype=float).reshape(3)
            expected_next = np.asarray(
                self.gt_gates[self.current_gate_idx], dtype=float
            ).reshape(3)
            race_direction = expected_next - previous_gt
            self.post_completion_direction_source = "gt_expected_next"
            self.gt_behavior_dependency_used = True
            self.gt_behavior_dependency_reason = "previous_horizon_fallback_gt_expected_next"
        elif self.approach_vector is not None:
            race_direction = np.asarray(self.approach_vector, dtype=float).reshape(3)
            self.post_completion_direction_source = "approach_vector"
        else:
            race_direction = center - completed_center
            self.post_completion_direction_source = "completed_gate_to_candidate"

        direction_norm = float(np.linalg.norm(race_direction))
        if not np.isfinite(direction_norm) or direction_norm < 1e-6:
            return reject("fallback_race_direction_unavailable")
        race_direction = race_direction / direction_norm
        completed_progress = float(np.dot(center - completed_center, race_direction))
        vehicle_progress = float(np.dot(center - vehicle_pos, race_direction))
        if completed_progress <= 0.0:
            return reject(f"fallback_not_ahead_of_completed_gate:{completed_progress:.2f}")
        if vehicle_progress <= 0.0:
            return reject(f"fallback_behind_vehicle:{vehicle_progress:.2f}")
        if expected_next is not None:
            expected_error = float(np.linalg.norm(center - expected_next))
            if expected_error > float(self.max_gate_jump):
                return reject(
                    f"fallback_too_far_from_expected_next:{expected_error:.2f}"
                    f">{float(self.max_gate_jump):.2f}"
                )

        valid, reason = self.validate_candidate_target(
            center, vehicle_pos, track_id=track_id
        )
        if not valid:
            return reject(f"fallback_candidate_invalid:{reason}")
        if self.race_progression.predefined_order is not None:
            return reject("fallback_cannot_modify_predefined_race_order")

        if track_id not in self.race_accepted_track_ids:
            self.race_accepted_track_ids.append(track_id)
        order = [
            self.canonical_track_id(tid)
            for tid in self.race_progression.inferred_order
            if self.canonical_track_id(tid) != track_id
        ]
        insert_at = min(max(int(self.race_progression.cursor), 0), len(order))
        order.insert(insert_at, track_id)
        self.race_progression.inferred_order = order
        self.race_order_track_ids = list(order)
        self.race_admitted_track_ids = list(self.race_accepted_track_ids)
        setattr(tr, "race_order_index", int(self.current_gate_idx))

        handoff["source"] = "previous_horizon_fallback"
        self.pending_lookahead_handoff = handoff
        self.next_track_after_completion_id = track_id
        self.next_track_available_after_completion = True
        self.promotion_fallback_previous_horizon_used = True
        self.promotion_fallback_rejection_reason = ""
        self.promotion_blocked_reason = ""
        print(
            "[LOOKAHEAD HANDOFF FALLBACK] accepted "
            f"track_id={track_id} center={center.tolist()} "
            f"hits={self.promotion_candidate_hits} "
            f"inliers={self.promotion_candidate_inliers} "
            f"outliers={self.promotion_candidate_outliers} "
            f"stability_blocker={self.promotion_candidate_stability_blocker or 'none'}"
        )
        return True

    def _try_post_completion_current_candidate_fallback(
        self,
        vehicle_pos,
        completed_track_id,
        completed_center,
    ):
        self.post_completion_candidate_promoted = False
        self.post_completion_candidate_track_id = None
        self.post_completion_candidate_rejected_reason = ""
        self.race_order_after_post_completion_fallback = []

        def reject(reason):
            self.post_completion_candidate_rejected_reason = str(reason)
            self.promotion_blocked_reason = str(reason)
            print(
                "[POST COMPLETION CANDIDATE FALLBACK] rejected "
                f"track_id={self.post_completion_candidate_track_id} reason={reason}"
            )
            return False

        if self.race_progression.predefined_order is not None:
            return reject("fallback_cannot_modify_predefined_race_order")

        # Refresh after marking the completed track so current-candidate selection
        # no longer treats the just-completed gate as eligible.
        self.refresh_race_order_from_memory()

        candidate_ids = [
            self.canonical_track_id(track_id)
            for track_id in self.current_gate_candidate_track_ids
            if self.canonical_track_id(track_id) is not None
        ]
        candidate_ids = list(dict.fromkeys(candidate_ids))
        if len(candidate_ids) != 1:
            return reject(f"fallback_candidate_count_not_one:{len(candidate_ids)}")

        track_id = candidate_ids[0]
        self.post_completion_candidate_track_id = track_id
        if self.selected_current_track_id is not None:
            selected_id = self.canonical_track_id(self.selected_current_track_id)
            if selected_id != track_id:
                return reject(f"fallback_selected_candidate_mismatch:{selected_id}!={track_id}")
        if track_id == self.canonical_track_id(completed_track_id):
            return reject("fallback_is_completed_track")
        if track_id in self.completed_track_ids_this_cycle:
            return reject("fallback_track_already_completed")
        if (
            self.race_gate_count is not None
            and track_id not in self.race_accepted_track_ids
            and len(self.race_accepted_track_ids) >= int(self.race_gate_count)
        ):
            return reject("fallback_race_gate_count_reached")

        tr = self.gate_memory.get_track_by_id(track_id)
        self._record_promotion_candidate_stability(tr)
        if tr is None:
            return reject("fallback_track_missing")
        if not bool(getattr(tr, "committed", False)):
            return reject("fallback_track_not_committed")

        center = np.asarray(tr.center, dtype=float).reshape(3)
        valid, reason = self.validate_planning_target(center)
        if not valid:
            return reject(f"fallback_invalid_target:{reason}")
        if self.is_near_completed_gate(center, radius=self.gate_memory.duplicate_merge_radius):
            return reject("fallback_duplicate_of_completed_gate")

        vehicle_pos = np.asarray(vehicle_pos, dtype=float).reshape(3)
        completed_center = np.asarray(completed_center, dtype=float).reshape(3)
        previous_gate_idx = self.current_gate_idx - 1
        expected_next = None
        if (
            self.gt_navigation_enabled()
            and
            0 <= previous_gate_idx < len(self.gt_gates)
            and 0 <= self.current_gate_idx < len(self.gt_gates)
        ):
            previous_gt = np.asarray(self.gt_gates[previous_gate_idx], dtype=float).reshape(3)
            expected_next = np.asarray(self.gt_gates[self.current_gate_idx], dtype=float).reshape(3)
            race_direction = expected_next - previous_gt
            self.post_completion_direction_source = "gt_expected_next"
            self.gt_behavior_dependency_used = True
            self.gt_behavior_dependency_reason = "post_completion_fallback_gt_expected_next"
        elif self.approach_vector is not None:
            race_direction = np.asarray(self.approach_vector, dtype=float).reshape(3)
            self.post_completion_direction_source = "approach_vector"
        else:
            race_direction = center - completed_center
            self.post_completion_direction_source = "completed_gate_to_candidate"

        direction_norm = float(np.linalg.norm(race_direction))
        if not np.isfinite(direction_norm) or direction_norm < 1e-6:
            return reject("fallback_race_direction_unavailable")
        race_direction = race_direction / direction_norm
        completed_progress = float(np.dot(center - completed_center, race_direction))
        vehicle_progress = float(np.dot(center - vehicle_pos, race_direction))
        if completed_progress <= 0.0:
            return reject(f"fallback_not_ahead_of_completed_gate:{completed_progress:.2f}")
        if vehicle_progress <= 0.0:
            return reject(f"fallback_behind_vehicle:{vehicle_progress:.2f}")
        if expected_next is not None:
            expected_error = float(np.linalg.norm(center - expected_next))
            if expected_error > float(self.max_gate_jump):
                return reject(
                    f"fallback_too_far_from_expected_next:{expected_error:.2f}"
                    f">{float(self.max_gate_jump):.2f}"
                )

        valid, reason = self.validate_candidate_target(center, vehicle_pos, track_id=track_id)
        if not valid:
            return reject(f"fallback_candidate_invalid:{reason}")

        if track_id not in self.race_accepted_track_ids:
            self.race_accepted_track_ids.append(track_id)
        order = [
            self.canonical_track_id(tid)
            for tid in self.race_progression.inferred_order
            if self.canonical_track_id(tid) != track_id
        ]
        insert_at = min(max(int(self.race_progression.cursor), 0), len(order))
        order.insert(insert_at, track_id)
        self.race_progression.inferred_order = order
        self.race_order_track_ids = list(order)
        self.race_admitted_track_ids = list(self.race_accepted_track_ids)
        self.race_order_after_post_completion_fallback = list(order)
        setattr(tr, "race_order_index", int(self.current_gate_idx))

        self.next_track_after_completion_id = track_id
        self.next_track_available_after_completion = True
        self.post_completion_candidate_promoted = True
        self.post_completion_candidate_rejected_reason = ""
        self.promotion_blocked_reason = ""
        print(
            "[POST COMPLETION CANDIDATE FALLBACK] accepted "
            f"track_id={track_id} center={center.tolist()} "
            f"hits={self.promotion_candidate_hits} "
            f"inliers={self.promotion_candidate_inliers} "
            f"outliers={self.promotion_candidate_outliers} "
            f"stability_blocker={self.promotion_candidate_stability_blocker or 'none'}"
        )
        return True

    def reset_crossing_debug(self):
        self.crossing_true_gate_center = np.full(3, np.nan, dtype=float)
        self.crossing_vehicle_position = np.full(3, np.nan, dtype=float)
        self.crossing_error = np.full(3, np.nan, dtype=float)
        self.crossing_lateral_error_xz = float("nan")

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

        self.target_clear_reason = reason

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
        self.active_target_center_at_plan = None
        self.active_target_latest_filtered_center = None
        self.active_target_shift_m = float("nan")
        self.active_target_shift_frames = 0
        self.active_target_shift_replan_triggered = False
        self.pending_active_target_correction = None
        self.approach_start_position = None
        pos = np.array([
            self.telemetry.pos["x"],
            self.telemetry.pos["y"],
            self.telemetry.pos["z"],
        ], dtype=float)
        self.perception_hold_position = pos.copy()
        telemetry_yaw = float(self.telemetry.rpy["yaw"])
        self.perception_hold_yaw = self.get_perception_yaw_hold_reference(telemetry_yaw)
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
        self._reset_pending_suffix_state(f"active_target_cleared:{reason}")
        print(f"[TARGET CLEAR] perception active target cleared reason={reason}")

    def _continue_existing_plan_after_completion(self, completed_track_id, next_track_id):
        """
        Advance active-target metadata to the next target already contained in
        the installed plan. This intentionally preserves the planner and its
        clock so the inter-gate segment after completion is not regenerated.
        """
        self.continued_existing_plan_after_completion = False
        self.continued_existing_plan_track_id = None
        self.continued_existing_plan_from_idx = -1
        self.continued_existing_plan_to_idx = -1
        self.gate_completion_replan_required = True

        if self.planner is None:
            return False
        if self.active_times is None:
            return False
        if completed_track_id is None or next_track_id is None:
            return False
        if len(self.active_target_track_ids) == 0 or len(self.active_target_gates) == 0:
            return False

        completed_id = self.canonical_track_id(completed_track_id)
        next_id = self.canonical_track_id(next_track_id)
        canonical_ids = [
            self.canonical_track_id(track_id)
            for track_id in self.active_target_track_ids
        ]

        if completed_id not in canonical_ids:
            return False

        completed_idx = canonical_ids.index(completed_id)
        next_idx = completed_idx + 1
        if next_idx >= len(canonical_ids):
            return False
        if canonical_ids[next_idx] != next_id:
            return False
        if next_idx >= len(self.active_target_gates):
            return False

        active_times = np.asarray(self.active_times, dtype=float).reshape(-1)
        if next_idx >= len(active_times):
            return False

        next_gate = np.asarray(self.active_target_gates[next_idx], dtype=float).reshape(3)
        if not np.all(np.isfinite(next_gate)):
            return False

        self.current_target_idx = next_idx
        self.active_target_track_id = next_id
        self.current_gate_pos = next_gate.copy()
        self.last_valid_target = next_gate.copy()
        self.active_target_center_at_plan = next_gate.copy()
        self.active_target_source = "continued_existing_plan_after_completion"
        self.next_valid_target_found = True
        self.next_target_installed_same_cycle = True
        self.target_retained_after_completion = True
        self.active_target_cleared = False
        self.no_active_target = False
        self.completed_gate_reference_blocked = False
        self.target_clear_reason = ""
        self.skipped_target_clear_after_completion = True
        self.post_completion_grace_until = 0.0
        self.post_completion_grace_active = False
        self.post_completion_grace_suppressed = True
        self.set_active_perception_target_geometry(
            next_gate,
            np.array([
                self.telemetry.pos["x"],
                self.telemetry.pos["y"],
                self.telemetry.pos["z"],
            ], dtype=float),
        )

        self.continued_existing_plan_after_completion = True
        self.continued_existing_plan_track_id = next_id
        self.continued_existing_plan_from_idx = completed_idx
        self.continued_existing_plan_to_idx = next_idx
        self.gate_completion_replan_required = False
        print(
            "[TARGET ADVANCE] continuing existing installed plan "
            f"completed_track_id={completed_id} next_track_id={next_id} "
            f"idx={completed_idx}->{next_idx}"
        )
        return True

    def _reset_pending_suffix_state(self, rejected_reason=""):
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

    def active_target_crossing_tau(self, target_idx):
        if self.active_times is None:
            return float("nan")
        times = np.asarray(self.active_times, dtype=float).reshape(-1)
        target_idx = int(target_idx)
        if target_idx < 0 or target_idx >= len(times):
            return float("nan")
        return float(np.sum(times[:target_idx + 1]))

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

    def _install_pending_suffix_after_completion(self, completed_track_id, next_track_id, pos):
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
        self.selected_next_gate_track_id = None
        self.selected_next_gate_stability_score = float("nan")
        self.selected_target_source = ""
        self.horizon_build_cursor = self.race_progression.cursor
        self.horizon_available_order = []
        self.horizon_selected_track_ids = []
        self.horizon_rejected_track_ids = []
        self.horizon_rejection_reason = ""
        self.planning_lookahead_track_ids = []
        self.planning_lookahead_source = ""
        self.planning_lookahead_used = False
        self.tentative_lookahead_used = False
        self.tentative_lookahead_track_ids = []
        self.tentative_lookahead_centers = ""
        self.tentative_lookahead_rejection_reason = ""
        self.append_lookahead_called = False
        self.append_lookahead_input_track_ids = []
        self.append_lookahead_selected_track_ids = []
        self.append_lookahead_selected_centers = ""
        self.append_lookahead_selected_types = ""
        self.horizon_track_decisions = {}
        self.planning_track_horizon_debug = ""
        self.planning_cycle_debug = ""

        if len(committed_tracks) == 0:
            self.planning_horizon_waypoint_types = "start"
            self._planning_target_waypoint_types = []
            return np.array([current_pos], dtype=float), [], []

        order = self.race_progression.order()
        self.horizon_available_order = list(order)
        target_tracks = []
        target_center_overrides = {}
        selected_track_ids = set()
        first_selected_order_idx = None

        if (
            self.race_progression.cursor < len(order)
            and self.selected_current_track_id is not None
            and self.canonical_track_id(order[self.race_progression.cursor])
            != self.canonical_track_id(self.selected_current_track_id)
        ):
            blocked_id = self.canonical_track_id(order[self.race_progression.cursor])
            self.horizon_rejected_track_ids.append(blocked_id)
            self.horizon_rejection_reason = "farther_future_track_cannot_be_hard_current"
            self.current_selection_rejection_reason = (
                f"track{blocked_id}:farther_future_track_cannot_be_hard_current;"
                f"selected_current_track={self.selected_current_track_id}"
            )
            self.horizon_track_decisions[blocked_id] = (
                "excluded:farther_future_track_cannot_be_hard_current"
            )
            self.planning_horizon_waypoint_types = "start"
            self._planning_target_waypoint_types = []
            return np.array([current_pos], dtype=float), [], []

        for order_idx in range(self.race_progression.cursor, len(order)):
            if len(target_tracks) >= max_gates_ahead:
                break

            track_id = order[order_idx]
            if track_id in selected_track_ids:
                continue

            tr = self.gate_memory.get_track_by_id(track_id)
            handoff = self.pending_lookahead_handoff
            is_handoff = bool(
                handoff is not None
                and self.canonical_track_id(handoff.get("track_id")) == self.canonical_track_id(track_id)
            )
            if tr is None or (not tr.committed and not is_handoff):
                self.horizon_rejected_track_ids.append(track_id)
                self.horizon_rejection_reason = "order_track_not_committed"
                self.horizon_track_decisions[track_id] = "excluded:order_track_not_committed"
                if is_handoff:
                    self.promotion_blocked_reason = "handoff_track_missing_or_not_committed"
                # Sequence integrity matters more than horizon length. With a
                # predefined race order, do not skip an unavailable next gate
                # and accidentally plan to a later gate.
                break
            if (
                self.use_lookahead_gate_filter
                and not getattr(tr, "is_stable", False)
                and track_id not in self.race_accepted_track_ids
                and not is_handoff
            ):
                self.last_perception_accepted = False
                self.last_perception_rejection_reason = "track_not_stable"
                self.active_target_admission_status = "pending_stability"
                self.horizon_rejected_track_ids.append(track_id)
                self.horizon_rejection_reason = "track_not_stable"
                self.horizon_track_decisions[track_id] = "excluded:track_not_stable"
                print(
                    f"[TARGET REJECT] reason=track_not_stable "
                    f"track_id={tr.id} blocked={getattr(tr, 'promotion_blocked_reason', '')}"
                )
                self.promotion_blocked_reason = (
                    getattr(tr, "promotion_blocked_reason", "")
                    or "track_not_stable"
                )
                break
            candidate_center = (
                np.asarray(handoff["center"], dtype=float).reshape(3)
                if is_handoff
                else np.asarray(tr.center, dtype=float).reshape(3)
            )
            valid, reason = self.validate_planning_target(candidate_center)
            if not valid:
                self.last_perception_accepted = False
                self.last_perception_rejection_reason = reason
                self.horizon_rejected_track_ids.append(track_id)
                self.horizon_rejection_reason = reason
                self.horizon_track_decisions[track_id] = f"excluded:{reason}"
                if is_handoff:
                    self.promotion_blocked_reason = f"planning_target_invalid:{reason}"
                print(f"[TARGET REJECT] reason={reason} track_id={tr.id} center={tr.center}")
                break
            if self.is_near_completed_gate(candidate_center):
                self.last_perception_accepted = False
                self.last_perception_rejection_reason = "already_completed_landmark"
                self.target_rejected_completed = True
                self.rejected_completed_this_lap = True
                self.horizon_rejected_track_ids.append(track_id)
                self.horizon_rejection_reason = "already_completed_landmark"
                self.horizon_track_decisions[track_id] = "excluded:already_completed_landmark"
                if is_handoff:
                    self.promotion_blocked_reason = "already_completed_landmark"
                print(f"[TARGET REJECT] reason=already_completed_landmark track_id={tr.id} center={tr.center}")
                continue
            if track_id not in self.race_accepted_track_ids and not is_handoff:
                self.last_perception_accepted = False
                self.last_perception_rejection_reason = "track_not_admitted_to_race"
                self.active_target_admission_status = "rejected"
                self.horizon_rejected_track_ids.append(track_id)
                self.horizon_rejection_reason = "track_not_admitted_to_race"
                self.horizon_track_decisions[track_id] = "excluded:track_not_admitted_to_race"
                print(f"[TARGET REJECT] reason=track_not_admitted_to_race track_id={tr.id} center={tr.center}")
                continue
            valid, reason = self.validate_candidate_target(candidate_center, current_pos, track_id=tr.id)
            if not valid:
                self.last_perception_accepted = False
                self.last_perception_rejection_reason = reason
                self.horizon_rejected_track_ids.append(track_id)
                self.horizon_rejection_reason = reason
                self.horizon_track_decisions[track_id] = f"excluded:{reason}"
                if is_handoff:
                    self.promotion_blocked_reason = f"candidate_target_invalid:{reason}"
                print(f"[TARGET REJECT] reason={reason} track_id={tr.id} center={tr.center}")
                continue
            target_tracks.append(tr)
            if is_handoff:
                target_center_overrides[track_id] = candidate_center.copy()
                self.promoted_lookahead_to_active = True
                self.promoted_track_id = track_id
                self.promoted_track_center = candidate_center.copy()
                self.promotion_blocked_reason = ""
                self.active_target_admission_status = "promoted_lookahead"
                print(
                    "[LOOKAHEAD HANDOFF] promoted "
                    f"track_id={track_id} center={candidate_center.tolist()} "
                    f"previous_ids={self.previous_horizon_track_ids} "
                    f"previous_types={self.previous_horizon_waypoint_types}"
                )
            self.horizon_track_decisions[track_id] = (
                "included:hard_current" if len(target_tracks) == 1 else "included:hard_stable"
            )
            selected_track_ids.add(track_id)
            self.valid_candidate_count += 1
            if len(target_tracks) == 1:
                self.selected_next_gate_track_id = tr.id
                self.selected_next_gate_stability_score = float(getattr(tr, "stability_score", np.nan))
                self.selected_target_source = "stable_track" if getattr(tr, "is_stable", False) else "race_admitted_track"
                print(
                    f"TRACK {tr.id} selected as next target: "
                    f"score={self.selected_next_gate_stability_score:.2f} "
                    f"source={self.selected_target_source}"
                )
            if first_selected_order_idx is None:
                first_selected_order_idx = order_idx

        target_gates = [
            target_center_overrides.get(tr.id, tr.center).copy()
            for tr in target_tracks
        ]
        target_track_ids = [tr.id for tr in target_tracks]
        target_waypoint_types = [
            "hard_current" if i == 0 else "hard_stable"
            for i in range(len(target_gates))
        ]

        if len(target_gates) == 0:
            self.planning_horizon_waypoint_types = "start"
            self._planning_target_waypoint_types = []
            return np.array([current_pos], dtype=float), [], []

        target_gates, target_track_ids, target_waypoint_types = self._append_planning_lookahead_targets(
            current_pos=np.asarray(current_pos, dtype=float).reshape(3),
            target_gates=target_gates,
            target_track_ids=target_track_ids,
            target_waypoint_types=target_waypoint_types,
            max_gates_ahead=max_gates_ahead,
            allow_raw_candidates=True,
        )
        self.update_lookahead_pipeline_debug()
        self.horizon_selected_track_ids = list(target_track_ids)
        self.planning_horizon_waypoint_types = " ".join(["start"] + target_waypoint_types)
        self._planning_target_waypoint_types = list(target_waypoint_types)

        if len(target_tracks) == 0:
            if len(target_gates) == 0:
                self.planning_horizon_waypoint_types = "start"
                self._planning_target_waypoint_types = []
                return np.array([current_pos], dtype=float), [], []

        if first_selected_order_idx is not None:
            self.race_progression.cursor = first_selected_order_idx

        if len(target_gates) > 0:
            self.last_valid_target = target_gates[0].copy()
            self.active_target_source = "memory_track"
            first_track = self.gate_memory.get_track_by_id(target_track_ids[0]) if target_track_ids[0] >= 0 else None
            if first_track is None or target_track_ids[0] not in self.race_accepted_track_ids:
                self.selected_target_source = "planning_lookahead"
            else:
                self.selected_target_source = (
                    "stable_track" if getattr(first_track, "is_stable", False) else "race_admitted_track"
                )
            self.active_target_track_id = target_track_ids[0]
            self.next_valid_target_found = True

        return np.vstack([current_pos] + target_gates), target_gates, target_track_ids

    def build_waypoint_horizon_from_gt(self, current_pos, max_gates_ahead=3):
        if not self.use_perception:
            max_gates_ahead = 1

        remaining_gates = self.gt_gates[self.current_gate_idx:self.current_gate_idx + max_gates_ahead]
        if len(remaining_gates) == 0:
            self.planning_horizon_waypoint_types = "start"
            self._planning_target_waypoint_types = []
            return np.array([current_pos], dtype=float), [], []

        target_gates = [np.asarray(g, dtype=float) for g in remaining_gates]
        target_track_ids = [-1] * len(target_gates)  # no memory-track IDs in GT mode
        target_waypoint_types = ["hard_current"] + ["hard_stable"] * max(0, len(target_gates) - 1)
        self.planning_horizon_waypoint_types = " ".join(["start"] + target_waypoint_types)
        self._planning_target_waypoint_types = list(target_waypoint_types)
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
            self.promoted_lookahead_to_active = False
            self.promoted_track_id = None
            self.promoted_track_center = np.full(3, np.nan, dtype=float)
            self.promotion_blocked_reason = ""
            self.promotion_normal_race_order_failed = False
            self.promotion_fallback_previous_horizon_used = False
            self.promotion_fallback_candidate_track_id = None
            self.promotion_fallback_candidate_center = np.full(3, np.nan, dtype=float)
            self.promotion_fallback_rejection_reason = ""
            self.previous_horizon_track_ids_at_completion = []
            self.previous_horizon_waypoint_types_at_completion = ""
            self.promoted_track_source = ""
            self.promotion_candidate_hits = 0
            self.promotion_candidate_inliers = 0
            self.promotion_candidate_outliers = 0
            self.promotion_candidate_camera_std = float("nan")
            self.promotion_candidate_center_std = float("nan")
            self.promotion_candidate_stability_blocker = ""
            self.post_completion_candidate_promoted = False
            self.post_completion_candidate_track_id = None
            self.post_completion_candidate_rejected_reason = ""
            self.race_order_after_post_completion_fallback = []
            self.post_completion_direction_source = ""
            self.continued_existing_plan_after_completion = False
            self.continued_existing_plan_track_id = None
            self.continued_existing_plan_from_idx = -1
            self.continued_existing_plan_to_idx = -1
            self.gate_completion_replan_required = True
            self.pending_suffix_installed = False
            self.gate_completion_triggered = False
            self.reset_crossing_debug()
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
            self.next_track_available_after_completion = False
            self.skipped_target_clear_after_completion = False
            self.next_track_after_completion_id = None
            self.next_target_installed_same_cycle = False
            self.target_clear_reason = ""
            self.post_completion_grace_suppressed = False

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
                self._reset_pending_suffix_state("no_active_target")
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
            self.previous_horizon_track_ids = list(self.planning_horizon_track_ids)
            self.previous_horizon_waypoint_types = self.planning_horizon_waypoint_types
            self.previous_horizon_track_ids_at_completion = list(
                self.previous_horizon_track_ids
            )
            self.previous_horizon_waypoint_types_at_completion = str(
                self.previous_horizon_waypoint_types
            )
            self.pending_lookahead_handoff = None
            if track_id in self.previous_horizon_track_ids:
                completed_horizon_idx = self.previous_horizon_track_ids.index(track_id)
                next_horizon_idx = completed_horizon_idx + 1
                if (
                    next_horizon_idx < len(self.previous_horizon_track_ids)
                    and next_horizon_idx < len(self.active_target_gates)
                ):
                    next_id = self.canonical_track_id(
                        self.previous_horizon_track_ids[next_horizon_idx]
                    )
                    waypoint_types = self.previous_horizon_waypoint_types.split()
                    next_type = (
                        waypoint_types[next_horizon_idx + 1]
                        if next_horizon_idx + 1 < len(waypoint_types)
                        else ""
                    )
                    if next_type in ("soft_committed_unstable", "soft_tentative", "hard_stable"):
                        self.pending_lookahead_handoff = {
                            "track_id": next_id,
                            "center": np.asarray(
                                self.active_target_gates[next_horizon_idx], dtype=float
                            ).reshape(3).copy(),
                            "waypoint_type": next_type,
                            "source": "race_order",
                        }
                        self._record_promotion_candidate_stability(
                            self.gate_memory.get_track_by_id(next_id)
                        )
            if 0 <= self.current_gate_idx < len(self.gt_gates):
                true_center = np.asarray(
                    self.gt_gates[self.current_gate_idx], dtype=float
                ).reshape(3)
                self.crossing_true_gate_center = true_center.copy()
                self.crossing_vehicle_position = pos.copy()
                self.crossing_error = pos - true_center
                self.crossing_lateral_error_xz = float(
                    np.linalg.norm(self.crossing_error[[0, 2]])
                )
                print(
                    "[GATE CROSSING DEBUG] "
                    f"gate_idx={self.current_gate_idx} track_id={track_id} "
                    f"vehicle={pos.tolist()} true={true_center.tolist()} "
                    f"error={self.crossing_error.tolist()} "
                    f"lateral_xz={self.crossing_lateral_error_xz:.3f}"
                )

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

            order_after_completion = self.race_progression.order()
            if self.race_progression.cursor < len(order_after_completion):
                self.next_track_after_completion_id = order_after_completion[self.race_progression.cursor]
                self.next_track_available_after_completion = True
            else:
                self.next_track_after_completion_id = None
                self.next_track_available_after_completion = False

            if self.pending_lookahead_handoff is not None:
                handoff_id = self.canonical_track_id(
                    self.pending_lookahead_handoff.get("track_id")
                )
                if self.next_track_after_completion_id is None:
                    self.promotion_normal_race_order_failed = True
                    self.promotion_blocked_reason = "next_race_order_track_missing"
                    fallback_used = self._try_previous_horizon_promotion_fallback(
                        vehicle_pos=pos,
                        completed_track_id=track_id,
                        completed_center=target,
                    )
                    if not fallback_used:
                        self._try_post_completion_current_candidate_fallback(
                            vehicle_pos=pos,
                            completed_track_id=track_id,
                            completed_center=target,
                        )
                elif handoff_id != self.canonical_track_id(self.next_track_after_completion_id):
                    self.promotion_normal_race_order_failed = True
                    self.promotion_blocked_reason = (
                        f"handoff_not_next_race_track:{handoff_id}!="
                        f"{self.next_track_after_completion_id}"
                    )
                    self.pending_lookahead_handoff = None
            elif self.next_track_after_completion_id is None:
                self.promotion_normal_race_order_failed = True
                self.promotion_blocked_reason = "next_race_order_track_missing"
                fallback_used = self._try_previous_horizon_promotion_fallback(
                    vehicle_pos=pos,
                    completed_track_id=track_id,
                    completed_center=target,
                )
                if not fallback_used:
                    self._try_post_completion_current_candidate_fallback(
                        vehicle_pos=pos,
                        completed_track_id=track_id,
                        completed_center=target,
                    )

            installed_next_target = False
            if self.next_track_available_after_completion:
                installed_next_target = self._continue_existing_plan_after_completion(
                    completed_track_id=track_id,
                    next_track_id=self.next_track_after_completion_id,
                )

            if not installed_next_target:
                installed_next_target = self._install_pending_suffix_after_completion(
                    completed_track_id=track_id,
                    next_track_id=self.next_track_after_completion_id,
                    pos=pos,
                )

            if not installed_next_target and self.next_track_available_after_completion:
                installed_next_target = self.path_plan(
                    replan_reason="gate_completed_next_track_available"
                )
            elif not installed_next_target and self.use_planning_lookahead_tracks:
                installed_next_target = self.path_plan(
                    replan_reason="gate_completed_planning_lookahead"
                )

            if installed_next_target and len(self.active_target_gates) > 0:
                if self.continued_existing_plan_after_completion:
                    self.promoted_track_source = "existing_installed_plan"
                elif self.pending_suffix_installed:
                    self.promoted_track_source = "pending_suffix"
                elif self.promoted_lookahead_to_active:
                    self.promoted_track_source = str(
                        (
                            self.pending_lookahead_handoff
                            or {}
                        ).get("source", "race_order")
                    )
                elif self.next_track_available_after_completion:
                    self.promoted_track_source = "race_order"
                if (
                    not self.continued_existing_plan_after_completion
                    and not self.pending_suffix_installed
                ):
                    self.active_target_source = (
                        "next_track_after_completion"
                        if self.next_track_available_after_completion
                        else "planning_lookahead_after_completion"
                    )
                self.next_target_installed_same_cycle = True
                self.skipped_target_clear_after_completion = True
                self.target_retained_after_completion = True
                self.active_target_cleared = False
                self.no_active_target = False
                self.completed_gate_reference_blocked = False
                self.post_completion_grace_until = 0.0
                self.post_completion_grace_active = False
                self.post_completion_grace_suppressed = True
                print(
                    "[TARGET ADVANCE] installed target after completion "
                    f"track_id={self.active_target_track_id} "
                    f"source={self.active_target_source}"
                )

            if not installed_next_target:
                # A completed perception target must not remain active while the
                # stack waits for a valid next gate. Replanning will install the
                # next target if one passes validation.
                self.clear_active_perception_target(reason="gate_completed")
            else:
                self.pending_lookahead_handoff = None
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

    def path_plan(self, gate_xyz=None, replan_reason="scheduled"):
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
        self.replan_reason = replan_reason
        self.last_target_z_clamped = False
        self.last_plan_started_at = plan_start
        self.last_plan_mode = "gt_single_gate" if not self.use_perception else "perception_horizon"
        self.last_plan_start_gate_idx = self.current_gate_idx if not self.use_perception else None
        self.last_plan_end_gate_idx = None
        self.replan_time = time.time()
        self.future_only_replan_preserved_active_segment = False
        self.future_only_replan_reason = ""
        self.replan_suppressed_reason = ""
        self.planning_horizon_track_ids = []
        self.planning_horizon_waypoint_count = 0
        self.planning_horizon_waypoints = ""
        self.planning_horizon_waypoint_types = "start"
        self._planning_target_waypoint_types = []
        self.gt_behavior_dependency_used = False
        self.gt_behavior_dependency_reason = ""
        self.terminal_extension_source = ""
        self.post_completion_direction_source = ""
        if self.gt_navigation_enabled():
            self.gt_behavior_dependency_used = True
            self.gt_behavior_dependency_reason = "explicit_gt_navigation"
        self.future_track_visible_before_completion = False
        self.future_track_blocked_reason = ""
        self.horizon_build_cursor = self.race_progression.cursor
        self.horizon_available_order = []
        self.horizon_selected_track_ids = []
        self.horizon_rejected_track_ids = []
        self.horizon_rejection_reason = ""
        self.planning_lookahead_track_ids = []
        self.planning_lookahead_source = ""
        self.planning_lookahead_used = False
        self.tentative_lookahead_used = False
        self.tentative_lookahead_track_ids = []
        self.tentative_lookahead_centers = ""
        self.tentative_lookahead_rejection_reason = ""
        self.append_lookahead_called = False
        self.append_lookahead_input_track_ids = []
        self.append_lookahead_selected_track_ids = []
        self.append_lookahead_selected_centers = ""
        self.append_lookahead_selected_types = ""
        self.passthrough_velocity_enabled = False
        self.passthrough_speed_used = float("nan")
        self.waypoint_velocity_log = np.full((3, 3), np.nan, dtype=float)
        self.internal_gate_velocity_nonzero = False
        self.terminal_velocity_mode = ""
        self.post_completion_horizon_has_future = False
        self.terminal_passthrough_extension_used = False
        self.terminal_passthrough_extension_point = np.full(3, np.nan, dtype=float)
        self.current_gate_treated_as_terminal = False
        self.first_segment_terminal_velocity_zero = False
        self.first_segment_min_v_ref_predicted = float("nan")
        self.reset_plan_geometric_validation_debug()

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
        previous_plan_state = {
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
        }

        def restore_previous_plan_state():
            self.active_target_gates = [
                g.copy() for g in previous_plan_state["active_target_gates"]
            ]
            self.active_target_track_ids = list(
                previous_plan_state["active_target_track_ids"]
            )
            self.current_target_idx = int(previous_plan_state["current_target_idx"])
            self.current_gate_pos = (
                None
                if previous_plan_state["current_gate_pos"] is None
                else previous_plan_state["current_gate_pos"].copy()
            )
            self.last_valid_target = (
                None
                if previous_plan_state["last_valid_target"] is None
                else previous_plan_state["last_valid_target"].copy()
            )
            self.active_target_track_id = previous_plan_state["active_target_track_id"]
            self.active_target_center_at_plan = (
                None
                if previous_plan_state["active_target_center_at_plan"] is None
                else previous_plan_state["active_target_center_at_plan"].copy()
            )

        waypoints, target_gates, target_track_ids = self.build_waypoint_horizon(
            pos,
            max_gates_ahead=1 if not self.use_perception else 3,
        )
        if self.use_perception:
            target_track_ids = [self.canonical_track_id(tid) for tid in target_track_ids]

        correction = self.pending_active_target_correction if self.use_perception else None
        if (
            correction is not None
            and len(target_gates) > 0
            and len(target_track_ids) > 0
            and self.canonical_track_id(target_track_ids[0]) == correction.get("track_id")
        ):
            corrected_target = np.asarray(correction["center"], dtype=float).reshape(3)
            print(
                "[ACTIVE TARGET SHIFT] applying smoothed correction "
                f"track_id={correction.get('track_id')} "
                f"old={target_gates[0]} corrected={corrected_target}"
            )
            target_gates[0] = corrected_target.copy()
            waypoints = np.vstack([pos] + target_gates)
        self.pending_active_target_correction = None

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
                self.finalize_planning_horizon_debug()
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
            self.finalize_planning_horizon_debug()
            self.last_plan_finished_at = time.time()
            self.last_plan_duration = self.last_plan_finished_at - plan_start
            return False

        if len(validated_target_gates) > 0:
            target_waypoint_types = list(self._planning_target_waypoint_types[:len(validated_target_gates)])
            target_gates = validated_target_gates
            target_track_ids = validated_target_track_ids

            if (
                self.use_perception
                and self.use_terminal_passthrough_extension
                and len(target_gates) == 1
                and not self.is_final_race_gate()
            ):
                extension = self.compute_terminal_passthrough_extension(pos, target_gates[0])
                if extension is not None and np.all(np.isfinite(extension)):
                    target_gates.append(np.asarray(extension, dtype=float).reshape(3))
                    target_track_ids.append(-2)
                    target_waypoint_types.append("terminal_extension")
                    self.terminal_passthrough_extension_used = True
                    self.terminal_passthrough_extension_point = np.asarray(
                        extension, dtype=float
                    ).reshape(3).copy()

            waypoints = np.vstack([pos] + target_gates)
            self.planning_horizon_waypoint_types = " ".join(["start"] + target_waypoint_types)
            self._planning_target_waypoint_types = list(target_waypoint_types)

        self.current_gate_treated_as_terminal = len(target_gates) == 1
        self.first_segment_terminal_velocity_zero = self.current_gate_treated_as_terminal
        if str(replan_reason).startswith("gate_completed") or replan_reason == "active_target_advanced":
            self.post_completion_horizon_has_future = len(target_gates) >= 2

        self.planning_horizon_track_ids = list(target_track_ids)
        self.planning_horizon_waypoint_count = int(len(waypoints))
        self.planning_horizon_waypoints = ";".join(
            f"{i}:{wp[0]:.2f},{wp[1]:.2f},{wp[2]:.2f}"
            for i, wp in enumerate(np.asarray(waypoints, dtype=float))
        )
        self.tentative_lookahead_centers = ";".join(
            f"{track_id}:{gate[0]:.2f},{gate[1]:.2f},{gate[2]:.2f}"
            for gate, track_id in zip(target_gates, target_track_ids)
            if track_id in self.tentative_lookahead_track_ids
        )
        self.tentative_lookahead_centers_at_plan = {
            int(track_id): np.asarray(gate, dtype=float).reshape(3).copy()
            for gate, track_id in zip(target_gates, target_track_ids)
            if track_id in self.tentative_lookahead_track_ids
        }

        previous_active_target = None
        if 0 <= self.current_target_idx < len(self.active_target_gates):
            previous_active_target = np.asarray(
                self.active_target_gates[self.current_target_idx], dtype=float
            ).reshape(3).copy()

        self.active_target_gates = [g.copy() for g in target_gates]
        self.active_target_track_ids = list(target_track_ids)
        self.current_target_idx = 0
        if self.use_perception and len(target_gates) > 0:
            self.last_valid_target = target_gates[0].copy()
            self.active_target_track_id = target_track_ids[0] if len(target_track_ids) > 0 else None
            self.next_valid_target_found = True
            self.active_target_cleared = False
            self.set_active_perception_target_geometry(target_gates[0], pos)
            self.active_target_center_at_plan = target_gates[0].copy()
            self.record_target_update_debug(
                previous=previous_active_target,
                new=target_gates[0],
                track_id=self.active_target_track_id,
                reason=replan_reason,
            )
            self.finalize_planning_horizon_debug()

        times_init = self.allocate_segment_times(
            waypoints,
            current_vel=vel,
            vmax=2.5,
            amax=2.0,
            T_min=1.0,
        )

        waypoint_velocities = None
        if self.use_perception:
            waypoint_velocities = self.compute_passthrough_waypoint_velocities(waypoints)
            self.passthrough_velocity_enabled = bool(self.use_passthrough_gate_velocities)
            self.passthrough_speed_used = (
                float(self.pass_through_speed)
                if self.passthrough_velocity_enabled
                else float("nan")
            )
            for i in range(min(3, len(waypoint_velocities))):
                self.waypoint_velocity_log[i] = waypoint_velocities[i]
            # Keep the current patch conservative: internal gates are pass-through,
            # but the final known horizon endpoint remains a stop point.
            v_end = np.zeros(3, dtype=float)
        else:
            # Preset waypoint mode should stop at the active gate. A nonzero
            # exit velocity is what lets the reference continue past the gate.
            v_end = np.zeros(3, dtype=float)
            self.terminal_velocity_mode = "zero_gt_endpoint"

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
        print("planning_horizon_track_ids:", self.planning_horizon_track_ids)
        print("passthrough_velocity_enabled:", self.passthrough_velocity_enabled)
        print("passthrough_speed_used:", self.passthrough_speed_used)
        print("waypoint velocities:", waypoint_velocities)
        print("internal_gate_velocity_nonzero:", self.internal_gate_velocity_nonzero)
        print("terminal_velocity_mode:", self.terminal_velocity_mode)

        # Fixed-time trajectory generation is a small linear solve. The SciPy
        # outer time optimizer caused second-scale offboard loop stalls in
        # perception mode, so online replans use the deterministic allocation.
        times_opt = times_init
        candidate_planner = MultiSegmentMinimumSnapPlanner()
        candidate_planner.update(
            waypoints=waypoints,
            times=times_opt,
            v_start=vel,
            v_end=v_end,
            a_start=np.zeros(3, dtype=float),
            a_end=np.zeros(3, dtype=float),
            j_start=np.zeros(3, dtype=float),
            j_end=np.zeros(3, dtype=float),
            waypoint_velocities=waypoint_velocities,
        )
        validation_ok, validation_debug = self.validate_minimum_snap_geometry(
            candidate_planner,
            waypoints,
        )
        selected_waypoints = np.asarray(waypoints, dtype=float).copy()
        selected_target_gates = [np.asarray(g, dtype=float).copy() for g in target_gates]
        selected_target_track_ids = list(target_track_ids)
        selected_target_waypoint_types = list(target_waypoint_types)
        selected_waypoint_velocities = waypoint_velocities
        self.reset_plan_geometric_validation_debug()

        if (
            not validation_ok
            and "z_corridor" in str(validation_debug.get("reason", ""))
            and np.isfinite(float(vel[2]))
            and float(vel[2]) < 0.0
        ):
            z_retry_vel = np.asarray(vel, dtype=float).reshape(3).copy()
            original_vz = float(z_retry_vel[2])
            z_retry_vel[2] = 0.0
            z_retry_planner = MultiSegmentMinimumSnapPlanner()
            z_retry_planner.update(
                waypoints=waypoints,
                times=times_opt,
                v_start=z_retry_vel,
                v_end=v_end,
                a_start=np.zeros(3, dtype=float),
                a_end=np.zeros(3, dtype=float),
                j_start=np.zeros(3, dtype=float),
                j_end=np.zeros(3, dtype=float),
                waypoint_velocities=waypoint_velocities,
            )
            z_retry_ok, z_retry_debug = self.validate_minimum_snap_geometry(
                z_retry_planner,
                waypoints,
            )
            if z_retry_ok:
                print(
                    "[PLAN GEOMETRY] z-corridor retry accepted "
                    f"v_start_z {original_vz:.3f}->0.000"
                )
                candidate_planner = z_retry_planner
                validation_ok = True
                validation_debug = z_retry_debug
                self.plan_z_fallback_reason = (
                    f"v_start_z_clamped_to_zero:{original_vz:.3f}"
                )
            else:
                print(
                    "[PLAN GEOMETRY] z-corridor retry failed "
                    f"v_start_z {original_vz:.3f}->0.000 "
                    f"reason={z_retry_debug['reason']} "
                    f"min_z={z_retry_debug['plan_min_z']:.3f}"
                )
                self.plan_z_fallback_reason = (
                    f"v_start_z_retry_failed:{z_retry_debug['reason']}"
                )

        if validation_ok:
            self.plan_min_z = float(validation_debug.get("plan_min_z", np.nan))
            self.plan_max_z = float(validation_debug.get("plan_max_z", np.nan))
            self.plan_z_undershoot_m = float(
                validation_debug.get("z_undershoot_m", 0.0)
            )
            self.plan_z_start_below_safe_min = bool(
                validation_debug.get("z_start_below_safe_min", False)
            )

        if not validation_ok:
            self.plan_geometric_validation_failed = True
            self.plan_validation_failed_segment_idx = int(validation_debug["segment_idx"])
            self.plan_max_backward_progress_m = float(
                validation_debug["max_backward_progress_m"]
            )
            self.plan_max_overshoot_m = float(validation_debug["max_overshoot_m"])
            self.plan_negative_progress_velocity_count = int(
                validation_debug["negative_progress_velocity_count"]
            )
            self.plan_validation_failure_reason = str(validation_debug["reason"])
            self.plan_z_corridor_failed = "z_corridor" in self.plan_validation_failure_reason
            self.plan_min_z = float(validation_debug.get("plan_min_z", np.nan))
            self.plan_max_z = float(validation_debug.get("plan_max_z", np.nan))
            self.plan_z_undershoot_m = float(
                validation_debug.get("z_undershoot_m", 0.0)
            )
            self.plan_z_start_below_safe_min = bool(
                validation_debug.get("z_start_below_safe_min", False)
            )
            print(
                "[PLAN GEOMETRY] candidate failed "
                f"segment={self.plan_validation_failed_segment_idx} "
                f"backward={self.plan_max_backward_progress_m:.3f}m "
                f"overshoot={self.plan_max_overshoot_m:.3f}m "
                f"neg_v_count={self.plan_negative_progress_velocity_count} "
                f"min_z={self.plan_min_z:.3f}m "
                f"z_undershoot={self.plan_z_undershoot_m:.3f}m "
                f"reason={self.plan_validation_failure_reason}"
            )

        if not validation_ok and len(target_gates) > 1:
            fallback_target_gates = [np.asarray(target_gates[0], dtype=float).copy()]
            fallback_target_track_ids = [target_track_ids[0]]
            fallback_target_waypoint_types = [
                target_waypoint_types[0] if len(target_waypoint_types) > 0 else "hard_current"
            ]
            fallback_terminal_extension_used = False
            fallback_terminal_extension_point = np.full(3, np.nan, dtype=float)

            if (
                self.use_perception
                and self.use_terminal_passthrough_extension
                and not self.is_final_race_gate()
            ):
                extension = self.compute_terminal_passthrough_extension(
                    pos,
                    fallback_target_gates[0],
                )
                if extension is not None and np.all(np.isfinite(extension)):
                    fallback_target_gates.append(
                        np.asarray(extension, dtype=float).reshape(3)
                    )
                    fallback_target_track_ids.append(-2)
                    fallback_target_waypoint_types.append("terminal_extension")
                    fallback_terminal_extension_used = True
                    fallback_terminal_extension_point = np.asarray(
                        extension,
                        dtype=float,
                    ).reshape(3).copy()

            fallback_waypoints = np.vstack([pos] + fallback_target_gates)
            fallback_times = self.allocate_segment_times(
                fallback_waypoints,
                current_vel=vel,
                vmax=2.5,
                amax=2.0,
                T_min=1.0,
            )
            fallback_waypoint_velocities = (
                self.compute_passthrough_waypoint_velocities(fallback_waypoints)
                if self.use_perception
                else None
            )
            fallback_planner = MultiSegmentMinimumSnapPlanner()
            fallback_planner.update(
                waypoints=fallback_waypoints,
                times=fallback_times,
                v_start=vel,
                v_end=v_end,
                a_start=np.zeros(3, dtype=float),
                a_end=np.zeros(3, dtype=float),
                j_start=np.zeros(3, dtype=float),
                j_end=np.zeros(3, dtype=float),
                waypoint_velocities=fallback_waypoint_velocities,
            )
            fallback_ok, fallback_debug = self.validate_minimum_snap_geometry(
                fallback_planner,
                fallback_waypoints,
            )
            if fallback_ok:
                print(
                    "[PLAN GEOMETRY] active-gate-only fallback accepted "
                    f"track_ids={fallback_target_track_ids} times={fallback_times}"
                )
                candidate_planner = fallback_planner
                times_opt = fallback_times
                selected_waypoints = fallback_waypoints.copy()
                selected_target_gates = [
                    np.asarray(g, dtype=float).copy()
                    for g in fallback_target_gates
                ]
                selected_target_track_ids = list(fallback_target_track_ids)
                selected_target_waypoint_types = list(fallback_target_waypoint_types)
                selected_waypoint_velocities = fallback_waypoint_velocities
                self.plan_geometric_fallback_used = True
                self.plan_min_z = float(fallback_debug.get("plan_min_z", np.nan))
                self.plan_max_z = float(fallback_debug.get("plan_max_z", np.nan))
                self.plan_z_undershoot_m = float(
                    fallback_debug.get("z_undershoot_m", 0.0)
                )
                self.plan_z_start_below_safe_min = bool(
                    fallback_debug.get("z_start_below_safe_min", False)
                )
                self.terminal_passthrough_extension_used = fallback_terminal_extension_used
                self.terminal_passthrough_extension_point = fallback_terminal_extension_point
            else:
                print(
                    "[PLAN GEOMETRY] active-gate-only fallback failed "
                    f"segment={fallback_debug['segment_idx']} "
                    f"backward={fallback_debug['max_backward_progress_m']:.3f}m "
                    f"overshoot={fallback_debug['max_overshoot_m']:.3f}m "
                    f"neg_v_count={fallback_debug['negative_progress_velocity_count']} "
                    f"reason={fallback_debug['reason']}"
                )
                self.plan_validation_failed_segment_idx = int(fallback_debug["segment_idx"])
                self.plan_max_backward_progress_m = float(
                    fallback_debug["max_backward_progress_m"]
                )
                self.plan_max_overshoot_m = float(fallback_debug["max_overshoot_m"])
                self.plan_negative_progress_velocity_count = int(
                    fallback_debug["negative_progress_velocity_count"]
                )
                self.plan_z_corridor_failed = "z_corridor" in str(
                    fallback_debug.get("reason", "")
                )
                self.plan_min_z = float(fallback_debug.get("plan_min_z", np.nan))
                self.plan_max_z = float(fallback_debug.get("plan_max_z", np.nan))
                self.plan_z_undershoot_m = float(
                    fallback_debug.get("z_undershoot_m", 0.0)
                )
                self.plan_z_start_below_safe_min = bool(
                    fallback_debug.get("z_start_below_safe_min", False)
                )
                self.plan_validation_failure_reason = (
                    f"fallback_failed:{fallback_debug['reason']}"
                )
                restore_previous_plan_state()
                self.last_plan_finished_at = time.time()
                self.last_plan_duration = self.last_plan_finished_at - plan_start
                return False
        elif not validation_ok:
            restore_previous_plan_state()
            self.last_plan_finished_at = time.time()
            self.last_plan_duration = self.last_plan_finished_at - plan_start
            return False

        self.planner = candidate_planner
        waypoints = selected_waypoints
        target_gates = selected_target_gates
        target_track_ids = selected_target_track_ids
        target_waypoint_types = selected_target_waypoint_types
        waypoint_velocities = selected_waypoint_velocities
        self.active_target_gates = [g.copy() for g in target_gates]
        self.active_target_track_ids = list(target_track_ids)
        self.current_target_idx = 0
        self.current_gate_treated_as_terminal = len(target_gates) == 1
        self.first_segment_terminal_velocity_zero = self.current_gate_treated_as_terminal
        self.post_completion_horizon_has_future = len(target_gates) >= 2
        self.waypoint_velocity_log = np.full((3, 3), np.nan, dtype=float)
        if waypoint_velocities is not None:
            for i in range(min(3, len(waypoint_velocities))):
                self.waypoint_velocity_log[i] = waypoint_velocities[i]
        self.planning_horizon_track_ids = list(target_track_ids)
        self.planning_horizon_waypoint_count = int(len(waypoints))
        self.planning_horizon_waypoints = ";".join(
            f"{i}:{wp[0]:.2f},{wp[1]:.2f},{wp[2]:.2f}"
            for i, wp in enumerate(np.asarray(waypoints, dtype=float))
        )
        self.planning_horizon_waypoint_types = " ".join(
            ["start"] + list(target_waypoint_types)
        )
        self._planning_target_waypoint_types = list(target_waypoint_types)
        if self.use_perception and len(target_gates) > 0:
            self.last_valid_target = target_gates[0].copy()
            self.active_target_track_id = target_track_ids[0] if len(target_track_ids) > 0 else None
            self.next_valid_target_found = True
            self.active_target_cleared = False
            self.set_active_perception_target_geometry(target_gates[0], pos)
            self.active_target_center_at_plan = target_gates[0].copy()

        if len(times_opt) > 0 and float(times_opt[0]) > 0.0:
            first_segment_speeds = []
            for tau in np.linspace(0.0, float(times_opt[0]), 101):
                _, velocity_ref, _ = self.planner.sample(float(tau))
                first_segment_speeds.append(float(np.linalg.norm(velocity_ref[:2])))
            self.first_segment_min_v_ref_predicted = min(first_segment_speeds)

        print("fixed segment times:", times_opt)
        print("fixed total horizon:", float(np.sum(times_opt)))
        print("plan_geometric_validation_failed:", self.plan_geometric_validation_failed)
        print("plan_geometric_fallback_used:", self.plan_geometric_fallback_used)
        print("plan_validation_failed_segment_idx:", self.plan_validation_failed_segment_idx)
        print("plan_max_backward_progress_m:", self.plan_max_backward_progress_m)
        print("plan_max_overshoot_m:", self.plan_max_overshoot_m)
        print("plan_negative_progress_velocity_count:", self.plan_negative_progress_velocity_count)
        print("plan_validation_failure_reason:", self.plan_validation_failure_reason)
        print("plan_z_corridor_failed:", self.plan_z_corridor_failed)
        print("plan_min_z:", self.plan_min_z)
        print("plan_max_z:", self.plan_max_z)
        print("plan_z_undershoot_m:", self.plan_z_undershoot_m)
        print("plan_z_fallback_reason:", self.plan_z_fallback_reason)
        print("plan_z_start_below_safe_min:", self.plan_z_start_below_safe_min)
        print("post_completion_horizon_has_future:", self.post_completion_horizon_has_future)
        print("terminal_passthrough_extension_used:", self.terminal_passthrough_extension_used)
        print("terminal_passthrough_extension_point:", self.terminal_passthrough_extension_point)
        print("current_gate_treated_as_terminal:", self.current_gate_treated_as_terminal)
        print("first_segment_terminal_velocity_zero:", self.first_segment_terminal_velocity_zero)
        print("first_segment_min_v_ref_predicted:", self.first_segment_min_v_ref_predicted)

        self.current_gate_pos = waypoints[1].copy()
        self.active_waypoints = waypoints.copy()
        self.active_times = np.asarray(times_opt, dtype=float).copy()
        self.trajectory_start_time = time.time()
        self.previous_sample_tau_used = 0.0
        self.previous_sample_tau_plan_id = None
        self.record_installed_plan_for_export(
            plan_source="normal_path_plan",
            replan_reason=replan_reason,
        )

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

    def check_active_target_shift_correction(self):
        self.active_target_shift_replan_triggered = False
        self.active_target_shift_suppressed = False
        self.distance_to_active_target_at_shift = float("nan")
        self.target_shift_xy = float("nan")
        self.target_shift_z = float("nan")
        self.shift_replan_allowed = False
        self.shift_replan_suppressed_reason = ""
        self.near_gate_suppression_overridden = False
        self.near_gate_override_reason = ""
        self.committed_target_error_to_filter = float("nan")
        self.committed_target_xy_error_to_filter = float("nan")
        self.committed_target_z_error_to_filter = float("nan")
        self.committed_target_error_to_GT = float("nan")
        self.latest_filter_error_to_GT = float("nan")
        self.target_update_improvement_m = float("nan")
        self.target_update_alpha_used = float("nan")
        self.target_update_aggressive_correction_used = False

        if not self.use_perception:
            return False
        if len(self.active_target_gates) == 0 or len(self.active_target_track_ids) == 0:
            self.active_target_shift_m = float("nan")
            self.active_target_shift_frames = 0
            self.active_target_latest_filtered_center = None
            return False
        if not (0 <= self.current_target_idx < len(self.active_target_track_ids)):
            return False
        if self.gate_completion_triggered or self.gate_plane_crossed:
            return False

        track_id = self.canonical_track_id(self.active_target_track_ids[self.current_target_idx])
        if track_id is None or track_id < 0:
            return False

        track = self.gate_memory.get_track_by_id(track_id)
        if track is None:
            return False

        latest = getattr(track, "filtered_center_world", None)
        if latest is None:
            latest = getattr(track, "center", None)
        if latest is None:
            return False

        latest = np.asarray(latest, dtype=float).reshape(3)
        if not np.all(np.isfinite(latest)):
            return False

        if self.active_target_center_at_plan is None:
            if 0 <= self.current_target_idx < len(self.active_target_gates):
                self.active_target_center_at_plan = np.asarray(
                    self.active_target_gates[self.current_target_idx],
                    dtype=float,
                ).reshape(3).copy()
            else:
                return False

        planned = np.asarray(self.active_target_center_at_plan, dtype=float).reshape(3)
        self.active_target_latest_filtered_center = latest.copy()
        raw_shift = latest - planned
        self.active_target_shift_m = float(np.linalg.norm(raw_shift))
        self.committed_target_error_to_filter = self.active_target_shift_m
        self.committed_target_xy_error_to_filter = float(
            np.linalg.norm(raw_shift[:2])
        )
        self.committed_target_z_error_to_filter = float(abs(raw_shift[2]))

        current_pos = np.array([
            self.telemetry.pos["x"],
            self.telemetry.pos["y"],
            self.telemetry.pos["z"],
        ], dtype=float)
        has_future_horizon_gate = self.current_target_idx + 1 < len(
            self.active_target_gates
        )
        has_future_race_gate = (
            self.race_gate_count is not None
            and self.current_gate_idx < self.race_gate_count - 1
        )
        active_is_internal_passthrough = bool(
            has_future_horizon_gate or has_future_race_gate
        )

        if 0 <= self.current_gate_idx < len(self.gt_gates):
            expected_gt = np.asarray(
                self.gt_gates[self.current_gate_idx], dtype=float
            ).reshape(3)
            if np.all(np.isfinite(expected_gt)):
                self.committed_target_error_to_GT = float(
                    np.linalg.norm(planned - expected_gt)
                )
                self.latest_filter_error_to_GT = float(
                    np.linalg.norm(latest - expected_gt)
                )
                self.target_update_improvement_m = (
                    self.committed_target_error_to_GT
                    - self.latest_filter_error_to_GT
                )
                if self.use_perception:
                    self.active_shift_gt_debug_only = True

        major_gt_improvement = bool(
            np.isfinite(self.target_update_improvement_m)
            and self.target_update_improvement_m > 0.30
        )
        major_gt_improvement_behavior = bool(
            self.gt_navigation_enabled() and major_gt_improvement
        )
        if active_is_internal_passthrough:
            latest_observation = (
                track.obs_history[-1]
                if len(getattr(track, "obs_history", [])) > 0
                else None
            )
            recent_observation = bool(
                latest_observation is not None
                and time.time() - float(latest_observation.timestamp)
                <= self.gate_memory.stale_time
            )
            latest_observation_accepted = bool(
                recent_observation and not bool(latest_observation.is_outlier)
            )
            latest_innovation = float("inf")
            if latest_observation is not None:
                observation_center = np.asarray(
                    latest_observation.center_world, dtype=float
                ).reshape(3)
                if np.all(np.isfinite(observation_center)):
                    latest_innovation = float(
                        np.linalg.norm(observation_center - latest)
                    )
            enough_history = bool(
                getattr(track, "committed", False)
                and int(getattr(track, "hits", 0))
                >= self.planning_lookahead_min_hits
                and int(getattr(track, "inlier_count", 0))
                >= self.gate_memory.min_hits_for_stable
            )
            reasonable_innovation = bool(
                np.isfinite(latest_innovation)
                and latest_innovation
                <= self.gate_memory.max_committed_match_distance
            )
            latest_valid, latest_invalid_reason = self.validate_candidate_target(
                latest, current_pos, track_id=track_id
            )
            if not enough_history:
                self.shift_replan_suppressed_reason = (
                    "active_filter_insufficient_history"
                )
                return False
            if not latest_observation_accepted:
                self.shift_replan_suppressed_reason = (
                    "active_filter_no_recent_accepted_detection"
                )
                return False
            if not reasonable_innovation:
                self.shift_replan_suppressed_reason = (
                    f"active_filter_innovation_high:{latest_innovation:.3f}"
                )
                return False
            if not latest_valid:
                self.shift_replan_suppressed_reason = (
                    f"active_filter_target_invalid:{latest_invalid_reason}"
                )
                return False

            small_noise_shift = bool(
                self.committed_target_xy_error_to_filter < 0.15
                and self.committed_target_z_error_to_filter < 0.10
            )
            if small_noise_shift and not major_gt_improvement_behavior:
                self.active_target_shift_frames = 0
                self.shift_replan_suppressed_reason = "active_filter_small_noise"
                return False

            material_filter_correction = bool(
                self.active_target_shift_m > self.active_target_shift_threshold_m
                or self.committed_target_xy_error_to_filter > 0.35
                or self.committed_target_z_error_to_filter > 0.25
                or major_gt_improvement_behavior
            )
        else:
            material_filter_correction = bool(
                self.active_target_shift_m > self.active_target_shift_threshold_m
            )
        if not material_filter_correction:
            self.active_target_shift_frames = 0
            return False

        self.active_target_shift_frames += 1
        if self.active_target_shift_frames < self.active_target_shift_required_frames:
            return False

        if time.time() - self.replan_time <= self.active_target_shift_replan_min_interval_s:
            return False

        if self.is_near_completed_gate(planned):
            return False

        correction_alpha = 0.3
        if (
            active_is_internal_passthrough
            and self.gt_navigation_enabled()
            and np.isfinite(self.target_update_improvement_m)
        ):
            if self.target_update_improvement_m > 0.75:
                correction_alpha = 0.7
            elif self.target_update_improvement_m > 0.40:
                correction_alpha = 0.5
        self.target_update_alpha_used = correction_alpha
        self.target_update_aggressive_correction_used = correction_alpha > 0.3
        corrected_target = (
            (1.0 - correction_alpha) * planned
            + correction_alpha * latest
        )
        proposed_shift = corrected_target - planned
        self.target_shift_xy = float(np.linalg.norm(proposed_shift[:2]))
        self.target_shift_z = float(abs(proposed_shift[2]))
        self.distance_to_active_target_at_shift = float(
            np.linalg.norm(planned - current_pos)
        )

        planned_valid, _ = self.validate_planning_target(planned)
        planned_target_stale = bool(
            time.time() - self.replan_time > self.gate_memory.stale_time
        )

        if self.approach_vector is not None and np.all(np.isfinite(self.approach_vector)):
            approach = np.asarray(self.approach_vector, dtype=float).reshape(3)
            progress_shift = abs(float(np.dot(raw_shift, approach)))
            lateral_shift = float(
                np.linalg.norm(raw_shift - np.dot(raw_shift, approach) * approach)
            )
        else:
            progress_shift = float(np.linalg.norm(raw_shift[:2]))
            lateral_shift = float(abs(raw_shift[2]))
        crossing_change_large = bool(
            progress_shift > self.current_gate_freeze_progress_margin
            or lateral_shift > self.current_gate_freeze_progress_margin
        )
        proposed_horizontal_shift_large = bool(
            self.target_shift_xy > self.max_current_gate_target_shift_near_gate
        )
        very_near_gate = bool(self.distance_to_active_target_at_shift <= 1.0)
        very_near_small_applied_shift = bool(
            very_near_gate
            and self.target_shift_xy < 0.25
            and self.target_shift_z < 0.15
            and not crossing_change_large
        )
        override_reasons = []
        if proposed_horizontal_shift_large:
            override_reasons.append("large_applied_xy_shift")
        if self.target_shift_z > 0.25:
            override_reasons.append("large_applied_z_shift")
        if major_gt_improvement_behavior and self.target_shift_xy > 0.20:
            override_reasons.append("gt_improvement_with_material_applied_shift")
        if crossing_change_large:
            override_reasons.append("crossing_change_large")
        near_gate_override_requested = bool(override_reasons)

        suppress_near_gate_shift = bool(
            self.freeze_current_gate_target_near_gate
            and active_is_internal_passthrough
            and self.distance_to_active_target_at_shift <= self.current_gate_freeze_distance
            and not proposed_horizontal_shift_large
            and not crossing_change_large
            and planned_valid
            and not planned_target_stale
            and (not near_gate_override_requested or very_near_small_applied_shift)
        )
        if (
            self.freeze_current_gate_target_near_gate
            and active_is_internal_passthrough
            and self.distance_to_active_target_at_shift
            <= self.current_gate_freeze_distance
            and near_gate_override_requested
        ):
            self.near_gate_suppression_overridden = True
            self.near_gate_override_reason = "|".join(override_reasons)
        if suppress_near_gate_shift:
            self.pending_active_target_correction = None
            self.active_target_shift_suppressed = True
            self.shift_replan_allowed = False
            self.shift_replan_suppressed_reason = (
                "internal_passthrough_near_gate_small_target_correction"
            )
            self.active_target_shift_frames = 0
            print(
                "[ACTIVE TARGET SHIFT] suppressed "
                f"track_id={track_id} distance={self.distance_to_active_target_at_shift:.3f}m "
                f"shift_xy={self.target_shift_xy:.3f}m shift_z={self.target_shift_z:.3f}m "
                f"progress_shift={progress_shift:.3f}m lateral_shift={lateral_shift:.3f}m "
                f"reason={self.shift_replan_suppressed_reason}"
            )
            return False

        self.shift_replan_allowed = True
        self.pending_active_target_correction = {
            "track_id": track_id,
            "center": corrected_target.copy(),
        }
        self.active_target_shift_replan_triggered = True
        print(
            "[ACTIVE TARGET SHIFT] replan requested "
            f"track_id={track_id} shift={self.active_target_shift_m:.3f}m "
            f"frames={self.active_target_shift_frames} "
            f"planned={planned} latest={latest} corrected={corrected_target} "
            f"alpha={correction_alpha:.2f} "
            f"improvement={self.target_update_improvement_m:.3f}m "
            f"near_override={self.near_gate_suppression_overridden} "
            f"override_reason={self.near_gate_override_reason or 'none'}"
        )
        return True

    def check_tentative_lookahead_shift_replan(self):
        self.tentative_lookahead_shift_replan_triggered = False
        self.tentative_lookahead_shift_m = float("nan")
        self.tentative_lookahead_shift_track_id = None

        if not self.use_perception or not self.use_tentative_lookahead_spline:
            return False
        if self.gate_completion_triggered or self.gate_plane_crossed:
            return False
        if time.time() - self.replan_time <= self.tentative_lookahead_replan_min_interval_s:
            return False
        if not self.tentative_lookahead_centers_at_plan:
            return False

        current_track_id = None
        if 0 <= self.current_target_idx < len(self.active_target_track_ids):
            current_track_id = self.canonical_track_id(self.active_target_track_ids[self.current_target_idx])

        best_shift = 0.0
        best_track_id = None
        for track_id, planned_center in self.tentative_lookahead_centers_at_plan.items():
            track_id = self.canonical_track_id(track_id)
            if track_id is None or track_id == current_track_id:
                continue
            tr = self.gate_memory.get_track_by_id(track_id)
            if tr is None:
                continue
            latest = getattr(tr, "filtered_center_world", None)
            if latest is None:
                latest = getattr(tr, "center", None)
            if latest is None:
                continue
            latest = np.asarray(latest, dtype=float).reshape(3)
            if not np.all(np.isfinite(latest)):
                continue
            planned_center = np.asarray(planned_center, dtype=float).reshape(3)
            shift = float(np.linalg.norm(latest - planned_center))
            if shift > best_shift:
                best_shift = shift
                best_track_id = track_id

        self.tentative_lookahead_shift_m = best_shift if best_track_id is not None else float("nan")
        self.tentative_lookahead_shift_track_id = best_track_id
        if best_track_id is None:
            return False
        threshold = float(self.tentative_lookahead_shift_replan_threshold)
        if (
            self.suppress_minor_tentative_lookahead_replans
            and len(self.active_target_gates) >= 2
        ):
            threshold = max(threshold, float(self.tentative_lookahead_replan_min_shift))
        if best_shift <= threshold:
            if self.suppress_minor_tentative_lookahead_replans:
                self.tentative_lookahead_replan_suppressed = True
                self.tentative_lookahead_replan_suppression_reason = (
                    "internal_passthrough_horizon_minor_lookahead_shift"
                )
                self.horizon_material_change_m = best_shift
                print(
                    "[TENTATIVE LOOKAHEAD SHIFT] replan suppressed "
                    f"track_id={best_track_id} shift={best_shift:.3f}m "
                    f"threshold={threshold:.3f}m "
                    f"reason={self.tentative_lookahead_replan_suppression_reason}"
                )
            return False

        self.tentative_lookahead_shift_replan_triggered = True
        print(
            "[TENTATIVE LOOKAHEAD SHIFT] replan requested "
            f"track_id={best_track_id} shift={best_shift:.3f}m"
        )
        return True

    # -------------------------------------------------------------------------
    # Control
    # -------------------------------------------------------------------------

    def get_perception_yaw_hold_reference(self, fallback_yaw):
        """
        Return the yaw that perception loss should hold.

        Losing a target should freeze the last commanded yaw reference instead
        of reseeding the hold yaw from telemetry at the loss instant. Reseeding
        from telemetry can create an artificial yaw step when perception drops
        out or when the active target is cleared.
        """
        fallback_yaw = float(fallback_yaw)
        if not np.isfinite(fallback_yaw):
            fallback_yaw = 0.0

        if self.has_commanded_yaw_reference:
            for yaw_ref in (self.ref_yaw, self.previous_yaw_cmd, self.last_desired_yaw):
                if yaw_ref is not None and np.isfinite(yaw_ref):
                    return float(yaw_ref)

            if np.isfinite(self.perception_hold_yaw):
                return float(self.perception_hold_yaw)

        return fallback_yaw

    def seed_yaw_hold(self, current_yaw_rad, reason=""):
        current_yaw_rad = float(current_yaw_rad)
        if not np.isfinite(current_yaw_rad):
            return

        self.perception_hold_yaw = current_yaw_rad
        self.ref_yaw = current_yaw_rad
        self.last_desired_yaw = current_yaw_rad
        self.previous_yaw_cmd = current_yaw_rad
        self.previous_yaw_cmd_log = current_yaw_rad
        self.raw_yaw_cmd = current_yaw_rad
        self.yaw_cmd_after_unwrap = current_yaw_rad
        self.yaw_hold_value = current_yaw_rad
        self.has_commanded_yaw_reference = True
        self.yaw_target_source = f"seed_yaw_hold:{reason}"
        print(f"[YAW HOLD SEED] reason={reason} yaw_deg={math.degrees(current_yaw_rad):.2f}")

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
            self.has_commanded_yaw_reference = True
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
        self.has_commanded_yaw_reference = True
        return limited

    def record_tracker_control_debug(self, state, dbg, roll_cmd, pitch_cmd, thrust_cmd):
        def vec(name):
            value = None if dbg is None else dbg.get(name)
            if value is None:
                return np.full(3, np.nan, dtype=float)
            arr = np.asarray(value, dtype=float).reshape(-1)
            if arr.size < 3:
                out = np.full(3, np.nan, dtype=float)
                out[:arr.size] = arr
                return out
            return arr[:3].copy()

        raw_velocity = np.array([
            self.telemetry.vel["vx"],
            self.telemetry.vel["vy"],
            self.telemetry.vel["vz"],
        ], dtype=float)
        velocity_input = np.asarray(state.vel, dtype=float).reshape(3)

        self.tracker_velocity_input = velocity_input.copy()
        self.tracker_velocity_was_sanitized = bool(
            not np.all(np.isfinite(raw_velocity))
            or not np.allclose(
                np.nan_to_num(raw_velocity, nan=0.0, posinf=0.0, neginf=0.0),
                velocity_input,
                equal_nan=False,
            )
        )
        self.tracker_e_p = vec("e_p")
        self.tracker_e_v = vec("e_v")
        self.tracker_a_ref = vec("a_ref")
        self.tracker_a_fb = vec("a_fb")
        self.tracker_a_cmd_raw = vec("a_cmd_raw_no_g")
        self.tracker_a_cmd_limited = vec("a_cmd_no_g")
        self.thrust_raw_before_clamp = float(
            dbg.get("thrust_raw_before_clamp", np.nan) if dbg is not None else np.nan
        )
        self.thrust_cmd_after_clamp = float(
            dbg.get("thrust_cmd_after_clamp", thrust_cmd) if dbg is not None else thrust_cmd
        )
        self.thrust_limited = bool(
            dbg.get("thrust_limited", False) if dbg is not None else False
        )
        self.hover_thrust = float(
            dbg.get("hover_thrust", self.tracker.thrust_hover) if dbg is not None else self.tracker.thrust_hover
        )
        self.vertical_thrust_after_tilt = float(
            thrust_cmd * math.cos(float(roll_cmd)) * math.cos(float(pitch_cmd))
        )

    def hold_no_target_control(self, state, current_yaw_rad):
        """
        Perception-only safety fallback for "no valid next gate".

        It publishes finite references and uses the normal tracker for altitude
        feedback and horizontal velocity damping without anchoring XY to the
        completed gate/completion point behind the vehicle.
        """
        if self.perception_hold_position is None:
            self.perception_hold_position = state.pos.copy()
            self.seed_yaw_hold(current_yaw_rad, reason="first_no_target_hold")
        self.perception_hold_yaw = self.get_perception_yaw_hold_reference(current_yaw_rad)

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
        self.record_tracker_control_debug(state, dbg, roll_cmd, pitch_cmd, thrust_cmd)

        print("No active perception target; damping velocity and holding altitude/yaw.")
        print("state pos:", state.pos)
        print("hold ref pos:", ref.pos)
        print("current yaw_deg:", math.degrees(current_yaw_rad))
        print("hold yaw_des:", self.perception_hold_yaw)
        print("hold yaw_des_deg:", math.degrees(self.perception_hold_yaw))
        print("previous yaw_cmd_deg:", math.degrees(self.previous_yaw_cmd) if self.previous_yaw_cmd is not None else float("nan"))
        print("has_commanded_yaw_reference:", self.has_commanded_yaw_reference)
        print("cmd roll pitch yaw thrust:", roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd)

        return roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd

    def attitude_control(self):
        current_yaw_rad = float(self.telemetry.rpy["yaw"])

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

        wall_tau = time.time() - self.trajectory_start_time
        self.time_elapsed = self.compute_reference_sample_tau(wall_tau, state)
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

        if self.use_perception and not self.last_perception_accepted:
            desired_yaw = current_yaw_rad
            self.perception_hold_yaw = desired_yaw
            self.yaw_target_source = "perception_lost_hold_current_yaw"
        elif self.use_perception and self.current_target_gate is not None:
            if self.is_near_completed_gate(self.current_target_gate):
                self.target_retained_after_completion = True
                self.clear_active_perception_target(reason="completed_target_in_control")
                print("Completed target reached control path; switching to hold.")
                return self.hold_no_target_control(state, current_yaw_rad)
            to_target = np.asarray(self.current_target_gate[:2], dtype=float) - state.pos[:2]
            if np.linalg.norm(to_target) > 1e-3:
                desired_yaw = np.arctan2(to_target[1], to_target[0])
                self.yaw_target_source = "active_target_camera_axis"
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
        self.record_tracker_control_debug(state, dbg, roll_cmd, pitch_cmd, thrust_cmd)

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

        # tracker yaw_cmd is in radians; telemetry RPY is stored in radians.
        api.telemetry.rpy["yaw"] = float(yaw)

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
