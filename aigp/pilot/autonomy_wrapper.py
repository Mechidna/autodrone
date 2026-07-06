from __future__ import annotations

import json
import math
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autonomy_core.core.competition_config import VADR_TS_002
from autonomy_core.controller.attitude_controller3 import (
    RPGHighLevelTracker,
    Reference,
    State,
)
from autonomy_core.planning.minimum_snap_planner_multi_time_optimized import (
    MultiSegmentMinimumSnapPlanner,
)
from autonomy_core.planning.trajectory_manager import allocate_segment_times
from autonomy_core.perception.gate_memory import GateMemory
from adaptive_hover_thrust import AdaptiveHoverThrust
from estimator_landmark_map import EstimatorLandmarkMap
from gate_pass_geometry import (
    GatePlanePassResult,
    check_gate_plane_pass,
    unit_vector_from_to,
)
from hover_acquisition import HoverAcquisition
from lateral_response_calibration import LateralResponseCalibration
from runtime_config import load_runtime_config
from target_manager import TargetManager
from thrust_scale_calibration import ThrustScaleCalibration
from vehicle_state_estimator import VehicleStateEstimator


GATE_MODEL_RE = re.compile(r"^racing_gate_(\d+)$")


@dataclass(frozen=True)
class AutonomyCommandRad:
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    thrust: float


class PyAIPilotAutonomyAPI:
    """
    Clean local autonomy entry point for pilot.

    This uses the same core modules as autonomy_api6 for the current
    ground-truth-gate path:
      MultiSegmentMinimumSnapPlanner -> RPGHighLevelTracker

    Perception inputs are hard-coded here but not used until the adapter starts
    feeding detections instead of gate centers.
    """

    def __init__(
        self,
        use_perception: bool | None = None,
        race_gate_count: int | None = None,
        pass_radius_m: float | None = None,
        config=None,
    ):
        self.config = config if config is not None else load_runtime_config()
        self.use_perception = (
            self.config.runtime.use_perception
            if use_perception is None
            else bool(use_perception)
        )
        self.pass_radius_m = (
            float(self.config.race.pass_radius_m)
            if pass_radius_m is None
            else float(pass_radius_m)
        )
        spec_gate_lateral_radius_m = float(VADR_TS_002.gate_inner_half_extent_m)
        configured_lateral_radius_m = float(self.config.race.pass_lateral_radius_m)
        self.gate_pass_lateral_radius_m = (
            spec_gate_lateral_radius_m
            if configured_lateral_radius_m <= 0.0
            else min(configured_lateral_radius_m, spec_gate_lateral_radius_m)
        )
        self.gate_plane_tolerance_m = max(
            0.0,
            float(self.config.race.pass_plane_tolerance_m),
        )
        self.near_plane_pass_enabled = bool(
            self.config.race.near_plane_pass_enabled
        )
        self.near_plane_pass_back_tolerance_m = max(
            0.0,
            float(self.config.race.near_plane_pass_back_tolerance_m),
        )
        self.near_plane_pass_forward_tolerance_m = max(
            0.0,
            float(self.config.race.near_plane_pass_forward_tolerance_m),
        )
        self.gate_pass_through_m = max(
            float(VADR_TS_002.gate_depth_m),
            float(self.config.race.pass_through_m),
            0.0,
        )
        self.camera_matrix = np.asarray(self.config.camera.matrix, dtype=float).reshape(3, 3)
        self.dist_coeffs = np.asarray(self.config.camera.dist_coeffs, dtype=float).reshape(-1)
        self.race_gate_count = (
            self.config.race.gate_count
            if race_gate_count is None
            else int(race_gate_count)
        )
        if self.race_gate_count is not None:
            self.race_gate_count = int(self.race_gate_count)

        self.gate_centers_neu = []
        self.gate_track_ids = []
        self._candidate_gate_track_ids = []
        self.gate_source_mode = str(self.config.gate_source.mode).lower()
        self.ground_truth_gate_positions_neu = [
            np.asarray(gate, dtype=float).reshape(3).copy()
            for gate in self.config.gate_source.known_gate_positions_neu
        ]
        self.ground_truth_gate_track_ids = [
            -(idx + 1) for idx in range(len(self.ground_truth_gate_positions_neu))
        ]
        self._last_gate_signature = None
        self._last_ground_truth_gate_print_signature = None
        self.active_track_count = 0

        self.current_gate_idx = 0
        self.current_gate_pos = None
        self.active_target_track_id = None
        self.active_target_center_at_plan = None
        self.active_target_latest_filtered_center = None
        self.last_active_target_center = None
        self.active_target_lost_time = None
        self.active_target_lost_grace_s = float(self.config.race.active_target_lost_grace_s)
        self.race_order_track_ids = []
        self.completed_track_ids = set()
        self.completed_gate_positions = []
        self.completed_gate_segments = []
        self.target_manager = TargetManager(
            race_gate_count=self.race_gate_count,
            active_target_lost_grace_s=self.active_target_lost_grace_s,
        )
        self.active_waypoints = None
        self.active_times = None
        self.active_waypoint_roles = []
        self.active_plan_generation = 0
        self.active_gate_normal = None
        self.previous_gate_pass_position = None
        self.gate_plane_crossed = False
        self.near_gate_but_not_crossed = False
        self.gate_progress_along_approach = float("nan")
        self.gate_lateral_error = float("nan")
        self.last_planned_gate_idx = -1
        self.trajectory_start_time = 0.0
        self.reference_tau_floor = 0.0
        self.reference_tau_raw = 0.0
        self.reference_tau_reason = "raw"
        self.reference_tau_vehicle = float("nan")
        self.reference_path_lag_m = 0.0
        self.reference_path_distance_m = float("nan")
        self.reference_progress_m = float("nan")
        self.reference_vehicle_progress_m = float("nan")
        self.reference_progress_clamp_tolerance_m = 0.10
        self.reference_progress_clamp_max_error_m = max(
            1.5,
            2.0 * float(self.gate_pass_lateral_radius_m),
        )
        self.reference_progress_clamp_max_tau_step_s = 1.0
        self.last_desired_yaw = 0.0
        self.last_yaw_target_source = "init"
        self.last_yaw_target = None
        self.replan_target_shift_m = float(self.config.planner.replan_target_shift_m)
        self.replan_after_trajectory_s = float(self.config.planner.replan_after_trajectory_s)
        self.replan_min_interval_s = float(self.config.planner.replan_min_interval_s)
        self.last_plan_wall_time = 0.0
        self.planning_horizon_gates = max(1, int(self.config.planner.horizon_gates))
        self.horizon_continuation_enabled = bool(
            self.config.planner.horizon_continuation_enabled
        )
        self.post_gate_exit_continuation_enabled = bool(
            self.config.planner.post_gate_exit_continuation_enabled
        )
        self.gate_corridor_enabled = bool(self.config.planner.gate_corridor_enabled)
        self.gate_corridor_length_m = max(
            0.0,
            float(self.config.planner.gate_corridor_length_m),
        )
        self.reference_progress_clamp_max_error_m = max(
            float(self.reference_progress_clamp_max_error_m),
            0.75 * float(self.gate_corridor_length_m),
        )
        self.passthrough_velocity_enabled = bool(
            self.config.planner.passthrough_velocity_enabled
        )
        self.passthrough_velocity_mode = str(
            self.config.planner.passthrough_velocity_mode or "fixed"
        ).lower()
        self.passthrough_speed_m_s = max(
            0.0,
            float(self.config.planner.passthrough_speed_m_s),
        )
        configured_passthrough_max_speed = float(
            self.config.planner.passthrough_speed_max_m_s
        )
        if configured_passthrough_max_speed <= 0.0:
            configured_passthrough_max_speed = self.passthrough_speed_m_s
        self.passthrough_speed_max_m_s = max(
            self.passthrough_speed_m_s,
            configured_passthrough_max_speed,
        )
        self.passthrough_turn_slowdown = float(
            np.clip(float(self.config.planner.passthrough_turn_slowdown), 0.0, 1.0)
        )
        self.terminal_velocity_enabled = bool(
            self.config.planner.terminal_velocity_enabled
        )
        configured_terminal_speed = float(self.config.planner.terminal_speed_m_s)
        if configured_terminal_speed <= 0.0:
            configured_terminal_speed = self.passthrough_speed_m_s
        self.terminal_speed_m_s = max(0.0, configured_terminal_speed)
        self.yaw_reference_motion_near_gate_enabled = bool(
            self.config.planner.yaw_reference_motion_near_gate_enabled
        )
        self.yaw_reference_motion_distance_m = max(
            0.0,
            float(self.config.planner.yaw_reference_motion_distance_m),
        )
        self.active_target_preempt_enabled = bool(
            self.config.planner.active_target_preempt_enabled
        )
        self.active_target_preempt_min_active_distance_m = max(
            0.0,
            float(self.config.planner.active_target_preempt_min_active_distance_m),
        )
        self.active_target_preempt_margin_m = max(
            0.0,
            float(self.config.planner.active_target_preempt_margin_m),
        )
        self.active_target_preempt_lateral_radius_m = max(
            0.0,
            float(self.config.planner.active_target_preempt_lateral_radius_m),
        )
        self.pending_active_target_preempt_track_id = None
        self.pending_active_target_preempt_details = None
        self.active_target_shift_enabled = bool(
            self.config.planner.active_target_shift_enabled
        )
        self.active_target_shift_threshold_m = float(
            self.config.planner.active_target_shift_threshold_m
        )
        self.active_target_shift_required_frames = max(
            1,
            int(self.config.planner.active_target_shift_required_frames),
        )
        self.active_target_shift_replan_min_interval_s = float(
            self.config.planner.active_target_shift_replan_min_interval_s
        )
        self.active_target_shift_alpha = float(
            np.clip(self.config.planner.active_target_shift_alpha, 0.0, 1.0)
        )
        self.active_target_shift_min_keypoint_conf = float(
            self.config.planner.active_target_shift_min_keypoint_conf
        )
        self.active_target_shift_max_reprojection_error = float(
            self.config.planner.active_target_shift_max_reprojection_error
        )
        self.active_target_shift_max_world_std_m = float(
            self.config.planner.active_target_shift_max_world_std_m
        )
        self.active_target_shift_max_step_m = max(
            0.0,
            float(self.config.planner.active_target_shift_max_step_m),
        )
        self.active_target_shift_max_total_m = float(
            self.config.planner.active_target_shift_max_total_m
        )
        self.active_target_shift_near_gate_distance_m = float(
            self.config.planner.active_target_shift_near_gate_distance_m
        )
        self.active_target_shift_max_near_gate_xy_m = float(
            self.config.planner.active_target_shift_max_near_gate_xy_m
        )
        self.active_target_shift_max_near_gate_z_m = float(
            self.config.planner.active_target_shift_max_near_gate_z_m
        )
        self.active_target_shift_defer_longitudinal_enabled = bool(
            self.config.planner.active_target_shift_defer_longitudinal_enabled
        )
        self.active_target_shift_longitudinal_min_m = max(
            0.0,
            float(self.config.planner.active_target_shift_longitudinal_min_m),
        )
        self.active_target_shift_longitudinal_lateral_max_m = max(
            0.0,
            float(self.config.planner.active_target_shift_longitudinal_lateral_max_m),
        )
        self.active_target_shift_longitudinal_filter_size = max(
            1,
            int(self.config.planner.active_target_shift_longitudinal_filter_size),
        )
        self.active_target_shift_longitudinal_exit_shift_max_m = max(
            0.0,
            float(self.config.planner.active_target_shift_longitudinal_exit_shift_max_m),
        )
        self.active_target_shift_longitudinal_enter_radius_m = max(
            0.0,
            float(self.config.planner.active_target_shift_longitudinal_enter_radius_m),
        )
        self.race_order_front_blocker_enabled = bool(
            self.config.planner.race_order_front_blocker_enabled
        )
        self.race_order_front_blocker_margin_m = max(
            0.0,
            float(self.config.planner.race_order_front_blocker_margin_m),
        )
        self.race_order_front_blocker_lateral_radius_m = max(
            0.0,
            float(self.config.planner.race_order_front_blocker_lateral_radius_m),
        )
        self.provisional_next_gate_enabled = bool(
            self.config.planner.provisional_next_gate_enabled
        )
        self.provisional_next_gate_min_hits = max(
            1,
            int(self.config.planner.provisional_next_gate_min_hits),
        )
        self.provisional_next_gate_max_age_s = max(
            0.0,
            float(self.config.planner.provisional_next_gate_max_age_s),
        )
        self.provisional_next_gate_min_keypoint_conf = float(
            self.config.planner.provisional_next_gate_min_keypoint_conf
        )
        self.provisional_next_gate_max_reprojection_error = float(
            self.config.planner.provisional_next_gate_max_reprojection_error
        )
        self.provisional_next_gate_max_world_std_m = float(
            self.config.planner.provisional_next_gate_max_world_std_m
        )
        self.provisional_next_gate_max_distance_m = max(
            0.0,
            float(self.config.planner.provisional_next_gate_max_distance_m),
        )
        self.provisional_next_gate_max_lateral_m = max(
            0.0,
            float(self.config.planner.provisional_next_gate_max_lateral_m),
        )
        self.provisional_next_gate_closer_margin_m = max(
            0.0,
            float(self.config.planner.provisional_next_gate_closer_margin_m),
        )
        self.provisional_next_gate_max_duration_s = max(
            0.0,
            float(self.config.planner.provisional_next_gate_max_duration_s),
        )
        self.provisional_next_gate_replan_shift_m = max(
            0.0,
            float(self.config.planner.provisional_next_gate_replan_shift_m),
        )
        configured_provisional_vmax = float(
            self.config.planner.provisional_next_gate_vmax_m_s
        )
        self.provisional_next_gate_vmax_m_s = max(
            0.0,
            configured_provisional_vmax,
        )
        self.spline_memory_live_override_m = max(
            0.0,
            float(self.config.planner.spline_memory_live_override_m),
        )
        self.active_target_shift_frames = 0
        self.active_target_shift_track_id = None
        self.active_target_shift_pending_kind = None
        self.deferred_longitudinal_shift_gate_idx = None
        self.deferred_longitudinal_shift_track_id = None
        self.deferred_longitudinal_shift_axis = None
        self.deferred_longitudinal_shift_samples = []
        self.deferred_longitudinal_shift_applied_generation = -1
        self.deferred_longitudinal_shift_pending_signature = None
        self.planner_vmax = float(self.config.planner.vmax)
        self.planner_amax = float(self.config.planner.amax)
        self.planner_t_min = float(self.config.planner.t_min)
        self.plan_validation_shape_enabled = bool(
            self.config.planner.plan_validation_shape_enabled
        )
        self.plan_validation_samples_per_segment = max(
            8,
            int(self.config.planner.plan_validation_samples_per_segment),
        )
        self.plan_validation_max_path_length_ratio = max(
            0.0,
            float(self.config.planner.plan_validation_max_path_length_ratio),
        )
        self.plan_validation_max_segment_path_length_ratio = max(
            0.0,
            float(self.config.planner.plan_validation_max_segment_path_length_ratio),
        )
        self.plan_validation_max_corridor_m = max(
            0.0,
            float(self.config.planner.plan_validation_max_corridor_m),
        )
        self.plan_validation_max_polyline_backtrack_m = max(
            0.0,
            float(self.config.planner.plan_validation_max_polyline_backtrack_m),
        )
        self.plan_validation_max_speed_m_s = max(
            0.0,
            float(self.config.planner.plan_validation_max_speed_m_s),
        )
        self.plan_validation_speed_tolerance_m_s = (
            max(1.0, 0.20 * self.plan_validation_max_speed_m_s)
            if self.plan_validation_max_speed_m_s > 0.0
            else 0.0
        )
        self.plan_validation_max_accel_m_s2 = max(
            0.0,
            float(self.config.planner.plan_validation_max_accel_m_s2),
        )
        self.plan_validation_max_acc_xy_m_s2 = max(
            0.0,
            float(self.config.planner.plan_validation_max_acc_xy_m_s2),
        )
        self.plan_validation_max_lateral_accel_m_s2 = max(
            0.0,
            float(self.config.planner.plan_validation_max_lateral_accel_m_s2),
        )
        self.plan_validation_max_acc_z_up_m_s2 = max(
            0.0,
            float(self.config.planner.plan_validation_max_acc_z_up_m_s2),
        )
        self.plan_validation_max_acc_z_down_m_s2 = max(
            0.0,
            float(self.config.planner.plan_validation_max_acc_z_down_m_s2),
        )
        self.plan_validation_max_z_overshoot_m = max(
            0.0,
            float(self.config.planner.plan_validation_max_z_overshoot_m),
        )
        self.plan_v_start_z_max_m_s = max(
            0.0,
            float(self.config.planner.plan_v_start_z_max_m_s),
        )
        self.safe_min_target_z = float(self.config.planner.safe_min_target_z)
        self.safe_max_target_z = float(self.config.planner.safe_max_target_z)
        self.target_z_mode = str(self.config.planner.target_z_mode).lower()
        self.expected_gate_altitude_m = float(self.config.race.expected_gate_altitude_m)
        self.max_detection_range_m = float(self.config.planner.max_detection_range_m)
        self.max_reprojection_error_for_memory = float(
            self.config.perception.max_reprojection_error_for_memory
        )
        self.min_keypoint_conf_for_memory = float(
            self.config.perception.min_keypoint_conf_for_memory
        )
        self.keypoint_border_margin_px = float(
            self.config.perception.keypoint_border_margin_px
        )
        self.min_quad_area_px2_for_memory = float(
            self.config.perception.min_quad_area_px2_for_memory
        )
        self.max_keypoint_opposite_edge_ratio = float(
            self.config.perception.max_keypoint_opposite_edge_ratio
        )
        self.max_pnp_size_depth_disagreement_m = float(
            self.config.perception.max_pnp_size_depth_disagreement_m
        )
        self.max_pnp_size_depth_disagreement_ratio = float(
            self.config.perception.max_pnp_size_depth_disagreement_ratio
        )
        self.min_depth_m_for_memory = float(self.config.perception.min_depth_m_for_memory)
        self.max_depth_m_for_memory = float(self.config.perception.max_depth_m_for_memory)
        self.reject_negative_depth = bool(self.config.perception.reject_negative_depth)
        gate_memory_config = self.config.gate_memory
        self.gate_memory = GateMemory(
            association_radius=gate_memory_config.association_radius,
            commit_radius=gate_memory_config.commit_radius,
            new_track_block_radius=gate_memory_config.new_track_block_radius,
            min_confidence_per_hit=gate_memory_config.min_confidence_per_hit,
            commit_hits=gate_memory_config.commit_hits,
            commit_confidence_sum=gate_memory_config.commit_confidence_sum,
            commit_spread_radius=gate_memory_config.commit_spread_radius,
            stale_time=gate_memory_config.stale_time,
            alpha=gate_memory_config.alpha,
            use_lookahead_gate_filter=gate_memory_config.use_lookahead_gate_filter,
            history_size=gate_memory_config.history_size,
            min_hits_for_stable=gate_memory_config.min_hits_for_stable,
            max_center_std_for_stable=gate_memory_config.max_center_std_for_stable,
            max_camera_std_for_stable=gate_memory_config.max_camera_std_for_stable,
            max_reprojection_error_for_stable=gate_memory_config.max_reprojection_error_for_stable,
            min_keypoint_conf_for_stable=gate_memory_config.min_keypoint_conf_for_stable,
            max_outlier_distance=gate_memory_config.max_outlier_distance,
            min_observation_time=gate_memory_config.min_observation_time,
        )
        self.gate_memory.max_committed_match_distance = (
            gate_memory_config.max_committed_match_distance
        )
        self.race_order_duplicate_radius_m = float(
            self.gate_memory.duplicate_merge_radius
        )
        state_estimation_config = self.config.state_estimation
        self.estimator_landmark_map = EstimatorLandmarkMap(
            min_hits=state_estimation_config.estimator_landmark_min_hits,
            min_observation_time_s=(
                state_estimation_config.estimator_landmark_min_observation_time_s
            ),
            max_center_std_m=state_estimation_config.estimator_landmark_max_center_std_m,
            max_camera_std_m=state_estimation_config.estimator_landmark_max_camera_std_m,
            max_reprojection_error=(
                state_estimation_config.estimator_landmark_max_reprojection_error
            ),
        )
        self.state_estimator = VehicleStateEstimator(self.config)
        self.shadow_state_estimator = (
            VehicleStateEstimator(self.config, mode_override="estimator")
            if bool(state_estimation_config.run_shadow_estimator)
            else None
        )
        self.last_state_estimate = None
        self.last_shadow_state_estimate = None
        self._last_gate_memory_frame_key = None
        self._last_stable_gate_print_signature = None
        self._last_race_order_print_signature = None
        self._last_race_order_duplicate_signature = None
        self._last_race_order_suffix_filter_signature = None
        self._last_race_order_front_blocker_signature = None
        self._last_race_order_closer_blocker_signature = None
        self._last_provisional_horizon_signature = None
        self._last_target_reject_signature = None
        self._last_plan_validation_reject_signature = None
        self._last_exit_tail_hold_signature = None
        self._last_perception_reject_print_time = 0.0
        self._last_trace_print_time = 0.0
        self._trace_period_s = 0.5
        self._last_shadow_trace_print_time = 0.0
        self._shadow_trace_period_s = float(
            state_estimation_config.shadow_trace_period_s
        )
        self.active_horizon_gate_indices = []
        self.active_horizon_track_ids = []
        self.active_horizon_targets = []
        self.active_plan_mode = ""
        self.active_terminal_velocity = np.zeros(3, dtype=float)
        self.active_terminal_velocity_policy = "unset"
        self.spline_memory_by_gate_idx: dict[int, dict] = {}
        self._last_spline_memory_signature = None
        self.post_gate_exit_until_s = 0.0
        self.post_gate_exit_reason = ""
        self.provisional_target_active = False
        self.provisional_target_track_id = None
        self.provisional_target_gate_idx = None
        self.provisional_target_center = None
        self.provisional_target_start_time = 0.0
        self.provisional_target_last_plan_time = 0.0
        self.provisional_target_plan_count = 0
        self._last_provisional_reject_signature = None
        self.last_gate_pass_preserved_plan = False
        print(
            "[GATE_SOURCE_CONFIG] "
            f"mode={self.gate_source_mode} "
            f"known_gates={len(self.ground_truth_gate_positions_neu)} "
            f"allow_ground_truth={int(bool(self.config.gate_source.allow_ground_truth))}",
            flush=True,
        )
        self._trace_canonical_gate_poses_from_active_world()

        self.planner = MultiSegmentMinimumSnapPlanner()
        self.tracker = RPGHighLevelTracker(
            mass=self.config.controller.mass,
            gravity=self.config.controller.gravity,
            kp=self.config.controller.kp,
            kv=self.config.controller.kv,
            max_tilt_deg=self.config.controller.max_tilt_deg,
            max_acc_xy=self.config.controller.max_acc_xy,
            max_acc_z_up=self.config.controller.max_acc_z_up,
            max_acc_z_down=self.config.controller.max_acc_z_down,
            lateral_accel_gain_xy=self.config.controller.lateral_accel_gain_xy,
            thrust_hover=self.config.controller.thrust_hover,
            thrust_min=self.config.controller.thrust_min,
            thrust_max=self.config.controller.thrust_max,
            max_acc_z_slew_m_s3=self.config.controller.max_acc_z_slew_m_s3,
            max_acc_z_slew_reset_s=self.config.controller.max_acc_z_slew_reset_s,
            near_reference_z_error_m=self.config.controller.near_reference_z_error_m,
            near_reference_vz_error_max_m_s=(
                self.config.controller.near_reference_vz_error_max_m_s
            ),
            near_reference_max_acc_z_up=(
                self.config.controller.near_reference_max_acc_z_up
            ),
            near_reference_max_acc_z_down=(
                self.config.controller.near_reference_max_acc_z_down
            ),
        )
        self.adaptive_hover = AdaptiveHoverThrust(
            initial_thrust=self.config.controller.thrust_hover,
            enabled=self.config.controller.adaptive_hover_enabled,
            gain=self.config.controller.adaptive_hover_gain,
            z_gain=self.config.controller.adaptive_hover_z_gain,
            min_value=self.config.controller.adaptive_hover_min,
            max_value=self.config.controller.adaptive_hover_max,
            max_signal=self.config.controller.adaptive_hover_max_signal,
            max_ref_vz=self.config.controller.adaptive_hover_max_ref_vz,
            max_ref_az=self.config.controller.adaptive_hover_max_ref_az,
            max_z_error=self.config.controller.adaptive_hover_max_z_error,
            saturation_margin=self.config.controller.adaptive_hover_saturation_margin,
            min_confidence=self.config.controller.adaptive_hover_min_confidence,
            fast_enabled=(
                bool(self.config.controller.adaptive_hover_fast_enabled)
                and str(self.config.state_estimation.mode).lower() == "estimator"
            ),
            fast_gain=self.config.controller.adaptive_hover_fast_gain,
            fast_min_z_error=self.config.controller.adaptive_hover_fast_min_z_error,
            fast_stable_signal=self.config.controller.adaptive_hover_fast_stable_signal,
            fast_stable_z_error=self.config.controller.adaptive_hover_fast_stable_z_error,
            fast_stable_samples=self.config.controller.adaptive_hover_fast_stable_samples,
            fast_decay_s=self.config.controller.adaptive_hover_fast_decay_s,
        )
        self.hover_thrust = float(self.adaptive_hover.value)
        self.hover_acquisition = HoverAcquisition(self.config)
        self._last_hover_acquisition_trace_time = 0.0
        self.thrust_scale_calibration = ThrustScaleCalibration(self.config)
        self._last_thrust_scale_calibration_trace_time = 0.0
        self.lateral_response_calibration = LateralResponseCalibration(self.config)
        self._last_lateral_response_calibration_trace_time = 0.0

    def update(self, snapshot) -> AutonomyCommandRad | None:
        snapshot.stable_gate_landmarks_neu = self._stable_gate_landmarks_neu()
        estimate = self.state_estimator.update(snapshot)
        self.last_state_estimate = estimate
        self._update_shadow_estimator(snapshot)

        if not estimate.valid:
            return None

        pos = np.asarray(estimate.pos_neu, dtype=float).reshape(3)
        vel = np.nan_to_num(
            np.asarray(estimate.vel_neu, dtype=float).reshape(3),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        yaw_rad = float(estimate.yaw_rad)

        if not np.all(np.isfinite(pos)):
            return None

        snapshot.pos_neu = pos.copy()
        snapshot.vel_neu = vel.copy()
        snapshot.pos_ned = np.array([pos[0], pos[1], -pos[2]], dtype=float)
        snapshot.vel_ned = np.array([vel[0], vel[1], -vel[2]], dtype=float)
        snapshot.latest_perception = (
            self.state_estimator.project_perception_with_estimated_state(
                getattr(snapshot, "latest_perception", None),
                estimate,
                snapshot,
            )
        )

        self._install_gate_centers(self._gates_from_snapshot(snapshot))

        if not self.hover_acquisition.completed:
            acquisition_result = self.hover_acquisition.update(
                snapshot=snapshot,
                estimate=estimate,
                hover_thrust=self.adaptive_hover.value,
            )
            if acquisition_result.debug.active or acquisition_result.debug.completed:
                self.adaptive_hover.set_value(
                    acquisition_result.hover_thrust,
                    status=f"acquisition_{acquisition_result.debug.status}",
                )
                self.hover_thrust = float(self.adaptive_hover.value)
                self.tracker.thrust_hover = self.hover_thrust
                self._trace_hover_acquisition(pos, acquisition_result.debug)

            if acquisition_result.command is not None:
                command = acquisition_result.command
                return AutonomyCommandRad(
                    roll_rad=command.roll_rad,
                    pitch_rad=command.pitch_rad,
                    yaw_rad=command.yaw_rad,
                    thrust=command.thrust,
                )

        if not self.thrust_scale_calibration.completed:
            calibration_result = self.thrust_scale_calibration.update(
                snapshot=snapshot,
                estimate=estimate,
                hover_thrust=self.adaptive_hover.value,
                hover_acquisition_completed=self.hover_acquisition.completed,
                current_thrust_from_acc_gain=self.tracker.thrust_from_acc_gain,
            )
            if calibration_result.thrust_from_acc_gain is not None:
                self.tracker.thrust_from_acc_gain = float(
                    calibration_result.thrust_from_acc_gain
                )
            if calibration_result.debug.active or calibration_result.debug.completed:
                self._trace_thrust_scale_calibration(
                    pos,
                    calibration_result.debug,
                )

            if calibration_result.command is not None:
                command = calibration_result.command
                return AutonomyCommandRad(
                    roll_rad=command.roll_rad,
                    pitch_rad=command.pitch_rad,
                    yaw_rad=command.yaw_rad,
                    thrust=command.thrust,
                )

        if not self.lateral_response_calibration.completed:
            lateral_result = self.lateral_response_calibration.update(
                snapshot=snapshot,
                estimate=estimate,
                hover_thrust=self.adaptive_hover.value,
                thrust_scale_calibration_completed=(
                    self.thrust_scale_calibration.completed
                ),
                current_lateral_accel_gain_xy=self.tracker.lateral_accel_gain_xy,
            )
            if lateral_result.lateral_accel_gain_xy is not None:
                self.tracker.lateral_accel_gain_xy = np.asarray(
                    lateral_result.lateral_accel_gain_xy,
                    dtype=float,
                ).reshape(2)
            if lateral_result.debug.active or lateral_result.debug.completed:
                self._trace_lateral_response_calibration(
                    pos,
                    lateral_result.debug,
                )

            if lateral_result.command is not None:
                command = lateral_result.command
                return AutonomyCommandRad(
                    roll_rad=command.roll_rad,
                    pitch_rad=command.pitch_rad,
                    yaw_rad=command.yaw_rad,
                    thrust=command.thrust,
                )

        shift_replanned = self._maybe_apply_active_target_shift(pos, vel)
        advanced = self._advance_gate_if_needed(pos, vel=vel)
        if not shift_replanned and self._should_plan(advanced, pos, vel):
            planned = self._path_plan(pos, vel)
            if not planned:
                planned = self._path_plan_provisional_next_gate(pos, vel)
            if not planned and self._active_plan_expired():
                self._clear_active_plan(reason="replan_failed_expired")

        if self.active_waypoints is None or self.planner.total_time <= 0.0:
            return None

        tau_raw = max(0.0, time.time() - self.trajectory_start_time)
        tau_raw = min(tau_raw, float(self.planner.total_time))
        tau = self._reference_progress_clamped_tau(tau_raw, pos)
        p_ref, v_ref, a_ref = self.planner.sample(tau)
        desired_yaw = self._desired_yaw(p_ref, v_ref, a_ref, pos, yaw_rad)

        state = State(pos=pos, vel=vel, yaw=yaw_rad)
        ref = Reference(
            pos=np.asarray(p_ref, dtype=float),
            vel=np.asarray(v_ref, dtype=float),
            acc=np.asarray(a_ref, dtype=float),
            yaw=desired_yaw,
        )

        self.tracker.thrust_hover = float(self.adaptive_hover.value)
        roll_rad, pitch_rad, yaw_cmd_rad, thrust, tracker_debug = (
            self.tracker.update(state, ref)
        )
        thrust = float(np.clip(thrust, 0.0, 1.0))
        hover_debug = self.adaptive_hover.update(
            state=state,
            ref=ref,
            thrust_cmd=thrust,
            estimator_valid=bool(estimate.valid),
            estimator_confidence=float(estimate.confidence),
        )
        self.hover_thrust = float(self.adaptive_hover.value)
        self.tracker.thrust_hover = self.hover_thrust

        # Preserve autonomy_api6's PX4/Gazebo sign convention.
        roll_rad = -float(roll_rad)
        pitch_rad = -float(pitch_rad)
        yaw_cmd_rad = float(yaw_cmd_rad)

        self._trace_autonomy(
            pos=pos,
            target=self.current_gate_pos,
            p_ref=p_ref,
            v_ref=v_ref,
            a_ref=a_ref,
            tau=tau,
            roll_rad=roll_rad,
            pitch_rad=pitch_rad,
            yaw_rad=yaw_cmd_rad,
            thrust=thrust,
            hover_debug=hover_debug,
            tracker_debug=tracker_debug,
        )

        return AutonomyCommandRad(
            roll_rad=roll_rad,
            pitch_rad=pitch_rad,
            yaw_rad=yaw_cmd_rad,
            thrust=thrust,
        )

    def _trace_hover_acquisition(self, pos, debug) -> None:
        now = time.time()
        period_s = float(self.config.hover_acquisition.print_period_s)
        if now - self._last_hover_acquisition_trace_time < period_s:
            return
        self._last_hover_acquisition_trace_time = now

        arr = np.asarray(pos, dtype=float).reshape(3)
        print(
            "hover_acquisition "
            f"status={debug.status} "
            f"active={int(debug.active)} "
            f"done={int(debug.completed)} "
            f"armed={debug.armed} "
            f"elapsed={debug.elapsed_s:.2f} "
            f"z_rel={debug.z_rel_m:.2f} "
            f"release_z={debug.release_z_m:.2f} "
            f"vz={debug.vz_m_s:.2f} "
            f"az={debug.az_m_s2:.2f} "
            f"thrust={debug.thrust:.3f} "
            f"hover={debug.hover_thrust:.3f} "
            f"z_hold_err={debug.z_hold_error_m:.2f} "
            f"z_hold_vz_err={debug.z_hold_vz_error_m_s:.2f} "
            f"z_hold_corr={debug.z_hold_thrust_correction:.3f} "
            f"overshoot_floor={debug.overshoot_thrust_floor:.3f} "
            f"lift={int(debug.lift_confirmed)} "
            f"stable={debug.stable_time_s:.2f} "
            f"pos_neu=({arr[0]:.2f},{arr[1]:.2f},{arr[2]:.2f})",
            flush=True,
        )

    def _trace_thrust_scale_calibration(self, pos, debug) -> None:
        now = time.time()
        period_s = float(self.config.thrust_scale_calibration.print_period_s)
        if (
            not bool(debug.completed)
            and now - self._last_thrust_scale_calibration_trace_time < period_s
        ):
            return
        self._last_thrust_scale_calibration_trace_time = now

        arr = np.asarray(pos, dtype=float).reshape(3)
        print(
            "thrust_scale_calibration "
            f"status={debug.status} "
            f"active={int(debug.active)} "
            f"done={int(debug.completed)} "
            f"armed={debug.armed} "
            f"elapsed={debug.elapsed_s:.2f} "
            f"z_rel={debug.z_rel_m:.2f} "
            f"vz={debug.vz_m_s:.2f} "
            f"az={debug.az_m_s2:.2f} "
            f"az_src={debug.accel_source} "
            f"thrust={debug.thrust:.3f} "
            f"hover={debug.hover_thrust:.3f} "
            f"delta={debug.thrust_delta:.3f} "
            f"z_hold_err={debug.z_hold_error_m:.2f} "
            f"z_hold_vz_err={debug.z_hold_vz_error_m_s:.2f} "
            f"z_hold_corr={debug.z_hold_thrust_correction:.3f} "
            f"samples={debug.samples} "
            f"accel_per_thrust={debug.accel_per_thrust:.2f} "
            f"thrust_gain={debug.thrust_from_acc_gain:.4f} "
            f"conf={debug.confidence:.2f} "
            f"pos_neu=({arr[0]:.2f},{arr[1]:.2f},{arr[2]:.2f})",
            flush=True,
        )

    def _trace_lateral_response_calibration(self, pos, debug) -> None:
        now = time.time()
        period_s = float(self.config.lateral_response_calibration.print_period_s)
        if (
            not bool(debug.completed)
            and now - self._last_lateral_response_calibration_trace_time < period_s
        ):
            return
        self._last_lateral_response_calibration_trace_time = now

        arr = np.asarray(pos, dtype=float).reshape(3)
        cmd_xy = debug.command_accel_xy_m_s2
        sample_cmd_xy = debug.sampled_command_accel_xy_m_s2
        acc_xy = debug.measured_accel_xy_m_s2
        sample_acc_xy = debug.sampled_measured_accel_xy_m_s2
        ratio_xy = debug.response_ratio_xy
        gain_xy = debug.lateral_accel_gain_xy
        samples_xy = debug.samples_xy
        signed_samples_xy = debug.signed_samples_xy
        print(
            "lateral_response_calibration "
            f"status={debug.status} "
            f"active={int(debug.active)} "
            f"done={int(debug.completed)} "
            f"armed={debug.armed} "
            f"elapsed={debug.elapsed_s:.2f} "
            f"xy_rel={debug.xy_rel_m:.2f} "
            f"vxy={debug.vxy_m_s:.2f} "
            f"z_rel={debug.z_rel_m:.2f} "
            f"z_hold_err={debug.z_hold_error_m:.2f} "
            f"z_hold_vz_err={debug.z_hold_vz_error_m_s:.2f} "
            f"z_hold_corr={debug.z_hold_thrust_correction:.3f} "
            f"acc_src={debug.accel_source} "
            f"cmd_acc_xy=({cmd_xy[0]:.2f},{cmd_xy[1]:.2f}) "
            f"meas_acc_xy=({acc_xy[0]:.2f},{acc_xy[1]:.2f}) "
            f"sample_cmd_xy=({sample_cmd_xy[0]:.2f},{sample_cmd_xy[1]:.2f}) "
            f"sample_meas_xy=({sample_acc_xy[0]:.2f},{sample_acc_xy[1]:.2f}) "
            f"sample_axis={debug.sampled_axis} "
            f"sample_age={debug.sampled_age_s:.2f} "
            f"sample_ratio={debug.sampled_response_ratio:.2f} "
            f"sample_status={debug.sampled_status} "
            f"cmd_deg=({math.degrees(debug.roll_rad):.2f},{math.degrees(debug.pitch_rad):.2f}) "
            f"thrust={debug.thrust:.3f} "
            f"hover={debug.hover_thrust:.3f} "
            f"ratio_xy=({ratio_xy[0]:.2f},{ratio_xy[1]:.2f}) "
            f"lat_gain=({gain_xy[0]:.3f},{gain_xy[1]:.3f}) "
            f"samples_xy=({samples_xy[0]},{samples_xy[1]}) "
            f"signed_samples_xy=({signed_samples_xy[0]},{signed_samples_xy[1]},"
            f"{signed_samples_xy[2]},{signed_samples_xy[3]}) "
            f"conf={debug.confidence:.2f} "
            f"pos_neu=({arr[0]:.2f},{arr[1]:.2f},{arr[2]:.2f})",
            flush=True,
        )

    def _update_shadow_estimator(self, snapshot) -> None:
        if self.shadow_state_estimator is None:
            return
        estimate = self.shadow_state_estimator.update(snapshot)
        self.last_shadow_state_estimate = estimate
        self._trace_shadow_estimator(estimate)

    def _trace_shadow_estimator(self, estimate) -> None:
        now = time.time()
        if now - self._last_shadow_trace_print_time < self._shadow_trace_period_s:
            return
        self._last_shadow_trace_print_time = now

        def fmt_vec(value) -> str:
            if value is None:
                return "None"
            try:
                arr = np.asarray(value, dtype=float).reshape(3)
            except (TypeError, ValueError):
                return "None"
            if not np.all(np.isfinite(arr)):
                return "None"
            return f"({arr[0]:.2f},{arr[1]:.2f},{arr[2]:.2f})"

        def fmt_float(value) -> str:
            try:
                out = float(value)
            except (TypeError, ValueError):
                return "nan"
            return f"{out:.3f}" if math.isfinite(out) else "nan"

        correction_source = str(
            getattr(estimate, "vision_correction_source", "")
        )
        correction_residual = getattr(
            estimate,
            "vision_correction_residual_m",
            None,
        )
        correction_count = int(
            getattr(estimate, "vision_correction_count", 0)
        )
        correction_txt = "none"
        if (
            correction_source
            and correction_residual is not None
            and math.isfinite(float(correction_residual))
        ):
            correction_txt = (
                f"{correction_source}@{float(correction_residual):.2f}/"
                f"{correction_count}"
            )

        truth_error = getattr(estimate, "truth_error_m", None)
        truth_error_txt = (
            "nan"
            if truth_error is None or not math.isfinite(float(truth_error))
            else f"{float(truth_error):.2f}"
        )

        print(
            "shadow_estimator_trace "
            f"valid={int(bool(getattr(estimate, 'valid', False)))} "
            f"truth_err={truth_error_txt} "
            f"imu_dt={fmt_float(getattr(estimate, 'imu_dt_s', None))} "
            f"raw_accel_body={fmt_vec(getattr(estimate, 'raw_accel_body', None))} "
            f"computed_acc_neu={fmt_vec(getattr(estimate, 'computed_acc_neu', None))} "
            f"pos_neu={fmt_vec(getattr(estimate, 'pos_neu', None))} "
            f"truth_pos_neu={fmt_vec(getattr(estimate, 'truth_pos_neu', None))} "
            f"est_minus_truth_neu={fmt_vec(getattr(estimate, 'position_error_neu', None))} "
            f"integrated_vel_neu={fmt_vec(getattr(estimate, 'vel_neu', None))} "
            f"visual_vel_neu={fmt_vec(getattr(estimate, 'visual_velocity_neu', None))} "
            f"visual_dt={fmt_float(getattr(estimate, 'visual_velocity_dt_s', None))} "
            f"visual_ref_reset={int(bool(getattr(estimate, 'visual_reference_reset', False)))} "
            f"visual_vel_reason={str(getattr(estimate, 'visual_velocity_reason', '') or 'none')} "
            f"mavlink_vel_neu={fmt_vec(getattr(estimate, 'truth_vel_neu', None))} "
            f"vel_minus_mavlink_neu={fmt_vec(getattr(estimate, 'velocity_error_neu', None))} "
            f"accel_bias_neu={fmt_vec(getattr(estimate, 'accel_bias_neu', None))} "
            f"corr={correction_txt}",
            flush=True,
        )

    def _trace_autonomy(
        self,
        pos,
        target,
        p_ref,
        v_ref,
        a_ref,
        tau: float,
        roll_rad: float,
        pitch_rad: float,
        yaw_rad: float,
        thrust: float,
        hover_debug=None,
        tracker_debug=None,
    ) -> None:
        now = time.time()
        if now - self._last_trace_print_time < self._trace_period_s:
            return
        self._last_trace_print_time = now

        def fmt_vec(value) -> str:
            arr = np.asarray(value, dtype=float).reshape(3)
            return f"({arr[0]:.2f},{arr[1]:.2f},{arr[2]:.2f})"

        target_txt = "None"
        dist_txt = "nan"
        if target is not None:
            target_arr = np.asarray(target, dtype=float).reshape(3)
            target_txt = fmt_vec(target_arr)
            dist_txt = f"{float(np.linalg.norm(np.asarray(pos) - target_arr)):.2f}"
        yaw_target_txt = "None"
        if self.last_yaw_target is not None:
            try:
                yaw_target_arr = np.asarray(self.last_yaw_target, dtype=float).reshape(3)
            except (TypeError, ValueError):
                yaw_target_arr = None
            if yaw_target_arr is not None and np.all(np.isfinite(yaw_target_arr)):
                yaw_target_txt = fmt_vec(yaw_target_arr)

        state_source = "unknown"
        truth_error_txt = "nan"
        truth_pos_txt = "None"
        correction_txt = "none"
        if self.last_state_estimate is not None:
            state_source = str(self.last_state_estimate.source)
            truth_error = self.last_state_estimate.truth_error_m
            if truth_error is not None and math.isfinite(float(truth_error)):
                truth_error_txt = f"{float(truth_error):.2f}"
            truth_pos = getattr(self.last_state_estimate, "truth_pos_neu", None)
            if truth_pos is not None:
                try:
                    truth_pos_arr = np.asarray(truth_pos, dtype=float).reshape(3)
                except (TypeError, ValueError):
                    truth_pos_arr = None
                if truth_pos_arr is not None and np.all(np.isfinite(truth_pos_arr)):
                    truth_pos_txt = fmt_vec(truth_pos_arr)
            correction_source = str(
                getattr(self.last_state_estimate, "vision_correction_source", "")
            )
            correction_residual = getattr(
                self.last_state_estimate,
                "vision_correction_residual_m",
                None,
            )
            correction_count = int(
                getattr(self.last_state_estimate, "vision_correction_count", 0)
            )
            if (
                correction_source
                and correction_residual is not None
                and math.isfinite(float(correction_residual))
                and correction_count > 0
            ):
                correction_txt = (
                    f"{correction_source}@{float(correction_residual):.2f}/{correction_count}"
                )

        hover_txt = "nan"
        hover_status = "none"
        hover_signal_txt = "0.00"
        hover_err_txt = "(0.00,0.00)"
        hover_gain_txt = "0.00"
        hover_fast_txt = "0.00"
        if hover_debug is not None:
            hover_txt = f"{float(hover_debug.value):.3f}"
            hover_status = str(hover_debug.status)
            hover_signal_txt = f"{float(hover_debug.signal):.2f}"
            hover_err_txt = (
                f"({float(hover_debug.z_error):.2f},{float(hover_debug.vz_error):.2f})"
            )
            hover_gain_txt = f"{float(hover_debug.gain):.3f}"
            hover_fast_txt = f"{float(hover_debug.fast_weight):.2f}"
        tracker_a_raw_z_txt = "nan"
        tracker_a_limit_z_txt = "nan"
        tracker_a_cmd_z_txt = "nan"
        tracker_near_z_txt = "0"
        tracker_vz_limited_txt = "0"
        tracker_slew_limited_txt = "0"
        if tracker_debug is not None:
            try:
                tracker_a_raw = np.asarray(
                    tracker_debug.get("a_cmd_raw_no_g"),
                    dtype=float,
                ).reshape(3)
                tracker_a_limit = np.asarray(
                    tracker_debug.get("a_cmd_accel_limited_no_g"),
                    dtype=float,
                ).reshape(3)
                tracker_a_cmd = np.asarray(
                    tracker_debug.get("a_cmd_no_g"),
                    dtype=float,
                ).reshape(3)
                tracker_a_raw_z_txt = f"{tracker_a_raw[2]:.2f}"
                tracker_a_limit_z_txt = f"{tracker_a_limit[2]:.2f}"
                tracker_a_cmd_z_txt = f"{tracker_a_cmd[2]:.2f}"
            except (TypeError, ValueError):
                pass
            tracker_near_z_txt = str(
                int(bool(tracker_debug.get("near_reference_vertical", False)))
            )
            tracker_vz_limited_txt = str(
                int(bool(tracker_debug.get("vertical_velocity_error_limited", False)))
            )
            tracker_slew_limited_txt = str(
                int(bool(tracker_debug.get("vertical_accel_slew_limited", False)))
            )

        target_diag = self.target_manager.diagnostics()
        lateral_gain = np.asarray(
            self.tracker.lateral_accel_gain_xy,
            dtype=float,
        ).reshape(2)
        target_shift_txt = "nan"
        target_shift_xy_txt = "nan"
        target_shift_z_txt = "nan"
        target_lock_age_txt = "nan"
        if math.isfinite(float(target_diag.shift_m)):
            target_shift_txt = f"{float(target_diag.shift_m):.2f}"
            target_shift_xy_txt = f"{float(target_diag.shift_xy_m):.2f}"
            target_shift_z_txt = f"{float(target_diag.shift_z_m):.2f}"
        if math.isfinite(float(target_diag.lock_age_s)):
            target_lock_age_txt = f"{float(target_diag.lock_age_s):.2f}"

        print(
            "autonomy_trace "
            f"gate_idx={self.current_gate_idx} "
            f"active_track={self.active_target_track_id} "
            f"tracks={self.active_track_count} "
            f"target_lock={int(target_diag.locked)} "
            f"target_event={target_diag.event} "
            f"target_shift={target_shift_txt} "
            f"target_shift_xy_z=({target_shift_xy_txt},{target_shift_z_txt}) "
            f"target_lock_age={target_lock_age_txt} "
            f"active_replan_held={int(target_diag.suppress_active_replan)} "
            f"state={state_source} "
            f"truth_err={truth_error_txt} "
            f"truth_pos_neu={truth_pos_txt} "
            f"corr={correction_txt} "
            f"tau={tau:.2f}/{float(self.planner.total_time):.2f} "
            f"tau_raw={float(self.reference_tau_raw):.2f} "
            f"tau_mode={self.reference_tau_reason} "
            f"tau_vehicle={self._fmt_float(self.reference_tau_vehicle, precision=2)} "
            f"path_lag={self._fmt_float(self.reference_path_lag_m, precision=2)} "
            f"path_dist={self._fmt_float(self.reference_path_distance_m, precision=2)} "
            f"dist={dist_txt} "
            f"pos_neu={fmt_vec(pos)} "
            f"target_neu={target_txt} "
            f"p_ref={fmt_vec(p_ref)} "
            f"v_ref={fmt_vec(v_ref)} "
            f"a_ref={fmt_vec(a_ref)} "
            f"yaw_source={self.last_yaw_target_source} "
            f"yaw_target_neu={yaw_target_txt} "
            f"cmd_deg=({math.degrees(roll_rad):.2f},{math.degrees(pitch_rad):.2f},{math.degrees(yaw_rad):.2f}) "
            f"thrust={thrust:.3f} "
            f"hover={hover_txt} "
            f"thrust_gain={float(self.tracker.thrust_from_acc_gain):.4f} "
            f"lat_gain=({lateral_gain[0]:.3f},{lateral_gain[1]:.3f}) "
            f"hover_adapt={hover_status} "
            f"hover_sig={hover_signal_txt} "
            f"hover_err_z_vz={hover_err_txt} "
            f"tracker_a_z_raw_limit_cmd=({tracker_a_raw_z_txt},{tracker_a_limit_z_txt},{tracker_a_cmd_z_txt}) "
            f"tracker_near_vz_slew=({tracker_near_z_txt},{tracker_vz_limited_txt},{tracker_slew_limited_txt}) "
            f"hover_gain={hover_gain_txt} "
            f"hover_fast={hover_fast_txt}",
            flush=True,
        )

    def _sync_target_manager_state(self, clear_unlocked_current: bool = False) -> None:
        diag = self.target_manager.diagnostics()
        self.current_gate_idx = int(diag.gate_idx)
        if not self.provisional_target_active or diag.center_at_plan is not None:
            self.active_target_track_id = diag.active_track_id
        self.completed_track_ids = set(self.target_manager.completed_track_ids)
        self.active_target_center_at_plan = (
            None if diag.center_at_plan is None else diag.center_at_plan.copy()
        )
        self.active_target_latest_filtered_center = (
            None if diag.latest_center is None else diag.latest_center.copy()
        )

        if diag.center_at_plan is not None:
            self.current_gate_pos = diag.center_at_plan.copy()
            self.last_active_target_center = diag.center_at_plan.copy()
            self.active_target_lost_time = None
        elif clear_unlocked_current and not self.provisional_target_active:
            self.current_gate_pos = None
            self.last_active_target_center = None
            self.active_target_lost_time = None

    def _reset_gate_pass_state(self) -> None:
        self.active_gate_normal = None
        self.previous_gate_pass_position = None
        self.gate_plane_crossed = False
        self.near_gate_but_not_crossed = False
        self.gate_progress_along_approach = float("nan")
        self.gate_lateral_error = float("nan")
        self._last_exit_tail_hold_signature = None

    def _initialize_gate_pass_tracking(
        self,
        *,
        pos: np.ndarray,
        target: np.ndarray,
        fallback_normal: np.ndarray | None = None,
    ) -> None:
        pos = np.asarray(pos, dtype=float).reshape(3)
        target = np.asarray(target, dtype=float).reshape(3)
        normal = unit_vector_from_to(pos, target, fallback=fallback_normal)
        self.active_gate_normal = None if normal is None else normal.copy()
        self.previous_gate_pass_position = pos.copy()
        self.gate_plane_crossed = False
        self.near_gate_but_not_crossed = False
        self.gate_lateral_error = float("nan")
        self.gate_progress_along_approach = (
            float(np.dot(pos - target, normal))
            if normal is not None
            else float("nan")
        )

    def _is_final_race_gate_index(self, gate_idx: int) -> bool:
        if self.race_gate_count is None:
            return False
        return int(gate_idx) >= int(self.race_gate_count) - 1

    def _has_future_race_gate(self, gate_idx: int) -> bool:
        if self.race_gate_count is None:
            return True
        return int(gate_idx) < int(self.race_gate_count)

    def _active_plan_remaining_s(self) -> float:
        if self.active_waypoints is None or self.planner.total_time <= 0.0:
            return 0.0
        elapsed = time.time() - float(self.trajectory_start_time)
        return max(0.0, float(self.planner.total_time) - elapsed)

    def _post_gate_exit_active(self, now: float | None = None) -> bool:
        now = time.time() if now is None else float(now)
        return (
            self.active_waypoints is not None
            and self.planner.total_time > 0.0
            and now <= float(self.post_gate_exit_until_s)
        )

    def _active_gate_exit_waypoint(self) -> tuple[np.ndarray | None, str]:
        if self.active_waypoints is None:
            return None, "none"
        roles = list(self.active_waypoint_roles or [])
        if not roles or len(roles) != len(self.active_waypoints):
            return None, "missing_roles"

        try:
            horizon_idx = self.active_horizon_gate_indices.index(
                int(self.current_gate_idx)
            )
        except ValueError:
            return None, "gate_not_in_active_horizon"
        if horizon_idx < 0:
            return None, "invalid_horizon_index"

        seen_gate_centers = 0
        for idx, role in enumerate(roles):
            if role != "gate_center":
                continue
            if seen_gate_centers != horizon_idx:
                seen_gate_centers += 1
                continue

            for exit_idx in range(idx + 1, len(roles)):
                exit_role = roles[exit_idx]
                if exit_role == "gate_center":
                    break
                if exit_role in ("gate_exit", "gate_exit_shifted"):
                    return (
                        np.asarray(
                            self.active_waypoints[exit_idx],
                            dtype=float,
                        ).reshape(3).copy(),
                        str(exit_role),
                    )
            return None, "missing_exit_after_center"

        return None, "missing_gate_center"

    def _reset_reference_progress_state(self) -> None:
        self.reference_tau_floor = 0.0
        self.reference_tau_raw = 0.0
        self.reference_tau_reason = "raw"
        self.reference_tau_vehicle = float("nan")
        self.reference_path_lag_m = 0.0
        self.reference_path_distance_m = float("nan")
        self.reference_progress_m = float("nan")
        self.reference_vehicle_progress_m = float("nan")

    def _reference_progress_clamped_tau(self, raw_tau: float, pos: np.ndarray) -> float:
        try:
            total_time = float(self.planner.total_time)
        except (TypeError, ValueError):
            total_time = 0.0
        if not math.isfinite(total_time) or total_time <= 0.0:
            self._reset_reference_progress_state()
            return 0.0

        raw_tau = float(np.clip(float(raw_tau), 0.0, total_time))
        tau = raw_tau
        reason = "raw"
        self.reference_tau_raw = raw_tau
        self.reference_tau_vehicle = float("nan")
        self.reference_path_lag_m = 0.0
        self.reference_path_distance_m = float("nan")
        self.reference_progress_m = float("nan")
        self.reference_vehicle_progress_m = float("nan")

        if (
            math.isfinite(float(self.reference_tau_floor))
            and self.reference_tau_floor > tau + 1e-6
        ):
            tau = min(total_time, float(self.reference_tau_floor))
            reason = "floor"

        projection = self._project_position_to_planner_path(pos)
        if projection is not None:
            vehicle_tau = float(projection["tau"])
            vehicle_progress = float(projection["progress_m"])
            path_distance = float(projection["distance_m"])
            reference_progress = self._planner_path_progress_at_tau(
                raw_tau,
                projection["taus"],
                projection["cumulative_m"],
            )
            path_lag = vehicle_progress - reference_progress
            self.reference_tau_vehicle = vehicle_tau
            self.reference_path_lag_m = path_lag
            self.reference_path_distance_m = path_distance
            self.reference_progress_m = reference_progress
            self.reference_vehicle_progress_m = vehicle_progress

            if (
                math.isfinite(path_lag)
                and math.isfinite(path_distance)
                and path_distance <= float(self.reference_progress_clamp_max_error_m)
                and path_lag > float(self.reference_progress_clamp_tolerance_m)
                and vehicle_tau > tau + 1e-6
            ):
                if self._post_gate_exit_active() or self.gate_plane_crossed:
                    max_tau = total_time
                    clamp_reason = "path_progress_post_gate"
                else:
                    max_tau = min(
                        total_time,
                        raw_tau + float(self.reference_progress_clamp_max_tau_step_s),
                    )
                    clamp_reason = "path_progress"
                clamped_tau = min(vehicle_tau, max_tau)
                if clamped_tau > tau + 1e-6:
                    tau = clamped_tau
                    reason = clamp_reason

        self.reference_tau_floor = max(
            float(self.reference_tau_floor),
            float(tau),
        )
        self.reference_tau_reason = reason
        return float(tau)

    def _project_position_to_planner_path(self, point: np.ndarray) -> dict | None:
        try:
            total_time = float(self.planner.total_time)
            point = np.asarray(point, dtype=float).reshape(3)
        except (TypeError, ValueError):
            return None
        if (
            not math.isfinite(total_time)
            or total_time <= 0.0
            or not np.all(np.isfinite(point))
        ):
            return None

        count = max(24, min(160, int(math.ceil(total_time * 8.0)) + 1))
        taus = np.linspace(0.0, total_time, count)
        positions = np.zeros((count, 3), dtype=float)
        cumulative = np.zeros(count, dtype=float)
        for idx, tau in enumerate(taus):
            try:
                sample_pos, _, _ = self.planner.sample(float(tau))
            except Exception:
                return None
            positions[idx] = np.asarray(sample_pos, dtype=float).reshape(3)
            if not np.all(np.isfinite(positions[idx])):
                return None
            if idx > 0:
                segment_length = float(np.linalg.norm(positions[idx] - positions[idx - 1]))
                if not math.isfinite(segment_length):
                    return None
                cumulative[idx] = cumulative[idx - 1] + max(0.0, segment_length)

        best_distance = float("inf")
        best_progress = 0.0
        best_tau = 0.0
        best_segment = -1
        for idx in range(count - 1):
            start = positions[idx]
            end = positions[idx + 1]
            delta = end - start
            segment_length_sq = float(np.dot(delta, delta))
            if segment_length_sq <= 1e-12:
                alpha = 0.0
                projection = start
            else:
                alpha = float(np.dot(point - start, delta) / segment_length_sq)
                alpha = float(np.clip(alpha, 0.0, 1.0))
                projection = start + alpha * delta
            distance = float(np.linalg.norm(point - projection))
            if not math.isfinite(distance) or distance >= best_distance:
                continue
            segment_progress = cumulative[idx + 1] - cumulative[idx]
            best_distance = distance
            best_progress = cumulative[idx] + alpha * segment_progress
            best_tau = float(taus[idx] + alpha * (taus[idx + 1] - taus[idx]))
            best_segment = idx

        if best_segment < 0:
            return None
        return {
            "distance_m": float(best_distance),
            "progress_m": float(best_progress),
            "tau": float(best_tau),
            "segment": int(best_segment),
            "taus": taus,
            "cumulative_m": cumulative,
        }

    @staticmethod
    def _planner_path_progress_at_tau(
        tau: float,
        taus: np.ndarray,
        cumulative_m: np.ndarray,
    ) -> float:
        if len(taus) == 0 or len(taus) != len(cumulative_m):
            return float("nan")
        tau = float(tau)
        if not math.isfinite(tau):
            return float("nan")
        if tau <= float(taus[0]):
            return float(cumulative_m[0])
        if tau >= float(taus[-1]):
            return float(cumulative_m[-1])
        idx = int(np.searchsorted(taus, tau, side="right") - 1)
        idx = max(0, min(idx, len(taus) - 2))
        dt = float(taus[idx + 1] - taus[idx])
        if abs(dt) <= 1e-12:
            return float(cumulative_m[idx])
        alpha = float(np.clip((tau - float(taus[idx])) / dt, 0.0, 1.0))
        return float(
            cumulative_m[idx]
            + alpha * (float(cumulative_m[idx + 1]) - float(cumulative_m[idx]))
        )

    def _completed_landmark_radius_m(self) -> float:
        return max(
            float(self.race_order_duplicate_radius_m),
            float(self.pass_radius_m),
            float(self.gate_pass_lateral_radius_m),
        )

    def _completed_segment_corridor_radius_m(self) -> float:
        return max(
            float(self.race_order_duplicate_radius_m),
            float(self.gate_pass_lateral_radius_m),
        )

    def _is_near_completed_landmark(
        self,
        center,
        *,
        radius: float | None = None,
    ) -> bool:
        center = self._finite_vec3_or_none(center)
        if center is None:
            return False
        radius = self._completed_landmark_radius_m() if radius is None else float(radius)
        if radius <= 0.0:
            return False
        for completed in self.completed_gate_positions:
            completed = self._finite_vec3_or_none(completed)
            if completed is None:
                continue
            if float(np.linalg.norm(center - completed)) <= radius:
                return True
        return False

    @staticmethod
    def _point_in_segment_corridor(
        point,
        start,
        end,
        *,
        radius: float,
    ) -> bool:
        point = np.asarray(point, dtype=float).reshape(3)
        start = np.asarray(start, dtype=float).reshape(3)
        end = np.asarray(end, dtype=float).reshape(3)
        if not (
            np.all(np.isfinite(point))
            and np.all(np.isfinite(start))
            and np.all(np.isfinite(end))
        ):
            return False
        segment = end - start
        denom = float(np.dot(segment, segment))
        if denom < 1e-9:
            return float(np.linalg.norm(point - start)) <= float(radius)
        t = float(np.dot(point - start, segment) / denom)
        if t < 0.0 or t > 1.0:
            return False
        closest = start + t * segment
        return float(np.linalg.norm(point - closest)) <= float(radius)

    def _is_inside_completed_segment_corridor(
        self,
        center,
        *,
        radius: float | None = None,
    ) -> bool:
        center = self._finite_vec3_or_none(center)
        if center is None:
            return False
        radius = (
            self._completed_segment_corridor_radius_m()
            if radius is None
            else float(radius)
        )
        if radius <= 0.0:
            return False
        for start, end in self.completed_gate_segments:
            if self._point_in_segment_corridor(
                center,
                start,
                end,
                radius=radius,
            ):
                return True
        return False

    def _record_completed_landmark(self, track_id, center) -> None:
        center = self._finite_vec3_or_none(center)
        if center is None:
            return
        if track_id is not None:
            try:
                self.completed_track_ids.add(int(track_id))
            except (TypeError, ValueError):
                pass

        if self._is_near_completed_landmark(center):
            return

        if self.completed_gate_positions:
            previous = np.asarray(self.completed_gate_positions[-1], dtype=float).reshape(3)
            if np.all(np.isfinite(previous)):
                self.completed_gate_segments.append((previous.copy(), center.copy()))
        self.completed_gate_positions.append(center.copy())

    def _target_rejection_reason(self, center, track_id=None) -> str:
        center = self._finite_vec3_or_none(center)
        if center is None:
            return "non_finite_target"
        if track_id is not None:
            try:
                if int(track_id) in self.completed_track_ids:
                    return "completed_track_id"
            except (TypeError, ValueError):
                pass
        if self._is_near_completed_landmark(center):
            return "duplicate_of_completed_landmark"
        if self._is_inside_completed_segment_corridor(center):
            return "completed_segment_corridor"
        return ""

    def _trace_target_rejection(
        self,
        *,
        reason: str,
        track_id,
        center,
        context: str,
    ) -> None:
        center = self._finite_vec3_or_none(center)
        rounded_center = (
            None
            if center is None
            else tuple(round(float(value), 2) for value in center)
        )
        signature = (
            str(context),
            str(reason),
            None if track_id is None else int(track_id),
            rounded_center,
        )
        if signature == self._last_target_reject_signature:
            return
        self._last_target_reject_signature = signature
        print(
            "target_validation_reject "
            f"context={context} "
            f"track={track_id if track_id is not None else 'none'} "
            f"reason={reason} "
            f"center={self._fmt_vec(center, precision=3) if center is not None else 'none'} "
            f"completed_tracks={sorted(int(t) for t in self.completed_track_ids)} "
            f"completed_landmarks={len(self.completed_gate_positions)} "
            f"completed_segments={len(self.completed_gate_segments)}",
            flush=True,
        )

    def _track_cluster_ids(
        self,
        track_id: int,
        committed_by_id: dict[int, object] | None = None,
        *,
        radius: float | None = None,
    ) -> list[int]:
        try:
            seed_id = int(track_id)
        except (TypeError, ValueError):
            return []
        if seed_id < 0:
            return [seed_id]

        if committed_by_id is None:
            committed_by_id = {
                int(track.id): track for track in self.gate_memory.get_committed_tracks()
            }
        if seed_id not in committed_by_id:
            track = self.gate_memory.get_track_by_id(seed_id)
            if track is not None:
                committed_by_id = dict(committed_by_id)
                committed_by_id[seed_id] = track
        if seed_id not in committed_by_id:
            return []

        radius = (
            float(self.race_order_duplicate_radius_m)
            if radius is None
            else float(radius)
        )
        if radius <= 0.0:
            return [seed_id]

        cluster: list[int] = []
        pending = [seed_id]
        while pending:
            current_id = int(pending.pop(0))
            if current_id in cluster:
                continue
            cluster.append(current_id)
            current_center = self._race_order_track_center(current_id, committed_by_id)
            if current_center is None:
                continue
            for other_id in sorted(committed_by_id):
                other_id = int(other_id)
                if other_id in cluster or other_id in pending:
                    continue
                other_center = self._race_order_track_center(other_id, committed_by_id)
                if other_center is None:
                    continue
                dist = float(np.linalg.norm(current_center - other_center))
                if math.isfinite(dist) and dist <= radius:
                    pending.append(other_id)
        return cluster

    def _best_duplicate_cluster_center(
        self,
        track_id,
        committed_by_id: dict[int, object] | None = None,
    ) -> tuple[np.ndarray | None, int | None, dict]:
        try:
            active_id = int(track_id)
        except (TypeError, ValueError):
            return None, None, {"ok": False, "reason": "invalid_track_id"}
        if active_id < 0:
            return None, active_id, {"ok": True, "reason": "ground_truth_track"}

        if committed_by_id is None:
            committed_by_id = {
                int(track.id): track for track in self.gate_memory.get_committed_tracks()
            }
        cluster_ids = self._track_cluster_ids(active_id, committed_by_id)
        best = None
        for candidate_id in cluster_ids:
            if candidate_id in self.completed_track_ids:
                continue
            track = committed_by_id.get(candidate_id)
            if track is None:
                track = self.gate_memory.get_track_by_id(candidate_id)
            center, quality = self._track_filtered_center_for_navigation(track)
            if center is None:
                continue
            score = self._race_order_track_score(candidate_id, committed_by_id)
            key = (*score, -abs(candidate_id - active_id), -candidate_id)
            if best is None or key > best[0]:
                best = (key, center.copy(), int(candidate_id), dict(quality))

        if best is not None:
            quality = best[3]
            quality["cluster_ids"] = tuple(cluster_ids)
            quality["source_track_id"] = int(best[2])
            return best[1], int(best[2]), quality

        track = committed_by_id.get(active_id)
        if track is None:
            track = self.gate_memory.get_track_by_id(active_id)
        center = self._track_navigation_center(track) if track is not None else None
        return center, active_id, {
            "ok": False,
            "reason": "fallback_exact_track" if center is not None else "missing_cluster_center",
            "cluster_ids": tuple(cluster_ids),
            "source_track_id": active_id,
        }

    def _physical_navigation_center(
        self,
        track_id,
        committed_by_id: dict[int, object] | None = None,
        *,
        require_fresh: bool = False,
    ) -> np.ndarray | None:
        center, _, quality = self._best_duplicate_cluster_center(track_id, committed_by_id)
        if require_fresh and not bool(quality.get("ok", False)):
            return None
        return None if center is None else center.copy()

    def _active_plan_expired(self, now: float | None = None) -> bool:
        if self.active_waypoints is None or self.planner.total_time <= 0.0:
            return False
        now = time.time() if now is None else float(now)
        elapsed = now - float(self.trajectory_start_time)
        return elapsed > float(self.planner.total_time) + float(self.replan_after_trajectory_s)

    def _clear_active_plan(self, *, reason: str) -> None:
        print(
            "active_plan_clear "
            f"reason={reason} "
            f"gate_idx={int(self.current_gate_idx)} "
            f"active_plan_mode={self.active_plan_mode} "
            f"terminal_policy={self.active_terminal_velocity_policy}",
            flush=True,
        )
        self.active_waypoints = None
        self.active_times = None
        self.active_waypoint_roles = []
        self.active_horizon_gate_indices = []
        self.active_horizon_track_ids = []
        self.active_horizon_targets = []
        self.active_plan_mode = ""
        self.active_terminal_velocity = np.zeros(3, dtype=float)
        self.active_terminal_velocity_policy = f"cleared:{reason}"
        self.post_gate_exit_until_s = 0.0
        self.post_gate_exit_reason = ""
        self._reset_reference_progress_state()

    def _trace_spline_memory(
        self,
        *,
        action: str,
        gate_idx: int,
        track_id,
        center,
        context: str,
        **fields,
    ) -> None:
        center = self._finite_vec3_or_none(center)
        rounded_center = (
            None
            if center is None
            else tuple(round(float(value), 2) for value in center)
        )
        signature = (
            str(action),
            int(gate_idx),
            None if track_id is None else int(track_id),
            rounded_center,
            str(context),
            tuple(sorted((str(key), str(value)) for key, value in fields.items())),
        )
        if signature == self._last_spline_memory_signature:
            return
        self._last_spline_memory_signature = signature

        extra = ""
        for key, value in fields.items():
            if isinstance(value, np.ndarray):
                extra += f" {key}={self._fmt_vec(value, precision=3)}"
            elif isinstance(value, (list, tuple)) and len(value) >= 3:
                extra += f" {key}={self._fmt_vec(value, precision=3)}"
            elif isinstance(value, float):
                extra += f" {key}={self._fmt_float(value, precision=3)}"
            elif value is None:
                extra += f" {key}=none"
            else:
                extra += f" {key}={value}"
        print(
            "spline_memory "
            f"action={action} "
            f"context={context} "
            f"gate_idx={int(gate_idx)} "
            f"track={track_id if track_id is not None else 'none'} "
            f"center_neu={self._fmt_vec(center, precision=3)}"
            f"{extra}",
            flush=True,
        )

    def _spline_memory_track_quality_ok(
        self,
        track_id,
        *,
        require_stable: bool,
        require_fresh: bool,
    ) -> tuple[bool, str]:
        try:
            track_id = None if track_id is None else int(track_id)
        except (TypeError, ValueError):
            return False, "invalid_track_id"
        if track_id is None or track_id < 0:
            return True, "synthetic_track"
        if not hasattr(self, "gate_memory"):
            return False, "missing_gate_memory"
        track = self.gate_memory.get_track_by_id(track_id)
        if track is None:
            return False, "missing_track"

        committed = bool(getattr(track, "committed", False))
        stable = bool(getattr(track, "is_stable", False))
        ever_stable = bool(getattr(track, "ever_stable", False))
        if require_stable and not (stable or ever_stable):
            return False, "not_stable"
        if not committed and not stable and not ever_stable:
            return False, "uncommitted_unstable"

        min_hits = max(
            1,
            int(self.gate_memory.min_hits_for_stable)
            if require_stable
            else min(int(self.gate_memory.commit_hits), int(self.gate_memory.min_hits_for_stable)),
        )
        inliers = int(getattr(track, "inlier_count", getattr(track, "hits", 0)))
        hits = int(getattr(track, "hits", 0))
        if max(inliers, hits) < min_hits:
            return False, "insufficient_hits"

        obs_history = getattr(track, "obs_history", [])
        if not obs_history:
            return False, "missing_observation"
        last_obs = obs_history[-1]
        if bool(getattr(last_obs, "is_outlier", False)):
            return False, "last_observation_outlier"
        if not bool(getattr(last_obs, "quality_ok", True)):
            return False, "last_observation_bad_quality"

        if require_fresh:
            last_seen = self._finite_float(getattr(track, "last_seen_time", 0.0), 0.0)
            stale_time = max(
                self._finite_float(getattr(self.gate_memory, "stale_time", 0.0), 0.0),
                float(self.provisional_next_gate_max_age_s),
            )
            age_s = time.time() - last_seen if last_seen > 0.0 else float("inf")
            if stale_time > 0.0 and (not math.isfinite(age_s) or age_s > stale_time):
                return False, "stale"

        reproj = self._finite_float(
            getattr(last_obs, "reprojection_error", float("nan")),
            float("nan"),
            allow_nan=True,
        )
        max_reproj = float(self.gate_memory.max_reprojection_error_for_stable)
        if max_reproj > 0.0 and (
            not math.isfinite(reproj) or reproj > max_reproj
        ):
            return False, "reprojection_error_high"

        kp_min = self._finite_float(
            getattr(last_obs, "keypoint_conf_min", float("nan")),
            float("nan"),
            allow_nan=True,
        )
        min_kp = float(self.gate_memory.min_keypoint_conf_for_stable)
        if min_kp > 0.0 and (not math.isfinite(kp_min) or kp_min < min_kp):
            return False, "keypoint_conf_low"

        world_std = self._finite_vec3_or_none(getattr(track, "center_world_std", None))
        if world_std is not None:
            world_std_norm = float(np.linalg.norm(world_std))
            max_world_std = float(self.gate_memory.max_center_std_for_stable)
            if max_world_std > 0.0 and (
                not math.isfinite(world_std_norm) or world_std_norm > max_world_std
            ):
                return False, "world_std_high"

        return True, "ok"

    def _spline_memory_record_allowed(
        self,
        *,
        source: str,
        track_id,
        is_current_target: bool = False,
    ) -> tuple[bool, str]:
        source = str(source)
        if source == "planning_horizon_candidate" or (
            source == "gate_horizon" and not bool(is_current_target)
        ):
            return self._spline_memory_track_quality_ok(
                track_id,
                require_stable=True,
                require_fresh=True,
            )
        return True, "installed_plan"

    def _spline_memory_live_override(
        self,
        *,
        gate_idx: int,
        memory_target: np.ndarray,
        memory_track_id,
        selected_target,
        selected_track_id,
        context: str,
    ) -> tuple[bool, np.ndarray | None, float]:
        selected = self._finite_vec3_or_none(selected_target)
        if selected is None:
            return False, None, float("inf")
        distance = float(np.linalg.norm(memory_target - selected))
        threshold = float(self.spline_memory_live_override_m)
        if threshold <= 0.0 or not math.isfinite(distance) or distance <= threshold:
            return False, selected, distance

        ok, reason = self._spline_memory_track_quality_ok(
            selected_track_id,
            require_stable=True,
            require_fresh=True,
        )
        if not ok:
            self._trace_spline_memory(
                action="keep",
                gate_idx=int(gate_idx),
                track_id=memory_track_id,
                center=memory_target,
                context=f"{context}:live_not_durable:{reason}",
                selected_track=(
                    "none" if selected_track_id is None else str(selected_track_id)
                ),
                selected_neu=selected,
                selected_dist_m=distance,
            )
            return False, selected, distance

        self._trace_spline_memory(
            action="yield",
            gate_idx=int(gate_idx),
            track_id=memory_track_id,
            center=memory_target,
            context=f"{context}:live_override",
            selected_track=("none" if selected_track_id is None else str(selected_track_id)),
            selected_neu=selected,
            selected_dist_m=distance,
        )
        self._clear_spline_memory_for_gate(
            int(gate_idx),
            reason=f"{context}_live_override",
            track_id=memory_track_id,
        )
        return True, selected.copy(), distance

    def _record_spline_memory_from_active_plan(self, *, source: str) -> None:
        if self.active_waypoints is None or not self.active_horizon_gate_indices:
            return
        now = time.time()
        for horizon_idx, gate_idx in enumerate(self.active_horizon_gate_indices):
            try:
                gate_idx = int(gate_idx)
            except (TypeError, ValueError):
                continue
            if gate_idx < int(self.current_gate_idx):
                continue
            target = (
                self.active_horizon_targets[horizon_idx]
                if horizon_idx < len(self.active_horizon_targets)
                else None
            )
            target = self._finite_vec3_or_none(target)
            if target is None:
                continue
            track_id = (
                self.active_horizon_track_ids[horizon_idx]
                if horizon_idx < len(self.active_horizon_track_ids)
                else None
            )
            try:
                track_id = None if track_id is None else int(track_id)
            except (TypeError, ValueError):
                track_id = None
            if track_id is not None and track_id in self.completed_track_ids:
                continue
            allowed, reason = self._spline_memory_record_allowed(
                source=str(source),
                track_id=track_id,
                is_current_target=(
                    int(horizon_idx) == 0 or int(gate_idx) == int(self.current_gate_idx)
                ),
            )
            if not allowed:
                self._trace_spline_memory(
                    action="skip_record",
                    gate_idx=int(gate_idx),
                    track_id=track_id,
                    center=target,
                    context=f"{source}:{reason}",
                    horizon_idx=int(horizon_idx),
                )
                continue
            record = {
                "gate_idx": int(gate_idx),
                "track_id": track_id,
                "center": target.copy(),
                "plan_generation": int(self.active_plan_generation),
                "plan_mode": str(self.active_plan_mode),
                "source": str(source),
                "validated_plan": bool(str(source) != "planning_horizon_candidate"),
                "created_time": float(now),
                "last_used_time": float(now),
                "completed": False,
            }
            self.spline_memory_by_gate_idx[int(gate_idx)] = record
            self._trace_spline_memory(
                action="record",
                gate_idx=int(gate_idx),
                track_id=track_id,
                center=target,
                context=str(source),
                plan_generation=int(self.active_plan_generation),
                plan_mode=str(self.active_plan_mode),
                horizon_idx=int(horizon_idx),
            )

    def _spline_memory_for_gate(
        self,
        gate_idx: int,
        track_id=None,
        *,
        validate_target: bool = True,
    ) -> tuple[np.ndarray, int | None, dict] | tuple[None, None, None]:
        try:
            gate_idx = int(gate_idx)
        except (TypeError, ValueError):
            return None, None, None
        record = self.spline_memory_by_gate_idx.get(gate_idx)
        if record is None or bool(record.get("completed", False)):
            return None, None, None
        target = self._finite_vec3_or_none(record.get("center"))
        if target is None:
            self.spline_memory_by_gate_idx.pop(gate_idx, None)
            return None, None, None
        memory_track_id = record.get("track_id")
        try:
            memory_track_id = None if memory_track_id is None else int(memory_track_id)
        except (TypeError, ValueError):
            memory_track_id = None
        if track_id is not None and memory_track_id is not None:
            try:
                if int(track_id) != int(memory_track_id):
                    return None, None, None
            except (TypeError, ValueError):
                return None, None, None
        if memory_track_id is not None and memory_track_id in self.completed_track_ids:
            self.spline_memory_by_gate_idx.pop(gate_idx, None)
            self._trace_spline_memory(
                action="clear",
                gate_idx=gate_idx,
                track_id=memory_track_id,
                center=target,
                context="completed_track",
            )
            return None, None, None
        source = str(record.get("source", ""))
        if source == "planning_horizon_candidate":
            allowed, reason = self._spline_memory_track_quality_ok(
                track_id=memory_track_id,
                require_stable=True,
                require_fresh=False,
            )
            if not allowed:
                self._trace_spline_memory(
                    action="reject",
                    gate_idx=gate_idx,
                    track_id=memory_track_id,
                    center=target,
                    context=f"weak_candidate_memory:{reason}",
                )
                self.spline_memory_by_gate_idx.pop(gate_idx, None)
                return None, None, None
        if validate_target:
            reason = self._target_rejection_reason(target, memory_track_id)
            if reason:
                self._trace_spline_memory(
                    action="reject",
                    gate_idx=gate_idx,
                    track_id=memory_track_id,
                    center=target,
                    context=reason,
                )
                if reason in (
                    "completed_track_id",
                    "duplicate_of_completed_landmark",
                    "completed_segment_corridor",
                ):
                    self.spline_memory_by_gate_idx.pop(gate_idx, None)
                return None, None, None
        record["last_used_time"] = float(time.time())
        return target.copy(), memory_track_id, record

    def _spline_memory_for_track_id(
        self,
        track_id,
        *,
        validate_target: bool = True,
    ) -> tuple[np.ndarray, int | None, int, dict] | tuple[None, None, None, None]:
        try:
            track_id = int(track_id)
        except (TypeError, ValueError):
            return None, None, None, None

        best = None
        now = time.time()
        max_age = max(
            float(self.provisional_next_gate_max_duration_s),
            self._finite_float(
                getattr(self.gate_memory, "stale_time", 0.0),
                0.0,
            ),
        )
        for gate_idx, record in self.spline_memory_by_gate_idx.items():
            if bool(record.get("completed", False)):
                continue
            memory_track_id = record.get("track_id")
            try:
                memory_track_id = None if memory_track_id is None else int(memory_track_id)
            except (TypeError, ValueError):
                memory_track_id = None
            if memory_track_id != track_id:
                continue
            created_time = self._finite_float(record.get("created_time", 0.0), 0.0)
            age_s = now - created_time if created_time > 0.0 else float("inf")
            if max_age > 0.0 and (not math.isfinite(age_s) or age_s > max_age):
                continue
            target, memory_track_id, checked_record = self._spline_memory_for_gate(
                int(gate_idx),
                track_id,
                validate_target=validate_target,
            )
            if target is None:
                continue
            key = (
                abs(int(gate_idx) - int(self.current_gate_idx)),
                int(gate_idx),
            )
            if best is None or key < best[0]:
                best = (key, target.copy(), memory_track_id, int(gate_idx), checked_record)

        if best is None:
            return None, None, None, None
        return best[1], best[2], best[3], best[4]

    def _record_spline_memory_candidates(
        self,
        *,
        gate_indices,
        track_ids,
        targets,
        source: str,
        skip_current: bool,
    ) -> None:
        if not gate_indices or not targets:
            return
        now = time.time()
        for horizon_idx, gate_idx in enumerate(gate_indices):
            try:
                gate_idx = int(gate_idx)
            except (TypeError, ValueError):
                continue
            if skip_current and gate_idx <= int(self.current_gate_idx):
                continue
            if gate_idx < int(self.current_gate_idx):
                continue
            target = (
                targets[horizon_idx]
                if horizon_idx < len(targets)
                else None
            )
            target = self._finite_vec3_or_none(target)
            if target is None:
                continue
            track_id = (
                track_ids[horizon_idx]
                if horizon_idx < len(track_ids)
                else None
            )
            try:
                track_id = None if track_id is None else int(track_id)
            except (TypeError, ValueError):
                track_id = None
            if track_id is not None and track_id in self.completed_track_ids:
                continue
            allowed, reason = self._spline_memory_record_allowed(
                source=str(source),
                track_id=track_id,
            )
            if not allowed:
                self._trace_spline_memory(
                    action="skip_record",
                    gate_idx=int(gate_idx),
                    track_id=track_id,
                    center=target,
                    context=f"{source}:{reason}",
                    plan_generation=int(self.active_plan_generation),
                    plan_mode=str(self.active_plan_mode or "candidate"),
                    horizon_idx=int(horizon_idx),
                )
                continue

            existing = self.spline_memory_by_gate_idx.get(gate_idx)
            existing_created = (
                self._finite_float(existing.get("created_time", 0.0), 0.0)
                if existing is not None
                else now
            )
            record = {
                "gate_idx": int(gate_idx),
                "track_id": track_id,
                "center": target.copy(),
                "plan_generation": int(self.active_plan_generation),
                "plan_mode": str(self.active_plan_mode or "candidate"),
                "source": str(source),
                "validated_plan": bool(str(source) != "planning_horizon_candidate"),
                "created_time": float(existing_created if existing is not None else now),
                "last_used_time": float(now),
                "completed": False,
            }
            self.spline_memory_by_gate_idx[int(gate_idx)] = record
            self._trace_spline_memory(
                action="record",
                gate_idx=int(gate_idx),
                track_id=track_id,
                center=target,
                context=str(source),
                plan_generation=int(self.active_plan_generation),
                plan_mode=str(self.active_plan_mode or "candidate"),
                horizon_idx=int(horizon_idx),
            )

    def _clear_spline_memory_for_gate(
        self,
        gate_idx: int,
        *,
        reason: str,
        track_id=None,
    ) -> None:
        try:
            gate_idx = int(gate_idx)
        except (TypeError, ValueError):
            return
        record = self.spline_memory_by_gate_idx.pop(gate_idx, None)
        if record is None:
            return
        memory_track_id = record.get("track_id", track_id)
        center = self._finite_vec3_or_none(record.get("center"))
        self._trace_spline_memory(
            action="clear",
            gate_idx=gate_idx,
            track_id=memory_track_id,
            center=center,
            context=str(reason),
        )

    def _spline_memory_override_target(
        self,
        *,
        gate_idx: int,
        selected_target,
        selected_track_id,
        context: str,
    ) -> tuple[np.ndarray | None, int | None, bool]:
        memory_target, memory_track_id, _ = self._spline_memory_for_gate(gate_idx)
        if memory_target is None:
            target = self._finite_vec3_or_none(selected_target)
            return target, selected_track_id, False

        selected = self._finite_vec3_or_none(selected_target)
        yielded, live_target, distance = self._spline_memory_live_override(
            gate_idx=int(gate_idx),
            memory_target=memory_target,
            memory_track_id=memory_track_id,
            selected_target=selected,
            selected_track_id=selected_track_id,
            context=str(context),
        )
        if yielded:
            return live_target, selected_track_id, False
        selected_track_text = "none"
        try:
            selected_track_text = (
                "none" if selected_track_id is None else str(int(selected_track_id))
            )
        except (TypeError, ValueError):
            selected_track_text = str(selected_track_id)
        self._trace_spline_memory(
            action="use",
            gate_idx=int(gate_idx),
            track_id=memory_track_id,
            center=memory_target,
            context=str(context),
            selected_track=selected_track_text,
            selected_neu=selected,
            selected_dist_m=distance,
        )
        return memory_target.copy(), memory_track_id, True

    def _apply_spline_memory_to_gate_entries(
        self,
        gates: list[np.ndarray],
        track_ids: list,
    ) -> tuple[list[np.ndarray], list]:
        gate_idx = int(self.current_gate_idx)
        memory_target, memory_track_id, _ = self._spline_memory_for_gate(gate_idx)
        if memory_target is None:
            return gates, track_ids

        out_gates = [
            np.asarray(gate, dtype=float).reshape(3).copy()
            for gate in gates
        ]
        out_track_ids = list(track_ids)
        if len(out_track_ids) < len(out_gates):
            out_track_ids.extend([None] * (len(out_gates) - len(out_track_ids)))

        if gate_idx > len(out_gates):
            return out_gates, out_track_ids

        selected = out_gates[gate_idx] if gate_idx < len(out_gates) else None
        selected_track_id = out_track_ids[gate_idx] if gate_idx < len(out_track_ids) else None
        yielded, live_target, distance = self._spline_memory_live_override(
            gate_idx=int(gate_idx),
            memory_target=memory_target,
            memory_track_id=memory_track_id,
            selected_target=selected,
            selected_track_id=selected_track_id,
            context="install_gate_centers",
        )
        if yielded:
            if gate_idx < len(out_gates):
                out_gates[gate_idx] = live_target.copy()
                out_track_ids[gate_idx] = selected_track_id
            return out_gates, out_track_ids
        if gate_idx == len(out_gates):
            out_gates.append(memory_target.copy())
            out_track_ids.append(memory_track_id)
        else:
            out_gates[gate_idx] = memory_target.copy()
            out_track_ids[gate_idx] = memory_track_id

        self._trace_spline_memory(
            action="inject",
            gate_idx=gate_idx,
            track_id=memory_track_id,
            center=memory_target,
            context="install_gate_centers",
            selected_track=(
                "none"
                if selected_track_id is None
                else str(selected_track_id)
            ),
            selected_neu=selected,
            selected_dist_m=distance,
        )
        return out_gates, out_track_ids

    @staticmethod
    def _finite_vec3_or_none(value) -> np.ndarray | None:
        try:
            arr = np.asarray(value, dtype=float).reshape(3)
        except (TypeError, ValueError):
            return None
        if not np.all(np.isfinite(arr)):
            return None
        return arr.copy()

    def _track_filtered_center_for_navigation(
        self,
        track,
    ) -> tuple[np.ndarray | None, dict]:
        details = {
            "ok": False,
            "reason": "missing_track",
            "reproj": float("nan"),
            "kp_min": float("nan"),
            "world_std": float("nan"),
            "age_s": float("nan"),
            "inliers": int(getattr(track, "inlier_count", 0)) if track is not None else 0,
            "stable": bool(getattr(track, "is_stable", False)) if track is not None else False,
        }
        if track is None or not bool(getattr(track, "committed", False)):
            return None, details

        center = self._finite_vec3_or_none(
            getattr(track, "filtered_center_world", None)
        )
        if center is None:
            details["reason"] = "missing_filtered_center"
            return None, details

        ever_stable = bool(getattr(track, "ever_stable", False))
        details["ever_stable"] = ever_stable
        if not bool(getattr(track, "is_stable", False)) and not ever_stable:
            details["reason"] = str(getattr(track, "promotion_blocked_reason", "")) or "unstable"
            return None, details

        min_inliers = max(1, int(getattr(self.gate_memory, "min_hits_for_stable", 1)))
        inliers = int(getattr(track, "inlier_count", 0))
        details["inliers"] = inliers
        if inliers < min_inliers:
            details["reason"] = "insufficient_inliers"
            return None, details

        last_seen = self._finite_float(getattr(track, "last_seen_time", 0.0), 0.0)
        stale_time = self._finite_float(getattr(self.gate_memory, "stale_time", 0.5), 0.5)
        age_s = time.time() - last_seen if last_seen > 0.0 else float("inf")
        details["age_s"] = age_s
        if last_seen <= 0.0 or age_s > stale_time:
            details["reason"] = "stale"
            return None, details

        world_std = getattr(track, "center_world_std", None)
        world_std_arr = self._finite_vec3_or_none(world_std)
        world_std_norm = (
            float(np.linalg.norm(world_std_arr))
            if world_std_arr is not None
            else float("inf")
        )
        details["world_std"] = world_std_norm
        max_world_std = float(self.active_target_shift_max_world_std_m)
        if max_world_std > 0.0 and (
            not math.isfinite(world_std_norm) or world_std_norm > max_world_std
        ):
            details["reason"] = "world_std_high"
            return None, details

        obs_history = getattr(track, "obs_history", [])
        if not obs_history:
            details["reason"] = "missing_observation"
            return None, details
        last_obs = obs_history[-1]
        if bool(getattr(last_obs, "is_outlier", False)):
            details["last_observation_outlier"] = True
        if bool(getattr(last_obs, "is_outlier", False)) and not ever_stable:
            details["reason"] = "last_observation_outlier"
            return None, details

        reproj = self._finite_float(
            getattr(last_obs, "reprojection_error", float("nan")),
            float("nan"),
            allow_nan=True,
        )
        details["reproj"] = reproj
        max_reproj = float(self.active_target_shift_max_reprojection_error)
        if max_reproj > 0.0 and (
            not math.isfinite(reproj) or reproj > max_reproj
        ):
            details["reason"] = "reprojection_error_high"
            return None, details

        kp_min = self._finite_float(
            getattr(last_obs, "keypoint_conf_min", float("nan")),
            float("nan"),
            allow_nan=True,
        )
        details["kp_min"] = kp_min
        min_kp = float(self.active_target_shift_min_keypoint_conf)
        if min_kp > 0.0 and (not math.isfinite(kp_min) or kp_min < min_kp):
            details["reason"] = "keypoint_conf_low"
            return None, details

        details["ok"] = True
        details["reason"] = "ok"
        return center, details

    def _active_track_filtered_center(self, track) -> np.ndarray | None:
        center, _ = self._track_filtered_center_for_navigation(track)
        return center

    def _track_navigation_center(self, track) -> np.ndarray | None:
        center, _ = self._track_filtered_center_for_navigation(track)
        if center is not None:
            return center
        return self._finite_vec3_or_none(getattr(track, "center", None))

    def _maybe_apply_active_target_shift(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
    ) -> bool:
        if (
            not self.active_target_shift_enabled
            or not self.use_perception
            or self.active_waypoints is None
            or self.planner.total_time <= 0.0
            or self.gate_plane_crossed
        ):
            return False

        diag = self.target_manager.diagnostics()
        if not diag.locked or diag.active_track_id is None or diag.center_at_plan is None:
            self.active_target_shift_frames = 0
            self.active_target_shift_pending_kind = None
            self.active_target_shift_track_id = None
            return False

        try:
            active_id = int(diag.active_track_id)
        except (TypeError, ValueError):
            self.active_target_shift_frames = 0
            self.active_target_shift_pending_kind = None
            self.active_target_shift_track_id = None
            return False
        if active_id < 0:
            self.active_target_shift_frames = 0
            self.active_target_shift_pending_kind = None
            self.active_target_shift_track_id = None
            return False

        if self.active_target_shift_track_id != active_id:
            self.active_target_shift_frames = 0
            self.active_target_shift_pending_kind = None
            self.active_target_shift_track_id = active_id
            self._reset_deferred_longitudinal_shift()

        planned = self._finite_vec3_or_none(diag.center_at_plan)
        if planned is None:
            self.active_target_shift_frames = 0
            self.active_target_shift_pending_kind = None
            return False

        planned = self._apply_target_z_policy(planned)

        latest, source_track_id, quality = self._best_duplicate_cluster_center(active_id)
        if latest is None or not bool(quality.get("ok", False)):
            self.active_target_shift_frames = 0
            self.active_target_shift_pending_kind = None
            if self._maybe_apply_deferred_longitudinal_exit_shift(
                pos=pos,
                vel=vel,
                gate_idx=int(self.current_gate_idx),
                track_id=active_id,
                target=planned,
            ):
                return True
            return False

        latest = self._apply_target_z_policy(latest)
        shift_vec = latest - planned
        shift_m = float(np.linalg.norm(shift_vec))
        shift_xy_m = float(np.linalg.norm(shift_vec[:2]))
        shift_z_m = float(abs(shift_vec[2]))
        if not math.isfinite(shift_m) or shift_m < self.active_target_shift_threshold_m:
            self.active_target_shift_frames = 0
            self.active_target_shift_pending_kind = None
            if self._maybe_apply_deferred_longitudinal_exit_shift(
                pos=pos,
                vel=vel,
                gate_idx=int(self.current_gate_idx),
                track_id=active_id,
                target=planned,
            ):
                return True
            return False
        max_total_m = float(self.active_target_shift_max_total_m)
        if max_total_m > 0.0 and shift_m > max_total_m:
            self.active_target_shift_frames = 0
            self.active_target_shift_pending_kind = None
            return False

        dist_to_target = float(np.linalg.norm(np.asarray(pos, dtype=float).reshape(3) - planned))

        longitudinal_axis = self._gate_corridor_axis(
            center=planned,
            reference=np.asarray(pos, dtype=float).reshape(3),
            preferred_normal=self.active_gate_normal,
        )
        longitudinal_m = float("nan")
        lateral_m = float("nan")
        mostly_longitudinal = False
        if longitudinal_axis is not None:
            longitudinal_m = float(np.dot(shift_vec, longitudinal_axis))
            lateral_vec = shift_vec - longitudinal_m * longitudinal_axis
            lateral_m = float(np.linalg.norm(lateral_vec))
            mostly_longitudinal = (
                self.active_target_shift_defer_longitudinal_enabled
                and abs(longitudinal_m)
                >= float(self.active_target_shift_longitudinal_min_m)
                and math.isfinite(lateral_m)
                and lateral_m
                < float(self.active_target_shift_longitudinal_lateral_max_m)
            )

        if mostly_longitudinal:
            if self.active_target_shift_pending_kind != "longitudinal_deferred":
                self.active_target_shift_frames = 0
                self.active_target_shift_pending_kind = "longitudinal_deferred"
            self.active_target_shift_frames += 1
            if self.active_target_shift_frames < self.active_target_shift_required_frames:
                return False
            self._record_deferred_longitudinal_shift(
                gate_idx=int(self.current_gate_idx),
                track_id=active_id,
                axis=longitudinal_axis,
                longitudinal_m=longitudinal_m,
                lateral_m=lateral_m,
                shift_m=shift_m,
                shift_z_m=shift_z_m,
                planned=planned,
                latest=latest,
                source_track_id=source_track_id,
                quality=quality,
            )
            self.active_target_shift_frames = 0
            self.active_target_shift_pending_kind = None
            return self._maybe_apply_deferred_longitudinal_exit_shift(
                pos=pos,
                vel=vel,
                gate_idx=int(self.current_gate_idx),
                track_id=active_id,
                target=planned,
            )

        if self._maybe_apply_deferred_longitudinal_exit_shift(
            pos=pos,
            vel=vel,
            gate_idx=int(self.current_gate_idx),
            track_id=active_id,
            target=planned,
        ):
            return True

        if dist_to_target <= self.active_target_shift_near_gate_distance_m:
            self.active_target_shift_frames = 0
            self.active_target_shift_pending_kind = None
            return False

        if self.active_target_shift_pending_kind != "active_correction":
            self.active_target_shift_frames = 0
            self.active_target_shift_pending_kind = "active_correction"
        self.active_target_shift_frames += 1
        if self.active_target_shift_frames < self.active_target_shift_required_frames:
            return False

        now = time.time()
        if now - self.last_plan_wall_time < self.active_target_shift_replan_min_interval_s:
            return False

        if self.active_target_shift_alpha <= 0.0:
            self.active_target_shift_frames = 0
            self.active_target_shift_pending_kind = None
            return False

        correction_vec = self.active_target_shift_alpha * shift_vec
        correction_m = float(np.linalg.norm(correction_vec))
        max_step_m = float(self.active_target_shift_max_step_m)
        if max_step_m > 0.0 and correction_m > max_step_m:
            correction_vec *= max_step_m / correction_m
            correction_m = max_step_m
        corrected = self._apply_target_z_policy(
            planned + correction_vec
        )
        target_idx = int(self.current_gate_idx)
        if target_idx < 0 or target_idx >= len(self.gate_centers_neu):
            self.active_target_shift_frames = 0
            self.active_target_shift_pending_kind = None
            return False

        print(
            "active_target_shift correction "
            f"gate_idx={target_idx} "
            f"track={active_id} "
            f"shift={shift_m:.2f} "
            f"shift_xy={shift_xy_m:.2f} "
            f"shift_z={shift_z_m:.2f} "
            f"shift_longitudinal_lateral=({self._fmt_float(longitudinal_m, precision=2)},"
            f"{self._fmt_float(lateral_m, precision=2)}) "
            f"step={correction_m:.2f} "
            f"frames={self.active_target_shift_frames} "
            f"dist_to_target={dist_to_target:.2f} "
            f"alpha={self.active_target_shift_alpha:.2f} "
            f"source_track={source_track_id if source_track_id is not None else 'none'} "
            f"cluster={quality.get('cluster_ids', ())} "
            f"reproj={self._fmt_float(quality.get('reproj'), precision=2)} "
            f"kp_min={self._fmt_float(quality.get('kp_min'), precision=2)} "
            f"world_std={self._fmt_float(quality.get('world_std'), precision=2)} "
            f"inliers={int(quality.get('inliers', 0))} "
            f"planned={self._fmt_vec(planned, precision=3)} "
            f"latest={self._fmt_vec(latest, precision=3)} "
            f"corrected={self._fmt_vec(corrected, precision=3)}",
            flush=True,
        )

        self.gate_centers_neu[target_idx] = corrected.copy()
        self.current_gate_pos = corrected.copy()
        self.last_active_target_center = latest.copy()
        self._reset_gate_pass_state()
        self.active_waypoints = None
        self.active_times = None
        self.active_waypoint_roles = []
        self.active_horizon_gate_indices = []
        self.active_horizon_track_ids = []
        self.active_horizon_targets = []
        self.active_plan_mode = ""
        self.active_terminal_velocity = np.zeros(3, dtype=float)
        self.active_terminal_velocity_policy = "cleared_active_shift"
        self.last_planned_gate_idx = -1
        self._last_gate_signature = None
        self.active_target_shift_frames = 0
        self.active_target_shift_pending_kind = None
        return self._path_plan(pos, vel)

    def _planning_horizon_targets(
        self,
        target_idx: int,
        target: np.ndarray,
        target_track_id,
        pos: np.ndarray | None = None,
        vel: np.ndarray | None = None,
    ) -> tuple[list[np.ndarray], list, list[int]]:
        targets = [np.asarray(target, dtype=float).reshape(3).copy()]
        track_ids = [target_track_id]
        gate_indices = [int(target_idx)]
        max_idx = int(target_idx) + int(self.planning_horizon_gates)
        if self.race_gate_count is not None:
            max_idx = min(max_idx, int(self.race_gate_count))
        elif len(self.gate_centers_neu) > 0:
            max_idx = min(max_idx, len(self.gate_centers_neu))
        for idx in range(int(target_idx) + 1, max_idx):
            selected_center = (
                self.gate_centers_neu[idx]
                if idx < len(self.gate_centers_neu)
                else None
            )
            selected_track_id = (
                self.gate_track_ids[idx] if idx < len(self.gate_track_ids) else None
            )
            center, track_id, _ = self._spline_memory_override_target(
                gate_idx=int(idx),
                selected_target=selected_center,
                selected_track_id=selected_track_id,
                context="planning_horizon",
            )
            if center is None:
                continue
            center = self._apply_target_z_policy(center)
            if not np.all(np.isfinite(center)):
                continue
            if track_id is not None and int(track_id) in self.completed_track_ids:
                continue
            if self.use_perception:
                reject_reason = self._target_rejection_reason(center, track_id)
                if reject_reason:
                    self._trace_target_rejection(
                        reason=reject_reason,
                        track_id=track_id,
                        center=center,
                        context="planning_horizon",
                    )
                    continue
            duplicate = self._duplicate_center_match(center, targets)
            if duplicate is not None:
                duplicate_idx, dist = duplicate
                duplicate_track = (
                    track_ids[duplicate_idx]
                    if duplicate_idx < len(track_ids)
                    else None
                )
                self._trace_target_rejection(
                    reason=f"duplicate_horizon_target:{float(dist):.2f}",
                    track_id=track_id,
                    center=center,
                    context=(
                        "planning_horizon:"
                        f"near={duplicate_track if duplicate_track is not None else 'none'}"
                    ),
                )
                continue
            targets.append(center.copy())
            track_ids.append(track_id)
            gate_indices.append(int(idx))
        self._append_provisional_horizon_targets(
            pos=pos,
            vel=vel,
            targets=targets,
            track_ids=track_ids,
            gate_indices=gate_indices,
        )
        return targets, track_ids, gate_indices

    def _append_provisional_horizon_targets(
        self,
        *,
        pos: np.ndarray | None,
        vel: np.ndarray | None,
        targets: list[np.ndarray],
        track_ids: list,
        gate_indices: list[int],
    ) -> None:
        if (
            not self.use_perception
            or not self.provisional_next_gate_enabled
            or pos is None
            or vel is None
            or not targets
            or len(targets) >= int(self.planning_horizon_gates)
            or not hasattr(self, "gate_memory")
        ):
            return

        if self.race_gate_count is not None and gate_indices:
            if int(gate_indices[-1]) >= int(self.race_gate_count) - 1:
                return

        pos = self._finite_vec3_or_none(pos)
        vel = self._finite_vec3_or_none(vel)
        if pos is None or vel is None:
            return

        accepted: list[tuple[int, int, float, float]] = []
        rejected: list[tuple[int, str, float]] = []
        now = time.time()

        def track_id_set() -> set[int]:
            out = set()
            for track_id in track_ids:
                if track_id is None:
                    continue
                try:
                    out.add(int(track_id))
                except (TypeError, ValueError):
                    continue
            return out

        def horizon_direction() -> np.ndarray | None:
            if len(targets) >= 2:
                direction = targets[-1] - targets[-2]
            else:
                direction = targets[-1] - pos
            norm = float(np.linalg.norm(direction))
            if math.isfinite(norm) and norm >= 1e-6:
                return direction / norm
            return self._provisional_forward_axis(pos, vel)

        while len(targets) < int(self.planning_horizon_gates):
            direction = horizon_direction()
            if direction is None:
                break

            anchor = np.asarray(targets[-1], dtype=float).reshape(3)
            anchor_projection = float(np.dot(anchor - pos, direction))
            existing_track_ids = track_id_set()
            best = None

            for track in sorted(self.gate_memory.tracks, key=lambda item: int(item.id)):
                track_id = int(track.id)
                distance_for_reject = float("nan")
                if track_id in existing_track_ids:
                    continue
                if track_id in self.completed_track_ids:
                    rejected.append((track_id, "completed_track", distance_for_reject))
                    continue

                center = self._provisional_track_center(track)
                if center is None:
                    rejected.append((track_id, "missing_center", distance_for_reject))
                    continue
                center = self._apply_target_z_policy(center)
                rel = center - pos
                distance = float(np.linalg.norm(rel))
                distance_for_reject = distance

                reject_reason = self._target_rejection_reason(center, track_id)
                if reject_reason:
                    rejected.append((track_id, reject_reason, distance_for_reject))
                    continue

                quality_ok, quality_reason, _ = self._provisional_track_quality(
                    track,
                    now,
                )
                if not quality_ok:
                    rejected.append((track_id, quality_reason, distance_for_reject))
                    continue

                duplicate = self._duplicate_center_match(center, targets)
                if duplicate is not None:
                    duplicate_idx, dist = duplicate
                    duplicate_track = (
                        track_ids[duplicate_idx]
                        if duplicate_idx < len(track_ids)
                        else None
                    )
                    rejected.append(
                        (
                            track_id,
                            "duplicate_horizon_target:"
                            f"{duplicate_track if duplicate_track is not None else 'none'}",
                            float(dist),
                        )
                    )
                    continue

                max_distance = float(self.provisional_next_gate_max_distance_m)
                if max_distance > 0.0 and (
                    not math.isfinite(distance) or distance > max_distance
                ):
                    rejected.append((track_id, "too_far", distance_for_reject))
                    continue

                projection = float(np.dot(rel, direction))
                if not math.isfinite(projection) or projection <= 0.50:
                    rejected.append((track_id, "not_in_front", distance_for_reject))
                    continue

                margin = max(0.50, float(self.provisional_next_gate_closer_margin_m))
                if projection <= anchor_projection + margin:
                    rejected.append((track_id, "not_beyond_horizon", distance_for_reject))
                    continue

                lateral_vec = rel - projection * direction
                lateral = float(np.linalg.norm(lateral_vec))
                max_lateral = float(self.provisional_next_gate_max_lateral_m)
                if max_lateral > 0.0 and (
                    not math.isfinite(lateral) or lateral > max_lateral
                ):
                    rejected.append((track_id, "lateral_high", distance_for_reject))
                    continue

                key = (
                    projection,
                    lateral,
                    distance,
                    -float(getattr(track, "hits", 0)),
                    float(track_id),
                )
                if best is None or key < best[0]:
                    best = (key, track_id, center.copy(), projection, lateral)

            if best is None:
                break

            _, track_id, center, projection, lateral = best
            next_gate_idx = (
                int(gate_indices[-1]) + 1
                if gate_indices
                else int(self.current_gate_idx) + len(targets)
            )
            if self.race_gate_count is not None:
                next_gate_idx = min(next_gate_idx, int(self.race_gate_count) - 1)
            targets.append(center.copy())
            track_ids.append(int(track_id))
            gate_indices.append(int(next_gate_idx))
            accepted.append((int(track_id), int(next_gate_idx), projection, lateral))

            if (
                self.race_gate_count is not None
                and next_gate_idx >= int(self.race_gate_count) - 1
            ):
                break

        self._trace_provisional_horizon_suffix(
            accepted=accepted,
            rejected=rejected,
            gate_idx=int(gate_indices[0] if gate_indices else self.current_gate_idx),
        )

    def _trace_provisional_horizon_suffix(
        self,
        *,
        accepted: list[tuple[int, int, float, float]],
        rejected: list[tuple[int, str, float]],
        gate_idx: int,
    ) -> None:
        if not accepted and not rejected:
            self._last_provisional_horizon_signature = None
            return

        signature = (
            tuple((track_id, gate_idx) for track_id, gate_idx, _, _ in accepted[:8]),
            tuple(
                (
                    track_id,
                    reason,
                    round(float(distance), 1) if math.isfinite(float(distance)) else "nan",
                )
                for track_id, reason, distance in rejected[:8]
            ),
        )
        if signature == self._last_provisional_horizon_signature:
            return
        self._last_provisional_horizon_signature = signature

        accepted_txt = (
            "none"
            if not accepted
            else "("
            + ",".join(
                f"{track_id}@{gate_idx}:proj={projection:.1f}:lat={lateral:.1f}"
                for track_id, gate_idx, projection, lateral in accepted[:8]
            )
            + ")"
        )
        rejected_txt = (
            "none"
            if not rejected
            else "("
            + ",".join(
                f"{track_id}:{reason}:dist={self._fmt_float(distance, precision=1)}"
                for track_id, reason, distance in rejected[:8]
            )
            + ")"
        )
        extra_rejected = "" if len(rejected) <= 8 else f" rejected_more={len(rejected) - 8}"
        print(
            "planning_horizon_provisional_suffix "
            f"gate_idx={int(gate_idx)} "
            f"accepted={accepted_txt} "
            f"rejected={rejected_txt}"
            f"{extra_rejected}",
            flush=True,
        )

    def _gate_corridor_axis(
        self,
        *,
        center: np.ndarray,
        reference: np.ndarray,
        preferred_normal: np.ndarray | None = None,
    ) -> np.ndarray | None:
        center = self._finite_vec3_or_none(center)
        reference = self._finite_vec3_or_none(reference)
        if center is None or reference is None:
            return None

        axis = self._finite_vec3_or_none(preferred_normal)
        if axis is not None:
            norm = float(np.linalg.norm(axis))
            if math.isfinite(norm) and norm > 1e-6:
                axis = axis / norm
            else:
                axis = None

        if axis is None:
            axis = unit_vector_from_to(reference, center)
        if axis is None:
            return None

        rel = center - reference
        rel_norm = float(np.linalg.norm(rel))
        if math.isfinite(rel_norm) and rel_norm > 1e-6 and float(np.dot(rel, axis)) < 0.0:
            axis = -axis
        return axis

    def _build_gate_corridor_waypoints(
        self,
        *,
        start: np.ndarray,
        targets,
        preferred_normals=None,
        exit_shift_by_target=None,
    ) -> tuple[np.ndarray, list[str]]:
        start = np.asarray(start, dtype=float).reshape(3)
        waypoints = [start.copy()]
        roles = ["start"]
        reference = start.copy()

        half_length = 0.5 * max(0.0, float(self.gate_corridor_length_m))
        use_corridor = bool(self.gate_corridor_enabled) and half_length > 1e-6
        preferred_normals = list(preferred_normals or [])
        exit_shift_by_target = list(exit_shift_by_target or [])

        for idx, raw_target in enumerate(targets):
            center = np.asarray(raw_target, dtype=float).reshape(3)
            normal = preferred_normals[idx] if idx < len(preferred_normals) else None
            axis = (
                self._gate_corridor_axis(
                    center=center,
                    reference=reference,
                    preferred_normal=normal,
                )
                if use_corridor
                else None
            )
            if axis is None:
                waypoints.append(center.copy())
                roles.append("gate_center")
                reference = center.copy()
                continue

            enter = center - axis * half_length
            exit_shift_m = (
                float(exit_shift_by_target[idx])
                if idx < len(exit_shift_by_target)
                else 0.0
            )
            if not math.isfinite(exit_shift_m):
                exit_shift_m = 0.0
            min_exit_after_center_m = max(float(VADR_TS_002.gate_depth_m), 0.25)
            exit_after_center_m = max(
                min_exit_after_center_m,
                half_length + exit_shift_m,
            )
            exit_point = center + axis * exit_after_center_m

            # If the current/reference point has already crossed the enter point,
            # do not send the vehicle backward just to line up.
            enter_progress = float(np.dot(enter - reference, axis))
            if math.isfinite(enter_progress) and enter_progress > 0.25:
                waypoints.append(enter)
                roles.append("gate_enter")

            waypoints.append(center.copy())
            roles.append("gate_center")
            waypoints.append(exit_point)
            roles.append("gate_exit")
            reference = exit_point

        return np.vstack(waypoints), roles

    def _reset_deferred_longitudinal_shift(self) -> None:
        self.deferred_longitudinal_shift_gate_idx = None
        self.deferred_longitudinal_shift_track_id = None
        self.deferred_longitudinal_shift_axis = None
        self.deferred_longitudinal_shift_samples = []
        self.deferred_longitudinal_shift_applied_generation = -1
        self.deferred_longitudinal_shift_pending_signature = None

    def _record_deferred_longitudinal_shift(
        self,
        *,
        gate_idx: int,
        track_id: int,
        axis: np.ndarray,
        longitudinal_m: float,
        lateral_m: float,
        shift_m: float,
        shift_z_m: float,
        planned: np.ndarray,
        latest: np.ndarray,
        source_track_id,
        quality: dict,
    ) -> float:
        if (
            self.deferred_longitudinal_shift_gate_idx != int(gate_idx)
            or self.deferred_longitudinal_shift_track_id != int(track_id)
        ):
            self.deferred_longitudinal_shift_gate_idx = int(gate_idx)
            self.deferred_longitudinal_shift_track_id = int(track_id)
            self.deferred_longitudinal_shift_samples = []
            self.deferred_longitudinal_shift_applied_generation = -1

        axis = np.asarray(axis, dtype=float).reshape(3)
        axis_norm = float(np.linalg.norm(axis))
        if math.isfinite(axis_norm) and axis_norm > 1e-6:
            self.deferred_longitudinal_shift_axis = axis / axis_norm

        self.deferred_longitudinal_shift_samples.append(float(longitudinal_m))
        self.deferred_longitudinal_shift_pending_signature = None
        max_samples = int(self.active_target_shift_longitudinal_filter_size)
        if len(self.deferred_longitudinal_shift_samples) > max_samples:
            self.deferred_longitudinal_shift_samples = (
                self.deferred_longitudinal_shift_samples[-max_samples:]
            )

        mean_m = float(np.mean(self.deferred_longitudinal_shift_samples))
        print(
            "active_target_shift_longitudinal_deferred "
            f"gate_idx={int(gate_idx)} "
            f"track={int(track_id)} "
            f"shift={float(shift_m):.2f} "
            f"longitudinal={float(longitudinal_m):.2f} "
            f"lateral={float(lateral_m):.2f} "
            f"shift_z={float(shift_z_m):.2f} "
            f"samples={len(self.deferred_longitudinal_shift_samples)} "
            f"mean={mean_m:.2f} "
            f"source_track={source_track_id if source_track_id is not None else 'none'} "
            f"reproj={self._fmt_float(quality.get('reproj'), precision=2)} "
            f"kp_min={self._fmt_float(quality.get('kp_min'), precision=2)} "
            f"world_std={self._fmt_float(quality.get('world_std'), precision=2)} "
            f"planned={self._fmt_vec(planned, precision=3)} "
            f"latest={self._fmt_vec(latest, precision=3)} "
            f"axis={self._fmt_vec(self.deferred_longitudinal_shift_axis, precision=3)}",
            flush=True,
        )
        return mean_m

    def _active_corridor_enter_status(
        self,
        pos: np.ndarray,
    ) -> tuple[bool, str, np.ndarray | None, float, float]:
        if self.active_waypoints is None or self.planner.total_time <= 0.0:
            return False, "no_active_plan", None, float("nan"), float("nan")
        roles = list(self.active_waypoint_roles or [])
        if not roles or len(roles) != len(self.active_waypoints):
            return False, "missing_roles", None, float("nan"), float("nan")
        if "gate_enter" not in roles:
            if "gate_center" in roles and "gate_exit" in roles:
                return True, "enter_absent_already_past", None, float("nan"), float("nan")
            if "gate_center" in roles:
                center_idx = roles.index("gate_center")
                if 0 <= center_idx < len(self.active_waypoints):
                    center = np.asarray(
                        self.active_waypoints[center_idx],
                        dtype=float,
                    ).reshape(3)
                    pos = np.asarray(pos, dtype=float).reshape(3)
                    distance_m = float(np.linalg.norm(pos - center))
                    progress_m = float("nan")
                    axis = self._finite_vec3_or_none(self.active_gate_normal)
                    if axis is not None:
                        axis_norm = float(np.linalg.norm(axis))
                        if math.isfinite(axis_norm) and axis_norm > 1e-6:
                            progress_m = float(np.dot(pos - center, axis / axis_norm))
                    close_radius_m = max(
                        float(self.active_target_shift_longitudinal_enter_radius_m),
                        float(self.active_target_shift_near_gate_distance_m),
                    )
                    if math.isfinite(distance_m) and distance_m <= close_radius_m:
                        return (
                            True,
                            "single_gate_close_to_center",
                            center,
                            distance_m,
                            progress_m,
                        )
            return False, "no_corridor_enter", None, float("nan"), float("nan")

        enter_idx = roles.index("gate_enter")
        if enter_idx < 0 or enter_idx >= len(self.active_waypoints):
            return False, "bad_enter_idx", None, float("nan"), float("nan")

        enter = np.asarray(self.active_waypoints[enter_idx], dtype=float).reshape(3)
        axis = None
        if "gate_center" in roles[enter_idx + 1 :]:
            center_idx = enter_idx + 1 + roles[enter_idx + 1 :].index("gate_center")
            center = np.asarray(self.active_waypoints[center_idx], dtype=float).reshape(3)
            direction = center - enter
            direction_norm = float(np.linalg.norm(direction))
            if math.isfinite(direction_norm) and direction_norm > 1e-6:
                axis = direction / direction_norm
        if axis is None:
            axis = self._finite_vec3_or_none(self.active_gate_normal)
        if axis is None:
            return False, "missing_axis", enter, float("nan"), float("nan")

        pos = np.asarray(pos, dtype=float).reshape(3)
        distance_m = float(np.linalg.norm(pos - enter))
        progress_m = float(np.dot(pos - enter, axis))
        entered = (
            math.isfinite(distance_m)
            and distance_m <= float(self.active_target_shift_longitudinal_enter_radius_m)
        ) or (
            math.isfinite(progress_m)
            and progress_m >= -float(self.active_target_shift_longitudinal_enter_radius_m)
        )
        return entered, "entered" if entered else "not_entered", enter, distance_m, progress_m

    def _maybe_apply_deferred_longitudinal_exit_shift(
        self,
        *,
        pos: np.ndarray,
        vel: np.ndarray,
        gate_idx: int,
        track_id: int,
        target: np.ndarray,
    ) -> bool:
        if (
            self.deferred_longitudinal_shift_gate_idx != int(gate_idx)
            or self.deferred_longitudinal_shift_track_id != int(track_id)
            or not self.deferred_longitudinal_shift_samples
            or self.deferred_longitudinal_shift_applied_generation
            == self.active_plan_generation
        ):
            return False

        entered, reason, enter, enter_dist_m, enter_progress_m = (
            self._active_corridor_enter_status(pos)
        )
        if not entered:
            mean_m = float(np.mean(self.deferred_longitudinal_shift_samples))
            signature = (
                int(gate_idx),
                int(track_id),
                len(self.deferred_longitudinal_shift_samples),
                round(mean_m, 2),
                str(reason),
            )
            if signature != self.deferred_longitudinal_shift_pending_signature:
                self.deferred_longitudinal_shift_pending_signature = signature
                print(
                    "active_target_shift_longitudinal_pending "
                    f"gate_idx={int(gate_idx)} "
                    f"track={int(track_id)} "
                    f"samples={len(self.deferred_longitudinal_shift_samples)} "
                    f"mean={mean_m:.2f} "
                    f"reason={reason} "
                    f"enter={self._fmt_vec(enter, precision=3) if enter is not None else 'none'} "
                    f"enter_dist={self._fmt_float(enter_dist_m, precision=2)} "
                    f"enter_progress={self._fmt_float(enter_progress_m, precision=2)}",
                    flush=True,
                )
            return False

        mean_m = float(np.mean(self.deferred_longitudinal_shift_samples))
        max_shift = float(self.active_target_shift_longitudinal_exit_shift_max_m)
        clamped_m = float(np.clip(mean_m, -max_shift, max_shift)) if max_shift > 0.0 else 0.0
        if abs(clamped_m) < 1e-6:
            return False

        axis = self._finite_vec3_or_none(self.deferred_longitudinal_shift_axis)
        if axis is None:
            axis = self._finite_vec3_or_none(self.active_gate_normal)
        if axis is None:
            return False
        axis_norm = float(np.linalg.norm(axis))
        if not math.isfinite(axis_norm) or axis_norm < 1e-6:
            return False
        axis = axis / axis_norm

        return self._path_plan_deferred_corridor_exit_shift(
            pos=np.asarray(pos, dtype=float).reshape(3),
            vel=np.asarray(vel, dtype=float).reshape(3),
            target=np.asarray(target, dtype=float).reshape(3),
            axis=axis,
            exit_shift_m=clamped_m,
            raw_mean_m=mean_m,
            gate_idx=int(gate_idx),
            track_id=int(track_id),
            enter_reason=reason,
            enter=enter,
            enter_dist_m=enter_dist_m,
            enter_progress_m=enter_progress_m,
        )

    def _compute_passthrough_waypoint_velocities(
        self,
        waypoints: np.ndarray,
    ) -> np.ndarray | None:
        if not self.passthrough_velocity_enabled or self.passthrough_speed_m_s <= 0.0:
            return None
        waypoints = np.asarray(waypoints, dtype=float)
        if waypoints.ndim != 2 or waypoints.shape[1] != 3 or len(waypoints) < 3:
            return None

        velocities = np.full_like(waypoints, np.nan, dtype=float)
        for idx in range(1, len(waypoints) - 1):
            incoming = waypoints[idx] - waypoints[idx - 1]
            outgoing = waypoints[idx + 1] - waypoints[idx]
            incoming_norm = float(np.linalg.norm(incoming))
            outgoing_norm = float(np.linalg.norm(outgoing))
            if (
                not math.isfinite(incoming_norm)
                or not math.isfinite(outgoing_norm)
                or incoming_norm < 1e-6
                or outgoing_norm < 1e-6
            ):
                continue
            incoming_dir = incoming / incoming_norm
            outgoing_dir = outgoing / outgoing_norm
            tangent = incoming_dir + outgoing_dir
            tangent_norm = float(np.linalg.norm(tangent))
            if not math.isfinite(tangent_norm) or tangent_norm < 1e-6:
                tangent = outgoing_dir
                tangent_norm = float(np.linalg.norm(tangent))
            if not math.isfinite(tangent_norm) or tangent_norm < 1e-6:
                continue
            direction = tangent / tangent_norm

            speed = self.passthrough_speed_m_s
            if self.passthrough_velocity_mode == "adaptive":
                dot = float(np.clip(np.dot(incoming_dir, outgoing_dir), -1.0, 1.0))
                turn_angle = float(math.acos(dot))
                turn_ratio = turn_angle / math.pi
                adaptive_speed = self.passthrough_speed_max_m_s * (
                    1.0 - self.passthrough_turn_slowdown * turn_ratio
                )
                speed = max(self.passthrough_speed_m_s, adaptive_speed)
                speed = min(
                    speed,
                    self.passthrough_speed_max_m_s,
                    float(self.planner_vmax),
                )
            velocities[idx] = speed * direction
        if not np.any(np.isfinite(velocities[1:-1])):
            return None
        return velocities

    def _compute_role_aware_waypoint_velocities(
        self,
        waypoints: np.ndarray,
        waypoint_roles: list[str] | tuple[str, ...],
    ) -> np.ndarray | None:
        velocities = self._compute_passthrough_waypoint_velocities(waypoints)
        if velocities is None:
            return None

        roles = list(waypoint_roles or [])
        if len(roles) != len(waypoints):
            return velocities

        has_corridor = "gate_enter" in roles and "gate_exit" in roles
        if not has_corridor:
            return velocities

        filtered = np.full_like(velocities, np.nan, dtype=float)
        for idx, role in enumerate(roles):
            # Constraining the gate center inside an enter/exit corridor can
            # create a backward lobe before the gate plane. Keep the approach
            # tangent at the enter point and let the center/exit derivatives be
            # solved by the minimum-snap continuity constraints.
            if role == "gate_enter":
                filtered[idx] = velocities[idx]

        if not np.any(np.isfinite(filtered[1:-1])):
            return None
        return filtered

    def _passthrough_velocity_policy_text(self) -> str:
        if not self.passthrough_velocity_enabled or self.passthrough_speed_m_s <= 0.0:
            return "disabled"
        return (
            f"{self.passthrough_velocity_mode}:"
            f"base={self.passthrough_speed_m_s:.2f}:"
            f"max={self.passthrough_speed_max_m_s:.2f}:"
            f"slowdown={self.passthrough_turn_slowdown:.2f}"
        )

    def _terminal_velocity_for_plan(
        self,
        *,
        waypoints: np.ndarray,
        plan_mode: str,
        horizon_gate_indices: list[int],
    ) -> tuple[np.ndarray, str]:
        zero = np.zeros(3, dtype=float)
        if not self.terminal_velocity_enabled:
            return zero, "zero_disabled"
        if self.terminal_speed_m_s <= 0.0:
            return zero, "zero_speed"
        if waypoints.ndim != 2 or waypoints.shape[1] != 3 or len(waypoints) < 3:
            return zero, "zero_no_terminal_segment"

        last_gate_idx = (
            int(horizon_gate_indices[-1])
            if horizon_gate_indices
            else int(self.current_gate_idx)
        )
        if self._is_final_race_gate_index(last_gate_idx):
            return zero, "zero_final_gate"
        if plan_mode not in (
            "gate_horizon",
            "single_gate_corridor",
            "single_gate_corridor_exit_shift",
            "single_gate_exit",
            "provisional_next_gate",
        ):
            return zero, f"zero_mode_{plan_mode}"

        direction = np.asarray(waypoints[-1] - waypoints[-2], dtype=float).reshape(3)
        norm = float(np.linalg.norm(direction))
        if not math.isfinite(norm) or norm < 1e-6:
            return zero, "zero_bad_terminal_direction"

        speed = min(float(self.terminal_speed_m_s), float(self.planner_vmax))
        velocity = speed * direction / norm
        return velocity, f"continue_{plan_mode}"

    def _sanitize_plan_start_velocity(
        self,
        v_start: np.ndarray,
        waypoints: np.ndarray,
    ) -> np.ndarray:
        velocity = np.nan_to_num(
            np.asarray(v_start, dtype=float).reshape(3),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if self.plan_v_start_z_max_m_s > 0.0:
            velocity[2] = float(
                np.clip(
                    velocity[2],
                    -float(self.plan_v_start_z_max_m_s),
                    float(self.plan_v_start_z_max_m_s),
                )
            )
        try:
            waypoints = np.asarray(waypoints, dtype=float)
        except (TypeError, ValueError):
            return velocity
        if (
            waypoints.ndim == 2
            and waypoints.shape[1] == 3
            and len(waypoints) >= 2
            and np.all(np.isfinite(waypoints[[0, -1], 2]))
        ):
            dz = float(waypoints[-1, 2] - waypoints[0, 2])
            if dz > 0.05 and velocity[2] < 0.0:
                velocity[2] = 0.0
            elif dz < -0.05 and velocity[2] > 0.0:
                velocity[2] = 0.0
        return velocity

    @staticmethod
    def _planner_v_start_used(
        planner: MultiSegmentMinimumSnapPlanner,
        fallback: np.ndarray,
    ) -> np.ndarray:
        value = getattr(planner, "_aigp_v_start_used", None)
        if value is None:
            value = getattr(planner, "v_start", None)
        if value is None:
            value = fallback
        try:
            value = np.asarray(value, dtype=float).reshape(3)
        except (TypeError, ValueError):
            value = np.asarray(fallback, dtype=float).reshape(3)
        return value.copy()

    @staticmethod
    def _planner_v_start_raw(
        planner: MultiSegmentMinimumSnapPlanner,
        fallback: np.ndarray,
    ) -> np.ndarray:
        value = getattr(planner, "_aigp_v_start_raw", None)
        if value is None:
            value = fallback
        try:
            value = np.asarray(value, dtype=float).reshape(3)
        except (TypeError, ValueError):
            value = np.asarray(fallback, dtype=float).reshape(3)
        return value.copy()

    def _build_minimum_snap_plan(
        self,
        *,
        waypoints: np.ndarray,
        times,
        v_start: np.ndarray,
        v_end: np.ndarray,
        waypoint_velocities,
    ) -> MultiSegmentMinimumSnapPlanner:
        raw_v_start = np.nan_to_num(
            np.asarray(v_start, dtype=float).reshape(3),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        used_v_start = self._sanitize_plan_start_velocity(raw_v_start, waypoints)
        planner = MultiSegmentMinimumSnapPlanner()
        planner.update(
            waypoints=waypoints,
            times=times,
            v_start=used_v_start,
            v_end=v_end,
            a_start=np.zeros(3, dtype=float),
            a_end=np.zeros(3, dtype=float),
            j_start=np.zeros(3, dtype=float),
            j_end=np.zeros(3, dtype=float),
            waypoint_velocities=waypoint_velocities,
        )
        planner._aigp_v_start_raw = raw_v_start.copy()
        planner._aigp_v_start_used = used_v_start.copy()
        return planner

    def _sample_planner_path_text(
        self,
        planner: MultiSegmentMinimumSnapPlanner,
        *,
        sample_count: int = 120,
    ) -> str:
        try:
            total_time = float(planner.total_time)
        except (TypeError, ValueError):
            return "none"
        if not math.isfinite(total_time) or total_time <= 0.0:
            return "none"
        count = max(2, min(200, int(sample_count)))
        samples = []
        for tau in np.linspace(0.0, total_time, count):
            try:
                position, _, _ = planner.sample(float(tau))
            except Exception:
                return "none"
            point = np.asarray(position, dtype=float).reshape(3)
            if not np.all(np.isfinite(point)):
                return "none"
            samples.append(self._fmt_vec(point, precision=3))
        return "[" + ";".join(samples) + "]"

    @staticmethod
    def _project_point_to_waypoint_polyline(
        point: np.ndarray,
        waypoints: np.ndarray,
    ) -> tuple[float, float, int]:
        point = np.asarray(point, dtype=float).reshape(3)
        waypoints = np.asarray(waypoints, dtype=float)
        if waypoints.ndim != 2 or waypoints.shape[1] != 3 or len(waypoints) < 2:
            return float("inf"), float("nan"), -1

        best_distance = float("inf")
        best_progress = 0.0
        best_segment = -1
        cumulative = 0.0
        for idx in range(len(waypoints) - 1):
            start = waypoints[idx]
            end = waypoints[idx + 1]
            delta = end - start
            segment_length = float(np.linalg.norm(delta))
            if not math.isfinite(segment_length):
                return float("inf"), float("nan"), -1
            if segment_length < 1e-6:
                distance = float(np.linalg.norm(point - start))
                progress = cumulative
            else:
                alpha = float(np.dot(point - start, delta) / (segment_length ** 2))
                alpha = float(np.clip(alpha, 0.0, 1.0))
                projection = start + alpha * delta
                distance = float(np.linalg.norm(point - projection))
                progress = cumulative + alpha * segment_length
            if math.isfinite(distance) and distance < best_distance:
                best_distance = distance
                best_progress = progress
                best_segment = idx
            cumulative += max(0.0, segment_length)
        return best_distance, best_progress, best_segment

    def _validate_minimum_snap_plan_shape(
        self,
        *,
        planner: MultiSegmentMinimumSnapPlanner,
        plan_mode: str,
        gate_idx: int,
        track_id,
    ) -> tuple[bool, dict]:
        if not self.plan_validation_shape_enabled:
            return True, {"reason": "shape_validation_disabled"}

        waypoints = getattr(planner, "waypoints", None)
        if waypoints is None:
            return True, {"reason": "shape_validation_no_waypoints"}
        try:
            waypoints = np.asarray(waypoints, dtype=float)
        except (TypeError, ValueError):
            return False, {
                "reason": "invalid_waypoints",
                "plan_mode": str(plan_mode),
                "gate_idx": int(gate_idx),
                "track_id": track_id,
            }
        if (
            waypoints.ndim != 2
            or waypoints.shape[1] != 3
            or len(waypoints) < 2
            or not np.all(np.isfinite(waypoints))
        ):
            return False, {
                "reason": "invalid_waypoints",
                "plan_mode": str(plan_mode),
                "gate_idx": int(gate_idx),
                "track_id": track_id,
            }

        times = np.asarray(getattr(planner, "times", []), dtype=float).reshape(-1)
        starts = np.asarray(
            getattr(planner, "segment_starts", []),
            dtype=float,
        ).reshape(-1)
        if (
            len(times) <= 0
            or len(starts) < len(times)
            or len(times) != len(waypoints) - 1
            or not np.all(np.isfinite(times))
            or np.any(times <= 0.0)
        ):
            return False, {
                "reason": "invalid_trajectory_times",
                "plan_mode": str(plan_mode),
                "gate_idx": int(gate_idx),
                "track_id": track_id,
            }

        waypoint_segment_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        waypoint_cumulative_lengths = np.concatenate(
            ([0.0], np.cumsum(waypoint_segment_lengths))
        )
        waypoint_path_length = float(np.sum(waypoint_segment_lengths))
        if not math.isfinite(waypoint_path_length) or waypoint_path_length < 1e-6:
            return False, {
                "reason": "degenerate_waypoint_path",
                "plan_mode": str(plan_mode),
                "gate_idx": int(gate_idx),
                "track_id": track_id,
            }

        max_speed = 0.0
        max_accel = 0.0
        max_accel_xy = 0.0
        max_lateral_accel = 0.0
        max_accel_z_up = 0.0
        max_accel_z_down = 0.0
        max_corridor = 0.0
        max_corridor_segment = -1
        max_polyline_backtrack = 0.0
        best_polyline_progress = 0.0
        path_length = 0.0
        previous_position = None
        segment_path_lengths = np.zeros(len(times), dtype=float)
        samples_per_segment = max(8, int(self.plan_validation_samples_per_segment))
        max_allowed_z = float(np.max(waypoints[:, 2])) + float(
            self.plan_validation_max_z_overshoot_m
        )
        min_allowed_z = float(np.min(waypoints[:, 2])) - float(
            self.plan_validation_max_z_overshoot_m
        )

        for segment_idx, duration in enumerate(times):
            duration = float(duration)
            segment_start = float(starts[segment_idx])
            segment_previous = None
            for tau in np.linspace(0.0, duration, samples_per_segment):
                sample_time = segment_start + float(tau)
                try:
                    position, velocity, acceleration = planner.sample(sample_time)
                except Exception as exc:
                    return False, {
                        "reason": f"sample_failed:{type(exc).__name__}",
                        "plan_mode": str(plan_mode),
                        "gate_idx": int(gate_idx),
                        "track_id": track_id,
                        "segment_idx": int(segment_idx),
                        "sample_time_s": float(sample_time),
                    }
                position = self._finite_vec3_or_none(position)
                velocity = self._finite_vec3_or_none(velocity)
                acceleration = self._finite_vec3_or_none(acceleration)
                if position is None or velocity is None or acceleration is None:
                    return False, {
                        "reason": "non_finite_sample",
                        "plan_mode": str(plan_mode),
                        "gate_idx": int(gate_idx),
                        "track_id": track_id,
                        "segment_idx": int(segment_idx),
                        "sample_time_s": float(sample_time),
                    }

                speed = float(np.linalg.norm(velocity))
                accel = float(np.linalg.norm(acceleration))
                accel_xy = float(np.linalg.norm(acceleration[:2]))
                speed_xy = float(np.linalg.norm(velocity[:2]))
                lateral_accel = 0.0
                if speed_xy > 1e-3:
                    velocity_xy = velocity[:2]
                    acceleration_xy = acceleration[:2]
                    tangent_accel = (
                        float(np.dot(acceleration_xy, velocity_xy))
                        / (speed_xy ** 2)
                    ) * velocity_xy
                    lateral_accel = float(np.linalg.norm(acceleration_xy - tangent_accel))
                accel_z = float(acceleration[2])
                accel_z_up = max(0.0, accel_z)
                accel_z_down = max(0.0, -accel_z)
                max_speed = max(max_speed, speed)
                max_accel = max(max_accel, accel)
                max_accel_xy = max(max_accel_xy, accel_xy)
                max_lateral_accel = max(max_lateral_accel, lateral_accel)
                max_accel_z_up = max(max_accel_z_up, accel_z_up)
                max_accel_z_down = max(max_accel_z_down, accel_z_down)
                if (
                    self.plan_validation_max_z_overshoot_m > 0.0
                    and position[2] > max_allowed_z
                ):
                    return False, {
                        "reason": "z_overshoot_too_large",
                        "plan_mode": str(plan_mode),
                        "gate_idx": int(gate_idx),
                        "track_id": track_id,
                        "segment_idx": int(segment_idx),
                        "sample_time_s": float(sample_time),
                        "z_m": float(position[2]),
                        "max_allowed_z_m": float(max_allowed_z),
                        "z_overshoot_m": float(position[2] - max_allowed_z),
                        "position": position.copy(),
                        "speed_m_s": float(speed),
                        "accel_m_s2": float(accel),
                        "accel_xy_m_s2": float(accel_xy),
                        "lateral_accel_m_s2": float(lateral_accel),
                        "accel_z_up_m_s2": float(accel_z_up),
                        "accel_z_down_m_s2": float(accel_z_down),
                    }
                if (
                    self.plan_validation_max_z_overshoot_m > 0.0
                    and position[2] < min_allowed_z
                ):
                    return False, {
                        "reason": "z_undershoot_too_large",
                        "plan_mode": str(plan_mode),
                        "gate_idx": int(gate_idx),
                        "track_id": track_id,
                        "segment_idx": int(segment_idx),
                        "sample_time_s": float(sample_time),
                        "z_m": float(position[2]),
                        "min_allowed_z_m": float(min_allowed_z),
                        "z_undershoot_m": float(min_allowed_z - position[2]),
                        "position": position.copy(),
                        "speed_m_s": float(speed),
                        "accel_m_s2": float(accel),
                        "accel_xy_m_s2": float(accel_xy),
                        "lateral_accel_m_s2": float(lateral_accel),
                        "accel_z_up_m_s2": float(accel_z_up),
                        "accel_z_down_m_s2": float(accel_z_down),
                    }
                if (
                    self.plan_validation_max_speed_m_s > 0.0
                    and speed
                    > self.plan_validation_max_speed_m_s
                    + self.plan_validation_speed_tolerance_m_s
                ):
                    return False, {
                        "reason": "max_speed_too_large",
                        "plan_mode": str(plan_mode),
                        "gate_idx": int(gate_idx),
                        "track_id": track_id,
                        "segment_idx": int(segment_idx),
                        "sample_time_s": float(sample_time),
                        "speed_m_s": float(speed),
                        "max_speed_m_s": float(self.plan_validation_max_speed_m_s),
                        "speed_tolerance_m_s": float(
                            self.plan_validation_speed_tolerance_m_s
                        ),
                        "position": position.copy(),
                    }
                if (
                    self.plan_validation_max_acc_xy_m_s2 > 0.0
                    and accel_xy > self.plan_validation_max_acc_xy_m_s2
                ):
                    return False, {
                        "reason": "max_acc_xy_too_large",
                        "plan_mode": str(plan_mode),
                        "gate_idx": int(gate_idx),
                        "track_id": track_id,
                        "segment_idx": int(segment_idx),
                        "sample_time_s": float(sample_time),
                        "accel_xy_m_s2": float(accel_xy),
                        "max_acc_xy_m_s2": float(
                            self.plan_validation_max_acc_xy_m_s2
                        ),
                        "speed_m_s": float(speed),
                        "accel_m_s2": float(accel),
                        "lateral_accel_m_s2": float(lateral_accel),
                        "position": position.copy(),
                    }
                if (
                    self.plan_validation_max_lateral_accel_m_s2 > 0.0
                    and lateral_accel > self.plan_validation_max_lateral_accel_m_s2
                ):
                    return False, {
                        "reason": "max_lateral_accel_too_large",
                        "plan_mode": str(plan_mode),
                        "gate_idx": int(gate_idx),
                        "track_id": track_id,
                        "segment_idx": int(segment_idx),
                        "sample_time_s": float(sample_time),
                        "lateral_accel_m_s2": float(lateral_accel),
                        "max_lateral_accel_m_s2": float(
                            self.plan_validation_max_lateral_accel_m_s2
                        ),
                        "speed_m_s": float(speed),
                        "speed_xy_m_s": float(speed_xy),
                        "accel_m_s2": float(accel),
                        "accel_xy_m_s2": float(accel_xy),
                        "position": position.copy(),
                    }
                if (
                    self.plan_validation_max_acc_z_up_m_s2 > 0.0
                    and accel_z_up > self.plan_validation_max_acc_z_up_m_s2
                ):
                    return False, {
                        "reason": "max_acc_z_up_too_large",
                        "plan_mode": str(plan_mode),
                        "gate_idx": int(gate_idx),
                        "track_id": track_id,
                        "segment_idx": int(segment_idx),
                        "sample_time_s": float(sample_time),
                        "accel_z_up_m_s2": float(accel_z_up),
                        "max_acc_z_up_m_s2": float(
                            self.plan_validation_max_acc_z_up_m_s2
                        ),
                        "speed_m_s": float(speed),
                        "accel_m_s2": float(accel),
                        "position": position.copy(),
                    }
                if (
                    self.plan_validation_max_acc_z_down_m_s2 > 0.0
                    and accel_z_down > self.plan_validation_max_acc_z_down_m_s2
                ):
                    return False, {
                        "reason": "max_acc_z_down_too_large",
                        "plan_mode": str(plan_mode),
                        "gate_idx": int(gate_idx),
                        "track_id": track_id,
                        "segment_idx": int(segment_idx),
                        "sample_time_s": float(sample_time),
                        "accel_z_down_m_s2": float(accel_z_down),
                        "max_acc_z_down_m_s2": float(
                            self.plan_validation_max_acc_z_down_m_s2
                        ),
                        "speed_m_s": float(speed),
                        "accel_m_s2": float(accel),
                        "position": position.copy(),
                    }
                if (
                    self.plan_validation_max_accel_m_s2 > 0.0
                    and accel > self.plan_validation_max_accel_m_s2
                ):
                    return False, {
                        "reason": "max_accel_too_large",
                        "plan_mode": str(plan_mode),
                        "gate_idx": int(gate_idx),
                        "track_id": track_id,
                        "segment_idx": int(segment_idx),
                        "sample_time_s": float(sample_time),
                        "accel_m_s2": float(accel),
                        "max_accel_m_s2": float(
                            self.plan_validation_max_accel_m_s2
                        ),
                        "position": position.copy(),
                    }

                corridor, polyline_progress, nearest_segment = (
                    self._project_point_to_waypoint_polyline(position, waypoints)
                )
                if 0 <= segment_idx < len(waypoints) - 1:
                    segment_start_wp = waypoints[segment_idx]
                    segment_delta_wp = waypoints[segment_idx + 1] - segment_start_wp
                    segment_length_wp = float(waypoint_segment_lengths[segment_idx])
                    if math.isfinite(segment_length_wp) and segment_length_wp >= 1e-6:
                        segment_alpha = float(
                            np.dot(position - segment_start_wp, segment_delta_wp)
                            / (segment_length_wp ** 2)
                        )
                        segment_alpha = float(np.clip(segment_alpha, 0.0, 1.0))
                        polyline_progress = (
                            float(waypoint_cumulative_lengths[segment_idx])
                            + segment_alpha * segment_length_wp
                        )
                if math.isfinite(corridor) and corridor > max_corridor:
                    max_corridor = corridor
                    max_corridor_segment = int(nearest_segment)
                if (
                    self.plan_validation_max_corridor_m > 0.0
                    and math.isfinite(corridor)
                    and corridor > self.plan_validation_max_corridor_m
                ):
                    return False, {
                        "reason": "corridor_deviation_too_large",
                        "plan_mode": str(plan_mode),
                        "gate_idx": int(gate_idx),
                        "track_id": track_id,
                        "segment_idx": int(segment_idx),
                        "nearest_waypoint_segment": int(nearest_segment),
                        "sample_time_s": float(sample_time),
                        "corridor_m": float(corridor),
                        "max_corridor_m": float(self.plan_validation_max_corridor_m),
                        "position": position.copy(),
                    }

                if math.isfinite(polyline_progress):
                    polyline_backtrack = max(
                        0.0,
                        float(best_polyline_progress - polyline_progress),
                    )
                    max_polyline_backtrack = max(
                        max_polyline_backtrack,
                        polyline_backtrack,
                    )
                    if (
                        self.plan_validation_max_polyline_backtrack_m > 0.0
                        and polyline_backtrack
                        > self.plan_validation_max_polyline_backtrack_m
                    ):
                        return False, {
                            "reason": "polyline_backtrack_too_large",
                            "plan_mode": str(plan_mode),
                            "gate_idx": int(gate_idx),
                            "track_id": track_id,
                            "segment_idx": int(segment_idx),
                            "sample_time_s": float(sample_time),
                            "polyline_backtrack_m": float(polyline_backtrack),
                            "max_polyline_backtrack_m": float(
                                self.plan_validation_max_polyline_backtrack_m
                            ),
                            "polyline_progress_m": float(polyline_progress),
                            "best_polyline_progress_m": float(best_polyline_progress),
                            "position": position.copy(),
                        }
                    best_polyline_progress = max(
                        best_polyline_progress,
                        float(polyline_progress),
                    )

                if segment_previous is not None:
                    segment_path_lengths[segment_idx] += float(
                        np.linalg.norm(position - segment_previous)
                    )
                segment_previous = position.copy()
                if previous_position is not None:
                    path_length += float(np.linalg.norm(position - previous_position))
                previous_position = position.copy()

        path_length_ratio = path_length / waypoint_path_length
        if (
            self.plan_validation_max_path_length_ratio > 0.0
            and math.isfinite(path_length_ratio)
            and path_length_ratio > self.plan_validation_max_path_length_ratio
        ):
            return False, {
                "reason": "path_length_ratio_too_large",
                "plan_mode": str(plan_mode),
                "gate_idx": int(gate_idx),
                "track_id": track_id,
                "path_length_ratio": float(path_length_ratio),
                "max_path_length_ratio": float(
                    self.plan_validation_max_path_length_ratio
                ),
                "path_length_m": float(path_length),
                "waypoint_path_length_m": float(waypoint_path_length),
                "corridor_m": float(max_corridor),
                "speed_m_s": float(max_speed),
                "accel_m_s2": float(max_accel),
                "accel_xy_m_s2": float(max_accel_xy),
                "lateral_accel_m_s2": float(max_lateral_accel),
                "accel_z_up_m_s2": float(max_accel_z_up),
                "accel_z_down_m_s2": float(max_accel_z_down),
            }

        if self.plan_validation_max_segment_path_length_ratio > 0.0:
            for segment_idx, (sampled_length, chord_length) in enumerate(
                zip(segment_path_lengths, waypoint_segment_lengths)
            ):
                chord_length = float(chord_length)
                if chord_length < 1e-6:
                    continue
                segment_ratio = float(sampled_length / chord_length)
                if (
                    math.isfinite(segment_ratio)
                    and segment_ratio
                    > self.plan_validation_max_segment_path_length_ratio
                ):
                    return False, {
                        "reason": "segment_path_length_ratio_too_large",
                        "plan_mode": str(plan_mode),
                        "gate_idx": int(gate_idx),
                        "track_id": track_id,
                        "segment_idx": int(segment_idx),
                        "segment_path_length_ratio": float(segment_ratio),
                        "max_segment_path_length_ratio": float(
                            self.plan_validation_max_segment_path_length_ratio
                        ),
                        "segment_path_length_m": float(sampled_length),
                        "segment_chord_length_m": float(chord_length),
                        "corridor_m": float(max_corridor),
                        "speed_m_s": float(max_speed),
                        "accel_m_s2": float(max_accel),
                        "accel_xy_m_s2": float(max_accel_xy),
                        "lateral_accel_m_s2": float(max_lateral_accel),
                        "accel_z_up_m_s2": float(max_accel_z_up),
                        "accel_z_down_m_s2": float(max_accel_z_down),
                    }

        return True, {
            "reason": "shape_validation_ok",
            "plan_mode": str(plan_mode),
            "gate_idx": int(gate_idx),
            "track_id": track_id,
            "path_length_ratio": float(path_length_ratio),
            "path_length_m": float(path_length),
            "waypoint_path_length_m": float(waypoint_path_length),
            "corridor_m": float(max_corridor),
            "nearest_waypoint_segment": int(max_corridor_segment),
            "polyline_backtrack_m": float(max_polyline_backtrack),
            "speed_m_s": float(max_speed),
            "accel_m_s2": float(max_accel),
            "accel_xy_m_s2": float(max_accel_xy),
            "lateral_accel_m_s2": float(max_lateral_accel),
            "accel_z_up_m_s2": float(max_accel_z_up),
            "accel_z_down_m_s2": float(max_accel_z_down),
        }

    def _planner_segment_derivative(
        self,
        planner: MultiSegmentMinimumSnapPlanner,
        *,
        segment_idx: int,
        tau: float,
        order: int,
    ) -> np.ndarray | None:
        coeffs = getattr(planner, "coeffs", None)
        if coeffs is None:
            return None
        try:
            coeffs = np.asarray(coeffs, dtype=float)
            segment_idx = int(segment_idx)
            tau = float(tau)
            order = int(order)
        except (TypeError, ValueError):
            return None
        if (
            coeffs.ndim != 3
            or coeffs.shape[1] != 3
            or segment_idx < 0
            or segment_idx >= coeffs.shape[0]
        ):
            return None
        values = np.zeros(3, dtype=float)
        try:
            for axis in range(3):
                values[axis] = planner._eval_poly(
                    coeffs[segment_idx, axis, :],
                    tau,
                    order=order,
                )
        except Exception:
            return None
        if not np.all(np.isfinite(values)):
            return None
        return values

    def _angle_between_deg(self, a, b) -> float:
        a = self._finite_vec3_or_none(a)
        b = self._finite_vec3_or_none(b)
        if a is None or b is None:
            return float("nan")
        a_norm = float(np.linalg.norm(a))
        b_norm = float(np.linalg.norm(b))
        if (
            not math.isfinite(a_norm)
            or not math.isfinite(b_norm)
            or a_norm < 1e-6
            or b_norm < 1e-6
        ):
            return float("nan")
        dot = float(np.dot(a, b) / (a_norm * b_norm))
        dot = float(np.clip(dot, -1.0, 1.0))
        return float(math.degrees(math.acos(dot)))

    def _trace_plan_boundary_continuity(
        self,
        *,
        planner: MultiSegmentMinimumSnapPlanner,
        waypoints: np.ndarray,
        plan_mode: str,
        horizon_track_ids,
        horizon_gate_indices,
    ) -> None:
        times = getattr(planner, "times", None)
        if times is None:
            return
        try:
            times = np.asarray(times, dtype=float).reshape(-1)
            waypoints = np.asarray(waypoints, dtype=float)
        except (TypeError, ValueError):
            return
        if (
            times.size < 2
            or waypoints.ndim != 2
            or waypoints.shape[1] != 3
            or waypoints.shape[0] != times.size + 1
        ):
            return

        for boundary_i in range(1, int(times.size)):
            left_segment = boundary_i - 1
            right_segment = boundary_i
            tau_s = float(np.sum(times[:boundary_i]))
            left_tau_s = float(times[left_segment])
            right_tau_s = 0.0

            left_velocity = self._planner_segment_derivative(
                planner,
                segment_idx=left_segment,
                tau=left_tau_s,
                order=1,
            )
            right_velocity = self._planner_segment_derivative(
                planner,
                segment_idx=right_segment,
                tau=right_tau_s,
                order=1,
            )
            left_acceleration = self._planner_segment_derivative(
                planner,
                segment_idx=left_segment,
                tau=left_tau_s,
                order=2,
            )
            right_acceleration = self._planner_segment_derivative(
                planner,
                segment_idx=right_segment,
                tau=right_tau_s,
                order=2,
            )
            left_jerk = self._planner_segment_derivative(
                planner,
                segment_idx=left_segment,
                tau=left_tau_s,
                order=3,
            )
            right_jerk = self._planner_segment_derivative(
                planner,
                segment_idx=right_segment,
                tau=right_tau_s,
                order=3,
            )
            if (
                left_velocity is None
                or right_velocity is None
                or left_acceleration is None
                or right_acceleration is None
                or left_jerk is None
                or right_jerk is None
            ):
                continue

            incoming_chord = waypoints[boundary_i] - waypoints[boundary_i - 1]
            outgoing_chord = waypoints[boundary_i + 1] - waypoints[boundary_i]
            velocity_at_waypoint = 0.5 * (left_velocity + right_velocity)
            speed_at_waypoint = float(np.linalg.norm(velocity_at_waypoint))
            turn_angle_in = self._angle_between_deg(incoming_chord, left_velocity)
            turn_angle_out = self._angle_between_deg(right_velocity, outgoing_chord)
            turn_angle_chord = self._angle_between_deg(incoming_chord, outgoing_chord)
            velocity_delta = float(np.linalg.norm(right_velocity - left_velocity))
            acceleration_delta = float(
                np.linalg.norm(right_acceleration - left_acceleration)
            )
            jerk_delta = float(np.linalg.norm(right_jerk - left_jerk))

            horizon_idx = int(boundary_i) - 1
            track_id = (
                horizon_track_ids[horizon_idx]
                if horizon_track_ids is not None and horizon_idx < len(horizon_track_ids)
                else None
            )
            gate_idx = (
                horizon_gate_indices[horizon_idx]
                if horizon_gate_indices is not None
                and horizon_idx < len(horizon_gate_indices)
                else None
            )
            print(
                "plan_boundary_continuity "
                f"plan_mode={plan_mode} "
                f"boundary_i={int(boundary_i)} "
                f"left_segment={int(left_segment)} "
                f"right_segment={int(right_segment)} "
                f"tau_s={tau_s:.3f} "
                f"gate_idx={gate_idx if gate_idx is not None else 'none'} "
                f"track={track_id if track_id is not None else 'none'} "
                f"waypoint_neu={self._fmt_vec(waypoints[boundary_i], precision=3)} "
                f"left_velocity_neu={self._fmt_vec(left_velocity, precision=6)} "
                f"right_velocity_neu={self._fmt_vec(right_velocity, precision=6)} "
                f"left_acceleration_neu={self._fmt_vec(left_acceleration, precision=6)} "
                f"right_acceleration_neu={self._fmt_vec(right_acceleration, precision=6)} "
                f"left_jerk_neu={self._fmt_vec(left_jerk, precision=6)} "
                f"right_jerk_neu={self._fmt_vec(right_jerk, precision=6)} "
                f"speed_at_waypoint={speed_at_waypoint:.6f} "
                f"turn_angle_in_deg={self._fmt_float(turn_angle_in, precision=3)} "
                f"turn_angle_out_deg={self._fmt_float(turn_angle_out, precision=3)} "
                f"turn_angle_chord_deg={self._fmt_float(turn_angle_chord, precision=3)} "
                f"velocity_delta={velocity_delta:.6f} "
                f"acceleration_delta={acceleration_delta:.6f} "
                f"jerk_delta={jerk_delta:.6f}",
                flush=True,
            )

    def _validate_active_gate_plan_crossing(
        self,
        *,
        planner: MultiSegmentMinimumSnapPlanner,
        target: np.ndarray,
        normal: np.ndarray | None,
        plan_mode: str,
        gate_idx: int,
        track_id,
    ) -> tuple[bool, dict]:
        target = self._finite_vec3_or_none(target)
        normal = self._finite_vec3_or_none(normal)
        if target is None:
            return False, {
                "reason": "non_finite_target",
                "plan_mode": str(plan_mode),
                "gate_idx": int(gate_idx),
                "track_id": track_id,
            }
        shape_valid, shape_details = self._validate_minimum_snap_plan_shape(
            planner=planner,
            plan_mode=plan_mode,
            gate_idx=gate_idx,
            track_id=track_id,
        )
        if not shape_valid:
            return False, shape_details
        if normal is None:
            return True, {"reason": "no_gate_normal"}
        normal_norm = float(np.linalg.norm(normal))
        if not math.isfinite(normal_norm) or normal_norm < 1e-6:
            return True, {"reason": "no_gate_normal"}
        if planner.total_time <= 0.0:
            return False, {
                "reason": "empty_trajectory",
                "plan_mode": str(plan_mode),
                "gate_idx": int(gate_idx),
                "track_id": track_id,
            }

        times = np.asarray(getattr(planner, "times", []), dtype=float).reshape(-1)
        starts = np.asarray(
            getattr(planner, "segment_starts", []),
            dtype=float,
        ).reshape(-1)
        if len(times) <= 0 or len(starts) < len(times):
            return False, {
                "reason": "invalid_trajectory_times",
                "plan_mode": str(plan_mode),
                "gate_idx": int(gate_idx),
                "track_id": track_id,
            }

        try:
            previous, _, _ = planner.sample(0.0)
        except Exception as exc:
            return False, {
                "reason": f"sample_failed:{type(exc).__name__}",
                "plan_mode": str(plan_mode),
                "gate_idx": int(gate_idx),
                "track_id": track_id,
            }

        closest = {
            "abs_progress_m": float("inf"),
            "lateral_error_m": float("nan"),
            "sample_time_s": 0.0,
            "position": previous.copy(),
        }
        n = normal / normal_norm
        previous_rel = np.asarray(previous, dtype=float).reshape(3) - target
        previous_progress = float(np.dot(previous_rel, n))
        best_progress = previous_progress
        backward_progress_tolerance_m = max(
            0.5,
            0.5 * float(self.gate_pass_lateral_radius_m),
        )
        initial_backward_progress_tolerance_m = min(
            backward_progress_tolerance_m,
            0.15,
        )
        samples_per_segment = 80
        for segment_idx, duration in enumerate(times):
            duration = float(duration)
            if not math.isfinite(duration) or duration <= 0.0:
                continue
            segment_start = float(starts[segment_idx])
            sample_count = max(4, int(samples_per_segment))
            for tau in np.linspace(0.0, duration, sample_count)[1:]:
                sample_time = segment_start + float(tau)
                try:
                    position, _, _ = planner.sample(sample_time)
                except Exception as exc:
                    return False, {
                        "reason": f"sample_failed:{type(exc).__name__}",
                        "plan_mode": str(plan_mode),
                        "gate_idx": int(gate_idx),
                        "track_id": track_id,
                        "segment_idx": int(segment_idx),
                    }

                rel = np.asarray(position, dtype=float).reshape(3) - target
                progress = float(np.dot(rel, n))
                lateral_vec = rel - progress * n
                lateral = float(np.linalg.norm(lateral_vec))
                abs_progress = abs(progress)
                if math.isfinite(abs_progress) and abs_progress < closest["abs_progress_m"]:
                    closest = {
                        "abs_progress_m": abs_progress,
                        "lateral_error_m": lateral,
                        "sample_time_s": sample_time,
                        "position": np.asarray(position, dtype=float).reshape(3).copy(),
                    }

                result = check_gate_plane_pass(
                    previous_position=previous,
                    position=position,
                    center=target,
                    normal=normal,
                    lateral_radius_m=self.gate_pass_lateral_radius_m,
                    plane_tolerance_m=self.gate_plane_tolerance_m,
                )
                if result.crossing_point is not None:
                    details = {
                        "reason": result.reason if result.passed else result.reason,
                        "plan_mode": str(plan_mode),
                        "gate_idx": int(gate_idx),
                        "track_id": track_id,
                        "segment_idx": int(segment_idx),
                        "sample_time_s": float(sample_time),
                        "distance_m": float(result.distance_m),
                        "plane_progress_m": float(result.signed_progress_m),
                        "previous_plane_progress_m": float(
                            result.previous_signed_progress_m
                        ),
                        "lateral_error_m": float(result.lateral_error_m),
                        "crossed_plane": bool(result.crossed_plane),
                        "crossing_point": (
                            None
                            if result.crossing_point is None
                            else np.asarray(result.crossing_point, dtype=float)
                            .reshape(3)
                            .copy()
                        ),
                    }
                    if result.passed:
                        return True, details
                    return False, details

                initial_window_s = min(1.0, 0.25 * float(planner.total_time))
                is_initial_backtrack = (
                    int(segment_idx) == 0
                    and float(sample_time) <= initial_window_s
                    and math.isfinite(progress)
                    and math.isfinite(best_progress)
                    and progress
                    < best_progress - float(initial_backward_progress_tolerance_m)
                )
                if is_initial_backtrack:
                    return False, {
                        "reason": "initial_backward_progress_before_crossing",
                        "plan_mode": str(plan_mode),
                        "gate_idx": int(gate_idx),
                        "track_id": track_id,
                        "segment_idx": int(segment_idx),
                        "sample_time_s": float(sample_time),
                        "progress_m": float(progress),
                        "best_progress_m": float(best_progress),
                        "backward_progress_m": float(best_progress - progress),
                        "lateral_error_m": float(lateral),
                        "position": np.asarray(position, dtype=float)
                        .reshape(3)
                        .copy(),
                    }

                if (
                    math.isfinite(progress)
                    and math.isfinite(best_progress)
                    and progress < best_progress - backward_progress_tolerance_m
                ):
                    return False, {
                        "reason": "backward_progress_before_crossing",
                        "plan_mode": str(plan_mode),
                        "gate_idx": int(gate_idx),
                        "track_id": track_id,
                        "segment_idx": int(segment_idx),
                        "sample_time_s": float(sample_time),
                        "progress_m": float(progress),
                        "best_progress_m": float(best_progress),
                        "backward_progress_m": float(best_progress - progress),
                        "lateral_error_m": float(lateral),
                        "position": np.asarray(position, dtype=float)
                        .reshape(3)
                        .copy(),
                    }
                if math.isfinite(progress) and (
                    not math.isfinite(best_progress) or progress > best_progress
                ):
                    best_progress = progress
                previous = np.asarray(position, dtype=float).reshape(3).copy()

        return False, {
            "reason": "no_active_gate_crossing",
            "plan_mode": str(plan_mode),
            "gate_idx": int(gate_idx),
            "track_id": track_id,
            "closest_abs_progress_m": float(closest["abs_progress_m"]),
            "closest_lateral_error_m": float(closest["lateral_error_m"]),
            "closest_sample_time_s": float(closest["sample_time_s"]),
            "closest_position": closest["position"],
        }

    def _trace_plan_validation_reject(
        self,
        details: dict,
        *,
        fallback: str,
    ) -> None:
        reason = str(details.get("reason", "unknown"))
        gate_idx = int(details.get("gate_idx", self.current_gate_idx))
        track_id = details.get("track_id", None)
        plan_mode = str(details.get("plan_mode", "unknown"))
        lateral = details.get("lateral_error_m", details.get("closest_lateral_error_m"))
        progress = details.get(
            "plane_progress_m",
            details.get("progress_m", details.get("closest_abs_progress_m")),
        )
        sample_time = details.get("sample_time_s", details.get("closest_sample_time_s"))
        crossing = details.get("crossing_point", details.get("closest_position"))
        crossing = self._finite_vec3_or_none(crossing)
        path_ratio = details.get("path_length_ratio")
        segment_ratio = details.get("segment_path_length_ratio")
        corridor = details.get("corridor_m")
        polyline_backtrack = details.get("polyline_backtrack_m")
        speed = details.get("speed_m_s")
        accel = details.get("accel_m_s2")
        accel_xy = details.get("accel_xy_m_s2")
        lateral_accel = details.get("lateral_accel_m_s2")
        accel_z_up = details.get("accel_z_up_m_s2")
        accel_z_down = details.get("accel_z_down_m_s2")
        z_overshoot = details.get("z_overshoot_m", details.get("z_undershoot_m"))
        signature = (
            gate_idx,
            None if track_id is None else int(track_id),
            plan_mode,
            reason,
            str(fallback),
            round(float(lateral), 2) if self._is_finite_number(lateral) else "nan",
            round(float(progress), 2) if self._is_finite_number(progress) else "nan",
        )
        if signature == self._last_plan_validation_reject_signature:
            return
        self._last_plan_validation_reject_signature = signature
        print(
            "plan_validation_reject "
            f"gate_idx={gate_idx} "
            f"track={track_id if track_id is not None else 'none'} "
            f"mode={plan_mode} "
            f"reason={reason} "
            f"lateral={self._fmt_float(lateral, precision=2)} "
            f"progress={self._fmt_float(progress, precision=2)} "
            f"sample_time={self._fmt_float(sample_time, precision=2)} "
            f"crossing={self._fmt_vec(crossing, precision=3) if crossing is not None else 'none'} "
            f"path_ratio={self._fmt_float(path_ratio, precision=3)} "
            f"segment_ratio={self._fmt_float(segment_ratio, precision=3)} "
            f"corridor={self._fmt_float(corridor, precision=2)} "
            f"polyline_backtrack={self._fmt_float(polyline_backtrack, precision=2)} "
            f"speed={self._fmt_float(speed, precision=2)} "
            f"accel={self._fmt_float(accel, precision=2)} "
            f"accel_xy={self._fmt_float(accel_xy, precision=2)} "
            f"lateral_accel={self._fmt_float(lateral_accel, precision=2)} "
            f"accel_z_up={self._fmt_float(accel_z_up, precision=2)} "
            f"accel_z_down={self._fmt_float(accel_z_down, precision=2)} "
            f"z_overshoot={self._fmt_float(z_overshoot, precision=2)} "
            f"fallback={fallback}",
            flush=True,
        )

    def _plan_validation_retry_scale(self, details: dict) -> float | None:
        reason = str(details.get("reason", ""))
        retry_specs = {
            "max_speed_too_large": ("speed_m_s", "max_speed_m_s", 1.0),
            "max_accel_too_large": ("accel_m_s2", "max_accel_m_s2", 0.5),
            "max_acc_xy_too_large": (
                "accel_xy_m_s2",
                "max_acc_xy_m_s2",
                0.5,
            ),
            "max_lateral_accel_too_large": (
                "lateral_accel_m_s2",
                "max_lateral_accel_m_s2",
                0.5,
            ),
            "max_acc_z_up_too_large": (
                "accel_z_up_m_s2",
                "max_acc_z_up_m_s2",
                0.5,
            ),
            "max_acc_z_down_too_large": (
                "accel_z_down_m_s2",
                "max_acc_z_down_m_s2",
                0.5,
            ),
        }
        if reason in retry_specs:
            value_key, limit_key, exponent = retry_specs[reason]
            value = self._finite_float(details.get(value_key), float("nan"))
            limit = self._finite_float(details.get(limit_key), float("nan"))
            if (
                not math.isfinite(value)
                or not math.isfinite(limit)
                or limit <= 0.0
                or value <= limit
            ):
                return None
            ratio = max(1.0, value / limit)
            return max(1.15, min(3.0, 1.15 * (ratio ** exponent)))
        if reason in ("z_overshoot_too_large", "z_undershoot_too_large"):
            return 1.35
        return None

    def _allow_provisional_direct_fallback(self, track, details: dict) -> bool:
        if track is None:
            return False
        for key in (
            "retained_spline_memory",
            "retained_provisional",
            "retained_memory",
        ):
            if self._finite_float(details.get(key), 0.0) > 0.5:
                return True

        committed = bool(getattr(track, "committed", False))
        stable = bool(
            getattr(track, "is_stable", False) or getattr(track, "ever_stable", False)
        )
        lateral = self._finite_float(details.get("lateral"), float("nan"))
        lateral_limit = max(
            float(self.gate_pass_lateral_radius_m),
            float(self.race_order_front_blocker_lateral_radius_m),
            float(self.race_order_duplicate_radius_m),
        )
        if math.isfinite(lateral) and lateral <= lateral_limit:
            return True
        return committed and stable and math.isfinite(lateral) and lateral <= lateral_limit

    def _path_plan_deferred_corridor_exit_shift(
        self,
        *,
        pos: np.ndarray,
        vel: np.ndarray,
        target: np.ndarray,
        axis: np.ndarray,
        exit_shift_m: float,
        raw_mean_m: float,
        gate_idx: int,
        track_id: int,
        enter_reason: str,
        enter: np.ndarray | None,
        enter_dist_m: float,
        enter_progress_m: float,
    ) -> bool:
        half_length = 0.5 * max(0.0, float(self.gate_corridor_length_m))
        min_exit_after_center_m = max(float(VADR_TS_002.gate_depth_m), 0.25)
        exit_after_center_m = max(
            min_exit_after_center_m,
            half_length + float(exit_shift_m),
        )
        shifted_exit = target + axis * exit_after_center_m
        waypoints = np.vstack([pos, target, shifted_exit])
        waypoint_roles = ["start", "gate_center", "gate_exit_shifted"]
        waypoint_velocities = self._compute_role_aware_waypoint_velocities(
            waypoints,
            waypoint_roles,
        )
        times = allocate_segment_times(
            waypoints,
            current_vel=vel,
            vmax=self.planner_vmax,
            amax=self.planner_amax,
            T_min=self.planner_t_min,
        )
        plan_mode = "single_gate_corridor_exit_shift"
        terminal_velocity, terminal_policy = self._terminal_velocity_for_plan(
            waypoints=waypoints,
            plan_mode=plan_mode,
            horizon_gate_indices=[int(gate_idx)],
        )
        planner = self._build_minimum_snap_plan(
            waypoints=waypoints,
            times=times,
            v_start=vel,
            v_end=terminal_velocity,
            waypoint_velocities=waypoint_velocities,
        )
        valid_plan, validation_details = self._validate_active_gate_plan_crossing(
            planner=planner,
            target=target,
            normal=axis,
            plan_mode=plan_mode,
            gate_idx=int(gate_idx),
            track_id=int(track_id),
        )
        if not valid_plan:
            self._trace_plan_validation_reject(
                validation_details,
                fallback="deferred_exit_shift_hold_current_plan",
            )
            print(
                "active_target_shift_longitudinal_exit_replan_reject "
                f"gate_idx={int(gate_idx)} "
                f"track={int(track_id)} "
                f"mean={float(raw_mean_m):.2f} "
                f"clamped={float(exit_shift_m):.2f} "
                f"reason={validation_details.get('reason', 'unknown')}",
                flush=True,
            )
            return False

        self.planner = planner
        self.current_gate_pos = target.copy()
        self.active_waypoints = waypoints.copy()
        self.active_times = np.asarray(times, dtype=float).copy()
        self.active_waypoint_roles = list(waypoint_roles)
        self.active_horizon_gate_indices = [int(gate_idx)]
        self.active_horizon_track_ids = [int(track_id)]
        self.active_horizon_targets = [target.copy()]
        self.active_plan_mode = plan_mode
        self.active_terminal_velocity = terminal_velocity.copy()
        self.active_terminal_velocity_policy = str(terminal_policy)
        self.post_gate_exit_until_s = 0.0
        self.post_gate_exit_reason = ""
        self.trajectory_start_time = time.time()
        self.last_plan_wall_time = self.trajectory_start_time
        self.last_planned_gate_idx = int(gate_idx)
        self.active_plan_generation += 1
        self._reset_reference_progress_state()
        self._record_spline_memory_from_active_plan(source=plan_mode)
        self.deferred_longitudinal_shift_applied_generation = self.active_plan_generation
        self.deferred_longitudinal_shift_samples = []
        self.deferred_longitudinal_shift_pending_signature = None
        self._reset_gate_pass_state()
        self._initialize_gate_pass_tracking(
            pos=pos,
            target=target,
            fallback_normal=axis,
        )

        waypoints_txt = "[" + ";".join(
            self._fmt_vec(waypoint, precision=3) for waypoint in waypoints
        ) + "]"
        times_txt = "(" + ",".join(f"{float(item):.3f}" for item in times) + ")"
        waypoint_roles_txt = "(" + ",".join(str(role) for role in waypoint_roles) + ")"
        waypoint_velocities_txt = "none"
        if waypoint_velocities is not None:
            waypoint_velocities_txt = "[" + ";".join(
                self._fmt_vec(velocity, precision=3)
                if np.all(np.isfinite(velocity))
                else "nan"
                for velocity in waypoint_velocities
            ) + "]"
        print(
            "active_target_shift_longitudinal_exit_replan "
            f"gate_idx={int(gate_idx)} "
            f"track={int(track_id)} "
            f"mean={float(raw_mean_m):.2f} "
            f"clamped={float(exit_shift_m):.2f} "
            f"exit_after_center={float(exit_after_center_m):.2f} "
            f"enter_reason={enter_reason} "
            f"enter={self._fmt_vec(enter, precision=3) if enter is not None else 'none'} "
            f"enter_dist={self._fmt_float(enter_dist_m, precision=2)} "
            f"enter_progress={self._fmt_float(enter_progress_m, precision=2)} "
            f"target_neu={self._fmt_vec(target, precision=3)} "
            f"axis={self._fmt_vec(axis, precision=3)} "
            f"shifted_exit={self._fmt_vec(shifted_exit, precision=3)}",
            flush=True,
        )
        plan_samples_txt = self._sample_planner_path_text(self.planner)
        print(
            "plan_install "
            f"gate_idx={int(gate_idx)} "
            f"track={int(track_id)} "
            f"mode={plan_mode} "
            f"horizon_tracks=({int(track_id)}) "
            f"horizon_gate_indices=({int(gate_idx)}) "
            f"total_time={float(self.planner.total_time):.3f} "
            f"segments={max(0, int(len(waypoints) - 1))} "
            f"target_neu={self._fmt_vec(target, precision=3)} "
            f"normal_neu={self._fmt_vec(axis, precision=3)} "
            f"v_start_neu={self._fmt_vec(self._planner_v_start_used(self.planner, vel), precision=3)} "
            f"v_start_raw_neu={self._fmt_vec(self._planner_v_start_raw(self.planner, vel), precision=3)} "
            f"v_end_neu={self._fmt_vec(terminal_velocity, precision=3)} "
            f"terminal_policy={terminal_policy} "
            f"passthrough_policy={self._passthrough_velocity_policy_text()} "
            f"waypoint_velocities_neu={waypoint_velocities_txt} "
            f"times_s={times_txt} "
            f"waypoint_roles={waypoint_roles_txt} "
            f"gate_corridor={int(self.gate_corridor_enabled)}:"
            f"{float(self.gate_corridor_length_m):.2f} "
            f"exit_shift_m={float(exit_shift_m):.2f} "
            f"waypoints_neu={waypoints_txt} "
            f"plan_samples_neu={plan_samples_txt}",
            flush=True,
        )
        self._trace_plan_boundary_continuity(
            planner=self.planner,
            waypoints=waypoints,
            plan_mode=plan_mode,
            horizon_track_ids=[int(track_id)],
            horizon_gate_indices=[int(gate_idx)],
        )
        return True

    def _path_plan(self, pos: np.ndarray, vel: np.ndarray) -> bool:
        pos = np.asarray(pos, dtype=float).reshape(3)
        vel = np.asarray(vel, dtype=float).reshape(3)
        target_idx = int(self.current_gate_idx)
        if target_idx < 0:
            return False

        selected_target = (
            self.gate_centers_neu[target_idx]
            if target_idx < len(self.gate_centers_neu)
            else None
        )
        target_track_id = (
            self.gate_track_ids[target_idx]
            if target_idx < len(self.gate_track_ids)
            else None
        )
        target, target_track_id, used_spline_memory = (
            self._spline_memory_override_target(
                gate_idx=target_idx,
                selected_target=selected_target,
                selected_track_id=target_track_id,
                context="path_plan",
            )
        )
        if target is None:
            return False
        target = self._apply_target_z_policy(target)
        target = self.target_manager.lock_target(
            gate_idx=target_idx,
            track_id=target_track_id,
            center_neu=target,
            reason="spline_memory_path_plan" if used_spline_memory else "path_plan",
        )
        self._sync_target_manager_state()

        self._initialize_gate_pass_tracking(
            pos=pos,
            target=target,
            fallback_normal=self.active_gate_normal,
        )
        normal = self.active_gate_normal

        horizon_targets, horizon_track_ids, horizon_gate_indices = (
            self._planning_horizon_targets(
                target_idx,
                target,
                target_track_id,
                pos=pos,
                vel=vel,
            )
        )
        self._record_spline_memory_candidates(
            gate_indices=horizon_gate_indices,
            track_ids=horizon_track_ids,
            targets=horizon_targets,
            source="planning_horizon_candidate",
            skip_current=True,
        )

        def build_candidate(targets, track_ids, gate_indices, *, allow_exit: bool = True):
            if len(targets) >= 2:
                preferred_normals = [normal] + [None] * max(0, len(targets) - 1)
                candidate_waypoints, candidate_waypoint_roles = (
                    self._build_gate_corridor_waypoints(
                        start=pos,
                        targets=targets,
                        preferred_normals=preferred_normals,
                    )
                )
                candidate_waypoint_velocities = (
                    self._compute_role_aware_waypoint_velocities(
                        candidate_waypoints,
                        candidate_waypoint_roles,
                    )
                )
                candidate_mode = "gate_horizon"
            elif normal is None or not allow_exit:
                candidate_waypoints = np.vstack([pos, target])
                candidate_waypoint_roles = ["start", "gate_center"]
                candidate_waypoint_velocities = None
                candidate_mode = "single_gate"
            elif self.gate_corridor_enabled and self.gate_corridor_length_m > 0.0:
                candidate_waypoints, candidate_waypoint_roles = (
                    self._build_gate_corridor_waypoints(
                        start=pos,
                        targets=[target],
                        preferred_normals=[normal],
                    )
                )
                candidate_waypoint_velocities = (
                    self._compute_role_aware_waypoint_velocities(
                        candidate_waypoints,
                        candidate_waypoint_roles,
                    )
                )
                candidate_mode = (
                    "single_gate_corridor"
                    if len(candidate_waypoints) >= 3
                    else "single_gate"
                )
            else:
                pass_through_target = target + normal * self.gate_pass_through_m
                candidate_waypoints = np.vstack([pos, target, pass_through_target])
                candidate_waypoint_roles = ["start", "gate_center", "gate_exit"]
                candidate_waypoint_velocities = None
                candidate_mode = "single_gate_exit"
            candidate_times = allocate_segment_times(
                candidate_waypoints,
                current_vel=vel,
                vmax=self.planner_vmax,
                amax=self.planner_amax,
                T_min=self.planner_t_min,
            )
            candidate_terminal_velocity, candidate_terminal_policy = (
                self._terminal_velocity_for_plan(
                    waypoints=candidate_waypoints,
                    plan_mode=candidate_mode,
                    horizon_gate_indices=gate_indices,
                )
            )
            candidate_planner = self._build_minimum_snap_plan(
                waypoints=candidate_waypoints,
                times=candidate_times,
                v_start=vel,
                v_end=candidate_terminal_velocity,
                waypoint_velocities=candidate_waypoint_velocities,
            )
            return {
                "planner": candidate_planner,
                "waypoints": candidate_waypoints,
                "times": candidate_times,
                "terminal_velocity": candidate_terminal_velocity,
                "terminal_policy": candidate_terminal_policy,
                "waypoint_velocities": candidate_waypoint_velocities,
                "mode": candidate_mode,
                "waypoint_roles": list(candidate_waypoint_roles),
                "horizon_targets": [
                    np.asarray(item, dtype=float).reshape(3).copy()
                    for item in targets
                ],
                "horizon_track_ids": list(track_ids),
                "horizon_gate_indices": list(gate_indices),
            }

        candidate = build_candidate(
            horizon_targets,
            horizon_track_ids,
            horizon_gate_indices,
        )
        valid_plan, validation_details = self._validate_active_gate_plan_crossing(
            planner=candidate["planner"],
            target=target,
            normal=normal,
            plan_mode=candidate["mode"],
            gate_idx=target_idx,
            track_id=target_track_id,
        )

        def retry_candidate_with_slower_timing(candidate, validation_details):
            initial_scale = self._plan_validation_retry_scale(validation_details)
            if initial_scale is None:
                return candidate, False, validation_details
            scale = float(initial_scale)
            best_details = validation_details
            for attempt in range(3):
                times = np.asarray(candidate["times"], dtype=float) * scale
                retry_planner = self._build_minimum_snap_plan(
                    waypoints=candidate["waypoints"],
                    times=times,
                    v_start=vel,
                    v_end=candidate["terminal_velocity"],
                    waypoint_velocities=candidate["waypoint_velocities"],
                )
                retry_valid, retry_details = self._validate_active_gate_plan_crossing(
                    planner=retry_planner,
                    target=target,
                    normal=normal,
                    plan_mode=candidate["mode"],
                    gate_idx=target_idx,
                    track_id=target_track_id,
                )
                print(
                    "plan_validation_retry "
                    f"gate_idx={target_idx} "
                    f"track={target_track_id if target_track_id is not None else 'none'} "
                    f"mode={candidate['mode']} "
                    f"attempt={attempt + 1} "
                    f"time_scale={scale:.2f} "
                    f"valid={int(retry_valid)} "
                    f"reason={retry_details.get('reason', 'ok')} "
                    f"speed={self._fmt_float(retry_details.get('speed_m_s'), precision=2)} "
                    f"accel={self._fmt_float(retry_details.get('accel_m_s2'), precision=2)} "
                    f"accel_xy={self._fmt_float(retry_details.get('accel_xy_m_s2'), precision=2)} "
                    f"lateral_accel={self._fmt_float(retry_details.get('lateral_accel_m_s2'), precision=2)} "
                    f"z_overshoot={self._fmt_float(retry_details.get('z_overshoot_m', retry_details.get('z_undershoot_m')), precision=2)}",
                    flush=True,
                )
                if retry_valid:
                    retried = dict(candidate)
                    retried["planner"] = retry_planner
                    retried["times"] = times
                    return retried, True, retry_details
                best_details = retry_details
                retry_scale = self._plan_validation_retry_scale(retry_details)
                if retry_scale is None:
                    break
                scale *= retry_scale
            return candidate, False, best_details

        if not valid_plan:
            candidate, valid_plan, validation_details = (
                retry_candidate_with_slower_timing(candidate, validation_details)
            )
        if not valid_plan and len(horizon_targets) >= 2:
            self._trace_plan_validation_reject(
                validation_details,
                fallback="active_gate_only",
            )
            fallback_candidate = build_candidate(
                [target.copy()],
                [target_track_id],
                [target_idx],
            )
            fallback_valid, fallback_details = self._validate_active_gate_plan_crossing(
                planner=fallback_candidate["planner"],
                target=target,
                normal=normal,
                plan_mode=fallback_candidate["mode"],
                gate_idx=target_idx,
                track_id=target_track_id,
            )
            if not fallback_valid:
                fallback_candidate, fallback_valid, fallback_details = (
                    retry_candidate_with_slower_timing(
                        fallback_candidate,
                        fallback_details,
                    )
                )
            if not fallback_valid and fallback_candidate["mode"] in (
                "single_gate_corridor",
                "single_gate_exit",
            ):
                self._trace_plan_validation_reject(
                    fallback_details,
                    fallback="single_gate",
                )
                direct_candidate = build_candidate(
                    [target.copy()],
                    [target_track_id],
                    [target_idx],
                    allow_exit=False,
                )
                direct_valid, direct_details = self._validate_active_gate_plan_crossing(
                    planner=direct_candidate["planner"],
                    target=target,
                    normal=normal,
                    plan_mode=direct_candidate["mode"],
                    gate_idx=target_idx,
                    track_id=target_track_id,
                )
                if direct_valid:
                    fallback_candidate = direct_candidate
                    fallback_valid = True
                else:
                    fallback_details = direct_details
            if not fallback_valid:
                self._trace_plan_validation_reject(
                    fallback_details,
                    fallback="none",
                )
                return False
            print(
                "plan_validation_fallback "
                f"gate_idx={target_idx} "
                f"track={target_track_id if target_track_id is not None else 'none'} "
                f"from_mode={candidate['mode']} "
                f"to_mode={fallback_candidate['mode']} "
                f"reason={validation_details.get('reason', 'unknown')}",
                flush=True,
            )
            candidate = fallback_candidate
        elif not valid_plan and candidate["mode"] in (
            "single_gate_corridor",
            "single_gate_exit",
        ):
            self._trace_plan_validation_reject(
                validation_details,
                fallback="single_gate",
            )
            fallback_candidate = build_candidate(
                [target.copy()],
                [target_track_id],
                [target_idx],
                allow_exit=False,
            )
            fallback_valid, fallback_details = self._validate_active_gate_plan_crossing(
                planner=fallback_candidate["planner"],
                target=target,
                normal=normal,
                plan_mode=fallback_candidate["mode"],
                gate_idx=target_idx,
                track_id=target_track_id,
            )
            if not fallback_valid:
                fallback_candidate, fallback_valid, fallback_details = (
                    retry_candidate_with_slower_timing(
                        fallback_candidate,
                        fallback_details,
                    )
                )
            if not fallback_valid:
                self._trace_plan_validation_reject(
                    fallback_details,
                    fallback="none",
                )
                return False
            print(
                "plan_validation_fallback "
                f"gate_idx={target_idx} "
                f"track={target_track_id if target_track_id is not None else 'none'} "
                f"from_mode={candidate['mode']} "
                f"to_mode={fallback_candidate['mode']} "
                f"reason={validation_details.get('reason', 'unknown')}",
                flush=True,
            )
            candidate = fallback_candidate
        elif not valid_plan:
            self._trace_plan_validation_reject(
                validation_details,
                fallback="none",
            )
            return False

        self.planner = candidate["planner"]
        waypoints = candidate["waypoints"]
        times = candidate["times"]
        terminal_velocity = candidate["terminal_velocity"]
        terminal_policy = candidate["terminal_policy"]
        waypoint_velocities = candidate["waypoint_velocities"]
        plan_mode = candidate["mode"]
        waypoint_roles = candidate["waypoint_roles"]
        horizon_targets = candidate["horizon_targets"]
        horizon_track_ids = candidate["horizon_track_ids"]
        horizon_gate_indices = candidate["horizon_gate_indices"]

        self.current_gate_pos = target.copy()
        self.active_waypoints = waypoints.copy()
        self.active_times = np.asarray(times, dtype=float).copy()
        self.active_waypoint_roles = list(waypoint_roles)
        self.active_horizon_gate_indices = list(horizon_gate_indices)
        self.active_horizon_track_ids = list(horizon_track_ids)
        self.active_horizon_targets = [
            np.asarray(item, dtype=float).reshape(3).copy()
            for item in horizon_targets
        ]
        self.active_plan_mode = str(plan_mode)
        self.active_terminal_velocity = terminal_velocity.copy()
        self.active_terminal_velocity_policy = str(terminal_policy)
        self.post_gate_exit_until_s = 0.0
        self.post_gate_exit_reason = ""
        self.trajectory_start_time = time.time()
        self.last_plan_wall_time = self.trajectory_start_time
        self.last_planned_gate_idx = int(self.current_gate_idx)
        self.active_plan_generation += 1
        self._reset_reference_progress_state()
        self._record_spline_memory_from_active_plan(source=str(plan_mode))
        waypoints_txt = "[" + ";".join(
            self._fmt_vec(waypoint, precision=3) for waypoint in waypoints
        ) + "]"
        times_txt = "(" + ",".join(f"{float(item):.3f}" for item in times) + ")"
        waypoint_roles_txt = "(" + ",".join(str(role) for role in waypoint_roles) + ")"
        plan_samples_txt = self._sample_planner_path_text(self.planner)
        horizon_tracks_txt = "(" + ",".join(
            str(track_id) if track_id is not None else "none"
            for track_id in horizon_track_ids
        ) + ")"
        horizon_indices_txt = "(" + ",".join(str(idx) for idx in horizon_gate_indices) + ")"
        waypoint_velocities_txt = "none"
        if waypoint_velocities is not None:
            waypoint_velocities_txt = "[" + ";".join(
                self._fmt_vec(velocity, precision=3)
                if np.all(np.isfinite(velocity))
                else "nan"
                for velocity in waypoint_velocities
            ) + "]"
        print(
            "plan_install "
            f"gate_idx={self.current_gate_idx} "
            f"track={target_track_id if target_track_id is not None else 'none'} "
            f"mode={plan_mode} "
            f"horizon_tracks={horizon_tracks_txt} "
            f"horizon_gate_indices={horizon_indices_txt} "
            f"total_time={float(self.planner.total_time):.3f} "
            f"segments={max(0, int(len(waypoints) - 1))} "
            f"target_neu={self._fmt_vec(target, precision=3)} "
            f"normal_neu={self._fmt_vec(normal, precision=3) if normal is not None else 'none'} "
            f"v_start_neu={self._fmt_vec(self._planner_v_start_used(self.planner, vel), precision=3)} "
            f"v_start_raw_neu={self._fmt_vec(self._planner_v_start_raw(self.planner, vel), precision=3)} "
            f"v_end_neu={self._fmt_vec(terminal_velocity, precision=3)} "
            f"terminal_policy={terminal_policy} "
            f"passthrough_policy={self._passthrough_velocity_policy_text()} "
            f"waypoint_velocities_neu={waypoint_velocities_txt} "
            f"times_s={times_txt} "
            f"waypoint_roles={waypoint_roles_txt} "
            f"gate_corridor={int(self.gate_corridor_enabled)}:"
            f"{float(self.gate_corridor_length_m):.2f} "
            f"waypoints_neu={waypoints_txt} "
            f"plan_samples_neu={plan_samples_txt}",
            flush=True,
        )
        self._trace_plan_boundary_continuity(
            planner=self.planner,
            waypoints=waypoints,
            plan_mode=str(plan_mode),
            horizon_track_ids=horizon_track_ids,
            horizon_gate_indices=horizon_gate_indices,
        )
        return True

    def _path_plan_provisional_next_gate(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
    ) -> bool:
        if (
            not self.provisional_next_gate_enabled
            or not self.use_perception
            or self.target_manager.locked
            or 0 <= self.current_gate_idx < len(self.gate_centers_neu)
        ):
            return False

        track, target, details = self._select_provisional_next_gate_candidate(pos, vel)
        if track is None or target is None:
            return False

        pos = np.asarray(pos, dtype=float).reshape(3)
        vel = np.asarray(vel, dtype=float).reshape(3)
        target = self._apply_target_z_policy(target)
        normal = unit_vector_from_to(pos, target)
        if normal is None:
            return False

        track_id = int(track.id)
        now = time.time()
        previous_track_id = self.provisional_target_track_id
        if (
            not self.provisional_target_active
            or previous_track_id is None
            or int(previous_track_id) != track_id
        ):
            self.provisional_target_start_time = now
            self.provisional_target_plan_count = 0

        provisional_vmax = float(self.provisional_next_gate_vmax_m_s)
        if provisional_vmax <= 0.0:
            provisional_vmax = float(self.planner_vmax)
        provisional_vmax = min(float(self.planner_vmax), provisional_vmax)

        def build_provisional_candidate(*, allow_exit: bool) -> dict:
            if (
                allow_exit
                and self.gate_corridor_enabled
                and self.gate_corridor_length_m > 0.0
            ):
                candidate_waypoints, candidate_waypoint_roles = (
                    self._build_gate_corridor_waypoints(
                        start=pos,
                        targets=[target],
                        preferred_normals=[normal],
                    )
                )
            elif allow_exit:
                pass_through_target = target + normal * self.gate_pass_through_m
                candidate_waypoints = np.vstack([pos, target, pass_through_target])
                candidate_waypoint_roles = ["start", "gate_center", "gate_exit"]
            else:
                candidate_waypoints = np.vstack([pos, target])
                candidate_waypoint_roles = ["start", "gate_center"]
            candidate_times = allocate_segment_times(
                candidate_waypoints,
                current_vel=vel,
                vmax=provisional_vmax,
                amax=self.planner_amax,
                T_min=self.planner_t_min,
            )
            candidate_terminal_velocity, candidate_terminal_policy = (
                self._terminal_velocity_for_plan(
                    waypoints=candidate_waypoints,
                    plan_mode="provisional_next_gate",
                    horizon_gate_indices=[int(self.current_gate_idx)],
                )
            )
            candidate_planner = self._build_minimum_snap_plan(
                waypoints=candidate_waypoints,
                times=candidate_times,
                v_start=vel,
                v_end=candidate_terminal_velocity,
                waypoint_velocities=None,
            )
            return {
                "planner": candidate_planner,
                "waypoints": candidate_waypoints,
                "times": candidate_times,
                "terminal_velocity": candidate_terminal_velocity,
                "terminal_policy": candidate_terminal_policy,
                "waypoint_roles": list(candidate_waypoint_roles),
                "mode": (
                    "provisional_next_gate"
                    if allow_exit
                    else "provisional_next_gate_direct"
                ),
            }

        candidate = build_provisional_candidate(allow_exit=True)
        valid_plan, validation_details = self._validate_active_gate_plan_crossing(
            planner=candidate["planner"],
            target=target,
            normal=normal,
            plan_mode=candidate["mode"],
            gate_idx=int(self.current_gate_idx),
            track_id=track_id,
        )
        if not valid_plan:
            def retry_provisional_candidate_with_slower_timing(candidate, validation_details):
                initial_scale = self._plan_validation_retry_scale(validation_details)
                if initial_scale is None:
                    return candidate, False, validation_details
                scale = float(initial_scale)
                best_details = validation_details
                for attempt in range(3):
                    times = np.asarray(candidate["times"], dtype=float) * scale
                    retry_planner = self._build_minimum_snap_plan(
                        waypoints=candidate["waypoints"],
                        times=times,
                        v_start=vel,
                        v_end=candidate["terminal_velocity"],
                        waypoint_velocities=None,
                    )
                    retry_valid, retry_details = self._validate_active_gate_plan_crossing(
                        planner=retry_planner,
                        target=target,
                        normal=normal,
                        plan_mode=candidate["mode"],
                        gate_idx=int(self.current_gate_idx),
                        track_id=track_id,
                    )
                    print(
                        "plan_validation_retry "
                        f"gate_idx={int(self.current_gate_idx)} "
                        f"track={track_id} "
                        f"mode={candidate['mode']} "
                        f"attempt={attempt + 1} "
                        f"time_scale={scale:.2f} "
                        f"valid={int(retry_valid)} "
                        f"reason={retry_details.get('reason', 'ok')} "
                        f"speed={self._fmt_float(retry_details.get('speed_m_s'), precision=2)} "
                        f"accel={self._fmt_float(retry_details.get('accel_m_s2'), precision=2)} "
                        f"accel_xy={self._fmt_float(retry_details.get('accel_xy_m_s2'), precision=2)} "
                        f"lateral_accel={self._fmt_float(retry_details.get('lateral_accel_m_s2'), precision=2)} "
                        f"z_overshoot={self._fmt_float(retry_details.get('z_overshoot_m', retry_details.get('z_undershoot_m')), precision=2)}",
                        flush=True,
                    )
                    if retry_valid:
                        retried = dict(candidate)
                        retried["planner"] = retry_planner
                        retried["times"] = times
                        return retried, True, retry_details
                    best_details = retry_details
                    retry_scale = self._plan_validation_retry_scale(retry_details)
                    if retry_scale is None:
                        break
                    scale *= retry_scale
                return candidate, False, best_details

            candidate, valid_plan, validation_details = (
                retry_provisional_candidate_with_slower_timing(
                    candidate,
                    validation_details,
                )
            )

        if not valid_plan:
            self._trace_plan_validation_reject(
                validation_details,
                fallback=(
                    "direct_target"
                    if self._allow_provisional_direct_fallback(track, details)
                    else "none_weak_direct_disabled"
                ),
            )
            if not self._allow_provisional_direct_fallback(track, details):
                print(
                    "provisional_next_gate_direct_reject "
                    f"gate_idx={int(self.current_gate_idx)} "
                    f"track={track_id} "
                    f"reason={validation_details.get('reason', 'unknown')} "
                    f"hits={self._fmt_float(details.get('hits'), precision=0)} "
                    f"lateral={self._fmt_float(details.get('lateral'), precision=2)} "
                    f"retained_spline_memory={self._fmt_float(details.get('retained_spline_memory'), precision=0)} "
                    f"retained_provisional={self._fmt_float(details.get('retained_provisional'), precision=0)}",
                    flush=True,
                )
                return False
            fallback_candidate = build_provisional_candidate(allow_exit=False)
            fallback_valid, fallback_details = self._validate_active_gate_plan_crossing(
                planner=fallback_candidate["planner"],
                target=target,
                normal=normal,
                plan_mode=fallback_candidate["mode"],
                gate_idx=int(self.current_gate_idx),
                track_id=track_id,
            )
            if not fallback_valid:
                fallback_candidate, fallback_valid, fallback_details = (
                    retry_provisional_candidate_with_slower_timing(
                        fallback_candidate,
                        fallback_details,
                    )
                )
            if not fallback_valid:
                self._trace_plan_validation_reject(
                    fallback_details,
                    fallback="none",
                )
                return False
            print(
                "plan_validation_fallback "
                f"gate_idx={int(self.current_gate_idx)} "
                f"track={track_id} "
                f"from_mode={candidate['mode']} "
                f"to_mode={fallback_candidate['mode']} "
                f"reason={validation_details.get('reason', 'unknown')}",
                flush=True,
            )
            candidate = fallback_candidate

        candidate_planner = candidate["planner"]
        waypoints = candidate["waypoints"]
        times = candidate["times"]
        terminal_velocity = candidate["terminal_velocity"]
        terminal_policy = candidate["terminal_policy"]
        waypoint_roles = candidate["waypoint_roles"]
        plan_mode = candidate["mode"]

        self.planner = candidate_planner
        self.provisional_target_active = True
        self.provisional_target_track_id = track_id
        self.provisional_target_gate_idx = int(self.current_gate_idx)
        self.provisional_target_center = target.copy()
        self.provisional_target_last_plan_time = now
        self.provisional_target_plan_count += 1
        self.current_gate_pos = target.copy()
        self.active_target_track_id = track_id
        self.last_active_target_center = target.copy()
        self.active_waypoints = waypoints.copy()
        self.active_times = np.asarray(times, dtype=float).copy()
        self.active_waypoint_roles = list(waypoint_roles)
        self.active_horizon_gate_indices = [int(self.current_gate_idx)]
        self.active_horizon_track_ids = [track_id]
        self.active_horizon_targets = [target.copy()]
        self.active_plan_mode = "provisional_next_gate"
        self.active_terminal_velocity = terminal_velocity.copy()
        self.active_terminal_velocity_policy = str(terminal_policy)
        self.trajectory_start_time = now
        self.last_plan_wall_time = now
        self.last_planned_gate_idx = int(self.current_gate_idx)
        self.active_plan_generation += 1
        self._reset_reference_progress_state()
        self._record_spline_memory_from_active_plan(source=str(plan_mode))
        self.post_gate_exit_until_s = 0.0
        self.post_gate_exit_reason = ""
        self._reset_gate_pass_state()

        waypoints_txt = "[" + ";".join(
            self._fmt_vec(waypoint, precision=3) for waypoint in waypoints
        ) + "]"
        times_txt = "(" + ",".join(f"{float(item):.3f}" for item in times) + ")"
        waypoint_roles_txt = "(" + ",".join(str(role) for role in waypoint_roles) + ")"
        plan_samples_txt = self._sample_planner_path_text(self.planner)
        nearest_plausible = details.get(
            "nearest_plausible_distance",
            details.get("nearest_stable_distance", float("inf")),
        )
        print(
            "plan_install "
            f"gate_idx={self.current_gate_idx} "
            f"track={track_id} "
            f"mode={plan_mode} "
            f"horizon_tracks=({track_id}) "
            f"horizon_gate_indices=({int(self.current_gate_idx)}) "
            f"total_time={float(self.planner.total_time):.3f} "
            f"segments={max(0, int(len(waypoints) - 1))} "
            f"target_neu={self._fmt_vec(target, precision=3)} "
            f"normal_neu={self._fmt_vec(normal, precision=3)} "
            f"v_start_neu={self._fmt_vec(self._planner_v_start_used(self.planner, vel), precision=3)} "
            f"v_start_raw_neu={self._fmt_vec(self._planner_v_start_raw(self.planner, vel), precision=3)} "
            f"v_end_neu={self._fmt_vec(terminal_velocity, precision=3)} "
            f"terminal_policy={terminal_policy} "
            f"times_s={times_txt} "
            f"waypoint_roles={waypoint_roles_txt} "
            f"gate_corridor={int(self.gate_corridor_enabled)}:"
            f"{float(self.gate_corridor_length_m):.2f} "
            f"waypoints_neu={waypoints_txt} "
            f"plan_samples_neu={plan_samples_txt} "
            f"hits={self._fmt_float(details.get('hits'), precision=0)} "
            f"age_s={self._fmt_float(details.get('age_s'), precision=2)} "
            f"reproj={self._fmt_float(details.get('reproj'), precision=2)} "
            f"kp_min={self._fmt_float(details.get('kp_min'), precision=2)} "
            f"world_std={self._fmt_float(details.get('world_std'), precision=2)} "
            f"distance={self._fmt_float(details.get('distance'), precision=2)} "
            f"projection={self._fmt_float(details.get('projection'), precision=2)} "
            f"lateral={self._fmt_float(details.get('lateral'), precision=2)} "
            f"nearest_plausible_distance={self._fmt_float(nearest_plausible, precision=2)} "
            f"nearest_stable_distance={self._fmt_float(nearest_plausible, precision=2)} "
            "pass_enabled=0",
            flush=True,
        )
        self._trace_plan_boundary_continuity(
            planner=self.planner,
            waypoints=waypoints,
            plan_mode=str(plan_mode),
            horizon_track_ids=[track_id],
            horizon_gate_indices=[int(self.current_gate_idx)],
        )
        return True

    def _apply_target_z_policy(self, target) -> np.ndarray:
        out = np.asarray(target, dtype=float).reshape(3).copy()
        if self.target_z_mode in ("expected", "expected_altitude", "fixed_expected"):
            out[2] = self.expected_gate_altitude_m
        else:
            out[2] = float(
                np.clip(
                    out[2],
                    self.safe_min_target_z,
                    self.safe_max_target_z,
                )
            )
        return out

    def _exit_tail_soft_pass_distance_m(self) -> float:
        return max(0.50, float(self.gate_pass_lateral_radius_m))

    def _expired_exit_tail_after_center_crossing(
        self,
        pos: np.ndarray,
    ) -> tuple[bool, np.ndarray | None, str, float]:
        if (
            not self.gate_plane_crossed
            or self.active_waypoints is None
            or self.planner.total_time <= 0.0
            or int(self.last_planned_gate_idx) != int(self.current_gate_idx)
            or not self._active_plan_expired()
        ):
            return False, None, "inactive", float("nan")

        exit_target, exit_role = self._active_gate_exit_waypoint()
        if exit_target is None:
            return False, None, str(exit_role), float("nan")

        pos = np.asarray(pos, dtype=float).reshape(3)
        distance = float(np.linalg.norm(pos - exit_target))
        if not math.isfinite(distance):
            return False, exit_target, str(exit_role), float("nan")
        return True, exit_target, str(exit_role), distance

    def _trace_expired_exit_tail_hold(
        self,
        *,
        pos: np.ndarray,
        exit_target: np.ndarray,
        exit_role: str,
        distance: float,
    ) -> None:
        signature = (
            int(self.current_gate_idx),
            self.active_target_track_id,
            str(exit_role),
            round(float(distance), 2) if math.isfinite(float(distance)) else "nan",
            str(self.active_plan_mode),
        )
        if signature == self._last_exit_tail_hold_signature:
            return
        self._last_exit_tail_hold_signature = signature
        print(
            "gate_exit_tail_hold_no_replan "
            f"gate_idx={int(self.current_gate_idx)} "
            f"track={self.active_target_track_id if self.active_target_track_id is not None else 'none'} "
            f"role={exit_role} "
            f"distance={self._fmt_float(distance, precision=3)} "
            f"soft_pass_distance={self._fmt_float(self._exit_tail_soft_pass_distance_m(), precision=3)} "
            f"pos_neu={self._fmt_vec(pos, precision=3)} "
            f"exit_neu={self._fmt_vec(exit_target, precision=3)} "
            f"active_plan_mode={self.active_plan_mode}",
            flush=True,
        )

    def _advance_gate_if_needed(
        self,
        pos: np.ndarray,
        vel: np.ndarray | None = None,
    ) -> bool:
        self.last_gate_pass_preserved_plan = False
        if self.provisional_target_active:
            return False
        if self.current_gate_pos is None and not self.gate_centers_neu:
            return False
        if (
            self.race_gate_count is not None
            and self.current_gate_idx >= self.race_gate_count
        ):
            return False

        if self.current_gate_pos is not None:
            target = self.current_gate_pos
        elif 0 <= self.current_gate_idx < len(self.gate_centers_neu):
            target = self.gate_centers_neu[self.current_gate_idx]
        else:
            return False

        pos = np.asarray(pos, dtype=float).reshape(3)
        vel = (
            np.zeros(3, dtype=float)
            if vel is None
            else np.nan_to_num(
                np.asarray(vel, dtype=float).reshape(3),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        )
        target = np.asarray(target, dtype=float).reshape(3)
        distance = float(np.linalg.norm(pos - target))
        pass_target, pass_target_role = self._active_gate_exit_waypoint()
        use_exit_pass_target = pass_target is not None
        normal = self.active_gate_normal
        if normal is None:
            start = (
                self.previous_gate_pass_position
                if self.previous_gate_pass_position is not None
                else pos
            )
            normal = unit_vector_from_to(start, target)
            self.active_gate_normal = None if normal is None else normal.copy()
        if normal is None:
            self.previous_gate_pass_position = pos.copy()
            self.gate_progress_along_approach = float("nan")
            return False

        previous_pos = self.previous_gate_pass_position
        if previous_pos is None:
            self.previous_gate_pass_position = pos.copy()
            self.gate_progress_along_approach = float(np.dot(pos - target, normal))
            return False

        center_pass_result = check_gate_plane_pass(
            previous_position=previous_pos,
            position=pos,
            center=target,
            normal=normal,
            lateral_radius_m=self.gate_pass_lateral_radius_m,
            plane_tolerance_m=self.gate_plane_tolerance_m,
            near_plane_pass_distance_m=(
                self.pass_radius_m if self.near_plane_pass_enabled else None
            ),
            near_plane_back_tolerance_m=self.near_plane_pass_back_tolerance_m,
            near_plane_forward_tolerance_m=self.near_plane_pass_forward_tolerance_m,
        )
        pass_result = center_pass_result
        pass_distance = distance
        if use_exit_pass_target:
            pass_result = check_gate_plane_pass(
                previous_position=previous_pos,
                position=pos,
                center=pass_target,
                normal=normal,
                lateral_radius_m=self.gate_pass_lateral_radius_m,
                plane_tolerance_m=self.gate_plane_tolerance_m,
                near_plane_pass_distance_m=None,
            )
            pass_distance = float(np.linalg.norm(pos - pass_target))
        self.previous_gate_pass_position = pos.copy()
        self.gate_plane_crossed = bool(
            self.gate_plane_crossed
            or center_pass_result.crossed_plane
            or (
                use_exit_pass_target
                and center_pass_result.passed
            )
        )
        self.near_gate_but_not_crossed = bool(
            not pass_result.passed and distance <= self.pass_radius_m
        )
        self.gate_progress_along_approach = float(center_pass_result.signed_progress_m)
        self.gate_lateral_error = float(center_pass_result.lateral_error_m)
        if use_exit_pass_target and not pass_result.passed:
            expired_tail, exit_target, exit_role, exit_distance = (
                self._expired_exit_tail_after_center_crossing(pos)
            )
            lateral_error = float(pass_result.lateral_error_m)
            soft_distance = self._exit_tail_soft_pass_distance_m()
            if (
                expired_tail
                and exit_target is not None
                and math.isfinite(exit_distance)
                and exit_distance <= soft_distance
                and math.isfinite(lateral_error)
                and lateral_error <= float(self.gate_pass_lateral_radius_m)
            ):
                pass_distance = float(exit_distance)
                pass_result = GatePlanePassResult(
                    passed=True,
                    reason="expired_exit_tail_near_exit",
                    distance_m=pass_distance,
                    signed_progress_m=float(pass_result.signed_progress_m),
                    previous_signed_progress_m=float(
                        pass_result.previous_signed_progress_m
                    ),
                    lateral_error_m=lateral_error,
                    crossed_plane=bool(pass_result.crossed_plane),
                    crossing_point=pos.copy(),
                )
                print(
                    "gate_exit_tail_soft_pass "
                    f"gate_idx={int(self.current_gate_idx)} "
                    f"track={self.active_target_track_id if self.active_target_track_id is not None else 'none'} "
                    f"role={exit_role} "
                    f"distance={pass_distance:.3f} "
                    f"soft_pass_distance={soft_distance:.3f} "
                    f"center_plane_progress={float(center_pass_result.signed_progress_m):.3f} "
                    f"exit_progress={float(pass_result.signed_progress_m):.3f} "
                    f"exit_lateral_error={lateral_error:.3f} "
                    f"pos_neu={self._fmt_vec(pos, precision=3)} "
                    f"exit_neu={self._fmt_vec(exit_target, precision=3)}",
                    flush=True,
                )
        if not pass_result.passed:
            if use_exit_pass_target and center_pass_result.passed:
                print(
                    "gate_center_pass_hold_exit "
                    f"gate_idx={int(self.current_gate_idx)} "
                    f"track={self.active_target_track_id if self.active_target_track_id is not None else 'none'} "
                    f"center_reason={center_pass_result.reason} "
                    f"center_distance={distance:.3f} "
                    f"center_plane_progress={float(center_pass_result.signed_progress_m):.3f} "
                    f"center_lateral_error={float(center_pass_result.lateral_error_m):.3f} "
                    f"pass_target_role={pass_target_role} "
                    f"pass_target_distance={pass_distance:.3f} "
                    f"pass_target_progress={float(pass_result.signed_progress_m):.3f} "
                    f"pass_target_neu={self._fmt_vec(pass_target, precision=3)}",
                    flush=True,
                )
            return False

        truth_pos = None
        truth_error = None
        if self.last_state_estimate is not None:
            truth_pos = getattr(self.last_state_estimate, "truth_pos_neu", None)
            truth_error = getattr(self.last_state_estimate, "truth_error_m", None)
        completed_gate_idx = int(self.current_gate_idx)
        completed_track_id = self.active_target_track_id
        next_gate_idx = completed_gate_idx + 1
        if self.race_gate_count is not None:
            next_gate_idx = min(next_gate_idx, int(self.race_gate_count))
        print(
            "gate_pass "
            f"gate_idx={completed_gate_idx} "
            f"track={completed_track_id if completed_track_id is not None else 'none'} "
            f"next_gate_idx={next_gate_idx} "
            f"reason={pass_target_role + '_' if use_exit_pass_target else ''}{pass_result.reason} "
            f"distance={distance:.3f} "
            f"plane_progress={float(pass_result.signed_progress_m):.3f} "
            f"lateral_error={float(pass_result.lateral_error_m):.3f} "
            f"center_plane_progress={float(center_pass_result.signed_progress_m):.3f} "
            f"center_lateral_error={float(center_pass_result.lateral_error_m):.3f} "
            f"pass_target_role={pass_target_role if use_exit_pass_target else 'gate_center'} "
            f"pass_target_distance={pass_distance:.3f} "
            f"pass_target_neu={self._fmt_vec(pass_target, precision=3) if use_exit_pass_target else self._fmt_vec(target, precision=3)} "
            f"pos_neu={self._fmt_vec(pos, precision=3)} "
            f"target_neu={self._fmt_vec(target, precision=3)} "
            f"truth_err={self._fmt_float(truth_error, precision=3)}",
            flush=True,
        )
        self._maybe_extend_race_for_forward_completion_candidate(
            completed_gate_idx=completed_gate_idx,
            completed_track_id=completed_track_id,
            pos=pos,
            target=target,
        )
        self.target_manager.mark_passed(
            pos_neu=pos,
            distance_m=pass_distance,
            pass_reason=(
                f"{pass_target_role}_{pass_result.reason}"
                if use_exit_pass_target
                else pass_result.reason
            ),
            plane_progress_m=pass_result.signed_progress_m,
            lateral_error_m=pass_result.lateral_error_m,
            truth_pos_neu=truth_pos,
            truth_error_m=truth_error,
        )
        self._record_completed_landmark(completed_track_id, target)
        self._clear_spline_memory_for_gate(
            completed_gate_idx,
            reason="passed",
            track_id=completed_track_id,
        )
        continued = self._continue_plan_after_gate_pass(
            completed_gate_idx=completed_gate_idx,
            next_gate_idx=next_gate_idx,
            pos=pos,
        )
        if not continued:
            self._sync_target_manager_state(clear_unlocked_current=True)
            planned_after_pass = self._path_plan(pos, vel)
            if not planned_after_pass:
                if self.target_manager.locked:
                    self.target_manager.clear_active(reason="post_pass_path_plan_failed")
                    self._sync_target_manager_state(clear_unlocked_current=True)
                planned_after_pass = self._path_plan_provisional_next_gate(pos, vel)
            if planned_after_pass:
                self.last_gate_pass_preserved_plan = True
                print(
                    "post_gate_fallback_plan "
                    f"completed_gate_idx={completed_gate_idx} "
                    f"next_gate_idx={next_gate_idx} "
                    f"mode={self.active_plan_mode} "
                    f"track={self.active_target_track_id if self.active_target_track_id is not None else 'none'} "
                    f"target_neu={self._fmt_vec(self.current_gate_pos, precision=3) if self.current_gate_pos is not None else 'none'}",
                    flush=True,
                )
            else:
                self.target_manager.clear_active(reason="post_pass_no_plan")
                self._sync_target_manager_state(clear_unlocked_current=True)
                self._reset_gate_pass_state()
                self.active_waypoints = None
                self.active_times = None
                self.active_waypoint_roles = []
                self.active_horizon_gate_indices = []
                self.active_horizon_track_ids = []
                self.active_horizon_targets = []
                self.active_plan_mode = ""
                self.active_terminal_velocity = np.zeros(3, dtype=float)
                self.active_terminal_velocity_policy = "cleared_after_pass"
        return True

    def _validated_horizon_continue_target(
        self,
        *,
        track_id,
        stored_target,
        next_gate_idx: int,
    ) -> tuple[bool, str, np.ndarray | None, int | None]:
        target = self._finite_vec3_or_none(stored_target)
        if target is None:
            return False, "non_finite_stored_target", None, None

        source_track_id = None
        if self.use_perception and track_id is not None:
            try:
                candidate_track_id = int(track_id)
            except (TypeError, ValueError):
                return False, "invalid_track_id", None, None
            if candidate_track_id >= 0:
                target, source_track_id, quality = self._best_duplicate_cluster_center(
                    candidate_track_id
                )
                if target is None or not bool(quality.get("ok", False)):
                    memory_target, memory_track_id, _ = self._spline_memory_for_gate(
                        next_gate_idx,
                        candidate_track_id,
                    )
                    if memory_target is not None:
                        return (
                            True,
                            "spline_memory",
                            self._apply_target_z_policy(memory_target),
                            memory_track_id,
                        )
                    return (
                        False,
                        str(quality.get("reason", "")) or "stale_horizon_track",
                        None,
                        source_track_id,
                    )
            else:
                source_track_id = candidate_track_id

        target = self._apply_target_z_policy(target)
        reason = self._target_rejection_reason(target, track_id)
        if reason:
            return False, reason, None, source_track_id

        return True, "ok", target.copy(), source_track_id

    def _continue_plan_after_gate_pass(
        self,
        *,
        completed_gate_idx: int,
        next_gate_idx: int,
        pos: np.ndarray,
    ) -> bool:
        if (
            not self.horizon_continuation_enabled
            or self.active_waypoints is None
            or self.planner.total_time <= 0.0
        ):
            return False

        pos = np.asarray(pos, dtype=float).reshape(3)
        now = time.time()
        remaining_s = self._active_plan_remaining_s()
        next_gate_idx = int(next_gate_idx)
        future_race_gate = self._has_future_race_gate(next_gate_idx)

        if future_race_gate and next_gate_idx in self.active_horizon_gate_indices:
            horizon_idx = self.active_horizon_gate_indices.index(next_gate_idx)
            if 0 <= horizon_idx < len(self.active_horizon_targets):
                target = self._apply_target_z_policy(
                    self.active_horizon_targets[horizon_idx]
                )
                track_id = (
                    self.active_horizon_track_ids[horizon_idx]
                    if horizon_idx < len(self.active_horizon_track_ids)
                    else None
                )
                valid, validation_reason, target, source_track_id = (
                    self._validated_horizon_continue_target(
                        track_id=track_id,
                        stored_target=target,
                        next_gate_idx=next_gate_idx,
                    )
                )
                if not valid or target is None:
                    self._trace_target_rejection(
                        reason=validation_reason,
                        track_id=track_id,
                        center=self.active_horizon_targets[horizon_idx],
                        context="horizon_continue",
                    )
                    return False
                if source_track_id is not None:
                    track_id = source_track_id
                locked_target = self.target_manager.lock_target(
                    gate_idx=next_gate_idx,
                    track_id=track_id,
                    center_neu=target,
                    reason=(
                        "spline_memory_horizon_continue"
                        if validation_reason == "spline_memory"
                        else "horizon_continue"
                    ),
                    now_s=now,
                )
                self._sync_target_manager_state()
                self._initialize_gate_pass_tracking(
                    pos=pos,
                    target=locked_target,
                    fallback_normal=None,
                )
                self.last_planned_gate_idx = next_gate_idx
                self.post_gate_exit_until_s = 0.0
                self.post_gate_exit_reason = ""
                self.last_gate_pass_preserved_plan = True
                print(
                    "horizon_continue_after_pass "
                    "mode=next_gate "
                    f"completed_gate_idx={int(completed_gate_idx)} "
                    f"next_gate_idx={next_gate_idx} "
                    f"track={track_id if track_id is not None else 'none'} "
                    f"horizon_idx={horizon_idx} "
                    f"remaining_s={remaining_s:.3f} "
                    f"target_neu={self._fmt_vec(locked_target, precision=3)} "
                    f"source={validation_reason} "
                    f"active_plan_mode={self.active_plan_mode} "
                    f"terminal_policy={self.active_terminal_velocity_policy}",
                    flush=True,
                )
                return True

        if (
            self.post_gate_exit_continuation_enabled
            and future_race_gate
            and self.active_plan_mode
            in (
                "single_gate_exit",
                "single_gate_corridor",
                "single_gate_corridor_exit_shift",
            )
            and remaining_s > 0.05
        ):
            self._sync_target_manager_state(clear_unlocked_current=True)
            self._reset_gate_pass_state()
            self.last_planned_gate_idx = next_gate_idx
            self.post_gate_exit_until_s = min(
                now + remaining_s + float(self.replan_after_trajectory_s),
                float(self.trajectory_start_time)
                + float(self.planner.total_time)
                + float(self.replan_after_trajectory_s),
            )
            self.post_gate_exit_reason = "waiting_for_future_gate_after_pass"
            self.last_gate_pass_preserved_plan = True
            print(
                "horizon_continue_after_pass "
                "mode=post_gate_exit "
                f"completed_gate_idx={int(completed_gate_idx)} "
                f"next_gate_idx={next_gate_idx} "
                f"remaining_s={remaining_s:.3f} "
                f"until_s={self.post_gate_exit_until_s:.3f} "
                f"active_plan_mode={self.active_plan_mode} "
                f"terminal_policy={self.active_terminal_velocity_policy} "
                f"reason={self.post_gate_exit_reason}",
                flush=True,
            )
            return True

        print(
            "horizon_continue_after_pass "
            "mode=none "
            f"completed_gate_idx={int(completed_gate_idx)} "
            f"next_gate_idx={next_gate_idx} "
            f"future_race_gate={int(bool(future_race_gate))} "
            f"remaining_s={remaining_s:.3f} "
            f"active_plan_mode={self.active_plan_mode} "
            f"horizon_gate_indices={self.active_horizon_gate_indices}",
            flush=True,
        )
        return False

    def _should_plan(
        self,
        advanced: bool,
        pos: np.ndarray,
        vel: np.ndarray,
    ) -> bool:
        if advanced:
            if self.last_gate_pass_preserved_plan:
                return False
            return True
        if self.provisional_target_active:
            return self._should_replan_provisional_target(pos, vel)
        if (
            self._post_gate_exit_active()
            and not self.target_manager.locked
            and not (0 <= self.current_gate_idx < len(self.gate_centers_neu))
        ):
            track, center, _ = self._select_provisional_next_gate_candidate(pos, vel)
            if track is not None and center is not None:
                return True
        if self.active_waypoints is None or self.planner.total_time <= 0.0:
            return True
        now = time.time()
        elapsed = now - self.trajectory_start_time
        if elapsed > float(self.planner.total_time) + self.replan_after_trajectory_s:
            expired_tail, exit_target, exit_role, exit_distance = (
                self._expired_exit_tail_after_center_crossing(pos)
            )
            if expired_tail and exit_target is not None:
                self._trace_expired_exit_tail_hold(
                    pos=pos,
                    exit_target=exit_target,
                    exit_role=exit_role,
                    distance=exit_distance,
                )
                return False
            return now - self.last_plan_wall_time >= self.replan_min_interval_s
        return self.last_planned_gate_idx != self.current_gate_idx

    def _desired_yaw(
        self,
        p_ref: np.ndarray,
        v_ref: np.ndarray,
        a_ref: np.ndarray,
        pos: np.ndarray,
        current_yaw: float,
    ) -> float:
        reference_motion_source = self._reference_motion_yaw_preference_source(pos)
        if reference_motion_source is not None:
            desired = self._reference_motion_yaw(v_ref, a_ref, self.last_desired_yaw)
            if np.isfinite(desired):
                self.last_yaw_target = None
                self.last_yaw_target_source = reference_motion_source
                self.last_desired_yaw = self._wrap_pi(desired)
                return self.last_desired_yaw

        target, source = self._yaw_target_center()
        self.last_yaw_target = None if target is None else target.copy()
        self.last_yaw_target_source = source
        if target is not None:
            behind, dot = self._yaw_target_behind_reference_motion(
                target=target,
                pos=pos,
                p_ref=p_ref,
                v_ref=v_ref,
                a_ref=a_ref,
            )
            if behind:
                desired = self._reference_motion_yaw(v_ref, a_ref, self.last_desired_yaw)
                if np.isfinite(desired):
                    self.last_yaw_target_source = (
                        f"reference_motion_{source}_behind_path"
                    )
                    self.last_desired_yaw = self._wrap_pi(desired)
                    print(
                        "yaw_target_reject "
                        f"source={source} "
                        f"reason=behind_reference_motion "
                        f"dot={self._fmt_float(dot, precision=3)} "
                        f"target_neu={self._fmt_vec(target, precision=3)} "
                        f"pos_neu={self._fmt_vec(pos, precision=3)}",
                        flush=True,
                    )
                    return self.last_desired_yaw

            to_target = np.asarray(target[:2], dtype=float) - pos[:2]
            if np.linalg.norm(to_target) > 0.30:
                desired = math.atan2(float(to_target[1]), float(to_target[0]))
                self.last_desired_yaw = self._wrap_pi(desired)
                return self.last_desired_yaw
            if np.isfinite(self.last_desired_yaw):
                self.last_yaw_target_source = f"{source}_near_hold"
                return self.last_desired_yaw
            self.last_yaw_target_source = "current_yaw_near_target"
            self.last_desired_yaw = self._wrap_pi(current_yaw)
            return self.last_desired_yaw

        desired = self._reference_motion_yaw(v_ref, a_ref, self.last_desired_yaw)
        self.last_yaw_target_source = "reference_motion"
        if not np.isfinite(desired):
            desired = current_yaw
            self.last_yaw_target_source = "current_yaw"
        self.last_desired_yaw = self._wrap_pi(desired)
        return self.last_desired_yaw

    def _yaw_target_behind_reference_motion(
        self,
        *,
        target: np.ndarray,
        pos: np.ndarray,
        p_ref: np.ndarray,
        v_ref: np.ndarray,
        a_ref: np.ndarray,
    ) -> tuple[bool, float]:
        target = self._finite_vec3_or_none(target)
        pos = self._finite_vec3_or_none(pos)
        if target is None or pos is None:
            return False, float("nan")

        direction_xy = None
        for vec in (v_ref, p_ref - pos, a_ref):
            try:
                candidate = np.asarray(vec, dtype=float).reshape(3)[:2]
            except (TypeError, ValueError):
                continue
            norm = float(np.linalg.norm(candidate))
            if math.isfinite(norm) and norm > 0.10:
                direction_xy = candidate / norm
                break
        if direction_xy is None:
            return False, float("nan")

        rel_xy = target[:2] - pos[:2]
        rel_norm = float(np.linalg.norm(rel_xy))
        if not math.isfinite(rel_norm) or rel_norm <= 0.30:
            return False, float("nan")

        dot = float(np.dot(rel_xy, direction_xy))
        return math.isfinite(dot) and dot <= 0.0, dot

    def _prefer_reference_motion_yaw_near_gate(self, pos: np.ndarray) -> bool:
        return self._reference_motion_yaw_preference_source(pos) is not None

    def _reference_motion_yaw_preference_source(self, pos: np.ndarray) -> str | None:
        if self.active_waypoints is None or self.planner.total_time <= 0.0:
            return None
        if self.gate_plane_crossed:
            exit_target, _ = self._active_gate_exit_waypoint()
            if exit_target is not None:
                return "reference_motion_after_center_crossing"
        if not self.yaw_reference_motion_near_gate_enabled:
            return None
        if self.yaw_reference_motion_distance_m <= 0.0:
            return None
        if self._post_gate_exit_active():
            return "reference_motion_post_gate_exit"
        if len(self.active_horizon_gate_indices) < 2:
            return None
        if self.current_gate_pos is None:
            return None
        try:
            pos = np.asarray(pos, dtype=float).reshape(3)
            target = np.asarray(self.current_gate_pos, dtype=float).reshape(3)
        except (TypeError, ValueError):
            return None
        distance = float(np.linalg.norm(pos - target))
        if (
            math.isfinite(distance)
            and distance <= float(self.yaw_reference_motion_distance_m)
        ):
            return "reference_motion_near_horizon_gate"
        return None

    def _yaw_target_center(self) -> tuple[np.ndarray | None, str]:
        if self.use_perception and self.active_target_track_id is not None:
            try:
                active_id = int(self.active_target_track_id)
            except (TypeError, ValueError):
                active_id = None
            if active_id is not None and active_id >= 0:
                diag = self.target_manager.diagnostics()
                try:
                    diag_active_id = (
                        None
                        if diag.active_track_id is None
                        else int(diag.active_track_id)
                    )
                except (TypeError, ValueError):
                    diag_active_id = None
                if (
                    diag.locked
                    and diag_active_id == active_id
                    and diag.event == "live_active_seen"
                ):
                    latest = self._finite_vec3_or_none(diag.latest_center)
                    if latest is not None:
                        return (
                            self._apply_target_z_policy(latest),
                            "active_target_latest",
                        )

                track = self.gate_memory.get_track_by_id(active_id)
                center, _ = self._track_filtered_center_for_navigation(track)
                if center is not None:
                    return self._apply_target_z_policy(center), "active_track_filtered"

        if self.current_gate_pos is not None:
            return (
                self._apply_target_z_policy(self.current_gate_pos),
                "locked_target",
            )
        return None, "none"

    def _gates_from_snapshot(self, snapshot) -> list[np.ndarray]:
        if self.gate_source_mode == "ground_truth":
            self._observe_perception_for_diagnostics(snapshot)
            return self._ground_truth_gates_from_config()

        if self.use_perception:
            perception_gates = self._perception_gates_from_snapshot(snapshot)
            if perception_gates:
                return perception_gates

        return self._track_gates_from_snapshot(snapshot)

    def _observe_perception_for_diagnostics(self, snapshot) -> None:
        latest_perception = getattr(snapshot, "latest_perception", None)
        if isinstance(latest_perception, dict):
            self._update_gate_memory(latest_perception)

    def _ground_truth_gates_from_config(self) -> list[np.ndarray]:
        gates = [gate.copy() for gate in self.ground_truth_gate_positions_neu]
        track_ids = list(self.ground_truth_gate_track_ids)
        if self.race_gate_count is not None:
            limit = min(int(self.race_gate_count), len(gates))
            gates = gates[:limit]
            track_ids = track_ids[:limit]

        self._candidate_gate_track_ids = track_ids
        signature = tuple(
            (track_id, *self._rounded_gate(gate, decimals=2))
            for track_id, gate in zip(track_ids, gates)
        )
        if signature != self._last_ground_truth_gate_print_signature:
            coords = " ".join(
                f"id={track_id}:({gate[0]:.2f}, {gate[1]:.2f}, {gate[2]:.2f})"
                for track_id, gate in zip(track_ids, gates)
            )
            print(f"autonomy_wrapper ground truth gates NEU: {coords}", flush=True)
            self._last_ground_truth_gate_print_signature = signature
        return gates

    def _trace_canonical_gate_poses_from_active_world(self) -> None:
        if str(self.config.runtime.runner_mode).lower() == "competition":
            return
        world_sdf = self._resolve_active_world_sdf()
        if world_sdf is None:
            return
        poses = self._canonical_gate_pose_records_from_sdf(world_sdf)
        if not poses:
            return
        payload = {
            "source": "sdf",
            "world_sdf": str(world_sdf),
            "poses": poses,
        }
        print(
            "canonical_gate_poses "
            + json.dumps(payload, separators=(",", ":"), sort_keys=True),
            flush=True,
        )

    def _resolve_active_world_sdf(self) -> Path | None:
        names: list[str] = []
        for env_name in ("AIGP_WORLD_SDF", "WORLD_SDF", "PX4_GZ_WORLD", "WORLD"):
            value = os.environ.get(env_name)
            if value:
                names.append(value)

        topic = str(
            getattr(
                self.config.perception_geometry_audit,
                "gazebo_dynamic_pose_topic",
                "",
            )
            or ""
        )
        match = re.match(r"^/world/([^/]+)/", topic)
        if match:
            names.append(match.group(1))

        seen: set[Path] = set()
        for value in names:
            for candidate in self._world_sdf_candidates(value):
                try:
                    candidate = candidate.expanduser().resolve()
                except OSError:
                    candidate = candidate.expanduser()
                if candidate in seen:
                    continue
                seen.add(candidate)
                if candidate.is_file():
                    return candidate
        return None

    @staticmethod
    def _world_sdf_candidates(value: str) -> list[Path]:
        text = str(value).strip()
        if not text:
            return []

        raw_path = Path(text).expanduser()
        candidates: list[Path] = []
        if raw_path.suffix == ".sdf" or raw_path.is_absolute() or "/" in text:
            candidates.append(raw_path)
            if raw_path.suffix != ".sdf":
                candidates.append(raw_path.with_suffix(".sdf"))

        world_name = raw_path.stem if raw_path.suffix == ".sdf" else raw_path.name
        if world_name and "/" not in world_name:
            world_roots = (
                Path.home()
                / "PX4-Autopilot"
                / "PX4-Autopilot"
                / "Tools"
                / "simulation"
                / "gz"
                / "worlds",
                Path.home()
                / "PX4-Autopilot"
                / "Tools"
                / "simulation"
                / "gz"
                / "worlds",
            )
            candidates.extend(root / f"{world_name}.sdf" for root in world_roots)
        return candidates

    def _canonical_gate_pose_records_from_sdf(self, world_sdf: Path) -> list[dict[str, object]]:
        try:
            root = ET.parse(world_sdf).getroot()
        except (OSError, ET.ParseError):
            return []

        records: list[dict[str, object]] = []
        models = []
        for model in root.findall(".//model"):
            name = str(model.get("name") or "")
            match = GATE_MODEL_RE.match(name)
            if not match:
                continue
            pose = model.find("pose")
            if pose is None:
                continue
            pose_values = self._parse_sdf_pose_values(pose.text)
            if pose_values is None:
                continue
            models.append((int(match.group(1)), name, pose_values))

        models.sort(key=lambda item: item[0])
        if self.race_gate_count is not None:
            models = models[: max(0, int(self.race_gate_count))]

        for order, (sdf_gate_index, model_name, pose_values) in enumerate(models):
            record = self._canonical_gate_pose_record_from_sdf_pose(
                gate_id=order,
                sdf_gate_index=sdf_gate_index,
                model_name=model_name,
                pose_values=pose_values,
            )
            if record is not None:
                records.append(record)
        return records

    @staticmethod
    def _parse_sdf_pose_values(text: str | None) -> tuple[float, float, float, float, float, float] | None:
        if text is None:
            return None
        try:
            values = tuple(float(part) for part in str(text).split())
        except ValueError:
            return None
        if len(values) != 6 or not all(math.isfinite(value) for value in values):
            return None
        return values

    @staticmethod
    def _canonical_gate_pose_record_from_sdf_pose(
        *,
        gate_id: int,
        sdf_gate_index: int,
        model_name: str,
        pose_values: tuple[float, float, float, float, float, float],
    ) -> dict[str, object] | None:
        x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad = pose_values
        rot = PyAIPilotAutonomyAPI._sdf_rpy_rotmat(roll_rad, pitch_rad, yaw_rad)
        local_center_sdf = np.array(
            [0.0, 0.0, float(VADR_TS_002.gate_outer_half_extent_m)],
            dtype=float,
        )
        center_sdf = np.array([x_m, y_m, z_m], dtype=float) + rot @ local_center_sdf
        normal_sdf = rot @ np.array([1.0, 0.0, 0.0], dtype=float)
        right_sdf = rot @ np.array([0.0, 1.0, 0.0], dtype=float)
        up_sdf = rot @ np.array([0.0, 0.0, 1.0], dtype=float)

        center_neu = PyAIPilotAutonomyAPI._sdf_vec_to_neu(center_sdf)
        normal_neu = PyAIPilotAutonomyAPI._unit_vec(
            PyAIPilotAutonomyAPI._sdf_vec_to_neu(normal_sdf)
        )
        right_neu = PyAIPilotAutonomyAPI._unit_vec(
            PyAIPilotAutonomyAPI._sdf_vec_to_neu(right_sdf)
        )
        up_neu = PyAIPilotAutonomyAPI._unit_vec(
            PyAIPilotAutonomyAPI._sdf_vec_to_neu(up_sdf)
        )
        if normal_neu is None or right_neu is None or up_neu is None:
            return None

        return {
            "id": int(gate_id),
            "sdf_gate_index": int(sdf_gate_index),
            "sdf_model": str(model_name),
            "center_neu": PyAIPilotAutonomyAPI._json_vec(center_neu),
            "right_axis_neu": PyAIPilotAutonomyAPI._json_vec(right_neu),
            "up_axis_neu": PyAIPilotAutonomyAPI._json_vec(up_neu),
            "normal_neu": PyAIPilotAutonomyAPI._json_vec(normal_neu),
            "sdf_pose": [float(value) for value in pose_values],
        }

    @staticmethod
    def _sdf_rpy_rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr, sr = math.cos(float(roll)), math.sin(float(roll))
        cp, sp = math.cos(float(pitch)), math.sin(float(pitch))
        cy, sy = math.cos(float(yaw)), math.sin(float(yaw))
        rx = np.array(
            [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
            dtype=float,
        )
        ry = np.array(
            [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
            dtype=float,
        )
        rz = np.array(
            [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )
        return rz @ ry @ rx

    @staticmethod
    def _sdf_vec_to_neu(value) -> np.ndarray:
        arr = np.asarray(value, dtype=float).reshape(3)
        return np.array([arr[1], arr[0], arr[2]], dtype=float)

    @staticmethod
    def _unit_vec(value) -> np.ndarray | None:
        arr = np.asarray(value, dtype=float).reshape(3)
        norm = float(np.linalg.norm(arr))
        if not math.isfinite(norm) or norm <= 1e-9:
            return None
        return arr / norm

    @staticmethod
    def _json_vec(value) -> list[float]:
        arr = np.asarray(value, dtype=float).reshape(-1)
        return [float(item) for item in arr]

    def _stable_gate_landmarks_neu(self) -> list[dict[str, object]]:
        return self.estimator_landmark_map.update_from_tracks(
            self.gate_memory.get_stable_tracks()
        )

    def _perception_gates_from_snapshot(self, snapshot) -> list[np.ndarray]:
        latest_perception = getattr(snapshot, "latest_perception", None)
        if isinstance(latest_perception, dict):
            self._update_gate_memory(latest_perception)

        pos = np.asarray(snapshot.pos_neu, dtype=float).reshape(3)
        committed_tracks = self.gate_memory.get_committed_tracks()
        committed_by_id = {int(track.id): track for track in committed_tracks}
        stable_tracks = self.gate_memory.get_stable_tracks()
        self._refresh_perception_race_order(
            stable_tracks=stable_tracks,
            committed_by_id=committed_by_id,
            current_pos=pos,
        )

        gates, track_ids = self._ordered_perception_gates(committed_by_id)
        self._candidate_gate_track_ids = list(track_ids)

        stable_signature = tuple(
            (int(track.id), *self._rounded_gate(track.center, decimals=2))
            for track in stable_tracks
        )
        if stable_signature != self._last_stable_gate_print_signature:
            coords = " ".join(
                f"id={track.id}:({track.center[0]:.2f}, {track.center[1]:.2f}, {track.center[2]:.2f})"
                for track in stable_tracks
            )
            print(f"autonomy_wrapper stable perception gates NEU: {coords}", flush=True)
            self._last_stable_gate_print_signature = stable_signature

        race_signature = tuple(
            (track_id, *self._rounded_gate(gate, decimals=2))
            for track_id, gate in zip(track_ids, gates)
        )
        if race_signature != self._last_race_order_print_signature:
            coords = " ".join(
                f"id={track_id}:({gate[0]:.2f}, {gate[1]:.2f}, {gate[2]:.2f})"
                for track_id, gate in zip(track_ids, gates)
            )
            print(
                f"autonomy_wrapper race order gates NEU: {coords}",
                flush=True,
            )
            self._last_race_order_print_signature = race_signature

        return gates

    def _race_order_track_center(
        self,
        track_id: int,
        committed_by_id: dict[int, object],
    ) -> np.ndarray | None:
        track = committed_by_id.get(int(track_id))
        if track is None and hasattr(self, "gate_memory"):
            track = self.gate_memory.get_track_by_id(int(track_id))
        if track is None:
            return None
        return self._track_navigation_center(track)

    def _race_order_duplicate_match(
        self,
        track_id: int,
        accepted_ids: list[int],
        committed_by_id: dict[int, object],
    ) -> tuple[int, int, float] | None:
        return self._race_order_duplicate_match_against_ids(
            track_id,
            accepted_ids,
            committed_by_id,
        )

    def _race_order_duplicate_match_against_ids(
        self,
        track_id: int,
        other_ids,
        committed_by_id: dict[int, object],
        radius: float | None = None,
    ) -> tuple[int, int, float] | None:
        radius = (
            float(self.race_order_duplicate_radius_m)
            if radius is None
            else float(radius)
        )
        if radius <= 0.0:
            return None
        center = self._race_order_track_center(track_id, committed_by_id)
        if center is None:
            return None
        best = None
        seen = set()
        for idx, accepted_id in enumerate(other_ids):
            accepted_id = int(accepted_id)
            if accepted_id == int(track_id) or accepted_id in seen:
                continue
            seen.add(accepted_id)
            accepted_center = self._race_order_track_center(
                accepted_id,
                committed_by_id,
            )
            if accepted_center is None:
                continue
            dist = float(np.linalg.norm(center - accepted_center))
            if not math.isfinite(dist) or dist > radius:
                continue
            if best is None or dist < best[2]:
                best = (int(idx), int(accepted_id), dist)
        return best

    def _race_order_track_score(
        self,
        track_id: int,
        committed_by_id: dict[int, object],
    ) -> tuple[float, float, float, float, float, float]:
        track = committed_by_id.get(int(track_id))
        if track is None:
            return (0.0, 0.0, 0.0, 0.0, -math.inf, -math.inf)

        reproj = self._finite_float(
            getattr(track, "reprojection_error_median", math.inf),
            math.inf,
        )
        std = getattr(track, "center_world_std", None)
        if std is not None:
            std = self._finite_vec3_or_none(std)
        std_norm = float(np.linalg.norm(std)) if std is not None else math.inf
        if not math.isfinite(reproj):
            reproj = math.inf
        if not math.isfinite(std_norm):
            std_norm = math.inf

        return (
            1.0 if bool(getattr(track, "is_stable", False)) else 0.0,
            self._finite_float(getattr(track, "stability_score", 0.0), 0.0),
            float(int(getattr(track, "inlier_count", 0))),
            float(int(getattr(track, "hits", 0))),
            -float(reproj),
            -float(std_norm),
        )

    def _trace_race_order_duplicate_skips(self, skips: list[tuple]) -> None:
        if not skips:
            self._last_race_order_duplicate_signature = None
            return
        signature = tuple(
            (
                str(action),
                int(track_id),
                int(other_id),
                round(float(dist), 2),
            )
            for action, track_id, other_id, dist in skips
        )
        if signature == self._last_race_order_duplicate_signature:
            return
        self._last_race_order_duplicate_signature = signature
        entries = " ".join(
            f"{action}={track_id}:near={other_id}:dist={float(dist):.2f}"
            for action, track_id, other_id, dist in skips[:8]
        )
        extra = "" if len(skips) <= 8 else f" more={len(skips) - 8}"
        print(
            "race_order duplicate_filter "
            f"radius={float(self.race_order_duplicate_radius_m):.2f} "
            f"{entries}{extra}",
            flush=True,
        )

    def _trace_race_order_suffix_filter(self, skips: list[tuple]) -> None:
        if not skips:
            self._last_race_order_suffix_filter_signature = None
            return

        def rounded_dist(value):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return "nan"
            return round(value, 2) if math.isfinite(value) else "nan"

        signature = tuple(
            (
                int(track_id),
                str(reason),
                None if near_id is None else int(near_id),
                rounded_dist(dist),
            )
            for track_id, reason, near_id, dist in skips[:8]
        )
        if signature == self._last_race_order_suffix_filter_signature:
            return
        self._last_race_order_suffix_filter_signature = signature

        entries = []
        for track_id, reason, near_id, dist in skips[:8]:
            entry = f"skip={int(track_id)}:{reason}"
            if near_id is not None:
                entry += f":near={int(near_id)}"
            try:
                dist = float(dist)
            except (TypeError, ValueError):
                dist = float("nan")
            if math.isfinite(dist):
                entry += f":dist={dist:.2f}"
            entries.append(entry)
        extra = "" if len(skips) <= 8 else f" more={len(skips) - 8}"
        print(
            "race_order suffix_filter "
            + " ".join(entries)
            + extra,
            flush=True,
        )

    def _duplicate_center_match(
        self,
        center,
        centers,
        *,
        radius: float | None = None,
    ) -> tuple[int, float] | None:
        center = self._finite_vec3_or_none(center)
        if center is None:
            return None
        radius = (
            float(self.race_order_duplicate_radius_m)
            if radius is None
            else float(radius)
        )
        if radius <= 0.0:
            return None
        best = None
        for idx, other in enumerate(centers):
            other = self._finite_vec3_or_none(other)
            if other is None:
                continue
            dist = float(np.linalg.norm(center - other))
            if not math.isfinite(dist) or dist > radius:
                continue
            if best is None or dist < best[1]:
                best = (int(idx), dist)
        return best

    def _refresh_perception_race_order(
        self,
        stable_tracks,
        committed_by_id: dict[int, object],
        current_pos: np.ndarray,
    ) -> None:
        current_pos = np.asarray(current_pos, dtype=float).reshape(3)

        active_id = self.active_target_track_id
        if active_id is not None:
            active_id = int(active_id)
        protected_ids = set(int(track_id) for track_id in self.completed_track_ids)
        if active_id is not None:
            protected_ids.add(int(active_id))

        accepted_ids = []
        duplicate_skips = []

        def accept_track_id(track_id: int) -> None:
            track_id = int(track_id)
            if track_id not in committed_by_id:
                return
            if track_id in accepted_ids:
                return
            duplicate = self._race_order_duplicate_match(
                track_id,
                accepted_ids,
                committed_by_id,
            )
            if duplicate is None:
                accepted_ids.append(track_id)
                return

            duplicate_idx, duplicate_id, dist = duplicate
            should_replace = (
                duplicate_id not in protected_ids
                and (
                    track_id in protected_ids
                    or self._race_order_track_score(track_id, committed_by_id)
                    > self._race_order_track_score(duplicate_id, committed_by_id)
                )
            )
            if should_replace:
                accepted_ids[duplicate_idx] = track_id
                duplicate_skips.append(("replace", duplicate_id, track_id, dist))
            else:
                duplicate_skips.append(("skip", track_id, duplicate_id, dist))

        for track_id in self.race_order_track_ids:
            accept_track_id(int(track_id))

        for track in stable_tracks:
            track_id = int(track.id)
            if track_id in self.completed_track_ids:
                continue
            if track_id in accepted_ids:
                continue
            accept_track_id(track_id)

        if active_id is not None and active_id in committed_by_id:
            if active_id not in accepted_ids:
                accepted_ids.insert(
                    min(max(int(self.current_gate_idx), 0), len(accepted_ids)),
                    active_id,
                )
        elif active_id is not None:
            active_id = None

        prefix_len = min(max(int(self.current_gate_idx), 0), len(accepted_ids))
        completed_prefix = accepted_ids[:prefix_len]
        candidate_ids = [
            track_id
            for track_id in accepted_ids[prefix_len:]
            if track_id not in self.completed_track_ids
        ]

        if (
            active_id is not None
            and active_id in committed_by_id
            and active_id not in candidate_ids
            and active_id not in completed_prefix
        ):
            duplicate = self._race_order_duplicate_match_against_ids(
                active_id,
                list(completed_prefix) + list(self.completed_track_ids),
                committed_by_id,
            )
            if duplicate is None:
                candidate_ids.insert(0, active_id)
            else:
                _, duplicate_id, dist = duplicate
                duplicate_skips.append(
                    ("skip_completed", active_id, duplicate_id, dist)
                )

        protected_duplicate_ids = list(completed_prefix) + list(self.completed_track_ids)
        filtered_candidate_ids = []
        filtered_candidate_centers: list[tuple[int, np.ndarray]] = []
        protected_candidate_centers: list[tuple[int, np.ndarray]] = []
        suffix_filter_skips = []

        def duplicate_guard_center(track_id: int) -> np.ndarray | None:
            track_id = int(track_id)
            if track_id not in committed_by_id:
                return None
            center = self._physical_navigation_center(
                track_id,
                committed_by_id,
                require_fresh=False,
            )
            if center is None:
                center = self._race_order_track_center(track_id, committed_by_id)
            return None if center is None else center.copy()

        def future_suffix_center(track_id: int) -> tuple[np.ndarray | None, str]:
            track_id = int(track_id)
            if track_id not in committed_by_id:
                return None, "missing_track"
            center = self._physical_navigation_center(
                track_id,
                committed_by_id,
                require_fresh=self.use_perception,
            )
            if center is None:
                memory_center, _, memory_gate_idx, _ = (
                    self._spline_memory_for_track_id(
                        track_id,
                        validate_target=False,
                    )
                )
                if memory_center is not None:
                    center = memory_center.copy()
                    self._trace_spline_memory(
                        action="use",
                        gate_idx=int(memory_gate_idx),
                        track_id=track_id,
                        center=center,
                        context="race_order_suffix_memory",
                    )
            if center is None:
                track = committed_by_id.get(track_id)
                retained_ok = bool(
                    track is not None
                    and (
                        bool(getattr(track, "committed", False))
                        or bool(getattr(track, "is_stable", False))
                        or bool(getattr(track, "ever_stable", False))
                    )
                )
                if retained_ok:
                    now = time.time()
                    last_seen = self._finite_float(
                        getattr(track, "last_seen_time", 0.0),
                        0.0,
                    )
                    max_age = max(
                        self._finite_float(
                            getattr(self.gate_memory, "stale_time", 0.0),
                            0.0,
                        ),
                        float(self.provisional_next_gate_max_duration_s),
                    )
                    age_s = now - last_seen if last_seen > 0.0 else float("inf")
                    obs_history = getattr(track, "obs_history", [])
                    last_obs = obs_history[-1] if obs_history else None
                    retained_ok = bool(
                        max_age > 0.0
                        and math.isfinite(age_s)
                        and age_s <= max_age
                        and last_obs is not None
                        and not bool(getattr(last_obs, "is_outlier", False))
                        and bool(getattr(last_obs, "quality_ok", True))
                    )
                if retained_ok:
                    center = self._race_order_track_center(track_id, committed_by_id)
                if center is None:
                    return None, "missing_fresh_center"
            reject_reason = self._target_rejection_reason(center, track_id)
            if reject_reason:
                return None, reject_reason
            return center.copy(), ""

        for protected_id in protected_duplicate_ids:
            protected_center = duplicate_guard_center(int(protected_id))
            if protected_center is not None:
                protected_candidate_centers.append((int(protected_id), protected_center))

        for track_id in candidate_ids:
            track_id = int(track_id)
            is_active_candidate = active_id is not None and track_id == int(active_id)
            if is_active_candidate:
                center = duplicate_guard_center(track_id)
                filtered_candidate_ids.append(track_id)
                if center is not None:
                    filtered_candidate_centers.append((track_id, center.copy()))
                continue
            else:
                center, reason = future_suffix_center(track_id)
                if center is None:
                    suffix_filter_skips.append((track_id, reason, None, float("nan")))
                    continue

            duplicate_guard = protected_candidate_centers + filtered_candidate_centers
            duplicate_centers = [item[1] for item in duplicate_guard]
            duplicate = (
                None
                if center is None
                else self._duplicate_center_match(center, duplicate_centers)
            )
            if duplicate is not None:
                duplicate_idx, dist = duplicate
                duplicate_id = duplicate_guard[duplicate_idx][0]
                duplicate_skips.append(
                    ("skip_completed", track_id, duplicate_id, dist)
                )
                suffix_filter_skips.append(
                    (track_id, "duplicate_suffix_or_completed", duplicate_id, dist)
                )
                continue
            filtered_candidate_ids.append(track_id)
            if center is not None:
                filtered_candidate_centers.append((track_id, center.copy()))
        candidate_ids = filtered_candidate_ids

        self._trace_race_order_suffix_filter(suffix_filter_skips)
        self._trace_race_order_duplicate_skips(duplicate_skips)

        self.pending_active_target_preempt_track_id = None
        self.pending_active_target_preempt_details = None
        ordered_suffix = self._order_track_ids_by_progress(
            candidate_ids=candidate_ids,
            current_pos=current_pos,
            committed_by_id=committed_by_id,
            active_id=active_id,
            protected_duplicate_ids=protected_duplicate_ids,
        )
        details = self.pending_active_target_preempt_details
        if (
            details is not None
            and ordered_suffix
            and int(ordered_suffix[0]) == int(details["new_track_id"])
        ):
            self.pending_active_target_preempt_track_id = int(details["new_track_id"])

        order = completed_prefix + ordered_suffix
        if self.race_gate_count is not None:
            order = order[: self.race_gate_count]
        self.race_order_track_ids = order

        if (
            self.active_target_track_id is None
            and 0 <= self.current_gate_idx < len(self.race_order_track_ids)
        ):
            self.active_target_track_id = self.race_order_track_ids[self.current_gate_idx]

    def _select_active_preempt_track(
        self,
        unique_ids: list[int],
        current_pos: np.ndarray,
        committed_by_id: dict[int, object],
        active_id,
    ) -> int | None:
        if (
            not self.active_target_preempt_enabled
            or not self.target_manager.locked
            or active_id is None
        ):
            return None

        active_id = int(active_id)
        if active_id not in unique_ids:
            return None

        active_center = self._race_order_track_center(active_id, committed_by_id)
        if active_center is None:
            return None

        current_pos = np.asarray(current_pos, dtype=float).reshape(3)
        active_vec = active_center - current_pos
        active_dist = float(np.linalg.norm(active_vec))
        if (
            not math.isfinite(active_dist)
            or active_dist < self.active_target_preempt_min_active_distance_m
            or active_dist < 1e-6
        ):
            return None

        direction = active_vec / active_dist
        margin = float(self.active_target_preempt_margin_m)
        lateral_limit = float(self.active_target_preempt_lateral_radius_m)
        if lateral_limit <= 0.0:
            lateral_limit = max(float(self.race_order_duplicate_radius_m), self.pass_radius_m)

        best = None
        for track_id in unique_ids:
            track_id = int(track_id)
            if track_id == active_id:
                continue
            center = self._race_order_track_center(track_id, committed_by_id)
            if center is None:
                continue

            rel = center - current_pos
            projection = float(np.dot(rel, direction))
            if not math.isfinite(projection) or projection <= 0.25:
                continue
            if projection >= active_dist - margin:
                continue

            lateral_vec = rel - projection * direction
            lateral = float(np.linalg.norm(lateral_vec))
            if not math.isfinite(lateral) or lateral > lateral_limit:
                continue

            candidate_dist = float(np.linalg.norm(rel))
            if not math.isfinite(candidate_dist) or candidate_dist + margin >= active_dist:
                continue

            key = (projection, candidate_dist, track_id)
            if best is None or key < best[0]:
                best = (
                    key,
                    {
                        "old_track_id": active_id,
                        "new_track_id": track_id,
                        "active_dist": active_dist,
                        "candidate_dist": candidate_dist,
                        "projection": projection,
                        "lateral": lateral,
                    },
                )

        if best is None:
            return None

        self.pending_active_target_preempt_details = best[1]
        return int(best[1]["new_track_id"])

    def _front_blocker_track_for_unlocked_order(
        self,
        first_id: int,
        unique_ids: list[int],
        current_pos: np.ndarray,
        committed_by_id: dict[int, object],
        protected_duplicate_ids=None,
    ) -> tuple[int | None, dict[str, float]]:
        if (
            not self.race_order_front_blocker_enabled
            or self.target_manager.locked
            or first_id not in committed_by_id
        ):
            return None, {}

        first_center = self._race_order_track_center(first_id, committed_by_id)
        if first_center is None:
            return None, {}

        current_pos = np.asarray(current_pos, dtype=float).reshape(3)
        first_vec = first_center - current_pos
        first_dist = float(np.linalg.norm(first_vec))
        if not math.isfinite(first_dist) or first_dist < 1e-6:
            return None, {}

        direction = first_vec / first_dist
        margin = float(self.race_order_front_blocker_margin_m)
        lateral_limit = float(self.race_order_front_blocker_lateral_radius_m)
        if lateral_limit <= 0.0:
            lateral_limit = max(float(self.race_order_duplicate_radius_m), self.pass_radius_m)

        unique_set = set(int(track_id) for track_id in unique_ids)
        duplicate_guard_ids = (
            unique_set
            | set(int(track_id) for track_id in self.completed_track_ids)
            | set(int(track_id) for track_id in (protected_duplicate_ids or []))
        )
        best = None
        for track_id, track in committed_by_id.items():
            track_id = int(track_id)
            if track_id == int(first_id) or track_id in unique_set:
                continue
            if track_id in self.completed_track_ids:
                continue
            duplicate = self._race_order_duplicate_match_against_ids(
                track_id,
                duplicate_guard_ids,
                committed_by_id,
            )
            if duplicate is not None:
                continue
            if not self._front_blocker_track_quality_ok(track):
                continue

            center = self._track_navigation_center(track)
            if center is None:
                continue
            rel = center - current_pos
            projection = float(np.dot(rel, direction))
            if not math.isfinite(projection) or projection <= 0.25:
                continue
            if projection + margin >= first_dist:
                continue

            lateral_vec = rel - projection * direction
            lateral = float(np.linalg.norm(lateral_vec))
            if not math.isfinite(lateral) or lateral > lateral_limit:
                continue

            distance = float(np.linalg.norm(rel))
            if not math.isfinite(distance):
                continue

            key = (projection, distance, track_id)
            if best is None or key < best[0]:
                best = (
                    key,
                    {
                        "track_id": track_id,
                        "first_track_id": int(first_id),
                        "first_dist": first_dist,
                        "projection": projection,
                        "distance": distance,
                        "lateral": lateral,
                    },
                )

        if best is None:
            return None, {}
        details = best[1]
        return int(details["track_id"]), details

    def _front_blocker_track_quality_ok(self, track) -> bool:
        if track is None or not bool(getattr(track, "committed", False)):
            return False

        min_hits = max(1, min(int(self.gate_memory.commit_hits), int(self.gate_memory.min_hits_for_stable)))
        if int(getattr(track, "hits", 0)) < min_hits:
            return False

        obs_history = getattr(track, "obs_history", [])
        if not obs_history:
            return False
        last_obs = obs_history[-1]
        if bool(getattr(last_obs, "is_outlier", False)):
            return False
        if not bool(getattr(last_obs, "quality_ok", True)):
            return False

        reproj = self._finite_float(
            getattr(last_obs, "reprojection_error", float("nan")),
            float("nan"),
            allow_nan=True,
        )
        max_reproj = float(self.max_reprojection_error_for_memory)
        if max_reproj > 0.0 and (
            not math.isfinite(reproj) or reproj > max_reproj
        ):
            return False

        kp_min = self._finite_float(
            getattr(last_obs, "keypoint_conf_min", float("nan")),
            float("nan"),
            allow_nan=True,
        )
        min_kp = float(self.min_keypoint_conf_for_memory)
        if min_kp > 0.0 and (not math.isfinite(kp_min) or kp_min < min_kp):
            return False

        return True

    def _forward_completion_candidate(
        self,
        *,
        pos: np.ndarray,
        target: np.ndarray,
        completed_track_id,
    ) -> tuple[int | None, dict]:
        if not self.use_perception or not hasattr(self, "gate_memory"):
            return None, {}

        target = self._finite_vec3_or_none(target)
        pos = self._finite_vec3_or_none(pos)
        if target is None or pos is None:
            return None, {}

        axis = None
        if self.completed_gate_positions:
            previous = self._finite_vec3_or_none(self.completed_gate_positions[-1])
            if previous is not None:
                delta = target - previous
                norm = float(np.linalg.norm(delta))
                if math.isfinite(norm) and norm >= 1e-6:
                    axis = delta / norm
        if axis is None:
            delta = target - pos
            norm = float(np.linalg.norm(delta))
            if math.isfinite(norm) and norm >= 1e-6:
                axis = delta / norm
        if axis is None:
            return None, {}

        margin = max(3.0, float(self.race_order_front_blocker_margin_m))
        lateral_limit = max(
            float(self.race_order_front_blocker_lateral_radius_m),
            float(self.race_order_duplicate_radius_m),
            float(self.gate_pass_lateral_radius_m),
        )
        if lateral_limit <= 0.0:
            lateral_limit = float(self.provisional_next_gate_max_lateral_m)

        protected_ids = set(int(track_id) for track_id in self.completed_track_ids)
        try:
            if completed_track_id is not None:
                protected_ids.add(int(completed_track_id))
        except (TypeError, ValueError):
            pass

        best = None
        for track in getattr(self.gate_memory, "tracks", []):
            try:
                track_id = int(track.id)
            except (TypeError, ValueError):
                continue
            if track_id in protected_ids:
                continue
            if not self._front_blocker_track_quality_ok(track):
                continue
            center = self._provisional_track_center(track)
            if center is None:
                continue
            if self._is_near_completed_landmark(center, radius=self._completed_landmark_radius_m()):
                continue

            rel = center - target
            projection = float(np.dot(rel, axis))
            if not math.isfinite(projection) or projection <= margin:
                continue
            lateral = float(np.linalg.norm(rel - projection * axis))
            if not math.isfinite(lateral) or lateral > lateral_limit:
                continue
            distance_from_vehicle = float(np.linalg.norm(center - pos))
            if not math.isfinite(distance_from_vehicle):
                continue

            key = (
                projection,
                lateral,
                distance_from_vehicle,
                -float(getattr(track, "hits", 0)),
                track_id,
            )
            if best is None or key < best[0]:
                best = (
                    key,
                    {
                        "track_id": track_id,
                        "center": center.copy(),
                        "projection": projection,
                        "lateral": lateral,
                        "distance": distance_from_vehicle,
                        "hits": int(getattr(track, "hits", 0)),
                    },
                )

        if best is None:
            return None, {}
        details = best[1]
        return int(details["track_id"]), details

    def _maybe_extend_race_for_forward_completion_candidate(
        self,
        *,
        completed_gate_idx: int,
        completed_track_id,
        pos: np.ndarray,
        target: np.ndarray,
    ) -> None:
        if self.race_gate_count is None:
            return
        if int(completed_gate_idx) < int(self.race_gate_count) - 1:
            return

        candidate_id, details = self._forward_completion_candidate(
            pos=pos,
            target=target,
            completed_track_id=completed_track_id,
        )
        if candidate_id is None:
            return

        old_count = int(self.race_gate_count)
        new_count = max(old_count + 1, int(completed_gate_idx) + 2)
        self.race_gate_count = new_count
        self.target_manager.race_gate_count = new_count
        if candidate_id not in self.race_order_track_ids:
            self.race_order_track_ids.append(candidate_id)

        print(
            "race_completion_deferred "
            f"completed_gate_idx={int(completed_gate_idx)} "
            f"completed_track={completed_track_id if completed_track_id is not None else 'none'} "
            f"old_gate_count={old_count} "
            f"new_gate_count={new_count} "
            f"forward_track={candidate_id} "
            f"projection={self._fmt_float(details.get('projection'), precision=2)} "
            f"lateral={self._fmt_float(details.get('lateral'), precision=2)} "
            f"distance={self._fmt_float(details.get('distance'), precision=2)} "
            f"hits={details.get('hits', 'none')} "
            f"center_neu={self._fmt_vec(details.get('center'), precision=3)}",
            flush=True,
        )

    def _track_plausible_blocker_center(self, track) -> np.ndarray | None:
        center = self._finite_vec3_or_none(
            getattr(track, "filtered_center_world", None)
        )
        if center is not None:
            return center
        return self._finite_vec3_or_none(getattr(track, "center", None))

    def _provisional_track_center(self, track) -> np.ndarray | None:
        center = self._finite_vec3_or_none(
            getattr(track, "filtered_center_world", None)
        )
        if center is not None:
            return center
        center = self._finite_vec3_or_none(getattr(track, "planning_center", None))
        if center is not None:
            return center
        return self._finite_vec3_or_none(getattr(track, "center", None))

    def _provisional_forward_axis(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
    ) -> np.ndarray | None:
        pos = np.asarray(pos, dtype=float).reshape(3)
        vel = np.asarray(vel, dtype=float).reshape(3)
        speed = float(np.linalg.norm(vel))
        if math.isfinite(speed) and speed >= 0.20:
            return vel / speed

        if self.completed_gate_positions:
            previous = self._finite_vec3_or_none(self.completed_gate_positions[-1])
            if previous is not None:
                direction = pos - previous
                norm = float(np.linalg.norm(direction))
                if math.isfinite(norm) and norm >= 0.20:
                    return direction / norm

        if self.active_gate_normal is not None:
            normal = self._finite_vec3_or_none(self.active_gate_normal)
            if normal is not None:
                norm = float(np.linalg.norm(normal))
                if math.isfinite(norm) and norm >= 1e-6:
                    return normal / norm

        return None

    def _provisional_track_quality(
        self,
        track,
        now: float,
    ) -> tuple[bool, str, dict[str, float]]:
        details: dict[str, float] = {}
        if track is None:
            return False, "missing_track", details

        hits = int(getattr(track, "hits", 0))
        details["hits"] = float(hits)
        if hits < int(self.provisional_next_gate_min_hits):
            return False, "insufficient_hits", details

        last_seen = self._finite_float(
            getattr(track, "last_seen_time", 0.0),
            0.0,
        )
        age_s = now - last_seen if last_seen > 0.0 else float("inf")
        details["age_s"] = float(age_s)
        max_age_s = float(self.provisional_next_gate_max_age_s)
        stable_retained = bool(
            getattr(track, "is_stable", False) or getattr(track, "ever_stable", False)
        )
        details["stable_retained"] = float(1.0 if stable_retained else 0.0)
        if stable_retained:
            max_age_s = max(
                max_age_s,
                self._finite_float(
                    getattr(self.gate_memory, "stale_time", max_age_s),
                    max_age_s,
                ),
            )
        if max_age_s > 0.0 and (last_seen <= 0.0 or age_s > max_age_s):
            return False, "stale", details

        obs_history = getattr(track, "obs_history", [])
        if not obs_history:
            return False, "missing_observation", details
        last_obs = obs_history[-1]
        if bool(getattr(last_obs, "is_outlier", False)):
            return False, "last_observation_outlier", details
        if not bool(getattr(last_obs, "quality_ok", True)):
            return (
                False,
                str(getattr(last_obs, "quality_reason", "")) or "quality_bad",
                details,
            )

        reproj = self._finite_float(
            getattr(last_obs, "reprojection_error", float("nan")),
            float("nan"),
            allow_nan=True,
        )
        details["reproj"] = reproj
        max_reproj = float(self.provisional_next_gate_max_reprojection_error)
        if max_reproj <= 0.0:
            max_reproj = float(self.max_reprojection_error_for_memory)
        if max_reproj > 0.0 and (
            not math.isfinite(reproj) or reproj > max_reproj
        ):
            return False, "reprojection_error_high", details

        kp_min = self._finite_float(
            getattr(last_obs, "keypoint_conf_min", float("nan")),
            float("nan"),
            allow_nan=True,
        )
        details["kp_min"] = kp_min
        min_kp = float(self.provisional_next_gate_min_keypoint_conf)
        if min_kp <= 0.0:
            min_kp = float(self.min_keypoint_conf_for_memory)
        if min_kp > 0.0 and (not math.isfinite(kp_min) or kp_min < min_kp):
            return False, "keypoint_conf_low", details

        world_std = self._finite_vec3_or_none(getattr(track, "center_world_std", None))
        world_std_norm = (
            float(np.linalg.norm(world_std))
            if world_std is not None
            else float("nan")
        )
        details["world_std"] = world_std_norm
        max_world_std = float(self.provisional_next_gate_max_world_std_m)
        if max_world_std > 0.0 and math.isfinite(world_std_norm):
            if world_std_norm > max_world_std:
                return False, "world_std_high", details

        return True, "ok", details

    def _nearest_plausible_forward_distance(
        self,
        *,
        pos: np.ndarray,
        direction: np.ndarray,
        exclude_track_id: int | None = None,
    ) -> float:
        best = float("inf")
        now = time.time()
        for track in sorted(self.gate_memory.tracks, key=lambda item: int(item.id)):
            track_id = int(track.id)
            if exclude_track_id is not None and track_id == int(exclude_track_id):
                continue
            if track_id in self.completed_track_ids:
                continue
            quality_ok, _, _ = self._closer_blocker_track_quality(track, now)
            if not quality_ok:
                continue
            center = self._track_plausible_blocker_center(track)
            if center is None:
                continue
            if self._target_rejection_reason(center, track_id):
                continue
            rel = center - pos
            projection = float(np.dot(rel, direction))
            if not math.isfinite(projection) or projection <= 0.25:
                continue
            lateral_vec = rel - projection * direction
            lateral = float(np.linalg.norm(lateral_vec))
            max_lateral = float(self.provisional_next_gate_max_lateral_m)
            if max_lateral > 0.0 and (
                not math.isfinite(lateral) or lateral > max_lateral
            ):
                continue
            distance = float(np.linalg.norm(rel))
            if math.isfinite(distance) and distance < best:
                best = distance
        return best

    def _trace_provisional_rejections(self, rejects: list[tuple]) -> None:
        if not rejects:
            self._last_provisional_reject_signature = None
            return
        signature = tuple(
            (
                int(track_id),
                str(reason),
                round(float(distance), 1) if math.isfinite(float(distance)) else "nan",
            )
            for track_id, reason, distance in rejects[:8]
        )
        if signature == self._last_provisional_reject_signature:
            return
        self._last_provisional_reject_signature = signature
        entries = " ".join(
            f"id={int(track_id)}:{reason}:dist={self._fmt_float(distance, precision=1)}"
            for track_id, reason, distance in rejects[:8]
        )
        extra = "" if len(rejects) <= 8 else f" more={len(rejects) - 8}"
        print(
            "provisional_next_gate_reject "
            f"gate_idx={int(self.current_gate_idx)} "
            f"{entries}{extra}",
            flush=True,
        )

    def _retained_provisional_next_gate_candidate(
        self,
        *,
        pos: np.ndarray,
        direction: np.ndarray,
        now: float,
    ) -> tuple[object, np.ndarray, dict[str, float]] | None:
        preferred_track_id = None
        if self.provisional_target_active:
            if (
                self.provisional_target_track_id is None
                or self.provisional_target_gate_idx is None
                or int(self.provisional_target_gate_idx) != int(self.current_gate_idx)
                or self._provisional_target_timed_out()
            ):
                return None
            try:
                preferred_track_id = int(self.provisional_target_track_id)
            except (TypeError, ValueError):
                return None

        memory_target, memory_track_id, record = self._spline_memory_for_gate(
            int(self.current_gate_idx),
            preferred_track_id,
            validate_target=False,
        )
        if memory_target is None:
            return None
        try:
            track_id = (
                int(memory_track_id)
                if memory_track_id is not None
                else int(preferred_track_id)
            )
        except (TypeError, ValueError):
            return None
        if track_id in self.completed_track_ids:
            return None

        max_age = max(
            float(self.provisional_next_gate_max_duration_s),
            self._finite_float(
                getattr(self.gate_memory, "stale_time", 0.0),
                0.0,
            ),
        )
        created_time = self._finite_float(record.get("created_time", 0.0), 0.0)
        memory_age_s = now - created_time if created_time > 0.0 else float("inf")
        if (
            not self.provisional_target_active
            and max_age > 0.0
            and (not math.isfinite(memory_age_s) or memory_age_s > max_age)
        ):
            return None

        track = self.gate_memory.get_track_by_id(track_id)
        if track is None:
            return None

        center = self._provisional_track_center(track)
        if center is None:
            center = memory_target.copy()
        if center is None:
            return None

        center = self._apply_target_z_policy(center)
        reject_reason = self._target_rejection_reason(center, track_id)
        if reject_reason:
            return None

        rel = center - pos
        distance = float(np.linalg.norm(rel))
        max_distance = float(self.provisional_next_gate_max_distance_m)
        if max_distance > 0.0 and (
            not math.isfinite(distance) or distance > max_distance
        ):
            return None

        projection = float(np.dot(rel, direction))
        if not math.isfinite(projection) or projection <= 0.50:
            return None

        lateral_vec = rel - projection * direction
        lateral = float(np.linalg.norm(lateral_vec))
        max_lateral = float(self.provisional_next_gate_max_lateral_m)
        if max_lateral > 0.0 and (
            not math.isfinite(lateral) or lateral > max_lateral
        ):
            return None

        quality_ok, quality_reason, quality = self._provisional_track_quality(
            track,
            now,
        )
        retained_memory = False
        if not quality_ok:
            if quality_reason != "stale":
                return None
            center = self._apply_target_z_policy(memory_target)
            rel = center - pos
            distance = float(np.linalg.norm(rel))
            if max_distance > 0.0 and (
                not math.isfinite(distance) or distance > max_distance
            ):
                return None
            projection = float(np.dot(rel, direction))
            if not math.isfinite(projection) or projection <= 0.50:
                return None
            lateral_vec = rel - projection * direction
            lateral = float(np.linalg.norm(lateral_vec))
            if max_lateral > 0.0 and (
                not math.isfinite(lateral) or lateral > max_lateral
            ):
                return None
            retained_memory = True

        nearest_plausible_dist = self._nearest_plausible_forward_distance(
            pos=pos,
            direction=direction,
            exclude_track_id=track_id,
        )
        details = dict(quality)
        details.update(
            {
                "track_id": float(track_id),
                "distance": distance,
                "projection": projection,
                "lateral": lateral,
                "nearest_plausible_distance": nearest_plausible_dist,
                "nearest_stable_distance": nearest_plausible_dist,
                "retained_provisional": 1.0,
                "retained_memory": 1.0 if retained_memory else 0.0,
                "retained_spline_memory": 1.0,
                "memory_age_s": memory_age_s,
            }
        )
        return track, center.copy(), details

    def _select_provisional_next_gate_candidate(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
    ) -> tuple[object | None, np.ndarray | None, dict[str, float]]:
        if (
            not self.provisional_next_gate_enabled
            or not self.use_perception
            or self.target_manager.locked
            or not self._has_future_race_gate(self.current_gate_idx)
        ):
            return None, None, {}

        pos = np.asarray(pos, dtype=float).reshape(3)
        direction = self._provisional_forward_axis(pos, vel)
        if direction is None:
            return None, None, {}

        now = time.time()
        retained = self._retained_provisional_next_gate_candidate(
            pos=pos,
            direction=direction,
            now=now,
        )
        best = None
        rejects: list[tuple] = []
        for track in sorted(self.gate_memory.tracks, key=lambda item: int(item.id)):
            track_id = int(track.id)
            distance_for_reject = float("nan")
            if track_id in self.completed_track_ids:
                rejects.append((track_id, "completed_track", distance_for_reject))
                continue

            center = self._provisional_track_center(track)
            if center is None:
                rejects.append((track_id, "missing_center", distance_for_reject))
                continue
            center = self._apply_target_z_policy(center)
            rel = center - pos
            distance = float(np.linalg.norm(rel))
            distance_for_reject = distance

            reject_reason = self._target_rejection_reason(center, track_id)
            if reject_reason:
                rejects.append((track_id, reject_reason, distance_for_reject))
                continue

            quality_ok, quality_reason, quality = self._provisional_track_quality(
                track,
                now,
            )
            if not quality_ok:
                rejects.append((track_id, quality_reason, distance_for_reject))
                continue

            max_distance = float(self.provisional_next_gate_max_distance_m)
            if max_distance > 0.0 and (
                not math.isfinite(distance) or distance > max_distance
            ):
                rejects.append((track_id, "too_far", distance_for_reject))
                continue

            projection = float(np.dot(rel, direction))
            if not math.isfinite(projection) or projection <= 0.50:
                rejects.append((track_id, "not_in_front", distance_for_reject))
                continue

            lateral_vec = rel - projection * direction
            lateral = float(np.linalg.norm(lateral_vec))
            max_lateral = float(self.provisional_next_gate_max_lateral_m)
            if max_lateral > 0.0 and (
                not math.isfinite(lateral) or lateral > max_lateral
            ):
                rejects.append((track_id, "lateral_high", distance_for_reject))
                continue

            nearest_plausible_dist = self._nearest_plausible_forward_distance(
                pos=pos,
                direction=direction,
                exclude_track_id=track_id,
            )
            closer_margin = float(self.provisional_next_gate_closer_margin_m)
            if math.isfinite(nearest_plausible_dist):
                if distance + closer_margin >= nearest_plausible_dist:
                    rejects.append(
                        (track_id, "not_closer_than_plausible", distance_for_reject)
                    )
                    continue

            details = dict(quality)
            details.update(
                {
                    "track_id": float(track_id),
                    "distance": distance,
                    "projection": projection,
                    "lateral": lateral,
                    "nearest_plausible_distance": nearest_plausible_dist,
                    "nearest_stable_distance": nearest_plausible_dist,
                }
            )
            key = (
                projection,
                distance,
                -float(getattr(track, "hits", 0)),
                float(track_id),
            )
            if best is None or key < best[0]:
                best = (key, track, center.copy(), details)

        if best is None:
            if retained is not None:
                self._last_provisional_reject_signature = None
                return retained
            self._trace_provisional_rejections(rejects)
            return None, None, {}

        if retained is not None:
            retained_track, retained_center, retained_details = retained
            retained_projection = float(retained_details.get("projection", math.inf))
            best_projection = float(best[3].get("projection", math.inf))
            try:
                best_track_id = int(best[1].id)
                retained_track_id = int(retained_track.id)
            except (TypeError, ValueError):
                best_track_id = None
                retained_track_id = None
            closer_margin = max(0.50, float(self.provisional_next_gate_closer_margin_m))
            if (
                best_track_id == retained_track_id
                or not math.isfinite(best_projection)
                or (
                    math.isfinite(retained_projection)
                    and best_projection + closer_margin >= retained_projection
                )
            ):
                self._last_provisional_reject_signature = None
                return retained_track, retained_center, retained_details

        self._last_provisional_reject_signature = None
        return best[1], best[2], best[3]

    def _clear_provisional_target(self, *, reason: str, clear_plan: bool = False) -> None:
        if self.provisional_target_active:
            print(
                "provisional_next_gate_clear "
                f"gate_idx={int(self.current_gate_idx)} "
                f"track={self.provisional_target_track_id if self.provisional_target_track_id is not None else 'none'} "
                f"reason={reason}",
                flush=True,
            )
        self.provisional_target_active = False
        self.provisional_target_track_id = None
        self.provisional_target_gate_idx = None
        self.provisional_target_center = None
        self.provisional_target_start_time = 0.0
        self.provisional_target_last_plan_time = 0.0
        self.provisional_target_plan_count = 0
        if clear_plan and self.active_plan_mode == "provisional_next_gate":
            self._clear_active_plan(reason=f"provisional_{reason}")

    def _provisional_target_timed_out(self) -> bool:
        if not self.provisional_target_active:
            return False
        max_duration = float(self.provisional_next_gate_max_duration_s)
        if max_duration <= 0.0:
            return False
        return time.time() - float(self.provisional_target_start_time) > max_duration

    def _should_replan_provisional_target(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
    ) -> bool:
        if not self.provisional_target_active:
            return False
        if self._provisional_target_timed_out():
            memory_target, _, _ = self._spline_memory_for_gate(
                int(self.current_gate_idx),
                self.provisional_target_track_id,
            )
            if memory_target is not None:
                self._clear_provisional_target(
                    reason="timeout_spline_memory_handoff",
                    clear_plan=False,
                )
                return True
            self._clear_provisional_target(reason="timeout", clear_plan=True)
            return False
        if self.active_waypoints is None or self.planner.total_time <= 0.0:
            return True
        if self._active_plan_expired():
            return True
        if time.time() - float(self.provisional_target_last_plan_time) < float(
            self.replan_min_interval_s
        ):
            return False

        track, center, _ = self._select_provisional_next_gate_candidate(pos, vel)
        if track is None or center is None:
            return False
        track_id = int(track.id)
        if self.provisional_target_track_id is None:
            return True
        if track_id != int(self.provisional_target_track_id):
            return True
        previous = self._finite_vec3_or_none(self.provisional_target_center)
        if previous is None:
            return True
        shift = float(np.linalg.norm(center - previous))
        return (
            math.isfinite(shift)
            and shift >= float(self.provisional_next_gate_replan_shift_m)
        )

    def _closer_blocker_track_quality(
        self,
        track,
        now: float,
    ) -> tuple[bool, str, dict[str, float]]:
        details: dict[str, float] = {}
        if track is None:
            return False, "missing_track", details

        committed = bool(getattr(track, "committed", False))
        stable = bool(getattr(track, "is_stable", False))
        ever_stable = bool(getattr(track, "ever_stable", False))
        details["committed"] = float(1.0 if committed else 0.0)
        details["stable"] = float(1.0 if stable else 0.0)
        details["ever_stable"] = float(1.0 if ever_stable else 0.0)
        hits = int(getattr(track, "hits", 0))
        min_hits = max(1, int(getattr(self.gate_memory, "commit_hits", 1)))
        details["hits"] = float(hits)
        if hits < min_hits:
            return False, "insufficient_hits", details

        last_seen = self._finite_float(
            getattr(track, "last_seen_time", 0.0),
            0.0,
        )
        stale_time = self._finite_float(
            getattr(self.gate_memory, "stale_time", 0.5),
            0.5,
        )
        age_s = now - last_seen if last_seen > 0.0 else float("inf")
        details["age_s"] = float(age_s)
        if last_seen <= 0.0 or age_s > stale_time:
            return False, "stale", details

        obs_history = getattr(track, "obs_history", [])
        if not obs_history:
            return False, "missing_observation", details
        last_obs = obs_history[-1]
        if bool(getattr(last_obs, "is_outlier", False)):
            return False, "last_observation_outlier", details
        if not bool(getattr(last_obs, "quality_ok", True)):
            return False, str(getattr(last_obs, "quality_reason", "")) or "quality_bad", details

        reproj = self._finite_float(
            getattr(last_obs, "reprojection_error", float("nan")),
            float("nan"),
            allow_nan=True,
        )
        details["reproj"] = reproj
        max_reproj = float(self.max_reprojection_error_for_memory)
        if max_reproj > 0.0 and (
            not math.isfinite(reproj) or reproj > max_reproj
        ):
            return False, "reprojection_error_high", details

        kp_min = self._finite_float(
            getattr(last_obs, "keypoint_conf_min", float("nan")),
            float("nan"),
            allow_nan=True,
        )
        details["kp_min"] = kp_min
        min_kp = float(self.min_keypoint_conf_for_memory)
        if min_kp > 0.0 and (not math.isfinite(kp_min) or kp_min < min_kp):
            return False, "keypoint_conf_low", details

        world_std = self._finite_vec3_or_none(getattr(track, "center_world_std", None))
        world_std_norm = (
            float(np.linalg.norm(world_std))
            if world_std is not None
            else float("nan")
        )
        details["world_std"] = world_std_norm
        max_plausible_std = max(2.0, float(self.race_order_duplicate_radius_m))
        if math.isfinite(world_std_norm) and world_std_norm > max_plausible_std:
            return False, "world_std_implausible", details

        if stable and not committed:
            reason = "stable_uncommitted_candidate"
        elif stable and committed:
            reason = "committed_stable_candidate"
        elif committed:
            reason = "committed_unstable"
        else:
            reason = "strong_uncommitted_candidate"
        return True, reason, details

    def _closer_plausible_blocker_for_target(
        self,
        first_id: int,
        unique_ids: list[int],
        current_pos: np.ndarray,
        committed_by_id: dict[int, object],
        protected_duplicate_ids=None,
    ) -> tuple[int | None, dict[str, float]]:
        if (
            not self.race_order_front_blocker_enabled
            or self.target_manager.locked
            or first_id not in committed_by_id
        ):
            return None, {}

        first_center = self._race_order_track_center(first_id, committed_by_id)
        if first_center is None:
            return None, {}

        current_pos = np.asarray(current_pos, dtype=float).reshape(3)
        first_vec = first_center - current_pos
        first_dist = float(np.linalg.norm(first_vec))
        if not math.isfinite(first_dist) or first_dist < 1e-6:
            return None, {}

        direction = first_vec / first_dist
        margin = float(self.race_order_front_blocker_margin_m)
        lateral_limit = float(self.race_order_front_blocker_lateral_radius_m)
        if lateral_limit <= 0.0:
            lateral_limit = max(float(self.race_order_duplicate_radius_m), self.pass_radius_m)

        unique_set = set(int(track_id) for track_id in unique_ids)
        duplicate_guard_ids = (
            unique_set
            | set(int(track_id) for track_id in self.completed_track_ids)
            | set(int(track_id) for track_id in (protected_duplicate_ids or []))
        )
        now = time.time()
        best = None
        for track in sorted(self.gate_memory.tracks, key=lambda tr: int(tr.id)):
            track_id = int(track.id)
            if track_id == int(first_id) or track_id in unique_set:
                continue
            if track_id in self.completed_track_ids:
                continue
            duplicate = self._race_order_duplicate_match_against_ids(
                track_id,
                duplicate_guard_ids,
                committed_by_id,
            )
            if duplicate is not None:
                continue

            quality_ok, quality_reason, quality_details = (
                self._closer_blocker_track_quality(track, now)
            )
            if not quality_ok:
                continue

            center = self._track_plausible_blocker_center(track)
            if center is None:
                continue

            rel = center - current_pos
            projection = float(np.dot(rel, direction))
            if not math.isfinite(projection) or projection <= 0.25:
                continue
            if projection + margin >= first_dist:
                continue

            lateral_vec = rel - projection * direction
            lateral = float(np.linalg.norm(lateral_vec))
            if not math.isfinite(lateral) or lateral > lateral_limit:
                continue

            distance = float(np.linalg.norm(rel))
            if not math.isfinite(distance) or distance + margin >= first_dist:
                continue

            key = (projection, distance, track_id)
            if best is None or key < best[0]:
                best = (
                    key,
                    {
                        "track_id": track_id,
                        "first_track_id": int(first_id),
                        "first_dist": first_dist,
                        "projection": projection,
                        "distance": distance,
                        "lateral": lateral,
                        "reason": quality_reason,
                        "hits": quality_details.get("hits", float("nan")),
                        "age_s": quality_details.get("age_s", float("nan")),
                        "reproj": quality_details.get("reproj", float("nan")),
                        "kp_min": quality_details.get("kp_min", float("nan")),
                        "world_std": quality_details.get("world_std", float("nan")),
                    },
                )

        if best is None:
            return None, {}
        details = best[1]
        return int(details["track_id"]), details

    def _trace_closer_plausible_blocker(
        self,
        details: dict[str, float],
    ) -> None:
        if not details:
            return
        signature = (
            int(details["track_id"]),
            int(details["first_track_id"]),
            round(float(details["first_dist"]), 1),
            round(float(details["projection"]), 1),
            round(float(details["lateral"]), 1),
            str(details.get("reason", "")),
        )
        if signature == self._last_race_order_closer_blocker_signature:
            return
        self._last_race_order_closer_blocker_signature = signature
        print(
            "race_order closer_plausible_blocker "
            f"track={int(details['track_id'])} "
            f"before={int(details['first_track_id'])} "
            f"reason={details.get('reason', '')} "
            f"first_dist={float(details['first_dist']):.2f} "
            f"projection={float(details['projection']):.2f} "
            f"distance={float(details['distance']):.2f} "
            f"lateral={float(details['lateral']):.2f} "
            f"hits={self._fmt_float(details.get('hits'), precision=0)} "
            f"age_s={self._fmt_float(details.get('age_s'), precision=2)} "
            f"reproj={self._fmt_float(details.get('reproj'), precision=2)} "
            f"kp_min={self._fmt_float(details.get('kp_min'), precision=2)} "
            f"world_std={self._fmt_float(details.get('world_std'), precision=2)}",
            flush=True,
        )

    def _trace_front_blocker_order(
        self,
        details: dict[str, float],
    ) -> None:
        if not details:
            return
        signature = (
            int(details["track_id"]),
            int(details["first_track_id"]),
            round(float(details["first_dist"]), 1),
            round(float(details["projection"]), 1),
            round(float(details["lateral"]), 1),
        )
        if signature == self._last_race_order_front_blocker_signature:
            return
        self._last_race_order_front_blocker_signature = signature
        print(
            "race_order front_blocker "
            f"track={int(details['track_id'])} "
            f"before={int(details['first_track_id'])} "
            f"first_dist={float(details['first_dist']):.2f} "
            f"projection={float(details['projection']):.2f} "
            f"distance={float(details['distance']):.2f} "
            f"lateral={float(details['lateral']):.2f}",
            flush=True,
        )

    def _order_track_ids_by_progress(
        self,
        candidate_ids: list[int],
        current_pos: np.ndarray,
        committed_by_id: dict[int, object],
        active_id,
        protected_duplicate_ids=None,
    ) -> list[int]:
        unique_ids = []
        for track_id in candidate_ids:
            track_id = int(track_id)
            if track_id in committed_by_id and track_id not in unique_ids:
                unique_ids.append(track_id)
        if not unique_ids:
            return []

        current_pos = np.asarray(current_pos, dtype=float).reshape(3)

        def center_for(track_id: int) -> np.ndarray:
            center = self._race_order_track_center(int(track_id), committed_by_id)
            if center is None:
                return np.asarray(
                    committed_by_id[int(track_id)].center,
                    dtype=float,
                ).reshape(3)
            return center

        preempt_id = self._select_active_preempt_track(
            unique_ids,
            current_pos,
            committed_by_id,
            active_id,
        )
        if preempt_id is not None:
            first_id = int(preempt_id)
        elif active_id is not None and int(active_id) in unique_ids:
            first_id = int(active_id)
        else:
            first_id = min(
                unique_ids,
                key=lambda track_id: (
                    float(np.linalg.norm(center_for(track_id) - current_pos)),
                    int(track_id),
                ),
            )

        if active_id is None:
            blocker_id, blocker_details = self._closer_plausible_blocker_for_target(
                first_id,
                unique_ids,
                current_pos,
                committed_by_id,
                protected_duplicate_ids=protected_duplicate_ids,
            )
            if blocker_id is not None:
                self._trace_closer_plausible_blocker(blocker_details)
                return []

            blocker_id, blocker_details = self._front_blocker_track_for_unlocked_order(
                first_id,
                unique_ids,
                current_pos,
                committed_by_id,
                protected_duplicate_ids=protected_duplicate_ids,
            )
            if blocker_id is not None:
                first_id = int(blocker_id)
                if blocker_id not in unique_ids:
                    unique_ids.insert(0, blocker_id)
                self._trace_front_blocker_order(blocker_details)

        rest = [track_id for track_id in unique_ids if track_id != first_id]
        if not rest:
            return [first_id]

        first_center = center_for(first_id)
        farthest_id = max(
            rest,
            key=lambda track_id: float(np.linalg.norm(center_for(track_id) - current_pos)),
        )
        course = center_for(farthest_id) - first_center
        norm = float(np.linalg.norm(course))
        if norm < 1e-6:
            course = first_center - current_pos
            norm = float(np.linalg.norm(course))
        if norm < 1e-6:
            rest.sort(
                key=lambda track_id: (
                    float(np.linalg.norm(center_for(track_id) - current_pos)),
                    int(track_id),
                )
            )
            return [first_id] + rest
        course = course / norm

        if float(np.dot(first_center - current_pos, course)) < 0.0:
            course = -course

        rest.sort(
            key=lambda track_id: (
                float(np.dot(center_for(track_id) - current_pos, course)),
                float(np.linalg.norm(center_for(track_id) - current_pos)),
                int(track_id),
            )
        )
        return [first_id] + rest

    def _ordered_perception_gates(
        self,
        committed_by_id: dict[int, object],
    ) -> tuple[list[np.ndarray], list[int]]:
        gates = []
        track_ids = []
        duplicate_skips = []
        now = time.time()

        current_idx = max(0, int(self.current_gate_idx))
        for order_idx, track_id in enumerate(self.race_order_track_ids):
            track_id = int(track_id)
            center = None
            track = committed_by_id.get(track_id)
            is_completed_prefix = order_idx < current_idx
            is_active_track = track_id == self.active_target_track_id
            if track is not None:
                center = self._physical_navigation_center(
                    track_id,
                    committed_by_id,
                    require_fresh=self.use_perception and not is_completed_prefix,
                )
                if center is None and is_completed_prefix:
                    center = self._track_navigation_center(track)
                if center is None and not is_completed_prefix:
                    memory_center, _, memory_gate_idx, _ = (
                        self._spline_memory_for_track_id(
                            track_id,
                            validate_target=False,
                        )
                    )
                    if memory_center is not None:
                        center = memory_center.copy()
                        self._trace_spline_memory(
                            action="use",
                            gate_idx=int(memory_gate_idx),
                            track_id=track_id,
                            center=center,
                            context="ordered_gate_memory",
                        )
            if center is None and is_active_track and self.last_active_target_center is not None:
                if self.active_target_lost_time is None:
                    self.active_target_lost_time = now
                if now - self.active_target_lost_time <= self.active_target_lost_grace_s:
                    center = self.last_active_target_center.copy()
            elif center is not None and is_active_track:
                self.last_active_target_center = center.copy()
                self.active_target_lost_time = None

            if center is None or not np.all(np.isfinite(center)):
                continue
            reject_reason = (
                ""
                if is_completed_prefix
                else self._target_rejection_reason(center, track_id)
            )
            if reject_reason:
                self._trace_target_rejection(
                    reason=reject_reason,
                    track_id=track_id,
                    center=center,
                    context="race_order",
                )
                continue

            if not is_completed_prefix:
                duplicate = self._duplicate_center_match(center, gates)
                if duplicate is not None:
                    duplicate_idx, dist = duplicate
                    duplicate_track = (
                        track_ids[duplicate_idx]
                        if duplicate_idx < len(track_ids)
                        else -1
                    )
                    duplicate_skips.append(
                        ("skip_ordered", track_id, duplicate_track, dist)
                    )
                    continue

            gates.append(center)
            track_ids.append(track_id)

        self._trace_race_order_duplicate_skips(duplicate_skips)
        return gates, track_ids

    def _detection_memory_quality(
        self,
        detection: dict,
        center_camera: np.ndarray | None,
    ) -> tuple[bool, str, dict[str, float]]:
        details: dict[str, float] = {}
        keypoints = self._detection_keypoints_px(detection)
        if keypoints is None:
            return False, "missing_keypoints", details

        conf = self._detection_keypoint_conf_array(detection)
        if conf is None:
            return False, "missing_keypoint_conf", details
        min_conf = float(np.min(conf))
        details["kp_min"] = min_conf
        threshold = float(self.min_keypoint_conf_for_memory)
        if threshold > 0.0 and min_conf < threshold:
            return False, f"keypoint_conf_low:{min_conf:.2f}", details

        margin = float(self.keypoint_border_margin_px)
        if margin > 0.0:
            width = float(self.config.camera.width)
            height = float(self.config.camera.height)
            x = keypoints[:, 0]
            y = keypoints[:, 1]
            if (
                np.any(x <= margin)
                or np.any(y <= margin)
                or np.any(x >= width - 1.0 - margin)
                or np.any(y >= height - 1.0 - margin)
            ):
                return False, "keypoint_on_image_border", details

        area = self._quad_area_px2(keypoints)
        details["quad_area_px2"] = area
        min_area = float(self.min_quad_area_px2_for_memory)
        if min_area > 0.0 and (not math.isfinite(area) or area < min_area):
            return False, f"quad_area_small:{area:.1f}", details

        edge_ratio = self._opposite_edge_ratio(keypoints)
        details["opposite_edge_ratio"] = edge_ratio
        max_edge_ratio = float(self.max_keypoint_opposite_edge_ratio)
        if max_edge_ratio > 0.0 and (
            not math.isfinite(edge_ratio) or edge_ratio > max_edge_ratio
        ):
            return False, f"opposite_edge_ratio_high:{edge_ratio:.2f}", details

        if center_camera is not None:
            z_size = self._keypoint_size_depth_m(keypoints)
            pnp_z = float(center_camera[2])
            details["z_from_width_height"] = z_size
            details["pnp_z"] = pnp_z
            if (
                math.isfinite(z_size)
                and z_size > 0.0
                and math.isfinite(pnp_z)
                and pnp_z > 0.0
            ):
                max_abs = float(self.max_pnp_size_depth_disagreement_m)
                max_ratio = float(self.max_pnp_size_depth_disagreement_ratio)
                allowed = 0.0
                if max_abs > 0.0:
                    allowed = max(allowed, max_abs)
                if max_ratio > 0.0:
                    allowed = max(allowed, max_ratio * z_size)
                disagreement = abs(pnp_z - z_size)
                details["size_depth_disagreement_m"] = disagreement
                if allowed > 0.0 and disagreement > allowed:
                    return (
                        False,
                        f"size_depth_disagreement:{disagreement:.2f}",
                        details,
                    )

        return True, "ok", details

    def _maybe_print_perception_memory_reject(
        self,
        detection: dict,
        reason: str,
        details: dict[str, float] | None = None,
    ) -> None:
        now = time.time()
        if now - self._last_perception_reject_print_time < 0.5:
            return
        self._last_perception_reject_print_time = now
        details = details or {}
        print(
            "perception_memory_reject "
            f"reason={reason} "
            f"det={self._blank_na(detection.get('detection_id'))} "
            f"reproj={self._fmt_float(detection.get('reprojection_error'), precision=2)} "
            f"kp_min={self._fmt_float(details.get('kp_min'), precision=2)} "
            f"area={self._fmt_float(details.get('quad_area_px2'), precision=1)} "
            f"edge_ratio={self._fmt_float(details.get('opposite_edge_ratio'), precision=2)} "
            f"pnp_z={self._fmt_float(details.get('pnp_z'), precision=2)} "
            f"z_size={self._fmt_float(details.get('z_from_width_height'), precision=2)}",
            flush=True,
        )

    @staticmethod
    def _detection_keypoints_px(detection: dict) -> np.ndarray | None:
        for key in ("keypoints_px", "yolo_keypoints", "ordered_corners", "raw_corners"):
            value = detection.get(key)
            if value is None:
                continue
            try:
                arr = np.asarray(value, dtype=float)
            except (TypeError, ValueError):
                continue
            if arr.ndim >= 2 and arr.shape[0] >= 4 and arr.shape[1] >= 2:
                keypoints = np.asarray(arr[:4, :2], dtype=float).reshape(4, 2)
                if np.all(np.isfinite(keypoints)):
                    return keypoints
        return None

    @staticmethod
    def _detection_keypoint_conf_array(detection: dict) -> np.ndarray | None:
        conf = detection.get("keypoint_conf")
        if conf is not None:
            try:
                arr = np.asarray(conf, dtype=float).reshape(-1)[:4]
            except (TypeError, ValueError):
                arr = np.empty(0, dtype=float)
            if arr.size >= 4 and np.all(np.isfinite(arr)):
                return arr.astype(float).copy()

        yolo_keypoints = detection.get("yolo_keypoints")
        if yolo_keypoints is not None:
            try:
                keypoints = np.asarray(yolo_keypoints, dtype=float)
            except (TypeError, ValueError):
                keypoints = np.empty((0, 0), dtype=float)
            if (
                keypoints.ndim >= 2
                and keypoints.shape[0] >= 4
                and keypoints.shape[1] >= 3
            ):
                arr = np.asarray(keypoints[:4, 2], dtype=float).reshape(4)
                if np.all(np.isfinite(arr)):
                    return arr.copy()

        return None

    @staticmethod
    def _quad_area_px2(keypoints: np.ndarray) -> float:
        pts = np.asarray(keypoints, dtype=float).reshape(4, 2)
        if not np.all(np.isfinite(pts)):
            return float("nan")
        return 0.5 * abs(
            float(
                np.dot(pts[:, 0], np.roll(pts[:, 1], -1))
                - np.dot(pts[:, 1], np.roll(pts[:, 0], -1))
            )
        )

    @staticmethod
    def _opposite_edge_ratio(keypoints: np.ndarray) -> float:
        pts = np.asarray(keypoints, dtype=float).reshape(4, 2)
        edges = np.asarray(
            [
                np.linalg.norm(pts[1] - pts[0]),
                np.linalg.norm(pts[2] - pts[1]),
                np.linalg.norm(pts[3] - pts[2]),
                np.linalg.norm(pts[0] - pts[3]),
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(edges)) or np.any(edges <= 1.0):
            return float("inf")
        top_bottom = max(edges[0], edges[2]) / max(min(edges[0], edges[2]), 1e-6)
        right_left = max(edges[1], edges[3]) / max(min(edges[1], edges[3]), 1e-6)
        return float(max(top_bottom, right_left))

    def _keypoint_size_depth_m(self, keypoints: np.ndarray) -> float:
        pts = np.asarray(keypoints, dtype=float).reshape(4, 2)
        if not np.all(np.isfinite(pts)):
            return float("nan")
        width_px = 0.5 * (
            np.linalg.norm(pts[1] - pts[0])
            + np.linalg.norm(pts[2] - pts[3])
        )
        height_px = 0.5 * (
            np.linalg.norm(pts[3] - pts[0])
            + np.linalg.norm(pts[2] - pts[1])
        )
        k = np.asarray(self.camera_matrix, dtype=float).reshape(3, 3)
        gate_size = float(self.config.perception.gate_size_m)
        depths = []
        if width_px > 1.0 and math.isfinite(float(k[0, 0])):
            depths.append(float(k[0, 0]) * gate_size / float(width_px))
        if height_px > 1.0 and math.isfinite(float(k[1, 1])):
            depths.append(float(k[1, 1]) * gate_size / float(height_px))
        if not depths:
            return float("nan")
        return float(np.mean(depths))

    def _update_gate_memory(self, latest_perception: dict) -> None:
        frame_key = self._perception_frame_key(latest_perception)
        if frame_key is not None and frame_key == self._last_gate_memory_frame_key:
            return
        self._last_gate_memory_frame_key = frame_key

        detections = latest_perception.get("detections")
        if not detections:
            return

        timestamp = self._finite_float(
            latest_perception.get("perception_wall_time"),
            time.time(),
        )
        for detection in sorted(detections, key=self._detection_sort_key):
            position = detection.get("gate_center_world")
            if position is None:
                continue

            arr = np.asarray(position, dtype=float)
            if arr.shape != (3,) or not np.all(np.isfinite(arr)):
                continue

            confidence = self._finite_float(
                detection.get("memory_confidence", detection.get("confidence")),
                0.0,
            )
            reprojection_error = self._finite_float(
                detection.get("reprojection_error"),
                np.nan,
                allow_nan=True,
            )
            if (
                self.max_reprojection_error_for_memory > 0.0
                and (
                    not math.isfinite(reprojection_error)
                    or reprojection_error > self.max_reprojection_error_for_memory
                )
            ):
                self._maybe_print_perception_memory_reject(
                    detection,
                    "reprojection_error_high",
                    {},
                )
                continue

            center_camera = detection.get("gate_center_camera")
            if center_camera is not None:
                center_camera = np.asarray(center_camera, dtype=float)
                if center_camera.shape != (3,) or not np.all(np.isfinite(center_camera)):
                    center_camera = None
            if center_camera is not None:
                depth_m = float(center_camera[2])
                if self.reject_negative_depth and depth_m <= 0.0:
                    continue
                if (
                    self.min_depth_m_for_memory > 0.0
                    and depth_m < self.min_depth_m_for_memory
                ):
                    continue
                if (
                    self.max_depth_m_for_memory > 0.0
                    and depth_m > self.max_depth_m_for_memory
                ):
                    continue
                if (
                    self.max_detection_range_m > 0.0
                    and float(np.linalg.norm(center_camera)) > self.max_detection_range_m
                ):
                    continue

            quality_ok, quality_reason, quality_details = (
                self._detection_memory_quality(detection, center_camera)
            )
            if not quality_ok:
                self._maybe_print_perception_memory_reject(
                    detection,
                    quality_reason,
                    quality_details,
                )
                continue

            keypoint_conf_min, keypoint_conf_mean = (
                self._detection_keypoint_conf_summary(detection)
            )
            memory_result = self.gate_memory.add_detection(
                center=arr.copy(),
                confidence=confidence,
                timestamp=timestamp,
                center_camera=center_camera,
                reprojection_error=reprojection_error,
                keypoint_conf_min=keypoint_conf_min,
                keypoint_conf_mean=keypoint_conf_mean,
                solver_name="latest_perception",
                quality_ok=True,
                quality_reason=quality_reason,
            )
            self._maybe_print_perception_chain_event(detection, memory_result)

        self.gate_memory.prune(timestamp)

    def _maybe_print_perception_chain_event(
        self,
        detection: dict,
        memory_result: dict | None,
    ) -> None:
        if not isinstance(memory_result, dict) or not memory_result.get("accepted", False):
            return

        events = []
        if memory_result.get("reason") == "new_track":
            events.append("new_track")
        if memory_result.get("committed_now", False):
            events.append("committed")
        if memory_result.get("stable_now", False):
            events.append("stable")
        if not events:
            return

        track_id = memory_result.get("track_id")
        track = None
        try:
            if track_id is not None:
                track = self.gate_memory.get_track_by_id(int(track_id))
        except (TypeError, ValueError):
            track = None

        for event in events:
            self._print_perception_chain_event(event, detection, memory_result, track)

    def _print_perception_chain_event(
        self,
        event: str,
        detection: dict,
        memory_result: dict,
        track,
    ) -> None:
        track_id = memory_result.get("track_id", "none")
        hits = getattr(track, "hits", "none")
        committed = getattr(track, "committed", memory_result.get("committed", False))
        stable = getattr(track, "is_stable", memory_result.get("stable", False))
        track_center = getattr(track, "center", memory_result.get("center"))
        world_std = getattr(track, "center_world_std", None)
        cam_median = self._track_camera_median(track)
        rpy_rad = detection.get("drone_rpy_rad_used")
        rpy_deg = None if rpy_rad is None else np.rad2deg(np.asarray(rpy_rad, dtype=float))
        method = str(detection.get("body_to_world_method_used", ""))

        print(
            "[PERCEPTION_CHAIN] "
            f"event={event} "
            f"track={track_id} "
            f"reason={memory_result.get('reason', '')} "
            f"hits={hits} "
            f"committed={bool(committed)} "
            f"stable={bool(stable)} "
            f"order={self._blank_na(detection.get('pnp_selected_order'))} "
            f"solver={self._blank_na(detection.get('pnp_selected_solver'))} "
            f"debug_best_order={self._blank_na(detection.get('pnp_debug_best_order'))} "
            f"reproj={self._fmt_float(detection.get('reprojection_error'), precision=2)} "
            f"conf={self._fmt_float(detection.get('memory_confidence'), precision=2)} "
            f"cam={self._fmt_vec(detection.get('gate_center_camera'), precision=2)} "
            f"body={self._fmt_vec(detection.get('gate_center_body_frd'), precision=2)} "
            f"world_neu={self._fmt_vec(detection.get('gate_center_world'), precision=2)} "
            f"world_ned={self._fmt_vec(detection.get('gate_center_world_ned'), precision=2)} "
            f"track_center={self._fmt_vec(track_center, precision=2)} "
            f"cam_median={self._fmt_vec(cam_median, precision=2)} "
            f"world_std={self._fmt_vec(world_std, precision=2)} "
            f"drone_ned={self._fmt_vec(detection.get('drone_pos_ned'), precision=2)} "
            f"rpy_deg={self._fmt_vec(rpy_deg, precision=1)} "
            f"kp={self._fmt_keypoints(detection)} "
            f"method={method}",
            flush=True,
        )

    @staticmethod
    def _track_camera_median(track):
        if track is None:
            return None
        camera_points = [
            obs.center_camera
            for obs in getattr(track, "obs_history", [])
            if getattr(obs, "center_camera", None) is not None
            and not bool(getattr(obs, "is_outlier", False))
        ]
        if not camera_points:
            return None
        arr = np.asarray(camera_points, dtype=float).reshape(-1, 3)
        if not np.all(np.isfinite(arr)):
            arr = arr[np.all(np.isfinite(arr), axis=1)]
        if arr.size == 0:
            return None
        return np.median(arr, axis=0)

    @staticmethod
    def _fmt_vec(value, precision: int = 2) -> str:
        if value is None:
            return "none"
        try:
            arr = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return "none"
        if arr.size < 3:
            return "none"
        vals = []
        for val in arr[:3]:
            if math.isfinite(float(val)):
                vals.append(f"{float(val):.{int(precision)}f}")
            else:
                vals.append("nan")
        return "(" + ",".join(vals) + ")"

    @staticmethod
    def _fmt_float(value, precision: int = 2) -> str:
        try:
            val = float(value)
        except (TypeError, ValueError):
            return "nan"
        if not math.isfinite(val):
            return "nan"
        return f"{val:.{int(precision)}f}"

    @staticmethod
    def _is_finite_number(value) -> bool:
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _blank_na(value) -> str:
        text = str(value or "")
        return text if text else "n/a"

    @staticmethod
    def _detection_keypoint_conf_summary(detection: dict) -> tuple[float, float]:
        conf = detection.get("keypoint_conf")
        if conf is None:
            yolo_keypoints = detection.get("yolo_keypoints")
            if yolo_keypoints is not None:
                try:
                    keypoints = np.asarray(yolo_keypoints, dtype=float)
                    if keypoints.ndim == 2 and keypoints.shape[0] >= 4 and keypoints.shape[1] >= 3:
                        conf = keypoints[:4, 2]
                except (TypeError, ValueError):
                    conf = None
        if conf is None:
            return float("nan"), float("nan")
        try:
            arr = np.asarray(conf, dtype=float).reshape(-1)[:4]
        except (TypeError, ValueError):
            return float("nan"), float("nan")
        if arr.size < 4 or not np.all(np.isfinite(arr)):
            return float("nan"), float("nan")
        return float(np.min(arr)), float(np.mean(arr))

    @staticmethod
    def _fmt_keypoints(detection: dict) -> str:
        keypoints = detection.get("keypoints_px")
        if keypoints is None:
            return "none"
        try:
            kp = np.asarray(keypoints, dtype=float)
        except (TypeError, ValueError):
            return "none"
        if kp.ndim != 2 or kp.shape[0] < 4 or kp.shape[1] < 2:
            return "none"

        conf = detection.get("keypoint_conf")
        conf_arr = None
        if conf is not None:
            try:
                conf_arr = np.asarray(conf, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                conf_arr = None

        items = []
        for idx in range(4):
            u = float(kp[idx, 0])
            v = float(kp[idx, 1])
            if not math.isfinite(u) or not math.isfinite(v):
                items.append(f"{idx}:nan,nan")
                continue
            if conf_arr is not None and conf_arr.size > idx and math.isfinite(float(conf_arr[idx])):
                items.append(f"{idx}:{u:.0f},{v:.0f},{float(conf_arr[idx]):.2f}")
            else:
                items.append(f"{idx}:{u:.0f},{v:.0f}")
        return "[" + " ".join(items) + "]"

    @staticmethod
    def _perception_frame_key(latest_perception: dict):
        frame_id = latest_perception.get("frame_id")
        if frame_id is not None:
            try:
                frame_id = int(frame_id)
            except (TypeError, ValueError):
                frame_id = -1
            if frame_id >= 0:
                return ("frame", frame_id)

        timestamp = latest_perception.get("perception_wall_time")
        if timestamp is None:
            return None
        try:
            return ("time", float(timestamp))
        except (TypeError, ValueError):
            return None

    def _track_gates_from_snapshot(self, snapshot) -> list[np.ndarray]:
        track_gates = getattr(snapshot, "track_gates", None)
        if not track_gates:
            self._candidate_gate_track_ids = []
            return []

        gates = []
        track_ids = []
        for gate in sorted(track_gates, key=self._track_gate_sort_key):
            position = gate.get("position_neu")
            if position is None and gate.get("position_ned") is not None:
                ned = np.asarray(gate["position_ned"], dtype=float).reshape(3)
                position = (ned[0], ned[1], -ned[2])
            if position is None:
                continue

            arr = np.asarray(position, dtype=float)
            if arr.shape == (3,) and np.all(np.isfinite(arr)):
                gates.append(arr.copy())
                track_ids.append(self._int_or_default(gate.get("gate_id"), len(track_ids)))

        self._candidate_gate_track_ids = track_ids
        return gates

    def _maybe_preempt_active_target(
        self,
        gates: list[np.ndarray],
        track_ids: list,
    ) -> bool:
        new_track_id = self.pending_active_target_preempt_track_id
        self.pending_active_target_preempt_track_id = None
        details = self.pending_active_target_preempt_details
        self.pending_active_target_preempt_details = None
        if (
            new_track_id is None
            or not self.target_manager.locked
            or not gates
            or not track_ids
        ):
            return False

        gate_idx = max(0, int(self.current_gate_idx))
        if gate_idx >= len(gates) or gate_idx >= len(track_ids):
            return False
        try:
            installed_track_id = int(track_ids[gate_idx])
        except (TypeError, ValueError):
            return False
        if installed_track_id != int(new_track_id):
            return False

        old_track_id = self.active_target_track_id
        if old_track_id is None:
            old_track_id = getattr(self.target_manager, "active_target_track_id", None)
        if old_track_id is not None and int(old_track_id) == int(new_track_id):
            return False

        active_dist = details.get("active_dist") if isinstance(details, dict) else None
        candidate_dist = details.get("candidate_dist") if isinstance(details, dict) else None
        projection = details.get("projection") if isinstance(details, dict) else None
        lateral = details.get("lateral") if isinstance(details, dict) else None
        print(
            "active_target_preempt "
            f"gate_idx={gate_idx} "
            f"old_track={old_track_id if old_track_id is not None else 'none'} "
            f"new_track={int(new_track_id)} "
            f"active_dist={self._fmt_float(active_dist, precision=2)} "
            f"candidate_dist={self._fmt_float(candidate_dist, precision=2)} "
            f"projection={self._fmt_float(projection, precision=2)} "
            f"lateral={self._fmt_float(lateral, precision=2)} "
            f"new_center={self._fmt_vec(gates[gate_idx], precision=3)}",
            flush=True,
        )

        self.target_manager.clear_active(reason="preempt")
        self._sync_target_manager_state(clear_unlocked_current=True)
        self._reset_gate_pass_state()
        self.active_waypoints = None
        self.active_times = None
        self.active_waypoint_roles = []
        self.active_horizon_gate_indices = []
        self.active_horizon_track_ids = []
        self.active_horizon_targets = []
        self.active_plan_mode = ""
        self.active_terminal_velocity = np.zeros(3, dtype=float)
        self.active_terminal_velocity_policy = "cleared_preempt"
        self.last_planned_gate_idx = -1
        self.active_target_shift_frames = 0
        self.active_target_shift_pending_kind = None
        self.active_target_shift_track_id = None
        self._reset_deferred_longitudinal_shift()
        self._last_gate_signature = None
        return True

    def _install_gate_centers(self, gates: list[np.ndarray]) -> None:
        track_ids = list(self._candidate_gate_track_ids)
        if len(track_ids) != len(gates):
            track_ids = [None] * len(gates)
        gates, track_ids = self._apply_spline_memory_to_gate_entries(gates, track_ids)

        preempted = self._maybe_preempt_active_target(gates, track_ids)
        signature = (
            int(self.current_gate_idx),
            self._gate_signature(gates, track_ids),
        )
        if not preempted and signature == self._last_gate_signature:
            return

        gates, track_ids = self.target_manager.update_live_targets(
            gate_idx=self.current_gate_idx,
            gates=gates,
            track_ids=track_ids,
        )
        self._sync_target_manager_state()

        previous_target = (
            self.current_gate_pos.copy()
            if self.current_gate_pos is not None
            else None
        )
        had_active_plan = (
            self.active_waypoints is not None
            and self.planner.total_time > 0.0
        )

        self.gate_centers_neu = [
            np.asarray(gate, dtype=float).reshape(3).copy()
            for gate in gates
        ]
        self.gate_track_ids = list(track_ids)
        self.current_gate_idx = max(0, int(self.current_gate_idx))
        if self.race_gate_count is not None:
            self.current_gate_idx = min(self.current_gate_idx, self.race_gate_count)
        self.active_track_count = max(
            0,
            len(self.gate_centers_neu) - self.current_gate_idx,
        )
        next_target = (
            self.gate_centers_neu[self.current_gate_idx].copy()
            if 0 <= self.current_gate_idx < len(self.gate_centers_neu)
            else None
        )
        self._last_gate_signature = signature
        provisional_handoff = bool(
            self.provisional_target_active and next_target is not None
        )

        if next_target is None:
            memory_target, memory_track_id, _ = self._spline_memory_for_gate(
                int(self.current_gate_idx)
            )
            if memory_target is not None:
                self.current_gate_pos = memory_target.copy()
                self.active_target_track_id = memory_track_id
                self.last_active_target_center = memory_target.copy()
                self.active_target_lost_time = None
                self._trace_spline_memory(
                    action="hold",
                    gate_idx=int(self.current_gate_idx),
                    track_id=memory_track_id,
                    center=memory_target,
                    context="no_live_gate_slot",
                )
                return
            if self.provisional_target_active:
                print(
                    "provisional_next_gate_continue "
                    f"gate_idx={int(self.current_gate_idx)} "
                    f"track={self.provisional_target_track_id if self.provisional_target_track_id is not None else 'none'} "
                    f"target_neu={self._fmt_vec(self.provisional_target_center, precision=3)}",
                    flush=True,
                )
                return
            self.current_gate_pos = None
            self.active_target_track_id = None
            self.last_active_target_center = None
            self.active_target_lost_time = None
            self._reset_gate_pass_state()
            if self._post_gate_exit_active():
                print(
                    "post_gate_exit_continue "
                    f"current_gate_idx={int(self.current_gate_idx)} "
                    f"until_s={self.post_gate_exit_until_s:.3f} "
                    f"reason={self.post_gate_exit_reason or 'active'}",
                    flush=True,
                )
                return
            self.active_waypoints = None
            self.active_times = None
            self.active_waypoint_roles = []
            self.active_horizon_gate_indices = []
            self.active_horizon_track_ids = []
            self.active_horizon_targets = []
            self.active_plan_mode = ""
            self.active_terminal_velocity = np.zeros(3, dtype=float)
            self.active_terminal_velocity_policy = "cleared_no_target"
            self.last_planned_gate_idx = -1
            return

        if self._post_gate_exit_active() and had_active_plan and not provisional_handoff:
            self.current_gate_pos = next_target.copy()
            self.active_target_track_id = (
                self.gate_track_ids[self.current_gate_idx]
                if self.current_gate_idx < len(self.gate_track_ids)
                else None
            )
            if self.active_target_track_id is not None:
                self.last_active_target_center = next_target.copy()
            print(
                "post_gate_exit_continue "
                f"current_gate_idx={int(self.current_gate_idx)} "
                f"track={self.active_target_track_id if self.active_target_track_id is not None else 'none'} "
                f"target_neu={self._fmt_vec(next_target, precision=3)} "
                f"until_s={self.post_gate_exit_until_s:.3f} "
                f"reason={self.post_gate_exit_reason or 'active'}",
                flush=True,
            )
            return

        if provisional_handoff:
            self._clear_provisional_target(reason="race_order_handoff")
            print(
                "provisional_next_gate_handoff "
                f"gate_idx={int(self.current_gate_idx)} "
                f"track={track_ids[self.current_gate_idx] if self.current_gate_idx < len(track_ids) else 'none'} "
                f"target_neu={self._fmt_vec(next_target, precision=3)}",
                flush=True,
            )

        target_shift = (
            float(np.linalg.norm(next_target - previous_target))
            if previous_target is not None
            else math.inf
        )
        active_locked_same_gate = (
            self.target_manager.locked
            and had_active_plan
            and self.last_planned_gate_idx == self.current_gate_idx
        )
        should_replan = (
            provisional_handoff
            or not had_active_plan
            or self.last_planned_gate_idx != self.current_gate_idx
            or (
                not active_locked_same_gate
                and target_shift > self.replan_target_shift_m
            )
        )

        if should_replan:
            self.current_gate_pos = next_target.copy()
            self.active_target_track_id = (
                self.gate_track_ids[self.current_gate_idx]
                if self.current_gate_idx < len(self.gate_track_ids)
                else None
            )
            if self.active_target_track_id is not None:
                self.last_active_target_center = next_target.copy()
            self._reset_gate_pass_state()
            self.active_waypoints = None
            self.active_times = None
            self.active_waypoint_roles = []
            self.active_horizon_gate_indices = []
            self.active_horizon_track_ids = []
            self.active_horizon_targets = []
            self.active_plan_mode = ""
            self.active_terminal_velocity = np.zeros(3, dtype=float)
            self.active_terminal_velocity_policy = "cleared_replan_needed"
            self.last_planned_gate_idx = -1

    @classmethod
    def _gate_signature(cls, gates: list[np.ndarray], track_ids=None) -> tuple:
        if track_ids is None or len(track_ids) != len(gates):
            track_ids = [None] * len(gates)
        return tuple(
            (track_id, *cls._rounded_gate(gate, decimals=1))
            for track_id, gate in zip(track_ids, gates)
        )

    @staticmethod
    def _rounded_gate(gate, decimals: int) -> tuple[float, float, float]:
        arr = np.asarray(gate, dtype=float).reshape(3)
        return tuple(round(float(value), int(decimals)) for value in arr)

    @staticmethod
    def _finite_float(value, default: float, allow_nan: bool = False) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float(default)
        if math.isfinite(out) or (allow_nan and math.isnan(out)):
            return out
        return float(default)

    @classmethod
    def _detection_sort_key(cls, detection) -> int:
        if not isinstance(detection, dict):
            return 0
        return cls._int_or_default(detection.get("detection_id"), 0)

    @classmethod
    def _track_gate_sort_key(cls, gate) -> int:
        if not isinstance(gate, dict):
            return 0
        return cls._int_or_default(gate.get("gate_id"), 0)

    @staticmethod
    def _int_or_default(value, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _reference_motion_yaw(v_ref, a_ref, last_yaw, eps=1e-3) -> float:
        v_xy = np.asarray(v_ref[:2], dtype=float)
        a_xy = np.asarray(a_ref[:2], dtype=float)

        if np.linalg.norm(v_xy) > eps:
            return math.atan2(float(v_xy[1]), float(v_xy[0]))
        if np.linalg.norm(a_xy) > eps:
            return math.atan2(float(a_xy[1]), float(a_xy[0]))
        return float(last_yaw)

    @staticmethod
    def _wrap_pi(angle: float) -> float:
        return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
