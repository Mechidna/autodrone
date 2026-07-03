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
from gate_pass_geometry import check_gate_plane_pass, unit_vector_from_to
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
        self.active_gate_normal = None
        self.previous_gate_pass_position = None
        self.gate_plane_crossed = False
        self.near_gate_but_not_crossed = False
        self.gate_progress_along_approach = float("nan")
        self.gate_lateral_error = float("nan")
        self.last_planned_gate_idx = -1
        self.trajectory_start_time = 0.0
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
        self.passthrough_velocity_enabled = bool(
            self.config.planner.passthrough_velocity_enabled
        )
        self.passthrough_speed_m_s = max(
            0.0,
            float(self.config.planner.passthrough_speed_m_s),
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
        self.active_target_shift_frames = 0
        self.active_target_shift_track_id = None
        self.planner_vmax = float(self.config.planner.vmax)
        self.planner_amax = float(self.config.planner.amax)
        self.planner_t_min = float(self.config.planner.t_min)
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
        self._last_target_reject_signature = None
        self._last_plan_validation_reject_signature = None
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
        advanced = self._advance_gate_if_needed(pos)
        if not shift_replanned and self._should_plan(advanced, pos, vel):
            planned = self._path_plan(pos, vel)
            if not planned:
                planned = self._path_plan_provisional_next_gate(pos, vel)
            if not planned and self._active_plan_expired():
                self._clear_active_plan(reason="replan_failed_expired")

        if self.active_waypoints is None or self.planner.total_time <= 0.0:
            return None

        tau = max(0.0, time.time() - self.trajectory_start_time)
        tau = min(tau, float(self.planner.total_time))
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
        roll_rad, pitch_rad, yaw_cmd_rad, thrust, _ = self.tracker.update(state, ref)
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
        self.active_horizon_gate_indices = []
        self.active_horizon_track_ids = []
        self.active_horizon_targets = []
        self.active_plan_mode = ""
        self.active_terminal_velocity = np.zeros(3, dtype=float)
        self.active_terminal_velocity_policy = f"cleared:{reason}"
        self.post_gate_exit_until_s = 0.0
        self.post_gate_exit_reason = ""

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
            self.active_target_shift_track_id = None
            return False

        try:
            active_id = int(diag.active_track_id)
        except (TypeError, ValueError):
            self.active_target_shift_frames = 0
            self.active_target_shift_track_id = None
            return False
        if active_id < 0:
            self.active_target_shift_frames = 0
            self.active_target_shift_track_id = None
            return False

        if self.active_target_shift_track_id != active_id:
            self.active_target_shift_frames = 0
            self.active_target_shift_track_id = active_id

        latest, source_track_id, quality = self._best_duplicate_cluster_center(active_id)
        if latest is None or not bool(quality.get("ok", False)):
            self.active_target_shift_frames = 0
            return False

        planned = self._finite_vec3_or_none(diag.center_at_plan)
        if planned is None:
            self.active_target_shift_frames = 0
            return False

        planned = self._apply_target_z_policy(planned)
        latest = self._apply_target_z_policy(latest)
        shift_vec = latest - planned
        shift_m = float(np.linalg.norm(shift_vec))
        shift_xy_m = float(np.linalg.norm(shift_vec[:2]))
        shift_z_m = float(abs(shift_vec[2]))
        if not math.isfinite(shift_m) or shift_m < self.active_target_shift_threshold_m:
            self.active_target_shift_frames = 0
            return False
        max_total_m = float(self.active_target_shift_max_total_m)
        if max_total_m > 0.0 and shift_m > max_total_m:
            self.active_target_shift_frames = 0
            return False

        dist_to_target = float(np.linalg.norm(np.asarray(pos, dtype=float).reshape(3) - planned))
        if dist_to_target <= self.active_target_shift_near_gate_distance_m:
            self.active_target_shift_frames = 0
            return False

        self.active_target_shift_frames += 1
        if self.active_target_shift_frames < self.active_target_shift_required_frames:
            return False

        now = time.time()
        if now - self.last_plan_wall_time < self.active_target_shift_replan_min_interval_s:
            return False

        if self.active_target_shift_alpha <= 0.0:
            self.active_target_shift_frames = 0
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
            return False

        print(
            "active_target_shift correction "
            f"gate_idx={target_idx} "
            f"track={active_id} "
            f"shift={shift_m:.2f} "
            f"shift_xy={shift_xy_m:.2f} "
            f"shift_z={shift_z_m:.2f} "
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
        self.active_horizon_gate_indices = []
        self.active_horizon_track_ids = []
        self.active_horizon_targets = []
        self.active_plan_mode = ""
        self.active_terminal_velocity = np.zeros(3, dtype=float)
        self.active_terminal_velocity_policy = "cleared_active_shift"
        self.last_planned_gate_idx = -1
        self._last_gate_signature = None
        self.active_target_shift_frames = 0
        return self._path_plan(pos, vel)

    def _planning_horizon_targets(
        self,
        target_idx: int,
        target: np.ndarray,
        target_track_id,
    ) -> tuple[list[np.ndarray], list, list[int]]:
        targets = [np.asarray(target, dtype=float).reshape(3).copy()]
        track_ids = [target_track_id]
        gate_indices = [int(target_idx)]
        max_idx = min(
            len(self.gate_centers_neu),
            int(target_idx) + self.planning_horizon_gates,
        )
        for idx in range(int(target_idx) + 1, max_idx):
            center = self._apply_target_z_policy(self.gate_centers_neu[idx])
            if not np.all(np.isfinite(center)):
                continue
            track_id = self.gate_track_ids[idx] if idx < len(self.gate_track_ids) else None
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
        return targets, track_ids, gate_indices

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
            direction = waypoints[idx + 1] - waypoints[idx - 1]
            norm = float(np.linalg.norm(direction))
            if not math.isfinite(norm) or norm < 1e-6:
                continue
            velocities[idx] = self.passthrough_speed_m_s * (direction / norm)
        if not np.any(np.isfinite(velocities[1:-1])):
            return None
        return velocities

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

    def _build_minimum_snap_plan(
        self,
        *,
        waypoints: np.ndarray,
        times,
        v_start: np.ndarray,
        v_end: np.ndarray,
        waypoint_velocities,
    ) -> MultiSegmentMinimumSnapPlanner:
        planner = MultiSegmentMinimumSnapPlanner()
        planner.update(
            waypoints=waypoints,
            times=times,
            v_start=v_start,
            v_end=v_end,
            a_start=np.zeros(3, dtype=float),
            a_end=np.zeros(3, dtype=float),
            j_start=np.zeros(3, dtype=float),
            j_end=np.zeros(3, dtype=float),
            waypoint_velocities=waypoint_velocities,
        )
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
            f"fallback={fallback}",
            flush=True,
        )

    def _path_plan(self, pos: np.ndarray, vel: np.ndarray) -> bool:
        if not self.gate_centers_neu:
            return False

        pos = np.asarray(pos, dtype=float).reshape(3)
        vel = np.asarray(vel, dtype=float).reshape(3)
        target_idx = int(self.current_gate_idx)
        if target_idx < 0 or target_idx >= len(self.gate_centers_neu):
            return False

        target = self._apply_target_z_policy(self.gate_centers_neu[target_idx])
        target_track_id = (
            self.gate_track_ids[target_idx]
            if target_idx < len(self.gate_track_ids)
            else None
        )
        target = self.target_manager.lock_target(
            gate_idx=target_idx,
            track_id=target_track_id,
            center_neu=target,
            reason="path_plan",
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
            )
        )

        def build_candidate(targets, track_ids, gate_indices, *, allow_exit: bool = True):
            if len(targets) >= 2:
                candidate_waypoints = np.vstack([pos, *targets])
                candidate_waypoint_velocities = (
                    self._compute_passthrough_waypoint_velocities(candidate_waypoints)
                )
                candidate_mode = "gate_horizon"
            elif normal is None or not allow_exit:
                candidate_waypoints = np.vstack([pos, target])
                candidate_waypoint_velocities = None
                candidate_mode = "single_gate"
            else:
                pass_through_target = target + normal * self.gate_pass_through_m
                candidate_waypoints = np.vstack([pos, target, pass_through_target])
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
            if not fallback_valid and fallback_candidate["mode"] == "single_gate_exit":
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
        elif not valid_plan and candidate["mode"] == "single_gate_exit":
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
        horizon_targets = candidate["horizon_targets"]
        horizon_track_ids = candidate["horizon_track_ids"]
        horizon_gate_indices = candidate["horizon_gate_indices"]

        self.current_gate_pos = target.copy()
        self.active_waypoints = waypoints.copy()
        self.active_times = np.asarray(times, dtype=float).copy()
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
        waypoints_txt = "[" + ";".join(
            self._fmt_vec(waypoint, precision=3) for waypoint in waypoints
        ) + "]"
        times_txt = "(" + ",".join(f"{float(item):.3f}" for item in times) + ")"
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
            f"v_start_neu={self._fmt_vec(vel, precision=3)} "
            f"v_end_neu={self._fmt_vec(terminal_velocity, precision=3)} "
            f"terminal_policy={terminal_policy} "
            f"waypoint_velocities_neu={waypoint_velocities_txt} "
            f"times_s={times_txt} "
            f"waypoints_neu={waypoints_txt} "
            f"plan_samples_neu={plan_samples_txt}",
            flush=True,
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
            if allow_exit:
                pass_through_target = target + normal * self.gate_pass_through_m
                candidate_waypoints = np.vstack([pos, target, pass_through_target])
            else:
                candidate_waypoints = np.vstack([pos, target])
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
            candidate_planner = MultiSegmentMinimumSnapPlanner()
            candidate_planner.update(
                waypoints=candidate_waypoints,
                times=candidate_times,
                v_start=vel,
                v_end=candidate_terminal_velocity,
                a_start=np.zeros(3, dtype=float),
                a_end=np.zeros(3, dtype=float),
                j_start=np.zeros(3, dtype=float),
                j_end=np.zeros(3, dtype=float),
                waypoint_velocities=None,
            )
            return {
                "planner": candidate_planner,
                "waypoints": candidate_waypoints,
                "times": candidate_times,
                "terminal_velocity": candidate_terminal_velocity,
                "terminal_policy": candidate_terminal_policy,
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
            self._trace_plan_validation_reject(
                validation_details,
                fallback="direct_target",
            )
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
        self.active_horizon_gate_indices = [int(self.current_gate_idx)]
        self.active_horizon_track_ids = [track_id]
        self.active_horizon_targets = [target.copy()]
        self.active_plan_mode = "provisional_next_gate"
        self.active_terminal_velocity = terminal_velocity.copy()
        self.active_terminal_velocity_policy = str(terminal_policy)
        self.trajectory_start_time = now
        self.last_plan_wall_time = now
        self.last_planned_gate_idx = int(self.current_gate_idx)
        self.post_gate_exit_until_s = 0.0
        self.post_gate_exit_reason = ""
        self._reset_gate_pass_state()

        waypoints_txt = "[" + ";".join(
            self._fmt_vec(waypoint, precision=3) for waypoint in waypoints
        ) + "]"
        times_txt = "(" + ",".join(f"{float(item):.3f}" for item in times) + ")"
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
            f"v_start_neu={self._fmt_vec(vel, precision=3)} "
            f"v_end_neu={self._fmt_vec(terminal_velocity, precision=3)} "
            f"terminal_policy={terminal_policy} "
            f"times_s={times_txt} "
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

    def _advance_gate_if_needed(self, pos: np.ndarray) -> bool:
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
        target = np.asarray(target, dtype=float).reshape(3)
        distance = float(np.linalg.norm(pos - target))
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

        pass_result = check_gate_plane_pass(
            previous_position=previous_pos,
            position=pos,
            center=target,
            normal=normal,
            lateral_radius_m=self.gate_pass_lateral_radius_m,
            plane_tolerance_m=self.gate_plane_tolerance_m,
        )
        self.previous_gate_pass_position = pos.copy()
        self.gate_plane_crossed = bool(self.gate_plane_crossed or pass_result.crossed_plane)
        self.near_gate_but_not_crossed = bool(
            not pass_result.passed and distance <= self.pass_radius_m
        )
        self.gate_progress_along_approach = float(pass_result.signed_progress_m)
        self.gate_lateral_error = float(pass_result.lateral_error_m)
        if not pass_result.passed:
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
            f"reason={pass_result.reason} "
            f"distance={distance:.3f} "
            f"plane_progress={float(pass_result.signed_progress_m):.3f} "
            f"lateral_error={float(pass_result.lateral_error_m):.3f} "
            f"pos_neu={self._fmt_vec(pos, precision=3)} "
            f"target_neu={self._fmt_vec(target, precision=3)} "
            f"truth_err={self._fmt_float(truth_error, precision=3)}",
            flush=True,
        )
        self.target_manager.mark_passed(
            pos_neu=pos,
            distance_m=distance,
            pass_reason=pass_result.reason,
            plane_progress_m=pass_result.signed_progress_m,
            lateral_error_m=pass_result.lateral_error_m,
            truth_pos_neu=truth_pos,
            truth_error_m=truth_error,
        )
        self._record_completed_landmark(completed_track_id, target)
        continued = self._continue_plan_after_gate_pass(
            completed_gate_idx=completed_gate_idx,
            next_gate_idx=next_gate_idx,
            pos=pos,
        )
        if not continued:
            self._sync_target_manager_state(clear_unlocked_current=True)
            self._reset_gate_pass_state()
            self.active_waypoints = None
            self.active_times = None
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
                valid, reject_reason, target, source_track_id = (
                    self._validated_horizon_continue_target(
                        track_id=track_id,
                        stored_target=target,
                        next_gate_idx=next_gate_idx,
                    )
                )
                if not valid or target is None:
                    self._trace_target_rejection(
                        reason=reject_reason,
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
                    reason="horizon_continue",
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
                    f"active_plan_mode={self.active_plan_mode} "
                    f"terminal_policy={self.active_terminal_velocity_policy}",
                    flush=True,
                )
                return True

        if (
            self.post_gate_exit_continuation_enabled
            and future_race_gate
            and self.active_plan_mode == "single_gate_exit"
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
        if self._prefer_reference_motion_yaw_near_gate(pos):
            desired = self._reference_motion_yaw(v_ref, a_ref, self.last_desired_yaw)
            if np.isfinite(desired):
                self.last_yaw_target = None
                self.last_yaw_target_source = "reference_motion_near_horizon_gate"
                self.last_desired_yaw = self._wrap_pi(desired)
                return self.last_desired_yaw

        target, source = self._yaw_target_center()
        self.last_yaw_target = None if target is None else target.copy()
        self.last_yaw_target_source = source
        if target is not None:
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

    def _prefer_reference_motion_yaw_near_gate(self, pos: np.ndarray) -> bool:
        if not self.yaw_reference_motion_near_gate_enabled:
            return False
        if self.yaw_reference_motion_distance_m <= 0.0:
            return False
        if self.active_waypoints is None or self.planner.total_time <= 0.0:
            return False
        if self._post_gate_exit_active():
            return True
        if len(self.active_horizon_gate_indices) < 2:
            return False
        if self.current_gate_pos is None:
            return False
        try:
            pos = np.asarray(pos, dtype=float).reshape(3)
            target = np.asarray(self.current_gate_pos, dtype=float).reshape(3)
        except (TypeError, ValueError):
            return False
        distance = float(np.linalg.norm(pos - target))
        return (
            math.isfinite(distance)
            and distance <= float(self.yaw_reference_motion_distance_m)
        )

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
            self._trace_provisional_rejections(rejects)
            return None, None, {}

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
        self.active_horizon_gate_indices = []
        self.active_horizon_track_ids = []
        self.active_horizon_targets = []
        self.active_plan_mode = ""
        self.active_terminal_velocity = np.zeros(3, dtype=float)
        self.active_terminal_velocity_policy = "cleared_preempt"
        self.last_planned_gate_idx = -1
        self.active_target_shift_frames = 0
        self.active_target_shift_track_id = None
        self._last_gate_signature = None
        return True

    def _install_gate_centers(self, gates: list[np.ndarray]) -> None:
        track_ids = list(self._candidate_gate_track_ids)
        if len(track_ids) != len(gates):
            track_ids = [None] * len(gates)

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
            self.active_horizon_gate_indices = []
            self.active_horizon_track_ids = []
            self.active_horizon_targets = []
            self.active_plan_mode = ""
            self.active_terminal_velocity = np.zeros(3, dtype=float)
            self.active_terminal_velocity_policy = "cleared_no_target"
            self.last_planned_gate_idx = -1
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
