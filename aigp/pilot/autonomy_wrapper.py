from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

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
from hover_acquisition import HoverAcquisition
from runtime_config import load_runtime_config
from vehicle_state_estimator import VehicleStateEstimator


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
        self._last_gate_signature = None
        self.active_track_count = 0

        self.current_gate_idx = 0
        self.current_gate_pos = None
        self.active_target_track_id = None
        self.last_active_target_center = None
        self.active_target_lost_time = None
        self.active_target_lost_grace_s = float(self.config.race.active_target_lost_grace_s)
        self.race_order_track_ids = []
        self.completed_track_ids = set()
        self.active_waypoints = None
        self.active_times = None
        self.last_planned_gate_idx = -1
        self.trajectory_start_time = 0.0
        self.last_desired_yaw = 0.0
        self.replan_target_shift_m = float(self.config.planner.replan_target_shift_m)
        self.replan_after_trajectory_s = float(self.config.planner.replan_after_trajectory_s)
        self.replan_min_interval_s = float(self.config.planner.replan_min_interval_s)
        self.last_plan_wall_time = 0.0
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
            max_outlier_distance=gate_memory_config.max_outlier_distance,
            min_observation_time=gate_memory_config.min_observation_time,
        )
        self.gate_memory.max_committed_match_distance = (
            gate_memory_config.max_committed_match_distance
        )
        self.state_estimator = VehicleStateEstimator(self.config)
        self.last_state_estimate = None
        self._last_gate_memory_frame_key = None
        self._last_stable_gate_print_signature = None
        self._last_race_order_print_signature = None
        self._last_trace_print_time = 0.0
        self._trace_period_s = 0.5

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

    def update(self, snapshot) -> AutonomyCommandRad | None:
        snapshot.stable_gate_landmarks_neu = self._stable_gate_landmarks_neu()
        estimate = self.state_estimator.update(snapshot)
        self.last_state_estimate = estimate

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

        advanced = self._advance_gate_if_needed(pos)
        if self._should_plan(advanced):
            self._path_plan(pos, vel)

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
            f"vz={debug.vz_m_s:.2f} "
            f"az={debug.az_m_s2:.2f} "
            f"thrust={debug.thrust:.3f} "
            f"hover={debug.hover_thrust:.3f} "
            f"lift={int(debug.lift_confirmed)} "
            f"stable={debug.stable_time_s:.2f} "
            f"pos_neu=({arr[0]:.2f},{arr[1]:.2f},{arr[2]:.2f})",
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

        state_source = "unknown"
        truth_error_txt = "nan"
        correction_txt = "none"
        if self.last_state_estimate is not None:
            state_source = str(self.last_state_estimate.source)
            truth_error = self.last_state_estimate.truth_error_m
            if truth_error is not None and math.isfinite(float(truth_error)):
                truth_error_txt = f"{float(truth_error):.2f}"
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

        print(
            "autonomy_trace "
            f"gate_idx={self.current_gate_idx} "
            f"active_track={self.active_target_track_id} "
            f"tracks={self.active_track_count} "
            f"state={state_source} "
            f"truth_err={truth_error_txt} "
            f"corr={correction_txt} "
            f"tau={tau:.2f}/{float(self.planner.total_time):.2f} "
            f"dist={dist_txt} "
            f"pos_neu={fmt_vec(pos)} "
            f"target_neu={target_txt} "
            f"p_ref={fmt_vec(p_ref)} "
            f"v_ref={fmt_vec(v_ref)} "
            f"a_ref={fmt_vec(a_ref)} "
            f"cmd_deg=({math.degrees(roll_rad):.2f},{math.degrees(pitch_rad):.2f},{math.degrees(yaw_rad):.2f}) "
            f"thrust={thrust:.3f} "
            f"hover={hover_txt} "
            f"hover_adapt={hover_status} "
            f"hover_sig={hover_signal_txt} "
            f"hover_err_z_vz={hover_err_txt} "
            f"hover_gain={hover_gain_txt} "
            f"hover_fast={hover_fast_txt}",
            flush=True,
        )

    def _path_plan(self, pos: np.ndarray, vel: np.ndarray) -> bool:
        if not self.gate_centers_neu:
            return False

        target_idx = int(self.current_gate_idx)
        if target_idx < 0 or target_idx >= len(self.gate_centers_neu):
            return False

        target = self._apply_target_z_policy(self.gate_centers_neu[target_idx])

        self.active_target_track_id = (
            self.gate_track_ids[target_idx]
            if target_idx < len(self.gate_track_ids)
            else None
        )
        if self.active_target_track_id is not None:
            self.last_active_target_center = target.copy()
            self.active_target_lost_time = None

        waypoints = np.vstack([pos, target])
        times = allocate_segment_times(
            waypoints,
            current_vel=vel,
            vmax=self.planner_vmax,
            amax=self.planner_amax,
            T_min=self.planner_t_min,
        )

        self.planner = MultiSegmentMinimumSnapPlanner()
        self.planner.update(
            waypoints=waypoints,
            times=times,
            v_start=vel,
            v_end=np.zeros(3, dtype=float),
            a_start=np.zeros(3, dtype=float),
            a_end=np.zeros(3, dtype=float),
            j_start=np.zeros(3, dtype=float),
            j_end=np.zeros(3, dtype=float),
        )

        self.current_gate_pos = target.copy()
        self.active_waypoints = waypoints.copy()
        self.active_times = np.asarray(times, dtype=float).copy()
        self.trajectory_start_time = time.time()
        self.last_plan_wall_time = self.trajectory_start_time
        self.last_planned_gate_idx = int(self.current_gate_idx)
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
        distance = float(np.linalg.norm(pos - target))
        if distance >= self.pass_radius_m:
            return False

        if self.active_target_track_id is not None:
            self.completed_track_ids.add(int(self.active_target_track_id))
        self.current_gate_idx += 1
        self.active_target_track_id = None
        self.last_active_target_center = None
        self.active_target_lost_time = None
        self.active_waypoints = None
        self.active_times = None
        return True

    def _should_plan(self, advanced: bool) -> bool:
        if advanced:
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
        target = self.current_gate_pos
        if target is not None:
            to_target = np.asarray(target[:2], dtype=float) - pos[:2]
            if np.linalg.norm(to_target) > 1e-3:
                desired = math.atan2(float(to_target[1]), float(to_target[0]))
                self.last_desired_yaw = self._wrap_pi(desired)
                return self.last_desired_yaw

        desired = self._reference_motion_yaw(v_ref, a_ref, self.last_desired_yaw)
        if not np.isfinite(desired):
            desired = current_yaw
        self.last_desired_yaw = self._wrap_pi(desired)
        return self.last_desired_yaw

    def _gates_from_snapshot(self, snapshot) -> list[np.ndarray]:
        if self.use_perception:
            perception_gates = self._perception_gates_from_snapshot(snapshot)
            if perception_gates:
                return perception_gates

        return self._track_gates_from_snapshot(snapshot)

    def _stable_gate_landmarks_neu(self) -> list[np.ndarray]:
        return [
            np.asarray(track.center, dtype=float).reshape(3).copy()
            for track in self.gate_memory.get_stable_tracks()
        ]

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

    def _refresh_perception_race_order(
        self,
        stable_tracks,
        committed_by_id: dict[int, object],
        current_pos: np.ndarray,
    ) -> None:
        current_pos = np.asarray(current_pos, dtype=float).reshape(3)

        accepted_ids = []
        for track_id in self.race_order_track_ids:
            track_id = int(track_id)
            if track_id in committed_by_id and track_id not in accepted_ids:
                accepted_ids.append(track_id)

        for track in stable_tracks:
            track_id = int(track.id)
            if track_id in self.completed_track_ids:
                continue
            if track_id in accepted_ids:
                continue
            if (
                self.race_gate_count is not None
                and len(accepted_ids) >= self.race_gate_count
            ):
                break
            accepted_ids.append(track_id)

        prefix_len = min(max(int(self.current_gate_idx), 0), len(accepted_ids))
        completed_prefix = accepted_ids[:prefix_len]
        candidate_ids = [
            track_id
            for track_id in accepted_ids[prefix_len:]
            if track_id not in self.completed_track_ids
        ]

        active_id = self.active_target_track_id
        if active_id is not None:
            active_id = int(active_id)

        if active_id is not None and active_id in committed_by_id:
            if active_id not in candidate_ids and active_id not in completed_prefix:
                candidate_ids.insert(0, active_id)
        elif active_id is not None:
            active_id = None

        ordered_suffix = self._order_track_ids_by_progress(
            candidate_ids=candidate_ids,
            current_pos=current_pos,
            committed_by_id=committed_by_id,
            active_id=active_id,
        )

        order = completed_prefix + ordered_suffix
        if self.race_gate_count is not None:
            order = order[: self.race_gate_count]
        self.race_order_track_ids = order

        if (
            self.active_target_track_id is None
            and 0 <= self.current_gate_idx < len(self.race_order_track_ids)
        ):
            self.active_target_track_id = self.race_order_track_ids[self.current_gate_idx]

    def _order_track_ids_by_progress(
        self,
        candidate_ids: list[int],
        current_pos: np.ndarray,
        committed_by_id: dict[int, object],
        active_id,
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
            return np.asarray(committed_by_id[int(track_id)].center, dtype=float).reshape(3)

        if active_id is not None and int(active_id) in unique_ids:
            first_id = int(active_id)
        else:
            first_id = min(
                unique_ids,
                key=lambda track_id: (
                    float(np.linalg.norm(center_for(track_id) - current_pos)),
                    int(track_id),
                ),
            )

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
        now = time.time()

        for track_id in self.race_order_track_ids:
            track_id = int(track_id)
            center = None
            track = committed_by_id.get(track_id)
            if track is not None:
                center = np.asarray(track.center, dtype=float).reshape(3).copy()
                if track_id == self.active_target_track_id:
                    self.last_active_target_center = center.copy()
                    self.active_target_lost_time = None
            elif (
                track_id == self.active_target_track_id
                and self.last_active_target_center is not None
            ):
                if self.active_target_lost_time is None:
                    self.active_target_lost_time = now
                if now - self.active_target_lost_time <= self.active_target_lost_grace_s:
                    center = self.last_active_target_center.copy()

            if center is None or not np.all(np.isfinite(center)):
                continue

            gates.append(center)
            track_ids.append(track_id)

        return gates, track_ids

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
                math.isfinite(reprojection_error)
                and reprojection_error > self.max_reprojection_error_for_memory
            ):
                continue

            center_camera = detection.get("gate_center_camera")
            if center_camera is not None:
                center_camera = np.asarray(center_camera, dtype=float)
                if center_camera.shape != (3,) or not np.all(np.isfinite(center_camera)):
                    center_camera = None
            if center_camera is not None:
                if self.reject_negative_depth and float(center_camera[2]) <= 0.0:
                    continue
                if (
                    self.max_detection_range_m > 0.0
                    and float(np.linalg.norm(center_camera)) > self.max_detection_range_m
                ):
                    continue

            self.gate_memory.add_detection(
                center=arr.copy(),
                confidence=confidence,
                timestamp=timestamp,
                center_camera=center_camera,
                reprojection_error=reprojection_error,
                solver_name="latest_perception",
            )

        self.gate_memory.prune(timestamp)

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

    def _install_gate_centers(self, gates: list[np.ndarray]) -> None:
        track_ids = list(self._candidate_gate_track_ids)
        if len(track_ids) != len(gates):
            track_ids = [None] * len(gates)

        signature = (
            int(self.current_gate_idx),
            self._gate_signature(gates, track_ids),
        )
        if signature == self._last_gate_signature:
            return

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

        if next_target is None:
            self.current_gate_pos = None
            self.active_target_track_id = None
            self.last_active_target_center = None
            self.active_target_lost_time = None
            self.active_waypoints = None
            self.active_times = None
            self.last_planned_gate_idx = -1
            return

        target_shift = (
            float(np.linalg.norm(next_target - previous_target))
            if previous_target is not None
            else math.inf
        )
        should_replan = (
            not had_active_plan
            or self.last_planned_gate_idx != self.current_gate_idx
            or target_shift > self.replan_target_shift_m
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
            self.active_waypoints = None
            self.active_times = None
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
