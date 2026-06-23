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


DEFAULT_CAMERA_MATRIX = np.array(
    [
        [320.0, 0.0, 320.0],
        [0.0, 320.0, 180.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)
DEFAULT_DIST_COEFFS = np.zeros(5, dtype=float)


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
        use_perception: bool = False,
        race_gate_count: int = 3,
        pass_radius_m: float = 1.25,
    ):
        self.use_perception = bool(use_perception)
        self.pass_radius_m = float(pass_radius_m)
        self.camera_matrix = DEFAULT_CAMERA_MATRIX.copy()
        self.dist_coeffs = DEFAULT_DIST_COEFFS.copy()

        self.gate_centers_neu = []
        self._last_gate_signature = None
        self.active_track_count = 0

        self.current_gate_idx = 0
        self.current_gate_pos = None
        self.active_waypoints = None
        self.active_times = None
        self.last_planned_gate_idx = -1
        self.trajectory_start_time = 0.0
        self.last_desired_yaw = 0.0
        self.replan_target_shift_m = 1.0
        self.replan_after_trajectory_s = 0.25
        self.gate_memory = GateMemory(
            association_radius=1.5,
            commit_radius=1.5,
            new_track_block_radius=4.5,
            min_confidence_per_hit=0.2,
            commit_hits=4,
            commit_confidence_sum=1.2,
            stale_time=3.0,
            alpha=0.35,
            min_hits_for_stable=6,
            max_center_std_for_stable=0.45,
            max_camera_std_for_stable=0.45,
            max_reprojection_error_for_stable=5.0,
        )
        self._last_gate_memory_frame_key = None
        self._last_stable_gate_print_signature = None
        self._last_trace_print_time = 0.0
        self._trace_period_s = 0.5

        self.planner = MultiSegmentMinimumSnapPlanner()
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

    def update(self, snapshot) -> AutonomyCommandRad | None:
        yaw_rad = float(getattr(snapshot, "yaw_rad", 0.0))

        if snapshot.pos_neu is None or snapshot.vel_neu is None:
            return None

        pos = np.asarray(snapshot.pos_neu, dtype=float).reshape(3)
        vel = np.nan_to_num(
            np.asarray(snapshot.vel_neu, dtype=float).reshape(3),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if not np.all(np.isfinite(pos)):
            return None

        self._install_gate_centers(self._gates_from_snapshot(snapshot))

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

        roll_rad, pitch_rad, yaw_cmd_rad, thrust, _ = self.tracker.update(state, ref)

        # Preserve autonomy_api6's PX4/Gazebo sign convention.
        roll_rad = -float(roll_rad)
        pitch_rad = -float(pitch_rad)
        yaw_cmd_rad = float(yaw_cmd_rad)
        thrust = float(np.clip(thrust, 0.0, 1.0))

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
        )

        return AutonomyCommandRad(
            roll_rad=roll_rad,
            pitch_rad=pitch_rad,
            yaw_rad=yaw_cmd_rad,
            thrust=thrust,
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

        print(
            "autonomy_trace "
            f"gate_idx={self.current_gate_idx} "
            f"tracks={self.active_track_count} "
            f"tau={tau:.2f}/{float(self.planner.total_time):.2f} "
            f"dist={dist_txt} "
            f"pos_neu={fmt_vec(pos)} "
            f"target_neu={target_txt} "
            f"p_ref={fmt_vec(p_ref)} "
            f"v_ref={fmt_vec(v_ref)} "
            f"a_ref={fmt_vec(a_ref)} "
            f"cmd_deg=({math.degrees(roll_rad):.2f},{math.degrees(pitch_rad):.2f},{math.degrees(yaw_rad):.2f}) "
            f"thrust={thrust:.3f}",
            flush=True,
        )

    def _path_plan(self, pos: np.ndarray, vel: np.ndarray) -> bool:
        if not self.gate_centers_neu:
            return False

        target = self.gate_centers_neu[
            min(self.current_gate_idx, len(self.gate_centers_neu) - 1)
        ].copy()
        if target[2] < 1.0:
            target[2] = 1.0

        waypoints = np.vstack([pos, target])
        times = allocate_segment_times(
            waypoints,
            current_vel=vel,
            vmax=2.5,
            amax=2.0,
            T_min=1.0,
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
        self.last_planned_gate_idx = int(self.current_gate_idx)
        return True

    def _advance_gate_if_needed(self, pos: np.ndarray) -> bool:
        if not self.gate_centers_neu:
            return False
        if self.current_gate_idx >= len(self.gate_centers_neu) - 1:
            return False

        target = (
            self.current_gate_pos
            if self.current_gate_pos is not None
            else self.gate_centers_neu[self.current_gate_idx]
        )
        distance = float(np.linalg.norm(pos - target))
        if distance >= self.pass_radius_m:
            return False

        self.current_gate_idx += 1
        self.active_waypoints = None
        self.active_times = None
        return True

    def _should_plan(self, advanced: bool) -> bool:
        if advanced:
            return True
        if self.active_waypoints is None or self.planner.total_time <= 0.0:
            return True
        elapsed = time.time() - self.trajectory_start_time
        if elapsed > float(self.planner.total_time) + self.replan_after_trajectory_s:
            return True
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

    def _perception_gates_from_snapshot(self, snapshot) -> list[np.ndarray]:
        latest_perception = getattr(snapshot, "latest_perception", None)
        if isinstance(latest_perception, dict):
            self._update_gate_memory(latest_perception)

        stable_tracks = self.gate_memory.get_stable_tracks()
        if not stable_tracks:
            return []

        gates = [track.center.copy() for track in stable_tracks]
        signature = tuple(
            (track.id, *self._rounded_gate(track.center, decimals=2))
            for track in stable_tracks
        )
        if signature != self._last_stable_gate_print_signature:
            coords = " ".join(
                f"id={track.id}:({track.center[0]:.2f}, {track.center[1]:.2f}, {track.center[2]:.2f})"
                for track in stable_tracks
            )
            print(f"autonomy_wrapper stable perception gates NEU: {coords}", flush=True)
            self._last_stable_gate_print_signature = signature
        return gates

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
            center_camera = detection.get("gate_center_camera")
            if center_camera is not None:
                center_camera = np.asarray(center_camera, dtype=float)
                if center_camera.shape != (3,) or not np.all(np.isfinite(center_camera)):
                    center_camera = None

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
            return []

        gates = []
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

        return gates

    def _install_gate_centers(self, gates: list[np.ndarray]) -> None:
        signature = self._gate_signature(gates)
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
        self.active_track_count = len(self.gate_centers_neu)
        self.current_gate_idx = min(
            int(self.current_gate_idx),
            max(0, len(self.gate_centers_neu) - 1),
        )
        next_target = (
            self.gate_centers_neu[self.current_gate_idx].copy()
            if self.gate_centers_neu
            else None
        )
        self._last_gate_signature = signature

        if next_target is None:
            self.current_gate_pos = None
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
            self.active_waypoints = None
            self.active_times = None
            self.last_planned_gate_idx = -1

    @classmethod
    def _gate_signature(cls, gates: list[np.ndarray]) -> tuple:
        return tuple(cls._rounded_gate(gate, decimals=1) for gate in gates)

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
