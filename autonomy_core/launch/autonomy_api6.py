import numpy as np
import time
import math
from autonomy_core.planning.minimum_snap_planner_multi_time_optimized import MultiSegmentMinimumSnapPlanner
from autonomy_core.launch.get_telemetry import GetTelemetry
from autonomy_core.controller.attitude_controller3 import RPGHighLevelTracker
from autonomy_core.perception.gate_perception import GatePerception
from autonomy_core.perception.gate_perception_node import GatePerceptionNode
from autonomy_core.perception.gate_memory import GateMemory
from dataclasses import dataclass


def compute_desired_yaw(v_ref, a_ref, last_yaw, eps=1e-3):
    v_xy = np.array(v_ref[:2], dtype=float)
    a_xy = np.array(a_ref[:2], dtype=float)

    if np.linalg.norm(v_xy) > eps:
        return np.arctan2(v_xy[1], v_xy[0])

    if np.linalg.norm(a_xy) > eps:
        return np.arctan2(a_xy[1], a_xy[0])

    return last_yaw


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
    def __init__(self, use_perception=False, race_gate_count=None):
        self.use_perception = use_perception

        self.gate_perception = GatePerception() if use_perception else None
        self.perception_node = GatePerceptionNode(self.gate_perception) if use_perception else None
        self.gate_memory = GateMemory(
            association_radius=2.0,
            commit_radius=2.0,
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

        # Landmark-memory stays forever.
        # This set is ONLY mission progress for the current lap/cycle.
        self.completed_track_ids_this_cycle = set()

        # Count of gate-pass events in the current cycle/lap.
        self.completed_gate_events_this_cycle = 0

        # If None, do not auto-reset cycles.
        # If an integer, reset only after this many gate passes.
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

        drone_yaw_rad = float(self.telemetry.rpy["yaw"]) * np.pi / 180.0

        det = self.perception_node.detect_gate(
            frame=frame,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            drone_pos=drone_pos,
            drone_yaw_rad=drone_yaw_rad,
        )

        now = time.time()

        if det is None:
            self.gate_memory.prune(now)
            return None

        self.gate_confidence = float(det["confidence"])

        result = self.gate_memory.add_detection(
            center=det["gate_center_world"],
            confidence=det["confidence"],
            timestamp=now,
        )

        self.gate_memory.prune(now)

        if result is not None:
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

    def get_committed_waypoints(self):
        committed = self.gate_memory.get_committed_centers()
        return [np.asarray(p, dtype=float) for p in committed]

    def maybe_reset_cycle_by_gate_count(self):
        """
        Reset lap/cycle only after a known number of gate passes.

        If race_gate_count is None:
            do not auto-reset.

        If race_gate_count is an integer:
            reset only after that many gate-pass events in this cycle.
        """
        if self.race_gate_count is None:
            return

        if self.completed_gate_events_this_cycle >= self.race_gate_count:
            print(
                f"Reached race_gate_count={self.race_gate_count}. "
                f"Resetting cycle."
            )
            self.completed_track_ids_this_cycle.clear()
            self.completed_gate_events_this_cycle = 0

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
        Build planning horizon from persistent gate landmarks, excluding only the gates
        completed in the current cycle/lap.

        Returns:
            waypoints: np.ndarray shape (N,3)
            target_gates: list[np.ndarray]
            target_track_ids: list[int]
        """
        self.maybe_reset_cycle_by_gate_count()

        committed_tracks = self.gate_memory.get_committed_tracks()
        if len(committed_tracks) == 0:
            return np.array([current_pos], dtype=float), [], []

        available_tracks = [
            tr for tr in committed_tracks
            if tr.id not in self.completed_track_ids_this_cycle
        ]

        if len(available_tracks) == 0:
            return np.array([current_pos], dtype=float), [], []

        # Simple first ordering rule:
        # prefer gates that are ahead in +y.
        ahead_tracks = [tr for tr in available_tracks if tr.center[1] > current_pos[1] + 1.0]

        # If nothing is ahead, fall back to available non-completed tracks.
        # This is important for loop-like or curved courses where "ahead in y" may not hold.
        candidate_tracks = ahead_tracks if len(ahead_tracks) > 0 else available_tracks

        # Sort by y as a simple baseline ordering.
        candidate_tracks.sort(key=lambda tr: tr.center[1])

        target_tracks = candidate_tracks[:max_gates_ahead]
        target_gates = [tr.center.copy() for tr in target_tracks]
        target_track_ids = [tr.id for tr in target_tracks]

        return np.vstack([current_pos] + target_gates), target_gates, target_track_ids

    def build_waypoint_horizon_from_gt(self, current_pos, max_gates_ahead=3):
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
            if not (0 <= self.current_target_idx < len(self.active_target_gates)):
                return False

            target = self.active_target_gates[self.current_target_idx]
            dist = np.linalg.norm(pos - target)

            if dist < threshold:
                track_id = None
                if 0 <= self.current_target_idx < len(self.active_target_track_ids):
                    track_id = self.active_target_track_ids[self.current_target_idx]

                print(
                    f"Perception mode: passed target gate {self.current_target_idx}: "
                    f"{target}, track_id={track_id}"
                )

                if track_id is not None and track_id >= 0:
                    self.completed_track_ids_this_cycle.add(track_id)

                # Count the gate-pass event for lap/cycle reset logic
                self.completed_gate_events_this_cycle += 1

                print(
                    f"completed_gate_events_this_cycle = "
                    f"{self.completed_gate_events_this_cycle}"
                )

                self.current_target_idx += 1
                return True

            return False

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
            uses remaining GT gates
        In perception mode:
            uses committed non-redundant gates from GateMemory,
            excluding only the gates completed in the current cycle.
        """
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
            max_gates_ahead=3,
        )

        print("=== REPLAN DEBUG ===")
        for tr in self.gate_memory.get_committed_tracks():
            print(f"track_id={tr.id}, center={tr.center}")
        print("active_target_track_ids:", self.active_target_track_ids)
        print("completed_track_ids_this_cycle:", sorted(self.completed_track_ids_this_cycle))
        print("completed_gate_events_this_cycle:", self.completed_gate_events_this_cycle)
        print("race_gate_count:", self.race_gate_count)
        print("telemetry pos:", pos)

        if len(waypoints) < 2 or len(target_gates) == 0:
            print("No valid gates available to plan to.")
            return False

        # Keep the currently planned target list stable until next replan
        self.active_target_gates = [g.copy() for g in target_gates]
        self.active_target_track_ids = list(target_track_ids)
        self.current_target_idx = 0

        times_init = self.allocate_segment_times(
            waypoints,
            current_vel=vel,
            vmax=2.5,
            amax=2.0,
            T_min=1.0,
        )

        v_end = self.compute_final_exit_velocity(target_gates, default_speed=2.5)

        print("planning waypoint horizon:")
        for i, wp in enumerate(waypoints):
            print(f"  wp[{i}] = {wp}")
        print("target_track_ids:", self.active_target_track_ids)
        print("completed_track_ids_this_cycle:", sorted(self.completed_track_ids_this_cycle))
        print("completed_gate_events_this_cycle:", self.completed_gate_events_this_cycle)
        print("race_gate_count:", self.race_gate_count)
        print("initial segment times:", times_init)
        print("initial total horizon:", float(np.sum(times_init)))
        print("terminal v_end:", v_end)

        times_opt, result = self.planner.optimize_times(
            waypoints=waypoints,
            times_init=times_init,
            v_start=vel,
            v_end=v_end,
            a_start=np.zeros(3, dtype=float),
            a_end=np.zeros(3, dtype=float),
            j_start=np.zeros(3, dtype=float),
            j_end=np.zeros(3, dtype=float),
            lambda_time=1.0,
            lambda_snap=0.01,
            t_min=0.1,
            maxiter=20,
        )

        print("optimized segment times:", times_opt)
        print("optimized total horizon:", float(np.sum(times_opt)))
        print("time optimization success:", result.success)
        print("time optimization message:", result.message)

        self.current_gate_pos = waypoints[1].copy()
        self.active_waypoints = waypoints.copy()
        self.active_times = np.asarray(times_opt, dtype=float).copy()
        self.trajectory_start_time = time.time()

        if not self.use_perception:
            self.last_planned_gate_idx = self.current_gate_idx

        total_time = self.planner.total_time
        for tau in np.linspace(0.0, total_time, 8):
            p, v, a = self.planner.sample(tau)
            print("tau =", tau)
            print("p =", p)
            print("v =", v)
            print("a =", a)

        return True

    # -------------------------------------------------------------------------
    # Control
    # -------------------------------------------------------------------------

    def attitude_control(self):
        current_yaw_rad = float(self.telemetry.rpy["yaw"]) * math.pi / 180.0

        pos = np.array([
            self.telemetry.pos["x"],
            self.telemetry.pos["y"],
            self.telemetry.pos["z"]
        ], dtype=float)

        state = State(
            pos=pos,
            vel=np.array([
                self.telemetry.vel["vx"],
                self.telemetry.vel["vy"],
                self.telemetry.vel["vz"]
            ], dtype=float),
            yaw=current_yaw_rad,
        )

        # Guard against calling before a plan exists
        if self.active_waypoints is None or self.planner.total_time <= 0.0:
            print("No active trajectory; returning hover-ish neutral command.")
            return 0.0, 0.0, current_yaw_rad, self.tracker.thrust_hover

        self.time_elapsed = time.time() - self.trajectory_start_time
        p_ref, v_ref, a_ref = self.planner.sample(self.time_elapsed)

        self.p_ref = np.array(p_ref, dtype=float)
        self.v_ref = np.array(v_ref, dtype=float)
        self.a_ref = np.array(a_ref, dtype=float)

        if 0 <= self.current_target_idx < len(self.active_target_gates):
            self.current_target_gate = self.active_target_gates[self.current_target_idx].copy()
        else:
            self.current_target_gate = self.current_gate_pos.copy()

        desired_yaw = compute_desired_yaw(v_ref, a_ref, self.last_desired_yaw)
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
