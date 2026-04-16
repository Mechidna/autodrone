import numpy as np
import time
import math
from autonomy_core.planning.minimum_snap_planner import GateTrajectoryPlanner
from autonomy_core.launch.get_telemetry import GetTelemetry
from autonomy_core.controller.attitude_controller3 import RPGHighLevelTracker
from autonomy_core.perception.gate_perception import GatePerception
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
    def __init__(self, use_perception=False):
        self.use_perception = use_perception
        self.gate_perception = GatePerception() if use_perception else None

        self.current_gate_pos = np.array([0.0, 0.0, 0.0], dtype=float)
        self.gate_confidence = 0.0
        self.planner = GateTrajectoryPlanner()
        self.telemetry = GetTelemetry()

        self.replan_time = 0.0
        self.trajectory_start_time = 0.0
        self.time_elapsed = 0.0
        self.segment_target = np.zeros(3, dtype=float)
        self.segment_duration = 1.0
        self.last_control_time = None
        self.error_z = 0.0
        self.last_desired_yaw = 0.0

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
            thrust_max=0.79,
        )

        # Ground-truth gates for planner debugging
        self.gt_gates = [
            np.array([0.0, 8.0, 1.5], dtype=float),
            np.array([0.8, 16.0, 1.5], dtype=float),
            np.array([-0.8, 24.0, 1.5], dtype=float),
        ]
        self.current_gate_idx = 0

    def choose_T(self, p0, v0, p1, vmax=1.5, amax=1.5, T_min=2.0):
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

    def process_frame(self, frame, camera_matrix, dist_coeffs):
        if not self.use_perception:
            # In planner-debug mode, do not use camera perception at all.
            return self.gt_gates[self.current_gate_idx].copy()

        perception = self.gate_perception.process(frame, camera_matrix, dist_coeffs)
        if perception is None or perception["confidence"] is None:
            print("No gate detected!")
            return np.zeros(3)

        self.gate_confidence = perception["confidence"]
        gate_t = perception["t"]
        gate_t = np.array([-gate_t[0][0], gate_t[2][0], -gate_t[1][0]], dtype=float)

        gate_xyz = np.array([
            self.telemetry.pos["x"],
            self.telemetry.pos["y"],
            self.telemetry.pos["z"],
        ], dtype=float) + gate_t

        self.current_gate_pos = gate_xyz
        return gate_xyz

    def get_current_target_gate(self):
        if self.current_gate_idx >= len(self.gt_gates):
            return self.gt_gates[-1].copy()
        return self.gt_gates[self.current_gate_idx].copy()

    def advance_gate_if_needed(self, threshold=1.0):
        pos = np.array([
            self.telemetry.pos["x"],
            self.telemetry.pos["y"],
            self.telemetry.pos["z"],
        ], dtype=float)

        target = self.get_current_target_gate()
        dist = np.linalg.norm(pos - target)

        if dist < threshold and self.current_gate_idx < len(self.gt_gates) - 1:
            self.current_gate_idx += 1
            print(f"Advancing to gate {self.current_gate_idx}: {self.gt_gates[self.current_gate_idx]}")

    def path_plan(self, gate_xyz=None):
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

        if gate_xyz is None:
            gate_xyz = self.get_current_target_gate()

        p1 = np.array(gate_xyz, dtype=float)
        v1 = np.array([0.0, 0.0, 0.0], dtype=float)

        T = self.choose_T(pos, vel, p1, vmax=1.5, amax=1.5)

        print("telemetry pos:", pos)
        print("planning to gate:", p1)
        print("Time Horizon:", T)

        self.current_gate_pos = p1
        self.segment_target = p1
        self.segment_duration = T
        self.trajectory_start_time = time.time()

        self.planner.update(pos, vel, p1, v1, T)

        for tau in np.linspace(0, T, 6):
            p, v, a = self.planner.sample(tau)
            print("tau =", tau)
            print("p =", p)
            print("v =", v)
            print("a =", a)

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

        self.time_elapsed = time.time() - self.trajectory_start_time
        p_ref, v_ref, a_ref = self.planner.sample(self.time_elapsed)

        desired_yaw = compute_desired_yaw(v_ref, a_ref, self.last_desired_yaw)
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
    api = AutonomyAPI(use_perception=False)

    api.telemetry.pos = {"x": 0.0, "y": 0.0, "z": 0.0}
    api.telemetry.vel = {"vx": 0.0, "vy": 0.0, "vz": 0.0}
    api.telemetry.rpy = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
    api.last_desired_yaw = 0.0

    # Build first trajectory from GT gate list
    api.path_plan()

    print("Starting mock trajectory test...")

    while True:
        roll, pitch, yaw, thrust = api.attitude_control()

        # Perfect tracking mock
        p, v, _ = api.planner.sample(api.time_elapsed)

        api.telemetry.pos["x"] = float(p[0])
        api.telemetry.pos["y"] = float(p[1])
        api.telemetry.pos["z"] = float(p[2])

        api.telemetry.vel["vx"] = float(v[0])
        api.telemetry.vel["vy"] = float(v[1])
        api.telemetry.vel["vz"] = float(v[2])

        # tracker yaw_cmd is in radians, but your telemetry code stores degrees
        api.telemetry.rpy["yaw"] = float(np.degrees(yaw))

        print(f"roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}, thrust={thrust:.2f}")

        # Gate progression
        api.advance_gate_if_needed(threshold=1.0)

        # Replan either when segment ends or when gate index changes
        if api.time_elapsed >= api.segment_duration:
            api.path_plan()

        time.sleep(0.02)