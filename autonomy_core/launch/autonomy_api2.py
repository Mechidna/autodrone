import numpy as np
import time
import math
from autonomy_core.planning.path_planner import GateTrajectoryPlanner
from autonomy_core.planning.path_visualizer2 import planner_visual
from autonomy_core.launch.get_telemetry import GetTelemetry
from autonomy_core.controller.attitude_controller2 import DroneAttitudeController
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
    pos: np.ndarray   # shape (3,), world frame, z-up
    vel: np.ndarray   # shape (3,), world frame, z-up
    yaw: float        # current yaw [rad]


@dataclass
class Reference:
    pos: np.ndarray   # shape (3,)
    vel: np.ndarray   # shape (3,)
    acc: np.ndarray   # shape (3,)
    yaw: float        # desired yaw [rad]
    yaw_rate: float = 0.0

class AutonomyAPI:
    def __init__(self):
        self.gate_perception = GatePerception()
        self.current_gate_pos = np.array([0.0, 0.0, 0.0], dtype=float)
        self.gate_confidence = 0.0
        self.planner = GateTrajectoryPlanner()
        self.telemetry = GetTelemetry()
        self.attitude_controller = DroneAttitudeController()
        self.replan_time = 0.0
        self.trajectory_start_time = 0.0
        self.time_elapsed = 0.0
        self.segment_target = np.zeros(3, dtype=float)
        self.segment_duration = 1.0
        self.last_control_time = None
        self.error_z = 0.0
        self.last_desired_yaw = float(self.telemetry.rpy["yaw"])
        self.tracker = RPGHighLevelTracker(
            mass=1.0,  # set your mass if you use it later
            gravity=9.81,
            kp=(2.5, 2.5, 3.5),
            kv=(2.0, 2.0, 2.6),
            max_tilt_deg=20.0,
            max_acc_xy=2.0,
            max_acc_z_up=2.5,
            max_acc_z_down=2.0,
            thrust_hover=0.74,  # tune from hover test
            thrust_min=0.60,
            thrust_max=0.79
        )

    def choose_T(self, p0, v0, p1, vmax=1.5, amax=1.5, T_min=2.0):
        dp = p1 - p0
        d = np.linalg.norm(dp)

        if d < 1e-6:
            return T_min

        dir_vec = dp / d
        v_along = np.dot(v0, dir_vec)

        # Base trapezoidal/triangular estimate
        t_acc = vmax / amax
        d_acc = 0.5 * amax * t_acc ** 2

        if d > 2 * d_acc:
            T_base = 2 * t_acc + (d - 2 * d_acc) / vmax
        else:
            T_base = 2 * np.sqrt(d / amax)

        # Inflate if moving the wrong way or too fast
        if v_along < 0:
            T_base += min(abs(v_along) / amax, 2.0)
        else:
            T_base -= min(v_along / (2 * amax), 0.5)

        return max(T_base, T_min)

    def process_frame(self, frame, camera_matrix, dist_coeffs):
        perception = self.gate_perception.process(frame, camera_matrix, dist_coeffs)
        if perception is None:
            print("No gate detected!")
            return np.zeros(3)  # or handle the "no gate" case as needed
        if perception["confidence"] is None:
            return np.zeros(3)
        else:
            self.gate_confidence = perception["confidence"]
            print(perception["confidence"])
        gate_t = perception["t"]
        gate_t = [-gate_t[0][0], gate_t[2][0], -gate_t[1][0]]
        gate_xyz = np.array([self.telemetry.pos["x"], self.telemetry.pos["y"], self.telemetry.pos["z"]], dtype=float) + np.array(gate_t)
        self.current_gate_pos = gate_xyz
        return gate_xyz

    def path_plan(self,gate_xyz):
        telemetry = self.telemetry
        self.replan_time = time.time()

        print("telemetry pos:", telemetry.pos)
        print("gate xyz used for planning:", self.current_gate_pos)
        print("segment target before replan:", self.segment_target)

        pos = np.array([
            telemetry.pos["x"],
            telemetry.pos["y"],
            telemetry.pos["z"]
        ], dtype=float)

        # p1 = np.array([-0.1, 10.0, 1.0], dtype=float)
        p1 = gate_xyz
        v1 = np.array([0.0, 0.0, 0.0], dtype=float)
        # T = 5

        vel = np.array([
            telemetry.vel["vx"],
            telemetry.vel["vy"],
            telemetry.vel["vz"]
        ], dtype=float)

        T = self.choose_T(pos, vel, p1, vmax=1.5, amax=1.5)
        print("Time Horizon:",T)
        self.segment_target = p1
        self.segment_duration = T
        self.trajectory_start_time = time.time()

        self.planner.update(pos, vel, p1, v1, T)

    # def attitude_control(self):           # <---------- Broken Control
    #     telemetry = self.telemetry
    #
    #     current_time = time.time()
    #     self.time_elapsed = current_time - self.trajectory_start_time
    #     if self.last_control_time is None:
    #         dt = 0.02
    #     else:
    #         dt = current_time - self.last_control_time
    #     self.last_control_time = current_time
    #     print("dt:",dt)
    #
    #     target_p, target_v, target_a = self.planner.sample(self.time_elapsed)
    #
    #     pos = np.array([
    #         telemetry.pos["x"],
    #         telemetry.pos["y"],
    #         telemetry.pos["z"]
    #     ], dtype=float)
    #
    #     vel = np.array([
    #         telemetry.vel["vx"],
    #         telemetry.vel["vy"],
    #         telemetry.vel["vz"]
    #     ], dtype=float)
    #
    #     rpy = np.array([
    #         telemetry.rpy["roll"],
    #         telemetry.rpy["pitch"],
    #         telemetry.rpy["yaw"]
    #     ], dtype=float)
    #
    #     # target_yaw = np.arctan2(target_v[1], target_v[0])
    #     target_yaw = float(rpy[2])
    #
    #     roll, pitch, yaw, thrust, error_z = self.attitude_controller.get_tilt_commands(
    #         target_p, target_v, target_a,
    #         pos, vel,
    #         float(rpy[0]), float(rpy[1]), float(rpy[2]),
    #         target_yaw,
    #         dt
    #     )
    #     self.error_z = error_z
    #
    #     return roll, pitch, yaw, thrust

    def attitude_control(self):                 # <----------- Working Control
        telemetry = self.telemetry
        planner = self.planner

        current_yaw_rad = float(telemetry.rpy["yaw"]) * math.pi / 180.0
        pos = np.array([
            telemetry.pos["x"],
            telemetry.pos["y"],
            telemetry.pos["z"]
        ], dtype=float)
        # current state, in z-up world frame
        state = State(
            pos=pos,
            vel=np.array([telemetry.vel["vx"], telemetry.vel["vy"], telemetry.vel["vz"]], dtype=float),
            yaw=current_yaw_rad,
        )

        t_now = time.time() - self.trajectory_start_time
        # reference from planner
        p_ref, v_ref, a_ref = planner.sample(t_now)

        desired_yaw = compute_desired_yaw(
            v_ref=v_ref,
            a_ref=a_ref,
            last_yaw=self.last_desired_yaw
        )
        self.last_desired_yaw = desired_yaw
        # desired_yaw = 90.0

        ref = Reference(
            pos=np.array(p_ref, dtype=float),
            vel=np.array(v_ref, dtype=float),
            acc=np.array(a_ref, dtype=float),
            yaw=desired_yaw,
        )

        roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd, dbg = self.tracker.update(state, ref)
        roll_cmd = -roll_cmd
        pitch_cmd = -pitch_cmd
        # yaw_cmd = 90.0 * math.pi / 180
        print("roll: ",roll_cmd)
        print("pitch: ",pitch_cmd)
        print(dbg)
        return roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd


if __name__ == "__main__":

    api = AutonomyAPI()

    # --------------------------------------------------
    # Initialize mock telemetry state
    # --------------------------------------------------
    api.telemetry.pos = {"x": 0.0, "y": 0.0, "z": 2.0}
    api.telemetry.vel = {"vx": 0.0, "vy": 0.0, "vz": 0.0}
    api.telemetry.rpy = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}

    # Build first trajectory
    api.path_plan()

    print("Starting mock trajectory test...")

    while True:

        # Run controller
        roll, pitch, yaw, thrust = api.attitude_control()

        # --------------------------------------------------
        # Mock drone physics (perfect tracking)
        # --------------------------------------------------
        p, v, _ = api.planner.sample(api.time_elapsed)

        api.telemetry.pos["x"] = float(p[0])
        api.telemetry.pos["y"] = float(p[1])
        api.telemetry.pos["z"] = float(p[2])

        api.telemetry.vel["vx"] = float(v[0])
        api.telemetry.vel["vy"] = float(v[1])
        api.telemetry.vel["vz"] = float(v[2])

        api.telemetry.rpy["yaw"] = float(yaw)

        # Print control outputs
        print(
            f"roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}, thrust={thrust:.2f}"
        )

        # --------------------------------------------------
        # When trajectory ends, create new one
        # --------------------------------------------------
        if api.time_elapsed >= api.segment_duration:

            p1 = np.random.uniform(0, 20, size=3)
            v1 = np.random.uniform(-2, 2, size=3)

            pos = np.array([
                api.telemetry.pos["x"],
                api.telemetry.pos["y"],
                api.telemetry.pos["z"]
            ])

            vel = np.array([
                api.telemetry.vel["vx"],
                api.telemetry.vel["vy"],
                api.telemetry.vel["vz"]
            ])

            T = np.random.uniform(4, 8)

            api.segment_target = p1
            api.segment_duration = T
            api.trajectory_start_time = time.time()

            api.planner.update(pos, vel, p1, v1, T)

            print("New trajectory generated")

        time.sleep(0.02)  # ~50Hz loop