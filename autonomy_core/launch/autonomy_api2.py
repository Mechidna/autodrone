import numpy as np
import time
from autonomy_core.planning.test import GateTrajectoryPlanner
from autonomy_core.planning.path_visualizer2 import planner_visual
from autonomy_core.launch.get_telemetry import GetTelemetry
from autonomy_core.controller.attitude_controller2 import DroneAttitudeController

class AutonomyAPI:
    def __init__(self):
        self.planner = GateTrajectoryPlanner()
        self.telemetry = GetTelemetry()
        self.attitude_controller = DroneAttitudeController()
        self.trajectory_start_time = 0

    def path_plan(self):
        telemetry = self.telemetry



        p1 = np.array([10, 5, 5])  # Target (Gate)
        v1 = np.array([5, 0, 0])  # Target Velocity (fast pass-through)
        T = 5.0  # Duration (s)

        p0 = np.array([telemetry.p0["x"], telemetry.p0["y"], telemetry.p0["z"]])
        pos = np.array([telemetry.pos["x"], telemetry.pos["y"], telemetry.pos["z"]])
        vel = np.array([telemetry.vel["vx"], telemetry.vel["vy"], telemetry.vel["vz"]])
        pos = np.array([20, 0, 10])  # Start Position
        vel = np.array([1, 0, 0])  # Start Velocity (moving forward)
        rpy = np.array([telemetry.rpy["roll"], telemetry.rpy["pitch"], telemetry.rpy["yaw"]])

        self.trajectory_start_time = time.time()
        self.planner.update(pos, vel, p1, v1, T)
        planner_visual(pos,p1,T)

    def attitude_control(self):
        telemetry = self.telemetry
        current_time = time.time()
        time_elapsed = current_time - self.trajectory_start_time
        target_p, target_v, target_a = self.planner.sample(time_elapsed)
        pos = np.array([telemetry.pos["x"], telemetry.pos["y"], telemetry.pos["z"]])
        vel = np.array([telemetry.vel["vx"], telemetry.vel["vy"], telemetry.vel["vz"]])
        rpy = np.array([telemetry.rpy["roll"], telemetry.rpy["pitch"], telemetry.rpy["yaw"]])
        target_yaw = np.arctan2(target_v[1], target_v[0])
        roll, pitch, yaw, thrust = self.attitude_controller.get_tilt_commands(target_p, target_v, target_a, pos, vel, float(rpy[2]), target_yaw)
        return roll, pitch, yaw, thrust

if __name__ == "__main__":
    autonomy_api = AutonomyAPI()
    autonomy_api.path_plan()
# # --- INSIDE YOUR 200 HZ LOOP ---
#
# # 1. Check the time
# current_time = time.time()
# time_elapsed = current_time - trajectory_start_time
#
# # 2. SAMPLE exactly one point from the polynomial
# # This is where 'np.polyval' happens for THIS specific millisecond
# target_p, target_v, target_a = planner.sample(time_elapsed)
#
# # 3. Use those specific values in your PID logic
# roll, pitch, yaw, thrust = controller.compute(
#     target_p, target_v, target_a,  # The "Sample"
#     estimated_p, estimated_v,      # From your EKF
#     target_yaw
# )