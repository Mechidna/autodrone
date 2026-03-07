from autonomy_core.perception.gate_perception import GatePerception
from autonomy_core.planning.gate_path_planner import GatePathPlanner
from autonomy_core.planning.path_visualizer import PathVisualizer
from autonomy_core.planning.trajectory_generator import TrajectoryGenerator
from autonomy_core.controller.attitude_controller import ClassicalController
import math
import numpy as np
def cam_to_enu_matrix(yaw_enu: float) -> np.ndarray:
    """
    Map OpenCV camera coords (x right, y down, z forward)
    into ENU world coords (E, N, U) using yaw only.

    For ENU:
      forward = [cos(yaw), sin(yaw), 0]
      right   = [sin(yaw), -cos(yaw), 0]
      down    = [0, 0, -1]
    """
    cy = math.cos(yaw_enu)
    sy = math.sin(yaw_enu)

    x_cam_enu = np.array([ sy, -cy,  0.0])  # right
    y_cam_enu = np.array([0.0,  0.0, -1.0]) # down
    z_cam_enu = np.array([ cy,  sy,  0.0])  # forward

    return np.column_stack([x_cam_enu, y_cam_enu, z_cam_enu])  # 3x3
def t_cam_to_enu(t_cam: np.ndarray, yaw_enu: float) -> np.ndarray:
    M = cam_to_enu_matrix(yaw_enu)
    return M @ t_cam.reshape(3,)

class AutonomyAPI:
    def __init__(self):
        self.gate_perception = GatePerception()
        self.path_planner = GatePathPlanner()
        self.path_visualizer = PathVisualizer()
        self.trajectory_generator = TrajectoryGenerator()

        self.controller = ClassicalController(
            kp_pos=(2.0, 2.0, 3.0),
            kd_vel=(1.5, 1.5, 2.0),
            hover_throttle=0.75,          # will tune
            throttle_accel_gain=0.05,
            yaw_mode="angle",
        )

    def process_frame(self, frame, camera_matrix, dist_coeffs,telemetry_pos,
                      telemetry_vel,telemetry_rpy,dt):
        current_state = {
            "pos": telemetry_pos,   # np.array([x,y,z])
            "vel": telemetry_vel,   # np.array([vx,vy,vz])
            "rpy": telemetry_rpy,   # np.array([x,y,z])
            "yaw": telemetry_rpy[2],   # float (rad)
        }
        perception = self.gate_perception.process(frame, camera_matrix, dist_coeffs)
        if not isinstance(perception, dict):
            print("No detection")
            return None
        ## yaw debugging
        t_cam = np.asarray(perception["t"], dtype=float).reshape(3, )
        yaw_enu = float(current_state["yaw"])  # you already pass ENU yaw in
        t_enu = t_cam_to_enu(t_cam, yaw_enu)
        perception["t_cam"] = t_cam
        perception["t"] = t_enu
        perception["R"] = np.eye(3)
        yaw = yaw_enu
        forward_enu = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=float)
        # Fake R so that R[:,2] == forward_enu (so your normal_axis=2 works)
        R_fake = np.eye(3)
        R_fake[:, 2] = forward_enu
        # choose a "right" axis for column 0 orthogonal to forward
        right_enu = np.array([math.sin(yaw), -math.cos(yaw), 0.0], dtype=float)
        R_fake[:, 0] = right_enu
        R_fake[:, 1] = np.array([0.0, 0.0, 1.0], dtype=float)  # up
        perception["R"] = R_fake
        ##
        plan = self.path_planner.plan(perception)
        self.path_visualizer.visualize(plan)
        if plan is None:
            print("No path planning")
            return None
        trajectory = self.trajectory_generator.step(plan,current_state,dt)
        if trajectory is None:
            print("No trajectory")
            return None
        cmd = self.controller.step(current_state, trajectory, dt)
        if cmd is None:
            print("No controller")
            return None
        return {
            "perception": perception,   # contains R, t, confidence
            "plan": plan,                # contains target_position, stage, gate_normal, etc.
            "trajectory": trajectory,
            "controller": cmd
        }

if __name__ == "__main__":
    import cv2
    import numpy as np
    from autonomy_core.perception.gate_perception import GatePerception
    from autonomy_core.planning.gate_path_planner import GatePathPlanner
    from autonomy_core.planning.trajectory_generator import TrajectoryGenerator
    from autonomy_core.planning.path_visualizer import PathVisualizer
    from autonomy_core.controller.attitude_controller import ClassicalController
    ## initialize
    autonomy = AutonomyAPI()
    ## mock data (delete later)
    telemetry_pos = np.zeros(3)
    telemetry_vel = np.zeros(3)
    telemetry_rpy = np.zeros(3)
    telemetry_yaw = telemetry_rpy[2]
    dt = 1.0 / 60.0
    ## mock gate image
    width = 640
    height = 480
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(
        frame,
        (200, 120),
        (440, 360),
        (0, 140, 255),  # Orange
        thickness=20
    )
    ## mock camera intrinsics
    fx = 600
    fy = 600
    cx = width / 2
    cy = height / 2
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1))
    ##
    ## loop
    while True:
        autonomy.process_frame(frame, camera_matrix, dist_coeffs,
                               telemetry_pos,telemetry_vel,telemetry_rpy,dt)