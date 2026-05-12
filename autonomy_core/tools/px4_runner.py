# px4_runner.py
import asyncio
import math
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from mavsdk import System
from mavsdk.offboard import Attitude, OffboardError

from autonomy_core.launch.get_telemetry import GetTelemetry
from autonomy_core.launch.autonomy_api6 import AutonomyAPI
from flight_logger import FlightLogger


# -------------------------------------------------
# Shared telemetry object
# -------------------------------------------------
telemetry = GetTelemetry()


def rad2deg(x):
    return x * 180.0 / math.pi


# -------------------------------------------------
# ROS2 camera adapter
# -------------------------------------------------
class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_adapter_node")

        self.bridge = CvBridge()

        self.frame = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.last_frame_time = None

        camera_info_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(
            Image,
            "/camera",
            self.image_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            "/camera_info",
            self.camera_info_callback,
            camera_info_qos,
        )

        self.get_logger().info("PerceptionNode started.")

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_matrix = np.array(msg.k, dtype=np.float32).reshape(3, 3)

        d = np.array(msg.d, dtype=np.float32) if len(msg.d) > 0 else np.zeros((5,), dtype=np.float32)
        self.dist_coeffs = d.reshape(-1, 1)

    def image_callback(self, msg: Image):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.last_frame_time = time.time()
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")


def ros_spin_thread(node):
    rclpy.spin(node)


# -------------------------------------------------
# Telemetry tracking tasks
# -------------------------------------------------
async def track_position(drone):
    async for pv in drone.telemetry.position_velocity_ned():
        pos_xyz = (
            pv.position.north_m,
            pv.position.east_m,
            -pv.position.down_m,
        )
        telemetry.telemetry_pos(pos_xyz, start_position=None)


async def track_velocity(drone):
    async for pv in drone.telemetry.position_velocity_ned():
        vel_xyz = (
            pv.velocity.north_m_s,
            pv.velocity.east_m_s,
            -pv.velocity.down_m_s,
        )
        telemetry.telemetry_vel(vel_xyz, start_velocity=None)


async def track_orientation(drone):
    async for att in drone.telemetry.attitude_euler():
        rpy_rad = (
            math.radians(att.roll_deg),
            math.radians(att.pitch_deg),
            math.radians(att.yaw_deg),
        )
        telemetry.telemetry_rpy(rpy_rad, start_orientation=None)


# -------------------------------------------------
# Main
# -------------------------------------------------
async def main():
    # ---------------- ROS2 startup ----------------
    rclpy.init()
    perception_node = PerceptionNode()
    ros_thread = threading.Thread(target=ros_spin_thread, args=(perception_node,), daemon=True)
    ros_thread.start()

    # ---------------- MAVSDK startup ----------------
    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected to drone.")
            break

    autonomy = AutonomyAPI(use_perception=True, race_gate_count=3)
    autonomy.telemetry = telemetry

    # -------------------------------------------------
    # Initial telemetry snapshot
    # -------------------------------------------------
    pv0 = await drone.telemetry.position_velocity_ned().__anext__()
    att0 = await drone.telemetry.attitude_euler().__anext__()

    start_position = (
        pv0.position.north_m,
        pv0.position.east_m,
        -pv0.position.down_m,
    )
    telemetry.telemetry_pos(start_position, start_position=start_position)

    start_velocity = (
        pv0.velocity.north_m_s,
        pv0.velocity.east_m_s,
        -pv0.velocity.down_m_s,
    )
    telemetry.telemetry_vel(start_velocity, start_velocity=start_velocity)

    start_orientation = (
        math.radians(att0.roll_deg),
        math.radians(att0.pitch_deg),
        math.radians(att0.yaw_deg),
    )
    telemetry.telemetry_rpy(start_orientation, start_orientation=start_orientation)

    # Start live telemetry tasks
    asyncio.create_task(track_position(drone))
    asyncio.create_task(track_velocity(drone))
    asyncio.create_task(track_orientation(drone))

    # Wait for ROS2 camera + camera_info
    print("Waiting for camera frame and intrinsics...")
    while (
        perception_node.frame is None
        or perception_node.camera_matrix is None
        or perception_node.dist_coeffs is None
    ):
        await asyncio.sleep(0.05)

    print("Camera data ready.")

    # Let telemetry populate once
    await asyncio.sleep(0.2)

    # -------------------------------------------------
    # Warm up perception and memory BEFORE arming
    # -------------------------------------------------
    print("Warming up perception memory before arming...")
    warmup_deadline = time.time() + 1.0

    while time.time() < warmup_deadline:
        frame = perception_node.frame
        camera_matrix = perception_node.camera_matrix
        dist_coeffs = perception_node.dist_coeffs

        if frame is not None and camera_matrix is not None and dist_coeffs is not None:
            result = autonomy.update_gate_memory_from_frame(frame, camera_matrix, dist_coeffs)

            if result is not None and result.get("committed_now", False):
                ok = autonomy.path_plan()
                if ok:
                    print("Initial committed gate found; trajectory planned.")
                    break

        await asyncio.sleep(0.05)

    # If no committed gate yet, that's okay.
    # attitude_control() should fall back to hover-ish neutral command until a plan exists.

    # -------------------------------------------------
    # Arm only after perception warmup / initial planning attempt
    # -------------------------------------------------
    await drone.action.arm()

    # Compute first command
    roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd = autonomy.attitude_control()

    print(
        f"Initial command: "
        f"roll={rad2deg(roll_cmd):.2f} deg, "
        f"pitch={rad2deg(pitch_cmd):.2f} deg, "
        f"yaw={rad2deg(yaw_cmd):.2f} deg, "
        f"thrust={thrust_cmd:.2f}"
    )

    # Send a few setpoints before starting offboard
    for _ in range(50):
        await drone.offboard.set_attitude(
            Attitude(
                roll_deg=rad2deg(roll_cmd),
                pitch_deg=rad2deg(pitch_cmd),
                yaw_deg=rad2deg(yaw_cmd),
                thrust_value=thrust_cmd,
            )
        )
        await asyncio.sleep(0.02)

    try:
        await drone.offboard.start()
        print("Offboard started.")
        flight_logger = FlightLogger("flight_log.csv")
    except OffboardError as e:
        print(f"Offboard start failed: {e._result.result}")
        await drone.action.disarm()
        perception_node.destroy_node()
        rclpy.shutdown()
        return

    try:
        while True:
            frame = perception_node.frame
            camera_matrix = perception_node.camera_matrix
            dist_coeffs = perception_node.dist_coeffs

            if frame is None or camera_matrix is None or dist_coeffs is None:
                print("Waiting for camera data...")
                await asyncio.sleep(0.05)
                continue

            # -------------------------------------------------
            # 1) Update gate memory from current frame
            # -------------------------------------------------
            mem_result = autonomy.update_gate_memory_from_frame(
                frame,
                camera_matrix,
                dist_coeffs,
            )

            # -------------------------------------------------
            # 2) Replan only when it makes sense
            # -------------------------------------------------
            should_replan = False

            # A new stable landmark gate was committed
            if mem_result is not None and mem_result.get("committed_now", False):
                print("New committed gate -> replanning.")
                should_replan = True

            # Current active target passed
            gate_changed = autonomy.advance_gate_if_needed(threshold=1.0)
            if gate_changed:
                print("Active target advanced -> replanning.")
                should_replan = True

            # Current trajectory finished
            if autonomy.planner.total_time > 0.0 and autonomy.time_elapsed >= autonomy.planner.total_time:
                print("Trajectory horizon exhausted -> replanning.")
                should_replan = True

            # Avoid absurdly frequent replans
            if should_replan and (time.time() - autonomy.replan_time > 0.3):
                ok = autonomy.path_plan()
                if not ok:
                    print("Replan requested, but no valid gates available yet.")

            # -------------------------------------------------
            # 3) Track current trajectory (or hover if none yet)
            # -------------------------------------------------
            roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd = autonomy.attitude_control()


            # flight logger
            p_ref = getattr(autonomy, "p_ref", None)
            v_ref = getattr(autonomy, "v_ref", None)
            a_ref = getattr(autonomy, "a_ref", None)

            target = getattr(autonomy, "current_gate", None)
            if target is None:
                target = getattr(autonomy, "current_target_gate", None)
            if target is None:
                target = getattr(autonomy, "target_gate", None)
            if target is None:
                target = getattr(autonomy, "active_gate", None)

            mode = getattr(autonomy, "mode", None)

            active_gate_idx = getattr(autonomy, "active_gate_idx", None)
            if active_gate_idx is None:
                active_gate_idx = getattr(autonomy, "current_gate_idx", None)
            if active_gate_idx is None:
                active_gate_idx = getattr(autonomy, "target_gate_idx", None)

            flight_logger.log(
                telemetry=telemetry,
                roll_cmd=roll_cmd,
                pitch_cmd=pitch_cmd,
                yaw_cmd=yaw_cmd,
                thrust_cmd=thrust_cmd,
                p_ref=p_ref,
                v_ref=v_ref,
                a_ref=a_ref,
                target=target,
                mode=mode,
                active_gate_idx=active_gate_idx,
            )

            await drone.offboard.set_attitude(
                Attitude(
                    roll_deg=rad2deg(roll_cmd),
                    pitch_deg=rad2deg(pitch_cmd),
                    yaw_deg=rad2deg(yaw_cmd),
                    thrust_value=thrust_cmd,
                )
            )

            print(
                f"cmd | roll={rad2deg(roll_cmd):6.2f} deg "
                f"pitch={rad2deg(pitch_cmd):6.2f} deg "
                f"yaw={rad2deg(yaw_cmd):7.2f} deg "
                f"thrust={thrust_cmd:4.2f}"
            )

            await asyncio.sleep(0.02)

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        try:
            flight_logger.close()
        except Exception:
            pass

        try:
            await drone.offboard.stop()
        except Exception:
            pass

        try:
            await drone.action.land()
        except Exception:
            pass

        perception_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    asyncio.run(main())