# px4_runner.py
import asyncio
import hashlib
import json
import math
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, CameraInfo
from tf2_msgs.msg import TFMessage
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
CAMERA_OFFSET_BODY = np.array([0.12, 0.03, 0.242], dtype=float)
GAZEBO_DYNAMIC_POSE_TOPIC = "/world/gate_test_1500mm_blue/dynamic_pose/info"
GAZEBO_MODEL_NAME = "x500_mono_cam_0"
GAZEBO_POSE_SOURCE = "gazebo_dynamic_pose_x500_mono_cam_0"


def rotate_vector_by_quaternion(quat_xyzw, vector):
    x, y, z, w = np.asarray(quat_xyzw, dtype=float).reshape(4)
    q_vec = np.array([x, y, z], dtype=float)
    vector = np.asarray(vector, dtype=float).reshape(3)
    return (
        vector
        + 2.0 * w * np.cross(q_vec, vector)
        + 2.0 * np.cross(q_vec, np.cross(q_vec, vector))
    )


def rad2deg(x):
    return x * 180.0 / math.pi


def hover_command(autonomy):
    yaw_rad = float(autonomy.telemetry.rpy["yaw"])
    return 0.0, 0.0, yaw_rad, autonomy.tracker.thrust_hover


# -------------------------------------------------
# ROS2 camera adapter
# -------------------------------------------------
class PerceptionNode(Node):
    def __init__(self, raw_dataset_saver=None, telemetry_obj=None):
        super().__init__("perception_adapter_node")

        self.bridge = CvBridge()
        self.raw_dataset_saver = raw_dataset_saver
        self.telemetry = telemetry_obj

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
        self.create_subscription(
            TFMessage,
            GAZEBO_DYNAMIC_POSE_TOPIC,
            self.gazebo_pose_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info("PerceptionNode started.")

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_matrix = np.array(msg.k, dtype=np.float32).reshape(3, 3)

        d = np.array(msg.d, dtype=np.float32) if len(msg.d) > 0 else np.zeros((5,), dtype=np.float32)
        self.dist_coeffs = d.reshape(-1, 1)

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.frame = frame
            self.last_frame_time = time.time()
            if self.raw_dataset_saver is not None and self.telemetry is not None:
                self.raw_dataset_saver.maybe_save_from_callback(
                    frame,
                    self.camera_matrix,
                    self.dist_coeffs,
                    self.telemetry,
                    msg.header.stamp,
                )
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def gazebo_pose_callback(self, msg: TFMessage):
        if self.raw_dataset_saver is None:
            return

        if len(msg.transforms) == 0:
            print("[RAW DATASET WARN] Gazebo dynamic pose TFMessage has no transforms; falling back to MAVSDK metadata.")
            self.raw_dataset_saver.latest_gazebo_pose = None
            return

        selected_transform = None
        selection_method = "child_frame_id"
        for transform in msg.transforms:
            if transform.child_frame_id != GAZEBO_MODEL_NAME:
                continue
            selected_transform = transform
            break

        if selected_transform is None:
            selected_transform = msg.transforms[0]
            selection_method = "fallback_transforms_0"

        translation = selected_transform.transform.translation
        rotation = selected_transform.transform.rotation
        stamp = selected_transform.header.stamp
        model_pos_world = np.array([
            translation.x,
            translation.y,
            translation.z,
        ], dtype=float)
        model_quat_world = np.array([
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w,
        ], dtype=float)
        camera_pos_world = (
            model_pos_world
            + rotate_vector_by_quaternion(model_quat_world, CAMERA_OFFSET_BODY)
        )

        self.raw_dataset_saver.update_gazebo_pose(
            model_pos_world=model_pos_world,
            model_quat_world=model_quat_world,
            camera_pos_world=camera_pos_world,
            camera_quat_world=model_quat_world.copy(),
            ros_stamp_sec=int(stamp.sec),
            ros_stamp_nanosec=int(stamp.nanosec),
            wall_time=time.time(),
            selection_method=selection_method,
        )


def ros_spin_thread(node):
    rclpy.spin(node)


class RawDatasetFrameSaver:
    def __init__(self, enabled=True, capture_hz=5.0, root_dir="~/datasets/gazebo_gate_capture"):
        self.enabled = bool(enabled)
        self.capture_period_s = 1.0 / float(capture_hz) if float(capture_hz) > 0.0 else 0.0
        self.root_dir = os.path.expanduser(root_dir)
        self.images_dir = os.path.join(self.root_dir, "images")
        self.metadata_dir = os.path.join(self.root_dir, "metadata")
        self.next_capture_time = 0.0
        self.last_saved_image_stamp = None
        self.saved_image_hash_poses = {}
        self.telemetry_ready = False
        self.latest_gazebo_pose = None
        self.frame_idx = 0

        if self.enabled:
            os.makedirs(self.images_dir, exist_ok=True)
            os.makedirs(self.metadata_dir, exist_ok=True)
            print(
                f"[RAW DATASET] saving raw frames at {float(capture_hz):.2f} Hz "
                f"to {self.root_dir}"
            )

    @staticmethod
    def _json_list(value):
        return np.asarray(value, dtype=float).tolist()

    @staticmethod
    def _telemetry_float(container, key):
        return float(container[key])

    @staticmethod
    def _image_hash_md5(frame):
        return hashlib.md5(np.ascontiguousarray(frame).tobytes()).hexdigest()

    @staticmethod
    def _pose_changed(previous, current, atol=1e-6):
        previous_pos, previous_rpy = previous
        current_pos, current_rpy = current
        return (
            not np.allclose(previous_pos, current_pos, atol=atol, rtol=0.0)
            or not np.allclose(previous_rpy, current_rpy, atol=atol, rtol=0.0)
        )

    def update_gazebo_pose(
        self,
        model_pos_world,
        model_quat_world,
        camera_pos_world,
        camera_quat_world,
        ros_stamp_sec,
        ros_stamp_nanosec,
        wall_time,
        selection_method,
    ):
        self.latest_gazebo_pose = {
            "gazebo_model_pos_world": self._json_list(model_pos_world),
            "gazebo_model_quat_world": self._json_list(model_quat_world),
            "gazebo_camera_pos_world": self._json_list(camera_pos_world),
            "gazebo_camera_quat_world": self._json_list(camera_quat_world),
            "gazebo_pose_ros_stamp_sec": int(ros_stamp_sec),
            "gazebo_pose_ros_stamp_nanosec": int(ros_stamp_nanosec),
            "gazebo_pose_wall_time": float(wall_time),
            "gazebo_pose_selection_method": str(selection_method),
        }

    def maybe_save_from_callback(self, frame, camera_matrix, dist_coeffs, telemetry_obj, image_stamp):
        if (
            not self.enabled
            or not self.telemetry_ready
            or frame is None
            or camera_matrix is None
            or dist_coeffs is None
        ):
            return None

        stamp = (int(image_stamp.sec), int(image_stamp.nanosec))
        if stamp == self.last_saved_image_stamp:
            return None

        now = time.time()
        if now < self.next_capture_time:
            return None

        if self.capture_period_s > 0.0:
            self.next_capture_time = now + self.capture_period_s

        self.frame_idx += 1
        image_filename = f"frame_{self.frame_idx:06d}.jpg"
        metadata_filename = f"frame_{self.frame_idx:06d}.json"
        image_path = os.path.join(self.images_dir, image_filename)
        metadata_path = os.path.join(self.metadata_dir, metadata_filename)

        image_height, image_width = frame.shape[:2]
        gazebo_pose = self.latest_gazebo_pose
        if gazebo_pose is None:
            gazebo_metadata = {
                "gazebo_model_pos_world": None,
                "gazebo_model_quat_world": None,
                "gazebo_camera_pos_world": None,
                "gazebo_camera_quat_world": None,
                "gazebo_pose_ros_stamp_sec": None,
                "gazebo_pose_ros_stamp_nanosec": None,
                "gazebo_pose_wall_time": None,
                "gazebo_pose_selection_method": None,
                "pose_source": None,
                "gazebo_pose_age_s": None,
            }
        else:
            gazebo_pose_age_s = now - float(gazebo_pose["gazebo_pose_wall_time"])
            if gazebo_pose_age_s > 0.05:
                print(
                    "[RAW DATASET WARN] stale Gazebo pose "
                    f"age={gazebo_pose_age_s:.3f}s image_stamp={stamp}"
                )
            gazebo_metadata = {
                "gazebo_model_pos_world": gazebo_pose["gazebo_model_pos_world"],
                "gazebo_model_quat_world": gazebo_pose["gazebo_model_quat_world"],
                "gazebo_camera_pos_world": gazebo_pose["gazebo_camera_pos_world"],
                "gazebo_camera_quat_world": gazebo_pose["gazebo_camera_quat_world"],
                "gazebo_pose_ros_stamp_sec": gazebo_pose["gazebo_pose_ros_stamp_sec"],
                "gazebo_pose_ros_stamp_nanosec": gazebo_pose["gazebo_pose_ros_stamp_nanosec"],
                "gazebo_pose_wall_time": gazebo_pose["gazebo_pose_wall_time"],
                "gazebo_pose_selection_method": gazebo_pose["gazebo_pose_selection_method"],
                "pose_source": GAZEBO_POSE_SOURCE,
                "gazebo_pose_age_s": float(gazebo_pose_age_s),
            }
        drone_pos = [
            self._telemetry_float(telemetry_obj.pos, "x"),
            self._telemetry_float(telemetry_obj.pos, "y"),
            self._telemetry_float(telemetry_obj.pos, "z"),
        ]
        drone_rpy_rad = [
            self._telemetry_float(telemetry_obj.rpy, "roll"),
            self._telemetry_float(telemetry_obj.rpy, "pitch"),
            self._telemetry_float(telemetry_obj.rpy, "yaw"),
        ]
        image_hash_md5 = self._image_hash_md5(frame)
        metadata = {
            "image_filename": image_filename,
            "timestamp": float(now),
            "metadata_write_time": float(now),
            "ros_image_stamp_sec": stamp[0],
            "ros_image_stamp_nanosec": stamp[1],
            "drone_pos": drone_pos,
            "drone_rpy_rad": drone_rpy_rad,
            "camera_matrix": self._json_list(camera_matrix),
            "dist_coeffs": self._json_list(dist_coeffs),
            "image_width": int(image_width),
            "image_height": int(image_height),
            "image_hash_md5": image_hash_md5,
        }
        metadata.update(gazebo_metadata)

        if not cv2.imwrite(image_path, frame):
            print(f"[RAW DATASET] failed to write image: {image_path}")
            self.frame_idx -= 1
            return None

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        current_pose = (np.asarray(drone_pos, dtype=float), np.asarray(drone_rpy_rad, dtype=float))
        previous_pose = self.saved_image_hash_poses.get(image_hash_md5)
        if previous_pose is not None and self._pose_changed(previous_pose, current_pose):
            print(
                "[RAW DATASET WARN] duplicate image hash saved with different telemetry "
                f"hash={image_hash_md5} image={image_filename}"
            )
        self.saved_image_hash_poses.setdefault(image_hash_md5, current_pose)
        self.last_saved_image_stamp = stamp

        if self.frame_idx % 50 == 0:
            print(f"[RAW DATASET] saved {self.frame_idx} frames")
            print(
                "[RAW DATASET] gazebo pose "
                f"selection={metadata.get('gazebo_pose_selection_method')} "
                f"model_pos={metadata.get('gazebo_model_pos_world')} "
                f"age_s={metadata.get('gazebo_pose_age_s')}"
            )

        return image_path, metadata_path


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
    save_raw_dataset_frames = False
    raw_dataset_capture_hz = 5.0
    startup_hover_s = 10.0
    raw_dataset_saver = RawDatasetFrameSaver(
        enabled=save_raw_dataset_frames,
        capture_hz=raw_dataset_capture_hz,
    )

    # ---------------- ROS2 startup ----------------
    rclpy.init()
    perception_node = PerceptionNode(
        raw_dataset_saver=raw_dataset_saver,
        telemetry_obj=telemetry,
    )
    ros_thread = threading.Thread(target=ros_spin_thread, args=(perception_node,), daemon=True)
    ros_thread.start()

    # ---------------- MAVSDK startup ----------------
    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected to drone.")
            break

    use_perception = True
    autonomy = AutonomyAPI(use_perception=use_perception, race_gate_count=3)
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
    raw_dataset_saver.telemetry_ready = True

    # Start live telemetry tasks
    asyncio.create_task(track_position(drone))
    asyncio.create_task(track_velocity(drone))
    asyncio.create_task(track_orientation(drone))

    use_camera = use_perception or save_raw_dataset_frames

    if use_camera:
        # Wait for ROS2 camera + camera_info when either perception or raw capture is active.
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

    if use_perception:
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

                gate_ready = (
                    result is not None
                    and (
                        result.get("stable_now", False)
                        or (
                            result.get("committed_now", False)
                            and not getattr(autonomy, "use_lookahead_gate_filter", True)
                        )
                    )
                )
                if gate_ready:
                    ok = autonomy.path_plan()
                    if ok:
                        print("Initial stable gate found; trajectory planned.")
                        break

            await asyncio.sleep(0.05)

        if (
            len(getattr(autonomy, "active_target_gates", [])) == 0
            and len(getattr(autonomy, "race_accepted_track_ids", [])) > 0
        ):
            ok = autonomy.path_plan()
            if ok:
                print("Initial stable gate was already admitted; trajectory planned.")

        # If no stable gate yet, that's okay.
        # attitude_control() should fall back to hover-ish neutral command until a plan exists.
    else:
        print("Perception disabled; planning initial trajectory from mock GT gates.")
        ok = autonomy.path_plan()
        if not ok:
            print("Initial mock-gate planning failed; vehicle will hold neutral command.")

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

        if startup_hover_s > 0.0:
            hover_started = time.time()
            print(f"[STARTUP HOVER] holding neutral hover for {startup_hover_s:.1f}s")
            while time.time() - hover_started < startup_hover_s:
                roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd = hover_command(autonomy)
                await drone.offboard.set_attitude(
                    Attitude(
                        roll_deg=rad2deg(roll_cmd),
                        pitch_deg=rad2deg(pitch_cmd),
                        yaw_deg=rad2deg(yaw_cmd),
                        thrust_value=thrust_cmd,
                    )
                )
                await asyncio.sleep(0.02)

        autonomy.seed_yaw_hold(float(telemetry.rpy["yaw"]), reason="after_startup_hover")
    except OffboardError as e:
        print(f"Offboard start failed: {e._result.result}")
        await drone.action.disarm()
        perception_node.destroy_node()
        rclpy.shutdown()
        return

    try:
        last_loop_time = time.time()

        while True:
            loop_start = time.time()
            loop_dt = loop_start - last_loop_time
            last_loop_time = loop_start

            mem_result = None
            replan_requested = False
            replan_duration = 0.0
            hold_command = False
            stale_command_suppressed = False
            autonomy.last_perception_replan_trigger = False

            if loop_dt > 0.1:
                print(f"[WARN] control loop gap {loop_dt:.3f}s; suppressing aggressive command.")
                stale_command_suppressed = True

            if use_camera:
                frame = perception_node.frame
                camera_matrix = perception_node.camera_matrix
                dist_coeffs = perception_node.dist_coeffs

                if frame is None or camera_matrix is None or dist_coeffs is None:
                    print("Waiting for camera data...")
                    await asyncio.sleep(0.05)
                    continue

            if use_perception:
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

            # A new future gate became usable for perception planning.
            if mem_result is not None and (
                mem_result.get("stable_now", False)
                or (
                    mem_result.get("committed_now", False)
                    and not getattr(autonomy, "use_lookahead_gate_filter", True)
                )
            ):
                print("New committed/stable gate -> replanning.")
                should_replan = True
                if use_perception:
                    autonomy.last_perception_replan_trigger = True

            if (
                use_perception
                and len(getattr(autonomy, "active_target_gates", [])) == 0
                and len(getattr(autonomy, "race_accepted_track_ids", [])) > 0
            ):
                should_replan = True
                autonomy.last_perception_replan_trigger = True

            # Current active target passed
            gate_changed = autonomy.advance_gate_if_needed(threshold=1.0)
            if gate_changed:
                print("Active target advanced -> replanning.")
                should_replan = True
                if use_perception:
                    autonomy.last_perception_replan_trigger = True

            # Current trajectory finished
            if autonomy.planner.total_time > 0.0 and autonomy.time_elapsed >= autonomy.planner.total_time:
                print("Trajectory horizon exhausted -> replanning.")
                should_replan = True
                if use_perception:
                    autonomy.last_perception_replan_trigger = True

            # Avoid absurdly frequent replans, except perception gate completion.
            # Once a perceived landmark is marked complete, the old target must
            # be replaced immediately so it cannot be tracked again.
            force_perception_gate_replan = use_perception and gate_changed
            if should_replan and (
                force_perception_gate_replan
                or time.time() - autonomy.replan_time > 0.3
            ):
                replan_requested = True
                hold_command = True
                replan_started = time.time()
                print(f"[REPLAN] start t={replan_started:.3f}")
                ok = autonomy.path_plan()
                replan_duration = time.time() - replan_started
                print(
                    f"[REPLAN] end duration={replan_duration:.3f}s "
                    f"mode={autonomy.last_plan_mode} "
                    f"start_gate={autonomy.last_plan_start_gate_idx} "
                    f"end_gate={autonomy.last_plan_end_gate_idx}"
                )
                if not ok:
                    print("Replan requested, but no valid gates available yet.")

            # -------------------------------------------------
            # 3) Track current trajectory (or hover if none yet)
            # -------------------------------------------------
            if (
                (hold_command or stale_command_suppressed)
                and use_perception
                and len(getattr(autonomy, "active_target_gates", [])) == 0
            ):
                roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd = autonomy.attitude_control()
            elif hold_command or stale_command_suppressed:
                roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd = hover_command(autonomy)
            else:
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
            if target is None and use_perception:
                target = getattr(autonomy, "hold_anchor", None)

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
                loop_dt=loop_dt,
                replan_requested=replan_requested,
                replan_duration=replan_duration,
                hold_command=hold_command,
                stale_command_suppressed=stale_command_suppressed,
                plan_mode=getattr(autonomy, "last_plan_mode", None),
                plan_start_gate_idx=getattr(autonomy, "last_plan_start_gate_idx", None),
                plan_end_gate_idx=getattr(autonomy, "last_plan_end_gate_idx", None),
                raw_gate=getattr(autonomy, "last_raw_gate_center", None),
                perception_accepted=getattr(autonomy, "last_perception_accepted", False),
                perception_rejection_reason=getattr(autonomy, "last_perception_rejection_reason", ""),
                last_valid_target=getattr(autonomy, "last_valid_target", None),
                target_z_clamped=getattr(autonomy, "last_target_z_clamped", False),
                perception_replan_trigger=getattr(autonomy, "last_perception_replan_trigger", False),
                distance_to_active_target=getattr(autonomy, "distance_to_active_target", float("nan")),
                gate_completion_triggered=getattr(autonomy, "gate_completion_triggered", False),
                completion_reason=getattr(autonomy, "completion_reason", ""),
                completed_gate_position=getattr(autonomy, "completed_gate_position", None),
                active_gate_idx_before=getattr(autonomy, "active_gate_idx_before", None),
                active_gate_idx_after=getattr(autonomy, "active_gate_idx_after", None),
                race_cursor_before=getattr(autonomy, "race_cursor_before", None),
                race_cursor_after=getattr(autonomy, "race_cursor_after", None),
                active_target_source=getattr(autonomy, "active_target_source", ""),
                target_rejected_completed=getattr(autonomy, "target_rejected_completed", False),
                candidate_track_id=getattr(autonomy, "candidate_track_id", None),
                candidate_center=getattr(autonomy, "candidate_center", None),
                candidate_order_score=getattr(autonomy, "candidate_order_score", float("nan")),
                rejected_wrong_order=getattr(autonomy, "rejected_wrong_order", False),
                rejected_duplicate=getattr(autonomy, "rejected_duplicate", False),
                rejected_completed_this_lap=getattr(autonomy, "rejected_completed_this_lap", False),
                race_cursor_advanced=getattr(autonomy, "race_cursor_advanced", False),
                active_gate_idx_advanced=getattr(autonomy, "active_gate_idx_advanced", False),
                completed_landmark_count=getattr(autonomy, "completed_landmark_count", 0),
                lap_reset_triggered=getattr(autonomy, "lap_reset_triggered", False),
                active_target_cleared=getattr(autonomy, "active_target_cleared", False),
                active_target_track_id=getattr(autonomy, "active_target_track_id", None),
                completed_gate_track_id=getattr(autonomy, "completed_gate_track_id", None),
                yaw_target_source=getattr(autonomy, "yaw_target_source", ""),
                target_retained_after_completion=getattr(autonomy, "target_retained_after_completion", False),
                next_valid_target_found=getattr(autonomy, "next_valid_target_found", False),
                valid_candidate_count=getattr(autonomy, "valid_candidate_count", 0),
                approach_vector=getattr(autonomy, "approach_vector", None),
                gate_progress_along_approach=getattr(autonomy, "gate_progress_along_approach", float("nan")),
                gate_lateral_error=getattr(autonomy, "gate_lateral_error", float("nan")),
                gate_plane_crossed=getattr(autonomy, "gate_plane_crossed", False),
                near_gate_but_not_crossed=getattr(autonomy, "near_gate_but_not_crossed", False),
                completion_blocked_reason=getattr(autonomy, "completion_blocked_reason", ""),
                no_active_target=getattr(autonomy, "no_active_target", False),
                no_target_control_mode=getattr(autonomy, "no_target_control_mode", ""),
                hold_anchor_source=getattr(autonomy, "hold_anchor_source", ""),
                hold_anchor=getattr(autonomy, "hold_anchor", None),
                velocity_damping_active=getattr(autonomy, "velocity_damping_active", False),
                completed_gate_reference_blocked=getattr(autonomy, "completed_gate_reference_blocked", False),
                p_ref_source=getattr(autonomy, "p_ref_source", ""),
                yaw_hold_value=math.degrees(getattr(autonomy, "yaw_hold_value", float("nan"))),
                yaw_hold_value_deg=math.degrees(getattr(autonomy, "yaw_hold_value", float("nan"))),
                current_yaw_rad_deg=math.degrees(float(getattr(autonomy.telemetry, "rpy", {}).get("yaw", float("nan")))),
                perception_hold_yaw_deg=math.degrees(getattr(autonomy, "perception_hold_yaw", float("nan"))),
                telemetry_yaw_deg=math.degrees(float(getattr(autonomy.telemetry, "rpy", {}).get("yaw", float("nan")))),
                previous_yaw_cmd_deg=math.degrees(getattr(autonomy, "previous_yaw_cmd_log", float("nan"))),
                raw_yaw_cmd_deg=math.degrees(getattr(autonomy, "raw_yaw_cmd", float("nan"))),
                yaw_cmd_after_unwrap_deg=math.degrees(getattr(autonomy, "yaw_cmd_after_unwrap", float("nan"))),
                has_commanded_yaw_reference=getattr(autonomy, "has_commanded_yaw_reference", False),
                yaw_rate_limited=getattr(autonomy, "yaw_rate_limited", False),
                post_completion_grace_active=getattr(autonomy, "post_completion_grace_active", False),
                next_track_available_after_completion=getattr(autonomy, "next_track_available_after_completion", False),
                skipped_target_clear_after_completion=getattr(autonomy, "skipped_target_clear_after_completion", False),
                next_track_after_completion_id=getattr(autonomy, "next_track_after_completion_id", None),
                next_target_installed_same_cycle=getattr(autonomy, "next_target_installed_same_cycle", False),
                target_clear_reason=getattr(autonomy, "target_clear_reason", ""),
                post_completion_grace_suppressed=getattr(autonomy, "post_completion_grace_suppressed", False),
                planning_horizon_track_ids=getattr(autonomy, "planning_horizon_track_ids", []),
                planning_horizon_waypoint_count=getattr(autonomy, "planning_horizon_waypoint_count", 0),
                planning_horizon_waypoints=getattr(autonomy, "planning_horizon_waypoints", ""),
                future_track_visible_before_completion=getattr(autonomy, "future_track_visible_before_completion", False),
                future_track_blocked_reason=getattr(autonomy, "future_track_blocked_reason", ""),
                horizon_build_cursor=getattr(autonomy, "horizon_build_cursor", 0),
                horizon_available_order=getattr(autonomy, "horizon_available_order", []),
                horizon_selected_track_ids=getattr(autonomy, "horizon_selected_track_ids", []),
                horizon_rejected_track_ids=getattr(autonomy, "horizon_rejected_track_ids", []),
                horizon_rejection_reason=getattr(autonomy, "horizon_rejection_reason", ""),
                planning_lookahead_track_ids=getattr(autonomy, "planning_lookahead_track_ids", []),
                planning_lookahead_source=getattr(autonomy, "planning_lookahead_source", ""),
                planning_lookahead_used=getattr(autonomy, "planning_lookahead_used", False),
                passthrough_velocity_enabled=getattr(autonomy, "passthrough_velocity_enabled", False),
                passthrough_speed_used=getattr(autonomy, "passthrough_speed_used", float("nan")),
                waypoint_velocity_log=getattr(autonomy, "waypoint_velocity_log", None),
                internal_gate_velocity_nonzero=getattr(autonomy, "internal_gate_velocity_nonzero", False),
                terminal_velocity_mode=getattr(autonomy, "terminal_velocity_mode", ""),
                replan_reason=getattr(autonomy, "replan_reason", ""),
                no_target_roll_source=getattr(autonomy, "no_target_roll_source", ""),
                no_target_pitch_source=getattr(autonomy, "no_target_pitch_source", ""),
                horizontal_hold_disabled_after_completion=getattr(autonomy, "horizontal_hold_disabled_after_completion", False),
                track_id=getattr(autonomy, "track_id", None),
                merged_into_track_id=getattr(autonomy, "merged_into_track_id", None),
                duplicate_merge_reason=getattr(autonomy, "duplicate_merge_reason", ""),
                race_order_track_ids=getattr(autonomy, "race_order_track_ids", []),
                race_order_inserted=getattr(autonomy, "race_order_inserted", False),
                race_order_rejected_reason=getattr(autonomy, "race_order_rejected_reason", ""),
                landmark_uncertainty=getattr(autonomy, "landmark_uncertainty", float("nan")),
                track_observations=getattr(autonomy, "track_observations", 0),
                completed_unique_gate_count=getattr(autonomy, "completed_unique_gate_count", 0),
                active_gate_idx_clamped_by_race_gate_count=getattr(autonomy, "active_gate_idx_clamped_by_race_gate_count", False),
                suspected_duplicate_track=getattr(autonomy, "suspected_duplicate_track", False),
                committed_track_centers=getattr(autonomy, "committed_track_centers_log", ""),
                pairwise_committed_track_distances=getattr(autonomy, "pairwise_committed_track_distances", ""),
                duplicate_radius_used=getattr(autonomy, "duplicate_radius_used", float("nan")),
                merge_candidate_pairs=getattr(autonomy, "merge_candidate_pairs", ""),
                merge_blocked_reason=getattr(autonomy, "merge_blocked_reason", ""),
                rejected_track_temporary_vs_permanent=getattr(autonomy, "rejected_track_temporary_vs_permanent", ""),
                active_target_admission_status=getattr(autonomy, "active_target_admission_status", ""),
                race_order_after_merge=getattr(autonomy, "race_order_after_merge", []),
                tentative_track_ids=getattr(autonomy, "tentative_track_ids", []),
                stable_track_ids=getattr(autonomy, "stable_track_ids", []),
                race_admitted_track_ids=getattr(autonomy, "race_admitted_track_ids", []),
                selected_next_gate_track_id=getattr(autonomy, "selected_next_gate_track_id", None),
                selected_next_gate_stability_score=getattr(autonomy, "selected_next_gate_stability_score", float("nan")),
                track_hits=getattr(autonomy, "track_observations", 0),
                track_history_len=getattr(autonomy, "track_history_len", 0),
                track_filtered_center=getattr(autonomy, "track_filtered_center", None),
                track_raw_latest_center=getattr(autonomy, "track_raw_latest_center", None),
                track_center_std=getattr(autonomy, "track_center_std", None),
                track_center_std_norm=getattr(autonomy, "track_center_std_norm", float("nan")),
                track_camera_std_norm=getattr(autonomy, "track_camera_std_norm", float("nan")),
                track_reprojection_error_mean=getattr(autonomy, "track_reprojection_error_mean", float("nan")),
                track_reprojection_error_median=getattr(autonomy, "track_reprojection_error_median", float("nan")),
                track_outlier_count=getattr(autonomy, "track_outlier_count", 0),
                track_inlier_count=getattr(autonomy, "track_inlier_count", 0),
                track_is_stable=getattr(autonomy, "track_is_stable", False),
                track_stability_score=getattr(autonomy, "track_stability_score", float("nan")),
                promotion_reason=getattr(autonomy, "promotion_reason", ""),
                promotion_blocked_reason=getattr(autonomy, "promotion_blocked_reason", ""),
                selected_target_source=getattr(autonomy, "selected_target_source", ""),
                raw_image_corners=getattr(autonomy, "last_raw_image_corners", None),
                ordered_image_corners=getattr(autonomy, "last_ordered_image_corners", None),
                pnp_rvec=getattr(autonomy, "last_pnp_rvec", None),
                pnp_tvec=getattr(autonomy, "last_pnp_tvec", None),
                gate_center_camera=getattr(autonomy, "last_gate_center_camera", None),
                gate_center_body=getattr(autonomy, "last_gate_center_body", None),
                gate_center_world_debug=getattr(autonomy, "last_gate_center_world_debug", None),
                gate_normal_world=getattr(autonomy, "last_gate_normal_world", None),
                gate_normal_camera=getattr(autonomy, "last_gate_normal_camera", None),
                detection_drone_pose=getattr(autonomy, "last_detection_drone_pose", None),
                reprojection_error=getattr(autonomy, "last_reprojection_error", float("nan")),
                reprojected_corners=getattr(autonomy, "last_reprojected_image_corners", None),
                corner_reprojection_error_px=getattr(
                    autonomy,
                    "last_corner_reprojection_error_px",
                    float("nan"),
                ),
                quad_center_x=getattr(autonomy, "last_quad_center_x", float("nan")),
                quad_center_y=getattr(autonomy, "last_quad_center_y", float("nan")),
                image_center_x=getattr(autonomy, "last_image_center_x", float("nan")),
                image_center_y=getattr(autonomy, "last_image_center_y", float("nan")),
                quad_center_offset_x=getattr(autonomy, "last_quad_center_offset_x", float("nan")),
                quad_center_offset_y=getattr(autonomy, "last_quad_center_offset_y", float("nan")),
                quad_width_px=getattr(autonomy, "last_quad_width_px", float("nan")),
                quad_height_px=getattr(autonomy, "last_quad_height_px", float("nan")),
                quad_aspect_ratio=getattr(autonomy, "last_quad_aspect_ratio", float("nan")),
                quad_area_px=getattr(autonomy, "last_quad_area_px", float("nan")),
                pnp_candidate_count=getattr(autonomy, "last_pnp_candidate_count", 0),
                chosen_pnp_candidate=getattr(autonomy, "last_chosen_pnp_candidate", None),
                pnp_candidate_0_error=getattr(autonomy, "last_pnp_candidate_0_error", float("nan")),
                pnp_candidate_1_error=getattr(autonomy, "last_pnp_candidate_1_error", float("nan")),
                transform_source=getattr(autonomy, "last_transform_source", ""),
                camera_to_body_matrix_used=getattr(
                    autonomy,
                    "last_camera_to_body_matrix_used",
                    None,
                ),
                body_to_world_method_used=getattr(
                    autonomy,
                    "last_body_to_world_method_used",
                    "",
                ),
                expected_gate_cam=getattr(autonomy, "expected_gate_cam", None),
                pnp_gate_cam=getattr(autonomy, "pnp_gate_cam", None),
                camera_error=getattr(autonomy, "camera_error", None),
                camera_error_norm=getattr(autonomy, "camera_error_norm", float("nan")),
                expected_gate_body=getattr(autonomy, "expected_gate_body", None),
                pnp_gate_body=getattr(autonomy, "pnp_gate_body", None),
                body_error=getattr(autonomy, "body_error", None),
                expected_gate_world=getattr(autonomy, "expected_gate_world", None),
                pnp_gate_world=getattr(autonomy, "pnp_gate_world", None),
                world_error=getattr(autonomy, "world_error", None),
                world_error_norm=getattr(autonomy, "world_error_norm", float("nan")),
                pnp_size_190_cam=getattr(autonomy, "pnp_size_190_cam", None),
                pnp_size_200_cam=getattr(autonomy, "pnp_size_200_cam", None),
                pnp_size_210_cam=getattr(autonomy, "pnp_size_210_cam", None),
                pnp_size_190_world=getattr(autonomy, "pnp_size_190_world", None),
                pnp_size_200_world=getattr(autonomy, "pnp_size_200_world", None),
                pnp_size_210_world=getattr(autonomy, "pnp_size_210_world", None),
                pnp_size_190_reproj_error=getattr(autonomy, "pnp_size_190_reproj_error", float("nan")),
                pnp_size_200_reproj_error=getattr(autonomy, "pnp_size_200_reproj_error", float("nan")),
                pnp_size_210_reproj_error=getattr(autonomy, "pnp_size_210_reproj_error", float("nan")),
                pnp_size_190_gt_error=getattr(autonomy, "pnp_size_190_gt_error", float("nan")),
                pnp_size_200_gt_error=getattr(autonomy, "pnp_size_200_gt_error", float("nan")),
                pnp_size_210_gt_error=getattr(autonomy, "pnp_size_210_gt_error", float("nan")),
                pnp_solver_used=getattr(autonomy, "pnp_solver_used", ""),
                live_solver_name=getattr(autonomy, "live_solver_name", ""),
                live_solver_world=getattr(autonomy, "live_solver_world", None),
                live_solver_reproj_error=getattr(autonomy, "live_solver_reproj_error", float("nan")),
                ippe_world_error_gt=getattr(autonomy, "ippe_world_error_gt", float("nan")),
                iterative_world_error_gt=getattr(autonomy, "iterative_world_error_gt", float("nan")),
                pnp_fallback_reason=getattr(autonomy, "pnp_fallback_reason", ""),
                pnp_best_debug_solver=getattr(autonomy, "pnp_best_debug_solver", ""),
                pnp_best_debug_order=getattr(autonomy, "pnp_best_debug_order", ""),
                pnp_current_world_error_gt=getattr(autonomy, "pnp_current_world_error_gt", float("nan")),
                pnp_best_world_error_gt=getattr(autonomy, "pnp_best_world_error_gt", float("nan")),
                pnp_current_cam=getattr(autonomy, "pnp_current_cam", None),
                pnp_best_cam=getattr(autonomy, "pnp_best_cam", None),
                pnp_current_world=getattr(autonomy, "pnp_current_world", None),
                pnp_best_world=getattr(autonomy, "pnp_best_world", None),
                pnp_current_reproj_error=getattr(autonomy, "pnp_current_reproj_error", float("nan")),
                pnp_best_reproj_error=getattr(autonomy, "pnp_best_reproj_error", float("nan")),
                pnp_candidate0_world=getattr(autonomy, "pnp_candidate0_world", None),
                pnp_candidate1_world=getattr(autonomy, "pnp_candidate1_world", None),
                pnp_candidate0_error=getattr(autonomy, "pnp_candidate0_error", float("nan")),
                pnp_candidate1_error=getattr(autonomy, "pnp_candidate1_error", float("nan")),
                pnp_gt_projected_center=getattr(autonomy, "pnp_gt_projected_center", None),
                pnp_gt_projected_quad_center_error_px=getattr(
                    autonomy,
                    "pnp_gt_projected_quad_center_error_px",
                    float("nan"),
                ),
                image_width=getattr(autonomy, "image_width", 0),
                image_height=getattr(autonomy, "image_height", 0),
                min_corner_x=getattr(autonomy, "min_corner_x", float("nan")),
                max_corner_x=getattr(autonomy, "max_corner_x", float("nan")),
                min_corner_y=getattr(autonomy, "min_corner_y", float("nan")),
                max_corner_y=getattr(autonomy, "max_corner_y", float("nan")),
                corner_margin_ok=getattr(autonomy, "corner_margin_ok", False),
                clipped_detection_rejected=getattr(autonomy, "clipped_detection_rejected", False),
                rejected_near_image_edge=getattr(autonomy, "rejected_near_image_edge", False),
                track_update_innovation=getattr(autonomy, "track_update_innovation", float("nan")),
                track_update_accepted=getattr(autonomy, "track_update_accepted", False),
                track_center_before_update=getattr(autonomy, "track_center_before_update", None),
                track_center_after_update=getattr(autonomy, "track_center_after_update", None),
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
