import threading
import time
import math
import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge

from frame_capture import CameraFrameCapture
from runtime_config import load_runtime_config


def _normalize_quat(quat_xyzw):
    quat = np.asarray(quat_xyzw, dtype=float).reshape(4)
    norm = float(np.linalg.norm(quat))
    if not math.isfinite(norm) or norm <= 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return quat / norm


def _quat_multiply(lhs_xyzw, rhs_xyzw):
    x1, y1, z1, w1 = _normalize_quat(lhs_xyzw)
    x2, y2, z2, w2 = _normalize_quat(rhs_xyzw)
    return _normalize_quat(
        np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=float,
        )
    )


def _quat_from_rpy(roll: float, pitch: float, yaw: float):
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return _normalize_quat(
        np.array(
            [
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
                cr * cp * cy + sr * sp * sy,
            ],
            dtype=float,
        )
    )


def _rotate_vector_by_quat(quat_xyzw, vector):
    x, y, z, w = _normalize_quat(quat_xyzw)
    q_vec = np.array([x, y, z], dtype=float)
    vector = np.asarray(vector, dtype=float).reshape(3)
    return (
        vector
        + 2.0 * w * np.cross(q_vec, vector)
        + 2.0 * np.cross(q_vec, np.cross(q_vec, vector))
    )


def _pose_from_transform(transform):
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    stamp = transform.header.stamp
    return (
        np.array([translation.x, translation.y, translation.z], dtype=float),
        _normalize_quat([rotation.x, rotation.y, rotation.z, rotation.w]),
        int(stamp.sec),
        int(stamp.nanosec),
        str(transform.child_frame_id),
    )


class RosCameraNode(Node):
    def __init__(self, data, camera_topic=None, camera_info_topic=None):
        super().__init__("ros_camera_rx_node")

        config = load_runtime_config()
        if camera_topic is None:
            camera_topic = config.vision.ros_camera_topic
        if camera_info_topic is None:
            camera_info_topic = config.vision.ros_camera_info_topic

        self.data = data
        self.bridge = CvBridge()
        self.frame_id_counter = 0
        self.frame_capture = CameraFrameCapture(source="ros2_camera")
        self.gazebo_pose_lock = threading.Lock()
        self.latest_gazebo_pose = None

        audit_config = config.perception_geometry_audit
        perception_world_pose_source = str(config.perception.world_pose_source).lower()
        runner_mode = str(config.runtime.runner_mode).lower()
        vision_source = str(config.vision.source).lower()
        audit_needs_gazebo_pose = (
            bool(audit_config.enabled)
            and str(audit_config.reference_pose_source).lower() in ("gazebo", "both")
        )
        self.gazebo_pose_enabled = (
            runner_mode != "competition"
            and vision_source == "ros"
            and (
                audit_needs_gazebo_pose
                or perception_world_pose_source == "gazebo_camera_sim"
            )
        )
        self.gazebo_model_name = str(audit_config.gazebo_model_name)
        self.gazebo_camera_link_name = str(audit_config.gazebo_camera_link_name)
        self.gazebo_allow_pose_fallback = bool(audit_config.gazebo_allow_pose_fallback)
        self.gazebo_mount_translation = np.asarray(
            audit_config.gazebo_camera_mount_xyz,
            dtype=float,
        ).reshape(3)
        self.gazebo_mount_quat = _quat_from_rpy(*audit_config.gazebo_camera_mount_rpy)
        self.gazebo_dynamic_pose_topic = self._resolve_gazebo_dynamic_pose_topic(
            audit_config.gazebo_dynamic_pose_topic
        )

        self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data,
        )

        if self.gazebo_pose_enabled and self.gazebo_dynamic_pose_topic:
            self.create_subscription(
                TFMessage,
                self.gazebo_dynamic_pose_topic,
                self.gazebo_pose_callback,
                qos_profile_sensor_data,
            )
            self.get_logger().info(
                "Listening for Gazebo dynamic pose on "
                f"{self.gazebo_dynamic_pose_topic} model={self.gazebo_model_name}"
            )
        elif self.gazebo_pose_enabled:
            self.get_logger().warn(
                "Gazebo pose is required but no dynamic pose topic resolved. "
                "Set WORLD/PX4_GZ_WORLD or perception_geometry_audit.gazebo_dynamic_pose_topic."
            )

        self.get_logger().info(f"Listening for ROS camera frames on {camera_topic}")

    @staticmethod
    def _resolve_gazebo_dynamic_pose_topic(topic):
        topic = str(topic or "").strip()
        if topic and topic.lower() != "auto":
            return topic
        world = (
            os.environ.get("PX4_GZ_WORLD")
            or os.environ.get("WORLD")
            or ""
        ).strip()
        if not world:
            return ""
        return f"/world/{world}/dynamic_pose/info"

    def _get_lock(self):
        if isinstance(self.data, dict):
            return self.data.get("lock")
        return None

    def _store_frame(self, frame_data):
        lock = self._get_lock()

        if lock is not None:
            with lock:
                self.data["latest_frame"] = frame_data
                self.data["vision_frame_count"] = self.data.get("vision_frame_count", 0) + 1
        else:
            self.data["latest_frame"] = frame_data
            self.data["vision_frame_count"] = self.data.get("vision_frame_count", 0) + 1

    def _store_camera_info(self, msg):
        camera_info = {
            "width": msg.width,
            "height": msg.height,
            "k": list(msg.k),
            "d": list(msg.d),
            "r": list(msg.r),
            "p": list(msg.p),
            "distortion_model": msg.distortion_model,
            "stamp_sec": int(msg.header.stamp.sec),
            "stamp_nanosec": int(msg.header.stamp.nanosec),
            "wall_time": time.time(),
        }

        lock = self._get_lock()

        if lock is not None:
            with lock:
                self.data["camera_info"] = camera_info
        else:
            self.data["camera_info"] = camera_info

    def camera_info_callback(self, msg: CameraInfo):
        self._store_camera_info(msg)

    def image_callback(self, msg: Image):
        try:
            # OpenCV BGR image, same convention as vision_rx.py after cv2.imdecode(..., IMREAD_COLOR)
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"Failed to convert ROS image: {exc}")
            return

        stamp_sec = int(msg.header.stamp.sec)
        stamp_nanosec = int(msg.header.stamp.nanosec)
        sim_time_ns = stamp_sec * 1_000_000_000 + stamp_nanosec

        self.frame_id_counter += 1
        gazebo_pose_snapshot = self.latest_gazebo_pose_snapshot()

        frame_data = {
            "frame_id": self.frame_id_counter,
            "image": img,
            "shape": img.shape,
            "sim_time_ns": sim_time_ns,
            "wall_time": time.time(),
            "gazebo_pose": gazebo_pose_snapshot,

            # ROS-specific metadata, harmless for controller/autonomy_adapter
            "ros_stamp_sec": stamp_sec,
            "ros_stamp_nanosec": stamp_nanosec,
            "ros_frame_id": msg.header.frame_id,
            "source": "ros2_camera",
        }

        self._store_frame(frame_data)
        self.frame_capture.maybe_capture(frame_data, img)

    def gazebo_pose_callback(self, msg: TFMessage):
        if len(msg.transforms) == 0:
            with self.gazebo_pose_lock:
                self.latest_gazebo_pose = None
            self._store_latest_gazebo_pose(None)
            return

        model_tf = self._find_model_transform(msg)
        camera_tf = self._find_camera_transform(msg)
        wall_time = time.time()

        if camera_tf is not None:
            camera_pos, camera_quat, sec, nanosec, child = _pose_from_transform(camera_tf)
            if model_tf is not None:
                model_pos, model_quat, _m_sec, _m_nanosec, _model_child = (
                    _pose_from_transform(model_tf)
                )
            else:
                model_pos, model_quat = camera_pos.copy(), camera_quat.copy()
            selection_method = f"camera_link:{child}"
        elif model_tf is not None:
            model_pos, model_quat, sec, nanosec, child = _pose_from_transform(model_tf)
            camera_pos = model_pos + _rotate_vector_by_quat(
                model_quat,
                self.gazebo_mount_translation,
            )
            camera_quat = _quat_multiply(model_quat, self.gazebo_mount_quat)
            selection_method = f"model_plus_mount:{child}"
        elif self.gazebo_allow_pose_fallback:
            model_pos, model_quat, sec, nanosec, child = _pose_from_transform(
                msg.transforms[0]
            )
            camera_pos = model_pos + _rotate_vector_by_quat(
                model_quat,
                self.gazebo_mount_translation,
            )
            camera_quat = _quat_multiply(model_quat, self.gazebo_mount_quat)
            selection_method = f"fallback_model_plus_mount:{child}"
        else:
            with self.gazebo_pose_lock:
                self.latest_gazebo_pose = None
            self._store_latest_gazebo_pose(None)
            return

        gazebo_pose = {
            "gazebo_model_pos_world": model_pos.copy(),
            "gazebo_model_quat_world": model_quat.copy(),
            "gazebo_camera_pos_world": camera_pos.copy(),
            "gazebo_camera_quat_world": camera_quat.copy(),
            "gazebo_pose_ros_stamp_sec": sec,
            "gazebo_pose_ros_stamp_nanosec": nanosec,
            "gazebo_pose_wall_time": wall_time,
            "gazebo_pose_selection_method": selection_method,
        }
        with self.gazebo_pose_lock:
            self.latest_gazebo_pose = gazebo_pose
        self._store_latest_gazebo_pose(gazebo_pose)

    def _find_model_transform(self, msg: TFMessage):
        for transform in msg.transforms:
            if str(transform.child_frame_id) == self.gazebo_model_name:
                return transform
        return None

    def _find_camera_transform(self, msg: TFMessage):
        exact_names = {
            self.gazebo_camera_link_name,
            f"{self.gazebo_model_name}::{self.gazebo_camera_link_name}",
            f"{self.gazebo_model_name}/{self.gazebo_camera_link_name}",
            f"{self.gazebo_model_name}::link::{self.gazebo_camera_link_name}",
        }
        for transform in msg.transforms:
            child = str(transform.child_frame_id)
            if child in exact_names:
                return transform
        for transform in msg.transforms:
            child = str(transform.child_frame_id)
            if (
                self.gazebo_model_name in child
                and child.endswith(self.gazebo_camera_link_name)
            ):
                return transform
        return None

    def latest_gazebo_pose_snapshot(self):
        with self.gazebo_pose_lock:
            if self.latest_gazebo_pose is None:
                return None
            return {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in self.latest_gazebo_pose.items()
            }

    def _store_latest_gazebo_pose(self, gazebo_pose):
        lock = self._get_lock()
        pose = None
        if isinstance(gazebo_pose, dict):
            pose = {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in gazebo_pose.items()
            }

        if lock is not None:
            with lock:
                self.data["latest_gazebo_pose"] = pose
            return

        self.data["latest_gazebo_pose"] = pose


class RosCameraRX:
    """
    Minimal ROS2 camera receiver that mirrors VisionRX's output.

    Writes:
        shared_data["latest_frame"] = {
            "frame_id": int,
            "image": np.ndarray BGR,
            "shape": tuple,
            "sim_time_ns": int,
            "wall_time": float,
            ...
        }

    So controller.py can use the same:
        frame = data.get("latest_frame")
    """

    def __init__(
        self,
        data,
        camera_topic=None,
        camera_info_topic=None,
        init_rclpy=True,
    ):
        self.data = data
        self.init_rclpy = init_rclpy
        self.is_running = True

        if self.init_rclpy and not rclpy.ok():
            rclpy.init()

        self.node = RosCameraNode(
            data=data,
            camera_topic=camera_topic,
            camera_info_topic=camera_info_topic,
        )

        self.thread = threading.Thread(
            target=self._spin_loop,
            daemon=False,
        )
        self.thread.start()

    def _spin_loop(self):
        while self.is_running and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def get_thread_for_join(self):
        self.is_running = False

        try:
            self.node.destroy_node()
        except Exception:
            pass

        return self.thread

    def shutdown(self):
        self.is_running = False

        try:
            self.node.destroy_node()
        except Exception:
            pass

        if self.init_rclpy and rclpy.ok():
            rclpy.shutdown()
