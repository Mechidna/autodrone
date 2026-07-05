#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage


DEFAULT_WORLD = (
    os.environ.get("PX4_GZ_WORLD")
    or os.environ.get("WORLD")
    or "gate_test_1500mm_blue_random"
)
DEFAULT_DYNAMIC_POSE_TOPIC = f"/world/{DEFAULT_WORLD}/dynamic_pose/info"
DEFAULT_CAPTURE_ROOT = "~/datasets/gazebo_gate_capture_racer"
DEFAULT_CAMERA_MOUNT_RPY = (0.0, -0.3490658503988659, 0.0)
DEFAULT_WORLD_SDF = (
    "/home/paolo/PX4-Autopilot/PX4-Autopilot/Tools/simulation/gz/worlds/"
    f"{DEFAULT_WORLD}.sdf"
)


def _parse_vec3(text: str, *, name: str) -> tuple[float, float, float]:
    parts = re.split(r"[,\s]+", str(text).strip())
    parts = [part for part in parts if part]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"{name} must contain exactly 3 numbers.")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _normalize_quat(quat_xyzw) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=float).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return quat / norm


def _quat_multiply(lhs_xyzw, rhs_xyzw) -> np.ndarray:
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


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
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


def _quat_to_rpy(quat_xyzw) -> np.ndarray:
    x, y, z, w = _normalize_quat(quat_xyzw)
    roll = math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    sin_pitch = 2.0 * (w * y - z * x)
    sin_pitch = max(-1.0, min(1.0, sin_pitch))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return np.array([roll, pitch, yaw], dtype=float)


def _rotate_vector_by_quat(quat_xyzw, vector) -> np.ndarray:
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
    return (
        np.array([translation.x, translation.y, translation.z], dtype=float),
        _normalize_quat([rotation.x, rotation.y, rotation.z, rotation.w]),
        int(transform.header.stamp.sec),
        int(transform.header.stamp.nanosec),
        str(transform.child_frame_id),
    )


def _json_list(value):
    return np.asarray(value, dtype=float).tolist()


def _image_hash_md5(frame: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(frame).tobytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    slug = slug.strip("._-")
    return slug or "run"


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


class GazeboYoloPoseCaptureNode(Node):
    def __init__(self, args):
        super().__init__("gazebo_yolo_pose_capture")
        self.args = args
        self.bridge = CvBridge()
        self.capture_period_s = (
            1.0 / float(args.capture_hz)
            if float(args.capture_hz) > 0.0
            else 0.0
        )
        self.next_capture_time_s = 0.0
        self.last_saved_stamp = None
        self.saved_count = 0
        self.last_warning_s: dict[str, float] = {}

        self.dataset_root = Path(os.path.expanduser(args.capture_root)).resolve()
        self.root_dir = self._prepare_run_directory()
        self.images_dir = self.root_dir / "images"
        self.metadata_dir = self.root_dir / "metadata"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.frame_idx = self._next_frame_index() - 1
        self.manifest_path = self.root_dir / "capture_manifest.json"
        self.manifest_relpath = _relative_to_root(self.manifest_path, self.root_dir)
        self.world_sdf_original_path = None
        self.world_sdf_original_sha256 = None
        self.world_sdf_snapshot_path = None
        self.world_sdf_snapshot_sha256 = None
        self.manifest_sha256 = None

        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_info_stamp = None
        self.latest_pose = None

        self.model_name = str(args.model_name)
        self.camera_link_name = str(args.camera_link_name)
        self.mount_translation = np.asarray(args.camera_mount_xyz, dtype=float).reshape(3)
        self.mount_quat = _quat_from_rpy(*args.camera_mount_rpy)
        self._write_capture_manifest()

        self.create_subscription(
            Image,
            args.camera_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            args.camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            TFMessage,
            args.dynamic_pose_topic,
            self.dynamic_pose_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            "capture run=%s hz=%.2f camera=%s camera_info=%s dynamic_pose=%s model=%s"
            % (
                self.root_dir,
                float(args.capture_hz),
                args.camera_topic,
                args.camera_info_topic,
                args.dynamic_pose_topic,
                self.model_name,
            )
        )

    def _prepare_run_directory(self) -> Path:
        if bool(self.args.flat_capture_layout):
            self.dataset_root.mkdir(parents=True, exist_ok=True)
            self.run_id = str(self.args.run_id or self.dataset_root.name)
            return self.dataset_root

        run_id = self.args.run_id
        if run_id is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            run_id = f"{timestamp}_{_safe_slug(self.args.world_name)}"
        self.run_id = _safe_slug(run_id)
        run_dir = self.dataset_root / "runs" / self.run_id
        suffix = 1
        while run_dir.exists() and any(run_dir.iterdir()):
            run_dir = self.dataset_root / "runs" / f"{self.run_id}_{suffix:02d}"
            suffix += 1
        self.run_id = run_dir.name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _write_capture_manifest(self) -> None:
        if not bool(self.args.no_world_sdf_snapshot):
            self._snapshot_world_sdf()

        manifest = {
            "schema": "gazebo_yolo_pose_capture.v1",
            "run_id": self.run_id,
            "created_wall_time": time.time(),
            "created_local_time": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
            "dataset_root": str(self.dataset_root),
            "run_dir": str(self.root_dir),
            "world_name": str(self.args.world_name),
            "world_sdf_original_path": (
                None if self.world_sdf_original_path is None else str(self.world_sdf_original_path)
            ),
            "world_sdf_original_sha256": self.world_sdf_original_sha256,
            "world_sdf_snapshot": (
                None
                if self.world_sdf_snapshot_path is None
                else _relative_to_root(self.world_sdf_snapshot_path, self.root_dir)
            ),
            "world_sdf_sha256": self.world_sdf_snapshot_sha256,
            "camera_topic": str(self.args.camera_topic),
            "camera_info_topic": str(self.args.camera_info_topic),
            "dynamic_pose_topic": str(self.args.dynamic_pose_topic),
            "model_name": self.model_name,
            "camera_link_name": self.camera_link_name,
            "camera_mount_xyz": _json_list(self.mount_translation),
            "camera_mount_rpy": list(float(v) for v in self.args.camera_mount_rpy),
            "capture_hz": float(self.args.capture_hz),
            "allow_pose_fallback": bool(self.args.allow_pose_fallback),
            "allow_missing_pose": bool(self.args.allow_missing_pose),
            "flat_capture_layout": bool(self.args.flat_capture_layout),
        }
        with self.manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
        self.manifest_sha256 = _sha256_file(self.manifest_path)

    def _snapshot_world_sdf(self) -> None:
        world_sdf = Path(os.path.expanduser(str(self.args.world_sdf))).resolve()
        if not world_sdf.exists():
            if bool(self.args.allow_missing_world_sdf):
                self.get_logger().warning(f"world SDF not found; manifest will omit snapshot: {world_sdf}")
                return
            raise RuntimeError(f"World SDF not found: {world_sdf}")

        snapshot_path = self.root_dir / "world.sdf"
        shutil.copy2(world_sdf, snapshot_path)
        self.world_sdf_original_path = world_sdf
        self.world_sdf_original_sha256 = _sha256_file(world_sdf)
        self.world_sdf_snapshot_path = snapshot_path
        self.world_sdf_snapshot_sha256 = _sha256_file(snapshot_path)

    def _next_frame_index(self) -> int:
        max_idx = 0
        for path in self.metadata_dir.glob("frame_*.json"):
            match = re.match(r"^frame_(\d+)\.json$", path.name)
            if match is None:
                continue
            max_idx = max(max_idx, int(match.group(1)))
        return max_idx + 1

    def _warn_periodic(self, key: str, message: str, period_s: float = 2.0) -> None:
        now = time.time()
        if now - float(self.last_warning_s.get(key, 0.0)) < period_s:
            return
        self.last_warning_s[key] = now
        self.get_logger().warning(message)

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_matrix = np.asarray(msg.k, dtype=float).reshape(3, 3)
        if len(msg.d) > 0:
            self.dist_coeffs = np.asarray(msg.d, dtype=float).reshape(-1, 1)
        else:
            self.dist_coeffs = np.zeros((5, 1), dtype=float)
        self.camera_info_stamp = (
            int(msg.header.stamp.sec),
            int(msg.header.stamp.nanosec),
        )

    def dynamic_pose_callback(self, msg: TFMessage):
        if len(msg.transforms) == 0:
            self.latest_pose = None
            return

        model_tf = self._find_model_transform(msg)
        camera_tf = self._find_camera_transform(msg)
        wall_time = time.time()

        if camera_tf is not None:
            camera_pos, camera_quat, sec, nanosec, child = _pose_from_transform(camera_tf)
            if model_tf is not None:
                model_pos, model_quat, _m_sec, _m_nanosec, model_child = _pose_from_transform(model_tf)
            else:
                model_pos, model_quat, model_child = camera_pos.copy(), camera_quat.copy(), child
            selection_method = f"camera_link:{child}"
        elif model_tf is not None:
            model_pos, model_quat, sec, nanosec, model_child = _pose_from_transform(model_tf)
            camera_pos = model_pos + _rotate_vector_by_quat(
                model_quat,
                self.mount_translation,
            )
            camera_quat = _quat_multiply(model_quat, self.mount_quat)
            selection_method = f"model_plus_mount:{model_child}"
        elif bool(self.args.allow_pose_fallback):
            fallback_tf = msg.transforms[0]
            model_pos, model_quat, sec, nanosec, model_child = _pose_from_transform(fallback_tf)
            camera_pos = model_pos + _rotate_vector_by_quat(
                model_quat,
                self.mount_translation,
            )
            camera_quat = _quat_multiply(model_quat, self.mount_quat)
            selection_method = f"fallback_model_plus_mount:{model_child}"
        else:
            self.latest_pose = None
            self._warn_periodic(
                "missing_pose",
                (
                    "dynamic_pose did not contain model/camera transform for "
                    f"{self.model_name}; not saving frames"
                ),
            )
            return

        self.latest_pose = {
            "gazebo_model_pos_world": model_pos,
            "gazebo_model_quat_world": model_quat,
            "gazebo_camera_pos_world": camera_pos,
            "gazebo_camera_quat_world": camera_quat,
            "gazebo_pose_ros_stamp_sec": sec,
            "gazebo_pose_ros_stamp_nanosec": nanosec,
            "gazebo_pose_wall_time": wall_time,
            "gazebo_pose_selection_method": selection_method,
        }

    def _find_model_transform(self, msg: TFMessage):
        for transform in msg.transforms:
            if str(transform.child_frame_id) == self.model_name:
                return transform
        return None

    def _find_camera_transform(self, msg: TFMessage):
        exact_names = {
            self.camera_link_name,
            f"{self.model_name}::{self.camera_link_name}",
            f"{self.model_name}/{self.camera_link_name}",
            f"{self.model_name}::link::{self.camera_link_name}",
        }
        for transform in msg.transforms:
            child = str(transform.child_frame_id)
            if child in exact_names:
                return transform
        for transform in msg.transforms:
            child = str(transform.child_frame_id)
            if self.model_name in child and child.endswith(self.camera_link_name):
                return transform
        return None

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"failed to convert ROS image: {exc}")
            return

        self._maybe_save_frame(frame, msg)

    def _maybe_save_frame(self, frame: np.ndarray, msg: Image):
        if self.camera_matrix is None or self.dist_coeffs is None:
            self._warn_periodic("missing_camera_info", "waiting for camera_info")
            return

        pose = self.latest_pose
        if pose is None and not bool(self.args.allow_missing_pose):
            self._warn_periodic("missing_pose_for_image", "waiting for Gazebo pose")
            return

        stamp = (int(msg.header.stamp.sec), int(msg.header.stamp.nanosec))
        if stamp == self.last_saved_stamp:
            return

        now = time.time()
        if now < self.next_capture_time_s:
            return
        if self.capture_period_s > 0.0:
            self.next_capture_time_s = now + self.capture_period_s

        self.frame_idx += 1
        image_filename = f"frame_{self.frame_idx:06d}.jpg"
        metadata_filename = f"frame_{self.frame_idx:06d}.json"
        image_path = self.images_dir / image_filename
        metadata_path = self.metadata_dir / metadata_filename

        if not cv2.imwrite(str(image_path), frame):
            self.get_logger().error(f"failed to write image: {image_path}")
            self.frame_idx -= 1
            return

        metadata = self._metadata(
            frame,
            image_filename=image_filename,
            stamp=stamp,
            wall_time=now,
            pose=pose,
        )
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self.last_saved_stamp = stamp
        self.saved_count += 1
        if self.saved_count == 1 or self.saved_count % max(1, int(self.args.print_every)) == 0:
            self.get_logger().info(
                f"saved={self.saved_count} image={image_path} pose={metadata['gazebo_pose_selection_method']}"
            )

        if int(self.args.max_frames) > 0 and self.saved_count >= int(self.args.max_frames):
            self.get_logger().info(f"max_frames reached: {self.saved_count}")
            rclpy.shutdown()

    def _metadata(self, frame, *, image_filename: str, stamp, wall_time: float, pose):
        image_height, image_width = frame.shape[:2]
        model_pos = None if pose is None else pose["gazebo_model_pos_world"]
        model_quat = None if pose is None else pose["gazebo_model_quat_world"]
        model_rpy = (
            np.zeros(3, dtype=float)
            if model_quat is None
            else _quat_to_rpy(model_quat)
        )

        # Legacy fallback keys expected by autolabel_gazebo_yolo_pose.py when no
        # gazebo_camera_* pose is available. The current capture path should use
        # gazebo_camera_* and therefore not rely on these.
        drone_pos = (
            np.zeros(3, dtype=float)
            if model_pos is None
            else np.array([model_pos[1], model_pos[0], model_pos[2]], dtype=float)
        )
        drone_rpy = np.array([model_rpy[0], model_rpy[1], -model_rpy[2]], dtype=float)

        gazebo_pose_age_s = (
            None
            if pose is None
            else wall_time - float(pose["gazebo_pose_wall_time"])
        )
        return {
            "image_filename": image_filename,
            "run_id": self.run_id,
            "capture_manifest": self.manifest_relpath,
            "capture_manifest_sha256": self.manifest_sha256,
            "world_name": str(self.args.world_name),
            "world_sdf_snapshot": (
                None
                if self.world_sdf_snapshot_path is None
                else _relative_to_root(self.world_sdf_snapshot_path, self.root_dir)
            ),
            "world_sdf_sha256": self.world_sdf_snapshot_sha256,
            "timestamp": float(wall_time),
            "metadata_write_time": float(wall_time),
            "ros_image_stamp_sec": int(stamp[0]),
            "ros_image_stamp_nanosec": int(stamp[1]),
            "drone_pos": _json_list(drone_pos),
            "drone_rpy_rad": _json_list(drone_rpy),
            "camera_matrix": _json_list(self.camera_matrix),
            "dist_coeffs": _json_list(self.dist_coeffs),
            "image_width": int(image_width),
            "image_height": int(image_height),
            "image_hash_md5": _image_hash_md5(frame),
            "gazebo_model_pos_world": (
                None if pose is None else _json_list(pose["gazebo_model_pos_world"])
            ),
            "gazebo_model_quat_world": (
                None if pose is None else _json_list(pose["gazebo_model_quat_world"])
            ),
            "gazebo_camera_pos_world": (
                None if pose is None else _json_list(pose["gazebo_camera_pos_world"])
            ),
            "gazebo_camera_quat_world": (
                None if pose is None else _json_list(pose["gazebo_camera_quat_world"])
            ),
            "gazebo_pose_ros_stamp_sec": (
                None if pose is None else int(pose["gazebo_pose_ros_stamp_sec"])
            ),
            "gazebo_pose_ros_stamp_nanosec": (
                None if pose is None else int(pose["gazebo_pose_ros_stamp_nanosec"])
            ),
            "gazebo_pose_wall_time": (
                None if pose is None else float(pose["gazebo_pose_wall_time"])
            ),
            "gazebo_pose_selection_method": (
                None if pose is None else str(pose["gazebo_pose_selection_method"])
            ),
            "pose_source": None if pose is None else f"gazebo_dynamic_pose_{self.model_name}",
            "gazebo_pose_age_s": gazebo_pose_age_s,
        }

    def finalize(self):
        if self.world_sdf_original_path is None or not self.world_sdf_original_path.exists():
            return
        current_sha256 = _sha256_file(self.world_sdf_original_path)
        if current_sha256 != self.world_sdf_original_sha256:
            self.get_logger().warning(
                "original world SDF changed during capture; labels should use this run's world.sdf snapshot"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Capture ROS2 camera frames plus Gazebo dynamic pose metadata for "
            "autolabel_gazebo_yolo_pose.py."
        )
    )
    parser.add_argument("--capture-root", default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--world-name", default=DEFAULT_WORLD)
    parser.add_argument("--world-sdf", default=DEFAULT_WORLD_SDF)
    parser.add_argument(
        "--flat-capture-layout",
        action="store_true",
        help="Use capture-root/images and capture-root/metadata instead of creating capture-root/runs/<run-id>.",
    )
    parser.add_argument(
        "--no-world-sdf-snapshot",
        action="store_true",
        help="Do not copy the active world SDF into the run directory.",
    )
    parser.add_argument(
        "--allow-missing-world-sdf",
        action="store_true",
        help="Start capture even if --world-sdf is missing. Not recommended for autolabeling.",
    )
    parser.add_argument("--capture-hz", type=float, default=5.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--camera-topic", default="/camera")
    parser.add_argument("--camera-info-topic", default="/camera_info")
    parser.add_argument("--dynamic-pose-topic", default=DEFAULT_DYNAMIC_POSE_TOPIC)
    parser.add_argument("--model-name", default="racer_mono_cam_0")
    parser.add_argument("--camera-link-name", default="camera_link")
    parser.add_argument(
        "--camera-mount-xyz",
        type=lambda text: _parse_vec3(text, name="--camera-mount-xyz"),
        default=(0.0, 0.0, 0.0),
        help="Camera pose translation relative to model pose, used if camera_link pose is absent.",
    )
    parser.add_argument(
        "--camera-mount-rpy",
        type=lambda text: _parse_vec3(text, name="--camera-mount-rpy"),
        default=DEFAULT_CAMERA_MOUNT_RPY,
        help="Camera mount RPY relative to model pose, used if camera_link pose is absent.",
    )
    parser.add_argument(
        "--allow-pose-fallback",
        action="store_true",
        help="If the target model is absent, use the first dynamic-pose transform plus mount.",
    )
    parser.add_argument(
        "--allow-missing-pose",
        action="store_true",
        help="Save frames even without Gazebo pose metadata. Not recommended for autolabeling.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = None
    try:
        node = GazeboYoloPoseCaptureNode(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node is not None:
                node.finalize()
                node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
