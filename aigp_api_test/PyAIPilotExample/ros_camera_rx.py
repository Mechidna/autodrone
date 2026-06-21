import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


ROS_CAMERA_TOPIC = "/camera"
ROS_CAMERA_INFO_TOPIC = "/camera_info"


class RosCameraNode(Node):
    def __init__(self, data, camera_topic=ROS_CAMERA_TOPIC, camera_info_topic=ROS_CAMERA_INFO_TOPIC):
        super().__init__("ros_camera_rx_node")

        self.data = data
        self.bridge = CvBridge()
        self.frame_id_counter = 0

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

        self.get_logger().info(f"Listening for ROS camera frames on {camera_topic}")

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

        frame_data = {
            "frame_id": self.frame_id_counter,
            "image": img,
            "shape": img.shape,
            "sim_time_ns": sim_time_ns,
            "wall_time": time.time(),

            # ROS-specific metadata, harmless for controller/autonomy_adapter
            "ros_stamp_sec": stamp_sec,
            "ros_stamp_nanosec": stamp_nanosec,
            "ros_frame_id": msg.header.frame_id,
            "source": "ros2_camera",
        }

        self._store_frame(frame_data)


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
        camera_topic=ROS_CAMERA_TOPIC,
        camera_info_topic=ROS_CAMERA_INFO_TOPIC,
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