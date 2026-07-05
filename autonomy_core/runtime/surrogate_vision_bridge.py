"""PX4/Gazebo surrogate camera-to-VADR UDP bridge for Phase 8.5E.

This module is surrogate-only. Importing it does not initialize ROS, import
cv2, open sockets, or interact with the competition runner. Live execution can
read a ROS sensor_msgs/Image topic, JPEG-encode pixels, packetize them with the
VADR `<IHHIIQ` vision header, and send UDP packets to 127.0.0.1:5600 by
default.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from autonomy_core.core.competition_config import RuntimeCompetitionConfig, VADR_TS_002


PHASE_8_5E = "8.5E"
SURROGATE_VISION_LABEL = "PX4/Gazebo surrogate vision bridge only"
DEFAULT_CAMERA_TOPIC = "/camera"
DEFAULT_SEND_HOST = "127.0.0.1"
DEFAULT_MAX_PAYLOAD_SIZE = 1200
DEFAULT_FRAME_ID_START = 1
DEFAULT_SIM_TIME_NS_START = 1_234_567_890

PHASE4B_NOT_SATISFIED = (
    "Phase 4B remains blocked pending real receive-only competition simulator "
    "telemetry evidence."
)
COMPETITION_READINESS_NOT_CLAIMED = (
    "Phase 8.5E is PX4/Gazebo surrogate vision evidence only and is not "
    "competition readiness."
)

FORBIDDEN_CAMERA_TOPIC_MARKERS = (
    "dynamic_pose",
    "ground_truth",
    "link_state",
    "model_state",
    "pose/info",
    "truth",
    "/tf",
    "tf_static",
    "camera_info",
    "depth",
)


class SurrogateVisionBridgeError(RuntimeError):
    """Raised when the surrogate vision bridge would violate Phase 8.5E gates."""


@dataclass(frozen=True)
class SurrogateVisionBridgeConfig:
    camera_topic: str = DEFAULT_CAMERA_TOPIC
    send_host: str = DEFAULT_SEND_HOST
    send_port: int = VADR_TS_002.vision_udp_port
    frames: int = 1
    timeout_s: float = 5.0
    max_payload_size: int = DEFAULT_MAX_PAYLOAD_SIZE
    jpeg_quality: int = 90
    frame_id_start: int = DEFAULT_FRAME_ID_START
    sim_time_ns_start: int = DEFAULT_SIM_TIME_NS_START
    resize_camera_to_competition: bool = False


@dataclass
class SurrogateVisionBridgeSummary:
    surrogate_label: str = SURROGATE_VISION_LABEL
    phase: str = PHASE_8_5E
    status: str = "not_started"
    fail_closed: bool = False
    safety_error: Optional[str] = None
    camera_topic: str = DEFAULT_CAMERA_TOPIC
    send_host: str = DEFAULT_SEND_HOST
    send_port: int = VADR_TS_002.vision_udp_port
    frames_requested: int = 0
    frames_captured: int = 0
    frames_sent: int = 0
    packets_sent: int = 0
    bytes_sent: int = 0
    packetization_errors: int = 0
    resize_applied_count: int = 0
    last_frame_id: Optional[int] = None
    last_sim_time_ns: Optional[int] = None
    image_shape: Optional[tuple[int, ...]] = None
    timed_out: bool = False
    command_sent_count: int = 0
    phase4b_satisfied: bool = False
    phase9_satisfied: bool = False
    competition_readiness_claimed: bool = False
    notes: tuple[str, ...] = (
        PHASE4B_NOT_SATISFIED,
        COMPETITION_READINESS_NOT_CLAIMED,
        "No MAVLink, heartbeat, setpoint, attitude target, position target, "
        "actuator, arm, offboard, reset, or command publication is performed.",
    )
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "surrogate_label": self.surrogate_label,
            "phase": self.phase,
            "status": self.status,
            "fail_closed": self.fail_closed,
            "safety_error": self.safety_error,
            "camera_topic": self.camera_topic,
            "send_host": self.send_host,
            "send_port": self.send_port,
            "frames_requested": self.frames_requested,
            "frames_captured": self.frames_captured,
            "frames_sent": self.frames_sent,
            "packets_sent": self.packets_sent,
            "bytes_sent": self.bytes_sent,
            "packetization_errors": self.packetization_errors,
            "resize_applied_count": self.resize_applied_count,
            "last_frame_id": self.last_frame_id,
            "last_sim_time_ns": self.last_sim_time_ns,
            "image_shape": None if self.image_shape is None else list(self.image_shape),
            "timed_out": self.timed_out,
            "command_sent_count": self.command_sent_count,
            "phase4b_satisfied": self.phase4b_satisfied,
            "phase9_satisfied": self.phase9_satisfied,
            "competition_readiness_claimed": self.competition_readiness_claimed,
            "notes.txt": list(self.notes),
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class PacketizedVisionFrame:
    frame_id: int
    sim_time_ns: int
    packets: tuple[bytes, ...]
    jpeg_size: int


class RosCameraFrameSource:
    """Lazy ROS Image source that reads only camera pixels."""

    def __init__(self, *, topic: str, timeout_s: float):
        self.topic = str(topic)
        self.timeout_s = float(timeout_s)

    def capture_frame(self) -> Any:
        return capture_ros_camera_frame_as_bgr(
            topic=self.topic,
            timeout_s=self.timeout_s,
        )


class UdpVisionPacketSender:
    """Lazy UDP sender for already-packetized VADR vision datagrams."""

    def __init__(
        self,
        *,
        send_host: str,
        send_port: int,
        socket_factory: Optional[Callable[[], Any]] = None,
    ):
        self.send_host = str(send_host)
        self.send_port = int(send_port)
        self.socket_factory = socket_factory

    def send_packets(self, packets: Sequence[bytes]) -> tuple[int, int]:
        sock = (
            self.socket_factory()
            if self.socket_factory is not None
            else _open_udp_socket()
        )
        close = getattr(sock, "close", None)
        try:
            packet_count = 0
            byte_count = 0
            for packet in packets:
                packet_bytes = bytes(packet)
                sock.sendto(packet_bytes, (self.send_host, self.send_port))
                packet_count += 1
                byte_count += len(packet_bytes)
            return packet_count, byte_count
        finally:
            if callable(close):
                close()


def run_surrogate_vision_bridge(
    config: SurrogateVisionBridgeConfig = SurrogateVisionBridgeConfig(),
    *,
    frame_source: Any = None,
    packet_sender: Any = None,
    jpeg_encoder: Optional[Callable[..., bytes]] = None,
    packetizer: Optional[Callable[..., Any]] = None,
    resizer: Optional[Callable[[Any, int, int], Any]] = None,
) -> SurrogateVisionBridgeSummary:
    """Capture frames, packetize them as VADR vision UDP, and send them.

    Tests should inject `frame_source`, `packet_sender`, and `jpeg_encoder` to
    keep execution deterministic and socket-free.
    """

    _validate_config(config)
    source = frame_source or RosCameraFrameSource(
        topic=config.camera_topic,
        timeout_s=config.timeout_s,
    )
    sender = packet_sender or UdpVisionPacketSender(
        send_host=config.send_host,
        send_port=config.send_port,
    )
    summary = SurrogateVisionBridgeSummary(
        status="running",
        camera_topic=config.camera_topic,
        send_host=config.send_host,
        send_port=config.send_port,
        frames_requested=int(config.frames),
    )

    for frame_index in range(int(config.frames)):
        frame = _capture_from_source(source)
        summary.frames_captured += 1

        frame, resized = validate_or_resize_frame_for_competition(
            frame,
            resize_enabled=config.resize_camera_to_competition,
            resizer=resizer,
        )
        if resized:
            summary.resize_applied_count += 1
        summary.image_shape = _frame_shape(frame)

        frame_id = int(config.frame_id_start) + frame_index
        sim_time_ns = _sim_time_ns_for_index(config, frame_index)
        try:
            packetized = packetize_frame_for_vadr(
                frame,
                frame_id=frame_id,
                sim_time_ns=sim_time_ns,
                max_payload_size=config.max_payload_size,
                jpeg_quality=config.jpeg_quality,
                jpeg_encoder=jpeg_encoder,
                packetizer=packetizer,
            )
        except Exception as exc:
            summary.packetization_errors += 1
            raise SurrogateVisionBridgeError(
                f"failed to packetize surrogate camera frame: {exc}"
            ) from exc

        packets_sent, bytes_sent = _send_packets(sender, packetized.packets)
        summary.frames_sent += 1
        summary.packets_sent += packets_sent
        summary.bytes_sent += bytes_sent
        summary.last_frame_id = packetized.frame_id
        summary.last_sim_time_ns = packetized.sim_time_ns

    summary.status = "surrogate_vision_bridge_complete"
    return summary


def fail_closed_summary(
    *,
    config: SurrogateVisionBridgeConfig,
    error: str,
) -> SurrogateVisionBridgeSummary:
    return SurrogateVisionBridgeSummary(
        status="fail_closed",
        fail_closed=True,
        safety_error=str(error),
        camera_topic=config.camera_topic,
        send_host=config.send_host,
        send_port=config.send_port,
        frames_requested=int(config.frames),
        errors=[str(error)],
    )


def validate_or_resize_frame_for_competition(
    frame: Any,
    *,
    resize_enabled: bool,
    config: RuntimeCompetitionConfig = VADR_TS_002,
    resizer: Optional[Callable[[Any, int, int], Any]] = None,
) -> tuple[Any, bool]:
    """Validate or explicitly resize camera pixels to official VADR resolution."""

    shape = _frame_shape(frame)
    expected_hw = (int(config.camera_height_px), int(config.camera_width_px))
    if len(shape) < 2:
        raise SurrogateVisionBridgeError("camera frame does not expose image shape")
    if shape[:2] == expected_hw:
        return frame, False
    if not resize_enabled:
        raise SurrogateVisionBridgeError(
            "camera frame is "
            f"{shape[1]}x{shape[0]}, but the competition vision path requires "
            f"{config.camera_width_px}x{config.camera_height_px}; configure the "
            "camera to the official resolution or rerun this surrogate bridge "
            "with --resize-camera-to-competition"
        )

    active_resizer = resizer or _resize_with_cv2
    resized = active_resizer(frame, int(config.camera_width_px), int(config.camera_height_px))
    resized_shape = _frame_shape(resized)
    if resized_shape[:2] != expected_hw:
        raise SurrogateVisionBridgeError(
            "resized camera frame does not match official VADR resolution"
        )
    return resized, True


def packetize_frame_for_vadr(
    frame: Any,
    *,
    frame_id: int,
    sim_time_ns: int,
    max_payload_size: int = DEFAULT_MAX_PAYLOAD_SIZE,
    jpeg_quality: int = 90,
    config: RuntimeCompetitionConfig = VADR_TS_002,
    jpeg_encoder: Optional[Callable[..., bytes]] = None,
    packetizer: Optional[Callable[..., Any]] = None,
) -> PacketizedVisionFrame:
    """JPEG-encode and VADR-packetize one already-sized camera frame."""

    active_encoder = jpeg_encoder or _load_jpeg_encoder()
    active_packetizer = packetizer or _load_packetizer()
    jpeg_bytes = active_encoder(frame, quality=int(jpeg_quality))
    packetized = active_packetizer(
        bytes(jpeg_bytes),
        frame_id=int(frame_id),
        sim_time_ns=int(sim_time_ns),
        max_payload_size=int(max_payload_size),
        config=config,
    )
    packets = tuple(bytes(packet) for packet in packetized.packets)
    return PacketizedVisionFrame(
        frame_id=int(frame_id),
        sim_time_ns=int(sim_time_ns),
        packets=packets,
        jpeg_size=int(packetized.jpeg_size),
    )


def capture_ros_camera_frame_as_bgr(*, topic: str, timeout_s: float) -> Any:
    """Capture one ROS Image frame lazily, without subscribing to truth topics."""

    _assert_camera_topic_safe(topic)
    try:
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.node import Node
        from sensor_msgs.msg import Image
    except ModuleNotFoundError as exc:
        raise SurrogateVisionBridgeError(
            "ROS camera bridge execution requires rclpy, sensor_msgs, and "
            "cv_bridge; these imports remain lazy and are only used when the "
            "bridge is explicitly run."
        ) from exc

    bridge = CvBridge()
    frame_box: dict[str, Any] = {}

    class _OneFrameNode(Node):
        def __init__(self) -> None:
            super().__init__("surrogate_vision_bridge_one_frame")
            self.create_subscription(Image, topic, self._callback, 1)

        def _callback(self, message: Any) -> None:
            if "frame" not in frame_box:
                frame_box["frame"] = bridge.imgmsg_to_cv2(
                    message,
                    desired_encoding="bgr8",
                )

    started_here = not rclpy.ok()
    if started_here:
        rclpy.init(args=None)
    node = _OneFrameNode()
    deadline = time.time() + float(timeout_s)
    try:
        while "frame" not in frame_box and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.destroy_node()
        if started_here:
            rclpy.shutdown()

    frame = frame_box.get("frame")
    if frame is None:
        raise SurrogateVisionBridgeError(
            f"timed out waiting for ROS camera frame on {topic!r}"
        )
    return frame


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 8.5E PX4/Gazebo surrogate vision bridge. Reads ROS camera "
            "pixels only, emits VADR JPEG UDP packets to 127.0.0.1:5600 by "
            "default, and sends no MAVLink or commands."
        )
    )
    parser.add_argument("--camera-topic", default=DEFAULT_CAMERA_TOPIC)
    parser.add_argument("--send-host", default=DEFAULT_SEND_HOST)
    parser.add_argument("--send-port", type=int, default=VADR_TS_002.vision_udp_port)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--max-payload-size", type=int, default=DEFAULT_MAX_PAYLOAD_SIZE)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--frame-id-start", type=int, default=DEFAULT_FRAME_ID_START)
    parser.add_argument("--sim-time-ns-start", type=int, default=DEFAULT_SIM_TIME_NS_START)
    parser.add_argument(
        "--resize-camera-to-competition",
        action="store_true",
        help=(
            "surrogate-only resize for non-640x360 cameras; not needed when "
            "Gazebo publishes the official VADR resolution"
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = SurrogateVisionBridgeConfig(
        camera_topic=str(args.camera_topic),
        send_host=str(args.send_host),
        send_port=int(args.send_port),
        frames=int(args.frames),
        timeout_s=float(args.timeout_s),
        max_payload_size=int(args.max_payload_size),
        jpeg_quality=int(args.jpeg_quality),
        frame_id_start=int(args.frame_id_start),
        sim_time_ns_start=int(args.sim_time_ns_start),
        resize_camera_to_competition=bool(args.resize_camera_to_competition),
    )
    try:
        summary = run_surrogate_vision_bridge(config)
        exit_code = 0
    except SurrogateVisionBridgeError as exc:
        summary = fail_closed_summary(config=config, error=str(exc))
        exit_code = 2

    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return exit_code


def _validate_config(config: SurrogateVisionBridgeConfig) -> None:
    if int(config.send_port) <= 0 or int(config.send_port) > 65535:
        raise SurrogateVisionBridgeError("send_port must be in [1, 65535]")
    if int(config.frames) <= 0:
        raise SurrogateVisionBridgeError("frames must be positive")
    if float(config.timeout_s) <= 0.0:
        raise SurrogateVisionBridgeError("timeout_s must be positive")
    if int(config.max_payload_size) <= 0:
        raise SurrogateVisionBridgeError("max_payload_size must be positive")
    if int(config.jpeg_quality) < 1 or int(config.jpeg_quality) > 100:
        raise SurrogateVisionBridgeError("jpeg_quality must be in [1, 100]")
    _assert_camera_topic_safe(config.camera_topic)


def _assert_camera_topic_safe(topic: str) -> None:
    normalized = str(topic).lower()
    for marker in FORBIDDEN_CAMERA_TOPIC_MARKERS:
        if marker in normalized:
            raise SurrogateVisionBridgeError(
                f"refusing to subscribe to non-image or truth-like topic {topic!r}"
            )


def _capture_from_source(source: Any) -> Any:
    capture = getattr(source, "capture_frame", None)
    if callable(capture):
        return capture()
    if callable(source):
        return source()
    raise SurrogateVisionBridgeError("frame_source must be callable or expose capture_frame()")


def _send_packets(sender: Any, packets: Sequence[bytes]) -> tuple[int, int]:
    send_packets = getattr(sender, "send_packets", None)
    if callable(send_packets):
        return send_packets(packets)
    if callable(sender):
        return sender(packets)
    raise SurrogateVisionBridgeError("packet_sender must be callable or expose send_packets()")


def _sim_time_ns_for_index(config: SurrogateVisionBridgeConfig, frame_index: int) -> int:
    period_ns = int(round(VADR_TS_002.vision_period_s * 1_000_000_000))
    return int(config.sim_time_ns_start) + int(frame_index) * period_ns


def _frame_shape(frame: Any) -> tuple[int, ...]:
    shape = getattr(frame, "shape", None)
    if shape is None:
        return ()
    return tuple(int(value) for value in shape)


def _resize_with_cv2(frame: Any, width: int, height: int) -> Any:
    cv2 = _import_cv2()
    return cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA)


def _load_jpeg_encoder() -> Callable[..., bytes]:
    from autonomy_core.tools.competition_vision_udp_loopback import encode_jpeg

    return encode_jpeg


def _load_packetizer() -> Callable[..., Any]:
    from autonomy_core.tools.competition_vision_udp_loopback import packetize_jpeg_bytes

    return packetize_jpeg_bytes


def _import_cv2() -> Any:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise SurrogateVisionBridgeError(
            "cv2 is required only when live bridge execution needs JPEG encode "
            "or explicit surrogate resizing"
        ) from exc
    return cv2


def _open_udp_socket() -> Any:
    import socket

    return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPETITION_READINESS_NOT_CLAIMED",
    "DEFAULT_CAMERA_TOPIC",
    "DEFAULT_FRAME_ID_START",
    "DEFAULT_MAX_PAYLOAD_SIZE",
    "DEFAULT_SEND_HOST",
    "DEFAULT_SIM_TIME_NS_START",
    "FORBIDDEN_CAMERA_TOPIC_MARKERS",
    "PHASE4B_NOT_SATISFIED",
    "PHASE_8_5E",
    "PacketizedVisionFrame",
    "RosCameraFrameSource",
    "SURROGATE_VISION_LABEL",
    "SurrogateVisionBridgeConfig",
    "SurrogateVisionBridgeError",
    "SurrogateVisionBridgeSummary",
    "UdpVisionPacketSender",
    "build_arg_parser",
    "capture_ros_camera_frame_as_bgr",
    "fail_closed_summary",
    "main",
    "packetize_frame_for_vadr",
    "run_surrogate_vision_bridge",
    "validate_or_resize_frame_for_competition",
]
