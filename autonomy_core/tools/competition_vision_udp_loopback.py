"""Local UDP loopback smoke tool for the competition image adapter.

Phase 8.5C only validates the VADR vision packet path:
mock image -> JPEG -> UDP packets -> CompetitionImageAdapter. Importing this
module does not open sockets or import cv2.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from autonomy_core.core.competition_config import RuntimeCompetitionConfig, VADR_TS_002
from autonomy_core.perception.competition_image_adapter import (
    CompetitionCameraFrame,
    CompetitionImageAdapter,
    CompetitionImageAdapterError,
    pack_vision_packet,
    parse_vision_packet_header,
)


DEFAULT_BIND_HOST = "0.0.0.0"
DEFAULT_SEND_HOST = "127.0.0.1"
DEFAULT_FRAME_ID = 1
DEFAULT_SIM_TIME_NS = 1_234_567_890
DEFAULT_MAX_PAYLOAD_SIZE = 1200


class VisionUdpLoopbackError(RuntimeError):
    """Raised when the local UDP vision loopback smoke path cannot complete."""


@dataclass(frozen=True)
class PacketizedJpegFrame:
    """A generated JPEG plus the VADR packets that carry it."""

    frame_id: int
    sim_time_ns: int
    jpeg_bytes: bytes
    packets: tuple[bytes, ...]

    @property
    def jpeg_size(self) -> int:
        return len(self.jpeg_bytes)

    @property
    def packet_count(self) -> int:
        return len(self.packets)


@dataclass
class VisionUdpLoopbackSummary:
    """Compact manual-smoke summary for Phase 8.5C."""

    bind_host: str
    send_host: str
    port: int
    packets_sent: int = 0
    packets_received: int = 0
    frames_completed: int = 0
    decode_failures: int = 0
    packets_rejected: int = 0
    duplicate_chunks: int = 0
    incomplete_frames_dropped: int = 0
    stale_frames_dropped: int = 0
    image_shape: Optional[tuple[int, ...]] = None
    image_dtype: Optional[str] = None
    image_stamp_sec: Optional[int] = None
    image_stamp_nanosec: Optional[int] = None
    camera_matrix: Optional[list[list[float]]] = None
    dist_coeffs: Optional[list[float]] = None
    gazebo_pose: Any = None
    image_pose_snapshot: Any = None
    mock_image_path: Optional[str] = None
    decoded_image_path: Optional[str] = None
    timed_out: bool = False
    errors: list[str] = field(default_factory=list)

    def record_frame(self, frame: CompetitionCameraFrame) -> None:
        self.frames_completed += 1
        self.image_shape = tuple(int(value) for value in frame.frame.shape)
        self.image_dtype = str(frame.frame.dtype)
        self.image_stamp_sec = int(frame.image_stamp_sec)
        self.image_stamp_nanosec = int(frame.image_stamp_nanosec)
        self.camera_matrix = [
            [float(value) for value in row] for row in frame.camera_matrix.tolist()
        ]
        self.dist_coeffs = [float(value) for value in np.ravel(frame.dist_coeffs)]
        self.gazebo_pose = frame.gazebo_pose
        self.image_pose_snapshot = frame.image_pose_snapshot

    def record_adapter_stats(self, adapter: CompetitionImageAdapter) -> None:
        stats = adapter.stats
        self.decode_failures = int(stats.decode_failures)
        self.packets_rejected = int(stats.packets_rejected)
        self.duplicate_chunks = int(stats.duplicate_chunks)
        self.incomplete_frames_dropped = int(stats.incomplete_frames_dropped)
        self.stale_frames_dropped = int(stats.stale_frames_dropped)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bind_host": self.bind_host,
            "send_host": self.send_host,
            "port": self.port,
            "packets_sent": self.packets_sent,
            "packets_received": self.packets_received,
            "frames_completed": self.frames_completed,
            "decode_failures": self.decode_failures,
            "packets_rejected": self.packets_rejected,
            "duplicate_chunks": self.duplicate_chunks,
            "incomplete_frames_dropped": self.incomplete_frames_dropped,
            "stale_frames_dropped": self.stale_frames_dropped,
            "image_shape": (
                None if self.image_shape is None else list(self.image_shape)
            ),
            "image_dtype": self.image_dtype,
            "image_stamp_sec": self.image_stamp_sec,
            "image_stamp_nanosec": self.image_stamp_nanosec,
            "camera_matrix": self.camera_matrix,
            "dist_coeffs": self.dist_coeffs,
            "gazebo_pose": self.gazebo_pose,
            "image_pose_snapshot": self.image_pose_snapshot,
            "mock_image_path": self.mock_image_path,
            "decoded_image_path": self.decoded_image_path,
            "timed_out": self.timed_out,
            "errors": list(self.errors),
        }


def build_mock_image(
    *,
    config: RuntimeCompetitionConfig = VADR_TS_002,
) -> np.ndarray:
    """Build a deterministic nonuniform BGR image at official resolution."""

    height = int(config.camera_height_px)
    width = int(config.camera_width_px)
    x = np.arange(width, dtype=np.uint16)
    y = np.arange(height, dtype=np.uint16).reshape(height, 1)

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = (x % 256).astype(np.uint8)
    frame[:, :, 1] = (y % 256).astype(np.uint8)
    frame[:, :, 2] = ((x + y) % 256).astype(np.uint8)

    # Add high-contrast structure so the saved mock image is easy to recognize.
    frame[height // 2 - 2 : height // 2 + 2, :, :] = (0, 255, 255)
    frame[:, width // 2 - 2 : width // 2 + 2, :] = (0, 255, 255)
    frame[40:100, 60:180, :] = (0, 0, 255)
    frame[height - 100 : height - 40, width - 180 : width - 60, :] = (255, 0, 0)
    return frame


def encode_jpeg(frame: np.ndarray, *, quality: int = 90) -> bytes:
    """JPEG-encode a BGR frame with lazy cv2 import."""

    if quality < 1 or quality > 100:
        raise ValueError("quality must be in [1, 100]")
    cv2 = _import_cv2()
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise VisionUdpLoopbackError("JPEG encode failed")
    return bytes(encoded)


def save_image(path: str | Path, frame: np.ndarray) -> str:
    """Save a BGR image with lazy cv2 import and return the path."""

    cv2 = _import_cv2()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise VisionUdpLoopbackError(f"failed to write image: {output_path}")
    return str(output_path)


def packetize_jpeg_bytes(
    jpeg_bytes: bytes,
    *,
    frame_id: int = DEFAULT_FRAME_ID,
    sim_time_ns: int = DEFAULT_SIM_TIME_NS,
    max_payload_size: int = DEFAULT_MAX_PAYLOAD_SIZE,
    config: RuntimeCompetitionConfig = VADR_TS_002,
) -> PacketizedJpegFrame:
    """Split encoded JPEG bytes into VADR `<IHHIIQ` UDP datagrams."""

    jpeg = bytes(jpeg_bytes)
    if not jpeg:
        raise ValueError("jpeg_bytes must not be empty")
    if max_payload_size <= 0:
        raise ValueError("max_payload_size must be positive")

    chunks = tuple(
        jpeg[index : index + max_payload_size]
        for index in range(0, len(jpeg), max_payload_size)
    )
    if len(chunks) > 0xFFFF:
        raise ValueError("too many chunks for uint16 total_chunks")

    packets = tuple(
        pack_vision_packet(
            frame_id=frame_id,
            chunk_id=chunk_id,
            total_chunks=len(chunks),
            jpeg_size=len(jpeg),
            payload=payload,
            sim_time_ns=sim_time_ns,
            config=config,
        )
        for chunk_id, payload in enumerate(chunks)
    )
    return PacketizedJpegFrame(
        frame_id=int(frame_id),
        sim_time_ns=int(sim_time_ns),
        jpeg_bytes=jpeg,
        packets=packets,
    )


def build_packetized_mock_frame(
    *,
    config: RuntimeCompetitionConfig = VADR_TS_002,
    frame_id: int = DEFAULT_FRAME_ID,
    sim_time_ns: int = DEFAULT_SIM_TIME_NS,
    max_payload_size: int = DEFAULT_MAX_PAYLOAD_SIZE,
    jpeg_quality: int = 90,
) -> tuple[np.ndarray, PacketizedJpegFrame]:
    frame = build_mock_image(config=config)
    jpeg_bytes = encode_jpeg(frame, quality=jpeg_quality)
    packetized = packetize_jpeg_bytes(
        jpeg_bytes,
        frame_id=frame_id,
        sim_time_ns=sim_time_ns,
        max_payload_size=max_payload_size,
        config=config,
    )
    return frame, packetized


def process_packets_with_adapter(
    packets: Iterable[bytes],
    *,
    adapter: Optional[CompetitionImageAdapter] = None,
) -> tuple[VisionUdpLoopbackSummary, Optional[CompetitionCameraFrame]]:
    """Process packet bytes without sockets for deterministic tests."""

    active_adapter = adapter if adapter is not None else CompetitionImageAdapter()
    summary = VisionUdpLoopbackSummary(
        bind_host="in_memory",
        send_host="in_memory",
        port=0,
    )
    completed_frame: Optional[CompetitionCameraFrame] = None
    for packet in packets:
        summary.packets_received += 1
        try:
            frame = active_adapter.process_packet(packet)
        except CompetitionImageAdapterError as exc:
            summary.errors.append(str(exc))
            continue
        if frame is not None:
            completed_frame = frame
            summary.record_frame(frame)
    summary.record_adapter_stats(active_adapter)
    return summary, completed_frame


def run_udp_loopback(
    *,
    bind_host: str = DEFAULT_BIND_HOST,
    send_host: str = DEFAULT_SEND_HOST,
    port: int = VADR_TS_002.vision_udp_port,
    frames: int = 1,
    timeout_s: float = 5.0,
    max_payload_size: int = DEFAULT_MAX_PAYLOAD_SIZE,
    jpeg_quality: int = 90,
    save_mock_image: Optional[str | Path] = None,
    save_decoded_image: Optional[str | Path] = None,
    config: RuntimeCompetitionConfig = VADR_TS_002,
) -> VisionUdpLoopbackSummary:
    """Run one local UDP loopback smoke test through CompetitionImageAdapter."""

    if frames <= 0:
        raise ValueError("frames must be positive")
    if port <= 0 or port > 65535:
        raise ValueError("port must be in [1, 65535]")
    if timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive")

    adapter = CompetitionImageAdapter(config=config)
    summary = VisionUdpLoopbackSummary(
        bind_host=str(bind_host),
        send_host=str(send_host),
        port=int(port),
    )
    first_source_frame: Optional[np.ndarray] = None
    all_packets: list[bytes] = []
    for index in range(frames):
        source_frame, packetized = build_packetized_mock_frame(
            config=config,
            frame_id=DEFAULT_FRAME_ID + index,
            sim_time_ns=DEFAULT_SIM_TIME_NS + index * int(config.vision_period_s * 1e9),
            max_payload_size=max_payload_size,
            jpeg_quality=jpeg_quality,
        )
        if first_source_frame is None:
            first_source_frame = source_frame
        all_packets.extend(packetized.packets)

    if save_mock_image is not None and first_source_frame is not None:
        summary.mock_image_path = save_image(save_mock_image, first_source_frame)

    deadline = time.monotonic() + float(timeout_s)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as rx_sock:
        rx_sock.bind((bind_host, int(port)))
        rx_sock.settimeout(0.05)

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as tx_sock:
            for packet in all_packets:
                tx_sock.sendto(packet, (send_host, int(port)))
                summary.packets_sent += 1

        last_frame: Optional[CompetitionCameraFrame] = None
        while summary.frames_completed < frames and time.monotonic() < deadline:
            try:
                packet, _addr = rx_sock.recvfrom(65536)
            except socket.timeout:
                continue
            summary.packets_received += 1
            try:
                frame = adapter.process_packet(packet)
            except CompetitionImageAdapterError as exc:
                summary.errors.append(str(exc))
                continue
            if frame is not None:
                last_frame = frame
                summary.record_frame(frame)

    if summary.frames_completed < frames:
        summary.timed_out = True
        summary.errors.append(
            f"timed out before completing {frames} frame(s)"
        )

    summary.record_adapter_stats(adapter)
    if save_decoded_image is not None and last_frame is not None:
        summary.decoded_image_path = save_image(save_decoded_image, last_frame.frame)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Local UDP loopback smoke test for VADR vision packets. This "
            "generates a mock image, sends VADR JPEG UDP packets, and feeds "
            "received datagrams to CompetitionImageAdapter only."
        )
    )
    parser.add_argument("--bind-host", default=DEFAULT_BIND_HOST)
    parser.add_argument("--send-host", default=DEFAULT_SEND_HOST)
    parser.add_argument("--port", type=int, default=VADR_TS_002.vision_udp_port)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--max-payload-size", type=int, default=DEFAULT_MAX_PAYLOAD_SIZE)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--save-mock-image", default=None)
    parser.add_argument("--save-decoded-image", default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = run_udp_loopback(
        bind_host=args.bind_host,
        send_host=args.send_host,
        port=args.port,
        frames=args.frames,
        timeout_s=args.timeout_s,
        max_payload_size=args.max_payload_size,
        jpeg_quality=args.jpeg_quality,
        save_mock_image=args.save_mock_image,
        save_decoded_image=args.save_decoded_image,
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 1 if summary.timed_out or summary.errors else 0


def _import_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise VisionUdpLoopbackError(
            "cv2 is required for explicit Phase 8.5C JPEG encode/decode smoke "
            "runs; install nothing from this tool and run inside an environment "
            "that already provides cv2."
        ) from exc
    return cv2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_BIND_HOST",
    "DEFAULT_FRAME_ID",
    "DEFAULT_MAX_PAYLOAD_SIZE",
    "DEFAULT_SEND_HOST",
    "DEFAULT_SIM_TIME_NS",
    "PacketizedJpegFrame",
    "VisionUdpLoopbackError",
    "VisionUdpLoopbackSummary",
    "build_arg_parser",
    "build_mock_image",
    "build_packetized_mock_frame",
    "encode_jpeg",
    "main",
    "packetize_jpeg_bytes",
    "parse_vision_packet_header",
    "process_packets_with_adapter",
    "run_udp_loopback",
    "save_image",
]
