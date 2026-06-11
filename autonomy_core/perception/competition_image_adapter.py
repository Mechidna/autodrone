"""VADR-TS-002 UDP vision packet parsing and JPEG frame adaptation.

This module is transport-safe: importing it does not open sockets or start
background work. It only parses packet bytes supplied by a future runner.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from autonomy_core.core.competition_config import RuntimeCompetitionConfig, VADR_TS_002


class CompetitionImageAdapterError(ValueError):
    """Raised for malformed competition vision packets."""


@dataclass(frozen=True)
class VisionPacketHeader:
    frame_id: int
    chunk_id: int
    total_chunks: int
    jpeg_size: int
    payload_size: int
    sim_time_ns: int


@dataclass(frozen=True)
class CompetitionCameraFrame:
    frame: np.ndarray
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_stamp_sec: int
    image_stamp_nanosec: int
    image_received_wall_time: float
    image_pose_snapshot: None = None
    gazebo_pose: None = None

    def update_gate_memory_kwargs(self) -> dict:
        return {
            "frame": self.frame,
            "camera_matrix": self.camera_matrix,
            "dist_coeffs": self.dist_coeffs,
            "image_stamp_sec": self.image_stamp_sec,
            "image_stamp_nanosec": self.image_stamp_nanosec,
            "image_received_wall_time": self.image_received_wall_time,
            "image_pose_snapshot": None,
            "gazebo_pose": None,
        }


@dataclass
class VisionAdapterStats:
    packets_received: int = 0
    packets_rejected: int = 0
    duplicate_chunks: int = 0
    inconsistent_chunks: int = 0
    incomplete_frames_dropped: int = 0
    stale_frames_dropped: int = 0
    decode_failures: int = 0
    completed_frames: int = 0


@dataclass
class _PartialFrame:
    total_chunks: int
    jpeg_size: int
    sim_time_ns: int
    first_seen_wall_time: float
    chunks: Dict[int, bytes] = field(default_factory=dict)
    received_bytes: int = 0


def parse_vision_packet_header(
    packet: bytes,
    *,
    config: RuntimeCompetitionConfig = VADR_TS_002,
) -> VisionPacketHeader:
    """Parse and validate the VADR-TS-002 vision packet header."""

    header_size = struct.calcsize(config.vision_header_format)
    if header_size != config.vision_header_size_bytes:
        raise CompetitionImageAdapterError(
            "configured vision header size does not match struct format"
        )
    if len(packet) < header_size:
        raise CompetitionImageAdapterError("vision packet is shorter than header")

    values = struct.unpack(config.vision_header_format, packet[:header_size])
    header = VisionPacketHeader(*map(int, values))
    payload_size = len(packet) - header_size

    if header.total_chunks <= 0:
        raise CompetitionImageAdapterError("total_chunks must be positive")
    if header.chunk_id >= header.total_chunks:
        raise CompetitionImageAdapterError("chunk_id must be less than total_chunks")
    if header.jpeg_size <= 0:
        raise CompetitionImageAdapterError("jpeg_size must be positive")
    if header.payload_size != payload_size:
        raise CompetitionImageAdapterError("payload_size does not match packet payload")
    if header.payload_size <= 0:
        raise CompetitionImageAdapterError("payload_size must be positive")
    if header.payload_size > header.jpeg_size:
        raise CompetitionImageAdapterError("payload_size cannot exceed jpeg_size")

    return header


def _decode_jpeg(jpeg_bytes: bytes) -> Optional[np.ndarray]:
    import cv2

    img_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


class CompetitionImageAdapter:
    """Reassemble chunked VADR-TS-002 JPEG packets into perception kwargs."""

    def __init__(
        self,
        *,
        config: RuntimeCompetitionConfig = VADR_TS_002,
        clock: Callable[[], float] = time.time,
        max_incomplete_frames: int = 8,
        max_frame_age_s: float = 1.0,
        jpeg_decoder: Callable[[bytes], Optional[np.ndarray]] = _decode_jpeg,
    ):
        if max_incomplete_frames <= 0:
            raise ValueError("max_incomplete_frames must be positive")
        if max_frame_age_s <= 0.0:
            raise ValueError("max_frame_age_s must be positive")

        self.config = config
        self.clock = clock
        self.max_incomplete_frames = int(max_incomplete_frames)
        self.max_frame_age_s = float(max_frame_age_s)
        self.jpeg_decoder = jpeg_decoder
        self.stats = VisionAdapterStats()
        self._frames: Dict[int, _PartialFrame] = {}

    @property
    def pending_frame_count(self) -> int:
        return len(self._frames)

    def process_packet(self, packet: bytes) -> Optional[CompetitionCameraFrame]:
        """Process one UDP datagram payload and emit a completed frame if ready."""

        self.stats.packets_received += 1
        now = float(self.clock())
        self._drop_stale_frames(now)

        try:
            header = parse_vision_packet_header(packet, config=self.config)
        except CompetitionImageAdapterError:
            self.stats.packets_rejected += 1
            raise

        payload = packet[self.config.vision_header_size_bytes :]
        partial = self._frames.get(header.frame_id)
        if partial is None:
            partial = _PartialFrame(
                total_chunks=header.total_chunks,
                jpeg_size=header.jpeg_size,
                sim_time_ns=header.sim_time_ns,
                first_seen_wall_time=now,
            )
            self._frames[header.frame_id] = partial
            self._enforce_frame_limit()
        elif (
            partial.total_chunks != header.total_chunks
            or partial.jpeg_size != header.jpeg_size
            or partial.sim_time_ns != header.sim_time_ns
        ):
            self.stats.inconsistent_chunks += 1
            self.stats.incomplete_frames_dropped += 1
            del self._frames[header.frame_id]
            raise CompetitionImageAdapterError(
                "chunk metadata is inconsistent with existing partial frame"
            )

        if header.chunk_id in partial.chunks:
            self.stats.duplicate_chunks += 1
            return None

        if partial.received_bytes + len(payload) > partial.jpeg_size:
            self.stats.packets_rejected += 1
            raise CompetitionImageAdapterError("received bytes exceed jpeg_size")

        partial.chunks[header.chunk_id] = payload
        partial.received_bytes += len(payload)

        if len(partial.chunks) != partial.total_chunks:
            return None

        return self._complete_frame(header.frame_id, partial, now)

    def _complete_frame(
        self,
        frame_id: int,
        partial: _PartialFrame,
        received_wall_time: float,
    ) -> CompetitionCameraFrame:
        missing = [
            chunk_id
            for chunk_id in range(partial.total_chunks)
            if chunk_id not in partial.chunks
        ]
        if missing:
            self.stats.incomplete_frames_dropped += 1
            del self._frames[frame_id]
            raise CompetitionImageAdapterError("frame is missing one or more chunks")

        jpeg_bytes = b"".join(partial.chunks[i] for i in range(partial.total_chunks))
        del self._frames[frame_id]

        if len(jpeg_bytes) != partial.jpeg_size:
            self.stats.packets_rejected += 1
            raise CompetitionImageAdapterError("reassembled JPEG size mismatch")

        frame = self.jpeg_decoder(jpeg_bytes)
        if frame is None:
            self.stats.decode_failures += 1
            return None
        if frame.shape[:2] != (self.config.camera_height_px, self.config.camera_width_px):
            self.stats.decode_failures += 1
            return None

        self.stats.completed_frames += 1
        image_stamp_sec, image_stamp_nanosec = sim_time_ns_to_stamp(partial.sim_time_ns)
        return CompetitionCameraFrame(
            frame=frame,
            camera_matrix=np.asarray(self.config.camera_matrix, dtype=float),
            dist_coeffs=np.asarray(self.config.dist_coeffs, dtype=float),
            image_stamp_sec=image_stamp_sec,
            image_stamp_nanosec=image_stamp_nanosec,
            image_received_wall_time=received_wall_time,
            image_pose_snapshot=None,
            gazebo_pose=None,
        )

    def _drop_stale_frames(self, now: float) -> None:
        stale_frame_ids = [
            frame_id
            for frame_id, partial in self._frames.items()
            if now - partial.first_seen_wall_time > self.max_frame_age_s
        ]
        for frame_id in stale_frame_ids:
            del self._frames[frame_id]
            self.stats.stale_frames_dropped += 1
            self.stats.incomplete_frames_dropped += 1

    def _enforce_frame_limit(self) -> None:
        while len(self._frames) > self.max_incomplete_frames:
            oldest_frame_id = min(
                self._frames,
                key=lambda frame_id: self._frames[frame_id].first_seen_wall_time,
            )
            del self._frames[oldest_frame_id]
            self.stats.incomplete_frames_dropped += 1


def sim_time_ns_to_stamp(sim_time_ns: int) -> tuple[int, int]:
    if sim_time_ns < 0:
        raise ValueError("sim_time_ns must be non-negative")
    sec = int(sim_time_ns // 1_000_000_000)
    nanosec = int(sim_time_ns % 1_000_000_000)
    return sec, nanosec


def pack_vision_packet(
    *,
    frame_id: int,
    chunk_id: int,
    total_chunks: int,
    jpeg_size: int,
    payload: bytes,
    sim_time_ns: int,
    config: RuntimeCompetitionConfig = VADR_TS_002,
) -> bytes:
    """Pack a VADR-TS-002 vision packet for deterministic fixtures."""

    payload = bytes(payload)
    header = struct.pack(
        config.vision_header_format,
        int(frame_id),
        int(chunk_id),
        int(total_chunks),
        int(jpeg_size),
        len(payload),
        int(sim_time_ns),
    )
    return header + payload


__all__ = [
    "CompetitionCameraFrame",
    "CompetitionImageAdapter",
    "CompetitionImageAdapterError",
    "VisionAdapterStats",
    "VisionPacketHeader",
    "pack_vision_packet",
    "parse_vision_packet_header",
    "sim_time_ns_to_stamp",
]
