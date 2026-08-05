import socket
import struct
import threading
import time
from collections import OrderedDict

import cv2
import numpy as np

from frame_capture import CameraFrameCapture
from runtime_config import load_runtime_config


def _configure_udp_socket_receive_buffer(sock, requested_bytes):
    requested = max(0, int(requested_bytes))
    if requested > 0:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, requested)
    return int(sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF))


class _CompletedFrameCache:
    """Bounded TTL cache used to suppress retransmitted completed frames."""

    def __init__(self, max_entries, ttl_s):
        self.max_entries = max(0, int(max_entries))
        self.ttl_ns = max(0, int(float(ttl_s) * 1e9))
        self._entries = OrderedDict()

    def add(self, key, completed_monotonic_ns):
        if self.max_entries <= 0 or self.ttl_ns <= 0:
            return
        self.prune(completed_monotonic_ns)
        self._entries.pop(key, None)
        self._entries[key] = {
            "completed_monotonic_ns": int(completed_monotonic_ns),
            "duplicate_reported": False,
        }
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def check_duplicate(self, key, now_monotonic_ns):
        self.prune(now_monotonic_ns)
        entry = self._entries.get(key)
        if entry is None:
            return False, False, None
        first_report = not bool(entry["duplicate_reported"])
        entry["duplicate_reported"] = True
        return True, first_report, int(entry["completed_monotonic_ns"])

    def prune(self, now_monotonic_ns):
        if self.max_entries <= 0 or self.ttl_ns <= 0:
            self._entries.clear()
            return
        cutoff = int(now_monotonic_ns) - self.ttl_ns
        while self._entries:
            first_key = next(iter(self._entries))
            completed_ns = int(
                self._entries[first_key]["completed_monotonic_ns"]
            )
            if completed_ns >= cutoff:
                break
            self._entries.popitem(last=False)

    def __len__(self):
        return len(self._entries)

class VisionRX:

    def __init__(
        self,
        data,
        bind_ip=None,
        port=None,
        socket_timeout_s=None,
        recv_bytes=None,
        socket_receive_buffer_bytes=None,
        header_format=None,
        max_pending_frames=None,
        stale_frame_timeout_s=None,
        completed_frame_cache_size=None,
        completed_frame_cache_ttl_s=None,
        max_jpeg_size_bytes=None,
        expected_width=None,
        expected_height=None,
        vio_recorder=None,
    ):
        config = load_runtime_config()
        self.data = data
        self.bind_ip = str(config.vision.udp_bind_ip if bind_ip is None else bind_ip)
        self.port = int(config.vision.udp_port if port is None else port)
        self.socket_timeout_s = float(
            config.vision.udp_socket_timeout_s
            if socket_timeout_s is None
            else socket_timeout_s
        )
        self.recv_bytes = int(config.vision.udp_recv_bytes if recv_bytes is None else recv_bytes)
        self.socket_receive_buffer_bytes = int(
            config.vision.udp_socket_receive_buffer_bytes
            if socket_receive_buffer_bytes is None
            else socket_receive_buffer_bytes
        )
        self.header_format = str(
            config.vision.packet_header_format
            if header_format is None
            else header_format
        )
        self.max_pending_frames = int(
            config.vision.max_pending_frames
            if max_pending_frames is None
            else max_pending_frames
        )
        self.stale_frame_timeout_s = float(
            config.vision.stale_frame_timeout_s
            if stale_frame_timeout_s is None
            else stale_frame_timeout_s
        )
        self.completed_frame_cache_size = int(
            config.vision.completed_frame_cache_size
            if completed_frame_cache_size is None
            else completed_frame_cache_size
        )
        self.completed_frame_cache_ttl_s = float(
            config.vision.completed_frame_cache_ttl_s
            if completed_frame_cache_ttl_s is None
            else completed_frame_cache_ttl_s
        )
        self.completed_frame_cache = _CompletedFrameCache(
            self.completed_frame_cache_size,
            self.completed_frame_cache_ttl_s,
        )
        self.max_jpeg_size_bytes = int(
            config.vision.max_jpeg_size_bytes
            if max_jpeg_size_bytes is None
            else max_jpeg_size_bytes
        )
        self.expected_width = int(
            config.camera.width if expected_width is None else expected_width
        )
        self.expected_height = int(
            config.camera.height if expected_height is None else expected_height
        )
        self.vio_recorder = vio_recorder
        self.frame_capture = CameraFrameCapture(source="udp_vision")
        self.socket_ready = threading.Event()
        self.bound_port = None
        self.thread = threading.Thread(
            target=self._vision_loop,
            daemon=False
        )
        self.is_running = True
        self.thread.start()

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    def _record_event(self, event_type, **details):
        recorder = self.vio_recorder
        if recorder is None or not recorder.enabled:
            return
        try:
            recorder.record_event(event_type, **details)
        except Exception as exc:
            print(f"WARNING: VIO recorder camera event failed: {exc}", flush=True)

    def _vision_loop(self):
        header_format = self.header_format
        header_sz = struct.calcsize(header_format)
        frames = {}  # frame_id -> received associated frame data

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        effective_receive_buffer_bytes = None
        try:
            effective_receive_buffer_bytes = _configure_udp_socket_receive_buffer(
                sock,
                self.socket_receive_buffer_bytes,
            )
        except OSError as exc:
            print(
                "WARNING: Unable to configure UDP camera receive buffer "
                f"requested={self.socket_receive_buffer_bytes}: {exc}",
                flush=True,
            )
            self._record_event(
                "camera_udp_socket_buffer_error",
                requested_receive_buffer_bytes=self.socket_receive_buffer_bytes,
                error=str(exc),
            )
        sock.bind((self.bind_ip, self.port))
        self.bound_port = int(sock.getsockname()[1])
        self.socket_ready.set()
        print(
            f"Listening for camera frames on {self.bind_ip}:{self.bound_port} "
            f"UDP_RCVBUF requested={self.socket_receive_buffer_bytes} "
            f"effective={effective_receive_buffer_bytes}...",
            flush=True,
        )
        self._record_event(
            "camera_udp_socket_config",
            requested_receive_buffer_bytes=self.socket_receive_buffer_bytes,
            effective_receive_buffer_bytes=effective_receive_buffer_bytes,
            max_pending_frames=self.max_pending_frames,
            stale_frame_timeout_s=self.stale_frame_timeout_s,
            completed_frame_cache_size=self.completed_frame_cache_size,
            completed_frame_cache_ttl_s=self.completed_frame_cache_ttl_s,
        )
        sock.settimeout(self.socket_timeout_s)

        while self.is_running:
            try:
                packet, addr = sock.recvfrom(self.recv_bytes)
            except socket.timeout:
                self._prune_stale_frames(frames)
                self.completed_frame_cache.prune(time.monotonic_ns())
                continue

            packet_wall_time_ns = time.time_ns()
            packet_monotonic_ns = time.monotonic_ns()
            if len(packet) < header_sz:
                self._record_event(
                    "camera_packet_rejected",
                    wall_time_ns=packet_wall_time_ns,
                    reason="short_header",
                    packet_size_bytes=len(packet),
                    header_size_bytes=header_sz,
                )
                continue

            header = packet[:header_sz]
            payload = packet[header_sz:]

            # frame_id - identifier for this vision frame
            # chunk_id - identifier for this chunk packet of data of this frame
            # total_chunks - total number of chunk packets that make up this frame
            # jpeg_size - full size of jpeg data
            # payload_size - size of this packet
            # sim_time_ns - frame's epoch timestamp in ns on the server
            frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns = struct.unpack(header_format, header)

            # Validate packet metadata before storing this chunk
            if total_chunks == 0:
                self._record_event(
                    "camera_packet_rejected",
                    wall_time_ns=packet_wall_time_ns,
                    reason="zero_total_chunks",
                    frame_id=frame_id,
                    chunk_id=chunk_id,
                )
                continue

            if chunk_id >= total_chunks:
                self._record_event(
                    "camera_packet_rejected",
                    wall_time_ns=packet_wall_time_ns,
                    reason="chunk_id_out_of_range",
                    frame_id=frame_id,
                    chunk_id=chunk_id,
                    total_chunks=total_chunks,
                )
                continue

            if payload_size != len(payload):
                self._record_event(
                    "camera_packet_rejected",
                    wall_time_ns=packet_wall_time_ns,
                    reason="payload_size_mismatch",
                    frame_id=frame_id,
                    chunk_id=chunk_id,
                    declared_payload_size=payload_size,
                    received_payload_size=len(payload),
                )
                continue

            if self.max_jpeg_size_bytes > 0 and jpeg_size > self.max_jpeg_size_bytes:
                self._record_event(
                    "camera_packet_rejected",
                    wall_time_ns=packet_wall_time_ns,
                    reason="jpeg_too_large",
                    frame_id=frame_id,
                    jpeg_size_bytes=jpeg_size,
                    max_jpeg_size_bytes=self.max_jpeg_size_bytes,
                )
                continue

            completed_key = (int(frame_id), int(sim_time_ns))
            (
                is_completed_duplicate,
                first_duplicate_report,
                completed_monotonic_ns,
            ) = self.completed_frame_cache.check_duplicate(
                completed_key,
                packet_monotonic_ns,
            )
            if is_completed_duplicate:
                if first_duplicate_report:
                    duplicate_age_ms = (
                        (packet_monotonic_ns - completed_monotonic_ns) / 1e6
                        if completed_monotonic_ns is not None
                        else None
                    )
                    self._record_event(
                        "camera_duplicate_frame_suppressed",
                        wall_time_ns=packet_wall_time_ns,
                        frame_id=frame_id,
                        sim_time_ns=sim_time_ns,
                        first_duplicate_chunk_id=chunk_id,
                        duplicate_age_ms=duplicate_age_ms,
                    )
                continue

            if frame_id not in frames:
                frames[frame_id] = {
                    "chunks": {},
                    "total": total_chunks,
                    "size": jpeg_size,
                    "time": sim_time_ns,
                    "first_seen_wall_time": time.time(),
                    "first_seen_wall_time_ns": packet_wall_time_ns,
                }
            else:
                existing = frames[frame_id]
                if (
                    int(existing["total"]) != int(total_chunks)
                    or int(existing["size"]) != int(jpeg_size)
                    or int(existing["time"]) != int(sim_time_ns)
                ):
                    self._record_event(
                        "camera_frame_dropped",
                        wall_time_ns=packet_wall_time_ns,
                        reason="metadata_changed_between_chunks",
                        frame_id=frame_id,
                        received_chunks=len(existing["chunks"]),
                    )
                    frames.pop(frame_id, None)
                    continue

            if chunk_id in frames[frame_id]["chunks"]:
                self._record_event(
                    "camera_duplicate_chunk",
                    wall_time_ns=packet_wall_time_ns,
                    frame_id=frame_id,
                    chunk_id=chunk_id,
                )
            frames[frame_id]["chunks"][chunk_id] = payload
            self._prune_stale_frames(frames)
            self._limit_pending_frames(frames)
            if frame_id not in frames:
                continue

            # Check if frame is complete
            if len(frames[frame_id]["chunks"]) == total_chunks:
                jpeg_bytes = bytearray()

                frame_complete = True
                for i in range(total_chunks):
                    if i not in frames[frame_id]["chunks"]:
                        print('Missing packet %s in frame %s' % (i, frame_id,))
                        self._record_event(
                            "camera_frame_dropped",
                            reason="missing_chunk_after_completion",
                            frame_id=frame_id,
                            missing_chunk_id=i,
                        )
                        frame_complete = False
                        continue
                    jpeg_bytes.extend(frames[frame_id]["chunks"][i])

                if not frame_complete:
                    del frames[frame_id]
                    continue

                if len(jpeg_bytes) != jpeg_size:
                    print(
                        f"JPEG size mismatch frame={frame_id}: "
                        f"got {len(jpeg_bytes)}, expected {jpeg_size}"
                    )
                    self._record_event(
                        "camera_frame_dropped",
                        reason="jpeg_size_mismatch",
                        frame_id=frame_id,
                        assembled_size_bytes=len(jpeg_bytes),
                        declared_size_bytes=jpeg_size,
                    )
                    del frames[frame_id]
                    continue

                self.completed_frame_cache.add(
                    completed_key,
                    time.monotonic_ns(),
                )

                recorder = self.vio_recorder
                if recorder is not None and recorder.enabled:
                    try:
                        recorder.record_camera_frame(
                            frame_id=frame_id,
                            sim_time_ns=sim_time_ns,
                            jpeg_bytes=bytes(jpeg_bytes),
                            receive_wall_time_ns=time.time_ns(),
                            width=self.expected_width,
                            height=self.expected_height,
                            total_chunks=total_chunks,
                        )
                    except Exception as exc:
                        print(
                            f"WARNING: VIO recorder camera frame failed: {exc}",
                            flush=True,
                        )

                img_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if image is not None:
                    if self._valid_image_shape(image):
                        self.process_frame(frame_id, image, sim_time_ns)
                    else:
                        height, width = image.shape[:2]
                        self._record_event(
                            "camera_image_shape_mismatch",
                            frame_id=frame_id,
                            width=int(width),
                            height=int(height),
                            expected_width=self.expected_width,
                            expected_height=self.expected_height,
                        )
                else:
                    print(f"Failed to decode frame: {frame_id}")
                    self._record_event(
                        "camera_decode_failed",
                        frame_id=frame_id,
                        sim_time_ns=sim_time_ns,
                        jpeg_size_bytes=jpeg_size,
                    )

                del frames[frame_id]

        for frame_id, frame in frames.items():
            self._record_event(
                "camera_frame_dropped",
                reason="receiver_shutdown_with_incomplete_frame",
                frame_id=frame_id,
                sim_time_ns=frame.get("time"),
                received_chunks=len(frame.get("chunks", {})),
                total_chunks=frame.get("total"),
            )
        sock.close()

    def _prune_stale_frames(self, frames):
        if self.stale_frame_timeout_s <= 0.0:
            return
        now = time.time()
        stale_ids = [
            frame_id
            for frame_id, frame in frames.items()
            if now - float(frame.get("first_seen_wall_time", now)) > self.stale_frame_timeout_s
        ]
        for frame_id in stale_ids:
            frame = frames.pop(frame_id, None)
            if frame is not None:
                self._record_event(
                    "camera_frame_dropped",
                    reason="stale_incomplete_frame",
                    frame_id=frame_id,
                    sim_time_ns=frame.get("time"),
                    received_chunks=len(frame.get("chunks", {})),
                    total_chunks=frame.get("total"),
                )

    def _limit_pending_frames(self, frames):
        if self.max_pending_frames <= 0:
            return
        while len(frames) > self.max_pending_frames:
            oldest_frame_id = min(
                frames,
                key=lambda frame_id: float(
                    frames[frame_id].get("first_seen_wall_time", 0.0)
                ),
            )
            frame = frames.pop(oldest_frame_id, None)
            if frame is not None:
                self._record_event(
                    "camera_frame_dropped",
                    reason="pending_frame_limit",
                    frame_id=oldest_frame_id,
                    sim_time_ns=frame.get("time"),
                    received_chunks=len(frame.get("chunks", {})),
                    total_chunks=frame.get("total"),
                )

    def _valid_image_shape(self, image):
        if self.expected_width is None and self.expected_height is None:
            return True
        height, width = image.shape[:2]
        if self.expected_width is not None and width != self.expected_width:
            print(
                f"Unexpected camera width frame={width}, expected {self.expected_width}",
                flush=True,
            )
            return False
        if self.expected_height is not None and height != self.expected_height:
            print(
                f"Unexpected camera height frame={height}, expected {self.expected_height}",
                flush=True,
            )
            return False
        return True

    def process_frame(self, frame_id, img, sim_time_ns=None):
        """
        Store the latest decoded camera frame for controller.py/autonomy_adapter.

        img is OpenCV BGR format because cv2.imdecode() returns BGR.
        """
        frame_data = {
            "frame_id": frame_id,
            "image": img,
            "shape": img.shape,
            "sim_time_ns": sim_time_ns,
            "wall_time": time.time(),
        }

        lock = self.data.get("lock") if isinstance(self.data, dict) else None

        if lock is not None:
            with lock:
                self.data["latest_frame"] = frame_data
                self.data["vision_frame_count"] = self.data.get("vision_frame_count", 0) + 1
        else:
            self.data["latest_frame"] = frame_data
            self.data["vision_frame_count"] = self.data.get("vision_frame_count", 0) + 1

        self.frame_capture.maybe_capture(frame_data, img)
