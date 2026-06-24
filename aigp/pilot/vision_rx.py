import socket
import struct
import threading
import time

import cv2
import numpy as np

from runtime_config import load_runtime_config

class VisionRX:

    def __init__(
        self,
        data,
        bind_ip=None,
        port=None,
        socket_timeout_s=None,
        recv_bytes=None,
        header_format=None,
        max_pending_frames=None,
        stale_frame_timeout_s=None,
        max_jpeg_size_bytes=None,
        expected_width=None,
        expected_height=None,
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
        self.thread = threading.Thread(
            target=self._vision_loop,
            daemon=False
        )
        self.is_running = True
        self.thread.start()

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    def _vision_loop(self):
        header_format = self.header_format
        header_sz = struct.calcsize(header_format)
        frames = {}  # frame_id -> received associated frame data

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.bind_ip, self.port))
        print(f"Listening for camera frames on {self.bind_ip}:{self.port}...")
        sock.settimeout(self.socket_timeout_s)

        while self.is_running:
            try:
                packet, addr = sock.recvfrom(self.recv_bytes)
            except socket.timeout:
                self._prune_stale_frames(frames)
                continue

            if len(packet) < header_sz:
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
                continue

            if chunk_id >= total_chunks:
                continue

            if payload_size != len(payload):
                continue

            if self.max_jpeg_size_bytes > 0 and jpeg_size > self.max_jpeg_size_bytes:
                continue

            if frame_id not in frames:
                frames[frame_id] = {
                    "chunks": {},
                    "total": total_chunks,
                    "size": jpeg_size,
                    "time": sim_time_ns,
                    "first_seen_wall_time": time.time(),
                }

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
                    del frames[frame_id]
                    continue

                img_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if image is not None:
                    if self._valid_image_shape(image):
                        self.process_frame(frame_id, image, sim_time_ns)
                else:
                    print(f"Failed to decode frame: {frame_id}")

                del frames[frame_id]

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
            frames.pop(frame_id, None)

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
            frames.pop(oldest_frame_id, None)

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
