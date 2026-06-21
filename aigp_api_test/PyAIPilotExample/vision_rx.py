import socket
import struct
import threading
import time

import cv2
import numpy as np

# Modify these properties if you want to run the server remotely for example
SIM_SERVER_UDP_IP = "0.0.0.0"
SIM_SERVER_UDP_PORT = 5600

class VisionRX:

    def __init__(self, data):
        self.data = data
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
        header_format = "<IHHIIQ"
        header_sz = struct.calcsize(header_format)
        frames = {}  # frame_id -> received associated frame data

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT))
        print("Listening for camera frames...")
        sock.settimeout(0.1)

        while self.is_running:
            try:
                packet, addr = sock.recvfrom(65536)
            except socket.timeout:
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

            if frame_id not in frames:
                frames[frame_id] = {
                    "chunks": {},
                    "total": total_chunks,
                    "size": jpeg_size,
                    "time": sim_time_ns
                }

            frames[frame_id]["chunks"][chunk_id] = payload

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
                    self.process_frame(frame_id, image, sim_time_ns)
                else:
                    print(f"Failed to decode frame: {frame_id}")

                del frames[frame_id]

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