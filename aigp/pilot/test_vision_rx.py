import socket
import struct
import time
import unittest

import cv2
import numpy as np

from runtime_config import load_runtime_config
from vision_rx import (
    VisionRX,
    _CompletedFrameCache,
    _configure_udp_socket_receive_buffer,
)


class _FakeSocket:
    def __init__(self):
        self.requested = None

    def setsockopt(self, level, option, value):
        self.requested = (level, option, value)

    def getsockopt(self, level, option):
        return self.requested[2]


class _RecorderSpy:
    enabled = True

    def __init__(self):
        self.frames = []
        self.events = []

    def record_camera_frame(self, **frame):
        self.frames.append(frame)

    def record_event(self, event_type, **details):
        self.events.append((event_type, details))


class CompletedFrameCacheTests(unittest.TestCase):
    def test_same_id_and_timestamp_is_reported_once_then_suppressed(self):
        cache = _CompletedFrameCache(max_entries=4, ttl_s=1.0)
        key = (7, 123_000)
        cache.add(key, 1_000_000_000)

        duplicate, first_report, completed_ns = cache.check_duplicate(
            key, 1_100_000_000
        )
        self.assertTrue(duplicate)
        self.assertTrue(first_report)
        self.assertEqual(completed_ns, 1_000_000_000)

        duplicate, first_report, _ = cache.check_duplicate(
            key, 1_200_000_000
        )
        self.assertTrue(duplicate)
        self.assertFalse(first_report)
        self.assertEqual(
            cache.check_duplicate((7, 124_000), 1_200_000_000)[0],
            False,
        )

    def test_cache_expires_and_evicts_old_entries(self):
        cache = _CompletedFrameCache(max_entries=2, ttl_s=0.5)
        cache.add((1, 1), 1_000_000_000)
        cache.add((2, 2), 1_100_000_000)
        cache.add((3, 3), 1_200_000_000)

        self.assertEqual(len(cache), 2)
        self.assertFalse(cache.check_duplicate((1, 1), 1_300_000_000)[0])
        self.assertTrue(cache.check_duplicate((2, 2), 1_300_000_000)[0])
        self.assertFalse(cache.check_duplicate((2, 2), 1_700_000_001)[0])


class UdpBufferTests(unittest.TestCase):
    def test_runtime_defaults_provide_burst_and_reassembly_headroom(self):
        vision = load_runtime_config().vision

        self.assertEqual(vision.udp_socket_receive_buffer_bytes, 8 * 1024 * 1024)
        self.assertEqual(vision.max_pending_frames, 64)
        self.assertEqual(vision.stale_frame_timeout_s, 1.0)
        self.assertEqual(vision.completed_frame_cache_size, 512)
        self.assertEqual(vision.completed_frame_cache_ttl_s, 5.0)

    def test_requested_socket_receive_buffer_is_applied_and_reported(self):
        sock = _FakeSocket()
        effective = _configure_udp_socket_receive_buffer(sock, 8 * 1024 * 1024)

        self.assertEqual(effective, 8 * 1024 * 1024)
        self.assertEqual(sock.requested[0], socket.SOL_SOCKET)
        self.assertEqual(sock.requested[1], socket.SO_RCVBUF)


class VisionReceiverIntegrationTests(unittest.TestCase):
    def test_retransmitted_frame_is_not_decoded_published_or_recorded_twice(self):
        recorder = _RecorderSpy()
        shared = {}
        receiver = VisionRX(
            shared,
            bind_ip="127.0.0.1",
            port=0,
            socket_timeout_s=0.01,
            recv_bytes=65536,
            socket_receive_buffer_bytes=8 * 1024 * 1024,
            header_format="<IHHIIQ",
            max_pending_frames=8,
            stale_frame_timeout_s=0.2,
            completed_frame_cache_size=8,
            completed_frame_cache_ttl_s=1.0,
            max_jpeg_size_bytes=100_000,
            expected_width=16,
            expected_height=12,
            vio_recorder=recorder,
        )
        self.assertTrue(receiver.socket_ready.wait(timeout=1.0))

        image = np.full((12, 16, 3), 127, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        jpeg = encoded.tobytes()

        chunk_size = (len(jpeg) + 2) // 3
        chunks = [
            jpeg[index:index + chunk_size]
            for index in range(0, len(jpeg), chunk_size)
        ]

        def packets(sim_time_ns):
            result = []
            for chunk_id, payload in enumerate(chunks):
                header = struct.pack(
                    "<IHHIIQ",
                    9,
                    chunk_id,
                    len(chunks),
                    len(jpeg),
                    len(payload),
                    sim_time_ns,
                )
                result.append(header + payload)
            return result

        def send_frame(sender, target, sim_time_ns):
            for item in packets(sim_time_ns):
                sender.sendto(item, target)

        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            target = ("127.0.0.1", receiver.bound_port)
            send_frame(sender, target, 100_000)
            self.assertTrue(self._wait_for(lambda: len(recorder.frames) == 1))

            send_frame(sender, target, 100_000)
            self.assertTrue(
                self._wait_for(
                    lambda: any(
                        event == "camera_duplicate_frame_suppressed"
                        for event, _ in recorder.events
                    )
                )
            )
            time.sleep(0.03)
            self.assertEqual(len(recorder.frames), 1)
            self.assertEqual(shared.get("vision_frame_count"), 1)

            send_frame(sender, target, 200_000)
            self.assertTrue(self._wait_for(lambda: len(recorder.frames) == 2))
            self.assertEqual(shared.get("vision_frame_count"), 2)
        finally:
            sender.close()
            thread = receiver.get_thread_for_join()
            thread.join(timeout=1.0)
            self.assertFalse(thread.is_alive())

        duplicate_events = [
            details
            for event, details in recorder.events
            if event == "camera_duplicate_frame_suppressed"
        ]
        self.assertEqual(len(duplicate_events), 1)
        self.assertEqual(duplicate_events[0]["frame_id"], 9)
        self.assertEqual(duplicate_events[0]["sim_time_ns"], 100_000)

    @staticmethod
    def _wait_for(predicate, timeout_s=1.0):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.005)
        return bool(predicate())


if __name__ == "__main__":
    unittest.main()
