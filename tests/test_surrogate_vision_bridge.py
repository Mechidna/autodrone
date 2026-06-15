import subprocess
import sys
import unittest

import numpy as np

from autonomy_core.perception.competition_image_adapter import (
    CompetitionImageAdapter,
    parse_vision_packet_header,
)
from autonomy_core.runtime.surrogate_vision_bridge import (
    DEFAULT_MAX_PAYLOAD_SIZE,
    PHASE_8_5E,
    SURROGATE_VISION_LABEL,
    SurrogateVisionBridgeConfig,
    SurrogateVisionBridgeError,
    UdpVisionPacketSender,
    build_arg_parser,
    fail_closed_summary,
    packetize_frame_for_vadr,
    run_surrogate_vision_bridge,
    validate_or_resize_frame_for_competition,
)


class FakeFrameSource:
    def __init__(self, frames):
        self.frames = list(frames)
        self.capture_calls = 0

    def capture_frame(self):
        self.capture_calls += 1
        if not self.frames:
            raise AssertionError("no fake frames remaining")
        return self.frames.pop(0)


class FakePacketSender:
    def __init__(self):
        self.calls = []

    def send_packets(self, packets):
        packet_tuple = tuple(bytes(packet) for packet in packets)
        self.calls.append(packet_tuple)
        return len(packet_tuple), sum(len(packet) for packet in packet_tuple)


class FakeSocket:
    def __init__(self):
        self.sent = []
        self.closed = False

    def sendto(self, packet, address):
        self.sent.append((bytes(packet), address))

    def close(self):
        self.closed = True


class SurrogateVisionBridgeTests(unittest.TestCase):
    def test_import_does_not_load_live_dependencies(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import sys; "
                    "import autonomy_core.runtime.surrogate_vision_bridge; "
                    "print('rclpy' in sys.modules, 'cv2' in sys.modules, "
                    "'pymavlink' in sys.modules, 'mavsdk' in sys.modules, "
                    "'autonomy_core.launch.autonomy_api6' in sys.modules)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.stdout.strip(), "False False False False False")

    def test_packetize_frame_for_vadr_uses_exact_header_without_cv2(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)

        packetized = packetize_frame_for_vadr(
            frame,
            frame_id=9,
            sim_time_ns=88_000_001,
            max_payload_size=4,
            jpeg_encoder=lambda _frame, quality: b"surrogate-jpeg",
        )

        self.assertEqual(packetized.frame_id, 9)
        self.assertEqual(packetized.sim_time_ns, 88_000_001)
        self.assertEqual(packetized.jpeg_size, len(b"surrogate-jpeg"))
        self.assertEqual(len(packetized.packets), 4)
        for chunk_id, packet in enumerate(packetized.packets):
            header = parse_vision_packet_header(packet)
            self.assertEqual(header.frame_id, 9)
            self.assertEqual(header.chunk_id, chunk_id)
            self.assertEqual(header.total_chunks, 4)
            self.assertEqual(header.jpeg_size, len(b"surrogate-jpeg"))
            self.assertLessEqual(header.payload_size, 4)
            self.assertEqual(header.sim_time_ns, 88_000_001)

    def test_run_bridge_sends_fake_frame_packets_without_socket(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame_source = FakeFrameSource([frame])
        packet_sender = FakePacketSender()

        summary = run_surrogate_vision_bridge(
            SurrogateVisionBridgeConfig(
                camera_topic="/camera",
                frames=1,
                max_payload_size=5,
                frame_id_start=44,
                sim_time_ns_start=1_234_567_890,
            ),
            frame_source=frame_source,
            packet_sender=packet_sender,
            jpeg_encoder=lambda _frame, quality: b"0123456789abc",
        )

        data = summary.to_dict()
        self.assertEqual(data["surrogate_label"], SURROGATE_VISION_LABEL)
        self.assertEqual(data["phase"], PHASE_8_5E)
        self.assertEqual(data["status"], "surrogate_vision_bridge_complete")
        self.assertEqual(data["frames_captured"], 1)
        self.assertEqual(data["frames_sent"], 1)
        self.assertEqual(data["packets_sent"], 3)
        self.assertGreater(data["bytes_sent"], 0)
        self.assertEqual(data["last_frame_id"], 44)
        self.assertEqual(data["last_sim_time_ns"], 1_234_567_890)
        self.assertEqual(data["image_shape"], [360, 640, 3])
        self.assertEqual(data["resize_applied_count"], 0)
        self.assertEqual(data["command_sent_count"], 0)
        self.assertFalse(data["phase4b_satisfied"])
        self.assertFalse(data["phase9_satisfied"])
        self.assertFalse(data["competition_readiness_claimed"])
        self.assertEqual(frame_source.capture_calls, 1)
        self.assertEqual(len(packet_sender.calls), 1)

    def test_bridge_packets_are_accepted_by_competition_image_adapter(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        packet_sender = FakePacketSender()
        run_surrogate_vision_bridge(
            SurrogateVisionBridgeConfig(frames=1, max_payload_size=3),
            frame_source=FakeFrameSource([frame]),
            packet_sender=packet_sender,
            jpeg_encoder=lambda _frame, quality: b"fake-jpeg",
        )
        adapter = CompetitionImageAdapter(
            jpeg_decoder=lambda _jpeg: np.zeros((360, 640, 3), dtype=np.uint8),
        )

        completed = None
        for packet in packet_sender.calls[0]:
            completed = adapter.process_packet(packet) or completed

        self.assertIsNotNone(completed)
        self.assertEqual(adapter.stats.completed_frames, 1)
        self.assertIsNone(completed.gazebo_pose)
        self.assertIsNone(completed.image_pose_snapshot)

    def test_rejects_non_competition_frame_size_without_explicit_resize(self):
        frame = np.zeros((960, 1280, 3), dtype=np.uint8)

        with self.assertRaises(SurrogateVisionBridgeError) as ctx:
            validate_or_resize_frame_for_competition(frame, resize_enabled=False)

        self.assertIn("requires 640x360", str(ctx.exception))

    def test_explicit_resize_path_can_be_tested_without_cv2(self):
        frame = np.zeros((960, 1280, 3), dtype=np.uint8)
        resized_frame = np.zeros((360, 640, 3), dtype=np.uint8)

        output, resized = validate_or_resize_frame_for_competition(
            frame,
            resize_enabled=True,
            resizer=lambda _frame, width, height: resized_frame,
        )

        self.assertTrue(resized)
        self.assertIs(output, resized_frame)

    def test_rejects_truth_like_camera_topic(self):
        with self.assertRaises(SurrogateVisionBridgeError):
            run_surrogate_vision_bridge(
                SurrogateVisionBridgeConfig(
                    camera_topic="/world/gate_test_1500mm_blue/dynamic_pose/info",
                ),
                frame_source=FakeFrameSource([np.zeros((360, 640, 3), dtype=np.uint8)]),
                packet_sender=FakePacketSender(),
                jpeg_encoder=lambda _frame, quality: b"fake-jpeg",
            )

    def test_udp_sender_can_use_injected_socket_factory(self):
        fake_socket = FakeSocket()
        sender = UdpVisionPacketSender(
            send_host="127.0.0.1",
            send_port=5600,
            socket_factory=lambda: fake_socket,
        )

        packets_sent, bytes_sent = sender.send_packets((b"abc", b"defg"))

        self.assertEqual(packets_sent, 2)
        self.assertEqual(bytes_sent, 7)
        self.assertEqual(
            fake_socket.sent,
            [
                (b"abc", ("127.0.0.1", 5600)),
                (b"defg", ("127.0.0.1", 5600)),
            ],
        )
        self.assertTrue(fake_socket.closed)

    def test_fail_closed_summary_never_claims_readiness(self):
        summary = fail_closed_summary(
            config=SurrogateVisionBridgeConfig(frames=2),
            error="example safety error",
        )
        data = summary.to_dict()

        self.assertEqual(data["status"], "fail_closed")
        self.assertTrue(data["fail_closed"])
        self.assertEqual(data["frames_requested"], 2)
        self.assertEqual(data["command_sent_count"], 0)
        self.assertFalse(data["phase4b_satisfied"])
        self.assertFalse(data["phase9_satisfied"])
        self.assertFalse(data["competition_readiness_claimed"])
        self.assertEqual(data["errors"], ["example safety error"])

    def test_arg_parser_defaults_match_phase_8_5e_bridge(self):
        args = build_arg_parser().parse_args([])

        self.assertEqual(args.camera_topic, "/camera")
        self.assertEqual(args.send_host, "127.0.0.1")
        self.assertEqual(args.send_port, 5600)
        self.assertEqual(args.frames, 1)
        self.assertEqual(args.max_payload_size, DEFAULT_MAX_PAYLOAD_SIZE)
        self.assertFalse(args.resize_camera_to_competition)


if __name__ == "__main__":
    unittest.main()
