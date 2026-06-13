import subprocess
import sys
import unittest

import numpy as np

from autonomy_core.core.competition_config import VADR_TS_002
from autonomy_core.perception.competition_image_adapter import (
    CompetitionImageAdapter,
    parse_vision_packet_header,
)
from autonomy_core.tools.competition_vision_udp_loopback import (
    DEFAULT_MAX_PAYLOAD_SIZE,
    build_arg_parser,
    build_mock_image,
    packetize_jpeg_bytes,
    process_packets_with_adapter,
)


class CompetitionVisionUdpLoopbackTests(unittest.TestCase):
    def test_import_does_not_load_cv2_or_open_runtime_dependencies(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import sys; "
                    "import autonomy_core.tools.competition_vision_udp_loopback; "
                    "print('cv2' in sys.modules, 'pymavlink' in sys.modules, "
                    "'mavsdk' in sys.modules, 'rclpy' in sys.modules)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.stdout.strip(), "False False False False")

    def test_build_mock_image_uses_official_resolution_and_nonuniform_pattern(self):
        frame = build_mock_image()

        self.assertEqual(frame.shape, (360, 640, 3))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertGreater(np.unique(frame.reshape(-1, 3), axis=0).shape[0], 10)
        np.testing.assert_array_equal(
            frame.shape[:2],
            (VADR_TS_002.camera_height_px, VADR_TS_002.camera_width_px),
        )

    def test_packetize_jpeg_bytes_uses_exact_vadr_headers(self):
        jpeg = bytes(range(250)) * 12
        packetized = packetize_jpeg_bytes(
            jpeg,
            frame_id=42,
            sim_time_ns=99_000_001,
            max_payload_size=500,
        )

        self.assertEqual(packetized.jpeg_size, len(jpeg))
        self.assertEqual(packetized.packet_count, 6)
        for chunk_id, packet in enumerate(packetized.packets):
            header = parse_vision_packet_header(packet)
            self.assertEqual(header.frame_id, 42)
            self.assertEqual(header.chunk_id, chunk_id)
            self.assertEqual(header.total_chunks, 6)
            self.assertEqual(header.jpeg_size, len(jpeg))
            self.assertLessEqual(header.payload_size, 500)
            self.assertEqual(header.sim_time_ns, 99_000_001)

    def test_process_packets_with_adapter_completes_frame_without_sockets(self):
        jpeg = b"fake-jpeg"
        packetized = packetize_jpeg_bytes(
            jpeg,
            frame_id=7,
            sim_time_ns=1_234_567_890,
            max_payload_size=4,
        )
        adapter = CompetitionImageAdapter(
            jpeg_decoder=lambda _jpeg: np.zeros((360, 640, 3), dtype=np.uint8),
        )

        summary, frame = process_packets_with_adapter(
            packetized.packets,
            adapter=adapter,
        )

        self.assertIsNotNone(frame)
        self.assertEqual(summary.packets_received, 3)
        self.assertEqual(summary.frames_completed, 1)
        self.assertEqual(summary.image_shape, (360, 640, 3))
        self.assertEqual(summary.image_dtype, "uint8")
        self.assertEqual(summary.image_stamp_sec, 1)
        self.assertEqual(summary.image_stamp_nanosec, 234_567_890)
        self.assertEqual(
            summary.camera_matrix,
            [[320.0, 0.0, 320.0], [0.0, 320.0, 180.0], [0.0, 0.0, 1.0]],
        )
        self.assertEqual(summary.dist_coeffs, [0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertIsNone(summary.gazebo_pose)
        self.assertIsNone(summary.image_pose_snapshot)
        self.assertFalse(summary.errors)

    def test_process_packets_records_rejections_without_sockets(self):
        summary, frame = process_packets_with_adapter([b"too-short"])

        self.assertIsNone(frame)
        self.assertEqual(summary.packets_received, 1)
        self.assertEqual(summary.frames_completed, 0)
        self.assertEqual(summary.packets_rejected, 1)
        self.assertIn("vision packet is shorter than header", summary.errors)

    def test_arg_parser_defaults_match_phase_8_5c_loopback(self):
        args = build_arg_parser().parse_args([])

        self.assertEqual(args.bind_host, "0.0.0.0")
        self.assertEqual(args.send_host, "127.0.0.1")
        self.assertEqual(args.port, 5600)
        self.assertEqual(args.frames, 1)
        self.assertEqual(args.max_payload_size, DEFAULT_MAX_PAYLOAD_SIZE)


if __name__ == "__main__":
    unittest.main()
