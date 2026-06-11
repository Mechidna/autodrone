import struct
import unittest

import cv2
import numpy as np

from autonomy_core.core.competition_config import RuntimeCompetitionConfig
from autonomy_core.perception.competition_image_adapter import (
    CompetitionImageAdapter,
    CompetitionImageAdapterError,
    pack_vision_packet,
    parse_vision_packet_header,
)


def make_test_jpeg():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:, :, 0] = 10
    frame[:, :, 1] = np.arange(640, dtype=np.uint8)
    frame[:, :, 2] = np.arange(360, dtype=np.uint8).reshape(360, 1)
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise AssertionError("fixture JPEG encode failed")
    return bytes(encoded)


def split_bytes(data, chunk_count):
    base = len(data) // chunk_count
    chunks = []
    start = 0
    for index in range(chunk_count):
        end = start + base
        if index == chunk_count - 1:
            end = len(data)
        chunks.append(data[start:end])
        start = end
    return chunks


class FakeClock:
    def __init__(self, value=100.0):
        self.value = float(value)

    def __call__(self):
        return self.value


class CompetitionImageAdapterTests(unittest.TestCase):
    def test_header_pack_and_parse_matches_exact_struct(self):
        config = RuntimeCompetitionConfig()
        payload = b"jpeg"
        packet = pack_vision_packet(
            frame_id=7,
            chunk_id=1,
            total_chunks=3,
            jpeg_size=12,
            payload=payload,
            sim_time_ns=1_234_567_890,
            config=config,
        )

        self.assertEqual(struct.calcsize(config.vision_header_format), 24)
        self.assertEqual(packet[:24], struct.pack("<IHHIIQ", 7, 1, 3, 12, 4, 1_234_567_890))

        header = parse_vision_packet_header(packet, config=config)

        self.assertEqual(header.frame_id, 7)
        self.assertEqual(header.chunk_id, 1)
        self.assertEqual(header.total_chunks, 3)
        self.assertEqual(header.jpeg_size, 12)
        self.assertEqual(header.payload_size, 4)
        self.assertEqual(header.sim_time_ns, 1_234_567_890)

    def test_reassembles_out_of_order_chunks_into_camera_frame_kwargs(self):
        clock = FakeClock(42.5)
        adapter = CompetitionImageAdapter(clock=clock)
        jpeg = make_test_jpeg()
        chunks = split_bytes(jpeg, 4)
        packets = [
            pack_vision_packet(
                frame_id=100,
                chunk_id=chunk_id,
                total_chunks=len(chunks),
                jpeg_size=len(jpeg),
                payload=chunk,
                sim_time_ns=9_876_543_210,
            )
            for chunk_id, chunk in enumerate(chunks)
        ]

        emitted = []
        for index in (2, 0, 3, 1):
            result = adapter.process_packet(packets[index])
            if result is not None:
                emitted.append(result)

        self.assertEqual(len(emitted), 1)
        camera_frame = emitted[0]
        self.assertEqual(camera_frame.frame.shape, (360, 640, 3))
        self.assertEqual(camera_frame.frame.dtype, np.uint8)
        np.testing.assert_allclose(
            camera_frame.camera_matrix,
            np.array([[320.0, 0.0, 320.0], [0.0, 320.0, 180.0], [0.0, 0.0, 1.0]]),
        )
        np.testing.assert_allclose(camera_frame.dist_coeffs, np.zeros(5))
        self.assertEqual(camera_frame.image_stamp_sec, 9)
        self.assertEqual(camera_frame.image_stamp_nanosec, 876_543_210)
        self.assertEqual(camera_frame.image_received_wall_time, 42.5)
        self.assertIsNone(camera_frame.image_pose_snapshot)
        self.assertIsNone(camera_frame.gazebo_pose)

        kwargs = camera_frame.update_gate_memory_kwargs()
        self.assertIs(kwargs["frame"], camera_frame.frame)
        self.assertIsNone(kwargs["image_pose_snapshot"])
        self.assertIsNone(kwargs["gazebo_pose"])
        self.assertEqual(adapter.stats.completed_frames, 1)
        self.assertEqual(adapter.pending_frame_count, 0)

    def test_payload_size_mismatch_rejects_packet(self):
        packet = struct.pack("<IHHIIQ", 1, 0, 1, 10, 9, 123) + b"1234"

        with self.assertRaises(CompetitionImageAdapterError):
            parse_vision_packet_header(packet)

    def test_invalid_chunk_id_rejects_packet(self):
        packet = pack_vision_packet(
            frame_id=1,
            chunk_id=2,
            total_chunks=2,
            jpeg_size=10,
            payload=b"1234",
            sim_time_ns=123,
        )

        with self.assertRaises(CompetitionImageAdapterError):
            parse_vision_packet_header(packet)

    def test_duplicate_chunk_does_not_emit_frame(self):
        adapter = CompetitionImageAdapter()
        jpeg = make_test_jpeg()
        chunks = split_bytes(jpeg, 2)
        first_packet = pack_vision_packet(
            frame_id=5,
            chunk_id=0,
            total_chunks=2,
            jpeg_size=len(jpeg),
            payload=chunks[0],
            sim_time_ns=100,
        )

        self.assertIsNone(adapter.process_packet(first_packet))
        self.assertIsNone(adapter.process_packet(first_packet))
        self.assertEqual(adapter.stats.duplicate_chunks, 1)
        self.assertEqual(adapter.stats.completed_frames, 0)

    def test_missing_chunk_does_not_emit_frame(self):
        adapter = CompetitionImageAdapter()
        jpeg = make_test_jpeg()
        chunks = split_bytes(jpeg, 3)

        for chunk_id in (0, 2):
            packet = pack_vision_packet(
                frame_id=6,
                chunk_id=chunk_id,
                total_chunks=3,
                jpeg_size=len(jpeg),
                payload=chunks[chunk_id],
                sim_time_ns=100,
            )
            self.assertIsNone(adapter.process_packet(packet))

        self.assertEqual(adapter.stats.completed_frames, 0)
        self.assertEqual(adapter.pending_frame_count, 1)

    def test_corrupt_jpeg_drops_without_emitting_frame(self):
        adapter = CompetitionImageAdapter()
        corrupt = b"not-a-jpeg"
        packet = pack_vision_packet(
            frame_id=8,
            chunk_id=0,
            total_chunks=1,
            jpeg_size=len(corrupt),
            payload=corrupt,
            sim_time_ns=100,
        )

        self.assertIsNone(adapter.process_packet(packet))
        self.assertEqual(adapter.stats.decode_failures, 1)
        self.assertEqual(adapter.stats.completed_frames, 0)
        self.assertEqual(adapter.pending_frame_count, 0)

    def test_wrong_resolution_decode_drops_without_resizing(self):
        def decode_wrong_resolution(_jpeg_bytes):
            return np.zeros((10, 20, 3), dtype=np.uint8)

        adapter = CompetitionImageAdapter(jpeg_decoder=decode_wrong_resolution)
        jpeg = b"fake-jpeg"
        packet = pack_vision_packet(
            frame_id=81,
            chunk_id=0,
            total_chunks=1,
            jpeg_size=len(jpeg),
            payload=jpeg,
            sim_time_ns=100,
        )

        self.assertIsNone(adapter.process_packet(packet))
        self.assertEqual(adapter.stats.decode_failures, 1)
        self.assertEqual(adapter.stats.completed_frames, 0)

    def test_stale_incomplete_frames_are_dropped(self):
        clock = FakeClock(10.0)
        adapter = CompetitionImageAdapter(clock=clock, max_frame_age_s=0.5)
        packet = pack_vision_packet(
            frame_id=9,
            chunk_id=0,
            total_chunks=2,
            jpeg_size=10,
            payload=b"12345",
            sim_time_ns=100,
        )

        self.assertIsNone(adapter.process_packet(packet))
        self.assertEqual(adapter.pending_frame_count, 1)

        clock.value = 10.6
        other_packet = pack_vision_packet(
            frame_id=10,
            chunk_id=0,
            total_chunks=2,
            jpeg_size=10,
            payload=b"12345",
            sim_time_ns=101,
        )
        self.assertIsNone(adapter.process_packet(other_packet))

        self.assertEqual(adapter.pending_frame_count, 1)
        self.assertEqual(adapter.stats.stale_frames_dropped, 1)
        self.assertEqual(adapter.stats.incomplete_frames_dropped, 1)

    def test_incomplete_frame_memory_is_bounded(self):
        adapter = CompetitionImageAdapter(max_incomplete_frames=2)

        for frame_id in (1, 2, 3):
            packet = pack_vision_packet(
                frame_id=frame_id,
                chunk_id=0,
                total_chunks=2,
                jpeg_size=10,
                payload=b"12345",
                sim_time_ns=frame_id,
            )
            self.assertIsNone(adapter.process_packet(packet))

        self.assertEqual(adapter.pending_frame_count, 2)
        self.assertEqual(adapter.stats.incomplete_frames_dropped, 1)


if __name__ == "__main__":
    unittest.main()
