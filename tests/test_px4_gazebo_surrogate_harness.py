import subprocess
import sys
import unittest

import numpy as np

from autonomy_core.perception.competition_image_adapter import (
    CompetitionImageAdapter,
    parse_vision_packet_header,
)
from autonomy_core.runtime.competition_guard import CompetitionGuardError
from autonomy_core.runtime.competition_runner import (
    CompetitionRunnerMode,
    CompetitionRunnerSafetyError,
)
from autonomy_core.runtime.px4_gazebo_surrogate_harness import (
    SURROGATE_LABEL,
    Px4EstimatedTelemetrySample,
    Px4GazeboSurrogateHarness,
    command_result_was_no_send,
    packetize_vadr_jpeg_bytes,
    telemetry_sample_to_fake_mavlink_messages,
)


class FakeClock:
    def __init__(self, value=100.0):
        self.value = float(value)

    def __call__(self):
        return self.value


class FakeAutonomy:
    def __init__(self):
        self.perception_updates = []

    def update_gate_memory_from_frame(self, **kwargs):
        self.perception_updates.append(dict(kwargs))


def sample():
    return Px4EstimatedTelemetrySample(
        position_ned_m=(1.0, 2.0, -3.0),
        velocity_ned_m_s=(0.1, 0.2, -0.3),
        yaw_rad=0.25,
        time_boot_ms=1234,
    )


def image_adapter(clock=None):
    return CompetitionImageAdapter(
        clock=(clock if clock is not None else FakeClock()),
        jpeg_decoder=lambda _jpeg: np.zeros((360, 640, 3), dtype=np.uint8),
    )


class Px4GazeboSurrogateHarnessTests(unittest.TestCase):
    def test_import_does_not_load_live_transport_or_image_dependencies(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import sys; "
                    "import autonomy_core.runtime.px4_gazebo_surrogate_harness; "
                    "print('pymavlink' in sys.modules, 'cv2' in sys.modules, "
                    "'mavsdk' in sys.modules, 'rclpy' in sys.modules)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.stdout.strip(), "False False False False")

    def test_px4_estimated_telemetry_converts_to_runner_fake_messages(self):
        local_position, attitude = telemetry_sample_to_fake_mavlink_messages(sample())

        self.assertEqual(local_position.get_type(), "LOCAL_POSITION_NED")
        self.assertEqual(attitude.get_type(), "ATTITUDE")
        self.assertEqual(local_position.x, 1.0)
        self.assertEqual(local_position.y, 2.0)
        self.assertEqual(local_position.z, -3.0)
        self.assertEqual(local_position.vz, -0.3)
        self.assertEqual(attitude.yaw, 0.25)
        self.assertEqual(attitude.time_boot_ms, 1234)

    def test_gazebo_truth_metadata_is_rejected_before_runner_use(self):
        harness = Px4GazeboSurrogateHarness()
        unsafe = Px4EstimatedTelemetrySample(
            position_ned_m=(1.0, 2.0, -3.0),
            velocity_ned_m_s=(0.0, 0.0, 0.0),
            yaw_rad=0.0,
            metadata={"gazebo_model_pose": [1.0, 2.0, 3.0]},
        )

        with self.assertRaises(CompetitionGuardError):
            harness.telemetry_messages([unsafe])

        self.assertEqual(harness.summary.guard_rejection_count, 1)

    def test_gazebo_truth_vision_metadata_is_rejected_before_packetization(self):
        harness = Px4GazeboSurrogateHarness()

        with self.assertRaises(CompetitionGuardError):
            harness.packetize_jpeg_frames(
                [b"jpeg"],
                metadata={"gazebo_camera_pose": [1.0, 2.0, 3.0]},
            )

        self.assertEqual(harness.summary.guard_rejection_count, 1)
        self.assertEqual(harness.summary.packetization_errors, 1)

    def test_fake_jpeg_packetization_matches_vadr_header_and_image_adapter(self):
        packets = packetize_vadr_jpeg_bytes(
            b"fake-jpeg",
            frame_id=42,
            sim_time_ns=1_234_567_890,
            max_payload_size=4,
        )

        self.assertEqual(len(packets), 3)
        first = parse_vision_packet_header(packets[0])
        self.assertEqual(first.frame_id, 42)
        self.assertEqual(first.chunk_id, 0)
        self.assertEqual(first.total_chunks, 3)
        self.assertEqual(first.jpeg_size, len(b"fake-jpeg"))
        self.assertEqual(first.sim_time_ns, 1_234_567_890)

        adapter = image_adapter()
        emitted = [adapter.process_packet(packet) for packet in packets]
        frames = [frame for frame in emitted if frame is not None]

        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].frame.shape, (360, 640, 3))
        self.assertIsNone(frames[0].gazebo_pose)
        self.assertIsNone(frames[0].image_pose_snapshot)

    def test_observe_surrogate_runs_runner_through_injected_telemetry(self):
        harness = Px4GazeboSurrogateHarness(clock=FakeClock(200.0))

        result = harness.run_observe_surrogate([sample()])

        self.assertEqual(result.mode, CompetitionRunnerMode.OBSERVE)
        self.assertEqual(result.telemetry_messages_processed, 3)
        self.assertTrue(result.heartbeat_seen)
        self.assertTrue(result.state_result.is_usable)
        np.testing.assert_allclose(result.state_result.vehicle_state.pos, [1.0, 2.0, 3.0])
        self.assertFalse(result.command_candidate_attempted)
        summary = harness.summary_dict()
        self.assertEqual(summary["surrogate_label"], SURROGATE_LABEL)
        self.assertFalse(summary["phase4b_satisfied"])
        self.assertFalse(summary["competition_readiness_claimed"])

    def test_vision_dry_run_surrogate_uses_injected_packets_and_autonomy(self):
        clock = FakeClock(300.0)
        harness = Px4GazeboSurrogateHarness(clock=clock)
        autonomy = FakeAutonomy()

        result = harness.run_vision_dry_run_surrogate(
            jpeg_frames=[b"jpeg"],
            autonomy=autonomy,
            image_adapter=image_adapter(clock),
        )

        self.assertEqual(result.mode, CompetitionRunnerMode.VISION_DRY_RUN)
        self.assertEqual(result.vision_packets_processed, 1)
        self.assertEqual(result.vision_frames_completed, 1)
        self.assertEqual(result.perception_update_calls, 1)
        self.assertEqual(len(autonomy.perception_updates), 1)
        self.assertIsNone(autonomy.perception_updates[0]["gazebo_pose"])
        self.assertEqual(harness.summary.frame_count, 1)
        self.assertEqual(harness.summary.completed_packetized_frames, 1)

    def test_command_dry_run_surrogate_builds_no_send_candidate(self):
        harness = Px4GazeboSurrogateHarness(clock=FakeClock(400.0))

        result = harness.run_command_dry_run_surrogate(
            [sample()],
            command_tuple=(0.1, -0.2, 0.3, 0.55),
        )

        self.assertEqual(result.mode, CompetitionRunnerMode.COMMAND_DRY_RUN)
        self.assertTrue(result.command_candidate_attempted)
        self.assertIsNotNone(result.command_result)
        self.assertTrue(result.command_result.accepted)
        self.assertTrue(command_result_was_no_send(result.command_result))
        self.assertFalse(result.command_publication_allowed)
        self.assertIn("phase6a_no_command_publication", result.command_blocked_reasons)
        self.assertIn("phase4b_telemetry_evidence_missing", result.command_blocked_reasons)
        self.assertIn("command_dry_run_no_send", result.command_blocked_reasons)
        summary = harness.summary_dict()
        self.assertEqual(summary["command_candidate_count"], 1)
        self.assertFalse(summary["phase4b_satisfied"])
        self.assertFalse(summary["competition_readiness_claimed"])

    def test_command_live_and_race_remain_fail_closed(self):
        harness = Px4GazeboSurrogateHarness()

        for mode in (CompetitionRunnerMode.COMMAND_LIVE, CompetitionRunnerMode.RACE):
            with self.subTest(mode=mode):
                with self.assertRaises(CompetitionRunnerSafetyError):
                    harness.build_runner(mode=mode)


if __name__ == "__main__":
    unittest.main()
