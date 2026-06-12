import math
import subprocess
import sys
import unittest

import numpy as np

from autonomy_core.perception.competition_image_adapter import (
    CompetitionImageAdapter,
    pack_vision_packet,
)
from autonomy_core.runtime.competition_guard import GAZEBO_TRUTH_POSE_SOURCE
from autonomy_core.runtime.competition_guard import (
    CompetitionGuard,
    CompetitionGuardError,
)
from autonomy_core.runtime.competition_runner import (
    CompetitionRunner,
    CompetitionRunnerConfig,
    CompetitionRunnerMode,
    CompetitionRunnerSafetyConfig,
    CompetitionRunnerSafetyError,
)


class FakeMessage:
    def __init__(self, message_type, message_id=0, **fields):
        self._message_type = message_type
        self._message_id = message_id
        for key, value in fields.items():
            setattr(self, key, value)

    def get_type(self):
        return self._message_type

    def get_msgId(self):
        return self._message_id

    def get_srcSystem(self):
        return 1

    def get_srcComponent(self):
        return 1

    def to_dict(self):
        return {
            "mavpackettype": self._message_type,
            **{
                key: value
                for key, value in vars(self).items()
                if not key.startswith("_")
            },
        }


class FakeMavlinkTransport:
    def __init__(self, messages):
        self.messages = list(messages)
        self.calls = 0

    def receive_messages(self):
        self.calls += 1
        messages = self.messages
        self.messages = []
        return messages


class FakeVisionTransport:
    def __init__(self, packets):
        self.packets = list(packets)
        self.calls = 0

    def receive_packets(self):
        self.calls += 1
        packets = self.packets
        self.packets = []
        return packets


class FakeFrame:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)

    def update_gate_memory_kwargs(self):
        return dict(self.kwargs)


class FakeImageAdapter:
    def __init__(self, frame):
        self.frame = frame

    def process_packet(self, _packet):
        return self.frame


class FakeAutonomy:
    def __init__(self):
        self.perception_updates = []
        self.attitude_control_calls = 0

    def update_gate_memory_from_frame(self, **kwargs):
        self.perception_updates.append(kwargs)

    def attitude_control(self):
        self.attitude_control_calls += 1
        return (0.1, -0.2, 0.3, 0.55)


class FakeClock:
    def __init__(self, start=10.0):
        self.now = float(start)

    def __call__(self):
        return self.now

    def set(self, value):
        self.now = float(value)


def local_position_message():
    return FakeMessage(
        "LOCAL_POSITION_NED",
        32,
        x=1.0,
        y=2.0,
        z=-3.0,
        vx=0.1,
        vy=0.2,
        vz=-0.3,
        time_boot_ms=1000,
    )


def attitude_message():
    return FakeMessage(
        "ATTITUDE",
        30,
        roll=0.0,
        pitch=0.0,
        yaw=0.25,
        rollspeed=0.0,
        pitchspeed=0.0,
        yawspeed=0.0,
        time_boot_ms=1000,
    )


class CompetitionRunnerSkeletonTests(unittest.TestCase):
    def test_import_does_not_import_pymavlink_or_cv2(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import sys; "
                    "import autonomy_core.runtime.competition_runner; "
                    "print('pymavlink' in sys.modules, 'cv2' in sys.modules)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.stdout.strip(), "False False")

    def test_live_command_modes_are_fail_closed(self):
        for mode in (CompetitionRunnerMode.COMMAND_LIVE, CompetitionRunnerMode.RACE):
            with self.subTest(mode=mode):
                with self.assertRaises(CompetitionRunnerSafetyError):
                    CompetitionRunner(config=CompetitionRunnerConfig(mode=mode))

    def test_live_command_modes_reject_gazebo_truth_before_fail_closed_mode(self):
        for mode in (CompetitionRunnerMode.COMMAND_LIVE, CompetitionRunnerMode.RACE):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(
                    CompetitionRunnerSafetyError,
                    "gazebo_truth_sim_only",
                ):
                    CompetitionRunner(
                        config=CompetitionRunnerConfig(
                            mode=mode,
                            perception_world_pose_source=GAZEBO_TRUTH_POSE_SOURCE,
                        )
                    )

    def test_command_publication_flag_is_rejected_even_in_dry_run_mode(self):
        with self.assertRaises(CompetitionRunnerSafetyError):
            CompetitionRunner(
                config=CompetitionRunnerConfig(
                    mode=CompetitionRunnerMode.COMMAND_DRY_RUN,
                    safety=CompetitionRunnerSafetyConfig(
                        command_publication_enabled=True,
                    ),
                )
            )

    def test_startup_metadata_rejects_gazebo_truth_paths(self):
        with self.assertRaisesRegex(
            CompetitionRunnerSafetyError,
            "Gazebo truth field",
        ):
            CompetitionRunner(
                config=CompetitionRunnerConfig(
                    mode=CompetitionRunnerMode.COMMAND_DRY_RUN,
                    startup_metadata={
                        "image_metadata": {
                            "gazebo_camera_pos_world": [1.0, 2.0, 3.0],
                        }
                    },
                )
            )

    def test_noncompetition_guard_allows_gazebo_diagnostics_at_runner_boundary(self):
        runner = CompetitionRunner(
            config=CompetitionRunnerConfig(
                mode=CompetitionRunnerMode.OBSERVE,
                perception_world_pose_source=GAZEBO_TRUTH_POSE_SOURCE,
                startup_metadata={"use_diagnostic_far_depth_correction": True},
            ),
            guard=CompetitionGuard(competition_mode=False),
        )

        result = runner.step()

        self.assertFalse(result.command_publication_allowed)
        self.assertIn("observe_mode_no_commands", result.command_blocked_reasons)

    def test_gazebo_truth_source_is_rejected_at_startup(self):
        with self.assertRaises(CompetitionRunnerSafetyError):
            CompetitionRunner(
                config=CompetitionRunnerConfig(
                    mode=CompetitionRunnerMode.OBSERVE,
                    perception_world_pose_source=GAZEBO_TRUTH_POSE_SOURCE,
                )
            )

    def test_observe_mode_processes_fake_telemetry_without_commands_or_vision(self):
        clock = FakeClock(100.0)
        transport = FakeMavlinkTransport(
            [
                FakeMessage("HEARTBEAT", 0, base_mode=0),
                local_position_message(),
                attitude_message(),
            ]
        )
        autonomy = FakeAutonomy()
        runner = CompetitionRunner(
            config=CompetitionRunnerConfig(mode=CompetitionRunnerMode.OBSERVE),
            mavlink_transport=transport,
            autonomy=autonomy,
            clock=clock,
        )

        result = runner.step()

        self.assertEqual(result.telemetry_messages_processed, 3)
        self.assertTrue(result.heartbeat_seen)
        self.assertEqual(result.heartbeat_age_s, 0.0)
        self.assertTrue(result.state_result.is_usable)
        np.testing.assert_allclose(result.state_result.vehicle_state.pos, [1.0, 2.0, 3.0])
        self.assertFalse(result.command_candidate_attempted)
        self.assertIsNone(result.command_result)
        self.assertFalse(result.command_publication_allowed)
        self.assertEqual(autonomy.attitude_control_calls, 0)
        self.assertEqual(autonomy.perception_updates, [])
        self.assertEqual(runner.stats.command_publications_sent, 0)
        self.assertEqual(transport.calls, 1)
        self.assertTrue(runner.telemetry_summary()["local_position_ned_available"])

    def test_vision_dry_run_uses_fake_packets_and_fake_autonomy_only(self):
        clock = FakeClock(200.0)
        fake_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        image_adapter = CompetitionImageAdapter(
            clock=clock,
            jpeg_decoder=lambda _jpeg: fake_frame,
        )
        packet = pack_vision_packet(
            frame_id=1,
            chunk_id=0,
            total_chunks=1,
            jpeg_size=4,
            payload=b"jpeg",
            sim_time_ns=1_234_567_890,
        )
        vision_transport = FakeVisionTransport([packet])
        autonomy = FakeAutonomy()
        runner = CompetitionRunner(
            config=CompetitionRunnerConfig(mode=CompetitionRunnerMode.VISION_DRY_RUN),
            image_adapter=image_adapter,
            vision_transport=vision_transport,
            autonomy=autonomy,
            clock=clock,
        )

        result = runner.step()

        self.assertEqual(result.vision_packets_processed, 1)
        self.assertEqual(result.vision_frames_completed, 1)
        self.assertEqual(result.perception_update_calls, 1)
        self.assertEqual(len(autonomy.perception_updates), 1)
        update_kwargs = autonomy.perception_updates[0]
        self.assertIsNone(update_kwargs["gazebo_pose"])
        self.assertIsNone(update_kwargs["image_pose_snapshot"])
        self.assertFalse(result.command_candidate_attempted)
        self.assertEqual(vision_transport.calls, 1)

    def test_vision_dry_run_rejects_non_none_gazebo_pose_before_fake_autonomy(self):
        autonomy = FakeAutonomy()
        image_adapter = FakeImageAdapter(
            FakeFrame(
                frame="frame",
                gazebo_pose={"gazebo_model_pos_world": [1.0, 2.0, 3.0]},
                image_pose_snapshot=None,
            )
        )
        runner = CompetitionRunner(
            config=CompetitionRunnerConfig(mode=CompetitionRunnerMode.VISION_DRY_RUN),
            image_adapter=image_adapter,
            autonomy=autonomy,
            clock=FakeClock(225.0),
        )

        with self.assertRaises(CompetitionGuardError):
            runner.step(vision_packets=[b"fake"])

        self.assertEqual(autonomy.perception_updates, [])

    def test_vision_dry_run_rejects_gazebo_metadata_before_fake_autonomy(self):
        autonomy = FakeAutonomy()
        image_adapter = FakeImageAdapter(
            FakeFrame(
                frame="frame",
                gazebo_pose=None,
                image_pose_snapshot=None,
                image_metadata={"gazebo_tf": {"map": "base_link"}},
            )
        )
        runner = CompetitionRunner(
            config=CompetitionRunnerConfig(mode=CompetitionRunnerMode.VISION_DRY_RUN),
            image_adapter=image_adapter,
            autonomy=autonomy,
            clock=FakeClock(226.0),
        )

        with self.assertRaises(CompetitionGuardError):
            runner.step(vision_packets=[b"fake"])

        self.assertEqual(autonomy.perception_updates, [])

    def test_command_dry_run_builds_candidate_but_publication_stays_blocked(self):
        clock = FakeClock(300.0)
        autonomy = FakeAutonomy()
        runner = CompetitionRunner(
            config=CompetitionRunnerConfig(mode=CompetitionRunnerMode.COMMAND_DRY_RUN),
            autonomy=autonomy,
            clock=clock,
        )

        result = runner.step(
            telemetry_messages=[
                FakeMessage("HEARTBEAT", 0, base_mode=0),
                local_position_message(),
                attitude_message(),
            ]
        )

        self.assertTrue(result.command_candidate_attempted)
        self.assertIsNotNone(result.command_result)
        self.assertTrue(result.command_result.accepted)
        self.assertIsNotNone(result.command_result.fields)
        self.assertFalse(result.command_result.fields.send_ready)
        self.assertFalse(result.command_publication_allowed)
        self.assertIn("phase6a_no_command_publication", result.command_blocked_reasons)
        self.assertIn("phase4b_telemetry_evidence_missing", result.command_blocked_reasons)
        self.assertIn("command_dry_run_no_send", result.command_blocked_reasons)
        self.assertEqual(autonomy.attitude_control_calls, 1)
        self.assertEqual(runner.stats.command_candidate_attempts, 1)
        self.assertEqual(runner.stats.command_candidates_accepted, 1)
        self.assertEqual(runner.stats.command_publications_attempted, 0)
        self.assertEqual(runner.stats.command_publications_sent, 0)

    def test_command_dry_run_processes_fake_telemetry_image_and_candidate_no_send(self):
        clock = FakeClock(350.0)
        fake_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        image_adapter = CompetitionImageAdapter(
            clock=clock,
            jpeg_decoder=lambda _jpeg: fake_frame,
        )
        packet = pack_vision_packet(
            frame_id=3,
            chunk_id=0,
            total_chunks=1,
            jpeg_size=4,
            payload=b"jpeg",
            sim_time_ns=2_000_000_001,
        )
        autonomy = FakeAutonomy()
        runner = CompetitionRunner(
            config=CompetitionRunnerConfig(mode=CompetitionRunnerMode.COMMAND_DRY_RUN),
            image_adapter=image_adapter,
            autonomy=autonomy,
            clock=clock,
        )

        result = runner.step(
            telemetry_messages=[
                FakeMessage("HEARTBEAT", 0, base_mode=0),
                local_position_message(),
                attitude_message(),
            ],
            vision_packets=[packet],
        )

        self.assertEqual(result.telemetry_messages_processed, 3)
        self.assertEqual(result.vision_packets_processed, 1)
        self.assertEqual(result.vision_frames_completed, 1)
        self.assertEqual(result.perception_update_calls, 1)
        self.assertEqual(len(autonomy.perception_updates), 1)
        self.assertTrue(result.command_candidate_attempted)
        self.assertIsNotNone(result.command_result)
        self.assertTrue(result.command_result.accepted)
        self.assertFalse(result.command_result.fields.send_ready)
        self.assertFalse(result.command_publication_allowed)
        self.assertIn("phase6a_no_command_publication", result.command_blocked_reasons)
        self.assertIn("phase4b_telemetry_evidence_missing", result.command_blocked_reasons)
        self.assertIn("command_dry_run_no_send", result.command_blocked_reasons)
        self.assertEqual(runner.stats.command_publications_attempted, 0)
        self.assertEqual(runner.stats.command_publications_sent, 0)

    def test_command_dry_run_does_not_call_autonomy_without_usable_state(self):
        autonomy = FakeAutonomy()
        runner = CompetitionRunner(
            config=CompetitionRunnerConfig(mode=CompetitionRunnerMode.COMMAND_DRY_RUN),
            autonomy=autonomy,
            clock=FakeClock(400.0),
        )

        result = runner.step(telemetry_messages=[FakeMessage("HEARTBEAT", 0)])

        self.assertFalse(result.command_candidate_attempted)
        self.assertIsNone(result.command_result)
        self.assertEqual(autonomy.attitude_control_calls, 0)
        self.assertIn("missing_local_position_ned", result.command_blocked_reasons)
        self.assertIn("command_candidate_blocked_invalid_state", result.events)

    def test_invalid_injected_transport_is_rejected_without_socket_fallback(self):
        runner = CompetitionRunner(
            config=CompetitionRunnerConfig(mode=CompetitionRunnerMode.OBSERVE),
            mavlink_transport=object(),
        )

        with self.assertRaises(CompetitionRunnerSafetyError):
            runner.step()

    def test_vision_packets_are_ignored_in_observe_mode(self):
        runner = CompetitionRunner(
            config=CompetitionRunnerConfig(mode=CompetitionRunnerMode.OBSERVE),
            clock=FakeClock(500.0),
        )

        result = runner.step(vision_packets=[b"not parsed in observe"])

        self.assertEqual(result.vision_packets_processed, 0)
        self.assertIn("vision_packets_ignored_in_mode", result.events)


if __name__ == "__main__":
    unittest.main()
