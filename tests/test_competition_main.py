import json
import subprocess
import sys
import unittest

from autonomy_core.runtime.competition_main import (
    CompetitionMainConfig,
    CompetitionMainSafetyError,
    PHASE_6E,
    run_competition_main,
)
from autonomy_core.runtime.competition_runner import CompetitionRunnerMode
from autonomy_core.runtime.competition_setup import CompetitionSetupConfig, build_competition_runtime


class FakeMessage:
    def __init__(self, message_type, **fields):
        self._message_type = message_type
        for key, value in fields.items():
            setattr(self, key, value)

    def get_type(self):
        return self._message_type

    def get_msgId(self):
        return 0

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
    def __init__(self, messages=()):
        self.messages = list(messages)
        self.close_calls = 0

    def receive_messages(self):
        messages = self.messages
        self.messages = []
        return messages

    def close(self):
        self.close_calls += 1

    def summary(self):
        return {"fake_mavlink": True, "remaining": len(self.messages)}


class FakeVisionTransport:
    def __init__(self, packets=()):
        self.packets = list(packets)
        self.close_calls = 0

    def receive_packets(self):
        packets = self.packets
        self.packets = []
        return packets

    def close(self):
        self.close_calls += 1

    def summary(self):
        return {"fake_vision": True, "remaining": len(self.packets)}


class FakeFrame:
    def update_gate_memory_kwargs(self):
        return {
            "frame": object(),
            "camera_matrix": object(),
            "dist_coeffs": object(),
            "image_stamp_sec": 1,
            "image_stamp_nanosec": 2,
            "image_received_wall_time": 3.0,
            "image_pose_snapshot": None,
            "gazebo_pose": None,
        }


class FakeImageAdapter:
    def __init__(self):
        self.packet_count = 0

    def process_packet(self, _packet):
        self.packet_count += 1
        return FakeFrame()


class FakeAutonomy:
    def __init__(self):
        self.perception_updates = []
        self.attitude_control_calls = 0

    def update_gate_memory_from_frame(self, **kwargs):
        self.perception_updates.append(kwargs)

    def attitude_control(self):
        self.attitude_control_calls += 1
        return (0.0, 0.0, 0.0, 0.5)


def usable_messages():
    return [
        FakeMessage("HEARTBEAT", base_mode=0),
        FakeMessage(
            "LOCAL_POSITION_NED",
            x=1.0,
            y=2.0,
            z=-3.0,
            vx=0.1,
            vy=0.2,
            vz=-0.3,
            time_boot_ms=1000,
        ),
        FakeMessage(
            "ATTITUDE",
            roll=0.0,
            pitch=0.0,
            yaw=0.25,
            rollspeed=0.0,
            pitchspeed=0.0,
            yawspeed=0.0,
            time_boot_ms=1000,
        ),
    ]


def components_for_mode(mode, *, with_vision=False, autonomy=None):
    return build_competition_runtime(
        CompetitionSetupConfig(mode=mode),
        mavlink_transport=FakeMavlinkTransport(usable_messages()),
        vision_transport=FakeVisionTransport([b"packet"] if with_vision else []),
        image_adapter=FakeImageAdapter(),
        autonomy=autonomy,
        clock=lambda: 100.0,
    )


class CompetitionMainTests(unittest.TestCase):
    def test_import_does_not_load_live_dependencies_or_autonomy(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import sys; "
                    "import autonomy_core.runtime.competition_main; "
                    "print('pymavlink' in sys.modules, 'cv2' in sys.modules, "
                    "'mavsdk' in sys.modules, 'rclpy' in sys.modules, "
                    "'autonomy_core.launch.autonomy_api6' in sys.modules)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.stdout.strip(), "False False False False False")

    def test_observe_runs_bounded_loop_with_fake_components(self):
        components = components_for_mode(CompetitionRunnerMode.OBSERVE)

        summary = run_competition_main(
            CompetitionMainConfig(mode=CompetitionRunnerMode.OBSERVE),
            components=components,
            clock=lambda: 100.0,
        ).to_dict()

        self.assertEqual(summary["status"], "dry_run_complete")
        self.assertEqual(summary["mode"], "observe")
        self.assertEqual(summary["steps_completed"], 1)
        self.assertEqual(summary["telemetry_messages_processed"], 3)
        self.assertTrue(summary["heartbeat_seen"])
        self.assertTrue(summary["state_usable"])
        self.assertEqual(summary["vision_packets_processed"], 0)
        self.assertEqual(summary["command_candidate_count"], 0)
        self.assertFalse(summary["command_publication_allowed"])
        self.assertEqual(summary["command_sent_count"], 0)
        self.assertFalse(summary["phase6e_satisfied"])
        self.assertFalse(summary["phase6e_receive_satisfied"])
        self.assertFalse(summary["competition_readiness_claimed"])

    def test_vision_dry_run_processes_fake_vision_without_send(self):
        autonomy = FakeAutonomy()
        components = components_for_mode(
            CompetitionRunnerMode.VISION_DRY_RUN,
            with_vision=True,
            autonomy=autonomy,
        )

        summary = run_competition_main(
            CompetitionMainConfig(mode=CompetitionRunnerMode.VISION_DRY_RUN),
            components=components,
            clock=lambda: 100.0,
        ).to_dict()

        self.assertEqual(summary["mode"], "vision_dry_run")
        self.assertEqual(summary["vision_packets_processed"], 1)
        self.assertEqual(summary["vision_frames_completed"], 1)
        self.assertEqual(summary["perception_update_calls"], 1)
        self.assertEqual(len(autonomy.perception_updates), 1)
        self.assertEqual(summary["command_sent_count"], 0)

    def test_phase6e_receive_verdict_passes_with_live_vision_transport_evidence(self):
        components = components_for_mode(
            CompetitionRunnerMode.VISION_DRY_RUN,
            with_vision=True,
            autonomy=None,
        )

        summary = run_competition_main(
            CompetitionMainConfig(
                mode=CompetitionRunnerMode.VISION_DRY_RUN,
                live_transports=True,
                evidence_label="px4_gazebo_surrogate",
            ),
            components=components,
            clock=lambda: 100.0,
        ).to_dict()

        self.assertEqual(summary["phase"], PHASE_6E)
        self.assertEqual(summary["evidence_label"], "px4_gazebo_surrogate")
        self.assertTrue(summary["live_transports_requested"])
        self.assertTrue(summary["phase6e_receive_satisfied"])
        self.assertTrue(summary["phase6e_satisfied"])
        self.assertFalse(summary["phase6e_perception_boundary_satisfied"])
        self.assertEqual(summary["telemetry_messages_processed"], 3)
        self.assertTrue(summary["heartbeat_seen"])
        self.assertTrue(summary["state_usable"])
        self.assertEqual(summary["vision_packets_processed"], 1)
        self.assertEqual(summary["vision_frames_completed"], 1)
        self.assertEqual(summary["perception_update_calls"], 0)
        self.assertFalse(summary["command_publication_allowed"])
        self.assertEqual(summary["command_sent_count"], 0)
        self.assertFalse(summary["phase4b_satisfied"])
        self.assertFalse(summary["competition_readiness_claimed"])
        self.assertTrue(
            summary["phase6e_success_criteria"]["vision_packets_processed_gt_0"]
        )
        self.assertTrue(
            summary["phase6e_success_criteria"]["vision_frames_completed_gt_0"]
        )
        self.assertFalse(
            summary["phase6e_success_criteria"]["perception_update_calls_gt_0"]
        )

    def test_phase6e_perception_boundary_verdict_requires_autonomy_update(self):
        autonomy = FakeAutonomy()
        components = components_for_mode(
            CompetitionRunnerMode.VISION_DRY_RUN,
            with_vision=True,
            autonomy=autonomy,
        )

        summary = run_competition_main(
            CompetitionMainConfig(
                mode=CompetitionRunnerMode.VISION_DRY_RUN,
                live_transports=True,
                evidence_label="fake_live_transport",
            ),
            components=components,
            clock=lambda: 100.0,
        ).to_dict()

        self.assertTrue(summary["phase6e_receive_satisfied"])
        self.assertTrue(summary["phase6e_perception_boundary_satisfied"])
        self.assertEqual(summary["perception_update_calls"], 1)
        self.assertEqual(len(autonomy.perception_updates), 1)
        self.assertIsNone(autonomy.perception_updates[0]["gazebo_pose"])
        self.assertIsNone(autonomy.perception_updates[0]["image_pose_snapshot"])

    def test_command_dry_run_builds_candidate_but_sends_nothing(self):
        components = components_for_mode(
            CompetitionRunnerMode.COMMAND_DRY_RUN,
            autonomy=FakeAutonomy(),
        )

        summary = run_competition_main(
            CompetitionMainConfig(mode=CompetitionRunnerMode.COMMAND_DRY_RUN),
            components=components,
            clock=lambda: 100.0,
        ).to_dict()

        self.assertEqual(summary["mode"], "command_dry_run")
        self.assertEqual(summary["command_candidate_count"], 1)
        self.assertFalse(summary["command_publication_allowed"])
        self.assertEqual(summary["command_sent_count"], 0)
        self.assertIn("command_dry_run_no_send", summary["command_blocked_reasons"])

    def test_live_command_and_race_fail_closed(self):
        for mode in (CompetitionRunnerMode.COMMAND_LIVE, CompetitionRunnerMode.RACE):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(CompetitionMainSafetyError, "fail-closed"):
                    run_competition_main(
                        CompetitionMainConfig(mode=mode),
                        components=components_for_mode(CompetitionRunnerMode.OBSERVE),
                    )

    def test_cli_without_live_transports_fails_closed_before_opening_sockets(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "autonomy_core.runtime.competition_main",
                "observe",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        data = json.loads(completed.stdout)

        self.assertEqual(completed.returncode, 2)
        self.assertEqual(data["status"], "fail_closed")
        self.assertIn("--live-transports", data["safety_error"])
        self.assertEqual(data["command_sent_count"], 0)
        self.assertFalse(data["competition_readiness_claimed"])

    def test_cli_command_live_fails_closed(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "autonomy_core.runtime.competition_main",
                "command_live",
                "--live-transports",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        data = json.loads(completed.stdout)

        self.assertEqual(completed.returncode, 2)
        self.assertEqual(data["status"], "fail_closed")
        self.assertIn("fail-closed", data["safety_error"])
        self.assertEqual(data["command_sent_count"], 0)


if __name__ == "__main__":
    unittest.main()
