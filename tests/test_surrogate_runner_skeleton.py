import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import autonomy_core.runtime.surrogate_runner as surrogate_runner_module
from autonomy_core.runtime.surrogate_runner import (
    EXECUTABLE_STAGE_8_5D_2_MODES,
    EXECUTABLE_STAGE_8_5D_3_MODES,
    FAIL_CLOSED_MODES,
    PHASE_8_5D_1,
    PHASE_8_5D_2,
    PHASE_8_5D_3,
    SUPPORTED_STAGE_8_5D_1_MODES,
    SURROGATE_LABEL,
    SurrogateRunner,
    SurrogateRunnerConfig,
    SurrogateRunnerMode,
    SurrogateRunnerSafetyError,
    normalize_surrogate_mode,
    receive_mavlink_messages,
)
from autonomy_core.tools.competition_vision_udp_loopback import (
    build_mock_image,
    save_image,
)


class FakeMavlinkMessage:
    def __init__(self, message_type, msg_id=0, **fields):
        self._message_type = message_type
        self._msg_id = msg_id
        self._fields = dict(fields)
        for key, value in fields.items():
            setattr(self, key, value)

    def get_type(self):
        return self._message_type

    def get_msgId(self):
        return self._msg_id

    def get_srcSystem(self):
        return 1

    def get_srcComponent(self):
        return 1

    def to_dict(self):
        return dict(self._fields)


def usable_px4_messages():
    return (
        FakeMavlinkMessage("HEARTBEAT", msg_id=0),
        FakeMavlinkMessage(
            "LOCAL_POSITION_NED",
            msg_id=32,
            time_boot_ms=1000,
            x=1.0,
            y=2.0,
            z=-3.0,
            vx=0.1,
            vy=0.2,
            vz=-0.3,
        ),
        FakeMavlinkMessage(
            "ATTITUDE",
            msg_id=30,
            time_boot_ms=1000,
            roll=0.0,
            pitch=0.0,
            yaw=1.25,
            rollspeed=0.0,
            pitchspeed=0.0,
            yawspeed=0.0,
        ),
    )


class FakeReceiveOnlyConnection:
    def __init__(self, messages):
        self.messages = list(messages)
        self.recv_calls = 0
        self.send_calls = 0

    def recv_match(self, *, blocking):
        self.recv_calls += 1
        if self.messages:
            return self.messages.pop(0)
        return None

    def send(self, *_args, **_kwargs):
        self.send_calls += 1
        raise AssertionError("surrogate receive helper must not send")


class SurrogateRunnerSkeletonTests(unittest.TestCase):
    def test_import_does_not_load_live_dependencies_or_autonomy(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import sys; "
                    "import autonomy_core.runtime.surrogate_runner; "
                    "print('pymavlink' in sys.modules, 'mavsdk' in sys.modules, "
                    "'rclpy' in sys.modules, 'cv2' in sys.modules, "
                    "'autonomy_core.launch.autonomy_api6' in sys.modules)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.stdout.strip(), "False False False False False")

    def test_all_supported_safe_modes_are_executable_by_phase_8_5d_3(self):
        self.assertFalse(SUPPORTED_STAGE_8_5D_1_MODES - EXECUTABLE_STAGE_8_5D_3_MODES)
        self.assertTrue(EXECUTABLE_STAGE_8_5D_2_MODES < EXECUTABLE_STAGE_8_5D_3_MODES)

    def test_receive_mavlink_messages_with_injected_connection_is_receive_only(self):
        connection = FakeReceiveOnlyConnection(usable_px4_messages())

        messages, summary = receive_mavlink_messages(
            endpoint="injected",
            duration_s=10.0,
            max_messages=3,
            connection=connection,
            clock=lambda: 1.0,
            sleep=lambda _seconds: None,
        )

        self.assertEqual(len(messages), 3)
        self.assertEqual(connection.recv_calls, 3)
        self.assertEqual(connection.send_calls, 0)
        self.assertTrue(summary["receive_only"])
        self.assertTrue(summary["local_position_ned_available"])
        self.assertTrue(summary["attitude_available"])

    def test_mock_vision_dry_run_executes_competition_runner_no_send(self):
        runner = SurrogateRunner(
            SurrogateRunnerConfig(
                mode=SurrogateRunnerMode.MOCK_VISION_DRY_RUN,
                max_payload_size=4096,
            )
        )

        data = runner.run().to_dict()

        self.assertEqual(data["phase"], PHASE_8_5D_2)
        self.assertEqual(data["mode"], "mock_vision_dry_run")
        self.assertEqual(data["source_kind"], "generated_mock")
        self.assertEqual(data["status"], "competition_stack_vision_dry_run_complete")
        self.assertTrue(data["competition_runner_executed"])
        self.assertTrue(data["fake_autonomy_used"])
        self.assertFalse(data["autonomy_instantiated"])
        self.assertFalse(data["sockets_opened"])
        self.assertFalse(data["command_publication_allowed"])
        self.assertEqual(data["command_sent_count"], 0)
        self.assertEqual(data["frame_count"], 1)
        self.assertEqual(data["completed_packetized_frames"], 1)
        self.assertGreater(data["vision_packets_processed"], 0)
        self.assertEqual(data["vision_frames_completed"], 1)
        self.assertEqual(data["perception_update_calls"], 1)
        self.assertEqual(data["image_shape"], [360, 640, 3])
        self.assertEqual(data["image_dtype"], "uint8")
        self.assertEqual(data["image_stamp_sec"], 1)
        self.assertEqual(data["image_stamp_nanosec"], 234567890)
        self.assertEqual(
            data["camera_matrix"],
            [[320.0, 0.0, 320.0], [0.0, 320.0, 180.0], [0.0, 0.0, 1.0]],
        )
        self.assertEqual(data["dist_coeffs"], [0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertIsNone(data["gazebo_pose"])
        self.assertIsNone(data["image_pose_snapshot"])
        self.assertFalse(data["phase4b_satisfied"])
        self.assertFalse(data["competition_readiness_claimed"])
        self.assertIn("vision_dry_run_mode_no_commands", data["command_blocked_reasons"])

    def test_saved_image_vision_dry_run_executes_with_saved_fixture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "fixture.jpg"
            save_image(image_path, build_mock_image())
            runner = SurrogateRunner(
                SurrogateRunnerConfig(
                    mode=SurrogateRunnerMode.SAVED_IMAGE_VISION_DRY_RUN,
                    input_image_path=str(image_path),
                    source_kind="saved_fixture",
                    max_payload_size=4096,
                )
            )

            data = runner.run().to_dict()

        self.assertEqual(data["phase"], PHASE_8_5D_2)
        self.assertEqual(data["mode"], "saved_image_vision_dry_run")
        self.assertEqual(data["source_kind"], "saved_fixture")
        self.assertEqual(data["source_image_path"], str(image_path))
        self.assertEqual(data["status"], "competition_stack_vision_dry_run_complete")
        self.assertTrue(data["competition_runner_executed"])
        self.assertEqual(data["vision_frames_completed"], 1)
        self.assertEqual(data["perception_update_calls"], 1)
        self.assertEqual(data["image_shape"], [360, 640, 3])
        self.assertFalse(data["command_publication_allowed"])
        self.assertEqual(data["command_sent_count"], 0)

    def test_px4_observe_executes_competition_runner_with_injected_telemetry(self):
        runner = SurrogateRunner(
            SurrogateRunnerConfig(
                mode=SurrogateRunnerMode.PX4_OBSERVE,
                injected_mavlink_messages=usable_px4_messages(),
            )
        )

        data = runner.run().to_dict()

        self.assertEqual(data["phase"], PHASE_8_5D_3)
        self.assertEqual(data["mode"], "px4_observe")
        self.assertEqual(data["status"], "competition_stack_px4_observe_complete")
        self.assertTrue(data["competition_runner_executed"])
        self.assertFalse(data["fake_autonomy_used"])
        self.assertFalse(data["sockets_opened"])
        self.assertFalse(data["pymavlink_connected"])
        self.assertEqual(data["telemetry_sample_count"], 3)
        self.assertEqual(
            data["telemetry_message_types"],
            ["HEARTBEAT", "LOCAL_POSITION_NED", "ATTITUDE"],
        )
        self.assertTrue(data["heartbeat_seen"])
        self.assertTrue(data["state_usable"])
        self.assertEqual(data["position_source"], "LOCAL_POSITION_NED")
        self.assertEqual(data["attitude_source"], "ATTITUDE")
        self.assertFalse(data["command_publication_allowed"])
        self.assertEqual(data["command_sent_count"], 0)
        self.assertIn("observe_mode_no_commands", data["command_blocked_reasons"])
        self.assertFalse(data["phase4b_satisfied"])
        self.assertFalse(data["competition_readiness_claimed"])

    def test_px4_vision_dry_run_pairs_injected_telemetry_with_mock_vision(self):
        runner = SurrogateRunner(
            SurrogateRunnerConfig(
                mode=SurrogateRunnerMode.PX4_VISION_DRY_RUN,
                vision_source_kind="generated_mock",
                injected_mavlink_messages=usable_px4_messages(),
                max_payload_size=4096,
            )
        )

        data = runner.run().to_dict()

        self.assertEqual(data["phase"], PHASE_8_5D_3)
        self.assertEqual(data["mode"], "px4_vision_dry_run")
        self.assertEqual(data["status"], "competition_stack_px4_vision_dry_run_complete")
        self.assertEqual(data["vision_source_kind"], "generated_mock")
        self.assertTrue(data["state_usable"])
        self.assertEqual(data["frame_count"], 1)
        self.assertEqual(data["vision_frames_completed"], 1)
        self.assertEqual(data["perception_update_calls"], 1)
        self.assertEqual(data["image_shape"], [360, 640, 3])
        self.assertIsNone(data["gazebo_pose"])
        self.assertIsNone(data["image_pose_snapshot"])
        self.assertFalse(data["command_publication_allowed"])
        self.assertEqual(data["command_sent_count"], 0)

    def test_px4_vision_dry_run_resizes_fake_ros_camera_frame_when_explicit(self):
        original_capture = surrogate_runner_module._capture_ros_camera_frame_as_bgr
        fake_frame = np.zeros((960, 1280, 3), dtype=np.uint8)
        surrogate_runner_module._capture_ros_camera_frame_as_bgr = (
            lambda *, topic, duration_s: fake_frame
        )
        try:
            runner = SurrogateRunner(
                SurrogateRunnerConfig(
                    mode=SurrogateRunnerMode.PX4_VISION_DRY_RUN,
                    vision_source_kind="ros_camera",
                    injected_mavlink_messages=usable_px4_messages(),
                    resize_camera_to_competition=True,
                    max_payload_size=4096,
                )
            )

            data = runner.run().to_dict()
        finally:
            surrogate_runner_module._capture_ros_camera_frame_as_bgr = original_capture

        self.assertEqual(data["phase"], PHASE_8_5D_3)
        self.assertEqual(data["mode"], "px4_vision_dry_run")
        self.assertEqual(data["source_kind"], "ros_camera_pixels_only")
        self.assertEqual(data["vision_source_kind"], "ros_camera")
        self.assertEqual(data["image_shape"], [360, 640, 3])
        self.assertEqual(data["vision_frames_completed"], 1)
        self.assertEqual(data["perception_update_calls"], 1)
        self.assertFalse(data["command_publication_allowed"])
        self.assertEqual(data["command_sent_count"], 0)
        self.assertFalse(data["phase4b_satisfied"])
        self.assertFalse(data["competition_readiness_claimed"])

    def test_px4_vision_dry_run_rejects_mismatched_ros_camera_without_resize_flag(self):
        original_capture = surrogate_runner_module._capture_ros_camera_frame_as_bgr
        fake_frame = np.zeros((960, 1280, 3), dtype=np.uint8)
        surrogate_runner_module._capture_ros_camera_frame_as_bgr = (
            lambda *, topic, duration_s: fake_frame
        )
        try:
            runner = SurrogateRunner(
                SurrogateRunnerConfig(
                    mode=SurrogateRunnerMode.PX4_VISION_DRY_RUN,
                    vision_source_kind="ros_camera",
                    injected_mavlink_messages=usable_px4_messages(),
                )
            )

            with self.assertRaisesRegex(
                SurrogateRunnerSafetyError,
                "--resize-camera-to-competition",
            ):
                runner.run()
        finally:
            surrogate_runner_module._capture_ros_camera_frame_as_bgr = original_capture

    def test_px4_command_dry_run_builds_no_send_candidate_with_injected_telemetry(self):
        runner = SurrogateRunner(
            SurrogateRunnerConfig(
                mode=SurrogateRunnerMode.PX4_COMMAND_DRY_RUN,
                vision_source_kind="generated_mock",
                injected_mavlink_messages=usable_px4_messages(),
                command_tuple=(0.0, 0.0, 1.25, 0.5),
                max_payload_size=4096,
            )
        )

        data = runner.run().to_dict()

        self.assertEqual(data["phase"], PHASE_8_5D_3)
        self.assertEqual(data["mode"], "px4_command_dry_run")
        self.assertEqual(data["status"], "competition_stack_px4_command_dry_run_complete")
        self.assertTrue(data["state_usable"])
        self.assertEqual(data["vision_frames_completed"], 1)
        self.assertEqual(data["perception_update_calls"], 1)
        self.assertEqual(data["command_candidate_count"], 1)
        self.assertFalse(data["command_publication_allowed"])
        self.assertEqual(data["command_sent_count"], 0)
        self.assertIn("command_dry_run_no_send", data["command_blocked_reasons"])
        self.assertIn(
            "phase4b_telemetry_evidence_missing",
            data["command_blocked_reasons"],
        )

    def test_fail_closed_modes_raise_safety_error(self):
        for mode in sorted(FAIL_CLOSED_MODES, key=lambda item: item.value):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(
                    SurrogateRunnerSafetyError,
                    "fail-closed",
                ):
                    SurrogateRunner(SurrogateRunnerConfig(mode=mode))

    def test_real_autonomy_and_command_send_flags_are_rejected(self):
        cases = (
            SurrogateRunnerConfig(
                mode=SurrogateRunnerMode.MOCK_VISION_DRY_RUN,
                use_real_autonomy=True,
            ),
            SurrogateRunnerConfig(
                mode=SurrogateRunnerMode.PX4_COMMAND_DRY_RUN,
                enable_surrogate_command_send=True,
            ),
            SurrogateRunnerConfig(
                mode=SurrogateRunnerMode.PX4_COMMAND_DRY_RUN,
                operator_command_send_confirmation=True,
            ),
        )

        for config in cases:
            with self.subTest(config=config):
                with self.assertRaises(SurrogateRunnerSafetyError):
                    SurrogateRunner(config)

    def test_normalize_unknown_mode_rejects(self):
        with self.assertRaisesRegex(SurrogateRunnerSafetyError, "unknown"):
            normalize_surrogate_mode("not_a_mode")

    def test_cli_supported_mode_prints_no_readiness_claim(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "autonomy_core.runtime.surrogate_runner",
                "mock_vision_dry_run",
                "--source-kind",
                "cli_test",
                "--max-payload-size",
                "4096",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        data = json.loads(completed.stdout)

        self.assertEqual(data["surrogate_label"], SURROGATE_LABEL)
        self.assertEqual(data["phase"], PHASE_8_5D_2)
        self.assertEqual(data["mode"], "mock_vision_dry_run")
        self.assertEqual(data["source_kind"], "cli_test")
        self.assertEqual(data["status"], "competition_stack_vision_dry_run_complete")
        self.assertTrue(data["competition_runner_executed"])
        self.assertEqual(data["vision_frames_completed"], 1)
        self.assertEqual(data["perception_update_calls"], 1)
        self.assertFalse(data["phase4b_satisfied"])
        self.assertFalse(data["competition_readiness_claimed"])
        self.assertFalse(data["command_publication_allowed"])
        self.assertEqual(data["command_sent_count"], 0)

    def test_saved_image_cli_without_input_fails_closed(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "autonomy_core.runtime.surrogate_runner",
                "saved_image_vision_dry_run",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        data = json.loads(completed.stdout)

        self.assertEqual(completed.returncode, 2)
        self.assertEqual(data["phase"], PHASE_8_5D_3)
        self.assertEqual(data["status"], "fail_closed")
        self.assertTrue(data["fail_closed"])
        self.assertIn("requires --input-image", data["safety_error"])
        self.assertEqual(data["command_sent_count"], 0)

    def test_cli_fail_closed_mode_returns_nonzero_json(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "autonomy_core.runtime.surrogate_runner",
                "px4_command_send",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        data = json.loads(completed.stdout)

        self.assertEqual(completed.returncode, 2)
        self.assertEqual(data["status"], "fail_closed")
        self.assertTrue(data["fail_closed"])
        self.assertIn("fail-closed", data["safety_error"])
        self.assertFalse(data["phase4b_satisfied"])
        self.assertFalse(data["competition_readiness_claimed"])
        self.assertEqual(data["command_sent_count"], 0)

    def test_cli_command_send_flag_is_rejected_even_for_dry_run_mode(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "autonomy_core.runtime.surrogate_runner",
                "px4_command_dry_run",
                "--enable-surrogate-command-send",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        data = json.loads(completed.stdout)

        self.assertEqual(completed.returncode, 2)
        self.assertEqual(data["status"], "fail_closed")
        self.assertIn("command send is not enabled", data["safety_error"])
        self.assertEqual(data["command_sent_count"], 0)


if __name__ == "__main__":
    unittest.main()
