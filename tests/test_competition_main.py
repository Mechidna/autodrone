import json
import subprocess
import sys
import unittest

from autonomy_core.runtime.competition_main import (
    CompetitionMainConfig,
    CompetitionMainSafetyError,
    PHASE_6E,
    PHASE_9B,
    PHASE_9C,
    PHASE_9D,
    PHASE_9E,
    PHASE_9E_1,
    PHASE_9E_2,
    PHASE_9E_3,
    PHASE_9E_4,
    run_competition_main,
)
from autonomy_core.runtime.competition_autonomy_factory import (
    COMPETITION_OFFICIAL_TRANSFORM_MODE,
)
from autonomy_core.runtime.competition_runner import CompetitionRunnerMode
from autonomy_core.runtime.competition_setup import CompetitionSetupConfig, build_competition_runtime
from autonomy_core.runtime.px4_gazebo_command_sender import (
    ATTITUDE_HOVER_BODY_RATES_IGNORE_TYPE_MASK,
    ATTITUDE_HOVER_ZERO_BODY_RATES,
    BODY_RATE_ATTITUDE_IGNORE_TYPE_MASK,
    BODY_RATE_DUMMY_QUATERNION,
    MAV_CMD_COMPONENT_ARM_DISARM,
    MAV_CMD_DO_SET_MODE,
    PX4_GAZEBO_SURROGATE_LABEL,
    PX4_GAZEBO_ARM_OFFBOARD_LABEL,
    PX4_GAZEBO_ATTITUDE_HOVER_SMOKE_LABEL,
    PX4_GAZEBO_BODY_RATE_SMOKE_LABEL,
    Px4GazeboCommandSenderConfig,
    Px4GazeboSetAttitudeTargetSender,
)
from autonomy_core.command.competition_command_adapter import (
    quaternion_wxyz_from_euler_zyx,
)


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
    def __init__(self, messages=(), summary_data=None):
        self.messages = list(messages)
        self.summary_data = summary_data
        self.close_calls = 0

    def receive_messages(self):
        messages = self.messages
        self.messages = []
        return messages

    def close(self):
        self.close_calls += 1

    def summary(self):
        if self.summary_data is not None:
            return self.summary_data
        return {"fake_mavlink": True, "remaining": len(self.messages)}


class RepeatingFakeMavlinkTransport(FakeMavlinkTransport):
    def receive_messages(self):
        return usable_messages()


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


class FakeTelemetry:
    def __init__(self):
        self.pos = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.vel = {"vx": 0.0, "vy": 0.0, "vz": 0.0}
        self.rpy = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        self.position_sample_time = float("nan")
        self.velocity_sample_time = float("nan")
        self.attitude_sample_time = float("nan")
        self.yaw_rad_raw = 0.0
        self.yaw_rad_perception = 0.0


class FakeAutonomy:
    def __init__(self):
        self.perception_updates = []
        self.attitude_control_calls = 0
        self.path_plan_calls = 0
        self.path_plan_reasons = []
        self.telemetry = FakeTelemetry()

    def update_gate_memory_from_frame(self, **kwargs):
        self.perception_updates.append(kwargs)

    def path_plan(self, replan_reason="scheduled"):
        self.path_plan_calls += 1
        self.path_plan_reasons.append(replan_reason)
        return True

    def attitude_control(self):
        self.attitude_control_calls += 1
        return (0.0, 0.0, 0.0, 0.5)


class HighThrustFakeAutonomy(FakeAutonomy):
    def attitude_control(self):
        self.attitude_control_calls += 1
        return (0.0, 0.0, 0.0, 0.85)


class FakeMav:
    def __init__(self):
        self.set_attitude_target_calls = []
        self.command_long_calls = []

    def set_attitude_target_send(self, *args):
        self.set_attitude_target_calls.append(args)

    def command_long_send(self, *args):
        self.command_long_calls.append(args)


class FakeConnection:
    def __init__(self):
        self.mav = FakeMav()


class IncrementingClock:
    def __init__(self, start=100.0, step=0.02):
        self.current = float(start) - float(step)
        self.step = float(step)

    def __call__(self):
        self.current += self.step
        return self.current


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


def px4_armed_offboard_summary():
    return {
        "message_types": {
            "HEARTBEAT": {
                "samples": [
                    {
                        "base_mode": 128,
                        "custom_mode": 6 << 16,
                    }
                ]
            }
        }
    }


def components_for_mode(
    mode,
    *,
    with_vision=False,
    autonomy=None,
    mavlink_summary=None,
):
    return build_competition_runtime(
        CompetitionSetupConfig(mode=mode),
        mavlink_transport=FakeMavlinkTransport(
            usable_messages(),
            summary_data=mavlink_summary,
        ),
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

    def test_phase9b_real_perception_dry_run_reports_success_with_fake_factory(self):
        captured_setup_configs = []
        autonomy = FakeAutonomy()

        def components_factory(setup_config):
            captured_setup_configs.append(setup_config)
            return components_for_mode(
                setup_config.mode,
                with_vision=True,
                autonomy=autonomy,
            )

        summary = run_competition_main(
            CompetitionMainConfig(
                mode=CompetitionRunnerMode.VISION_DRY_RUN,
                live_transports=True,
                use_real_autonomy=True,
                real_perception=True,
                allow_legacy_yolo_default=True,
                evidence_label="px4_gazebo_surrogate_phase9b",
            ),
            components_factory=components_factory,
            clock=lambda: 100.0,
        ).to_dict()

        self.assertEqual(summary["phase"], PHASE_9B)
        self.assertEqual(summary["evidence_label"], "px4_gazebo_surrogate_phase9b")
        self.assertTrue(summary["use_real_autonomy"])
        self.assertTrue(summary["real_perception_requested"])
        self.assertTrue(summary["allow_legacy_yolo_default"])
        self.assertEqual(
            summary["perception_transform_mode"],
            COMPETITION_OFFICIAL_TRANSFORM_MODE,
        )
        self.assertTrue(summary["phase6e_receive_satisfied"])
        self.assertTrue(summary["phase6e_perception_boundary_satisfied"])
        self.assertTrue(summary["phase9b_perception_dry_run_satisfied"])
        self.assertEqual(summary["perception_update_calls"], 1)
        self.assertFalse(summary["command_publication_allowed"])
        self.assertEqual(summary["command_sent_count"], 0)
        self.assertFalse(summary["phase4b_satisfied"])
        self.assertFalse(summary["competition_readiness_claimed"])
        self.assertTrue(summary["phase9b_success_criteria"]["use_real_autonomy"])
        self.assertTrue(summary["phase9b_success_criteria"]["real_perception_requested"])
        self.assertTrue(
            summary["phase9b_success_criteria"]["legacy_yolo_default_acknowledged"]
        )
        self.assertTrue(
            summary["phase9b_success_criteria"][
                "official_competition_transform_selected"
            ]
        )
        self.assertTrue(captured_setup_configs[0].autonomy_profile.use_perception)
        self.assertTrue(
            captured_setup_configs[0].autonomy_profile.allow_legacy_yolo_default
        )
        self.assertEqual(
            captured_setup_configs[0].autonomy_profile.perception_transform_mode,
            COMPETITION_OFFICIAL_TRANSFORM_MODE,
        )
        self.assertIsNone(autonomy.perception_updates[0]["gazebo_pose"])
        self.assertIsNone(autonomy.perception_updates[0]["image_pose_snapshot"])

    def test_phase9b_requires_real_autonomy(self):
        with self.assertRaisesRegex(CompetitionMainSafetyError, "--use-real-autonomy"):
            run_competition_main(
                CompetitionMainConfig(
                    mode=CompetitionRunnerMode.VISION_DRY_RUN,
                    live_transports=True,
                    real_perception=True,
                    allow_legacy_yolo_default=True,
                ),
                components_factory=lambda _config: components_for_mode(
                    CompetitionRunnerMode.VISION_DRY_RUN,
                    with_vision=True,
                    autonomy=FakeAutonomy(),
                ),
            )

    def test_phase9b_requires_legacy_yolo_ack_for_live_real_autonomy(self):
        with self.assertRaisesRegex(CompetitionMainSafetyError, "hardcoded YOLO"):
            run_competition_main(
                CompetitionMainConfig(
                    mode=CompetitionRunnerMode.VISION_DRY_RUN,
                    live_transports=True,
                    use_real_autonomy=True,
                    real_perception=True,
                )
            )

    def test_phase9b_requires_official_competition_transform(self):
        with self.assertRaisesRegex(CompetitionMainSafetyError, "competition_official_ned"):
            run_competition_main(
                CompetitionMainConfig(
                    mode=CompetitionRunnerMode.VISION_DRY_RUN,
                    live_transports=True,
                    use_real_autonomy=True,
                    real_perception=True,
                    allow_legacy_yolo_default=True,
                    perception_transform_mode="physical_direct_rad_x_mirror",
                ),
                components_factory=lambda _config: components_for_mode(
                    CompetitionRunnerMode.VISION_DRY_RUN,
                    with_vision=True,
                    autonomy=FakeAutonomy(),
                ),
            )

    def test_phase9c_command_candidate_dry_run_reports_success_with_fake_factory(self):
        captured_setup_configs = []
        autonomy = FakeAutonomy()

        def components_factory(setup_config):
            captured_setup_configs.append(setup_config)
            return components_for_mode(
                setup_config.mode,
                with_vision=True,
                autonomy=autonomy,
            )

        summary = run_competition_main(
            CompetitionMainConfig(
                mode=CompetitionRunnerMode.COMMAND_DRY_RUN,
                live_transports=True,
                use_real_autonomy=True,
                real_perception=True,
                allow_legacy_yolo_default=True,
                evidence_label="px4_gazebo_surrogate_phase9c",
            ),
            components_factory=components_factory,
            clock=lambda: 100.0,
        ).to_dict()

        self.assertEqual(summary["phase"], PHASE_9C)
        self.assertEqual(summary["mode"], "command_dry_run")
        self.assertEqual(summary["evidence_label"], "px4_gazebo_surrogate_phase9c")
        self.assertTrue(summary["phase6e_receive_satisfied"])
        self.assertTrue(summary["phase6e_perception_boundary_satisfied"])
        self.assertTrue(summary["phase9c_command_dry_run_satisfied"])
        self.assertEqual(summary["perception_update_calls"], 1)
        self.assertEqual(summary["autonomy_telemetry_sync_count"], 1)
        self.assertEqual(summary["planning_attempt_count"], 1)
        self.assertEqual(summary["planning_success_count"], 1)
        self.assertEqual(summary["planning_failure_count"], 0)
        self.assertEqual(summary["command_candidate_count"], 1)
        self.assertEqual(summary["command_candidate_accepted_count"], 1)
        self.assertEqual(summary["command_candidate_rejection_count"], 0)
        self.assertFalse(summary["command_publication_allowed"])
        self.assertEqual(summary["command_sent_count"], 0)
        self.assertFalse(summary["phase4b_satisfied"])
        self.assertFalse(summary["competition_readiness_claimed"])
        self.assertEqual(autonomy.path_plan_calls, 1)
        self.assertEqual(
            autonomy.path_plan_reasons,
            ["phase9c_command_dry_run"],
        )
        self.assertEqual(autonomy.attitude_control_calls, 1)
        self.assertEqual(autonomy.telemetry.pos, {"x": 1.0, "y": 2.0, "z": 3.0})
        self.assertEqual(
            autonomy.telemetry.vel,
            {"vx": 0.1, "vy": 0.2, "vz": 0.3},
        )
        self.assertEqual(autonomy.telemetry.rpy["yaw"], 0.25)
        self.assertEqual(
            captured_setup_configs[0].autonomy_profile.perception_transform_mode,
            COMPETITION_OFFICIAL_TRANSFORM_MODE,
        )

        command_result = summary["last_command_result"]
        self.assertTrue(command_result["accepted"])
        fields = command_result["fields"]
        self.assertEqual(fields["message_name"], "SET_ATTITUDE_TARGET")
        self.assertEqual(fields["target_system"], 1)
        self.assertEqual(fields["target_component"], 1)
        self.assertFalse(fields["send_ready"])
        self.assertEqual(fields["thrust"], 0.5)
        self.assertIn("dry-run only", fields["send_blocked_reason"])
        self.assertTrue(
            summary["phase9c_success_criteria"]["last_command_result_no_send"]
        )
        self.assertTrue(
            summary["phase9c_success_criteria"]["planning_success_count_gt_0"]
        )

    def test_phase9c_requires_command_dry_run_for_real_command_candidates(self):
        with self.assertRaisesRegex(CompetitionMainSafetyError, "Phase 9C command_dry_run"):
            run_competition_main(
                CompetitionMainConfig(
                    mode=CompetitionRunnerMode.OBSERVE,
                    live_transports=True,
                    use_real_autonomy=True,
                    real_perception=True,
                    allow_legacy_yolo_default=True,
                ),
                components_factory=lambda _config: components_for_mode(
                    CompetitionRunnerMode.OBSERVE,
                    autonomy=FakeAutonomy(),
                ),
            )

    def test_phase9d_requires_explicit_surrogate_command_ack(self):
        with self.assertRaisesRegex(CompetitionMainSafetyError, "ack-px4"):
            run_competition_main(
                CompetitionMainConfig(
                    mode=CompetitionRunnerMode.COMMAND_DRY_RUN,
                    live_transports=True,
                    use_real_autonomy=True,
                    real_perception=True,
                    allow_legacy_yolo_default=True,
                    px4_gazebo_command_send=True,
                ),
                components=components_for_mode(
                    CompetitionRunnerMode.COMMAND_DRY_RUN,
                    with_vision=True,
                    autonomy=FakeAutonomy(),
                ),
            )

    def test_phase9d_surrogate_command_send_reports_success_with_fake_connection(self):
        connection = FakeConnection()
        sender = Px4GazeboSetAttitudeTargetSender(
            connection=connection,
            clock=lambda: 100.0,
        )
        autonomy = FakeAutonomy()

        summary = run_competition_main(
            CompetitionMainConfig(
                mode=CompetitionRunnerMode.COMMAND_DRY_RUN,
                live_transports=True,
                use_real_autonomy=True,
                real_perception=True,
                allow_legacy_yolo_default=True,
                px4_gazebo_command_send=True,
                ack_px4_gazebo_surrogate_command_send=True,
                px4_gazebo_command_max_count=1,
                evidence_label="px4_gazebo_surrogate_phase9d",
            ),
            components=components_for_mode(
                CompetitionRunnerMode.COMMAND_DRY_RUN,
                with_vision=True,
                autonomy=autonomy,
            ),
            command_sender=sender,
            clock=lambda: 100.0,
        ).to_dict()

        self.assertEqual(summary["phase"], PHASE_9D)
        self.assertEqual(summary["mode"], "command_dry_run")
        self.assertEqual(summary["evidence_label"], "px4_gazebo_surrogate_phase9d")
        self.assertTrue(summary["px4_gazebo_command_send_requested"])
        self.assertTrue(summary["px4_gazebo_command_send_acknowledged"])
        self.assertTrue(summary["command_publication_allowed"])
        self.assertEqual(summary["command_candidate_count"], 1)
        self.assertEqual(summary["command_candidate_accepted_count"], 1)
        self.assertEqual(summary["command_send_attempt_count"], 1)
        self.assertEqual(summary["command_sent_count"], 1)
        self.assertEqual(summary["command_send_rejection_count"], 0)
        self.assertTrue(summary["phase9d_surrogate_command_send_satisfied"])
        self.assertFalse(summary["phase4b_satisfied"])
        self.assertFalse(summary["competition_readiness_claimed"])
        self.assertEqual(len(connection.mav.set_attitude_target_calls), 1)
        self.assertEqual(
            summary["last_command_send_result"]["surrogate_label"],
            PX4_GAZEBO_SURROGATE_LABEL,
        )
        self.assertTrue(summary["last_command_send_result"]["sent"])
        self.assertFalse(summary["last_command_send_result"]["phase4b_satisfied"])
        self.assertFalse(
            summary["last_command_send_result"]["competition_readiness_claimed"]
        )
        self.assertFalse(summary["last_command_result"]["fields"]["send_ready"])
        self.assertTrue(
            summary["phase9d_success_criteria"][
                "last_command_send_surrogate_labeled"
            ]
        )

    def test_phase9d_rejects_send_when_current_frame_is_not_fresh(self):
        connection = FakeConnection()
        sender = Px4GazeboSetAttitudeTargetSender(connection=connection)

        summary = run_competition_main(
            CompetitionMainConfig(
                mode=CompetitionRunnerMode.COMMAND_DRY_RUN,
                live_transports=True,
                use_real_autonomy=True,
                real_perception=True,
                allow_legacy_yolo_default=True,
                px4_gazebo_command_send=True,
                ack_px4_gazebo_surrogate_command_send=True,
            ),
            components=components_for_mode(
                CompetitionRunnerMode.COMMAND_DRY_RUN,
                with_vision=False,
                autonomy=FakeAutonomy(),
            ),
            command_sender=sender,
            clock=lambda: 100.0,
        ).to_dict()

        self.assertEqual(summary["phase"], PHASE_9D)
        self.assertEqual(summary["command_sent_count"], 0)
        self.assertEqual(summary["command_send_rejection_count"], 1)
        self.assertEqual(
            summary["last_command_send_result"]["rejection_reason"],
            "image_not_fresh",
        )
        self.assertFalse(summary["phase9d_surrogate_command_send_satisfied"])
        self.assertEqual(len(connection.mav.set_attitude_target_calls), 0)

    def test_phase9e_requires_explicit_arm_ack(self):
        with self.assertRaisesRegex(CompetitionMainSafetyError, "ack-px4-gazebo-surrogate-arm"):
            run_competition_main(
                CompetitionMainConfig(
                    mode=CompetitionRunnerMode.COMMAND_DRY_RUN,
                    live_transports=True,
                    use_real_autonomy=True,
                    real_perception=True,
                    allow_legacy_yolo_default=True,
                    px4_gazebo_arm=True,
                ),
                components=components_for_mode(
                    CompetitionRunnerMode.COMMAND_DRY_RUN,
                    with_vision=True,
                    autonomy=FakeAutonomy(),
                ),
            )

    def test_phase9e_requires_explicit_offboard_ack(self):
        with self.assertRaisesRegex(
            CompetitionMainSafetyError,
            "ack-px4-gazebo-surrogate-offboard",
        ):
            run_competition_main(
                CompetitionMainConfig(
                    mode=CompetitionRunnerMode.COMMAND_DRY_RUN,
                    live_transports=True,
                    use_real_autonomy=True,
                    real_perception=True,
                    allow_legacy_yolo_default=True,
                    px4_gazebo_offboard=True,
                ),
                components=components_for_mode(
                    CompetitionRunnerMode.COMMAND_DRY_RUN,
                    with_vision=True,
                    autonomy=FakeAutonomy(),
                ),
            )

    def test_phase9e_arm_offboard_reports_success_with_fake_connection_and_state(self):
        connection = FakeConnection()
        sender = Px4GazeboSetAttitudeTargetSender(connection=connection)

        summary = run_competition_main(
            CompetitionMainConfig(
                mode=CompetitionRunnerMode.COMMAND_DRY_RUN,
                live_transports=True,
                use_real_autonomy=True,
                real_perception=True,
                allow_legacy_yolo_default=True,
                px4_gazebo_arm=True,
                ack_px4_gazebo_surrogate_arm=True,
                px4_gazebo_offboard=True,
                ack_px4_gazebo_surrogate_offboard=True,
                px4_gazebo_command_send=True,
                ack_px4_gazebo_surrogate_command_send=True,
                evidence_label="px4_gazebo_surrogate_phase9e",
            ),
            components=components_for_mode(
                CompetitionRunnerMode.COMMAND_DRY_RUN,
                with_vision=True,
                autonomy=FakeAutonomy(),
                mavlink_summary=px4_armed_offboard_summary(),
            ),
            command_sender=sender,
            clock=lambda: 100.0,
        ).to_dict()

        self.assertEqual(summary["phase"], PHASE_9E)
        self.assertEqual(summary["evidence_label"], "px4_gazebo_surrogate_phase9e")
        self.assertTrue(summary["px4_gazebo_arm_requested"])
        self.assertTrue(summary["px4_gazebo_arm_acknowledged"])
        self.assertTrue(summary["px4_gazebo_offboard_requested"])
        self.assertTrue(summary["px4_gazebo_offboard_acknowledged"])
        self.assertEqual(summary["px4_gazebo_arm_offboard_label"], PX4_GAZEBO_ARM_OFFBOARD_LABEL)
        self.assertEqual(summary["arm_attempt_count"], 1)
        self.assertEqual(summary["arm_sent_count"], 1)
        self.assertEqual(summary["offboard_attempt_count"], 1)
        self.assertEqual(summary["offboard_sent_count"], 1)
        self.assertTrue(summary["armed_state_observed"])
        self.assertTrue(summary["offboard_state_observed"])
        self.assertEqual(summary["command_sent_count"], 1)
        self.assertTrue(summary["phase9e_surrogate_arm_offboard_satisfied"])
        self.assertFalse(summary["phase4b_satisfied"])
        self.assertFalse(summary["competition_readiness_claimed"])
        self.assertEqual(len(connection.mav.command_long_calls), 2)
        self.assertEqual(
            connection.mav.command_long_calls[0][2],
            MAV_CMD_COMPONENT_ARM_DISARM,
        )
        self.assertEqual(connection.mav.command_long_calls[1][2], MAV_CMD_DO_SET_MODE)
        self.assertEqual(len(connection.mav.set_attitude_target_calls), 1)
        self.assertEqual(summary["last_arm_result"]["surrogate_label"], PX4_GAZEBO_ARM_OFFBOARD_LABEL)
        self.assertEqual(summary["last_offboard_result"]["surrogate_label"], PX4_GAZEBO_ARM_OFFBOARD_LABEL)

    def test_phase9e1_streams_cached_setpoint_after_image_goes_stale(self):
        clock = IncrementingClock()
        connection = FakeConnection()
        sender = Px4GazeboSetAttitudeTargetSender(connection=connection)
        components = build_competition_runtime(
            CompetitionSetupConfig(mode=CompetitionRunnerMode.COMMAND_DRY_RUN),
            mavlink_transport=RepeatingFakeMavlinkTransport(
                summary_data=px4_armed_offboard_summary()
            ),
            vision_transport=FakeVisionTransport([b"packet"]),
            image_adapter=FakeImageAdapter(),
            autonomy=FakeAutonomy(),
            clock=clock,
        )

        summary = run_competition_main(
            CompetitionMainConfig(
                mode=CompetitionRunnerMode.COMMAND_DRY_RUN,
                live_transports=True,
                use_real_autonomy=True,
                real_perception=True,
                allow_legacy_yolo_default=True,
                px4_gazebo_arm=True,
                ack_px4_gazebo_surrogate_arm=True,
                px4_gazebo_offboard=True,
                ack_px4_gazebo_surrogate_offboard=True,
                px4_gazebo_command_send=True,
                ack_px4_gazebo_surrogate_command_send=True,
                px4_gazebo_continuous_setpoint_stream=True,
                px4_gazebo_command_max_count=4,
                px4_gazebo_command_max_age_s=1.0,
                px4_gazebo_offboard_prestream_count=2,
                evidence_label="px4_gazebo_surrogate_phase9e1",
                steps=4,
            ),
            components=components,
            command_sender=sender,
            clock=clock,
        ).to_dict()

        self.assertEqual(summary["phase"], PHASE_9E_1)
        self.assertTrue(summary["px4_gazebo_continuous_setpoint_stream_requested"])
        self.assertEqual(summary["px4_gazebo_command_max_age_s"], 1.0)
        self.assertEqual(summary["vision_frames_completed"], 1)
        self.assertEqual(summary["perception_update_calls"], 1)
        self.assertGreater(summary["command_candidate_count"], 1)
        self.assertEqual(summary["setpoint_stream_cache_update_count"], 1)
        self.assertGreaterEqual(summary["setpoint_stream_reused_count"], 1)
        self.assertEqual(summary["setpoint_stream_stale_rejection_count"], 0)
        self.assertEqual(summary["command_sent_count"], 4)
        self.assertEqual(summary["command_send_rejection_count"], 0)
        self.assertTrue(summary["phase9e_surrogate_arm_offboard_satisfied"])
        self.assertTrue(summary["phase9e1_continuous_setpoint_stream_satisfied"])
        self.assertTrue(summary["phase9e1_success_criteria"]["command_sent_count_gt_1"])
        self.assertTrue(
            summary["phase9e1_success_criteria"][
                "setpoint_stream_reused_count_gt_0"
            ]
        )
        self.assertFalse(summary["phase4b_satisfied"])
        self.assertFalse(summary["competition_readiness_claimed"])
        self.assertEqual(len(connection.mav.set_attitude_target_calls), 4)
        self.assertEqual(len(connection.mav.command_long_calls), 2)

    def test_phase9e2_clamps_surrogate_thrust_without_changing_command_candidate(self):
        connection = FakeConnection()
        sender = Px4GazeboSetAttitudeTargetSender(
            config=Px4GazeboCommandSenderConfig(
                enable_surrogate_thrust_clamp=True,
                surrogate_thrust_clamp_max=0.76,
            ),
            connection=connection,
        )
        autonomy = HighThrustFakeAutonomy()

        summary = run_competition_main(
            CompetitionMainConfig(
                mode=CompetitionRunnerMode.COMMAND_DRY_RUN,
                live_transports=True,
                use_real_autonomy=True,
                real_perception=True,
                allow_legacy_yolo_default=True,
                px4_gazebo_command_send=True,
                ack_px4_gazebo_surrogate_command_send=True,
                px4_gazebo_surrogate_thrust_clamp=True,
                px4_gazebo_surrogate_thrust_clamp_max=0.76,
                px4_gazebo_command_max_count=1,
                evidence_label="px4_gazebo_surrogate_phase9e2",
            ),
            components=components_for_mode(
                CompetitionRunnerMode.COMMAND_DRY_RUN,
                with_vision=True,
                autonomy=autonomy,
            ),
            command_sender=sender,
            clock=lambda: 100.0,
        ).to_dict()

        self.assertEqual(summary["phase"], PHASE_9E_2)
        self.assertEqual(summary["evidence_label"], "px4_gazebo_surrogate_phase9e2")
        self.assertTrue(summary["px4_gazebo_surrogate_thrust_clamp_requested"])
        self.assertEqual(summary["px4_gazebo_surrogate_thrust_clamp_max"], 0.76)
        self.assertEqual(summary["last_command_result"]["fields"]["thrust"], 0.85)
        self.assertEqual(summary["command_sent_count"], 1)
        self.assertTrue(summary["last_command_send_result"]["sent"])
        self.assertTrue(summary["last_command_send_result"]["thrust_clamped"])
        self.assertEqual(summary["last_command_send_result"]["raw_thrust"], 0.85)
        self.assertEqual(summary["last_command_send_result"]["thrust"], 0.76)
        self.assertEqual(summary["last_command_send_result"]["stream_source"], "current")
        self.assertEqual(connection.mav.set_attitude_target_calls[0][8], 0.76)
        self.assertEqual(
            summary["command_sender_summary"]["stats"]["thrust_clamp_count"],
            1,
        )
        self.assertTrue(summary["phase9e2_surrogate_command_safety_satisfied"])
        self.assertTrue(
            summary["phase9e2_success_criteria"]["last_command_send_thrust_clamped"]
        )
        self.assertFalse(summary["phase4b_satisfied"])
        self.assertFalse(summary["competition_readiness_claimed"])

    def test_phase9e2_clamp_requires_explicit_command_send(self):
        with self.assertRaisesRegex(CompetitionMainSafetyError, "requires --px4"):
            run_competition_main(
                CompetitionMainConfig(
                    mode=CompetitionRunnerMode.COMMAND_DRY_RUN,
                    live_transports=True,
                    use_real_autonomy=True,
                    real_perception=True,
                    allow_legacy_yolo_default=True,
                    px4_gazebo_surrogate_thrust_clamp=True,
                    px4_gazebo_surrogate_thrust_clamp_max=0.76,
                ),
                components=components_for_mode(
                    CompetitionRunnerMode.COMMAND_DRY_RUN,
                    with_vision=True,
                    autonomy=HighThrustFakeAutonomy(),
                ),
            )

    def test_phase9e3_body_rate_smoke_reports_success_without_real_autonomy(self):
        connection = FakeConnection()
        sender = Px4GazeboSetAttitudeTargetSender(connection=connection)

        summary = run_competition_main(
            CompetitionMainConfig(
                mode=CompetitionRunnerMode.COMMAND_DRY_RUN,
                live_transports=True,
                px4_gazebo_body_rate_smoke=True,
                ack_px4_gazebo_surrogate_body_rate_smoke=True,
                px4_gazebo_body_roll_rate=0.1,
                px4_gazebo_body_pitch_rate=-0.2,
                px4_gazebo_body_yaw_rate=0.3,
                px4_gazebo_body_rate_thrust=0.74,
                px4_gazebo_command_max_count=1,
                evidence_label="px4_gazebo_surrogate_phase9e3",
            ),
            components=components_for_mode(CompetitionRunnerMode.COMMAND_DRY_RUN),
            command_sender=sender,
            clock=lambda: 100.0,
        ).to_dict()

        self.assertEqual(summary["phase"], PHASE_9E_3)
        self.assertEqual(summary["evidence_label"], "px4_gazebo_surrogate_phase9e3")
        self.assertFalse(summary["use_real_autonomy"])
        self.assertFalse(summary["real_perception_requested"])
        self.assertTrue(summary["px4_gazebo_body_rate_smoke_requested"])
        self.assertTrue(summary["px4_gazebo_body_rate_smoke_acknowledged"])
        self.assertEqual(summary["px4_gazebo_body_rate_smoke_label"], PX4_GAZEBO_BODY_RATE_SMOKE_LABEL)
        self.assertEqual(summary["body_rate_type_mask"], BODY_RATE_ATTITUDE_IGNORE_TYPE_MASK)
        self.assertEqual(summary["body_rate_q"], list(BODY_RATE_DUMMY_QUATERNION))
        self.assertEqual(summary["body_roll_rate"], 0.1)
        self.assertEqual(summary["body_pitch_rate"], -0.2)
        self.assertEqual(summary["body_yaw_rate"], 0.3)
        self.assertEqual(summary["body_rate_thrust"], 0.74)
        self.assertEqual(summary["body_rate_command_sent_count"], 1)
        self.assertEqual(summary["body_rate_command_rejection_count"], 0)
        self.assertEqual(summary["command_sent_count"], 0)
        self.assertTrue(summary["command_publication_allowed"])
        self.assertTrue(summary["phase9e3_body_rate_interface_satisfied"])
        self.assertEqual(
            summary["last_body_rate_command_send_result"]["type_mask"],
            BODY_RATE_ATTITUDE_IGNORE_TYPE_MASK,
        )
        self.assertEqual(
            summary["last_body_rate_command_send_result"]["q"],
            list(BODY_RATE_DUMMY_QUATERNION),
        )
        self.assertEqual(
            summary["last_body_rate_command_send_result"]["surrogate_label"],
            PX4_GAZEBO_BODY_RATE_SMOKE_LABEL,
        )
        self.assertFalse(
            summary["last_body_rate_command_send_result"]["phase4b_satisfied"]
        )
        self.assertFalse(
            summary["last_body_rate_command_send_result"][
                "competition_readiness_claimed"
            ]
        )
        self.assertEqual(
            connection.mav.set_attitude_target_calls[0],
            (
                100000,
                1,
                1,
                BODY_RATE_ATTITUDE_IGNORE_TYPE_MASK,
                BODY_RATE_DUMMY_QUATERNION,
                0.1,
                -0.2,
                0.3,
                0.74,
            ),
        )
        self.assertFalse(summary["phase4b_satisfied"])
        self.assertFalse(summary["competition_readiness_claimed"])

    def test_phase9e3_requires_explicit_ack_and_thrust(self):
        with self.assertRaisesRegex(CompetitionMainSafetyError, "body-rate-smoke"):
            run_competition_main(
                CompetitionMainConfig(
                    mode=CompetitionRunnerMode.COMMAND_DRY_RUN,
                    live_transports=True,
                    px4_gazebo_body_rate_smoke=True,
                    px4_gazebo_body_rate_thrust=0.74,
                ),
                components=components_for_mode(CompetitionRunnerMode.COMMAND_DRY_RUN),
            )

        with self.assertRaisesRegex(CompetitionMainSafetyError, "body-rate-thrust"):
            run_competition_main(
                CompetitionMainConfig(
                    mode=CompetitionRunnerMode.COMMAND_DRY_RUN,
                    live_transports=True,
                    px4_gazebo_body_rate_smoke=True,
                    ack_px4_gazebo_surrogate_body_rate_smoke=True,
                ),
                components=components_for_mode(CompetitionRunnerMode.COMMAND_DRY_RUN),
            )

    def test_phase9e3_rejects_mixing_with_autonomy_command_send(self):
        with self.assertRaisesRegex(CompetitionMainSafetyError, "cannot be combined"):
            run_competition_main(
                CompetitionMainConfig(
                    mode=CompetitionRunnerMode.COMMAND_DRY_RUN,
                    live_transports=True,
                    use_real_autonomy=True,
                    real_perception=True,
                    allow_legacy_yolo_default=True,
                    px4_gazebo_command_send=True,
                    ack_px4_gazebo_surrogate_command_send=True,
                    px4_gazebo_body_rate_smoke=True,
                    ack_px4_gazebo_surrogate_body_rate_smoke=True,
                    px4_gazebo_body_rate_thrust=0.74,
                ),
                components=components_for_mode(
                    CompetitionRunnerMode.COMMAND_DRY_RUN,
                    with_vision=True,
                    autonomy=FakeAutonomy(),
                ),
            )

    def test_phase9e3_body_rate_prestream_can_satisfy_offboard_gate(self):
        clock = IncrementingClock()
        connection = FakeConnection()
        sender = Px4GazeboSetAttitudeTargetSender(connection=connection)
        components = build_competition_runtime(
            CompetitionSetupConfig(mode=CompetitionRunnerMode.COMMAND_DRY_RUN),
            mavlink_transport=RepeatingFakeMavlinkTransport(
                summary_data=px4_armed_offboard_summary()
            ),
            vision_transport=FakeVisionTransport([]),
            image_adapter=FakeImageAdapter(),
            autonomy=None,
            clock=clock,
        )

        summary = run_competition_main(
            CompetitionMainConfig(
                mode=CompetitionRunnerMode.COMMAND_DRY_RUN,
                live_transports=True,
                px4_gazebo_body_rate_smoke=True,
                ack_px4_gazebo_surrogate_body_rate_smoke=True,
                px4_gazebo_body_rate_thrust=0.74,
                px4_gazebo_command_max_count=3,
                px4_gazebo_offboard=True,
                ack_px4_gazebo_surrogate_offboard=True,
                px4_gazebo_offboard_prestream_count=2,
                steps=4,
            ),
            components=components,
            command_sender=sender,
            clock=clock,
        ).to_dict()

        self.assertEqual(summary["phase"], PHASE_9E_3)
        self.assertEqual(summary["body_rate_command_sent_count"], 3)
        self.assertEqual(summary["offboard_sent_count"], 1)
        self.assertTrue(summary["offboard_state_observed"])
        self.assertTrue(
            summary["phase9e3_success_criteria"][
                "offboard_prestream_count_satisfied"
            ]
        )
        self.assertTrue(summary["phase9e3_body_rate_interface_satisfied"])
        self.assertEqual(len(connection.mav.set_attitude_target_calls), 3)
        self.assertEqual(connection.mav.command_long_calls[0][2], MAV_CMD_DO_SET_MODE)

    def test_phase9e4_attitude_hover_smoke_reports_success_without_real_autonomy(self):
        connection = FakeConnection()
        sender = Px4GazeboSetAttitudeTargetSender(connection=connection)
        expected_q = quaternion_wxyz_from_euler_zyx(
            roll_rad=0.0,
            pitch_rad=0.0,
            yaw_rad=0.25,
        )

        summary = run_competition_main(
            CompetitionMainConfig(
                mode=CompetitionRunnerMode.COMMAND_DRY_RUN,
                live_transports=True,
                px4_gazebo_attitude_hover_smoke=True,
                ack_px4_gazebo_surrogate_attitude_hover_smoke=True,
                px4_gazebo_attitude_hover_thrust=0.74,
                px4_gazebo_command_max_count=1,
                evidence_label="px4_gazebo_surrogate_phase9e4",
            ),
            components=components_for_mode(CompetitionRunnerMode.COMMAND_DRY_RUN),
            command_sender=sender,
            clock=lambda: 100.0,
        ).to_dict()

        self.assertEqual(summary["phase"], PHASE_9E_4)
        self.assertEqual(summary["evidence_label"], "px4_gazebo_surrogate_phase9e4")
        self.assertFalse(summary["use_real_autonomy"])
        self.assertFalse(summary["real_perception_requested"])
        self.assertTrue(summary["px4_gazebo_attitude_hover_smoke_requested"])
        self.assertTrue(summary["px4_gazebo_attitude_hover_smoke_acknowledged"])
        self.assertEqual(
            summary["px4_gazebo_attitude_hover_smoke_label"],
            PX4_GAZEBO_ATTITUDE_HOVER_SMOKE_LABEL,
        )
        self.assertEqual(
            summary["attitude_hover_type_mask"],
            ATTITUDE_HOVER_BODY_RATES_IGNORE_TYPE_MASK,
        )
        self.assertEqual(
            summary["attitude_hover_body_rates"],
            list(ATTITUDE_HOVER_ZERO_BODY_RATES),
        )
        self.assertEqual(summary["attitude_hover_roll_rad"], 0.0)
        self.assertEqual(summary["attitude_hover_pitch_rad"], 0.0)
        self.assertAlmostEqual(summary["attitude_hover_yaw_rad"], 0.25)
        self.assertEqual(summary["attitude_hover_yaw_source"], "current_state")
        self.assertEqual(summary["attitude_hover_thrust"], 0.74)
        self.assertEqual(summary["attitude_hover_command_sent_count"], 1)
        self.assertEqual(summary["attitude_hover_command_rejection_count"], 0)
        self.assertEqual(summary["command_sent_count"], 0)
        self.assertTrue(summary["command_publication_allowed"])
        self.assertTrue(summary["phase9e4_attitude_hover_interface_satisfied"])
        self.assertEqual(
            summary["last_attitude_hover_command_send_result"]["type_mask"],
            ATTITUDE_HOVER_BODY_RATES_IGNORE_TYPE_MASK,
        )
        self.assertEqual(
            summary["last_attitude_hover_command_send_result"]["q"],
            list(expected_q),
        )
        self.assertEqual(
            summary["last_attitude_hover_command_send_result"]["yaw_source"],
            "current_state",
        )
        self.assertEqual(
            summary["last_attitude_hover_command_send_result"]["surrogate_label"],
            PX4_GAZEBO_ATTITUDE_HOVER_SMOKE_LABEL,
        )
        self.assertFalse(
            summary["last_attitude_hover_command_send_result"]["phase4b_satisfied"]
        )
        self.assertFalse(
            summary["last_attitude_hover_command_send_result"][
                "competition_readiness_claimed"
            ]
        )
        self.assertEqual(
            connection.mav.set_attitude_target_calls[0],
            (
                100000,
                1,
                1,
                ATTITUDE_HOVER_BODY_RATES_IGNORE_TYPE_MASK,
                expected_q,
                0.0,
                0.0,
                0.0,
                0.74,
            ),
        )
        self.assertFalse(summary["phase4b_satisfied"])
        self.assertFalse(summary["competition_readiness_claimed"])

    def test_phase9e4_can_use_explicit_yaw(self):
        connection = FakeConnection()
        sender = Px4GazeboSetAttitudeTargetSender(connection=connection)
        expected_q = quaternion_wxyz_from_euler_zyx(
            roll_rad=0.0,
            pitch_rad=0.0,
            yaw_rad=1.0,
        )

        summary = run_competition_main(
            CompetitionMainConfig(
                mode=CompetitionRunnerMode.COMMAND_DRY_RUN,
                live_transports=True,
                px4_gazebo_attitude_hover_smoke=True,
                ack_px4_gazebo_surrogate_attitude_hover_smoke=True,
                px4_gazebo_attitude_hover_yaw_rad=1.0,
                px4_gazebo_attitude_hover_thrust=0.74,
                px4_gazebo_command_max_count=1,
            ),
            components=components_for_mode(CompetitionRunnerMode.COMMAND_DRY_RUN),
            command_sender=sender,
            clock=lambda: 100.0,
        ).to_dict()

        self.assertEqual(summary["phase"], PHASE_9E_4)
        self.assertEqual(summary["attitude_hover_yaw_source"], "explicit")
        self.assertAlmostEqual(summary["attitude_hover_yaw_rad"], 1.0)
        self.assertEqual(
            connection.mav.set_attitude_target_calls[0][4],
            expected_q,
        )
        self.assertTrue(
            summary["phase9e4_success_criteria"][
                "last_attitude_hover_yaw_matches_explicit"
            ]
        )

    def test_phase9e4_requires_explicit_ack_and_thrust(self):
        with self.assertRaisesRegex(CompetitionMainSafetyError, "attitude-hover-smoke"):
            run_competition_main(
                CompetitionMainConfig(
                    mode=CompetitionRunnerMode.COMMAND_DRY_RUN,
                    live_transports=True,
                    px4_gazebo_attitude_hover_smoke=True,
                    px4_gazebo_attitude_hover_thrust=0.74,
                ),
                components=components_for_mode(CompetitionRunnerMode.COMMAND_DRY_RUN),
            )

        with self.assertRaisesRegex(CompetitionMainSafetyError, "attitude-hover-thrust"):
            run_competition_main(
                CompetitionMainConfig(
                    mode=CompetitionRunnerMode.COMMAND_DRY_RUN,
                    live_transports=True,
                    px4_gazebo_attitude_hover_smoke=True,
                    ack_px4_gazebo_surrogate_attitude_hover_smoke=True,
                ),
                components=components_for_mode(CompetitionRunnerMode.COMMAND_DRY_RUN),
            )

    def test_phase9e4_rejects_mixing_with_body_rate_smoke(self):
        with self.assertRaisesRegex(CompetitionMainSafetyError, "cannot be combined"):
            run_competition_main(
                CompetitionMainConfig(
                    mode=CompetitionRunnerMode.COMMAND_DRY_RUN,
                    live_transports=True,
                    px4_gazebo_body_rate_smoke=True,
                    ack_px4_gazebo_surrogate_body_rate_smoke=True,
                    px4_gazebo_body_rate_thrust=0.74,
                    px4_gazebo_attitude_hover_smoke=True,
                    ack_px4_gazebo_surrogate_attitude_hover_smoke=True,
                    px4_gazebo_attitude_hover_thrust=0.74,
                ),
                components=components_for_mode(CompetitionRunnerMode.COMMAND_DRY_RUN),
            )

    def test_phase9e4_attitude_hover_prestream_can_satisfy_offboard_gate(self):
        clock = IncrementingClock()
        connection = FakeConnection()
        sender = Px4GazeboSetAttitudeTargetSender(connection=connection)
        components = build_competition_runtime(
            CompetitionSetupConfig(mode=CompetitionRunnerMode.COMMAND_DRY_RUN),
            mavlink_transport=RepeatingFakeMavlinkTransport(
                summary_data=px4_armed_offboard_summary()
            ),
            vision_transport=FakeVisionTransport([]),
            image_adapter=FakeImageAdapter(),
            autonomy=None,
            clock=clock,
        )

        summary = run_competition_main(
            CompetitionMainConfig(
                mode=CompetitionRunnerMode.COMMAND_DRY_RUN,
                live_transports=True,
                px4_gazebo_attitude_hover_smoke=True,
                ack_px4_gazebo_surrogate_attitude_hover_smoke=True,
                px4_gazebo_attitude_hover_thrust=0.74,
                px4_gazebo_command_max_count=3,
                px4_gazebo_offboard=True,
                ack_px4_gazebo_surrogate_offboard=True,
                px4_gazebo_offboard_prestream_count=2,
                steps=4,
            ),
            components=components,
            command_sender=sender,
            clock=clock,
        ).to_dict()

        self.assertEqual(summary["phase"], PHASE_9E_4)
        self.assertEqual(summary["attitude_hover_command_sent_count"], 3)
        self.assertEqual(summary["offboard_sent_count"], 1)
        self.assertTrue(summary["offboard_state_observed"])
        self.assertTrue(
            summary["phase9e4_success_criteria"][
                "offboard_prestream_count_satisfied"
            ]
        )
        self.assertTrue(summary["phase9e4_attitude_hover_interface_satisfied"])
        self.assertEqual(len(connection.mav.set_attitude_target_calls), 3)
        self.assertEqual(connection.mav.command_long_calls[0][2], MAV_CMD_DO_SET_MODE)

    def test_phase9e_is_incomplete_without_armed_telemetry_evidence(self):
        connection = FakeConnection()
        sender = Px4GazeboSetAttitudeTargetSender(connection=connection)

        summary = run_competition_main(
            CompetitionMainConfig(
                mode=CompetitionRunnerMode.COMMAND_DRY_RUN,
                live_transports=True,
                use_real_autonomy=True,
                real_perception=True,
                allow_legacy_yolo_default=True,
                px4_gazebo_arm=True,
                ack_px4_gazebo_surrogate_arm=True,
            ),
            components=components_for_mode(
                CompetitionRunnerMode.COMMAND_DRY_RUN,
                with_vision=True,
                autonomy=FakeAutonomy(),
            ),
            command_sender=sender,
            clock=lambda: 100.0,
        ).to_dict()

        self.assertEqual(summary["phase"], PHASE_9E)
        self.assertEqual(summary["arm_sent_count"], 1)
        self.assertFalse(summary["armed_state_observed"])
        self.assertFalse(summary["phase9e_surrogate_arm_offboard_satisfied"])
        self.assertFalse(summary["competition_readiness_claimed"])

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
