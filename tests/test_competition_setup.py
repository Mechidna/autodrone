import subprocess
import sys
import unittest

from autonomy_core.runtime.competition_guard import GAZEBO_TRUTH_POSE_SOURCE
from autonomy_core.runtime.competition_autonomy_factory import (
    COMPETITION_SAFE_POSE_SOURCE,
    CompetitionAutonomyProfile,
)
from autonomy_core.runtime.competition_runner import CompetitionRunnerMode
from autonomy_core.runtime.competition_setup import (
    CompetitionSetupConfig,
    build_competition_runtime,
    create_autonomy_api,
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
    def __init__(self, messages=()):
        self.messages = list(messages)
        self.started = False
        self.close_calls = 0

    @property
    def is_started(self):
        return self.started

    def receive_messages(self):
        messages = self.messages
        self.messages = []
        return messages

    def close(self):
        self.close_calls += 1


class FakeVisionTransport:
    def __init__(self, packets=()):
        self.packets = list(packets)
        self.started = False
        self.close_calls = 0

    @property
    def is_started(self):
        return self.started

    def receive_packets(self):
        packets = self.packets
        self.packets = []
        return packets

    def close(self):
        self.close_calls += 1


class FakeAutonomy:
    def __init__(self, **kwargs):
        self.constructor_kwargs = dict(kwargs)
        self.perception_updates = []
        self.attitude_control_calls = 0
        self.perception_world_pose_source = GAZEBO_TRUTH_POSE_SOURCE
        self.save_perception_debug_frames = True
        self.use_diagnostic_far_depth_correction = True
        self.image_gazebo_pose_snapshot = {"pose": "truth"}

    def update_gate_memory_from_frame(self, **kwargs):
        self.perception_updates.append(kwargs)

    def attitude_control(self):
        self.attitude_control_calls += 1
        return (0.0, 0.0, 0.0, 0.5)


class CompetitionSetupTests(unittest.TestCase):
    def test_import_does_not_load_live_dependencies_or_autonomy(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import sys; "
                    "import autonomy_core.runtime.competition_setup; "
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

    def test_default_setup_constructs_components_without_starting_transports(self):
        components = build_competition_runtime()

        self.assertEqual(components.runner.mode, CompetitionRunnerMode.OBSERVE)
        self.assertIsNone(components.autonomy)
        self.assertFalse(components.mavlink_transport.is_started)
        self.assertFalse(components.vision_transport.is_started)
        self.assertIs(components.runner.mavlink_transport, components.mavlink_transport)
        self.assertIs(components.runner.vision_transport, components.vision_transport)
        self.assertIs(components.runner.guard, components.guard)
        self.assertIs(components.runner.state_adapter, components.state_adapter)
        self.assertIs(components.runner.image_adapter, components.image_adapter)
        self.assertIs(components.runner.command_adapter, components.command_adapter)

    def test_setup_uses_injected_transports_and_fake_autonomy(self):
        mavlink = FakeMavlinkTransport(
            [
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
        )
        vision = FakeVisionTransport()
        autonomy = FakeAutonomy()
        components = build_competition_runtime(
            CompetitionSetupConfig(mode=CompetitionRunnerMode.OBSERVE),
            mavlink_transport=mavlink,
            vision_transport=vision,
            autonomy=autonomy,
            clock=lambda: 100.0,
        )

        result = components.runner.step()

        self.assertEqual(result.telemetry_messages_processed, 3)
        self.assertTrue(result.heartbeat_seen)
        self.assertTrue(result.state_result.is_usable)
        self.assertIs(components.autonomy, autonomy)
        self.assertFalse(result.command_publication_allowed)
        self.assertFalse(mavlink.is_started)
        self.assertFalse(vision.is_started)

    def test_create_autonomy_api_uses_explicit_factory_only_when_called(self):
        calls = []

        def factory(**kwargs):
            calls.append(kwargs)
            return FakeAutonomy(**kwargs)

        autonomy = create_autonomy_api(autonomy_factory=factory)

        self.assertEqual(
            calls,
            [
                {
                    "use_perception": False,
                    "race_gate_count": None,
                    "race_gate_order": None,
                    "save_perception_debug_frames": False,
                    "use_lookahead_gate_filter": True,
                }
            ],
        )
        self.assertIsInstance(autonomy, FakeAutonomy)
        self.assertFalse(autonomy.constructor_kwargs["use_perception"])
        self.assertEqual(
            autonomy.perception_world_pose_source,
            COMPETITION_SAFE_POSE_SOURCE,
        )
        self.assertFalse(autonomy.save_perception_debug_frames)

    def test_setup_can_use_explicit_autonomy_factory(self):
        calls = []

        def factory(**kwargs):
            calls.append(kwargs)
            return FakeAutonomy(**kwargs)

        components = build_competition_runtime(
            CompetitionSetupConfig(
                use_real_autonomy=True,
                autonomy_profile=CompetitionAutonomyProfile(
                    use_perception=True,
                    allow_legacy_yolo_default=True,
                ),
            ),
            mavlink_transport=FakeMavlinkTransport(),
            vision_transport=FakeVisionTransport(),
            autonomy_factory=factory,
        )

        self.assertEqual(calls[0]["use_perception"], True)
        self.assertFalse(calls[0]["save_perception_debug_frames"])
        self.assertIsInstance(components.autonomy, FakeAutonomy)
        self.assertEqual(
            components.autonomy.perception_world_pose_source,
            COMPETITION_SAFE_POSE_SOURCE,
        )
        self.assertFalse(components.autonomy.save_perception_debug_frames)
        self.assertFalse(components.autonomy.use_diagnostic_far_depth_correction)
        self.assertIsNone(components.autonomy.image_gazebo_pose_snapshot)
        self.assertTrue(components.autonomy.competition_autonomy_profile_active)
        self.assertFalse(components.mavlink_transport.is_started)
        self.assertFalse(components.vision_transport.is_started)

    def test_setup_rejects_gazebo_truth_source(self):
        with self.assertRaisesRegex(Exception, "gazebo_truth_sim_only"):
            build_competition_runtime(
                CompetitionSetupConfig(
                    perception_world_pose_source=GAZEBO_TRUTH_POSE_SOURCE,
                ),
                mavlink_transport=FakeMavlinkTransport(),
                vision_transport=FakeVisionTransport(),
            )

    def test_components_close_delegates_to_transports(self):
        mavlink = FakeMavlinkTransport()
        vision = FakeVisionTransport()
        components = build_competition_runtime(
            mavlink_transport=mavlink,
            vision_transport=vision,
        )

        components.close()

        self.assertEqual(mavlink.close_calls, 1)
        self.assertEqual(vision.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
