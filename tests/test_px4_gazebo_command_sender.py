import subprocess
import sys
import unittest

from autonomy_core.command.competition_command_adapter import (
    AUTONOMY_ATTITUDE_ANGLE_TYPE_MASK,
    build_dry_run_set_attitude_target_fields,
    quaternion_wxyz_from_euler_zyx,
)
from autonomy_core.runtime.px4_gazebo_command_sender import (
    ATTITUDE_HOVER_BODY_RATES_IGNORE_TYPE_MASK,
    ATTITUDE_HOVER_ZERO_BODY_RATES,
    BODY_RATE_ATTITUDE_IGNORE_TYPE_MASK,
    BODY_RATE_DUMMY_QUATERNION,
    MAV_CMD_COMPONENT_ARM_DISARM,
    MAV_CMD_DO_SET_MODE,
    MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
    PX4_CUSTOM_MAIN_MODE_OFFBOARD,
    PX4_GAZEBO_ARM_OFFBOARD_LABEL,
    PX4_GAZEBO_ATTITUDE_HOVER_SMOKE_LABEL,
    PX4_GAZEBO_BODY_RATE_SMOKE_LABEL,
    PX4_GAZEBO_SURROGATE_LABEL,
    Px4GazeboCommandSenderConfig,
    Px4GazeboSetAttitudeTargetSender,
)


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


def valid_fields(sequence=1, thrust=0.5, roll=0.0, pitch=0.0):
    return build_dry_run_set_attitude_target_fields(
        (roll, pitch, 0.25, thrust),
        time_boot_ms=1234,
        target_system=1,
        target_component=1,
        sequence=sequence,
    )


class Px4GazeboCommandSenderTests(unittest.TestCase):
    def test_import_does_not_load_live_dependencies(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import sys; "
                    "import autonomy_core.runtime.px4_gazebo_command_sender; "
                    "print('pymavlink' in sys.modules, 'cv2' in sys.modules, "
                    "'mavsdk' in sys.modules, 'rclpy' in sys.modules)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.stdout.strip(), "False False False False")

    def test_sends_set_attitude_target_fields_to_injected_connection(self):
        connection = FakeConnection()
        sender = Px4GazeboSetAttitudeTargetSender(
            connection=connection,
            clock=lambda: 100.0,
        )

        result = sender.send_set_attitude_target(valid_fields(), now_s=100.0)

        self.assertTrue(result.sent)
        self.assertEqual(result.surrogate_label, PX4_GAZEBO_SURROGATE_LABEL)
        self.assertFalse(result.phase4b_satisfied)
        self.assertFalse(result.competition_readiness_claimed)
        self.assertEqual(len(connection.mav.set_attitude_target_calls), 1)
        self.assertEqual(
            connection.mav.set_attitude_target_calls[0],
            valid_fields().as_pymavlink_args(),
        )
        self.assertEqual(sender.stats.sent_count, 1)

    def test_rejects_second_command_at_or_above_100hz(self):
        connection = FakeConnection()
        sender = Px4GazeboSetAttitudeTargetSender(connection=connection)

        first = sender.send_set_attitude_target(valid_fields(sequence=1), now_s=10.0)
        second = sender.send_set_attitude_target(valid_fields(sequence=2), now_s=10.01)

        self.assertTrue(first.sent)
        self.assertFalse(second.sent)
        self.assertEqual(second.rejection_reason, "command_rate_limit")
        self.assertEqual(len(connection.mav.set_attitude_target_calls), 1)

    def test_rejects_unsafe_thrust(self):
        connection = FakeConnection()
        sender = Px4GazeboSetAttitudeTargetSender(
            config=Px4GazeboCommandSenderConfig(max_thrust=0.6),
            connection=connection,
        )

        result = sender.send_set_attitude_target(valid_fields(thrust=0.85), now_s=10.0)

        self.assertFalse(result.sent)
        self.assertEqual(result.rejection_reason, "thrust_safety_limit")
        self.assertEqual(len(connection.mav.set_attitude_target_calls), 0)

    def test_surrogate_thrust_clamp_changes_sent_field_only(self):
        connection = FakeConnection()
        sender = Px4GazeboSetAttitudeTargetSender(
            config=Px4GazeboCommandSenderConfig(
                enable_surrogate_thrust_clamp=True,
                surrogate_thrust_clamp_max=0.76,
            ),
            connection=connection,
        )

        result = sender.send_set_attitude_target(
            valid_fields(thrust=0.85),
            now_s=10.0,
            stream_source="current",
        )

        self.assertTrue(result.sent)
        self.assertTrue(result.thrust_clamped)
        self.assertEqual(result.raw_thrust, 0.85)
        self.assertEqual(result.thrust, 0.76)
        self.assertEqual(result.stream_source, "current")
        self.assertEqual(connection.mav.set_attitude_target_calls[0][8], 0.76)
        self.assertEqual(sender.stats.thrust_clamp_count, 1)
        self.assertEqual(sender.stats.last_raw_thrust, 0.85)
        self.assertEqual(sender.stats.last_sent_thrust, 0.76)
        self.assertEqual(sender.stats.stream_source_counts, {"current": 1})
        summary = sender.summary()
        self.assertTrue(summary["safety"]["enable_surrogate_thrust_clamp"])
        self.assertEqual(summary["stats"]["thrust_clamp_count"], 1)
        self.assertEqual(summary["stats"]["max_raw_thrust"], 0.85)
        self.assertEqual(summary["stats"]["max_sent_thrust"], 0.76)

    def test_records_stream_source_and_send_gap(self):
        connection = FakeConnection()
        sender = Px4GazeboSetAttitudeTargetSender(connection=connection)

        first = sender.send_set_attitude_target(
            valid_fields(sequence=1),
            now_s=10.0,
            stream_source="current",
        )
        second = sender.send_set_attitude_target(
            valid_fields(sequence=2),
            now_s=10.02,
            stream_source="cached",
        )

        self.assertTrue(first.sent)
        self.assertTrue(second.sent)
        self.assertAlmostEqual(second.send_gap_s, 0.02)
        self.assertEqual(
            sender.stats.stream_source_counts,
            {"cached": 1, "current": 1},
        )
        self.assertAlmostEqual(sender.stats.max_send_gap_s, 0.02)

    def test_sends_body_rate_set_attitude_target_to_injected_connection(self):
        connection = FakeConnection()
        sender = Px4GazeboSetAttitudeTargetSender(connection=connection)

        result = sender.send_body_rate_set_attitude_target(
            target_system=1,
            target_component=1,
            body_roll_rate=0.1,
            body_pitch_rate=-0.2,
            body_yaw_rate=0.3,
            thrust=0.74,
            time_boot_ms=1234,
            sequence=7,
            now_s=10.0,
        )

        self.assertTrue(result.sent)
        self.assertEqual(result.surrogate_label, PX4_GAZEBO_BODY_RATE_SMOKE_LABEL)
        self.assertEqual(result.type_mask, BODY_RATE_ATTITUDE_IGNORE_TYPE_MASK)
        self.assertEqual(result.q, BODY_RATE_DUMMY_QUATERNION)
        self.assertEqual(result.body_roll_rate, 0.1)
        self.assertEqual(result.body_pitch_rate, -0.2)
        self.assertEqual(result.body_yaw_rate, 0.3)
        self.assertEqual(result.thrust, 0.74)
        self.assertFalse(result.phase4b_satisfied)
        self.assertFalse(result.competition_readiness_claimed)
        self.assertEqual(
            connection.mav.set_attitude_target_calls[0],
            (
                1234,
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
        self.assertEqual(sender.stats.sent_count, 1)
        self.assertEqual(sender.stats.stream_source_counts, {"body_rate_smoke": 1})

    def test_sends_attitude_hover_set_attitude_target_to_injected_connection(self):
        connection = FakeConnection()
        sender = Px4GazeboSetAttitudeTargetSender(connection=connection)
        expected_q = quaternion_wxyz_from_euler_zyx(
            roll_rad=0.0,
            pitch_rad=0.0,
            yaw_rad=0.25,
        )

        result = sender.send_attitude_hover_set_attitude_target(
            target_system=1,
            target_component=1,
            roll_rad=0.0,
            pitch_rad=0.0,
            yaw_rad=0.25,
            thrust=0.74,
            time_boot_ms=1234,
            sequence=8,
            now_s=10.0,
        )

        self.assertTrue(result.sent)
        self.assertEqual(result.surrogate_label, PX4_GAZEBO_ATTITUDE_HOVER_SMOKE_LABEL)
        self.assertEqual(result.type_mask, ATTITUDE_HOVER_BODY_RATES_IGNORE_TYPE_MASK)
        self.assertEqual(result.type_mask, AUTONOMY_ATTITUDE_ANGLE_TYPE_MASK)
        self.assertEqual(result.q, expected_q)
        self.assertEqual(result.body_roll_rate, ATTITUDE_HOVER_ZERO_BODY_RATES[0])
        self.assertEqual(result.body_pitch_rate, ATTITUDE_HOVER_ZERO_BODY_RATES[1])
        self.assertEqual(result.body_yaw_rate, ATTITUDE_HOVER_ZERO_BODY_RATES[2])
        self.assertEqual(result.roll_rad, 0.0)
        self.assertEqual(result.pitch_rad, 0.0)
        self.assertAlmostEqual(result.yaw_rad, 0.25)
        self.assertEqual(result.thrust, 0.74)
        self.assertFalse(result.phase4b_satisfied)
        self.assertFalse(result.competition_readiness_claimed)
        self.assertEqual(
            connection.mav.set_attitude_target_calls[0],
            (
                1234,
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
        self.assertEqual(sender.stats.sent_count, 1)
        self.assertEqual(sender.stats.stream_source_counts, {"attitude_hover_smoke": 1})

    def test_attitude_hover_smoke_rejects_unsafe_pitch(self):
        connection = FakeConnection()
        sender = Px4GazeboSetAttitudeTargetSender(
            config=Px4GazeboCommandSenderConfig(max_abs_roll_pitch_rad=0.1),
            connection=connection,
        )

        result = sender.send_attitude_hover_set_attitude_target(
            target_system=1,
            target_component=1,
            roll_rad=0.0,
            pitch_rad=0.2,
            yaw_rad=0.25,
            thrust=0.74,
            time_boot_ms=1234,
            now_s=10.0,
        )

        self.assertFalse(result.sent)
        self.assertEqual(result.rejection_reason, "pitch_safety_limit")
        self.assertEqual(len(connection.mav.set_attitude_target_calls), 0)

    def test_body_rate_smoke_rejects_unsafe_body_rate(self):
        connection = FakeConnection()
        sender = Px4GazeboSetAttitudeTargetSender(
            config=Px4GazeboCommandSenderConfig(max_abs_body_rate_rad_s=0.5),
            connection=connection,
        )

        result = sender.send_body_rate_set_attitude_target(
            target_system=1,
            target_component=1,
            body_roll_rate=0.6,
            body_pitch_rate=0.0,
            body_yaw_rate=0.0,
            thrust=0.74,
            time_boot_ms=1234,
            now_s=10.0,
        )

        self.assertFalse(result.sent)
        self.assertEqual(result.rejection_reason, "body_roll_rate_safety_limit")
        self.assertEqual(len(connection.mav.set_attitude_target_calls), 0)

    def test_body_rate_smoke_obeys_rate_limit(self):
        connection = FakeConnection()
        sender = Px4GazeboSetAttitudeTargetSender(connection=connection)

        first = sender.send_body_rate_set_attitude_target(
            target_system=1,
            target_component=1,
            body_roll_rate=0.0,
            body_pitch_rate=0.0,
            body_yaw_rate=0.0,
            thrust=0.74,
            time_boot_ms=1000,
            sequence=1,
            now_s=10.0,
        )
        second = sender.send_body_rate_set_attitude_target(
            target_system=1,
            target_component=1,
            body_roll_rate=0.0,
            body_pitch_rate=0.0,
            body_yaw_rate=0.0,
            thrust=0.74,
            time_boot_ms=1010,
            sequence=2,
            now_s=10.01,
        )

        self.assertTrue(first.sent)
        self.assertFalse(second.sent)
        self.assertEqual(second.rejection_reason, "command_rate_limit")
        self.assertEqual(len(connection.mav.set_attitude_target_calls), 1)

    def test_surrogate_thrust_clamp_requires_a_bound(self):
        with self.assertRaisesRegex(ValueError, "at least one clamp bound"):
            Px4GazeboSetAttitudeTargetSender(
                config=Px4GazeboCommandSenderConfig(
                    enable_surrogate_thrust_clamp=True
                ),
                connection=FakeConnection(),
            )

    def test_rejects_unsafe_roll(self):
        connection = FakeConnection()
        sender = Px4GazeboSetAttitudeTargetSender(connection=connection)

        result = sender.send_set_attitude_target(valid_fields(roll=0.8), now_s=10.0)

        self.assertFalse(result.sent)
        self.assertEqual(result.rejection_reason, "roll_safety_limit")
        self.assertEqual(len(connection.mav.set_attitude_target_calls), 0)

    def test_summary_is_surrogate_only(self):
        sender = Px4GazeboSetAttitudeTargetSender(connection=FakeConnection())

        summary = sender.summary()

        self.assertEqual(summary["surrogate_label"], PX4_GAZEBO_SURROGATE_LABEL)
        self.assertEqual(
            summary["arm_offboard_surrogate_label"],
            PX4_GAZEBO_ARM_OFFBOARD_LABEL,
        )
        self.assertEqual(
            summary["attitude_hover_smoke_surrogate_label"],
            PX4_GAZEBO_ATTITUDE_HOVER_SMOKE_LABEL,
        )
        self.assertFalse(summary["phase4b_satisfied"])
        self.assertFalse(summary["competition_readiness_claimed"])

    def test_sends_arm_command_long_to_injected_connection(self):
        connection = FakeConnection()
        sender = Px4GazeboSetAttitudeTargetSender(connection=connection)

        result = sender.send_arm_command(
            target_system=1,
            target_component=1,
            now_s=10.0,
        )

        self.assertTrue(result.sent)
        self.assertEqual(result.command_name, "MAV_CMD_COMPONENT_ARM_DISARM")
        self.assertEqual(result.command_id, MAV_CMD_COMPONENT_ARM_DISARM)
        self.assertEqual(result.surrogate_label, PX4_GAZEBO_ARM_OFFBOARD_LABEL)
        self.assertFalse(result.phase4b_satisfied)
        self.assertFalse(result.competition_readiness_claimed)
        self.assertEqual(
            connection.mav.command_long_calls[0],
            (1, 1, MAV_CMD_COMPONENT_ARM_DISARM, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        self.assertEqual(sender.stats.arm_attempts, 1)
        self.assertEqual(sender.stats.arm_sent_count, 1)

    def test_sends_offboard_mode_command_long_to_injected_connection(self):
        connection = FakeConnection()
        sender = Px4GazeboSetAttitudeTargetSender(connection=connection)

        result = sender.send_offboard_mode_command(
            target_system=1,
            target_component=1,
            now_s=10.0,
        )

        self.assertTrue(result.sent)
        self.assertEqual(result.command_name, "MAV_CMD_DO_SET_MODE_OFFBOARD")
        self.assertEqual(result.command_id, MAV_CMD_DO_SET_MODE)
        self.assertEqual(
            connection.mav.command_long_calls[0],
            (
                1,
                1,
                MAV_CMD_DO_SET_MODE,
                0,
                float(MAV_MODE_FLAG_CUSTOM_MODE_ENABLED),
                float(PX4_CUSTOM_MAIN_MODE_OFFBOARD),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
        )
        self.assertEqual(sender.stats.offboard_attempts, 1)
        self.assertEqual(sender.stats.offboard_sent_count, 1)

    def test_lifecycle_command_rejects_missing_command_long_api(self):
        class ConnectionWithoutCommandLong:
            class Mav:
                pass

            mav = Mav()

        sender = Px4GazeboSetAttitudeTargetSender(
            connection=ConnectionWithoutCommandLong()
        )

        result = sender.send_arm_command(target_system=1, target_component=1)

        self.assertFalse(result.sent)
        self.assertIn("command_long_send", result.rejection_reason)
        self.assertEqual(sender.stats.lifecycle_rejection_count, 1)


if __name__ == "__main__":
    unittest.main()
