import math
import subprocess
import sys
import unittest

from autonomy_core.command.competition_command_adapter import (
    AUTONOMY_ATTITUDE_ANGLE_TYPE_MASK,
    AUTONOMY_TUPLE_SEMANTICS,
    COMMAND_SEND_BLOCKED_REASON,
    PYAIPILOT_BODY_RATE_REFERENCE_TYPE_MASK,
    SET_ATTITUDE_TARGET_MESSAGE_NAME,
    CompetitionCommandAdapterError,
    CompetitionDryRunCommandAdapter,
    autonomy_attitude_command_from_tuple,
    build_dry_run_set_attitude_target_fields,
    quaternion_wxyz_from_euler_zyx,
)


class CompetitionCommandAdapterTests(unittest.TestCase):
    def test_import_does_not_import_pymavlink(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import sys; "
                    "import autonomy_core.command.competition_command_adapter; "
                    "print('pymavlink' in sys.modules)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.stdout.strip(), "False")

    def test_autonomy_tuple_semantics_are_attitude_angles_not_body_rates(self):
        command = autonomy_attitude_command_from_tuple((0.1, -0.2, 0.3, 0.55))

        self.assertEqual(command.roll_rad, 0.1)
        self.assertEqual(command.pitch_rad, -0.2)
        self.assertEqual(command.yaw_rad, 0.3)
        self.assertEqual(command.thrust, 0.55)
        self.assertIn("attitude angle", command.semantics)
        self.assertIn("not a body-rate command", command.semantics)

    def test_zero_command_maps_to_dry_run_attitude_target_fields(self):
        fields = build_dry_run_set_attitude_target_fields(
            (0.0, 0.0, 0.0, 0.5),
            time_boot_ms=1234,
            target_system=1,
            target_component=2,
            sequence=7,
        )

        self.assertEqual(fields.message_name, SET_ATTITUDE_TARGET_MESSAGE_NAME)
        self.assertEqual(fields.time_boot_ms, 1234)
        self.assertEqual(fields.target_system, 1)
        self.assertEqual(fields.target_component, 2)
        self.assertEqual(fields.type_mask, AUTONOMY_ATTITUDE_ANGLE_TYPE_MASK)
        self.assertEqual(fields.q, (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(fields.body_roll_rate, 0.0)
        self.assertEqual(fields.body_pitch_rate, 0.0)
        self.assertEqual(fields.body_yaw_rate, 0.0)
        self.assertEqual(fields.thrust, 0.5)
        self.assertEqual(fields.sequence, 7)
        self.assertFalse(fields.send_ready)
        self.assertEqual(fields.send_blocked_reason, COMMAND_SEND_BLOCKED_REASON)
        self.assertEqual(
            fields.as_pymavlink_args(),
            (1234, 1, 2, AUTONOMY_ATTITUDE_ANGLE_TYPE_MASK, fields.q, 0.0, 0.0, 0.0, 0.5),
        )

    def test_yaw_command_uses_mavlink_quaternion_wxyz_order(self):
        yaw = math.pi / 2.0
        expected = (
            math.cos(yaw / 2.0),
            0.0,
            0.0,
            math.sin(yaw / 2.0),
        )

        actual = quaternion_wxyz_from_euler_zyx(
            roll_rad=0.0,
            pitch_rad=0.0,
            yaw_rad=yaw,
        )

        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value)

    def test_reference_body_rate_mask_is_documented_but_not_used_for_angle_tuple(self):
        self.assertEqual(PYAIPILOT_BODY_RATE_REFERENCE_TYPE_MASK, 128)
        self.assertEqual(AUTONOMY_ATTITUDE_ANGLE_TYPE_MASK, 7)

    def test_rejects_nonfinite_values_and_invalid_thrust(self):
        with self.assertRaises(CompetitionCommandAdapterError):
            autonomy_attitude_command_from_tuple((0.0, math.nan, 0.0, 0.5))
        with self.assertRaises(CompetitionCommandAdapterError):
            autonomy_attitude_command_from_tuple((math.inf, 0.0, 0.0, 0.5))
        with self.assertRaises(CompetitionCommandAdapterError):
            autonomy_attitude_command_from_tuple((0.0, 0.0, 0.0, 1.1))
        with self.assertRaises(CompetitionCommandAdapterError):
            autonomy_attitude_command_from_tuple((0.0, 0.0, 0.0))

    def test_rejects_invalid_target_and_timestamp_fields(self):
        with self.assertRaises(CompetitionCommandAdapterError):
            build_dry_run_set_attitude_target_fields(
                (0.0, 0.0, 0.0, 0.5),
                time_boot_ms=-1,
                target_system=1,
                target_component=1,
            )
        with self.assertRaises(CompetitionCommandAdapterError):
            build_dry_run_set_attitude_target_fields(
                (0.0, 0.0, 0.0, 0.5),
                time_boot_ms=1,
                target_system=256,
                target_component=1,
            )

    def test_dry_run_adapter_rate_limits_without_sending(self):
        adapter = CompetitionDryRunCommandAdapter()

        first = adapter.build_set_attitude_target(
            (0.0, 0.0, 0.0, 0.5),
            time_boot_ms=10,
            target_system=1,
            target_component=1,
            now_s=100.0,
        )
        limited = adapter.build_set_attitude_target(
            (0.0, 0.0, 0.0, 0.5),
            time_boot_ms=11,
            target_system=1,
            target_component=1,
            now_s=100.005,
        )
        later = adapter.build_set_attitude_target(
            (0.0, 0.0, 0.0, 0.5),
            time_boot_ms=12,
            target_system=1,
            target_component=1,
            now_s=100.011,
        )

        self.assertTrue(first.accepted)
        self.assertIsNotNone(first.fields)
        self.assertFalse(first.fields.send_ready)
        self.assertFalse(limited.accepted)
        self.assertEqual(limited.rejection_reason, "command_rate_limit")
        self.assertTrue(later.accepted)

    def test_invalid_command_does_not_consume_rate_slot(self):
        adapter = CompetitionDryRunCommandAdapter()

        invalid = adapter.build_set_attitude_target(
            (0.0, 0.0, 0.0, math.nan),
            time_boot_ms=10,
            target_system=1,
            target_component=1,
            now_s=100.0,
        )
        valid = adapter.build_set_attitude_target(
            (0.0, 0.0, 0.0, 0.5),
            time_boot_ms=11,
            target_system=1,
            target_component=1,
            now_s=100.001,
        )

        self.assertFalse(invalid.accepted)
        self.assertIn("thrust must be finite", invalid.rejection_reason)
        self.assertTrue(valid.accepted)
        self.assertIsNotNone(valid.fields)
        self.assertEqual(valid.fields.source_tuple_semantics, AUTONOMY_TUPLE_SEMANTICS)

    def test_invalid_now_rejects_without_sending(self):
        adapter = CompetitionDryRunCommandAdapter()

        result = adapter.build_set_attitude_target(
            (0.0, 0.0, 0.0, 0.5),
            time_boot_ms=11,
            target_system=1,
            target_component=1,
            now_s=math.inf,
        )

        self.assertFalse(result.accepted)
        self.assertIsNone(result.fields)
        self.assertIn("now_s must be finite", result.rejection_reason)


if __name__ == "__main__":
    unittest.main()
