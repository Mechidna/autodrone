import subprocess
import sys
import unittest
from dataclasses import replace

from autonomy_core.command.competition_command_adapter import (
    SET_ATTITUDE_TARGET_MESSAGE_NAME,
    build_dry_run_set_attitude_target_fields,
)
from autonomy_core.runtime.competition_setpoint_streamer import (
    SETPOINT_SOURCE_AUTONOMY,
    SETPOINT_SOURCE_FALLBACK,
    SETPOINT_SOURCE_NONE,
    CompetitionSetpointStreamConfig,
    CompetitionSetpointStreamer,
    CompetitionSetpointStreamerError,
)


def _fields(*, roll=0.0, pitch=0.0, yaw=0.0, thrust=0.5):
    return build_dry_run_set_attitude_target_fields(
        (roll, pitch, yaw, thrust),
        time_boot_ms=0,
        target_system=1,
        target_component=1,
    )


class CompetitionSetpointStreamerTest(unittest.TestCase):
    def test_import_safety_does_not_load_live_dependencies(self):
        probe = (
            "import sys\n"
            "import autonomy_core.runtime.competition_setpoint_streamer\n"
            "print(' '.join(str(name in sys.modules) for name in "
            "('pymavlink', 'cv2', 'mavsdk', 'rclpy')))\n"
        )
        completed = subprocess.run(
            [sys.executable, "-B", "-c", probe],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual("False False False False", completed.stdout.strip())

    def test_rejects_stream_rate_at_or_above_vadr_limit(self):
        with self.assertRaisesRegex(CompetitionSetpointStreamerError, "strictly below"):
            CompetitionSetpointStreamer(
                config=CompetitionSetpointStreamConfig(stream_rate_hz=100.0),
                fallback_fields=_fields(),
            )

        with self.assertRaisesRegex(CompetitionSetpointStreamerError, "positive"):
            CompetitionSetpointStreamer(
                config=CompetitionSetpointStreamConfig(stream_rate_hz=0.0),
                fallback_fields=_fields(),
            )

    def test_emits_fallback_when_autonomy_command_is_missing(self):
        streamer = CompetitionSetpointStreamer(fallback_fields=_fields(thrust=0.74))

        decision = streamer.step(now_s=10.0)

        self.assertTrue(decision.should_emit)
        self.assertEqual(SETPOINT_SOURCE_FALLBACK, decision.source)
        self.assertEqual("autonomy_command_missing", decision.reason)
        self.assertEqual(1, decision.sequence)
        self.assertEqual(10000, decision.fields.time_boot_ms)
        self.assertEqual(1, decision.fields.sequence)
        self.assertEqual(0.74, decision.fields.thrust)
        self.assertFalse(decision.phase4b_satisfied)
        self.assertFalse(decision.competition_readiness_claimed)
        self.assertEqual(1, streamer.stats.fallback_emit_count)

    def test_emits_fresh_autonomy_then_rate_limits(self):
        streamer = CompetitionSetpointStreamer(fallback_fields=_fields(thrust=0.74))
        streamer.update_autonomy_fields(_fields(yaw=1.57, thrust=0.6), now_s=20.0)

        first = streamer.step(now_s=20.0)
        limited = streamer.step(now_s=20.01)
        second = streamer.step(now_s=20.05)

        self.assertTrue(first.should_emit)
        self.assertEqual(SETPOINT_SOURCE_AUTONOMY, first.source)
        self.assertEqual("autonomy_command_fresh", first.reason)
        self.assertEqual(1, first.sequence)
        self.assertFalse(limited.should_emit)
        self.assertEqual(SETPOINT_SOURCE_NONE, limited.source)
        self.assertEqual("stream_rate_wait", limited.reason)
        self.assertTrue(second.should_emit)
        self.assertEqual(SETPOINT_SOURCE_AUTONOMY, second.source)
        self.assertEqual(2, second.sequence)
        self.assertEqual(1, streamer.stats.rate_limited_count)
        self.assertEqual(2, streamer.stats.autonomy_emit_count)

    def test_stale_autonomy_falls_back_to_hold_fields(self):
        config = CompetitionSetpointStreamConfig(
            stream_rate_hz=20.0,
            autonomy_command_fresh_s=0.5,
        )
        streamer = CompetitionSetpointStreamer(
            config=config,
            fallback_fields=_fields(thrust=0.74),
        )
        streamer.update_autonomy_fields(_fields(yaw=1.57, thrust=0.6), now_s=0.0)

        first = streamer.step(now_s=0.0)
        fallback = streamer.step(now_s=1.0)

        self.assertTrue(first.should_emit)
        self.assertEqual(SETPOINT_SOURCE_AUTONOMY, first.source)
        self.assertTrue(fallback.should_emit)
        self.assertEqual(SETPOINT_SOURCE_FALLBACK, fallback.source)
        self.assertEqual("autonomy_command_stale", fallback.reason)
        self.assertEqual(1.0, fallback.command_age_s)
        self.assertEqual(0.74, fallback.fields.thrust)
        self.assertEqual(1, streamer.stats.stale_autonomy_count)
        self.assertEqual(1, streamer.stats.fallback_emit_count)

    def test_no_fallback_blocks_missing_or_stale_autonomy(self):
        config = CompetitionSetpointStreamConfig(
            stream_rate_hz=20.0,
            autonomy_command_fresh_s=0.5,
        )
        streamer = CompetitionSetpointStreamer(config=config)

        missing = streamer.step(now_s=0.0)
        streamer.update_autonomy_fields(_fields(thrust=0.6), now_s=0.1)
        fresh = streamer.step(now_s=0.1)
        stale = streamer.step(now_s=1.0)

        self.assertFalse(missing.should_emit)
        self.assertEqual("fallback_command_missing", missing.reason)
        self.assertTrue(fresh.should_emit)
        self.assertEqual(SETPOINT_SOURCE_AUTONOMY, fresh.source)
        self.assertFalse(stale.should_emit)
        self.assertEqual("fallback_command_missing", stale.reason)
        self.assertEqual(2, streamer.stats.missing_fallback_count)

    def test_rejects_invalid_fields(self):
        invalid = replace(_fields(), message_name="OTHER")
        streamer = CompetitionSetpointStreamer()

        with self.assertRaisesRegex(
            CompetitionSetpointStreamerError,
            SET_ATTITUDE_TARGET_MESSAGE_NAME,
        ):
            streamer.update_autonomy_fields(invalid, now_s=0.0)

        self.assertEqual(1, streamer.stats.invalid_update_count)

    def test_summary_never_claims_competition_readiness(self):
        streamer = CompetitionSetpointStreamer(fallback_fields=_fields(thrust=0.74))
        streamer.step(now_s=1.0)

        summary = streamer.summary()

        self.assertFalse(summary["phase4b_satisfied"])
        self.assertFalse(summary["competition_readiness_claimed"])
        self.assertEqual(1, summary["stats"]["emit_count"])
        self.assertEqual(SETPOINT_SOURCE_FALLBACK, summary["stats"]["last_source"])


if __name__ == "__main__":
    unittest.main()
