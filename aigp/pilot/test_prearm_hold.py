import unittest

from prearm_hold import hold_before_competition_arm


class _RecorderSpy:
    enabled = True

    def __init__(self, trace):
        self.trace = trace

    def record_event(self, event_type, **details):
        self.trace.append(("event", event_type, details))
        return True


class CompetitionPrearmSensorHoldTests(unittest.TestCase):
    def test_hold_records_before_and_after_sleep(self):
        trace = []
        times = iter((10.0, 13.05))

        elapsed_s = hold_before_competition_arm(
            3.0,
            recorder=_RecorderSpy(trace),
            sleep_fn=lambda duration: trace.append(("sleep", duration)),
            monotonic_fn=lambda: next(times),
            print_fn=lambda message, **kwargs: trace.append(("print", message)),
        )

        self.assertAlmostEqual(elapsed_s, 3.05)
        self.assertEqual(trace[0][1], "competition_prearm_sensor_hold_started")
        self.assertEqual(trace[2], ("sleep", 3.0))
        self.assertEqual(trace[3][1], "competition_prearm_sensor_hold_completed")
        self.assertIn("complete", trace[4][1])

    def test_zero_duration_does_nothing(self):
        trace = []

        elapsed_s = hold_before_competition_arm(
            0.0,
            recorder=_RecorderSpy(trace),
            sleep_fn=lambda duration: trace.append(("sleep", duration)),
            monotonic_fn=lambda: 0.0,
            print_fn=lambda message, **kwargs: trace.append(("print", message)),
        )

        self.assertEqual(elapsed_s, 0.0)
        self.assertEqual(trace, [])

    def test_invalid_duration_is_rejected(self):
        for duration_s in (-1.0, float("nan"), float("inf")):
            with self.subTest(duration_s=duration_s):
                with self.assertRaises(ValueError):
                    hold_before_competition_arm(duration_s)


if __name__ == "__main__":
    unittest.main()
