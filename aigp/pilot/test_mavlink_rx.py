import math
import unittest
from dataclasses import replace
from types import SimpleNamespace

from mavlink_rx import MAVLinkRX
from runtime_config import load_runtime_config


def _config(runner_mode, *, yaw_inverted):
    config = load_runtime_config()
    return replace(
        config,
        runtime=replace(
            config.runtime,
            runner_mode=runner_mode,
            competition_yaw_inverted=yaw_inverted,
        ),
    )


def _attitude_message(yaw_deg, yawspeed):
    return SimpleNamespace(
        roll=0.1,
        pitch=-0.2,
        yaw=math.radians(yaw_deg),
        rollspeed=0.01,
        pitchspeed=-0.02,
        yawspeed=yawspeed,
        time_boot_ms=1234,
    )


class MAVLinkRXAttitudeTests(unittest.TestCase):
    def test_competition_inverts_incoming_yaw_and_yaw_rate(self):
        data = {}
        receiver = MAVLinkRX(
            None,
            data,
            config=_config("competition", yaw_inverted=True),
        )

        receiver.on_attitude(_attitude_message(-173.1, 0.25))

        attitude = data["attitude"]
        self.assertAlmostEqual(math.degrees(attitude["yaw"]), 173.1)
        self.assertAlmostEqual(attitude["yawspeed"], -0.25)
        self.assertAlmostEqual(
            math.degrees(attitude["yaw_mavlink_raw"]),
            -173.1,
        )
        self.assertAlmostEqual(attitude["yawspeed_mavlink_raw"], 0.25)
        self.assertTrue(attitude["competition_yaw_inverted"])

    def test_disabled_or_px4_mode_leaves_incoming_yaw_unchanged(self):
        for runner_mode, yaw_inverted in (
            ("competition", False),
            ("px4", True),
        ):
            with self.subTest(
                runner_mode=runner_mode,
                yaw_inverted=yaw_inverted,
            ):
                data = {}
                receiver = MAVLinkRX(
                    None,
                    data,
                    config=_config(
                        runner_mode,
                        yaw_inverted=yaw_inverted,
                    ),
                )

                receiver.on_attitude(_attitude_message(-42.0, 0.3))

                attitude = data["attitude"]
                self.assertAlmostEqual(
                    math.degrees(attitude["yaw"]),
                    -42.0,
                )
                self.assertAlmostEqual(attitude["yawspeed"], 0.3)
                self.assertFalse(attitude["competition_yaw_inverted"])


if __name__ == "__main__":
    unittest.main()
