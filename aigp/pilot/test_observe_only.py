import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from controller import Controller
from runtime_config import load_runtime_config


class _MavSenderSpy:
    def __init__(self):
        self.calls = []

    def command_long_send(self, *args):
        self.calls.append(args)


class _ConnectionSpy:
    def __init__(self):
        self.mav = _MavSenderSpy()
        self.target_system = 1
        self.target_component = 1
        self.mode_calls = []

    def mode_mapping(self):
        raise AssertionError("observe-only controller must not query flight modes")

    def set_mode(self, mode_name):
        self.mode_calls.append(mode_name)


class ObserveOnlyTests(unittest.TestCase):
    def test_environment_override_enables_observe_only(self):
        with patch.dict(os.environ, {"OBSERVE_ONLY": "true"}, clear=False):
            config = load_runtime_config()

        self.assertTrue(config.runtime.observe_only)

    def test_controller_methods_cannot_transmit_in_observe_only(self):
        with patch.dict(os.environ, {"OBSERVE_ONLY": "true"}, clear=False):
            config = load_runtime_config()
        connection = _ConnectionSpy()
        controller = Controller(
            connection,
            {"latest_autonomy_command": SimpleNamespace()},
            system_boot_ms=0,
            config=config,
        )

        with patch("controller.time.sleep"):
            self.assertFalse(controller.update())
        self.assertFalse(controller._send_attitude(1.0, 2.0, 3.0, 0.5))
        self.assertFalse(controller.arm())
        self.assertFalse(controller.send_sim_reset_command())
        self.assertFalse(controller.set_mode("OFFBOARD"))

        self.assertEqual(connection.mav.calls, [])
        self.assertEqual(connection.mode_calls, [])

    def test_live_openvins_locks_control_yaw_before_wire_inversion(self):
        environment = {
            "OBSERVE_ONLY": "false",
            "AIGP_LIVE_OPENVINS": "true",
            "AIGP_LIVE_OPENVINS_LOCK_YAW": "true",
            "AIGP_LIVE_OPENVINS_YAW_DEG": "180.0",
        }
        with patch.dict(os.environ, environment, clear=False):
            config = load_runtime_config()
            controller = Controller(
                _ConnectionSpy(),
                {},
                system_boot_ms=0,
                config=config,
            )

        with patch("controller.send_attitude_angle_flight_control") as send:
            controller._send_attitude(1.0, 2.0, 30.0, 0.5)
            controller._send_attitude(1.0, 2.0, -70.0, 0.5)

        first_yaw = send.call_args_list[0].args[4]
        second_yaw = send.call_args_list[1].args[4]
        self.assertAlmostEqual(first_yaw, second_yaw)
        self.assertAlmostEqual(abs(first_yaw), 180.0)


if __name__ == "__main__":
    unittest.main()
