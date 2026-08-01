import os
import unittest
from unittest.mock import patch

from runtime_config import load_runtime_config


class CompetitionGroundTruthDebugTests(unittest.TestCase):
    def test_competition_ground_truth_is_rejected_without_debug_opt_in(self):
        with patch.dict(
            os.environ,
            {
                "RUNNER_MODE": "competition",
                "GATE_SOURCE_MODE": "ground_truth",
                "ALLOW_COMPETITION_GROUND_TRUTH_DEBUG": "false",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "ALLOW_COMPETITION_GROUND_TRUTH_DEBUG=true",
            ):
                load_runtime_config()

    def test_competition_ground_truth_loads_with_debug_opt_in(self):
        with patch.dict(
            os.environ,
            {
                "RUNNER_MODE": "competition",
                "GATE_SOURCE_MODE": "ground_truth",
                "ALLOW_COMPETITION_GROUND_TRUTH_DEBUG": "true",
            },
            clear=False,
        ):
            config = load_runtime_config()

        self.assertEqual(config.gate_source.mode, "ground_truth")
        self.assertTrue(config.gate_source.allow_ground_truth)
        self.assertTrue(
            config.gate_source.allow_competition_ground_truth_debug
        )
        self.assertGreater(len(config.gate_source.known_gate_positions_neu), 0)


class ControllerRuntimeConfigTests(unittest.TestCase):
    def test_tilt_compensation_and_plan_validation_limit_are_enabled(self):
        with patch.dict(
            os.environ,
            {
                "RUNNER_MODE": "competition",
                "ALLOW_COMPETITION_GROUND_TRUTH_DEBUG": "true",
            },
            clear=False,
        ):
            config = load_runtime_config()

        self.assertTrue(config.controller.tilt_thrust_compensation_enabled)
        self.assertEqual(
            config.planner.plan_validation_max_acc_xy_m_s2,
            7.0,
        )
        self.assertTrue(config.planner.forward_progress_constraint_enabled)
        self.assertEqual(config.planner.forward_progress_min_speed_m_s, 0.0)


if __name__ == "__main__":
    unittest.main()
