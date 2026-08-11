import os
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

from runtime_config import _validate, load_runtime_config


class CompetitionGroundTruthDebugTests(unittest.TestCase):
    def test_competition_ground_truth_is_rejected_without_debug_opt_in(self):
        config = load_runtime_config()
        config = replace(
            config,
            runtime=replace(config.runtime, runner_mode="competition"),
            gate_source=replace(
                config.gate_source,
                mode="ground_truth",
                allow_ground_truth=True,
                allow_competition_ground_truth_debug=False,
            ),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "ALLOW_COMPETITION_GROUND_TRUTH_DEBUG=true",
        ):
            _validate(config)

    def test_competition_ground_truth_loads_with_debug_opt_in(self):
        config = load_runtime_config()
        config = replace(
            config,
            runtime=replace(config.runtime, runner_mode="competition"),
            gate_source=replace(
                config.gate_source,
                mode="ground_truth",
                allow_ground_truth=True,
                allow_competition_ground_truth_debug=True,
            ),
        )
        _validate(config)

        self.assertEqual(config.gate_source.mode, "ground_truth")
        self.assertTrue(config.gate_source.allow_ground_truth)
        self.assertTrue(
            config.gate_source.allow_competition_ground_truth_debug
        )
        self.assertGreater(len(config.gate_source.known_gate_positions_neu), 0)


class SafeRuntimeConfigTests(unittest.TestCase):
    def test_repository_default_is_observe_only_and_disarmed(self):
        with patch.dict(os.environ, {}, clear=True):
            config = load_runtime_config()

        self.assertTrue(config.runtime.observe_only)
        self.assertFalse(config.runtime.px4_offboard_enabled)
        self.assertFalse(config.runtime.px4_arm)
        self.assertFalse(config.runtime.competition_arm)
        self.assertFalse(config.hover_acquisition.enabled)
        self.assertFalse(config.thrust_scale_calibration.enabled)
        self.assertFalse(config.lateral_response_calibration.enabled)
        self.assertFalse(config.experimental_gate_vio_alignment.enabled)
        self.assertEqual(config.gate_source.mode, "perception")
        self.assertFalse(config.gate_source.allow_ground_truth)
        self.assertFalse(
            config.gate_source.allow_competition_ground_truth_debug
        )
        self.assertFalse(
            config.gate_memory.known_position_commit_filter_enabled
        )

    def test_missing_config_falls_back_to_observe_only_and_disarmed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = os.path.join(temp_dir, "missing-runtime.toml")
            with patch.dict(os.environ, {}, clear=True):
                config = load_runtime_config(missing_path)

        self.assertTrue(config.runtime.observe_only)
        self.assertFalse(config.runtime.px4_offboard_enabled)
        self.assertFalse(config.runtime.px4_arm)
        self.assertFalse(config.runtime.competition_arm)
        self.assertFalse(config.hover_acquisition.enabled)
        self.assertFalse(config.thrust_scale_calibration.enabled)
        self.assertFalse(config.lateral_response_calibration.enabled)
        self.assertFalse(config.experimental_gate_vio_alignment.enabled)
        self.assertEqual(config.gate_source.mode, "perception")
        self.assertFalse(config.gate_source.allow_ground_truth)
        self.assertFalse(
            config.gate_source.allow_competition_ground_truth_debug
        )
        self.assertFalse(
            config.gate_memory.known_position_commit_filter_enabled
        )


class ControllerRuntimeConfigTests(unittest.TestCase):
    def test_competition_prearm_sensor_hold_environment_override(self):
        with patch.dict(
            os.environ,
            {
                "RUNNER_MODE": "competition",
                "ALLOW_COMPETITION_GROUND_TRUTH_DEBUG": "true",
                "COMPETITION_PREARM_SENSOR_HOLD_S": "4.5",
            },
            clear=False,
        ):
            config = load_runtime_config()

        self.assertEqual(config.runtime.competition_prearm_sensor_hold_s, 4.5)

    def test_tilt_compensation_and_plan_validation_guards_are_enabled(self):
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
        self.assertGreaterEqual(
            config.planner.plan_validation_max_acc_xy_m_s2,
            config.controller.max_acc_xy,
        )
        self.assertTrue(config.planner.forward_progress_constraint_enabled)
        self.assertEqual(config.planner.forward_progress_min_speed_m_s, 0.0)


class ExperimentalGateVioAlignmentConfigTests(unittest.TestCase):
    def test_gate_vio_alignment_is_disabled_by_default(self):
        with patch.dict(
            os.environ,
            {
                "RUNNER_MODE": "competition",
                "ALLOW_COMPETITION_GROUND_TRUTH_DEBUG": "true",
                "EXPERIMENTAL_GATE_VIO_ALIGNMENT": "false",
            },
            clear=False,
        ):
            config = load_runtime_config()

        self.assertFalse(config.experimental_gate_vio_alignment.enabled)
        self.assertGreaterEqual(
            config.experimental_gate_vio_alignment.min_consistent_frames,
            2,
        )

    def test_enabling_alignment_without_perception_is_rejected(self):
        config = load_runtime_config()
        config = replace(
            config,
            runtime=replace(config.runtime, use_perception=False),
            experimental_gate_vio_alignment=replace(
                config.experimental_gate_vio_alignment,
                enabled=True,
            ),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "runtime.use_perception=true",
        ):
            _validate(config)


if __name__ == "__main__":
    unittest.main()
