import math
import os
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import perception_adapter as perception_adapter_module
from autonomy_adapter import AutonomyAdapter
from autonomy_wrapper import PyAIPilotAutonomyAPI
from controller import Controller, get_current_yaw_deg
from hover_acquisition import HoverAcquisition
from lateral_response_calibration import LateralResponseCalibration
from runtime_config import load_runtime_config
from thrust_scale_calibration import ThrustScaleCalibration


def _calibration_only_config():
    config = load_runtime_config()
    return replace(
        config,
        runtime=replace(
            config.runtime,
            runner_mode="competition",
            calibration_only=True,
            perception_hold=False,
        ),
    )


def _normal_config(runner_mode):
    config = load_runtime_config()
    return replace(
        config,
        runtime=replace(
            config.runtime,
            runner_mode=runner_mode,
            calibration_only=False,
            perception_hold=False,
            startup_observation_duration_s=0.0,
        ),
    )


def _perception_hold_config():
    config = load_runtime_config()
    return replace(
        config,
        runtime=replace(
            config.runtime,
            runner_mode="competition",
            use_perception=True,
            calibration_only=False,
            perception_hold=True,
        ),
    )


class _RecordingMav:
    def __init__(self):
        self.attitude_targets = []
        self.position_targets = []

    def set_attitude_target_send(self, *args):
        self.attitude_targets.append(args)

    def set_position_target_local_ned_send(self, *args):
        self.position_targets.append(args)


class _Connection:
    def __init__(self):
        self.target_system = 1
        self.target_component = 1
        self.mav = _RecordingMav()


def _quaternion_yaw_deg(quaternion):
    w, x, y, z = quaternion
    yaw_rad = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return math.degrees(yaw_rad)


class _ForbiddenHoverHold:
    def __init__(self):
        self.reset_called = False

    def reset(self):
        self.reset_called = True

    def update_and_send(self, *_args, **_kwargs):
        raise AssertionError("attitude-only fallback must not call HoverHold")


class _RecordingHoverHold:
    def __init__(self):
        self.update_calls = 0
        self.reset_called = False

    def reset(self):
        self.reset_called = True

    def update_and_send(self, *_args, **_kwargs):
        self.update_calls += 1
        return True


class _StateEstimator:
    def __init__(self, estimate):
        self.estimate = estimate

    def update(self, _snapshot):
        return self.estimate

    def project_perception_with_estimated_state(
        self,
        latest_perception,
        _estimate,
        _snapshot,
    ):
        return latest_perception


class CalibrationOnlyTests(unittest.TestCase):
    def test_environment_override_enables_and_disables_mode(self):
        with patch.dict(
            os.environ,
            {"CALIBRATION_ONLY": "true", "PERCEPTION_HOLD": "false"},
        ):
            self.assertTrue(load_runtime_config().runtime.calibration_only)

        with patch.dict(
            os.environ,
            {"CALIBRATION_ONLY": "false", "PERCEPTION_HOLD": "false"},
        ):
            self.assertFalse(load_runtime_config().runtime.calibration_only)

        with patch.dict(
            os.environ,
            {"COMPETITION_YAW_INVERTED": "false"},
        ):
            self.assertFalse(
                load_runtime_config().runtime.competition_yaw_inverted
            )

    def test_perception_adapter_does_not_start_in_calibration_only_mode(self):
        data = {}
        with patch.object(
            perception_adapter_module,
            "PerceptionWrapper",
            side_effect=AssertionError("YOLO/perception must not be constructed"),
        ):
            adapter = perception_adapter_module.PerceptionAdapter(
                data,
                config=_calibration_only_config(),
            )

        self.assertFalse(adapter.enabled)
        self.assertFalse(adapter.is_running)
        self.assertIsNone(adapter.thread)
        self.assertEqual(data["latest_perception_status"], "calibration_only")

    def test_snapshot_does_not_require_camera_frame(self):
        adapter = AutonomyAdapter.__new__(AutonomyAdapter)
        adapter.config = _calibration_only_config()
        race_status = {
            "sim_boot_time_ms": 4_000,
            "race_start_boot_time_ms": 5_000,
        }

        snapshot = adapter.build_snapshot(
            frame=None,
            attitude={
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -math.pi,
                "wall_time": 1.0,
            },
            imu={
                "accel_xyz": (0.0, 0.0, -9.81),
                "gyro_xyz": (0.0, 0.0, 0.0),
                "wall_time": 1.0,
            },
            race_status=race_status,
        )

        self.assertIsNone(snapshot.image_bgr)
        self.assertEqual(snapshot.image_shape, ())
        self.assertEqual(snapshot.frame_id, -1)
        self.assertEqual(snapshot.race_status, race_status)
        self.assertIsNot(snapshot.race_status, race_status)

    def test_no_command_fallback_is_attitude_only_and_preserves_yaw(self):
        connection = _Connection()
        data = {
            "latest_autonomy_command": None,
            "latest_autonomy_command_status": "missing_inputs",
            "attitude": {"yaw": -math.pi},
        }
        controller = Controller(
            connection,
            data,
            system_boot_ms=0,
            config=_calibration_only_config(),
        )
        hover_hold = _ForbiddenHoverHold()
        controller.hover_hold = hover_hold

        with patch("controller.time.sleep", return_value=None):
            controller.update()

        self.assertTrue(hover_hold.reset_called)
        self.assertEqual(connection.mav.position_targets, [])
        self.assertEqual(len(connection.mav.attitude_targets), 1)
        attitude_args = connection.mav.attitude_targets[0]
        quaternion = attitude_args[4]
        self.assertAlmostEqual(abs(quaternion[3]), 1.0)
        self.assertAlmostEqual(
            attitude_args[8],
            controller.fallback_thrust,
        )
        self.assertAlmostEqual(get_current_yaw_deg(data), -180.0)

    def test_calibrations_wait_for_explicit_armed_state(self):
        config = _calibration_only_config()
        snapshot = SimpleNamespace(
            accel_xyz=np.array([0.0, 0.0, -9.81], dtype=float),
            roll_rad=0.0,
            pitch_rad=0.0,
            yaw_rad=0.0,
            armed=None,
        )
        estimate = SimpleNamespace(
            valid=True,
            confidence=1.0,
            pos_neu=np.array([0.0, 0.0, 1.0], dtype=float),
            vel_neu=np.zeros(3, dtype=float),
            yaw_rad=0.0,
        )

        acquisition_stage = HoverAcquisition(config)
        thrust_scale_stage = ThrustScaleCalibration(config)
        lateral_stage = LateralResponseCalibration(config)

        acquisition = acquisition_stage.update(
            snapshot=snapshot,
            estimate=estimate,
            hover_thrust=0.3,
            now=0.0,
        )
        thrust_scale = thrust_scale_stage.update(
            snapshot=snapshot,
            estimate=estimate,
            hover_thrust=0.3,
            hover_acquisition_completed=True,
            current_thrust_from_acc_gain=1.0 / 9.81,
            now=0.0,
        )
        lateral = lateral_stage.update(
            snapshot=snapshot,
            estimate=estimate,
            hover_thrust=0.3,
            thrust_scale_calibration_completed=True,
            current_lateral_accel_gain_xy=np.ones(2, dtype=float),
            now=0.0,
        )

        self.assertEqual(acquisition.debug.status, "waiting_armed")
        self.assertEqual(thrust_scale.debug.status, "waiting_armed")
        self.assertEqual(lateral.debug.status, "waiting_armed")
        for stage in (
            acquisition_stage,
            thrust_scale_stage,
            lateral_stage,
        ):
            self.assertFalse(stage.completed)
            self.assertIsNone(stage.start_time)

    def test_completed_calibrations_hold_learned_thrust_without_planning(self):
        api, snapshot, estimate = self._completed_calibration_api(
            pos_neu=[1.0, 2.0, 3.0],
            vel_neu=[0.0, 0.0, 0.0],
            yaw_rad=-math.pi,
        )
        self.assertFalse(api.use_perception)

        api.adaptive_hover.set_value(0.31, status="test")
        command = api.update(snapshot)

        self.assertIsNotNone(command)
        self.assertAlmostEqual(command.roll_rad, 0.0)
        self.assertAlmostEqual(command.pitch_rad, 0.0)
        self.assertAlmostEqual(command.yaw_rad, -math.pi)
        self.assertAlmostEqual(command.thrust, 0.31)

        estimate.pos_neu = np.array([1.0, 2.0, 3.2], dtype=float)
        estimate.vel_neu = np.array([0.0, 0.0, 0.5], dtype=float)
        damped_command = api.update(snapshot)
        self.assertLess(damped_command.thrust, 0.31)

        api._install_gate_centers.assert_not_called()
        api._should_plan.assert_not_called()

    def test_failed_thrust_calibration_holds_and_blocks_lateral_stage(self):
        api, snapshot, estimate = self._completed_calibration_api(
            pos_neu=[1.0, 2.0, 3.0],
            vel_neu=[0.2, 0.0, 0.4],
            yaw_rad=-math.pi,
        )
        api.adaptive_hover.set_value(0.267, status="test")
        api.thrust_scale_calibration.succeeded = False
        api.lateral_response_calibration.completed = False
        api.lateral_response_calibration.update = Mock(
            side_effect=AssertionError(
                "failed thrust calibration must not unlock lateral calibration"
            )
        )
        api._trace_thrust_scale_failure_hold = Mock()

        command = api.update(snapshot)

        self.assertIsNotNone(command)
        self.assertGreater(math.hypot(command.roll_rad, command.pitch_rad), 0.0)
        self.assertLessEqual(
            max(abs(command.roll_rad), abs(command.pitch_rad)),
            math.radians(api.config.lateral_response_calibration.max_tilt_deg),
        )
        self.assertAlmostEqual(command.yaw_rad, -math.pi)
        self.assertLess(command.thrust, 0.267)
        self.assertAlmostEqual(api._thrust_scale_failure_hold_z_m, 3.0)
        np.testing.assert_allclose(
            api._thrust_scale_failure_hold_xy_m,
            np.array([1.0, 2.0], dtype=float),
        )
        api.lateral_response_calibration.update.assert_not_called()
        api._trace_thrust_scale_failure_hold.assert_called_once()

        estimate.pos_neu = np.array([1.1, 2.0, 2.8], dtype=float)
        estimate.vel_neu = np.array([0.0, 0.0, -0.2], dtype=float)
        recovery = api.update(snapshot)
        self.assertGreater(recovery.thrust, 0.267)
        self.assertAlmostEqual(api._thrust_scale_failure_hold_z_m, 3.0)
        np.testing.assert_allclose(
            api._thrust_scale_failure_hold_xy_m,
            np.array([1.0, 2.0], dtype=float),
        )

    def test_failed_lateral_calibration_holds_xy_z_and_never_claims_completion(self):
        api, snapshot, estimate = self._completed_calibration_api(
            pos_neu=[1.0, 2.0, 3.0],
            vel_neu=[0.3, 0.0, 0.4],
            yaw_rad=0.0,
        )
        api.adaptive_hover.set_value(0.267, status="test")
        prior_gain = np.array([0.9, 1.1], dtype=float)
        api.tracker.lateral_accel_gain_xy = prior_gain.copy()
        api.lateral_response_calibration.completed = False
        api.lateral_response_calibration.succeeded = False
        def fail_lateral_calibration(**_kwargs):
            api.lateral_response_calibration.completed = True
            api.lateral_response_calibration.succeeded = False
            return SimpleNamespace(
                lateral_accel_gain_xy=np.array([1.25, 0.5], dtype=float),
                succeeded=False,
                command=None,
                debug=SimpleNamespace(
                    active=False,
                    completed=True,
                    succeeded=False,
                ),
            )

        api.lateral_response_calibration.update = Mock(
            side_effect=fail_lateral_calibration
        )
        api._trace_lateral_response_calibration = Mock()
        api._trace_lateral_response_failure_hold = Mock()

        command = api.update(snapshot)

        self.assertIsNotNone(command)
        self.assertLess(command.pitch_rad, 0.0)
        self.assertLessEqual(
            abs(command.pitch_rad),
            math.radians(api.config.lateral_response_calibration.max_tilt_deg),
        )
        self.assertLess(command.thrust, 0.267)
        np.testing.assert_allclose(
            api._lateral_response_failure_hold_xy_m,
            np.array([1.0, 2.0], dtype=float),
        )
        self.assertAlmostEqual(api._lateral_response_failure_hold_z_m, 3.0)
        self.assertIsNone(api._calibration_only_hold_xy_m)
        self.assertIsNone(api._calibration_only_hold_z_m)
        np.testing.assert_allclose(
            api.tracker.lateral_accel_gain_xy,
            prior_gain,
        )
        api.lateral_response_calibration.update.assert_called_once()
        api._trace_lateral_response_calibration.assert_called_once()
        api._trace_lateral_response_failure_hold.assert_called_once()
        api._trace_calibration_only_hold.assert_not_called()
        api._should_plan.assert_not_called()

        estimate.pos_neu = np.array([1.2, 2.0, 2.8], dtype=float)
        estimate.vel_neu = np.array([0.0, 0.0, -0.2], dtype=float)
        recovery = api.update(snapshot)
        self.assertGreater(recovery.thrust, 0.267)
        np.testing.assert_allclose(
            api._lateral_response_failure_hold_xy_m,
            np.array([1.0, 2.0], dtype=float),
        )
        self.assertAlmostEqual(api._lateral_response_failure_hold_z_m, 3.0)

    def test_disabled_or_skipped_lateral_stage_does_not_block_post_calibration_hold(self):
        for stage_overrides in (
            {"enabled": False, "estimator_mode_only": False},
            {"enabled": True, "estimator_mode_only": True},
        ):
            with self.subTest(**stage_overrides):
                api, snapshot, _estimate = self._completed_calibration_api(
                    pos_neu=[1.0, 2.0, 3.0],
                    vel_neu=[0.0, 0.0, 0.0],
                    yaw_rad=0.0,
                )
                api.lateral_response_calibration.completed = False
                api.lateral_response_calibration.succeeded = False
                api.lateral_response_calibration.enabled = stage_overrides["enabled"]
                api.lateral_response_calibration.estimator_mode_only = (
                    stage_overrides["estimator_mode_only"]
                )
                api._trace_lateral_response_failure_hold = Mock()

                command = api.update(snapshot)

                self.assertIsNotNone(command)
                self.assertTrue(api.lateral_response_calibration.completed)
                self.assertFalse(api.lateral_response_calibration.succeeded)
                self.assertIsNotNone(api._calibration_only_hold_xy_m)
                self.assertIsNotNone(api._calibration_only_hold_z_m)
                api._trace_lateral_response_failure_hold.assert_not_called()
                api._trace_calibration_only_hold.assert_called_once()

    def test_post_calibration_hold_brakes_and_sends_only_attitude_targets(self):
        api, snapshot, _estimate = self._completed_calibration_api(
            pos_neu=[1.0, 2.0, 3.0],
            vel_neu=[0.3, 0.0, 0.0],
            yaw_rad=0.0,
        )
        api.adaptive_hover.set_value(0.31, status="test")
        api.tracker.lateral_accel_gain_xy = np.array([1.25, 1.0], dtype=float)

        command = api.update(snapshot)

        expected_pitch = -math.atan2(0.4 * 1.25, 9.81)
        self.assertAlmostEqual(command.roll_rad, 0.0, places=6)
        self.assertAlmostEqual(command.pitch_rad, expected_pitch, places=6)
        self.assertAlmostEqual(command.thrust, 0.31)
        np.testing.assert_allclose(
            api._calibration_only_hold_xy_m,
            np.array([1.0, 2.0], dtype=float),
        )

        connection = _Connection()
        data = {
            "latest_autonomy_command": SimpleNamespace(
                roll_deg=math.degrees(command.roll_rad),
                pitch_deg=math.degrees(command.pitch_rad),
                yaw_deg=math.degrees(command.yaw_rad),
                thrust=command.thrust,
            ),
            "latest_autonomy_command_status": "ok",
            "attitude": {"yaw": 0.0},
        }
        controller = Controller(
            connection,
            data,
            system_boot_ms=0,
            config=_calibration_only_config(),
        )
        hover_hold = _ForbiddenHoverHold()
        controller.hover_hold = hover_hold

        with patch("controller.time.sleep", return_value=None):
            controller.update()

        self.assertTrue(hover_hold.reset_called)
        self.assertEqual(connection.mav.position_targets, [])
        self.assertEqual(len(connection.mav.attitude_targets), 1)

    def test_post_calibration_hold_brakes_positive_x_at_yaw_pi(self):
        api, snapshot, _estimate = self._completed_calibration_api(
            pos_neu=[1.0, 2.0, 3.0],
            vel_neu=[0.3, 0.0, 0.0],
            yaw_rad=-math.pi,
        )
        api.tracker.lateral_accel_gain_xy = np.ones(2, dtype=float)

        command = api.update(snapshot)

        expected_pitch = math.atan2(0.4, 9.81)
        self.assertAlmostEqual(command.roll_rad, 0.0, places=6)
        self.assertAlmostEqual(command.pitch_rad, expected_pitch, places=6)
        self.assertAlmostEqual(command.yaw_rad, -math.pi)

    def test_polarity_mismatch_failure_uses_level_attitude(self):
        api, snapshot, _estimate = self._completed_calibration_api(
            pos_neu=[1.0, 2.0, 3.0],
            vel_neu=[0.3, -0.2, 0.0],
            yaw_rad=-math.pi,
        )
        api.lateral_response_calibration.completed = True
        api.lateral_response_calibration.succeeded = False
        api.lateral_response_calibration.last_debug = SimpleNamespace(
            status="polarity_mismatch_fallback"
        )

        command = api.update(snapshot)

        self.assertAlmostEqual(command.roll_rad, 0.0)
        self.assertAlmostEqual(command.pitch_rad, 0.0)
        self.assertAlmostEqual(command.yaw_rad, -math.pi)

    def test_post_calibration_hold_restores_captured_xy_position(self):
        api, snapshot, estimate = self._completed_calibration_api(
            pos_neu=[1.0, 2.0, 3.0],
            vel_neu=[0.0, 0.0, 0.0],
            yaw_rad=0.0,
        )

        initial_command = api.update(snapshot)
        self.assertAlmostEqual(initial_command.roll_rad, 0.0)
        self.assertAlmostEqual(initial_command.pitch_rad, 0.0)
        np.testing.assert_allclose(
            api._calibration_only_hold_xy_m,
            np.array([1.0, 2.0], dtype=float),
        )

        estimate.pos_neu = np.array([1.5, 2.0, 3.0], dtype=float)
        negative_pitch_command = api.update(snapshot)
        self.assertLess(negative_pitch_command.pitch_rad, 0.0)

        estimate.pos_neu = np.array([0.5, 2.0, 3.0], dtype=float)
        positive_pitch_command = api.update(snapshot)
        self.assertGreater(positive_pitch_command.pitch_rad, 0.0)
        np.testing.assert_allclose(
            api._calibration_only_hold_xy_m,
            np.array([1.0, 2.0], dtype=float),
        )

    def test_post_calibration_hold_rotates_braking_with_yaw(self):
        api, snapshot, _estimate = self._completed_calibration_api(
            pos_neu=[1.0, 2.0, 3.0],
            vel_neu=[0.3, 0.0, 0.0],
            yaw_rad=math.pi / 2.0,
        )
        api.tracker.lateral_accel_gain_xy = np.ones(2, dtype=float)

        command = api.update(snapshot)

        expected_roll = math.atan2(0.4, 9.81)
        self.assertAlmostEqual(command.roll_rad, expected_roll, places=6)
        self.assertAlmostEqual(command.pitch_rad, 0.0, places=6)
        self.assertAlmostEqual(command.yaw_rad, math.pi / 2.0)

    def test_post_calibration_hold_caps_physical_accel_and_tilt(self):
        api, snapshot, _estimate = self._completed_calibration_api(
            pos_neu=[1.0, 2.0, 3.0],
            vel_neu=[100.0, 100.0, 0.0],
            yaw_rad=0.0,
        )
        api.tracker.lateral_accel_gain_xy = np.array([100.0, 100.0], dtype=float)

        command = api.update(snapshot)

        trace_args = api._trace_calibration_only_hold.call_args.args
        hold_debug = trace_args[5]
        physical_accel = np.asarray(
            hold_debug["accel_xy_m_s2"],
            dtype=float,
        )
        attitude_accel = np.asarray(
            hold_debug["attitude_accel_xy_m_s2"],
            dtype=float,
        )
        physical_limit = min(
            float(api.config.controller.max_acc_xy),
            float(api.lateral_response_calibration.probe_accel_m_s2),
        )
        self.assertAlmostEqual(
            float(np.linalg.norm(physical_accel)),
            physical_limit,
            places=6,
        )
        np.testing.assert_allclose(attitude_accel, physical_accel * 100.0)
        max_tilt_rad = math.radians(
            float(api.config.lateral_response_calibration.max_tilt_deg)
        )
        self.assertLessEqual(abs(command.roll_rad), max_tilt_rad + 1.0e-12)
        self.assertLessEqual(abs(command.pitch_rad), max_tilt_rad + 1.0e-12)
        self.assertAlmostEqual(command.thrust, api.adaptive_hover.value)

    @staticmethod
    def _completed_calibration_api(
        *,
        pos_neu,
        vel_neu,
        yaw_rad,
    ):
        config = _calibration_only_config()
        api = PyAIPilotAutonomyAPI(
            use_perception=True,
            config=config,
        )
        estimate = SimpleNamespace(
            valid=True,
            confidence=1.0,
            pos_neu=np.asarray(pos_neu, dtype=float),
            vel_neu=np.asarray(vel_neu, dtype=float),
            yaw_rad=float(yaw_rad),
        )
        api.state_estimator = _StateEstimator(estimate)
        api.shadow_state_estimator = None
        api.hover_acquisition.completed = True
        api.thrust_scale_calibration.completed = True
        api.thrust_scale_calibration.succeeded = True
        api.lateral_response_calibration.completed = True
        api.lateral_response_calibration.succeeded = True
        api._install_gate_centers = Mock(
            side_effect=AssertionError("calibration-only mode must not install gates")
        )
        api._should_plan = Mock(
            side_effect=AssertionError("calibration-only mode must not plan")
        )
        api._trace_calibration_only_hold = Mock()
        snapshot = SimpleNamespace(latest_perception=None)
        return api, snapshot, estimate


class PerceptionHoldTests(unittest.TestCase):
    def test_environment_override_enables_and_disables_mode(self):
        with patch.dict(
            os.environ,
            {"CALIBRATION_ONLY": "false", "PERCEPTION_HOLD": "true"},
        ):
            self.assertTrue(load_runtime_config().runtime.perception_hold)

        with patch.dict(
            os.environ,
            {"CALIBRATION_ONLY": "false", "PERCEPTION_HOLD": "false"},
        ):
            self.assertFalse(load_runtime_config().runtime.perception_hold)

    def test_calibration_only_and_perception_hold_are_mutually_exclusive(self):
        with patch.dict(
            os.environ,
            {"CALIBRATION_ONLY": "true", "PERCEPTION_HOLD": "true"},
        ):
            with self.assertRaisesRegex(RuntimeError, "mutually exclusive"):
                load_runtime_config()

    def test_invalid_perception_hold_environment_value_fails_closed(self):
        with patch.dict(
            os.environ,
            {"CALIBRATION_ONLY": "false", "PERCEPTION_HOLD": "ture"},
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Invalid boolean environment variable PERCEPTION_HOLD",
            ):
                load_runtime_config()

    def test_perception_adapter_starts_in_perception_hold_mode(self):
        config = _perception_hold_config()
        fake_wrapper = Mock()
        fake_thread = Mock()
        with (
            patch.object(
                perception_adapter_module,
                "PerceptionWrapper",
                return_value=fake_wrapper,
            ) as wrapper_ctor,
            patch.object(
                perception_adapter_module.threading,
                "Thread",
                return_value=fake_thread,
            ) as thread_ctor,
        ):
            adapter = perception_adapter_module.PerceptionAdapter(
                {},
                config=config,
            )

        self.assertTrue(adapter.enabled)
        self.assertTrue(adapter.is_running)
        self.assertIs(adapter.adapter, fake_wrapper)
        wrapper_ctor.assert_called_once_with(config=config)
        self.assertIs(thread_ctor.call_args.kwargs["target"].__self__, adapter)
        self.assertEqual(
            thread_ctor.call_args.kwargs["target"].__func__,
            adapter._loop.__func__,
        )
        self.assertFalse(thread_ctor.call_args.kwargs["daemon"])
        fake_thread.start.assert_called_once_with()

    def test_perception_hold_control_does_not_require_a_camera_frame(self):
        adapter = AutonomyAdapter.__new__(AutonomyAdapter)
        adapter.config = _perception_hold_config()

        snapshot = adapter.build_snapshot(
            frame=None,
            attitude={
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -math.pi,
                "wall_time": 1.0,
            },
            imu={
                "accel_xyz": (0.0, 0.0, -9.81),
                "gyro_xyz": (0.0, 0.0, 0.0),
                "wall_time": 1.0,
            },
        )

        self.assertIsNone(snapshot.image_bgr)
        self.assertEqual(snapshot.image_shape, ())
        self.assertEqual(snapshot.frame_id, -1)

    def test_completed_calibrations_observe_memory_and_hold_without_planning(self):
        api, snapshot, estimate = self._completed_perception_hold_api(
            pos_neu=[1.0, 2.0, 3.0],
            vel_neu=[0.3, 0.0, 0.4],
            yaw_rad=0.25,
        )
        api.adaptive_hover.set_value(0.31, status="test")
        api.tracker.lateral_accel_gain_xy = np.array([1.1, 1.0], dtype=float)

        with patch(
            "autonomy_wrapper.time.monotonic",
            side_effect=[0.0, 1.0, 1.6],
        ):
            first_command = api.update(snapshot)

            self.assertTrue(api.use_perception)
            self.assertTrue(api._perception_hold_active)
            self.assertFalse(api._perception_hold_memory_active)
            self.assertEqual(api._last_gate_memory_frame_key, ("frame", 10))
            api._observe_perception_hold_memory.assert_not_called()
            api._install_gate_centers.assert_not_called()
            np.testing.assert_allclose(
                api._perception_hold_xy_m,
                np.array([1.0, 2.0], dtype=float),
            )
            self.assertAlmostEqual(api._perception_hold_z_m, 3.0)
            self.assertAlmostEqual(api._perception_hold_yaw_rad, 0.25)
            self.assertLess(first_command.pitch_rad, 0.0)
            self.assertLess(first_command.thrust, 0.31)
            self.assertAlmostEqual(first_command.yaw_rad, 0.25)

            snapshot.frame_id = 11
            snapshot.image_wall_time = 2.0
            snapshot.latest_perception = {
                "frame_id": 11,
                "image_wall_time": 2.0,
                "perception_wall_time": 2.1,
                "detections": [],
            }
            estimate.pos_neu = np.array([1.2, 2.0, 2.8], dtype=float)
            estimate.vel_neu = np.zeros(3, dtype=float)
            estimate.yaw_rad = -1.0
            recovery = api.update(snapshot)
            self.assertFalse(api._perception_hold_memory_active)

            snapshot.frame_id = 12
            snapshot.image_wall_time = 2.6
            snapshot.latest_perception = {
                "frame_id": 12,
                "image_wall_time": 2.6,
                "perception_wall_time": 2.7,
                "detections": [],
            }
            api.update(snapshot)
            self.assertTrue(api._perception_hold_memory_active)
            api._observe_perception_hold_memory.assert_not_called()

        snapshot.frame_id = 13
        snapshot.image_wall_time = 3.0
        snapshot.latest_perception = {
            "frame_id": 13,
            "image_wall_time": 3.0,
            "perception_wall_time": 3.1,
            "detections": [],
        }
        api.update(snapshot)

        api._observe_perception_hold_memory.assert_called_once_with(snapshot)
        api._install_gate_centers.assert_not_called()
        self.assertEqual(api.active_track_count, 1)
        self.assertLess(recovery.pitch_rad, 0.0)
        self.assertGreater(recovery.thrust, 0.31)
        self.assertAlmostEqual(recovery.yaw_rad, 0.25)
        np.testing.assert_allclose(
            api._perception_hold_xy_m,
            np.array([1.0, 2.0], dtype=float),
        )
        self.assertAlmostEqual(api._perception_hold_z_m, 3.0)
        api._maybe_apply_active_target_shift.assert_not_called()
        api._advance_gate_if_needed.assert_not_called()
        api._should_plan.assert_not_called()
        api._path_plan.assert_not_called()
        api._path_plan_provisional_next_gate.assert_not_called()
        self.assertEqual(api._trace_perception_hold.call_count, 4)

    def test_source_image_barrier_rejects_in_flight_calibration_frame(self):
        api, snapshot, _estimate = self._completed_perception_hold_api(
            pos_neu=[0.0, 0.0, 1.0],
            vel_neu=[0.0, 0.0, 0.0],
            yaw_rad=0.0,
        )
        api._perception_hold_frame_barrier = 20
        api._perception_hold_image_time_barrier = 5.0
        def add_first_track(_latest_perception, *, snapshot):
            api.gate_memory.add_detection(
                center=np.array([5.0, 0.0, 1.0], dtype=float),
                confidence=1.0,
                timestamp=6.1,
            )

        api._update_gate_memory = Mock(side_effect=add_first_track)

        snapshot.latest_perception = {
            "frame_id": 20,
            "image_wall_time": 5.0,
            "perception_wall_time": 6.0,
            "detections": [],
        }
        PyAIPilotAutonomyAPI._observe_perception_hold_memory(api, snapshot)
        api._update_gate_memory.assert_not_called()

        snapshot.latest_perception = {
            "frame_id": 21,
            "image_wall_time": 5.1,
            "perception_wall_time": 6.1,
            "detections": [],
        }
        PyAIPilotAutonomyAPI._observe_perception_hold_memory(api, snapshot)
        api._update_gate_memory.assert_called_once_with(
            snapshot.latest_perception,
            snapshot=snapshot,
        )
        self.assertEqual(len(api.gate_memory.tracks), 1)
        self.assertEqual(api.gate_memory.tracks[0].id, 0)

    def test_failed_thrust_calibration_preempts_perception_hold(self):
        api, snapshot, _estimate = self._completed_perception_hold_api(
            pos_neu=[1.0, 2.0, 3.0],
            vel_neu=[0.2, 0.0, 0.2],
            yaw_rad=0.0,
        )
        api.thrust_scale_calibration.succeeded = False
        api.lateral_response_calibration.completed = False
        api.lateral_response_calibration.update = Mock(
            side_effect=AssertionError(
                "failed thrust calibration must not unlock lateral calibration"
            )
        )
        api._trace_thrust_scale_failure_hold = Mock()

        command = api.update(snapshot)

        self.assertIsNotNone(command)
        self.assertFalse(api._perception_hold_active)
        self.assertIsNone(api._perception_hold_xy_m)
        self.assertIsNone(api._perception_hold_z_m)
        self.assertIsNone(api._perception_hold_yaw_rad)
        api._observe_perception_hold_memory.assert_not_called()
        api.lateral_response_calibration.update.assert_not_called()
        api._trace_thrust_scale_failure_hold.assert_called_once()
        api._trace_perception_hold.assert_not_called()
        api._should_plan.assert_not_called()

    @staticmethod
    def _completed_perception_hold_api(
        *,
        pos_neu,
        vel_neu,
        yaw_rad,
    ):
        config = _perception_hold_config()
        api = PyAIPilotAutonomyAPI(
            use_perception=True,
            config=config,
        )
        estimate = SimpleNamespace(
            valid=True,
            confidence=1.0,
            pos_neu=np.asarray(pos_neu, dtype=float),
            vel_neu=np.asarray(vel_neu, dtype=float),
            yaw_rad=float(yaw_rad),
        )
        api.state_estimator = _StateEstimator(estimate)
        api.shadow_state_estimator = None
        api.hover_acquisition.completed = True
        api.thrust_scale_calibration.completed = True
        api.thrust_scale_calibration.succeeded = True
        api.lateral_response_calibration.completed = True
        api.lateral_response_calibration.succeeded = True
        api._observe_perception_hold_memory = Mock(return_value=1)
        api._install_gate_centers = Mock(
            side_effect=AssertionError(
                "perception-hold mode must not install navigation targets"
            )
        )
        api._maybe_apply_active_target_shift = Mock(
            side_effect=AssertionError("perception-hold mode must not shift targets")
        )
        api._advance_gate_if_needed = Mock(
            side_effect=AssertionError("perception-hold mode must not advance gates")
        )
        api._should_plan = Mock(
            side_effect=AssertionError("perception-hold mode must not plan")
        )
        api._path_plan = Mock(
            side_effect=AssertionError("perception-hold mode must not plan")
        )
        api._path_plan_provisional_next_gate = Mock(
            side_effect=AssertionError("perception-hold mode must not plan")
        )
        api._trace_perception_hold = Mock()
        snapshot = SimpleNamespace(
            frame_id=10,
            image_wall_time=1.0,
            latest_perception={
                "frame_id": 10,
                "image_wall_time": 1.0,
                "perception_wall_time": 1.0,
                "detections": [],
            }
        )
        return api, snapshot, estimate


class CompetitionFallbackTests(unittest.TestCase):
    def test_attitude_boundary_changes_only_competition_pitch_polarity(self):
        competition_api, _snapshot, _estimate = self._no_plan_api("competition")
        px4_api, _snapshot, _estimate = self._no_plan_api("px4")

        competition = competition_api._attitude_command_boundary(0.1, 0.2)
        px4 = px4_api._attitude_command_boundary(0.1, 0.2)

        self.assertEqual(competition, (-0.1, 0.2))
        self.assertEqual(px4, (-0.1, -0.2))

    def test_active_flight_applies_configured_competition_nose_down_minimum(self):
        api, _snapshot, _estimate = self._no_plan_api("competition")
        api.config = replace(
            api.config,
            controller=replace(
                api.config.controller,
                attitude_slew_limit_enabled=False,
                flight_nose_down_enabled=True,
                flight_min_nose_down_deg=5.0,
            ),
        )

        _roll_rad, pitch_rad = api._attitude_command_boundary(
            0.0,
            0.0,
            apply_flight_nose_down=True,
        )
        self.assertAlmostEqual(math.degrees(pitch_rad), 5.0)

        _roll_rad, pitch_rad = api._attitude_command_boundary(
            0.0,
            math.radians(9.0),
            apply_flight_nose_down=True,
        )
        self.assertAlmostEqual(math.degrees(pitch_rad), 9.0)

    def test_nose_down_minimum_does_not_affect_nonflight_or_px4_commands(self):
        competition_api, _snapshot, _estimate = self._no_plan_api("competition")
        px4_api, _snapshot, _estimate = self._no_plan_api("px4")
        for api in (competition_api, px4_api):
            api.config = replace(
                api.config,
                controller=replace(
                    api.config.controller,
                    attitude_slew_limit_enabled=False,
                    flight_nose_down_enabled=True,
                    flight_min_nose_down_deg=5.0,
                ),
            )

        _roll_rad, competition_pitch = (
            competition_api._attitude_command_boundary(0.0, 0.0)
        )
        _roll_rad, px4_pitch = px4_api._attitude_command_boundary(
            0.0,
            0.0,
            apply_flight_nose_down=True,
        )

        self.assertAlmostEqual(competition_pitch, 0.0)
        self.assertAlmostEqual(px4_pitch, 0.0)

    def test_attitude_slew_limit_caps_roll_and_pitch_rate(self):
        api, _snapshot, _estimate = self._no_plan_api("competition")
        api.config = replace(
            api.config,
            controller=replace(
                api.config.controller,
                attitude_slew_limit_enabled=True,
                max_roll_slew_rate_deg_s=30.0,
                max_pitch_slew_rate_deg_s=20.0,
            ),
        )
        api._remember_attitude_slew_state(0.0, 0.0, now=10.0)

        roll_rad, pitch_rad = api._attitude_command_boundary(
            math.radians(-20.0),
            math.radians(20.0),
            now=10.1,
        )

        self.assertAlmostEqual(math.degrees(roll_rad), 3.0)
        self.assertAlmostEqual(math.degrees(pitch_rad), 2.0)
        self.assertTrue(api._attitude_slew_limited)
        self.assertAlmostEqual(
            math.degrees(api._attitude_slew_target_roll_rad),
            20.0,
        )
        self.assertAlmostEqual(
            math.degrees(api._attitude_slew_target_pitch_rad),
            20.0,
        )

    def test_attitude_slew_limit_can_be_disabled(self):
        api, _snapshot, _estimate = self._no_plan_api("competition")
        api.config = replace(
            api.config,
            controller=replace(
                api.config.controller,
                attitude_slew_limit_enabled=False,
            ),
        )
        api._remember_attitude_slew_state(0.0, 0.0, now=10.0)

        roll_rad, pitch_rad = api._attitude_command_boundary(
            math.radians(-20.0),
            math.radians(20.0),
            now=10.1,
        )

        self.assertAlmostEqual(math.degrees(roll_rad), 20.0)
        self.assertAlmostEqual(math.degrees(pitch_rad), 20.0)
        self.assertFalse(api._attitude_slew_limited)

    def test_tilt_compensation_uses_final_slew_limited_pitch(self):
        api, _snapshot, _estimate = self._no_plan_api("competition")
        api.config = replace(
            api.config,
            controller=replace(
                api.config.controller,
                attitude_slew_limit_enabled=True,
                max_roll_slew_rate_deg_s=40.0,
                max_pitch_slew_rate_deg_s=40.0,
            ),
        )
        api.tracker.tilt_thrust_compensation_enabled = True
        api.tracker.thrust_min = 0.0
        api.tracker.thrust_max = 1.0
        api._remember_attitude_slew_state(0.0, 0.0, now=10.0)

        roll_rad, pitch_rad = api._attitude_command_boundary(
            0.0,
            math.radians(30.0),
            now=10.05,
        )
        tracker_debug = {
            "thrust_uncompensated": 0.251,
            "tilt_vertical_fraction": math.cos(math.radians(30.0)),
            "tilt_thrust_compensation_factor": (
                1.0 / math.cos(math.radians(30.0))
            ),
        }

        thrust = api._finalize_tracker_thrust(
            tracker_thrust=0.251 / math.cos(math.radians(30.0)),
            tracker_debug=tracker_debug,
            roll_rad=roll_rad,
            pitch_rad=pitch_rad,
        )

        self.assertAlmostEqual(math.degrees(pitch_rad), 2.0)
        self.assertAlmostEqual(
            thrust,
            0.251 / math.cos(math.radians(2.0)),
        )
        self.assertLess(thrust, 0.252)
        self.assertEqual(
            tracker_debug["tilt_compensation_attitude_source"],
            "final_command",
        )

    def test_attitude_slew_limit_smooths_transition_to_hold(self):
        api, _snapshot, _estimate = self._no_plan_api("competition")
        api.config = replace(
            api.config,
            controller=replace(
                api.config.controller,
                attitude_slew_limit_enabled=True,
                max_roll_slew_rate_deg_s=30.0,
                max_pitch_slew_rate_deg_s=30.0,
            ),
        )
        api._remember_attitude_slew_state(
            math.radians(-8.0),
            math.radians(9.0),
            now=10.0,
        )

        roll_rad, pitch_rad = api._apply_attitude_slew_limit(
            0.0,
            0.0,
            now=10.1,
        )

        self.assertAlmostEqual(math.degrees(roll_rad), -5.0)
        self.assertAlmostEqual(math.degrees(pitch_rad), 6.0)
        self.assertTrue(api._attitude_slew_limited)

    def test_controller_no_command_uses_attitude_fallback_in_competition(self):
        connection = _Connection()
        data = {
            "latest_autonomy_command": None,
            "latest_autonomy_command_status": "missing_inputs",
            "attitude": {"yaw": math.pi / 2.0},
        }
        controller = Controller(
            connection,
            data,
            system_boot_ms=0,
            config=_normal_config("competition"),
        )
        hover_hold = _ForbiddenHoverHold()
        controller.hover_hold = hover_hold

        with patch("controller.time.sleep", return_value=None):
            controller.update()

        self.assertTrue(hover_hold.reset_called)
        self.assertEqual(connection.mav.position_targets, [])
        self.assertEqual(len(connection.mav.attitude_targets), 1)
        attitude_args = connection.mav.attitude_targets[0]
        self.assertAlmostEqual(attitude_args[8], controller.fallback_thrust)
        self.assertAlmostEqual(get_current_yaw_deg(data), 90.0)
        self.assertAlmostEqual(
            _quaternion_yaw_deg(attitude_args[4]),
            -90.0,
        )

    def test_controller_inverts_only_competition_outgoing_yaw(self):
        for runner_mode, expected_wire_yaw_deg in (
            ("competition", -35.0),
            ("px4", 35.0),
        ):
            with self.subTest(runner_mode=runner_mode):
                connection = _Connection()
                data = {
                    "latest_autonomy_command": SimpleNamespace(
                        roll_deg=0.0,
                        pitch_deg=0.0,
                        yaw_deg=35.0,
                        thrust=0.3,
                    ),
                    "latest_autonomy_command_status": "ok",
                }
                controller = Controller(
                    connection,
                    data,
                    system_boot_ms=0,
                    config=_normal_config(runner_mode),
                )

                with patch("controller.time.sleep", return_value=None):
                    controller.update()

                attitude_args = connection.mav.attitude_targets[0]
                self.assertAlmostEqual(
                    _quaternion_yaw_deg(attitude_args[4]),
                    expected_wire_yaw_deg,
                )

    def test_controller_startup_observation_transmits_nothing(self):
        connection = _Connection()
        data = {
            "latest_autonomy_command": None,
            "latest_autonomy_command_status": "startup_observation_observing",
            "attitude": {"yaw": math.pi / 2.0},
        }
        controller = Controller(
            connection,
            data,
            system_boot_ms=0,
            config=_normal_config("competition"),
        )
        hover_hold = _ForbiddenHoverHold()
        controller.hover_hold = hover_hold

        with patch("controller.time.sleep", return_value=None):
            controller.update()

        self.assertTrue(hover_hold.reset_called)
        self.assertEqual(connection.mav.position_targets, [])
        self.assertEqual(connection.mav.attitude_targets, [])

    def test_controller_no_command_keeps_position_hold_in_px4(self):
        connection = _Connection()
        data = {
            "latest_autonomy_command": None,
            "latest_autonomy_command_status": "missing_inputs",
            "attitude": {"yaw": 0.0},
        }
        controller = Controller(
            connection,
            data,
            system_boot_ms=0,
            config=_normal_config("px4"),
        )
        hover_hold = _RecordingHoverHold()
        controller.hover_hold = hover_hold

        with patch("controller.time.sleep", return_value=None):
            controller.update()

        self.assertEqual(hover_hold.update_calls, 1)
        self.assertFalse(hover_hold.reset_called)
        self.assertEqual(connection.mav.attitude_targets, [])

    def test_competition_no_plan_holds_learned_thrust_with_vertical_damping(self):
        api, snapshot, estimate = self._no_plan_api("competition")
        api.adaptive_hover.set_value(0.278, status="test")
        api._trace_competition_no_plan_hold = Mock()

        command = api.update(snapshot)

        self.assertIsNotNone(command)
        self.assertAlmostEqual(command.roll_rad, 0.0)
        self.assertAlmostEqual(command.pitch_rad, 0.0)
        self.assertAlmostEqual(command.yaw_rad, -math.pi)
        self.assertAlmostEqual(command.thrust, 0.278)
        self.assertAlmostEqual(api._competition_no_plan_hold_z_m, 3.0)

        estimate.pos_neu = np.array([1.0, 2.0, 2.8], dtype=float)
        estimate.vel_neu = np.array([0.0, 0.0, -0.5], dtype=float)
        recovery_command = api.update(snapshot)

        self.assertGreater(recovery_command.thrust, 0.278)
        self.assertAlmostEqual(recovery_command.thrust, 0.338)
        self.assertAlmostEqual(api._competition_no_plan_hold_z_m, 3.0)

    def test_no_target_search_advances_then_descends_after_settle(self):
        api, snapshot, estimate = self._no_plan_api("competition")
        snapshot.armed = True
        snapshot.race_status = {
            "sim_boot_time_ms": 2_000,
            "race_start_boot_time_ms": 1_000,
        }
        api._trace_no_target_search = Mock()

        initial = api._competition_no_target_search_command(
            snapshot=snapshot,
            pos=estimate.pos_neu,
            vel=estimate.vel_neu,
            yaw_rad=estimate.yaw_rad,
            now=10.0,
            wall_time=100.0,
        )

        self.assertEqual(api._no_target_search_state, "loss_grace")
        self.assertAlmostEqual(api._competition_no_plan_hold_z_m, 3.0)
        self.assertAlmostEqual(initial.pitch_rad, 0.0)

        estimate.vel_neu = np.array([0.4, 0.0, 0.0], dtype=float)
        api._competition_no_target_search_command(
            snapshot=snapshot,
            pos=estimate.pos_neu,
            vel=estimate.vel_neu,
            yaw_rad=estimate.yaw_rad,
            now=11.1,
            wall_time=101.1,
        )
        self.assertEqual(api._no_target_search_state, "settling")
        self.assertAlmostEqual(api._competition_no_plan_hold_z_m, 3.0)

        estimate.vel_neu = np.zeros(3, dtype=float)
        api._competition_no_target_search_command(
            snapshot=snapshot,
            pos=estimate.pos_neu,
            vel=estimate.vel_neu,
            yaw_rad=estimate.yaw_rad,
            now=11.2,
            wall_time=101.2,
        )
        self.assertEqual(api._no_target_search_state, "advancing")
        self.assertAlmostEqual(api._competition_no_plan_hold_z_m, 3.0)
        np.testing.assert_allclose(
            api._competition_no_plan_hold_xy_m,
            np.array([-1.0, 2.0], dtype=float),
            atol=1e-8,
        )

        estimate.pos_neu[:2] = api._competition_no_plan_hold_xy_m
        estimate.vel_neu = np.array([0.4, 0.0, 0.0], dtype=float)
        api._competition_no_target_search_command(
            snapshot=snapshot,
            pos=estimate.pos_neu,
            vel=estimate.vel_neu,
            yaw_rad=estimate.yaw_rad,
            now=11.3,
            wall_time=101.3,
        )
        self.assertEqual(api._no_target_search_state, "forward_settling")
        self.assertAlmostEqual(api._competition_no_plan_hold_z_m, 3.0)

        estimate.vel_neu = np.zeros(3, dtype=float)
        api._competition_no_target_search_command(
            snapshot=snapshot,
            pos=estimate.pos_neu,
            vel=estimate.vel_neu,
            yaw_rad=estimate.yaw_rad,
            now=11.4,
            wall_time=101.4,
        )
        self.assertEqual(api._no_target_search_state, "descending")
        self.assertAlmostEqual(
            api._competition_no_plan_hold_z_m,
            3.0 - api.no_target_search_descent_rate_m_s * 0.1,
        )
        np.testing.assert_allclose(
            api._competition_no_plan_hold_xy_m,
            np.array([-1.0, 2.0], dtype=float),
        )

    def test_no_target_search_freezes_descent_and_centers_credible_candidate(self):
        api, snapshot, estimate = self._no_plan_api("competition")
        snapshot.armed = True
        snapshot.race_status = {
            "sim_boot_time_ms": 2_000,
            "race_start_boot_time_ms": 1_000,
        }
        api._trace_no_target_search = Mock()
        observation = SimpleNamespace(
            quality_ok=True,
            is_outlier=False,
            keypoint_conf_min=0.95,
            reprojection_error=0.25,
            image_area_px2=120.0,
        )
        candidate = SimpleNamespace(
            id=42,
            hits=1,
            committed=False,
            is_stable=False,
            last_seen_time=200.0,
            last_image_center_px=np.array([500.0, 180.0], dtype=float),
            center=np.array([-25.94, -0.54, 1.14], dtype=float),
            filtered_center_world=None,
            obs_history=[observation],
        )
        api.gate_memory.tracks = [candidate]

        api._competition_no_target_search_command(
            snapshot=snapshot,
            pos=estimate.pos_neu,
            vel=estimate.vel_neu,
            yaw_rad=estimate.yaw_rad,
            now=20.0,
            wall_time=200.0,
        )
        command = api._competition_no_target_search_command(
            snapshot=snapshot,
            pos=estimate.pos_neu,
            vel=estimate.vel_neu,
            yaw_rad=estimate.yaw_rad,
            now=20.1,
            wall_time=200.1,
        )

        self.assertEqual(api._no_target_search_state, "acquiring")
        self.assertEqual(api._no_target_search_candidate_track_id, 42)
        self.assertAlmostEqual(api._competition_no_plan_hold_z_m, 3.0)
        self.assertGreater(command.yaw_rad, -math.pi)

    def test_no_target_search_descends_until_candidate_is_vertically_centered(
        self,
    ):
        api, snapshot, estimate = self._no_plan_api("competition")
        snapshot.armed = True
        snapshot.race_status = {
            "sim_boot_time_ms": 2_000,
            "race_start_boot_time_ms": 1_000,
        }
        api._trace_no_target_search = Mock()
        observation = SimpleNamespace(
            quality_ok=True,
            is_outlier=False,
            keypoint_conf_min=0.95,
            reprojection_error=0.25,
            image_area_px2=120.0,
        )
        candidate = SimpleNamespace(
            id=42,
            hits=1,
            committed=False,
            is_stable=False,
            last_seen_time=200.0,
            last_image_center_px=np.array([320.0, 330.0], dtype=float),
            center=np.array([-25.94, -0.54, 1.14], dtype=float),
            filtered_center_world=None,
            obs_history=[observation],
        )
        api.gate_memory.tracks = [candidate]

        api._competition_no_target_search_command(
            snapshot=snapshot,
            pos=estimate.pos_neu,
            vel=estimate.vel_neu,
            yaw_rad=estimate.yaw_rad,
            now=20.0,
            wall_time=200.0,
        )
        api._competition_no_target_search_command(
            snapshot=snapshot,
            pos=estimate.pos_neu,
            vel=estimate.vel_neu,
            yaw_rad=estimate.yaw_rad,
            now=20.1,
            wall_time=200.1,
        )

        self.assertEqual(api._no_target_search_state, "centering_vertical")
        self.assertAlmostEqual(
            api._competition_no_plan_hold_z_m,
            3.0
            - api.no_target_search_center_vertical_max_descent_rate_m_s * 0.1,
        )

        estimate.pos_neu[2] = api._competition_no_plan_hold_z_m
        candidate.last_seen_time = 200.2
        candidate.last_image_center_px[1] = 180.0
        api._competition_no_target_search_command(
            snapshot=snapshot,
            pos=estimate.pos_neu,
            vel=estimate.vel_neu,
            yaw_rad=estimate.yaw_rad,
            now=20.2,
            wall_time=200.2,
        )

        self.assertEqual(api._no_target_search_state, "acquiring")
        self.assertAlmostEqual(
            api._competition_no_plan_hold_z_m,
            estimate.pos_neu[2],
        )

    def test_no_target_search_does_not_replace_an_admitted_target(self):
        api, _snapshot, _estimate = self._no_plan_api("competition")
        api.current_gate_pos = np.array([-20.0, 0.0, 1.0], dtype=float)

        self.assertFalse(api._no_target_search_should_run())

    def test_px4_no_plan_preserves_existing_none_fallback(self):
        api, snapshot, _estimate = self._no_plan_api("px4")

        command = api.update(snapshot)

        self.assertIsNone(command)

    def test_lateral_completion_hands_directly_to_competition_attitude_hold(self):
        api, snapshot, _estimate = self._no_plan_api("competition")
        api.adaptive_hover.set_value(0.278, status="test")
        api.lateral_response_calibration.completed = False
        api.lateral_response_calibration.update = Mock(
            side_effect=self._complete_lateral_calibration
        )
        api._trace_lateral_response_calibration = Mock()
        api._trace_competition_no_plan_hold = Mock()

        command = api.update(snapshot)

        self.assertIsNotNone(command)
        self.assertAlmostEqual(command.thrust, 0.278)
        api.lateral_response_calibration.update.assert_called_once()

    @staticmethod
    def _complete_lateral_calibration(**_kwargs):
        return SimpleNamespace(
            lateral_accel_gain_xy=None,
            succeeded=True,
            command=None,
            debug=SimpleNamespace(active=False, completed=True, succeeded=True),
        )

    @staticmethod
    def _no_plan_api(runner_mode):
        config = _normal_config(runner_mode)
        api = PyAIPilotAutonomyAPI(
            use_perception=True,
            config=config,
        )
        estimate = SimpleNamespace(
            valid=True,
            confidence=1.0,
            pos_neu=np.array([1.0, 2.0, 3.0], dtype=float),
            vel_neu=np.zeros(3, dtype=float),
            yaw_rad=-math.pi,
        )
        api.state_estimator = _StateEstimator(estimate)
        api.shadow_state_estimator = None
        api.hover_acquisition.completed = True
        api.thrust_scale_calibration.completed = True
        api.thrust_scale_calibration.succeeded = True
        api.lateral_response_calibration.completed = True
        api.lateral_response_calibration.succeeded = True
        api.active_waypoints = None
        api.planner.total_time = 0.0
        api._gates_from_snapshot = Mock(return_value=[])
        api._install_gate_centers = Mock()
        api._maybe_apply_active_target_shift = Mock(return_value=False)
        api._advance_gate_if_needed = Mock(return_value=False)
        api._should_plan = Mock(return_value=False)
        snapshot = SimpleNamespace(latest_perception=None)
        return api, snapshot, estimate


class FixedCalibrationConfigTests(unittest.TestCase):
    def test_disabled_calibrations_seed_tracker_with_fixed_values(self):
        config = load_runtime_config()

        self.assertFalse(config.hover_acquisition.enabled)
        self.assertFalse(config.thrust_scale_calibration.enabled)
        self.assertFalse(config.lateral_response_calibration.enabled)
        self.assertAlmostEqual(config.controller.thrust_hover, 0.264)
        self.assertAlmostEqual(config.controller.thrust_from_acc_gain, 0.0154)
        np.testing.assert_allclose(
            config.controller.lateral_accel_gain_xy,
            [1.37, 1.39],
        )

        api = PyAIPilotAutonomyAPI(use_perception=True, config=config)

        self.assertAlmostEqual(api.tracker.thrust_hover, 0.264)
        self.assertAlmostEqual(api.tracker.thrust_from_acc_gain, 0.0154)
        np.testing.assert_allclose(
            api.tracker.lateral_accel_gain_xy,
            [1.37, 1.39],
        )


class StartupObservationTests(unittest.TestCase):
    def test_runtime_enables_ten_second_observation(self):
        self.assertAlmostEqual(
            load_runtime_config().runtime.startup_observation_duration_s,
            10.0,
        )

    def test_gate_memory_updates_before_flight_behaviors(self):
        config = _normal_config("competition")
        config = replace(
            config,
            runtime=replace(
                config.runtime,
                startup_observation_duration_s=10.0,
            ),
        )
        api = PyAIPilotAutonomyAPI(use_perception=True, config=config)
        estimate = SimpleNamespace(
            valid=True,
            confidence=1.0,
            pos_neu=np.zeros(3, dtype=float),
            vel_neu=np.zeros(3, dtype=float),
            yaw_rad=0.0,
        )
        api.state_estimator = _StateEstimator(estimate)
        api.shadow_state_estimator = None

        def ingest_detection(_snapshot):
            api.gate_memory.add_detection(
                center=np.array([10.0, 1.0, 2.0], dtype=float),
                confidence=1.0,
                timestamp=1.0,
            )
            return []

        api._gates_from_snapshot = Mock(side_effect=ingest_detection)
        api._install_gate_centers = Mock()
        api.hover_acquisition.update = Mock(
            side_effect=AssertionError(
                "hover acquisition must remain delayed during observation"
            )
        )
        snapshot = SimpleNamespace(
            latest_perception={
                "frame_id": 1,
                "image_wall_time": 1.0,
                "perception_wall_time": 1.0,
                "detections": [],
            }
        )

        command = api.update(snapshot)

        self.assertIsNone(command)
        self.assertTrue(api.startup_observation_active)
        self.assertEqual(api.startup_observation_status, "observing")
        self.assertEqual(len(api.gate_memory.tracks), 1)
        api.hover_acquisition.update.assert_not_called()

    def test_observation_releases_flight_after_duration(self):
        config = _normal_config("competition")
        config = replace(
            config,
            runtime=replace(
                config.runtime,
                startup_observation_duration_s=10.0,
            ),
        )
        api = PyAIPilotAutonomyAPI(use_perception=True, config=config)
        snapshot = SimpleNamespace(latest_perception={"frame_id": 1})

        self.assertTrue(
            api._startup_observation_blocks_flight(snapshot, now=100.0)
        )
        self.assertTrue(
            api._startup_observation_blocks_flight(snapshot, now=109.99)
        )
        self.assertFalse(
            api._startup_observation_blocks_flight(snapshot, now=110.0)
        )
        self.assertFalse(api.startup_observation_active)
        self.assertEqual(api.startup_observation_status, "complete")


if __name__ == "__main__":
    unittest.main()
