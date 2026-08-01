import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from lateral_response_calibration import LateralResponseCalibration
from runtime_config import load_runtime_config


def _config(**overrides):
    config = load_runtime_config()
    values = {
        "enabled": True,
        "estimator_mode_only": False,
        "require_thrust_scale_calibration": True,
        "require_armed": True,
        "initial_delay_s": 0.0,
        "min_duration_s": 0.6,
        "max_duration_s": 2.0,
        "phase_duration_s": 0.15,
        "settle_duration_s": 0.02,
        "probe_accel_m_s2": 0.6,
        "min_probe_accel_m_s2": 0.2,
        "max_probe_accel_m_s2": 1.0,
        "max_tilt_deg": 5.0,
        "min_samples_per_axis": 2,
        "accel_filter_alpha": 1.0,
        "accel_deadband_m_s2": 0.0,
        "min_abs_accel_m_s2": 0.0,
        "max_cross_axis_ratio": 0.0,
        "max_sign_ratio_disagreement": 0.35,
        "z_hold_enabled": True,
        "z_hold_kp": 0.10,
        "z_hold_kv": 0.08,
        "z_hold_max_correction": 0.08,
        "max_gain": 2.0,
        "result_alpha": 1.0,
    }
    values.update(overrides)
    return replace(
        config,
        lateral_response_calibration=replace(
            config.lateral_response_calibration,
            **values,
        ),
    )


def _snapshot(
    *,
    accel_neu_xy,
    gravity=9.81,
    roll_rad=0.0,
    pitch_rad=0.0,
    yaw_rad=0.0,
    position_wall_time=None,
):
    return SimpleNamespace(
        roll_rad=float(roll_rad),
        pitch_rad=float(pitch_rad),
        yaw_rad=float(yaw_rad),
        accel_xyz=np.array(
            [accel_neu_xy[0], accel_neu_xy[1], -gravity],
            dtype=float,
        ),
        position_wall_time=position_wall_time,
        armed=True,
    )


def _estimate(pos_neu=None, vel_neu=None):
    return SimpleNamespace(
        valid=True,
        pos_neu=np.array(
            [0.0, 0.0, 1.0] if pos_neu is None else pos_neu,
            dtype=float,
        ),
        vel_neu=np.array(
            [0.0, 0.0, 0.0] if vel_neu is None else vel_neu,
            dtype=float,
        ),
        yaw_rad=0.0,
    )


def test_learns_lateral_accel_gain_xy_from_small_probes():
    calibrator = LateralResponseCalibration(_config())
    response_xy = np.array([0.75, 1.25], dtype=float)
    last_command_accel_xy = np.zeros(2, dtype=float)
    pos_neu = np.array([0.0, 0.0, 1.0], dtype=float)
    vel_neu = np.zeros(3, dtype=float)
    result = None

    for idx in range(60):
        measured_accel_xy = response_xy * last_command_accel_xy
        if idx > 0:
            vel_neu[:2] += measured_accel_xy * 0.05
            pos_neu += vel_neu * 0.05
        result = calibrator.update(
            snapshot=_snapshot(
                accel_neu_xy=measured_accel_xy,
                position_wall_time=idx * 0.05,
            ),
            estimate=_estimate(pos_neu=pos_neu, vel_neu=vel_neu),
            hover_thrust=0.5,
            thrust_scale_calibration_completed=True,
            current_lateral_accel_gain_xy=np.ones(2, dtype=float),
            now=idx * 0.05,
        )
        last_command_accel_xy = np.asarray(
            result.debug.command_accel_xy_m_s2,
            dtype=float,
        )
        if result.debug.completed:
            break

    assert result is not None
    assert result.debug.status == "calibrated"
    assert result.succeeded
    assert result.debug.succeeded
    assert calibrator.succeeded
    assert result.debug.samples_xy[0] >= 2
    assert result.debug.samples_xy[1] >= 2
    np.testing.assert_allclose(
        result.lateral_accel_gain_xy,
        np.array([1.0 / response_xy[0], 1.0 / response_xy[1]]),
        atol=1e-9,
    )


def test_rejects_lateral_gain_when_probe_signs_disagree():
    calibrator = LateralResponseCalibration(
        _config(
            max_sign_ratio_disagreement=0.20,
            min_duration_s=0.6,
            max_duration_s=1.1,
        )
    )
    last_command_accel_xy = np.zeros(2, dtype=float)
    pos_neu = np.array([0.0, 0.0, 1.0], dtype=float)
    vel_neu = np.zeros(3, dtype=float)
    result = None

    for idx in range(80):
        measured_accel_xy = np.zeros(2, dtype=float)
        axis = int(np.argmax(np.abs(last_command_accel_xy)))
        command = float(last_command_accel_xy[axis])
        if abs(command) > 0.0:
            if axis == 0:
                ratio = 0.50 if command > 0.0 else 1.00
            else:
                ratio = 1.00
            measured_accel_xy[axis] = ratio * command
        if idx > 0:
            vel_neu[:2] += measured_accel_xy * 0.05
            pos_neu += vel_neu * 0.05

        result = calibrator.update(
            snapshot=_snapshot(
                accel_neu_xy=measured_accel_xy,
                position_wall_time=idx * 0.05,
            ),
            estimate=_estimate(pos_neu=pos_neu, vel_neu=vel_neu),
            hover_thrust=0.5,
            thrust_scale_calibration_completed=True,
            current_lateral_accel_gain_xy=np.ones(2, dtype=float),
            now=idx * 0.05,
        )
        last_command_accel_xy = np.asarray(
            result.debug.command_accel_xy_m_s2,
            dtype=float,
        )
        if result.debug.completed:
            break

    assert result is not None
    assert result.debug.status == "timeout_fallback"
    assert result.debug.signed_samples_xy[0] >= 2
    assert result.debug.signed_samples_xy[1] >= 2
    assert result.debug.signed_samples_xy[2] >= 2
    assert result.debug.signed_samples_xy[3] >= 2
    assert result.lateral_accel_gain_xy is None
    assert not result.succeeded
    assert not result.debug.succeeded
    assert not calibrator.succeeded


def test_rejects_lateral_gain_outside_configured_bounds():
    calibrator = LateralResponseCalibration(
        _config(
            min_gain=0.5,
            max_gain=1.25,
            min_duration_s=0.6,
            max_duration_s=1.1,
        )
    )
    response_xy = np.array([0.5, 1.0], dtype=float)
    last_command_accel_xy = np.zeros(2, dtype=float)
    pos_neu = np.array([0.0, 0.0, 1.0], dtype=float)
    vel_neu = np.zeros(3, dtype=float)
    result = None

    for idx in range(80):
        measured_accel_xy = response_xy * last_command_accel_xy
        if idx > 0:
            vel_neu[:2] += measured_accel_xy * 0.05
            pos_neu += vel_neu * 0.05
        result = calibrator.update(
            snapshot=_snapshot(
                accel_neu_xy=measured_accel_xy,
                position_wall_time=idx * 0.05,
            ),
            estimate=_estimate(pos_neu=pos_neu, vel_neu=vel_neu),
            hover_thrust=0.5,
            thrust_scale_calibration_completed=True,
            current_lateral_accel_gain_xy=np.ones(2, dtype=float),
            now=idx * 0.05,
        )
        last_command_accel_xy = np.asarray(
            result.debug.command_accel_xy_m_s2,
            dtype=float,
        )
        if result.debug.completed:
            break

    assert result is not None
    assert result.debug.status == "timeout_fallback"
    assert result.debug.signed_samples_xy[0] >= 2
    assert result.debug.signed_samples_xy[1] >= 2
    assert result.debug.signed_samples_xy[2] >= 2
    assert result.debug.signed_samples_xy[3] >= 2
    assert result.lateral_accel_gain_xy is None
    assert not result.succeeded
    assert not calibrator.succeeded


def test_yaw_pi_world_x_accel_uses_observed_competition_pitch_polarity():
    calibrator = LateralResponseCalibration(_config())
    magnitude = 0.4

    _, negative_x_pitch, _ = calibrator._command_from_accel(
        command_accel_xy=np.array([-magnitude, 0.0], dtype=float),
        yaw_rad=math.pi,
        hover_thrust=0.5,
    )
    _, positive_x_pitch, _ = calibrator._command_from_accel(
        command_accel_xy=np.array([magnitude, 0.0], dtype=float),
        yaw_rad=math.pi,
        hover_thrust=0.5,
    )

    expected = math.atan2(magnitude, 9.81)
    np.testing.assert_allclose(negative_x_pitch, expected, atol=1e-12)
    np.testing.assert_allclose(positive_x_pitch, -expected, atol=1e-12)


def test_px4_mode_retains_legacy_pitch_bridge():
    config = _config()
    config = replace(
        config,
        runtime=replace(config.runtime, runner_mode="px4"),
    )
    calibrator = LateralResponseCalibration(config)

    _, pitch, _ = calibrator._command_from_accel(
        command_accel_xy=np.array([-0.4, 0.0], dtype=float),
        yaw_rad=math.pi,
        hover_thrust=0.5,
    )

    np.testing.assert_allclose(pitch, -math.atan2(0.4, 9.81), atol=1e-12)


def test_rejects_persistent_command_to_kinematic_polarity_mismatch():
    calibrator = LateralResponseCalibration(
        _config(min_samples_per_axis=2, min_duration_s=0.6, max_duration_s=2.0)
    )
    last_command_accel_xy = np.zeros(2, dtype=float)
    pos_neu = np.array([0.0, 0.0, 1.0], dtype=float)
    vel_neu = np.zeros(3, dtype=float)
    result = None

    for idx in range(20):
        imu_accel_xy = 0.8 * last_command_accel_xy
        kinematic_accel_xy = imu_accel_xy.copy()
        kinematic_accel_xy[0] *= -1.0
        if idx > 0:
            vel_neu[:2] += kinematic_accel_xy * 0.05
            pos_neu += vel_neu * 0.05
        result = calibrator.update(
            snapshot=_snapshot(
                accel_neu_xy=imu_accel_xy,
                position_wall_time=idx * 0.05,
            ),
            estimate=_estimate(pos_neu=pos_neu, vel_neu=vel_neu),
            hover_thrust=0.5,
            thrust_scale_calibration_completed=True,
            current_lateral_accel_gain_xy=np.ones(2, dtype=float),
            now=idx * 0.05,
        )
        last_command_accel_xy = np.asarray(
            result.debug.command_accel_xy_m_s2,
            dtype=float,
        )
        if result.debug.completed:
            break

    assert result is not None
    assert result.debug.status == "polarity_mismatch_fallback"
    assert result.debug.polarity_mismatch_detected
    assert result.debug.kinematic_mismatch_streak_xy[0] >= 2
    assert result.lateral_accel_gain_xy is None
    assert not result.succeeded
    assert not calibrator.succeeded


def test_calibration_local_imu_rotation_corrects_only_competition_pitch_sign():
    calibrator = LateralResponseCalibration(_config())
    snapshot = _snapshot(
        accel_neu_xy=np.zeros(2, dtype=float),
        pitch_rad=math.radians(2.0),
        yaw_rad=math.pi,
        position_wall_time=0.0,
    )

    imu_accel_xy, _ = calibrator._accel_sources_xy_neu(
        snapshot,
        np.zeros(2, dtype=float),
        0.0,
    )

    assert imu_accel_xy[0] < 0.0
    np.testing.assert_allclose(imu_accel_xy[1], 0.0, atol=1e-12)


def test_velocity_accel_ignores_duplicate_position_samples():
    calibrator = LateralResponseCalibration(_config())

    first = calibrator._velocity_accel_xy(
        snapshot=_snapshot(
            accel_neu_xy=np.zeros(2),
            position_wall_time=10.0,
        ),
        vel_xy=np.array([0.0, 0.0]),
        now=20.0,
    )
    duplicate = calibrator._velocity_accel_xy(
        snapshot=_snapshot(
            accel_neu_xy=np.zeros(2),
            position_wall_time=10.0,
        ),
        vel_xy=np.array([99.0, 0.0]),
        now=20.1,
    )
    fresh = calibrator._velocity_accel_xy(
        snapshot=_snapshot(
            accel_neu_xy=np.zeros(2),
            position_wall_time=10.1,
        ),
        vel_xy=np.array([0.1, 0.0]),
        now=20.2,
    )

    assert np.all(np.isnan(first))
    assert np.all(np.isnan(duplicate))
    np.testing.assert_allclose(fresh, np.array([1.0, 0.0]), atol=1e-12)


def test_velocity_accel_resets_on_regressed_position_timestamp():
    calibrator = LateralResponseCalibration(_config())

    calibrator._velocity_accel_xy(
        snapshot=_snapshot(
            accel_neu_xy=np.zeros(2),
            position_wall_time=10.0,
        ),
        vel_xy=np.array([0.0, 0.0]),
        now=20.0,
    )
    regressed = calibrator._velocity_accel_xy(
        snapshot=_snapshot(
            accel_neu_xy=np.zeros(2),
            position_wall_time=9.0,
        ),
        vel_xy=np.array([5.0, 0.0]),
        now=20.1,
    )
    recovered = calibrator._velocity_accel_xy(
        snapshot=_snapshot(
            accel_neu_xy=np.zeros(2),
            position_wall_time=9.1,
        ),
        vel_xy=np.array([5.2, 0.0]),
        now=20.2,
    )

    assert np.all(np.isnan(regressed))
    np.testing.assert_allclose(recovered, np.array([2.0, 0.0]), atol=1e-12)


def test_disabled_and_skipped_stages_complete_without_claiming_success():
    for config, expected_status in (
        (_config(enabled=False), "disabled"),
        (_config(estimator_mode_only=True), "skipped_state_mode"),
    ):
        calibrator = LateralResponseCalibration(config)
        result = calibrator.update(
            snapshot=_snapshot(accel_neu_xy=np.zeros(2)),
            estimate=_estimate(),
            hover_thrust=0.5,
            thrust_scale_calibration_completed=True,
            current_lateral_accel_gain_xy=np.ones(2, dtype=float),
            now=0.0,
        )

        assert result.debug.status == expected_status
        assert result.debug.completed
        assert result.lateral_accel_gain_xy is None
        assert not result.succeeded
        assert not result.debug.succeeded
        assert calibrator.completed
        assert not calibrator.succeeded


def test_z_hold_adds_bounded_thrust_when_calibration_drops():
    calibrator = LateralResponseCalibration(
        _config(
            initial_delay_s=0.5,
            z_hold_kp=0.10,
            z_hold_kv=0.08,
            z_hold_max_correction=0.08,
        )
    )

    first = calibrator.update(
        snapshot=_snapshot(accel_neu_xy=np.zeros(2)),
        estimate=_estimate(pos_neu=[0.0, 0.0, 1.0]),
        hover_thrust=0.5,
        thrust_scale_calibration_completed=True,
        current_lateral_accel_gain_xy=np.ones(2, dtype=float),
        now=0.0,
    )
    assert first.command is not None
    assert first.command.thrust == 0.5

    dropped = calibrator.update(
        snapshot=_snapshot(accel_neu_xy=np.zeros(2)),
        estimate=_estimate(
            pos_neu=[0.0, 0.0, 0.7],
            vel_neu=[0.0, 0.0, -0.2],
        ),
        hover_thrust=0.5,
        thrust_scale_calibration_completed=True,
        current_lateral_accel_gain_xy=np.ones(2, dtype=float),
        now=0.05,
    )

    assert dropped.command is not None
    np.testing.assert_allclose(dropped.debug.z_hold_error_m, 0.3, atol=1e-9)
    np.testing.assert_allclose(dropped.debug.z_hold_vz_error_m_s, 0.2, atol=1e-9)
    np.testing.assert_allclose(
        dropped.debug.z_hold_thrust_correction,
        0.046,
        atol=1e-9,
    )
    np.testing.assert_allclose(dropped.command.thrust, 0.546, atol=1e-9)
