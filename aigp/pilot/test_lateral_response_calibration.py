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


def _snapshot(*, accel_neu_xy, gravity=9.81):
    return SimpleNamespace(
        roll_rad=0.0,
        pitch_rad=0.0,
        yaw_rad=0.0,
        accel_xyz=np.array(
            [accel_neu_xy[0], accel_neu_xy[1], -gravity],
            dtype=float,
        ),
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
    result = None

    for idx in range(60):
        measured_accel_xy = response_xy * last_command_accel_xy
        result = calibrator.update(
            snapshot=_snapshot(accel_neu_xy=measured_accel_xy),
            estimate=_estimate(),
            hover_thrust=0.5,
            thrust_scale_calibration_completed=True,
            current_lateral_accel_gain_xy=np.ones(2, dtype=float),
            now=idx * 0.05,
        )
        last_command_accel_xy = np.asarray(
            result.debug.command_accel_xy_m_s2,
            dtype=float,
        )
        if result.lateral_accel_gain_xy is not None:
            break

    assert result is not None
    assert result.debug.status == "calibrated"
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

        result = calibrator.update(
            snapshot=_snapshot(accel_neu_xy=measured_accel_xy),
            estimate=_estimate(),
            hover_thrust=0.5,
            thrust_scale_calibration_completed=True,
            current_lateral_accel_gain_xy=np.ones(2, dtype=float),
            now=idx * 0.05,
        )
        last_command_accel_xy = np.asarray(
            result.debug.command_accel_xy_m_s2,
            dtype=float,
        )
        if result.lateral_accel_gain_xy is not None:
            break

    assert result is not None
    assert result.debug.status == "timeout_fallback"
    assert result.debug.signed_samples_xy[0] >= 2
    assert result.debug.signed_samples_xy[1] >= 2
    assert result.debug.signed_samples_xy[2] >= 2
    assert result.debug.signed_samples_xy[3] >= 2
    np.testing.assert_allclose(result.lateral_accel_gain_xy, np.ones(2), atol=1e-9)


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
