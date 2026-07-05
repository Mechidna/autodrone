from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from runtime_config import load_runtime_config
from thrust_scale_calibration import ThrustScaleCalibration


def _config(**overrides):
    config = load_runtime_config()
    values = {
        "enabled": True,
        "estimator_mode_only": False,
        "require_hover_acquisition": True,
        "require_armed": True,
        "initial_delay_s": 0.0,
        "min_duration_s": 0.4,
        "max_duration_s": 1.0,
        "phase_duration_s": 0.15,
        "settle_duration_s": 0.02,
        "probe_delta_thrust": 0.04,
        "min_probe_delta_thrust": 0.02,
        "max_probe_delta_thrust": 0.08,
        "min_samples": 4,
        "accel_filter_alpha": 1.0,
        "accel_deadband_m_s2": 0.0,
        "min_abs_accel_m_s2": 0.0,
    }
    values.update(overrides)
    return replace(
        config,
        thrust_scale_calibration=replace(
            config.thrust_scale_calibration,
            **values,
        ),
    )


def _snapshot(*, az_neu, gravity=9.81):
    return SimpleNamespace(
        roll_rad=0.0,
        pitch_rad=0.0,
        yaw_rad=0.0,
        accel_xyz=np.array([0.0, 0.0, -gravity - az_neu], dtype=float),
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


def test_learns_thrust_from_acc_gain_from_small_probes():
    config = _config()
    calibrator = ThrustScaleCalibration(config)
    hover_thrust = 0.5
    accel_per_thrust = 8.0
    last_delta = 0.0
    result = None

    for idx in range(30):
        result = calibrator.update(
            snapshot=_snapshot(az_neu=accel_per_thrust * last_delta),
            estimate=_estimate(),
            hover_thrust=hover_thrust,
            hover_acquisition_completed=True,
            current_thrust_from_acc_gain=1.0 / 9.81,
            now=idx * 0.05,
        )
        if result.command is not None:
            last_delta = result.command.thrust - hover_thrust
        if result.thrust_from_acc_gain is not None:
            break

    assert result is not None
    assert result.debug.status == "calibrated"
    assert result.debug.samples >= config.thrust_scale_calibration.min_samples
    assert result.thrust_from_acc_gain is not None
    assert abs(result.thrust_from_acc_gain - (1.0 / accel_per_thrust)) < 1e-9


def test_z_hold_adds_bounded_thrust_when_calibration_drops():
    config = _config(
        initial_delay_s=0.0,
        z_hold_enabled=True,
        z_hold_kp=0.10,
        z_hold_kv=0.08,
        z_hold_max_correction=0.08,
    )
    calibrator = ThrustScaleCalibration(config)

    first = calibrator.update(
        snapshot=_snapshot(az_neu=0.0),
        estimate=_estimate(pos_neu=[0.0, 0.0, 1.0]),
        hover_thrust=0.5,
        hover_acquisition_completed=True,
        current_thrust_from_acc_gain=1.0 / 9.81,
        now=0.0,
    )
    assert first.command is not None

    dropped = calibrator.update(
        snapshot=_snapshot(az_neu=0.0),
        estimate=_estimate(
            pos_neu=[0.0, 0.0, 0.7],
            vel_neu=[0.0, 0.0, -0.2],
        ),
        hover_thrust=0.5,
        hover_acquisition_completed=True,
        current_thrust_from_acc_gain=1.0 / 9.81,
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
    np.testing.assert_allclose(dropped.command.thrust, 0.586, atol=1e-9)
