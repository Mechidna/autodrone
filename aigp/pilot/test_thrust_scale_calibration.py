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
        "min_samples_per_sign": 2,
        "neutral_max_abs_vz_m_s": 0.2,
        "neutral_max_abs_accel_m_s2": 0.5,
        "max_sign_slope_disagreement": 0.35,
        "accel_filter_alpha": 1.0,
        "accel_deadband_m_s2": 0.0,
        "min_abs_accel_m_s2": 0.0,
        "min_gain": 0.001,
        "max_gain": 1.0,
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
    accel_bias = 0.25
    last_delta = 0.0
    result = None

    for idx in range(30):
        result = calibrator.update(
            snapshot=_snapshot(
                az_neu=accel_bias + accel_per_thrust * last_delta,
            ),
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
    assert result.succeeded
    assert result.debug.succeeded
    assert result.debug.samples >= config.thrust_scale_calibration.min_samples
    assert (
        result.debug.positive_samples
        >= config.thrust_scale_calibration.min_samples_per_sign
    )
    assert (
        result.debug.negative_samples
        >= config.thrust_scale_calibration.min_samples_per_sign
    )
    assert result.thrust_from_acc_gain is not None
    assert abs(result.thrust_from_acc_gain - (1.0 / accel_per_thrust)) < 1e-9


def test_neutral_dwell_requires_continuous_stability():
    config = _config(initial_delay_s=0.2, min_duration_s=0.1)
    calibrator = ThrustScaleCalibration(config)

    for now, vz_m_s, expected_status in (
        (0.0, 0.0, "neutral_settle"),
        (0.1, 0.3, "waiting_neutral"),
        (0.2, 0.0, "neutral_settle"),
        (0.35, 0.0, "neutral_settle"),
    ):
        result = calibrator.update(
            snapshot=_snapshot(az_neu=0.0),
            estimate=_estimate(vel_neu=[0.0, 0.0, vz_m_s]),
            hover_thrust=0.5,
            hover_acquisition_completed=True,
            current_thrust_from_acc_gain=1.0 / 9.81,
            now=now,
        )
        assert result.command is not None
        assert result.debug.status == expected_status
        np.testing.assert_allclose(result.command.thrust, 0.5, atol=1e-9)

    started = calibrator.update(
        snapshot=_snapshot(az_neu=0.0),
        estimate=_estimate(),
        hover_thrust=0.5,
        hover_acquisition_completed=True,
        current_thrust_from_acc_gain=1.0 / 9.81,
        now=0.4,
    )
    assert started.command is not None
    assert started.debug.status == "probing"
    np.testing.assert_allclose(started.command.thrust, 0.54, atol=1e-9)


def test_positive_only_samples_cannot_succeed():
    config = _config(
        min_duration_s=0.1,
        max_duration_s=0.6,
        phase_duration_s=10.0,
    )
    calibrator = ThrustScaleCalibration(config)
    last_delta = 0.0
    result = None

    for idx in range(20):
        result = calibrator.update(
            snapshot=_snapshot(az_neu=8.0 * last_delta),
            estimate=_estimate(),
            hover_thrust=0.5,
            hover_acquisition_completed=True,
            current_thrust_from_acc_gain=1.0 / 9.81,
            now=idx * 0.05,
        )
        if result.command is not None:
            last_delta = result.command.thrust - 0.5
        if result.debug.completed:
            break

    assert result is not None
    assert result.debug.status == "timeout_fallback"
    assert result.debug.positive_samples >= config.thrust_scale_calibration.min_samples
    assert result.debug.negative_samples == 0
    assert result.thrust_from_acc_gain is None
    assert not result.succeeded
    assert not calibrator.succeeded


def test_sign_slope_disagreement_is_rejected():
    config = _config(min_duration_s=0.1, max_duration_s=0.8)
    calibrator = ThrustScaleCalibration(config)
    last_delta = 0.0
    result = None

    for idx in range(30):
        slope = 8.0 if last_delta >= 0.0 else 2.0
        result = calibrator.update(
            snapshot=_snapshot(az_neu=slope * last_delta),
            estimate=_estimate(),
            hover_thrust=0.5,
            hover_acquisition_completed=True,
            current_thrust_from_acc_gain=1.0 / 9.81,
            now=idx * 0.05,
        )
        if result.command is not None:
            last_delta = result.command.thrust - 0.5
        if result.debug.completed:
            break

    assert result is not None
    assert result.debug.status == "timeout_fallback"
    assert result.debug.sign_slope_disagreement > 0.35
    assert result.thrust_from_acc_gain is None
    assert not result.succeeded


def test_out_of_range_gain_is_rejected_instead_of_clamped_to_success():
    config = _config(
        min_duration_s=0.1,
        max_duration_s=0.8,
        min_gain=0.008,
        max_gain=0.04,
    )
    calibrator = ThrustScaleCalibration(config)
    last_delta = 0.0
    result = None

    for idx in range(30):
        result = calibrator.update(
            snapshot=_snapshot(az_neu=20.0 * last_delta),
            estimate=_estimate(),
            hover_thrust=0.5,
            hover_acquisition_completed=True,
            current_thrust_from_acc_gain=1.0 / 9.81,
            now=idx * 0.05,
        )
        if result.command is not None:
            last_delta = result.command.thrust - 0.5
        if result.debug.completed:
            break

    assert result is not None
    assert result.debug.status == "timeout_fallback"
    np.testing.assert_allclose(result.debug.accel_per_thrust, 20.0, atol=1e-9)
    assert result.thrust_from_acc_gain is None
    assert not result.succeeded
    assert not calibrator.succeeded


def test_motion_limit_completes_without_claiming_success():
    config = _config(initial_delay_s=0.0, max_abs_vz_m_s=0.4)
    calibrator = ThrustScaleCalibration(config)
    calibrator.update(
        snapshot=_snapshot(az_neu=0.0),
        estimate=_estimate(),
        hover_thrust=0.5,
        hover_acquisition_completed=True,
        current_thrust_from_acc_gain=1.0 / 9.81,
        now=0.0,
    )

    result = calibrator.update(
        snapshot=_snapshot(az_neu=0.0),
        estimate=_estimate(vel_neu=[0.0, 0.0, 0.5]),
        hover_thrust=0.5,
        hover_acquisition_completed=True,
        current_thrust_from_acc_gain=1.0 / 9.81,
        now=0.05,
    )

    assert result.debug.status == "motion_limited_fallback"
    assert result.debug.completed
    assert not result.debug.succeeded
    assert result.thrust_from_acc_gain is None
    assert not result.succeeded


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
