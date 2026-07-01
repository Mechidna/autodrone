from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from hover_acquisition import HoverAcquisition
from runtime_config import load_runtime_config


def _config(**overrides):
    config = load_runtime_config()
    values = {
        "enabled": True,
        "estimator_mode_only": False,
        "require_armed": True,
        "initial_thrust": 0.5,
        "min_thrust": 0.0,
        "max_probe_thrust": 0.85,
        "thrust_step_per_s": 0.25,
        "thrust_trim_step_per_s": 0.12,
        "velocity_gain": 0.35,
        "accel_gain": 0.05,
        "accel_deadband_m_s2": 0.30,
        "target_vz_m_s": 0.10,
        "max_up_vz_m_s": 0.80,
        "max_relative_z_m": 2.0,
        "max_settle_vz_m_s": 0.60,
        "min_duration_s": 0.10,
        "max_duration_s": 1.00,
        "stable_duration_s": 0.10,
        "stable_vz_abs_m_s": 0.25,
        "stable_accel_abs_m_s2": 0.80,
        "lift_confirm_z_m": 0.15,
        "lift_confirm_vz_m_s": 0.15,
        "relative_airborne_z_m": 0.25,
        "min_release_z_m": 0.15,
        "min_confidence": 0.0,
        "overshoot_thrust_step_per_s": 0.60,
        "overshoot_max_thrust_drop": 0.0,
        "z_hold_enabled": True,
        "z_hold_kp": 0.10,
        "z_hold_kv": 0.08,
        "z_hold_max_correction": 0.08,
        "reset_hover_on_disarm": True,
        "release_on_timeout_while_unstable": False,
        "print_period_s": 0.5,
    }
    values.update(overrides)
    return replace(
        config,
        hover_acquisition=replace(config.hover_acquisition, **values),
    )


def _snapshot(*, az_neu=0.0, gravity=9.81):
    return SimpleNamespace(
        roll_rad=0.0,
        pitch_rad=0.0,
        yaw_rad=0.0,
        accel_xyz=np.array([0.0, 0.0, -gravity - az_neu], dtype=float),
        armed=True,
    )


def _estimate(*, z, vz=0.0):
    return SimpleNamespace(
        valid=True,
        confidence=1.0,
        pos_neu=np.array([0.0, 0.0, z], dtype=float),
        vel_neu=np.array([0.0, 0.0, vz], dtype=float),
        yaw_rad=0.0,
    )


def test_does_not_release_after_falling_below_relative_release_height():
    acquisition = HoverAcquisition(_config())

    first = acquisition.update(
        snapshot=_snapshot(),
        estimate=_estimate(z=0.0),
        hover_thrust=0.5,
        now=0.0,
    )
    assert first.command is not None

    lifted = acquisition.update(
        snapshot=_snapshot(),
        estimate=_estimate(z=0.20, vz=0.20),
        hover_thrust=first.hover_thrust,
        now=0.10,
    )
    assert lifted.debug.lift_confirmed

    low = acquisition.update(
        snapshot=_snapshot(),
        estimate=_estimate(z=0.02, vz=0.0),
        hover_thrust=lifted.hover_thrust,
        now=0.35,
    )

    assert low.command is not None
    assert not low.debug.completed
    assert acquisition.completed is False
    assert low.debug.z_rel_m < low.debug.release_z_m
    assert low.debug.z_hold_thrust_correction > 0.0


def test_overshoot_recovery_has_no_artificial_thrust_floor():
    acquisition = HoverAcquisition(_config())

    start = acquisition.update(
        snapshot=_snapshot(),
        estimate=_estimate(z=0.0),
        hover_thrust=0.5,
        now=0.0,
    )
    lifted = acquisition.update(
        snapshot=_snapshot(),
        estimate=_estimate(z=0.20, vz=0.20),
        hover_thrust=start.hover_thrust,
        now=0.10,
    )
    overshoot = acquisition.update(
        snapshot=_snapshot(),
        estimate=_estimate(z=0.60, vz=1.00),
        hover_thrust=lifted.hover_thrust,
        now=1.00,
    )

    assert overshoot.command is not None
    assert np.isnan(overshoot.debug.overshoot_thrust_floor)
    assert overshoot.command.thrust < lifted.command.thrust
    np.testing.assert_allclose(overshoot.hover_thrust, lifted.hover_thrust)


def test_hover_estimate_updates_only_after_stable_hover_sample():
    acquisition = HoverAcquisition(_config(stable_duration_s=0.05))

    start = acquisition.update(
        snapshot=_snapshot(),
        estimate=_estimate(z=0.0),
        hover_thrust=0.5,
        now=0.0,
    )
    moving = acquisition.update(
        snapshot=_snapshot(),
        estimate=_estimate(z=0.25, vz=0.8),
        hover_thrust=start.hover_thrust,
        now=0.20,
    )

    assert moving.command is not None
    assert moving.debug.lift_confirmed
    np.testing.assert_allclose(moving.hover_thrust, 0.5)

    stable_start = acquisition.update(
        snapshot=_snapshot(),
        estimate=_estimate(z=0.25, vz=0.0),
        hover_thrust=moving.hover_thrust,
        now=0.30,
    )
    assert stable_start.command is not None
    assert not stable_start.debug.completed

    stable = acquisition.update(
        snapshot=_snapshot(),
        estimate=_estimate(z=0.25, vz=0.0),
        hover_thrust=stable_start.hover_thrust,
        now=0.36,
    )

    assert stable.debug.completed
    np.testing.assert_allclose(stable.hover_thrust, acquisition.command_thrust)
