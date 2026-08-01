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
        "require_race_start": False,
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


def _snapshot(*, az_neu=0.0, gravity=9.81, armed=True, race_status=None):
    return SimpleNamespace(
        roll_rad=0.0,
        pitch_rad=0.0,
        yaw_rad=0.0,
        accel_xyz=np.array([0.0, 0.0, -gravity - az_neu], dtype=float),
        armed=armed,
        race_status=race_status,
    )


def _estimate(*, z, vz=0.0):
    return SimpleNamespace(
        valid=True,
        confidence=1.0,
        pos_neu=np.array([0.0, 0.0, z], dtype=float),
        vel_neu=np.array([0.0, 0.0, vz], dtype=float),
        yaw_rad=0.0,
    )


def _race_status(*, sim_boot_time_ms, race_start_boot_time_ms):
    return {
        "sim_boot_time_ms": sim_boot_time_ms,
        "race_start_boot_time_ms": race_start_boot_time_ms,
    }


def test_waits_for_race_start_without_advancing_probe():
    acquisition = HoverAcquisition(
        _config(
            require_race_start=True,
            initial_thrust=0.30,
        )
    )
    scheduled_start = 5_000

    first = acquisition.update(
        snapshot=_snapshot(
            race_status=_race_status(
                sim_boot_time_ms=1_000,
                race_start_boot_time_ms=scheduled_start,
            )
        ),
        estimate=_estimate(z=2.0),
        hover_thrust=0.30,
        now=0.0,
    )
    still_waiting = acquisition.update(
        snapshot=_snapshot(
            race_status=_race_status(
                sim_boot_time_ms=4_999,
                race_start_boot_time_ms=scheduled_start,
            )
        ),
        estimate=_estimate(z=2.1),
        hover_thrust=first.hover_thrust,
        now=4.99,
    )

    for result in (first, still_waiting):
        assert result.command is not None
        assert result.debug.status == "waiting_race_start"
        assert result.debug.elapsed_s == 0.0
        assert result.command.thrust == 0.30
        assert result.hover_thrust == 0.30
    assert acquisition.start_time is None
    assert acquisition.last_update_time is None
    assert acquisition.initial_z is None
    assert acquisition.stable_since is None
    assert acquisition.command_thrust == 0.30
    assert not acquisition.lift_confirmed


def test_race_start_initializes_clock_and_altitude_reference_at_go():
    acquisition = HoverAcquisition(
        _config(
            require_race_start=True,
            initial_thrust=0.30,
        )
    )
    scheduled_start = 5_000

    acquisition.update(
        snapshot=_snapshot(
            race_status=_race_status(
                sim_boot_time_ms=4_900,
                race_start_boot_time_ms=scheduled_start,
            )
        ),
        estimate=_estimate(z=2.0),
        hover_thrust=0.30,
        now=4.9,
    )
    started = acquisition.update(
        snapshot=_snapshot(
            race_status=_race_status(
                sim_boot_time_ms=scheduled_start,
                race_start_boot_time_ms=scheduled_start,
            )
        ),
        estimate=_estimate(z=2.2),
        hover_thrust=0.30,
        now=5.0,
    )

    assert started.command is not None
    assert started.debug.status == "seeking_lift"
    assert started.debug.elapsed_s == 0.0
    assert started.debug.dt_s == 0.0
    assert started.debug.z_rel_m == 0.0
    assert started.command.thrust == 0.30
    assert acquisition.start_time == 5.0
    assert acquisition.initial_z == 2.2

    after_go = acquisition.update(
        snapshot=_snapshot(
            race_status=_race_status(
                sim_boot_time_ms=5_100,
                race_start_boot_time_ms=scheduled_start,
            )
        ),
        estimate=_estimate(z=2.25),
        hover_thrust=started.hover_thrust,
        now=5.1,
    )
    assert np.isclose(after_go.debug.elapsed_s, 0.1)
    assert np.isclose(after_go.debug.z_rel_m, 0.05)


def test_race_gate_fails_closed_for_missing_or_unready_status():
    unready_statuses = (
        (None, "waiting_race_status"),
        ({}, "waiting_race_status"),
        (
            _race_status(sim_boot_time_ms=1_000, race_start_boot_time_ms=-1),
            "waiting_race_start",
        ),
        (
            _race_status(sim_boot_time_ms=1_000, race_start_boot_time_ms=0),
            "waiting_race_start",
        ),
        (
            _race_status(sim_boot_time_ms=1_000, race_start_boot_time_ms=1_001),
            "waiting_race_start",
        ),
    )

    for race_status, expected_status in unready_statuses:
        acquisition = HoverAcquisition(_config(require_race_start=True))
        result = acquisition.update(
            snapshot=_snapshot(race_status=race_status),
            estimate=_estimate(z=0.0),
            hover_thrust=0.5,
            now=10.0,
        )
        assert result.debug.status == expected_status
        assert acquisition.start_time is None


def test_arming_remains_required_even_when_race_has_started():
    acquisition = HoverAcquisition(_config(require_race_start=True))
    result = acquisition.update(
        snapshot=_snapshot(
            armed=False,
            race_status=_race_status(
                sim_boot_time_ms=5_001,
                race_start_boot_time_ms=5_000,
            ),
        ),
        estimate=_estimate(z=0.0),
        hover_thrust=0.5,
        now=5.001,
    )

    assert result.debug.status == "waiting_armed"
    assert acquisition.start_time is None


def test_px4_mode_does_not_require_competition_race_status():
    config = _config(require_race_start=True)
    config = replace(
        config,
        runtime=replace(config.runtime, runner_mode="px4"),
    )
    acquisition = HoverAcquisition(config)

    result = acquisition.update(
        snapshot=_snapshot(race_status=None),
        estimate=_estimate(z=0.0),
        hover_thrust=0.5,
        now=1.0,
    )

    assert result.debug.status == "seeking_lift"
    assert acquisition.start_time == 1.0


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


def test_relaxed_timeout_can_release_without_forcing_startup_climb():
    acquisition = HoverAcquisition(
        _config(
            release_on_timeout_while_unstable=True,
            min_release_z_m=0.15,
            lift_confirm_z_m=0.15,
            relative_airborne_z_m=0.25,
            max_duration_s=1.0,
        )
    )

    start = acquisition.update(
        snapshot=_snapshot(),
        estimate=_estimate(z=-20.0, vz=0.0),
        hover_thrust=0.5,
        now=0.0,
    )
    assert start.command is not None
    assert not start.debug.lift_confirmed

    timed_out = acquisition.update(
        snapshot=_snapshot(),
        estimate=_estimate(z=-20.0, vz=0.0),
        hover_thrust=start.hover_thrust,
        now=1.10,
    )

    assert timed_out.command is None
    assert timed_out.debug.completed
    assert timed_out.debug.status == "timeout_safe_lift_unconfirmed"


def test_relaxed_timeout_does_not_release_while_high_above_start():
    acquisition = HoverAcquisition(
        _config(
            release_on_timeout_while_unstable=True,
            min_release_z_m=0.0,
            lift_confirm_z_m=0.0,
            relative_airborne_z_m=0.0,
            max_duration_s=1.0,
            max_relative_z_m=2.0,
        )
    )

    start = acquisition.update(
        snapshot=_snapshot(),
        estimate=_estimate(z=0.0, vz=0.0),
        hover_thrust=0.5,
        now=0.0,
    )
    assert start.command is not None

    high = acquisition.update(
        snapshot=_snapshot(),
        estimate=_estimate(z=8.0, vz=0.0),
        hover_thrust=start.hover_thrust,
        now=1.10,
    )

    assert high.command is not None
    assert not high.debug.completed
    assert high.debug.status == "overshoot_recover"


def test_timeout_remains_fail_closed_when_unstable_release_is_disabled():
    acquisition = HoverAcquisition(
        _config(
            release_on_timeout_while_unstable=False,
            min_release_z_m=0.15,
            lift_confirm_z_m=0.15,
            relative_airborne_z_m=0.15,
            max_duration_s=1.0,
        )
    )

    start = acquisition.update(
        snapshot=_snapshot(),
        estimate=_estimate(z=0.0, vz=0.0),
        hover_thrust=0.5,
        now=0.0,
    )
    lifted = acquisition.update(
        snapshot=_snapshot(),
        estimate=_estimate(z=0.20, vz=0.20),
        hover_thrust=start.hover_thrust,
        now=0.10,
    )
    timed_out = acquisition.update(
        snapshot=_snapshot(az_neu=2.0),
        estimate=_estimate(z=0.20, vz=0.10),
        hover_thrust=lifted.hover_thrust,
        now=1.10,
    )

    assert timed_out.command is not None
    assert not timed_out.debug.completed
    assert timed_out.debug.status == "timeout_recovering"
    assert acquisition.completed is False


def test_lift_confirmed_below_release_height_keeps_climbing_through_timeout():
    acquisition = HoverAcquisition(
        _config(
            initial_thrust=0.30,
            z_hold_enabled=False,
            min_release_z_m=0.25,
            lift_confirm_z_m=0.15,
            lift_confirm_vz_m_s=0.15,
            relative_airborne_z_m=0.25,
            max_duration_s=1.0,
            release_on_timeout_while_unstable=False,
        )
    )

    start = acquisition.update(
        snapshot=_snapshot(),
        estimate=_estimate(z=0.0, vz=0.0),
        hover_thrust=0.30,
        now=10.0,
    )
    lifted = acquisition.update(
        snapshot=_snapshot(),
        estimate=_estimate(z=0.15, vz=0.23),
        hover_thrust=start.hover_thrust,
        now=10.39,
    )
    plateau = acquisition.update(
        snapshot=_snapshot(),
        estimate=_estimate(z=0.20, vz=0.0),
        hover_thrust=lifted.hover_thrust,
        now=10.90,
    )
    timed_out_1 = acquisition.update(
        snapshot=_snapshot(),
        estimate=_estimate(z=0.20, vz=0.0),
        hover_thrust=plateau.hover_thrust,
        now=11.10,
    )
    timed_out_2 = acquisition.update(
        snapshot=_snapshot(),
        estimate=_estimate(z=0.20, vz=0.0),
        hover_thrust=timed_out_1.hover_thrust,
        now=11.30,
    )

    assert lifted.debug.lift_confirmed
    assert plateau.debug.status == "settling"
    assert plateau.debug.z_rel_m < plateau.debug.release_z_m
    assert timed_out_1.debug.status == "timeout_recovering"
    assert timed_out_2.debug.status == "timeout_recovering"
    assert all(
        result.command is not None
        for result in (plateau, timed_out_1, timed_out_2)
    )
    assert timed_out_1.command.thrust > plateau.command.thrust
    assert timed_out_2.command.thrust > timed_out_1.command.thrust
    assert not acquisition.completed


def test_lift_confirmed_at_release_height_returns_to_zero_vz_target():
    acquisition = HoverAcquisition(
        _config(
            initial_thrust=0.30,
            z_hold_enabled=False,
            min_release_z_m=0.25,
            lift_confirm_z_m=0.15,
            lift_confirm_vz_m_s=0.15,
            relative_airborne_z_m=0.25,
            max_duration_s=10.0,
            stable_duration_s=5.0,
        )
    )

    start = acquisition.update(
        snapshot=_snapshot(),
        estimate=_estimate(z=0.0, vz=0.0),
        hover_thrust=0.30,
        now=10.0,
    )
    acquisition.update(
        snapshot=_snapshot(),
        estimate=_estimate(z=0.25, vz=0.20),
        hover_thrust=start.hover_thrust,
        now=10.40,
    )
    at_release_1 = acquisition.update(
        snapshot=_snapshot(),
        estimate=_estimate(z=0.25, vz=0.0),
        hover_thrust=start.hover_thrust,
        now=10.80,
    )
    at_release_2 = acquisition.update(
        snapshot=_snapshot(),
        estimate=_estimate(z=0.25, vz=0.0),
        hover_thrust=at_release_1.hover_thrust,
        now=11.00,
    )

    assert at_release_1.command is not None
    assert at_release_2.command is not None
    np.testing.assert_allclose(
        at_release_2.command.thrust,
        at_release_1.command.thrust,
    )


def test_ground_contact_acceleration_does_not_reverse_takeoff_ramp():
    acquisition = HoverAcquisition(
        _config(
            initial_thrust=0.27,
            thrust_step_per_s=0.25,
        )
    )

    start = acquisition.update(
        snapshot=_snapshot(),
        estimate=_estimate(z=0.0, vz=0.0),
        hover_thrust=0.27,
        now=10.0,
    )
    ground_contact = acquisition.update(
        snapshot=_snapshot(az_neu=2.0),
        estimate=_estimate(z=0.0, vz=0.0),
        hover_thrust=start.hover_thrust,
        now=10.5,
    )

    assert ground_contact.command is not None
    assert not ground_contact.debug.lift_confirmed
    np.testing.assert_allclose(ground_contact.command.thrust, 0.395)


def test_overshoot_recovery_floor_can_be_disabled():
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


def test_overshoot_recovery_floor_clamps_internal_and_final_thrust():
    acquisition = HoverAcquisition(
        _config(
            initial_thrust=0.27,
            overshoot_max_thrust_drop=0.25,
            min_release_z_m=0.25,
            z_hold_enabled=True,
            z_hold_max_correction=0.08,
        )
    )

    start = acquisition.update(
        snapshot=_snapshot(),
        estimate=_estimate(z=0.0, vz=0.0),
        hover_thrust=0.27,
        now=10.0,
    )
    takeoff_power = acquisition.update(
        snapshot=_snapshot(),
        estimate=_estimate(z=0.0, vz=0.0),
        hover_thrust=start.hover_thrust,
        now=11.0,
    )
    lifted = acquisition.update(
        snapshot=_snapshot(),
        estimate=_estimate(z=0.20, vz=0.20),
        hover_thrust=takeoff_power.hover_thrust,
        now=11.02,
    )
    overshoot = acquisition.update(
        snapshot=_snapshot(az_neu=4.0),
        estimate=_estimate(z=0.60, vz=1.00),
        hover_thrust=lifted.hover_thrust,
        now=11.52,
    )

    assert takeoff_power.command is not None
    assert lifted.command is not None
    assert overshoot.command is not None
    np.testing.assert_allclose(takeoff_power.command.thrust, 0.52)
    np.testing.assert_allclose(overshoot.debug.overshoot_thrust_floor, 0.27)
    np.testing.assert_allclose(acquisition.command_thrust, 0.27)
    np.testing.assert_allclose(overshoot.command.thrust, 0.27)
    assert overshoot.debug.z_hold_thrust_correction < 0.0


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

    assert stable.command is None
    assert stable.debug.completed
    assert stable.debug.status == "stable"
    np.testing.assert_allclose(stable.debug.z_hold_thrust_correction, -0.01)
    np.testing.assert_allclose(stable.hover_thrust, stable.debug.thrust)
    np.testing.assert_allclose(stable.debug.hover_thrust, stable.debug.thrust)
    np.testing.assert_allclose(acquisition.hover_thrust, stable.debug.thrust)
    np.testing.assert_allclose(stable.hover_thrust, acquisition.command_thrust)
    np.testing.assert_allclose(stable.hover_thrust, stable_start.command.thrust)
