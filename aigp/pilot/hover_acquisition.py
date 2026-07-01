from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from autonomy_core.core.frame_conventions import (
    body_frd_to_local_ned_rotmat,
    local_ned_to_neu,
)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


@dataclass(frozen=True)
class HoverAcquisitionCommand:
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    thrust: float


@dataclass(frozen=True)
class HoverAcquisitionDebug:
    status: str
    active: bool
    completed: bool
    thrust: float
    hover_thrust: float
    elapsed_s: float
    dt_s: float
    z_rel_m: float
    vz_m_s: float
    az_m_s2: float
    lift_confirmed: bool
    stable_time_s: float
    armed: Optional[bool]
    release_z_m: float
    z_hold_error_m: float
    z_hold_vz_error_m_s: float
    z_hold_thrust_correction: float
    overshoot_thrust_floor: float


@dataclass(frozen=True)
class HoverAcquisitionResult:
    command: Optional[HoverAcquisitionCommand]
    hover_thrust: float
    debug: HoverAcquisitionDebug


class HoverAcquisition:
    """
    Competition-safe hover-thrust bootstrap for attitude-thrust control.

    This module does not target a known altitude or gate height. It commands
    level attitude and learns from relative vertical response before the race
    planner starts issuing gate-tracking commands.
    """

    def __init__(self, config):
        section = config.hover_acquisition
        self.state_mode = str(config.state_estimation.mode).lower()
        self.gravity_m_s2 = float(config.state_estimation.gravity_m_s2)

        self.enabled = bool(section.enabled)
        self.estimator_mode_only = bool(section.estimator_mode_only)
        self.require_armed = bool(section.require_armed)
        self.initial_thrust = _clamp(section.initial_thrust, 0.0, 1.0)
        self.min_thrust = _clamp(section.min_thrust, 0.0, 1.0)
        self.max_probe_thrust = _clamp(section.max_probe_thrust, 0.0, 1.0)
        if self.max_probe_thrust < self.min_thrust:
            self.min_thrust, self.max_probe_thrust = (
                self.max_probe_thrust,
                self.min_thrust,
            )

        self.thrust_step_per_s = max(0.0, float(section.thrust_step_per_s))
        self.thrust_trim_step_per_s = max(0.0, float(section.thrust_trim_step_per_s))
        self.velocity_gain = max(0.0, float(section.velocity_gain))
        self.accel_gain = max(0.0, float(section.accel_gain))
        self.accel_deadband_m_s2 = max(0.0, float(section.accel_deadband_m_s2))
        self.target_vz_m_s = float(section.target_vz_m_s)
        self.max_up_vz_m_s = max(0.0, float(section.max_up_vz_m_s))
        self.max_relative_z_m = max(0.0, float(section.max_relative_z_m))
        self.max_settle_vz_m_s = max(0.0, float(section.max_settle_vz_m_s))
        self.min_duration_s = max(0.0, float(section.min_duration_s))
        self.max_duration_s = max(0.0, float(section.max_duration_s))
        self.stable_duration_s = max(0.0, float(section.stable_duration_s))
        self.stable_vz_abs_m_s = max(0.0, float(section.stable_vz_abs_m_s))
        self.stable_accel_abs_m_s2 = max(0.0, float(section.stable_accel_abs_m_s2))
        self.lift_confirm_z_m = max(0.0, float(section.lift_confirm_z_m))
        self.lift_confirm_vz_m_s = max(0.0, float(section.lift_confirm_vz_m_s))
        self.relative_airborne_z_m = max(0.0, float(section.relative_airborne_z_m))
        self.min_release_z_m = max(0.0, float(section.min_release_z_m))
        self.min_confidence = _clamp(section.min_confidence, 0.0, 1.0)
        self.overshoot_thrust_step_per_s = max(
            0.0,
            float(section.overshoot_thrust_step_per_s),
        )
        self.overshoot_max_thrust_drop = max(
            0.0,
            float(section.overshoot_max_thrust_drop),
        )
        self.z_hold_enabled = bool(section.z_hold_enabled)
        self.z_hold_kp = max(0.0, float(section.z_hold_kp))
        self.z_hold_kv = max(0.0, float(section.z_hold_kv))
        self.z_hold_max_correction = max(
            0.0,
            float(section.z_hold_max_correction),
        )
        self.reset_hover_on_disarm = bool(section.reset_hover_on_disarm)
        self.release_on_timeout_while_unstable = bool(
            section.release_on_timeout_while_unstable
        )

        self.completed = False
        self.start_time: Optional[float] = None
        self.last_update_time: Optional[float] = None
        self.initial_z: Optional[float] = None
        self.hover_thrust = self.initial_thrust
        self.command_thrust = self.initial_thrust
        self.lift_confirmed = False
        self.lift_confirmed_thrust: Optional[float] = None
        self.stable_since: Optional[float] = None
        self.last_debug = self._debug(
            status="init",
            active=False,
            completed=False,
            thrust=self.hover_thrust,
            hover_thrust=self.hover_thrust,
            elapsed_s=0.0,
            dt_s=0.0,
            z_rel_m=0.0,
            vz_m_s=0.0,
            az_m_s2=0.0,
            stable_time_s=0.0,
            armed=None,
        )

    def update(
        self,
        *,
        snapshot,
        estimate,
        hover_thrust: float,
        now: Optional[float] = None,
    ) -> HoverAcquisitionResult:
        now = time.monotonic() if now is None else float(now)
        current_hover = _clamp(
            self._finite_float(hover_thrust, self.initial_thrust),
            self.min_thrust,
            self.max_probe_thrust,
        )

        if not self.enabled:
            self.completed = True
            return self._inactive("disabled", current_hover)
        if self.estimator_mode_only and self.state_mode != "estimator":
            self.completed = True
            return self._inactive("skipped_state_mode", current_hover)
        if self.completed:
            return self._inactive("complete", current_hover)
        if not bool(getattr(estimate, "valid", False)):
            return self._inactive("invalid_state", current_hover)
        if float(getattr(estimate, "confidence", 0.0)) < self.min_confidence:
            return self._inactive("low_confidence", current_hover)

        state_z, state_vz, yaw_rad = self._state_terms(snapshot, estimate)
        az_m_s2 = self._vertical_accel_neu(snapshot)
        armed = self._armed(snapshot)

        if self.require_armed and armed is False:
            self._reset_runtime()
            thrust = (
                self.initial_thrust
                if self.reset_hover_on_disarm
                else max(current_hover, self.initial_thrust)
            )
            self.hover_thrust = thrust
            self.command_thrust = thrust
            debug = self._debug(
                status="waiting_armed",
                active=True,
                completed=False,
                thrust=thrust,
                hover_thrust=thrust,
                elapsed_s=0.0,
                dt_s=0.0,
                z_rel_m=0.0,
                vz_m_s=state_vz,
                az_m_s2=az_m_s2,
                stable_time_s=0.0,
                armed=armed,
            )
            return HoverAcquisitionResult(
                command=HoverAcquisitionCommand(
                    roll_rad=0.0,
                    pitch_rad=0.0,
                    yaw_rad=yaw_rad,
                    thrust=thrust,
                ),
                hover_thrust=thrust,
                debug=debug,
            )

        if self.start_time is None:
            self.start_time = now
            self.last_update_time = now
            self.initial_z = state_z
            self.hover_thrust = (
                self.initial_thrust
                if self.reset_hover_on_disarm
                else max(current_hover, self.initial_thrust)
            )
            self.command_thrust = self.hover_thrust
            self.lift_confirmed = state_z >= self.relative_airborne_z_m
            self.lift_confirmed_thrust = (
                self.command_thrust if self.lift_confirmed else None
            )
            self.stable_since = None
            elapsed_s = 0.0
            dt_s = 0.0
        else:
            elapsed_s = max(0.0, now - self.start_time)
            dt_s = now - float(self.last_update_time or now)
            if not math.isfinite(dt_s) or dt_s < 0.0 or dt_s > 1.0:
                dt_s = 0.0
            self.last_update_time = now

        z_rel_m = state_z - float(
            self.initial_z if self.initial_z is not None else state_z
        )
        self._update_lift_confirmation(
            state_z=state_z,
            z_rel_m=z_rel_m,
            vz_m_s=state_vz,
            command_thrust=self.command_thrust,
        )
        overshoot = self._overshoot(z_rel_m=z_rel_m, vz_m_s=state_vz)
        if overshoot:
            self.lift_confirmed = True
            if self.lift_confirmed_thrust is None:
                self.lift_confirmed_thrust = self.command_thrust
            self.stable_since = None
        (
            z_hold_error_m,
            z_hold_vz_error_m_s,
            z_hold_thrust_correction,
        ) = self._z_hold_terms(z_rel_m=z_rel_m, vz_m_s=state_vz)
        overshoot_thrust_floor = math.nan

        if dt_s > 0.0:
            self.command_thrust = self._next_command_thrust(
                command_thrust=self.command_thrust,
                vz_m_s=state_vz,
                az_m_s2=az_m_s2,
                dt_s=dt_s,
                overshoot=overshoot,
            )
            if self._stable_sample_ok(
                elapsed_s=elapsed_s,
                z_rel_m=z_rel_m,
                vz_m_s=state_vz,
                az_m_s2=az_m_s2,
            ):
                self.hover_thrust = self._learn_hover_estimate(
                    hover_thrust=self.hover_thrust,
                    command_thrust=self.command_thrust,
                    dt_s=dt_s,
                )

        command_thrust = self._command_with_z_hold(z_hold_thrust_correction)

        stable_time_s = self._stable_time(
            now=now,
            elapsed_s=elapsed_s,
            z_rel_m=z_rel_m,
            vz_m_s=state_vz,
            az_m_s2=az_m_s2,
        )
        timed_out = self.max_duration_s > 0.0 and elapsed_s >= self.max_duration_s
        release_blocked = (
            timed_out
            and not self.release_on_timeout_while_unstable
            and self._release_unsafe(z_rel_m=z_rel_m, vz_m_s=state_vz, overshoot=overshoot)
        )
        if stable_time_s >= self.stable_duration_s or (timed_out and not release_blocked):
            self.completed = True
            status = "stable" if stable_time_s >= self.stable_duration_s else "timeout"
            if timed_out and not self.lift_confirmed:
                status = "timeout_lift_unconfirmed"
            if stable_time_s >= self.stable_duration_s:
                self.hover_thrust = self.command_thrust
            debug = self._debug(
                status=status,
                active=False,
                completed=True,
                thrust=command_thrust,
                hover_thrust=self.hover_thrust,
                elapsed_s=elapsed_s,
                dt_s=dt_s,
                z_rel_m=z_rel_m,
                vz_m_s=state_vz,
                az_m_s2=az_m_s2,
                stable_time_s=stable_time_s,
                armed=armed,
                release_z_m=self.min_release_z_m,
                z_hold_error_m=z_hold_error_m,
                z_hold_vz_error_m_s=z_hold_vz_error_m_s,
                z_hold_thrust_correction=z_hold_thrust_correction,
                overshoot_thrust_floor=overshoot_thrust_floor,
            )
            return HoverAcquisitionResult(
                command=None,
                hover_thrust=self.hover_thrust,
                debug=debug,
            )

        if overshoot:
            status = "overshoot_recover"
        elif release_blocked:
            status = "timeout_recovering"
        else:
            status = "seeking_lift" if not self.lift_confirmed else "settling"
        debug = self._debug(
            status=status,
            active=True,
            completed=False,
            thrust=command_thrust,
            hover_thrust=self.hover_thrust,
            elapsed_s=elapsed_s,
            dt_s=dt_s,
            z_rel_m=z_rel_m,
            vz_m_s=state_vz,
            az_m_s2=az_m_s2,
            stable_time_s=stable_time_s,
            armed=armed,
            release_z_m=self.min_release_z_m,
            z_hold_error_m=z_hold_error_m,
            z_hold_vz_error_m_s=z_hold_vz_error_m_s,
            z_hold_thrust_correction=z_hold_thrust_correction,
            overshoot_thrust_floor=overshoot_thrust_floor,
        )
        return HoverAcquisitionResult(
            command=HoverAcquisitionCommand(
                roll_rad=0.0,
                pitch_rad=0.0,
                yaw_rad=yaw_rad,
                thrust=command_thrust,
            ),
            hover_thrust=self.hover_thrust,
            debug=debug,
        )

    def _next_command_thrust(
        self,
        *,
        command_thrust: float,
        vz_m_s: float,
        az_m_s2: float,
        dt_s: float,
        overshoot: bool,
    ) -> float:
        accel_feedback = 0.0
        accel_valid = math.isfinite(az_m_s2)
        if accel_valid:
            accel_feedback = -self.accel_gain * az_m_s2

        target_vz = 0.0 if self.lift_confirmed else self.target_vz_m_s
        rate = self.velocity_gain * (target_vz - vz_m_s) + accel_feedback

        up_limit = self.thrust_trim_step_per_s if self.lift_confirmed else self.thrust_step_per_s
        down_limit = (
            self.overshoot_thrust_step_per_s
            if overshoot
            else self.thrust_trim_step_per_s
        )
        rate = _clamp(rate, -down_limit, up_limit)

        if not self.lift_confirmed:
            weak_accel = (not accel_valid) or az_m_s2 < self.accel_deadband_m_s2
            if vz_m_s < self.lift_confirm_vz_m_s and weak_accel:
                rate = max(rate, self.thrust_step_per_s)

        if overshoot:
            rate = min(rate, -down_limit)
        elif self.max_up_vz_m_s > 0.0 and vz_m_s > self.max_up_vz_m_s:
            rate = min(rate, -down_limit)

        thrust = command_thrust + rate * dt_s
        return _clamp(thrust, self.min_thrust, self.max_probe_thrust)

    def _overshoot(self, *, z_rel_m: float, vz_m_s: float) -> bool:
        high_above_band = (
            self.max_relative_z_m > 0.0 and z_rel_m >= self.max_relative_z_m
        )
        if (
            high_above_band
            and self.max_settle_vz_m_s > 0.0
            and vz_m_s < -self.max_settle_vz_m_s
        ):
            high_above_band = False
        return high_above_band or (
            self.max_up_vz_m_s > 0.0 and vz_m_s > self.max_up_vz_m_s
        )

    def _release_unsafe(self, *, z_rel_m: float, vz_m_s: float, overshoot: bool) -> bool:
        if not self.lift_confirmed:
            return True
        if self.min_release_z_m > 0.0 and z_rel_m < self.min_release_z_m:
            return True
        if overshoot:
            return True
        if self.max_settle_vz_m_s > 0.0 and abs(vz_m_s) > self.max_settle_vz_m_s:
            return True
        if self.max_relative_z_m > 0.0 and z_rel_m > self.max_relative_z_m:
            return True
        return False

    def _stable_time(
        self,
        *,
        now: float,
        elapsed_s: float,
        z_rel_m: float,
        vz_m_s: float,
        az_m_s2: float,
    ) -> float:
        if self._stable_sample_ok(
            elapsed_s=elapsed_s,
            z_rel_m=z_rel_m,
            vz_m_s=vz_m_s,
            az_m_s2=az_m_s2,
        ):
            if self.stable_since is None:
                self.stable_since = now
            return max(0.0, now - self.stable_since)

        self.stable_since = None
        return 0.0

    def _stable_sample_ok(
        self,
        *,
        elapsed_s: float,
        z_rel_m: float,
        vz_m_s: float,
        az_m_s2: float,
    ) -> bool:
        accel_stable = (
            not math.isfinite(az_m_s2)
            or abs(az_m_s2) <= self.stable_accel_abs_m_s2
        )
        release_height_ok = (
            self.min_release_z_m <= 0.0 or z_rel_m >= self.min_release_z_m
        )
        upper_height_ok = (
            self.max_relative_z_m <= 0.0 or z_rel_m <= self.max_relative_z_m
        )
        return (
            self.lift_confirmed
            and elapsed_s >= self.min_duration_s
            and release_height_ok
            and upper_height_ok
            and abs(vz_m_s) <= self.stable_vz_abs_m_s
            and accel_stable
        )

    def _update_lift_confirmation(
        self,
        *,
        state_z: float,
        z_rel_m: float,
        vz_m_s: float,
        command_thrust: float,
    ) -> None:
        if self.lift_confirmed:
            return
        self.lift_confirmed = (
            state_z >= self.relative_airborne_z_m
            or z_rel_m >= self.lift_confirm_z_m
            or vz_m_s >= self.lift_confirm_vz_m_s
        )
        if self.lift_confirmed:
            self.lift_confirmed_thrust = float(command_thrust)

    def _z_hold_terms(self, *, z_rel_m: float, vz_m_s: float) -> tuple[float, float, float]:
        if (
            not self.z_hold_enabled
            or not self.lift_confirmed
            or self.z_hold_max_correction <= 0.0
        ):
            return 0.0, 0.0, 0.0

        z_error_m = float(self.min_release_z_m - z_rel_m)
        vz_error_m_s = float(-vz_m_s)
        correction = self.z_hold_kp * z_error_m + self.z_hold_kv * vz_error_m_s
        if not math.isfinite(correction):
            correction = 0.0
        if self.max_relative_z_m > 0.0 and z_rel_m > self.max_relative_z_m:
            correction = min(correction, 0.0)
        correction = _clamp(
            correction,
            -self.z_hold_max_correction,
            self.z_hold_max_correction,
        )
        return z_error_m, vz_error_m_s, correction

    def _command_with_z_hold(self, z_hold_thrust_correction: float) -> float:
        thrust = self.command_thrust + self._finite_float(
            z_hold_thrust_correction,
            0.0,
        )
        return _clamp(thrust, self.min_thrust, self.max_probe_thrust)

    def _learn_hover_estimate(
        self,
        *,
        hover_thrust: float,
        command_thrust: float,
        dt_s: float,
    ) -> float:
        if dt_s <= 0.0:
            return hover_thrust
        alpha = _clamp(dt_s / max(self.stable_duration_s, 0.05), 0.0, 1.0)
        learned = float(hover_thrust) + alpha * (
            float(command_thrust) - float(hover_thrust)
        )
        return _clamp(learned, self.min_thrust, self.max_probe_thrust)

    def _overshoot_thrust_floor(self) -> float:
        if self.overshoot_max_thrust_drop <= 0.0:
            return math.nan
        if self.lift_confirmed_thrust is None:
            return math.nan
        return _clamp(
            float(self.lift_confirmed_thrust) - self.overshoot_max_thrust_drop,
            self.min_thrust,
            self.max_probe_thrust,
        )

    def _inactive(self, status: str, hover_thrust: float) -> HoverAcquisitionResult:
        debug = self._debug(
            status=status,
            active=False,
            completed=self.completed,
            thrust=hover_thrust,
            hover_thrust=hover_thrust,
            elapsed_s=0.0,
            dt_s=0.0,
            z_rel_m=0.0,
            vz_m_s=0.0,
            az_m_s2=0.0,
            stable_time_s=0.0,
            armed=None,
        )
        return HoverAcquisitionResult(command=None, hover_thrust=hover_thrust, debug=debug)

    def _reset_runtime(self) -> None:
        self.start_time = None
        self.last_update_time = None
        self.initial_z = None
        self.hover_thrust = self.initial_thrust
        self.command_thrust = self.initial_thrust
        self.lift_confirmed = False
        self.lift_confirmed_thrust = None
        self.stable_since = None

    def _debug(
        self,
        *,
        status: str,
        active: bool,
        completed: bool,
        thrust: float,
        hover_thrust: float,
        elapsed_s: float,
        dt_s: float,
        z_rel_m: float,
        vz_m_s: float,
        az_m_s2: float,
        stable_time_s: float,
        armed: Optional[bool],
        release_z_m: float = 0.0,
        z_hold_error_m: float = 0.0,
        z_hold_vz_error_m_s: float = 0.0,
        z_hold_thrust_correction: float = 0.0,
        overshoot_thrust_floor: float = math.nan,
    ) -> HoverAcquisitionDebug:
        self.last_debug = HoverAcquisitionDebug(
            status=str(status),
            active=bool(active),
            completed=bool(completed),
            thrust=float(thrust),
            hover_thrust=float(hover_thrust),
            elapsed_s=float(elapsed_s),
            dt_s=float(dt_s),
            z_rel_m=float(z_rel_m),
            vz_m_s=float(vz_m_s),
            az_m_s2=float(az_m_s2),
            lift_confirmed=bool(self.lift_confirmed),
            stable_time_s=float(stable_time_s),
            armed=armed,
            release_z_m=float(release_z_m),
            z_hold_error_m=float(z_hold_error_m),
            z_hold_vz_error_m_s=float(z_hold_vz_error_m_s),
            z_hold_thrust_correction=float(z_hold_thrust_correction),
            overshoot_thrust_floor=float(overshoot_thrust_floor),
        )
        return self.last_debug

    def _state_terms(self, snapshot, estimate) -> tuple[float, float, float]:
        pos = np.asarray(getattr(estimate, "pos_neu"), dtype=float).reshape(3)
        vel = np.asarray(getattr(estimate, "vel_neu"), dtype=float).reshape(3)
        yaw_rad = self._finite_float(
            getattr(estimate, "yaw_rad", None),
            self._finite_float(getattr(snapshot, "yaw_rad", None), 0.0),
        )
        return float(pos[2]), float(vel[2]), float(yaw_rad)

    def _vertical_accel_neu(self, snapshot) -> float:
        acc_body = self._vec3(getattr(snapshot, "accel_xyz", None))
        if acc_body is None:
            return math.nan

        rot_ned_body = body_frd_to_local_ned_rotmat(
            self._finite_float(getattr(snapshot, "roll_rad", None), 0.0),
            self._finite_float(getattr(snapshot, "pitch_rad", None), 0.0),
            self._finite_float(getattr(snapshot, "yaw_rad", None), 0.0),
        )
        acc_ned = rot_ned_body @ acc_body
        acc_ned = acc_ned + np.array([0.0, 0.0, self.gravity_m_s2], dtype=float)
        acc_neu = local_ned_to_neu(acc_ned)
        return float(acc_neu[2])

    @staticmethod
    def _armed(snapshot) -> Optional[bool]:
        value = getattr(snapshot, "armed", None)
        if value is not None:
            return bool(value)
        heartbeat = getattr(snapshot, "heartbeat", None)
        if isinstance(heartbeat, dict) and heartbeat.get("armed") is not None:
            return bool(heartbeat["armed"])
        return None

    @staticmethod
    def _vec3(value) -> Optional[np.ndarray]:
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=float).reshape(3)
        except (TypeError, ValueError):
            return None
        if np.all(np.isfinite(arr)):
            return arr.copy()
        return None

    @staticmethod
    def _finite_float(value, default: float) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float(default)
        return out if math.isfinite(out) else float(default)
