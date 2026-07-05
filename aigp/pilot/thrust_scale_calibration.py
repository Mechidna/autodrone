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
class ThrustScaleCalibrationCommand:
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    thrust: float


@dataclass(frozen=True)
class ThrustScaleCalibrationDebug:
    status: str
    active: bool
    completed: bool
    elapsed_s: float
    dt_s: float
    thrust: float
    hover_thrust: float
    thrust_delta: float
    az_m_s2: float
    vz_m_s: float
    z_rel_m: float
    z_hold_error_m: float
    z_hold_vz_error_m_s: float
    z_hold_thrust_correction: float
    samples: int
    accel_per_thrust: float
    thrust_from_acc_gain: float
    confidence: float
    accel_source: str
    armed: Optional[bool]


@dataclass(frozen=True)
class ThrustScaleCalibrationResult:
    command: Optional[ThrustScaleCalibrationCommand]
    thrust_from_acc_gain: Optional[float]
    debug: ThrustScaleCalibrationDebug


class ThrustScaleCalibration:
    """
    Learn normalized thrust units per vertical m/s^2 after hover acquisition.

    The calibration is competition-safe: it does not use gate truth, altitude
    targets, or simulator-only signals. It holds level attitude near the learned
    hover thrust, applies small alternating thrust probes, and estimates the
    local slope from onboard vertical acceleration response.
    """

    def __init__(self, config):
        section = config.thrust_scale_calibration
        self.state_mode = str(config.state_estimation.mode).lower()
        self.gravity_m_s2 = float(config.state_estimation.gravity_m_s2)

        self.enabled = bool(section.enabled)
        self.estimator_mode_only = bool(section.estimator_mode_only)
        self.require_hover_acquisition = bool(section.require_hover_acquisition)
        self.require_armed = bool(section.require_armed)
        self.initial_delay_s = max(0.0, float(section.initial_delay_s))
        self.min_duration_s = max(0.0, float(section.min_duration_s))
        self.max_duration_s = max(0.0, float(section.max_duration_s))
        self.phase_duration_s = max(0.05, float(section.phase_duration_s))
        self.settle_duration_s = max(0.0, float(section.settle_duration_s))
        self.probe_delta_thrust = abs(float(section.probe_delta_thrust))
        self.min_probe_delta_thrust = abs(float(section.min_probe_delta_thrust))
        self.max_probe_delta_thrust = abs(float(section.max_probe_delta_thrust))
        self.probe_delta_thrust = _clamp(
            self.probe_delta_thrust,
            self.min_probe_delta_thrust,
            self.max_probe_delta_thrust,
        )
        self.min_samples = max(1, int(section.min_samples))
        self.accel_filter_alpha = _clamp(section.accel_filter_alpha, 0.0, 1.0)
        self.accel_deadband_m_s2 = max(0.0, float(section.accel_deadband_m_s2))
        self.min_abs_accel_m_s2 = max(0.0, float(section.min_abs_accel_m_s2))
        self.max_abs_accel_m_s2 = max(0.0, float(section.max_abs_accel_m_s2))
        self.max_abs_vz_m_s = max(0.0, float(section.max_abs_vz_m_s))
        self.max_relative_z_m = max(0.0, float(section.max_relative_z_m))
        self.z_hold_enabled = bool(section.z_hold_enabled)
        self.z_hold_kp = max(0.0, float(section.z_hold_kp))
        self.z_hold_kv = max(0.0, float(section.z_hold_kv))
        self.z_hold_max_correction = max(
            0.0,
            float(section.z_hold_max_correction),
        )
        self.min_gain = max(1e-6, float(section.min_gain))
        self.max_gain = max(self.min_gain, float(section.max_gain))
        self.result_alpha = _clamp(section.result_alpha, 0.0, 1.0)

        self.completed = False
        self.start_time: Optional[float] = None
        self.last_update_time: Optional[float] = None
        self.initial_z: Optional[float] = None
        self.command_delta = 0.0
        self.command_delta_since: Optional[float] = None
        self.filtered_az: Optional[float] = None
        self.prev_vz: Optional[float] = None
        self.prev_vz_time: Optional[float] = None
        self.sum_delta_accel = 0.0
        self.sum_delta_sq = 0.0
        self.samples = 0
        self.last_gain = self._default_gain()
        self.last_debug = self._debug(
            status="init",
            active=False,
            completed=False,
            elapsed_s=0.0,
            dt_s=0.0,
            thrust=0.0,
            hover_thrust=0.0,
            thrust_delta=0.0,
            az_m_s2=math.nan,
            vz_m_s=0.0,
            z_rel_m=0.0,
            z_hold_error_m=0.0,
            z_hold_vz_error_m_s=0.0,
            z_hold_thrust_correction=0.0,
            accel_source="none",
            armed=None,
        )

    def update(
        self,
        *,
        snapshot,
        estimate,
        hover_thrust: float,
        hover_acquisition_completed: bool,
        current_thrust_from_acc_gain: float,
        now: Optional[float] = None,
    ) -> ThrustScaleCalibrationResult:
        now = time.monotonic() if now is None else float(now)
        hover = _clamp(self._finite_float(hover_thrust, 0.5), 0.0, 1.0)
        current_gain = self._valid_gain(current_thrust_from_acc_gain)
        if current_gain is not None:
            self.last_gain = current_gain

        if not self.enabled:
            self.completed = True
            return self._inactive("disabled", hover)
        if self.estimator_mode_only and self.state_mode != "estimator":
            self.completed = True
            return self._inactive("skipped_state_mode", hover)
        if self.completed:
            return self._inactive("complete", hover)
        if self.require_hover_acquisition and not hover_acquisition_completed:
            return self._inactive("waiting_hover_acquisition", hover)
        if not bool(getattr(estimate, "valid", False)):
            return self._inactive("invalid_state", hover)

        state_z, state_vz, yaw_rad = self._state_terms(snapshot, estimate)
        armed = self._armed(snapshot)
        if self.require_armed and armed is False:
            self._reset_runtime()
            debug = self._debug(
                status="waiting_armed",
                active=True,
                completed=False,
                elapsed_s=0.0,
                dt_s=0.0,
                thrust=hover,
                hover_thrust=hover,
                thrust_delta=0.0,
                az_m_s2=math.nan,
                vz_m_s=state_vz,
                z_rel_m=0.0,
                z_hold_error_m=0.0,
                z_hold_vz_error_m_s=0.0,
                z_hold_thrust_correction=0.0,
                accel_source="none",
                armed=armed,
            )
            return ThrustScaleCalibrationResult(
                command=ThrustScaleCalibrationCommand(0.0, 0.0, yaw_rad, hover),
                thrust_from_acc_gain=None,
                debug=debug,
            )

        if self.start_time is None:
            self.start_time = now
            self.last_update_time = now
            self.initial_z = state_z
            self.command_delta = 0.0
            self.command_delta_since = now
            self.filtered_az = None
            self.prev_vz = state_vz
            self.prev_vz_time = now
            elapsed_s = 0.0
            dt_s = 0.0
        else:
            elapsed_s = max(0.0, now - self.start_time)
            dt_s = now - float(self.last_update_time or now)
            if not math.isfinite(dt_s) or dt_s < 0.0 or dt_s > 1.0:
                dt_s = 0.0
            self.last_update_time = now

        z_rel_m = state_z - float(self.initial_z if self.initial_z is not None else state_z)
        (
            z_hold_error_m,
            z_hold_vz_error_m_s,
            z_hold_thrust_correction,
        ) = self._z_hold_terms(z_rel_m=z_rel_m, vz_m_s=state_vz)
        az_raw, accel_source = self._vertical_accel_neu(snapshot, state_vz, now)
        az_m_s2 = self._filtered_accel(az_raw)
        self._record_sample(
            now=now,
            thrust_delta=self.command_delta,
            az_m_s2=az_m_s2,
            vz_m_s=state_vz,
            z_hold_thrust_correction=z_hold_thrust_correction,
        )

        accel_per_thrust = self._accel_per_thrust()
        estimated_gain = self._gain_from_slope(accel_per_thrust)
        enough_sample_count = self.samples >= self.min_samples
        if enough_sample_count and estimated_gain is not None:
            self.last_gain = self._blend_gain(self.last_gain, estimated_gain)

        motion_limited = self._motion_limited(z_rel_m=z_rel_m, vz_m_s=state_vz)
        timed_out = self.max_duration_s > 0.0 and elapsed_s >= self.max_duration_s
        enough_samples = enough_sample_count and elapsed_s >= self.min_duration_s
        valid_calibration = enough_samples and estimated_gain is not None
        if motion_limited or timed_out or valid_calibration:
            self.completed = True
            if valid_calibration:
                status = "calibrated"
            elif enough_sample_count and estimated_gain is not None:
                status = "motion_limited_calibrated" if motion_limited else "timeout_calibrated"
            elif motion_limited:
                status = "motion_limited_fallback"
            else:
                status = "timeout_fallback"
            debug = self._debug(
                status=status,
                active=False,
                completed=True,
                elapsed_s=elapsed_s,
                dt_s=dt_s,
                thrust=hover,
                hover_thrust=hover,
                thrust_delta=0.0,
                az_m_s2=az_m_s2,
                vz_m_s=state_vz,
                z_rel_m=z_rel_m,
                z_hold_error_m=z_hold_error_m,
                z_hold_vz_error_m_s=z_hold_vz_error_m_s,
                z_hold_thrust_correction=z_hold_thrust_correction,
                accel_source=accel_source,
                armed=armed,
            )
            return ThrustScaleCalibrationResult(
                command=None,
                thrust_from_acc_gain=self.last_gain,
                debug=debug,
            )

        desired_delta = self._desired_probe_delta(elapsed_s)
        if abs(desired_delta - self.command_delta) > 1e-6:
            self.command_delta = desired_delta
            self.command_delta_since = now

        thrust = _clamp(hover + self.command_delta + z_hold_thrust_correction, 0.0, 1.0)
        actual_delta = thrust - hover
        expected_delta = self.command_delta + z_hold_thrust_correction
        if abs(actual_delta - expected_delta) > 1e-6:
            self.command_delta = actual_delta
            self.command_delta_since = now

        status = "delay" if elapsed_s < self.initial_delay_s else "probing"
        debug = self._debug(
            status=status,
            active=True,
            completed=False,
            elapsed_s=elapsed_s,
            dt_s=dt_s,
            thrust=thrust,
            hover_thrust=hover,
            thrust_delta=self.command_delta,
            az_m_s2=az_m_s2,
            vz_m_s=state_vz,
            z_rel_m=z_rel_m,
            z_hold_error_m=z_hold_error_m,
            z_hold_vz_error_m_s=z_hold_vz_error_m_s,
            z_hold_thrust_correction=z_hold_thrust_correction,
            accel_source=accel_source,
            armed=armed,
        )
        return ThrustScaleCalibrationResult(
            command=ThrustScaleCalibrationCommand(
                roll_rad=0.0,
                pitch_rad=0.0,
                yaw_rad=yaw_rad,
                thrust=thrust,
            ),
            thrust_from_acc_gain=None,
            debug=debug,
        )

    def _desired_probe_delta(self, elapsed_s: float) -> float:
        if elapsed_s < self.initial_delay_s:
            return 0.0
        phase = int((elapsed_s - self.initial_delay_s) / self.phase_duration_s)
        return self.probe_delta_thrust if phase % 2 == 0 else -self.probe_delta_thrust

    def _record_sample(
        self,
        *,
        now: float,
        thrust_delta: float,
        az_m_s2: float,
        vz_m_s: float,
        z_hold_thrust_correction: float,
    ) -> None:
        if abs(z_hold_thrust_correction) > 1e-3:
            return
        if abs(thrust_delta) < self.min_probe_delta_thrust:
            return
        if self.command_delta_since is None:
            return
        if now - self.command_delta_since < self.settle_duration_s:
            return
        if not math.isfinite(az_m_s2):
            return
        if self.max_abs_accel_m_s2 > 0.0 and abs(az_m_s2) > self.max_abs_accel_m_s2:
            return
        if self.max_abs_vz_m_s > 0.0 and abs(vz_m_s) > self.max_abs_vz_m_s:
            return
        if abs(az_m_s2) < self.min_abs_accel_m_s2:
            return

        accel = 0.0 if abs(az_m_s2) <= self.accel_deadband_m_s2 else az_m_s2
        if thrust_delta * accel <= 0.0:
            return

        self.sum_delta_accel += thrust_delta * accel
        self.sum_delta_sq += thrust_delta * thrust_delta
        self.samples += 1

    def _accel_per_thrust(self) -> float:
        if self.sum_delta_sq <= 1e-9:
            return math.nan
        return self.sum_delta_accel / self.sum_delta_sq

    def _gain_from_slope(self, accel_per_thrust: float) -> Optional[float]:
        if not math.isfinite(accel_per_thrust) or accel_per_thrust <= 1e-6:
            return None
        gain = 1.0 / accel_per_thrust
        if not math.isfinite(gain):
            return None
        return _clamp(gain, self.min_gain, self.max_gain)

    def _blend_gain(self, current_gain: float, estimated_gain: float) -> float:
        alpha = self.result_alpha
        return _clamp(
            (1.0 - alpha) * float(current_gain) + alpha * float(estimated_gain),
            self.min_gain,
            self.max_gain,
        )

    def _z_hold_terms(self, *, z_rel_m: float, vz_m_s: float) -> tuple[float, float, float]:
        if not self.z_hold_enabled or self.z_hold_max_correction <= 0.0:
            return 0.0, 0.0, 0.0

        z_error_m = float(-z_rel_m)
        vz_error_m_s = float(-vz_m_s)
        correction = self.z_hold_kp * z_error_m + self.z_hold_kv * vz_error_m_s
        if not math.isfinite(correction):
            correction = 0.0
        correction = _clamp(
            correction,
            -self.z_hold_max_correction,
            self.z_hold_max_correction,
        )
        return z_error_m, vz_error_m_s, correction

    def _motion_limited(self, *, z_rel_m: float, vz_m_s: float) -> bool:
        if self.max_relative_z_m > 0.0 and abs(z_rel_m) > self.max_relative_z_m:
            return True
        if self.max_abs_vz_m_s > 0.0 and abs(vz_m_s) > self.max_abs_vz_m_s:
            return True
        return False

    def _inactive(self, status: str, hover_thrust: float) -> ThrustScaleCalibrationResult:
        debug = self._debug(
            status=status,
            active=False,
            completed=self.completed,
            elapsed_s=0.0,
            dt_s=0.0,
            thrust=hover_thrust,
            hover_thrust=hover_thrust,
            thrust_delta=0.0,
            az_m_s2=math.nan,
            vz_m_s=0.0,
            z_rel_m=0.0,
            z_hold_error_m=0.0,
            z_hold_vz_error_m_s=0.0,
            z_hold_thrust_correction=0.0,
            accel_source="none",
            armed=None,
        )
        return ThrustScaleCalibrationResult(
            command=None,
            thrust_from_acc_gain=None,
            debug=debug,
        )

    def _reset_runtime(self) -> None:
        self.start_time = None
        self.last_update_time = None
        self.initial_z = None
        self.command_delta = 0.0
        self.command_delta_since = None
        self.filtered_az = None
        self.prev_vz = None
        self.prev_vz_time = None
        self.sum_delta_accel = 0.0
        self.sum_delta_sq = 0.0
        self.samples = 0

    def _debug(
        self,
        *,
        status: str,
        active: bool,
        completed: bool,
        elapsed_s: float,
        dt_s: float,
        thrust: float,
        hover_thrust: float,
        thrust_delta: float,
        az_m_s2: float,
        vz_m_s: float,
        z_rel_m: float,
        z_hold_error_m: float,
        z_hold_vz_error_m_s: float,
        z_hold_thrust_correction: float,
        accel_source: str,
        armed: Optional[bool],
    ) -> ThrustScaleCalibrationDebug:
        accel_per_thrust = self._accel_per_thrust()
        confidence = min(1.0, float(self.samples) / float(self.min_samples))
        self.last_debug = ThrustScaleCalibrationDebug(
            status=str(status),
            active=bool(active),
            completed=bool(completed),
            elapsed_s=float(elapsed_s),
            dt_s=float(dt_s),
            thrust=float(thrust),
            hover_thrust=float(hover_thrust),
            thrust_delta=float(thrust_delta),
            az_m_s2=float(az_m_s2),
            vz_m_s=float(vz_m_s),
            z_rel_m=float(z_rel_m),
            z_hold_error_m=float(z_hold_error_m),
            z_hold_vz_error_m_s=float(z_hold_vz_error_m_s),
            z_hold_thrust_correction=float(z_hold_thrust_correction),
            samples=int(self.samples),
            accel_per_thrust=float(accel_per_thrust),
            thrust_from_acc_gain=float(self.last_gain),
            confidence=float(confidence),
            accel_source=str(accel_source),
            armed=armed,
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

    def _vertical_accel_neu(
        self,
        snapshot,
        state_vz: float,
        now: float,
    ) -> tuple[float, str]:
        acc_body = self._vec3(getattr(snapshot, "accel_xyz", None))
        if acc_body is not None:
            rot_ned_body = body_frd_to_local_ned_rotmat(
                self._finite_float(getattr(snapshot, "roll_rad", None), 0.0),
                self._finite_float(getattr(snapshot, "pitch_rad", None), 0.0),
                self._finite_float(getattr(snapshot, "yaw_rad", None), 0.0),
            )
            acc_ned = rot_ned_body @ acc_body
            acc_ned = acc_ned + np.array([0.0, 0.0, self.gravity_m_s2], dtype=float)
            acc_neu = local_ned_to_neu(acc_ned)
            return float(acc_neu[2]), "imu"

        if self.prev_vz is not None and self.prev_vz_time is not None:
            dt_s = now - self.prev_vz_time
            prev_vz = self.prev_vz
            self.prev_vz = state_vz
            self.prev_vz_time = now
            if math.isfinite(dt_s) and 0.01 <= dt_s <= 0.5:
                return float((state_vz - prev_vz) / dt_s), "velocity"

        self.prev_vz = state_vz
        self.prev_vz_time = now
        return math.nan, "none"

    def _filtered_accel(self, az_raw: float) -> float:
        if not math.isfinite(az_raw):
            return math.nan
        if self.filtered_az is None or not math.isfinite(self.filtered_az):
            self.filtered_az = float(az_raw)
        else:
            alpha = self.accel_filter_alpha
            self.filtered_az = (1.0 - alpha) * self.filtered_az + alpha * float(az_raw)
        return float(self.filtered_az)

    def _valid_gain(self, value: float) -> Optional[float]:
        try:
            gain = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(gain):
            return None
        return _clamp(gain, self.min_gain, self.max_gain)

    def _default_gain(self) -> float:
        gravity = self.gravity_m_s2 if self.gravity_m_s2 > 1e-6 else 9.81
        return _clamp(1.0 / gravity, self.min_gain, self.max_gain)

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
