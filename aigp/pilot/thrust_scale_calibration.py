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
    succeeded: bool
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
    positive_samples: int
    negative_samples: int
    accel_per_thrust: float
    positive_accel_per_thrust: float
    negative_accel_per_thrust: float
    sign_slope_disagreement: float
    neutral_accel_m_s2: float
    thrust_from_acc_gain: float
    confidence: float
    accel_source: str
    armed: Optional[bool]


@dataclass(frozen=True)
class ThrustScaleCalibrationResult:
    command: Optional[ThrustScaleCalibrationCommand]
    thrust_from_acc_gain: Optional[float]
    succeeded: bool
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
        self.min_samples_per_sign = max(1, int(section.min_samples_per_sign))
        self.neutral_max_abs_vz_m_s = max(
            0.0,
            float(section.neutral_max_abs_vz_m_s),
        )
        self.neutral_max_abs_accel_m_s2 = max(
            0.0,
            float(section.neutral_max_abs_accel_m_s2),
        )
        self.max_sign_slope_disagreement = max(
            0.0,
            float(section.max_sign_slope_disagreement),
        )
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
        self.succeeded = False
        self.start_time: Optional[float] = None
        self.probe_start_time: Optional[float] = None
        self.neutral_stable_since: Optional[float] = None
        self.last_update_time: Optional[float] = None
        self.initial_z: Optional[float] = None
        self.command_delta = 0.0
        self.command_delta_since: Optional[float] = None
        self.filtered_az: Optional[float] = None
        self.prev_vz: Optional[float] = None
        self.prev_vz_time: Optional[float] = None
        self.neutral_accel_sum = 0.0
        self.neutral_accel_samples = 0
        self.neutral_accel_m_s2 = math.nan
        self.signed_delta_sums = np.zeros(2, dtype=float)
        self.signed_accel_sums = np.zeros(2, dtype=float)
        self.signed_sample_counts = np.zeros(2, dtype=int)
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
        if current_gain is not None and not self.succeeded:
            self.last_gain = current_gain

        if not self.enabled:
            self.completed = True
            self.succeeded = False
            return self._inactive("disabled", hover)
        if self.estimator_mode_only and self.state_mode != "estimator":
            self.completed = True
            self.succeeded = False
            return self._inactive("skipped_state_mode", hover)
        if self.completed:
            return self._inactive("complete" if self.succeeded else "failed", hover)
        if self.require_hover_acquisition and not hover_acquisition_completed:
            return self._inactive("waiting_hover_acquisition", hover)
        if not bool(getattr(estimate, "valid", False)):
            return self._inactive("invalid_state", hover)

        state_z, state_vz, yaw_rad = self._state_terms(snapshot, estimate)
        armed = self._armed(snapshot)
        if self.require_armed and armed is not True:
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
                succeeded=False,
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

        motion_limited = self._motion_limited(z_rel_m=z_rel_m, vz_m_s=state_vz)
        timed_out = self.max_duration_s > 0.0 and elapsed_s >= self.max_duration_s
        if motion_limited or timed_out:
            self.completed = True
            self.succeeded = False
            status = "motion_limited_fallback" if motion_limited else "timeout_fallback"
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
                thrust_from_acc_gain=None,
                succeeded=False,
                debug=debug,
            )

        if self.probe_start_time is None:
            neutral_stable = self._neutral_is_stable(
                az_m_s2=az_m_s2,
                vz_m_s=state_vz,
                z_hold_thrust_correction=z_hold_thrust_correction,
            )
            if neutral_stable:
                if self.neutral_stable_since is None:
                    self.neutral_stable_since = now
                    self.neutral_accel_sum = 0.0
                    self.neutral_accel_samples = 0
                self.neutral_accel_sum += az_m_s2
                self.neutral_accel_samples += 1
            else:
                self.neutral_stable_since = None
                self.neutral_accel_sum = 0.0
                self.neutral_accel_samples = 0

            stable_elapsed_s = (
                max(0.0, now - self.neutral_stable_since)
                if self.neutral_stable_since is not None
                else 0.0
            )
            dwell_complete = (
                neutral_stable
                and self.neutral_accel_samples > 0
                and stable_elapsed_s + 1e-9 >= self.initial_delay_s
            )
            if dwell_complete:
                self.neutral_accel_m_s2 = (
                    self.neutral_accel_sum / float(self.neutral_accel_samples)
                )
                self.probe_start_time = now
                self.initial_z = state_z
                z_rel_m = 0.0
                self.command_delta = 0.0
                self.command_delta_since = now
            else:
                self.command_delta = 0.0
                thrust = _clamp(hover + z_hold_thrust_correction, 0.0, 1.0)
                status = "neutral_settle" if neutral_stable else "waiting_neutral"
                debug = self._debug(
                    status=status,
                    active=True,
                    completed=False,
                    elapsed_s=elapsed_s,
                    dt_s=dt_s,
                    thrust=thrust,
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
                    command=ThrustScaleCalibrationCommand(
                        roll_rad=0.0,
                        pitch_rad=0.0,
                        yaw_rad=yaw_rad,
                        thrust=thrust,
                    ),
                    thrust_from_acc_gain=None,
                    succeeded=False,
                    debug=debug,
                )

        probe_elapsed_s = max(0.0, now - float(self.probe_start_time))
        sample_az_m_s2 = az_raw if math.isfinite(az_raw) else az_m_s2
        self._record_sample(
            now=now,
            thrust_delta=self.command_delta,
            az_m_s2=sample_az_m_s2,
            vz_m_s=state_vz,
            z_hold_thrust_correction=z_hold_thrust_correction,
        )
        estimated_gain = self._validated_gain()
        valid_calibration = (
            probe_elapsed_s >= self.min_duration_s and estimated_gain is not None
        )
        if valid_calibration:
            self.last_gain = self._blend_gain(self.last_gain, estimated_gain)
            self.completed = True
            self.succeeded = True
            debug = self._debug(
                status="calibrated",
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
                succeeded=True,
                debug=debug,
            )

        desired_delta = self._desired_probe_delta(probe_elapsed_s)
        if abs(desired_delta - self.command_delta) > 1e-6:
            self.command_delta = desired_delta
            self.command_delta_since = now

        thrust = _clamp(hover + self.command_delta + z_hold_thrust_correction, 0.0, 1.0)
        actual_delta = thrust - hover
        expected_delta = self.command_delta + z_hold_thrust_correction
        if abs(actual_delta - expected_delta) > 1e-6:
            self.command_delta = actual_delta
            self.command_delta_since = now

        positive_samples = int(self.signed_sample_counts[1])
        negative_samples = int(self.signed_sample_counts[0])
        status = (
            "probing_unbalanced"
            if self.samples >= self.min_samples
            and (
                positive_samples < self.min_samples_per_sign
                or negative_samples < self.min_samples_per_sign
            )
            else "probing"
        )
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
            succeeded=False,
            debug=debug,
        )

    def _desired_probe_delta(self, elapsed_s: float) -> float:
        phase = int(elapsed_s / self.phase_duration_s)
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

        sign_index = 1 if thrust_delta > 0.0 else 0
        self.signed_delta_sums[sign_index] += thrust_delta
        self.signed_accel_sums[sign_index] += az_m_s2
        self.signed_sample_counts[sign_index] += 1
        self.samples += 1

    def _accel_per_thrust(self) -> float:
        if np.any(self.signed_sample_counts <= 0):
            return math.nan
        mean_delta = self.signed_delta_sums / self.signed_sample_counts
        mean_accel = self.signed_accel_sums / self.signed_sample_counts
        delta_span = float(mean_delta[1] - mean_delta[0])
        if delta_span <= 1e-9:
            return math.nan
        return float((mean_accel[1] - mean_accel[0]) / delta_span)

    def _sign_slopes(self) -> tuple[float, float, float]:
        if (
            np.any(self.signed_sample_counts <= 0)
            or not math.isfinite(self.neutral_accel_m_s2)
        ):
            return math.nan, math.nan, math.nan

        mean_delta = self.signed_delta_sums / self.signed_sample_counts
        mean_accel = self.signed_accel_sums / self.signed_sample_counts
        positive_delta = float(mean_delta[1])
        negative_delta = float(mean_delta[0])
        if positive_delta <= 1e-9 or negative_delta >= -1e-9:
            return math.nan, math.nan, math.nan

        positive_response = float(mean_accel[1] - self.neutral_accel_m_s2)
        negative_response = float(self.neutral_accel_m_s2 - mean_accel[0])
        if abs(positive_response) <= self.accel_deadband_m_s2:
            positive_response = 0.0
        if abs(negative_response) <= self.accel_deadband_m_s2:
            negative_response = 0.0

        positive_slope = positive_response / positive_delta
        negative_slope = negative_response / (-negative_delta)
        scale = max(
            0.5 * (abs(positive_slope) + abs(negative_slope)),
            1e-9,
        )
        disagreement = abs(positive_slope - negative_slope) / scale
        return positive_slope, negative_slope, disagreement

    def _validated_gain(self) -> Optional[float]:
        positive_count = int(self.signed_sample_counts[1])
        negative_count = int(self.signed_sample_counts[0])
        if (
            self.samples < self.min_samples
            or positive_count < self.min_samples_per_sign
            or negative_count < self.min_samples_per_sign
        ):
            return None

        positive_slope, negative_slope, disagreement = self._sign_slopes()
        if (
            not math.isfinite(positive_slope)
            or not math.isfinite(negative_slope)
            or positive_slope <= 0.0
            or negative_slope <= 0.0
            or not math.isfinite(disagreement)
            or disagreement > self.max_sign_slope_disagreement
        ):
            return None

        minimum_response = max(
            self.min_abs_accel_m_s2,
            self.accel_deadband_m_s2,
        )
        if (
            positive_slope * self.probe_delta_thrust < minimum_response
            or negative_slope * self.probe_delta_thrust < minimum_response
        ):
            return None

        return self._gain_from_slope(self._accel_per_thrust())

    def _neutral_is_stable(
        self,
        *,
        az_m_s2: float,
        vz_m_s: float,
        z_hold_thrust_correction: float,
    ) -> bool:
        if not math.isfinite(az_m_s2) or not math.isfinite(vz_m_s):
            return False
        if abs(z_hold_thrust_correction) > 1e-3:
            return False
        if (
            self.neutral_max_abs_vz_m_s > 0.0
            and abs(vz_m_s) > self.neutral_max_abs_vz_m_s
        ):
            return False
        if (
            self.neutral_max_abs_accel_m_s2 > 0.0
            and abs(az_m_s2) > self.neutral_max_abs_accel_m_s2
        ):
            return False
        return True

    def _gain_from_slope(self, accel_per_thrust: float) -> Optional[float]:
        if not math.isfinite(accel_per_thrust) or accel_per_thrust <= 1e-6:
            return None
        gain = 1.0 / accel_per_thrust
        if not math.isfinite(gain):
            return None
        if gain < self.min_gain or gain > self.max_gain:
            return None
        return float(gain)

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
            succeeded=self.succeeded,
            debug=debug,
        )

    def _reset_runtime(self) -> None:
        self.start_time = None
        self.probe_start_time = None
        self.neutral_stable_since = None
        self.last_update_time = None
        self.initial_z = None
        self.command_delta = 0.0
        self.command_delta_since = None
        self.filtered_az = None
        self.prev_vz = None
        self.prev_vz_time = None
        self.neutral_accel_sum = 0.0
        self.neutral_accel_samples = 0
        self.neutral_accel_m_s2 = math.nan
        self.signed_delta_sums.fill(0.0)
        self.signed_accel_sums.fill(0.0)
        self.signed_sample_counts.fill(0)
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
        positive_samples = int(self.signed_sample_counts[1])
        negative_samples = int(self.signed_sample_counts[0])
        positive_slope, negative_slope, disagreement = self._sign_slopes()
        confidence = min(
            1.0,
            float(self.samples) / float(self.min_samples),
            float(positive_samples) / float(self.min_samples_per_sign),
            float(negative_samples) / float(self.min_samples_per_sign),
        )
        self.last_debug = ThrustScaleCalibrationDebug(
            status=str(status),
            active=bool(active),
            completed=bool(completed),
            succeeded=bool(self.succeeded),
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
            positive_samples=positive_samples,
            negative_samples=negative_samples,
            accel_per_thrust=float(accel_per_thrust),
            positive_accel_per_thrust=float(positive_slope),
            negative_accel_per_thrust=float(negative_slope),
            sign_slope_disagreement=float(disagreement),
            neutral_accel_m_s2=float(self.neutral_accel_m_s2),
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
