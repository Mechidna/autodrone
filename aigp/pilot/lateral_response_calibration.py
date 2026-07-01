from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from autonomy_core.controller.attitude_controller3 import normalize, rotmat_to_euler_zyx
from autonomy_core.core.frame_conventions import (
    body_frd_to_local_ned_rotmat,
    local_ned_to_neu,
)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


@dataclass(frozen=True)
class LateralResponseCalibrationCommand:
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    thrust: float


@dataclass(frozen=True)
class LateralResponseCalibrationDebug:
    status: str
    active: bool
    completed: bool
    elapsed_s: float
    dt_s: float
    roll_rad: float
    pitch_rad: float
    thrust: float
    hover_thrust: float
    command_accel_xy_m_s2: tuple[float, float]
    sampled_command_accel_xy_m_s2: tuple[float, float]
    measured_accel_xy_m_s2: tuple[float, float]
    sampled_measured_accel_xy_m_s2: tuple[float, float]
    sampled_axis: int
    sampled_age_s: float
    sampled_response_ratio: float
    sampled_status: str
    response_ratio_xy: tuple[float, float]
    lateral_accel_gain_xy: tuple[float, float]
    samples_xy: tuple[int, int]
    signed_samples_xy: tuple[int, int, int, int]
    z_hold_error_m: float
    z_hold_vz_error_m_s: float
    z_hold_thrust_correction: float
    confidence: float
    accel_source: str
    xy_rel_m: float
    vxy_m_s: float
    z_rel_m: float
    armed: Optional[bool]


@dataclass(frozen=True)
class LateralResponseCalibrationResult:
    command: Optional[LateralResponseCalibrationCommand]
    lateral_accel_gain_xy: Optional[np.ndarray]
    debug: LateralResponseCalibrationDebug


class LateralResponseCalibration:
    """
    Learn XY acceleration response for attitude-thrust control.

    This phase is competition-safe: it uses only attitude/thrust commands,
    state estimate, and IMU/velocity response. It does not use simulator truth,
    known gates, or absolute global position.
    """

    def __init__(self, config):
        section = config.lateral_response_calibration
        self.state_mode = str(config.state_estimation.mode).lower()
        self.gravity_m_s2 = float(config.state_estimation.gravity_m_s2)

        self.enabled = bool(section.enabled)
        self.estimator_mode_only = bool(section.estimator_mode_only)
        self.require_thrust_scale_calibration = bool(
            section.require_thrust_scale_calibration
        )
        self.require_armed = bool(section.require_armed)
        self.initial_delay_s = max(0.0, float(section.initial_delay_s))
        self.min_duration_s = max(0.0, float(section.min_duration_s))
        self.max_duration_s = max(0.0, float(section.max_duration_s))
        self.phase_duration_s = max(0.05, float(section.phase_duration_s))
        self.settle_duration_s = max(0.0, float(section.settle_duration_s))
        self.probe_accel_m_s2 = abs(float(section.probe_accel_m_s2))
        self.min_probe_accel_m_s2 = abs(float(section.min_probe_accel_m_s2))
        self.max_probe_accel_m_s2 = abs(float(section.max_probe_accel_m_s2))
        self.max_tilt_rad = math.radians(max(0.0, float(section.max_tilt_deg)))
        self.min_samples_per_axis = max(1, int(section.min_samples_per_axis))
        self.accel_filter_alpha = _clamp(section.accel_filter_alpha, 0.0, 1.0)
        self.accel_deadband_m_s2 = max(0.0, float(section.accel_deadband_m_s2))
        self.min_abs_accel_m_s2 = max(0.0, float(section.min_abs_accel_m_s2))
        self.max_abs_accel_m_s2 = max(0.0, float(section.max_abs_accel_m_s2))
        self.max_cross_axis_ratio = max(0.0, float(section.max_cross_axis_ratio))
        self.max_abs_vxy_m_s = max(0.0, float(section.max_abs_vxy_m_s))
        self.max_xy_displacement_m = max(0.0, float(section.max_xy_displacement_m))
        self.max_relative_z_m = max(0.0, float(section.max_relative_z_m))
        self.max_sign_ratio_disagreement = max(
            0.0,
            float(section.max_sign_ratio_disagreement),
        )
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
        self.tilt_thrust_compensation = bool(section.tilt_thrust_compensation)

        max_accel_from_tilt = self.gravity_m_s2 * math.tan(self.max_tilt_rad)
        if max_accel_from_tilt > 0.0:
            self.max_probe_accel_m_s2 = min(
                self.max_probe_accel_m_s2,
                max_accel_from_tilt,
            )
        self.probe_accel_m_s2 = _clamp(
            self.probe_accel_m_s2,
            self.min_probe_accel_m_s2,
            self.max_probe_accel_m_s2,
        )

        self.completed = False
        self.start_time: Optional[float] = None
        self.last_update_time: Optional[float] = None
        self.initial_pos: Optional[np.ndarray] = None
        self.command_accel_xy = np.zeros(2, dtype=float)
        self.command_accel_since: Optional[float] = None
        self.filtered_accel_xy: Optional[np.ndarray] = None
        self.prev_vel_xy: Optional[np.ndarray] = None
        self.prev_vel_time: Optional[float] = None
        self.signed_ratio_sums = np.zeros((2, 2), dtype=float)
        self.signed_sample_counts = np.zeros((2, 2), dtype=int)
        self.ratio_sums = np.zeros(2, dtype=float)
        self.sample_counts = np.zeros(2, dtype=int)
        self.last_sample_command_accel_xy = np.zeros(2, dtype=float)
        self.last_sample_measured_accel_xy = np.full(2, math.nan, dtype=float)
        self.last_sample_axis = -1
        self.last_sample_age_s = math.nan
        self.last_sample_response_ratio = math.nan
        self.last_sample_status = "init"
        self.last_gain_xy = np.ones(2, dtype=float)
        self.last_debug = self._debug(
            status="init",
            active=False,
            completed=False,
            elapsed_s=0.0,
            dt_s=0.0,
            roll_rad=0.0,
            pitch_rad=0.0,
            thrust=0.0,
            hover_thrust=0.0,
            measured_accel_xy=np.full(2, math.nan, dtype=float),
            accel_source="none",
            xy_rel_m=0.0,
            vxy_m_s=0.0,
            z_rel_m=0.0,
            armed=None,
        )

    def update(
        self,
        *,
        snapshot,
        estimate,
        hover_thrust: float,
        thrust_scale_calibration_completed: bool,
        current_lateral_accel_gain_xy,
        now: Optional[float] = None,
    ) -> LateralResponseCalibrationResult:
        now = time.monotonic() if now is None else float(now)
        hover = _clamp(self._finite_float(hover_thrust, 0.5), 0.0, 1.0)
        current_gain = self._valid_gain_xy(current_lateral_accel_gain_xy)
        if current_gain is not None:
            self.last_gain_xy = current_gain

        if not self.enabled:
            self.completed = True
            return self._inactive("disabled", hover)
        if self.estimator_mode_only and self.state_mode != "estimator":
            self.completed = True
            return self._inactive("skipped_state_mode", hover)
        if self.completed:
            return self._inactive("complete", hover)
        if self.require_thrust_scale_calibration and not thrust_scale_calibration_completed:
            return self._inactive("waiting_thrust_scale", hover)
        if not bool(getattr(estimate, "valid", False)):
            return self._inactive("invalid_state", hover)

        pos, vel, yaw_rad = self._state_terms(snapshot, estimate)
        armed = self._armed(snapshot)
        if self.require_armed and armed is False:
            self._reset_runtime()
            debug = self._debug(
                status="waiting_armed",
                active=True,
                completed=False,
                elapsed_s=0.0,
                dt_s=0.0,
                roll_rad=0.0,
                pitch_rad=0.0,
                thrust=hover,
                hover_thrust=hover,
                measured_accel_xy=np.full(2, math.nan, dtype=float),
                accel_source="none",
                xy_rel_m=0.0,
                vxy_m_s=float(np.linalg.norm(vel[:2])),
                z_rel_m=0.0,
                armed=armed,
            )
            return LateralResponseCalibrationResult(
                command=LateralResponseCalibrationCommand(0.0, 0.0, yaw_rad, hover),
                lateral_accel_gain_xy=None,
                debug=debug,
            )

        if self.start_time is None:
            self.start_time = now
            self.last_update_time = now
            self.initial_pos = pos.copy()
            self.command_accel_xy = np.zeros(2, dtype=float)
            self.command_accel_since = now
            self.filtered_accel_xy = None
            self.prev_vel_xy = vel[:2].copy()
            self.prev_vel_time = now
            elapsed_s = 0.0
            dt_s = 0.0
        else:
            elapsed_s = max(0.0, now - self.start_time)
            dt_s = now - float(self.last_update_time or now)
            if not math.isfinite(dt_s) or dt_s < 0.0 or dt_s > 1.0:
                dt_s = 0.0
            self.last_update_time = now

        initial = self.initial_pos if self.initial_pos is not None else pos
        rel = pos - initial
        xy_rel_m = float(np.linalg.norm(rel[:2]))
        z_rel_m = float(rel[2])
        vxy_m_s = float(np.linalg.norm(vel[:2]))
        z_hold_error_m = float(-z_rel_m)
        z_hold_vz_error_m_s = float(-vel[2])
        z_hold_thrust_correction = self._z_hold_thrust_correction(
            z_error_m=z_hold_error_m,
            vz_error_m_s=z_hold_vz_error_m_s,
        )
        accel_raw, accel_source = self._accel_xy_neu(snapshot, vel[:2], now)
        measured_accel_xy = self._filtered_accel_xy(accel_raw)
        self._record_sample(
            now=now,
            command_accel_xy=self.command_accel_xy,
            measured_accel_xy=measured_accel_xy,
            vxy_m_s=vxy_m_s,
        )

        response_ratio = self._response_ratio_xy()
        enough_samples = self._has_enough_signed_samples()
        estimated_gain = self._gain_from_ratio(response_ratio) if enough_samples else None
        if estimated_gain is not None:
            self.last_gain_xy = self._blend_gain(self.last_gain_xy, estimated_gain)

        motion_limited = self._motion_limited(
            xy_rel_m=xy_rel_m,
            vxy_m_s=vxy_m_s,
            z_rel_m=z_rel_m,
        )
        timed_out = self.max_duration_s > 0.0 and elapsed_s >= self.max_duration_s
        valid_calibration = (
            enough_samples
            and estimated_gain is not None
            and elapsed_s >= self.min_duration_s
        )
        if motion_limited or timed_out or valid_calibration:
            self.completed = True
            if valid_calibration:
                status = "calibrated"
            elif enough_samples and estimated_gain is not None:
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
                roll_rad=0.0,
                pitch_rad=0.0,
                thrust=hover,
                hover_thrust=hover,
                measured_accel_xy=measured_accel_xy,
                accel_source=accel_source,
                xy_rel_m=xy_rel_m,
                vxy_m_s=vxy_m_s,
                z_rel_m=z_rel_m,
                z_hold_error_m=z_hold_error_m,
                z_hold_vz_error_m_s=z_hold_vz_error_m_s,
                z_hold_thrust_correction=z_hold_thrust_correction,
                armed=armed,
            )
            return LateralResponseCalibrationResult(
                command=None,
                lateral_accel_gain_xy=self.last_gain_xy.copy(),
                debug=debug,
            )

        desired_accel_xy = self._desired_probe_accel(elapsed_s)
        if np.linalg.norm(desired_accel_xy - self.command_accel_xy) > 1e-6:
            self.command_accel_xy = desired_accel_xy
            self.command_accel_since = now

        roll_rad, pitch_rad, thrust = self._command_from_accel(
            command_accel_xy=self.command_accel_xy,
            yaw_rad=yaw_rad,
            hover_thrust=hover,
            thrust_correction=z_hold_thrust_correction,
        )
        status = "delay" if elapsed_s < self.initial_delay_s else "probing"
        debug = self._debug(
            status=status,
            active=True,
            completed=False,
            elapsed_s=elapsed_s,
            dt_s=dt_s,
            roll_rad=roll_rad,
            pitch_rad=pitch_rad,
            thrust=thrust,
            hover_thrust=hover,
            measured_accel_xy=measured_accel_xy,
            accel_source=accel_source,
            xy_rel_m=xy_rel_m,
            vxy_m_s=vxy_m_s,
            z_rel_m=z_rel_m,
            z_hold_error_m=z_hold_error_m,
            z_hold_vz_error_m_s=z_hold_vz_error_m_s,
            z_hold_thrust_correction=z_hold_thrust_correction,
            armed=armed,
        )
        return LateralResponseCalibrationResult(
            command=LateralResponseCalibrationCommand(
                roll_rad=roll_rad,
                pitch_rad=pitch_rad,
                yaw_rad=yaw_rad,
                thrust=thrust,
            ),
            lateral_accel_gain_xy=None,
            debug=debug,
        )

    def _desired_probe_accel(self, elapsed_s: float) -> np.ndarray:
        if elapsed_s < self.initial_delay_s:
            return np.zeros(2, dtype=float)
        phase = int((elapsed_s - self.initial_delay_s) / self.phase_duration_s) % 4
        accel = self.probe_accel_m_s2
        if phase == 0:
            return np.array([accel, 0.0], dtype=float)
        if phase == 1:
            return np.array([-accel, 0.0], dtype=float)
        if phase == 2:
            return np.array([0.0, accel], dtype=float)
        return np.array([0.0, -accel], dtype=float)

    def _command_from_accel(
        self,
        *,
        command_accel_xy: np.ndarray,
        yaw_rad: float,
        hover_thrust: float,
        thrust_correction: float = 0.0,
    ) -> tuple[float, float, float]:
        a_des = np.array(
            [
                float(command_accel_xy[0]),
                float(command_accel_xy[1]),
                self.gravity_m_s2,
            ],
            dtype=float,
        )
        z_b_des = normalize(a_des)
        if z_b_des is None:
            z_b_des = np.array([0.0, 0.0, 1.0], dtype=float)

        x_c = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0], dtype=float)
        y_c = np.array([-math.sin(yaw_rad), math.cos(yaw_rad), 0.0], dtype=float)
        x_b_des = normalize(np.cross(y_c, z_b_des))
        if x_b_des is None:
            x_b_des = x_c
        y_b_des = normalize(np.cross(z_b_des, x_b_des))
        if y_b_des is None:
            y_b_des = y_c

        r_des = np.column_stack((x_b_des, y_b_des, z_b_des))
        roll_des, pitch_des, _ = rotmat_to_euler_zyx(r_des)

        roll_cmd = -_clamp(roll_des, -self.max_tilt_rad, self.max_tilt_rad)
        pitch_cmd = -_clamp(pitch_des, -self.max_tilt_rad, self.max_tilt_rad)
        thrust = float(hover_thrust)
        if self.tilt_thrust_compensation:
            tilt_norm = min(self.max_tilt_rad, math.hypot(roll_cmd, pitch_cmd))
            thrust = thrust / max(0.85, math.cos(tilt_norm))
        thrust += self._finite_float(thrust_correction, 0.0)
        return roll_cmd, pitch_cmd, _clamp(thrust, 0.0, 1.0)

    def _z_hold_thrust_correction(
        self,
        *,
        z_error_m: float,
        vz_error_m_s: float,
    ) -> float:
        if not self.z_hold_enabled or self.z_hold_max_correction <= 0.0:
            return 0.0
        correction = self.z_hold_kp * float(z_error_m)
        correction += self.z_hold_kv * float(vz_error_m_s)
        if not math.isfinite(correction):
            return 0.0
        return _clamp(
            correction,
            -self.z_hold_max_correction,
            self.z_hold_max_correction,
        )

    def _record_sample(
        self,
        *,
        now: float,
        command_accel_xy: np.ndarray,
        measured_accel_xy: np.ndarray,
        vxy_m_s: float,
    ) -> None:
        self.last_sample_command_accel_xy = np.asarray(
            command_accel_xy,
            dtype=float,
        ).reshape(2).copy()
        self.last_sample_measured_accel_xy = np.asarray(
            measured_accel_xy,
            dtype=float,
        ).reshape(2).copy()
        self.last_sample_axis = -1
        self.last_sample_age_s = math.nan
        self.last_sample_response_ratio = math.nan
        self.last_sample_status = "not_evaluated"

        if self.command_accel_since is None:
            self.last_sample_status = "no_command_time"
            return
        sample_age_s = now - self.command_accel_since
        self.last_sample_age_s = float(sample_age_s)
        if sample_age_s < self.settle_duration_s:
            self.last_sample_status = "settling"
            return
        if not np.all(np.isfinite(measured_accel_xy)):
            self.last_sample_status = "invalid_accel"
            return
        if self.max_abs_vxy_m_s > 0.0 and abs(vxy_m_s) > self.max_abs_vxy_m_s:
            self.last_sample_status = "high_vxy"
            return
        if self.max_abs_accel_m_s2 > 0.0:
            if float(np.linalg.norm(measured_accel_xy)) > self.max_abs_accel_m_s2:
                self.last_sample_status = "high_accel"
                return

        axis = int(np.argmax(np.abs(command_accel_xy)))
        self.last_sample_axis = axis
        command_mag = abs(float(command_accel_xy[axis]))
        if command_mag < self.min_probe_accel_m_s2:
            self.last_sample_status = "probe_small"
            return
        sign = 1.0 if command_accel_xy[axis] >= 0.0 else -1.0
        along = sign * float(measured_accel_xy[axis])
        cross = float(measured_accel_xy[1 - axis])
        if abs(along) <= self.accel_deadband_m_s2:
            self.last_sample_status = "deadband"
            return
        if along < self.min_abs_accel_m_s2:
            self.last_sample_status = "wrong_sign_or_weak"
            return
        if (
            self.max_cross_axis_ratio > 0.0
            and abs(cross) > self.max_cross_axis_ratio * max(abs(along), 1e-6)
        ):
            self.last_sample_status = "cross_axis"
            return

        response_ratio = along / command_mag
        self.last_sample_response_ratio = float(response_ratio)
        self.last_sample_status = "accepted"
        sign_index = 1 if command_accel_xy[axis] >= 0.0 else 0
        self.signed_ratio_sums[axis, sign_index] += response_ratio
        self.signed_sample_counts[axis, sign_index] += 1
        self.ratio_sums[axis] += response_ratio
        self.sample_counts[axis] += 1

    def _response_ratio_xy(self) -> np.ndarray:
        ratio = np.full(2, math.nan, dtype=float)
        signed_means = self._signed_response_means()
        for axis in range(2):
            neg = float(signed_means[axis, 0])
            pos = float(signed_means[axis, 1])
            if not math.isfinite(neg) or not math.isfinite(pos):
                continue
            if neg <= 1e-6 or pos <= 1e-6:
                continue
            mean = 0.5 * (neg + pos)
            rel_disagreement = abs(pos - neg) / max(mean, 1e-6)
            if rel_disagreement > self.max_sign_ratio_disagreement:
                continue
            ratio[axis] = mean
        return ratio

    def _signed_response_means(self) -> np.ndarray:
        means = np.full((2, 2), math.nan, dtype=float)
        for axis in range(2):
            for sign_index in range(2):
                count = int(self.signed_sample_counts[axis, sign_index])
                if count >= self.min_samples_per_axis:
                    means[axis, sign_index] = (
                        self.signed_ratio_sums[axis, sign_index] / float(count)
                    )
        return means

    def _has_enough_signed_samples(self) -> bool:
        return bool(np.all(self.signed_sample_counts >= self.min_samples_per_axis))

    def _gain_from_ratio(self, response_ratio_xy: np.ndarray) -> Optional[np.ndarray]:
        if not np.all(np.isfinite(response_ratio_xy)):
            return None
        if np.any(response_ratio_xy <= 1e-6):
            return None
        return np.clip(1.0 / response_ratio_xy, self.min_gain, self.max_gain)

    def _blend_gain(self, current_gain: np.ndarray, estimated_gain: np.ndarray) -> np.ndarray:
        alpha = self.result_alpha
        return np.clip(
            (1.0 - alpha) * current_gain + alpha * estimated_gain,
            self.min_gain,
            self.max_gain,
        )

    def _motion_limited(self, *, xy_rel_m: float, vxy_m_s: float, z_rel_m: float) -> bool:
        if self.max_xy_displacement_m > 0.0 and xy_rel_m > self.max_xy_displacement_m:
            return True
        if self.max_abs_vxy_m_s > 0.0 and vxy_m_s > self.max_abs_vxy_m_s:
            return True
        if self.max_relative_z_m > 0.0 and abs(z_rel_m) > self.max_relative_z_m:
            return True
        return False

    def _inactive(self, status: str, hover_thrust: float) -> LateralResponseCalibrationResult:
        debug = self._debug(
            status=status,
            active=False,
            completed=self.completed,
            elapsed_s=0.0,
            dt_s=0.0,
            roll_rad=0.0,
            pitch_rad=0.0,
            thrust=hover_thrust,
            hover_thrust=hover_thrust,
            measured_accel_xy=np.full(2, math.nan, dtype=float),
            accel_source="none",
            xy_rel_m=0.0,
            vxy_m_s=0.0,
            z_rel_m=0.0,
            armed=None,
        )
        return LateralResponseCalibrationResult(
            command=None,
            lateral_accel_gain_xy=None,
            debug=debug,
        )

    def _reset_runtime(self) -> None:
        self.start_time = None
        self.last_update_time = None
        self.initial_pos = None
        self.command_accel_xy = np.zeros(2, dtype=float)
        self.command_accel_since = None
        self.filtered_accel_xy = None
        self.prev_vel_xy = None
        self.prev_vel_time = None
        self.signed_ratio_sums[:, :] = 0.0
        self.signed_sample_counts[:, :] = 0
        self.ratio_sums[:] = 0.0
        self.sample_counts[:] = 0
        self.last_sample_command_accel_xy = np.zeros(2, dtype=float)
        self.last_sample_measured_accel_xy = np.full(2, math.nan, dtype=float)
        self.last_sample_axis = -1
        self.last_sample_age_s = math.nan
        self.last_sample_response_ratio = math.nan
        self.last_sample_status = "reset"

    def _debug(
        self,
        *,
        status: str,
        active: bool,
        completed: bool,
        elapsed_s: float,
        dt_s: float,
        roll_rad: float,
        pitch_rad: float,
        thrust: float,
        hover_thrust: float,
        measured_accel_xy: np.ndarray,
        accel_source: str,
        xy_rel_m: float,
        vxy_m_s: float,
        z_rel_m: float,
        armed: Optional[bool],
        z_hold_error_m: float = 0.0,
        z_hold_vz_error_m_s: float = 0.0,
        z_hold_thrust_correction: float = 0.0,
    ) -> LateralResponseCalibrationDebug:
        ratio = self._response_ratio_xy()
        required_samples = 4 * float(self.min_samples_per_axis)
        confidence = min(
            1.0,
            float(np.sum(self.signed_sample_counts)) / required_samples,
        )
        self.last_debug = LateralResponseCalibrationDebug(
            status=str(status),
            active=bool(active),
            completed=bool(completed),
            elapsed_s=float(elapsed_s),
            dt_s=float(dt_s),
            roll_rad=float(roll_rad),
            pitch_rad=float(pitch_rad),
            thrust=float(thrust),
            hover_thrust=float(hover_thrust),
            command_accel_xy_m_s2=(
                float(self.command_accel_xy[0]),
                float(self.command_accel_xy[1]),
            ),
            sampled_command_accel_xy_m_s2=(
                float(self.last_sample_command_accel_xy[0]),
                float(self.last_sample_command_accel_xy[1]),
            ),
            measured_accel_xy_m_s2=(
                float(measured_accel_xy[0]),
                float(measured_accel_xy[1]),
            ),
            sampled_measured_accel_xy_m_s2=(
                float(self.last_sample_measured_accel_xy[0]),
                float(self.last_sample_measured_accel_xy[1]),
            ),
            sampled_axis=int(self.last_sample_axis),
            sampled_age_s=float(self.last_sample_age_s),
            sampled_response_ratio=float(self.last_sample_response_ratio),
            sampled_status=str(self.last_sample_status),
            response_ratio_xy=(float(ratio[0]), float(ratio[1])),
            lateral_accel_gain_xy=(
                float(self.last_gain_xy[0]),
                float(self.last_gain_xy[1]),
            ),
            samples_xy=(int(self.sample_counts[0]), int(self.sample_counts[1])),
            signed_samples_xy=(
                int(self.signed_sample_counts[0, 0]),
                int(self.signed_sample_counts[0, 1]),
                int(self.signed_sample_counts[1, 0]),
                int(self.signed_sample_counts[1, 1]),
            ),
            z_hold_error_m=float(z_hold_error_m),
            z_hold_vz_error_m_s=float(z_hold_vz_error_m_s),
            z_hold_thrust_correction=float(z_hold_thrust_correction),
            confidence=float(confidence),
            accel_source=str(accel_source),
            xy_rel_m=float(xy_rel_m),
            vxy_m_s=float(vxy_m_s),
            z_rel_m=float(z_rel_m),
            armed=armed,
        )
        return self.last_debug

    def _state_terms(self, snapshot, estimate) -> tuple[np.ndarray, np.ndarray, float]:
        pos = np.asarray(getattr(estimate, "pos_neu"), dtype=float).reshape(3)
        vel = np.asarray(getattr(estimate, "vel_neu"), dtype=float).reshape(3)
        yaw_rad = self._finite_float(
            getattr(estimate, "yaw_rad", None),
            self._finite_float(getattr(snapshot, "yaw_rad", None), 0.0),
        )
        return pos.copy(), vel.copy(), float(yaw_rad)

    def _accel_xy_neu(
        self,
        snapshot,
        vel_xy: np.ndarray,
        now: float,
    ) -> tuple[np.ndarray, str]:
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
            return np.asarray(acc_neu[:2], dtype=float), "imu"

        if self.prev_vel_xy is not None and self.prev_vel_time is not None:
            dt_s = now - self.prev_vel_time
            prev_vel = self.prev_vel_xy.copy()
            self.prev_vel_xy = np.asarray(vel_xy, dtype=float).reshape(2).copy()
            self.prev_vel_time = now
            if math.isfinite(dt_s) and 0.01 <= dt_s <= 0.5:
                return (self.prev_vel_xy - prev_vel) / dt_s, "velocity"

        self.prev_vel_xy = np.asarray(vel_xy, dtype=float).reshape(2).copy()
        self.prev_vel_time = now
        return np.full(2, math.nan, dtype=float), "none"

    def _filtered_accel_xy(self, accel_raw: np.ndarray) -> np.ndarray:
        accel = np.asarray(accel_raw, dtype=float).reshape(2)
        if not np.all(np.isfinite(accel)):
            return np.full(2, math.nan, dtype=float)
        if self.filtered_accel_xy is None or not np.all(np.isfinite(self.filtered_accel_xy)):
            self.filtered_accel_xy = accel.copy()
        else:
            alpha = self.accel_filter_alpha
            self.filtered_accel_xy = (1.0 - alpha) * self.filtered_accel_xy + alpha * accel
        return self.filtered_accel_xy.copy()

    def _valid_gain_xy(self, value) -> Optional[np.ndarray]:
        try:
            arr = np.asarray(value, dtype=float).reshape(2)
        except (TypeError, ValueError):
            return None
        if not np.all(np.isfinite(arr)):
            return None
        return np.clip(arr, self.min_gain, self.max_gain)

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
