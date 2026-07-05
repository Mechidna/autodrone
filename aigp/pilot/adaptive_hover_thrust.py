from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


@dataclass(frozen=True)
class AdaptiveHoverDebug:
    value: float
    status: str
    signal: float = 0.0
    z_error: float = 0.0
    vz_error: float = 0.0
    dt: float = 0.0
    gain: float = 0.0
    fast_weight: float = 0.0


class AdaptiveHoverThrust:
    """
    Slow hover-thrust estimator for normalized attitude-thrust commands.

    The estimate is adjusted from vertical position/velocity tracking error.
    It intentionally uses the same state and reference that the controller uses;
    it does not read MAVLink local position or odometry directly.
    """

    def __init__(
        self,
        *,
        initial_thrust: float = 0.5,
        enabled: bool = True,
        gain: float = 0.04,
        z_gain: float = 0.30,
        min_value: float = 0.0,
        max_value: float = 1.0,
        max_signal: float = 1.0,
        max_ref_vz: float = 1.0,
        max_ref_az: float = 1.5,
        max_z_error: float = 2.0,
        saturation_margin: float = 0.03,
        min_confidence: float = 0.0,
        fast_enabled: bool = False,
        fast_gain: float = 0.25,
        fast_min_z_error: float = 0.05,
        fast_stable_signal: float = 0.08,
        fast_stable_z_error: float = 0.15,
        fast_stable_samples: int = 20,
        fast_decay_s: float = 3.0,
    ):
        self.enabled = bool(enabled)
        self.gain = max(0.0, float(gain))
        self.z_gain = float(z_gain)
        self.min_value = _clamp(min_value, 0.0, 1.0)
        self.max_value = _clamp(max_value, 0.0, 1.0)
        if self.max_value < self.min_value:
            self.min_value, self.max_value = self.max_value, self.min_value
        self.max_signal = max(0.0, float(max_signal))
        self.max_ref_vz = max(0.0, float(max_ref_vz))
        self.max_ref_az = max(0.0, float(max_ref_az))
        self.max_z_error = max(0.0, float(max_z_error))
        self.saturation_margin = _clamp(saturation_margin, 0.0, 0.49)
        self.min_confidence = _clamp(min_confidence, 0.0, 1.0)
        self.fast_enabled = bool(fast_enabled)
        self.fast_gain = max(self.gain, float(fast_gain))
        self.fast_min_z_error = max(0.0, float(fast_min_z_error))
        self.fast_stable_signal = max(0.0, float(fast_stable_signal))
        self.fast_stable_z_error = max(0.0, float(fast_stable_z_error))
        self.fast_stable_samples = max(1, int(fast_stable_samples))
        self.fast_decay_s = max(0.0, float(fast_decay_s))
        self.fast_weight = 1.0 if self.fast_enabled else 0.0
        self.stable_sample_count = 0

        self.value = _clamp(initial_thrust, self.min_value, self.max_value)
        self.last_update_time: float | None = None
        self.last_debug = AdaptiveHoverDebug(value=self.value, status="init")

    def set_value(self, value: float, *, status: str = "external") -> AdaptiveHoverDebug:
        self.value = _clamp(value, self.min_value, self.max_value)
        return self._debug(status)

    def update(
        self,
        *,
        state,
        ref,
        thrust_cmd: float,
        estimator_valid: bool = True,
        estimator_confidence: float = 1.0,
        now: float | None = None,
    ) -> AdaptiveHoverDebug:
        now = time.monotonic() if now is None else float(now)
        if self.last_update_time is None:
            self.last_update_time = now
            return self._debug("first_sample")

        dt = now - self.last_update_time
        self.last_update_time = now
        if not math.isfinite(dt) or dt <= 0.0 or dt > 1.0:
            return self._debug("bad_dt", dt=max(0.0, dt if math.isfinite(dt) else 0.0))

        if not self.enabled:
            return self._debug("disabled", dt=dt)
        if self.gain <= 0.0:
            return self._debug("zero_gain", dt=dt)
        if not bool(estimator_valid):
            return self._debug("invalid_state", dt=dt)
        if float(estimator_confidence) < self.min_confidence:
            return self._debug("low_confidence", dt=dt)

        state_z, state_vz, ref_z, ref_vz, ref_az = self._vertical_terms(state, ref)
        if not all(math.isfinite(v) for v in (state_z, state_vz, ref_z, ref_vz, ref_az)):
            return self._debug("nonfinite", dt=dt)

        z_error = ref_z - state_z
        vz_error = ref_vz - state_vz
        if self.max_z_error > 0.0 and abs(z_error) > self.max_z_error:
            return self._debug("large_z_error", z_error=z_error, vz_error=vz_error, dt=dt)
        if self.max_ref_vz > 0.0 and abs(ref_vz) > self.max_ref_vz:
            return self._debug("moving_reference", z_error=z_error, vz_error=vz_error, dt=dt)
        if self.max_ref_az > 0.0 and abs(ref_az) > self.max_ref_az:
            return self._debug("accelerating_reference", z_error=z_error, vz_error=vz_error, dt=dt)

        thrust = float(thrust_cmd)
        if not math.isfinite(thrust):
            return self._debug("bad_thrust", z_error=z_error, vz_error=vz_error, dt=dt)
        if (
            thrust <= self.saturation_margin
            or thrust >= 1.0 - self.saturation_margin
        ):
            return self._debug("saturated", z_error=z_error, vz_error=vz_error, dt=dt)

        signal = vz_error + self.z_gain * z_error
        signal = _clamp(signal, -self.max_signal, self.max_signal)
        gain = self._gain_for_sample(
            signal=signal,
            z_error=z_error,
            ref_vz=ref_vz,
            dt=dt,
        )
        self.value = _clamp(
            self.value + gain * signal * dt,
            self.min_value,
            self.max_value,
        )
        status = "fast" if gain > self.gain else "active"
        return self._debug(
            status,
            signal=signal,
            z_error=z_error,
            vz_error=vz_error,
            dt=dt,
            gain=gain,
        )

    def _gain_for_sample(
        self,
        *,
        signal: float,
        z_error: float,
        ref_vz: float,
        dt: float,
    ) -> float:
        stable = (
            abs(float(signal)) <= self.fast_stable_signal
            and abs(float(z_error)) <= self.fast_stable_z_error
        )
        if stable:
            self.stable_sample_count += 1
        else:
            self.stable_sample_count = 0

        if (
            self.fast_enabled
            and self.fast_weight > 0.0
            and self.stable_sample_count >= self.fast_stable_samples
        ):
            if self.fast_decay_s <= 0.0:
                self.fast_weight = 0.0
            else:
                self.fast_weight = max(0.0, self.fast_weight - dt / self.fast_decay_s)

        fast_allowed = (
            self.fast_enabled
            and self.fast_weight > 0.0
            and z_error > self.fast_min_z_error
            and ref_vz >= 0.0
            and signal > 0.0
        )
        if not fast_allowed:
            return self.gain

        return self.gain + self.fast_weight * (self.fast_gain - self.gain)

    def _debug(
        self,
        status: str,
        *,
        signal: float = 0.0,
        z_error: float = 0.0,
        vz_error: float = 0.0,
        dt: float = 0.0,
        gain: float | None = None,
    ) -> AdaptiveHoverDebug:
        self.last_debug = AdaptiveHoverDebug(
            value=float(self.value),
            status=str(status),
            signal=float(signal),
            z_error=float(z_error),
            vz_error=float(vz_error),
            dt=float(dt),
            gain=float(self.gain if gain is None else gain),
            fast_weight=float(self.fast_weight),
        )
        return self.last_debug

    @staticmethod
    def _vertical_terms(state, ref) -> tuple[float, float, float, float, float]:
        state_pos = np.asarray(state.pos, dtype=float).reshape(3)
        state_vel = np.asarray(state.vel, dtype=float).reshape(3)
        ref_pos = np.asarray(ref.pos, dtype=float).reshape(3)
        ref_vel = np.asarray(ref.vel, dtype=float).reshape(3)
        ref_acc = np.asarray(ref.acc, dtype=float).reshape(3)
        return (
            float(state_pos[2]),
            float(state_vel[2]),
            float(ref_pos[2]),
            float(ref_vel[2]),
            float(ref_acc[2]),
        )
