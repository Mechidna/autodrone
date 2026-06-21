"""Generic competition setpoint stream policy.

This module is import-safe and transport-free. It decides which already-built
SET_ATTITUDE_TARGET fields should be emitted at a fixed cadence, but it never
opens sockets, imports pymavlink, arms PX4, switches Offboard, or sends
commands. Transport-specific layers consume the returned fields.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

from autonomy_core.command.competition_command_adapter import (
    SET_ATTITUDE_TARGET_MESSAGE_NAME,
    DryRunSetAttitudeTargetFields,
)
from autonomy_core.core.competition_config import RuntimeCompetitionConfig, VADR_TS_002


DEFAULT_COMPETITION_SETPOINT_STREAM_HZ = 20.0
DEFAULT_AUTONOMY_COMMAND_FRESH_S = 0.5

SETPOINT_SOURCE_AUTONOMY = "autonomy"
SETPOINT_SOURCE_FALLBACK = "fallback"
SETPOINT_SOURCE_NONE = "none"


class CompetitionSetpointStreamerError(ValueError):
    """Raised when stream policy inputs are invalid."""


@dataclass(frozen=True)
class CompetitionSetpointStreamConfig:
    """Cadence and freshness policy for streamed command fields."""

    stream_rate_hz: float = DEFAULT_COMPETITION_SETPOINT_STREAM_HZ
    autonomy_command_fresh_s: float = DEFAULT_AUTONOMY_COMMAND_FRESH_S
    require_fallback: bool = True
    competition_config: RuntimeCompetitionConfig = VADR_TS_002

    @property
    def stream_period_s(self) -> float:
        return 1.0 / float(self.stream_rate_hz)


@dataclass(frozen=True)
class CompetitionSetpointStreamDecision:
    """Result of one setpoint stream policy step."""

    should_emit: bool
    fields: Optional[DryRunSetAttitudeTargetFields]
    source: str
    reason: str
    now_s: float
    sequence: Optional[int] = None
    command_age_s: Optional[float] = None
    stream_period_s: Optional[float] = None
    next_due_s: Optional[float] = None
    phase4b_satisfied: bool = False
    competition_readiness_claimed: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "should_emit": self.should_emit,
            "source": self.source,
            "reason": self.reason,
            "now_s": self.now_s,
            "sequence": self.sequence,
            "command_age_s": self.command_age_s,
            "stream_period_s": self.stream_period_s,
            "next_due_s": self.next_due_s,
            "phase4b_satisfied": self.phase4b_satisfied,
            "competition_readiness_claimed": self.competition_readiness_claimed,
            "message_name": None if self.fields is None else self.fields.message_name,
        }


@dataclass
class CompetitionSetpointStreamStats:
    """Counters for deterministic stream diagnostics."""

    step_count: int = 0
    autonomy_update_count: int = 0
    fallback_update_count: int = 0
    emit_count: int = 0
    autonomy_emit_count: int = 0
    fallback_emit_count: int = 0
    no_emit_count: int = 0
    rate_limited_count: int = 0
    stale_autonomy_count: int = 0
    missing_autonomy_count: int = 0
    missing_fallback_count: int = 0
    invalid_update_count: int = 0
    last_emit_wall_time_s: Optional[float] = None
    max_emit_gap_s: Optional[float] = None
    last_source: str = SETPOINT_SOURCE_NONE
    last_reason: str = ""
    last_command_age_s: Optional[float] = None
    last_sequence: Optional[int] = None

    def record_emit(
        self,
        *,
        now_s: float,
        source: str,
        reason: str,
        command_age_s: Optional[float],
        sequence: int,
    ) -> None:
        if self.last_emit_wall_time_s is not None:
            gap_s = float(now_s) - float(self.last_emit_wall_time_s)
            self.max_emit_gap_s = _optional_max(self.max_emit_gap_s, gap_s)
        self.last_emit_wall_time_s = float(now_s)
        self.emit_count += 1
        if source == SETPOINT_SOURCE_AUTONOMY:
            self.autonomy_emit_count += 1
        elif source == SETPOINT_SOURCE_FALLBACK:
            self.fallback_emit_count += 1
        self.last_source = source
        self.last_reason = reason
        self.last_command_age_s = command_age_s
        self.last_sequence = int(sequence)

    def record_no_emit(self, *, reason: str) -> None:
        self.no_emit_count += 1
        self.last_source = SETPOINT_SOURCE_NONE
        self.last_reason = reason
        if reason == "stream_rate_wait":
            self.rate_limited_count += 1
        elif reason == "autonomy_command_missing":
            self.missing_autonomy_count += 1
        elif reason == "fallback_command_missing":
            self.missing_fallback_count += 1
        elif reason == "autonomy_command_stale":
            self.stale_autonomy_count += 1

    def to_dict(self) -> dict[str, object]:
        return {
            "step_count": self.step_count,
            "autonomy_update_count": self.autonomy_update_count,
            "fallback_update_count": self.fallback_update_count,
            "emit_count": self.emit_count,
            "autonomy_emit_count": self.autonomy_emit_count,
            "fallback_emit_count": self.fallback_emit_count,
            "no_emit_count": self.no_emit_count,
            "rate_limited_count": self.rate_limited_count,
            "stale_autonomy_count": self.stale_autonomy_count,
            "missing_autonomy_count": self.missing_autonomy_count,
            "missing_fallback_count": self.missing_fallback_count,
            "invalid_update_count": self.invalid_update_count,
            "last_emit_wall_time_s": self.last_emit_wall_time_s,
            "max_emit_gap_s": self.max_emit_gap_s,
            "last_source": self.last_source,
            "last_reason": self.last_reason,
            "last_command_age_s": self.last_command_age_s,
            "last_sequence": self.last_sequence,
        }


class CompetitionSetpointStreamer:
    """Select fresh autonomy fields or fallback fields at a fixed stream rate."""

    def __init__(
        self,
        *,
        config: CompetitionSetpointStreamConfig = CompetitionSetpointStreamConfig(),
        fallback_fields: Optional[DryRunSetAttitudeTargetFields] = None,
    ):
        _validate_config(config)
        self.config = config
        self.stats = CompetitionSetpointStreamStats()
        self._latest_autonomy_fields: Optional[DryRunSetAttitudeTargetFields] = None
        self._latest_autonomy_update_s: Optional[float] = None
        self._fallback_fields: Optional[DryRunSetAttitudeTargetFields] = None
        self._last_emit_s: Optional[float] = None
        self._sequence = 0
        if fallback_fields is not None:
            self.update_fallback_fields(fallback_fields)

    def update_autonomy_fields(
        self,
        fields: DryRunSetAttitudeTargetFields,
        *,
        now_s: float,
    ) -> None:
        """Cache the latest accepted autonomy command fields."""

        try:
            _validate_fields(fields)
            now = _finite_float(now_s, name="now_s")
        except CompetitionSetpointStreamerError:
            self.stats.invalid_update_count += 1
            raise
        self._latest_autonomy_fields = fields
        self._latest_autonomy_update_s = now
        self.stats.autonomy_update_count += 1

    def update_fallback_fields(self, fields: DryRunSetAttitudeTargetFields) -> None:
        """Set the fallback/hold fields used when autonomy is missing or stale."""

        try:
            _validate_fields(fields)
        except CompetitionSetpointStreamerError:
            self.stats.invalid_update_count += 1
            raise
        self._fallback_fields = fields
        self.stats.fallback_update_count += 1

    def step(self, *, now_s: float) -> CompetitionSetpointStreamDecision:
        """Return the next fields to emit, or a no-emit decision."""

        now = _finite_float(now_s, name="now_s")
        self.stats.step_count += 1

        if self._last_emit_s is not None:
            next_due_s = float(self._last_emit_s) + self.config.stream_period_s
            if now < next_due_s:
                self.stats.record_no_emit(reason="stream_rate_wait")
                return CompetitionSetpointStreamDecision(
                    should_emit=False,
                    fields=None,
                    source=SETPOINT_SOURCE_NONE,
                    reason="stream_rate_wait",
                    now_s=now,
                    stream_period_s=self.config.stream_period_s,
                    next_due_s=next_due_s,
                )

        fields, source, reason, command_age_s = self._select_fields(now)
        if fields is None:
            self.stats.record_no_emit(reason=reason)
            return CompetitionSetpointStreamDecision(
                should_emit=False,
                fields=None,
                source=SETPOINT_SOURCE_NONE,
                reason=reason,
                now_s=now,
                command_age_s=command_age_s,
                stream_period_s=self.config.stream_period_s,
                next_due_s=None,
            )

        self._sequence += 1
        emitted_fields = replace(
            fields,
            time_boot_ms=_time_boot_ms_from_wall_time(now),
            sequence=self._sequence,
        )
        self._last_emit_s = now
        self.stats.record_emit(
            now_s=now,
            source=source,
            reason=reason,
            command_age_s=command_age_s,
            sequence=self._sequence,
        )
        return CompetitionSetpointStreamDecision(
            should_emit=True,
            fields=emitted_fields,
            source=source,
            reason=reason,
            now_s=now,
            sequence=self._sequence,
            command_age_s=command_age_s,
            stream_period_s=self.config.stream_period_s,
            next_due_s=now + self.config.stream_period_s,
        )

    def summary(self) -> dict[str, object]:
        return {
            "stream_rate_hz": float(self.config.stream_rate_hz),
            "stream_period_s": self.config.stream_period_s,
            "autonomy_command_fresh_s": float(self.config.autonomy_command_fresh_s),
            "require_fallback": bool(self.config.require_fallback),
            "stats": self.stats.to_dict(),
            "phase4b_satisfied": False,
            "competition_readiness_claimed": False,
        }

    def _select_fields(
        self,
        now_s: float,
    ) -> tuple[Optional[DryRunSetAttitudeTargetFields], str, str, Optional[float]]:
        if self._latest_autonomy_fields is None:
            if self._fallback_fields is None:
                return None, SETPOINT_SOURCE_NONE, "fallback_command_missing", None
            return self._fallback_fields, SETPOINT_SOURCE_FALLBACK, "autonomy_command_missing", None

        if self._latest_autonomy_update_s is None:
            if self._fallback_fields is None:
                return None, SETPOINT_SOURCE_NONE, "fallback_command_missing", None
            return self._fallback_fields, SETPOINT_SOURCE_FALLBACK, "autonomy_command_time_missing", None

        command_age_s = float(now_s) - float(self._latest_autonomy_update_s)
        if command_age_s < 0.0:
            return None, SETPOINT_SOURCE_NONE, "autonomy_command_time_in_future", command_age_s
        if command_age_s <= float(self.config.autonomy_command_fresh_s):
            return (
                self._latest_autonomy_fields,
                SETPOINT_SOURCE_AUTONOMY,
                "autonomy_command_fresh",
                command_age_s,
            )

        self.stats.stale_autonomy_count += 1
        if self._fallback_fields is None:
            if self.config.require_fallback:
                return None, SETPOINT_SOURCE_NONE, "fallback_command_missing", command_age_s
            return None, SETPOINT_SOURCE_NONE, "autonomy_command_stale", command_age_s
        return (
            self._fallback_fields,
            SETPOINT_SOURCE_FALLBACK,
            "autonomy_command_stale",
            command_age_s,
        )


def _validate_config(config: CompetitionSetpointStreamConfig) -> None:
    stream_rate_hz = _finite_float(config.stream_rate_hz, name="stream_rate_hz")
    if stream_rate_hz <= 0.0:
        raise CompetitionSetpointStreamerError("stream_rate_hz must be positive")
    if not config.competition_config.command_rate_is_allowed(stream_rate_hz):
        raise CompetitionSetpointStreamerError(
            "stream_rate_hz must remain strictly below the VADR command limit"
        )
    if config.stream_period_s <= config.competition_config.command_period_lower_bound_exclusive_s:
        raise CompetitionSetpointStreamerError(
            "stream_period_s must preserve the strict command period lower bound"
        )
    command_fresh_s = _finite_float(
        config.autonomy_command_fresh_s,
        name="autonomy_command_fresh_s",
    )
    if command_fresh_s <= 0.0:
        raise CompetitionSetpointStreamerError(
            "autonomy_command_fresh_s must be positive"
        )


def _validate_fields(fields: DryRunSetAttitudeTargetFields) -> None:
    if not isinstance(fields, DryRunSetAttitudeTargetFields):
        raise CompetitionSetpointStreamerError(
            "fields must be DryRunSetAttitudeTargetFields"
        )
    if fields.message_name != SET_ATTITUDE_TARGET_MESSAGE_NAME:
        raise CompetitionSetpointStreamerError(
            "fields must represent SET_ATTITUDE_TARGET"
        )
    if len(tuple(fields.q)) != 4:
        raise CompetitionSetpointStreamerError("fields.q must contain 4 components")
    for index, component in enumerate(tuple(fields.q)):
        _finite_float(component, name=f"fields.q[{index}]")
    _finite_float(fields.body_roll_rate, name="fields.body_roll_rate")
    _finite_float(fields.body_pitch_rate, name="fields.body_pitch_rate")
    _finite_float(fields.body_yaw_rate, name="fields.body_yaw_rate")
    thrust = _finite_float(fields.thrust, name="fields.thrust")
    if not 0.0 <= thrust <= 1.0:
        raise CompetitionSetpointStreamerError("fields.thrust must be in 0..1")


def _finite_float(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise CompetitionSetpointStreamerError(f"{name} must be finite")
    return result


def _time_boot_ms_from_wall_time(now_s: float) -> int:
    return int(max(0.0, float(now_s)) * 1000.0) & 0xFFFFFFFF


def _optional_max(
    current: Optional[float],
    candidate: float,
) -> float:
    if current is None:
        return float(candidate)
    return max(float(current), float(candidate))


__all__ = [
    "DEFAULT_AUTONOMY_COMMAND_FRESH_S",
    "DEFAULT_COMPETITION_SETPOINT_STREAM_HZ",
    "SETPOINT_SOURCE_AUTONOMY",
    "SETPOINT_SOURCE_FALLBACK",
    "SETPOINT_SOURCE_NONE",
    "CompetitionSetpointStreamConfig",
    "CompetitionSetpointStreamDecision",
    "CompetitionSetpointStreamStats",
    "CompetitionSetpointStreamer",
    "CompetitionSetpointStreamerError",
]
