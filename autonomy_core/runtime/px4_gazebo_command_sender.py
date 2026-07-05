"""PX4/Gazebo-only MAVLink command sender for Phase 9D/9E.

This module is import-safe: it does not import pymavlink, open sockets, start
PX4/Gazebo, arm, set offboard mode, or publish commands on import. It can send
only when a caller explicitly provides an already-open MAVLink connection or a
connection provider.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

from autonomy_core.command.competition_command_adapter import (
    AUTONOMY_ATTITUDE_ANGLE_TYPE_MASK,
    PYAIPILOT_BODY_RATE_REFERENCE_TYPE_MASK,
    SET_ATTITUDE_TARGET_MESSAGE_NAME,
    DryRunSetAttitudeTargetFields,
    build_dry_run_set_attitude_target_fields,
)
from autonomy_core.core.competition_config import (
    PYAIPILOT_EXAMPLE_REFERENCE,
    VADR_TS_002,
)


PX4_GAZEBO_SURROGATE_LABEL = "PX4/Gazebo surrogate command-send only"
PX4_GAZEBO_ARM_OFFBOARD_LABEL = "PX4/Gazebo surrogate arm/offboard only"
PX4_GAZEBO_BODY_RATE_SMOKE_LABEL = "PX4/Gazebo surrogate body-rate smoke only"
PX4_GAZEBO_ATTITUDE_HOVER_SMOKE_LABEL = (
    "PX4/Gazebo surrogate attitude-angle hover smoke only"
)
ATTITUDE_HOVER_BODY_RATES_IGNORE_TYPE_MASK = AUTONOMY_ATTITUDE_ANGLE_TYPE_MASK
ATTITUDE_HOVER_ZERO_BODY_RATES = (0.0, 0.0, 0.0)
BODY_RATE_ATTITUDE_IGNORE_TYPE_MASK = PYAIPILOT_BODY_RATE_REFERENCE_TYPE_MASK
BODY_RATE_DUMMY_QUATERNION = (1.0, 0.0, 0.0, 0.0)

MAV_CMD_COMPONENT_ARM_DISARM = 400
MAV_CMD_DO_SET_MODE = 176
MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1
PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6


class Px4GazeboCommandSenderError(RuntimeError):
    """Raised when the surrogate sender cannot validate or publish a command."""


@dataclass(frozen=True)
class Px4GazeboCommandSenderConfig:
    """Safety bounds for the PX4/Gazebo-only Phase 9D sender."""

    surrogate_label: str = PX4_GAZEBO_SURROGATE_LABEL
    allow_dry_run_fields: bool = True
    min_command_period_s: float = VADR_TS_002.command_period_lower_bound_exclusive_s
    min_thrust: float = PYAIPILOT_EXAMPLE_REFERENCE.normalized_thrust_range[0]
    max_thrust: float = PYAIPILOT_EXAMPLE_REFERENCE.normalized_thrust_range[1]
    max_abs_roll_pitch_rad: float = 0.7
    max_abs_body_rate_rad_s: float = 2.0
    enable_surrogate_thrust_clamp: bool = False
    surrogate_thrust_clamp_min: Optional[float] = None
    surrogate_thrust_clamp_max: Optional[float] = None


@dataclass(frozen=True)
class Px4GazeboCommandSendResult:
    """Result of one explicit PX4/Gazebo surrogate send attempt."""

    attempted: bool
    sent: bool
    rejection_reason: str = ""
    sent_at_s: Optional[float] = None
    message_name: str = SET_ATTITUDE_TARGET_MESSAGE_NAME
    sequence: Optional[int] = None
    target_system: Optional[int] = None
    target_component: Optional[int] = None
    type_mask: Optional[int] = None
    q: tuple[float, float, float, float] = ()
    body_roll_rate: Optional[float] = None
    body_pitch_rate: Optional[float] = None
    body_yaw_rate: Optional[float] = None
    roll_rad: Optional[float] = None
    pitch_rad: Optional[float] = None
    yaw_rad: Optional[float] = None
    raw_thrust: Optional[float] = None
    thrust: Optional[float] = None
    thrust_clamped: bool = False
    stream_source: str = ""
    send_gap_s: Optional[float] = None
    surrogate_label: str = PX4_GAZEBO_SURROGATE_LABEL
    phase4b_satisfied: bool = False
    competition_readiness_claimed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "sent": self.sent,
            "rejection_reason": self.rejection_reason,
            "sent_at_s": self.sent_at_s,
            "message_name": self.message_name,
            "sequence": self.sequence,
            "target_system": self.target_system,
            "target_component": self.target_component,
            "type_mask": self.type_mask,
            "q": list(self.q),
            "body_roll_rate": self.body_roll_rate,
            "body_pitch_rate": self.body_pitch_rate,
            "body_yaw_rate": self.body_yaw_rate,
            "roll_rad": self.roll_rad,
            "pitch_rad": self.pitch_rad,
            "yaw_rad": self.yaw_rad,
            "raw_thrust": self.raw_thrust,
            "thrust": self.thrust,
            "thrust_clamped": self.thrust_clamped,
            "stream_source": self.stream_source,
            "send_gap_s": self.send_gap_s,
            "surrogate_label": self.surrogate_label,
            "phase4b_satisfied": self.phase4b_satisfied,
            "competition_readiness_claimed": self.competition_readiness_claimed,
        }


@dataclass(frozen=True)
class Px4GazeboBodyRateSetAttitudeTargetFields:
    """Fixed body-rate SET_ATTITUDE_TARGET fields for Phase 9E.3 smoke tests."""

    message_name: str
    time_boot_ms: int
    target_system: int
    target_component: int
    type_mask: int
    q: tuple[float, float, float, float]
    body_roll_rate: float
    body_pitch_rate: float
    body_yaw_rate: float
    thrust: float
    sequence: Optional[int] = None

    def as_pymavlink_args(self) -> tuple[
        int,
        int,
        int,
        int,
        tuple[float, float, float, float],
        float,
        float,
        float,
        float,
    ]:
        return (
            self.time_boot_ms,
            self.target_system,
            self.target_component,
            self.type_mask,
            self.q,
            self.body_roll_rate,
            self.body_pitch_rate,
            self.body_yaw_rate,
            self.thrust,
        )


@dataclass(frozen=True)
class Px4GazeboLifecycleCommandResult:
    """Result of one PX4/Gazebo-only arm/offboard lifecycle command."""

    attempted: bool
    sent: bool
    command_name: str
    command_id: int
    target_system: Optional[int] = None
    target_component: Optional[int] = None
    params: tuple[float, float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    rejection_reason: str = ""
    sent_at_s: Optional[float] = None
    surrogate_label: str = PX4_GAZEBO_ARM_OFFBOARD_LABEL
    phase4b_satisfied: bool = False
    competition_readiness_claimed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "sent": self.sent,
            "command_name": self.command_name,
            "command_id": self.command_id,
            "target_system": self.target_system,
            "target_component": self.target_component,
            "params": list(self.params),
            "rejection_reason": self.rejection_reason,
            "sent_at_s": self.sent_at_s,
            "surrogate_label": self.surrogate_label,
            "phase4b_satisfied": self.phase4b_satisfied,
            "competition_readiness_claimed": self.competition_readiness_claimed,
        }


@dataclass
class Px4GazeboCommandSenderStats:
    send_attempts: int = 0
    sent_count: int = 0
    arm_attempts: int = 0
    arm_sent_count: int = 0
    offboard_attempts: int = 0
    offboard_sent_count: int = 0
    rejection_count: int = 0
    lifecycle_rejection_count: int = 0
    last_send_wall_time: Optional[float] = None
    last_lifecycle_wall_time: Optional[float] = None
    thrust_clamp_count: int = 0
    last_raw_thrust: Optional[float] = None
    last_sent_thrust: Optional[float] = None
    min_raw_thrust: Optional[float] = None
    max_raw_thrust: Optional[float] = None
    min_sent_thrust: Optional[float] = None
    max_sent_thrust: Optional[float] = None
    last_yaw_rad: Optional[float] = None
    min_yaw_rad: Optional[float] = None
    max_yaw_rad: Optional[float] = None
    last_send_gap_s: Optional[float] = None
    max_send_gap_s: Optional[float] = None
    stream_source_counts: dict[str, int] = field(default_factory=dict)
    rejection_reasons: dict[str, int] = field(default_factory=dict)

    def record_rejection(self, reason: str) -> None:
        self.rejection_count += 1
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1

    def record_sent(
        self,
        *,
        raw_fields: Any,
        sent_fields: Any,
        stream_source: str,
        thrust_clamped: bool,
        send_gap_s: Optional[float],
    ) -> None:
        self.sent_count += 1
        raw_thrust = float(raw_fields.thrust)
        sent_thrust = float(sent_fields.thrust)
        _roll, _pitch, yaw = _euler_zyx_from_quaternion_wxyz(
            tuple(float(component) for component in sent_fields.q)
        )
        source = stream_source or "unspecified"
        self.stream_source_counts[source] = self.stream_source_counts.get(source, 0) + 1
        if thrust_clamped:
            self.thrust_clamp_count += 1
        self.last_raw_thrust = raw_thrust
        self.last_sent_thrust = sent_thrust
        self.min_raw_thrust = _optional_min(self.min_raw_thrust, raw_thrust)
        self.max_raw_thrust = _optional_max(self.max_raw_thrust, raw_thrust)
        self.min_sent_thrust = _optional_min(self.min_sent_thrust, sent_thrust)
        self.max_sent_thrust = _optional_max(self.max_sent_thrust, sent_thrust)
        self.last_yaw_rad = yaw
        self.min_yaw_rad = _optional_min(self.min_yaw_rad, yaw)
        self.max_yaw_rad = _optional_max(self.max_yaw_rad, yaw)
        self.last_send_gap_s = send_gap_s
        if send_gap_s is not None:
            self.max_send_gap_s = _optional_max(self.max_send_gap_s, send_gap_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "send_attempts": self.send_attempts,
            "sent_count": self.sent_count,
            "arm_attempts": self.arm_attempts,
            "arm_sent_count": self.arm_sent_count,
            "offboard_attempts": self.offboard_attempts,
            "offboard_sent_count": self.offboard_sent_count,
            "rejection_count": self.rejection_count,
            "lifecycle_rejection_count": self.lifecycle_rejection_count,
            "last_send_wall_time": self.last_send_wall_time,
            "last_lifecycle_wall_time": self.last_lifecycle_wall_time,
            "thrust_clamp_count": self.thrust_clamp_count,
            "last_raw_thrust": self.last_raw_thrust,
            "last_sent_thrust": self.last_sent_thrust,
            "min_raw_thrust": self.min_raw_thrust,
            "max_raw_thrust": self.max_raw_thrust,
            "min_sent_thrust": self.min_sent_thrust,
            "max_sent_thrust": self.max_sent_thrust,
            "last_yaw_rad": self.last_yaw_rad,
            "min_yaw_rad": self.min_yaw_rad,
            "max_yaw_rad": self.max_yaw_rad,
            "last_send_gap_s": self.last_send_gap_s,
            "max_send_gap_s": self.max_send_gap_s,
            "stream_source_counts": dict(sorted(self.stream_source_counts.items())),
            "rejection_reasons": dict(sorted(self.rejection_reasons.items())),
        }


class Px4GazeboSetAttitudeTargetSender:
    """Send validated SET_ATTITUDE_TARGET fields to PX4/Gazebo only."""

    def __init__(
        self,
        *,
        config: Px4GazeboCommandSenderConfig = Px4GazeboCommandSenderConfig(),
        connection: Any = None,
        connection_provider: Optional[Callable[[], Any]] = None,
        clock: Callable[[], float] = time.time,
    ):
        if config.min_command_period_s < VADR_TS_002.command_period_lower_bound_exclusive_s:
            raise ValueError("min_command_period_s must preserve the <100 Hz limit")
        if config.min_thrust < 0.0 or config.max_thrust > 1.0:
            raise ValueError("thrust bounds must remain inside 0..1")
        if config.min_thrust > config.max_thrust:
            raise ValueError("min_thrust must be <= max_thrust")
        if config.max_abs_roll_pitch_rad <= 0.0:
            raise ValueError("max_abs_roll_pitch_rad must be positive")
        if config.max_abs_body_rate_rad_s <= 0.0:
            raise ValueError("max_abs_body_rate_rad_s must be positive")
        if config.enable_surrogate_thrust_clamp:
            if (
                config.surrogate_thrust_clamp_min is None
                and config.surrogate_thrust_clamp_max is None
            ):
                raise ValueError(
                    "surrogate thrust clamp requires at least one clamp bound"
                )
            _validate_optional_clamp_bound(
                config.surrogate_thrust_clamp_min,
                name="surrogate_thrust_clamp_min",
                min_allowed=config.min_thrust,
                max_allowed=config.max_thrust,
            )
            _validate_optional_clamp_bound(
                config.surrogate_thrust_clamp_max,
                name="surrogate_thrust_clamp_max",
                min_allowed=config.min_thrust,
                max_allowed=config.max_thrust,
            )
            if (
                config.surrogate_thrust_clamp_min is not None
                and config.surrogate_thrust_clamp_max is not None
                and config.surrogate_thrust_clamp_min > config.surrogate_thrust_clamp_max
            ):
                raise ValueError(
                    "surrogate_thrust_clamp_min must be <= surrogate_thrust_clamp_max"
                )

        self.config = config
        self._connection = connection
        self._connection_provider = connection_provider
        self.clock = clock
        self.stats = Px4GazeboCommandSenderStats()
        self._last_sent_at_s: Optional[float] = None

    def send_set_attitude_target(
        self,
        fields: DryRunSetAttitudeTargetFields,
        *,
        now_s: Optional[float] = None,
        stream_source: str = "",
    ) -> Px4GazeboCommandSendResult:
        """Validate and send one SET_ATTITUDE_TARGET message."""

        self.stats.send_attempts += 1
        now = float(self.clock() if now_s is None else now_s)
        send_gap_s = None if self._last_sent_at_s is None else now - self._last_sent_at_s
        send_fields = fields
        thrust_clamped = False
        try:
            send_fields, thrust_clamped = self._apply_surrogate_thrust_clamp(fields)
            self._validate_fields(send_fields)
            self._check_rate(now)
            connection = self._connection_for_send()
            send = getattr(getattr(connection, "mav", None), "set_attitude_target_send", None)
            if not callable(send):
                raise Px4GazeboCommandSenderError(
                    "mavlink connection does not expose mav.set_attitude_target_send"
                )
            send(*send_fields.as_pymavlink_args())
        except (Px4GazeboCommandSenderError, ValueError, TypeError) as exc:
            reason = str(exc)
            self.stats.record_rejection(reason)
            return _result_from_fields(
                send_fields,
                attempted=True,
                sent=False,
                rejection_reason=reason,
                sent_at_s=None,
                surrogate_label=self.config.surrogate_label,
                raw_fields=fields,
                thrust_clamped=thrust_clamped,
                stream_source=stream_source,
                send_gap_s=send_gap_s,
            )

        self._last_sent_at_s = now
        self.stats.last_send_wall_time = now
        self.stats.record_sent(
            raw_fields=fields,
            sent_fields=send_fields,
            stream_source=stream_source,
            thrust_clamped=thrust_clamped,
            send_gap_s=send_gap_s,
        )
        return _result_from_fields(
            send_fields,
            attempted=True,
            sent=True,
            rejection_reason="",
            sent_at_s=now,
            surrogate_label=self.config.surrogate_label,
            raw_fields=fields,
            thrust_clamped=thrust_clamped,
            stream_source=stream_source,
            send_gap_s=send_gap_s,
        )

    def send_body_rate_set_attitude_target(
        self,
        *,
        target_system: int,
        target_component: int,
        body_roll_rate: float,
        body_pitch_rate: float,
        body_yaw_rate: float,
        thrust: float,
        time_boot_ms: int,
        sequence: Optional[int] = None,
        now_s: Optional[float] = None,
        stream_source: str = "body_rate_smoke",
    ) -> Px4GazeboCommandSendResult:
        """Send one fixed body-rate SET_ATTITUDE_TARGET smoke command."""

        self.stats.send_attempts += 1
        now = float(self.clock() if now_s is None else now_s)
        send_gap_s = None if self._last_sent_at_s is None else now - self._last_sent_at_s
        fields = Px4GazeboBodyRateSetAttitudeTargetFields(
            message_name=SET_ATTITUDE_TARGET_MESSAGE_NAME,
            time_boot_ms=int(time_boot_ms),
            target_system=int(target_system),
            target_component=int(target_component),
            type_mask=BODY_RATE_ATTITUDE_IGNORE_TYPE_MASK,
            q=BODY_RATE_DUMMY_QUATERNION,
            body_roll_rate=float(body_roll_rate),
            body_pitch_rate=float(body_pitch_rate),
            body_yaw_rate=float(body_yaw_rate),
            thrust=float(thrust),
            sequence=sequence,
        )
        try:
            fields = self._normalized_body_rate_fields(fields)
            self._check_rate(now)
            connection = self._connection_for_send()
            send = getattr(getattr(connection, "mav", None), "set_attitude_target_send", None)
            if not callable(send):
                raise Px4GazeboCommandSenderError(
                    "mavlink connection does not expose mav.set_attitude_target_send"
                )
            send(*fields.as_pymavlink_args())
        except (Px4GazeboCommandSenderError, ValueError, TypeError) as exc:
            reason = str(exc)
            self.stats.record_rejection(reason)
            return _result_from_fields(
                fields,
                attempted=True,
                sent=False,
                rejection_reason=reason,
                sent_at_s=None,
                surrogate_label=PX4_GAZEBO_BODY_RATE_SMOKE_LABEL,
                raw_fields=fields,
                thrust_clamped=False,
                stream_source=stream_source,
                send_gap_s=send_gap_s,
            )

        self._last_sent_at_s = now
        self.stats.last_send_wall_time = now
        self.stats.record_sent(
            raw_fields=fields,
            sent_fields=fields,
            stream_source=stream_source,
            thrust_clamped=False,
            send_gap_s=send_gap_s,
        )
        return _result_from_fields(
            fields,
            attempted=True,
            sent=True,
            rejection_reason="",
            sent_at_s=now,
            surrogate_label=PX4_GAZEBO_BODY_RATE_SMOKE_LABEL,
            raw_fields=fields,
            thrust_clamped=False,
            stream_source=stream_source,
            send_gap_s=send_gap_s,
        )

    def send_attitude_hover_set_attitude_target(
        self,
        *,
        target_system: int,
        target_component: int,
        roll_rad: float,
        pitch_rad: float,
        yaw_rad: float,
        thrust: float,
        time_boot_ms: int,
        sequence: Optional[int] = None,
        now_s: Optional[float] = None,
        stream_source: str = "attitude_hover_smoke",
    ) -> Px4GazeboCommandSendResult:
        """Send one fixed attitude-angle SET_ATTITUDE_TARGET hover smoke command."""

        self.stats.send_attempts += 1
        now = float(self.clock() if now_s is None else now_s)
        send_gap_s = None if self._last_sent_at_s is None else now - self._last_sent_at_s
        fields: Optional[DryRunSetAttitudeTargetFields] = None
        try:
            fields = build_dry_run_set_attitude_target_fields(
                (float(roll_rad), float(pitch_rad), float(yaw_rad), float(thrust)),
                time_boot_ms=int(time_boot_ms),
                target_system=int(target_system),
                target_component=int(target_component),
                sequence=sequence,
            )
            self._validate_fields(fields)
            self._check_rate(now)
            connection = self._connection_for_send()
            send = getattr(getattr(connection, "mav", None), "set_attitude_target_send", None)
            if not callable(send):
                raise Px4GazeboCommandSenderError(
                    "mavlink connection does not expose mav.set_attitude_target_send"
                )
            send(*fields.as_pymavlink_args())
        except (Px4GazeboCommandSenderError, ValueError, TypeError) as exc:
            reason = str(exc)
            self.stats.record_rejection(reason)
            if fields is None:
                return Px4GazeboCommandSendResult(
                    attempted=True,
                    sent=False,
                    rejection_reason=reason,
                    sent_at_s=None,
                    message_name=SET_ATTITUDE_TARGET_MESSAGE_NAME,
                    sequence=sequence,
                    target_system=None,
                    target_component=None,
                    type_mask=ATTITUDE_HOVER_BODY_RATES_IGNORE_TYPE_MASK,
                    q=(),
                    body_roll_rate=None,
                    body_pitch_rate=None,
                    body_yaw_rate=None,
                    roll_rad=None,
                    pitch_rad=None,
                    yaw_rad=None,
                    raw_thrust=None,
                    thrust=None,
                    thrust_clamped=False,
                    stream_source=stream_source,
                    send_gap_s=send_gap_s,
                    surrogate_label=PX4_GAZEBO_ATTITUDE_HOVER_SMOKE_LABEL,
                )
            return _result_from_fields(
                fields,
                attempted=True,
                sent=False,
                rejection_reason=reason,
                sent_at_s=None,
                surrogate_label=PX4_GAZEBO_ATTITUDE_HOVER_SMOKE_LABEL,
                raw_fields=fields,
                thrust_clamped=False,
                stream_source=stream_source,
                send_gap_s=send_gap_s,
            )

        self._last_sent_at_s = now
        self.stats.last_send_wall_time = now
        self.stats.record_sent(
            raw_fields=fields,
            sent_fields=fields,
            stream_source=stream_source,
            thrust_clamped=False,
            send_gap_s=send_gap_s,
        )
        return _result_from_fields(
            fields,
            attempted=True,
            sent=True,
            rejection_reason="",
            sent_at_s=now,
            surrogate_label=PX4_GAZEBO_ATTITUDE_HOVER_SMOKE_LABEL,
            raw_fields=fields,
            thrust_clamped=False,
            stream_source=stream_source,
            send_gap_s=send_gap_s,
        )

    def send_arm_command(
        self,
        *,
        target_system: int,
        target_component: int,
        arm: bool = True,
        now_s: Optional[float] = None,
    ) -> Px4GazeboLifecycleCommandResult:
        """Send PX4/Gazebo-only MAV_CMD_COMPONENT_ARM_DISARM."""

        self.stats.arm_attempts += 1
        params = (1.0 if arm else 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        result = self._send_command_long(
            command_name="MAV_CMD_COMPONENT_ARM_DISARM",
            command_id=MAV_CMD_COMPONENT_ARM_DISARM,
            target_system=target_system,
            target_component=target_component,
            params=params,
            now_s=now_s,
        )
        if result.sent:
            self.stats.arm_sent_count += 1
        return result

    def send_offboard_mode_command(
        self,
        *,
        target_system: int,
        target_component: int,
        now_s: Optional[float] = None,
    ) -> Px4GazeboLifecycleCommandResult:
        """Send PX4/Gazebo-only MAV_CMD_DO_SET_MODE for PX4 Offboard."""

        self.stats.offboard_attempts += 1
        params = (
            float(MAV_MODE_FLAG_CUSTOM_MODE_ENABLED),
            float(PX4_CUSTOM_MAIN_MODE_OFFBOARD),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        result = self._send_command_long(
            command_name="MAV_CMD_DO_SET_MODE_OFFBOARD",
            command_id=MAV_CMD_DO_SET_MODE,
            target_system=target_system,
            target_component=target_component,
            params=params,
            now_s=now_s,
        )
        if result.sent:
            self.stats.offboard_sent_count += 1
        return result

    def summary(self) -> dict[str, Any]:
        return {
            "surrogate_label": self.config.surrogate_label,
            "arm_offboard_surrogate_label": PX4_GAZEBO_ARM_OFFBOARD_LABEL,
            "body_rate_smoke_surrogate_label": PX4_GAZEBO_BODY_RATE_SMOKE_LABEL,
            "attitude_hover_smoke_surrogate_label": (
                PX4_GAZEBO_ATTITUDE_HOVER_SMOKE_LABEL
            ),
            "phase4b_satisfied": False,
            "competition_readiness_claimed": False,
            "body_rate_smoke": {
                "type_mask": BODY_RATE_ATTITUDE_IGNORE_TYPE_MASK,
                "q": list(BODY_RATE_DUMMY_QUATERNION),
            },
            "attitude_hover_smoke": {
                "type_mask": ATTITUDE_HOVER_BODY_RATES_IGNORE_TYPE_MASK,
                "body_rates": list(ATTITUDE_HOVER_ZERO_BODY_RATES),
            },
            "stats": self.stats.to_dict(),
            "safety": {
                "allow_dry_run_fields": self.config.allow_dry_run_fields,
                "min_command_period_s": self.config.min_command_period_s,
                "min_thrust": self.config.min_thrust,
                "max_thrust": self.config.max_thrust,
                "max_abs_roll_pitch_rad": self.config.max_abs_roll_pitch_rad,
                "max_abs_body_rate_rad_s": self.config.max_abs_body_rate_rad_s,
                "enable_surrogate_thrust_clamp": (
                    self.config.enable_surrogate_thrust_clamp
                ),
                "surrogate_thrust_clamp_min": (
                    self.config.surrogate_thrust_clamp_min
                ),
                "surrogate_thrust_clamp_max": (
                    self.config.surrogate_thrust_clamp_max
                ),
            },
        }

    def _connection_for_send(self) -> Any:
        connection = self._connection
        if connection is None and self._connection_provider is not None:
            connection = self._connection_provider()
        if connection is None:
            raise Px4GazeboCommandSenderError("mavlink connection unavailable")
        return connection

    def _send_command_long(
        self,
        *,
        command_name: str,
        command_id: int,
        target_system: int,
        target_component: int,
        params: tuple[float, float, float, float, float, float, float],
        now_s: Optional[float],
    ) -> Px4GazeboLifecycleCommandResult:
        now = float(self.clock() if now_s is None else now_s)
        try:
            target_system = _uint8(target_system, "target_system")
            target_component = _uint8(target_component, "target_component")
            params = tuple(_finite(value, "command_long_param") for value in params)
            connection = self._connection_for_send()
            send = getattr(getattr(connection, "mav", None), "command_long_send", None)
            if not callable(send):
                raise Px4GazeboCommandSenderError(
                    "mavlink connection does not expose mav.command_long_send"
                )
            send(
                target_system,
                target_component,
                int(command_id),
                0,
                *params,
            )
        except (Px4GazeboCommandSenderError, ValueError, TypeError) as exc:
            reason = str(exc)
            self.stats.record_rejection(reason)
            self.stats.lifecycle_rejection_count += 1
            return Px4GazeboLifecycleCommandResult(
                attempted=True,
                sent=False,
                command_name=command_name,
                command_id=int(command_id),
                target_system=target_system if isinstance(target_system, int) else None,
                target_component=(
                    target_component if isinstance(target_component, int) else None
                ),
                params=params if isinstance(params, tuple) else (),
                rejection_reason=reason,
                sent_at_s=None,
            )

        self.stats.last_lifecycle_wall_time = now
        return Px4GazeboLifecycleCommandResult(
            attempted=True,
            sent=True,
            command_name=command_name,
            command_id=int(command_id),
            target_system=target_system,
            target_component=target_component,
            params=params,
            rejection_reason="",
            sent_at_s=now,
        )

    def _check_rate(self, now_s: float) -> None:
        if not math.isfinite(now_s):
            raise Px4GazeboCommandSenderError("now_s must be finite")
        if self._last_sent_at_s is None:
            return
        elapsed = now_s - self._last_sent_at_s
        if elapsed <= self.config.min_command_period_s:
            raise Px4GazeboCommandSenderError("command_rate_limit")

    def _validate_fields(self, fields: DryRunSetAttitudeTargetFields) -> None:
        if fields.message_name != SET_ATTITUDE_TARGET_MESSAGE_NAME:
            raise Px4GazeboCommandSenderError("message_name_must_be_SET_ATTITUDE_TARGET")
        if fields.type_mask != AUTONOMY_ATTITUDE_ANGLE_TYPE_MASK:
            raise Px4GazeboCommandSenderError("unexpected_type_mask")
        if fields.send_ready and not self.config.allow_dry_run_fields:
            raise Px4GazeboCommandSenderError("unexpected_send_ready_field")
        if not fields.send_ready and not self.config.allow_dry_run_fields:
            raise Px4GazeboCommandSenderError("dry_run_fields_not_allowed")
        if len(fields.q) != 4:
            raise Px4GazeboCommandSenderError("quaternion_must_have_four_components")
        q = tuple(_finite(component, "q") for component in fields.q)
        norm = math.sqrt(sum(component * component for component in q))
        if abs(norm - 1.0) > 1e-3:
            raise Px4GazeboCommandSenderError("quaternion_norm_invalid")

        roll, pitch, _yaw = _euler_zyx_from_quaternion_wxyz(q)
        if abs(roll) > self.config.max_abs_roll_pitch_rad:
            raise Px4GazeboCommandSenderError("roll_safety_limit")
        if abs(pitch) > self.config.max_abs_roll_pitch_rad:
            raise Px4GazeboCommandSenderError("pitch_safety_limit")

        for name, value in (
            ("body_roll_rate", fields.body_roll_rate),
            ("body_pitch_rate", fields.body_pitch_rate),
            ("body_yaw_rate", fields.body_yaw_rate),
        ):
            body_rate = _finite(value, name)
            if abs(body_rate) > self.config.max_abs_body_rate_rad_s:
                raise Px4GazeboCommandSenderError(f"{name}_safety_limit")

        thrust = _finite(fields.thrust, "thrust")
        if not self.config.min_thrust <= thrust <= self.config.max_thrust:
            raise Px4GazeboCommandSenderError("thrust_safety_limit")

    def _normalized_body_rate_fields(
        self,
        fields: Px4GazeboBodyRateSetAttitudeTargetFields,
    ) -> Px4GazeboBodyRateSetAttitudeTargetFields:
        if fields.message_name != SET_ATTITUDE_TARGET_MESSAGE_NAME:
            raise Px4GazeboCommandSenderError("message_name_must_be_SET_ATTITUDE_TARGET")
        if fields.type_mask != BODY_RATE_ATTITUDE_IGNORE_TYPE_MASK:
            raise Px4GazeboCommandSenderError("unexpected_body_rate_type_mask")

        target_system = _uint8(fields.target_system, "target_system")
        target_component = _uint8(fields.target_component, "target_component")
        time_boot_ms = _uint32(fields.time_boot_ms, "time_boot_ms")

        if tuple(fields.q) != BODY_RATE_DUMMY_QUATERNION:
            raise Px4GazeboCommandSenderError("body_rate_q_must_be_dummy_identity")
        q = tuple(_finite(component, "q") for component in fields.q)
        norm = math.sqrt(sum(component * component for component in q))
        if abs(norm - 1.0) > 1e-3:
            raise Px4GazeboCommandSenderError("quaternion_norm_invalid")

        body_roll_rate = _finite(fields.body_roll_rate, "body_roll_rate")
        body_pitch_rate = _finite(fields.body_pitch_rate, "body_pitch_rate")
        body_yaw_rate = _finite(fields.body_yaw_rate, "body_yaw_rate")
        for name, value in (
            ("body_roll_rate", body_roll_rate),
            ("body_pitch_rate", body_pitch_rate),
            ("body_yaw_rate", body_yaw_rate),
        ):
            if abs(value) > self.config.max_abs_body_rate_rad_s:
                raise Px4GazeboCommandSenderError(f"{name}_safety_limit")

        thrust = _finite(fields.thrust, "thrust")
        if not self.config.min_thrust <= thrust <= self.config.max_thrust:
            raise Px4GazeboCommandSenderError("thrust_safety_limit")

        return replace(
            fields,
            time_boot_ms=time_boot_ms,
            target_system=target_system,
            target_component=target_component,
            q=q,
            body_roll_rate=body_roll_rate,
            body_pitch_rate=body_pitch_rate,
            body_yaw_rate=body_yaw_rate,
            thrust=thrust,
        )

    def _apply_surrogate_thrust_clamp(
        self,
        fields: DryRunSetAttitudeTargetFields,
    ) -> tuple[DryRunSetAttitudeTargetFields, bool]:
        raw_thrust = _finite(fields.thrust, "raw_thrust")
        if not 0.0 <= raw_thrust <= 1.0:
            raise Px4GazeboCommandSenderError("raw_thrust_outside_normalized_range")
        if not self.config.enable_surrogate_thrust_clamp:
            return fields, False

        clamped_thrust = raw_thrust
        if self.config.surrogate_thrust_clamp_min is not None:
            clamped_thrust = max(
                clamped_thrust,
                float(self.config.surrogate_thrust_clamp_min),
            )
        if self.config.surrogate_thrust_clamp_max is not None:
            clamped_thrust = min(
                clamped_thrust,
                float(self.config.surrogate_thrust_clamp_max),
            )
        if clamped_thrust == raw_thrust:
            return fields, False
        return replace(fields, thrust=clamped_thrust), True


def _result_from_fields(
    fields: Any,
    *,
    attempted: bool,
    sent: bool,
    rejection_reason: str,
    sent_at_s: Optional[float],
    surrogate_label: str,
    raw_fields: Optional[Any] = None,
    thrust_clamped: bool = False,
    stream_source: str = "",
    send_gap_s: Optional[float] = None,
) -> Px4GazeboCommandSendResult:
    roll, pitch, yaw = _euler_zyx_from_quaternion_wxyz(
        tuple(float(component) for component in fields.q)
    )
    raw = raw_fields or fields
    return Px4GazeboCommandSendResult(
        attempted=attempted,
        sent=sent,
        rejection_reason=rejection_reason,
        sent_at_s=sent_at_s,
        message_name=fields.message_name,
        sequence=fields.sequence,
        target_system=fields.target_system,
        target_component=fields.target_component,
        type_mask=fields.type_mask,
        q=tuple(float(component) for component in fields.q),
        body_roll_rate=float(fields.body_roll_rate),
        body_pitch_rate=float(fields.body_pitch_rate),
        body_yaw_rate=float(fields.body_yaw_rate),
        roll_rad=roll,
        pitch_rad=pitch,
        yaw_rad=yaw,
        raw_thrust=float(raw.thrust),
        thrust=float(fields.thrust),
        thrust_clamped=bool(thrust_clamped),
        stream_source=str(stream_source),
        send_gap_s=send_gap_s,
        surrogate_label=surrogate_label,
    )


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise Px4GazeboCommandSenderError(f"{name}_must_be_finite")
    return result


def _uint8(value: int, name: str) -> int:
    result = int(value)
    if not 0 <= result <= 255:
        raise Px4GazeboCommandSenderError(f"{name}_must_fit_uint8")
    return result


def _uint32(value: int, name: str) -> int:
    result = int(value)
    if not 0 <= result <= 0xFFFFFFFF:
        raise Px4GazeboCommandSenderError(f"{name}_must_fit_uint32")
    return result


def _validate_optional_clamp_bound(
    value: Optional[float],
    *,
    name: str,
    min_allowed: float,
    max_allowed: float,
) -> None:
    if value is None:
        return
    bound = float(value)
    if not math.isfinite(bound):
        raise ValueError(f"{name} must be finite")
    if not min_allowed <= bound <= max_allowed:
        raise ValueError(f"{name} must remain inside thrust safety bounds")


def _optional_min(current: Optional[float], value: float) -> float:
    return value if current is None else min(current, value)


def _optional_max(current: Optional[float], value: float) -> float:
    return value if current is None else max(current, value)


def _euler_zyx_from_quaternion_wxyz(
    q: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


__all__ = [
    "BODY_RATE_ATTITUDE_IGNORE_TYPE_MASK",
    "BODY_RATE_DUMMY_QUATERNION",
    "MAV_CMD_COMPONENT_ARM_DISARM",
    "MAV_CMD_DO_SET_MODE",
    "MAV_MODE_FLAG_CUSTOM_MODE_ENABLED",
    "PX4_CUSTOM_MAIN_MODE_OFFBOARD",
    "PX4_GAZEBO_ARM_OFFBOARD_LABEL",
    "PX4_GAZEBO_BODY_RATE_SMOKE_LABEL",
    "PX4_GAZEBO_SURROGATE_LABEL",
    "Px4GazeboBodyRateSetAttitudeTargetFields",
    "Px4GazeboCommandSendResult",
    "Px4GazeboCommandSenderConfig",
    "Px4GazeboCommandSenderError",
    "Px4GazeboCommandSenderStats",
    "Px4GazeboLifecycleCommandResult",
    "Px4GazeboSetAttitudeTargetSender",
]
