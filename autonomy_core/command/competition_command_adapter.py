"""Dry-run competition command adapter skeleton.

This module is intentionally import-safe and dependency-free. It prepares
SET_ATTITUDE_TARGET-style field dictionaries for review and tests, but it does
not import pymavlink, open sockets, or send commands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from autonomy_core.core.competition_config import (
    PYAIPILOT_EXAMPLE_REFERENCE,
    VADR_TS_002,
    RuntimeCompetitionConfig,
)


SET_ATTITUDE_TARGET_MESSAGE_NAME = "SET_ATTITUDE_TARGET"

MAVLINK_ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE = 1
MAVLINK_ATTITUDE_TARGET_TYPEMASK_BODY_PITCH_RATE_IGNORE = 2
MAVLINK_ATTITUDE_TARGET_TYPEMASK_BODY_YAW_RATE_IGNORE = 4
MAVLINK_ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE = 128

AUTONOMY_ATTITUDE_ANGLE_TYPE_MASK = (
    MAVLINK_ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE
    | MAVLINK_ATTITUDE_TARGET_TYPEMASK_BODY_PITCH_RATE_IGNORE
    | MAVLINK_ATTITUDE_TARGET_TYPEMASK_BODY_YAW_RATE_IGNORE
)
PYAIPILOT_BODY_RATE_REFERENCE_TYPE_MASK = MAVLINK_ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE

AUTONOMY_TUPLE_SEMANTICS = (
    "AutonomyAPI.attitude_control tuple: roll_rad attitude angle, "
    "pitch_rad attitude angle, yaw_rad attitude angle, normalized thrust 0..1. "
    "The tuple is not a body-rate command."
)
COMMAND_SEND_BLOCKED_REASON = (
    "Phase 5A is dry-run only; live command publication is blocked until "
    "Phase 4B proves usable competition telemetry and a later phase explicitly "
    "enables sending."
)


class CompetitionCommandAdapterError(ValueError):
    """Raised when a command tuple cannot be represented safely."""


@dataclass(frozen=True)
class AutonomyAttitudeCommand:
    """Documented AutonomyAPI attitude tuple semantics."""

    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    thrust: float
    semantics: str = AUTONOMY_TUPLE_SEMANTICS


@dataclass(frozen=True)
class DryRunSetAttitudeTargetFields:
    """Would-be MAVLink SET_ATTITUDE_TARGET fields, without serialization/send."""

    message_name: str
    time_boot_ms: int
    target_system: int
    target_component: int
    type_mask: int
    q: Tuple[float, float, float, float]
    body_roll_rate: float
    body_pitch_rate: float
    body_yaw_rate: float
    thrust: float
    source_tuple_semantics: str
    send_ready: bool
    send_blocked_reason: str
    sequence: Optional[int] = None

    def as_pymavlink_args(self) -> Tuple[
        int,
        int,
        int,
        int,
        Tuple[float, float, float, float],
        float,
        float,
        float,
        float,
    ]:
        """Return positional args matching set_attitude_target_send order."""

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
class DryRunCommandResult:
    accepted: bool
    fields: Optional[DryRunSetAttitudeTargetFields]
    rejection_reason: str = ""


class DryRunCommandRateGate:
    """Strictly enforce the VADR-TS-002 command-rate ceiling in dry-run code."""

    def __init__(self, *, config: RuntimeCompetitionConfig = VADR_TS_002):
        self.min_period_s = config.command_period_lower_bound_exclusive_s
        self.last_accepted_monotonic_s: Optional[float] = None

    def check_and_record(self, now_s: float) -> tuple[bool, str]:
        now = _finite_float(now_s, name="now_s")
        if self.last_accepted_monotonic_s is None:
            self.last_accepted_monotonic_s = now
            return True, ""

        elapsed = now - self.last_accepted_monotonic_s
        if elapsed <= self.min_period_s:
            return False, "command_rate_limit"

        self.last_accepted_monotonic_s = now
        return True, ""


class CompetitionDryRunCommandAdapter:
    """Build dry-run command fields and record rejections without sending."""

    def __init__(self, *, config: RuntimeCompetitionConfig = VADR_TS_002):
        self.config = config
        self.rate_gate = DryRunCommandRateGate(config=config)

    def build_set_attitude_target(
        self,
        command_tuple: Sequence[float],
        *,
        time_boot_ms: int,
        target_system: int,
        target_component: int,
        now_s: float,
        sequence: Optional[int] = None,
    ) -> DryRunCommandResult:
        try:
            fields = build_dry_run_set_attitude_target_fields(
                command_tuple,
                time_boot_ms=time_boot_ms,
                target_system=target_system,
                target_component=target_component,
                sequence=sequence,
            )
        except CompetitionCommandAdapterError as exc:
            return DryRunCommandResult(
                accepted=False,
                fields=None,
                rejection_reason=str(exc),
            )

        try:
            allowed, reason = self.rate_gate.check_and_record(now_s)
        except CompetitionCommandAdapterError as exc:
            return DryRunCommandResult(
                accepted=False,
                fields=None,
                rejection_reason=str(exc),
            )
        if not allowed:
            return DryRunCommandResult(
                accepted=False,
                fields=None,
                rejection_reason=reason,
            )

        return DryRunCommandResult(accepted=True, fields=fields)


def autonomy_attitude_command_from_tuple(
    command_tuple: Sequence[float],
) -> AutonomyAttitudeCommand:
    if len(command_tuple) != 4:
        raise CompetitionCommandAdapterError(
            "command tuple must contain roll, pitch, yaw, thrust"
        )

    roll_rad, pitch_rad, yaw_rad, thrust = (
        _finite_float(command_tuple[0], name="roll_rad"),
        _finite_float(command_tuple[1], name="pitch_rad"),
        _finite_float(command_tuple[2], name="yaw_rad"),
        _finite_float(command_tuple[3], name="thrust"),
    )

    thrust_min, thrust_max = PYAIPILOT_EXAMPLE_REFERENCE.normalized_thrust_range
    if not thrust_min <= thrust <= thrust_max:
        raise CompetitionCommandAdapterError("thrust must be normalized in 0..1")

    return AutonomyAttitudeCommand(
        roll_rad=roll_rad,
        pitch_rad=pitch_rad,
        yaw_rad=yaw_rad,
        thrust=thrust,
    )


def quaternion_wxyz_from_euler_zyx(
    *,
    roll_rad: float,
    pitch_rad: float,
    yaw_rad: float,
) -> Tuple[float, float, float, float]:
    """Build a MAVLink quaternion in w, x, y, z order from attitude angles."""

    roll = _finite_float(roll_rad, name="roll_rad")
    pitch = _finite_float(pitch_rad, name="pitch_rad")
    yaw = _finite_float(yaw_rad, name="yaw_rad")

    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    q = (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )
    norm = math.sqrt(sum(component * component for component in q))
    if norm <= 0.0 or not math.isfinite(norm):
        raise CompetitionCommandAdapterError("attitude quaternion is invalid")
    return tuple(component / norm for component in q)


def build_dry_run_set_attitude_target_fields(
    command_tuple: Sequence[float],
    *,
    time_boot_ms: int,
    target_system: int,
    target_component: int,
    sequence: Optional[int] = None,
) -> DryRunSetAttitudeTargetFields:
    command = autonomy_attitude_command_from_tuple(command_tuple)

    return DryRunSetAttitudeTargetFields(
        message_name=SET_ATTITUDE_TARGET_MESSAGE_NAME,
        time_boot_ms=_uint32(time_boot_ms, name="time_boot_ms"),
        target_system=_uint8(target_system, name="target_system"),
        target_component=_uint8(target_component, name="target_component"),
        type_mask=AUTONOMY_ATTITUDE_ANGLE_TYPE_MASK,
        q=quaternion_wxyz_from_euler_zyx(
            roll_rad=command.roll_rad,
            pitch_rad=command.pitch_rad,
            yaw_rad=command.yaw_rad,
        ),
        body_roll_rate=0.0,
        body_pitch_rate=0.0,
        body_yaw_rate=0.0,
        thrust=command.thrust,
        source_tuple_semantics=command.semantics,
        send_ready=False,
        send_blocked_reason=COMMAND_SEND_BLOCKED_REASON,
        sequence=None if sequence is None else _uint32(sequence, name="sequence"),
    )


def _finite_float(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise CompetitionCommandAdapterError(f"{name} must be finite")
    return result


def _uint8(value: int, *, name: str) -> int:
    result = int(value)
    if not 0 <= result <= 255:
        raise CompetitionCommandAdapterError(f"{name} must fit uint8")
    return result


def _uint32(value: int, *, name: str) -> int:
    result = int(value)
    if not 0 <= result <= 0xFFFFFFFF:
        raise CompetitionCommandAdapterError(f"{name} must fit uint32")
    return result


__all__ = [
    "AUTONOMY_ATTITUDE_ANGLE_TYPE_MASK",
    "AUTONOMY_TUPLE_SEMANTICS",
    "COMMAND_SEND_BLOCKED_REASON",
    "MAVLINK_ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE",
    "MAVLINK_ATTITUDE_TARGET_TYPEMASK_BODY_PITCH_RATE_IGNORE",
    "MAVLINK_ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE",
    "MAVLINK_ATTITUDE_TARGET_TYPEMASK_BODY_YAW_RATE_IGNORE",
    "PYAIPILOT_BODY_RATE_REFERENCE_TYPE_MASK",
    "SET_ATTITUDE_TARGET_MESSAGE_NAME",
    "AutonomyAttitudeCommand",
    "CompetitionCommandAdapterError",
    "CompetitionDryRunCommandAdapter",
    "DryRunCommandRateGate",
    "DryRunCommandResult",
    "DryRunSetAttitudeTargetFields",
    "autonomy_attitude_command_from_tuple",
    "build_dry_run_set_attitude_target_fields",
    "quaternion_wxyz_from_euler_zyx",
]
