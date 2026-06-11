"""Competition MAVLink telemetry to internal VehicleState adapters.

This module is import-safe and has no pymavlink dependency. It operates on
message-like objects so tests and future receive-only tools can pass fake or
real MAVLink messages without changing existing runtime telemetry behavior.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from autonomy_core.core.types import VehicleState


LOCAL_POSITION_NED = "LOCAL_POSITION_NED"
ODOMETRY = "ODOMETRY"
ATTITUDE = "ATTITUDE"


class CompetitionStateAdapterError(ValueError):
    """Raised when a MAVLink telemetry message cannot produce a valid state."""


@dataclass(frozen=True)
class TimedMavlinkMessage:
    message: Any
    received_wall_time: float


@dataclass(frozen=True)
class CompetitionStateResult:
    vehicle_state: Optional[VehicleState]
    position_source: str = ""
    attitude_source: str = ""
    position_age_s: float = math.nan
    attitude_age_s: float = math.nan
    missing_reasons: tuple[str, ...] = ()

    @property
    def is_usable(self) -> bool:
        return self.vehicle_state is not None and len(self.missing_reasons) == 0


def _field(message: Any, name: str) -> Any:
    if isinstance(message, dict):
        return message[name]
    return getattr(message, name)


def _message_type(message: Any) -> str:
    get_type = getattr(message, "get_type", None)
    if callable(get_type):
        return str(get_type())
    if isinstance(message, dict) and "type" in message:
        return str(message["type"])
    return str(getattr(message, "mavpackettype", ""))


def _finite_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise CompetitionStateAdapterError(f"{name} must be finite")
    return result


def _finite_vector(values: Any, *, name: str, size: int = 3) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(size)
    if not np.all(np.isfinite(result)):
        raise CompetitionStateAdapterError(f"{name} must contain only finite values")
    return result


def ned_position_to_internal_z_up(x: Any, y: Any, z: Any) -> np.ndarray:
    """Convert MAVLink local NED position to current internal z-up position."""

    north = _finite_float(x, name="position x/north")
    east = _finite_float(y, name="position y/east")
    down = _finite_float(z, name="position z/down")
    return np.array([north, east, -down], dtype=float)


def ned_velocity_to_internal_z_up(vx: Any, vy: Any, vz: Any) -> np.ndarray:
    """Convert MAVLink local NED velocity to current internal z-up velocity."""

    north = _finite_float(vx, name="velocity x/north")
    east = _finite_float(vy, name="velocity y/east")
    down = _finite_float(vz, name="velocity z/down")
    return np.array([north, east, -down], dtype=float)


def yaw_from_attitude(message: Any) -> float:
    """Return MAVLink ATTITUDE yaw in the existing internal yaw convention."""

    return _finite_float(_field(message, "yaw"), name="attitude yaw")


def yaw_from_quaternion_wxyz(q: Any) -> float:
    """Extract ZYX yaw from a MAVLink quaternion in w, x, y, z order."""

    quat = _finite_vector(q, name="odometry quaternion", size=4)
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def vehicle_state_from_local_position_ned(
    local_position_ned_message: Any,
    attitude_message: Any,
) -> VehicleState:
    """Build a VehicleState from LOCAL_POSITION_NED plus ATTITUDE messages."""

    pos = ned_position_to_internal_z_up(
        _field(local_position_ned_message, "x"),
        _field(local_position_ned_message, "y"),
        _field(local_position_ned_message, "z"),
    )
    vel = ned_velocity_to_internal_z_up(
        _field(local_position_ned_message, "vx"),
        _field(local_position_ned_message, "vy"),
        _field(local_position_ned_message, "vz"),
    )
    yaw = yaw_from_attitude(attitude_message)
    return VehicleState(pos=pos, vel=vel, yaw=yaw)


def vehicle_state_from_odometry(odometry_message: Any) -> VehicleState:
    """Build a VehicleState from a MAVLink ODOMETRY message."""

    pos = ned_position_to_internal_z_up(
        _field(odometry_message, "x"),
        _field(odometry_message, "y"),
        _field(odometry_message, "z"),
    )
    vel = ned_velocity_to_internal_z_up(
        _field(odometry_message, "vx"),
        _field(odometry_message, "vy"),
        _field(odometry_message, "vz"),
    )
    yaw = yaw_from_quaternion_wxyz(_field(odometry_message, "q"))
    return VehicleState(pos=pos, vel=vel, yaw=yaw)


class CompetitionStateAdapter:
    """Small state adapter skeleton for receive-only MAVLink telemetry."""

    def __init__(self, *, clock: Callable[[], float] = time.time):
        self.clock = clock
        self.latest_local_position_ned: Optional[TimedMavlinkMessage] = None
        self.latest_odometry: Optional[TimedMavlinkMessage] = None
        self.latest_attitude: Optional[TimedMavlinkMessage] = None

    def ingest_message(
        self,
        message: Any,
        *,
        received_wall_time: Optional[float] = None,
    ) -> None:
        received = float(self.clock() if received_wall_time is None else received_wall_time)
        timed = TimedMavlinkMessage(message=message, received_wall_time=received)
        msg_type = _message_type(message)

        if msg_type == LOCAL_POSITION_NED:
            self.latest_local_position_ned = timed
        elif msg_type == ODOMETRY:
            self.latest_odometry = timed
        elif msg_type == ATTITUDE:
            self.latest_attitude = timed

    def latest_result(
        self,
        *,
        now: Optional[float] = None,
        max_age_s: float = 0.5,
    ) -> CompetitionStateResult:
        current_time = float(self.clock() if now is None else now)
        reasons: list[str] = []

        odometry = self._fresh(self.latest_odometry, current_time, max_age_s)
        if odometry is not None:
            try:
                state = vehicle_state_from_odometry(odometry.message)
            except (AttributeError, KeyError, CompetitionStateAdapterError) as exc:
                reasons.append(f"invalid_odometry:{exc}")
            else:
                age = current_time - odometry.received_wall_time
                return CompetitionStateResult(
                    vehicle_state=state,
                    position_source=ODOMETRY,
                    attitude_source=ODOMETRY,
                    position_age_s=age,
                    attitude_age_s=age,
                )

        local_position = self._fresh(
            self.latest_local_position_ned,
            current_time,
            max_age_s,
        )
        attitude = self._fresh(self.latest_attitude, current_time, max_age_s)

        if self.latest_odometry is not None and odometry is None:
            reasons.append("stale_odometry")
        if self.latest_local_position_ned is None:
            reasons.append("missing_local_position_ned")
        elif local_position is None:
            reasons.append("stale_local_position_ned")
        if self.latest_attitude is None:
            reasons.append("missing_attitude")
        elif attitude is None:
            reasons.append("stale_attitude")

        if local_position is None or attitude is None:
            return CompetitionStateResult(
                vehicle_state=None,
                missing_reasons=tuple(reasons),
            )

        try:
            state = vehicle_state_from_local_position_ned(
                local_position.message,
                attitude.message,
            )
        except (AttributeError, KeyError, CompetitionStateAdapterError) as exc:
            return CompetitionStateResult(
                vehicle_state=None,
                missing_reasons=tuple([*reasons, f"invalid_local_position_or_attitude:{exc}"]),
            )

        return CompetitionStateResult(
            vehicle_state=state,
            position_source=LOCAL_POSITION_NED,
            attitude_source=ATTITUDE,
            position_age_s=current_time - local_position.received_wall_time,
            attitude_age_s=current_time - attitude.received_wall_time,
            missing_reasons=tuple(reasons),
        )

    @staticmethod
    def _fresh(
        timed_message: Optional[TimedMavlinkMessage],
        now: float,
        max_age_s: float,
    ) -> Optional[TimedMavlinkMessage]:
        if timed_message is None:
            return None
        if now - timed_message.received_wall_time > float(max_age_s):
            return None
        return timed_message


__all__ = [
    "ATTITUDE",
    "LOCAL_POSITION_NED",
    "ODOMETRY",
    "CompetitionStateAdapter",
    "CompetitionStateAdapterError",
    "CompetitionStateResult",
    "TimedMavlinkMessage",
    "ned_position_to_internal_z_up",
    "ned_velocity_to_internal_z_up",
    "vehicle_state_from_local_position_ned",
    "vehicle_state_from_odometry",
    "yaw_from_attitude",
    "yaw_from_quaternion_wxyz",
]
