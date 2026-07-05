"""Receive-only MAVLink telemetry inventory tool for Phase 4A.

Importing this module does not open sockets or import pymavlink. Running the
CLI opens a receive-only MAVLink connection and records message inventory; it
does not send commands, heartbeats, or setpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


TIMESTAMP_FIELD_NAMES = (
    "time_boot_ms",
    "time_usec",
    "timestamp",
    "timestamp_sample",
    "ts1",
    "tc1",
)


def _message_type(message: Any) -> str:
    get_type = getattr(message, "get_type", None)
    if callable(get_type):
        return str(get_type())
    return str(getattr(message, "mavpackettype", type(message).__name__))


def _message_id(message: Any) -> Optional[int]:
    get_msg_id = getattr(message, "get_msgId", None)
    if callable(get_msg_id):
        return int(get_msg_id())
    msg_id = getattr(message, "msgid", None)
    return None if msg_id is None else int(msg_id)


def _source_system(message: Any) -> Optional[int]:
    get_src_system = getattr(message, "get_srcSystem", None)
    if callable(get_src_system):
        return int(get_src_system())
    src_system = getattr(message, "_srcSystem", None)
    return None if src_system is None else int(src_system)


def _source_component(message: Any) -> Optional[int]:
    get_src_component = getattr(message, "get_srcComponent", None)
    if callable(get_src_component):
        return int(get_src_component())
    src_component = getattr(message, "_srcComponent", None)
    return None if src_component is None else int(src_component)


def _message_fields(message: Any) -> dict[str, Any]:
    to_dict = getattr(message, "to_dict", None)
    if callable(to_dict):
        raw = dict(to_dict())
    else:
        raw = {
            key: value
            for key, value in vars(message).items()
            if not key.startswith("_")
        }
    raw.pop("mavpackettype", None)
    return raw


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    return repr(value)


@dataclass
class MessageInventory:
    message_type: str
    message_id: Optional[int]
    count: int = 0
    first_seen_wall_time: float = 0.0
    last_seen_wall_time: float = 0.0
    system_ids: set[int] = field(default_factory=set)
    component_ids: set[int] = field(default_factory=set)
    field_names: set[str] = field(default_factory=set)
    timestamp_fields: set[str] = field(default_factory=set)
    samples: list[dict[str, Any]] = field(default_factory=list)

    def record(self, message: Any, *, received_wall_time: float, sample_limit: int) -> None:
        if self.count == 0:
            self.first_seen_wall_time = received_wall_time
        self.count += 1
        self.last_seen_wall_time = received_wall_time

        system_id = _source_system(message)
        component_id = _source_component(message)
        if system_id is not None:
            self.system_ids.add(system_id)
        if component_id is not None:
            self.component_ids.add(component_id)

        fields = _message_fields(message)
        self.field_names.update(fields)
        self.timestamp_fields.update(
            field_name
            for field_name in fields
            if field_name in TIMESTAMP_FIELD_NAMES
        )
        if len(self.samples) < sample_limit:
            self.samples.append({key: _jsonable(value) for key, value in fields.items()})

    def observed_rate_hz(self) -> float:
        if self.count < 2:
            return 0.0
        elapsed = self.last_seen_wall_time - self.first_seen_wall_time
        if elapsed <= 0.0:
            return 0.0
        return (self.count - 1) / elapsed

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_type": self.message_type,
            "message_id": self.message_id,
            "count": self.count,
            "observed_rate_hz": self.observed_rate_hz(),
            "first_seen_wall_time": self.first_seen_wall_time,
            "last_seen_wall_time": self.last_seen_wall_time,
            "system_ids": sorted(self.system_ids),
            "component_ids": sorted(self.component_ids),
            "field_names": sorted(self.field_names),
            "timestamp_fields": sorted(self.timestamp_fields),
            "samples": self.samples,
        }


class MavlinkTelemetryInventory:
    """In-memory MAVLink message inventory for receive-only observe mode."""

    def __init__(self, *, sample_limit_per_type: int = 3):
        self.sample_limit_per_type = int(sample_limit_per_type)
        self.messages: dict[str, MessageInventory] = {}

    def observe_message(
        self,
        message: Any,
        *,
        received_wall_time: Optional[float] = None,
    ) -> None:
        message_type = _message_type(message)
        if message_type == "BAD_DATA":
            return
        message_id = _message_id(message)
        now = time.time() if received_wall_time is None else float(received_wall_time)
        inventory = self.messages.get(message_type)
        if inventory is None:
            inventory = MessageInventory(
                message_type=message_type,
                message_id=message_id,
            )
            self.messages[message_type] = inventory
        inventory.record(
            message,
            received_wall_time=now,
            sample_limit=self.sample_limit_per_type,
        )

    def summary(self) -> dict[str, Any]:
        message_summaries = {
            message_type: inventory.to_dict()
            for message_type, inventory in sorted(self.messages.items())
        }
        return {
            "receive_only": True,
            "message_types": message_summaries,
            "local_position_ned_available": "LOCAL_POSITION_NED" in self.messages,
            "odometry_available": "ODOMETRY" in self.messages,
            "attitude_available": "ATTITUDE" in self.messages,
        }


def open_mavlink_connection(endpoint: str):
    from pymavlink import mavutil

    return mavutil.mavlink_connection(endpoint)


def run_receive_only_observe(
    *,
    endpoint: str = "udpin:127.0.0.1:14550",
    duration_s: float = 10.0,
    poll_sleep_s: float = 0.001,
    connection: Any = None,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run receive-only telemetry inventory. No commands or heartbeats are sent."""

    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if poll_sleep_s < 0.0:
        raise ValueError("poll_sleep_s must be non-negative")

    mavlink_connection = connection
    if mavlink_connection is None:
        mavlink_connection = open_mavlink_connection(endpoint)

    inventory = MavlinkTelemetryInventory()
    started = float(clock())
    deadline = started + float(duration_s)

    while float(clock()) < deadline:
        message = mavlink_connection.recv_match(blocking=False)
        now = float(clock())
        if message is None:
            if poll_sleep_s > 0.0:
                sleep(poll_sleep_s)
            continue
        inventory.observe_message(message, received_wall_time=now)

    summary = inventory.summary()
    summary.update(
        {
            "endpoint": endpoint,
            "duration_s": float(duration_s),
            "started_wall_time": started,
            "finished_wall_time": float(clock()),
        }
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Receive-only MAVLink telemetry inventory for competition adapter "
            "Phase 4A. This does not send commands, heartbeats, or setpoints."
        )
    )
    parser.add_argument(
        "--endpoint",
        default="udpin:127.0.0.1:14550",
        help="pymavlink endpoint, default: udpin:127.0.0.1:14550",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=10.0,
        help="receive-only observe duration in seconds",
    )
    parser.add_argument(
        "--poll-sleep-s",
        type=float,
        default=0.001,
        help="sleep interval when no message is available",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = run_receive_only_observe(
        endpoint=args.endpoint,
        duration_s=args.duration_s,
        poll_sleep_s=args.poll_sleep_s,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
