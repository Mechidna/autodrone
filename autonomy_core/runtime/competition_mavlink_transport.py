"""Import-safe MAVLink receive transport for the competition runtime.

Phase 6B only receives MAVLink messages. It does not send heartbeats,
setpoints, attitude targets, actuator commands, arm/offboard/reset commands,
or any other MAVLink command.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from autonomy_core.tools.competition_mavlink_observe import MavlinkTelemetryInventory


DEFAULT_MAVLINK_ENDPOINT = "udpin:0.0.0.0:14540"


class CompetitionMavlinkTransportError(RuntimeError):
    """Raised when the MAVLink receive transport cannot operate safely."""


@dataclass(frozen=True)
class CompetitionMavlinkTransportConfig:
    endpoint: str = DEFAULT_MAVLINK_ENDPOINT
    poll_sleep_s: float = 0.001
    max_messages_per_poll: int = 64
    sample_limit_per_type: int = 3


@dataclass
class CompetitionMavlinkTransportStats:
    started: bool = False
    messages_received: int = 0
    bad_data_ignored: int = 0
    receive_errors: int = 0
    heartbeat_count: int = 0
    last_message_wall_time: Optional[float] = None
    last_heartbeat_wall_time: Optional[float] = None
    message_type_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "started": self.started,
            "messages_received": self.messages_received,
            "bad_data_ignored": self.bad_data_ignored,
            "receive_errors": self.receive_errors,
            "heartbeat_count": self.heartbeat_count,
            "last_message_wall_time": self.last_message_wall_time,
            "last_heartbeat_wall_time": self.last_heartbeat_wall_time,
            "message_type_counts": dict(sorted(self.message_type_counts.items())),
        }


class CompetitionMavlinkTransport:
    """Nonblocking MAVLink receive-only transport.

    The transport is dependency-injected into `CompetitionRunner` via its
    `receive_messages()` method. It deliberately has no send API.
    """

    def __init__(
        self,
        *,
        config: CompetitionMavlinkTransportConfig = CompetitionMavlinkTransportConfig(),
        connection: Any = None,
        connection_factory: Optional[Callable[[str], Any]] = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if config.poll_sleep_s < 0.0:
            raise ValueError("poll_sleep_s must be non-negative")
        if config.max_messages_per_poll <= 0:
            raise ValueError("max_messages_per_poll must be positive")
        if config.sample_limit_per_type <= 0:
            raise ValueError("sample_limit_per_type must be positive")

        self.config = config
        self._connection = connection
        self._connection_factory = connection_factory
        self._owns_connection = connection is None
        self.clock = clock
        self.sleep = sleep
        self.stats = CompetitionMavlinkTransportStats(started=connection is not None)
        self.inventory = MavlinkTelemetryInventory(
            sample_limit_per_type=config.sample_limit_per_type
        )

    @property
    def is_started(self) -> bool:
        return self._connection is not None

    @property
    def active_connection(self) -> Any:
        """Return the already-open connection without opening a socket."""

        return self._connection

    def start(self) -> "CompetitionMavlinkTransport":
        """Open the configured MAVLink endpoint if not already injected."""

        if self._connection is None:
            factory = self._connection_factory or open_mavlink_connection
            self._connection = factory(self.config.endpoint)
            self._owns_connection = True
        self.stats.started = True
        return self

    def close(self) -> None:
        close = getattr(self._connection, "close", None)
        if self._connection is not None and self._owns_connection and callable(close):
            close()
        self._connection = None
        self.stats.started = False

    def receive_messages(
        self,
        *,
        max_messages: Optional[int] = None,
        duration_s: float = 0.0,
    ) -> tuple[Any, ...]:
        """Poll MAVLink messages without sending anything."""

        if duration_s < 0.0:
            raise ValueError("duration_s must be non-negative")
        limit = self.config.max_messages_per_poll if max_messages is None else int(max_messages)
        if limit <= 0:
            raise ValueError("max_messages must be positive")

        self.start()
        messages: list[Any] = []
        deadline = None if duration_s <= 0.0 else float(self.clock()) + float(duration_s)

        while len(messages) < limit:
            try:
                message = self._connection.recv_match(blocking=False)
            except ConnectionResetError:
                self.stats.receive_errors += 1
                break
            except OSError:
                self.stats.receive_errors += 1
                break

            now = float(self.clock())
            if message is None:
                if deadline is not None and now < deadline:
                    if self.config.poll_sleep_s > 0.0:
                        self.sleep(self.config.poll_sleep_s)
                    continue
                break

            message_type = _message_type(message)
            if message_type == "BAD_DATA":
                self.stats.bad_data_ignored += 1
                continue

            self._record_message(message, message_type, now)
            messages.append(message)

        return tuple(messages)

    def summary(self) -> dict[str, Any]:
        summary = self.inventory.summary()
        summary.update(
            {
                "endpoint": self.config.endpoint,
                "receive_only": True,
                "stats": self.stats.to_dict(),
            }
        )
        return summary

    def _record_message(self, message: Any, message_type: str, now: float) -> None:
        self.inventory.observe_message(message, received_wall_time=now)
        self.stats.messages_received += 1
        self.stats.last_message_wall_time = now
        self.stats.message_type_counts[message_type] = (
            self.stats.message_type_counts.get(message_type, 0) + 1
        )
        if message_type == "HEARTBEAT":
            self.stats.heartbeat_count += 1
            self.stats.last_heartbeat_wall_time = now


class _MavlinkModuleLoadError(CompetitionMavlinkTransportError):
    pass


def open_mavlink_connection(endpoint: str):
    """Open a pymavlink connection lazily for explicit live execution only."""

    try:
        from pymavlink import mavutil
    except ModuleNotFoundError as exc:
        raise _MavlinkModuleLoadError(
            "pymavlink is required only when starting the live MAVLink transport"
        ) from exc
    return mavutil.mavlink_connection(endpoint)


def _message_type(message: Any) -> str:
    get_type = getattr(message, "get_type", None)
    if callable(get_type):
        return str(get_type())
    if isinstance(message, dict) and "type" in message:
        return str(message["type"])
    return str(getattr(message, "mavpackettype", type(message).__name__))


__all__ = [
    "DEFAULT_MAVLINK_ENDPOINT",
    "CompetitionMavlinkTransport",
    "CompetitionMavlinkTransportConfig",
    "CompetitionMavlinkTransportError",
    "CompetitionMavlinkTransportStats",
    "open_mavlink_connection",
]
