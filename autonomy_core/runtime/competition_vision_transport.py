"""Import-safe UDP vision receive transport for the competition runtime.

Phase 6B only receives raw VADR UDP datagrams and surfaces packet bytes to
`CompetitionRunner`. JPEG decoding and header validation remain in
`CompetitionImageAdapter`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from autonomy_core.core.competition_config import VADR_TS_002


DEFAULT_VISION_BIND_HOST = "0.0.0.0"
DEFAULT_VISION_PORT = VADR_TS_002.vision_udp_port
DEFAULT_RECV_BUFFER_SIZE = 65536


class CompetitionVisionTransportError(RuntimeError):
    """Raised when the UDP vision receive transport cannot operate safely."""


@dataclass(frozen=True)
class CompetitionVisionTransportConfig:
    bind_host: str = DEFAULT_VISION_BIND_HOST
    port: int = DEFAULT_VISION_PORT
    recv_buffer_size: int = DEFAULT_RECV_BUFFER_SIZE
    max_packets_per_poll: int = 128
    poll_sleep_s: float = 0.001


@dataclass
class CompetitionVisionTransportStats:
    started: bool = False
    packets_received: int = 0
    bytes_received: int = 0
    receive_errors: int = 0
    receive_timeouts: int = 0
    last_packet_wall_time: Optional[float] = None
    source_address_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "started": self.started,
            "packets_received": self.packets_received,
            "bytes_received": self.bytes_received,
            "receive_errors": self.receive_errors,
            "receive_timeouts": self.receive_timeouts,
            "last_packet_wall_time": self.last_packet_wall_time,
            "source_address_counts": dict(sorted(self.source_address_counts.items())),
        }


class CompetitionVisionTransport:
    """Nonblocking UDP receive transport for raw VADR vision packets."""

    def __init__(
        self,
        *,
        config: CompetitionVisionTransportConfig = CompetitionVisionTransportConfig(),
        socket_obj: Any = None,
        socket_factory: Optional[Callable[[str, int, int], Any]] = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if config.port <= 0 or config.port > 65535:
            raise ValueError("port must be in [1, 65535]")
        if config.recv_buffer_size <= 0:
            raise ValueError("recv_buffer_size must be positive")
        if config.max_packets_per_poll <= 0:
            raise ValueError("max_packets_per_poll must be positive")
        if config.poll_sleep_s < 0.0:
            raise ValueError("poll_sleep_s must be non-negative")

        self.config = config
        self._socket = socket_obj
        self._socket_factory = socket_factory
        self._owns_socket = socket_obj is None
        self.clock = clock
        self.sleep = sleep
        self.stats = CompetitionVisionTransportStats(started=socket_obj is not None)

    @property
    def is_started(self) -> bool:
        return self._socket is not None

    def start(self) -> "CompetitionVisionTransport":
        """Bind the UDP vision socket if not already injected."""

        if self._socket is None:
            factory = self._socket_factory or open_udp_vision_socket
            self._socket = factory(
                self.config.bind_host,
                self.config.port,
                self.config.recv_buffer_size,
            )
            self._owns_socket = True
        self.stats.started = True
        return self

    def close(self) -> None:
        close = getattr(self._socket, "close", None)
        if self._socket is not None and self._owns_socket and callable(close):
            close()
        self._socket = None
        self.stats.started = False

    def receive_packets(
        self,
        *,
        max_packets: Optional[int] = None,
        duration_s: float = 0.0,
    ) -> tuple[bytes, ...]:
        """Poll raw UDP datagrams without parsing image payloads."""

        if duration_s < 0.0:
            raise ValueError("duration_s must be non-negative")
        limit = self.config.max_packets_per_poll if max_packets is None else int(max_packets)
        if limit <= 0:
            raise ValueError("max_packets must be positive")

        self.start()
        packets: list[bytes] = []
        deadline = None if duration_s <= 0.0 else float(self.clock()) + float(duration_s)

        while len(packets) < limit:
            try:
                packet, address = self._socket.recvfrom(self.config.recv_buffer_size)
            except (BlockingIOError, TimeoutError):
                self.stats.receive_timeouts += 1
                now = float(self.clock())
                if deadline is not None and now < deadline:
                    if self.config.poll_sleep_s > 0.0:
                        self.sleep(self.config.poll_sleep_s)
                    continue
                break
            except OSError:
                self.stats.receive_errors += 1
                break

            now = float(self.clock())
            packet_bytes = bytes(packet)
            self._record_packet(packet_bytes, address, now)
            packets.append(packet_bytes)

        return tuple(packets)

    def summary(self) -> dict[str, Any]:
        return {
            "bind_host": self.config.bind_host,
            "port": self.config.port,
            "receive_only": True,
            "stats": self.stats.to_dict(),
        }

    def _record_packet(self, packet: bytes, address: Any, now: float) -> None:
        address_key = _address_key(address)
        self.stats.packets_received += 1
        self.stats.bytes_received += len(packet)
        self.stats.last_packet_wall_time = now
        self.stats.source_address_counts[address_key] = (
            self.stats.source_address_counts.get(address_key, 0) + 1
        )


def open_udp_vision_socket(bind_host: str, port: int, _recv_buffer_size: int):
    """Bind a UDP socket lazily for explicit live execution only."""

    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((bind_host, int(port)))
    sock.setblocking(False)
    return sock


def _address_key(address: Any) -> str:
    if isinstance(address, tuple) and len(address) >= 2:
        return f"{address[0]}:{address[1]}"
    return str(address)


__all__ = [
    "DEFAULT_RECV_BUFFER_SIZE",
    "DEFAULT_VISION_BIND_HOST",
    "DEFAULT_VISION_PORT",
    "CompetitionVisionTransport",
    "CompetitionVisionTransportConfig",
    "CompetitionVisionTransportError",
    "CompetitionVisionTransportStats",
    "open_udp_vision_socket",
]
