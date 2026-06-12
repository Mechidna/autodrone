"""Import-safe competition runner skeleton with injected fake transports only.

Phase 6A intentionally does not implement live MAVLink, live UDP vision,
command publication, race mode, or real AutonomyAPI ownership. Callers may
inject fake transports and fake autonomy objects for deterministic tests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Optional, Sequence

from autonomy_core.command.competition_command_adapter import (
    CompetitionDryRunCommandAdapter,
    DryRunCommandResult,
)
from autonomy_core.core.competition_config import RuntimeCompetitionConfig, VADR_TS_002
from autonomy_core.core.competition_state_adapter import (
    CompetitionStateAdapter,
    CompetitionStateResult,
)
from autonomy_core.perception.competition_image_adapter import (
    CompetitionCameraFrame,
    CompetitionImageAdapter,
    CompetitionImageAdapterError,
)
from autonomy_core.runtime.competition_guard import (
    CompetitionGuard,
    CompetitionGuardError,
)
from autonomy_core.tools.competition_mavlink_observe import MavlinkTelemetryInventory


class CompetitionRunnerMode(str, Enum):
    OBSERVE = "observe"
    VISION_DRY_RUN = "vision_dry_run"
    COMMAND_DRY_RUN = "command_dry_run"
    COMMAND_LIVE = "command_live"
    RACE = "race"


LIVE_COMMAND_MODES = frozenset(
    {
        CompetitionRunnerMode.COMMAND_LIVE,
        CompetitionRunnerMode.RACE,
    }
)

VISION_MODES = frozenset(
    {
        CompetitionRunnerMode.VISION_DRY_RUN,
        CompetitionRunnerMode.COMMAND_DRY_RUN,
    }
)

COMMAND_CANDIDATE_MODES = frozenset({CompetitionRunnerMode.COMMAND_DRY_RUN})


class CompetitionRunnerSafetyError(RuntimeError):
    """Raised when a requested runner mode violates Phase 6A safety gates."""


@dataclass(frozen=True)
class CompetitionRunnerSafetyConfig:
    """Fail-closed switches for the Phase 6A runner skeleton."""

    phase4b_telemetry_evidence_available: bool = False
    command_publication_enabled: bool = False
    allow_live_command_modes: bool = False
    require_fresh_state_for_command_candidate: bool = True
    max_state_age_s: float = 0.5


@dataclass(frozen=True)
class CompetitionRunnerConfig:
    mode: CompetitionRunnerMode | str = CompetitionRunnerMode.OBSERVE
    target_system: int = 1
    target_component: int = 1
    perception_world_pose_source: Optional[str] = None
    safety: CompetitionRunnerSafetyConfig = field(
        default_factory=CompetitionRunnerSafetyConfig
    )


@dataclass
class CompetitionRunnerStats:
    steps: int = 0
    telemetry_messages_processed: int = 0
    heartbeats_received: int = 0
    vision_packets_processed: int = 0
    vision_frames_completed: int = 0
    perception_update_calls: int = 0
    command_candidate_attempts: int = 0
    command_candidates_accepted: int = 0
    command_rejections: int = 0
    command_publications_attempted: int = 0
    command_publications_sent: int = 0


@dataclass(frozen=True)
class CompetitionRunnerStepResult:
    mode: CompetitionRunnerMode
    now_s: float
    telemetry_messages_processed: int
    heartbeat_seen: bool
    heartbeat_age_s: Optional[float]
    state_result: CompetitionStateResult
    vision_packets_processed: int
    vision_frames_completed: int
    perception_update_calls: int
    command_candidate_attempted: bool
    command_result: Optional[DryRunCommandResult]
    command_publication_allowed: bool
    command_blocked_reasons: tuple[str, ...]
    events: tuple[str, ...]


class CompetitionRunner:
    """Coordinate competition adapters through injected transports.

    This skeleton has no live transport implementation. If `mavlink_transport`
    or `vision_transport` is supplied, it must be an injected fake/test object
    exposing `receive_messages()` or `receive_packets()` respectively.
    """

    def __init__(
        self,
        *,
        config: CompetitionRunnerConfig = CompetitionRunnerConfig(),
        competition_config: RuntimeCompetitionConfig = VADR_TS_002,
        guard: Optional[CompetitionGuard] = None,
        state_adapter: Optional[CompetitionStateAdapter] = None,
        image_adapter: Optional[CompetitionImageAdapter] = None,
        command_adapter: Optional[CompetitionDryRunCommandAdapter] = None,
        telemetry_inventory: Optional[MavlinkTelemetryInventory] = None,
        autonomy: Any = None,
        mavlink_transport: Any = None,
        vision_transport: Any = None,
        clock: Callable[[], float] = time.time,
    ):
        self.config = config
        self.mode = normalize_runner_mode(config.mode)
        self.competition_config = competition_config
        self.guard = CompetitionGuard() if guard is None else guard
        self.state_adapter = (
            CompetitionStateAdapter(clock=clock)
            if state_adapter is None
            else state_adapter
        )
        self.image_adapter = (
            CompetitionImageAdapter(clock=clock)
            if image_adapter is None
            else image_adapter
        )
        self.command_adapter = (
            CompetitionDryRunCommandAdapter(config=competition_config)
            if command_adapter is None
            else command_adapter
        )
        self.telemetry_inventory = (
            MavlinkTelemetryInventory()
            if telemetry_inventory is None
            else telemetry_inventory
        )
        self.autonomy = autonomy
        self.mavlink_transport = mavlink_transport
        self.vision_transport = vision_transport
        self.clock = clock
        self.stats = CompetitionRunnerStats()
        self.last_heartbeat_wall_time: Optional[float] = None

        self._assert_startup_safety()

    def step(
        self,
        *,
        telemetry_messages: Optional[Iterable[Any]] = None,
        vision_packets: Optional[Iterable[bytes]] = None,
    ) -> CompetitionRunnerStepResult:
        """Process one injected/fake batch without opening sockets or sending."""

        now = float(self.clock())
        self.stats.steps += 1
        events: list[str] = []

        self.guard.assert_competition_safe(
            perception_world_pose_source=self.config.perception_world_pose_source,
            runner_inputs={
                "telemetry_messages": telemetry_messages,
                "vision_packets": vision_packets,
            },
            mode=self.mode.value,
            command_enabled=False,
        )

        telemetry_batch = list(
            telemetry_messages
            if telemetry_messages is not None
            else _receive_from_injected_transport(
                self.mavlink_transport,
                method_name="receive_messages",
            )
        )
        telemetry_processed = self._process_telemetry(telemetry_batch, now, events)

        state_result = self.state_adapter.latest_result(
            now=now,
            max_age_s=self.config.safety.max_state_age_s,
        )

        vision_batch = []
        if self.mode in VISION_MODES:
            vision_batch = list(
                vision_packets
                if vision_packets is not None
                else _receive_from_injected_transport(
                    self.vision_transport,
                    method_name="receive_packets",
                )
            )
        elif vision_packets:
            events.append("vision_packets_ignored_in_mode")

        vision_completed, perception_updates = self._process_vision(
            vision_batch,
            events,
        )

        command_result: Optional[DryRunCommandResult] = None
        command_candidate_attempted = False
        command_blocked_reasons = self._command_blocked_reasons(state_result)

        if self.mode in COMMAND_CANDIDATE_MODES:
            command_candidate_attempted, command_result = self._build_command_candidate(
                state_result,
                now,
                events,
            )
            if command_result is not None and not command_result.accepted:
                command_blocked_reasons = tuple(
                    [*command_blocked_reasons, command_result.rejection_reason]
                )

        result = CompetitionRunnerStepResult(
            mode=self.mode,
            now_s=now,
            telemetry_messages_processed=telemetry_processed,
            heartbeat_seen=self.last_heartbeat_wall_time is not None,
            heartbeat_age_s=self.heartbeat_age_s(now),
            state_result=state_result,
            vision_packets_processed=len(vision_batch),
            vision_frames_completed=vision_completed,
            perception_update_calls=perception_updates,
            command_candidate_attempted=command_candidate_attempted,
            command_result=command_result,
            command_publication_allowed=False,
            command_blocked_reasons=tuple(command_blocked_reasons),
            events=tuple(events),
        )
        return result

    def heartbeat_age_s(self, now_s: Optional[float] = None) -> Optional[float]:
        if self.last_heartbeat_wall_time is None:
            return None
        now = float(self.clock() if now_s is None else now_s)
        return now - self.last_heartbeat_wall_time

    def telemetry_summary(self) -> dict[str, Any]:
        return self.telemetry_inventory.summary()

    def _assert_startup_safety(self) -> None:
        safety = self.config.safety
        if self.mode in LIVE_COMMAND_MODES:
            raise CompetitionRunnerSafetyError(
                f"{self.mode.value} is not enabled in the Phase 6A runner skeleton"
            )
        if safety.allow_live_command_modes:
            raise CompetitionRunnerSafetyError(
                "Phase 6A does not allow live command modes"
            )
        if safety.command_publication_enabled:
            raise CompetitionRunnerSafetyError(
                "Phase 6A does not allow command publication"
            )
        if self.mode == CompetitionRunnerMode.COMMAND_DRY_RUN:
            if not safety.phase4b_telemetry_evidence_available:
                # This mode may still build fake dry-run candidates, but it cannot
                # be interpreted as command readiness.
                pass

        try:
            self.guard.assert_competition_safe(
                perception_world_pose_source=self.config.perception_world_pose_source,
                mode=self.mode.value,
                command_enabled=False,
            )
        except CompetitionGuardError as exc:
            raise CompetitionRunnerSafetyError(str(exc)) from exc

    def _process_telemetry(
        self,
        messages: Sequence[Any],
        now: float,
        events: list[str],
    ) -> int:
        processed = 0
        for message in messages:
            message_type = _message_type(message)
            self.telemetry_inventory.observe_message(
                message,
                received_wall_time=now,
            )
            if message_type == "HEARTBEAT":
                self.last_heartbeat_wall_time = now
                self.stats.heartbeats_received += 1
            self.state_adapter.ingest_message(message, received_wall_time=now)
            processed += 1

        self.stats.telemetry_messages_processed += processed
        if processed == 0:
            events.append("no_telemetry_messages")
        return processed

    def _process_vision(
        self,
        packets: Sequence[bytes],
        events: list[str],
    ) -> tuple[int, int]:
        completed = 0
        perception_updates = 0
        for packet in packets:
            self.stats.vision_packets_processed += 1
            try:
                frame = self.image_adapter.process_packet(packet)
            except CompetitionImageAdapterError as exc:
                events.append(f"vision_packet_rejected:{exc}")
                continue
            if frame is None:
                continue
            completed += 1
            self.stats.vision_frames_completed += 1
            perception_updates += self._update_gate_memory_if_injected(frame, events)

        return completed, perception_updates

    def _update_gate_memory_if_injected(
        self,
        frame: CompetitionCameraFrame,
        events: list[str],
    ) -> int:
        if self.autonomy is None:
            events.append("no_autonomy_for_perception_update")
            return 0

        update_gate_memory = getattr(
            self.autonomy,
            "update_gate_memory_from_frame",
            None,
        )
        if not callable(update_gate_memory):
            events.append("autonomy_missing_update_gate_memory_from_frame")
            return 0

        kwargs = self.guard.perception_update_kwargs(
            **frame.update_gate_memory_kwargs()
        )
        update_gate_memory(**kwargs)
        self.stats.perception_update_calls += 1
        return 1

    def _build_command_candidate(
        self,
        state_result: CompetitionStateResult,
        now: float,
        events: list[str],
    ) -> tuple[bool, Optional[DryRunCommandResult]]:
        if (
            self.config.safety.require_fresh_state_for_command_candidate
            and not state_result.is_usable
        ):
            self.stats.command_rejections += 1
            events.append("command_candidate_blocked_invalid_state")
            return False, None

        if self.autonomy is None:
            self.stats.command_rejections += 1
            events.append("command_candidate_blocked_missing_autonomy")
            return False, None

        attitude_control = getattr(self.autonomy, "attitude_control", None)
        if not callable(attitude_control):
            self.stats.command_rejections += 1
            events.append("command_candidate_blocked_missing_attitude_control")
            return False, None

        self.stats.command_candidate_attempts += 1
        command_tuple = attitude_control()
        result = self.command_adapter.build_set_attitude_target(
            command_tuple,
            time_boot_ms=_time_boot_ms_from_wall_time(now),
            target_system=self.config.target_system,
            target_component=self.config.target_component,
            now_s=now,
            sequence=self.stats.command_candidate_attempts,
        )
        if result.accepted:
            self.stats.command_candidates_accepted += 1
        else:
            self.stats.command_rejections += 1
        return True, result

    def _command_blocked_reasons(
        self,
        state_result: CompetitionStateResult,
    ) -> tuple[str, ...]:
        reasons = ["phase6a_no_command_publication"]
        if not self.config.safety.phase4b_telemetry_evidence_available:
            reasons.append("phase4b_telemetry_evidence_missing")
        if self.mode == CompetitionRunnerMode.COMMAND_DRY_RUN:
            reasons.append("command_dry_run_no_send")
        else:
            reasons.append(f"{self.mode.value}_mode_no_commands")
        if not state_result.is_usable:
            reasons.extend(state_result.missing_reasons)
        return tuple(reasons)


def normalize_runner_mode(mode: CompetitionRunnerMode | str) -> CompetitionRunnerMode:
    if isinstance(mode, CompetitionRunnerMode):
        return mode
    try:
        return CompetitionRunnerMode(str(mode))
    except ValueError as exc:
        raise CompetitionRunnerSafetyError(f"unknown competition runner mode: {mode}") from exc


def _message_type(message: Any) -> str:
    get_type = getattr(message, "get_type", None)
    if callable(get_type):
        return str(get_type())
    if isinstance(message, dict) and "type" in message:
        return str(message["type"])
    return str(getattr(message, "mavpackettype", type(message).__name__))


def _receive_from_injected_transport(
    transport: Any,
    *,
    method_name: str,
) -> tuple[Any, ...]:
    if transport is None:
        return ()
    method = getattr(transport, method_name, None)
    if not callable(method):
        raise CompetitionRunnerSafetyError(
            f"injected transport must expose {method_name}()"
        )
    return tuple(method())


def _time_boot_ms_from_wall_time(now_s: float) -> int:
    return int(max(0.0, float(now_s)) * 1000.0) & 0xFFFFFFFF


__all__ = [
    "COMMAND_CANDIDATE_MODES",
    "LIVE_COMMAND_MODES",
    "VISION_MODES",
    "CompetitionRunner",
    "CompetitionRunnerConfig",
    "CompetitionRunnerMode",
    "CompetitionRunnerSafetyConfig",
    "CompetitionRunnerSafetyError",
    "CompetitionRunnerStats",
    "CompetitionRunnerStepResult",
    "normalize_runner_mode",
]
