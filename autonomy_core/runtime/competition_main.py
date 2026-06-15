"""Import-safe competition executable lifecycle for Phase 6D/6E.

This module parses/executes bounded dry-run modes around `competition_setup.py`.
It does not send commands, enable race mode, or start live transports unless the
CLI explicitly requests `--live-transports`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from autonomy_core.runtime.competition_mavlink_transport import (
    CompetitionMavlinkTransportConfig,
    DEFAULT_MAVLINK_ENDPOINT,
)
from autonomy_core.runtime.competition_runner import (
    CompetitionRunnerMode,
    CompetitionRunnerSafetyConfig,
)
from autonomy_core.runtime.competition_setup import (
    CompetitionRuntimeComponents,
    CompetitionSetupConfig,
    build_competition_runtime,
)
from autonomy_core.runtime.competition_vision_transport import (
    CompetitionVisionTransportConfig,
    DEFAULT_VISION_BIND_HOST,
    DEFAULT_VISION_PORT,
)


PHASE_6D = "6D"
PHASE_6E = "6E"
PHASE4B_NOT_SATISFIED = "Phase 4B remains blocked pending real competition telemetry evidence."
COMPETITION_READINESS_NOT_CLAIMED = "Dry-run output is not competition readiness evidence."
PHASE6E_SURROGATE_LIMITATION = (
    "Phase 6E live receive dry-run can use PX4/Gazebo surrogate evidence, but "
    "that does not satisfy Phase 4B or Phase 9 real competition simulator evidence."
)


class CompetitionMainSafetyError(RuntimeError):
    """Raised when the Phase 6D executable would violate safety gates."""


@dataclass(frozen=True)
class CompetitionMainConfig:
    mode: CompetitionRunnerMode | str = CompetitionRunnerMode.OBSERVE
    steps: int = 1
    duration_s: float = 0.0
    step_sleep_s: float = 0.0
    live_transports: bool = False
    use_real_autonomy: bool = False
    target_system: int = 1
    target_component: int = 1
    mavlink_endpoint: str = DEFAULT_MAVLINK_ENDPOINT
    mavlink_max_messages_per_poll: int = 64
    vision_bind_host: str = DEFAULT_VISION_BIND_HOST
    vision_port: int = DEFAULT_VISION_PORT
    vision_max_packets_per_poll: int = 128
    evidence_label: str = "unspecified"


@dataclass
class CompetitionMainSummary:
    phase: str = PHASE_6D
    mode: str = CompetitionRunnerMode.OBSERVE.value
    status: str = "not_started"
    fail_closed: bool = False
    safety_error: Optional[str] = None
    evidence_label: str = "unspecified"
    live_transports_requested: bool = False
    use_real_autonomy: bool = False
    steps_requested: int = 0
    steps_completed: int = 0
    duration_s: float = 0.0
    telemetry_messages_processed: int = 0
    heartbeat_seen: bool = False
    heartbeat_age_s: Optional[float] = None
    state_usable: bool = False
    state_missing_reasons: tuple[str, ...] = ()
    position_source: str = ""
    attitude_source: str = ""
    vision_packets_processed: int = 0
    vision_frames_completed: int = 0
    perception_update_calls: int = 0
    command_candidate_count: int = 0
    command_publication_allowed: bool = False
    command_sent_count: int = 0
    command_blocked_reasons: tuple[str, ...] = ()
    runner_events: tuple[str, ...] = ()
    mavlink_transport_summary: Optional[dict[str, Any]] = None
    vision_transport_summary: Optional[dict[str, Any]] = None
    phase4b_satisfied: bool = False
    phase6e_receive_satisfied: bool = False
    phase6e_perception_boundary_satisfied: bool = False
    phase6e_satisfied: bool = False
    phase6e_success_criteria: dict[str, bool] = field(default_factory=dict)
    competition_readiness_claimed: bool = False
    notes: tuple[str, ...] = (
        PHASE4B_NOT_SATISFIED,
        COMPETITION_READINESS_NOT_CLAIMED,
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "mode": self.mode,
            "status": self.status,
            "fail_closed": self.fail_closed,
            "safety_error": self.safety_error,
            "evidence_label": self.evidence_label,
            "live_transports_requested": self.live_transports_requested,
            "use_real_autonomy": self.use_real_autonomy,
            "steps_requested": self.steps_requested,
            "steps_completed": self.steps_completed,
            "duration_s": self.duration_s,
            "telemetry_messages_processed": self.telemetry_messages_processed,
            "heartbeat_seen": self.heartbeat_seen,
            "heartbeat_age_s": self.heartbeat_age_s,
            "state_usable": self.state_usable,
            "state_missing_reasons": list(self.state_missing_reasons),
            "position_source": self.position_source,
            "attitude_source": self.attitude_source,
            "vision_packets_processed": self.vision_packets_processed,
            "vision_frames_completed": self.vision_frames_completed,
            "perception_update_calls": self.perception_update_calls,
            "command_candidate_count": self.command_candidate_count,
            "command_publication_allowed": self.command_publication_allowed,
            "command_sent_count": self.command_sent_count,
            "command_blocked_reasons": list(self.command_blocked_reasons),
            "runner_events": list(self.runner_events),
            "mavlink_transport_summary": self.mavlink_transport_summary,
            "vision_transport_summary": self.vision_transport_summary,
            "phase4b_satisfied": self.phase4b_satisfied,
            "phase6e_receive_satisfied": self.phase6e_receive_satisfied,
            "phase6e_perception_boundary_satisfied": (
                self.phase6e_perception_boundary_satisfied
            ),
            "phase6e_satisfied": self.phase6e_satisfied,
            "phase6e_success_criteria": dict(sorted(self.phase6e_success_criteria.items())),
            "competition_readiness_claimed": self.competition_readiness_claimed,
            "notes": list(self.notes),
        }


def run_competition_main(
    config: CompetitionMainConfig = CompetitionMainConfig(),
    *,
    components: Optional[CompetitionRuntimeComponents] = None,
    components_factory: Callable[..., CompetitionRuntimeComponents] = build_competition_runtime,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> CompetitionMainSummary:
    """Run a bounded Phase 6D dry-run loop."""

    mode = _normalize_mode(config.mode)
    _assert_main_config_safe(config, mode, components=components)

    started = float(clock())
    active_components = components or components_factory(_setup_config_from_main(config, mode))
    steps_completed = 0
    aggregate = _Aggregate()
    deadline = None if config.duration_s <= 0.0 else started + float(config.duration_s)

    try:
        for _step_index in range(int(config.steps)):
            if deadline is not None and float(clock()) > deadline:
                break
            result = active_components.runner.step()
            aggregate.record(result)
            steps_completed += 1
            if config.step_sleep_s > 0.0:
                sleep(float(config.step_sleep_s))
    finally:
        close = getattr(active_components, "close", None)
        if callable(close):
            close()

    finished = float(clock())
    phase6e_criteria = _phase6e_success_criteria(
        config=config,
        mode=mode,
        aggregate=aggregate,
    )
    phase6e_receive_satisfied = _phase6e_receive_satisfied(phase6e_criteria, mode)
    phase6e_perception_boundary_satisfied = bool(
        phase6e_criteria.get("perception_update_calls_gt_0", False)
    )

    return CompetitionMainSummary(
        phase=_phase_for_config(config),
        mode=mode.value,
        status="dry_run_complete",
        evidence_label=str(config.evidence_label),
        live_transports_requested=bool(config.live_transports),
        use_real_autonomy=bool(config.use_real_autonomy),
        steps_requested=int(config.steps),
        steps_completed=steps_completed,
        duration_s=max(0.0, finished - started),
        telemetry_messages_processed=aggregate.telemetry_messages_processed,
        heartbeat_seen=aggregate.heartbeat_seen,
        heartbeat_age_s=aggregate.heartbeat_age_s,
        state_usable=aggregate.state_usable,
        state_missing_reasons=aggregate.state_missing_reasons,
        position_source=aggregate.position_source,
        attitude_source=aggregate.attitude_source,
        vision_packets_processed=aggregate.vision_packets_processed,
        vision_frames_completed=aggregate.vision_frames_completed,
        perception_update_calls=aggregate.perception_update_calls,
        command_candidate_count=aggregate.command_candidate_count,
        command_publication_allowed=False,
        command_sent_count=0,
        command_blocked_reasons=aggregate.command_blocked_reasons,
        runner_events=aggregate.runner_events,
        mavlink_transport_summary=_transport_summary(active_components.mavlink_transport),
        vision_transport_summary=_transport_summary(active_components.vision_transport),
        phase6e_receive_satisfied=phase6e_receive_satisfied,
        phase6e_perception_boundary_satisfied=phase6e_perception_boundary_satisfied,
        phase6e_satisfied=phase6e_receive_satisfied,
        phase6e_success_criteria=phase6e_criteria,
        notes=_notes_for_config(config),
    )


def fail_closed_summary(
    *,
    mode: CompetitionRunnerMode | str,
    error: str,
    live_transports_requested: bool = False,
    use_real_autonomy: bool = False,
) -> CompetitionMainSummary:
    normalized = _mode_value(mode)
    return CompetitionMainSummary(
        phase=PHASE_6E if live_transports_requested else PHASE_6D,
        mode=normalized,
        status="fail_closed",
        fail_closed=True,
        safety_error=str(error),
        evidence_label="fail_closed",
        live_transports_requested=bool(live_transports_requested),
        use_real_autonomy=bool(use_real_autonomy),
        command_publication_allowed=False,
        command_sent_count=0,
        command_blocked_reasons=("phase6d_fail_closed",),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 6D/6E competition main executable. Runs bounded dry-run "
            "modes around competition_setup.py. Commands are never sent."
        )
    )
    parser.add_argument(
        "mode",
        choices=[mode.value for mode in CompetitionRunnerMode],
        help="competition runner mode",
    )
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--step-sleep-s", type=float, default=0.0)
    parser.add_argument(
        "--live-transports",
        action="store_true",
        help="explicitly allow production receive transports to open sockets",
    )
    parser.add_argument(
        "--use-real-autonomy",
        action="store_true",
        help="explicitly construct real AutonomyAPI through competition_setup.py",
    )
    parser.add_argument("--target-system", type=int, default=1)
    parser.add_argument("--target-component", type=int, default=1)
    parser.add_argument("--mavlink-endpoint", default=DEFAULT_MAVLINK_ENDPOINT)
    parser.add_argument("--mavlink-max-messages-per-poll", type=int, default=64)
    parser.add_argument("--vision-bind-host", default=DEFAULT_VISION_BIND_HOST)
    parser.add_argument("--vision-port", type=int, default=DEFAULT_VISION_PORT)
    parser.add_argument("--vision-max-packets-per-poll", type=int, default=128)
    parser.add_argument(
        "--evidence-label",
        default="unspecified",
        help=(
            "operator label for the dry-run evidence, for example "
            "px4_gazebo_surrogate or competition_sim_observe"
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = CompetitionMainConfig(
        mode=args.mode,
        steps=int(args.steps),
        duration_s=float(args.duration_s),
        step_sleep_s=float(args.step_sleep_s),
        live_transports=bool(args.live_transports),
        use_real_autonomy=bool(args.use_real_autonomy),
        target_system=int(args.target_system),
        target_component=int(args.target_component),
        mavlink_endpoint=str(args.mavlink_endpoint),
        mavlink_max_messages_per_poll=int(args.mavlink_max_messages_per_poll),
        vision_bind_host=str(args.vision_bind_host),
        vision_port=int(args.vision_port),
        vision_max_packets_per_poll=int(args.vision_max_packets_per_poll),
        evidence_label=str(args.evidence_label),
    )
    try:
        summary = run_competition_main(config)
        exit_code = 0
    except CompetitionMainSafetyError as exc:
        summary = fail_closed_summary(
            mode=args.mode,
            error=str(exc),
            live_transports_requested=bool(args.live_transports),
            use_real_autonomy=bool(args.use_real_autonomy),
        )
        exit_code = 2

    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return exit_code


@dataclass
class _Aggregate:
    telemetry_messages_processed: int = 0
    heartbeat_seen: bool = False
    heartbeat_age_s: Optional[float] = None
    state_usable: bool = False
    state_missing_reasons: tuple[str, ...] = ()
    position_source: str = ""
    attitude_source: str = ""
    vision_packets_processed: int = 0
    vision_frames_completed: int = 0
    perception_update_calls: int = 0
    command_candidate_count: int = 0
    command_blocked_reasons: tuple[str, ...] = ()
    runner_events: tuple[str, ...] = ()

    def record(self, result: Any) -> None:
        self.telemetry_messages_processed += int(result.telemetry_messages_processed)
        self.heartbeat_seen = self.heartbeat_seen or bool(result.heartbeat_seen)
        self.heartbeat_age_s = result.heartbeat_age_s
        self.state_usable = bool(result.state_result.is_usable)
        self.state_missing_reasons = tuple(result.state_result.missing_reasons)
        self.position_source = str(result.state_result.position_source)
        self.attitude_source = str(result.state_result.attitude_source)
        self.vision_packets_processed += int(result.vision_packets_processed)
        self.vision_frames_completed += int(result.vision_frames_completed)
        self.perception_update_calls += int(result.perception_update_calls)
        if result.command_candidate_attempted:
            self.command_candidate_count += 1
        self.command_blocked_reasons = tuple(result.command_blocked_reasons)
        self.runner_events = tuple([*self.runner_events, *result.events])


def _setup_config_from_main(
    config: CompetitionMainConfig,
    mode: CompetitionRunnerMode,
) -> CompetitionSetupConfig:
    return CompetitionSetupConfig(
        mode=mode,
        target_system=config.target_system,
        target_component=config.target_component,
        use_real_autonomy=config.use_real_autonomy,
        runner_safety=CompetitionRunnerSafetyConfig(
            phase4b_telemetry_evidence_available=False,
            command_publication_enabled=False,
            allow_live_command_modes=False,
        ),
        mavlink_transport_config=CompetitionMavlinkTransportConfig(
            endpoint=config.mavlink_endpoint,
            max_messages_per_poll=config.mavlink_max_messages_per_poll,
        ),
        vision_transport_config=CompetitionVisionTransportConfig(
            bind_host=config.vision_bind_host,
            port=config.vision_port,
            max_packets_per_poll=config.vision_max_packets_per_poll,
        ),
    )


def _assert_main_config_safe(
    config: CompetitionMainConfig,
    mode: CompetitionRunnerMode,
    *,
    components: Optional[CompetitionRuntimeComponents],
) -> None:
    if int(config.steps) <= 0:
        raise CompetitionMainSafetyError("steps must be positive")
    if float(config.duration_s) < 0.0:
        raise CompetitionMainSafetyError("duration_s must be non-negative")
    if float(config.step_sleep_s) < 0.0:
        raise CompetitionMainSafetyError("step_sleep_s must be non-negative")
    if mode in {CompetitionRunnerMode.COMMAND_LIVE, CompetitionRunnerMode.RACE}:
        raise CompetitionMainSafetyError(
            f"{mode.value} is fail-closed in Phase 6D"
        )
    if components is None and not config.live_transports:
        raise CompetitionMainSafetyError(
            "Phase 6D requires injected components or explicit --live-transports"
        )


def _phase_for_config(config: CompetitionMainConfig) -> str:
    return PHASE_6E if config.live_transports else PHASE_6D


def _notes_for_config(config: CompetitionMainConfig) -> tuple[str, ...]:
    notes = [
        PHASE4B_NOT_SATISFIED,
        COMPETITION_READINESS_NOT_CLAIMED,
    ]
    if config.live_transports:
        notes.append(PHASE6E_SURROGATE_LIMITATION)
        notes.append(
            "Phase 6E validates production receive transports only; command "
            "publication remains disabled."
        )
    return tuple(notes)


def _phase6e_success_criteria(
    *,
    config: CompetitionMainConfig,
    mode: CompetitionRunnerMode,
    aggregate: _Aggregate,
) -> dict[str, bool]:
    criteria = {
        "live_transports_requested": bool(config.live_transports),
        "telemetry_messages_processed_gt_0": aggregate.telemetry_messages_processed > 0,
        "heartbeat_seen": bool(aggregate.heartbeat_seen),
        "state_usable": bool(aggregate.state_usable),
        "command_publication_allowed_false": True,
        "command_sent_count_zero": True,
        "phase4b_not_satisfied": True,
        "competition_readiness_not_claimed": True,
    }
    if mode in {
        CompetitionRunnerMode.VISION_DRY_RUN,
        CompetitionRunnerMode.COMMAND_DRY_RUN,
    }:
        criteria.update(
            {
                "vision_packets_processed_gt_0": aggregate.vision_packets_processed > 0,
                "vision_frames_completed_gt_0": aggregate.vision_frames_completed > 0,
                "perception_update_calls_gt_0": aggregate.perception_update_calls > 0,
            }
        )
    if mode == CompetitionRunnerMode.COMMAND_DRY_RUN:
        criteria["command_candidate_count_gt_0"] = aggregate.command_candidate_count > 0
    return criteria


def _phase6e_receive_satisfied(
    criteria: dict[str, bool],
    mode: CompetitionRunnerMode,
) -> bool:
    required = [
        "live_transports_requested",
        "telemetry_messages_processed_gt_0",
        "heartbeat_seen",
        "state_usable",
        "command_publication_allowed_false",
        "command_sent_count_zero",
        "phase4b_not_satisfied",
        "competition_readiness_not_claimed",
    ]
    if mode in {
        CompetitionRunnerMode.VISION_DRY_RUN,
        CompetitionRunnerMode.COMMAND_DRY_RUN,
    }:
        required.extend(
            [
                "vision_packets_processed_gt_0",
                "vision_frames_completed_gt_0",
            ]
        )
    if mode == CompetitionRunnerMode.COMMAND_DRY_RUN:
        required.append("command_candidate_count_gt_0")
    return all(bool(criteria.get(name, False)) for name in required)


def _normalize_mode(mode: CompetitionRunnerMode | str) -> CompetitionRunnerMode:
    if isinstance(mode, CompetitionRunnerMode):
        return mode
    try:
        return CompetitionRunnerMode(str(mode))
    except ValueError as exc:
        raise CompetitionMainSafetyError(f"unknown competition mode: {mode}") from exc


def _mode_value(mode: CompetitionRunnerMode | str) -> str:
    return mode.value if isinstance(mode, CompetitionRunnerMode) else str(mode)


def _transport_summary(transport: Any) -> Optional[dict[str, Any]]:
    summary = getattr(transport, "summary", None)
    if callable(summary):
        return summary()
    return None


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPETITION_READINESS_NOT_CLAIMED",
    "PHASE4B_NOT_SATISFIED",
    "PHASE_6D",
    "PHASE_6E",
    "PHASE6E_SURROGATE_LIMITATION",
    "CompetitionMainConfig",
    "CompetitionMainSafetyError",
    "CompetitionMainSummary",
    "build_arg_parser",
    "fail_closed_summary",
    "main",
    "run_competition_main",
]
