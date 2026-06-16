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
from autonomy_core.runtime.competition_autonomy_factory import (
    COMPETITION_OFFICIAL_TRANSFORM_MODE,
    CompetitionAutonomyProfile,
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
PHASE_9B = "9B"
PHASE_9C = "9C"
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
    real_perception: bool = False
    allow_legacy_yolo_default: bool = False
    perception_transform_mode: str = COMPETITION_OFFICIAL_TRANSFORM_MODE
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
    real_perception_requested: bool = False
    allow_legacy_yolo_default: bool = False
    perception_transform_mode: str = COMPETITION_OFFICIAL_TRANSFORM_MODE
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
    autonomy_telemetry_sync_count: int = 0
    planning_attempt_count: int = 0
    planning_success_count: int = 0
    planning_failure_count: int = 0
    command_candidate_count: int = 0
    command_candidate_accepted_count: int = 0
    command_candidate_rejection_count: int = 0
    last_command_result: Optional[dict[str, Any]] = None
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
    phase9b_perception_dry_run_satisfied: bool = False
    phase9b_success_criteria: dict[str, bool] = field(default_factory=dict)
    phase9c_command_dry_run_satisfied: bool = False
    phase9c_success_criteria: dict[str, bool] = field(default_factory=dict)
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
            "real_perception_requested": self.real_perception_requested,
            "allow_legacy_yolo_default": self.allow_legacy_yolo_default,
            "perception_transform_mode": self.perception_transform_mode,
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
            "autonomy_telemetry_sync_count": self.autonomy_telemetry_sync_count,
            "planning_attempt_count": self.planning_attempt_count,
            "planning_success_count": self.planning_success_count,
            "planning_failure_count": self.planning_failure_count,
            "command_candidate_count": self.command_candidate_count,
            "command_candidate_accepted_count": self.command_candidate_accepted_count,
            "command_candidate_rejection_count": self.command_candidate_rejection_count,
            "last_command_result": self.last_command_result,
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
            "phase9b_perception_dry_run_satisfied": (
                self.phase9b_perception_dry_run_satisfied
            ),
            "phase9b_success_criteria": dict(sorted(self.phase9b_success_criteria.items())),
            "phase9c_command_dry_run_satisfied": (
                self.phase9c_command_dry_run_satisfied
            ),
            "phase9c_success_criteria": dict(sorted(self.phase9c_success_criteria.items())),
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
    phase9b_criteria = _phase9b_success_criteria(
        config=config,
        mode=mode,
        aggregate=aggregate,
        phase6e_receive_satisfied=phase6e_receive_satisfied,
        phase6e_perception_boundary_satisfied=phase6e_perception_boundary_satisfied,
    )
    phase9b_satisfied = _phase9b_satisfied(phase9b_criteria)
    phase9c_criteria = _phase9c_success_criteria(
        config=config,
        mode=mode,
        aggregate=aggregate,
        phase6e_receive_satisfied=phase6e_receive_satisfied,
        phase6e_perception_boundary_satisfied=phase6e_perception_boundary_satisfied,
    )
    phase9c_satisfied = _phase9c_satisfied(phase9c_criteria)

    return CompetitionMainSummary(
        phase=_phase_for_config(config),
        mode=mode.value,
        status="dry_run_complete",
        evidence_label=str(config.evidence_label),
        live_transports_requested=bool(config.live_transports),
        use_real_autonomy=bool(config.use_real_autonomy),
        real_perception_requested=bool(config.real_perception),
        allow_legacy_yolo_default=bool(config.allow_legacy_yolo_default),
        perception_transform_mode=str(config.perception_transform_mode),
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
        autonomy_telemetry_sync_count=aggregate.autonomy_telemetry_sync_count,
        planning_attempt_count=aggregate.planning_attempt_count,
        planning_success_count=aggregate.planning_success_count,
        planning_failure_count=aggregate.planning_failure_count,
        command_candidate_count=aggregate.command_candidate_count,
        command_candidate_accepted_count=aggregate.command_candidate_accepted_count,
        command_candidate_rejection_count=aggregate.command_candidate_rejection_count,
        last_command_result=aggregate.last_command_result,
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
        phase9b_perception_dry_run_satisfied=phase9b_satisfied,
        phase9b_success_criteria=phase9b_criteria,
        phase9c_command_dry_run_satisfied=phase9c_satisfied,
        phase9c_success_criteria=phase9c_criteria,
        notes=_notes_for_config(config),
    )


def fail_closed_summary(
    *,
    mode: CompetitionRunnerMode | str,
    error: str,
    live_transports_requested: bool = False,
    use_real_autonomy: bool = False,
    real_perception_requested: bool = False,
    allow_legacy_yolo_default: bool = False,
    perception_transform_mode: str = COMPETITION_OFFICIAL_TRANSFORM_MODE,
) -> CompetitionMainSummary:
    normalized = _mode_value(mode)
    return CompetitionMainSummary(
        phase=_phase_for_flags(
            mode=normalized,
            live_transports=live_transports_requested,
            real_perception=real_perception_requested,
        ),
        mode=normalized,
        status="fail_closed",
        fail_closed=True,
        safety_error=str(error),
        evidence_label="fail_closed",
        live_transports_requested=bool(live_transports_requested),
        use_real_autonomy=bool(use_real_autonomy),
        real_perception_requested=bool(real_perception_requested),
        allow_legacy_yolo_default=bool(allow_legacy_yolo_default),
        perception_transform_mode=str(perception_transform_mode),
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
    parser.add_argument(
        "--real-perception",
        action="store_true",
        help=(
            "Phase 9B: enable the competition-safe AutonomyAPI profile with "
            "use_perception=True. In command_dry_run this becomes Phase 9C; "
            "commands are still never sent"
        ),
    )
    parser.add_argument(
        "--allow-legacy-yolo-default",
        action="store_true",
        help=(
            "temporary Phase 9B acknowledgment that AutonomyAPI still uses its "
            "legacy hardcoded YOLO weights path"
        ),
    )
    parser.add_argument(
        "--perception-transform-mode",
        default=COMPETITION_OFFICIAL_TRANSFORM_MODE,
        help=(
            "Phase 9B.2 perception transform mode. Real-perception competition "
            f"dry-runs require {COMPETITION_OFFICIAL_TRANSFORM_MODE!r}."
        ),
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
        real_perception=bool(args.real_perception),
        allow_legacy_yolo_default=bool(args.allow_legacy_yolo_default),
        perception_transform_mode=str(args.perception_transform_mode),
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
            real_perception_requested=bool(args.real_perception),
            allow_legacy_yolo_default=bool(args.allow_legacy_yolo_default),
            perception_transform_mode=str(args.perception_transform_mode),
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
    autonomy_telemetry_sync_count: int = 0
    planning_attempt_count: int = 0
    planning_success_count: int = 0
    planning_failure_count: int = 0
    command_candidate_count: int = 0
    command_candidate_accepted_count: int = 0
    command_candidate_rejection_count: int = 0
    last_command_result: Optional[dict[str, Any]] = None
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
        if getattr(result, "autonomy_telemetry_synced", False):
            self.autonomy_telemetry_sync_count += 1
        if getattr(result, "planning_attempted", False):
            self.planning_attempt_count += 1
            if result.planning_succeeded:
                self.planning_success_count += 1
            else:
                self.planning_failure_count += 1
        if result.command_candidate_attempted:
            self.command_candidate_count += 1
            if result.command_result is not None and result.command_result.accepted:
                self.command_candidate_accepted_count += 1
            else:
                self.command_candidate_rejection_count += 1
        if result.command_result is not None:
            self.last_command_result = _command_result_to_dict(result.command_result)
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
        autonomy_profile=CompetitionAutonomyProfile(
            use_perception=bool(config.real_perception),
            allow_legacy_yolo_default=bool(config.allow_legacy_yolo_default),
            perception_transform_mode=str(config.perception_transform_mode),
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
    if config.real_perception and mode not in {
        CompetitionRunnerMode.VISION_DRY_RUN,
        CompetitionRunnerMode.COMMAND_DRY_RUN,
    }:
        raise CompetitionMainSafetyError(
            "real perception is only enabled for Phase 9B vision_dry_run or "
            "Phase 9C command_dry_run"
        )
    if config.real_perception and not config.use_real_autonomy:
        raise CompetitionMainSafetyError(
            "Phase 9B/9C requires --use-real-autonomy with --real-perception"
        )
    if config.real_perception and not config.live_transports and components is None:
        raise CompetitionMainSafetyError(
            "Phase 9B/9C requires --live-transports or injected test components"
        )
    if config.real_perception and not config.allow_legacy_yolo_default and components is None:
        raise CompetitionMainSafetyError(
            "Phase 9B/9C currently requires --allow-legacy-yolo-default because "
            "AutonomyAPI still has a hardcoded YOLO weights path"
        )
    if (
        config.real_perception
        and str(config.perception_transform_mode) != COMPETITION_OFFICIAL_TRANSFORM_MODE
    ):
        raise CompetitionMainSafetyError(
            "Phase 9B.2/9C real perception requires "
            f"--perception-transform-mode {COMPETITION_OFFICIAL_TRANSFORM_MODE}"
        )
    if components is None and not config.live_transports:
        raise CompetitionMainSafetyError(
            "Phase 6D requires injected components or explicit --live-transports"
        )


def _phase_for_config(config: CompetitionMainConfig) -> str:
    if config.real_perception:
        if _mode_value(config.mode) == CompetitionRunnerMode.COMMAND_DRY_RUN.value:
            return PHASE_9C
        return PHASE_9B
    return PHASE_6E if config.live_transports else PHASE_6D


def _phase_for_flags(
    *,
    mode: str,
    live_transports: bool,
    real_perception: bool,
) -> str:
    if real_perception:
        if str(mode) == CompetitionRunnerMode.COMMAND_DRY_RUN.value:
            return PHASE_9C
        return PHASE_9B
    return PHASE_6E if live_transports else PHASE_6D


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
    if config.real_perception:
        notes.append(
            "Phase 9B/9C runs real perception through the competition-safe "
            "AutonomyAPI profile; command publication remains disabled."
        )
        notes.append(
            "Phase 9B.2 selects the official competition camera/NED transform "
            "for real-perception dry-runs."
        )
        if config.allow_legacy_yolo_default:
            notes.append(
                "Legacy YOLO default was explicitly acknowledged for this "
                "temporary dry-run; replace with explicit YOLO config later."
            )
        if _mode_value(config.mode) == CompetitionRunnerMode.COMMAND_DRY_RUN.value:
            notes.append(
                "Phase 9C builds no-send command candidates only; PX4/Gazebo "
                "surrogate command candidates do not prove competition command "
                "acceptance."
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


def _phase9b_success_criteria(
    *,
    config: CompetitionMainConfig,
    mode: CompetitionRunnerMode,
    aggregate: _Aggregate,
    phase6e_receive_satisfied: bool,
    phase6e_perception_boundary_satisfied: bool,
) -> dict[str, bool]:
    return {
        "mode_is_vision_dry_run": mode == CompetitionRunnerMode.VISION_DRY_RUN,
        "live_transports_requested": bool(config.live_transports),
        "use_real_autonomy": bool(config.use_real_autonomy),
        "real_perception_requested": bool(config.real_perception),
        "legacy_yolo_default_acknowledged": bool(config.allow_legacy_yolo_default),
        "official_competition_transform_selected": (
            str(config.perception_transform_mode) == COMPETITION_OFFICIAL_TRANSFORM_MODE
        ),
        "phase6e_receive_satisfied": bool(phase6e_receive_satisfied),
        "phase6e_perception_boundary_satisfied": bool(
            phase6e_perception_boundary_satisfied
        ),
        "perception_update_calls_gt_0": aggregate.perception_update_calls > 0,
        "command_publication_allowed_false": True,
        "command_sent_count_zero": True,
        "phase4b_not_satisfied": True,
        "competition_readiness_not_claimed": True,
    }


def _phase9b_satisfied(criteria: dict[str, bool]) -> bool:
    required = [
        "mode_is_vision_dry_run",
        "live_transports_requested",
        "use_real_autonomy",
        "real_perception_requested",
        "official_competition_transform_selected",
        "phase6e_receive_satisfied",
        "phase6e_perception_boundary_satisfied",
        "perception_update_calls_gt_0",
        "command_publication_allowed_false",
        "command_sent_count_zero",
        "phase4b_not_satisfied",
        "competition_readiness_not_claimed",
    ]
    return all(bool(criteria.get(name, False)) for name in required)


def _phase9c_success_criteria(
    *,
    config: CompetitionMainConfig,
    mode: CompetitionRunnerMode,
    aggregate: _Aggregate,
    phase6e_receive_satisfied: bool,
    phase6e_perception_boundary_satisfied: bool,
) -> dict[str, bool]:
    command_result = aggregate.last_command_result or {}
    fields = command_result.get("fields") or {}
    return {
        "mode_is_command_dry_run": mode == CompetitionRunnerMode.COMMAND_DRY_RUN,
        "live_transports_requested": bool(config.live_transports),
        "use_real_autonomy": bool(config.use_real_autonomy),
        "real_perception_requested": bool(config.real_perception),
        "legacy_yolo_default_acknowledged": bool(config.allow_legacy_yolo_default),
        "official_competition_transform_selected": (
            str(config.perception_transform_mode) == COMPETITION_OFFICIAL_TRANSFORM_MODE
        ),
        "phase6e_receive_satisfied": bool(phase6e_receive_satisfied),
        "phase6e_perception_boundary_satisfied": bool(
            phase6e_perception_boundary_satisfied
        ),
        "perception_update_calls_gt_0": aggregate.perception_update_calls > 0,
        "autonomy_telemetry_sync_count_gt_0": (
            aggregate.autonomy_telemetry_sync_count > 0
        ),
        "planning_attempt_count_gt_0": aggregate.planning_attempt_count > 0,
        "planning_success_count_gt_0": aggregate.planning_success_count > 0,
        "command_candidate_count_gt_0": aggregate.command_candidate_count > 0,
        "command_candidate_accepted_count_gt_0": (
            aggregate.command_candidate_accepted_count > 0
        ),
        "last_command_result_accepted": bool(command_result.get("accepted", False)),
        "last_command_result_no_send": fields.get("send_ready") is False,
        "last_command_message_is_set_attitude_target": (
            fields.get("message_name") == "SET_ATTITUDE_TARGET"
        ),
        "command_publication_allowed_false": True,
        "command_sent_count_zero": True,
        "phase4b_not_satisfied": True,
        "competition_readiness_not_claimed": True,
    }


def _phase9c_satisfied(criteria: dict[str, bool]) -> bool:
    required = [
        "mode_is_command_dry_run",
        "live_transports_requested",
        "use_real_autonomy",
        "real_perception_requested",
        "official_competition_transform_selected",
        "phase6e_receive_satisfied",
        "phase6e_perception_boundary_satisfied",
        "perception_update_calls_gt_0",
        "autonomy_telemetry_sync_count_gt_0",
        "planning_attempt_count_gt_0",
        "planning_success_count_gt_0",
        "command_candidate_count_gt_0",
        "command_candidate_accepted_count_gt_0",
        "last_command_result_accepted",
        "last_command_result_no_send",
        "last_command_message_is_set_attitude_target",
        "command_publication_allowed_false",
        "command_sent_count_zero",
        "phase4b_not_satisfied",
        "competition_readiness_not_claimed",
    ]
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


def _command_result_to_dict(result: Any) -> dict[str, Any]:
    fields = getattr(result, "fields", None)
    fields_dict = None
    if fields is not None:
        fields_dict = {
            "message_name": fields.message_name,
            "time_boot_ms": fields.time_boot_ms,
            "target_system": fields.target_system,
            "target_component": fields.target_component,
            "type_mask": fields.type_mask,
            "q": list(fields.q),
            "body_roll_rate": fields.body_roll_rate,
            "body_pitch_rate": fields.body_pitch_rate,
            "body_yaw_rate": fields.body_yaw_rate,
            "thrust": fields.thrust,
            "source_tuple_semantics": fields.source_tuple_semantics,
            "send_ready": fields.send_ready,
            "send_blocked_reason": fields.send_blocked_reason,
            "sequence": fields.sequence,
            "pymavlink_args": list(fields.as_pymavlink_args()),
        }

    return {
        "accepted": bool(getattr(result, "accepted", False)),
        "rejection_reason": str(getattr(result, "rejection_reason", "")),
        "fields": fields_dict,
    }


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPETITION_READINESS_NOT_CLAIMED",
    "PHASE4B_NOT_SATISFIED",
    "PHASE_6D",
    "PHASE_6E",
    "PHASE_9B",
    "PHASE_9C",
    "PHASE6E_SURROGATE_LIMITATION",
    "CompetitionMainConfig",
    "CompetitionMainSafetyError",
    "CompetitionMainSummary",
    "build_arg_parser",
    "fail_closed_summary",
    "main",
    "run_competition_main",
]
