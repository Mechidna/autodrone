"""Import-safe setup wiring for the competition runtime.

Phase 6C constructs competition runtime components but does not provide a CLI,
start live sockets, run loops, or send commands. Real AutonomyAPI construction
is lazy and explicit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from autonomy_core.command.competition_command_adapter import CompetitionDryRunCommandAdapter
from autonomy_core.core.competition_config import RuntimeCompetitionConfig, VADR_TS_002
from autonomy_core.core.competition_state_adapter import CompetitionStateAdapter
from autonomy_core.perception.competition_image_adapter import CompetitionImageAdapter
from autonomy_core.runtime.competition_autonomy_factory import (
    CompetitionAutonomyProfile,
    create_competition_autonomy_api,
)
from autonomy_core.runtime.competition_guard import CompetitionGuard
from autonomy_core.runtime.competition_mavlink_transport import (
    CompetitionMavlinkTransport,
    CompetitionMavlinkTransportConfig,
)
from autonomy_core.runtime.competition_runner import (
    CompetitionRunner,
    CompetitionRunnerConfig,
    CompetitionRunnerMode,
    CompetitionRunnerSafetyConfig,
)
from autonomy_core.runtime.competition_vision_transport import (
    CompetitionVisionTransport,
    CompetitionVisionTransportConfig,
)
from autonomy_core.tools.competition_mavlink_observe import MavlinkTelemetryInventory


class CompetitionSetupError(RuntimeError):
    """Raised when competition runtime setup cannot be built safely."""


@dataclass(frozen=True)
class CompetitionSetupConfig:
    """Passive setup configuration for Phase 6C wiring."""

    mode: CompetitionRunnerMode | str = CompetitionRunnerMode.OBSERVE
    target_system: int = 1
    target_component: int = 1
    perception_world_pose_source: Optional[str] = None
    startup_metadata: Any = None
    competition_mode: bool = True
    use_real_autonomy: bool = False
    autonomy_profile: CompetitionAutonomyProfile = field(
        default_factory=CompetitionAutonomyProfile
    )
    autonomy_kwargs: Mapping[str, Any] = field(default_factory=dict)
    competition_config: RuntimeCompetitionConfig = VADR_TS_002
    runner_safety: CompetitionRunnerSafetyConfig = field(
        default_factory=CompetitionRunnerSafetyConfig
    )
    mavlink_transport_config: CompetitionMavlinkTransportConfig = field(
        default_factory=CompetitionMavlinkTransportConfig
    )
    vision_transport_config: CompetitionVisionTransportConfig = field(
        default_factory=CompetitionVisionTransportConfig
    )


@dataclass
class CompetitionRuntimeComponents:
    """Constructed competition runtime components for a future executable."""

    config: CompetitionSetupConfig
    runner: CompetitionRunner
    guard: CompetitionGuard
    state_adapter: CompetitionStateAdapter
    image_adapter: CompetitionImageAdapter
    command_adapter: CompetitionDryRunCommandAdapter
    mavlink_transport: CompetitionMavlinkTransport
    vision_transport: CompetitionVisionTransport
    telemetry_inventory: MavlinkTelemetryInventory
    autonomy: Any = None

    def close(self) -> None:
        """Close owned transport resources if a caller started them later."""

        close_mavlink = getattr(self.mavlink_transport, "close", None)
        if callable(close_mavlink):
            close_mavlink()
        close_vision = getattr(self.vision_transport, "close", None)
        if callable(close_vision):
            close_vision()


def build_competition_runtime(
    config: CompetitionSetupConfig = CompetitionSetupConfig(),
    *,
    guard: Optional[CompetitionGuard] = None,
    state_adapter: Optional[CompetitionStateAdapter] = None,
    image_adapter: Optional[CompetitionImageAdapter] = None,
    command_adapter: Optional[CompetitionDryRunCommandAdapter] = None,
    mavlink_transport: Optional[CompetitionMavlinkTransport] = None,
    vision_transport: Optional[CompetitionVisionTransport] = None,
    telemetry_inventory: Optional[MavlinkTelemetryInventory] = None,
    autonomy: Any = None,
    autonomy_factory: Optional[Callable[..., Any]] = None,
    clock: Callable[[], float] = time.time,
) -> CompetitionRuntimeComponents:
    """Construct competition runtime components without starting live IO."""

    active_guard = guard or CompetitionGuard(competition_mode=config.competition_mode)
    active_guard.assert_competition_safe(
        perception_world_pose_source=config.perception_world_pose_source,
        runner_inputs=config.startup_metadata,
        mode=str(_mode_value(config.mode)),
        command_enabled=config.runner_safety.command_publication_enabled,
    )

    active_state_adapter = state_adapter or CompetitionStateAdapter(clock=clock)
    active_image_adapter = image_adapter or CompetitionImageAdapter(clock=clock)
    active_command_adapter = command_adapter or CompetitionDryRunCommandAdapter(
        config=config.competition_config
    )
    active_mavlink_transport = mavlink_transport or CompetitionMavlinkTransport(
        config=config.mavlink_transport_config,
        clock=clock,
    )
    active_vision_transport = vision_transport or CompetitionVisionTransport(
        config=config.vision_transport_config,
        clock=clock,
    )
    active_telemetry_inventory = telemetry_inventory or MavlinkTelemetryInventory()
    active_autonomy = autonomy
    if active_autonomy is None and config.use_real_autonomy:
        active_autonomy = create_autonomy_api(
            profile=config.autonomy_profile,
            autonomy_factory=autonomy_factory,
            autonomy_kwargs=config.autonomy_kwargs,
            guard=active_guard,
        )

    runner_config = CompetitionRunnerConfig(
        mode=config.mode,
        target_system=config.target_system,
        target_component=config.target_component,
        perception_world_pose_source=config.perception_world_pose_source,
        startup_metadata=config.startup_metadata,
        safety=config.runner_safety,
    )
    runner = CompetitionRunner(
        config=runner_config,
        competition_config=config.competition_config,
        guard=active_guard,
        state_adapter=active_state_adapter,
        image_adapter=active_image_adapter,
        command_adapter=active_command_adapter,
        telemetry_inventory=active_telemetry_inventory,
        autonomy=active_autonomy,
        mavlink_transport=active_mavlink_transport,
        vision_transport=active_vision_transport,
        clock=clock,
    )

    return CompetitionRuntimeComponents(
        config=config,
        runner=runner,
        guard=active_guard,
        state_adapter=active_state_adapter,
        image_adapter=active_image_adapter,
        command_adapter=active_command_adapter,
        mavlink_transport=active_mavlink_transport,
        vision_transport=active_vision_transport,
        telemetry_inventory=active_telemetry_inventory,
        autonomy=active_autonomy,
    )


def create_autonomy_api(
    *,
    profile: CompetitionAutonomyProfile = CompetitionAutonomyProfile(),
    autonomy_factory: Optional[Callable[..., Any]] = None,
    autonomy_kwargs: Optional[Mapping[str, Any]] = None,
    guard: Optional[CompetitionGuard] = None,
) -> Any:
    """Create a competition-safe AutonomyAPI only when explicitly requested."""

    return create_competition_autonomy_api(
        profile=profile,
        autonomy_factory=autonomy_factory,
        autonomy_kwargs=autonomy_kwargs,
        guard=guard,
    )


def _mode_value(mode: CompetitionRunnerMode | str) -> str:
    return mode.value if isinstance(mode, CompetitionRunnerMode) else str(mode)


__all__ = [
    "CompetitionRuntimeComponents",
    "CompetitionSetupConfig",
    "CompetitionSetupError",
    "CompetitionAutonomyProfile",
    "build_competition_runtime",
    "create_autonomy_api",
]
