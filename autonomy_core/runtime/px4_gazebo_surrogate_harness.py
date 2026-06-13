"""PX4/Gazebo surrogate harness scaffold for CompetitionRunner.

Phase 8.5A is surrogate confidence only. This module never opens sockets,
starts PX4/Gazebo/MAVSDK/ROS, sends commands, or provides competition
readiness evidence. It feeds the competition runner through injected fake
transports only.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from autonomy_core.command.competition_command_adapter import DryRunCommandResult
from autonomy_core.core.competition_config import RuntimeCompetitionConfig, VADR_TS_002
from autonomy_core.perception.competition_image_adapter import (
    CompetitionImageAdapter,
    pack_vision_packet,
)
from autonomy_core.runtime.competition_guard import CompetitionGuard, CompetitionGuardError
from autonomy_core.runtime.competition_runner import (
    CompetitionRunner,
    CompetitionRunnerConfig,
    CompetitionRunnerMode,
    CompetitionRunnerSafetyConfig,
    CompetitionRunnerSafetyError,
    CompetitionRunnerStepResult,
)


SURROGATE_LABEL = "PX4/Gazebo surrogate only"
PHASE4B_NOT_SATISFIED_REASON = (
    "PX4/Gazebo surrogate evidence does not satisfy Phase 4B real competition "
    "simulator receive-only telemetry evidence."
)


class Px4GazeboSurrogateHarnessError(ValueError):
    """Raised when surrogate-only fixture input is malformed or unsafe."""


@dataclass(frozen=True)
class SurrogateMavlinkMessage:
    """Small MAVLink-like message object for injected runner tests."""

    message_type: str
    message_id: int
    fields: Mapping[str, Any] = field(default_factory=dict)
    source_system: int = 1
    source_component: int = 1

    def __getattr__(self, name: str) -> Any:
        try:
            return self.fields[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def get_type(self) -> str:
        return self.message_type

    def get_msgId(self) -> int:
        return int(self.message_id)

    def get_srcSystem(self) -> int:
        return int(self.source_system)

    def get_srcComponent(self) -> int:
        return int(self.source_component)

    def to_dict(self) -> dict[str, Any]:
        return {"mavpackettype": self.message_type, **dict(self.fields)}


@dataclass(frozen=True)
class Px4EstimatedTelemetrySample:
    """Estimated PX4/MAVSDK local telemetry, never Gazebo truth."""

    position_ned_m: tuple[float, float, float]
    velocity_ned_m_s: tuple[float, float, float]
    yaw_rad: float
    time_boot_ms: int = 0
    roll_rad: float = 0.0
    pitch_rad: float = 0.0
    metadata: Optional[Mapping[str, Any]] = None
    source_system: int = 1
    source_component: int = 1


@dataclass
class Px4GazeboSurrogateSummary:
    """Compact result summary; explicitly not competition readiness evidence."""

    surrogate_label: str = SURROGATE_LABEL
    frame_count: int = 0
    completed_packetized_frames: int = 0
    packetization_errors: int = 0
    telemetry_sample_count: int = 0
    stale_telemetry_count: int = 0
    command_candidate_count: int = 0
    command_blocked_reasons: list[str] = field(default_factory=list)
    guard_rejection_count: int = 0
    phase4b_satisfied: bool = False
    competition_readiness_claimed: bool = False

    def record_step(self, result: CompetitionRunnerStepResult) -> None:
        if result.command_candidate_attempted:
            self.command_candidate_count += 1
        for reason in result.command_blocked_reasons:
            if reason not in self.command_blocked_reasons:
                self.command_blocked_reasons.append(reason)
        if any(reason.startswith("stale_") for reason in result.state_result.missing_reasons):
            self.stale_telemetry_count += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "surrogate_label": self.surrogate_label,
            "frame_count": self.frame_count,
            "completed_packetized_frames": self.completed_packetized_frames,
            "packetization_errors": self.packetization_errors,
            "telemetry_sample_count": self.telemetry_sample_count,
            "stale_telemetry_count": self.stale_telemetry_count,
            "command_candidate_count": self.command_candidate_count,
            "command_blocked_reasons": list(self.command_blocked_reasons),
            "guard_rejection_count": self.guard_rejection_count,
            "phase4b_satisfied": self.phase4b_satisfied,
            "competition_readiness_claimed": self.competition_readiness_claimed,
        }


@dataclass(frozen=True)
class SurrogateDecodedFrame:
    """Minimal decoded frame stand-in for deterministic tests without cv2."""

    height: int
    width: int
    channels: int = 3
    dtype: str = "uint8"

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, self.channels)


@dataclass(frozen=True)
class Px4GazeboSurrogateScenarioStep:
    """One deterministic scenario step for injected runner execution."""

    wall_time_s: float
    telemetry_samples: tuple[Px4EstimatedTelemetrySample, ...] = ()
    jpeg_frames: tuple[bytes, ...] = ()
    sim_time_ns: Optional[int] = None
    frame_period_ns: int = 33_333_333
    telemetry_include_heartbeat: bool = True
    vision_metadata: Any = None


@dataclass(frozen=True)
class Px4GazeboSurrogateScenario:
    """Surrogate-only scenario definition; not competition evidence."""

    name: str
    mode: CompetitionRunnerMode | str
    steps: tuple[Px4GazeboSurrogateScenarioStep, ...]
    command_tuple: tuple[float, ...] = (0.0, 0.0, 0.0, 0.5)


@dataclass(frozen=True)
class Px4GazeboSurrogateScenarioResult:
    """Deterministic scenario result summary."""

    scenario_name: str
    mode: CompetitionRunnerMode
    step_results: tuple[CompetitionRunnerStepResult, ...]
    telemetry_message_count: int
    vision_packet_count: int
    frame_count: int
    command_candidate_count: int
    summary: Mapping[str, Any]

    @property
    def phase4b_satisfied(self) -> bool:
        return bool(self.summary.get("phase4b_satisfied", False))

    @property
    def competition_readiness_claimed(self) -> bool:
        return bool(self.summary.get("competition_readiness_claimed", False))


class InjectedTelemetryTransport:
    """One-shot fake telemetry transport for CompetitionRunner injection."""

    def __init__(self, messages: Iterable[Any]):
        self._messages = list(messages)
        self.calls = 0

    def receive_messages(self) -> tuple[Any, ...]:
        self.calls += 1
        messages = tuple(self._messages)
        self._messages = []
        return messages


class InjectedVisionPacketTransport:
    """One-shot fake vision transport for CompetitionRunner injection."""

    def __init__(self, packets: Iterable[bytes]):
        self._packets = [bytes(packet) for packet in packets]
        self.calls = 0

    def receive_packets(self) -> tuple[bytes, ...]:
        self.calls += 1
        packets = tuple(self._packets)
        self._packets = []
        return packets


class _SurrogateCommandAutonomy:
    """Tiny injected autonomy stand-in used only for dry-run command candidates."""

    def __init__(self, command_tuple: Sequence[float]):
        self.command_tuple = tuple(command_tuple)
        self.attitude_control_calls = 0
        self.perception_updates: list[dict[str, Any]] = []

    def attitude_control(self) -> tuple[float, ...]:
        self.attitude_control_calls += 1
        return self.command_tuple

    def update_gate_memory_from_frame(self, **kwargs: Any) -> None:
        self.perception_updates.append(dict(kwargs))


class _ScenarioClock:
    def __init__(self, start: float = 0.0):
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def set(self, value: float) -> None:
        self.now = _finite_float(value, name="wall_time_s")


def make_surrogate_heartbeat(
    *,
    source_system: int = 1,
    source_component: int = 1,
) -> SurrogateMavlinkMessage:
    return SurrogateMavlinkMessage(
        message_type="HEARTBEAT",
        message_id=0,
        fields={"base_mode": 0, "system_status": 0},
        source_system=source_system,
        source_component=source_component,
    )


def telemetry_sample_to_fake_mavlink_messages(
    sample: Px4EstimatedTelemetrySample,
    *,
    guard: Optional[CompetitionGuard] = None,
) -> tuple[SurrogateMavlinkMessage, SurrogateMavlinkMessage]:
    """Convert estimated PX4/MAVSDK telemetry to runner-accepted fake messages."""

    active_guard = CompetitionGuard() if guard is None else guard
    active_guard.assert_no_gazebo_truth_fields(
        sample.metadata,
        context="PX4/Gazebo surrogate telemetry metadata",
    )
    position = _finite_vector3(sample.position_ned_m, name="position_ned_m")
    velocity = _finite_vector3(sample.velocity_ned_m_s, name="velocity_ned_m_s")
    yaw = _finite_float(sample.yaw_rad, name="yaw_rad")
    roll = _finite_float(sample.roll_rad, name="roll_rad")
    pitch = _finite_float(sample.pitch_rad, name="pitch_rad")
    time_boot_ms = _uint32(sample.time_boot_ms, name="time_boot_ms")

    common = {
        "source_system": sample.source_system,
        "source_component": sample.source_component,
    }
    local_position = SurrogateMavlinkMessage(
        message_type="LOCAL_POSITION_NED",
        message_id=32,
        fields={
            "x": position[0],
            "y": position[1],
            "z": position[2],
            "vx": velocity[0],
            "vy": velocity[1],
            "vz": velocity[2],
            "time_boot_ms": time_boot_ms,
        },
        **common,
    )
    attitude = SurrogateMavlinkMessage(
        message_type="ATTITUDE",
        message_id=30,
        fields={
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "rollspeed": 0.0,
            "pitchspeed": 0.0,
            "yawspeed": 0.0,
            "time_boot_ms": time_boot_ms,
        },
        **common,
    )
    return local_position, attitude


def telemetry_samples_to_fake_mavlink_messages(
    samples: Iterable[Px4EstimatedTelemetrySample],
    *,
    include_heartbeat: bool = True,
    guard: Optional[CompetitionGuard] = None,
) -> tuple[SurrogateMavlinkMessage, ...]:
    active_guard = CompetitionGuard() if guard is None else guard
    messages: list[SurrogateMavlinkMessage] = []
    if include_heartbeat:
        messages.append(make_surrogate_heartbeat())
    for sample in samples:
        messages.extend(telemetry_sample_to_fake_mavlink_messages(sample, guard=active_guard))
    return tuple(messages)


def packetize_vadr_jpeg_bytes(
    jpeg_bytes: bytes,
    *,
    frame_id: int,
    sim_time_ns: int,
    max_payload_size: int = 1200,
    config: RuntimeCompetitionConfig = VADR_TS_002,
    guard: Optional[CompetitionGuard] = None,
    metadata: Any = None,
) -> tuple[bytes, ...]:
    """Packetize encoded JPEG bytes into fake VADR `<IHHIIQ` packets."""

    active_guard = CompetitionGuard() if guard is None else guard
    active_guard.assert_no_gazebo_truth_fields(
        metadata,
        context="PX4/Gazebo surrogate vision metadata",
    )
    jpeg = bytes(jpeg_bytes)
    if len(jpeg) == 0:
        raise Px4GazeboSurrogateHarnessError("jpeg_bytes must not be empty")
    if max_payload_size <= 0:
        raise Px4GazeboSurrogateHarnessError("max_payload_size must be positive")
    _uint32(frame_id, name="frame_id")
    _uint64(sim_time_ns, name="sim_time_ns")

    chunks = [
        jpeg[index : index + max_payload_size]
        for index in range(0, len(jpeg), max_payload_size)
    ]
    if len(chunks) > 0xFFFF:
        raise Px4GazeboSurrogateHarnessError("too many JPEG chunks for VADR header")

    return tuple(
        pack_vision_packet(
            frame_id=frame_id,
            chunk_id=chunk_id,
            total_chunks=len(chunks),
            jpeg_size=len(jpeg),
            payload=chunk,
            sim_time_ns=sim_time_ns,
            config=config,
        )
        for chunk_id, chunk in enumerate(chunks)
    )


def packetize_vadr_frame_array(
    frame: Any,
    *,
    frame_id: int,
    sim_time_ns: int,
    max_payload_size: int = 1200,
    config: RuntimeCompetitionConfig = VADR_TS_002,
    guard: Optional[CompetitionGuard] = None,
    metadata: Any = None,
) -> tuple[bytes, ...]:
    """Lazy optional helper for future live surrogate image-pixel runs."""

    shape = getattr(frame, "shape", None)
    if shape is None or len(shape) < 2:
        raise Px4GazeboSurrogateHarnessError("frame must expose image shape")
    if tuple(shape[:2]) != (config.camera_height_px, config.camera_width_px):
        raise Px4GazeboSurrogateHarnessError(
            "frame shape must match official competition camera resolution"
        )

    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise Px4GazeboSurrogateHarnessError(
            "cv2 is required only for explicit frame-array JPEG encoding"
        ) from exc

    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise Px4GazeboSurrogateHarnessError("JPEG encoding failed")
    return packetize_vadr_jpeg_bytes(
        bytes(encoded),
        frame_id=frame_id,
        sim_time_ns=sim_time_ns,
        max_payload_size=max_payload_size,
        config=config,
        guard=guard,
        metadata=metadata,
    )


def deterministic_surrogate_jpeg_decoder(
    _jpeg_bytes: bytes,
    *,
    config: RuntimeCompetitionConfig = VADR_TS_002,
) -> SurrogateDecodedFrame:
    """Decode stand-in for deterministic tests; does not inspect JPEG bytes."""

    return SurrogateDecodedFrame(
        height=config.camera_height_px,
        width=config.camera_width_px,
        channels=3,
    )


@dataclass
class Px4GazeboSurrogateHarness:
    """Surrogate-only runner scaffold using injected fake transports."""

    config: RuntimeCompetitionConfig = VADR_TS_002
    guard: CompetitionGuard = field(default_factory=CompetitionGuard)
    clock: Callable[[], float] = time.time
    summary: Px4GazeboSurrogateSummary = field(default_factory=Px4GazeboSurrogateSummary)

    def telemetry_messages(
        self,
        samples: Iterable[Px4EstimatedTelemetrySample],
        *,
        include_heartbeat: bool = True,
    ) -> tuple[SurrogateMavlinkMessage, ...]:
        sample_list = list(samples)
        self.summary.telemetry_sample_count += len(sample_list)
        try:
            return telemetry_samples_to_fake_mavlink_messages(
                sample_list,
                include_heartbeat=include_heartbeat,
                guard=self.guard,
            )
        except CompetitionGuardError:
            self.summary.guard_rejection_count += 1
            raise

    def packetize_jpeg_frames(
        self,
        jpeg_frames: Iterable[bytes],
        *,
        starting_frame_id: int = 1,
        starting_sim_time_ns: int = 0,
        frame_period_ns: int = 33_333_333,
        max_payload_size: int = 1200,
        metadata: Any = None,
    ) -> tuple[bytes, ...]:
        packets: list[bytes] = []
        for offset, jpeg in enumerate(jpeg_frames):
            self.summary.frame_count += 1
            try:
                packets.extend(
                    packetize_vadr_jpeg_bytes(
                        jpeg,
                        frame_id=starting_frame_id + offset,
                        sim_time_ns=starting_sim_time_ns + offset * frame_period_ns,
                        max_payload_size=max_payload_size,
                        config=self.config,
                        guard=self.guard,
                        metadata=metadata,
                    )
                )
            except CompetitionGuardError:
                self.summary.guard_rejection_count += 1
                self.summary.packetization_errors += 1
                raise
            except Px4GazeboSurrogateHarnessError:
                self.summary.packetization_errors += 1
                raise
            self.summary.completed_packetized_frames += 1
        return tuple(packets)

    def build_runner(
        self,
        *,
        mode: CompetitionRunnerMode | str,
        telemetry_messages: Iterable[Any] = (),
        vision_packets: Iterable[bytes] = (),
        autonomy: Any = None,
        image_adapter: Optional[CompetitionImageAdapter] = None,
    ) -> CompetitionRunner:
        runner_mode = mode if isinstance(mode, CompetitionRunnerMode) else CompetitionRunnerMode(str(mode))
        if runner_mode in (
            CompetitionRunnerMode.COMMAND_LIVE,
            CompetitionRunnerMode.RACE,
        ):
            # Preserve runner fail-closed behavior; do not pre-filter it here.
            return CompetitionRunner(
                config=CompetitionRunnerConfig(mode=runner_mode),
                competition_config=self.config,
                guard=self.guard,
                clock=self.clock,
            )

        return CompetitionRunner(
            config=CompetitionRunnerConfig(
                mode=runner_mode,
                safety=CompetitionRunnerSafetyConfig(
                    phase4b_telemetry_evidence_available=False,
                    command_publication_enabled=False,
                    allow_live_command_modes=False,
                ),
            ),
            competition_config=self.config,
            guard=self.guard,
            clock=self.clock,
            mavlink_transport=InjectedTelemetryTransport(telemetry_messages),
            vision_transport=InjectedVisionPacketTransport(vision_packets),
            image_adapter=image_adapter,
            autonomy=autonomy,
        )

    def run_observe_surrogate(
        self,
        samples: Iterable[Px4EstimatedTelemetrySample],
    ) -> CompetitionRunnerStepResult:
        messages = self.telemetry_messages(samples)
        runner = self.build_runner(
            mode=CompetitionRunnerMode.OBSERVE,
            telemetry_messages=messages,
        )
        return self._step_and_record(runner)

    def run_vision_dry_run_surrogate(
        self,
        *,
        jpeg_frames: Iterable[bytes],
        autonomy: Any = None,
        image_adapter: Optional[CompetitionImageAdapter] = None,
    ) -> CompetitionRunnerStepResult:
        packets = self.packetize_jpeg_frames(jpeg_frames)
        runner = self.build_runner(
            mode=CompetitionRunnerMode.VISION_DRY_RUN,
            vision_packets=packets,
            autonomy=autonomy,
            image_adapter=image_adapter,
        )
        return self._step_and_record(runner)

    def run_command_dry_run_surrogate(
        self,
        samples: Iterable[Px4EstimatedTelemetrySample],
        *,
        command_tuple: Sequence[float] = (0.0, 0.0, 0.0, 0.5),
        jpeg_frames: Iterable[bytes] = (),
        image_adapter: Optional[CompetitionImageAdapter] = None,
    ) -> CompetitionRunnerStepResult:
        messages = self.telemetry_messages(samples)
        packets = self.packetize_jpeg_frames(jpeg_frames) if jpeg_frames else ()
        autonomy = _SurrogateCommandAutonomy(command_tuple)
        runner = self.build_runner(
            mode=CompetitionRunnerMode.COMMAND_DRY_RUN,
            telemetry_messages=messages,
            vision_packets=packets,
            autonomy=autonomy,
            image_adapter=image_adapter,
        )
        return self._step_and_record(runner)

    def run_scenario(
        self,
        scenario: Px4GazeboSurrogateScenario,
        *,
        autonomy: Any = None,
        image_adapter: Optional[CompetitionImageAdapter] = None,
        jpeg_decoder: Optional[Callable[[bytes], Any]] = None,
    ) -> Px4GazeboSurrogateScenarioResult:
        """Run a deterministic surrogate scenario through injected batches only."""

        steps = tuple(scenario.steps)
        self._validate_scenario_steps(steps)
        runner_mode = (
            scenario.mode
            if isinstance(scenario.mode, CompetitionRunnerMode)
            else CompetitionRunnerMode(str(scenario.mode))
        )
        scenario_clock = _ScenarioClock(steps[0].wall_time_s if steps else 0.0)
        active_autonomy = autonomy
        if runner_mode == CompetitionRunnerMode.COMMAND_DRY_RUN and active_autonomy is None:
            active_autonomy = _SurrogateCommandAutonomy(scenario.command_tuple)
        active_image_adapter = image_adapter
        if active_image_adapter is None and runner_mode in (
            CompetitionRunnerMode.VISION_DRY_RUN,
            CompetitionRunnerMode.COMMAND_DRY_RUN,
        ):
            decoder = jpeg_decoder
            if decoder is None:
                decoder = lambda jpeg: deterministic_surrogate_jpeg_decoder(
                    jpeg,
                    config=self.config,
                )
            active_image_adapter = CompetitionImageAdapter(
                clock=scenario_clock,
                jpeg_decoder=decoder,
            )

        runner = CompetitionRunner(
            config=CompetitionRunnerConfig(
                mode=runner_mode,
                safety=CompetitionRunnerSafetyConfig(
                    phase4b_telemetry_evidence_available=False,
                    command_publication_enabled=False,
                    allow_live_command_modes=False,
                ),
            ),
            competition_config=self.config,
            guard=self.guard,
            clock=scenario_clock,
            image_adapter=active_image_adapter,
            autonomy=active_autonomy,
        )

        next_frame_id = 1
        telemetry_message_count = 0
        vision_packet_count = 0
        step_results: list[CompetitionRunnerStepResult] = []
        for step_index, step in enumerate(steps):
            scenario_clock.set(step.wall_time_s)
            messages = self.telemetry_messages(
                step.telemetry_samples,
                include_heartbeat=step.telemetry_include_heartbeat,
            )
            telemetry_message_count += len(messages)

            packets: tuple[bytes, ...] = ()
            if step.jpeg_frames:
                sim_time_ns = (
                    step.sim_time_ns
                    if step.sim_time_ns is not None
                    else int(step.wall_time_s * 1_000_000_000)
                )
                packets = self.packetize_jpeg_frames(
                    step.jpeg_frames,
                    starting_frame_id=next_frame_id,
                    starting_sim_time_ns=sim_time_ns,
                    frame_period_ns=step.frame_period_ns,
                    metadata=step.vision_metadata,
                )
                next_frame_id += len(step.jpeg_frames)
                vision_packet_count += len(packets)

            result = runner.step(
                telemetry_messages=messages,
                vision_packets=packets,
            )
            self.summary.record_step(result)
            step_results.append(result)

        summary = self.summary_dict()
        return Px4GazeboSurrogateScenarioResult(
            scenario_name=scenario.name,
            mode=runner_mode,
            step_results=tuple(step_results),
            telemetry_message_count=telemetry_message_count,
            vision_packet_count=vision_packet_count,
            frame_count=self.summary.frame_count,
            command_candidate_count=self.summary.command_candidate_count,
            summary=summary,
        )

    def _step_and_record(self, runner: CompetitionRunner) -> CompetitionRunnerStepResult:
        result = runner.step()
        self.summary.record_step(result)
        return result

    def summary_dict(self) -> dict[str, Any]:
        return self.summary.to_dict()

    def _validate_scenario_steps(
        self,
        steps: Sequence[Px4GazeboSurrogateScenarioStep],
    ) -> None:
        previous_wall_time: Optional[float] = None
        previous_time_boot_ms: Optional[int] = None
        previous_sim_time_ns: Optional[int] = None
        for step in steps:
            wall_time = _finite_float(step.wall_time_s, name="wall_time_s")
            if previous_wall_time is not None and wall_time <= previous_wall_time:
                raise Px4GazeboSurrogateHarnessError(
                    "scenario wall_time_s values must be strictly increasing"
                )
            previous_wall_time = wall_time

            if step.frame_period_ns <= 0:
                raise Px4GazeboSurrogateHarnessError("frame_period_ns must be positive")
            if step.sim_time_ns is not None:
                sim_time_ns = _uint64(step.sim_time_ns, name="sim_time_ns")
                if previous_sim_time_ns is not None and sim_time_ns < previous_sim_time_ns:
                    raise Px4GazeboSurrogateHarnessError(
                        "scenario sim_time_ns values must be monotonic"
                    )
                previous_sim_time_ns = sim_time_ns

            for sample in step.telemetry_samples:
                time_boot_ms = _uint32(sample.time_boot_ms, name="time_boot_ms")
                if (
                    previous_time_boot_ms is not None
                    and time_boot_ms < previous_time_boot_ms
                ):
                    raise Px4GazeboSurrogateHarnessError(
                        "scenario telemetry time_boot_ms values must be monotonic"
                    )
                previous_time_boot_ms = time_boot_ms


def command_result_was_no_send(result: Optional[DryRunCommandResult]) -> bool:
    return result is not None and result.fields is not None and not result.fields.send_ready


def _finite_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise Px4GazeboSurrogateHarnessError(f"{name} must be finite")
    return result


def _finite_vector3(value: Any, *, name: str) -> tuple[float, float, float]:
    try:
        raw = tuple(value)
    except TypeError as exc:
        raise Px4GazeboSurrogateHarnessError(f"{name} must be a 3-vector") from exc
    if len(raw) != 3:
        raise Px4GazeboSurrogateHarnessError(f"{name} must be a 3-vector")
    return tuple(_finite_float(component, name=name) for component in raw)


def _uint32(value: Any, *, name: str) -> int:
    result = int(value)
    if result < 0 or result > 0xFFFFFFFF:
        raise Px4GazeboSurrogateHarnessError(f"{name} must fit uint32")
    return result


def _uint64(value: Any, *, name: str) -> int:
    result = int(value)
    if result < 0 or result > 0xFFFFFFFFFFFFFFFF:
        raise Px4GazeboSurrogateHarnessError(f"{name} must fit uint64")
    return result


__all__ = [
    "PHASE4B_NOT_SATISFIED_REASON",
    "SURROGATE_LABEL",
    "InjectedTelemetryTransport",
    "InjectedVisionPacketTransport",
    "Px4EstimatedTelemetrySample",
    "Px4GazeboSurrogateHarness",
    "Px4GazeboSurrogateHarnessError",
    "Px4GazeboSurrogateScenario",
    "Px4GazeboSurrogateScenarioResult",
    "Px4GazeboSurrogateScenarioStep",
    "Px4GazeboSurrogateSummary",
    "SurrogateDecodedFrame",
    "SurrogateMavlinkMessage",
    "command_result_was_no_send",
    "deterministic_surrogate_jpeg_decoder",
    "make_surrogate_heartbeat",
    "packetize_vadr_frame_array",
    "packetize_vadr_jpeg_bytes",
    "telemetry_sample_to_fake_mavlink_messages",
    "telemetry_samples_to_fake_mavlink_messages",
]
