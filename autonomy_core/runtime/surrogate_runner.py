"""Import-safe PX4/Gazebo surrogate runner.

Phase 8.5D-3 adds bounded receive-only PX4 MAVLink surrogate modes that
feed telemetry, generated/saved image packets, and optional lazy ROS camera
pixels into the competition runner through injected batches. It does not
send heartbeats, setpoints, attitude targets, actuator commands, or any
other MAVLink command.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional


SURROGATE_LABEL = "PX4/Gazebo surrogate only"
PHASE_8_5D_1 = "8.5D-1"
PHASE_8_5D_2 = "8.5D-2"
PHASE_8_5D_3 = "8.5D-3"
PHASE = PHASE_8_5D_3
PHASE4B_NOT_SATISFIED = (
    "PX4/Gazebo surrogate output does not satisfy Phase 4B real competition "
    "simulator receive-only telemetry evidence."
)
COMPETITION_READINESS_NOT_CLAIMED = (
    "Surrogate runner output is not competition telemetry, command, race, or "
    "submitted-run readiness evidence."
)
DEFAULT_MAVLINK_ENDPOINT = "udpin:0.0.0.0:14540"
DEFAULT_CAMERA_TOPIC = "/camera/image_raw"


class SurrogateRunnerMode(str, Enum):
    MOCK_VISION_DRY_RUN = "mock_vision_dry_run"
    SAVED_IMAGE_VISION_DRY_RUN = "saved_image_vision_dry_run"
    PX4_OBSERVE = "px4_observe"
    PX4_VISION_DRY_RUN = "px4_vision_dry_run"
    PX4_COMMAND_DRY_RUN = "px4_command_dry_run"
    PX4_COMMAND_SEND = "px4_command_send"
    RACE = "race"
    COMPETITION_LIVE = "competition_live"


SUPPORTED_STAGE_8_5D_1_MODES = frozenset(
    {
        SurrogateRunnerMode.MOCK_VISION_DRY_RUN,
        SurrogateRunnerMode.SAVED_IMAGE_VISION_DRY_RUN,
        SurrogateRunnerMode.PX4_OBSERVE,
        SurrogateRunnerMode.PX4_VISION_DRY_RUN,
        SurrogateRunnerMode.PX4_COMMAND_DRY_RUN,
    }
)

EXECUTABLE_STAGE_8_5D_2_MODES = frozenset(
    {
        SurrogateRunnerMode.MOCK_VISION_DRY_RUN,
        SurrogateRunnerMode.SAVED_IMAGE_VISION_DRY_RUN,
    }
)

EXECUTABLE_STAGE_8_5D_3_MODES = frozenset(
    {
        *EXECUTABLE_STAGE_8_5D_2_MODES,
        SurrogateRunnerMode.PX4_OBSERVE,
        SurrogateRunnerMode.PX4_VISION_DRY_RUN,
        SurrogateRunnerMode.PX4_COMMAND_DRY_RUN,
    }
)

FAIL_CLOSED_MODES = frozenset(
    {
        SurrogateRunnerMode.PX4_COMMAND_SEND,
        SurrogateRunnerMode.RACE,
        SurrogateRunnerMode.COMPETITION_LIVE,
    }
)

VISION_SOURCE_KINDS = frozenset({"none", "generated_mock", "saved_image", "ros_camera"})


class SurrogateRunnerSafetyError(RuntimeError):
    """Raised when a surrogate runner mode violates Phase 8.5D gates."""


@dataclass(frozen=True)
class SurrogateRunnerConfig:
    mode: SurrogateRunnerMode | str = SurrogateRunnerMode.MOCK_VISION_DRY_RUN
    source_kind: str = "none"
    vision_source_kind: str = "none"
    input_image_path: Optional[str] = None
    resize_input_image: bool = False
    resize_camera_to_competition: bool = False
    frame_id: int = 1
    sim_time_ns: int = 1_234_567_890
    max_payload_size: int = 1200
    jpeg_quality: int = 90
    mavlink_endpoint: str = DEFAULT_MAVLINK_ENDPOINT
    duration_s: float = 5.0
    poll_sleep_s: float = 0.001
    max_messages: int = 0
    injected_mavlink_messages: tuple[Any, ...] = field(default_factory=tuple)
    camera_topic: str = DEFAULT_CAMERA_TOPIC
    command_tuple: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.5)
    use_real_autonomy: bool = False
    enable_surrogate_command_send: bool = False
    operator_command_send_confirmation: bool = False


@dataclass
class SurrogateRunnerSummary:
    surrogate_label: str = SURROGATE_LABEL
    phase: str = PHASE_8_5D_1
    mode: str = SurrogateRunnerMode.MOCK_VISION_DRY_RUN.value
    source_kind: str = "none"
    vision_source_kind: str = "none"
    status: str = "not_started"
    fail_closed: bool = False
    safety_error: Optional[str] = None
    telemetry_sample_count: int = 0
    telemetry_message_types: tuple[str, ...] = ()
    telemetry_summary: Optional[dict[str, Any]] = None
    mavlink_endpoint: Optional[str] = None
    duration_s: Optional[float] = None
    heartbeat_seen: bool = False
    heartbeat_age_s: Optional[float] = None
    position_source: str = ""
    attitude_source: str = ""
    frame_count: int = 0
    completed_packetized_frames: int = 0
    vision_packets_processed: int = 0
    vision_frames_completed: int = 0
    perception_update_calls: int = 0
    state_usable: bool = False
    state_missing_reasons: tuple[str, ...] = ()
    command_candidate_count: int = 0
    command_publication_allowed: bool = False
    command_sent_count: int = 0
    command_blocked_reasons: tuple[str, ...] = (
        "phase8_5d_1_no_runtime_execution",
        "phase4b_telemetry_evidence_missing",
    )
    guard_rejection_count: int = 0
    sockets_opened: bool = False
    ros_initialized: bool = False
    mavsdk_connected: bool = False
    pymavlink_connected: bool = False
    cv2_loaded_by_runner: bool = False
    fake_autonomy_used: bool = False
    autonomy_instantiated: bool = False
    competition_runner_executed: bool = False
    source_image_path: Optional[str] = None
    camera_topic: Optional[str] = None
    image_shape: Optional[tuple[int, ...]] = None
    image_dtype: Optional[str] = None
    image_stamp_sec: Optional[int] = None
    image_stamp_nanosec: Optional[int] = None
    camera_matrix: Optional[list[list[float]]] = None
    dist_coeffs: Optional[list[float]] = None
    gazebo_pose: Any = None
    image_pose_snapshot: Any = None
    runner_events: tuple[str, ...] = ()
    phase4b_satisfied: bool = False
    competition_readiness_claimed: bool = False
    notes: tuple[str, ...] = (
        PHASE4B_NOT_SATISFIED,
        COMPETITION_READINESS_NOT_CLAIMED,
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "surrogate_label": self.surrogate_label,
            "phase": self.phase,
            "mode": self.mode,
            "source_kind": self.source_kind,
            "vision_source_kind": self.vision_source_kind,
            "status": self.status,
            "fail_closed": self.fail_closed,
            "safety_error": self.safety_error,
            "telemetry_sample_count": self.telemetry_sample_count,
            "telemetry_message_types": list(self.telemetry_message_types),
            "telemetry_summary": self.telemetry_summary,
            "mavlink_endpoint": self.mavlink_endpoint,
            "duration_s": self.duration_s,
            "heartbeat_seen": self.heartbeat_seen,
            "heartbeat_age_s": self.heartbeat_age_s,
            "position_source": self.position_source,
            "attitude_source": self.attitude_source,
            "frame_count": self.frame_count,
            "completed_packetized_frames": self.completed_packetized_frames,
            "vision_packets_processed": self.vision_packets_processed,
            "vision_frames_completed": self.vision_frames_completed,
            "perception_update_calls": self.perception_update_calls,
            "state_usable": self.state_usable,
            "state_missing_reasons": list(self.state_missing_reasons),
            "command_candidate_count": self.command_candidate_count,
            "command_publication_allowed": self.command_publication_allowed,
            "command_sent_count": self.command_sent_count,
            "command_blocked_reasons": list(self.command_blocked_reasons),
            "guard_rejection_count": self.guard_rejection_count,
            "sockets_opened": self.sockets_opened,
            "ros_initialized": self.ros_initialized,
            "mavsdk_connected": self.mavsdk_connected,
            "pymavlink_connected": self.pymavlink_connected,
            "cv2_loaded_by_runner": self.cv2_loaded_by_runner,
            "fake_autonomy_used": self.fake_autonomy_used,
            "autonomy_instantiated": self.autonomy_instantiated,
            "competition_runner_executed": self.competition_runner_executed,
            "source_image_path": self.source_image_path,
            "camera_topic": self.camera_topic,
            "image_shape": None if self.image_shape is None else list(self.image_shape),
            "image_dtype": self.image_dtype,
            "image_stamp_sec": self.image_stamp_sec,
            "image_stamp_nanosec": self.image_stamp_nanosec,
            "camera_matrix": self.camera_matrix,
            "dist_coeffs": self.dist_coeffs,
            "gazebo_pose": self.gazebo_pose,
            "image_pose_snapshot": self.image_pose_snapshot,
            "runner_events": list(self.runner_events),
            "phase4b_satisfied": self.phase4b_satisfied,
            "competition_readiness_claimed": self.competition_readiness_claimed,
            "notes.txt": list(self.notes),
        }


class _FakeSurrogateAutonomy:
    """Injected autonomy boundary for surrogate dry-runs only."""

    def __init__(self, command_tuple: tuple[float, float, float, float]) -> None:
        self.command_tuple = command_tuple
        self.perception_updates: list[dict[str, Any]] = []

    def update_gate_memory_from_frame(self, **kwargs: Any) -> None:
        self.perception_updates.append(dict(kwargs))

    def attitude_control(self) -> tuple[float, float, float, float]:
        return self.command_tuple


class SurrogateRunner:
    """PX4/Gazebo surrogate entrypoint into competition-stack dry-runs."""

    def __init__(self, config: SurrogateRunnerConfig = SurrogateRunnerConfig()):
        self.config = config
        self.mode = normalize_surrogate_mode(config.mode)
        self._assert_startup_safety()

    def run(self) -> SurrogateRunnerSummary:
        """Run the selected supported surrogate mode."""

        if self.mode == SurrogateRunnerMode.MOCK_VISION_DRY_RUN:
            return self._run_mock_vision_dry_run()
        if self.mode == SurrogateRunnerMode.SAVED_IMAGE_VISION_DRY_RUN:
            return self._run_saved_image_vision_dry_run()
        if self.mode == SurrogateRunnerMode.PX4_OBSERVE:
            return self._run_px4_surrogate(competition_mode="observe")
        if self.mode == SurrogateRunnerMode.PX4_VISION_DRY_RUN:
            return self._run_px4_surrogate(competition_mode="vision_dry_run")
        if self.mode == SurrogateRunnerMode.PX4_COMMAND_DRY_RUN:
            return self._run_px4_surrogate(competition_mode="command_dry_run")

        return SurrogateRunnerSummary(
            phase=PHASE_8_5D_1,
            mode=self.mode.value,
            source_kind=self.config.source_kind,
            status="skeleton_ready_no_runtime_execution",
        )

    def _run_mock_vision_dry_run(self) -> SurrogateRunnerSummary:
        packets, source_kind, source_image_path, ros_initialized = self._build_vision_packets(
            required=True,
            default_source="generated_mock",
        )
        return self._run_vision_packets(
            packets=packets,
            source_kind=source_kind,
            vision_source_kind=self._resolved_vision_source_kind(default="generated_mock"),
            source_image_path=source_image_path,
            ros_initialized=ros_initialized,
        )

    def _run_saved_image_vision_dry_run(self) -> SurrogateRunnerSummary:
        packets, source_kind, source_image_path, ros_initialized = self._build_vision_packets(
            required=True,
            default_source="saved_image",
        )
        return self._run_vision_packets(
            packets=packets,
            source_kind=source_kind,
            vision_source_kind=self._resolved_vision_source_kind(default="saved_image"),
            source_image_path=source_image_path,
            ros_initialized=ros_initialized,
        )

    def _run_vision_packets(
        self,
        *,
        packets: tuple[bytes, ...],
        source_kind: str,
        vision_source_kind: str,
        source_image_path: Optional[str],
        ros_initialized: bool = False,
    ) -> SurrogateRunnerSummary:
        CompetitionImageAdapter, CompetitionRunner, config_types = _load_competition_runner_types()
        CompetitionRunnerConfig, CompetitionRunnerMode, CompetitionRunnerSafetyConfig = config_types

        autonomy = _FakeSurrogateAutonomy(self.config.command_tuple)
        runner = CompetitionRunner(
            config=CompetitionRunnerConfig(
                mode=CompetitionRunnerMode.VISION_DRY_RUN,
                safety=CompetitionRunnerSafetyConfig(
                    phase4b_telemetry_evidence_available=False,
                    command_publication_enabled=False,
                    allow_live_command_modes=False,
                ),
            ),
            image_adapter=CompetitionImageAdapter(),
            autonomy=autonomy,
        )
        result = runner.step(vision_packets=packets)
        last_update = autonomy.perception_updates[-1] if autonomy.perception_updates else {}
        return _summary_from_step_result(
            phase=PHASE_8_5D_2,
            mode=self.mode,
            source_kind=source_kind,
            vision_source_kind=vision_source_kind,
            status="competition_stack_vision_dry_run_complete",
            source_image_path=source_image_path,
            packets=packets,
            telemetry_messages=(),
            telemetry_summary=None,
            mavlink_endpoint=None,
            duration_s=None,
            camera_topic=None,
            result=result,
            last_update=last_update,
            sockets_opened=False,
            pymavlink_connected=False,
            ros_initialized=ros_initialized,
            fake_autonomy_used=True,
        )

    def _run_px4_surrogate(self, *, competition_mode: str) -> SurrogateRunnerSummary:
        CompetitionImageAdapter, CompetitionRunner, config_types = _load_competition_runner_types()
        CompetitionRunnerConfig, CompetitionRunnerMode, CompetitionRunnerSafetyConfig = config_types

        telemetry_messages, telemetry_summary, sockets_opened, pymavlink_connected = (
            self._collect_mavlink_messages()
        )
        packets: tuple[bytes, ...] = ()
        source_kind = _source_kind(self.config, default="px4_estimated_telemetry")
        source_image_path = None
        ros_initialized = False

        if competition_mode in {"vision_dry_run", "command_dry_run"}:
            packets, source_kind, source_image_path, ros_initialized = self._build_vision_packets(
                required=competition_mode == "vision_dry_run",
                default_source="none",
            )

        mode_map = {
            "observe": CompetitionRunnerMode.OBSERVE,
            "vision_dry_run": CompetitionRunnerMode.VISION_DRY_RUN,
            "command_dry_run": CompetitionRunnerMode.COMMAND_DRY_RUN,
        }
        runner_mode = mode_map[competition_mode]
        autonomy = None
        if runner_mode in {
            CompetitionRunnerMode.VISION_DRY_RUN,
            CompetitionRunnerMode.COMMAND_DRY_RUN,
        }:
            autonomy = _FakeSurrogateAutonomy(self.config.command_tuple)

        runner = CompetitionRunner(
            config=CompetitionRunnerConfig(
                mode=runner_mode,
                safety=CompetitionRunnerSafetyConfig(
                    phase4b_telemetry_evidence_available=False,
                    command_publication_enabled=False,
                    allow_live_command_modes=False,
                ),
            ),
            image_adapter=CompetitionImageAdapter(),
            autonomy=autonomy,
        )
        result = runner.step(
            telemetry_messages=telemetry_messages,
            vision_packets=packets,
        )
        last_update = {}
        if autonomy is not None and autonomy.perception_updates:
            last_update = autonomy.perception_updates[-1]

        status = {
            "observe": "competition_stack_px4_observe_complete",
            "vision_dry_run": "competition_stack_px4_vision_dry_run_complete",
            "command_dry_run": "competition_stack_px4_command_dry_run_complete",
        }[competition_mode]
        return _summary_from_step_result(
            phase=PHASE_8_5D_3,
            mode=self.mode,
            source_kind=source_kind,
            vision_source_kind=self._resolved_vision_source_kind(default="none"),
            status=status,
            source_image_path=source_image_path,
            packets=packets,
            telemetry_messages=telemetry_messages,
            telemetry_summary=telemetry_summary,
            mavlink_endpoint=self.config.mavlink_endpoint,
            duration_s=self.config.duration_s,
            camera_topic=self.config.camera_topic if ros_initialized else None,
            result=result,
            last_update=last_update,
            sockets_opened=sockets_opened,
            pymavlink_connected=pymavlink_connected,
            ros_initialized=ros_initialized,
            fake_autonomy_used=autonomy is not None,
        )

    def _collect_mavlink_messages(
        self,
    ) -> tuple[tuple[Any, ...], dict[str, Any], bool, bool]:
        if self.config.injected_mavlink_messages:
            messages = tuple(self.config.injected_mavlink_messages)
            return messages, _inventory_summary_for_messages(messages), False, False

        messages, summary = receive_mavlink_messages(
            endpoint=self.config.mavlink_endpoint,
            duration_s=self.config.duration_s,
            poll_sleep_s=self.config.poll_sleep_s,
            max_messages=self.config.max_messages,
        )
        return messages, summary, True, True

    def _build_vision_packets(
        self,
        *,
        required: bool,
        default_source: str,
    ) -> tuple[tuple[bytes, ...], str, Optional[str], bool]:
        vision_source = self._resolved_vision_source_kind(default=default_source)
        if vision_source == "none":
            if required:
                raise SurrogateRunnerSafetyError(
                    "this mode requires --vision-source generated_mock, saved_image, or ros_camera"
                )
            return (), _source_kind(self.config, default="px4_estimated_telemetry"), None, False

        if vision_source == "generated_mock":
            build_packetized_mock_frame = _load_build_packetized_mock_frame()
            _source_frame, packetized = build_packetized_mock_frame(
                frame_id=self.config.frame_id,
                sim_time_ns=self.config.sim_time_ns,
                max_payload_size=self.config.max_payload_size,
                jpeg_quality=self.config.jpeg_quality,
            )
            return (
                packetized.packets,
                _source_kind(self.config, default="generated_mock"),
                None,
                False,
            )

        if vision_source == "saved_image":
            if not self.config.input_image_path:
                raise SurrogateRunnerSafetyError(
                    f"{self.mode.value} with saved_image requires --input-image"
                )
            frame = _load_saved_image(
                self.config.input_image_path,
                resize_input_image=self.config.resize_input_image,
            )
            encode_jpeg, packetize_jpeg_bytes = _load_saved_image_packetizers()
            jpeg_bytes = encode_jpeg(frame, quality=self.config.jpeg_quality)
            packetized = packetize_jpeg_bytes(
                jpeg_bytes,
                frame_id=self.config.frame_id,
                sim_time_ns=self.config.sim_time_ns,
                max_payload_size=self.config.max_payload_size,
            )
            return (
                packetized.packets,
                _source_kind(self.config, default="saved_image"),
                str(Path(self.config.input_image_path)),
                False,
            )

        if vision_source == "ros_camera":
            frame = _capture_ros_camera_frame_as_bgr(
                topic=self.config.camera_topic,
                duration_s=self.config.duration_s,
            )
            frame = _resize_ros_camera_frame_for_competition(
                frame,
                resize_enabled=self.config.resize_camera_to_competition,
            )
            encode_jpeg, packetize_jpeg_bytes = _load_saved_image_packetizers()
            jpeg_bytes = encode_jpeg(frame, quality=self.config.jpeg_quality)
            packetized = packetize_jpeg_bytes(
                jpeg_bytes,
                frame_id=self.config.frame_id,
                sim_time_ns=self.config.sim_time_ns,
                max_payload_size=self.config.max_payload_size,
            )
            return (
                packetized.packets,
                _source_kind(self.config, default="ros_camera_pixels_only"),
                None,
                True,
            )

        raise SurrogateRunnerSafetyError(f"unknown vision source: {vision_source}")

    def _resolved_vision_source_kind(self, *, default: str) -> str:
        value = str(self.config.vision_source_kind or default)
        if value == "none" and default != "none":
            value = default
        return value

    def _assert_startup_safety(self) -> None:
        if self.mode in FAIL_CLOSED_MODES:
            raise SurrogateRunnerSafetyError(
                f"{self.mode.value} is fail-closed in the current Phase 8.5D implementation"
            )
        if self.config.use_real_autonomy:
            raise SurrogateRunnerSafetyError(
                "real AutonomyAPI ownership is not enabled in Phase 8.5D-3"
            )
        if self.config.enable_surrogate_command_send:
            raise SurrogateRunnerSafetyError(
                "surrogate command send is not enabled in Phase 8.5D-3"
            )
        if self.config.operator_command_send_confirmation:
            raise SurrogateRunnerSafetyError(
                "operator command-send confirmation is reserved for a later "
                "PX4/Gazebo-only command-send phase"
            )
        if self.mode not in SUPPORTED_STAGE_8_5D_1_MODES:
            raise SurrogateRunnerSafetyError(
                f"{self.mode.value} is not supported in the current Phase 8.5D implementation"
            )
        vision_source = str(self.config.vision_source_kind or "none")
        if vision_source not in VISION_SOURCE_KINDS:
            raise SurrogateRunnerSafetyError(f"unknown vision source: {vision_source}")
        if self.config.duration_s <= 0.0:
            raise SurrogateRunnerSafetyError("duration_s must be positive")
        if self.config.poll_sleep_s < 0.0:
            raise SurrogateRunnerSafetyError("poll_sleep_s must be non-negative")
        if self.config.max_messages < 0:
            raise SurrogateRunnerSafetyError("max_messages must be non-negative")


def normalize_surrogate_mode(mode: SurrogateRunnerMode | str) -> SurrogateRunnerMode:
    if isinstance(mode, SurrogateRunnerMode):
        return mode
    try:
        return SurrogateRunnerMode(str(mode))
    except ValueError as exc:
        raise SurrogateRunnerSafetyError(f"unknown surrogate runner mode: {mode}") from exc


def fail_closed_summary(
    *,
    mode: SurrogateRunnerMode | str,
    source_kind: str = "none",
    error: str,
) -> SurrogateRunnerSummary:
    normalized_mode = mode.value if isinstance(mode, SurrogateRunnerMode) else str(mode)
    phase = (
        PHASE_8_5D_3
        if normalized_mode in {mode.value for mode in EXECUTABLE_STAGE_8_5D_3_MODES}
        else PHASE_8_5D_1
    )
    return SurrogateRunnerSummary(
        phase=phase,
        mode=normalized_mode,
        source_kind=source_kind,
        status="fail_closed",
        fail_closed=True,
        safety_error=str(error),
        command_blocked_reasons=(
            "phase8_5d_fail_closed",
            "phase4b_telemetry_evidence_missing",
        ),
    )


def run_surrogate_runner(config: SurrogateRunnerConfig) -> SurrogateRunnerSummary:
    runner = SurrogateRunner(config)
    return runner.run()


def receive_mavlink_messages(
    *,
    endpoint: str = DEFAULT_MAVLINK_ENDPOINT,
    duration_s: float = 5.0,
    poll_sleep_s: float = 0.001,
    max_messages: int = 0,
    connection: Any = None,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Receive MAVLink messages without sending anything.

    This helper opens a pymavlink connection only when `connection` is not
    provided. It never calls wait_heartbeat(), mav.send(), heartbeat_send(), or
    any setpoint/command method.
    """

    if duration_s <= 0.0:
        raise SurrogateRunnerSafetyError("duration_s must be positive")
    if poll_sleep_s < 0.0:
        raise SurrogateRunnerSafetyError("poll_sleep_s must be non-negative")
    if max_messages < 0:
        raise SurrogateRunnerSafetyError("max_messages must be non-negative")

    mavlink_connection = connection if connection is not None else _open_mavlink_connection(endpoint)
    messages: list[Any] = []
    inventory = _new_mavlink_inventory()
    started = float(clock())
    deadline = started + float(duration_s)

    while float(clock()) < deadline:
        message = mavlink_connection.recv_match(blocking=False)
        now = float(clock())
        if message is None:
            if poll_sleep_s > 0.0:
                sleep(poll_sleep_s)
            continue
        if _message_type(message) == "BAD_DATA":
            continue
        messages.append(message)
        inventory.observe_message(message, received_wall_time=now)
        if max_messages and len(messages) >= max_messages:
            break

    summary = inventory.summary()
    summary.update(
        {
            "endpoint": endpoint,
            "duration_s": float(duration_s),
            "started_wall_time": started,
            "finished_wall_time": float(clock()),
            "message_count": len(messages),
            "receive_only": True,
        }
    )
    return tuple(messages), summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 8.5D PX4/Gazebo surrogate runner. Implemented modes feed "
            "receive-only PX4 MAVLink telemetry and generated/saved/optional "
            "ROS camera pixels through CompetitionRunner injected batches. No "
            "commands are sent."
        )
    )
    parser.add_argument(
        "mode",
        choices=[mode.value for mode in SurrogateRunnerMode],
        help="surrogate runner mode",
    )
    parser.add_argument(
        "--source-kind",
        default="none",
        help="source label included in the surrogate summary",
    )
    parser.add_argument(
        "--vision-source",
        choices=sorted(VISION_SOURCE_KINDS),
        default="none",
        dest="vision_source_kind",
        help="vision source for PX4 vision/command dry-runs",
    )
    parser.add_argument(
        "--input-image",
        default=None,
        help="image path required by saved_image vision source",
    )
    parser.add_argument(
        "--resize-input-image",
        action="store_true",
        help=(
            "resize saved image to the official 640x360 surrogate resolution; "
            "without this flag saved images must already match"
        ),
    )
    parser.add_argument(
        "--resize-camera-to-competition",
        action="store_true",
        help=(
            "surrogate-only: resize ROS camera pixels to the official 640x360 "
            "VADR resolution before JPEG packetization"
        ),
    )
    parser.add_argument(
        "--frame-id",
        type=int,
        default=1,
        help="VADR frame_id for generated/saved-image dry-runs",
    )
    parser.add_argument(
        "--sim-time-ns",
        type=int,
        default=1_234_567_890,
        help="VADR sim_time_ns for generated/saved-image dry-runs",
    )
    parser.add_argument(
        "--max-payload-size",
        type=int,
        default=1200,
        help="maximum VADR JPEG payload bytes per packet",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality for generated/saved-image packetization",
    )
    parser.add_argument(
        "--mavlink-endpoint",
        default=DEFAULT_MAVLINK_ENDPOINT,
        help=f"receive-only pymavlink endpoint, default: {DEFAULT_MAVLINK_ENDPOINT}",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=5.0,
        help="bounded receive duration for PX4 surrogate modes",
    )
    parser.add_argument(
        "--poll-sleep-s",
        type=float,
        default=0.001,
        help="sleep interval when no MAVLink message is available",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=0,
        help="optional receive cap; 0 means duration-only",
    )
    parser.add_argument(
        "--camera-topic",
        default=DEFAULT_CAMERA_TOPIC,
        help="ROS sensor_msgs/Image topic used only with --vision-source ros_camera",
    )
    parser.add_argument("--command-roll", type=float, default=0.0)
    parser.add_argument("--command-pitch", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--command-thrust", type=float, default=0.5)
    parser.add_argument(
        "--use-real-autonomy",
        action="store_true",
        help="reserved for a later phase; rejected in Phase 8.5D-3",
    )
    parser.add_argument(
        "--enable-surrogate-command-send",
        action="store_true",
        help="reserved for a later PX4/Gazebo-only command-send phase; rejected now",
    )
    parser.add_argument(
        "--i-understand-this-sends-to-px4-gazebo-only",
        action="store_true",
        dest="operator_command_send_confirmation",
        help="reserved for a later command-send phase; rejected now",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = SurrogateRunnerConfig(
        mode=args.mode,
        source_kind=args.source_kind,
        vision_source_kind=args.vision_source_kind,
        input_image_path=args.input_image,
        resize_input_image=bool(args.resize_input_image),
        resize_camera_to_competition=bool(args.resize_camera_to_competition),
        frame_id=int(args.frame_id),
        sim_time_ns=int(args.sim_time_ns),
        max_payload_size=int(args.max_payload_size),
        jpeg_quality=int(args.jpeg_quality),
        mavlink_endpoint=str(args.mavlink_endpoint),
        duration_s=float(args.duration_s),
        poll_sleep_s=float(args.poll_sleep_s),
        max_messages=int(args.max_messages),
        camera_topic=str(args.camera_topic),
        command_tuple=(
            float(args.command_roll),
            float(args.command_pitch),
            float(args.command_yaw),
            float(args.command_thrust),
        ),
        use_real_autonomy=bool(args.use_real_autonomy),
        enable_surrogate_command_send=bool(args.enable_surrogate_command_send),
        operator_command_send_confirmation=bool(args.operator_command_send_confirmation),
    )
    try:
        summary = run_surrogate_runner(config)
        exit_code = 0
    except SurrogateRunnerSafetyError as exc:
        summary = fail_closed_summary(
            mode=args.mode,
            source_kind=args.source_kind,
            error=str(exc),
        )
        exit_code = 2

    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return exit_code


def _source_kind(config: SurrogateRunnerConfig, *, default: str) -> str:
    return default if config.source_kind == "none" else str(config.source_kind)


def _summary_from_step_result(
    *,
    phase: str,
    mode: SurrogateRunnerMode,
    source_kind: str,
    vision_source_kind: str,
    status: str,
    source_image_path: Optional[str],
    packets: tuple[bytes, ...],
    telemetry_messages: tuple[Any, ...],
    telemetry_summary: Optional[dict[str, Any]],
    mavlink_endpoint: Optional[str],
    duration_s: Optional[float],
    camera_topic: Optional[str],
    result: Any,
    last_update: dict[str, Any],
    sockets_opened: bool,
    pymavlink_connected: bool,
    ros_initialized: bool,
    fake_autonomy_used: bool,
) -> SurrogateRunnerSummary:
    frame = last_update.get("frame")
    camera_matrix = last_update.get("camera_matrix")
    dist_coeffs = last_update.get("dist_coeffs")
    notes = [
        PHASE4B_NOT_SATISFIED,
        COMPETITION_READINESS_NOT_CLAIMED,
        f"processed {len(telemetry_messages)} receive-only MAVLink message(s)",
        f"processed {len(packets)} injected VADR vision packet(s)",
    ]
    return SurrogateRunnerSummary(
        phase=phase,
        mode=mode.value,
        source_kind=source_kind,
        vision_source_kind=vision_source_kind,
        status=status,
        telemetry_sample_count=len(telemetry_messages),
        telemetry_message_types=tuple(_message_type(message) for message in telemetry_messages),
        telemetry_summary=telemetry_summary,
        mavlink_endpoint=mavlink_endpoint if sockets_opened else None,
        duration_s=duration_s if sockets_opened else None,
        heartbeat_seen=bool(result.heartbeat_seen),
        heartbeat_age_s=result.heartbeat_age_s,
        position_source=str(result.state_result.position_source),
        attitude_source=str(result.state_result.attitude_source),
        frame_count=1 if packets else 0,
        completed_packetized_frames=1 if result.vision_frames_completed else 0,
        vision_packets_processed=int(result.vision_packets_processed),
        vision_frames_completed=int(result.vision_frames_completed),
        perception_update_calls=int(result.perception_update_calls),
        state_usable=bool(result.state_result.is_usable),
        state_missing_reasons=tuple(result.state_result.missing_reasons),
        command_candidate_count=1 if result.command_candidate_attempted else 0,
        command_publication_allowed=False,
        command_sent_count=0,
        command_blocked_reasons=tuple(result.command_blocked_reasons),
        sockets_opened=sockets_opened,
        ros_initialized=ros_initialized,
        pymavlink_connected=pymavlink_connected,
        cv2_loaded_by_runner="cv2" in sys.modules,
        fake_autonomy_used=fake_autonomy_used,
        autonomy_instantiated=False,
        competition_runner_executed=True,
        source_image_path=source_image_path,
        camera_topic=camera_topic,
        image_shape=(
            None
            if frame is None
            else tuple(int(value) for value in getattr(frame, "shape", ()))
        ),
        image_dtype=None if frame is None else str(getattr(frame, "dtype", "")),
        image_stamp_sec=last_update.get("image_stamp_sec"),
        image_stamp_nanosec=last_update.get("image_stamp_nanosec"),
        camera_matrix=(
            None
            if camera_matrix is None
            else [[float(value) for value in row] for row in camera_matrix.tolist()]
        ),
        dist_coeffs=(
            None
            if dist_coeffs is None
            else [float(value) for value in dist_coeffs.reshape(-1).tolist()]
        ),
        gazebo_pose=last_update.get("gazebo_pose"),
        image_pose_snapshot=last_update.get("image_pose_snapshot"),
        runner_events=tuple(result.events),
        notes=tuple(notes),
    )


def _load_competition_runner_types():
    from autonomy_core.perception.competition_image_adapter import CompetitionImageAdapter
    from autonomy_core.runtime.competition_runner import (
        CompetitionRunner,
        CompetitionRunnerConfig,
        CompetitionRunnerMode,
        CompetitionRunnerSafetyConfig,
    )

    return (
        CompetitionImageAdapter,
        CompetitionRunner,
        (CompetitionRunnerConfig, CompetitionRunnerMode, CompetitionRunnerSafetyConfig),
    )


def _load_build_packetized_mock_frame():
    from autonomy_core.tools.competition_vision_udp_loopback import (
        build_packetized_mock_frame,
    )

    return build_packetized_mock_frame


def _load_saved_image_packetizers():
    from autonomy_core.tools.competition_vision_udp_loopback import (
        encode_jpeg,
        packetize_jpeg_bytes,
    )

    return encode_jpeg, packetize_jpeg_bytes


def _load_saved_image(
    input_image_path: str,
    *,
    resize_input_image: bool,
):
    from autonomy_core.core.competition_config import VADR_TS_002

    cv2 = _import_cv2()
    path = Path(input_image_path)
    if not path.exists():
        raise SurrogateRunnerSafetyError(f"input image does not exist: {path}")
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise SurrogateRunnerSafetyError(f"failed to read image: {path}")
    expected_shape = (
        VADR_TS_002.camera_height_px,
        VADR_TS_002.camera_width_px,
    )
    if tuple(frame.shape[:2]) != expected_shape:
        if not resize_input_image:
            raise SurrogateRunnerSafetyError(
                "input image must be 640x360 unless --resize-input-image is set"
            )
        frame = cv2.resize(
            frame,
            (VADR_TS_002.camera_width_px, VADR_TS_002.camera_height_px),
            interpolation=cv2.INTER_AREA,
        )
    return frame


def _capture_ros_camera_frame_as_bgr(*, topic: str, duration_s: float):
    """Capture one ROS Image frame lazily, without using TF or Gazebo truth."""

    try:
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.node import Node
        from sensor_msgs.msg import Image
    except ModuleNotFoundError as exc:
        raise SurrogateRunnerSafetyError(
            "ros_camera vision source requires rclpy, sensor_msgs, and cv_bridge; "
            "imports remain lazy and no ROS is initialized unless this source is selected"
        ) from exc

    bridge = CvBridge()
    frame_box: dict[str, Any] = {}

    class _OneFrameNode(Node):
        def __init__(self) -> None:
            super().__init__("surrogate_runner_one_frame")
            self.create_subscription(Image, topic, self._callback, 1)

        def _callback(self, message: Any) -> None:
            if "frame" not in frame_box:
                frame_box["frame"] = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")

    started_here = not rclpy.ok()
    if started_here:
        rclpy.init(args=None)
    node = _OneFrameNode()
    deadline = time.time() + float(duration_s)
    try:
        while "frame" not in frame_box and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.destroy_node()
        if started_here:
            rclpy.shutdown()

    frame = frame_box.get("frame")
    if frame is None:
        raise SurrogateRunnerSafetyError(
            f"timed out waiting for ROS camera frame on {topic!r}"
        )
    return frame


def _resize_ros_camera_frame_for_competition(frame: Any, *, resize_enabled: bool):
    """Validate or explicitly resize ROS camera pixels to VADR resolution."""

    from autonomy_core.core.competition_config import VADR_TS_002

    shape = tuple(int(value) for value in getattr(frame, "shape", ()))
    expected_hw = (VADR_TS_002.camera_height_px, VADR_TS_002.camera_width_px)
    if len(shape) < 2:
        raise SurrogateRunnerSafetyError("ROS camera frame does not expose image shape")
    if shape[:2] == expected_hw:
        return frame
    if not resize_enabled:
        raise SurrogateRunnerSafetyError(
            "ROS camera frame is "
            f"{shape[1]}x{shape[0]}, but the competition adapter requires "
            f"{VADR_TS_002.camera_width_px}x{VADR_TS_002.camera_height_px}; "
            "rerun with --resize-camera-to-competition for surrogate-only resizing"
        )

    cv2 = _import_cv2()
    return cv2.resize(
        frame,
        (VADR_TS_002.camera_width_px, VADR_TS_002.camera_height_px),
        interpolation=cv2.INTER_AREA,
    )


def _import_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise SurrogateRunnerSafetyError(
            "cv2 is required only when executing image surrogate dry-runs"
        ) from exc
    return cv2


def _open_mavlink_connection(endpoint: str):
    try:
        from pymavlink import mavutil
    except ModuleNotFoundError as exc:
        raise SurrogateRunnerSafetyError(
            "pymavlink is required only when executing live PX4 surrogate receive modes"
        ) from exc
    return mavutil.mavlink_connection(endpoint)


def _new_mavlink_inventory():
    from autonomy_core.tools.competition_mavlink_observe import MavlinkTelemetryInventory

    return MavlinkTelemetryInventory()


def _inventory_summary_for_messages(messages: tuple[Any, ...]) -> dict[str, Any]:
    inventory = _new_mavlink_inventory()
    for message in messages:
        inventory.observe_message(message, received_wall_time=time.time())
    summary = inventory.summary()
    summary.update(
        {
            "endpoint": "injected",
            "duration_s": 0.0,
            "message_count": len(messages),
            "receive_only": True,
        }
    )
    return summary


def _message_type(message: Any) -> str:
    get_type = getattr(message, "get_type", None)
    if callable(get_type):
        return str(get_type())
    if isinstance(message, dict) and "type" in message:
        return str(message["type"])
    return str(getattr(message, "mavpackettype", type(message).__name__))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPETITION_READINESS_NOT_CLAIMED",
    "DEFAULT_CAMERA_TOPIC",
    "DEFAULT_MAVLINK_ENDPOINT",
    "FAIL_CLOSED_MODES",
    "EXECUTABLE_STAGE_8_5D_2_MODES",
    "EXECUTABLE_STAGE_8_5D_3_MODES",
    "PHASE",
    "PHASE_8_5D_1",
    "PHASE_8_5D_2",
    "PHASE_8_5D_3",
    "PHASE4B_NOT_SATISFIED",
    "SUPPORTED_STAGE_8_5D_1_MODES",
    "SURROGATE_LABEL",
    "SurrogateRunner",
    "SurrogateRunnerConfig",
    "SurrogateRunnerMode",
    "SurrogateRunnerSafetyError",
    "SurrogateRunnerSummary",
    "VISION_SOURCE_KINDS",
    "build_arg_parser",
    "fail_closed_summary",
    "main",
    "normalize_surrogate_mode",
    "receive_mavlink_messages",
    "run_surrogate_runner",
]
