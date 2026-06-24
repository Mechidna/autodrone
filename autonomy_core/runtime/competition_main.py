"""Import-safe competition executable lifecycle for Phase 6D/6E.

This module parses/executes bounded dry-run modes around `competition_setup.py`.
It does not enable race mode or start live transports unless the CLI explicitly
requests `--live-transports`. Phase 9D can send PX4/Gazebo surrogate commands
only behind explicit acknowledgement flags; that output is not competition
readiness evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

from autonomy_core.command.competition_command_adapter import (
    build_dry_run_set_attitude_target_fields,
    quaternion_wxyz_from_euler_zyx,
)
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
from autonomy_core.runtime.px4_gazebo_command_sender import (
    ATTITUDE_HOVER_BODY_RATES_IGNORE_TYPE_MASK,
    ATTITUDE_HOVER_ZERO_BODY_RATES,
    BODY_RATE_ATTITUDE_IGNORE_TYPE_MASK,
    BODY_RATE_DUMMY_QUATERNION,
    PX4_GAZEBO_ARM_OFFBOARD_LABEL,
    PX4_GAZEBO_ATTITUDE_HOVER_SMOKE_LABEL,
    PX4_GAZEBO_BODY_RATE_SMOKE_LABEL,
    PX4_GAZEBO_SURROGATE_LABEL,
    Px4GazeboCommandSenderConfig,
    Px4GazeboSetAttitudeTargetSender,
)
from autonomy_core.runtime.competition_setpoint_streamer import (
    CompetitionSetpointStreamConfig,
    CompetitionSetpointStreamer,
)


PHASE_6D = "6D"
PHASE_6E = "6E"
PHASE_9B = "9B"
PHASE_9C = "9C"
PHASE_9D = "9D"
PHASE_9E = "9E"
PHASE_9E_1 = "9E.1"
PHASE_9E_2 = "9E.2"
PHASE_9E_3 = "9E.3"
PHASE_9E_4 = "9E.4"
PHASE_9F = "9F"
PHASE_9F_1 = "9F.1"
PHASE_9F_2 = "9F.2"
PHASE_9F_3B = "9F.3B"
PHASE_9F_3C = "9F.3C"
PHASE4B_NOT_SATISFIED = "Phase 4B remains blocked pending real competition telemetry evidence."
COMPETITION_READINESS_NOT_CLAIMED = "Dry-run output is not competition readiness evidence."
PHASE6E_SURROGATE_LIMITATION = (
    "Phase 6E live receive dry-run can use PX4/Gazebo surrogate evidence, but "
    "that does not satisfy Phase 4B or Phase 9 real competition simulator evidence."
)
PHASE9D_SURROGATE_LIMITATION = (
    "Phase 9D sends PX4/Gazebo surrogate commands only; it does not satisfy "
    "Phase 4B, real competition simulator evidence, or competition readiness."
)
PHASE9E_SURROGATE_LIMITATION = (
    "Phase 9E arms/sets mode for PX4/Gazebo surrogate testing only; it does "
    "not satisfy Phase 4B, real competition simulator evidence, or competition readiness."
)
PHASE9E1_SURROGATE_LIMITATION = (
    "Phase 9E.1 continuously streams bounded PX4/Gazebo surrogate setpoints "
    "from the latest accepted competition command candidate; it does not "
    "satisfy Phase 4B, real competition simulator evidence, or competition readiness."
)
PHASE9E2_SURROGATE_LIMITATION = (
    "Phase 9E.2 applies PX4/Gazebo surrogate-only command safety clamps and "
    "stream diagnostics at the sender boundary; it does not change competition "
    "controller output or satisfy Phase 4B, real competition simulator evidence, "
    "or competition readiness."
)
PHASE9E3_SURROGATE_LIMITATION = (
    "Phase 9E.3 sends fixed PX4/Gazebo surrogate-only body-rate "
    "SET_ATTITUDE_TARGET messages without using perception, planning, or "
    "AutonomyAPI.attitude_control(); it does not satisfy Phase 4B, real "
    "competition simulator evidence, or competition readiness."
)
PHASE9E4_SURROGATE_LIMITATION = (
    "Phase 9E.4 sends fixed PX4/Gazebo surrogate-only attitude-angle hover "
    "SET_ATTITUDE_TARGET messages to mirror the legacy MAVSDK hover interface; "
    "it does not satisfy Phase 4B, real competition simulator evidence, or "
    "competition readiness."
)
PHASE9F_SURROGATE_LIMITATION = (
    "Phase 9F closes the PX4/Gazebo surrogate autonomy loop through real "
    "competition-stack perception, planning, dry-run command adaptation, and "
    "PX4/Gazebo-only MAVLink command sending; it does not satisfy Phase 4B, "
    "real competition simulator evidence, or competition readiness."
)
PHASE9F1_SURROGATE_LIMITATION = (
    "Phase 9F.1 applies a PX4/Gazebo surrogate-only debug yaw override at the "
    "MAVLink sender boundary and tightens full-loop health criteria; it does "
    "not change AutonomyAPI, planner, controller, perception, or competition "
    "command semantics."
)
PHASE9F2_SURROGATE_LIMITATION = (
    "Phase 9F.2 adds a PX4/Gazebo surrogate-only fixed-rate cached setpoint "
    "stream before Offboard mode request; it does not change AutonomyAPI, "
    "planner, controller, perception, or competition command semantics."
)
PHASE9F3B_SURROGATE_LIMITATION = (
    "Phase 9F.3B routes the PX4/Gazebo surrogate fixed-rate sender through "
    "the generic competition setpoint streamer policy; it does not change "
    "AutonomyAPI, planner, controller, perception, or competition command "
    "semantics."
)
PHASE9F3C_SURROGATE_LIMITATION = (
    "Phase 9F.3C adds an explicit PX4/Gazebo surrogate-only fallback/hold "
    "setpoint for the generic streamer when autonomy commands are missing or "
    "stale; it does not change planner, controller, perception, or competition "
    "command semantics."
)
PX4_GAZEBO_FULL_AUTONOMY_LOOP_LABEL = (
    "PX4/Gazebo surrogate full autonomy loop only"
)
PX4_GAZEBO_DEBUG_YAW_OVERRIDE_LABEL = (
    "PX4/Gazebo surrogate debug yaw override only"
)
PX4_GAZEBO_FIXED_RATE_SETPOINT_STREAM_LABEL = (
    "PX4/Gazebo surrogate fixed-rate setpoint stream only"
)
PX4_GAZEBO_GENERIC_SETPOINT_STREAMER_LABEL = (
    "PX4/Gazebo surrogate generic setpoint streamer integration only"
)
PX4_GAZEBO_GENERIC_SETPOINT_FALLBACK_LABEL = (
    "PX4/Gazebo surrogate generic setpoint fallback only"
)
PHASE9F_MIN_COMMAND_SEND_RATE_HZ = 10.0
PHASE9F_MIN_VISION_FRAME_RATE_HZ = 5.0
PHASE9F_MAX_SEND_GAP_S = 0.5
PHASE9F2_DEFAULT_SETPOINT_STREAM_HZ = 20.0
PHASE9F2_MAX_SETPOINT_STREAM_HZ = 99.0


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
    px4_gazebo_command_send: bool = False
    ack_px4_gazebo_surrogate_command_send: bool = False
    px4_gazebo_command_max_count: int = 1
    px4_gazebo_command_max_heartbeat_age_s: float = 1.5
    px4_gazebo_command_min_period_s: float = 0.01
    px4_gazebo_command_min_thrust: float = 0.0
    px4_gazebo_command_max_thrust: float = 1.0
    px4_gazebo_command_max_abs_roll_pitch_rad: float = 0.7
    px4_gazebo_command_max_abs_body_rate_rad_s: float = 2.0
    px4_gazebo_surrogate_thrust_clamp: bool = False
    px4_gazebo_surrogate_thrust_clamp_min: Optional[float] = None
    px4_gazebo_surrogate_thrust_clamp_max: Optional[float] = None
    px4_gazebo_continuous_setpoint_stream: bool = False
    px4_gazebo_command_max_age_s: float = 0.5
    px4_gazebo_arm: bool = False
    ack_px4_gazebo_surrogate_arm: bool = False
    px4_gazebo_offboard: bool = False
    ack_px4_gazebo_surrogate_offboard: bool = False
    px4_gazebo_arm_max_attempts: int = 1
    px4_gazebo_offboard_max_attempts: int = 1
    px4_gazebo_offboard_prestream_count: int = 0
    px4_gazebo_body_rate_smoke: bool = False
    ack_px4_gazebo_surrogate_body_rate_smoke: bool = False
    px4_gazebo_body_roll_rate: float = 0.0
    px4_gazebo_body_pitch_rate: float = 0.0
    px4_gazebo_body_yaw_rate: float = 0.0
    px4_gazebo_body_rate_thrust: Optional[float] = None
    px4_gazebo_attitude_hover_smoke: bool = False
    ack_px4_gazebo_surrogate_attitude_hover_smoke: bool = False
    px4_gazebo_attitude_hover_roll_rad: float = 0.0
    px4_gazebo_attitude_hover_pitch_rad: float = 0.0
    px4_gazebo_attitude_hover_yaw_rad: Optional[float] = None
    px4_gazebo_attitude_hover_thrust: Optional[float] = None
    px4_gazebo_full_autonomy_loop: bool = False
    ack_px4_gazebo_surrogate_full_autonomy_loop: bool = False
    px4_gazebo_debug_yaw_override_rad: Optional[float] = None
    ack_px4_gazebo_surrogate_debug_yaw_override: bool = False
    px4_gazebo_fixed_rate_setpoint_stream: bool = False
    ack_px4_gazebo_surrogate_fixed_rate_setpoint_stream: bool = False
    px4_gazebo_setpoint_stream_hz: float = PHASE9F2_DEFAULT_SETPOINT_STREAM_HZ
    px4_gazebo_setpoint_stream_burst_limit: int = 2
    px4_gazebo_generic_setpoint_streamer: bool = False
    ack_px4_gazebo_surrogate_generic_setpoint_streamer: bool = False
    px4_gazebo_generic_setpoint_fallback: bool = False
    ack_px4_gazebo_surrogate_generic_setpoint_fallback: bool = False
    px4_gazebo_generic_fallback_roll_rad: float = 0.0
    px4_gazebo_generic_fallback_pitch_rad: float = 0.0
    px4_gazebo_generic_fallback_yaw_rad: Optional[float] = None
    px4_gazebo_generic_fallback_thrust: Optional[float] = None


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
    px4_gazebo_command_send_requested: bool = False
    px4_gazebo_command_send_acknowledged: bool = False
    px4_gazebo_surrogate_label: str = PX4_GAZEBO_SURROGATE_LABEL
    px4_gazebo_surrogate_thrust_clamp_requested: bool = False
    px4_gazebo_surrogate_thrust_clamp_min: Optional[float] = None
    px4_gazebo_surrogate_thrust_clamp_max: Optional[float] = None
    px4_gazebo_arm_requested: bool = False
    px4_gazebo_arm_acknowledged: bool = False
    px4_gazebo_offboard_requested: bool = False
    px4_gazebo_offboard_acknowledged: bool = False
    px4_gazebo_offboard_prestream_count: int = 0
    px4_gazebo_arm_offboard_label: str = PX4_GAZEBO_ARM_OFFBOARD_LABEL
    px4_gazebo_body_rate_smoke_requested: bool = False
    px4_gazebo_body_rate_smoke_acknowledged: bool = False
    px4_gazebo_body_rate_smoke_label: str = PX4_GAZEBO_BODY_RATE_SMOKE_LABEL
    body_rate_type_mask: Optional[int] = None
    body_rate_q: tuple[float, float, float, float] = ()
    body_roll_rate: Optional[float] = None
    body_pitch_rate: Optional[float] = None
    body_yaw_rate: Optional[float] = None
    body_rate_thrust: Optional[float] = None
    body_rate_command_sent_count: int = 0
    body_rate_command_rejection_count: int = 0
    last_body_rate_command_send_result: Optional[dict[str, Any]] = None
    px4_gazebo_attitude_hover_smoke_requested: bool = False
    px4_gazebo_attitude_hover_smoke_acknowledged: bool = False
    px4_gazebo_attitude_hover_smoke_label: str = PX4_GAZEBO_ATTITUDE_HOVER_SMOKE_LABEL
    attitude_hover_type_mask: Optional[int] = None
    attitude_hover_body_rates: tuple[float, float, float] = ()
    attitude_hover_roll_rad: Optional[float] = None
    attitude_hover_pitch_rad: Optional[float] = None
    attitude_hover_yaw_rad: Optional[float] = None
    attitude_hover_yaw_source: str = ""
    attitude_hover_thrust: Optional[float] = None
    attitude_hover_command_sent_count: int = 0
    attitude_hover_command_rejection_count: int = 0
    last_attitude_hover_command_send_result: Optional[dict[str, Any]] = None
    px4_gazebo_full_autonomy_loop_requested: bool = False
    px4_gazebo_full_autonomy_loop_acknowledged: bool = False
    px4_gazebo_full_autonomy_loop_label: str = PX4_GAZEBO_FULL_AUTONOMY_LOOP_LABEL
    phase9f_command_backend: str = ""
    phase9f_command_type_mask: Optional[int] = None
    px4_gazebo_debug_yaw_override_requested: bool = False
    px4_gazebo_debug_yaw_override_acknowledged: bool = False
    px4_gazebo_debug_yaw_override_rad: Optional[float] = None
    px4_gazebo_debug_yaw_override_applied_count: int = 0
    px4_gazebo_debug_yaw_override_label: str = PX4_GAZEBO_DEBUG_YAW_OVERRIDE_LABEL
    px4_gazebo_fixed_rate_setpoint_stream_requested: bool = False
    px4_gazebo_fixed_rate_setpoint_stream_acknowledged: bool = False
    px4_gazebo_setpoint_stream_hz: float = PHASE9F2_DEFAULT_SETPOINT_STREAM_HZ
    px4_gazebo_setpoint_stream_burst_limit: int = 2
    px4_gazebo_fixed_rate_setpoint_stream_label: str = (
        PX4_GAZEBO_FIXED_RATE_SETPOINT_STREAM_LABEL
    )
    fixed_rate_setpoint_stream_iteration_count: int = 0
    fixed_rate_setpoint_stream_attempt_count: int = 0
    fixed_rate_setpoint_stream_sent_count: int = 0
    fixed_rate_setpoint_stream_rejection_count: int = 0
    px4_gazebo_generic_setpoint_streamer_requested: bool = False
    px4_gazebo_generic_setpoint_streamer_acknowledged: bool = False
    px4_gazebo_generic_setpoint_streamer_label: str = (
        PX4_GAZEBO_GENERIC_SETPOINT_STREAMER_LABEL
    )
    generic_setpoint_streamer_summary: Optional[dict[str, Any]] = None
    px4_gazebo_generic_setpoint_fallback_requested: bool = False
    px4_gazebo_generic_setpoint_fallback_acknowledged: bool = False
    px4_gazebo_generic_setpoint_fallback_label: str = (
        PX4_GAZEBO_GENERIC_SETPOINT_FALLBACK_LABEL
    )
    generic_setpoint_fallback_roll_rad: Optional[float] = None
    generic_setpoint_fallback_pitch_rad: Optional[float] = None
    generic_setpoint_fallback_yaw_rad: Optional[float] = None
    generic_setpoint_fallback_yaw_source: str = ""
    generic_setpoint_fallback_thrust: Optional[float] = None
    generic_setpoint_fallback_update_count: int = 0
    generic_setpoint_fallback_rejection_count: int = 0
    generic_setpoint_fallback_last_rejection: str = ""
    phase9f_command_send_rate_hz: Optional[float] = None
    phase9f_vision_frame_rate_hz: Optional[float] = None
    phase9f_max_send_gap_s: Optional[float] = None
    arm_attempt_count: int = 0
    arm_sent_count: int = 0
    arm_rejection_count: int = 0
    last_arm_result: Optional[dict[str, Any]] = None
    offboard_attempt_count: int = 0
    offboard_sent_count: int = 0
    offboard_rejection_count: int = 0
    last_offboard_result: Optional[dict[str, Any]] = None
    armed_state_observed: bool = False
    offboard_state_observed: bool = False
    command_send_attempt_count: int = 0
    command_sent_count: int = 0
    command_send_rejection_count: int = 0
    last_command_send_result: Optional[dict[str, Any]] = None
    px4_gazebo_continuous_setpoint_stream_requested: bool = False
    px4_gazebo_command_max_age_s: float = 0.5
    setpoint_stream_cache_update_count: int = 0
    setpoint_stream_reused_count: int = 0
    setpoint_stream_stale_rejection_count: int = 0
    last_cached_command_age_s: Optional[float] = None
    command_sender_summary: Optional[dict[str, Any]] = None
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
    phase9d_surrogate_command_send_satisfied: bool = False
    phase9d_success_criteria: dict[str, bool] = field(default_factory=dict)
    phase9e_surrogate_arm_offboard_satisfied: bool = False
    phase9e_success_criteria: dict[str, bool] = field(default_factory=dict)
    phase9e1_continuous_setpoint_stream_satisfied: bool = False
    phase9e1_success_criteria: dict[str, bool] = field(default_factory=dict)
    phase9e2_surrogate_command_safety_satisfied: bool = False
    phase9e2_success_criteria: dict[str, bool] = field(default_factory=dict)
    phase9e3_body_rate_interface_satisfied: bool = False
    phase9e3_success_criteria: dict[str, bool] = field(default_factory=dict)
    phase9e4_attitude_hover_interface_satisfied: bool = False
    phase9e4_success_criteria: dict[str, bool] = field(default_factory=dict)
    phase9f_full_autonomy_loop_satisfied: bool = False
    phase9f_success_criteria: dict[str, bool] = field(default_factory=dict)
    phase9f2_fixed_rate_setpoint_stream_satisfied: bool = False
    phase9f2_success_criteria: dict[str, bool] = field(default_factory=dict)
    phase9f3b_generic_setpoint_streamer_satisfied: bool = False
    phase9f3b_success_criteria: dict[str, bool] = field(default_factory=dict)
    phase9f3c_fallback_setpoint_satisfied: bool = False
    phase9f3c_success_criteria: dict[str, bool] = field(default_factory=dict)
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
            "px4_gazebo_command_send_requested": (
                self.px4_gazebo_command_send_requested
            ),
            "px4_gazebo_command_send_acknowledged": (
                self.px4_gazebo_command_send_acknowledged
            ),
            "px4_gazebo_surrogate_label": self.px4_gazebo_surrogate_label,
            "px4_gazebo_surrogate_thrust_clamp_requested": (
                self.px4_gazebo_surrogate_thrust_clamp_requested
            ),
            "px4_gazebo_surrogate_thrust_clamp_min": (
                self.px4_gazebo_surrogate_thrust_clamp_min
            ),
            "px4_gazebo_surrogate_thrust_clamp_max": (
                self.px4_gazebo_surrogate_thrust_clamp_max
            ),
            "px4_gazebo_arm_requested": self.px4_gazebo_arm_requested,
            "px4_gazebo_arm_acknowledged": self.px4_gazebo_arm_acknowledged,
            "px4_gazebo_offboard_requested": self.px4_gazebo_offboard_requested,
            "px4_gazebo_offboard_acknowledged": (
                self.px4_gazebo_offboard_acknowledged
            ),
            "px4_gazebo_offboard_prestream_count": (
                self.px4_gazebo_offboard_prestream_count
            ),
            "px4_gazebo_arm_offboard_label": self.px4_gazebo_arm_offboard_label,
            "px4_gazebo_body_rate_smoke_requested": (
                self.px4_gazebo_body_rate_smoke_requested
            ),
            "px4_gazebo_body_rate_smoke_acknowledged": (
                self.px4_gazebo_body_rate_smoke_acknowledged
            ),
            "px4_gazebo_body_rate_smoke_label": (
                self.px4_gazebo_body_rate_smoke_label
            ),
            "body_rate_type_mask": self.body_rate_type_mask,
            "body_rate_q": list(self.body_rate_q),
            "body_roll_rate": self.body_roll_rate,
            "body_pitch_rate": self.body_pitch_rate,
            "body_yaw_rate": self.body_yaw_rate,
            "body_rate_thrust": self.body_rate_thrust,
            "body_rate_command_sent_count": self.body_rate_command_sent_count,
            "body_rate_command_rejection_count": (
                self.body_rate_command_rejection_count
            ),
            "last_body_rate_command_send_result": (
                self.last_body_rate_command_send_result
            ),
            "px4_gazebo_attitude_hover_smoke_requested": (
                self.px4_gazebo_attitude_hover_smoke_requested
            ),
            "px4_gazebo_attitude_hover_smoke_acknowledged": (
                self.px4_gazebo_attitude_hover_smoke_acknowledged
            ),
            "px4_gazebo_attitude_hover_smoke_label": (
                self.px4_gazebo_attitude_hover_smoke_label
            ),
            "attitude_hover_type_mask": self.attitude_hover_type_mask,
            "attitude_hover_body_rates": list(self.attitude_hover_body_rates),
            "attitude_hover_roll_rad": self.attitude_hover_roll_rad,
            "attitude_hover_pitch_rad": self.attitude_hover_pitch_rad,
            "attitude_hover_yaw_rad": self.attitude_hover_yaw_rad,
            "attitude_hover_yaw_source": self.attitude_hover_yaw_source,
            "attitude_hover_thrust": self.attitude_hover_thrust,
            "attitude_hover_command_sent_count": (
                self.attitude_hover_command_sent_count
            ),
            "attitude_hover_command_rejection_count": (
                self.attitude_hover_command_rejection_count
            ),
            "last_attitude_hover_command_send_result": (
                self.last_attitude_hover_command_send_result
            ),
            "px4_gazebo_full_autonomy_loop_requested": (
                self.px4_gazebo_full_autonomy_loop_requested
            ),
            "px4_gazebo_full_autonomy_loop_acknowledged": (
                self.px4_gazebo_full_autonomy_loop_acknowledged
            ),
            "px4_gazebo_full_autonomy_loop_label": (
                self.px4_gazebo_full_autonomy_loop_label
            ),
            "phase9f_command_backend": self.phase9f_command_backend,
            "phase9f_command_type_mask": self.phase9f_command_type_mask,
            "px4_gazebo_debug_yaw_override_requested": (
                self.px4_gazebo_debug_yaw_override_requested
            ),
            "px4_gazebo_debug_yaw_override_acknowledged": (
                self.px4_gazebo_debug_yaw_override_acknowledged
            ),
            "px4_gazebo_debug_yaw_override_rad": (
                self.px4_gazebo_debug_yaw_override_rad
            ),
            "px4_gazebo_debug_yaw_override_applied_count": (
                self.px4_gazebo_debug_yaw_override_applied_count
            ),
            "px4_gazebo_debug_yaw_override_label": (
                self.px4_gazebo_debug_yaw_override_label
            ),
            "px4_gazebo_fixed_rate_setpoint_stream_requested": (
                self.px4_gazebo_fixed_rate_setpoint_stream_requested
            ),
            "px4_gazebo_fixed_rate_setpoint_stream_acknowledged": (
                self.px4_gazebo_fixed_rate_setpoint_stream_acknowledged
            ),
            "px4_gazebo_setpoint_stream_hz": self.px4_gazebo_setpoint_stream_hz,
            "px4_gazebo_setpoint_stream_burst_limit": (
                self.px4_gazebo_setpoint_stream_burst_limit
            ),
            "px4_gazebo_fixed_rate_setpoint_stream_label": (
                self.px4_gazebo_fixed_rate_setpoint_stream_label
            ),
            "fixed_rate_setpoint_stream_iteration_count": (
                self.fixed_rate_setpoint_stream_iteration_count
            ),
            "fixed_rate_setpoint_stream_attempt_count": (
                self.fixed_rate_setpoint_stream_attempt_count
            ),
            "fixed_rate_setpoint_stream_sent_count": (
                self.fixed_rate_setpoint_stream_sent_count
            ),
            "fixed_rate_setpoint_stream_rejection_count": (
                self.fixed_rate_setpoint_stream_rejection_count
            ),
            "px4_gazebo_generic_setpoint_streamer_requested": (
                self.px4_gazebo_generic_setpoint_streamer_requested
            ),
            "px4_gazebo_generic_setpoint_streamer_acknowledged": (
                self.px4_gazebo_generic_setpoint_streamer_acknowledged
            ),
            "px4_gazebo_generic_setpoint_streamer_label": (
                self.px4_gazebo_generic_setpoint_streamer_label
            ),
            "generic_setpoint_streamer_summary": (
                None
                if self.generic_setpoint_streamer_summary is None
                else dict(self.generic_setpoint_streamer_summary)
            ),
            "px4_gazebo_generic_setpoint_fallback_requested": (
                self.px4_gazebo_generic_setpoint_fallback_requested
            ),
            "px4_gazebo_generic_setpoint_fallback_acknowledged": (
                self.px4_gazebo_generic_setpoint_fallback_acknowledged
            ),
            "px4_gazebo_generic_setpoint_fallback_label": (
                self.px4_gazebo_generic_setpoint_fallback_label
            ),
            "generic_setpoint_fallback_roll_rad": (
                self.generic_setpoint_fallback_roll_rad
            ),
            "generic_setpoint_fallback_pitch_rad": (
                self.generic_setpoint_fallback_pitch_rad
            ),
            "generic_setpoint_fallback_yaw_rad": (
                self.generic_setpoint_fallback_yaw_rad
            ),
            "generic_setpoint_fallback_yaw_source": (
                self.generic_setpoint_fallback_yaw_source
            ),
            "generic_setpoint_fallback_thrust": self.generic_setpoint_fallback_thrust,
            "generic_setpoint_fallback_update_count": (
                self.generic_setpoint_fallback_update_count
            ),
            "generic_setpoint_fallback_rejection_count": (
                self.generic_setpoint_fallback_rejection_count
            ),
            "generic_setpoint_fallback_last_rejection": (
                self.generic_setpoint_fallback_last_rejection
            ),
            "phase9f_command_send_rate_hz": self.phase9f_command_send_rate_hz,
            "phase9f_vision_frame_rate_hz": self.phase9f_vision_frame_rate_hz,
            "phase9f_max_send_gap_s": self.phase9f_max_send_gap_s,
            "arm_attempt_count": self.arm_attempt_count,
            "arm_sent_count": self.arm_sent_count,
            "arm_rejection_count": self.arm_rejection_count,
            "last_arm_result": self.last_arm_result,
            "offboard_attempt_count": self.offboard_attempt_count,
            "offboard_sent_count": self.offboard_sent_count,
            "offboard_rejection_count": self.offboard_rejection_count,
            "last_offboard_result": self.last_offboard_result,
            "armed_state_observed": self.armed_state_observed,
            "offboard_state_observed": self.offboard_state_observed,
            "command_send_attempt_count": self.command_send_attempt_count,
            "command_sent_count": self.command_sent_count,
            "command_send_rejection_count": self.command_send_rejection_count,
            "last_command_send_result": self.last_command_send_result,
            "px4_gazebo_continuous_setpoint_stream_requested": (
                self.px4_gazebo_continuous_setpoint_stream_requested
            ),
            "px4_gazebo_command_max_age_s": self.px4_gazebo_command_max_age_s,
            "setpoint_stream_cache_update_count": (
                self.setpoint_stream_cache_update_count
            ),
            "setpoint_stream_reused_count": self.setpoint_stream_reused_count,
            "setpoint_stream_stale_rejection_count": (
                self.setpoint_stream_stale_rejection_count
            ),
            "last_cached_command_age_s": self.last_cached_command_age_s,
            "command_sender_summary": self.command_sender_summary,
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
            "phase9d_surrogate_command_send_satisfied": (
                self.phase9d_surrogate_command_send_satisfied
            ),
            "phase9d_success_criteria": dict(sorted(self.phase9d_success_criteria.items())),
            "phase9e_surrogate_arm_offboard_satisfied": (
                self.phase9e_surrogate_arm_offboard_satisfied
            ),
            "phase9e_success_criteria": dict(sorted(self.phase9e_success_criteria.items())),
            "phase9e1_continuous_setpoint_stream_satisfied": (
                self.phase9e1_continuous_setpoint_stream_satisfied
            ),
            "phase9e1_success_criteria": dict(
                sorted(self.phase9e1_success_criteria.items())
            ),
            "phase9e2_surrogate_command_safety_satisfied": (
                self.phase9e2_surrogate_command_safety_satisfied
            ),
            "phase9e2_success_criteria": dict(
                sorted(self.phase9e2_success_criteria.items())
            ),
            "phase9e3_body_rate_interface_satisfied": (
                self.phase9e3_body_rate_interface_satisfied
            ),
            "phase9e3_success_criteria": dict(
                sorted(self.phase9e3_success_criteria.items())
            ),
            "phase9e4_attitude_hover_interface_satisfied": (
                self.phase9e4_attitude_hover_interface_satisfied
            ),
            "phase9e4_success_criteria": dict(
                sorted(self.phase9e4_success_criteria.items())
            ),
            "phase9f_full_autonomy_loop_satisfied": (
                self.phase9f_full_autonomy_loop_satisfied
            ),
            "phase9f_success_criteria": dict(
                sorted(self.phase9f_success_criteria.items())
            ),
            "phase9f2_fixed_rate_setpoint_stream_satisfied": (
                self.phase9f2_fixed_rate_setpoint_stream_satisfied
            ),
            "phase9f2_success_criteria": dict(
                sorted(self.phase9f2_success_criteria.items())
            ),
            "phase9f3b_generic_setpoint_streamer_satisfied": (
                self.phase9f3b_generic_setpoint_streamer_satisfied
            ),
            "phase9f3b_success_criteria": dict(
                sorted(self.phase9f3b_success_criteria.items())
            ),
            "phase9f3c_fallback_setpoint_satisfied": (
                self.phase9f3c_fallback_setpoint_satisfied
            ),
            "phase9f3c_success_criteria": dict(
                sorted(self.phase9f3c_success_criteria.items())
            ),
            "competition_readiness_claimed": self.competition_readiness_claimed,
            "notes.txt": list(self.notes),
        }


def run_competition_main(
    config: CompetitionMainConfig = CompetitionMainConfig(),
    *,
    components: Optional[CompetitionRuntimeComponents] = None,
    components_factory: Callable[..., CompetitionRuntimeComponents] = build_competition_runtime,
    command_sender: Optional[Any] = None,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> CompetitionMainSummary:
    """Run a bounded Phase 6D dry-run loop."""

    mode = _normalize_mode(config.mode)
    _assert_main_config_safe(config, mode, components=components)

    started = float(clock())
    active_components = components or components_factory(_setup_config_from_main(config, mode))
    active_command_sender = _command_sender_for_config(
        config,
        active_components,
        command_sender=command_sender,
        clock=clock,
    )
    setpoint_streamer = _setpoint_streamer_for_config(config)
    steps_completed = 0
    aggregate = _Aggregate()
    deadline = None if config.duration_s <= 0.0 else started + float(config.duration_s)

    try:
        for _step_index in range(int(config.steps)):
            if deadline is not None and float(clock()) > deadline:
                break
            result = active_components.runner.step()
            aggregate.record(result)
            _maybe_update_phase9f3b_setpoint_streamer(
                config=config,
                result=result,
                setpoint_streamer=setpoint_streamer,
            )
            _maybe_update_phase9f3c_fallback_setpoint(
                config=config,
                result=result,
                aggregate=aggregate,
                setpoint_streamer=setpoint_streamer,
            )
            fixed_rate_results = _maybe_send_phase9f2_fixed_rate_setpoints(
                config=config,
                result=result,
                aggregate=aggregate,
                command_sender=active_command_sender,
                setpoint_streamer=setpoint_streamer,
                clock=clock,
                sleep=sleep,
            )
            for fixed_rate_result in fixed_rate_results:
                aggregate.record_command_send(fixed_rate_result)
                aggregate.record_fixed_rate_setpoint_stream(fixed_rate_result)
            lifecycle_results = _maybe_send_phase9e_lifecycle(
                config=config,
                result=result,
                aggregate=aggregate,
                command_sender=active_command_sender,
            )
            for lifecycle_name, lifecycle_result in lifecycle_results:
                aggregate.record_lifecycle_send(lifecycle_name, lifecycle_result)
            body_rate_result = _maybe_send_phase9e3_body_rate_smoke(
                config=config,
                result=result,
                aggregate=aggregate,
                command_sender=active_command_sender,
            )
            if body_rate_result is not None:
                aggregate.record_body_rate_command_send(body_rate_result)
            attitude_hover_result = _maybe_send_phase9e4_attitude_hover_smoke(
                config=config,
                result=result,
                aggregate=aggregate,
                command_sender=active_command_sender,
            )
            if attitude_hover_result is not None:
                aggregate.record_attitude_hover_command_send(attitude_hover_result)
            if not config.px4_gazebo_fixed_rate_setpoint_stream:
                send_result = _maybe_send_phase9d_command(
                    config=config,
                    result=result,
                    aggregate=aggregate,
                    command_sender=active_command_sender,
                )
                if send_result is not None:
                    aggregate.record_command_send(send_result)
            steps_completed += 1
            if config.step_sleep_s > 0.0:
                sleep(float(config.step_sleep_s))
    finally:
        close = getattr(active_components, "close", None)
        if callable(close):
            close()

    finished = float(clock())
    if setpoint_streamer is not None:
        aggregate.generic_setpoint_streamer_summary = setpoint_streamer.summary()
    elapsed_s = max(0.0, finished - started)
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
    phase9d_criteria = _phase9d_success_criteria(
        config=config,
        mode=mode,
        aggregate=aggregate,
        phase6e_receive_satisfied=phase6e_receive_satisfied,
        phase6e_perception_boundary_satisfied=phase6e_perception_boundary_satisfied,
    )
    phase9d_satisfied = _phase9d_satisfied(phase9d_criteria)
    mavlink_transport_summary = _transport_summary(active_components.mavlink_transport)
    vision_transport_summary = _transport_summary(active_components.vision_transport)
    armed_state_observed = _armed_state_observed(mavlink_transport_summary)
    offboard_state_observed = _offboard_state_observed(mavlink_transport_summary)
    phase9e_criteria = _phase9e_success_criteria(
        config=config,
        mode=mode,
        aggregate=aggregate,
        phase6e_receive_satisfied=phase6e_receive_satisfied,
        phase6e_perception_boundary_satisfied=phase6e_perception_boundary_satisfied,
        armed_state_observed=armed_state_observed,
        offboard_state_observed=offboard_state_observed,
    )
    phase9e_satisfied = _phase9e_satisfied(phase9e_criteria, config)
    phase9e1_criteria = _phase9e1_success_criteria(
        config=config,
        mode=mode,
        aggregate=aggregate,
        phase9e_satisfied=phase9e_satisfied,
    )
    phase9e1_satisfied = _phase9e1_satisfied(phase9e1_criteria, config)
    command_sender_summary = _transport_summary(active_command_sender)
    phase9e2_criteria = _phase9e2_success_criteria(
        config=config,
        mode=mode,
        aggregate=aggregate,
        command_sender_summary=command_sender_summary,
    )
    phase9e2_satisfied = _phase9e2_satisfied(phase9e2_criteria, config)
    phase9e3_criteria = _phase9e3_success_criteria(
        config=config,
        mode=mode,
        aggregate=aggregate,
        armed_state_observed=armed_state_observed,
        offboard_state_observed=offboard_state_observed,
    )
    phase9e3_satisfied = _phase9e3_satisfied(phase9e3_criteria, config)
    phase9e4_criteria = _phase9e4_success_criteria(
        config=config,
        mode=mode,
        aggregate=aggregate,
        armed_state_observed=armed_state_observed,
        offboard_state_observed=offboard_state_observed,
    )
    phase9e4_satisfied = _phase9e4_satisfied(phase9e4_criteria, config)
    phase9f_criteria = _phase9f_success_criteria(
        config=config,
        mode=mode,
        aggregate=aggregate,
        duration_s=elapsed_s,
        command_sender_summary=command_sender_summary,
        phase6e_receive_satisfied=phase6e_receive_satisfied,
        phase6e_perception_boundary_satisfied=phase6e_perception_boundary_satisfied,
        phase9c_satisfied=phase9c_satisfied,
        phase9d_satisfied=phase9d_satisfied,
        phase9e_satisfied=phase9e_satisfied,
    )
    phase9f_satisfied = _phase9f_satisfied(phase9f_criteria, config)
    phase9f2_criteria = _phase9f2_success_criteria(
        config=config,
        mode=mode,
        aggregate=aggregate,
        phase9f_satisfied=phase9f_satisfied,
    )
    phase9f2_satisfied = _phase9f2_satisfied(phase9f2_criteria, config)
    phase9f3b_criteria = _phase9f3b_success_criteria(
        config=config,
        mode=mode,
        aggregate=aggregate,
        phase9f2_satisfied=phase9f2_satisfied,
    )
    phase9f3b_satisfied = _phase9f3b_satisfied(phase9f3b_criteria, config)
    phase9f3c_criteria = _phase9f3c_success_criteria(
        config=config,
        mode=mode,
        aggregate=aggregate,
    )
    phase9f3c_satisfied = _phase9f3c_satisfied(phase9f3c_criteria, config)

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
        duration_s=elapsed_s,
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
        command_publication_allowed=bool(
            config.px4_gazebo_command_send
            or config.px4_gazebo_arm
            or config.px4_gazebo_offboard
            or config.px4_gazebo_body_rate_smoke
            or config.px4_gazebo_attitude_hover_smoke
        ),
        px4_gazebo_command_send_requested=bool(config.px4_gazebo_command_send),
        px4_gazebo_command_send_acknowledged=bool(
            config.ack_px4_gazebo_surrogate_command_send
        ),
        px4_gazebo_surrogate_thrust_clamp_requested=bool(
            config.px4_gazebo_surrogate_thrust_clamp
        ),
        px4_gazebo_surrogate_thrust_clamp_min=(
            config.px4_gazebo_surrogate_thrust_clamp_min
        ),
        px4_gazebo_surrogate_thrust_clamp_max=(
            config.px4_gazebo_surrogate_thrust_clamp_max
        ),
        px4_gazebo_arm_requested=bool(config.px4_gazebo_arm),
        px4_gazebo_arm_acknowledged=bool(config.ack_px4_gazebo_surrogate_arm),
        px4_gazebo_offboard_requested=bool(config.px4_gazebo_offboard),
        px4_gazebo_offboard_acknowledged=bool(
            config.ack_px4_gazebo_surrogate_offboard
        ),
        px4_gazebo_offboard_prestream_count=int(
            config.px4_gazebo_offboard_prestream_count
        ),
        px4_gazebo_body_rate_smoke_requested=bool(
            config.px4_gazebo_body_rate_smoke
        ),
        px4_gazebo_body_rate_smoke_acknowledged=bool(
            config.ack_px4_gazebo_surrogate_body_rate_smoke
        ),
        body_rate_type_mask=(
            BODY_RATE_ATTITUDE_IGNORE_TYPE_MASK
            if config.px4_gazebo_body_rate_smoke
            else None
        ),
        body_rate_q=(
            BODY_RATE_DUMMY_QUATERNION
            if config.px4_gazebo_body_rate_smoke
            else ()
        ),
        body_roll_rate=(
            float(config.px4_gazebo_body_roll_rate)
            if config.px4_gazebo_body_rate_smoke
            else None
        ),
        body_pitch_rate=(
            float(config.px4_gazebo_body_pitch_rate)
            if config.px4_gazebo_body_rate_smoke
            else None
        ),
        body_yaw_rate=(
            float(config.px4_gazebo_body_yaw_rate)
            if config.px4_gazebo_body_rate_smoke
            else None
        ),
        body_rate_thrust=(
            float(config.px4_gazebo_body_rate_thrust)
            if config.px4_gazebo_body_rate_thrust is not None
            else None
        ),
        body_rate_command_sent_count=aggregate.body_rate_command_sent_count,
        body_rate_command_rejection_count=(
            aggregate.body_rate_command_rejection_count
        ),
        last_body_rate_command_send_result=(
            aggregate.last_body_rate_command_send_result
        ),
        px4_gazebo_attitude_hover_smoke_requested=bool(
            config.px4_gazebo_attitude_hover_smoke
        ),
        px4_gazebo_attitude_hover_smoke_acknowledged=bool(
            config.ack_px4_gazebo_surrogate_attitude_hover_smoke
        ),
        attitude_hover_type_mask=(
            ATTITUDE_HOVER_BODY_RATES_IGNORE_TYPE_MASK
            if config.px4_gazebo_attitude_hover_smoke
            else None
        ),
        attitude_hover_body_rates=(
            ATTITUDE_HOVER_ZERO_BODY_RATES
            if config.px4_gazebo_attitude_hover_smoke
            else ()
        ),
        attitude_hover_roll_rad=(
            float(config.px4_gazebo_attitude_hover_roll_rad)
            if config.px4_gazebo_attitude_hover_smoke
            else None
        ),
        attitude_hover_pitch_rad=(
            float(config.px4_gazebo_attitude_hover_pitch_rad)
            if config.px4_gazebo_attitude_hover_smoke
            else None
        ),
        attitude_hover_yaw_rad=aggregate.last_attitude_hover_yaw_rad,
        attitude_hover_yaw_source=aggregate.last_attitude_hover_yaw_source,
        attitude_hover_thrust=(
            float(config.px4_gazebo_attitude_hover_thrust)
            if config.px4_gazebo_attitude_hover_thrust is not None
            else None
        ),
        attitude_hover_command_sent_count=(
            aggregate.attitude_hover_command_sent_count
        ),
        attitude_hover_command_rejection_count=(
            aggregate.attitude_hover_command_rejection_count
        ),
        last_attitude_hover_command_send_result=(
            aggregate.last_attitude_hover_command_send_result
        ),
        px4_gazebo_full_autonomy_loop_requested=bool(
            config.px4_gazebo_full_autonomy_loop
        ),
        px4_gazebo_full_autonomy_loop_acknowledged=bool(
            config.ack_px4_gazebo_surrogate_full_autonomy_loop
        ),
        phase9f_command_backend=(
            "attitude_angle_quaternion_set_attitude_target"
            if config.px4_gazebo_full_autonomy_loop
            else ""
        ),
        phase9f_command_type_mask=(
            ATTITUDE_HOVER_BODY_RATES_IGNORE_TYPE_MASK
            if config.px4_gazebo_full_autonomy_loop
            else None
        ),
        px4_gazebo_debug_yaw_override_requested=(
            config.px4_gazebo_debug_yaw_override_rad is not None
        ),
        px4_gazebo_debug_yaw_override_acknowledged=bool(
            config.ack_px4_gazebo_surrogate_debug_yaw_override
        ),
        px4_gazebo_debug_yaw_override_rad=(
            float(config.px4_gazebo_debug_yaw_override_rad)
            if config.px4_gazebo_debug_yaw_override_rad is not None
            else None
        ),
        px4_gazebo_debug_yaw_override_applied_count=(
            aggregate.debug_yaw_override_applied_count
        ),
        px4_gazebo_fixed_rate_setpoint_stream_requested=bool(
            config.px4_gazebo_fixed_rate_setpoint_stream
        ),
        px4_gazebo_fixed_rate_setpoint_stream_acknowledged=bool(
            config.ack_px4_gazebo_surrogate_fixed_rate_setpoint_stream
        ),
        px4_gazebo_setpoint_stream_hz=float(config.px4_gazebo_setpoint_stream_hz),
        px4_gazebo_setpoint_stream_burst_limit=int(
            config.px4_gazebo_setpoint_stream_burst_limit
        ),
        fixed_rate_setpoint_stream_iteration_count=(
            aggregate.fixed_rate_setpoint_stream_iteration_count
        ),
        fixed_rate_setpoint_stream_attempt_count=(
            aggregate.fixed_rate_setpoint_stream_attempt_count
        ),
        fixed_rate_setpoint_stream_sent_count=(
            aggregate.fixed_rate_setpoint_stream_sent_count
        ),
        fixed_rate_setpoint_stream_rejection_count=(
            aggregate.fixed_rate_setpoint_stream_rejection_count
        ),
        px4_gazebo_generic_setpoint_streamer_requested=bool(
            config.px4_gazebo_generic_setpoint_streamer
        ),
        px4_gazebo_generic_setpoint_streamer_acknowledged=bool(
            config.ack_px4_gazebo_surrogate_generic_setpoint_streamer
        ),
        generic_setpoint_streamer_summary=aggregate.generic_setpoint_streamer_summary,
        px4_gazebo_generic_setpoint_fallback_requested=bool(
            config.px4_gazebo_generic_setpoint_fallback
        ),
        px4_gazebo_generic_setpoint_fallback_acknowledged=bool(
            config.ack_px4_gazebo_surrogate_generic_setpoint_fallback
        ),
        generic_setpoint_fallback_roll_rad=(
            float(config.px4_gazebo_generic_fallback_roll_rad)
            if config.px4_gazebo_generic_setpoint_fallback
            else None
        ),
        generic_setpoint_fallback_pitch_rad=(
            float(config.px4_gazebo_generic_fallback_pitch_rad)
            if config.px4_gazebo_generic_setpoint_fallback
            else None
        ),
        generic_setpoint_fallback_yaw_rad=(
            aggregate.last_generic_setpoint_fallback_yaw_rad
        ),
        generic_setpoint_fallback_yaw_source=(
            aggregate.last_generic_setpoint_fallback_yaw_source
        ),
        generic_setpoint_fallback_thrust=(
            float(config.px4_gazebo_generic_fallback_thrust)
            if config.px4_gazebo_generic_fallback_thrust is not None
            else None
        ),
        generic_setpoint_fallback_update_count=(
            aggregate.generic_setpoint_fallback_update_count
        ),
        generic_setpoint_fallback_rejection_count=(
            aggregate.generic_setpoint_fallback_rejection_count
        ),
        generic_setpoint_fallback_last_rejection=(
            aggregate.generic_setpoint_fallback_last_rejection
        ),
        phase9f_command_send_rate_hz=_rate_or_none(
            aggregate.command_sent_count,
            elapsed_s,
        ),
        phase9f_vision_frame_rate_hz=_rate_or_none(
            aggregate.vision_frames_completed,
            elapsed_s,
        ),
        phase9f_max_send_gap_s=_phase9f_max_send_gap_s(command_sender_summary),
        arm_attempt_count=aggregate.arm_attempt_count,
        arm_sent_count=aggregate.arm_sent_count,
        arm_rejection_count=aggregate.arm_rejection_count,
        last_arm_result=aggregate.last_arm_result,
        offboard_attempt_count=aggregate.offboard_attempt_count,
        offboard_sent_count=aggregate.offboard_sent_count,
        offboard_rejection_count=aggregate.offboard_rejection_count,
        last_offboard_result=aggregate.last_offboard_result,
        armed_state_observed=armed_state_observed,
        offboard_state_observed=offboard_state_observed,
        command_send_attempt_count=aggregate.command_send_attempt_count,
        command_sent_count=aggregate.command_sent_count,
        command_send_rejection_count=aggregate.command_send_rejection_count,
        last_command_send_result=aggregate.last_command_send_result,
        px4_gazebo_continuous_setpoint_stream_requested=bool(
            config.px4_gazebo_continuous_setpoint_stream
        ),
        px4_gazebo_command_max_age_s=float(config.px4_gazebo_command_max_age_s),
        setpoint_stream_cache_update_count=aggregate.setpoint_stream_cache_update_count,
        setpoint_stream_reused_count=aggregate.setpoint_stream_reused_count,
        setpoint_stream_stale_rejection_count=(
            aggregate.setpoint_stream_stale_rejection_count
        ),
        last_cached_command_age_s=aggregate.last_cached_command_age_s,
        command_sender_summary=command_sender_summary,
        command_blocked_reasons=aggregate.command_blocked_reasons,
        runner_events=aggregate.runner_events,
        mavlink_transport_summary=mavlink_transport_summary,
        vision_transport_summary=vision_transport_summary,
        phase6e_receive_satisfied=phase6e_receive_satisfied,
        phase6e_perception_boundary_satisfied=phase6e_perception_boundary_satisfied,
        phase6e_satisfied=phase6e_receive_satisfied,
        phase6e_success_criteria=phase6e_criteria,
        phase9b_perception_dry_run_satisfied=phase9b_satisfied,
        phase9b_success_criteria=phase9b_criteria,
        phase9c_command_dry_run_satisfied=phase9c_satisfied,
        phase9c_success_criteria=phase9c_criteria,
        phase9d_surrogate_command_send_satisfied=phase9d_satisfied,
        phase9d_success_criteria=phase9d_criteria,
        phase9e_surrogate_arm_offboard_satisfied=phase9e_satisfied,
        phase9e_success_criteria=phase9e_criteria,
        phase9e1_continuous_setpoint_stream_satisfied=phase9e1_satisfied,
        phase9e1_success_criteria=phase9e1_criteria,
        phase9e2_surrogate_command_safety_satisfied=phase9e2_satisfied,
        phase9e2_success_criteria=phase9e2_criteria,
        phase9e3_body_rate_interface_satisfied=phase9e3_satisfied,
        phase9e3_success_criteria=phase9e3_criteria,
        phase9e4_attitude_hover_interface_satisfied=phase9e4_satisfied,
        phase9e4_success_criteria=phase9e4_criteria,
        phase9f_full_autonomy_loop_satisfied=phase9f_satisfied,
        phase9f_success_criteria=phase9f_criteria,
        phase9f2_fixed_rate_setpoint_stream_satisfied=phase9f2_satisfied,
        phase9f2_success_criteria=phase9f2_criteria,
        phase9f3b_generic_setpoint_streamer_satisfied=phase9f3b_satisfied,
        phase9f3b_success_criteria=phase9f3b_criteria,
        phase9f3c_fallback_setpoint_satisfied=phase9f3c_satisfied,
        phase9f3c_success_criteria=phase9f3c_criteria,
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
    px4_gazebo_command_send_requested: bool = False,
    px4_gazebo_command_send_acknowledged: bool = False,
    px4_gazebo_surrogate_thrust_clamp_requested: bool = False,
    px4_gazebo_surrogate_thrust_clamp_min: Optional[float] = None,
    px4_gazebo_surrogate_thrust_clamp_max: Optional[float] = None,
    px4_gazebo_continuous_setpoint_stream_requested: bool = False,
    px4_gazebo_arm_requested: bool = False,
    px4_gazebo_arm_acknowledged: bool = False,
    px4_gazebo_offboard_requested: bool = False,
    px4_gazebo_offboard_acknowledged: bool = False,
    px4_gazebo_body_rate_smoke_requested: bool = False,
    px4_gazebo_body_rate_smoke_acknowledged: bool = False,
    body_roll_rate: Optional[float] = None,
    body_pitch_rate: Optional[float] = None,
    body_yaw_rate: Optional[float] = None,
    body_rate_thrust: Optional[float] = None,
    px4_gazebo_attitude_hover_smoke_requested: bool = False,
    px4_gazebo_attitude_hover_smoke_acknowledged: bool = False,
    attitude_hover_roll_rad: Optional[float] = None,
    attitude_hover_pitch_rad: Optional[float] = None,
    attitude_hover_yaw_rad: Optional[float] = None,
    attitude_hover_thrust: Optional[float] = None,
    px4_gazebo_full_autonomy_loop_requested: bool = False,
    px4_gazebo_full_autonomy_loop_acknowledged: bool = False,
    px4_gazebo_debug_yaw_override_requested: bool = False,
    px4_gazebo_debug_yaw_override_acknowledged: bool = False,
    px4_gazebo_debug_yaw_override_rad: Optional[float] = None,
    px4_gazebo_fixed_rate_setpoint_stream_requested: bool = False,
    px4_gazebo_fixed_rate_setpoint_stream_acknowledged: bool = False,
    px4_gazebo_generic_setpoint_streamer_requested: bool = False,
    px4_gazebo_generic_setpoint_streamer_acknowledged: bool = False,
    px4_gazebo_generic_setpoint_fallback_requested: bool = False,
    px4_gazebo_generic_setpoint_fallback_acknowledged: bool = False,
    generic_setpoint_fallback_thrust: Optional[float] = None,
) -> CompetitionMainSummary:
    normalized = _mode_value(mode)
    return CompetitionMainSummary(
        phase=_phase_for_flags(
            mode=normalized,
            live_transports=live_transports_requested,
            real_perception=real_perception_requested,
            px4_gazebo_command_send=px4_gazebo_command_send_requested,
            px4_gazebo_surrogate_thrust_clamp=(
                px4_gazebo_surrogate_thrust_clamp_requested
            ),
            px4_gazebo_continuous_setpoint_stream=(
                px4_gazebo_continuous_setpoint_stream_requested
            ),
            px4_gazebo_arm=px4_gazebo_arm_requested,
            px4_gazebo_offboard=px4_gazebo_offboard_requested,
            px4_gazebo_body_rate_smoke=px4_gazebo_body_rate_smoke_requested,
            px4_gazebo_attitude_hover_smoke=(
                px4_gazebo_attitude_hover_smoke_requested
            ),
            px4_gazebo_full_autonomy_loop=(
                px4_gazebo_full_autonomy_loop_requested
            ),
            px4_gazebo_debug_yaw_override=(
                px4_gazebo_debug_yaw_override_requested
            ),
            px4_gazebo_fixed_rate_setpoint_stream=(
                px4_gazebo_fixed_rate_setpoint_stream_requested
            ),
            px4_gazebo_generic_setpoint_streamer=(
                px4_gazebo_generic_setpoint_streamer_requested
            ),
            px4_gazebo_generic_setpoint_fallback=(
                px4_gazebo_generic_setpoint_fallback_requested
            ),
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
        px4_gazebo_command_send_requested=bool(px4_gazebo_command_send_requested),
        px4_gazebo_command_send_acknowledged=bool(
            px4_gazebo_command_send_acknowledged
        ),
        px4_gazebo_surrogate_thrust_clamp_requested=bool(
            px4_gazebo_surrogate_thrust_clamp_requested
        ),
        px4_gazebo_surrogate_thrust_clamp_min=(
            px4_gazebo_surrogate_thrust_clamp_min
        ),
        px4_gazebo_surrogate_thrust_clamp_max=(
            px4_gazebo_surrogate_thrust_clamp_max
        ),
        px4_gazebo_continuous_setpoint_stream_requested=bool(
            px4_gazebo_continuous_setpoint_stream_requested
        ),
        px4_gazebo_arm_requested=bool(px4_gazebo_arm_requested),
        px4_gazebo_arm_acknowledged=bool(px4_gazebo_arm_acknowledged),
        px4_gazebo_offboard_requested=bool(px4_gazebo_offboard_requested),
        px4_gazebo_offboard_acknowledged=bool(px4_gazebo_offboard_acknowledged),
        px4_gazebo_body_rate_smoke_requested=bool(
            px4_gazebo_body_rate_smoke_requested
        ),
        px4_gazebo_body_rate_smoke_acknowledged=bool(
            px4_gazebo_body_rate_smoke_acknowledged
        ),
        body_rate_type_mask=(
            BODY_RATE_ATTITUDE_IGNORE_TYPE_MASK
            if px4_gazebo_body_rate_smoke_requested
            else None
        ),
        body_rate_q=(
            BODY_RATE_DUMMY_QUATERNION
            if px4_gazebo_body_rate_smoke_requested
            else ()
        ),
        body_roll_rate=body_roll_rate,
        body_pitch_rate=body_pitch_rate,
        body_yaw_rate=body_yaw_rate,
        body_rate_thrust=body_rate_thrust,
        px4_gazebo_attitude_hover_smoke_requested=bool(
            px4_gazebo_attitude_hover_smoke_requested
        ),
        px4_gazebo_attitude_hover_smoke_acknowledged=bool(
            px4_gazebo_attitude_hover_smoke_acknowledged
        ),
        attitude_hover_type_mask=(
            ATTITUDE_HOVER_BODY_RATES_IGNORE_TYPE_MASK
            if px4_gazebo_attitude_hover_smoke_requested
            else None
        ),
        attitude_hover_body_rates=(
            ATTITUDE_HOVER_ZERO_BODY_RATES
            if px4_gazebo_attitude_hover_smoke_requested
            else ()
        ),
        attitude_hover_roll_rad=attitude_hover_roll_rad,
        attitude_hover_pitch_rad=attitude_hover_pitch_rad,
        attitude_hover_yaw_rad=attitude_hover_yaw_rad,
        attitude_hover_yaw_source=(
            "explicit" if attitude_hover_yaw_rad is not None else "current_state"
        ),
        attitude_hover_thrust=attitude_hover_thrust,
        px4_gazebo_full_autonomy_loop_requested=bool(
            px4_gazebo_full_autonomy_loop_requested
        ),
        px4_gazebo_full_autonomy_loop_acknowledged=bool(
            px4_gazebo_full_autonomy_loop_acknowledged
        ),
        phase9f_command_backend=(
            "attitude_angle_quaternion_set_attitude_target"
            if px4_gazebo_full_autonomy_loop_requested
            else ""
        ),
        phase9f_command_type_mask=(
            ATTITUDE_HOVER_BODY_RATES_IGNORE_TYPE_MASK
            if px4_gazebo_full_autonomy_loop_requested
            else None
        ),
        px4_gazebo_debug_yaw_override_requested=bool(
            px4_gazebo_debug_yaw_override_requested
        ),
        px4_gazebo_debug_yaw_override_acknowledged=bool(
            px4_gazebo_debug_yaw_override_acknowledged
        ),
        px4_gazebo_debug_yaw_override_rad=px4_gazebo_debug_yaw_override_rad,
        px4_gazebo_fixed_rate_setpoint_stream_requested=bool(
            px4_gazebo_fixed_rate_setpoint_stream_requested
        ),
        px4_gazebo_fixed_rate_setpoint_stream_acknowledged=bool(
            px4_gazebo_fixed_rate_setpoint_stream_acknowledged
        ),
        px4_gazebo_generic_setpoint_streamer_requested=bool(
            px4_gazebo_generic_setpoint_streamer_requested
        ),
        px4_gazebo_generic_setpoint_streamer_acknowledged=bool(
            px4_gazebo_generic_setpoint_streamer_acknowledged
        ),
        px4_gazebo_generic_setpoint_fallback_requested=bool(
            px4_gazebo_generic_setpoint_fallback_requested
        ),
        px4_gazebo_generic_setpoint_fallback_acknowledged=bool(
            px4_gazebo_generic_setpoint_fallback_acknowledged
        ),
        generic_setpoint_fallback_thrust=generic_setpoint_fallback_thrust,
        command_sent_count=0,
        command_blocked_reasons=("phase6d_fail_closed",),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Competition main executable. Runs bounded dry-run modes around "
            "competition_setup.py. Phase 9D PX4/Gazebo command sending is "
            "available only behind explicit surrogate acknowledgement flags."
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
    parser.add_argument(
        "--px4-gazebo-command-send",
        action="store_true",
        help=(
            "Phase 9D only: send accepted competition command-adapter fields "
            "to PX4/Gazebo through the already-open MAVLink connection"
        ),
    )
    parser.add_argument(
        "--ack-px4-gazebo-surrogate-command-send",
        action="store_true",
        help=(
            "required acknowledgement that Phase 9D command send is "
            "PX4/Gazebo surrogate-only evidence, not competition readiness"
        ),
    )
    parser.add_argument("--px4-gazebo-command-max-count", type=int, default=1)
    parser.add_argument("--px4-gazebo-command-max-heartbeat-age-s", type=float, default=1.5)
    parser.add_argument("--px4-gazebo-command-min-period-s", type=float, default=0.01)
    parser.add_argument("--px4-gazebo-command-min-thrust", type=float, default=0.0)
    parser.add_argument("--px4-gazebo-command-max-thrust", type=float, default=1.0)
    parser.add_argument("--px4-gazebo-command-max-abs-roll-pitch-rad", type=float, default=0.7)
    parser.add_argument("--px4-gazebo-command-max-abs-body-rate-rad-s", type=float, default=2.0)
    parser.add_argument(
        "--px4-gazebo-surrogate-thrust-clamp",
        action="store_true",
        help=(
            "Phase 9E.2 only: clamp PX4/Gazebo surrogate thrust at the sender "
            "boundary without changing AutonomyAPI/controller output"
        ),
    )
    parser.add_argument(
        "--px4-gazebo-surrogate-thrust-clamp-min",
        type=float,
        default=None,
        help="optional Phase 9E.2 lower clamp bound for surrogate thrust",
    )
    parser.add_argument(
        "--px4-gazebo-surrogate-thrust-clamp-max",
        type=float,
        default=None,
        help="optional Phase 9E.2 upper clamp bound for surrogate thrust",
    )
    parser.add_argument(
        "--px4-gazebo-continuous-setpoint-stream",
        action="store_true",
        help=(
            "Phase 9E.1 only: continue streaming the latest accepted "
            "competition command candidate to PX4/Gazebo while it remains fresh"
        ),
    )
    parser.add_argument(
        "--px4-gazebo-command-max-age-s",
        type=float,
        default=0.5,
        help="maximum age for reusing the latest accepted command candidate",
    )
    parser.add_argument(
        "--px4-gazebo-arm",
        action="store_true",
        help="Phase 9E only: send PX4/Gazebo surrogate arm command",
    )
    parser.add_argument(
        "--ack-px4-gazebo-surrogate-arm",
        action="store_true",
        help="required acknowledgement for PX4/Gazebo surrogate arm command",
    )
    parser.add_argument(
        "--px4-gazebo-offboard",
        action="store_true",
        help="Phase 9E only: request PX4/Gazebo Offboard mode via MAV_CMD_DO_SET_MODE",
    )
    parser.add_argument(
        "--ack-px4-gazebo-surrogate-offboard",
        action="store_true",
        help="required acknowledgement for PX4/Gazebo surrogate offboard mode request",
    )
    parser.add_argument("--px4-gazebo-arm-max-attempts", type=int, default=1)
    parser.add_argument("--px4-gazebo-offboard-max-attempts", type=int, default=1)
    parser.add_argument(
        "--px4-gazebo-offboard-prestream-count",
        type=int,
        default=0,
        help=(
            "Phase 9E only: require this many sent attitude targets before "
            "attempting PX4 Offboard mode"
        ),
    )
    parser.add_argument(
        "--px4-gazebo-body-rate-smoke",
        action="store_true",
        help=(
            "Phase 9E.3 only: send fixed PX4/Gazebo surrogate body-rate "
            "SET_ATTITUDE_TARGET messages without real perception/planning"
        ),
    )
    parser.add_argument(
        "--ack-px4-gazebo-surrogate-body-rate-smoke",
        action="store_true",
        help="required acknowledgement for PX4/Gazebo body-rate smoke commands",
    )
    parser.add_argument("--px4-gazebo-body-roll-rate", type=float, default=0.0)
    parser.add_argument("--px4-gazebo-body-pitch-rate", type=float, default=0.0)
    parser.add_argument("--px4-gazebo-body-yaw-rate", type=float, default=0.0)
    parser.add_argument(
        "--px4-gazebo-body-rate-thrust",
        type=float,
        default=None,
        help="explicit normalized thrust for Phase 9E.3 body-rate smoke",
    )
    parser.add_argument(
        "--px4-gazebo-attitude-hover-smoke",
        action="store_true",
        help=(
            "Phase 9E.4 only: send fixed PX4/Gazebo surrogate attitude-angle "
            "hover SET_ATTITUDE_TARGET messages without real perception/planning"
        ),
    )
    parser.add_argument(
        "--ack-px4-gazebo-surrogate-attitude-hover-smoke",
        action="store_true",
        help="required acknowledgement for PX4/Gazebo attitude-hover smoke commands",
    )
    parser.add_argument("--px4-gazebo-attitude-hover-roll-rad", type=float, default=0.0)
    parser.add_argument("--px4-gazebo-attitude-hover-pitch-rad", type=float, default=0.0)
    parser.add_argument(
        "--px4-gazebo-attitude-hover-yaw-rad",
        type=float,
        default=None,
        help=(
            "optional explicit yaw for Phase 9E.4; defaults to latest estimated "
            "telemetry yaw to mirror legacy hover"
        ),
    )
    parser.add_argument(
        "--px4-gazebo-attitude-hover-thrust",
        type=float,
        default=None,
        help="explicit normalized thrust for Phase 9E.4 attitude-hover smoke",
    )
    parser.add_argument(
        "--px4-gazebo-full-autonomy-loop",
        action="store_true",
        help=(
            "Phase 9F only: label and gate a full PX4/Gazebo surrogate "
            "autonomy loop through real perception, planning, command dry-run "
            "adaptation, and PX4/Gazebo-only command send"
        ),
    )
    parser.add_argument(
        "--ack-px4-gazebo-surrogate-full-autonomy-loop",
        action="store_true",
        help=(
            "required acknowledgement that Phase 9F is PX4/Gazebo "
            "surrogate-only evidence, not competition readiness"
        ),
    )
    parser.add_argument(
        "--px4-gazebo-debug-yaw-override-rad",
        type=float,
        default=None,
        help=(
            "Phase 9F.1 only: override only the outgoing PX4/Gazebo "
            "surrogate MAVLink yaw at the sender boundary for debugging"
        ),
    )
    parser.add_argument(
        "--ack-px4-gazebo-surrogate-debug-yaw-override",
        action="store_true",
        help=(
            "required acknowledgement that Phase 9F.1 yaw override is "
            "PX4/Gazebo surrogate-only and does not change controller output"
        ),
    )
    parser.add_argument(
        "--px4-gazebo-fixed-rate-setpoint-stream",
        action="store_true",
        help=(
            "Phase 9F.2 only: send cached accepted competition command fields "
            "through the PX4/Gazebo surrogate MAVLink sender at a fixed stream "
            "cadence before Offboard request"
        ),
    )
    parser.add_argument(
        "--ack-px4-gazebo-surrogate-fixed-rate-setpoint-stream",
        action="store_true",
        help=(
            "required acknowledgement that Phase 9F.2 fixed-rate setpoint "
            "streaming is PX4/Gazebo surrogate-only evidence"
        ),
    )
    parser.add_argument(
        "--px4-gazebo-setpoint-stream-hz",
        type=float,
        default=PHASE9F2_DEFAULT_SETPOINT_STREAM_HZ,
        help="Phase 9F.2 PX4/Gazebo surrogate setpoint stream target rate",
    )
    parser.add_argument(
        "--px4-gazebo-setpoint-stream-burst-limit",
        type=int,
        default=2,
        help=(
            "maximum Phase 9F.2 cached setpoints to send after each runner step"
        ),
    )
    parser.add_argument(
        "--px4-gazebo-generic-setpoint-streamer",
        action="store_true",
        help=(
            "Phase 9F.3B only: route PX4/Gazebo surrogate fixed-rate sends "
            "through CompetitionSetpointStreamer policy"
        ),
    )
    parser.add_argument(
        "--ack-px4-gazebo-surrogate-generic-setpoint-streamer",
        action="store_true",
        help=(
            "required acknowledgement that Phase 9F.3B generic setpoint "
            "streamer integration is PX4/Gazebo surrogate-only evidence"
        ),
    )
    parser.add_argument(
        "--px4-gazebo-generic-setpoint-fallback",
        action="store_true",
        help=(
            "Phase 9F.3C only: enable explicit PX4/Gazebo surrogate fallback/"
            "hold setpoints in CompetitionSetpointStreamer when autonomy "
            "commands are missing or stale"
        ),
    )
    parser.add_argument(
        "--ack-px4-gazebo-surrogate-generic-setpoint-fallback",
        action="store_true",
        help=(
            "required acknowledgement that Phase 9F.3C fallback setpoints are "
            "PX4/Gazebo surrogate-only and not competition readiness"
        ),
    )
    parser.add_argument("--px4-gazebo-generic-fallback-roll-rad", type=float, default=0.0)
    parser.add_argument("--px4-gazebo-generic-fallback-pitch-rad", type=float, default=0.0)
    parser.add_argument(
        "--px4-gazebo-generic-fallback-yaw-rad",
        type=float,
        default=None,
        help=(
            "optional Phase 9F.3C fallback yaw. If omitted, current state yaw "
            "is used for the fallback/hold setpoint"
        ),
    )
    parser.add_argument(
        "--px4-gazebo-generic-fallback-thrust",
        type=float,
        default=None,
        help="required explicit normalized thrust for Phase 9F.3C fallback setpoints",
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
        px4_gazebo_command_send=bool(args.px4_gazebo_command_send),
        ack_px4_gazebo_surrogate_command_send=bool(
            args.ack_px4_gazebo_surrogate_command_send
        ),
        px4_gazebo_command_max_count=int(args.px4_gazebo_command_max_count),
        px4_gazebo_command_max_heartbeat_age_s=float(
            args.px4_gazebo_command_max_heartbeat_age_s
        ),
        px4_gazebo_command_min_period_s=float(args.px4_gazebo_command_min_period_s),
        px4_gazebo_command_min_thrust=float(args.px4_gazebo_command_min_thrust),
        px4_gazebo_command_max_thrust=float(args.px4_gazebo_command_max_thrust),
        px4_gazebo_command_max_abs_roll_pitch_rad=float(
            args.px4_gazebo_command_max_abs_roll_pitch_rad
        ),
        px4_gazebo_command_max_abs_body_rate_rad_s=float(
            args.px4_gazebo_command_max_abs_body_rate_rad_s
        ),
        px4_gazebo_surrogate_thrust_clamp=bool(
            args.px4_gazebo_surrogate_thrust_clamp
        ),
        px4_gazebo_surrogate_thrust_clamp_min=(
            args.px4_gazebo_surrogate_thrust_clamp_min
        ),
        px4_gazebo_surrogate_thrust_clamp_max=(
            args.px4_gazebo_surrogate_thrust_clamp_max
        ),
        px4_gazebo_continuous_setpoint_stream=bool(
            args.px4_gazebo_continuous_setpoint_stream
        ),
        px4_gazebo_command_max_age_s=float(args.px4_gazebo_command_max_age_s),
        px4_gazebo_arm=bool(args.px4_gazebo_arm),
        ack_px4_gazebo_surrogate_arm=bool(args.ack_px4_gazebo_surrogate_arm),
        px4_gazebo_offboard=bool(args.px4_gazebo_offboard),
        ack_px4_gazebo_surrogate_offboard=bool(
            args.ack_px4_gazebo_surrogate_offboard
        ),
        px4_gazebo_arm_max_attempts=int(args.px4_gazebo_arm_max_attempts),
        px4_gazebo_offboard_max_attempts=int(args.px4_gazebo_offboard_max_attempts),
        px4_gazebo_offboard_prestream_count=int(
            args.px4_gazebo_offboard_prestream_count
        ),
        px4_gazebo_body_rate_smoke=bool(args.px4_gazebo_body_rate_smoke),
        ack_px4_gazebo_surrogate_body_rate_smoke=bool(
            args.ack_px4_gazebo_surrogate_body_rate_smoke
        ),
        px4_gazebo_body_roll_rate=float(args.px4_gazebo_body_roll_rate),
        px4_gazebo_body_pitch_rate=float(args.px4_gazebo_body_pitch_rate),
        px4_gazebo_body_yaw_rate=float(args.px4_gazebo_body_yaw_rate),
        px4_gazebo_body_rate_thrust=args.px4_gazebo_body_rate_thrust,
        px4_gazebo_attitude_hover_smoke=bool(args.px4_gazebo_attitude_hover_smoke),
        ack_px4_gazebo_surrogate_attitude_hover_smoke=bool(
            args.ack_px4_gazebo_surrogate_attitude_hover_smoke
        ),
        px4_gazebo_attitude_hover_roll_rad=float(
            args.px4_gazebo_attitude_hover_roll_rad
        ),
        px4_gazebo_attitude_hover_pitch_rad=float(
            args.px4_gazebo_attitude_hover_pitch_rad
        ),
        px4_gazebo_attitude_hover_yaw_rad=args.px4_gazebo_attitude_hover_yaw_rad,
        px4_gazebo_attitude_hover_thrust=args.px4_gazebo_attitude_hover_thrust,
        px4_gazebo_full_autonomy_loop=bool(args.px4_gazebo_full_autonomy_loop),
        ack_px4_gazebo_surrogate_full_autonomy_loop=bool(
            args.ack_px4_gazebo_surrogate_full_autonomy_loop
        ),
        px4_gazebo_debug_yaw_override_rad=(
            args.px4_gazebo_debug_yaw_override_rad
        ),
        ack_px4_gazebo_surrogate_debug_yaw_override=bool(
            args.ack_px4_gazebo_surrogate_debug_yaw_override
        ),
        px4_gazebo_fixed_rate_setpoint_stream=bool(
            args.px4_gazebo_fixed_rate_setpoint_stream
        ),
        ack_px4_gazebo_surrogate_fixed_rate_setpoint_stream=bool(
            args.ack_px4_gazebo_surrogate_fixed_rate_setpoint_stream
        ),
        px4_gazebo_setpoint_stream_hz=float(args.px4_gazebo_setpoint_stream_hz),
        px4_gazebo_setpoint_stream_burst_limit=int(
            args.px4_gazebo_setpoint_stream_burst_limit
        ),
        px4_gazebo_generic_setpoint_streamer=bool(
            args.px4_gazebo_generic_setpoint_streamer
        ),
        ack_px4_gazebo_surrogate_generic_setpoint_streamer=bool(
            args.ack_px4_gazebo_surrogate_generic_setpoint_streamer
        ),
        px4_gazebo_generic_setpoint_fallback=bool(
            args.px4_gazebo_generic_setpoint_fallback
        ),
        ack_px4_gazebo_surrogate_generic_setpoint_fallback=bool(
            args.ack_px4_gazebo_surrogate_generic_setpoint_fallback
        ),
        px4_gazebo_generic_fallback_roll_rad=float(
            args.px4_gazebo_generic_fallback_roll_rad
        ),
        px4_gazebo_generic_fallback_pitch_rad=float(
            args.px4_gazebo_generic_fallback_pitch_rad
        ),
        px4_gazebo_generic_fallback_yaw_rad=(
            args.px4_gazebo_generic_fallback_yaw_rad
        ),
        px4_gazebo_generic_fallback_thrust=args.px4_gazebo_generic_fallback_thrust,
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
            px4_gazebo_command_send_requested=bool(args.px4_gazebo_command_send),
            px4_gazebo_command_send_acknowledged=bool(
                args.ack_px4_gazebo_surrogate_command_send
            ),
            px4_gazebo_surrogate_thrust_clamp_requested=bool(
                args.px4_gazebo_surrogate_thrust_clamp
            ),
            px4_gazebo_surrogate_thrust_clamp_min=(
                args.px4_gazebo_surrogate_thrust_clamp_min
            ),
            px4_gazebo_surrogate_thrust_clamp_max=(
                args.px4_gazebo_surrogate_thrust_clamp_max
            ),
            px4_gazebo_continuous_setpoint_stream_requested=bool(
                args.px4_gazebo_continuous_setpoint_stream
            ),
            px4_gazebo_arm_requested=bool(args.px4_gazebo_arm),
            px4_gazebo_arm_acknowledged=bool(args.ack_px4_gazebo_surrogate_arm),
            px4_gazebo_offboard_requested=bool(args.px4_gazebo_offboard),
            px4_gazebo_offboard_acknowledged=bool(
                args.ack_px4_gazebo_surrogate_offboard
            ),
            px4_gazebo_body_rate_smoke_requested=bool(
                args.px4_gazebo_body_rate_smoke
            ),
            px4_gazebo_body_rate_smoke_acknowledged=bool(
                args.ack_px4_gazebo_surrogate_body_rate_smoke
            ),
            body_roll_rate=float(args.px4_gazebo_body_roll_rate),
            body_pitch_rate=float(args.px4_gazebo_body_pitch_rate),
            body_yaw_rate=float(args.px4_gazebo_body_yaw_rate),
            body_rate_thrust=args.px4_gazebo_body_rate_thrust,
            px4_gazebo_attitude_hover_smoke_requested=bool(
                args.px4_gazebo_attitude_hover_smoke
            ),
            px4_gazebo_attitude_hover_smoke_acknowledged=bool(
                args.ack_px4_gazebo_surrogate_attitude_hover_smoke
            ),
            attitude_hover_roll_rad=float(args.px4_gazebo_attitude_hover_roll_rad),
            attitude_hover_pitch_rad=float(args.px4_gazebo_attitude_hover_pitch_rad),
            attitude_hover_yaw_rad=args.px4_gazebo_attitude_hover_yaw_rad,
            attitude_hover_thrust=args.px4_gazebo_attitude_hover_thrust,
            px4_gazebo_full_autonomy_loop_requested=bool(
                args.px4_gazebo_full_autonomy_loop
            ),
            px4_gazebo_full_autonomy_loop_acknowledged=bool(
                args.ack_px4_gazebo_surrogate_full_autonomy_loop
            ),
            px4_gazebo_debug_yaw_override_requested=(
                args.px4_gazebo_debug_yaw_override_rad is not None
            ),
            px4_gazebo_debug_yaw_override_acknowledged=bool(
                args.ack_px4_gazebo_surrogate_debug_yaw_override
            ),
            px4_gazebo_debug_yaw_override_rad=(
                args.px4_gazebo_debug_yaw_override_rad
            ),
            px4_gazebo_fixed_rate_setpoint_stream_requested=bool(
                args.px4_gazebo_fixed_rate_setpoint_stream
            ),
            px4_gazebo_fixed_rate_setpoint_stream_acknowledged=bool(
                args.ack_px4_gazebo_surrogate_fixed_rate_setpoint_stream
            ),
            px4_gazebo_generic_setpoint_streamer_requested=bool(
                args.px4_gazebo_generic_setpoint_streamer
            ),
            px4_gazebo_generic_setpoint_streamer_acknowledged=bool(
                args.ack_px4_gazebo_surrogate_generic_setpoint_streamer
            ),
            px4_gazebo_generic_setpoint_fallback_requested=bool(
                args.px4_gazebo_generic_setpoint_fallback
            ),
            px4_gazebo_generic_setpoint_fallback_acknowledged=bool(
                args.ack_px4_gazebo_surrogate_generic_setpoint_fallback
            ),
            generic_setpoint_fallback_thrust=args.px4_gazebo_generic_fallback_thrust,
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
    cached_command_fields: Any = None
    cached_command_wall_time_s: Optional[float] = None
    setpoint_stream_cache_update_count: int = 0
    setpoint_stream_reused_count: int = 0
    setpoint_stream_stale_rejection_count: int = 0
    last_cached_command_age_s: Optional[float] = None
    arm_attempt_count: int = 0
    arm_sent_count: int = 0
    arm_rejection_count: int = 0
    last_arm_result: Optional[dict[str, Any]] = None
    offboard_attempt_count: int = 0
    offboard_sent_count: int = 0
    offboard_rejection_count: int = 0
    last_offboard_result: Optional[dict[str, Any]] = None
    body_rate_command_sent_count: int = 0
    body_rate_command_rejection_count: int = 0
    last_body_rate_command_send_result: Optional[dict[str, Any]] = None
    attitude_hover_command_sent_count: int = 0
    attitude_hover_command_rejection_count: int = 0
    last_attitude_hover_command_send_result: Optional[dict[str, Any]] = None
    last_attitude_hover_yaw_rad: Optional[float] = None
    last_attitude_hover_yaw_source: str = ""
    command_send_attempt_count: int = 0
    command_sent_count: int = 0
    command_send_rejection_count: int = 0
    last_command_send_result: Optional[dict[str, Any]] = None
    debug_yaw_override_applied_count: int = 0
    fixed_rate_setpoint_stream_iteration_count: int = 0
    fixed_rate_setpoint_stream_attempt_count: int = 0
    fixed_rate_setpoint_stream_sent_count: int = 0
    fixed_rate_setpoint_stream_rejection_count: int = 0
    generic_setpoint_streamer_summary: Optional[dict[str, Any]] = None
    generic_setpoint_fallback_update_count: int = 0
    generic_setpoint_fallback_rejection_count: int = 0
    generic_setpoint_fallback_last_rejection: str = ""
    last_generic_setpoint_fallback_yaw_rad: Optional[float] = None
    last_generic_setpoint_fallback_yaw_source: str = ""
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
            if (
                result.command_result.accepted
                and result.command_result.fields is not None
                and _phase9d_current_command_rejection_reason(result) == ""
            ):
                self.cached_command_fields = result.command_result.fields
                self.cached_command_wall_time_s = float(result.now_s)
                self.last_cached_command_age_s = 0.0
                self.setpoint_stream_cache_update_count += 1
        self.command_blocked_reasons = tuple(result.command_blocked_reasons)
        self.runner_events = tuple([*self.runner_events, *result.events])

    def record_command_send(self, send_result: dict[str, Any]) -> None:
        if bool(send_result.get("attempted", False)):
            self.command_send_attempt_count += 1
        if bool(send_result.get("sent", False)):
            self.command_sent_count += 1
        else:
            self.command_send_rejection_count += 1
        self.last_command_send_result = dict(send_result)

    def record_fixed_rate_setpoint_stream(
        self,
        send_result: dict[str, Any],
    ) -> None:
        if bool(send_result.get("attempted", False)):
            self.fixed_rate_setpoint_stream_attempt_count += 1
        if bool(send_result.get("sent", False)):
            self.fixed_rate_setpoint_stream_sent_count += 1
        else:
            self.fixed_rate_setpoint_stream_rejection_count += 1

    def record_body_rate_command_send(self, send_result: dict[str, Any]) -> None:
        if bool(send_result.get("sent", False)):
            self.body_rate_command_sent_count += 1
        else:
            self.body_rate_command_rejection_count += 1
        self.last_body_rate_command_send_result = dict(send_result)

    def record_attitude_hover_command_send(self, send_result: dict[str, Any]) -> None:
        if bool(send_result.get("sent", False)):
            self.attitude_hover_command_sent_count += 1
        else:
            self.attitude_hover_command_rejection_count += 1
        self.last_attitude_hover_command_send_result = dict(send_result)
        yaw = send_result.get("yaw_rad")
        if yaw is not None:
            self.last_attitude_hover_yaw_rad = float(yaw)
        yaw_source = send_result.get("yaw_source")
        if yaw_source:
            self.last_attitude_hover_yaw_source = str(yaw_source)

    def record_lifecycle_send(
        self,
        lifecycle_name: str,
        lifecycle_result: dict[str, Any],
    ) -> None:
        if lifecycle_name == "arm":
            if bool(lifecycle_result.get("attempted", False)):
                self.arm_attempt_count += 1
            if bool(lifecycle_result.get("sent", False)):
                self.arm_sent_count += 1
            else:
                self.arm_rejection_count += 1
            self.last_arm_result = dict(lifecycle_result)
            return

        if lifecycle_name == "offboard":
            if bool(lifecycle_result.get("attempted", False)):
                self.offboard_attempt_count += 1
            if bool(lifecycle_result.get("sent", False)):
                self.offboard_sent_count += 1
            else:
                self.offboard_rejection_count += 1
            self.last_offboard_result = dict(lifecycle_result)


def _command_sender_for_config(
    config: CompetitionMainConfig,
    components: CompetitionRuntimeComponents,
    *,
    command_sender: Optional[Any],
    clock: Callable[[], float],
) -> Optional[Any]:
    if command_sender is not None:
        return command_sender
    if not (
        config.px4_gazebo_command_send
        or config.px4_gazebo_arm
        or config.px4_gazebo_offboard
        or config.px4_gazebo_body_rate_smoke
        or config.px4_gazebo_attitude_hover_smoke
    ):
        return None

    sender_config = Px4GazeboCommandSenderConfig(
        min_command_period_s=float(config.px4_gazebo_command_min_period_s),
        min_thrust=float(config.px4_gazebo_command_min_thrust),
        max_thrust=float(config.px4_gazebo_command_max_thrust),
        max_abs_roll_pitch_rad=float(config.px4_gazebo_command_max_abs_roll_pitch_rad),
        max_abs_body_rate_rad_s=float(
            config.px4_gazebo_command_max_abs_body_rate_rad_s
        ),
        enable_surrogate_thrust_clamp=bool(
            config.px4_gazebo_surrogate_thrust_clamp
        ),
        surrogate_thrust_clamp_min=config.px4_gazebo_surrogate_thrust_clamp_min,
        surrogate_thrust_clamp_max=config.px4_gazebo_surrogate_thrust_clamp_max,
    )
    return Px4GazeboSetAttitudeTargetSender(
        config=sender_config,
        connection_provider=lambda: getattr(
            components.mavlink_transport,
            "active_connection",
            None,
        ),
        clock=clock,
    )


def _setpoint_streamer_for_config(
    config: CompetitionMainConfig,
) -> Optional[CompetitionSetpointStreamer]:
    if not config.px4_gazebo_generic_setpoint_streamer:
        return None
    return CompetitionSetpointStreamer(
        config=CompetitionSetpointStreamConfig(
            stream_rate_hz=float(config.px4_gazebo_setpoint_stream_hz),
            autonomy_command_fresh_s=float(config.px4_gazebo_command_max_age_s),
            require_fallback=bool(config.px4_gazebo_generic_setpoint_fallback),
        )
    )


def _maybe_update_phase9f3b_setpoint_streamer(
    *,
    config: CompetitionMainConfig,
    result: Any,
    setpoint_streamer: Optional[CompetitionSetpointStreamer],
) -> None:
    if setpoint_streamer is None:
        return
    if not config.px4_gazebo_generic_setpoint_streamer:
        return
    if (
        result.command_result is None
        or not result.command_result.accepted
        or result.command_result.fields is None
    ):
        return
    if _phase9d_current_command_rejection_reason(result):
        return
    setpoint_streamer.update_autonomy_fields(
        result.command_result.fields,
        now_s=float(result.now_s),
    )


def _maybe_update_phase9f3c_fallback_setpoint(
    *,
    config: CompetitionMainConfig,
    result: Any,
    aggregate: _Aggregate,
    setpoint_streamer: Optional[CompetitionSetpointStreamer],
) -> None:
    if not config.px4_gazebo_generic_setpoint_fallback:
        return
    if setpoint_streamer is None:
        aggregate.generic_setpoint_fallback_rejection_count += 1
        aggregate.generic_setpoint_fallback_last_rejection = (
            "generic_setpoint_streamer_unavailable"
        )
        return

    yaw_rad, yaw_source, yaw_rejection = _phase9f3c_fallback_yaw(config, result)
    if yaw_rejection:
        aggregate.generic_setpoint_fallback_rejection_count += 1
        aggregate.generic_setpoint_fallback_last_rejection = yaw_rejection
        return
    if config.px4_gazebo_generic_fallback_thrust is None:
        aggregate.generic_setpoint_fallback_rejection_count += 1
        aggregate.generic_setpoint_fallback_last_rejection = (
            "generic_fallback_thrust_missing"
        )
        return

    try:
        fields = build_dry_run_set_attitude_target_fields(
            (
                float(config.px4_gazebo_generic_fallback_roll_rad),
                float(config.px4_gazebo_generic_fallback_pitch_rad),
                float(yaw_rad),
                float(config.px4_gazebo_generic_fallback_thrust),
            ),
            time_boot_ms=_time_boot_ms_from_wall_time(result.now_s),
            target_system=int(config.target_system),
            target_component=int(config.target_component),
        )
        setpoint_streamer.update_fallback_fields(fields)
    except Exception as exc:
        aggregate.generic_setpoint_fallback_rejection_count += 1
        aggregate.generic_setpoint_fallback_last_rejection = str(exc)
        return

    aggregate.generic_setpoint_fallback_update_count += 1
    aggregate.last_generic_setpoint_fallback_yaw_rad = float(yaw_rad)
    aggregate.last_generic_setpoint_fallback_yaw_source = str(yaw_source)


def _phase9f3c_fallback_yaw(
    config: CompetitionMainConfig,
    result: Any,
) -> tuple[Optional[float], str, str]:
    if config.px4_gazebo_generic_fallback_yaw_rad is not None:
        yaw = float(config.px4_gazebo_generic_fallback_yaw_rad)
        if not math.isfinite(yaw):
            return None, "explicit", "generic_fallback_yaw_must_be_finite"
        return yaw, "explicit", ""

    state = getattr(result.state_result, "vehicle_state", None)
    yaw = getattr(state, "yaw", None)
    if yaw is None:
        return None, "current_state", "generic_fallback_current_yaw_unavailable"
    yaw_float = float(yaw)
    if not math.isfinite(yaw_float):
        return None, "current_state", "generic_fallback_current_yaw_must_be_finite"
    return yaw_float, "current_state", ""


def _maybe_send_phase9e_lifecycle(
    *,
    config: CompetitionMainConfig,
    result: Any,
    aggregate: _Aggregate,
    command_sender: Optional[Any],
) -> tuple[tuple[str, dict[str, Any]], ...]:
    if not (config.px4_gazebo_arm or config.px4_gazebo_offboard):
        return ()
    if command_sender is None:
        return (("arm", _phase9e_rejected_lifecycle_result("command_sender_unavailable")),)

    lifecycle_results: list[tuple[str, dict[str, Any]]] = []
    rejection = _phase9e_lifecycle_pre_send_rejection_reason(config, result)
    if rejection:
        if config.px4_gazebo_arm and aggregate.arm_attempt_count < int(
            config.px4_gazebo_arm_max_attempts
        ):
            lifecycle_results.append(
                ("arm", _phase9e_rejected_lifecycle_result(rejection))
            )
        if config.px4_gazebo_offboard and aggregate.offboard_attempt_count < int(
            config.px4_gazebo_offboard_max_attempts
        ):
            lifecycle_results.append(
                ("offboard", _phase9e_rejected_lifecycle_result(rejection))
            )
        return tuple(lifecycle_results)

    if config.px4_gazebo_arm and aggregate.arm_attempt_count < int(
        config.px4_gazebo_arm_max_attempts
    ):
        send_arm = getattr(command_sender, "send_arm_command", None)
        if not callable(send_arm):
            lifecycle_results.append(
                ("arm", _phase9e_rejected_lifecycle_result("sender_missing_arm_method"))
            )
        else:
            arm_result = send_arm(
                target_system=config.target_system,
                target_component=config.target_component,
                arm=True,
                now_s=result.now_s,
            )
            lifecycle_results.append(("arm", _result_to_dict(arm_result)))

    attitude_setpoints_sent = (
        aggregate.command_sent_count + aggregate.body_rate_command_sent_count
        + aggregate.attitude_hover_command_sent_count
    )
    offboard_prestream_ready = attitude_setpoints_sent >= int(
        config.px4_gazebo_offboard_prestream_count
    )
    if (
        config.px4_gazebo_offboard
        and offboard_prestream_ready
        and aggregate.offboard_attempt_count < int(config.px4_gazebo_offboard_max_attempts)
    ):
        send_offboard = getattr(command_sender, "send_offboard_mode_command", None)
        if not callable(send_offboard):
            lifecycle_results.append(
                (
                    "offboard",
                    _phase9e_rejected_lifecycle_result("sender_missing_offboard_method"),
                )
            )
        else:
            offboard_result = send_offboard(
                target_system=config.target_system,
                target_component=config.target_component,
                now_s=result.now_s,
            )
            lifecycle_results.append(("offboard", _result_to_dict(offboard_result)))

    return tuple(lifecycle_results)


def _phase9e_lifecycle_pre_send_rejection_reason(
    config: CompetitionMainConfig,
    result: Any,
) -> str:
    if not result.heartbeat_seen:
        return "heartbeat_missing"
    if result.heartbeat_age_s is None:
        return "heartbeat_age_missing"
    if float(result.heartbeat_age_s) > float(config.px4_gazebo_command_max_heartbeat_age_s):
        return "heartbeat_stale"
    if not result.state_result.is_usable:
        return "state_unusable"
    return ""


def _phase9e_rejected_lifecycle_result(reason: str) -> dict[str, Any]:
    return {
        "attempted": False,
        "sent": False,
        "command_name": "",
        "command_id": 0,
        "target_system": None,
        "target_component": None,
        "params": [],
        "rejection_reason": str(reason),
        "sent_at_s": None,
        "surrogate_label": PX4_GAZEBO_ARM_OFFBOARD_LABEL,
        "phase4b_satisfied": False,
        "competition_readiness_claimed": False,
    }


def _maybe_send_phase9e3_body_rate_smoke(
    *,
    config: CompetitionMainConfig,
    result: Any,
    aggregate: _Aggregate,
    command_sender: Optional[Any],
) -> Optional[dict[str, Any]]:
    if not config.px4_gazebo_body_rate_smoke:
        return None
    if command_sender is None:
        return _phase9e3_rejected_body_rate_result("command_sender_unavailable")
    if aggregate.body_rate_command_sent_count >= int(config.px4_gazebo_command_max_count):
        return None

    base_rejection = _phase9d_base_pre_send_rejection_reason(config, result)
    if base_rejection:
        return _phase9e3_rejected_body_rate_result(base_rejection)

    send_method = getattr(command_sender, "send_body_rate_set_attitude_target", None)
    if not callable(send_method):
        return _phase9e3_rejected_body_rate_result(
            "command_sender_missing_body_rate_send_method"
        )
    send_result = send_method(
        target_system=config.target_system,
        target_component=config.target_component,
        body_roll_rate=config.px4_gazebo_body_roll_rate,
        body_pitch_rate=config.px4_gazebo_body_pitch_rate,
        body_yaw_rate=config.px4_gazebo_body_yaw_rate,
        thrust=config.px4_gazebo_body_rate_thrust,
        time_boot_ms=_time_boot_ms_from_wall_time(result.now_s),
        sequence=aggregate.body_rate_command_sent_count + 1,
        now_s=result.now_s,
        stream_source="body_rate_smoke",
    )
    return _result_to_dict(send_result)


def _phase9e3_rejected_body_rate_result(reason: str) -> dict[str, Any]:
    return {
        "attempted": True,
        "sent": False,
        "rejection_reason": str(reason),
        "sent_at_s": None,
        "message_name": "SET_ATTITUDE_TARGET",
        "sequence": None,
        "target_system": None,
        "target_component": None,
        "type_mask": BODY_RATE_ATTITUDE_IGNORE_TYPE_MASK,
        "q": list(BODY_RATE_DUMMY_QUATERNION),
        "body_roll_rate": None,
        "body_pitch_rate": None,
        "body_yaw_rate": None,
        "thrust": None,
        "surrogate_label": PX4_GAZEBO_BODY_RATE_SMOKE_LABEL,
        "phase4b_satisfied": False,
        "competition_readiness_claimed": False,
    }


def _maybe_send_phase9e4_attitude_hover_smoke(
    *,
    config: CompetitionMainConfig,
    result: Any,
    aggregate: _Aggregate,
    command_sender: Optional[Any],
) -> Optional[dict[str, Any]]:
    if not config.px4_gazebo_attitude_hover_smoke:
        return None
    if command_sender is None:
        return _phase9e4_rejected_attitude_hover_result(
            "command_sender_unavailable",
            yaw_source=_attitude_hover_yaw_source(config),
        )
    if aggregate.attitude_hover_command_sent_count >= int(
        config.px4_gazebo_command_max_count
    ):
        return None

    base_rejection = _phase9d_base_pre_send_rejection_reason(config, result)
    if base_rejection:
        return _phase9e4_rejected_attitude_hover_result(
            base_rejection,
            yaw_source=_attitude_hover_yaw_source(config),
        )

    yaw_rad, yaw_source, yaw_rejection = _phase9e4_attitude_hover_yaw(config, result)
    if yaw_rejection:
        return _phase9e4_rejected_attitude_hover_result(
            yaw_rejection,
            yaw_source=yaw_source,
        )

    send_method = getattr(command_sender, "send_attitude_hover_set_attitude_target", None)
    if not callable(send_method):
        return _phase9e4_rejected_attitude_hover_result(
            "command_sender_missing_attitude_hover_send_method",
            yaw_source=yaw_source,
        )
    send_result = send_method(
        target_system=config.target_system,
        target_component=config.target_component,
        roll_rad=config.px4_gazebo_attitude_hover_roll_rad,
        pitch_rad=config.px4_gazebo_attitude_hover_pitch_rad,
        yaw_rad=yaw_rad,
        thrust=config.px4_gazebo_attitude_hover_thrust,
        time_boot_ms=_time_boot_ms_from_wall_time(result.now_s),
        sequence=aggregate.attitude_hover_command_sent_count + 1,
        now_s=result.now_s,
        stream_source="attitude_hover_smoke",
    )
    result_dict = _result_to_dict(send_result)
    result_dict["yaw_source"] = yaw_source
    return result_dict


def _phase9e4_attitude_hover_yaw(
    config: CompetitionMainConfig,
    result: Any,
) -> tuple[Optional[float], str, str]:
    if config.px4_gazebo_attitude_hover_yaw_rad is not None:
        yaw = float(config.px4_gazebo_attitude_hover_yaw_rad)
        if not math.isfinite(yaw):
            return None, "explicit", "attitude_hover_yaw_must_be_finite"
        return yaw, "explicit", ""

    state = getattr(result.state_result, "vehicle_state", None)
    yaw = getattr(state, "yaw", None)
    if yaw is None:
        return None, "current_state", "attitude_hover_current_yaw_unavailable"
    yaw_float = float(yaw)
    if not math.isfinite(yaw_float):
        return None, "current_state", "attitude_hover_current_yaw_must_be_finite"
    return yaw_float, "current_state", ""


def _attitude_hover_yaw_source(config: CompetitionMainConfig) -> str:
    if config.px4_gazebo_attitude_hover_yaw_rad is not None:
        return "explicit"
    return "current_state"


def _phase9e4_rejected_attitude_hover_result(
    reason: str,
    *,
    yaw_source: str,
) -> dict[str, Any]:
    return {
        "attempted": True,
        "sent": False,
        "rejection_reason": str(reason),
        "sent_at_s": None,
        "message_name": "SET_ATTITUDE_TARGET",
        "sequence": None,
        "target_system": None,
        "target_component": None,
        "type_mask": ATTITUDE_HOVER_BODY_RATES_IGNORE_TYPE_MASK,
        "q": [],
        "body_roll_rate": 0.0,
        "body_pitch_rate": 0.0,
        "body_yaw_rate": 0.0,
        "roll_rad": None,
        "pitch_rad": None,
        "yaw_rad": None,
        "yaw_source": str(yaw_source),
        "thrust": None,
        "surrogate_label": PX4_GAZEBO_ATTITUDE_HOVER_SMOKE_LABEL,
        "phase4b_satisfied": False,
        "competition_readiness_claimed": False,
    }


def _maybe_send_phase9d_command(
    *,
    config: CompetitionMainConfig,
    result: Any,
    aggregate: _Aggregate,
    command_sender: Optional[Any],
) -> Optional[dict[str, Any]]:
    if not config.px4_gazebo_command_send:
        return None
    if command_sender is None:
        return _phase9d_rejected_send_result("command_sender_unavailable")
    if aggregate.command_sent_count >= int(config.px4_gazebo_command_max_count):
        return None

    base_rejection = _phase9d_base_pre_send_rejection_reason(config, result)
    if base_rejection:
        return _phase9d_rejected_send_result(base_rejection)

    current_rejection = _phase9d_current_command_rejection_reason(result)
    stream_source = "current"
    if current_rejection:
        if not config.px4_gazebo_continuous_setpoint_stream:
            return _phase9d_rejected_send_result(current_rejection)
        fields_or_rejection = _phase9e1_cached_command_fields(
            config=config,
            result=result,
            aggregate=aggregate,
        )
        if isinstance(fields_or_rejection, str):
            if fields_or_rejection == "cached_command_stale":
                aggregate.setpoint_stream_stale_rejection_count += 1
            return _phase9d_rejected_send_result(fields_or_rejection)
        fields = fields_or_rejection
        stream_source = "cached"
        aggregate.setpoint_stream_reused_count += 1
    else:
        fields = result.command_result.fields
    fields, override_metadata = _phase9f_apply_debug_yaw_override(
        config=config,
        fields=fields,
        aggregate=aggregate,
    )
    send_method = getattr(command_sender, "send_set_attitude_target", None)
    if not callable(send_method):
        return _phase9d_rejected_send_result("command_sender_missing_send_method")
    send_result = send_method(fields, now_s=result.now_s, stream_source=stream_source)
    to_dict = getattr(send_result, "to_dict", None)
    if callable(to_dict):
        result_dict = to_dict()
    else:
        result_dict = dict(send_result)
    result_dict.update(override_metadata)
    return result_dict


def _maybe_send_phase9f2_fixed_rate_setpoints(
    *,
    config: CompetitionMainConfig,
    result: Any,
    aggregate: _Aggregate,
    command_sender: Optional[Any],
    setpoint_streamer: Optional[CompetitionSetpointStreamer],
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> tuple[dict[str, Any], ...]:
    if not config.px4_gazebo_fixed_rate_setpoint_stream:
        return ()
    aggregate.fixed_rate_setpoint_stream_iteration_count += 1
    if not config.px4_gazebo_command_send:
        return ()
    if command_sender is None:
        return (_phase9d_rejected_send_result("command_sender_unavailable"),)
    if aggregate.command_sent_count >= int(config.px4_gazebo_command_max_count):
        return ()

    base_rejection = _phase9d_base_pre_send_rejection_reason(config, result)
    if base_rejection:
        return ()
    if aggregate.cached_command_fields is None and not (
        config.px4_gazebo_generic_setpoint_streamer
        and config.px4_gazebo_generic_setpoint_fallback
    ):
        return ()

    send_method = getattr(command_sender, "send_set_attitude_target", None)
    if not callable(send_method):
        return (_phase9d_rejected_send_result("command_sender_missing_send_method"),)

    results: list[dict[str, Any]] = []
    stream_period_s = 1.0 / float(config.px4_gazebo_setpoint_stream_hz)
    remaining = int(config.px4_gazebo_command_max_count) - aggregate.command_sent_count
    burst_count = max(
        0,
        min(int(config.px4_gazebo_setpoint_stream_burst_limit), remaining),
    )
    for burst_index in range(burst_count):
        if burst_index > 0:
            sleep(stream_period_s)
        send_time_s = _phase9f2_next_send_time_s(
            command_sender=command_sender,
            clock=clock,
            sleep=sleep,
            min_period_s=float(config.px4_gazebo_command_min_period_s),
        )
        fields_or_rejection, stream_source, streamer_metadata, send_time_s = (
            _phase9f_fixed_rate_fields_for_time(
                config=config,
                now_s=send_time_s,
                aggregate=aggregate,
                setpoint_streamer=setpoint_streamer,
                sleep=sleep,
            )
        )
        if isinstance(fields_or_rejection, str):
            if fields_or_rejection in {
                "cached_command_stale",
                "generic_setpoint_streamer:autonomy_command_stale",
            }:
                aggregate.setpoint_stream_stale_rejection_count += 1
            results.append(_phase9d_rejected_send_result(fields_or_rejection))
            break

        fields = fields_or_rejection
        aggregate.setpoint_stream_reused_count += 1
        fields, override_metadata = _phase9f_apply_debug_yaw_override(
            config=config,
            fields=fields,
            aggregate=aggregate,
        )
        send_result = send_method(
            fields,
            now_s=send_time_s,
            stream_source=stream_source,
        )
        result_dict = _result_to_dict(send_result)
        result_dict.update(override_metadata)
        result_dict.update(streamer_metadata)
        result_dict["fixed_rate_setpoint_stream"] = True
        result_dict["fixed_rate_setpoint_stream_hz"] = float(
            config.px4_gazebo_setpoint_stream_hz
        )
        result_dict["fixed_rate_setpoint_stream_label"] = (
            PX4_GAZEBO_FIXED_RATE_SETPOINT_STREAM_LABEL
        )
        results.append(result_dict)

        if not bool(result_dict.get("sent", False)):
            break
    return tuple(results)


def _phase9f_fixed_rate_fields_for_time(
    *,
    config: CompetitionMainConfig,
    now_s: float,
    aggregate: _Aggregate,
    setpoint_streamer: Optional[CompetitionSetpointStreamer],
    sleep: Callable[[float], None],
) -> tuple[Any, str, dict[str, Any], float]:
    if not config.px4_gazebo_generic_setpoint_streamer:
        fields_or_rejection = _cached_command_fields_for_time(
            config=config,
            now_s=now_s,
            aggregate=aggregate,
        )
        return fields_or_rejection, "fixed_rate_cached", {}, float(now_s)

    if setpoint_streamer is None:
        return (
            "generic_setpoint_streamer_unavailable",
            "generic_setpoint_streamer",
            {},
            float(now_s),
        )

    effective_now_s = float(now_s)
    decision = setpoint_streamer.step(now_s=now_s)
    if (
        not decision.should_emit
        and decision.reason == "stream_rate_wait"
        and decision.next_due_s is not None
    ):
        wait_s = max(0.0, float(decision.next_due_s) - float(now_s))
        if wait_s > 0.0:
            sleep(wait_s)
        effective_now_s = float(decision.next_due_s)
        decision = setpoint_streamer.step(now_s=effective_now_s)

    metadata = {
        "generic_setpoint_streamer": True,
        "generic_setpoint_streamer_label": PX4_GAZEBO_GENERIC_SETPOINT_STREAMER_LABEL,
        "generic_setpoint_streamer_source": decision.source,
        "generic_setpoint_streamer_reason": decision.reason,
        "generic_setpoint_streamer_sequence": decision.sequence,
        "generic_setpoint_streamer_command_age_s": decision.command_age_s,
        "generic_setpoint_streamer_phase4b_satisfied": decision.phase4b_satisfied,
        "generic_setpoint_streamer_competition_readiness_claimed": (
            decision.competition_readiness_claimed
        ),
    }
    if decision.source == "fallback":
        metadata["generic_setpoint_fallback"] = True
        metadata["generic_setpoint_fallback_label"] = (
            PX4_GAZEBO_GENERIC_SETPOINT_FALLBACK_LABEL
        )
    if decision.command_age_s is not None:
        aggregate.last_cached_command_age_s = float(decision.command_age_s)
    if not decision.should_emit or decision.fields is None:
        return (
            f"generic_setpoint_streamer:{decision.reason}",
            "generic_setpoint_streamer",
            metadata,
            effective_now_s,
        )
    return decision.fields, f"generic_{decision.source}", metadata, effective_now_s


def _phase9f2_next_send_time_s(
    *,
    command_sender: Any,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    min_period_s: float,
) -> float:
    now_s = float(clock())
    last_sent_s = _command_sender_last_send_wall_time(command_sender)
    if last_sent_s is None:
        return now_s

    remaining_s = float(min_period_s) - (now_s - float(last_sent_s))
    if remaining_s > 0.0:
        sleep(remaining_s)
        now_s = float(clock())
    return now_s


def _command_sender_last_send_wall_time(command_sender: Any) -> Optional[float]:
    stats = getattr(command_sender, "stats", None)
    last_sent_s = getattr(stats, "last_send_wall_time", None)
    if last_sent_s is None:
        return None
    return float(last_sent_s)


def _phase9f_apply_debug_yaw_override(
    *,
    config: CompetitionMainConfig,
    fields: Any,
    aggregate: _Aggregate,
) -> tuple[Any, dict[str, Any]]:
    if config.px4_gazebo_debug_yaw_override_rad is None:
        return fields, {}

    yaw_override = float(config.px4_gazebo_debug_yaw_override_rad)
    roll_rad, pitch_rad, original_yaw_rad = _euler_zyx_from_quaternion_wxyz(
        tuple(float(component) for component in fields.q)
    )
    override_q = quaternion_wxyz_from_euler_zyx(
        roll_rad=roll_rad,
        pitch_rad=pitch_rad,
        yaw_rad=yaw_override,
    )
    aggregate.debug_yaw_override_applied_count += 1
    return (
        replace(fields, q=override_q),
        {
            "debug_yaw_override_applied": True,
            "debug_yaw_override_label": PX4_GAZEBO_DEBUG_YAW_OVERRIDE_LABEL,
            "debug_yaw_original_rad": original_yaw_rad,
            "debug_yaw_override_rad": yaw_override,
        },
    )


def _phase9d_base_pre_send_rejection_reason(
    config: CompetitionMainConfig,
    result: Any,
) -> str:
    if not result.heartbeat_seen:
        return "heartbeat_missing"
    if result.heartbeat_age_s is None:
        return "heartbeat_age_missing"
    if float(result.heartbeat_age_s) > float(config.px4_gazebo_command_max_heartbeat_age_s):
        return "heartbeat_stale"
    if not result.state_result.is_usable:
        return "state_unusable"
    return ""


def _phase9d_current_command_rejection_reason(result: Any) -> str:
    if int(result.vision_frames_completed) <= 0:
        return "image_not_fresh"
    if int(result.perception_update_calls) <= 0:
        return "perception_not_fresh"
    if not getattr(result, "autonomy_telemetry_synced", False):
        return "autonomy_telemetry_not_synced"
    if result.planning_succeeded is not True:
        return "planning_not_succeeded"
    if not result.command_candidate_attempted:
        return "command_candidate_not_attempted"
    if result.command_result is None:
        return "command_result_missing"
    if not result.command_result.accepted:
        return f"command_result_rejected:{result.command_result.rejection_reason}"
    if result.command_result.fields is None:
        return "command_fields_missing"
    return ""


def _phase9e1_cached_command_fields(
    *,
    config: CompetitionMainConfig,
    result: Any,
    aggregate: _Aggregate,
) -> Any:
    return _cached_command_fields_for_time(
        config=config,
        now_s=float(result.now_s),
        aggregate=aggregate,
    )


def _cached_command_fields_for_time(
    *,
    config: CompetitionMainConfig,
    now_s: float,
    aggregate: _Aggregate,
) -> Any:
    if aggregate.cached_command_fields is None:
        return "cached_command_missing"
    if aggregate.cached_command_wall_time_s is None:
        return "cached_command_time_missing"

    age_s = float(now_s) - float(aggregate.cached_command_wall_time_s)
    aggregate.last_cached_command_age_s = age_s
    if age_s < 0.0:
        return "cached_command_time_in_future"
    if age_s > float(config.px4_gazebo_command_max_age_s):
        return "cached_command_stale"

    return replace(
        aggregate.cached_command_fields,
        time_boot_ms=_time_boot_ms_from_wall_time(now_s),
    )


def _time_boot_ms_from_wall_time(now_s: float) -> int:
    return int(max(0.0, float(now_s)) * 1000.0) & 0xFFFFFFFF


def _phase9d_rejected_send_result(reason: str) -> dict[str, Any]:
    return {
        "attempted": False,
        "sent": False,
        "rejection_reason": str(reason),
        "sent_at_s": None,
        "message_name": "SET_ATTITUDE_TARGET",
        "sequence": None,
        "target_system": None,
        "target_component": None,
        "type_mask": None,
        "q": [],
        "body_roll_rate": None,
        "body_pitch_rate": None,
        "body_yaw_rate": None,
        "thrust": None,
        "surrogate_label": PX4_GAZEBO_SURROGATE_LABEL,
        "phase4b_satisfied": False,
        "competition_readiness_claimed": False,
    }


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
    if config.px4_gazebo_full_autonomy_loop:
        if mode != CompetitionRunnerMode.COMMAND_DRY_RUN:
            raise CompetitionMainSafetyError(
                "Phase 9F full autonomy loop requires command_dry_run"
            )
        if not config.ack_px4_gazebo_surrogate_full_autonomy_loop:
            raise CompetitionMainSafetyError(
                "Phase 9F requires "
                "--ack-px4-gazebo-surrogate-full-autonomy-loop"
            )
        if not config.live_transports and components is None:
            raise CompetitionMainSafetyError(
                "Phase 9F requires --live-transports or injected test components"
            )
        if not config.use_real_autonomy or not config.real_perception:
            raise CompetitionMainSafetyError(
                "Phase 9F requires --use-real-autonomy and --real-perception"
            )
        if (
            str(config.perception_transform_mode)
            != COMPETITION_OFFICIAL_TRANSFORM_MODE
        ):
            raise CompetitionMainSafetyError(
                "Phase 9F requires "
                f"--perception-transform-mode {COMPETITION_OFFICIAL_TRANSFORM_MODE}"
            )
        if not config.px4_gazebo_command_send:
            raise CompetitionMainSafetyError(
                "Phase 9F requires --px4-gazebo-command-send"
            )
        if not config.ack_px4_gazebo_surrogate_command_send:
            raise CompetitionMainSafetyError(
                "Phase 9F requires --ack-px4-gazebo-surrogate-command-send"
            )
        if not config.px4_gazebo_continuous_setpoint_stream:
            raise CompetitionMainSafetyError(
                "Phase 9F requires --px4-gazebo-continuous-setpoint-stream"
            )
        if not config.px4_gazebo_arm or not config.ack_px4_gazebo_surrogate_arm:
            raise CompetitionMainSafetyError(
                "Phase 9F requires --px4-gazebo-arm and "
                "--ack-px4-gazebo-surrogate-arm"
            )
        if (
            not config.px4_gazebo_offboard
            or not config.ack_px4_gazebo_surrogate_offboard
        ):
            raise CompetitionMainSafetyError(
                "Phase 9F requires --px4-gazebo-offboard and "
                "--ack-px4-gazebo-surrogate-offboard"
            )
        if config.px4_gazebo_body_rate_smoke or config.px4_gazebo_attitude_hover_smoke:
            raise CompetitionMainSafetyError(
                "Phase 9F cannot be combined with fixed smoke command modes"
            )
    if config.px4_gazebo_debug_yaw_override_rad is not None:
        if not config.px4_gazebo_full_autonomy_loop:
            raise CompetitionMainSafetyError(
                "Phase 9F.1 debug yaw override requires "
                "--px4-gazebo-full-autonomy-loop"
            )
        if not config.ack_px4_gazebo_surrogate_debug_yaw_override:
            raise CompetitionMainSafetyError(
                "Phase 9F.1 debug yaw override requires "
                "--ack-px4-gazebo-surrogate-debug-yaw-override"
            )
        yaw_override = float(config.px4_gazebo_debug_yaw_override_rad)
        if not math.isfinite(yaw_override):
            raise CompetitionMainSafetyError(
                "px4_gazebo_debug_yaw_override_rad must be finite"
            )
    if (
        config.ack_px4_gazebo_surrogate_debug_yaw_override
        and config.px4_gazebo_debug_yaw_override_rad is None
    ):
        raise CompetitionMainSafetyError(
            "--ack-px4-gazebo-surrogate-debug-yaw-override requires "
            "--px4-gazebo-debug-yaw-override-rad"
        )
    if config.px4_gazebo_fixed_rate_setpoint_stream:
        if not config.px4_gazebo_full_autonomy_loop:
            raise CompetitionMainSafetyError(
                "Phase 9F.2 fixed-rate setpoint stream requires "
                "--px4-gazebo-full-autonomy-loop"
            )
        if not config.ack_px4_gazebo_surrogate_fixed_rate_setpoint_stream:
            raise CompetitionMainSafetyError(
                "Phase 9F.2 fixed-rate setpoint stream requires "
                "--ack-px4-gazebo-surrogate-fixed-rate-setpoint-stream"
            )
        stream_hz = float(config.px4_gazebo_setpoint_stream_hz)
        if not math.isfinite(stream_hz) or stream_hz <= 0.0:
            raise CompetitionMainSafetyError(
                "px4_gazebo_setpoint_stream_hz must be finite and positive"
            )
        if stream_hz > PHASE9F2_MAX_SETPOINT_STREAM_HZ:
            raise CompetitionMainSafetyError(
                "px4_gazebo_setpoint_stream_hz must remain below 100 Hz"
            )
        if (1.0 / stream_hz) < float(config.px4_gazebo_command_min_period_s):
            raise CompetitionMainSafetyError(
                "px4_gazebo_setpoint_stream_hz conflicts with command min period"
            )
        if int(config.px4_gazebo_setpoint_stream_burst_limit) <= 0:
            raise CompetitionMainSafetyError(
                "px4_gazebo_setpoint_stream_burst_limit must be positive"
            )
        if (
            config.px4_gazebo_offboard
            and int(config.px4_gazebo_offboard_prestream_count) <= 0
        ):
            raise CompetitionMainSafetyError(
                "Phase 9F.2 requires positive "
                "--px4-gazebo-offboard-prestream-count before Offboard request"
            )
    if (
        config.ack_px4_gazebo_surrogate_fixed_rate_setpoint_stream
        and not config.px4_gazebo_fixed_rate_setpoint_stream
    ):
        raise CompetitionMainSafetyError(
            "--ack-px4-gazebo-surrogate-fixed-rate-setpoint-stream requires "
            "--px4-gazebo-fixed-rate-setpoint-stream"
        )
    if config.px4_gazebo_generic_setpoint_streamer:
        if not config.px4_gazebo_fixed_rate_setpoint_stream:
            raise CompetitionMainSafetyError(
                "Phase 9F.3B generic setpoint streamer requires "
                "--px4-gazebo-fixed-rate-setpoint-stream"
            )
        if not config.ack_px4_gazebo_surrogate_generic_setpoint_streamer:
            raise CompetitionMainSafetyError(
                "Phase 9F.3B generic setpoint streamer requires "
                "--ack-px4-gazebo-surrogate-generic-setpoint-streamer"
            )
    if (
        config.ack_px4_gazebo_surrogate_generic_setpoint_streamer
        and not config.px4_gazebo_generic_setpoint_streamer
    ):
        raise CompetitionMainSafetyError(
            "--ack-px4-gazebo-surrogate-generic-setpoint-streamer requires "
            "--px4-gazebo-generic-setpoint-streamer"
        )
    if config.px4_gazebo_generic_setpoint_fallback:
        if not config.px4_gazebo_generic_setpoint_streamer:
            raise CompetitionMainSafetyError(
                "Phase 9F.3C fallback setpoint requires "
                "--px4-gazebo-generic-setpoint-streamer"
            )
        if not config.ack_px4_gazebo_surrogate_generic_setpoint_fallback:
            raise CompetitionMainSafetyError(
                "Phase 9F.3C fallback setpoint requires "
                "--ack-px4-gazebo-surrogate-generic-setpoint-fallback"
            )
        if config.px4_gazebo_generic_fallback_thrust is None:
            raise CompetitionMainSafetyError(
                "Phase 9F.3C fallback setpoint requires "
                "--px4-gazebo-generic-fallback-thrust"
            )
        fallback_values = (
            float(config.px4_gazebo_generic_fallback_roll_rad),
            float(config.px4_gazebo_generic_fallback_pitch_rad),
            float(config.px4_gazebo_generic_fallback_thrust),
        )
        if config.px4_gazebo_generic_fallback_yaw_rad is not None:
            fallback_values = (
                *fallback_values,
                float(config.px4_gazebo_generic_fallback_yaw_rad),
            )
        if not all(math.isfinite(value) for value in fallback_values):
            raise CompetitionMainSafetyError(
                "Phase 9F.3C fallback roll/pitch/yaw/thrust must be finite"
            )
        fallback_thrust = float(config.px4_gazebo_generic_fallback_thrust)
        if not (
            float(config.px4_gazebo_command_min_thrust)
            <= fallback_thrust
            <= float(config.px4_gazebo_command_max_thrust)
        ):
            raise CompetitionMainSafetyError(
                "Phase 9F.3C fallback thrust must be within command thrust bounds"
            )
        max_abs_rp = float(config.px4_gazebo_command_max_abs_roll_pitch_rad)
        if (
            abs(float(config.px4_gazebo_generic_fallback_roll_rad)) > max_abs_rp
            or abs(float(config.px4_gazebo_generic_fallback_pitch_rad)) > max_abs_rp
        ):
            raise CompetitionMainSafetyError(
                "Phase 9F.3C fallback roll/pitch exceed command attitude bounds"
            )
    if (
        config.ack_px4_gazebo_surrogate_generic_setpoint_fallback
        and not config.px4_gazebo_generic_setpoint_fallback
    ):
        raise CompetitionMainSafetyError(
            "--ack-px4-gazebo-surrogate-generic-setpoint-fallback requires "
            "--px4-gazebo-generic-setpoint-fallback"
        )
    if config.px4_gazebo_command_send:
        if config.px4_gazebo_body_rate_smoke or config.px4_gazebo_attitude_hover_smoke:
            raise CompetitionMainSafetyError(
                "Phase 9E fixed smoke modes cannot be combined with "
                "--px4-gazebo-command-send"
            )
        if mode != CompetitionRunnerMode.COMMAND_DRY_RUN:
            raise CompetitionMainSafetyError(
                "Phase 9D PX4/Gazebo command send requires command_dry_run"
            )
        if not config.ack_px4_gazebo_surrogate_command_send:
            raise CompetitionMainSafetyError(
                "Phase 9D requires --ack-px4-gazebo-surrogate-command-send"
            )
        if not config.live_transports and components is None:
            raise CompetitionMainSafetyError(
                "Phase 9D requires --live-transports or injected test components"
            )
        if not config.use_real_autonomy or not config.real_perception:
            raise CompetitionMainSafetyError(
                "Phase 9D requires --use-real-autonomy and --real-perception"
            )
        if int(config.px4_gazebo_command_max_count) <= 0:
            raise CompetitionMainSafetyError(
                "px4_gazebo_command_max_count must be positive"
            )
        if float(config.px4_gazebo_command_max_heartbeat_age_s) <= 0.0:
            raise CompetitionMainSafetyError(
                "px4_gazebo_command_max_heartbeat_age_s must be positive"
            )
        if float(config.px4_gazebo_command_min_period_s) < 0.01:
            raise CompetitionMainSafetyError(
                "px4_gazebo_command_min_period_s must preserve the <100 Hz limit"
            )
        if float(config.px4_gazebo_command_min_thrust) < 0.0:
            raise CompetitionMainSafetyError("px4_gazebo_command_min_thrust must be >= 0")
        if float(config.px4_gazebo_command_max_thrust) > 1.0:
            raise CompetitionMainSafetyError("px4_gazebo_command_max_thrust must be <= 1")
        if (
            float(config.px4_gazebo_command_min_thrust)
            > float(config.px4_gazebo_command_max_thrust)
        ):
            raise CompetitionMainSafetyError(
                "px4_gazebo_command_min_thrust must be <= max thrust"
            )
    if config.px4_gazebo_body_rate_smoke:
        if config.px4_gazebo_attitude_hover_smoke:
            raise CompetitionMainSafetyError(
                "Phase 9E.3 body-rate smoke cannot be combined with "
                "Phase 9E.4 attitude-hover smoke"
            )
        if mode != CompetitionRunnerMode.COMMAND_DRY_RUN:
            raise CompetitionMainSafetyError(
                "Phase 9E.3 body-rate smoke requires command_dry_run"
            )
        if not config.ack_px4_gazebo_surrogate_body_rate_smoke:
            raise CompetitionMainSafetyError(
                "Phase 9E.3 requires "
                "--ack-px4-gazebo-surrogate-body-rate-smoke"
            )
        if not config.live_transports and components is None:
            raise CompetitionMainSafetyError(
                "Phase 9E.3 requires --live-transports or injected test components"
            )
        if config.use_real_autonomy or config.real_perception:
            raise CompetitionMainSafetyError(
                "Phase 9E.3 must not use real AutonomyAPI or real perception"
            )
        if config.px4_gazebo_body_rate_thrust is None:
            raise CompetitionMainSafetyError(
                "Phase 9E.3 requires explicit --px4-gazebo-body-rate-thrust"
            )
        if int(config.px4_gazebo_command_max_count) <= 0:
            raise CompetitionMainSafetyError(
                "px4_gazebo_command_max_count must be positive"
            )
        if float(config.px4_gazebo_command_max_heartbeat_age_s) <= 0.0:
            raise CompetitionMainSafetyError(
                "px4_gazebo_command_max_heartbeat_age_s must be positive"
            )
        if float(config.px4_gazebo_command_min_period_s) < 0.01:
            raise CompetitionMainSafetyError(
                "px4_gazebo_command_min_period_s must preserve the <100 Hz limit"
            )
        body_rate_thrust = float(config.px4_gazebo_body_rate_thrust)
        if not math.isfinite(body_rate_thrust):
            raise CompetitionMainSafetyError(
                "px4_gazebo_body_rate_thrust must be finite"
            )
        if not (
            float(config.px4_gazebo_command_min_thrust)
            <= body_rate_thrust
            <= float(config.px4_gazebo_command_max_thrust)
        ):
            raise CompetitionMainSafetyError(
                "px4_gazebo_body_rate_thrust must remain inside command thrust bounds"
            )
        for name, value in (
            ("px4_gazebo_body_roll_rate", config.px4_gazebo_body_roll_rate),
            ("px4_gazebo_body_pitch_rate", config.px4_gazebo_body_pitch_rate),
            ("px4_gazebo_body_yaw_rate", config.px4_gazebo_body_yaw_rate),
        ):
            rate = float(value)
            if not math.isfinite(rate):
                raise CompetitionMainSafetyError(f"{name} must be finite")
            if abs(rate) > float(config.px4_gazebo_command_max_abs_body_rate_rad_s):
                raise CompetitionMainSafetyError(f"{name} exceeds body-rate safety limit")
    if config.px4_gazebo_attitude_hover_smoke:
        if mode != CompetitionRunnerMode.COMMAND_DRY_RUN:
            raise CompetitionMainSafetyError(
                "Phase 9E.4 attitude-hover smoke requires command_dry_run"
            )
        if not config.ack_px4_gazebo_surrogate_attitude_hover_smoke:
            raise CompetitionMainSafetyError(
                "Phase 9E.4 requires "
                "--ack-px4-gazebo-surrogate-attitude-hover-smoke"
            )
        if not config.live_transports and components is None:
            raise CompetitionMainSafetyError(
                "Phase 9E.4 requires --live-transports or injected test components"
            )
        if config.use_real_autonomy or config.real_perception:
            raise CompetitionMainSafetyError(
                "Phase 9E.4 must not use real AutonomyAPI or real perception"
            )
        if config.px4_gazebo_attitude_hover_thrust is None:
            raise CompetitionMainSafetyError(
                "Phase 9E.4 requires explicit --px4-gazebo-attitude-hover-thrust"
            )
        if int(config.px4_gazebo_command_max_count) <= 0:
            raise CompetitionMainSafetyError(
                "px4_gazebo_command_max_count must be positive"
            )
        if float(config.px4_gazebo_command_max_heartbeat_age_s) <= 0.0:
            raise CompetitionMainSafetyError(
                "px4_gazebo_command_max_heartbeat_age_s must be positive"
            )
        if float(config.px4_gazebo_command_min_period_s) < 0.01:
            raise CompetitionMainSafetyError(
                "px4_gazebo_command_min_period_s must preserve the <100 Hz limit"
            )
        attitude_hover_thrust = float(config.px4_gazebo_attitude_hover_thrust)
        if not math.isfinite(attitude_hover_thrust):
            raise CompetitionMainSafetyError(
                "px4_gazebo_attitude_hover_thrust must be finite"
            )
        if not (
            float(config.px4_gazebo_command_min_thrust)
            <= attitude_hover_thrust
            <= float(config.px4_gazebo_command_max_thrust)
        ):
            raise CompetitionMainSafetyError(
                "px4_gazebo_attitude_hover_thrust must remain inside command thrust bounds"
            )
        for name, value in (
            (
                "px4_gazebo_attitude_hover_roll_rad",
                config.px4_gazebo_attitude_hover_roll_rad,
            ),
            (
                "px4_gazebo_attitude_hover_pitch_rad",
                config.px4_gazebo_attitude_hover_pitch_rad,
            ),
        ):
            angle = float(value)
            if not math.isfinite(angle):
                raise CompetitionMainSafetyError(f"{name} must be finite")
            if abs(angle) > float(config.px4_gazebo_command_max_abs_roll_pitch_rad):
                raise CompetitionMainSafetyError(f"{name} exceeds attitude safety limit")
        if config.px4_gazebo_attitude_hover_yaw_rad is not None:
            yaw = float(config.px4_gazebo_attitude_hover_yaw_rad)
            if not math.isfinite(yaw):
                raise CompetitionMainSafetyError(
                    "px4_gazebo_attitude_hover_yaw_rad must be finite"
                )
    if config.px4_gazebo_surrogate_thrust_clamp:
        if not config.px4_gazebo_command_send:
            raise CompetitionMainSafetyError(
                "Phase 9E.2 thrust clamp requires --px4-gazebo-command-send"
            )
        if not config.ack_px4_gazebo_surrogate_command_send:
            raise CompetitionMainSafetyError(
                "Phase 9E.2 thrust clamp requires "
                "--ack-px4-gazebo-surrogate-command-send"
            )
        clamp_min = config.px4_gazebo_surrogate_thrust_clamp_min
        clamp_max = config.px4_gazebo_surrogate_thrust_clamp_max
        if clamp_min is None and clamp_max is None:
            raise CompetitionMainSafetyError(
                "Phase 9E.2 thrust clamp requires at least one clamp bound"
            )
        if clamp_min is not None and (
            float(clamp_min) < float(config.px4_gazebo_command_min_thrust)
            or float(clamp_min) > float(config.px4_gazebo_command_max_thrust)
        ):
            raise CompetitionMainSafetyError(
                "px4_gazebo_surrogate_thrust_clamp_min must remain inside "
                "command thrust safety bounds"
            )
        if clamp_max is not None and (
            float(clamp_max) < float(config.px4_gazebo_command_min_thrust)
            or float(clamp_max) > float(config.px4_gazebo_command_max_thrust)
        ):
            raise CompetitionMainSafetyError(
                "px4_gazebo_surrogate_thrust_clamp_max must remain inside "
                "command thrust safety bounds"
            )
        if (
            clamp_min is not None
            and clamp_max is not None
            and float(clamp_min) > float(clamp_max)
        ):
            raise CompetitionMainSafetyError(
                "px4_gazebo_surrogate_thrust_clamp_min must be <= clamp max"
            )
    if config.px4_gazebo_continuous_setpoint_stream:
        if mode != CompetitionRunnerMode.COMMAND_DRY_RUN:
            raise CompetitionMainSafetyError(
                "Phase 9E.1 continuous setpoint stream requires command_dry_run"
            )
        if not config.px4_gazebo_command_send:
            raise CompetitionMainSafetyError(
                "continuous setpoint stream requires --px4-gazebo-command-send"
            )
        if not config.ack_px4_gazebo_surrogate_command_send:
            raise CompetitionMainSafetyError(
                "continuous setpoint stream requires --ack-px4-gazebo-surrogate-command-send"
            )
        if float(config.px4_gazebo_command_max_age_s) <= 0.0:
            raise CompetitionMainSafetyError(
                "px4_gazebo_command_max_age_s must be positive"
            )
    if config.px4_gazebo_arm or config.px4_gazebo_offboard:
        if mode != CompetitionRunnerMode.COMMAND_DRY_RUN:
            raise CompetitionMainSafetyError(
                "Phase 9E PX4/Gazebo arm/offboard requires command_dry_run"
            )
        if config.px4_gazebo_arm and not config.ack_px4_gazebo_surrogate_arm:
            raise CompetitionMainSafetyError(
                "Phase 9E requires --ack-px4-gazebo-surrogate-arm"
            )
        if (
            config.px4_gazebo_offboard
            and not config.ack_px4_gazebo_surrogate_offboard
        ):
            raise CompetitionMainSafetyError(
                "Phase 9E requires --ack-px4-gazebo-surrogate-offboard"
            )
        if not config.live_transports and components is None:
            raise CompetitionMainSafetyError(
                "Phase 9E requires --live-transports or injected test components"
            )
        if (
            not config.px4_gazebo_body_rate_smoke
            and not config.px4_gazebo_attitude_hover_smoke
            and (not config.use_real_autonomy or not config.real_perception)
        ):
            raise CompetitionMainSafetyError(
                "Phase 9E requires --use-real-autonomy and --real-perception"
            )
        if int(config.px4_gazebo_arm_max_attempts) <= 0:
            raise CompetitionMainSafetyError(
                "px4_gazebo_arm_max_attempts must be positive"
            )
        if int(config.px4_gazebo_offboard_max_attempts) <= 0:
            raise CompetitionMainSafetyError(
                "px4_gazebo_offboard_max_attempts must be positive"
            )
        if int(config.px4_gazebo_offboard_prestream_count) < 0:
            raise CompetitionMainSafetyError(
                "px4_gazebo_offboard_prestream_count must be non-negative"
            )
        if (
            config.px4_gazebo_offboard
            and int(config.px4_gazebo_offboard_prestream_count) > 0
            and not (
                config.px4_gazebo_command_send or config.px4_gazebo_body_rate_smoke
                or config.px4_gazebo_attitude_hover_smoke
            )
        ):
            raise CompetitionMainSafetyError(
                "offboard prestream requires --px4-gazebo-command-send or "
                "--px4-gazebo-body-rate-smoke or "
                "--px4-gazebo-attitude-hover-smoke"
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
    if config.px4_gazebo_generic_setpoint_fallback:
        return PHASE_9F_3C
    if config.px4_gazebo_generic_setpoint_streamer:
        return PHASE_9F_3B
    if config.px4_gazebo_fixed_rate_setpoint_stream:
        return PHASE_9F_2
    if config.px4_gazebo_debug_yaw_override_rad is not None:
        return PHASE_9F_1
    if config.px4_gazebo_full_autonomy_loop:
        return PHASE_9F
    if config.px4_gazebo_attitude_hover_smoke:
        return PHASE_9E_4
    if config.px4_gazebo_body_rate_smoke:
        return PHASE_9E_3
    if config.px4_gazebo_surrogate_thrust_clamp:
        return PHASE_9E_2
    if config.px4_gazebo_continuous_setpoint_stream:
        return PHASE_9E_1
    if config.px4_gazebo_arm or config.px4_gazebo_offboard:
        return PHASE_9E
    if config.px4_gazebo_command_send:
        return PHASE_9D
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
    px4_gazebo_command_send: bool = False,
    px4_gazebo_surrogate_thrust_clamp: bool = False,
    px4_gazebo_continuous_setpoint_stream: bool = False,
    px4_gazebo_arm: bool = False,
    px4_gazebo_offboard: bool = False,
    px4_gazebo_body_rate_smoke: bool = False,
    px4_gazebo_attitude_hover_smoke: bool = False,
    px4_gazebo_full_autonomy_loop: bool = False,
    px4_gazebo_debug_yaw_override: bool = False,
    px4_gazebo_fixed_rate_setpoint_stream: bool = False,
    px4_gazebo_generic_setpoint_streamer: bool = False,
    px4_gazebo_generic_setpoint_fallback: bool = False,
) -> str:
    if px4_gazebo_generic_setpoint_fallback:
        return PHASE_9F_3C
    if px4_gazebo_generic_setpoint_streamer:
        return PHASE_9F_3B
    if px4_gazebo_fixed_rate_setpoint_stream:
        return PHASE_9F_2
    if px4_gazebo_debug_yaw_override:
        return PHASE_9F_1
    if px4_gazebo_full_autonomy_loop:
        return PHASE_9F
    if px4_gazebo_attitude_hover_smoke:
        return PHASE_9E_4
    if px4_gazebo_body_rate_smoke:
        return PHASE_9E_3
    if px4_gazebo_surrogate_thrust_clamp:
        return PHASE_9E_2
    if px4_gazebo_continuous_setpoint_stream:
        return PHASE_9E_1
    if px4_gazebo_arm or px4_gazebo_offboard:
        return PHASE_9E
    if px4_gazebo_command_send:
        return PHASE_9D
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
    if config.px4_gazebo_command_send:
        notes.append(PHASE9D_SURROGATE_LIMITATION)
        notes.append(
            "Phase 9D command sending is gated for PX4/Gazebo SITL only and "
            "must not be interpreted as real competition simulator evidence."
        )
    if config.px4_gazebo_arm or config.px4_gazebo_offboard:
        notes.append(PHASE9E_SURROGATE_LIMITATION)
        notes.append(
            "Phase 9E arm/offboard setup is gated for PX4/Gazebo SITL only; "
            "real competition arm/offboard behavior remains deferred."
        )
    if config.px4_gazebo_continuous_setpoint_stream:
        notes.append(PHASE9E1_SURROGATE_LIMITATION)
        notes.append(
            "Phase 9E.1 streams only cached accepted competition command "
            "fields with heartbeat/state freshness and command-age limits."
        )
    if config.px4_gazebo_surrogate_thrust_clamp:
        notes.append(PHASE9E2_SURROGATE_LIMITATION)
        notes.append(
            "Phase 9E.2 reports raw controller thrust separately from the "
            "PX4/Gazebo surrogate thrust actually sent over MAVLink."
        )
    if config.px4_gazebo_body_rate_smoke:
        notes.append(PHASE9E3_SURROGATE_LIMITATION)
        notes.append(
            "Phase 9E.3 fixed body-rate smoke is independent from perception, "
            "planning, AutonomyAPI.attitude_control(), and the competition "
            "angle-to-quaternion command adapter."
        )
    if config.px4_gazebo_attitude_hover_smoke:
        notes.append(PHASE9E4_SURROGATE_LIMITATION)
        notes.append(
            "Phase 9E.4 fixed attitude-angle hover smoke uses roll=0, pitch=0, "
            "current-or-explicit yaw, and explicit thrust to mirror legacy "
            "MAVSDK attitude hover semantics."
        )
    if config.px4_gazebo_full_autonomy_loop:
        notes.append(PHASE9F_SURROGATE_LIMITATION)
        notes.append(
            "Phase 9F uses the existing competition command dry-run adapter "
            "backend: attitude-angle quaternion SET_ATTITUDE_TARGET with type "
            f"mask {ATTITUDE_HOVER_BODY_RATES_IGNORE_TYPE_MASK}; body-rate "
            "competition protocol work remains a separate decision."
        )
    if config.px4_gazebo_debug_yaw_override_rad is not None:
        notes.append(PHASE9F1_SURROGATE_LIMITATION)
        notes.append(
            "Phase 9F.1 preserves AutonomyAPI/controller yaw in the dry-run "
            "command result and replaces only the outgoing PX4/Gazebo "
            "surrogate MAVLink quaternion yaw before send."
        )
    if config.px4_gazebo_fixed_rate_setpoint_stream:
        notes.append(PHASE9F2_SURROGATE_LIMITATION)
        notes.append(
            "Phase 9F.2 sends cached accepted competition command fields at "
            "the configured PX4/Gazebo surrogate stream cadence before the "
            "Offboard mode request gate; it is still PX4/Gazebo-only evidence."
        )
    if config.px4_gazebo_generic_setpoint_streamer:
        notes.append(PHASE9F3B_SURROGATE_LIMITATION)
        notes.append(
            "Phase 9F.3B keeps command publication in the PX4/Gazebo surrogate "
            "sender, but sources cadence and freshness decisions from "
            "CompetitionSetpointStreamer."
        )
    if config.px4_gazebo_generic_setpoint_fallback:
        notes.append(PHASE9F3C_SURROGATE_LIMITATION)
        notes.append(
            "Phase 9F.3C streams explicit fallback/hold fields when autonomy "
            "commands are missing or stale; this is PX4/Gazebo-only evidence "
            "and not competition readiness."
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


def _phase9d_success_criteria(
    *,
    config: CompetitionMainConfig,
    mode: CompetitionRunnerMode,
    aggregate: _Aggregate,
    phase6e_receive_satisfied: bool,
    phase6e_perception_boundary_satisfied: bool,
) -> dict[str, bool]:
    last_send = aggregate.last_command_send_result or {}
    return {
        "mode_is_command_dry_run": mode == CompetitionRunnerMode.COMMAND_DRY_RUN,
        "live_transports_requested": bool(config.live_transports),
        "use_real_autonomy": bool(config.use_real_autonomy),
        "real_perception_requested": bool(config.real_perception),
        "legacy_yolo_default_acknowledged": bool(config.allow_legacy_yolo_default),
        "official_competition_transform_selected": (
            str(config.perception_transform_mode) == COMPETITION_OFFICIAL_TRANSFORM_MODE
        ),
        "px4_gazebo_command_send_requested": bool(config.px4_gazebo_command_send),
        "px4_gazebo_command_send_acknowledged": bool(
            config.ack_px4_gazebo_surrogate_command_send
        ),
        "phase6e_receive_satisfied": bool(phase6e_receive_satisfied),
        "phase6e_perception_boundary_satisfied": bool(
            phase6e_perception_boundary_satisfied
        ),
        "heartbeat_seen": bool(aggregate.heartbeat_seen),
        "state_usable": bool(aggregate.state_usable),
        "perception_update_calls_gt_0": aggregate.perception_update_calls > 0,
        "autonomy_telemetry_sync_count_gt_0": (
            aggregate.autonomy_telemetry_sync_count > 0
        ),
        "planning_success_count_gt_0": aggregate.planning_success_count > 0,
        "command_candidate_count_gt_0": aggregate.command_candidate_count > 0,
        "command_candidate_accepted_count_gt_0": (
            aggregate.command_candidate_accepted_count > 0
        ),
        "command_send_attempt_count_gt_0": aggregate.command_send_attempt_count > 0,
        "command_sent_count_gt_0": aggregate.command_sent_count > 0,
        "command_sent_count_lte_max": (
            aggregate.command_sent_count <= int(config.px4_gazebo_command_max_count)
        ),
        "last_command_send_result_sent": bool(last_send.get("sent", False)),
        "last_command_send_surrogate_labeled": (
            last_send.get("surrogate_label") == PX4_GAZEBO_SURROGATE_LABEL
        ),
        "last_command_send_phase4b_false": last_send.get("phase4b_satisfied") is False,
        "last_command_send_competition_readiness_false": (
            last_send.get("competition_readiness_claimed") is False
        ),
        "phase4b_not_satisfied": True,
        "competition_readiness_not_claimed": True,
    }


def _phase9d_satisfied(criteria: dict[str, bool]) -> bool:
    required = [
        "mode_is_command_dry_run",
        "live_transports_requested",
        "use_real_autonomy",
        "real_perception_requested",
        "official_competition_transform_selected",
        "px4_gazebo_command_send_requested",
        "px4_gazebo_command_send_acknowledged",
        "phase6e_receive_satisfied",
        "phase6e_perception_boundary_satisfied",
        "heartbeat_seen",
        "state_usable",
        "perception_update_calls_gt_0",
        "autonomy_telemetry_sync_count_gt_0",
        "planning_success_count_gt_0",
        "command_candidate_count_gt_0",
        "command_candidate_accepted_count_gt_0",
        "command_send_attempt_count_gt_0",
        "command_sent_count_gt_0",
        "command_sent_count_lte_max",
        "last_command_send_result_sent",
        "last_command_send_surrogate_labeled",
        "last_command_send_phase4b_false",
        "last_command_send_competition_readiness_false",
        "phase4b_not_satisfied",
        "competition_readiness_not_claimed",
    ]
    return all(bool(criteria.get(name, False)) for name in required)


def _phase9e_success_criteria(
    *,
    config: CompetitionMainConfig,
    mode: CompetitionRunnerMode,
    aggregate: _Aggregate,
    phase6e_receive_satisfied: bool,
    phase6e_perception_boundary_satisfied: bool,
    armed_state_observed: bool,
    offboard_state_observed: bool,
) -> dict[str, bool]:
    arm_requested = bool(config.px4_gazebo_arm)
    offboard_requested = bool(config.px4_gazebo_offboard)
    criteria = {
        "mode_is_command_dry_run": mode == CompetitionRunnerMode.COMMAND_DRY_RUN,
        "live_transports_requested": bool(config.live_transports),
        "use_real_autonomy": bool(config.use_real_autonomy),
        "real_perception_requested": bool(config.real_perception),
        "legacy_yolo_default_acknowledged": bool(config.allow_legacy_yolo_default),
        "official_competition_transform_selected": (
            str(config.perception_transform_mode) == COMPETITION_OFFICIAL_TRANSFORM_MODE
        ),
        "px4_gazebo_arm_requested": arm_requested,
        "px4_gazebo_arm_acknowledged": bool(config.ack_px4_gazebo_surrogate_arm),
        "px4_gazebo_offboard_requested": offboard_requested,
        "px4_gazebo_offboard_acknowledged": bool(
            config.ack_px4_gazebo_surrogate_offboard
        ),
        "phase6e_receive_satisfied": bool(phase6e_receive_satisfied),
        "phase6e_perception_boundary_satisfied": bool(
            phase6e_perception_boundary_satisfied
        ),
        "heartbeat_seen": bool(aggregate.heartbeat_seen),
        "state_usable": bool(aggregate.state_usable),
        "perception_update_calls_gt_0": aggregate.perception_update_calls > 0,
        "autonomy_telemetry_sync_count_gt_0": (
            aggregate.autonomy_telemetry_sync_count > 0
        ),
        "planning_success_count_gt_0": aggregate.planning_success_count > 0,
        "command_candidate_count_gt_0": aggregate.command_candidate_count > 0,
        "phase4b_not_satisfied": True,
        "competition_readiness_not_claimed": True,
    }
    if arm_requested:
        criteria.update(
            {
                "arm_attempt_count_gt_0": aggregate.arm_attempt_count > 0,
                "arm_sent_count_gt_0": aggregate.arm_sent_count > 0,
                "arm_sent_count_lte_max": (
                    aggregate.arm_sent_count <= int(config.px4_gazebo_arm_max_attempts)
                ),
                "armed_state_observed": bool(armed_state_observed),
            }
        )
    if offboard_requested:
        criteria.update(
            {
                "offboard_attempt_count_gt_0": aggregate.offboard_attempt_count > 0,
                "offboard_sent_count_gt_0": aggregate.offboard_sent_count > 0,
                "offboard_sent_count_lte_max": (
                    aggregate.offboard_sent_count
                    <= int(config.px4_gazebo_offboard_max_attempts)
                ),
                "offboard_prestream_count_satisfied": (
                    aggregate.command_sent_count
                    >= int(config.px4_gazebo_offboard_prestream_count)
                ),
                "offboard_state_observed": bool(offboard_state_observed),
            }
        )
    return criteria


def _phase9e_satisfied(
    criteria: dict[str, bool],
    config: CompetitionMainConfig,
) -> bool:
    required = [
        "mode_is_command_dry_run",
        "live_transports_requested",
        "use_real_autonomy",
        "real_perception_requested",
        "official_competition_transform_selected",
        "phase6e_receive_satisfied",
        "phase6e_perception_boundary_satisfied",
        "heartbeat_seen",
        "state_usable",
        "perception_update_calls_gt_0",
        "autonomy_telemetry_sync_count_gt_0",
        "planning_success_count_gt_0",
        "command_candidate_count_gt_0",
        "phase4b_not_satisfied",
        "competition_readiness_not_claimed",
    ]
    if config.px4_gazebo_arm:
        required.extend(
            [
                "px4_gazebo_arm_requested",
                "px4_gazebo_arm_acknowledged",
                "arm_attempt_count_gt_0",
                "arm_sent_count_gt_0",
                "arm_sent_count_lte_max",
                "armed_state_observed",
            ]
        )
    if config.px4_gazebo_offboard:
        required.extend(
            [
                "px4_gazebo_offboard_requested",
                "px4_gazebo_offboard_acknowledged",
                "offboard_attempt_count_gt_0",
                "offboard_sent_count_gt_0",
                "offboard_sent_count_lte_max",
                "offboard_prestream_count_satisfied",
                "offboard_state_observed",
            ]
        )
    return all(bool(criteria.get(name, False)) for name in required)


def _phase9e1_success_criteria(
    *,
    config: CompetitionMainConfig,
    mode: CompetitionRunnerMode,
    aggregate: _Aggregate,
    phase9e_satisfied: bool,
) -> dict[str, bool]:
    return {
        "mode_is_command_dry_run": mode == CompetitionRunnerMode.COMMAND_DRY_RUN,
        "px4_gazebo_continuous_setpoint_stream_requested": bool(
            config.px4_gazebo_continuous_setpoint_stream
        ),
        "px4_gazebo_command_send_requested": bool(config.px4_gazebo_command_send),
        "px4_gazebo_command_send_acknowledged": bool(
            config.ack_px4_gazebo_surrogate_command_send
        ),
        "phase9e_surrogate_arm_offboard_satisfied": bool(phase9e_satisfied),
        "command_sent_count_gt_1": aggregate.command_sent_count > 1,
        "command_sent_count_lte_max": (
            aggregate.command_sent_count <= int(config.px4_gazebo_command_max_count)
        ),
        "setpoint_stream_cache_update_count_gt_0": (
            aggregate.setpoint_stream_cache_update_count > 0
        ),
        "setpoint_stream_reused_count_gt_0": (
            aggregate.setpoint_stream_reused_count > 0
        ),
        "setpoint_stream_stale_rejection_count_zero": (
            aggregate.setpoint_stream_stale_rejection_count == 0
        ),
        "last_cached_command_age_within_limit": (
            aggregate.last_cached_command_age_s is not None
            and aggregate.last_cached_command_age_s
            <= float(config.px4_gazebo_command_max_age_s)
        ),
        "phase4b_not_satisfied": True,
        "competition_readiness_not_claimed": True,
    }


def _phase9e1_satisfied(
    criteria: dict[str, bool],
    config: CompetitionMainConfig,
) -> bool:
    if not config.px4_gazebo_continuous_setpoint_stream:
        return False
    required = [
        "mode_is_command_dry_run",
        "px4_gazebo_continuous_setpoint_stream_requested",
        "px4_gazebo_command_send_requested",
        "px4_gazebo_command_send_acknowledged",
        "phase9e_surrogate_arm_offboard_satisfied",
        "command_sent_count_gt_1",
        "command_sent_count_lte_max",
        "setpoint_stream_cache_update_count_gt_0",
        "setpoint_stream_reused_count_gt_0",
        "setpoint_stream_stale_rejection_count_zero",
        "last_cached_command_age_within_limit",
        "phase4b_not_satisfied",
        "competition_readiness_not_claimed",
    ]
    return all(bool(criteria.get(name, False)) for name in required)


def _phase9e2_success_criteria(
    *,
    config: CompetitionMainConfig,
    mode: CompetitionRunnerMode,
    aggregate: _Aggregate,
    command_sender_summary: Optional[dict[str, Any]],
) -> dict[str, bool]:
    last_send = aggregate.last_command_send_result or {}
    sender_stats = (
        command_sender_summary.get("stats", {})
        if isinstance(command_sender_summary, dict)
        else {}
    )
    return {
        "mode_is_command_dry_run": mode == CompetitionRunnerMode.COMMAND_DRY_RUN,
        "px4_gazebo_command_send_requested": bool(config.px4_gazebo_command_send),
        "px4_gazebo_command_send_acknowledged": bool(
            config.ack_px4_gazebo_surrogate_command_send
        ),
        "surrogate_thrust_clamp_requested": bool(
            config.px4_gazebo_surrogate_thrust_clamp
        ),
        "surrogate_thrust_clamp_bound_configured": (
            config.px4_gazebo_surrogate_thrust_clamp_min is not None
            or config.px4_gazebo_surrogate_thrust_clamp_max is not None
        ),
        "command_send_attempt_count_gt_0": aggregate.command_send_attempt_count > 0,
        "command_sent_count_gt_0": aggregate.command_sent_count > 0,
        "command_sent_count_lte_max": (
            aggregate.command_sent_count <= int(config.px4_gazebo_command_max_count)
        ),
        "last_command_send_result_sent": bool(last_send.get("sent", False)),
        "last_command_send_surrogate_labeled": (
            last_send.get("surrogate_label") == PX4_GAZEBO_SURROGATE_LABEL
        ),
        "last_command_send_thrust_clamped": bool(
            last_send.get("thrust_clamped", False)
        ),
        "last_command_send_raw_thrust_recorded": (
            last_send.get("raw_thrust") is not None
        ),
        "last_command_send_stream_source_recorded": bool(
            last_send.get("stream_source")
        ),
        "sender_thrust_clamp_count_gt_0": (
            int(sender_stats.get("thrust_clamp_count", 0)) > 0
        ),
        "phase4b_not_satisfied": True,
        "competition_readiness_not_claimed": True,
    }


def _phase9e2_satisfied(
    criteria: dict[str, bool],
    config: CompetitionMainConfig,
) -> bool:
    if not config.px4_gazebo_surrogate_thrust_clamp:
        return False
    required = [
        "mode_is_command_dry_run",
        "px4_gazebo_command_send_requested",
        "px4_gazebo_command_send_acknowledged",
        "surrogate_thrust_clamp_requested",
        "surrogate_thrust_clamp_bound_configured",
        "command_send_attempt_count_gt_0",
        "command_sent_count_gt_0",
        "command_sent_count_lte_max",
        "last_command_send_result_sent",
        "last_command_send_surrogate_labeled",
        "last_command_send_thrust_clamped",
        "last_command_send_raw_thrust_recorded",
        "last_command_send_stream_source_recorded",
        "sender_thrust_clamp_count_gt_0",
        "phase4b_not_satisfied",
        "competition_readiness_not_claimed",
    ]
    return all(bool(criteria.get(name, False)) for name in required)


def _phase9e3_success_criteria(
    *,
    config: CompetitionMainConfig,
    mode: CompetitionRunnerMode,
    aggregate: _Aggregate,
    armed_state_observed: bool,
    offboard_state_observed: bool,
) -> dict[str, bool]:
    last_send = aggregate.last_body_rate_command_send_result or {}
    requested_thrust = config.px4_gazebo_body_rate_thrust
    criteria = {
        "mode_is_command_dry_run": mode == CompetitionRunnerMode.COMMAND_DRY_RUN,
        "live_transports_requested": bool(config.live_transports),
        "body_rate_smoke_requested": bool(config.px4_gazebo_body_rate_smoke),
        "body_rate_smoke_acknowledged": bool(
            config.ack_px4_gazebo_surrogate_body_rate_smoke
        ),
        "use_real_autonomy_false": not bool(config.use_real_autonomy),
        "real_perception_requested_false": not bool(config.real_perception),
        "heartbeat_seen": bool(aggregate.heartbeat_seen),
        "state_usable": bool(aggregate.state_usable),
        "body_rate_command_sent_count_gt_0": (
            aggregate.body_rate_command_sent_count > 0
        ),
        "body_rate_command_rejection_count_zero": (
            aggregate.body_rate_command_rejection_count == 0
        ),
        "body_rate_command_sent_count_lte_max": (
            aggregate.body_rate_command_sent_count
            <= int(config.px4_gazebo_command_max_count)
        ),
        "last_body_rate_command_sent": bool(last_send.get("sent", False)),
        "last_body_rate_type_mask_attitude_ignore": (
            last_send.get("type_mask") == BODY_RATE_ATTITUDE_IGNORE_TYPE_MASK
        ),
        "last_body_rate_q_identity": (
            tuple(last_send.get("q") or ()) == BODY_RATE_DUMMY_QUATERNION
        ),
        "last_body_roll_rate_matches": _float_equal(
            last_send.get("body_roll_rate"),
            config.px4_gazebo_body_roll_rate,
        ),
        "last_body_pitch_rate_matches": _float_equal(
            last_send.get("body_pitch_rate"),
            config.px4_gazebo_body_pitch_rate,
        ),
        "last_body_yaw_rate_matches": _float_equal(
            last_send.get("body_yaw_rate"),
            config.px4_gazebo_body_yaw_rate,
        ),
        "last_body_rate_thrust_matches": (
            requested_thrust is not None
            and _float_equal(last_send.get("thrust"), requested_thrust)
        ),
        "last_body_rate_surrogate_labeled": (
            last_send.get("surrogate_label") == PX4_GAZEBO_BODY_RATE_SMOKE_LABEL
        ),
        "last_body_rate_phase4b_false": last_send.get("phase4b_satisfied") is False,
        "last_body_rate_competition_readiness_false": (
            last_send.get("competition_readiness_claimed") is False
        ),
        "phase4b_not_satisfied": True,
        "competition_readiness_not_claimed": True,
    }
    if config.px4_gazebo_arm:
        criteria["armed_state_observed"] = bool(armed_state_observed)
    if config.px4_gazebo_offboard:
        criteria["offboard_state_observed"] = bool(offboard_state_observed)
        criteria["offboard_prestream_count_satisfied"] = (
            aggregate.command_sent_count + aggregate.body_rate_command_sent_count
            >= int(config.px4_gazebo_offboard_prestream_count)
        )
    return criteria


def _phase9e3_satisfied(
    criteria: dict[str, bool],
    config: CompetitionMainConfig,
) -> bool:
    if not config.px4_gazebo_body_rate_smoke:
        return False
    required = [
        "mode_is_command_dry_run",
        "live_transports_requested",
        "body_rate_smoke_requested",
        "body_rate_smoke_acknowledged",
        "use_real_autonomy_false",
        "real_perception_requested_false",
        "heartbeat_seen",
        "state_usable",
        "body_rate_command_sent_count_gt_0",
        "body_rate_command_rejection_count_zero",
        "body_rate_command_sent_count_lte_max",
        "last_body_rate_command_sent",
        "last_body_rate_type_mask_attitude_ignore",
        "last_body_rate_q_identity",
        "last_body_roll_rate_matches",
        "last_body_pitch_rate_matches",
        "last_body_yaw_rate_matches",
        "last_body_rate_thrust_matches",
        "last_body_rate_surrogate_labeled",
        "last_body_rate_phase4b_false",
        "last_body_rate_competition_readiness_false",
        "phase4b_not_satisfied",
        "competition_readiness_not_claimed",
    ]
    if config.px4_gazebo_arm:
        required.append("armed_state_observed")
    if config.px4_gazebo_offboard:
        required.extend(
            [
                "offboard_state_observed",
                "offboard_prestream_count_satisfied",
            ]
        )
    return all(bool(criteria.get(name, False)) for name in required)


def _phase9e4_success_criteria(
    *,
    config: CompetitionMainConfig,
    mode: CompetitionRunnerMode,
    aggregate: _Aggregate,
    armed_state_observed: bool,
    offboard_state_observed: bool,
) -> dict[str, bool]:
    last_send = aggregate.last_attitude_hover_command_send_result or {}
    requested_thrust = config.px4_gazebo_attitude_hover_thrust
    requested_yaw = config.px4_gazebo_attitude_hover_yaw_rad
    last_yaw = last_send.get("yaw_rad")
    criteria = {
        "mode_is_command_dry_run": mode == CompetitionRunnerMode.COMMAND_DRY_RUN,
        "live_transports_requested": bool(config.live_transports),
        "attitude_hover_smoke_requested": bool(
            config.px4_gazebo_attitude_hover_smoke
        ),
        "attitude_hover_smoke_acknowledged": bool(
            config.ack_px4_gazebo_surrogate_attitude_hover_smoke
        ),
        "use_real_autonomy_false": not bool(config.use_real_autonomy),
        "real_perception_requested_false": not bool(config.real_perception),
        "heartbeat_seen": bool(aggregate.heartbeat_seen),
        "state_usable": bool(aggregate.state_usable),
        "attitude_hover_command_sent_count_gt_0": (
            aggregate.attitude_hover_command_sent_count > 0
        ),
        "attitude_hover_command_rejection_count_zero": (
            aggregate.attitude_hover_command_rejection_count == 0
        ),
        "attitude_hover_command_sent_count_lte_max": (
            aggregate.attitude_hover_command_sent_count
            <= int(config.px4_gazebo_command_max_count)
        ),
        "last_attitude_hover_command_sent": bool(last_send.get("sent", False)),
        "last_attitude_hover_type_mask_body_rates_ignore": (
            last_send.get("type_mask") == ATTITUDE_HOVER_BODY_RATES_IGNORE_TYPE_MASK
        ),
        "last_attitude_hover_body_rates_zero": (
            _float_equal(last_send.get("body_roll_rate"), 0.0)
            and _float_equal(last_send.get("body_pitch_rate"), 0.0)
            and _float_equal(last_send.get("body_yaw_rate"), 0.0)
        ),
        "last_attitude_hover_roll_matches": _float_equal(
            last_send.get("roll_rad"),
            config.px4_gazebo_attitude_hover_roll_rad,
        ),
        "last_attitude_hover_pitch_matches": _float_equal(
            last_send.get("pitch_rad"),
            config.px4_gazebo_attitude_hover_pitch_rad,
        ),
        "last_attitude_hover_yaw_available": last_yaw is not None,
        "last_attitude_hover_yaw_matches_explicit": (
            requested_yaw is None or _float_equal(last_yaw, requested_yaw)
        ),
        "last_attitude_hover_yaw_source_expected": (
            last_send.get("yaw_source") == _attitude_hover_yaw_source(config)
        ),
        "last_attitude_hover_thrust_matches": (
            requested_thrust is not None
            and _float_equal(last_send.get("thrust"), requested_thrust)
        ),
        "last_attitude_hover_surrogate_labeled": (
            last_send.get("surrogate_label") == PX4_GAZEBO_ATTITUDE_HOVER_SMOKE_LABEL
        ),
        "last_attitude_hover_phase4b_false": (
            last_send.get("phase4b_satisfied") is False
        ),
        "last_attitude_hover_competition_readiness_false": (
            last_send.get("competition_readiness_claimed") is False
        ),
        "phase4b_not_satisfied": True,
        "competition_readiness_not_claimed": True,
    }
    if config.px4_gazebo_arm:
        criteria["armed_state_observed"] = bool(armed_state_observed)
    if config.px4_gazebo_offboard:
        criteria["offboard_state_observed"] = bool(offboard_state_observed)
        criteria["offboard_prestream_count_satisfied"] = (
            aggregate.command_sent_count
            + aggregate.body_rate_command_sent_count
            + aggregate.attitude_hover_command_sent_count
            >= int(config.px4_gazebo_offboard_prestream_count)
        )
    return criteria


def _phase9e4_satisfied(
    criteria: dict[str, bool],
    config: CompetitionMainConfig,
) -> bool:
    if not config.px4_gazebo_attitude_hover_smoke:
        return False
    required = [
        "mode_is_command_dry_run",
        "live_transports_requested",
        "attitude_hover_smoke_requested",
        "attitude_hover_smoke_acknowledged",
        "use_real_autonomy_false",
        "real_perception_requested_false",
        "heartbeat_seen",
        "state_usable",
        "attitude_hover_command_sent_count_gt_0",
        "attitude_hover_command_rejection_count_zero",
        "attitude_hover_command_sent_count_lte_max",
        "last_attitude_hover_command_sent",
        "last_attitude_hover_type_mask_body_rates_ignore",
        "last_attitude_hover_body_rates_zero",
        "last_attitude_hover_roll_matches",
        "last_attitude_hover_pitch_matches",
        "last_attitude_hover_yaw_available",
        "last_attitude_hover_yaw_matches_explicit",
        "last_attitude_hover_yaw_source_expected",
        "last_attitude_hover_thrust_matches",
        "last_attitude_hover_surrogate_labeled",
        "last_attitude_hover_phase4b_false",
        "last_attitude_hover_competition_readiness_false",
        "phase4b_not_satisfied",
        "competition_readiness_not_claimed",
    ]
    if config.px4_gazebo_arm:
        required.append("armed_state_observed")
    if config.px4_gazebo_offboard:
        required.extend(
            [
                "offboard_state_observed",
                "offboard_prestream_count_satisfied",
            ]
        )
    return all(bool(criteria.get(name, False)) for name in required)


def _phase9f_success_criteria(
    *,
    config: CompetitionMainConfig,
    mode: CompetitionRunnerMode,
    aggregate: _Aggregate,
    duration_s: float,
    command_sender_summary: Optional[dict[str, Any]],
    phase6e_receive_satisfied: bool,
    phase6e_perception_boundary_satisfied: bool,
    phase9c_satisfied: bool,
    phase9d_satisfied: bool,
    phase9e_satisfied: bool,
) -> dict[str, bool]:
    last_send = aggregate.last_command_send_result or {}
    command_result = aggregate.last_command_result or {}
    fields = command_result.get("fields") or {}
    q = last_send.get("q") or ()
    command_send_rate_hz = _rate_or_none(aggregate.command_sent_count, duration_s)
    vision_frame_rate_hz = _rate_or_none(aggregate.vision_frames_completed, duration_s)
    max_send_gap_s = _phase9f_max_send_gap_s(command_sender_summary)
    debug_yaw_requested = config.px4_gazebo_debug_yaw_override_rad is not None
    debug_yaw_override_rad = (
        float(config.px4_gazebo_debug_yaw_override_rad)
        if debug_yaw_requested
        else None
    )
    return {
        "mode_is_command_dry_run": mode == CompetitionRunnerMode.COMMAND_DRY_RUN,
        "live_transports_requested": bool(config.live_transports),
        "use_real_autonomy": bool(config.use_real_autonomy),
        "real_perception_requested": bool(config.real_perception),
        "legacy_yolo_default_acknowledged": bool(config.allow_legacy_yolo_default),
        "official_competition_transform_selected": (
            str(config.perception_transform_mode) == COMPETITION_OFFICIAL_TRANSFORM_MODE
        ),
        "px4_gazebo_full_autonomy_loop_requested": bool(
            config.px4_gazebo_full_autonomy_loop
        ),
        "px4_gazebo_full_autonomy_loop_acknowledged": bool(
            config.ack_px4_gazebo_surrogate_full_autonomy_loop
        ),
        "px4_gazebo_command_send_requested": bool(config.px4_gazebo_command_send),
        "px4_gazebo_command_send_acknowledged": bool(
            config.ack_px4_gazebo_surrogate_command_send
        ),
        "px4_gazebo_arm_requested": bool(config.px4_gazebo_arm),
        "px4_gazebo_arm_acknowledged": bool(config.ack_px4_gazebo_surrogate_arm),
        "px4_gazebo_offboard_requested": bool(config.px4_gazebo_offboard),
        "px4_gazebo_offboard_acknowledged": bool(
            config.ack_px4_gazebo_surrogate_offboard
        ),
        "px4_gazebo_continuous_setpoint_stream_requested": bool(
            config.px4_gazebo_continuous_setpoint_stream
        ),
        "fixed_smoke_modes_disabled": (
            not config.px4_gazebo_body_rate_smoke
            and not config.px4_gazebo_attitude_hover_smoke
        ),
        "phase6e_receive_satisfied": bool(phase6e_receive_satisfied),
        "phase6e_perception_boundary_satisfied": bool(
            phase6e_perception_boundary_satisfied
        ),
        "phase9c_command_dry_run_satisfied": bool(phase9c_satisfied),
        "phase9d_surrogate_command_send_satisfied": bool(phase9d_satisfied),
        "phase9e_surrogate_arm_offboard_satisfied": bool(phase9e_satisfied),
        "heartbeat_seen": bool(aggregate.heartbeat_seen),
        "state_usable": bool(aggregate.state_usable),
        "vision_frames_completed_gt_0": aggregate.vision_frames_completed > 0,
        "perception_update_calls_gt_0": aggregate.perception_update_calls > 0,
        "autonomy_telemetry_sync_count_gt_0": (
            aggregate.autonomy_telemetry_sync_count > 0
        ),
        "planning_success_count_gt_0": aggregate.planning_success_count > 0,
        "command_candidate_accepted_count_gt_0": (
            aggregate.command_candidate_accepted_count > 0
        ),
        "command_sent_count_gt_0": aggregate.command_sent_count > 0,
        "command_sent_count_lte_max": (
            aggregate.command_sent_count <= int(config.px4_gazebo_command_max_count)
        ),
        "command_send_rejection_count_zero": (
            aggregate.command_send_rejection_count == 0
        ),
        "setpoint_stream_stale_rejection_count_zero": (
            aggregate.setpoint_stream_stale_rejection_count == 0
        ),
        "phase9f_command_send_rate_hz_gt_10": (
            command_send_rate_hz is not None
            and command_send_rate_hz > PHASE9F_MIN_COMMAND_SEND_RATE_HZ
        ),
        "phase9f_vision_frame_rate_hz_gt_5": (
            vision_frame_rate_hz is not None
            and vision_frame_rate_hz > PHASE9F_MIN_VISION_FRAME_RATE_HZ
        ),
        "phase9f_max_send_gap_s_lte_0_5": (
            max_send_gap_s is not None
            and max_send_gap_s <= PHASE9F_MAX_SEND_GAP_S
        ),
        "last_command_send_result_sent": bool(last_send.get("sent", False)),
        "last_command_send_surrogate_labeled": (
            last_send.get("surrogate_label") == PX4_GAZEBO_SURROGATE_LABEL
        ),
        "last_command_result_attitude_angle_type_mask": (
            fields.get("type_mask") == ATTITUDE_HOVER_BODY_RATES_IGNORE_TYPE_MASK
        ),
        "last_command_send_attitude_angle_type_mask": (
            last_send.get("type_mask") == ATTITUDE_HOVER_BODY_RATES_IGNORE_TYPE_MASK
        ),
        "last_command_send_has_quaternion": len(tuple(q)) == 4,
        "last_command_send_body_rates_zero": (
            _float_equal(last_send.get("body_roll_rate"), 0.0)
            and _float_equal(last_send.get("body_pitch_rate"), 0.0)
            and _float_equal(last_send.get("body_yaw_rate"), 0.0)
        ),
        "debug_yaw_override_not_requested_or_acknowledged": (
            not debug_yaw_requested
            or bool(config.ack_px4_gazebo_surrogate_debug_yaw_override)
        ),
        "debug_yaw_override_not_requested_or_applied": (
            not debug_yaw_requested
            or aggregate.debug_yaw_override_applied_count > 0
        ),
        "debug_yaw_override_not_requested_or_last_send_marked": (
            not debug_yaw_requested
            or bool(last_send.get("debug_yaw_override_applied", False))
        ),
        "debug_yaw_override_not_requested_or_last_send_matches": (
            not debug_yaw_requested
            or (
                _float_equal(last_send.get("debug_yaw_override_rad"), debug_yaw_override_rad)
                and _float_equal(last_send.get("yaw_rad"), debug_yaw_override_rad)
            )
        ),
        "last_command_send_phase4b_false": last_send.get("phase4b_satisfied") is False,
        "last_command_send_competition_readiness_false": (
            last_send.get("competition_readiness_claimed") is False
        ),
        "phase4b_not_satisfied": True,
        "competition_readiness_not_claimed": True,
    }


def _phase9f_satisfied(
    criteria: dict[str, bool],
    config: CompetitionMainConfig,
) -> bool:
    if not config.px4_gazebo_full_autonomy_loop:
        return False
    required = [
        "mode_is_command_dry_run",
        "live_transports_requested",
        "use_real_autonomy",
        "real_perception_requested",
        "legacy_yolo_default_acknowledged",
        "official_competition_transform_selected",
        "px4_gazebo_full_autonomy_loop_requested",
        "px4_gazebo_full_autonomy_loop_acknowledged",
        "px4_gazebo_command_send_requested",
        "px4_gazebo_command_send_acknowledged",
        "px4_gazebo_arm_requested",
        "px4_gazebo_arm_acknowledged",
        "px4_gazebo_offboard_requested",
        "px4_gazebo_offboard_acknowledged",
        "px4_gazebo_continuous_setpoint_stream_requested",
        "fixed_smoke_modes_disabled",
        "phase6e_receive_satisfied",
        "phase6e_perception_boundary_satisfied",
        "phase9c_command_dry_run_satisfied",
        "phase9d_surrogate_command_send_satisfied",
        "phase9e_surrogate_arm_offboard_satisfied",
        "heartbeat_seen",
        "state_usable",
        "vision_frames_completed_gt_0",
        "perception_update_calls_gt_0",
        "autonomy_telemetry_sync_count_gt_0",
        "planning_success_count_gt_0",
        "command_candidate_accepted_count_gt_0",
        "command_sent_count_gt_0",
        "command_sent_count_lte_max",
        "command_send_rejection_count_zero",
        "setpoint_stream_stale_rejection_count_zero",
        "phase9f_command_send_rate_hz_gt_10",
        "phase9f_vision_frame_rate_hz_gt_5",
        "phase9f_max_send_gap_s_lte_0_5",
        "last_command_send_result_sent",
        "last_command_send_surrogate_labeled",
        "last_command_result_attitude_angle_type_mask",
        "last_command_send_attitude_angle_type_mask",
        "last_command_send_has_quaternion",
        "last_command_send_body_rates_zero",
        "debug_yaw_override_not_requested_or_acknowledged",
        "debug_yaw_override_not_requested_or_applied",
        "debug_yaw_override_not_requested_or_last_send_marked",
        "debug_yaw_override_not_requested_or_last_send_matches",
        "last_command_send_phase4b_false",
        "last_command_send_competition_readiness_false",
        "phase4b_not_satisfied",
        "competition_readiness_not_claimed",
    ]
    return all(bool(criteria.get(name, False)) for name in required)


def _phase9f2_success_criteria(
    *,
    config: CompetitionMainConfig,
    mode: CompetitionRunnerMode,
    aggregate: _Aggregate,
    phase9f_satisfied: bool,
) -> dict[str, bool]:
    stream_hz = float(config.px4_gazebo_setpoint_stream_hz)
    last_send = aggregate.last_command_send_result or {}
    return {
        "mode_is_command_dry_run": mode == CompetitionRunnerMode.COMMAND_DRY_RUN,
        "px4_gazebo_fixed_rate_setpoint_stream_requested": bool(
            config.px4_gazebo_fixed_rate_setpoint_stream
        ),
        "px4_gazebo_fixed_rate_setpoint_stream_acknowledged": bool(
            config.ack_px4_gazebo_surrogate_fixed_rate_setpoint_stream
        ),
        "px4_gazebo_full_autonomy_loop_requested": bool(
            config.px4_gazebo_full_autonomy_loop
        ),
        "px4_gazebo_continuous_setpoint_stream_requested": bool(
            config.px4_gazebo_continuous_setpoint_stream
        ),
        "stream_hz_positive": stream_hz > 0.0,
        "stream_hz_below_100": stream_hz <= PHASE9F2_MAX_SETPOINT_STREAM_HZ,
        "burst_limit_positive": (
            int(config.px4_gazebo_setpoint_stream_burst_limit) > 0
        ),
        "offboard_prestream_count_positive": (
            not config.px4_gazebo_offboard
            or int(config.px4_gazebo_offboard_prestream_count) > 0
        ),
        "offboard_prestream_count_satisfied": (
            not config.px4_gazebo_offboard
            or aggregate.command_sent_count
            >= int(config.px4_gazebo_offboard_prestream_count)
        ),
        "fixed_rate_iteration_count_gt_0": (
            aggregate.fixed_rate_setpoint_stream_iteration_count > 0
        ),
        "fixed_rate_attempt_count_gt_0": (
            aggregate.fixed_rate_setpoint_stream_attempt_count > 0
        ),
        "fixed_rate_sent_count_gt_0": (
            aggregate.fixed_rate_setpoint_stream_sent_count > 0
        ),
        "fixed_rate_rejection_count_zero": (
            aggregate.fixed_rate_setpoint_stream_rejection_count == 0
        ),
        "fixed_rate_sent_count_equals_command_sent_count": (
            aggregate.fixed_rate_setpoint_stream_sent_count
            == aggregate.command_sent_count
        ),
        "last_command_send_fixed_rate_marked": bool(
            last_send.get("fixed_rate_setpoint_stream", False)
        ),
        "last_command_send_surrogate_labeled": (
            last_send.get("surrogate_label") == PX4_GAZEBO_SURROGATE_LABEL
        ),
        "phase9f_full_autonomy_loop_satisfied": bool(phase9f_satisfied),
        "phase4b_not_satisfied": True,
        "competition_readiness_not_claimed": True,
    }


def _phase9f2_satisfied(
    criteria: dict[str, bool],
    config: CompetitionMainConfig,
) -> bool:
    if not config.px4_gazebo_fixed_rate_setpoint_stream:
        return False
    required = [
        "mode_is_command_dry_run",
        "px4_gazebo_fixed_rate_setpoint_stream_requested",
        "px4_gazebo_fixed_rate_setpoint_stream_acknowledged",
        "px4_gazebo_full_autonomy_loop_requested",
        "px4_gazebo_continuous_setpoint_stream_requested",
        "stream_hz_positive",
        "stream_hz_below_100",
        "burst_limit_positive",
        "offboard_prestream_count_positive",
        "offboard_prestream_count_satisfied",
        "fixed_rate_iteration_count_gt_0",
        "fixed_rate_attempt_count_gt_0",
        "fixed_rate_sent_count_gt_0",
        "fixed_rate_rejection_count_zero",
        "fixed_rate_sent_count_equals_command_sent_count",
        "last_command_send_fixed_rate_marked",
        "last_command_send_surrogate_labeled",
        "phase9f_full_autonomy_loop_satisfied",
        "phase4b_not_satisfied",
        "competition_readiness_not_claimed",
    ]
    return all(bool(criteria.get(name, False)) for name in required)


def _phase9f3b_success_criteria(
    *,
    config: CompetitionMainConfig,
    mode: CompetitionRunnerMode,
    aggregate: _Aggregate,
    phase9f2_satisfied: bool,
) -> dict[str, bool]:
    last_send = aggregate.last_command_send_result or {}
    streamer_summary = aggregate.generic_setpoint_streamer_summary or {}
    streamer_stats = streamer_summary.get("stats", {})
    if not isinstance(streamer_stats, dict):
        streamer_stats = {}
    return {
        "mode_is_command_dry_run": mode == CompetitionRunnerMode.COMMAND_DRY_RUN,
        "px4_gazebo_generic_setpoint_streamer_requested": bool(
            config.px4_gazebo_generic_setpoint_streamer
        ),
        "px4_gazebo_generic_setpoint_streamer_acknowledged": bool(
            config.ack_px4_gazebo_surrogate_generic_setpoint_streamer
        ),
        "px4_gazebo_fixed_rate_setpoint_stream_requested": bool(
            config.px4_gazebo_fixed_rate_setpoint_stream
        ),
        "px4_gazebo_fixed_rate_setpoint_stream_acknowledged": bool(
            config.ack_px4_gazebo_surrogate_fixed_rate_setpoint_stream
        ),
        "phase9f2_fixed_rate_setpoint_stream_satisfied": bool(phase9f2_satisfied),
        "generic_setpoint_streamer_summary_present": bool(streamer_summary),
        "generic_setpoint_streamer_emit_count_gt_0": (
            int(streamer_stats.get("emit_count", 0)) > 0
        ),
        "generic_setpoint_streamer_invalid_update_count_zero": (
            int(streamer_stats.get("invalid_update_count", 0)) == 0
        ),
        "last_command_send_generic_streamer_marked": bool(
            last_send.get("generic_setpoint_streamer", False)
        ),
        "last_command_send_generic_streamer_label": (
            last_send.get("generic_setpoint_streamer_label")
            == PX4_GAZEBO_GENERIC_SETPOINT_STREAMER_LABEL
        ),
        "last_command_send_generic_streamer_phase4b_false": (
            last_send.get("generic_setpoint_streamer_phase4b_satisfied") is False
        ),
        "last_command_send_generic_streamer_readiness_false": (
            last_send.get(
                "generic_setpoint_streamer_competition_readiness_claimed"
            )
            is False
        ),
        "phase4b_not_satisfied": True,
        "competition_readiness_not_claimed": True,
    }


def _phase9f3b_satisfied(
    criteria: dict[str, bool],
    config: CompetitionMainConfig,
) -> bool:
    if not config.px4_gazebo_generic_setpoint_streamer:
        return False
    required = [
        "mode_is_command_dry_run",
        "px4_gazebo_generic_setpoint_streamer_requested",
        "px4_gazebo_generic_setpoint_streamer_acknowledged",
        "px4_gazebo_fixed_rate_setpoint_stream_requested",
        "px4_gazebo_fixed_rate_setpoint_stream_acknowledged",
        "phase9f2_fixed_rate_setpoint_stream_satisfied",
        "generic_setpoint_streamer_summary_present",
        "generic_setpoint_streamer_emit_count_gt_0",
        "generic_setpoint_streamer_invalid_update_count_zero",
        "last_command_send_generic_streamer_marked",
        "last_command_send_generic_streamer_label",
        "last_command_send_generic_streamer_phase4b_false",
        "last_command_send_generic_streamer_readiness_false",
        "phase4b_not_satisfied",
        "competition_readiness_not_claimed",
    ]
    return all(bool(criteria.get(name, False)) for name in required)


def _phase9f3c_success_criteria(
    *,
    config: CompetitionMainConfig,
    mode: CompetitionRunnerMode,
    aggregate: _Aggregate,
) -> dict[str, bool]:
    last_send = aggregate.last_command_send_result or {}
    streamer_summary = aggregate.generic_setpoint_streamer_summary or {}
    streamer_stats = streamer_summary.get("stats", {})
    if not isinstance(streamer_stats, dict):
        streamer_stats = {}
    return {
        "mode_is_command_dry_run": mode == CompetitionRunnerMode.COMMAND_DRY_RUN,
        "px4_gazebo_generic_setpoint_fallback_requested": bool(
            config.px4_gazebo_generic_setpoint_fallback
        ),
        "px4_gazebo_generic_setpoint_fallback_acknowledged": bool(
            config.ack_px4_gazebo_surrogate_generic_setpoint_fallback
        ),
        "px4_gazebo_generic_setpoint_streamer_requested": bool(
            config.px4_gazebo_generic_setpoint_streamer
        ),
        "px4_gazebo_generic_setpoint_streamer_acknowledged": bool(
            config.ack_px4_gazebo_surrogate_generic_setpoint_streamer
        ),
        "fallback_thrust_configured": (
            config.px4_gazebo_generic_fallback_thrust is not None
        ),
        "fallback_update_count_gt_0": (
            aggregate.generic_setpoint_fallback_update_count > 0
        ),
        "fallback_update_rejection_count_zero": (
            aggregate.generic_setpoint_fallback_rejection_count == 0
        ),
        "generic_setpoint_streamer_summary_present": bool(streamer_summary),
        "generic_setpoint_streamer_fallback_update_count_gt_0": (
            int(streamer_stats.get("fallback_update_count", 0)) > 0
        ),
        "generic_setpoint_streamer_fallback_emit_count_gt_0": (
            int(streamer_stats.get("fallback_emit_count", 0)) > 0
        ),
        "fixed_rate_sent_count_gt_0": (
            aggregate.fixed_rate_setpoint_stream_sent_count > 0
        ),
        "fixed_rate_rejection_count_zero": (
            aggregate.fixed_rate_setpoint_stream_rejection_count == 0
        ),
        "command_send_rejection_count_zero": (
            aggregate.command_send_rejection_count == 0
        ),
        "last_command_send_generic_fallback_marked": bool(
            last_send.get("generic_setpoint_fallback", False)
        ),
        "last_command_send_generic_fallback_label": (
            last_send.get("generic_setpoint_fallback_label")
            == PX4_GAZEBO_GENERIC_SETPOINT_FALLBACK_LABEL
        ),
        "phase4b_not_satisfied": True,
        "competition_readiness_not_claimed": True,
    }


def _phase9f3c_satisfied(
    criteria: dict[str, bool],
    config: CompetitionMainConfig,
) -> bool:
    if not config.px4_gazebo_generic_setpoint_fallback:
        return False
    required = [
        "mode_is_command_dry_run",
        "px4_gazebo_generic_setpoint_fallback_requested",
        "px4_gazebo_generic_setpoint_fallback_acknowledged",
        "px4_gazebo_generic_setpoint_streamer_requested",
        "px4_gazebo_generic_setpoint_streamer_acknowledged",
        "fallback_thrust_configured",
        "fallback_update_count_gt_0",
        "fallback_update_rejection_count_zero",
        "generic_setpoint_streamer_summary_present",
        "generic_setpoint_streamer_fallback_update_count_gt_0",
        "generic_setpoint_streamer_fallback_emit_count_gt_0",
        "fixed_rate_sent_count_gt_0",
        "fixed_rate_rejection_count_zero",
        "command_send_rejection_count_zero",
        "last_command_send_generic_fallback_marked",
        "last_command_send_generic_fallback_label",
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


def _result_to_dict(result: Any) -> dict[str, Any]:
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return dict(result)


def _armed_state_observed(mavlink_summary: Optional[dict[str, Any]]) -> bool:
    for sample in _mavlink_summary_samples(mavlink_summary, "HEARTBEAT"):
        base_mode = _int_or_none(sample.get("base_mode"))
        if base_mode is not None and base_mode & 128:
            return True
    return False


def _offboard_state_observed(mavlink_summary: Optional[dict[str, Any]]) -> bool:
    for sample in _mavlink_summary_samples(mavlink_summary, "HEARTBEAT"):
        custom_mode = _int_or_none(sample.get("custom_mode"))
        if custom_mode is None:
            continue
        px4_main_mode = (custom_mode >> 16) & 0xFF
        if px4_main_mode == 6:
            return True
    return False


def _mavlink_summary_samples(
    mavlink_summary: Optional[dict[str, Any]],
    message_type: str,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(mavlink_summary, dict):
        return ()
    message_types = mavlink_summary.get("message_types")
    if not isinstance(message_types, dict):
        return ()
    entry = message_types.get(message_type)
    if not isinstance(entry, dict):
        return ()
    samples = entry.get("samples")
    if not isinstance(samples, list):
        return ()
    return tuple(sample for sample in samples if isinstance(sample, dict))


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_equal(left: Any, right: Any, *, abs_tol: float = 1e-9) -> bool:
    try:
        left_float = float(left)
        right_float = float(right)
    except (TypeError, ValueError):
        return False
    return math.isfinite(left_float) and math.isclose(
        left_float,
        right_float,
        rel_tol=0.0,
        abs_tol=abs_tol,
    )


def _euler_zyx_from_quaternion_wxyz(
    q: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    if len(q) != 4:
        raise ValueError("q must contain four wxyz components")
    w, x, y, z = (float(component) for component in q)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("q must have a finite non-zero norm")
    w /= norm
    x /= norm
    y /= norm
    z /= norm

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _rate_or_none(count: int, duration_s: float) -> Optional[float]:
    duration = float(duration_s)
    if duration <= 0.0 or not math.isfinite(duration):
        return None
    return float(count) / duration


def _phase9f_max_send_gap_s(
    command_sender_summary: Optional[dict[str, Any]],
) -> Optional[float]:
    if not isinstance(command_sender_summary, dict):
        return None
    stats = command_sender_summary.get("stats")
    if not isinstance(stats, dict):
        return None
    value = stats.get("max_send_gap_s")
    if value is None:
        return None
    try:
        gap = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(gap):
        return None
    return gap


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
    "PHASE_9D",
    "PHASE_9E",
    "PHASE_9E_1",
    "PHASE_9E_2",
    "PHASE_9E_3",
    "PHASE_9E_4",
    "PHASE_9F",
    "PHASE_9F_1",
    "PHASE6E_SURROGATE_LIMITATION",
    "PHASE9D_SURROGATE_LIMITATION",
    "PHASE9E_SURROGATE_LIMITATION",
    "PHASE9E1_SURROGATE_LIMITATION",
    "PHASE9E2_SURROGATE_LIMITATION",
    "PHASE9E3_SURROGATE_LIMITATION",
    "PHASE9E4_SURROGATE_LIMITATION",
    "PHASE9F_SURROGATE_LIMITATION",
    "PHASE9F1_SURROGATE_LIMITATION",
    "PX4_GAZEBO_DEBUG_YAW_OVERRIDE_LABEL",
    "PX4_GAZEBO_FULL_AUTONOMY_LOOP_LABEL",
    "CompetitionMainConfig",
    "CompetitionMainSafetyError",
    "CompetitionMainSummary",
    "build_arg_parser",
    "fail_closed_summary",
    "main",
    "run_competition_main",
]
