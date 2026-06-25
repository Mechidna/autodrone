from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "runtime.toml"


@dataclass(frozen=True)
class RuntimeSection:
    runner_mode: str
    use_perception: bool
    control_hz: float
    flow_status_period_s: float
    join_timeout_s: float
    shutdown_land_wait_s: float
    px4_offboard_enabled: bool
    px4_offboard_prime_count: int
    px4_offboard_mode: str
    px4_arm: bool
    competition_arm: bool


@dataclass(frozen=True)
class MavlinkSection:
    ip: str
    port_px4: int
    port_competition: int
    target_system: int
    target_component: int

    def port_for_mode(self, runner_mode: str) -> int:
        if str(runner_mode).lower() == "competition":
            return int(self.port_competition)
        return int(self.port_px4)


@dataclass(frozen=True)
class TelemetrySection:
    prefer_odometry: bool
    require_local_position: bool
    max_state_age_s: float
    store_highres_imu: bool
    store_timesync: bool


@dataclass(frozen=True)
class StateEstimationSection:
    mode: str
    mavlink_truth_logging: bool
    init_position_source: str
    use_imu_prediction: bool
    use_vision_correction: bool
    vision_correction_source: str
    max_imu_dt_s: float
    max_state_age_s: float
    vision_correction_alpha: float
    vision_correction_max_residual_m: float
    vision_correction_min_confidence: float
    known_gate_positions_neu: tuple[tuple[float, float, float], ...]
    gravity_m_s2: float
    max_imu_accel_m_s2: float
    max_imu_velocity_m_s: float


@dataclass(frozen=True)
class TimesyncSection:
    request_hz: float


@dataclass(frozen=True)
class VisionSection:
    source: str
    udp_bind_ip: str
    udp_port: int
    udp_socket_timeout_s: float
    udp_recv_bytes: int
    packet_header_format: str
    max_pending_frames: int
    stale_frame_timeout_s: float
    max_jpeg_size_bytes: int
    ros_camera_topic: str
    ros_camera_info_topic: str


@dataclass(frozen=True)
class CameraSection:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: tuple[float, ...]

    @property
    def matrix(self) -> tuple[tuple[float, float, float], ...]:
        return (
            (self.fx, 0.0, self.cx),
            (0.0, self.fy, self.cy),
            (0.0, 0.0, 1.0),
        )


@dataclass(frozen=True)
class PerceptionSection:
    enabled: bool
    backend: str
    hz: float
    gate_size_m: float
    yolo_model_path: Optional[str]
    preprocess_mode: str
    yolo_conf: float
    yolo_imgsz: int
    yolo_device: Optional[int | str]
    transform_mode: str
    world_pose_source: str
    max_reprojection_error_for_memory: float
    reject_negative_depth: bool


@dataclass(frozen=True)
class GateMemorySection:
    association_radius: float
    commit_radius: float
    new_track_block_radius: float
    min_confidence_per_hit: float
    commit_hits: int
    commit_confidence_sum: float
    commit_spread_radius: float
    stale_time: float
    alpha: float
    use_lookahead_gate_filter: bool
    history_size: int
    min_hits_for_stable: int
    max_center_std_for_stable: float
    max_camera_std_for_stable: float
    max_reprojection_error_for_stable: float
    max_outlier_distance: float
    max_committed_match_distance: float
    min_observation_time: float


@dataclass(frozen=True)
class RaceSection:
    gate_count: Optional[int]
    pass_radius_m: float
    clear_radius_m: float
    debounce_s: float
    expected_gate_altitude_m: float
    active_target_lost_grace_s: float


@dataclass(frozen=True)
class PlannerSection:
    vmax: float
    amax: float
    t_min: float
    safe_min_target_z: float
    safe_max_target_z: float
    replan_target_shift_m: float
    replan_after_trajectory_s: float
    replan_min_interval_s: float
    max_detection_range_m: float
    target_z_mode: str


@dataclass(frozen=True)
class ControllerSection:
    mass: float
    gravity: float
    kp: tuple[float, float, float]
    kv: tuple[float, float, float]
    max_tilt_deg: float
    max_acc_xy: float
    max_acc_z_up: float
    max_acc_z_down: float
    thrust_hover: float
    thrust_min: float
    thrust_max: float
    fallback_thrust: float
    adaptive_hover_enabled: bool
    adaptive_hover_gain: float
    adaptive_hover_z_gain: float
    adaptive_hover_min: float
    adaptive_hover_max: float
    adaptive_hover_max_signal: float
    adaptive_hover_max_ref_vz: float
    adaptive_hover_max_ref_az: float
    adaptive_hover_max_z_error: float
    adaptive_hover_saturation_margin: float
    adaptive_hover_min_confidence: float
    max_yaw_rate_deg_s: float
    command_print_period_s: float


@dataclass(frozen=True)
class CommandSection:
    type: str
    stream_hz: float
    max_hz: float


@dataclass(frozen=True)
class HoverSection:
    stale_after_s: float
    takeoff_alt_m: float
    takeoff_window_s: float
    stationary_speed_m_s: float
    near_ground_abs_z_m: float
    post_race_hold_mode: str


@dataclass(frozen=True)
class PilotConfig:
    profile: str
    path: Path
    runtime: RuntimeSection
    mavlink: MavlinkSection
    telemetry: TelemetrySection
    state_estimation: StateEstimationSection
    timesync: TimesyncSection
    vision: VisionSection
    camera: CameraSection
    perception: PerceptionSection
    gate_memory: GateMemorySection
    race: RaceSection
    planner: PlannerSection
    controller: ControllerSection
    command: CommandSection
    hover: HoverSection


def load_runtime_config(path: str | os.PathLike[str] | None = None) -> PilotConfig:
    config_path = Path(path) if path is not None else CONFIG_PATH
    raw: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("rb") as f:
            raw = tomllib.load(f)

    runtime_raw = _section(raw, "runtime")
    mavlink_raw = _section(raw, "mavlink")
    telemetry_raw = _section(raw, "telemetry")
    state_estimation_raw = _section(raw, "state_estimation")
    timesync_raw = _section(raw, "timesync")
    vision_raw = _section(raw, "vision")
    camera_raw = _section(_section(raw, "camera"), "competition")
    perception_raw = _section(raw, "perception")
    gate_memory_raw = _section(raw, "gate_memory")
    race_raw = _section(raw, "race")
    planner_raw = _section(raw, "planner")
    controller_raw = _section(raw, "controller")
    command_raw = _section(raw, "command")
    hover_raw = _section(raw, "hover")

    runner_mode = _env_str("RUNNER_MODE", _str(runtime_raw, "runner_mode", "px4")).lower()
    control_hz = _float(runtime_raw, "control_hz", _float(command_raw, "stream_hz", 50.0))
    command_stream_hz = _float(command_raw, "stream_hz", control_hz)
    mavlink_port_override = _env_int("MAVLINK_PORT", None)

    port_px4 = _int(mavlink_raw, "port_px4", 14540)
    port_competition = _int(mavlink_raw, "port_competition", 14550)
    if mavlink_port_override is not None:
        if runner_mode == "competition":
            port_competition = mavlink_port_override
        else:
            port_px4 = mavlink_port_override

    config = PilotConfig(
        profile=_str(raw, "profile", "py_aipilot_example"),
        path=config_path,
        runtime=RuntimeSection(
            runner_mode=runner_mode,
            use_perception=_bool(runtime_raw, "use_perception", True),
            control_hz=control_hz,
            flow_status_period_s=_float(runtime_raw, "flow_status_period_s", 1.0),
            join_timeout_s=_float(runtime_raw, "join_timeout_s", 1.0),
            shutdown_land_wait_s=_float(runtime_raw, "shutdown_land_wait_s", 2.0),
            px4_offboard_enabled=_bool(runtime_raw, "px4_offboard_enabled", True),
            px4_offboard_prime_count=_int(runtime_raw, "px4_offboard_prime_count", 100),
            px4_offboard_mode=_str(runtime_raw, "px4_offboard_mode", "OFFBOARD"),
            px4_arm=_bool(runtime_raw, "px4_arm", True),
            competition_arm=_bool(runtime_raw, "competition_arm", True),
        ),
        mavlink=MavlinkSection(
            ip=_env_str("MAVLINK_IP", _str(mavlink_raw, "ip", "127.0.0.1")),
            port_px4=port_px4,
            port_competition=port_competition,
            target_system=_int(mavlink_raw, "target_system", 1),
            target_component=_int(mavlink_raw, "target_component", 1),
        ),
        telemetry=TelemetrySection(
            prefer_odometry=_bool(telemetry_raw, "prefer_odometry", False),
            require_local_position=_bool(telemetry_raw, "require_local_position", True),
            max_state_age_s=_float(telemetry_raw, "max_state_age_s", 0.5),
            store_highres_imu=_bool(telemetry_raw, "store_highres_imu", True),
            store_timesync=_bool(telemetry_raw, "store_timesync", True),
        ),
        state_estimation=StateEstimationSection(
            mode=_str(state_estimation_raw, "mode", "auto").lower(),
            mavlink_truth_logging=_bool(
                state_estimation_raw,
                "mavlink_truth_logging",
                True,
            ),
            init_position_source=_str(
                state_estimation_raw,
                "init_position_source",
                "zero",
            ).lower(),
            use_imu_prediction=_bool(state_estimation_raw, "use_imu_prediction", True),
            use_vision_correction=_bool(state_estimation_raw, "use_vision_correction", True),
            vision_correction_source=_str(
                state_estimation_raw,
                "vision_correction_source",
                "known_gates",
            ).lower(),
            max_imu_dt_s=_float(state_estimation_raw, "max_imu_dt_s", 0.05),
            max_state_age_s=_float(
                state_estimation_raw,
                "max_state_age_s",
                _float(telemetry_raw, "max_state_age_s", 0.5),
            ),
            vision_correction_alpha=_float(
                state_estimation_raw,
                "vision_correction_alpha",
                0.25,
            ),
            vision_correction_max_residual_m=_float(
                state_estimation_raw,
                "vision_correction_max_residual_m",
                3.0,
            ),
            vision_correction_min_confidence=_float(
                state_estimation_raw,
                "vision_correction_min_confidence",
                0.2,
            ),
            known_gate_positions_neu=tuple(
                _vec3_tuple(item)
                for item in (state_estimation_raw.get("known_gate_positions_neu", ()) or ())
            ),
            gravity_m_s2=_float(state_estimation_raw, "gravity_m_s2", 9.81),
            max_imu_accel_m_s2=_float(state_estimation_raw, "max_imu_accel_m_s2", 20.0),
            max_imu_velocity_m_s=_float(state_estimation_raw, "max_imu_velocity_m_s", 8.0),
        ),
        timesync=TimesyncSection(
            request_hz=_float(timesync_raw, "request_hz", 10.0),
        ),
        vision=VisionSection(
            source=_env_str("VISION_SOURCE", _str(vision_raw, "source", "udp")).lower(),
            udp_bind_ip=_str(vision_raw, "udp_bind_ip", "0.0.0.0"),
            udp_port=_int(vision_raw, "udp_port", 5600),
            udp_socket_timeout_s=_float(vision_raw, "udp_socket_timeout_s", 0.1),
            udp_recv_bytes=_int(vision_raw, "udp_recv_bytes", 65536),
            packet_header_format=_str(vision_raw, "packet_header_format", "<IHHIIQ"),
            max_pending_frames=_int(vision_raw, "max_pending_frames", 8),
            stale_frame_timeout_s=_float(vision_raw, "stale_frame_timeout_s", 0.5),
            max_jpeg_size_bytes=_int(vision_raw, "max_jpeg_size_bytes", 4_000_000),
            ros_camera_topic=_str(vision_raw, "ros_camera_topic", "/camera"),
            ros_camera_info_topic=_str(vision_raw, "ros_camera_info_topic", "/camera_info"),
        ),
        camera=CameraSection(
            width=_int(camera_raw, "width", 640),
            height=_int(camera_raw, "height", 360),
            fx=_float(camera_raw, "fx", 320.0),
            fy=_float(camera_raw, "fy", 320.0),
            cx=_float(camera_raw, "cx", 320.0),
            cy=_float(camera_raw, "cy", 180.0),
            dist_coeffs=tuple(_float_list(camera_raw.get("dist_coeffs"), [0.0] * 5)),
        ),
        perception=PerceptionSection(
            enabled=_bool(perception_raw, "enabled", True),
            backend=_env_str("PERCEPTION_BACKEND", _str(perception_raw, "backend", "blue")).lower(),
            hz=_env_float("PERCEPTION_HZ", _float(perception_raw, "hz", 30.0)),
            gate_size_m=_float(perception_raw, "gate_size_m", 1.5),
            yolo_model_path=_env_str_optional(
                "YOLO_MODEL_PATH",
                _str_optional(perception_raw, "yolo_model_path", None),
            ),
            preprocess_mode=_str(perception_raw, "preprocess_mode", "raw"),
            yolo_conf=_float(perception_raw, "yolo_conf", 0.1),
            yolo_imgsz=_int(perception_raw, "yolo_imgsz", 640),
            yolo_device=_optional_device(perception_raw.get("yolo_device")),
            transform_mode=_str(perception_raw, "transform_mode", "competition_official_ned"),
            world_pose_source=_str(perception_raw, "world_pose_source", "mavsdk"),
            max_reprojection_error_for_memory=_float(
                perception_raw,
                "max_reprojection_error_for_memory",
                20.0,
            ),
            reject_negative_depth=_bool(perception_raw, "reject_negative_depth", True),
        ),
        gate_memory=GateMemorySection(
            association_radius=_float(gate_memory_raw, "association_radius", 1.5),
            commit_radius=_float(gate_memory_raw, "commit_radius", 1.5),
            new_track_block_radius=_float(gate_memory_raw, "new_track_block_radius", 4.5),
            min_confidence_per_hit=_float(gate_memory_raw, "min_confidence_per_hit", 0.2),
            commit_hits=_int(gate_memory_raw, "commit_hits", 4),
            commit_confidence_sum=_float(gate_memory_raw, "commit_confidence_sum", 1.2),
            commit_spread_radius=_float(gate_memory_raw, "commit_spread_radius", 1.0),
            stale_time=_float(gate_memory_raw, "stale_time", 3.0),
            alpha=_float(gate_memory_raw, "alpha", 0.35),
            use_lookahead_gate_filter=_bool(gate_memory_raw, "use_lookahead_gate_filter", True),
            history_size=_int(gate_memory_raw, "history_size", 15),
            min_hits_for_stable=_int(gate_memory_raw, "min_hits_for_stable", 6),
            max_center_std_for_stable=_float(gate_memory_raw, "max_center_std_for_stable", 0.45),
            max_camera_std_for_stable=_float(gate_memory_raw, "max_camera_std_for_stable", 0.45),
            max_reprojection_error_for_stable=_float(
                gate_memory_raw,
                "max_reprojection_error_for_stable",
                5.0,
            ),
            max_outlier_distance=_float(gate_memory_raw, "max_outlier_distance", 0.75),
            max_committed_match_distance=_float(
                gate_memory_raw,
                "max_committed_match_distance",
                0.8,
            ),
            min_observation_time=_float(gate_memory_raw, "min_observation_time", 0.25),
        ),
        race=RaceSection(
            gate_count=_int_optional(
                race_raw,
                "gate_count",
                _int_optional(runtime_raw, "race_gate_count", 3),
            ),
            pass_radius_m=_float(race_raw, "pass_radius_m", _float(planner_raw, "pass_radius_m", 1.25)),
            clear_radius_m=_float(race_raw, "clear_radius_m", 1.75),
            debounce_s=_float(race_raw, "debounce_s", 0.75),
            expected_gate_altitude_m=_float(race_raw, "expected_gate_altitude_m", 1.5),
            active_target_lost_grace_s=_float(race_raw, "active_target_lost_grace_s", 2.0),
        ),
        planner=PlannerSection(
            vmax=_float(planner_raw, "vmax", 2.5),
            amax=_float(planner_raw, "amax", 2.0),
            t_min=_float(planner_raw, "t_min", 1.0),
            safe_min_target_z=_float(planner_raw, "safe_min_target_z", 1.0),
            safe_max_target_z=_float(planner_raw, "safe_max_target_z", 3.0),
            replan_target_shift_m=_float(planner_raw, "replan_target_shift_m", 1.0),
            replan_after_trajectory_s=_float(planner_raw, "replan_after_trajectory_s", 0.25),
            replan_min_interval_s=_float(planner_raw, "replan_min_interval_s", 0.3),
            max_detection_range_m=_float(planner_raw, "max_detection_range_m", 25.0),
            target_z_mode=_str(planner_raw, "target_z_mode", "observed_clamped"),
        ),
        controller=ControllerSection(
            mass=_float(controller_raw, "mass", 1.0),
            gravity=_float(controller_raw, "gravity", 9.81),
            kp=_float_tuple(controller_raw.get("kp"), (2.5, 2.5, 3.5), 3),
            kv=_float_tuple(controller_raw.get("kv"), (2.0, 2.0, 2.6), 3),
            max_tilt_deg=_float(controller_raw, "max_tilt_deg", 20.0),
            max_acc_xy=_float(controller_raw, "max_acc_xy", 2.0),
            max_acc_z_up=_float(controller_raw, "max_acc_z_up", 2.5),
            max_acc_z_down=_float(controller_raw, "max_acc_z_down", 2.0),
            thrust_hover=_float(
                controller_raw,
                "thrust_hover_initial",
                _float(controller_raw, "thrust_hover", 0.5),
            ),
            thrust_min=0.0,
            thrust_max=1.0,
            fallback_thrust=_float(
                controller_raw,
                "fallback_thrust_initial",
                _float(
                    controller_raw,
                    "fallback_thrust",
                    _float(
                        controller_raw,
                        "thrust_hover_initial",
                        _float(controller_raw, "thrust_hover", 0.5),
                    ),
                ),
            ),
            adaptive_hover_enabled=_bool(controller_raw, "adaptive_hover_enabled", True),
            adaptive_hover_gain=_float(controller_raw, "adaptive_hover_gain", 0.04),
            adaptive_hover_z_gain=_float(controller_raw, "adaptive_hover_z_gain", 0.30),
            adaptive_hover_min=_float(controller_raw, "adaptive_hover_min", 0.0),
            adaptive_hover_max=_float(controller_raw, "adaptive_hover_max", 1.0),
            adaptive_hover_max_signal=_float(controller_raw, "adaptive_hover_max_signal", 1.0),
            adaptive_hover_max_ref_vz=_float(controller_raw, "adaptive_hover_max_ref_vz", 1.0),
            adaptive_hover_max_ref_az=_float(controller_raw, "adaptive_hover_max_ref_az", 1.5),
            adaptive_hover_max_z_error=_float(controller_raw, "adaptive_hover_max_z_error", 2.0),
            adaptive_hover_saturation_margin=_float(
                controller_raw,
                "adaptive_hover_saturation_margin",
                0.03,
            ),
            adaptive_hover_min_confidence=_float(
                controller_raw,
                "adaptive_hover_min_confidence",
                0.0,
            ),
            max_yaw_rate_deg_s=_float(controller_raw, "max_yaw_rate_deg_s", 90.0),
            command_print_period_s=_float(controller_raw, "command_print_period_s", 1.0),
        ),
        command=CommandSection(
            type=_str(command_raw, "type", "set_attitude_target"),
            stream_hz=command_stream_hz,
            max_hz=_float(command_raw, "max_hz", 100.0),
        ),
        hover=HoverSection(
            stale_after_s=_float(hover_raw, "stale_after_s", 1.0),
            takeoff_alt_m=_float(hover_raw, "takeoff_alt_m", 1.5),
            takeoff_window_s=_float(hover_raw, "takeoff_window_s", 10.0),
            stationary_speed_m_s=_float(hover_raw, "stationary_speed_m_s", 0.05),
            near_ground_abs_z_m=_float(hover_raw, "near_ground_abs_z_m", 0.25),
            post_race_hold_mode=_str(hover_raw, "post_race_hold_mode", "position_hold"),
        ),
    )
    _validate(config)
    return config


def _validate(config: PilotConfig) -> None:
    if config.runtime.runner_mode not in ("px4", "competition"):
        raise RuntimeError(
            f"Invalid runner_mode={config.runtime.runner_mode!r}. "
            "Use runner_mode='px4' or runner_mode='competition'."
        )
    if config.vision.source not in ("udp", "ros"):
        raise RuntimeError(
            f"Invalid vision.source={config.vision.source!r}. "
            "Use vision.source='udp' or vision.source='ros'."
        )
    if config.state_estimation.mode not in ("mavlink", "auto", "estimator"):
        raise RuntimeError(
            f"Invalid state_estimation.mode={config.state_estimation.mode!r}. "
            "Use 'mavlink', 'auto', or 'estimator'."
        )
    if config.state_estimation.init_position_source not in ("zero", "mavlink_once"):
        raise RuntimeError(
            "state_estimation.init_position_source must be 'zero' or 'mavlink_once'."
        )
    if config.state_estimation.vision_correction_source not in (
        "none",
        "known_gates",
        "stable_tracks",
        "known_gates_or_stable_tracks",
    ):
        raise RuntimeError(
            "state_estimation.vision_correction_source must be 'none', "
            "'known_gates', 'stable_tracks', or 'known_gates_or_stable_tracks'."
        )
    if config.command.stream_hz <= 0.0:
        raise RuntimeError("command.stream_hz must be positive.")
    if config.command.max_hz <= 0.0:
        raise RuntimeError("command.max_hz must be positive.")
    if config.command.stream_hz >= config.command.max_hz:
        raise RuntimeError(
            f"command.stream_hz={config.command.stream_hz} must be below "
            f"command.max_hz={config.command.max_hz}."
        )
    if config.runtime.control_hz <= 0.0:
        raise RuntimeError("runtime.control_hz must be positive.")
    if config.runtime.control_hz >= config.command.max_hz:
        raise RuntimeError(
            f"runtime.control_hz={config.runtime.control_hz} must be below "
            f"command.max_hz={config.command.max_hz}."
        )
    if not 0.0 <= config.controller.thrust_min <= config.controller.thrust_max <= 1.0:
        raise RuntimeError("controller thrust_min/thrust_max must stay within [0.0, 1.0].")
    if not 0.0 <= config.controller.thrust_hover <= 1.0:
        raise RuntimeError("controller thrust_hover_initial must stay within [0.0, 1.0].")
    if not 0.0 <= config.controller.fallback_thrust <= 1.0:
        raise RuntimeError("controller fallback_thrust must stay within [0.0, 1.0].")
    if not 0.0 <= config.controller.adaptive_hover_min <= config.controller.adaptive_hover_max <= 1.0:
        raise RuntimeError(
            "controller adaptive_hover_min/adaptive_hover_max must stay within [0.0, 1.0]."
        )


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    return value if isinstance(value, dict) else {}


def _str(raw: dict[str, Any], key: str, default: str) -> str:
    value = raw.get(key, default)
    return str(default if value is None else value)


def _str_optional(raw: dict[str, Any], key: str, default: Optional[str]) -> Optional[str]:
    value = raw.get(key, default)
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _int(raw: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(raw.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _int_optional(raw: dict[str, Any], key: str, default: Optional[int]) -> Optional[int]:
    value = raw.get(key, default)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(raw: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(raw.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _bool(raw: dict[str, Any], key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "on")
    return bool(value)


def _float_list(value: Any, default: list[float]) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return [float(item) for item in default]
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return [float(item) for item in default]


def _float_tuple(value: Any, default: tuple[float, ...], length: int) -> tuple[float, ...]:
    values = _float_list(value, list(default))
    if len(values) != length:
        values = list(default)
    return tuple(float(item) for item in values)


def _vec3_tuple(value: Any) -> tuple[float, float, float]:
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        values = ()
    if len(values) != 3:
        raise RuntimeError("known_gate_positions_neu entries must contain exactly 3 numbers.")
    return values


def _optional_device(value: Any) -> Optional[int | str]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value)
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return text


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _env_str_optional(name: str, default: Optional[str]) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return default
    return value if value else None


def _env_int(name: str, default: Optional[int]) -> Optional[int]:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)
