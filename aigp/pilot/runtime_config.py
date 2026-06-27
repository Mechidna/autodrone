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
    allow_known_gate_correction: bool
    max_imu_dt_s: float
    max_state_age_s: float
    vision_correction_alpha: float
    vision_correction_alpha_xy: float
    vision_correction_alpha_z: float
    vision_correction_max_delta_m: float
    vision_correction_max_residual_m: float
    vision_correction_min_confidence: float
    estimator_landmark_min_hits: int
    estimator_landmark_min_observation_time_s: float
    estimator_landmark_max_center_std_m: float
    estimator_landmark_max_camera_std_m: float
    estimator_landmark_max_reprojection_error: float
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
    body_translation_m: tuple[float, float, float]
    mount_profile: str
    yaw_correction_deg: float

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
    min_depth_m_for_memory: float
    max_depth_m_for_memory: float
    reject_negative_depth: bool


@dataclass(frozen=True)
class PerceptionGeometryAuditSection:
    enabled: bool
    print_period_s: float
    max_prints: int
    max_match_distance_m: float
    known_gate_positions_neu: tuple[tuple[float, float, float], ...]
    gate_right_axis_neu: tuple[float, float, float]
    gate_up_axis_neu: tuple[float, float, float]


@dataclass(frozen=True)
class GateSourceSection:
    mode: str
    allow_ground_truth: bool
    known_gate_positions_neu: tuple[tuple[float, float, float], ...]


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
    adaptive_hover_fast_enabled: bool
    adaptive_hover_fast_gain: float
    adaptive_hover_fast_min_z_error: float
    adaptive_hover_fast_stable_signal: float
    adaptive_hover_fast_stable_z_error: float
    adaptive_hover_fast_stable_samples: int
    adaptive_hover_fast_decay_s: float
    max_yaw_rate_deg_s: float
    command_print_period_s: float


@dataclass(frozen=True)
class HoverAcquisitionSection:
    enabled: bool
    estimator_mode_only: bool
    require_armed: bool
    initial_thrust: float
    min_thrust: float
    max_probe_thrust: float
    thrust_step_per_s: float
    thrust_trim_step_per_s: float
    velocity_gain: float
    accel_gain: float
    accel_deadband_m_s2: float
    target_vz_m_s: float
    max_up_vz_m_s: float
    max_relative_z_m: float
    max_settle_vz_m_s: float
    min_duration_s: float
    max_duration_s: float
    stable_duration_s: float
    stable_vz_abs_m_s: float
    stable_accel_abs_m_s2: float
    lift_confirm_z_m: float
    lift_confirm_vz_m_s: float
    relative_airborne_z_m: float
    min_confidence: float
    overshoot_thrust_step_per_s: float
    reset_hover_on_disarm: bool
    release_on_timeout_while_unstable: bool
    print_period_s: float


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
    perception_geometry_audit: PerceptionGeometryAuditSection
    gate_source: GateSourceSection
    gate_memory: GateMemorySection
    race: RaceSection
    planner: PlannerSection
    controller: ControllerSection
    hover_acquisition: HoverAcquisitionSection
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
    camera_parent_raw = _section(raw, "camera")
    camera_raw = _section(camera_parent_raw, "competition")
    camera_mount_raw = _section(camera_parent_raw, "mount")
    perception_raw = _section(raw, "perception")
    perception_geometry_audit_raw = _section(raw, "perception_geometry_audit")
    gate_source_raw = _section(raw, "gate_source")
    gate_memory_raw = _section(raw, "gate_memory")
    race_raw = _section(raw, "race")
    planner_raw = _section(raw, "planner")
    controller_raw = _section(raw, "controller")
    hover_acquisition_raw = _section(raw, "hover_acquisition")
    command_raw = _section(raw, "command")
    hover_raw = _section(raw, "hover")

    runner_mode = _env_str("RUNNER_MODE", _str(runtime_raw, "runner_mode", "px4")).lower()
    control_hz = _float(runtime_raw, "control_hz", _float(command_raw, "stream_hz", 50.0))
    command_stream_hz = _float(command_raw, "stream_hz", control_hz)
    mavlink_port_override = _env_int("MAVLINK_PORT", None)
    camera_mount_profile, camera_body_translation_m, camera_yaw_correction_deg = _resolve_camera_mount(
        _env_str(
            "CAMERA_MOUNT_PROFILE",
            _str(camera_mount_raw, "profile", _str(camera_raw, "mount_profile", "auto")),
        ),
        runner_mode=runner_mode,
        competition_body_translation_m=_float_tuple(
            camera_mount_raw.get("competition_body_translation_m"),
            (0.0, 0.0, 0.0),
            3,
        ),
        competition_yaw_correction_deg=_float(
            camera_mount_raw,
            "competition_yaw_correction_deg",
            0.0,
        ),
        px4_x500_body_translation_m=_float_tuple(
            camera_mount_raw.get("px4_x500_mono_cam_body_translation_m"),
            (0.12, -0.03, -0.242),
            3,
        ),
        px4_x500_yaw_correction_deg=_float(
            camera_mount_raw,
            "px4_x500_mono_cam_yaw_correction_deg",
            0.0,
        ),
        custom_body_translation_m=_float_tuple(
            camera_mount_raw.get("custom_body_translation_m"),
            _float_tuple(camera_raw.get("body_translation_m"), (0.0, 0.0, 0.0), 3),
            3,
        ),
        custom_yaw_correction_deg=_float(
            camera_mount_raw,
            "custom_yaw_correction_deg",
            _float(camera_raw, "yaw_correction_deg", 0.0),
        ),
    )

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
                "stable_tracks",
            ).lower(),
            allow_known_gate_correction=_bool(
                state_estimation_raw,
                "allow_known_gate_correction",
                False,
            ),
            max_imu_dt_s=_float(state_estimation_raw, "max_imu_dt_s", 0.05),
            max_state_age_s=_float(
                state_estimation_raw,
                "max_state_age_s",
                _float(telemetry_raw, "max_state_age_s", 0.5),
            ),
            vision_correction_alpha=_float(
                state_estimation_raw,
                "vision_correction_alpha",
                0.10,
            ),
            vision_correction_alpha_xy=_float(
                state_estimation_raw,
                "vision_correction_alpha_xy",
                _float(state_estimation_raw, "vision_correction_alpha", 0.10),
            ),
            vision_correction_alpha_z=_float(
                state_estimation_raw,
                "vision_correction_alpha_z",
                0.0,
            ),
            vision_correction_max_delta_m=_float(
                state_estimation_raw,
                "vision_correction_max_delta_m",
                0.5,
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
            estimator_landmark_min_hits=_int(
                state_estimation_raw,
                "estimator_landmark_min_hits",
                12,
            ),
            estimator_landmark_min_observation_time_s=_float(
                state_estimation_raw,
                "estimator_landmark_min_observation_time_s",
                0.75,
            ),
            estimator_landmark_max_center_std_m=_float(
                state_estimation_raw,
                "estimator_landmark_max_center_std_m",
                0.25,
            ),
            estimator_landmark_max_camera_std_m=_float(
                state_estimation_raw,
                "estimator_landmark_max_camera_std_m",
                0.25,
            ),
            estimator_landmark_max_reprojection_error=_float(
                state_estimation_raw,
                "estimator_landmark_max_reprojection_error",
                2.5,
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
            body_translation_m=camera_body_translation_m,
            mount_profile=camera_mount_profile,
            yaw_correction_deg=camera_yaw_correction_deg,
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
                5.0,
            ),
            min_depth_m_for_memory=_float(
                perception_raw,
                "min_depth_m_for_memory",
                0.5,
            ),
            max_depth_m_for_memory=_float(
                perception_raw,
                "max_depth_m_for_memory",
                _float(planner_raw, "max_detection_range_m", 25.0),
            ),
            reject_negative_depth=_bool(perception_raw, "reject_negative_depth", True),
        ),
        perception_geometry_audit=PerceptionGeometryAuditSection(
            enabled=_bool(perception_geometry_audit_raw, "enabled", False),
            print_period_s=_float(perception_geometry_audit_raw, "print_period_s", 0.5),
            max_prints=_int(perception_geometry_audit_raw, "max_prints", 80),
            max_match_distance_m=_float(
                perception_geometry_audit_raw,
                "max_match_distance_m",
                5.0,
            ),
            known_gate_positions_neu=tuple(
                _vec3_tuple(item)
                for item in (
                    perception_geometry_audit_raw.get("known_gate_positions_neu", ())
                    or ()
                )
            ),
            gate_right_axis_neu=_vec3_tuple(
                perception_geometry_audit_raw.get("gate_right_axis_neu", (1.0, 0.0, 0.0))
            ),
            gate_up_axis_neu=_vec3_tuple(
                perception_geometry_audit_raw.get("gate_up_axis_neu", (0.0, 0.0, 1.0))
            ),
        ),
        gate_source=GateSourceSection(
            mode=_env_str(
                "GATE_SOURCE_MODE",
                _str(gate_source_raw, "mode", "perception"),
            ).lower(),
            allow_ground_truth=_bool(gate_source_raw, "allow_ground_truth", False),
            known_gate_positions_neu=tuple(
                _vec3_tuple(item)
                for item in (
                    gate_source_raw.get("known_gate_positions_neu", ())
                    or ()
                )
            ),
        ),
        gate_memory=GateMemorySection(
            association_radius=_float(gate_memory_raw, "association_radius", 1.5),
            commit_radius=_float(gate_memory_raw, "commit_radius", 1.5),
            new_track_block_radius=_float(gate_memory_raw, "new_track_block_radius", 4.5),
            min_confidence_per_hit=_float(gate_memory_raw, "min_confidence_per_hit", 0.2),
            commit_hits=_int(gate_memory_raw, "commit_hits", 6),
            commit_confidence_sum=_float(gate_memory_raw, "commit_confidence_sum", 1.8),
            commit_spread_radius=_float(gate_memory_raw, "commit_spread_radius", 0.60),
            stale_time=_float(gate_memory_raw, "stale_time", 3.0),
            alpha=_float(gate_memory_raw, "alpha", 0.35),
            use_lookahead_gate_filter=_bool(gate_memory_raw, "use_lookahead_gate_filter", True),
            history_size=_int(gate_memory_raw, "history_size", 15),
            min_hits_for_stable=_int(gate_memory_raw, "min_hits_for_stable", 10),
            max_center_std_for_stable=_float(gate_memory_raw, "max_center_std_for_stable", 0.30),
            max_camera_std_for_stable=_float(gate_memory_raw, "max_camera_std_for_stable", 0.30),
            max_reprojection_error_for_stable=_float(
                gate_memory_raw,
                "max_reprojection_error_for_stable",
                3.0,
            ),
            max_outlier_distance=_float(gate_memory_raw, "max_outlier_distance", 0.60),
            max_committed_match_distance=_float(
                gate_memory_raw,
                "max_committed_match_distance",
                0.60,
            ),
            min_observation_time=_float(gate_memory_raw, "min_observation_time", 0.50),
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
            adaptive_hover_fast_enabled=_bool(
                controller_raw,
                "adaptive_hover_fast_enabled",
                True,
            ),
            adaptive_hover_fast_gain=_float(
                controller_raw,
                "adaptive_hover_fast_gain",
                0.25,
            ),
            adaptive_hover_fast_min_z_error=_float(
                controller_raw,
                "adaptive_hover_fast_min_z_error",
                0.05,
            ),
            adaptive_hover_fast_stable_signal=_float(
                controller_raw,
                "adaptive_hover_fast_stable_signal",
                0.08,
            ),
            adaptive_hover_fast_stable_z_error=_float(
                controller_raw,
                "adaptive_hover_fast_stable_z_error",
                0.15,
            ),
            adaptive_hover_fast_stable_samples=_int(
                controller_raw,
                "adaptive_hover_fast_stable_samples",
                20,
            ),
            adaptive_hover_fast_decay_s=_float(
                controller_raw,
                "adaptive_hover_fast_decay_s",
                3.0,
            ),
            max_yaw_rate_deg_s=_float(controller_raw, "max_yaw_rate_deg_s", 90.0),
            command_print_period_s=_float(controller_raw, "command_print_period_s", 1.0),
        ),
        hover_acquisition=HoverAcquisitionSection(
            enabled=_bool(hover_acquisition_raw, "enabled", True),
            estimator_mode_only=_bool(
                hover_acquisition_raw,
                "estimator_mode_only",
                True,
            ),
            require_armed=_bool(hover_acquisition_raw, "require_armed", True),
            initial_thrust=_float(
                hover_acquisition_raw,
                "initial_thrust",
                _float(controller_raw, "thrust_hover_initial", 0.5),
            ),
            min_thrust=_float(hover_acquisition_raw, "min_thrust", 0.0),
            max_probe_thrust=_float(hover_acquisition_raw, "max_probe_thrust", 0.85),
            thrust_step_per_s=_float(
                hover_acquisition_raw,
                "thrust_step_per_s",
                0.25,
            ),
            thrust_trim_step_per_s=_float(
                hover_acquisition_raw,
                "thrust_trim_step_per_s",
                0.12,
            ),
            velocity_gain=_float(hover_acquisition_raw, "velocity_gain", 0.35),
            accel_gain=_float(hover_acquisition_raw, "accel_gain", 0.05),
            accel_deadband_m_s2=_float(
                hover_acquisition_raw,
                "accel_deadband_m_s2",
                0.30,
            ),
            target_vz_m_s=_float(hover_acquisition_raw, "target_vz_m_s", 0.10),
            max_up_vz_m_s=_float(hover_acquisition_raw, "max_up_vz_m_s", 0.80),
            max_relative_z_m=_float(hover_acquisition_raw, "max_relative_z_m", 2.0),
            max_settle_vz_m_s=_float(hover_acquisition_raw, "max_settle_vz_m_s", 0.60),
            min_duration_s=_float(hover_acquisition_raw, "min_duration_s", 0.75),
            max_duration_s=_float(hover_acquisition_raw, "max_duration_s", 4.0),
            stable_duration_s=_float(
                hover_acquisition_raw,
                "stable_duration_s",
                0.50,
            ),
            stable_vz_abs_m_s=_float(
                hover_acquisition_raw,
                "stable_vz_abs_m_s",
                0.25,
            ),
            stable_accel_abs_m_s2=_float(
                hover_acquisition_raw,
                "stable_accel_abs_m_s2",
                0.80,
            ),
            lift_confirm_z_m=_float(hover_acquisition_raw, "lift_confirm_z_m", 0.15),
            lift_confirm_vz_m_s=_float(
                hover_acquisition_raw,
                "lift_confirm_vz_m_s",
                0.15,
            ),
            relative_airborne_z_m=_float(
                hover_acquisition_raw,
                "relative_airborne_z_m",
                _float(hover_acquisition_raw, "airborne_z_m", 0.25),
            ),
            min_confidence=_float(hover_acquisition_raw, "min_confidence", 0.0),
            overshoot_thrust_step_per_s=_float(
                hover_acquisition_raw,
                "overshoot_thrust_step_per_s",
                0.60,
            ),
            reset_hover_on_disarm=_bool(
                hover_acquisition_raw,
                "reset_hover_on_disarm",
                True,
            ),
            release_on_timeout_while_unstable=_bool(
                hover_acquisition_raw,
                "release_on_timeout_while_unstable",
                False,
            ),
            print_period_s=_float(hover_acquisition_raw, "print_period_s", 0.5),
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
    if config.camera.mount_profile not in ("competition", "px4_x500_mono_cam", "custom"):
        raise RuntimeError(
            f"Invalid camera mount profile={config.camera.mount_profile!r}. "
            "Use 'auto', 'competition', 'px4_x500_mono_cam', or 'custom'."
        )
    if (
        config.runtime.runner_mode == "competition"
        and any(abs(float(item)) > 1e-9 for item in config.camera.body_translation_m)
    ):
        raise RuntimeError(
            "camera.body_translation_m must resolve to [0, 0, 0] in "
            "runner_mode='competition'. VADR-TS-002 defines the camera and body "
            "frames as the same origin; PX4 validation camera offsets must stay "
            "in runner_mode='px4'."
        )
    if (
        config.runtime.runner_mode == "competition"
        and abs(float(config.camera.yaw_correction_deg)) > 1e-9
    ):
        raise RuntimeError(
            "camera.yaw_correction_deg must resolve to 0.0 in "
            "runner_mode='competition'. VADR-TS-002 defines the official camera/body "
            "transform; PX4 validation yaw corrections must stay in runner_mode='px4'."
        )
    if config.vision.source not in ("udp", "ros"):
        raise RuntimeError(
            f"Invalid vision.source={config.vision.source!r}. "
            "Use vision.source='udp' or vision.source='ros'."
        )
    if config.runtime.runner_mode == "competition" and config.perception_geometry_audit.enabled:
        raise RuntimeError(
            "perception_geometry_audit is debug-only and may use known sim gate "
            "positions. Disable it before running runner_mode='competition'."
        )
    if config.gate_source.mode not in ("perception", "ground_truth"):
        raise RuntimeError(
            f"Invalid gate_source.mode={config.gate_source.mode!r}. "
            "Use 'perception' or 'ground_truth'."
        )
    if config.gate_source.mode == "ground_truth":
        if config.runtime.runner_mode == "competition":
            raise RuntimeError(
                "gate_source.mode='ground_truth' is debug-only and not competition-valid. "
                "VADR-TS-002 does not provide fixed gate coordinates to contestant "
                "software."
            )
        if not config.gate_source.allow_ground_truth:
            raise RuntimeError(
                "gate_source.mode='ground_truth' requires "
                "gate_source.allow_ground_truth=true to make the sim-truth dependency "
                "explicit."
            )
        if not config.gate_source.known_gate_positions_neu:
            raise RuntimeError(
                "gate_source.known_gate_positions_neu must contain at least one gate "
                "when gate_source.mode='ground_truth'."
            )
        if (
            config.race.gate_count is not None
            and len(config.gate_source.known_gate_positions_neu) < config.race.gate_count
        ):
            raise RuntimeError(
                "gate_source.known_gate_positions_neu must contain at least "
                "race.gate_count entries when gate_source.mode='ground_truth'."
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
    source_uses_known_gates = config.state_estimation.vision_correction_source in (
        "known_gates",
        "known_gates_or_stable_tracks",
    )
    known_gate_positions_configured = bool(
        config.state_estimation.known_gate_positions_neu
    )
    known_gate_correction_active = (
        config.state_estimation.mode == "estimator"
        and config.state_estimation.use_vision_correction
        and (source_uses_known_gates or known_gate_positions_configured)
    )
    if (
        known_gate_correction_active
        and not config.state_estimation.allow_known_gate_correction
    ):
        raise RuntimeError(
            "Known gate correction is disabled for the estimator path because "
            "gate coordinates/altitude are not provided by VADR-TS-002. Use "
            "vision_correction_source='stable_tracks' with "
            "known_gate_positions_neu=[] for competition-style validation, or "
            "set state_estimation.allow_known_gate_correction=true only for "
            "explicit sim-truth debugging."
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
    for key, value in (
        ("vision_correction_alpha", config.state_estimation.vision_correction_alpha),
        ("vision_correction_alpha_xy", config.state_estimation.vision_correction_alpha_xy),
        ("vision_correction_alpha_z", config.state_estimation.vision_correction_alpha_z),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise RuntimeError(f"state_estimation.{key} must stay within [0.0, 1.0].")
    if config.state_estimation.vision_correction_max_delta_m < 0.0:
        raise RuntimeError("state_estimation.vision_correction_max_delta_m must be non-negative.")
    if config.state_estimation.vision_correction_max_residual_m < 0.0:
        raise RuntimeError("state_estimation.vision_correction_max_residual_m must be non-negative.")
    if config.state_estimation.estimator_landmark_min_hits < 1:
        raise RuntimeError("state_estimation.estimator_landmark_min_hits must be at least 1.")
    for key, value in (
        (
            "estimator_landmark_min_observation_time_s",
            config.state_estimation.estimator_landmark_min_observation_time_s,
        ),
        (
            "estimator_landmark_max_center_std_m",
            config.state_estimation.estimator_landmark_max_center_std_m,
        ),
        (
            "estimator_landmark_max_camera_std_m",
            config.state_estimation.estimator_landmark_max_camera_std_m,
        ),
        (
            "estimator_landmark_max_reprojection_error",
            config.state_estimation.estimator_landmark_max_reprojection_error,
        ),
        ("perception.min_depth_m_for_memory", config.perception.min_depth_m_for_memory),
        ("perception.max_depth_m_for_memory", config.perception.max_depth_m_for_memory),
        (
            "perception_geometry_audit.print_period_s",
            config.perception_geometry_audit.print_period_s,
        ),
        (
            "perception_geometry_audit.max_match_distance_m",
            config.perception_geometry_audit.max_match_distance_m,
        ),
    ):
        if float(value) < 0.0:
            raise RuntimeError(f"{key} must be non-negative.")
    if config.perception_geometry_audit.max_prints < 0:
        raise RuntimeError("perception_geometry_audit.max_prints must be non-negative.")
    if (
        config.perception.max_depth_m_for_memory > 0.0
        and config.perception.min_depth_m_for_memory > config.perception.max_depth_m_for_memory
    ):
        raise RuntimeError(
            "perception.min_depth_m_for_memory must be <= max_depth_m_for_memory."
        )
    if not 0.0 <= config.controller.adaptive_hover_min <= config.controller.adaptive_hover_max <= 1.0:
        raise RuntimeError(
            "controller adaptive_hover_min/adaptive_hover_max must stay within [0.0, 1.0]."
        )
    if not 0.0 <= config.hover_acquisition.min_thrust <= config.hover_acquisition.max_probe_thrust <= 1.0:
        raise RuntimeError(
            "hover_acquisition min_thrust/max_probe_thrust must stay within [0.0, 1.0]."
        )
    if not 0.0 <= config.hover_acquisition.initial_thrust <= 1.0:
        raise RuntimeError("hover_acquisition initial_thrust must stay within [0.0, 1.0].")
    if config.hover_acquisition.max_duration_s <= 0.0:
        raise RuntimeError("hover_acquisition max_duration_s must be positive.")
    for key, value in (
        ("max_relative_z_m", config.hover_acquisition.max_relative_z_m),
        ("max_settle_vz_m_s", config.hover_acquisition.max_settle_vz_m_s),
        (
            "overshoot_thrust_step_per_s",
            config.hover_acquisition.overshoot_thrust_step_per_s,
        ),
    ):
        if float(value) < 0.0:
            raise RuntimeError(f"hover_acquisition.{key} must be non-negative.")


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    return value if isinstance(value, dict) else {}


def _resolve_camera_mount(
    profile: str,
    *,
    runner_mode: str,
    competition_body_translation_m: tuple[float, float, float],
    competition_yaw_correction_deg: float,
    px4_x500_body_translation_m: tuple[float, float, float],
    px4_x500_yaw_correction_deg: float,
    custom_body_translation_m: tuple[float, float, float],
    custom_yaw_correction_deg: float,
) -> tuple[str, tuple[float, float, float], float]:
    requested = str(profile or "auto").strip().lower()
    if requested == "auto":
        requested = "competition" if str(runner_mode).lower() == "competition" else "px4_x500_mono_cam"

    if requested == "competition":
        return "competition", competition_body_translation_m, float(competition_yaw_correction_deg)
    if requested == "px4_x500_mono_cam":
        return "px4_x500_mono_cam", px4_x500_body_translation_m, float(px4_x500_yaw_correction_deg)
    if requested == "custom":
        return "custom", custom_body_translation_m, float(custom_yaw_correction_deg)
    return requested, custom_body_translation_m, float(custom_yaw_correction_deg)


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
