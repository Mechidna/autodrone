"""Configuration scaffolding for future runtime profiles.

These dataclasses mirror current AutonomyAPI defaults but are not yet the
authoritative source of runtime values. Wire them into runtime code only in a
later behavior-preserving phase with explicit default parity checks.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from autonomy_core.core.competition_config import VADR_TS_002


Vector3 = Tuple[float, float, float]
GateList = Tuple[Vector3, ...]


@dataclass(frozen=True)
class VehicleLimitsConfig:
    """Vehicle/controller limits copied from the current tracker setup."""

    mass: float = 1.0
    gravity: float = 9.81
    max_tilt_deg: float = 20.0
    max_acc_xy: float = 2.0
    max_acc_z_up: float = 2.5
    max_acc_z_down: float = 2.0
    thrust_hover: float = 0.74
    thrust_min: float = 0.60
    thrust_max: float = 0.85


@dataclass(frozen=True)
class ControllerConfig:
    """Controller gains and command limits copied from AutonomyAPI defaults."""

    kp: Vector3 = (2.5, 2.5, 3.5)
    kv: Vector3 = (2.0, 2.0, 2.6)
    max_yaw_rate_deg_s: float = 90.0
    no_target_grace_s: float = 0.75
    limits: VehicleLimitsConfig = field(default_factory=VehicleLimitsConfig)


@dataclass(frozen=True)
class PlannerConfig:
    """Planner and horizon defaults copied from the current hard-coded values."""

    reference_progress_tau_lead_s: float = 0.25
    reference_projection_sample_count: int = 100
    continue_previous_trajectory_during_replan: bool = True
    max_trajectory_holdover_s: float = 0.75
    hover_on_replan_without_valid_trajectory: bool = True
    command_stale_safety_threshold_s: float = 0.5
    installed_plan_sample_count: int = 160
    safe_min_target_z: float = 1.0
    safe_max_target_z: float = 3.0
    max_detection_range: float = 25.0
    max_gate_jump: float = 12.0
    completed_gate_position_radius: float = 1.5
    use_passthrough_gate_velocities: bool = False
    pass_through_speed: float = 3.0
    use_planning_lookahead_tracks: bool = True
    use_raw_rejected_planning_lookahead: bool = False
    planning_lookahead_min_hits: int = 6
    use_tentative_lookahead_spline: bool = True
    lookahead_min_hits: int = 3
    lookahead_min_confidence_sum: float = 0.8
    lookahead_max_reprojection_error: float = 8.0
    lookahead_max_distance: float = 25.0
    use_terminal_passthrough_extension: bool = True
    terminal_passthrough_extension_distance: float = 4.0
    suppress_minor_tentative_lookahead_replans: bool = True
    tentative_lookahead_replan_min_shift: float = 0.75
    tentative_lookahead_shift_replan_threshold: float = 0.5
    tentative_lookahead_replan_min_interval_s: float = 0.5
    raw_planning_lookahead_ttl_s: float = 1.25
    default_gt_gates: GateList = (
        (0.0, 8.0, 1.5),
        (0.8, 16.0, 1.5),
        (-0.8, 24.0, 1.5),
        (0.8, 16.0, 1.5),
        (0.0, 8.0, 1.5),
        (0.0, 0.0, 1.5),
    )


@dataclass(frozen=True)
class PerceptionConfig:
    """Perception, transform, and GateMemory defaults copied for future wiring."""

    camera_offset_body: Vector3 = (0.12, 0.03, 0.242)
    gate_size: float = VADR_TS_002.gate_inner_square_m
    yolo_model_path: str = (
        "/home/paolo/datasets/gazebo_gate_yolo_pose_ab_runs/partial/weights/best.pt"
    )
    preprocess_mode: str = "raw"
    yolo_conf: float = 0.1
    yolo_imgsz: int = 640
    yolo_device: int = 0
    association_radius: float = 1.5
    commit_radius: float = 1.5
    new_track_block_radius: float = 4.5
    min_confidence_per_hit: float = 0.2
    commit_hits: int = 4
    commit_confidence_sum: float = 1.2
    stale_time: float = 3.0
    alpha: float = 0.35
    use_lookahead_gate_filter: bool = True
    min_hits_for_stable: int = 6
    max_center_std_for_stable: float = 0.45
    max_camera_std_for_stable: float = 0.45
    max_reprojection_error_for_stable: float = 5.0
    perception_transform_mode: str = "physical_direct_rad_x_mirror"
    perception_world_pose_source: str = "gazebo_truth_sim_only"
    perception_world_pose_sources: Tuple[str, ...] = (
        "mavsdk",
        "gazebo_truth_sim_only",
    )
    perception_transform_modes: Tuple[str, ...] = (
        "legacy_scaled_yaw",
        "direct_rad",
        "physical_direct_rad",
        "physical_direct_rad_x_mirror",
        "competition_official_ned",
        "yaw_minus_pi_over_2",
        "pi_over_2_minus_yaw",
        "neg_yaw",
        "neg_yaw_plus_pi_over_2",
        "physical_mavsdk_yaw_aligned",
    )
    use_diagnostic_far_depth_correction: bool = False
    max_plausible_gate_speed: float = 12.0
    gate_jump_margin: float = 12.0


@dataclass(frozen=True)
class LoggingConfig:
    """Logging-mode scaffold only; FlightLogger behavior is unchanged."""

    mode: str = "debug_full"
    available_modes: Tuple[str, ...] = (
        "debug_full",
        "debug_light",
        "race",
        "off",
    )
    save_perception_debug_frames: bool = True
    debug_verbose_overlay: bool = False
    sidecar_plan_logging_enabled: bool = True
    preserve_current_schema: bool = True


@dataclass(frozen=True)
class RuntimeConfig:
    """Top-level scaffold for future competition/runtime profile selection."""

    use_perception: bool = False
    race_gate_count: Optional[int] = None
    race_gate_order: Optional[Tuple[int, ...]] = None
    save_perception_debug_frames: bool = True
    use_lookahead_gate_filter: bool = True
    race_pass_radius: float = 1.25
    race_clear_radius: float = 1.75
    race_advance_debounce_s: float = 0.75
    race_allow_laps: bool = True
    perception_single_lap_no_reset: bool = True
    vehicle_limits: VehicleLimitsConfig = field(default_factory=VehicleLimitsConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


__all__ = [
    "ControllerConfig",
    "LoggingConfig",
    "PerceptionConfig",
    "PlannerConfig",
    "RuntimeConfig",
    "VehicleLimitsConfig",
]
