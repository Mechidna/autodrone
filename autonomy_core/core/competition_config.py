"""Passive constants for the VADR-TS-002 competition interface.

This module is intentionally import-safe and side-effect free. It does not wire
competition constants into existing runtime defaults or import runtime modules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


Matrix3x3 = Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]
Vector3 = Tuple[float, float, float]


@dataclass(frozen=True)
class RuntimeCompetitionConfig:
    """Official VADR-TS-002 Issue 00.02 competition constants."""

    spec_id: str = "VADR-TS-002"
    spec_issue: str = "00.02"
    spec_date: str = "2026-05-08"

    mavlink_transport: str = "udp_mavlink"
    protocol_world_frame: str = "mavlink_local_ned"
    internal_world_frame: str = "z_up"

    vision_udp_port: int = 5600
    vision_header_format: str = "<IHHIIQ"
    vision_header_size_bytes: int = 24
    vision_header_fields: Tuple[str, ...] = (
        "frame_id:uint32",
        "chunk_id:uint16",
        "total_chunks:uint16",
        "jpeg_size:uint32",
        "payload_size:uint32",
        "sim_time_ns:uint64",
    )

    camera_width_px: int = 640
    camera_height_px: int = 360
    camera_fx_px: float = 320.0
    camera_fy_px: float = 320.0
    camera_cx_px: float = 320.0
    camera_cy_px: float = 180.0
    camera_distortion_model: str = "zero"
    camera_dist_coeff_count: int = 5
    camera_body_translation_m: Vector3 = (0.0, 0.0, 0.0)
    camera_tilt_up_deg: float = 20.0

    physics_rate_hz: float = 120.0
    vision_rate_hz: float = 30.0
    heartbeat_min_hz: float = 2.0
    command_rate_upper_bound_exclusive_hz: float = 100.0

    gate_outer_square_mm: int = 2700
    gate_inner_square_mm: int = 1500
    gate_depth_mm: int = 260
    drone_chassis_length_mm: int = 280
    drone_chassis_width_mm: int = 280
    drone_chassis_height_mm: int = 160
    race_max_duration_s: int = 8 * 60

    @property
    def camera_matrix(self) -> Matrix3x3:
        return (
            (self.camera_fx_px, 0.0, self.camera_cx_px),
            (0.0, self.camera_fy_px, self.camera_cy_px),
            (0.0, 0.0, 1.0),
        )

    @property
    def dist_coeffs(self) -> Tuple[float, ...]:
        return tuple(0.0 for _ in range(self.camera_dist_coeff_count))

    @property
    def camera_resolution(self) -> Tuple[int, int]:
        return (self.camera_width_px, self.camera_height_px)

    @property
    def camera_tilt_up_rad(self) -> float:
        return math.radians(self.camera_tilt_up_deg)

    @property
    def physics_period_s(self) -> float:
        return 1.0 / self.physics_rate_hz

    @property
    def vision_period_s(self) -> float:
        return 1.0 / self.vision_rate_hz

    @property
    def heartbeat_period_max_s(self) -> float:
        return 1.0 / self.heartbeat_min_hz

    @property
    def command_period_lower_bound_exclusive_s(self) -> float:
        return 1.0 / self.command_rate_upper_bound_exclusive_hz

    def command_rate_is_allowed(self, rate_hz: float) -> bool:
        return float(rate_hz) < self.command_rate_upper_bound_exclusive_hz


@dataclass(frozen=True)
class PyAIPilotExampleReferenceConfig:
    """Read-only reference-code defaults from third_party/PyAIPilotExample."""

    mavlink_default_ip: str = "127.0.0.1"
    mavlink_default_udp_port: int = 14550
    mavlink_connection_scheme: str = "udpin"
    vision_bind_ip: str = "0.0.0.0"
    vision_udp_port: int = 5600
    vision_header_format: str = "<IHHIIQ"
    vision_udp_recv_bytes: int = 65536
    timesync_request_hz: float = 10.0
    controller_update_hz: float = 250.0
    attitude_target_uses_body_rates: bool = True
    attitude_target_ignores_quaternion: bool = True
    attitude_target_dummy_quaternion_wxyz: Tuple[float, float, float, float] = (
        1.0,
        0.0,
        0.0,
        0.0,
    )
    normalized_thrust_range: Tuple[float, float] = (0.0, 1.0)
    default_controller_path: str = "set_actuator_control_target"


VADR_TS_002 = RuntimeCompetitionConfig()
PYAIPILOT_EXAMPLE_REFERENCE = PyAIPilotExampleReferenceConfig()


__all__ = [
    "PYAIPILOT_EXAMPLE_REFERENCE",
    "PyAIPilotExampleReferenceConfig",
    "RuntimeCompetitionConfig",
    "VADR_TS_002",
]
