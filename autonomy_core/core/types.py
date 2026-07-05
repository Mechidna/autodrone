"""Shared data contracts for autonomy runtime seams.

These dataclasses are intentionally passive containers. Phase 2 adds them as an
importable contract layer without changing existing runtime call sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class State:
    """Current vehicle state in the existing internal z-up world frame."""

    pos: np.ndarray
    vel: np.ndarray
    yaw: float


VehicleState = State


@dataclass
class Reference:
    """Trajectory reference consumed by the current high-level tracker."""

    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    yaw: float
    yaw_rate: float = 0.0


ReferenceState = Reference


@dataclass
class ControlCommand:
    """Attitude/thrust command returned by control policy seams."""

    roll: float
    pitch: float
    yaw: float
    thrust: float


@dataclass
class CameraModel:
    """Camera intrinsics/distortion used by perception pipelines."""

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray


@dataclass
class CameraFrame:
    """Image plus timing and pose metadata passed into perception."""

    frame: np.ndarray
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_stamp_sec: int = 0
    image_stamp_nanosec: int = 0
    image_received_wall_time: float = float("nan")
    image_pose_snapshot: Optional[Any] = None
    gazebo_pose: Optional[Any] = None


@dataclass
class GateTarget:
    """Candidate or active gate target in the planner world frame."""

    center: np.ndarray
    track_id: Optional[int] = None
    source: str = ""
    confidence: float = float("nan")


@dataclass
class PlanDebugInfo:
    """Minimal plan metadata mirrored by existing AutonomyAPI log fields."""

    plan_id: int = 0
    mode: str = ""
    start_gate_idx: Optional[int] = None
    end_gate_idx: Optional[int] = None
    duration: float = float("nan")
    replan_reason: str = ""
