"""Passive frame-convention helpers for competition adapter boundaries.

Nothing in this module is wired into runtime perception or control. These
helpers document and test the protocol-boundary conventions before any adapter
uses them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autonomy_core.core.competition_config import RuntimeCompetitionConfig, VADR_TS_002


@dataclass(frozen=True)
class FrameConvention:
    name: str
    x_axis: str
    y_axis: str
    z_axis: str


MAVLINK_LOCAL_NED = FrameConvention(
    name="mavlink_local_ned",
    x_axis="north",
    y_axis="east",
    z_axis="down",
)

MAVLINK_BODY_FRD = FrameConvention(
    name="mavlink_body_frd",
    x_axis="forward",
    y_axis="right",
    z_axis="down",
)

INTERNAL_BODY_FLU = FrameConvention(
    name="internal_body_flu",
    x_axis="forward",
    y_axis="left",
    z_axis="up",
)

OPENCV_CAMERA_OPTICAL = FrameConvention(
    name="opencv_camera_optical",
    x_axis="right",
    y_axis="down",
    z_axis="forward",
)


def official_camera_matrix(
    config: RuntimeCompetitionConfig = VADR_TS_002,
) -> np.ndarray:
    return np.asarray(config.camera_matrix, dtype=float)


def official_dist_coeffs(
    config: RuntimeCompetitionConfig = VADR_TS_002,
) -> np.ndarray:
    return np.asarray(config.dist_coeffs, dtype=float)


def camera_translation_body_frd(
    config: RuntimeCompetitionConfig = VADR_TS_002,
) -> np.ndarray:
    return np.asarray(config.camera_body_translation_m, dtype=float).reshape(3)


def opencv_camera_to_mavlink_body_frd_rotmat(
    tilt_up_rad: float,
) -> np.ndarray:
    """Camera optical frame to MAVLink body FRD.

    MAVLink body FRD axes are x-forward, y-right, z-down. OpenCV optical camera
    axes are x-right, y-down, z-forward. A positive upward tilt rotates the
    camera optical axis toward body up, i.e. negative body z/down.

    Returned matrix maps camera-frame vectors into body-FRD vectors.
    """

    c = float(np.cos(tilt_up_rad))
    s = float(np.sin(tilt_up_rad))
    return np.array(
        [
            [0.0, s, c],
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
        ],
        dtype=float,
    )


def mavlink_body_frd_to_opencv_camera_rotmat(
    tilt_up_rad: float,
) -> np.ndarray:
    """MAVLink body FRD to OpenCV optical camera rotation matrix."""

    return opencv_camera_to_mavlink_body_frd_rotmat(tilt_up_rad).T


def official_camera_to_body_frd_rotmat(
    config: RuntimeCompetitionConfig = VADR_TS_002,
) -> np.ndarray:
    return opencv_camera_to_mavlink_body_frd_rotmat(config.camera_tilt_up_rad)


def official_body_frd_to_camera_rotmat(
    config: RuntimeCompetitionConfig = VADR_TS_002,
) -> np.ndarray:
    return mavlink_body_frd_to_opencv_camera_rotmat(config.camera_tilt_up_rad)


def body_frd_point_to_camera_optical(
    point_body_frd: np.ndarray,
    *,
    config: RuntimeCompetitionConfig = VADR_TS_002,
) -> np.ndarray:
    point_body = np.asarray(point_body_frd, dtype=float).reshape(3)
    translation = camera_translation_body_frd(config)
    return official_body_frd_to_camera_rotmat(config) @ (point_body - translation)


def project_body_frd_point_to_pixel(
    point_body_frd: np.ndarray,
    *,
    config: RuntimeCompetitionConfig = VADR_TS_002,
) -> np.ndarray:
    point_camera = body_frd_point_to_camera_optical(point_body_frd, config=config)
    z = float(point_camera[2])
    if z <= 0.0:
        raise ValueError("point is not in front of the OpenCV camera")

    camera_matrix = official_camera_matrix(config)
    u = camera_matrix[0, 0] * (point_camera[0] / z) + camera_matrix[0, 2]
    v = camera_matrix[1, 1] * (point_camera[1] / z) + camera_matrix[1, 2]
    return np.array([u, v], dtype=float)


__all__ = [
    "FrameConvention",
    "INTERNAL_BODY_FLU",
    "MAVLINK_BODY_FRD",
    "MAVLINK_LOCAL_NED",
    "OPENCV_CAMERA_OPTICAL",
    "body_frd_point_to_camera_optical",
    "camera_translation_body_frd",
    "mavlink_body_frd_to_opencv_camera_rotmat",
    "official_body_frd_to_camera_rotmat",
    "official_camera_matrix",
    "official_camera_to_body_frd_rotmat",
    "official_dist_coeffs",
    "opencv_camera_to_mavlink_body_frd_rotmat",
    "project_body_frd_point_to_pixel",
]
