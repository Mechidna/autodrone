from dataclasses import replace

import cv2
import numpy as np

from feature_visual_odometry import FeatureVisualOdometry
from runtime_config import load_runtime_config


def _config(**overrides):
    config = load_runtime_config()
    feature_visual_odometry = replace(
        config.feature_visual_odometry,
        enabled=True,
        fuse_velocity=True,
        detector="gftt",
        max_features=500,
        quality_level=0.001,
        min_distance_px=5.0,
        min_tracks=12,
        redetect_below_tracks=20,
        min_inliers=8,
        min_inlier_ratio=0.25,
        min_dt_s=0.01,
        max_dt_s=2.0,
        max_forward_backward_error_px=2.5,
        max_median_flow_px=120.0,
        ransac_threshold_px=2.0,
        min_scale_speed_m_s=0.01,
        max_visual_speed_m_s=20.0,
        max_velocity_innovation_m_s=20.0,
        trace=False,
        **overrides,
    )
    return replace(config, feature_visual_odometry=feature_visual_odometry)


def _perception(timestamp: float, frame_id: int) -> dict:
    return {
        "frame_id": frame_id,
        "image_wall_time": timestamp,
        "perception_wall_time": timestamp,
        "camera_matrix": np.array(
            [
                [320.0, 0.0, 320.0],
                [0.0, 320.0, 180.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        "camera_to_body": np.eye(3, dtype=float),
        "perception_yaw_correction_rad": 0.0,
    }


def _render(points_camera: np.ndarray) -> np.ndarray:
    image = np.zeros((360, 640), dtype=np.uint8)
    fx = fy = 320.0
    cx = 320.0
    cy = 180.0
    for x, y, z in points_camera:
        if z <= 0.1:
            continue
        u = int(round(fx * x / z + cx))
        v = int(round(fy * y / z + cy))
        if 4 <= u < 636 and 4 <= v < 356:
            cv2.circle(image, (u, v), 2, 255, -1)
            cv2.line(image, (u - 3, v), (u + 3, v), 180, 1)
            cv2.line(image, (u, v - 3), (u, v + 3), 180, 1)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def test_feature_vo_outputs_scaled_visual_velocity_from_klt_tracks():
    rng = np.random.default_rng(4)
    points = np.column_stack(
        [
            rng.uniform(-2.0, 2.0, 160),
            rng.uniform(-1.0, 1.0, 160),
            rng.uniform(4.0, 10.0, 160),
        ]
    )
    camera_delta = np.array([0.04, 0.00, 0.12], dtype=float)
    first_image = _render(points)
    second_image = _render(points - camera_delta)

    vo = FeatureVisualOdometry(_config())
    first = vo.update(
        first_image,
        _perception(1.0, 1),
        roll_rad=0.0,
        pitch_rad=0.0,
        yaw_rad=0.0,
        estimated_pos_neu=np.zeros(3),
        estimated_vel_neu=np.array([0.5, 0.0, 0.0]),
        now=1.0,
    )
    second = vo.update(
        second_image,
        _perception(1.2, 2),
        roll_rad=0.0,
        pitch_rad=0.0,
        yaw_rad=0.0,
        estimated_pos_neu=np.zeros(3),
        estimated_vel_neu=np.array([0.5, 0.0, 0.0]),
        now=1.2,
    )

    assert not first.valid
    assert first.reason == "no_reference"
    assert second.valid
    assert second.reason == "feature_vo"
    assert second.feature_count >= 8
    assert second.velocity_neu is not None
    assert np.isclose(np.linalg.norm(second.velocity_neu), 0.5, atol=1e-6)
