"""Gazebo truth diagnostic helpers used by AutonomyAPI wrappers."""

import math

import numpy as np


def _gazebo_model_pose_to_planner(gazebo_pose, quaternion_xyzw_to_rotmat):
    """Convert Gazebo model truth into the established planner frame."""
    if not isinstance(gazebo_pose, dict):
        return None, None
    try:
        position_gazebo = np.asarray(
            gazebo_pose["gazebo_model_pos_world"], dtype=float
        ).reshape(3)
        rotation_gazebo = quaternion_xyzw_to_rotmat(
            np.asarray(
                gazebo_pose["gazebo_model_quat_world"], dtype=float
            ).reshape(4)
        )
    except (KeyError, TypeError, ValueError):
        return None, None
    if not (
        np.all(np.isfinite(position_gazebo))
        and np.all(np.isfinite(rotation_gazebo))
    ):
        return None, None

    # Gazebo world x/y map to planner y/x. The body reflection is required
    # to keep the result a proper rotation and yields yaw=90deg-gazebo_yaw.
    world_gazebo_to_planner = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    body_planner_to_gazebo = np.diag([1.0, -1.0, 1.0])
    position_planner = world_gazebo_to_planner @ position_gazebo
    rotation_planner_body = (
        world_gazebo_to_planner
        @ rotation_gazebo
        @ body_planner_to_gazebo
    )
    return position_planner, rotation_planner_body


def reset_gazebo_pose_comparison_debug(api):
    api.gazebo_model_pos_world = np.full(3, np.nan, dtype=float)
    api.gazebo_model_quat_world = np.full(4, np.nan, dtype=float)
    api.gazebo_camera_pos_world = np.full(3, np.nan, dtype=float)
    api.gazebo_camera_quat_world = np.full(4, np.nan, dtype=float)
    api.gazebo_pose_wall_time = float("nan")
    api.gazebo_pose_age_s = float("nan")
    api.gazebo_model_yaw_deg = float("nan")
    api.gazebo_camera_yaw_deg = float("nan")
    api.mavsdk_minus_gazebo_pos = np.full(3, np.nan, dtype=float)
    api.mavsdk_minus_gazebo_yaw_deg = float("nan")
    api.gate_world_mavsdk = np.full(3, np.nan, dtype=float)
    api.gate_world_gazebo = np.full(3, np.nan, dtype=float)
    api.gate_world_mavsdk_error_to_gt = float("nan")
    api.gate_world_gazebo_error_to_gt = float("nan")
    api.required_yaw_deg_from_pnp_to_gt = float("nan")
    api.mavsdk_yaw_minus_required_deg = float("nan")
    api.gazebo_yaw_minus_required_deg = float("nan")


def capture_gazebo_pose_debug(
    api,
    gazebo_pose,
    mavsdk_pos,
    mavsdk_yaw_rad,
    process_wall_time,
):
    if not isinstance(gazebo_pose, dict):
        return
    try:
        api.gazebo_model_pos_world = np.asarray(
            gazebo_pose["gazebo_model_pos_world"], dtype=float
        ).reshape(3)
        api.gazebo_model_quat_world = np.asarray(
            gazebo_pose["gazebo_model_quat_world"], dtype=float
        ).reshape(4)
        api.gazebo_camera_pos_world = np.asarray(
            gazebo_pose["gazebo_camera_pos_world"], dtype=float
        ).reshape(3)
        api.gazebo_camera_quat_world = np.asarray(
            gazebo_pose["gazebo_camera_quat_world"], dtype=float
        ).reshape(4)
        api.gazebo_pose_wall_time = float(gazebo_pose["gazebo_pose_wall_time"])
    except (KeyError, TypeError, ValueError):
        api.reset_gazebo_pose_comparison_debug()
        return

    api.gazebo_pose_age_s = float(process_wall_time) - api.gazebo_pose_wall_time
    api.gazebo_model_yaw_deg = api._quaternion_xyzw_yaw_deg(
        api.gazebo_model_quat_world
    )
    api.gazebo_camera_yaw_deg = api._quaternion_xyzw_yaw_deg(
        api.gazebo_camera_quat_world
    )
    gazebo_model_pos_planner = np.array([
        api.gazebo_model_pos_world[1],
        api.gazebo_model_pos_world[0],
        api.gazebo_model_pos_world[2],
    ], dtype=float)
    api.mavsdk_minus_gazebo_pos = (
        np.asarray(mavsdk_pos, dtype=float).reshape(3)
        - gazebo_model_pos_planner
    )
    api.mavsdk_minus_gazebo_yaw_deg = api._wrap_degrees(
        math.degrees(float(mavsdk_yaw_rad)) - api.gazebo_model_yaw_deg
    )


def compute_gazebo_pose_gate_comparison_debug(
    api,
    pnp_camera,
    mavsdk_pos,
    mavsdk_rpy_raw,
    perception_rpy,
):
    if pnp_camera is None or len(api.gt_gates) == 0:
        return
    pnp_camera = np.asarray(pnp_camera, dtype=float).reshape(3)
    mavsdk_pos = np.asarray(mavsdk_pos, dtype=float).reshape(3)
    mavsdk_rpy_raw = np.asarray(mavsdk_rpy_raw, dtype=float).reshape(3)
    perception_rpy = np.asarray(perception_rpy, dtype=float).reshape(3)
    if not np.all(np.isfinite(pnp_camera)):
        return

    gate_idx = int(np.clip(api.current_gate_idx, 0, len(api.gt_gates) - 1))
    gt_gate = np.asarray(api.gt_gates[gate_idx], dtype=float).reshape(3)
    r_body_camera = api.body_camera_matrix_for_mode(api.perception_transform_mode)
    body_vec = api.camera_offset_body + r_body_camera @ pnp_camera
    r_wb_uncorrected = api.perception_node._rpy_to_rotmat(
        float(mavsdk_rpy_raw[0]),
        float(mavsdk_rpy_raw[1]),
        float(mavsdk_rpy_raw[2]),
    )
    r_wb_corrected = api.perception_node._rpy_to_rotmat(
        float(perception_rpy[0]),
        float(perception_rpy[1]),
        float(perception_rpy[2]),
    )
    api.gate_world_uncorrected = mavsdk_pos + r_wb_uncorrected @ body_vec
    api.gate_world_corrected = mavsdk_pos + r_wb_corrected @ body_vec
    api.gate_world_mavsdk = api.gate_world_uncorrected.copy()
    api.gate_world_mavsdk_error_to_gt = float(
        np.linalg.norm(api.gate_world_mavsdk - gt_gate)
    )

    r_wb_gazebo = api._quaternion_xyzw_to_rotmat(api.gazebo_model_quat_world)
    if (
        np.all(np.isfinite(api.gazebo_model_pos_world))
        and np.all(np.isfinite(r_wb_gazebo))
    ):
        api.gate_world_gazebo = (
            api.gazebo_model_pos_world + r_wb_gazebo @ body_vec
        )
        api.gate_world_gazebo_error_to_gt = float(
            np.linalg.norm(api.gate_world_gazebo - gt_gate)
        )

    # For Rz(yaw) @ Ry(pitch) @ Rx(roll), solve the yaw that aligns
    # the pitch/roll-adjusted body vector with the GT horizontal bearing.
    r_no_yaw = api.perception_node._rpy_to_rotmat(
        float(mavsdk_rpy_raw[0]),
        float(mavsdk_rpy_raw[1]),
        0.0,
    )
    leveled_body_vec = r_no_yaw @ body_vec
    desired_delta = gt_gate - mavsdk_pos
    if (
        np.linalg.norm(leveled_body_vec[:2]) > 1e-9
        and np.linalg.norm(desired_delta[:2]) > 1e-9
    ):
        required_yaw = math.atan2(
            float(desired_delta[1]), float(desired_delta[0])
        ) - math.atan2(
            float(leveled_body_vec[1]), float(leveled_body_vec[0])
        )
        api.required_yaw_deg_from_pnp_to_gt = api._wrap_degrees(
            math.degrees(required_yaw)
        )
        api.mavsdk_yaw_minus_required_deg = api._wrap_degrees(
            math.degrees(float(mavsdk_rpy_raw[2]))
            - api.required_yaw_deg_from_pnp_to_gt
        )
        if np.isfinite(api.gazebo_model_yaw_deg):
            api.gazebo_yaw_minus_required_deg = api._wrap_degrees(
                api.gazebo_model_yaw_deg
                - api.required_yaw_deg_from_pnp_to_gt
            )
