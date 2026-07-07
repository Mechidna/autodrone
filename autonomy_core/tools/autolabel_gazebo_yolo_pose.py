#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CAPTURE_ROOT = "~/datasets/gazebo_gate_capture"
DEFAULT_OUTPUT_ROOT = "~/datasets/gazebo_gate_yolo_pose"
DEFAULT_WORLD_SDF = (
    "/home/paolo/PX4-Autopilot/PX4-Autopilot/Tools/simulation/gz/worlds/"
    "gate_test_1500mm_blue_random.sdf"
)

GATE_CENTERS_WORLD_MAVSDK = (
    np.array([0.0, 8.0, 1.5], dtype=float),
    np.array([0.8, 16.0, 1.5], dtype=float),
    np.array([-0.8, 24.0, 1.5], dtype=float),
)
GATE_CENTERS_WORLD_GAZEBO = (
    np.array([8.0, 0.0, 1.35], dtype=float),
    np.array([16.0, 0.8, 1.35], dtype=float),
    np.array([24.0, -0.8, 1.35], dtype=float),
)
GATE_YAWS_RAD_MAVSDK = (0.0, 0.0, 0.0)
GATE_YAWS_RAD_GAZEBO = (
    np.pi / 2.0,
    np.pi / 2.0,
    np.pi / 2.0,
)
INNER_OPENING_M = 1.5
OUTER_GATE_M = 2.7
GATE_DEPTH_M = 0.260
GATE_EXIT_FACE_OFFSET_M = GATE_DEPTH_M * 0.5
SELF_OCCLUSION_ENDPOINT_CLEARANCE_M = 0.02
KEYPOINT_LAYOUT_INNER4 = "inner4"
KEYPOINT_LAYOUT_INNER4_OUTER4 = "inner4_outer4"
KEYPOINT_LAYOUT_CHOICES = (KEYPOINT_LAYOUT_INNER4, KEYPOINT_LAYOUT_INNER4_OUTER4)
KEYPOINT_LAYOUT_FLIP_IDX = {
    KEYPOINT_LAYOUT_INNER4: (1, 0, 3, 2),
    KEYPOINT_LAYOUT_INNER4_OUTER4: (1, 0, 3, 2, 5, 4, 7, 6),
}
KEYPOINT_LAYOUT_NAMES = {
    KEYPOINT_LAYOUT_INNER4: ("TL", "TR", "BR", "BL"),
    KEYPOINT_LAYOUT_INNER4_OUTER4: (
        "I-TL",
        "I-TR",
        "I-BR",
        "I-BL",
        "O-TL",
        "O-TR",
        "O-BR",
        "O-BL",
    ),
}
GATE_CENTER_Z_OFFSET_M = 1.35
SDF_GATE_YAW_OFFSET_RAD = np.pi / 2.0
MIN_PARTIAL_BBOX_AREA_PX2 = 64.0
GATE_OCCLUSION_PADDING_M = 0.0
DEFAULT_MAX_GATE_LABEL_DISTANCE_M = 200.0
# Legacy MAVSDK fallback only. Current racer_mono_cam captures should carry
# gazebo_camera_* pose metadata from aigp/tools/capture_gazebo_yolo_pose.py.
CAMERA_OFFSET_BODY = np.array([0.12, 0.03, 0.242], dtype=float)
ROLL_SIGN = -1.0
PITCH_SIGN = -1.0
YAW_SIGN = 1.0
YAW_OFFSET_RAD = 0.0
# OpenCV camera frame:
#   x = right
#   y = down
#   z = forward
#
# Body frame:
#   x = forward
#   y = left
#   z = up
#
# Therefore:
#   camera z forward -> body x
#   camera x right   -> -body y
#   camera y down    -> -body z

R_BODY_CAMERA = np.array([
    [0.0,  0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
], dtype=float)


def _parse_pose(text: str | None, *, context: str) -> tuple[float, float, float, float, float, float]:
    if text is None:
        raise RuntimeError(f"{context} has no <pose> text.")
    values = tuple(float(part) for part in str(text).split())
    if len(values) != 6:
        raise RuntimeError(f"{context} pose must have 6 values, got {len(values)}.")
    return values


def _parse_pose_or_default(
    text: str | None,
    *,
    context: str,
) -> tuple[float, float, float, float, float, float]:
    if text is None or not str(text).strip():
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return _parse_pose(text, context=context)


def _parse_box_size(text: str | None, *, context: str) -> np.ndarray:
    if text is None:
        raise RuntimeError(f"{context} has no <size> text.")
    values = np.asarray([float(part) for part in str(text).split()], dtype=float)
    if values.shape != (3,):
        raise RuntimeError(f"{context} box size must have 3 values, got {len(values)}.")
    return values


def _transform_from_pose_values(
    pose_values: tuple[float, float, float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    x, y, z, roll, pitch, yaw = pose_values
    return np.array([x, y, z], dtype=float), rpy_to_rotmat(roll, pitch, yaw)


def _compose_transform(
    parent_translation: np.ndarray,
    parent_rotation: np.ndarray,
    child_pose_values: tuple[float, float, float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    child_translation, child_rotation = _transform_from_pose_values(child_pose_values)
    return (
        parent_translation + parent_rotation @ child_translation,
        parent_rotation @ child_rotation,
    )


def _load_model_box_occluders(
    model: ET.Element,
    *,
    model_name: str,
    model_translation: np.ndarray,
    model_rotation: np.ndarray,
) -> tuple[dict, ...]:
    occluders = []
    for link in model.findall("./link"):
        link_name = link.get("name", "link")
        link_pose = _parse_pose_or_default(
            link.findtext("pose"),
            context=f"{model_name}:{link_name}",
        )
        link_translation, link_rotation = _compose_transform(
            model_translation,
            model_rotation,
            link_pose,
        )

        for visual_index, visual in enumerate(link.findall("./visual")):
            size_text = visual.findtext("./geometry/box/size")
            if size_text is None:
                continue

            visual_name = visual.get("name", f"visual_{visual_index}")
            size = _parse_box_size(
                size_text,
                context=f"{model_name}:{link_name}:{visual_name}",
            )
            if np.any(size <= 0.0):
                continue

            visual_pose = _parse_pose_or_default(
                visual.findtext("pose"),
                context=f"{model_name}:{link_name}:{visual_name}",
            )
            box_translation, box_rotation = _compose_transform(
                link_translation,
                link_rotation,
                visual_pose,
            )
            occluders.append({
                "name": f"{model_name}/{link_name}/{visual_name}",
                "center_world": box_translation,
                "rotation_world_from_box": box_rotation,
                "half_extents": 0.5 * size,
            })

    return tuple(occluders)


def load_world_sdf_gate_geometry(
    world_sdf,
    *,
    gate_center_z_offset_m: float = GATE_CENTER_Z_OFFSET_M,
    sdf_gate_yaw_offset_rad: float = SDF_GATE_YAW_OFFSET_RAD,
) -> dict:
    """Load racing_gate_* centers/yaws from a Gazebo world SDF.

    Gazebo/SDF world coordinates are x-forward, y-lateral, z-up for the local
    PX4 worlds. The gate model's inner-opening center is offset upward from the
    model root by gate_center_z_offset_m. The gate opening width is along the
    gate model's local +Y axis, so the labeler's right-axis yaw is SDF yaw + pi/2.
    """

    path = Path(os.path.expanduser(str(world_sdf)))
    if not path.exists():
        raise RuntimeError(f"World SDF not found: {path}")

    root = ET.parse(path).getroot()
    world = root.find("world")
    if world is None:
        world = root

    gates = []
    gate_name_re = re.compile(r"^racing_gate_(\d+)$")
    for model in world.findall("./model"):
        name = model.get("name", "")
        match = gate_name_re.match(name)
        if match is None:
            continue

        gate_idx = int(match.group(1))
        pose_values = _parse_pose_or_default(
            model.findtext("pose"),
            context=f"{path}:{name}",
        )
        sdf_x, sdf_y, sdf_z, _roll, _pitch, sdf_yaw = pose_values
        model_translation, model_rotation = _transform_from_pose_values(pose_values)
        center_gazebo = (
            model_translation
            + model_rotation @ np.array([0.0, 0.0, float(gate_center_z_offset_m)], dtype=float)
        )
        yaw_gazebo = float(sdf_yaw) + float(sdf_gate_yaw_offset_rad)
        gazebo_occluders = _load_model_box_occluders(
            model,
            model_name=name,
            model_translation=model_translation,
            model_rotation=model_rotation,
        )

        # Existing MAVSDK/NEU-like debug metadata swaps SDF x/y.
        center_mavsdk = np.array(
            [sdf_y, sdf_x, sdf_z + float(gate_center_z_offset_m)],
            dtype=float,
        )
        yaw_mavsdk = -float(sdf_yaw)
        gates.append((
            gate_idx,
            center_gazebo,
            yaw_gazebo,
            center_mavsdk,
            yaw_mavsdk,
            gazebo_occluders,
        ))

    if not gates:
        raise RuntimeError(f"No racing_gate_* models found in {path}")

    gates.sort(key=lambda item: item[0])
    return {
        "source": f"world_sdf:{path}",
        "gazebo_centers": tuple(item[1] for item in gates),
        "gazebo_yaws": tuple(item[2] for item in gates),
        "gazebo_occluders": tuple(item[5] for item in gates),
        "gazebo_frame": "gazebo_world_sdf",
        "mavsdk_centers": tuple(item[3] for item in gates),
        "mavsdk_yaws": tuple(item[4] for item in gates),
        "mavsdk_occluders": tuple(() for _item in gates),
        "mavsdk_frame": "mavsdk_neu_from_sdf",
    }


def legacy_hardcoded_gate_geometry() -> dict:
    return {
        "source": "legacy_hardcoded",
        "gazebo_centers": GATE_CENTERS_WORLD_GAZEBO,
        "gazebo_yaws": GATE_YAWS_RAD_GAZEBO,
        "gazebo_occluders": tuple(() for _center in GATE_CENTERS_WORLD_GAZEBO),
        "gazebo_frame": "legacy_gazebo",
        "mavsdk_centers": GATE_CENTERS_WORLD_MAVSDK,
        "mavsdk_yaws": GATE_YAWS_RAD_MAVSDK,
        "mavsdk_occluders": tuple(() for _center in GATE_CENTERS_WORLD_MAVSDK),
        "mavsdk_frame": "legacy_mavsdk",
    }


def resolve_gate_geometry(args, world_sdf=None) -> dict:
    if bool(getattr(args, "legacy_hardcoded_gates", False)):
        return legacy_hardcoded_gate_geometry()
    selected_world_sdf = world_sdf or getattr(args, "world_sdf", None) or DEFAULT_WORLD_SDF
    return load_world_sdf_gate_geometry(
        selected_world_sdf,
        gate_center_z_offset_m=float(args.gate_center_z_offset_m),
        sdf_gate_yaw_offset_rad=float(args.sdf_gate_yaw_offset_rad),
    )


def rpy_to_rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=float)


def quaternion_to_rotmat(quat_xyzw) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, dtype=float).reshape(4)
    norm = np.linalg.norm([x, y, z, w])
    if norm <= 0.0:
        return np.eye(3, dtype=float)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)


def gate_axes_world(gate_yaw_rad: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return right, up, and exit-normal axes for the SDF gate model."""
    right = np.array([np.cos(gate_yaw_rad), np.sin(gate_yaw_rad), 0.0], dtype=float)
    up = np.array([0.0, 0.0, 1.0], dtype=float)
    # The SDF gate boxes use local X as the 0.260 m depth and local +Y as the
    # opening-width axis. gate_yaw_rad is the world yaw of local +Y, so local +X
    # is right x up. In the generated race worlds, local +X is the exit face.
    exit_normal = np.cross(right, up)
    exit_normal /= np.linalg.norm(exit_normal) + 1e-12
    return right, up, exit_normal


def gate_inner_corners_world(center_world: np.ndarray, gate_yaw_rad: float) -> np.ndarray:
    center_world = np.asarray(center_world, dtype=float).reshape(3)
    half = INNER_OPENING_M / 2.0

    right, up, exit_normal = gate_axes_world(gate_yaw_rad)
    face_center_world = center_world + GATE_EXIT_FACE_OFFSET_M * exit_normal

    return np.array([
        face_center_world - half * right + half * up,  # TL
        face_center_world + half * right + half * up,  # TR
        face_center_world + half * right - half * up,  # BR
        face_center_world - half * right - half * up,  # BL
    ], dtype=float)


def gate_outer_corners_world(center_world: np.ndarray, gate_yaw_rad: float) -> np.ndarray:
    center_world = np.asarray(center_world, dtype=float).reshape(3)
    half = OUTER_GATE_M / 2.0

    right, up, exit_normal = gate_axes_world(gate_yaw_rad)
    # Outer exit-face corners are usually hidden from the drone by the near
    # frame, so label the visible entry face for the outer silhouette.
    face_center_world = center_world - GATE_EXIT_FACE_OFFSET_M * exit_normal

    return np.array([
        face_center_world - half * right + half * up,  # TL
        face_center_world + half * right + half * up,  # TR
        face_center_world + half * right - half * up,  # BR
        face_center_world - half * right - half * up,  # BL
    ], dtype=float)


def world_to_camera(points_world, drone_pos, drone_rpy_rad):
    points_world = np.asarray(points_world, dtype=float).reshape(-1, 3)
    drone_pos = np.asarray(drone_pos, dtype=float).reshape(3)
    roll, pitch, yaw = np.asarray(drone_rpy_rad, dtype=float).reshape(3)

    roll = ROLL_SIGN * float(roll)
    pitch = PITCH_SIGN * float(pitch)
    yaw = YAW_SIGN * float(yaw) + YAW_OFFSET_RAD

    r_wb = rpy_to_rotmat(roll, pitch, yaw)
    r_bw = r_wb.T
    r_camera_body = R_BODY_CAMERA.T

    points_body = (r_bw @ (points_world - drone_pos).T).T - CAMERA_OFFSET_BODY
    points_camera = (r_camera_body @ points_body.T).T
    return points_camera


def gazebo_world_to_camera_body(
    points_world,
    camera_pos_world,
    camera_quat_world,
    gazebo_rotation_mode="direct",
):
    points_world = np.asarray(points_world, dtype=float).reshape(-1, 3)
    camera_pos_world = np.asarray(camera_pos_world, dtype=float).reshape(3)
    r_wc = quaternion_to_rotmat(camera_quat_world)

    rel_world = points_world - camera_pos_world
    if gazebo_rotation_mode == "transpose":
        points_cam_body = (r_wc.T @ rel_world.T).T
    elif gazebo_rotation_mode == "direct":
        points_cam_body = (r_wc @ rel_world.T).T
    else:
        raise ValueError(f"Unsupported gazebo_rotation_mode: {gazebo_rotation_mode}")

    return points_cam_body


def gazebo_body_to_camera_optical(points_body, gazebo_optical_mode="current"):
    points_body = np.asarray(points_body, dtype=float).reshape(-1, 3)
    if gazebo_optical_mode in ("current", "flip_y"):
        points_camera = (R_BODY_CAMERA.T @ points_body.T).T
        if gazebo_optical_mode == "flip_y":
            points_camera[:, 1] *= -1.0
    elif gazebo_optical_mode == "physical":
        points_camera = np.column_stack([
            points_body[:, 1],
            -points_body[:, 2],
            points_body[:, 0],
        ])
    elif gazebo_optical_mode == "physical_minus_y":
        points_camera = np.column_stack([
            -points_body[:, 1],
            -points_body[:, 2],
            points_body[:, 0],
        ])
    else:
        raise ValueError(f"Unsupported gazebo_optical_mode: {gazebo_optical_mode}")

    return points_camera


def gazebo_world_to_camera(
    points_world,
    camera_pos_world,
    camera_quat_world,
    gazebo_rotation_mode="direct",
    gazebo_optical_mode="current",
):
    points_body = gazebo_world_to_camera_body(
        points_world,
        camera_pos_world,
        camera_quat_world,
        gazebo_rotation_mode=gazebo_rotation_mode,
    )
    return gazebo_body_to_camera_optical(points_body, gazebo_optical_mode=gazebo_optical_mode)


def metadata_has_gazebo_pose(metadata):
    return (
        metadata.get("gazebo_camera_pos_world") is not None
        and metadata.get("gazebo_camera_quat_world") is not None
    )


def metadata_camera_position_world(metadata):
    if metadata_has_gazebo_pose(metadata):
        return np.asarray(metadata["gazebo_camera_pos_world"], dtype=float).reshape(3)
    return np.asarray(metadata["drone_pos"], dtype=float).reshape(3)


def metadata_world_to_camera(
    points_world,
    metadata,
    gazebo_rotation_mode="direct",
    gazebo_optical_mode="current",
):
    if metadata_has_gazebo_pose(metadata):
        return gazebo_world_to_camera(
            points_world,
            metadata["gazebo_camera_pos_world"],
            metadata["gazebo_camera_quat_world"],
            gazebo_rotation_mode=gazebo_rotation_mode,
            gazebo_optical_mode=gazebo_optical_mode,
        )

    return world_to_camera(
        points_world,
        metadata["drone_pos"],
        metadata["drone_rpy_rad"],
    )


def metadata_pose_source(metadata):
    return "gazebo" if metadata_has_gazebo_pose(metadata) else "mavsdk"


def metadata_gate_centers_frame(metadata, gate_geometry=None):
    if gate_geometry is None:
        return "gazebo" if metadata_has_gazebo_pose(metadata) else "mavsdk"
    return (
        gate_geometry["gazebo_frame"]
        if metadata_has_gazebo_pose(metadata)
        else gate_geometry["mavsdk_frame"]
    )


def metadata_gate_centers(metadata, gate_geometry=None):
    if gate_geometry is not None:
        if metadata_has_gazebo_pose(metadata):
            return gate_geometry["gazebo_centers"]
        return gate_geometry["mavsdk_centers"]
    if metadata_has_gazebo_pose(metadata):
        return GATE_CENTERS_WORLD_GAZEBO
    return GATE_CENTERS_WORLD_MAVSDK


def metadata_gate_yaws(metadata, gate_geometry=None):
    if gate_geometry is not None:
        if metadata_has_gazebo_pose(metadata):
            return gate_geometry["gazebo_yaws"]
        return gate_geometry["mavsdk_yaws"]
    if metadata_has_gazebo_pose(metadata):
        return GATE_YAWS_RAD_GAZEBO
    return GATE_YAWS_RAD_MAVSDK


def metadata_gate_occluders(metadata, gate_geometry=None):
    if gate_geometry is None:
        return ()
    if metadata_has_gazebo_pose(metadata):
        return gate_geometry.get("gazebo_occluders", ())
    return gate_geometry.get("mavsdk_occluders", ())


def gate_right_axis(gate_yaw_rad):
    return np.array([np.cos(gate_yaw_rad), np.sin(gate_yaw_rad), 0.0], dtype=float)


def segment_intersects_obb(
    segment_start_world,
    segment_end_world,
    box: dict,
    *,
    padding_m: float = 0.0,
    t_min: float = 1e-5,
    t_max: float = 1.0 - 1e-5,
) -> bool:
    segment_start_world = np.asarray(segment_start_world, dtype=float).reshape(3)
    segment_end_world = np.asarray(segment_end_world, dtype=float).reshape(3)
    center_world = np.asarray(box["center_world"], dtype=float).reshape(3)
    rotation_world_from_box = np.asarray(box["rotation_world_from_box"], dtype=float).reshape(3, 3)
    half_extents = (
        np.asarray(box["half_extents"], dtype=float).reshape(3)
        + max(0.0, float(padding_m))
    )

    start_box = rotation_world_from_box.T @ (segment_start_world - center_world)
    end_box = rotation_world_from_box.T @ (segment_end_world - center_world)
    direction_box = end_box - start_box

    enter_t = float(t_min)
    exit_t = float(t_max)
    for axis in range(3):
        if abs(direction_box[axis]) < 1e-12:
            if start_box[axis] < -half_extents[axis] or start_box[axis] > half_extents[axis]:
                return False
            continue

        t1 = (-half_extents[axis] - start_box[axis]) / direction_box[axis]
        t2 = (half_extents[axis] - start_box[axis]) / direction_box[axis]
        if t1 > t2:
            t1, t2 = t2, t1
        enter_t = max(enter_t, float(t1))
        exit_t = min(exit_t, float(t2))
        if enter_t > exit_t:
            return False

    return exit_t >= float(t_min) and enter_t <= float(t_max)


def keypoint_gate_occlusion(
    camera_pos_world,
    keypoints_world,
    *,
    target_gate_idx: int,
    gate_occluders,
    padding_m: float = GATE_OCCLUSION_PADDING_M,
) -> tuple[np.ndarray, list[str]]:
    keypoints_world = np.asarray(keypoints_world, dtype=float).reshape(-1, 3)
    camera_pos_world = np.asarray(camera_pos_world, dtype=float).reshape(3)
    occluded = np.zeros(len(keypoints_world), dtype=bool)
    occluder_names = [""] * len(keypoints_world)

    for point_idx, keypoint_world in enumerate(keypoints_world):
        for gate_idx, boxes in enumerate(gate_occluders):
            if gate_idx == int(target_gate_idx):
                continue
            for box in boxes:
                if segment_intersects_obb(
                    camera_pos_world,
                    keypoint_world,
                    box,
                    padding_m=padding_m,
                ):
                    occluded[point_idx] = True
                    occluder_names[point_idx] = str(box.get("name", f"gate_{gate_idx}"))
                    break
            if occluded[point_idx]:
                break

    return occluded, occluder_names


def keypoint_same_gate_occlusion(
    camera_pos_world,
    keypoints_world,
    *,
    target_gate_idx: int,
    gate_occluders,
    padding_m: float = GATE_OCCLUSION_PADDING_M,
    endpoint_clearance_m: float = SELF_OCCLUSION_ENDPOINT_CLEARANCE_M,
) -> tuple[np.ndarray, list[str]]:
    keypoints_world = np.asarray(keypoints_world, dtype=float).reshape(-1, 3)
    camera_pos_world = np.asarray(camera_pos_world, dtype=float).reshape(3)
    occluded = np.zeros(len(keypoints_world), dtype=bool)
    occluder_names = [""] * len(keypoints_world)

    gate_occluders = tuple(gate_occluders or ())
    target_gate_idx = int(target_gate_idx)
    if target_gate_idx < 0 or target_gate_idx >= len(gate_occluders):
        return occluded, occluder_names

    boxes = tuple(gate_occluders[target_gate_idx])
    endpoint_clearance_m = max(0.0, float(endpoint_clearance_m))
    for point_idx, keypoint_world in enumerate(keypoints_world):
        segment_length = float(np.linalg.norm(keypoint_world - camera_pos_world))
        if segment_length <= endpoint_clearance_m:
            continue
        t_max = 1.0 - endpoint_clearance_m / segment_length
        t_max = min(1.0 - 1e-5, max(0.0, t_max))
        for box in boxes:
            if segment_intersects_obb(
                camera_pos_world,
                keypoint_world,
                box,
                padding_m=padding_m,
                t_max=t_max,
            ):
                occluded[point_idx] = True
                box_name = str(box.get("name", f"gate_{target_gate_idx}"))
                occluder_names[point_idx] = f"self:{box_name}"
                break

    return occluded, occluder_names


def merge_keypoint_occlusions(
    primary_occluded,
    primary_occluder_names,
    secondary_occluded,
    secondary_occluder_names,
) -> tuple[np.ndarray, list[str]]:
    primary_occluded = np.asarray(primary_occluded, dtype=bool).reshape(-1)
    secondary_occluded = np.asarray(secondary_occluded, dtype=bool).reshape(-1)
    if primary_occluded.shape != secondary_occluded.shape:
        raise ValueError(
            "Occlusion masks must have the same shape: "
            f"{primary_occluded.shape} != {secondary_occluded.shape}"
        )

    occluded = primary_occluded | secondary_occluded
    occluder_names = list(primary_occluder_names)
    if len(occluder_names) != len(occluded):
        occluder_names = [""] * len(occluded)
    for idx, is_secondary_only in enumerate(secondary_occluded & ~primary_occluded):
        if is_secondary_only:
            occluder_names[idx] = str(secondary_occluder_names[idx])
    return occluded, occluder_names


def project_camera_points(points_camera, camera_matrix, dist_coeffs):
    points_camera = np.asarray(points_camera, dtype=float).reshape(-1, 3)
    camera_matrix = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
    dist_coeffs = np.asarray(dist_coeffs, dtype=float).reshape(-1, 1)

    image_points, _ = cv2.projectPoints(
        points_camera.reshape(-1, 1, 3),
        np.zeros((3, 1), dtype=float),
        np.zeros((3, 1), dtype=float),
        camera_matrix,
        dist_coeffs,
    )
    return image_points.reshape(-1, 2)


def order_points_tl_tr_br_bl(points_image):
    points_image = np.asarray(points_image, dtype=float).reshape(4, 2)
    return points_image[order_points_tl_tr_br_bl_indices(points_image)]


def order_points_tl_tr_br_bl_indices(points_image):
    points_image = np.asarray(points_image, dtype=float).reshape(4, 2)
    by_y_indices = np.argsort(points_image[:, 1])
    top_indices = by_y_indices[:2]
    bottom_indices = by_y_indices[2:]
    top_indices = top_indices[np.argsort(points_image[top_indices, 0])]
    bottom_indices = bottom_indices[np.argsort(points_image[bottom_indices, 0])]
    tl_idx, tr_idx = top_indices
    bl_idx, br_idx = bottom_indices
    return np.array([tl_idx, tr_idx, br_idx, bl_idx], dtype=int)


def inside_image(points_image, width, height):
    points_image = np.asarray(points_image, dtype=float).reshape(-1, 2)
    x = points_image[:, 0]
    y = points_image[:, 1]
    return bool(np.all((x >= 0.0) & (x < float(width)) & (y >= 0.0) & (y < float(height))))


def points_inside_image(points_image, width, height):
    points_image = np.asarray(points_image, dtype=float).reshape(-1, 2)
    return (
        np.isfinite(points_image).all(axis=1)
        & (points_image[:, 0] >= 0.0)
        & (points_image[:, 0] < float(width))
        & (points_image[:, 1] >= 0.0)
        & (points_image[:, 1] < float(height))
    )


def clip_polygon_to_image(points_image, width, height):
    polygon = [np.asarray(point, dtype=float) for point in np.asarray(points_image).reshape(-1, 2)]
    boundaries = (
        (0, 0.0, True),
        (0, float(width - 1), False),
        (1, 0.0, True),
        (1, float(height - 1), False),
    )

    for axis, boundary, keep_greater in boundaries:
        if not polygon:
            break
        clipped = []
        previous = polygon[-1]
        previous_inside = (
            previous[axis] >= boundary if keep_greater else previous[axis] <= boundary
        )
        for current in polygon:
            current_inside = (
                current[axis] >= boundary if keep_greater else current[axis] <= boundary
            )
            if current_inside != previous_inside:
                denominator = current[axis] - previous[axis]
                if abs(denominator) > 1e-12:
                    ratio = (boundary - previous[axis]) / denominator
                    intersection = previous + ratio * (current - previous)
                    intersection[axis] = boundary
                    clipped.append(intersection)
            if current_inside:
                clipped.append(current)
            previous = current
            previous_inside = current_inside
        polygon = clipped

    return np.asarray(polygon, dtype=float).reshape(-1, 2)


def bbox_corners_from_points(points_image):
    points_image = np.asarray(points_image, dtype=float).reshape(-1, 2)
    x_min = float(np.min(points_image[:, 0]))
    x_max = float(np.max(points_image[:, 0]))
    y_min = float(np.min(points_image[:, 1]))
    y_max = float(np.max(points_image[:, 1]))
    return np.array(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
        dtype=float,
    )


def normalize_keypoint_layout(value):
    layout = str(value or KEYPOINT_LAYOUT_INNER4).strip().lower()
    if layout not in KEYPOINT_LAYOUT_CHOICES:
        raise ValueError(
            f"Unsupported keypoint layout {value!r}; use one of {KEYPOINT_LAYOUT_CHOICES}."
        )
    return layout


def keypoint_layout_count(keypoint_layout):
    return len(KEYPOINT_LAYOUT_FLIP_IDX[normalize_keypoint_layout(keypoint_layout)])


def yolo_pose_line(
    bbox_points_image,
    keypoints_image,
    keypoint_visibility,
    width,
    height,
    class_id=0,
):
    bbox_points_image = np.asarray(bbox_points_image, dtype=float).reshape(-1, 2)
    keypoints_image = np.asarray(keypoints_image, dtype=float).reshape(-1, 2)
    raw_visibility = np.asarray(keypoint_visibility).reshape(-1)
    if raw_visibility.shape[0] != keypoints_image.shape[0]:
        raise ValueError(
            "keypoint_visibility count must match keypoints_image count: "
            f"{raw_visibility.shape[0]} != {keypoints_image.shape[0]}"
        )
    if raw_visibility.dtype == np.bool_:
        keypoint_visibility = np.where(raw_visibility, 2, 0).astype(int)
    else:
        keypoint_visibility = np.clip(raw_visibility.astype(int), 0, 2)
    xs = bbox_points_image[:, 0]
    ys = bbox_points_image[:, 1]

    x_min = float(np.min(xs))
    x_max = float(np.max(xs))
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))

    cx = ((x_min + x_max) * 0.5) / float(width)
    cy = ((y_min + y_max) * 0.5) / float(height)
    w = (x_max - x_min) / float(width)
    h = (y_max - y_min) / float(height)

    values = [class_id, cx, cy, w, h]
    for (x, y), visibility in zip(keypoints_image, keypoint_visibility):
        if int(visibility) > 0:
            values.extend([
                float(x) / float(width),
                float(y) / float(height),
                int(visibility),
            ])
        else:
            values.extend([0.0, 0.0, 0])

    return " ".join(
        str(v) if isinstance(v, int) else f"{float(v):.8f}"
        for v in values
    )


def load_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "image_filename": str(metadata["image_filename"]),
        "timestamp": float(metadata["timestamp"]),
        "drone_pos": np.asarray(metadata["drone_pos"], dtype=float).reshape(3),
        "drone_rpy_rad": np.asarray(metadata["drone_rpy_rad"], dtype=float).reshape(3),
        "camera_matrix": np.asarray(metadata["camera_matrix"], dtype=float).reshape(3, 3),
        "dist_coeffs": np.asarray(metadata.get("dist_coeffs", [0, 0, 0, 0, 0]), dtype=float).reshape(-1, 1),
        "image_width": int(metadata["image_width"]),
        "image_height": int(metadata["image_height"]),
        "gazebo_camera_pos_world": (
            np.asarray(metadata["gazebo_camera_pos_world"], dtype=float).reshape(3)
            if metadata.get("gazebo_camera_pos_world") is not None
            else None
        ),
        "gazebo_camera_quat_world": (
            np.asarray(metadata["gazebo_camera_quat_world"], dtype=float).reshape(4)
            if metadata.get("gazebo_camera_quat_world") is not None
            else None
        ),
    }


def draw_preview(image, labels):
    preview = image.copy()
    outer_colors = ((0, 180, 255), (255, 128, 0), (0, 128, 255))
    inner_colors = ((0, 255, 0), (255, 0, 0), (0, 255, 255))

    for label in labels:
        gate_idx = label["gate_idx"]
        bbox_points = label["bbox_points_image"]
        keypoints = label["keypoints_image"]
        raw_visibility = np.asarray(
            label.get("keypoint_yolo_visibility", label["keypoint_visibility"])
        ).reshape(-1)
        if raw_visibility.dtype == np.bool_:
            keypoint_visibility = np.where(raw_visibility, 2, 0).astype(int)
        else:
            keypoint_visibility = np.clip(raw_visibility.astype(int), 0, 2)
        keypoints_clipped = np.asarray(label["projected_keypoints_clipped"], dtype=float)
        outer_color = outer_colors[gate_idx % len(outer_colors)]
        inner_color = inner_colors[gate_idx % len(inner_colors)]
        bbox_pts = np.round(bbox_points).astype(int).reshape(-1, 2)
        keypoint_pts = np.round(keypoints).astype(int).reshape(-1, 2)

        cv2.polylines(preview, [bbox_pts], isClosed=True, color=outer_color, thickness=2)
        if len(keypoint_pts) >= 4 and np.all(keypoint_visibility[:4] == 2):
            cv2.polylines(preview, [keypoint_pts[:4]], isClosed=True, color=inner_color, thickness=2)
        if len(keypoint_pts) >= 8 and np.all(keypoint_visibility[4:8] == 2):
            cv2.polylines(preview, [keypoint_pts[4:8]], isClosed=True, color=outer_color, thickness=2)
        names = KEYPOINT_LAYOUT_NAMES.get(
            KEYPOINT_LAYOUT_INNER4_OUTER4 if len(keypoint_pts) >= 8 else KEYPOINT_LAYOUT_INNER4,
            tuple(f"KP{idx}" for idx in range(len(keypoint_pts))),
        )
        for point_idx, (x, y) in enumerate(keypoint_pts):
            point_color = inner_color if point_idx < 4 else outer_color
            point_name = names[point_idx] if point_idx < len(names) else f"KP{point_idx}"
            if keypoint_visibility[point_idx] == 2:
                cv2.circle(preview, (int(x), int(y)), 4, point_color, -1)
                marker = ""
                text_origin = (int(x) + 4, int(y) - 4)
                text_color = point_color
            elif keypoint_visibility[point_idx] == 1:
                cv2.drawMarker(
                    preview,
                    (int(x), int(y)),
                    (0, 165, 255),
                    markerType=cv2.MARKER_DIAMOND,
                    markerSize=10,
                    thickness=2,
                )
                marker = ":occluded"
                text_origin = (int(x) + 4, int(y) - 4)
                text_color = (0, 165, 255)
            else:
                clipped_x, clipped_y = np.round(keypoints_clipped[point_idx]).astype(int)
                cv2.drawMarker(
                    preview,
                    (int(clipped_x), int(clipped_y)),
                    (0, 0, 255),
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=10,
                    thickness=2,
                )
                marker = ":missing"
                text_origin = (int(clipped_x) + 4, int(clipped_y) - 4)
                text_color = (0, 0, 255)
            cv2.putText(
                preview,
                f"g{gate_idx}:{point_name}{marker}",
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                text_color,
                1,
                cv2.LINE_AA,
            )

    return preview


def write_preview_sheets(preview_paths, output_root, sheet_size=25, thumb_width=320):
    if not preview_paths:
        return 0

    sheets_dir = output_root / "preview_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for sheet_index, start in enumerate(range(0, len(preview_paths), sheet_size)):
        paths = preview_paths[start:start + sheet_size]
        thumbnails = []
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            scale = float(thumb_width) / float(image.shape[1])
            thumb = cv2.resize(
                image,
                (thumb_width, max(1, int(round(image.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA,
            )
            cv2.putText(
                thumb,
                path.name,
                (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            thumbnails.append(thumb)
        if not thumbnails:
            continue
        rows = []
        for row_start in range(0, len(thumbnails), 5):
            row_images = thumbnails[row_start:row_start + 5]
            while len(row_images) < 5:
                row_images.append(np.zeros_like(thumbnails[0]))
            rows.append(cv2.hconcat(row_images))
        sheet = cv2.vconcat(rows)
        cv2.imwrite(str(sheets_dir / f"sheet_{sheet_index:03d}.jpg"), sheet)
        written += 1
    return written


def split_name(index, val_ratio):
    if val_ratio <= 0.0:
        return "train"
    if val_ratio >= 1.0:
        return "val"

    stride = max(int(round(1.0 / val_ratio)), 1)
    return "val" if index % stride == 0 else "train"


def ensure_output_dirs(output_root, draw_preview_enabled):
    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    if draw_preview_enabled:
        (output_root / "previews").mkdir(parents=True, exist_ok=True)


def write_yaml(output_root, keypoint_layout=KEYPOINT_LAYOUT_INNER4):
    keypoint_layout = normalize_keypoint_layout(keypoint_layout)
    keypoint_count = keypoint_layout_count(keypoint_layout)
    flip_idx = ", ".join(str(index) for index in KEYPOINT_LAYOUT_FLIP_IDX[keypoint_layout])
    yaml_path = output_root / "gate_pose.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {output_root}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"kpt_shape: [{keypoint_count}, 3]\n")
        f.write(f"flip_idx: [{flip_idx}]\n")
        f.write("names:\n")
        f.write("  0: gate\n")
    return yaml_path


def print_first_frame_debug(
    metadata,
    candidates,
    selected_gate_idx,
    gazebo_rotation_mode,
    gazebo_optical_mode,
    gate_geometry=None,
):
    raw_roll, raw_pitch, raw_yaw = np.asarray(metadata["drone_rpy_rad"], dtype=float).reshape(3)
    corrected_roll = ROLL_SIGN * float(raw_roll)
    corrected_pitch = PITCH_SIGN * float(raw_pitch)
    corrected_yaw = YAW_SIGN * float(raw_yaw) + YAW_OFFSET_RAD
    print("[AUTOLABEL DEBUG] first processed frame")
    print(f"[AUTOLABEL DEBUG] pose_source_used={metadata_pose_source(metadata)}")
    print(f"[AUTOLABEL DEBUG] gazebo_rotation_mode={gazebo_rotation_mode}")
    print(f"[AUTOLABEL DEBUG] gazebo_optical_mode={gazebo_optical_mode}")
    if gate_geometry is not None:
        print(f"[AUTOLABEL DEBUG] gate_geometry_source={gate_geometry['source']}")
    print(f"[AUTOLABEL DEBUG] gate_centers_frame={metadata_gate_centers_frame(metadata, gate_geometry)}")
    print(f"[AUTOLABEL DEBUG] camera_pos_world={metadata.get('gazebo_camera_pos_world')}")
    print(f"[AUTOLABEL DEBUG] camera_quat_world={metadata.get('gazebo_camera_quat_world')}")
    print(
        "[AUTOLABEL DEBUG] raw_rpy_deg="
        f"[{np.degrees(raw_roll):.3f}, {np.degrees(raw_pitch):.3f}, {np.degrees(raw_yaw):.3f}]"
    )
    print(
        "[AUTOLABEL DEBUG] corrected_rpy_deg="
        f"[{np.degrees(corrected_roll):.3f}, "
        f"{np.degrees(corrected_pitch):.3f}, "
        f"{np.degrees(corrected_yaw):.3f}]"
    )

    for candidate in candidates:
        gate_idx = candidate["gate_idx"]
        depth = candidate["mean_depth"]
        center = np.asarray(candidate["gate_center_world"], dtype=float)
        right_axis = np.asarray(candidate["gate_right_axis_world"], dtype=float)
        corners = np.asarray(candidate["corners_image"], dtype=float)
        clipped_corners = np.asarray(candidate["projected_keypoints_clipped"], dtype=float)
        center_text = np.array2string(center, precision=3, suppress_small=False)
        right_axis_text = np.array2string(right_axis, precision=3, suppress_small=False)
        corners_text = np.array2string(corners, precision=2, suppress_small=False)
        clipped_text = np.array2string(clipped_corners, precision=2, suppress_small=False)
        yolo_visibility_text = np.array2string(
            np.asarray(candidate["keypoint_yolo_visibility"], dtype=int),
            separator=",",
        )
        status = "accepted" if candidate["accepted"] else f"rejected:{candidate['reason']}"
        print(
            f"[AUTOLABEL DEBUG] gate={gate_idx} "
            f"gate_center_world={center_text} "
            f"gate_yaw_rad={candidate['gate_yaw_rad']:.6f} "
            f"gate_right_axis_world={right_axis_text} "
            f"gate_center_distance_m={candidate['gate_center_distance_m']:.3f} "
            f"mean_camera_depth={depth:.3f} "
            f"status={status} "
            f"labeled_keypoint_count={candidate['labeled_keypoint_count']} "
            f"visible_keypoint_count={candidate['visible_keypoint_count']} "
            f"occluded_keypoint_count={candidate['occluded_keypoint_count']} "
            f"keypoint_yolo_visibility={yolo_visibility_text} "
            f"partial_gate_label={candidate['partial_gate_label']} "
            f"rejected_partial_reason={candidate['rejected_partial_reason']} "
            f"projected_keypoints_raw={corners_text} "
            f"projected_keypoints_clipped={clipped_text} "
            f"bbox_clipped={candidate['bbox_clipped']}"
        )
        if gate_idx == 0:
            points_body = candidate.get("points_body", None)
            points_camera = candidate.get("points_camera", None)
            if points_body is not None:
                body_text = np.array2string(np.asarray(points_body, dtype=float), precision=3, suppress_small=False)
                print(f"[AUTOLABEL DEBUG] gate=0 points_body={body_text}")
            if points_camera is not None:
                camera_text = np.array2string(np.asarray(points_camera, dtype=float), precision=3, suppress_small=False)
                print(f"[AUTOLABEL DEBUG] gate=0 points_camera={camera_text}")
            print(f"[AUTOLABEL DEBUG] gate=0 projected_pixels={corners_text}")

    if selected_gate_idx is None:
        print("[AUTOLABEL DEBUG] selected_gate=None")
    else:
        print(f"[AUTOLABEL DEBUG] selected_gate={selected_gate_idx}")


def print_frame_summary_debug(
    frame_name,
    metadata,
    labels,
    gazebo_rotation_mode,
    gazebo_optical_mode,
    gate_geometry=None,
):
    raw_roll, raw_pitch, raw_yaw = np.asarray(metadata["drone_rpy_rad"], dtype=float).reshape(3)
    corrected_roll = ROLL_SIGN * float(raw_roll)
    corrected_pitch = PITCH_SIGN * float(raw_pitch)
    corrected_yaw = YAW_SIGN * float(raw_yaw) + YAW_OFFSET_RAD

    if labels:
        selected_gate_idx = labels[0]["gate_idx"]
        bbox_points = labels[0]["bbox_points_image"]
        bbox_points = np.asarray(bbox_points, dtype=float).reshape(4, 2)
        x_min = float(np.min(bbox_points[:, 0]))
        x_max = float(np.max(bbox_points[:, 0]))
        y_min = float(np.min(bbox_points[:, 1]))
        y_max = float(np.max(bbox_points[:, 1]))
        bbox_center = ((x_min + x_max) * 0.5, (y_min + y_max) * 0.5)
        bbox_size = (x_max - x_min, y_max - y_min)
    else:
        selected_gate_idx = None
        bbox_center = (float("nan"), float("nan"))
        bbox_size = (float("nan"), float("nan"))

    print(
        f"[AUTOLABEL DEBUG] frame={frame_name} "
        f"pose_source={metadata_pose_source(metadata)} "
        f"gazebo_rotation_mode={gazebo_rotation_mode} "
        f"gazebo_optical_mode={gazebo_optical_mode} "
        f"gate_centers_frame={metadata_gate_centers_frame(metadata, gate_geometry)} "
        f"camera_pos_world={metadata.get('gazebo_camera_pos_world')} "
        f"camera_quat_world={metadata.get('gazebo_camera_quat_world')} "
        f"raw_rpy_deg=[{np.degrees(raw_roll):.3f}, {np.degrees(raw_pitch):.3f}, {np.degrees(raw_yaw):.3f}] "
        f"corrected_rpy_deg=[{np.degrees(corrected_roll):.3f}, "
        f"{np.degrees(corrected_pitch):.3f}, {np.degrees(corrected_yaw):.3f}] "
        f"selected_gate={selected_gate_idx} "
        f"bbox_center_px=[{bbox_center[0]:.2f}, {bbox_center[1]:.2f}] "
        f"bbox_size_px=[{bbox_size[0]:.2f}, {bbox_size[1]:.2f}]"
    )


def build_labels(
    metadata,
    label_all_visible_gates=False,
    allow_partial_gates=False,
    min_partial_bbox_area_px2=MIN_PARTIAL_BBOX_AREA_PX2,
    keypoint_layout=KEYPOINT_LAYOUT_INNER4,
    order_image_corners=True,
    gazebo_rotation_mode="direct",
    gazebo_optical_mode="current",
    gate_geometry=None,
    enable_gate_occlusion=True,
    gate_occlusion_padding_m=GATE_OCCLUSION_PADDING_M,
    max_gate_label_distance_m=DEFAULT_MAX_GATE_LABEL_DISTANCE_M,
    debug=False,
):
    keypoint_layout = normalize_keypoint_layout(keypoint_layout)
    keypoint_count = keypoint_layout_count(keypoint_layout)
    width = int(metadata["image_width"])
    height = int(metadata["image_height"])
    labels = []
    candidates = []
    gate_centers = metadata_gate_centers(metadata, gate_geometry)
    gate_yaws = metadata_gate_yaws(metadata, gate_geometry)
    gate_occluders = metadata_gate_occluders(metadata, gate_geometry)
    has_gate_occluders = any(len(boxes) > 0 for boxes in gate_occluders)
    use_gate_occlusion = (
        bool(enable_gate_occlusion)
        and metadata_has_gazebo_pose(metadata)
        and has_gate_occluders
    )
    label_camera_pos_world = metadata_camera_position_world(metadata)
    camera_pos_world = label_camera_pos_world if use_gate_occlusion else None
    max_gate_label_distance_m = float(max_gate_label_distance_m)
    limit_gate_distance = (
        max_gate_label_distance_m > 0.0
        and np.isfinite(max_gate_label_distance_m)
    )

    for gate_idx, (center, yaw) in enumerate(zip(gate_centers, gate_yaws)):
        center_world = np.asarray(center, dtype=float).reshape(3)
        gate_center_distance_m = float(np.linalg.norm(center_world - label_camera_pos_world))
        inner_corners_world = gate_inner_corners_world(center_world, yaw)
        outer_corners_world = gate_outer_corners_world(center_world, yaw)
        points_body = None
        outer_points_body = None
        if metadata_has_gazebo_pose(metadata):
            points_body = gazebo_world_to_camera_body(
                inner_corners_world,
                metadata["gazebo_camera_pos_world"],
                metadata["gazebo_camera_quat_world"],
                gazebo_rotation_mode=gazebo_rotation_mode,
            )
            inner_corners_camera = gazebo_body_to_camera_optical(
                points_body,
                gazebo_optical_mode=gazebo_optical_mode,
            )
            outer_points_body = gazebo_world_to_camera_body(
                outer_corners_world,
                metadata["gazebo_camera_pos_world"],
                metadata["gazebo_camera_quat_world"],
                gazebo_rotation_mode=gazebo_rotation_mode,
            )
            outer_corners_camera = gazebo_body_to_camera_optical(
                outer_points_body,
                gazebo_optical_mode=gazebo_optical_mode,
            )
        else:
            inner_corners_camera = metadata_world_to_camera(
                inner_corners_world,
                metadata,
                gazebo_rotation_mode=gazebo_rotation_mode,
                gazebo_optical_mode=gazebo_optical_mode,
            )
            outer_corners_camera = metadata_world_to_camera(
                outer_corners_world,
                metadata,
                gazebo_rotation_mode=gazebo_rotation_mode,
                gazebo_optical_mode=gazebo_optical_mode,
            )
        mean_depth = float(np.mean(inner_corners_camera[:, 2]))
        accepted = True
        reason = ""
        rejected_partial_reason = ""
        keypoints_image = np.full((keypoint_count, 2), np.nan, dtype=float)
        projected_keypoints_clipped = np.full((keypoint_count, 2), np.nan, dtype=float)
        keypoint_yolo_visibility = np.zeros(keypoint_count, dtype=int)
        keypoint_occluded = np.zeros(keypoint_count, dtype=bool)
        keypoint_occluders = [""] * keypoint_count
        labeled_keypoint_count = 0
        visible_keypoint_count = 0
        occluded_keypoint_count = 0
        bbox_points_image = np.full((4, 2), np.nan, dtype=float)
        bbox_clipped = np.full(4, np.nan, dtype=float)
        partial_gate_label = False

        if limit_gate_distance and gate_center_distance_m > max_gate_label_distance_m:
            accepted = False
            reason = "gate_center_distance_exceeds_limit"
        elif not allow_partial_gates and not np.all(inner_corners_camera[:, 2] > 0.0):
            accepted = False
            reason = "inner_corner_behind_camera"
        elif not allow_partial_gates and not np.all(outer_corners_camera[:, 2] > 0.0):
            accepted = False
            reason = "outer_corner_behind_camera"
        else:
            inner_keypoints_image = project_camera_points(
                inner_corners_camera,
                metadata["camera_matrix"],
                metadata["dist_coeffs"],
            )
            bbox_points_image = project_camera_points(
                outer_corners_camera,
                metadata["camera_matrix"],
                metadata["dist_coeffs"],
            )
            outer_keypoints_image = bbox_points_image.copy()
            if order_image_corners:
                corner_order = order_points_tl_tr_br_bl_indices(inner_keypoints_image)
                inner_keypoints_image = inner_keypoints_image[corner_order]
                inner_corners_world = inner_corners_world[corner_order]
                inner_corners_camera = inner_corners_camera[corner_order]
                if points_body is not None:
                    points_body = points_body[corner_order]
                outer_corner_order = order_points_tl_tr_br_bl_indices(bbox_points_image)
                bbox_points_image = bbox_points_image[outer_corner_order]
                outer_keypoints_image = outer_keypoints_image[outer_corner_order]
                outer_corners_world = outer_corners_world[outer_corner_order]
                outer_corners_camera = outer_corners_camera[outer_corner_order]
                if outer_points_body is not None:
                    outer_points_body = outer_points_body[outer_corner_order]

            inner_keypoint_labeled = points_inside_image(inner_keypoints_image, width, height)
            inner_keypoint_labeled &= np.asarray(inner_corners_camera[:, 2] > 0.0, dtype=bool)
            outer_keypoint_labeled = points_inside_image(outer_keypoints_image, width, height)
            outer_keypoint_labeled &= np.asarray(outer_corners_camera[:, 2] > 0.0, dtype=bool)

            inner_projected_clipped = np.column_stack(
                [
                    np.clip(inner_keypoints_image[:, 0], 0.0, float(width - 1)),
                    np.clip(inner_keypoints_image[:, 1], 0.0, float(height - 1)),
                ]
            )
            outer_projected_clipped = np.column_stack(
                [
                    np.clip(outer_keypoints_image[:, 0], 0.0, float(width - 1)),
                    np.clip(outer_keypoints_image[:, 1], 0.0, float(height - 1)),
                ]
            )
            inner_keypoint_occluded = np.zeros(4, dtype=bool)
            outer_keypoint_occluded = np.zeros(4, dtype=bool)
            inner_keypoint_occluders = [""] * 4
            outer_keypoint_occluders = [""] * 4
            if use_gate_occlusion and camera_pos_world is not None:
                inner_other_occluded, inner_other_occluders = keypoint_gate_occlusion(
                    camera_pos_world,
                    inner_corners_world,
                    target_gate_idx=gate_idx,
                    gate_occluders=gate_occluders,
                    padding_m=gate_occlusion_padding_m,
                )
                inner_self_occluded, inner_self_occluders = keypoint_same_gate_occlusion(
                    camera_pos_world,
                    inner_corners_world,
                    target_gate_idx=gate_idx,
                    gate_occluders=gate_occluders,
                    padding_m=gate_occlusion_padding_m,
                )
                inner_keypoint_occluded, inner_keypoint_occluders = merge_keypoint_occlusions(
                    inner_other_occluded,
                    inner_other_occluders,
                    inner_self_occluded,
                    inner_self_occluders,
                )
                inner_keypoint_occluded &= inner_keypoint_labeled
                outer_other_occluded, outer_other_occluders = keypoint_gate_occlusion(
                    camera_pos_world,
                    outer_corners_world,
                    target_gate_idx=gate_idx,
                    gate_occluders=gate_occluders,
                    padding_m=gate_occlusion_padding_m,
                )
                outer_self_occluded, outer_self_occluders = keypoint_same_gate_occlusion(
                    camera_pos_world,
                    outer_corners_world,
                    target_gate_idx=gate_idx,
                    gate_occluders=gate_occluders,
                    padding_m=gate_occlusion_padding_m,
                )
                outer_keypoint_occluded, outer_keypoint_occluders = merge_keypoint_occlusions(
                    outer_other_occluded,
                    outer_other_occluders,
                    outer_self_occluded,
                    outer_self_occluders,
                )
                outer_keypoint_occluded &= outer_keypoint_labeled

            inner_visibility = np.zeros(4, dtype=int)
            inner_visibility[inner_keypoint_labeled & inner_keypoint_occluded] = 1
            inner_visibility[inner_keypoint_labeled & ~inner_keypoint_occluded] = 2
            outer_visibility = np.zeros(4, dtype=int)
            outer_visibility[outer_keypoint_labeled & outer_keypoint_occluded] = 1
            outer_visibility[outer_keypoint_labeled & ~outer_keypoint_occluded] = 2
            if keypoint_layout == KEYPOINT_LAYOUT_INNER4:
                keypoints_image = inner_keypoints_image
                projected_keypoints_clipped = inner_projected_clipped
                keypoint_yolo_visibility = inner_visibility
                keypoint_occluded = inner_keypoint_occluded
                keypoint_occluders = inner_keypoint_occluders
            else:
                keypoints_image = np.vstack([inner_keypoints_image, outer_keypoints_image])
                projected_keypoints_clipped = np.vstack(
                    [inner_projected_clipped, outer_projected_clipped]
                )
                keypoint_yolo_visibility = np.concatenate([inner_visibility, outer_visibility])
                keypoint_occluded = np.concatenate(
                    [inner_keypoint_occluded, outer_keypoint_occluded]
                )
                keypoint_occluders = inner_keypoint_occluders + outer_keypoint_occluders
            labeled_keypoint_count = int(np.sum(keypoint_yolo_visibility > 0))
            visible_keypoint_count = int(np.sum(keypoint_yolo_visibility == 2))
            occluded_keypoint_count = int(np.sum(keypoint_yolo_visibility == 1))

            if allow_partial_gates:
                partial_gate_label = labeled_keypoint_count < keypoint_count
                if labeled_keypoint_count < 2:
                    accepted = False
                    reason = "fewer_than_2_labeled_keypoints"
                    rejected_partial_reason = reason
                else:
                    positive_outer = bbox_points_image[
                        np.asarray(outer_corners_camera[:, 2] > 0.0, dtype=bool)
                    ]
                    clipped_polygon = (
                        clip_polygon_to_image(positive_outer, width, height)
                        if len(positive_outer) >= 3
                        else np.empty((0, 2), dtype=float)
                    )
                    if len(clipped_polygon) < 3:
                        accepted = False
                        reason = "projected_outer_gate_does_not_intersect_image"
                        rejected_partial_reason = reason
                    else:
                        bbox_points_image = bbox_corners_from_points(clipped_polygon)
                        x_min, y_min = bbox_points_image[0]
                        x_max, y_max = bbox_points_image[2]
                        bbox_clipped = np.array([x_min, y_min, x_max, y_max], dtype=float)
                        bbox_area = float((x_max - x_min) * (y_max - y_min))
                        if bbox_area < float(min_partial_bbox_area_px2):
                            accepted = False
                            reason = "clipped_bbox_area_too_small"
                            rejected_partial_reason = reason
            else:
                if not inside_image(bbox_points_image, width, height):
                    accepted = False
                    reason = "outer_corner_outside_image"
                else:
                    x_min = float(np.min(bbox_points_image[:, 0]))
                    x_max = float(np.max(bbox_points_image[:, 0]))
                    y_min = float(np.min(bbox_points_image[:, 1]))
                    y_max = float(np.max(bbox_points_image[:, 1]))
                    bbox_clipped = np.array([x_min, y_min, x_max, y_max], dtype=float)

        candidate = {
            "keypoint_layout": keypoint_layout,
            "gate_idx": gate_idx,
            "gate_id": gate_idx,
            "race_order": gate_idx,
            "gate_center_world": center_world,
            "gate_center_distance_m": gate_center_distance_m,
            "gate_yaw_rad": float(yaw),
            "gate_right_axis_world": gate_right_axis(float(yaw)),
            "points_body": points_body,
            "points_camera": inner_corners_camera.copy(),
            "bbox_points_image": bbox_points_image,
            "corners_image": keypoints_image,
            "projected_keypoints_raw": keypoints_image.copy(),
            "projected_keypoints_clipped": projected_keypoints_clipped,
            "keypoint_visibility": keypoint_yolo_visibility,
            "keypoint_yolo_visibility": keypoint_yolo_visibility,
            "keypoint_occluded": keypoint_occluded,
            "keypoint_occluders": keypoint_occluders,
            "labeled_keypoint_count": labeled_keypoint_count,
            "visible_keypoint_count": visible_keypoint_count,
            "occluded_keypoint_count": occluded_keypoint_count,
            "partial_gate_label": bool(partial_gate_label),
            "rejected_partial_reason": rejected_partial_reason,
            "bbox_clipped": bbox_clipped,
            "mean_depth": mean_depth,
            "accepted": accepted,
            "reason": reason,
        }
        candidates.append(candidate)

        if accepted:
            labels.append({
                "keypoint_layout": keypoint_layout,
                "gate_idx": gate_idx,
                "gate_id": gate_idx,
                "race_order": gate_idx,
                "gate_center_distance_m": gate_center_distance_m,
                "bbox_points_image": bbox_points_image.copy(),
                "keypoints_image": keypoints_image.copy(),
                "keypoint_visibility": keypoint_yolo_visibility.copy(),
                "keypoint_yolo_visibility": keypoint_yolo_visibility.copy(),
                "keypoint_occluded": keypoint_occluded.copy(),
                "keypoint_occluders": list(keypoint_occluders),
                "projected_keypoints_raw": keypoints_image.copy(),
                "projected_keypoints_clipped": projected_keypoints_clipped.copy(),
                "labeled_keypoint_count": labeled_keypoint_count,
                "visible_keypoint_count": visible_keypoint_count,
                "occluded_keypoint_count": occluded_keypoint_count,
                "partial_gate_label": bool(partial_gate_label),
                "rejected_partial_reason": rejected_partial_reason,
                "bbox_clipped": bbox_clipped.copy(),
                "mean_depth": mean_depth,
            })

    selected_gate_idx = None
    if labels and not label_all_visible_gates:
        selected = min(
            (candidate for candidate in candidates if candidate["accepted"]),
            key=lambda candidate: candidate["mean_depth"],
        )
        selected_gate_idx = selected["gate_idx"]
        labels = [next(label for label in labels if label["gate_idx"] == selected_gate_idx)]
    elif labels:
        selected_gate_idx = "all_visible"

    if debug:
        print_first_frame_debug(
            metadata,
            candidates,
            selected_gate_idx,
            gazebo_rotation_mode,
            gazebo_optical_mode,
            gate_geometry=gate_geometry,
        )

    return labels, candidates


def _json_array(value):
    return json.dumps(np.asarray(value, dtype=float).tolist(), separators=(",", ":"))


def diagnostic_row(
    frame_name,
    candidate,
    *,
    run_id="",
    source_image_filename=None,
    gate_geometry_source="",
):
    return {
        "run_id": run_id,
        "image_filename": frame_name,
        "source_image_filename": source_image_filename or frame_name,
        "gate_geometry_source": gate_geometry_source,
        "keypoint_layout": candidate.get("keypoint_layout", KEYPOINT_LAYOUT_INNER4),
        "gate_id": candidate["gate_id"],
        "race_order": candidate["race_order"],
        "accepted": bool(candidate["accepted"]),
        "rejection_reason": candidate["reason"],
        "labeled_keypoint_count": candidate["labeled_keypoint_count"],
        "visible_keypoint_count": candidate["visible_keypoint_count"],
        "occluded_keypoint_count": candidate["occluded_keypoint_count"],
        "partial_gate_label": bool(candidate["partial_gate_label"]),
        "rejected_partial_reason": candidate["rejected_partial_reason"],
        "gate_center_distance_m": candidate["gate_center_distance_m"],
        "mean_depth_m": candidate["mean_depth"],
        "projected_keypoints_raw": _json_array(candidate["projected_keypoints_raw"]),
        "projected_keypoints_clipped": _json_array(candidate["projected_keypoints_clipped"]),
        "keypoint_visibility": json.dumps(
            np.asarray(candidate["keypoint_visibility"], dtype=int).tolist(),
            separators=(",", ":"),
        ),
        "keypoint_occluders": json.dumps(candidate["keypoint_occluders"], separators=(",", ":")),
        "bbox_clipped": _json_array(candidate["bbox_clipped"]),
    }


def _safe_filename_prefix(value: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    prefix = prefix.strip("._-")
    return prefix or "run"


def _load_manifest(run_root: Path) -> dict:
    manifest_path = run_root / "capture_manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _run_world_sdf_path(args, capture_root: Path, run_root: Path, manifest: dict) -> Path | None:
    if bool(getattr(args, "legacy_hardcoded_gates", False)):
        return None

    if getattr(args, "world_sdf", None):
        return Path(os.path.expanduser(str(args.world_sdf)))

    manifest_snapshot = manifest.get("world_sdf_snapshot")
    if manifest_snapshot:
        snapshot_path = run_root / str(manifest_snapshot)
        if snapshot_path.exists():
            return snapshot_path

    run_snapshot = run_root / "world.sdf"
    if run_snapshot.exists():
        return run_snapshot

    root_snapshot = capture_root / "world.sdf"
    if root_snapshot.exists():
        return root_snapshot

    manifest_original = manifest.get("world_sdf_original_path")
    if manifest_original:
        original_path = Path(os.path.expanduser(str(manifest_original)))
        if original_path.exists():
            return original_path

    return Path(DEFAULT_WORLD_SDF)


def _capture_run_infos(args, capture_root: Path) -> list[dict]:
    capture_root = capture_root.resolve()
    runs = []
    runs_dir = capture_root / "runs"
    if not bool(args.flat_capture_root) and runs_dir.is_dir():
        for run_root in sorted(runs_dir.glob(args.run_glob)):
            if not run_root.is_dir():
                continue
            images_dir = run_root / "images"
            metadata_dir = run_root / "metadata"
            if not images_dir.is_dir() or not metadata_dir.is_dir():
                continue
            manifest = _load_manifest(run_root)
            run_id = str(manifest.get("run_id") or run_root.name)
            runs.append({
                "run_id": run_id,
                "prefix": "" if bool(args.no_run_prefix) else _safe_filename_prefix(run_id),
                "run_root": run_root,
                "images_dir": images_dir,
                "metadata_dir": metadata_dir,
                "world_sdf": _run_world_sdf_path(args, capture_root, run_root, manifest),
                "manifest": manifest,
            })

    if runs:
        return runs

    images_dir = capture_root / "images"
    metadata_dir = capture_root / "metadata"
    if images_dir.is_dir() and metadata_dir.is_dir():
        manifest = _load_manifest(capture_root)
        run_id = str(manifest.get("run_id") or capture_root.name)
        return [{
            "run_id": run_id,
            "prefix": "",
            "run_root": capture_root,
            "images_dir": images_dir,
            "metadata_dir": metadata_dir,
            "world_sdf": _run_world_sdf_path(args, capture_root, capture_root, manifest),
            "manifest": manifest,
        }]

    raise RuntimeError(f"No capture runs or flat images/metadata found in {capture_root}")


def _prefixed_image_name(prefix: str, image_name: str) -> str:
    image_name = Path(image_name).name
    if not prefix:
        return image_name
    return f"{prefix}_{image_name}"


def process_dataset(args):
    capture_root = Path(os.path.expanduser(args.capture_root))
    output_root = Path(os.path.expanduser(args.output_root))
    capture_runs = _capture_run_infos(args, capture_root)

    if not args.preview_only:
        ensure_output_dirs(output_root, args.draw_preview)
        yaml_path = write_yaml(output_root, keypoint_layout=args.keypoint_layout)
    elif args.draw_preview:
        (output_root / "previews").mkdir(parents=True, exist_ok=True)
        yaml_path = None
    else:
        yaml_path = None

    processed = 0
    copied = 0
    labeled_gates = 0
    partial_labeled_gates = 0
    preview_count = 0
    preview_paths = []
    diagnostic_rows = []

    for capture_run in capture_runs:
        run_id = capture_run["run_id"]
        images_dir = capture_run["images_dir"]
        metadata_dir = capture_run["metadata_dir"]
        metadata_paths = sorted(metadata_dir.glob("frame_*.json"))
        if not metadata_paths:
            print(f"[WARN] no metadata files found in {metadata_dir}")
            continue

        gate_geometry = resolve_gate_geometry(args, world_sdf=capture_run["world_sdf"])
        gate_occluder_box_count = sum(
            len(boxes) for boxes in gate_geometry.get("gazebo_occluders", ())
        )
        print(
            f"[AUTOLABEL] run={run_id} frames={len(metadata_paths)} "
            f"gate_geometry_source={gate_geometry['source']} "
            f"gates={len(gate_geometry['gazebo_centers'])} "
            f"gate_occluder_boxes={gate_occluder_box_count}"
        )

        for metadata_path in metadata_paths:
            if args.preview_only and processed >= args.max_preview:
                break

            metadata = load_metadata(metadata_path)
            image_path = images_dir / metadata["image_filename"]
            if not image_path.exists():
                print(f"[WARN] missing image for {metadata_path.name}: {image_path}")
                continue

            output_image_name = _prefixed_image_name(
                capture_run["prefix"],
                metadata["image_filename"],
            )
            labels, candidates = build_labels(
                metadata,
                label_all_visible_gates=args.label_all_visible_gates,
                allow_partial_gates=args.allow_partial_gates,
                min_partial_bbox_area_px2=args.min_partial_bbox_area_px2,
                keypoint_layout=args.keypoint_layout,
                order_image_corners=not args.no_image_order_corners,
                gazebo_rotation_mode=args.gazebo_rotation_mode,
                gazebo_optical_mode=args.gazebo_optical_mode,
                gate_geometry=gate_geometry,
                enable_gate_occlusion=not args.disable_gate_occlusion,
                gate_occlusion_padding_m=args.gate_occlusion_padding_m,
                max_gate_label_distance_m=args.max_gate_label_distance_m,
                debug=(processed == 0),
            )
            diagnostic_rows.extend(
                diagnostic_row(
                    output_image_name,
                    candidate,
                    run_id=run_id,
                    source_image_filename=metadata["image_filename"],
                    gate_geometry_source=gate_geometry["source"],
                )
                for candidate in candidates
            )
            if processed < 10:
                print_frame_summary_debug(
                    output_image_name,
                    metadata,
                    labels,
                    args.gazebo_rotation_mode,
                    args.gazebo_optical_mode,
                    gate_geometry=gate_geometry,
                )

            global_index = processed
            processed += 1
            labeled_gates += len(labels)
            partial_labeled_gates += sum(
                int(label["partial_gate_label"]) for label in labels
            )

            image = None
            if args.draw_preview:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    print(f"[WARN] could not read image for preview: {image_path}")
                else:
                    preview = draw_preview(image, labels)
                    preview_path = output_root / "previews" / output_image_name
                    cv2.imwrite(str(preview_path), preview)
                    preview_paths.append(preview_path)
                    preview_count += 1

            if args.preview_only:
                continue

            split = split_name(global_index, args.val_ratio)
            output_image_path = output_root / "images" / split / output_image_name
            output_label_path = output_root / "labels" / split / f"{Path(output_image_name).stem}.txt"

            shutil.copy2(image_path, output_image_path)
            with open(output_label_path, "w", encoding="utf-8") as f:
                for label in labels:
                    f.write(
                        yolo_pose_line(
                            label["bbox_points_image"],
                            label["keypoints_image"],
                            label["keypoint_visibility"],
                            metadata["image_width"],
                            metadata["image_height"],
                        )
                    )
                    f.write("\n")
            copied += 1

        if args.preview_only and processed >= args.max_preview:
            break

    if processed == 0:
        raise RuntimeError(f"No metadata files processed from {capture_root}")

    preview_sheet_count = (
        write_preview_sheets(preview_paths, output_root)
        if args.draw_preview
        else 0
    )
    diagnostics_path = output_root / "autolabel_diagnostics.csv"
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_fields = [
        "run_id",
        "image_filename",
        "source_image_filename",
        "gate_geometry_source",
        "keypoint_layout",
        "gate_id",
        "race_order",
        "accepted",
        "rejection_reason",
        "labeled_keypoint_count",
        "visible_keypoint_count",
        "occluded_keypoint_count",
        "partial_gate_label",
        "rejected_partial_reason",
        "gate_center_distance_m",
        "mean_depth_m",
        "projected_keypoints_raw",
        "projected_keypoints_clipped",
        "keypoint_visibility",
        "keypoint_occluders",
        "bbox_clipped",
    ]
    with open(diagnostics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=diagnostic_fields)
        writer.writeheader()
        writer.writerows(diagnostic_rows)

    print(
        f"[AUTOLABEL] processed={processed} images_copied={copied} "
        f"labeled_gates={labeled_gates} partial_labeled_gates={partial_labeled_gates} "
        f"previews={preview_count} preview_sheets={preview_sheet_count}"
    )
    print(f"[AUTOLABEL] wrote {diagnostics_path}")
    if yaml_path is not None:
        print(f"[AUTOLABEL] wrote {yaml_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Project known Gazebo gate corners into raw captures and write YOLO pose labels."
    )
    parser.add_argument("--capture-root", default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--world-sdf",
        default=None,
        help=(
            "Gazebo world SDF containing racing_gate_* poses. If omitted, each "
            "capture run uses its own world.sdf snapshot or manifest path."
        ),
    )
    parser.add_argument(
        "--run-glob",
        default="*",
        help="Glob under capture-root/runs for multi-run capture datasets.",
    )
    parser.add_argument(
        "--flat-capture-root",
        action="store_true",
        help="Force legacy capture-root/images and capture-root/metadata layout.",
    )
    parser.add_argument(
        "--no-run-prefix",
        action="store_true",
        help="Do not prefix output filenames with run id. Unsafe for multi-run output unless frame names are unique.",
    )
    parser.add_argument(
        "--legacy-hardcoded-gates",
        action="store_true",
        help="Use the old three hard-coded x500 gate centers/yaws.",
    )
    parser.add_argument(
        "--gate-center-z-offset-m",
        type=float,
        default=GATE_CENTER_Z_OFFSET_M,
        help="Vertical offset from racing_gate_* root pose to inner-opening center.",
    )
    parser.add_argument(
        "--sdf-gate-yaw-offset-rad",
        type=float,
        default=SDF_GATE_YAW_OFFSET_RAD,
        help="Yaw offset from SDF gate root yaw to the labeler right-axis yaw.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--draw-preview", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--max-preview", type=int, default=50)
    parser.add_argument("--label-all-visible-gates", action="store_true")
    parser.add_argument(
        "--keypoint-layout",
        choices=KEYPOINT_LAYOUT_CHOICES,
        default=KEYPOINT_LAYOUT_INNER4,
        help=(
            "YOLO pose keypoint layout. inner4 keeps the existing inner-opening "
            "TL/TR/BR/BL labels. inner4_outer4 appends outer-frame TL/TR/BR/BL "
            "for 8-keypoint training."
        ),
    )
    parser.add_argument(
        "--allow-partial-gates",
        action="store_true",
        help="Label gates with at least two labeled layout keypoints using 0 0 0 for missing points.",
    )
    parser.add_argument(
        "--min-partial-bbox-area-px2",
        type=float,
        default=MIN_PARTIAL_BBOX_AREA_PX2,
    )
    parser.add_argument(
        "--max-gate-label-distance-m",
        type=float,
        default=DEFAULT_MAX_GATE_LABEL_DISTANCE_M,
        help=(
            "Do not label gates whose 3D gate-center distance from the camera "
            "is greater than this many meters. Use <=0 or inf to disable."
        ),
    )
    parser.add_argument(
        "--disable-gate-occlusion",
        action="store_true",
        help="Do not mark keypoints occluded by same-gate or other-gate frame boxes.",
    )
    parser.add_argument(
        "--gate-occlusion-padding-m",
        type=float,
        default=GATE_OCCLUSION_PADDING_M,
        help=(
            "Optional padding added to gate frame boxes during same-gate and "
            "other-gate occlusion ray tests."
        ),
    )
    parser.add_argument("--no-image-order-corners", action="store_true")
    parser.add_argument(
        "--gazebo-rotation-mode",
        choices=("transpose", "direct"),
        default="direct",
    )
    parser.add_argument(
        "--gazebo-optical-mode",
        choices=("current", "flip_y", "physical", "physical_minus_y"),
        default="current",
    )
    args = parser.parse_args()
    if np.isnan(float(args.max_gate_label_distance_m)):
        parser.error("--max-gate-label-distance-m must not be NaN.")
    return args


def main():
    args = parse_args()
    if args.preview_only:
        args.draw_preview = True
    process_dataset(args)


if __name__ == "__main__":
    main()
