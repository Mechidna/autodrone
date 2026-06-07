#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CAPTURE_ROOT = "~/datasets/gazebo_gate_capture"
DEFAULT_OUTPUT_ROOT = "~/datasets/gazebo_gate_yolo_pose"

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
MIN_PARTIAL_BBOX_AREA_PX2 = 64.0
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


def gate_inner_corners_world(center_world: np.ndarray, gate_yaw_rad: float) -> np.ndarray:
    center_world = np.asarray(center_world, dtype=float).reshape(3)
    half = INNER_OPENING_M / 2.0

    # yaw=0 means the gate plane is vertical with normal along +world Y.
    right = np.array([np.cos(gate_yaw_rad), np.sin(gate_yaw_rad), 0.0], dtype=float)
    up = np.array([0.0, 0.0, 1.0], dtype=float)

    return np.array([
        center_world - half * right + half * up,  # TL
        center_world + half * right + half * up,  # TR
        center_world + half * right - half * up,  # BR
        center_world - half * right - half * up,  # BL
    ], dtype=float)


def gate_outer_corners_world(center_world: np.ndarray, gate_yaw_rad: float) -> np.ndarray:
    center_world = np.asarray(center_world, dtype=float).reshape(3)
    half = OUTER_GATE_M / 2.0

    right = np.array([np.cos(gate_yaw_rad), np.sin(gate_yaw_rad), 0.0], dtype=float)
    up = np.array([0.0, 0.0, 1.0], dtype=float)

    return np.array([
        center_world - half * right + half * up,  # TL
        center_world + half * right + half * up,  # TR
        center_world + half * right - half * up,  # BR
        center_world - half * right - half * up,  # BL
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


def metadata_gate_centers_frame(metadata):
    return "gazebo" if metadata_has_gazebo_pose(metadata) else "mavsdk"


def metadata_gate_centers(metadata):
    if metadata_has_gazebo_pose(metadata):
        return GATE_CENTERS_WORLD_GAZEBO
    return GATE_CENTERS_WORLD_MAVSDK


def metadata_gate_yaws(metadata):
    if metadata_has_gazebo_pose(metadata):
        return GATE_YAWS_RAD_GAZEBO
    return GATE_YAWS_RAD_MAVSDK


def gate_right_axis(gate_yaw_rad):
    return np.array([np.cos(gate_yaw_rad), np.sin(gate_yaw_rad), 0.0], dtype=float)


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
    by_y = points_image[np.argsort(points_image[:, 1])]
    top = by_y[:2]
    bottom = by_y[2:]
    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]
    tl, tr = top
    bl, br = bottom
    return np.array([tl, tr, br, bl], dtype=float)


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


def yolo_pose_line(
    bbox_points_image,
    keypoints_image,
    keypoint_visibility,
    width,
    height,
    class_id=0,
):
    bbox_points_image = np.asarray(bbox_points_image, dtype=float).reshape(-1, 2)
    keypoints_image = np.asarray(keypoints_image, dtype=float).reshape(4, 2)
    keypoint_visibility = np.asarray(keypoint_visibility, dtype=bool).reshape(4)
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
    for (x, y), visible in zip(keypoints_image, keypoint_visibility):
        if visible:
            values.extend([float(x) / float(width), float(y) / float(height), 2])
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
    names = ("TL", "TR", "BR", "BL")

    for label in labels:
        gate_idx = label["gate_idx"]
        bbox_points = label["bbox_points_image"]
        keypoints = label["keypoints_image"]
        keypoint_visibility = np.asarray(label["keypoint_visibility"], dtype=bool)
        keypoints_clipped = np.asarray(label["projected_keypoints_clipped"], dtype=float)
        outer_color = outer_colors[gate_idx % len(outer_colors)]
        inner_color = inner_colors[gate_idx % len(inner_colors)]
        bbox_pts = np.round(bbox_points).astype(int).reshape(-1, 2)
        keypoint_pts = np.round(keypoints).astype(int).reshape(-1, 2)

        cv2.polylines(preview, [bbox_pts], isClosed=True, color=outer_color, thickness=2)
        visible_points = keypoint_pts[keypoint_visibility]
        if len(visible_points) == 4:
            cv2.polylines(preview, [visible_points], isClosed=True, color=inner_color, thickness=2)
        for point_idx, (x, y) in enumerate(keypoint_pts):
            if keypoint_visibility[point_idx]:
                cv2.circle(preview, (int(x), int(y)), 4, inner_color, -1)
                marker = ""
                text_origin = (int(x) + 4, int(y) - 4)
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
            cv2.putText(
                preview,
                f"g{gate_idx}:{names[point_idx]}{marker}",
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                inner_color if keypoint_visibility[point_idx] else (0, 0, 255),
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


def write_yaml(output_root):
    yaml_path = output_root / "gate_pose.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {output_root}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("kpt_shape: [4, 3]\n")
        f.write("flip_idx: [1, 0, 3, 2]\n")
        f.write("names:\n")
        f.write("  0: gate\n")
    return yaml_path


def print_first_frame_debug(metadata, candidates, selected_gate_idx, gazebo_rotation_mode, gazebo_optical_mode):
    raw_roll, raw_pitch, raw_yaw = np.asarray(metadata["drone_rpy_rad"], dtype=float).reshape(3)
    corrected_roll = ROLL_SIGN * float(raw_roll)
    corrected_pitch = PITCH_SIGN * float(raw_pitch)
    corrected_yaw = YAW_SIGN * float(raw_yaw) + YAW_OFFSET_RAD
    print("[AUTOLABEL DEBUG] first processed frame")
    print(f"[AUTOLABEL DEBUG] pose_source_used={metadata_pose_source(metadata)}")
    print(f"[AUTOLABEL DEBUG] gazebo_rotation_mode={gazebo_rotation_mode}")
    print(f"[AUTOLABEL DEBUG] gazebo_optical_mode={gazebo_optical_mode}")
    print(f"[AUTOLABEL DEBUG] gate_centers_frame={metadata_gate_centers_frame(metadata)}")
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
        status = "accepted" if candidate["accepted"] else f"rejected:{candidate['reason']}"
        print(
            f"[AUTOLABEL DEBUG] gate={gate_idx} "
            f"gate_center_world={center_text} "
            f"gate_yaw_rad={candidate['gate_yaw_rad']:.6f} "
            f"gate_right_axis_world={right_axis_text} "
            f"mean_camera_depth={depth:.3f} "
            f"status={status} "
            f"visible_keypoint_count={candidate['visible_keypoint_count']} "
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


def print_frame_summary_debug(frame_name, metadata, labels, gazebo_rotation_mode, gazebo_optical_mode):
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
        f"gate_centers_frame={metadata_gate_centers_frame(metadata)} "
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
    order_image_corners=True,
    gazebo_rotation_mode="direct",
    gazebo_optical_mode="current",
    debug=False,
):
    width = int(metadata["image_width"])
    height = int(metadata["image_height"])
    labels = []
    candidates = []
    gate_centers = metadata_gate_centers(metadata)
    gate_yaws = metadata_gate_yaws(metadata)

    for gate_idx, (center, yaw) in enumerate(zip(gate_centers, gate_yaws)):
        inner_corners_world = gate_inner_corners_world(center, yaw)
        outer_corners_world = gate_outer_corners_world(center, yaw)
        points_body = None
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
        keypoints_image = np.full((4, 2), np.nan, dtype=float)
        projected_keypoints_clipped = np.full((4, 2), np.nan, dtype=float)
        keypoint_visibility = np.zeros(4, dtype=bool)
        bbox_points_image = np.full((4, 2), np.nan, dtype=float)
        bbox_clipped = np.full(4, np.nan, dtype=float)
        partial_gate_label = False

        if not allow_partial_gates and not np.all(inner_corners_camera[:, 2] > 0.0):
            accepted = False
            reason = "inner_corner_behind_camera"
        elif not allow_partial_gates and not np.all(outer_corners_camera[:, 2] > 0.0):
            accepted = False
            reason = "outer_corner_behind_camera"
        else:
            keypoints_image = project_camera_points(
                inner_corners_camera,
                metadata["camera_matrix"],
                metadata["dist_coeffs"],
            )
            bbox_points_image = project_camera_points(
                outer_corners_camera,
                metadata["camera_matrix"],
                metadata["dist_coeffs"],
            )
            if order_image_corners:
                keypoints_image = order_points_tl_tr_br_bl(keypoints_image)
                bbox_points_image = order_points_tl_tr_br_bl(bbox_points_image)

            keypoint_visibility = points_inside_image(keypoints_image, width, height)
            keypoint_visibility &= np.asarray(inner_corners_camera[:, 2] > 0.0, dtype=bool)
            projected_keypoints_clipped = np.column_stack(
                [
                    np.clip(keypoints_image[:, 0], 0.0, float(width - 1)),
                    np.clip(keypoints_image[:, 1], 0.0, float(height - 1)),
                ]
            )

            if allow_partial_gates:
                visible_keypoint_count = int(np.sum(keypoint_visibility))
                partial_gate_label = visible_keypoint_count < 4
                if visible_keypoint_count < 2:
                    accepted = False
                    reason = "fewer_than_2_visible_keypoints"
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
            "gate_idx": gate_idx,
            "gate_id": gate_idx,
            "race_order": gate_idx,
            "gate_center_world": np.asarray(center, dtype=float).reshape(3),
            "gate_yaw_rad": float(yaw),
            "gate_right_axis_world": gate_right_axis(float(yaw)),
            "points_body": points_body,
            "points_camera": inner_corners_camera.copy(),
            "bbox_points_image": bbox_points_image,
            "corners_image": keypoints_image,
            "projected_keypoints_raw": keypoints_image.copy(),
            "projected_keypoints_clipped": projected_keypoints_clipped,
            "keypoint_visibility": keypoint_visibility,
            "visible_keypoint_count": int(np.sum(keypoint_visibility)),
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
                "gate_idx": gate_idx,
                "gate_id": gate_idx,
                "race_order": gate_idx,
                "bbox_points_image": bbox_points_image.copy(),
                "keypoints_image": keypoints_image.copy(),
                "keypoint_visibility": keypoint_visibility.copy(),
                "projected_keypoints_raw": keypoints_image.copy(),
                "projected_keypoints_clipped": projected_keypoints_clipped.copy(),
                "visible_keypoint_count": int(np.sum(keypoint_visibility)),
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
        )

    return labels, candidates


def _json_array(value):
    return json.dumps(np.asarray(value, dtype=float).tolist(), separators=(",", ":"))


def diagnostic_row(frame_name, candidate):
    return {
        "image_filename": frame_name,
        "gate_id": candidate["gate_id"],
        "race_order": candidate["race_order"],
        "accepted": bool(candidate["accepted"]),
        "rejection_reason": candidate["reason"],
        "visible_keypoint_count": candidate["visible_keypoint_count"],
        "partial_gate_label": bool(candidate["partial_gate_label"]),
        "rejected_partial_reason": candidate["rejected_partial_reason"],
        "mean_depth_m": candidate["mean_depth"],
        "projected_keypoints_raw": _json_array(candidate["projected_keypoints_raw"]),
        "projected_keypoints_clipped": _json_array(candidate["projected_keypoints_clipped"]),
        "keypoint_visibility": json.dumps(
            np.asarray(candidate["keypoint_visibility"], dtype=int).tolist(),
            separators=(",", ":"),
        ),
        "bbox_clipped": _json_array(candidate["bbox_clipped"]),
    }


def process_dataset(args):
    capture_root = Path(os.path.expanduser(args.capture_root))
    images_dir = capture_root / "images"
    metadata_dir = capture_root / "metadata"
    output_root = Path(os.path.expanduser(args.output_root))

    metadata_paths = sorted(metadata_dir.glob("frame_*.json"))
    if args.preview_only:
        metadata_paths = metadata_paths[:args.max_preview]

    if not metadata_paths:
        raise RuntimeError(f"No metadata files found in {metadata_dir}")

    if not args.preview_only:
        ensure_output_dirs(output_root, args.draw_preview)
        yaml_path = write_yaml(output_root)
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

    for index, metadata_path in enumerate(metadata_paths):
        metadata = load_metadata(metadata_path)
        image_path = images_dir / metadata["image_filename"]
        if not image_path.exists():
            print(f"[WARN] missing image for {metadata_path.name}: {image_path}")
            continue

        labels, candidates = build_labels(
            metadata,
            label_all_visible_gates=args.label_all_visible_gates,
            allow_partial_gates=args.allow_partial_gates,
            min_partial_bbox_area_px2=args.min_partial_bbox_area_px2,
            order_image_corners=not args.no_image_order_corners,
            gazebo_rotation_mode=args.gazebo_rotation_mode,
            gazebo_optical_mode=args.gazebo_optical_mode,
            debug=(processed == 0),
        )
        diagnostic_rows.extend(
            diagnostic_row(metadata["image_filename"], candidate)
            for candidate in candidates
        )
        if processed < 10:
            print_frame_summary_debug(
                metadata["image_filename"],
                metadata,
                labels,
                args.gazebo_rotation_mode,
                args.gazebo_optical_mode,
            )
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
                preview_path = output_root / "previews" / image_path.name
                cv2.imwrite(str(preview_path), preview)
                preview_paths.append(preview_path)
                preview_count += 1

        if args.preview_only:
            continue

        split = split_name(index, args.val_ratio)
        output_image_path = output_root / "images" / split / image_path.name
        output_label_path = output_root / "labels" / split / f"{image_path.stem}.txt"

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

    preview_sheet_count = (
        write_preview_sheets(preview_paths, output_root)
        if args.draw_preview
        else 0
    )
    diagnostics_path = output_root / "autolabel_diagnostics.csv"
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_fields = [
        "image_filename",
        "gate_id",
        "race_order",
        "accepted",
        "rejection_reason",
        "visible_keypoint_count",
        "partial_gate_label",
        "rejected_partial_reason",
        "mean_depth_m",
        "projected_keypoints_raw",
        "projected_keypoints_clipped",
        "keypoint_visibility",
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
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--draw-preview", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--max-preview", type=int, default=50)
    parser.add_argument("--label-all-visible-gates", action="store_true")
    parser.add_argument(
        "--allow-partial-gates",
        action="store_true",
        help="Label gates with at least two visible inner keypoints using 0 0 0 for missing points.",
    )
    parser.add_argument(
        "--min-partial-bbox-area-px2",
        type=float,
        default=MIN_PARTIAL_BBOX_AREA_PX2,
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
    return parser.parse_args()


def main():
    args = parse_args()
    if args.preview_only:
        args.draw_preview = True
    process_dataset(args)


if __name__ == "__main__":
    main()
