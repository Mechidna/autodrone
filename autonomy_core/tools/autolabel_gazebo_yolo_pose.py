#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CAPTURE_ROOT = "~/datasets/gazebo_gate_capture"
DEFAULT_OUTPUT_ROOT = "~/datasets/gazebo_gate_yolo_pose"

GATE_CENTERS_WORLD = (
    np.array([0.0, 8.0, 1.5], dtype=float),
    np.array([0.8, 16.0, 1.5], dtype=float),
    np.array([-0.8, 24.0, 1.5], dtype=float),
)
GATE_YAWS_RAD = (0.0, 0.0, 0.0)
INNER_OPENING_M = 1.5
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


def inside_image(points_image, width, height):
    points_image = np.asarray(points_image, dtype=float).reshape(-1, 2)
    x = points_image[:, 0]
    y = points_image[:, 1]
    return bool(np.all((x >= 0.0) & (x < float(width)) & (y >= 0.0) & (y < float(height))))


def yolo_pose_line(points_image, width, height, class_id=0):
    points_image = np.asarray(points_image, dtype=float).reshape(4, 2)
    xs = points_image[:, 0]
    ys = points_image[:, 1]

    x_min = float(np.min(xs))
    x_max = float(np.max(xs))
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))

    cx = ((x_min + x_max) * 0.5) / float(width)
    cy = ((y_min + y_max) * 0.5) / float(height)
    w = (x_max - x_min) / float(width)
    h = (y_max - y_min) / float(height)

    values = [class_id, cx, cy, w, h]
    for x, y in points_image:
        values.extend([float(x) / float(width), float(y) / float(height), 2])

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
    }


def draw_preview(image, labels):
    preview = image.copy()
    colors = ((0, 255, 0), (255, 0, 0), (0, 128, 255))
    names = ("TL", "TR", "BR", "BL")

    for gate_idx, points in labels:
        color = colors[gate_idx % len(colors)]
        pts = np.round(points).astype(int).reshape(-1, 2)
        cv2.polylines(preview, [pts], isClosed=True, color=color, thickness=2)
        for point_idx, (x, y) in enumerate(pts):
            cv2.circle(preview, (int(x), int(y)), 4, color, -1)
            cv2.putText(
                preview,
                f"g{gate_idx}:{names[point_idx]}",
                (int(x) + 4, int(y) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )

    return preview


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


def print_first_frame_debug(metadata, candidates, selected_gate_idx):
    raw_roll, raw_pitch, raw_yaw = np.asarray(metadata["drone_rpy_rad"], dtype=float).reshape(3)
    corrected_roll = ROLL_SIGN * float(raw_roll)
    corrected_pitch = PITCH_SIGN * float(raw_pitch)
    corrected_yaw = YAW_SIGN * float(raw_yaw) + YAW_OFFSET_RAD
    print("[AUTOLABEL DEBUG] first processed frame")
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
        corners = np.asarray(candidate["corners_image"], dtype=float)
        corners_text = np.array2string(corners, precision=2, suppress_small=False)
        status = "accepted" if candidate["accepted"] else f"rejected:{candidate['reason']}"
        print(
            f"[AUTOLABEL DEBUG] gate={gate_idx} "
            f"mean_camera_depth={depth:.3f} "
            f"status={status} "
            f"projected_corners_px={corners_text}"
        )

    if selected_gate_idx is None:
        print("[AUTOLABEL DEBUG] selected_gate=None")
    else:
        print(f"[AUTOLABEL DEBUG] selected_gate={selected_gate_idx}")


def print_frame_summary_debug(frame_name, metadata, labels):
    raw_roll, raw_pitch, raw_yaw = np.asarray(metadata["drone_rpy_rad"], dtype=float).reshape(3)
    corrected_roll = ROLL_SIGN * float(raw_roll)
    corrected_pitch = PITCH_SIGN * float(raw_pitch)
    corrected_yaw = YAW_SIGN * float(raw_yaw) + YAW_OFFSET_RAD

    if labels:
        selected_gate_idx, points_image = labels[0]
        points_image = np.asarray(points_image, dtype=float).reshape(4, 2)
        x_min = float(np.min(points_image[:, 0]))
        x_max = float(np.max(points_image[:, 0]))
        y_min = float(np.min(points_image[:, 1]))
        y_max = float(np.max(points_image[:, 1]))
        bbox_center = ((x_min + x_max) * 0.5, (y_min + y_max) * 0.5)
        bbox_size = (x_max - x_min, y_max - y_min)
    else:
        selected_gate_idx = None
        bbox_center = (float("nan"), float("nan"))
        bbox_size = (float("nan"), float("nan"))

    print(
        f"[AUTOLABEL DEBUG] frame={frame_name} "
        f"raw_rpy_deg=[{np.degrees(raw_roll):.3f}, {np.degrees(raw_pitch):.3f}, {np.degrees(raw_yaw):.3f}] "
        f"corrected_rpy_deg=[{np.degrees(corrected_roll):.3f}, "
        f"{np.degrees(corrected_pitch):.3f}, {np.degrees(corrected_yaw):.3f}] "
        f"selected_gate={selected_gate_idx} "
        f"bbox_center_px=[{bbox_center[0]:.2f}, {bbox_center[1]:.2f}] "
        f"bbox_size_px=[{bbox_size[0]:.2f}, {bbox_size[1]:.2f}]"
    )


def build_labels(metadata, label_all_visible_gates=False, debug=False):
    width = int(metadata["image_width"])
    height = int(metadata["image_height"])
    labels = []
    candidates = []

    for gate_idx, (center, yaw) in enumerate(zip(GATE_CENTERS_WORLD, GATE_YAWS_RAD)):
        corners_world = gate_inner_corners_world(center, yaw)
        corners_camera = world_to_camera(
            corners_world,
            metadata["drone_pos"],
            metadata["drone_rpy_rad"],
        )
        mean_depth = float(np.mean(corners_camera[:, 2]))
        accepted = True
        reason = ""
        corners_image = np.full((4, 2), np.nan, dtype=float)

        if not np.all(corners_camera[:, 2] > 0.0):
            accepted = False
            reason = "corner_behind_camera"
        else:
            corners_image = project_camera_points(
                corners_camera,
                metadata["camera_matrix"],
                metadata["dist_coeffs"],
            )

            if not inside_image(corners_image, width, height):
                accepted = False
                reason = "corner_outside_image"

        candidate = {
            "gate_idx": gate_idx,
            "corners_image": corners_image,
            "mean_depth": mean_depth,
            "accepted": accepted,
            "reason": reason,
        }
        candidates.append(candidate)

        if accepted:
            labels.append((gate_idx, corners_image))

    selected_gate_idx = None
    if labels and not label_all_visible_gates:
        selected = min(
            (candidate for candidate in candidates if candidate["accepted"]),
            key=lambda candidate: candidate["mean_depth"],
        )
        selected_gate_idx = selected["gate_idx"]
        labels = [(selected["gate_idx"], selected["corners_image"])]
    elif labels:
        selected_gate_idx = "all_visible"

    if debug:
        print_first_frame_debug(metadata, candidates, selected_gate_idx)

    return labels


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
    preview_count = 0

    for index, metadata_path in enumerate(metadata_paths):
        metadata = load_metadata(metadata_path)
        image_path = images_dir / metadata["image_filename"]
        if not image_path.exists():
            print(f"[WARN] missing image for {metadata_path.name}: {image_path}")
            continue

        labels = build_labels(
            metadata,
            label_all_visible_gates=args.label_all_visible_gates,
            debug=(processed == 0),
        )
        if processed < 10:
            print_frame_summary_debug(metadata["image_filename"], metadata, labels)
        processed += 1
        labeled_gates += len(labels)

        image = None
        if args.draw_preview:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"[WARN] could not read image for preview: {image_path}")
            else:
                preview = draw_preview(image, labels)
                preview_path = output_root / "previews" / image_path.name
                cv2.imwrite(str(preview_path), preview)
                preview_count += 1

        if args.preview_only:
            continue

        split = split_name(index, args.val_ratio)
        output_image_path = output_root / "images" / split / image_path.name
        output_label_path = output_root / "labels" / split / f"{image_path.stem}.txt"

        shutil.copy2(image_path, output_image_path)
        with open(output_label_path, "w", encoding="utf-8") as f:
            for _, points_image in labels:
                f.write(yolo_pose_line(points_image, metadata["image_width"], metadata["image_height"]))
                f.write("\n")
        copied += 1

    print(
        f"[AUTOLABEL] processed={processed} images_copied={copied} "
        f"labeled_gates={labeled_gates} previews={preview_count}"
    )
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
    return parser.parse_args()


def main():
    args = parse_args()
    if args.preview_only:
        args.draw_preview = True
    process_dataset(args)


if __name__ == "__main__":
    main()
