#!/usr/bin/env python3
"""Audit camera/IMU rotational consistency in an AIGP OpenVINS replay.

The audit tracks sparse image features, integrates the replay's already
converted body-FRD gyroscope samples between camera timestamps, and ranks a
grid of camera mounting and timing hypotheses.  Translation is fitted
independently for every image pair, so the ranking is based on epipolar
consistency rather than incorrectly treating all optical flow as rotation.

This is a calibration diagnostic only.  It never modifies the replay and it
does not feed ground truth to OpenVINS.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


DEFAULT_REPLAY_NAMES = {
    "highres_imu": "openvins_replay_imuframefix_bracketed",
    "scaled_imu": "openvins_replay_scaled_imu_bracketed",
    "scaled_imu2": "openvins_replay_scaled_imu2_bracketed",
    "scaled_imu3": "openvins_replay_scaled_imu3_bracketed",
    "hil_sensor": "openvins_replay_hil_sensor_bracketed",
}


class AuditError(RuntimeError):
    """The replay cannot support a trustworthy camera/IMU audit."""


@dataclass(frozen=True)
class CameraFrame:
    timestamp_s: float
    path: Path
    source_frame_id: int


@dataclass(frozen=True)
class TrackedPair:
    first_timestamp_s: float
    second_timestamp_s: float
    first_points_px: np.ndarray
    second_points_px: np.ndarray
    median_flow_px: float

    @property
    def track_count(self) -> int:
        return int(self.first_points_px.shape[0])


@dataclass(frozen=True)
class ImuSeries:
    timestamps_s: np.ndarray
    angular_velocity_rad_s: np.ndarray


def camera_to_imu_rotation(tilt_up_deg: float) -> np.ndarray:
    """OpenCV optical camera to MAVLink body/IMU FRD rotation."""

    tilt = math.radians(float(tilt_up_deg))
    sine = math.sin(tilt)
    cosine = math.cos(tilt)
    return np.asarray(
        (
            (0.0, sine, cosine),
            (1.0, 0.0, 0.0),
            (0.0, cosine, -sine),
        ),
        dtype=np.float64,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise AuditError(f"missing required CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AuditError(f"CSV contains no rows: {path}")
    return rows


def _load_manifest(replay: Path) -> dict[str, Any]:
    path = replay / "manifest.json"
    if not path.is_file():
        raise AuditError(f"missing replay manifest: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"unable to read replay manifest {path}: {exc}") from exc
    if manifest.get("format") != "aigp_openvins_replay":
        raise AuditError(f"unsupported replay format in {path}")
    return manifest


def _load_frames(replay: Path) -> list[CameraFrame]:
    rows = _read_csv(replay / "cam0" / "data.csv")
    frames: list[CameraFrame] = []
    for row in rows:
        path = replay / "cam0" / row["filename"]
        if not path.is_file():
            raise AuditError(f"camera image is missing: {path}")
        frames.append(
            CameraFrame(
                timestamp_s=float(row["timestamp"]),
                path=path,
                source_frame_id=int(row.get("source_frame_id", -1)),
            )
        )
    timestamps = np.asarray([frame.timestamp_s for frame in frames])
    if len(frames) < 3 or np.any(np.diff(timestamps) <= 0.0):
        raise AuditError("camera timestamps must contain at least three increasing rows")
    return frames


def _load_imu(replay: Path) -> ImuSeries:
    rows = _read_csv(replay / "imu.csv")
    timestamps = np.asarray([float(row["timestamp"]) for row in rows], dtype=np.float64)
    angular_velocity = np.asarray(
        [
            (float(row["wx"]), float(row["wy"]), float(row["wz"]))
            for row in rows
        ],
        dtype=np.float64,
    )
    if len(rows) < 3 or np.any(np.diff(timestamps) <= 0.0):
        raise AuditError("IMU timestamps must contain at least three increasing rows")
    if not np.all(np.isfinite(angular_velocity)):
        raise AuditError("IMU angular velocity contains non-finite values")
    return ImuSeries(timestamps, angular_velocity)


def _evenly_spaced_start_indices(
    frame_count: int,
    pair_gap: int,
    max_pairs: int,
) -> list[int]:
    possible = frame_count - pair_gap
    if possible <= 0:
        return []
    count = min(possible, max_pairs)
    if count == possible:
        return list(range(possible))
    return sorted(
        {
            int(round(value))
            for value in np.linspace(0, possible - 1, num=count)
        }
    )


def _read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2:
        raise AuditError(f"OpenCV could not decode camera image: {path}")
    return image


def _track_pair(
    first: CameraFrame,
    second: CameraFrame,
    *,
    max_features: int,
    min_feature_distance_px: float,
    max_forward_backward_error_px: float,
    border_margin_px: float,
) -> TrackedPair | None:
    first_image = _read_gray(first.path)
    second_image = _read_gray(second.path)
    if first_image.shape != second_image.shape:
        raise AuditError(
            f"camera resolution changed between {first.path} and {second.path}"
        )

    first_points = cv2.goodFeaturesToTrack(
        first_image,
        maxCorners=int(max_features),
        qualityLevel=0.01,
        minDistance=float(min_feature_distance_px),
        blockSize=7,
        useHarrisDetector=False,
    )
    if first_points is None or len(first_points) == 0:
        return None

    lk = {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    }
    second_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        first_image, second_image, first_points, None, **lk
    )
    if second_points is None or forward_status is None:
        return None
    reverse_points, reverse_status, _ = cv2.calcOpticalFlowPyrLK(
        second_image, first_image, second_points, None, **lk
    )
    if reverse_points is None or reverse_status is None:
        return None

    p0 = first_points.reshape(-1, 2).astype(np.float64)
    p1 = second_points.reshape(-1, 2).astype(np.float64)
    p0_reverse = reverse_points.reshape(-1, 2).astype(np.float64)
    status = forward_status.reshape(-1).astype(bool)
    status &= reverse_status.reshape(-1).astype(bool)
    status &= np.all(np.isfinite(p1), axis=1)
    status &= np.all(np.isfinite(p0_reverse), axis=1)
    status &= np.linalg.norm(p0_reverse - p0, axis=1) <= float(
        max_forward_backward_error_px
    )

    height, width = first_image.shape
    margin = float(border_margin_px)
    status &= p1[:, 0] >= margin
    status &= p1[:, 0] < width - margin
    status &= p1[:, 1] >= margin
    status &= p1[:, 1] < height - margin

    p0 = p0[status]
    p1 = p1[status]
    if len(p0) == 0:
        return None
    return TrackedPair(
        first_timestamp_s=first.timestamp_s,
        second_timestamp_s=second.timestamp_s,
        first_points_px=p0,
        second_points_px=p1,
        median_flow_px=float(np.median(np.linalg.norm(p1 - p0, axis=1))),
    )


def _track_frame_pairs(
    frames: Sequence[CameraFrame],
    *,
    pair_gap: int,
    max_pairs: int,
    max_features: int,
    min_tracks: int,
    min_feature_distance_px: float,
    max_forward_backward_error_px: float,
    border_margin_px: float,
) -> tuple[list[TrackedPair], int]:
    start_indices = _evenly_spaced_start_indices(
        len(frames), pair_gap, max_pairs
    )
    tracked: list[TrackedPair] = []
    for index in start_indices:
        pair = _track_pair(
            frames[index],
            frames[index + pair_gap],
            max_features=max_features,
            min_feature_distance_px=min_feature_distance_px,
            max_forward_backward_error_px=max_forward_backward_error_px,
            border_margin_px=border_margin_px,
        )
        if pair is not None and pair.track_count >= min_tracks:
            tracked.append(pair)
    return tracked, len(start_indices)


def _interpolated_angular_velocity(imu: ImuSeries, timestamp_s: float) -> np.ndarray:
    return np.asarray(
        [
            np.interp(timestamp_s, imu.timestamps_s, imu.angular_velocity_rad_s[:, axis])
            for axis in range(3)
        ],
        dtype=np.float64,
    )


def integrate_imu_frame_rotation(
    imu: ImuSeries,
    start_s: float,
    end_s: float,
) -> np.ndarray:
    """Return the rotation mapping IMU-frame coordinates at start to end."""

    if not start_s < end_s:
        raise AuditError("rotation integration requires start_s < end_s")
    if start_s < imu.timestamps_s[0] or end_s > imu.timestamps_s[-1]:
        raise AuditError("rotation integration interval lies outside the IMU data")

    first_internal = int(np.searchsorted(imu.timestamps_s, start_s, side="right"))
    last_internal = int(np.searchsorted(imu.timestamps_s, end_s, side="left"))
    times = np.concatenate(
        (
            np.asarray([start_s]),
            imu.timestamps_s[first_internal:last_internal],
            np.asarray([end_s]),
        )
    )
    values = np.vstack(
        (
            _interpolated_angular_velocity(imu, start_s),
            imu.angular_velocity_rad_s[first_internal:last_internal],
            _interpolated_angular_velocity(imu, end_s),
        )
    )

    rotation_end_from_start = np.eye(3, dtype=np.float64)
    for index, duration_s in enumerate(np.diff(times)):
        midpoint_rate = 0.5 * (values[index] + values[index + 1])
        step_rotation, _ = cv2.Rodrigues(-midpoint_rate * float(duration_s))
        rotation_end_from_start = step_rotation @ rotation_end_from_start
    return rotation_end_from_start


def _normalized_bearings(
    points_px: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    mirror_x: bool,
) -> np.ndarray:
    x = (points_px[:, 0] - float(cx)) / float(fx)
    if mirror_x:
        # Flipping both the raster and principal point about width - 1 reduces
        # exactly to negating normalized x; no image re-encoding is needed.
        x = -x
    y = (points_px[:, 1] - float(cy)) / float(fy)
    return np.column_stack((x, y, np.ones_like(x)))


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def epipolar_residuals_px(
    first_bearings: np.ndarray,
    second_bearings: np.ndarray,
    rotation_second_from_first: np.ndarray,
    focal_scale_px: float,
) -> np.ndarray:
    """Fit translation and return Sampson residuals for one frame pair."""

    rotated_first = (rotation_second_from_first @ first_bearings.T).T
    design = np.cross(rotated_first, second_bearings)
    normal = design.T @ design
    _, eigenvectors = np.linalg.eigh(normal)
    translation = eigenvectors[:, 0]
    essential = _skew(translation) @ rotation_second_from_first

    ex1 = (essential @ first_bearings.T).T
    etx2 = (essential.T @ second_bearings.T).T
    numerator = np.abs(np.sum(second_bearings * ex1, axis=1))
    denominator = np.sqrt(
        ex1[:, 0] ** 2
        + ex1[:, 1] ** 2
        + etx2[:, 0] ** 2
        + etx2[:, 1] ** 2
    )
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 1e-12)
    return float(focal_scale_px) * numerator[valid] / denominator[valid]


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = float((np.trace(rotation) - 1.0) * 0.5)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _candidate_values(minimum: float, maximum: float, step: float) -> list[float]:
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise AuditError("candidate limits must be finite")
    if step <= 0.0 or not math.isfinite(step):
        raise AuditError("candidate step must be finite and positive")
    if maximum < minimum:
        raise AuditError("candidate maximum must be >= minimum")
    count = int(math.floor((maximum - minimum) / step + 1e-9))
    values = [minimum + index * step for index in range(count + 1)]
    if not math.isclose(values[-1], maximum, abs_tol=step * 1e-6):
        values.append(maximum)
    return [float(value) for value in values]


def _score_candidate(
    pairs: Sequence[TrackedPair],
    imu_rotations: Sequence[np.ndarray | None],
    *,
    rotation_camera_to_imu: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    mirror_x: bool,
    min_rotation_deg: float,
) -> dict[str, Any] | None:
    pair_medians: list[float] = []
    all_residuals: list[np.ndarray] = []
    used_tracks = 0
    rotation_degrees: list[float] = []
    focal_scale = math.sqrt(float(fx) * float(fy))

    for pair, imu_rotation in zip(pairs, imu_rotations):
        if imu_rotation is None:
            continue
        rotation_deg = _rotation_angle_deg(imu_rotation)
        if rotation_deg < min_rotation_deg:
            continue
        camera_rotation = (
            rotation_camera_to_imu.T
            @ imu_rotation
            @ rotation_camera_to_imu
        )
        first = _normalized_bearings(
            pair.first_points_px,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            mirror_x=mirror_x,
        )
        second = _normalized_bearings(
            pair.second_points_px,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            mirror_x=mirror_x,
        )
        residuals = epipolar_residuals_px(
            first, second, camera_rotation, focal_scale
        )
        if residuals.size < 8:
            continue
        pair_medians.append(float(np.median(residuals)))
        all_residuals.append(residuals)
        used_tracks += int(residuals.size)
        rotation_degrees.append(rotation_deg)

    if not pair_medians:
        return None
    combined = np.concatenate(all_residuals)
    return {
        "score_median_pair_sampson_px": float(np.median(pair_medians)),
        "score_p90_pair_sampson_px": float(np.percentile(pair_medians, 90.0)),
        "residual_median_px": float(np.median(combined)),
        "residual_p90_px": float(np.percentile(combined, 90.0)),
        "scored_pairs": len(pair_medians),
        "scored_tracks": used_tracks,
        "rotation_median_deg": float(np.median(rotation_degrees)),
        "rotation_max_deg": float(np.max(rotation_degrees)),
    }


def _confidence_summary(
    ranked: Sequence[dict[str, Any]],
    *,
    tilt_values: Sequence[float],
    offset_values_ms: Sequence[float],
) -> dict[str, Any]:
    if not ranked:
        return {"status": "fail_no_candidates"}
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    different_mirror = next(
        (
            candidate
            for candidate in ranked[1:]
            if candidate["mirror_x"] != best["mirror_x"]
        ),
        None,
    )
    score = float(best["score_median_pair_sampson_px"])
    second_score = (
        float(second["score_median_pair_sampson_px"])
        if second is not None
        else float("nan")
    )
    mirror_score = (
        float(different_mirror["score_median_pair_sampson_px"])
        if different_mirror is not None
        else float("nan")
    )
    at_tilt_boundary = math.isclose(best["tilt_up_deg"], tilt_values[0]) or math.isclose(
        best["tilt_up_deg"], tilt_values[-1]
    )
    at_offset_boundary = math.isclose(
        best["imu_time_offset_ms"], offset_values_ms[0]
    ) or math.isclose(best["imu_time_offset_ms"], offset_values_ms[-1])

    if best["scored_pairs"] < 20:
        status = "inconclusive_too_few_rotating_pairs"
    elif at_tilt_boundary or at_offset_boundary:
        status = "inconclusive_best_at_search_boundary"
    elif not math.isfinite(score) or score > 3.0:
        status = "inconclusive_high_residual"
    else:
        status = "pass_candidate_found"

    return {
        "status": status,
        "best_to_second_score_ratio": (
            score / second_score if second_score > 0.0 else None
        ),
        "best_to_opposite_mirror_score_ratio": (
            score / mirror_score if mirror_score > 0.0 else None
        ),
        "best_at_tilt_boundary": at_tilt_boundary,
        "best_at_time_offset_boundary": at_offset_boundary,
        "notes": [
            "Lower Sampson residual is better.",
            "A best candidate on a search boundary requires a wider follow-up sweep.",
            "This audit estimates rotational extrinsics and timing, not camera translation.",
            "mirror_x means horizontally flip the raster before OpenVINS and use cx' = width - 1 - cx.",
        ],
    }


def audit_replay(
    replay: Path,
    *,
    run_id: str,
    output: Path,
    pair_gap: int = 3,
    max_pairs: int = 180,
    max_features: int = 400,
    min_tracks: int = 25,
    min_feature_distance_px: float = 10.0,
    max_forward_backward_error_px: float = 1.0,
    border_margin_px: float = 4.0,
    min_rotation_deg: float = 0.2,
    tilt_min_deg: float = -30.0,
    tilt_max_deg: float = 30.0,
    tilt_step_deg: float = 5.0,
    offset_min_ms: float = -50.0,
    offset_max_ms: float = 50.0,
    offset_step_ms: float = 2.0,
    top_candidates: int = 20,
) -> dict[str, Any]:
    replay = replay.resolve()
    manifest = _load_manifest(replay)
    calibration = manifest.get("calibration", {})
    intrinsics = calibration.get("intrinsics")
    resolution = calibration.get("resolution")
    if not isinstance(intrinsics, list) or len(intrinsics) != 4:
        raise AuditError("replay manifest is missing four camera intrinsics")
    if not isinstance(resolution, list) or len(resolution) != 2:
        raise AuditError("replay manifest is missing camera resolution")
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    width, height = (int(value) for value in resolution)
    if fx <= 0.0 or fy <= 0.0 or width <= 0 or height <= 0:
        raise AuditError("replay manifest contains invalid camera calibration")

    frames = _load_frames(replay)
    imu = _load_imu(replay)
    tracked_pairs, attempted_pairs = _track_frame_pairs(
        frames,
        pair_gap=pair_gap,
        max_pairs=max_pairs,
        max_features=max_features,
        min_tracks=min_tracks,
        min_feature_distance_px=min_feature_distance_px,
        max_forward_backward_error_px=max_forward_backward_error_px,
        border_margin_px=border_margin_px,
    )
    if len(tracked_pairs) < 10:
        raise AuditError(
            f"only {len(tracked_pairs)} usable image pairs; at least 10 required"
        )

    tilt_values = _candidate_values(tilt_min_deg, tilt_max_deg, tilt_step_deg)
    offset_values_ms = _candidate_values(
        offset_min_ms, offset_max_ms, offset_step_ms
    )
    candidates: list[dict[str, Any]] = []
    for offset_ms in offset_values_ms:
        offset_s = offset_ms / 1000.0
        imu_rotations: list[np.ndarray | None] = []
        for pair in tracked_pairs:
            start_s = pair.first_timestamp_s + offset_s
            end_s = pair.second_timestamp_s + offset_s
            if start_s < imu.timestamps_s[0] or end_s > imu.timestamps_s[-1]:
                imu_rotations.append(None)
                continue
            imu_rotations.append(
                integrate_imu_frame_rotation(imu, start_s, end_s)
            )
        for tilt_deg in tilt_values:
            rotation = camera_to_imu_rotation(tilt_deg)
            for mirror_x in (False, True):
                score = _score_candidate(
                    tracked_pairs,
                    imu_rotations,
                    rotation_camera_to_imu=rotation,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    mirror_x=mirror_x,
                    min_rotation_deg=min_rotation_deg,
                )
                if score is None:
                    continue
                candidates.append(
                    {
                        "mirror_x": mirror_x,
                        "image_transform": "flip_x" if mirror_x else "none",
                        "tilt_up_deg": tilt_deg,
                        "imu_time_offset_ms": offset_ms,
                        "time_convention": "imu_time = camera_time + offset",
                        "R_camera_to_imu_frd": rotation.tolist(),
                        **score,
                    }
                )

    if not candidates:
        raise AuditError("no calibration candidate had usable rotating image pairs")
    candidates.sort(key=lambda item: item["score_median_pair_sampson_px"])
    track_counts = np.asarray([pair.track_count for pair in tracked_pairs], dtype=float)
    median_flows = np.asarray([pair.median_flow_px for pair in tracked_pairs], dtype=float)
    report = {
        "format": "aigp_openvins_camera_imu_audit",
        "format_version": 1,
        "run_id": run_id,
        "replay": str(replay),
        "ground_truth_used": False,
        "method": {
            "feature_detector": "Shi-Tomasi goodFeaturesToTrack",
            "feature_tracker": "pyramidal Lucas-Kanade with forward/backward check",
            "score": "median per-pair Sampson epipolar residual after fitting translation independently per pair",
            "rotation_integration": "midpoint SO(3) integration of replay body-FRD angular velocity",
        },
        "camera": {
            "resolution": [width, height],
            "intrinsics": [fx, fy, cx, cy],
            "mirrored_intrinsics_if_selected": [fx, fy, width - 1.0 - cx, cy],
        },
        "tracking": {
            "pair_gap_frames": pair_gap,
            "attempted_pairs": attempted_pairs,
            "usable_pairs": len(tracked_pairs),
            "min_tracks_required": min_tracks,
            "tracks_total": int(np.sum(track_counts)),
            "tracks_median_per_pair": float(np.median(track_counts)),
            "tracks_p10_per_pair": float(np.percentile(track_counts, 10.0)),
            "tracks_p90_per_pair": float(np.percentile(track_counts, 90.0)),
            "median_flow_px": float(np.median(median_flows)),
        },
        "search": {
            "tilt_up_deg": {
                "minimum": tilt_min_deg,
                "maximum": tilt_max_deg,
                "step": tilt_step_deg,
                "count": len(tilt_values),
            },
            "imu_time_offset_ms": {
                "minimum": offset_min_ms,
                "maximum": offset_max_ms,
                "step": offset_step_ms,
                "count": len(offset_values_ms),
                "convention": "imu_time = camera_time + offset",
            },
            "mirror_x": [False, True],
            "minimum_integrated_rotation_deg": min_rotation_deg,
            "candidate_count": len(candidates),
        },
        "best_candidate": candidates[0],
        "ranked_candidates": candidates[: max(1, int(top_candidates))],
        "confidence": _confidence_summary(
            candidates,
            tilt_values=tilt_values,
            offset_values_ms=offset_values_ms,
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return report


def _default_replay(root: Path, run_id: str, imu_source: str) -> Path:
    try:
        replay_name = DEFAULT_REPLAY_NAMES[imu_source]
    except KeyError as exc:
        supported = ", ".join(sorted(DEFAULT_REPLAY_NAMES))
        raise AuditError(
            f"unsupported IMU source {imu_source!r}; choose one of {supported}"
        ) from exc
    return root / "aigp" / "logs" / "runs" / run_id / replay_name


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run ID under aigp/logs/runs")
    parser.add_argument("--imu-source", default="highres_imu")
    parser.add_argument(
        "--replay",
        type=Path,
        help="explicit replay directory; overrides --imu-source path selection",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pair-gap", type=int, default=3)
    parser.add_argument("--max-pairs", type=int, default=180)
    parser.add_argument("--max-features", type=int, default=400)
    parser.add_argument("--min-tracks", type=int, default=25)
    parser.add_argument("--min-rotation-deg", type=float, default=0.2)
    parser.add_argument("--tilt-min-deg", type=float, default=-30.0)
    parser.add_argument("--tilt-max-deg", type=float, default=30.0)
    parser.add_argument("--tilt-step-deg", type=float, default=5.0)
    parser.add_argument("--offset-min-ms", type=float, default=-50.0)
    parser.add_argument("--offset-max-ms", type=float, default=50.0)
    parser.add_argument("--offset-step-ms", type=float, default=2.0)
    parser.add_argument("--top-candidates", type=int, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.pair_gap < 1:
        print("ERROR: --pair-gap must be at least 1", file=sys.stderr)
        return 2
    if args.max_pairs < 10 or args.max_features < 25 or args.min_tracks < 8:
        print(
            "ERROR: --max-pairs must be >=10, --max-features >=25, and --min-tracks >=8",
            file=sys.stderr,
        )
        return 2

    root = Path(__file__).resolve().parents[2]
    replay = args.replay or _default_replay(root, args.run, args.imu_source)
    output = args.output or replay / "camera_imu_audit_report.json"
    try:
        report = audit_replay(
            replay,
            run_id=args.run,
            output=output,
            pair_gap=args.pair_gap,
            max_pairs=args.max_pairs,
            max_features=args.max_features,
            min_tracks=args.min_tracks,
            min_rotation_deg=args.min_rotation_deg,
            tilt_min_deg=args.tilt_min_deg,
            tilt_max_deg=args.tilt_max_deg,
            tilt_step_deg=args.tilt_step_deg,
            offset_min_ms=args.offset_min_ms,
            offset_max_ms=args.offset_max_ms,
            offset_step_ms=args.offset_step_ms,
            top_candidates=args.top_candidates,
        )
    except (AuditError, cv2.error, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    best = report["best_candidate"]
    tracking = report["tracking"]
    confidence = report["confidence"]
    print(f"Camera/IMU audit report: {output.resolve()}")
    print(
        "Tracked pairs: "
        f"{tracking['usable_pairs']}/{tracking['attempted_pairs']}; "
        f"median tracks {tracking['tracks_median_per_pair']:.1f}"
    )
    print(
        "Best candidate: "
        f"image={best['image_transform']}; "
        f"tilt_up_deg={best['tilt_up_deg']:.3f}; "
        f"imu_time=camera_time{best['imu_time_offset_ms']:+.3f} ms; "
        f"score={best['score_median_pair_sampson_px']:.4f} px"
    )
    print(f"Confidence: {confidence['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
