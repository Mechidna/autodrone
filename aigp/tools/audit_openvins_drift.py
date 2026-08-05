#!/usr/bin/env python3
"""Decompose OpenVINS drift into scale, attitude, velocity, and visual effects.

Captured MAVLink truth is evaluation-only. It defines the same single startup
frame alignment used by ``diagnose_openvins_replay.py`` and is never supplied to
OpenVINS or used to change an estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


_TOOLS_DIR = Path(__file__).resolve().parent
_AIGP_DIR = _TOOLS_DIR.parent
_REPO_ROOT = _AIGP_DIR.parent
for _path in (_TOOLS_DIR, _AIGP_DIR / "pilot", _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from diagnose_openvins_replay import (  # noqa: E402
    DiagnosticError,
    _finite_summary,
    _read_numeric_csv,
    _startup_aligned_position_errors,
    quaternion_xyzw_to_matrix,
    replay_truth_body_to_ned,
    replay_truth_local_ned,
)
from runtime_config import load_runtime_config  # noqa: E402


ESTIMATE_FIELDS = (
    "timestamp",
    "q_GtoI_x",
    "q_GtoI_y",
    "q_GtoI_z",
    "q_GtoI_w",
    "p_IinG_x",
    "p_IinG_y",
    "p_IinG_z",
    "v_IinG_x",
    "v_IinG_y",
    "v_IinG_z",
)
VISUAL_FIELDS = (
    "camera_timestamp",
    "initialized",
    "state_advanced",
    "msckf_accepted_features",
    "slam_features_in_state",
    "active_tracks",
)


class DriftAuditError(RuntimeError):
    """The replay cannot be decomposed safely."""


def _ned_to_neu(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=float).copy()
    output[..., 2] *= -1.0
    return output


def _wrap_angle_rad(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def rotation_matrices_to_euler_zyx(rotations: np.ndarray) -> np.ndarray:
    """Return body-to-world roll, pitch, yaw for non-singular ZYX rotations."""

    rotations = np.asarray(rotations, dtype=float)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise DriftAuditError("rotations must have shape (N, 3, 3)")
    pitch = np.arcsin(np.clip(-rotations[:, 2, 0], -1.0, 1.0))
    roll = np.arctan2(rotations[:, 2, 1], rotations[:, 2, 2])
    yaw = np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0])
    return np.column_stack((roll, pitch, yaw))


def orientation_error_series(
    estimated_body_to_ned: np.ndarray,
    truth_body_to_ned: np.ndarray,
) -> dict[str, np.ndarray]:
    estimated = np.asarray(estimated_body_to_ned, dtype=float)
    truth = np.asarray(truth_body_to_ned, dtype=float)
    if estimated.shape != truth.shape or estimated.shape[1:] != (3, 3):
        raise DriftAuditError("estimated/truth rotation arrays must have equal shape")

    estimate_euler = rotation_matrices_to_euler_zyx(estimated)
    truth_euler = rotation_matrices_to_euler_zyx(truth)
    euler_error_deg = np.degrees(_wrap_angle_rad(estimate_euler - truth_euler))

    relative = np.einsum("nji,njk->nik", truth, estimated)
    traces = np.trace(relative, axis1=1, axis2=2)
    geodesic_deg = np.degrees(
        np.arccos(np.clip((traces - 1.0) * 0.5, -1.0, 1.0))
    )

    truth_forward = truth[:, :, 0]
    estimated_forward = estimated[:, :, 0]
    truth_xy = truth_forward[:, :2]
    estimated_xy = estimated_forward[:, :2]
    truth_xy /= np.maximum(np.linalg.norm(truth_xy, axis=1)[:, None], 1e-12)
    estimated_xy /= np.maximum(
        np.linalg.norm(estimated_xy, axis=1)[:, None], 1e-12
    )
    heading_deg = np.degrees(
        np.arctan2(
            truth_xy[:, 0] * estimated_xy[:, 1]
            - truth_xy[:, 1] * estimated_xy[:, 0],
            np.sum(truth_xy * estimated_xy, axis=1),
        )
    )
    down_dot = np.sum(truth[:, :, 2] * estimated[:, :, 2], axis=1)
    tilt_deg = np.degrees(np.arccos(np.clip(down_dot, -1.0, 1.0)))
    return {
        "roll_deg": euler_error_deg[:, 0],
        "pitch_deg": euler_error_deg[:, 1],
        "yaw_deg": euler_error_deg[:, 2],
        "heading_deg": heading_deg,
        "tilt_deg": tilt_deg,
        "geodesic_deg": geodesic_deg,
    }


def best_position_scale_multiplier(
    estimated_position: np.ndarray,
    truth_position: np.ndarray,
) -> float:
    """Return the no-offset scale multiplying startup-relative VIO position."""

    estimated = np.asarray(estimated_position, dtype=float)
    truth = np.asarray(truth_position, dtype=float)
    estimate_relative = estimated - estimated[0]
    truth_relative = truth - truth[0]
    denominator = float(np.sum(estimate_relative * estimate_relative))
    if denominator <= 1e-12:
        raise DriftAuditError("estimated trajectory has no displacement")
    return float(np.sum(estimate_relative * truth_relative) / denominator)


def vector_angle_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    cosine = np.full(first.shape[0], np.nan, dtype=float)
    valid = denominator > 1e-12
    cosine[valid] = np.sum(first[valid] * second[valid], axis=1) / denominator[valid]
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def longest_true_run(mask: np.ndarray) -> Optional[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    best: Optional[tuple[int, int]] = None
    start: Optional[int] = None
    for index, value in enumerate(np.append(mask, False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            candidate = (start, index - 1)
            if best is None or candidate[1] - candidate[0] > best[1] - best[0]:
                best = candidate
            start = None
    return best


def _stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float).reshape(-1)
    output = _finite_summary(values)
    finite = values[np.isfinite(values)]
    if finite.size:
        output["std"] = float(np.std(finite))
        output["mad"] = float(np.median(np.abs(finite - np.median(finite))))
    return output


def _absolute_stats(values: np.ndarray) -> dict[str, Any]:
    return _stats(np.abs(np.asarray(values, dtype=float)))


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    if abs(denominator) <= 1e-12:
        return None
    return float(numerator / denominator)


def _pearson(first: np.ndarray, second: np.ndarray) -> Optional[float]:
    first = np.asarray(first, dtype=float).reshape(-1)
    second = np.asarray(second, dtype=float).reshape(-1)
    valid = np.isfinite(first) & np.isfinite(second)
    if np.count_nonzero(valid) < 3:
        return None
    first = first[valid]
    second = second[valid]
    if np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _track_tangents(
    timestamp: np.ndarray,
    truth_position: np.ndarray,
    truth_velocity: np.ndarray,
) -> np.ndarray:
    velocity = np.asarray(truth_velocity, dtype=float).copy()
    speed = np.linalg.norm(velocity, axis=1)
    fallback = np.gradient(truth_position, timestamp, axis=0)
    use_fallback = speed < 0.25
    velocity[use_fallback] = fallback[use_fallback]
    norms = np.linalg.norm(velocity, axis=1)
    return velocity / np.maximum(norms[:, None], 1e-12)


def _match_visual_diagnostics(replay: Path, timestamp: np.ndarray) -> dict[str, np.ndarray]:
    path = replay / "openvins_update_diagnostics.csv"
    data = _read_numeric_csv(path, VISUAL_FIELDS)
    initialized = data[:, 1] > 0.5
    data = data[initialized]
    if data.shape[0] < timestamp.size:
        raise DriftAuditError(
            "visual diagnostics contain fewer initialized rows than estimates"
        )
    indices = np.searchsorted(data[:, 0], timestamp)
    indices = np.clip(indices, 0, data.shape[0] - 1)
    left = np.clip(indices - 1, 0, data.shape[0] - 1)
    choose_left = np.abs(data[left, 0] - timestamp) < np.abs(
        data[indices, 0] - timestamp
    )
    indices = np.where(choose_left, left, indices)
    worst_delta = float(np.max(np.abs(data[indices, 0] - timestamp)))
    if worst_delta > 1e-6:
        raise DriftAuditError(
            "visual diagnostics do not match estimate timestamps; worst delta "
            f"is {worst_delta * 1000.0:.3f} ms"
        )
    matched = data[indices]
    return {
        "state_advanced": matched[:, 2],
        "accepted_features": matched[:, 3],
        "slam_features": matched[:, 4],
        "active_tracks": matched[:, 5],
        "worst_timestamp_delta_ms": np.asarray(worst_delta * 1000.0),
    }


def _load_trajectory(replay: Path) -> dict[str, Any]:
    estimate = _read_numeric_csv(replay / "openvins_estimate.csv", ESTIMATE_FIELDS)
    timestamp = estimate[:, 0]
    if np.any(np.diff(timestamp) <= 0.0):
        raise DriftAuditError("estimate timestamps are not strictly increasing")
    truth = replay_truth_local_ned(replay, timestamp)
    if truth is None:
        raise DriftAuditError("captured local-position truth is unavailable")
    truth_position, truth_velocity, translation_method = truth
    fallback = np.tile((0.0, 0.0, 0.0, 1.0), (estimate.shape[0], 1))
    truth_body_to_ned, orientation_method = replay_truth_body_to_ned(
        replay, timestamp, fallback
    )
    estimate_global_to_imu = np.transpose(
        quaternion_xyzw_to_matrix(estimate[:, 1:5]), (0, 2, 1)
    )
    estimated_position, _ = _startup_aligned_position_errors(
        estimate[:, 5:8],
        estimate_global_to_imu,
        truth_position,
        truth_body_to_ned,
    )
    rotation_global_to_ned = truth_body_to_ned[0] @ estimate_global_to_imu[0]
    estimated_body_to_ned = np.einsum(
        "ij,njk->nik",
        rotation_global_to_ned,
        np.transpose(estimate_global_to_imu, (0, 2, 1)),
    )
    estimated_velocity = np.einsum(
        "ij,nj->ni", rotation_global_to_ned, estimate[:, 8:11]
    )
    return {
        "timestamp": timestamp,
        "estimated_position_ned": estimated_position,
        "truth_position_ned": truth_position,
        "estimated_velocity_ned": estimated_velocity,
        "truth_velocity_ned": truth_velocity,
        "estimated_body_to_ned": estimated_body_to_ned,
        "truth_body_to_ned": truth_body_to_ned,
        "translation_method": translation_method,
        "orientation_method": orientation_method,
        "startup_rotation_global_to_ned": rotation_global_to_ned,
    }


def compute_window_metrics(
    timestamp: np.ndarray,
    estimated_position: np.ndarray,
    truth_position: np.ndarray,
    position_error: np.ndarray,
    visual: dict[str, np.ndarray],
    *,
    window_s: float,
    min_displacement_m: float,
) -> list[dict[str, Any]]:
    rows = []
    for end in range(1, timestamp.size):
        start = int(np.searchsorted(timestamp, timestamp[end] - window_s))
        duration = float(timestamp[end] - timestamp[start])
        if duration < 0.8 * window_s:
            continue
        estimate_delta = estimated_position[end] - estimated_position[start]
        truth_delta = truth_position[end] - truth_position[start]
        estimate_distance = float(np.linalg.norm(estimate_delta))
        truth_distance = float(np.linalg.norm(truth_delta))
        if truth_distance < min_displacement_m:
            continue
        direction_error = float(
            vector_angle_deg(estimate_delta[None, :], truth_delta[None, :])[0]
        )
        selected = slice(start, end + 1)
        rows.append(
            {
                "start_index": start,
                "end_index": end,
                "start_timestamp_s": float(timestamp[start]),
                "end_timestamp_s": float(timestamp[end]),
                "duration_s": duration,
                "estimated_displacement_m": estimate_distance,
                "truth_displacement_m": truth_distance,
                "estimated_to_truth_scale": estimate_distance / truth_distance,
                "direction_error_deg": direction_error,
                "position_error_growth_m": float(
                    position_error[end] - position_error[start]
                ),
                "accepted_features": int(
                    np.sum(visual["accepted_features"][selected])
                ),
                "mean_active_tracks": float(
                    np.mean(visual["active_tracks"][selected])
                ),
                "active_track_zero_fraction": float(
                    np.mean(visual["active_tracks"][selected] == 0)
                ),
                "mean_slam_features": float(
                    np.mean(visual["slam_features"][selected])
                ),
            }
        )
    return rows


def _source_config(replay: Path) -> Optional[Path]:
    snapshot = replay.parent / "vio_dataset" / "runtime.toml"
    if snapshot.is_file():
        return snapshot
    current = _AIGP_DIR / "config" / "runtime.toml"
    return current if current.is_file() else None


def _gate_checkpoints(
    replay: Path,
    timestamp: np.ndarray,
    truth_position_ned: np.ndarray,
    position_error: np.ndarray,
    cumulative_scale: np.ndarray,
    orientation: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], Optional[str]]:
    config_path = _source_config(replay)
    if config_path is None:
        return [], None
    config = load_runtime_config(config_path)
    gates_neu = np.asarray(config.gate_source.known_gate_positions_neu, dtype=float)
    if gates_neu.size == 0:
        return [], str(config_path)
    truth_neu = _ned_to_neu(truth_position_ned)
    checkpoints = []
    for gate_index, gate in enumerate(gates_neu.reshape(-1, 3)):
        distances = np.linalg.norm(truth_neu - gate, axis=1)
        index = int(np.argmin(distances))
        checkpoints.append(
            {
                "gate_index": gate_index,
                "gate_position_neu_m": gate.tolist(),
                "timestamp_s": float(timestamp[index]),
                "closest_truth_distance_m": float(distances[index]),
                "position_error_m": float(position_error[index]),
                "cumulative_path_scale": (
                    None
                    if not math.isfinite(cumulative_scale[index])
                    else float(cumulative_scale[index])
                ),
                "heading_error_deg": float(orientation["heading_deg"][index]),
                "tilt_error_deg": float(orientation["tilt_deg"][index]),
            }
        )
    return checkpoints, str(config_path)


def _run_summary(
    run: Optional[tuple[int, int]],
    timestamp: np.ndarray,
    position_error: np.ndarray,
) -> Optional[dict[str, Any]]:
    if run is None:
        return None
    start, end = run
    return {
        "start_index": start,
        "end_index": end,
        "rows": end - start + 1,
        "start_timestamp_s": float(timestamp[start]),
        "end_timestamp_s": float(timestamp[end]),
        "duration_s": float(timestamp[end] - timestamp[start]),
        "position_error_change_m": float(
            position_error[end] - position_error[start]
        ),
    }


def _classify(
    *,
    endpoint_scale: float,
    path_scale: float,
    local_scale: dict[str, Any],
    scale_improvement_fraction: float,
    heading_p95_deg: float,
    tilt_p95_deg: float,
    velocity_direction_p95_deg: float,
    dropout_growth_ratio: Optional[float],
) -> dict[str, Any]:
    local_median = float(local_scale.get("median", math.nan))
    scale_deviation = max(
        abs(endpoint_scale - 1.0),
        abs(path_scale - 1.0),
        abs(local_median - 1.0) if math.isfinite(local_median) else 0.0,
    )
    scale_material = bool(scale_deviation >= 0.05)
    scale_explains_error = bool(scale_improvement_fraction >= 0.25)
    attitude_material = bool(
        heading_p95_deg >= 5.0 or tilt_p95_deg >= 5.0
    )
    velocity_direction_material = bool(velocity_direction_p95_deg >= 10.0)
    dropout_coupled = bool(
        dropout_growth_ratio is not None and dropout_growth_ratio >= 2.0
    )

    if scale_material and attitude_material:
        dominant = "mixed_scale_and_attitude_drift"
        next_experiment = (
            "sweep camera intrinsics first, then camera-to-IMU rotation; "
            "do not combine both changes in one replay"
        )
    elif scale_material and scale_explains_error:
        dominant = "systematic_metric_scale_error"
        next_experiment = (
            "run a controlled focal-length/intrinsics sweep with all other "
            "OpenVINS settings fixed"
        )
    elif attitude_material or velocity_direction_material:
        dominant = "attitude_or_extrinsic_drift"
        next_experiment = (
            "run a controlled camera-to-IMU rotation/convention sweep with "
            "intrinsics fixed"
        )
    elif dropout_coupled:
        dominant = "visual_tracking_dropout_coupled_drift"
        next_experiment = (
            "inspect and replay the longest active-track dropout intervals"
        )
    elif scale_material:
        dominant = "scale_present_but_not_sufficient"
        next_experiment = (
            "inspect time-varying direction and bias errors before applying a "
            "single scale correction"
        )
    else:
        dominant = "mixed_or_unresolved_drift"
        next_experiment = "inspect the highest-error-growth windows individually"
    return {
        "dominant_error": dominant,
        "systematic_scale_material": scale_material,
        "single_scale_counterfactual_explains_at_least_25_percent": (
            scale_explains_error
        ),
        "attitude_error_material": attitude_material,
        "velocity_direction_error_material": velocity_direction_material,
        "active_track_dropout_coupled": dropout_coupled,
        "recommended_next_experiment": next_experiment,
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        result = value.item()
        if isinstance(result, float) and not math.isfinite(result):
            return None
        return result
    raise TypeError(f"cannot serialize {type(value).__name__}")


def audit_replay(
    replay: Path,
    *,
    output_dir: Optional[Path] = None,
    window_s: float = 2.0,
    min_window_displacement_m: float = 1.0,
    min_velocity_m_s: float = 0.5,
) -> dict[str, Any]:
    replay = replay.resolve()
    if not replay.is_dir():
        raise DriftAuditError(f"replay directory does not exist: {replay}")
    if window_s <= 0.0 or min_window_displacement_m <= 0.0:
        raise DriftAuditError("window and displacement thresholds must be positive")
    output_dir = (
        output_dir.resolve()
        if output_dir is not None
        else replay / "drift_decomposition_audit"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory = _load_trajectory(replay)
    timestamp = trajectory["timestamp"]
    estimated_position = trajectory["estimated_position_ned"]
    truth_position = trajectory["truth_position_ned"]
    estimated_velocity = trajectory["estimated_velocity_ned"]
    truth_velocity = trajectory["truth_velocity_ned"]
    visual = _match_visual_diagnostics(replay, timestamp)
    orientation = orientation_error_series(
        trajectory["estimated_body_to_ned"], trajectory["truth_body_to_ned"]
    )

    position_error_vector = estimated_position - truth_position
    position_error = np.linalg.norm(position_error_vector, axis=1)
    tangents = _track_tangents(
        timestamp, truth_position, truth_velocity
    )
    along_error = np.sum(position_error_vector * tangents, axis=1)
    cross_error = np.linalg.norm(
        position_error_vector - along_error[:, None] * tangents, axis=1
    )

    estimate_relative = estimated_position - estimated_position[0]
    truth_relative = truth_position - truth_position[0]
    estimate_displacement = np.linalg.norm(estimate_relative, axis=1)
    truth_displacement = np.linalg.norm(truth_relative, axis=1)
    displacement_scale = np.full(timestamp.size, np.nan)
    displaced = truth_displacement >= min_window_displacement_m
    displacement_scale[displaced] = (
        estimate_displacement[displaced] / truth_displacement[displaced]
    )
    displacement_direction_error = vector_angle_deg(
        estimate_relative, truth_relative
    )

    estimate_steps = np.linalg.norm(np.diff(estimated_position, axis=0), axis=1)
    truth_steps = np.linalg.norm(np.diff(truth_position, axis=0), axis=1)
    estimate_path = np.concatenate(([0.0], np.cumsum(estimate_steps)))
    truth_path = np.concatenate(([0.0], np.cumsum(truth_steps)))
    cumulative_scale = np.full(timestamp.size, np.nan)
    path_valid = truth_path >= min_window_displacement_m
    cumulative_scale[path_valid] = estimate_path[path_valid] / truth_path[path_valid]
    endpoint_scale = estimate_displacement[-1] / truth_displacement[-1]
    path_scale = estimate_path[-1] / truth_path[-1]

    scale_multiplier = best_position_scale_multiplier(
        estimated_position, truth_position
    )
    scaled_position = truth_position[0] + scale_multiplier * estimate_relative
    scaled_error = np.linalg.norm(scaled_position - truth_position, axis=1)
    raw_p95 = float(np.percentile(position_error, 95.0))
    scaled_p95 = float(np.percentile(scaled_error, 95.0))
    scale_improvement_fraction = max(
        0.0, (raw_p95 - scaled_p95) / max(raw_p95, 1e-12)
    )

    estimate_speed = np.linalg.norm(estimated_velocity, axis=1)
    truth_speed = np.linalg.norm(truth_velocity, axis=1)
    moving = (estimate_speed >= min_velocity_m_s) & (
        truth_speed >= min_velocity_m_s
    )
    speed_scale = np.full(timestamp.size, np.nan)
    speed_scale[moving] = estimate_speed[moving] / truth_speed[moving]
    velocity_direction_error = vector_angle_deg(
        estimated_velocity, truth_velocity
    )
    velocity_direction_error[~moving] = np.nan
    truth_velocity_unit = truth_velocity / np.maximum(
        truth_speed[:, None], 1e-12
    )
    velocity_error_vector = estimated_velocity - truth_velocity
    velocity_along_error = np.sum(
        velocity_error_vector * truth_velocity_unit, axis=1
    )
    velocity_cross_error = np.linalg.norm(
        velocity_error_vector
        - velocity_along_error[:, None] * truth_velocity_unit,
        axis=1,
    )
    velocity_along_error[~moving] = np.nan
    velocity_cross_error[~moving] = np.nan

    windows = compute_window_metrics(
        timestamp,
        estimated_position,
        truth_position,
        position_error,
        visual,
        window_s=window_s,
        min_displacement_m=min_window_displacement_m,
    )
    window_scale = np.asarray(
        [row["estimated_to_truth_scale"] for row in windows], dtype=float
    )
    window_direction = np.asarray(
        [row["direction_error_deg"] for row in windows], dtype=float
    )
    window_growth = np.asarray(
        [row["position_error_growth_m"] for row in windows], dtype=float
    )
    window_active = np.asarray(
        [row["mean_active_tracks"] for row in windows], dtype=float
    )
    window_accepted = np.asarray(
        [row["accepted_features"] for row in windows], dtype=float
    )

    dt = np.diff(timestamp)
    error_growth_rate = np.zeros(timestamp.size, dtype=float)
    error_growth_rate[1:] = np.diff(position_error) / dt
    positive_growth_rate = np.maximum(error_growth_rate, 0.0)
    active_dropout = visual["active_tracks"] <= 0.0
    accepted_zero = visual["accepted_features"] <= 0.0
    dropout_growth = positive_growth_rate[active_dropout]
    tracked_growth = positive_growth_rate[~active_dropout]
    dropout_growth_ratio = _safe_ratio(
        float(np.mean(dropout_growth)) if dropout_growth.size else math.nan,
        float(np.mean(tracked_growth)) if tracked_growth.size else math.nan,
    )
    longest_active_dropout = _run_summary(
        longest_true_run(active_dropout), timestamp, position_error
    )
    longest_accepted_zero = _run_summary(
        longest_true_run(accepted_zero), timestamp, position_error
    )

    checkpoints, config_path = _gate_checkpoints(
        replay,
        timestamp,
        truth_position,
        position_error,
        cumulative_scale,
        orientation,
    )

    local_scale_summary = _stats(window_scale)
    heading_abs = _absolute_stats(orientation["heading_deg"])
    tilt_abs = _absolute_stats(orientation["tilt_deg"])
    velocity_direction_summary = _stats(velocity_direction_error)
    classification = _classify(
        endpoint_scale=endpoint_scale,
        path_scale=path_scale,
        local_scale=local_scale_summary,
        scale_improvement_fraction=scale_improvement_fraction,
        heading_p95_deg=float(heading_abs["p95"]),
        tilt_p95_deg=float(tilt_abs["p95"]),
        velocity_direction_p95_deg=float(velocity_direction_summary["p95"]),
        dropout_growth_ratio=dropout_growth_ratio,
    )

    first_half = np.flatnonzero(position_error >= 0.5)
    first_two = np.flatnonzero(position_error >= 2.0)
    timeseries_rows = []
    for index in range(timestamp.size):
        timeseries_rows.append(
            {
                "timestamp": f"{timestamp[index]:.9f}",
                "estimated_x_ned": estimated_position[index, 0],
                "estimated_y_ned": estimated_position[index, 1],
                "estimated_z_ned": estimated_position[index, 2],
                "truth_x_ned": truth_position[index, 0],
                "truth_y_ned": truth_position[index, 1],
                "truth_z_ned": truth_position[index, 2],
                "position_error_m": position_error[index],
                "position_error_x_ned_m": position_error_vector[index, 0],
                "position_error_y_ned_m": position_error_vector[index, 1],
                "position_error_z_ned_m": position_error_vector[index, 2],
                "along_track_error_m": along_error[index],
                "cross_track_error_m": cross_error[index],
                "displacement_scale": _csv_value(displacement_scale[index]),
                "cumulative_path_scale": _csv_value(cumulative_scale[index]),
                "displacement_direction_error_deg": _csv_value(
                    displacement_direction_error[index]
                ),
                "roll_error_deg": orientation["roll_deg"][index],
                "pitch_error_deg": orientation["pitch_deg"][index],
                "yaw_error_deg": orientation["yaw_deg"][index],
                "heading_error_deg": orientation["heading_deg"][index],
                "tilt_error_deg": orientation["tilt_deg"][index],
                "attitude_geodesic_error_deg": orientation["geodesic_deg"][index],
                "estimated_speed_m_s": estimate_speed[index],
                "truth_speed_m_s": truth_speed[index],
                "speed_scale": _csv_value(speed_scale[index]),
                "velocity_direction_error_deg": _csv_value(
                    velocity_direction_error[index]
                ),
                "velocity_along_error_m_s": _csv_value(
                    velocity_along_error[index]
                ),
                "velocity_cross_error_m_s": _csv_value(
                    velocity_cross_error[index]
                ),
                "msckf_accepted_features": int(
                    visual["accepted_features"][index]
                ),
                "slam_features_in_state": int(visual["slam_features"][index]),
                "active_tracks": int(visual["active_tracks"][index]),
                "active_track_dropout": int(active_dropout[index]),
                "position_error_growth_rate_m_s": error_growth_rate[index],
            }
        )

    timeseries_path = output_dir / "openvins_drift_timeseries.csv"
    windows_path = output_dir / "openvins_drift_windows.csv"
    report_path = output_dir / "openvins_drift_audit_report.json"
    _write_csv(timeseries_path, tuple(timeseries_rows[0]), timeseries_rows)
    if windows:
        _write_csv(windows_path, tuple(windows[0]), windows)
    else:
        _write_csv(
            windows_path,
            (
                "start_index",
                "end_index",
                "start_timestamp_s",
                "end_timestamp_s",
                "duration_s",
                "estimated_displacement_m",
                "truth_displacement_m",
                "estimated_to_truth_scale",
                "direction_error_deg",
                "position_error_growth_m",
                "accepted_features",
                "mean_active_tracks",
                "active_track_zero_fraction",
                "mean_slam_features",
            ),
            [],
        )

    report = {
        "format": "aigp_openvins_drift_decomposition",
        "format_version": 1,
        "replay": str(replay),
        "status": "fail" if float(np.percentile(position_error, 95.0)) > 0.5 else "pass",
        "classification": classification,
        "truth_separation": {
            "truth_fed_to_openvins": False,
            "truth_used_for_single_startup_frame_alignment": True,
            "truth_used_for_audit_only": True,
            "trajectory_wide_alignment_or_scale_fed_to_estimator": False,
        },
        "samples": {
            "rows": int(timestamp.size),
            "first_timestamp_s": float(timestamp[0]),
            "last_timestamp_s": float(timestamp[-1]),
            "duration_s": float(timestamp[-1] - timestamp[0]),
        },
        "position": {
            "error_m": _stats(position_error),
            "final_error_m": float(position_error[-1]),
            "first_error_over_0_5_m_timestamp_s": (
                None if first_half.size == 0 else float(timestamp[first_half[0]])
            ),
            "first_error_over_2_m_timestamp_s": (
                None if first_two.size == 0 else float(timestamp[first_two[0]])
            ),
            "axis_absolute_error_m": {
                "x_ned": _absolute_stats(position_error_vector[:, 0]),
                "y_ned": _absolute_stats(position_error_vector[:, 1]),
                "z_ned": _absolute_stats(position_error_vector[:, 2]),
            },
            "along_track_signed_error_m": _stats(along_error),
            "along_track_absolute_error_m": _absolute_stats(along_error),
            "cross_track_error_m": _stats(cross_error),
            "final_error_ned_m": position_error_vector[-1].tolist(),
        },
        "scale": {
            "endpoint_displacement_m": {
                "estimated": float(estimate_displacement[-1]),
                "truth": float(truth_displacement[-1]),
                "estimated_to_truth_ratio": float(endpoint_scale),
            },
            "path_length_m": {
                "estimated": float(estimate_path[-1]),
                "truth": float(truth_path[-1]),
                "estimated_to_truth_ratio": float(path_scale),
            },
            "startup_displacement_ratio": _stats(displacement_scale),
            "window_duration_s": window_s,
            "window_displacement_ratio": local_scale_summary,
            "window_displacement_direction_error_deg": _stats(window_direction),
            "best_single_scale_counterfactual": {
                "multiplier_applied_to_openvins": scale_multiplier,
                "uncorrected_position_error_p95_m": raw_p95,
                "scaled_position_error_p95_m": scaled_p95,
                "p95_improvement_fraction": scale_improvement_fraction,
                "scaled_position_error_m": _stats(scaled_error),
                "evaluation_only": True,
            },
        },
        "attitude": {
            "roll_absolute_error_deg": _absolute_stats(orientation["roll_deg"]),
            "pitch_absolute_error_deg": _absolute_stats(orientation["pitch_deg"]),
            "yaw_absolute_error_deg": _absolute_stats(orientation["yaw_deg"]),
            "heading_absolute_error_deg": heading_abs,
            "tilt_absolute_error_deg": tilt_abs,
            "geodesic_error_deg": _stats(orientation["geodesic_deg"]),
            "final_signed_error_deg": {
                "roll": float(orientation["roll_deg"][-1]),
                "pitch": float(orientation["pitch_deg"][-1]),
                "yaw": float(orientation["yaw_deg"][-1]),
                "heading": float(orientation["heading_deg"][-1]),
            },
        },
        "velocity": {
            "moving_threshold_m_s": min_velocity_m_s,
            "moving_samples": int(np.count_nonzero(moving)),
            "speed_ratio": _stats(speed_scale),
            "speed_absolute_error_m_s": _absolute_stats(
                estimate_speed - truth_speed
            ),
            "direction_error_deg": velocity_direction_summary,
            "along_track_absolute_error_m_s": _absolute_stats(
                velocity_along_error
            ),
            "cross_track_error_m_s": _stats(velocity_cross_error),
        },
        "visual_coupling": {
            "semantics": (
                "active_tracks=0 is treated as a tracking dropout; zero accepted "
                "MSCKF features alone is not, because track updates are batched"
            ),
            "active_track_dropout_rows": int(np.count_nonzero(active_dropout)),
            "active_track_dropout_fraction": float(np.mean(active_dropout)),
            "accepted_feature_zero_rows": int(np.count_nonzero(accepted_zero)),
            "accepted_feature_zero_fraction": float(np.mean(accepted_zero)),
            "longest_active_track_dropout": longest_active_dropout,
            "longest_accepted_feature_zero_run": longest_accepted_zero,
            "positive_error_growth_rate_m_s": {
                "during_active_track_dropout": _stats(dropout_growth),
                "while_active_tracks_present": _stats(tracked_growth),
                "dropout_to_tracked_mean_ratio": dropout_growth_ratio,
            },
            "window_correlations": {
                "mean_active_tracks_vs_position_error_growth": _pearson(
                    window_active, window_growth
                ),
                "accepted_features_vs_position_error_growth": _pearson(
                    window_accepted, window_growth
                ),
                "mean_active_tracks_vs_scale_ratio": _pearson(
                    window_active, window_scale
                ),
            },
            "visual_timestamp_match_max_delta_ms": float(
                visual["worst_timestamp_delta_ms"]
            ),
        },
        "gate_checkpoints_audit_only": checkpoints,
        "source_runtime_config": config_path,
        "truth_sources": {
            "translation": trajectory["translation_method"],
            "orientation": trajectory["orientation_method"],
        },
        "outputs": {
            "report": str(report_path),
            "timeseries": str(timeseries_path),
            "windows": str(windows_path),
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_value) + "\n",
        encoding="utf-8",
    )
    return report


def _csv_value(value: float) -> Any:
    return "" if not math.isfinite(float(value)) else float(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--window-s", type=float, default=2.0)
    parser.add_argument("--min-window-displacement-m", type=float, default=1.0)
    parser.add_argument("--min-velocity-m-s", type=float, default=0.5)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = audit_replay(
            args.replay,
            output_dir=args.output_dir,
            window_s=args.window_s,
            min_window_displacement_m=args.min_window_displacement_m,
            min_velocity_m_s=args.min_velocity_m_s,
        )
    except (DriftAuditError, DiagnosticError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    classification = report["classification"]
    print(f"Drift audit: {report['status']}")
    print(f"Dominant error: {classification['dominant_error']}")
    print(
        "Endpoint/path scale: "
        f"{report['scale']['endpoint_displacement_m']['estimated_to_truth_ratio']:.3f}/"
        f"{report['scale']['path_length_m']['estimated_to_truth_ratio']:.3f}"
    )
    print(
        "Attitude p95: "
        f"heading={report['attitude']['heading_absolute_error_deg']['p95']:.3f} deg; "
        f"tilt={report['attitude']['tilt_absolute_error_deg']['p95']:.3f} deg"
    )
    print(
        "Velocity direction p95: "
        f"{report['velocity']['direction_error_deg']['p95']:.3f} deg"
    )
    print(
        "Next experiment: "
        f"{classification['recommended_next_experiment']}"
    )
    print(f"Report: {report['outputs']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
