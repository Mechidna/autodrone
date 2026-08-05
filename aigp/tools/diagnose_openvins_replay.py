#!/usr/bin/env python3
"""Diagnose OpenVINS sensor consistency and accepted visual updates.

Truth is used only by this offline evaluator. It is never written into the
IMU/camera replay or supplied to OpenVINS.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


GRAVITY_NED_M_S2 = np.array((0.0, 0.0, 9.81), dtype=float)
DEFAULT_SMOOTHING_WINDOW_S = 0.25
GYRO_DYNAMIC_THRESHOLD_RAD_S = 0.02
IMU_SOURCE_REPLAY_NAMES = {
    "highres_imu": "openvins_replay_imuframefix_bracketed",
    "scaled_imu": "openvins_replay_scaled_imu_bracketed",
    "scaled_imu2": "openvins_replay_scaled_imu2_bracketed",
    "scaled_imu3": "openvins_replay_scaled_imu3_bracketed",
    "hil_sensor": "openvins_replay_hil_sensor_bracketed",
}


class DiagnosticError(RuntimeError):
    """A replay cannot be evaluated safely."""


def _read_numeric_csv(path: Path, fields: Sequence[str]) -> np.ndarray:
    if not path.is_file():
        raise DiagnosticError(f"missing required file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(fields) - set(reader.fieldnames or ())
        if missing:
            raise DiagnosticError(
                f"{path} is missing columns: {', '.join(sorted(missing))}"
            )
        try:
            values = [[float(row[field]) for field in fields] for row in reader]
        except (TypeError, ValueError) as exc:
            raise DiagnosticError(f"non-numeric value in {path}: {exc}") from exc
    if not values:
        raise DiagnosticError(f"CSV contains no rows: {path}")
    result = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(result)):
        raise DiagnosticError(f"CSV contains non-finite values: {path}")
    return result


def _require_increasing(timestamps: np.ndarray, label: str) -> None:
    if timestamps.size < 3 or np.any(np.diff(timestamps) <= 0.0):
        raise DiagnosticError(f"{label} timestamps are not strictly increasing")


def quaternion_xyzw_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    """Return Hamilton active rotation matrices for xyzw quaternions."""

    quaternions = np.asarray(quaternions, dtype=float)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise DiagnosticError("quaternions must have shape (N, 4)")
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms < 1e-9):
        raise DiagnosticError("truth contains a zero-length quaternion")
    x, y, z, w = (quaternions / norms[:, None]).T
    return np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=1,
    ).reshape((-1, 3, 3))


def euler_body_frd_to_ned_matrix(
    roll_rad: np.ndarray,
    pitch_rad: np.ndarray,
    yaw_rad: np.ndarray,
) -> np.ndarray:
    """Return body-FRD to local-NED matrices for aligned Euler samples."""

    roll = np.asarray(roll_rad, dtype=float).reshape(-1)
    pitch = np.asarray(pitch_rad, dtype=float).reshape(-1)
    yaw = np.asarray(yaw_rad, dtype=float).reshape(-1)
    if not (roll.shape == pitch.shape == yaw.shape):
        raise DiagnosticError("roll, pitch, and yaw arrays must have equal shape")
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.stack(
        (
            cy * cp,
            cy * sp * sr - sy * cr,
            cy * sp * cr + sy * sr,
            sy * cp,
            sy * sp * sr + cy * cr,
            sy * sp * cr - cy * sr,
            -sp,
            cp * sr,
            cp * cr,
        ),
        axis=1,
    ).reshape((-1, 3, 3))


def nearest_rotation_samples(
    source_time: np.ndarray,
    source_rotations: np.ndarray,
    target_time: np.ndarray,
    *,
    max_delta_s: float = 0.05,
) -> tuple[np.ndarray, float]:
    """Align rotation samples by nearest timestamp without interpolating Euler angles."""

    source_time = np.asarray(source_time, dtype=float).reshape(-1)
    target_time = np.asarray(target_time, dtype=float).reshape(-1)
    _require_increasing(source_time, "attitude truth")
    _require_increasing(target_time, "orientation target")
    if source_rotations.shape != (source_time.size, 3, 3):
        raise DiagnosticError("attitude rotation array has unexpected shape")
    insertion = np.searchsorted(source_time, target_time)
    right = np.clip(insertion, 0, source_time.size - 1)
    left = np.clip(insertion - 1, 0, source_time.size - 1)
    choose_right = np.abs(source_time[right] - target_time) < np.abs(
        source_time[left] - target_time
    )
    indices = np.where(choose_right, right, left)
    deltas = np.abs(source_time[indices] - target_time)
    worst_delta = float(np.max(deltas))
    if worst_delta > max_delta_s:
        raise DiagnosticError(
            "attitude truth is not sufficiently aligned with odometry: "
            f"worst nearest-sample delta {worst_delta * 1000.0:.3f} ms"
        )
    return source_rotations[indices], worst_delta


def _resolve_source_dataset(replay: Path, recorded_path: str) -> Path:
    """Resolve a replay's source dataset across Windows and WSL paths."""

    recorded = Path(recorded_path)
    if recorded.is_dir():
        return recorded
    sibling = replay.parent / "vio_dataset"
    if sibling.is_dir():
        return sibling
    return recorded


def replay_truth_local_ned(
    replay: Path,
    target_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    """Load physical local-NED position/velocity from the source capture."""

    manifest_path = replay / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise DiagnosticError(f"unable to read replay manifest: {exc}") from exc
    source_dataset_text = manifest.get("source_dataset")
    clock = manifest.get("clock_alignment", {})
    if not source_dataset_text or not {
        "imu_to_server_offset_ns",
        "replay_origin_server_ns",
    }.issubset(clock):
        return None

    source_path = (
        _resolve_source_dataset(replay, source_dataset_text)
        / "truth"
        / "local_position_ned.csv"
    )
    local = _read_numeric_csv(
        source_path,
        ("time_boot_ms", "x", "y", "z", "vx", "vy", "vz"),
    )
    local_by_time = {int(row[0]): row for row in local}
    local = np.asarray(
        [local_by_time[key] for key in sorted(local_by_time)],
        dtype=float,
    )
    source_time = (
        local[:, 0] * 1_000_000.0
        + float(clock["imu_to_server_offset_ns"])
        - float(clock["replay_origin_server_ns"])
    ) / 1_000_000_000.0
    _require_increasing(source_time, "local-position truth")
    target_time = np.asarray(target_time, dtype=float).reshape(-1)
    insertion = np.searchsorted(source_time, target_time)
    right = np.clip(insertion, 0, source_time.size - 1)
    left = np.clip(insertion - 1, 0, source_time.size - 1)
    choose_right = np.abs(source_time[right] - target_time) < np.abs(
        source_time[left] - target_time
    )
    indices = np.where(choose_right, right, left)
    deltas = np.abs(source_time[indices] - target_time)
    worst_delta = float(np.max(deltas))
    if worst_delta > 0.05:
        raise DiagnosticError(
            "local-position truth is not sufficiently aligned with odometry: "
            f"worst nearest-sample delta {worst_delta * 1000.0:.3f} ms"
        )
    return local[indices, 1:4], local[indices, 4:7], {
        "source": "captured_mavlink_local_position_ned",
        "source_file": str(source_path),
        "frame": "local_ned",
        "nearest_sample_max_delta_ms": worst_delta * 1000.0,
        "odometry_translation_used": False,
    }


def replay_truth_body_to_ned(
    replay: Path,
    truth_time: np.ndarray,
    fallback_quaternions_xyzw: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolve the physical body-to-NED attitude used for IMU evaluation.

    Competition ODOMETRY quaternions do not share the physical roll convention
    observed in ATTITUDE and vehicle motion.  Prepared VQ1 replays retain their
    source dataset path and clock mapping, so use the captured ATTITUDE stream
    with the same empirically verified boundary as live lateral calibration:
    (+roll, -pitch, competition-adjusted yaw).  Synthetic/standalone replays
    without that provenance retain the legacy quaternion interpretation.
    """

    manifest_path = replay / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise DiagnosticError(f"unable to read replay manifest: {exc}") from exc
        source_dataset_text = manifest.get("source_dataset")
        clock = manifest.get("clock_alignment", {})
        if source_dataset_text and {
            "imu_to_server_offset_ns",
            "replay_origin_server_ns",
        }.issubset(clock):
            source_dataset = _resolve_source_dataset(replay, source_dataset_text)
            attitude_path = source_dataset / "truth" / "attitude.csv"
            attitude = _read_numeric_csv(
                attitude_path,
                ("time_boot_ms", "roll", "pitch", "yaw"),
            )
            # ATTITUDE can contain retransmitted timestamps. Keep the last row.
            attitude_by_time = {int(row[0]): row for row in attitude}
            attitude = np.asarray(
                [attitude_by_time[key] for key in sorted(attitude_by_time)],
                dtype=float,
            )
            attitude_time = (
                attitude[:, 0] * 1_000_000.0
                + float(clock["imu_to_server_offset_ns"])
                - float(clock["replay_origin_server_ns"])
            ) / 1_000_000_000.0
            source_rotations = euler_body_frd_to_ned_matrix(
                attitude[:, 1],
                -attitude[:, 2],
                attitude[:, 3],
            )
            rotations, worst_delta = nearest_rotation_samples(
                attitude_time,
                source_rotations,
                truth_time,
            )
            return rotations, {
                "source": "captured_mavlink_attitude",
                "transform": "body_frd_to_local_ned(+roll,-pitch,competition_yaw)",
                "source_file": str(attitude_path),
                "nearest_sample_max_delta_ms": worst_delta * 1000.0,
                "odometry_quaternion_used": False,
            }

    return quaternion_xyzw_to_matrix(fallback_quaternions_xyzw), {
        "source": "truth_csv_quaternion",
        "transform": "Hamilton active body_frd_to_local_ned",
        "source_file": str(replay / "truth.csv"),
        "nearest_sample_max_delta_ms": None,
        "odometry_quaternion_used": True,
    }


def _window_samples(timestamps: np.ndarray, seconds: float) -> int:
    median_period = float(np.median(np.diff(timestamps)))
    samples = max(1, int(round(seconds / median_period)))
    if samples % 2 == 0:
        samples += 1
    maximum = timestamps.size if timestamps.size % 2 == 1 else timestamps.size - 1
    return min(samples, maximum)


def _centered_moving_average(values: np.ndarray, samples: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    was_vector = values.ndim == 1
    if was_vector:
        values = values[:, None]
    if values.ndim != 2:
        raise DiagnosticError("moving-average input must be one- or two-dimensional")
    samples = max(1, min(int(samples), values.shape[0]))
    if samples % 2 == 0:
        samples -= 1
    if samples <= 1:
        result = values.copy()
    else:
        padding = samples // 2
        padded = np.pad(values, ((padding, padding), (0, 0)), mode="edge")
        kernel = np.full(samples, 1.0 / samples)
        result = np.column_stack(
            [np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(values.shape[1])]
        )
    return result[:, 0] if was_vector else result


def _differentiate(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.gradient(values[:, axis], timestamps) for axis in range(values.shape[1])]
    )


def _vector_rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((left - right) ** 2, axis=1))))


def _axis_metrics(
    measured: np.ndarray,
    expected: np.ndarray,
    *,
    dynamic_threshold: float = 0.10,
    unit_key: str = "m_s2",
    signal_description: str = "specific force",
) -> dict[str, Any]:
    measured_mean = float(np.mean(measured))
    expected_mean = float(np.mean(expected))
    measured_std = float(np.std(measured))
    expected_std = float(np.std(expected))
    if measured_std > 1e-12 and expected_std > 1e-12:
        correlation = float(np.corrcoef(measured, expected)[0, 1])
        gain = float(np.polyfit(measured, expected, 1)[0])
    else:
        correlation = None
        gain = None
    rmse = float(np.sqrt(np.mean((measured - expected) ** 2)))
    sign_flip_rmse = float(np.sqrt(np.mean((-measured - expected) ** 2)))
    std_ratio = measured_std / expected_std if expected_std > 1e-12 else None

    status = "pass"
    reason = f"dynamic signal agrees with truth-derived {signal_description}"
    if expected_std < dynamic_threshold:
        reason = (
            f"truth-derived {signal_description} does not sufficiently excite "
            "this axis; sign and scale are not observable"
        )
    if expected_std >= dynamic_threshold and (std_ratio is None or std_ratio < 0.20):
        status = "fail_missing_dynamic_signal"
        reason = "measured axis has less than 20% of the expected dynamic variation"
    elif (
        expected_std >= dynamic_threshold
        and correlation is not None
        and correlation < -0.80
    ):
        status = "fail_opposite_sign"
        reason = "measured and expected dynamics have opposite sign"
    elif (
        expected_std >= dynamic_threshold
        and correlation is not None
        and correlation < 0.80
    ):
        status = "warn_low_correlation"
        reason = "measured dynamics correlate poorly with the expected signal"
    elif (
        expected_std >= dynamic_threshold
        and std_ratio is not None
        and not 0.5 <= std_ratio <= 2.0
    ):
        status = "fail_scale"
        reason = "measured dynamic scale differs by more than twofold"

    return {
        "status": status,
        "reason": reason,
        f"measured_mean_{unit_key}": measured_mean,
        f"expected_mean_{unit_key}": expected_mean,
        f"measured_std_{unit_key}": measured_std,
        f"expected_std_{unit_key}": expected_std,
        "measured_to_expected_std_ratio": std_ratio,
        "correlation": correlation,
        "expected_from_measured_gain": gain,
        f"rmse_{unit_key}": rmse,
        f"sign_flip_rmse_{unit_key}": sign_flip_rmse,
    }


def _metrics_for_interval(
    timestamps: np.ndarray,
    measured: np.ndarray,
    expected: np.ndarray,
    start_s: float,
    end_s: float,
) -> dict[str, Any] | None:
    selected = (timestamps >= start_s) & (timestamps <= end_s)
    if int(np.count_nonzero(selected)) < 10:
        return None
    return {
        "start_s": start_s,
        "end_s": end_s,
        "samples": int(np.count_nonzero(selected)),
        "axes": {
            axis: _axis_metrics(measured[selected, index], expected[selected, index])
            for index, axis in enumerate("xyz")
        },
    }


def analyze_accelerometer(replay: Path, smoothing_window_s: float) -> dict[str, Any]:
    imu = _read_numeric_csv(replay / "imu.csv", ("timestamp", "ax", "ay", "az"))
    truth = _read_numeric_csv(
        replay / "truth.csv",
        ("timestamp", "px", "py", "pz", "vx", "vy", "vz", "qx", "qy", "qz", "qw"),
    )
    imu_time, measured = imu[:, 0], imu[:, 1:4]
    truth_time = truth[:, 0]
    truth_position = truth[:, 1:4]
    reported_velocity = truth[:, 4:7]
    body_to_ned, orientation_method = replay_truth_body_to_ned(
        replay,
        truth_time,
        truth[:, 7:11],
    )
    _require_increasing(imu_time, "IMU")
    _require_increasing(truth_time, "truth")

    local_truth = replay_truth_local_ned(replay, truth_time)
    translation_method: dict[str, Any]
    if local_truth is not None:
        truth_position, velocity_ned, translation_method = local_truth
        velocity_frame = "local_ned"
    else:
        velocity_ned = reported_velocity
        translation_method = {
            "source": "truth_csv_odometry",
            "source_file": str(replay / "truth.csv"),
            "frame": "auto",
            "nearest_sample_max_delta_ms": None,
            "odometry_translation_used": True,
        }

    velocity_check_samples = _window_samples(truth_time, 0.50)
    position_rate = _centered_moving_average(
        _differentiate(truth_position, truth_time), velocity_check_samples
    )
    velocity_check_interior = (
        (truth_time >= truth_time[0] + 0.50)
        & (truth_time <= truth_time[-1] - 0.50)
    )
    selected_velocity_rmse = _vector_rmse(
        velocity_ned[velocity_check_interior],
        position_rate[velocity_check_interior],
    )
    raw_velocity_rmse: float | None = None
    body_velocity_rmse: float | None = None
    if local_truth is None:
        body_velocity_in_ned = np.einsum(
            "nij,nj->ni", body_to_ned, reported_velocity
        )
        raw_velocity_rmse = selected_velocity_rmse
        body_velocity_rmse = _vector_rmse(
            body_velocity_in_ned[velocity_check_interior],
            position_rate[velocity_check_interior],
        )
        if body_velocity_rmse < 0.25 * raw_velocity_rmse:
            velocity_frame = "body_frd"
            velocity_ned = body_velocity_in_ned
            selected_velocity_rmse = body_velocity_rmse
        elif raw_velocity_rmse < 0.25 * body_velocity_rmse:
            velocity_frame = "local_ned"
        else:
            raise DiagnosticError(
                "truth velocity frame is ambiguous; refusing to derive acceleration"
            )

    truth_smoothing_samples = _window_samples(truth_time, smoothing_window_s)
    velocity_ned_smoothed = _centered_moving_average(
        velocity_ned, truth_smoothing_samples
    )
    acceleration_ned = _differentiate(velocity_ned_smoothed, truth_time)
    specific_force_body = np.einsum(
        "nji,nj->ni", body_to_ned, acceleration_ned - GRAVITY_NED_M_S2
    )

    overlap = (imu_time >= truth_time[0]) & (imu_time <= truth_time[-1])
    if int(np.count_nonzero(overlap)) < 100:
        raise DiagnosticError("less than 100 overlapping IMU/truth samples")
    overlap_time = imu_time[overlap]
    overlap_measured = measured[overlap]
    overlap_expected = np.column_stack(
        [
            np.interp(overlap_time, truth_time, specific_force_body[:, axis])
            for axis in range(3)
        ]
    )
    imu_smoothing_samples = _window_samples(overlap_time, smoothing_window_s)
    overlap_measured = _centered_moving_average(
        overlap_measured, imu_smoothing_samples
    )
    overlap_expected = _centered_moving_average(
        overlap_expected, imu_smoothing_samples
    )

    edge_margin_s = smoothing_window_s
    interior = (overlap_time >= overlap_time[0] + edge_margin_s) & (
        overlap_time <= overlap_time[-1] - edge_margin_s
    )
    axes = {
        axis: _axis_metrics(
            overlap_measured[interior, index], overlap_expected[interior, index]
        )
        for index, axis in enumerate("xyz")
    }
    failed_axes = [axis for axis, metrics in axes.items() if metrics["status"].startswith("fail")]
    warning_axes = [axis for axis, metrics in axes.items() if metrics["status"].startswith("warn")]

    intervals = []
    for start_s, end_s in ((8.0, 9.0), (9.0, 12.0), (12.0, 15.0)):
        interval = _metrics_for_interval(
            overlap_time, overlap_measured, overlap_expected, start_s, end_s
        )
        if interval is not None:
            intervals.append(interval)

    return {
        "status": "fail" if failed_axes else ("warn" if warning_axes else "pass"),
        "method": {
            "truth_orientation": orientation_method,
            "truth_translation": translation_method,
            "truth_velocity_selected_frame": velocity_frame,
            "specific_force_equation": "f_body = R_body_to_NED^T * (a_NED - [0,0,9.81])",
            "smoothing_window_s": smoothing_window_s,
            "truth_smoothing_samples": truth_smoothing_samples,
            "imu_smoothing_samples": imu_smoothing_samples,
            "truth_is_evaluation_only": True,
        },
        "truth_velocity_frame_check": {
            "raw_as_local_ned_vector_rmse_m_s": raw_velocity_rmse,
            "body_frd_rotated_to_ned_vector_rmse_m_s": body_velocity_rmse,
            "selected_vector_rmse_m_s": selected_velocity_rmse,
            "selected": velocity_frame,
        },
        "overlap_samples": int(np.count_nonzero(interior)),
        "axes": axes,
        "failed_axes": failed_axes,
        "warning_axes": warning_axes,
        "intervals": intervals,
    }


def _rotation_vectors(rotation_matrices: np.ndarray) -> np.ndarray:
    """Return SO(3) logarithms for rotations known to be close to identity."""

    traces = np.trace(rotation_matrices, axis1=1, axis2=2)
    angles = np.arccos(np.clip((traces - 1.0) * 0.5, -1.0, 1.0))
    if np.any(angles >= math.pi - 1e-3):
        raise DiagnosticError(
            "truth orientation changes by nearly 180 degrees between samples"
        )
    antisymmetric = np.column_stack(
        (
            rotation_matrices[:, 2, 1] - rotation_matrices[:, 1, 2],
            rotation_matrices[:, 0, 2] - rotation_matrices[:, 2, 0],
            rotation_matrices[:, 1, 0] - rotation_matrices[:, 0, 1],
        )
    )
    scale = np.full_like(angles, 0.5)
    nonzero = angles > 1e-8
    scale[nonzero] = angles[nonzero] / (2.0 * np.sin(angles[nonzero]))
    return antisymmetric * scale[:, None]


def analyze_gyroscope(replay: Path, smoothing_window_s: float) -> dict[str, Any]:
    """Compare replay angular rates with truth orientation increments."""

    imu = _read_numeric_csv(replay / "imu.csv", ("timestamp", "wx", "wy", "wz"))
    truth = _read_numeric_csv(
        replay / "truth.csv", ("timestamp", "qx", "qy", "qz", "qw")
    )
    imu_time, measured = imu[:, 0], imu[:, 1:4]
    truth_time = truth[:, 0]
    _require_increasing(imu_time, "IMU")
    _require_increasing(truth_time, "truth")

    body_to_ned, orientation_method = replay_truth_body_to_ned(
        replay,
        truth_time,
        truth[:, 1:5],
    )
    body_increment = np.einsum(
        "nji,njk->nik", body_to_ned[:-1], body_to_ned[1:]
    )
    truth_midpoint_time = 0.5 * (truth_time[:-1] + truth_time[1:])
    angular_rate_body = _rotation_vectors(body_increment) / np.diff(truth_time)[:, None]

    overlap = (imu_time >= truth_midpoint_time[0]) & (
        imu_time <= truth_midpoint_time[-1]
    )
    if int(np.count_nonzero(overlap)) < 100:
        raise DiagnosticError("less than 100 overlapping IMU/truth gyro samples")
    overlap_time = imu_time[overlap]
    overlap_measured = measured[overlap]
    overlap_expected = np.column_stack(
        [
            np.interp(overlap_time, truth_midpoint_time, angular_rate_body[:, axis])
            for axis in range(3)
        ]
    )
    smoothing_samples = _window_samples(overlap_time, smoothing_window_s)
    overlap_measured = _centered_moving_average(overlap_measured, smoothing_samples)
    overlap_expected = _centered_moving_average(overlap_expected, smoothing_samples)
    interior = (overlap_time >= overlap_time[0] + smoothing_window_s) & (
        overlap_time <= overlap_time[-1] - smoothing_window_s
    )
    if int(np.count_nonzero(interior)) < 100:
        raise DiagnosticError("less than 100 interior gyro samples after smoothing")

    axes = {
        axis: _axis_metrics(
            overlap_measured[interior, index],
            overlap_expected[interior, index],
            dynamic_threshold=GYRO_DYNAMIC_THRESHOLD_RAD_S,
            unit_key="rad_s",
            signal_description="body angular rate",
        )
        for index, axis in enumerate("xyz")
    }
    failed_axes = [
        axis for axis, metrics in axes.items() if metrics["status"].startswith("fail")
    ]
    warning_axes = [
        axis for axis, metrics in axes.items() if metrics["status"].startswith("warn")
    ]
    return {
        "status": "fail" if failed_axes else ("warn" if warning_axes else "pass"),
        "method": {
            "truth_orientation": orientation_method,
            "angular_rate_equation": "Log(R_body_to_NED[k]^T * R_body_to_NED[k+1]) / dt",
            "comparison_frame": "body_frd",
            "smoothing_window_s": smoothing_window_s,
            "imu_smoothing_samples": smoothing_samples,
            "dynamic_threshold_rad_s": GYRO_DYNAMIC_THRESHOLD_RAD_S,
            "truth_is_evaluation_only": True,
        },
        "overlap_samples": int(np.count_nonzero(interior)),
        "axes": axes,
        "failed_axes": failed_axes,
        "warning_axes": warning_axes,
    }


def _startup_aligned_position_errors(
    estimate_position_global: np.ndarray,
    estimate_global_to_imu: np.ndarray,
    truth_position_ned: np.ndarray,
    truth_body_to_ned: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Align the VIO origin/orientation once at startup, without scale fitting."""

    rotation_global_to_ned = truth_body_to_ned[0] @ estimate_global_to_imu[0]
    relative_estimate = estimate_position_global - estimate_position_global[0]
    aligned = (
        np.einsum("ij,nj->ni", rotation_global_to_ned, relative_estimate)
        + truth_position_ned[0]
    )
    return aligned, np.linalg.norm(aligned - truth_position_ned, axis=1)


def analyze_estimate(replay: Path) -> dict[str, Any]:
    estimate_path = replay / "openvins_estimate.csv"
    if not estimate_path.is_file():
        return {"status": "not_available", "path": str(estimate_path)}
    estimate = _read_numeric_csv(
        estimate_path,
        (
            "timestamp", "q_GtoI_x", "q_GtoI_y", "q_GtoI_z", "q_GtoI_w",
            "p_IinG_x", "p_IinG_y", "p_IinG_z",
            "v_IinG_x", "v_IinG_y", "v_IinG_z",
        ),
    )
    truth = _read_numeric_csv(
        replay / "truth.csv", ("timestamp", "vx", "vy", "vz")
    )
    estimate_speed = np.linalg.norm(estimate[:, 8:11], axis=1)
    truth_speed = np.linalg.norm(truth[:, 1:4], axis=1)
    expected_speed = np.interp(estimate[:, 0], truth[:, 0], truth_speed)
    speed_error = np.abs(estimate_speed - expected_speed)
    over_five = np.flatnonzero(speed_error >= 5.0)
    speed_error_summary = _finite_summary(speed_error)
    position_evaluation: dict[str, Any] = {
        "available": False,
        "reason": "captured local-position truth is unavailable",
    }
    local_truth = replay_truth_local_ned(replay, estimate[:, 0])
    if local_truth is not None:
        truth_position_ned, _, translation_method = local_truth
        fallback_attitude = np.tile(
            np.asarray((0.0, 0.0, 0.0, 1.0)), (estimate.shape[0], 1)
        )
        truth_body_to_ned, orientation_method = replay_truth_body_to_ned(
            replay, estimate[:, 0], fallback_attitude
        )
        # OpenVINS stores JPL q_GtoI; the Hamilton matrix for the same numeric
        # coefficients is its transpose.
        estimate_global_to_imu = np.transpose(
            quaternion_xyzw_to_matrix(estimate[:, 1:5]), (0, 2, 1)
        )
        aligned_position, position_error = _startup_aligned_position_errors(
            estimate[:, 5:8],
            estimate_global_to_imu,
            truth_position_ned,
            truth_body_to_ned,
        )
        over_half_meter = np.flatnonzero(position_error >= 0.5)
        over_two_meters = np.flatnonzero(position_error >= 2.0)
        position_summary = _finite_summary(position_error)
        estimated_path_length_m = float(
            np.sum(np.linalg.norm(np.diff(aligned_position, axis=0), axis=1))
        )
        truth_path_length_m = float(
            np.sum(np.linalg.norm(np.diff(truth_position_ned, axis=0), axis=1))
        )
        estimated_displacement_m = float(
            np.linalg.norm(aligned_position[-1] - aligned_position[0])
        )
        truth_displacement_m = float(
            np.linalg.norm(truth_position_ned[-1] - truth_position_ned[0])
        )
        position_status = (
            "pass"
            if position_summary["p95"] <= 0.5 and position_summary["max"] <= 1.0
            else "fail"
        )
        position_evaluation = {
            "available": True,
            "status": position_status,
            "alignment": (
                "single startup origin and full orientation; rigid rotation/"
                "translation only; no scale fitting or trajectory-wide alignment"
            ),
            "translation_truth": translation_method,
            "orientation_truth": orientation_method,
            "error_m": position_summary,
            "final_error_m": float(position_error[-1]),
            "path_length_m": {
                "estimated": estimated_path_length_m,
                "truth": truth_path_length_m,
                "estimated_to_truth_ratio": (
                    estimated_path_length_m / truth_path_length_m
                    if truth_path_length_m > 0.0
                    else None
                ),
            },
            "endpoint_displacement_m": {
                "estimated": estimated_displacement_m,
                "truth": truth_displacement_m,
                "estimated_to_truth_ratio": (
                    estimated_displacement_m / truth_displacement_m
                    if truth_displacement_m > 0.0
                    else None
                ),
            },
            "first_error_over_0_5_m_s": (
                None
                if over_half_meter.size == 0
                else float(estimate[over_half_meter[0], 0])
            ),
            "first_error_over_2_m_s": (
                None
                if over_two_meters.size == 0
                else float(estimate[over_two_meters[0], 0])
            ),
            "final_aligned_position_ned_m": aligned_position[-1].tolist(),
            "final_truth_position_ned_m": truth_position_ned[-1].tolist(),
        }
    speed_status = "fail" if over_five.size else "pass"
    position_status = position_evaluation.get("status", "not_available")
    return {
        "status": (
            "fail"
            if speed_status == "fail" or position_status == "fail"
            else "pass"
        ),
        "speed_status": speed_status,
        "rows": int(estimate.shape[0]),
        "first_timestamp_s": float(estimate[0, 0]),
        "last_timestamp_s": float(estimate[-1, 0]),
        "final_position_norm_m": float(np.linalg.norm(estimate[-1, 5:8])),
        "final_estimated_speed_m_s": float(estimate_speed[-1]),
        "final_truth_speed_m_s": float(expected_speed[-1]),
        "max_estimated_speed_m_s": float(np.max(estimate_speed)),
        "max_truth_speed_m_s": float(np.max(truth_speed)),
        "speed_error_m_s": speed_error_summary,
        "first_speed_error_over_5_m_s": (
            None if over_five.size == 0 else float(estimate[over_five[0], 0])
        ),
        "startup_aligned_position": position_evaluation,
    }


def _count_summary(values: np.ndarray) -> dict[str, Any]:
    return {
        "sum": int(np.sum(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "min": int(np.min(values)),
        "max": int(np.max(values)),
        "nonzero_rows": int(np.count_nonzero(values)),
        "zero_fraction": float(np.mean(values == 0)),
    }


def _finite_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0, "nonfinite_count": int(values.size)}
    return {
        "count": int(finite.size),
        "nonfinite_count": int(values.size - finite.size),
        "min": float(np.min(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p25": float(np.percentile(finite, 25)),
        "median": float(np.median(finite)),
        "p75": float(np.percentile(finite, 75)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def _visual_health(
    state_advanced_fraction: float,
    msckf: np.ndarray,
    slam: np.ndarray,
    active: np.ndarray,
) -> tuple[str, str]:
    if state_advanced_fraction < 0.99:
        return (
            "fail_state_continuity",
            "the estimator state did not advance on at least 99% of initialized camera rows",
        )
    # MSCKF features are normally consumed in batches when tracks are lost or
    # marginalized, so zero on an individual frame is not itself a failure.
    # Fewer than one accepted feature per four camera rows over an interval,
    # with fewer than one persistent SLAM feature on average, means visual
    # constraints are too sparse to bound inertial drift.
    if int(np.sum(msckf)) < 0.25 * msckf.size and float(np.mean(slam)) < 1.0:
        return (
            "fail_sparse_visual_constraints",
            "too few MSCKF features survive geometric/gating checks and too few SLAM features persist",
        )
    if float(np.mean(active == 0)) > 0.20:
        return (
            "warn_tracking_dropouts",
            "more than 20% of initialized camera rows have no triangulated active tracks",
        )
    return "pass", "visual constraints and state continuity meet diagnostic thresholds"


MSCKF_FUNNEL_FIELDS = (
    "msckf_lost_candidates",
    "msckf_marginal_candidates",
    "msckf_maxtrack_candidates",
    "msckf_candidates_before_limit",
    "msckf_candidates_after_limit",
    "msckf_input_features",
    "msckf_input_measurements",
    "msckf_rejected_too_few",
    "msckf_rejected_triangulation",
    "msckf_triangulation_bad_condition",
    "msckf_triangulation_depth_too_near",
    "msckf_triangulation_depth_too_far",
    "msckf_triangulation_invalid_numeric",
    "msckf_triangulation_other",
    "msckf_rejected_refinement",
    "msckf_refinement_depth_too_near",
    "msckf_refinement_depth_too_far",
    "msckf_refinement_baseline_ratio",
    "msckf_refinement_invalid_numeric",
    "msckf_refinement_other",
    "msckf_chi2_tested",
    "msckf_rejected_chi2",
    "msckf_accepted_features",
    "msckf_accepted_measurements",
    "msckf_chi2_ratio_mean",
    "msckf_chi2_ratio_max",
)

MSCKF_SELECTION_FIELDS = (
    "msckf_geometry_valid_features",
    "msckf_post_chi2_features",
    "msckf_rejected_selection_limit",
    "msckf_selected_features",
)

FEATURE_GEOMETRY_FIELDS = (
    "camera_timestamp",
    "result",
    "triangulation_success",
    "refinement_success",
    "chi2_tested",
    "accepted",
    "triangulation_failure_reason",
    "refinement_failure_reason",
    "observation_count",
    "track_duration_s",
    "max_parallax_deg",
    "max_camera_baseline_m",
    "condition_number",
    "linear_depth_m",
    "linear_range_m",
    "refined_depth_m",
    "refined_range_m",
    "chi2_ratio",
)

FEATURE_IMAGE_FIELDS = (
    "first_u_px",
    "first_v_px",
    "last_u_px",
    "last_v_px",
    "mean_u_px",
    "mean_v_px",
    "min_u_px",
    "max_u_px",
    "min_v_px",
    "max_v_px",
)

FEATURE_SELECTION_FIELDS = (
    "selected_for_update",
    "selection_rank",
    "selection_grid_row",
    "selection_grid_col",
)

FEATURE_RESULT_NAMES = {
    0: "pending",
    1: "triangulation_rejected",
    2: "refinement_rejected",
    3: "chi2_rejected",
    4: "accepted",
    5: "selection_rejected",
}

SLAM_FEATURE_IMAGE_FIELDS = (
    "camera_timestamp",
    "feature_id",
    "tracked_in_current_frame",
    "u_px",
    "v_px",
    "depth_m",
    "update_fail_count",
    "should_marg",
)


def analyze_feature_geometry(replay: Path) -> dict[str, Any]:
    path = replay / "openvins_feature_geometry.csv"
    if not path.is_file():
        return {
            "available": False,
            "path": str(path),
            "reason": "per-feature geometry instrumentation has not been replayed",
        }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = sorted(set(FEATURE_GEOMETRY_FIELDS) - fieldnames)
        if missing:
            return {
                "available": False,
                "path": str(path),
                "reason": "per-feature geometry CSV is missing required columns",
                "missing_columns": missing,
            }
        image_fields_available = set(FEATURE_IMAGE_FIELDS).issubset(fieldnames)
        selection_fields_available = set(FEATURE_SELECTION_FIELDS).issubset(
            fieldnames
        )
        loaded_fields = FEATURE_GEOMETRY_FIELDS + (
            FEATURE_IMAGE_FIELDS if image_fields_available else ()
        ) + (
            FEATURE_SELECTION_FIELDS if selection_fields_available else ()
        )
        try:
            rows = [
                [float(row[field]) for field in loaded_fields]
                for row in reader
            ]
        except (TypeError, ValueError) as exc:
            raise DiagnosticError(f"non-numeric value in {path}: {exc}") from exc
    if not rows:
        return {"available": True, "path": str(path), "rows": 0, "groups": {}}

    data = np.asarray(rows, dtype=float)
    values = {
        field: data[:, index]
        for index, field in enumerate(loaded_fields)
    }
    return _finish_feature_geometry_analysis(
        replay,
        path,
        data,
        values,
        image_fields_available,
        selection_fields_available,
    )


def analyze_slam_feature_images(replay: Path) -> dict[str, Any]:
    """Describe where persistent SLAM landmarks live in the camera raster."""
    path = replay / "openvins_slam_feature_diagnostics.csv"
    if not path.is_file():
        return {
            "available": False,
            "path": str(path),
            "reason": "persistent-SLAM image instrumentation has not been replayed",
        }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = sorted(set(SLAM_FEATURE_IMAGE_FIELDS) - fieldnames)
        if missing:
            return {
                "available": False,
                "path": str(path),
                "reason": "persistent-SLAM CSV is missing required columns",
                "missing_columns": missing,
            }
        rows = []
        try:
            for row in reader:
                rows.append(
                    [
                        float(row[field]) if row[field] != "" else math.nan
                        for field in SLAM_FEATURE_IMAGE_FIELDS
                    ]
                )
        except (TypeError, ValueError) as exc:
            raise DiagnosticError(f"non-numeric value in {path}: {exc}") from exc
    if not rows:
        return {
            "available": True,
            "path": str(path),
            "state_rows": 0,
            "unique_features": 0,
        }

    data = np.asarray(rows, dtype=float)
    values = {
        field: data[:, index]
        for index, field in enumerate(SLAM_FEATURE_IMAGE_FIELDS)
    }
    try:
        manifest = json.loads((replay / "manifest.json").read_text(encoding="utf-8"))
        width, height = [
            float(value) for value in manifest["calibration"]["resolution"]
        ]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        width = height = math.nan
    if not all(math.isfinite(value) and value > 0.0 for value in (width, height)):
        return {
            "available": False,
            "path": str(path),
            "reason": "replay manifest has no valid camera resolution",
        }

    timestamps = values["camera_timestamp"]
    feature_ids = values["feature_id"].astype(np.int64)
    tracked = values["tracked_in_current_frame"] > 0.5
    u_px = values["u_px"]
    v_px = values["v_px"]
    spatial_valid = (
        tracked
        & np.isfinite(u_px)
        & np.isfinite(v_px)
        & (u_px >= 0.0)
        & (u_px < width)
        & (v_px >= 0.0)
        & (v_px < height)
    )
    x_index = np.full(data.shape[0], -1, dtype=int)
    y_index = np.full(data.shape[0], -1, dtype=int)
    x_index[spatial_valid] = np.minimum(
        2, (3.0 * u_px[spatial_valid] / width).astype(int)
    )
    y_index[spatial_valid] = np.minimum(
        2, (3.0 * v_px[spatial_valid] / height).astype(int)
    )

    landmark_rows: list[dict[str, float | int]] = []
    for feature_id in np.unique(feature_ids):
        selected = feature_ids == feature_id
        observed = selected & spatial_valid
        if not np.any(observed):
            continue
        landmark_rows.append(
            {
                "feature_id": int(feature_id),
                "median_u_px": float(np.median(u_px[observed])),
                "median_v_px": float(np.median(v_px[observed])),
                "state_lifetime_s": float(
                    np.max(timestamps[selected]) - np.min(timestamps[selected])
                ),
                "state_updates": int(np.count_nonzero(selected)),
                "tracked_updates": int(np.count_nonzero(observed)),
            }
        )

    assigned_u = np.asarray(
        [row["median_u_px"] for row in landmark_rows], dtype=float
    )
    assigned_v = np.asarray(
        [row["median_v_px"] for row in landmark_rows], dtype=float
    )
    assigned_x = np.minimum(2, (3.0 * assigned_u / width).astype(int))
    assigned_y = np.minimum(2, (3.0 * assigned_v / height).astype(int))
    lifetimes = np.asarray(
        [row["state_lifetime_s"] for row in landmark_rows], dtype=float
    )
    state_updates = np.asarray(
        [row["state_updates"] for row in landmark_rows], dtype=float
    )
    tracked_updates = np.asarray(
        [row["tracked_updates"] for row in landmark_rows], dtype=float
    )
    spatial_rows = int(np.count_nonzero(spatial_valid))
    assigned_total = len(landmark_rows)

    def region_report(sample_selected: np.ndarray, assigned_selected: np.ndarray) -> dict[str, Any]:
        sample_count = int(np.count_nonzero(sample_selected))
        assigned_count = int(np.count_nonzero(assigned_selected))
        return {
            "sample_rows": sample_count,
            "sample_fraction": sample_count / spatial_rows if spatial_rows else None,
            "unique_features_seen": int(
                np.unique(feature_ids[sample_selected]).size
            ),
            "assigned_unique_features": assigned_count,
            "assigned_fraction": (
                assigned_count / assigned_total if assigned_total else None
            ),
            "assigned_state_lifetime_s": _finite_summary(
                lifetimes[assigned_selected]
            ),
            "assigned_state_updates": _finite_summary(
                state_updates[assigned_selected]
            ),
            "assigned_tracked_updates": _finite_summary(
                tracked_updates[assigned_selected]
            ),
        }

    vertical_names = ("top", "middle", "bottom")
    horizontal_names = ("left", "center", "right")
    grid = {}
    for y, vertical_name in enumerate(vertical_names):
        for x, horizontal_name in enumerate(horizontal_names):
            grid[f"{vertical_name}_{horizontal_name}"] = region_report(
                spatial_valid & (x_index == x) & (y_index == y),
                (assigned_x == x) & (assigned_y == y),
            )

    top_samples = spatial_valid & (v_px < height / 3.0)
    top_assigned = assigned_v < height / 3.0
    top_third = region_report(top_samples, top_assigned)
    top_third["mask_rows"] = int(round(height / 3.0))

    camera_times = np.unique(timestamps)
    occupancy = np.asarray(
        [np.count_nonzero(timestamps == timestamp) for timestamp in camera_times],
        dtype=int,
    )
    tracked_occupancy = np.asarray(
        [
            np.count_nonzero((timestamps == timestamp) & spatial_valid)
            for timestamp in camera_times
        ],
        dtype=int,
    )
    top_occupancy = np.asarray(
        [
            np.count_nonzero((timestamps == timestamp) & top_samples)
            for timestamp in camera_times
        ],
        dtype=int,
    )
    return {
        "available": True,
        "path": str(path),
        "coordinate": "current raw distorted cam0 pixel for each persistent landmark still tracked",
        "resolution": [int(width), int(height)],
        "state_rows": int(data.shape[0]),
        "tracked_spatial_rows": spatial_rows,
        "untracked_or_invalid_rows": int(data.shape[0] - spatial_rows),
        "unique_features_in_state": int(np.unique(feature_ids).size),
        "unique_features_with_pixels": assigned_total,
        "per_camera_occupancy": {
            "state": _count_summary(occupancy),
            "tracked_with_pixels": _count_summary(tracked_occupancy),
            "tracked_in_top_third": _count_summary(top_occupancy),
        },
        "landmark_lifecycle": {
            "state_lifetime_s": _finite_summary(lifetimes),
            "state_updates": _finite_summary(state_updates),
            "tracked_updates": _finite_summary(tracked_updates),
        },
        "top_third_counterfactual_mask": top_third,
        "grid_3x3": grid,
        "semantics": {
            "sample_rows": "landmark-camera observations, so long-lived landmarks contribute repeatedly",
            "assigned_unique_features": "each landmark counted once at its median pixel location",
            "state_lifetime_s": "time from first to last post-promotion appearance in the persistent state",
        },
    }


def _finish_feature_geometry_analysis(
    replay: Path,
    path: Path,
    data: np.ndarray,
    values: dict[str, np.ndarray],
    image_fields_available: bool,
    selection_fields_available: bool,
) -> dict[str, Any]:
    results = values["result"].astype(int)
    triangulation_reasons = values["triangulation_failure_reason"].astype(int)
    groups = {
        "all_attempted": np.ones(data.shape[0], dtype=bool),
        "bad_condition_rejected": (results == 1) & (triangulation_reasons == 1),
        "triangulation_passed": values["triangulation_success"] > 0.5,
        "accepted": values["accepted"] > 0.5,
    }
    if selection_fields_available:
        groups["selection_rejected"] = results == 5
        groups["post_chi2_valid"] = (results == 4) | (results == 5)
    before_12_s = values["camera_timestamp"] <= 12.0
    groups.update(
        {
            "bad_condition_rejected_before_12_s": (
                groups["bad_condition_rejected"] & before_12_s
            ),
            "triangulation_passed_before_12_s": (
                groups["triangulation_passed"] & before_12_s
            ),
            "accepted_before_12_s": groups["accepted"] & before_12_s,
        }
    )
    metric_names = (
        "observation_count",
        "track_duration_s",
        "max_parallax_deg",
        "max_camera_baseline_m",
        "condition_number",
        "linear_depth_m",
        "linear_range_m",
        "refined_depth_m",
        "refined_range_m",
        "chi2_ratio",
    ) + (FEATURE_IMAGE_FIELDS if image_fields_available else ()) + (
        ("selection_rank",) if selection_fields_available else ()
    )
    group_reports = {
        name: {
            "rows": int(np.count_nonzero(selected)),
            "metrics": {
                metric: _finite_summary(values[metric][selected])
                for metric in metric_names
            },
        }
        for name, selected in groups.items()
    }

    bad = groups["bad_condition_rejected"]
    bad_condition = values["condition_number"][bad]
    bad_condition = bad_condition[np.isfinite(bad_condition)]
    bad_parallax = values["max_parallax_deg"][bad]
    bad_parallax = bad_parallax[np.isfinite(bad_parallax)]
    condition_threshold = 10_000.0
    image_region_analysis: dict[str, Any] = {
        "available": False,
        "reason": "per-feature image-coordinate instrumentation has not been replayed",
    }
    if image_fields_available:
        try:
            manifest = json.loads(
                (replay / "manifest.json").read_text(encoding="utf-8")
            )
            width, height = [
                float(value) for value in manifest["calibration"]["resolution"]
            ]
            cx, cy = [
                float(value) for value in manifest["calibration"]["intrinsics"][2:4]
            ]
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            width = height = cx = cy = math.nan
        if all(math.isfinite(value) and value > 0.0 for value in (width, height)):
            mean_u = values["mean_u_px"]
            mean_v = values["mean_v_px"]
            spatial_valid = (
                np.isfinite(mean_u)
                & np.isfinite(mean_v)
                & (mean_u >= 0.0)
                & (mean_u < width)
                & (mean_v >= 0.0)
                & (mean_v < height)
            )
            x_index = np.full(data.shape[0], -1, dtype=int)
            y_index = np.full(data.shape[0], -1, dtype=int)
            x_index[spatial_valid] = np.minimum(
                2, (3.0 * mean_u[spatial_valid] / width).astype(int)
            )
            y_index[spatial_valid] = np.minimum(
                2, (3.0 * mean_v[spatial_valid] / height).astype(int)
            )

            def image_subset_report(selected: np.ndarray) -> dict[str, Any]:
                rows_selected = int(np.count_nonzero(selected))
                accepted_rows = int(np.count_nonzero(selected & groups["accepted"]))
                bad_rows = int(
                    np.count_nonzero(selected & groups["bad_condition_rejected"])
                )
                return {
                    "rows": rows_selected,
                    "accepted": accepted_rows,
                    "bad_condition_rejected": bad_rows,
                    "acceptance_fraction": (
                        accepted_rows / rows_selected if rows_selected else None
                    ),
                    "bad_condition_fraction": (
                        bad_rows / rows_selected if rows_selected else None
                    ),
                    "parallax_deg": _finite_summary(
                        values["max_parallax_deg"][selected]
                    ),
                }

            region_names = (
                ("top", "middle", "bottom"),
                ("left", "center", "right"),
            )
            grid = {}
            for y, vertical_name in enumerate(region_names[0]):
                for x, horizontal_name in enumerate(region_names[1]):
                    selected = spatial_valid & (x_index == x) & (y_index == y)
                    grid[f"{vertical_name}_{horizontal_name}"] = (
                        image_subset_report(selected)
                    )

            radius_fraction = np.full(data.shape[0], np.nan, dtype=float)
            corner_radius = math.sqrt((width / 2.0) ** 2 + (height / 2.0) ** 2)
            radius_fraction[spatial_valid] = np.sqrt(
                (mean_u[spatial_valid] - cx) ** 2
                + (mean_v[spatial_valid] - cy) ** 2
            ) / corner_radius
            radial_bins = {
                "inner_third": spatial_valid & (radius_fraction < 1.0 / 3.0),
                "middle_third": spatial_valid
                & (radius_fraction >= 1.0 / 3.0)
                & (radius_fraction < 2.0 / 3.0),
                "outer_third": spatial_valid & (radius_fraction >= 2.0 / 3.0),
            }

            vertical_zone_reports: dict[str, dict[str, Any]] = {}
            vertical_indices: dict[str, np.ndarray] = {}
            for coordinate_name, coordinate_values in (
                ("birth", values["first_v_px"]),
                ("mean", values["mean_v_px"]),
                ("terminal", values["last_v_px"]),
            ):
                valid = (
                    np.isfinite(coordinate_values)
                    & (coordinate_values >= 0.0)
                    & (coordinate_values < height)
                )
                indices = np.full(data.shape[0], -1, dtype=int)
                indices[valid] = np.minimum(
                    2, (3.0 * coordinate_values[valid] / height).astype(int)
                )
                vertical_indices[coordinate_name] = indices
                vertical_zone_reports[coordinate_name] = {
                    name: image_subset_report(valid & (indices == index))
                    for index, name in enumerate(("top", "middle", "bottom"))
                }

            accepted_transitions = {}
            for birth_index, birth_name in enumerate(("top", "middle", "bottom")):
                for terminal_index, terminal_name in enumerate(
                    ("top", "middle", "bottom")
                ):
                    accepted_transitions[f"{birth_name}_to_{terminal_name}"] = int(
                        np.count_nonzero(
                            groups["accepted"]
                            & (vertical_indices["birth"] == birth_index)
                            & (vertical_indices["terminal"] == terminal_index)
                        )
                    )
            startup = values["camera_timestamp"] <= 6.0
            startup_accepted = startup & groups["accepted"]
            startup_born_top = startup_accepted & (
                vertical_indices["birth"] == 0
            )
            image_region_analysis = {
                "available": True,
                "coordinate": "mean raw distorted pixel position over the track",
                "resolution": [int(width), int(height)],
                "optical_center_px": [cx, cy],
                "valid_rows": int(np.count_nonzero(spatial_valid)),
                "invalid_rows": int(data.shape[0] - np.count_nonzero(spatial_valid)),
                "grid_3x3": grid,
                "radial": {
                    name: image_subset_report(selected)
                    for name, selected in radial_bins.items()
                },
                "vertical_track_flow": {
                    "zones": vertical_zone_reports,
                    "accepted_birth_to_terminal_counts": accepted_transitions,
                    "startup_through_6_s": {
                        "accepted": int(np.count_nonzero(startup_accepted)),
                        "accepted_born_in_top_third": int(
                            np.count_nonzero(startup_born_top)
                        ),
                        "accepted_born_in_top_third_fraction": (
                            float(
                                np.count_nonzero(startup_born_top)
                                / np.count_nonzero(startup_accepted)
                            )
                            if np.any(startup_accepted)
                            else None
                        ),
                    },
                },
            }
        else:
            image_region_analysis = {
                "available": False,
                "reason": "replay manifest has no valid camera resolution",
            }
    selection_analysis: dict[str, Any] = {
        "available": False,
        "reason": "geometry-ranked selection instrumentation has not been replayed",
    }
    if selection_fields_available:
        selected = values["selected_for_update"] > 0.5
        grid_row = values["selection_grid_row"].astype(int)
        grid_col = values["selection_grid_col"].astype(int)
        grid_valid = (
            (grid_row >= 0)
            & (grid_row < 3)
            & (grid_col >= 0)
            & (grid_col < 3)
        )
        vertical_names = ("top", "middle", "bottom")
        horizontal_names = ("left", "center", "right")
        selection_analysis = {
            "available": True,
            "post_chi2_candidates": int(
                np.count_nonzero((results == 4) | (results == 5))
            ),
            "selected": int(np.count_nonzero(selected)),
            "rejected_by_limit": int(np.count_nonzero(results == 5)),
            "grid_3x3": {
                f"{vertical_names[row]}_{horizontal_names[column]}": {
                    "candidates": int(
                        np.count_nonzero(
                            grid_valid & (grid_row == row) & (grid_col == column)
                        )
                    ),
                    "selected": int(
                        np.count_nonzero(
                            selected & (grid_row == row) & (grid_col == column)
                        )
                    ),
                }
                for row in range(3)
                for column in range(3)
            },
        }
    return {
        "available": True,
        "path": str(path),
        "rows": int(data.shape[0]),
        "result_counts": {
            name: int(np.count_nonzero(results == code))
            for code, name in FEATURE_RESULT_NAMES.items()
        },
        "groups": group_reports,
        "bad_condition_threshold_analysis": {
            "configured_threshold": condition_threshold,
            "condition_number_bins": {
                "at_or_below_threshold": int(
                    np.count_nonzero(bad_condition <= condition_threshold)
                ),
                "one_to_ten_times_threshold": int(
                    np.count_nonzero(
                        (bad_condition > condition_threshold)
                        & (bad_condition <= 10 * condition_threshold)
                    )
                ),
                "ten_to_one_hundred_times_threshold": int(
                    np.count_nonzero(
                        (bad_condition > 10 * condition_threshold)
                        & (bad_condition <= 100 * condition_threshold)
                    )
                ),
                "above_one_hundred_times_threshold": int(
                    np.count_nonzero(bad_condition > 100 * condition_threshold)
                ),
                "nonfinite": int(np.count_nonzero(bad) - bad_condition.size),
            },
            "parallax_bins_deg": {
                "below_0_1": int(np.count_nonzero(bad_parallax < 0.1)),
                "from_0_1_to_0_5": int(
                    np.count_nonzero(
                        (bad_parallax >= 0.1) & (bad_parallax < 0.5)
                    )
                ),
                "from_0_5_to_1": int(
                    np.count_nonzero(
                        (bad_parallax >= 0.5) & (bad_parallax < 1.0)
                    )
                ),
                "from_1_to_2": int(
                    np.count_nonzero(
                        (bad_parallax >= 1.0) & (bad_parallax < 2.0)
                    )
                ),
                "at_least_2": int(np.count_nonzero(bad_parallax >= 2.0)),
                "nonfinite": int(np.count_nonzero(bad) - bad_parallax.size),
            },
        },
        "image_region_analysis": image_region_analysis,
        "geometry_ranked_selection": selection_analysis,
        "semantics": {
            "result_codes": FEATURE_RESULT_NAMES,
            "failure_reason_codes": {
                0: "none",
                1: "bad_condition",
                2: "depth_too_near",
                3: "depth_too_far",
                4: "baseline_ratio",
                5: "invalid_numeric",
            },
            "max_parallax_deg": "maximum pairwise angle between observed rays expressed in a common frame",
            "max_camera_baseline_m": "maximum pairwise distance between camera clone positions",
            "condition_number": "condition number of the same 3x3 linear triangulation matrix used by OpenVINS",
            "linear_depth_m": "anchor-frame z from the unconstrained linear solution; it may be nonphysical for rejected tracks",
            "image_regions": "thirds of the 640x360 raster selected by each track's mean raw pixel position",
        },
    }


def _analyze_msckf_rejection_funnel(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        fieldnames = tuple(csv.DictReader(handle).fieldnames or ())
    missing = sorted(set(MSCKF_FUNNEL_FIELDS) - set(fieldnames))
    if missing:
        return {
            "available": False,
            "reason": "OpenVINS rejection-counter patch was not present for this replay",
            "missing_columns": missing,
        }

    selection_fields_available = set(MSCKF_SELECTION_FIELDS).issubset(
        fieldnames
    )
    loaded_funnel_fields = MSCKF_FUNNEL_FIELDS + (
        MSCKF_SELECTION_FIELDS if selection_fields_available else ()
    )

    fields = ("initialized",) + loaded_funnel_fields
    data = _read_numeric_csv(path, fields)
    post = data[data[:, 0] > 0.5, 1:]
    if post.size == 0:
        return {"available": True, "status": "fail_no_initialized_rows"}
    values = {
        field: post[:, index]
        for index, field in enumerate(loaded_funnel_fields)
    }
    totals = {
        field: int(np.sum(values[field]))
        for field in loaded_funnel_fields
        if field not in ("msckf_chi2_ratio_mean", "msckf_chi2_ratio_max")
    }
    input_features = totals["msckf_input_features"]
    too_few = totals["msckf_rejected_too_few"]
    triangulation = totals["msckf_rejected_triangulation"]
    refinement = totals["msckf_rejected_refinement"]
    chi2_tested = totals["msckf_chi2_tested"]
    chi2_rejected = totals["msckf_rejected_chi2"]
    selection_rejected = (
        totals["msckf_rejected_selection_limit"]
        if selection_fields_available
        else 0
    )
    accepted = totals["msckf_accepted_features"]
    accounted = (
        too_few
        + triangulation
        + refinement
        + chi2_rejected
        + selection_rejected
        + accepted
    )
    triangulation_reasons = {
        "bad_condition": totals["msckf_triangulation_bad_condition"],
        "depth_too_near": totals["msckf_triangulation_depth_too_near"],
        "depth_too_far": totals["msckf_triangulation_depth_too_far"],
        "invalid_numeric": totals["msckf_triangulation_invalid_numeric"],
        "other": totals["msckf_triangulation_other"],
    }
    refinement_reasons = {
        "depth_too_near": totals["msckf_refinement_depth_too_near"],
        "depth_too_far": totals["msckf_refinement_depth_too_far"],
        "baseline_ratio": totals["msckf_refinement_baseline_ratio"],
        "invalid_numeric": totals["msckf_refinement_invalid_numeric"],
        "other": totals["msckf_refinement_other"],
    }

    rejection_counts = {
        "too_few_measurements": too_few,
        "triangulation": triangulation,
        "refinement": refinement,
        "chi2": chi2_rejected,
    }
    if selection_fields_available:
        rejection_counts["selection_limit"] = selection_rejected
    dominant_stage = max(rejection_counts, key=rejection_counts.get)
    accepted_fraction = accepted / input_features if input_features else 0.0
    if input_features == 0:
        status = "fail_no_msckf_candidates"
    elif accepted_fraction < 0.01:
        status = f"fail_dominant_{dominant_stage}_rejection"
    elif accepted_fraction < 0.05:
        status = f"warn_dominant_{dominant_stage}_rejection"
    else:
        status = "pass"

    tested_per_row = values["msckf_chi2_tested"]
    tested_rows = tested_per_row > 0.0
    weighted_chi2_ratio_mean = (
        float(
            np.sum(
                values["msckf_chi2_ratio_mean"][tested_rows]
                * tested_per_row[tested_rows]
            )
            / np.sum(tested_per_row[tested_rows])
        )
        if np.any(tested_rows)
        else None
    )
    return {
        "available": True,
        "status": status,
        "dominant_rejection_stage": dominant_stage,
        "initialized_rows": int(post.shape[0]),
        "totals": totals,
        "accounting": {
            "input_features": input_features,
            "accounted_features": accounted,
            "difference": input_features - accounted,
        },
        "triangulation_reasons": {
            "counts": triangulation_reasons,
            "accounted": sum(triangulation_reasons.values()),
            "difference": triangulation - sum(triangulation_reasons.values()),
            "dominant": max(triangulation_reasons, key=triangulation_reasons.get),
        },
        "refinement_reasons": {
            "counts": refinement_reasons,
            "accounted": sum(refinement_reasons.values()),
            "difference": refinement - sum(refinement_reasons.values()),
            "dominant": max(refinement_reasons, key=refinement_reasons.get),
        },
        "fractions_of_input": {
            "too_few_measurements": too_few / input_features if input_features else None,
            "triangulation": triangulation / input_features if input_features else None,
            "refinement": refinement / input_features if input_features else None,
            "chi2": chi2_rejected / input_features if input_features else None,
            "selection_limit": (
                selection_rejected / input_features
                if selection_fields_available and input_features
                else None
            ),
            "accepted": accepted_fraction if input_features else None,
        },
        "conditional_fractions": {
            "triangulation_rejected_after_track_length_check": (
                triangulation / (input_features - too_few)
                if input_features > too_few
                else None
            ),
            "refinement_rejected_after_triangulation": (
                refinement / (input_features - too_few - triangulation)
                if input_features > too_few + triangulation
                else None
            ),
            "chi2_rejected_of_tested": (
                chi2_rejected / chi2_tested if chi2_tested else None
            ),
            "accepted_of_chi2_tested": accepted / chi2_tested if chi2_tested else None,
            "selected_of_post_chi2_valid": (
                totals["msckf_selected_features"]
                / totals["msckf_post_chi2_features"]
                if selection_fields_available
                and totals["msckf_post_chi2_features"]
                else None
            ),
        },
        "chi2_ratio_to_threshold": {
            "weighted_mean": weighted_chi2_ratio_mean,
            "maximum": float(np.max(values["msckf_chi2_ratio_max"])),
            "interpretation": "values above 1.0 fail the configured chi-square gate",
        },
        "per_update": {
            field: _count_summary(values[field].astype(int))
            for field in loaded_funnel_fields
            if field not in ("msckf_chi2_ratio_mean", "msckf_chi2_ratio_max")
        },
        "geometry_ranked_selection": {
            "available": selection_fields_available,
            "geometry_valid_features": (
                totals["msckf_geometry_valid_features"]
                if selection_fields_available
                else None
            ),
            "post_chi2_features": (
                totals["msckf_post_chi2_features"]
                if selection_fields_available
                else None
            ),
            "rejected_by_limit": selection_rejected,
            "selected_features": (
                totals["msckf_selected_features"]
                if selection_fields_available
                else None
            ),
        },
    }


def analyze_visual_updates(replay: Path) -> dict[str, Any]:
    path = replay / "openvins_update_diagnostics.csv"
    if not path.is_file():
        return {
            "status": "pending_rerun",
            "path": str(path),
            "reason": "the instrumented runner has not produced per-camera diagnostics yet",
        }
    data = _read_numeric_csv(
        path,
        (
            "camera_timestamp", "initialized", "state_advanced",
            "msckf_features_used", "slam_features_in_state", "active_tracks",
        ),
    )
    initialized = data[:, 1] > 0.5
    if not np.any(initialized):
        return {"status": "fail", "reason": "OpenVINS never initialized"}
    post = data[initialized]
    msckf = post[:, 3].astype(int)
    slam = post[:, 4].astype(int)
    active = post[:, 5].astype(int)
    state_advanced_fraction = float(np.mean(post[:, 2] > 0.5))
    status, reason = _visual_health(
        state_advanced_fraction, msckf, slam, active
    )
    windows: list[dict[str, Any]] = []
    for start_s, end_s in ((8.0, 9.0), (9.0, 12.0), (12.0, 15.0)):
        selected = (post[:, 0] >= start_s) & (post[:, 0] <= end_s)
        if np.any(selected):
            window_status, window_reason = _visual_health(
                float(np.mean(post[selected, 2] > 0.5)),
                msckf[selected],
                slam[selected],
                active[selected],
            )
            windows.append(
                {
                    "status": window_status,
                    "reason": window_reason,
                    "start_s": start_s,
                    "end_s": end_s,
                    "camera_updates": int(np.count_nonzero(selected)),
                    "msckf_features_used": _count_summary(msckf[selected]),
                    "slam_features_in_state": _count_summary(slam[selected]),
                    "active_tracks": _count_summary(active[selected]),
                }
            )
    return {
        "status": status,
        "reason": reason,
        "camera_rows": int(data.shape[0]),
        "initialized_rows": int(post.shape[0]),
        "state_advanced_fraction": state_advanced_fraction,
        "msckf_features_used": _count_summary(msckf),
        "slam_features_in_state": _count_summary(slam),
        "active_tracks": _count_summary(active),
        "msckf_rejection_funnel": _analyze_msckf_rejection_funnel(path),
        "feature_geometry": analyze_feature_geometry(replay),
        "slam_feature_images": analyze_slam_feature_images(replay),
        "windows": windows,
        "semantics": (
            "msckf_features_used is OpenVINS get_good_features_MSCKF() after "
            "triangulation and chi-square gating"
        ),
    }


def analyze_replay(
    replay: Path, smoothing_window_s: float = DEFAULT_SMOOTHING_WINDOW_S
) -> dict[str, Any]:
    replay = replay.resolve()
    if not replay.is_dir():
        raise DiagnosticError(f"replay directory does not exist: {replay}")
    accelerometer = analyze_accelerometer(replay, smoothing_window_s)
    gyroscope = analyze_gyroscope(replay, smoothing_window_s)
    estimate = analyze_estimate(replay)
    visual_updates = analyze_visual_updates(replay)
    blockers = []
    for axis in accelerometer["failed_axes"]:
        blockers.append(
            {
                "code": f"accelerometer_{axis}_{accelerometer['axes'][axis]['status']}",
                "detail": accelerometer["axes"][axis]["reason"],
            }
        )
    for axis in gyroscope["failed_axes"]:
        blockers.append(
            {
                "code": f"gyroscope_{axis}_{gyroscope['axes'][axis]['status']}",
                "detail": gyroscope["axes"][axis]["reason"],
            }
        )
    if estimate["status"] == "fail":
        blockers.append(
            {
                "code": "openvins_velocity_divergence",
                "detail": "estimated velocity diverges materially from truth",
            }
        )
    if visual_updates["status"].startswith("fail"):
        blockers.append(
            {
                "code": visual_updates["status"],
                "detail": visual_updates["reason"],
            }
        )
    return {
        "format": "aigp_openvins_diagnostic_report",
        "format_version": 5,
        "replay": str(replay),
        "status": "fail" if blockers else "pass",
        "accelerometer": accelerometer,
        "gyroscope": gyroscope,
        "estimate": estimate,
        "visual_updates": visual_updates,
        "blockers": blockers,
        "truth_fed_to_estimator": False,
    }


def _default_runs_root() -> Path:
    return Path(__file__).resolve().parents[1] / "logs" / "runs"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help="run ID under aigp/logs/runs")
    source.add_argument("--replay", type=Path, help="OpenVINS replay directory")
    parser.add_argument(
        "--imu-source",
        choices=tuple(IMU_SOURCE_REPLAY_NAMES),
        default="highres_imu",
        help=(
            "IMU source replay to diagnose when --run is used "
            "(default: highres_imu)"
        ),
    )
    parser.add_argument("--output", type=Path, help="diagnostic JSON output path")
    parser.add_argument(
        "--smoothing-window-s",
        type=float,
        default=DEFAULT_SMOOTHING_WINDOW_S,
        help="centered smoothing window used before differentiation (default: 0.25)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not 0.05 <= args.smoothing_window_s <= 2.0:
        print("ERROR: --smoothing-window-s must be between 0.05 and 2.0", file=sys.stderr)
        return 2
    replay = (
        _default_runs_root()
        / args.run
        / IMU_SOURCE_REPLAY_NAMES[args.imu_source]
        if args.run
        else args.replay
    )
    output = args.output or replay / "openvins_diagnostic_report.json"
    try:
        report = analyze_replay(replay, args.smoothing_window_s)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(output)
    except (DiagnosticError, OSError, ValueError, np.linalg.LinAlgError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Diagnostic report: {output.resolve()}")
    print(
        "Truth velocity frame: "
        f"{report['accelerometer']['truth_velocity_frame_check']['selected']}"
    )
    for axis in "xyz":
        metrics = report["accelerometer"]["axes"][axis]
        print(
            f"Accel {axis}: {metrics['status']}; correlation "
            f"{metrics['correlation']!s}; measured/expected dynamic std "
            f"{metrics['measured_to_expected_std_ratio']!s}"
        )
    for axis in "xyz":
        metrics = report["gyroscope"]["axes"][axis]
        print(
            f"Gyro {axis}: {metrics['status']}; correlation "
            f"{metrics['correlation']!s}; measured/expected dynamic std "
            f"{metrics['measured_to_expected_std_ratio']!s}"
        )
    print(f"Visual updates: {report['visual_updates']['status']}")
    funnel = report["visual_updates"].get("msckf_rejection_funnel", {})
    if funnel.get("available"):
        totals = funnel["totals"]
        print(
            "MSCKF funnel: "
            f"input={totals['msckf_input_features']}; "
            f"too_few={totals['msckf_rejected_too_few']}; "
            f"triangulation={totals['msckf_rejected_triangulation']}; "
            f"refinement={totals['msckf_rejected_refinement']}; "
            f"chi2={totals['msckf_rejected_chi2']}; "
            + (
                f"selection={totals['msckf_rejected_selection_limit']}; "
                if "msckf_rejected_selection_limit" in totals
                else ""
            )
            + f"accepted={totals['msckf_accepted_features']}"
        )
        tri_reasons = funnel.get("triangulation_reasons", {})
        refine_reasons = funnel.get("refinement_reasons", {})
        if tri_reasons:
            print(
                "Triangulation reasons: "
                + "; ".join(
                    f"{name}={count}"
                    for name, count in tri_reasons["counts"].items()
                )
            )
        if refine_reasons:
            print(
                "Refinement reasons: "
                + "; ".join(
                    f"{name}={count}"
                    for name, count in refine_reasons["counts"].items()
                )
            )
        geometry = report["visual_updates"].get("feature_geometry", {})
        if geometry.get("available") and geometry.get("rows", 0):
            bad = geometry["groups"]["bad_condition_rejected"]
            accepted = geometry["groups"]["accepted"]
            print(
                "Feature geometry: "
                f"rows={geometry['rows']}; "
                f"bad-condition parallax median="
                f"{bad['metrics']['max_parallax_deg'].get('median')} deg; "
                f"bad-condition cond median="
                f"{bad['metrics']['condition_number'].get('median')}; "
                f"accepted parallax median="
                f"{accepted['metrics']['max_parallax_deg'].get('median')} deg"
            )
            image_regions = geometry.get("image_region_analysis", {})
            if image_regions.get("available"):
                grid = image_regions["grid_3x3"]
                ranked = sorted(
                    (
                        (name, region)
                        for name, region in grid.items()
                        if region["rows"] > 0
                        and region["acceptance_fraction"] is not None
                    ),
                    key=lambda item: item[1]["acceptance_fraction"],
                    reverse=True,
                )
                bottom_accepted = sum(
                    region["accepted"]
                    for name, region in grid.items()
                    if name.startswith("bottom_")
                )
                accepted_total = accepted["rows"]
                best_name, best_region = ranked[0]
                worst_name, worst_region = ranked[-1]
                print(
                    "Feature image regions: "
                    f"best={best_name} "
                    f"({100.0 * best_region['acceptance_fraction']:.2f}% accepted); "
                    f"worst={worst_name} "
                    f"({100.0 * worst_region['acceptance_fraction']:.2f}% accepted); "
                    f"bottom-third accepted share="
                    f"{100.0 * bottom_accepted / accepted_total:.2f}%"
                )
        slam_images = report["visual_updates"].get("slam_feature_images", {})
        if (
            slam_images.get("available")
            and slam_images.get("unique_features_with_pixels", 0)
        ):
            top = slam_images["top_third_counterfactual_mask"]
            lifetime = top["assigned_state_lifetime_s"]
            print(
                "Persistent SLAM image regions: "
                f"top-third sample share={100.0 * top['sample_fraction']:.2f}%; "
                f"top-third landmark share={100.0 * top['assigned_fraction']:.2f}%; "
                f"top-third lifetime median={lifetime.get('median')} s"
            )
    print(f"Overall: {report['status']}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
