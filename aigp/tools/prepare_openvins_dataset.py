#!/usr/bin/env python3
"""Validate an AIGP VIO capture and prepare deterministic OpenVINS replay data.

The simulator camera timestamp is already in the TIMESYNC server clock.  IMU
``time_usec`` is a boot clock, so this tool estimates its constant offset to
the server clock from the lower envelope of receive delays.  It deliberately
does not regress the IMU clock against packet arrival time: UDP/MAVLink queue
jitter is one-sided and would appear as false clock drift.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


NS_PER_SECOND = 1_000_000_000
US_TO_NS = 1_000
IMU_OFFSET_QUANTILE = 0.02
CAMERA_TILT_UP_DEG = 20.0
DEFAULT_CAMERA_TILT_UP_DEG = CAMERA_TILT_UP_DEG
FEATURE_MAX_DISTANCE_M = 200.0
DEFAULT_MAX_CLONES = 11
DEFAULT_MAX_MSCKF_IN_UPDATE = 40
DEFAULT_MAX_SLAM = 50
DEFAULT_MAX_SLAM_IN_UPDATE = 25
DEFAULT_NUM_PTS = 300
DEFAULT_FAST_THRESHOLD = 15
DEFAULT_MIN_PX_DIST = 10
DEFAULT_CAMERA_IMU_TIME_OFFSET_MS = 0.0
DEFAULT_CAMERA_FOCAL_PX = None
DEFAULT_MASK_TOP_ROWS = 0
DEFAULT_MASK_GUIDE_CONE = False
DEFAULT_CAMERA_IMAGE_MODE = "grayscale"
CAMERA_IMAGE_MODES = ("grayscale", "red", "red_fixed", "red_sidehist")
DEFAULT_HISTOGRAM_METHOD = "HISTOGRAM"
HISTOGRAM_METHODS = ("NONE", "HISTOGRAM", "CLAHE")
GUIDE_CONE_MASK_REFERENCE_RESOLUTION = [640, 360]
GUIDE_CONE_MASK_POLYGON_NORMALIZED = [
    [285.0 / 640.0, 130.0 / 360.0],
    [355.0 / 640.0, 130.0 / 360.0],
    [590.0 / 640.0, 359.0 / 360.0],
    [50.0 / 640.0, 359.0 / 360.0],
]
DEFAULT_STATIONARY_START = "first"
DEFAULT_INITIALIZATION_MODE = "jerk"
DEFAULT_IMU_GAP_POLICY = "reject"
DEFAULT_MAX_IMU_GAP_MS = 120.0
IMU_GAP_REJECT_THRESHOLD_MS = 30.0
INITIALIZATION_WINDOW_S = 2.0
INITIALIZATION_MIN_RATE_HZ = 100.0
INITIALIZATION_MAX_GYRO_NORM_RAD_S = 0.1
INITIALIZATION_MAX_ACCEL_MEAN_ERROR_M_S2 = 0.5
AUTO_STATIONARY_MAX_ACCEL_STD_M_S2 = 0.25
IMU_GAP_POLICIES = ("reject", "allow", "interpolate")
STATIONARY_START_POLICIES = ("first", "auto")
INITIALIZATION_MODES = ("jerk", "stationary")
COMPETITION_GYRO_SIGN = (-1.0, -1.0, -1.0)
COMPETITION_ACCEL_SIGN = (1.0, 1.0, 1.0)
SPEC_GYRO_SIGN = (1.0, 1.0, 1.0)
SUPPORTED_IMU_SOURCES = (
    "highres_imu",
    "scaled_imu",
    "scaled_imu2",
    "scaled_imu3",
    "hil_sensor",
    "raw_imu",
)


class DatasetError(RuntimeError):
    """The capture cannot safely be replayed as a VIO dataset."""


@dataclass(frozen=True)
class AffineClockFit:
    local_origin_ns: int
    server_origin_ns: int
    intercept_s: float
    slope: float
    pair_count: int
    selected_pair_count: int
    median_rtt_ms: float
    p95_rtt_ms: float

    def local_to_server_ns(self, local_ns: int) -> float:
        elapsed_s = (local_ns - self.local_origin_ns) / NS_PER_SECOND
        return self.server_origin_ns + (
            self.intercept_s + self.slope * elapsed_s
        ) * NS_PER_SECOND


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise DatasetError(f"missing required file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise DatasetError(f"CSV contains no data rows: {path}")
    return rows


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise DatasetError("cannot calculate a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, probability)) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _strictly_increasing(values: Sequence[int], label: str) -> None:
    for previous, current in zip(values, values[1:]):
        if current <= previous:
            raise DatasetError(
                f"{label} is not strictly increasing: {previous} then {current}"
            )


def _fit_timesync(rows: Sequence[dict[str, str]]) -> AffineClockFit:
    requests: dict[int, list[int]] = {}
    pairs: list[tuple[int, int, int]] = []

    for row in rows:
        direction = row["direction"].strip().lower()
        wall_ns = int(row["wall_time_ns"])
        tc1 = int(row["tc1"])
        ts1 = int(row["ts1"])
        if direction == "tx" and ts1 == 0:
            requests.setdefault(tc1, []).append(wall_ns)
        elif direction == "rx" and ts1 in requests and requests[ts1]:
            tx_wall_ns = requests[ts1].pop(0)
            if wall_ns >= tx_wall_ns:
                local_midpoint_ns = (tx_wall_ns + wall_ns) // 2
                pairs.append((local_midpoint_ns, tc1, wall_ns - tx_wall_ns))

    if len(pairs) < 10:
        raise DatasetError(
            f"only {len(pairs)} complete TIMESYNC exchanges; at least 10 required"
        )

    rtts_ns = [pair[2] for pair in pairs]
    cutoff_ns = _percentile(rtts_ns, 0.5)
    selected = [pair for pair in pairs if pair[2] <= cutoff_ns]
    if len(selected) < 5:
        raise DatasetError("too few low-RTT TIMESYNC exchanges for clock fit")

    local_origin_ns = selected[0][0]
    server_origin_ns = selected[0][1]
    xs = [(pair[0] - local_origin_ns) / NS_PER_SECOND for pair in selected]
    ys = [(pair[1] - server_origin_ns) / NS_PER_SECOND for pair in selected]
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    variance_x = sum((x - mean_x) ** 2 for x in xs)
    if variance_x <= 0.0:
        raise DatasetError("TIMESYNC exchanges have no usable time span")
    slope = sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)
    ) / variance_x
    intercept_s = mean_y - slope * mean_x

    if not 0.9995 <= slope <= 1.0005:
        raise DatasetError(f"implausible local/server clock slope: {slope:.9f}")

    return AffineClockFit(
        local_origin_ns=local_origin_ns,
        server_origin_ns=server_origin_ns,
        intercept_s=intercept_s,
        slope=slope,
        pair_count=len(pairs),
        selected_pair_count=len(selected),
        median_rtt_ms=_percentile(rtts_ns, 0.5) / 1e6,
        p95_rtt_ms=_percentile(rtts_ns, 0.95) / 1e6,
    )


def _rate_hz(timestamps_ns: Sequence[int]) -> float:
    if len(timestamps_ns) < 2 or timestamps_ns[-1] <= timestamps_ns[0]:
        raise DatasetError("not enough timestamp span to calculate a rate")
    return (len(timestamps_ns) - 1) * NS_PER_SECOND / (
        timestamps_ns[-1] - timestamps_ns[0]
    )


def _check_source_manifest(dataset: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("format") != "aigp_vio_dataset":
        raise DatasetError("unsupported source dataset format")
    if manifest.get("status") != "complete" or not manifest.get("clean_shutdown"):
        raise DatasetError("source dataset did not finish with a clean shutdown")
    if manifest.get("errors"):
        raise DatasetError(f"source recorder reported errors: {manifest['errors']}")
    counts = manifest.get("counts", {})
    if any(int(value) for value in counts.get("dropped_queue_full", {}).values()):
        raise DatasetError("source recorder dropped queued measurements")
    if any(int(value) for value in counts.get("write_failures", {}).values()):
        raise DatasetError("source recorder had write failures")
    if not (dataset / "runtime.toml").is_file():
        raise DatasetError("source dataset is missing its runtime.toml snapshot")


def _load_imu_source(
    dataset: Path,
    manifest: dict[str, Any],
    imu_source: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    source = str(imu_source).strip().lower()
    if source not in SUPPORTED_IMU_SOURCES:
        raise DatasetError(
            f"unsupported IMU source {imu_source!r}; choose from "
            + ", ".join(SUPPORTED_IMU_SOURCES)
        )
    if source == "raw_imu":
        raise DatasetError(
            "RAW_IMU is device-specific and unscaled; capture is supported, "
            "but OpenVINS replay requires a calibrated SI conversion"
        )

    files = manifest.get("files", {})
    source_files = files.get("imu_sources", {})
    if source == "highres_imu":
        relative_path = source_files.get(source) or files.get("imu")
        gyro_sign = COMPETITION_GYRO_SIGN
        profile = "vq1_highres_imu_physical_body_rate_corrected"
        reason = (
            "VQ1 physical body angular velocity derived from captured "
            "ATTITUDE and controlled roll excitation matches HIGHRES_IMU as "
            "(-xgyro, -ygyro, -zgyro)"
        )
    else:
        relative_path = source_files.get(source)
        gyro_sign = SPEC_GYRO_SIGN
        profile = f"mavlink_{source}_spec_axes"
        reason = (
            "MAVLink scaled/SI alternative source converted to SI during "
            "capture and replayed with specification-defined body axes"
        )
    if not relative_path:
        raise DatasetError(
            f"dataset manifest does not advertise IMU source {source!r}"
        )

    rows = _read_csv(dataset / relative_path)
    required = ("time_usec", "xacc", "yacc", "zacc", "xgyro", "ygyro", "zgyro")
    missing = [name for name in required if name not in rows[0]]
    if missing:
        raise DatasetError(
            f"IMU source {source!r} is missing columns: {', '.join(missing)}"
        )
    for row_index, row in enumerate(rows, start=2):
        for name in required:
            if str(row.get(name, "")).strip() == "":
                raise DatasetError(
                    f"IMU source {source!r} has an empty {name} at CSV row {row_index}"
                )

    source_row_count = len(rows)
    rows_by_timestamp = {int(row["time_usec"]): row for row in rows}
    rows = [rows_by_timestamp[timestamp] for timestamp in sorted(rows_by_timestamp)]
    if rows and int(rows[-1]["time_usec"]) >= 1_000_000_000_000:
        raise DatasetError(
            f"IMU source {source!r} appears to use UNIX-epoch time; "
            "this replay path currently requires a boot-clock timestamp"
        )
    return rows, {
        "source": source,
        "source_file": str(relative_path),
        "source_rows": source_row_count,
        "unique_rows": len(rows),
        "timestamp_kind": rows[0].get("timestamp_kind") or "boot",
        "gyro_sign_xyz": gyro_sign,
        "accel_sign_xyz": COMPETITION_ACCEL_SIGN,
        "profile": profile,
        "reason": reason,
    }


def _validate_jpegs(
    dataset: Path, camera_rows: Sequence[dict[str, str]]
) -> list[Path]:
    paths: list[Path] = []
    for row in camera_rows:
        path = dataset / "camera" / row["filename"]
        if not path.is_file():
            raise DatasetError(f"missing camera image: {path}")
        expected_size = int(row["jpeg_size_bytes"])
        if path.stat().st_size != expected_size:
            raise DatasetError(
                f"JPEG size mismatch for {path.name}: "
                f"expected {expected_size}, got {path.stat().st_size}"
            )
        with path.open("rb") as handle:
            start = handle.read(2)
            handle.seek(-2, os.SEEK_END)
            end = handle.read(2)
        if start != b"\xff\xd8" or end != b"\xff\xd9":
            raise DatasetError(f"invalid JPEG markers: {path}")
        paths.append(path)
    return paths


def _camera_to_imu_matrix(
    camera_tilt_up_deg: float = DEFAULT_CAMERA_TILT_UP_DEG,
) -> list[list[float]]:
    tilt = math.radians(camera_tilt_up_deg)
    sine = math.sin(tilt)
    cosine = math.cos(tilt)
    # OpenCV optical (right, down, forward) -> MAVLink body/IMU FRD.
    return [
        [0.0, sine, cosine],
        [1.0, 0.0, 0.0],
        [0.0, cosine, -sine],
    ]


def _render_estimator_config(
    max_clones: int = DEFAULT_MAX_CLONES,
    track_frequency_hz: float = 30.0,
    max_msckf_in_update: int = DEFAULT_MAX_MSCKF_IN_UPDATE,
    max_slam: int = DEFAULT_MAX_SLAM,
    max_slam_in_update: int = DEFAULT_MAX_SLAM_IN_UPDATE,
    initialization_mode: str = DEFAULT_INITIALIZATION_MODE,
    geometry_ranked_msckf: bool = False,
    histogram_method: str = DEFAULT_HISTOGRAM_METHOD,
    num_pts: int = DEFAULT_NUM_PTS,
    fast_threshold: int = DEFAULT_FAST_THRESHOLD,
    min_px_dist: int = DEFAULT_MIN_PX_DIST,
) -> str:
    stationary_initialization = initialization_mode == "stationary"
    try_zupt = str(stationary_initialization).lower()
    zupt_only_at_beginning = str(stationary_initialization).lower()
    return f"""%YAML:1.0
verbosity: "INFO"
use_fej: true
integration: "rk4"
use_stereo: false
max_cameras: 1
calib_cam_extrinsics: false
calib_cam_intrinsics: false
calib_cam_timeoffset: false
calib_imu_intrinsics: false
calib_imu_g_sensitivity: false
max_clones: {max_clones}
max_slam: {max_slam}
max_slam_in_update: {max_slam_in_update}
max_msckf_in_update: {max_msckf_in_update}
msckf_geometry_ranked_selection: {str(geometry_ranked_msckf).lower()}
dt_slam_delay: 1
gravity_mag: 9.81
feat_rep_msckf: "GLOBAL_3D"
feat_rep_slam: "ANCHORED_MSCKF_INVERSE_DEPTH"
feat_rep_aruco: "ANCHORED_MSCKF_INVERSE_DEPTH"
fi_max_dist: 200.0

try_zupt: {try_zupt}
zupt_chi2_multipler: 0
zupt_max_velocity: 0.5
zupt_noise_multiplier: 50
zupt_max_disparity: 0.5
zupt_only_at_beginning: {zupt_only_at_beginning}

init_window_time: 2.0
init_imu_thresh: 0.5
init_max_disparity: 2.0
init_max_features: 50
init_dyn_use: false
init_dyn_mle_opt_calib: false
init_dyn_mle_max_iter: 50
init_dyn_mle_max_time: 0.05
init_dyn_mle_max_threads: 6
init_dyn_num_pose: 6
init_dyn_min_deg: 10.0
init_dyn_inflation_ori: 10
init_dyn_inflation_vel: 100
init_dyn_inflation_bg: 10
init_dyn_inflation_ba: 100
init_dyn_min_rec_cond: 1e-12
init_dyn_bias_g: [0.0, 0.0, 0.0]
init_dyn_bias_a: [0.0, 0.0, 0.0]

record_timing_information: false
record_timing_filepath: "/tmp/openvins_timing.txt"
save_total_state: false
filepath_est: "/tmp/openvins_estimate.txt"
filepath_std: "/tmp/openvins_deviation.txt"
filepath_gt: "/tmp/openvins_groundtruth.txt"

use_klt: true
num_pts: {num_pts}
fast_threshold: {fast_threshold}
grid_x: 8
grid_y: 5
min_px_dist: {min_px_dist}
knn_ratio: 0.65
track_frequency: {track_frequency_hz:.9f}
downsample_cameras: false
num_opencv_threads: 4
histogram_method: "{histogram_method}"

use_aruco: false
num_aruco: 1024
downsize_aruco: false

up_msckf_sigma_px: 1.0
up_msckf_chi2_multipler: 1
up_slam_sigma_px: 1.0
up_slam_chi2_multipler: 1
up_aruco_sigma_px: 1.0
up_aruco_chi2_multipler: 1
use_mask: false

relative_config_imu: "kalibr_imu_chain.yaml"
relative_config_imucam: "kalibr_imucam_chain.yaml"
"""


def _render_imu_config(rate_hz: float) -> str:
    return f"""%YAML:1.0
imu0:
  T_i_b:
    - [1.0, 0.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0, 0.0]
    - [0.0, 0.0, 1.0, 0.0]
    - [0.0, 0.0, 0.0, 1.0]
  accelerometer_noise_density: 0.01
  accelerometer_random_walk: 0.001
  gyroscope_noise_density: 0.001
  gyroscope_random_walk: 0.0001
  rostopic: /imu0
  time_offset: 0.0
  update_rate: {rate_hz:.9f}
  model: "kalibr"
  Tw:
    - [1.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  R_IMUtoGYRO:
    - [1.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  Ta:
    - [1.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  R_IMUtoACC:
    - [1.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  Tg:
    - [0.0, 0.0, 0.0]
    - [0.0, 0.0, 0.0]
    - [0.0, 0.0, 0.0]
"""


def _render_camera_config(
    camera: dict[str, Any],
    camera_imu_timeoffset_s: float = 0.0,
    camera_tilt_up_deg: float = DEFAULT_CAMERA_TILT_UP_DEG,
) -> str:
    rotation = _camera_to_imu_matrix(camera_tilt_up_deg)
    translation = [float(value) for value in camera["body_translation_m"]]
    matrix_rows = [
        rotation[row] + [translation[row]] for row in range(3)
    ] + [[0.0, 0.0, 0.0, 1.0]]
    matrix_yaml = "\n".join(
        "      - [" + ", ".join(f"{value:.12f}" for value in row) + "]"
        for row in matrix_rows
    )
    coefficients = [float(value) for value in camera["dist_coeffs"][:4]]
    return f"""%YAML:1.0
cam0:
  T_imu_cam:
{matrix_yaml}
  camera_model: pinhole
  distortion_model: radtan
  distortion_coeffs: [{', '.join(f'{value:.12f}' for value in coefficients)}]
  intrinsics: [{float(camera['fx']):.12f}, {float(camera['fy']):.12f}, {float(camera['cx']):.12f}, {float(camera['cy']):.12f}]
  resolution: [{int(camera['width'])}, {int(camera['height'])}]
  rostopic: /cam0/image_raw
  timeshift_cam_imu: {camera_imu_timeoffset_s:.9f}
  cam_overlaps: []
"""


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(fieldnames)
        writer.writerows(rows)


def _format_time(timestamp_ns: float, origin_ns: float) -> str:
    return f"{(timestamp_ns - origin_ns) / NS_PER_SECOND:.9f}"


def _link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _imu_window_stats(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[float], list[float]]:
    gyro_norms = [
        math.sqrt(
            sum(float(row[name]) ** 2 for name in ("xgyro", "ygyro", "zgyro"))
        )
        for row in rows
    ]
    accel_norms = [
        math.sqrt(
            sum(float(row[name]) ** 2 for name in ("xacc", "yacc", "zacc"))
        )
        for row in rows
    ]
    return gyro_norms, accel_norms


def _find_stationary_start(
    rows: Sequence[dict[str, Any]], timestamps_ns: Sequence[int]
) -> int:
    """Find the earliest complete OpenVINS initialization window using IMU only."""
    minimum_rows = math.ceil(INITIALIZATION_MIN_RATE_HZ * INITIALIZATION_WINDOW_S)
    window_ns = int(round(INITIALIZATION_WINDOW_S * NS_PER_SECOND))
    end = 0
    for start in range(len(rows)):
        end = max(end, start)
        while end < len(rows) and timestamps_ns[end] - timestamps_ns[start] < window_ns:
            end += 1
        if end >= len(rows):
            break
        window_rows = rows[start : end + 1]
        if len(window_rows) < minimum_rows:
            continue
        gyro_norms, accel_norms = _imu_window_stats(window_rows)
        if max(gyro_norms) > INITIALIZATION_MAX_GYRO_NORM_RAD_S:
            continue
        if (
            abs(statistics.fmean(accel_norms) - 9.81)
            > INITIALIZATION_MAX_ACCEL_MEAN_ERROR_M_S2
        ):
            continue
        if statistics.pstdev(accel_norms) > AUTO_STATIONARY_MAX_ACCEL_STD_M_S2:
            continue
        return start
    raise DatasetError(
        "no usable two-second stationary IMU window was found; "
        "automatic selection uses IMU measurements only"
    )


def _apply_imu_gap_policy(
    rows: Sequence[dict[str, Any]],
    timestamps_ns: Sequence[int],
    *,
    policy: str,
    max_imu_gap_ms: float,
) -> tuple[
    list[dict[str, Any]],
    list[int],
    list[dict[str, Any]],
    int,
    float,
]:
    """Validate or interpolate long gaps without consulting truth data."""
    raw_gaps_ns = [
        current - previous
        for previous, current in zip(timestamps_ns, timestamps_ns[1:])
    ]
    if not raw_gaps_ns:
        raise DatasetError("not enough IMU rows to inspect timestamp gaps")
    raw_gap_max_ms = max(raw_gaps_ns) / 1e6
    effective_limit_ms = (
        IMU_GAP_REJECT_THRESHOLD_MS if policy == "reject" else max_imu_gap_ms
    )
    if raw_gap_max_ms > effective_limit_ms:
        raise DatasetError(
            f"IMU has a {raw_gap_max_ms:.3f} ms gap; "
            f"{policy} policy requires <= {effective_limit_ms:g} ms"
        )

    long_gap_indices = [
        index
        for index, gap_ns in enumerate(raw_gaps_ns)
        if gap_ns > IMU_GAP_REJECT_THRESHOLD_MS * 1e6
    ]
    gap_events: list[dict[str, Any]] = []
    if policy != "interpolate" or not long_gap_indices:
        for index in long_gap_indices:
            gap_events.append(
                {
                    "start_time_usec": int(timestamps_ns[index] // US_TO_NS),
                    "end_time_usec": int(timestamps_ns[index + 1] // US_TO_NS),
                    "raw_gap_ms": raw_gaps_ns[index] / 1e6,
                    "inserted_rows": 0,
                    "policy": policy,
                }
            )
        return list(rows), list(timestamps_ns), gap_events, 0, effective_limit_ms

    ordinary_gaps_ns = [
        gap_ns
        for gap_ns in raw_gaps_ns
        if gap_ns <= IMU_GAP_REJECT_THRESHOLD_MS * 1e6
    ]
    if not ordinary_gaps_ns:
        raise DatasetError("cannot infer the nominal IMU period for interpolation")
    nominal_gap_ns = statistics.median(ordinary_gaps_ns)
    output_rows: list[dict[str, Any]] = [dict(rows[0])]
    output_timestamps_ns = [int(timestamps_ns[0])]
    inserted_rows = 0
    sensor_fields = ("xacc", "yacc", "zacc", "xgyro", "ygyro", "zgyro")

    for index, (previous_row, current_row) in enumerate(zip(rows, rows[1:])):
        previous_ns = int(timestamps_ns[index])
        current_ns = int(timestamps_ns[index + 1])
        gap_ns = current_ns - previous_ns
        inserted_for_gap = 0
        if gap_ns > IMU_GAP_REJECT_THRESHOLD_MS * 1e6:
            segment_count = max(2, math.ceil(gap_ns / nominal_gap_ns))
            for segment in range(1, segment_count):
                fraction = segment / segment_count
                synthetic_ns = round(previous_ns + fraction * gap_ns)
                synthetic = dict(previous_row)
                synthetic["time_usec"] = synthetic_ns // US_TO_NS
                for field in sensor_fields:
                    start_value = float(previous_row[field])
                    end_value = float(current_row[field])
                    synthetic[field] = start_value + fraction * (
                        end_value - start_value
                    )
                if previous_row.get("wall_time_ns") and current_row.get("wall_time_ns"):
                    previous_wall_ns = int(previous_row["wall_time_ns"])
                    current_wall_ns = int(current_row["wall_time_ns"])
                    synthetic["wall_time_ns"] = round(
                        previous_wall_ns
                        + fraction * (current_wall_ns - previous_wall_ns)
                    )
                synthetic["interpolated"] = "1"
                output_rows.append(synthetic)
                output_timestamps_ns.append(synthetic_ns)
                inserted_rows += 1
                inserted_for_gap += 1
            gap_events.append(
                {
                    "start_time_usec": previous_ns // US_TO_NS,
                    "end_time_usec": current_ns // US_TO_NS,
                    "raw_gap_ms": gap_ns / 1e6,
                    "inserted_rows": inserted_for_gap,
                    "policy": policy,
                }
            )
        output_rows.append(dict(current_row))
        output_timestamps_ns.append(current_ns)

    _strictly_increasing(output_timestamps_ns, "gap-processed IMU timestamps")
    return (
        output_rows,
        output_timestamps_ns,
        gap_events,
        inserted_rows,
        effective_limit_ms,
    )


def prepare_dataset(
    dataset: Path,
    output: Path,
    *,
    imu_source: str = "highres_imu",
    max_clones: int = DEFAULT_MAX_CLONES,
    camera_stride: int = 1,
    max_msckf_in_update: int = DEFAULT_MAX_MSCKF_IN_UPDATE,
    max_slam: int = DEFAULT_MAX_SLAM,
    max_slam_in_update: int = DEFAULT_MAX_SLAM_IN_UPDATE,
    geometry_ranked_msckf: bool = False,
    camera_imu_timeoffset_ms: float = DEFAULT_CAMERA_IMU_TIME_OFFSET_MS,
    camera_focal_px: float | None = DEFAULT_CAMERA_FOCAL_PX,
    camera_tilt_up_deg: float = DEFAULT_CAMERA_TILT_UP_DEG,
    mask_top_rows: int = DEFAULT_MASK_TOP_ROWS,
    mask_guide_cone: bool = DEFAULT_MASK_GUIDE_CONE,
    camera_image_mode: str = DEFAULT_CAMERA_IMAGE_MODE,
    histogram_method: str = DEFAULT_HISTOGRAM_METHOD,
    num_pts: int = DEFAULT_NUM_PTS,
    fast_threshold: int = DEFAULT_FAST_THRESHOLD,
    min_px_dist: int = DEFAULT_MIN_PX_DIST,
    stationary_start: str = DEFAULT_STATIONARY_START,
    initialization_mode: str = DEFAULT_INITIALIZATION_MODE,
    imu_gap_policy: str = DEFAULT_IMU_GAP_POLICY,
    max_imu_gap_ms: float = DEFAULT_MAX_IMU_GAP_MS,
) -> dict[str, Any]:
    dataset = dataset.resolve()
    output = output.resolve()
    if output.exists():
        raise DatasetError(f"output already exists: {output}")
    if not 2 <= max_clones <= 100:
        raise DatasetError("max_clones must be between 2 and 100")
    if not 1 <= camera_stride <= 8:
        raise DatasetError("camera_stride must be between 1 and 8")
    if not 1 <= max_msckf_in_update <= 1000:
        raise DatasetError("max_msckf_in_update must be between 1 and 1000")
    if not 0 <= max_slam <= 500:
        raise DatasetError("max_slam must be between 0 and 500")
    if not 1 <= max_slam_in_update <= 500:
        raise DatasetError("max_slam_in_update must be between 1 and 500")
    if not 1 <= num_pts <= 5000:
        raise DatasetError("num_pts must be between 1 and 5000")
    if not 1 <= fast_threshold <= 255:
        raise DatasetError("fast_threshold must be between 1 and 255")
    if not 1 <= min_px_dist <= 100:
        raise DatasetError("min_px_dist must be between 1 and 100")
    if not math.isfinite(camera_imu_timeoffset_ms):
        raise DatasetError("camera_imu_timeoffset_ms must be finite")
    if not -100.0 <= camera_imu_timeoffset_ms <= 100.0:
        raise DatasetError("camera_imu_timeoffset_ms must be between -100 and 100")
    if camera_focal_px is not None:
        if not math.isfinite(camera_focal_px):
            raise DatasetError("camera_focal_px must be finite")
        if not 100.0 <= camera_focal_px <= 1000.0:
            raise DatasetError("camera_focal_px must be between 100 and 1000")
    if not math.isfinite(camera_tilt_up_deg):
        raise DatasetError("camera_tilt_up_deg must be finite")
    if not -89.0 <= camera_tilt_up_deg <= 89.0:
        raise DatasetError("camera_tilt_up_deg must be between -89 and 89")
    if not 0 <= mask_top_rows < 360:
        raise DatasetError("mask_top_rows must be between 0 and 359")
    if camera_image_mode not in CAMERA_IMAGE_MODES:
        raise DatasetError(
            "camera_image_mode must be one of: " + ", ".join(CAMERA_IMAGE_MODES)
        )
    if histogram_method not in HISTOGRAM_METHODS:
        raise DatasetError(
            "histogram_method must be one of: " + ", ".join(HISTOGRAM_METHODS)
        )
    if camera_image_mode == "red_fixed" and histogram_method != "NONE":
        raise DatasetError(
            "red_fixed camera image mode requires histogram_method NONE"
        )
    if camera_image_mode == "red_sidehist" and histogram_method != "NONE":
        raise DatasetError(
            "red_sidehist camera image mode requires histogram_method NONE"
        )
    if camera_image_mode == "red_sidehist" and (
        mask_top_rows != 0 or mask_guide_cone
    ):
        raise DatasetError(
            "red_sidehist camera image mode requires both spatial masks off"
        )
    if stationary_start not in STATIONARY_START_POLICIES:
        raise DatasetError(
            "stationary_start must be one of: "
            + ", ".join(STATIONARY_START_POLICIES)
        )
    if initialization_mode not in INITIALIZATION_MODES:
        raise DatasetError(
            "initialization_mode must be one of: "
            + ", ".join(INITIALIZATION_MODES)
        )
    if imu_gap_policy not in IMU_GAP_POLICIES:
        raise DatasetError(
            "imu_gap_policy must be one of: " + ", ".join(IMU_GAP_POLICIES)
        )
    if not math.isfinite(max_imu_gap_ms) or not 30.0 <= max_imu_gap_ms <= 1000.0:
        raise DatasetError("max_imu_gap_ms must be between 30 and 1000")
    camera_imu_timeoffset_ns = int(round(camera_imu_timeoffset_ms * 1e6))
    camera_imu_timeoffset_s = camera_imu_timeoffset_ns / NS_PER_SECOND

    manifest_path = dataset / "manifest.json"
    if not manifest_path.is_file():
        raise DatasetError(f"missing source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _check_source_manifest(dataset, manifest)

    imu_rows, imu_source_info = _load_imu_source(
        dataset, manifest, imu_source
    )
    camera_rows = _read_csv(dataset / manifest["files"]["camera_index"])
    timesync_rows = _read_csv(dataset / manifest["files"]["timesync"])
    truth_rows = _read_csv(dataset / manifest["files"]["odometry_truth"])

    imu_rows.sort(key=lambda row: int(row["time_usec"]))
    camera_rows.sort(key=lambda row: int(row["sim_time_ns"]))
    source_imu_rows = imu_rows
    source_imu_boot_ns = [
        int(row["time_usec"]) * US_TO_NS for row in source_imu_rows
    ]
    camera_server_ns = [int(row["sim_time_ns"]) for row in camera_rows]
    _strictly_increasing(source_imu_boot_ns, "IMU time_usec")
    _strictly_increasing(camera_server_ns, "camera sim_time_ns")

    clock_fit = _fit_timesync(timesync_rows)
    imu_offset_samples_ns = [
        clock_fit.local_to_server_ns(int(row["wall_time_ns"])) - boot_ns
        for row, boot_ns in zip(source_imu_rows, source_imu_boot_ns)
    ]
    imu_to_server_offset_ns = int(
        round(_percentile(imu_offset_samples_ns, IMU_OFFSET_QUANTILE))
    )

    stationary_start_index = 0
    if stationary_start == "auto":
        stationary_start_index = _find_stationary_start(
            source_imu_rows, source_imu_boot_ns
        )
    stationary_start_offset_s = (
        source_imu_boot_ns[stationary_start_index] - source_imu_boot_ns[0]
    ) / NS_PER_SECOND
    raw_imu_rows = source_imu_rows[stationary_start_index:]
    raw_imu_boot_ns = source_imu_boot_ns[stationary_start_index:]
    raw_imu_gaps_ms = [
        (current - previous) / 1e6
        for previous, current in zip(raw_imu_boot_ns, raw_imu_boot_ns[1:])
    ]
    (
        imu_rows,
        imu_boot_ns,
        imu_gap_events,
        interpolated_imu_rows,
        effective_max_imu_gap_ms,
    ) = _apply_imu_gap_policy(
        raw_imu_rows,
        raw_imu_boot_ns,
        policy=imu_gap_policy,
        max_imu_gap_ms=max_imu_gap_ms,
    )
    mapped_imu_ns = [boot_ns + imu_to_server_offset_ns for boot_ns in imu_boot_ns]
    origin_ns = mapped_imu_ns[0]
    overlap_indices = [
        index
        for index, timestamp_ns in enumerate(camera_server_ns)
        if (
            mapped_imu_ns[0]
            <= timestamp_ns + camera_imu_timeoffset_ns
            < mapped_imu_ns[-1]
        )
    ]
    if len(overlap_indices) < 100:
        raise DatasetError(
            f"only {len(overlap_indices)} camera frames overlap the IMU interval"
        )
    selected_camera_indices = overlap_indices[::camera_stride]
    first_camera_index = selected_camera_indices[0]
    last_camera_index = selected_camera_indices[-1]
    selected_camera_rows = [camera_rows[index] for index in selected_camera_indices]
    selected_camera_ns = [camera_server_ns[index] for index in selected_camera_indices]
    source_image_paths = _validate_jpegs(dataset, selected_camera_rows)

    future_imu_bracket_ms: list[float] = []
    imu_bracket_index = 0
    for camera_timestamp_ns in selected_camera_ns:
        effective_imu_timestamp_ns = (
            camera_timestamp_ns + camera_imu_timeoffset_ns
        )
        while (
            imu_bracket_index < len(mapped_imu_ns)
            and mapped_imu_ns[imu_bracket_index] <= effective_imu_timestamp_ns
        ):
            imu_bracket_index += 1
        if imu_bracket_index >= len(mapped_imu_ns):
            raise DatasetError(
                "a retained camera frame has no strictly newer IMU sample"
            )
        future_imu_bracket_ms.append(
            (
                mapped_imu_ns[imu_bracket_index]
                - effective_imu_timestamp_ns
            )
            / 1e6
        )

    raw_imu_rate_hz = _rate_hz(raw_imu_boot_ns)
    imu_rate_hz = _rate_hz(imu_boot_ns)
    camera_rate_hz = _rate_hz(selected_camera_ns)
    imu_gaps_ms = [
        (current - previous) / 1e6
        for previous, current in zip(imu_boot_ns, imu_boot_ns[1:])
    ]
    camera_gaps_ms = [
        (current - previous) / 1e6
        for previous, current in zip(selected_camera_ns, selected_camera_ns[1:])
    ]
    if raw_imu_rate_hz < 100.0:
        raise DatasetError(
            f"raw IMU rate is only {raw_imu_rate_hz:.3f} Hz; require >= 100 Hz"
        )
    if (
        imu_gap_policy == "interpolate"
        and max(imu_gaps_ms) > IMU_GAP_REJECT_THRESHOLD_MS
    ):
        raise DatasetError("IMU interpolation did not repair every accepted long gap")
    expected_camera_rate_hz = 30.0 / camera_stride
    if not 0.8 * expected_camera_rate_hz <= camera_rate_hz <= 1.2 * expected_camera_rate_hz:
        raise DatasetError(
            f"camera rate is {camera_rate_hz:.3f} Hz; expected about "
            f"{expected_camera_rate_hz:.3f} Hz for stride {camera_stride}"
        )

    initial_rows = [
        row
        for row, timestamp_ns in zip(imu_rows, imu_boot_ns)
        if timestamp_ns - imu_boot_ns[0] <= INITIALIZATION_WINDOW_S * NS_PER_SECOND
    ]
    gyro_norms, accel_norms = _imu_window_stats(initial_rows)
    minimum_initial_rows = math.ceil(
        INITIALIZATION_MIN_RATE_HZ * INITIALIZATION_WINDOW_S
    )
    if (
        len(initial_rows) < minimum_initial_rows
        or max(gyro_norms) > INITIALIZATION_MAX_GYRO_NORM_RAD_S
    ):
        raise DatasetError("the first two seconds are not a usable stationary IMU window")
    if (
        abs(statistics.fmean(accel_norms) - 9.81)
        > INITIALIZATION_MAX_ACCEL_MEAN_ERROR_M_S2
    ):
        raise DatasetError("initial accelerometer norm is inconsistent with gravity")

    source_camera = manifest["camera"]
    camera = dict(source_camera)
    if camera_focal_px is not None:
        camera["fx"] = float(camera_focal_px)
        camera["fy"] = float(camera_focal_px)
    if int(camera["width"]) != 640 or int(camera["height"]) != 360:
        raise DatasetError("the VQ1 calibration expects 640x360 images")
    if camera.get("mount_profile") != "competition":
        raise DatasetError("the VQ1 calibration expects the competition camera mount")

    temp_output = output.parent / f".{output.name}.tmp-{os.getpid()}"
    if temp_output.exists():
        raise DatasetError(f"temporary output path already exists: {temp_output}")
    temp_output.mkdir(parents=True)
    try:
        (temp_output / "cam0" / "data").mkdir(parents=True)
        (temp_output / "config").mkdir()

        _write_csv(
            temp_output / "imu.csv",
            ("timestamp", "wx", "wy", "wz", "ax", "ay", "az"),
            (
                (
                    _format_time(timestamp_ns, origin_ns),
                    imu_source_info["gyro_sign_xyz"][0] * float(row["xgyro"]),
                    imu_source_info["gyro_sign_xyz"][1] * float(row["ygyro"]),
                    imu_source_info["gyro_sign_xyz"][2] * float(row["zgyro"]),
                    imu_source_info["accel_sign_xyz"][0] * float(row["xacc"]),
                    imu_source_info["accel_sign_xyz"][1] * float(row["yacc"]),
                    imu_source_info["accel_sign_xyz"][2] * float(row["zacc"]),
                )
                for row, timestamp_ns in zip(imu_rows, mapped_imu_ns)
            ),
        )
        _write_csv(
            temp_output / "imu_gap_events.csv",
            (
                "start_time_usec",
                "end_time_usec",
                "raw_gap_ms",
                "inserted_rows",
                "policy",
            ),
            (
                (
                    event["start_time_usec"],
                    event["end_time_usec"],
                    event["raw_gap_ms"],
                    event["inserted_rows"],
                    event["policy"],
                )
                for event in imu_gap_events
            ),
        )

        link_modes: set[str] = set()
        camera_output_rows: list[tuple[str, str, int, int]] = []
        for sequence, (row, timestamp_ns, source) in enumerate(
            zip(selected_camera_rows, selected_camera_ns, source_image_paths), start=1
        ):
            output_name = f"{sequence:06d}_{int(row['frame_id']):06d}.jpg"
            link_modes.add(
                _link_or_copy(source, temp_output / "cam0" / "data" / output_name)
            )
            camera_output_rows.append(
                (
                    _format_time(timestamp_ns, origin_ns),
                    f"data/{output_name}",
                    int(row["frame_id"]),
                    int(timestamp_ns),
                )
            )
        _write_csv(
            temp_output / "cam0" / "data.csv",
            ("timestamp", "filename", "source_frame_id", "source_sim_time_ns"),
            camera_output_rows,
        )

        # Truth contains legitimate repeated MAVLink packets. Keep the last row
        # for each source timestamp; truth is evaluation-only and never fed to VIO.
        truth_by_time = {int(row["time_usec"]): row for row in truth_rows}
        truth_output_rows = []
        for time_usec in sorted(truth_by_time):
            row = truth_by_time[time_usec]
            timestamp_ns = time_usec * US_TO_NS + imu_to_server_offset_ns
            truth_output_rows.append(
                (
                    _format_time(timestamp_ns, origin_ns),
                    row["x"], row["y"], row["z"],
                    row["qx"], row["qy"], row["qz"], row["qw"],
                    row["vx"], row["vy"], row["vz"],
                )
            )
        _write_csv(
            temp_output / "truth.csv",
            ("timestamp", "px", "py", "pz", "qx", "qy", "qz", "qw", "vx", "vy", "vz"),
            truth_output_rows,
        )

        (temp_output / "config" / "estimator_config.yaml").write_text(
            _render_estimator_config(
                max_clones=max_clones,
                track_frequency_hz=camera_rate_hz,
                max_msckf_in_update=max_msckf_in_update,
                max_slam=max_slam,
                max_slam_in_update=max_slam_in_update,
                initialization_mode=initialization_mode,
                geometry_ranked_msckf=geometry_ranked_msckf,
                histogram_method=histogram_method,
                num_pts=num_pts,
                fast_threshold=fast_threshold,
                min_px_dist=min_px_dist,
            ),
            encoding="utf-8",
            newline="\n",
        )
        (temp_output / "config" / "kalibr_imu_chain.yaml").write_text(
            _render_imu_config(imu_rate_hz), encoding="utf-8", newline="\n"
        )
        (temp_output / "config" / "kalibr_imucam_chain.yaml").write_text(
            _render_camera_config(
                camera,
                camera_imu_timeoffset_s,
                camera_tilt_up_deg,
            ),
            encoding="utf-8",
            newline="\n",
        )

        report: dict[str, Any] = {
            "format": "aigp_openvins_replay",
            "format_version": 3,
            "source_dataset": str(dataset),
            "source_manifest_status": manifest["status"],
            "clock_alignment": {
                "camera_clock": "TIMESYNC server/simulator nanoseconds",
                "imu_clock": (
                    f"MAVLink {imu_source_info['source'].upper()} "
                    f"{imu_source_info['timestamp_kind']} microseconds"
                ),
                "method": "TIMESYNC low-RTT affine local/server fit plus 2% lower-envelope IMU receive offset",
                "timesync_pairs": clock_fit.pair_count,
                "timesync_pairs_used": clock_fit.selected_pair_count,
                "local_to_server_slope": clock_fit.slope,
                "local_to_server_drift_ppm": (clock_fit.slope - 1.0) * 1e6,
                "local_to_server_intercept_s": clock_fit.intercept_s,
                "timesync_rtt_median_ms": clock_fit.median_rtt_ms,
                "timesync_rtt_p95_ms": clock_fit.p95_rtt_ms,
                "imu_to_server_offset_ns": imu_to_server_offset_ns,
                "replay_origin_server_ns": origin_ns,
            },
            "imu": {
                "source": imu_source_info["source"],
                "source_file": imu_source_info["source_file"],
                "source_rows": imu_source_info["source_rows"],
                "unique_source_rows": imu_source_info["unique_rows"],
                "raw_rows_after_stationary_trim": len(raw_imu_rows),
                "rows": len(imu_rows),
                "rate_hz": imu_rate_hz,
                "raw_rate_hz": raw_imu_rate_hz,
                "span_s": (imu_boot_ns[-1] - imu_boot_ns[0]) / NS_PER_SECOND,
                "gap_median_ms": _percentile(imu_gaps_ms, 0.5),
                "gap_p95_ms": _percentile(imu_gaps_ms, 0.95),
                "gap_p99_ms": _percentile(imu_gaps_ms, 0.99),
                "gap_max_ms": max(imu_gaps_ms),
                "raw_gap_median_ms": _percentile(raw_imu_gaps_ms, 0.5),
                "raw_gap_p95_ms": _percentile(raw_imu_gaps_ms, 0.95),
                "raw_gap_p99_ms": _percentile(raw_imu_gaps_ms, 0.99),
                "raw_gap_max_ms": max(raw_imu_gaps_ms),
                "gap_policy": {
                    "mode": imu_gap_policy,
                    "normal_gap_threshold_ms": IMU_GAP_REJECT_THRESHOLD_MS,
                    "maximum_permitted_raw_gap_ms": effective_max_imu_gap_ms,
                    "long_gap_count": len(imu_gap_events),
                    "interpolated_rows": interpolated_imu_rows,
                    "interpolation_method": (
                        "piecewise linear between adjacent measured IMU samples"
                        if imu_gap_policy == "interpolate"
                        else None
                    ),
                    "truth_used": False,
                    "audit_file": "imu_gap_events.csv",
                },
                "initialization": {
                    "start_policy": stationary_start,
                    "source_rows_trimmed_before": stationary_start_index,
                    "source_start_offset_s": stationary_start_offset_s,
                    "window_s": INITIALIZATION_WINDOW_S,
                    "selection_inputs": "IMU only",
                    "truth_used": False,
                    "auto_accel_norm_std_limit_m_s2": (
                        AUTO_STATIONARY_MAX_ACCEL_STD_M_S2
                        if stationary_start == "auto"
                        else None
                    ),
                },
                "initial_stationary_rows": len(initial_rows),
                "initial_gyro_norm_max_rad_s": max(gyro_norms),
                "initial_accel_norm_mean_m_s2": statistics.fmean(accel_norms),
                "initial_accel_norm_std_m_s2": statistics.pstdev(accel_norms),
            },
            "camera": {
                "source_rows": len(camera_rows),
                "source_overlap_rows": len(overlap_indices),
                "overlap_rows": len(selected_camera_rows),
                "stride": camera_stride,
                "trimmed_before": first_camera_index,
                "trimmed_after": len(camera_rows) - last_camera_index - 1,
                "first_source_frame_id": int(selected_camera_rows[0]["frame_id"]),
                "last_source_frame_id": int(selected_camera_rows[-1]["frame_id"]),
                "rate_hz": camera_rate_hz,
                "span_s": (selected_camera_ns[-1] - selected_camera_ns[0]) / NS_PER_SECOND,
                "gap_median_ms": _percentile(camera_gaps_ms, 0.5),
                "gap_p95_ms": _percentile(camera_gaps_ms, 0.95),
                "gap_max_ms": max(camera_gaps_ms),
                "materialization": "+".join(sorted(link_modes)),
                "future_imu_bracket": {
                    "required_by_runner": True,
                    "rows": len(future_imu_bracket_ms),
                    "lead_median_ms": _percentile(future_imu_bracket_ms, 0.5),
                    "lead_p95_ms": _percentile(future_imu_bracket_ms, 0.95),
                    "lead_max_ms": max(future_imu_bracket_ms),
                },
            },
            "estimator_policy": {
                "camera_imu_timeoffset_calibration": False,
                "configured_camera_to_imu_timeoffset_s": camera_imu_timeoffset_s,
                "configured_camera_to_imu_timeoffset_ms": camera_imu_timeoffset_ms,
                "camera_focal_override_px": camera_focal_px,
                "camera_tilt_up_deg": camera_tilt_up_deg,
                "feature_max_distance_m": FEATURE_MAX_DISTANCE_M,
                "max_clones": max_clones,
                "max_msckf_in_update": max_msckf_in_update,
                "max_slam": max_slam,
                "max_slam_in_update": max_slam_in_update,
                "geometry_ranked_msckf": geometry_ranked_msckf,
                "num_pts": num_pts,
                "fast_threshold": fast_threshold,
                "min_px_dist": min_px_dist,
                "camera_stride": camera_stride,
                "mask_top_rows": mask_top_rows,
                "mask_guide_cone": bool(mask_guide_cone),
                "camera_image_mode": camera_image_mode,
                "camera_image_preprocessing": (
                    "decode color JPEG, extract OpenCV BGR channel 2 (red), "
                    "calculate each frame's equalization LUT using only pixels "
                    "outside the guide-cone envelope, and apply it to the full "
                    "red image without masking any features"
                    if camera_image_mode == "red_sidehist"
                    else (
                        "decode color JPEG, extract OpenCV BGR channel 2 (red), "
                        "derive one bounded gamma LUT from the first two seconds "
                        "with target median 48/255, and reuse that LUT unchanged"
                        if camera_image_mode == "red_fixed"
                        else (
                            "decode color JPEG and feed the OpenCV BGR channel at "
                            "index 2, corresponding to red"
                            if camera_image_mode == "red"
                            else "decode JPEG directly as OpenCV grayscale"
                        )
                    )
                ),
                "histogram_method": histogram_method,
                "guide_cone_mask_reference_resolution": (
                    GUIDE_CONE_MASK_REFERENCE_RESOLUTION
                ),
                "guide_cone_mask_polygon_normalized": (
                    GUIDE_CONE_MASK_POLYGON_NORMALIZED
                ),
                "stationary_start": stationary_start,
                "initialization_mode": initialization_mode,
                "try_zupt": initialization_mode == "stationary",
                "zupt_only_at_beginning": initialization_mode == "stationary",
                "imu_gap_policy": imu_gap_policy,
                "max_imu_gap_ms": effective_max_imu_gap_ms,
                "feature_mask_activation": (
                    "guide_cone_from_first_camera_frame_and_top_rows_after_"
                    "estimator_initialization"
                    if mask_guide_cone and mask_top_rows > 0
                    else (
                        "from_first_camera_frame"
                        if mask_guide_cone
                        else "after_estimator_initialization"
                    )
                ),
                "track_frequency_hz": camera_rate_hz,
                "camera_feed_policy": (
                    "feed every IMU through the first sample strictly newer "
                    "than camera timestamp plus the configured camera-to-IMU "
                    "time offset before feeding that camera"
                ),
                "reason": (
                    "offline clocks are already aligned; freezing the offset "
                    "and bracketing camera times prevents IMU extrapolation. "
                    + (
                        "Startup-only zero-velocity support permits static "
                        "initialization without requiring a large takeoff jerk."
                        if initialization_mode == "stationary"
                        else "The default initializer waits for a visual and inertial jerk."
                    )
                ),
            },
            "calibration": {
                "camera_model": "pinhole-radtan",
                "intrinsics": [camera[key] for key in ("fx", "fy", "cx", "cy")],
                "source_intrinsics": [
                    source_camera[key] for key in ("fx", "fy", "cx", "cy")
                ],
                "focal_length_override_px": camera_focal_px,
                "distortion_coeffs": camera["dist_coeffs"][:4],
                "resolution": [camera["width"], camera["height"]],
                "camera_tilt_up_deg": camera_tilt_up_deg,
                "nominal_camera_tilt_up_deg": DEFAULT_CAMERA_TILT_UP_DEG,
                "R_camera_to_imu_frd": _camera_to_imu_matrix(
                    camera_tilt_up_deg
                ),
                "p_camera_in_imu_m": camera["body_translation_m"],
                "imu_frame": "MAVLink body FRD",
                "imu_input_transform": {
                    "profile": imu_source_info["profile"],
                    "source": f"MAVLink {imu_source_info['source'].upper()}",
                    "angular_velocity_output_from_source": [
                        f"{'+' if sign >= 0.0 else '-'}{axis}gyro"
                        for axis, sign in zip(
                            "xyz", imu_source_info["gyro_sign_xyz"]
                        )
                    ],
                    "gyroscope_sign_xyz": list(
                        imu_source_info["gyro_sign_xyz"]
                    ),
                    "accelerometer_output_from_source": [
                        "+xacc",
                        "+yacc",
                        "+zacc",
                    ],
                    "accelerometer_sign_xyz": list(
                        imu_source_info["accel_sign_xyz"]
                    ),
                    "reason": imu_source_info["reason"],
                    "source_capture_modified": False,
                },
            },
            "truth": {
                "source_rows": len(truth_rows),
                "unique_rows": len(truth_output_rows),
                "frame": "MAVLink local NED / body FRD",
                "fed_to_estimator": False,
            },
            "validation": "pass",
        }
        (temp_output / "manifest.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temp_output.replace(output)
    except Exception:
        shutil.rmtree(temp_output, ignore_errors=True)
        raise

    return report


def _default_runs_root() -> Path:
    return Path(__file__).resolve().parents[1] / "logs" / "runs"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help="run ID under aigp/logs/runs")
    source.add_argument("--dataset", type=Path, help="path to a vio_dataset directory")
    parser.add_argument("--output", type=Path, help="output replay directory")
    parser.add_argument(
        "--imu-source",
        choices=SUPPORTED_IMU_SOURCES,
        default="highres_imu",
        help=(
            "captured MAVLink IMU source (default: highres_imu); RAW_IMU "
            "capture is diagnostic-only until calibrated"
        ),
    )
    parser.add_argument(
        "--max-clones",
        type=int,
        default=DEFAULT_MAX_CLONES,
        help=f"number of cloned camera poses (default: {DEFAULT_MAX_CLONES})",
    )
    parser.add_argument(
        "--camera-stride",
        type=int,
        default=1,
        help="retain every Nth camera frame while keeping the full-rate IMU (default: 1)",
    )
    parser.add_argument(
        "--max-msckf-in-update",
        type=int,
        default=DEFAULT_MAX_MSCKF_IN_UPDATE,
        help=(
            "maximum MSCKF features processed in one update "
            f"(default: {DEFAULT_MAX_MSCKF_IN_UPDATE})"
        ),
    )
    parser.add_argument(
        "--max-slam",
        type=int,
        default=DEFAULT_MAX_SLAM,
        help=(
            "maximum persistent SLAM landmarks; use 0 for pure MSCKF "
            f"(default: {DEFAULT_MAX_SLAM})"
        ),
    )
    parser.add_argument(
        "--max-slam-in-update",
        type=int,
        default=DEFAULT_MAX_SLAM_IN_UPDATE,
        help=(
            "maximum persistent SLAM landmarks used in one update "
            f"(default: {DEFAULT_MAX_SLAM_IN_UPDATE})"
        ),
    )
    parser.add_argument(
        "--geometry-ranked-msckf",
        action="store_true",
        help=(
            "apply the MSCKF feature cap after geometric and chi-square "
            "validation using deterministic 3x3 image-grid ranking"
        ),
    )
    parser.add_argument(
        "--num-pts",
        type=int,
        default=DEFAULT_NUM_PTS,
        help=f"target number of KLT features (default: {DEFAULT_NUM_PTS})",
    )
    parser.add_argument(
        "--fast-threshold",
        type=int,
        default=DEFAULT_FAST_THRESHOLD,
        help=f"FAST detector threshold (default: {DEFAULT_FAST_THRESHOLD})",
    )
    parser.add_argument(
        "--min-px-dist",
        type=int,
        default=DEFAULT_MIN_PX_DIST,
        help=(
            "minimum pixel spacing between detected features "
            f"(default: {DEFAULT_MIN_PX_DIST})"
        ),
    )
    parser.add_argument(
        "--camera-imu-offset-ms",
        type=float,
        default=DEFAULT_CAMERA_IMU_TIME_OFFSET_MS,
        help=(
            "OpenVINS camera-to-IMU time offset in milliseconds, using "
            "imu_time = camera_time + offset (default: 0)"
        ),
    )
    parser.add_argument(
        "--camera-focal-px",
        type=float,
        default=DEFAULT_CAMERA_FOCAL_PX,
        help=(
            "override both fx and fy while keeping cx/cy and distortion fixed; "
            "intended only for controlled calibration sweeps"
        ),
    )
    parser.add_argument(
        "--camera-tilt-up-deg",
        type=float,
        default=DEFAULT_CAMERA_TILT_UP_DEG,
        help=(
            "override camera mounting pitch in T_imu_cam while preserving "
            "the optical-to-FRD convention and lever arm; intended only for "
            "controlled extrinsic sweeps (default: 20)"
        ),
    )
    parser.add_argument(
        "--mask-top-rows",
        type=int,
        default=DEFAULT_MASK_TOP_ROWS,
        help=(
            "exclude this many rows from the top of each 360-row camera image "
            f"during tracking (default: {DEFAULT_MASK_TOP_ROWS})"
        ),
    )
    parser.add_argument(
        "--mask-guide-cone",
        action="store_true",
        help=(
            "exclude the animated central guide-cone trapezoid from every "
            "camera frame, including estimator initialization"
        ),
    )
    parser.add_argument(
        "--camera-image-mode",
        choices=CAMERA_IMAGE_MODES,
        default=DEFAULT_CAMERA_IMAGE_MODE,
        help=(
            "monochrome image supplied to OpenVINS: normal grayscale, raw red "
            "channel, red with one fixed startup-derived contrast LUT, or red "
            "with per-frame equalization statistics taken only from outside "
            "the guide envelope (default: grayscale)"
        ),
    )
    parser.add_argument(
        "--histogram-method",
        choices=HISTOGRAM_METHODS,
        default=DEFAULT_HISTOGRAM_METHOD,
        help=(
            "OpenVINS image histogram preprocessing: NONE, HISTOGRAM, or "
            "CLAHE (default: HISTOGRAM)"
        ),
    )
    parser.add_argument(
        "--stationary-start",
        choices=STATIONARY_START_POLICIES,
        default=DEFAULT_STATIONARY_START,
        help=(
            "use the first IMU row or automatically trim to the earliest "
            "two-second stationary IMU-only window (default: first)"
        ),
    )
    parser.add_argument(
        "--initialization-mode",
        choices=INITIALIZATION_MODES,
        default=DEFAULT_INITIALIZATION_MODE,
        help=(
            "wait for a large visual/inertial jerk, or enable startup-only "
            "zero-velocity support for static initialization (default: jerk)"
        ),
    )
    parser.add_argument(
        "--imu-gap-policy",
        choices=IMU_GAP_POLICIES,
        default=DEFAULT_IMU_GAP_POLICY,
        help=(
            "reject gaps over 30 ms, allow an audited raw gap, or fill it by "
            "linear interpolation between adjacent IMU samples (default: reject)"
        ),
    )
    parser.add_argument(
        "--max-imu-gap-ms",
        type=float,
        default=DEFAULT_MAX_IMU_GAP_MS,
        help=(
            "largest raw gap accepted by allow/interpolate policies "
            f"(default: {DEFAULT_MAX_IMU_GAP_MS:g} ms)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.run:
        dataset = _default_runs_root() / args.run / "vio_dataset"
    else:
        dataset = args.dataset
    if args.output:
        output = args.output
    elif args.imu_source == "highres_imu":
        output = dataset.parent / "openvins_replay_imuframefix_bracketed"
    else:
        output = dataset.parent / f"openvins_replay_{args.imu_source}_bracketed"
    if not args.output and args.camera_stride != 1:
        output = output.with_name(f"{output.name}_stride{args.camera_stride}")
    if not args.output and args.geometry_ranked_msckf:
        output = output.with_name(f"{output.name}_geoselect")
    if not args.output and args.max_slam != DEFAULT_MAX_SLAM:
        output = output.with_name(f"{output.name}_slam{args.max_slam}")
    if (
        not args.output
        and args.max_slam_in_update != DEFAULT_MAX_SLAM_IN_UPDATE
    ):
        output = output.with_name(
            f"{output.name}_slamupd{args.max_slam_in_update}"
        )
    if not args.output and (
        args.num_pts != DEFAULT_NUM_PTS
        or args.fast_threshold != DEFAULT_FAST_THRESHOLD
        or args.min_px_dist != DEFAULT_MIN_PX_DIST
    ):
        output = output.with_name(
            f"{output.name}_pts{args.num_pts}_fast{args.fast_threshold}"
            f"_px{args.min_px_dist}"
        )
    if not args.output and args.camera_imu_offset_ms != 0.0:
        sign = "p" if args.camera_imu_offset_ms > 0.0 else "m"
        magnitude = f"{abs(args.camera_imu_offset_ms):g}".replace(".", "p")
        output = output.with_name(f"{output.name}_dt{sign}{magnitude}ms")
    if not args.output and args.camera_focal_px is not None:
        focal = f"{args.camera_focal_px:g}".replace(".", "p")
        output = output.with_name(f"{output.name}_fx{focal}")
    if (
        not args.output
        and args.camera_tilt_up_deg != DEFAULT_CAMERA_TILT_UP_DEG
    ):
        sign = "" if args.camera_tilt_up_deg >= 0.0 else "m"
        magnitude = f"{abs(args.camera_tilt_up_deg):g}".replace(".", "p")
        output = output.with_name(f"{output.name}_tilt{sign}{magnitude}deg")
    if not args.output and args.mask_top_rows != 0:
        output = output.with_name(f"{output.name}_masktop{args.mask_top_rows}")
    if not args.output and args.mask_guide_cone:
        output = output.with_name(f"{output.name}_maskguidecone")
    if not args.output and args.camera_image_mode != DEFAULT_CAMERA_IMAGE_MODE:
        image_suffix = (
            "redchannel"
            if args.camera_image_mode == "red"
            else (
                "redfixed"
                if args.camera_image_mode == "red_fixed"
                else "redsidehist"
            )
        )
        output = output.with_name(f"{output.name}_{image_suffix}")
    if not args.output and args.histogram_method != DEFAULT_HISTOGRAM_METHOD:
        output = output.with_name(f"{output.name}_hist{args.histogram_method.lower()}")
    if not args.output and args.stationary_start != DEFAULT_STATIONARY_START:
        output = output.with_name(f"{output.name}_autostart")
    if not args.output and args.initialization_mode != DEFAULT_INITIALIZATION_MODE:
        output = output.with_name(
            f"{output.name}_init{args.initialization_mode}"
        )
    if not args.output and args.imu_gap_policy != DEFAULT_IMU_GAP_POLICY:
        gap_limit = f"{args.max_imu_gap_ms:g}".replace(".", "p")
        output = output.with_name(
            f"{output.name}_gap{args.imu_gap_policy}{gap_limit}ms"
        )
    try:
        report = prepare_dataset(
            dataset,
            output,
            imu_source=args.imu_source,
            max_clones=args.max_clones,
            camera_stride=args.camera_stride,
            max_msckf_in_update=args.max_msckf_in_update,
            max_slam=args.max_slam,
            max_slam_in_update=args.max_slam_in_update,
            geometry_ranked_msckf=args.geometry_ranked_msckf,
            num_pts=args.num_pts,
            fast_threshold=args.fast_threshold,
            min_px_dist=args.min_px_dist,
            camera_imu_timeoffset_ms=args.camera_imu_offset_ms,
            camera_focal_px=args.camera_focal_px,
            camera_tilt_up_deg=args.camera_tilt_up_deg,
            mask_top_rows=args.mask_top_rows,
            mask_guide_cone=args.mask_guide_cone,
            camera_image_mode=args.camera_image_mode,
            histogram_method=args.histogram_method,
            stationary_start=args.stationary_start,
            initialization_mode=args.initialization_mode,
            imu_gap_policy=args.imu_gap_policy,
            max_imu_gap_ms=args.max_imu_gap_ms,
        )
    except (DatasetError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"OpenVINS replay dataset: {output.resolve()}")
    print(
        f"IMU {report['imu']['source']}: {report['imu']['rows']} rows at "
        f"{report['imu']['rate_hz']:.3f} Hz; "
        f"max gap {report['imu']['gap_max_ms']:.3f} ms; raw max "
        f"{report['imu']['raw_gap_max_ms']:.3f} ms; "
        f"policy {report['imu']['gap_policy']['mode']}"
    )
    print(
        "Initialization: "
        f"{report['imu']['initialization']['start_policy']} start, "
        f"{report['estimator_policy']['initialization_mode']} mode, trimmed "
        f"{report['imu']['initialization']['source_rows_trimmed_before']} IMU rows "
        f"({report['imu']['initialization']['source_start_offset_s']:.3f} s)"
    )
    print(
        f"Camera: {report['camera']['overlap_rows']} overlap frames at "
        f"{report['camera']['rate_hz']:.3f} Hz; source IDs "
        f"{report['camera']['first_source_frame_id']}.."
        f"{report['camera']['last_source_frame_id']}"
    )
    print(
        "IMU brackets: "
        f"{report['camera']['future_imu_bracket']['rows']} frames, "
        f"median future lead "
        f"{report['camera']['future_imu_bracket']['lead_median_ms']:.3f} ms, "
        "time-offset calibration disabled"
    )
    print(
        "Clock: "
        f"{report['clock_alignment']['timesync_pairs_used']} low-RTT pairs, "
        f"{report['clock_alignment']['local_to_server_drift_ppm']:.3f} ppm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
