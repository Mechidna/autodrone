#!/usr/bin/env python3
"""Evaluate gate-map corrections on a recorded OpenVINS trajectory.

This is deliberately an offline shadow experiment.  The detector and alignment
receive camera images, OpenVINS poses, and the configured gate map.  Captured
MAVLink truth is loaded on a separate path and is used only after each estimate
has been produced, to score the result and audit PnP associations.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import sys
from collections import Counter, deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np


_TOOLS_DIR = Path(__file__).resolve().parent
_AIGP_DIR = _TOOLS_DIR.parent
_REPO_ROOT = _AIGP_DIR.parent
for _path in (_TOOLS_DIR, _AIGP_DIR / "pilot", _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from diagnose_openvins_replay import (  # noqa: E402
    DiagnosticError,
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
RAW_DETECTION_FIELDS = (
    "timestamp",
    "frame_id",
    "detection_index",
    "confidence",
    "memory_confidence",
    "yolo_confidence",
    "reprojection_error",
    "quad_area_px2",
    "camera_x",
    "camera_y",
    "camera_z",
    "camera_corrected_x",
    "camera_corrected_y",
    "camera_corrected_z",
    "body_x",
    "body_y",
    "body_z",
    "pnp_solver",
    "pnp_order",
)


class ExperimentError(RuntimeError):
    """The requested replay cannot be evaluated safely."""


@dataclass(frozen=True)
class AlignmentOptions:
    min_confidence: float
    max_reprojection_error: float
    min_depth_m: float
    max_depth_m: float
    initial_association_radius_m: float
    association_radius_m: float
    temporal_window_s: float
    min_consistent_frames: int
    consistency_radius_m: float
    max_initial_offset_m: float
    max_update_innovation_m: float
    correction_alpha: float
    max_step_m: float
    stale_after_s: float

    @classmethod
    def from_config(cls, section: Any) -> "AlignmentOptions":
        return cls(
            min_confidence=float(section.min_confidence),
            max_reprojection_error=float(section.max_reprojection_error),
            min_depth_m=float(section.min_depth_m),
            max_depth_m=float(section.max_depth_m),
            initial_association_radius_m=float(
                section.initial_association_radius_m
            ),
            association_radius_m=float(section.association_radius_m),
            temporal_window_s=float(section.temporal_window_s),
            min_consistent_frames=int(section.min_consistent_frames),
            consistency_radius_m=float(section.consistency_radius_m),
            max_initial_offset_m=float(section.max_initial_offset_m),
            max_update_innovation_m=float(section.max_update_innovation_m),
            correction_alpha=float(section.correction_alpha),
            max_step_m=float(section.max_step_m),
            stale_after_s=float(section.stale_after_s),
        )


@dataclass(frozen=True)
class _OffsetObservation:
    timestamp: float
    offset_neu: np.ndarray
    association_residual_m: float
    detection_index: int


class OfflineGateMapAligner:
    """Translation-only map alignment with no access to evaluation truth."""

    def __init__(
        self,
        known_gates_neu: Sequence[Sequence[float]],
        options: AlignmentOptions,
    ) -> None:
        self.known_gates = np.asarray(known_gates_neu, dtype=float).reshape(-1, 3)
        if self.known_gates.shape[0] == 0:
            raise ExperimentError("the runtime config contains no known gate map")
        if not np.all(np.isfinite(self.known_gates)):
            raise ExperimentError("the known gate map contains non-finite values")
        self.options = options
        self.offset_neu = np.zeros(3, dtype=float)
        self.initialized = False
        self.last_accepted_timestamp: Optional[float] = None
        self.history_by_gate: dict[int, deque[_OffsetObservation]] = {}

    def update(
        self,
        timestamp: float,
        raw_position_neu: np.ndarray,
        detections: list[dict[str, Any]],
    ) -> dict[str, Any]:
        timestamp = float(timestamp)
        raw_position = _vec3(raw_position_neu, "raw OpenVINS position")
        self._prune(timestamp)
        radius = (
            self.options.association_radius_m
            if self.initialized
            else self.options.initial_association_radius_m
        )
        predicted_vehicle = raw_position + (
            self.offset_neu if self.initialized else 0.0
        )
        observations: dict[int, _OffsetObservation] = {}

        for detection in detections:
            detection["geometry_ok"] = self._geometry_ok(detection)
            detection["alignment_gate_index"] = None
            detection["alignment_association_residual_m"] = None
            detection["candidate_offset_neu"] = None
            detection["alignment_candidate_ok"] = False
            if not detection["geometry_ok"]:
                continue

            rel_neu = _vec3(detection["relative_gate_neu"], "relative gate")
            predicted_gate = predicted_vehicle + rel_neu
            gate_index, association_residual = self._nearest_gate(predicted_gate)
            detection["alignment_gate_index"] = gate_index
            detection["alignment_association_residual_m"] = association_residual
            if association_residual > radius:
                continue

            candidate_offset = (
                self.known_gates[gate_index] - raw_position - rel_neu
            )
            detection["candidate_offset_neu"] = candidate_offset.copy()
            if (
                not self.initialized
                and np.linalg.norm(candidate_offset)
                > self.options.max_initial_offset_m
            ):
                continue
            if (
                self.initialized
                and np.linalg.norm(candidate_offset - self.offset_neu)
                > self.options.max_update_innovation_m
            ):
                continue
            detection["alignment_candidate_ok"] = True
            observation = _OffsetObservation(
                timestamp=timestamp,
                offset_neu=candidate_offset,
                association_residual_m=association_residual,
                detection_index=int(detection["detection_index"]),
            )
            prior = observations.get(gate_index)
            if prior is None or (
                observation.association_residual_m
                < prior.association_residual_m
            ):
                observations[gate_index] = observation

        for gate_index, observation in observations.items():
            self.history_by_gate.setdefault(gate_index, deque()).append(observation)
        self._prune(timestamp)

        candidates: list[tuple[int, float, float, int, np.ndarray]] = []
        for gate_index, current in observations.items():
            consistent = self._consistent_candidate(gate_index, current)
            if consistent is None:
                continue
            target, support, spread = consistent
            candidates.append(
                (
                    -support,
                    spread,
                    current.association_residual_m,
                    gate_index,
                    target,
                )
            )

        accepted = False
        accepted_gate: Optional[int] = None
        support = 0
        spread: Optional[float] = None
        association_residual: Optional[float] = None
        if candidates:
            (
                negative_support,
                spread,
                association_residual,
                accepted_gate,
                target,
            ) = min(candidates, key=lambda item: item[:3])
            support = -negative_support
            if not self.initialized:
                self.offset_neu = target.copy()
                self.initialized = True
                reason = "initialized"
            else:
                innovation = target - self.offset_neu
                step = self.options.correction_alpha * innovation
                step_norm = float(np.linalg.norm(step))
                if step_norm > self.options.max_step_m:
                    step *= self.options.max_step_m / max(step_norm, 1e-12)
                self.offset_neu = self.offset_neu + step
                reason = "updated"
            accepted = True
            self.last_accepted_timestamp = timestamp
        elif not observations:
            reason = (
                "no_geometry_measurements"
                if not any(item.get("geometry_ok") for item in detections)
                else "no_gated_measurements"
            )
        else:
            reason = "awaiting_temporal_consistency"
            support = max(
                (
                    self._inlier_support(gate_index)
                    for gate_index in observations
                ),
                default=0,
            )

        age = (
            None
            if self.last_accepted_timestamp is None
            else max(0.0, timestamp - self.last_accepted_timestamp)
        )
        fresh = bool(
            self.initialized
            and age is not None
            and age <= self.options.stale_after_s
        )
        if self.initialized and not accepted and not fresh:
            reason = f"stale:{reason}"
        aligned = (
            raw_position + self.offset_neu
            if self.initialized
            else raw_position.copy()
        )
        return {
            "accepted": accepted,
            "initialized": self.initialized,
            "fresh": fresh,
            "reason": reason,
            "gate_index": accepted_gate,
            "support": support,
            "spread_m": spread,
            "association_residual_m": association_residual,
            "last_update_age_s": age,
            "offset_neu": self.offset_neu.copy(),
            "aligned_position_neu": aligned,
        }

    def _geometry_ok(self, detection: dict[str, Any]) -> bool:
        confidence = 0.0
        for key in ("memory_confidence", "confidence", "yolo_confidence"):
            candidate = _finite_float(detection.get(key))
            if candidate is not None:
                confidence = max(0.0, min(candidate, 1.0))
                break
        if confidence < self.options.min_confidence:
            return False
        reprojection = _finite_float(detection.get("reprojection_error"))
        if (
            reprojection is not None
            and self.options.max_reprojection_error > 0.0
            and reprojection > self.options.max_reprojection_error
        ):
            return False
        camera = np.asarray(detection.get("gate_center_camera"), dtype=float).reshape(3)
        if not np.all(np.isfinite(camera)):
            return False
        depth = float(camera[2])
        if depth <= 0.0 or depth < self.options.min_depth_m:
            return False
        return not (
            self.options.max_depth_m > 0.0
            and depth > self.options.max_depth_m
        )

    def _nearest_gate(self, point_neu: np.ndarray) -> tuple[int, float]:
        distances = np.linalg.norm(self.known_gates - point_neu, axis=1)
        gate_index = int(np.argmin(distances))
        return gate_index, float(distances[gate_index])

    def _prune(self, timestamp: float) -> None:
        cutoff = timestamp - self.options.temporal_window_s
        empty = []
        for gate_index, history in self.history_by_gate.items():
            while history and history[0].timestamp < cutoff:
                history.popleft()
            if not history:
                empty.append(gate_index)
        for gate_index in empty:
            self.history_by_gate.pop(gate_index, None)

    def _inlier_support(self, gate_index: int) -> int:
        history = self.history_by_gate.get(gate_index)
        if not history:
            return 0
        offsets = np.asarray([item.offset_neu for item in history])
        median = np.median(offsets, axis=0)
        return int(
            np.count_nonzero(
                np.linalg.norm(offsets - median, axis=1)
                <= self.options.consistency_radius_m
            )
        )

    def _consistent_candidate(
        self,
        gate_index: int,
        current: _OffsetObservation,
    ) -> Optional[tuple[np.ndarray, int, float]]:
        history = self.history_by_gate.get(gate_index)
        if not history:
            return None
        offsets = np.asarray([item.offset_neu for item in history])
        median = np.median(offsets, axis=0)
        deviations = np.linalg.norm(offsets - median, axis=1)
        inliers = deviations <= self.options.consistency_radius_m
        support = int(np.count_nonzero(inliers))
        if support < self.options.min_consistent_frames:
            return None
        if (
            np.linalg.norm(current.offset_neu - median)
            > self.options.consistency_radius_m
        ):
            return None
        target = np.median(offsets[inliers], axis=0)
        spread = float(np.max(np.linalg.norm(offsets[inliers] - target, axis=1)))
        return target, support, spread


def _finite_float(value: Any, *, default: Optional[float] = None) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _vec3(value: Any, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float).reshape(3)
    except (TypeError, ValueError) as exc:
        raise ExperimentError(f"{label} is not a three-vector") from exc
    if not np.all(np.isfinite(result)):
        raise ExperimentError(f"{label} contains non-finite values")
    return result.copy()


def _ned_to_neu(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=float).copy()
    output[..., 2] *= -1.0
    return output


def _summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "p95": float(np.percentile(finite, 95.0)),
        "max": float(np.max(finite)),
    }


def _read_camera_rows(replay: Path) -> list[dict[str, Any]]:
    path = replay / "cam0" / "data.csv"
    if not path.is_file():
        raise ExperimentError(f"missing camera index: {path}")
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp", "filename", "source_frame_id"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ExperimentError(
                f"camera index is missing columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            rows.append(
                {
                    "timestamp": float(row["timestamp"]),
                    "filename": str(row["filename"]),
                    "frame_id": int(row["source_frame_id"]),
                    "sim_time_ns": int(row.get("source_sim_time_ns") or 0),
                }
            )
    if not rows:
        raise ExperimentError("camera index contains no rows")
    return rows


def _match_camera_rows(
    camera_rows: list[dict[str, Any]],
    estimate_times: np.ndarray,
) -> list[dict[str, Any]]:
    camera_times = np.asarray([row["timestamp"] for row in camera_rows])
    indices = np.searchsorted(camera_times, estimate_times)
    result = []
    worst_delta = 0.0
    for timestamp, insertion in zip(estimate_times, indices):
        candidates = {
            max(0, min(int(insertion), camera_times.size - 1)),
            max(0, min(int(insertion) - 1, camera_times.size - 1)),
        }
        index = min(candidates, key=lambda value: abs(camera_times[value] - timestamp))
        delta = abs(float(camera_times[index] - timestamp))
        worst_delta = max(worst_delta, delta)
        result.append(camera_rows[index])
    if worst_delta > 1e-6:
        raise ExperimentError(
            "camera and OpenVINS outputs do not share timestamps; worst delta "
            f"is {worst_delta * 1000.0:.3f} ms"
        )
    if len({row["frame_id"] for row in result}) != len(result):
        raise ExperimentError("camera-to-estimate matching reused a frame")
    return result


def _source_runtime_config(replay: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise ExperimentError(f"runtime config does not exist: {explicit}")
        return explicit.resolve()
    snapshot = replay.parent / "vio_dataset" / "runtime.toml"
    if snapshot.is_file():
        return snapshot.resolve()
    fallback = _AIGP_DIR / "config" / "runtime.toml"
    if fallback.is_file():
        return fallback.resolve()
    raise ExperimentError("unable to find a runtime config snapshot")


def _load_openvins_trajectory(replay: Path) -> dict[str, Any]:
    estimate = _read_numeric_csv(replay / "openvins_estimate.csv", ESTIMATE_FIELDS)
    estimate_times = estimate[:, 0]
    if np.any(np.diff(estimate_times) <= 0.0):
        raise ExperimentError("OpenVINS estimate timestamps are not increasing")
    truth = replay_truth_local_ned(replay, estimate_times)
    if truth is None:
        raise ExperimentError("captured local-position truth is unavailable")
    truth_position_ned, truth_velocity_ned, translation_method = truth
    fallback = np.tile(np.asarray((0.0, 0.0, 0.0, 1.0)), (estimate.shape[0], 1))
    truth_body_to_ned, orientation_method = replay_truth_body_to_ned(
        replay, estimate_times, fallback
    )
    estimate_global_to_imu = np.transpose(
        quaternion_xyzw_to_matrix(estimate[:, 1:5]), (0, 2, 1)
    )
    aligned_position_ned, _ = _startup_aligned_position_errors(
        estimate[:, 5:8],
        estimate_global_to_imu,
        truth_position_ned,
        truth_body_to_ned,
    )
    rotation_global_to_ned = truth_body_to_ned[0] @ estimate_global_to_imu[0]
    estimate_body_to_ned = np.einsum(
        "ij,njk->nik",
        rotation_global_to_ned,
        np.transpose(estimate_global_to_imu, (0, 2, 1)),
    )
    aligned_velocity_ned = np.einsum(
        "ij,nj->ni", rotation_global_to_ned, estimate[:, 8:11]
    )
    return {
        "estimate": estimate,
        "timestamp": estimate_times,
        "raw_position_neu": _ned_to_neu(aligned_position_ned),
        "raw_velocity_neu": _ned_to_neu(aligned_velocity_ned),
        "estimated_body_to_ned": estimate_body_to_ned,
        "truth_position_neu": _ned_to_neu(truth_position_ned),
        "truth_velocity_neu": _ned_to_neu(truth_velocity_ned),
        "truth_body_to_ned": truth_body_to_ned,
        "translation_method": translation_method,
        "orientation_method": orientation_method,
        "startup_rotation_global_to_ned": rotation_global_to_ned,
    }


def _detection_from_csv(row: dict[str, str]) -> dict[str, Any]:
    return {
        "timestamp": float(row["timestamp"]),
        "frame_id": int(row["frame_id"]),
        "detection_index": int(row["detection_index"]),
        "confidence": float(row["confidence"]),
        "memory_confidence": float(row["memory_confidence"]),
        "yolo_confidence": float(row["yolo_confidence"]),
        "reprojection_error": float(row["reprojection_error"]),
        "quad_area_px2": float(row["quad_area_px2"]),
        "gate_center_camera": np.asarray(
            [row["camera_x"], row["camera_y"], row["camera_z"]], dtype=float
        ),
        "gate_center_camera_corrected": np.asarray(
            [
                row["camera_corrected_x"],
                row["camera_corrected_y"],
                row["camera_corrected_z"],
            ],
            dtype=float,
        ),
        "gate_center_body_frd": np.asarray(
            [row["body_x"], row["body_y"], row["body_z"]], dtype=float
        ),
        "pnp_solver": row["pnp_solver"],
        "pnp_order": row["pnp_order"],
    }


def _load_cached_detections(
    output_dir: Path,
    camera_rows: list[dict[str, Any]],
) -> Optional[dict[int, list[dict[str, Any]]]]:
    frame_path = output_dir / "openvins_gate_detection_frames.csv"
    detection_path = output_dir / "openvins_gate_detections_raw.csv"
    if not frame_path.is_file() or not detection_path.is_file():
        return None
    with frame_path.open(newline="", encoding="utf-8") as handle:
        frame_rows = list(csv.DictReader(handle))
    expected = [row["frame_id"] for row in camera_rows]
    cached = [int(row["frame_id"]) for row in frame_rows]
    if cached != expected:
        return None
    result = {frame_id: [] for frame_id in expected}
    with detection_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            detection = _detection_from_csv(row)
            result.setdefault(detection["frame_id"], []).append(detection)
    return result


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _extract_detections(
    replay: Path,
    output_dir: Path,
    camera_rows: list[dict[str, Any]],
    config: Any,
    *,
    force: bool,
    max_frames: Optional[int],
) -> tuple[dict[int, list[dict[str, Any]]], bool]:
    if not force and max_frames is None:
        cached = _load_cached_detections(output_dir, camera_rows)
        if cached is not None:
            print(
                f"Reusing cached PnP detections for {len(camera_rows)} frames",
                flush=True,
            )
            return cached, True

    try:
        import cv2
        from perception_wrapper import PerceptionWrapper
    except ImportError as exc:
        raise ExperimentError(
            "offline detection requires OpenCV and the controller environment"
        ) from exc

    selected_rows = camera_rows if max_frames is None else camera_rows[:max_frames]
    state_estimation = replace(config.state_estimation, mode="estimator")
    perception = replace(config.perception, world_pose_source="camera_only")
    detector_config = replace(
        config,
        state_estimation=state_estimation,
        perception=perception,
    )
    result: dict[int, list[dict[str, Any]]] = {}
    raw_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    perception_log_path = output_dir / "openvins_gate_perception.log"
    with perception_log_path.open("w", encoding="utf-8") as perception_log:
        with contextlib.redirect_stdout(perception_log):
            wrapper = PerceptionWrapper(config=detector_config)
        for index, camera_row in enumerate(selected_rows, start=1):
            image_path = replay / "cam0" / camera_row["filename"]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ExperimentError(f"unable to read camera image: {image_path}")
            frame = {
                "frame_id": camera_row["frame_id"],
                "sim_time_ns": camera_row["sim_time_ns"],
                "wall_time": camera_row["timestamp"],
                "image": image,
            }
            with contextlib.redirect_stdout(perception_log):
                latest = wrapper.update(frame=frame)
            detections = []
            for detection_index, source in enumerate(latest.get("detections", ())):
                detection = {
                    "timestamp": camera_row["timestamp"],
                    "frame_id": camera_row["frame_id"],
                    "detection_index": detection_index,
                    "confidence": float(source.get("confidence", 0.0)),
                    "memory_confidence": float(
                        source.get("memory_confidence", source.get("confidence", 0.0))
                    ),
                    "yolo_confidence": float(
                        source.get("yolo_confidence", source.get("confidence", 0.0))
                    ),
                    "reprojection_error": float(
                        source.get("reprojection_error", math.nan)
                    ),
                    "quad_area_px2": float(source.get("quad_area_px2", math.nan)),
                    "gate_center_camera": np.asarray(
                        source.get("gate_center_camera"), dtype=float
                    ).reshape(3),
                    "gate_center_camera_corrected": np.asarray(
                        source.get("gate_center_camera_corrected"), dtype=float
                    ).reshape(3),
                    "gate_center_body_frd": np.asarray(
                        source.get("gate_center_body_frd"), dtype=float
                    ).reshape(3),
                    "pnp_solver": str(source.get("pnp_selected_solver", "")),
                    "pnp_order": str(source.get("pnp_selected_order", "")),
                }
                detections.append(detection)
                camera = detection["gate_center_camera"]
                corrected = detection["gate_center_camera_corrected"]
                body = detection["gate_center_body_frd"]
                raw_rows.append(
                    {
                        "timestamp": f"{camera_row['timestamp']:.9f}",
                        "frame_id": camera_row["frame_id"],
                        "detection_index": detection_index,
                        "confidence": detection["confidence"],
                        "memory_confidence": detection["memory_confidence"],
                        "yolo_confidence": detection["yolo_confidence"],
                        "reprojection_error": detection["reprojection_error"],
                        "quad_area_px2": detection["quad_area_px2"],
                        "camera_x": camera[0],
                        "camera_y": camera[1],
                        "camera_z": camera[2],
                        "camera_corrected_x": corrected[0],
                        "camera_corrected_y": corrected[1],
                        "camera_corrected_z": corrected[2],
                        "body_x": body[0],
                        "body_y": body[1],
                        "body_z": body[2],
                        "pnp_solver": detection["pnp_solver"],
                        "pnp_order": detection["pnp_order"],
                    }
                )
            result[camera_row["frame_id"]] = detections
            frame_rows.append(
                {
                    "timestamp": f"{camera_row['timestamp']:.9f}",
                    "frame_id": camera_row["frame_id"],
                    "filename": camera_row["filename"],
                    "detection_count": len(detections),
                }
            )
            if index == 1 or index % 50 == 0 or index == len(selected_rows):
                print(
                    f"PnP detection {index}/{len(selected_rows)}; "
                    f"detections={len(raw_rows)}",
                    flush=True,
                )

    _write_csv(
        output_dir / "openvins_gate_detection_frames.csv",
        ("timestamp", "frame_id", "filename", "detection_count"),
        frame_rows,
    )
    _write_csv(
        output_dir / "openvins_gate_detections_raw.csv",
        RAW_DETECTION_FIELDS,
        raw_rows,
    )
    return result, False


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"cannot serialize {type(value).__name__}")


def evaluate(
    replay: Path,
    *,
    config_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    force_detections: bool = False,
    max_frames: Optional[int] = None,
) -> dict[str, Any]:
    replay = replay.resolve()
    if not replay.is_dir():
        raise ExperimentError(f"replay directory does not exist: {replay}")
    output_dir = (
        output_dir.resolve()
        if output_dir is not None
        else replay / "gate_alignment_experiment"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = _source_runtime_config(replay, config_path)
    config = load_runtime_config(resolved_config)
    known_gates = np.asarray(config.gate_source.known_gate_positions_neu, dtype=float)
    options = AlignmentOptions.from_config(config.experimental_gate_vio_alignment)
    trajectory = _load_openvins_trajectory(replay)
    camera_rows = _match_camera_rows(
        _read_camera_rows(replay), trajectory["timestamp"]
    )
    if max_frames is not None:
        if max_frames < 1:
            raise ExperimentError("--max-frames must be positive")
        original_count = int(trajectory["timestamp"].size)
        camera_rows = camera_rows[:max_frames]
        for key, value in list(trajectory.items()):
            if isinstance(value, np.ndarray) and value.shape[:1] == (
                original_count,
            ):
                trajectory[key] = value[:max_frames]

    detections_by_frame, reused_detections = _extract_detections(
        replay,
        output_dir,
        camera_rows,
        config,
        force=force_detections,
        max_frames=None,
    )
    aligner = OfflineGateMapAligner(known_gates, options)
    camera_translation = np.asarray(config.camera.body_translation_m, dtype=float)
    timeseries_rows = []
    detection_rows = []
    raw_errors = []
    aligned_errors = []
    fresh_errors = []
    velocity_errors = []
    offset_errors = []
    pnp_map_errors = []
    geometry_pnp_map_errors = []
    candidate_pnp_map_errors = []
    candidate_offset_errors = []
    association_correct = []
    accepted_gates: set[int] = set()
    truth_matched_gates: set[int] = set()
    reasons: Counter[str] = Counter()
    per_gate_audit = {
        index: {
            "all_pnp_errors": [],
            "geometry_pnp_errors": [],
            "candidate_pnp_errors": [],
            "candidate_offset_errors": [],
        }
        for index in range(known_gates.shape[0])
    }

    count = len(camera_rows)
    for index, camera_row in enumerate(camera_rows):
        timestamp = float(trajectory["timestamp"][index])
        raw_position = trajectory["raw_position_neu"][index]
        truth_position = trajectory["truth_position_neu"][index]
        truth_velocity = trajectory["truth_velocity_neu"][index]
        raw_velocity = trajectory["raw_velocity_neu"][index]
        estimate_rotation = trajectory["estimated_body_to_ned"][index]
        truth_rotation = trajectory["truth_body_to_ned"][index]
        detections = [
            dict(item)
            for item in detections_by_frame.get(camera_row["frame_id"], ())
        ]
        for detection in detections:
            gate_body = _vec3(
                detection["gate_center_body_frd"], "PnP body position"
            )
            levered_body = camera_translation + gate_body
            detection["relative_gate_neu"] = _ned_to_neu(
                estimate_rotation @ levered_body
            )
            truth_gate_neu = truth_position + _ned_to_neu(
                truth_rotation @ levered_body
            )
            truth_distances = np.linalg.norm(known_gates - truth_gate_neu, axis=1)
            truth_gate_index = int(np.argmin(truth_distances))
            truth_map_error = float(truth_distances[truth_gate_index])
            detection["truth_projected_gate_neu"] = truth_gate_neu
            detection["truth_gate_index"] = truth_gate_index
            detection["truth_map_error_m"] = truth_map_error
            pnp_map_errors.append(truth_map_error)
            per_gate_audit[truth_gate_index]["all_pnp_errors"].append(
                truth_map_error
            )
            if truth_map_error <= 6.0:
                truth_matched_gates.add(truth_gate_index)

        decision = aligner.update(timestamp, raw_position, detections)
        reasons[decision["reason"]] += 1
        if decision["accepted"] and decision["gate_index"] is not None:
            accepted_gates.add(int(decision["gate_index"]))
        aligned_position = decision["aligned_position_neu"]
        raw_error = float(np.linalg.norm(raw_position - truth_position))
        aligned_error = float(np.linalg.norm(aligned_position - truth_position))
        velocity_error = float(np.linalg.norm(raw_velocity - truth_velocity))
        required_offset = truth_position - raw_position
        offset_error = float(np.linalg.norm(decision["offset_neu"] - required_offset))
        raw_errors.append(raw_error)
        aligned_errors.append(aligned_error)
        velocity_errors.append(velocity_error)
        offset_errors.append(offset_error)
        if decision["fresh"]:
            fresh_errors.append(aligned_error)

        for detection in detections:
            selected_gate = detection.get("alignment_gate_index")
            truth_gate_index = int(detection["truth_gate_index"])
            if detection.get("alignment_candidate_ok") and detection["truth_map_error_m"] <= 6.0:
                association_correct.append(int(selected_gate) == truth_gate_index)
            candidate = detection.get("candidate_offset_neu")
            candidate_error = (
                None
                if candidate is None
                else float(np.linalg.norm(candidate - required_offset))
            )
            if detection.get("geometry_ok"):
                geometry_pnp_map_errors.append(detection["truth_map_error_m"])
                per_gate_audit[truth_gate_index]["geometry_pnp_errors"].append(
                    detection["truth_map_error_m"]
                )
            if detection.get("alignment_candidate_ok"):
                candidate_pnp_map_errors.append(detection["truth_map_error_m"])
                per_gate_audit[truth_gate_index]["candidate_pnp_errors"].append(
                    detection["truth_map_error_m"]
                )
                if candidate_error is not None:
                    candidate_offset_errors.append(candidate_error)
                    per_gate_audit[truth_gate_index][
                        "candidate_offset_errors"
                    ].append(candidate_error)
            detection_rows.append(
                {
                    "timestamp": f"{timestamp:.9f}",
                    "frame_id": camera_row["frame_id"],
                    "detection_index": detection["detection_index"],
                    "confidence": detection["confidence"],
                    "reprojection_error": detection["reprojection_error"],
                    "depth_m": detection["gate_center_camera"][2],
                    "geometry_ok": int(detection.get("geometry_ok", False)),
                    "candidate_ok": int(
                        detection.get("alignment_candidate_ok", False)
                    ),
                    "selected_gate_index": _csv_optional(selected_gate),
                    "association_residual_m": _csv_optional(
                        detection.get("alignment_association_residual_m")
                    ),
                    "truth_gate_index_audit_only": truth_gate_index,
                    "truth_pnp_map_error_m_audit_only": detection[
                        "truth_map_error_m"
                    ],
                    "association_correct_audit_only": (
                        ""
                        if selected_gate is None
                        else int(int(selected_gate) == truth_gate_index)
                    ),
                    "candidate_offset_error_m_audit_only": _csv_optional(
                        candidate_error
                    ),
                }
            )

        offset = decision["offset_neu"]
        timeseries_rows.append(
            {
                "timestamp": f"{timestamp:.9f}",
                "frame_id": camera_row["frame_id"],
                "detections": len(detections),
                "alignment_initialized": int(decision["initialized"]),
                "alignment_accepted": int(decision["accepted"]),
                "alignment_fresh": int(decision["fresh"]),
                "alignment_reason": decision["reason"],
                "alignment_gate_index": _csv_optional(decision["gate_index"]),
                "alignment_support": decision["support"],
                "offset_x": offset[0],
                "offset_y": offset[1],
                "offset_z": offset[2],
                "raw_x": raw_position[0],
                "raw_y": raw_position[1],
                "raw_z": raw_position[2],
                "aligned_x": aligned_position[0],
                "aligned_y": aligned_position[1],
                "aligned_z": aligned_position[2],
                "truth_x_audit_only": truth_position[0],
                "truth_y_audit_only": truth_position[1],
                "truth_z_audit_only": truth_position[2],
                "raw_error_m_audit_only": raw_error,
                "aligned_error_m_audit_only": aligned_error,
                "required_offset_x_audit_only": required_offset[0],
                "required_offset_y_audit_only": required_offset[1],
                "required_offset_z_audit_only": required_offset[2],
                "offset_error_m_audit_only": offset_error,
                "velocity_error_m_s_audit_only": velocity_error,
            }
        )
        if (index + 1) % 100 == 0 or index + 1 == count:
            print(f"Alignment scoring {index + 1}/{count}", flush=True)

    timeseries_path = output_dir / "openvins_gate_alignment_timeseries.csv"
    detections_path = output_dir / "openvins_gate_alignment_detections.csv"
    _write_csv(timeseries_path, tuple(timeseries_rows[0]), timeseries_rows)
    _write_csv(
        detections_path,
        (
            "timestamp",
            "frame_id",
            "detection_index",
            "confidence",
            "reprojection_error",
            "depth_m",
            "geometry_ok",
            "candidate_ok",
            "selected_gate_index",
            "association_residual_m",
            "truth_gate_index_audit_only",
            "truth_pnp_map_error_m_audit_only",
            "association_correct_audit_only",
            "candidate_offset_error_m_audit_only",
        ),
        detection_rows,
    )

    raw_summary = _summary(raw_errors)
    aligned_summary = _summary(aligned_errors)
    fresh_summary = _summary(fresh_errors)
    association_accuracy = (
        None
        if not association_correct
        else float(np.mean(np.asarray(association_correct, dtype=float)))
    )
    initialized_rows = [
        row for row in timeseries_rows if row["alignment_initialized"]
    ]
    accepted_rows = [row for row in timeseries_rows if row["alignment_accepted"]]
    per_gate_report = {}
    for gate_index, audit in per_gate_audit.items():
        per_gate_report[str(gate_index)] = {
            "position_neu_m": known_gates[gate_index].tolist(),
            "all_detections": len(audit["all_pnp_errors"]),
            "geometry_accepted": len(audit["geometry_pnp_errors"]),
            "alignment_candidates": len(audit["candidate_pnp_errors"]),
            "all_pnp_map_error_m": _summary(audit["all_pnp_errors"]),
            "geometry_pnp_map_error_m": _summary(
                audit["geometry_pnp_errors"]
            ),
            "candidate_pnp_map_error_m": _summary(
                audit["candidate_pnp_errors"]
            ),
            "candidate_offset_error_m_audit_only": _summary(
                audit["candidate_offset_errors"]
            ),
        }
    blockers = []
    if not initialized_rows:
        blockers.append("gate alignment never initialized")
    if len(accepted_gates) < 3:
        blockers.append(
            f"alignment updates covered only {len(accepted_gates)} configured gates"
        )
    if aligned_summary.get("p95", math.inf) > 0.5:
        blockers.append(
            "aligned p95 position error exceeds the 0.5 m control target"
        )
    if aligned_summary.get("max", math.inf) > 1.0:
        blockers.append("aligned maximum position error exceeds 1.0 m")
    if association_accuracy is not None and association_accuracy < 0.90:
        blockers.append("gate association audit accuracy is below 90%")

    report = {
        "format": "aigp_openvins_gate_alignment_experiment",
        "format_version": 1,
        "status": "pass" if not blockers else "fail",
        "usable_for_control": not blockers,
        "blockers": blockers,
        "replay": str(replay),
        "runtime_config": str(resolved_config),
        "outputs": {
            "report": str(output_dir / "openvins_gate_alignment_report.json"),
            "timeseries": str(timeseries_path),
            "detections": str(detections_path),
            "raw_detection_cache": str(
                output_dir / "openvins_gate_detections_raw.csv"
            ),
            "perception_log": str(
                output_dir / "openvins_gate_perception.log"
            ),
        },
        "truth_separation": {
            "truth_fed_to_detector": False,
            "truth_fed_to_aligner": False,
            "truth_used_for_startup_frame_alignment": True,
            "truth_used_for_scoring_and_pnp_audit_only": True,
            "note": (
                "The OpenVINS frame is rigidly aligned once at its first output "
                "using captured position and attitude, with no scale fit."
            ),
        },
        "frames": {
            "evaluated": count,
            "first_timestamp_s": float(trajectory["timestamp"][0]),
            "last_timestamp_s": float(trajectory["timestamp"][-1]),
            "exact_camera_timestamp_matches": count,
        },
        "detector": {
            "cache_reused": reused_detections,
            "detections": len(detection_rows),
            "frames_with_detections": len(
                {row["frame_id"] for row in detection_rows}
            ),
            "geometry_accepted": int(
                sum(row["geometry_ok"] for row in detection_rows)
            ),
            "configured_backend": config.perception.backend,
            "configured_model": config.perception.yolo_model_path,
            "configured_preprocess_mode": config.perception.preprocess_mode,
        },
        "alignment": {
            "model": "translation_only",
            "options": options.__dict__,
            "initialized": bool(initialized_rows),
            "initialization_timestamp_s": (
                None
                if not initialized_rows
                else float(initialized_rows[0]["timestamp"])
            ),
            "accepted_updates": len(accepted_rows),
            "accepted_gate_indices": sorted(accepted_gates),
            "truth_audit_gate_indices": sorted(truth_matched_gates),
            "final_offset_neu_m": aligner.offset_neu.tolist(),
            "reason_counts": dict(sorted(reasons.items())),
        },
        "position_error_m": {
            "raw_startup_aligned_openvins": raw_summary,
            "gate_aligned": aligned_summary,
            "gate_aligned_while_fresh": fresh_summary,
            "final_raw": raw_errors[-1],
            "final_gate_aligned": aligned_errors[-1],
            "p95_improvement_m": (
                raw_summary.get("p95", math.nan)
                - aligned_summary.get("p95", math.nan)
            ),
        },
        "velocity_error_m_s": _summary(velocity_errors),
        "alignment_offset_error_m_audit_only": _summary(offset_errors),
        "pnp_audit": {
            "all_truth_projected_center_to_nearest_map_gate_error_m": _summary(
                pnp_map_errors
            ),
            "geometry_accepted_map_error_m": _summary(
                geometry_pnp_map_errors
            ),
            "alignment_candidate_map_error_m": _summary(
                candidate_pnp_map_errors
            ),
            "candidate_offset_error_m": _summary(candidate_offset_errors),
            "association_samples": len(association_correct),
            "association_accuracy": association_accuracy,
            "per_truth_gate": per_gate_report,
        },
        "truth_sources": {
            "translation": trajectory["translation_method"],
            "orientation": trajectory["orientation_method"],
        },
    }
    report_path = output_dir / "openvins_gate_alignment_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_value) + "\n",
        encoding="utf-8",
    )
    return report


def _csv_optional(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        help="runtime TOML; defaults to the source capture's snapshot",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="defaults to REPLAY/gate_alignment_experiment",
    )
    parser.add_argument(
        "--force-detections",
        action="store_true",
        help="rerun YOLO/PnP instead of reusing the experiment cache",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="development-only prefix limit; omit for a valid full experiment",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate(
            args.replay,
            config_path=args.config,
            output_dir=args.output_dir,
            force_detections=args.force_detections,
            max_frames=args.max_frames,
        )
    except (ExperimentError, DiagnosticError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        "Gate alignment experiment: "
        f"{report['status']}; usable_for_control="
        f"{int(report['usable_for_control'])}"
    )
    print(f"Report: {report['outputs']['report']}")
    raw = report["position_error_m"]["raw_startup_aligned_openvins"]
    aligned = report["position_error_m"]["gate_aligned"]
    print(
        f"Position p95: raw={raw.get('p95', math.nan):.3f} m; "
        f"gate-aligned={aligned.get('p95', math.nan):.3f} m"
    )
    for blocker in report["blockers"]:
        print(f"BLOCKER: {blocker}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
