from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from autonomy_core.core.frame_conventions import (
    body_frd_to_local_ned_rotmat,
    local_ned_to_neu,
    perception_rpy_for_transform,
)


@dataclass(frozen=True)
class GateVioAlignmentResult:
    raw_pos_neu: np.ndarray
    aligned_pos_neu: np.ndarray
    offset_neu: np.ndarray
    enabled: bool
    initialized: bool
    accepted: bool
    reason: str
    support_count: int = 0
    landmark_source: str = ""
    association_residual_m: Optional[float] = None
    candidate_spread_m: Optional[float] = None
    last_update_age_s: Optional[float] = None


@dataclass(frozen=True)
class _OffsetObservation:
    wall_time: float
    frame_key: object
    offset_neu: np.ndarray
    association_residual_m: float


class ExperimentalGateVioAlignment:
    """Robust translation-only alignment between a local VIO frame and a gate map.

    Gate PnP observations update only ``offset_neu``. The caller's raw VIO pose and
    velocity are never mutated. Yaw and scale are deliberately outside this first
    experiment because a single gate-center observation cannot observe either one
    safely.
    """

    def __init__(self, config):
        self.config = config
        self.options = config.experimental_gate_vio_alignment
        self.known_gates = tuple(
            np.asarray(position, dtype=float).reshape(3).copy()
            for position in config.gate_source.known_gate_positions_neu
        )
        self.offset_neu = np.zeros(3, dtype=float)
        self.initialized = False
        self.last_frame_key = None
        self.last_accepted_wall_time: Optional[float] = None
        self.history_by_landmark: dict[str, deque[_OffsetObservation]] = {}
        self.last_trace_wall_time = 0.0
        self.last_trace_signature = None

    def reset(self) -> None:
        self.offset_neu = np.zeros(3, dtype=float)
        self.initialized = False
        self.last_frame_key = None
        self.last_accepted_wall_time = None
        self.history_by_landmark.clear()
        self.last_trace_wall_time = 0.0
        self.last_trace_signature = None

    def update(
        self,
        raw_pos_neu,
        snapshot,
        *,
        now: Optional[float] = None,
    ) -> GateVioAlignmentResult:
        raw_pos = self._vec3(raw_pos_neu)
        if raw_pos is None:
            raise ValueError("raw_pos_neu must be a finite three-vector")
        wall_now = time.time() if now is None else float(now)

        if not bool(self.options.enabled):
            return self._result(raw_pos, False, "disabled", wall_now)
        if not self.known_gates:
            return self._traced_result(raw_pos, False, "no_known_gate_map", wall_now)

        latest_perception = getattr(snapshot, "latest_perception", None)
        if not isinstance(latest_perception, dict):
            return self._traced_result(raw_pos, False, "no_perception", wall_now)

        frame_key = self._frame_key(latest_perception)
        if frame_key is None:
            frame_key = ("object", id(latest_perception))
        if frame_key == self.last_frame_key:
            return self._result(raw_pos, False, "duplicate_frame", wall_now)
        self.last_frame_key = frame_key

        detections = latest_perception.get("detections")
        if not detections:
            self._prune_history(wall_now)
            return self._traced_result(raw_pos, False, "no_detections", wall_now)

        rpy_used = perception_rpy_for_transform(
            np.array(
                [
                    float(getattr(snapshot, "roll_rad", 0.0)),
                    float(getattr(snapshot, "pitch_rad", 0.0)),
                    float(getattr(snapshot, "yaw_rad", 0.0)),
                ],
                dtype=float,
            ),
            transform_mode=str(latest_perception.get("transform_mode", "")),
            yaw_correction_rad=self._finite_float(
                latest_perception.get("perception_yaw_correction_rad"),
                default=0.0,
            ),
        )
        rot_ned_body = body_frd_to_local_ned_rotmat(*rpy_used)
        camera_translation_body = self._vec3(
            latest_perception.get("camera_translation_body")
        )
        if camera_translation_body is None:
            camera_translation_body = np.zeros(3, dtype=float)

        association_radius = float(
            self.options.association_radius_m
            if self.initialized
            else self.options.initial_association_radius_m
        )
        aligned_prediction = raw_pos + (
            self.offset_neu if self.initialized else np.zeros(3, dtype=float)
        )
        observations: dict[str, _OffsetObservation] = {}

        for detection in detections:
            if not isinstance(detection, dict) or not self._geometry_ok(detection):
                continue
            gate_body = self._vec3(
                detection.get("gate_center_body_frd", detection.get("gate_center_body"))
            )
            if gate_body is None:
                continue

            rel_neu = local_ned_to_neu(
                rot_ned_body @ (camera_translation_body + gate_body)
            )
            predicted_gate_neu = aligned_prediction + rel_neu
            landmark = self._nearest_gate(predicted_gate_neu, association_radius)
            if landmark is None:
                continue
            landmark_position, landmark_source, association_residual = landmark
            candidate_offset = landmark_position - raw_pos - rel_neu

            if (
                not self.initialized
                and np.linalg.norm(candidate_offset) > self.options.max_initial_offset_m
            ):
                continue
            if (
                self.initialized
                and np.linalg.norm(candidate_offset - self.offset_neu)
                > self.options.max_update_innovation_m
            ):
                continue

            observation = _OffsetObservation(
                wall_time=wall_now,
                frame_key=frame_key,
                offset_neu=candidate_offset,
                association_residual_m=association_residual,
            )
            previous = observations.get(landmark_source)
            if (
                previous is None
                or observation.association_residual_m
                < previous.association_residual_m
            ):
                observations[landmark_source] = observation

        if not observations:
            self._prune_history(wall_now)
            return self._traced_result(
                raw_pos,
                False,
                "no_gated_measurements",
                wall_now,
            )

        for source, observation in observations.items():
            history = self.history_by_landmark.setdefault(source, deque())
            history.append(observation)
        self._prune_history(wall_now)

        candidates = []
        for source, current_observation in observations.items():
            consistent = self._consistent_candidate(source, current_observation)
            if consistent is not None:
                target_offset, support_count, spread_m = consistent
                candidates.append(
                    (
                        -support_count,
                        spread_m,
                        current_observation.association_residual_m,
                        source,
                        target_offset,
                        support_count,
                    )
                )

        if not candidates:
            max_support = max(
                (len(self.history_by_landmark[source]) for source in observations),
                default=0,
            )
            return self._traced_result(
                raw_pos,
                False,
                "awaiting_temporal_consistency",
                wall_now,
                support_count=max_support,
            )

        (
            _,
            spread_m,
            association_residual,
            source,
            target_offset,
            support_count,
        ) = min(candidates, key=lambda item: item[:3])

        if not self.initialized:
            self.offset_neu = target_offset.copy()
            self.initialized = True
            reason = "initialized"
        else:
            innovation = target_offset - self.offset_neu
            step = float(self.options.correction_alpha) * innovation
            step_norm = float(np.linalg.norm(step))
            max_step = float(self.options.max_step_m)
            if step_norm > max_step:
                step *= max_step / max(step_norm, 1e-12)
            self.offset_neu = self.offset_neu + step
            reason = "updated"

        self.last_accepted_wall_time = wall_now
        return self._traced_result(
            raw_pos,
            True,
            reason,
            wall_now,
            support_count=support_count,
            landmark_source=source,
            association_residual_m=association_residual,
            candidate_spread_m=spread_m,
        )

    def _consistent_candidate(
        self,
        source: str,
        current: _OffsetObservation,
    ) -> Optional[tuple[np.ndarray, int, float]]:
        history = self.history_by_landmark.get(source)
        if not history:
            return None
        offsets = np.asarray([item.offset_neu for item in history], dtype=float)
        median = np.median(offsets, axis=0)
        deviations = np.linalg.norm(offsets - median, axis=1)
        inliers = deviations <= float(self.options.consistency_radius_m)
        support_count = int(np.count_nonzero(inliers))
        if support_count < int(self.options.min_consistent_frames):
            return None
        if float(np.linalg.norm(current.offset_neu - median)) > float(
            self.options.consistency_radius_m
        ):
            return None
        inlier_offsets = offsets[inliers]
        target = np.median(inlier_offsets, axis=0)
        spread_m = float(np.max(np.linalg.norm(inlier_offsets - target, axis=1)))
        return target, support_count, spread_m

    def _prune_history(self, now: float) -> None:
        cutoff = float(now) - float(self.options.temporal_window_s)
        empty_sources = []
        for source, history in self.history_by_landmark.items():
            while history and history[0].wall_time < cutoff:
                history.popleft()
            if not history:
                empty_sources.append(source)
        for source in empty_sources:
            self.history_by_landmark.pop(source, None)

    def _nearest_gate(
        self,
        predicted_gate_neu: np.ndarray,
        max_distance_m: float,
    ) -> Optional[tuple[np.ndarray, str, float]]:
        distances = [
            float(np.linalg.norm(predicted_gate_neu - position))
            for position in self.known_gates
        ]
        if not distances:
            return None
        index = int(np.argmin(distances))
        residual = distances[index]
        if max_distance_m > 0.0 and residual > max_distance_m:
            return None
        return self.known_gates[index], f"known_gate:{index}", residual

    def _geometry_ok(self, detection: dict) -> bool:
        confidence = self._confidence(detection)
        if confidence < float(self.options.min_confidence):
            return False

        reprojection_error = self._finite_float(detection.get("reprojection_error"))
        if (
            reprojection_error is not None
            and float(self.options.max_reprojection_error) > 0.0
            and reprojection_error > float(self.options.max_reprojection_error)
        ):
            return False

        center_camera = self._vec3(detection.get("gate_center_camera"))
        if center_camera is None:
            return True
        depth_m = float(center_camera[2])
        if depth_m <= 0.0:
            return False
        if depth_m < float(self.options.min_depth_m):
            return False
        if (
            float(self.options.max_depth_m) > 0.0
            and depth_m > float(self.options.max_depth_m)
        ):
            return False
        return True

    def _result(
        self,
        raw_pos: np.ndarray,
        accepted: bool,
        reason: str,
        now: float,
        *,
        support_count: int = 0,
        landmark_source: str = "",
        association_residual_m: Optional[float] = None,
        candidate_spread_m: Optional[float] = None,
    ) -> GateVioAlignmentResult:
        age_s = (
            None
            if self.last_accepted_wall_time is None
            else max(0.0, float(now) - self.last_accepted_wall_time)
        )
        if (
            self.initialized
            and not accepted
            and age_s is not None
            and age_s > float(self.options.stale_after_s)
        ):
            reason = f"stale:{reason}"
        aligned_pos = raw_pos + self.offset_neu if self.initialized else raw_pos.copy()
        return GateVioAlignmentResult(
            raw_pos_neu=raw_pos.copy(),
            aligned_pos_neu=aligned_pos.copy(),
            offset_neu=self.offset_neu.copy(),
            enabled=bool(self.options.enabled),
            initialized=bool(self.initialized),
            accepted=bool(accepted),
            reason=str(reason),
            support_count=int(support_count),
            landmark_source=str(landmark_source),
            association_residual_m=association_residual_m,
            candidate_spread_m=candidate_spread_m,
            last_update_age_s=age_s,
        )

    def _traced_result(self, *args, **kwargs) -> GateVioAlignmentResult:
        result = self._result(*args, **kwargs)
        self._trace(result)
        return result

    def _trace(self, result: GateVioAlignmentResult) -> None:
        if not bool(self.options.trace):
            return
        now = time.time()
        signature = (
            result.initialized,
            result.accepted,
            result.reason,
            result.landmark_source,
        )
        if (
            signature == self.last_trace_signature
            and now - self.last_trace_wall_time < float(self.options.trace_period_s)
        ):
            return
        self.last_trace_signature = signature
        self.last_trace_wall_time = now
        residual = (
            "nan"
            if result.association_residual_m is None
            else f"{result.association_residual_m:.2f}"
        )
        spread = (
            "nan"
            if result.candidate_spread_m is None
            else f"{result.candidate_spread_m:.2f}"
        )
        print(
            "experimental_gate_vio_alignment "
            f"initialized={int(result.initialized)} "
            f"accepted={int(result.accepted)} "
            f"reason={result.reason} "
            f"gate={result.landmark_source or 'none'} "
            f"support={result.support_count} "
            f"association_residual_m={residual} "
            f"spread_m={spread} "
            f"offset_neu=({result.offset_neu[0]:.2f},"
            f"{result.offset_neu[1]:.2f},{result.offset_neu[2]:.2f})",
            flush=True,
        )

    @staticmethod
    def _frame_key(latest_perception: dict) -> Optional[object]:
        frame_id = latest_perception.get("frame_id")
        if frame_id is not None:
            try:
                frame_id = int(frame_id)
            except (TypeError, ValueError):
                frame_id = -1
            if frame_id >= 0:
                return ("frame", frame_id)
        for key in ("image_sim_time_ns", "image_wall_time", "perception_wall_time"):
            value = latest_perception.get(key)
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric) and numeric > 0.0:
                return (key, numeric)
        return None

    @classmethod
    def _confidence(cls, detection: dict) -> float:
        for key in ("memory_confidence", "confidence", "yolo_confidence"):
            value = cls._finite_float(detection.get(key))
            if value is not None:
                return max(0.0, min(value, 1.0))
        return 0.0

    @staticmethod
    def _vec3(value) -> Optional[np.ndarray]:
        if value is None:
            return None
        try:
            out = np.asarray(value, dtype=float).reshape(3)
        except (TypeError, ValueError):
            return None
        return out.copy() if np.all(np.isfinite(out)) else None

    @staticmethod
    def _finite_float(value, *, default: Optional[float] = None) -> Optional[float]:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return default
        return out if math.isfinite(out) else default
