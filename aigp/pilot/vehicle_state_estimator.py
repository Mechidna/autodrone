from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from autonomy_core.core.frame_conventions import (
    body_frd_to_local_ned_rotmat,
    local_ned_to_neu,
    local_neu_to_ned,
)


@dataclass(frozen=True)
class VehicleStateEstimate:
    pos_neu: np.ndarray
    vel_neu: np.ndarray
    yaw_rad: float
    source: str
    valid: bool
    confidence: float
    wall_time: float
    truth_pos_neu: Optional[np.ndarray] = None
    truth_vel_neu: Optional[np.ndarray] = None
    truth_error_m: Optional[float] = None
    vision_correction_source: str = ""
    vision_correction_residual_m: Optional[float] = None
    vision_correction_count: int = 0
    reason: str = ""


class VehicleStateEstimator:
    """
    Small state-provider layer for the pilot autonomy wrapper.

    Modes:
    - mavlink: require fresh MAVLink local position/odometry.
    - auto: use MAVLink state when fresh, otherwise fall back to estimator.
    - estimator: ignore MAVLink state for control and keep it only as truth.
    """

    def __init__(self, config):
        self.config = config
        self.mode = str(config.state_estimation.mode).lower()
        self.initialized = False
        self.pos_neu = np.zeros(3, dtype=float)
        self.vel_neu = np.zeros(3, dtype=float)
        self.last_wall_time: Optional[float] = None
        self.last_source = "uninitialized"
        self.last_vision_correction = {
            "source": "",
            "residual_m": None,
            "count": 0,
        }

    def update(self, snapshot) -> VehicleStateEstimate:
        truth = self.mavlink_state_from_snapshot(snapshot)

        if self.mode in ("mavlink", "auto") and truth is not None:
            self._anchor_to_truth(truth)
            return truth

        if self.mode == "mavlink":
            return self._invalid("missing_fresh_mavlink_state", truth)

        return self._estimator_update(snapshot, truth)

    def mavlink_state_from_snapshot(self, snapshot) -> Optional[VehicleStateEstimate]:
        pos = self._vec3(getattr(snapshot, "pos_neu", None))
        vel = self._vec3(getattr(snapshot, "vel_neu", None))
        if pos is None or vel is None:
            return None

        wall_time = self._finite_float(getattr(snapshot, "position_wall_time", None))
        if wall_time is None:
            wall_time = time.time()

        max_age_s = float(self.config.state_estimation.max_state_age_s)
        if max_age_s > 0.0 and time.time() - wall_time > max_age_s:
            return None

        source = str(getattr(snapshot, "position_source", None) or "mavlink")
        return VehicleStateEstimate(
            pos_neu=pos.copy(),
            vel_neu=vel.copy(),
            yaw_rad=float(getattr(snapshot, "yaw_rad", 0.0)),
            source=source,
            valid=True,
            confidence=1.0,
            wall_time=wall_time,
            truth_pos_neu=pos.copy()
            if self.config.state_estimation.mavlink_truth_logging
            else None,
            truth_vel_neu=vel.copy()
            if self.config.state_estimation.mavlink_truth_logging
            else None,
            truth_error_m=0.0
            if self.config.state_estimation.mavlink_truth_logging
            else None,
            reason="fresh_mavlink_state",
        )

    def project_perception_with_estimated_state(
        self,
        latest_perception,
        estimate: VehicleStateEstimate,
        snapshot,
    ):
        if not isinstance(latest_perception, dict) or not estimate.valid:
            return latest_perception

        if not bool(self.config.state_estimation.use_vision_correction):
            return latest_perception

        detections = latest_perception.get("detections")
        if not detections:
            return latest_perception

        camera_translation_body = self._vec3(
            latest_perception.get("camera_translation_body")
        )
        if camera_translation_body is None:
            camera_translation_body = np.zeros(3, dtype=float)

        pos_ned = local_neu_to_ned(estimate.pos_neu)
        rot_ned_body = body_frd_to_local_ned_rotmat(
            float(getattr(snapshot, "roll_rad", 0.0)),
            float(getattr(snapshot, "pitch_rad", 0.0)),
            float(getattr(snapshot, "yaw_rad", estimate.yaw_rad)),
        )

        projected = dict(latest_perception)
        projected["world_frame"] = "estimator_local_neu"
        projected["world_pose_source"] = estimate.source
        projected["detections"] = []

        for detection in detections:
            if not isinstance(detection, dict):
                continue
            out = copy.copy(detection)
            gate_body = self._vec3(
                out.get("gate_center_body_frd", out.get("gate_center_body"))
            )
            if gate_body is None:
                projected["detections"].append(out)
                continue

            gate_world_ned = pos_ned + rot_ned_body @ (
                camera_translation_body + gate_body
            )
            gate_world_neu = local_ned_to_neu(gate_world_ned)
            out["gate_center_world_ned"] = gate_world_ned.copy()
            out["gate_center_world"] = gate_world_neu.copy()
            out["drone_pos_ned"] = pos_ned.copy()
            out["drone_pos_neu"] = estimate.pos_neu.copy()
            out["drone_state_source"] = estimate.source
            out["body_to_world_method_used"] = (
                f"vehicle_state_estimator:{estimate.source}"
            )

            gate_normal_body = self._vec3(
                out.get("gate_normal_body_frd", out.get("gate_normal_body"))
            )
            if gate_normal_body is not None:
                normal_world_ned = rot_ned_body @ gate_normal_body
                out["gate_normal_world_ned"] = normal_world_ned.copy()
                out["gate_normal_world"] = local_ned_to_neu(normal_world_ned)

            projected["detections"].append(out)

        return projected

    def _estimator_update(self, snapshot, truth) -> VehicleStateEstimate:
        now = self._snapshot_wall_time(snapshot)
        yaw_rad = float(getattr(snapshot, "yaw_rad", 0.0))

        if not self.initialized:
            self._initialize(snapshot, truth, now)

        if bool(self.config.state_estimation.use_imu_prediction):
            self._predict_with_imu(snapshot, now)
        else:
            self.last_wall_time = now

        correction = self._correct_with_vision(snapshot)

        truth_pos = truth.pos_neu.copy() if truth is not None else None
        truth_vel = truth.vel_neu.copy() if truth is not None else None
        truth_error = (
            float(np.linalg.norm(self.pos_neu - truth.pos_neu))
            if truth is not None
            else None
        )
        confidence = 0.5 if truth is None else max(0.2, 1.0 / (1.0 + truth_error))

        return VehicleStateEstimate(
            pos_neu=self.pos_neu.copy(),
            vel_neu=self.vel_neu.copy(),
            yaw_rad=yaw_rad,
            source="estimator",
            valid=True,
            confidence=float(confidence),
            wall_time=now,
            truth_pos_neu=truth_pos,
            truth_vel_neu=truth_vel,
            truth_error_m=truth_error,
            vision_correction_source=str(correction["source"]),
            vision_correction_residual_m=correction["residual_m"],
            vision_correction_count=int(correction["count"]),
            reason="imu_vision_estimator",
        )

    def _correct_with_vision(self, snapshot) -> dict[str, object]:
        self.last_vision_correction = {
            "source": "",
            "residual_m": None,
            "count": 0,
        }

        if not bool(self.config.state_estimation.use_vision_correction):
            return self.last_vision_correction

        source_mode = str(self.config.state_estimation.vision_correction_source).lower()
        if source_mode == "none":
            return self.last_vision_correction

        latest_perception = getattr(snapshot, "latest_perception", None)
        if not isinstance(latest_perception, dict):
            return self.last_vision_correction

        detections = latest_perception.get("detections")
        if not detections:
            return self.last_vision_correction

        landmarks = self._vision_landmarks(snapshot, source_mode)
        if not landmarks:
            return self.last_vision_correction

        rot_ned_body = body_frd_to_local_ned_rotmat(
            float(getattr(snapshot, "roll_rad", 0.0)),
            float(getattr(snapshot, "pitch_rad", 0.0)),
            float(getattr(snapshot, "yaw_rad", 0.0)),
        )
        camera_translation_body = self._vec3(
            latest_perception.get("camera_translation_body")
        )
        if camera_translation_body is None:
            camera_translation_body = np.zeros(3, dtype=float)

        max_residual = float(self.config.state_estimation.vision_correction_max_residual_m)
        min_confidence = float(self.config.state_estimation.vision_correction_min_confidence)
        measurements = []
        weights = []
        residuals = []
        sources = []

        for detection in detections:
            if not isinstance(detection, dict):
                continue
            confidence = self._confidence(detection)
            if confidence < min_confidence:
                continue

            if not self._detection_geometry_ok(detection):
                continue

            gate_body = self._vec3(
                detection.get("gate_center_body_frd", detection.get("gate_center_body"))
            )
            if gate_body is None:
                continue

            rel_neu = local_ned_to_neu(rot_ned_body @ (camera_translation_body + gate_body))
            predicted_gate = self.pos_neu + rel_neu
            landmark = self._nearest_landmark(predicted_gate, landmarks, max_residual)
            if landmark is None:
                continue

            landmark_pos, landmark_source = landmark
            measured_pos = landmark_pos - rel_neu
            residual = float(np.linalg.norm(measured_pos - self.pos_neu))
            if max_residual > 0.0 and residual > max_residual:
                continue

            measurements.append(measured_pos)
            weights.append(max(confidence, 1e-3))
            residuals.append(residual)
            sources.append(landmark_source)

        if not measurements:
            return self.last_vision_correction

        measured_pos = np.average(
            np.asarray(measurements, dtype=float),
            axis=0,
            weights=np.asarray(weights, dtype=float),
        )
        alpha = float(np.clip(self.config.state_estimation.vision_correction_alpha, 0.0, 1.0))
        alpha_xy = float(
            np.clip(
                getattr(
                    self.config.state_estimation,
                    "vision_correction_alpha_xy",
                    alpha,
                ),
                0.0,
                1.0,
            )
        )
        alpha_z = float(
            np.clip(
                getattr(
                    self.config.state_estimation,
                    "vision_correction_alpha_z",
                    0.0,
                ),
                0.0,
                1.0,
            )
        )
        delta = measured_pos - self.pos_neu
        max_delta = float(
            max(
                0.0,
                getattr(
                    self.config.state_estimation,
                    "vision_correction_max_delta_m",
                    0.0,
                ),
            )
        )
        if max_delta > 0.0:
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > max_delta:
                delta = delta * (max_delta / max(delta_norm, 1e-9))
        self.pos_neu[:2] = self.pos_neu[:2] + alpha_xy * delta[:2]
        self.pos_neu[2] = self.pos_neu[2] + alpha_z * delta[2]

        correction = {
            "source": "+".join(sorted(set(sources))),
            "residual_m": float(np.average(residuals, weights=weights)),
            "count": int(len(measurements)),
        }
        self.last_vision_correction = correction
        return correction

    def _vision_landmarks(self, snapshot, source_mode: str) -> list[tuple[np.ndarray, str]]:
        landmarks: list[tuple[np.ndarray, str]] = []

        if source_mode in ("stable_tracks", "known_gates_or_stable_tracks"):
            stable_landmarks = (
                getattr(snapshot, "stable_gate_landmarks_neu", None) or []
            )
            for idx, item in enumerate(stable_landmarks):
                label = f"stable_track:{idx}"
                position = item
                if isinstance(item, dict):
                    position = item.get("position_neu")
                    landmark_id = item.get("track_id", item.get("landmark_id", idx))
                    label = f"stable_track:{landmark_id}"
                arr = self._vec3(position)
                if arr is not None:
                    landmarks.append((arr, label))

        if source_mode == "known_gates_or_stable_tracks" and landmarks:
            return landmarks

        allow_known_gate_correction = bool(
            getattr(
                self.config.state_estimation,
                "allow_known_gate_correction",
                False,
            )
        )
        if (
            source_mode in ("known_gates", "known_gates_or_stable_tracks")
            and allow_known_gate_correction
        ):
            known_positions = self.config.state_estimation.known_gate_positions_neu
            for idx, position in enumerate(known_positions):
                arr = self._vec3(position)
                if arr is not None:
                    landmarks.append((arr, f"known_gate:{idx}"))

            for gate in getattr(snapshot, "track_gates", None) or []:
                if not isinstance(gate, dict):
                    continue
                arr = self._vec3(gate.get("position_neu"))
                if arr is None and gate.get("position_ned") is not None:
                    ned = self._vec3(gate.get("position_ned"))
                    if ned is not None:
                        arr = local_ned_to_neu(ned)
                if arr is not None:
                    gate_id = gate.get("gate_id", len(landmarks))
                    landmarks.append((arr, f"track_gate:{gate_id}"))

        return landmarks

    def _detection_geometry_ok(self, detection: dict) -> bool:
        reprojection_error = self._finite_float(detection.get("reprojection_error"))
        if (
            reprojection_error is not None
            and reprojection_error
            > float(self.config.perception.max_reprojection_error_for_memory)
        ):
            return False

        center_camera = self._vec3(detection.get("gate_center_camera"))
        if center_camera is None:
            return True

        depth_m = float(center_camera[2])
        if bool(self.config.perception.reject_negative_depth) and depth_m <= 0.0:
            return False

        min_depth = float(
            getattr(self.config.perception, "min_depth_m_for_memory", 0.0)
        )
        max_depth = float(
            getattr(self.config.perception, "max_depth_m_for_memory", 0.0)
        )
        if min_depth > 0.0 and depth_m < min_depth:
            return False
        if max_depth > 0.0 and depth_m > max_depth:
            return False
        return True

    @staticmethod
    def _nearest_landmark(
        predicted_gate: np.ndarray,
        landmarks: list[tuple[np.ndarray, str]],
        max_residual: float,
    ) -> Optional[tuple[np.ndarray, str]]:
        best = None
        best_dist = float("inf")
        for landmark_pos, source in landmarks:
            dist = float(np.linalg.norm(predicted_gate - landmark_pos))
            if dist < best_dist:
                best = (landmark_pos, source)
                best_dist = dist
        if best is None:
            return None
        if max_residual > 0.0 and best_dist > max_residual:
            return None
        return best

    @classmethod
    def _confidence(cls, detection: dict) -> float:
        for key in ("memory_confidence", "confidence", "yolo_confidence"):
            value = cls._finite_float(detection.get(key))
            if value is not None:
                return max(0.0, min(float(value), 1.0))
        return 0.0

    def _initialize(self, snapshot, truth, now: float) -> None:
        source = str(self.config.state_estimation.init_position_source).lower()
        if source == "mavlink_once" and truth is not None:
            self.pos_neu = truth.pos_neu.copy()
            self.vel_neu = truth.vel_neu.copy()
            self.last_source = "estimator_init_mavlink_once"
        else:
            self.pos_neu = np.zeros(3, dtype=float)
            self.vel_neu = np.zeros(3, dtype=float)
            self.last_source = "estimator_init_zero"

        self.last_wall_time = now
        self.initialized = True

    def _anchor_to_truth(self, truth: VehicleStateEstimate) -> None:
        self.pos_neu = truth.pos_neu.copy()
        self.vel_neu = truth.vel_neu.copy()
        self.last_wall_time = truth.wall_time
        self.initialized = True
        self.last_source = truth.source

    def _predict_with_imu(self, snapshot, now: float) -> None:
        if self.last_wall_time is None:
            self.last_wall_time = now
            return

        dt = max(0.0, now - float(self.last_wall_time))
        dt = min(dt, float(self.config.state_estimation.max_imu_dt_s))
        self.last_wall_time = now
        if dt <= 0.0:
            return

        acc_body = self._vec3(getattr(snapshot, "accel_xyz", None))
        if acc_body is None:
            return

        rot_ned_body = body_frd_to_local_ned_rotmat(
            float(getattr(snapshot, "roll_rad", 0.0)),
            float(getattr(snapshot, "pitch_rad", 0.0)),
            float(getattr(snapshot, "yaw_rad", 0.0)),
        )
        acc_ned = rot_ned_body @ acc_body
        acc_ned = acc_ned + np.array(
            [0.0, 0.0, float(self.config.state_estimation.gravity_m_s2)],
            dtype=float,
        )
        acc_neu = local_ned_to_neu(acc_ned)

        max_acc = float(self.config.state_estimation.max_imu_accel_m_s2)
        acc_norm = float(np.linalg.norm(acc_neu))
        if max_acc > 0.0 and acc_norm > max_acc:
            acc_neu = acc_neu * (max_acc / max(acc_norm, 1e-9))

        self.pos_neu = self.pos_neu + self.vel_neu * dt + 0.5 * acc_neu * dt * dt
        self.vel_neu = self.vel_neu + acc_neu * dt

        max_vel = float(self.config.state_estimation.max_imu_velocity_m_s)
        vel_norm = float(np.linalg.norm(self.vel_neu))
        if max_vel > 0.0 and vel_norm > max_vel:
            self.vel_neu = self.vel_neu * (max_vel / max(vel_norm, 1e-9))

    def _invalid(self, reason: str, truth=None) -> VehicleStateEstimate:
        truth_pos = truth.pos_neu.copy() if truth is not None else None
        truth_vel = truth.vel_neu.copy() if truth is not None else None
        return VehicleStateEstimate(
            pos_neu=np.full(3, np.nan, dtype=float),
            vel_neu=np.full(3, np.nan, dtype=float),
            yaw_rad=0.0,
            source=self.mode,
            valid=False,
            confidence=0.0,
            wall_time=time.time(),
            truth_pos_neu=truth_pos,
            truth_vel_neu=truth_vel,
            truth_error_m=None,
            reason=reason,
        )

    @staticmethod
    def _snapshot_wall_time(snapshot) -> float:
        for attr in ("imu_wall_time", "attitude_wall_time", "image_wall_time"):
            value = VehicleStateEstimator._finite_float(getattr(snapshot, attr, None))
            if value is not None and value > 0.0:
                return value
        return time.time()

    @staticmethod
    def _vec3(value) -> Optional[np.ndarray]:
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=float).reshape(3)
        except (TypeError, ValueError):
            return None
        if np.all(np.isfinite(arr)):
            return arr.copy()
        return None

    @staticmethod
    def _finite_float(value) -> Optional[float]:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if math.isfinite(out):
            return out
        return None
