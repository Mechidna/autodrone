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
from feature_visual_odometry import FeatureVisualOdometry
from visual_odometry import GateKeypointVisualOdometry, VisualOdometryMeasurement


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
    imu_dt_s: Optional[float] = None
    raw_accel_body: Optional[np.ndarray] = None
    computed_acc_neu: Optional[np.ndarray] = None
    position_error_neu: Optional[np.ndarray] = None
    velocity_error_neu: Optional[np.ndarray] = None
    visual_velocity_neu: Optional[np.ndarray] = None
    visual_velocity_dt_s: Optional[float] = None
    visual_velocity_reason: str = ""
    visual_reference_reset: bool = False
    accel_bias_neu: Optional[np.ndarray] = None


class VehicleStateEstimator:
    """
    Small state-provider layer for the pilot autonomy wrapper.

    Modes:
    - mavlink: require fresh MAVLink local position/odometry.
    - auto: use MAVLink state when fresh, otherwise fall back to estimator.
    - estimator: ignore MAVLink state for control and keep it only as truth.
    """

    def __init__(self, config, *, mode_override: Optional[str] = None):
        self.config = config
        self.mode = str(
            config.state_estimation.mode if mode_override is None else mode_override
        ).lower()
        self.initialized = False
        self.pos_neu = np.zeros(3, dtype=float)
        self.vel_neu = np.zeros(3, dtype=float)
        self.accel_bias_neu = np.zeros(3, dtype=float)
        self.last_wall_time: Optional[float] = None
        self.last_visual_update_time: Optional[float] = None
        self.last_visual_measured_pos_neu: Optional[np.ndarray] = None
        self.last_visual_debug_print_time = 0.0
        self.last_visual_debug_signature: Optional[tuple[object, ...]] = None
        self.last_temporal_vo_trace_time = 0.0
        self.last_temporal_vo_trace_signature: Optional[tuple[object, ...]] = None
        self.last_feature_vo_trace_time = 0.0
        self.last_feature_vo_trace_signature: Optional[tuple[object, ...]] = None
        self.last_source = "uninitialized"
        self.visual_odometry = GateKeypointVisualOdometry(config)
        self.feature_visual_odometry = FeatureVisualOdometry(config)
        self.last_vision_correction = {
            "source": "",
            "residual_m": None,
            "count": 0,
            "visual_velocity_neu": None,
            "visual_dt_s": None,
            "visual_velocity_reason": "",
            "visual_reference_reset": False,
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

        if self._perception_world_pose_source(latest_perception) == "gazebo_camera_sim":
            projected = dict(latest_perception)
            projected["world_frame"] = "gazebo_camera_sim_neu"
            projected["world_pose_source"] = "gazebo_camera_sim"
            return projected

        camera_translation_body = self._vec3(
            latest_perception.get("camera_translation_body")
        )
        if camera_translation_body is None:
            camera_translation_body = np.zeros(3, dtype=float)

        pos_ned = local_neu_to_ned(estimate.pos_neu)
        rpy_used = np.array(
            [
                float(getattr(snapshot, "roll_rad", 0.0)),
                float(getattr(snapshot, "pitch_rad", 0.0)),
                float(getattr(snapshot, "yaw_rad", estimate.yaw_rad)),
            ],
            dtype=float,
        )
        rpy_used[2] += self._perception_yaw_correction_rad(latest_perception)
        rot_ned_body = body_frd_to_local_ned_rotmat(*rpy_used)

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
            out["drone_rpy_rad_used"] = rpy_used.copy()
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

        imu_debug = {}
        if bool(self.config.state_estimation.use_imu_prediction):
            imu_debug = self._predict_with_imu(snapshot, now)
        else:
            self.last_wall_time = now

        correction = self._correct_with_vision(snapshot, now)

        truth_pos = truth.pos_neu.copy() if truth is not None else None
        truth_vel = truth.vel_neu.copy() if truth is not None else None
        position_error = (
            self.pos_neu - truth.pos_neu
            if truth is not None
            else None
        )
        velocity_error = (
            self.vel_neu - truth.vel_neu
            if truth is not None
            else None
        )
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
            imu_dt_s=imu_debug.get("dt_s"),
            raw_accel_body=imu_debug.get("raw_accel_body"),
            computed_acc_neu=imu_debug.get("computed_acc_neu"),
            position_error_neu=(
                None if position_error is None else position_error.copy()
            ),
            velocity_error_neu=(
                None if velocity_error is None else velocity_error.copy()
            ),
            visual_velocity_neu=correction.get("visual_velocity_neu"),
            visual_velocity_dt_s=correction.get("visual_dt_s"),
            visual_velocity_reason=str(correction.get("visual_velocity_reason", "")),
            visual_reference_reset=bool(correction.get("visual_reference_reset", False)),
            accel_bias_neu=self.accel_bias_neu.copy(),
        )

    def _correct_with_vision(
        self,
        snapshot,
        now: Optional[float] = None,
    ) -> dict[str, object]:
        visual_now = self._snapshot_wall_time(snapshot) if now is None else float(now)
        self.last_vision_correction = self._empty_vision_correction()

        if not bool(self.config.state_estimation.use_vision_correction):
            return self._vision_no_correction("disabled", visual_now)

        latest_perception = getattr(snapshot, "latest_perception", None)
        feature_vo_measurement = self._measure_feature_vo(
            snapshot,
            latest_perception if isinstance(latest_perception, dict) else None,
            visual_now,
        )
        temporal_vo_measurement = None

        source_mode = str(self.config.state_estimation.vision_correction_source).lower()
        if source_mode == "none":
            visual_odometry_correction = self._apply_preferred_visual_odometry(
                feature_vo_measurement,
                temporal_vo_measurement,
                visual_now,
            )
            return self._prefer_visual_odometry_no_correction(
                visual_odometry_correction,
                "source_none",
                visual_now,
            )

        if not isinstance(latest_perception, dict):
            visual_odometry_correction = self._apply_preferred_visual_odometry(
                feature_vo_measurement,
                temporal_vo_measurement,
                visual_now,
            )
            return self._prefer_visual_odometry_no_correction(
                visual_odometry_correction,
                "no_perception",
                visual_now,
            )

        detections = latest_perception.get("detections")
        if not detections:
            visual_odometry_correction = self._apply_preferred_visual_odometry(
                feature_vo_measurement,
                temporal_vo_measurement,
                visual_now,
            )
            return self._prefer_visual_odometry_no_correction(
                visual_odometry_correction,
                "no_detections",
                visual_now,
            )

        if self._perception_world_pose_source(latest_perception) == "gazebo_camera_sim":
            visual_odometry_correction = self._apply_preferred_visual_odometry(
                feature_vo_measurement,
                temporal_vo_measurement,
                visual_now,
            )
            return self._prefer_visual_odometry_no_correction(
                visual_odometry_correction,
                "gazebo_camera_sim_world_pose",
                visual_now,
            )

        temporal_vo_measurement = self._measure_temporal_vo(
            latest_perception,
            snapshot,
            visual_now,
        )
        visual_odometry_correction = self._apply_preferred_visual_odometry(
            feature_vo_measurement,
            temporal_vo_measurement,
            visual_now,
        )

        landmarks = self._vision_landmarks(snapshot, source_mode)
        if not landmarks:
            return self._prefer_visual_odometry_no_correction(
                visual_odometry_correction,
                "no_landmarks",
                visual_now,
            )

        rot_ned_body = body_frd_to_local_ned_rotmat(
            float(getattr(snapshot, "roll_rad", 0.0)),
            float(getattr(snapshot, "pitch_rad", 0.0)),
            float(getattr(snapshot, "yaw_rad", 0.0))
            + self._perception_yaw_correction_rad(latest_perception),
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
            return self._prefer_visual_odometry_no_correction(
                visual_odometry_correction,
                "no_measurements",
                visual_now,
            )

        unique_sources = sorted(set(sources))
        correction_count = len(unique_sources)
        weighted_residual = float(np.average(residuals, weights=weights))
        min_measurements = int(
            max(
                1,
                getattr(
                    self.config.state_estimation,
                    "vision_correction_min_measurements",
                    2,
                ),
            )
        )
        single_landmark_max_residual = float(
            max(
                0.0,
                getattr(
                    self.config.state_estimation,
                    "vision_correction_single_landmark_max_residual_m",
                    0.35,
                ),
            )
        )
        max_avg_residual = float(
            max(
                0.0,
                getattr(
                    self.config.state_estimation,
                    "vision_correction_max_avg_residual_m",
                    0.8,
                ),
            )
        )

        enough_support = correction_count >= min_measurements
        low_residual_single = (
            correction_count == 1
            and single_landmark_max_residual > 0.0
            and weighted_residual <= single_landmark_max_residual
        )
        if not enough_support and not low_residual_single:
            return self._prefer_visual_odometry_rejection(
                visual_odometry_correction,
                "reject:insufficient_landmark_support",
                weighted_residual,
                correction_count,
                visual_now,
            )

        if max_avg_residual > 0.0 and weighted_residual > max_avg_residual:
            return self._prefer_visual_odometry_rejection(
                visual_odometry_correction,
                "reject:high_avg_residual",
                weighted_residual,
                correction_count,
                visual_now,
            )

        measured_pos = np.average(
            np.asarray(measurements, dtype=float),
            axis=0,
            weights=np.asarray(weights, dtype=float),
        )
        visual_update = self._apply_visual_filter_update(
            measured_pos,
            visual_now,
            residual_m=weighted_residual,
            measurement_count=correction_count,
        )
        visual_velocity = visual_update["visual_velocity_neu"]

        correction = {
            "source": "+".join(unique_sources),
            "residual_m": weighted_residual,
            "count": int(correction_count),
            "visual_velocity_neu": (
                None if visual_velocity is None else visual_velocity.copy()
            ),
            "visual_dt_s": visual_update["visual_dt_s"],
            "visual_velocity_reason": str(visual_update["visual_velocity_reason"]),
            "visual_reference_reset": bool(visual_update["visual_reference_reset"]),
        }
        correction = self._merge_temporal_vo(correction, visual_odometry_correction)
        self.last_vision_correction = correction
        self._trace_visual_correction(correction, visual_now)
        return correction

    def _measure_temporal_vo(
        self,
        latest_perception: dict,
        snapshot,
        now: float,
    ) -> VisualOdometryMeasurement:
        measurement = self.visual_odometry.update(
            latest_perception,
            roll_rad=float(getattr(snapshot, "roll_rad", 0.0)),
            pitch_rad=float(getattr(snapshot, "pitch_rad", 0.0)),
            yaw_rad=float(getattr(snapshot, "yaw_rad", 0.0)),
            estimated_pos_neu=self.pos_neu,
            estimated_vel_neu=self.vel_neu,
            now=now,
        )
        self._trace_temporal_vo(measurement, now)
        return measurement

    def _measure_feature_vo(
        self,
        snapshot,
        latest_perception: Optional[dict],
        now: float,
    ) -> VisualOdometryMeasurement:
        measurement = self.feature_visual_odometry.update(
            getattr(snapshot, "image_bgr", None),
            latest_perception,
            roll_rad=float(getattr(snapshot, "roll_rad", 0.0)),
            pitch_rad=float(getattr(snapshot, "pitch_rad", 0.0)),
            yaw_rad=float(getattr(snapshot, "yaw_rad", 0.0)),
            estimated_pos_neu=self.pos_neu,
            estimated_vel_neu=self.vel_neu,
            now=now,
        )
        self._trace_feature_vo(measurement, now)
        return measurement

    def _apply_preferred_visual_odometry(
        self,
        feature_measurement: Optional[VisualOdometryMeasurement],
        temporal_measurement: Optional[VisualOdometryMeasurement],
        now: float,
    ) -> Optional[dict[str, object]]:
        if (
            feature_measurement is not None
            and feature_measurement.valid
            and feature_measurement.velocity_neu is not None
            and bool(self.config.feature_visual_odometry.fuse_velocity)
        ):
            return self._apply_visual_odometry_measurement(
                feature_measurement,
                "feature_vo",
                now,
                apply_position=False,
            )

        if (
            temporal_measurement is not None
            and temporal_measurement.valid
            and temporal_measurement.velocity_neu is not None
        ):
            return self._apply_visual_odometry_measurement(
                temporal_measurement,
                "gate_keypoint_vo",
                now,
                apply_position=True,
            )

        return None

    def _apply_visual_odometry_measurement(
        self,
        measurement: VisualOdometryMeasurement,
        source: str,
        now: float,
        *,
        apply_position: bool,
    ) -> dict[str, object]:
        if not measurement.valid or measurement.velocity_neu is None:
            return self._empty_vision_correction()

        if apply_position:
            self._correct_position_from_temporal_vo(measurement)
        self._correct_velocity_and_bias_from_visual_velocity(
            measurement.velocity_neu,
            now,
            dt_override_s=measurement.dt_s,
        )

        return {
            "source": source,
            "residual_m": measurement.residual_m,
            "count": int(measurement.feature_count),
            "visual_velocity_neu": measurement.velocity_neu.copy(),
            "visual_dt_s": measurement.dt_s,
            "visual_velocity_reason": source,
            "visual_reference_reset": bool(measurement.reference_reset),
            "wall_time": float(now),
        }

    def _correct_position_from_temporal_vo(
        self,
        measurement: VisualOdometryMeasurement,
    ) -> None:
        if measurement.pose_neu is None:
            return

        target = np.asarray(measurement.pose_neu, dtype=float).reshape(3)
        if not np.all(np.isfinite(target)):
            return

        delta = target - self.pos_neu
        max_delta = float(max(0.0, self.config.visual_odometry.max_position_delta_m))
        if max_delta > 0.0:
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > max_delta:
                delta = delta * (max_delta / max(delta_norm, 1e-9))

        alpha_xy = float(
            np.clip(self.config.visual_odometry.position_alpha_xy, 0.0, 1.0)
        )
        alpha_z = float(np.clip(self.config.visual_odometry.position_alpha_z, 0.0, 1.0))
        self.pos_neu[:2] = self.pos_neu[:2] + alpha_xy * delta[:2]
        self.pos_neu[2] = self.pos_neu[2] + alpha_z * delta[2]

    def _prefer_temporal_vo(
        self,
        temporal_vo_correction: Optional[dict[str, object]],
        fallback: dict[str, object],
    ) -> dict[str, object]:
        if temporal_vo_correction is None:
            return fallback
        self.last_vision_correction = temporal_vo_correction
        self._trace_visual_correction(
            temporal_vo_correction,
            self._finite_float(temporal_vo_correction.get("wall_time")),
        )
        return temporal_vo_correction

    def _prefer_visual_odometry_no_correction(
        self,
        visual_odometry_correction: Optional[dict[str, object]],
        reason: str,
        now: Optional[float],
    ) -> dict[str, object]:
        if visual_odometry_correction is None:
            return self._vision_no_correction(reason, now)
        return self._prefer_temporal_vo(
            visual_odometry_correction,
            self._empty_vision_correction(),
        )

    def _prefer_visual_odometry_rejection(
        self,
        visual_odometry_correction: Optional[dict[str, object]],
        reason: str,
        residual_m: float,
        count: int,
        now: Optional[float],
    ) -> dict[str, object]:
        if visual_odometry_correction is None:
            return self._vision_rejection(reason, residual_m, count, now)
        return self._prefer_temporal_vo(
            visual_odometry_correction,
            self._empty_vision_correction(),
        )

    @staticmethod
    def _merge_temporal_vo(
        correction: dict[str, object],
        temporal_vo_correction: Optional[dict[str, object]],
    ) -> dict[str, object]:
        if temporal_vo_correction is None:
            return correction

        merged = dict(correction)
        source = str(merged.get("source", "") or "")
        vo_source = str(temporal_vo_correction.get("source", "") or "visual_odometry")
        if source and source != vo_source:
            merged["source"] = f"{source}+{vo_source}"
        else:
            merged["source"] = vo_source
        merged["visual_velocity_neu"] = temporal_vo_correction.get("visual_velocity_neu")
        merged["visual_dt_s"] = temporal_vo_correction.get("visual_dt_s")
        merged["visual_velocity_reason"] = temporal_vo_correction.get(
            "visual_velocity_reason",
            vo_source,
        )
        merged["visual_reference_reset"] = temporal_vo_correction.get(
            "visual_reference_reset",
            False,
        )
        return merged

    @staticmethod
    def _perception_world_pose_source(latest_perception) -> str:
        if not isinstance(latest_perception, dict):
            return ""
        return str(latest_perception.get("world_pose_source", "")).lower()

    def _apply_visual_filter_update(
        self,
        measured_pos: np.ndarray,
        now: float,
        *,
        residual_m: float,
        measurement_count: int,
    ) -> dict[str, object]:
        alpha = float(
            np.clip(self.config.state_estimation.vision_correction_alpha, 0.0, 1.0)
        )
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

        visual_velocity, visual_dt, visual_reason = (
            self._visual_velocity_from_measurement(measured_pos, now)
        )
        visual_reference_reset = False
        if visual_velocity is not None:
            velocity_quality_reason = self._visual_velocity_quality_rejection(
                visual_velocity,
                residual_m=residual_m,
                measurement_count=measurement_count,
            )
            if velocity_quality_reason:
                visual_velocity = None
                visual_reason = velocity_quality_reason
            else:
                self._correct_velocity_and_bias_from_visual_velocity(
                    visual_velocity,
                    now,
                )
                self._store_visual_velocity_reference(measured_pos, now)
                visual_reference_reset = True
        elif self._should_reset_visual_velocity_reference(now):
            self._store_visual_velocity_reference(measured_pos, now)
            visual_reference_reset = True
            if visual_reason in ("no_reference", "nonpositive_dt", "dt_above_max"):
                visual_reason = f"{visual_reason}:reset_reference"

        return {
            "visual_velocity_neu": visual_velocity,
            "visual_dt_s": visual_dt,
            "visual_velocity_reason": visual_reason,
            "visual_reference_reset": visual_reference_reset,
        }

    def _store_visual_velocity_reference(
        self,
        measured_pos: np.ndarray,
        now: float,
    ) -> None:
        self.last_visual_measured_pos_neu = measured_pos.copy()
        self.last_visual_update_time = float(now)

    def _should_reset_visual_velocity_reference(self, now: float) -> bool:
        if (
            self.last_visual_update_time is None
            or self.last_visual_measured_pos_neu is None
        ):
            return True

        dt = float(now) - float(self.last_visual_update_time)
        if dt <= 0.0:
            return True

        max_dt = float(
            max(
                0.0,
                getattr(
                    self.config.state_estimation,
                    "vision_correction_max_velocity_dt_s",
                    0.0,
                ),
            )
        )
        return max_dt > 0.0 and dt > max_dt

    def _visual_velocity_from_measurement(
        self,
        measured_pos: np.ndarray,
        now: float,
    ) -> tuple[Optional[np.ndarray], Optional[float], str]:
        if (
            self.last_visual_update_time is None
            or self.last_visual_measured_pos_neu is None
        ):
            return None, None, "no_reference"

        dt = float(now) - float(self.last_visual_update_time)
        min_dt = float(
            max(
                0.0,
                getattr(
                    self.config.state_estimation,
                    "vision_correction_min_velocity_dt_s",
                    0.0,
                ),
            )
        )
        max_dt = float(
            max(
                0.0,
                getattr(
                    self.config.state_estimation,
                    "vision_correction_max_velocity_dt_s",
                    0.0,
                ),
            )
        )
        if dt <= 0.0:
            return None, dt, "nonpositive_dt"
        if dt < min_dt:
            return None, dt, "dt_below_min"
        if max_dt > 0.0 and dt > max_dt:
            return None, dt, "dt_above_max"

        return (measured_pos - self.last_visual_measured_pos_neu) / dt, dt, "computed"

    def _visual_velocity_quality_rejection(
        self,
        visual_velocity: np.ndarray,
        *,
        residual_m: float,
        measurement_count: int,
    ) -> str:
        min_measurements = int(
            max(
                1,
                getattr(
                    self.config.state_estimation,
                    "vision_correction_velocity_min_measurements",
                    1,
                ),
            )
        )
        if int(measurement_count) < min_measurements:
            return "reject:visual_velocity_support"

        max_residual = float(
            max(
                0.0,
                getattr(
                    self.config.state_estimation,
                    "vision_correction_velocity_max_residual_m",
                    0.0,
                ),
            )
        )
        if max_residual > 0.0 and float(residual_m) > max_residual:
            return "reject:visual_velocity_residual"

        speed = float(np.linalg.norm(visual_velocity))
        max_speed = float(
            max(
                0.0,
                getattr(
                    self.config.state_estimation,
                    "vision_correction_max_visual_speed_m_s",
                    0.0,
                ),
            )
        )
        if max_speed > 0.0 and speed > max_speed:
            return "reject:visual_velocity_speed"

        innovation = float(np.linalg.norm(visual_velocity - self.vel_neu))
        max_innovation = float(
            max(
                0.0,
                getattr(
                    self.config.state_estimation,
                    "vision_correction_max_velocity_innovation_m_s",
                    0.0,
                ),
            )
        )
        if max_innovation > 0.0 and innovation > max_innovation:
            return "reject:visual_velocity_innovation"

        return ""

    def _correct_velocity_and_bias_from_visual_velocity(
        self,
        visual_velocity: np.ndarray,
        now: float,
        *,
        dt_override_s: Optional[float] = None,
    ) -> None:
        velocity_delta = visual_velocity - self.vel_neu
        max_velocity_delta = float(
            max(
                0.0,
                getattr(
                    self.config.state_estimation,
                    "vision_correction_max_velocity_delta_m_s",
                    0.0,
                ),
            )
        )
        if max_velocity_delta > 0.0:
            delta_norm = float(np.linalg.norm(velocity_delta))
            if delta_norm > max_velocity_delta:
                velocity_delta = velocity_delta * (
                    max_velocity_delta / max(delta_norm, 1e-9)
                )

        vel_alpha_xy = float(
            np.clip(
                getattr(
                    self.config.state_estimation,
                    "vision_correction_velocity_alpha_xy",
                    0.0,
                ),
                0.0,
                1.0,
            )
        )
        vel_alpha_z = float(
            np.clip(
                getattr(
                    self.config.state_estimation,
                    "vision_correction_velocity_alpha_z",
                    0.0,
                ),
                0.0,
                1.0,
            )
        )
        self.vel_neu[:2] = self.vel_neu[:2] + vel_alpha_xy * velocity_delta[:2]
        self.vel_neu[2] = self.vel_neu[2] + vel_alpha_z * velocity_delta[2]

        if dt_override_s is None:
            if self.last_visual_update_time is None:
                return
            dt = float(now) - float(self.last_visual_update_time)
        else:
            dt = float(dt_override_s)
        if dt <= 0.0:
            return

        bias_delta = -velocity_delta / dt
        max_bias_delta = float(
            max(
                0.0,
                getattr(
                    self.config.state_estimation,
                    "vision_correction_max_bias_delta_m_s2",
                    0.0,
                ),
            )
        )
        if max_bias_delta > 0.0:
            bias_delta_norm = float(np.linalg.norm(bias_delta))
            if bias_delta_norm > max_bias_delta:
                bias_delta = bias_delta * (
                    max_bias_delta / max(bias_delta_norm, 1e-9)
                )

        bias_alpha_xy = float(
            np.clip(
                getattr(
                    self.config.state_estimation,
                    "vision_correction_bias_alpha_xy",
                    0.0,
                ),
                0.0,
                1.0,
            )
        )
        bias_alpha_z = float(
            np.clip(
                getattr(
                    self.config.state_estimation,
                    "vision_correction_bias_alpha_z",
                    0.0,
                ),
                0.0,
                1.0,
            )
        )
        self.accel_bias_neu[:2] = (
            self.accel_bias_neu[:2] + bias_alpha_xy * bias_delta[:2]
        )
        self.accel_bias_neu[2] = self.accel_bias_neu[2] + bias_alpha_z * bias_delta[2]

        max_bias = float(
            max(
                0.0,
                getattr(
                    self.config.state_estimation,
                    "vision_correction_max_accel_bias_m_s2",
                    0.0,
                ),
            )
        )
        if max_bias > 0.0:
            bias_norm = float(np.linalg.norm(self.accel_bias_neu))
            if bias_norm > max_bias:
                self.accel_bias_neu = self.accel_bias_neu * (
                    max_bias / max(bias_norm, 1e-9)
                )

    def _trace_temporal_vo(
        self,
        measurement: VisualOdometryMeasurement,
        now: Optional[float],
    ) -> None:
        if not bool(self.config.visual_odometry.trace):
            return
        if now is None:
            now = time.time()

        signature = (
            bool(measurement.valid),
            str(measurement.reason),
            int(measurement.feature_count),
            int(measurement.track_count),
        )
        elapsed = float(now) - float(self.last_temporal_vo_trace_time)
        min_period = float(max(0.0, self.config.visual_odometry.trace_period_s))
        if signature == self.last_temporal_vo_trace_signature and elapsed < min_period:
            return

        self.last_temporal_vo_trace_signature = signature
        self.last_temporal_vo_trace_time = float(now)

        residual = measurement.residual_m
        residual_txt = (
            "nan"
            if residual is None or not math.isfinite(float(residual))
            else f"{float(residual):.2f}"
        )
        dt = measurement.dt_s
        dt_txt = "nan" if dt is None or not math.isfinite(float(dt)) else f"{float(dt):.3f}"
        speed = measurement.speed_m_s
        speed_txt = (
            "nan"
            if speed is None or not math.isfinite(float(speed))
            else f"{float(speed):.2f}"
        )
        print(
            "temporal_vo_debug "
            f"valid={int(measurement.valid)} "
            f"reason={measurement.reason} "
            f"dt={dt_txt} "
            f"features={int(measurement.feature_count)} "
            f"tracks={int(measurement.track_count)} "
            f"residual={residual_txt} "
            f"speed={speed_txt} "
            f"delta_neu={self._fmt_vec_debug(measurement.delta_neu)} "
            f"vel_neu={self._fmt_vec_debug(measurement.velocity_neu)} "
            f"vo_pose_neu={self._fmt_vec_debug(measurement.pose_neu)}",
            flush=True,
        )

    def _trace_feature_vo(
        self,
        measurement: VisualOdometryMeasurement,
        now: Optional[float],
    ) -> None:
        if not bool(self.config.feature_visual_odometry.trace):
            return
        if now is None:
            now = time.time()

        signature = (
            bool(measurement.valid),
            str(measurement.reason),
            int(measurement.feature_count),
            int(measurement.track_count),
        )
        elapsed = float(now) - float(self.last_feature_vo_trace_time)
        min_period = float(max(0.0, self.config.feature_visual_odometry.trace_period_s))
        if signature == self.last_feature_vo_trace_signature and elapsed < min_period:
            return

        self.last_feature_vo_trace_signature = signature
        self.last_feature_vo_trace_time = float(now)

        residual = measurement.residual_m
        residual_txt = (
            "nan"
            if residual is None or not math.isfinite(float(residual))
            else f"{float(residual):.2f}"
        )
        dt = measurement.dt_s
        dt_txt = "nan" if dt is None or not math.isfinite(float(dt)) else f"{float(dt):.3f}"
        speed = measurement.speed_m_s
        speed_txt = (
            "nan"
            if speed is None or not math.isfinite(float(speed))
            else f"{float(speed):.2f}"
        )
        print(
            "feature_vo_debug "
            f"valid={int(measurement.valid)} "
            f"reason={measurement.reason} "
            f"dt={dt_txt} "
            f"inliers={int(measurement.feature_count)} "
            f"tracks={int(measurement.track_count)} "
            f"residual_px={residual_txt} "
            f"speed={speed_txt} "
            f"delta_neu={self._fmt_vec_debug(measurement.delta_neu)} "
            f"vel_neu={self._fmt_vec_debug(measurement.velocity_neu)} "
            f"vo_pose_neu={self._fmt_vec_debug(measurement.pose_neu)}",
            flush=True,
        )

    def _vision_rejection(
        self,
        reason: str,
        residual_m: float,
        count: int,
        now: Optional[float] = None,
    ) -> dict[str, object]:
        correction = {
            "source": str(reason),
            "residual_m": float(residual_m),
            "count": int(count),
            "visual_velocity_neu": None,
            "visual_dt_s": self._visual_reference_dt(now),
            "visual_velocity_reason": str(reason),
            "visual_reference_reset": False,
        }
        self.last_vision_correction = correction
        self._trace_visual_correction(correction, now)
        return correction

    @staticmethod
    def _empty_vision_correction() -> dict[str, object]:
        return {
            "source": "",
            "residual_m": None,
            "count": 0,
            "visual_velocity_neu": None,
            "visual_dt_s": None,
            "visual_velocity_reason": "",
            "visual_reference_reset": False,
        }

    def _vision_no_correction(
        self,
        reason: str,
        now: Optional[float] = None,
    ) -> dict[str, object]:
        correction = self._empty_vision_correction()
        correction["source"] = str(reason)
        correction["visual_dt_s"] = self._visual_reference_dt(now)
        correction["visual_velocity_reason"] = str(reason)
        self.last_vision_correction = correction
        self._trace_visual_correction(correction, now)
        return correction

    def _visual_reference_dt(self, now: Optional[float]) -> Optional[float]:
        if now is None or self.last_visual_update_time is None:
            return None
        return float(now) - float(self.last_visual_update_time)

    def _trace_visual_correction(
        self,
        correction: dict[str, object],
        now: Optional[float],
    ) -> None:
        if not self._should_trace_visual_correction(correction, now):
            return

        source = str(correction.get("source", ""))
        residual = correction.get("residual_m")
        residual_txt = (
            "nan"
            if residual is None or not math.isfinite(float(residual))
            else f"{float(residual):.2f}"
        )
        visual_dt = correction.get("visual_dt_s")
        visual_dt_txt = (
            "nan"
            if visual_dt is None or not math.isfinite(float(visual_dt))
            else f"{float(visual_dt):.3f}"
        )
        visual_velocity = correction.get("visual_velocity_neu")
        accepted = bool(
            source
            and not source.startswith("reject:")
            and correction.get("residual_m") is not None
        )
        print(
            "estimator_visual_debug "
            f"mode={self.mode} "
            f"accepted={int(accepted)} "
            f"source={source or 'none'} "
            f"residual={residual_txt} "
            f"count={int(correction.get('count', 0))} "
            f"visual_dt={visual_dt_txt} "
            f"visual_ref_reset={int(bool(correction.get('visual_reference_reset', False)))} "
            f"visual_vel_reason={str(correction.get('visual_velocity_reason', '')) or 'none'} "
            f"visual_vel_neu={self._fmt_vec_debug(visual_velocity)} "
            f"pos_neu={self._fmt_vec_debug(self.pos_neu)} "
            f"vel_neu={self._fmt_vec_debug(self.vel_neu)} "
            f"accel_bias_neu={self._fmt_vec_debug(self.accel_bias_neu)}",
            flush=True,
        )

    def _should_trace_visual_correction(
        self,
        correction: dict[str, object],
        now: Optional[float],
    ) -> bool:
        if now is None:
            now = time.time()

        source = str(correction.get("source", ""))
        reason = str(correction.get("visual_velocity_reason", ""))
        has_velocity = correction.get("visual_velocity_neu") is not None
        signature = (
            source,
            reason,
            bool(correction.get("visual_reference_reset", False)),
            has_velocity,
        )

        min_period_s = 0.50
        elapsed = float(now) - float(self.last_visual_debug_print_time)
        if signature == self.last_visual_debug_signature and elapsed < min_period_s:
            return False

        self.last_visual_debug_signature = signature
        self.last_visual_debug_print_time = float(now)
        return True

    @staticmethod
    def _fmt_vec_debug(value) -> str:
        if value is None:
            return "None"
        try:
            arr = np.asarray(value, dtype=float).reshape(3)
        except (TypeError, ValueError):
            return "None"
        if not np.all(np.isfinite(arr)):
            return "None"
        return f"({arr[0]:.2f},{arr[1]:.2f},{arr[2]:.2f})"

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

        self.accel_bias_neu = np.zeros(3, dtype=float)
        self.last_visual_update_time = None
        self.last_visual_measured_pos_neu = None
        self.last_visual_debug_print_time = 0.0
        self.last_visual_debug_signature = None
        self.last_temporal_vo_trace_time = 0.0
        self.last_temporal_vo_trace_signature = None
        self.last_feature_vo_trace_time = 0.0
        self.last_feature_vo_trace_signature = None
        self.visual_odometry.reset()
        self.feature_visual_odometry.reset()
        self.last_wall_time = now
        self.initialized = True

    def _anchor_to_truth(self, truth: VehicleStateEstimate) -> None:
        self.pos_neu = truth.pos_neu.copy()
        self.vel_neu = truth.vel_neu.copy()
        self.accel_bias_neu = np.zeros(3, dtype=float)
        self.last_visual_update_time = None
        self.last_visual_measured_pos_neu = None
        self.last_visual_debug_print_time = 0.0
        self.last_visual_debug_signature = None
        self.last_temporal_vo_trace_time = 0.0
        self.last_temporal_vo_trace_signature = None
        self.last_feature_vo_trace_time = 0.0
        self.last_feature_vo_trace_signature = None
        self.visual_odometry.reset()
        self.feature_visual_odometry.reset()
        self.last_wall_time = truth.wall_time
        self.initialized = True
        self.last_source = truth.source

    def _predict_with_imu(self, snapshot, now: float) -> dict[str, object]:
        debug: dict[str, object] = {
            "dt_s": None,
            "raw_accel_body": None,
            "computed_acc_neu": None,
        }
        if self.last_wall_time is None:
            self.last_wall_time = now
            return debug

        dt = max(0.0, now - float(self.last_wall_time))
        dt = min(dt, float(self.config.state_estimation.max_imu_dt_s))
        self.last_wall_time = now
        debug["dt_s"] = float(dt)
        if dt <= 0.0:
            return debug

        acc_body = self._vec3(getattr(snapshot, "accel_xyz", None))
        if acc_body is None:
            return debug
        debug["raw_accel_body"] = acc_body.copy()

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
        acc_neu = acc_neu - self.accel_bias_neu
        debug["computed_acc_neu"] = acc_neu.copy()

        self.pos_neu = self.pos_neu + self.vel_neu * dt + 0.5 * acc_neu * dt * dt
        self.vel_neu = self.vel_neu + acc_neu * dt

        max_vel = float(self.config.state_estimation.max_imu_velocity_m_s)
        vel_norm = float(np.linalg.norm(self.vel_neu))
        if max_vel > 0.0 and vel_norm > max_vel:
            self.vel_neu = self.vel_neu * (max_vel / max(vel_norm, 1e-9))
        return debug

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
    def _perception_yaw_correction_rad(latest_perception: dict) -> float:
        if not isinstance(latest_perception, dict):
            return 0.0
        value = latest_perception.get("perception_yaw_correction_rad")
        try:
            out = float(value)
        except (TypeError, ValueError):
            return 0.0
        return out if math.isfinite(out) else 0.0

    @staticmethod
    def _finite_float(value) -> Optional[float]:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if math.isfinite(out):
            return out
        return None
