from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from autonomy_core.core.frame_conventions import (
    body_frd_to_local_ned_rotmat,
    local_ned_to_neu,
)


@dataclass(frozen=True)
class VisualOdometryMeasurement:
    valid: bool
    reason: str
    delta_neu: Optional[np.ndarray] = None
    velocity_neu: Optional[np.ndarray] = None
    pose_neu: Optional[np.ndarray] = None
    dt_s: Optional[float] = None
    feature_count: int = 0
    track_count: int = 0
    residual_m: Optional[float] = None
    speed_m_s: Optional[float] = None
    reference_reset: bool = False


@dataclass(frozen=True)
class _GateObservation:
    detection_index: int
    center_px: np.ndarray
    depth_m: float
    corners_neu: dict[int, np.ndarray]


class GateKeypointVisualOdometry:
    """
    Temporal VO from semantic gate keypoints.

    Each gate corner is treated as a static feature. For a matched gate seen in
    two frames, vehicle displacement is the previous corner-relative vector
    minus the current corner-relative vector, expressed in local NEU.
    """

    def __init__(self, config):
        self.config = config
        self.previous_observations: list[_GateObservation] = []
        self.previous_time: Optional[float] = None
        self.previous_frame_key: Optional[object] = None
        self.pose_neu: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.previous_observations = []
        self.previous_time = None
        self.previous_frame_key = None
        self.pose_neu = None

    def update(
        self,
        latest_perception,
        *,
        roll_rad: float,
        pitch_rad: float,
        yaw_rad: float,
        estimated_pos_neu: np.ndarray,
        estimated_vel_neu: np.ndarray,
        now: float,
    ) -> VisualOdometryMeasurement:
        if not bool(self.config.visual_odometry.enabled):
            return VisualOdometryMeasurement(False, "disabled")

        if not isinstance(latest_perception, dict):
            return VisualOdometryMeasurement(False, "no_perception")

        frame_key = self._frame_key(latest_perception)
        if frame_key is not None and frame_key == self.previous_frame_key:
            return VisualOdometryMeasurement(False, "duplicate_frame")

        timestamp = self._timestamp(latest_perception, now)
        observations = self._observations(
            latest_perception,
            roll_rad=roll_rad,
            pitch_rad=pitch_rad,
            yaw_rad=yaw_rad,
        )
        if not observations:
            return VisualOdometryMeasurement(False, "no_gate_keypoint_observations")

        if self.previous_time is None or not self.previous_observations:
            self._store_reference(
                observations,
                timestamp,
                frame_key,
                estimated_pos_neu,
            )
            return VisualOdometryMeasurement(
                False,
                "no_reference",
                reference_reset=True,
            )

        dt = timestamp - float(self.previous_time)
        min_dt = float(max(0.0, self.config.visual_odometry.min_dt_s))
        max_dt = float(max(0.0, self.config.visual_odometry.max_dt_s))
        if dt <= 0.0:
            return VisualOdometryMeasurement(False, "nonpositive_dt", dt_s=dt)
        if dt < min_dt:
            return VisualOdometryMeasurement(False, "dt_below_min", dt_s=dt)
        if max_dt > 0.0 and dt > max_dt:
            self._store_reference(
                observations,
                timestamp,
                frame_key,
                estimated_pos_neu,
            )
            return VisualOdometryMeasurement(
                False,
                "dt_above_max",
                dt_s=dt,
                reference_reset=True,
            )

        matches = self._match_observations(observations)
        corner_deltas = []
        for previous, current in matches:
            common = sorted(set(previous.corners_neu) & set(current.corners_neu))
            for corner_index in common:
                corner_deltas.append(
                    previous.corners_neu[corner_index] - current.corners_neu[corner_index]
                )

        feature_count = len(corner_deltas)
        min_features = int(max(1, self.config.visual_odometry.min_features))
        if feature_count < min_features:
            return VisualOdometryMeasurement(
                False,
                "insufficient_features",
                dt_s=dt,
                feature_count=feature_count,
                track_count=len(matches),
            )

        deltas = np.asarray(corner_deltas, dtype=float).reshape(-1, 3)
        delta_neu = np.median(deltas, axis=0)
        residuals = np.linalg.norm(deltas - delta_neu, axis=1)
        residual_m = float(np.median(residuals))
        max_residual = float(max(0.0, self.config.visual_odometry.max_corner_delta_std_m))
        if max_residual > 0.0 and residual_m > max_residual:
            return VisualOdometryMeasurement(
                False,
                "corner_delta_residual_high",
                delta_neu=delta_neu.copy(),
                dt_s=dt,
                feature_count=feature_count,
                track_count=len(matches),
                residual_m=residual_m,
            )

        velocity_neu = delta_neu / dt
        speed_m_s = float(np.linalg.norm(velocity_neu))
        max_speed = float(max(0.0, self.config.visual_odometry.max_visual_speed_m_s))
        if max_speed > 0.0 and speed_m_s > max_speed:
            return VisualOdometryMeasurement(
                False,
                "visual_speed_high",
                delta_neu=delta_neu.copy(),
                velocity_neu=velocity_neu.copy(),
                dt_s=dt,
                feature_count=feature_count,
                track_count=len(matches),
                residual_m=residual_m,
                speed_m_s=speed_m_s,
            )

        innovation = float(
            np.linalg.norm(velocity_neu - np.asarray(estimated_vel_neu, dtype=float).reshape(3))
        )
        max_innovation = float(
            max(0.0, self.config.visual_odometry.max_velocity_innovation_m_s)
        )
        if max_innovation > 0.0 and innovation > max_innovation:
            return VisualOdometryMeasurement(
                False,
                "velocity_innovation_high",
                delta_neu=delta_neu.copy(),
                velocity_neu=velocity_neu.copy(),
                dt_s=dt,
                feature_count=feature_count,
                track_count=len(matches),
                residual_m=residual_m,
                speed_m_s=speed_m_s,
            )

        if self.pose_neu is None:
            self.pose_neu = np.asarray(estimated_pos_neu, dtype=float).reshape(3).copy()
        self.pose_neu = self.pose_neu + delta_neu
        pose_neu = self.pose_neu.copy()
        self._store_reference(observations, timestamp, frame_key, pose_neu)

        return VisualOdometryMeasurement(
            True,
            "gate_keypoint_vo",
            delta_neu=delta_neu.copy(),
            velocity_neu=velocity_neu.copy(),
            pose_neu=pose_neu,
            dt_s=dt,
            feature_count=feature_count,
            track_count=len(matches),
            residual_m=residual_m,
            speed_m_s=speed_m_s,
        )

    def _store_reference(
        self,
        observations: list[_GateObservation],
        timestamp: float,
        frame_key: Optional[object],
        pose_neu,
    ) -> None:
        self.previous_observations = list(observations)
        self.previous_time = float(timestamp)
        self.previous_frame_key = frame_key
        self.pose_neu = np.asarray(pose_neu, dtype=float).reshape(3).copy()

    def _observations(
        self,
        latest_perception: dict,
        *,
        roll_rad: float,
        pitch_rad: float,
        yaw_rad: float,
    ) -> list[_GateObservation]:
        detections = latest_perception.get("detections")
        if not detections:
            return []

        camera_to_body = self._matrix3(latest_perception.get("camera_to_body"))
        if camera_to_body is None:
            return []

        camera_translation_body = self._vec3(
            latest_perception.get("camera_translation_body")
        )
        if camera_translation_body is None:
            camera_translation_body = np.zeros(3, dtype=float)

        yaw_used = float(yaw_rad) + self._perception_yaw_correction_rad(latest_perception)
        rot_ned_body = body_frd_to_local_ned_rotmat(
            float(roll_rad),
            float(pitch_rad),
            yaw_used,
        )

        observations: list[_GateObservation] = []
        for detection_index, detection in enumerate(detections):
            if not isinstance(detection, dict):
                continue
            observation = self._observation_from_detection(
                detection,
                detection_index=detection_index,
                camera_to_body=camera_to_body,
                camera_translation_body=camera_translation_body,
                rot_ned_body=rot_ned_body,
            )
            if observation is not None:
                observations.append(observation)
        return observations

    def _observation_from_detection(
        self,
        detection: dict,
        *,
        detection_index: int,
        camera_to_body: np.ndarray,
        camera_translation_body: np.ndarray,
        rot_ned_body: np.ndarray,
    ) -> Optional[_GateObservation]:
        keypoints_px = self._keypoints_px(detection)
        if keypoints_px is None:
            return None

        keypoint_conf = self._keypoint_conf(detection, len(keypoints_px))
        min_conf = float(max(0.0, self.config.visual_odometry.min_keypoint_conf))
        valid_indices = [
            idx
            for idx, conf in enumerate(keypoint_conf[: len(keypoints_px)])
            if math.isfinite(float(conf)) and float(conf) >= min_conf
        ]
        if len(valid_indices) < int(max(1, self.config.visual_odometry.min_features)):
            return None

        object_points = self._object_points(detection)
        rvec = self._vec3(detection.get("rvec"))
        tvec = self._vec3(detection.get("tvec"))
        if object_points is None or rvec is None or tvec is None:
            return None

        object_points = object_points[: len(keypoints_px)]
        if object_points.shape[0] <= max(valid_indices):
            return None

        try:
            rot_obj_to_camera, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        except cv2.error:
            return None

        corners_camera = (rot_obj_to_camera @ object_points.T).T + tvec.reshape(1, 3)
        if not np.all(np.isfinite(corners_camera)):
            return None

        corners_neu: dict[int, np.ndarray] = {}
        for idx in valid_indices:
            corner_body = camera_translation_body + camera_to_body @ corners_camera[idx]
            corners_neu[idx] = local_ned_to_neu(rot_ned_body @ corner_body)

        if not corners_neu:
            return None

        center_px = np.mean(keypoints_px[valid_indices], axis=0)
        center_camera = self._vec3(detection.get("gate_center_camera"))
        if center_camera is None:
            center_camera = np.mean(corners_camera[valid_indices], axis=0)
        depth_m = float(center_camera[2])
        if not math.isfinite(depth_m) or depth_m <= 0.0:
            return None

        return _GateObservation(
            detection_index=int(detection_index),
            center_px=np.asarray(center_px, dtype=float).reshape(2),
            depth_m=depth_m,
            corners_neu=corners_neu,
        )

    def _match_observations(
        self,
        current_observations: list[_GateObservation],
    ) -> list[tuple[_GateObservation, _GateObservation]]:
        matches: list[tuple[_GateObservation, _GateObservation]] = []
        used_previous: set[int] = set()
        max_shift_px = float(max(0.0, self.config.visual_odometry.max_match_center_shift_px))
        max_depth_delta = float(max(0.0, self.config.visual_odometry.max_match_depth_delta_m))

        for current in current_observations:
            best_index = None
            best_score = float("inf")
            for previous_index, previous in enumerate(self.previous_observations):
                if previous_index in used_previous:
                    continue
                common = set(previous.corners_neu) & set(current.corners_neu)
                if not common:
                    continue
                center_shift = float(np.linalg.norm(current.center_px - previous.center_px))
                if max_shift_px > 0.0 and center_shift > max_shift_px:
                    continue
                depth_delta = abs(float(current.depth_m) - float(previous.depth_m))
                if max_depth_delta > 0.0 and depth_delta > max_depth_delta:
                    continue
                score = center_shift + 10.0 * depth_delta
                if score < best_score:
                    best_index = previous_index
                    best_score = score

            if best_index is not None:
                used_previous.add(best_index)
                matches.append((self.previous_observations[best_index], current))

        return matches

    @staticmethod
    def _frame_key(latest_perception: dict) -> Optional[object]:
        frame_id = latest_perception.get("frame_id")
        if frame_id is not None:
            try:
                return ("frame_id", int(frame_id))
            except (TypeError, ValueError):
                pass
        timestamp = latest_perception.get("perception_wall_time")
        try:
            value = float(timestamp)
        except (TypeError, ValueError):
            return None
        return ("time", value) if math.isfinite(value) else None

    @staticmethod
    def _timestamp(latest_perception: dict, now: float) -> float:
        for key in ("perception_wall_time", "image_wall_time", "timestamp"):
            try:
                value = float(latest_perception.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
        return float(now)

    @staticmethod
    def _keypoints_px(detection: dict) -> Optional[np.ndarray]:
        for key in ("keypoints_px", "yolo_keypoints", "ordered_corners"):
            value = detection.get(key)
            if value is None:
                continue
            try:
                arr = np.asarray(value, dtype=float)
            except (TypeError, ValueError):
                continue
            if arr.ndim == 2 and arr.shape[0] >= 4 and arr.shape[1] >= 2:
                out = arr[:4, :2].reshape(4, 2)
                if np.all(np.isfinite(out)):
                    return out
        return None

    @staticmethod
    def _keypoint_conf(detection: dict, count: int) -> np.ndarray:
        value = detection.get("keypoint_conf")
        if value is not None:
            try:
                arr = np.asarray(value, dtype=float).reshape(-1)
                if arr.shape[0] >= count:
                    return arr[:count]
            except (TypeError, ValueError):
                pass

        value = detection.get("yolo_keypoints")
        if value is not None:
            try:
                arr = np.asarray(value, dtype=float)
                if arr.ndim == 2 and arr.shape[0] >= count and arr.shape[1] >= 3:
                    return arr[:count, 2]
            except (TypeError, ValueError):
                pass

        return np.ones(count, dtype=float)

    @staticmethod
    def _object_points(detection: dict) -> Optional[np.ndarray]:
        value = detection.get("object_points_m")
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return None
        if arr.ndim == 2 and arr.shape[0] >= 4 and arr.shape[1] >= 3:
            out = arr[:4, :3].reshape(4, 3)
            if np.all(np.isfinite(out)):
                return out
        return None

    @staticmethod
    def _vec3(value) -> Optional[np.ndarray]:
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=float).reshape(3)
        except (TypeError, ValueError):
            return None
        return arr.copy() if np.all(np.isfinite(arr)) else None

    @staticmethod
    def _matrix3(value) -> Optional[np.ndarray]:
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=float).reshape(3, 3)
        except (TypeError, ValueError):
            return None
        return arr.copy() if np.all(np.isfinite(arr)) else None

    @staticmethod
    def _perception_yaw_correction_rad(latest_perception: dict) -> float:
        try:
            value = float(latest_perception.get("perception_yaw_correction_rad"))
        except (TypeError, ValueError):
            return 0.0
        return value if math.isfinite(value) else 0.0
