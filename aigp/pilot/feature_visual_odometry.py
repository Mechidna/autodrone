from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from autonomy_core.core.competition_config import VADR_TS_002
from autonomy_core.core.frame_conventions import (
    body_frd_to_local_ned_rotmat,
    local_ned_to_neu,
    official_camera_to_body_frd_rotmat,
)
from visual_odometry import VisualOdometryMeasurement


class FeatureVisualOdometry:
    """
    Sparse image-feature visual odometry.

    This is intentionally relative-only: it uses image features and camera
    calibration to estimate motion direction, then scales that direction with
    the estimator's current inertial speed. It does not use known gate
    coordinates, map truth, or fabricated altitude.
    """

    def __init__(self, config):
        self.config = config
        self.previous_gray: Optional[np.ndarray] = None
        self.previous_points: Optional[np.ndarray] = None
        self.previous_time: Optional[float] = None
        self.previous_frame_key: Optional[object] = None
        self.pose_neu: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.previous_gray = None
        self.previous_points = None
        self.previous_time = None
        self.previous_frame_key = None
        self.pose_neu = None

    def update(
        self,
        image_bgr,
        latest_perception,
        *,
        roll_rad: float,
        pitch_rad: float,
        yaw_rad: float,
        estimated_pos_neu: np.ndarray,
        estimated_vel_neu: np.ndarray,
        now: float,
    ) -> VisualOdometryMeasurement:
        cfg = self.config.feature_visual_odometry
        if not bool(cfg.enabled):
            return VisualOdometryMeasurement(False, "disabled")

        gray = self._gray_image(image_bgr)
        if gray is None:
            return VisualOdometryMeasurement(False, "no_image")

        frame_key = self._frame_key(image_bgr, latest_perception)
        if frame_key is not None and frame_key == self.previous_frame_key:
            return VisualOdometryMeasurement(False, "duplicate_frame")

        timestamp = self._timestamp(image_bgr, latest_perception, now)
        if self.previous_gray is None or self.previous_points is None:
            self._store_reference(gray, timestamp, frame_key, estimated_pos_neu)
            return VisualOdometryMeasurement(
                False,
                "no_reference",
                reference_reset=True,
            )

        dt = timestamp - float(self.previous_time)
        min_dt = float(max(0.0, cfg.min_dt_s))
        max_dt = float(max(0.0, cfg.max_dt_s))
        if dt <= 0.0:
            return VisualOdometryMeasurement(False, "nonpositive_dt", dt_s=dt)
        if dt < min_dt:
            return VisualOdometryMeasurement(False, "dt_below_min", dt_s=dt)
        if max_dt > 0.0 and dt > max_dt:
            self._store_reference(gray, timestamp, frame_key, estimated_pos_neu)
            return VisualOdometryMeasurement(
                False,
                "dt_above_max",
                dt_s=dt,
                reference_reset=True,
            )

        prev_pts, curr_pts, fb_err = self._track_features(gray)
        track_count = int(curr_pts.shape[0])
        min_tracks = int(max(1, cfg.min_tracks))
        if track_count < min_tracks:
            self._store_reference(gray, timestamp, frame_key, estimated_pos_neu)
            return VisualOdometryMeasurement(
                False,
                "insufficient_tracks",
                dt_s=dt,
                feature_count=track_count,
                track_count=track_count,
                reference_reset=True,
            )

        flow_px = np.linalg.norm(curr_pts - prev_pts, axis=1)
        median_flow = float(np.median(flow_px)) if flow_px.size else 0.0
        max_flow = float(max(0.0, cfg.max_median_flow_px))
        if max_flow > 0.0 and median_flow > max_flow:
            self._store_reference(gray, timestamp, frame_key, estimated_pos_neu)
            return VisualOdometryMeasurement(
                False,
                "flow_too_large",
                dt_s=dt,
                feature_count=track_count,
                track_count=track_count,
                residual_m=median_flow,
                reference_reset=True,
            )

        camera_matrix = self._camera_matrix(latest_perception)
        if camera_matrix is None:
            self._store_reference(gray, timestamp, frame_key, estimated_pos_neu)
            return VisualOdometryMeasurement(
                False,
                "missing_camera_matrix",
                dt_s=dt,
                feature_count=track_count,
                track_count=track_count,
                reference_reset=True,
            )

        pose = self._relative_pose(prev_pts, curr_pts, fb_err, camera_matrix)
        if pose is None:
            self._store_reference(gray, timestamp, frame_key, estimated_pos_neu)
            return VisualOdometryMeasurement(
                False,
                "essential_pose_failed",
                dt_s=dt,
                feature_count=track_count,
                track_count=track_count,
                reference_reset=True,
            )
        direction_camera, inlier_count, residual_px, inlier_curr_pts = pose

        min_inliers = int(max(1, cfg.min_inliers))
        min_ratio = float(max(0.0, min(float(cfg.min_inlier_ratio), 1.0)))
        inlier_ratio = inlier_count / max(track_count, 1)
        if inlier_count < min_inliers or inlier_ratio < min_ratio:
            self._store_reference(gray, timestamp, frame_key, estimated_pos_neu)
            return VisualOdometryMeasurement(
                False,
                "insufficient_inliers",
                dt_s=dt,
                feature_count=inlier_count,
                track_count=track_count,
                residual_m=residual_px,
                reference_reset=True,
            )

        direction_neu = self._direction_neu(
            direction_camera,
            latest_perception,
            roll_rad=roll_rad,
            pitch_rad=pitch_rad,
            yaw_rad=yaw_rad,
        )
        if direction_neu is None:
            self._store_reference(gray, timestamp, frame_key, estimated_pos_neu)
            return VisualOdometryMeasurement(
                False,
                "invalid_motion_direction",
                dt_s=dt,
                feature_count=inlier_count,
                track_count=track_count,
                residual_m=residual_px,
                reference_reset=True,
            )

        estimated_vel = np.asarray(estimated_vel_neu, dtype=float).reshape(3)
        speed_estimate = float(np.linalg.norm(estimated_vel))
        min_scale_speed = float(max(0.0, cfg.min_scale_speed_m_s))
        if speed_estimate < min_scale_speed:
            self._store_reference(
                gray,
                timestamp,
                frame_key,
                estimated_pos_neu,
                points=inlier_curr_pts,
            )
            return VisualOdometryMeasurement(
                False,
                "scale_speed_low",
                dt_s=dt,
                feature_count=inlier_count,
                track_count=track_count,
                residual_m=residual_px,
                reference_reset=True,
            )

        predicted_delta = estimated_vel * dt
        if float(np.dot(direction_neu, predicted_delta)) < 0.0:
            direction_neu = -direction_neu

        delta_neu = direction_neu * float(np.linalg.norm(predicted_delta))
        velocity_neu = delta_neu / dt
        speed_m_s = float(np.linalg.norm(velocity_neu))

        max_speed = float(max(0.0, cfg.max_visual_speed_m_s))
        if max_speed > 0.0 and speed_m_s > max_speed:
            self._store_reference(gray, timestamp, frame_key, estimated_pos_neu)
            return VisualOdometryMeasurement(
                False,
                "visual_speed_high",
                delta_neu=delta_neu.copy(),
                velocity_neu=velocity_neu.copy(),
                dt_s=dt,
                feature_count=inlier_count,
                track_count=track_count,
                residual_m=residual_px,
                speed_m_s=speed_m_s,
                reference_reset=True,
            )

        innovation = float(np.linalg.norm(velocity_neu - estimated_vel))
        max_innovation = float(max(0.0, cfg.max_velocity_innovation_m_s))
        if max_innovation > 0.0 and innovation > max_innovation:
            self._store_reference(
                gray,
                timestamp,
                frame_key,
                estimated_pos_neu,
                points=inlier_curr_pts,
            )
            return VisualOdometryMeasurement(
                False,
                "velocity_innovation_high",
                delta_neu=delta_neu.copy(),
                velocity_neu=velocity_neu.copy(),
                dt_s=dt,
                feature_count=inlier_count,
                track_count=track_count,
                residual_m=residual_px,
                speed_m_s=speed_m_s,
                reference_reset=True,
            )

        if self.pose_neu is None:
            self.pose_neu = np.asarray(estimated_pos_neu, dtype=float).reshape(3).copy()
        self.pose_neu = self.pose_neu + delta_neu
        pose_neu = self.pose_neu.copy()
        self._store_reference(
            gray,
            timestamp,
            frame_key,
            pose_neu,
            points=inlier_curr_pts,
        )

        return VisualOdometryMeasurement(
            True,
            "feature_vo",
            delta_neu=delta_neu.copy(),
            velocity_neu=velocity_neu.copy(),
            pose_neu=pose_neu,
            dt_s=dt,
            feature_count=inlier_count,
            track_count=track_count,
            residual_m=residual_px,
            speed_m_s=speed_m_s,
        )

    def _store_reference(
        self,
        gray: np.ndarray,
        timestamp: float,
        frame_key: Optional[object],
        pose_neu,
        *,
        points: Optional[np.ndarray] = None,
    ) -> None:
        if points is None or points.shape[0] < int(self.config.feature_visual_odometry.redetect_below_tracks):
            points = self._detect_features(gray)

        self.previous_gray = gray.copy()
        self.previous_points = points.reshape(-1, 1, 2).astype(np.float32)
        self.previous_time = float(timestamp)
        self.previous_frame_key = frame_key
        self.pose_neu = np.asarray(pose_neu, dtype=float).reshape(3).copy()

    def _detect_features(self, gray: np.ndarray) -> np.ndarray:
        cfg = self.config.feature_visual_odometry
        detector = str(cfg.detector).lower()
        if detector == "orb":
            orb = cv2.ORB_create(nfeatures=int(max(1, cfg.max_features)))
            keypoints = orb.detect(gray, None)
            points = np.asarray([kp.pt for kp in keypoints], dtype=np.float32)
            if points.size:
                return points.reshape(-1, 2)

        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=int(max(1, cfg.max_features)),
            qualityLevel=float(max(1e-6, cfg.quality_level)),
            minDistance=float(max(1.0, cfg.min_distance_px)),
            blockSize=int(max(3, cfg.block_size_px)),
        )
        if points is None:
            return np.empty((0, 2), dtype=np.float32)
        return points.reshape(-1, 2).astype(np.float32)

    def _track_features(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config.feature_visual_odometry
        prev = np.asarray(self.previous_points, dtype=np.float32).reshape(-1, 1, 2)
        if prev.size == 0:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        lk_params = dict(
            winSize=(int(cfg.lk_win_size_px), int(cfg.lk_win_size_px)),
            maxLevel=int(max(0, cfg.lk_max_level)),
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                int(max(1, cfg.lk_max_iter)),
                float(max(1e-6, cfg.lk_epsilon)),
            ),
        )
        curr, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            gray,
            prev,
            None,
            **lk_params,
        )
        if curr is None or status is None:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        back, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            self.previous_gray,
            curr,
            None,
            **lk_params,
        )
        if back is None or back_status is None:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        prev_2d = prev.reshape(-1, 2)
        curr_2d = curr.reshape(-1, 2)
        back_2d = back.reshape(-1, 2)
        fb_err = np.linalg.norm(prev_2d - back_2d, axis=1)
        good = (
            (status.reshape(-1) == 1)
            & (back_status.reshape(-1) == 1)
            & np.all(np.isfinite(curr_2d), axis=1)
            & np.isfinite(fb_err)
            & (fb_err <= float(max(0.0, cfg.max_forward_backward_error_px)))
        )

        height, width = gray.shape[:2]
        border = float(max(0.0, cfg.border_margin_px))
        if border > 0.0:
            good &= (
                (curr_2d[:, 0] >= border)
                & (curr_2d[:, 0] < width - border)
                & (curr_2d[:, 1] >= border)
                & (curr_2d[:, 1] < height - border)
            )

        return prev_2d[good], curr_2d[good], fb_err[good]

    def _relative_pose(
        self,
        prev_pts: np.ndarray,
        curr_pts: np.ndarray,
        fb_err: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> Optional[tuple[np.ndarray, int, float, np.ndarray]]:
        cfg = self.config.feature_visual_odometry
        if prev_pts.shape[0] < 5 or curr_pts.shape[0] < 5:
            return None

        try:
            essential, mask = cv2.findEssentialMat(
                prev_pts.astype(np.float32),
                curr_pts.astype(np.float32),
                camera_matrix,
                method=cv2.RANSAC,
                prob=float(min(max(cfg.ransac_prob, 0.0), 1.0)),
                threshold=float(max(0.1, cfg.ransac_threshold_px)),
            )
        except cv2.error:
            return None
        if essential is None or mask is None:
            return None
        essential = np.asarray(essential, dtype=float)
        if essential.shape[0] > 3:
            essential = essential[:3, :3]
        if essential.shape != (3, 3):
            return None

        inlier_mask = mask.reshape(-1).astype(bool)
        if int(np.count_nonzero(inlier_mask)) < 5:
            return None

        try:
            _, _, translation, pose_mask = cv2.recoverPose(
                essential,
                prev_pts.astype(np.float32),
                curr_pts.astype(np.float32),
                camera_matrix,
                mask=mask,
            )
        except cv2.error:
            return None

        pose_inliers = pose_mask.reshape(-1).astype(bool) if pose_mask is not None else inlier_mask
        combined = inlier_mask & pose_inliers
        inlier_count = int(np.count_nonzero(combined))
        if inlier_count <= 0:
            return None

        residual_px = float(np.median(fb_err[combined])) if fb_err.size else 0.0
        direction_camera = np.asarray(translation, dtype=float).reshape(3)
        inlier_curr_pts = curr_pts[combined].reshape(-1, 2).astype(np.float32)
        return direction_camera, inlier_count, residual_px, inlier_curr_pts

    def _direction_neu(
        self,
        direction_camera: np.ndarray,
        latest_perception,
        *,
        roll_rad: float,
        pitch_rad: float,
        yaw_rad: float,
    ) -> Optional[np.ndarray]:
        norm = float(np.linalg.norm(direction_camera))
        if not math.isfinite(norm) or norm <= 1e-9:
            return None

        camera_to_body = self._camera_to_body(latest_perception)
        yaw_used = float(yaw_rad) + self._perception_yaw_correction_rad(latest_perception)
        rot_ned_body = body_frd_to_local_ned_rotmat(
            float(roll_rad),
            float(pitch_rad),
            yaw_used,
        )
        direction_body = camera_to_body @ (direction_camera / norm)
        direction_neu = local_ned_to_neu(rot_ned_body @ direction_body)
        direction_norm = float(np.linalg.norm(direction_neu))
        if not math.isfinite(direction_norm) or direction_norm <= 1e-9:
            return None
        return direction_neu / direction_norm

    def _camera_matrix(self, latest_perception) -> Optional[np.ndarray]:
        if isinstance(latest_perception, dict):
            matrix = latest_perception.get("camera_matrix")
            if matrix is not None:
                try:
                    arr = np.asarray(matrix, dtype=float).reshape(3, 3)
                    if np.all(np.isfinite(arr)):
                        return arr.copy()
                except (TypeError, ValueError):
                    pass
        try:
            arr = np.asarray(self.config.camera.matrix, dtype=float).reshape(3, 3)
        except (TypeError, ValueError):
            return None
        return arr.copy() if np.all(np.isfinite(arr)) else None

    @staticmethod
    def _camera_to_body(latest_perception) -> np.ndarray:
        if isinstance(latest_perception, dict):
            matrix = latest_perception.get("camera_to_body")
            if matrix is not None:
                try:
                    arr = np.asarray(matrix, dtype=float).reshape(3, 3)
                    if np.all(np.isfinite(arr)):
                        return arr.copy()
                except (TypeError, ValueError):
                    pass
        return np.asarray(official_camera_to_body_frd_rotmat(VADR_TS_002), dtype=float).reshape(3, 3)

    @staticmethod
    def _perception_yaw_correction_rad(latest_perception) -> float:
        if not isinstance(latest_perception, dict):
            return 0.0
        try:
            value = float(latest_perception.get("perception_yaw_correction_rad"))
        except (TypeError, ValueError):
            return 0.0
        return value if math.isfinite(value) else 0.0

    @staticmethod
    def _gray_image(image_bgr) -> Optional[np.ndarray]:
        if image_bgr is None:
            return None
        if isinstance(image_bgr, dict):
            image_bgr = image_bgr.get("image")
        try:
            image = np.asarray(image_bgr)
        except (TypeError, ValueError):
            return None
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3 and image.shape[2] >= 3:
            gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            return None
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        return gray

    @staticmethod
    def _frame_key(image_bgr, latest_perception) -> Optional[object]:
        for source in (latest_perception, image_bgr):
            if not isinstance(source, dict):
                continue
            frame_id = source.get("frame_id")
            if frame_id is not None:
                try:
                    return ("frame_id", int(frame_id))
                except (TypeError, ValueError):
                    pass
        return None

    @staticmethod
    def _timestamp(image_bgr, latest_perception, now: float) -> float:
        for source in (latest_perception, image_bgr):
            if not isinstance(source, dict):
                continue
            for key in ("image_wall_time", "wall_time", "timestamp", "perception_wall_time"):
                try:
                    value = float(source.get(key))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value) and value > 0.0:
                    return value
        return float(now)
