# Verified with:
# python3 -m compileall autonomy_core/estimation
# A smoke test where seeded gate landmarks corrected a +1 m odometry bias to about -0.978 m estimated drift.
# To continue wiring this into the rest of the stack, the missing interfaces are:
# 1. px4_runner.py should feed odometry into estimator.update_odometry(...) each telemetry update.
# 2. GatePerceptionNode.detect_gate(...) already computes gate_center_body; that should feed estimator.update_gate_detection(...) before GateMemory or planning.
# 3. AutonomyAPI should use estimator.get_state().position, .velocity, and .yaw instead of raw telemetry.pos/rpy for planning and control.
# 4. GateMemory and the estimator should be reconciled: either estimator landmarks replace GateMemory, or GateMemory supplies stable landmark_ids into estimator updates.
# 5. For AlphaPilot-style behavior, seed rough known gate locations via set_landmark(...) / set_landmarks(...); otherwise the estimator can build a map, but it cannot correct initial drift until it has an external or previously committed reference.

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


def _as_vec3(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be convertible to shape (3,), got {arr.shape}")
    return arr


def _wrap_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _yaw_to_rotmat(yaw: float) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _dyaw_rotmat(yaw: float) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array(
        [
            [-s, -c, 0.0],
            [c, -s, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )


@dataclass
class OdometrySample:
    """
    Odometry/VIO/PX4 state in a locally consistent odometry frame.

    Position and velocity should be z-up and expressed in the same convention
    used by the rest of the autonomy stack.
    """

    position: np.ndarray
    velocity: np.ndarray
    yaw: float
    timestamp: float
    covariance: Optional[np.ndarray] = None


@dataclass
class GateDetection:
    """
    Relative gate-center measurement.

    `position_body` is the vector from the drone/body origin to the observed
    gate center, expressed in the drone body frame. If a perception module
    reports camera-frame translation, convert it to body frame before calling
    the estimator.
    """

    position_body: np.ndarray
    timestamp: float
    confidence: float = 1.0
    landmark_id: Optional[int] = None
    measurement_covariance: Optional[np.ndarray] = None


@dataclass
class GateLandmark:
    id: int
    center: np.ndarray
    covariance: np.ndarray
    hits: int = 1
    confidence_sum: float = 1.0
    committed: bool = False
    last_seen_time: float = 0.0
    recent_errors: list = field(default_factory=list)


@dataclass
class EstimatedState:
    """
    Drift-corrected state for planning/control.
    """

    position: np.ndarray
    velocity: np.ndarray
    yaw: float
    timestamp: float
    odom_position: np.ndarray
    odom_yaw: float
    drift_position: np.ndarray
    drift_yaw: float
    covariance: np.ndarray


class DriftCorrectingGateEKF:
    """
    Lightweight gate-landmark estimator for the racing stack.

    The filter keeps a slow correction from odometry/VIO/PX4 coordinates into
    the gate-map frame:

        p_corrected = p_odom + drift_position
        yaw_corrected = yaw_odom + drift_yaw

    Gate detections update that correction against persistent landmarks. This
    is intentionally narrower than a full VIO EKF: it is an integration layer
    that can sit between telemetry/perception and the existing planner without
    requiring changes to either of those packages.
    """

    def __init__(
        self,
        drift_process_std: Tuple[float, float] = (0.08, np.deg2rad(2.0)),
        initial_drift_std: Tuple[float, float] = (1.0, np.deg2rad(15.0)),
        default_gate_measurement_std: float = 0.35,
        association_radius: float = 2.0,
        commit_hits: int = 3,
        commit_confidence_sum: float = 2.0,
        min_confidence: float = 0.2,
        uncommitted_alpha: float = 0.35,
        committed_alpha: float = 0.02,
        max_update_distance: float = 6.0,
    ):
        pos_std, yaw_std = initial_drift_std
        self._x = np.zeros(4, dtype=float)  # [dx, dy, dz, dyaw]
        self._P = np.diag([pos_std**2, pos_std**2, pos_std**2, yaw_std**2])

        proc_pos_std, proc_yaw_std = drift_process_std
        self._Q_per_second = np.diag(
            [
                proc_pos_std**2,
                proc_pos_std**2,
                proc_pos_std**2,
                proc_yaw_std**2,
            ]
        )

        self.default_gate_measurement_std = float(default_gate_measurement_std)
        self.association_radius = float(association_radius)
        self.commit_hits = int(commit_hits)
        self.commit_confidence_sum = float(commit_confidence_sum)
        self.min_confidence = float(min_confidence)
        self.uncommitted_alpha = float(uncommitted_alpha)
        self.committed_alpha = float(committed_alpha)
        self.max_update_distance = float(max_update_distance)

        self._odom: Optional[OdometrySample] = None
        self._next_landmark_id = 0
        self.landmarks: Dict[int, GateLandmark] = {}

    @property
    def drift_position(self) -> np.ndarray:
        return self._x[:3].copy()

    @property
    def drift_yaw(self) -> float:
        return float(self._x[3])

    @property
    def covariance(self) -> np.ndarray:
        return self._P.copy()

    def reset(self) -> None:
        self._x[:] = 0.0
        self._P[:] = np.diag(np.diag(self._P))
        self._odom = None
        self.landmarks.clear()
        self._next_landmark_id = 0

    def update_odometry(
        self,
        position,
        velocity,
        yaw: float,
        timestamp: float,
        covariance: Optional[np.ndarray] = None,
    ) -> EstimatedState:
        """
        Add the latest odometry/VIO/PX4 sample and propagate drift uncertainty.
        """
        sample = OdometrySample(
            position=_as_vec3(position, "position"),
            velocity=_as_vec3(velocity, "velocity"),
            yaw=float(yaw),
            timestamp=float(timestamp),
            covariance=None if covariance is None else np.asarray(covariance, dtype=float),
        )

        if self._odom is not None:
            dt = max(0.0, sample.timestamp - self._odom.timestamp)
            self._P = self._P + self._Q_per_second * dt

        self._odom = sample
        self._x[3] = _wrap_pi(self._x[3])
        return self.get_state()

    def predict_gate_world(self, position_body) -> np.ndarray:
        if self._odom is None:
            raise RuntimeError("Cannot project gate detection before odometry is available.")

        rel_body = _as_vec3(position_body, "position_body")
        yaw = self._odom.yaw + self._x[3]
        return self._odom.position + self._x[:3] + _yaw_to_rotmat(yaw) @ rel_body

    def update_gate_detection(
        self,
        position_body,
        timestamp: float,
        confidence: float = 1.0,
        landmark_id: Optional[int] = None,
        measurement_covariance: Optional[np.ndarray] = None,
    ) -> Optional[dict]:
        """
        Fuse a relative gate-center detection.

        Returns a small diagnostics dictionary, or None if the detection was
        rejected due to missing odometry or low confidence.
        """
        if self._odom is None:
            return None

        confidence = float(confidence)
        if confidence < self.min_confidence:
            return {
                "accepted": False,
                "reason": "low_confidence",
                "landmark_id": landmark_id,
            }

        detection = GateDetection(
            position_body=_as_vec3(position_body, "position_body"),
            timestamp=float(timestamp),
            confidence=confidence,
            landmark_id=landmark_id,
            measurement_covariance=(
                None
                if measurement_covariance is None
                else np.asarray(measurement_covariance, dtype=float)
            ),
        )

        predicted_center = self.predict_gate_world(detection.position_body)
        landmark = self._resolve_landmark(detection, predicted_center)

        if landmark is None:
            landmark = self._create_landmark(detection, predicted_center)
            return {
                "accepted": True,
                "reason": "new_landmark",
                "landmark_id": landmark.id,
                "committed": landmark.committed,
                "residual_norm": 0.0,
                "corrected_state": self.get_state(),
            }

        residual = landmark.center - predicted_center
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm > self.max_update_distance:
            return {
                "accepted": False,
                "reason": "residual_too_large",
                "landmark_id": landmark.id,
                "committed": landmark.committed,
                "residual_norm": residual_norm,
            }

        self._apply_drift_update(detection, residual)

        corrected_center = self.predict_gate_world(detection.position_body)
        self._update_landmark(landmark, detection, corrected_center)

        return {
            "accepted": True,
            "reason": "updated_landmark",
            "landmark_id": landmark.id,
            "committed": landmark.committed,
            "residual_norm": residual_norm,
            "corrected_state": self.get_state(),
            "landmark_center": landmark.center.copy(),
        }

    def update_gate_detection_obj(self, detection: GateDetection) -> Optional[dict]:
        return self.update_gate_detection(
            position_body=detection.position_body,
            timestamp=detection.timestamp,
            confidence=detection.confidence,
            landmark_id=detection.landmark_id,
            measurement_covariance=detection.measurement_covariance,
        )

    def get_state(self) -> EstimatedState:
        if self._odom is None:
            raise RuntimeError("No odometry has been provided yet.")

        return EstimatedState(
            position=self._odom.position + self._x[:3],
            velocity=self._odom.velocity.copy(),
            yaw=_wrap_pi(self._odom.yaw + self._x[3]),
            timestamp=self._odom.timestamp,
            odom_position=self._odom.position.copy(),
            odom_yaw=float(self._odom.yaw),
            drift_position=self._x[:3].copy(),
            drift_yaw=float(self._x[3]),
            covariance=self._P.copy(),
        )

    def get_committed_landmarks(self) -> Iterable[GateLandmark]:
        return tuple(
            sorted(
                (lm for lm in self.landmarks.values() if lm.committed),
                key=lambda lm: lm.id,
            )
        )

    def set_landmark(
        self,
        landmark_id: int,
        center,
        covariance: Optional[np.ndarray] = None,
        committed: bool = True,
    ) -> GateLandmark:
        """
        Seed or overwrite a gate landmark from a known/rough race map.

        This is the hook that lets the estimator correct odometry drift before
        the vehicle has built a map solely from its own biased odometry frame.
        """
        center = _as_vec3(center, "center")
        landmark_id = int(landmark_id)

        if covariance is None:
            covariance = np.eye(3, dtype=float)
        covariance = np.asarray(covariance, dtype=float).reshape(3, 3)

        landmark = GateLandmark(
            id=landmark_id,
            center=center.copy(),
            covariance=covariance,
            hits=self.commit_hits if committed else 1,
            confidence_sum=self.commit_confidence_sum if committed else 1.0,
            committed=bool(committed),
            last_seen_time=0.0,
        )
        self.landmarks[landmark_id] = landmark
        self._next_landmark_id = max(self._next_landmark_id, landmark_id + 1)
        return landmark

    def set_landmarks(self, landmarks: Dict[int, np.ndarray]) -> None:
        for landmark_id, center in landmarks.items():
            self.set_landmark(landmark_id, center)

    def _resolve_landmark(
        self,
        detection: GateDetection,
        predicted_center: np.ndarray,
    ) -> Optional[GateLandmark]:
        if detection.landmark_id is not None and detection.landmark_id in self.landmarks:
            return self.landmarks[detection.landmark_id]

        best = None
        best_dist = float("inf")
        for landmark in self.landmarks.values():
            dist = float(np.linalg.norm(predicted_center - landmark.center))
            if dist < self.association_radius and dist < best_dist:
                best = landmark
                best_dist = dist

        return best

    def _create_landmark(
        self,
        detection: GateDetection,
        center: np.ndarray,
    ) -> GateLandmark:
        landmark_id = (
            self._next_landmark_id
            if detection.landmark_id is None
            else int(detection.landmark_id)
        )
        self._next_landmark_id = max(self._next_landmark_id, landmark_id + 1)

        cov = detection.measurement_covariance
        if cov is None:
            var = self.default_gate_measurement_std**2 / max(detection.confidence, 1e-3)
            cov = np.eye(3, dtype=float) * var

        landmark = GateLandmark(
            id=landmark_id,
            center=center.copy(),
            covariance=np.asarray(cov, dtype=float).reshape(3, 3),
            hits=1,
            confidence_sum=detection.confidence,
            committed=False,
            last_seen_time=detection.timestamp,
        )
        self._maybe_commit(landmark)
        self.landmarks[landmark.id] = landmark
        return landmark

    def _update_landmark(
        self,
        landmark: GateLandmark,
        detection: GateDetection,
        corrected_center: np.ndarray,
    ) -> None:
        alpha = self.committed_alpha if landmark.committed else self.uncommitted_alpha
        err = float(np.linalg.norm(corrected_center - landmark.center))

        landmark.center = (1.0 - alpha) * landmark.center + alpha * corrected_center
        landmark.hits += 1
        landmark.confidence_sum += detection.confidence
        landmark.last_seen_time = detection.timestamp
        landmark.recent_errors.append(err)
        if len(landmark.recent_errors) > 30:
            landmark.recent_errors = landmark.recent_errors[-30:]

        self._maybe_commit(landmark)

    def _maybe_commit(self, landmark: GateLandmark) -> None:
        if landmark.committed:
            return
        if landmark.hits < self.commit_hits:
            return
        if landmark.confidence_sum < self.commit_confidence_sum:
            return
        landmark.committed = True

    def _measurement_covariance(self, detection: GateDetection) -> np.ndarray:
        if detection.measurement_covariance is not None:
            return np.asarray(detection.measurement_covariance, dtype=float).reshape(3, 3)

        confidence = max(detection.confidence, 1e-3)
        var = self.default_gate_measurement_std**2 / confidence
        return np.eye(3, dtype=float) * var

    def _apply_drift_update(self, detection: GateDetection, residual: np.ndarray) -> None:
        yaw = self._odom.yaw + self._x[3]
        rel_body = detection.position_body

        H = np.zeros((3, 4), dtype=float)
        H[:, :3] = np.eye(3)
        H[:, 3] = _dyaw_rotmat(yaw) @ rel_body

        R_meas = self._measurement_covariance(detection)
        S = H @ self._P @ H.T + R_meas
        K = self._P @ H.T @ np.linalg.inv(S)

        self._x = self._x + K @ residual
        self._x[3] = _wrap_pi(self._x[3])

        I = np.eye(4, dtype=float)
        # Joseph form is slightly more work, but keeps P symmetric/PSD better.
        KH = K @ H
        self._P = (I - KH) @ self._P @ (I - KH).T + K @ R_meas @ K.T
        self._P = 0.5 * (self._P + self._P.T)


class EstimatorLayer(DriftCorrectingGateEKF):
    """
    Compatibility alias for stack-level code.

    Use `update_odometry(...)` for VIO/PX4 pose updates and
    `update_gate_detection(...)` for body-frame gate observations.
    """
