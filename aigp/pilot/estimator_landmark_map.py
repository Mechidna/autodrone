from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class EstimatorLandmark:
    track_id: int
    position_neu: np.ndarray
    first_seen_time: float
    last_seen_time: float
    hits: int
    center_std_m: float
    camera_std_m: float
    reprojection_error: float
    source: str = "stable_gate_track"

    def as_snapshot(self) -> dict[str, object]:
        return {
            "track_id": int(self.track_id),
            "position_neu": self.position_neu.copy(),
            "first_seen_time": float(self.first_seen_time),
            "last_seen_time": float(self.last_seen_time),
            "hits": int(self.hits),
            "center_std_m": float(self.center_std_m),
            "camera_std_m": float(self.camera_std_m),
            "reprojection_error": float(self.reprojection_error),
            "source": self.source,
        }


class EstimatorLandmarkMap:
    """
    Competition-safe visual landmark map for estimator correction.

    The map is populated only from stable perception tracks. Once a track is
    admitted, its position is frozen so vehicle-state correction does not chase
    gate memory that was itself projected through the current vehicle estimate.
    """

    def __init__(
        self,
        *,
        min_hits: int,
        min_observation_time_s: float,
        max_center_std_m: float,
        max_camera_std_m: float,
        max_reprojection_error: float,
    ) -> None:
        self.min_hits = max(1, int(min_hits))
        self.min_observation_time_s = max(0.0, float(min_observation_time_s))
        self.max_center_std_m = max(0.0, float(max_center_std_m))
        self.max_camera_std_m = max(0.0, float(max_camera_std_m))
        self.max_reprojection_error = max(0.0, float(max_reprojection_error))
        self._landmarks: dict[int, EstimatorLandmark] = {}
        self.last_rejected: dict[int, str] = {}
        self.last_admitted_ids: list[int] = []

    def update_from_tracks(self, tracks: Iterable[object]) -> list[dict[str, object]]:
        self.last_rejected = {}
        self.last_admitted_ids = []

        for track in sorted(list(tracks or []), key=lambda item: int(getattr(item, "id", 0))):
            track_id = int(getattr(track, "id", -1))
            reason = self._admission_rejection_reason(track)
            if reason:
                self.last_rejected[track_id] = reason
                continue

            center = np.asarray(getattr(track, "center"), dtype=float).reshape(3)
            center_std = self._std_norm(getattr(track, "center_world_std", None))
            camera_std = self._std_norm(getattr(track, "center_camera_std", None))
            reproj = self._finite_float(
                getattr(track, "reprojection_error_median", math.nan),
                math.nan,
            )
            first_seen = self._finite_float(getattr(track, "first_seen_time", 0.0), 0.0)
            last_seen = self._finite_float(getattr(track, "last_seen_time", first_seen), first_seen)
            hits = int(getattr(track, "hits", 0))

            existing = self._landmarks.get(track_id)
            if existing is None:
                self._landmarks[track_id] = EstimatorLandmark(
                    track_id=track_id,
                    position_neu=center.copy(),
                    first_seen_time=first_seen,
                    last_seen_time=last_seen,
                    hits=hits,
                    center_std_m=center_std,
                    camera_std_m=camera_std,
                    reprojection_error=reproj,
                )
                self.last_admitted_ids.append(track_id)
            else:
                self._landmarks[track_id] = replace(
                    existing,
                    last_seen_time=max(existing.last_seen_time, last_seen),
                    hits=max(existing.hits, hits),
                    center_std_m=center_std,
                    camera_std_m=camera_std,
                    reprojection_error=reproj,
                )

        return [landmark.as_snapshot() for landmark in self.landmarks()]

    def landmarks(self) -> list[EstimatorLandmark]:
        return [
            self._landmarks[track_id]
            for track_id in sorted(self._landmarks)
        ]

    def _admission_rejection_reason(self, track: object) -> str:
        if not bool(getattr(track, "committed", False)):
            return "not_committed"
        if not bool(getattr(track, "is_stable", False)):
            return str(getattr(track, "promotion_blocked_reason", "")) or "not_stable"

        hits = int(getattr(track, "hits", 0))
        inlier_count = int(getattr(track, "inlier_count", hits))
        if hits < self.min_hits or inlier_count < self.min_hits:
            return "insufficient_hits"

        first_seen = self._finite_float(getattr(track, "first_seen_time", 0.0), 0.0)
        last_seen = self._finite_float(getattr(track, "last_seen_time", first_seen), first_seen)
        if last_seen - first_seen < self.min_observation_time_s:
            return "observation_span_short"

        try:
            center = np.asarray(getattr(track, "center"), dtype=float).reshape(3)
        except (TypeError, ValueError):
            return "invalid_center"
        if not np.all(np.isfinite(center)):
            return "invalid_center"

        center_std = self._std_norm(getattr(track, "center_world_std", None))
        if not math.isfinite(center_std) or center_std > self.max_center_std_m:
            return "center_std_high"

        camera_std = self._std_norm(getattr(track, "center_camera_std", None))
        if not math.isfinite(camera_std) or camera_std > self.max_camera_std_m:
            return "camera_std_high"

        reproj = self._finite_float(getattr(track, "reprojection_error_median", math.nan), math.nan)
        if math.isfinite(reproj) and reproj > self.max_reprojection_error:
            return "reprojection_error_high"

        return ""

    @staticmethod
    def _std_norm(value) -> float:
        if value is None:
            return math.inf
        try:
            arr = np.asarray(value, dtype=float).reshape(3)
        except (TypeError, ValueError):
            return math.inf
        if not np.all(np.isfinite(arr)):
            return math.inf
        return float(np.linalg.norm(arr))

    @staticmethod
    def _finite_float(value, default: float) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float(default)
        return out if math.isfinite(out) else float(default)
