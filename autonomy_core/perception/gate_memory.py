import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GateObservation:
    timestamp: float
    center_world: np.ndarray
    center_camera: Optional[np.ndarray] = None
    keypoints_px: Optional[np.ndarray] = None
    image_center_px: Optional[np.ndarray] = None
    image_size_px: Optional[np.ndarray] = None
    image_area_px2: float = float("nan")
    reprojection_error: float = float("nan")
    confidence: float = 0.0
    keypoint_conf_min: float = float("nan")
    keypoint_conf_mean: float = float("nan")
    quality_ok: bool = True
    quality_reason: str = ""
    solver_name: str = ""
    active_gate_idx: Optional[int] = None
    hit_index: int = 0
    is_outlier: bool = False
    hard_outlier: bool = False


@dataclass
class GateTrack:
    id: int
    center: np.ndarray
    confidence_sum: float = 0.0
    hits: int = 0
    first_seen_time: float = 0.0
    last_seen_time: float = 0.0
    committed: bool = False
    planning_center: Optional[np.ndarray] = None
    planning_locked: bool = False
    planning_lock_reason: str = ""
    last_image_center_px: Optional[np.ndarray] = None
    last_image_size_px: Optional[np.ndarray] = None
    last_image_seen_time: float = 0.0

    # Raw measurement history used for commit consistency checks / debugging
    measurement_history: List[np.ndarray] = field(default_factory=list)
    obs_history: List[GateObservation] = field(default_factory=list)

    # Residual history after commit: distance from new observation to locked center
    recent_errors: List[float] = field(default_factory=list)
    filtered_center_world: Optional[np.ndarray] = None
    filtered_center_camera: Optional[np.ndarray] = None
    center_world_std: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    center_camera_std: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    reprojection_error_mean: float = float("nan")
    reprojection_error_median: float = float("nan")
    image_area_px2_median: float = float("nan")
    required_hits_for_stable: int = 0
    is_stable: bool = False
    stability_score: float = 0.0
    promotion_blocked_reason: str = "insufficient_observations"
    outlier_count: int = 0
    inlier_count: int = 0
    ever_stable: bool = False
    visible_miss_count: int = 0
    visible_miss_start_time: float = 0.0
    visible_miss_time_s: float = 0.0
    last_expected_visible_time: float = 0.0
    visibility_state: str = "unknown"
    negative_evidence_score: float = 0.0
    commit_blocked_reason: str = ""
    known_position_match_index: Optional[int] = None
    known_position_distance_m: float = float("nan")

    def append_observation(
        self,
        measurement: np.ndarray,
        confidence: float,
        timestamp: float,
        center_camera: Optional[np.ndarray] = None,
        keypoints_px: Optional[np.ndarray] = None,
        reprojection_error: float = float("nan"),
        keypoint_conf_min: float = float("nan"),
        keypoint_conf_mean: float = float("nan"),
        quality_ok: bool = True,
        quality_reason: str = "",
        solver_name: str = "",
        active_gate_idx: Optional[int] = None,
        history_size: int = 15,
        is_outlier: bool = False,
    ):
        measurement = np.asarray(measurement, dtype=float).reshape(3)
        camera = None
        if center_camera is not None:
            camera = np.asarray(center_camera, dtype=float).reshape(3)
        image_keypoints = None
        image_center = None
        image_size = None
        image_area = float("nan")
        if keypoints_px is not None:
            try:
                candidate = np.asarray(keypoints_px, dtype=float).reshape(-1, 2)
            except (TypeError, ValueError):
                candidate = np.empty((0, 2), dtype=float)
            if candidate.shape[0] >= 4 and np.all(np.isfinite(candidate)):
                image_keypoints = candidate.copy()
                image_center = np.mean(candidate, axis=0)
                image_size = np.ptp(candidate, axis=0)
                image_area = 0.5 * abs(
                    float(
                        np.dot(candidate[:, 0], np.roll(candidate[:, 1], -1))
                        - np.dot(candidate[:, 1], np.roll(candidate[:, 0], -1))
                    )
                )

        self.confidence_sum += float(confidence)
        self.hits += 1
        self.visible_miss_count = 0
        self.visible_miss_start_time = 0.0
        self.visible_miss_time_s = 0.0
        self.last_expected_visible_time = 0.0
        self.visibility_state = "visible_hit"
        self.negative_evidence_score = 0.0
        if self.first_seen_time <= 0.0:
            self.first_seen_time = float(timestamp)
        self.last_seen_time = float(timestamp)
        if image_center is not None and image_size is not None:
            self.last_image_center_px = image_center.copy()
            self.last_image_size_px = image_size.copy()
            self.last_image_seen_time = float(timestamp)
        self.measurement_history.append(measurement.copy())
        obs = GateObservation(
            timestamp=float(timestamp),
            center_world=measurement.copy(),
            center_camera=None if camera is None else camera.copy(),
            keypoints_px=None if image_keypoints is None else image_keypoints.copy(),
            image_center_px=None if image_center is None else image_center.copy(),
            image_size_px=None if image_size is None else image_size.copy(),
            image_area_px2=float(image_area),
            reprojection_error=float(reprojection_error),
            confidence=float(confidence),
            keypoint_conf_min=float(keypoint_conf_min),
            keypoint_conf_mean=float(keypoint_conf_mean),
            quality_ok=bool(quality_ok),
            quality_reason=str(quality_reason or ""),
            solver_name=str(solver_name or ""),
            active_gate_idx=active_gate_idx,
            hit_index=int(self.hits),
            is_outlier=bool(is_outlier),
            hard_outlier=bool(is_outlier),
        )
        self.obs_history.append(obs)
        if len(self.obs_history) > history_size:
            self.obs_history = self.obs_history[-history_size:]
        if len(self.measurement_history) > max(history_size, 30):
            self.measurement_history = self.measurement_history[-max(history_size, 30):]

    def update(
        self,
        measurement: np.ndarray,
        confidence: float,
        timestamp: float,
        alpha: float = 0.35,
        center_camera: Optional[np.ndarray] = None,
        keypoints_px: Optional[np.ndarray] = None,
        reprojection_error: float = float("nan"),
        keypoint_conf_min: float = float("nan"),
        keypoint_conf_mean: float = float("nan"),
        quality_ok: bool = True,
        quality_reason: str = "",
        solver_name: str = "",
        active_gate_idx: Optional[int] = None,
        history_size: int = 15,
    ):
        measurement = np.asarray(measurement, dtype=float).reshape(3)
        self.append_observation(
            measurement=measurement,
            confidence=confidence,
            timestamp=timestamp,
            center_camera=center_camera,
            keypoints_px=keypoints_px,
            reprojection_error=reprojection_error,
            keypoint_conf_min=keypoint_conf_min,
            keypoint_conf_mean=keypoint_conf_mean,
            quality_ok=quality_ok,
            quality_reason=quality_reason,
            solver_name=solver_name,
            active_gate_idx=active_gate_idx,
            history_size=history_size,
        )
        # Robust center for uncommitted tracks: do not let one accepted
        # measurement pull the candidate center. Boundary/outlier rejection
        # happens before this; the median handles residual jitter.
        recent = np.asarray(self.recent_measurements(10), dtype=float)
        self.center = np.median(recent, axis=0)

    def observe_without_moving(
        self,
        measurement: np.ndarray,
        confidence: float,
        timestamp: float,
        center_camera: Optional[np.ndarray] = None,
        keypoints_px: Optional[np.ndarray] = None,
        reprojection_error: float = float("nan"),
        keypoint_conf_min: float = float("nan"),
        keypoint_conf_mean: float = float("nan"),
        quality_ok: bool = True,
        quality_reason: str = "",
        solver_name: str = "",
        active_gate_idx: Optional[int] = None,
        history_size: int = 15,
        is_outlier: bool = False,
    ):
        """
        Record that this committed landmark was seen again, but do not move its center.
        """
        measurement = np.asarray(measurement, dtype=float).reshape(3)

        self.append_observation(
            measurement=measurement,
            confidence=confidence,
            timestamp=timestamp,
            center_camera=center_camera,
            keypoints_px=keypoints_px,
            reprojection_error=reprojection_error,
            keypoint_conf_min=keypoint_conf_min,
            keypoint_conf_mean=keypoint_conf_mean,
            quality_ok=quality_ok,
            quality_reason=quality_reason,
            solver_name=solver_name,
            active_gate_idx=active_gate_idx,
            history_size=history_size,
            is_outlier=is_outlier,
        )

        reference = (
            self.planning_center
            if self.planning_center is not None
            else self.center
        )
        err = float(np.linalg.norm(measurement - reference))
        self.recent_errors.append(err)
        if is_outlier:
            self.outlier_count += 1

        # keep this bounded
        if len(self.recent_errors) > 30:
            self.recent_errors = self.recent_errors[-30:]

    def recent_measurements(self, n: int = 10) -> List[np.ndarray]:
        if n <= 0:
            return []
        return self.measurement_history[-n:]


class GateMemory:
    """
    Stores candidate and committed gate centers as persistent world landmarks.

    Key policy:
    - Association prefers COMMITTED tracks first, then candidate tracks
    - New tracks are blocked near committed landmarks
    - Candidates commit only after repeated + spatially consistent observations
    - Committed landmarks do not move anymore
    """

    def __init__(
        self,
        association_radius: float = 2.0,
        commit_radius: float = 2.0,
        new_track_block_radius: float = 3.5,
        min_confidence_per_hit: float = 0.5,
        commit_hits: int = 3,
        commit_confidence_sum: float = 2.0,
        commit_spread_radius: float = 1.0,
        stale_time: float = 5.0,
        alpha: float = 0.35,
        use_lookahead_gate_filter: bool = True,
        history_size: int = 15,
        min_hits_for_stable: int = 5,
        max_center_std_for_stable: float = 0.35,
        max_camera_std_for_stable: float = 0.35,
        max_reprojection_error_for_stable: float = 5.0,
        min_keypoint_conf_for_stable: float = 0.0,
        max_outlier_distance: float = 0.75,
        min_observation_time: float = 0.25,
        temporal_image_association_enabled: bool = True,
        temporal_image_association_max_age_s: float = 1.0,
        temporal_image_association_max_center_distance_px: float = 2.5,
        temporal_image_association_max_size_ratio: float = 1.5,
        temporal_image_association_min_depth_m: float = 20.0,
        tiny_detection_area_px2: float = 50.0,
        tiny_detection_min_hits_for_stable: int = 6,
        tiny_detection_min_observation_time: float = 0.5,
        known_position_commit_filter_enabled: bool = False,
        known_position_commit_radius_m: float = 5.0,
        known_gate_positions_neu=None,
    ):
        self.association_radius = float(association_radius)
        self.commit_radius = float(commit_radius)
        self.new_track_block_radius = float(new_track_block_radius)
        self.min_confidence_per_hit = float(min_confidence_per_hit)
        self.commit_hits = int(commit_hits)
        self.commit_confidence_sum = float(commit_confidence_sum)
        self.commit_spread_radius = float(commit_spread_radius)
        self.stale_time = float(stale_time)
        self.alpha = float(alpha)
        self.use_lookahead_gate_filter = bool(use_lookahead_gate_filter)
        self.history_size = int(history_size)
        self.min_hits_for_stable = int(min_hits_for_stable)
        self.max_center_std_for_stable = float(max_center_std_for_stable)
        self.max_camera_std_for_stable = float(max_camera_std_for_stable)
        self.max_reprojection_error_for_stable = float(max_reprojection_error_for_stable)
        self.min_keypoint_conf_for_stable = float(min_keypoint_conf_for_stable)
        self.max_outlier_distance = float(max_outlier_distance)
        self.min_observation_time = float(min_observation_time)
        self.temporal_image_association_enabled = bool(
            temporal_image_association_enabled
        )
        self.temporal_image_association_max_age_s = max(
            0.0,
            float(temporal_image_association_max_age_s),
        )
        self.temporal_image_association_max_center_distance_px = max(
            0.0,
            float(temporal_image_association_max_center_distance_px),
        )
        self.temporal_image_association_max_size_ratio = max(
            1.0,
            float(temporal_image_association_max_size_ratio),
        )
        self.temporal_image_association_min_depth_m = max(
            0.0,
            float(temporal_image_association_min_depth_m),
        )
        self.tiny_detection_area_px2 = max(
            0.0,
            float(tiny_detection_area_px2),
        )
        self.tiny_detection_min_hits_for_stable = max(
            1,
            int(tiny_detection_min_hits_for_stable),
        )
        self.tiny_detection_min_observation_time = max(
            0.0,
            float(tiny_detection_min_observation_time),
        )
        self.known_position_commit_filter_enabled = bool(
            known_position_commit_filter_enabled
        )
        self.known_position_commit_radius_m = max(
            0.0,
            float(known_position_commit_radius_m),
        )
        known_positions = (
            ()
            if known_gate_positions_neu is None
            else known_gate_positions_neu
        )
        self.known_gate_positions_neu = np.asarray(
            known_positions,
            dtype=float,
        ).reshape(-1, 3)
        self.max_committed_match_distance = 0.8  # meters (start here)
        # Approximate physical gate aperture is about 2 m in the simulator's
        # perception model. Split/offset PnP landmarks for the same visual gate
        # can land on opposite sides of that aperture, so duplicate suppression
        # must be wider than raw association while still geometry-only.
        self.estimated_gate_size = 2.0
        self.duplicate_merge_radius = max(5.0, 2.5 * self.estimated_gate_size)
        self.last_merge_event = {
            "merged": False,
            "source_id": None,
            "target_id": None,
            "reason": "",
            "distance": float("nan"),
        }
        self.last_pairwise_distances = []
        self.last_merge_candidate_pairs = []
        self.last_merge_blocked_reason = ""
        self.max_candidate_update_innovation = max(2.0, self.commit_radius)
        self.last_update_innovation = float("nan")
        self.last_update_accepted = False
        self.last_track_center_before = None
        self.last_track_center_after = None
        self.last_track_stable_now = False
        self.last_track_state_change = ""
        self.last_outlier_rejected = False
        self.nearest_track_id = None
        self.nearest_track_distance = float("nan")
        self.nearest_track_hits = 0
        self.nearest_track_committed = False
        self.nearest_track_stable = False
        self.association_attempted = False
        self.association_success = False
        self.duplicate_rejection_reason = ""
        self.last_association_mode = ""
        self.last_image_association_center_distance_px = float("nan")
        self.last_image_association_size_ratio = float("nan")
        self.last_image_association_age_s = float("nan")

        self._next_id = 0
        self.tracks: List[GateTrack] = []

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    @staticmethod
    def _image_signature(
        keypoints_px: Optional[np.ndarray],
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if keypoints_px is None:
            return None, None
        try:
            points = np.asarray(keypoints_px, dtype=float).reshape(-1, 2)
        except (TypeError, ValueError):
            return None, None
        if points.shape[0] < 4 or not np.all(np.isfinite(points)):
            return None, None
        center = np.mean(points, axis=0)
        size = np.ptp(points, axis=0)
        if np.any(size <= 0.0):
            return None, None
        return center, size

    @staticmethod
    def _image_size_ratio(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=float).reshape(2)
        b = np.asarray(b, dtype=float).reshape(2)
        if np.any(a <= 0.0) or np.any(b <= 0.0):
            return float("inf")
        ratios = np.maximum(a / b, b / a)
        return float(np.max(ratios))

    def _find_recent_image_track(
        self,
        *,
        keypoints_px: Optional[np.ndarray],
        center_camera: Optional[np.ndarray],
        timestamp: float,
        committed_only: bool = False,
        uncommitted_only: bool = False,
    ) -> tuple[Optional[GateTrack], float, float, float]:
        if not self.temporal_image_association_enabled:
            return None, float("nan"), float("nan"), float("nan")

        image_center, image_size = self._image_signature(keypoints_px)
        if image_center is None or image_size is None:
            return None, float("nan"), float("nan"), float("nan")

        if center_camera is None:
            return None, float("nan"), float("nan"), float("nan")
        try:
            camera = np.asarray(center_camera, dtype=float).reshape(3)
        except (TypeError, ValueError):
            return None, float("nan"), float("nan"), float("nan")
        if not np.all(np.isfinite(camera)):
            return None, float("nan"), float("nan"), float("nan")
        if float(camera[2]) < self.temporal_image_association_min_depth_m:
            return None, float("nan"), float("nan"), float("nan")

        best = None
        best_score = float("inf")
        best_center_distance = float("nan")
        best_size_ratio = float("nan")
        best_age = float("nan")
        for tr in self.tracks:
            if committed_only and not tr.committed:
                continue
            if uncommitted_only and tr.committed:
                continue
            if tr.last_image_center_px is None or tr.last_image_size_px is None:
                continue

            age = float(timestamp) - float(tr.last_image_seen_time)
            if age < -1e-6 or age > self.temporal_image_association_max_age_s:
                continue
            center_distance = float(
                np.linalg.norm(image_center - tr.last_image_center_px)
            )
            if (
                center_distance
                > self.temporal_image_association_max_center_distance_px
            ):
                continue
            size_ratio = self._image_size_ratio(image_size, tr.last_image_size_px)
            if size_ratio > self.temporal_image_association_max_size_ratio:
                continue

            score = center_distance + max(0.0, size_ratio - 1.0)
            if score < best_score:
                best = tr
                best_score = score
                best_center_distance = center_distance
                best_size_ratio = size_ratio
                best_age = age

        return best, best_center_distance, best_size_ratio, best_age

    def _record_image_association_debug(
        self,
        *,
        mode: str,
        center_distance_px: float,
        size_ratio: float,
        age_s: float,
    ) -> None:
        self.last_association_mode = str(mode)
        self.last_image_association_center_distance_px = float(center_distance_px)
        self.last_image_association_size_ratio = float(size_ratio)
        self.last_image_association_age_s = float(age_s)

    def _find_best_track(self, center: np.ndarray, committed_only: bool = False, uncommitted_only: bool = False) -> Optional[GateTrack]:
        best = None
        best_dist = float("inf")

        for tr in self.tracks:
            if committed_only and not tr.committed:
                continue
            if uncommitted_only and tr.committed:
                continue

            d = self._distance(center, tr.center)
            if d < self.association_radius and d < best_dist:
                best = tr
                best_dist = d

        return best

    def _find_nearest_track(self, center: np.ndarray, radius: Optional[float] = None) -> Optional[GateTrack]:
        best = None
        best_dist = float("inf")
        max_radius = float("inf") if radius is None else float(radius)
        for tr in self.tracks:
            d = self._distance(center, tr.center)
            if d < max_radius and d < best_dist:
                best = tr
                best_dist = d
        return best

    def _record_nearest_track_debug(self, center: np.ndarray):
        nearest = self._find_nearest_track(
            center,
            radius=max(self.association_radius, self.new_track_block_radius),
        )
        self.nearest_track_id = None
        self.nearest_track_distance = float("nan")
        self.nearest_track_hits = 0
        self.nearest_track_committed = False
        self.nearest_track_stable = False
        if nearest is not None:
            self.nearest_track_id = nearest.id
            self.nearest_track_distance = self._distance(center, nearest.center)
            self.nearest_track_hits = int(nearest.hits)
            self.nearest_track_committed = bool(nearest.committed)
            self.nearest_track_stable = bool(nearest.is_stable)
        return nearest

    def _near_any_committed(self, center: np.ndarray, radius: Optional[float] = None) -> bool:
        r = self.new_track_block_radius if radius is None else float(radius)
        for tr in self.tracks:
            if not tr.committed:
                continue
            if self._distance(center, tr.center) < r:
                return True
        return False

    def _create_track(
        self,
        center: np.ndarray,
        confidence: float,
        timestamp: float,
        center_camera: Optional[np.ndarray] = None,
        keypoints_px: Optional[np.ndarray] = None,
        reprojection_error: float = float("nan"),
        keypoint_conf_min: float = float("nan"),
        keypoint_conf_mean: float = float("nan"),
        quality_ok: bool = True,
        quality_reason: str = "",
        solver_name: str = "",
        active_gate_idx: Optional[int] = None,
    ) -> GateTrack:
        center = np.asarray(center, dtype=float).reshape(3)

        tr = GateTrack(
            id=self._next_id,
            center=center.copy(),
            confidence_sum=float(confidence),
            hits=0,
            first_seen_time=float(timestamp),
            last_seen_time=float(timestamp),
            committed=False,
            measurement_history=[],
            recent_errors=[],
        )
        tr.append_observation(
            measurement=center,
            confidence=confidence,
            timestamp=timestamp,
            center_camera=center_camera,
            keypoints_px=keypoints_px,
            reprojection_error=reprojection_error,
            keypoint_conf_min=keypoint_conf_min,
            keypoint_conf_mean=keypoint_conf_mean,
            quality_ok=quality_ok,
            quality_reason=quality_reason,
            solver_name=solver_name,
            active_gate_idx=active_gate_idx,
            history_size=self.history_size,
        )
        self._next_id += 1
        self.tracks.append(tr)
        self._update_track_filter(tr)
        return tr

    def _candidate_spread(self, tr: GateTrack, n_recent: int = 10) -> float:
        """
        Maximum distance of recent accepted inlier measurements from the
        candidate center. Temporal image association may retain alternative
        PnP depth hypotheses as outliers; those must not make a dominant,
        consistent depth cluster impossible to commit.
        """
        observations = [
            obs
            for obs in tr.obs_history[-n_recent:]
            if bool(obs.quality_ok) and not bool(obs.is_outlier)
        ]
        measurements = [obs.center_world for obs in observations]
        if len(measurements) == 0:
            return float("inf")

        center = tr.center
        dists = [self._distance(m, center) for m in measurements]
        return float(max(dists)) if len(dists) > 0 else float("inf")

    def _maybe_commit(self, tr: GateTrack):
        if tr.committed:
            return

        quality_obs = [
            obs for obs in tr.obs_history if bool(obs.quality_ok) and not bool(obs.is_outlier)
        ]
        if len(quality_obs) < self.commit_hits:
            return

        if tr.hits < self.commit_hits:
            return

        quality_confidence_sum = float(
            sum(float(obs.confidence) for obs in quality_obs)
        )
        if quality_confidence_sum < self.commit_confidence_sum:
            return

        spread = self._candidate_spread(tr, n_recent=10)
        if spread > self.commit_spread_radius:
            return

        if self.known_position_commit_filter_enabled:
            if self.known_gate_positions_neu.shape[0] == 0:
                tr.commit_blocked_reason = "known_position_filter_no_references"
                tr.known_position_match_index = None
                tr.known_position_distance_m = float("inf")
                return

            distances = np.linalg.norm(
                self.known_gate_positions_neu - tr.center.reshape(1, 3),
                axis=1,
            )
            nearest_index = int(np.argmin(distances))
            nearest_distance = float(distances[nearest_index])
            tr.known_position_match_index = nearest_index
            tr.known_position_distance_m = nearest_distance
            if nearest_distance > self.known_position_commit_radius_m:
                tr.commit_blocked_reason = "outside_known_position_commit_radius"
                return

        # avoid committing duplicates near an already committed landmark
        for other in self.tracks:
            if other.id == tr.id:
                continue
            if other.committed and self._distance(tr.center, other.center) < self.commit_radius:
                return

        tr.commit_blocked_reason = ""
        tr.committed = True
        tr.planning_center = tr.center.copy()
        tr.center = tr.planning_center.copy()
        if not self.use_lookahead_gate_filter:
            tr.is_stable = True
            tr.filtered_center_world = tr.center.copy()
            tr.promotion_blocked_reason = ""

    def lock_track_for_planning(
        self,
        track_id: int,
        reason: str = "prearm_gate_acquisition",
    ) -> bool:
        tr = self.get_track_by_id(int(track_id))
        if tr is None or not tr.committed:
            return False
        if not (tr.is_stable or tr.ever_stable):
            return False
        if tr.planning_locked:
            return True

        candidates = (
            tr.filtered_center_world,
            tr.planning_center,
            tr.center,
        )
        locked_center = None
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                arr = np.asarray(candidate, dtype=float).reshape(3)
            except (TypeError, ValueError):
                continue
            if np.all(np.isfinite(arr)):
                locked_center = arr.copy()
                break
        if locked_center is None:
            return False

        tr.planning_center = locked_center.copy()
        tr.center = locked_center.copy()
        tr.planning_locked = True
        tr.planning_lock_reason = str(reason or "planning_locked")
        return True

    def _track_uncertainty(self, tr: GateTrack) -> float:
        measurements = tr.recent_measurements(20)
        if len(measurements) <= 1:
            return float("inf")
        dists = [self._distance(m, tr.center) for m in measurements]
        return float(np.sqrt(np.mean(np.square(dists))))

    def _finite_errors(self, observations: List[GateObservation]) -> np.ndarray:
        errors = [obs.reprojection_error for obs in observations if np.isfinite(obs.reprojection_error)]
        return np.asarray(errors, dtype=float)

    def _update_track_filter(self, tr: GateTrack):
        was_stable = bool(tr.is_stable)
        if tr.committed and tr.planning_center is None:
            tr.planning_center = tr.center.copy()
        observations = list(tr.obs_history[-self.history_size:])
        if len(observations) == 0:
            tr.promotion_blocked_reason = "no_observations"
            return

        quality_observations = [obs for obs in observations if bool(obs.quality_ok)]
        if len(quality_observations) == 0:
            tr.inlier_count = 0
            tr.outlier_count = sum(1 for obs in tr.obs_history if obs.is_outlier)
            tr.filtered_center_world = tr.center.copy()
            tr.filtered_center_camera = None
            tr.center_world_std = np.full(3, np.nan)
            tr.center_camera_std = np.full(3, np.nan)
            tr.reprojection_error_mean = float("nan")
            tr.reprojection_error_median = float("nan")
            tr.promotion_blocked_reason = "no_quality_observations"
            tr.is_stable = False
            tr.stability_score = 0.0
            return

        if tr.committed:
            filter_observations = [
                obs for obs in quality_observations if not bool(obs.is_outlier)
            ]
        else:
            # Candidate depth modes are allowed to compete over time. Robust
            # outliers are therefore recomputed for uncommitted tracks instead
            # of permanently pinning the candidate to its first PnP solution.
            filter_observations = [
                obs for obs in quality_observations if not bool(obs.hard_outlier)
            ]
            for obs in filter_observations:
                obs.is_outlier = False
        if len(filter_observations) == 0:
            filter_observations = quality_observations

        world = np.asarray([obs.center_world for obs in filter_observations], dtype=float)
        median_world = np.median(world, axis=0)
        dists = np.linalg.norm(world - median_world, axis=1)
        inlier_mask = dists <= self.max_outlier_distance
        inlier_obs = [obs for obs, keep in zip(filter_observations, inlier_mask) if keep]
        outliers = [obs for obs, keep in zip(filter_observations, inlier_mask) if not keep]
        for obs, keep in zip(filter_observations, inlier_mask):
            obs.is_outlier = bool(obs.hard_outlier or not keep)

        tr.inlier_count = len(inlier_obs)
        tr.outlier_count = sum(1 for obs in tr.obs_history if obs.is_outlier)

        if len(inlier_obs) > 0:
            inlier_world = np.asarray([obs.center_world for obs in inlier_obs], dtype=float)
            filtered_world = np.mean(inlier_world, axis=0)
            tr.filtered_center_world = filtered_world.copy()
            tr.center_world_std = np.std(inlier_world, axis=0) if len(inlier_world) > 1 else np.zeros(3)
            if not tr.committed and not tr.is_stable:
                tr.center = filtered_world.copy()
        else:
            tr.filtered_center_world = tr.center.copy()
            tr.center_world_std = np.full(3, np.nan)

        camera_obs = [obs.center_camera for obs in inlier_obs if obs.center_camera is not None]
        if len(camera_obs) > 0:
            camera = np.asarray(camera_obs, dtype=float)
            tr.filtered_center_camera = np.mean(camera, axis=0)
            tr.center_camera_std = np.std(camera, axis=0) if len(camera) > 1 else np.zeros(3)
        else:
            tr.filtered_center_camera = None
            tr.center_camera_std = np.full(3, np.nan)

        errors = self._finite_errors(inlier_obs)
        if errors.size > 0:
            tr.reprojection_error_mean = float(np.mean(errors))
            tr.reprojection_error_median = float(np.median(errors))
        else:
            tr.reprojection_error_mean = float("nan")
            tr.reprojection_error_median = float("nan")

        span_observations = inlier_obs if len(inlier_obs) > 0 else quality_observations
        span = float(span_observations[-1].timestamp - span_observations[0].timestamp)
        image_areas = np.asarray(
            [
                float(obs.image_area_px2)
                for obs in inlier_obs
                if np.isfinite(obs.image_area_px2) and obs.image_area_px2 > 0.0
            ],
            dtype=float,
        )
        tr.image_area_px2_median = (
            float(np.median(image_areas))
            if image_areas.size > 0
            else float("nan")
        )
        required_hits_for_stable = int(self.min_hits_for_stable)
        required_observation_time = float(self.min_observation_time)
        if (
            self.tiny_detection_area_px2 > 0.0
            and np.isfinite(tr.image_area_px2_median)
            and tr.image_area_px2_median <= self.tiny_detection_area_px2
        ):
            required_hits_for_stable = max(
                required_hits_for_stable,
                int(self.tiny_detection_min_hits_for_stable),
            )
            required_observation_time = max(
                required_observation_time,
                float(self.tiny_detection_min_observation_time),
            )
        tr.required_hits_for_stable = required_hits_for_stable
        world_std_norm = float(np.linalg.norm(tr.center_world_std)) if np.all(np.isfinite(tr.center_world_std)) else float("inf")
        reproj = tr.reprojection_error_median
        reproj_ok = not np.isfinite(reproj) or reproj <= self.max_reprojection_error_for_stable
        min_kp_conf = float("nan")
        stable_obs = inlier_obs[-max(1, required_hits_for_stable):]
        kp_conf = [
            float(obs.keypoint_conf_min)
            for obs in stable_obs
            if np.isfinite(obs.keypoint_conf_min)
        ]
        if kp_conf:
            min_kp_conf = float(min(kp_conf))
        kp_conf_ok = (
            self.min_keypoint_conf_for_stable <= 0.0
            or (
                len(kp_conf) >= min(required_hits_for_stable, len(stable_obs))
                and min_kp_conf >= self.min_keypoint_conf_for_stable
            )
        )

        reason = ""
        if tr.hits < required_hits_for_stable:
            reason = "insufficient_hits"
        elif tr.inlier_count < required_hits_for_stable:
            reason = "insufficient_inliers"
        elif not kp_conf_ok:
            reason = "keypoint_conf_low"
        elif world_std_norm > self.max_center_std_for_stable:
            reason = "center_std_high"
        elif not reproj_ok:
            reason = "reprojection_error_high"
        elif span < required_observation_time:
            reason = "observation_span_short"

        tr.promotion_blocked_reason = reason
        if reason:
            tr.is_stable = False
            tr.stability_score = 0.0
        else:
            tr.is_stable = True
            if not tr.committed:
                tr.center = tr.filtered_center_world.copy()
            elif tr.planning_center is None:
                tr.planning_center = tr.center.copy()
            std_score = 1.0 - min(world_std_norm / max(self.max_center_std_for_stable, 1e-6), 1.0)
            reproj_score = 1.0
            if np.isfinite(reproj):
                reproj_score = 1.0 - min(reproj / max(self.max_reprojection_error_for_stable, 1e-6), 1.0)
            hit_score = min(
                tr.hits / max(float(required_hits_for_stable), 1.0),
                2.0,
            ) / 2.0
            tr.stability_score = float(0.45 * std_score + 0.35 * reproj_score + 0.20 * hit_score)

        if tr.committed and tr.planning_center is not None:
            tr.center = tr.planning_center.copy()

        tr.ever_stable = bool(tr.ever_stable or tr.is_stable)
        if tr.is_stable and not was_stable:
            self.last_track_stable_now = True
            self.last_track_state_change = "stable"
            print(
                f"TRACK {tr.id} stable: hits={tr.hits} std={world_std_norm:.2f} "
                f"reproj={tr.reprojection_error_median:.2f} "
                f"kp={min_kp_conf:.2f} area={tr.image_area_px2_median:.1f} "
                f"required_hits={required_hits_for_stable} "
                f"score={tr.stability_score:.2f}"
            )
        elif not tr.is_stable and tr.promotion_blocked_reason:
            print(
                f"TRACK {tr.id} tentative: hits={tr.hits} std={world_std_norm:.2f} "
                f"kp={min_kp_conf:.2f} area={tr.image_area_px2_median:.1f} "
                f"required_hits={required_hits_for_stable} "
                f"blocked={tr.promotion_blocked_reason}"
            )

    def merge_track_into(self, source_id: int, target_id: int, reason: str = "duplicate_committed_track"):
        if source_id == target_id:
            return None

        source = self.get_track_by_id(source_id)
        target = self.get_track_by_id(target_id)
        if source is None or target is None:
            return None
        if source.planning_locked or target.planning_locked:
            return None

        source_center = (
            source.planning_center
            if source.planning_center is not None
            else source.center
        ).copy()
        target_center = (
            target.planning_center
            if target.planning_center is not None
            else target.center
        ).copy()
        source_weight = max(float(source.hits), 1.0)
        target_weight = max(float(target.hits), 1.0)
        merged_center = (
            target_center * target_weight + source_center * source_weight
        ) / (target_weight + source_weight)
        target.center = merged_center.copy()
        target.planning_center = merged_center.copy()
        target.confidence_sum += source.confidence_sum
        target.hits += source.hits
        target.last_seen_time = max(target.last_seen_time, source.last_seen_time)
        target.committed = target.committed or source.committed
        target.ever_stable = bool(target.ever_stable or source.ever_stable)
        target.measurement_history.extend(source.measurement_history)
        target.obs_history.extend(source.obs_history)
        target.obs_history = target.obs_history[-self.history_size:]
        target.recent_errors.extend(source.recent_errors)
        if len(target.recent_errors) > 30:
            target.recent_errors = target.recent_errors[-30:]
        self._update_track_filter(target)

        self.tracks = [tr for tr in self.tracks if tr.id != source_id]
        self.last_merge_event = {
            "merged": True,
            "source_id": int(source_id),
            "target_id": int(target_id),
            "reason": reason,
            "distance": self._distance(source_center, target_center),
        }
        return self.last_merge_event.copy()

    def merge_duplicate_committed_tracks(self, radius: Optional[float] = None):
        radius = self.duplicate_merge_radius if radius is None else float(radius)
        self.last_merge_event = {
            "merged": False,
            "source_id": None,
            "target_id": None,
            "reason": "",
            "distance": float("nan"),
        }
        self.last_pairwise_distances = []
        self.last_merge_candidate_pairs = []
        self.last_merge_blocked_reason = "fewer_than_two_committed_tracks"

        committed = self.get_committed_tracks()
        for i, a in enumerate(committed):
            for b in committed[i + 1:]:
                dist = self._distance(a.center, b.center)
                self.last_pairwise_distances.append((a.id, b.id, dist))
                if dist >= radius:
                    self.last_merge_blocked_reason = f"all_pairs_above_radius:{radius:.2f}"
                    continue
                self.last_merge_candidate_pairs.append((a.id, b.id, dist))
                self.last_merge_blocked_reason = ""

                if (a.hits, -a.id) >= (b.hits, -b.id):
                    target, source = a, b
                else:
                    target, source = b, a

                return self.merge_track_into(
                    source_id=source.id,
                    target_id=target.id,
                    reason=f"committed_centers_within:{dist:.2f}",
                )

        return self.last_merge_event.copy()

    def track_uncertainty(self, track_id: int) -> float:
        tr = self.get_track_by_id(track_id)
        if tr is None:
            return float("nan")
        return self._track_uncertainty(tr)

    def prune(self, now: float):
        kept = []
        for tr in self.tracks:
            age = now - tr.last_seen_time
            if tr.committed:
                # committed landmarks persist
                kept.append(tr)
            else:
                if age <= self.stale_time:
                    kept.append(tr)
        self.tracks = kept

    def add_detection(
        self,
        center: np.ndarray,
        confidence: float,
        timestamp: float,
        center_camera: Optional[np.ndarray] = None,
        keypoints_px: Optional[np.ndarray] = None,
        reprojection_error: float = float("nan"),
        keypoint_conf_min: float = float("nan"),
        keypoint_conf_mean: float = float("nan"),
        solver_name: str = "",
        active_gate_idx: Optional[int] = None,
        quality_ok: bool = True,
        quality_reason: str = "",
        allow_new_track: bool = True,
        allow_candidate_commit: bool = True,
        admission_reason: str = "",
    ):
        """
        Add a new world-frame gate center measurement.

        Association order:
        1) nearest COMMITTED track
        2) recent image-space match to a COMMITTED track
        3) nearest UNCOMMITTED track
        4) recent image-space match to an UNCOMMITTED track
        5) if near committed landmark, reject creating a new track
        6) honor an optional temporary new-track admission freeze
        7) otherwise create a new candidate track
        """
        center = np.asarray(center, dtype=float).reshape(3)
        confidence = float(confidence)
        self.last_update_innovation = float("nan")
        self.last_update_accepted = False
        self.last_track_center_before = None
        self.last_track_center_after = None
        self.last_track_stable_now = False
        self.last_track_state_change = ""
        self.last_outlier_rejected = False
        self.nearest_track_id = None
        self.nearest_track_distance = float("nan")
        self.nearest_track_hits = 0
        self.nearest_track_committed = False
        self.nearest_track_stable = False
        self.association_attempted = False
        self.association_success = False
        self.duplicate_rejection_reason = ""
        self.last_association_mode = ""
        self.last_image_association_center_distance_px = float("nan")
        self.last_image_association_size_ratio = float("nan")
        self.last_image_association_age_s = float("nan")

        if confidence < self.min_confidence_per_hit:
            return {
                "accepted": False,
                "reason": "low_confidence",
                "track_id": None,
                "committed_now": False,
                "committed": False,
                "center": center,
            }

        if not bool(quality_ok):
            return {
                "accepted": False,
                "reason": f"quality_rejected:{quality_reason or 'unknown'}",
                "track_id": None,
                "committed_now": False,
                "committed": False,
                "center": center,
            }

        nearest = self._record_nearest_track_debug(center)

        # 1-2) Prefer a world-space committed match, then a recent
        # image-space identity match. The latter prevents a tiny repeated box
        # from spawning a new landmark solely because PnP depth jumped.
        association_mode = "world"
        image_center_distance_px = float("nan")
        image_size_ratio = float("nan")
        image_age_s = float("nan")
        tr = self._find_best_track(center, committed_only=True)
        if tr is None:
            (
                tr,
                image_center_distance_px,
                image_size_ratio,
                image_age_s,
            ) = self._find_recent_image_track(
                keypoints_px=keypoints_px,
                center_camera=center_camera,
                timestamp=timestamp,
                committed_only=True,
            )
            if tr is not None:
                association_mode = "image"
                self._record_image_association_debug(
                    mode="image_committed",
                    center_distance_px=image_center_distance_px,
                    size_ratio=image_size_ratio,
                    age_s=image_age_s,
                )
        if tr is None and nearest is not None and nearest.committed:
            # A committed-but-not-stable landmark must continue receiving hits.
            # The wider duplicate radius should block only new-track creation,
            # not observations that belong to the nearest existing track.
            tr = nearest
            association_mode = "nearest_world"
        if tr is not None:
            dist = self._distance(center, tr.center)
            self.last_update_innovation = dist
            self.last_track_center_before = tr.center.copy()
            self.association_attempted = True
            if association_mode == "image":
                print(
                    "temporal_image_association "
                    f"track={tr.id} state=committed "
                    f"age_s={image_age_s:.3f} "
                    f"center_delta_px={image_center_distance_px:.2f} "
                    f"size_ratio={image_size_ratio:.2f} "
                    f"world_innovation_m={dist:.2f}"
                )
            if association_mode == "image" and image_age_s <= 1e-6:
                self.duplicate_rejection_reason = "same_frame_image_duplicate"
                return {
                    "accepted": False,
                    "reason": "same_frame_image_duplicate",
                    "track_id": tr.id,
                    "committed_now": False,
                    "committed": True,
                    "stable": tr.is_stable,
                    "matched_visually": True,
                    "center": tr.center.copy(),
                }
            print(
                f"[ASSOC] trying match: dist={dist:.2f}, track_id={tr.id}, "
                f"mode={association_mode}"
            )
            ever_stable = bool(getattr(tr, "ever_stable", False) or tr.is_stable)
            match_threshold = (
                max(self.max_outlier_distance, self.max_committed_match_distance)
                if ever_stable
                else max(self.commit_radius, self.max_committed_match_distance)
            )
            if dist <= match_threshold:
                tr.observe_without_moving(
                    center,
                    confidence,
                    timestamp,
                    center_camera=center_camera,
                    keypoints_px=keypoints_px,
                    reprojection_error=reprojection_error,
                    keypoint_conf_min=keypoint_conf_min,
                    keypoint_conf_mean=keypoint_conf_mean,
                    quality_ok=quality_ok,
                    quality_reason=quality_reason,
                    solver_name=solver_name,
                    active_gate_idx=active_gate_idx,
                    history_size=self.history_size,
                )
                self._update_track_filter(tr)
                self.last_update_accepted = True
                self.association_success = True
                self.last_track_center_after = tr.center.copy()

                if dist > 2.0:
                    print(f"[WARN] large residual on committed track {tr.id}: {dist:.2f} m")

                return {
                    "accepted": True,
                    "reason": (
                        "matched_committed_track_image"
                        if association_mode == "image"
                        else "matched_committed_track"
                    ),
                    "track_id": tr.id,
                    "committed_now": False,
                    "committed": True,
                    "stable_now": self.last_track_stable_now,
                    "stable": tr.is_stable,
                    "matched_visually": association_mode == "image",
                    "association_mode": association_mode,
                    "center": tr.center.copy(),
                }

            tr.observe_without_moving(
                center,
                confidence,
                timestamp,
                center_camera=center_camera,
                keypoints_px=keypoints_px,
                reprojection_error=reprojection_error,
                keypoint_conf_min=keypoint_conf_min,
                keypoint_conf_mean=keypoint_conf_mean,
                quality_ok=quality_ok,
                quality_reason=quality_reason,
                solver_name=solver_name,
                active_gate_idx=active_gate_idx,
                history_size=self.history_size,
                is_outlier=True,
            )
            self._update_track_filter(tr)
            self.last_outlier_rejected = True
            self.last_track_center_after = tr.center.copy()
            print(
                f"TRACK {tr.id} observation rejected as outlier: "
                f"distance={dist:.2f} m from committed center "
                f"threshold={match_threshold:.2f}"
            )
            return {
                "accepted": False,
                "reason": (
                    f"committed_track_image_depth_outlier:{dist:.2f}"
                    if association_mode == "image"
                    else f"committed_track_outlier:{dist:.2f}"
                ),
                "track_id": tr.id,
                "committed_now": False,
                "committed": True,
                "stable_now": False,
                "stable": tr.is_stable,
                "matched_visually": association_mode == "image",
                "association_mode": association_mode,
                "center": tr.center.copy(),
            }

        # 3-4) Else try a world-space candidate, followed by a recent
        # image-space candidate.
        association_mode = "world"
        tr = self._find_best_track(center, uncommitted_only=True)
        if tr is None:
            (
                tr,
                image_center_distance_px,
                image_size_ratio,
                image_age_s,
            ) = self._find_recent_image_track(
                keypoints_px=keypoints_px,
                center_camera=center_camera,
                timestamp=timestamp,
                uncommitted_only=True,
            )
            if tr is not None:
                association_mode = "image"
                self._record_image_association_debug(
                    mode="image_candidate",
                    center_distance_px=image_center_distance_px,
                    size_ratio=image_size_ratio,
                    age_s=image_age_s,
                )
        if tr is not None:
            innovation = self._distance(center, tr.center)
            self.last_update_innovation = innovation
            self.last_track_center_before = tr.center.copy()
            self.association_attempted = True
            if association_mode == "image":
                print(
                    "temporal_image_association "
                    f"track={tr.id} state=candidate "
                    f"age_s={image_age_s:.3f} "
                    f"center_delta_px={image_center_distance_px:.2f} "
                    f"size_ratio={image_size_ratio:.2f} "
                    f"world_innovation_m={innovation:.2f}"
                )
            if association_mode == "image" and image_age_s <= 1e-6:
                self.last_track_center_after = tr.center.copy()
                self.duplicate_rejection_reason = "same_frame_image_duplicate"
                return {
                    "accepted": False,
                    "reason": "same_frame_image_duplicate",
                    "track_id": tr.id,
                    "committed_now": False,
                    "committed": False,
                    "stable": tr.is_stable,
                    "matched_visually": True,
                    "center": tr.center.copy(),
                }
            if (
                innovation > self.max_candidate_update_innovation
                and association_mode != "image"
            ):
                self.last_track_center_after = tr.center.copy()
                self.duplicate_rejection_reason = f"candidate_update_innovation_too_large:{innovation:.2f}"
                return {
                    "accepted": False,
                    "reason": f"candidate_update_innovation_too_large:{innovation:.2f}",
                    "track_id": tr.id,
                    "committed_now": False,
                    "committed": tr.committed,
                    "center": center,
                }
            committed_before = tr.committed
            tr.update(
                center,
                confidence,
                timestamp,
                alpha=self.alpha,
                center_camera=center_camera,
                keypoints_px=keypoints_px,
                reprojection_error=reprojection_error,
                keypoint_conf_min=keypoint_conf_min,
                keypoint_conf_mean=keypoint_conf_mean,
                quality_ok=quality_ok,
                quality_reason=quality_reason,
                solver_name=solver_name,
                active_gate_idx=active_gate_idx,
                history_size=self.history_size,
            )
            if association_mode == "image":
                # Recompute candidate depth clusters before commit. This lets
                # a later, repeatedly supported PnP mode replace the first
                # hypothesis without creating a second landmark.
                self._update_track_filter(tr)
                latest_is_outlier = bool(tr.obs_history[-1].is_outlier)
                if allow_candidate_commit and not latest_is_outlier:
                    self._maybe_commit(tr)
                    self._update_track_filter(tr)
            else:
                if allow_candidate_commit:
                    self._maybe_commit(tr)
                self._update_track_filter(tr)
                latest_is_outlier = bool(tr.obs_history[-1].is_outlier)
            self.last_update_accepted = not latest_is_outlier
            self.association_success = True
            self.last_track_center_after = tr.center.copy()

            return {
                "accepted": not latest_is_outlier,
                "reason": (
                    "image_track_depth_outlier"
                    if latest_is_outlier
                    else (
                        "updated_track_image"
                        if association_mode == "image"
                        else "updated_track"
                    )
                ),
                "track_id": tr.id,
                "committed_now": (not committed_before and tr.committed),
                "committed": tr.committed,
                "stable_now": self.last_track_stable_now,
                "stable": tr.is_stable,
                "matched_visually": association_mode == "image",
                "association_mode": association_mode,
                "center": tr.center.copy(),
                "commit_suppressed": bool(
                    not allow_candidate_commit and not committed_before
                ),
            }

        # 5) Block spawning a new track too close to a committed landmark
        if self._near_any_committed(center, radius=self.new_track_block_radius):
            self.duplicate_rejection_reason = "near_existing_committed_landmark"
            print(
                "[ASSOC REJECT] near_existing_committed_landmark "
                f"nearest_track_id={self.nearest_track_id} "
                f"nearest_dist={self.nearest_track_distance:.2f} "
                f"nearest_hits={self.nearest_track_hits} "
                f"nearest_committed={self.nearest_track_committed} "
                f"nearest_stable={self.nearest_track_stable} "
                f"association_attempted={self.association_attempted} "
                f"association_success={self.association_success}"
            )
            return {
                "accepted": False,
                "reason": "near_existing_committed_landmark",
                "track_id": None,
                "committed_now": False,
                "committed": True,
                "center": center,
            }

        # 6) A caller may temporarily freeze track admission while still
        # allowing observations to refresh already-known landmarks. This is
        # used while traversing an active gate, when its frame can fill the
        # image and create convincing but short-lived false gate detections.
        if not allow_new_track:
            reason = str(admission_reason or "policy")
            self.duplicate_rejection_reason = f"new_track_suppressed:{reason}"
            return {
                "accepted": False,
                "reason": f"new_track_suppressed:{reason}",
                "track_id": None,
                "committed_now": False,
                "committed": False,
                "center": center,
            }

        # 7) Otherwise create a new candidate track
        tr = self._create_track(
            center,
            confidence,
            timestamp,
            center_camera=center_camera,
            keypoints_px=keypoints_px,
            reprojection_error=reprojection_error,
            keypoint_conf_min=keypoint_conf_min,
            keypoint_conf_mean=keypoint_conf_mean,
            quality_ok=quality_ok,
            quality_reason=quality_reason,
            solver_name=solver_name,
            active_gate_idx=active_gate_idx,
        )
        self.last_update_innovation = 0.0
        self.last_update_accepted = True
        self.last_track_center_before = center.copy()
        committed_before = tr.committed
        self._maybe_commit(tr)
        self.last_track_center_after = tr.center.copy()

        return {
            "accepted": True,
            "reason": "new_track",
            "track_id": tr.id,
            "committed_now": (not committed_before and tr.committed),
            "committed": tr.committed,
            "stable_now": self.last_track_stable_now,
            "stable": tr.is_stable,
            "center": tr.center.copy(),
        }

    def get_committed_tracks(self) -> List[GateTrack]:
        out = [tr for tr in self.tracks if tr.committed]
        # Preserve persistent landmark identity. Race ordering is handled by
        # RaceProgression, not by spatial coordinates.
        out.sort(key=lambda tr: tr.id)
        return out

    def get_committed_centers(self) -> List[np.ndarray]:
        return [tr.center.copy() for tr in self.get_committed_tracks()]

    def get_stable_tracks(self) -> List[GateTrack]:
        out = [tr for tr in self.tracks if tr.committed and tr.is_stable]
        out.sort(key=lambda tr: tr.id)
        return out

    def tentative_track_ids(self) -> List[int]:
        return [tr.id for tr in self.tracks if not tr.is_stable]

    def stable_track_ids(self) -> List[int]:
        return [tr.id for tr in self.tracks if tr.is_stable]

    def get_track_by_id(self, track_id: int) -> Optional[GateTrack]:
        for tr in self.tracks:
            if tr.id == track_id:
                return tr
        return None

    def find_committed_track_near_center(self, center: np.ndarray, radius: float = 2.0) -> Optional[GateTrack]:
        center = np.asarray(center, dtype=float).reshape(3)

        best = None
        best_dist = float("inf")

        for tr in self.tracks:
            if not tr.committed:
                continue

            d = self._distance(center, tr.center)
            if d < radius and d < best_dist:
                best = tr
                best_dist = d

        return best
