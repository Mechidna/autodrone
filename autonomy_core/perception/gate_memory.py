import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GateTrack:
    id: int
    center: np.ndarray
    confidence_sum: float = 0.0
    hits: int = 0
    last_seen_time: float = 0.0
    committed: bool = False

    # Raw measurement history used for commit consistency checks / debugging
    measurement_history: List[np.ndarray] = field(default_factory=list)

    # Residual history after commit: distance from new observation to locked center
    recent_errors: List[float] = field(default_factory=list)

    def update(self, measurement: np.ndarray, confidence: float, timestamp: float, alpha: float = 0.35):
        measurement = np.asarray(measurement, dtype=float).reshape(3)

        # exponential smoothing / fusion while still uncommitted
        self.center = (1.0 - alpha) * self.center + alpha * measurement

        self.confidence_sum += float(confidence)
        self.hits += 1
        self.last_seen_time = float(timestamp)
        self.measurement_history.append(measurement.copy())

    def observe_without_moving(self, measurement: np.ndarray, confidence: float, timestamp: float):
        """
        Record that this committed landmark was seen again, but do not move its center.
        """
        measurement = np.asarray(measurement, dtype=float).reshape(3)

        self.confidence_sum += float(confidence)
        self.hits += 1
        self.last_seen_time = float(timestamp)
        self.measurement_history.append(measurement.copy())

        err = float(np.linalg.norm(measurement - self.center))
        self.recent_errors.append(err)

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
        self.max_committed_match_distance = 0.8  # meters (start here)

        self._next_id = 0
        self.tracks: List[GateTrack] = []

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

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

    def _near_any_committed(self, center: np.ndarray, radius: Optional[float] = None) -> bool:
        r = self.new_track_block_radius if radius is None else float(radius)
        for tr in self.tracks:
            if not tr.committed:
                continue
            if self._distance(center, tr.center) < r:
                return True
        return False

    def _create_track(self, center: np.ndarray, confidence: float, timestamp: float) -> GateTrack:
        center = np.asarray(center, dtype=float).reshape(3)

        tr = GateTrack(
            id=self._next_id,
            center=center.copy(),
            confidence_sum=float(confidence),
            hits=1,
            last_seen_time=float(timestamp),
            committed=False,
            measurement_history=[center.copy()],
            recent_errors=[],
        )
        self._next_id += 1
        self.tracks.append(tr)
        return tr

    def _candidate_spread(self, tr: GateTrack, n_recent: int = 10) -> float:
        """
        Maximum distance of recent measurements from the candidate center.
        """
        measurements = tr.recent_measurements(n_recent)
        if len(measurements) == 0:
            return float("inf")

        center = tr.center
        dists = [self._distance(m, center) for m in measurements]
        return float(max(dists)) if len(dists) > 0 else float("inf")

    def _maybe_commit(self, tr: GateTrack):
        if tr.committed:
            return

        if tr.hits < self.commit_hits:
            return

        if tr.confidence_sum < self.commit_confidence_sum:
            return

        spread = self._candidate_spread(tr, n_recent=10)
        if spread > self.commit_spread_radius:
            return

        # avoid committing duplicates near an already committed landmark
        for other in self.tracks:
            if other.id == tr.id:
                continue
            if other.committed and self._distance(tr.center, other.center) < self.commit_radius:
                return

        tr.committed = True

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

    def add_detection(self, center: np.ndarray, confidence: float, timestamp: float):
        """
        Add a new world-frame gate center measurement.

        Association order:
        1) nearest COMMITTED track
        2) nearest UNCOMMITTED track
        3) if near committed landmark, reject creating a new track
        4) otherwise create a new candidate track
        """
        center = np.asarray(center, dtype=float).reshape(3)
        confidence = float(confidence)

        if confidence < self.min_confidence_per_hit:
            return {
                "accepted": False,
                "reason": "low_confidence",
                "track_id": None,
                "committed_now": False,
                "committed": False,
                "center": center,
            }

        # 1) Prefer matching an existing committed landmark
        tr = self._find_best_track(center, committed_only=True)
        if tr is not None:
            dist = self._distance(center, tr.center)
            print(f"[ASSOC] trying match: dist={dist:.2f}, track_id={tr.id}")
            if dist < self.max_committed_match_distance:
                tr.observe_without_moving(center, confidence, timestamp)

                if dist > 2.0:
                    print(f"[WARN] large residual on committed track {tr.id}: {dist:.2f} m")

                return {
                    "accepted": True,
                    "reason": "matched_committed_track",
                    "track_id": tr.id,
                    "committed_now": False,
                    "committed": True,
                    "center": tr.center.copy(),
                }
            else:
                # too far → do NOT match this committed track
                pass

        # 2) Else try matching an uncommitted candidate
        tr = self._find_best_track(center, uncommitted_only=True)
        if tr is not None:
            committed_before = tr.committed
            tr.update(center, confidence, timestamp, alpha=self.alpha)
            self._maybe_commit(tr)

            return {
                "accepted": True,
                "reason": "updated_track",
                "track_id": tr.id,
                "committed_now": (not committed_before and tr.committed),
                "committed": tr.committed,
                "center": tr.center.copy(),
            }

        # 3) Block spawning a new track too close to a committed landmark
        if self._near_any_committed(center, radius=self.new_track_block_radius):
            return {
                "accepted": False,
                "reason": "near_existing_committed_landmark",
                "track_id": None,
                "committed_now": False,
                "committed": True,
                "center": center,
            }

        # 4) Otherwise create a new candidate track
        tr = self._create_track(center, confidence, timestamp)
        committed_before = tr.committed
        self._maybe_commit(tr)

        return {
            "accepted": True,
            "reason": "new_track",
            "track_id": tr.id,
            "committed_now": (not committed_before and tr.committed),
            "committed": tr.committed,
            "center": tr.center.copy(),
        }

    def get_committed_tracks(self) -> List[GateTrack]:
        out = [tr for tr in self.tracks if tr.committed]
        out.sort(key=lambda tr: tr.center[1])
        return out

    def get_committed_centers(self) -> List[np.ndarray]:
        return [tr.center.copy() for tr in self.get_committed_tracks()]

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