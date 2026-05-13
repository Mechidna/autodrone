import time
from typing import Iterable, List, Optional, Sequence

import numpy as np


class RaceProgression:
    """
    Deterministic race-order state machine.

    The progression cursor advances through gate track IDs, not through spatial
    sorting. A predefined order is preferred when available; otherwise newly
    committed track IDs are appended in discovery/commit order. Gate landmarks
    remain persistent and can be revisited on later laps.
    """

    def __init__(
        self,
        race_order: Optional[Sequence[int]] = None,
        pass_radius: float = 1.25,
        clear_radius: float = 1.75,
        advance_debounce_s: float = 0.75,
        allow_laps: bool = True,
    ):
        self.predefined_order = None if race_order is None else [int(x) for x in race_order]
        self.inferred_order: List[int] = []
        self.cursor = 0
        self.lap = 0
        self.pass_radius = float(pass_radius)
        self.clear_radius = max(float(clear_radius), self.pass_radius)
        self.advance_debounce_s = float(advance_debounce_s)
        self.allow_laps = bool(allow_laps)

        self.last_passed_track_id = None
        self.last_advance_time = -float("inf")
        self.waiting_for_clear_track_id = None

    def reset(self):
        self.cursor = 0
        self.lap = 0
        self.last_passed_track_id = None
        self.last_advance_time = -float("inf")
        self.waiting_for_clear_track_id = None

    def set_predefined_order(self, race_order: Optional[Sequence[int]]):
        self.predefined_order = None if race_order is None else [int(x) for x in race_order]
        self.reset()

    def sync_committed_tracks(self, tracks: Iterable) -> None:
        """
        Extend fallback order with newly committed track IDs.

        This intentionally uses persistent track identity, not position. If the
        perception stack cannot provide semantic gate IDs, discovery order is
        the only deterministic fallback that supports non-monotonic geometry.
        """
        known = set(self.inferred_order)
        new_ids = [int(tr.id) for tr in tracks if int(tr.id) not in known]
        self.inferred_order.extend(sorted(new_ids))

    def order(self) -> List[int]:
        if self.predefined_order is not None:
            return list(self.predefined_order)
        return list(self.inferred_order)

    def current_track_id(self) -> Optional[int]:
        order = self.order()
        if len(order) == 0:
            return None
        if self.cursor >= len(order):
            if not self.allow_laps:
                return None
            self.cursor = self.cursor % len(order)
        return order[self.cursor]

    def horizon_track_ids(self, max_gates_ahead: int) -> List[int]:
        order = self.order()
        if len(order) == 0 or max_gates_ahead <= 0:
            return []

        horizon = []
        idx = self.cursor
        steps = 0
        max_steps = max_gates_ahead if self.allow_laps else max(0, len(order) - idx)

        while len(horizon) < max_gates_ahead and steps < max_steps:
            if idx >= len(order):
                if not self.allow_laps:
                    break
                idx = 0

            track_id = order[idx]

            # Do not immediately re-target the same gate while the vehicle has
            # not cleared it. This only suppresses adjacent duplicate IDs; it
            # does not prevent normal lap wrap-around after other gates.
            if not (
                len(horizon) == 0
                and track_id == self.last_passed_track_id
                and self.waiting_for_clear_track_id == track_id
            ):
                horizon.append(track_id)

            idx += 1
            steps += 1

        return horizon

    def update_clearance(self, current_pos, track_by_id) -> None:
        if self.waiting_for_clear_track_id is None:
            return

        track = track_by_id(self.waiting_for_clear_track_id)
        if track is None:
            self.waiting_for_clear_track_id = None
            return

        dist = float(np.linalg.norm(np.asarray(current_pos, dtype=float).reshape(3) - track.center))
        if dist > self.clear_radius:
            self.waiting_for_clear_track_id = None

    def try_advance(self, current_pos, track_by_id, now: Optional[float] = None) -> dict:
        """
        Advance only when the active sequence gate is reached and debounce/clear
        rules allow it. Returns diagnostics for caller logging.
        """
        now = time.time() if now is None else float(now)
        current_id = self.current_track_id()
        if current_id is None:
            return {"advanced": False, "reason": "no_active_gate", "track_id": None}

        self.update_clearance(current_pos, track_by_id)

        if now - self.last_advance_time < self.advance_debounce_s:
            return {"advanced": False, "reason": "debounce", "track_id": current_id}

        if self.waiting_for_clear_track_id == current_id:
            return {"advanced": False, "reason": "waiting_for_clear", "track_id": current_id}

        track = track_by_id(current_id)
        if track is None:
            return {"advanced": False, "reason": "target_not_committed", "track_id": current_id}

        current_pos = np.asarray(current_pos, dtype=float).reshape(3)
        dist = float(np.linalg.norm(current_pos - track.center))
        if dist > self.pass_radius:
            return {
                "advanced": False,
                "reason": "outside_pass_radius",
                "track_id": current_id,
                "distance": dist,
            }

        self.last_passed_track_id = current_id
        self.waiting_for_clear_track_id = current_id
        self.last_advance_time = now
        self.cursor += 1

        order = self.order()
        if len(order) > 0 and self.cursor >= len(order):
            if self.allow_laps:
                self.cursor = 0
                self.lap += 1
            else:
                self.cursor = len(order)

        return {
            "advanced": True,
            "reason": "passed_gate",
            "track_id": current_id,
            "distance": dist,
            "next_track_id": self.current_track_id(),
            "lap": self.lap,
            "cursor": self.cursor,
        }

