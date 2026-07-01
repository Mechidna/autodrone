from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TargetDiagnostics:
    gate_idx: int
    active_track_id: Optional[int]
    locked: bool
    lock_age_s: float
    shift_m: float
    shift_xy_m: float
    shift_z_m: float
    event: str
    suppress_active_replan: bool
    center_at_plan: Optional[np.ndarray]
    latest_center: Optional[np.ndarray]


class TargetManager:
    """
    Minimal active-target owner for the pilot stack.

    Perception may keep refining landmarks, but the current approach target is a
    planned contract. Once locked, live landmark updates are diagnostic inputs
    until the gate is passed; they do not directly move the active target.
    """

    def __init__(
        self,
        *,
        race_gate_count: Optional[int],
        active_target_lost_grace_s: float,
        shift_print_threshold_m: float = 0.25,
        shift_print_period_s: float = 1.0,
    ):
        self.race_gate_count = None if race_gate_count is None else int(race_gate_count)
        self.active_target_lost_grace_s = float(active_target_lost_grace_s)
        self.shift_print_threshold_m = float(shift_print_threshold_m)
        self.shift_print_period_s = float(shift_print_period_s)

        self.current_gate_idx = 0
        self.completed_track_ids: set[int] = set()
        self.active_target_track_id: Optional[int] = None
        self.center_at_plan: Optional[np.ndarray] = None
        self.latest_center: Optional[np.ndarray] = None
        self.lock_time_s: Optional[float] = None
        self.active_target_lost_time_s: Optional[float] = None
        self.last_event = "init"
        self.suppress_active_replan = False

        self.shift_m = float("nan")
        self.shift_xy_m = float("nan")
        self.shift_z_m = float("nan")
        self._last_shift_print_time_s = 0.0
        self._last_shift_print_bucket = -1

    @property
    def locked(self) -> bool:
        return self.center_at_plan is not None

    def set_gate_index(self, gate_idx: int) -> None:
        self.current_gate_idx = max(0, int(gate_idx))
        if self.race_gate_count is not None:
            self.current_gate_idx = min(self.current_gate_idx, self.race_gate_count)

    def lock_target(
        self,
        *,
        gate_idx: int,
        track_id,
        center_neu,
        reason: str,
        now_s: Optional[float] = None,
    ) -> np.ndarray:
        center = self._as_vec3(center_neu)
        now_s = time.time() if now_s is None else float(now_s)
        self.set_gate_index(gate_idx)
        self.active_target_track_id = self._canonical_track_id(track_id)
        self.center_at_plan = center.copy()
        self.latest_center = center.copy()
        self.lock_time_s = now_s
        self.active_target_lost_time_s = None
        self.last_event = f"lock:{reason}"
        self.suppress_active_replan = False
        self._update_shift(center)
        print(
            "target_manager lock "
            f"gate_idx={self.current_gate_idx} "
            f"track={self._track_text(self.active_target_track_id)} "
            f"center={self._fmt_vec(center)} "
            f"reason={reason}",
            flush=True,
        )
        return center.copy()

    def update_live_targets(
        self,
        *,
        gate_idx: int,
        gates: list[np.ndarray],
        track_ids: list,
        now_s: Optional[float] = None,
    ) -> tuple[list[np.ndarray], list]:
        now_s = time.time() if now_s is None else float(now_s)
        self.set_gate_index(gate_idx)

        entries = [
            (track_ids[i] if i < len(track_ids) else None, self._as_vec3(gate))
            for i, gate in enumerate(gates)
        ]

        if not self.locked:
            self.suppress_active_replan = False
            return [gate.copy() for _, gate in entries], [track_id for track_id, _ in entries]

        active_id = self.active_target_track_id
        live_idx = self._find_active_entry(entries, active_id)
        if live_idx is not None:
            self.latest_center = entries[live_idx][1].copy()
            self.active_target_lost_time_s = None
            self.last_event = "live_active_seen"
            self._update_shift(self.latest_center)
            self._maybe_print_shift()
        elif self.active_target_lost_time_s is None:
            self.active_target_lost_time_s = now_s
            self.last_event = "live_active_lost"
            print(
                "target_manager active_lost "
                f"gate_idx={self.current_gate_idx} "
                f"track={self._track_text(active_id)} "
                f"grace_s={self.active_target_lost_grace_s:.2f}",
                flush=True,
            )
        elif now_s - self.active_target_lost_time_s > self.active_target_lost_grace_s:
            self.last_event = "live_active_missing_after_grace"

        locked_entry = (active_id, self.center_at_plan.copy())
        if active_id is not None:
            entries = [
                entry
                for entry in entries
                if self._canonical_track_id(entry[0]) != active_id
            ]

        insert_idx = min(max(self.current_gate_idx, 0), len(entries))
        if active_id is not None:
            entries.insert(insert_idx, locked_entry)
        elif self.current_gate_idx < len(entries):
            entries[insert_idx] = locked_entry
        else:
            entries.insert(insert_idx, locked_entry)

        self.suppress_active_replan = True
        return [gate.copy() for _, gate in entries], [track_id for track_id, _ in entries]

    def mark_passed(
        self,
        *,
        pos_neu,
        distance_m: float,
        truth_pos_neu=None,
        truth_error_m=None,
        pass_reason: str | None = None,
        plane_progress_m: float | None = None,
        lateral_error_m: float | None = None,
    ) -> None:
        completed_id = self.active_target_track_id
        if completed_id is not None:
            self.completed_track_ids.add(int(completed_id))

        extra_txt = ""
        if pass_reason:
            extra_txt += f" reason={pass_reason}"
        if plane_progress_m is not None:
            try:
                plane_progress = float(plane_progress_m)
            except (TypeError, ValueError):
                plane_progress = float("nan")
            if math.isfinite(plane_progress):
                extra_txt += f" plane={plane_progress:.2f}"
        if lateral_error_m is not None:
            try:
                lateral_error = float(lateral_error_m)
            except (TypeError, ValueError):
                lateral_error = float("nan")
            if math.isfinite(lateral_error):
                extra_txt += f" lateral={lateral_error:.2f}"
        if truth_pos_neu is not None:
            try:
                extra_txt += f" truth_pos={self._fmt_vec(self._as_vec3(truth_pos_neu))}"
            except (TypeError, ValueError):
                pass
        if truth_error_m is not None:
            try:
                truth_error = float(truth_error_m)
            except (TypeError, ValueError):
                truth_error = float("nan")
            if math.isfinite(truth_error):
                extra_txt += f" truth_err={truth_error:.2f}"

        print(
            "target_manager passed "
            f"gate_idx={self.current_gate_idx} "
            f"track={self._track_text(completed_id)} "
            f"dist={float(distance_m):.2f} "
            f"pos={self._fmt_vec(self._as_vec3(pos_neu))}"
            f"{extra_txt}",
            flush=True,
        )
        self.current_gate_idx += 1
        if self.race_gate_count is not None:
            self.current_gate_idx = min(self.current_gate_idx, self.race_gate_count)
        self.clear_active(reason="passed")

    def clear_active(self, *, reason: str) -> None:
        self.active_target_track_id = None
        self.center_at_plan = None
        self.latest_center = None
        self.lock_time_s = None
        self.active_target_lost_time_s = None
        self.shift_m = float("nan")
        self.shift_xy_m = float("nan")
        self.shift_z_m = float("nan")
        self.suppress_active_replan = False
        self.last_event = f"clear:{reason}"

    def diagnostics(self, now_s: Optional[float] = None) -> TargetDiagnostics:
        now_s = time.time() if now_s is None else float(now_s)
        lock_age_s = (
            float("nan")
            if self.lock_time_s is None
            else max(0.0, now_s - float(self.lock_time_s))
        )
        return TargetDiagnostics(
            gate_idx=int(self.current_gate_idx),
            active_track_id=self.active_target_track_id,
            locked=self.locked,
            lock_age_s=lock_age_s,
            shift_m=float(self.shift_m),
            shift_xy_m=float(self.shift_xy_m),
            shift_z_m=float(self.shift_z_m),
            event=str(self.last_event),
            suppress_active_replan=bool(self.suppress_active_replan),
            center_at_plan=(
                None if self.center_at_plan is None else self.center_at_plan.copy()
            ),
            latest_center=None if self.latest_center is None else self.latest_center.copy(),
        )

    def _update_shift(self, latest_center: np.ndarray) -> None:
        if self.center_at_plan is None:
            self.shift_m = float("nan")
            self.shift_xy_m = float("nan")
            self.shift_z_m = float("nan")
            return
        delta = self._as_vec3(latest_center) - self.center_at_plan
        self.shift_m = float(np.linalg.norm(delta))
        self.shift_xy_m = float(np.linalg.norm(delta[:2]))
        self.shift_z_m = float(abs(delta[2]))

    def _maybe_print_shift(self) -> None:
        if not math.isfinite(self.shift_m) or self.shift_m < self.shift_print_threshold_m:
            return
        bucket = int(self.shift_m / max(self.shift_print_threshold_m, 1e-6))
        now_s = time.time()
        if (
            bucket == self._last_shift_print_bucket
            and now_s - self._last_shift_print_time_s < self.shift_print_period_s
        ):
            return
        self._last_shift_print_bucket = bucket
        self._last_shift_print_time_s = now_s
        print(
            "target_manager active_shift "
            f"gate_idx={self.current_gate_idx} "
            f"track={self._track_text(self.active_target_track_id)} "
            f"shift={self.shift_m:.2f} "
            f"xy={self.shift_xy_m:.2f} "
            f"z={self.shift_z_m:.2f} "
            f"locked={self._fmt_vec(self.center_at_plan)} "
            f"latest={self._fmt_vec(self.latest_center)} "
            "active_replan=held",
            flush=True,
        )

    @staticmethod
    def _find_active_entry(entries, active_id: Optional[int]) -> Optional[int]:
        if active_id is None:
            return None
        for idx, (track_id, _) in enumerate(entries):
            if TargetManager._canonical_track_id(track_id) == active_id:
                return idx
        return None

    @staticmethod
    def _canonical_track_id(track_id) -> Optional[int]:
        if track_id is None:
            return None
        try:
            return int(track_id)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_vec3(value) -> np.ndarray:
        arr = np.asarray(value, dtype=float).reshape(3)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"target vector is not finite: {value!r}")
        return arr

    @staticmethod
    def _fmt_vec(value) -> str:
        if value is None:
            return "None"
        arr = np.asarray(value, dtype=float).reshape(3)
        return f"({arr[0]:.2f},{arr[1]:.2f},{arr[2]:.2f})"

    @staticmethod
    def _track_text(track_id: Optional[int]) -> str:
        return "none" if track_id is None else str(int(track_id))
