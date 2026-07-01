from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_EPS = 1e-9


@dataclass(frozen=True)
class GatePlanePassResult:
    passed: bool
    reason: str
    distance_m: float
    signed_progress_m: float
    previous_signed_progress_m: float
    lateral_error_m: float
    crossed_plane: bool
    crossing_point: np.ndarray | None


def unit_vector_from_to(start, target, fallback=None) -> np.ndarray | None:
    start_arr = np.asarray(start, dtype=float).reshape(3)
    target_arr = np.asarray(target, dtype=float).reshape(3)
    delta = target_arr - start_arr
    norm = float(np.linalg.norm(delta))
    if norm > _EPS and np.all(np.isfinite(delta)):
        return delta / norm

    if fallback is None:
        return None
    fallback_arr = np.asarray(fallback, dtype=float).reshape(3)
    fallback_norm = float(np.linalg.norm(fallback_arr))
    if fallback_norm > _EPS and np.all(np.isfinite(fallback_arr)):
        return fallback_arr / fallback_norm
    return None


def check_gate_plane_pass(
    *,
    previous_position,
    position,
    center,
    normal,
    lateral_radius_m: float,
    plane_tolerance_m: float,
) -> GatePlanePassResult:
    prev = np.asarray(previous_position, dtype=float).reshape(3)
    pos = np.asarray(position, dtype=float).reshape(3)
    gate_center = np.asarray(center, dtype=float).reshape(3)
    normal_arr = np.asarray(normal, dtype=float).reshape(3)

    distance = float(np.linalg.norm(pos - gate_center))
    invalid = (
        not np.all(np.isfinite(prev))
        or not np.all(np.isfinite(pos))
        or not np.all(np.isfinite(gate_center))
        or not np.all(np.isfinite(normal_arr))
    )
    if invalid:
        return GatePlanePassResult(
            passed=False,
            reason="invalid_gate_pass_geometry",
            distance_m=distance,
            signed_progress_m=float("nan"),
            previous_signed_progress_m=float("nan"),
            lateral_error_m=float("nan"),
            crossed_plane=False,
            crossing_point=None,
        )

    normal_norm = float(np.linalg.norm(normal_arr))
    if normal_norm <= _EPS:
        return GatePlanePassResult(
            passed=False,
            reason="missing_gate_plane_normal",
            distance_m=distance,
            signed_progress_m=float("nan"),
            previous_signed_progress_m=float("nan"),
            lateral_error_m=float("nan"),
            crossed_plane=False,
            crossing_point=None,
        )

    lateral_radius = float(lateral_radius_m)
    if not np.isfinite(lateral_radius) or lateral_radius <= 0.0:
        return GatePlanePassResult(
            passed=False,
            reason="invalid_gate_lateral_radius",
            distance_m=distance,
            signed_progress_m=float("nan"),
            previous_signed_progress_m=float("nan"),
            lateral_error_m=float("nan"),
            crossed_plane=False,
            crossing_point=None,
        )

    n = normal_arr / normal_norm
    previous_progress = float(np.dot(prev - gate_center, n))
    progress = float(np.dot(pos - gate_center, n))
    tolerance = max(0.0, float(plane_tolerance_m))

    crossed = previous_progress <= 0.0 <= progress
    moving_toward_exit = progress >= previous_progress - 1e-6
    on_plane = abs(progress) <= tolerance and moving_toward_exit

    crossing_point = None
    if crossed:
        denom = progress - previous_progress
        if abs(denom) > _EPS:
            alpha = float(np.clip(-previous_progress / denom, 0.0, 1.0))
            crossing_point = prev + alpha * (pos - prev)
        else:
            crossing_point = pos.copy()
    elif on_plane:
        crossing_point = pos.copy()

    if crossing_point is None:
        rel = pos - gate_center
    else:
        rel = crossing_point - gate_center
    lateral_vec = rel - float(np.dot(rel, n)) * n
    lateral_error = float(np.linalg.norm(lateral_vec))

    if crossing_point is None:
        return GatePlanePassResult(
            passed=False,
            reason="not_past_gate_plane",
            distance_m=distance,
            signed_progress_m=progress,
            previous_signed_progress_m=previous_progress,
            lateral_error_m=lateral_error,
            crossed_plane=False,
            crossing_point=None,
        )

    if lateral_error > lateral_radius:
        return GatePlanePassResult(
            passed=False,
            reason=f"lateral_error_too_large:{lateral_error:.2f}",
            distance_m=distance,
            signed_progress_m=progress,
            previous_signed_progress_m=previous_progress,
            lateral_error_m=lateral_error,
            crossed_plane=bool(crossed),
            crossing_point=crossing_point,
        )

    return GatePlanePassResult(
        passed=True,
        reason="crossed_gate_plane" if crossed else "on_gate_plane",
        distance_m=distance,
        signed_progress_m=progress,
        previous_signed_progress_m=previous_progress,
        lateral_error_m=lateral_error,
        crossed_plane=bool(crossed),
        crossing_point=crossing_point,
    )
