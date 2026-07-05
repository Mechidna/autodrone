"""Minimum-snap trajectory validation helpers.

This module is deliberately stateless. AutonomyAPI keeps the public wrappers and
plan_* debug fields for compatibility with the runner/logger surface.
"""

from __future__ import annotations

import numpy as np


def reset_plan_geometric_validation_debug(target) -> None:
    """Reset AutonomyAPI-compatible plan validation debug fields."""

    target.plan_geometric_validation_failed = False
    target.plan_geometric_fallback_used = False
    target.plan_validation_failed_segment_idx = -1
    target.plan_max_backward_progress_m = 0.0
    target.plan_max_overshoot_m = 0.0
    target.plan_negative_progress_velocity_count = 0
    target.plan_validation_failure_reason = ""
    target.plan_z_corridor_failed = False
    target.plan_min_z = float("nan")
    target.plan_max_z = float("nan")
    target.plan_z_undershoot_m = 0.0
    target.plan_z_fallback_reason = ""
    target.plan_z_start_below_safe_min = False


def validate_minimum_snap_geometry(
    planner,
    waypoints,
    safe_min_target_z,
    samples_per_segment=80,
    backward_tolerance_m=0.15,
    overshoot_tolerance_m=0.35,
    negative_velocity_tolerance=4,
    endpoint_margin_fraction=0.08,
    z_corridor_tolerance_m=0.05,
    z_endpoint_undershoot_tolerance_m=0.20,
):
    """Validate trajectory monotonicity and z-corridor constraints."""

    waypoints = np.asarray(waypoints, dtype=float)
    times = np.asarray(getattr(planner, "times", []), dtype=float).reshape(-1)
    segment_starts = np.asarray(
        getattr(planner, "segment_starts", []), dtype=float
    ).reshape(-1)
    if len(waypoints) < 2 or len(times) != len(waypoints) - 1:
        return False, {
            "segment_idx": -1,
            "max_backward_progress_m": 0.0,
            "max_overshoot_m": 0.0,
            "negative_progress_velocity_count": 0,
            "plan_min_z": float("nan"),
            "plan_max_z": float("nan"),
            "z_undershoot_m": 0.0,
            "z_start_below_safe_min": False,
            "reason": "invalid_validation_inputs",
        }

    worst = {
        "segment_idx": -1,
        "max_backward_progress_m": 0.0,
        "max_overshoot_m": 0.0,
        "negative_progress_velocity_count": 0,
        "plan_min_z": float("nan"),
        "plan_max_z": float("nan"),
        "z_undershoot_m": 0.0,
        "z_start_below_safe_min": False,
        "reason": "",
    }
    plan_min_z = float("inf")
    plan_max_z = float("-inf")
    worst_z_undershoot = 0.0

    for segment_idx in range(len(times)):
        p0 = waypoints[segment_idx]
        p1 = waypoints[segment_idx + 1]
        delta = p1 - p0
        segment_length = float(np.linalg.norm(delta))
        if not np.isfinite(segment_length) or segment_length < 1e-6:
            continue
        direction = delta / segment_length
        duration = float(times[segment_idx])
        segment_start = float(segment_starts[segment_idx])
        sample_count = max(3, int(samples_per_segment))
        progress_values = []
        z_values = []
        negative_velocity_count = 0

        for tau in np.linspace(0.0, duration, sample_count):
            p, v, _ = planner.sample(segment_start + float(tau))
            progress = float(np.dot(p - p0, direction))
            progress_values.append(progress)
            z = float(p[2])
            z_values.append(z)
            if np.isfinite(z):
                plan_min_z = min(plan_min_z, z)
                plan_max_z = max(plan_max_z, z)
            s_dot = float(np.dot(v, direction))
            normalized_tau = float(tau) / duration if duration > 1e-6 else 1.0
            if (
                normalized_tau < 1.0 - float(endpoint_margin_fraction)
                and s_dot < -1e-3
            ):
                negative_velocity_count += 1

        max_backward = 0.0
        max_seen = progress_values[0]
        for progress in progress_values[1:]:
            max_backward = max(max_backward, max_seen - progress)
            max_seen = max(max_seen, progress)

        min_progress = min(progress_values)
        max_progress = max(progress_values)
        min_z = min(z_values) if z_values else float("nan")
        max_overshoot = max(
            0.0,
            max_progress - segment_length,
            -min_progress,
        )
        z_start = float(p0[2])
        z_end = float(p1[2])
        z_start_below_safe_min = bool(
            np.isfinite(z_start)
            and z_start < float(safe_min_target_z)
        )
        if z_start_below_safe_min:
            segment_floor = z_start - float(z_corridor_tolerance_m)
        else:
            segment_floor = max(
                float(safe_min_target_z) - float(z_corridor_tolerance_m),
                min(z_start, z_end) - float(z_endpoint_undershoot_tolerance_m),
            )
        z_undershoot = max(0.0, segment_floor - min_z) if np.isfinite(min_z) else 0.0
        worst_z_undershoot = max(worst_z_undershoot, z_undershoot)

        failed_reasons = []
        if max_backward > backward_tolerance_m:
            failed_reasons.append("backward_progress")
        if max_overshoot > overshoot_tolerance_m:
            failed_reasons.append("segment_overshoot")
        if negative_velocity_count > int(negative_velocity_tolerance):
            failed_reasons.append("negative_progress_velocity")
        if z_undershoot > 0.0:
            failed_reasons.append("z_corridor")

        if (
            max_backward > worst["max_backward_progress_m"]
            or max_overshoot > worst["max_overshoot_m"]
            or negative_velocity_count > worst["negative_progress_velocity_count"]
            or z_undershoot > worst["z_undershoot_m"]
        ):
            worst = {
                "segment_idx": int(segment_idx),
                "max_backward_progress_m": float(max_backward),
                "max_overshoot_m": float(max_overshoot),
                "negative_progress_velocity_count": int(negative_velocity_count),
                "plan_min_z": float(plan_min_z),
                "plan_max_z": float(plan_max_z),
                "z_undershoot_m": float(z_undershoot),
                "z_start_below_safe_min": bool(z_start_below_safe_min),
                "reason": ",".join(failed_reasons),
            }

        if failed_reasons:
            worst["segment_idx"] = int(segment_idx)
            worst["plan_min_z"] = float(plan_min_z)
            worst["plan_max_z"] = float(plan_max_z)
            worst["z_undershoot_m"] = float(z_undershoot)
            worst["z_start_below_safe_min"] = bool(z_start_below_safe_min)
            worst["reason"] = ",".join(failed_reasons)
            return False, worst

    if not worst["reason"]:
        worst["reason"] = "ok"
    worst["plan_min_z"] = float(plan_min_z) if np.isfinite(plan_min_z) else float("nan")
    worst["plan_max_z"] = float(plan_max_z) if np.isfinite(plan_max_z) else float("nan")
    worst["z_undershoot_m"] = float(worst_z_undershoot)
    worst["z_start_below_safe_min"] = bool(
        np.any(waypoints[:-1, 2] < float(safe_min_target_z))
    )
    return True, worst
