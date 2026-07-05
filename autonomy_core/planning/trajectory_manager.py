"""Low-risk trajectory helper functions used by AutonomyAPI.

Phase 5 keeps installed-plan ownership and planning state on AutonomyAPI. These
helpers are pure calculations or simple reads that do not own planner state.
"""

from __future__ import annotations

import numpy as np


def choose_T(p0, v0, p1, vmax=2.5, amax=2.0, T_min=1.0):
    """Allocate a conservative segment duration for one waypoint interval."""

    dp = p1 - p0
    d = np.linalg.norm(dp)

    if d < 1e-6:
        return T_min

    dir_vec = dp / d
    v_along = np.dot(v0, dir_vec)

    t_acc = vmax / amax
    d_acc = 0.5 * amax * t_acc**2

    if d > 2 * d_acc:
        T_base = 2 * t_acc + (d - 2 * d_acc) / vmax
    else:
        T_base = 2 * np.sqrt(d / amax)

    if v_along < 0:
        T_base += min(abs(v_along) / amax, 2.0)
    else:
        T_base -= min(v_along / (2 * amax), 0.5)

    return max(T_base, T_min)


def allocate_segment_times(waypoints, current_vel, vmax=2.5, amax=2.0, T_min=1.0):
    """Allocate per-segment times for the current waypoint horizon."""

    times = []

    for i in range(len(waypoints) - 1):
        p0 = waypoints[i]
        p1 = waypoints[i + 1]

        if i == 0:
            v0 = current_vel
        else:
            v0 = np.zeros(3, dtype=float)

        T = choose_T(p0, v0, p1, vmax=vmax, amax=amax, T_min=T_min)
        times.append(T)

    return np.asarray(times, dtype=float)


def active_target_crossing_tau(active_times, target_idx):
    """Return the cumulative crossing tau for an active target index."""

    if active_times is None:
        return float("nan")
    times = np.asarray(active_times, dtype=float).reshape(-1)
    target_idx = int(target_idx)
    if target_idx < 0 or target_idx >= len(times):
        return float("nan")
    return float(np.sum(times[:target_idx + 1]))
