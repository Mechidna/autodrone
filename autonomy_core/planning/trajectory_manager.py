"""Low-risk trajectory helper functions used by AutonomyAPI.

Phase 5 keeps installed-plan ownership and planning state on AutonomyAPI. These
helpers are pure calculations or simple reads that do not own planner state.
"""

from __future__ import annotations

import numpy as np


def choose_T(p0, v0, p1, vmax=2.5, amax=2.0, T_min=1.0, v1=None):
    """Allocate a conservative segment duration for one waypoint interval.

    ``v1`` is optional for backwards compatibility.  When it is supplied, the
    duration uses the velocity projected along the segment at both endpoints.
    This avoids allocating every interior race segment as if the vehicle stops
    at each gate even when the minimum-snap solve has explicit passthrough
    velocities.
    """

    dp = p1 - p0
    d = np.linalg.norm(dp)

    if d < 1e-6:
        return T_min

    dir_vec = dp / d
    v_along = float(np.dot(v0, dir_vec))

    t_acc = vmax / amax
    d_acc = 0.5 * amax * t_acc**2

    if d > 2 * d_acc:
        T_base = 2 * t_acc + (d - 2 * d_acc) / vmax
    else:
        T_base = 2 * np.sqrt(d / amax)

    if v1 is None:
        if v_along < 0:
            T_base += min(abs(v_along) / amax, 2.0)
        else:
            T_base -= min(v_along / (2 * amax), 0.5)

        return max(T_base, T_min)

    # Boundary-velocity-aware trapezoidal/triangular timing.  Only motion along
    # the segment advances the interval; a tangent velocity at a turning gate
    # naturally contributes its projection to both adjacent segments.
    end_along = float(np.dot(v1, dir_vec))
    reverse_time = 0.0
    effective_distance = float(d)
    if v_along < 0.0:
        reverse_time = min(abs(v_along) / amax, 2.0)
        effective_distance += 0.5 * min(abs(v_along), 2.0 * amax) ** 2 / amax
        v_along = 0.0

    start_speed = float(np.clip(v_along, 0.0, vmax))
    end_speed = float(np.clip(end_along, 0.0, vmax))
    minimum_transition_distance = abs(end_speed**2 - start_speed**2) / (2.0 * amax)
    if effective_distance + 1e-9 < minimum_transition_distance:
        # The requested boundary speeds are not acceleration-feasible over
        # this short interval.  Keep a finite kinematic estimate and let the
        # existing spline validator reduce/reject the offending gate speed.
        average_speed = max(0.5 * (start_speed + end_speed), 1e-3)
        T_base = effective_distance / average_speed
        return max(reverse_time + T_base, T_min)

    peak_sq = amax * effective_distance + 0.5 * (
        start_speed**2 + end_speed**2
    )
    peak_speed = float(np.sqrt(max(peak_sq, 0.0)))
    if peak_speed <= vmax:
        T_base = (
            max(0.0, peak_speed - start_speed)
            + max(0.0, peak_speed - end_speed)
        ) / amax
    else:
        accel_distance = max(0.0, vmax**2 - start_speed**2) / (2.0 * amax)
        decel_distance = max(0.0, vmax**2 - end_speed**2) / (2.0 * amax)
        cruise_distance = max(
            0.0,
            effective_distance - accel_distance - decel_distance,
        )
        T_base = (
            max(0.0, vmax - start_speed) / amax
            + cruise_distance / vmax
            + max(0.0, vmax - end_speed) / amax
        )

    return max(reverse_time + T_base, T_min)


def allocate_segment_times(
    waypoints,
    current_vel,
    vmax=2.5,
    amax=2.0,
    T_min=1.0,
    waypoint_velocities=None,
    terminal_vel=None,
):
    """Allocate per-segment times for the current waypoint horizon.

    If waypoint/terminal velocities are supplied, they are used as the actual
    entry and exit conditions for each segment.  NaN waypoint rows retain the
    legacy zero-velocity assumption for that waypoint.
    """

    waypoints = np.asarray(waypoints, dtype=float)
    times = []
    use_boundary_velocities = waypoint_velocities is not None or terminal_vel is not None
    boundary_velocities = np.zeros_like(waypoints, dtype=float)
    boundary_velocities[0] = np.asarray(current_vel, dtype=float).reshape(3)

    if waypoint_velocities is not None:
        supplied = np.asarray(waypoint_velocities, dtype=float)
        if supplied.shape != waypoints.shape:
            raise ValueError("waypoint_velocities must match waypoints shape")
        finite_rows = np.all(np.isfinite(supplied), axis=1)
        boundary_velocities[finite_rows] = supplied[finite_rows]

    if terminal_vel is not None:
        terminal = np.asarray(terminal_vel, dtype=float).reshape(3)
        if np.all(np.isfinite(terminal)):
            boundary_velocities[-1] = terminal

    for i in range(len(waypoints) - 1):
        p0 = waypoints[i]
        p1 = waypoints[i + 1]

        if use_boundary_velocities:
            v0 = boundary_velocities[i]
            v1 = boundary_velocities[i + 1]
        else:
            v0 = current_vel if i == 0 else np.zeros(3, dtype=float)
            v1 = None

        T = choose_T(
            p0,
            v0,
            p1,
            vmax=vmax,
            amax=amax,
            T_min=T_min,
            v1=v1,
        )
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
