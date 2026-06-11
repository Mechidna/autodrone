"""Non-commanding control debug helpers for AutonomyAPI."""

import math

import numpy as np


def _debug_vec(dbg, name):
    value = None if dbg is None else dbg.get(name)
    if value is None:
        return np.full(3, np.nan, dtype=float)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size < 3:
        out = np.full(3, np.nan, dtype=float)
        out[:arr.size] = arr
        return out
    return arr[:3].copy()


def compute_tracker_control_debug_fields(
    telemetry,
    state,
    dbg,
    roll_cmd,
    pitch_cmd,
    thrust_cmd,
    tracker_thrust_hover,
):
    """Return logger-visible tracker debug fields without changing commands."""
    raw_velocity = np.array([
        telemetry.vel["vx"],
        telemetry.vel["vy"],
        telemetry.vel["vz"],
    ], dtype=float)
    velocity_input = np.asarray(state.vel, dtype=float).reshape(3)

    return {
        "tracker_velocity_input": velocity_input.copy(),
        "tracker_velocity_was_sanitized": bool(
            not np.all(np.isfinite(raw_velocity))
            or not np.allclose(
                np.nan_to_num(raw_velocity, nan=0.0, posinf=0.0, neginf=0.0),
                velocity_input,
                equal_nan=False,
            )
        ),
        "tracker_e_p": _debug_vec(dbg, "e_p"),
        "tracker_e_v": _debug_vec(dbg, "e_v"),
        "tracker_a_ref": _debug_vec(dbg, "a_ref"),
        "tracker_a_fb": _debug_vec(dbg, "a_fb"),
        "tracker_a_cmd_raw": _debug_vec(dbg, "a_cmd_raw_no_g"),
        "tracker_a_cmd_limited": _debug_vec(dbg, "a_cmd_no_g"),
        "thrust_raw_before_clamp": float(
            dbg.get("thrust_raw_before_clamp", np.nan) if dbg is not None else np.nan
        ),
        "thrust_cmd_after_clamp": float(
            dbg.get("thrust_cmd_after_clamp", thrust_cmd) if dbg is not None else thrust_cmd
        ),
        "thrust_limited": bool(
            dbg.get("thrust_limited", False) if dbg is not None else False
        ),
        "hover_thrust": float(
            dbg.get("hover_thrust", tracker_thrust_hover)
            if dbg is not None
            else tracker_thrust_hover
        ),
        "vertical_thrust_after_tilt": float(
            thrust_cmd * math.cos(float(roll_cmd)) * math.cos(float(pitch_cmd))
        ),
    }
