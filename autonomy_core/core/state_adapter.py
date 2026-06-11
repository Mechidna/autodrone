"""Adapters from runtime telemetry objects into shared core state types."""

from __future__ import annotations

import numpy as np

from autonomy_core.core.types import VehicleState


def vehicle_state_from_telemetry(telemetry) -> VehicleState:
    """Build a VehicleState from the current GetTelemetry-style object.

    The existing runtime already stores PX4 NED values as internal z-up values
    before they reach AutonomyAPI. This adapter preserves that convention and
    only sanitizes velocity, matching the previous attitude_control behavior.
    """

    pos = np.array(
        [
            telemetry.pos["x"],
            telemetry.pos["y"],
            telemetry.pos["z"],
        ],
        dtype=float,
    )
    vel = np.nan_to_num(
        np.array(
            [
                telemetry.vel["vx"],
                telemetry.vel["vy"],
                telemetry.vel["vz"],
            ],
            dtype=float,
        ),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    yaw = float(telemetry.rpy["yaw"])
    return VehicleState(pos=pos, vel=vel, yaw=yaw)
