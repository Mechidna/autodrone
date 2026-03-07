import time

import numpy as np


class GetTelemetry:
    def __init__(self):
        # Initialize timestamps to 0 so they print immediately the first time
        self.last_vel_time = 0
        self.last_rpy_time = 0
        init_position = (0.0, 0.0, 0.0)
        init_velocity = (0.0, 0.0, 0.0)
        init_orientation = (0.0, 0.0, 0.0)
        p_key = ("x", "y", "z")
        v_key = ("vx", "vy", "vz")
        o_key = ("roll", "pitch", "yaw")
        self.p0 = dict(zip(p_key, map(float, init_position)))
        self.pos = dict(zip(p_key, map(float, init_position)))
        self.vel = dict(zip(v_key, map(float, init_velocity)))
        self.rpy = dict(zip(o_key, map(float, init_orientation)))

    def telemetry_pos(self, position, start_position):
        # Position is usually printed once at the start, so no timer needed here
        self.p0 = {
            "x": float(start_position[0]),
            "y": float(start_position[1]),
            "z": float(start_position[2])
        }
        current_time = time.time()
        self.pos = {
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2])
        }
        if start_position is not None:
            self._print_vel(start_position, prefix="Initial Vel")
        elif current_time - self.last_vel_time >= 1.0:
            self._print_pos(position, prefix="Live Vel")
            self.last_vel_time = current_time

    def telemetry_vel(self, velocity, start_velocity):
        current_time = time.time()
        self.vel = {
            "vx": velocity[0],
            "vy": velocity[1],
            "vz": velocity[2]
        }
        # Priority: If start_velocity is provided, we print it (Initial snapshot)
        # Otherwise, check if 1.0 second has passed for live updates
        if start_velocity is not None:
            self._print_vel(start_velocity, prefix="Initial Vel")
        elif current_time - self.last_vel_time >= 1.0:
            self._print_vel(velocity, prefix="Live Vel")
            self.last_vel_time = current_time

    def telemetry_rpy(self, orientation, start_orientation):
        current_time = time.time()
        self.rpy = {
            "roll": orientation[0],
            "pitch": orientation[1],
            "yaw": orientation[2]
        }
        if start_orientation is not None:
            self._print_rpy(start_orientation, prefix="Initial RPY")
        elif current_time - self.last_rpy_time >= 1.0:
            self._print_rpy(orientation, prefix="Live RPY")
            self.last_rpy_time = current_time

    # Helper methods to keep the logic clean
    def _print_pos(self, data, prefix):
        pos = {"x": round(float(data[0]), 3), "y": round(float(data[1]), 3), "z": round(float(data[2]), 3)}
        print(f"{prefix}: {data}")

    def _print_vel(self, data, prefix):
        vel = {"vx": round(float(data[0]), 3), "vy": round(float(data[1]), 3), "vz": round(float(data[2]), 3)}
        print(f"{prefix}: {vel}")

    def _print_rpy(self, data, prefix):
        rpy = {"roll": round(float(data[0]), 1), "pitch": round(float(data[1]), 1), "yaw": round(float(data[2]), 1)}
        print(f"{prefix}: {rpy}")
