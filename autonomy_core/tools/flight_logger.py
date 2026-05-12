import csv
import time
import math
import numpy as np


class FlightLogger:
    def __init__(self, filename="flight_log.csv"):
        self.log_file = open(filename, "w", newline="")
        self.writer = csv.writer(self.log_file)

        self.writer.writerow([
            "t",

            # Actual drone state
            "px", "py", "pz",
            "vx", "vy", "vz",
            "ax", "ay", "az",

            # Commanded attitude/thrust
            "roll_cmd_deg",
            "pitch_cmd_deg",
            "yaw_cmd_deg",
            "thrust_cmd",

            # Planner reference
            "p_ref_x", "p_ref_y", "p_ref_z",
            "v_ref_x", "v_ref_y", "v_ref_z",
            "a_ref_x", "a_ref_y", "a_ref_z",

            # Target gate / waypoint
            "target_x", "target_y", "target_z",

            # Debug info
            "tracking_err_x",
            "tracking_err_y",
            "tracking_err_z",
            "tracking_err_norm",
            "mode",
            "active_gate_idx",
        ])

        self.t0 = time.time()
        self.last_t = None
        self.last_vel = None

    def _vec3(self, value):
        """
        Converts list/tuple/np.array/dict/None into a 3-value numpy vector.
        Returns [nan, nan, nan] if unavailable.
        """
        if value is None:
            return np.array([np.nan, np.nan, np.nan], dtype=float)

        if isinstance(value, dict):
            # Supports {"x":..., "y":..., "z":...}
            if all(k in value for k in ["x", "y", "z"]):
                return np.array([value["x"], value["y"], value["z"]], dtype=float)

            # Supports {"north":..., "east":..., "up":...}
            if all(k in value for k in ["north", "east", "up"]):
                return np.array([value["north"], value["east"], value["up"]], dtype=float)

            return np.array([np.nan, np.nan, np.nan], dtype=float)

        arr = np.array(value, dtype=float).reshape(-1)

        if arr.size >= 3:
            return arr[:3]

        return np.array([np.nan, np.nan, np.nan], dtype=float)

    def log(
        self,
        telemetry,
        roll_cmd,
        pitch_cmd,
        yaw_cmd,
        thrust_cmd,
        p_ref=None,
        v_ref=None,
        a_ref=None,
        target=None,
        mode=None,
        active_gate_idx=None,
    ):
        now = time.time()
        t = now - self.t0

        # Your GetTelemetry seems to store pos/vel as dicts
        pos = self._vec3(getattr(telemetry, "pos", None))
        vel = self._vec3(getattr(telemetry, "vel", None))

        if self.last_t is None or self.last_vel is None:
            acc = np.array([0.0, 0.0, 0.0])
        else:
            dt = now - self.last_t
            acc = (vel - self.last_vel) / dt if dt > 1e-6 else np.zeros(3)

        p_ref = self._vec3(p_ref)
        v_ref = self._vec3(v_ref)
        a_ref = self._vec3(a_ref)
        target = self._vec3(target)

        err = p_ref - pos
        err_norm = float(np.linalg.norm(err)) if not np.any(np.isnan(err)) else np.nan

        self.writer.writerow([
            t,

            pos[0], pos[1], pos[2],
            vel[0], vel[1], vel[2],
            acc[0], acc[1], acc[2],

            math.degrees(roll_cmd),
            math.degrees(pitch_cmd),
            math.degrees(yaw_cmd),
            thrust_cmd,

            p_ref[0], p_ref[1], p_ref[2],
            v_ref[0], v_ref[1], v_ref[2],
            a_ref[0], a_ref[1], a_ref[2],

            target[0], target[1], target[2],

            err[0], err[1], err[2],
            err_norm,
            mode,
            active_gate_idx,
        ])

        self.log_file.flush()

        self.last_t = now
        self.last_vel = vel.copy()

    def close(self):
        self.log_file.close()