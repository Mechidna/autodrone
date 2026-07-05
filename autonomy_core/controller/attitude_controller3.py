import time

import numpy as np
from dataclasses import dataclass


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def normalize(v, eps=1e-9):
    n = np.linalg.norm(v)
    if n < eps:
        return None
    return v / n


def rotmat_to_euler_zyx(R):
    """
    Returns roll, pitch, yaw using ZYX convention:
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    pitch = np.arcsin(-clamp(R[2, 0], -1.0, 1.0))

    if abs(np.cos(pitch)) < 1e-6:
        # near gimbal lock
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])

    return roll, pitch, yaw


@dataclass
class State:
    pos: np.ndarray   # shape (3,), world frame, z-up
    vel: np.ndarray   # shape (3,), world frame, z-up
    yaw: float        # current yaw [rad]


@dataclass
class Reference:
    pos: np.ndarray   # shape (3,)
    vel: np.ndarray   # shape (3,)
    acc: np.ndarray   # shape (3,)
    yaw: float        # desired yaw [rad]
    yaw_rate: float = 0.0


class RPGHighLevelTracker:
    """
    Minimal RPG-style high-level controller:
      a_des = a_ref + Kp*(p_ref - p) + Kv*(v_ref - v) + g

    Then constructs desired attitude from a_des and desired yaw,
    and outputs roll/pitch/yaw/thrust.

    This is the scalable outer-loop piece. PX4 can keep doing the
    inner attitude stabilization.
    """
    def __init__(
        self,
        mass=1.0,
        gravity=9.81,
        kp=(3.0, 3.0, 4.0),
        kv=(2.2, 2.2, 2.8),
        max_tilt_deg=25.0,
        max_acc_xy=4.0,
        max_acc_z_up=4.0,
        max_acc_z_down=3.0,
        thrust_hover=0.5,
        thrust_min=0.30,
        thrust_max=0.80,
        thrust_from_acc_gain=None,
        lateral_accel_gain_xy=(1.0, 1.0),
        max_acc_z_slew_m_s3=0.0,
        max_acc_z_slew_reset_s=0.5,
        near_reference_z_error_m=0.0,
        near_reference_vz_error_max_m_s=0.0,
        near_reference_max_acc_z_up=0.0,
        near_reference_max_acc_z_down=0.0,
    ):
        self.m = float(mass)
        self.g = float(gravity)

        self.Kp = np.array(kp, dtype=float)
        self.Kv = np.array(kv, dtype=float)

        self.max_tilt = np.deg2rad(max_tilt_deg)
        self.max_acc_xy = float(max_acc_xy)
        self.max_acc_z_up = float(max_acc_z_up)
        self.max_acc_z_down = float(max_acc_z_down)

        self.thrust_hover = float(thrust_hover)
        self.thrust_min = float(thrust_min)
        self.thrust_max = float(thrust_max)

        # If thrust is normalized and roughly linear around hover:
        # thrust ≈ hover + gain * (a_z / g)
        self.thrust_from_acc_gain = (
            1.0 / self.g if thrust_from_acc_gain is None else float(thrust_from_acc_gain)
        )
        self.lateral_accel_gain_xy = np.asarray(
            lateral_accel_gain_xy,
            dtype=float,
        ).reshape(2)
        self.max_acc_z_slew_m_s3 = max(0.0, float(max_acc_z_slew_m_s3))
        self.max_acc_z_slew_reset_s = max(0.0, float(max_acc_z_slew_reset_s))
        self.near_reference_z_error_m = max(0.0, float(near_reference_z_error_m))
        self.near_reference_vz_error_max_m_s = max(
            0.0,
            float(near_reference_vz_error_max_m_s),
        )
        self.near_reference_max_acc_z_up = max(
            0.0,
            float(near_reference_max_acc_z_up),
        )
        self.near_reference_max_acc_z_down = max(
            0.0,
            float(near_reference_max_acc_z_down),
        )
        self._last_acc_z_cmd = None
        self._last_update_time = None

    def reset_vertical_limiter(self):
        self._last_acc_z_cmd = None
        self._last_update_time = None

    def _limit_acceleration(self, a_cmd_no_g, *, near_reference=False):
        """
        Limit commanded translational acceleration before gravity is added.
        This is more practical than limiting total a_des directly.
        """
        a = a_cmd_no_g.copy()

        # Horizontal limit
        a_xy = a[:2]
        n_xy = np.linalg.norm(a_xy)
        if n_xy > self.max_acc_xy:
            a[:2] = a_xy * (self.max_acc_xy / n_xy)

        # Vertical limit
        max_acc_z_up = self.max_acc_z_up
        max_acc_z_down = self.max_acc_z_down
        if near_reference:
            if self.near_reference_max_acc_z_up > 0.0:
                max_acc_z_up = min(max_acc_z_up, self.near_reference_max_acc_z_up)
            if self.near_reference_max_acc_z_down > 0.0:
                max_acc_z_down = min(max_acc_z_down, self.near_reference_max_acc_z_down)
        if a[2] > max_acc_z_up:
            a[2] = max_acc_z_up
        if a[2] < -max_acc_z_down:
            a[2] = -max_acc_z_down

        return a

    def _slew_limit_vertical_acceleration(self, a_cmd_no_g):
        a = a_cmd_no_g.copy()
        now = time.monotonic()
        previous = self._last_acc_z_cmd
        previous_time = self._last_update_time
        limited = False

        if (
            previous is not None
            and previous_time is not None
            and self.max_acc_z_slew_m_s3 > 0.0
        ):
            dt = now - previous_time
            if dt > 0.0 and (
                self.max_acc_z_slew_reset_s <= 0.0
                or dt <= self.max_acc_z_slew_reset_s
            ):
                max_delta = self.max_acc_z_slew_m_s3 * dt
                before = float(a[2])
                a[2] = clamp(a[2], previous - max_delta, previous + max_delta)
                limited = abs(float(a[2]) - before) > 1e-9

        self._last_acc_z_cmd = float(a[2])
        self._last_update_time = now
        return a, limited

    def _construct_R_des(self, z_b_des, yaw_des):
        """
        Build desired orientation from desired body z-axis and yaw.
        Same idea as RPG Section 4.1.2:
        choose heading in world xy plane, then construct orthonormal basis. :contentReference[oaicite:1]{index=1}
        """
        x_c = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0], dtype=float)
        y_c = np.array([-np.sin(yaw_des), np.cos(yaw_des), 0.0], dtype=float)

        x_b_des = normalize(np.cross(y_c, z_b_des))
        if x_b_des is None:
            # singular case: thrust nearly aligned with y_c
            x_b_des = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0], dtype=float)

        y_b_des = normalize(np.cross(z_b_des, x_b_des))
        if y_b_des is None:
            y_b_des = np.array([-np.sin(yaw_des), np.cos(yaw_des), 0.0], dtype=float)

        R_des = np.column_stack((x_b_des, y_b_des, z_b_des))
        return R_des

    def update(self, state: State, ref: Reference):
        # Position / velocity errors
        e_p = ref.pos - state.pos
        e_v = ref.vel - state.vel

        near_reference_vertical = (
            self.near_reference_z_error_m > 0.0
            and abs(float(e_p[2])) <= self.near_reference_z_error_m
        )
        e_v_for_control = e_v.copy()
        vertical_velocity_error_limited = False
        if (
            near_reference_vertical
            and self.near_reference_vz_error_max_m_s > 0.0
        ):
            before = float(e_v_for_control[2])
            e_v_for_control[2] = clamp(
                before,
                -self.near_reference_vz_error_max_m_s,
                self.near_reference_vz_error_max_m_s,
            )
            vertical_velocity_error_limited = abs(float(e_v_for_control[2]) - before) > 1e-9

        # RPG-style high-level acceleration command:
        # a_fb + a_ref + gravity compensation. :contentReference[oaicite:2]{index=2}
        a_fb = self.Kp * e_p + self.Kv * e_v_for_control
        a_cmd_raw_no_g = ref.acc + a_fb

        # Saturate translational demand to something your drone can actually do
        a_cmd_accel_limited_no_g = self._limit_acceleration(
            a_cmd_raw_no_g,
            near_reference=near_reference_vertical,
        )
        a_cmd_no_g, vertical_accel_slew_limited = (
            self._slew_limit_vertical_acceleration(a_cmd_accel_limited_no_g)
        )

        # Calibrate lateral acceleration demand into the attitude command path.
        a_cmd_attitude_no_g = a_cmd_no_g.copy()
        a_cmd_attitude_no_g[:2] = (
            a_cmd_attitude_no_g[:2] * self.lateral_accel_gain_xy
        )

        # Add gravity compensation (z-up world frame)
        a_des = a_cmd_attitude_no_g + np.array([0.0, 0.0, self.g], dtype=float)

        # Desired body z-axis aligns with desired acceleration direction
        z_b_des = normalize(a_des)
        if z_b_des is None:
            z_b_des = np.array([0.0, 0.0, 1.0], dtype=float)

        # Desired attitude
        R_des = self._construct_R_des(z_b_des, ref.yaw)
        roll_des, pitch_des, yaw_des_from_R = rotmat_to_euler_zyx(R_des)

        # Optional tilt clamp as a final safety layer
        roll_des = clamp(roll_des, -self.max_tilt, self.max_tilt)
        pitch_des = clamp(pitch_des, -self.max_tilt, self.max_tilt)

        # Yaw command: use reference yaw directly, not R-derived yaw, for clarity
        yaw_cmd = wrap_pi(ref.yaw)

        # Collective thrust:
        # RPG computes commanded thrust from desired accel projected onto current body z axis. :contentReference[oaicite:3]{index=3}
        # For PX4 normalized thrust, a practical approximation is to map desired vertical accel to normalized thrust.
        thrust_raw_before_clamp = self.thrust_hover + self.thrust_from_acc_gain * a_cmd_no_g[2]
        thrust_cmd = thrust_raw_before_clamp
        thrust_cmd = clamp(thrust_cmd, self.thrust_min, self.thrust_max)

        # r1, p1, y1 = rotmat_to_euler_zyx(R_des)
        # r2, p2, y2 = rotmat_to_euler_zyx(R_des.T)
        #
        # print("R_des   :", np.degrees([r1, p1, y1]))
        # print("R_des.T :", np.degrees([r2, p2, y2]))
        # print("z_b_des :", z_b_des)

        debug = {
            "e_p": e_p,
            "e_v": e_v,
            "e_v_for_control": e_v_for_control,
            "a_ref": ref.acc,
            "a_fb": a_fb,
            "a_cmd_raw_no_g": a_cmd_raw_no_g,
            "a_cmd_accel_limited_no_g": a_cmd_accel_limited_no_g,
            "a_cmd_no_g": a_cmd_no_g,
            "a_cmd_attitude_no_g": a_cmd_attitude_no_g,
            "a_des": a_des,
            "z_b_des": z_b_des,
            "R_des": R_des,
            "yaw_des_from_R": yaw_des_from_R,
            "thrust_raw_before_clamp": thrust_raw_before_clamp,
            "thrust_cmd_after_clamp": thrust_cmd,
            "thrust_limited": bool(thrust_cmd != thrust_raw_before_clamp),
            "hover_thrust": self.thrust_hover,
            "lateral_accel_gain_xy": self.lateral_accel_gain_xy.copy(),
            "near_reference_vertical": bool(near_reference_vertical),
            "vertical_velocity_error_limited": bool(vertical_velocity_error_limited),
            "vertical_accel_slew_limited": bool(vertical_accel_slew_limited),
        }

        return roll_des, pitch_des, yaw_cmd, thrust_cmd, debug
