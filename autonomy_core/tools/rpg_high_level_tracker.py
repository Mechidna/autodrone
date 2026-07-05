#!/usr/bin/env python3
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
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])

    return roll, pitch, yaw


@dataclass
class State:
    pos: np.ndarray   # shape (3,), world frame, z-up: [north, east, up]
    vel: np.ndarray   # shape (3,), world frame, z-up: [vn, ve, vup]
    yaw: float        # current yaw [rad]


@dataclass
class Reference:
    pos: np.ndarray   # shape (3,), world frame, z-up: [north, east, up]
    vel: np.ndarray   # shape (3,)
    acc: np.ndarray   # shape (3,)
    yaw: float        # desired yaw [rad]
    yaw_rate: float = 0.0


class RPGHighLevelTracker:
    """
    Minimal RPG-style high-level controller.

    Computes:

        a_cmd = a_ref + Kp * position_error + Kv * velocity_error

    Then converts the desired acceleration into desired roll, pitch, yaw,
    and normalized thrust.

    This is an outer-loop controller. PX4 still handles inner attitude/rate
    stabilization.
    """

    def __init__(
        self,
        mass=1.0,
        gravity=9.81,
        kp=(0.8, 0.8, 2.0),
        kv=(1.2, 1.2, 1.5),
        max_tilt_deg=15.0,
        max_acc_xy=1.5,
        max_acc_z_up=2.0,
        max_acc_z_down=1.5,
        thrust_hover=0.74,
        thrust_min=0.30,
        thrust_max=0.85,
        thrust_from_acc_gain=None,
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

        # Normalized thrust approximation:
        # thrust ≈ hover + gain * vertical_accel
        self.thrust_from_acc_gain = (
            1.0 / self.g if thrust_from_acc_gain is None else float(thrust_from_acc_gain)
        )

    def _limit_acceleration(self, a_cmd_no_g):
        """
        Limit commanded translational acceleration before gravity is added.
        """
        a = a_cmd_no_g.copy()

        # Horizontal acceleration limit
        a_xy = a[:2]
        n_xy = np.linalg.norm(a_xy)
        if n_xy > self.max_acc_xy:
            a[:2] = a_xy * (self.max_acc_xy / n_xy)

        # Vertical acceleration limits, z-up
        if a[2] > self.max_acc_z_up:
            a[2] = self.max_acc_z_up
        if a[2] < -self.max_acc_z_down:
            a[2] = -self.max_acc_z_down

        return a

    def _construct_R_des(self, z_b_des, yaw_des):
        """
        Build desired orientation from desired body z-axis and yaw.

        World frame here is z-up:
            x = north
            y = east
            z = up
        """
        x_c = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0], dtype=float)
        y_c = np.array([-np.sin(yaw_des), np.cos(yaw_des), 0.0], dtype=float)

        x_b_des = normalize(np.cross(y_c, z_b_des))
        if x_b_des is None:
            x_b_des = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0], dtype=float)

        y_b_des = normalize(np.cross(z_b_des, x_b_des))
        if y_b_des is None:
            y_b_des = np.array([-np.sin(yaw_des), np.cos(yaw_des), 0.0], dtype=float)

        R_des = np.column_stack((x_b_des, y_b_des, z_b_des))
        return R_des

    def update(self, state: State, ref: Reference):
        e_p = ref.pos - state.pos
        e_v = ref.vel - state.vel

        a_fb = self.Kp * e_p + self.Kv * e_v
        a_cmd_raw_no_g = ref.acc + a_fb

        a_cmd_no_g = self._limit_acceleration(a_cmd_raw_no_g)

        # Add gravity compensation in z-up world frame
        a_des = a_cmd_no_g + np.array([0.0, 0.0, self.g], dtype=float)

        z_b_des = normalize(a_des)
        if z_b_des is None:
            z_b_des = np.array([0.0, 0.0, 1.0], dtype=float)

        R_des = self._construct_R_des(z_b_des, ref.yaw)
        roll_des, pitch_des, yaw_des_from_R = rotmat_to_euler_zyx(R_des)

        # Final tilt safety clamp
        roll_des = clamp(roll_des, -self.max_tilt, self.max_tilt)
        pitch_des = clamp(pitch_des, -self.max_tilt, self.max_tilt)

        yaw_cmd = wrap_pi(ref.yaw)

        thrust_raw_before_clamp = (
            self.thrust_hover + self.thrust_from_acc_gain * a_cmd_no_g[2]
        )
        thrust_cmd = clamp(
            thrust_raw_before_clamp,
            self.thrust_min,
            self.thrust_max,
        )

        debug = {
            "e_p": e_p,
            "e_v": e_v,
            "a_ref": ref.acc,
            "a_fb": a_fb,
            "a_cmd_raw_no_g": a_cmd_raw_no_g,
            "a_cmd_no_g": a_cmd_no_g,
            "a_des": a_des,
            "z_b_des": z_b_des,
            "R_des": R_des,
            "yaw_des_from_R": yaw_des_from_R,
            "thrust_raw_before_clamp": thrust_raw_before_clamp,
            "thrust_cmd_after_clamp": thrust_cmd,
            "thrust_limited": bool(thrust_cmd != thrust_raw_before_clamp),
            "hover_thrust": self.thrust_hover,
        }

        return roll_des, pitch_des, yaw_cmd, thrust_cmd, debug