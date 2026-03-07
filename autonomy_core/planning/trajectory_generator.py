import numpy as np


def _norm(v: np.ndarray, eps: float = 1e-9) -> float:
    return float(np.linalg.norm(v) + eps)


def _unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _wrap_pi(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class TrajectoryGenerator:
    """
    Receding-horizon trajectory generator.

    Inputs:
      - current_state: {"pos": (3,), "vel": (3,), "yaw": float}  yaw optional
      - plan from GatePathPlanner.plan()

    Outputs:
      desired_state:
        {
          "pos_d": (3,),
          "vel_d": (3,),
          "acc_d": (3,),
          "yaw_d": float,
          "yaw_rate_d": float,
          "stage": str
        }

    Notes:
      - All vectors are in the same frame as your perception/planner (likely camera/body/world).
      - If you're in camera frame, this still works, but your downstream controller must match that frame.
    """

    def __init__(
        self,
        cruise_speed=2.0,          # m/s nominal forward speed
        max_speed=3.0,             # m/s speed clamp
        max_acc=3.0,               # m/s^2 accel clamp
        slow_radius=6.0,           # m: begin slowing down as you near target
        stop_radius=0.5,           # m: consider "at target"
        yaw_mode="velocity",       # "velocity" or "gate_normal"
        yaw_rate_limit=np.deg2rad(90.0),  # rad/s clamp
        lookahead_time=0.10,       # sec: for position lookahead
    ):
        self.cruise_speed = float(cruise_speed)
        self.max_speed = float(max_speed)
        self.max_acc = float(max_acc)
        self.slow_radius = float(slow_radius)
        self.stop_radius = float(stop_radius)
        self.yaw_mode = str(yaw_mode)
        self.yaw_rate_limit = float(yaw_rate_limit)
        self.lookahead_time = float(lookahead_time)

        # Internal integrator for smooth velocity command
        self._vel_cmd = np.zeros(3, dtype=float)

    def reset(self):
        self._vel_cmd[:] = 0.0

    def step(self, plan: dict, current_state: dict, dt: float) -> dict:
        if plan is None:
            return None

        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dt must be > 0")

        pos = np.asarray(current_state.get("pos", [0, 0, 0]), dtype=float).reshape(3)
        vel = np.asarray(current_state.get("vel", [0, 0, 0]), dtype=float).reshape(3)
        yaw = float(current_state.get("yaw", 0.0))

        stage = plan.get("stage", "UNKNOWN")
        target = np.asarray(plan["target_position"], dtype=float).reshape(3)

        # -----------------------------
        # 1) Compute desired speed profile based on distance
        # -----------------------------
        to_target = target - pos
        dist = _norm(to_target)

        # Scale speed down as you get close (simple, effective)
        # dist >= slow_radius -> cruise_speed
        # dist <= stop_radius -> 0
        if dist <= self.stop_radius:
            speed_des = 0.0
        elif dist >= self.slow_radius:
            speed_des = self.cruise_speed
        else:
            # linear ramp down
            alpha = (dist - self.stop_radius) / max(self.slow_radius - self.stop_radius, 1e-6)
            speed_des = self.cruise_speed * _clamp(alpha, 0.0, 1.0)

        speed_des = _clamp(speed_des, 0.0, self.max_speed)

        dir_des = _unit(to_target)
        vel_des = dir_des * speed_des

        # -----------------------------
        # 2) Acceleration-limited velocity command (smooth)
        # -----------------------------
        # Move commanded velocity toward vel_des with accel limit
        dv = vel_des - self._vel_cmd
        dv_norm = np.linalg.norm(dv)

        max_dv = self.max_acc * dt
        if dv_norm > max_dv and dv_norm > 1e-9:
            dv = dv * (max_dv / dv_norm)

        self._vel_cmd = self._vel_cmd + dv

        # Clamp commanded speed
        vcmd_norm = np.linalg.norm(self._vel_cmd)
        if vcmd_norm > self.max_speed:
            self._vel_cmd = self._vel_cmd * (self.max_speed / vcmd_norm)

        # Desired acceleration (for controllers that use it)
        acc_d = (self._vel_cmd - vel) / max(dt, 1e-6)
        acc_norm = np.linalg.norm(acc_d)
        if acc_norm > self.max_acc:
            acc_d = acc_d * (self.max_acc / acc_norm)

        # -----------------------------
        # 3) Position desired (receding horizon lookahead)
        # -----------------------------
        pos_d = pos + self._vel_cmd * self.lookahead_time

        # Optionally, don't look past the target when close
        if dist < self.slow_radius:
            # keep pos_d on the line, but not beyond target
            proj = np.dot(pos_d - pos, dir_des)
            proj = _clamp(proj, 0.0, dist)
            pos_d = pos + dir_des * proj

        # -----------------------------
        # 4) Yaw desired
        # -----------------------------
        if self.yaw_mode == "gate_normal" and "gate_normal" in plan:
            # Face along gate normal (useful for passing through)
            n = np.asarray(plan["gate_normal"], dtype=float).reshape(3)
            n = _unit(n)
            # yaw from x-y projection
            yaw_d = float(np.arctan2(n[1], n[0]))
        else:
            # Default: face direction of motion (or to_target if stopped)
            heading_vec = self._vel_cmd if np.linalg.norm(self._vel_cmd) > 1e-3 else to_target
            hv = _unit(heading_vec)
            yaw_d = float(np.arctan2(hv[1], hv[0]))

        # Yaw rate command (simple P on angle)
        yaw_err = _wrap_pi(yaw_d - yaw)
        yaw_rate_d = _clamp(yaw_err / max(dt, 1e-6), -self.yaw_rate_limit, self.yaw_rate_limit)

        traj = {
            "pos_d": pos_d,
            "vel_d": self._vel_cmd.copy(),
            "acc_d": acc_d,
            "yaw_d": yaw_d,
            "yaw_rate_d": yaw_rate_d,
            "stage": stage,
            "dist_to_target": dist,
            "speed_des": speed_des,
        }
        # Debug prints
        print("\n------ Trajectory ------")
        print("Current Position:", pos)
        print("Desired Position:", pos_d)
        print("Current Velocity:", vel)
        print("Desired Velocity:", self._vel_cmd.copy())
        print("Desired Speed:", round(speed_des,2))
        print("Desired Acceleration", acc_d)
        print("Desired Yaw:", yaw_d)
        print("Desired Yaw Rate:", yaw_rate_d)
        print("Stage:", stage)
        print("Distance:", round(dist,2))

        return traj