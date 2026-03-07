import numpy as np


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _wrap_pi(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _rot_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body-to-world rotation matrix from roll/pitch/yaw (ZYX convention)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    # Rz(yaw)*Ry(pitch)*Rx(roll)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr           ]
    ], dtype=float)


class ClassicalController:
    """
    Converts desired trajectory (pos/vel/acc/yaw) into classical commands:
      throttle, roll, pitch, yaw (or yaw_rate).

    You must feed telemetry in a CONSISTENT world frame:
      pos, vel in world frame
      roll/pitch/yaw describing body orientation relative to world

    If your sim uses ENU vs NED you must adapt signs accordingly.
    """

    def __init__(
        self,
        # Outer-loop gains (tune these first)
        kp_pos=(2.0, 2.0, 3.0),
        kd_vel=(1.5, 1.5, 2.0),

        # Limits
        max_tilt_deg=35.0,      # max roll/pitch magnitude
        max_yaw_rate_deg=180.0, # only used if yaw_mode="rate"

        # Throttle mapping
        hover_throttle=0.5,     # throttle that approximately hovers
        throttle_accel_gain=0.05,  # how much throttle changes per (m/s^2) vertical accel
        throttle_min=0.0,
        throttle_max=1.0,

        # Yaw interpretation
        yaw_mode="angle",  # "angle" or "rate"

        # Gravity handling
        gravity=9.81,
        world_up_axis=2,   # 2 for z-up (ENU). If NED, you'd use z-down logic.
    ):
        self.kp_pos = np.array(kp_pos, dtype=float)
        self.kd_vel = np.array(kd_vel, dtype=float)

        self.max_tilt = np.deg2rad(float(max_tilt_deg))
        self.max_yaw_rate = np.deg2rad(float(max_yaw_rate_deg))

        self.hover_throttle = float(hover_throttle)
        self.throttle_accel_gain = float(throttle_accel_gain)
        self.throttle_min = float(throttle_min)
        self.throttle_max = float(throttle_max)

        self.yaw_mode = str(yaw_mode)
        self.g = float(gravity)
        self.up_axis = int(world_up_axis)

    def step(self, current_state: dict, traj: dict, dt: float) -> dict:
        """
        current_state:
          {
            "pos": np.array([x,y,z]),
            "vel": np.array([vx,vy,vz]),
            "rpy": np.array([roll,pitch,yaw])   # radians
            # optionally "R": 3x3 rotation (body->world)
          }

        traj:
          {
            "pos_d": np.array([x,y,z]),
            "vel_d": np.array([vx,vy,vz]),
            "acc_d": np.array([ax,ay,az]),
            "yaw_d": float,
            "yaw_rate_d": float
          }
        """
        dt = float(dt)
        if dt <= 0:
            raise ValueError("dt must be > 0")

        pos = np.asarray(current_state["pos"], dtype=float).reshape(3)
        vel = np.asarray(current_state["vel"], dtype=float).reshape(3)

        if "R" in current_state:
            R_bw = np.asarray(current_state["R"], dtype=float).reshape(3, 3)  # body->world
            # extract yaw for yaw-hold mode if needed
            yaw = float(np.arctan2(R_bw[1, 0], R_bw[0, 0]))
        else:
            rpy = np.asarray(current_state["rpy"], dtype=float).reshape(3)
            roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
            R_bw = _rot_from_rpy(roll, pitch, yaw)

        pos_d = np.asarray(traj["pos_d"], dtype=float).reshape(3)
        vel_d = np.asarray(traj["vel_d"], dtype=float).reshape(3)
        acc_ff = np.asarray(traj.get("acc_d", [0, 0, 0]), dtype=float).reshape(3)

        yaw_d = float(traj.get("yaw_d", yaw))
        yaw_rate_d = float(traj.get("yaw_rate_d", 0.0))

        # ---------------------------------------
        # 1) Outer-loop: desired acceleration in world frame
        # ---------------------------------------
        e_p = pos_d - pos
        e_v = vel_d - vel

        a_cmd_world = self.kp_pos * e_p + self.kd_vel * e_v + acc_ff

        # ---------------------------------------
        # 2) Throttle from vertical acceleration
        # ---------------------------------------
        # For ENU z-up:
        #   hover when a_cmd_world[z] == 0  (i.e. no extra accel)
        #   increase throttle for +z accel
        az = float(a_cmd_world[self.up_axis])

        throttle = self.hover_throttle + self.throttle_accel_gain * az
        throttle = _clamp(throttle, self.throttle_min, self.throttle_max)

        # ---------------------------------------
        # 3) Roll/Pitch from lateral acceleration + desired yaw
        # ---------------------------------------
        # Standard small-angle mapping (works well up to ~30-40 degrees).
        # For ENU:
        #   ax -> pitch forward/back
        #   ay -> roll left/right
        ax = float(a_cmd_world[0])
        ay = float(a_cmd_world[1])

        # Rotate desired lateral accel into a "yaw-aligned" frame so roll/pitch commands
        # don't fight the yaw controller.
        cy, sy = np.cos(yaw_d), np.sin(yaw_d)

        # accel in yaw frame (x' forward, y' left)
        a_xp =  cy * ax + sy * ay
        a_yp = -sy * ax + cy * ay

        # Map to pitch/roll
        # pitch forward (negative pitch in many conventions) depends on sim convention.
        # This mapping assumes:
        #   +pitch tips nose UP (aircraft), which is opposite of many robotics frames.
        # We'll use a common robotics convention:
        #   pitch_cmd = -a_x / g
        #   roll_cmd  =  a_y / g
        pitch_cmd = _clamp(-a_xp / self.g, -np.tan(self.max_tilt), np.tan(self.max_tilt))
        roll_cmd  = _clamp( a_yp / self.g, -np.tan(self.max_tilt), np.tan(self.max_tilt))

        # Convert from tan(theta) approximation to angles
        pitch_cmd = np.arctan(pitch_cmd)
        roll_cmd  = np.arctan(roll_cmd)

        # ---------------------------------------
        # 4) Yaw command
        # ---------------------------------------
        if self.yaw_mode == "rate":
            yaw_cmd = _clamp(yaw_rate_d, -self.max_yaw_rate, self.max_yaw_rate)
        else:
            # angle mode: command yaw angle
            yaw_cmd = yaw_d  # your sim/controller interprets this as desired yaw angle

        yaw_cmd = yaw_cmd + 90

        cmd = {
            "throttle": float(throttle),
            "roll": float(roll_cmd),
            "pitch": float(pitch_cmd),
            "yaw": float(yaw_cmd),
            "yaw_mode": self.yaw_mode,
            "debug": {
                "e_p": e_p,
                "e_v": e_v,
                "a_cmd_world": a_cmd_world,
            }
        }

        # Debug prints
        print("\n------ Commands ------")
        print("Throttle:", float(throttle))
        print("Roll:", float(roll_cmd))
        print("Pitch:", float(pitch_cmd))
        print("Yaw:", float(yaw_cmd))
        print("Yaw Mode:", self.yaw_mode)
        print("e_p:", e_p)
        print("e_v:", e_v)
        print("a_cmd_world:", a_cmd_world)

        return cmd