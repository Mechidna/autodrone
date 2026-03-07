import math
from dataclasses import dataclass

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi

@dataclass
class HoverCmd:
    roll: float    # rad
    pitch: float   # rad
    yaw: float     # rad (or deg depending on API)
    throttle: float  # 0..1

class SelfTuningHoverENU:
    """
    Uses only classical commands:
      - throttle (0..1)
      - roll, pitch (rad)
      - yaw (rad)

    Uses telemetry:
      - position ENU (x,y,z)
      - velocity ENU (vx,vy,vz)
      - roll, pitch, yaw

    Key idea:
      - estimate vertical acceleration az from vz derivative
      - adapt hover_throttle so az -> 0 when roll/pitch ~ 0
      - altitude loop outputs throttle adjustments around hover_throttle
      - XY position loop outputs desired accel -> tilt (roll/pitch)
    """

    def __init__(self):
        # --- throttle / hover estimation ---
        self.hover_throttle = 0.55     # initial guess
        self.k_hover = 0.05            # slow adaptation rate (per m/s^2)
        self.hover_min, self.hover_max = 0.25, 0.80

        # --- vertical control gains (work well with adaptive hover) ---
        self.kpz = 1.2                 # z position -> throttle
        self.kdz = 0.35                # vz damping -> throttle
        self.kiz = 0.15                # integrator to remove residual bias
        self.iz = 0.0
        self.iz_lim = 0.25

        # --- XY control (position PD -> accel -> tilt) ---
        self.kpxy = 0.8
        self.kdxy = 0.5
        self.max_tilt_deg = 12.0
        self.max_tilt = math.radians(self.max_tilt_deg)
        self.g = 9.81

        # --- throttle limits ---
        self.thr_min, self.thr_max = 0.05, 0.95

        # --- internal state for acceleration estimate ---
        self._vz_prev = None
        self._az_filt = 0.0
        self._az_alpha = 0.2           # low-pass on accel estimate (0..1)

        # targets
        self.x_t = None
        self.y_t = None
        self.z_t = None
        self.yaw_t = None

    def set_target_here(self, x, y, z, yaw):
        self.x_t, self.y_t, self.z_t, self.yaw_t = float(x), float(y), float(z), float(yaw)
        self.iz = 0.0
        self._vz_prev = None

    def update(self, x, y, z, vx, vy, vz, roll, pitch, yaw, dt) -> HoverCmd:
        dt = max(1e-3, float(dt))

        if self.x_t is None:
            # first call: lock target at current position/yaw
            self.set_target_here(x, y, z, yaw)

        # -------------------------
        # 1) Estimate vertical accel az from vz
        # -------------------------
        if self._vz_prev is None:
            az = 0.0
        else:
            az = (vz - self._vz_prev) / dt
        self._vz_prev = vz

        # low-pass filter accel (helps noisy telemetry)
        self._az_filt = (1 - self._az_alpha) * self._az_filt + self._az_alpha * az
        az_f = self._az_filt

        # compensate for tilt reducing vertical component of thrust:
        # when tilted, same throttle yields less vertical accel.
        tilt = math.sqrt(roll*roll + pitch*pitch)
        c = max(0.5, math.cos(tilt))   # avoid division blow-up
        az_level_equiv = az_f / c

        # -------------------------
        # 2) Adapt hover throttle (only when approximately level)
        # -------------------------
        # Idea: if we're level and az != 0, adjust hover_throttle to drive az -> 0
        if tilt < math.radians(8.0):
            self.hover_throttle = clamp(
                self.hover_throttle - self.k_hover * az_level_equiv * dt,
                self.hover_min, self.hover_max
            )

        # -------------------------
        # 3) Altitude hold -> throttle
        # -------------------------
        ez = self.z_t - z
        self.iz = clamp(self.iz + ez * dt, -self.iz_lim, self.iz_lim)

        # throttle around hover_throttle
        thr = (self.hover_throttle
               + self.kpz * ez
               - self.kdz * vz
               + self.kiz * self.iz)

        thr = clamp(thr, self.thr_min, self.thr_max)

        # -------------------------
        # 4) XY hold -> desired accel -> roll/pitch
        # -------------------------
        ex = self.x_t - x
        ey = self.y_t - y

        ax = self.kpxy * ex - self.kdxy * vx
        ay = self.kpxy * ey - self.kdxy * vy

        # Map desired ENU accelerations to pitch/roll.
        # Small-angle approx:
        #   ax ≈  g * pitch
        #   ay ≈ -g * roll
        pitch_cmd = clamp(ax / self.g, -self.max_tilt, self.max_tilt)
        roll_cmd  = clamp(-ay / self.g, -self.max_tilt, self.max_tilt)

        # -------------------------
        # 5) Yaw hold
        # -------------------------
        yaw_err = wrap_pi(self.yaw_t - yaw)
        yaw_cmd = yaw + yaw_err  # direct setpoint (or just yaw_t)
        yaw_cmd = self.yaw_t

        return HoverCmd(roll=roll_cmd, pitch=pitch_cmd, yaw=yaw_cmd, throttle=thr)