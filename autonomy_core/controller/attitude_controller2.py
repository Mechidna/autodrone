import numpy as np


class DroneAttitudeController:
    def __init__(self, kp=1.5, kd=0.5, ki=0.1,
                 kp_z=0.25, kd_z=0.20, ki_z=0.05,
                 hover_thrust=0.55):
        self.kp = kp
        self.kd = kd
        self.ki = ki

        # Z gains
        self.kp_z = kp_z
        self.kd_z = kd_z
        self.ki_z = ki_z

        # Adaptive hover thrust
        self.hover_thrust = hover_thrust
        self.k_hover = 0.05
        self.hover_min = 0.25
        self.hover_max = 0.80

        self.g = 9.81

        # XY integrator
        self.error_i = np.array([0.0, 0.0])
        self.max_i = 2.0

        # Z integrator
        self.error_iz = 0.0
        self.max_iz = 0.25

        self.max_tilt = np.radians(10)
        self.max_thrust = 0.79
        self.min_thrust = 0.05

        # For vertical acceleration estimate
        self._vz_prev = None
        self._az_filt = 0.0
        self._az_alpha = 0.2

    def reset(self):
        self.error_i[:] = 0.0
        self.error_iz = 0.0
        self._vz_prev = None
        self._az_filt = 0.0

    def get_tilt_commands(
        self,
        target_p, target_v, target_a,
        actual_p, actual_v,
        actual_roll, actual_pitch, actual_yaw,
        target_yaw,
        dt
    ):
        dt = max(1e-3, float(dt))
        # Debug
        print("target_p:", target_p)
        print("target_v:", target_v)
        print("target_a:", target_a)
        print("actual_p:", actual_p)
        print("actual_v:", actual_v)
        print("actual_roll:", actual_roll)
        print("actual_pitch:", actual_pitch)
        print("actual_yaw:", actual_yaw)
        print("target_yaw:", target_yaw)
        # Debug End
        # ----------------------------
        # XY position control
        # ----------------------------
        error_p = target_p[:2] - actual_p[:2]
        error_v = target_v[:2] - actual_v[:2]

        self.error_i += error_p * dt
        self.error_i = np.clip(self.error_i, -self.max_i, self.max_i)

        accel_world_x = (
            target_a[0]
            + self.kp * error_p[0]
            + self.kd * error_v[0]
            + self.ki * self.error_i[0]
        )
        accel_world_y = (
            target_a[1]
            + self.kp * error_p[1]
            + self.kd * error_v[1]
            + self.ki * self.error_i[1]
        )

        cos_y = np.cos(actual_yaw)
        sin_y = np.sin(actual_yaw)

        accel_body_x = accel_world_x * cos_y + accel_world_y * sin_y
        accel_body_y = -accel_world_x * sin_y + accel_world_y * cos_y

        pitch = np.arctan2(accel_body_x, self.g)
        roll = np.arctan2(-accel_body_y, self.g)

        pitch = np.clip(pitch, -self.max_tilt, self.max_tilt)
        roll = np.clip(roll, -self.max_tilt, self.max_tilt)

        # ----------------------------
        # Z control -> thrust (hybrid adaptive version)
        # ----------------------------
        z = actual_p[2]
        vz = actual_v[2]

        error_z = target_p[2] - z
        error_vz = target_v[2] - vz

        # Estimate vertical acceleration from telemetry
        if self._vz_prev is None:
            az = 0.0
        else:
            az = (vz - self._vz_prev) / dt
        self._vz_prev = vz

        self._az_filt = (1.0 - self._az_alpha) * self._az_filt + self._az_alpha * az
        az_f = self._az_filt

        # Tilt compensation for hover adaptation
        tilt_cos = max(0.5, np.cos(actual_roll) * np.cos(actual_pitch))
        az_level_equiv = az_f / tilt_cos

        # Only adapt hover thrust when nearly level
        tilt_mag = np.sqrt(actual_roll**2 + actual_pitch**2)
        near_hover = (
                abs(error_z) < 0.2 and
                abs(vz) < 0.15 and
                abs(target_v[2]) < 0.15 and
                abs(target_a[2]) < 0.5 and
                tilt_mag < np.radians(8.0)
        )

        if near_hover:
            self.hover_thrust -= self.k_hover * az_level_equiv * dt
            self.hover_thrust = np.clip(self.hover_thrust, self.hover_min, self.hover_max)

        # Z integrator
        self.error_iz += error_z * dt
        self.error_iz = np.clip(self.error_iz, -self.max_iz, self.max_iz)

        # Direct throttle law around learned hover thrust
        thrust = (
            self.hover_thrust
            + self.kp_z * error_z
            + self.kd_z * error_vz
            + self.ki_z * self.error_iz
        )

        # Optional feedforward contribution from target vertical accel
        # This is intentionally mild so it does not dominate.
        # thrust += 0.5 * (target_a[2] / self.g) * self.hover_thrust

        # Compensate lost vertical authority when tilted
        thrust = thrust / tilt_cos

        thrust = np.clip(thrust, self.min_thrust, self.max_thrust)
        print("error_z:", error_z)
        print("error_vz:", error_vz)
        print("hover_thrust:", self.hover_thrust)
        print("thrust:", thrust)
        return roll, pitch, target_yaw, thrust, error_z