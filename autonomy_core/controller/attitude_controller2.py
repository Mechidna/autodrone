import numpy as np


class DroneAttitudeController:
    def __init__(self, kp=1.5, kd=0.5, ki=0.1):
        self.kp = kp  # Position gain (The "Spring")
        self.kd = kd  # Velocity gain (The "Damper")
        self.ki = ki  # Integral gain (The "Bias Corrector")
        self.g = 9.81  # Gravity (m/s^2)

        # Error accumulation for the Integral term
        self.error_i = np.array([0.0, 0.0])
        self.max_i = 2.0  # Cap the integral to prevent "windup"

        # Safety limits
        self.max_tilt = np.radians(35)  # Max 35 degrees tilt

    def get_tilt_commands(self, target_p, target_v, target_a, actual_p, actual_v, actual_yaw, target_yaw):
        """
        Maps Trajectory + EKF State to Roll, Pitch, Yaw, and Z.
        """
        # 1. Calculate World-Frame Errors
        error_p = target_p[:2] - actual_p[:2]
        error_v = target_v[:2] - actual_v[:2]

        # 2. Update Integral Term (Self-Tuning for wind/weight)
        self.error_i += error_p * 0.005  # Assuming 200Hz (dt=0.005)
        self.error_i = np.clip(self.error_i, -self.max_i, self.max_i)

        # 3. Desired Acceleration in WORLD frame (Feed-Forward + PID)
        accel_world_x = target_a[0] + (self.kp * error_p[0]) + (self.kd * error_v[0]) + (self.ki * self.error_i[0])
        accel_world_y = target_a[1] + (self.kp * error_p[1]) + (self.kd * error_v[1]) + (self.ki * self.error_i[1])

        # 4. COORDINATE TRANSFORMATION (Crucial!)
        # We need to rotate world-frame acceleration into the drone's BODY frame
        # so that 'Pitch' always moves the drone forward relative to its NOSE.
        cos_y = np.cos(actual_yaw)
        sin_y = np.sin(actual_yaw)

        accel_body_x = accel_world_x * cos_y + accel_world_y * sin_y
        accel_body_y = -accel_world_x * sin_y + accel_world_y * cos_y

        # 5. Map Accel to Pitch and Roll
        # atan2 is more robust than simple division for larger angles
        pitch = np.arctan2(accel_body_x, self.g)
        roll = np.arctan2(-accel_body_y, self.g)

        # 6. Safety Clipping
        pitch = np.clip(pitch, -self.max_tilt, self.max_tilt)
        roll = np.clip(roll, -self.max_tilt, self.max_tilt)

        # 7. Target Z (Altitude) and Yaw
        # We pass target_z directly as AirSim handles the vertical PID internally
        target_z = target_p[2]

        return roll, pitch, target_yaw, target_z