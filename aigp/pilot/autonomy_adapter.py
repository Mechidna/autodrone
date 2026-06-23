from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np

from autonomy_wrapper import PyAIPilotAutonomyAPI


@dataclass
class AttitudeCommand:
    kind: Literal["attitude"]
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    thrust: float


@dataclass
class AutonomyInputSnapshot:
    # Image
    image_bgr: np.ndarray
    image_shape: tuple
    frame_id: int
    image_sim_time_ns: Optional[int]
    image_wall_time: float

    # Attitude, radians
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    roll_rate_rad_s: float
    pitch_rate_rad_s: float
    yaw_rate_rad_s: float
    attitude_time_boot_ms: Optional[int]
    attitude_wall_time: float

    # IMU
    accel_xyz: np.ndarray
    gyro_xyz: np.ndarray
    imu_time_usec: Optional[int]
    imu_wall_time: float

    # Optional local position/velocity
    pos_ned: Optional[np.ndarray]
    vel_ned: Optional[np.ndarray]
    pos_neu: Optional[np.ndarray]
    vel_neu: Optional[np.ndarray]

    # Optional race track data
    track_gates: Optional[list[dict[str, Any]]]

    # Optional perception landmarks
    latest_perception: Optional[dict[str, Any]]

    # Optional timing
    timesync: Optional[dict[str, Any]]


class AutonomyAdapter:
    def __init__(self):
        self.autonomy = PyAIPilotAutonomyAPI(
            use_perception=True,
            race_gate_count=3,
        )
        self.last_snapshot: Optional[AutonomyInputSnapshot] = None

    @staticmethod
    def _vec3(value, default=None):
        if value is None:
            return default
        arr = np.asarray(value, dtype=float)
        if arr.shape != (3,):
            return default
        return arr

    def build_snapshot(
        self,
        frame,
        attitude,
        imu,
        timesync=None,
        local_position_ned=None,
        odometry=None,
        track_gates=None,
        latest_perception=None,
    ) -> AutonomyInputSnapshot:
        image_bgr = frame["image"]

        accel_xyz = self._vec3(imu.get("accel_xyz"))
        if accel_xyz is None:
            accel_xyz = np.array(
                [
                    imu.get("xacc", 0.0),
                    imu.get("yacc", 0.0),
                    imu.get("zacc", 0.0),
                ],
                dtype=float,
            )

        gyro_xyz = self._vec3(imu.get("gyro_xyz"))
        if gyro_xyz is None:
            gyro_xyz = np.array(
                [
                    imu.get("xgyro", 0.0),
                    imu.get("ygyro", 0.0),
                    imu.get("zgyro", 0.0),
                ],
                dtype=float,
            )

        pos_ned = None
        vel_ned = None
        pos_neu = None
        vel_neu = None
        if odometry is not None:
            pos_ned = self._vec3(odometry.get("pos_ned"))
            vel_ned = self._vec3(odometry.get("vel_ned"))
            pos_neu = self._vec3(odometry.get("pos_neu"))
            vel_neu = self._vec3(odometry.get("vel_neu"))

        elif local_position_ned is not None:
            pos_ned = self._vec3(local_position_ned.get("pos_ned"))
            vel_ned = self._vec3(local_position_ned.get("vel_ned"))
            pos_neu = self._vec3(local_position_ned.get("pos_neu"))
            vel_neu = self._vec3(local_position_ned.get("vel_neu"))

            # Fallback if aliases are missing
            if pos_ned is None:
                pos_ned = np.array(
                    [
                        local_position_ned.get("x", 0.0),
                        local_position_ned.get("y", 0.0),
                        local_position_ned.get("z", 0.0),
                    ],
                    dtype=float,
                )
            if vel_ned is None:
                vel_ned = np.array(
                    [
                        local_position_ned.get("vx", 0.0),
                        local_position_ned.get("vy", 0.0),
                        local_position_ned.get("vz", 0.0),
                    ],
                    dtype=float,
                )
            if pos_neu is None and pos_ned is not None:
                pos_neu = np.array([pos_ned[0], pos_ned[1], -pos_ned[2]], dtype=float)
            if vel_neu is None and vel_ned is not None:
                vel_neu = np.array([vel_ned[0], vel_ned[1], -vel_ned[2]], dtype=float)

        return AutonomyInputSnapshot(
            image_bgr=image_bgr,
            image_shape=tuple(frame.get("shape", image_bgr.shape)),
            frame_id=int(frame.get("frame_id", -1)),
            image_sim_time_ns=frame.get("sim_time_ns"),
            image_wall_time=float(frame.get("wall_time", 0.0)),

            roll_rad=float(attitude.get("roll", 0.0)),
            pitch_rad=float(attitude.get("pitch", 0.0)),
            yaw_rad=float(attitude.get("yaw", 0.0)),
            roll_rate_rad_s=float(attitude.get("rollspeed", 0.0)),
            pitch_rate_rad_s=float(attitude.get("pitchspeed", 0.0)),
            yaw_rate_rad_s=float(attitude.get("yawspeed", 0.0)),
            attitude_time_boot_ms=attitude.get("time_boot_ms"),
            attitude_wall_time=float(attitude.get("wall_time", 0.0)),

            accel_xyz=accel_xyz,
            gyro_xyz=gyro_xyz,
            imu_time_usec=imu.get("time_usec"),
            imu_wall_time=float(imu.get("wall_time", 0.0)),

            pos_ned=pos_ned,
            vel_ned=vel_ned,
            pos_neu=pos_neu,
            vel_neu=vel_neu,

            track_gates=track_gates,
            latest_perception=latest_perception,
            timesync=timesync,
        )

    def update(
        self,
        frame,
        attitude,
        imu,
        timesync=None,
        local_position_ned=None,
        odometry=None,
        track_gates=None,
        latest_perception=None,
    ):
        """
        Platform-neutral autonomy update.

        Works for both:
          PX4/Gazebo:
              frame from ros_camera_rx.py
              telemetry from mavlink_rx.py

          Competition:
              frame from vision_rx.py
              telemetry from mavlink_rx.py
        """

        snapshot = self.build_snapshot(
            frame=frame,
            attitude=attitude,
            imu=imu,
            timesync=timesync,
            local_position_ned=local_position_ned,
            odometry=odometry,
            track_gates=track_gates,
            latest_perception=latest_perception,
        )

        self.last_snapshot = snapshot

        cmd = self.autonomy.update(snapshot)
        if cmd is None:
            return None

        return AttitudeCommand(
            kind="attitude",
            roll_deg=float(np.degrees(cmd.roll_rad)),
            pitch_deg=float(np.degrees(cmd.pitch_rad)),
            yaw_deg=float(np.degrees(cmd.yaw_rad)),
            thrust=float(cmd.thrust),
        )
