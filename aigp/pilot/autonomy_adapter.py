from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np

from autonomy_wrapper import PyAIPilotAutonomyAPI
from runtime_config import load_runtime_config


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
    position_source: Optional[str]
    position_wall_time: Optional[float]

    # Optional race track data
    track_gates: Optional[list[dict[str, Any]]]

    # Optional perception landmarks
    latest_perception: Optional[dict[str, Any]]

    # Optional timing
    timesync: Optional[dict[str, Any]]

    # Optional vehicle/system state
    armed: Optional[bool]
    heartbeat: Optional[dict[str, Any]]


class AutonomyAdapter:
    def __init__(self, config=None):
        self.config = config if config is not None else load_runtime_config()
        self.autonomy = PyAIPilotAutonomyAPI(
            use_perception=self.config.runtime.use_perception,
            race_gate_count=self.config.race.gate_count,
            pass_radius_m=self.config.race.pass_radius_m,
            config=self.config,
        )
        self.last_snapshot: Optional[AutonomyInputSnapshot] = None
        self.latest_state_estimate: Optional[dict[str, Any]] = None

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
        armed=None,
        heartbeat=None,
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
        position_source = None
        position_wall_time = None
        position_sources = (
            (("odometry", odometry), ("local_position_ned", local_position_ned))
            if self.config.telemetry.prefer_odometry
            else (("local_position_ned", local_position_ned), ("odometry", odometry))
        )
        for source_name, source in position_sources:
            if source is None:
                continue
            pos_ned = self._vec3(source.get("pos_ned"))
            vel_ned = self._vec3(source.get("vel_ned"))
            pos_neu = self._vec3(source.get("pos_neu"))
            vel_neu = self._vec3(source.get("vel_neu"))

            # Fallback if aliases are missing
            if pos_ned is None:
                pos_ned = np.array(
                    [
                        source.get("x", 0.0),
                        source.get("y", 0.0),
                        source.get("z", 0.0),
                    ],
                    dtype=float,
                )
            if vel_ned is None:
                vel_ned = np.array(
                    [
                        source.get("vx", 0.0),
                        source.get("vy", 0.0),
                        source.get("vz", 0.0),
                    ],
                    dtype=float,
                )
            if pos_neu is None and pos_ned is not None:
                pos_neu = np.array([pos_ned[0], pos_ned[1], -pos_ned[2]], dtype=float)
            if vel_neu is None and vel_ned is not None:
                vel_neu = np.array([vel_ned[0], vel_ned[1], -vel_ned[2]], dtype=float)
            if pos_ned is not None or pos_neu is not None:
                position_source = source_name
                wall_time = source.get("wall_time")
                position_wall_time = None if wall_time is None else float(wall_time)
                break

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
            position_source=position_source,
            position_wall_time=position_wall_time,

            track_gates=track_gates,
            latest_perception=latest_perception,
            timesync=timesync,
            armed=None if armed is None else bool(armed),
            heartbeat=heartbeat if isinstance(heartbeat, dict) else None,
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
        armed=None,
        heartbeat=None,
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
            armed=armed,
            heartbeat=heartbeat,
        )

        self.last_snapshot = snapshot

        cmd = self.autonomy.update(snapshot)
        self.latest_state_estimate = self._state_estimate_dict(
            getattr(self.autonomy, "last_state_estimate", None)
        )
        if cmd is None:
            return None

        return AttitudeCommand(
            kind="attitude",
            roll_deg=float(np.degrees(cmd.roll_rad)),
            pitch_deg=float(np.degrees(cmd.pitch_rad)),
            yaw_deg=float(np.degrees(cmd.yaw_rad)),
            thrust=float(cmd.thrust),
        )

    @staticmethod
    def _state_estimate_dict(estimate) -> Optional[dict[str, Any]]:
        if estimate is None:
            return None

        try:
            pos_neu = np.asarray(estimate.pos_neu, dtype=float).reshape(3)
            vel_neu = np.asarray(estimate.vel_neu, dtype=float).reshape(3)
        except (TypeError, ValueError):
            return None

        if not np.all(np.isfinite(pos_neu)):
            return None

        pos_ned = np.array([pos_neu[0], pos_neu[1], -pos_neu[2]], dtype=float)
        vel_ned = np.array([vel_neu[0], vel_neu[1], -vel_neu[2]], dtype=float)
        return {
            "valid": bool(estimate.valid),
            "source": str(estimate.source),
            "confidence": float(estimate.confidence),
            "reason": str(estimate.reason),
            "pos_neu": tuple(float(value) for value in pos_neu),
            "vel_neu": tuple(float(value) for value in vel_neu),
            "pos_ned": tuple(float(value) for value in pos_ned),
            "vel_ned": tuple(float(value) for value in vel_ned),
            "yaw_rad": float(estimate.yaw_rad),
            "wall_time": float(estimate.wall_time),
            "truth_error_m": (
                None
                if estimate.truth_error_m is None
                else float(estimate.truth_error_m)
            ),
            "vision_correction_source": str(
                getattr(estimate, "vision_correction_source", "")
            ),
            "vision_correction_residual_m": (
                None
                if getattr(estimate, "vision_correction_residual_m", None) is None
                else float(estimate.vision_correction_residual_m)
            ),
            "vision_correction_count": int(
                getattr(estimate, "vision_correction_count", 0)
            ),
        }
