import math
import time
from dataclasses import dataclass
from typing import Optional

from pymavlink import mavutil


POSITION_HOLD_BASE_MASK = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
)


@dataclass(frozen=True)
class HoverHoldTarget:
    x: float
    y: float
    z: float
    yaw_rad: Optional[float]
    mode: str = "hold"
    source: str = "mavlink"


class HoverHold:
    def __init__(
        self,
        stale_after_s: float = 1.0,
        takeoff_alt_m: float = 1.5,
        takeoff_window_s: float = 10.0,
        stationary_speed_m_s: float = 0.05,
        near_ground_abs_z_m: float = 0.25,
    ):
        self.stale_after_s = float(stale_after_s)
        self.takeoff_alt_m = float(takeoff_alt_m)
        self.takeoff_window_s = float(takeoff_window_s)
        self.stationary_speed_m_s = float(stationary_speed_m_s)
        self.near_ground_abs_z_m = float(near_ground_abs_z_m)
        self.target: Optional[HoverHoldTarget] = None

    def reset(self):
        self.target = None

    def update_and_send(self, mavlink_conn, system_boot_ms, data) -> bool:
        if self.target is None:
            startup_elapsed_s = time.time() - (float(system_boot_ms) / 1000.0)
            self.target = self._capture_target(data, startup_elapsed_s)

            if self.target is None:
                return False

            print(
                "hover_hold captured:",
                f"mode={self.target.mode}",
                f"source={self.target.source}",
                f"x={self.target.x:.2f}",
                f"y={self.target.y:.2f}",
                f"z={self.target.z:.2f}",
                f"yaw={self.target.yaw_rad:.2f}" if self.target.yaw_rad is not None else "yaw=ignored",
                flush=True,
            )

        self._send_position_hold(mavlink_conn, system_boot_ms, self.target)
        return True

    def _capture_target(self, data, startup_elapsed_s: float) -> Optional[HoverHoldTarget]:
        lock = data.get("lock") if isinstance(data, dict) else None

        if lock is not None:
            with lock:
                odometry = data.get("odometry")
                local_position_ned = data.get("local_position_ned")
                state_estimate = data.get("latest_state_estimate")
                attitude = data.get("attitude")
                active_track_count = data.get("latest_autonomy_active_track_count")
                track_gates = data.get("track_gates")
        else:
            odometry = data.get("odometry")
            local_position_ned = data.get("local_position_ned")
            state_estimate = data.get("latest_state_estimate")
            attitude = data.get("attitude")
            active_track_count = data.get("latest_autonomy_active_track_count")
            track_gates = data.get("track_gates")

        estimate = self._fresh_estimator_state(state_estimate)
        source = "estimator" if estimate is not None else "mavlink"
        if estimate is not None:
            position, velocity, yaw_rad = estimate
        else:
            position = self._fresh_position_ned(odometry)
            if position is None:
                position = self._fresh_position_ned(local_position_ned)
            velocity = self._fresh_velocity_ned(odometry)
            if velocity is None:
                velocity = self._fresh_velocity_ned(local_position_ned)
            yaw_rad = self._fresh_yaw(attitude)
        if position is None:
            return None

        z = position[2]
        mode = "hold"

        if self._should_takeoff(
            position=position,
            velocity=velocity,
            startup_elapsed_s=startup_elapsed_s,
            active_track_count=active_track_count,
            track_gates=track_gates,
        ):
            z = position[2] - self.takeoff_alt_m
            mode = "takeoff_hold"

        return HoverHoldTarget(
            x=position[0],
            y=position[1],
            z=z,
            yaw_rad=yaw_rad,
            mode=mode,
            source=source,
        )

    def _fresh_estimator_state(
        self,
        state_estimate,
    ) -> Optional[tuple[tuple[float, float, float], Optional[tuple[float, float, float]], Optional[float]]]:
        if not isinstance(state_estimate, dict) or self._is_stale(state_estimate):
            return None
        if not bool(state_estimate.get("valid", False)):
            return None
        if str(state_estimate.get("source", "")).lower() != "estimator":
            return None

        position = self._tuple3(state_estimate.get("pos_ned"))
        if position is None:
            pos_neu = self._tuple3(state_estimate.get("pos_neu"))
            if pos_neu is not None:
                position = (pos_neu[0], pos_neu[1], -pos_neu[2])
        if position is None:
            return None

        velocity = self._tuple3(state_estimate.get("vel_ned"))
        if velocity is None:
            vel_neu = self._tuple3(state_estimate.get("vel_neu"))
            if vel_neu is not None:
                velocity = (vel_neu[0], vel_neu[1], -vel_neu[2])

        yaw_rad = None
        try:
            yaw = float(state_estimate.get("yaw_rad"))
        except (TypeError, ValueError):
            yaw = float("nan")
        if math.isfinite(yaw):
            yaw_rad = yaw

        return position, velocity, yaw_rad

    @staticmethod
    def _tuple3(value) -> Optional[tuple[float, float, float]]:
        if value is None:
            return None
        try:
            if len(value) != 3:
                return None
            out = tuple(float(item) for item in value)
        except (TypeError, ValueError):
            return None
        if all(math.isfinite(item) for item in out):
            return out
        return None

    def _fresh_position_ned(self, source) -> Optional[tuple[float, float, float]]:
        if not isinstance(source, dict) or self._is_stale(source):
            return None

        pos_ned = source.get("pos_ned")
        if pos_ned is not None and len(pos_ned) == 3:
            position = tuple(float(value) for value in pos_ned)
        elif all(key in source for key in ("x", "y", "z")):
            position = (
                float(source["x"]),
                float(source["y"]),
                float(source["z"]),
            )
        else:
            return None

        if all(math.isfinite(value) for value in position):
            return position
        return None

    def _fresh_velocity_ned(self, source) -> Optional[tuple[float, float, float]]:
        if not isinstance(source, dict) or self._is_stale(source):
            return None

        vel_ned = source.get("vel_ned")
        if vel_ned is not None and len(vel_ned) == 3:
            velocity = tuple(float(value) for value in vel_ned)
        elif all(key in source for key in ("vx", "vy", "vz")):
            velocity = (
                float(source["vx"]),
                float(source["vy"]),
                float(source["vz"]),
            )
        else:
            return None

        if all(math.isfinite(value) for value in velocity):
            return velocity
        return None

    def _should_takeoff(
        self,
        *,
        position: tuple[float, float, float],
        velocity: Optional[tuple[float, float, float]],
        startup_elapsed_s: float,
        active_track_count,
        track_gates,
    ) -> bool:
        return (
            self._no_active_tracks(active_track_count, track_gates)
            and self._is_stationary(velocity)
            and abs(float(position[2])) <= self.near_ground_abs_z_m
            and 0.0 <= float(startup_elapsed_s) < self.takeoff_window_s
        )

    def _no_active_tracks(self, active_track_count, track_gates) -> bool:
        if track_gates:
            return False
        if active_track_count is not None:
            try:
                return int(active_track_count) <= 0
            except (TypeError, ValueError):
                return False
        return True

    def _is_stationary(self, velocity: Optional[tuple[float, float, float]]) -> bool:
        if velocity is None:
            return False
        speed = math.sqrt(sum(float(value) * float(value) for value in velocity))
        return speed <= self.stationary_speed_m_s

    def _fresh_yaw(self, attitude) -> Optional[float]:
        if not isinstance(attitude, dict) or self._is_stale(attitude):
            return None

        yaw = attitude.get("yaw")
        if yaw is None:
            return None

        yaw_rad = float(yaw)
        if math.isfinite(yaw_rad):
            return yaw_rad
        return None

    def _is_stale(self, source) -> bool:
        wall_time = source.get("wall_time")
        if wall_time is None:
            return False
        return time.time() - float(wall_time) > self.stale_after_s

    def _send_position_hold(self, mavlink_conn, system_boot_ms, target: HoverHoldTarget):
        now_ms = int(time.time() * 1000)
        yaw_rad = 0.0 if target.yaw_rad is None else target.yaw_rad
        type_mask = POSITION_HOLD_BASE_MASK

        if target.yaw_rad is None:
            type_mask |= mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE

        mavlink_conn.mav.set_position_target_local_ned_send(
            now_ms - system_boot_ms,
            mavlink_conn.target_system,
            mavlink_conn.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            target.x,
            target.y,
            target.z,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            yaw_rad,
            0.0,
        )
