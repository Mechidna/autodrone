#!/usr/bin/env python3
import argparse
import math
import time

import numpy as np
from pymavlink import mavutil

from rpg_high_level_tracker import RPGHighLevelTracker, State, Reference


RATE_HZ = 20.0
DT = 1.0 / RATE_HZ


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def euler_to_quaternion(roll, pitch, yaw):
    """
    Returns quaternion in MAVLink order: [w, x, y, z]
    roll/pitch/yaw are radians.
    """
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [w, x, y, z]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--conn", default="udpin:0.0.0.0:14540")

    parser.add_argument(
        "--control-mode",
        choices=["position", "attitude", "attitude_pid"],
        default="position",
        help=(
            "position = SET_POSITION_TARGET_LOCAL_NED, "
            "attitude = fixed SET_ATTITUDE_TARGET, "
            "attitude_pid = feedback position controller -> attitude target"
        ),
    )

    # Shared target, expressed as NED horizontal + positive-up altitude
    parser.add_argument("--target-north", type=float, default=0.0)
    parser.add_argument("--target-east", type=float, default=7.0)
    parser.add_argument("--target-up", type=float, default=1.5)

    # Fixed attitude mode
    parser.add_argument("--roll-deg", type=float, default=0.0)
    parser.add_argument("--pitch-deg", type=float, default=-5.0)
    parser.add_argument("--yaw-deg", type=float, default=90.0)
    parser.add_argument(
        "--thrust",
        type=float,
        default=0.74,
        help="Normalized collective thrust, 0..1. Tune carefully in SITL.",
    )
    parser.add_argument(
        "--attitude-seconds",
        type=float,
        default=5.0,
        help="How long to hold fixed attitude command.",
    )

    # Feedback attitude PID mode
    parser.add_argument("--pid-seconds", type=float, default=20.0)
    parser.add_argument("--hover-thrust", type=float, default=0.74)
    parser.add_argument("--max-tilt-deg", type=float, default=15.0)

    # Conservative default gains for SITL smoke test
    parser.add_argument("--kp-xy", type=float, default=0.8)
    parser.add_argument("--kp-z", type=float, default=2.0)
    parser.add_argument("--kv-xy", type=float, default=1.2)
    parser.add_argument("--kv-z", type=float, default=1.5)

    args = parser.parse_args()

    master = mavutil.mavlink_connection(args.conn, source_system=42)

    print("Waiting for PX4 heartbeat...")
    master.wait_heartbeat()
    print(
        f"Connected to PX4 sys={master.target_system} "
        f"comp={master.target_component}"
    )

    boot_time = time.time()

    def now_ms():
        return int((time.time() - boot_time) * 1000)

    # ------------------------------------------------------------
    # Position setpoint sender
    # ------------------------------------------------------------
    pos_only_mask = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
    )

    def send_position_ned(x, y, z):
        master.mav.set_position_target_local_ned_send(
            now_ms(),
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            pos_only_mask,
            x, y, z,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0,
        )

    # ------------------------------------------------------------
    # Attitude setpoint sender
    # ------------------------------------------------------------
    attitude_mask = (
        mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE |
        mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_PITCH_RATE_IGNORE |
        mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_YAW_RATE_IGNORE
    )

    def send_attitude_rad(roll, pitch, yaw, thrust):
        q = euler_to_quaternion(roll, pitch, yaw)

        master.mav.set_attitude_target_send(
            now_ms(),
            master.target_system,
            master.target_component,
            attitude_mask,
            q,
            0.0, 0.0, 0.0,
            clamp(thrust, 0.0, 1.0),
        )

    def send_attitude_deg(roll_deg, pitch_deg, yaw_deg, thrust):
        send_attitude_rad(
            math.radians(roll_deg),
            math.radians(pitch_deg),
            math.radians(yaw_deg),
            thrust,
        )

    # ------------------------------------------------------------
    # PX4 state feedback reader
    # ------------------------------------------------------------
    latest_local_pos = None
    latest_attitude = None

    def poll_px4_state():
        nonlocal latest_local_pos, latest_attitude

        while True:
            msg = master.recv_match(
                type=["LOCAL_POSITION_NED", "ATTITUDE"],
                blocking=False,
            )

            if msg is None:
                break

            msg_type = msg.get_type()

            if msg_type == "LOCAL_POSITION_NED":
                latest_local_pos = msg
            elif msg_type == "ATTITUDE":
                latest_attitude = msg

        if latest_local_pos is None or latest_attitude is None:
            return None

        # PX4 LOCAL_POSITION_NED:
        #   x  = north
        #   y  = east
        #   z  = down
        #   vx = north velocity
        #   vy = east velocity
        #   vz = down velocity
        #
        # Tracker expects z-up:
        #   pos = [north, east, up]
        #   vel = [north, east, up_velocity]
        pos_up = np.array(
            [
                latest_local_pos.x,
                latest_local_pos.y,
                -latest_local_pos.z,
            ],
            dtype=float,
        )

        vel_up = np.array(
            [
                latest_local_pos.vx,
                latest_local_pos.vy,
                -latest_local_pos.vz,
            ],
            dtype=float,
        )

        return State(
            pos=pos_up,
            vel=vel_up,
            yaw=latest_attitude.yaw,
        )

    # ------------------------------------------------------------
    # PX4 mode / arm / land
    # ------------------------------------------------------------
    def set_mode(mode_name):
        modes = master.mode_mapping()
        print("Available modes:", modes)

        if modes is None or mode_name not in modes:
            raise RuntimeError(f"Mode {mode_name} not available")

        print(f"Setting mode: {mode_name}")
        master.set_mode(mode_name)

    def arm():
        print("Arming...")
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1, 0, 0, 0, 0, 0, 0,
        )

    def land():
        print("Landing...")
        try:
            set_mode("AUTO.LAND")
        except Exception:
            master.mav.command_long_send(
                master.target_system,
                master.target_component,
                mavutil.mavlink.MAV_CMD_NAV_LAND,
                0,
                0, 0, 0, 0, 0, 0, 0,
            )

    # ------------------------------------------------------------
    # Stream helpers
    # ------------------------------------------------------------
    def stream_position_for(seconds, x, y, z):
        end = time.time() + seconds
        while time.time() < end:
            send_position_ned(x, y, z)
            poll_px4_state()
            time.sleep(DT)

    def stream_attitude_fixed_for(seconds, roll_deg, pitch_deg, yaw_deg, thrust):
        end = time.time() + seconds
        while time.time() < end:
            send_attitude_deg(roll_deg, pitch_deg, yaw_deg, thrust)
            poll_px4_state()
            time.sleep(DT)

    def stream_attitude_pid_for(seconds, tracker, ref):
        end = time.time() + seconds
        next_print = 0.0

        while time.time() < end:
            state = poll_px4_state()

            if state is None:
                # Keep Offboard alive while waiting for feedback.
                send_attitude_deg(0.0, 0.0, args.yaw_deg, args.hover_thrust)
                time.sleep(DT)
                continue

            roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd, debug = tracker.update(state, ref)

            # Convention bridge:
            #
            # The tracker is written in a z-up world frame.
            # PX4/MAVLink attitude target is aeronautical/body FRD style.
            #
            # These sign flips are a practical starting bridge for this SITL test.
            # Verify signs before tuning gains.
            roll_px4 = -roll_cmd
            pitch_px4 = -pitch_cmd
            yaw_px4 = yaw_cmd

            send_attitude_rad(roll_px4, pitch_px4, yaw_px4, thrust_cmd)

            now = time.time()
            if now >= next_print:
                e_p = debug["e_p"]
                a_cmd = debug["a_cmd_no_g"]

                print(
                    "att_pid "
                    f"pos_neu={np.round(state.pos, 2)} "
                    f"err_neu={np.round(e_p, 2)} "
                    f"a_cmd={np.round(a_cmd, 2)} "
                    f"rpy_px4_deg={np.round(np.degrees([roll_px4, pitch_px4, yaw_px4]), 1)} "
                    f"thrust={thrust_cmd:.2f}"
                )

                next_print = now + 1.0

            time.sleep(DT)

    # ------------------------------------------------------------
    # Main mission
    # ------------------------------------------------------------
    target_ned_x = args.target_north
    target_ned_y = args.target_east
    target_ned_z = -abs(args.target_up)

    target_up_frame = np.array(
        [args.target_north, args.target_east, args.target_up],
        dtype=float,
    )

    try:
        if args.control_mode == "position":
            print("Priming Offboard with position setpoints...")
            stream_position_for(2.0, 0.0, 0.0, target_ned_z)

            set_mode("OFFBOARD")
            time.sleep(0.5)
            arm()

            print(f"Position hold at up={args.target_up:.2f} m...")
            stream_position_for(5.0, 0.0, 0.0, target_ned_z)

            print(
                f"Flying to NED target: "
                f"north={target_ned_x:.2f}, "
                f"east={target_ned_y:.2f}, "
                f"down={target_ned_z:.2f}"
            )
            stream_position_for(8.0, target_ned_x, target_ned_y, target_ned_z)

            print("Holding target...")
            stream_position_for(5.0, target_ned_x, target_ned_y, target_ned_z)

        elif args.control_mode == "attitude":
            print("Priming Offboard with position setpoints for safe takeoff...")
            stream_position_for(2.0, 0.0, 0.0, target_ned_z)

            set_mode("OFFBOARD")
            time.sleep(0.5)
            arm()

            print(f"Taking off/holding at up={args.target_up:.2f} m...")
            stream_position_for(5.0, 0.0, 0.0, target_ned_z)

            print(
                "Switching to fixed attitude setpoint: "
                f"roll={args.roll_deg:.1f} deg, "
                f"pitch={args.pitch_deg:.1f} deg, "
                f"yaw={args.yaw_deg:.1f} deg, "
                f"thrust={args.thrust:.2f}"
            )
            stream_attitude_fixed_for(
                args.attitude_seconds,
                args.roll_deg,
                args.pitch_deg,
                args.yaw_deg,
                args.thrust,
            )

        elif args.control_mode == "attitude_pid":
            ref = Reference(
                pos=target_up_frame,
                vel=np.zeros(3),
                acc=np.zeros(3),
                yaw=math.radians(args.yaw_deg),
                yaw_rate=0.0,
            )

            tracker = RPGHighLevelTracker(
                kp=(args.kp_xy, args.kp_xy, args.kp_z),
                kv=(args.kv_xy, args.kv_xy, args.kv_z),
                max_tilt_deg=args.max_tilt_deg,
                max_acc_xy=1.5,
                max_acc_z_up=2.0,
                max_acc_z_down=1.5,
                thrust_hover=args.hover_thrust,
                thrust_min=0.30,
                thrust_max=0.85,
            )

            print("Waiting for PX4 LOCAL_POSITION_NED and ATTITUDE...")
            while poll_px4_state() is None:
                time.sleep(0.05)

            print("Priming Offboard with neutral attitude setpoints...")
            stream_attitude_fixed_for(
                2.0,
                0.0,
                0.0,
                args.yaw_deg,
                args.hover_thrust,
            )

            set_mode("OFFBOARD")
            time.sleep(0.5)
            arm()

            print(
                f"Flying with attitude feedback to "
                f"north={args.target_north:.2f}, "
                f"east={args.target_east:.2f}, "
                f"up={args.target_up:.2f}"
            )

            stream_attitude_pid_for(args.pid_seconds, tracker, ref)

    finally:
        land()
        time.sleep(2.0)


if __name__ == "__main__":
    main()