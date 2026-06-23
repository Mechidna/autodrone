#!/usr/bin/env python3
import math
import socket
import struct
import threading
import time

import cv2
import numpy as np
from pymavlink import mavutil


CLIENT_MAVLINK_IP = "127.0.0.1"
CLIENT_MAVLINK_PORT = 14540

CLIENT_VISION_IP = "127.0.0.1"
CLIENT_VISION_PORT = 5600

MAVLINK_HZ = 50
VISION_HZ = 30

running = True


def vision_loop():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    frame_id = 0
    header_format = "<IHHIIQ"
    max_payload = 60000

    while running:
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(
            img,
            f"fake frame {frame_id}",
            (40, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        ok, encoded = cv2.imencode(".jpg", img)
        if not ok:
            continue

        jpeg = encoded.tobytes()
        total_chunks = math.ceil(len(jpeg) / max_payload)
        sim_time_ns = time.time_ns()

        for chunk_id in range(total_chunks):
            start = chunk_id * max_payload
            chunk = jpeg[start:start + max_payload]

            header = struct.pack(
                header_format,
                frame_id,
                chunk_id,
                total_chunks,
                len(jpeg),
                len(chunk),
                sim_time_ns,
            )

            sock.sendto(
                header + chunk,
                (CLIENT_VISION_IP, CLIENT_VISION_PORT),
            )

        frame_id += 1
        time.sleep(1.0 / VISION_HZ)


def mavlink_loop():
    conn = mavutil.mavlink_connection(
        f"udpout:{CLIENT_MAVLINK_IP}:{CLIENT_MAVLINK_PORT}",
        source_system=1,
        source_component=1,
    )

    boot = time.time()
    last_print = 0.0

    print(f"Fake sim sending MAVLink to {CLIENT_MAVLINK_IP}:{CLIENT_MAVLINK_PORT}")

    while running:
        now = time.time()
        time_boot_ms = int((now - boot) * 1000)
        time_boot_us = int((now - boot) * 1_000_000)

        # Simulator -> Client: HEARTBEAT
        conn.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_QUADROTOR,
            mavutil.mavlink.MAV_AUTOPILOT_GENERIC,
            0,
            0,
            mavutil.mavlink.MAV_STATE_ACTIVE,
        )

        # Simulator -> Client: ATTITUDE
        conn.mav.attitude_send(
            time_boot_ms,
            0.0,  # roll
            0.0,  # pitch
            0.0,  # yaw
            0.0,  # rollspeed
            0.0,  # pitchspeed
            0.0,  # yawspeed
        )

        # Simulator -> Client: HIGHRES_IMU
        conn.mav.highres_imu_send(
            time_boot_us,
            0.0, 0.0, -9.81,   # xacc, yacc, zacc
            0.0, 0.0, 0.0,     # gyro
            0.0, 0.0, 0.0,     # mag
            1013.25,           # abs_pressure
            0.0,               # diff_pressure
            0.0,               # pressure_alt
            25.0,              # temperature
            0,                 # fields_updated
        )

        # Receive client commands
        while True:
            msg = conn.recv_match(blocking=False)
            if msg is None:
                break

            msg_type = msg.get_type()

            if msg_type == "SET_POSITION_TARGET_LOCAL_NED":
                if now - last_print > 0.25:
                    print(
                        "RX SET_POSITION_TARGET_LOCAL_NED "
                        f"frame={msg.coordinate_frame} "
                        f"mask={msg.type_mask} "
                        f"pos=({msg.x:.2f},{msg.y:.2f},{msg.z:.2f}) "
                        f"vel=({msg.vx:.2f},{msg.vy:.2f},{msg.vz:.2f})"
                    )
                    last_print = now

            elif msg_type == "SET_ATTITUDE_TARGET":
                if now - last_print > 0.25:
                    print(
                        "RX SET_ATTITUDE_TARGET "
                        f"mask={msg.type_mask} "
                        f"q={[round(v, 3) for v in msg.q]} "
                        f"rates=({msg.body_roll_rate:.2f},"
                        f"{msg.body_pitch_rate:.2f},"
                        f"{msg.body_yaw_rate:.2f}) "
                        f"thrust={msg.thrust:.2f}"
                    )
                    last_print = now

            elif msg_type == "TIMESYNC":
                # Basic response: echo client timestamp in ts1, put sim time in tc1
                conn.mav.timesync_send(
                    int(time.time_ns()),
                    msg.tc1,
                )

            elif msg_type == "COMMAND_LONG":
                print(
                    "RX COMMAND_LONG "
                    f"command={msg.command} "
                    f"param1={msg.param1}"
                )

        time.sleep(1.0 / MAVLINK_HZ)


def main():
    global running

    vt = threading.Thread(target=vision_loop, daemon=True)
    mt = threading.Thread(target=mavlink_loop, daemon=True)

    vt.start()
    mt.start()

    print("Fake competition sim running. Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        running = False
        print("Stopping fake sim...")


if __name__ == "__main__":
    main()