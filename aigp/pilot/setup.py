from pymavlink import mavutil
from timesync import TimeSync
from vision_rx import VisionRX
from mavlink_rx import MAVLinkRX
from controller import Controller
from ros_camera_rx import RosCameraRX
from autonomy_adapter import AutonomyAdapter
from perception_adapter import PerceptionAdapter
import os

def setup_components(shared_data, system_boot_ms, server_ip, server_udp_port):
    # -------------------------------
    # Mavlink Connection
    # -------------------------------
    # Start a connection listening on a UDP port
    sim_conn = mavutil.mavlink_connection('udpin:%s:%s' % (server_ip, server_udp_port,))
    print("Waiting for heartbeat...", flush=True)
    sim_conn.wait_heartbeat()
    print(f"Connected to system: {sim_conn.target_system}", flush=True)

    # -------------------------------
    # Setup Mavlink msg receiver
    # -------------------------------
    print("Setting up MAVLink rx...", flush=True)
    mavlink_rx = MAVLinkRX.create_mavlink_rx(sim_conn, shared_data)

    # -------------------------------
    # Timesync request Loop
    # -------------------------------
    print("Setting up Timesync loop...", flush=True)
    # ts_loop = TimeSync(sim_conn, shared_data)
    ts_loop = TimeSync.create_timesync(sim_conn, shared_data)

    # -------------------------------
    # Connect Vision receiver
    # -------------------------------
    vision_source = os.getenv("VISION_SOURCE", "udp").lower()

    if vision_source == "ros":
        print("Setting up ROS2 camera rx...", flush=True)
        vision_rx = RosCameraRX(shared_data, camera_topic="/camera")
    else:
        print("Setting up UDP competition vision rx...", flush=True)
        vision_rx = VisionRX(shared_data)

    # -------------------------------
    # Main control loop
    # -------------------------------
    perception_adapter = PerceptionAdapter(shared_data)
    autonomy_adapter = AutonomyAdapter()
    controller = Controller(sim_conn, shared_data, system_boot_ms)

    return {
        'vision_rx': vision_rx,
        'mavlink_rx': mavlink_rx,
        'ts_loop': ts_loop,
        'sim_conn': sim_conn,
        'controller': controller,
        'autonomy_adapter': autonomy_adapter,
        'perception_adapter': perception_adapter,
    }
