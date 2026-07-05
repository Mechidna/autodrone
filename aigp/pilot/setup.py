from pymavlink import mavutil
from timesync import TimeSync
from vision_rx import VisionRX
from mavlink_rx import MAVLinkRX
from controller import Controller
from autonomy_adapter import AutonomyAdapter
from perception_adapter import PerceptionAdapter


def setup_components(shared_data, system_boot_ms, config):
    # -------------------------------
    # Mavlink Connection
    # -------------------------------
    # Start a connection listening on a UDP port
    server_ip = config.mavlink.ip
    server_udp_port = config.mavlink.port_for_mode(config.runtime.runner_mode)
    sim_conn = mavutil.mavlink_connection('udpin:%s:%s' % (server_ip, server_udp_port,))
    print("Waiting for heartbeat...", flush=True)
    sim_conn.wait_heartbeat()
    sim_conn.target_system = int(config.mavlink.target_system)
    sim_conn.target_component = int(config.mavlink.target_component)
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
    ts_loop = TimeSync.create_timesync(
        sim_conn,
        shared_data,
        hz=config.timesync.request_hz,
    )

    # -------------------------------
    # Connect Vision receiver
    # -------------------------------
    vision_source = config.vision.source

    if vision_source == "ros":
        from ros_camera_rx import RosCameraRX

        print("Setting up ROS2 camera rx...", flush=True)
        vision_rx = RosCameraRX(
            shared_data,
            camera_topic=config.vision.ros_camera_topic,
            camera_info_topic=config.vision.ros_camera_info_topic,
        )
    else:
        print("Setting up UDP competition vision rx...", flush=True)
        vision_rx = VisionRX(
            shared_data,
            bind_ip=config.vision.udp_bind_ip,
            port=config.vision.udp_port,
            socket_timeout_s=config.vision.udp_socket_timeout_s,
            recv_bytes=config.vision.udp_recv_bytes,
            header_format=config.vision.packet_header_format,
            max_pending_frames=config.vision.max_pending_frames,
            stale_frame_timeout_s=config.vision.stale_frame_timeout_s,
            max_jpeg_size_bytes=config.vision.max_jpeg_size_bytes,
            expected_width=config.camera.width,
            expected_height=config.camera.height,
        )

    # -------------------------------
    # Main control loop
    # -------------------------------
    perception_adapter = PerceptionAdapter(shared_data, config=config)
    autonomy_adapter = AutonomyAdapter(config=config)
    controller = Controller(sim_conn, shared_data, system_boot_ms, config=config)

    return {
        'vision_rx': vision_rx,
        'mavlink_rx': mavlink_rx,
        'ts_loop': ts_loop,
        'sim_conn': sim_conn,
        'controller': controller,
        'autonomy_adapter': autonomy_adapter,
        'perception_adapter': perception_adapter,
    }
