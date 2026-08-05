from pymavlink import mavutil
from timesync import TimeSync
from vision_rx import VisionRX
from mavlink_rx import (
    ALTERNATIVE_IMU_MESSAGE_IDS,
    HIGHRES_IMU_MESSAGE_ID,
    MAVLinkRX,
    configure_mavlink_udp_receive_buffer,
    mavlink_message_source_target,
    request_mavlink_message_rate,
)
from controller import Controller
from autonomy_adapter import AutonomyAdapter
from perception_adapter import PerceptionAdapter
from vio_dataset_recorder import VioDatasetRecorder
from live_openvins import LiveOpenVins, RecorderFanout


def setup_components(shared_data, system_boot_ms, config):
    # -------------------------------
    # Mavlink Connection
    # -------------------------------
    # Start a connection listening on a UDP port
    server_ip = config.mavlink.ip
    server_udp_port = config.mavlink.port_for_mode(config.runtime.runner_mode)
    sim_conn = mavutil.mavlink_connection('udpin:%s:%s' % (server_ip, server_udp_port,))
    requested_mavlink_buffer = int(
        config.mavlink.udp_socket_receive_buffer_bytes
    )
    effective_mavlink_buffer = configure_mavlink_udp_receive_buffer(
        sim_conn,
        requested_mavlink_buffer,
    )
    print(
        "MAVLink UDP_RCVBUF "
        f"requested={requested_mavlink_buffer} "
        f"effective={effective_mavlink_buffer}",
        flush=True,
    )
    print("Waiting for heartbeat...", flush=True)
    heartbeat = sim_conn.wait_heartbeat()
    interval_target_system, interval_target_component = (
        mavlink_message_source_target(
            heartbeat,
            fallback_system=config.mavlink.target_system,
            fallback_component=config.mavlink.target_component,
        )
    )
    sim_conn.target_system = int(config.mavlink.target_system)
    sim_conn.target_component = int(config.mavlink.target_component)
    print(
        "Connected: "
        f"control_target={sim_conn.target_system}:{sim_conn.target_component} "
        f"telemetry_request_target={interval_target_system}:"
        f"{interval_target_component}",
        flush=True,
    )

    dataset_vio_recorder = VioDatasetRecorder.from_environment(config=config)
    if dataset_vio_recorder.enabled:
        print(
            f"Recording VIO dataset to: {dataset_vio_recorder.root}",
            flush=True,
        )
        dataset_vio_recorder.record_event(
            "mavlink_udp_socket_config",
            requested_receive_buffer_bytes=requested_mavlink_buffer,
            effective_receive_buffer_bytes=effective_mavlink_buffer,
            control_target_system=int(sim_conn.target_system),
            control_target_component=int(sim_conn.target_component),
            telemetry_request_target_system=interval_target_system,
            telemetry_request_target_component=interval_target_component,
        )
    live_openvins = LiveOpenVins.from_environment(shared_data, config)
    vio_recorder = RecorderFanout(dataset_vio_recorder, live_openvins)

    # -------------------------------
    # Setup Mavlink msg receiver
    # -------------------------------
    print("Setting up MAVLink rx...", flush=True)
    mavlink_rx = MAVLinkRX.create_mavlink_rx(
        sim_conn,
        shared_data,
        config=config,
        vio_recorder=vio_recorder,
    )

    if config.mavlink.request_highres_imu_rate:
        requested_imu_rate_hz = float(config.mavlink.highres_imu_rate_hz)
        try:
            interval_usec = request_mavlink_message_rate(
                sim_conn,
                HIGHRES_IMU_MESSAGE_ID,
                requested_imu_rate_hz,
                target_system=interval_target_system,
                target_component=interval_target_component,
            )
            print(
                "Requested HIGHRES_IMU "
                f"rate={requested_imu_rate_hz:.1f}Hz "
                f"interval={interval_usec}us "
                f"target={interval_target_system}:{interval_target_component}",
                flush=True,
            )
            vio_recorder.record_event(
                "mavlink_message_interval_requested",
                message="HIGHRES_IMU",
                message_id=HIGHRES_IMU_MESSAGE_ID,
                requested_rate_hz=requested_imu_rate_hz,
                interval_usec=interval_usec,
                target_system=interval_target_system,
                target_component=interval_target_component,
            )
        except Exception as exc:
            print(
                f"WARNING: HIGHRES_IMU rate request failed: {exc}",
                flush=True,
            )
            vio_recorder.record_event(
                "mavlink_message_interval_request_failed",
                message="HIGHRES_IMU",
                message_id=HIGHRES_IMU_MESSAGE_ID,
                requested_rate_hz=requested_imu_rate_hz,
                target_system=interval_target_system,
                target_component=interval_target_component,
                error=f"{type(exc).__name__}: {exc}",
            )

    if config.mavlink.request_alternative_imu_rates:
        alternative_rate_hz = float(config.mavlink.alternative_imu_rate_hz)
        for message_name, message_id in ALTERNATIVE_IMU_MESSAGE_IDS.items():
            try:
                interval_usec = request_mavlink_message_rate(
                    sim_conn,
                    message_id,
                    alternative_rate_hz,
                    target_system=interval_target_system,
                    target_component=interval_target_component,
                )
                print(
                    f"Requested {message_name} "
                    f"rate={alternative_rate_hz:.1f}Hz "
                    f"interval={interval_usec}us "
                    f"target={interval_target_system}:"
                    f"{interval_target_component}",
                    flush=True,
                )
                vio_recorder.record_event(
                    "mavlink_message_interval_requested",
                    message=message_name,
                    message_id=message_id,
                    requested_rate_hz=alternative_rate_hz,
                    interval_usec=interval_usec,
                    target_system=interval_target_system,
                    target_component=interval_target_component,
                )
            except Exception as exc:
                print(
                    f"WARNING: {message_name} rate request failed: {exc}",
                    flush=True,
                )
                vio_recorder.record_event(
                    "mavlink_message_interval_request_failed",
                    message=message_name,
                    message_id=message_id,
                    requested_rate_hz=alternative_rate_hz,
                    target_system=interval_target_system,
                    target_component=interval_target_component,
                    error=f"{type(exc).__name__}: {exc}",
                )

    # -------------------------------
    # Timesync request Loop
    # -------------------------------
    print("Setting up Timesync loop...", flush=True)
    # ts_loop = TimeSync(sim_conn, shared_data)
    ts_loop = TimeSync.create_timesync(
        sim_conn,
        shared_data,
        hz=config.timesync.request_hz,
        vio_recorder=vio_recorder,
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
            socket_receive_buffer_bytes=(
                config.vision.udp_socket_receive_buffer_bytes
            ),
            header_format=config.vision.packet_header_format,
            max_pending_frames=config.vision.max_pending_frames,
            stale_frame_timeout_s=config.vision.stale_frame_timeout_s,
            completed_frame_cache_size=(
                config.vision.completed_frame_cache_size
            ),
            completed_frame_cache_ttl_s=(
                config.vision.completed_frame_cache_ttl_s
            ),
            max_jpeg_size_bytes=config.vision.max_jpeg_size_bytes,
            expected_width=config.camera.width,
            expected_height=config.camera.height,
            vio_recorder=vio_recorder,
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
        'vio_recorder': vio_recorder,
        'live_openvins': live_openvins,
    }
