import os
import time
import threading

from setup import setup_components


RUNNER_MODE = os.getenv("RUNNER_MODE", "px4").lower()

if RUNNER_MODE not in ("px4", "competition"):
    raise RuntimeError(
        f"Invalid RUNNER_MODE={RUNNER_MODE}. "
        "Use RUNNER_MODE=px4 or RUNNER_MODE=competition."
    )

SIM_SERVER_UDP_IP = os.getenv("MAVLINK_IP", "127.0.0.1")

if RUNNER_MODE == "px4":
    default_udp_port = 14540
else:
    default_udp_port = 14550

SIM_SERVER_UDP_PORT = int(os.getenv("MAVLINK_PORT", str(default_udp_port)))

print(
    f"Starting main.py with "
    f"RUNNER_MODE={RUNNER_MODE}, "
    f"VISION_SOURCE={os.getenv('VISION_SOURCE', 'udp')}, "
    f"MAVLINK={SIM_SERVER_UDP_IP}:{SIM_SERVER_UDP_PORT}",
    flush=True,
)

system_boot_ms = int(time.time() * 1000)

shared_data = {
    "lock": threading.Lock()
}

components = setup_components(
    shared_data,
    system_boot_ms,
    SIM_SERVER_UDP_IP,
    SIM_SERVER_UDP_PORT,
)

controller = components["controller"]
ts_loop = components["ts_loop"]
mavlink_rx = components["mavlink_rx"]
vision_rx = components["vision_rx"]
autonomy_adapter = components["autonomy_adapter"]
perception_adapter = components["perception_adapter"]


def update_autonomy_command():
    lock = shared_data.get("lock")

    with lock:
        frame = shared_data.get("latest_frame")
        attitude = shared_data.get("attitude")
        imu = shared_data.get("highres_imu")
        timesync = shared_data.get("timesync")
        local_position_ned = shared_data.get("local_position_ned")
        odometry = shared_data.get("odometry")
        track_gates = shared_data.get("track_gates")
        latest_perception = shared_data.get("latest_perception")

    if frame is None or attitude is None or imu is None:
        with lock:
            shared_data["latest_autonomy_command"] = None
            shared_data["latest_autonomy_command_wall_time"] = time.time()
            shared_data["latest_autonomy_command_status"] = "missing_inputs"
            shared_data["latest_autonomy_active_track_count"] = 0
        return

    active_track_count = 0
    try:
        cmd = autonomy_adapter.update(
            frame=frame,
            attitude=attitude,
            imu=imu,
            timesync=timesync,
            local_position_ned=local_position_ned,
            odometry=odometry,
            track_gates=track_gates,
            latest_perception=latest_perception,
        )
        active_track_count = int(
            getattr(autonomy_adapter.autonomy, "active_track_count", 0)
        )
        status = "ok"
    except Exception as exc:
        cmd = None
        status = f"error:{exc}"

    with lock:
        shared_data["latest_autonomy_command"] = cmd
        shared_data["latest_autonomy_command_wall_time"] = time.time()
        shared_data["latest_autonomy_command_status"] = status
        shared_data["latest_autonomy_active_track_count"] = active_track_count


if RUNNER_MODE == "px4":
    print("Priming PX4 Offboard stream...", flush=True)

    for _ in range(100):  # 2 seconds at 50 Hz
        update_autonomy_command()
        controller.update()

    print("Switching to OFFBOARD...", flush=True)
    controller.set_mode("OFFBOARD")

    print("Arming drone...", flush=True)
    controller.arm()

elif RUNNER_MODE == "competition":
    # Competition/fake-sim mode should not use PX4 OFFBOARD.
    # The pilot includes arm(), so keep it here for compatibility.
    # If the real competition sim ignores this, that is okay.
    print("Competition mode: arming and streaming commands without PX4 OFFBOARD.", flush=True)
    controller.arm()


print("Starting control loop...", flush=True)

try:
    next_print = time.time()

    while True:
        update_autonomy_command()
        controller.update()

        if time.time() >= next_print:
            lock = shared_data.get("lock")

            with lock:
                has_attitude = "attitude" in shared_data
                has_imu = "highres_imu" in shared_data
                has_frame = "latest_frame" in shared_data
                has_perception = shared_data.get("latest_perception") is not None
                has_command = shared_data.get("latest_autonomy_command") is not None
                command_status = shared_data.get("latest_autonomy_command_status", "unknown")
                perception_status = shared_data.get("latest_perception_status", "unknown")
                counts = dict(shared_data.get("mavlink_message_counts", {}))

            print(
                "flow status:",
                has_attitude,
                has_imu,
                has_frame,
                has_perception,
                has_command,
                command_status,
                perception_status,
                counts,
                flush=True,
            )

            next_print = time.time() + 1.0

except KeyboardInterrupt:
    print("Stopping...", flush=True)

    if RUNNER_MODE == "px4":
        # Optional clean PX4 shutdown if your Controller has land().
        if hasattr(controller, "land"):
            try:
                controller.land()
                time.sleep(2.0)
            except Exception as exc:
                print(f"Landing request failed: {exc}", flush=True)

finally:
    for name, component in [
        ("timesync", ts_loop),
        ("mavlink_rx", mavlink_rx),
        ("vision_rx", vision_rx),
        ("perception_adapter", perception_adapter),
    ]:
        thread = component.get_thread_for_join()
        if thread is not None:
            thread.join(timeout=1.0)

    print("Client exited!", flush=True)
