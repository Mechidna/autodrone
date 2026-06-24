import time
import threading

from setup import setup_components
from runtime_config import load_runtime_config


CONFIG = load_runtime_config()
RUNNER_MODE = CONFIG.runtime.runner_mode
SIM_SERVER_UDP_IP = CONFIG.mavlink.ip
SIM_SERVER_UDP_PORT = CONFIG.mavlink.port_for_mode(RUNNER_MODE)

print(
    f"Starting main.py with "
    f"RUNNER_MODE={RUNNER_MODE}, "
    f"VISION_SOURCE={CONFIG.vision.source}, "
    f"MAVLINK={SIM_SERVER_UDP_IP}:{SIM_SERVER_UDP_PORT}, "
    f"CONFIG={CONFIG.path}",
    flush=True,
)

system_boot_ms = int(time.time() * 1000)

shared_data = {
    "lock": threading.Lock()
}

components = setup_components(
    shared_data,
    system_boot_ms,
    CONFIG,
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
    if CONFIG.runtime.px4_offboard_enabled:
        print("Priming PX4 Offboard stream...", flush=True)

        for _ in range(CONFIG.runtime.px4_offboard_prime_count):
            update_autonomy_command()
            controller.update()

        print(f"Switching to {CONFIG.runtime.px4_offboard_mode}...", flush=True)
        controller.set_mode(CONFIG.runtime.px4_offboard_mode)

    if CONFIG.runtime.px4_arm:
        print("Arming drone...", flush=True)
        controller.arm()

elif RUNNER_MODE == "competition":
    if CONFIG.runtime.competition_arm:
        print("Competition mode: arming and streaming commands without PX4 OFFBOARD.", flush=True)
        controller.arm()
    else:
        print("Competition mode: streaming commands without PX4 OFFBOARD or arm.", flush=True)


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

            next_print = time.time() + CONFIG.runtime.flow_status_period_s

except KeyboardInterrupt:
    print("Stopping...", flush=True)

    if RUNNER_MODE == "px4":
        # Optional clean PX4 shutdown if your Controller has land().
        if hasattr(controller, "land"):
            try:
                controller.land()
                time.sleep(CONFIG.runtime.shutdown_land_wait_s)
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
            thread.join(timeout=CONFIG.runtime.join_timeout_s)

    print("Client exited!", flush=True)
