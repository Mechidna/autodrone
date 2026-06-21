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


if RUNNER_MODE == "px4":
    print("Priming PX4 Offboard stream...", flush=True)

    for _ in range(100):  # 2 seconds at 50 Hz
        controller.update()

    print("Switching to OFFBOARD...", flush=True)
    controller.set_mode("OFFBOARD")

    print("Arming drone...", flush=True)
    controller.arm()

elif RUNNER_MODE == "competition":
    # Competition/fake-sim mode should not use PX4 OFFBOARD.
    # The PyAIPilotExample includes arm(), so keep it here for compatibility.
    # If the real competition sim ignores this, that is okay.
    print("Competition mode: arming and streaming commands without PX4 OFFBOARD.", flush=True)
    controller.arm()


print("Starting control loop...", flush=True)

try:
    next_print = time.time()

    while True:
        controller.update()

        if time.time() >= next_print:
            lock = shared_data.get("lock")

            with lock:
                has_attitude = "attitude" in shared_data
                has_imu = "highres_imu" in shared_data
                has_frame = "latest_frame" in shared_data
                counts = dict(shared_data.get("mavlink_message_counts", {}))

            print(
                "flow status:",
                has_attitude,
                has_imu,
                has_frame,
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
    ]:
        thread = component.get_thread_for_join()
        if thread is not None:
            thread.join(timeout=1.0)

    print("Client exited!", flush=True)