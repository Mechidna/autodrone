import time
import threading
import os
import math

from prearm_hold import hold_before_competition_arm
from setup import setup_components
from runtime_config import load_runtime_config


CONFIG = load_runtime_config()
RUNNER_MODE = CONFIG.runtime.runner_mode
OBSERVE_ONLY = bool(CONFIG.runtime.observe_only)
SIM_SERVER_UDP_IP = CONFIG.mavlink.ip
SIM_SERVER_UDP_PORT = CONFIG.mavlink.port_for_mode(RUNNER_MODE)

print(
    f"Starting main.py with "
    f"RUNNER_MODE={RUNNER_MODE}, "
    f"OBSERVE_ONLY={OBSERVE_ONLY}, "
    f"COMPETITION_YAW_INVERTED={CONFIG.runtime.competition_yaw_inverted}, "
    f"CALIBRATION_ONLY={CONFIG.runtime.calibration_only}, "
    f"PERCEPTION_HOLD={CONFIG.runtime.perception_hold}, "
    f"STARTUP_OBSERVATION={CONFIG.runtime.startup_observation_duration_s:.1f}s, "
    f"COMPETITION_PREARM_SENSOR_HOLD="
    f"{CONFIG.runtime.competition_prearm_sensor_hold_s:.1f}s, "
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
vio_recorder = components["vio_recorder"]
LIVE_OPENVINS = os.environ.get("AIGP_LIVE_OPENVINS", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LIVE_OPENVINS_YAW_DEG = float(
    os.environ.get("AIGP_LIVE_OPENVINS_YAW_DEG", "180.0")
)
LIVE_OPENVINS_MAX_YAW_ERROR_DEG = float(
    os.environ.get("AIGP_LIVE_OPENVINS_MAX_YAW_ERROR_DEG", "25.0")
)
LIVE_OPENVINS_MAX_SPEED_M_S = float(
    os.environ.get("AIGP_LIVE_OPENVINS_MAX_SPEED_M_S", "15.0")
)
LIVE_OPENVINS_MAX_POSITION_JUMP_M = float(
    os.environ.get("AIGP_LIVE_OPENVINS_MAX_POSITION_JUMP_M", "2.50")
)
LIVE_OPENVINS_WARN_POSITION_JUMP_M = float(
    os.environ.get("AIGP_LIVE_OPENVINS_WARN_POSITION_JUMP_M", "0.75")
)
_live_openvins_last_position = None


def _wrapped_angle_error_deg(value_deg, reference_deg):
    return (float(value_deg) - float(reference_deg) + 180.0) % 360.0 - 180.0


def enforce_live_openvins_safety():
    """Fail closed before a bad live VIO state can drive the vehicle away."""
    global _live_openvins_last_position

    if not LIVE_OPENVINS or OBSERVE_ONLY:
        return

    calibrator = getattr(
        autonomy_adapter.autonomy,
        "lateral_response_calibration",
        None,
    )
    if (
        calibrator is not None
        and bool(getattr(calibrator, "enabled", True))
        and bool(getattr(calibrator, "completed", False))
        and not bool(getattr(calibrator, "succeeded", False))
    ):
        status = str(getattr(getattr(calibrator, "last_debug", None), "status", "failed"))
        controller.disarm()
        raise RuntimeError(
            "LIVE_OPENVINS lateral consistency calibration failed "
            f"({status}); vehicle disarmed"
        )

    with shared_data["lock"]:
        attitude = shared_data.get("external_vio_attitude")
        position = shared_data.get("external_vio_local_position_ned")
    if not isinstance(attitude, dict) or not isinstance(position, dict):
        return

    yaw_deg = math.degrees(float(attitude.get("yaw", 0.0)))
    yaw_error_deg = abs(
        _wrapped_angle_error_deg(yaw_deg, LIVE_OPENVINS_YAW_DEG)
    )
    velocity = position.get("vel_neu") or position.get("vel_ned")
    speed_m_s = (
        math.sqrt(sum(float(value) ** 2 for value in velocity))
        if velocity is not None
        else 0.0
    )
    pos_neu = position.get("pos_neu")
    openvins_timestamp = position.get("openvins_timestamp")
    position_jump_m = 0.0
    position_jump_limit_m = LIVE_OPENVINS_MAX_POSITION_JUMP_M
    position_jump_dt_s = 0.0
    expected_motion_m = 0.0
    if pos_neu is not None and openvins_timestamp is not None:
        current_position = (
            float(openvins_timestamp),
            tuple(float(value) for value in pos_neu),
            speed_m_s,
        )
        previous_position = _live_openvins_last_position
        if previous_position is not None:
            previous_timestamp, previous_pos_neu, previous_speed_m_s = (
                previous_position
            )
            dt_s = current_position[0] - previous_timestamp
            if dt_s > 0.0:
                position_jump_dt_s = dt_s
                position_jump_m = math.sqrt(
                    sum(
                        (value - previous_value) ** 2
                        for value, previous_value in zip(
                            current_position[1],
                            previous_pos_neu,
                        )
                    )
                )
                expected_motion_m = (
                    max(speed_m_s, previous_speed_m_s) * dt_s
                )
                position_jump_limit_m = max(
                    LIVE_OPENVINS_MAX_POSITION_JUMP_M,
                    expected_motion_m + 0.50,
                )
            if dt_s != 0.0:
                _live_openvins_last_position = current_position
        else:
            _live_openvins_last_position = current_position
    if (
        position_jump_m > LIVE_OPENVINS_WARN_POSITION_JUMP_M
        and position_jump_m <= position_jump_limit_m
    ):
        print(
            "LIVE_OPENVINS position correction warning: "
            f"jump_m={position_jump_m:.2f} "
            f"dt_s={position_jump_dt_s:.3f} "
            f"expected_motion_m={expected_motion_m:.2f} "
            f"hard_limit_m={position_jump_limit_m:.2f}",
            flush=True,
        )
    if (
        yaw_error_deg > LIVE_OPENVINS_MAX_YAW_ERROR_DEG
        or speed_m_s > LIVE_OPENVINS_MAX_SPEED_M_S
        or position_jump_m > position_jump_limit_m
    ):
        controller.disarm()
        raise RuntimeError(
            "LIVE_OPENVINS divergence watchdog tripped; vehicle disarmed: "
            f"yaw_error_deg={yaw_error_deg:.1f} "
            f"speed_m_s={speed_m_s:.1f} "
            f"position_jump_m={position_jump_m:.2f} "
            f"position_jump_limit_m={position_jump_limit_m:.2f} "
            f"position_jump_dt_s={position_jump_dt_s:.3f} "
            f"expected_motion_m={expected_motion_m:.2f}"
        )


def startup_observation_active():
    return bool(
        getattr(
            autonomy_adapter.autonomy,
            "startup_observation_active",
            False,
        )
    )


def startup_observation_command_status():
    observation_status = str(
        getattr(
            autonomy_adapter.autonomy,
            "startup_observation_status",
            "waiting_perception",
        )
    )
    return f"startup_observation_{observation_status}"


def update_autonomy_command():
    lock = shared_data.get("lock")

    with lock:
        frame = shared_data.get("latest_frame")
        attitude = (
            shared_data.get("external_vio_attitude")
            or shared_data.get("attitude")
        )
        imu = shared_data.get("highres_imu")
        timesync = shared_data.get("timesync")
        local_position_ned = (
            shared_data.get("external_vio_local_position_ned")
            or shared_data.get("local_position_ned")
        )
        odometry = shared_data.get("odometry")
        track_gates = shared_data.get("track_gates")
        race_status = shared_data.get("race_status")
        collision = shared_data.get("collision")
        latest_perception = shared_data.get("latest_perception")
        armed = shared_data.get("armed")
        heartbeat = shared_data.get("heartbeat")

    if (
        attitude is None
        or imu is None
        or (
            frame is None
            and not (
                CONFIG.runtime.calibration_only
                or CONFIG.runtime.perception_hold
            )
        )
    ):
        with lock:
            shared_data["latest_autonomy_command"] = None
            shared_data["latest_autonomy_command_wall_time"] = time.time()
            shared_data["latest_autonomy_command_status"] = (
                startup_observation_command_status()
                if startup_observation_active()
                else "missing_inputs"
            )
            shared_data["latest_autonomy_active_track_count"] = 0
        return

    active_track_count = 0
    latest_state_estimate = None
    try:
        cmd = autonomy_adapter.update(
            frame=frame,
            attitude=attitude,
            imu=imu,
            timesync=timesync,
            local_position_ned=local_position_ned,
            odometry=odometry,
            track_gates=track_gates,
            race_status=race_status,
            collision=collision,
            latest_perception=latest_perception,
            armed=armed,
            heartbeat=heartbeat,
        )
        active_track_count = int(
            getattr(autonomy_adapter.autonomy, "active_track_count", 0)
        )
        latest_state_estimate = getattr(autonomy_adapter, "latest_state_estimate", None)
        status = (
            startup_observation_command_status()
            if startup_observation_active()
            else "ok"
        )
    except Exception as exc:
        cmd = None
        latest_state_estimate = getattr(autonomy_adapter, "latest_state_estimate", None)
        status = f"error:{exc}"

    with lock:
        shared_data["latest_autonomy_command"] = cmd
        shared_data["latest_autonomy_command_wall_time"] = time.time()
        shared_data["latest_autonomy_command_status"] = status
        shared_data["latest_autonomy_active_track_count"] = active_track_count
        shared_data["latest_state_estimate"] = latest_state_estimate


if OBSERVE_ONLY:
    print(
        "OBSERVE_ONLY active: skipping offboard priming, mode changes, "
        "arming, autonomy updates, and all controller transmissions.",
        flush=True,
    )
    with shared_data["lock"]:
        shared_data["latest_autonomy_command"] = None
        shared_data["latest_autonomy_command_wall_time"] = time.time()
        shared_data["latest_autonomy_command_status"] = "observe_only"
        shared_data["latest_autonomy_active_track_count"] = 0
elif RUNNER_MODE == "px4":
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
        hold_before_competition_arm(
            CONFIG.runtime.competition_prearm_sensor_hold_s,
            recorder=vio_recorder,
        )
        if LIVE_OPENVINS:
            print(
                "LIVE_OPENVINS: validating synchronized IMU/camera input "
                "before arming...",
                flush=True,
            )
            deadline = time.time() + 12.0
            while time.time() < deadline:
                with shared_data["lock"]:
                    vio_status = shared_data.get(
                        "external_vio_status",
                        "starting",
                    )
                if vio_status in {"waiting_initialization", "tracking"}:
                    print(
                        "LIVE_OPENVINS synchronized sensor input confirmed; "
                        f"prearm status={vio_status}.",
                        flush=True,
                    )
                    break
                print(
                    f"LIVE_OPENVINS prearm status={vio_status}",
                    flush=True,
                )
                time.sleep(0.5)
            else:
                raise RuntimeError(
                    "LIVE_OPENVINS did not receive synchronized camera/IMU "
                    "input before the prearm deadline; refusing to arm"
                )
            # Static initialization needs the first post-stationary acceleration
            # edge. If ATTITUDE telemetry is absent, preserve the verified VQ1
            # startup heading while the level bootstrap command creates it.
            with shared_data["lock"]:
                if shared_data.get("attitude") is None:
                    shared_data["external_vio_attitude"] = {
                        "roll": 0.0,
                        "pitch": 0.0,
                        "yaw": math.pi,
                        "rollspeed": 0.0,
                        "pitchspeed": 0.0,
                        "yawspeed": 0.0,
                        "wall_time": time.time(),
                        "source": "live_openvins_bootstrap",
                    }
        print("Competition mode: arming and streaming commands without PX4 OFFBOARD.", flush=True)
        controller.arm()
        if LIVE_OPENVINS:
            with shared_data["lock"]:
                vio_status = shared_data.get("external_vio_status")
            if vio_status != "tracking":
                print(
                    "LIVE_OPENVINS: applying bounded level-hover bootstrap "
                    "for static initialization...",
                    flush=True,
                )
                bootstrap_deadline = time.time() + 6.0
                while time.time() < bootstrap_deadline:
                    with shared_data["lock"]:
                        vio_position = shared_data.get(
                            "external_vio_local_position_ned"
                        )
                        vio_status = shared_data.get(
                            "external_vio_status",
                            "waiting_initialization",
                        )
                        shared_data["latest_autonomy_command"] = None
                        shared_data["latest_autonomy_command_wall_time"] = (
                            time.time()
                        )
                        shared_data["latest_autonomy_command_status"] = (
                            "live_vio_bootstrap"
                        )
                    if vio_status == "tracking" and vio_position is not None:
                        print(
                            "LIVE_OPENVINS tracking established after "
                            "bootstrap motion.",
                            flush=True,
                        )
                        break
                    controller.update()
                else:
                    controller.disarm()
                    raise RuntimeError(
                        "LIVE_OPENVINS did not initialize during bounded "
                        "bootstrap motion; vehicle disarmed"
                    )
    else:
        print("Competition mode: streaming commands without PX4 OFFBOARD or arm.", flush=True)


print("Starting control loop...", flush=True)

try:
    next_print = time.time()

    while True:
        if not OBSERVE_ONLY:
            update_autonomy_command()
            enforce_live_openvins_safety()
            controller.update()
        else:
            # Keep the receive/record threads alive without ever entering a
            # code path that can emit a flight-control setpoint.
            time.sleep(1.0 / max(1.0, CONFIG.runtime.control_hz))

        if time.time() >= next_print:
            lock = shared_data.get("lock")

            with lock:
                has_attitude = (
                    "external_vio_attitude" in shared_data
                    or "attitude" in shared_data
                )
                has_imu = "highres_imu" in shared_data
                has_frame = "latest_frame" in shared_data
                has_perception = shared_data.get("latest_perception") is not None
                has_command = shared_data.get("latest_autonomy_command") is not None
                command_status = shared_data.get("latest_autonomy_command_status", "unknown")
                perception_status = shared_data.get("latest_perception_status", "unknown")
                counts = dict(shared_data.get("mavlink_message_counts", {}))
                mavlink_rx_diagnostics = dict(
                    shared_data.get("mavlink_rx_diagnostics", {})
                )
                external_vio_status = shared_data.get(
                    "external_vio_status",
                    "disabled",
                )

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
                f"mavlink_rx={mavlink_rx_diagnostics}",
                f"external_vio={external_vio_status}",
                flush=True,
            )

            next_print = time.time() + CONFIG.runtime.flow_status_period_s

except KeyboardInterrupt:
    print("Stopping...", flush=True)

    if not OBSERVE_ONLY and RUNNER_MODE == "px4":
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

    if vio_recorder.enabled:
        vio_recorder.close()
        print(
            f"VIO inputs closed: {vio_recorder.root} "
            f"counts={vio_recorder.summary()}",
            flush=True,
        )

    print("Client exited!", flush=True)
