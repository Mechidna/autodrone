from collections import Counter, OrderedDict
import socket
import struct
import time
import threading

from pymavlink import mavutil

from autonomy_core.core.frame_conventions import (
    competition_yaw_boundary_rad,
    competition_yaw_rate_boundary,
)

ENCAPSULATED_RACE_STATUS_MSG_ID = 1
ENCAPSULATED_TRACK_INFO_MSG_ID = 2
SCALED_IMU_MESSAGE_ID = 26
RAW_IMU_MESSAGE_ID = 27
HIGHRES_IMU_MESSAGE_ID = 105
HIL_SENSOR_MESSAGE_ID = 107
SCALED_IMU2_MESSAGE_ID = 116
SCALED_IMU3_MESSAGE_ID = 129

STANDARD_GRAVITY_M_S2 = 9.80665
MILLIG_TO_M_S2 = STANDARD_GRAVITY_M_S2 / 1000.0
MRAD_TO_RAD = 1.0 / 1000.0
MILLIGAUSS_TO_GAUSS = 1.0 / 1000.0

ALTERNATIVE_IMU_MESSAGE_IDS = {
    "SCALED_IMU": SCALED_IMU_MESSAGE_ID,
    "RAW_IMU": RAW_IMU_MESSAGE_ID,
    "HIL_SENSOR": HIL_SENSOR_MESSAGE_ID,
    "SCALED_IMU2": SCALED_IMU2_MESSAGE_ID,
    "SCALED_IMU3": SCALED_IMU3_MESSAGE_ID,
}


def configure_mavlink_udp_receive_buffer(mavlink_connection, requested_bytes):
    """Set and report the kernel receive buffer on a pymavlink UDP connection."""
    requested = max(0, int(requested_bytes))
    udp_socket = getattr(mavlink_connection, "port", None)
    if udp_socket is None:
        raise RuntimeError("MAVLink connection does not expose its UDP socket.")
    if requested > 0:
        udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, requested)
    return int(udp_socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF))


def mavlink_message_source_target(
    message,
    *,
    fallback_system,
    fallback_component,
):
    """Resolve an addressable source target from a received MAVLink message."""

    def source_id(method_name, fallback):
        method = getattr(message, method_name, None)
        try:
            value = int(method()) if callable(method) else int(fallback)
        except (TypeError, ValueError):
            value = int(fallback)
        return value if 1 <= value <= 255 else int(fallback)

    return (
        source_id("get_srcSystem", fallback_system),
        source_id("get_srcComponent", fallback_component),
    )


def request_mavlink_message_rate(
    mavlink_connection,
    message_id,
    rate_hz,
    *,
    target_system=None,
    target_component=None,
):
    """Request a MAVLink message interval and return the interval in microseconds."""
    rate = float(rate_hz)
    if rate <= 0.0:
        raise ValueError("MAVLink message rate must be positive.")
    interval_usec = max(1, int(round(1_000_000.0 / rate)))
    resolved_target_system = (
        mavlink_connection.target_system
        if target_system is None
        else target_system
    )
    resolved_target_component = (
        mavlink_connection.target_component
        if target_component is None
        else target_component
    )
    mavlink_connection.mav.command_long_send(
        int(resolved_target_system),
        int(resolved_target_component),
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        float(message_id),
        float(interval_usec),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    return interval_usec


class MAVLinkRX:

    def __init__(self, mavlink_connection, data, config=None, vio_recorder=None):
        self.mavlink_conn = mavlink_connection
        self.data = data
        self.config = config
        self.vio_recorder = vio_recorder
        mavlink_config = getattr(config, "mavlink", None)
        self.deduplicate_packets = bool(
            getattr(mavlink_config, "deduplicate_packets", True)
        )
        self.duplicate_cache_size = max(
            0,
            int(getattr(mavlink_config, "duplicate_cache_size", 1024)),
        )
        self.duplicate_cache_ttl_s = max(
            0.0,
            float(getattr(mavlink_config, "duplicate_cache_ttl_s", 0.10)),
        )
        self.competition_yaw_inverted = bool(
            config is not None
            and str(config.runtime.runner_mode).lower() == "competition"
            and config.runtime.competition_yaw_inverted
        )
        self.thread = None
        self.is_running = False

        # Local RX-internal buffers. These do not need the shared_data lock
        # because only this MAVLinkRX thread modifies them.
        self.track_chunks = {}
        self.expected_num_track_chunks = {}
        self._recent_packets = OrderedDict()
        self._last_sequence_by_source = {}
        self._diagnostics = Counter()
        self._duplicates_by_type = Counter()
        self._missing_by_source = Counter()
        self._out_of_order_by_source = Counter()

    @classmethod
    def create_mavlink_rx(
        cls,
        mavlink_connection,
        data,
        config=None,
        vio_recorder=None,
    ):
        rx = cls(
            mavlink_connection,
            data,
            config=config,
            vio_recorder=vio_recorder,
        )
        rx.thread = threading.Thread(
            target=rx.mavlink_receive_loop,
            daemon=False,
        )
        rx.is_running = True
        rx.thread.start()
        return rx

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    # --------------------------------------------------------------------------------------
    # Shared-data helpers
    # --------------------------------------------------------------------------------------

    def _get_lock(self):
        if isinstance(self.data, dict):
            return self.data.get("lock")
        return None

    def _store(self, key, value):
        self._update_shared({key: value})

    def _record(self, method_name, *args, **kwargs):
        recorder = self.vio_recorder
        if recorder is None or not recorder.enabled:
            return
        try:
            getattr(recorder, method_name)(*args, **kwargs)
        except Exception as exc:
            print(f"WARNING: VIO recorder {method_name} failed: {exc}", flush=True)

    def _update_shared(self, updates):
        """
        Atomically update one or more keys in shared_data.

        Use this whenever MAVLinkRX writes to self.data.
        """
        lock = self._get_lock()

        if lock is not None:
            with lock:
                self.data.update(updates)
        else:
            self.data.update(updates)

    def _note_message(self, msg_type):
        """
        Record latest MAVLink message type and per-message counts.
        """
        now = time.time()
        lock = self._get_lock()

        if lock is not None:
            with lock:
                self._note_message_unlocked(msg_type, now)
        else:
            self._note_message_unlocked(msg_type, now)

    def _note_message_unlocked(self, msg_type, now):
        self.data["latest_mavlink_msg_type"] = msg_type
        self.data["latest_mavlink_rx_wall_time"] = now

        counts = self.data.setdefault("mavlink_message_counts", {})
        counts[msg_type] = counts.get(msg_type, 0) + 1

    @staticmethod
    def _message_int(msg, method_name):
        method = getattr(msg, method_name, None)
        if not callable(method):
            return None
        try:
            return int(method())
        except (TypeError, ValueError):
            return None

    def _message_metadata(self, msg):
        return {
            "mavlink_seq": self._message_int(msg, "get_seq"),
            "mavlink_src_system": self._message_int(msg, "get_srcSystem"),
            "mavlink_src_component": self._message_int(msg, "get_srcComponent"),
        }

    def _packet_fingerprint(self, msg):
        get_buffer = getattr(msg, "get_msgbuf", None)
        if not callable(get_buffer):
            return None
        try:
            raw_packet = bytes(get_buffer())
        except (TypeError, ValueError):
            return None
        if not raw_packet:
            return None
        metadata = self._message_metadata(msg)
        return (
            metadata["mavlink_src_system"],
            metadata["mavlink_src_component"],
            raw_packet,
        )

    def _is_duplicate_packet(self, msg, now_monotonic):
        if (
            not self.deduplicate_packets
            or self.duplicate_cache_size <= 0
            or self.duplicate_cache_ttl_s <= 0.0
        ):
            return False
        fingerprint = self._packet_fingerprint(msg)
        if fingerprint is None:
            return False

        stale_before = now_monotonic - self.duplicate_cache_ttl_s
        while self._recent_packets:
            _, oldest_time = next(iter(self._recent_packets.items()))
            if oldest_time >= stale_before:
                break
            self._recent_packets.popitem(last=False)

        previous_time = self._recent_packets.get(fingerprint)
        is_duplicate = (
            previous_time is not None
            and now_monotonic - previous_time <= self.duplicate_cache_ttl_s
        )
        self._recent_packets[fingerprint] = now_monotonic
        self._recent_packets.move_to_end(fingerprint)
        while len(self._recent_packets) > self.duplicate_cache_size:
            self._recent_packets.popitem(last=False)
        return is_duplicate

    def _note_sequence(self, msg):
        metadata = self._message_metadata(msg)
        sequence = metadata["mavlink_seq"]
        if sequence is None:
            return
        source = (
            metadata["mavlink_src_system"],
            metadata["mavlink_src_component"],
        )
        source_key = f"{source[0]}:{source[1]}"
        previous = self._last_sequence_by_source.get(source)
        self._last_sequence_by_source[source] = sequence
        if previous is None:
            return

        advance = (sequence - previous) & 0xFF
        if advance == 0:
            self._diagnostics["same_sequence_packets"] += 1
        elif advance <= 127:
            missing = advance - 1
            if missing:
                self._diagnostics["inferred_packets_lost"] += missing
                self._missing_by_source[source_key] += missing
        else:
            self._diagnostics["out_of_order_packets"] += 1
            self._out_of_order_by_source[source_key] += 1

    def diagnostics_snapshot(self):
        return {
            "received_packets": int(self._diagnostics["received_packets"]),
            "accepted_packets": int(self._diagnostics["accepted_packets"]),
            "duplicates_suppressed": int(
                self._diagnostics["duplicates_suppressed"]
            ),
            "same_sequence_packets": int(
                self._diagnostics["same_sequence_packets"]
            ),
            "inferred_packets_lost": int(
                self._diagnostics["inferred_packets_lost"]
            ),
            "out_of_order_packets": int(
                self._diagnostics["out_of_order_packets"]
            ),
            "duplicates_by_type": dict(self._duplicates_by_type),
            "missing_by_source": dict(self._missing_by_source),
            "out_of_order_by_source": dict(self._out_of_order_by_source),
        }

    def _publish_diagnostics(self):
        self._update_shared({"mavlink_rx_diagnostics": self.diagnostics_snapshot()})

    # --------------------------------------------------------------------------------------
    # Receive loop
    # --------------------------------------------------------------------------------------

    def mavlink_receive_loop(self):
        """
        Continuously receive MAVLink messages without blocking.

        This RX class stores the latest useful telemetry into shared_data.
        Optional/non-spec messages are kept as optional debug data if they arrive.
        """
        try:
            while self.is_running:
                try:
                    msg = self.mavlink_conn.recv_match(blocking=False)
                except ConnectionResetError:
                    print(
                        "WARNING: ConnectionResetError was thrown. "
                        "No longer listening to MAVLink port.",
                        flush=True,
                    )
                    return

                if msg is None:
                    time.sleep(0.001)
                    continue
                self.process_message(msg)
        finally:
            self._publish_diagnostics()
            self._record(
                "record_event",
                "mavlink_rx_summary",
                **self.diagnostics_snapshot(),
            )

    def process_message(self, msg):
        """Validate, deduplicate, account for, and dispatch one MAVLink packet."""
        msg_type = msg.get_type()
        if msg_type == "BAD_DATA":
            return False

        self._diagnostics["received_packets"] += 1
        if self._is_duplicate_packet(msg, time.monotonic()):
            self._diagnostics["duplicates_suppressed"] += 1
            self._duplicates_by_type[msg_type] += 1
            if self._diagnostics["duplicates_suppressed"] % 64 == 1:
                self._publish_diagnostics()
            return False

        self._diagnostics["accepted_packets"] += 1
        self._note_sequence(msg)
        self._note_message(msg_type)

        if self._diagnostics["accepted_packets"] % 256 == 0:
            self._publish_diagnostics()

        if msg_type == "HEARTBEAT":
            self.on_heartbeat(msg)
        elif msg_type == "TIMESYNC":
            self.on_timesync(msg)
        elif msg_type == "ATTITUDE":
            self.on_attitude(msg)
        elif msg_type == "SCALED_IMU":
            self.on_scaled_imu(msg, "scaled_imu")
        elif msg_type == "RAW_IMU":
            self.on_raw_imu(msg)
        elif msg_type == "HIGHRES_IMU":
            self.on_highres_imu(msg)
        elif msg_type == "HIL_SENSOR":
            self.on_hil_sensor(msg)
        elif msg_type == "SCALED_IMU2":
            self.on_scaled_imu(msg, "scaled_imu2")
        elif msg_type == "SCALED_IMU3":
            self.on_scaled_imu(msg, "scaled_imu3")
        elif msg_type == "LOCAL_POSITION_NED":
            self.on_local_position_ned(msg)
        elif msg_type == "ODOMETRY":
            self.on_odometry(msg)
        elif msg_type == "ENCAPSULATED_DATA":
            self.on_encapsulated_data(msg)
        elif msg_type == "ACTUATOR_OUTPUT_STATUS":
            self.on_actuator_output_status(msg)
        elif msg_type == "COLLISION":
            self.on_collision(msg)
        elif msg_type == "DATA_TRANSMISSION_HANDSHAKE":
            self.on_data_transmission_handshake(msg)
        elif msg_type == "COMMAND_ACK":
            self.on_command_ack(msg)
        return True

    # --------------------------------------------------------------------------------------
    # Standard / expected telemetry
    # --------------------------------------------------------------------------------------

    def on_heartbeat(self, msg):
        armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

        heartbeat = {
            "armed": armed,
            "base_mode": msg.base_mode,
            "custom_mode": msg.custom_mode,
            "system_status": msg.system_status,
            "type": msg.type,
            "autopilot": msg.autopilot,
            "mavlink_version": msg.mavlink_version,
            "wall_time": time.time(),
        }

        self._update_shared({
            "heartbeat": heartbeat,
            "armed": armed,
        })

    def on_timesync(self, msg):
        wall_time_ns = time.time_ns()
        timesync = {
            "ts1": msg.ts1,
            "tc1": msg.tc1,
            "wall_time": wall_time_ns / 1e9,
            "wall_time_ns": wall_time_ns,
        }

        self._record(
            "record_timesync",
            direction="rx",
            tc1=msg.tc1,
            ts1=msg.ts1,
            wall_time_ns=wall_time_ns,
        )
        self._store("timesync", timesync)

    def on_attitude(self, msg):
        wall_time_ns = time.time_ns()
        raw_yaw = float(msg.yaw)
        raw_yawspeed = float(msg.yawspeed)
        attitude = {
            "roll": msg.roll,
            "pitch": msg.pitch,
            "yaw": competition_yaw_boundary_rad(
                raw_yaw,
                inverted=self.competition_yaw_inverted,
            ),
            "rollspeed": msg.rollspeed,
            "pitchspeed": msg.pitchspeed,
            "yawspeed": competition_yaw_rate_boundary(
                raw_yawspeed,
                inverted=self.competition_yaw_inverted,
            ),
            "yaw_mavlink_raw": raw_yaw,
            "yawspeed_mavlink_raw": raw_yawspeed,
            "competition_yaw_inverted": self.competition_yaw_inverted,
            "time_boot_ms": msg.time_boot_ms,
            "wall_time": wall_time_ns / 1e9,
            "wall_time_ns": wall_time_ns,
        }
        attitude.update(self._message_metadata(msg))

        self._record("record_attitude", attitude)
        self._store("attitude", attitude)

    def on_highres_imu(self, msg):
        wall_time_ns = time.time_ns()
        highres_imu = {
            "source": "highres_imu",
            "message_id": HIGHRES_IMU_MESSAGE_ID,
            "units_profile": "si",
            "xacc": msg.xacc,
            "yacc": msg.yacc,
            "zacc": msg.zacc,
            "xgyro": msg.xgyro,
            "ygyro": msg.ygyro,
            "zgyro": msg.zgyro,
            "xmag": msg.xmag,
            "ymag": msg.ymag,
            "zmag": msg.zmag,
            "abs_pressure": msg.abs_pressure,
            "diff_pressure": msg.diff_pressure,
            "pressure_alt": msg.pressure_alt,
            "temperature": msg.temperature,
            "fields_updated": msg.fields_updated,
            "time_usec": msg.time_usec,
            "sensor_id": getattr(msg, "id", 0),

            # Convenience aliases
            "accel_xyz": (msg.xacc, msg.yacc, msg.zacc),
            "gyro_xyz": (msg.xgyro, msg.ygyro, msg.zgyro),

            "wall_time": wall_time_ns / 1e9,
            "wall_time_ns": wall_time_ns,
        }
        highres_imu.update(self._message_metadata(msg))

        self._record("record_imu", highres_imu)
        self._store("highres_imu", highres_imu)

    def on_scaled_imu(self, msg, source):
        """Normalize a MAVLink SCALED_IMU* sample while retaining raw fields."""
        source = str(source).lower()
        message_ids = {
            "scaled_imu": SCALED_IMU_MESSAGE_ID,
            "scaled_imu2": SCALED_IMU2_MESSAGE_ID,
            "scaled_imu3": SCALED_IMU3_MESSAGE_ID,
        }
        if source not in message_ids:
            raise ValueError(f"unsupported scaled IMU source: {source}")
        wall_time_ns = time.time_ns()
        time_boot_ms = int(msg.time_boot_ms)
        sample = {
            "source": source,
            "message_id": message_ids[source],
            "units_profile": "mG_mrad_s_mgauss_cdegC",
            "time_boot_ms": time_boot_ms,
            "time_usec": time_boot_ms * 1000,
            "timestamp_kind": "boot",
            "wall_time": wall_time_ns / 1e9,
            "wall_time_ns": wall_time_ns,
            "sensor_id": int(getattr(msg, "id", 0)),
            "xacc_raw": int(msg.xacc),
            "yacc_raw": int(msg.yacc),
            "zacc_raw": int(msg.zacc),
            "xgyro_raw": int(msg.xgyro),
            "ygyro_raw": int(msg.ygyro),
            "zgyro_raw": int(msg.zgyro),
            "xmag_raw": int(msg.xmag),
            "ymag_raw": int(msg.ymag),
            "zmag_raw": int(msg.zmag),
            "temperature_raw": int(getattr(msg, "temperature", 0)),
            "xacc": float(msg.xacc) * MILLIG_TO_M_S2,
            "yacc": float(msg.yacc) * MILLIG_TO_M_S2,
            "zacc": float(msg.zacc) * MILLIG_TO_M_S2,
            "xgyro": float(msg.xgyro) * MRAD_TO_RAD,
            "ygyro": float(msg.ygyro) * MRAD_TO_RAD,
            "zgyro": float(msg.zgyro) * MRAD_TO_RAD,
            "xmag": float(msg.xmag) * MILLIGAUSS_TO_GAUSS,
            "ymag": float(msg.ymag) * MILLIGAUSS_TO_GAUSS,
            "zmag": float(msg.zmag) * MILLIGAUSS_TO_GAUSS,
            "temperature": (
                ""
                if int(getattr(msg, "temperature", 0)) == 0
                else float(msg.temperature) / 100.0
            ),
        }
        sample["accel_xyz"] = (sample["xacc"], sample["yacc"], sample["zacc"])
        sample["gyro_xyz"] = (
            sample["xgyro"], sample["ygyro"], sample["zgyro"]
        )
        sample.update(self._message_metadata(msg))
        self._record("record_imu_source", source, sample)
        self._store(source, sample)

    def on_raw_imu(self, msg):
        """Capture device-specific RAW_IMU values without inventing SI scaling."""
        wall_time_ns = time.time_ns()
        sample = {
            "source": "raw_imu",
            "message_id": RAW_IMU_MESSAGE_ID,
            "units_profile": "device_specific_unscaled",
            "time_usec": int(msg.time_usec),
            "time_boot_ms": "",
            "timestamp_kind": "epoch_or_boot",
            "wall_time": wall_time_ns / 1e9,
            "wall_time_ns": wall_time_ns,
            "sensor_id": int(getattr(msg, "id", 0)),
            "xacc_raw": int(msg.xacc),
            "yacc_raw": int(msg.yacc),
            "zacc_raw": int(msg.zacc),
            "xgyro_raw": int(msg.xgyro),
            "ygyro_raw": int(msg.ygyro),
            "zgyro_raw": int(msg.zgyro),
            "xmag_raw": int(msg.xmag),
            "ymag_raw": int(msg.ymag),
            "zmag_raw": int(msg.zmag),
            "temperature_raw": int(getattr(msg, "temperature", 0)),
        }
        sample.update(self._message_metadata(msg))
        self._record("record_imu_source", "raw_imu", sample)
        self._store("raw_imu", sample)

    def on_hil_sensor(self, msg):
        """Capture the simulator HIL_SENSOR SI stream when it is available."""
        wall_time_ns = time.time_ns()
        sample = {
            "source": "hil_sensor",
            "message_id": HIL_SENSOR_MESSAGE_ID,
            "units_profile": "si",
            "time_usec": int(msg.time_usec),
            "time_boot_ms": "",
            "timestamp_kind": "epoch_or_boot",
            "wall_time": wall_time_ns / 1e9,
            "wall_time_ns": wall_time_ns,
            "sensor_id": int(getattr(msg, "id", 0)),
            "fields_updated": int(getattr(msg, "fields_updated", 0)),
            "xacc": float(msg.xacc),
            "yacc": float(msg.yacc),
            "zacc": float(msg.zacc),
            "xgyro": float(msg.xgyro),
            "ygyro": float(msg.ygyro),
            "zgyro": float(msg.zgyro),
            "xmag": float(msg.xmag),
            "ymag": float(msg.ymag),
            "zmag": float(msg.zmag),
            "temperature": float(msg.temperature),
        }
        sample["accel_xyz"] = (sample["xacc"], sample["yacc"], sample["zacc"])
        sample["gyro_xyz"] = (
            sample["xgyro"], sample["ygyro"], sample["zgyro"]
        )
        sample.update(self._message_metadata(msg))
        self._record("record_imu_source", "hil_sensor", sample)
        self._store("hil_sensor", sample)

    # --------------------------------------------------------------------------------------
    # Optional telemetry/debug messages
    # --------------------------------------------------------------------------------------

    def on_local_position_ned(self, msg):
        wall_time_ns = time.time_ns()
        local_position_ned = {
            "x": msg.x,
            "y": msg.y,
            "z": msg.z,
            "vx": msg.vx,
            "vy": msg.vy,
            "vz": msg.vz,
            "time_boot_ms": msg.time_boot_ms,

            # NED aliases
            "pos_ned": (msg.x, msg.y, msg.z),
            "vel_ned": (msg.vx, msg.vy, msg.vz),

            # z-up aliases for autonomy code that uses up-positive
            "pos_neu": (msg.x, msg.y, -msg.z),
            "vel_neu": (msg.vx, msg.vy, -msg.vz),

            "wall_time": wall_time_ns / 1e9,
            "wall_time_ns": wall_time_ns,
        }
        local_position_ned.update(self._message_metadata(msg))

        self._record("record_local_position", local_position_ned)
        self._store("local_position_ned", local_position_ned)

    def on_odometry(self, msg):
        wall_time_ns = time.time_ns()
        q_wxyz = tuple(msg.q)
        q_xyzw = (msg.q[1], msg.q[2], msg.q[3], msg.q[0])

        odometry = {
            "x": msg.x,
            "y": msg.y,
            "z": msg.z,
            "vx": msg.vx,
            "vy": msg.vy,
            "vz": msg.vz,
            "q_wxyz": q_wxyz,
            "q_xyzw": q_xyzw,
            "rollspeed": msg.rollspeed,
            "pitchspeed": msg.pitchspeed,
            "yawspeed": msg.yawspeed,
            "time_usec": msg.time_usec,
            "reset_counter": msg.reset_counter,
            "frame_id": getattr(msg, "frame_id", None),
            "child_frame_id": getattr(msg, "child_frame_id", None),
            "estimator_type": getattr(msg, "estimator_type", None),
            "quality": getattr(msg, "quality", None),
            "pose_covariance": tuple(getattr(msg, "pose_covariance", ()) or ()),
            "velocity_covariance": tuple(
                getattr(msg, "velocity_covariance", ()) or ()
            ),

            # NED aliases
            "pos_ned": (msg.x, msg.y, msg.z),
            "vel_ned": (msg.vx, msg.vy, msg.vz),

            # z-up aliases
            "pos_neu": (msg.x, msg.y, -msg.z),
            "vel_neu": (msg.vx, msg.vy, -msg.vz),

            "wall_time": wall_time_ns / 1e9,
            "wall_time_ns": wall_time_ns,
        }
        odometry.update(self._message_metadata(msg))

        self._record("record_odometry", odometry)
        self._store("odometry", odometry)

    def on_command_ack(self, msg):
        command_ack = {
            "command": int(msg.command),
            "result": int(msg.result),
            "progress": getattr(msg, "progress", None),
            "result_param2": getattr(msg, "result_param2", None),
            "target_system": getattr(msg, "target_system", None),
            "target_component": getattr(msg, "target_component", None),
            "wall_time_ns": time.time_ns(),
        }
        command_ack.update(self._message_metadata(msg))
        self._store("latest_command_ack", command_ack)
        if command_ack["command"] == int(
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL
        ):
            self._record(
                "record_event",
                "mavlink_message_interval_ack",
                **command_ack,
            )

    def on_actuator_output_status(self, msg):
        actuators = list(msg.actuator)

        actuator_output_status = {
            "time_usec": msg.time_usec,
            "actuator": actuators,
            "wall_time": time.time(),
        }

        if len(actuators) >= 4:
            actuator_output_status.update({
                "motor_front_left": actuators[0],
                "motor_front_right": actuators[1],
                "motor_back_left": actuators[2],
                "motor_back_right": actuators[3],
            })

        self._store("actuator_output_status", actuator_output_status)

    def on_collision(self, msg):
        collision = {
            "id": msg.id,

            # Example collision IDs:
            # 1001 - Gate
            # 1002 - Environment
            "threat_level": msg.threat_level,
            "impact": msg.horizontal_minimum_delta,
            "wall_time": time.time(),
        }

        self._store("collision", collision)

    # --------------------------------------------------------------------------------------
    # Optional custom/race data
    # --------------------------------------------------------------------------------------

    def on_encapsulated_data(self, msg):
        if not msg:
            return

        raw_payload = bytes(msg.data)

        if len(raw_payload) < 1:
            return

        data_type = int(raw_payload[0])
        self._store("latest_encapsulated_data_type", data_type)

        if data_type == ENCAPSULATED_RACE_STATUS_MSG_ID:
            self.on_race_status(msg)

        elif data_type == ENCAPSULATED_TRACK_INFO_MSG_ID:
            self.on_track_data_packet(msg)

    def on_race_status(self, msg):
        raw_payload = bytes(msg.data)

        fmt = "<BQqqIq"
        needed = struct.calcsize(fmt)

        if len(raw_payload) < needed:
            return

        (
            data_type,
            sim_boot_time_ms,
            race_start_boot_time_ms,
            race_finish_time_ns,
            active_gate_index,
            last_gate_race_time,
        ) = struct.unpack_from(fmt, raw_payload)

        race_status = {
            "data_type": data_type,
            "sim_boot_time_ms": sim_boot_time_ms,
            "race_start_boot_time_ms": race_start_boot_time_ms,
            "race_finish_time_ns": race_finish_time_ns,
            "active_gate_index": active_gate_index,
            "last_gate_race_time": last_gate_race_time,
            "wall_time": time.time(),
        }

        self._store("race_status", race_status)

    def on_data_transmission_handshake(self, msg):
        # DATA_TRANSMISSION_HANDSHAKE appears to be repurposed by the example
        # for upcoming track-data packets.
        track_data_transfer_id = msg.width

        self.track_chunks[track_data_transfer_id] = {}
        self.expected_num_track_chunks[track_data_transfer_id] = msg.packets

        handshake = {
            "transfer_id": track_data_transfer_id,
            "expected_packets": msg.packets,
            "payload": msg.payload,
            "size": msg.size,
            "width": msg.width,
            "height": msg.height,
            "type": msg.type,
            "jpg_quality": msg.jpg_quality,
            "wall_time": time.time(),
        }

        self._store("latest_track_data_handshake", handshake)

    def on_track_data_packet(self, msg):
        raw_payload = bytes(msg.data)

        fmt = "<BH"
        needed = struct.calcsize(fmt)

        if len(raw_payload) < needed:
            return

        data_type, transfer_id = struct.unpack_from(fmt, raw_payload)

        if transfer_id not in self.expected_num_track_chunks:
            return

        chunk_payload = raw_payload[needed:]
        self.track_chunks[transfer_id][msg.seqnr] = chunk_payload

        if len(self.track_chunks[transfer_id]) == self.expected_num_track_chunks[transfer_id]:
            full_payload = bytes()

            for i in range(len(self.track_chunks[transfer_id])):
                full_payload += self.track_chunks[transfer_id][i]

            del self.track_chunks[transfer_id]
            del self.expected_num_track_chunks[transfer_id]

            self.on_track_data(full_payload)

    def on_track_data(self, payload):
        if len(payload) < 2:
            return

        num_gates, = struct.unpack_from("<H", payload)
        payload = payload[2:]

        gates = []
        gate_fmt = "<Hfffffffff"
        gate_size = struct.calcsize(gate_fmt)

        for _ in range(num_gates):
            if len(payload) < gate_size:
                break

            (
                gate_id,
                position_ned_x,
                position_ned_y,
                position_ned_z,
                orientation_ned_w,
                orientation_ned_x,
                orientation_ned_y,
                orientation_ned_z,
                width,
                height,
            ) = struct.unpack_from(gate_fmt, payload)

            gates.append({
                "gate_id": gate_id,
                "position_ned": (
                    position_ned_x,
                    position_ned_y,
                    position_ned_z,
                ),
                "position_neu": (
                    position_ned_x,
                    position_ned_y,
                    -position_ned_z,
                ),
                "orientation_ned_wxyz": (
                    orientation_ned_w,
                    orientation_ned_x,
                    orientation_ned_y,
                    orientation_ned_z,
                ),
                "width": width,
                "height": height,
            })

            payload = payload[gate_size:]

        track_data = {
            "num_gates": num_gates,
            "gates": gates,
            "wall_time": time.time(),
        }

        self._update_shared({
            "track_data": track_data,
            "track_gates": gates,
        })
