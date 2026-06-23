import struct
import time
import threading

from pymavlink import mavutil

ENCAPSULATED_RACE_STATUS_MSG_ID = 1
ENCAPSULATED_TRACK_INFO_MSG_ID = 2


class MAVLinkRX:

    def __init__(self, mavlink_connection, data):
        self.mavlink_conn = mavlink_connection
        self.data = data
        self.thread = None
        self.is_running = False

        # Local RX-internal buffers. These do not need the shared_data lock
        # because only this MAVLinkRX thread modifies them.
        self.track_chunks = {}
        self.expected_num_track_chunks = {}

    @classmethod
    def create_mavlink_rx(cls, mavlink_connection, data):
        rx = cls(mavlink_connection, data)
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

    # --------------------------------------------------------------------------------------
    # Receive loop
    # --------------------------------------------------------------------------------------

    def mavlink_receive_loop(self):
        """
        Continuously receive MAVLink messages without blocking.

        This RX class stores the latest useful telemetry into shared_data.
        Optional/non-spec messages are kept as optional debug data if they arrive.
        """
        while self.is_running:

            try:
                msg = self.mavlink_conn.recv_match(blocking=False)
            except ConnectionResetError:
                print("WARNING: ConnectionResetError was thrown. No longer listening to MAVLink port.")
                return

            if msg is None:
                time.sleep(0.001)
                continue

            msg_type = msg.get_type()

            if msg_type == "BAD_DATA":
                continue

            self._note_message(msg_type)

            if msg_type == "HEARTBEAT":
                self.on_heartbeat(msg)

            elif msg_type == "TIMESYNC":
                self.on_timesync(msg)

            elif msg_type == "ATTITUDE":
                self.on_attitude(msg)

            elif msg_type == "HIGHRES_IMU":
                self.on_highres_imu(msg)

            # Optional/debug telemetry. These may not be guaranteed by the spec,
            # but storing them is useful if they arrive.
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
        timesync = {
            "ts1": msg.ts1,
            "tc1": msg.tc1,
            "wall_time": time.time(),
        }

        self._store("timesync", timesync)

    def on_attitude(self, msg):
        attitude = {
            "roll": msg.roll,
            "pitch": msg.pitch,
            "yaw": msg.yaw,
            "rollspeed": msg.rollspeed,
            "pitchspeed": msg.pitchspeed,
            "yawspeed": msg.yawspeed,
            "time_boot_ms": msg.time_boot_ms,
            "wall_time": time.time(),
        }

        self._store("attitude", attitude)

    def on_highres_imu(self, msg):
        highres_imu = {
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

            # Convenience aliases
            "accel_xyz": (msg.xacc, msg.yacc, msg.zacc),
            "gyro_xyz": (msg.xgyro, msg.ygyro, msg.zgyro),

            "wall_time": time.time(),
        }

        self._store("highres_imu", highres_imu)

    # --------------------------------------------------------------------------------------
    # Optional telemetry/debug messages
    # --------------------------------------------------------------------------------------

    def on_local_position_ned(self, msg):
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

            "wall_time": time.time(),
        }

        self._store("local_position_ned", local_position_ned)

    def on_odometry(self, msg):
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

            # NED aliases
            "pos_ned": (msg.x, msg.y, msg.z),
            "vel_ned": (msg.vx, msg.vy, msg.vz),

            # z-up aliases
            "pos_neu": (msg.x, msg.y, -msg.z),
            "vel_neu": (msg.vx, msg.vy, -msg.vz),

            "wall_time": time.time(),
        }

        self._store("odometry", odometry)

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