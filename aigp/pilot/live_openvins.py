"""Live ROS-free OpenVINS bridge between Windows Python and WSL.

The bridge reuses the VIO recorder callbacks so it receives every HIGHRES_IMU
sample and every completed JPEG, not the lower-rate snapshots consumed by the
control loop.  A small line/binary protocol feeds the existing C++ OpenVINS
runner through a ``wsl.exe`` subprocess pipe.
"""

from __future__ import annotations

from collections import deque
import math
import os
from pathlib import Path
import queue
import shlex
import statistics
import subprocess
import threading
import time
from types import SimpleNamespace
from typing import Any

from experimental_gate_vio_alignment import ExperimentalGateVioAlignment


def _windows_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = resolved.as_posix().split(":", 1)[-1]
    return f"/mnt/{drive}{tail}"


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("empty percentile input")
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, probability)) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _quat_xyzw_matrix(q: tuple[float, float, float, float]) -> list[list[float]]:
    x, y, z, w = q
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-9:
        raise ValueError("zero OpenVINS quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def _openvins_body_to_ned(
    q: tuple[float, float, float, float],
) -> list[list[float]]:
    # OpenVINS stores JPL q_GtoI.  The Hamilton matrix for the same numeric
    # coefficients is R_ItoG.  The fixed startup alignment measured on all five
    # VQ1 captures is R_GtoNED=diag(-1,+1,-1).
    rotation_imu_to_global = _quat_xyzw_matrix(q)
    return [
        [-value for value in rotation_imu_to_global[0]],
        list(rotation_imu_to_global[1]),
        [-value for value in rotation_imu_to_global[2]],
    ]


def _matrix_to_rpy(rotation: list[list[float]]) -> tuple[float, float, float]:
    roll = math.atan2(rotation[2][1], rotation[2][2])
    pitch = math.atan2(
        -rotation[2][0],
        math.hypot(rotation[2][1], rotation[2][2]),
    )
    yaw = math.atan2(rotation[1][0], rotation[0][0])
    return roll, pitch, yaw


class LiveOpenVins:
    """Recorder-compatible live OpenVINS transport."""

    def __init__(self, shared_data: dict[str, Any], config: Any):
        self.shared_data = shared_data
        self.config = config
        self.enabled = os.environ.get("AIGP_LIVE_OPENVINS", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.root = "live_openvins"
        self._events: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=8192)
        self._stop = threading.Event()
        self._process: subprocess.Popen[bytes] | None = None
        self._worker: threading.Thread | None = None
        self._reader: threading.Thread | None = None
        self._camera_sequence = 0
        camera_stride_text = os.environ.get(
            "AIGP_LIVE_OPENVINS_CAMERA_STRIDE",
            "2",
        ).strip()
        try:
            self.camera_stride = int(camera_stride_text)
        except ValueError as exc:
            raise RuntimeError(
                "AIGP_LIVE_OPENVINS_CAMERA_STRIDE must be an integer between 1 and 8"
            ) from exc
        if not 1 <= self.camera_stride <= 8:
            raise RuntimeError(
                "AIGP_LIVE_OPENVINS_CAMERA_STRIDE must be between 1 and 8"
            )
        self._counts = {
            "imu_received": 0,
            "camera_received": 0,
            "imu_sent": 0,
            "camera_sent": 0,
            "states_received": 0,
            "queue_dropped": 0,
            "interpolated_imu": 0,
        }
        self._origin_position_global: tuple[float, float, float] | None = None
        self._last_trace_time = 0.0
        self._lock_published_yaw = os.environ.get(
            "AIGP_LIVE_OPENVINS_LOCK_YAW", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._published_yaw_rad = math.radians(
            float(os.environ.get("AIGP_LIVE_OPENVINS_YAW_DEG", "180.0"))
        )
        self._gate_alignment = (
            ExperimentalGateVioAlignment(config) if config is not None else None
        )

        if self.enabled:
            self._start()

    @classmethod
    def from_environment(cls, shared_data: dict[str, Any], config: Any):
        return cls(shared_data, config)

    def _start(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "openvins"
            / "live_config"
            / "estimator_config.yaml"
        )
        wsl_config = _windows_to_wsl(config_path)
        runner = os.environ.get(
            "AIGP_OPENVINS_LIVE_RUNNER",
            "$HOME/.cache/aigp_openvins_runner/run_vq1_dataset",
        )
        command = (
            f"exec {runner} --live {shlex.quote(wsl_config)} red"
        )
        self._process = subprocess.Popen(
            ["wsl.exe", "bash", "-lc", command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="live-openvins-writer",
            daemon=True,
        )
        self._reader = threading.Thread(
            target=self._reader_loop,
            name="live-openvins-reader",
            daemon=True,
        )
        self._worker.start()
        self._reader.start()
        self._publish_status("starting")
        print(
            f"LIVE_OPENVINS enabled config={config_path} runner={runner} "
            f"camera_stride={self.camera_stride} "
            f"locked_yaw={int(self._lock_published_yaw)} "
            f"gate_map_alignment="
            f"{int(self._gate_alignment is not None and bool(self._gate_alignment.options.enabled))}",
            flush=True,
        )

    def _enqueue(self, kind: str, payload: Any) -> bool:
        if not self.enabled or self._stop.is_set():
            return False
        try:
            self._events.put_nowait((kind, payload))
            return True
        except queue.Full:
            self._counts["queue_dropped"] += 1
            self._publish_status("input_queue_full")
            return False

    def record_imu(self, sample: dict[str, Any]) -> bool:
        self._counts["imu_received"] += 1
        return self._enqueue("imu", dict(sample))

    def record_camera_frame(self, **frame: Any) -> bool:
        self._counts["camera_received"] += 1
        sequence = self._camera_sequence
        self._camera_sequence += 1
        # Default to the validated offline camera_stride=2 configuration, while
        # permitting an explicit full-rate live experiment with stride=1.
        if sequence % self.camera_stride:
            return True
        copied = dict(frame)
        copied["jpeg_bytes"] = bytes(frame["jpeg_bytes"])
        return self._enqueue("camera", copied)

    def record_timesync(self, **sample: Any) -> bool:
        return self._enqueue("timesync", dict(sample))

    def record_event(self, *args: Any, **kwargs: Any) -> bool:
        return self.enabled

    def record_imu_source(self, *args: Any, **kwargs: Any) -> bool:
        return False

    def record_attitude(self, *args: Any, **kwargs: Any) -> bool:
        return False

    def record_local_position(self, *args: Any, **kwargs: Any) -> bool:
        return False

    def record_odometry(self, *args: Any, **kwargs: Any) -> bool:
        return False

    def _publish_status(self, status: str) -> None:
        lock = self.shared_data.get("lock")
        if lock is None:
            self.shared_data["external_vio_status"] = status
            return
        with lock:
            self.shared_data["external_vio_status"] = status

    @staticmethod
    def _clock_offset(pairs: deque[tuple[int, int]]) -> int | None:
        if len(pairs) < 5:
            return None
        ordered = sorted(pairs, key=lambda item: item[0])
        selected = ordered[: max(5, len(ordered) // 2)]
        return int(round(statistics.median(item[1] for item in selected)))

    def _worker_loop(self) -> None:
        requests: dict[int, int] = {}
        sync_pairs: deque[tuple[int, int]] = deque(maxlen=100)
        raw_imus: list[dict[str, Any]] = []
        raw_cameras: list[dict[str, Any]] = []
        mapped_imus: list[tuple[int, tuple[float, ...]]] = []
        pending_cameras: list[dict[str, Any]] = []
        server_minus_wall_ns: int | None = None
        imu_to_server_ns: int | None = None
        origin_ns: int | None = None
        last_mapped_imu_ns: int | None = None
        last_sent_imu: tuple[float, tuple[float, ...]] | None = None

        def map_imu(sample: dict[str, Any]) -> None:
            nonlocal last_mapped_imu_ns, origin_ns
            if imu_to_server_ns is None:
                return
            timestamp_ns = int(sample["time_usec"]) * 1000 + imu_to_server_ns
            if last_mapped_imu_ns is not None and timestamp_ns <= last_mapped_imu_ns:
                return
            values = (
                -float(sample["xgyro"]),
                -float(sample["ygyro"]),
                -float(sample["zgyro"]),
                float(sample["xacc"]),
                float(sample["yacc"]),
                float(sample["zacc"]),
            )
            if not all(math.isfinite(value) for value in values):
                return
            if origin_ns is None:
                origin_ns = timestamp_ns
            mapped_imus.append((timestamp_ns, values))
            last_mapped_imu_ns = timestamp_ns

        def send_imu(timestamp_ns: int, values: tuple[float, ...]) -> None:
            nonlocal last_sent_imu
            assert origin_ns is not None
            timestamp = (timestamp_ns - origin_ns) / 1e9
            if last_sent_imu is not None:
                previous_time, previous_values = last_sent_imu
                gap = timestamp - previous_time
                if 0.03 < gap <= 0.12:
                    segments = max(2, int(math.ceil(gap / 0.008333333)))
                    for index in range(1, segments):
                        fraction = index / segments
                        interp_time = previous_time + fraction * gap
                        interp_values = tuple(
                            start + fraction * (end - start)
                            for start, end in zip(previous_values, values)
                        )
                        self._write_imu(interp_time, interp_values)
                        self._counts["interpolated_imu"] += 1
            self._write_imu(timestamp, values)
            last_sent_imu = (timestamp, values)

        def flush_ready_cameras() -> None:
            if origin_ns is None:
                return
            pending_cameras.sort(key=lambda item: int(item["sim_time_ns"]))
            while pending_cameras and mapped_imus:
                camera = pending_cameras[0]
                camera_ns = int(camera["sim_time_ns"])
                future_index = next(
                    (
                        index
                        for index, (timestamp_ns, _) in enumerate(mapped_imus)
                        if timestamp_ns > camera_ns
                    ),
                    None,
                )
                if future_index is None:
                    return
                if camera_ns < mapped_imus[0][0]:
                    pending_cameras.pop(0)
                    continue
                for timestamp_ns, values in mapped_imus[: future_index + 1]:
                    send_imu(timestamp_ns, values)
                del mapped_imus[: future_index + 1]
                pending_cameras.pop(0)
                self._write_camera(
                    (camera_ns - origin_ns) / 1e9,
                    int(camera["frame_id"]),
                    bytes(camera["jpeg_bytes"]),
                )

        try:
            while not self._stop.is_set():
                try:
                    kind, payload = self._events.get(timeout=0.05)
                except queue.Empty:
                    flush_ready_cameras()
                    continue

                if kind == "timesync":
                    direction = str(payload.get("direction", "")).lower()
                    tc1 = int(payload.get("tc1", 0))
                    ts1 = int(payload.get("ts1", 0))
                    wall_ns = int(payload.get("wall_time_ns", 0))
                    if direction == "tx" and ts1 == 0:
                        requests[tc1] = wall_ns
                    elif direction == "rx" and ts1 in requests:
                        tx_wall = requests.pop(ts1)
                        if wall_ns >= tx_wall:
                            midpoint = (tx_wall + wall_ns) // 2
                            sync_pairs.append((wall_ns - tx_wall, tc1 - midpoint))
                            server_minus_wall_ns = self._clock_offset(sync_pairs)
                elif kind == "imu":
                    if imu_to_server_ns is None:
                        raw_imus.append(payload)
                    else:
                        map_imu(payload)
                elif kind == "camera":
                    if imu_to_server_ns is None:
                        raw_cameras.append(payload)
                    else:
                        pending_cameras.append(payload)

                if (
                    imu_to_server_ns is None
                    and server_minus_wall_ns is not None
                    and len(raw_imus) >= 50
                ):
                    candidates = [
                        int(sample["wall_time_ns"])
                        + server_minus_wall_ns
                        - int(sample["time_usec"]) * 1000
                        for sample in raw_imus
                    ]
                    imu_to_server_ns = int(round(_percentile(candidates, 0.02)))
                    for sample in raw_imus:
                        map_imu(sample)
                    pending_cameras.extend(raw_cameras)
                    raw_imus.clear()
                    raw_cameras.clear()
                    self._publish_status("clock_aligned_waiting_initialization")
                    print(
                        "LIVE_OPENVINS clock aligned "
                        f"timesync_pairs={len(sync_pairs)} "
                        f"imu_to_server_ms={imu_to_server_ns / 1e6:.3f}",
                        flush=True,
                    )
                flush_ready_cameras()
        except Exception as exc:
            self._publish_status(f"writer_error:{type(exc).__name__}:{exc}")
            print(f"LIVE_OPENVINS writer failed: {exc}", flush=True)

    def _write_imu(self, timestamp: float, values: tuple[float, ...]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise RuntimeError("OpenVINS subprocess is not running")
        line = "I {:.17g} {}\n".format(
            timestamp,
            " ".join(f"{value:.17g}" for value in values),
        )
        process.stdin.write(line.encode("ascii"))
        process.stdin.flush()
        self._counts["imu_sent"] += 1

    def _write_camera(self, timestamp: float, frame_id: int, jpeg: bytes) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise RuntimeError("OpenVINS subprocess is not running")
        process.stdin.write(
            f"C {timestamp:.17g} {frame_id} {len(jpeg)}\n".encode("ascii")
        )
        process.stdin.write(jpeg)
        process.stdin.write(b"\n")
        process.stdin.flush()
        self._counts["camera_sent"] += 1

    def _reader_loop(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            for raw_line in iter(process.stdout.readline, b""):
                line = raw_line.decode("utf-8", errors="replace").strip()
                if line.startswith("AIGP_VIO_STATE "):
                    self._consume_state(line)
                elif line.startswith("AIGP_VIO_WAIT"):
                    self._publish_status("waiting_initialization")
                    print(f"LIVE_OPENVINS {line}", flush=True)
                elif line.startswith("ERROR:"):
                    self._publish_status(f"runner_error:{line}")
                    print(f"LIVE_OPENVINS {line}", flush=True)
            if not self._stop.is_set():
                code = process.poll()
                self._publish_status(f"runner_exited:{code}")
                print(f"LIVE_OPENVINS runner exited code={code}", flush=True)
        except Exception as exc:
            self._publish_status(f"reader_error:{type(exc).__name__}:{exc}")
            print(f"LIVE_OPENVINS reader failed: {exc}", flush=True)

    def _consume_state(self, line: str) -> None:
        fields = line.split()
        if len(fields) != 12:
            raise ValueError(f"unexpected OpenVINS state field count: {len(fields)}")
        values = [float(value) for value in fields[1:]]
        timestamp = values[0]
        q = tuple(values[1:5])
        p = tuple(values[5:8])
        v = tuple(values[8:11])
        if self._origin_position_global is None:
            self._origin_position_global = p
        origin = self._origin_position_global
        relative = tuple(value - start for value, start in zip(p, origin))
        raw_pos_neu = (-relative[0], relative[1], relative[2])
        vel_neu = (-v[0], v[1], v[2])
        roll, pitch, raw_yaw = _matrix_to_rpy(_openvins_body_to_ned(q))
        yaw = self._published_yaw_rad if self._lock_published_yaw else raw_yaw
        now = time.time()

        lock = self.shared_data.get("lock")
        if lock is None:
            latest_perception = self.shared_data.get("latest_perception")
        else:
            with lock:
                latest_perception = self.shared_data.get("latest_perception")
        if self._gate_alignment is None:
            alignment = SimpleNamespace(
                aligned_pos_neu=raw_pos_neu,
                offset_neu=(0.0, 0.0, 0.0),
                initialized=False,
                accepted=False,
                reason="disabled",
            )
        else:
            alignment = self._gate_alignment.update(
                raw_pos_neu,
                SimpleNamespace(
                    roll_rad=roll,
                    pitch_rad=pitch,
                    yaw_rad=yaw,
                    latest_perception=latest_perception,
                ),
                now=now,
            )
        pos_neu = tuple(float(value) for value in alignment.aligned_pos_neu)
        pos_ned = (pos_neu[0], pos_neu[1], -pos_neu[2])
        vel_ned = (vel_neu[0], vel_neu[1], -vel_neu[2])
        attitude = {
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "raw_openvins_yaw": raw_yaw,
            "rollspeed": 0.0,
            "pitchspeed": 0.0,
            "yawspeed": 0.0,
            "wall_time": now,
            "source": "live_openvins",
        }
        local_position = {
            "pos_ned": pos_ned,
            "vel_ned": vel_ned,
            "pos_neu": pos_neu,
            "vel_neu": vel_neu,
            "x": pos_ned[0],
            "y": pos_ned[1],
            "z": pos_ned[2],
            "vx": vel_ned[0],
            "vy": vel_ned[1],
            "vz": vel_ned[2],
            "wall_time": now,
            "source": "live_openvins",
            "openvins_timestamp": timestamp,
            "raw_pos_neu": raw_pos_neu,
            "gate_alignment_initialized": alignment.initialized,
            "gate_alignment_accepted": alignment.accepted,
            "gate_alignment_offset_neu": tuple(
                float(value) for value in alignment.offset_neu
            ),
            "gate_alignment_reason": alignment.reason,
        }
        if lock is None:
            self.shared_data["external_vio_attitude"] = attitude
            self.shared_data["external_vio_local_position_ned"] = local_position
            self.shared_data["external_vio_status"] = "tracking"
        else:
            with lock:
                self.shared_data["external_vio_attitude"] = attitude
                self.shared_data["external_vio_local_position_ned"] = local_position
                self.shared_data["external_vio_status"] = "tracking"
        self._counts["states_received"] += 1
        if now - self._last_trace_time >= 1.0:
            self._last_trace_time = now
            print(
                "LIVE_OPENVINS state "
                f"t={timestamp:.3f} "
                f"pos_neu=({pos_neu[0]:.2f},{pos_neu[1]:.2f},{pos_neu[2]:.2f}) "
                f"vel_neu=({vel_neu[0]:.2f},{vel_neu[1]:.2f},{vel_neu[2]:.2f}) "
                f"rpy_deg=({math.degrees(roll):.1f},{math.degrees(pitch):.1f},"
                f"{math.degrees(yaw):.1f}) "
                f"raw_yaw_deg={math.degrees(raw_yaw):.1f} "
                f"gate_offset=({alignment.offset_neu[0]:.2f},"
                f"{alignment.offset_neu[1]:.2f},{alignment.offset_neu[2]:.2f})",
                flush=True,
            )

    def close(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        process = self._process
        # The writer owns stdin.  Let it leave its loop before appending the
        # quit record so Q can never be inserted in the middle of a JPEG.
        if self._worker is not None:
            self._worker.join(timeout=1.0)
        if process is not None and process.stdin is not None:
            try:
                process.stdin.write(b"Q\n")
                process.stdin.flush()
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        if process is not None:
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.terminate()
        if self._reader is not None:
            self._reader.join(timeout=1.0)

    def summary(self) -> dict[str, int]:
        return dict(self._counts)


class RecorderFanout:
    """Present multiple recorder-compatible sinks as one receiver callback."""

    def __init__(self, *recorders: Any):
        self.recorders = tuple(
            recorder
            for recorder in recorders
            if recorder is not None and bool(getattr(recorder, "enabled", False))
        )
        self.enabled = bool(self.recorders)
        self.root = next(
            (getattr(recorder, "root") for recorder in self.recorders),
            "disabled",
        )

    def __getattr__(self, name: str):
        if not name.startswith("record_"):
            raise AttributeError(name)

        def forward(*args: Any, **kwargs: Any) -> bool:
            accepted = False
            for recorder in self.recorders:
                method = getattr(recorder, name, None)
                if method is None:
                    continue
                accepted = bool(method(*args, **kwargs)) or accepted
            return accepted

        return forward

    def close(self) -> None:
        for recorder in self.recorders:
            recorder.close()

    def summary(self) -> dict[str, Any]:
        return {
            type(recorder).__name__: recorder.summary()
            for recorder in self.recorders
        }
