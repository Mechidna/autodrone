from __future__ import annotations

import atexit
import csv
import datetime as dt
import json
import math
import os
import platform
import queue
import shutil
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any


class VioDatasetRecorder:
    """Asynchronously preserve the raw inputs needed for offline VIO testing."""

    MAVLINK_METADATA_FIELDS = (
        "mavlink_seq",
        "mavlink_src_system",
        "mavlink_src_component",
    )

    CAMERA_FIELDS = (
        "sim_time_ns",
        "frame_id",
        "filename",
        "receive_wall_time_ns",
        "width",
        "height",
        "jpeg_size_bytes",
        "total_chunks",
    )
    IMU_FIELDS = (
        "source",
        "message_id",
        "units_profile",
        "time_usec",
        "wall_time_ns",
        "xacc",
        "yacc",
        "zacc",
        "xgyro",
        "ygyro",
        "zgyro",
        "xmag",
        "ymag",
        "zmag",
        "abs_pressure",
        "diff_pressure",
        "pressure_alt",
        "temperature",
        "fields_updated",
        "sensor_id",
    ) + MAVLINK_METADATA_FIELDS
    ALTERNATIVE_IMU_FIELDS = (
        "source",
        "message_id",
        "units_profile",
        "timestamp_kind",
        "time_usec",
        "time_boot_ms",
        "wall_time_ns",
        "sensor_id",
        "xacc_raw",
        "yacc_raw",
        "zacc_raw",
        "xgyro_raw",
        "ygyro_raw",
        "zgyro_raw",
        "xmag_raw",
        "ymag_raw",
        "zmag_raw",
        "temperature_raw",
        "xacc",
        "yacc",
        "zacc",
        "xgyro",
        "ygyro",
        "zgyro",
        "xmag",
        "ymag",
        "zmag",
        "temperature",
        "fields_updated",
    ) + MAVLINK_METADATA_FIELDS
    TIMESYNC_FIELDS = (
        "direction",
        "wall_time_ns",
        "tc1",
        "ts1",
    )
    ATTITUDE_FIELDS = (
        "time_boot_ms",
        "wall_time_ns",
        "roll",
        "pitch",
        "yaw",
        "rollspeed",
        "pitchspeed",
        "yawspeed",
        "yaw_mavlink_raw",
        "yawspeed_mavlink_raw",
        "competition_yaw_inverted",
    ) + MAVLINK_METADATA_FIELDS
    LOCAL_POSITION_FIELDS = (
        "time_boot_ms",
        "wall_time_ns",
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
    ) + MAVLINK_METADATA_FIELDS
    ODOMETRY_FIELDS = (
        "time_usec",
        "wall_time_ns",
        "frame_id",
        "child_frame_id",
        "x",
        "y",
        "z",
        "qw",
        "qx",
        "qy",
        "qz",
        "vx",
        "vy",
        "vz",
        "rollspeed",
        "pitchspeed",
        "yawspeed",
        "reset_counter",
        "estimator_type",
        "quality",
        "pose_covariance_json",
        "velocity_covariance_json",
    ) + MAVLINK_METADATA_FIELDS

    CSV_OUTPUTS = {
        "camera": ("camera/data.csv", CAMERA_FIELDS),
        "imu": ("imu/data.csv", IMU_FIELDS),
        "imu_raw": ("imu/raw_imu.csv", ALTERNATIVE_IMU_FIELDS),
        "imu_scaled": ("imu/scaled_imu.csv", ALTERNATIVE_IMU_FIELDS),
        "imu_scaled2": ("imu/scaled_imu2.csv", ALTERNATIVE_IMU_FIELDS),
        "imu_scaled3": ("imu/scaled_imu3.csv", ALTERNATIVE_IMU_FIELDS),
        "imu_hil": ("imu/hil_sensor.csv", ALTERNATIVE_IMU_FIELDS),
        "timesync": ("timesync/data.csv", TIMESYNC_FIELDS),
        "attitude": ("truth/attitude.csv", ATTITUDE_FIELDS),
        "local_position": ("truth/local_position_ned.csv", LOCAL_POSITION_FIELDS),
        "odometry": ("truth/odometry.csv", ODOMETRY_FIELDS),
    }

    def __init__(self, root: str | os.PathLike[str] | None, *, config=None, queue_size=512):
        self.enabled = bool(root)
        self.root = Path(root).expanduser().resolve() if self.enabled else None
        self.config = config
        self.queue_size = max(16, min(8192, int(queue_size)))
        self.close_timeout_s = self._env_float("AIGP_VIO_CLOSE_TIMEOUT_S", 30.0)
        self.started_wall_time_ns = time.time_ns()
        self.started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
        self._queue: queue.Queue[dict[str, Any]] | None = None
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._close_lock = threading.Lock()
        self._counter_lock = threading.Lock()
        self._manifest_lock = threading.Lock()
        self._closed = False
        self._frame_sequence = 0
        self._accepted = Counter()
        self._written = Counter()
        self._dropped = Counter()
        self._write_failures = Counter()
        self._errors: list[str] = []
        self._max_queue_depth = 0
        self._files: dict[str, Any] = {}
        self._writers: dict[str, csv.DictWriter] = {}
        self._events_file = None

        if not self.enabled or self.root is None:
            return

        self._create_layout()
        self._queue = queue.Queue(maxsize=self.queue_size)
        self._worker = threading.Thread(
            target=self._writer_loop,
            name="aigp_vio_dataset_writer",
            daemon=True,
        )
        self._worker.start()
        self._write_manifest(status="recording", clean_shutdown=False)
        atexit.register(self.close)

    @classmethod
    def from_environment(cls, *, config=None) -> "VioDatasetRecorder":
        root = os.environ.get("AIGP_VIO_RECORD_DIR", "").strip()
        queue_size = cls._env_int("AIGP_VIO_QUEUE_SIZE", 512)
        return cls(root or None, config=config, queue_size=queue_size)

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, default))
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            value = float(os.environ.get(name, default))
        except (TypeError, ValueError):
            return float(default)
        return value if math.isfinite(value) and value >= 0.0 else float(default)

    def _create_layout(self) -> None:
        assert self.root is not None
        if self.root.exists() and any(self.root.iterdir()):
            raise RuntimeError(
                f"VIO dataset directory is not empty: {self.root}. "
                "Choose a new run id or recording directory."
            )
        (self.root / "camera" / "data").mkdir(parents=True, exist_ok=True)
        (self.root / "imu").mkdir(parents=True, exist_ok=True)
        (self.root / "timesync").mkdir(parents=True, exist_ok=True)
        (self.root / "truth").mkdir(parents=True, exist_ok=True)

        for kind, (relative_path, fieldnames) in self.CSV_OUTPUTS.items():
            path = self.root / relative_path
            handle = path.open("w", encoding="utf-8", newline="", buffering=1)
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            self._files[kind] = handle
            self._writers[kind] = writer

        self._events_file = (self.root / "events.jsonl").open(
            "w", encoding="utf-8", buffering=1
        )

        config_path = Path(getattr(self.config, "path", ""))
        if config_path.is_file():
            shutil.copyfile(config_path, self.root / "runtime.toml")

    def record_camera_frame(
        self,
        *,
        frame_id: int,
        sim_time_ns: int,
        jpeg_bytes: bytes,
        receive_wall_time_ns: int,
        width: int,
        height: int,
        total_chunks: int,
    ) -> bool:
        if not self.enabled:
            return False
        with self._counter_lock:
            self._frame_sequence += 1
            sequence = self._frame_sequence
        filename = (
            f"frame_{sequence:08d}_{int(sim_time_ns)}_{int(frame_id)}.jpg"
        )
        row = {
            "sim_time_ns": int(sim_time_ns),
            "frame_id": int(frame_id),
            "filename": f"data/{filename}",
            "receive_wall_time_ns": int(receive_wall_time_ns),
            "width": int(width),
            "height": int(height),
            "jpeg_size_bytes": len(jpeg_bytes),
            "total_chunks": int(total_chunks),
        }
        return self._enqueue(
            "camera",
            {
                "kind": "camera",
                "row": row,
                "filename": filename,
                "jpeg_bytes": bytes(jpeg_bytes),
            },
        )

    def record_imu(self, sample: dict[str, Any]) -> bool:
        return self._enqueue_row("imu", sample, self.IMU_FIELDS)

    def record_imu_source(self, source: str, sample: dict[str, Any]) -> bool:
        kind_by_source = {
            "raw_imu": "imu_raw",
            "scaled_imu": "imu_scaled",
            "scaled_imu2": "imu_scaled2",
            "scaled_imu3": "imu_scaled3",
            "hil_sensor": "imu_hil",
        }
        normalized_source = str(source).strip().lower()
        try:
            kind = kind_by_source[normalized_source]
        except KeyError as exc:
            raise ValueError(f"unsupported IMU source: {source}") from exc
        row = dict(sample)
        row["source"] = normalized_source
        return self._enqueue_row(kind, row, self.ALTERNATIVE_IMU_FIELDS)

    def record_timesync(
        self,
        *,
        direction: str,
        tc1: int,
        ts1: int,
        wall_time_ns: int,
    ) -> bool:
        return self._enqueue_row(
            "timesync",
            {
                "direction": str(direction),
                "tc1": int(tc1),
                "ts1": int(ts1),
                "wall_time_ns": int(wall_time_ns),
            },
            self.TIMESYNC_FIELDS,
        )

    def record_attitude(self, sample: dict[str, Any]) -> bool:
        return self._enqueue_row("attitude", sample, self.ATTITUDE_FIELDS)

    def record_local_position(self, sample: dict[str, Any]) -> bool:
        return self._enqueue_row(
            "local_position", sample, self.LOCAL_POSITION_FIELDS
        )

    def record_odometry(self, sample: dict[str, Any]) -> bool:
        q = tuple(sample.get("q_wxyz") or ())
        row = dict(sample)
        row.update(
            {
                "qw": q[0] if len(q) > 0 else "",
                "qx": q[1] if len(q) > 1 else "",
                "qy": q[2] if len(q) > 2 else "",
                "qz": q[3] if len(q) > 3 else "",
                "pose_covariance_json": self._compact_json(
                    sample.get("pose_covariance", ())
                ),
                "velocity_covariance_json": self._compact_json(
                    sample.get("velocity_covariance", ())
                ),
            }
        )
        return self._enqueue_row("odometry", row, self.ODOMETRY_FIELDS)

    def record_event(self, event_type: str, **details: Any) -> bool:
        event = {
            "event": str(event_type),
            "wall_time_ns": int(details.pop("wall_time_ns", time.time_ns())),
            **details,
        }
        return self._enqueue("event", {"kind": "event", "event": event})

    def _enqueue_row(self, kind: str, sample: dict[str, Any], fields) -> bool:
        if not self.enabled:
            return False
        row = {field: sample.get(field, "") for field in fields}
        return self._enqueue(kind, {"kind": kind, "row": row})

    def _enqueue(self, kind: str, item: dict[str, Any]) -> bool:
        if not self.enabled or self._queue is None or self._closed:
            return False
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            with self._counter_lock:
                self._dropped[kind] += 1
            return False

        depth = self._queue.qsize()
        with self._counter_lock:
            self._accepted[kind] += 1
            self._max_queue_depth = max(self._max_queue_depth, depth)
        return True

    def _writer_loop(self) -> None:
        assert self._queue is not None
        try:
            while not self._stop_event.is_set() or not self._queue.empty():
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                kind = str(item.get("kind", "unknown"))
                try:
                    self._write_item(item)
                except Exception as exc:
                    with self._counter_lock:
                        self._write_failures[kind] += 1
                        if len(self._errors) < 20:
                            self._errors.append(f"{kind}: {type(exc).__name__}: {exc}")
                finally:
                    self._queue.task_done()
        finally:
            self._close_outputs()
            if self._stop_event.is_set():
                self._write_manifest(status="complete", clean_shutdown=True)

    def _write_item(self, item: dict[str, Any]) -> None:
        kind = str(item["kind"])
        if kind == "camera":
            assert self.root is not None
            image_path = self.root / "camera" / "data" / str(item["filename"])
            image_path.write_bytes(item["jpeg_bytes"])
            self._writers[kind].writerow(item["row"])
        elif kind == "event":
            assert self._events_file is not None
            self._events_file.write(self._compact_json(item["event"]) + "\n")
        else:
            self._writers[kind].writerow(item["row"])
        with self._counter_lock:
            self._written[kind] += 1

    def close(self) -> None:
        if not self.enabled:
            return
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._stop_event.set()

        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=self.close_timeout_s)
        clean = worker is None or not worker.is_alive()
        self._write_manifest(
            status="complete" if clean else "incomplete_writer_timeout",
            clean_shutdown=clean,
        )

    def summary(self) -> dict[str, Any]:
        with self._counter_lock:
            return {
                "accepted": dict(self._accepted),
                "written": dict(self._written),
                "dropped_queue_full": dict(self._dropped),
                "write_failures": dict(self._write_failures),
                "max_queue_depth": int(self._max_queue_depth),
                "queue_capacity": int(self.queue_size),
            }

    def _close_outputs(self) -> None:
        for handle in self._files.values():
            try:
                handle.close()
            except OSError:
                pass
        if self._events_file is not None:
            try:
                self._events_file.close()
            except OSError:
                pass

    def _write_manifest(self, *, status: str, clean_shutdown: bool) -> None:
        if self.root is None:
            return
        config = self.config
        camera = getattr(config, "camera", None)
        manifest = {
            "format": "aigp_vio_dataset",
            "format_version": 2,
            "status": status,
            "clean_shutdown": bool(clean_shutdown),
            "started_utc": self.started_utc,
            "started_wall_time_ns": int(self.started_wall_time_ns),
            "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "updated_wall_time_ns": time.time_ns(),
            "host": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "pid": os.getpid(),
            },
            "runtime": {
                "profile": getattr(config, "profile", None),
                "runner_mode": getattr(getattr(config, "runtime", None), "runner_mode", None),
                "observe_only": getattr(
                    getattr(config, "runtime", None), "observe_only", None
                ),
                "vision_source": getattr(getattr(config, "vision", None), "source", None),
                "config_source": str(getattr(config, "path", "")) or None,
                "config_snapshot": "runtime.toml",
            },
            "camera": {
                "width": getattr(camera, "width", None),
                "height": getattr(camera, "height", None),
                "fx": getattr(camera, "fx", None),
                "fy": getattr(camera, "fy", None),
                "cx": getattr(camera, "cx", None),
                "cy": getattr(camera, "cy", None),
                "dist_coeffs": list(getattr(camera, "dist_coeffs", ()) or ()),
                "body_translation_m": list(
                    getattr(camera, "body_translation_m", ()) or ()
                ),
                "mount_profile": getattr(camera, "mount_profile", None),
                "yaw_correction_deg": getattr(camera, "yaw_correction_deg", None),
            },
            "clock_notes": {
                "camera": "sim_time_ns from the UDP frame header",
                "imu": "per-source MAVLink boot/epoch-or-boot timestamp retained in each CSV",
                "timesync": "raw tc1/ts1 plus local wall_time_ns for both tx and rx",
                "truth": "MAVLink timestamps; evaluation only, not a VIO input",
            },
            "files": {
                "camera_index": "camera/data.csv",
                "camera_images": "camera/data/*.jpg",
                "imu": "imu/data.csv",
                "imu_sources": {
                    "highres_imu": "imu/data.csv",
                    "raw_imu": "imu/raw_imu.csv",
                    "scaled_imu": "imu/scaled_imu.csv",
                    "scaled_imu2": "imu/scaled_imu2.csv",
                    "scaled_imu3": "imu/scaled_imu3.csv",
                    "hil_sensor": "imu/hil_sensor.csv",
                },
                "timesync": "timesync/data.csv",
                "attitude_truth": "truth/attitude.csv",
                "local_position_truth": "truth/local_position_ned.csv",
                "odometry_truth": "truth/odometry.csv",
                "events": "events.jsonl",
            },
            "counts": self.summary(),
            "errors": list(self._errors),
        }
        temp_path = self.root / "manifest.json.tmp"
        manifest_path = self.root / "manifest.json"
        with self._manifest_lock:
            temp_path.write_text(
                json.dumps(self._jsonable(manifest), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temp_path.replace(manifest_path)

    @classmethod
    def _compact_json(cls, value: Any) -> str:
        return json.dumps(cls._jsonable(value), separators=(",", ":"), sort_keys=True)

    @classmethod
    def _jsonable(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): cls._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._jsonable(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, float) and not math.isfinite(value):
            if math.isnan(value):
                return "nan"
            return "inf" if value > 0.0 else "-inf"
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)
