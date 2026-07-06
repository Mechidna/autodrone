from __future__ import annotations

import atexit
import json
import math
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

import cv2


class CameraFrameCapture:
    """Optional debug camera frame capture for replay_debug_map.py."""

    def __init__(self, *, source: str):
        self.source = str(source)
        root = os.environ.get("AIGP_CAMERA_CAPTURE_DIR", "").strip()
        self.enabled = bool(root)
        self.root = Path(root).expanduser() if self.enabled else None
        self.index_file = None
        self.next_capture_wall_time = 0.0
        self.count = 0
        self.saved_count = 0
        self.drop_count = 0
        self.write_failure_count = 0
        self.period_s = 1.0 / max(0.1, self._env_float("AIGP_CAMERA_CAPTURE_HZ", 30.0))
        self.jpeg_quality = int(
            max(1, min(100, self._env_float("AIGP_CAMERA_CAPTURE_JPEG_QUALITY", 80.0)))
        )
        self.queue_size = int(
            max(1, min(512, self._env_float("AIGP_CAMERA_CAPTURE_QUEUE_SIZE", 4.0)))
        )
        self.close_timeout_s = max(
            0.0,
            min(10.0, self._env_float("AIGP_CAMERA_CAPTURE_CLOSE_TIMEOUT_S", 1.0)),
        )
        self._queue: queue.Queue[dict[str, Any]] | None = None
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._counter_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False

        if not self.enabled or self.root is None:
            return

        self.root.mkdir(parents=True, exist_ok=True)
        self.index_file = (self.root / "index.jsonl").open("a", encoding="utf-8", buffering=1)
        self._queue = queue.Queue(maxsize=self.queue_size)
        self._worker = threading.Thread(
            target=self._writer_loop,
            name=f"aigp_camera_capture_{self.source}",
            daemon=True,
        )
        self._worker.start()
        atexit.register(self.close)

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            value = float(os.environ.get(name, default))
        except (TypeError, ValueError):
            return float(default)
        return value if math.isfinite(value) else float(default)

    def maybe_capture(self, frame_data: dict[str, Any], image) -> None:
        if (
            not self.enabled
            or self.root is None
            or self.index_file is None
            or self._queue is None
            or self._closed
        ):
            return

        wall_time = float(frame_data.get("wall_time") or time.time())
        if wall_time < self.next_capture_wall_time:
            return
        self.next_capture_wall_time = wall_time + self.period_s

        if self._queue.full():
            self._record_drop()
            return

        with self._counter_lock:
            self.count += 1
            sequence = self.count

        filename = f"frame_{sequence:06d}.jpg"
        path = self.root / filename

        height, width = image.shape[:2]
        event = {
            "source": self.source,
            "frame_id": frame_data.get("frame_id"),
            "wall_time": wall_time,
            "sim_time_ns": frame_data.get("sim_time_ns"),
            "ros_stamp_sec": frame_data.get("ros_stamp_sec"),
            "ros_stamp_nanosec": frame_data.get("ros_stamp_nanosec"),
            "width": int(width),
            "height": int(height),
            "path": filename,
        }

        item = {
            "event": event,
            "image": image.copy(),
            "path": path,
            "enqueue_wall_time": time.time(),
        }
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self._record_drop()

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

    def _record_drop(self) -> None:
        with self._counter_lock:
            self.drop_count += 1

    def _record_write_success(self) -> dict[str, int]:
        with self._counter_lock:
            self.saved_count += 1
            return {
                "capture_saved_total": int(self.saved_count),
                "capture_dropped_total": int(self.drop_count),
                "capture_write_failed_total": int(self.write_failure_count),
            }

    def _record_write_failure(self) -> None:
        with self._counter_lock:
            self.write_failure_count += 1

    def _writer_loop(self) -> None:
        assert self._queue is not None
        assert self.index_file is not None

        try:
            while not self._stop_event.is_set() or not self._queue.empty():
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                try:
                    self._write_item(item)
                finally:
                    self._queue.task_done()
        finally:
            self.index_file.close()

    def _write_item(self, item: dict[str, Any]) -> None:
        image = item["image"]
        path = Path(item["path"])
        write_start = time.time()
        ok = cv2.imwrite(
            str(path),
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
        )
        write_end = time.time()
        if not ok:
            self._record_write_failure()
            return

        event = dict(item["event"])
        event.update(self._record_write_success())
        event.update(
            {
                "capture_enqueue_wall_time": item.get("enqueue_wall_time"),
                "capture_write_wall_time": write_end,
                "capture_write_duration_ms": round((write_end - write_start) * 1000.0, 3),
                "capture_queue_depth_after_write": (
                    self._queue.qsize() if self._queue is not None else 0
                ),
                "capture_queue_size": int(self.queue_size),
                "capture_jpeg_quality": int(self.jpeg_quality),
            }
        )
        self.index_file.write(json.dumps(event, sort_keys=True) + "\n")
