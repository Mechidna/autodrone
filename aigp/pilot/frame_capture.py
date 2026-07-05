from __future__ import annotations

import json
import math
import os
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
        self.period_s = 1.0 / max(0.1, self._env_float("AIGP_CAMERA_CAPTURE_HZ", 30.0))
        self.jpeg_quality = int(
            max(1, min(100, self._env_float("AIGP_CAMERA_CAPTURE_JPEG_QUALITY", 80.0)))
        )

        if not self.enabled or self.root is None:
            return

        self.root.mkdir(parents=True, exist_ok=True)
        self.index_file = (self.root / "index.jsonl").open("a", encoding="utf-8", buffering=1)

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            value = float(os.environ.get(name, default))
        except (TypeError, ValueError):
            return float(default)
        return value if math.isfinite(value) else float(default)

    def maybe_capture(self, frame_data: dict[str, Any], image) -> None:
        if not self.enabled or self.root is None or self.index_file is None:
            return

        wall_time = float(frame_data.get("wall_time") or time.time())
        if wall_time < self.next_capture_wall_time:
            return
        self.next_capture_wall_time = wall_time + self.period_s

        self.count += 1
        filename = f"frame_{self.count:06d}.jpg"
        path = self.root / filename
        ok = cv2.imwrite(
            str(path),
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
        )
        if not ok:
            return

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
        self.index_file.write(json.dumps(event, sort_keys=True) + "\n")
