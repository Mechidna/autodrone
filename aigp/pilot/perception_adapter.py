import threading
import time

from perception_wrapper import PerceptionWrapper
from runtime_config import load_runtime_config


class PerceptionAdapter:
    def __init__(self, data, hz=None, config=None):
        self.data = data
        self.config = config if config is not None else load_runtime_config()
        self.enabled = bool(self.config.perception.enabled)
        self.hz = float(hz if hz is not None else self.config.perception.hz)
        self.period_s = 1.0 / max(1.0, self.hz)
        self.last_processed_frame_id = None
        self.thread = None
        self.is_running = False
        if not self.enabled:
            self._write_status("disabled")
            return
        self.adapter = PerceptionWrapper(config=self.config)
        self.is_running = True
        self.thread = threading.Thread(
            target=self._loop,
            daemon=False,
        )
        self.thread.start()

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    def _get_lock(self):
        if isinstance(self.data, dict):
            return self.data.get("lock")
        return None

    def _read_inputs(self):
        lock = self._get_lock()
        if lock is not None:
            with lock:
                return {
                    "frame": self.data.get("latest_frame"),
                    "attitude": self.data.get("attitude"),
                    "odometry": self.data.get("odometry"),
                    "local_position_ned": self.data.get("local_position_ned"),
                }

        return {
            "frame": self.data.get("latest_frame"),
            "attitude": self.data.get("attitude"),
            "odometry": self.data.get("odometry"),
            "local_position_ned": self.data.get("local_position_ned"),
        }

    def _write_perception(self, latest_perception, status):
        lock = self._get_lock()
        updates = {
            "latest_perception": latest_perception,
            "latest_perception_status": status,
            "latest_perception_wall_time": time.time(),
        }

        if lock is not None:
            with lock:
                self.data.update(updates)
                self.data["perception_update_count"] = self.data.get("perception_update_count", 0) + 1
            return

        self.data.update(updates)
        self.data["perception_update_count"] = self.data.get("perception_update_count", 0) + 1

    def _write_status(self, status):
        lock = self._get_lock()
        updates = {
            "latest_perception_status": status,
            "latest_perception_wall_time": time.time(),
        }

        if lock is not None:
            with lock:
                self.data.update(updates)
            return

        self.data.update(updates)

    def _loop(self):
        while self.is_running:
            loop_start = time.monotonic()
            inputs = self._read_inputs()
            frame = inputs["frame"]

            if frame is None:
                self._write_status("no_frame")
                time.sleep(self.period_s)
                continue

            frame_id = int(frame.get("frame_id", -1))
            if frame_id == self.last_processed_frame_id:
                time.sleep(min(0.005, self.period_s))
                continue

            self.last_processed_frame_id = frame_id

            try:
                latest_perception = self.adapter.update(
                    frame=frame,
                    attitude=inputs["attitude"],
                    odometry=inputs["odometry"],
                    local_position_ned=inputs["local_position_ned"],
                )
                status = "ok"
            except Exception as exc:
                print(f"[perception_adapter] update failed: {exc}", flush=True)
                latest_perception = None
                status = f"error:{exc}"

            self._write_perception(latest_perception, status)

            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, self.period_s - elapsed))
