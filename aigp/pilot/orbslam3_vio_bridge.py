from __future__ import annotations

import base64
import json
import math
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 45530
DEFAULT_HZ = 30.0


@dataclass(frozen=True)
class OrbSlam3VioBridgeConfig:
    enabled: bool = False
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    hz: float = DEFAULT_HZ
    connect_timeout_s: float = 0.25
    request_timeout_s: float = 0.25
    jpeg_quality: int = 90
    max_stale_input_s: float = 0.5
    max_imu_samples_per_frame: int = 64
    source: str = "orb_slam3"


class OrbSlam3IpcClient:
    """
    Line-delimited JSON TCP client for an external ORB-SLAM3 wrapper.

    The Python runtime remains the owner of shared_data. The ORB-SLAM3 process
    sees serialized requests only.
    """

    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        connect_timeout_s: float = 0.25,
        request_timeout_s: float = 0.25,
    ):
        self.host = str(host)
        self.port = int(port)
        self.connect_timeout_s = float(connect_timeout_s)
        self.request_timeout_s = float(request_timeout_s)
        self._sock: Optional[socket.socket] = None
        self._reader = None

    def close(self) -> None:
        reader = self._reader
        self._reader = None
        if reader is not None:
            try:
                reader.close()
            except OSError:
                pass

        sock = self._sock
        self._sock = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._ensure_connected()
        assert self._sock is not None
        assert self._reader is not None

        data = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self._sock.settimeout(self.request_timeout_s)
        try:
            self._sock.sendall(data + b"\n")
            line = self._reader.readline()
        except OSError:
            self.close()
            raise

        if not line:
            self.close()
            raise RuntimeError("ORB-SLAM3 IPC connection closed")

        response = json.loads(line.decode("utf-8"))
        if not isinstance(response, dict):
            raise RuntimeError("ORB-SLAM3 IPC response must be a JSON object")
        return response

    def _ensure_connected(self) -> None:
        if self._sock is not None and self._reader is not None:
            return

        sock = socket.create_connection(
            (self.host, self.port),
            timeout=self.connect_timeout_s,
        )
        self._sock = sock
        self._reader = sock.makefile("rb")


class OrbSlam3VioBridge:
    """
    Python-side bridge from pilot shared_data to an external ORB-SLAM3 process.

    Reads:
      shared_data["latest_frame"]
      shared_data["highres_imu"]
      shared_data["highres_imu_buffer"] when present
      shared_data["camera_info"] when present

    Writes:
      shared_data["vio_estimate"]
      shared_data["vio_bridge_status"]
    """

    def __init__(
        self,
        shared_data: dict[str, Any],
        *,
        config=None,
        bridge_config: OrbSlam3VioBridgeConfig | None = None,
        client: Any = None,
        autostart: bool = True,
    ):
        self.shared_data = shared_data
        self.config = config
        self.bridge_config = bridge_config or self._config_from_runtime(config)
        self.client = client or OrbSlam3IpcClient(
            host=self.bridge_config.host,
            port=self.bridge_config.port,
            connect_timeout_s=self.bridge_config.connect_timeout_s,
            request_timeout_s=self.bridge_config.request_timeout_s,
        )
        self.thread: Optional[threading.Thread] = None
        self.is_running = False
        self.last_frame_id: Optional[int] = None
        self.last_frame_timestamp_ns: Optional[int] = None
        self.last_imu_time_usec: Optional[int] = None
        self.last_motion_position_neu: Optional[np.ndarray] = None
        self.last_motion_position_wall_time: Optional[float] = None
        self.reset_counter = 0

        if autostart and self.bridge_config.enabled:
            self.start()

    @classmethod
    def create_orbslam3_vio_bridge(
        cls,
        shared_data: dict[str, Any],
        *,
        config=None,
        bridge_config: OrbSlam3VioBridgeConfig | None = None,
        client: Any = None,
    ) -> "OrbSlam3VioBridge":
        return cls(
            shared_data,
            config=config,
            bridge_config=bridge_config,
            client=client,
            autostart=True,
        )

    def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(
            target=self._run_loop,
            name="orbslam3-vio-bridge",
            daemon=False,
        )
        self.thread.start()

    def get_thread_for_join(self):
        self.is_running = False
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
        return self.thread

    def step_once(self) -> Optional[dict[str, Any]]:
        snapshot = self._snapshot_inputs()
        if snapshot is None:
            self._store_status("waiting_for_inputs")
            return None

        frame = snapshot["frame"]
        frame_id = self._int_or_none(frame.get("frame_id"))
        if frame_id is not None and frame_id == self.last_frame_id:
            self._store_status("duplicate_frame")
            return None

        payload = self._build_request(snapshot)
        diagnostics = self._request_diagnostics(snapshot, payload)
        self._store_diagnostics(diagnostics)
        imu_payload = payload.get("imu")
        if not imu_payload:
            self._store_status("waiting_for_new_imu")
            return None
        if not self._has_usable_imu_window(imu_payload):
            clock = diagnostics.get("clock", {})
            self._store_status(
                "waiting_for_imu_span",
                imu_count=clock.get("imu_count"),
                imu_span_s=clock.get("imu_span_s"),
            )
            return None
        started = time.time()
        response = self.client.request(payload)
        estimate = self._estimate_from_response(
            response,
            frame=frame,
            request_wall_time=started,
            response_wall_time=time.time(),
        )
        estimate["diagnostics"] = diagnostics
        self._store_estimate(estimate)
        self._store_status(
            estimate.get("status", "unknown"),
            valid=estimate.get("valid"),
            tracking_state=estimate.get("tracking_state"),
            tracking_state_name=estimate.get("tracking_state_name"),
            tracked_features=estimate.get("tracked_features"),
            failure_reason=estimate.get("failure_reason"),
        )
        self.last_frame_id = frame_id
        frame_payload = payload.get("frame", {})
        if isinstance(frame_payload, dict):
            self.last_frame_timestamp_ns = self._int_or_none(
                frame_payload.get("timestamp_ns")
            )
        imu_payload = payload.get("imu", [])
        if isinstance(imu_payload, list) and imu_payload:
            last_imu = imu_payload[-1]
            if isinstance(last_imu, dict):
                self.last_imu_time_usec = self._int_or_none(last_imu.get("time_usec"))
        return estimate

    def _has_usable_imu_window(self, imu_payload: Any) -> bool:
        if not isinstance(imu_payload, list):
            return False

        times_usec = []
        for sample in imu_payload:
            if not isinstance(sample, dict):
                continue
            time_usec = self._int_or_none(sample.get("time_usec"))
            if time_usec is not None:
                times_usec.append(time_usec)

        if len(times_usec) < 2:
            return False
        return max(times_usec) > min(times_usec)

    def _run_loop(self) -> None:
        period_s = 1.0 / max(float(self.bridge_config.hz), 1e-6)
        next_tick = time.time()

        while self.is_running:
            now = time.time()
            if now < next_tick:
                time.sleep(min(next_tick - now, 0.01))
                continue
            next_tick = now + period_s

            try:
                self.step_once()
            except Exception as exc:
                self._store_status(f"error:{exc}")
                self._store_estimate(
                    {
                        "valid": False,
                        "status": "lost",
                        "source": self.bridge_config.source,
                        "failure_reason": str(exc),
                        "wall_time": time.time(),
                    }
                )
                close = getattr(self.client, "close", None)
                if callable(close):
                    close()

    def _snapshot_inputs(self) -> Optional[dict[str, Any]]:
        lock = self._lock()
        if lock is not None:
            with lock:
                return self._snapshot_inputs_unlocked()
        return self._snapshot_inputs_unlocked()

    def _snapshot_inputs_unlocked(self) -> Optional[dict[str, Any]]:
        frame = self.shared_data.get("latest_frame")
        imu = self.shared_data.get("highres_imu")
        if not isinstance(frame, dict) or not isinstance(imu, dict):
            return None

        frame_wall_time = self._float_or_none(frame.get("wall_time"))
        imu_wall_time = self._float_or_none(imu.get("wall_time"))
        now = time.time()
        max_stale = float(self.bridge_config.max_stale_input_s)
        if max_stale > 0.0:
            if frame_wall_time is not None and now - frame_wall_time > max_stale:
                return None
            if imu_wall_time is not None and now - imu_wall_time > max_stale:
                return None

        image = frame.get("image")
        if image is None:
            return None

        return {
            "frame": dict(frame),
            "imu": dict(imu),
            "imu_buffer": list(self.shared_data.get("highres_imu_buffer", []) or []),
            "camera_info": self._copy_optional_dict(self.shared_data.get("camera_info")),
            "attitude": self._copy_optional_dict(self.shared_data.get("attitude")),
            "local_position_ned": self._copy_optional_dict(
                self.shared_data.get("local_position_ned")
            ),
            "odometry": self._copy_optional_dict(self.shared_data.get("odometry")),
            "timesync": self._copy_optional_dict(self.shared_data.get("timesync")),
            "heartbeat": self._copy_optional_dict(self.shared_data.get("heartbeat")),
        }

    def _build_request(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        frame = snapshot["frame"]
        image = np.asarray(frame["image"])
        image_jpeg_b64 = self._encode_image_b64(image)
        timestamp_ns = self._timestamp_ns(frame)
        imu_samples = self._imu_samples(
            snapshot["imu"],
            snapshot["imu_buffer"],
            previous_frame_timestamp_ns=self.last_frame_timestamp_ns,
            current_frame_timestamp_ns=timestamp_ns,
        )
        camera = self._camera_payload(snapshot.get("camera_info"), image)

        return {
            "type": "track_monocular_inertial",
            "source": "python_shared_data",
            "frame": {
                "frame_id": self._int_or_none(frame.get("frame_id")),
                "timestamp_ns": timestamp_ns,
                "wall_time": self._float_or_none(frame.get("wall_time")),
                "shape": list(image.shape),
                "encoding": "bgr8_jpeg_base64",
                "jpeg_quality": int(self.bridge_config.jpeg_quality),
                "data": image_jpeg_b64,
            },
            "imu": imu_samples,
            "camera": camera,
            "timesync": snapshot.get("timesync"),
            "heartbeat": snapshot.get("heartbeat"),
        }

    def _imu_samples(
        self,
        latest_imu: dict[str, Any],
        imu_buffer: list[Any],
        *,
        previous_frame_timestamp_ns: Optional[int],
        current_frame_timestamp_ns: Optional[int],
    ) -> list[dict[str, Any]]:
        max_samples = int(max(1, self.bridge_config.max_imu_samples_per_frame))
        samples: list[dict[str, Any]] = []

        for item in imu_buffer:
            if isinstance(item, dict):
                sample = self._imu_sample(item)
                if sample is not None:
                    samples.append(sample)

        latest = self._imu_sample(latest_imu)
        if latest is not None:
            latest_time = latest.get("time_usec")
            if not samples or samples[-1].get("time_usec") != latest_time:
                samples.append(latest)

        if current_frame_timestamp_ns is None:
            return samples[-max_samples:]

        timed_samples: dict[int, dict[str, Any]] = {}
        for sample in samples:
            sample_time_ns = self._imu_sample_time_ns(sample)
            if sample_time_ns is None:
                continue
            timed_samples[sample_time_ns] = sample

        if not timed_samples:
            return samples[-max_samples:]

        current_ns = int(current_frame_timestamp_ns)
        previous_ns = self._int_or_none(previous_frame_timestamp_ns)
        if previous_ns is not None and previous_ns < current_ns:
            window_start_ns = previous_ns
        else:
            startup_window_s = max(
                0.05,
                min(0.25, 1.0 / max(float(self.bridge_config.hz), 1e-6)),
            )
            window_start_ns = current_ns - int(startup_window_s * 1_000_000_000)

        ordered = sorted(timed_samples.items())
        windowed_items = [
            (sample_time_ns, sample)
            for sample_time_ns, sample in ordered
            if window_start_ns < sample_time_ns <= current_ns
        ]

        boundary_gap_ns = 50_000_000
        if len(windowed_items) < 2:
            previous_items = [
                (sample_time_ns, sample)
                for sample_time_ns, sample in ordered
                if sample_time_ns <= window_start_ns
            ]
            if previous_items:
                previous_time_ns, previous_sample = previous_items[-1]
                if window_start_ns - previous_time_ns <= boundary_gap_ns:
                    windowed_items.insert(0, (previous_time_ns, previous_sample))

        if len(windowed_items) < 2:
            next_items = [
                (sample_time_ns, sample)
                for sample_time_ns, sample in ordered
                if sample_time_ns > current_ns
            ]
            if next_items:
                next_time_ns, next_sample = next_items[0]
                if next_time_ns - current_ns <= boundary_gap_ns:
                    windowed_items.append((next_time_ns, next_sample))

        if windowed_items:
            return self._only_unsent_imu_samples(
                [sample for _, sample in windowed_items[-max_samples:]]
            )

        before_or_at_frame = [
            sample for sample_time_ns, sample in ordered if sample_time_ns <= current_ns
        ]
        if before_or_at_frame:
            return self._only_unsent_imu_samples(before_or_at_frame[-max_samples:])

        return self._only_unsent_imu_samples([ordered[0][1]])

    def _only_unsent_imu_samples(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        last_sent = self._int_or_none(self.last_imu_time_usec)
        if last_sent is None:
            return samples

        filtered = []
        for sample in samples:
            sample_time = self._int_or_none(sample.get("time_usec"))
            if sample_time is None or sample_time > last_sent:
                filtered.append(sample)
        return filtered

    def _imu_sample(self, imu: dict[str, Any]) -> Optional[dict[str, Any]]:
        accel = self._vec3(imu.get("accel_xyz"))
        gyro = self._vec3(imu.get("gyro_xyz"))
        if accel is None:
            accel = self._vec3([imu.get("xacc"), imu.get("yacc"), imu.get("zacc")])
        if gyro is None:
            gyro = self._vec3([imu.get("xgyro"), imu.get("ygyro"), imu.get("zgyro")])
        if accel is None or gyro is None:
            return None

        return {
            "time_usec": self._int_or_none(imu.get("time_usec")),
            "wall_time": self._float_or_none(imu.get("wall_time")),
            "accel_xyz": accel.tolist(),
            "gyro_xyz": gyro.tolist(),
        }

    def _imu_sample_time_ns(self, sample: dict[str, Any]) -> Optional[int]:
        time_usec = self._int_or_none(sample.get("time_usec"))
        if time_usec is None:
            return None
        return int(time_usec) * 1000

    def _camera_payload(self, camera_info: Optional[dict[str, Any]], image: np.ndarray) -> dict[str, Any]:
        height, width = image.shape[:2]
        if camera_info:
            k = list(camera_info.get("k") or [])
            d = list(camera_info.get("d") or [])
            if len(k) >= 9:
                return {
                    "width": int(camera_info.get("width") or width),
                    "height": int(camera_info.get("height") or height),
                    "fx": float(k[0]),
                    "fy": float(k[4]),
                    "cx": float(k[2]),
                    "cy": float(k[5]),
                    "dist_coeffs": [float(v) for v in d],
                    "distortion_model": str(camera_info.get("distortion_model", "")),
                    "source": "camera_info",
                }

        camera = getattr(self.config, "camera", None)
        if camera is not None:
            return {
                "width": int(getattr(camera, "width", width)),
                "height": int(getattr(camera, "height", height)),
                "fx": float(getattr(camera, "fx", 0.0)),
                "fy": float(getattr(camera, "fy", 0.0)),
                "cx": float(getattr(camera, "cx", 0.0)),
                "cy": float(getattr(camera, "cy", 0.0)),
                "dist_coeffs": [float(v) for v in getattr(camera, "dist_coeffs", ())],
                "body_translation_m": [
                    float(v) for v in getattr(camera, "body_translation_m", (0.0, 0.0, 0.0))
                ],
                "mount_profile": str(getattr(camera, "mount_profile", "")),
                "yaw_correction_deg": float(getattr(camera, "yaw_correction_deg", 0.0)),
                "source": "runtime_config",
            }

        return {
            "width": int(width),
            "height": int(height),
            "fx": 0.0,
            "fy": 0.0,
            "cx": 0.0,
            "cy": 0.0,
            "dist_coeffs": [],
            "source": "image_shape_only",
        }

    def _estimate_from_response(
        self,
        response: dict[str, Any],
        *,
        frame: dict[str, Any],
        request_wall_time: float,
        response_wall_time: float,
    ) -> dict[str, Any]:
        status = str(response.get("status", "unknown"))
        valid = bool(response.get("valid", status in ("tracking", "ok")))

        pos_neu = self._vec3(response.get("pos_neu"))
        if pos_neu is None:
            pos_ned = self._vec3(response.get("pos_ned"))
            if pos_ned is not None:
                pos_neu = np.array([pos_ned[0], pos_ned[1], -pos_ned[2]], dtype=float)

        vel_neu = self._vec3(response.get("vel_neu"))
        if vel_neu is None:
            vel_ned = self._vec3(response.get("vel_ned"))
            if vel_ned is not None:
                vel_neu = np.array([vel_ned[0], vel_ned[1], -vel_ned[2]], dtype=float)

        quat_xyzw = self._quat_xyzw(response.get("quat_xyzw"))
        if quat_xyzw is None:
            quat_wxyz = self._quat_wxyz(response.get("quat_wxyz"))
            if quat_wxyz is not None:
                quat_xyzw = np.array(
                    [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
                    dtype=float,
                )

        rpy = self._rpy_from_response(response, quat_xyzw)
        estimate = {
            "valid": bool(valid and pos_neu is not None),
            "status": status,
            "source": str(response.get("source", self.bridge_config.source)),
            "timestamp_s": self._float_or_none(response.get("timestamp_s")),
            "timestamp_ns": self._int_or_none(response.get("timestamp_ns")),
            "frame_id": self._int_or_none(frame.get("frame_id")),
            "image_sim_time_ns": self._int_or_none(frame.get("sim_time_ns")),
            "wall_time": response_wall_time,
            "latency_s": max(0.0, response_wall_time - request_wall_time),
            "failure_reason": str(response.get("failure_reason", "")),
            "reset_counter": int(response.get("reset_counter", self.reset_counter) or 0),
            "tracking_state": self._int_or_none(response.get("tracking_state")),
            "tracking_state_name": str(response.get("tracking_state_name", "")),
            "tracked_features": self._int_or_none(response.get("tracked_features")),
            "inlier_features": self._int_or_none(response.get("inlier_features")),
            "outlier_features": self._int_or_none(response.get("outlier_features")),
            "keyframes": self._int_or_none(response.get("keyframes")),
            "map_points": self._int_or_none(response.get("map_points")),
            "map_count": self._int_or_none(response.get("map_count")),
            "request_count": self._int_or_none(response.get("request_count")),
            "imu_initialized": bool(response.get("imu_initialized", False)),
            "inertial_ba1": bool(response.get("inertial_ba1", False)),
            "inertial_ba2": bool(response.get("inertial_ba2", False)),
            "bad_imu": bool(response.get("bad_imu", False)),
            "local_mapper_initializing": bool(
                response.get("local_mapper_initializing", False)
            ),
            "keyframes_in_queue": self._int_or_none(response.get("keyframes_in_queue")),
            "local_mapper_init_time_s": self._float_or_none(
                response.get("local_mapper_init_time_s")
            ),
            "local_mapper_matches_inliers": self._int_or_none(
                response.get("local_mapper_matches_inliers")
            ),
            "tracker_matches_inliers": self._int_or_none(
                response.get("tracker_matches_inliers")
            ),
            "current_frame_id": self._int_or_none(response.get("current_frame_id")),
            "initial_frame_id": self._int_or_none(response.get("initial_frame_id")),
            "current_frame_keypoints": self._int_or_none(
                response.get("current_frame_keypoints")
            ),
            "initial_frame_keypoints": self._int_or_none(
                response.get("initial_frame_keypoints")
            ),
            "mono_ready_to_initialize": bool(
                response.get("mono_ready_to_initialize", False)
            ),
            "created_map": bool(response.get("created_map", False)),
            "max_frames": self._int_or_none(response.get("max_frames")),
            "frames_to_reset_imu": self._int_or_none(
                response.get("frames_to_reset_imu")
            ),
            "initializer_reason": str(response.get("initializer_reason", "")),
            "initializer_matches": self._int_or_none(
                response.get("initializer_matches")
            ),
            "initializer_triangulated": self._int_or_none(
                response.get("initializer_triangulated")
            ),
            "initializer_tracked_map_points": self._int_or_none(
                response.get("initializer_tracked_map_points")
            ),
            "initializer_median_depth": self._float_or_none(
                response.get("initializer_median_depth")
            ),
            "initializer_elapsed_s": self._float_or_none(
                response.get("initializer_elapsed_s")
            ),
            "initializer_reconstruct_success": bool(
                response.get("initializer_reconstruct_success", False)
            ),
            "pose_covariance": response.get("pose_covariance"),
            "velocity_covariance": response.get("velocity_covariance"),
            "bias_covariance": response.get("bias_covariance"),
        }

        if pos_neu is not None:
            estimate["pos_neu"] = tuple(float(v) for v in pos_neu)
            estimate["pos_ned"] = (float(pos_neu[0]), float(pos_neu[1]), float(-pos_neu[2]))
        if vel_neu is not None:
            estimate["vel_neu"] = tuple(float(v) for v in vel_neu)
            estimate["vel_ned"] = (float(vel_neu[0]), float(vel_neu[1]), float(-vel_neu[2]))
        if quat_xyzw is not None:
            estimate["quat_xyzw"] = tuple(float(v) for v in quat_xyzw)
        if rpy is not None:
            estimate["roll_rad"] = float(rpy[0])
            estimate["pitch_rad"] = float(rpy[1])
            estimate["yaw_rad"] = float(rpy[2])

        return estimate

    def _store_estimate(self, estimate: dict[str, Any]) -> None:
        lock = self._lock()
        if lock is not None:
            with lock:
                self.shared_data["vio_estimate"] = estimate
                self.shared_data["vio_estimate_wall_time"] = time.time()
        else:
            self.shared_data["vio_estimate"] = estimate
            self.shared_data["vio_estimate_wall_time"] = time.time()

    def _store_diagnostics(self, diagnostics: dict[str, Any]) -> None:
        lock = self._lock()
        if lock is not None:
            with lock:
                self.shared_data["vio_diagnostics"] = diagnostics
                self.shared_data["vio_diagnostics_wall_time"] = time.time()
        else:
            self.shared_data["vio_diagnostics"] = diagnostics
            self.shared_data["vio_diagnostics_wall_time"] = time.time()

    def _store_status(self, status: str, **extra: Any) -> None:
        lock = self._lock()
        value = {
            "status": str(status),
            "wall_time": time.time(),
            "source": self.bridge_config.source,
        }
        value.update({k: v for k, v in extra.items() if v is not None})
        if lock is not None:
            with lock:
                self.shared_data["vio_bridge_status"] = value
        else:
            self.shared_data["vio_bridge_status"] = value

    def _encode_image_b64(self, image: np.ndarray) -> str:
        quality = int(np.clip(self.bridge_config.jpeg_quality, 1, 100))
        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if not ok:
            raise RuntimeError("failed to JPEG-encode VIO frame")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def _timestamp_ns(self, frame: dict[str, Any]) -> Optional[int]:
        sim_time_ns = self._int_or_none(frame.get("sim_time_ns"))
        if sim_time_ns is not None:
            return sim_time_ns
        wall_time = self._float_or_none(frame.get("wall_time"))
        if wall_time is None:
            return None
        return int(wall_time * 1_000_000_000)

    def _request_diagnostics(
        self,
        snapshot: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        frame = snapshot["frame"]
        imu_samples = payload.get("imu", [])
        if not isinstance(imu_samples, list):
            imu_samples = []
        camera = payload.get("camera")
        if not isinstance(camera, dict):
            camera = {}

        clock = self._clock_diagnostics(frame, imu_samples)
        imu = self._imu_diagnostics(imu_samples)
        motion = self._motion_diagnostics(snapshot, imu)
        return {
            "clock": clock,
            "imu": imu,
            "motion": motion,
            "extrinsic": {
                "camera_source": camera.get("source"),
                "mount_profile": camera.get("mount_profile"),
                "body_translation_m": camera.get("body_translation_m"),
                "yaw_correction_deg": camera.get("yaw_correction_deg"),
                "camera_width": camera.get("width"),
                "camera_height": camera.get("height"),
                "fx": camera.get("fx"),
                "fy": camera.get("fy"),
                "cx": camera.get("cx"),
                "cy": camera.get("cy"),
                "note": "Python passes raw MAVLink HIGHRES_IMU accel/gyro through unchanged; ORB-SLAM3 applies IMU.T_b_c1 from its YAML.",
            },
        }

    def _clock_diagnostics(
        self,
        frame: dict[str, Any],
        imu_samples: list[Any],
    ) -> dict[str, Any]:
        image_timestamp_ns = self._timestamp_ns(frame)
        image_time_s = None if image_timestamp_ns is None else image_timestamp_ns / 1e9
        image_wall_time_s = self._float_or_none(frame.get("wall_time"))

        imu_times_s: list[float] = []
        imu_wall_times_s: list[float] = []
        for sample in imu_samples:
            if not isinstance(sample, dict):
                continue
            time_usec = self._int_or_none(sample.get("time_usec"))
            if time_usec is not None:
                imu_times_s.append(time_usec / 1e6)
            wall_time = self._float_or_none(sample.get("wall_time"))
            if wall_time is not None:
                imu_wall_times_s.append(wall_time)

        imu_first_s = imu_times_s[0] if imu_times_s else None
        imu_last_s = imu_times_s[-1] if imu_times_s else None
        imu_span_s = (
            float(imu_last_s - imu_first_s)
            if imu_first_s is not None and imu_last_s is not None
            else None
        )
        imu_dts = [
            imu_times_s[i] - imu_times_s[i - 1]
            for i in range(1, len(imu_times_s))
            if math.isfinite(imu_times_s[i] - imu_times_s[i - 1])
        ]

        image_minus_imu_last_s = (
            float(image_time_s - imu_last_s)
            if image_time_s is not None and imu_last_s is not None
            else None
        )
        nearest_imu_dt_s = (
            min(abs(image_time_s - t) for t in imu_times_s)
            if image_time_s is not None and imu_times_s
            else None
        )
        image_inside_imu_window = (
            bool(imu_first_s <= image_time_s <= imu_last_s)
            if image_time_s is not None
            and imu_first_s is not None
            and imu_last_s is not None
            else None
        )
        time_domain_warning = bool(
            nearest_imu_dt_s is None
            or nearest_imu_dt_s > 0.25
            or (
                image_minus_imu_last_s is not None
                and abs(image_minus_imu_last_s) > 1.0
            )
        )

        return {
            "image_timestamp_ns": image_timestamp_ns,
            "image_time_s": image_time_s,
            "image_wall_time_s": image_wall_time_s,
            "imu_count": len(imu_times_s),
            "imu_first_s": imu_first_s,
            "imu_last_s": imu_last_s,
            "imu_span_s": imu_span_s,
            "imu_dt_mean_s": float(np.mean(imu_dts)) if imu_dts else None,
            "imu_dt_max_s": float(np.max(imu_dts)) if imu_dts else None,
            "image_minus_imu_last_s": image_minus_imu_last_s,
            "nearest_imu_dt_s": None if nearest_imu_dt_s is None else float(nearest_imu_dt_s),
            "image_inside_imu_window": image_inside_imu_window,
            "latest_imu_wall_age_s": (
                time.time() - imu_wall_times_s[-1] if imu_wall_times_s else None
            ),
            "image_wall_age_s": (
                time.time() - image_wall_time_s if image_wall_time_s is not None else None
            ),
            "time_domain_warning": time_domain_warning,
        }

    def _imu_diagnostics(self, imu_samples: list[Any]) -> dict[str, Any]:
        accels = []
        gyros = []
        for sample in imu_samples:
            if not isinstance(sample, dict):
                continue
            accel = self._vec3(sample.get("accel_xyz"))
            gyro = self._vec3(sample.get("gyro_xyz"))
            if accel is not None:
                accels.append(accel)
            if gyro is not None:
                gyros.append(gyro)

        result: dict[str, Any] = {
            "count": len(accels),
            "frame_convention": "raw MAVLink HIGHRES_IMU body frame as received",
        }
        if accels:
            accel_arr = np.vstack(accels)
            accel_mean = np.mean(accel_arr, axis=0)
            accel_std = np.std(accel_arr, axis=0)
            accel_norm = np.linalg.norm(accel_arr, axis=1)
            dominant_idx = int(np.argmax(np.abs(accel_mean)))
            dominant_axis = ("x", "y", "z")[dominant_idx]
            dominant_sign = "+" if accel_mean[dominant_idx] >= 0.0 else "-"
            accel_dynamic = np.linalg.norm(accel_arr - accel_mean, axis=1)
            result.update(
                {
                    "accel_mean_xyz": [float(v) for v in accel_mean],
                    "accel_std_xyz": [float(v) for v in accel_std],
                    "accel_norm_mean_m_s2": float(np.mean(accel_norm)),
                    "accel_norm_min_m_s2": float(np.min(accel_norm)),
                    "accel_norm_max_m_s2": float(np.max(accel_norm)),
                    "accel_norm_minus_g_m_s2": float(np.mean(accel_norm) - 9.80665),
                    "accel_dynamic_mean_m_s2": float(np.mean(accel_dynamic)),
                    "dominant_accel_axis": f"{dominant_sign}{dominant_axis}",
                }
            )
        if gyros:
            gyro_arr = np.vstack(gyros)
            gyro_norm = np.linalg.norm(gyro_arr, axis=1)
            result.update(
                {
                    "gyro_mean_xyz_rad_s": [float(v) for v in np.mean(gyro_arr, axis=0)],
                    "gyro_norm_mean_rad_s": float(np.mean(gyro_norm)),
                    "gyro_norm_max_rad_s": float(np.max(gyro_norm)),
                }
            )
        return result

    def _motion_diagnostics(
        self,
        snapshot: dict[str, Any],
        imu_diagnostics: dict[str, Any],
    ) -> dict[str, Any]:
        position_neu, velocity_neu, position_source, position_wall_time = (
            self._position_velocity_neu(snapshot)
        )
        position_delta_m = None
        position_dt_s = None
        derived_speed_m_s = None
        if (
            position_neu is not None
            and position_wall_time is not None
            and self.last_motion_position_neu is not None
            and self.last_motion_position_wall_time is not None
        ):
            position_dt_s = position_wall_time - self.last_motion_position_wall_time
            if position_dt_s > 1e-6:
                position_delta_m = float(
                    np.linalg.norm(position_neu - self.last_motion_position_neu)
                )
                derived_speed_m_s = position_delta_m / position_dt_s

        if position_neu is not None:
            self.last_motion_position_neu = position_neu.copy()
            self.last_motion_position_wall_time = position_wall_time

        velocity_speed_m_s = (
            float(np.linalg.norm(velocity_neu)) if velocity_neu is not None else None
        )
        speed_m_s = velocity_speed_m_s
        if speed_m_s is None:
            speed_m_s = derived_speed_m_s

        gyro_max = self._float_or_none(imu_diagnostics.get("gyro_norm_max_rad_s"))
        accel_dynamic = self._float_or_none(
            imu_diagnostics.get("accel_dynamic_mean_m_s2")
        )
        low_excitation = None
        if speed_m_s is not None and gyro_max is not None and accel_dynamic is not None:
            low_excitation = bool(
                speed_m_s < 0.2 and gyro_max < 0.05 and accel_dynamic < 0.15
            )

        return {
            "position_source": position_source,
            "position_delta_m": position_delta_m,
            "position_dt_s": position_dt_s,
            "derived_speed_m_s": derived_speed_m_s,
            "velocity_speed_m_s": velocity_speed_m_s,
            "speed_m_s": speed_m_s,
            "gyro_norm_max_rad_s": gyro_max,
            "accel_dynamic_mean_m_s2": accel_dynamic,
            "low_excitation_hint": low_excitation,
        }

    def _position_velocity_neu(
        self,
        snapshot: dict[str, Any],
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str], Optional[float]]:
        for source_name in ("odometry", "local_position_ned"):
            source = snapshot.get(source_name)
            if not isinstance(source, dict):
                continue
            pos_neu = self._vec3(source.get("pos_neu"))
            vel_neu = self._vec3(source.get("vel_neu"))
            pos_ned = self._vec3(source.get("pos_ned"))
            vel_ned = self._vec3(source.get("vel_ned"))
            if pos_neu is None and pos_ned is not None:
                pos_neu = np.array([pos_ned[0], pos_ned[1], -pos_ned[2]], dtype=float)
            if vel_neu is None and vel_ned is not None:
                vel_neu = np.array([vel_ned[0], vel_ned[1], -vel_ned[2]], dtype=float)
            if pos_neu is None:
                x = self._float_or_none(source.get("x"))
                y = self._float_or_none(source.get("y"))
                z = self._float_or_none(source.get("z"))
                if x is not None and y is not None and z is not None:
                    pos_neu = np.array([x, y, -z], dtype=float)
            if vel_neu is None:
                vx = self._float_or_none(source.get("vx"))
                vy = self._float_or_none(source.get("vy"))
                vz = self._float_or_none(source.get("vz"))
                if vx is not None and vy is not None and vz is not None:
                    vel_neu = np.array([vx, vy, -vz], dtype=float)
            if pos_neu is not None or vel_neu is not None:
                return (
                    pos_neu,
                    vel_neu,
                    source_name,
                    self._float_or_none(source.get("wall_time")),
                )
        return None, None, None, None

    def _lock(self):
        if isinstance(self.shared_data, dict):
            return self.shared_data.get("lock")
        return None

    @staticmethod
    def _config_from_runtime(config) -> OrbSlam3VioBridgeConfig:
        section = getattr(config, "orbslam3_vio", None)
        if section is None:
            return OrbSlam3VioBridgeConfig()
        return OrbSlam3VioBridgeConfig(
            enabled=bool(getattr(section, "enabled", False)),
            host=str(getattr(section, "host", DEFAULT_HOST)),
            port=int(getattr(section, "port", DEFAULT_PORT)),
            hz=float(getattr(section, "hz", DEFAULT_HZ)),
            connect_timeout_s=float(getattr(section, "connect_timeout_s", 0.25)),
            request_timeout_s=float(getattr(section, "request_timeout_s", 0.25)),
            jpeg_quality=int(getattr(section, "jpeg_quality", 90)),
            max_stale_input_s=float(getattr(section, "max_stale_input_s", 0.5)),
            max_imu_samples_per_frame=int(
                getattr(section, "max_imu_samples_per_frame", 64)
            ),
            source=str(getattr(section, "source", "orb_slam3")),
        )

    @staticmethod
    def _copy_optional_dict(value) -> Optional[dict[str, Any]]:
        return dict(value) if isinstance(value, dict) else None

    @staticmethod
    def _vec3(value) -> Optional[np.ndarray]:
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=float).reshape(3)
        except (TypeError, ValueError):
            return None
        if not np.all(np.isfinite(arr)):
            return None
        return arr

    @staticmethod
    def _quat_xyzw(value) -> Optional[np.ndarray]:
        if value is None:
            return None
        try:
            quat = np.asarray(value, dtype=float).reshape(4)
        except (TypeError, ValueError):
            return None
        norm = float(np.linalg.norm(quat))
        if not math.isfinite(norm) or norm <= 0.0:
            return None
        return quat / norm

    @staticmethod
    def _quat_wxyz(value) -> Optional[np.ndarray]:
        return OrbSlam3VioBridge._quat_xyzw(value)

    @staticmethod
    def _rpy_from_response(
        response: dict[str, Any],
        quat_xyzw: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        rpy = OrbSlam3VioBridge._vec3(
            [
                response.get("roll_rad"),
                response.get("pitch_rad"),
                response.get("yaw_rad"),
            ]
        )
        if rpy is not None:
            return rpy
        if quat_xyzw is None:
            return None

        x, y, z, w = quat_xyzw
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw], dtype=float)

    @staticmethod
    def _int_or_none(value) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _float_or_none(value) -> Optional[float]:
        if value is None:
            return None
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if math.isfinite(out) else None
