import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from autonomy_core.core.frame_conventions import (
    official_camera_to_body_frd_rotmat,
)
from orbslam3_vio_bridge import OrbSlam3VioBridge, OrbSlam3VioBridgeConfig


class FakeOrbSlam3Client:
    def __init__(self, response):
        self.response = response
        self.requests = []
        self.closed = False

    def request(self, payload):
        self.requests.append(payload)
        return dict(self.response)

    def close(self):
        self.closed = True


def _runtime_config():
    return SimpleNamespace(
        camera=SimpleNamespace(
            width=640,
            height=360,
            fx=320.0,
            fy=320.0,
            cx=320.0,
            cy=180.0,
            dist_coeffs=(0.0, 0.0, 0.0, 0.0, 0.0),
            body_translation_m=(0.0, 0.0, 0.0),
            mount_profile="racer_mono_cam",
            yaw_correction_deg=0.0,
        )
    )


def _shared_data():
    now = time.time()
    return {
        "lock": threading.Lock(),
        "latest_frame": {
            "frame_id": 7,
            "image": np.zeros((360, 640, 3), dtype=np.uint8),
            "shape": (360, 640, 3),
            "sim_time_ns": 123_456_789,
            "wall_time": now,
        },
        "highres_imu": {
            "xacc": 0.1,
            "yacc": 0.2,
            "zacc": -9.7,
            "xgyro": 0.01,
            "ygyro": 0.02,
            "zgyro": 0.03,
            "time_usec": 123_456,
            "wall_time": now,
        },
        "highres_imu_buffer": [
            {
                "xacc": 0.0,
                "yacc": 0.0,
                "zacc": -9.81,
                "xgyro": 0.0,
                "ygyro": 0.0,
                "zgyro": 0.0,
                "time_usec": 103_456,
                "wall_time": now,
            }
        ],
    }


def _imu_at(time_usec):
    return {
        "xacc": 0.0,
        "yacc": 0.0,
        "zacc": -9.81,
        "xgyro": 0.0,
        "ygyro": 0.0,
        "zgyro": 0.0,
        "time_usec": time_usec,
        "wall_time": time.time(),
    }


def test_step_once_sends_vio_request_and_writes_estimate():
    shared_data = _shared_data()
    client = FakeOrbSlam3Client(
        {
            "valid": True,
            "status": "tracking",
            "source": "orb_slam3",
            "pos_neu": [1.0, 2.0, 3.0],
            "vel_neu": [0.4, 0.5, 0.6],
            "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            "tracking_state": 2,
            "tracking_state_name": "tracking",
            "tracked_features": 120,
            "inlier_features": 100,
            "keyframes": 4,
        }
    )
    bridge = OrbSlam3VioBridge(
        shared_data,
        config=_runtime_config(),
        bridge_config=OrbSlam3VioBridgeConfig(enabled=False),
        client=client,
        autostart=False,
    )

    estimate = bridge.step_once()

    assert len(client.requests) == 1
    request = client.requests[0]
    assert request["type"] == "track_monocular_inertial"
    assert request["frame"]["frame_id"] == 7
    assert request["frame"]["timestamp_ns"] == 123_456_789
    assert request["frame"]["encoding"] == "bgr8_jpeg_base64"
    assert request["imu"][-1]["accel_xyz"] == [0.1, 0.2, -9.7]
    assert request["imu"][-1]["gyro_xyz"] == [0.01, 0.02, 0.03]
    assert request["camera"]["source"] == "runtime_config"
    assert request["camera"]["fx"] == 320.0

    assert estimate["valid"] is True
    assert estimate["status"] == "tracking"
    assert estimate["pos_neu"] == (1.0, 2.0, 3.0)
    assert estimate["pos_ned"] == (1.0, 2.0, -3.0)
    assert estimate["vel_neu"] == (0.4, 0.5, 0.6)
    assert estimate["frame_id"] == 7
    assert estimate["tracking_state"] == 2
    assert estimate["tracking_state_name"] == "tracking"
    assert estimate["tracked_features"] == 120

    assert shared_data["vio_estimate"] == estimate
    assert shared_data["vio_estimate_wall_time"] >= estimate["wall_time"]
    assert shared_data["vio_bridge_status"]["status"] == "tracking"
    assert shared_data["vio_bridge_status"]["tracking_state"] == 2
    assert shared_data["vio_bridge_status"]["tracked_features"] == 120

    diagnostics = shared_data["vio_diagnostics"]
    assert estimate["diagnostics"] == diagnostics
    assert diagnostics["clock"]["imu_count"] == 2
    assert diagnostics["clock"]["nearest_imu_dt_s"] < 1e-3
    assert diagnostics["imu"]["dominant_accel_axis"] == "-z"
    assert diagnostics["extrinsic"]["mount_profile"] == "racer_mono_cam"


def test_imu_samples_are_windowed_to_image_interval():
    shared_data = _shared_data()
    shared_data["latest_frame"]["frame_id"] = 8
    shared_data["latest_frame"]["sim_time_ns"] = 1_100_000_000
    shared_data["highres_imu_buffer"] = [
        _imu_at(time_usec) for time_usec in range(800_000, 1_100_001, 20_000)
    ]
    shared_data["highres_imu"] = _imu_at(1_120_000)
    client = FakeOrbSlam3Client(
        {
            "valid": False,
            "status": "initializing",
            "tracking_state": 1,
            "tracking_state_name": "initializing",
        }
    )
    bridge = OrbSlam3VioBridge(
        shared_data,
        config=_runtime_config(),
        bridge_config=OrbSlam3VioBridgeConfig(enabled=False, hz=10),
        client=client,
        autostart=False,
    )
    bridge.last_frame_timestamp_ns = 1_000_000_000

    bridge.step_once()

    imu_times = [sample["time_usec"] for sample in client.requests[0]["imu"]]
    assert imu_times == [1_020_000, 1_040_000, 1_060_000, 1_080_000, 1_100_000]
    assert 1_120_000 not in imu_times

    diagnostics = shared_data["vio_diagnostics"]
    assert diagnostics["clock"]["imu_count"] == 5
    assert diagnostics["clock"]["imu_span_s"] == pytest.approx(0.08)
    assert diagnostics["clock"]["image_minus_imu_last_s"] == pytest.approx(0.0)


def test_short_imu_window_includes_near_previous_boundary_sample():
    shared_data = _shared_data()
    shared_data["latest_frame"]["frame_id"] = 9
    shared_data["latest_frame"]["sim_time_ns"] = 1_020_000_000
    shared_data["highres_imu_buffer"] = [
        _imu_at(time_usec) for time_usec in (980_000, 1_000_000, 1_020_000)
    ]
    shared_data["highres_imu"] = _imu_at(1_020_000)
    client = FakeOrbSlam3Client(
        {
            "valid": False,
            "status": "initializing",
            "tracking_state": 1,
            "tracking_state_name": "initializing",
        }
    )
    bridge = OrbSlam3VioBridge(
        shared_data,
        config=_runtime_config(),
        bridge_config=OrbSlam3VioBridgeConfig(enabled=False, hz=10),
        client=client,
        autostart=False,
    )
    bridge.last_frame_timestamp_ns = 1_000_000_000

    bridge.step_once()

    imu_times = [sample["time_usec"] for sample in client.requests[0]["imu"]]
    assert imu_times == [1_000_000, 1_020_000]

    diagnostics = shared_data["vio_diagnostics"]
    assert diagnostics["clock"]["imu_count"] == 2
    assert diagnostics["clock"]["imu_span_s"] == pytest.approx(0.02)


def test_step_once_waits_when_only_one_new_imu_sample_is_available():
    shared_data = _shared_data()
    shared_data["latest_frame"]["frame_id"] = 10
    shared_data["latest_frame"]["sim_time_ns"] = 1_020_000_000
    shared_data["highres_imu_buffer"] = [
        _imu_at(time_usec) for time_usec in (980_000, 1_000_000, 1_020_000)
    ]
    shared_data["highres_imu"] = _imu_at(1_020_000)
    client = FakeOrbSlam3Client(
        {
            "valid": False,
            "status": "initializing",
            "tracking_state": 1,
            "tracking_state_name": "initializing",
        }
    )
    bridge = OrbSlam3VioBridge(
        shared_data,
        config=_runtime_config(),
        bridge_config=OrbSlam3VioBridgeConfig(enabled=False, hz=10),
        client=client,
        autostart=False,
    )
    bridge.last_frame_timestamp_ns = 1_000_000_000
    bridge.last_imu_time_usec = 1_000_000

    assert bridge.step_once() is None

    assert client.requests == []
    assert shared_data["vio_bridge_status"]["status"] == "waiting_for_imu_span"
    assert shared_data["vio_bridge_status"]["imu_count"] == 1


def test_step_once_waits_when_no_new_imu_sample_survives_filtering():
    shared_data = _shared_data()
    shared_data["latest_frame"]["frame_id"] = 11
    shared_data["latest_frame"]["sim_time_ns"] = 1_020_000_000
    shared_data["highres_imu_buffer"] = [
        _imu_at(time_usec) for time_usec in (980_000, 1_000_000, 1_020_000)
    ]
    shared_data["highres_imu"] = _imu_at(1_020_000)
    client = FakeOrbSlam3Client(
        {
            "valid": False,
            "status": "initializing",
            "tracking_state": 1,
            "tracking_state_name": "initializing",
        }
    )
    bridge = OrbSlam3VioBridge(
        shared_data,
        config=_runtime_config(),
        bridge_config=OrbSlam3VioBridgeConfig(enabled=False, hz=10),
        client=client,
        autostart=False,
    )
    bridge.last_frame_timestamp_ns = 1_000_000_000
    bridge.last_imu_time_usec = 1_020_000

    assert bridge.step_once() is None

    assert client.requests == []
    assert shared_data["vio_bridge_status"]["status"] == "waiting_for_new_imu"
    assert shared_data["vio_diagnostics"]["clock"]["imu_count"] == 0
    assert shared_data["vio_diagnostics"]["clock"]["imu_span_s"] is None


def test_step_once_sends_after_skipped_frame_accumulates_two_new_imu_samples():
    shared_data = _shared_data()
    shared_data["latest_frame"]["frame_id"] = 10
    shared_data["latest_frame"]["sim_time_ns"] = 1_020_000_000
    shared_data["highres_imu_buffer"] = [
        _imu_at(time_usec) for time_usec in (1_000_000, 1_020_000)
    ]
    shared_data["highres_imu"] = _imu_at(1_020_000)
    client = FakeOrbSlam3Client(
        {
            "valid": False,
            "status": "initializing",
            "tracking_state": 1,
            "tracking_state_name": "initializing",
        }
    )
    bridge = OrbSlam3VioBridge(
        shared_data,
        config=_runtime_config(),
        bridge_config=OrbSlam3VioBridgeConfig(enabled=False, hz=10),
        client=client,
        autostart=False,
    )
    bridge.last_frame_timestamp_ns = 1_000_000_000
    bridge.last_imu_time_usec = 1_000_000

    assert bridge.step_once() is None
    assert bridge.last_frame_id is None
    assert bridge.last_frame_timestamp_ns == 1_000_000_000
    assert client.requests == []

    shared_data["latest_frame"]["frame_id"] = 11
    shared_data["latest_frame"]["sim_time_ns"] = 1_040_000_000
    shared_data["highres_imu_buffer"] = [
        _imu_at(time_usec) for time_usec in (1_000_000, 1_020_000, 1_040_000)
    ]
    shared_data["highres_imu"] = _imu_at(1_040_000)

    bridge.step_once()

    imu_times = [sample["time_usec"] for sample in client.requests[0]["imu"]]
    assert imu_times == [1_020_000, 1_040_000]
    assert bridge.last_frame_id == 11
    assert bridge.last_imu_time_usec == 1_040_000


def test_step_once_waits_when_no_new_imu_sample_is_available():
    shared_data = _shared_data()
    shared_data["latest_frame"]["frame_id"] = 10
    shared_data["latest_frame"]["sim_time_ns"] = 1_020_000_000
    shared_data["highres_imu_buffer"] = [_imu_at(1_000_000)]
    shared_data["highres_imu"] = _imu_at(1_000_000)
    client = FakeOrbSlam3Client(
        {
            "valid": False,
            "status": "initializing",
            "tracking_state": 1,
            "tracking_state_name": "initializing",
        }
    )
    bridge = OrbSlam3VioBridge(
        shared_data,
        config=_runtime_config(),
        bridge_config=OrbSlam3VioBridgeConfig(enabled=False, hz=10),
        client=client,
        autostart=False,
    )
    bridge.last_frame_timestamp_ns = 1_000_000_000
    bridge.last_imu_time_usec = 1_000_000

    assert bridge.step_once() is None

    assert client.requests == []
    assert shared_data["vio_bridge_status"]["status"] == "waiting_for_new_imu"


def test_orbslam3_yaml_matches_bridge_rate_and_official_transform():
    path = Path(__file__).resolve().parents[1] / "config" / "orbslam3_racer_mono_inertial.yaml"
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    assert fs.isOpened()
    try:
        assert fs.getNode("Camera.fps").real() == pytest.approx(20.0)

        t_b_c = fs.getNode("IMU.T_b_c1").mat()
        assert t_b_c.shape == (4, 4)
        np.testing.assert_allclose(
            t_b_c[:3, :3],
            official_camera_to_body_frd_rotmat(),
            atol=1e-6,
        )
        np.testing.assert_allclose(t_b_c[:3, 3], np.zeros(3), atol=1e-12)
    finally:
        fs.release()


def test_step_once_ignores_duplicate_frame():
    shared_data = _shared_data()
    client = FakeOrbSlam3Client(
        {
            "valid": True,
            "status": "tracking",
            "pos_neu": [1.0, 2.0, 3.0],
        }
    )
    bridge = OrbSlam3VioBridge(
        shared_data,
        config=_runtime_config(),
        bridge_config=OrbSlam3VioBridgeConfig(enabled=False),
        client=client,
        autostart=False,
    )

    assert bridge.step_once() is not None
    assert bridge.step_once() is None

    assert len(client.requests) == 1
    assert shared_data["vio_bridge_status"]["status"] == "duplicate_frame"
