import sys
from types import ModuleType, SimpleNamespace

fake_pymavlink = ModuleType("pymavlink")
fake_pymavlink.mavutil = SimpleNamespace()
sys.modules.setdefault("pymavlink", fake_pymavlink)

from mavlink_rx import MAVLinkRX
from runtime_config import load_runtime_config


def _highres_imu_msg(time_usec):
    return SimpleNamespace(
        xacc=0.1,
        yacc=0.2,
        zacc=-9.7,
        xgyro=0.01,
        ygyro=0.02,
        zgyro=0.03,
        xmag=0.0,
        ymag=0.0,
        zmag=0.0,
        abs_pressure=0.0,
        diff_pressure=0.0,
        pressure_alt=0.0,
        temperature=20.0,
        fields_updated=0,
        time_usec=time_usec,
    )


def test_runtime_config_keeps_orbslam3_vio_disabled_by_default(tmp_path):
    config = load_runtime_config(tmp_path / "missing_runtime.toml")

    assert config.orbslam3_vio.enabled is False
    assert config.orbslam3_vio.host == "127.0.0.1"
    assert config.orbslam3_vio.port == 45530


def test_runtime_config_reads_orbslam3_vio_section(tmp_path):
    config_path = tmp_path / "runtime.toml"
    config_path.write_text(
        """
[orbslam3_vio]
enabled = true
host = "127.0.0.2"
port = 45531
hz = 20
connect_timeout_s = 1.5
request_timeout_s = 2.5
jpeg_quality = 80
max_stale_input_s = 1.0
max_imu_samples_per_frame = 32
source = "orb_slam3_test"
""",
        encoding="utf-8",
    )

    config = load_runtime_config(config_path)

    assert config.orbslam3_vio.enabled is True
    assert config.orbslam3_vio.host == "127.0.0.2"
    assert config.orbslam3_vio.port == 45531
    assert config.orbslam3_vio.hz == 20.0
    assert config.orbslam3_vio.connect_timeout_s == 1.5
    assert config.orbslam3_vio.request_timeout_s == 2.5
    assert config.orbslam3_vio.jpeg_quality == 80
    assert config.orbslam3_vio.max_stale_input_s == 1.0
    assert config.orbslam3_vio.max_imu_samples_per_frame == 32
    assert config.orbslam3_vio.source == "orb_slam3_test"


def test_mavlink_rx_stores_rolling_highres_imu_buffer():
    shared_data = {}
    rx = MAVLinkRX(None, shared_data, highres_imu_buffer_size=2)

    rx.on_highres_imu(_highres_imu_msg(100))
    rx.on_highres_imu(_highres_imu_msg(200))
    rx.on_highres_imu(_highres_imu_msg(300))

    assert shared_data["highres_imu"]["time_usec"] == 300
    assert [item["time_usec"] for item in shared_data["highres_imu_buffer"]] == [
        200,
        300,
    ]
