import csv
import json
import tempfile
import unittest
from pathlib import Path

from runtime_config import load_runtime_config
from vio_dataset_recorder import VioDatasetRecorder


class VioDatasetRecorderTests(unittest.TestCase):
    def test_writes_lossless_camera_and_synchronized_sensor_files(self):
        jpeg = b"\xff\xd8original-simulator-jpeg\x00\xff\xd9"

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "vio_dataset"
            recorder = VioDatasetRecorder(
                root,
                config=load_runtime_config(),
                queue_size=32,
            )

            self.assertTrue(
                recorder.record_camera_frame(
                    frame_id=7,
                    sim_time_ns=123_456_789,
                    jpeg_bytes=jpeg,
                    receive_wall_time_ns=987_654_321,
                    width=640,
                    height=360,
                    total_chunks=3,
                )
            )
            self.assertTrue(
                recorder.record_imu(
                    {
                        "time_usec": 123_400,
                        "wall_time_ns": 987_600_000,
                        "xacc": 1.0,
                        "yacc": 2.0,
                        "zacc": 3.0,
                        "xgyro": 0.1,
                        "ygyro": 0.2,
                        "zgyro": 0.3,
                        "fields_updated": 4095,
                        "mavlink_seq": 17,
                        "mavlink_src_system": 1,
                        "mavlink_src_component": 2,
                    }
                )
            )
            self.assertTrue(
                recorder.record_imu_source(
                    "scaled_imu",
                    {
                        "source": "scaled_imu",
                        "message_id": 26,
                        "units_profile": "mG_mrad_s_mgauss_cdegC",
                        "timestamp_kind": "boot",
                        "time_boot_ms": 124,
                        "time_usec": 124_000,
                        "wall_time_ns": 987_601_000,
                        "xacc_raw": 1000,
                        "yacc_raw": -250,
                        "zacc_raw": -1000,
                        "xacc": 9.80665,
                        "yacc": -2.4516625,
                        "zacc": -9.80665,
                        "xgyro": 0.1,
                        "ygyro": 0.2,
                        "zgyro": 0.3,
                    },
                )
            )
            self.assertTrue(
                recorder.record_timesync(
                    direction="rx",
                    tc1=1_000,
                    ts1=2_000,
                    wall_time_ns=3_000,
                )
            )
            self.assertTrue(
                recorder.record_odometry(
                    {
                        "time_usec": 123_400,
                        "wall_time_ns": 987_600_000,
                        "x": 1.0,
                        "y": 2.0,
                        "z": -3.0,
                        "q_wxyz": (1.0, 0.0, 0.0, 0.0),
                        "vx": 4.0,
                        "vy": 5.0,
                        "vz": -6.0,
                        "pose_covariance": (0.1, 0.2),
                        "velocity_covariance": (0.3, 0.4),
                    }
                )
            )
            self.assertTrue(
                recorder.record_event(
                    "camera_duplicate_chunk",
                    frame_id=7,
                    chunk_id=1,
                    wall_time_ns=987_654_000,
                )
            )
            recorder.close()

            with (root / "camera" / "data.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                camera_rows = list(csv.DictReader(handle))
            self.assertEqual(len(camera_rows), 1)
            self.assertEqual(camera_rows[0]["sim_time_ns"], "123456789")
            self.assertEqual(camera_rows[0]["frame_id"], "7")
            self.assertEqual(camera_rows[0]["jpeg_size_bytes"], str(len(jpeg)))
            self.assertEqual(
                (root / "camera" / camera_rows[0]["filename"]).read_bytes(),
                jpeg,
            )

            with (root / "imu" / "data.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                imu_rows = list(csv.DictReader(handle))
            self.assertEqual(len(imu_rows), 1)
            self.assertEqual(imu_rows[0]["time_usec"], "123400")
            self.assertEqual(imu_rows[0]["zgyro"], "0.3")
            self.assertEqual(imu_rows[0]["mavlink_seq"], "17")
            self.assertEqual(imu_rows[0]["mavlink_src_system"], "1")
            self.assertEqual(imu_rows[0]["mavlink_src_component"], "2")

            with (root / "imu" / "scaled_imu.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                scaled_rows = list(csv.DictReader(handle))
            self.assertEqual(len(scaled_rows), 1)
            self.assertEqual(scaled_rows[0]["source"], "scaled_imu")
            self.assertEqual(scaled_rows[0]["xacc_raw"], "1000")
            self.assertEqual(scaled_rows[0]["xacc"], "9.80665")

            with (root / "timesync" / "data.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                timesync_rows = list(csv.DictReader(handle))
            self.assertEqual(timesync_rows[0]["direction"], "rx")
            self.assertEqual(timesync_rows[0]["tc1"], "1000")

            with (root / "truth" / "odometry.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                odometry_rows = list(csv.DictReader(handle))
            self.assertEqual(odometry_rows[0]["qw"], "1.0")
            self.assertEqual(odometry_rows[0]["pose_covariance_json"], "[0.1,0.2]")

            events = [
                json.loads(line)
                for line in (root / "events.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(events[0]["event"], "camera_duplicate_chunk")

            manifest = json.loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "complete")
            self.assertTrue(manifest["clean_shutdown"])
            self.assertEqual(manifest["counts"]["accepted"]["camera"], 1)
            self.assertEqual(manifest["counts"]["written"]["imu"], 1)
            self.assertEqual(manifest["counts"]["written"]["imu_scaled"], 1)
            self.assertEqual(
                manifest["files"]["imu_sources"]["scaled_imu"],
                "imu/scaled_imu.csv",
            )
            self.assertEqual(manifest["counts"]["dropped_queue_full"], {})
            self.assertTrue((root / "runtime.toml").is_file())

    def test_disabled_recorder_does_not_create_files(self):
        recorder = VioDatasetRecorder(None)
        self.assertFalse(recorder.enabled)
        self.assertFalse(recorder.record_imu({"time_usec": 1}))
        recorder.close()


if __name__ == "__main__":
    unittest.main()
