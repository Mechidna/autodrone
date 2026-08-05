import csv
import json
import math
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_openvins_dataset import DatasetError, prepare_dataset


def write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class PrepareOpenVinsDatasetTests(unittest.TestCase):
    def make_dataset(self, root: Path) -> Path:
        dataset = root / "vio_dataset"
        dataset.mkdir(parents=True)
        (dataset / "runtime.toml").write_text("# snapshot\n", encoding="utf-8")

        boot_start_us = 5_000_000
        server_start_ns = 1_700_000_000_000_000_000
        imu_rows = []
        for index in range(481):
            time_us = boot_start_us + index * 8_333
            source_ns = server_start_ns + (time_us - boot_start_us) * 1_000
            receive_delay_ns = 2_000_000 + (index % 7) * 100_000
            imu_rows.append(
                {
                    "time_usec": time_us,
                    "wall_time_ns": source_ns + receive_delay_ns,
                    "xacc": 0.0,
                    "yacc": 0.0,
                    "zacc": -9.81,
                    "xgyro": 0.1 if index >= 360 else 0.0,
                    "ygyro": 0.2 if index >= 360 else 0.0,
                    "zgyro": -0.3 if index >= 360 else 0.0,
                }
            )
        write_csv(
            dataset / "imu" / "data.csv",
            ("time_usec", "wall_time_ns", "xacc", "yacc", "zacc", "xgyro", "ygyro", "zgyro"),
            imu_rows,
        )
        scaled_rows = [
            {
                "source": "scaled_imu",
                "time_usec": row["time_usec"],
                "wall_time_ns": row["wall_time_ns"],
                "xacc": row["xacc"],
                "yacc": row["yacc"],
                "zacc": row["zacc"],
                "xgyro": row["xgyro"],
                "ygyro": row["ygyro"],
                "zgyro": row["zgyro"],
            }
            for row in imu_rows
        ]
        write_csv(
            dataset / "imu" / "scaled_imu.csv",
            (
                "source", "time_usec", "wall_time_ns", "xacc", "yacc",
                "zacc", "xgyro", "ygyro", "zgyro",
            ),
            scaled_rows,
        )

        timesync_rows = []
        for index in range(20):
            tx_ns = server_start_ns + index * 150_000_000
            rtt_ns = 1_000_000 + (index % 4) * 100_000
            timesync_rows.extend(
                (
                    {"direction": "tx", "wall_time_ns": tx_ns, "tc1": tx_ns, "ts1": 0},
                    {
                        "direction": "rx",
                        "wall_time_ns": tx_ns + rtt_ns,
                        "tc1": tx_ns + rtt_ns // 2,
                        "ts1": tx_ns,
                    },
                )
            )
        write_csv(
            dataset / "timesync" / "data.csv",
            ("direction", "wall_time_ns", "tc1", "ts1"),
            timesync_rows,
        )

        camera_rows = []
        image_dir = dataset / "camera" / "data"
        image_dir.mkdir(parents=True)
        jpeg = b"\xff\xd8synthetic\xff\xd9"
        for index in range(121):
            sim_ns = server_start_ns + 10_000_000 + index * 33_333_333
            filename = f"data/frame_{index:06d}.jpg"
            (dataset / "camera" / filename).write_bytes(jpeg)
            camera_rows.append(
                {
                    "sim_time_ns": sim_ns,
                    "frame_id": 100 + index,
                    "filename": filename,
                    "receive_wall_time_ns": sim_ns + 3_000_000,
                    "width": 640,
                    "height": 360,
                    "jpeg_size_bytes": len(jpeg),
                    "total_chunks": 1,
                }
            )
        write_csv(
            dataset / "camera" / "data.csv",
            (
                "sim_time_ns", "frame_id", "filename", "receive_wall_time_ns",
                "width", "height", "jpeg_size_bytes", "total_chunks",
            ),
            camera_rows,
        )

        truth_rows = []
        for index in range(4):
            truth_rows.append(
                {
                    "time_usec": boot_start_us + index * 1_000_000,
                    "x": index,
                    "y": 0,
                    "z": 0,
                    "qw": 1,
                    "qx": 0,
                    "qy": 0,
                    "qz": 0,
                    "vx": 1,
                    "vy": 0,
                    "vz": 0,
                }
            )
        truth_rows.append(dict(truth_rows[-1]))
        write_csv(
            dataset / "truth" / "odometry.csv",
            ("time_usec", "x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz"),
            truth_rows,
        )

        manifest = {
            "format": "aigp_vio_dataset",
            "format_version": 1,
            "status": "complete",
            "clean_shutdown": True,
            "errors": [],
            "counts": {"dropped_queue_full": {}, "write_failures": {}},
            "files": {
                "imu": "imu/data.csv",
                "imu_sources": {
                    "highres_imu": "imu/data.csv",
                    "scaled_imu": "imu/scaled_imu.csv",
                },
                "camera_index": "camera/data.csv",
                "timesync": "timesync/data.csv",
                "odometry_truth": "truth/odometry.csv",
            },
            "camera": {
                "width": 640,
                "height": 360,
                "fx": 320.0,
                "fy": 320.0,
                "cx": 320.0,
                "cy": 180.0,
                "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
                "body_translation_m": [0.0, 0.0, 0.0],
                "mount_profile": "competition",
            },
        }
        (dataset / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return dataset

    def test_prepares_aligned_replay_with_config_and_deduplicated_truth(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self.make_dataset(Path(temp_dir))
            output = Path(temp_dir) / "openvins_replay"

            report = prepare_dataset(dataset, output)

            self.assertEqual(report["validation"], "pass")
            self.assertEqual(report["imu"]["rows"], 481)
            self.assertGreater(report["imu"]["rate_hz"], 119.9)
            self.assertEqual(report["camera"]["overlap_rows"], 120)
            self.assertEqual(report["camera"]["stride"], 1)
            self.assertEqual(report["truth"]["source_rows"], 5)
            self.assertEqual(report["truth"]["unique_rows"], 4)
            self.assertAlmostEqual(
                report["clock_alignment"]["local_to_server_drift_ppm"], 0.0, places=5
            )

            with (output / "imu.csv").open(newline="", encoding="utf-8") as handle:
                imu_output = list(csv.DictReader(handle))
            self.assertEqual(imu_output[0]["timestamp"], "0.000000000")
            self.assertEqual(imu_output[1]["timestamp"], "0.008333000")
            self.assertEqual(imu_output[360]["wx"], "-0.1")
            self.assertEqual(imu_output[360]["wy"], "-0.2")
            self.assertEqual(imu_output[360]["wz"], "0.3")
            self.assertEqual(
                report["calibration"]["imu_input_transform"]["gyroscope_sign_xyz"],
                [-1.0, -1.0, -1.0],
            )
            self.assertFalse(
                report["calibration"]["imu_input_transform"]["source_capture_modified"]
            )
            self.assertEqual(
                report["camera"]["future_imu_bracket"]["rows"],
                report["camera"]["overlap_rows"],
            )
            self.assertGreater(
                report["camera"]["future_imu_bracket"]["lead_median_ms"], 0.0
            )
            self.assertFalse(
                report["estimator_policy"]["camera_imu_timeoffset_calibration"]
            )
            estimator_config = (
                output / "config" / "estimator_config.yaml"
            ).read_text(encoding="utf-8")
            self.assertIn("calib_cam_timeoffset: false", estimator_config)
            self.assertIn("max_clones: 11", estimator_config)
            self.assertIn("max_slam: 50", estimator_config)
            self.assertIn("max_slam_in_update: 25", estimator_config)
            self.assertIn("max_msckf_in_update: 40", estimator_config)
            self.assertIn(
                "msckf_geometry_ranked_selection: false", estimator_config
            )
            self.assertIn("track_frequency: 30.000", estimator_config)
            self.assertIn("num_pts: 300", estimator_config)
            self.assertIn("fast_threshold: 15", estimator_config)
            self.assertIn("min_px_dist: 10", estimator_config)
            self.assertIn("fi_max_dist: 200.0", estimator_config)
            self.assertIn("try_zupt: false", estimator_config)
            self.assertIn("zupt_only_at_beginning: false", estimator_config)
            self.assertEqual(report["estimator_policy"]["max_clones"], 11)
            self.assertEqual(
                report["estimator_policy"]["max_msckf_in_update"], 40
            )
            self.assertEqual(report["estimator_policy"]["max_slam"], 50)
            self.assertEqual(
                report["estimator_policy"]["max_slam_in_update"], 25
            )
            self.assertFalse(
                report["estimator_policy"]["geometry_ranked_msckf"]
            )
            self.assertEqual(
                report["estimator_policy"]["configured_camera_to_imu_timeoffset_ms"],
                0.0,
            )
            self.assertEqual(report["estimator_policy"]["camera_stride"], 1)
            self.assertEqual(report["estimator_policy"]["mask_top_rows"], 0)
            self.assertFalse(report["estimator_policy"]["mask_guide_cone"])
            self.assertEqual(
                report["estimator_policy"]["camera_image_mode"], "grayscale"
            )
            self.assertEqual(
                report["estimator_policy"]["histogram_method"], "HISTOGRAM"
            )
            self.assertEqual(
                report["estimator_policy"]["feature_max_distance_m"], 200.0
            )
            self.assertEqual(report["estimator_policy"]["num_pts"], 300)
            self.assertEqual(report["estimator_policy"]["fast_threshold"], 15)
            self.assertEqual(report["estimator_policy"]["min_px_dist"], 10)

            with (output / "cam0" / "data.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                camera_output = list(csv.DictReader(handle))
            self.assertEqual(camera_output[0]["source_frame_id"], "100")
            self.assertEqual(camera_output[-1]["source_frame_id"], "219")
            self.assertTrue((output / "cam0" / camera_output[0]["filename"]).is_file())

            camera_config = (
                output / "config" / "kalibr_imucam_chain.yaml"
            ).read_text(encoding="utf-8")
            self.assertIn("timeshift_cam_imu: 0.000000000", camera_config)
            self.assertIn("0.342020143326", camera_config)
            self.assertIn("0.939692620786", camera_config)
            self.assertTrue(math.isclose(report["imu"]["initial_accel_norm_mean_m_s2"], 9.81))

    def test_refuses_to_overwrite_an_existing_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self.make_dataset(Path(temp_dir))
            output = Path(temp_dir) / "openvins_replay"
            output.mkdir()
            with self.assertRaisesRegex(DatasetError, "output already exists"):
                prepare_dataset(dataset, output)

    def test_overrides_only_focal_length_for_controlled_intrinsics_sweep(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self.make_dataset(Path(temp_dir))
            output = Path(temp_dir) / "openvins_replay_fx280"

            report = prepare_dataset(dataset, output, camera_focal_px=280.0)

            camera_config = (
                output / "config" / "kalibr_imucam_chain.yaml"
            ).read_text(encoding="utf-8")
            self.assertIn(
                "intrinsics: [280.000000000000, 280.000000000000, "
                "320.000000000000, 180.000000000000]",
                camera_config,
            )
            self.assertEqual(
                report["calibration"]["source_intrinsics"],
                [320.0, 320.0, 320.0, 180.0],
            )
            self.assertEqual(
                report["calibration"]["intrinsics"],
                [280.0, 280.0, 320.0, 180.0],
            )
            self.assertEqual(
                report["estimator_policy"]["camera_focal_override_px"],
                280.0,
            )

    def test_rejects_out_of_range_focal_length_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self.make_dataset(Path(temp_dir))
            with self.assertRaisesRegex(DatasetError, "between 100 and 1000"):
                prepare_dataset(
                    dataset,
                    Path(temp_dir) / "openvins_replay_fx50",
                    camera_focal_px=50.0,
                )

    def test_overrides_only_camera_tilt_for_controlled_extrinsics_sweep(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self.make_dataset(Path(temp_dir))
            output = Path(temp_dir) / "openvins_replay_tilt15deg"

            report = prepare_dataset(
                dataset,
                output,
                camera_tilt_up_deg=15.0,
            )

            camera_config = (
                output / "config" / "kalibr_imucam_chain.yaml"
            ).read_text(encoding="utf-8")
            sine = math.sin(math.radians(15.0))
            cosine = math.cos(math.radians(15.0))
            self.assertIn(
                "      - [0.000000000000, "
                f"{sine:.12f}, {cosine:.12f}, 0.000000000000]",
                camera_config,
            )
            self.assertIn(
                "      - [0.000000000000, "
                f"{cosine:.12f}, {-sine:.12f}, 0.000000000000]",
                camera_config,
            )
            self.assertEqual(
                report["estimator_policy"]["camera_tilt_up_deg"],
                15.0,
            )
            self.assertEqual(report["calibration"]["camera_tilt_up_deg"], 15.0)
            self.assertEqual(
                report["calibration"]["nominal_camera_tilt_up_deg"],
                20.0,
            )
            self.assertEqual(
                report["calibration"]["p_camera_in_imu_m"],
                [0.0, 0.0, 0.0],
            )

    def test_rejects_out_of_range_camera_tilt_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self.make_dataset(Path(temp_dir))

            with self.assertRaisesRegex(
                DatasetError, "camera_tilt_up_deg must be between -89 and 89"
            ):
                prepare_dataset(
                    dataset,
                    Path(temp_dir) / "openvins_replay_bad_tilt",
                    camera_tilt_up_deg=90.0,
                )

    def test_writes_requested_clone_window(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self.make_dataset(Path(temp_dir))
            output = Path(temp_dir) / "openvins_replay_clones20"

            report = prepare_dataset(dataset, output, max_clones=20)

            estimator_config = (
                output / "config" / "estimator_config.yaml"
            ).read_text(encoding="utf-8")
            self.assertIn("max_clones: 20", estimator_config)
            self.assertEqual(report["estimator_policy"]["max_clones"], 20)

    def test_prepares_camera_stride_with_full_rate_imu(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self.make_dataset(Path(temp_dir))
            output = Path(temp_dir) / "openvins_replay_stride2"

            report = prepare_dataset(dataset, output, camera_stride=2)

            self.assertEqual(report["imu"]["rows"], 481)
            self.assertEqual(report["camera"]["source_overlap_rows"], 120)
            self.assertEqual(report["camera"]["overlap_rows"], 60)
            self.assertEqual(report["camera"]["stride"], 2)
            self.assertTrue(
                math.isclose(report["camera"]["rate_hz"], 15.0, rel_tol=1e-4)
            )
            estimator_config = (
                output / "config" / "estimator_config.yaml"
            ).read_text(encoding="utf-8")
            self.assertIn("max_clones: 11", estimator_config)
            self.assertIn("track_frequency: 15.000", estimator_config)
            with (output / "cam0" / "data.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                camera_output = list(csv.DictReader(handle))
            self.assertEqual(camera_output[0]["source_frame_id"], "100")
            self.assertEqual(camera_output[1]["source_frame_id"], "102")

    def test_writes_requested_msckf_update_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self.make_dataset(Path(temp_dir))
            output = Path(temp_dir) / "openvins_replay_msckf100"

            report = prepare_dataset(
                dataset,
                output,
                max_msckf_in_update=100,
            )

            estimator_config = (
                output / "config" / "estimator_config.yaml"
            ).read_text(encoding="utf-8")
            self.assertIn("max_msckf_in_update: 100", estimator_config)
            self.assertEqual(
                report["estimator_policy"]["max_msckf_in_update"], 100
            )

    def test_writes_requested_klt_detector_settings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self.make_dataset(Path(temp_dir))
            output = Path(temp_dir) / "openvins_replay_dense_tracker"

            report = prepare_dataset(
                dataset,
                output,
                num_pts=500,
                fast_threshold=10,
                min_px_dist=7,
            )

            estimator_config = (
                output / "config" / "estimator_config.yaml"
            ).read_text(encoding="utf-8")
            self.assertIn("num_pts: 500", estimator_config)
            self.assertIn("fast_threshold: 10", estimator_config)
            self.assertIn("min_px_dist: 7", estimator_config)
            self.assertEqual(report["estimator_policy"]["num_pts"], 500)
            self.assertEqual(report["estimator_policy"]["fast_threshold"], 10)
            self.assertEqual(report["estimator_policy"]["min_px_dist"], 7)

    def test_disables_persistent_slam_for_pure_msckf_replay(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self.make_dataset(Path(temp_dir))
            output = Path(temp_dir) / "openvins_replay_slam0"

            report = prepare_dataset(dataset, output, max_slam=0)

            estimator_config = (
                output / "config" / "estimator_config.yaml"
            ).read_text(encoding="utf-8")
            self.assertIn("max_slam: 0", estimator_config)
            self.assertEqual(report["estimator_policy"]["max_slam"], 0)

    def test_writes_requested_persistent_slam_update_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self.make_dataset(Path(temp_dir))
            output = Path(temp_dir) / "openvins_replay_slamupd50"

            report = prepare_dataset(
                dataset,
                output,
                max_slam=100,
                max_slam_in_update=50,
            )

            estimator_config = (
                output / "config" / "estimator_config.yaml"
            ).read_text(encoding="utf-8")
            self.assertIn("max_slam: 100", estimator_config)
            self.assertIn("max_slam_in_update: 50", estimator_config)
            self.assertEqual(report["estimator_policy"]["max_slam"], 100)
            self.assertEqual(
                report["estimator_policy"]["max_slam_in_update"], 50
            )

    def test_enables_post_validation_geometry_ranked_msckf_selection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self.make_dataset(Path(temp_dir))
            output = Path(temp_dir) / "openvins_replay_geoselect"

            report = prepare_dataset(
                dataset,
                output,
                max_msckf_in_update=20,
                geometry_ranked_msckf=True,
            )

            estimator_config = (
                output / "config" / "estimator_config.yaml"
            ).read_text(encoding="utf-8")
            self.assertIn("max_msckf_in_update: 20", estimator_config)
            self.assertIn(
                "msckf_geometry_ranked_selection: true", estimator_config
            )
            self.assertTrue(
                report["estimator_policy"]["geometry_ranked_msckf"]
            )

    def test_records_requested_top_image_mask(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self.make_dataset(Path(temp_dir))
            output = Path(temp_dir) / "openvins_replay_masktop120"

            report = prepare_dataset(dataset, output, mask_top_rows=120)

            self.assertEqual(report["estimator_policy"]["mask_top_rows"], 120)
            self.assertEqual(
                report["estimator_policy"]["feature_mask_activation"],
                "after_estimator_initialization",
            )

    def test_records_requested_guide_cone_mask_from_first_frame(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self.make_dataset(Path(temp_dir))
            output = Path(temp_dir) / "openvins_replay_maskguidecone"

            report = prepare_dataset(dataset, output, mask_guide_cone=True)
            policy = report["estimator_policy"]

            self.assertTrue(policy["mask_guide_cone"])
            self.assertEqual(
                policy["feature_mask_activation"],
                "from_first_camera_frame",
            )
            self.assertEqual(
                policy["guide_cone_mask_reference_resolution"],
                [640, 360],
            )
            self.assertEqual(len(policy["guide_cone_mask_polygon_normalized"]), 4)

    def test_records_requested_red_camera_channel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self.make_dataset(Path(temp_dir))
            output = Path(temp_dir) / "openvins_replay_redchannel"

            report = prepare_dataset(dataset, output, camera_image_mode="red")
            policy = report["estimator_policy"]

            self.assertEqual(policy["camera_image_mode"], "red")
            self.assertIn("BGR channel at index 2", policy["camera_image_preprocessing"])
            self.assertEqual(policy["mask_top_rows"], 0)
            self.assertFalse(policy["mask_guide_cone"])

    def test_writes_requested_histogram_method(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self.make_dataset(Path(temp_dir))
            output = Path(temp_dir) / "openvins_replay_histnone"

            report = prepare_dataset(dataset, output, histogram_method="NONE")

            estimator_config = (
                output / "config" / "estimator_config.yaml"
            ).read_text(encoding="utf-8")
            self.assertIn('histogram_method: "NONE"', estimator_config)
            self.assertEqual(report["estimator_policy"]["histogram_method"], "NONE")

    def test_records_fixed_red_contrast_policy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self.make_dataset(Path(temp_dir))
            output = Path(temp_dir) / "openvins_replay_redfixed_histnone"

            report = prepare_dataset(
                dataset,
                output,
                camera_image_mode="red_fixed",
                histogram_method="NONE",
            )
            policy = report["estimator_policy"]

            self.assertEqual(policy["camera_image_mode"], "red_fixed")
            self.assertEqual(policy["histogram_method"], "NONE")
            self.assertIn("reuse that LUT unchanged", policy["camera_image_preprocessing"])

    def test_fixed_red_contrast_rejects_adaptive_histogram(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self.make_dataset(Path(temp_dir))
            output = Path(temp_dir) / "openvins_replay_invalid"

            with self.assertRaisesRegex(
                DatasetError, "red_fixed.*requires histogram_method NONE"
            ):
                prepare_dataset(
                    dataset,
                    output,
                    camera_image_mode="red_fixed",
                    histogram_method="HISTOGRAM",
                )

    def test_records_side_histogram_without_feature_masks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self.make_dataset(Path(temp_dir))
            output = Path(temp_dir) / "openvins_replay_redsidehist_histnone"

            report = prepare_dataset(
                dataset,
                output,
                camera_image_mode="red_sidehist",
                histogram_method="NONE",
            )
            policy = report["estimator_policy"]

            self.assertEqual(policy["camera_image_mode"], "red_sidehist")
            self.assertEqual(policy["histogram_method"], "NONE")
            self.assertEqual(policy["mask_top_rows"], 0)
            self.assertFalse(policy["mask_guide_cone"])
            self.assertIn(
                "without masking any features",
                policy["camera_image_preprocessing"],
            )

    def test_side_histogram_rejects_spatial_feature_masks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self.make_dataset(Path(temp_dir))
            output = Path(temp_dir) / "openvins_replay_invalid"

            with self.assertRaisesRegex(
                DatasetError, "red_sidehist.*requires both spatial masks off"
            ):
                prepare_dataset(
                    dataset,
                    output,
                    camera_image_mode="red_sidehist",
                    histogram_method="NONE",
                    mask_top_rows=100,
                )

    def test_writes_and_brackets_requested_camera_imu_offset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self.make_dataset(Path(temp_dir))
            offset_output = Path(temp_dir) / "openvins_replay_dtp4ms"

            report = prepare_dataset(
                dataset,
                offset_output,
                camera_imu_timeoffset_ms=4.0,
            )

            camera_config = (
                offset_output / "config" / "kalibr_imucam_chain.yaml"
            ).read_text(encoding="utf-8")
            self.assertIn("timeshift_cam_imu: 0.004000000", camera_config)
            self.assertEqual(
                report["estimator_policy"]["configured_camera_to_imu_timeoffset_s"],
                0.004,
            )
            self.assertEqual(
                report["estimator_policy"]["configured_camera_to_imu_timeoffset_ms"],
                4.0,
            )
            with (offset_output / "imu.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                imu_timestamps = [
                    float(row["timestamp"]) for row in csv.DictReader(handle)
                ]
            with (offset_output / "cam0" / "data.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                camera_timestamps = [
                    float(row["timestamp"]) for row in csv.DictReader(handle)
                ]
            expected_leads_ms = []
            imu_index = 0
            for camera_timestamp in camera_timestamps:
                effective_timestamp = camera_timestamp + 0.004
                while (
                    imu_index < len(imu_timestamps)
                    and imu_timestamps[imu_index] <= effective_timestamp
                ):
                    imu_index += 1
                self.assertLess(imu_index, len(imu_timestamps))
                expected_leads_ms.append(
                    (imu_timestamps[imu_index] - effective_timestamp) * 1000.0
                )
            self.assertTrue(all(lead > 0.0 for lead in expected_leads_ms))
            self.assertAlmostEqual(
                report["camera"]["future_imu_bracket"]["lead_median_ms"],
                statistics.median(expected_leads_ms),
                places=6,
            )

    def test_selects_scaled_imu_with_si_units_and_spec_axis_signs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self.make_dataset(Path(temp_dir))
            output = Path(temp_dir) / "openvins_replay_scaled"

            report = prepare_dataset(
                dataset,
                output,
                imu_source="scaled_imu",
            )

            self.assertEqual(report["imu"]["source"], "scaled_imu")
            self.assertEqual(
                report["calibration"]["imu_input_transform"]["gyroscope_sign_xyz"],
                [1.0, 1.0, 1.0],
            )
            with (output / "imu.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[360]["wx"], "0.1")
            self.assertEqual(rows[360]["wy"], "0.2")
            self.assertEqual(rows[360]["wz"], "-0.3")

    def test_auto_stationary_start_trims_dynamic_capture_prefix_using_only_imu(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self.make_dataset(Path(temp_dir))
            imu_path = dataset / "imu" / "data.csv"
            with imu_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames
                imu_rows = list(reader)
            for row in imu_rows[:40]:
                row["xgyro"] = "1.0"
            write_csv(imu_path, fieldnames, imu_rows)

            with self.assertRaisesRegex(DatasetError, "first two seconds"):
                prepare_dataset(dataset, Path(temp_dir) / "strict_start")

            output = Path(temp_dir) / "auto_start"
            report = prepare_dataset(
                dataset,
                output,
                stationary_start="auto",
                initialization_mode="stationary",
            )

            initialization = report["imu"]["initialization"]
            self.assertEqual(initialization["start_policy"], "auto")
            self.assertEqual(initialization["source_rows_trimmed_before"], 40)
            self.assertGreater(initialization["source_start_offset_s"], 0.3)
            self.assertEqual(initialization["selection_inputs"], "IMU only")
            self.assertFalse(initialization["truth_used"])
            estimator_config = (
                output / "config" / "estimator_config.yaml"
            ).read_text(encoding="utf-8")
            self.assertIn("try_zupt: true", estimator_config)
            self.assertIn("zupt_only_at_beginning: true", estimator_config)
            self.assertTrue(report["estimator_policy"]["try_zupt"])
            self.assertTrue(
                report["estimator_policy"]["zupt_only_at_beginning"]
            )
            with (output / "imu.csv").open(newline="", encoding="utf-8") as handle:
                output_rows = list(csv.DictReader(handle))
            self.assertEqual(output_rows[0]["timestamp"], "0.000000000")

    def test_gap_policies_reject_allow_and_interpolate_an_imu_dropout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self.make_dataset(Path(temp_dir))
            imu_path = dataset / "imu" / "data.csv"
            with imu_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames
                imu_rows = list(reader)
            del imu_rows[241:250]
            write_csv(imu_path, fieldnames, imu_rows)

            with self.assertRaisesRegex(DatasetError, "reject policy"):
                prepare_dataset(dataset, Path(temp_dir) / "rejected")

            allow_output = Path(temp_dir) / "allowed"
            allow_report = prepare_dataset(
                dataset,
                allow_output,
                imu_gap_policy="allow",
                max_imu_gap_ms=100.0,
            )
            self.assertEqual(allow_report["imu"]["gap_policy"]["mode"], "allow")
            self.assertEqual(
                allow_report["imu"]["gap_policy"]["interpolated_rows"], 0
            )
            self.assertGreater(allow_report["imu"]["gap_max_ms"], 80.0)
            self.assertFalse(allow_report["imu"]["gap_policy"]["truth_used"])
            with (allow_output / "imu_gap_events.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                allow_events = list(csv.DictReader(handle))
            self.assertEqual(len(allow_events), 1)
            self.assertEqual(allow_events[0]["inserted_rows"], "0")

            interpolate_output = Path(temp_dir) / "interpolated"
            interpolate_report = prepare_dataset(
                dataset,
                interpolate_output,
                imu_gap_policy="interpolate",
                max_imu_gap_ms=100.0,
            )
            gap_policy = interpolate_report["imu"]["gap_policy"]
            self.assertEqual(gap_policy["mode"], "interpolate")
            self.assertEqual(gap_policy["interpolated_rows"], 9)
            self.assertEqual(interpolate_report["imu"]["rows"], 481)
            self.assertGreater(interpolate_report["imu"]["raw_gap_max_ms"], 80.0)
            self.assertLess(interpolate_report["imu"]["gap_max_ms"], 9.0)
            self.assertIn("adjacent", gap_policy["interpolation_method"])
            self.assertFalse(gap_policy["truth_used"])
            with (interpolate_output / "imu_gap_events.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                interpolate_events = list(csv.DictReader(handle))
            self.assertEqual(interpolate_events[0]["inserted_rows"], "9")

    def test_refuses_raw_imu_without_device_calibration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self.make_dataset(Path(temp_dir))
            output = Path(temp_dir) / "openvins_replay_raw"

            with self.assertRaisesRegex(DatasetError, "device-specific and unscaled"):
                prepare_dataset(dataset, output, imu_source="raw_imu")


if __name__ == "__main__":
    unittest.main()
