import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_openvins_replay import (
    _startup_aligned_position_errors,
    analyze_gyroscope,
    analyze_replay,
    euler_body_frd_to_ned_matrix,
    replay_truth_body_to_ned,
    replay_truth_local_ned,
)


def write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class DiagnoseOpenVinsReplayTests(unittest.TestCase):
    def test_startup_position_alignment_uses_rotation_and_no_scale_fit(self):
        estimate_position = np.asarray(
            ((4.0, 2.0, 1.0), (5.0, 2.0, 1.0), (6.0, 2.0, 1.0))
        )
        global_to_imu = np.tile(np.eye(3), (3, 1, 1))
        body_to_ned = np.tile(
            np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
            (3, 1, 1),
        )
        truth_position = np.asarray(
            ((10.0, 20.0, -2.0), (10.0, 21.0, -2.0), (10.0, 22.0, -2.0))
        )

        aligned, errors = _startup_aligned_position_errors(
            estimate_position, global_to_imu, truth_position, body_to_ned
        )

        np.testing.assert_allclose(aligned, truth_position)
        np.testing.assert_allclose(errors, 0.0)

    def test_uses_captured_attitude_and_local_position_for_competition_truth(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            replay = root / "replay"
            dataset = root / "vio_dataset"
            replay.mkdir()
            timestamps_ms = (1000, 1010, 1020)
            rolls = (0.1, 0.2, 0.3)
            pitches = (0.05, 0.10, 0.15)
            yaws = (0.4, 0.4, 0.4)
            write_csv(
                dataset / "truth" / "attitude.csv",
                ("time_boot_ms", "roll", "pitch", "yaw"),
                (
                    {
                        "time_boot_ms": timestamp,
                        "roll": roll,
                        "pitch": pitch,
                        "yaw": yaw,
                    }
                    for timestamp, roll, pitch, yaw in zip(
                        timestamps_ms, rolls, pitches, yaws
                    )
                ),
            )
            write_csv(
                dataset / "truth" / "local_position_ned.csv",
                ("time_boot_ms", "x", "y", "z", "vx", "vy", "vz"),
                (
                    {
                        "time_boot_ms": timestamp,
                        "x": index,
                        "y": 2 * index,
                        "z": -index,
                        "vx": 1.0,
                        "vy": 2.0,
                        "vz": -1.0,
                    }
                    for index, timestamp in enumerate(timestamps_ms)
                ),
            )
            (replay / "manifest.json").write_text(
                json.dumps(
                    {
                        # Simulate a manifest prepared under another OS. The
                        # sibling vio_dataset remains portable across WSL and
                        # Windows and should be selected as the fallback.
                        "source_dataset": "/mnt/z/nonexistent/vio_dataset",
                        "clock_alignment": {
                            "imu_to_server_offset_ns": 0,
                            "replay_origin_server_ns": 1_000_000_000,
                        },
                    }
                ),
                encoding="utf-8",
            )
            target_time = np.array((0.0, 0.01, 0.02))
            fallback = np.tile(np.array((0.0, 0.0, 0.0, 1.0)), (3, 1))

            rotations, orientation_method = replay_truth_body_to_ned(
                replay, target_time, fallback
            )
            expected_rotations = euler_body_frd_to_ned_matrix(
                np.asarray(rolls), -np.asarray(pitches), np.asarray(yaws)
            )
            np.testing.assert_allclose(rotations, expected_rotations)
            self.assertFalse(orientation_method["odometry_quaternion_used"])

            local_truth = replay_truth_local_ned(replay, target_time)
            self.assertIsNotNone(local_truth)
            position, velocity, translation_method = local_truth
            np.testing.assert_allclose(position[-1], (2.0, 4.0, -2.0))
            np.testing.assert_allclose(velocity[-1], (1.0, 2.0, -1.0))
            self.assertFalse(translation_method["odometry_translation_used"])

    def make_replay(self, root: Path) -> Path:
        replay = root / "replay"
        replay.mkdir()
        timestamps = np.arange(0.0, 10.001, 0.01)
        yaw = math.pi / 2.0
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)

        velocity_body = np.column_stack(
            (
                2.0 + np.sin(0.7 * timestamps),
                0.8 * np.cos(0.5 * timestamps),
                np.zeros_like(timestamps),
            )
        )
        # Constant +90 degree yaw: body x maps to NED y and body y to NED -x.
        velocity_ned = np.column_stack(
            (-velocity_body[:, 1], velocity_body[:, 0], velocity_body[:, 2])
        )
        positions = np.zeros_like(velocity_ned)
        dt = np.diff(timestamps)
        positions[1:] = np.cumsum(
            0.5 * (velocity_ned[1:] + velocity_ned[:-1]) * dt[:, None], axis=0
        )

        acceleration_body = np.column_stack(
            (
                0.7 * np.cos(0.7 * timestamps),
                -0.4 * np.sin(0.5 * timestamps),
                np.full_like(timestamps, -9.81),
            )
        )
        write_csv(
            replay / "truth.csv",
            ("timestamp", "px", "py", "pz", "vx", "vy", "vz", "qx", "qy", "qz", "qw"),
            (
                {
                    "timestamp": timestamp,
                    "px": position[0],
                    "py": position[1],
                    "pz": position[2],
                    "vx": velocity[0],
                    "vy": velocity[1],
                    "vz": velocity[2],
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": qz,
                    "qw": qw,
                }
                for timestamp, position, velocity in zip(
                    timestamps, positions, velocity_body
                )
            ),
        )
        # Deliberately remove the dynamic y acceleration while retaining x/z.
        write_csv(
            replay / "imu.csv",
            ("timestamp", "wx", "wy", "wz", "ax", "ay", "az"),
            (
                {
                    "timestamp": timestamp,
                    "wx": 0.0,
                    "wy": 0.0,
                    "wz": 0.0,
                    "ax": acceleration[0],
                    "ay": 0.0,
                    "az": acceleration[2],
                }
                for timestamp, acceleration in zip(timestamps, acceleration_body)
            ),
        )
        write_csv(
            replay / "openvins_update_diagnostics.csv",
            (
                "camera_timestamp", "source_frame_id", "initialized",
                "state_timestamp", "state_advanced", "msckf_features_used",
                "slam_features_in_state", "active_tracks",
                "msckf_lost_candidates", "msckf_marginal_candidates",
                "msckf_maxtrack_candidates", "msckf_candidates_before_limit",
                "msckf_candidates_after_limit", "msckf_input_features",
                "msckf_input_measurements", "msckf_rejected_too_few",
                "msckf_rejected_triangulation",
                "msckf_triangulation_bad_condition",
                "msckf_triangulation_depth_too_near",
                "msckf_triangulation_depth_too_far",
                "msckf_triangulation_invalid_numeric",
                "msckf_triangulation_other", "msckf_rejected_refinement",
                "msckf_refinement_depth_too_near",
                "msckf_refinement_depth_too_far",
                "msckf_refinement_baseline_ratio",
                "msckf_refinement_invalid_numeric",
                "msckf_refinement_other",
                "msckf_chi2_tested", "msckf_rejected_chi2",
                "msckf_geometry_valid_features", "msckf_post_chi2_features",
                "msckf_rejected_selection_limit", "msckf_selected_features",
                "msckf_accepted_features", "msckf_accepted_measurements",
                "msckf_chi2_ratio_mean", "msckf_chi2_ratio_max",
                "position_norm", "speed", "bias_gyro_norm", "bias_accel_norm",
            ),
            (
                {
                    "camera_timestamp": timestamp,
                    "source_frame_id": index,
                    "initialized": 1,
                    "state_timestamp": timestamp,
                    "state_advanced": 1,
                    "msckf_features_used": 20,
                    "slam_features_in_state": 10,
                    "active_tracks": 100,
                    "msckf_lost_candidates": 20,
                    "msckf_marginal_candidates": 0,
                    "msckf_maxtrack_candidates": 0,
                    "msckf_candidates_before_limit": 20,
                    "msckf_candidates_after_limit": 20,
                    "msckf_input_features": 20,
                    "msckf_input_measurements": 120,
                    "msckf_rejected_too_few": 0,
                    "msckf_rejected_triangulation": 0,
                    "msckf_triangulation_bad_condition": 0,
                    "msckf_triangulation_depth_too_near": 0,
                    "msckf_triangulation_depth_too_far": 0,
                    "msckf_triangulation_invalid_numeric": 0,
                    "msckf_triangulation_other": 0,
                    "msckf_rejected_refinement": 0,
                    "msckf_refinement_depth_too_near": 0,
                    "msckf_refinement_depth_too_far": 0,
                    "msckf_refinement_baseline_ratio": 0,
                    "msckf_refinement_invalid_numeric": 0,
                    "msckf_refinement_other": 0,
                    "msckf_chi2_tested": 20,
                    "msckf_rejected_chi2": 0,
                    "msckf_geometry_valid_features": 20,
                    "msckf_post_chi2_features": 20,
                    "msckf_rejected_selection_limit": 0,
                    "msckf_selected_features": 20,
                    "msckf_accepted_features": 20,
                    "msckf_accepted_measurements": 120,
                    "msckf_chi2_ratio_mean": 0.2,
                    "msckf_chi2_ratio_max": 0.4,
                    "position_norm": 0,
                    "speed": 0,
                    "bias_gyro_norm": 0,
                    "bias_accel_norm": 0,
                }
                for index, timestamp in enumerate(timestamps[::3])
            ),
        )
        write_csv(
            replay / "openvins_feature_geometry.csv",
            (
                "camera_timestamp", "source_frame_id", "feature_id", "result",
                "first_u_px", "first_v_px", "last_u_px", "last_v_px",
                "mean_u_px", "mean_v_px", "min_u_px", "max_u_px",
                "min_v_px", "max_v_px",
                "triangulation_success", "refinement_success", "chi2_tested",
                "selected_for_update", "selection_rank",
                "selection_grid_row", "selection_grid_col", "accepted",
                "triangulation_failure_reason",
                "refinement_failure_reason", "observation_count",
                "track_duration_s", "max_parallax_deg",
                "max_camera_baseline_m", "condition_number", "linear_depth_m",
                "linear_range_m", "refined_depth_m", "refined_range_m",
                "chi2_ratio",
            ),
            (
                {
                    "camera_timestamp": 1.0,
                    "source_frame_id": index,
                    "feature_id": index,
                    "result": result,
                    "first_u_px": mean_u - 2.0,
                    "first_v_px": mean_v - 1.0,
                    "last_u_px": mean_u + 2.0,
                    "last_v_px": mean_v + 1.0,
                    "mean_u_px": mean_u,
                    "mean_v_px": mean_v,
                    "min_u_px": mean_u - 2.0,
                    "max_u_px": mean_u + 2.0,
                    "min_v_px": mean_v - 1.0,
                    "max_v_px": mean_v + 1.0,
                    "triangulation_success": int(result != 1),
                    "refinement_success": int(result not in (1, 2)),
                    "chi2_tested": int(result in (3, 4)),
                    "selected_for_update": int(result == 4),
                    "selection_rank": 1 if result == 4 else 0,
                    "selection_grid_row": int(mean_v // 120),
                    "selection_grid_col": int(mean_u // (640 / 3)),
                    "accepted": int(result == 4),
                    "triangulation_failure_reason": int(result == 1),
                    "refinement_failure_reason": 5 if result == 2 else 0,
                    "observation_count": observations,
                    "track_duration_s": duration,
                    "max_parallax_deg": parallax,
                    "max_camera_baseline_m": baseline,
                    "condition_number": condition,
                    "linear_depth_m": depth,
                    "linear_range_m": depth,
                    "refined_depth_m": depth if result != 1 else math.nan,
                    "refined_range_m": depth if result != 1 else math.nan,
                    "chi2_ratio": chi2,
                }
                for index, (
                    result, observations, duration, parallax, baseline,
                    condition, depth, chi2, mean_u, mean_v,
                ) in enumerate(
                    (
                        (1, 3, 0.067, 0.2, 0.05, 20_000.0, 80.0, math.nan, 320.0, 180.0),
                        (2, 5, 0.133, 0.7, 0.15, 500.0, 60.0, math.nan, 500.0, 80.0),
                        (3, 7, 0.200, 1.5, 0.30, 100.0, 30.0, 2.0, 500.0, 300.0),
                        (4, 8, 0.233, 2.5, 0.40, 50.0, 20.0, 0.5, 50.0, 300.0),
                    )
                )
            ),
        )
        write_csv(
            replay / "openvins_slam_feature_diagnostics.csv",
            (
                "camera_timestamp", "source_frame_id", "feature_id",
                "tracked_in_current_frame", "u_px", "v_px", "depth_m",
                "anchor_camera_id", "unique_camera_id",
                "anchor_clone_timestamp", "update_fail_count", "should_marg",
            ),
            (
                {
                    "camera_timestamp": timestamp,
                    "source_frame_id": index,
                    "feature_id": feature_id,
                    "tracked_in_current_frame": 1,
                    "u_px": u_px,
                    "v_px": v_px,
                    "depth_m": 20.0,
                    "anchor_camera_id": 0,
                    "unique_camera_id": 0,
                    "anchor_clone_timestamp": 1.0,
                    "update_fail_count": 0,
                    "should_marg": 0,
                }
                for index, (timestamp, feature_id, u_px, v_px) in enumerate(
                    (
                        (1.0, 101, 300.0, 60.0),
                        (2.0, 101, 310.0, 65.0),
                        (3.0, 101, 320.0, 70.0),
                        (1.0, 102, 50.0, 300.0),
                    )
                )
            ),
        )
        (replay / "manifest.json").write_text(
            json.dumps(
                {
                    "calibration": {
                        "resolution": [640, 360],
                        "intrinsics": [320.0, 320.0, 320.0, 180.0],
                    }
                }
            ),
            encoding="utf-8",
        )
        return replay

    def test_flags_missing_accelerometer_axis_and_summarizes_updates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            replay = self.make_replay(Path(temp_dir))

            report = analyze_replay(replay)

            self.assertEqual(
                report["accelerometer"]["truth_velocity_frame_check"]["selected"],
                "body_frd",
            )
            self.assertEqual(report["accelerometer"]["axes"]["x"]["status"], "pass")
            self.assertEqual(
                report["accelerometer"]["axes"]["y"]["status"],
                "fail_missing_dynamic_signal",
            )
            self.assertEqual(report["accelerometer"]["axes"]["z"]["status"], "pass")
            self.assertEqual(report["visual_updates"]["status"], "pass")
            self.assertEqual(
                report["visual_updates"]["msckf_features_used"]["median"], 20.0
            )
            funnel = report["visual_updates"]["msckf_rejection_funnel"]
            self.assertTrue(funnel["available"])
            self.assertEqual(funnel["status"], "pass")
            self.assertEqual(
                funnel["totals"]["msckf_input_features"],
                funnel["totals"]["msckf_accepted_features"],
            )
            self.assertTrue(
                funnel["geometry_ranked_selection"]["available"]
            )
            self.assertEqual(
                funnel["geometry_ranked_selection"]["selected_features"],
                funnel["totals"]["msckf_accepted_features"],
            )
            geometry = report["visual_updates"]["feature_geometry"]
            self.assertTrue(geometry["available"])
            self.assertEqual(geometry["rows"], 4)
            self.assertEqual(
                geometry["result_counts"]["triangulation_rejected"], 1
            )
            self.assertEqual(
                geometry["groups"]["bad_condition_rejected"]["metrics"]
                ["condition_number"]["median"],
                20_000.0,
            )
            self.assertTrue(
                geometry["geometry_ranked_selection"]["available"]
            )
            self.assertEqual(
                geometry["geometry_ranked_selection"]["selected"], 1
            )
            self.assertEqual(
                geometry["bad_condition_threshold_analysis"]
                ["parallax_bins_deg"]["from_0_1_to_0_5"],
                1,
            )
            image_regions = geometry["image_region_analysis"]
            self.assertTrue(image_regions["available"])
            self.assertEqual(
                image_regions["grid_3x3"]["bottom_left"]["accepted"], 1
            )
            self.assertEqual(
                image_regions["grid_3x3"]["middle_center"]
                ["bad_condition_rejected"],
                1,
            )
            self.assertEqual(
                image_regions["vertical_track_flow"]["zones"]["birth"]
                ["bottom"]["accepted"],
                1,
            )
            slam_images = report["visual_updates"]["slam_feature_images"]
            self.assertTrue(slam_images["available"])
            self.assertEqual(slam_images["unique_features_in_state"], 2)
            self.assertEqual(
                slam_images["top_third_counterfactual_mask"]
                ["assigned_unique_features"],
                1,
            )
            self.assertEqual(
                slam_images["top_third_counterfactual_mask"]
                ["assigned_state_lifetime_s"]["median"],
                2.0,
            )
            self.assertFalse(report["truth_fed_to_estimator"])

    def test_flags_opposite_gyroscope_axis(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            replay = Path(temp_dir)
            timestamps = np.arange(0.0, 10.001, 0.01)
            pitch = 0.3 * np.sin(0.8 * timestamps)
            pitch_rate = 0.24 * np.cos(0.8 * timestamps)
            write_csv(
                replay / "truth.csv",
                ("timestamp", "qx", "qy", "qz", "qw"),
                (
                    {
                        "timestamp": timestamp,
                        "qx": 0.0,
                        "qy": math.sin(angle / 2.0),
                        "qz": 0.0,
                        "qw": math.cos(angle / 2.0),
                    }
                    for timestamp, angle in zip(timestamps, pitch)
                ),
            )
            write_csv(
                replay / "imu.csv",
                ("timestamp", "wx", "wy", "wz"),
                (
                    {
                        "timestamp": timestamp,
                        "wx": 0.0,
                        "wy": -rate,
                        "wz": 0.0,
                    }
                    for timestamp, rate in zip(timestamps, pitch_rate)
                ),
            )

            report = analyze_gyroscope(replay, 0.25)

            self.assertEqual(report["axes"]["x"]["status"], "pass")
            self.assertEqual(report["axes"]["y"]["status"], "fail_opposite_sign")
            self.assertLess(report["axes"]["y"]["correlation"], -0.99)
            self.assertEqual(report["axes"]["z"]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
