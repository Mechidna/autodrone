import subprocess
import sys
import unittest

from autonomy_core.perception.competition_image_adapter import parse_vision_packet_header
from autonomy_core.runtime.competition_guard import CompetitionGuardError
from autonomy_core.runtime.competition_runner import (
    CompetitionRunnerMode,
    CompetitionRunnerSafetyError,
)
from autonomy_core.runtime.px4_gazebo_surrogate_harness import (
    SURROGATE_LABEL,
    Px4EstimatedTelemetrySample,
    Px4GazeboSurrogateHarness,
    Px4GazeboSurrogateHarnessError,
    Px4GazeboSurrogateScenario,
    Px4GazeboSurrogateScenarioStep,
    command_result_was_no_send,
)


def sample(index, *, time_boot_ms=None, metadata=None):
    value = float(index)
    return Px4EstimatedTelemetrySample(
        position_ned_m=(value, value + 1.0, -(value + 2.0)),
        velocity_ned_m_s=(0.1 * value, 0.2 * value, -0.3 * value),
        yaw_rad=0.1 * value,
        time_boot_ms=(1000 + 100 * index if time_boot_ms is None else time_boot_ms),
        metadata=metadata,
    )


class Px4GazeboSurrogateScenarioTests(unittest.TestCase):
    def test_import_does_not_load_live_transport_or_image_dependencies(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import sys; "
                    "import autonomy_core.runtime.px4_gazebo_surrogate_harness; "
                    "print('pymavlink' in sys.modules, 'cv2' in sys.modules, "
                    "'mavsdk' in sys.modules, 'rclpy' in sys.modules)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.stdout.strip(), "False False False False")

    def test_multi_step_observe_scenario_converts_estimated_telemetry(self):
        harness = Px4GazeboSurrogateHarness()
        scenario = Px4GazeboSurrogateScenario(
            name="observe_three_samples",
            mode=CompetitionRunnerMode.OBSERVE,
            steps=(
                Px4GazeboSurrogateScenarioStep(10.0, telemetry_samples=(sample(1),)),
                Px4GazeboSurrogateScenarioStep(10.1, telemetry_samples=(sample(2),)),
                Px4GazeboSurrogateScenarioStep(10.2, telemetry_samples=(sample(3),)),
            ),
        )

        result = harness.run_scenario(scenario)

        self.assertEqual(result.mode, CompetitionRunnerMode.OBSERVE)
        self.assertEqual(len(result.step_results), 3)
        self.assertEqual(result.telemetry_message_count, 9)
        self.assertEqual(result.summary["telemetry_sample_count"], 3)
        self.assertTrue(all(step.state_result.is_usable for step in result.step_results))
        self.assertFalse(result.phase4b_satisfied)
        self.assertFalse(result.competition_readiness_claimed)

    def test_multi_frame_vision_scenario_has_deterministic_packet_counts(self):
        harness = Px4GazeboSurrogateHarness()
        packets = harness.packetize_jpeg_frames(
            [b"frame-a", b"frame-b"],
            starting_frame_id=7,
            starting_sim_time_ns=1_000,
        )
        headers = [parse_vision_packet_header(packet) for packet in packets]

        self.assertEqual([header.frame_id for header in headers], [7, 8])
        self.assertEqual([header.chunk_id for header in headers], [0, 0])
        self.assertEqual([header.total_chunks for header in headers], [1, 1])
        self.assertEqual([header.sim_time_ns for header in headers], [1_000, 33_334_333])

        harness = Px4GazeboSurrogateHarness()
        scenario = Px4GazeboSurrogateScenario(
            name="vision_three_frames",
            mode=CompetitionRunnerMode.VISION_DRY_RUN,
            steps=(
                Px4GazeboSurrogateScenarioStep(
                    20.0,
                    jpeg_frames=(b"frame-1", b"frame-2"),
                    sim_time_ns=2_000,
                ),
                Px4GazeboSurrogateScenarioStep(
                    20.2,
                    jpeg_frames=(b"frame-3",),
                    sim_time_ns=70_000_000,
                ),
            ),
        )

        result = harness.run_scenario(scenario)

        self.assertEqual(result.mode, CompetitionRunnerMode.VISION_DRY_RUN)
        self.assertEqual(result.frame_count, 3)
        self.assertEqual(result.vision_packet_count, 3)
        self.assertEqual(result.summary["completed_packetized_frames"], 3)
        self.assertEqual(sum(step.vision_frames_completed for step in result.step_results), 3)
        self.assertFalse(result.phase4b_satisfied)

    def test_command_dry_run_scenario_remains_no_send(self):
        harness = Px4GazeboSurrogateHarness()
        scenario = Px4GazeboSurrogateScenario(
            name="command_no_send",
            mode=CompetitionRunnerMode.COMMAND_DRY_RUN,
            command_tuple=(0.1, -0.2, 0.3, 0.55),
            steps=(
                Px4GazeboSurrogateScenarioStep(30.0, telemetry_samples=(sample(1),)),
            ),
        )

        result = harness.run_scenario(scenario)
        command_result = result.step_results[0].command_result

        self.assertEqual(result.command_candidate_count, 1)
        self.assertTrue(command_result.accepted)
        self.assertTrue(command_result_was_no_send(command_result))
        self.assertFalse(result.step_results[0].command_publication_allowed)
        self.assertIn("phase6a_no_command_publication", result.summary["command_blocked_reasons"])
        self.assertIn("phase4b_telemetry_evidence_missing", result.summary["command_blocked_reasons"])
        self.assertIn("command_dry_run_no_send", result.summary["command_blocked_reasons"])

    def test_mixed_telemetry_and_image_scenario_summary_counts_are_deterministic(self):
        harness = Px4GazeboSurrogateHarness()
        scenario = Px4GazeboSurrogateScenario(
            name="mixed_command_dry_run",
            mode=CompetitionRunnerMode.COMMAND_DRY_RUN,
            steps=(
                Px4GazeboSurrogateScenarioStep(
                    40.0,
                    telemetry_samples=(sample(1),),
                    jpeg_frames=(b"frame-1", b"frame-2"),
                    sim_time_ns=4_000,
                ),
            ),
        )

        result = harness.run_scenario(scenario)

        self.assertEqual(result.telemetry_message_count, 3)
        self.assertEqual(result.vision_packet_count, 2)
        self.assertEqual(result.frame_count, 2)
        self.assertEqual(result.command_candidate_count, 1)
        self.assertEqual(result.summary["surrogate_label"], SURROGATE_LABEL)
        self.assertFalse(result.summary["phase4b_satisfied"])
        self.assertFalse(result.summary["competition_readiness_claimed"])

    def test_non_monotonic_timestamps_are_rejected(self):
        harness = Px4GazeboSurrogateHarness()
        wall_time_scenario = Px4GazeboSurrogateScenario(
            name="non_monotonic_wall_time",
            mode=CompetitionRunnerMode.OBSERVE,
            steps=(
                Px4GazeboSurrogateScenarioStep(50.0, telemetry_samples=(sample(1),)),
                Px4GazeboSurrogateScenarioStep(49.9, telemetry_samples=(sample(2),)),
            ),
        )
        telemetry_time_scenario = Px4GazeboSurrogateScenario(
            name="non_monotonic_boot_time",
            mode=CompetitionRunnerMode.OBSERVE,
            steps=(
                Px4GazeboSurrogateScenarioStep(
                    60.0,
                    telemetry_samples=(sample(1, time_boot_ms=2000),),
                ),
                Px4GazeboSurrogateScenarioStep(
                    60.1,
                    telemetry_samples=(sample(2, time_boot_ms=1999),),
                ),
            ),
        )

        with self.assertRaisesRegex(Px4GazeboSurrogateHarnessError, "wall_time_s"):
            harness.run_scenario(wall_time_scenario)
        with self.assertRaisesRegex(Px4GazeboSurrogateHarnessError, "time_boot_ms"):
            harness.run_scenario(telemetry_time_scenario)

    def test_gazebo_truth_metadata_is_rejected_before_runner_use(self):
        harness = Px4GazeboSurrogateHarness()
        scenario = Px4GazeboSurrogateScenario(
            name="gazebo_truth_rejected",
            mode=CompetitionRunnerMode.OBSERVE,
            steps=(
                Px4GazeboSurrogateScenarioStep(
                    70.0,
                    telemetry_samples=(
                        sample(1, metadata={"gazebo_model_pose": [1.0, 2.0, 3.0]}),
                    ),
                ),
            ),
        )

        with self.assertRaises(CompetitionGuardError):
            harness.run_scenario(scenario)

        self.assertEqual(harness.summary.guard_rejection_count, 1)

    def test_command_live_and_race_remain_fail_closed_for_scenarios(self):
        for mode in (CompetitionRunnerMode.COMMAND_LIVE, CompetitionRunnerMode.RACE):
            with self.subTest(mode=mode):
                harness = Px4GazeboSurrogateHarness()
                scenario = Px4GazeboSurrogateScenario(
                    name=f"{mode.value}_blocked",
                    mode=mode,
                    steps=(
                        Px4GazeboSurrogateScenarioStep(80.0, telemetry_samples=(sample(1),)),
                    ),
                )

                with self.assertRaises(CompetitionRunnerSafetyError):
                    harness.run_scenario(scenario)


if __name__ == "__main__":
    unittest.main()
