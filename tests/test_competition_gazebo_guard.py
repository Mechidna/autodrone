import unittest

from autonomy_core.runtime.competition_guard import (
    CompetitionGuard,
    CompetitionGuardError,
    GAZEBO_TRUTH_POSE_SOURCE,
)


class CompetitionGuardTests(unittest.TestCase):
    def test_rejects_gazebo_truth_pose_source(self):
        guard = CompetitionGuard()

        with self.assertRaises(CompetitionGuardError):
            guard.assert_perception_world_pose_source(GAZEBO_TRUTH_POSE_SOURCE)

    def test_allows_non_gazebo_pose_source(self):
        guard = CompetitionGuard()

        guard.assert_perception_world_pose_source("mavsdk")

    def test_rejects_non_none_gazebo_pose_before_perception(self):
        guard = CompetitionGuard()

        with self.assertRaises(CompetitionGuardError):
            guard.assert_no_gazebo_pose({"gazebo_model_pos_world": [1.0, 2.0, 3.0]})

    def test_forces_gazebo_pose_none_in_perception_kwargs(self):
        guard = CompetitionGuard()

        safe_kwargs = guard.perception_update_kwargs(frame="frame", gazebo_pose=None)

        self.assertIsNone(safe_kwargs["gazebo_pose"])
        self.assertEqual(safe_kwargs["frame"], "frame")

    def test_rejects_gazebo_truth_in_image_metadata(self):
        guard = CompetitionGuard()
        metadata = {
            "stamp": 123,
            "gazebo_model_pos_world": [1.0, 2.0, 3.0],
        }

        with self.assertRaises(CompetitionGuardError):
            guard.assert_competition_safe(image_metadata=metadata)

    def test_command_enabled_mode_rejects_gazebo_truth_inputs(self):
        guard = CompetitionGuard()

        with self.assertRaises(CompetitionGuardError):
            guard.assert_competition_safe(
                perception_world_pose_source=GAZEBO_TRUTH_POSE_SOURCE,
                command_enabled=True,
            )

    def test_command_enabled_mode_rejects_runner_gazebo_pose(self):
        guard = CompetitionGuard()

        with self.assertRaises(CompetitionGuardError):
            guard.assert_competition_safe(
                perception_world_pose_source="mavsdk",
                mode="command_live",
                runner_inputs={"latest_gazebo_pose": {"pose": "truth"}},
            )

    def test_guard_is_passive_when_not_in_competition_mode(self):
        guard = CompetitionGuard(competition_mode=False)

        guard.assert_competition_safe(
            perception_world_pose_source=GAZEBO_TRUTH_POSE_SOURCE,
            gazebo_pose={"gazebo_model_pos_world": [1.0, 2.0, 3.0]},
            command_enabled=True,
        )


if __name__ == "__main__":
    unittest.main()
