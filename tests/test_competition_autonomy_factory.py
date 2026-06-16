import subprocess
import sys
import unittest

from autonomy_core.runtime.competition_autonomy_factory import (
    COMPETITION_AUTONOMY_PROFILE_NAME,
    COMPETITION_OFFICIAL_TRANSFORM_MODE,
    COMPETITION_SAFE_POSE_SOURCE,
    CompetitionAutonomyProfile,
    CompetitionAutonomyProfileError,
    apply_competition_autonomy_profile,
    create_competition_autonomy_api,
    validate_competition_autonomy_api,
)
from autonomy_core.runtime.competition_guard import GAZEBO_TRUTH_POSE_SOURCE


class FakeAutonomy:
    def __init__(self, **kwargs):
        self.constructor_kwargs = dict(kwargs)
        self.perception_world_pose_source = GAZEBO_TRUTH_POSE_SOURCE
        self.perception_world_pose_source_used = GAZEBO_TRUTH_POSE_SOURCE
        self.perception_transform_mode = "physical_direct_rad_x_mirror"
        self.save_perception_debug_frames = True
        self.use_diagnostic_far_depth_correction = True
        self.image_gazebo_pose_snapshot = {"pose": "truth"}
        self.print_perception_transform_startup_calls = 0

    def print_perception_transform_startup(self):
        self.print_perception_transform_startup_calls += 1


class CompetitionAutonomyFactoryTests(unittest.TestCase):
    def test_import_does_not_load_real_autonomy_or_live_dependencies(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import sys; "
                    "import autonomy_core.runtime.competition_autonomy_factory; "
                    "print('autonomy_core.launch.autonomy_api6' in sys.modules, "
                    "'cv2' in sys.modules, 'pymavlink' in sys.modules, "
                    "'mavsdk' in sys.modules, 'rclpy' in sys.modules)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.stdout.strip(), "False False False False False")

    def test_create_competition_autonomy_api_uses_fake_factory_lazily(self):
        calls = []

        def factory(**kwargs):
            calls.append(dict(kwargs))
            return FakeAutonomy(**kwargs)

        autonomy = create_competition_autonomy_api(autonomy_factory=factory)

        self.assertEqual(
            calls,
            [
                {
                    "use_perception": False,
                    "race_gate_count": None,
                    "race_gate_order": None,
                    "save_perception_debug_frames": False,
                    "use_lookahead_gate_filter": True,
                }
            ],
        )
        self.assertEqual(autonomy.perception_world_pose_source, COMPETITION_SAFE_POSE_SOURCE)
        self.assertEqual(
            autonomy.perception_world_pose_source_used,
            COMPETITION_SAFE_POSE_SOURCE,
        )
        self.assertEqual(
            autonomy.perception_transform_mode,
            COMPETITION_OFFICIAL_TRANSFORM_MODE,
        )
        self.assertFalse(autonomy.save_perception_debug_frames)
        self.assertFalse(autonomy.use_diagnostic_far_depth_correction)
        self.assertIsNone(autonomy.image_gazebo_pose_snapshot)
        self.assertTrue(autonomy.competition_autonomy_profile_active)
        self.assertEqual(
            autonomy.competition_autonomy_profile_name,
            COMPETITION_AUTONOMY_PROFILE_NAME,
        )
        self.assertEqual(autonomy.competition_yolo_config_source, "not_loaded_in_profile")
        self.assertEqual(autonomy.print_perception_transform_startup_calls, 0)

    def test_rejects_gazebo_truth_pose_source(self):
        profile = CompetitionAutonomyProfile(
            perception_world_pose_source=GAZEBO_TRUTH_POSE_SOURCE,
        )

        with self.assertRaisesRegex(CompetitionAutonomyProfileError, "gazebo_truth"):
            create_competition_autonomy_api(
                profile=profile,
                autonomy_factory=lambda **kwargs: FakeAutonomy(**kwargs),
            )

    def test_rejects_debug_frame_writes(self):
        profile = CompetitionAutonomyProfile(save_perception_debug_frames=True)

        with self.assertRaisesRegex(
            CompetitionAutonomyProfileError,
            "save_perception_debug_frames=False",
        ):
            profile.constructor_kwargs()

    def test_rejects_diagnostic_far_depth_correction(self):
        profile = CompetitionAutonomyProfile(
            use_diagnostic_far_depth_correction=True,
        )

        with self.assertRaisesRegex(
            CompetitionAutonomyProfileError,
            "use_diagnostic_far_depth_correction=False",
        ):
            profile.constructor_kwargs()

    def test_rejects_real_perception_without_explicit_yolo_acknowledgment(self):
        profile = CompetitionAutonomyProfile(use_perception=True)

        with self.assertRaisesRegex(
            CompetitionAutonomyProfileError,
            "explicit YOLO configuration",
        ):
            profile.constructor_kwargs()

    def test_allows_real_perception_only_with_explicit_legacy_yolo_acknowledgment(self):
        profile = CompetitionAutonomyProfile(
            use_perception=True,
            allow_legacy_yolo_default=True,
        )
        autonomy = create_competition_autonomy_api(
            profile=profile,
            autonomy_factory=lambda **kwargs: FakeAutonomy(**kwargs),
        )

        self.assertTrue(autonomy.constructor_kwargs["use_perception"])
        self.assertEqual(
            autonomy.perception_transform_mode,
            COMPETITION_OFFICIAL_TRANSFORM_MODE,
        )
        self.assertEqual(autonomy.print_perception_transform_startup_calls, 1)
        self.assertEqual(
            autonomy.competition_yolo_config_source,
            "legacy_autonomyapi_default_explicitly_acknowledged",
        )

    def test_rejects_non_official_competition_transform_mode(self):
        profile = CompetitionAutonomyProfile(
            perception_transform_mode="physical_direct_rad_x_mirror",
        )

        with self.assertRaisesRegex(CompetitionAutonomyProfileError, "perception_transform_mode"):
            create_competition_autonomy_api(
                profile=profile,
                autonomy_factory=lambda **kwargs: FakeAutonomy(**kwargs),
            )

    def test_rejects_extra_kwargs_that_override_profile_owned_fields(self):
        profile = CompetitionAutonomyProfile(
            extra_autonomy_kwargs={"use_perception": False},
        )

        with self.assertRaisesRegex(CompetitionAutonomyProfileError, "owns"):
            profile.constructor_kwargs()

    def test_rejects_extra_kwargs_with_gazebo_truth_content(self):
        profile = CompetitionAutonomyProfile(
            extra_autonomy_kwargs={"gazebo_pose": {"pose": "truth"}},
        )

        with self.assertRaisesRegex(CompetitionAutonomyProfileError, "forbidden"):
            profile.constructor_kwargs()

    def test_validate_detects_profile_drift_after_application(self):
        autonomy = FakeAutonomy()
        apply_competition_autonomy_profile(autonomy)
        autonomy.perception_world_pose_source = GAZEBO_TRUTH_POSE_SOURCE

        with self.assertRaisesRegex(CompetitionAutonomyProfileError, "gazebo_truth"):
            validate_competition_autonomy_api(autonomy)

    def test_validate_detects_transform_mode_drift_after_application(self):
        autonomy = FakeAutonomy()
        apply_competition_autonomy_profile(autonomy)
        autonomy.perception_transform_mode = "physical_direct_rad_x_mirror"

        with self.assertRaisesRegex(CompetitionAutonomyProfileError, "perception_transform_mode"):
            validate_competition_autonomy_api(autonomy)


if __name__ == "__main__":
    unittest.main()
