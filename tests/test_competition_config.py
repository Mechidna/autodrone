import importlib
import math
import sys
import unittest


class CompetitionConfigTests(unittest.TestCase):
    def test_import_has_no_runtime_side_effects(self):
        sys.modules.pop("autonomy_core.core.competition_config", None)
        sys.modules.pop("autonomy_core.launch.autonomy_api6", None)

        module = importlib.import_module("autonomy_core.core.competition_config")

        self.assertTrue(hasattr(module, "RuntimeCompetitionConfig"))
        self.assertNotIn("autonomy_core.launch.autonomy_api6", sys.modules)

    def test_official_camera_matrix_matches_vadr_ts_002(self):
        from autonomy_core.core.competition_config import RuntimeCompetitionConfig

        config = RuntimeCompetitionConfig()

        self.assertEqual(
            config.camera_matrix,
            (
                (320.0, 0.0, 320.0),
                (0.0, 320.0, 180.0),
                (0.0, 0.0, 1.0),
            ),
        )

    def test_official_distortion_is_all_zero(self):
        from autonomy_core.core.competition_config import RuntimeCompetitionConfig

        config = RuntimeCompetitionConfig()

        self.assertGreater(len(config.dist_coeffs), 0)
        self.assertTrue(all(value == 0.0 for value in config.dist_coeffs))

    def test_official_rates_and_periods_match_vadr_ts_002(self):
        from autonomy_core.core.competition_config import RuntimeCompetitionConfig

        config = RuntimeCompetitionConfig()

        self.assertEqual(config.physics_rate_hz, 120.0)
        self.assertEqual(config.vision_rate_hz, 30.0)
        self.assertEqual(config.heartbeat_min_hz, 2.0)
        self.assertEqual(config.command_rate_upper_bound_exclusive_hz, 100.0)
        self.assertTrue(config.command_rate_is_allowed(99.999))
        self.assertFalse(config.command_rate_is_allowed(100.0))
        self.assertAlmostEqual(config.physics_period_s, 1.0 / 120.0)
        self.assertAlmostEqual(config.vision_period_s, 1.0 / 30.0)
        self.assertAlmostEqual(config.heartbeat_period_max_s, 0.5)
        self.assertAlmostEqual(config.command_period_lower_bound_exclusive_s, 0.01)

    def test_official_camera_tilt_is_degrees_with_radian_helper(self):
        from autonomy_core.core.competition_config import RuntimeCompetitionConfig

        config = RuntimeCompetitionConfig()

        self.assertEqual(config.camera_tilt_up_deg, 20.0)
        self.assertAlmostEqual(config.camera_tilt_up_rad, math.radians(20.0))

    def test_official_geometry_and_race_duration(self):
        from autonomy_core.core.competition_config import RuntimeCompetitionConfig

        config = RuntimeCompetitionConfig()

        self.assertEqual(config.gate_outer_square_mm, 2700)
        self.assertEqual(config.gate_inner_square_mm, 1500)
        self.assertEqual(config.gate_depth_mm, 260)
        self.assertEqual(config.drone_chassis_length_mm, 280)
        self.assertEqual(config.drone_chassis_width_mm, 280)
        self.assertEqual(config.drone_chassis_height_mm, 160)
        self.assertEqual(config.race_max_duration_s, 480)

    def test_vision_header_constants_are_explicit(self):
        from autonomy_core.core.competition_config import RuntimeCompetitionConfig

        config = RuntimeCompetitionConfig()

        self.assertEqual(config.vision_udp_port, 5600)
        self.assertEqual(config.vision_header_format, "<IHHIIQ")
        self.assertEqual(config.vision_header_size_bytes, 24)
        self.assertEqual(
            config.vision_header_fields,
            (
                "frame_id:uint32",
                "chunk_id:uint16",
                "total_chunks:uint16",
                "jpeg_size:uint32",
                "payload_size:uint32",
                "sim_time_ns:uint64",
            ),
        )

    def test_reference_defaults_are_separate_from_official_constants(self):
        from autonomy_core.core.competition_config import (
            PyAIPilotExampleReferenceConfig,
            RuntimeCompetitionConfig,
        )

        official = RuntimeCompetitionConfig()
        reference = PyAIPilotExampleReferenceConfig()

        self.assertEqual(reference.mavlink_default_ip, "127.0.0.1")
        self.assertEqual(reference.mavlink_default_udp_port, 14550)
        self.assertEqual(reference.mavlink_connection_scheme, "udpin")
        self.assertEqual(reference.vision_bind_ip, "0.0.0.0")
        self.assertEqual(reference.vision_udp_port, official.vision_udp_port)
        self.assertEqual(reference.vision_header_format, official.vision_header_format)
        self.assertEqual(reference.vision_udp_recv_bytes, 65536)
        self.assertEqual(reference.timesync_request_hz, 10.0)
        self.assertEqual(reference.controller_update_hz, 250.0)
        self.assertGreater(
            reference.controller_update_hz,
            official.command_rate_upper_bound_exclusive_hz,
        )
        self.assertEqual(reference.default_controller_path, "set_actuator_control_target")


if __name__ == "__main__":
    unittest.main()
