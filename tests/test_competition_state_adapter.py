import math
import unittest

import numpy as np

from autonomy_core.core.competition_state_adapter import (
    ATTITUDE,
    LOCAL_POSITION_NED,
    ODOMETRY,
    CompetitionStateAdapter,
    CompetitionStateAdapterError,
    ned_position_to_internal_z_up,
    ned_velocity_to_internal_z_up,
    vehicle_state_from_local_position_ned,
    vehicle_state_from_odometry,
    yaw_from_quaternion_wxyz,
)


class FakeMessage:
    def __init__(self, message_type, **fields):
        self._message_type = message_type
        for key, value in fields.items():
            setattr(self, key, value)

    def get_type(self):
        return self._message_type


class CompetitionStateAdapterTests(unittest.TestCase):
    def test_ned_position_and_velocity_convert_to_internal_z_up(self):
        np.testing.assert_allclose(
            ned_position_to_internal_z_up(1.0, 2.0, 3.0),
            np.array([1.0, 2.0, -3.0]),
        )
        np.testing.assert_allclose(
            ned_velocity_to_internal_z_up(4.0, 5.0, -6.0),
            np.array([4.0, 5.0, 6.0]),
        )

    def test_local_position_and_attitude_build_vehicle_state(self):
        local_position = FakeMessage(
            LOCAL_POSITION_NED,
            x=10.0,
            y=-2.0,
            z=3.5,
            vx=1.0,
            vy=2.0,
            vz=-0.5,
            time_boot_ms=123,
        )
        attitude = FakeMessage(
            ATTITUDE,
            roll=0.1,
            pitch=-0.2,
            yaw=1.25,
            rollspeed=0.0,
            pitchspeed=0.0,
            yawspeed=0.0,
            time_boot_ms=124,
        )

        state = vehicle_state_from_local_position_ned(local_position, attitude)

        np.testing.assert_allclose(state.pos, np.array([10.0, -2.0, -3.5]))
        np.testing.assert_allclose(state.vel, np.array([1.0, 2.0, 0.5]))
        self.assertEqual(state.yaw, 1.25)

    def test_odometry_builds_vehicle_state_from_quaternion(self):
        yaw = 0.75
        odometry = FakeMessage(
            ODOMETRY,
            x=1.0,
            y=2.0,
            z=-3.0,
            vx=4.0,
            vy=5.0,
            vz=6.0,
            q=[math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
            rollspeed=0.0,
            pitchspeed=0.0,
            yawspeed=0.0,
            time_usec=1000,
            reset_counter=0,
        )

        state = vehicle_state_from_odometry(odometry)

        np.testing.assert_allclose(state.pos, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(state.vel, np.array([4.0, 5.0, -6.0]))
        self.assertAlmostEqual(state.yaw, yaw)

    def test_yaw_from_quaternion_wxyz(self):
        yaw = -1.2
        q = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]

        self.assertAlmostEqual(yaw_from_quaternion_wxyz(q), yaw)

    def test_nonfinite_position_rejected(self):
        with self.assertRaises(CompetitionStateAdapterError):
            ned_position_to_internal_z_up(1.0, math.nan, 2.0)

    def test_adapter_reports_missing_position_without_fabricating_state(self):
        adapter = CompetitionStateAdapter(clock=lambda: 10.0)
        adapter.ingest_message(
            FakeMessage(ATTITUDE, yaw=0.5, roll=0.0, pitch=0.0),
            received_wall_time=10.0,
        )

        result = adapter.latest_result(now=10.0)

        self.assertFalse(result.is_usable)
        self.assertIsNone(result.vehicle_state)
        self.assertIn("missing_local_position_ned", result.missing_reasons)

    def test_adapter_reports_stale_telemetry(self):
        adapter = CompetitionStateAdapter(clock=lambda: 11.0)
        adapter.ingest_message(
            FakeMessage(
                LOCAL_POSITION_NED,
                x=1.0,
                y=2.0,
                z=3.0,
                vx=0.0,
                vy=0.0,
                vz=0.0,
            ),
            received_wall_time=9.0,
        )
        adapter.ingest_message(
            FakeMessage(ATTITUDE, yaw=0.5, roll=0.0, pitch=0.0),
            received_wall_time=9.0,
        )

        result = adapter.latest_result(now=11.0, max_age_s=0.5)

        self.assertFalse(result.is_usable)
        self.assertIsNone(result.vehicle_state)
        self.assertIn("stale_local_position_ned", result.missing_reasons)
        self.assertIn("stale_attitude", result.missing_reasons)

    def test_adapter_prefers_fresh_odometry_when_available(self):
        adapter = CompetitionStateAdapter(clock=lambda: 20.0)
        yaw = 0.25
        adapter.ingest_message(
            FakeMessage(
                ODOMETRY,
                x=1.0,
                y=2.0,
                z=3.0,
                vx=4.0,
                vy=5.0,
                vz=6.0,
                q=[math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
            ),
            received_wall_time=19.8,
        )

        result = adapter.latest_result(now=20.0, max_age_s=0.5)

        self.assertTrue(result.is_usable)
        self.assertEqual(result.position_source, ODOMETRY)
        self.assertEqual(result.attitude_source, ODOMETRY)
        np.testing.assert_allclose(result.vehicle_state.pos, np.array([1.0, 2.0, -3.0]))
        np.testing.assert_allclose(result.vehicle_state.vel, np.array([4.0, 5.0, -6.0]))
        self.assertAlmostEqual(result.vehicle_state.yaw, yaw)

    def test_adapter_uses_local_position_and_attitude_when_odometry_absent(self):
        adapter = CompetitionStateAdapter(clock=lambda: 30.0)
        adapter.ingest_message(
            FakeMessage(
                LOCAL_POSITION_NED,
                x=1.0,
                y=2.0,
                z=3.0,
                vx=4.0,
                vy=5.0,
                vz=6.0,
            ),
            received_wall_time=29.9,
        )
        adapter.ingest_message(
            FakeMessage(ATTITUDE, yaw=-0.5, roll=0.0, pitch=0.0),
            received_wall_time=29.8,
        )

        result = adapter.latest_result(now=30.0, max_age_s=0.5)

        self.assertTrue(result.is_usable)
        self.assertEqual(result.position_source, LOCAL_POSITION_NED)
        self.assertEqual(result.attitude_source, ATTITUDE)
        np.testing.assert_allclose(result.vehicle_state.pos, np.array([1.0, 2.0, -3.0]))
        np.testing.assert_allclose(result.vehicle_state.vel, np.array([4.0, 5.0, -6.0]))
        self.assertAlmostEqual(result.vehicle_state.yaw, -0.5)


if __name__ == "__main__":
    unittest.main()
