import subprocess
import sys
import unittest

from autonomy_core.tools.competition_mavlink_observe import (
    MavlinkTelemetryInventory,
    run_receive_only_observe,
)


class FakeMessage:
    def __init__(
        self,
        message_type,
        message_id,
        *,
        system_id=1,
        component_id=1,
        **fields,
    ):
        self._message_type = message_type
        self._message_id = message_id
        self._system_id = system_id
        self._component_id = component_id
        self._fields = fields

    def get_type(self):
        return self._message_type

    def get_msgId(self):
        return self._message_id

    def get_srcSystem(self):
        return self._system_id

    def get_srcComponent(self):
        return self._component_id

    def to_dict(self):
        return {"mavpackettype": self._message_type, **self._fields}


class FakeConnection:
    def __init__(self, messages):
        self.messages = list(messages)
        self.recv_calls = 0

    def recv_match(self, *, blocking=False):
        self.recv_calls += 1
        self.last_blocking = blocking
        if self.messages:
            return self.messages.pop(0)
        return None


class FakeClock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        current = self.value
        self.value += 0.1
        return current


class MavlinkObserveTests(unittest.TestCase):
    def test_import_does_not_import_pymavlink(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import sys; "
                    "import autonomy_core.tools.competition_mavlink_observe; "
                    "print('pymavlink' in sys.modules)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.stdout.strip(), "False")

    def test_inventory_records_message_fields_ids_sources_and_rates(self):
        inventory = MavlinkTelemetryInventory(sample_limit_per_type=2)

        inventory.observe_message(
            FakeMessage(
                "LOCAL_POSITION_NED",
                32,
                system_id=42,
                component_id=7,
                x=1.0,
                y=2.0,
                z=3.0,
                vx=4.0,
                vy=5.0,
                vz=6.0,
                time_boot_ms=123,
            ),
            received_wall_time=10.0,
        )
        inventory.observe_message(
            FakeMessage(
                "LOCAL_POSITION_NED",
                32,
                system_id=42,
                component_id=7,
                x=2.0,
                y=3.0,
                z=4.0,
                vx=5.0,
                vy=6.0,
                vz=7.0,
                time_boot_ms=223,
            ),
            received_wall_time=10.5,
        )

        summary = inventory.summary()
        local_position = summary["message_types"]["LOCAL_POSITION_NED"]

        self.assertTrue(summary["receive_only"])
        self.assertTrue(summary["local_position_ned_available"])
        self.assertFalse(summary["odometry_available"])
        self.assertEqual(local_position["message_id"], 32)
        self.assertEqual(local_position["count"], 2)
        self.assertAlmostEqual(local_position["observed_rate_hz"], 2.0)
        self.assertEqual(local_position["system_ids"], [42])
        self.assertEqual(local_position["component_ids"], [7])
        self.assertIn("x", local_position["field_names"])
        self.assertIn("time_boot_ms", local_position["timestamp_fields"])
        self.assertEqual(len(local_position["samples"]), 2)

    def test_run_receive_only_observe_uses_nonblocking_recv_and_sends_nothing(self):
        messages = [
            FakeMessage("HEARTBEAT", 0, base_mode=0),
            FakeMessage("ATTITUDE", 30, yaw=0.25, time_boot_ms=100),
            FakeMessage("ODOMETRY", 331, x=1.0, y=2.0, z=3.0, time_usec=1000),
        ]
        connection = FakeConnection(messages)
        clock = FakeClock()

        summary = run_receive_only_observe(
            duration_s=0.8,
            connection=connection,
            clock=clock,
            sleep=lambda _seconds: None,
            poll_sleep_s=0.0,
        )

        self.assertTrue(summary["receive_only"])
        self.assertIn("HEARTBEAT", summary["message_types"])
        self.assertTrue(summary["odometry_available"])
        self.assertTrue(summary["attitude_available"])
        self.assertFalse(connection.last_blocking)

    def test_bad_data_is_ignored(self):
        inventory = MavlinkTelemetryInventory()

        inventory.observe_message(FakeMessage("BAD_DATA", -1), received_wall_time=1.0)

        self.assertEqual(inventory.summary()["message_types"], {})


if __name__ == "__main__":
    unittest.main()
