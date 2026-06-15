import subprocess
import sys
import unittest

from autonomy_core.runtime.competition_mavlink_transport import (
    CompetitionMavlinkTransport,
    CompetitionMavlinkTransportConfig,
    DEFAULT_MAVLINK_ENDPOINT,
)
from autonomy_core.runtime.competition_runner import (
    CompetitionRunner,
    CompetitionRunnerConfig,
    CompetitionRunnerMode,
)


class FakeMessage:
    def __init__(
        self,
        message_type,
        message_id=0,
        *,
        system_id=1,
        component_id=1,
        **fields,
    ):
        self._message_type = message_type
        self._message_id = message_id
        self._system_id = system_id
        self._component_id = component_id
        self._fields = dict(fields)
        for key, value in fields.items():
            setattr(self, key, value)

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
        self.close_calls = 0
        self.send_calls = 0
        self.wait_heartbeat_calls = 0
        self.blocking_values = []

    def recv_match(self, *, blocking=False):
        self.recv_calls += 1
        self.blocking_values.append(blocking)
        if self.messages:
            return self.messages.pop(0)
        return None

    def close(self):
        self.close_calls += 1

    def send(self, *_args, **_kwargs):
        self.send_calls += 1
        raise AssertionError("MAVLink receive transport must not send")

    def wait_heartbeat(self):
        self.wait_heartbeat_calls += 1
        raise AssertionError("MAVLink receive transport must not wait_heartbeat")


class FakeClock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        current = self.value
        self.value += 0.1
        return current


class CompetitionMavlinkTransportTests(unittest.TestCase):
    def test_import_does_not_import_pymavlink(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import sys; "
                    "import autonomy_core.runtime.competition_mavlink_transport; "
                    "print('pymavlink' in sys.modules)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.stdout.strip(), "False")

    def test_receive_messages_is_nonblocking_receive_only_and_records_summary(self):
        connection = FakeConnection(
            [
                FakeMessage("HEARTBEAT", 0, base_mode=0),
                FakeMessage("BAD_DATA", -1),
                FakeMessage("ODOMETRY", 331, x=1.0, y=2.0, z=3.0, time_usec=123),
            ]
        )
        transport = CompetitionMavlinkTransport(
            connection=connection,
            clock=FakeClock(),
            sleep=lambda _seconds: None,
        )

        messages = transport.receive_messages(max_messages=2)
        summary = transport.summary()

        self.assertEqual([message.get_type() for message in messages], ["HEARTBEAT", "ODOMETRY"])
        self.assertTrue(all(value is False for value in connection.blocking_values))
        self.assertEqual(connection.send_calls, 0)
        self.assertEqual(connection.wait_heartbeat_calls, 0)
        self.assertEqual(summary["endpoint"], DEFAULT_MAVLINK_ENDPOINT)
        self.assertTrue(summary["receive_only"])
        self.assertTrue(summary["odometry_available"])
        self.assertEqual(summary["stats"]["messages_received"], 2)
        self.assertEqual(summary["stats"]["bad_data_ignored"], 1)
        self.assertEqual(summary["stats"]["heartbeat_count"], 1)
        self.assertEqual(summary["stats"]["message_type_counts"]["ODOMETRY"], 1)

    def test_transport_can_start_from_injected_factory(self):
        created = []

        def factory(endpoint):
            created.append(endpoint)
            return FakeConnection([])

        transport = CompetitionMavlinkTransport(connection_factory=factory)
        transport.start()

        self.assertEqual(created, [DEFAULT_MAVLINK_ENDPOINT])
        self.assertTrue(transport.is_started)

    def test_close_closes_owned_connection(self):
        connection = FakeConnection([])
        transport = CompetitionMavlinkTransport(
            connection_factory=lambda _endpoint: connection,
        )

        transport.start()
        transport.close()

        self.assertEqual(connection.close_calls, 1)
        self.assertFalse(transport.is_started)

    def test_transport_feeds_competition_runner_without_live_socket(self):
        transport = CompetitionMavlinkTransport(
            connection=FakeConnection(
                [
                    FakeMessage("HEARTBEAT", 0, base_mode=0),
                    FakeMessage(
                        "LOCAL_POSITION_NED",
                        32,
                        x=1.0,
                        y=2.0,
                        z=-3.0,
                        vx=0.1,
                        vy=0.2,
                        vz=-0.3,
                        time_boot_ms=1000,
                    ),
                    FakeMessage(
                        "ATTITUDE",
                        30,
                        roll=0.0,
                        pitch=0.0,
                        yaw=0.25,
                        rollspeed=0.0,
                        pitchspeed=0.0,
                        yawspeed=0.0,
                        time_boot_ms=1000,
                    ),
                ]
            )
        )
        runner = CompetitionRunner(
            config=CompetitionRunnerConfig(mode=CompetitionRunnerMode.OBSERVE),
            mavlink_transport=transport,
            clock=lambda: 100.0,
        )

        result = runner.step()

        self.assertEqual(result.telemetry_messages_processed, 3)
        self.assertTrue(result.heartbeat_seen)
        self.assertTrue(result.state_result.is_usable)
        self.assertFalse(result.command_publication_allowed)


if __name__ == "__main__":
    unittest.main()
