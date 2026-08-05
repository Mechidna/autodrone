import math
import socket
import unittest
from dataclasses import replace
from types import SimpleNamespace

from mavlink_rx import (
    ALTERNATIVE_IMU_MESSAGE_IDS,
    HIGHRES_IMU_MESSAGE_ID,
    MILLIG_TO_M_S2,
    MAVLinkRX,
    configure_mavlink_udp_receive_buffer,
    mavlink_message_source_target,
    request_mavlink_message_rate,
)
from runtime_config import load_runtime_config


def _config(runner_mode, *, yaw_inverted):
    config = load_runtime_config()
    return replace(
        config,
        runtime=replace(
            config.runtime,
            runner_mode=runner_mode,
            competition_yaw_inverted=yaw_inverted,
        ),
    )


def _attitude_message(yaw_deg, yawspeed):
    return SimpleNamespace(
        roll=0.1,
        pitch=-0.2,
        yaw=math.radians(yaw_deg),
        rollspeed=0.01,
        pitchspeed=-0.02,
        yawspeed=yawspeed,
        time_boot_ms=1234,
    )


def _imu_message(time_usec):
    return SimpleNamespace(
        xacc=1.0,
        yacc=2.0,
        zacc=3.0,
        xgyro=0.1,
        ygyro=0.2,
        zgyro=0.3,
        xmag=0.0,
        ymag=0.0,
        zmag=0.0,
        abs_pressure=1013.0,
        diff_pressure=0.0,
        pressure_alt=12.0,
        temperature=20.0,
        fields_updated=4095,
        time_usec=time_usec,
    )


def _wire_imu_message(time_usec, *, sequence, raw_packet):
    message = _imu_message(time_usec)
    message.get_type = lambda: "HIGHRES_IMU"
    message.get_seq = lambda: sequence
    message.get_srcSystem = lambda: 1
    message.get_srcComponent = lambda: 1
    message.get_msgbuf = lambda: raw_packet
    return message


class _FakeSocket:
    def __init__(self):
        self.requested = None

    def setsockopt(self, level, option, value):
        self.requested = (level, option, value)

    def getsockopt(self, level, option):
        return self.requested[2]


class _CommandSenderSpy:
    def __init__(self):
        self.calls = []

    def command_long_send(self, *args):
        self.calls.append(args)


class _RecorderSpy:
    enabled = True

    def __init__(self):
        self.imu_samples = []
        self.imu_sources = []

    def record_imu(self, sample):
        self.imu_samples.append(dict(sample))

    def record_imu_source(self, source, sample):
        self.imu_sources.append((source, dict(sample)))


class MAVLinkRXAttitudeTests(unittest.TestCase):
    def test_competition_inverts_incoming_yaw_and_yaw_rate(self):
        data = {}
        receiver = MAVLinkRX(
            None,
            data,
            config=_config("competition", yaw_inverted=True),
        )

        receiver.on_attitude(_attitude_message(-173.1, 0.25))

        attitude = data["attitude"]
        self.assertAlmostEqual(math.degrees(attitude["yaw"]), 173.1)
        self.assertAlmostEqual(attitude["yawspeed"], -0.25)
        self.assertAlmostEqual(
            math.degrees(attitude["yaw_mavlink_raw"]),
            -173.1,
        )
        self.assertAlmostEqual(attitude["yawspeed_mavlink_raw"], 0.25)
        self.assertTrue(attitude["competition_yaw_inverted"])

    def test_disabled_or_px4_mode_leaves_incoming_yaw_unchanged(self):
        for runner_mode, yaw_inverted in (
            ("competition", False),
            ("px4", True),
        ):
            with self.subTest(
                runner_mode=runner_mode,
                yaw_inverted=yaw_inverted,
            ):
                data = {}
                receiver = MAVLinkRX(
                    None,
                    data,
                    config=_config(
                        runner_mode,
                        yaw_inverted=yaw_inverted,
                    ),
                )

                receiver.on_attitude(_attitude_message(-42.0, 0.3))

                attitude = data["attitude"]
                self.assertAlmostEqual(
                    math.degrees(attitude["yaw"]),
                    -42.0,
                )
                self.assertAlmostEqual(attitude["yawspeed"], 0.3)
                self.assertFalse(attitude["competition_yaw_inverted"])

    def test_every_imu_callback_reaches_recorder_before_latest_value_replacement(self):
        data = {}
        recorder = _RecorderSpy()
        receiver = MAVLinkRX(
            None,
            data,
            config=_config("competition", yaw_inverted=True),
            vio_recorder=recorder,
        )

        receiver.on_highres_imu(_imu_message(100))
        receiver.on_highres_imu(_imu_message(200))

        self.assertEqual(
            [sample["time_usec"] for sample in recorder.imu_samples],
            [100, 200],
        )
        self.assertEqual(data["highres_imu"]["time_usec"], 200)
        self.assertIsInstance(recorder.imu_samples[0]["wall_time_ns"], int)

    def test_exact_wire_packet_retransmission_is_suppressed(self):
        data = {}
        recorder = _RecorderSpy()
        receiver = MAVLinkRX(
            None,
            data,
            config=_config("competition", yaw_inverted=True),
            vio_recorder=recorder,
        )
        packet = b"same-complete-mavlink-packet"

        self.assertTrue(
            receiver.process_message(
                _wire_imu_message(100, sequence=7, raw_packet=packet)
            )
        )
        self.assertFalse(
            receiver.process_message(
                _wire_imu_message(100, sequence=7, raw_packet=packet)
            )
        )

        self.assertEqual(len(recorder.imu_samples), 1)
        self.assertEqual(receiver.diagnostics_snapshot()["duplicates_suppressed"], 1)

    def test_equal_sensor_timestamp_is_kept_when_wire_packets_differ(self):
        data = {}
        recorder = _RecorderSpy()
        receiver = MAVLinkRX(
            None,
            data,
            config=_config("competition", yaw_inverted=True),
            vio_recorder=recorder,
        )

        receiver.process_message(
            _wire_imu_message(100, sequence=7, raw_packet=b"packet-7")
        )
        receiver.process_message(
            _wire_imu_message(100, sequence=8, raw_packet=b"packet-8")
        )

        self.assertEqual(len(recorder.imu_samples), 2)
        self.assertEqual(receiver.diagnostics_snapshot()["duplicates_suppressed"], 0)
        self.assertEqual(recorder.imu_samples[1]["mavlink_seq"], 8)

    def test_scaled_imu_is_converted_to_si_and_recorded_separately(self):
        data = {}
        recorder = _RecorderSpy()
        receiver = MAVLinkRX(None, data, vio_recorder=recorder)
        message = SimpleNamespace(
            time_boot_ms=1234,
            xacc=1000,
            yacc=-250,
            zacc=-1000,
            xgyro=100,
            ygyro=-200,
            zgyro=300,
            xmag=10,
            ymag=20,
            zmag=30,
            temperature=2500,
        )

        receiver.on_scaled_imu(message, "scaled_imu")

        sample = data["scaled_imu"]
        self.assertEqual(sample["time_usec"], 1_234_000)
        self.assertAlmostEqual(sample["xacc"], 1000 * MILLIG_TO_M_S2)
        self.assertAlmostEqual(sample["ygyro"], -0.2)
        self.assertAlmostEqual(sample["temperature"], 25.0)
        self.assertEqual(sample["xacc_raw"], 1000)
        self.assertEqual(recorder.imu_sources[0][0], "scaled_imu")

    def test_raw_imu_remains_unscaled(self):
        data = {}
        recorder = _RecorderSpy()
        receiver = MAVLinkRX(None, data, vio_recorder=recorder)
        message = SimpleNamespace(
            time_usec=456,
            xacc=1,
            yacc=2,
            zacc=3,
            xgyro=4,
            ygyro=5,
            zgyro=6,
            xmag=7,
            ymag=8,
            zmag=9,
            id=2,
            temperature=100,
        )

        receiver.on_raw_imu(message)

        self.assertEqual(data["raw_imu"]["yacc_raw"], 2)
        self.assertNotIn("yacc", data["raw_imu"])
        self.assertEqual(data["raw_imu"]["sensor_id"], 2)
        self.assertEqual(recorder.imu_sources[0][0], "raw_imu")

    def test_sequence_gap_is_reported_as_inferred_packet_loss(self):
        receiver = MAVLinkRX(
            None,
            {},
            config=_config("competition", yaw_inverted=True),
        )
        receiver.process_message(
            _wire_imu_message(100, sequence=10, raw_packet=b"packet-10")
        )
        receiver.process_message(
            _wire_imu_message(200, sequence=13, raw_packet=b"packet-13")
        )

        diagnostics = receiver.diagnostics_snapshot()
        self.assertEqual(diagnostics["inferred_packets_lost"], 2)
        self.assertEqual(diagnostics["missing_by_source"], {"1:1": 2})


class MAVLinkTransportConfigurationTests(unittest.TestCase):
    def test_runtime_defaults_enable_primary_imu_without_alternative_probe(self):
        mavlink = load_runtime_config().mavlink

        self.assertEqual(
            mavlink.udp_socket_receive_buffer_bytes,
            8 * 1024 * 1024,
        )
        self.assertTrue(mavlink.deduplicate_packets)
        self.assertEqual(mavlink.duplicate_cache_size, 1024)
        self.assertAlmostEqual(mavlink.duplicate_cache_ttl_s, 0.10)
        self.assertTrue(mavlink.request_highres_imu_rate)
        self.assertEqual(mavlink.highres_imu_rate_hz, 120.0)
        self.assertFalse(mavlink.request_alternative_imu_rates)
        self.assertEqual(mavlink.alternative_imu_rate_hz, 120.0)
        self.assertEqual(
            ALTERNATIVE_IMU_MESSAGE_IDS,
            {
                "SCALED_IMU": 26,
                "RAW_IMU": 27,
                "HIL_SENSOR": 107,
                "SCALED_IMU2": 116,
                "SCALED_IMU3": 129,
            },
        )

    def test_requested_socket_receive_buffer_is_applied_and_reported(self):
        fake_socket = _FakeSocket()
        connection = SimpleNamespace(port=fake_socket)

        effective = configure_mavlink_udp_receive_buffer(
            connection,
            8 * 1024 * 1024,
        )

        self.assertEqual(effective, 8 * 1024 * 1024)
        self.assertEqual(fake_socket.requested[0], socket.SOL_SOCKET)
        self.assertEqual(fake_socket.requested[1], socket.SO_RCVBUF)

    def test_highres_imu_rate_request_uses_expected_interval(self):
        sender = _CommandSenderSpy()
        connection = SimpleNamespace(
            mav=sender,
            target_system=1,
            target_component=2,
        )

        interval_usec = request_mavlink_message_rate(
            connection,
            HIGHRES_IMU_MESSAGE_ID,
            120.0,
        )

        self.assertEqual(interval_usec, 8333)
        call = sender.calls[0]
        self.assertEqual(call[0:2], (1, 2))
        self.assertEqual(call[4], float(HIGHRES_IMU_MESSAGE_ID))
        self.assertEqual(call[5], 8333.0)

    def test_message_rate_request_can_target_heartbeat_source_without_mutating_control_target(self):
        sender = _CommandSenderSpy()
        connection = SimpleNamespace(
            mav=sender,
            target_system=1,
            target_component=1,
        )
        heartbeat = SimpleNamespace(
            get_srcSystem=lambda: 1,
            get_srcComponent=lambda: 200,
        )
        target_system, target_component = mavlink_message_source_target(
            heartbeat,
            fallback_system=connection.target_system,
            fallback_component=connection.target_component,
        )

        request_mavlink_message_rate(
            connection,
            HIGHRES_IMU_MESSAGE_ID,
            120.0,
            target_system=target_system,
            target_component=target_component,
        )

        self.assertEqual(sender.calls[0][0:2], (1, 200))
        self.assertEqual(
            (connection.target_system, connection.target_component),
            (1, 1),
        )

    def test_message_source_target_falls_back_for_invalid_source_ids(self):
        heartbeat = SimpleNamespace(
            get_srcSystem=lambda: 0,
            get_srcComponent=lambda: None,
        )

        target = mavlink_message_source_target(
            heartbeat,
            fallback_system=4,
            fallback_component=9,
        )

        self.assertEqual(target, (4, 9))


if __name__ == "__main__":
    unittest.main()
