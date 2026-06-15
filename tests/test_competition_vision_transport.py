import subprocess
import sys
import unittest

from autonomy_core.perception.competition_image_adapter import CompetitionCameraFrame
from autonomy_core.runtime.competition_runner import (
    CompetitionRunner,
    CompetitionRunnerConfig,
    CompetitionRunnerMode,
)
from autonomy_core.runtime.competition_vision_transport import (
    CompetitionVisionTransport,
    CompetitionVisionTransportConfig,
    DEFAULT_VISION_BIND_HOST,
    DEFAULT_VISION_PORT,
)


class FakeSocket:
    def __init__(self, packets):
        self.packets = list(packets)
        self.recv_calls = 0
        self.close_calls = 0
        self.send_calls = 0
        self.bound = None
        self.blocking = None

    def recvfrom(self, _size):
        self.recv_calls += 1
        if self.packets:
            return self.packets.pop(0)
        raise BlockingIOError()

    def bind(self, address):
        self.bound = address

    def setblocking(self, value):
        self.blocking = value

    def close(self):
        self.close_calls += 1

    def sendto(self, *_args, **_kwargs):
        self.send_calls += 1
        raise AssertionError("Vision receive transport must not send")


class FakeImageAdapter:
    def __init__(self):
        self.packets = []

    def process_packet(self, packet):
        self.packets.append(packet)
        return CompetitionCameraFrame(
            frame=FakeArray((360, 640, 3), "uint8"),
            camera_matrix=FakeMatrix([[320.0, 0.0, 320.0], [0.0, 320.0, 180.0], [0.0, 0.0, 1.0]]),
            dist_coeffs=FakeMatrix([[0.0, 0.0, 0.0, 0.0, 0.0]]),
            image_stamp_sec=1,
            image_stamp_nanosec=2,
            image_received_wall_time=3.0,
        )


class FakeArray:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class FakeMatrix:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values

    def reshape(self, *_args):
        return self


class FakeAutonomy:
    def __init__(self):
        self.perception_updates = []

    def update_gate_memory_from_frame(self, **kwargs):
        self.perception_updates.append(kwargs)


class CompetitionVisionTransportTests(unittest.TestCase):
    def test_import_does_not_import_cv2_or_pymavlink(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import sys; "
                    "import autonomy_core.runtime.competition_vision_transport; "
                    "print('cv2' in sys.modules, 'pymavlink' in sys.modules)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.stdout.strip(), "False False")

    def test_receive_packets_is_nonblocking_receive_only_and_records_summary(self):
        socket = FakeSocket(
            [
                (b"packet-1", ("127.0.0.1", 10001)),
                (b"packet-2", ("127.0.0.1", 10001)),
            ]
        )
        transport = CompetitionVisionTransport(
            socket_obj=socket,
            sleep=lambda _seconds: None,
        )

        packets = transport.receive_packets(max_packets=3)
        summary = transport.summary()

        self.assertEqual(packets, (b"packet-1", b"packet-2"))
        self.assertEqual(socket.send_calls, 0)
        self.assertTrue(summary["receive_only"])
        self.assertEqual(summary["bind_host"], DEFAULT_VISION_BIND_HOST)
        self.assertEqual(summary["port"], DEFAULT_VISION_PORT)
        self.assertEqual(summary["stats"]["packets_received"], 2)
        self.assertEqual(summary["stats"]["bytes_received"], len(b"packet-1packet-2"))
        self.assertEqual(summary["stats"]["receive_timeouts"], 1)
        self.assertEqual(summary["stats"]["source_address_counts"]["127.0.0.1:10001"], 2)

    def test_transport_can_start_from_injected_socket_factory(self):
        fake_socket = FakeSocket([])
        created = []

        def factory(bind_host, port, recv_buffer_size):
            created.append((bind_host, port, recv_buffer_size))
            return fake_socket

        transport = CompetitionVisionTransport(socket_factory=factory)
        transport.start()

        self.assertEqual(created, [(DEFAULT_VISION_BIND_HOST, DEFAULT_VISION_PORT, 65536)])
        self.assertTrue(transport.is_started)

    def test_close_closes_owned_socket(self):
        fake_socket = FakeSocket([])
        transport = CompetitionVisionTransport(socket_factory=lambda *_args: fake_socket)

        transport.start()
        transport.close()

        self.assertEqual(fake_socket.close_calls, 1)
        self.assertFalse(transport.is_started)

    def test_transport_feeds_competition_runner_without_live_socket(self):
        transport = CompetitionVisionTransport(
            socket_obj=FakeSocket([(b"raw-vadr-packet", ("127.0.0.1", 5600))])
        )
        image_adapter = FakeImageAdapter()
        autonomy = FakeAutonomy()
        runner = CompetitionRunner(
            config=CompetitionRunnerConfig(mode=CompetitionRunnerMode.VISION_DRY_RUN),
            vision_transport=transport,
            image_adapter=image_adapter,
            autonomy=autonomy,
            clock=lambda: 100.0,
        )

        result = runner.step()

        self.assertEqual(result.vision_packets_processed, 1)
        self.assertEqual(result.vision_frames_completed, 1)
        self.assertEqual(result.perception_update_calls, 1)
        self.assertEqual(image_adapter.packets, [b"raw-vadr-packet"])
        self.assertEqual(len(autonomy.perception_updates), 1)
        self.assertFalse(result.command_publication_allowed)

    def test_invalid_config_rejects_bad_port(self):
        with self.assertRaises(ValueError):
            CompetitionVisionTransport(
                config=CompetitionVisionTransportConfig(port=70000)
            )


if __name__ == "__main__":
    unittest.main()
