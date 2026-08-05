import unittest

from timesync import TimeSync


class _RecorderSpy:
    enabled = True

    def __init__(self):
        self.samples = []

    def record_timesync(self, **sample):
        self.samples.append(sample)


class _MavSender:
    def __init__(self):
        self.sent = []
        self.owner = None

    def timesync_send(self, tc1, ts1):
        self.sent.append((tc1, ts1))
        self.owner.is_running = False


class _Connection:
    def __init__(self):
        self.mav = _MavSender()


class TimeSyncRecorderTests(unittest.TestCase):
    def test_outbound_request_is_recorded_with_the_exact_sent_timestamp(self):
        connection = _Connection()
        recorder = _RecorderSpy()
        timesync = TimeSync(
            connection,
            {},
            hz=1_000_000.0,
            vio_recorder=recorder,
        )
        connection.mav.owner = timesync
        timesync.is_running = True

        timesync.timesync_loop()

        self.assertEqual(len(connection.mav.sent), 1)
        self.assertEqual(len(recorder.samples), 1)
        tc1, ts1 = connection.mav.sent[0]
        self.assertEqual(recorder.samples[0]["direction"], "tx")
        self.assertEqual(recorder.samples[0]["tc1"], tc1)
        self.assertEqual(recorder.samples[0]["ts1"], ts1)
        self.assertEqual(recorder.samples[0]["wall_time_ns"], tc1)


if __name__ == "__main__":
    unittest.main()
