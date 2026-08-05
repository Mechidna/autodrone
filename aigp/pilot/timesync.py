import time
import threading

from pymavlink import mavutil

TIMESYNC_REQUEST_HZ = 10

class TimeSync:

    def __init__(
        self,
        mavlink_connection,
        data,
        hz=TIMESYNC_REQUEST_HZ,
        vio_recorder=None,
    ):
        self.mavlink_conn = mavlink_connection
        self.data = data
        self.hz = float(hz)
        self.vio_recorder = vio_recorder
        self.thread = None
        self.is_running = False

    @classmethod
    def create_timesync(
        cls,
        mavlink_connection,
        data,
        hz=TIMESYNC_REQUEST_HZ,
        vio_recorder=None,
    ):
        ts = cls(
            mavlink_connection,
            data,
            hz=hz,
            vio_recorder=vio_recorder,
        )
        ts.thread = threading.Thread(
            target=ts.timesync_loop,
            daemon = False
        )
        ts.is_running = True
        ts.thread.start()
        return ts

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    def timesync_loop(self):
        while self.is_running:
            now = int(time.time_ns())
            self.mavlink_conn.mav.timesync_send(
                now,  # tc1 = client time
                0     # ts1 = 0 (request)
            )
            recorder = self.vio_recorder
            if recorder is not None and recorder.enabled:
                try:
                    recorder.record_timesync(
                        direction="tx",
                        tc1=now,
                        ts1=0,
                        wall_time_ns=now,
                    )
                except Exception as exc:
                    print(f"WARNING: VIO recorder timesync tx failed: {exc}", flush=True)
            time.sleep(1.0 / max(1.0, self.hz))
