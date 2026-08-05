"""Exercise the Windows callback bridge using a raw captured VIO dataset."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import threading
import time


PILOT_DIR = Path(__file__).resolve().parents[1] / "pilot"
sys.path.insert(0, str(PILOT_DIR))

from live_openvins import LiveOpenVins  # noqa: E402


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--timeout-s", type=float, default=45.0)
    args = parser.parse_args()
    dataset = args.dataset.resolve()

    events: list[tuple[int, str, dict[str, str]]] = []
    for row in read_rows(dataset / "imu" / "data.csv"):
        events.append((int(row["wall_time_ns"]), "imu", row))
    for row in read_rows(dataset / "camera" / "data.csv"):
        events.append((int(row["receive_wall_time_ns"]), "camera", row))
    for row in read_rows(dataset / "timesync" / "data.csv"):
        events.append((int(row["wall_time_ns"]), "timesync", row))
    events.sort(key=lambda item: item[0])

    os.environ["AIGP_LIVE_OPENVINS"] = "1"
    shared = {"lock": threading.Lock()}
    bridge = LiveOpenVins(shared, config=None)
    try:
        for _, kind, row in events:
            if kind == "imu":
                bridge.record_imu(
                    {
                        "time_usec": int(row["time_usec"]),
                        "wall_time_ns": int(row["wall_time_ns"]),
                        **{
                            name: float(row[name])
                            for name in (
                                "xacc",
                                "yacc",
                                "zacc",
                                "xgyro",
                                "ygyro",
                                "zgyro",
                            )
                        },
                    }
                )
            elif kind == "camera":
                image_path = dataset / "camera" / row["filename"]
                bridge.record_camera_frame(
                    sim_time_ns=int(row["sim_time_ns"]),
                    frame_id=int(row["frame_id"]),
                    receive_wall_time_ns=int(row["receive_wall_time_ns"]),
                    width=int(row["width"]),
                    height=int(row["height"]),
                    total_chunks=int(row["total_chunks"]),
                    jpeg_bytes=image_path.read_bytes(),
                )
            else:
                bridge.record_timesync(
                    direction=row["direction"],
                    tc1=int(row["tc1"]),
                    ts1=int(row["ts1"]),
                    wall_time_ns=int(row["wall_time_ns"]),
                )

        deadline = time.time() + args.timeout_s
        while time.time() < deadline:
            summary = bridge.summary()
            if summary["states_received"] >= 10:
                break
            process = bridge._process
            if process is not None and process.poll() is not None:
                break
            time.sleep(0.1)
        summary = bridge.summary()
        status = shared.get("external_vio_status")
        print(f"bridge status={status} counts={summary}", flush=True)
        if summary["states_received"] < 10 or status != "tracking":
            print("FAIL: callback bridge did not establish tracking", flush=True)
            return 1
        state = shared["external_vio_local_position_ned"]
        attitude = shared["external_vio_attitude"]
        print(
            "PASS: callback bridge tracking "
            f"pos_neu={state['pos_neu']} vel_neu={state['vel_neu']} "
            f"yaw_rad={attitude['yaw']:.6f}",
            flush=True,
        )
        return 0
    finally:
        bridge.close()


if __name__ == "__main__":
    raise SystemExit(main())
