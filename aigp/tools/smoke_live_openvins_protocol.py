"""Feed a prepared replay through the live OpenVINS stdin protocol."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shlex
import subprocess
import threading


def windows_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = resolved.as_posix().split(":", 1)[-1]
    return f"/mnt/{drive}{tail}"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--max-cameras", type=int, default=120)
    args = parser.parse_args()

    replay = args.replay.resolve()
    project = Path(__file__).resolve().parents[2]
    config = (
        project / "aigp" / "openvins" / "live_config" / "estimator_config.yaml"
    )
    imus = read_rows(replay / "imu.csv")
    cameras = read_rows(replay / "cam0" / "data.csv")[: args.max_cameras]
    command = (
        "exec $HOME/.cache/aigp_openvins_runner/run_vq1_dataset --live "
        f"{shlex.quote(windows_to_wsl(config))} red"
    )
    process = subprocess.Popen(
        ["wsl.exe", "bash", "-lc", command],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    states: list[str] = []
    errors: list[str] = []

    def read_output() -> None:
        for raw in iter(process.stdout.readline, b""):
            line = raw.decode("utf-8", errors="replace").strip()
            if line.startswith("AIGP_VIO_STATE "):
                states.append(line)
            elif line.startswith("AIGP_VIO_WAIT") or line.startswith("ERROR:"):
                print(line, flush=True)
            if line.startswith("ERROR:"):
                errors.append(line)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    imu_index = 0
    try:
        for camera in cameras:
            camera_time = float(camera["timestamp"])
            while (
                imu_index < len(imus)
                and float(imus[imu_index]["timestamp"]) <= camera_time
            ):
                row = imus[imu_index]
                values = " ".join(
                    row[name] for name in ("wx", "wy", "wz", "ax", "ay", "az")
                )
                process.stdin.write(
                    f"I {row['timestamp']} {values}\n".encode("ascii")
                )
                imu_index += 1
            if imu_index >= len(imus):
                raise RuntimeError("camera has no future IMU bracket")
            row = imus[imu_index]
            values = " ".join(
                row[name] for name in ("wx", "wy", "wz", "ax", "ay", "az")
            )
            process.stdin.write(
                f"I {row['timestamp']} {values}\n".encode("ascii")
            )
            imu_index += 1
            jpeg = (replay / "cam0" / camera["filename"]).read_bytes()
            process.stdin.write(
                (
                    f"C {camera['timestamp']} {camera['source_frame_id']} "
                    f"{len(jpeg)}\n"
                ).encode("ascii")
            )
            process.stdin.write(jpeg)
            process.stdin.write(b"\n")
            process.stdin.flush()
        process.stdin.write(b"Q\n")
        process.stdin.flush()
        process.stdin.close()
        code = process.wait(timeout=90.0)
        reader.join(timeout=2.0)
    finally:
        if process.poll() is None:
            process.terminate()

    if code != 0 or errors or not states:
        print(
            f"FAIL: exit={code} cameras={len(cameras)} states={len(states)} "
            f"errors={errors}",
            flush=True,
        )
        return 1
    print(
        f"PASS: live protocol cameras={len(cameras)} states={len(states)}",
        flush=True,
    )
    print(f"first: {states[0]}", flush=True)
    print(f"last:  {states[-1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
