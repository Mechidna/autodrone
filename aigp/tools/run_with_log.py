#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any


STRUCTURED_PREFIXES = (
    ("autonomy_trace ", "autonomy_trace"),
    ("shadow_estimator_trace ", "shadow_estimator_trace"),
    ("hover_acquisition ", "hover_acquisition"),
    ("thrust_scale_calibration ", "thrust_scale_calibration"),
    ("lateral_response_calibration ", "lateral_response_calibration"),
    ("plan_install ", "plan_install"),
    ("plan_boundary_continuity ", "plan_boundary_continuity"),
    ("plan_validation_reject ", "plan_validation_reject"),
    ("gate_pass ", "gate_pass"),
    ("gate_center_pass_hold_exit ", "gate_center_pass_hold_exit"),
    ("post_gate_exit_continue ", "post_gate_exit_continue"),
    ("target_validation_reject ", "target_validation_reject"),
    ("planning_horizon_provisional_suffix ", "planning_horizon_provisional_suffix"),
    ("active_target_shift correction ", "active_target_shift"),
    (
        "active_target_shift_longitudinal_deferred ",
        "active_target_shift_longitudinal_deferred",
    ),
    (
        "active_target_shift_longitudinal_pending ",
        "active_target_shift_longitudinal_pending",
    ),
    (
        "active_target_shift_longitudinal_exit_replan ",
        "active_target_shift_longitudinal_exit_replan",
    ),
    (
        "active_target_shift_longitudinal_exit_replan_reject ",
        "active_target_shift_longitudinal_exit_replan_reject",
    ),
    ("target_manager lock ", "target_lock"),
    ("target_manager passed ", "target_passed"),
    ("target_manager active_lost ", "target_active_lost"),
    ("spline_memory ", "spline_memory"),
    ("[PERCEPTION_CHAIN] ", "perception_chain"),
    ("[PERCEPTION_DEBUG] ", "perception_debug"),
)

JSON_PREFIXES = (
    ("canonical_gate_poses ", "canonical_gate_poses"),
)

GATE_LIST_PREFIXES = (
    ("autonomy_wrapper ground truth gates NEU:", "ground_truth_gates"),
    ("autonomy_wrapper stable perception gates NEU:", "stable_perception_gates"),
    ("autonomy_wrapper race order gates NEU:", "race_order_gates"),
)

ALERT_MARKERS = (
    "Traceback",
    "[ERROR]",
    "[WARN]",
    " error:",
    "Exception",
    "failed",
    "Failed",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0.0 else "-inf"
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _load_runtime_config(repo: Path) -> dict[str, Any] | None:
    path = repo / "aigp" / "config" / "runtime.toml"
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return None


def _metadata(command: list[str], cwd: Path, repo: Path, run_id: str) -> dict[str, Any]:
    status_short = _run_git(["status", "--short"], repo)
    return {
        "event": "run_start",
        "run_id": run_id,
        "wall_time": time.time(),
        "iso_time": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cwd": str(cwd),
        "repo": str(repo),
        "command": command,
        "env": {
            key: os.environ[key]
            for key in (
                "RUNNER_MODE",
                "VISION_SOURCE",
                "MAVLINK_IP",
                "MAVLINK_PORT",
                "GATE_SOURCE_MODE",
                "PERCEPTION_BACKEND",
                "PERCEPTION_WORLD_POSE_SOURCE",
                "YOLO_MODEL_PATH",
                "CAMERA_MOUNT_PROFILE",
                "WORLD",
            )
            if key in os.environ
        },
        "git": {
            "branch": _run_git(["branch", "--show-current"], repo),
            "commit": _run_git(["rev-parse", "HEAD"], repo),
            "status_short": status_short.splitlines() if status_short else [],
        },
        "runtime_config": _jsonable(_load_runtime_config(repo)),
    }


def _parse_key_value_line(line: str, event: str, body: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for token in shlex.split(body, posix=True):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key:
            fields[key] = _coerce_scalar(value)
    return {
        "event": event,
        "wall_time": time.time(),
        "raw": line,
        "fields": fields,
    }


def _coerce_scalar(value: str) -> Any:
    lower = value.lower()
    if lower in ("true", "false"):
        return lower == "true"
    if lower in ("none", "null", "n/a"):
        return None
    if value[:1] in ("[", "{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_flow_status(line: str) -> dict[str, Any]:
    parts = line.split(maxsplit=8)
    fields: dict[str, Any] = {}
    labels = (
        "has_attitude",
        "has_imu",
        "has_frame",
        "has_perception",
        "has_command",
        "command_status",
        "perception_status",
    )
    for label, value in zip(labels, parts[2:]):
        if label.startswith("has_"):
            fields[label] = value == "True"
        else:
            fields[label] = value
    return {
        "event": "flow_status",
        "wall_time": time.time(),
        "raw": line,
        "fields": fields,
    }


def _event_from_line(line: str) -> dict[str, Any] | None:
    stripped = line.rstrip("\n")
    if not stripped:
        return None

    if stripped.startswith("flow status:"):
        return _parse_flow_status(stripped)

    if stripped.startswith("Starting main.py"):
        return {
            "event": "pilot_start",
            "wall_time": time.time(),
            "raw": stripped,
        }

    for prefix, event in STRUCTURED_PREFIXES:
        if stripped.startswith(prefix):
            return _parse_key_value_line(stripped, event, stripped[len(prefix) :])

    for prefix, event in JSON_PREFIXES:
        if stripped.startswith(prefix):
            body = stripped[len(prefix) :].strip()
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                payload = {"parse_error": True, "text": body}
            if not isinstance(payload, dict):
                payload = {"value": payload}
            payload = dict(payload)
            payload["event"] = event
            payload["wall_time"] = time.time()
            payload["raw"] = stripped
            return payload

    for prefix, event in GATE_LIST_PREFIXES:
        if stripped.startswith(prefix):
            return {
                "event": event,
                "wall_time": time.time(),
                "raw": stripped,
                "text": stripped[len(prefix) :].strip(),
            }

    if any(marker in stripped for marker in ALERT_MARKERS):
        return {
            "event": "terminal_alert",
            "wall_time": time.time(),
            "raw": stripped,
        }

    return None


def _default_command(repo: Path) -> list[str]:
    return [sys.executable, str(repo / "aigp" / "pilot" / "main.py")]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the AIGP pilot with two per-run log files: run.log and debug.jsonl."
        )
    )
    parser.add_argument(
        "--log-root",
        default=str(_repo_root() / "aigp" / "logs" / "runs"),
        help="Directory where timestamped run folders are created.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run folder name. Defaults to UTC timestamp.",
    )
    parser.add_argument(
        "--capture-camera-frames",
        action="store_true",
        help=(
            "Save decoded camera frames under the run directory for replay_debug_map.py. "
            "This can create many JPEG files."
        ),
    )
    parser.add_argument(
        "--camera-capture-hz",
        type=float,
        default=30.0,
        help="Maximum camera frame capture rate when --capture-camera-frames is set.",
    )
    parser.add_argument(
        "--camera-jpeg-quality",
        type=int,
        default=80,
        help="JPEG quality for captured replay camera frames.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Optional command to run after '--'. Defaults to the pilot main.py.",
    )
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args


def main() -> int:
    repo = _repo_root()
    cwd = Path.cwd()
    args = _parse_args()
    run_id = args.run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.log_root).expanduser()
    if not run_dir.is_absolute():
        run_dir = cwd / run_dir
    run_dir = run_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    command = list(args.command) if args.command else _default_command(repo)
    run_log_path = run_dir / "run.log"
    debug_path = run_dir / "debug.jsonl"
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if bool(args.capture_camera_frames):
        camera_dir = run_dir / "camera_frames"
        camera_dir.mkdir(parents=True, exist_ok=True)
        env["AIGP_CAMERA_CAPTURE_DIR"] = str(camera_dir)
        env["AIGP_CAMERA_CAPTURE_HZ"] = str(max(0.1, float(args.camera_capture_hz)))
        env["AIGP_CAMERA_CAPTURE_JPEG_QUALITY"] = str(
            max(1, min(100, int(args.camera_jpeg_quality)))
        )

    print(f"logging run to {run_dir}", flush=True)
    print(f"command: {' '.join(shlex.quote(part) for part in command)}", flush=True)
    if bool(args.capture_camera_frames):
        print(
            "camera frame capture: "
            f"{run_dir / 'camera_frames'} "
            f"hz={max(0.1, float(args.camera_capture_hz)):.1f} "
            f"jpeg_quality={max(1, min(100, int(args.camera_jpeg_quality)))}",
            flush=True,
        )

    start = time.time()
    proc: subprocess.Popen[str] | None = None
    returncode = 1

    with run_log_path.open("w", encoding="utf-8", buffering=1) as run_log, debug_path.open(
        "w",
        encoding="utf-8",
        buffering=1,
    ) as debug_log:

        def write_event(event: dict[str, Any]) -> None:
            debug_log.write(json.dumps(_jsonable(event), sort_keys=True) + "\n")

        write_event(_metadata(command, cwd, repo, run_id))
        try:
            proc = subprocess.Popen(
                command,
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                run_log.write(line)
                event = _event_from_line(line)
                if event is not None:
                    write_event(event)
            returncode = proc.wait()
        except KeyboardInterrupt:
            write_event(
                {
                    "event": "run_interrupted",
                    "wall_time": time.time(),
                }
            )
            if proc is not None and proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                returncode = proc.wait()
            else:
                returncode = 130
        finally:
            elapsed_s = time.time() - start
            write_event(
                {
                    "event": "run_end",
                    "wall_time": time.time(),
                    "elapsed_s": elapsed_s,
                    "returncode": returncode,
                    "run_log": str(run_log_path),
                    "debug_jsonl": str(debug_path),
                }
            )

    print(f"run log: {run_log_path}", flush=True)
    print(f"debug log: {debug_path}", flush=True)
    return int(returncode)


if __name__ == "__main__":
    raise SystemExit(main())
