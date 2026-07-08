#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import numpy as np
    from autonomy_core.planning.minimum_snap_planner_multi_time_optimized import (
        MultiSegmentMinimumSnapPlanner,
    )
except Exception:  # pragma: no cover - replay still works with logged samples only.
    np = None
    MultiSegmentMinimumSnapPlanner = None


VECTOR_RE = re.compile(r"\(([^()]*)\)")
GATE_RE = re.compile(r"id=([^:\s]+):\(([^()]*)\)")


def _repo_root() -> Path:
    return _REPO_ROOT


def _latest_run(log_root: Path) -> Path:
    runs = [path for path in log_root.iterdir() if path.is_dir()]
    if not runs:
        raise FileNotFoundError(f"no run directories found in {log_root}")
    return sorted(runs, key=lambda path: path.name)[-1]


def _resolve_run(value: str | None) -> Path:
    repo = _repo_root()
    log_root = repo / "aigp" / "logs" / "runs"
    if value is None or value == "latest":
        return _latest_run(log_root)

    path = Path(value).expanduser()
    if path.is_file():
        return path.parent
    if path.is_dir():
        return path

    candidate = log_root / value
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"run not found: {value}")


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _parse_vec(value: Any, *, dims: int | None = None) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        vals = [_finite_float(item) for item in value]
    else:
        text = str(value).strip()
        match = VECTOR_RE.fullmatch(text)
        if match:
            text = match.group(1)
        vals = [_finite_float(part.strip()) for part in text.split(",")]
    if any(item is None for item in vals):
        return None
    out = [float(item) for item in vals if item is not None]
    if dims is not None and len(out) != dims:
        return None
    return out


def _parse_vec_list(value: Any) -> list[list[float]]:
    if value is None:
        return []
    text = str(value)
    out: list[list[float]] = []
    for match in VECTOR_RE.finditer(text):
        vec = _parse_vec(match.group(1), dims=3)
        if vec is not None:
            out.append(vec)
    return out


def _parse_int_list(value: Any) -> list[int | None]:
    if value is None:
        return []
    text = str(value).strip()
    match = VECTOR_RE.fullmatch(text)
    if match:
        text = match.group(1)
    out: list[int | None] = []
    for part in text.split(","):
        item = part.strip()
        if not item or item.lower() in ("none", "nan"):
            out.append(None)
            continue
        try:
            out.append(int(item))
        except ValueError:
            out.append(None)
    return out


def _parse_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    text = str(value).strip()
    match = VECTOR_RE.fullmatch(text)
    if match:
        text = match.group(1)
    out: list[float] = []
    for part in text.split(","):
        val = _finite_float(part.strip())
        if val is not None:
            out.append(float(val))
    return out


def _parse_waypoint_velocity_list(
    value: Any,
    *,
    expected_count: int,
) -> list[list[float]] | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("none", "null"):
        return None
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    rows: list[list[float]] = []
    for part in text.split(";"):
        item = part.strip()
        if not item or item.lower() in ("none", "nan"):
            rows.append([math.nan, math.nan, math.nan])
            continue
        vec = _parse_vec(item, dims=3)
        if vec is None:
            return None
        rows.append(vec)
    if expected_count > 0 and len(rows) != expected_count:
        return None
    return rows


def _reconstruct_plan_samples(
    *,
    waypoints: list[list[float]],
    times: list[float],
    v_start: list[float] | None,
    v_end: list[float] | None,
    waypoint_velocities: list[list[float]] | None,
    sample_count: int = 120,
) -> list[list[float]]:
    if np is None or MultiSegmentMinimumSnapPlanner is None:
        return []
    if len(waypoints) < 2 or len(times) != len(waypoints) - 1:
        return []
    if v_start is None or v_end is None:
        return []
    try:
        planner = MultiSegmentMinimumSnapPlanner()
        planner.update(
            waypoints=np.asarray(waypoints, dtype=float),
            times=np.asarray(times, dtype=float),
            v_start=np.asarray(v_start, dtype=float).reshape(3),
            v_end=np.asarray(v_end, dtype=float).reshape(3),
            a_start=np.zeros(3, dtype=float),
            a_end=np.zeros(3, dtype=float),
            j_start=np.zeros(3, dtype=float),
            j_end=np.zeros(3, dtype=float),
            waypoint_velocities=None
            if waypoint_velocities is None
            else np.asarray(waypoint_velocities, dtype=float),
        )
        total_time = float(planner.total_time)
        if not math.isfinite(total_time) or total_time <= 0.0:
            return []
        count = max(2, min(200, int(sample_count)))
        samples = []
        for tau in np.linspace(0.0, total_time, count):
            point, _, _ = planner.sample(float(tau))
            arr = np.asarray(point, dtype=float).reshape(3)
            if not np.all(np.isfinite(arr)):
                return []
            samples.append([float(arr[0]), float(arr[1]), float(arr[2])])
        return samples
    except Exception:
        return []


def _parse_gate_list(text: str) -> list[dict[str, Any]]:
    gates = []
    for order, match in enumerate(GATE_RE.finditer(text or "")):
        raw_id = match.group(1)
        try:
            gate_id: int | str = int(raw_id)
        except ValueError:
            gate_id = raw_id
        vec = _parse_vec(match.group(2), dims=3)
        if vec is None:
            continue
        gates.append(
            {
                "id": gate_id,
                "order": order,
                "p": vec,
            }
        )
    return gates


def _compact_canonical_gate_poses(event: dict[str, Any], start: float) -> dict[str, Any]:
    gates: list[dict[str, Any]] = []
    raw_poses = event.get("poses")
    if not isinstance(raw_poses, list):
        raw_poses = []
    for order, item in enumerate(raw_poses):
        if not isinstance(item, dict):
            continue
        center = _parse_vec(item.get("center_neu"), dims=3)
        if center is None:
            continue
        gate_id = item.get("id", order)
        try:
            gate_id = int(gate_id)
        except (TypeError, ValueError):
            pass
        gate = {
            "id": gate_id,
            "order": order,
            "p": center,
            "center_neu": center,
            "right_axis_neu": _parse_vec(item.get("right_axis_neu"), dims=3),
            "up_axis_neu": _parse_vec(item.get("up_axis_neu"), dims=3),
            "normal_neu": _parse_vec(item.get("normal_neu"), dims=3),
            "sdf_pose": _parse_vec(item.get("sdf_pose"), dims=6),
            "sdf_gate_index": item.get("sdf_gate_index"),
            "sdf_model": item.get("sdf_model"),
        }
        gates.append(gate)
    return {
        "t": _event_time(event, start),
        "kind": "canonical_gate_poses",
        "source": event.get("source"),
        "world_sdf": event.get("world_sdf"),
        "gates": gates,
    }


def _known_gates_from_runtime(runtime_config: Any) -> list[dict[str, Any]]:
    if not isinstance(runtime_config, dict):
        return []
    candidate_sections = (
        runtime_config.get("gate_source") or {},
        runtime_config.get("perception_geometry_audit") or {},
        runtime_config.get("state_estimation") or {},
    )
    positions = []
    for section in candidate_sections:
        values = section.get("known_gate_positions_neu")
        if isinstance(values, list) and values:
            positions = values
            break

    gates = []
    for order, pos in enumerate(positions):
        vec = _parse_vec(pos, dims=3)
        if vec is not None:
            gates.append({"id": order, "order": order, "p": vec})
    return gates


def _gate_size_from_runtime(runtime_config: Any) -> float:
    if isinstance(runtime_config, dict):
        perception = runtime_config.get("perception") or {}
        gate_size = _finite_float(perception.get("gate_size_m"))
        if gate_size is not None and gate_size > 0:
            return gate_size
    return 1.5


def _field_vec(fields: dict[str, Any], name: str, *, dims: int = 3) -> list[float] | None:
    return _parse_vec(fields.get(name), dims=dims)


def _event_time(event: dict[str, Any], start: float) -> float:
    value = _finite_float(event.get("wall_time"))
    if value is None:
        return 0.0
    return max(0.0, value - start)


def _compact_frame(event: dict[str, Any], start: float) -> dict[str, Any] | None:
    fields = event.get("fields") or {}
    pos = _field_vec(fields, "pos_neu")
    truth = _field_vec(fields, "truth_pos_neu")
    pref = _field_vec(fields, "p_ref")
    if pos is None and truth is None and pref is None:
        return None
    return {
        "t": _event_time(event, start),
        "gate_idx": fields.get("gate_idx"),
        "active_track": fields.get("active_track"),
        "tracks": fields.get("tracks"),
        "target_lock": fields.get("target_lock"),
        "target_event": fields.get("target_event"),
        "target_shift": fields.get("target_shift"),
        "dist": fields.get("dist"),
        "yaw_source": fields.get("yaw_source"),
        "pos": pos,
        "truth": truth,
        "p_ref": pref,
        "target": _field_vec(fields, "target_neu"),
        "yaw_target": _field_vec(fields, "yaw_target_neu"),
        "v_ref": _field_vec(fields, "v_ref"),
        "cmd_deg": _field_vec(fields, "cmd_deg"),
    }


def _compact_plan(event: dict[str, Any], start: float) -> dict[str, Any]:
    fields = event.get("fields") or {}
    waypoints = _parse_vec_list(fields.get("waypoints_neu"))
    times = _parse_float_list(fields.get("times_s"))
    v_start = _field_vec(fields, "v_start_neu")
    v_end = _field_vec(fields, "v_end_neu")
    waypoint_velocities = _parse_waypoint_velocity_list(
        fields.get("waypoint_velocities_neu"),
        expected_count=len(waypoints),
    )
    samples = _parse_vec_list(fields.get("plan_samples_neu"))
    if not samples:
        samples = _reconstruct_plan_samples(
            waypoints=waypoints,
            times=times,
            v_start=v_start,
            v_end=v_end,
            waypoint_velocities=waypoint_velocities,
        )
    return {
        "t": _event_time(event, start),
        "gate_idx": fields.get("gate_idx"),
        "track": fields.get("track"),
        "mode": fields.get("mode"),
        "horizon_tracks": _parse_int_list(fields.get("horizon_tracks")),
        "horizon_gate_indices": _parse_int_list(fields.get("horizon_gate_indices")),
        "total_time": fields.get("total_time"),
        "segments": fields.get("segments"),
        "terminal_policy": fields.get("terminal_policy"),
        "target": _field_vec(fields, "target_neu"),
        "normal": _field_vec(fields, "normal_neu"),
        "times": times,
        "v_start": v_start,
        "v_end": v_end,
        "waypoints": waypoints,
        "samples": samples,
    }


def _compact_rejected_plan(event: dict[str, Any], start: float) -> dict[str, Any]:
    fields = event.get("fields") or {}
    out = _compact_plan(event, start)
    out.update(
        {
            "reason": fields.get("reason"),
            "fallback": fields.get("fallback"),
            "sample_time": fields.get("sample_time"),
            "lateral": fields.get("lateral"),
            "progress": fields.get("progress"),
            "crossing": _field_vec(fields, "crossing"),
            "path_ratio": fields.get("path_ratio"),
            "segment_ratio": fields.get("segment_ratio"),
            "corridor": fields.get("corridor"),
            "polyline_backtrack": fields.get("polyline_backtrack"),
            "speed": fields.get("speed"),
            "accel": fields.get("accel"),
            "accel_xy": fields.get("accel_xy"),
            "lateral_accel": fields.get("lateral_accel"),
            "accel_z_up": fields.get("accel_z_up"),
            "accel_z_down": fields.get("accel_z_down"),
            "z_overshoot": fields.get("z_overshoot"),
        }
    )
    return out


def _compact_pass(event: dict[str, Any], start: float) -> dict[str, Any]:
    fields = event.get("fields") or {}
    return {
        "t": _event_time(event, start),
        "gate_idx": fields.get("gate_idx"),
        "track": fields.get("track"),
        "next_gate_idx": fields.get("next_gate_idx"),
        "reason": fields.get("reason"),
        "distance": fields.get("distance"),
        "lateral_error": fields.get("lateral_error"),
        "pos": _field_vec(fields, "pos_neu"),
        "target": _field_vec(fields, "target_neu"),
    }


def _compact_lock(event: dict[str, Any], start: float) -> dict[str, Any]:
    fields = event.get("fields") or {}
    return {
        "t": _event_time(event, start),
        "gate_idx": fields.get("gate_idx"),
        "track": fields.get("track"),
        "reason": fields.get("reason"),
        "center": _field_vec(fields, "center"),
    }


def _compact_shift(event: dict[str, Any], start: float) -> dict[str, Any]:
    fields = event.get("fields") or {}
    return {
        "t": _event_time(event, start),
        "gate_idx": fields.get("gate_idx"),
        "track": fields.get("track"),
        "source_track": fields.get("source_track"),
        "shift": fields.get("shift"),
        "shift_xy": fields.get("shift_xy"),
        "shift_z": fields.get("shift_z"),
        "planned": _field_vec(fields, "planned"),
        "latest": _field_vec(fields, "latest"),
        "corrected": _field_vec(fields, "corrected"),
    }


def _compact_map(event: dict[str, Any], start: float) -> dict[str, Any]:
    return {
        "t": _event_time(event, start),
        "kind": event.get("event"),
        "gates": _parse_gate_list(str(event.get("text") or "")),
    }


def _downsample(items: list[dict[str, Any]], max_items: int) -> list[dict[str, Any]]:
    if max_items <= 0 or len(items) <= max_items:
        return items
    step = (len(items) - 1) / float(max_items - 1)
    out = []
    last_idx = -1
    for i in range(max_items):
        idx = round(i * step)
        if idx == last_idx:
            continue
        out.append(items[idx])
        last_idx = idx
    return out


def _load_camera_frames(run_dir: Path, start: float) -> list[dict[str, Any]]:
    index_path = run_dir / "camera_frames" / "index.jsonl"
    if not index_path.exists():
        return []

    frames: list[dict[str, Any]] = []
    try:
        with index_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(raw, dict):
                    continue
                wall_time = _finite_float(raw.get("wall_time"))
                rel_path = str(raw.get("path") or "").strip()
                if wall_time is None or not rel_path:
                    continue
                path = index_path.parent / rel_path
                frames.append(
                    {
                        "t": max(0.0, float(wall_time) - start),
                        "frame_id": raw.get("frame_id"),
                        "source": raw.get("source"),
                        "path": str(Path("camera_frames") / rel_path),
                        "width": raw.get("width"),
                        "height": raw.get("height"),
                        "sim_time_ns": raw.get("sim_time_ns"),
                        "ros_stamp_sec": raw.get("ros_stamp_sec"),
                        "ros_stamp_nanosec": raw.get("ros_stamp_nanosec"),
                        "exists": path.exists(),
                    }
                )
    except OSError:
        return []
    return sorted(frames, key=lambda item: float(item.get("t") or 0.0))


def _load_debug(path: Path, *, max_frames: int) -> dict[str, Any]:
    events = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not events:
        raise ValueError(f"no JSON events found in {path}")

    start = min(
        float(event["wall_time"])
        for event in events
        if _finite_float(event.get("wall_time")) is not None
    )
    frames = []
    plans = []
    rejected_plans = []
    passes = []
    locks = []
    shifts = []
    maps = []
    canonical_gate_poses = []
    alerts = []
    run_start = None
    run_end = None

    for event in events:
        kind = event.get("event")
        if kind == "run_start":
            run_start = event
        elif kind == "run_end":
            run_end = event
        elif kind == "autonomy_trace":
            frame = _compact_frame(event, start)
            if frame is not None:
                frames.append(frame)
        elif kind == "plan_install":
            plans.append(_compact_plan(event, start))
        elif kind == "plan_candidate_reject":
            rejected_plans.append(_compact_rejected_plan(event, start))
        elif kind == "gate_pass":
            passes.append(_compact_pass(event, start))
        elif kind == "target_lock":
            locks.append(_compact_lock(event, start))
        elif kind == "active_target_shift":
            shifts.append(_compact_shift(event, start))
        elif kind in (
            "ground_truth_gates",
            "stable_perception_gates",
            "race_order_gates",
        ):
            maps.append(_compact_map(event, start))
        elif kind == "canonical_gate_poses":
            canonical_gate_poses.append(_compact_canonical_gate_poses(event, start))
        elif kind == "terminal_alert":
            alerts.append(
                {
                    "t": _event_time(event, start),
                    "text": str(event.get("raw") or "")[:220],
                }
            )

    frames = _downsample(frames, max_frames)
    duration = max(
        [0.0]
        + [float(item["t"]) for item in frames]
        + [float(item["t"]) for item in plans]
        + [float(item["t"]) for item in rejected_plans]
        + [float(item["t"]) for item in passes]
    )
    env = {}
    runtime_config = None
    if isinstance(run_start, dict):
        env = run_start.get("env") or {}
        runtime_config = run_start.get("runtime_config")
    camera_frames = _load_camera_frames(path.parent, start)

    return {
        "run_id": path.parent.name,
        "debug_path": str(path),
        "duration": duration,
        "env": env,
        "runtime_config": runtime_config,
        "known_gates": _known_gates_from_runtime(runtime_config),
        "gate_size_m": _gate_size_from_runtime(runtime_config),
        "returncode": None if run_end is None else run_end.get("returncode"),
        "frames": frames,
        "plans": plans,
        "rejected_plans": rejected_plans,
        "passes": passes,
        "locks": locks,
        "shifts": shifts,
        "maps": maps,
        "canonical_gate_poses": canonical_gate_poses,
        "camera_frames": camera_frames,
        "alerts": alerts[-20:],
        "counts": {
            "frames": len(frames),
            "plans": len(plans),
            "rejected_plans": len(rejected_plans),
            "passes": len(passes),
            "locks": len(locks),
            "shifts": len(shifts),
            "maps": len(maps),
            "canonical_gate_poses": len(canonical_gate_poses),
            "camera_frames": len(camera_frames),
            "alerts": len(alerts),
        },
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AIGP Debug Replay - __RUN_ID__</title>
<style>
:root {
  color-scheme: dark;
  --bg: #101418;
  --panel: #171d23;
  --muted: #9aa7b3;
  --text: #eef4f8;
  --grid: #28313a;
  --accent: #63c7ff;
  --truth: #60d394;
  --ref: #ffb454;
  --plan: #d38cff;
  --rejected-plan: #ff8a4c;
  --race: #47d18c;
  --stable: #ffd166;
  --gt: #d6dde5;
  --pass: #ff6b6b;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.35 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
header {
  padding: 14px 18px;
  background: #0b0e11;
  border-bottom: 1px solid #28313a;
}
h1 {
  margin: 0 0 4px;
  font-size: 18px;
  font-weight: 650;
}
.sub { color: var(--muted); font-size: 12px; }
main {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 360px;
  gap: 12px;
  padding: 12px;
}
.panel {
  background: var(--panel);
  border: 1px solid #29333d;
  border-radius: 8px;
  overflow: hidden;
}
.canvas-wrap { position: relative; min-height: 620px; }
canvas {
  display: block;
  width: 100%;
  background: #101820;
}
#mapCanvas { height: 620px; }
#altCanvas { height: 180px; border-top: 1px solid #29333d; }
#cameraCanvas { width: 100%; height: 202px; border-radius: 6px; background: #080c10; }
#recordCanvas { display: none; }
.toolbar {
  display: grid;
  grid-template-columns: auto auto auto 1fr auto auto;
  gap: 10px;
  align-items: center;
  padding: 10px;
  border-bottom: 1px solid #29333d;
}
button, select {
  color: var(--text);
  background: #202a33;
  border: 1px solid #394653;
  border-radius: 6px;
  padding: 7px 10px;
}
input[type="range"] { width: 100%; }
.checks {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px 14px;
  padding: 10px;
  border-bottom: 1px solid #29333d;
  color: #dbe5ec;
  font-size: 13px;
}
label { white-space: nowrap; }
aside {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.side-section {
  padding: 12px;
}
.kv {
  display: grid;
  grid-template-columns: 128px 1fr;
  gap: 4px 8px;
  font-size: 13px;
}
.kv div:nth-child(odd) { color: var(--muted); }
.legend {
  display: grid;
  grid-template-columns: 16px 1fr;
  gap: 6px 8px;
  color: #dbe5ec;
  font-size: 13px;
}
.swatch { width: 14px; height: 14px; border-radius: 4px; margin-top: 2px; }
.events {
  max-height: 280px;
  overflow: auto;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  color: #d5dde5;
  white-space: pre-wrap;
}
.record-status {
  color: var(--muted);
  font-size: 12px;
  margin-top: 8px;
}
.hint { color: var(--muted); font-size: 12px; margin-top: 8px; }
@media (max-width: 1000px) {
  main { grid-template-columns: 1fr; }
  #mapCanvas { height: 520px; }
  .checks { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
</head>
<body>
<header>
  <h1>AIGP Debug Replay: __RUN_ID__</h1>
  <div class="sub" id="subtitle"></div>
</header>
<main>
  <section class="panel">
    <div class="toolbar">
      <button id="playBtn">Play</button>
      <button id="saveVideoBtn">Save video</button>
      <select id="viewMode">
        <option value="xy">Top-down x/y</option>
        <option value="yz">Side y/z</option>
        <option value="3d" selected>3D orbit</option>
      </select>
      <input id="slider" type="range" min="0" max="0" value="0">
      <select id="speed">
        <option value="0.5">0.5x</option>
        <option value="1" selected>1x</option>
        <option value="2">2x</option>
        <option value="4">4x</option>
        <option value="8">8x</option>
      </select>
      <span id="timeLabel">0.0s</span>
    </div>
    <div class="checks">
      <label><input type="checkbox" id="showTruth" checked> truth path</label>
      <label><input type="checkbox" id="showRef" checked> p_ref path</label>
      <label><input type="checkbox" id="showPlan" checked> plan spline</label>
      <label><input type="checkbox" id="showRejectedPlans" checked> failed plans</label>
      <label><input type="checkbox" id="showRace" checked> race order</label>
      <label><input type="checkbox" id="showStable" checked> stable map</label>
      <label><input type="checkbox" id="showGt" checked> ground truth</label>
      <label><input type="checkbox" id="showLabels" checked> labels</label>
      <label><input type="checkbox" id="followDrone" checked> follow drone</label>
    </div>
    <div class="canvas-wrap">
      <canvas id="mapCanvas"></canvas>
      <canvas id="altCanvas"></canvas>
    </div>
  </section>
  <aside>
    <section class="panel side-section">
      <h2 style="margin:0 0 10px;font-size:15px;">Camera</h2>
      <canvas id="cameraCanvas"></canvas>
      <div class="record-status" id="videoStatus"></div>
    </section>
    <section class="panel side-section">
      <h2 style="margin:0 0 10px;font-size:15px;">Current Frame</h2>
      <div class="kv" id="frameInfo"></div>
    </section>
    <section class="panel side-section">
      <h2 style="margin:0 0 10px;font-size:15px;">Legend</h2>
      <div class="legend">
        <div class="swatch" style="background:var(--accent)"></div><div>drone / odometry path</div>
        <div class="swatch" style="background:var(--truth)"></div><div>truth path</div>
        <div class="swatch" style="background:var(--ref)"></div><div>controller reference p_ref</div>
        <div class="swatch" style="background:var(--plan)"></div><div>latest installed minimum-snap spline</div>
        <div class="swatch" style="background:var(--rejected-plan)"></div><div>failed validation candidates</div>
        <div class="swatch" style="background:var(--race)"></div><div>internal race order map</div>
        <div class="swatch" style="background:var(--stable)"></div><div>stable perception tracks</div>
        <div class="swatch" style="background:var(--pass)"></div><div>passed gate marker</div>
      </div>
      <div class="hint">Use Top-down for lateral map order, Side y/z for over/under failures, and 3D orbit for gate/drone pose context. Drag and scroll the 3D view to inspect.</div>
    </section>
    <section class="panel side-section">
      <h2 style="margin:0 0 10px;font-size:15px;">Recent Events</h2>
      <div class="events" id="events"></div>
    </section>
  </aside>
</main>
<canvas id="recordCanvas"></canvas>
<script>
const DATA = __DATA__;

const mapCanvas = document.getElementById('mapCanvas');
const altCanvas = document.getElementById('altCanvas');
const cameraCanvas = document.getElementById('cameraCanvas');
const recordCanvas = document.getElementById('recordCanvas');
const slider = document.getElementById('slider');
const playBtn = document.getElementById('playBtn');
const saveVideoBtn = document.getElementById('saveVideoBtn');
const viewMode = document.getElementById('viewMode');
const speedSel = document.getElementById('speed');
const timeLabel = document.getElementById('timeLabel');
const frameInfo = document.getElementById('frameInfo');
const eventsBox = document.getElementById('events');
const subtitle = document.getElementById('subtitle');
const videoStatus = document.getElementById('videoStatus');
const checks = ['showTruth','showRef','showPlan','showRejectedPlans','showRace','showStable','showGt','showLabels','followDrone']
  .reduce((acc, id) => { acc[id] = document.getElementById(id); return acc; }, {});

const frames = DATA.frames || [];
const cameraFrames = DATA.camera_frames || [];
slider.max = Math.max(0, frames.length - 1);
subtitle.textContent = `${DATA.debug_path} | frames=${DATA.counts.frames}, camera=${DATA.counts.camera_frames || 0}, plans=${DATA.counts.plans}, failed_plans=${DATA.counts.rejected_plans || 0}, passes=${DATA.counts.passes}, shifts=${DATA.counts.shifts}, return=${DATA.returncode}`;
const gateSizeM = Number(DATA.gate_size_m || 1.5);

let frameIndex = 0;
let playing = false;
let lastTick = performance.now();
let playTime = frames.length ? Number(frames[0].t || 0) : 0;
let recording = false;

function finite(v) { return Number.isFinite(v); }
function pxy(p) { return p && finite(p[0]) && finite(p[1]); }
function p3(p) { return p && finite(p[0]) && finite(p[1]) && finite(p[2]); }
function pz(p) { return p && finite(p[2]); }
function fmt(v, n=2) { return Number.isFinite(Number(v)) ? Number(v).toFixed(n) : 'n/a'; }
function fmtVec(p) { return p ? `(${fmt(p[0])}, ${fmt(p[1])}, ${fmt(p[2])})` : 'none'; }
function add3(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function sub3(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function mul3(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
function dot3(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function cross3(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}
function len3(a) { return Math.hypot(a[0], a[1], a[2]); }
function norm3(a, fallback=[0, 1, 0]) {
  const n = len3(a);
  return n > 1e-6 ? [a[0] / n, a[1] / n, a[2] / n] : fallback.slice();
}
function dist3(a, b) { return p3(a) && p3(b) ? len3(sub3(a, b)) : Infinity; }

function latestAt(items, t, predicate = null) {
  let best = null;
  for (const item of items || []) {
    if (item.t <= t && (!predicate || predicate(item))) best = item;
    if (item.t > t) break;
  }
  return best;
}

function recentAt(items, t, count, predicate = null) {
  const out = [];
  for (const item of items || []) {
    if (item.t <= t && (!predicate || predicate(item))) out.push(item);
    if (item.t > t) break;
  }
  return out.slice(-count);
}

function planCurvePoints(plan) {
  if (!plan) return [];
  if (plan.samples && plan.samples.length >= 2) return plan.samples;
  return plan.waypoints || [];
}

function rejectedPlansForTime(t) {
  return recentAt(DATA.rejected_plans || [], t, 10)
    .filter(p => Number(t) - Number(p.t || 0) <= 8);
}

function frameIndexForTime(t) {
  if (!frames.length) return 0;
  if (t <= frames[0].t) return 0;
  let lo = 0;
  let hi = frames.length - 1;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (frames[mid].t <= t) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

function cameraFrameIndexForTime(t) {
  if (!cameraFrames.length) return -1;
  if (t <= cameraFrames[0].t) return 0;
  let lo = 0;
  let hi = cameraFrames.length - 1;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (cameraFrames[mid].t <= t) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

function cameraFrameForTime(t) {
  const idx = cameraFrameIndexForTime(t);
  return idx >= 0 ? cameraFrames[idx] : null;
}

const imageCache = new Map();

function loadCameraImage(item) {
  if (!item || !item.path) return Promise.resolve(null);
  const key = item.path;
  const cached = imageCache.get(key);
  if (cached) {
    if (cached.loaded) return Promise.resolve(cached.img);
    if (cached.failed) return Promise.resolve(null);
    return cached.promise;
  }
  const img = new Image();
  const entry = {
    img,
    loaded: false,
    failed: false,
    promise: null,
  };
  entry.promise = new Promise(resolve => {
    img.onload = () => {
      entry.loaded = true;
      resolve(img);
    };
    img.onerror = () => {
      entry.failed = true;
      resolve(null);
    };
  });
  imageCache.set(key, entry);
  img.src = key;
  return entry.promise;
}

function drawCenteredImage(ctx, img, x, y, w, h) {
  const iw = img.naturalWidth || img.width || 1;
  const ih = img.naturalHeight || img.height || 1;
  const scale = Math.min(w / iw, h / ih);
  const dw = iw * scale;
  const dh = ih * scale;
  ctx.drawImage(img, x + (w - dw) / 2, y + (h - dh) / 2, dw, dh);
}

function drawCamera() {
  const { ctx, w, h } = resizeCanvas(cameraCanvas);
  ctx.fillStyle = '#080c10';
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = '#9aa7b3';
  ctx.font = '12px ui-monospace, monospace';

  if (!cameraFrames.length) {
    ctx.fillText('No camera frames captured for this run.', 12, 24);
    if (videoStatus) {
      videoStatus.textContent = 'Run with --capture-camera-frames to include /camera images.';
    }
    return;
  }

  const t = frames[frameIndex] ? Number(frames[frameIndex].t || 0) : 0;
  const item = cameraFrameForTime(t);
  if (!item) {
    ctx.fillText('Waiting for camera frame...', 12, 24);
    return;
  }

  const cached = imageCache.get(item.path);
  if (cached && cached.loaded) {
    drawCenteredImage(ctx, cached.img, 0, 0, w, h);
  } else {
    ctx.fillText('Loading camera frame...', 12, 24);
    loadCameraImage(item).then(() => {
      const current = cameraFrameForTime(frames[frameIndex] ? Number(frames[frameIndex].t || 0) : 0);
      if (current && current.path === item.path) drawCamera();
    });
  }

  ctx.fillStyle = 'rgba(0, 0, 0, 0.62)';
  ctx.fillRect(0, h - 24, w, 24);
  ctx.fillStyle = '#eef4f8';
  ctx.fillText(
    `camera t=${fmt(item.t, 2)}s frame=${item.frame_id ?? 'n/a'} source=${item.source ?? 'n/a'}`,
    10,
    h - 8,
  );
}

function allPoints() {
  const pts = [];
  for (const f of frames) {
    if (pxy(f.pos)) pts.push(f.pos);
    if (pxy(f.truth)) pts.push(f.truth);
    if (pxy(f.p_ref)) pts.push(f.p_ref);
    if (pxy(f.target)) pts.push(f.target);
  }
  for (const p of DATA.plans || []) for (const w of p.waypoints || []) if (pxy(w)) pts.push(w);
  for (const p of DATA.rejected_plans || []) {
    for (const w of planCurvePoints(p)) if (pxy(w)) pts.push(w);
  }
  for (const g of DATA.known_gates || []) if (pxy(g.p)) pts.push(g.p);
  for (const m of DATA.maps || []) for (const g of m.gates || []) if (pxy(g.p)) pts.push(g.p);
  for (const gp of DATA.passes || []) {
    if (pxy(gp.pos)) pts.push(gp.pos);
    if (pxy(gp.target)) pts.push(gp.target);
  }
  return pts;
}

const globalPts = allPoints();
const globalBounds = (() => {
  const xs = globalPts.map(p => p[0]);
  const ys = globalPts.map(p => p[1]);
  const zs = globalPts.filter(pz).map(p => p[2]);
  const pad = 4;
  return {
    minX: Math.min(...xs, -1) - pad,
    maxX: Math.max(...xs, 1) + pad,
    minY: Math.min(...ys, -1) - pad,
    maxY: Math.max(...ys, 1) + pad,
    minZ: Math.min(...zs, 0) - 1,
    maxZ: Math.max(...zs, 1) + 1,
  };
})();

const orbit = {
  yaw: -0.75,
  pitch: 0.55,
  distance: null,
  dragging: false,
  lastX: 0,
  lastY: 0,
};

function resizeCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: rect.width, h: rect.height };
}

function makeProjector(w, h, frame) {
  const mode = viewMode.value || 'xy';
  const side = mode === 'yz';
  const hIdx = side ? 1 : 0;
  const vIdx = side ? 2 : 1;
  const labelH = side ? 'y' : 'x';
  const labelV = side ? 'z' : 'y';
  let b = side
    ? {
        minH: globalBounds.minY,
        maxH: globalBounds.maxY,
        minV: globalBounds.minZ,
        maxV: globalBounds.maxZ,
      }
    : {
        minH: globalBounds.minX,
        maxH: globalBounds.maxX,
        minV: globalBounds.minY,
        maxV: globalBounds.maxY,
      };
  const valid = p => p && finite(p[hIdx]) && finite(p[vIdx]);
  if (checks.followDrone.checked && frame && valid(frame.pos)) {
    const spanH = side ? 70 : 70 * w / Math.max(h, 1);
    const spanV = side ? 35 : 70;
    b = {
      minH: frame.pos[hIdx] - spanH / 2,
      maxH: frame.pos[hIdx] + spanH / 2,
      minV: frame.pos[vIdx] - spanV / 2,
      maxV: frame.pos[vIdx] + spanV / 2,
    };
  }
  const pad = 42;
  const sx = (w - 2 * pad) / Math.max(1e-6, b.maxH - b.minH);
  const sy = (h - 2 * pad) / Math.max(1e-6, b.maxV - b.minV);
  const s = Math.min(sx, sy);
  return {
    b,
    s,
    mode,
    hIdx,
    vIdx,
    labelH,
    labelV,
    valid,
    x: p => pad + (p[hIdx] - b.minH) * s,
    y: p => h - pad - (p[vIdx] - b.minV) * s,
  };
}

function drawLine(ctx, pts, proj, color, width=2, alpha=1) {
  const filtered = pts.filter(p => proj.valid(p));
  if (filtered.length < 2) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(proj.x(filtered[0]), proj.y(filtered[0]));
  for (const p of filtered.slice(1)) ctx.lineTo(proj.x(p), proj.y(p));
  ctx.stroke();
  ctx.restore();
}

function drawCircle(ctx, proj, p, radius, color, fill=true) {
  if (!proj.valid(p)) return;
  ctx.beginPath();
  ctx.arc(proj.x(p), proj.y(p), radius, 0, Math.PI * 2);
  if (fill) {
    ctx.fillStyle = color;
    ctx.fill();
  } else {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

function drawDiamond(ctx, proj, p, radius, color) {
  if (!proj.valid(p)) return;
  const x = proj.x(p), y = proj.y(p);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(x, y - radius);
  ctx.lineTo(x + radius, y);
  ctx.lineTo(x, y + radius);
  ctx.lineTo(x - radius, y);
  ctx.closePath();
  ctx.fill();
}

function drawLabel(ctx, proj, p, text, color, dx=7, dy=-7) {
  if (!checks.showLabels.checked || !proj.valid(p)) return;
  ctx.fillStyle = color;
  ctx.font = '12px ui-monospace, monospace';
  ctx.fillText(text, proj.x(p) + dx, proj.y(p) + dy);
}

function drawGrid(ctx, proj, w, h) {
  ctx.fillStyle = '#101820';
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = '#28313a';
  ctx.lineWidth = 1;
  const stepH = 10;
  const stepV = proj.mode === 'yz' ? 5 : 10;
  const minH = Math.floor(proj.b.minH / stepH) * stepH;
  const maxH = Math.ceil(proj.b.maxH / stepH) * stepH;
  const minV = Math.floor(proj.b.minV / stepV) * stepV;
  const maxV = Math.ceil(proj.b.maxV / stepV) * stepV;
  ctx.beginPath();
  for (let axisH = minH; axisH <= maxH; axisH += stepH) {
    const p = [0, 0, 0];
    p[proj.hIdx] = axisH;
    const px = proj.x(p);
    ctx.moveTo(px, 0);
    ctx.lineTo(px, h);
  }
  for (let axisV = minV; axisV <= maxV; axisV += stepV) {
    const p = [0, 0, 0];
    p[proj.vIdx] = axisV;
    const py = proj.y(p);
    ctx.moveTo(0, py);
    ctx.lineTo(w, py);
  }
  ctx.stroke();
  ctx.fillStyle = '#73818d';
  ctx.font = '12px ui-monospace, monospace';
  ctx.fillText(proj.labelH, w - 24, h - 14);
  ctx.fillText(proj.labelV, 14, 22);
  ctx.fillText(proj.mode === 'yz' ? 'side y/z' : 'top-down x/y', 14, h - 14);
}

function drawDrone(ctx, proj, frame) {
  const p = frame.pos || frame.truth;
  if (!proj.valid(p)) return;
  let angle = -Math.PI / 2;
  const prev = frames[Math.max(0, frameIndex - 1)];
  if (prev && proj.valid(prev.pos)) {
    const dh = p[proj.hIdx] - prev.pos[proj.hIdx];
    const dv = p[proj.vIdx] - prev.pos[proj.vIdx];
    if (Math.hypot(dh, dv) > 0.02) angle = Math.atan2(-dv, dh);
  }
  const x = proj.x(p), y = proj.y(p);
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  ctx.fillStyle = '#63c7ff';
  ctx.strokeStyle = '#061018';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(10, 0);
  ctx.lineTo(-8, -6);
  ctx.lineTo(-4, 0);
  ctx.lineTo(-8, 6);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function makeProjector3D(w, h, frame) {
  const rangeX = globalBounds.maxX - globalBounds.minX;
  const rangeY = globalBounds.maxY - globalBounds.minY;
  const rangeZ = globalBounds.maxZ - globalBounds.minZ;
  const range = Math.max(rangeX, rangeY, rangeZ * 3, 20);
  if (!orbit.distance) orbit.distance = range * 1.35;
  const target = checks.followDrone.checked && p3(frame.pos)
    ? frame.pos
    : [
        (globalBounds.minX + globalBounds.maxX) / 2,
        (globalBounds.minY + globalBounds.maxY) / 2,
        (globalBounds.minZ + globalBounds.maxZ) / 2,
      ];
  const pitch = Math.max(-1.25, Math.min(1.25, orbit.pitch));
  const yaw = orbit.yaw;
  const dist = Math.max(5, orbit.distance);
  const offset = [
    Math.sin(yaw) * Math.cos(pitch) * dist,
    -Math.cos(yaw) * Math.cos(pitch) * dist,
    Math.sin(pitch) * dist,
  ];
  const cam = add3(target, offset);
  const forward = norm3(sub3(target, cam), [0, 1, 0]);
  // Match onboard camera imagery: screen-right follows MAVLink/OpenCV
  // body-right, not the opposite-handed external orbit convention.
  let right = norm3(cross3([0, 0, 1], forward), [1, 0, 0]);
  if (len3(right) < 1e-6) right = [1, 0, 0];
  const up = norm3(cross3(forward, right), [0, 0, 1]);
  const focal = Math.min(w, h) * 0.92;
  return {
    cam,
    target,
    forward,
    right,
    up,
    dist,
    project(p) {
      if (!p3(p)) return null;
      const rel = sub3(p, cam);
      const depth = dot3(rel, forward);
      if (depth <= 0.15) return null;
      return {
        x: w / 2 + dot3(rel, right) * focal / depth,
        y: h / 2 - dot3(rel, up) * focal / depth,
        depth,
      };
    },
  };
}

function draw3DLine(ctx, proj, pts, color, width=2, alpha=1) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  let active = false;
  let drew = false;
  for (const p of pts || []) {
    const q = proj.project(p);
    if (!q) {
      active = false;
      continue;
    }
    if (!active) {
      ctx.moveTo(q.x, q.y);
      active = true;
    } else {
      ctx.lineTo(q.x, q.y);
      drew = true;
    }
  }
  if (drew) ctx.stroke();
  ctx.restore();
}

function draw3DMarker(ctx, proj, p, radius, color, fill=true, alpha=1) {
  const q = proj.project(p);
  if (!q) return;
  const scale = Math.max(0.55, Math.min(1.5, 40 / q.depth));
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  ctx.arc(q.x, q.y, radius * scale, 0, Math.PI * 2);
  if (fill) {
    ctx.fillStyle = color;
    ctx.fill();
  } else {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
  }
  ctx.restore();
}

function draw3DLabel(ctx, proj, p, text, color, dx=7, dy=-7) {
  if (!checks.showLabels.checked) return;
  const q = proj.project(p);
  if (!q) return;
  ctx.fillStyle = color;
  ctx.font = '12px ui-monospace, monospace';
  ctx.fillText(text, q.x + dx, q.y + dy);
}

function draw3DFloor(ctx, proj, w, h) {
  ctx.fillStyle = '#101820';
  ctx.fillRect(0, 0, w, h);
  const step = 10;
  const x0 = Math.floor(globalBounds.minX / step) * step;
  const x1 = Math.ceil(globalBounds.maxX / step) * step;
  const y0 = Math.floor(globalBounds.minY / step) * step;
  const y1 = Math.ceil(globalBounds.maxY / step) * step;
  ctx.strokeStyle = '#28313a';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = x0; x <= x1; x += step) {
    const a = proj.project([x, y0, 0]);
    const b = proj.project([x, y1, 0]);
    if (a && b) {
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
    }
  }
  for (let y = y0; y <= y1; y += step) {
    const a = proj.project([x0, y, 0]);
    const b = proj.project([x1, y, 0]);
    if (a && b) {
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
    }
  }
  ctx.stroke();
  draw3DLine(ctx, proj, [[0, 0, 0], [5, 0, 0]], '#ff8a8a', 2, 0.9);
  draw3DLine(ctx, proj, [[0, 0, 0], [0, 5, 0]], '#8affbf', 2, 0.9);
  draw3DLine(ctx, proj, [[0, 0, 0], [0, 0, 5]], '#8cc8ff', 2, 0.9);
  draw3DLabel(ctx, proj, [5, 0, 0], 'x', '#ff8a8a');
  draw3DLabel(ctx, proj, [0, 5, 0], 'y', '#8affbf');
  draw3DLabel(ctx, proj, [0, 0, 5], 'z', '#8cc8ff');
  ctx.fillStyle = '#73818d';
  ctx.font = '12px ui-monospace, monospace';
  ctx.fillText('3D orbit: drag rotate, wheel zoom', 14, h - 14);
}

function gateNormalFromNeighbors(g, gates) {
  const idx = Math.max(0, gates.indexOf(g));
  const prev = idx > 0 ? gates[idx - 1] : null;
  const next = idx < gates.length - 1 ? gates[idx + 1] : null;
  if (next && p3(next.p) && p3(g.p)) return norm3(sub3(next.p, g.p), [0, 1, 0]);
  if (prev && p3(prev.p) && p3(g.p)) return norm3(sub3(g.p, prev.p), [0, 1, 0]);
  return [0, 1, 0];
}

function planNormalForGate(g, t) {
  let best = null;
  for (const plan of DATA.plans || []) {
    if (plan.t > t) break;
    if (!p3(plan.normal)) continue;
    const gateIds = [String(g.id), String(g.order)];
    const planGateIds = [
      String(plan.gate_idx),
      ...(plan.horizon_gate_indices || []).map(v => String(v)),
    ];
    const idMatch = gateIds.some(id => planGateIds.includes(id));
    const targetMatch = dist3(plan.target, g.p) < 3.0;
    const waypointMatch = (plan.waypoints || []).some(w => dist3(w, g.p) < 2.0);
    if (idMatch || targetMatch || waypointMatch) best = plan.normal;
  }
  return best;
}

function gateAxes(normal) {
  const n = norm3(normal, [0, 1, 0]);
  let up = sub3([0, 0, 1], mul3(n, dot3([0, 0, 1], n)));
  up = norm3(up, [0, 0, 1]);
  let right = norm3(cross3(up, n), [1, 0, 0]);
  if (len3(right) < 1e-6) right = [1, 0, 0];
  return { normal: n, up, right };
}

function gateAxesForGate(g, fallbackNormal) {
  const fallback = gateAxes(fallbackNormal);
  const normal = p3(g.normal_neu) ? norm3(g.normal_neu, fallback.normal) : fallback.normal;
  const up = p3(g.up_axis_neu) ? norm3(g.up_axis_neu, fallback.up) : fallback.up;
  let right = p3(g.right_axis_neu) ? norm3(g.right_axis_neu, fallback.right) : null;
  if (!right || len3(right) < 1e-6) right = norm3(cross3(up, normal), fallback.right);
  return { normal, up, right };
}

function draw3DGate(ctx, proj, g, gates, t, color, label, alpha=0.13) {
  if (!p3(g.p)) return;
  const normal = planNormalForGate(g, t) || gateNormalFromNeighbors(g, gates);
  const axes = gateAxesForGate(g, normal);
  const half = Math.max(0.1, gateSizeM / 2);
  const c = g.p;
  const corners = [
    add3(add3(c, mul3(axes.right, -half)), mul3(axes.up, half)),
    add3(add3(c, mul3(axes.right, half)), mul3(axes.up, half)),
    add3(add3(c, mul3(axes.right, half)), mul3(axes.up, -half)),
    add3(add3(c, mul3(axes.right, -half)), mul3(axes.up, -half)),
  ];
  const qs = corners.map(p => proj.project(p));
  if (qs.some(q => !q)) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(qs[0].x, qs[0].y);
  for (const q of qs.slice(1)) ctx.lineTo(q.x, q.y);
  ctx.closePath();
  ctx.fill();
  ctx.globalAlpha = 0.95;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.restore();
  draw3DLine(ctx, proj, [c, add3(c, mul3(axes.normal, 1.25))], color, 1.5, 0.55);
  draw3DLabel(ctx, proj, c, label, color);
}

function canonicalGates(t) {
  const canonical = latestAt(DATA.canonical_gate_poses || [], t);
  if (canonical && canonical.gates && canonical.gates.length) return canonical.gates;
  const gt = latestAt(DATA.maps, t, m => m.kind === 'ground_truth_gates');
  if (gt && gt.gates && gt.gates.length) return gt.gates;
  if (DATA.known_gates && DATA.known_gates.length) return DATA.known_gates;
  const race = latestAt(DATA.maps, t, m => m.kind === 'race_order_gates');
  if (race && race.gates && race.gates.length) return race.gates;
  const stable = latestAt(DATA.maps, t, m => m.kind === 'stable_perception_gates');
  return stable && stable.gates ? stable.gates : [];
}

function droneHeading3D(frame) {
  const p = frame.pos || frame.truth;
  if (frame.cmd_deg && finite(frame.cmd_deg[2])) {
    const yawRad = frame.cmd_deg[2] * Math.PI / 180;
    return norm3([Math.cos(yawRad), Math.sin(yawRad), 0], [0, 1, 0]);
  }
  if (p3(p) && p3(frame.yaw_target) && dist3(frame.yaw_target, p) > 0.2) {
    return norm3(sub3(frame.yaw_target, p), [0, 1, 0]);
  }
  if (p3(frame.v_ref) && len3(frame.v_ref) > 0.05) return norm3(frame.v_ref, [0, 1, 0]);
  const prev = frames[Math.max(0, frameIndex - 1)];
  if (prev && p3(prev.pos) && p3(p) && dist3(p, prev.pos) > 0.03) {
    return norm3(sub3(p, prev.pos), [0, 1, 0]);
  }
  return [0, 1, 0];
}

function draw3DDrone(ctx, proj, frame) {
  const p = frame.pos || frame.truth;
  if (!p3(p)) return;
  const forward = droneHeading3D(frame);
  let right = norm3(cross3([0, 0, 1], forward), [1, 0, 0]);
  if (len3(right) < 1e-6) right = [1, 0, 0];
  const nose = add3(p, mul3(forward, 0.9));
  const tail = add3(p, mul3(forward, -0.45));
  const left = add3(tail, mul3(right, -0.35));
  const rightPt = add3(tail, mul3(right, 0.35));
  const top = add3(p, [0, 0, 0.35]);
  const nq = proj.project(nose);
  const lq = proj.project(left);
  const rq = proj.project(rightPt);
  const tq = proj.project(top);
  if (!nq || !lq || !rq) return;
  ctx.save();
  ctx.fillStyle = '#63c7ff';
  ctx.strokeStyle = '#061018';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(nq.x, nq.y);
  ctx.lineTo(lq.x, lq.y);
  ctx.lineTo(rq.x, rq.y);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  if (tq) {
    ctx.strokeStyle = '#a7f3ff';
    ctx.beginPath();
    ctx.moveTo(nq.x, nq.y);
    ctx.lineTo(tq.x, tq.y);
    ctx.lineTo(lq.x, lq.y);
    ctx.moveTo(tq.x, tq.y);
    ctx.lineTo(rq.x, rq.y);
    ctx.stroke();
  }
  ctx.restore();
  draw3DLabel(ctx, proj, p, 'drone', '#a7f3ff', 9, 12);
}

function drawMap3D() {
  if (!frames.length) return;
  frameIndex = Math.max(0, Math.min(frames.length - 1, frameIndex));
  const frame = frames[frameIndex];
  const {ctx, w, h} = resizeCanvas(mapCanvas);
  const proj = makeProjector3D(w, h, frame);
  draw3DFloor(ctx, proj, w, h);

  const t = frame.t;
  const trail = frames.slice(0, frameIndex + 1);
  const baseGates = canonicalGates(t);
  const stable = latestAt(DATA.maps, t, m => m.kind === 'stable_perception_gates');
  const race = latestAt(DATA.maps, t, m => m.kind === 'race_order_gates');

  if (checks.showGt.checked) {
    for (const g of baseGates) draw3DGate(ctx, proj, g, baseGates, t, '#d6dde5', `gt${g.id}`, 0.08);
  }
  if (checks.showStable.checked && stable) {
    for (const g of stable.gates) draw3DGate(ctx, proj, g, stable.gates, t, '#ffd166', `s${g.id}`, 0.11);
  }
  if (checks.showRace.checked && race) {
    for (const g of race.gates) draw3DGate(ctx, proj, g, race.gates, t, '#47d18c', `${g.order}:${g.id}`, 0.15);
  }

  draw3DLine(ctx, proj, trail.map(f => f.pos), '#63c7ff', 2.5, 0.95);
  if (checks.showTruth.checked) draw3DLine(ctx, proj, trail.map(f => f.truth), '#60d394', 2, 0.85);
  if (checks.showRef.checked) draw3DLine(ctx, proj, trail.map(f => f.p_ref), '#ffb454', 2, 0.9);

  const rejectedPlans = rejectedPlansForTime(t);
  if (checks.showRejectedPlans.checked) {
    for (const rejected of rejectedPlans) {
      const rejectedCurve = planCurvePoints(rejected);
      if (rejectedCurve.length) draw3DLine(ctx, proj, rejectedCurve, '#ff8a4c', 1.5, 0.38);
    }
    const latestRejected = rejectedPlans[rejectedPlans.length - 1];
    if (latestRejected) {
      const labelPoint = latestRejected.target || planCurvePoints(latestRejected).slice(-1)[0];
      draw3DLabel(ctx, proj, labelPoint, `reject ${latestRejected.reason || 'plan'}`, '#ffb282', 8, 14);
    }
  }

  const plan = latestAt(DATA.plans, t);
  const planCurve = planCurvePoints(plan);
  if (checks.showPlan.checked && plan && planCurve.length) {
    draw3DLine(ctx, proj, planCurve, '#d38cff', 3, 0.9);
    const planWaypoints = plan.waypoints || [];
    for (let i = 0; i < planWaypoints.length; i++) {
      draw3DMarker(ctx, proj, planWaypoints[i], 5, '#d38cff', true);
      draw3DLabel(ctx, proj, planWaypoints[i], `w${i}`, '#e4c3ff');
    }
  }

  for (const gp of DATA.passes || []) {
    if (gp.t > t) break;
    const p = gp.pos || gp.target;
    draw3DMarker(ctx, proj, p, 7, '#ff6b6b', false);
    draw3DLabel(ctx, proj, p, `pass g${gp.gate_idx}`, '#ff8a8a', 8, 14);
  }
  draw3DMarker(ctx, proj, frame.target, 7, '#ffffff', false);
  draw3DLabel(ctx, proj, frame.target, 'target', '#ffffff');
  draw3DMarker(ctx, proj, frame.yaw_target, 4, '#a7f3ff', false);
  draw3DLabel(ctx, proj, frame.yaw_target, 'yaw', '#a7f3ff');
  draw3DDrone(ctx, proj, frame);
}

function drawMap() {
  if (!frames.length) return;
  mapCanvas.style.cursor = viewMode.value === '3d' ? (orbit.dragging ? 'grabbing' : 'grab') : 'default';
  if (viewMode.value === '3d') {
    drawMap3D();
    return;
  }
  frameIndex = Math.max(0, Math.min(frames.length - 1, frameIndex));
  const frame = frames[frameIndex];
  const {ctx, w, h} = resizeCanvas(mapCanvas);
  const proj = makeProjector(w, h, frame);
  drawGrid(ctx, proj, w, h);

  const t = frame.t;
  const trail = frames.slice(0, frameIndex + 1);
  drawLine(ctx, trail.map(f => f.pos), proj, '#63c7ff', 2.5, 0.95);
  if (checks.showTruth.checked) drawLine(ctx, trail.map(f => f.truth), proj, '#60d394', 2, 0.85);
  if (checks.showRef.checked) drawLine(ctx, trail.map(f => f.p_ref), proj, '#ffb454', 2, 0.9);

  const canonical = latestAt(DATA.canonical_gate_poses || [], t);
  const gt = (
    canonical
    || latestAt(DATA.maps, t, m => m.kind === 'ground_truth_gates')
    || ((DATA.known_gates && DATA.known_gates.length) ? {gates: DATA.known_gates} : null)
  );
  const stable = latestAt(DATA.maps, t, m => m.kind === 'stable_perception_gates');
  const race = latestAt(DATA.maps, t, m => m.kind === 'race_order_gates');

  if (checks.showGt.checked && gt) {
    for (const g of gt.gates) {
      drawCircle(ctx, proj, g.p, 6, '#d6dde5', false);
      drawLabel(ctx, proj, g.p, `gt${g.id}`, '#d6dde5');
    }
  }
  if (checks.showStable.checked && stable) {
    for (const g of stable.gates) {
      drawCircle(ctx, proj, g.p, 4, '#ffd166', true);
      drawLabel(ctx, proj, g.p, `s${g.id}`, '#ffd166');
    }
  }
  if (checks.showRace.checked && race) {
    for (const g of race.gates) {
      drawDiamond(ctx, proj, g.p, 6, '#47d18c');
      drawLabel(ctx, proj, g.p, `${g.order}:${g.id}`, '#47d18c');
    }
  }

  const rejectedPlans = rejectedPlansForTime(t);
  if (checks.showRejectedPlans.checked) {
    for (const rejected of rejectedPlans) {
      const rejectedCurve = planCurvePoints(rejected);
      if (rejectedCurve.length) drawLine(ctx, rejectedCurve, proj, '#ff8a4c', 1.5, 0.38);
    }
    const latestRejected = rejectedPlans[rejectedPlans.length - 1];
    if (latestRejected) {
      const labelPoint = latestRejected.target || planCurvePoints(latestRejected).slice(-1)[0];
      drawLabel(ctx, proj, labelPoint, `reject ${latestRejected.reason || 'plan'}`, '#ffb282', 8, 14);
    }
  }

  const plan = latestAt(DATA.plans, t);
  const planCurve = planCurvePoints(plan);
  if (checks.showPlan.checked && plan && planCurve.length) {
    drawLine(ctx, planCurve, proj, '#d38cff', 3, 0.9);
    const planWaypoints = plan.waypoints || [];
    for (let i = 0; i < planWaypoints.length; i++) {
      drawCircle(ctx, proj, planWaypoints[i], 5, '#d38cff', true);
      drawLabel(ctx, proj, planWaypoints[i], `w${i}`, '#e4c3ff');
    }
  }

  for (const gp of DATA.passes || []) {
    if (gp.t > t) break;
    const p = gp.pos || gp.target;
    if (!proj.valid(p)) continue;
    drawCircle(ctx, proj, p, 7, '#ff6b6b', false);
    drawLabel(ctx, proj, p, `pass g${gp.gate_idx}`, '#ff8a8a', 8, 14);
  }

  if (proj.valid(frame.target)) {
    drawCircle(ctx, proj, frame.target, 7, '#ffffff', false);
    drawLabel(ctx, proj, frame.target, 'target', '#ffffff');
  }
  if (proj.valid(frame.yaw_target)) {
    drawCircle(ctx, proj, frame.yaw_target, 4, '#a7f3ff', false);
    drawLabel(ctx, proj, frame.yaw_target, 'yaw', '#a7f3ff');
  }
  drawDrone(ctx, proj, frame);
}

function drawAlt() {
  if (!frames.length) return;
  const {ctx, w, h} = resizeCanvas(altCanvas);
  ctx.fillStyle = '#111920';
  ctx.fillRect(0, 0, w, h);
  const padL = 48, padR = 14, padT = 16, padB = 28;
  const tMax = Math.max(1, DATA.duration || frames[frames.length - 1].t || 1);
  const zMin = globalBounds.minZ, zMax = globalBounds.maxZ;
  const sx = t => padL + (t / tMax) * (w - padL - padR);
  const sy = z => h - padB - ((z - zMin) / Math.max(1e-6, zMax - zMin)) * (h - padT - padB);
  ctx.strokeStyle = '#28313a';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let z = Math.ceil(zMin / 5) * 5; z <= zMax; z += 5) {
    ctx.moveTo(padL, sy(z));
    ctx.lineTo(w - padR, sy(z));
  }
  ctx.stroke();
  function plot(key, color) {
    const pts = frames.filter(f => pz(f[key]));
    if (pts.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(sx(pts[0].t), sy(pts[0][key][2]));
    for (const f of pts.slice(1)) ctx.lineTo(sx(f.t), sy(f[key][2]));
    ctx.stroke();
  }
  if (checks.showTruth.checked) plot('truth', '#60d394');
  plot('pos', '#63c7ff');
  if (checks.showRef.checked) plot('p_ref', '#ffb454');
  const frame = frames[frameIndex];
  const x = sx(frame.t);
  ctx.strokeStyle = '#ffffff';
  ctx.beginPath();
  ctx.moveTo(x, padT);
  ctx.lineTo(x, h - padB);
  ctx.stroke();
  ctx.fillStyle = '#9aa7b3';
  ctx.font = '12px ui-monospace, monospace';
  ctx.fillText('z (m)', 8, 18);
  ctx.fillText(`${fmt(zMin,1)}`, 8, h - padB);
  ctx.fillText(`${fmt(zMax,1)}`, 8, padT + 4);
}

function updateInfo() {
  if (!frames.length) return;
  const f = frames[frameIndex];
  const t = f.t || 0;
  timeLabel.textContent = `${fmt(t, 1)}s / frame ${frameIndex + 1}/${frames.length}`;
  const plan = latestAt(DATA.plans, t);
  const planCurve = planCurvePoints(plan);
  const rejectedPlans = rejectedPlansForTime(t);
  const race = latestAt(DATA.maps, t, m => m.kind === 'race_order_gates');
  const stable = latestAt(DATA.maps, t, m => m.kind === 'stable_perception_gates');
  const cameraFrame = cameraFrameForTime(t);
  const viewText = viewMode.value === '3d'
    ? '3D orbit'
    : (viewMode.value === 'yz' ? 'side y/z' : 'top-down x/y');
  const rows = [
    ['t', `${fmt(t, 2)} s`],
    ['view', viewText],
    ['gate_idx', f.gate_idx ?? 'n/a'],
    ['active_track', f.active_track ?? 'n/a'],
    ['target_event', f.target_event ?? 'n/a'],
    ['dist', fmt(f.dist)],
    ['pos', fmtVec(f.pos)],
    ['truth', fmtVec(f.truth)],
    ['target', fmtVec(f.target)],
    ['yaw_source', f.yaw_source ?? 'n/a'],
    ['cmd_deg', fmtVec(f.cmd_deg)],
    ['plan', plan ? `${plan.mode} g=${plan.gate_idx} tracks=(${(plan.horizon_tracks||[]).join(',')})` : 'none'],
    ['plan samples', plan ? `${planCurve.length} curve / ${(plan.waypoints || []).length} waypoints` : 'none'],
    ['failed plans', `${rejectedPlans.length} recent / ${DATA.counts.rejected_plans || 0} total`],
    ['camera frame', cameraFrame ? `${cameraFrame.frame_id ?? 'n/a'} @ ${fmt(cameraFrame.t, 2)}s` : 'none'],
    ['race gates', race ? race.gates.length : 0],
    ['stable gates', stable ? stable.gates.length : 0],
  ];
  frameInfo.innerHTML = rows.map(([k, v]) => `<div>${k}</div><div>${v}</div>`).join('');

  const ev = [
    ...recentAt(DATA.plans, t, 5).map(p => `[${fmt(p.t,1)}] plan ${p.mode} gate=${p.gate_idx} tracks=(${(p.horizon_tracks||[]).join(',')})`),
    ...recentAt(DATA.rejected_plans || [], t, 5).map(p => `[${fmt(p.t,1)}] reject ${p.mode} gate=${p.gate_idx} reason=${p.reason ?? 'n/a'} fallback=${p.fallback ?? 'n/a'}`),
    ...recentAt(DATA.shifts, t, 5).map(s => `[${fmt(s.t,1)}] shift gate=${s.gate_idx} track=${s.track} shift=${fmt(s.shift)}`),
    ...recentAt(DATA.passes, t, 5).map(g => `[${fmt(g.t,1)}] pass gate=${g.gate_idx} track=${g.track} d=${fmt(g.distance)} lat=${fmt(g.lateral_error)}`),
    ...recentAt(DATA.alerts, t, 3).map(a => `[${fmt(a.t,1)}] alert ${a.text}`),
  ].sort((a, b) => {
    const ta = Number((a.match(/\[([0-9.]+)\]/) || [0,0])[1]);
    const tb = Number((b.match(/\[([0-9.]+)\]/) || [0,0])[1]);
    return ta - tb;
  }).slice(-12);
  eventsBox.textContent = ev.join('\n') || 'No recent structured events.';
}

function draw() {
  drawMap();
  drawAlt();
  drawCamera();
  updateInfo();
  slider.value = String(frameIndex);
}

function tick(now) {
  if (playing && frames.length > 1) {
    const dt = (now - lastTick) / 1000;
    const speed = Number(speedSel.value || 1);
    playTime += dt * speed;
    frameIndex = frameIndexForTime(playTime);
    if (frameIndex >= frames.length - 1) {
      frameIndex = frames.length - 1;
      playTime = Number(frames[frameIndex].t || playTime);
      playing = false;
      playBtn.textContent = 'Play';
    }
    draw();
  }
  lastTick = now;
  requestAnimationFrame(tick);
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function mediaRecorderMimeType() {
  const candidates = [
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm',
  ];
  for (const item of candidates) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(item)) return item;
  }
  return '';
}

function composeRecordFrame(ctx, t) {
  const w = recordCanvas.width;
  const h = recordCanvas.height;
  ctx.fillStyle = '#101418';
  ctx.fillRect(0, 0, w, h);

  const leftW = 1280;
  const mapH = 840;
  const altH = h - mapH;
  const rightX = leftW;
  const rightW = w - leftW;

  ctx.drawImage(mapCanvas, 0, 0, leftW, mapH);
  ctx.drawImage(altCanvas, 0, mapH, leftW, altH);

  ctx.fillStyle = '#080c10';
  ctx.fillRect(rightX, 0, rightW, h);
  ctx.drawImage(cameraCanvas, rightX, 0, rightW, 360);

  const f = frames[frameIndex] || {};
  const cam = cameraFrameForTime(t);
  ctx.fillStyle = '#eef4f8';
  ctx.font = '24px ui-monospace, monospace';
  ctx.fillText(`AIGP replay ${DATA.run_id}`, rightX + 24, 420);
  ctx.font = '20px ui-monospace, monospace';
  ctx.fillText(`t ${fmt(t, 2)} s`, rightX + 24, 462);
  ctx.fillText(`gate ${f.gate_idx ?? 'n/a'} track ${f.active_track ?? 'n/a'}`, rightX + 24, 500);
  ctx.fillText(`pos ${fmtVec(f.pos)}`, rightX + 24, 538);
  ctx.fillText(`target ${fmtVec(f.target)}`, rightX + 24, 576);
  ctx.fillText(`camera ${cam ? `${cam.frame_id ?? 'n/a'} @ ${fmt(cam.t, 2)}s` : 'none'}`, rightX + 24, 614);

  ctx.fillStyle = '#9aa7b3';
  ctx.font = '16px ui-monospace, monospace';
  ctx.fillText('3D orbit + follow drone, encoded at 30 fps real-time playback', rightX + 24, h - 36);
}

async function recordReplayVideo() {
  if (recording || !frames.length) return;
  if (!window.MediaRecorder || !recordCanvas.captureStream) {
    videoStatus.textContent = 'MediaRecorder/canvas capture is not supported in this browser.';
    return;
  }

  recording = true;
  saveVideoBtn.disabled = true;
  const previous = {
    frameIndex,
    playTime,
    playing,
    viewMode: viewMode.value,
    followDrone: checks.followDrone.checked,
  };

  playing = false;
  playBtn.textContent = 'Play';
  viewMode.value = '3d';
  checks.followDrone.checked = true;

  const fps = 30;
  const startT = Number(frames[0].t || 0);
  const endT = Math.max(startT, Number(DATA.duration || frames[frames.length - 1].t || startT));
  const totalFrames = Math.max(1, Math.ceil((endT - startT) * fps));
  recordCanvas.width = 1920;
  recordCanvas.height = 1080;
  const recordCtx = recordCanvas.getContext('2d');
  const mimeType = mediaRecorderMimeType();
  const stream = recordCanvas.captureStream(fps);
  const chunks = [];
  let recorder;

  try {
    recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  } catch (err) {
    videoStatus.textContent = `Could not start recorder: ${err}`;
    saveVideoBtn.disabled = false;
    recording = false;
    return;
  }

  const stopped = new Promise(resolve => {
    recorder.onstop = resolve;
  });
  recorder.ondataavailable = event => {
    if (event.data && event.data.size > 0) chunks.push(event.data);
  };

  try {
    recorder.start();
    for (let i = 0; i <= totalFrames; i++) {
      const t = Math.min(endT, startT + i / fps);
      frameIndex = frameIndexForTime(t);
      playTime = t;
      const cam = cameraFrameForTime(t);
      await loadCameraImage(cam);
      draw();
      composeRecordFrame(recordCtx, t);
      videoStatus.textContent = `recording ${i}/${totalFrames} frames (${fmt((i / Math.max(1, totalFrames)) * 100, 0)}%)`;
      await sleep(1000 / fps);
    }
  } finally {
    if (recorder.state !== 'inactive') recorder.stop();
    await stopped;
    for (const track of stream.getTracks()) track.stop();
  }

  const blob = new Blob(chunks, { type: mimeType || 'video/webm' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${DATA.run_id || 'aigp_replay'}_replay.webm`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);

  frameIndex = previous.frameIndex;
  playTime = previous.playTime;
  playing = previous.playing;
  viewMode.value = previous.viewMode;
  checks.followDrone.checked = previous.followDrone;
  playBtn.textContent = playing ? 'Pause' : 'Play';
  videoStatus.textContent = `saved ${link.download}`;
  saveVideoBtn.disabled = false;
  recording = false;
  draw();
}

mapCanvas.addEventListener('pointerdown', event => {
  if (viewMode.value !== '3d') return;
  orbit.dragging = true;
  orbit.lastX = event.clientX;
  orbit.lastY = event.clientY;
  mapCanvas.setPointerCapture(event.pointerId);
  draw();
});
mapCanvas.addEventListener('pointermove', event => {
  if (viewMode.value !== '3d' || !orbit.dragging) return;
  const dx = event.clientX - orbit.lastX;
  const dy = event.clientY - orbit.lastY;
  orbit.lastX = event.clientX;
  orbit.lastY = event.clientY;
  orbit.yaw += dx * 0.008;
  orbit.pitch = Math.max(-1.25, Math.min(1.25, orbit.pitch + dy * 0.006));
  draw();
});
mapCanvas.addEventListener('pointerup', event => {
  if (!orbit.dragging) return;
  orbit.dragging = false;
  try { mapCanvas.releasePointerCapture(event.pointerId); } catch (err) {}
  draw();
});
mapCanvas.addEventListener('pointercancel', () => {
  orbit.dragging = false;
  draw();
});
mapCanvas.addEventListener('wheel', event => {
  if (viewMode.value !== '3d') return;
  event.preventDefault();
  const rangeX = globalBounds.maxX - globalBounds.minX;
  const rangeY = globalBounds.maxY - globalBounds.minY;
  const rangeZ = globalBounds.maxZ - globalBounds.minZ;
  const base = Math.max(rangeX, rangeY, rangeZ * 3, 20);
  const cur = orbit.distance || base * 1.35;
  const next = cur * Math.exp(event.deltaY * 0.001);
  orbit.distance = Math.max(5, Math.min(base * 5, next));
  draw();
}, {passive: false});

playBtn.addEventListener('click', () => {
  if (frameIndex >= frames.length - 1) frameIndex = 0;
  playTime = Number(frames[frameIndex].t || 0);
  playing = !playing;
  playBtn.textContent = playing ? 'Pause' : 'Play';
  lastTick = performance.now();
});
saveVideoBtn.addEventListener('click', recordReplayVideo);
slider.addEventListener('input', () => {
  frameIndex = Number(slider.value);
  playTime = Number((frames[frameIndex] && frames[frameIndex].t) || 0);
  playing = false;
  playBtn.textContent = 'Play';
  draw();
});
viewMode.addEventListener('change', draw);
for (const el of Object.values(checks)) el.addEventListener('change', draw);
window.addEventListener('resize', draw);

if (!frames.length) {
  document.body.innerHTML = '<main><section class="panel side-section">No autonomy_trace frames found in this debug log.</section></main>';
} else {
  draw();
  requestAnimationFrame(tick);
}
</script>
</body>
</html>
"""


def _write_html(data: dict[str, Any], out_path: Path) -> None:
    encoded = json.dumps(data, separators=(",", ":"), allow_nan=False)
    encoded = encoded.replace("<", "\\u003c")
    html = (
        HTML_TEMPLATE.replace("__RUN_ID__", str(data["run_id"]))
        .replace("__DATA__", encoded)
    )
    out_path.write_text(html, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a self-contained HTML replay from an AIGP run debug.jsonl."
        )
    )
    parser.add_argument(
        "--run",
        default="latest",
        help=(
            "Run directory, run id under aigp/logs/runs, debug.jsonl path, or 'latest'. "
            "Defaults to latest."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output HTML path. Defaults to <run>/replay_debug_map.html.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=5000,
        help="Downsample autonomy_trace frames to this maximum. Use 0 to keep all.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        run_dir = _resolve_run(args.run)
        debug_path = run_dir if run_dir.is_file() else run_dir / "debug.jsonl"
        if not debug_path.exists():
            raise FileNotFoundError(f"debug log not found: {debug_path}")
        data = _load_debug(debug_path, max_frames=max(0, int(args.max_frames)))
        out_path = Path(args.out).expanduser() if args.out else debug_path.parent / "replay_debug_map.html"
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_html(data, out_path)
    except Exception as exc:
        print(f"replay_debug_map: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {out_path}")
    print(
        "summary: "
        f"frames={data['counts']['frames']} "
        f"plans={data['counts']['plans']} "
        f"passes={data['counts']['passes']} "
        f"maps={data['counts']['maps']} "
        f"duration={data['duration']:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
