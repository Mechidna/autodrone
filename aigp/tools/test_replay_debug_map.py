from __future__ import annotations

import json
from pathlib import Path

from aigp.tools import replay_debug_map


def _write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in items),
        encoding="utf-8",
    )


def test_load_debug_replays_camera_startup_with_mavlink_truth(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    debug_path = run_dir / "debug.jsonl"
    _write_jsonl(
        debug_path,
        [
            {"event": "run_start", "wall_time": 100.0, "runtime_config": {}},
            {
                "event": "shadow_estimator_trace",
                "wall_time": 102.0,
                "fields": {
                    "pos_neu": "(99.0,98.0,97.0)",
                    "truth_pos_neu": "(1.0,2.0,3.0)",
                },
            },
            {
                "event": "hover_acquisition",
                "wall_time": 104.0,
                "fields": {"pos_neu": "(1.1,2.1,3.1)", "status": "settling"},
            },
            {
                "event": "autonomy_trace",
                "wall_time": 106.0,
                "fields": {
                    "pos_neu": "(2.0,3.0,4.0)",
                    "truth_pos_neu": "(2.0,3.0,4.0)",
                    "p_ref": "(2.5,3.5,4.5)",
                    "gate_idx": 0,
                },
            },
            {
                "event": "shadow_estimator_trace",
                "wall_time": 108.0,
                "fields": {
                    "pos_neu": "(199.0,198.0,197.0)",
                    "truth_pos_neu": "(3.0,4.0,5.0)",
                },
            },
            {"event": "run_end", "wall_time": 109.0, "returncode": 0},
        ],
    )
    _write_jsonl(
        run_dir / "camera_frames" / "index.jsonl",
        [
            {"wall_time": 100.5, "frame_id": 1, "path": "frame_000001.jpg"},
            {"wall_time": 103.0, "frame_id": 2, "path": "frame_000002.jpg"},
            {"wall_time": 107.0, "frame_id": 3, "path": "frame_000003.jpg"},
            {"wall_time": 109.0, "frame_id": 4, "path": "frame_000004.jpg"},
        ],
    )

    data = replay_debug_map._load_debug(debug_path, max_frames=0)

    assert data["counts"]["autonomy_frames"] == 1
    assert data["counts"]["state_frames"] == 4
    assert data["counts"]["camera_frames"] == 4
    assert data["counts"]["frames"] == 8
    assert data["duration"] == 9.0

    first = data["frames"][0]
    assert first["t"] == 0.5
    assert first["timeline_source"] == "camera"
    assert first["state_source"] == "mavlink_truth"
    assert first["pos"] == [1.0, 2.0, 3.0]
    assert first["truth"] == [1.0, 2.0, 3.0]

    during_flight = next(
        frame
        for frame in data["frames"]
        if frame["timeline_source"] == "camera" and frame["t"] == 7.0
    )
    assert during_flight["state_source"] == "autonomy_trace"
    assert during_flight["pos"] == [2.0, 3.0, 4.0]
    assert during_flight["p_ref"] == [2.5, 3.5, 4.5]

    after_flight = next(
        frame
        for frame in data["frames"]
        if frame["timeline_source"] == "camera" and frame["t"] == 9.0
    )
    assert after_flight["phase"] == "postflight"
    assert after_flight["state_source"] == "mavlink_truth"
    assert after_flight["pos"] == [3.0, 4.0, 5.0]
    assert [99.0, 98.0, 97.0] not in [
        frame["pos"] for frame in data["frames"]
    ]
    assert [199.0, 198.0, 197.0] not in [
        frame["pos"] for frame in data["frames"]
    ]


def test_load_debug_can_replay_camera_without_state_telemetry(tmp_path: Path) -> None:
    run_dir = tmp_path / "camera_only"
    debug_path = run_dir / "debug.jsonl"
    _write_jsonl(
        debug_path,
        [
            {"event": "run_start", "wall_time": 20.0, "runtime_config": {}},
            {"event": "run_end", "wall_time": 22.0, "returncode": 0},
        ],
    )
    _write_jsonl(
        run_dir / "camera_frames" / "index.jsonl",
        [{"wall_time": 20.25, "frame_id": 1, "path": "frame_000001.jpg"}],
    )

    data = replay_debug_map._load_debug(debug_path, max_frames=0)

    assert data["counts"]["state_frames"] == 0
    assert data["counts"]["frames"] == 1
    assert data["frames"][0]["t"] == 0.25
    assert data["frames"][0]["phase"] == "camera"
    assert data["frames"][0]["state_source"] == "camera_only"
    assert data["frames"][0]["pos"] is None
