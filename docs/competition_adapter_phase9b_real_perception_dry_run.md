# Phase 9B Real Perception Dry-Run

Status: implemented as an explicit no-command dry-run mode.

## Purpose

Phase 9B runs the production competition receive path with the Phase 9A
competition-safe `AutonomyAPI` profile and real perception enabled.

This phase proves that decoded competition-style frames can reach
`AutonomyAPI.update_gate_memory_from_frame(...)` with:

- `gazebo_pose=None`
- `image_pose_snapshot=None`
- command publication disabled

## What Was Added

`autonomy_core/runtime/competition_main.py` now supports:

- `--real-perception`
- `--allow-legacy-yolo-default`
- `--perception-transform-mode` with the Phase 9B.2 default
  `competition_official_ned`

`--real-perception` requires:

- `vision_dry_run`
- `--use-real-autonomy`
- `--live-transports` unless tests inject fake components

For the current repository state, `--allow-legacy-yolo-default` is also required
for live real-perception dry-runs because `AutonomyAPI` still contains a
hardcoded legacy YOLO weights path. This is an explicit temporary
acknowledgment, not a competition-ready configuration pattern.

Phase 9B.2 selects `competition_official_ned` through the competition-safe
profile. Legacy `AutonomyAPI` and `px4_runner` defaults remain
`physical_direct_rad_x_mirror`.

## Manual PX4/Gazebo Surrogate Test

Terminal 1, start the competition receive/perception dry-run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m autonomy_core.runtime.competition_main \
  vision_dry_run \
  --live-transports \
  --evidence-label px4_gazebo_surrogate_phase9b \
  --use-real-autonomy \
  --real-perception \
  --allow-legacy-yolo-default \
  --mavlink-endpoint udpin:0.0.0.0:14540 \
  --vision-bind-host 0.0.0.0 \
  --vision-port 5600 \
  --steps 1000 \
  --step-sleep-s 0.02
```

Terminal 2, while Terminal 1 is running, bridge Gazebo camera frames:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m autonomy_core.runtime.surrogate_vision_bridge \
  --camera-topic /camera \
  --send-host 127.0.0.1 \
  --send-port 5600 \
  --frames 5 \
  --timeout-s 5
```

## Success Criteria

Phase 9B passes when Terminal 1 reports:

- `phase == "9B"`
- `mode == "vision_dry_run"`
- `live_transports_requested == true`
- `use_real_autonomy == true`
- `real_perception_requested == true`
- `perception_transform_mode == "competition_official_ned"`
- `phase6e_receive_satisfied == true`
- `phase6e_perception_boundary_satisfied == true`
- `phase9b_perception_dry_run_satisfied == true`
- `perception_update_calls > 0`
- `vision_packets_processed > 0`
- `vision_frames_completed > 0`
- `command_publication_allowed == false`
- `command_sent_count == 0`
- `phase4b_satisfied == false`
- `competition_readiness_claimed == false`

The run may still report no gate detections if the model does not recognize the
current Gazebo scene. That is a perception/model result, not a transport or
adapter failure, as long as perception updates are called without crashing.

For Phase 9B.2 transform validation, the near-gate world/internal `z` should no
longer be rejected with `z_below_safe_min` solely because of the selected
camera transform.

## Not Claimed

- No commands are sent.
- Phase 9C command candidate dry-run is not started.
- Phase 9D real competition simulator stages are not started.
- Phase 4B remains blocked pending real receive-only competition simulator
  telemetry evidence.
- Competition readiness, command readiness, and race readiness are not claimed.
