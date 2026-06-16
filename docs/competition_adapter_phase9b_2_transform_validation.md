# Phase 9B.2 Official Transform Validation

Status: implemented as an opt-in competition transform path for receive-only real-perception dry-runs.

## Purpose

Phase 9B.2 fixes the Phase 9B transform boundary without changing legacy runner defaults. The tilted-camera PX4/Gazebo surrogate run showed that PnP still produced repeatable camera-frame `tvec` values, but the active legacy transform mode `physical_direct_rad_x_mirror` mapped the tilted camera's positive OpenCV camera `y` component into negative internal/world `z`, causing all detections to be rejected by `z_below_safe_min`.

## What Changed

- Added `official_camera_to_internal_body_flu_rotmat(...)` in `autonomy_core/core/frame_conventions.py`.
- Added an opt-in `competition_official_ned` transform mode in `autonomy_core/launch/autonomy_api6.py`.
- The new mode uses the official VADR camera tilt and a proper rotation with `det(R)=+1`.
- The existing legacy default remains `physical_direct_rad_x_mirror` for `px4_runner` and non-competition `AutonomyAPI` construction.
- The competition-safe AutonomyAPI profile now selects `competition_official_ned` for real-perception competition dry-runs.
- `competition_main.py` reports the selected perception transform mode and fails closed if Phase 9B real perception is requested with a non-official transform.

## Not Changed

- No PnP scoring changes.
- No YOLO threshold changes.
- No gate admission, GateMemory, race progression, planner, controller, yaw/thrust tuning, or logger schema changes.
- No live command publication.
- No `command_live` or `race` enablement.
- No surrogate-side compensation for competition camera math.

## Manual PX4/Gazebo Surrogate Re-Test

Run the same receive-only Phase 9B command. The official transform is selected by default when `--real-perception` uses the competition-safe profile:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m autonomy_core.runtime.competition_main \
  vision_dry_run \
  --live-transports \
  --evidence-label px4_gazebo_surrogate_phase9b_2_official_transform \
  --use-real-autonomy \
  --real-perception \
  --allow-legacy-yolo-default \
  --mavlink-endpoint udpin:0.0.0.0:14540 \
  --vision-bind-host 0.0.0.0 \
  --vision-port 5600 \
  --steps 1000 \
  --step-sleep-s 0.02
```

While it runs, bridge Gazebo camera frames:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m autonomy_core.runtime.surrogate_vision_bridge \
  --camera-topic /camera \
  --send-host 127.0.0.1 \
  --send-port 5600 \
  --frames 5 \
  --timeout-s 5
```

## Success Criteria

Phase 9B.2 passes as a receive-only transform validation when:

- Terminal output reports `perception_transform_mode == "competition_official_ned"`.
- Startup diagnostics for the active mode report `det(R_body_camera)=1.000`.
- `phase9b_perception_dry_run_satisfied == true`.
- `perception_update_calls > 0`.
- `vision_frames_completed > 0`.
- The near-gate selected world/internal `z` is not rejected with `z_below_safe_min`.
- If transform ground-truth diagnostics are available, selected transform error is best or within 10% of best and preferably less than 0.5 m for near gates.
- `command_publication_allowed == false`.
- `command_sent_count == 0`.
- `phase4b_satisfied == false`.
- `competition_readiness_claimed == false`.

## Remaining Questions Before Phase 9C

Phase 9B.2 validates the camera/body/world transform path only. Before Phase 9C command-candidate dry-run is allowed, the receive-only output should show stable, plausible gate memory behavior using the official transform. If yaw/image warnings remain, quantify them before using perception output for command candidates.
