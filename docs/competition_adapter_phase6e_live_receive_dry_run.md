# Phase 6E Live Receive Dry-Run

Status: implemented for receive-only dry-runs.

## Purpose

Phase 6E proves that the production competition receive path can ingest
MAVLink telemetry and VADR UDP vision packets through the same transport modules
intended for the real competition simulator:

```text
MAVLink UDP endpoint
  -> competition_mavlink_transport.py
  -> CompetitionRunner.step(...)
  -> competition_state_adapter.py

UDP 0.0.0.0:5600 VADR JPEG packets
  -> competition_vision_transport.py
  -> CompetitionRunner.step(...)
  -> competition_image_adapter.py
```

This phase does not enable commands, race mode, or competition readiness.

## What Was Added

`autonomy_core/runtime/competition_main.py` now reports explicit Phase 6E
criteria when `--live-transports` is used:

- `phase: "6E"`
- `evidence_label`
- `phase6e_receive_satisfied`
- `phase6e_perception_boundary_satisfied`
- `phase6e_satisfied`
- `phase6e_success_criteria`

`phase6e_receive_satisfied=true` means the production receive transports
processed enough telemetry and vision to satisfy the receive dry-run criteria.

`phase6e_perception_boundary_satisfied=true` only means an injected or explicitly
enabled autonomy object received at least one perception update with
`gazebo_pose=None` and `image_pose_snapshot=None`.

The default safe PX4/Gazebo run does not instantiate real `AutonomyAPI`, so it
can pass `phase6e_receive_satisfied` while
`phase6e_perception_boundary_satisfied=false`.

## Manual PX4/Gazebo Surrogate Test

Terminal 1, start the competition receive dry-run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m autonomy_core.runtime.competition_main \
  vision_dry_run \
  --live-transports \
  --evidence-label px4_gazebo_surrogate \
  --mavlink-endpoint udpin:0.0.0.0:14540 \
  --vision-bind-host 0.0.0.0 \
  --vision-port 5600 \
  --steps 1000 \
  --step-sleep-s 0.02
```

Terminal 2, while Terminal 1 is still running, bridge Gazebo camera frames:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m autonomy_core.runtime.surrogate_vision_bridge \
  --camera-topic /camera \
  --send-host 127.0.0.1 \
  --send-port 5600 \
  --frames 5 \
  --timeout-s 5
```

## Success Criteria

For the receive dry-run to pass:

- `phase == "6E"`
- `live_transports_requested == true`
- `evidence_label == "px4_gazebo_surrogate"` or another accurate label
- `telemetry_messages_processed > 0`
- `heartbeat_seen == true`
- `state_usable == true`
- `vision_packets_processed > 0`
- `vision_frames_completed > 0`
- `phase6e_receive_satisfied == true`
- `phase6e_satisfied == true`
- `command_publication_allowed == false`
- `command_sent_count == 0`
- `phase4b_satisfied == false`
- `competition_readiness_claimed == false`

Optional stronger criterion:

- `phase6e_perception_boundary_satisfied == true`

That optional criterion requires an injected fake autonomy in tests or an
explicit later real-autonomy dry-run. It is not required for the current safe
PX4/Gazebo transport receive check.

## Not Claimed

- Phase 4B real competition telemetry evidence is still blocked.
- Phase 9 real competition simulator dry-run is not started.
- Command readiness is not claimed.
- Race readiness is not claimed.
- Competition readiness is not claimed.
- No MAVLink commands, heartbeats, setpoints, attitude targets, position
  targets, actuator commands, arm/offboard/reset commands, or command
  publication are added.
