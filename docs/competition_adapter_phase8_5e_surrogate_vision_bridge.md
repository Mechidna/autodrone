# Phase 8.5E Surrogate Vision Bridge

Status: implemented as PX4/Gazebo surrogate-only tooling.

## Purpose

`autonomy_core/runtime/surrogate_vision_bridge.py` lets the local PX4/Gazebo
camera mimic the competition simulator vision interface:

```text
ROS /camera pixels only
  -> optional explicit resize to 640x360
  -> JPEG encode
  -> VADR <IHHIIQ> packetization
  -> UDP send to 127.0.0.1:5600
```

This does not satisfy Phase 4B, Phase 9, telemetry readiness, command
readiness, race readiness, or competition readiness.

## Guardrails

- Importing the bridge does not initialize ROS, import `cv2`, open sockets, or
  instantiate `AutonomyAPI`.
- ROS, `cv_bridge`, `cv2`, and UDP sockets are only used when the bridge CLI is
  explicitly run.
- The bridge reads image pixels only. It rejects truth-like topics such as
  dynamic pose, TF, camera info, depth, ground truth, link state, or model state.
- The bridge sends no MAVLink, heartbeat, setpoint, attitude target, position
  target, actuator, arm, offboard, reset, or command messages.
- Production competition modules do not import this surrogate bridge.

## Current Gazebo Camera

The local Gazebo camera has been configured to publish `640x360`, matching the
official VADR resolution. That means the bridge can run without
`--resize-camera-to-competition` for the current setup.

## Manual Smoke Flow

Terminal 1: run the production competition receive dry-run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m autonomy_core.runtime.competition_main \
  vision_dry_run \
  --live-transports \
  --mavlink-endpoint udpin:0.0.0.0:14540 \
  --vision-bind-host 0.0.0.0 \
  --vision-port 5600 \
  --steps 100 \
  --step-sleep-s 0.02
```

Terminal 2: send ROS camera frames as VADR UDP vision packets:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m autonomy_core.runtime.surrogate_vision_bridge \
  --camera-topic /camera \
  --send-host 127.0.0.1 \
  --send-port 5600 \
  --frames 1 \
  --timeout-s 5
```

Expected competition-main dry-run evidence:

- `heartbeat_seen=true`
- `state_usable=true`, if PX4 is streaming usable estimated telemetry
- `vision_packets_processed>0`
- `vision_frames_completed>=1`
- `perception_update_calls>=1`
- `command_publication_allowed=false`
- `command_sent_count=0`
- `phase4b_satisfied=false`
- `competition_readiness_claimed=false`

## Tests

Deterministic tests are in `tests/test_surrogate_vision_bridge.py`.

They prove:

- bridge import safety
- exact VADR header packetization without ROS, `cv2`, or sockets
- packets produced by the bridge are accepted by `CompetitionImageAdapter`
- non-`640x360` frames are rejected unless explicit surrogate resize is enabled
- resize can be tested with an injected fake resizer, without importing `cv2`
- truth-like topics are rejected
- injected UDP sender behavior is deterministic
- no readiness or command-send claims are made
