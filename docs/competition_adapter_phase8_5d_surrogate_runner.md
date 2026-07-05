# Competition Adapter Phase 8.5D Surrogate Runner

Status: Phase 8.5D-1, Phase 8.5D-2, Phase 8.5D-3, and Phase 8.5D-3A code paths implemented. Live PX4/Gazebo evidence is still user-run/local evidence only and does not satisfy Phase 4B.

Date: 2026-06-13

## Scope

Phase 8.5D creates the path toward running the competition stack from
PX4/Gazebo surrogate inputs without extending the legacy `px4_runner.py` path.

The entrypoint is:

- `autonomy_core/runtime/surrogate_runner.py`

Tests are in:

- `tests/test_surrogate_runner_skeleton.py`

The runner is explicitly surrogate-only. It is for local confidence that the
competition adapters and `CompetitionRunner` can be exercised end-to-end with
PX4/Gazebo-like inputs. It does not claim competition telemetry readiness,
command readiness, race readiness, or Phase 4B completion.

## Implemented Stages

Phase 8.5D-1 implemented the import-safe runner skeleton:

- CLI modes and JSON summary contract.
- Surrogate-only labels and readiness non-claims.
- Fail-closed `px4_command_send`, `race`, and `competition_live` modes.

Phase 8.5D-2 implemented no-send image dry-runs:

- `mock_vision_dry_run`
- `saved_image_vision_dry_run`

These modes feed generated or saved `640x360` images through JPEG encoding,
VADR `<IHHIIQ` packetization, `CompetitionRunner.step(...)`,
`CompetitionImageAdapter`, and injected fake autonomy.

Phase 8.5D-3 implemented receive-only PX4 surrogate modes:

- `px4_observe`
- `px4_vision_dry_run`
- `px4_command_dry_run`

These modes can collect bounded receive-only MAVLink telemetry from an explicit
pymavlink endpoint such as `udpin:0.0.0.0:14540`, or use injected fake messages
in deterministic tests. They pass messages into `CompetitionRunner.step(...)`
without sending heartbeats, setpoints, attitude targets, actuator commands, arm,
offboard, reset, or any other MAVLink command.

Vision for PX4 surrogate modes can come from:

- `--vision-source generated_mock`
- `--vision-source saved_image --input-image PATH`
- `--vision-source ros_camera --camera-topic TOPIC`

Phase 8.5D-3A added explicit surrogate-only ROS camera resizing:

- Use `--resize-camera-to-competition` with `--vision-source ros_camera` when
  the ROS camera publishes a non-competition image size such as `1280x960`.
- The frame is resized to the official VADR `640x360` size before JPEG
  packetization.
- Without this flag, non-`640x360` ROS frames fail closed with a clear message.
- `CompetitionImageAdapter` remains strict and still requires official
  competition resolution.

The ROS camera path is lazy and optional. It imports ROS/cv_bridge only when
`ros_camera` is explicitly selected. It captures image pixels only and does not
use Gazebo model pose, Gazebo camera pose, TF, link state, world pose, depth
truth, gate truth, track truth, or pose snapshots.

## Mode Behavior

Implemented executable modes:

- `mock_vision_dry_run`: generated mock image through competition vision path.
- `saved_image_vision_dry_run`: saved image through competition vision path.
- `px4_observe`: receive-only MAVLink telemetry through `CompetitionRunner` observe mode.
- `px4_vision_dry_run`: receive-only MAVLink telemetry plus selected vision source through `CompetitionRunner` vision dry-run mode.
- `px4_command_dry_run`: receive-only MAVLink telemetry plus optional selected vision source through `CompetitionRunner` command dry-run mode; it may build a would-be command candidate but never sends it.

Fail-closed modes:

- `px4_command_send`
- `race`
- `competition_live`

## Safety Contract

Importing `surrogate_runner.py` does not:

- open sockets;
- initialize ROS;
- connect MAVSDK;
- import `pymavlink`, `mavsdk`, `rclpy`, or `cv2`;
- instantiate `AutonomyAPI`;
- publish commands.

Runtime socket use is explicit and only for PX4 surrogate receive modes when no
injected MAVLink messages are provided. The receive helper calls
`recv_match(blocking=False)` only. It does not call `wait_heartbeat()`,
`heartbeat_send()`, `mav.send()`, setpoint sends, command sends, arm, offboard,
or reset.

The CLI always reports:

- `surrogate_label = "PX4/Gazebo surrogate only"`
- `phase4b_satisfied = false`
- `competition_readiness_claimed = false`
- `command_publication_allowed = false`
- `command_sent_count = 0`

## Example Generated-Image Dry-Run

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m autonomy_core.runtime.surrogate_runner \
  mock_vision_dry_run \
  --source-kind generated_mock \
  --max-payload-size 4096
```

Expected status:

```text
status: competition_stack_vision_dry_run_complete
competition_runner_executed: true
vision_frames_completed: 1
perception_update_calls: 1
gazebo_pose: null
image_pose_snapshot: null
phase4b_satisfied: false
competition_readiness_claimed: false
command_sent_count: 0
```

## Example Saved-Image Dry-Run

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m autonomy_core.runtime.surrogate_runner \
  saved_image_vision_dry_run \
  --input-image /tmp/competition_vision_loopback_mock.jpg \
  --source-kind saved_image \
  --max-payload-size 4096
```

The input image must already be `640x360` unless `--resize-input-image` is
provided.

## Example PX4 Observe Surrogate

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m autonomy_core.runtime.surrogate_runner \
  px4_observe \
  --mavlink-endpoint udpin:0.0.0.0:14540 \
  --duration-s 5 \
  --max-messages 200
```

This should be run only when PX4/Gazebo is already streaming MAVLink to the
selected endpoint. It is receive-only and surrogate-only.

Useful fields to inspect:

- `telemetry_sample_count`
- `telemetry_message_types`
- `telemetry_summary.message_types`
- `heartbeat_seen`
- `state_usable`
- `position_source`
- `attitude_source`
- `command_sent_count`

## Example PX4 Vision Dry-Run With Generated Vision

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m autonomy_core.runtime.surrogate_runner \
  px4_vision_dry_run \
  --mavlink-endpoint udpin:0.0.0.0:14540 \
  --duration-s 5 \
  --max-messages 200 \
  --vision-source generated_mock \
  --max-payload-size 4096
```

This validates PX4 receive-only telemetry plus competition vision packetization
and perception boundary wiring. The image is generated mock data unless a saved
image or ROS camera source is selected.

## Example PX4 Command Dry-Run With Generated Vision

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m autonomy_core.runtime.surrogate_runner \
  px4_command_dry_run \
  --mavlink-endpoint udpin:0.0.0.0:14540 \
  --duration-s 5 \
  --max-messages 200 \
  --vision-source generated_mock \
  --command-roll 0.0 \
  --command-pitch 0.0 \
  --command-yaw 0.0 \
  --command-thrust 0.5
```

This may produce a dry-run command candidate if telemetry is usable, but it
still reports `command_publication_allowed=false` and `command_sent_count=0`.

## Example ROS Camera Pixel Source

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m autonomy_core.runtime.surrogate_runner \
  px4_vision_dry_run \
  --mavlink-endpoint udpin:0.0.0.0:14540 \
  --duration-s 5 \
  --max-messages 200 \
  --vision-source ros_camera \
  --camera-topic /camera \
  --resize-camera-to-competition
```

This path is optional and lazy. It requires ROS 2, `sensor_msgs`, and
`cv_bridge` to be available in the active environment. It must be treated as
PX4/Gazebo surrogate evidence only.

## Fail-Closed Example

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m autonomy_core.runtime.surrogate_runner \
  px4_command_send
```

Expected status:

```text
status: fail_closed
fail_closed: true
command_sent_count: 0
```

## Remaining 8.5D Work

Not implemented yet:

- real `AutonomyAPI` ownership;
- PX4/Gazebo command send;
- race mode;
- competition live mode;
- using surrogate results to satisfy Phase 4B.

Still requires local/manual evidence:

- a real PX4/Gazebo receive-only run against `14540` or another explicit endpoint;
- a real ROS/Gazebo camera pixel run if `ros_camera` is used;
- review of telemetry rates, frame rates, state freshness, perception update counts, command candidate counts, stale blocks, and guard rejections from local logs.
