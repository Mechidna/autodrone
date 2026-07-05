# Phase 9F.1: PX4/Gazebo Debug Yaw Override And Stream Health Gate

Phase 9F.1 is a PX4/Gazebo surrogate-only diagnostic extension to Phase 9F.
It was added after the first full surrogate autonomy loop showed that software
integration completed, but the vehicle did not physically complete the gate run.

## What It Adds

- A sender-boundary yaw override for the PX4/Gazebo surrogate command path.
- Explicit acknowledgement before the override can run.
- JSON fields proving whether the override was requested, acknowledged, and
  applied.
- Stricter Phase 9F pass criteria for command stream health.

## What It Does Not Change

- No planner changes.
- No controller changes.
- No perception changes.
- No PnP or transform changes.
- No `AutonomyAPI.attitude_control()` behavior changes.
- No competition command readiness claim.
- No Phase 4B completion claim.

## Yaw Override Boundary

The override applies only after the competition dry-run command adapter has
already produced `SET_ATTITUDE_TARGET` attitude-angle fields.

The implementation:

- Reads the autonomy-generated outgoing quaternion.
- Recovers roll, pitch, and original yaw.
- Preserves roll and pitch.
- Replaces yaw with `--px4-gazebo-debug-yaw-override-rad`.
- Rebuilds the quaternion.
- Sends only through the PX4/Gazebo surrogate command sender.

The dry-run command result still records the original autonomy/controller yaw,
so the override cannot hide what the controller requested.

## Required Flags

Phase 9F.1 requires all Phase 9F gates plus:

```bash
--px4-gazebo-debug-yaw-override-rad 1.57079632679
--ack-px4-gazebo-surrogate-debug-yaw-override
```

Without the acknowledgement, `competition_main.py` fails closed.

## Stream Health Criteria

Phase 9F now requires more than "some commands were sent":

- Command send rate must be greater than `10 Hz`.
- Vision completed frame rate must be greater than `5 Hz`.
- Maximum command-send gap must be less than or equal to `0.5 s`.
- Command send rejection count must be zero.
- Stale cached setpoint rejection count must be zero.

This prevents a sparse command stream with multi-second gaps from reporting
`phase9f_full_autonomy_loop_satisfied=true`.

## Manual Use

Append the yaw override flags to the existing Phase 9F command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m autonomy_core.runtime.competition_main \
  command_dry_run \
  --live-transports \
  --mavlink-endpoint udpin:0.0.0.0:14540 \
  --vision-bind-host 0.0.0.0 \
  --vision-port 5600 \
  --steps 1200 \
  --step-sleep-s 0.02 \
  --use-real-autonomy \
  --real-perception \
  --allow-legacy-yolo-default \
  --px4-gazebo-full-autonomy-loop \
  --ack-px4-gazebo-surrogate-full-autonomy-loop \
  --px4-gazebo-command-send \
  --ack-px4-gazebo-surrogate-command-send \
  --px4-gazebo-continuous-setpoint-stream \
  --px4-gazebo-command-max-count 1200 \
  --px4-gazebo-arm \
  --ack-px4-gazebo-surrogate-arm \
  --px4-gazebo-offboard \
  --ack-px4-gazebo-surrogate-offboard \
  --px4-gazebo-debug-yaw-override-rad 1.57079632679 \
  --ack-px4-gazebo-surrogate-debug-yaw-override
```

Use `Ctrl+C` if vehicle behavior is unsafe. This can move the local PX4/Gazebo
vehicle.

## Expected JSON

- `phase == "9F.1"`
- `px4_gazebo_debug_yaw_override_requested == true`
- `px4_gazebo_debug_yaw_override_acknowledged == true`
- `px4_gazebo_debug_yaw_override_applied_count > 0`
- `last_command_send_result.debug_yaw_override_applied == true`
- `last_command_send_result.debug_yaw_override_rad` equals the requested yaw
- `last_command_send_result.yaw_rad` equals the requested yaw
- `phase4b_satisfied == false`
- `competition_readiness_claimed == false`

If the stream health criteria fail, the run may still provide useful diagnostic
evidence, but `phase9f_full_autonomy_loop_satisfied` must remain `false`.
