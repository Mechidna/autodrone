# Phase 9F.3B - Generic Streamer Sender Integration

## Status

Implemented as an explicit PX4/Gazebo surrogate-only opt-in path in:

- `autonomy_core/runtime/competition_main.py`
- `tests/test_competition_main.py`

This phase uses the generic streamer from:

- `autonomy_core/runtime/competition_setpoint_streamer.py`

## Purpose

Phase 9F.3B wires `CompetitionSetpointStreamer` into the PX4/Gazebo surrogate
fixed-rate sender path.

This keeps the reusable setpoint policy separate from PX4/Gazebo MAVLink
session mechanics:

- `CompetitionSetpointStreamer` decides whether fields should emit.
- `competition_main.py` owns PX4/Gazebo surrogate send gating.
- `Px4GazeboSetAttitudeTargetSender` remains the only local sender.

## CLI Flags

9F.3B is enabled only when all 9F/9F.2 gates are already enabled plus:

```bash
--px4-gazebo-generic-setpoint-streamer
--ack-px4-gazebo-surrogate-generic-setpoint-streamer
```

It also requires the existing fixed-rate stream flags:

```bash
--px4-gazebo-fixed-rate-setpoint-stream
--ack-px4-gazebo-surrogate-fixed-rate-setpoint-stream
```

## Behavior

When enabled:

- Accepted command candidates update `CompetitionSetpointStreamer`.
- Fixed-rate send attempts ask the streamer for the next eligible fields.
- The streamer enforces the configured cadence and command freshness policy.
- The PX4/Gazebo sender sends only fields emitted by the streamer.
- Sent result metadata records `generic_setpoint_streamer=true`.
- Summary output includes `generic_setpoint_streamer_summary`.

The old 9F.2 direct cached-field path remains available when the 9F.3B flag is
not set.

## Hard Boundary

This phase does not:

- Add live competition command mode.
- Enable `command_live` or `race`.
- Change `CompetitionRunner` command publication gates.
- Change `AutonomyAPI.attitude_control()`.
- Change planner, controller, perception, PnP, transform selection, race
  progression, yaw, thrust, hover, or no-target behavior.
- Change command adapter semantics.
- Add MAVSDK, ROS, or new transports.
- Claim Phase 4B, real competition simulator telemetry evidence, command
  readiness, race readiness, submitted-run readiness, or competition readiness.

## Tests

The deterministic tests prove:

- 9F.3B reports `phase="9F.3B"`.
- The generic streamer flag and acknowledgement are required.
- The underlying 9F.2 fixed-rate sender criteria still pass.
- `generic_setpoint_streamer_summary` records updates and emits.
- The last command send is marked as a generic streamer send.
- Phase 4B and competition readiness remain false.

## Manual Run Shape

Use the same Phase 9F.2 command, adding:

```bash
--px4-gazebo-generic-setpoint-streamer \
--ack-px4-gazebo-surrogate-generic-setpoint-streamer
```

Success criteria:

- `"phase": "9F.3B"`
- `"phase9f3b_generic_setpoint_streamer_satisfied": true`
- `"phase9f2_fixed_rate_setpoint_stream_satisfied": true`
- `"fixed_rate_setpoint_stream_rejection_count": 0`
- `"command_send_rejection_count": 0`
- `"generic_setpoint_streamer_summary"` has `stats.emit_count > 0`
- `"phase4b_satisfied": false`
- `"competition_readiness_claimed": false`
