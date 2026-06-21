# Phase 9F.3C - Generic Streamer Fallback Setpoint

## Status

Implemented as an explicit PX4/Gazebo surrogate-only opt-in path in:

- `autonomy_core/runtime/competition_main.py`
- `tests/test_competition_main.py`

This phase extends the generic streamer path from:

- `autonomy_core/runtime/competition_setpoint_streamer.py`

## Purpose

Phase 9F.3C adds an explicit fallback/hold setpoint for the PX4/Gazebo
surrogate fixed-rate command stream.

The problem addressed by this phase is the Phase 9F.3B failure mode where the
full autonomy loop sent valid setpoints initially, then rejected stale autonomy
commands once perception/planning stopped producing fresh command candidates.
PX4 Offboard-style control still needs a continuous stream, so the surrogate
sender needs a clearly labeled fallback while autonomy output is missing or
stale.

## CLI Flags

9F.3C is enabled only when the 9F/9F.2/9F.3B gates are already enabled plus:

```bash
--px4-gazebo-generic-setpoint-fallback \
--ack-px4-gazebo-surrogate-generic-setpoint-fallback \
--px4-gazebo-generic-fallback-thrust 0.74
```

Optional fallback attitude flags:

```bash
--px4-gazebo-generic-fallback-roll-rad 0.0 \
--px4-gazebo-generic-fallback-pitch-rad 0.0 \
--px4-gazebo-generic-fallback-yaw-rad 1.57079632679
```

If fallback yaw is omitted, the fallback uses the current vehicle yaw from the
state adapter. Roll and pitch default to `0.0`. Thrust has no default and must be
provided explicitly.

## Behavior

When enabled:

- `competition_main.py` builds fallback `SET_ATTITUDE_TARGET` dry-run fields with
  the existing command adapter.
- `CompetitionSetpointStreamer` receives those fallback fields each runner step.
- If the latest autonomy command is missing or stale, the streamer emits fallback
  fields instead of returning a stale-command rejection.
- Fixed-rate PX4/Gazebo surrogate sends are marked with
  `generic_setpoint_fallback=true` and the fallback label.
- Summary output records fallback update/rejection counts and phase-specific
  success criteria.

## Hard Boundary

This phase does not:

- Enable `command_live` or `race`.
- Add live competition transports.
- Change `CompetitionRunner` command publication gates.
- Change `AutonomyAPI.attitude_control()`.
- Change planner, controller, perception, PnP, transform selection, race
  progression, yaw, thrust, hover, or no-target behavior.
- Change command adapter semantics.
- Add MAVSDK, ROS, or new transport dependencies.
- Claim Phase 4B, real competition simulator telemetry evidence, command
  readiness, race readiness, submitted-run readiness, or competition readiness.

## Acceptance Criteria

A successful 9F.3C run reports:

- `phase == "9F.3C"`
- `px4_gazebo_generic_setpoint_fallback_requested == true`
- `px4_gazebo_generic_setpoint_fallback_acknowledged == true`
- `generic_setpoint_fallback_update_count > 0`
- `generic_setpoint_fallback_rejection_count == 0`
- `generic_setpoint_streamer_summary.stats.fallback_emit_count > 0`
- `fixed_rate_setpoint_stream_rejection_count == 0`
- `command_send_rejection_count == 0`
- Last command send is marked with `generic_setpoint_fallback == true`
- `phase9f3c_fallback_setpoint_satisfied == true`
- `phase4b_satisfied == false`
- `competition_readiness_claimed == false`

## Manual Run Shape

Start from the Phase 9F.3B command and add:

```bash
--px4-gazebo-generic-setpoint-fallback \
--ack-px4-gazebo-surrogate-generic-setpoint-fallback \
--px4-gazebo-generic-fallback-thrust 0.74
```

If you want to hold a gate-facing yaw for debugging, add an explicit yaw:

```bash
--px4-gazebo-generic-fallback-yaw-rad 1.57079632679
```

The fallback thrust should be treated as PX4/Gazebo surrogate tuning only. It is
not a competition controller retune and is not evidence that the real competition
simulator will accept or behave the same way.
