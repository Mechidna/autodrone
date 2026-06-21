# Phase 9F.3A - Generic Competition Setpoint Streamer

## Status

Implemented as an import-safe, transport-free policy module:

- `autonomy_core/runtime/competition_setpoint_streamer.py`
- `tests/test_competition_setpoint_streamer.py`

## Purpose

Phase 9F.3A separates the generic command-stream policy from the
PX4/Gazebo-specific Offboard lifecycle and sender code.

The streamer decides which already-built `SET_ATTITUDE_TARGET` fields should
be emitted at a fixed cadence:

- Prefer fresh autonomy command fields.
- Fall back to explicit hold/fallback fields when autonomy is missing or stale.
- Reject stream rates at or above the VADR command-rate ceiling.
- Preserve `phase4b_satisfied=false`.
- Preserve `competition_readiness_claimed=false`.

## Hard Boundary

This phase does not:

- Open sockets.
- Import `pymavlink`, `cv2`, `mavsdk`, or `rclpy`.
- Send MAVLink commands.
- Arm PX4.
- Switch Offboard mode.
- Start Gazebo, PX4, ROS, or the competition simulator.
- Call perception, planning, controller code, or `AutonomyAPI`.
- Change planner, controller, perception, PnP, transform, race progression, or
  command adapter semantics.
- Claim Phase 4B, real competition telemetry evidence, command readiness, race
  readiness, submitted-run readiness, or competition readiness.

Transport-specific code must consume the returned fields. The generic streamer
only returns a decision object.

## Policy

Inputs:

- `DryRunSetAttitudeTargetFields` from the competition command adapter.
- A monotonic/wall-clock timestamp supplied by the caller.
- Optional fallback/hold fields.

Outputs:

- `CompetitionSetpointStreamDecision`
- `CompetitionSetpointStreamStats`

The streamer:

- Emits no faster than `stream_rate_hz`.
- Requires `stream_rate_hz < 100 Hz` per VADR-TS-002.
- Defaults to `20 Hz`.
- Treats autonomy command fields as stale after `0.5 s` by default.
- Rewrites `time_boot_ms` and `sequence` only on emitted decisions.
- Emits fallback fields when autonomy is missing or stale and fallback fields
  are configured.
- Produces no-send decisions when neither fresh autonomy nor fallback fields are
  available.

## Tests

The deterministic tests prove:

- Importing the streamer does not load live dependencies.
- Rates at or above `100 Hz` are rejected.
- Missing autonomy emits fallback fields when configured.
- Fresh autonomy is preferred over fallback fields.
- Calls inside the configured stream period are rate-limited.
- Stale autonomy falls back to hold fields.
- Missing fallback blocks emission when autonomy is unavailable or stale.
- Invalid fields are rejected.
- Summaries never claim Phase 4B or competition readiness.

## Next Work

Phase 9F.3B should wire this generic streamer into the PX4/Gazebo sender path
as a local surrogate integration step. That later phase may handle PX4/Gazebo
Offboard prestream/session mechanics, but it must remain separate from the
generic competition policy in this phase.
