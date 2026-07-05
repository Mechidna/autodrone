# Phase 9E.4 - PX4/Gazebo Attitude-Angle Hover Interface Smoke

Status: implemented for PX4/Gazebo surrogate smoke testing.

## Purpose

Phase 9E.4 isolates the raw MAVLink interface needed to reproduce the legacy
`px4_runner` startup hover behavior without using MAVSDK. It sends fixed
`SET_ATTITUDE_TARGET` attitude-angle hover setpoints through the competition
MAVLink transport and PX4/Gazebo sender path.

This phase exists because Phase 9E.3 body-rate mode proved the command path can
send body-rate setpoints, but zero body rates are not the same as a level hover.
Legacy `px4_runner` used MAVSDK absolute attitude setpoints: roll `0`, pitch `0`,
yaw held at current/reference yaw, and normalized thrust near hover.

## Command Under Test

Phase 9E.4 sends `SET_ATTITUDE_TARGET` with:

- `type_mask = 7`, meaning body roll/pitch/yaw rates are ignored.
- `q = quaternion(roll=0, pitch=0, yaw=current_state_yaw)` by default.
- Optional explicit yaw via `--px4-gazebo-attitude-hover-yaw-rad`.
- `body_roll_rate = body_pitch_rate = body_yaw_rate = 0.0`.
- Explicit normalized thrust via `--px4-gazebo-attitude-hover-thrust`.

This mirrors the legacy MAVSDK `Attitude(roll_deg=0, pitch_deg=0, yaw_deg=..., thrust_value=...)`
interface more closely than Phase 9E.3 body-rate mode.

## Hard Boundary

- PX4/Gazebo surrogate only.
- No Phase 4B claim.
- No real competition simulator telemetry claim.
- No competition command readiness claim.
- No race readiness or submitted-run readiness claim.
- Do not enable `command_live` or `race`.
- Do not use Gazebo truth for state, target selection, command readiness, command feedback, or pass/fail criteria.
- Do not call perception, planning, or `AutonomyAPI.attitude_control()` to generate this fixed hover command.
- Do not retune planner, controller, thrust, yaw, gains, hover behavior, no-target behavior, or race progression.
- Keep this path behind explicit PX4/Gazebo surrogate acknowledgement flags.

## Implementation

- `autonomy_core/runtime/px4_gazebo_command_sender.py` exposes
  `send_attitude_hover_set_attitude_target(...)`.
- `autonomy_core/runtime/competition_main.py` exposes Phase 9E.4 CLI flags and
  summary fields.
- The default yaw source is the latest usable estimated telemetry yaw from the
  competition state adapter.
- An explicit yaw can be supplied for diagnostic runs.
- The path is rejected if combined with Phase 9D `--px4-gazebo-command-send` or
  Phase 9E.3 body-rate smoke.
- Existing PX4/Gazebo arm/offboard flags may be reused.
- Offboard prestream can be satisfied by fixed attitude-hover setpoints.

## Manual Test Pattern

Run this with PX4/Gazebo already publishing MAVLink to `14540`:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m autonomy_core.runtime.competition_main \
  command_dry_run \
  --live-transports \
  --mavlink-endpoint udpin:0.0.0.0:14540 \
  --steps 600 \
  --step-sleep-s 0.02 \
  --evidence-label px4_gazebo_surrogate_phase9e4_attitude_hover_smoke \
  --px4-gazebo-arm \
  --ack-px4-gazebo-surrogate-arm \
  --px4-gazebo-offboard \
  --ack-px4-gazebo-surrogate-offboard \
  --px4-gazebo-offboard-prestream-count 10 \
  --px4-gazebo-attitude-hover-smoke \
  --ack-px4-gazebo-surrogate-attitude-hover-smoke \
  --px4-gazebo-attitude-hover-roll-rad 0.0 \
  --px4-gazebo-attitude-hover-pitch-rad 0.0 \
  --px4-gazebo-attitude-hover-thrust 0.74 \
  --px4-gazebo-command-max-count 600
```

No surrogate vision bridge is required for this first interface smoke because
perception and planning are intentionally excluded.

## Success Criteria

Phase 9E.4 passes as an interface smoke when the JSON summary reports:

- `phase = "9E.4"`.
- `px4_gazebo_attitude_hover_smoke_requested = true`.
- `px4_gazebo_attitude_hover_smoke_acknowledged = true`.
- `attitude_hover_type_mask = 7`.
- `attitude_hover_body_rates = [0.0, 0.0, 0.0]`.
- `attitude_hover_yaw_source = "current_state"` unless an explicit yaw was supplied.
- `attitude_hover_thrust` equals the requested thrust.
- `attitude_hover_command_sent_count > 0`.
- `attitude_hover_command_rejection_count = 0`.
- `heartbeat_seen = true`.
- `state_usable = true`.
- `armed_state_observed = true` if arming was requested.
- `offboard_state_observed = true` if Offboard was requested.
- `phase9e4_attitude_hover_interface_satisfied = true`.
- `phase4b_satisfied = false`.
- `competition_readiness_claimed = false`.

Manual vehicle behavior should resemble a short level hover more closely than
Phase 9E.3 body-rate smoke. If it still drifts or failsafes, inspect PX4 mode,
setpoint stream timing, thrust value, and failsafe state before changing
competition autonomy behavior.

## Failure / Inconclusive Criteria

The phase is not a clean pass if any of these occur:

- No heartbeat or stale heartbeat.
- State unusable.
- No attitude-hover command sends.
- Any sender rejection.
- PX4 does not arm when arming was requested.
- PX4 does not enter Offboard when Offboard was requested.
- PX4 immediately failsafes, RTLs, disarms, or rejects mode/command setup.
- The sent command uses body-rate mode (`type_mask=128`) instead of attitude-angle mode (`type_mask=7`).
- Any Gazebo truth path is used for readiness.

## Follow-Up

If Phase 9E.4 behaves like the legacy hover, the next work should compare the
competition command adapter's attitude-angle output against this fixed hover
path before deciding whether to keep attitude-angle command output or build a
separate body-rate feedback controller.
