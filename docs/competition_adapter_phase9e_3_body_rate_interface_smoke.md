# Phase 9E.3 - PX4/Gazebo Body-Rate SET_ATTITUDE_TARGET Interface Smoke

Status: implemented for PX4/Gazebo surrogate smoke testing.

## Purpose

Phase 9E.3 isolates the MAVLink command interface before debugging autonomy
behavior. It proves whether PX4/Gazebo can arm, accept Offboard mode if needed,
and respond to a continuous `SET_ATTITUDE_TARGET` body-rate stream that mirrors
the PyAIPilotExample attitude-control example.

This phase does not use perception, planning, or `AutonomyAPI.attitude_control()`
to generate commands. It sends fixed, explicit body-rate setpoints so interface
problems can be separated from controller behavior.

## Why This Phase Exists

Phase 9E.2 proved the current quaternion-attitude command path can transmit
commands and clamp thrust, but PX4/Gazebo still entered failsafe/RTL and did not
show stable Offboard behavior.

The read-only PyAIPilotExample reference shows an alternate attitude-control
path using:

```text
SET_ATTITUDE_TARGET
q = [1, 0, 0, 0] dummy quaternion
type_mask = ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
body_roll_rate = ROLL_RATE
body_pitch_rate = PITCH_RATE
body_yaw_rate = YAW_RATE
thrust = THRUST
```

That is body-rate mode. It is not the same as the current competition adapter
command path, which uses attitude quaternion mode and ignores body rates.

Phase 9E.3 tests body-rate mode directly against PX4/Gazebo before deciding how
to adapt autonomy controller output.

## Hard Boundary

- PX4/Gazebo surrogate only.
- No Phase 4B claim.
- No real competition simulator telemetry claim.
- No competition command readiness claim.
- No race readiness or submitted-run readiness claim.
- Do not enable `command_live` or `race`.
- Do not use Gazebo truth for state, target selection, command readiness, or
  pass/fail criteria.
- Do not call perception.
- Do not call planning.
- Do not call `AutonomyAPI.attitude_control()` for the fixed body-rate smoke.
- Do not convert autonomy RPY output into body rates in this phase.
- Do not retune planner, controller, yaw, thrust, hover, or no-target behavior.
- Do not modify `CompetitionRunner` command publication semantics.
- Keep the body-rate sender behind explicit PX4/Gazebo surrogate flags.

## Command Under Test

The fixed smoke command should send `SET_ATTITUDE_TARGET` with:

```text
type_mask = ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
q = [1, 0, 0, 0]
body_roll_rate = explicit fixed roll rate, default 0.0 rad/s
body_pitch_rate = explicit fixed pitch rate, default 0.0 rad/s
body_yaw_rate = explicit fixed yaw rate, default 0.0 rad/s
thrust = explicit fixed normalized thrust, default selected by operator
```

The first safe manual value should be near the local PX4/Gazebo hover thrust,
for example:

```text
body_roll_rate = 0.0
body_pitch_rate = 0.0
body_yaw_rate = 0.0
thrust = 0.74
```

The thrust value must be explicit. Do not silently reuse controller thrust or a
hidden default.

## Expected Implementation Shape

Sender support exists for a fixed body-rate `SET_ATTITUDE_TARGET` smoke path
using an already-open MAVLink connection from the competition MAVLink transport.

Suggested explicit flags:

```bash
--px4-gazebo-body-rate-smoke
--ack-px4-gazebo-surrogate-body-rate-smoke
--px4-gazebo-body-roll-rate 0.0
--px4-gazebo-body-pitch-rate 0.0
--px4-gazebo-body-yaw-rate 0.0
--px4-gazebo-body-rate-thrust 0.74
--px4-gazebo-command-max-count 300
```

If arming/offboard is needed for PX4/Gazebo, reuse the existing explicit
surrogate lifecycle flags:

```bash
--px4-gazebo-arm
--ack-px4-gazebo-surrogate-arm
--px4-gazebo-offboard
--ack-px4-gazebo-surrogate-offboard
--px4-gazebo-offboard-prestream-count 10
```

The body-rate smoke path should be independent from:

```text
AutonomyAPI.attitude_control()
CompetitionDryRunCommandAdapter angle-to-quaternion mode
surrogate thrust clamp used by Phase 9E.2
perception/planning freshness gates
```

It may still require fresh heartbeat and usable state because those are safety
preconditions for sending to PX4/Gazebo.

Implemented behavior:

- `autonomy_core/runtime/px4_gazebo_command_sender.py` exposes an explicit
  `send_body_rate_set_attitude_target(...)` method.
- `autonomy_core/runtime/competition_main.py` exposes the Phase 9E.3 flags and
  JSON summary fields.
- The body-rate smoke path is rejected if combined with
  `--px4-gazebo-command-send`.
- The body-rate smoke path is rejected if real AutonomyAPI/perception is
  requested.
- Existing PX4/Gazebo arm/offboard flags may be reused, and Offboard prestream
  can be satisfied by fixed body-rate setpoints.
- The path still requires heartbeat and usable estimated state before sending.

## JSON Summary Requirements

The summary should include:

- `phase = "9E.3"`
- `px4_gazebo_body_rate_smoke_requested`
- `px4_gazebo_body_rate_smoke_acknowledged`
- `body_rate_type_mask`
- `body_rate_q`
- `body_roll_rate`
- `body_pitch_rate`
- `body_yaw_rate`
- `body_rate_thrust`
- `body_rate_command_sent_count`
- `body_rate_command_rejection_count`
- `last_body_rate_command_send_result`
- `armed_state_observed`
- `offboard_state_observed` if offboard is requested
- `phase9e3_body_rate_interface_satisfied`
- `phase4b_satisfied = false`
- `competition_readiness_claimed = false`

## Success Criteria

Phase 9E.3 passes as an interface smoke when:

- `phase` is `"9E.3"`.
- Explicit body-rate smoke flag is present.
- Explicit body-rate smoke acknowledgement is present.
- `type_mask` equals `ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE`.
- `q` equals `[1, 0, 0, 0]`.
- Body rates equal the requested fixed values.
- Thrust equals the requested fixed value.
- Command send count is greater than zero.
- Command rejection count is zero.
- Heartbeat is seen and fresh.
- State is usable.
- Armed state is observed if arming was requested.
- Offboard state is observed if offboard was requested and required for the run.
- PX4 does not immediately enter repeated failsafe/RTL during the bounded smoke.
- `phase4b_satisfied=false`.
- `competition_readiness_claimed=false`.

## Failure / Inconclusive Criteria

The phase is not considered a clean interface pass if any of these occur:

- No heartbeat.
- Stale heartbeat.
- State unusable.
- Command send count remains zero.
- Command sender rejects fixed body-rate fields.
- PX4 repeatedly enters failsafe/RTL.
- PX4 arms but immediately disarms.
- PX4 Offboard mode is requested but not observed.
- The sent command uses quaternion-attitude mode instead of body-rate mode.
- Any Gazebo truth path is used to satisfy readiness.

If PX4/Gazebo still fails with fixed body rates and hover-ish thrust, the
problem is likely PX4 Offboard lifecycle, setpoint stream timing, or MAVLink
mode acceptance rather than perception/planning/controller behavior.

## Manual Test Pattern

Terminal 1, run the fixed body-rate smoke through the competition MAVLink
transport and PX4/Gazebo sender path:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m autonomy_core.runtime.competition_main \
  command_dry_run \
  --live-transports \
  --mavlink-endpoint udpin:0.0.0.0:14540 \
  --steps 600 \
  --step-sleep-s 0.02 \
  --evidence-label px4_gazebo_surrogate_phase9e3_body_rate_smoke \
  --px4-gazebo-arm \
  --ack-px4-gazebo-surrogate-arm \
  --px4-gazebo-offboard \
  --ack-px4-gazebo-surrogate-offboard \
  --px4-gazebo-offboard-prestream-count 10 \
  --px4-gazebo-body-rate-smoke \
  --ack-px4-gazebo-surrogate-body-rate-smoke \
  --px4-gazebo-body-roll-rate 0.0 \
  --px4-gazebo-body-pitch-rate 0.0 \
  --px4-gazebo-body-yaw-rate 0.0 \
  --px4-gazebo-body-rate-thrust 0.74 \
  --px4-gazebo-command-max-count 300
```

No surrogate vision bridge is required for this first interface smoke because
perception and planning are intentionally excluded.

## Follow-Up After 9E.3

If Phase 9E.3 passes, the next separate phase should add a body-rate command
adapter mode for autonomy output:

```text
AutonomyAPI.attitude_control()
  -> roll_angle, pitch_angle, yaw_angle, thrust
  -> explicit angle-to-body-rate adapter
  -> SET_ATTITUDE_TARGET body-rate mode
```

That later phase should choose and test conservative conversion gains. It must
not be bundled into Phase 9E.3.
