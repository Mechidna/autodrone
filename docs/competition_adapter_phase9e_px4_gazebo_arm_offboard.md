# Phase 9E - PX4/Gazebo Surrogate Arm/Offboard Bring-Up

Status: implemented as a PX4/Gazebo surrogate-only lifecycle gate.

## What This Phase Does

- Keeps `CompetitionRunner` in `command_dry_run`.
- Adds explicit PX4/Gazebo-only arming and offboard mode commands.
- Reuses the already-open MAVLink connection from the competition MAVLink transport.
- Keeps perception, planning, control, and command candidate generation inside the competition stack.
- Records arm/offboard attempts, sent counts, last lifecycle command result, and telemetry evidence.

## What This Phase Does Not Claim

- It does not satisfy Phase 4B.
- It does not prove real competition simulator telemetry.
- It does not prove real competition command acceptance.
- It does not claim competition readiness, race readiness, or submitted-run readiness.
- It does not enable `command_live` or `race`.
- It does not use Gazebo truth.

## Explicit Flags

Phase 9E lifecycle commands are disabled by default.

Arming requires both:

```bash
--px4-gazebo-arm
--ack-px4-gazebo-surrogate-arm
```

PX4 Offboard mode request requires both:

```bash
--px4-gazebo-offboard
--ack-px4-gazebo-surrogate-offboard
```

If PX4 needs a setpoint stream before mode entry, require one explicitly:

```bash
--px4-gazebo-offboard-prestream-count 10
```

Attitude-target command sending still requires the Phase 9D flags:

```bash
--px4-gazebo-command-send
--ack-px4-gazebo-surrogate-command-send
```

## Commands Sent

- Arm: `MAV_CMD_COMPONENT_ARM_DISARM` with `param1=1`.
- Offboard: `MAV_CMD_DO_SET_MODE` with PX4 custom main mode `6`.
- Attitude target: existing Phase 9D `SET_ATTITUDE_TARGET` path from the competition command adapter.

## Phase 9E.1 Follow-Up

If PX4 arms or accepts Offboard but then disarms because setpoints are not
streamed continuously, use Phase 9E.1:

```bash
--px4-gazebo-continuous-setpoint-stream
--px4-gazebo-command-max-age-s 0.5
```

Phase 9E.1 still uses `command_dry_run` and the explicit PX4/Gazebo surrogate
send acknowledgements. It reuses only the latest accepted competition command
candidate while heartbeat, state, and cached-command age remain fresh.

## Telemetry Evidence

The summary checks:

- Armed state from `HEARTBEAT.base_mode & 128`.
- PX4 Offboard state from `HEARTBEAT.custom_mode`, where `(custom_mode >> 16) & 0xFF == 6`.

If the lifecycle commands are sent but telemetry does not prove the requested state, Phase 9E is marked incomplete rather than successful.

## First Manual Test Pattern

Terminal 1:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m autonomy_core.runtime.competition_main \
  command_dry_run \
  --live-transports \
  --use-real-autonomy \
  --real-perception \
  --allow-legacy-yolo-default \
  --mavlink-endpoint udpin:0.0.0.0:14540 \
  --vision-bind-host 0.0.0.0 \
  --vision-port 5600 \
  --steps 300 \
  --step-sleep-s 0.02 \
  --evidence-label px4_gazebo_surrogate_phase9e_arm_offboard \
  --px4-gazebo-arm \
  --ack-px4-gazebo-surrogate-arm \
  --px4-gazebo-offboard \
  --ack-px4-gazebo-surrogate-offboard \
  --px4-gazebo-offboard-prestream-count 10 \
  --px4-gazebo-command-send \
  --ack-px4-gazebo-surrogate-command-send \
  --px4-gazebo-command-max-count 20
```

Terminal 2:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m autonomy_core.runtime.surrogate_vision_bridge \
  --camera-topic /camera \
  --send-host 127.0.0.1 \
  --send-port 5600 \
  --frames 60 \
  --timeout-s 15
```

## Success Criteria

- `phase` is `"9E"`.
- `px4_gazebo_arm_requested=true`.
- `px4_gazebo_arm_acknowledged=true`.
- `arm_sent_count > 0`.
- `armed_state_observed=true`.
- If offboard was requested, `offboard_sent_count > 0`.
- If offboard prestream was requested, `command_sent_count` reaches the requested prestream count before success.
- If offboard was requested, `offboard_state_observed=true`.
- `state_usable=true`.
- `vision_frames_completed > 0`.
- `perception_update_calls > 0`.
- `command_candidate_count > 0`.
- If command-send is requested, `command_sent_count > 0`.
- `phase9e_surrogate_arm_offboard_satisfied=true`.
- `phase4b_satisfied=false`.
- `competition_readiness_claimed=false`.

If `armed_state_observed=false` or `offboard_state_observed=false`, inspect the MAVLink telemetry summary before increasing command count.
