# Phase 9E.2 - PX4/Gazebo Surrogate Command Safety Clamp And Stream Diagnostics

Status: implemented as a PX4/Gazebo surrogate-only sender-boundary safety layer.

## What This Phase Does

- Keeps `CompetitionRunner` in `command_dry_run`.
- Keeps `command_live` and `race` fail-closed.
- Leaves `AutonomyAPI.attitude_control()` and controller tuning unchanged.
- Leaves `CompetitionDryRunCommandAdapter` output unchanged.
- Adds an explicit PX4/Gazebo-only thrust clamp at the final MAVLink sender boundary.
- Records raw controller thrust separately from the thrust actually sent to PX4/Gazebo.
- Records command stream source (`current` or `cached`), yaw range, thrust ranges, clamp count, and send gaps.

## Why This Was Added

The Phase 9E.1 PX4/Gazebo run showed that commands were transmitted, but the local SITL vehicle climbed aggressively and oscillated. The existing controller can output thrust up to the configured `0.85` maximum, while local hover thrust is about `0.74`. Phase 9E.2 adds an explicit surrogate-only clamp for local safety without retuning competition behavior.

## Explicit Flags

The clamp is disabled by default. It requires the Phase 9D command-send flags plus an explicit clamp flag and at least one clamp bound:

```bash
--px4-gazebo-command-send \
--ack-px4-gazebo-surrogate-command-send \
--px4-gazebo-surrogate-thrust-clamp \
--px4-gazebo-surrogate-thrust-clamp-max 0.76
```

The clamp applies after a command candidate has already passed the competition command adapter. The JSON summary still reports the original command candidate thrust under `last_command_result.fields.thrust`.

## Safety Boundaries

- PX4/Gazebo surrogate only.
- No Phase 4B claim.
- No competition readiness claim.
- No `command_live` or `race` enablement.
- No Gazebo truth for state, perception, target selection, command readiness, or command feedback.
- No planner, controller, YOLO, PnP, perception scoring, race progression, logger schema, or legacy runner behavior changes.
- No hidden hover-thrust retune.
- The clamp is at the sender boundary only and is opt-in.

## Summary Fields

The JSON summary includes:

- `phase: "9E.2"`
- `px4_gazebo_surrogate_thrust_clamp_requested`
- `px4_gazebo_surrogate_thrust_clamp_min`
- `px4_gazebo_surrogate_thrust_clamp_max`
- `last_command_result.fields.thrust` for the raw competition command candidate
- `last_command_send_result.raw_thrust`
- `last_command_send_result.thrust`
- `last_command_send_result.thrust_clamped`
- `last_command_send_result.stream_source`
- `last_command_send_result.send_gap_s`
- `command_sender_summary.stats.thrust_clamp_count`
- `command_sender_summary.stats.min_raw_thrust`
- `command_sender_summary.stats.max_raw_thrust`
- `command_sender_summary.stats.min_sent_thrust`
- `command_sender_summary.stats.max_sent_thrust`
- `command_sender_summary.stats.stream_source_counts`
- `command_sender_summary.stats.max_send_gap_s`
- `phase9e2_surrogate_command_safety_satisfied`
- `phase9e2_success_criteria`

## Manual Test Pattern

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
  --steps 600 \
  --step-sleep-s 0.02 \
  --evidence-label px4_gazebo_surrogate_phase9e2_clamped_stream \
  --px4-gazebo-arm \
  --ack-px4-gazebo-surrogate-arm \
  --px4-gazebo-offboard \
  --ack-px4-gazebo-surrogate-offboard \
  --px4-gazebo-offboard-prestream-count 10 \
  --px4-gazebo-command-send \
  --ack-px4-gazebo-surrogate-command-send \
  --px4-gazebo-continuous-setpoint-stream \
  --px4-gazebo-command-max-count 300 \
  --px4-gazebo-command-max-age-s 0.5 \
  --px4-gazebo-surrogate-thrust-clamp \
  --px4-gazebo-surrogate-thrust-clamp-max 0.76
```

Terminal 2:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m autonomy_core.runtime.surrogate_vision_bridge \
  --camera-topic /camera \
  --send-host 127.0.0.1 \
  --send-port 5600 \
  --frames 300 \
  --timeout-s 20
```

## Success Criteria

- `phase` is `"9E.2"`.
- `state_usable=true`.
- `heartbeat_seen=true` and heartbeat age is fresh.
- `vision_frames_completed > 0`.
- `perception_update_calls > 0`.
- `command_candidate_count > 0`.
- `command_sent_count > 0`.
- `last_command_result.fields.thrust` may show the raw controller output.
- `last_command_send_result.raw_thrust` records the same raw output before clamp.
- `last_command_send_result.thrust <= --px4-gazebo-surrogate-thrust-clamp-max`.
- `last_command_send_result.thrust_clamped=true` for at least one sent command.
- `command_sender_summary.stats.thrust_clamp_count > 0`.
- `command_sender_summary.stats.max_sent_thrust <= --px4-gazebo-surrogate-thrust-clamp-max`.
- `phase9e2_surrogate_command_safety_satisfied=true`.
- `phase4b_satisfied=false`.
- `competition_readiness_claimed=false`.

If the vehicle still climbs or oscillates with a clamp near hover thrust, do not retune planner/controller in this phase. Record the JSON summary, PX4 console status, and observed motion, then use a separate investigation phase for command semantics, mode state, and attitude/thrust response.
