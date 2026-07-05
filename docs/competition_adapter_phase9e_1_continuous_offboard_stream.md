# Phase 9E.1 - PX4/Gazebo Continuous Offboard Setpoint Stream

Status: implemented as a PX4/Gazebo surrogate-only command-stream gate.

## What This Phase Does

- Keeps `CompetitionRunner` in `command_dry_run`.
- Keeps `command_live` and `race` fail-closed.
- Adds an explicit PX4/Gazebo-only continuous setpoint stream flag.
- Caches the latest accepted competition command candidate only from a step with fresh vision/perception gates.
- Reuses that cached command for later sends only while heartbeat, state, and command age remain within configured limits.
- Keeps all actual sending outside `CompetitionRunner`, in the Phase 9D/9E CLI/lifecycle layer.

## Why This Was Added

The Phase 9E PX4/Gazebo run proved that arm/offboard commands and attitude targets could be transmitted, but PX4 did not remain active because setpoint publication was tied to fresh perception frames. PX4 Offboard behavior expects a continuing setpoint stream. Phase 9E.1 decouples the PX4/Gazebo surrogate setpoint stream from per-frame image freshness while still requiring the cached setpoint to originate from the competition stack.

## Explicit Flags

Continuous streaming is disabled by default. It requires the Phase 9D send flags plus the Phase 9E.1 stream flag:

```bash
--px4-gazebo-command-send \
--ack-px4-gazebo-surrogate-command-send \
--px4-gazebo-continuous-setpoint-stream
```

The cached command age limit is controlled by:

```bash
--px4-gazebo-command-max-age-s 0.5
```

Arming and Offboard mode still require the Phase 9E flags:

```bash
--px4-gazebo-arm \
--ack-px4-gazebo-surrogate-arm \
--px4-gazebo-offboard \
--ack-px4-gazebo-surrogate-offboard
```

## Safety Boundaries

- PX4/Gazebo surrogate only.
- No Phase 4B claim.
- No competition readiness claim.
- No `command_live` or `race` enablement.
- No Gazebo truth for state, gate position, target selection, command readiness, or command feedback.
- No changes to planner, controller, YOLO, PnP, perception scoring, race progression, logger schema, or legacy `px4_runner` behavior.
- Command rate remains bounded by the existing `<100 Hz` command sender gate.

## Summary Fields

The JSON summary includes:

- `phase: "9E.1"`
- `px4_gazebo_continuous_setpoint_stream_requested`
- `px4_gazebo_command_max_age_s`
- `setpoint_stream_cache_update_count`
- `setpoint_stream_reused_count`
- `setpoint_stream_stale_rejection_count`
- `last_cached_command_age_s`
- `phase9e1_continuous_setpoint_stream_satisfied`
- `phase9e1_success_criteria`

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
  --evidence-label px4_gazebo_surrogate_phase9e1_stream \
  --px4-gazebo-arm \
  --ack-px4-gazebo-surrogate-arm \
  --px4-gazebo-offboard \
  --ack-px4-gazebo-surrogate-offboard \
  --px4-gazebo-offboard-prestream-count 10 \
  --px4-gazebo-command-send \
  --ack-px4-gazebo-surrogate-command-send \
  --px4-gazebo-continuous-setpoint-stream \
  --px4-gazebo-command-max-count 300 \
  --px4-gazebo-command-max-age-s 0.5
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

- `phase` is `"9E.1"`.
- `state_usable=true`.
- `heartbeat_seen=true` and heartbeat age is fresh.
- `vision_frames_completed > 0`.
- `perception_update_calls > 0`.
- `command_candidate_count > 0`.
- `setpoint_stream_cache_update_count > 0`.
- `setpoint_stream_reused_count > 0`.
- `setpoint_stream_stale_rejection_count = 0`.
- `command_sent_count > 1` and remains `<= --px4-gazebo-command-max-count`.
- `armed_state_observed=true` if arm was requested.
- `offboard_state_observed=true` if offboard was requested.
- `phase9e1_continuous_setpoint_stream_satisfied=true`.
- `phase4b_satisfied=false`.
- `competition_readiness_claimed=false`.

If PX4 arms but then disarms again, inspect `setpoint_stream_reused_count`, `command_sent_count`, `last_command_send_result`, and `command_sender_summary.stats.rejection_reasons` before changing thrust or control gains.
