# Phase 9D - PX4/Gazebo Surrogate Command-Send Safety Gate

Status: implemented as a PX4/Gazebo surrogate-only command-send gate.

## What This Phase Does

- Keeps `CompetitionRunner` in `command_dry_run` mode.
- Uses the existing competition stack to receive PX4/Gazebo MAVLink telemetry.
- Uses `surrogate_vision_bridge.py` to feed Gazebo camera frames as VADR-style UDP `5600` packets.
- Runs the competition-safe `AutonomyAPI` profile through perception, planning, and control.
- Converts the resulting attitude tuple through `CompetitionDryRunCommandAdapter`.
- Sends the resulting `SET_ATTITUDE_TARGET` fields to PX4/Gazebo only when explicit Phase 9D flags and safety gates pass.

## What This Phase Does Not Claim

- It does not satisfy Phase 4B.
- It does not prove real competition simulator telemetry.
- It does not prove real competition command acceptance.
- It does not claim competition readiness, race readiness, or submitted-run readiness.
- It does not enable `command_live` or `race`.
- It does not use Gazebo truth for state, perception, target selection, or command readiness.

## Safety Gates

Phase 9D command sending requires:

- `mode=command_dry_run`
- `--live-transports`
- `--use-real-autonomy`
- `--real-perception`
- `--px4-gazebo-command-send`
- `--ack-px4-gazebo-surrogate-command-send`
- a fresh heartbeat
- usable competition state
- a fresh decoded image/perception update in the current step
- synced autonomy telemetry
- successful planning
- accepted command-adapter output
- command count bounded by `--px4-gazebo-command-max-count`
- command period preserving the `<100 Hz` requirement
- finite, bounded `SET_ATTITUDE_TARGET` fields

## Manual Smoke Test Pattern

Terminal 1, run the competition stack with Phase 9D send gated to one command:

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
  --steps 150 \
  --step-sleep-s 0.02 \
  --evidence-label px4_gazebo_surrogate_phase9d \
  --px4-gazebo-command-send \
  --ack-px4-gazebo-surrogate-command-send \
  --px4-gazebo-command-max-count 1
```

Terminal 2, feed Gazebo camera frames as VADR-style UDP packets:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m autonomy_core.runtime.surrogate_vision_bridge \
  --camera-topic /camera \
  --send-host 127.0.0.1 \
  --send-port 5600 \
  --frames 20 \
  --timeout-s 10
```

## Success Criteria

Expected JSON fields for a one-command smoke test:

- `phase` is `"9D"`
- `mode` is `"command_dry_run"`
- `phase9d_surrogate_command_send_satisfied` is `true`
- `phase4b_satisfied` is `false`
- `competition_readiness_claimed` is `false`
- `px4_gazebo_command_send_requested` is `true`
- `px4_gazebo_command_send_acknowledged` is `true`
- `state_usable` is `true`
- `vision_frames_completed > 0`
- `perception_update_calls > 0`
- `command_candidate_count > 0`
- `command_send_attempt_count > 0`
- `command_sent_count` is `1` for the one-command smoke test
- `command_send_rejection_count` is `0`
- `last_command_send_result.sent` is `true`
- `last_command_send_result.surrogate_label` is `"PX4/Gazebo surrogate command-send only"`

If `command_sent_count` stays `0`, inspect `last_command_send_result.rejection_reason`.

Common rejection reasons:

- `heartbeat_missing`
- `heartbeat_stale`
- `state_unusable`
- `image_not_fresh`
- `perception_not_fresh`
- `autonomy_telemetry_not_synced`
- `planning_not_succeeded`
- `command_candidate_not_attempted`
- `command_rate_limit`
- `thrust_safety_limit`
- `roll_safety_limit`
- `pitch_safety_limit`
- `mavlink connection unavailable`

## Implementation Notes

- The new sender is `autonomy_core/runtime/px4_gazebo_command_sender.py`.
- The sender is import-safe and does not import `pymavlink`.
- The sender reuses the already-open MAVLink connection from `competition_mavlink_transport.py`.
- `competition_mavlink_transport.py` remains receive-first; it only exposes the already-open connection for the explicit Phase 9D sender.
- `CompetitionRunner` remains no-send and still reports dry-run command candidates.
- `command_live` and `race` remain fail-closed.
