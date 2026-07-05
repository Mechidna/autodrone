# Phase 9F.2: PX4/Gazebo Fixed-Rate Setpoint Stream Gate

Phase 9F.2 is a PX4/Gazebo surrogate-only diagnostic extension to Phase 9F. It
addresses the Phase 9F.1 result where perception, planning, command adaptation,
and MAVLink command sending all executed, but PX4/Gazebo saw sparse/stale
setpoint streaming and Offboard mode was not observed.

## Scope

- Uses the existing competition-stack perception, planning, and dry-run command
  adapter output.
- Sends only cached accepted dry-run `SET_ATTITUDE_TARGET` fields through the
  PX4/Gazebo surrogate command sender.
- Sends cached setpoints before the arm/offboard lifecycle gate in each bounded
  loop iteration.
- Requires an explicit positive Offboard prestream count when Offboard is
  requested.
- Keeps the command backend as the current attitude-angle quaternion
  `SET_ATTITUDE_TARGET` path.

## Safety Gates

Phase 9F.2 requires:

- `--px4-gazebo-full-autonomy-loop`
- `--ack-px4-gazebo-surrogate-full-autonomy-loop`
- `--px4-gazebo-command-send`
- `--ack-px4-gazebo-surrogate-command-send`
- `--px4-gazebo-continuous-setpoint-stream`
- `--px4-gazebo-fixed-rate-setpoint-stream`
- `--ack-px4-gazebo-surrogate-fixed-rate-setpoint-stream`
- `--px4-gazebo-offboard-prestream-count` greater than zero when Offboard is
  requested
- setpoint stream rate greater than zero and below `100 Hz`

## What It Does Not Claim

- It does not satisfy Phase 4B.
- It does not satisfy real competition simulator telemetry evidence.
- It does not claim competition command readiness, race readiness, submitted-run
  readiness, or competition readiness.
- It does not change planner, controller, YOLO, PnP, perception, transform
  selection, command adapter semantics, or race progression.
- It does not enable `command_live` or `race`.

## Success Criteria

The JSON summary must report:

- `phase == "9F.2"`
- fixed-rate stream requested and acknowledged
- fixed-rate stream attempt and sent counts greater than zero
- fixed-rate stream rejection count is zero
- command send rejection count is zero
- stale cached setpoint rejection count is zero
- Offboard prestream count is positive and satisfied
- all normal Phase 9F receive, perception, command candidate, arm/offboard, and
  send gates pass
- `phase9f2_fixed_rate_setpoint_stream_satisfied=true`
- `phase4b_satisfied=false`
- `competition_readiness_claimed=false`

## Example Command

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m autonomy_core.runtime.competition_main \
  command_dry_run \
  --live-transports \
  --mavlink-endpoint udpin:0.0.0.0:14540 \
  --vision-bind-host 0.0.0.0 \
  --vision-port 5600 \
  --use-real-autonomy \
  --real-perception \
  --allow-legacy-yolo-default \
  --px4-gazebo-full-autonomy-loop \
  --ack-px4-gazebo-surrogate-full-autonomy-loop \
  --px4-gazebo-command-send \
  --ack-px4-gazebo-surrogate-command-send \
  --px4-gazebo-continuous-setpoint-stream \
  --px4-gazebo-fixed-rate-setpoint-stream \
  --ack-px4-gazebo-surrogate-fixed-rate-setpoint-stream \
  --px4-gazebo-arm \
  --ack-px4-gazebo-surrogate-arm \
  --px4-gazebo-offboard \
  --ack-px4-gazebo-surrogate-offboard \
  --px4-gazebo-offboard-prestream-count 20 \
  --px4-gazebo-command-max-count 500 \
  --px4-gazebo-command-max-age-s 2.0 \
  --px4-gazebo-setpoint-stream-hz 20 \
  --px4-gazebo-setpoint-stream-burst-limit 2 \
  --steps 600 \
  --step-sleep-s 0.02
```

Run the Phase 8.5E surrogate vision bridge in a separate terminal so UDP `5600`
is populated with VADR-style packets.
