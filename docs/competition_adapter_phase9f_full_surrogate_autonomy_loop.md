# Phase 9F: PX4/Gazebo Full Surrogate Autonomy Loop

Phase 9F labels and gates the first full PX4/Gazebo surrogate loop through the
competition stack:

- PX4/Gazebo estimated telemetry enters through `competition_mavlink_transport.py`.
- Gazebo camera frames are packetized by `surrogate_vision_bridge.py` into VADR
  UDP vision packets on port `5600`.
- `competition_vision_transport.py` receives those UDP packets.
- `competition_image_adapter.py` reassembles and decodes VADR JPEG frames.
- `CompetitionRunner` calls the competition-safe `AutonomyAPI` profile with
  `gazebo_pose=None` and `image_pose_snapshot=None`.
- `AutonomyAPI.path_plan()` and `AutonomyAPI.attitude_control()` generate a
  command candidate.
- The competition dry-run command adapter converts that candidate to
  `SET_ATTITUDE_TARGET` attitude-angle fields.
- The PX4/Gazebo-only command sender publishes those fields to the local SITL
  MAVLink connection when all explicit surrogate safety gates are enabled.

This phase is surrogate-only. It does not satisfy Phase 4B, real competition
simulator telemetry evidence, race readiness, submitted-run readiness, or
competition readiness.

## Command Backend

Phase 9F intentionally uses the current competition dry-run command adapter
backend:

- MAVLink message: `SET_ATTITUDE_TARGET`
- Type mask: `7`
- Semantics: attitude quaternion is used; body roll/pitch/yaw rates are ignored
- Source tuple: current `AutonomyAPI.attitude_control()` roll/pitch/yaw/thrust

The PyAIPilotExample body-rate command protocol remains a separate future
decision. Phase 9F does not convert autonomy output into body rates.

## Required Gates

`competition_main.py` requires all of these for Phase 9F:

- `command_dry_run` mode
- `--live-transports` or injected test components
- `--use-real-autonomy`
- `--real-perception`
- `--allow-legacy-yolo-default` until the YOLO path is externalized
- `--px4-gazebo-full-autonomy-loop`
- `--ack-px4-gazebo-surrogate-full-autonomy-loop`
- `--px4-gazebo-command-send`
- `--ack-px4-gazebo-surrogate-command-send`
- `--px4-gazebo-continuous-setpoint-stream`
- `--px4-gazebo-arm`
- `--ack-px4-gazebo-surrogate-arm`
- `--px4-gazebo-offboard`
- `--ack-px4-gazebo-surrogate-offboard`

Phase 9F rejects fixed Phase 9E.3/9E.4 smoke command modes.

## Phase 9F.1 Debug Yaw Override

Phase 9F.1 is a PX4/Gazebo surrogate-only diagnostic layer on top of Phase 9F.
It exists because the first Phase 9F manual run showed the competition controller
commanding yaw away from the gate-facing direction while the local PX4/Gazebo
vehicle was otherwise receiving the full-loop command stream.

Phase 9F.1 adds:

- `--px4-gazebo-debug-yaw-override-rad <yaw>`
- `--ack-px4-gazebo-surrogate-debug-yaw-override`

The override is applied only at the PX4/Gazebo MAVLink sender boundary. It
preserves the autonomy-computed roll, pitch, and thrust, decomposes the outgoing
dry-run quaternion, replaces yaw with the explicit debug value, and rebuilds the
quaternion before send.

It does not change:

- `AutonomyAPI.attitude_control()` output
- planner behavior
- controller behavior
- perception behavior
- competition command-adapter semantics
- real competition command readiness

JSON fields identify the override:

- `phase == "9F.1"` when the override is enabled
- `px4_gazebo_debug_yaw_override_requested`
- `px4_gazebo_debug_yaw_override_acknowledged`
- `px4_gazebo_debug_yaw_override_rad`
- `px4_gazebo_debug_yaw_override_applied_count`
- `last_command_send_result.debug_yaw_original_rad`
- `last_command_send_result.debug_yaw_override_rad`
- `last_command_send_result.debug_yaw_override_applied`

## Manual PX4/Gazebo Test

Terminal 1: start PX4/Gazebo with the VADR-like 640x360 camera and MAVLink
streaming to `14540`.

Terminal 2: start the surrogate vision bridge for enough frames to cover the
run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m autonomy_core.runtime.surrogate_vision_bridge \
  --camera-topic /camera \
  --send-host 127.0.0.1 \
  --send-port 5600 \
  --frames 1200 \
  --timeout-s 5
```

Terminal 3: run the gated full surrogate loop:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m autonomy_core.runtime.competition_main \
  command_dry_run \
  --live-transports \
  --mavlink-endpoint udpin:0.0.0.0:14540 \
  --vision-bind-host 0.0.0.0 \
  --vision-port 5600 \
  --steps 1200 \
  --step-sleep-s 0.02 \
  --use-real-autonomy \
  --real-perception \
  --allow-legacy-yolo-default \
  --px4-gazebo-full-autonomy-loop \
  --ack-px4-gazebo-surrogate-full-autonomy-loop \
  --px4-gazebo-command-send \
  --ack-px4-gazebo-surrogate-command-send \
  --px4-gazebo-continuous-setpoint-stream \
  --px4-gazebo-command-max-count 1200 \
  --px4-gazebo-arm \
  --ack-px4-gazebo-surrogate-arm \
  --px4-gazebo-offboard \
  --ack-px4-gazebo-surrogate-offboard
```

Optional Phase 9F.1 yaw-forcing diagnostic, for example gate-facing yaw near
`90 deg`:

```bash
  --px4-gazebo-debug-yaw-override-rad 1.57079632679 \
  --ack-px4-gazebo-surrogate-debug-yaw-override
```

Use `Ctrl+C` if the vehicle behavior is unsafe. This path can move the local
PX4/Gazebo vehicle.

## JSON Success Criteria

Expected high-level fields:

- `phase == "9F"`
- `status == "dry_run_complete"`
- `px4_gazebo_full_autonomy_loop_requested == true`
- `px4_gazebo_full_autonomy_loop_acknowledged == true`
- `phase9f_command_backend == "attitude_angle_quaternion_set_attitude_target"`
- `phase9f_command_type_mask == 7`
- `heartbeat_seen == true`
- `state_usable == true`
- `vision_frames_completed > 0`
- `perception_update_calls > 0`
- `planning_success_count > 0`
- `command_candidate_accepted_count > 0`
- `command_sent_count > 0`
- `command_send_rejection_count == 0`
- `setpoint_stream_stale_rejection_count == 0`
- `phase9f_command_send_rate_hz > 10.0`
- `phase9f_vision_frame_rate_hz > 5.0`
- `phase9f_max_send_gap_s <= 0.5`
- `armed_state_observed == true`
- `offboard_state_observed == true`
- `phase9f_full_autonomy_loop_satisfied == true`
- `phase4b_satisfied == false`
- `competition_readiness_claimed == false`

Manual validation still matters. Phase 9F does not automatically score whether
the vehicle passed all three gates; it only proves that the PX4/Gazebo surrogate
can drive the competition-stack perception/planning/command-send loop.
