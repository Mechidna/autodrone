# Competition Adapter Phase 8.5A PX4/Gazebo Surrogate Harness

Status: scaffold implemented; no live PX4/Gazebo run performed.

Date: 2026-06-12

## Scope

Phase 8.5A adds an import-safe PX4/Gazebo surrogate harness scaffold for
runner wiring confidence only. It is not competition simulator evidence and
does not satisfy Phase 4B, Phase 9, telemetry readiness, command readiness,
race readiness, or competition readiness.

The harness is implemented in
`autonomy_core/runtime/px4_gazebo_surrogate_harness.py`.

## Surrogate-Only Rules

- All outputs are labeled `PX4/Gazebo surrogate only`.
- `phase4b_satisfied` is always `false`.
- `competition_readiness_claimed` is always `false`.
- PX4/MAVSDK telemetry inputs must be estimated telemetry only.
- Gazebo camera inputs may be used only as raw image pixels or pre-encoded
  JPEG bytes.
- Gazebo model pose, Gazebo camera pose, Gazebo TF/transform, link state,
  world pose, depth truth, and other Gazebo-truth metadata are rejected before
  runner use.
- `CompetitionRunner` is fed only through injected fake telemetry and vision
  transports.
- `command_live` and `race` remain fail-closed.
- No commands, heartbeats, setpoints, attitude targets, position targets,
  motor commands, or MAVLink commands are sent.

## Implemented Scaffold

The module provides:

- `Px4EstimatedTelemetrySample` for PX4/MAVSDK estimated local-NED telemetry.
- `SurrogateMavlinkMessage` for fake MAVLink-like `LOCAL_POSITION_NED`,
  `ATTITUDE`, and `HEARTBEAT` messages accepted by the existing state adapter
  and runner inventory.
- `telemetry_sample_to_fake_mavlink_messages(...)` and
  `telemetry_samples_to_fake_mavlink_messages(...)`.
- `packetize_vadr_jpeg_bytes(...)` for deterministic fake VADR `<IHHIIQ`
  packetization of already-encoded JPEG bytes.
- `packetize_vadr_frame_array(...)` for future explicit frame-array JPEG
  encoding with lazy `cv2` import only inside that called function.
- `InjectedTelemetryTransport` and `InjectedVisionPacketTransport` one-shot
  fake transports.
- `Px4GazeboSurrogateHarness` with:
  - `run_observe_surrogate(...)`
  - `run_vision_dry_run_surrogate(...)`
  - `run_command_dry_run_surrogate(...)`
  - `build_runner(...)`
  - `summary_dict()`
- `Px4GazeboSurrogateSummary` with:
  - `surrogate_label`
  - `frame_count`
  - `completed_packetized_frames`
  - `packetization_errors`
  - `telemetry_sample_count`
  - `stale_telemetry_count`
  - `command_candidate_count`
  - `command_blocked_reasons`
  - `guard_rejection_count`
  - `phase4b_satisfied = false`
  - `competition_readiness_claimed = false`

## What The Tests Prove

`tests/test_px4_gazebo_surrogate_harness.py` proves:

- Importing the harness does not load `pymavlink`, `cv2`, `mavsdk`, or `rclpy`.
- PX4 estimated local-NED telemetry converts into runner-accepted fake
  `LOCAL_POSITION_NED` and `ATTITUDE` messages.
- Gazebo-truth metadata in telemetry or vision inputs is rejected before
  runner use.
- Fake JPEG bytes packetize into valid VADR `<IHHIIQ` packets accepted by
  `CompetitionImageAdapter` with an injected fake decoder.
- Observe surrogate mode runs through injected fake telemetry.
- Vision dry-run surrogate mode runs through injected fake vision packets and
  injected fake autonomy.
- Command dry-run surrogate mode can produce a no-send command candidate while
  preserving Phase 6A and Phase 4B command-blocked reasons.
- `command_live` and `race` remain fail-closed.
- The summary never claims Phase 4B or competition readiness.

The tests do not require PX4, Gazebo, MAVSDK, ROS, `cv2`, `pymavlink`, YOLO
weights, GPU, network access, or simulator sockets.

## Future Manual PX4/Gazebo Run Requirements

A future manual surrogate run must record:

- Exact PX4/MAVSDK estimated telemetry sources and rates.
- Confirmation that no Gazebo pose, camera pose, TF, link state, depth truth,
  world pose, or model pose metadata was passed into the harness.
- Camera configuration evidence for `640x360`, `fx=fy=320`, `cx=320`,
  `cy=180`, zero distortion, same body/camera origin, and `20 deg` upward tilt.
- Frame rate, packetization errors, image adapter decode/completion counts,
  telemetry freshness, command candidate rate, command-blocked reasons, and
  guard rejection count.
- An explicit statement that results are PX4/Gazebo surrogate-only and do not
  satisfy Phase 4B or Phase 9 real competition simulator stages.
