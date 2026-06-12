# Competition Adapter Phase 8 Deterministic Tests

Date: 2026-06-12.

Status: implemented for the deterministic, no-live-transport adapter boundary.

## Scope

Phase 8 consolidates deterministic fixture coverage for the competition
adapter boundary. These tests do not require the competition simulator, Gazebo,
PX4, MAVSDK, live MAVLink, UDP sockets, GPU, YOLO weights, or network access.

No live transport, command publication, `command_live`, or `race` behavior was
added.

## Fixture Coverage

Config constants:

- `tests/test_competition_config.py`
- Covers official VADR-TS-002 constants, camera matrix, zero distortion, rates,
  command-rate ceiling, camera tilt, geometry constants, vision header fields,
  and separation from `PyAIPilotExample` reference defaults.

Vision packet and image adapter:

- `tests/test_competition_image_adapter.py`
- Covers exact little-endian `<IHHIIQ` header parsing, `24` byte header size,
  out-of-order chunk reassembly, official camera metadata, timestamp fields,
  missing chunks, corrupt JPEG payloads, duplicate chunks, wrong resolution,
  stale incomplete frames, and bounded incomplete-frame memory.

Camera model and frame conventions:

- `tests/test_competition_camera_model.py`
- Covers official intrinsics, zero distortion, zero translation, MAVLink/body
  and OpenCV camera conventions, `20 deg` upward tilt, inverse rotations,
  projection to principal point, and a projection sanity check that catches the
  common tilt sign inversion.

State adapter:

- `tests/test_competition_state_adapter.py`
- Covers MAVLink local-NED position and velocity conversion to internal z-up
  `VehicleState`, `LOCAL_POSITION_NED + ATTITUDE`, `ODOMETRY`, quaternion yaw,
  nonfinite rejection, missing position rejection, stale telemetry rejection,
  and source preference.

MAVLink observe tooling:

- `tests/test_competition_mavlink_observe.py`
- Covers import safety, telemetry inventory fields, message IDs, source IDs,
  rates, timestamp fields, nonblocking receive-only behavior, no sends, and
  `BAD_DATA` filtering.

Command adapter:

- `tests/test_competition_command_adapter.py`
- Covers the Phase 5A decision that the existing AutonomyAPI tuple is
  roll/pitch/yaw attitude angles in radians plus normalized thrust, not body
  rates.
- Covers dry-run `SET_ATTITUDE_TARGET`-style fields: quaternion `w, x, y, z`,
  body-rate ignore mask, target IDs, timestamps, sequence, normalized thrust,
  and `send_ready = false`.
- Covers nonfinite tuple values, invalid tuple length, invalid thrust, invalid
  target/timestamp fields, invalid clock values, and strict below-`100 Hz`
  dry-run rate limiting.
- Documents the `PyAIPilotExample` body-rate example as non-equivalent
  reference evidence, not as the active tuple mapping.

Gazebo guard:

- `tests/test_competition_gazebo_guard.py`
- Covers rejection of `gazebo_truth_sim_only`, non-`None` `gazebo_pose`, Gazebo
  model/camera/TF/transform metadata, pose snapshots, and enabled
  Gazebo-only far-depth diagnostic flags.
- Covers passive behavior when `competition_mode = false`.

Runner skeleton:

- `tests/test_competition_runner_skeleton.py`
- Covers import safety without `pymavlink` or `cv2`, observe mode with fake
  heartbeat/telemetry, vision dry-run with fake image packets, command dry-run
  with fake telemetry plus fake image plus dry-run command candidate, no-send
  behavior, command-blocked reasons, invalid state gating, invalid injected
  transport rejection, and fail-closed `command_live`/`race`.

## Known Boundaries

- Phase 8 tests do not prove real competition simulator telemetry availability.
- Phase 8 tests do not prove the real simulator accepts any
  `SET_ATTITUDE_TARGET` packet semantics.
- Stale live-image command eligibility is a future runner/freshness gate; Phase
  8 only covers current deterministic runner state gating and image adapter
  stale incomplete-frame handling.
- Phase 8.25 geometry audit remains a separate future phase.
- Phase 8.5 PX4/Gazebo surrogate harness remains a separate future phase.

## Phase 4B Status

Phase 4B remains blocked pending real receive-only telemetry evidence from the
competition simulator. Phase 8 deterministic fixtures do not complete Phase 4B
and do not claim telemetry readiness, command readiness, or race readiness.
