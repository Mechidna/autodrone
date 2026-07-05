# Competition Adapter Phase 5A Command Semantics Inventory

Date: 2026-06-11.

Status: dry-run boundary prepared; live command publication remains blocked.

## Scope

Phase 5A documents command semantics and adds a dry-run-only command adapter
skeleton. It does not enable command sending, sockets, a runner mode, Gazebo
evidence, or competition readiness.

Phase 4B remains blocked pending real receive-only telemetry evidence from the
competition simulator. That block also prevents command-enabled work.

## Existing AutonomyAPI Tuple Semantics

`AutonomyAPI.attitude_control()` returns:

- `roll_cmd`: attitude angle in radians.
- `pitch_cmd`: attitude angle in radians.
- `yaw_cmd`: yaw attitude angle in radians.
- `thrust_cmd`: normalized collective thrust in the existing controller
  convention.

Evidence:

- `RPGHighLevelTracker.update(...)` returns `roll_des`, `pitch_des`, `yaw_cmd`,
  and `thrust_cmd`.
- `AutonomyAPI.attitude_control()` and no-target hold paths return that
  four-tuple after current sign handling for roll and pitch.
- `px4_runner.py` converts roll, pitch, and yaw from radians to degrees before
  constructing `mavsdk.offboard.Attitude`.
- `px4_runner.py` passes `thrust_cmd` through as `thrust_value`.

Decision:

- Treat the tuple as attitude-angle commands plus normalized thrust.
- Do not treat the tuple as body roll, pitch, or yaw rates.
- Do not retune gains, yaw policy, hover thrust, thrust limits, or no-target
  behavior in the command adapter.

## Read-Only Reference Command Evidence

`third_party/PyAIPilotExample/controller.py` demonstrates:

- `set_attitude_target_send(...)`.
- Quaternion order `w, x, y, z`.
- Body rates in radians per second.
- Normalized thrust `0.0 .. 1.0`.
- Type mask `ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE`.
- Dummy quaternion `[1, 0, 0, 0]` because attitude is ignored in that example.

That reference example is body-rate mode. It does not prove the competition
simulator will accept the current AutonomyAPI attitude-angle tuple directly.

## Phase 5A Dry-Run Adapter Decision

The dry-run adapter prepares deterministic `SET_ATTITUDE_TARGET`-style fields
for the documented AutonomyAPI tuple:

- Message name: `SET_ATTITUDE_TARGET`.
- Time field: caller-provided `time_boot_ms`.
- Target IDs: caller-provided `target_system` and `target_component`.
- Type mask: ignore body roll, pitch, and yaw rates.
- Quaternion: generated from roll, pitch, and yaw attitude angles in MAVLink
  `w, x, y, z` order.
- Body rates: zeroed because the source tuple is not a body-rate command.
- Thrust: preserved as normalized `0.0 .. 1.0`.
- Send status: always `send_ready = False`.

This is a candidate dry-run representation only. It is not live command
readiness.

## Send Blockers

Command sending remains blocked until all are true:

- Phase 4B real receive-only observe evidence proves usable competition
  telemetry for state.
- A later phase explicitly enables send behavior.
- The simulator accepts and documents the selected `SET_ATTITUDE_TARGET`
  semantics, including quaternion/body-rate/type-mask behavior.
- Command units, target IDs, timestamp semantics, and rate limits are validated
  against real competition simulator behavior or a known-good packet capture.

## Explicit Non-Evidence

Do not use any of these as proof of command readiness:

- Gazebo behavior.
- Phase 4A fake telemetry fixtures.
- Presence of handlers in `PyAIPilotExample`.
- Race status or track metadata.
- Visual detections.
- Stale telemetry.
- The Phase 5A dry-run adapter itself.
