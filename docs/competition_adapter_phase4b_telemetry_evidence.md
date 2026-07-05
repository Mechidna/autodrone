# Competition Adapter Phase 4B Telemetry Evidence Review

Date: 2026-06-11.

Status: blocked pending live receive-only simulator evidence.

## Scope

Phase 4B decides whether the competition adapter can rely on live simulator
MAVLink telemetry for the internal z-up `VehicleState`.

This phase must be based on actual receive-only observe output from the
simulator. It must not assume `LOCAL_POSITION_NED` or `ODOMETRY` is usable just
because Phase 4A has handlers or unit fixtures for those messages.

## Evidence Available In This Workspace

No saved receive-only simulator observe output was present in the workspace at
the start of Phase 4B.

The Phase 4A observe tool exists at:

- `autonomy_core/tools/competition_mavlink_observe.py`

The Phase 4A state adapter skeleton exists at:

- `autonomy_core/core/competition_state_adapter.py`

Those files are not live simulator evidence. They only provide receive-only
tooling and deterministic adapter fixtures.

## Current Decision

Do not finalize the state adapter beyond the Phase 4A skeleton.

Do not proceed toward command-enabled work until one of these evidence-backed
conditions is true:

- `ODOMETRY` is observed from the live simulator with finite `x`, `y`, `z`,
  `vx`, `vy`, `vz`, and `q` fields at a usable rate and freshness.
- `LOCAL_POSITION_NED` is observed from the live simulator with finite `x`, `y`,
  `z`, `vx`, `vy`, and `vz` fields, and `ATTITUDE` is observed with finite yaw
  at a usable rate and freshness.

If neither condition is satisfied by live observe evidence, position is
unavailable for the current planner/controller assumptions and state estimation
becomes a P0 blocker before command enablement.

## Required Local Observe Command

Run this only when the simulator is running and accessible:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m autonomy_core.tools.competition_mavlink_observe --endpoint udpin:127.0.0.1:14550 --duration-s 30
```

Save the JSON output as a Phase 4B evidence artifact. The artifact should
include the full `message_types` object, not just the top-level availability
booleans.

## Evidence Review Checklist

For each observed message relevant to state, record:

- MAVLink message name and ID.
- System IDs and component IDs.
- Field names and sample field values.
- Timestamp fields such as `time_boot_ms` or `time_usec`.
- Observed receive rate.
- Whether fields required for position, velocity, and yaw are finite.
- Whether the observed rate and freshness are acceptable for runner-level
  command gating.

## Mapping Decision Rules

Prefer `ODOMETRY` if live evidence proves it is available and valid:

- Position mapping: MAVLink local NED `x`, `y`, `z` to internal
  `[x, y, -z]`.
- Velocity mapping: MAVLink local NED `vx`, `vy`, `vz` to internal
  `[vx, vy, -vz]`.
- Yaw mapping: extract yaw from MAVLink quaternion `q` in `w, x, y, z` order.

Use `LOCAL_POSITION_NED + ATTITUDE` only if live evidence proves both are
available and valid:

- Position mapping: MAVLink local NED `x`, `y`, `z` to internal
  `[x, y, -z]`.
- Velocity mapping: MAVLink local NED `vx`, `vy`, `vz` to internal
  `[vx, vy, -vz]`.
- Yaw mapping: MAVLink `ATTITUDE.yaw` as the internal yaw candidate, subject to
  live evidence confirming the simulator uses the expected radians convention.

Do not fabricate position from:

- Gates or visual detections.
- Race status or track metadata.
- Stale telemetry.
- Gazebo truth.
- Simulator-only diagnostic fields.

## Blocker Outcome If Evidence Is Missing Or Ambiguous

If live observe evidence does not prove usable local position/odometry, create a
separate P0 state-estimation plan before any command-enabled phase.

That plan may evaluate VIO/EKF or another state-estimation bridge, but it must
remain separate from the thin competition adapter branch unless explicitly
rescoped.
