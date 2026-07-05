# Phase 9C: Command Candidate Dry-Run, No Send

Status: implemented as a no-send dry-run boundary.

## Purpose

Phase 9C runs the competition receive/perception/planning/control path far enough
to build would-be command fields through `competition_command_adapter.py`, while
keeping all command publication disabled.

This phase answers one question only:

- Given fresh competition-style telemetry and decoded competition-style images,
  can the competition stack produce a deterministic no-send `SET_ATTITUDE_TARGET`
  candidate?

It does not prove PX4, Gazebo, or the real competition simulator accepts the
command.

## Runtime Path

The Phase 9C CLI path is:

```bash
python3 -B -m autonomy_core.runtime.competition_main \
  command_dry_run \
  --live-transports \
  --use-real-autonomy \
  --real-perception \
  --allow-legacy-yolo-default \
  --mavlink-endpoint udpin:0.0.0.0:14540 \
  --vision-bind-host 0.0.0.0 \
  --vision-port 5600 \
  --steps 1000 \
  --step-sleep-s 0.02
```

Expected external setup for PX4/Gazebo surrogate testing:

- PX4/Gazebo publishes MAVLink telemetry to the configured endpoint.
- `surrogate_vision_bridge.py` emits Gazebo camera frames as VADR-style UDP
  JPEG packets to `127.0.0.1:5600`.
- The Gazebo camera is already configured for the official `640x360` VADR
  resolution, so no surrogate resize is expected for the current setup.

## Safety Properties

Phase 9C keeps these safety gates closed:

- `command_publication_allowed=false`
- `command_sent_count=0`
- no heartbeat, setpoint, actuator, arm, offboard, reset, attitude target, or
  position target is sent by `competition_main.py` or `competition_runner.py`
- `command_live` and `race` remain fail-closed
- Phase 4B remains blocked pending real receive-only competition simulator
  telemetry evidence
- competition readiness is not claimed

## What Was Added

Phase 9C adds no-send command-candidate accounting to the competition runtime:

- `command_dry_run --real-perception` is labeled as `phase="9C"`.
- The runner copies usable `CompetitionStateAdapter` output into the injected
  `AutonomyAPI.telemetry` object before planning/control in command dry-run.
- The runner attempts `AutonomyAPI.path_plan(replan_reason="phase9c_command_dry_run")`
  when that method is available.
- The runner then calls `AutonomyAPI.attitude_control()` and passes the returned
  tuple to `CompetitionDryRunCommandAdapter`.
- The JSON summary reports planning counts, command candidate counts, accepted
  and rejected candidates, and the last dry-run `SET_ATTITUDE_TARGET` fields.

## Success Criteria

A Phase 9C run reports `phase9c_command_dry_run_satisfied=true` only when all of
these are true:

- mode is `command_dry_run`
- live receive transports were explicitly requested
- the competition-safe `AutonomyAPI` profile was requested
- official `competition_official_ned` transform mode was selected
- telemetry receive criteria passed
- perception boundary criteria passed
- autonomy telemetry sync happened at least once
- `path_plan(...)` was attempted and succeeded at least once
- at least one command candidate was attempted and accepted by the dry-run
  command adapter
- the last command candidate is a no-send `SET_ATTITUDE_TARGET`
- command publication remained disabled
- command sent count remained zero
- Phase 4B and competition readiness remain false

If `path_plan(...)` fails because no stable perception target exists, Phase 9C
can still show a command candidate, but `phase9c_command_dry_run_satisfied` must
remain false until the planning success criterion passes.

## Deterministic Tests

Tests are in `tests/test_competition_main.py` and use fake injected transports,
fake image adapter, and fake autonomy. They do not instantiate real `AutonomyAPI`,
run YOLO, open sockets, run PX4/Gazebo, or send commands.

The focused test verifies:

- Phase 9C is selected for `command_dry_run --real-perception`.
- competition state is copied into the autonomy telemetry object.
- `path_plan(...)` is called with `phase9c_command_dry_run`.
- `attitude_control()` is called once.
- a dry-run `SET_ATTITUDE_TARGET` field dictionary is emitted.
- `send_ready=false`, `command_publication_allowed=false`, and
  `command_sent_count=0`.

## Remaining Blockers

Phase 9C does not remove these blockers:

- Phase 4B is still blocked until real competition simulator receive-only
  telemetry evidence is available.
- PX4/Gazebo surrogate output is still surrogate-only confidence evidence.
- A real competition simulator or official equivalent must still validate
  telemetry availability, command semantics, and command acceptance before any
  live command phase.
