# Competition Adapter Phase 6A Fake-Transport Runner Skeleton

Date: 2026-06-11.

Status: fake-transport skeleton added; live transport and command publication
remain blocked.

## Scope

Phase 6A adds an import-safe runner boundary that coordinates the existing
competition adapters through dependency injection. It is not live simulator
bring-up and it is not command readiness.

The runner skeleton must not:

- Open MAVLink sockets.
- Open UDP vision sockets.
- Send heartbeats, setpoints, actuator commands, attitude targets, or position
  targets.
- Run Gazebo or use Gazebo evidence.
- Run the competition simulator.
- Own or instantiate real `AutonomyAPI`.
- Call perception, YOLO, PnP, planner, or controller code directly.

## Implemented Boundary

`autonomy_core/runtime/competition_runner.py` defines:

- Runner modes: `observe`, `vision_dry_run`, `command_dry_run`,
  `command_live`, and `race`.
- Fail-closed safety gates for `command_live`, `race`, and any command
  publication flag.
- Injected fake transport hooks:
  - `receive_messages()` for fake MAVLink telemetry batches.
  - `receive_packets()` for fake vision packet batches.
- State processing through `CompetitionStateAdapter`.
- Vision packet processing through `CompetitionImageAdapter`.
- Optional fake-autonomy perception update calls only when an object is
  explicitly injected.
- Optional fake-autonomy `attitude_control()` calls only in `command_dry_run`
  and only after a usable fake state exists.
- Dry-run command candidate generation through
  `CompetitionDryRunCommandAdapter`.

## Command Publication Status

Command publication is always disabled in Phase 6A.

`command_dry_run` may build a deterministic dry-run command candidate with
`send_ready = false`, but the runner still records blocking reasons:

- `phase6a_no_command_publication`
- `phase4b_telemetry_evidence_missing`
- `command_dry_run_no_send`

`command_live` and `race` fail at startup in this phase.

## Phase 4B Dependency

Phase 4B remains blocked pending real receive-only telemetry evidence from the
competition simulator. Fake telemetry accepted by Phase 6A tests is only a
runner wiring fixture and must not be treated as competition telemetry
readiness.

## Test Coverage

`tests/test_competition_runner_skeleton.py` covers:

- Import safety without `pymavlink` or `cv2`.
- Fail-closed live command modes.
- Rejection of command publication flags.
- Rejection of Gazebo truth pose source.
- Fake telemetry observe-mode processing.
- Fake vision dry-run processing with fake JPEG decoding and fake autonomy.
- Fake command dry-run candidate generation with publication still blocked.
- No fake autonomy command call when state is unusable.
- Rejection of invalid injected transport objects instead of socket fallback.
