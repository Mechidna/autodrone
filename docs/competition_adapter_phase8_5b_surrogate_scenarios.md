# Competition Adapter Phase 8.5B Surrogate Scenario Matrix

Status: deterministic surrogate scenario fixtures implemented; no live
PX4/Gazebo run performed.

Date: 2026-06-12

## Scope

Phase 8.5B extends the Phase 8.5A PX4/Gazebo surrogate harness with
deterministic multi-step scenario fixtures. These fixtures provide regression
coverage for runner wiring, packetization counts, telemetry conversion counts,
guard rejection, and no-send command dry-run behavior.

This remains surrogate-only evidence. It does not satisfy Phase 4B, Phase 9,
competition telemetry readiness, command readiness, race readiness, or
competition readiness.

## Implemented Scenario Support

`autonomy_core/runtime/px4_gazebo_surrogate_harness.py` now includes:

- `Px4GazeboSurrogateScenarioStep`
- `Px4GazeboSurrogateScenario`
- `Px4GazeboSurrogateScenarioResult`
- `SurrogateDecodedFrame`
- `deterministic_surrogate_jpeg_decoder(...)`
- `Px4GazeboSurrogateHarness.run_scenario(...)`

Scenario execution:

- Calls the existing `CompetitionRunner.step(...)` path.
- Uses explicit injected fake telemetry batches and fake VADR vision packets.
- Does not add new `CompetitionRunner` modes.
- Does not open sockets.
- Does not import or require MAVSDK, ROS, `pymavlink`, `cv2`, PX4, Gazebo,
  YOLO weights, GPU, or network access.
- Does not call real `AutonomyAPI` or real perception.
- Uses a deterministic fake JPEG decoder by default for scenario tests.
- Keeps `command_dry_run` no-send.
- Keeps `command_live` and `race` fail-closed.

## Deterministic Checks

The scenario helper validates:

- Strictly increasing scenario wall-clock timestamps.
- Monotonic telemetry `time_boot_ms` values.
- Monotonic provided image `sim_time_ns` values.
- Positive frame periods.
- PX4/MAVSDK estimated telemetry conversion into runner-accepted fake messages.
- Fake JPEG byte packetization into VADR `<IHHIIQ` packets.
- Runner mode used.
- Command dry-run no-send behavior.
- `phase4b_satisfied = false`.
- `competition_readiness_claimed = false`.
- Gazebo-truth metadata rejection before runner use.

## Scenario Fixtures

`tests/test_px4_gazebo_surrogate_scenarios.py` covers:

- Multi-step observe scenario with several fake PX4 estimated telemetry samples.
- Multi-frame `vision_dry_run` scenario with deterministic fake VADR packets.
- `command_dry_run` scenario with a no-send command candidate.
- Mixed telemetry plus image `command_dry_run` scenario with deterministic
  summary counts.
- Rejection of non-monotonic wall-clock and telemetry timestamps.
- Rejection of Gazebo-truth metadata before runner use.
- Fail-closed `command_live` and `race` scenario attempts.
- Import safety without loading `pymavlink`, `cv2`, `mavsdk`, or `rclpy`.

## Explicit Non-Claims

- No live competition MAVLink transport was added.
- No live UDP vision transport was added.
- No UDP port was opened.
- No PX4, Gazebo, MAVSDK, ROS, or simulator process was started.
- No heartbeats, setpoints, attitude targets, position targets, motor commands,
  or MAVLink commands were sent.
- No protected vendor files were modified.
- Phase 4B remains blocked pending real receive-only competition simulator
  telemetry evidence.
- Phase 9 real competition simulator stages were not started.
