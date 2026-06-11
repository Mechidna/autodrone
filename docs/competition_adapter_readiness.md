# Competition Adapter Readiness Checkpoint

Phase 12 checkpoint. This document records whether the refactor has created enough seams to begin a competition-adapter planning branch. It is a checklist only and does not authorize runtime behavior changes.

## Scope And Compatibility Position

- `AutonomyAPI` remains the runtime facade.
- `px4_runner.py` remains compatible with the existing facade fields and command tuple.
- `flight_logger.py` remains compatible with the existing logger-visible fields.
- No competition adapter, MPC tracker, EKF/VIO, logging-mode switch, or behavior change is implemented by this checkpoint.
- The current simulation/GT diagnostics behavior remains available and should be explicitly disabled or isolated in a later production/competition profile.

## 1. State Seam

Status: present enough for adapter planning.

- `autonomy_core/core/types.py` defines shared passive contracts including `State`, `VehicleState`, `Reference`, `ReferenceState`, `ControlCommand`, `CameraModel`, `CameraFrame`, `GateTarget`, and `PlanDebugInfo`.
- `autonomy_core/core/state_adapter.py` defines `vehicle_state_from_telemetry(telemetry)`.
- `AutonomyAPI.attitude_control()` uses the telemetry adapter path while preserving the current telemetry object and internal z-up convention.
- `AutonomyAPI` still accepts and owns the current `GetTelemetry` object, so runner compatibility is preserved.

Remaining adapter work:

- Add a competition telemetry/state adapter that produces `VehicleState` without depending on `GetTelemetry` internals.
- Decide where external timestamps and estimator health should live; current `VehicleState` is intentionally minimal.

## 2. Camera / Frame Seam

Status: partial seam present, transport adapter still needed.

- `CameraFrame` and `CameraModel` exist in `autonomy_core/core/types.py`.
- `AutonomyAPI.update_gate_memory_from_frame(...)` remains the public perception wrapper and still owns the current competition-facing image/camera input shape: frame, camera matrix, distortion coefficients, image timestamps, optional image pose snapshot, and optional Gazebo pose.
- `autonomy_core/perception/gate_pipeline.py` contains only low-risk detection-flow debug formatting helpers.

Remaining adapter work:

- Add a competition image transport receiver that converts incoming image chunks and camera metadata into the current `update_gate_memory_from_frame(...)` arguments or a `CameraFrame` wrapper.
- Define a production camera model source that does not depend on Gazebo or local calibration assumptions.
- Keep frame-convention changes out of the adapter unless explicitly tested; transform behavior remains in `AutonomyAPI` and perception modules.

## 3. Command Seam

Status: facade-compatible, not yet adapter-ready.

- `AutonomyAPI.attitude_control()` still returns the exact tuple `(roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd)`.
- `ControlCommand` exists in `autonomy_core/core/types.py`, but runtime command output has not been rewired to use it.
- Command mapping and tracker ownership remain in `AutonomyAPI` and the current `RPGHighLevelTracker` path.
- Phase 7R intentionally kept command-producing yaw/no-target/control helpers inside `AutonomyAPI` after the failed Phase 7 behavior-preservation attempt.

Remaining adapter work:

- Add a competition command adapter that wraps the current tuple into the required competition protocol without changing `attitude_control()` behavior.
- Do not replace the tracker or add MPC until a separate behavior branch with sim acceptance is planned.

## 4. Planning Seam

Status: useful seams present; `path_plan(...)` intentionally remains the facade.

Extracted modules:

- `autonomy_core/planning/plan_validator.py`: minimum-snap geometric and z-corridor validation helpers.
- `autonomy_core/planning/trajectory_manager.py`: low-risk trajectory/reference helpers such as segment timing and active-target crossing tau.
- `autonomy_core/planning/suffix_planner.py`: pending suffix state and install helpers.
- `autonomy_core/planning/target_horizon.py`: horizon-building facade helpers.

Current facade behavior:

- `AutonomyAPI.path_plan(...)` remains the public wrapper for planning and still owns planner solve/install behavior.
- Planner solve/install logic, validation fallback semantics, virtual-clock behavior, suffix behavior, and planner reference timing remain behavior-compatible with the pre-refactor facade.

Remaining adapter work:

- Keep `path_plan(...)` as the integration point for a competition adapter until planner state ownership is explicitly separated.
- Add planner-profile configuration only after default parity tests exist.

## 5. Race / Target Seam

Status: partial seam present; high-risk progression logic remains inside `AutonomyAPI`.

Extracted modules:

- `autonomy_core/planning/target_validation.py`: target validation, duplicate checks, canonical track ID helpers.
- `autonomy_core/racing/race_admission.py`: landmark merge and race-admission helpers.
- `autonomy_core/racing/gate_advancement.py`: small gate crossing / target-clear helpers only.

Still tightly coupled inside `AutonomyAPI`:

- `advance_gate_if_needed(...)` remains in `AutonomyAPI` because it owns race cursor updates, active gate index updates, same-plan continuation, pending suffix install priority, and full replan fallback.
- `_continue_existing_plan_after_completion(...)` remains in `AutonomyAPI` because it is coupled to installed plan timing and active target metadata.
- Post-completion fallback helpers remain in `AutonomyAPI` because they mutate race order/admission and GT/perception fallback state.

Remaining adapter work:

- Treat `advance_gate_if_needed(...)` as a guarded behavior seam, not a refactor seam.
- Any gate-opening or low-crossing validation is behavior work and should happen in a separate bugfix branch.

## 6. Perception Seam

Status: limited pipeline seam present; live perception remains coupled by design.

- `autonomy_core/perception/gate_pipeline.py` contains detection-flow debug initialization/finalization helpers.
- `AutonomyAPI.update_gate_memory_from_frame(...)` still owns YOLO, PnP, transform, GateMemory update/admission, Gazebo truth diagnostic mixing, and logger-visible perception fields.
- Full `process_frame(...)`, PnP candidate selection, transform conventions, GateMemory admission behavior, and GT diagnostic behavior were intentionally not moved.

Remaining adapter work:

- Add an input adapter around `update_gate_memory_from_frame(...)` first, not a deep perception rewrite.
- Separate production perception from Gazebo-only diagnostics only after preserving log fields and sim behavior.
- Keep YOLO model/threshold changes and domain randomization out of adapter scaffolding.

## 7. Debug / Gazebo Seam

Status: improved isolation, but Gazebo behavior still exists in runtime paths.

- `autonomy_core/debug/gazebo_diagnostics.py` contains Gazebo pose conversion and comparison debug helpers.
- `AutonomyAPI` keeps wrappers for the moved Gazebo diagnostic methods.
- Gazebo truth diagnostics are more isolated than before, but `AutonomyAPI` still contains sim-only branches and fields, including the current default `perception_world_pose_source = "gazebo_truth_sim_only"`.

Remaining adapter work:

- Add a runtime profile or explicit production guard that prevents Gazebo truth pose-source usage in real competition runs.
- Keep transform convention changes separate from adapter work.
- Do not remove Gazebo diagnostics until a replacement debug/profile strategy exists.

## 8. Config / Logging Seam

Status: scaffolding present, not authoritative.

- `autonomy_core/core/config.py` defines `VehicleLimitsConfig`, `ControllerConfig`, `PlannerConfig`, `PerceptionConfig`, `LoggingConfig`, and `RuntimeConfig`.
- The config classes mirror current defaults but are not wired into runtime code.
- `LoggingConfig` defines future modes: `debug_full`, `debug_light`, `race`, and `off`.
- `FlightLogger` behavior, logging columns, sidecar plan logging, and debug-frame behavior are unchanged.

Remaining adapter work:

- Wire config into `AutonomyAPI` only with explicit default parity checks.
- Add logging modes as an opt-in behavior change after schema compatibility is defined.

## 9. Compatibility Summary

- `AutonomyAPI` remains the single facade used by runner and logger code.
- `px4_runner.py` was not changed by this checkpoint.
- `flight_logger.py` was not changed by this checkpoint.
- Public method names and return shapes needed by runner/logger remain available.
- The refactor intentionally preserved behavior; sim acceptance was used phase-by-phase before this checkpoint.

## 10. Known Remaining Risks And TODOs

- Low gate crossing / gate-opening validation remains behavior work, not refactor work.
- MPC tracker remains future work.
- Competition API adapter remains future work.
- VIO/EKF remains future work.
- YOLO/domain randomization remains future work.
- Gazebo truth defaults must be reviewed before any real competition run.
- `advance_gate_if_needed(...)` is still high-risk and should not be modified casually.

## Readiness Decision

The refactor has created enough seams to start a competition-adapter planning branch. The next branch should be adapter-first and behavior-preserving: wrap the existing facade inputs/outputs before changing planner, perception, control, or logging internals.

Top blockers before actual competition integration:

1. Define and implement the competition image/metadata transport adapter into `update_gate_memory_from_frame(...)` or `CameraFrame`.
2. Define and implement the competition command transport adapter from `(roll, pitch, yaw, thrust)` without changing controller behavior.
3. Disable or profile-guard Gazebo truth pose-source defaults for non-sim runs.
4. Add config wiring with default parity tests before using runtime profiles or logging modes.
5. Add state-estimator/VIO/EKF integration planning without changing the current telemetry adapter path.
