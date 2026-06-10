# AutonomyAPI Refactor Plan

## Purpose

Refactor `autonomy_core/launch/autonomy_api6.py` from a large monolithic runtime file into smaller modules with clear boundaries while preserving current behavior.

This refactor prepares the stack for:

- competition interfacing
- MPC integration
- logging modes
- vehicle and camera configuration
- state-source abstraction for MAVSDK, competition MAVLink, VIO/EKF, and simulated sources
- isolating Gazebo truth diagnostics from production decision logic

This plan is staged intentionally. It is not a rewrite, not a tuning pass, and not the competition adapter implementation.

## Non-Negotiable Constraints

- Preserve behavior unless a phase explicitly says otherwise.
- Keep `AutonomyAPI` as the public facade during the refactor.
- Keep `autonomy_core/tools/px4_runner.py` compatible during every phase.
- Keep `autonomy_core/tools/flight_logger.py` compatible during every phase.
- Do not change planner math.
- Do not change controller gains or control laws.
- Do not change frame conventions during extraction phases.
- Do not change perception thresholds, YOLO settings, PnP selection, or gate-memory admission behavior.
- Do not remove Gazebo truth diagnostics yet.
- Do not let Gazebo truth diagnostics become more coupled to production decision logic.
- Do not implement MPC during this refactor.
- Do not implement the competition API adapter until the state, camera, and command seams exist.
- Do not implement VIO/EKF/state estimation during this refactor.
- Do not introduce broad formatting-only diffs.
- Implement one phase per Codex turn unless explicitly instructed otherwise.
- Stop before editing if `git status --short` shows unexpected source changes not made by the current task.
- Stop immediately if a runtime Python file changes unexpectedly while working.

## Current Files Inspected For This Plan

- `autonomy_core/launch/autonomy_api6.py`
- `autonomy_core/tools/px4_runner.py`
- `autonomy_core/tools/flight_logger.py`
- `autonomy_core/controller/attitude_controller3.py`
- `autonomy_core/planning/minimum_snap_planner_multi_time_optimized.py`
- `autonomy_core/perception/gate_memory.py`
- `autonomy_core/perception/gate_perception_yolo.py`
- `autonomy_core/launch/race_progression.py`

The file `docs/refactor_autonomy_api6_plan.md` did not exist when this plan was reviewed. This repository currently has this root-level `refactor_plan.md` as the plan source.

## Existing Modules To Preserve

These modules already exist and should not be renamed during early phases:

- `autonomy_core/launch/autonomy_api6.py`
- `autonomy_core/launch/get_telemetry.py`
- `autonomy_core/launch/race_progression.py`
- `autonomy_core/tools/px4_runner.py`
- `autonomy_core/tools/flight_logger.py`
- `autonomy_core/controller/attitude_controller3.py`
- `autonomy_core/planning/minimum_snap_planner_multi_time_optimized.py`
- `autonomy_core/perception/gate_memory.py`
- `autonomy_core/perception/gate_perception_yolo.py`
- `autonomy_core/perception/gate_perception_node.py`
- `autonomy_core/perception/corner_measurement.py`

## Current Responsibilities In `autonomy_api6.py`

`autonomy_api6.py` currently combines at least these responsibilities:

- `AutonomyAPI` facade and mutable runtime state
- telemetry access and internal state construction
- YOLO/PnP perception orchestration
- camera/body/world transform handling
- gate-memory updates and track admission
- race-order/progression coordination
- waypoint horizon construction
- minimum-snap trajectory generation and installation
- geometric plan validation
- z-corridor validation and fallback handling
- active target shift detection and replanning triggers
- pending suffix / same-plan continuation behavior
- reference sampling and virtual-clock progress logic
- no-target hold behavior
- yaw continuity and yaw-rate limiting
- PD/geometric attitude command generation via `RPGHighLevelTracker`
- Gazebo truth diagnostics and overlays
- installed-plan export rows for sidecar logging
- large sets of debug/log fields consumed by `FlightLogger`

## Public API Surface To Preserve

`AutonomyAPI` must continue to expose these methods because `px4_runner.py` calls them directly:

- `__init__(...)`
- `initialize_perception_yaw_correction(...)`
- `update_gate_memory_from_frame(...)`
- `path_plan(...)`
- `attitude_control()`
- `seed_yaw_hold(...)`
- `reset_target_update_event_debug()`
- `advance_gate_if_needed(...)`
- `check_active_target_shift_correction(...)`
- `check_tentative_lookahead_new_candidate_replan(...)`
- `check_tentative_lookahead_shift_replan(...)`
- `prepare_pending_suffix_for_future_only_replan(...)`

These methods may become wrappers, but their names, call signatures, return shapes, and side effects must remain compatible until a later runner migration phase explicitly changes them.

## Public Fields And Compatibility Surface To Preserve

`px4_runner.py` and `flight_logger.py` access many `AutonomyAPI` fields directly through `getattr(...)`, direct assignment, or snapshot/restore helpers. During extraction phases, either keep these attributes on `AutonomyAPI` or provide compatibility properties that behave identically.

Must preserve direct runtime fields and objects:

- `telemetry`
- `planner`
- `tracker`
- `use_perception`
- `use_lookahead_gate_filter`
- `active_waypoints`
- `active_times`
- `active_target_gates`
- `active_target_track_ids`
- `current_target_idx`
- `current_target_gate`
- `current_gate_pos`
- `active_target_track_id`
- `last_valid_target`
- `active_target_center`
- `active_target_center_at_plan`
- `active_target_latest_filtered_center`
- `trajectory_start_time`
- `time_elapsed`
- `wall_tau`
- `previous_sample_tau_used`
- `previous_sample_tau_plan_id`
- `p_ref`, `v_ref`, `a_ref`, `ref_yaw`
- `replan_time`
- `last_plan_mode`, `last_plan_start_gate_idx`, `last_plan_end_gate_idx`, `last_plan_duration`
- `installed_plan_export_rows`
- `active_plan_id`
- `planning_horizon_track_ids`
- `planning_horizon_waypoint_count`
- `planning_horizon_waypoints`
- `planning_horizon_waypoint_types`
- `_planning_target_waypoint_types`
- `pending_suffix_*` fields
- `future_only_replan_*` fields
- `previous_trajectory_*` fields
- `trajectory_expired_s`
- `command_stale_age_s`
- `hover_due_to_*` fields
- `continued_previous_trajectory_during_replan`
- `last_perception_replan_trigger`
- `has_commanded_yaw_reference`
- `previous_yaw_cmd`, `previous_yaw_cmd_log`
- `last_desired_yaw`
- `perception_hold_yaw`
- `hover_yaw_*` fields
- `no_active_target`, `no_target_control_mode`, `hold_anchor`, `hold_anchor_source`
- `velocity_damping_active`
- `completed_gate_reference_blocked`
- `p_ref_source`
- `yaw_hold_value`
- `distance_to_active_target`
- `gate_completion_triggered`
- `completion_reason`
- `completed_gate_position`
- `active_gate_idx_before`, `active_gate_idx_after`
- `race_cursor_before`, `race_cursor_after`
- `target_update_*` fields
- `crossing_*` fields
- `tracker_*` debug fields
- `thrust_*` debug fields
- `plan_*` validation debug fields
- perception/debug fields consumed by `FlightLogger.log(...)`, including YOLO, PnP, transform, Gazebo comparison, image timing, track, lookahead, and race-order fields

This is not a complete field inventory. Phase 1 must generate a more complete inventory before any large move.

## Current Known Stable Behaviors To Preserve

These behaviors are considered known stable unless Phase 1 inventory proves otherwise:

- Sidecar installed-plan logging through `installed_plan_export_rows` and `FlightLogger.log_installed_plan_rows(...)`.
- Geometric validation of minimum-snap plans before installation.
- Z-corridor lower validation, including startup relaxation when starting below the safe minimum altitude.
- Progress-aware virtual clock / reference sampling that limits trajectory time based on vehicle progress.
- Same-plan continuation after gate completion when the existing plan can continue safely.
- Pending suffix behavior for future-only replans and post-completion installation, if currently active.
- GT-independent perception behavior should remain available; Gazebo truth may be diagnostic but should not be required for production-safe perception.
- `use_passthrough_gate_velocities` is currently initialized as disabled; preserve that default unless an explicit later behavior phase changes it.
- Existing hover/no-target hold behavior, including post-completion grace behavior and yaw-hold continuity.
- Existing active-target shift and tentative-lookahead replan suppression thresholds.
- Existing debug frame behavior and log field population, unless a phase explicitly moves diagnostics behind a compatible wrapper.

## Target Module Layout

The final target can evolve, but module names should stay consistent across this plan unless Phase 1 revises them with evidence.

```text
autonomy_core/
  core/
    __init__.py
    types.py              # VehicleState, CameraFrame, CameraModel, ControlCommand, etc.
    state_adapter.py      # Telemetry/state extraction helpers.
    config.py             # Later vehicle/camera/runtime config, no behavior change initially.

  perception/
    gate_perception_yolo.py      # Existing YOLO/PnP implementation.
    gate_perception_node.py      # Existing ROS/Gazebo adapter, dev only.
    gate_memory.py               # Existing gate memory/tracking.
    gate_pipeline.py             # New orchestration seam for frame -> detections -> memory.

  planning/
    minimum_snap_planner_multi_time_optimized.py  # Existing solver.
    plan_validator.py                           # New validation seam.
    trajectory_manager.py                       # New installed-plan/reference seam.
    suffix_planner.py                           # New pending suffix seam.

  control/
    attitude_controller3.py      # Existing tracker implementation stays available.
    attitude_control.py          # New AutonomyAPI control-policy extraction.
    command_mapper.py            # Later command abstraction seam.

  debug/
    gazebo_diagnostics.py        # New debug-only Gazebo truth diagnostics seam.

  tools/
    px4_runner.py                # Existing dev runtime, must remain compatible.
    flight_logger.py             # Existing logger, must remain compatible.
```

Do not create `tracker_mpc.py`, competition runner modules, or VIO/EKF modules during this refactor.

## General Validation Commands

Run only commands appropriate for the phase. Do not run simulation unless the phase explicitly asks for it and the user approves that workflow.

Baseline docs-only validation:

```bash
git status --short
git diff -- refactor_plan.md
```

Python syntax validation after runtime edits:

```bash
python3 -m py_compile autonomy_core/launch/autonomy_api6.py
python3 -m py_compile autonomy_core/tools/px4_runner.py
python3 -m py_compile autonomy_core/tools/flight_logger.py
```

Import smoke test after runtime edits:

```bash
python3 - <<'PY'
from autonomy_core.launch.autonomy_api6 import AutonomyAPI
api = AutonomyAPI(use_perception=False, race_gate_count=3)
print(type(api).__name__)
print(hasattr(api, 'path_plan'), hasattr(api, 'attitude_control'))
PY
```

Optional expanded syntax check after broad extraction:

```bash
python3 -m compileall autonomy_core
```

Do not run sim in documentation-only phases.

## Rollback And Stop Rules

- Before each phase, run `git status --short`.
- If unexpected modified Python files are present, stop and ask how to proceed.
- Prefer small commits/checkpoints after each successful phase.
- If a validation command fails after a phase, do not start the next phase.
- First try to fix within the current phase scope.
- If the fix requires behavior changes, stop and report the issue instead.
- If extraction becomes tangled, revert only the current phase changes, not unrelated user work.
- Never use destructive git commands unless explicitly requested.

## Phase 1 - Inventory And Dependency Map

Purpose:
Document the current public surface and method groups before moving code.

Files to edit:

- `refactor_plan.md`
- optional `docs/autonomy_api6_method_inventory.csv`

Files not to edit:

- all runtime Python files

Expected behavior:

- No runtime behavior changes.
- No imports or code paths change.

Tasks:

- Inventory all methods in `AutonomyAPI` by responsibility.
- Inventory all `AutonomyAPI` methods called by `px4_runner.py`.
- Inventory all `AutonomyAPI` fields directly assigned by `px4_runner.py`.
- Inventory fields copied by `snapshot_trajectory_state(...)` and restored by `restore_trajectory_state(...)`.
- Inventory major logger field groups consumed by `FlightLogger.log(...)`.
- Mark methods as candidate targets: core, perception, planning, control, debug, compatibility wrapper, or verify.
- Identify methods that use Gazebo truth for diagnostics versus decision logic.
- Identify fields that must stay on the facade through compatibility properties.

Validation commands:

```bash
git status --short
git diff -- refactor_plan.md
git diff -- docs/autonomy_api6_method_inventory.csv
```

Success criteria:

- The plan and optional inventory describe what will move before code moves.
- No runtime Python files are modified.
- Uncertain items are marked `verify in Phase 1 inventory` instead of guessed.

Risks:

- Missing a logger-consumed field can break CSV logging after later phases.
- Missing a direct runner assignment can break replan/hover/safety behavior.

Suggested Codex prompt:

```text
Implement Phase 1 from refactor_plan.md only.

Scope:
- Do not edit runtime Python files.
- Build or update docs/autonomy_api6_method_inventory.csv with AutonomyAPI methods grouped by responsibility and proposed target module.
- Update refactor_plan.md only if the inventory reveals inaccurate API or phase details.
- Preserve behavior; docs only.
- Run git diff for refactor_plan.md and the CSV.
- Stop if unexpected source changes are present.
```

## Phase 2 - Extract Shared Types

Purpose:
Create neutral data contracts without changing existing behavior or call sites broadly.

Files to edit:

- `autonomy_core/core/__init__.py`
- `autonomy_core/core/types.py`
- `autonomy_core/launch/autonomy_api6.py` only if needed for imports of moved dataclasses/functions

Files not to edit:

- `autonomy_core/tools/px4_runner.py`
- `autonomy_core/tools/flight_logger.py`
- planner, perception, and controller implementation files unless syntax requires import-only changes

Expected behavior:

- Existing `AutonomyAPI` constructor and methods behave the same.
- Existing `State` and `Reference` semantics remain identical if moved or aliased.
- Existing dict-based telemetry remains supported.

Candidate types:

- `VehicleState`
- `ReferenceState` or existing `Reference` compatibility type
- `ControlCommand`
- `CameraFrame`
- `CameraModel`
- `GateTarget`
- `PlanDebugInfo`

Do not change:

- frame conventions
- signs of altitude or velocity
- yaw wrapping behavior
- tracker math
- planner sampling behavior
- perception output structures

Validation commands:

```bash
python3 -m py_compile autonomy_core/core/types.py
python3 -m py_compile autonomy_core/launch/autonomy_api6.py
python3 - <<'PY'
from autonomy_core.launch.autonomy_api6 import AutonomyAPI
api = AutonomyAPI(use_perception=False, race_gate_count=3)
print(type(api).__name__)
PY
```

Success criteria:

- Types import successfully.
- `AutonomyAPI(use_perception=False)` still instantiates.
- Public methods and public fields remain on `AutonomyAPI`.

Risks:

- Moving `State` or `Reference` can break `RPGHighLevelTracker.update(...)` if attributes differ.
- Dataclass defaults with mutable `np.ndarray` values can introduce shared-state bugs. Use `default_factory` where needed.

Suggested Codex prompt:

```text
Implement Phase 2 from refactor_plan.md only.

Scope:
- Add autonomy_core/core/types.py and __init__.py.
- Move or alias only simple dataclasses and pure helpers if safe.
- Preserve AutonomyAPI public behavior and px4_runner.py compatibility.
- Do not move perception, planning, or control logic.
- Do not change frame conventions or math.
- Run py_compile and the AutonomyAPI instantiation smoke test.
- Report changed files and any compatibility risks.
```

## Phase 3 - Add State Adapter Seam

Purpose:
Centralize telemetry-to-state extraction so future PX4, competition, and estimator sources can feed the same core.

Files to edit:

- `autonomy_core/core/state_adapter.py`
- `autonomy_core/launch/autonomy_api6.py`
- possibly `autonomy_core/core/types.py`

Files not to edit:

- `autonomy_core/tools/px4_runner.py`
- `autonomy_core/tools/flight_logger.py`
- planner solver
- perception implementation
- controller gains/logic

Expected behavior:

- `AutonomyAPI.telemetry` remains a public mutable field.
- `px4_runner.py` can still assign `autonomy.telemetry = telemetry`.
- `attitude_control()` returns the same tuple shape `(roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd)`.

Tasks:

- Add helper to build `VehicleState` from the current `GetTelemetry`-style object.
- Replace direct telemetry reads only in low-risk control entry points first: `attitude_control(...)` and `hold_no_target_control(...)` if feasible.
- Keep all existing debug fields populated.
- Keep state frame convention exactly as currently used: current internal position uses positive-up `z` from the PX4 runner conversion.

Do not change:

- `px4_runner.py` telemetry conversion from NED down to positive-up `z`
- yaw correction behavior
- velocity sanitization behavior
- no-target hold behavior

Validation commands:

```bash
python3 -m py_compile autonomy_core/core/state_adapter.py
python3 -m py_compile autonomy_core/launch/autonomy_api6.py
python3 - <<'PY'
from autonomy_core.launch.autonomy_api6 import AutonomyAPI
api = AutonomyAPI(use_perception=False, race_gate_count=3)
cmd = api.attitude_control()
print(len(cmd), all(x == x for x in cmd))
PY
```

Success criteria:

- `attitude_control()` still works before a plan exists.
- No caller changes are required in `px4_runner.py`.
- State extraction is now testable independently.

Risks:

- Small timestamp or yaw-source changes could affect perception snapshot timing.
- Accidentally changing altitude sign would invalidate planning and control.

Suggested Codex prompt:

```text
Implement Phase 3 from refactor_plan.md only.

Scope:
- Add autonomy_core/core/state_adapter.py.
- Centralize current telemetry object -> VehicleState extraction.
- Replace direct telemetry reads only in attitude_control and hold_no_target_control if safe.
- Preserve all public AutonomyAPI fields and method return shapes.
- Do not change frame conventions, controller behavior, or px4_runner.py.
- Run py_compile and the attitude_control smoke test.
- Stop if unrelated source changes are present.
```

## Phase 4 - Extract Plan Validation

Purpose:
Separate trajectory validation from the main runtime state machine while preserving installed-plan acceptance/rejection behavior.

Files to edit:

- `autonomy_core/planning/plan_validator.py`
- `autonomy_core/launch/autonomy_api6.py`
- possibly `autonomy_core/planning/__init__.py` if needed

Files not to edit:

- `autonomy_core/planning/minimum_snap_planner_multi_time_optimized.py`
- `autonomy_core/tools/px4_runner.py`
- `autonomy_core/tools/flight_logger.py`
- controller and perception modules

Expected behavior:

- `AutonomyAPI.validate_minimum_snap_geometry(...)` remains callable or wrapped if any internal code still uses it.
- Plan validation debug fields remain on `AutonomyAPI` for logging.
- Validation thresholds and failure reasons do not change.

Move or wrap:

- `reset_plan_geometric_validation_debug(...)`
- `validate_minimum_snap_geometry(...)`
- z-corridor validation details
- debug result assignment helpers if they can move without behavior changes

Do not move yet:

- `path_plan(...)`
- planner solve/install logic
- suffix install logic
- active target/horizon construction

Validation commands:

```bash
python3 -m py_compile autonomy_core/planning/plan_validator.py
python3 -m py_compile autonomy_core/launch/autonomy_api6.py
python3 - <<'PY'
from autonomy_core.launch.autonomy_api6 import AutonomyAPI
api = AutonomyAPI(use_perception=False, race_gate_count=3)
print(hasattr(api, 'validate_minimum_snap_geometry'))
print(api.plan_geometric_validation_failed)
PY
```

Success criteria:

- Validation module imports.
- Existing `path_plan(...)` still reaches validation and writes the same debug fields.
- No planner output changes are intended.

Risks:

- Validation uses many `AutonomyAPI` thresholds/debug fields; moving too much at once can break logging.
- A pure function may be cleaner, but compatibility may require a wrapper that copies results back to `AutonomyAPI`.

Suggested Codex prompt:

```text
Implement Phase 4 from refactor_plan.md only.

Scope:
- Extract plan validation into autonomy_core/planning/plan_validator.py.
- Keep AutonomyAPI wrappers and debug fields compatible.
- Do not move path_plan or planner solve/install logic.
- Preserve validation thresholds, failure reasons, and z-corridor behavior exactly.
- Run py_compile and the validation smoke test.
- Report any fields that made extraction unsafe.
```

## Phase 5 - Extract Trajectory Manager Seam

Purpose:
Move installed-plan ownership and reference sampling toward a dedicated planning/runtime seam without changing path planning behavior.

Files to edit:

- `autonomy_core/planning/trajectory_manager.py`
- `autonomy_core/launch/autonomy_api6.py`
- possibly `autonomy_core/planning/__init__.py`

Files not to edit:

- `autonomy_core/tools/px4_runner.py`
- `autonomy_core/tools/flight_logger.py`
- perception modules
- controller modules
- minimum-snap solver internals

Expected behavior:

- `AutonomyAPI.path_plan(...)` remains the public entry point.
- `AutonomyAPI.prepare_pending_suffix_for_future_only_replan(...)` remains the public entry point.
- `px4_runner.py` snapshot/restore helpers still work.
- `installed_plan_export_rows` still logs through `FlightLogger.log_installed_plan_rows(...)`.

Candidate move or wrap:

- `choose_T(...)`
- `allocate_segment_times(...)`
- `record_installed_plan_for_export(...)`
- `compute_reference_sample_tau(...)`
- `nearest_tau_on_active_plan_xy(...)`
- `clamp_reference_altitude(...)` only if it can move without changing perception/control coupling
- active plan metadata ownership, if compatibility properties are straightforward

Do not move yet:

- `path_plan(...)` wholesale unless Phase 5 inventory shows it is safe
- `build_waypoint_horizon_*` if still tightly coupled to perception/race state
- suffix creation/installation internals unless Phase 6 is included separately

Validation commands:

```bash
python3 -m py_compile autonomy_core/planning/trajectory_manager.py
python3 -m py_compile autonomy_core/launch/autonomy_api6.py
python3 - <<'PY'
from autonomy_core.launch.autonomy_api6 import AutonomyAPI
api = AutonomyAPI(use_perception=False, race_gate_count=3)
print(hasattr(api, 'path_plan'), hasattr(api, 'planner'))
print(type(api.installed_plan_export_rows).__name__)
PY
```

Success criteria:

- Public plan fields remain readable from `AutonomyAPI`.
- Installed-plan export rows are still populated when a valid plan is installed.
- No replan trigger behavior changes.

Risks:

- `px4_runner.py` deep-copies `autonomy.planner` and several active-plan fields; compatibility must remain exact.
- Virtual-clock sampling and active target completion depend on shared state.

Suggested Codex prompt:

```text
Implement Phase 5 from refactor_plan.md only.

Scope:
- Add autonomy_core/planning/trajectory_manager.py.
- Extract only low-risk installed-plan helpers and reference sampling helpers.
- Keep AutonomyAPI.path_plan as the public wrapper and do not rewrite planning behavior.
- Preserve active plan fields used by px4_runner snapshot/restore and FlightLogger.
- Do not change minimum-snap solver code.
- Run py_compile and the trajectory-manager smoke test.
- Stop if extraction would require behavior changes.
```

## Phase 6 - Extract Suffix Planner

Purpose:
Separate future-only replans, pending suffix creation, and pending suffix installation from the main facade.

Files to edit:

- `autonomy_core/planning/suffix_planner.py`
- `autonomy_core/launch/autonomy_api6.py`

Files not to edit:

- `px4_runner.py`
- `flight_logger.py`
- controller/perception modules
- minimum-snap solver internals

Expected behavior:

- Same-plan continuation remains first priority after gate completion.
- Pending suffix behavior remains second priority.
- Full replan remains fallback behavior.
- Existing `pending_suffix_*` fields remain readable from `AutonomyAPI`.

Candidate move or wrap:

- `_reset_pending_suffix_state(...)`
- `active_target_crossing_tau(...)`
- `prepare_pending_suffix_for_future_only_replan(...)`
- `_install_pending_suffix_after_completion(...)`

Validation commands:

```bash
python3 -m py_compile autonomy_core/planning/suffix_planner.py
python3 -m py_compile autonomy_core/launch/autonomy_api6.py
```

Success criteria:

- Public wrapper methods remain callable.
- Existing suffix debug/log fields remain on the facade.
- No unnecessary `active_target_advanced` full replan is introduced when same-plan continuation should handle completion.

Risks:

- Suffix logic is tightly coupled to active target IDs, race progression, and installed plan state.
- This phase should be skipped or split further if Phase 1 inventory shows too much coupling.

## Phase 7 - Extract Control Boundary

Purpose:
Prepare for future MPC by separating current control policy from the facade, without implementing MPC.

Files to edit:

- `autonomy_core/control/attitude_control.py`
- `autonomy_core/control/command_mapper.py` only if useful as a behavior-preserving seam
- `autonomy_core/launch/autonomy_api6.py`

Files not to edit:

- `autonomy_core/controller/attitude_controller3.py`, unless import-only changes are unavoidable
- `px4_runner.py`
- `flight_logger.py`
- planner/perception modules

Expected behavior:

- `AutonomyAPI.attitude_control()` still returns `(roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd)`.
- Current `RPGHighLevelTracker` remains the tracker.
- Current gain values and thrust limits remain unchanged.
- Roll/pitch sign convention remains unchanged.

Candidate move or wrap:

- `hold_no_target_control(...)`
- `get_perception_yaw_hold_reference(...)`
- `seed_yaw_hold(...)`
- `continuous_yaw_command(...)`
- `record_tracker_control_debug(...)`
- `attitude_control(...)` internals

Validation commands:

```bash
python3 -m py_compile autonomy_core/control/attitude_control.py
python3 -m py_compile autonomy_core/launch/autonomy_api6.py
python3 - <<'PY'
from autonomy_core.launch.autonomy_api6 import AutonomyAPI
api = AutonomyAPI(use_perception=False, race_gate_count=3)
cmd = api.attitude_control()
print(cmd)
PY
```

Success criteria:

- No-controller-plan hover behavior remains identical.
- Yaw continuity fields remain populated for logger compatibility.
- Tracker debug fields remain populated.

Risks:

- Sign errors in roll/pitch are easy to introduce.
- Yaw unwrapping/rate limiting state depends on previous command fields.

## Phase 8 - Extract Perception Pipeline

Purpose:
Move frame-processing orchestration to a perception module while preserving YOLO/PnP/memory behavior.

Files to edit:

- `autonomy_core/perception/gate_pipeline.py`
- `autonomy_core/launch/autonomy_api6.py`

Files not to edit:

- `gate_perception_yolo.py`, unless import-only changes are needed
- `gate_memory.py`, unless import-only changes are needed
- `px4_runner.py`
- `flight_logger.py`
- planner/control modules

Expected behavior:

- `AutonomyAPI.update_gate_memory_from_frame(...)` remains the public wrapper.
- Perception debug fields remain on `AutonomyAPI` or are mirrored back compatibly.
- Gazebo truth arguments may still be accepted for diagnostics.

Candidate move or wrap:

- `process_frame(...)`
- camera/body transform helpers that are perception-only
- `update_gate_memory_from_frame(...)` internals
- detection flow debug helpers if they can move without breaking logger fields
- supplemental detection logic if still perception-only

Validation commands:

```bash
python3 -m py_compile autonomy_core/perception/gate_pipeline.py
python3 -m py_compile autonomy_core/launch/autonomy_api6.py
```

Success criteria:

- Perception-enabled runner can still call the same facade method.
- `GateMemory` behavior and thresholds are unchanged.
- YOLO model loading behavior is unchanged.

Risks:

- Perception code is highly coupled to debug overlays and Gazebo truth diagnostics.
- Moving transform helpers before frame contracts are explicit may hide frame bugs.

## Phase 9 - Extract Race And Target Selection Helpers

Purpose:
Separate race-order, candidate validation, and target/horizon selection from unrelated control/perception code.

Files to edit:

- Verify target module in Phase 1 inventory. Likely one of:
  - `autonomy_core/planning/target_horizon.py`
  - `autonomy_core/racing/target_selection.py`
  - keep inside `trajectory_manager.py` if coupling is still high
- `autonomy_core/launch/autonomy_api6.py`

Files not to edit:

- `autonomy_core/launch/race_progression.py`, unless import-only changes are needed
- `px4_runner.py`
- `flight_logger.py`

Expected behavior:

- Race progression remains sequence-based over persistent gate track IDs.
- Landmark memory is not deleted when a gate is passed.
- Predefined race order behavior remains unchanged.

Candidate move or wrap:

- `validate_perception_gate_center(...)`
- `is_near_completed_gate(...)`
- `find_duplicate_committed_track(...)`
- `validate_planning_target(...)`
- `validate_candidate_target(...)`
- `canonical_track_id(...)`
- landmark merge helpers
- race-order assignment helpers
- `build_waypoint_horizon_from_memory(...)`
- `build_waypoint_horizon_from_gt(...)`
- `build_waypoint_horizon(...)`
- `advance_gate_if_needed(...)` internals if safe

Validation commands:

```bash
python3 -m py_compile autonomy_core/launch/autonomy_api6.py
```

Success criteria:

- Gate advancement behavior remains unchanged.
- Race-order log/debug fields remain populated.
- No target skipping or duplicate target regression is introduced.

Risks:

- Candidate filtering, race order, and perception memory are tightly coupled.
- This phase may need to be split after Phase 1 inventory.

## Phase 10 - Extract Gazebo Diagnostics

Purpose:
Keep Gazebo truth diagnostics available while isolating them from production decision logic.

Files to edit:

- `autonomy_core/debug/__init__.py`
- `autonomy_core/debug/gazebo_diagnostics.py`
- `autonomy_core/launch/autonomy_api6.py`

Files not to edit:

- `px4_runner.py` unless a later explicit compatibility phase allows it
- planner/control/perception behavior modules

Expected behavior:

- Gazebo debug values still populate in the PX4/Gazebo dev runner.
- Production-safe perception can run without Gazebo truth fields.
- `gazebo_truth_sim_only` is not removed in this phase, but its decision impact is documented and made easier to disable later.

Candidate move or wrap:

- `_gazebo_model_pose_to_planner(...)`
- Gazebo pose comparison helpers
- GT projection helpers
- RMSE/debug overlay helpers
- transform sweep diagnostics
- size-depth and PnP formulation diagnostics if they are debug-only

Validation commands:

```bash
python3 -m py_compile autonomy_core/debug/gazebo_diagnostics.py
python3 -m py_compile autonomy_core/launch/autonomy_api6.py
```

Success criteria:

- Debug diagnostics remain available in Gazebo.
- The main production path can avoid requiring Gazebo pose objects.
- Logger field names remain compatible.

Risks:

- Some methods combine diagnostics with candidate selection. Mark uncertain methods `verify before moving` rather than forcing extraction.

## Phase 11 - Config And Logging Mode Seams

Purpose:
Prepare for race/performance logging modes and vehicle configuration without changing defaults.

Files to edit:

- `autonomy_core/core/config.py`
- optional config docs or examples
- `autonomy_core/launch/autonomy_api6.py`

Files not to edit:

- `flight_logger.py` unless an explicit compatibility wrapper is needed
- `px4_runner.py` unless an explicit optional config pass is approved

Expected behavior:

- Current default parameters remain exactly the same.
- Hard-coded values may be mirrored into config objects but not changed.
- Existing constructor arguments keep working.

Validation commands:

```bash
python3 -m py_compile autonomy_core/core/config.py
python3 -m py_compile autonomy_core/launch/autonomy_api6.py
```

Success criteria:

- `AutonomyAPI(...)` still works with old arguments.
- Config objects make future competition/vehicle setup easier without changing behavior.

Risks:

- Accidentally changing defaults will change behavior even if code structure looks cleaner.

## Phase 12 - Competition Adapter Preparation Checkpoint

Purpose:
Confirm the refactor has created enough seams before starting competition interface work.

Files to edit:

- docs only unless gaps are found and explicitly approved

Expected behavior:

- No runtime behavior changes.

Checklist:

- There is a state adapter seam.
- There is a camera frame/model seam or clear next step.
- There is a command output seam or clear next step.
- `AutonomyAPI` remains the facade.
- `px4_runner.py` remains compatible.
- Gazebo truth diagnostics are isolated enough to disable for production.
- Logger fields are either preserved on the facade or proxied compatibly.

Do not implement:

- competition UDP vision receiver
- MAVLink competition client
- MPC
- EKF/VIO

Validation commands:

```bash
git status --short
git diff -- refactor_plan.md
```

Success criteria:

- The next plan can focus on competition interfacing without further monolith surgery.

## Future Work Not Part Of This Refactor

These items are explicitly out of scope until the refactor phases create stable seams:

- MPC tracker implementation.
- VIO/EKF/state estimation integration.
- Dynamic randomized gate generator.
- YOLO training and domain randomization.
- Competition API adapter.
- UDP JPEG chunk receiver for the competition vision stream.
- MAVLink2 competition client and heartbeat/command-rate compliance layer.
- Logging performance/race mode.
- Search/reacquisition mode.
- Controller tuning.
- Perception threshold tuning.
- Planner behavior changes.
- Removing Gazebo/PX4 development support.

## Recommended First Prompt After This Plan Is Finalized

```text
Implement Phase 1 from refactor_plan.md only.

Read autonomy_core/launch/autonomy_api6.py, autonomy_core/tools/px4_runner.py, and autonomy_core/tools/flight_logger.py. Do not edit runtime Python files.

Create or update docs/autonomy_api6_method_inventory.csv with:
- AutonomyAPI method name
- responsibility group
- proposed target module
- public/private/runner-called status
- runner/logger field dependencies if obvious
- risk notes

Update refactor_plan.md only if the inventory reveals incorrect method names, missing public API, or unsafe phase ordering.

Run:
- git status --short
- git diff -- refactor_plan.md
- git diff -- docs/autonomy_api6_method_inventory.csv

Stop if unexpected source changes are present. Summarize uncertain items as verify-in-next-phase rather than guessing.
```
