# Competition Adapter Phase 8.25A Geometry Audit

Status: complete read-only audit.

Date: 2026-06-12

Scope:

- Audit gate and drone geometry constants before interpreting perception, replay, or PX4/Gazebo surrogate output.
- Compare current code against VADR-TS-002 geometry.
- Preserve current PnP scoring, gate admission, clearance, race progression, logging, live transport, and command behavior.

Official VADR-TS-002 geometry:

- Gate outer square: `2700 mm` (`2.7 m`).
- Gate inner square: `1500 mm` (`1.5 m`).
- Gate depth: `260 mm` (`0.26 m`).
- Drone chassis: `280 mm x 280 mm x 160 mm`.

## Central Constants Check

Official gate and drone dimensions are centralized passively in
`autonomy_core/core/competition_config.py`:

- `RuntimeCompetitionConfig.gate_outer_square_mm = 2700`
- `RuntimeCompetitionConfig.gate_inner_square_mm = 1500`
- `RuntimeCompetitionConfig.gate_depth_mm = 260`
- `RuntimeCompetitionConfig.drone_chassis_length_mm = 280`
- `RuntimeCompetitionConfig.drone_chassis_width_mm = 280`
- `RuntimeCompetitionConfig.drone_chassis_height_mm = 160`
- `VADR_TS_002` exports the default frozen config instance.

These constants are not wired into existing runtime defaults. Current runtime
perception and memory paths still contain duplicated or behavior-specific
values. No centralization or runtime wiring change was made in this audit pass.

## Geometry Inventory

| File | Constant or value | Units | Status vs VADR | Affects | Notes and risk |
| --- | --- | --- | --- | --- | --- |
| `autonomy_core/core/competition_config.py` | `gate_outer_square_mm = 2700` | mm | Matches | adapter metadata/tests/docs | Central passive official constant. |
| `autonomy_core/core/competition_config.py` | `gate_inner_square_mm = 1500` | mm | Matches | adapter metadata/tests/docs | Central passive official constant. |
| `autonomy_core/core/competition_config.py` | `gate_depth_mm = 260` | mm | Matches | adapter metadata/tests/docs | No active PnP or clearance use found. |
| `autonomy_core/core/competition_config.py` | `drone_chassis_length_mm = 280`, `drone_chassis_width_mm = 280`, `drone_chassis_height_mm = 160` | mm | Matches | adapter metadata/tests/docs | No active clearance use found. |
| `tests/test_competition_config.py` | assertions for `2700`, `1500`, `260`, `280`, `280`, `160` | mm | Matches | tests | Deterministic coverage exists for the central passive constants. |
| `autonomy_core/core/config.py` | `PerceptionConfig.gate_size = 1.5` | m | Matches inner square | future runtime config if wired | Duplicated from spec instead of derived from `RuntimeCompetitionConfig`. |
| `autonomy_core/launch/autonomy_api6.py` | `GatePerception(gate_size=1.5, ...)` | m | Matches inner square if YOLO keypoints are inner corners | active YOLO PnP object points and size-depth debug | Behavior-sensitive duplicate of the official inner opening; not changed. |
| `autonomy_core/perception/gate_perception_yolo.py` | constructor default `gate_size=1.5` | m | Matches inner square | active YOLO PnP object points and apparent size-depth estimate | Comments state YOLO keypoints are inner opening corners. This is the active intended model. |
| `autonomy_core/perception/gate_perception_yolo.py` | `model_points = +/- gate_size / 2.0` with default `1.5` | m | Matches inner square planar corners | PnP object points | Official gate depth is not represented; object model is planar. |
| `autonomy_core/perception/gate_perception_yolo.py` | comment: use `gate_size=2.7` for HSV outer contour detection | m | Matches outer square | docs/comment only | Correct as guidance, but not centralized. |
| `autonomy_core/perception/gate_perception_yolo.py` | explicit debug sweep `sizes=(1.40, 1.50, 1.60)` | m | Includes official inner square | PnP debug/diagnostics | Brackets official inner opening. Do not interpret as final scoring without checking downstream debug consumers. |
| `autonomy_core/perception/gate_perception_yolo.py` | `solve_pnp_gate_size_sweep(... sizes=(1.90, 2.00, 2.10))` default | m | Stale/ambiguous | PnP debug if called without explicit sizes | Active call passes `1.40/1.50/1.60`; default remains old and should be handled in Phase 8.25B if retained. |
| `autonomy_core/perception/gate_perception.py` | constructor default `gate_size=1.5` | m | Matches inner square, but comments describe outer-frame detection | legacy perception PnP | Ambiguous legacy path. Its explicit debug sweep uses outer sizes, but default object points are inner size. |
| `autonomy_core/perception/gate_perception.py` | explicit debug sweep `sizes=(2.60, 2.70, 2.80)` | m | Includes official outer square | legacy PnP debug/diagnostics | Matches outer-frame comment. |
| `autonomy_core/perception/gate_perception.py` | `solve_pnp_gate_size_sweep(... sizes=(1.90, 2.00, 2.10))` default | m | Stale/ambiguous | legacy PnP debug if called without explicit sizes | Does not match official inner or outer square. |
| `autonomy_core/perception/gate_perception_orange.py` | constructor default `gate_size=2.0` | m | Stale | legacy HSV/orange PnP object points | Does not match official `1.5 m` inner or `2.7 m` outer square. Behavior-sensitive if this path is used. |
| `autonomy_core/perception/gate_perception_orange.py` | explicit/default sweep `sizes=(1.90, 2.00, 2.10)` | m | Stale | legacy HSV/orange PnP debug | Does not match official geometry. |
| `autonomy_core/perception/synthetic_pnp_sanity.py` | `GatePerception(gate_size=2.0, ...)` | m | Stale | synthetic PnP tool/test script | Should not be used to validate competition geometry without update. |
| `autonomy_core/tools/autolabel_gazebo_yolo_pose.py` | `INNER_OPENING_M = 1.5` | m | Matches inner square | Gazebo/autolabel tooling only | Uses Gazebo/reference metadata and must not be competition truth. Constant itself matches official geometry. |
| `autonomy_core/tools/autolabel_gazebo_yolo_pose.py` | `OUTER_GATE_M = 2.7` | m | Matches outer square | Gazebo/autolabel tooling only | Uses Gazebo/reference metadata and must not be competition truth. Constant itself matches official geometry. |
| `autonomy_core/perception/gate_memory.py` | default `association_radius = 2.0`, `commit_radius = 2.0`, `new_track_block_radius = 3.5` | m | Ambiguous behavior thresholds | track association/admission | Not physical gate dimensions. Changing them would alter behavior. |
| `autonomy_core/launch/autonomy_api6.py` | `GateMemory(association_radius=1.5, commit_radius=1.5, new_track_block_radius=4.5, ...)` | m | Ambiguous behavior thresholds | active memory/admission | Not physical gate dimensions, but values overlap official inner size and can be misread during analysis. |
| `autonomy_core/core/config.py` | `association_radius = 1.5`, `commit_radius = 1.5`, `new_track_block_radius = 4.5` | m | Ambiguous behavior thresholds | future runtime config if wired | Mirrors current behavior; not official geometry constants. |
| `autonomy_core/perception/gate_memory.py` | `estimated_gate_size = 2.0`; `duplicate_merge_radius = max(5.0, 2.5 * estimated_gate_size)` | m | Stale/ambiguous | duplicate committed-track merge and race admission | Comment says approximate simulator perception aperture. This is behavior-sensitive and should not be silently changed. |
| `autonomy_core/racing/race_admission.py` | uses `api.gate_memory.duplicate_merge_radius` | m | Inherits stale/ambiguous source | race admission and duplicate merge | If `estimated_gate_size` changes, race admission can change. |
| `docs/autonomy_math_reference.md` | documents `estimated_gate_size = 2.0` and `duplicate_merge_radius` formula | m | Stale/ambiguous docs | docs | Matches current code, not official VADR geometry. |
| `autonomy_core/launch/race_progression.py` | `pass_radius = 1.25`, `clear_radius = 1.75` | m | Ambiguous behavior thresholds | race progression/clearance state machine | Not official gate or drone geometry. Do not change in a geometry-constant fix without a behavior review. |
| `autonomy_core/launch/autonomy_api6.py` | `RaceProgression(pass_radius=1.25, clear_radius=1.75, ...)` | m | Ambiguous behavior thresholds | active race progression | Runtime override of the same behavior thresholds. |
| `autonomy_core/core/config.py` | `race_pass_radius = 1.25`, `race_clear_radius = 1.75` | m | Ambiguous behavior thresholds | future runtime config if wired | Mirrors current behavior; not official physical geometry. |
| `autonomy_core/launch/autonomy_api6.py` | `completed_gate_position_radius = 1.5` | m | Ambiguous behavior threshold | completed-gate duplicate filtering | Not a gate dimension even though it equals the inner square size. |
| `autonomy_core/core/config.py` | `completed_gate_position_radius = 1.5` | m | Ambiguous behavior threshold | future runtime config if wired | Mirrors current behavior. |
| `autonomy_core/launch/autonomy_api6.py` | `current_gate_freeze_distance = 2.0`, `current_gate_freeze_progress_margin = 1.5`, `gate_pass_radius = 0.75` | m | Ambiguous behavior thresholds | active target freeze and gate advancement | Not official physical geometry. |
| `autonomy_core/racing/gate_advancement.py` | uses `api.gate_pass_radius` and `api.race_progression.pass_radius` | m | Inherits behavior thresholds | race gate advancement | Current pass-through proxy does not explicitly validate physical aperture or drone chassis clearance. |
| `autonomy_core/planning/target_validation.py` | uses `completed_gate_position_radius` and `gate_memory.commit_radius` | m | Inherits behavior thresholds | target validation and duplicate rejection | Not physical geometry. |
| `autonomy_core/planning/target_horizon.py` | uses `gate_memory.commit_radius`; calls `race_progression.update_clearance(...)` | m | Inherits behavior thresholds | horizon filtering and clearance update | Not physical geometry. |
| `autonomy_core/perception/gate_perception_node.py` | passes through `gate_size_sweep` debug entries | m, inferred | Ambiguous debug plumbing | perception debug world conversion | No fixed geometry value found in this file. |
| `autonomy_core/launch/autonomy_api6.py`, `autonomy_core/tools/px4_runner.py`, `autonomy_core/tools/flight_logger.py` | `pnp_size_190`, `pnp_size_200`, `pnp_size_210` debug/log fields | m labels, inferred | Stale/ambiguous for active YOLO sweep | logging/debug only | Active YOLO sweep emits `140/150/160`, but logger-facing fields still expect `190/200/210`. This can make perception-debug interpretation misleading without changing behavior. |
| `autonomy_core/tools/plot_flight_xy.py` | mock gate positions with `z = 1.5` | m, inferred | Not gate geometry | plotting/demo tooling | False-positive search hit; center height, not gate size. |
| `autonomy_core/tools/px4_runner.py` | topic name `gate_test_1500mm_blue` | text | Not gate geometry | Gazebo/PX4 runner topic | False-positive search hit; protected by competition Gazebo guard for competition runner paths. |

## Depth And Drone Clearance Findings

- Official gate depth (`260 mm`) exists only in passive competition config and its deterministic test.
- Active PnP object points are planar square corners with `z = 0`; no current runtime use of gate depth was found.
- Official drone chassis dimensions exist only in passive competition config and its deterministic test.
- No runtime clearance check was found that uses the `280 mm x 280 mm x 160 mm` chassis dimensions.
- Current completion/clearance behavior is based on radius/progress proxies (`pass_radius`, `clear_radius`, `gate_pass_radius`), not explicit aperture/chassis clearance.

Risk before interpreting perception or surrogate output:

- PnP distance and pose depend on `gate_size`; the active YOLO path appears aligned to the official inner opening, but it is duplicated instead of sourced from central constants.
- Legacy perception paths and synthetic PnP tooling contain `2.0 m` geometry that does not match official inner or outer square dimensions.
- Duplicate-merge and race-admission behavior still use `estimated_gate_size = 2.0` as an approximate simulator aperture, but the resulting `duplicate_merge_radius` is a behavior threshold, not a geometry constant.
- Existing `pnp_size_190/200/210` logging fields can obscure active YOLO debug sweeps around `1.40/1.50/1.60`; do not use those fields alone to judge competition PnP geometry.
- Surrogate PX4/Gazebo results must not be interpreted as competition evidence until these stale/ambiguous paths are either fixed or explicitly excluded.

## Test Coverage Audit

Existing deterministic coverage:

- `tests/test_competition_config.py` asserts official gate outer square, inner square, depth, drone chassis dimensions, and race duration in `RuntimeCompetitionConfig`.
- Phase 8 camera and image adapter tests do not use gate/drone geometry constants.

Missing or deferred coverage:

- No deterministic test currently proves active YOLO `GatePerception` object points are derived from `RuntimeCompetitionConfig.gate_inner_square_mm`.
- No deterministic test currently proves `GatePerception.model_points_for_size(1.5)` produces the official inner-corner `+/-0.75 m` square without importing YOLO weights.
- No deterministic test currently covers the stale/default `1.90/2.00/2.10` debug sweep defaults.
- No deterministic test currently covers legacy `gate_perception_orange.py` or `synthetic_pnp_sanity.py` against official dimensions.
- No deterministic test currently covers whether `pnp_size_190/200/210` logging remains meaningful after the active YOLO sweep moved to `1.40/1.50/1.60`.
- No deterministic test currently covers explicit drone/chassis clearance because runtime logic does not use those dimensions.

## Phase 8.25A Proposed Phase 8.25B Follow-Up

Do not silently change the values below in Phase 8.25A. A follow-up should be
small, reviewed, and fixture-backed because these values can alter PnP pose,
target admission, duplicate merging, clearance, logging interpretation, or race
progression.

Suggested Phase 8.25B scope:

- Add passive meter-derived helpers or constants for official gate inner, outer, depth, and drone chassis dimensions if needed.
- Add import-safe deterministic tests for pure object-point helpers without requiring YOLO, GPU, or model weights.
- Decide whether active YOLO `gate_size=1.5` should be sourced from `RuntimeCompetitionConfig` while preserving the exact numeric default.
- Decide whether stale `1.90/2.00/2.10` sweep defaults should be removed, renamed, or replaced with official inner/outer fixture values.
- Decide whether `GateMemory.estimated_gate_size = 2.0` is a behavior threshold that should remain as-is, or a stale physical assumption that needs a separate admission/merge behavior test.
- Decide whether `pnp_size_190/200/210` log fields should be generalized or supplemented without breaking existing logger schema.
- Defer any explicit drone/chassis clearance behavior until a separate behavior phase; adding such a check would change race progression semantics.

## Phase 8.25B Cleanup Results

Status: complete for passive centralization and no-behavior-change cleanup.

Centralized passive helpers added to `autonomy_core/core/competition_config.py`:

- `millimeters_to_meters(...)`
- `planar_square_object_points_m(...)`
- `RuntimeCompetitionConfig.gate_outer_square_m`
- `RuntimeCompetitionConfig.gate_inner_square_m`
- `RuntimeCompetitionConfig.gate_depth_m`
- `RuntimeCompetitionConfig.gate_outer_half_extent_m`
- `RuntimeCompetitionConfig.gate_inner_half_extent_m`
- `RuntimeCompetitionConfig.drone_chassis_m`
- `RuntimeCompetitionConfig.gate_inner_object_points_m`

Safe replacements made with no numeric behavior change:

- `autonomy_core/core/config.py` now derives `PerceptionConfig.gate_size` from `VADR_TS_002.gate_inner_square_m`; the value remains `1.5 m`.
- `autonomy_core/perception/gate_perception_yolo.py` now derives its default `gate_size` from `VADR_TS_002.gate_inner_square_m`; the value remains `1.5 m`.
- `autonomy_core/perception/gate_perception_yolo.py` now builds planar square model points through `planar_square_object_points_m(gate_size)`; the default object points remain `+/-0.75 m` inner-corner points with `z = 0`.
- `autonomy_core/launch/autonomy_api6.py` now passes `VADR_TS_002.gate_inner_square_m` into the active YOLO `GatePerception` constructor; the value remains `1.5 m`.

Deterministic tests added:

- Official meter helpers assert `2.7 m`, `1.5 m`, `0.26 m`, `1.35 m`, `0.75 m`, and `0.28 x 0.28 x 0.16 m`.
- Official inner-gate object points assert the active YOLO planar convention without importing YOLO, GPU code, or model weights.
- The runtime config scaffold test asserts `PerceptionConfig.gate_size` tracks `RuntimeCompetitionConfig.gate_inner_square_m`.
- A source-level test asserts the active YOLO file and `AutonomyAPI` constructor reference the official inner-square helper without importing `gate_perception_yolo`.

Intentionally unchanged behavior-sensitive values:

- `GateMemory.estimated_gate_size = 2.0` and `duplicate_merge_radius = max(5.0, 2.5 * estimated_gate_size)` remain unchanged because they affect duplicate committed-track merge behavior and race admission.
- `association_radius`, `commit_radius`, `new_track_block_radius`, `completed_gate_position_radius`, `pass_radius`, `clear_radius`, `gate_pass_radius`, current-gate freeze distance, and race-admission thresholds remain unchanged because they affect target admission, clearance, race progression, and low-gate crossing behavior.
- Legacy `gate_perception.py`, `gate_perception_orange.py`, and `synthetic_pnp_sanity.py` geometry defaults remain unchanged because they are old diagnostic/perception paths and changing them could alter non-competition behavior or tool expectations.
- `pnp_size_190/200/210` logger fields remain unchanged because renaming or generalizing them would change logger schema compatibility.
- Gate depth and drone chassis constants remain passive only; no explicit aperture/chassis clearance behavior was added.

A later perception/planning behavior phase must decide whether stale legacy PnP defaults, duplicate-merge sizing, and pass/clearance proxies should change. Any such change needs focused fixtures because it can alter PnP pose, target admission, duplicate merging, clearance, logging interpretation, or race progression.
