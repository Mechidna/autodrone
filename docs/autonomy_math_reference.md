# Autonomy Math Reference

This document describes the current autonomy-stack math end to end. It is a reference for the behavior that exists now, not a proposal to change behavior. Future improvements are listed separately in each relevant section.

Primary runtime facade: `autonomy_core/launch/autonomy_api6.py::AutonomyAPI`.

## 1. Coordinate Frames And State Conventions

### Current Behavior

The runtime uses an internal world/planner frame with positive `z` upward. PX4 telemetry arrives from MAVSDK as NED and is converted in `autonomy_core/tools/px4_runner.py` before it is stored on `GetTelemetry`:

```math
p = [north, east, -down]^T
```

```math
v = [v_north, v_east, -v_down]^T
```

The current yaw stored in telemetry is:

```math
\psi = radians(yaw\_deg)
```

with no sign flip in `px4_runner.py::track_orientation(...)`. Roll and pitch are also converted from degrees to radians and stored directly.

The planner, gate memory, race logic, trajectory sampler, and controller all consume this internal positive-up frame.

Camera-to-world conversion uses:

```math
p_{gate}^{world} = p_{vehicle}^{world} + R_{world,body}(roll,pitch,yaw) (p_{camera\_offset}^{body} + R_{body,camera} p_{gate}^{camera})
```

where `p_camera_offset_body = [0.12, 0.03, 0.242]^T` in `AutonomyAPI.__init__(...)`.

The current default perception transform mode is `physical_direct_rad_x_mirror`, whose camera-to-body matrix is:

```math
R_{body,camera} =
\begin{bmatrix}
0 & 0 & 1 \\
1 & 0 & 0 \\
0 & -1 & 0
\end{bmatrix}
```

from `AutonomyAPI.physical_x_mirror_body_camera_matrix(...)`.

Other available matrices include `AutonomyAPI.physical_body_camera_matrix(...)`:

```math
R_{body,camera} =
\begin{bmatrix}
0 & 0 & 1 \\
-1 & 0 & 0 \\
0 & -1 & 0
\end{bmatrix}
```

and `AutonomyAPI.legacy_body_camera_matrix(...)`:

```math
R_{body,camera} =
\begin{bmatrix}
-1 & 0 & 0 \\
0 & 0 & 1 \\
0 & -1 & 0
\end{bmatrix}
```

`R_world_body` is built from roll, pitch, yaw in `GatePerceptionNode._rpy_to_rotmat(...)`, reached through `AutonomyAPI.transform_gate_camera_to_world(...)`.

Gazebo truth conversion uses a separate diagnostic/sim path. `autonomy_core/debug/gazebo_diagnostics.py::_gazebo_model_pose_to_planner(...)` maps Gazebo world into planner world with:

```math
R_{world,gazebo\rightarrow planner} =
\begin{bmatrix}
0 & 1 & 0 \\
1 & 0 & 0 \\
0 & 0 & 1
\end{bmatrix}
```

and applies a body reflection:

```math
R_{body,planner\rightarrow gazebo} = diag(1,-1,1)
```

so:

```math
p_{planner} = R_{world,gazebo\rightarrow planner} p_{gazebo}
```

```math
R_{planner,body} = R_{world,gazebo\rightarrow planner} R_{gazebo,body} R_{body,planner\rightarrow gazebo}
```

### Scripts / Functions

- `autonomy_core/tools/px4_runner.py::track_position(...)`
- `autonomy_core/tools/px4_runner.py::track_velocity(...)`
- `autonomy_core/tools/px4_runner.py::track_orientation(...)`
- `autonomy_core/launch/get_telemetry.py::GetTelemetry.telemetry_pos(...)`
- `autonomy_core/launch/get_telemetry.py::GetTelemetry.telemetry_vel(...)`
- `autonomy_core/launch/get_telemetry.py::GetTelemetry.telemetry_rpy(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.perception_rpy_for_mode(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.body_camera_matrix_for_mode(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.transform_gate_camera_to_world(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.transform_gate_camera_to_world_for_pose_source(...)`
- `autonomy_core/debug/gazebo_diagnostics.py::_gazebo_model_pose_to_planner(...)`

### Verify In Code

- Exact semantic labels for the body axes are implicit in transform matrices and PX4 command usage. The code clearly uses positive-up `z`, but does not define a formal body-frame convention document.

### Future Improvement

- Add a dedicated frame-convention document and tests that assert camera, body, PX4/NED, planner, and Gazebo mappings with known poses.

## 2. Telemetry And Vehicle State

### Current Behavior

`VehicleState` is a passive alias of `State` in `autonomy_core/core/types.py`:

```python
VehicleState = State
```

with fields:

```math
state = (p, v, \psi)
```

where:

```math
p = [x,y,z]^T
```

```math
v = [v_x,v_y,v_z]^T
```

```math
\psi = yaw
```

`autonomy_core/core/state_adapter.py::vehicle_state_from_telemetry(...)` builds `VehicleState` from the current `GetTelemetry` object:

```math
p = [telemetry.pos["x"], telemetry.pos["y"], telemetry.pos["z"]]^T
```

```math
v = nan\_to\_num([telemetry.vel["vx"], telemetry.vel["vy"], telemetry.vel["vz"]]^T)
```

NaN, positive infinity, and negative infinity velocity values are replaced with `0.0`. Position is not sanitized in this adapter.

Yaw comes from:

```math
\psi = telemetry.rpy["yaw"]
```

Roll and pitch remain available in `telemetry.rpy` for perception transforms, but `VehicleState` only carries yaw for control.

### Scripts / Functions

- `autonomy_core/core/types.py::State`
- `autonomy_core/core/types.py::VehicleState`
- `autonomy_core/core/state_adapter.py::vehicle_state_from_telemetry(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.attitude_control(...)`
- `autonomy_core/control/attitude_control.py::compute_tracker_control_debug_fields(...)`

### Future Improvement

- Add timestamp and estimator-health fields to a competition state adapter without changing the existing `VehicleState` behavior until parity tests exist.

## 3. Perception: YOLO Detection To Gate Pose

### Current Behavior

`autonomy_core/perception/gate_perception_yolo.py::GatePerception` runs Ultralytics YOLO pose detection. It expects four semantic keypoints in this order:

```text
TL, TR, BR, BL
```

The 3D model points are the inner gate-opening corners with gate size `s_gate = 1.5 m` by default:

```math
s = s_{gate}/2
```

```math
P_{model} =
\{[-s,s,0], [s,s,0], [s,-s,0], [-s,-s,0]\}
```

YOLO candidate detection applies these checks in `detect_gate_candidates(...)`:

- bounding-box confidence at least `yolo_conf`
- at least four keypoints
- finite keypoint coordinates
- keypoints within image bounds with a `5 px` tolerance
- mean keypoint confidence at least `0.10`
- keypoint polygon area at least `50 px^2`

The YOLO candidate score is:

```math
score_{yolo} = box\_conf + 0.25\ mean\_keypoint\_conf + 0.10\ min(area/40000, 1)
```

Similar detections are deduplicated if both are true:

```math
\|c_i - c_j\| < 12 px
```

```math
\|size_i - size_j\| < 20 px
```

Pose estimation uses `cv2.solvePnPGeneric(..., SOLVEPNP_IPPE_SQUARE)` across allowed corner-order permutations and also tries `cv2.solvePnP(..., SOLVEPNP_ITERATIVE)` on the default order.

For a candidate with rotation `R`, translation `t`, normal `n = R[:,2]`, image points `u_i`, model points `P_i`, and camera model `(K, dist)`, reprojection error is:

```math
e_{reproj} = \sqrt{\frac{1}{N}\sum_i \|project(K,dist,R,t,P_i)-u_i\|^2}
```

implemented by `GatePerception.compute_reprojection_error(...)`.

Approximate size-derived depth is:

```math
z_{size,width} = f_x s_{gate} / width_{px}
```

```math
z_{size,height} = f_y s_{gate} / height_{px}
```

```math
z_{size} = mean(valid\ depths)
```

Candidate PnP scoring starts from negative reprojection error:

```math
score = -e_{reproj}
```

and applies penalties/bonuses in `GatePerception.score_pnp_candidate(...)`:

- non-finite reprojection: `-1e6`
- non-positive depth: `-1e6`
- depth outside `(0.5, 30.0) m`: `-50`
- grazing normal when `abs(n_z) < 0.15`: `-25`
- size-depth disagreement above `1.0 m`: `-10 * disagreement`
- lateral angle/image-center disagreement above `0.25`: `-25 * disagreement`
- normal bonus: `+0.25 * abs(n_z)`

where:

```math
lateral\_angle = t_x/t_z
```

```math
image\_center\_offset = (mean(u_x)-c_x)/f_x
```

AutonomyAPI then may rescore PnP candidates in world coordinates with `select_pnp_candidate_for_live_geometry(...)`. That score includes z-height error, reprojection error, size-depth error, range penalty, route-normal penalty, temporal track consistency, and active-target lateral jump penalty:

```math
score = -1.20e_z -0.20e_{reproj} -0.35e_{size\_depth} -0.15e_{range} -0.50e_{route} -0.80e_{temporal} -1.00e_{lateral} + bonus_{temporal}
```

If the best world-geometry score is below `-50`, the lowest-reprojection candidate is used as fallback.

After PnP, `AutonomyAPI.update_gate_memory_from_frame(...)` rejects detections with `reprojection_error > 6.0 px`, invalid image bounds, unsafe target z, too-far detections, or excessive jumps from the last valid target.

### Scripts / Functions

- `autonomy_core/perception/gate_perception_yolo.py::GatePerception.detect_gate_candidates(...)`
- `autonomy_core/perception/gate_perception_yolo.py::GatePerception.order_corners(...)`
- `autonomy_core/perception/gate_perception_yolo.py::GatePerception.estimate_pose(...)`
- `autonomy_core/perception/gate_perception_yolo.py::GatePerception.score_pnp_candidate(...)`
- `autonomy_core/perception/gate_perception_yolo.py::GatePerception.compute_reprojection_error(...)`
- `autonomy_core/perception/gate_perception_yolo.py::GatePerception.project_model_points(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.select_pnp_candidate_for_live_geometry(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.update_gate_memory_from_frame(...)`

### Future Improvement

- Retrain or validate YOLO for far gates and domain-randomized lighting, but keep that separate from refactor/adapter work.

## 4. Camera/Body/World Transforms

### Current Behavior

The production camera transform path for a camera-frame gate center is:

```math
p_{gate}^{body} = R_{body,camera} p_{gate}^{camera}
```

```math
p_{gate}^{world,mavsdk} = p_{vehicle}^{world} + R_{world,body}(r,p,\psi) (p_{camera\_offset}^{body} + p_{gate}^{body})
```

`AutonomyAPI.perception_rpy_for_mode(...)` chooses the yaw/roll/pitch convention. In the current default `physical_direct_rad_x_mirror`, raw telemetry radians are used directly with a possible yaw correction:

```math
\psi_{perception} = wrap\_pi(\psi_{telemetry} + \Delta\psi_{correction})
```

`AutonomyAPI.initialize_perception_yaw_correction(...)` can initialize yaw correction from Gazebo truth:

```math
\psi_{expected,planner} = wrap\_pi(\pi/2 - \psi_{gazebo})
```

```math
\Delta\psi = wrap\_pi(\psi_{expected,planner} - \psi_{telemetry})
```

`AutonomyAPI.transform_gate_camera_to_world_for_pose_source(...)` computes both MAVSDK and Gazebo-derived world estimates. With `perception_world_pose_source = "gazebo_truth_sim_only"`, the selected world point uses Gazebo truth if available, otherwise MAVSDK fallback. This is a sim-only behavior path, but it is currently reachable through the runtime facade when `gazebo_pose` is supplied.

### Scripts / Functions

- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.perception_rpy_for_mode(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.initialize_perception_yaw_correction(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.update_dynamic_gazebo_perception_yaw_correction(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.body_camera_matrix_for_mode(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.transform_gate_camera_to_world(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.transform_gate_camera_to_world_for_pose_source(...)`
- `autonomy_core/debug/gazebo_diagnostics.py::_gazebo_model_pose_to_planner(...)`

### Verify In Code

- `GatePerceptionNode._rpy_to_rotmat(...)` should be read with any future transform change. This document assumes the current ZYX convention implied by calls and comments.

### Future Improvement

- Add explicit transform unit tests with known camera points and poses before competition use.

## 5. Gate Memory / Tracking

### Current Behavior

`GateMemory` stores persistent `GateTrack` landmarks. Each observation stores world center, optional camera center, reprojection error, confidence, solver name, active gate index, and outlier flag.

Distance is Euclidean:

```math
d(a,b) = \|a-b\|_2
```

Association order in `GateMemory.add_detection(...)` is:

1. match nearest committed track
2. match nearest uncommitted track
3. reject new track if near a committed landmark
4. otherwise create a new candidate track

Uncommitted track centers are robustly updated with the median of recent measurements:

```math
c_{track} = median(m_{recent})
```

Committed tracks do not move on direct observations. They append residuals:

```math
e_{residual} = \|m - c_{track}\|_2
```

Filtering in `_update_track_filter(...)` uses the median world point to reject outliers:

```math
m_{median} = median(m_i)
```

```math
inlier_i = \|m_i - m_{median}\| \le max\_outlier\_distance
```

Filtered world and camera centers are means of inlier observations:

```math
c_{filtered,world} = mean(m_i^{world})
```

```math
c_{filtered,camera} = mean(m_i^{camera})
```

Standard deviations are component-wise standard deviations over inliers:

```math
\sigma_{world} = std(m_i^{world})
```

```math
\sigma_{camera} = std(m_i^{camera})
```

A track becomes stable when all of these pass:

- hits at least `min_hits_for_stable` (`6` in AutonomyAPI defaults)
- inlier count at least `min_hits_for_stable`
- `||center_world_std|| <= max_center_std_for_stable` (`0.45`)
- `||center_camera_std|| <= max_camera_std_for_stable` (`0.45`)
- median reprojection error finite and `<= max_reprojection_error_for_stable` (`5.0`), or reprojection unavailable
- observation span at least `min_observation_time`

Stability score is:

```math
score = 0.45s_{std} + 0.35s_{reproj} + 0.20s_{hits}
```

where:

```math
s_{std} = 1 - min(||\sigma_{world}||/max\_center\_std, 1)
```

```math
s_{reproj} = 1 - min(e_{reproj,median}/max\_reproj, 1)
```

```math
s_{hits} = min(hits/min\_hits, 2)/2
```

Commit criteria in `_maybe_commit(...)` are:

- hits at least `commit_hits` (`4` in AutonomyAPI defaults)
- confidence sum at least `commit_confidence_sum` (`1.2`)
- max recent spread from candidate center within `commit_spread_radius`
- not within `commit_radius` of an existing committed landmark

Duplicate committed tracks are merged when their centers are closer than `duplicate_merge_radius`, currently `max(5.0, 2.5 * estimated_gate_size)` with `estimated_gate_size = 2.0`.

Track merge uses hit-weighted center averaging:

```math
c_{target,new} = \frac{h_t c_t + h_s c_s}{h_t + h_s}
```

### Scripts / Functions

- `autonomy_core/perception/gate_memory.py::GateObservation`
- `autonomy_core/perception/gate_memory.py::GateTrack`
- `autonomy_core/perception/gate_memory.py::GateTrack.update(...)`
- `autonomy_core/perception/gate_memory.py::GateTrack.observe_without_moving(...)`
- `autonomy_core/perception/gate_memory.py::GateMemory.add_detection(...)`
- `autonomy_core/perception/gate_memory.py::GateMemory._update_track_filter(...)`
- `autonomy_core/perception/gate_memory.py::GateMemory._maybe_commit(...)`
- `autonomy_core/perception/gate_memory.py::GateMemory.merge_duplicate_committed_tracks(...)`
- `autonomy_core/perception/gate_memory.py::GateMemory.merge_track_into(...)`

### Future Improvement

- Replace heuristic filtering with a proper landmark estimator only after preserving current track/admission behavior in tests.

## 6. Race Order And Target Selection

### Current Behavior

`RaceProgression` is a cursor over persistent gate track IDs, not a spatial sorting system. If `race_gate_order` is provided, that order wins. Otherwise `race_progression.inferred_order` is built from admitted track IDs.

Key state fields:

- `race_order_track_ids`: current order exposed for logging
- `race_admitted_track_ids` / `race_accepted_track_ids`: admitted candidates
- `race_progression.cursor`: index of next gate to complete
- `completed_track_ids_this_cycle`: completed landmarks for the current lap/cycle

Target validation checks are intentionally geometric/safety checks, not semantic gate-opening checks.

`validate_perception_gate_center(...)` rejects:

- non-finite center
- `z < safe_min_target_z`
- `z > safe_max_target_z`
- vehicle distance greater than `max_detection_range`
- jump from `last_valid_target` greater than `max_gate_jump`

`validate_candidate_target(...)` rejects:

- invalid planning target
- near already completed gate
- duplicate committed track within `commit_radius`
- too far from vehicle
- implausible jump from the last completed valid gate:

```math
jump = \|c - c_{last\_completed}\|
```

```math
max\_jump = gate\_jump\_margin + max\_plausible\_gate\_speed \Delta t
```

reject if:

```math
jump > max\_jump
```

Race admission requires committed and, when `use_lookahead_gate_filter` is true, stable tracks. Duplicate accepted race gates within `duplicate_merge_radius` are merged.

`assign_race_order_from_progress(...)` chooses the nearest valid candidate as the current gate, then ranks future candidates by projected progress along an inferred course direction.

### Scripts / Functions

- `autonomy_core/launch/race_progression.py::RaceProgression`
- `autonomy_core/planning/target_validation.py::canonical_track_id(...)`
- `autonomy_core/planning/target_validation.py::validate_perception_gate_center(...)`
- `autonomy_core/planning/target_validation.py::is_near_completed_gate(...)`
- `autonomy_core/planning/target_validation.py::find_duplicate_committed_track(...)`
- `autonomy_core/planning/target_validation.py::validate_planning_target(...)`
- `autonomy_core/planning/target_validation.py::validate_candidate_target(...)`
- `autonomy_core/racing/race_admission.py::accept_track_into_race_order(...)`
- `autonomy_core/racing/race_admission.py::assign_race_order_from_progress(...)`
- `autonomy_core/racing/race_admission.py::refresh_race_order_from_memory(...)`

### Future Improvement

- Add explicit gate-opening validation as behavior work, not as a refactor.

## 7. Waypoint Horizon Construction

### Current Behavior

`build_waypoint_horizon(...)` returns:

```math
waypoints = [p_{current}, g_0, g_1, ...]
```

with parallel lists:

- `target_gates`
- `target_track_ids`
- `_planning_target_waypoint_types`

In GT mode, `build_waypoint_horizon_from_gt(...)` uses only the current GT gate because `path_plan(...)` passes `max_gates_ahead=1` when perception is disabled.

In perception mode, `build_waypoint_horizon_from_memory(...)` walks race order from the current race cursor and validates each target. The first target is normally `hard_current`; later stable race-order targets are `hard_stable`.

Lookahead appending in `_append_planning_lookahead_targets(...)` can add:

- `hard_stable`: stable committed tracks passing hit/range/duplicate/completed checks
- `soft_committed_unstable`: committed but unstable tracks that pass tentative lookahead checks
- `soft_tentative`: uncommitted tentative tracks or raw clamped candidates that pass tentative lookahead checks
- `terminal_extension`: a synthetic point past the active gate when only one gate is in the horizon and terminal extension is enabled

Terminal passthrough extension computes a direction from, in priority order:

1. next GT gate in GT navigation
2. perception gate normal
3. previous completed gate to current gate
4. approach vector
5. current vehicle velocity
6. vehicle-to-current-gate vector

and returns:

```math
p_{extension} = p_{gate} + d_{extension}\ \frac{direction}{\|direction\|}
```

where `d_extension = terminal_passthrough_extension_distance` (`4.0` by default).

### Scripts / Functions

- `autonomy_core/planning/target_horizon.py::build_waypoint_horizon(...)`
- `autonomy_core/planning/target_horizon.py::build_waypoint_horizon_from_memory(...)`
- `autonomy_core/planning/target_horizon.py::build_waypoint_horizon_from_gt(...)`
- `autonomy_core/planning/target_horizon.py::_append_planning_lookahead_targets(...)`
- `autonomy_core/planning/target_horizon.py::finalize_planning_horizon_debug(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.compute_terminal_passthrough_extension(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.path_plan(...)`

### Future Improvement

- Separate horizon policy from facade state only after sim acceptance for current target/race behavior.

## 8. Minimum-Snap Trajectory Generation

### Current Behavior

The planner is `autonomy_core/planning/minimum_snap_planner_multi_time_optimized.py::MultiSegmentMinimumSnapPlanner`.

Each segment uses a seventh-order polynomial per axis in local segment time:

```math
p_k(t) = \sum_{i=0}^{7} c_{k,i} t^i, \quad t \in [0,T_k]
```

The snap objective per axis is:

```math
J = \int_0^{T_k} \left(\frac{d^4 p_k(t)}{dt^4}\right)^2 dt
```

For coefficients `c`, the QP form is:

```math
minimize\quad \frac{1}{2} c^T Q c
```

```math
subject\ to\quad A c = b
```

The implementation solves the KKT system directly:

```math
\begin{bmatrix}
Q + \epsilon I & A^T \\
A & 0
\end{bmatrix}
\begin{bmatrix}
c \\
\lambda
\end{bmatrix}
=
\begin{bmatrix}
0 \\
b
\end{bmatrix}
```

with small regularization `epsilon = 1e-10`.

Constraints include:

- segment start/end positions equal supplied waypoints
- start velocity, acceleration, jerk
- end velocity, acceleration, jerk
- internal derivative continuity for orders 1 through 6

If internal waypoint velocities are supplied, derivative-1 continuity and derivative-6 continuity at that boundary are replaced by explicit end/start velocity constraints to keep the square `8M` system.

The polynomial derivative basis row is:

```math
\frac{d^r}{dt^r} t^i = \frac{i!}{(i-r)!} t^{i-r}
```

implemented as a falling factorial in `_basis_row(...)`.

Segment durations come from `trajectory_manager.choose_T(...)`. For segment distance `d`, max velocity `vmax`, max acceleration `amax`:

```math
t_{acc} = vmax/amax
```

```math
d_{acc} = 0.5\ amax\ t_{acc}^2
```

If `d > 2 d_acc`:

```math
T = 2t_{acc} + \frac{d - 2d_{acc}}{vmax}
```

else:

```math
T = 2\sqrt{d/amax}
```

Then the duration is adjusted for initial velocity along the segment direction and clamped to at least `T_min`.

`active_times` are the segment durations currently installed on `AutonomyAPI`. `active_target_crossing_tau(active_times, target_idx)` returns:

```math
\tau_{cross}(i) = \sum_{k=0}^{i} T_k
```

### Scripts / Functions

- `autonomy_core/planning/minimum_snap_planner_multi_time_optimized.py::MultiSegmentMinimumSnapPlanner.update(...)`
- `autonomy_core/planning/minimum_snap_planner_multi_time_optimized.py::MultiSegmentMinimumSnapPlanner._solve_axis(...)`
- `autonomy_core/planning/minimum_snap_planner_multi_time_optimized.py::MultiSegmentMinimumSnapPlanner._build_constraints(...)`
- `autonomy_core/planning/minimum_snap_planner_multi_time_optimized.py::MultiSegmentMinimumSnapPlanner._segment_Q(...)`
- `autonomy_core/planning/minimum_snap_planner_multi_time_optimized.py::MultiSegmentMinimumSnapPlanner.sample(...)`
- `autonomy_core/planning/minimum_snap_planner_multi_time_optimized.py::MultiSegmentMinimumSnapPlanner.sample_full(...)`
- `autonomy_core/planning/trajectory_manager.py::choose_T(...)`
- `autonomy_core/planning/trajectory_manager.py::allocate_segment_times(...)`
- `autonomy_core/planning/trajectory_manager.py::active_target_crossing_tau(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.path_plan(...)`

### Future Improvement

- Add dynamic feasibility/retiming if higher-speed trajectories need stricter acceleration, thrust, and gate-opening constraints.

## 9. Plan Validation

### Current Behavior

`autonomy_core/planning/plan_validator.py::validate_minimum_snap_geometry(...)` samples each segment and validates progress and z-corridor behavior.

For each segment from waypoint `p0` to `p1`:

```math
\Delta = p_1 - p_0
```

```math
L = \|\Delta\|
```

```math
\hat d = \Delta/L
```

Sampled progress is:

```math
s(t) = (p(t)-p_0)^T \hat d
```

Backward progress is max loss from the best previous progress:

```math
max\_backward = \max_t (\max_{\tau<t} s(\tau) - s(t))
```

Segment overshoot is:

```math
max\_overshoot = \max(0, \max(s)-L, -\min(s))
```

Negative progress velocity counts samples, except near the endpoint margin, where:

```math
\dot s(t) = v(t)^T \hat d < -10^{-3}
```

Z-corridor validation samples `z(t)` and computes a segment floor. If start z is already below the safe minimum, the floor is relaxed to:

```math
z_{floor} = z_{start} - z_{corridor\_tolerance}
```

otherwise:

```math
z_{floor} = \max(safe\_min\_target\_z - z_{corridor\_tolerance}, \min(z_{start},z_{end}) - z_{endpoint\_undershoot\_tolerance})
```

Failure if:

```math
z_{floor} - \min(z(t)) > 0
```

AutonomyAPI default thresholds are passed from `AutonomyAPI.validate_minimum_snap_geometry(...)`:

- backward tolerance: `0.15 m`
- overshoot tolerance: `0.35 m`
- negative velocity tolerance: `4` samples
- endpoint margin fraction: `0.08`
- z corridor tolerance: `0.05 m`
- z endpoint undershoot tolerance: `0.20 m`

If the z-corridor validation fails and the current vertical velocity is downward, `path_plan(...)` retries with:

```math
v_{start,z} = 0
```

when original `v_start_z < 0`.

If a multi-target horizon fails validation, `path_plan(...)` attempts an active-gate-only fallback, optionally with terminal extension.

### What Is Not Currently Validated

- gate-opening crossing through the physical aperture
- full thrust feasibility over the trajectory
- tilt feasibility over the entire trajectory
- collision with gate frame or obstacles
- model-predictive contouring/progress constraints

### Scripts / Functions

- `autonomy_core/planning/plan_validator.py::validate_minimum_snap_geometry(...)`
- `autonomy_core/planning/plan_validator.py::reset_plan_geometric_validation_debug(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.validate_minimum_snap_geometry(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.path_plan(...)`

### Future Improvement

- Add full dynamic feasibility and gate-opening validation in a behavior branch.

## 10. Same-Plan Continuation And Pending Suffix

### Current Behavior

After a gate completion, the priority is:

1. continue the already installed plan if the next target is already in it
2. install a pending suffix if valid
3. fall back to a normal full replan

Same-plan continuation in `_continue_existing_plan_after_completion(...)` preserves the active planner and clock. It only advances `current_target_idx` and active target metadata when the next race-order track is the next target in the installed plan.

Pending suffix creation is used for future-only replans. `prepare_pending_suffix_for_future_only_replan(...)` samples the active plan at the current active target crossing time:

```math
\tau_{splice} = \sum_{k=0}^{current\_target\_idx} T_k
```

Then it samples:

```math
(p_s, v_s, a_s, j_s, snap_s) = planner.sample\_full(\tau_{splice})
```

The suffix waypoints are:

```math
[p_s, g_{future,0}, g_{future,1}, ...]
```

The suffix planner uses boundary conditions:

```math
v_{start}=v_s,\quad a_{start}=a_s,\quad j_{start}=j_s
```

```math
v_{end}=0,\quad a_{end}=0,\quad j_{end}=0
```

The suffix is validated with the same minimum-snap geometry validator. It is installed after completion only if:

- pending suffix is valid
- splice track matches completed track
- first suffix track matches the next race target if a next target exists
- current vehicle position is within `2.0 m` of suffix start

### Scripts / Functions

- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI._continue_existing_plan_after_completion(...)`
- `autonomy_core/planning/suffix_planner.py::reset_pending_suffix_state(...)`
- `autonomy_core/planning/suffix_planner.py::prepare_pending_suffix_for_future_only_replan(...)`
- `autonomy_core/planning/suffix_planner.py::install_pending_suffix_after_completion(...)`
- `autonomy_core/planning/trajectory_manager.py::active_target_crossing_tau(...)`
- `autonomy_core/tools/px4_runner.py` replan loop around future-only replan reasons

### Future Improvement

- Add suffix/replan equivalence tests using recorded plans before changing suffix policy.

## 11. Reference Sampling And Virtual Clock

### Current Behavior

`AutonomyAPI.attitude_control(...)` computes wall-clock trajectory time:

```math
\tau_{wall} = time.time() - trajectory\_start\_time
```

Then `compute_reference_sample_tau(...)` computes the actual sample time. Initial value:

```math
\tau_{sample} = \min(\max(0,\tau_{wall}), planner.total\_time)
```

In perception mode, if an active plan exists and the race is not complete, the virtual clock limits reference progress based on the nearest XY point on the active plan:

```math
\tau_{vehicle} = \arg\min_{\tau_i} \|p_{plan,xy}(\tau_i) - p_{vehicle,xy}\|
```

where the current implementation searches `reference_projection_sample_count` evenly spaced samples.

Allowed sample time is:

```math
\tau_{allowed} = \min(planner.total\_time, \tau_{vehicle} + reference\_progress\_tau\_lead\_s)
```

```math
\tau_{sample} = \min(\tau_{sample}, \tau_{allowed})
```

Finally, sample tau is monotonic per active plan:

```math
\tau_{sample} = \max(previous\_sample\_tau\_used, \tau_{sample})
```

and clamped to total plan time.

The sampled reference is:

```math
(p_{ref}, v_{ref}, a_{ref}) = planner.sample(\tau_{sample})
```

In perception mode, `clamp_reference_altitude(...)` applies a lower z clamp:

```math
p_{ref,z} = \max(p_{ref,z}, safe\_min\_target\_z)
```

and if clamped:

```math
v_{ref,z} = \max(v_{ref,z}, 0)
```

```math
a_{ref,z} = \max(a_{ref,z}, 0)
```

### Scripts / Functions

- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.nearest_tau_on_active_plan_xy(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.compute_reference_sample_tau(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.clamp_reference_altitude(...)`
- `autonomy_core/planning/minimum_snap_planner_multi_time_optimized.py::MultiSegmentMinimumSnapPlanner.sample(...)`
- `autonomy_core/planning/minimum_snap_planner_multi_time_optimized.py::MultiSegmentMinimumSnapPlanner.sample_full(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.attitude_control(...)`

### Future Improvement

- Replace the XY sampled nearest-point virtual clock with contouring/progress MPC if competition behavior needs tighter progress control.

## 12. Geometric / PD Tracker

### Current Behavior

The controller is `autonomy_core/controller/attitude_controller3.py::RPGHighLevelTracker`.

Position and velocity errors are:

```math
e_p = p_{ref} - p
```

```math
e_v = v_{ref} - v
```

Feedback acceleration is:

```math
a_{fb} = K_p \odot e_p + K_v \odot e_v
```

Feedforward plus feedback, before gravity, is:

```math
a_{cmd,no\_g,raw} = a_{ref} + a_{fb}
```

Acceleration limiting is applied before gravity:

```math
\|a_{xy}\| \le max\_acc\_xy
```

```math
-max\_acc\_z\_down \le a_z \le max\_acc\_z\_up
```

Gravity compensation in positive-up world frame is:

```math
a_{des} = a_{cmd,no\_g} + [0,0,g]^T
```

Desired body z-axis:

```math
z_b^{des} = \frac{a_{des}}{\|a_{des}\|}
```

Given desired yaw `psi_des`, heading helper axes are:

```math
x_c = [\cos\psi, \sin\psi, 0]^T
```

```math
y_c = [-\sin\psi, \cos\psi, 0]^T
```

Desired body axes:

```math
x_b^{des} = normalize(y_c \times z_b^{des})
```

```math
y_b^{des} = normalize(z_b^{des} \times x_b^{des})
```

```math
R_{des} = [x_b^{des}\ y_b^{des}\ z_b^{des}]
```

Roll, pitch, yaw are extracted with ZYX convention:

```math
R = R_z(yaw) R_y(pitch) R_x(roll)
```

Roll and pitch are clamped:

```math
roll,pitch \in [-max\_tilt, max\_tilt]
```

Yaw command is the wrapped reference yaw:

```math
\psi_{cmd} = wrap\_pi(\psi_{ref})
```

Thrust mapping is a normalized approximation around hover:

```math
thrust_{raw} = thrust_{hover} + gain\ a_{cmd,no\_g,z}
```

where default `gain = 1/g` unless overridden.

Then:

```math
thrust = clamp(thrust_{raw}, thrust_{min}, thrust_{max})
```

AutonomyAPI flips roll and pitch after tracker output:

```math
roll_{cmd} = -roll_{tracker}
```

```math
pitch_{cmd} = -pitch_{tracker}
```

Tracker debug fields record errors, feedforward, feedback, limited acceleration, raw/clamped thrust, and vertical thrust after tilt:

```math
thrust_{vertical} = thrust\ cos(roll)\ cos(pitch)
```

### Scripts / Functions

- `autonomy_core/controller/attitude_controller3.py::RPGHighLevelTracker.update(...)`
- `autonomy_core/controller/attitude_controller3.py::RPGHighLevelTracker._limit_acceleration(...)`
- `autonomy_core/controller/attitude_controller3.py::RPGHighLevelTracker._construct_R_des(...)`
- `autonomy_core/controller/attitude_controller3.py::rotmat_to_euler_zyx(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.attitude_control(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.hold_no_target_control(...)`
- `autonomy_core/control/attitude_control.py::compute_tracker_control_debug_fields(...)`

### Future Improvement

- Add translational MPC or contouring MPC in a behavior branch, not inside the current refactor seam.

## 13. Yaw Policy

### Current Behavior

Yaw policy is owned by `AutonomyAPI`, not the extracted debug helper.

Generic reference yaw helper:

```math
\psi_{des} = atan2(v_y, v_x)\quad if\ \|v_{xy}\| > eps
```

else:

```math
\psi_{des} = atan2(a_y, a_x)\quad if\ \|a_{xy}\| > eps
```

else last yaw is reused.

In perception mode with a valid current target, active-target yaw is:

```math
\psi_{target} = atan2(g_y - p_y, g_x - p_x)
```

and `yaw_target_source = "active_target_camera_axis"`.

If perception is enabled and the last perception update was not accepted, current behavior sets:

```math
\psi_{des} = \psi_{current}
```

and `yaw_target_source = "perception_lost_hold_current_yaw"`.

Yaw unwrapping and rate limiting are in `continuous_yaw_command(...)`:

```math
\psi_{unwrap} = \psi_{prev} + wrap\_pi(\psi_{raw} - \psi_{prev})
```

```math
\Delta\psi = \psi_{unwrap} - \psi_{prev}
```

```math
|\Delta\psi| \le max\_yaw\_rate\ \Delta t
```

`seed_yaw_hold(...)` initializes all yaw hold/reference fields to the current yaw. `get_perception_yaw_hold_reference(...)` prefers `ref_yaw`, `previous_yaw_cmd`, `last_desired_yaw`, then `perception_hold_yaw` before falling back to telemetry yaw.

### Scripts / Functions

- `autonomy_core/launch/autonomy_api6.py::compute_desired_yaw(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.get_perception_yaw_hold_reference(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.seed_yaw_hold(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.continuous_yaw_command(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.hold_no_target_control(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.attitude_control(...)`
- `autonomy_core/tools/px4_runner.py::hover_command(...)`

### Future Improvement

- Do not change yaw as part of refactor. Any yaw behavior change should be a separate behavior branch with sim acceptance.

## 14. Gate Completion / Crossing Logic

### Current Behavior

In perception mode, `advance_gate_if_needed(...)` checks the current active target. It validates the target, checks crossing geometry, verifies cursor/track ID consistency, then advances race state.

Approach geometry is initialized in `set_active_perception_target_geometry(...)`:

```math
approach = g - p_{start}
```

```math
\hat a = approach/\|approach\|
```

Gate pass geometry in `compute_gate_pass_geometry(...)` uses:

```math
rel = p_{vehicle} - g
```

```math
progress = rel^T \hat a
```

```math
lateral = rel - progress\ \hat a
```

```math
e_{lateral} = \|lateral\|
```

Plane crossing is detected when previous progress was non-positive and current progress is non-negative:

```math
progress_{prev} \le 0 \le progress
```

Completion requires both:

```math
e_{lateral} < gate\_pass\_radius
```

and either:

```math
progress > gate\_progress\_threshold
```

or plane crossing is true.

If close to the target by race pass radius but not complete, debug fields show whether lateral error or plane progress blocked completion.

On completion, `advance_gate_if_needed(...)`:

- records completed track ID and completed position
- records crossing debug against GT gate when available
- increments race cursor
- increments `current_gate_idx`
- updates completed counts
- tries same-plan continuation
- tries pending suffix installation
- tries full replan fallback
- clears target and enters no-target hold if no next target installs

In GT mode, completion is simpler:

```math
\|p - g\| < threshold
```

then `current_gate_idx += 1`.

### Scripts / Functions

- `autonomy_core/racing/gate_advancement.py::compute_gate_pass_geometry(...)`
- `autonomy_core/racing/gate_advancement.py::reset_crossing_debug(...)`
- `autonomy_core/racing/gate_advancement.py::clear_active_perception_target(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.set_active_perception_target_geometry(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.advance_gate_if_needed(...)`
- `autonomy_core/launch/race_progression.py::RaceProgression`

### Current Limitation

- There is no explicit gate-opening validation that proves the vehicle passed through the physical gate aperture. Lateral gate pass radius is the current proxy.

## 15. Logging And Sidecar Plan Math

### Current Behavior

`autonomy_core/tools/flight_logger.py::FlightLogger` writes:

- `flight_log.csv`: main per-loop telemetry, command, reference, perception, race, planning, validation, yaw, transform, and debug fields
- `flight_log_plans.csv`: sidecar installed-plan samples

The sidecar is populated by `AutonomyAPI.record_installed_plan_for_export(...)`. Each installed plan increments:

```math
active\_plan\_id = active\_plan\_id + 1
```

It samples `installed_plan_sample_count` evenly spaced `tau` values from `0` to `planner.total_time`:

```math
\tau_i \in linspace(0, T_{total}, N)
```

For each sample:

```math
(p_i, v_i, a_i) = planner.sample\_full(\tau_i)
```

and logs:

- plan ID
- plan source
- replan reason
- tau
- x/y/z
- vx/vy/vz
- ax/ay/az
- active segment durations
- active target track IDs
- waypoint types

The main log includes actual position/velocity, commands, tracker debug fields, `p_ref/v_ref/a_ref`, active target, active plan ID, gate/race state, perception/PnP/debug fields, validation fields, yaw fields, and transform diagnostics.

These logs can compare actual vs sampled reference vs installed plan by joining main rows on `active_plan_id` and sidecar rows on `plan_id`.

### Scripts / Functions

- `autonomy_core/tools/flight_logger.py::FlightLogger.__init__(...)`
- `autonomy_core/tools/flight_logger.py::FlightLogger.log_installed_plan_rows(...)`
- `autonomy_core/tools/flight_logger.py::FlightLogger.log(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.record_installed_plan_for_export(...)`
- `autonomy_core/tools/px4_runner.py` control/replan loop and logger calls

### Future Improvement

- Wire `LoggingConfig` modes only after preserving schema compatibility and bounding I/O in competition mode.

## 16. Gazebo Diagnostics / GT Math

### Current Behavior

Gazebo diagnostics are partially isolated in `autonomy_core/debug/gazebo_diagnostics.py`, but Gazebo truth can still affect selected perception world pose when `perception_world_pose_source = "gazebo_truth_sim_only"` and `gazebo_pose` is supplied to `update_gate_memory_from_frame(...)`.

Debug conversion from Gazebo model pose to planner pose is:

```math
p_{planner} =
\begin{bmatrix}
0 & 1 & 0 \\
1 & 0 & 0 \\
0 & 0 & 1
\end{bmatrix}
p_{gazebo}
```

```math
R_{planner,body} = R_{swap} R_{gazebo,body} diag(1,-1,1)
```

`capture_gazebo_pose_debug(...)` stores Gazebo model/camera pose, computes pose age, yaw, and MAVSDK-minus-Gazebo position/yaw deltas.

`compute_gazebo_pose_gate_comparison_debug(...)` computes gate-world estimates through uncorrected MAVSDK yaw, corrected perception yaw, and Gazebo yaw, then compares them to the configured GT gate:

```math
e_{mavsdk\rightarrow gt} = \|p_{gate}^{mavsdk} - p_{gate}^{gt}\|
```

```math
e_{gazebo\rightarrow gt} = \|p_{gate}^{gazebo} - p_{gate}^{gt}\|
```

It also solves a debug yaw needed to align the pitch/roll-adjusted body vector with the GT horizontal bearing:

```math
\psi_{required} = atan2(\Delta_y,\Delta_x) - atan2(b_y,b_x)
```

`AutonomyAPI.apply_diagnostic_far_depth_correction(...)` is an opt-in diagnostic path. By default `use_diagnostic_far_depth_correction = False`. If enabled, future detections with depth above thresholds can scale camera depth before world transform:

```math
p_{camera,corrected} = factor\ p_{camera}
```

where factor is `1/0.95` for depth at least `8 m`, and `1/0.915` for depth above `14 m`.

### Scripts / Functions

- `autonomy_core/debug/gazebo_diagnostics.py::_gazebo_model_pose_to_planner(...)`
- `autonomy_core/debug/gazebo_diagnostics.py::reset_gazebo_pose_comparison_debug(...)`
- `autonomy_core/debug/gazebo_diagnostics.py::capture_gazebo_pose_debug(...)`
- `autonomy_core/debug/gazebo_diagnostics.py::compute_gazebo_pose_gate_comparison_debug(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.transform_gate_camera_to_world_for_pose_source(...)`
- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI.apply_diagnostic_far_depth_correction(...)`
- `autonomy_core/tools/px4_runner.py::PerceptionNode.gazebo_pose_callback(...)`

### Production Caveat

- Competition/production runs must not pass or use Gazebo truth as live pose input. The default `gazebo_truth_sim_only` source must be guarded or replaced before real competition use.

## 17. Known Current Limitations

- The planner is not full kinodynamic planning. It solves fixed-duration minimum-snap splines and then validates selected geometric properties.
- There is no MPC yet. The controller is a geometric/PD high-level tracker with acceleration limiting and normalized thrust mapping.
- Vehicle limits and gains are still mostly hard-coded in runtime. `autonomy_core/core/config.py` is scaffolding and not authoritative yet.
- Gate-opening completion validation is not implemented. Current perception completion uses approach-plane and lateral-radius checks.
- Perception/PnP estimates can bias target centers, especially at range, and GateMemory filtering is heuristic.
- The virtual clock is XY/progress based and sampled, not a full contouring MPC or continuous closest-point projection.
- Logging can be heavy in debug mode because the main CSV, plan sidecar, and perception debug-frame paths carry many fields.
- Gazebo truth diagnostics must be disabled or isolated for competition. Current sim defaults can use Gazebo truth as selected perception pose source when supplied.
- Current command interface remains roll/pitch/yaw/thrust; no competition command adapter is implemented yet.

## 18. Future Math Upgrades

### Translational MPC

Replace or augment the PD/geometric outer loop with an MPC that optimizes future position, velocity, acceleration, thrust, and tilt constraints over a horizon.

Possible objective:

```math
\min \sum_k \|p_k-p_{ref,k}\|_Q^2 + \|v_k-v_{ref,k}\|_R^2 + \|u_k\|_S^2
```

subject to discrete dynamics and actuator constraints.

### Contouring / Progress MPC

Replace the sampled XY virtual clock with a progress state and contouring error:

```math
e_c = p - p_{path}(s)
```

```math
\dot s \ge 0
```

and optimize progress while penalizing contouring error.

### Dynamic Feasibility / Retiming

Retiming could explicitly enforce acceleration, jerk, thrust, tilt, and velocity limits instead of relying on fixed segment-time heuristics and post-validation.

### Gate-Opening Validation

Add explicit gate aperture geometry checks. For example, transform the vehicle crossing point into a gate-local frame and require:

```math
|x_{gate}| < opening\_half\_width - margin
```

```math
|z_{gate}| < opening\_half\_height - margin
```

This is behavior work and must be validated in sim.

### Camera / State Estimator Abstraction

Add competition adapters for image/camera metadata and vehicle odometry. Preserve the current `AutonomyAPI` facade until adapter parity is proven.

### VIO / EKF Integration

Replace or supplement MAVSDK telemetry with an estimator abstraction that provides covariance, freshness, and frame IDs.

### YOLO / Domain-Randomized Training

Train and validate YOLO on far-gate, lighting, blur, and viewpoint variation. Keep model-threshold changes separate from refactor changes.
