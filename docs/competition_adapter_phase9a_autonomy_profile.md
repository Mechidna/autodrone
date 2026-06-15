# Phase 9A Competition-Safe AutonomyAPI Profile

Status: implemented as a construction/profile boundary.

## Purpose

Phase 9A keeps the existing `autonomy_core/launch/autonomy_api6.py`
`AutonomyAPI` facade, but prevents competition code from using legacy Gazebo
truth defaults directly.

This is not a planner, controller, YOLO, PnP, race progression, or gate
advancement rewrite.

## Added Boundary

Primary module:

- `autonomy_core/runtime/competition_autonomy_factory.py`

The module is import-safe:

- It does not import `AutonomyAPI` on import.
- It does not import `cv2`, `pymavlink`, MAVSDK, ROS, or `rclpy`.
- It does not open sockets.
- It does not instantiate real perception.

`competition_setup.py` now routes explicit real-autonomy construction through
this factory when `use_real_autonomy=True`.

## Profile Behavior

The default competition profile:

- sets `perception_world_pose_source = "mavsdk"`
- sets `perception_world_pose_source_used = "mavsdk"`
- sets `save_perception_debug_frames = False`
- sets `use_diagnostic_far_depth_correction = False`
- clears `image_gazebo_pose_snapshot`
- marks the object with `competition_autonomy_profile_active = True`
- records `competition_autonomy_profile_name`
- keeps command publication outside `AutonomyAPI`

The profile rejects:

- `perception_world_pose_source = "gazebo_truth_sim_only"`
- enabled diagnostic far-depth correction
- enabled debug-frame writes
- extra constructor kwargs that contain Gazebo/truth/pose-snapshot fields
- real perception if it would silently rely on the legacy hardcoded YOLO path

For a temporary dry-run using the existing hardcoded YOLO default, the caller
must explicitly set `allow_legacy_yolo_default=True`. This is intentionally
visible and should be replaced later by a YOLO-path-aware config/factory if
needed.

## What This Does Not Claim

- Phase 9B real perception dry-run is not started.
- Phase 9C command candidate dry-run is not started.
- Phase 9D real competition simulator evidence is not started.
- Phase 4B remains blocked pending real receive-only competition simulator
  telemetry evidence.
- Command readiness, race readiness, and competition readiness are not claimed.

## Tests

Deterministic tests cover:

- import safety
- fake-factory construction without real `AutonomyAPI`
- safe field application
- rejection of Gazebo truth pose source
- rejection of unsafe diagnostic/debug settings
- explicit acknowledgment required for legacy YOLO defaults
- setup integration through `competition_setup.py`
