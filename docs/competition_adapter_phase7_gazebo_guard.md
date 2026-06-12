# Competition Adapter Phase 7 Gazebo Guard

Date: 2026-06-12.

Status: implemented for the current competition runner boundary.

## Scope

Phase 7 hardens the competition-mode guard that protects the adapter runner
from Gazebo truth paths. It does not add live transport, command publication,
`command_live`, or `race` behavior.

The guard remains import-safe and does not import `pymavlink`, `cv2`, Gazebo
diagnostics, or `AutonomyAPI`.

## Rejected In Competition Mode

The guard rejects:

- `perception_world_pose_source = "gazebo_truth_sim_only"`.
- Any non-`None` `gazebo_pose` passed toward perception updates.
- Runner or image metadata containing Gazebo model pose fields.
- Runner or image metadata containing Gazebo camera pose fields.
- Runner or image metadata containing Gazebo TF/transform fields.
- Gazebo-truth pose snapshots, including non-empty `image_pose_snapshot` and
  `image_gazebo_pose_snapshot` fields.
- Enabled Gazebo-only diagnostic far-depth correction flags such as
  `use_diagnostic_far_depth_correction = true`.

The competition runner boundary calls the guard at startup and before any
injected fake autonomy `update_gate_memory_from_frame(...)` call. Perception
kwargs returned through the guard always set `gazebo_pose = None`.

## Preserved Outside Competition Mode

`CompetitionGuard(competition_mode=False)` remains passive for sim/Gazebo
diagnostic settings. Existing Gazebo diagnostics, logger schema, `px4_runner.py`,
and `AutonomyAPI` internals were not modified.

## Command And Telemetry Status

Live command modes remain disabled:

- `command_live` fails closed.
- `race` fails closed.
- `command_dry_run` remains no-send and records Phase 4B/Phase 6A blocking
  reasons.

Phase 4B remains blocked pending real receive-only telemetry evidence from the
competition simulator. This guard work does not claim telemetry readiness or
command readiness.
