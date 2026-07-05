# Competition Adapter Phase 0 Inventory

Date: 2026-06-11
Branch: `feature/competition-adapter-plan`
Source plan: `docs/competition_adapter_plan.md`

This is Phase 0 only. It records current seams, protected paths, and known risks before adapter implementation. It does not authorize runtime behavior changes.

## Scope Guard

Completed in this phase:

- Ran required pre-touch git checks.
- Inspected the current competition adapter plan.
- Confirmed current branch and dirty-worktree state.
- Confirmed there was no existing test layout and added only minimal test-layout scaffolding.
- Inventoried current `AutonomyAPI` facade seams.
- Inventoried current state, image, command, runner, Gazebo-truth, and logging seams.
- Read vendor example files under `third_party/PyAIPilotExample/**` as immutable reference code only.

Not done in this phase:

- No competition guard implementation.
- No config constants implementation.
- No UDP vision adapter.
- No state adapter.
- No command adapter.
- No competition runner.
- No behavior tests.
- No planner, controller, perception, PnP, YOLO, race progression, logging, Gazebo diagnostic, or existing runner changes.
- No live simulator communication.
- No dependency installs or updates.
- No formatters, lint autofix, codemods, cleanup scripts, or dependency update tools.

## Required Pre-Touch Checks

Commands requested by the plan were run before file edits:

```bash
git status --short
git branch --show-current
git status --short -- third_party/pilot || true
git diff --name-only -- third_party/pilot || true
git diff --cached --name-only -- third_party/pilot || true
```

Observed output summary:

- Current branch: `feature/competition-adapter-plan`.
- Dirty worktree before Phase 0 edits included:
  - `D __pycache__/__init__.cpython-312.pyc`
  - `?? autonomy_core/debug_frames/`
  - `?? "flight logs/"`
  - `?? third_party/`
- Vendor path status:
  - `?? third_party/PyAIPilotExample/`
  - No unstaged diff output for files inside `third_party/PyAIPilotExample/**`.
  - No cached diff output for files inside `third_party/PyAIPilotExample/**`.

Phase 0 did not edit, format, move, rename, delete, generate into, stage, or commit anything under `third_party/PyAIPilotExample/**`.

## Protected Vendor Reference

`third_party/PyAIPilotExample/**` is immutable vendor/reference code.

Allowed:

- Read and inspect.
- Import or execute for reference if needed by a future phase.
- Copy into `tmp_path`, `/tmp`, or another generated test directory if mutable test fixtures are needed later.

Forbidden:

- Edit.
- Format.
- Move.
- Rename.
- Delete.
- Generate files into the vendor tree.
- Patch examples to make tests pass.
- Stage or commit vendor changes.

Vendor files inspected read-only in this phase:

- `third_party/PyAIPilotExample/vision_rx.py`
- `third_party/PyAIPilotExample/mavlink_rx.py`
- `third_party/PyAIPilotExample/controller.py`

## Test Layout Inventory

No existing `test`, `tests`, or `testing` directory was found outside `.git` and the protected vendor tree.

Root project test/config files found:

- `pyproject.toml`

`pyproject.toml` currently has:

- Package metadata for `autonomy-core`.
- No project runtime dependencies declared.
- No pytest configuration.
- No test dependency group.

Phase 0 scaffolding added:

- `tests/README.md`

No behavior tests were added.

## Active Runtime Facade

The current active facade used by the existing runner is:

- `autonomy_core/launch/autonomy_api6.py::AutonomyAPI`

Evidence:

- `autonomy_core/tools/px4_runner.py` imports `AutonomyAPI` from `autonomy_core.launch.autonomy_api6`.
- `px4_runner.py` instantiates `AutonomyAPI(use_perception=use_perception, race_gate_count=3)`.

Facade methods to preserve for the competition adapter:

- `AutonomyAPI.update_gate_memory_from_frame(...)`
- `AutonomyAPI.path_plan(...)`
- `AutonomyAPI.advance_gate_if_needed(...)`
- `AutonomyAPI.attitude_control()`

Phase 0 constraint:

- Do not refactor or modify `advance_gate_if_needed(...)`.
- Do not change planner, controller, perception, PnP, YOLO, logging, Gazebo diagnostics, or existing runner behavior.

## Current Data Contracts

Shared passive contracts already exist in `autonomy_core/core/types.py`:

- `VehicleState`
- `CameraFrame`
- `CameraModel`
- `ControlCommand`
- `Reference`
- `GateTarget`
- `PlanDebugInfo`

Current telemetry adapter seam:

- `autonomy_core/core/state_adapter.py::vehicle_state_from_telemetry(telemetry)`

Current adapter behavior:

- Reads `telemetry.pos["x"]`, `telemetry.pos["y"]`, `telemetry.pos["z"]`.
- Reads `telemetry.vel["vx"]`, `telemetry.vel["vy"]`, `telemetry.vel["vz"]`.
- Sanitizes velocity NaN/Inf values to zero.
- Reads `telemetry.rpy["yaw"]`.
- Returns internal z-up `VehicleState`.

Risk:

- The current state adapter assumes position, velocity, and yaw are already in the internal convention before reaching `AutonomyAPI`.
- The competition adapter must handle MAVLink/NED at the protocol boundary without changing this internal convention.

## Current State And Telemetry Path

Existing PX4 runner telemetry path:

- Uses MAVSDK `position_velocity_ned()`.
- Converts `position.down_m` to internal z-up by negating it.
- Converts `velocity.down_m_s` to internal z-up by negating it.
- Uses MAVSDK `attitude_euler()` and converts roll, pitch, yaw degrees to radians.
- Writes values into the shared `GetTelemetry` object.
- `AutonomyAPI.attitude_control()` calls `vehicle_state_from_telemetry(self.telemetry)`.

Vendor MAVLink reference observations from `third_party/PyAIPilotExample/mavlink_rx.py`:

- Handles `HEARTBEAT`.
- Handles `TIMESYNC`.
- Handles `ATTITUDE`.
- Handles `LOCAL_POSITION_NED`.
- Handles `ODOMETRY`.
- Handles `HIGHRES_IMU`.
- Handles `ENCAPSULATED_DATA`.
- Handles `ACTUATOR_OUTPUT_STATUS`.
- Handles `COLLISION`.
- `LOCAL_POSITION_NED` fields are read as `x`, `y`, `z`, `vx`, `vy`, `vz`, `time_boot_ms`.
- `ODOMETRY` fields are read as position, quaternion, velocity, body rates, `time_usec`, and reset count.

Risk:

- The plan still requires live or provided-example observe-mode verification that the simulator actually emits usable local position/odometry during the current competition scenario.
- If local position/odometry is absent, state estimation becomes a P0 blocker before command-enabled runs.

## Current Image And Perception Path

Existing PX4 runner image path:

- Uses ROS2 camera frames and camera info through `px4_runner.py`.
- Waits for `perception_node.frame`, `perception_node.camera_matrix`, and `perception_node.dist_coeffs`.
- Calls `AutonomyAPI.update_gate_memory_from_frame(...)`.
- Passes `gazebo_pose=perception_node.latest_gazebo_pose_snapshot()` in current sim path.

Current perception input seam:

- `AutonomyAPI.update_gate_memory_from_frame(frame, camera_matrix, dist_coeffs, image_stamp_sec=0, image_stamp_nanosec=0, image_received_wall_time=np.nan, image_pose_snapshot=None, gazebo_pose=None)`

Current perception behavior:

- `GatePerception` in `gate_perception_yolo.py` imports `ultralytics.YOLO`.
- YOLO model path, confidence, image size, and device are hard-coded in current `AutonomyAPI` defaults.
- YOLO/PnP behavior is owned by the current perception path and must not be changed in Phase 0.
- `GatePerceptionNode` documents OpenCV optical frame as x-right, y-down, z-forward, and body/world convention as x-forward, y-left, z-up.

Vendor vision reference observations from `third_party/PyAIPilotExample/vision_rx.py`:

- UDP port is `5600`.
- Header format is `<IHHIIQ`.
- Header size is computed with `struct.calcsize(header_format)`.
- Header fields are `frame_id`, `chunk_id`, `total_chunks`, `jpeg_size`, `payload_size`, `sim_time_ns`.
- JPEG chunks are reassembled by `frame_id`.
- JPEG decode uses `cv2.imdecode(..., cv2.IMREAD_COLOR)`.

Risk:

- Current runner image source is ROS2, not the competition UDP JPEG stream.
- Current production camera constants from VADR-TS-002 are not implemented yet.
- Current Gazebo pose injection must be guarded in later phases before competition mode.

## Current Command Path

Existing command seam:

- `AutonomyAPI.attitude_control()` returns `(roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd)`.

Existing PX4 runner command path:

- Converts roll, pitch, and yaw from radians to degrees for MAVSDK `Attitude`.
- Sends `mavsdk.offboard.Attitude(roll_deg=..., pitch_deg=..., yaw_deg=..., thrust_value=...)`.
- Uses a 50 Hz loop in existing runner sections that sleep `0.02` seconds.

Vendor command reference observations from `third_party/PyAIPilotExample/controller.py`:

- Uses `pymavlink`.
- Shows `set_attitude_target_send(...)`.
- Documents quaternion order as `(w, x, y, z)`.
- Documents body rates as radians per second.
- Documents thrust as normalized `0.0 .. 1.0`.
- Example attitude path uses `ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE`, a dummy quaternion, and body rates.
- Shows `set_position_target_local_ned_send(...)` with `MAV_FRAME_LOCAL_NED`.
- Vendor controller loop constant is `CONTROL_HZ = 250`, which exceeds the VADR-TS-002 command-rate requirement and must not be copied as-is into competition output.

Risk:

- Competition command adapter must verify exact `SET_ATTITUDE_TARGET` semantics before live command output.
- Current `AutonomyAPI` returns attitude angles, while the vendor example demonstrates body-rate attitude target usage. This must be resolved at the protocol boundary in a later phase.
- Do not retune thrust, yaw, gains, hover behavior, or no-target behavior in Phase 0.

## Current Gazebo Truth Paths

Current production-risk defaults and paths found:

- `AutonomyAPI.__init__` sets `self.perception_world_pose_source = "gazebo_truth_sim_only"`.
- `RuntimeConfig` scaffolding mirrors `perception_world_pose_source = "gazebo_truth_sim_only"`.
- `AutonomyAPI.update_gate_memory_from_frame(...)` accepts `gazebo_pose`.
- `AutonomyAPI.update_gate_memory_from_frame(...)` calls `initialize_perception_yaw_correction(gazebo_pose)`.
- `AutonomyAPI.transform_gate_camera_to_world_for_pose_source(...)` can select Gazebo truth when `perception_world_pose_source == "gazebo_truth_sim_only"`.
- `px4_runner.py` subscribes to Gazebo dynamic pose data and passes `latest_gazebo_pose_snapshot()` into perception calls.
- `flight_logger.py` includes Gazebo pose/source columns and diagnostic far-depth correction fields.
- `autonomy_core/debug/gazebo_diagnostics.py` contains Gazebo truth diagnostic helpers used by `AutonomyAPI` wrappers.

Phase 0 result:

- These paths are inventoried only.
- No guard or behavior change was implemented.
- Later Phase 0.5/Phase 7 work must fail fast for competition mode before command-enabled runs.

## Current Config And Logging Seams

Config scaffold:

- `autonomy_core/core/config.py` contains passive dataclasses for vehicle limits, controller, planner, perception, logging, and runtime config.
- The file states these dataclasses mirror current defaults and are not authoritative runtime sources yet.
- `LoggingConfig` lists future modes: `debug_full`, `debug_light`, `race`, and `off`.

Logging seam:

- `FlightLogger` currently preserves a large schema with perception, planning, command, MAVSDK, and Gazebo diagnostic fields.
- Logging schema and default logging volume were not changed in Phase 0.

Risk:

- Competition race logging must be bounded later, but Phase 0 does not alter logger behavior.

## Existing MAVLink/PX4 Helpers

Current repo helpers:

- Existing runtime control is MAVSDK-based in `autonomy_core/tools/px4_runner.py`.
- No production competition UDP MAVLink runner exists yet.
- No `autonomy_core/runtime/competition_runner.py` exists yet.
- No `autonomy_core/command/competition_command_adapter.py` exists yet.

Vendor references:

- `third_party/PyAIPilotExample/setup.py` creates a `pymavlink` UDP connection.
- `third_party/PyAIPilotExample/mavlink_rx.py` receives simulator MAVLink messages.
- `third_party/PyAIPilotExample/controller.py` sends actuator, attitude target, and position target commands.
- `third_party/PyAIPilotExample/timesync.py` sends `TIMESYNC`.

Vendor rule:

- These files are reference evidence only and must remain immutable.

## Phase 0.25 PyAIPilotExample Protocol Evidence Addendum

Phase 0.25 read-only reference inspection updated this inventory with concrete facts from `third_party/PyAIPilotExample/**`. No runtime files, tests, or vendor files were modified.

Files inspected as immutable source evidence:

- `third_party/PyAIPilotExample/main.py`
- `third_party/PyAIPilotExample/setup.py`
- `third_party/PyAIPilotExample/vision_rx.py`
- `third_party/PyAIPilotExample/mavlink_rx.py`
- `third_party/PyAIPilotExample/controller.py`
- `third_party/PyAIPilotExample/timesync.py`
- `third_party/PyAIPilotExample/requirements.txt`

MAVLink connection facts:

- `main.py` defaults simulator MAVLink access to `SIM_SERVER_UDP_IP = "127.0.0.1"` and `SIM_SERVER_UDP_PORT = 14550`.
- `setup.py` creates the connection with `mavutil.mavlink_connection("udpin:%s:%s" % (server_ip, server_udp_port))`.
- `setup.py` calls `sim_conn.wait_heartbeat()` before constructing receiver/controller components.
- `setup.py` constructs `MAVLinkRX.create_mavlink_rx(...)`, `TimeSync(sim_conn, shared_data)`, `VisionRX(shared_data)`, and `Controller(...)`.
- `setup.py` constructs `TimeSync(...)` directly instead of calling `TimeSync.create_timesync(...)`; as written, the reference setup does not appear to start the timesync thread.

Vision packet facts:

- `vision_rx.py` binds UDP vision to `0.0.0.0:5600`.
- Header format is little-endian `<IHHIIQ`.
- Header fields are `frame_id`, `chunk_id`, `total_chunks`, `jpeg_size`, `payload_size`, and `sim_time_ns`.
- UDP receive size is `65536` bytes.
- Frames are reassembled by `frame_id`.
- Chunks are concatenated in ascending `chunk_id` order.
- JPEG bytes are decoded with `cv2.imdecode(img_array, cv2.IMREAD_COLOR)`.
- The decoded image is OpenCV-compatible BGR data unless a later fixture proves otherwise.
- The reference receiver does not provide the stricter parser validation required by the plan. Future adapter implementation must still validate header length, chunk IDs, total chunks, JPEG size, payload size, duplicate chunks, stale incomplete frames, and decode failures.

Telemetry/message facts:

- `mavlink_rx.py` polls `recv_match(blocking=False)` and sleeps `0.001` seconds when no message is available.
- It handles `HEARTBEAT`, `TIMESYNC`, `ATTITUDE`, `LOCAL_POSITION_NED`, `ODOMETRY`, `HIGHRES_IMU`, `ENCAPSULATED_DATA`, `ACTUATOR_OUTPUT_STATUS`, `COLLISION`, and `DATA_TRANSMISSION_HANDSHAKE`.
- `HEARTBEAT` checks `MAV_MODE_FLAG_SAFETY_ARMED`.
- `TIMESYNC` reads `ts1` and `tc1`.
- `ATTITUDE` reads roll, pitch, yaw, rollspeed, pitchspeed, yawspeed, and `time_boot_ms`.
- `LOCAL_POSITION_NED` reads `x`, `y`, `z`, `vx`, `vy`, `vz`, and `time_boot_ms`.
- `ODOMETRY` reads position, quaternion, velocity, roll/pitch/yaw rates, `time_usec`, and `reset_counter`.
- `HIGHRES_IMU` reads accelerometer, gyro, and `time_usec`.
- `DATA_TRANSMISSION_HANDSHAKE` is used for upcoming track data packets.
- Presence of `LOCAL_POSITION_NED` and `ODOMETRY` handlers is not proof those messages are emitted or usable during live competition runs; observe mode must verify actual availability.

Race/track data facts:

- Encapsulated race status payload starts with data type `1`.
- Race status is decoded with struct `<BQqqIq`.
- Encapsulated track info payload starts with data type `2`.
- Track gate data includes gate ID, NED position, NED orientation quaternion, width, and height.

Race/track data caution:

- Simulator-provided race status may be logged for diagnostics.
- Simulator-provided track/gate geometry must not replace perception or GateMemory behavior in this adapter branch unless rules/specs explicitly allow it and a separate behavior change is approved.

Timesync facts:

- `timesync.py` defines `TIMESYNC_REQUEST_HZ = 10`.
- `timesync.py` sends `timesync_send(now, 0)` with `now = time.time_ns()`.
- Reference setup lifecycle must be verified before copying timesync behavior because `setup.py` does not appear to start the timesync thread.

Command facts:

- `controller.py` includes examples for `SET_ACTUATOR_CONTROL_TARGET`, `SET_ATTITUDE_TARGET`, `SET_POSITION_TARGET_LOCAL_NED`, arm, and simulator reset command `31000`.
- `SET_ATTITUDE_TARGET` example uses type mask `ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE`.
- `SET_ATTITUDE_TARGET` example sends dummy quaternion `[1, 0, 0, 0]`.
- `SET_ATTITUDE_TARGET` example sends body roll/pitch/yaw rates in `rad/s`.
- `SET_ATTITUDE_TARGET` example sends normalized thrust `0.0 .. 1.0`.
- `SET_POSITION_TARGET_LOCAL_NED` example uses `MAV_FRAME_LOCAL_NED`.
- `SET_POSITION_TARGET_LOCAL_NED` example masks ignored position, acceleration, yaw, and yaw-rate fields while using velocity fields.
- Default `Controller.update()` calls motor-control output, not the attitude-target or position-target example path.
- `CONTROL_HZ = 250`, which exceeds the VADR-TS-002 command-rate limit and must not be copied into competition command publishing.

Dependency facts:

- `requirements.txt` lists `pymavlink`, `opencv-python`, `numpy`, `matplotlib`, and `keyboard`.
- Phase 0.25 did not install or update dependencies.

Carry-forward decisions:

- Future observe mode must record exact MAVLink message names, IDs, field samples, rates, timestamp sources, system/component IDs, and whether `LOCAL_POSITION_NED` or `ODOMETRY` provides usable position and velocity.
- Future command adapter work must document whether `AutonomyAPI.attitude_control()` returns attitude angles, body rates, yaw angle, yaw rate, normalized efforts, or another convention before mapping to `SET_ATTITUDE_TARGET`.
- If command tuple semantics are unclear, Phase 5 must stop and add a command-semantics note before coding the MAVLink mapping.
- Future command code must not copy vendor `CONTROL_HZ = 250`.
- Future implementation must not modify `third_party/PyAIPilotExample/**`.

## Phase 0 Risks To Carry Forward

- Gazebo truth defaults still exist in `AutonomyAPI` and config scaffolding.
- Existing image path is ROS2 camera frame based, not UDP JPEG based.
- Existing state path assumes local position/velocity/yaw have already been converted into internal z-up convention.
- Live simulator telemetry availability remains unresolved; local position/odometry must be verified before command enablement.
- Current command output is MAVSDK attitude, while competition requires MAVLink UDP command messages.
- Vendor command example uses body-rate attitude target semantics, while existing `AutonomyAPI` returns roll/pitch/yaw/thrust.
- Vendor default controller path sends motor/actuator control, not the attitude-target or position-target examples; do not copy it as the adapter command path.
- Vendor `CONTROL_HZ = 250` exceeds the VADR-TS-002 command-rate limit and must not be copied.
- Vendor setup constructs `TimeSync(...)` directly and does not appear to start the timesync thread; lifecycle needs verification before copying timesync behavior.
- Vendor track/race data may be useful for diagnostics but must not replace perception/GateMemory behavior without separate approval.
- `ultralytics`, ROS2, MAVSDK, OpenCV, and other runtime dependencies are not declared in `pyproject.toml`; import validation may be environment-dependent.
- Existing debug frames and flight logs are present as untracked artifacts and were not modified.
- `third_party/PyAIPilotExample/` is untracked vendor/reference material and must not be modified by adapter implementation.

## Phase 0 Completion Criteria

Phase 0 is complete when this inventory is present and the worktree only contains docs/minimal test-layout additions from this phase plus pre-existing unrelated artifacts.

Confirmed:

- Active facade methods are listed.
- Current runner import path is listed.
- Current Gazebo-truth paths are listed.
- Existing MAVLink/PX4 command references are listed.
- Test layout status is listed.
- Protected vendor rule is listed.
- No runtime code was changed.
