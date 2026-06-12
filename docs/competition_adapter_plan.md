# Competition Adapter Plan v2

This plan replaces the first adapter plan as the implementation guide for the next competition branch. It is based on VADR-TS-002 Issue 00.02, dated May 8, 2026, plus the current readiness checkpoint. The goal is adapter-first and behavior-preserving integration: make the existing `AutonomyAPI` stack speak the competition interface correctly before changing planner, perception, control, or estimator behavior.

## Hard Constraint - Protected Vendor Reference

`third_party/PyAIPilotExample/**` is read-only vendor/reference code.

Allowed actions:

- Read, inspect, import, execute, and test against files under `third_party/PyAIPilotExample/**`.
- Copy files from `third_party/PyAIPilotExample/**` into `tmp_path`, `/tmp`, or another temporary test directory when a test needs mutable fixtures.

Forbidden actions:

- Do not edit, format, move, rename, delete, generate into, or commit any file under `third_party/PyAIPilotExample/**`.
- Do not apply automated formatters, linters with write/fix mode, codemods, patch tools, or cleanup scripts to `third_party/PyAIPilotExample/**`.
- Do not vendor-modify `PyAIPilotExample` to make tests pass. If the example disagrees with this plan or the implementation, update adapter code, tests, or this plan outside the protected vendor tree.

Implementation rule:

- Treat every file under `third_party/PyAIPilotExample/**` as immutable source evidence. Any writable test artifact derived from it must live outside that tree.

## Priority Definitions

- P0: must be done before meaningful simulator runs or command-enabled runs.
- P1: required before a submitted/timed competition run.
- P2: deferred unless discovered to be a blocker.

## Source Of Truth

Use VADR-TS-002 Issue 00.02 as the active spec. Ignore older conflicting VADR-TS-001 assumptions unless they are explicitly repeated by VADR-TS-002 or the provided simulator example code.

Required competition constants:

- MAVLink transport: UDP MAVLink-compatible transport.
- World/state convention: MAVLink local NED at the protocol boundary.
- Internal runtime convention: existing z-up `VehicleState` inside `AutonomyAPI` and planner/control code.
- Vision transport: UDP port `5600`.
- Vision packet format: little-endian `24` byte metadata header followed by JPEG payload bytes.
- Vision header struct: `<IHHIIQ` as `frame_id:uint32`, `chunk_id:uint16`, `total_chunks:uint16`, `jpeg_size:uint32`, `payload_size:uint32`, `sim_time_ns:uint64`.
- Vision frame handling: reassemble JPEG chunks by `frame_id` before decoding.
- Camera resolution: `640x360`.
- Camera intrinsics: `fx=320`, `fy=320`, `cx=320`, `cy=180`.
- Camera matrix: `[[320, 0, 320], [0, 320, 180], [0, 0, 1]]`.
- Distortion coefficients: all zeros.
- Camera/body translation: same origin.
- Camera/body rotation: camera tilted `20 deg` upward relative to body.
- Simulator physics rate: `120 Hz`.
- Vision rate: `30 Hz`.
- Heartbeat minimum: `2 Hz`.
- Command output rate: strictly below `100 Hz`.
- Gate model: `2700 mm` outer square, `1500 mm` inner square, `260 mm` depth.
- Drone chassis: `280 mm x 280 mm x 160 mm`.
- Race limit: maximum submitted run duration is `8 min`.

## Hard Non-Goals For This Branch

Do not implement these unless a phase explicitly discovers they are a blocker:

- MPC or tracker replacement.
- EKF/VIO, except if the live simulator does not provide usable local position/odometry.
- YOLO retraining, domain randomization, confidence retuning, or model replacement.
- PnP scoring changes.
- Pixel orientation changes, image resizing, or undistortion changes without a failing fixture proving the mismatch.
- Deep planner changes.
- `advance_gate_if_needed(...)` or gate progression refactors.
- Thrust mapping, yaw behavior, controller gains, hover behavior, or no-target behavior changes.
- Logging schema changes during bring-up.
- Gazebo truth usage in competition mode.
- Any modification to `third_party/PyAIPilotExample/**`.

## Existing Facade Seams To Preserve

Primary integration points:

- `AutonomyAPI.update_gate_memory_from_frame(frame, camera_matrix, dist_coeffs, image_stamp_sec=0, image_stamp_nanosec=0, image_received_wall_time=np.nan, image_pose_snapshot=None, gazebo_pose=None)`
- `AutonomyAPI.path_plan(...)`
- `AutonomyAPI.advance_gate_if_needed(...)`
- `AutonomyAPI.attitude_control()` returning `(roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd)`
- `autonomy_core/core/types.py::VehicleState`
- `autonomy_core/core/types.py::CameraFrame`
- `autonomy_core/core/types.py::CameraModel`
- `autonomy_core/core/types.py::ControlCommand`

Competition work belongs at the protocol boundary. Keep the facade behavior stable until adapter tests and dry runs prove the boundary conversions are correct.

## Phase 0 - Branch Rules And Inventory

Priority: P0.

Purpose:

- Prevent implementation drift before files are added.
- Record the exact current seams that the adapters must call.

Implementation tasks:

- Create this branch as adapter-first and behavior-preserving.
- Confirm the active `AutonomyAPI` import path used by existing runners.
- Confirm current test layout. If no test layout exists, create one without moving runtime code.
- Search for current Gazebo-truth defaults and call sites before implementing the guard.
- Search for any existing MAVLink/PX4 command helpers that can be reused without changing behavior.
- Confirm `third_party/PyAIPilotExample/**` is treated as immutable vendor/reference code before running edits, formatters, test fixture generation, or cleanup scripts.

Expected output:

- No runtime behavior changes.
- A short implementation note or commit message listing the active facade methods and the files that contain Gazebo-truth paths.

Acceptance criteria:

- Existing imports still work.
- Existing runner files are not modified except for optional import-safe helper extraction if needed.
- No planner, controller, YOLO, PnP, race progression, or logging behavior changes.
- No files under `third_party/PyAIPilotExample/**` are modified, formatted, moved, renamed, deleted, generated, or committed.

## Phase 0.25 - PyAIPilotExample Protocol Evidence Addendum

Priority: P0.

Purpose:

- Capture concrete protocol facts from the read-only `third_party/PyAIPilotExample/**` reference files before implementation.
- Convert the example-code findings into explicit adapter risks and decision gates.
- Keep this phase docs/tooling-only; no runtime behavior changes.

Primary files to update:

- `docs/competition_adapter_plan.md`
- `docs/competition_adapter_phase0_inventory.md`
- Optional: `scripts/check_protected_paths.sh` if still missing.

Implementation tasks:

- Read, but do not modify, these reference files:
  - `third_party/PyAIPilotExample/vision_rx.py`
  - `third_party/PyAIPilotExample/mavlink_rx.py`
  - `third_party/PyAIPilotExample/controller.py`
  - `third_party/PyAIPilotExample/timesync.py`
  - `third_party/PyAIPilotExample/setup.py`
  - `third_party/PyAIPilotExample/main.py`

- Record the following reference-code facts in the Phase 0 inventory:
  - `main.py` defaults MAVLink simulator access to `SIM_SERVER_UDP_IP = "127.0.0.1"` and `SIM_SERVER_UDP_PORT = 14550`.
  - `setup.py` creates the MAVLink connection with `mavutil.mavlink_connection("udpin:%s:%s" % (server_ip, server_udp_port))`.
  - `setup.py` calls `sim_conn.wait_heartbeat()` before creating the MAVLink RX, timesync, vision RX, and controller components.
  - `setup.py` constructs `TimeSync(sim_conn, shared_data)` directly; it does not call `TimeSync.create_timesync(...)`, so the reference code as written does not appear to start the timesync thread.
  - `vision_rx.py` binds the UDP vision socket to `0.0.0.0:5600`.
  - `vision_rx.py` uses header format little-endian `<IHHIIQ`, whose fields are `frame_id`, `chunk_id`, `total_chunks`, `jpeg_size`, `payload_size`, and `sim_time_ns`.
  - `vision_rx.py` uses `struct.calcsize("<IHHIIQ")` for header size, receives up to `65536` bytes per UDP datagram, and reassembles chunks by `frame_id`.
  - `vision_rx.py` concatenates chunks in `chunk_id` order, decodes JPEG bytes with `cv2.imdecode(..., cv2.IMREAD_COLOR)`, and passes an OpenCV BGR image to `process_frame(...)`.
  - The reference vision receiver does not validate all fields required by this plan; adapter implementation must still validate header length, chunk IDs, total chunks, JPEG size, payload size, duplicate chunks, stale incomplete frames, and decode failures.
  - `mavlink_rx.py` polls `recv_match(blocking=False)` and sleeps `0.001` seconds when no message is available.
  - Candidate telemetry messages in `mavlink_rx.py` include `HEARTBEAT`, `TIMESYNC`, `ATTITUDE`, `LOCAL_POSITION_NED`, `ODOMETRY`, `HIGHRES_IMU`, `ENCAPSULATED_DATA`, `ACTUATOR_OUTPUT_STATUS`, `COLLISION`, and `DATA_TRANSMISSION_HANDSHAKE`.
  - `mavlink_rx.py` reads `ATTITUDE` roll, pitch, yaw, rollspeed, pitchspeed, yawspeed, and `time_boot_ms`.
  - `mavlink_rx.py` reads `LOCAL_POSITION_NED` fields `x`, `y`, `z`, `vx`, `vy`, `vz`, and `time_boot_ms`.
  - `mavlink_rx.py` reads `ODOMETRY` position, quaternion, velocity, body rates, `time_usec`, and `reset_counter`.
  - `ENCAPSULATED_DATA` may contain race status and track information; record it during observe mode but do not feed track/gate positions into autonomy behavior unless separately approved.
  - `mavlink_rx.py` decodes race status with struct `<BQqqIq`.
  - `mavlink_rx.py` decodes track gate data as NED position, NED orientation quaternion, width, and height.
  - `timesync.py` defines `TIMESYNC_REQUEST_HZ = 10` and sends `timesync_send(now, 0)` with `now = time.time_ns()`.
  - `controller.py` command examples include `SET_ACTUATOR_CONTROL_TARGET`, `SET_ATTITUDE_TARGET`, `SET_POSITION_TARGET_LOCAL_NED`, arm, and simulator reset command `31000`.
  - `controller.py` `SET_ATTITUDE_TARGET` example uses body rates in `rad/s`, normalized thrust `0.0 .. 1.0`, type mask `ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE`, and dummy quaternion `[1, 0, 0, 0]`.
  - `controller.py` `SET_POSITION_TARGET_LOCAL_NED` example uses `MAV_FRAME_LOCAL_NED`, velocity fields, and masks ignored position, acceleration, yaw, and yaw-rate fields.
  - `controller.py` default `Controller.update()` calls motor-control output, not the attitude-target or position-target example paths.
  - `controller.py` has `CONTROL_HZ = 250`, which exceeds the VADR-TS-002 command-rate limit and must not be copied into competition command publishing.
  - `requirements.txt` lists reference dependencies including `pymavlink`, `opencv-python`, `numpy`, `matplotlib`, and `keyboard`; Phase 0.25 does not install dependencies.

- Add an explicit Phase 5 precondition:
  - Before implementing `SET_ATTITUDE_TARGET` mapping, document whether existing `AutonomyAPI.attitude_control()` returns attitude angles, body rates, yaw angle, yaw rate, normalized efforts, or another convention.
  - If tuple semantics are unclear, stop Phase 5 and add a command-semantics note before coding the MAVLink mapping.

- Add an explicit Phase 4 observe-mode requirement:
  - Observe mode must record exact MAVLink message names, IDs, fields, field samples, rates, timestamp sources, system/component IDs, and whether `LOCAL_POSITION_NED` or `ODOMETRY` provides usable position and velocity.
  - Do not assume local position/odometry just because `mavlink_rx.py` has handlers for those messages.

- Add an explicit race/track-data caution:
  - Simulator-provided race status may be logged for diagnostics.
  - Simulator-provided track/gate geometry must not be used as a replacement for perception/gate-memory behavior in this adapter branch unless rules/specs explicitly allow it and a separate behavior change is approved.

Acceptance criteria:

- `docs/competition_adapter_phase0_inventory.md` has a “PyAIPilotExample protocol evidence” section.
- `docs/competition_adapter_plan.md` contains this Phase 0.25 section or an equivalent addendum.
- Phase 4 and Phase 5 decision gates include the telemetry and command-semantics clarifications above.
- No runtime files are modified.
- No files under `third_party/PyAIPilotExample/**` are modified, formatted, moved, renamed, deleted, generated into, staged, or committed.
- If `scripts/check_protected_paths.sh` is added, it is non-runtime tooling only and does not install hooks automatically.

Do not change:

- Adapter implementation.
- Existing runners.
- Planner, controller, YOLO, PnP, perception, race progression, logging, or Gazebo diagnostics.
- Files under `third_party/PyAIPilotExample/**`.

## Phase 0.5 - Minimal Gazebo Guard Skeleton

Priority: P0.

Purpose:

- Add the smallest competition-mode guard before any meaningful simulator integration.
- Prevent Gazebo truth defaults from leaking into adapter bring-up.

Primary files to add or update:

- `autonomy_core/runtime/competition_guard.py` or equivalent.
- `autonomy_core/runtime/competition_runner.py` once the runner exists.
- Tests such as `tests/test_competition_gazebo_guard.py`.

Implementation tasks:

- Define an explicit competition mode/profile flag that adapters and the runner can check.
- Reject `perception_world_pose_source = "gazebo_truth_sim_only"` in competition mode.
- Force `gazebo_pose=None` for all competition calls into `AutonomyAPI.update_gate_memory_from_frame(...)`.
- Reject image metadata or runner inputs that contain Gazebo model pose, Gazebo camera pose, Gazebo TF, or Gazebo-only pose snapshots.
- Make command-enabled modes refuse to start if any Gazebo truth path is active.
- Keep the skeleton small; the full production guard remains Phase 7.

Acceptance criteria:

- Competition-mode import/static tests prove `gazebo_truth_sim_only` is rejected.
- Competition-mode tests prove non-`None` `gazebo_pose` is rejected before perception calls.
- Command-enabled runner modes cannot start when the guard detects active Gazebo truth paths.
- Existing sim/Gazebo diagnostics remain available outside competition mode.

Do not change:

- Existing Gazebo diagnostic behavior for simulation.
- Logger columns that currently expose Gazebo diagnostic fields.
- Perception transform behavior.

## Phase 1 - Competition Config Constants

Priority: P0.

Purpose:

- Freeze VADR-TS-002 Issue 00.02 constants in one importable place.
- Avoid older assumptions leaking into adapter code.

Primary files to add or update:

- `autonomy_core/core/competition_config.py` or an equivalent config module.
- `autonomy_core/core/config.py` only if adding a passive `RuntimeCompetitionConfig` is cleaner.
- Tests under the project test directory, for example `tests/test_competition_config.py`.

Implementation tasks:

- Add a frozen `RuntimeCompetitionConfig` or equivalent immutable constants container.
- Include the exact constants listed in the `Source Of Truth` section.
- Provide helper methods or properties for `camera_matrix`, `dist_coeffs`, command period, heartbeat period, and camera tilt radians.
- Keep this config passive. Do not silently wire it into existing runtime defaults yet.
- Make older sim/Gazebo defaults explicit by name so they cannot be mistaken for competition defaults.

Acceptance criteria:

- Importing the config has no side effects.
- A test asserts the official camera matrix equals `[[320, 0, 320], [0, 320, 180], [0, 0, 1]]`.
- A test asserts distortion is all zeros.
- A test asserts command rate limit is below `100 Hz`, vision rate is `30 Hz`, heartbeat minimum is at least `2 Hz`, and physics rate is `120 Hz`.
- A test asserts camera tilt is represented as `20 deg` upward and converted to radians only by helpers.

Do not change:

- Existing `RuntimeConfig` behavior.
- Existing perception defaults.
- Existing controller or planner defaults.

## Phase 2 - UDP Vision Receiver And Image Adapter

Priority: P0.

Purpose:

- Receive the competition JPEG stream on UDP port `5600`.
- Reassemble chunked frames and decode them into the exact frame object accepted by `AutonomyAPI.update_gate_memory_from_frame(...)`.

Primary files to add:

- `autonomy_core/perception/competition_image_adapter.py`
- Optional transport-only helper, for example `autonomy_core/runtime/competition_vision_udp.py`, if keeping sockets out of perception code is cleaner.
- Tests such as `tests/test_competition_image_adapter.py`.

Packet/parser tasks:

- Implement a parser for the VADR-TS-002 little-endian `24` byte metadata header.
- Use the expected struct `<IHHIIQ`: `frame_id:uint32`, `chunk_id:uint16`, `total_chunks:uint16`, `jpeg_size:uint32`, `payload_size:uint32`, `sim_time_ns:uint64`.
- Confirm that struct against VADR-TS-002 and the provided `PyAIPilotExample/vision_rx.py` before coding; cross-check `PyAIPilotExample/mavlink_rx.py` only for MAVLink and timing context. If either VADR-TS-002 or `vision_rx.py` disagrees with this plan, stop and update the plan/tests before implementation.
- Treat `third_party/PyAIPilotExample/vision_rx.py` and `third_party/PyAIPilotExample/mavlink_rx.py` as read-only reference files; copy them to a temporary directory first if a test needs mutable example data.
- Extract and validate `frame_id`, `chunk_id`, `total_chunks`, `jpeg_size`, `payload_size`, and `sim_time_ns`.
- Reject packets with malformed header length, invalid `chunk_id`, invalid `total_chunks`, negative or impossible sizes, `payload_size` mismatch, or payload that would exceed `jpeg_size`.
- Reassemble chunks by `frame_id`.
- Drop incomplete stale frames using bounded memory.
- Count dropped chunks, duplicate chunks, incomplete frames, decode failures, and completed frames.

Decode/adapter tasks:

- Decode the completed JPEG into the same array layout currently consumed by perception.
- Preserve pixel orientation.
- Do not resize, undistort, rotate, mirror, or normalize pixels in this branch unless a deterministic fixture proves VADR-TS-002 requires it.
- Produce a `CameraFrame` or equivalent tuple containing `frame`, official `camera_matrix`, zero `dist_coeffs`, `image_stamp_sec`, `image_stamp_nanosec`, `image_received_wall_time`, `image_pose_snapshot=None`, and `gazebo_pose=None`.
- Pass decoded frames into `AutonomyAPI.update_gate_memory_from_frame(...)` from the runner, not from a background socket callback that hides errors.

Acceptance criteria:

- One fixture encodes a small image as JPEG, splits it into multiple VADR-TS-002 packets, feeds packets out of order if the spec permits it, and asserts exactly one decoded frame is emitted.
- One fixture packs and unpacks the exact `<IHHIIQ` header and asserts the decoded fields and `24` byte size.
- The decoded frame has expected shape `(360, 640, channels)` or a documented OpenCV-compatible shape accepted by the current perception code.
- Timestamps are preserved or wall-time fallback is explicit.
- Corrupt or missing chunks do not emit a frame.
- Memory for incomplete frame assemblies is bounded.

Do not change:

- YOLO thresholds.
- PnP scoring.
- Frame orientation.
- Undistortion policy.
- `AutonomyAPI.update_gate_memory_from_frame(...)` internals unless needed for an explicit Gazebo guard in a later phase.

## Phase 3 - Production Camera Model And Frame Convention Boundary

Priority: P0.

Purpose:

- Make the official camera intrinsics and `20 deg` upward camera tilt available to perception without changing perception behavior blindly.
- Put all NED/body/camera convention conversions at the adapter boundary or behind explicit tested helpers.

Primary files to add or update:

- `autonomy_core/perception/competition_image_adapter.py`
- `autonomy_core/core/competition_config.py`
- Optional math helper module only if needed, for example `autonomy_core/core/frame_conventions.py`.
- Tests such as `tests/test_competition_camera_model.py`.

Implementation tasks:

- Feed `camera_matrix = [[320, 0, 320], [0, 320, 180], [0, 0, 1]]` into the existing perception path.
- Feed zero distortion coefficients into the existing perception path.
- Represent camera/body translation as zero.
- Represent the `20 deg` upward camera tilt explicitly as a transform with a documented sign convention.
- Document the current image-processing camera convention used by PnP/OpenCV in code comments or tests.
- Document the MAVLink/body/NED convention used by telemetry and command adapters.
- Add deterministic transform tests for identity body pose plus `20 deg` camera tilt.
- Add a projection sanity fixture: with identity body pose and a point straight ahead in the documented body/NED convention, apply the `20 deg` upward camera tilt and verify the projected pixel moves in the expected image direction.

Acceptance criteria:

- Official intrinsics are asserted exactly.
- Distortion is asserted all-zero.
- Camera tilt is not hidden in arbitrary constants or legacy offsets.
- Tests make it clear which direction is NED, which direction is internal z-up, and which camera axis convention is used by OpenCV/perception.
- The projection sanity fixture fails on the common camera-tilt sign inversion.

High-risk warning:

- Current code already has camera-frame and transform assumptions. Do not change transforms in the perception core as part of this phase. If a mismatch is found, add a failing fixture first, then fix only the boundary conversion needed to satisfy that fixture.

## Phase 4 - MAVLink Telemetry And State Adapter

Priority: P0.

Purpose:

- Convert competition MAVLink/NED telemetry into the internal z-up `VehicleState` expected by the current runtime.
- Resolve immediately whether the live simulator emits usable local position/odometry.
- Treat telemetry discovery as the first live-simulator milestone before enabling vision-driven or command-enabled runs.

Primary files to add:

- `autonomy_core/core/competition_state_adapter.py`
- Optional transport helper, for example `autonomy_core/runtime/competition_mavlink.py`.
- Tests such as `tests/test_competition_state_adapter.py`.
- Phase 4B evidence review note: `docs/competition_adapter_phase4b_telemetry_evidence.md`.

Verification tasks before finalizing adapter behavior:

- Inspect the provided `PyAIPilotExample/mavlink_rx.py`.
- Treat `third_party/PyAIPilotExample/mavlink_rx.py` as read-only reference code; do not patch it to expose fields, alter timing, or simplify tests.
- Run `observe` mode against the live simulator before building out command-enabled behavior and record the exact MAVLink message names, IDs, fields, rates, and timestamps emitted by the simulator.
- This inventory can be performed by a minimal standalone observe script or a thin early runner skeleton; the full `competition_runner.py` does not need to exist yet.
- Observe mode must record message IDs, system/component IDs, field samples, timestamp sources, observed rates, and freshness for each relevant MAVLink message.
- Observe mode must specifically record whether `LOCAL_POSITION_NED` or `ODOMETRY` is emitted and whether either message provides usable position and velocity for the current planner/controller assumptions.
- Cross-check VADR-TS-002 telemetry expectations, including `ATTITUDE`, `HIGHRES_IMU`, `TIMESYNC`, `HEARTBEAT`, command messages, orientation, linear velocities, and system status flags.
- Specifically verify whether usable local position or odometry exists.
- Record whether velocity is world-frame NED or body-frame.
- Record whether attitude is quaternion, Euler, yaw-only, or another representation.
- Record timestamp fields and freshness semantics.
- Record estimator/system flags that indicate validity.

Telemetry discovery gate:

- Do not assume local position/odometry exists just because the current runtime needs it.
- Do not assume local position/odometry exists just because `third_party/PyAIPilotExample/mavlink_rx.py` has handlers for `LOCAL_POSITION_NED` and `ODOMETRY`.
- If no live receive-only observe output is available, Phase 4B remains blocked; do not finalize adapter behavior or proceed toward command-enabled phases.
- If the live simulator emits usable local position/odometry, implement the thin adapter mapping from that message into `VehicleState`.
- If the live simulator does not emit usable local position/odometry, stop before command enablement and create a separate P0 state-estimation/VIO/EKF plan.
- Do not hide missing position by fabricating position from stale data, gate observations, or Gazebo truth.

State conversion tasks:

- Convert NED position to internal z-up position using a tested mapping.
- Convert NED velocity to internal z-up velocity using the same convention.
- Convert attitude/yaw into internal yaw convention expected by `VehicleState`.
- Preserve timestamps and freshness metadata at runner level if `VehicleState` remains minimal.
- Validate finite numeric values before constructing `VehicleState`.
- Treat missing/stale telemetry as a runner-level failsafe condition.

Telemetry ambiguity decision:

- If local position/odometry is available and fresh, continue with the thin state adapter.
- If only attitude/orientation, linear velocities, and system flags are available, mark position unavailable and stop before enabling commands.
- If position/odometry is not emitted, VIO/EKF or another state-estimation bridge becomes a P0 unblocker for command-enabled runs. It remains outside this adapter branch until that blocker is explicitly scoped.

Acceptance criteria:

- Observe-mode output records the exact MAVLink telemetry available from the simulator.
- A NED fixture converts into the exact expected internal z-up `VehicleState.pos`, `VehicleState.vel`, and `VehicleState.yaw`.
- Missing position/odometry is represented explicitly and causes command output to remain disabled.
- Stale telemetry is detected by timestamp/freshness checks.
- No existing `GetTelemetry` internals are required by the competition adapter.

Do not change:

- `autonomy_core/core/state_adapter.py::vehicle_state_from_telemetry(...)` behavior.
- Internal planner/controller state convention.
- Planner behavior to compensate for missing telemetry.

Note for user, run this locally when the simulator is running:
- PYTHONDONTWRITEBYTECODE=1 python3 -B -m autonomy_core.tools.competition_mavlink_observe --endpoint udpin:127.0.0.1:14550 --duration-s 30

## Phase 5 - Command Adapter For MAVLink SET_ATTITUDE_TARGET

Priority: P0.

Purpose:

- Wrap the existing `(roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd)` output into MAVLink command messages without retuning control behavior.

Primary files to add:

- `autonomy_core/command/competition_command_adapter.py`
- Create `autonomy_core/command/__init__.py` if the package does not exist.
- Tests such as `tests/test_competition_command_adapter.py`.
- Phase 5A command semantics note: `docs/competition_adapter_phase5a_command_semantics.md`.

Implementation tasks:

- Accept either the raw tuple `(roll, pitch, yaw, thrust)` or `ControlCommand`.
- Before implementing the MAVLink mapping, document whether existing `AutonomyAPI.attitude_control()` outputs roll/pitch/yaw attitude angles, body rates, yaw angle, yaw rate, normalized efforts, or another convention.
- If tuple semantics are unclear, stop Phase 5 and add a command-semantics note before coding the MAVLink adapter.
- Phase 5A documents the current tuple as roll/pitch/yaw attitude angles in radians plus normalized thrust; it is not a body-rate command.
- Phase 5A may add only a dry-run `SET_ATTITUDE_TARGET` field builder with `send_ready = false`; live command publication remains blocked while Phase 4B lacks real competition telemetry evidence.
- Validate finite values and expected units.
- After comparing with the simulator example, record the confirmed units and ranges for roll, pitch, yaw, and thrust, including radians versus degrees and normalized thrust versus any simulator-specific thrust field.
- Convert roll/pitch/yaw into the MAVLink `SET_ATTITUDE_TARGET` representation required by the simulator.
- Verify whether the simulator expects attitude quaternion fields, body-rate fields, ignored fields via type mask, or a combination.
- Verify roll/pitch/yaw units and frame semantics before constructing the quaternion or attitude target payload.
- Compare the existing tuple semantics against the read-only `controller.py` evidence that its `SET_ATTITUDE_TARGET` example ignores attitude quaternion and uses body rates plus normalized thrust.
- Preserve the existing thrust value except for protocol-level field formatting or clamping already required by MAVLink message validity.
- Attach timestamps, sequence counters, target system/component, and type masks if required by the MAVLink library or simulator example.
- Enforce command publication rate strictly below `100 Hz`.
- Provide an explicit dry-run path that serializes/logs the would-be command without sending it.
- Record rejected commands and rejection reasons.

Initial command path:

- Use `SET_ATTITUDE_TARGET` first because `AutonomyAPI.attitude_control()` already returns roll, pitch, yaw, and thrust.
- Do not use `SET_POSITION_TARGET_LOCAL_NED` as the first path unless the simulator rejects attitude-target control or the spec example requires position-target control.

Acceptance criteria:

- A fixture command tuple maps to deterministic MAVLink fields.
- A serialization fixture compares `SET_ATTITUDE_TARGET` fields, quaternion/body-rate/type-mask semantics, target IDs, and thrust against the provided read-only `PyAIPilotExample` behavior or a known-good packet capture.
- Command rate limiting is tested with a fake clock.
- NaN, infinity, missing fields, and stale state reject command publication.
- Dry-run mode produces the same message object or serialized fields without sending over UDP.

Do not change:

- Controller gains.
- Thrust hover/min/max tuning.
- Yaw policy.
- No-target behavior.
- Tracker implementation.

## Phase 6 - MAVLink Transport And Dry-Run Competition Runner

Priority: P0.

Purpose:

- Add a dedicated competition runner that owns transport, heartbeat, telemetry polling, image ingestion, planning/control scheduling, and command publication mode.

Primary files to add:

- `autonomy_core/runtime/competition_runner.py`
- Create `autonomy_core/runtime/__init__.py` if the package does not exist.
- Optional transport wrappers in `autonomy_core/runtime/competition_mavlink.py` and `autonomy_core/runtime/competition_vision_udp.py`.
- Tests such as `tests/test_competition_runner_dry_run.py` for import-safe or fake-transport behavior.
- Phase 6A fake-transport runner note: `docs/competition_adapter_phase6a_runner_skeleton.md`.

Runner responsibilities:

- Run the minimal competition guard before starting command-capable modes.
- Establish UDP MAVLink communication.
- Receive simulator heartbeats.
- Send client heartbeat/keepalive at or above `2 Hz` minimum.
- Process telemetry through `competition_state_adapter.py`.
- Process vision through `competition_image_adapter.py`.
- Call `AutonomyAPI.update_gate_memory_from_frame(...)` for completed frames.
- Call `AutonomyAPI.path_plan(...)` using existing scheduling behavior where possible.
- Call `AutonomyAPI.advance_gate_if_needed(...)` only as current runtime already does; do not refactor progression.
- Call `AutonomyAPI.attitude_control()` for the command tuple.
- Publish commands through `competition_command_adapter.py` only when command mode is enabled and safety/freshness gates pass.
- Keep command publication below `100 Hz`.

Runner modes:

- `observe`: heartbeat and telemetry only; no vision ingestion and no commands. This is the first live-simulator milestone and must record the exact MAVLink telemetry inventory.
- `vision_dry_run`: heartbeat, telemetry, and image ingestion; no commands.
- `command_dry_run`: full pipeline but command adapter does not send UDP commands.
- `command_live`: full pipeline with command sending enabled, only after explicit operator selection outside submitted timed runs.
- `race`: autonomous timed-run mode with no human interaction after start and bounded logging.

Phase 6A status:

- Add only an import-safe runner skeleton with fake/injected transports.
- Do not implement live MAVLink transport or live UDP vision transport.
- Do not enable `command_live` or `race`; these modes must fail closed in Phase 6A.
- `command_dry_run` may build dry-run command candidates through the Phase 5A adapter, but command publication remains blocked because Phase 4B lacks real competition telemetry evidence.

Minimum runner logs:

- Heartbeat received/sent status and age.
- MAVLink connection status.
- Exact MAVLink message names, IDs, fields used, observed rates, and timestamp sources during observe mode.
- Telemetry message type, freshness, and validity.
- Vision frame rate.
- Dropped/duplicate/corrupt image chunks.
- JPEG decode failures.
- `AutonomyAPI` perception update call count.
- Command candidate rate.
- Command publish rate.
- Command rejection/failsafe reasons.
- Gate-order progression summary sufficient to debug invalid runs.

Acceptance criteria:

- Runner imports without opening sockets unless explicitly started.
- Fake transports can exercise heartbeat, telemetry, image, and dry-run command flow.
- Commands are disabled by default.
- Live command output requires an explicit mode flag/config value.
- Missing heartbeat, stale telemetry, stale image, invalid state, or Gazebo guard failure prevents command publication.

Do not change:

- `px4_runner.py` compatibility.
- `flight_logger.py` schema.
- Existing sim/Gazebo runner behavior.

## Phase 7 - Hard Gazebo-Truth Production Guard

Priority: P0.

Purpose:

- Fail fast if competition runtime tries to use Gazebo truth.
- Expand the Phase 0.5 guard skeleton into the full production guard.
- Remove the highest-risk compliance issue before command-enabled simulator runs.

Primary files to add or update:

- `autonomy_core/runtime/competition_guard.py` or equivalent.
- `autonomy_core/runtime/competition_runner.py`
- Optional guarded hook in `AutonomyAPI` if the runner-level guard cannot cover all paths.
- Tests such as `tests/test_competition_gazebo_guard.py`.
- Phase 7 guard note: `docs/competition_adapter_phase7_gazebo_guard.md`.

Phase 7 status:

- Implemented for the current competition runner boundary.
- No live transport, command publication, telemetry readiness, or command readiness is claimed.
- Existing `px4_runner.py`, `AutonomyAPI`, Gazebo diagnostics, and logger schema remain unchanged.

Guard requirements:

- Preserve and harden all Phase 0.5 guard checks.
- Competition runtime must pass `gazebo_pose=None` into `AutonomyAPI.update_gate_memory_from_frame(...)`.
- Competition runtime must reject `perception_world_pose_source = "gazebo_truth_sim_only"`.
- Competition runtime must not subscribe to or consume Gazebo model pose, Gazebo camera pose, Gazebo TF, or Gazebo-only depth correction fields.
- Competition runtime must fail fast if any image metadata contains a Gazebo pose snapshot.
- Competition runtime must fail fast if diagnostic far-depth correction depends on Gazebo-only assumptions.
- Guard must run before sockets start sending live commands.

Acceptance criteria:

- A competition-mode test proves `gazebo_pose` is rejected.
- A competition-mode test proves `perception_world_pose_source = "gazebo_truth_sim_only"` is rejected.
- A competition-mode test proves the runner cannot start in command-enabled mode if Gazebo truth paths are active.
- Sim/Gazebo diagnostics remain available outside competition mode.

Do not change:

- Debug/Gazebo diagnostic code used by simulation unless required to add explicit profile checks.
- Logger columns that currently record Gazebo diagnostic fields.

## Phase 8 - Deterministic Adapter Test Suite

Priority: P1, but should be written alongside P0 code whenever practical.

Purpose:

- Prove all competition boundary conversions before live command output.

Required fixture tests:

- Config fixture: VADR-TS-002 constants match official values.
- Vision header fixture: the exact little-endian `<IHHIIQ` header is `24` bytes and unpacks into `frame_id`, `chunk_id`, `total_chunks`, `jpeg_size`, `payload_size`, and `sim_time_ns`.
- Image fixture: encoded/chunked JPEG frame decodes to expected array shape, dtype, timestamp fields, and official camera metadata.
- Image rejection fixture: missing/corrupt/duplicate chunks do not emit a bad frame.
- Camera model fixture: official intrinsics, zero distortion, zero translation, and `20 deg` upward tilt are represented deterministically.
- Camera projection sanity fixture: identity body pose plus `20 deg` upward camera tilt moves a straight-ahead point in the expected image direction.
- State fixture: known MAVLink/NED telemetry converts to internal z-up `VehicleState` exactly.
- State rejection fixture: missing or stale local position/odometry blocks command eligibility.
- Command fixture: one known command tuple maps to exact MAVLink command fields.
- Command serialization fixture: `SET_ATTITUDE_TARGET` quaternion/body-rate/type-mask fields match `PyAIPilotExample` behavior or a known-good packet capture.
- Command rate fixture: command adapter never publishes at or above `100 Hz`.
- Gazebo guard fixture: competition mode rejects Gazebo truth defaults and non-`None` `gazebo_pose`.
- Runner fixture: dry-run mode processes fake heartbeat, telemetry, image, and command candidate without sending a UDP command.

Acceptance criteria:

- Tests are deterministic and do not require network, simulator, GPU, YOLO weights, Gazebo, or live MAVLink.
- Tests do not write under `third_party/PyAIPilotExample/**`; any mutable copies of example files or payloads live under `tmp_path`, `/tmp`, or another generated test fixture directory outside the vendor tree.
- Tests use fake clocks for rate/freshness checks.
- Tests fail loudly on unit or frame-convention ambiguity.

## Phase 8.25 - Early Gate Geometry Constants Audit

Priority: P1 before interpreting perception or surrogate results.

Purpose:

- Audit gate and drone geometry constants before using offline replay, Gazebo/PX4 surrogate runs, or perception dry-run output to judge behavior.
- Prevent perception conclusions from being based on stale gate dimensions while preserving current PnP scoring behavior.

Implementation tasks:

- Read-only audit current PnP object points, gate-size constants, gate-clearance assumptions, and drone-size assumptions.
- Compare current constants to VADR-TS-002: `2700 mm` outer square, `1500 mm` inner square, `260 mm` depth, drone `280 mm x 280 mm x 160 mm`.
- Record every file and constant that already matches the spec.
- Record every file and constant that appears stale or ambiguous.
- If updates are needed, update only centralized constants and deterministic fixtures first.
- Do not change PnP candidate scoring, gate admission thresholds, low-gate crossing behavior, or race progression.

Acceptance criteria:

- Early geometry audit output is available before Phase 8.5 surrogate interpretation.
- Official gate/drone dimensions are recorded in one central config path or explicitly mapped to existing constants.
- Any stale or ambiguous geometry usage is listed before behavior tuning or perception-result interpretation.
- No scoring or progression behavior changes are made in this phase.

## Phase 8.5 - PX4/Gazebo Surrogate Competition Harness

Priority: P1 surrogate confidence only; does not unblock Phase 4B.

Purpose:

- Use PX4/Gazebo as a surrogate to validate competition runner wiring, timing pressure, camera packetization, state mapping, and dry-run command candidates when the real competition simulator is unavailable.
- Keep surrogate evidence explicitly separate from real competition simulator evidence.

Scope:

- Use PX4/MAVSDK estimated telemetry only; do not feed Gazebo model pose, Gazebo camera pose, Gazebo TF, or Gazebo truth into competition runner inputs.
- Configure or approximate the Gazebo camera to the VADR-TS-002 model: `640x360`, `fx=fy=320`, `cx=320`, `cy=180`, zero distortion, same body/camera origin, and `20 deg` upward tilt.
- Convert PX4/MAVSDK telemetry into MAVLink-like fake messages accepted by `CompetitionRunner` injected transports.
- JPEG-encode and packetize Gazebo camera frames into fake VADR `<IHHIIQ` vision packets.
- Feed telemetry and vision into `CompetitionRunner` through injected transports only.
- Keep command adapter in dry-run mode; `command_live` and `race` remain disabled.
- Label all logs and artifacts as PX4/Gazebo surrogate evidence, not competition simulator evidence.

Non-goals:

- Do not mark Phase 4B complete from PX4/Gazebo data.
- Do not claim competition telemetry readiness, command readiness, or race readiness.
- Do not add live competition MAVLink transport or live competition UDP vision transport.
- Do not send heartbeats, setpoints, actuator commands, attitude targets, position targets, or any MAVLink command from the competition runner.
- Do not use Gazebo truth to fabricate state, gate position, or command readiness.

Acceptance criteria:

- Surrogate harness can run with command publication disabled.
- Runner inputs are produced through fake/injected transports, not direct production sockets.
- Gazebo-truth guard remains active and rejects any Gazebo truth metadata that reaches the competition runner.
- Harness logs frame rate, packetization errors, telemetry freshness, command candidate rate, and command-blocked reasons.
- Results are explicitly documented as surrogate-only and cannot satisfy Phase 4B.

## Phase 9 - Offline Replay And Dry-Run Bring-Up

Priority: P1.

Purpose:

- Validate adapter behavior progressively before command-enabled runs, using real competition simulator evidence where simulator stages are required.
- Keep PX4/Gazebo surrogate evidence in Phase 8.5; do not use it to satisfy competition simulator observe, vision, command, or live stages.

Stage 1: import/static validation.

- Import all new modules.
- Instantiate config and adapter classes.
- Confirm no sockets open during import.
- Confirm no Gazebo guard failures in non-competition sim mode.

Stage 2: offline fixture/replay validation.

- Replay saved image packets through the image adapter.
- Replay saved MAVLink telemetry through the state adapter.
- Run `AutonomyAPI.update_gate_memory_from_frame(...)` with `gazebo_pose=None`.
- Run `path_plan(...)` and `attitude_control()` only if required inputs are valid.
- Keep command adapter in dry-run mode.

Stage 3: simulator observe mode.

- Start MAVLink heartbeat and telemetry receive only.
- Confirm heartbeat age and telemetry freshness.
- Record exact MAVLink messages, IDs, rates, fields, timestamps, and system-status flags emitted by the simulator.
- Verify whether local position/odometry is available.
- Stop here and scope a P0 state-estimation plan if usable local position/odometry is absent.
- Do not ingest vision and do not send commands.
- This stage requires the real competition simulator or an official equivalent; PX4/Gazebo surrogate output does not satisfy it.

Stage 4: simulator vision dry-run.

- Enable UDP vision receive and JPEG decode.
- Pass frames into perception with official camera metadata and `gazebo_pose=None`.
- Log frame rate, dropped chunks, decode failures, and perception update count.
- Do not send commands.
- This stage requires the real competition simulator or an official equivalent; Phase 8.5 only provides surrogate confidence.

Stage 5: simulator command dry-run.

- Run the full pipeline through command adapter dry-run.
- Verify command units and rate below `100 Hz`.
- Verify stale telemetry/image rejects command candidates.
- Do not send UDP command messages.
- This stage requires real competition telemetry and command-semantics evidence; PX4/Gazebo does not prove competition command acceptance.

Stage 6: live command output under external safety controls.

- Enable command sending only after Stages 1-5 pass.
- Keep detailed but bounded logs.
- Stop immediately on heartbeat loss, stale telemetry, stale image, invalid command, or Gazebo guard failure.

Acceptance criteria:

- Each stage has logs sufficient to diagnose the next failure without changing behavior mid-stage.
- Command-enabled mode is never the first simulator run.
- If state telemetry lacks position/odometry, stop after observe mode and scope the state-estimation unblocker.

## Phase 10 - Race Logging, Compliance, And Bounded I/O

Priority: P1 before submitted timed runs.

Purpose:

- Make competition runs autonomous, compliant, and not timing-limited by debug I/O.

Implementation tasks:

- Add a competition-safe `race` or `off` logging mode when the runner is ready for timed attempts.
- Bound debug-frame writes and sidecar logging so they cannot block `30 Hz` image handling or sub-`100 Hz` command output.
- Preserve existing logging schema during bring-up unless a separate migration is explicitly planned.
- Add a runner-level submitted-run mode that disables manual interaction after start.
- Add an `8 min` maximum run timer for submitted runs.
- Log start/intermediate/finish gate progression clearly enough to debug invalid runs.
- Log command rejection/failsafe events compactly.

Acceptance criteria:

- Submitted-run mode can run without human interaction after start.
- Manual command injection is impossible in submitted-run mode.
- Logging mode is bounded and does not write blocking debug frames in race mode.
- Run timer stops or safes the vehicle after `8 min` according to competition safety requirements.

Do not change:

- Planner behavior.
- Race progression internals.
- Logger schema during debug bring-up.

## Phase 11 - Gate Geometry And Target Constants Audit

Priority: P1 before submitted timed runs.

Purpose:

- Perform the final pre-submission geometry audit after the early Phase 8.25 audit and any fixture/constant updates.
- Ensure submitted-run constants match VADR-TS-002 without changing scoring behavior prematurely.

Implementation tasks:

- Re-run the Phase 8.25 geometry audit before submitted/timed attempts.
- Audit PnP object points and gate-size constants.
- Audit gate-clearance and drone-size assumptions.
- Compare current constants to VADR-TS-002: `2700 mm` outer square, `1500 mm` inner square, `260 mm` depth, drone `280 mm x 280 mm x 160 mm`.
- Confirm any earlier constant/fixture updates are still in effect.
- Add tests that assert official geometry constants are used where the adapter or PnP model expects them.

Acceptance criteria:

- Official gate/drone dimensions are recorded in one place.
- Any code still using old dimensions is listed explicitly.
- PnP scoring behavior is unchanged unless a separate tested behavior fix is approved.

Do not change:

- Candidate scoring.
- Gate admission thresholds.
- Low-gate crossing logic.
- `advance_gate_if_needed(...)`.

## Phase 12 - Deferred Work Gate

Priority: P2 unless a previous phase proves a blocker.

Do not start these until the adapter, tests, Gazebo guard, and dry-run stages are stable:

- MPC or tracker replacement.
- EKF/VIO or estimator fusion, unless Phase 4 proves local position/odometry is unavailable.
- YOLO retraining or domain randomization.
- PnP scoring changes.
- Gate-opening validation changes.
- Low-gate crossing behavior work.
- Race progression refactors.
- Deep planner changes.
- Behavior retuning for thrust, yaw, no-target behavior, or hover behavior.

Escalation rule:

- If the simulator lacks usable local position/odometry, create a new P0 state-estimation plan before enabling commands. Do not bury that work inside the thin adapter branch.

## Practical Implementation Order

Use this order for Codex implementation tasks:

1. Complete Phase 0 inventory.
2. Add the minimal Gazebo guard skeleton.
3. Add `RuntimeCompetitionConfig` or equivalent passive constants with tests.
4. Add `competition_image_adapter.py` for exact `<IHHIIQ` UDP header parsing, JPEG reassembly/decode, timestamps, official intrinsics, and no Gazebo pose.
5. Add explicit production camera model helpers, camera tilt tests, and the projection sanity fixture.
6. Run observe-mode MAVLink telemetry inventory against the simulator or provided example before assuming local position/odometry exists; this can be done with a minimal standalone observe script or a thin early runner skeleton before the full `competition_runner.py` exists.
7. Add `competition_state_adapter.py` only after telemetry availability and frame semantics are known.
8. Add `competition_command_adapter.py` for tuple-to-`SET_ATTITUDE_TARGET` dry-run and publish paths with exact serialization tests and rate limiting.
9. Add `competition_runner.py` in `observe`, `vision_dry_run`, and `command_dry_run` modes before any live command mode.
10. Expand the hard Gazebo-truth guard and run guard tests before command-enabled simulator runs.
11. Run deterministic adapter tests.
12. Run the Phase 8.25 early gate/drone geometry audit before interpreting perception or surrogate output.
13. Build the Phase 8.5 PX4/Gazebo surrogate harness if the real competition simulator remains unavailable; keep results labeled surrogate-only.
14. Run offline replay validation against saved frames and telemetry.
15. Run real competition simulator observe/vision/command dry-run stages when available; PX4/Gazebo surrogate output does not satisfy Phase 4B or Phase 9 simulator stages.
16. Enable live command output below `100 Hz` only after heartbeat, telemetry freshness, image timing, Gazebo guard, and command units are verified against the real competition simulator or an official equivalent.
17. Add bounded race logging, no-human-interaction safeguards, and the `8 min` submitted-run timer.
18. Complete the final Phase 11 gate-geometry audit before submitted runs.

## Definition Of Done For The Adapter Branch

The adapter branch is complete when:

- Competition constants are centralized and tested against VADR-TS-002 Issue 00.02.
- The minimal Gazebo guard skeleton exists before adapter bring-up, and the full guard fails fast on any competition-mode Gazebo truth path.
- UDP JPEG vision packets are reassembled, decoded, timestamped, and passed to `AutonomyAPI.update_gate_memory_from_frame(...)` with official camera metadata.
- The UDP vision header parser is tested against the exact little-endian `<IHHIIQ` `24` byte struct.
- Camera tilt direction is covered by a projection sanity fixture.
- Observe mode records the exact simulator MAVLink telemetry inventory before command enablement.
- MAVLink telemetry is converted from NED into internal z-up `VehicleState`, or missing position/odometry is explicitly identified as a P0 blocker.
- Existing `AutonomyAPI.attitude_control()` output is converted to MAVLink `SET_ATTITUDE_TARGET` without controller retuning, and serialization semantics are verified against the example or a known-good capture.
- `competition_runner.py` can run in dry-run modes with heartbeat, telemetry, vision, planning/control calls, and command dry-run logging.
- Deterministic tests cover image, camera, state, command, Gazebo guard, and runner dry-run behavior.
- No adapter implementation, test, formatter, or generated fixture modifies `third_party/PyAIPilotExample/**`.
- Command publication is disabled by default and constrained below `100 Hz` when enabled.
- Race/timed-run mode has bounded logging and no human interaction after start.

The shortest compliant path is thin adapters, exact frame/unit/timing tests, no Gazebo truth, dry-run validation, then live commands only after telemetry and command units are proven.
