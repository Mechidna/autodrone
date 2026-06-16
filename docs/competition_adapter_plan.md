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
- `autonomy_core/runtime/competition_mavlink_transport.py`
- `autonomy_core/runtime/competition_vision_transport.py`
- `autonomy_core/runtime/competition_setup.py`
- `autonomy_core/runtime/competition_main.py`
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

### Phase 6B - Production Competition Transport Modules

Priority: P0 for real transport dry-runs.

Purpose:

- Add production competition transport modules that mirror the reference
  `mavlink_rx.py` and `vision_rx.py` entry points without depending on
  PX4/Gazebo surrogate modules.
- Keep `CompetitionRunner` dependency-injected and free of socket ownership.

Primary files to add:

- `autonomy_core/runtime/competition_mavlink_transport.py`
- `autonomy_core/runtime/competition_vision_transport.py`
- Tests such as `tests/test_competition_mavlink_transport.py`
- Tests such as `tests/test_competition_vision_transport.py`

MAVLink transport requirements:

- Importing the module must not open sockets or import `pymavlink`.
- When explicitly started, open a MAVLink UDP endpoint such as
  `udpin:0.0.0.0:14540` or the official competition endpoint.
- Receive messages with nonblocking polling equivalent to
  `recv_match(blocking=False)`.
- Surface received MAVLink message objects or message-like wrappers to
  `CompetitionRunner.step(telemetry_messages=...)`.
- Record heartbeat status, message names, counts, rates, timestamp fields,
  source system/component IDs, and receive errors.
- Do not send heartbeats, setpoints, attitude targets, position targets,
  actuator commands, arm/offboard/reset commands, or any MAVLink command in
  Phase 6B.

Vision transport requirements:

- Importing the module must not open sockets or import `cv2`.
- When explicitly started, bind UDP vision on `0.0.0.0:5600` by default.
- Receive raw datagrams using the VADR `<IHHIIQ` JPEG-chunk packet format.
- Surface raw packet bytes to
  `CompetitionRunner.step(vision_packets=...)`; do not duplicate
  `CompetitionImageAdapter` parsing logic inside the transport.
- Record packet count, byte count, source address, dropped/timeout counts, and
  socket receive errors.
- Keep `CompetitionImageAdapter` strict about official `640x360` decoded
  frames.

Acceptance criteria:

- Both transports are import-safe.
- Unit tests use fake sockets/connections and do not bind live ports.
- Manual smoke commands may be added, but live sockets open only on explicit
  execution.
- No command publication is added.
- No files under `third_party/PyAIPilotExample/**` are modified.

Status:

- Implemented import-safe receive-only transport modules:
  `autonomy_core/runtime/competition_mavlink_transport.py` and
  `autonomy_core/runtime/competition_vision_transport.py`.
- The MAVLink transport lazily opens `pymavlink` only when explicitly started,
  receives with nonblocking `recv_match(blocking=False)`, records message
  inventory/statistics, and surfaces raw message objects through
  `receive_messages()`.
- The vision transport lazily binds UDP only when explicitly started, receives
  raw datagrams for the VADR `5600` vision path, records packet/source
  statistics, and surfaces raw packet bytes through `receive_packets()`.
- Deterministic tests are available in
  `tests/test_competition_mavlink_transport.py` and
  `tests/test_competition_vision_transport.py`; they use fake connections and
  fake sockets only.
- No setup module, main executable, live receive dry-run, surrogate vision
  bridge, command publication, heartbeat send, setpoint send, or command send
  was added in Phase 6B.

### Phase 6C - Competition Setup Wiring

Priority: P0 after Phase 6B.

Purpose:

- Add setup/wiring code analogous to `PyAIPilotExample/setup.py`, but using the
  production competition modules and existing `AutonomyAPI` facade.
- Keep executable lifecycle separate from dependency construction.

Primary file to add:

- `autonomy_core/runtime/competition_setup.py`

Setup responsibilities:

- Construct `CompetitionRunner`.
- Construct `CompetitionGuard`.
- Construct `CompetitionStateAdapter`, `CompetitionImageAdapter`, and
  `CompetitionDryRunCommandAdapter`.
- Construct `competition_mavlink_transport.py` and
  `competition_vision_transport.py` objects only when setup is explicitly
  called.
- Construct `AutonomyAPI` only when explicitly requested by a live/dry-run
  executable path; importing setup must not instantiate it.
- Apply the competition-mode Gazebo truth guard before any perception or
  command-capable mode is allowed to start.
- Keep command publication disabled by default.

Acceptance criteria:

- Importing `competition_setup.py` has no side effects.
- Tests can construct setup components with fake transports and fake autonomy.
- No live sockets open unless the caller explicitly requests live transports.
- No command publication is enabled.
- No planner, controller, YOLO, PnP, race progression, logging schema, or
  Gazebo diagnostic behavior changes.

Status:

- Implemented import-safe setup wiring in
  `autonomy_core/runtime/competition_setup.py`.
- `build_competition_runtime(...)` constructs `CompetitionRunner`,
  `CompetitionGuard`, `CompetitionStateAdapter`, `CompetitionImageAdapter`,
  `CompetitionDryRunCommandAdapter`, `CompetitionMavlinkTransport`, and
  `CompetitionVisionTransport` without opening sockets or starting loops.
- Setup supports injected fake transports and fake autonomy for tests.
- Real `AutonomyAPI` construction is lazy and requires explicit
  `use_real_autonomy=True` or an explicit `create_autonomy_api(...)` call; tests
  use a fake factory and do not instantiate real `AutonomyAPI`.
- Setup runs the competition-mode Gazebo truth guard before constructing the
  runner.
- Deterministic tests are available in `tests/test_competition_setup.py`.
- No Phase 6D `competition_main.py`, Phase 6E live receive dry-run,
  Phase 8.5E surrogate vision bridge, command publication, heartbeat send,
  setpoint send, or command send was added in Phase 6C.

### Phase 6D - Competition Main Executable

Priority: P0 after Phase 6C.

Purpose:

- Add the competition executable/lifecycle layer analogous to
  `PyAIPilotExample/main.py`.
- Keep it separate from setup construction and from the pure
  `CompetitionRunner` coordinator.

Primary file to add:

- `autonomy_core/runtime/competition_main.py`

CLI modes:

- `observe`: MAVLink heartbeat/telemetry only; no vision; no commands.
- `vision_dry_run`: MAVLink telemetry plus UDP `5600` vision; no commands.
- `command_dry_run`: full receive pipeline and no-send command candidates.
- `command_live`: present only as fail-closed until a later explicit command
  enablement phase.
- `race`: present only as fail-closed until race safeguards are implemented.

Executable requirements:

- Importing the module must not open sockets, import `pymavlink` or `cv2`, or
  instantiate `AutonomyAPI`.
- Running a mode explicitly may open the configured MAVLink endpoint and UDP
  vision socket.
- The loop must call `CompetitionRunner.step(...)` with batches from the
  production transports.
- The loop must report heartbeat, telemetry freshness, vision packet/frame
  counts, perception update counts, command candidate counts, command-blocked
  reasons, and guard rejections.
- Command publication must remain disabled in Phase 6D.

Acceptance criteria:

- CLI smoke tests can run with fake transports without live sockets.
- `command_live` and `race` fail closed.
- Manual live dry-runs are possible only when explicitly invoked.
- No command send path is added.

Status:

- Implemented import-safe CLI/lifecycle layer in
  `autonomy_core/runtime/competition_main.py`.
- The executable supports bounded dry-run summaries for `observe`,
  `vision_dry_run`, and `command_dry_run` through
  `competition_setup.build_competition_runtime(...)`.
- `command_live` and `race` remain fail-closed.
- Live transports require explicit `--live-transports`; tests use injected
  fake components and do not open sockets.
- Real `AutonomyAPI` is not instantiated by default.
- Summary output reports heartbeat/state status, telemetry counts, vision
  packet/frame counts, perception update counts, command candidate count,
  command-blocked reasons, and explicit no-readiness flags.
- Deterministic tests are available in `tests/test_competition_main.py`.
- No Phase 6E full live receive dry-run, Phase 8.5E surrogate vision bridge,
  command publication, heartbeat send, setpoint send, or command send was added
  in Phase 6D.

### Phase 6E - Full Receive Dry-Run With Live Transports

Priority: P0/P1 before Phase 9 dry-run interpretation.

Purpose:

- Prove the production competition receive path is wired end-to-end before
  offline replay, real competition simulator dry-runs, or command enablement.

Expected live receive path:

```text
MAVLink UDP endpoint
  -> competition_mavlink_transport.py
  -> CompetitionRunner.step(telemetry_messages=...)
  -> competition_state_adapter.py

UDP 0.0.0.0:5600 VADR JPEG packets
  -> competition_vision_transport.py
  -> CompetitionRunner.step(vision_packets=...)
  -> competition_image_adapter.py
  -> AutonomyAPI.update_gate_memory_from_frame(...)
```

PX4/Gazebo surrogate setup for Phase 6E:

- PX4/Gazebo sends MAVLink UDP directly to the configured competition MAVLink
  endpoint.
- `surrogate_vision_bridge.py` sends Gazebo camera pixels as VADR JPEG UDP
  packets to `127.0.0.1:5600`.
- `competition_main.py vision_dry_run` receives both through the same transport
  modules intended for the real competition simulator.

Acceptance criteria:

- `observe` receives heartbeat and usable telemetry without command
  publication.
- `vision_dry_run` receives MAVLink telemetry and UDP `5600` vision packets,
  completes at least one decoded frame, and calls the perception boundary with
  `gazebo_pose=None` and `image_pose_snapshot=None`.
- `command_dry_run` may build no-send command candidates when state/image
  freshness gates pass.
- `command_publication_allowed=false` and `command_sent_count=0` remain hard
  outputs.
- Logs are clearly marked as dry-run and, when PX4/Gazebo is used, surrogate
  evidence only.
- Phase 4B remains blocked unless the real competition simulator or official
  equivalent is used.

Status:

- Implemented receive-only Phase 6E verdict reporting in
  `autonomy_core/runtime/competition_main.py`.
- When `--live-transports` is used, summaries are labeled `phase: "6E"` and
  include `evidence_label`, `phase6e_receive_satisfied`,
  `phase6e_perception_boundary_satisfied`, `phase6e_satisfied`, and
  `phase6e_success_criteria`.
- `phase6e_receive_satisfied=true` requires live transports, telemetry
  messages, heartbeat, usable state, zero command publication, and, for
  `vision_dry_run` or `command_dry_run`, at least one received vision packet and
  completed frame.
- `phase6e_perception_boundary_satisfied=true` is separate and requires an
  injected or explicitly enabled autonomy object to receive a perception update
  with `gazebo_pose=None` and `image_pose_snapshot=None`.
- Manual PX4/Gazebo surrogate receive instructions and success criteria are
  documented in `docs/competition_adapter_phase6e_live_receive_dry_run.md`.
- Deterministic tests are available in `tests/test_competition_main.py`.
- Command publication remains disabled; `command_live` and `race` remain
  fail-closed.
- Phase 4B, Phase 9, command readiness, race readiness, and competition
  readiness are still not claimed.

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
- Command serialization fixture: `SET_ATTITUDE_TARGET` quaternion/body-rate/type-mask fields match the documented Phase 5A dry-run decision; `PyAIPilotExample` body-rate behavior remains non-equivalent reference evidence unless a later known-good packet capture proves otherwise.
- Command rate fixture: command adapter never publishes at or above `100 Hz`.
- Gazebo guard fixture: competition mode rejects Gazebo truth defaults and non-`None` `gazebo_pose`.
- Runner fixture: dry-run mode processes fake heartbeat, telemetry, image, and command candidate without sending a UDP command.

Phase 8 status:

- Implemented for the deterministic, no-live-transport adapter boundary.
- Coverage note: `docs/competition_adapter_phase8_deterministic_tests.md`.
- Phase 8.25 geometry audit and Phase 8.5 PX4/Gazebo surrogate harness remain separate future phases.
- No telemetry readiness, command readiness, race readiness, or Phase 4B completion is claimed.

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

Phase 8.25A status:

- Complete as a read-only audit in `docs/competition_adapter_phase8_25_geometry_audit.md`.
- Official gate/drone dimensions are centralized passively in `autonomy_core/core/competition_config.py`.
- Active YOLO PnP geometry, legacy stale geometry, duplicate-merge geometry assumptions, clearance proxies, and logging/debug geometry fields are inventoried.
- No runtime constants, PnP scoring, gate admission thresholds, clearance behavior, race progression, live transport, or command publication behavior were changed.
- Any fixes remain deferred to a separate Phase 8.25B or later behavior-reviewed phase.

Phase 8.25B status:

- Complete for passive geometry helper centralization and no-behavior-change cleanup.
- `RuntimeCompetitionConfig` exposes official meter helpers, inner/outer half extents, drone chassis meters, and planar inner-gate object points.
- Active YOLO default gate size and object-point construction now reference the official inner-square helpers while preserving the same `1.5 m` default and `+/-0.75 m` planar model points.
- Behavior-sensitive values remain unchanged: duplicate merge sizing, association/commit radii, race pass/clear radii, low-gate crossing proxies, legacy perception defaults, and logger field names.
- Phase 8.5 remains separate and not started; Phase 4B remains blocked pending real receive-only competition simulator telemetry evidence.

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

Phase 8.5A status:

- Scaffold implemented in `autonomy_core/runtime/px4_gazebo_surrogate_harness.py`.
- The scaffold converts PX4/MAVSDK estimated telemetry samples into fake runner-accepted MAVLink-like messages and packetizes pre-encoded JPEG bytes into fake VADR `<IHHIIQ` packets.
- The scaffold feeds `CompetitionRunner` only through injected fake transports and keeps `command_dry_run` no-send.
- `command_live` and `race` remain fail-closed; no live competition transport or command publication was added.
- Documentation is available in `docs/competition_adapter_phase8_5_px4_gazebo_surrogate_harness.md`.
- This is scaffold/deterministic-test coverage only; no real PX4/Gazebo surrogate run was performed, Phase 8.5 is not fully complete, Phase 4B remains blocked, and Phase 9 was not started.

Phase 8.5B status:

- Deterministic surrogate scenario fixtures are implemented in `tests/test_px4_gazebo_surrogate_scenarios.py`.
- The surrogate harness now supports multi-step scenario definitions with fake PX4 estimated telemetry samples, fake encoded JPEG byte frames, explicit runner modes, and deterministic timestamp validation.
- Scenario execution still uses the existing `CompetitionRunner.step(...)` path with injected fake batches only; no new runner modes, live transports, sockets, or command publication were added.
- Documentation is available in `docs/competition_adapter_phase8_5b_surrogate_scenarios.md`.
- This remains surrogate-only regression coverage; no real PX4/Gazebo run was performed, Phase 8.5 is not fully complete, Phase 4B remains blocked, and Phase 9 was not started.

### Phase 8.5C - Local UDP Vision Loopback For Image Adapter Only

Priority: P1 surrogate/protocol confidence only; does not unblock Phase 4B or Phase 9.

Purpose:

- Make the competition vision packet path easy to understand and manually test before adding any full runner or live simulator transport.
- Prove that `competition_image_adapter.py` can do the equivalent receiver-side work intended by `third_party/PyAIPilotExample/vision_rx.py`: receive UDP datagrams on `0.0.0.0:5600`, parse the exact `<IHHIIQ` header, reassemble JPEG chunks by `frame_id`, decode a completed JPEG image, attach official camera metadata, and emit a `CompetitionCameraFrame`.
- Provide a local mock sender that generates an actual in-memory `640x360` test image, JPEG-encodes it, packetizes it with the VADR header, and sends the packets to `127.0.0.1:5600`.

Scope:

- Add a standalone, explicit smoke tool, for example `autonomy_core/tools/competition_vision_udp_loopback.py`.
- The tool may open a UDP receiver socket only when explicitly executed, never on import.
- The receiver side binds to `0.0.0.0:5600` by default to mirror `PyAIPilotExample/vision_rx.py`.
- The sender side sends mock VADR packets to `127.0.0.1:5600` by default.
- The smoke tool should support a single-process loopback mode so one command can start the receiver, generate the mock image, send packets, process them through `CompetitionImageAdapter.process_packet(...)`, and print a compact summary.
- Any `cv2` import for JPEG encode/decode must be lazy and limited to explicit CLI execution or helper calls; importing competition runtime modules must remain safe.
- Use official `RuntimeCompetitionConfig` values for resolution, camera matrix, distortion, vision header format, and default port.

Non-goals:

- Do not start PX4, Gazebo, MAVSDK, ROS, the competition simulator, or any simulator process.
- Do not call `CompetitionRunner`, `AutonomyAPI`, perception, YOLO, PnP, planner, controller, state adapter, command adapter, or command publication paths.
- Do not add live competition MAVLink transport.
- Do not add the full live competition UDP vision transport to `CompetitionRunner`.
- Do not ingest Gazebo camera frames in this phase; that remains a later surrogate bridge phase.
- Do not send heartbeats, setpoints, actuator commands, attitude targets, position targets, or any MAVLink command.
- Do not claim Phase 4B, Phase 9, telemetry readiness, command readiness, race readiness, or competition readiness.
- Do not modify `third_party/PyAIPilotExample/**`.

Implementation tasks:

- Add a small loopback smoke module with import-safe functions for:
  - building a deterministic `640x360` mock image with a visible nonuniform pattern;
  - JPEG-encoding that image when `cv2` is available;
  - packetizing the JPEG bytes with `pack_vision_packet(...)` or equivalent VADR `<IHHIIQ` fields;
  - sending packets over UDP to a configurable host/port, default `127.0.0.1:5600`;
  - receiving datagrams from a configurable bind host/port, default `0.0.0.0:5600`;
  - passing received datagrams directly to `CompetitionImageAdapter.process_packet(...)`;
  - stopping after an expected number of completed frames or a timeout.
- Print or return a summary containing:
  - `bind_host`, `port`, `packets_sent`, `packets_received`, `frames_completed`, `decode_failures`, `packets_rejected`, `image_shape`, `image_stamp_sec`, `image_stamp_nanosec`, `camera_matrix`, `dist_coeffs`, `gazebo_pose`, and `image_pose_snapshot`.
- Add deterministic unit tests for packet-building and adapter processing without opening sockets.
- Add an opt-in/manual UDP loopback smoke command in docs. Unit tests should not bind `5600` by default because the port may be in use.
- If `cv2` is unavailable, fail the manual smoke command with a clear message and do not install dependencies.

Suggested manual command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m autonomy_core.tools.competition_vision_udp_loopback \
  --bind-host 0.0.0.0 \
  --send-host 127.0.0.1 \
  --port 5600 \
  --frames 1 \
  --timeout-s 5 \
  --save-mock-image /tmp/competition_vision_loopback_mock.jpg \
  --save-decoded-image /tmp/competition_vision_loopback_decoded.jpg
```

Expected manual smoke output:

```text
packets_sent > 0
packets_received == packets_sent
frames_completed == 1
image_shape == [360, 640, 3]
gazebo_pose == None
image_pose_snapshot == None
camera_matrix == [[320, 0, 320], [0, 320, 180], [0, 0, 1]]
dist_coeffs == [0, 0, 0, 0, 0]
mock_image_path == /tmp/competition_vision_loopback_mock.jpg
decoded_image_path == /tmp/competition_vision_loopback_decoded.jpg
```

Acceptance criteria:

- Importing the loopback module does not open sockets and does not import `cv2`.
- The explicit loopback command binds `0.0.0.0:5600`, sends mock VADR UDP packets to `127.0.0.1:5600`, and completes one decoded frame through `CompetitionImageAdapter`.
- The adapter statistics and summary prove that real UDP datagrams, not direct in-memory injection, reached `process_packet(...)`.
- The completed frame uses official competition camera metadata and has `gazebo_pose=None` and `image_pose_snapshot=None`.
- Corrupt or incomplete UDP packet sequences are rejected or time out deterministically.
- Protected vendor files remain unchanged.

Phase 8.5C status:

- Implemented for generated mock-image UDP loopback in `autonomy_core/tools/competition_vision_udp_loopback.py`.
- The tool builds a deterministic `640x360` mock image, lazily JPEG-encodes it, packetizes it with the VADR `<IHHIIQ` header, sends packets to `127.0.0.1:5600`, receives on `0.0.0.0:5600`, and passes datagrams directly to `CompetitionImageAdapter.process_packet(...)`.
- Optional `--save-mock-image` and `--save-decoded-image` outputs let the exact source and decoded images be inspected manually.
- Deterministic no-socket tests are available in `tests/test_competition_vision_udp_loopback.py`; the actual UDP bind/send path remains an explicit smoke command.
- Documentation is available in `docs/competition_adapter_phase8_5c_udp_vision_loopback.md`.
- This does not call `CompetitionRunner`, `AutonomyAPI`, perception, telemetry, command, simulator, or Gazebo paths; Phase 4B and Phase 9 remain blocked/not started.

### Phase 8.5D - Surrogate Runner Full Competition-Stack Bridge

Priority: P1/P2 staged surrogate confidence. This phase can provide strong
local integration confidence, but it must not unblock Phase 4B, Phase 9, or any
real competition readiness claim.

Purpose:

- Add a dedicated `surrogate_runner.py` entrypoint that lets developers run the
  competition stack against PX4/Gazebo-generated surrogate data instead of the
  legacy `px4_runner.py` path.
- Treat PX4/Gazebo as a mock competition input generator: estimated telemetry
  plus camera pixels in, competition-formatted telemetry and VADR JPEG packets
  out.
- Exercise the competition modules end to end:
  `CompetitionRunner`, `competition_state_adapter.py`,
  `competition_image_adapter.py`, `competition_guard.py`,
  `AutonomyAPI.update_gate_memory_from_frame(...)`, existing planner/control
  calls where safe, and `competition_command_adapter.py`.
- Lay the foundation for an explicitly opt-in PX4/Gazebo closed-loop control
  stage after no-send dry-run evidence is stable.

Architecture:

```text
PX4/Gazebo estimated telemetry + Gazebo camera pixels
  -> surrogate_runner.py
  -> PX4 estimated telemetry adapter
  -> fake MAVLink-like telemetry messages
  -> VADR JPEG packetizer for camera frames
  -> CompetitionRunner.step(telemetry_messages=..., vision_packets=...)
  -> competition_state_adapter.py
  -> competition_image_adapter.py
  -> competition_guard.py
  -> AutonomyAPI boundary
  -> competition_command_adapter.py
  -> dry-run command fields first
  -> optional PX4/Gazebo-only command sender in a later gated substage
```

Required separation from legacy runtime:

- Do not route this through `autonomy_core/tools/px4_runner.py`.
- Do not reuse `px4_runner.py` control loops, Gazebo truth pose plumbing,
  offboard setup, logging schema, or runner behavior.
- Use `px4_runner.py` only as read-only implementation context if needed.
- `surrogate_runner.py` must be the new local bridge into the competition
  pipeline.

Surrogate input rules:

- Telemetry input may come from PX4/MAVSDK estimated telemetry or receive-only
  MAVLink telemetry such as `HEARTBEAT`, `ATTITUDE`, `LOCAL_POSITION_NED`, and
  `ODOMETRY`.
- Camera input may come from Gazebo camera pixels, saved image files, or a mock
  generated image source.
- Gazebo camera pixels are allowed only as image pixels.
- The runner must reject Gazebo model pose, Gazebo camera pose, Gazebo TF,
  Gazebo link state, Gazebo world pose, depth truth, gate truth, track truth,
  and any pose snapshot derived from Gazebo truth.
- Camera frames must be encoded/packetized into VADR `<IHHIIQ` packets before
  entering the competition image path.
- PX4/Gazebo telemetry must be converted into the same message shape consumed
  by `CompetitionRunner` and `competition_state_adapter.py`.

Recommended module boundaries:

- `autonomy_core/runtime/surrogate_runner.py`
  - CLI entrypoint and orchestration.
  - May lazily import optional PX4/Gazebo dependencies only when an explicit
    live surrogate mode is requested.
  - Must not open sockets, initialize ROS, connect MAVSDK, import YOLO weights,
    or instantiate `AutonomyAPI` on import.
- `autonomy_core/runtime/surrogate_sources.py` or equivalent, if useful.
  - PX4/MAVLink receive-only telemetry source.
  - Gazebo/ROS camera image source.
  - Saved-image/mock-image source.
- `autonomy_core/runtime/surrogate_command_sink.py` or equivalent, if useful.
  - No-send command logger first.
  - Later PX4/Gazebo-only command sender behind an explicit safety gate.
- Existing `px4_gazebo_surrogate_harness.py`.
  - Keep deterministic unit-test/scenario helpers here; do not turn it into the
    live runner entrypoint.

Stage 8.5D-1: import-safe runner skeleton.

- Add `surrogate_runner.py` with CLI modes and explicit labels:
  `surrogate_label = "PX4/Gazebo surrogate only"`.
- Supported initial modes:
  - `mock_vision_dry_run`
  - `saved_image_vision_dry_run`
  - `px4_observe`
  - `px4_vision_dry_run`
  - `px4_command_dry_run`
- Unsupported/fail-closed initial modes:
  - `px4_command_send`
  - `race`
  - `competition_live`
- Importing the module must not open sockets, initialize ROS, import
  `pymavlink`, `mavsdk`, `rclpy`, `cv2`, or instantiate `AutonomyAPI`.
- The CLI must print that Phase 4B and competition readiness are not satisfied.

Phase 8.5D-1 status:

- Implemented the import-safe skeleton in `autonomy_core/runtime/surrogate_runner.py`.
- The skeleton defines supported initial modes: `mock_vision_dry_run`, `saved_image_vision_dry_run`, `px4_observe`, `px4_vision_dry_run`, and `px4_command_dry_run`.
- `px4_command_send`, `race`, and `competition_live` fail closed.
- The CLI prints a surrogate-only JSON summary with `phase4b_satisfied=false`, `competition_readiness_claimed=false`, `command_publication_allowed=false`, and `command_sent_count=0`.
- Import safety and fail-closed behavior are covered by `tests/test_surrogate_runner_skeleton.py`.
- Documentation is available in `docs/competition_adapter_phase8_5d_surrogate_runner.md`.
- No live sources, `CompetitionRunner` execution, `AutonomyAPI` instantiation, Gazebo/PX4 integration, or command publication were added.

Stage 8.5D-2: no-send mock/saved-image competition stack run.

- Feed generated mock images or saved images through the same VADR packetizer
  used by Phase 8.5C.
- Feed optional fake telemetry samples or saved telemetry fixtures through the
  existing surrogate telemetry conversion helpers.
- Call `CompetitionRunner.step(...)` with explicit injected batches.
- Allow an injected fake autonomy first.
- Then allow explicit `--use-real-autonomy` only after import and smoke tests
  prove it does not silently enable legacy Gazebo truth paths.
- Keep command publication disabled.

Acceptance for 8.5D-2:

- A mock or saved image reaches `CompetitionRunner`, completes through
  `competition_image_adapter.py`, and calls
  `update_gate_memory_from_frame(...)` with `gazebo_pose=None` and
  `image_pose_snapshot=None`.
- If telemetry is provided, it reaches `competition_state_adapter.py` and
  produces internal z-up `VehicleState`.
- If command dry-run is requested, the command adapter produces
  `SET_ATTITUDE_TARGET`-style dry-run fields and `send_ready=False`.
- Summary includes frame count, completed frames, telemetry samples, state
  usability, perception update count, command candidate count, command-blocked
  reasons, guard rejections, and explicit `phase4b_satisfied=false`.

Phase 8.5D-2 status:

- Implemented generated-image and saved-image no-send dry-runs in `autonomy_core/runtime/surrogate_runner.py`.
- `mock_vision_dry_run` now builds a deterministic mock image, JPEG-encodes it, packetizes it with the VADR `<IHHIIQ` header, and feeds injected packets into `CompetitionRunner.step(...)`.
- `saved_image_vision_dry_run` now loads a saved image, requires `640x360` unless `--resize-input-image` is supplied, JPEG-encodes it, packetizes it, and feeds injected packets into `CompetitionRunner.step(...)`.
- Both implemented modes use injected fake autonomy only and verify that perception kwargs keep `gazebo_pose=None` and `image_pose_snapshot=None`.
- Both implemented modes keep command publication disabled and report `phase4b_satisfied=false`, `competition_readiness_claimed=false`, and `command_sent_count=0`.
- Tests are available in `tests/test_surrogate_runner_skeleton.py`.
- Still deferred: fake/saved telemetry fixtures, live PX4/Gazebo telemetry, live Gazebo camera input, real `AutonomyAPI` ownership, telemetry-backed command dry-run, and PX4/Gazebo command send.

Stage 8.5D-3: live PX4/Gazebo receive-only surrogate run.

- Receive PX4/Gazebo estimated telemetry from the same kind of local source
  already proven by the `14540` receive-only smoke test.
- Receive Gazebo camera pixels from a live camera source, but only as pixels.
- Convert telemetry and images into competition-runner injected batches.
- Run `observe`, `vision_dry_run`, and `command_dry_run` loops.
- Do not send any control output.
- Record rates for telemetry, frames, packetization, perception updates, command
  candidates, stale state/image blocks, and guard rejection events.

Acceptance for 8.5D-3:

- The runner processes live surrogate telemetry and live surrogate camera
  pixels through competition adapters for a bounded duration.
- `CompetitionRunner` remains the only pipeline coordinator.
- No Gazebo truth metadata reaches the competition runner or AutonomyAPI call.
- No command, heartbeat, setpoint, attitude target, position target, actuator
  target, arm, offboard, or reset message is sent.
- Logs are clearly labeled as surrogate-only.

Phase 8.5D-3 status:

- Implemented receive-only PX4 surrogate modes in `autonomy_core/runtime/surrogate_runner.py`.
- `px4_observe` now receives bounded MAVLink telemetry from an explicit endpoint such as `udpin:0.0.0.0:14540`, or from injected fake messages in deterministic tests, and feeds it into `CompetitionRunner.step(...)` observe mode.
- `px4_vision_dry_run` now pairs receive-only MAVLink telemetry with generated, saved-image, or optional lazy ROS camera pixel sources, packetizes images as VADR `<IHHIIQ`, and feeds injected batches into `CompetitionRunner.step(...)`.
- `px4_command_dry_run` now uses the same receive-only telemetry path and may build a no-send dry-run command candidate through `competition_command_adapter.py`; `command_publication_allowed=false` and `command_sent_count=0` remain hard outputs.
- The optional `ros_camera` source imports ROS/cv_bridge lazily only when explicitly selected and must use image pixels only, never Gazebo pose, TF, link state, world pose, depth truth, gate truth, track truth, or pose snapshots.
- Deterministic tests cover injected MAVLink telemetry, no-send receive helper behavior, PX4 observe mode, PX4 vision dry-run, and PX4 command dry-run without opening sockets or importing live dependencies.
- No live PX4/Gazebo run was performed by Codex; local/manual PX4/Gazebo evidence is still needed before interpreting this as surrogate runtime evidence.
- Phase 4B remains blocked pending real receive-only competition simulator telemetry evidence. Phase 8.5D-3 surrogate output does not claim competition telemetry readiness, command readiness, race readiness, or submitted-run readiness.

Phase 8.5D-3A status:

- Implemented surrogate-only ROS camera resizing in `autonomy_core/runtime/surrogate_runner.py`.
- The resize applies only when `--vision-source ros_camera` is selected and `--resize-camera-to-competition` is explicitly provided.
- Non-competition-sized ROS camera frames, such as `1280x960`, are resized to the official VADR `640x360` resolution before JPEG encoding and VADR `<IHHIIQ` packetization.
- If a ROS camera frame is not `640x360` and `--resize-camera-to-competition` is not provided, the surrogate runner fails closed with a clear message instead of silently feeding a frame that `CompetitionImageAdapter` will reject.
- `CompetitionImageAdapter` remains strict and still rejects decoded frames that are not official competition resolution.
- Deterministic tests use a fake `1280x960` ROS camera frame and injected telemetry to prove explicit resize reaches `CompetitionRunner`, completes one vision frame, and calls the fake perception boundary without command publication.
- No command sending, Gazebo truth usage, Phase 4B claim, telemetry readiness claim, command readiness claim, race readiness claim, or competition readiness claim was added.

Stage 8.5D-4: closed-loop PX4/Gazebo command-send foundation.

This substage is intentionally later and must not be implemented until
8.5D-1 through 8.5D-3 are stable and reviewed.

- Add a PX4/Gazebo-only command sink behind a hard explicit flag such as
  `--enable-surrogate-command-send`.
- This flag must be unavailable in competition mode and must not change
  `CompetitionRunner` live command/race behavior.
- Require separate operator confirmation in the command line, for example:
  `--i-understand-this-sends-to-px4-gazebo-only`.
- Continue to route command creation through `competition_command_adapter.py`.
- Verify units before send:
  - roll, pitch, yaw in radians;
  - quaternion/body-rate/type-mask semantics;
  - thrust normalized or mapped explicitly to the chosen PX4/Gazebo command API;
  - command rate strictly below `100 Hz`;
  - stale telemetry/image blocks command send;
  - command send stops on timeout, guard rejection, stale state, stale image, or
    user interrupt.
- The command sink may target PX4/Gazebo through MAVSDK Offboard or MAVLink
  `SET_ATTITUDE_TARGET`, but the chosen path must be documented and tested
  separately.
- Do not arm automatically unless a later explicit safety-reviewed task adds
  that behavior.
- Do not use Gazebo truth to stabilize or validate commands.

Acceptance for 8.5D-4:

- Closed-loop command-send is available only in a PX4/Gazebo surrogate mode.
- The default remains no-send.
- `command_live` and `race` in `CompetitionRunner` remain fail-closed.
- Every sent command is traceable to a dry-run command candidate and includes
  timestamp, rate, state freshness, image freshness, units, and block/allow
  decision.
- Logs state that this is PX4/Gazebo surrogate command-send evidence only and
  does not satisfy Phase 4B, Phase 9, or competition command readiness.

Hard non-goals for all of Phase 8.5D:

- Do not modify `third_party/PyAIPilotExample/**`.
- Do not claim Phase 4B completion from PX4/Gazebo.
- Do not claim competition telemetry readiness, command readiness, race
  readiness, or competition readiness.
- Do not add live competition MAVLink or live competition UDP vision transport
  to `CompetitionRunner`.
- Do not use Gazebo truth for state, gate position, target validation, command
  readiness, perception correction, or debugging shortcuts in competition-path
  calls.
- Do not change planner, controller, YOLO, PnP scoring, perception core
  behavior, race progression, `advance_gate_if_needed(...)`, RuntimeConfig
  defaults, logging schema, thrust tuning, yaw tuning, hover behavior, or
  no-target behavior.
- Do not send commands in any mode except the later explicit PX4/Gazebo-only
  command-send substage.

Required tests before any live surrogate command-send:

- `surrogate_runner.py` import safety: no `pymavlink`, `mavsdk`, `rclpy`, `cv2`,
  sockets, or `AutonomyAPI` instantiation on import.
- Mock image and saved image reach `CompetitionRunner` through VADR packetized
  vision batches.
- Fake telemetry reaches `CompetitionRunner` through MAVLink-like telemetry
  batches.
- Guard rejects Gazebo truth metadata in telemetry, image metadata, runner
  startup metadata, and autonomy perception kwargs.
- `command_dry_run` produces no-send command candidates and records
  command-blocked reasons.
- `px4_command_send`, `race`, and `competition_live` modes fail closed until
  their explicit later safety gates exist.
- If a command-send sink is later added, tests must prove the default is no-send
  and the send path cannot run without both explicit CLI safety flags.

Suggested manual progression:

1. Run `surrogate_runner.py mock_vision_dry_run` with generated image input.
2. Run `surrogate_runner.py saved_image_vision_dry_run` with a saved Gazebo
   camera frame.
3. Run `surrogate_runner.py px4_observe` against PX4/Gazebo telemetry only.
4. Run `surrogate_runner.py px4_vision_dry_run` with live telemetry and live
   camera pixels, no commands.
5. Run `surrogate_runner.py px4_command_dry_run` to generate command candidates,
   still no-send.
6. Only after review, add and test explicit PX4/Gazebo-only command-send.

Expected summary fields:

- `surrogate_label`
- `mode`
- `source_kind`
- `telemetry_sample_count`
- `frame_count`
- `completed_packetized_frames`
- `vision_packets_processed`
- `vision_frames_completed`
- `perception_update_calls`
- `state_usable`
- `state_missing_reasons`
- `command_candidate_count`
- `command_publication_allowed`
- `command_sent_count`
- `command_blocked_reasons`
- `guard_rejection_count`
- `phase4b_satisfied=false`
- `competition_readiness_claimed=false`

### Phase 8.5E - Surrogate Vision Bridge To UDP 5600

Priority: P1 surrogate transport confidence only; does not unblock Phase 4B or
Phase 9 real simulator stages.

Purpose:

- Add a dedicated surrogate bridge that makes PX4/Gazebo camera output mimic
  the competition simulator vision interface.
- Keep the bridge separate from production competition code. Production
  competition modules must not import this bridge.

Primary file to add:

- `autonomy_core/runtime/surrogate_vision_bridge.py`

Bridge responsibility:

```text
Gazebo/ROS camera topic
  -> image pixels only
  -> optional validation or explicit resize to VADR resolution
  -> JPEG encode
  -> VADR <IHHIIQ> packetization
  -> UDP send to 127.0.0.1:5600 by default
```

Current Gazebo camera evidence:

- The local PX4/Gazebo camera has been configured to publish `640x360` images
  matching the official VADR resolution.
- The `640x360` Gazebo camera can feed the bridge without resize.
- The bridge should keep an explicit resize option for non-matching cameras,
  but resizing should not be required for the current configured Gazebo camera.

Requirements:

- Importing the bridge must not initialize ROS, import `cv2`, open sockets, or
  start simulator communication.
- Live execution may lazily import ROS/cv_bridge and `cv2` only when explicitly
  started.
- The bridge must never read Gazebo model pose, Gazebo camera pose, Gazebo TF,
  Gazebo link state, Gazebo world pose, depth truth, gate truth, track truth, or
  pose snapshots.
- The bridge must send only VADR UDP vision packets; it must not send MAVLink
  telemetry, heartbeats, setpoints, attitude targets, position targets, actuator
  commands, arm/offboard/reset commands, or any other command.
- The default send target should be `127.0.0.1:5600`, so
  `competition_vision_transport.py` can bind `0.0.0.0:5600` and receive the
  same packet shape expected from the real competition simulator.
- Use official `RuntimeCompetitionConfig` constants for resolution, header
  format, default UDP port, camera metadata, and timing labels.

Acceptance criteria:

- Deterministic tests can packetize fake in-memory camera frames without ROS,
  `cv2`, or sockets.
- Import safety tests prove the bridge does not load ROS, `cv2`, `pymavlink`,
  MAVSDK, or `AutonomyAPI` on import.
- A manual PX4/Gazebo smoke command can read `/camera`, JPEG-encode and
  packetize frames, and send VADR UDP packets to `127.0.0.1:5600`.
- When paired with Phase 6E, `competition_main.py vision_dry_run` receives the
  bridge output through the production `competition_vision_transport.py`.
- All output is labeled PX4/Gazebo surrogate evidence only and cannot satisfy
  Phase 4B, Phase 9 real competition simulator evidence, command readiness,
  race readiness, or competition readiness.

Status:

- Implemented surrogate-only bridge in
  `autonomy_core/runtime/surrogate_vision_bridge.py`.
- The bridge lazily reads ROS `sensor_msgs/Image` pixels, validates or
  explicitly resizes frames to the official `640x360` VADR resolution,
  JPEG-encodes frames, packetizes them with the exact `<IHHIIQ` vision header,
  and sends UDP packets to `127.0.0.1:5600` by default.
- The current local Gazebo camera publishes `640x360`, so the bridge can feed
  the production `competition_vision_transport.py` without resize.
- The bridge is surrogate-only and is not imported by production competition
  modules.
- Importing the bridge does not initialize ROS, import `cv2`, open sockets,
  import `pymavlink`, import MAVSDK, or instantiate `AutonomyAPI`.
- The bridge rejects truth-like ROS topics and never reads Gazebo model pose,
  camera pose, TF, link state, world pose, depth truth, gate truth, track truth,
  or pose snapshots.
- The bridge sends no MAVLink, heartbeats, setpoints, attitude targets,
  position targets, actuator commands, arm/offboard/reset commands, or command
  messages.
- Deterministic tests are available in `tests/test_surrogate_vision_bridge.py`.
- Manual PX4/Gazebo smoke instructions are documented in
  `docs/competition_adapter_phase8_5e_surrogate_vision_bridge.md`.
- This does not mark Phase 4B, Phase 9, telemetry readiness, command readiness,
  race readiness, or competition readiness complete.

## Phase 9 - Autonomy Profile, Offline Replay, And Dry-Run Bring-Up

Priority: P1.

Purpose:

- Validate adapter behavior progressively before command-enabled runs.
- Add a competition-safe `AutonomyAPI` profile before real perception is enabled.
- Keep PX4/Gazebo surrogate evidence in Phase 8.5/Phase 6E; do not use it to
  satisfy real competition simulator observe, vision, command, or live stages.
- Do not enable command publication in Phase 9A, 9B, or 9C.

### Phase 9A - Competition-Safe AutonomyAPI Profile

Priority: P1 before real perception dry-runs.

Purpose:

- Construct the existing `AutonomyAPI` through a competition-safe factory or
  profile instead of using legacy defaults directly.
- Keep `autonomy_api6.py` behavior-preserving; do not rewrite planner,
  controller, YOLO, PnP, race progression, or `advance_gate_if_needed(...)`.

Required behavior:

- Instantiate `AutonomyAPI` only when explicitly requested by caller/CLI.
- Force `perception_world_pose_source` to a non-Gazebo source such as `mavsdk`
  or a future competition state source; reject `gazebo_truth_sim_only`.
- Ensure competition perception calls pass `gazebo_pose=None` and
  `image_pose_snapshot=None`.
- Disable or bound debug-frame writes for dry-run safety.
- Keep `use_diagnostic_far_depth_correction=False`.
- Verify the profile does not consume Gazebo model pose, Gazebo camera pose,
  Gazebo TF, depth truth, gate truth, track truth, or pose snapshots.
- Make YOLO weights/config explicit; do not silently rely on a local hardcoded
  path without a visible config failure or override.
- Validate camera metadata is provided by the competition image adapter, not by
  legacy Gazebo defaults.

Acceptance criteria:

- Importing setup/main/profile modules does not instantiate real `AutonomyAPI`.
- Tests can build the competition-safe profile with a fake factory.
- Guard tests prove `gazebo_truth_sim_only` and non-`None` Gazebo pose inputs are
  rejected in competition mode.
- No planner, controller, perception thresholds, PnP scoring, gate progression,
  or logging schema behavior is retuned in this phase.

Status:

- Implemented competition-safe profile/factory in
  `autonomy_core/runtime/competition_autonomy_factory.py`.
- `competition_setup.py` routes explicit `use_real_autonomy=True` construction
  through the safe profile.
- Importing setup/main/profile modules does not instantiate real `AutonomyAPI`
  or load `cv2`, `pymavlink`, MAVSDK, ROS, or `rclpy`.
- The profile forces non-Gazebo pose source defaults, disables debug-frame
  writes, disables diagnostic far-depth correction, clears Gazebo pose
  snapshots, and marks the returned object as competition-profiled.
- Real perception cannot silently rely on the legacy hardcoded YOLO path; a
  caller must explicitly acknowledge that temporary dry-run behavior or add a
  YOLO-path-aware factory.
- Deterministic tests are available in
  `tests/test_competition_autonomy_factory.py` and
  `tests/test_competition_setup.py`.
- Documentation is available in
  `docs/competition_adapter_phase9a_autonomy_profile.md`.
- Phase 9B, Phase 9C, Phase 9D, Phase 4B, command readiness, race readiness,
  and competition readiness are still not claimed.

### Phase 9B - Real Perception Dry-Run, No Commands

Priority: P1 after Phase 9A.

Purpose:

- Run `competition_main.py vision_dry_run` with the competition-safe
  `AutonomyAPI` profile.
- Prove decoded competition-style frames can reach
  `AutonomyAPI.update_gate_memory_from_frame(...)` without Gazebo truth.

Acceptance criteria:

- `phase6e_receive_satisfied=true` or equivalent receive evidence exists before
  the run is interpreted.
- `phase6e_perception_boundary_satisfied=true` or an equivalent Phase 9B
  perception boundary flag is true.
- `perception_update_calls > 0`.
- `gazebo_pose=None` and `image_pose_snapshot=None` are passed to
  `update_gate_memory_from_frame(...)`.
- The run logs frame rate, dropped/duplicate/corrupt chunks, JPEG decode
  failures, and perception update count.
- `command_publication_allowed=false` and `command_sent_count=0`.
- Do not call `path_plan(...)` or `attitude_control()` unless explicitly needed
  for a later Phase 9C dry-run.
- PX4/Gazebo surrogate evidence can provide confidence but does not satisfy
  real competition simulator evidence requirements.

Status:

- Implemented explicit Phase 9B dry-run controls in
  `autonomy_core/runtime/competition_main.py`.
- `competition_main.py vision_dry_run` now supports `--real-perception` with
  the Phase 9A competition-safe `AutonomyAPI` profile.
- `--real-perception` requires `--use-real-autonomy` and `vision_dry_run`; live
  operator runs require `--live-transports`.
- The current live real-perception path requires
  `--allow-legacy-yolo-default` as an explicit temporary acknowledgment because
  `AutonomyAPI` still has a hardcoded legacy YOLO weights path.
- Summary output includes `phase: "9B"`,
  `phase9b_perception_dry_run_satisfied`, and
  `phase9b_success_criteria`.
- Command publication remains disabled; `command_live` and `race` remain
  fail-closed.
- Deterministic tests use fake components/factories and do not instantiate real
  `AutonomyAPI`, run YOLO, open sockets, or send commands.
- Manual PX4/Gazebo surrogate instructions and success criteria are documented
  in `docs/competition_adapter_phase9b_real_perception_dry_run.md`.
- Phase 9B.2 adds the opt-in `competition_official_ned` transform mode for
  real-perception competition dry-runs. Legacy `AutonomyAPI` and `px4_runner`
  defaults remain `physical_direct_rad_x_mirror`.
- The competition-safe `AutonomyAPI` profile now selects
  `competition_official_ned`, and `competition_main.py --real-perception`
  fails closed if a non-official transform is requested.
- Phase 9B.2 transform validation instructions and success criteria are
  documented in
  `docs/competition_adapter_phase9b_2_transform_validation.md`.
- Phase 9C, Phase 9D, Phase 4B, command readiness, race readiness, and
  competition readiness are still not claimed.

### Phase 9C - Command Candidate Dry-Run, No Send

Priority: P1 after Phase 9B.

Purpose:

- Run the full receive/perception/planning/control path far enough to build
  no-send command candidates through `competition_command_adapter.py`.
- Verify command units and rate limits without publishing any command.

Acceptance criteria:

- Phase 9A competition-safe `AutonomyAPI` profile is active.
- Phase 9B perception dry-run has passed or an offline replay fixture provides
  valid perception/state inputs.
- Phase 9B.2 official transform validation has passed, including no
  `z_below_safe_min` rejection for the near gate caused by the selected
  transform and no mirrored `det(R)=-1` active competition transform.
- `command_candidate_count > 0` when state/image freshness gates pass.
- Stale telemetry, stale image, invalid state, invalid command tuple, NaN, or
  infinity rejects command candidates deterministically.
- `command_publication_allowed=false`.
- `command_sent_count=0`.
- Command tuple semantics, quaternion/body-rate/type-mask fields, target IDs,
  thrust units, and command-rate checks are logged and test-covered.
- PX4/Gazebo no-send command candidates do not prove competition command
  acceptance.

Status:

- Implemented Phase 9C no-send command-candidate dry-run in
  `autonomy_core/runtime/competition_main.py` and
  `autonomy_core/runtime/competition_runner.py`.
- `competition_main.py command_dry_run --real-perception --use-real-autonomy`
  is labeled as `phase: "9C"` and remains no-send.
- The runner syncs usable `CompetitionStateAdapter` output into the
  competition-profiled `AutonomyAPI.telemetry` object before command dry-run
  planning/control.
- The runner attempts `path_plan(replan_reason="phase9c_command_dry_run")`
  when available, then calls `attitude_control()` and converts the tuple through
  `CompetitionDryRunCommandAdapter`.
- Summary output includes planning attempt/success/failure counts, accepted and
  rejected command candidate counts, `phase9c_success_criteria`,
  `phase9c_command_dry_run_satisfied`, and the final no-send
  `SET_ATTITUDE_TARGET` fields.
- Deterministic tests use fake transports/autonomy only and do not instantiate
  real `AutonomyAPI`, open sockets, run YOLO, run PX4/Gazebo, or send commands.
- Documentation is available in
  `docs/competition_adapter_phase9c_command_dry_run.md`.
- Phase 4B, Phase 9D, command readiness, race readiness, and competition
  readiness are still not claimed.

### Phase 9D - Real Competition Simulator Dry-Run Stages

Priority: P1 when the real competition simulator or official equivalent is
available.

Purpose:

- Repeat observe, vision dry-run, and command dry-run against the real
  competition simulator before any command-enabled mode.
- Use real competition evidence to resolve Phase 4B and command semantics.

Stages:

- Observe mode: receive heartbeat and telemetry only; record exact MAVLink
  messages, IDs, rates, fields, timestamps, and system-status flags.
- Telemetry decision: verify whether usable local position/odometry exists; if
  absent, stop and scope a P0 state-estimation/VIO/EKF unblocker.
- Vision dry-run: receive UDP `5600` VADR JPEG packets, decode frames, and call
  the competition-safe perception boundary with `gazebo_pose=None`.
- Command dry-run: build no-send command candidates only after telemetry and
  image freshness gates pass.
- Live command output under external safety controls is a later explicit phase,
  not automatic Phase 9D work.

Acceptance criteria:

- Each stage has logs sufficient to diagnose the next failure without changing
  behavior mid-stage.
- Command-enabled mode is never the first real simulator run.
- Phase 4B is marked complete only from real competition simulator or official
  equivalent receive-only telemetry evidence.
- If usable local position/odometry is missing, stop after observe mode and
  scope the state-estimation unblocker.
- PX4/Gazebo surrogate output and local UDP loopback output do not satisfy
  Phase 9D.

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
9. Add Phase 6A `competition_runner.py` in `observe`, `vision_dry_run`, and `command_dry_run` modes with fake/injected transports before any live command mode.
10. Add Phase 6B production receive transports: `competition_mavlink_transport.py` and `competition_vision_transport.py`.
11. Add Phase 6C `competition_setup.py` to construct transports, guards, adapters, `AutonomyAPI`, and `CompetitionRunner` only when explicitly called.
12. Add Phase 6D `competition_main.py` with `observe`, `vision_dry_run`, and `command_dry_run` CLI modes; `command_live` and `race` remain fail-closed.
13. Expand the hard Gazebo-truth guard and run guard tests before command-enabled simulator runs.
14. Run deterministic adapter tests.
15. Run the Phase 8.25 early gate/drone geometry audit before interpreting perception or surrogate output.
16. Build the Phase 8.5 PX4/Gazebo surrogate harness if the real competition simulator remains unavailable; keep results labeled surrogate-only.
17. Add Phase 8.5C local UDP vision loopback smoke tooling for `CompetitionImageAdapter` only; bind `0.0.0.0:5600`, send mock VADR JPEG packets to `127.0.0.1:5600`, and do not call the runner, autonomy, perception, telemetry, or command paths.
18. Add Phase 8.5D `surrogate_runner.py` in stages: import-safe/no-send first, mock/saved-image competition stack runs second, live PX4/Gazebo receive-only runs third, and PX4/Gazebo-only command-send only after a separate explicit safety gate.
19. Add Phase 8.5E `surrogate_vision_bridge.py` so PX4/Gazebo camera pixels can be emitted as VADR UDP packets to `127.0.0.1:5600` for the production vision transport.
20. Run Phase 6E full receive dry-run with live transports: PX4/Gazebo MAVLink directly into `competition_mavlink_transport.py`, surrogate bridge UDP `5600` into `competition_vision_transport.py`, and `competition_main.py vision_dry_run`; no commands.
21. Add Phase 9A competition-safe `AutonomyAPI` profile/factory before any real perception dry-run; do not use legacy defaults directly.
22. Run Phase 9B real perception dry-run with the safe profile, `gazebo_pose=None`, `image_pose_snapshot=None`, and no commands.
23. Run Phase 9B.2 official transform validation with `competition_official_ned`; do not proceed if the selected transform still causes near-gate `z_below_safe_min` rejection or mirrored `det(R)=-1` competition output.
24. Run Phase 9C command candidate dry-run with the safe profile; build no-send command candidates only.
25. Run Phase 9D real competition simulator observe/vision/command dry-run stages when available; PX4/Gazebo surrogate output and local UDP loopback output do not satisfy Phase 4B or Phase 9D simulator stages.
26. Enable live command output below `100 Hz` only after heartbeat, telemetry freshness, image timing, Gazebo guard, and command units are verified against the real competition simulator or an official equivalent.
27. Add bounded race logging, no-human-interaction safeguards, and the `8 min` submitted-run timer.
28. Complete the final Phase 11 gate-geometry audit before submitted runs.

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
