# AIGP Autonomy Stack

This directory contains the active AIGP pilot stack used for drone-racing
control, perception, planning, and PX4/Gazebo validation. The current runtime is
centered on `aigp/pilot/main.py` and `aigp/config/runtime.toml`.

The stack has two practical operating profiles:

- Linux development and PX4/Gazebo validation
- Windows native competition runtime with UDP vision and MAVLink

The legacy monolithic runner is not the recommended entry point for new users.

![AIGP debug replay](docs/20260707_185854_replay.gif)


## Confirmed Target Specs

These are the target environments this stack is written for and should be kept
compatible with.

| Area | Confirmed target |
| --- | --- |
| Linux OS | Ubuntu 24.04 LTS |
| ROS | ROS 2 Jazzy |
| Gazebo | Gazebo Sim 8.x through ROS/Gazebo bridge |
| Python | Python 3.12 |
| Windows | Windows 11 with Python 3.12 for competition runtime only |
| GPU | NVIDIA CUDA-capable GPU recommended for YOLO race use |
| CPU fallback | Works for import/smoke tests, not recommended for real-time YOLO |
| Camera image | 640x360 competition camera model |
| Runtime model | MAVLink telemetry plus UDP image stream |
| YOLO model path | `aigp/models/gate_yolo_pose_8k/best.pt` |

Python 3.11 is also acceptable for most of the code. Python 3.10 is not
recommended because the runtime imports `tomllib`.

## What This Stack Does

The active runtime performs:

1. MAVLink telemetry receive through `pymavlink`.
2. UDP or ROS image receive.
3. YOLO pose detection of race gates.
4. PnP gate pose estimation from keypoints.
5. Gate memory, race-order filtering, and target selection.
6. Minimum-snap trajectory generation.
7. Attitude/thrust command streaming back over MAVLink.

Main entry point:

```bash
python3 ./aigp/pilot/main.py
```

Logged/debug entry point:

```bash
python3 ./aigp/tools/run_with_log.py
```

Primary configuration:

```text
aigp/config/runtime.toml
```

## Repository Layout

```text
aigp/
  config/
    runtime.toml                 # Main runtime configuration
  models/
    gate_yolo_pose_8k/
      best.pt                    # Trained YOLO pose weights
  pilot/
    main.py                      # Active runner
    setup.py                     # MAVLink, vision, perception, controller setup
    autonomy_wrapper.py          # Planning, target selection, race logic
    perception_wrapper.py        # YOLO/PnP perception integration
    controller.py                # MAVLink command streaming
    vision_rx.py                 # UDP competition image receiver
    ros_camera_rx.py             # ROS camera receiver for Linux sim/debug only
  tools/
    run_with_log.py              # Run stack and save compact debug logs
    randomize_gate_world.py      # Generate randomized Gazebo gate worlds
    capture_gazebo_yolo_pose.py  # Capture raw sim frames and metadata
    replay_debug_map.py          # Visual replay of debug.jsonl

autonomy_core/
  perception/                    # Gate perception, YOLO, PnP helpers
  planning/                      # Minimum-snap and validation modules
  racing/                        # Gate advancement and race admission logic
  core/                          # Frame conventions and shared types
```

## Required YOLO Files

For runtime, only the trained weights are required:

```text
aigp/models/gate_yolo_pose_8k/best.pt
```

Recommended optional files for reproducibility:

```text
aigp/models/gate_yolo_pose_8k/args.yaml
aigp/models/gate_yolo_pose_8k/gate_pose.yaml
```

The configured path should be repo-relative:

```toml
[perception]
yolo_model_path = "aigp/models/gate_yolo_pose_8k/best.pt"
yolo_keypoint_layout = "inner4_outer4"
yolo_keypoint_order = "image"
```

This works on Linux and Windows as long as the stack is launched from the repo
root.

If committing `.pt` files to GitHub, use Git LFS if the file is large:

```bash
git lfs track "*.pt"
git add .gitattributes aigp/models/gate_yolo_pose_8k/best.pt
```

## Python Dependencies

The current `pyproject.toml` does not install all runtime dependencies. Install
them explicitly.

Core runtime:

```bash
python -m pip install -e .
python -m pip install numpy scipy opencv-python pymavlink ultralytics
```

YOLO uses PyTorch through Ultralytics. For GPU use, install the PyTorch build
matching your CUDA driver before running real-time YOLO.

Useful import test:

```bash
python -c "import cv2, numpy, scipy, pymavlink, ultralytics, torch; print('ok', torch.cuda.is_available())"
```

## Linux Setup

Use Linux for PX4, Gazebo, ROS, dataset capture, autolabeling, and deep debug.

### Linux Requirements

Known-good target:

- Ubuntu 24.04 LTS
- Python 3.12
- ROS 2 Jazzy
- Gazebo Sim 8.x
- `ros_gz_bridge`
- NVIDIA GPU for practical YOLO speed

### Linux Virtual Environment

From repo root:

```bash
python3.12 -m venv .venv_ctrl
source .venv_ctrl/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install numpy scipy opencv-python pymavlink ultralytics
```

If using GPU, install the correct PyTorch CUDA package for your driver.

### Linux PX4/Gazebo Runtime

Use PX4 mode for local sim validation:

```bash
export WORLD=gate_test_1500mm_blue_random
export VISION_SOURCE=ros
export RUNNER_MODE=px4
export PERCEPTION_BACKEND=yolo
export PERCEPTION_WORLD_POSE_SOURCE=gazebo_camera_sim

python3 ./aigp/tools/run_with_log.py
```

This mode expects ROS camera topics and Gazebo dynamic pose when using
`gazebo_camera_sim`.

Typical ROS/Gazebo bridge:

```bash
ros2 run ros_gz_bridge parameter_bridge \
  "/world/${WORLD}/model/racer_mono_cam_0/link/camera_link/sensor/camera/image@sensor_msgs/msg/Image[gz.msgs.Image" \
  "/world/${WORLD}/model/racer_mono_cam_0/link/camera_link/sensor/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo" \
  "/world/${WORLD}/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V" \
  --ros-args \
  -r "/world/${WORLD}/model/racer_mono_cam_0/link/camera_link/sensor/camera/image:=/camera" \
  -r "/world/${WORLD}/model/racer_mono_cam_0/link/camera_link/sensor/camera/camera_info:=/camera_info"
```

If `PERCEPTION_WORLD_POSE_SOURCE=gazebo_camera_sim`, the stack needs the same
`WORLD` environment variable in the terminal running the stack.

### Linux Competition-Like Runtime

To run the same core path used on Windows, avoid ROS and use UDP vision:

```bash
export RUNNER_MODE=competition
export VISION_SOURCE=udp
export PERCEPTION_BACKEND=yolo
export PERCEPTION_WORLD_POSE_SOURCE=mavsdk
export CAMERA_MOUNT_PROFILE=competition
export MAVLINK_IP=0.0.0.0
export MAVLINK_PORT=14550
export YOLO_MODEL_PATH=aigp/models/gate_yolo_pose_8k/best.pt

python3 ./aigp/pilot/main.py
```

Before competition mode, make sure this is disabled in `runtime.toml`:

```toml
[perception_geometry_audit]
enabled = false
```

The runtime intentionally refuses to start competition mode with geometry audit
enabled because audit/debug code can use simulator-only truth.

## Windows Setup

Windows is recommended only for the native competition runtime:

- UDP MAVLink
- UDP image stream
- YOLO perception
- MAVLink attitude/thrust command output

Do not use native Windows for:

- ROS camera input
- Gazebo camera pose debug
- PX4 SITL/Gazebo orchestration
- Gazebo training-dataset capture/autolabeling

Use Linux or WSL/Linux for those.

The competition UDP runtime can record a raw camera/IMU VIO dataset natively
on Windows; see **VIO Dataset Capture** below. Run OpenVINS and the later
dataset conversion/replay step inside WSL2.

### Windows Requirements

Known target:

- Windows 11
- Python 3.12
- NVIDIA GPU recommended
- Windows Firewall allowing Python UDP traffic
- YOLO weights copied into the repo

For a step-by-step PyCharm setup with smoke tests and firewall checks, see
[`docs/windows_setup_pycharm.md`](docs/windows_setup_pycharm.md).

### Windows Virtual Environment

From PowerShell in the repo root:

```powershell
py -3.12 -m venv .venv_ctrl
.\.venv_ctrl\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install numpy scipy opencv-python pymavlink ultralytics
```

Import test:

```powershell
python -c "import cv2, numpy, scipy, pymavlink, ultralytics, torch; print('ok', torch.cuda.is_available())"
```

### Windows Competition Configuration

Use these runtime settings:

```toml
[runtime]
runner_mode = "competition"
calibration_only = false
perception_hold = false
competition_arm = false  # first Windows smoke test only; set true when ready to arm

[vision]
source = "udp"

[camera.mount]
profile = "competition"
competition_body_translation_m = [0.0, 0.0, 0.0]
competition_yaw_correction_deg = 0.0

[perception]
backend = "yolo"
world_pose_source = "mavsdk"
yolo_model_path = "aigp/models/gate_yolo_pose_8k/best.pt"
yolo_keypoint_layout = "inner4_outer4"
yolo_keypoint_order = "image"

[perception_geometry_audit]
enabled = false

[gate_source]
mode = "perception"
allow_ground_truth = false

[state_estimation]
allow_known_gate_correction = false
```

Current development configs may keep `perception_geometry_audit.enabled = true`
for PX4/Gazebo debugging. Competition mode will refuse to start until that is
disabled. For a first Windows networking/perception smoke test, also set
`competition_arm = false` so the stack cannot arm while you are checking packet
flow.

### Experimental gate-to-VIO alignment

`[experimental_gate_vio_alignment]` is a disabled-by-default localization
experiment. It uses temporally consistent gate PnP observations and the metric
positions in `[gate_source].known_gate_positions_neu` to maintain a separate
translation offset from the raw estimator/VIO frame into the map frame. It does
not write gate position, velocity, yaw, or scale into the underlying estimator.
When enabled, the older direct landmark position correction is bypassed so the
two methods cannot correct the same state simultaneously.

To observe it on the shadow estimator without changing the flight state source:

```toml
[runtime]
use_perception = true

[state_estimation]
mode = "mavlink"
run_shadow_estimator = true

[experimental_gate_vio_alignment]
enabled = true
```

The startup line must report `control_path=0 shadow_path=1`. To use its aligned
position for control, the state-estimation path must eventually be the live VIO
provider (currently `mode = "estimator"` uses the in-process Python estimator;
the repository does not yet stream live OpenVINS output into Windows). The
feature can also be toggled for one process with
`EXPERIMENTAL_GATE_VIO_ALIGNMENT=true` or `false`. Enabling it requires
perception and a non-empty metric gate map; configuration loading fails instead
of silently running without either input.

This first implementation estimates translation only. A single gate center is
not sufficient to safely estimate map yaw or VIO scale.

PowerShell preflight:

```powershell
$env:RUNNER_MODE="competition"
$env:VISION_SOURCE="udp"
$env:PERCEPTION_BACKEND="yolo"
$env:PERCEPTION_WORLD_POSE_SOURCE="mavsdk"
$env:CAMERA_MOUNT_PROFILE="competition"
$env:MAVLINK_IP="0.0.0.0"
$env:MAVLINK_PORT="14550"
$env:YOLO_MODEL_PATH="$PWD\aigp\models\gate_yolo_pose_8k\best.pt"

python -c "import sys; sys.path.insert(0,'aigp/pilot'); from runtime_config import load_runtime_config; c=load_runtime_config(); print(c.runtime.runner_mode, c.runtime.competition_arm, c.vision.source, c.camera.mount_profile, c.perception.yolo_model_path)"
```

PowerShell launch:

```powershell
cd <path-to-your-autonomy_core-clone>

$env:RUNNER_MODE="competition"
$env:VISION_SOURCE="udp"
$env:PERCEPTION_BACKEND="yolo"
$env:PERCEPTION_WORLD_POSE_SOURCE="mavsdk"
$env:CAMERA_MOUNT_PROFILE="competition"
$env:MAVLINK_IP="0.0.0.0"
$env:MAVLINK_PORT="14550"
$env:YOLO_MODEL_PATH="$PWD\aigp\models\gate_yolo_pose_8k\best.pt"

python .\aigp\pilot\main.py
```

Use `MAVLINK_IP=0.0.0.0` when another process or machine sends MAVLink to the
Windows computer. Use `127.0.0.1` only when the sender is local.

### Windows Firewall

If the stack hangs at:

```text
Waiting for heartbeat...
```

then MAVLink is not reaching the process. Check:

- Windows Firewall allows Python.
- Inbound UDP `14550` is allowed for MAVLink.
- `MAVLINK_PORT` matches the sender.
- The sender is targeting the Windows machine IP.
- `MAVLINK_IP=0.0.0.0` is used for external senders.

If MAVLink connects but perception never updates, check:

- UDP vision port is `5600` unless changed.
- Inbound UDP `5600` is allowed for camera frames.
- Firewall allows UDP vision packets.
- The competition sender is using the expected packet format.
- `VISION_SOURCE=udp`.

## Important Runtime Modes

### `RUNNER_MODE=px4`

Used for Linux PX4/Gazebo validation.

Behavior:

- Opens MAVLink on `mavlink.port_px4`, default `14540`.
- Primes PX4 Offboard if enabled.
- Can set PX4 mode to `OFFBOARD`.
- Can arm PX4.

### `RUNNER_MODE=competition`

Used for competition runtime.

Behavior:

- Opens MAVLink on `mavlink.port_competition`, default `14550`.
- Does not set PX4 Offboard mode.
- Streams commands directly.
- Can arm if `competition_arm=true`.
- Rejects debug-only sim truth modes.

### Competition Calibration-Only Mode

Use calibration-only mode to run hover acquisition, thrust-scale calibration,
and lateral-response calibration without starting YOLO, gate tracking, or
trajectory planning. After all enabled calibration stages finish, the stack
continues streaming a level `SET_ATTITUDE_TARGET` at the learned hover thrust
and current yaw, with bounded vertical position/velocity damping. It does not
enter the `SET_POSITION_TARGET_LOCAL_NED`
`HoverHold` fallback.

For an armed competition-simulator calibration run:

```toml
[runtime]
runner_mode = "competition"
calibration_only = true
perception_hold = false
prearm_gate_acquisition = false
competition_arm = true
```

The equivalent temporary PowerShell override is:

```powershell
$env:CALIBRATION_ONLY="true"
$env:PREARM_GATE_ACQUISITION="false"
python .\aigp\tools\run_with_log.py
Remove-Item Env:\CALIBRATION_ONLY
Remove-Item Env:\PREARM_GATE_ACQUISITION
```

`competition_arm` has no environment override, so it must be enabled in
`runtime.toml` when the stack is responsible for arming. Leave it `false` if
the simulator is armed separately.

Calibration-only mode automatically suppresses the perception worker even if
`runtime.use_perception` and `perception.enabled` remain `true`. Camera frames
are not required in this mode; attitude, IMU, armed state, and the configured
position estimate remain required by the calibration stages.

A clean completion is visible in the run log as:

```text
hover_acquisition ... done=1 status=stable
thrust_scale_calibration ... done=1 status=calibrated
lateral_response_calibration ... done=1 status=calibrated
calibration_only_hold command_type=SET_ATTITUDE_TARGET ... thrust=...
```

`timeout_fallback` or `motion_limited_fallback` means the sequence finished
using fallback values, not that calibration succeeded cleanly. In competition
mode, stopping the client does not currently send a land or disarm command;
stop or reset the simulator safely rather than relying on `Ctrl+C` to disarm.

### Competition Perception-Hold Mode

Use perception-hold mode for an armed perception diagnostic after calibration.
It runs hover acquisition, thrust-scale calibration, and lateral-response
calibration, captures the resulting XY/Z/yaw pose, and holds that pose with
`SET_ATTITUDE_TARGET`. YOLO and GateMemory continue running, but perceived
landmarks are not fed into the state estimator, no navigation target is
installed, and all trajectory planning, gate advancement, and gate-pass logic
are bypassed.

```toml
[runtime]
runner_mode = "competition"
use_perception = true
calibration_only = false
perception_hold = true
prearm_gate_acquisition = false
perception_hold_settle_speed_m_s = 0.15
perception_hold_settle_duration_s = 0.50
competition_arm = true

[state_estimation]
mode = "mavlink"

[gate_source]
mode = "perception"
```

The temporary PowerShell override is:

```powershell
$env:CALIBRATION_ONLY="false"
$env:PERCEPTION_HOLD="true"
$env:PREARM_GATE_ACQUISITION="false"
python .\aigp\tools\run_with_log.py
Remove-Item Env:\PERCEPTION_HOLD
Remove-Item Env:\PREARM_GATE_ACQUISITION
```

GateMemory is deliberately quarantined during calibration and while residual
motion settles. It also rejects perception results originating from camera
frames captured before the post-calibration barrier, including inference that
was still in flight at handoff. A missing or failed camera stream prevents the
perception diagnostic from succeeding, but does not interrupt the MAVLink-based
flight hold.

A healthy run shows all enabled calibrations completing with `succeeded=1`,
then recurring lines such as:

```text
perception_hold reason=post_calibration_perception_only ... memory_tracks=...
[PERCEPTION_CHAIN] event=committed ...
[PERCEPTION_CHAIN] event=stable ...
```

There should be no `plan_install`, target-shift, gate-pass, or race-advance
events. Stop the client and reset the simulator after this diagnostic; do not
reuse a perception-hold process across a disarm/re-arm or simulator reset.

### Pre-Control Gate Acquisition

For a normal perception flight, pre-control acquisition lets the passive
MAVLink and camera receivers build GateMemory before the client emits any
flight-control or thrust commands. In the official competition simulator it
can also continue while the externally armed vehicle is still pinned by the
Ready countdown:

```toml
[runtime]
calibration_only = false
perception_hold = false
prearm_gate_acquisition = true
prearm_gate_min_duration_s = 2.0
prearm_gate_min_stable = 1
prearm_gate_max_age_s = 0.75
prearm_gate_ready_updates = 2
```

The collection duration starts with the first valid projected perception
result. Only fresh, stable, committed tracks count, and every consecutive
readiness update must match a stable track in that new YOLO result. Their
current filtered centers are locked for planning at handoff. GateMemory is
then frozen during hover/thrust/lateral calibration so vehicle motion cannot
rewrite the pre-arm map, and it resumes only on a newer post-calibration
camera result. Acquisition waits indefinitely for readiness. During this
phase the controller is not updated, so no `SET_ATTITUDE_TARGET` or thrust
command is emitted.

PX4 still requires a disarmed heartbeat. In competition mode, an armed
heartbeat is accepted only while fresh race-status timing proves the race has
not started. The projected countdown deadline is checked between status
packets; if the race starts before a gate is locked, startup reports
`race_started_before_prearm_ready` and does not enter the control loop. Reset
the simulator before retrying. When the competition heartbeat is already
armed, the later duplicate arm request is skipped.

Start the client before pressing **Ready**. Model and perception initialization
can take longer than the simulator's five-second countdown, so launching after
Ready may correctly report `race_started_before_prearm_ready` before gate
acquisition has had time to run.

A successful startup includes:

```text
prearm_gate_lock track=... center_neu=(...) hits=... score=...
prearm_gate_acquisition ready ... locked=[...]
```

To temporarily bypass this wait for a non-flight diagnostic:

```powershell
$env:PREARM_GATE_ACQUISITION="false"
```

## Environment Overrides

The runtime supports these useful environment variables:

| Variable | Purpose |
| --- | --- |
| `RUNNER_MODE` | `px4` or `competition` |
| `CALIBRATION_ONLY` | `true` runs calibration stages, then attitude-only hover |
| `PERCEPTION_HOLD` | `true` calibrates, then holds XY/Z/yaw while observing perception only |
| `PREARM_GATE_ACQUISITION` | `true` requires fresh committed gates before any local flight-control output |
| `VISION_SOURCE` | `udp` or `ros` |
| `PERCEPTION_BACKEND` | `yolo`, `blue`, or `orange` |
| `PERCEPTION_HZ` | Perception loop rate |
| `PERCEPTION_WORLD_POSE_SOURCE` | `mavsdk`, `camera_only`, `none`, `estimator`, `gazebo_camera_sim` |
| `CAMERA_MOUNT_PROFILE` | `competition`, `racer_mono_cam`, `px4_x500_mono_cam`, `custom`, `auto` |
| `MAVLINK_IP` | IP/interface used by `pymavlink` UDP input |
| `MAVLINK_PORT` | Overrides port for the selected runner mode |
| `YOLO_MODEL_PATH` | Path to YOLO `.pt` weights |
| `GATE_SOURCE_MODE` | `perception` or `ground_truth` |
| `AIGP_VIO_RECORD_DIR` | Enables raw VIO recording at the specified directory |
| `AIGP_VIO_QUEUE_SIZE` | Pending asynchronous dataset writes; default `512` |

There is no environment override for `perception_geometry_audit.enabled` or
`runtime.competition_arm`; edit `runtime.toml` before competition runs.

## Debug Logging

For Linux/PX4 debugging, prefer:

```bash
python3 ./aigp/tools/run_with_log.py
```

Logs are written under:

```text
aigp/logs/runs/<run_id>/
  stdout.log
  debug.jsonl
```

Generate the visual replay:

```bash
python3 ./aigp/tools/replay_debug_map.py \
  aigp/logs/runs/<run_id>/debug.jsonl
```

Open the generated:

```text
aigp/logs/runs/<run_id>/replay_debug_map.html
```

## VIO Dataset Capture

For a Windows competition-simulator run, start the normal logged runner with
the opt-in VIO flag:

```powershell
python .\aigp\tools\run_with_log.py --capture-vio-dataset
```

To inventory or capture a simulator that does not provide flight state, add
the fail-safe observe-only policy:

```powershell
python .\aigp\tools\run_with_log.py `
  --observe-only `
  --capture-vio-dataset `
  --vio-dataset-queue-size 2048 `
  --run-id vq2_vq1_observe_01
```

Observe-only mode skips PX4 offboard priming, mode changes, arming, autonomy
updates, controller updates, simulator reset commands, and shutdown landing.
The controller also independently refuses to transmit if one of those methods
is called accidentally. MAVLink TIMESYNC and best-effort telemetry-rate
requests remain enabled because they are sensor-capture protocol traffic, not
flight-control commands. `run.log` must contain `OBSERVE_ONLY=True` and
`OBSERVE_ONLY active`; `manifest.json` records `runtime.observe_only: true`.

This does not enable VIO or change the state source used for control. It adds a
`vio_dataset` directory to the timestamped run and asynchronously records:

```text
vio_dataset/
  manifest.json
  runtime.toml
  events.jsonl
  camera/data.csv
  camera/data/*.jpg
  imu/data.csv                 # HIGHRES_IMU
  imu/hil_sensor.csv
  imu/scaled_imu.csv
  imu/scaled_imu2.csv
  imu/scaled_imu3.csv
  imu/raw_imu.csv              # diagnostic-only, device-specific scale
  timesync/data.csv
  truth/attitude.csv
  truth/local_position_ned.csv
  truth/odometry.csv
```

The camera files are the original reassembled simulator JPEG payloads; they are
not decoded and re-encoded for this dataset. `camera/data.csv` preserves each
frame's original `sim_time_ns`, the IMU CSVs preserve every received source
timestamp plus raw/unit metadata, and the TIMESYNC file includes both
transmitted and received `tc1`/`ts1` values. The `truth` files are for offline
scoring only and must not be fed to OpenVINS.

After stopping the run, open `manifest.json` and require all
`dropped_queue_full` and `write_failures` counts to be zero. Also check
`events.jsonl` for `camera_frame_dropped`, `camera_packet_rejected`, or
`camera_decode_failed`. If the disk falls behind, repeat with a larger bounded
queue, for example:

```powershell
python .\aigp\tools\run_with_log.py `
  --capture-vio-dataset `
  --vio-dataset-queue-size 2048
```

The UDP receiver requests an 8 MiB kernel receive buffer before binding,
retains up to 64 incomplete frames for one second, and suppresses completed
frame retransmissions for five seconds using `(frame_id, sim_time_ns)`. At
startup it prints both the requested and effective `UDP_RCVBUF`; verify the
effective value is at least 8388608 on Windows. A
`camera_duplicate_frame_suppressed` event is expected when the simulator
retransmits a frame and does not represent a dataset duplicate. Completed
camera rows should still be unique by frame ID and simulator timestamp.

The MAVLink receiver separately requests an 8 MiB kernel receive buffer and a
120 Hz `HIGHRES_IMU` interval. During `--capture-vio-dataset` only, it also
requests `HIL_SENSOR`, `SCALED_IMU`, `SCALED_IMU2`, `SCALED_IMU3`, and
`RAW_IMU` at 120 Hz so their six-axis signals can be compared. These requests
target the component that sent the startup heartbeat, independently of the
configured flight-control target. They are best-effort: startup continues if
the endpoint rejects or does not support one, and any matching `COMMAND_ACK`
is written to `events.jsonl`. Normal flight runs do not request the extra
streams. Exact byte-identical MAVLink packet
retransmissions received within 100 ms are suppressed. Sensor values are never
deduplicated merely because their timestamps or values match. IMU and truth CSV
rows include `mavlink_seq`, `mavlink_src_system`, and
`mavlink_src_component`; the final `mavlink_rx_summary` event reports duplicate,
sequence-gap, and out-of-order counts. Verify the startup log reports an
effective MAVLink `UDP_RCVBUF` of at least 8388608 bytes, then require a stable
IMU rate of at least 100 Hz before treating a capture as the final OpenVINS
benchmark.

From WSL2, the default captures are visible under:

```bash
/mnt/c/dev/autonomy_core/aigp/logs/runs/<run_id>/vio_dataset
```

Use `aigp/tools/prepare_openvins_dataset.py --imu-source ...` to build a
source-specific offline replay. The full Windows/WSL workflow and supported
source names are in `aigp/openvins/README.md`.

## Dataset Capture And Autolabeling

These tools are Linux/Gazebo-only.

Capture Gazebo frames:

```bash
export WORLD=gate_test_1500mm_blue_random

python3 ./aigp/tools/capture_gazebo_yolo_pose.py \
  --capture-root ~/datasets/gazebo_gate_capture_racer \
  --capture-hz 10 \
  --dynamic-pose-topic /world/$WORLD/dynamic_pose/info \
  --allow-pose-fallback
```

Autolabel captures for the 8-keypoint training dataset:

```bash
python3 ./autonomy_core/tools/autolabel_gazebo_yolo_pose.py \
  --capture-root ~/datasets/gazebo_gate_capture_racer \
  --output-root ~/datasets/gazebo_gate_yolo_pose_racer_8k \
  --keypoint-layout inner4_outer4 \
  --allow-partial-gates \
  --label-all-visible-gates \
  --draw-preview \
  --gazebo-rotation-mode transpose \
  --gazebo-optical-mode physical_minus_y
```

The generated `gate_pose.yaml` will use `kpt_shape: [8, 3]` with keypoints
ordered as inner TL/TR/BR/BL followed by outer TL/TR/BR/BL. The inner keypoints
are projected on the gate exit face, 0.130 m downstream from the SDF gate center
plane. The outer keypoints are projected on the entry/visible face, 0.130 m
upstream, so they stay on the visible frame silhouette. Omit
`--keypoint-layout inner4_outer4` to keep generating the existing 4-keypoint
inner-corner dataset, also on the exit face. Keypoints hidden by the same gate's
own frame or by another gate frame are kept labeled with YOLO visibility `1`.

Train the 8-keypoint model:

```bash
yolo pose train \
  model=~/datasets/drone-racing-dataset/yolo11n-pose.pt \
  data=~/datasets/gazebo_gate_yolo_pose_racer_8k/gate_pose.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  project=~/datasets/gazebo_gate_yolo_pose_racer_8k_runs \
  name=inner4_outer4
```

To run the 8-keypoint perception/PnP path after training, set
`perception.yolo_model_path` in `aigp/config/runtime.toml` to the new
`.../weights/best.pt` and set `perception.yolo_keypoint_layout =
"inner4_outer4"`. Use `yolo_keypoint_layout = "inner4"` with the existing
4-keypoint model.

Randomize a world:

```bash
python3 ./aigp/tools/randomize_gate_world.py \
  --seed 1001 \
  --gate-count 10 \
  --gate-spacing-m 12 \
  --sequential-gate-height-step-m 5 \
  --randomize-positions true \
  --randomize-lighting true \
  --gate-rgb 1.0 0.45 0.0 \
  --update-runtime true
```

Restart Gazebo/PX4 after randomizing a world. Gazebo will keep the old world if
it is already running.

## Troubleshooting

### `Waiting for heartbeat...`

MAVLink is not reaching the runner.

Check:

- Correct `RUNNER_MODE`.
- Correct `MAVLINK_PORT`.
- `MAVLINK_IP=0.0.0.0` for external senders.
- Firewall allows UDP.
- Sender is targeting the correct host.

### `perception_geometry_audit is debug-only`

Competition mode refuses to start while audit is enabled.

Fix:

```toml
[perception_geometry_audit]
enabled = false
```

### No camera frames

Check:

- `VISION_SOURCE=udp` for competition.
- UDP vision sender uses port `5600`.
- Firewall allows UDP.
- Packet format matches `packet_header_format = "<IHHIIQ"`.

### YOLO model fails to load

Check:

- `aigp/models/gate_yolo_pose_8k/best.pt` exists.
- `YOLO_MODEL_PATH` points to the correct file.
- `ultralytics` and `torch` import.
- CUDA device exists if `yolo_device = 0`.

For CPU smoke tests, set:

```toml
yolo_device = "cpu"
```

### Windows path issues

Prefer repo-relative paths in `runtime.toml`:

```toml
yolo_model_path = "aigp/models/gate_yolo_pose_8k/best.pt"
```

If using an absolute Windows path in TOML, use forward slashes:

```toml
yolo_model_path = "C:/Users/<you>/autonomy_core/aigp/models/gate_yolo_pose_8k/best.pt"
```

### ROS import errors on Windows

Do not use ROS mode on native Windows.

Use:

```powershell
$env:VISION_SOURCE="udp"
```

## Safety Notes

- `runner_mode="competition"` must not depend on Gazebo truth, known gate
  positions, or ROS sim pose.
- `perception_geometry_audit` is for PX4/Gazebo debug only.
- `gate_source.mode="ground_truth"` is debug-only and invalid for competition.
- `gazebo_camera_sim` is PX4/Gazebo debug-only.
- Competition camera mount must resolve to zero translation and zero yaw
  correction.

## Quick Start Summary

Linux PX4/Gazebo debug:

```bash
source .venv_ctrl/bin/activate
export WORLD=gate_test_1500mm_blue_random
export RUNNER_MODE=px4
export VISION_SOURCE=ros
export PERCEPTION_BACKEND=yolo
export PERCEPTION_WORLD_POSE_SOURCE=gazebo_camera_sim
python3 ./aigp/tools/run_with_log.py
```

Windows competition:

```powershell
.\.venv_ctrl\Scripts\Activate.ps1
$env:RUNNER_MODE="competition"
$env:VISION_SOURCE="udp"
$env:PERCEPTION_BACKEND="yolo"
$env:PERCEPTION_WORLD_POSE_SOURCE="mavsdk"
$env:CAMERA_MOUNT_PROFILE="competition"
$env:MAVLINK_IP="0.0.0.0"
$env:MAVLINK_PORT="14550"
$env:YOLO_MODEL_PATH="$PWD\aigp\models\gate_yolo_pose_8k\best.pt"
python .\aigp\pilot\main.py
```
