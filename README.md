# AIGP Autonomy Stack

This directory contains the active AIGP pilot stack used for drone-racing
control, perception, planning, and PX4/Gazebo validation. The current runtime is
centered on `aigp/pilot/main.py` and `aigp/config/runtime.toml`.

The stack has two practical operating profiles:

- Linux development and PX4/Gazebo validation
- Windows native competition runtime with UDP vision and MAVLink

The legacy monolithic runner is not the recommended entry point for new users.

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
- dataset capture/autolabeling

Use Linux or WSL/Linux for those.

### Windows Requirements

Known target:

- Windows 11
- Python 3.12
- NVIDIA GPU recommended
- Windows Firewall allowing Python UDP traffic
- YOLO weights copied into the repo

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

## Environment Overrides

The runtime supports these useful environment variables:

| Variable | Purpose |
| --- | --- |
| `RUNNER_MODE` | `px4` or `competition` |
| `VISION_SOURCE` | `udp` or `ros` |
| `PERCEPTION_BACKEND` | `yolo`, `blue`, or `orange` |
| `PERCEPTION_HZ` | Perception loop rate |
| `PERCEPTION_WORLD_POSE_SOURCE` | `mavsdk`, `camera_only`, `none`, `estimator`, `gazebo_camera_sim` |
| `CAMERA_MOUNT_PROFILE` | `competition`, `racer_mono_cam`, `px4_x500_mono_cam`, `custom`, `auto` |
| `MAVLINK_IP` | IP/interface used by `pymavlink` UDP input |
| `MAVLINK_PORT` | Overrides port for the selected runner mode |
| `YOLO_MODEL_PATH` | Path to YOLO `.pt` weights |
| `GATE_SOURCE_MODE` | `perception` or `ground_truth` |

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
  model=/home/paolo/datasets/drone-racing-dataset/yolo11n-pose.pt \
  data=/home/paolo/datasets/gazebo_gate_yolo_pose_racer_8k/gate_pose.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  project=/home/paolo/datasets/gazebo_gate_yolo_pose_racer_8k_runs \
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
yolo_model_path = "C:/Users/paolo/autonomy_core/aigp/models/gate_yolo_pose_8k/best.pt"
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
