# Windows PyCharm Setup Guide

This guide is for running the AIGP stack on native Windows for the competition
runtime only. It does not set up ROS, Gazebo, PX4 SITL, dataset capture, or
autolabeling on Windows.

Use Windows for:

- UDP camera frames
- MAVLink telemetry
- YOLO gate perception
- MAVLink attitude/thrust command output

Use Linux for:

- ROS camera input
- Gazebo camera pose debug
- PX4/Gazebo orchestration
- Dataset capture and autolabeling

## 1. Install Python 3.12

1. Download Python 3.12 for Windows from:

   ```text
   https://www.python.org/downloads/windows/
   ```

2. Run the installer.

3. On the first installer screen, check:

   ```text
   Add python.exe to PATH
   ```

4. Click:

   ```text
   Install Now
   ```

5. Open PowerShell and verify Python:

   ```powershell
   py -3.12 --version
   python --version
   ```

Expected result:

```text
Python 3.12.x
```

If `python --version` does not show Python 3.12, use `py -3.12` in commands
until the PATH issue is fixed.

## 2. Install Git

1. Download Git for Windows:

   ```text
   https://git-scm.com/download/win
   ```

2. Install with the default options.

3. Open PowerShell and verify:

   ```powershell
   git --version
   ```

## 3. Clone Or Open The Repo

Choose a simple location without spaces in the path. This guide uses:

```text
C:\dev\autonomy_core
```

If the folder does not exist yet:

```powershell
mkdir C:\dev
cd C:\dev
git clone <your-repo-url> autonomy_core
cd C:\dev\autonomy_core
```

Replace `<your-repo-url>` with the URL copied from GitHub's green `Code`
button. For example, it will usually look like an HTTPS URL ending in
`.git`.

If the repo already exists, go there:

```powershell
cd C:\dev\autonomy_core
```

Verify you are in the right folder:

```powershell
dir
```

You should see files such as:

```text
README.md
pyproject.toml
aigp
autonomy_core
docs
```

## 4. Install PyCharm

1. Install PyCharm Community or Professional.

2. Open PyCharm.

3. Select:

   ```text
   Open
   ```

4. Choose:

   ```text
   C:\dev\autonomy_core
   ```

5. Wait for PyCharm to index the project.

## 5. Create The PyCharm Virtual Environment

In PyCharm:

1. Open:

   ```text
   File > Settings > Project: autonomy_core > Python Interpreter
   ```

2. Click:

   ```text
   Add Interpreter
   ```

3. Choose:

   ```text
   Add Local Interpreter
   ```

4. Choose:

   ```text
   Virtualenv Environment
   ```

5. Use these settings:

   ```text
   Environment: New
   Location: C:\dev\autonomy_core\.venv_ctrl
   Base interpreter: Python 3.12
   ```

6. Apply the settings.

If PyCharm cannot find Python 3.12, use the PowerShell check from step 1 and
confirm Python was installed with PATH enabled.

## 6. Install Python Dependencies

Open the PyCharm terminal. It should start in:

```text
C:\dev\autonomy_core
```

If it does not, run:

```powershell
cd C:\dev\autonomy_core
```

Activate the virtual environment:

```powershell
.\.venv_ctrl\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again:

```powershell
.\.venv_ctrl\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install numpy scipy opencv-python pymavlink ultralytics
```

YOLO installs PyTorch through Ultralytics if needed. For real-time GPU use, you
may need a CUDA-compatible PyTorch install that matches the Windows NVIDIA
driver. First get the CPU/import path working, then tune GPU setup.

## 7. Add The YOLO Model

The runtime expects:

```text
C:\dev\autonomy_core\aigp\models\gate_yolo_pose_8k\best.pt
```

Create the folder if needed:

```powershell
mkdir .\aigp\models\gate_yolo_pose_8k
```

Copy the trained model into that folder and name it:

```text
best.pt
```

Verify:

```powershell
dir .\aigp\models\gate_yolo_pose_8k
```

You should see:

```text
best.pt
```

## 8. Edit Runtime Settings For Windows

Open:

```text
aigp\config\runtime.toml
```

For a first Windows smoke test, use these settings:

```toml
[runtime]
runner_mode = "competition"
competition_arm = false

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

Important: keep `competition_arm = false` for the first smoke test. Change it
to `true` only when MAVLink, UDP vision, perception, and command behavior have
all been checked.

## 9. Run Import Smoke Test

In the PyCharm terminal:

```powershell
python -c "import cv2, numpy, scipy, pymavlink, ultralytics, torch; print('ok', torch.cuda.is_available())"
```

Expected result:

```text
ok False
```

or:

```text
ok True
```

`False` means PyTorch is not using CUDA. That is acceptable for a first import
test, but may be too slow for real-time YOLO.

## 10. Run Config Preflight

In the PyCharm terminal:

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

Expected result should look like:

```text
competition False udp competition C:\dev\autonomy_core\aigp\models\gate_yolo_pose_8k\best.pt
```

If `competition_arm` prints `True`, stop and set it to `false` before the first
smoke test.

## 11. Allow Windows Firewall

The stack needs inbound UDP.

Allow:

- MAVLink UDP port `14550`
- Camera UDP port `5600`
- Python through Windows Firewall

When Windows shows a firewall prompt for Python, allow private networks. If the
sender is on another computer or robot network, confirm the Windows network is
classified correctly.

## 12. First Run With Arming Disabled

In the PyCharm terminal:

```powershell
cd C:\dev\autonomy_core
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

This first run should not arm because `competition_arm = false`.

If it prints:

```text
Waiting for heartbeat...
```

then MAVLink is not reaching the process.

Check:

- The sender is targeting the Windows computer IP.
- The sender is using UDP port `14550`.
- `MAVLINK_IP` is `0.0.0.0`.
- Windows Firewall allows inbound UDP `14550`.

If MAVLink works but camera frames do not arrive, check:

- The vision sender is using UDP port `5600`.
- Windows Firewall allows inbound UDP `5600`.
- `VISION_SOURCE` is `udp`.
- The packet format matches `packet_header_format = "<IHHIIQ"`.

## 13. PyCharm Run Configuration

After the terminal run works, create a PyCharm run configuration.

1. Open:

   ```text
   Run > Edit Configurations
   ```

2. Add:

   ```text
   Python
   ```

3. Set:

   ```text
   Name: AIGP Windows Competition
   Script path: C:\dev\autonomy_core\aigp\pilot\main.py
   Working directory: C:\dev\autonomy_core
   Python interpreter: C:\dev\autonomy_core\.venv_ctrl
   ```

4. Add environment variables:

   ```text
   RUNNER_MODE=competition
   VISION_SOURCE=udp
   PERCEPTION_BACKEND=yolo
   PERCEPTION_WORLD_POSE_SOURCE=mavsdk
   CAMERA_MOUNT_PROFILE=competition
   MAVLINK_IP=0.0.0.0
   MAVLINK_PORT=14550
   YOLO_MODEL_PATH=C:\dev\autonomy_core\aigp\models\gate_yolo_pose_8k\best.pt
   ```

5. Save and run.

## 14. Turn On Arming Only When Ready

After the following are confirmed:

- Import smoke test passes.
- Config preflight prints `competition`.
- YOLO model loads.
- MAVLink heartbeat is received.
- Camera frames are received.
- Perception is producing gate detections.
- Command output is understood and expected.

Then change:

```toml
competition_arm = true
```

Do not enable arming during dependency setup or firewall debugging.

## 15. Working With Codex

When asking Codex for help on Windows, include:

- The command you ran.
- The full error output.
- Whether you are in PowerShell, PyCharm terminal, or Git Bash.
- The current folder from:

  ```powershell
  pwd
  ```

- The Python version from:

  ```powershell
  python --version
  ```

- The active interpreter from:

  ```powershell
  python -c "import sys; print(sys.executable)"
  ```

This makes path, virtualenv, and dependency problems much faster to diagnose.

## Common Fixes

### PowerShell Cannot Activate `.venv_ctrl`

Run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then:

```powershell
.\.venv_ctrl\Scripts\Activate.ps1
```

### `ModuleNotFoundError`

Make sure the virtualenv is active, then reinstall:

```powershell
python -m pip install -e .
python -m pip install numpy scipy opencv-python pymavlink ultralytics
```

### YOLO Model Not Found

Use the repo-relative TOML path:

```toml
yolo_model_path = "aigp/models/gate_yolo_pose_8k/best.pt"
```

Then verify the file exists:

```powershell
dir .\aigp\models\gate_yolo_pose_8k\best.pt
```

### `Waiting for heartbeat...`

MAVLink is not reaching Python.

Check:

- Sender target IP.
- UDP port `14550`.
- Windows Firewall.
- `MAVLINK_IP=0.0.0.0`.

### No Camera Frames

Camera UDP is not reaching Python.

Check:

- UDP port `5600`.
- Windows Firewall.
- `VISION_SOURCE=udp`.
- Sender packet format.

### CUDA Is Not Available

If the import smoke test prints:

```text
ok False
```

the stack can still import, but YOLO may run slowly. After the basic stack works,
install the PyTorch CUDA build that matches the Windows NVIDIA driver.
