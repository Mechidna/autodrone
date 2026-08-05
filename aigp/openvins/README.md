# VQ1 OpenVINS offline replay

This path validates an `aigp_vio_dataset`, aligns the camera and IMU clocks,
trims images to the IMU interval, materializes a replay directory, and feeds it
to OpenVINS without ROS. A VIO capture requests and records `HIGHRES_IMU`,
`HIL_SENSOR`, `SCALED_IMU`, `SCALED_IMU2`, `SCALED_IMU3`, and diagnostic-only
`RAW_IMU` in parallel. Normal flight runs request only `HIGHRES_IMU`.

## Capture an IMU comparison run

From PowerShell:

```powershell
python .\aigp\tools\run_with_log.py `
  --capture-vio-dataset `
  --vio-dataset-queue-size 2048 `
  --run-id vq1_openvins_05

$manifest = Get-Content .\aigp\logs\runs\vq1_openvins_05\vio_dataset\manifest.json | ConvertFrom-Json
$manifest.counts.written
$manifest.counts.dropped_queue_full
```

For a simulator with no attitude, local-position, or odometry telemetry, make
the first capture without any flight-control transmissions:

```powershell
python .\aigp\tools\run_with_log.py `
  --observe-only `
  --capture-vio-dataset `
  --vio-dataset-queue-size 2048 `
  --run-id vq2_vq1_observe_01
```

Stop it with Ctrl+C after 20-30 seconds. This capture inventories camera and
IMU availability safely; the current replay preparation remains
truth-dependent and is a separate next step if the sensor inventory passes.

Startup reports the configured flight-control target separately from the
heartbeat-derived telemetry request target. In the VQ1 simulator these should
normally read `control_target=1:1 telemetry_request_target=1:200`, and each
message-interval request should show `target=1:200`.

The capture command temporarily requests all candidate streams at 120 Hz. A
source is usable only when its written count is nonzero and the dropped and
write-failure counts are zero. Try `hil_sensor` first when present, then the
available `scaled_imu*` streams. `RAW_IMU` is retained for inspection but is
not replayable because MAVLink defines its scale as device-specific.

Prepare one candidate at a time. For example:

```powershell
.\.venv_ctrl\Scripts\python.exe aigp\tools\prepare_openvins_dataset.py `
  --run vq1_openvins_05 `
  --imu-source hil_sensor
Get-Content aigp\logs\runs\vq1_openvins_05\openvins_replay_hil_sensor_bracketed\manifest.json
```

Then open Ubuntu/WSL and run that source:

```bash
cd /mnt/c/dev/autonomy_core
bash aigp/openvins/build_and_run_vq1.sh vq1_openvins_05 hil_sensor
```

Back in PowerShell, check all six inertial axes and the visual updates:

```powershell
.\.venv_ctrl\Scripts\python.exe aigp\tools\diagnose_openvins_replay.py `
  --run vq1_openvins_05 `
  --imu-source hil_sensor
```

Replace `hil_sensor` consistently with `scaled_imu`, `scaled_imu2`, or
`scaled_imu3` to evaluate another captured source. The source-specific replay
directories allow the results to coexist.

## Existing HIGHRES_IMU replay

From PowerShell, prepare and inspect the replay data:

```powershell
.\.venv_ctrl\Scripts\python.exe aigp\tools\prepare_openvins_dataset.py --run vq1_openvins_03
Get-Content aigp\logs\runs\vq1_openvins_03\openvins_replay_imuframefix_bracketed\manifest.json
```

Then open Ubuntu/WSL and run:

```bash
cd /mnt/c/dev/autonomy_core
bash aigp/openvins/build_and_run_vq1.sh vq1_openvins_03
```

If OpenVINS is not at `~/src/open_vins`, set it explicitly:

```bash
OPENVINS_ROOT=/path/to/open_vins \
  bash aigp/openvins/build_and_run_vq1.sh vq1_openvins_03
```

The runner writes both
`openvins_replay_imuframefix_bracketed/openvins_estimate.csv` and
`openvins_replay_imuframefix_bracketed/openvins_update_diagnostics.csv`. The
second file records the complete MSCKF rejection funnel: candidates, input
measurements, too-short tracks, triangulation failures, nonlinear-refinement
failures, their initializer-level reasons, chi-square tests/rejections,
accepted features, active tracks, SLAM features, and state continuity for each
camera update. The runner also writes `openvins_feature_geometry.csv`, with one
row per feature that reaches triangulation: observation count, track duration,
maximum angular parallax, maximum camera baseline, linear-system condition
number, linear/refined depth, chi-square ratio, and final rejection stage. The
build script applies all diagnostics patches under `patches/` idempotently to
the WSL OpenVINS checkout and rebuilds the affected library.
The runner exits with a nonzero status if configuration parsing, image
decoding, or estimator initialization fails.

To run a controlled 20-clone experiment without overwriting the 11-clone
baseline, pass the clone count as the third argument:

```bash
cd /mnt/c/dev/autonomy_core
bash aigp/openvins/build_and_run_vq1.sh vq1_openvins_03 highres_imu 20
```

This writes to
`openvins_replay_imuframefix_bracketed_clones20/`.

To test a longer temporal baseline with the same full-rate IMU, retain every
second camera frame (about 15 Hz) while keeping the default 11 clones:

```bash
cd /mnt/c/dev/autonomy_core
bash aigp/openvins/build_and_run_vq1.sh vq1_openvins_03 highres_imu 11 2
```

This preserves the 30 Hz baseline and writes to
`openvins_replay_imuframefix_bracketed_stride2/`.

To exclude the animated central guide-cone overlay from initialization and
tracking, pass `1` as the thirteenth argument. All preceding arguments remain
explicit so this writes a separate controlled replay:

```bash
bash aigp/openvins/build_and_run_vq1.sh \
  vq1_openvins_yexcite_noperception_04 highres_imu 30 2 10 0 0 \
  auto allow 120 stationary legacy 1
```

This writes to the corresponding `_maskguidecone_...` replay directory. The
resolution-scaled polygon and its first-frame activation are recorded in that
replay's manifest.

To suppress the animated cyan guide without removing its entire geometric
envelope, pass `red` as the fourteenth argument and leave both spatial masks
at zero. The runner decodes each recorded color JPEG and supplies its red
channel to OpenVINS as the monochrome image:

```bash
bash aigp/openvins/build_and_run_vq1.sh \
  vq1_openvins_yexcite_noperception_04 highres_imu 30 2 10 0 0 \
  auto allow 120 stationary legacy 0 red
```

This controlled replay writes to the corresponding `_redchannel_...`
directory. Normal grayscale remains the default, and the replay manifest
records the selected camera-image mode.

To keep the red-channel suppression from being re-amplified by global
histogram equalization, pass `NONE` as the fifteenth argument while leaving
both spatial masks at zero:

```bash
bash aigp/openvins/build_and_run_vq1.sh \
  vq1_openvins_yexcite_noperception_04 highres_imu 30 2 10 0 0 \
  auto allow 120 stationary legacy 0 red NONE
```

This writes a separate `_redchannel_histnone_...` replay. `HISTOGRAM` remains
the default for existing commands; `CLAHE` is also accepted for controlled
comparison.

The sixteenth through eighteenth arguments control KLT feature density:
`num_pts`, `fast_threshold`, and `min_px_dist`. For example, this preserves the
30-clone geometry-ranked baseline while testing a denser detector:

```bash
bash aigp/openvins/build_and_run_vq1.sh \
  vq1_openvins_yexcite_noperception_04 highres_imu 30 2 10 0 0 \
  auto allow 120 stationary geometry 0 red HISTOGRAM 500 10 7
```

The defaults remain `300`, `15`, and `10`. Non-default values are included in
the replay directory name and recorded in its manifest.

The nineteenth argument controls the maximum number of persistent SLAM
landmarks. Set it to `0` for a pure-MSCKF replay while leaving the default at
`50` for existing commands:

```bash
bash aigp/openvins/build_and_run_vq1.sh \
  vq1_openvins_yexcite_noperception_04 highres_imu 30 2 10 0 0 \
  auto allow 120 stationary geometry 0 red HISTOGRAM 500 15 7 0
```

Non-default values are recorded in the manifest and add `_slam0` (or the
selected limit) to the replay directory name.

The twentieth argument controls how many persistent landmarks may be used in
one SLAM update. The default is `25`; this example raises both the landmark
capacity and per-update limit:

```bash
bash aigp/openvins/build_and_run_vq1.sh \
  vq1_openvins_yexcite_noperception_04 highres_imu 30 2 10 0 0 \
  auto allow 120 stationary geometry 0 red HISTOGRAM 500 15 7 100 50
```

A non-default update limit adds `_slamupd50` (or the selected limit) to the
replay directory name and manifest.

The twenty-first argument is an experimental focal-length override in pixels.
It sets both `fx` and `fy` while preserving the recorded principal point,
distortion, images, IMU, timing, and all estimator settings. Use `native` (the
default) to retain the captured calibration:

```bash
bash aigp/openvins/build_and_run_vq1.sh \
  vq1_openvins_yexcite_noperception_06 highres_imu 30 2 10 0 0 \
  auto allow 120 stationary geometry 0 red HISTOGRAM 500 15 7 100 25 325
```

The override is recorded in the replay manifest and adds `_fx325` (or the
selected value) to the replay directory name.

The twenty-second argument is an experimental camera-to-IMU mounting-pitch
override in degrees. It changes only the rotation in `T_imu_cam`; native
`fx=fy=320`, the zero lever arm, images, IMU, timing, and estimator settings
remain fixed. The default is the nominal 20-degree upward tilt:

```bash
bash aigp/openvins/build_and_run_vq1.sh \
  vq1_openvins_yexcite_noperception_06 highres_imu 30 2 10 0 0 \
  auto allow 120 stationary geometry 0 red HISTOGRAM 500 15 7 100 25 \
  native 20.8
```

A non-default angle is recorded in the replay manifest and adds a suffix such
as `_tilt20p8deg`. Keep this override experimental: the run-06 sweep found a
sharp local optimum at 20.8 degrees, but its 4.78 m p95 error remains well
above the 0.5 m control-readiness target. The complete controlled sweep and
cross-run comparison are in
`aigp/logs/runs/vq1_openvins_yexcite_noperception_06/openvins_camera_tilt_sweep.md`.

Prove that one prepared replay is software-deterministic before comparing
calibrations. This runs isolated hard-linked copies, preserves every output,
and requires byte-identical SHA-256 hashes for the estimate, update funnel,
feature geometry, and persistent-SLAM diagnostics:

```bash
bash aigp/openvins/audit_replay_determinism.sh \
  /mnt/c/dev/autonomy_core/aigp/logs/runs/RUN_ID/REPLAY_DIRECTORY \
  5 audit_label
```

Audit outputs are written beside the replay under `determinism_audits/`; the
source replay and its existing estimates are not overwritten.

For a temporally fixed contrast enhancement, use `red_fixed` with OpenVINS
histogram processing disabled:

```bash
bash aigp/openvins/build_and_run_vq1.sh \
  vq1_openvins_yexcite_noperception_04 highres_imu 30 2 10 0 0 \
  auto allow 120 stationary legacy 0 red_fixed NONE
```

The runner collects one red-channel histogram over the first two replay
seconds, computes a bounded gamma LUT that maps its median toward 48/255, and
then applies that exact LUT to every frame. It never adapts the LUT after
startup. The resulting `_redfixed_histnone_...` directory remains separate
from the raw-red and per-frame-histogram experiments. `red_fixed` rejects any
OpenVINS histogram method other than `NONE` to prevent double normalization.

To adapt contrast per frame without allowing the animated raceline to control
the equalization statistics, use `red_sidehist`:

```bash
bash aigp/openvins/build_and_run_vq1.sh \
  vq1_openvins_yexcite_noperception_04 highres_imu 30 2 10 0 0 \
  auto allow 120 stationary legacy 0 red_sidehist NONE
```

For each frame, the runner computes the histogram from the pixels outside the
audited guide-cone polygon and applies the resulting LUT to the complete red
image. The feature mask remains empty, so features inside the guide region are
still tracked. This mode requires both spatial masks off and OpenVINS
histogram processing `NONE`, and writes a separate
`_redsidehist_histnone_...` replay.

Back in PowerShell, produce the sensor/update report:

```powershell
.\.venv_ctrl\Scripts\python.exe aigp\tools\diagnose_openvins_replay.py `
  --run vq1_openvins_03
Get-Content aigp\logs\runs\vq1_openvins_03\openvins_replay_imuframefix_bracketed\openvins_diagnostic_report.json
```

Audit camera/IMU rotational calibration independently of OpenVINS:

```powershell
.\.venv_ctrl\Scripts\python.exe aigp\tools\audit_openvins_camera_imu.py `
  --run vq1_openvins_03
Get-Content aigp\logs\runs\vq1_openvins_03\openvins_replay_imuframefix_bracketed\camera_imu_audit_report.json
```

The audit tracks image features and integrates the corrected body-FRD gyro
between camera timestamps. It searches original versus horizontally flipped
images, camera tilt, and camera/IMU time offset. Each image pair fits its own
translation direction before calculating a Sampson epipolar residual, so
forward motion is not incorrectly counted as rotational error. Lower scores
are better. `imu_time_offset_ms` uses
`imu_time = camera_time + offset`. The audit is read-only with respect to the
replay, does not use ground truth, and estimates rotational extrinsics rather
than camera translation.

The sensor audit uses captured `LOCAL_POSITION_NED` translation and the
empirically verified physical ATTITUDE boundary `(+roll, -pitch,
competition-adjusted yaw)`. Raw ODOMETRY orientation/velocity are excluded from
this audit because their competition-frame conventions are internally paired
but do not describe the physical IMU attitude. Truth remains evaluation-only
and is never fed to OpenVINS.

The VQ1 competition stream reports all three HIGHRES_IMU gyro axes opposite
the physical body angular rates verified by ATTITUDE, local motion, and the
controlled roll excitation. Conversion therefore writes OpenVINS angular
velocity as `(-xgyro, -ygyro, -zgyro)` and records the transform in the
generated manifest. Accelerometer axes are unchanged. The previous
pitch-only replay and earlier failed baselines are preserved.

Every camera update is fed only after OpenVINS has received the first IMU
sample strictly newer than that camera timestamp. This brackets interpolation
and prevents last-sample extrapolation. Online camera/IMU time-offset
calibration is disabled because conversion has already aligned both clocks.

Calibration assumptions are explicit in the generated manifest and YAML:

- camera: 640x360 pinhole, `fx=fy=320`, principal point `(320, 180)`, zero distortion;
- camera optical to IMU/body FRD: competition camera pitched 20 degrees upward;
- camera and IMU/body origins: coincident;
- feature triangulation maximum distance: 200 m (`fi_max_dist`);
- IMU: selected source converted to acceleration in m/s^2 and angular rate in
  rad/s; the generated manifest records its source, units, and axis transform.
