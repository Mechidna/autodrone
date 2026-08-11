# Repository Instructions for Codex

## Scope

These instructions apply to the entire repository. The active OpenVINS work is
currently developed on the `openvins` branch.

## Mission and Current State

This repository contains an autonomy stack for drone-racing research using
PX4/Gazebo, MAVLink, YOLO gate perception, trajectory generation, and an
experimental OpenVINS visual-inertial odometry path.

OpenVINS is **not control-ready**. The competition objective was not completed
with OpenVINS, and the current runner must be treated as an instrumented
research integration rather than a working localization product. Preserve that
distinction in code, documentation, commit messages, and portfolio claims.

The checked-in runtime must remain safe by default:

- `runtime.observe_only = true`
- competition arming and PX4 offboard entry disabled
- calibration motion disabled unless explicitly requested for a controlled test
- simulator/ground-truth modes disabled for competition use
- experimental gate-to-VIO alignment disabled

Do not enable flight-control output, arming, offboard entry, or experimental
localization in the default configuration. Changes that can command a vehicle
require an explicit user request and must retain an immediate abort/disarm path.

## Read Before Changing OpenVINS or Gazebo Code

Read these files before proposing or implementing OpenVINS/Gazebo work:

1. `README.md`
2. `aigp/openvins/README.md`
3. `docs/case-studies/openvins-vio/README.md`
4. `aigp/config/runtime.toml`
5. the specific capture, replay, diagnostic, or runtime code being changed

The case study is the authoritative summary of what was tried, what improved,
what failed, and why deployment was rejected. The OpenVINS README is the
authoritative command and output reference.

## Verified Experimental Boundary

The strongest offline result changed only the camera-to-IMU rotation from the
nominal 20-degree upward pitch to 20.8 degrees:

- primary-replay position-error p95: 20.821 m to 4.779 m
- primary-replay final error: 20.578 m to 5.682 m
- three-capture mean p95: 18.812 m to 7.077 m
- three-capture worst p95 at 20.8 degrees: 14.042 m
- first error above the internal 0.5 m target: 8.498 s
- exact prepared replay: five byte-identical executions on one build/machine

This is evidence of parameter sensitivity, not proof that 20.8 degrees is the
physical calibration. Capture 04 performed better at 20 degrees. The production
default remains 20 degrees until a fresh paired experiment justifies changing
it.

The focal-length sweep did not justify changing the native 320 px focal length.
Translation-only gate alignment made the nominal trajectory worse and was
reported as `usable_for_control=false`. Live attempts ended in calibration
failure or divergence-watchdog disarm. Do not describe any of these as a
successful autonomous flight.

## Truth and Evidence Rules

Preserve the estimator/evaluation boundary:

- Camera and IMU measurements may be fed to OpenVINS.
- Captured truth must not be fed to OpenVINS, the gate detector, or a correction
  intended to represent truth-free localization.
- Truth may be used after estimation for one documented startup-frame rigid
  alignment, scoring, and audit-only counterfactuals.
- Never feed a trajectory-wide scale or pose fit back into the estimator and
  then report the result as independent localization.
- Mark any simulator truth, known gate map, or debug-only source prominently.
- State the dataset count, controlled variable, fixed variables, metrics, and
  acceptance threshold for every experiment.
- Do not generalize a single replay or a single machine's determinism result.

Prefer a negative result with a clear boundary over an unsupported success
claim.

## Reproducible Python Environment

Python 3.12 is required for development and runtime use. CI verifies Python
3.12 on Linux and Windows; earlier Python versions are unsupported. Ubuntu
24.04 is the primary Linux target.

From the repository root:

```bash
python3.12 -m venv .venv_ctrl
source .venv_ctrl/bin/activate
python -m pip install --require-hashes -r requirements/test.lock
python -m pip install --no-build-isolation --no-deps -e .
python -m pip check
python -m pytest -q
```

The locked environment covers repository development and unit tests. It does
not include Ultralytics or a hardware-specific PyTorch/CUDA build. Install that
hardware layer separately when live YOLO inference is required.

The unit suite does not require Gazebo, PX4, flight hardware, or raw replay
logs. Run the relevant tests after each change and the complete suite before
calling a cross-cutting change complete.

## OpenVINS Dependency and Patches

OpenVINS is an external checkout, not a submodule. The known local integration
was based on:

```text
repository: https://github.com/rpng/open_vins.git
base revision: 69488123ed9362dd44b6f28e7f4680abbff1442b
default checkout: ~/src/open_vins
```

For a reproducible clean setup, check out that base revision and let
`aigp/openvins/build_and_run_vq1.sh` apply the repository patches. Do not assume
the patches apply to newer upstream revisions. If intentionally upgrading
OpenVINS, make that a separate change, record the new revision, refresh patches
as needed, and rerun deterministic and accuracy audits.

The build script derives this repository's root from its own path and should
work on native Linux. Examples containing `/mnt/c/dev/autonomy_core` are WSL
examples; on native Ubuntu, run from the actual clone path.

The replay build requires a prepared dataset under `aigp/logs/runs`. Those raw
datasets are intentionally excluded from Git. A fresh clone can build and run
unit tests but cannot reproduce an old replay unless the dataset is copied
separately or regenerated.

## PX4/Gazebo Boundary

PX4, Gazebo, ROS 2, and the custom race world are external to this repository.
The expected development target is Ubuntu 24.04, ROS 2 Jazzy, and Gazebo Sim
8.x. The commonly referenced world is `gate_test_1500mm_blue_random`, generated
from `gate_test_1500mm_blue.sdf`; neither world file is currently versioned
here.

Before claiming a fresh-clone Gazebo workflow works:

1. record the PX4 repository URL and exact revision;
2. document how the base race world/model assets are obtained;
3. remove or parameterize machine-specific paths;
4. verify ROS/Gazebo topics and MAVLink endpoints from a clean Ubuntu checkout;
5. run observe-only capture before allowing any control output.

Known portability debt:

- `aigp/tools/randomize_gate_world.py` hard-codes
  `/home/paolo/PX4-Autopilot/PX4-Autopilot/Tools/simulation/gz/worlds`.
- `aigp/tools/capture_gazebo_yolo_pose.py` uses the same machine-specific root
  for its default world SDF.

Do not add a new absolute home-directory path. Prefer an explicit CLI option,
then an environment variable, then a documented relative/default lookup.

## Valid Next OpenVINS Work

Prioritize work in this order unless the user requests a different bounded
task:

1. Make the native Ubuntu/PX4/Gazebo setup portable and record dependency pins.
2. Verify the locked install, unit suite, OpenVINS build, and observe-only data
   capture from a clean checkout.
3. Generate a fresh synchronized capture and replay it as a paired 20-versus-
   20.8-degree experiment with every other variable fixed.
4. Diagnose time-varying direction, bias, timing, and extrinsic errors; do not
   apply a single scale correction as a substitute.
5. Require success on unseen captures before any observe-only live validation.
6. Consider connection to control only after predeclared accuracy thresholds
   and live safety checks pass.

Do not spend effort tuning the flight controller around a state estimate that
has already failed the localization acceptance threshold.

## Repository Hygiene

- Inspect `git status` before editing and preserve unrelated user changes.
- Do not commit `aigp/logs`, replay directories, camera frames, build trees,
  virtual environments, `__pycache__`, `.pyc` files, or machine-specific paths.
- The case-study CSV files are small curated evidence and are intentionally
  tracked despite the general `*.csv` ignore rule.
- Existing model weights are tracked. Discuss Git LFS or release artifacts
  before adding new large binaries.
- Do not add malformed directories whose names encode Windows paths, including
  paths resembling `C...Users...`.
- Keep configuration paths repository-relative or configurable.
- Use separate output directories for controlled OpenVINS experiments so an
  existing baseline is never overwritten.
- Keep generated manifests with experimental outputs and record all non-default
  parameters.

## Implementation and Validation Expectations

- Prefer the smallest change that tests one hypothesis.
- Add or update tests for behavior changes that can be exercised without the
  simulator.
- For sensor-frame, timing, calibration, or estimator changes, provide a
  controlled before/after audit rather than judging a trajectory visually.
- Prove one prepared replay is deterministic before comparing close parameter
  variants.
- Treat initialization, continued state publication, and accepted visual
  updates as necessary but insufficient; control readiness depends on bounded
  state error across captures.
- Update the relevant README when commands, assumptions, outputs, or safety
  behavior change.
- Run `git diff --check` before handing off work.
- Report exactly which tests and experiments ran, and distinguish tests not run
  from tests that failed.

## Definition of Done

An OpenVINS/Gazebo change is complete only when:

- the intended behavior is implemented without weakening safe defaults;
- relevant unit tests pass;
- installation/build/run instructions match a clean Ubuntu checkout;
- dependency and external-repository revisions are recorded;
- evidence preserves truth separation and controlled-variable discipline;
- new generated data remains outside Git unless it is a deliberately curated,
  small review artifact; and
- limitations and remaining control blockers are stated explicitly.
