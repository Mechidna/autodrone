#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_ID="${1:-vq1_openvins_03}"
IMU_SOURCE="${2:-highres_imu}"
MAX_CLONES="${3:-11}"
CAMERA_STRIDE="${4:-1}"
MAX_MSCKF_IN_UPDATE="${5:-40}"
CAMERA_IMU_OFFSET_MS="${6:-0}"
MASK_TOP_ROWS="${7:-0}"
STATIONARY_START="${8:-first}"
IMU_GAP_POLICY="${9:-reject}"
MAX_IMU_GAP_MS="${10:-120}"
INITIALIZATION_MODE="${11:-jerk}"
MSCKF_SELECTION="${12:-legacy}"
MASK_GUIDE_CONE="${13:-0}"
CAMERA_IMAGE_MODE="${14:-grayscale}"
HISTOGRAM_METHOD="${15:-HISTOGRAM}"
NUM_PTS="${16:-300}"
FAST_THRESHOLD="${17:-15}"
MIN_PX_DIST="${18:-10}"
MAX_SLAM="${19:-50}"
MAX_SLAM_IN_UPDATE="${20:-25}"
CAMERA_FOCAL_PX="${21:-native}"
CAMERA_TILT_UP_DEG="${22:-20}"
BUILD_JOBS="${AIGP_OPENVINS_BUILD_JOBS:-2}"
case "${IMU_SOURCE}" in
  highres_imu|scaled_imu|scaled_imu2|scaled_imu3|hil_sensor) ;;
  *)
    echo "ERROR: unsupported IMU source: ${IMU_SOURCE}" >&2
    exit 64
    ;;
esac
if ! [[ "${MAX_CLONES}" =~ ^[0-9]+$ ]] || (( MAX_CLONES < 2 || MAX_CLONES > 100 )); then
  echo "ERROR: max clones must be an integer between 2 and 100: ${MAX_CLONES}" >&2
  exit 64
fi
if ! [[ "${CAMERA_STRIDE}" =~ ^[0-9]+$ ]] || (( CAMERA_STRIDE < 1 || CAMERA_STRIDE > 8 )); then
  echo "ERROR: camera stride must be an integer between 1 and 8: ${CAMERA_STRIDE}" >&2
  exit 64
fi
if ! [[ "${MAX_MSCKF_IN_UPDATE}" =~ ^[0-9]+$ ]] || (( MAX_MSCKF_IN_UPDATE < 1 || MAX_MSCKF_IN_UPDATE > 1000 )); then
  echo "ERROR: max MSCKF features per update must be an integer between 1 and 1000: ${MAX_MSCKF_IN_UPDATE}" >&2
  exit 64
fi
if ! [[ "${CAMERA_IMU_OFFSET_MS}" =~ ^-?[0-9]+$ ]] || (( CAMERA_IMU_OFFSET_MS < -100 || CAMERA_IMU_OFFSET_MS > 100 )); then
  echo "ERROR: camera-to-IMU offset must be an integer from -100 to 100 ms: ${CAMERA_IMU_OFFSET_MS}" >&2
  exit 64
fi
if ! [[ "${MASK_TOP_ROWS}" =~ ^[0-9]+$ ]] || (( MASK_TOP_ROWS < 0 || MASK_TOP_ROWS > 359 )); then
  echo "ERROR: top mask rows must be an integer from 0 to 359: ${MASK_TOP_ROWS}" >&2
  exit 64
fi
case "${STATIONARY_START}" in
  first|auto) ;;
  *)
    echo "ERROR: stationary start must be first or auto: ${STATIONARY_START}" >&2
    exit 64
    ;;
esac
case "${IMU_GAP_POLICY}" in
  reject|allow|interpolate) ;;
  *)
    echo "ERROR: IMU gap policy must be reject, allow, or interpolate: ${IMU_GAP_POLICY}" >&2
    exit 64
    ;;
esac
if ! [[ "${MAX_IMU_GAP_MS}" =~ ^[0-9]+$ ]] || (( MAX_IMU_GAP_MS < 30 || MAX_IMU_GAP_MS > 1000 )); then
  echo "ERROR: max IMU gap must be an integer from 30 to 1000 ms: ${MAX_IMU_GAP_MS}" >&2
  exit 64
fi
case "${INITIALIZATION_MODE}" in
  jerk|stationary) ;;
  *)
    echo "ERROR: initialization mode must be jerk or stationary: ${INITIALIZATION_MODE}" >&2
    exit 64
    ;;
esac
case "${MSCKF_SELECTION}" in
  legacy|geometry) ;;
  *)
    echo "ERROR: MSCKF selection must be legacy or geometry: ${MSCKF_SELECTION}" >&2
    exit 64
    ;;
esac
case "${MASK_GUIDE_CONE}" in
  0|1) ;;
  *)
    echo "ERROR: guide-cone mask must be 0 or 1" >&2
    exit 64
    ;;
esac
case "${CAMERA_IMAGE_MODE}" in
  grayscale|red|red_fixed|red_sidehist) ;;
  *)
    echo "ERROR: camera image mode must be grayscale, red, red_fixed, or red_sidehist: ${CAMERA_IMAGE_MODE}" >&2
    exit 64
    ;;
esac
if [[ "${CAMERA_IMAGE_MODE}" == "red_fixed" && "${HISTOGRAM_METHOD}" != "NONE" ]]; then
  echo "ERROR: red_fixed camera image mode requires histogram method NONE" >&2
  exit 64
fi
if [[ "${CAMERA_IMAGE_MODE}" == "red_sidehist" && "${HISTOGRAM_METHOD}" != "NONE" ]]; then
  echo "ERROR: red_sidehist camera image mode requires histogram method NONE" >&2
  exit 64
fi
if [[ "${CAMERA_IMAGE_MODE}" == "red_sidehist" && ( "${MASK_TOP_ROWS}" != "0" || "${MASK_GUIDE_CONE}" != "0" ) ]]; then
  echo "ERROR: red_sidehist camera image mode requires both spatial masks off" >&2
  exit 64
fi
case "${HISTOGRAM_METHOD}" in
  NONE|HISTOGRAM|CLAHE) ;;
  *)
    echo "ERROR: histogram method must be NONE, HISTOGRAM, or CLAHE: ${HISTOGRAM_METHOD}" >&2
    exit 64
    ;;
esac
if ! [[ "${NUM_PTS}" =~ ^[0-9]+$ ]] || (( NUM_PTS < 1 || NUM_PTS > 5000 )); then
  echo "ERROR: num_pts must be an integer between 1 and 5000: ${NUM_PTS}" >&2
  exit 64
fi
if ! [[ "${FAST_THRESHOLD}" =~ ^[0-9]+$ ]] || (( FAST_THRESHOLD < 1 || FAST_THRESHOLD > 255 )); then
  echo "ERROR: FAST threshold must be an integer between 1 and 255: ${FAST_THRESHOLD}" >&2
  exit 64
fi
if ! [[ "${MIN_PX_DIST}" =~ ^[0-9]+$ ]] || (( MIN_PX_DIST < 1 || MIN_PX_DIST > 100 )); then
  echo "ERROR: minimum pixel distance must be an integer between 1 and 100: ${MIN_PX_DIST}" >&2
  exit 64
fi
if ! [[ "${MAX_SLAM}" =~ ^[0-9]+$ ]] || (( MAX_SLAM < 0 || MAX_SLAM > 500 )); then
  echo "ERROR: max SLAM landmarks must be an integer between 0 and 500: ${MAX_SLAM}" >&2
  exit 64
fi
if ! [[ "${MAX_SLAM_IN_UPDATE}" =~ ^[0-9]+$ ]] || (( MAX_SLAM_IN_UPDATE < 1 || MAX_SLAM_IN_UPDATE > 500 )); then
  echo "ERROR: max SLAM landmarks per update must be an integer between 1 and 500: ${MAX_SLAM_IN_UPDATE}" >&2
  exit 64
fi
if [[ "${CAMERA_FOCAL_PX}" != "native" ]]; then
  if ! [[ "${CAMERA_FOCAL_PX}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "ERROR: camera focal length must be native or a positive number: ${CAMERA_FOCAL_PX}" >&2
    exit 64
  fi
  CAMERA_FOCAL_WHOLE="${CAMERA_FOCAL_PX%%.*}"
  if (( CAMERA_FOCAL_WHOLE < 100 || CAMERA_FOCAL_WHOLE > 1000 )); then
    echo "ERROR: camera focal length must be from 100 to 1000 px: ${CAMERA_FOCAL_PX}" >&2
    exit 64
  fi
fi
if ! [[ "${CAMERA_TILT_UP_DEG}" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
  echo "ERROR: camera tilt must be a number from -89 to 89 degrees: ${CAMERA_TILT_UP_DEG}" >&2
  exit 64
fi
if ! [[ "${BUILD_JOBS}" =~ ^[0-9]+$ ]] || (( BUILD_JOBS < 1 || BUILD_JOBS > 16 )); then
  echo "ERROR: AIGP_OPENVINS_BUILD_JOBS must be an integer between 1 and 16: ${BUILD_JOBS}" >&2
  exit 64
fi
OPENVINS_ROOT="${OPENVINS_ROOT:-${HOME}/src/open_vins}"
OPENVINS_BUILD="${OPENVINS_ROOT}/ov_msckf/build"
RUNNER_BUILD="${AIGP_OPENVINS_BUILD_DIR:-${HOME}/.cache/aigp_openvins_runner}"
DIAGNOSTICS_PATCH="${SCRIPT_DIR}/patches/msckf_rejection_diagnostics.patch"
FAILURE_REASONS_PATCH="${SCRIPT_DIR}/patches/feature_initializer_failure_reasons.patch"
FEATURE_GEOMETRY_PATCH="${SCRIPT_DIR}/patches/msckf_feature_geometry_diagnostics.patch"
FEATURE_IMAGE_PATCH="${SCRIPT_DIR}/patches/msckf_feature_image_diagnostics.patch"
GEOMETRY_SELECTION_PATCH="${SCRIPT_DIR}/patches/msckf_geometry_ranked_selection.patch"
if [[ "${IMU_SOURCE}" == "highres_imu" ]]; then
  REPLAY_NAME="openvins_replay_imuframefix_bracketed"
else
  REPLAY_NAME="openvins_replay_${IMU_SOURCE}_bracketed"
fi
if [[ "${MAX_CLONES}" != "11" ]]; then
  REPLAY_NAME="${REPLAY_NAME}_clones${MAX_CLONES}"
fi
if [[ "${CAMERA_STRIDE}" != "1" ]]; then
  REPLAY_NAME="${REPLAY_NAME}_stride${CAMERA_STRIDE}"
fi
if [[ "${MAX_MSCKF_IN_UPDATE}" != "40" ]]; then
  REPLAY_NAME="${REPLAY_NAME}_msckf${MAX_MSCKF_IN_UPDATE}"
fi
if [[ "${MSCKF_SELECTION}" == "geometry" ]]; then
  REPLAY_NAME="${REPLAY_NAME}_geoselect"
fi
if [[ "${MAX_SLAM}" != "50" ]]; then
  REPLAY_NAME="${REPLAY_NAME}_slam${MAX_SLAM}"
fi
if [[ "${MAX_SLAM_IN_UPDATE}" != "25" ]]; then
  REPLAY_NAME="${REPLAY_NAME}_slamupd${MAX_SLAM_IN_UPDATE}"
fi
if [[ "${NUM_PTS}" != "300" || "${FAST_THRESHOLD}" != "15" || "${MIN_PX_DIST}" != "10" ]]; then
  REPLAY_NAME="${REPLAY_NAME}_pts${NUM_PTS}_fast${FAST_THRESHOLD}_px${MIN_PX_DIST}"
fi
if (( CAMERA_IMU_OFFSET_MS > 0 )); then
  REPLAY_NAME="${REPLAY_NAME}_dtp${CAMERA_IMU_OFFSET_MS}ms"
elif (( CAMERA_IMU_OFFSET_MS < 0 )); then
  REPLAY_NAME="${REPLAY_NAME}_dtm${CAMERA_IMU_OFFSET_MS#-}ms"
fi
if (( MASK_TOP_ROWS > 0 )); then
  REPLAY_NAME="${REPLAY_NAME}_masktop${MASK_TOP_ROWS}"
fi
if [[ "${MASK_GUIDE_CONE}" == "1" ]]; then
  REPLAY_NAME="${REPLAY_NAME}_maskguidecone"
fi
if [[ "${CAMERA_IMAGE_MODE}" == "red" ]]; then
  REPLAY_NAME="${REPLAY_NAME}_redchannel"
elif [[ "${CAMERA_IMAGE_MODE}" == "red_fixed" ]]; then
  REPLAY_NAME="${REPLAY_NAME}_redfixed"
elif [[ "${CAMERA_IMAGE_MODE}" == "red_sidehist" ]]; then
  REPLAY_NAME="${REPLAY_NAME}_redsidehist"
fi
if [[ "${HISTOGRAM_METHOD}" != "HISTOGRAM" ]]; then
  REPLAY_NAME="${REPLAY_NAME}_hist${HISTOGRAM_METHOD,,}"
fi
if [[ "${STATIONARY_START}" == "auto" ]]; then
  REPLAY_NAME="${REPLAY_NAME}_autostart"
fi
if [[ "${INITIALIZATION_MODE}" != "jerk" ]]; then
  REPLAY_NAME="${REPLAY_NAME}_init${INITIALIZATION_MODE}"
fi
if [[ "${IMU_GAP_POLICY}" != "reject" ]]; then
  REPLAY_NAME="${REPLAY_NAME}_gap${IMU_GAP_POLICY}${MAX_IMU_GAP_MS}ms"
fi
if [[ "${CAMERA_FOCAL_PX}" != "native" ]]; then
  FOCAL_SUFFIX="${CAMERA_FOCAL_PX//./p}"
  REPLAY_NAME="${REPLAY_NAME}_fx${FOCAL_SUFFIX}"
fi
if [[ "${CAMERA_TILT_UP_DEG}" != "20" && "${CAMERA_TILT_UP_DEG}" != "20.0" ]]; then
  TILT_SUFFIX="${CAMERA_TILT_UP_DEG//-/m}"
  TILT_SUFFIX="${TILT_SUFFIX//./p}"
  REPLAY_NAME="${REPLAY_NAME}_tilt${TILT_SUFFIX}deg"
fi
REPLAY_DIR="${PROJECT_ROOT}/aigp/logs/runs/${RUN_ID}/${REPLAY_NAME}"

if [[ ! -f "${OPENVINS_ROOT}/ov_msckf/CMakeLists.txt" ]]; then
  echo "ERROR: OpenVINS was not found at ${OPENVINS_ROOT}" >&2
  echo "Set OPENVINS_ROOT to your open_vins checkout." >&2
  exit 2
fi

if grep -q "struct MsckfUpdateDiagnostics" "${OPENVINS_ROOT}/ov_msckf/src/update/UpdaterMSCKF.h"; then
  echo "OpenVINS MSCKF rejection diagnostics patch is already applied."
elif git -C "${OPENVINS_ROOT}" apply --recount --check "${DIAGNOSTICS_PATCH}"; then
  git -C "${OPENVINS_ROOT}" apply --recount "${DIAGNOSTICS_PATCH}"
  echo "Applied OpenVINS MSCKF rejection diagnostics patch."
else
  echo "ERROR: OpenVINS diagnostics patch does not apply cleanly." >&2
  echo "Inspect ${DIAGNOSTICS_PATCH} and the checkout at ${OPENVINS_ROOT}." >&2
  exit 2
fi

if grep -q "triangulation_bad_condition" "${OPENVINS_ROOT}/ov_msckf/src/update/UpdaterMSCKF.h"; then
  echo "OpenVINS feature-initializer failure-reason patch is already applied."
elif git -C "${OPENVINS_ROOT}" apply --recount --check "${FAILURE_REASONS_PATCH}"; then
  git -C "${OPENVINS_ROOT}" apply --recount "${FAILURE_REASONS_PATCH}"
  echo "Applied OpenVINS feature-initializer failure-reason patch."
else
  echo "ERROR: OpenVINS feature-initializer failure-reason patch does not apply cleanly." >&2
  echo "Inspect ${FAILURE_REASONS_PATCH} and the checkout at ${OPENVINS_ROOT}." >&2
  exit 2
fi

if grep -q "MsckfFeatureGeometryDiagnostics" "${OPENVINS_ROOT}/ov_msckf/src/update/UpdaterMSCKF.h"; then
  echo "OpenVINS per-feature geometry diagnostics patch is already applied."
elif git -C "${OPENVINS_ROOT}" apply --recount --check "${FEATURE_GEOMETRY_PATCH}"; then
  git -C "${OPENVINS_ROOT}" apply --recount "${FEATURE_GEOMETRY_PATCH}"
  echo "Applied OpenVINS per-feature geometry diagnostics patch."
else
  echo "ERROR: OpenVINS per-feature geometry diagnostics patch does not apply cleanly." >&2
  echo "Inspect ${FEATURE_GEOMETRY_PATCH} and the checkout at ${OPENVINS_ROOT}." >&2
  exit 2
fi

if grep -q "mean_u_px" "${OPENVINS_ROOT}/ov_msckf/src/update/UpdaterMSCKF.h"; then
  echo "OpenVINS per-feature image diagnostics patch is already applied."
elif git -C "${OPENVINS_ROOT}" apply --recount --check "${FEATURE_IMAGE_PATCH}"; then
  git -C "${OPENVINS_ROOT}" apply --recount "${FEATURE_IMAGE_PATCH}"
  echo "Applied OpenVINS per-feature image diagnostics patch."
else
  echo "ERROR: OpenVINS per-feature image diagnostics patch does not apply cleanly." >&2
  echo "Inspect ${FEATURE_IMAGE_PATCH} and the checkout at ${OPENVINS_ROOT}." >&2
  exit 2
fi

if grep -q "msckf_geometry_ranked_selection" "${OPENVINS_ROOT}/ov_msckf/src/state/StateOptions.h"; then
  echo "OpenVINS geometry-ranked MSCKF selection patch is already applied."
elif git -C "${OPENVINS_ROOT}" apply --recount --check "${GEOMETRY_SELECTION_PATCH}"; then
  git -C "${OPENVINS_ROOT}" apply --recount "${GEOMETRY_SELECTION_PATCH}"
  echo "Applied OpenVINS geometry-ranked MSCKF selection patch."
else
  echo "ERROR: OpenVINS geometry-ranked selection patch does not apply cleanly." >&2
  echo "Inspect ${GEOMETRY_SELECTION_PATCH} and the checkout at ${OPENVINS_ROOT}." >&2
  exit 2
fi

GEOMETRY_SELECTION_ARGS=()
if [[ "${MSCKF_SELECTION}" == "geometry" ]]; then
  GEOMETRY_SELECTION_ARGS+=(--geometry-ranked-msckf)
fi
GUIDE_CONE_ARGS=()
if [[ "${MASK_GUIDE_CONE}" == "1" ]]; then
  GUIDE_CONE_ARGS+=(--mask-guide-cone)
fi
FOCAL_ARGS=()
if [[ "${CAMERA_FOCAL_PX}" != "native" ]]; then
  FOCAL_ARGS+=(--camera-focal-px "${CAMERA_FOCAL_PX}")
fi
TILT_ARGS=(--camera-tilt-up-deg "${CAMERA_TILT_UP_DEG}")

if [[ ! -f "${REPLAY_DIR}/manifest.json" ]]; then
  python3 "${PROJECT_ROOT}/aigp/tools/prepare_openvins_dataset.py" \
    --run "${RUN_ID}" \
    --imu-source "${IMU_SOURCE}" \
    --max-clones "${MAX_CLONES}" \
    --camera-stride "${CAMERA_STRIDE}" \
    --max-msckf-in-update "${MAX_MSCKF_IN_UPDATE}" \
    --max-slam "${MAX_SLAM}" \
    --max-slam-in-update "${MAX_SLAM_IN_UPDATE}" \
    "${GEOMETRY_SELECTION_ARGS[@]}" \
    --num-pts "${NUM_PTS}" \
    --fast-threshold "${FAST_THRESHOLD}" \
    --min-px-dist "${MIN_PX_DIST}" \
    --camera-imu-offset-ms "${CAMERA_IMU_OFFSET_MS}" \
    "${FOCAL_ARGS[@]}" \
    "${TILT_ARGS[@]}" \
    --mask-top-rows "${MASK_TOP_ROWS}" \
    "${GUIDE_CONE_ARGS[@]}" \
    --camera-image-mode "${CAMERA_IMAGE_MODE}" \
    --histogram-method "${HISTOGRAM_METHOD}" \
    --stationary-start "${STATIONARY_START}" \
    --initialization-mode "${INITIALIZATION_MODE}" \
    --imu-gap-policy "${IMU_GAP_POLICY}" \
    --max-imu-gap-ms "${MAX_IMU_GAP_MS}" \
    --output "${REPLAY_DIR}"
fi

if [[ ! -f "${OPENVINS_BUILD}/CMakeCache.txt" ]]; then
  cmake -S "${OPENVINS_ROOT}/ov_msckf" -B "${OPENVINS_BUILD}" -DENABLE_ROS=OFF
fi
cmake --build "${OPENVINS_BUILD}" --parallel "${BUILD_JOBS}"

cmake \
  -S "${SCRIPT_DIR}/runner" \
  -B "${RUNNER_BUILD}" \
  -DOPENVINS_ROOT="${OPENVINS_ROOT}"
cmake --build "${RUNNER_BUILD}" --parallel "${BUILD_JOBS}"

"${RUNNER_BUILD}/run_vq1_dataset" \
  "${REPLAY_DIR}" \
  "${REPLAY_DIR}/openvins_estimate.csv" \
  "${REPLAY_DIR}/openvins_update_diagnostics.csv" \
  "${MASK_TOP_ROWS}" \
  "${MASK_GUIDE_CONE}" \
  "${CAMERA_IMAGE_MODE}"
