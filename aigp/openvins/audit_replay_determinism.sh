#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 3 )); then
  echo "usage: audit_replay_determinism.sh REPLAY_DIRECTORY [RUN_COUNT] [AUDIT_LABEL]" >&2
  exit 64
fi

REPLAY_DIR="$(realpath "$1")"
RUN_COUNT="${2:-5}"
AUDIT_LABEL="${3:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUNNER="${AIGP_OPENVINS_RUNNER:-${HOME}/.cache/aigp_openvins_runner/run_vq1_dataset}"

if ! [[ "${RUN_COUNT}" =~ ^[0-9]+$ ]] || (( RUN_COUNT < 2 || RUN_COUNT > 20 )); then
  echo "ERROR: run count must be an integer between 2 and 20: ${RUN_COUNT}" >&2
  exit 64
fi
if [[ ! -d "${REPLAY_DIR}" ]]; then
  echo "ERROR: replay directory does not exist: ${REPLAY_DIR}" >&2
  exit 2
fi
if [[ ! -x "${RUNNER}" ]]; then
  echo "ERROR: OpenVINS replay runner is missing or not executable: ${RUNNER}" >&2
  exit 2
fi
for required in imu.csv cam0/data.csv config/estimator_config.yaml manifest.json; do
  if [[ ! -f "${REPLAY_DIR}/${required}" ]]; then
    echo "ERROR: replay input is missing: ${REPLAY_DIR}/${required}" >&2
    exit 2
  fi
done

readarray -t REPLAY_POLICY < <(
  python3 - "${REPLAY_DIR}/manifest.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle)
policy = manifest["estimator_policy"]
print(int(policy["mask_top_rows"]))
print(1 if policy["mask_guide_cone"] else 0)
print(policy["camera_image_mode"])
PY
)
MASK_TOP_ROWS="${REPLAY_POLICY[0]}"
MASK_GUIDE_CONE="${REPLAY_POLICY[1]}"
CAMERA_IMAGE_MODE="${REPLAY_POLICY[2]}"

RUN_PARENT="$(dirname "${REPLAY_DIR}")"
REPLAY_NAME="$(basename "${REPLAY_DIR}")"
AUDIT_DIR="${RUN_PARENT}/determinism_audits/${REPLAY_NAME}_${RUN_COUNT}x_${AUDIT_LABEL}"
if [[ -e "${AUDIT_DIR}" ]]; then
  echo "ERROR: refusing to overwrite existing audit directory: ${AUDIT_DIR}" >&2
  exit 2
fi
mkdir -p "${AUDIT_DIR}"

INPUT_HASHES="${AUDIT_DIR}/input_hashes.txt"
sha256sum \
  "${REPLAY_DIR}/imu.csv" \
  "${REPLAY_DIR}/cam0/data.csv" \
  "${REPLAY_DIR}/config/estimator_config.yaml" \
  "${REPLAY_DIR}/config/kalibr_imucam_chain.yaml" \
  > "${INPUT_HASHES}"

FILES=(
  openvins_estimate.csv
  openvins_update_diagnostics.csv
  openvins_feature_geometry.csv
  openvins_slam_feature_diagnostics.csv
)

for (( index = 1; index <= RUN_COUNT; ++index )); do
  printf -v LABEL "replay_%02d" "${index}"
  OUTPUT_DIR="${AUDIT_DIR}/${LABEL}"
  ISOLATED_REPLAY="${OUTPUT_DIR}/replay"
  mkdir -p "${ISOLATED_REPLAY}"

  # Hard-link immutable replay inputs so each execution sees identical bytes
  # without duplicating the camera dataset. OpenVINS outputs are newly created
  # inside each isolated directory and cannot overwrite the source replay.
  cp -al "${REPLAY_DIR}/imu.csv" "${ISOLATED_REPLAY}/imu.csv"
  cp -al "${REPLAY_DIR}/cam0" "${ISOLATED_REPLAY}/cam0"
  cp -al "${REPLAY_DIR}/config" "${ISOLATED_REPLAY}/config"

  echo "START ${LABEL}"
  "${RUNNER}" \
    "${ISOLATED_REPLAY}" \
    "${ISOLATED_REPLAY}/openvins_estimate.csv" \
    "${ISOLATED_REPLAY}/openvins_update_diagnostics.csv" \
    "${MASK_TOP_ROWS}" \
    "${MASK_GUIDE_CONE}" \
    "${CAMERA_IMAGE_MODE}" \
    > "${OUTPUT_DIR}/runner.log" 2>&1

  sha256sum "${FILES[@]/#/${ISOLATED_REPLAY}/}" \
    > "${OUTPUT_DIR}/sha256.txt"
  wc -l "${FILES[@]/#/${ISOLATED_REPLAY}/}" \
    > "${OUTPUT_DIR}/row_counts.txt"
  echo "DONE ${LABEL}"
done

STATUS="pass"
SUMMARY="${AUDIT_DIR}/comparison.txt"
: > "${SUMMARY}"
for filename in "${FILES[@]}"; do
  reference="$(sha256sum "${AUDIT_DIR}/replay_01/replay/${filename}" | cut -d' ' -f1)"
  identical=1
  for (( index = 2; index <= RUN_COUNT; ++index )); do
    printf -v LABEL "replay_%02d" "${index}"
    candidate="$(sha256sum "${AUDIT_DIR}/${LABEL}/replay/${filename}" | cut -d' ' -f1)"
    if [[ "${candidate}" != "${reference}" ]]; then
      identical=0
      STATUS="fail"
    fi
  done
  printf '%s sha256=%s identical=%s\n' \
    "${filename}" "${reference}" "${identical}" | tee -a "${SUMMARY}"
done

printf 'status=%s\nruns=%s\nreplay=%s\naudit_dir=%s\n' \
  "${STATUS}" "${RUN_COUNT}" "${REPLAY_DIR}" "${AUDIT_DIR}" \
  | tee -a "${SUMMARY}"

if [[ "${STATUS}" != "pass" ]]; then
  exit 1
fi
