#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
MODEL_NAME=$(basename -- "${SCRIPT_DIR}")
MODE=${CINN_MODE:?CINN_MODE must be dynamic or cinn}
TASK=${TASK:-train}
PYTHON_BIN=${PYTHON_BIN:-python}
PADDLEFORMERS_DIR=${PADDLEFORMERS_DIR:-${REPO_DIR}/../PaddleFormers}
PADDLEFORMERS_DIR=$(cd -- "${PADDLEFORMERS_DIR}" && pwd)
COMMIT_FILE=${SCRIPT_DIR}/paddleformers.commit
if [[ ! -f "${COMMIT_FILE}" ]]; then
  echo "ERROR: missing ${COMMIT_FILE}" >&2
  exit 2
fi
EXPECTED_COMMIT=$(tr -d '[:space:]' < "${COMMIT_FILE}")
ACTUAL_COMMIT=$(git -C "${PADDLEFORMERS_DIR}" rev-parse HEAD)

# A model may provide one patch, or select one explicitly with PATCH_FILE.
PATCH_FILE=${PATCH_FILE:-}
if [[ -z "${PATCH_FILE}" ]]; then
  PATCH_CANDIDATES=()
  for candidate in "${SCRIPT_DIR}/patches"/*.patch; do
    [[ -f "${candidate}" ]] && PATCH_CANDIDATES+=("${candidate}")
  done
  if [[ "${#PATCH_CANDIDATES[@]}" -gt 1 ]]; then
    echo "ERROR: multiple patches found; set PATCH_FILE explicitly" >&2
    printf '  %s\n' "${PATCH_CANDIDATES[@]}" >&2
    exit 3
  elif [[ "${#PATCH_CANDIDATES[@]}" -eq 1 ]]; then
    PATCH_FILE=${PATCH_CANDIDATES[0]}
  fi
elif [[ "${PATCH_FILE}" != /* ]]; then
  PATCH_FILE=${SCRIPT_DIR}/${PATCH_FILE}
fi

if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "ERROR: PaddleFormers commit mismatch" >&2
  echo "expected=${EXPECTED_COMMIT}" >&2
  echo "actual=${ACTUAL_COMMIT}" >&2
  exit 2
fi

if [[ -n "${PATCH_FILE}" ]]; then
  if git -C "${PADDLEFORMERS_DIR}" apply --check --reverse "${PATCH_FILE}" >/dev/null 2>&1; then
    echo "PaddleFormers patch is already applied (${MODEL_NAME})."
  elif git -C "${PADDLEFORMERS_DIR}" apply --check "${PATCH_FILE}" >/dev/null 2>&1; then
    git -C "${PADDLEFORMERS_DIR}" apply "${PATCH_FILE}"
    echo "Applied ${MODEL_NAME} PaddleFormers compatibility patch."
  else
    echo "ERROR: PaddleFormers tree is neither clean-patchable nor already patched." >&2
    echo "Inspect: git -C ${PADDLEFORMERS_DIR} status --short" >&2
    exit 3
  fi
else
  echo "No PaddleFormers compatibility patch configured for ${MODEL_NAME}."
fi

resolve_path() {
  local value=$1
  if [[ "${value}" = /* ]]; then
    printf '%s\n' "${value}"
  else
    case "${value}" in
      "${MODEL_NAME}"/*) printf '%s\n' "${REPO_DIR}/${value}" ;;
      *) printf '%s\n' "${SCRIPT_DIR}/${value}" ;;
    esac
  fi
}

if [[ -z "${CHECKPOINT_DIR:-}" ]]; then
  LOCATION_FILE=${SCRIPT_DIR}/checkpoint/LOCATION.local
  if [[ -f "${LOCATION_FILE}" ]]; then
    CHECKPOINT_DIR=$(sed -n 's/^path=//p' "${LOCATION_FILE}" | head -n 1)
  fi
fi
if [[ -z "${CHECKPOINT_DIR:-}" ]]; then
  echo "ERROR: set CHECKPOINT_DIR or add path=<absolute-path> to checkpoint/LOCATION.local" >&2
  exit 4
fi
CHECKPOINT_DIR=$(resolve_path "${CHECKPOINT_DIR}")

if [[ -z "${DATA_DIR:-}" ]]; then
  if [[ "${TASK}" == "infer" ]]; then
    DATA_DIR=${SCRIPT_DIR}/data/eval
  else
    DATA_DIR=${SCRIPT_DIR}/data/train
  fi
fi
DATA_DIR=$(resolve_path "${DATA_DIR}")
GPU_ID=${GPU_ID:-0}
STEPS=${STEPS:-20}
WARMUP_STEPS=${WARMUP_STEPS:-2}
LEARNING_RATE=${LEARNING_RATE:-0}
OUTPUT=${OUTPUT:-${SCRIPT_DIR}/results/${TASK}-${MODE}.json}
LOG=${LOG:-${SCRIPT_DIR}/logs/${TASK}-${MODE}.log}
OUTPUT=$(resolve_path "${OUTPUT}")
LOG=$(resolve_path "${LOG}")
mkdir -p "$(dirname -- "${OUTPUT}")" "$(dirname -- "${LOG}")"

COMMAND=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_model.py"
  --mode "${MODE}"
  --task "${TASK}"
  --checkpoint "${CHECKPOINT_DIR}"
  --data "${DATA_DIR}"
  --steps "${STEPS}"
  --warmup-steps "${WARMUP_STEPS}"
  --learning-rate "${LEARNING_RATE}"
  --device gpu:0
  --output "${OUTPUT}"
)

echo "model=${MODEL_NAME} mode=${MODE} task=${TASK} gpu=${GPU_ID}"
echo "paddleformers=${PADDLEFORMERS_DIR}@${ACTUAL_COMMIT}"
echo "checkpoint=${CHECKPOINT_DIR}"
echo "data=${DATA_DIR}"
CUDA_VISIBLE_DEVICES=${GPU_ID} PYTHONPATH="${PADDLEFORMERS_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${COMMAND[@]}" 2>&1 | tee "${LOG}"

if [[ "${MODE}" == "cinn" ]]; then
  COMPILE_COUNT=$(grep -c 'Compiling subgraph with CINN backend' "${LOG}" || true)
  if [[ "${COMPILE_COUNT}" -lt 1 ]]; then
    echo "ERROR: CINN compile marker was not found in ${LOG}" >&2
    exit 5
  fi
  echo "cinn_compile_count=${COMPILE_COUNT}"
fi
