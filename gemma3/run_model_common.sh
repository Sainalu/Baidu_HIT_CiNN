#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
MODE=${CINN_MODE:?CINN_MODE must be dynamic or cinn}
TASK=${TASK:-train}
PYTHON_BIN=${PYTHON_BIN:-python}
PADDLEFORMERS_DIR=${PADDLEFORMERS_DIR:-${REPO_DIR}/../PaddleFormers}
PADDLEFORMERS_DIR=$(cd -- "${PADDLEFORMERS_DIR}" && pwd)
EXPECTED_COMMIT=$(tr -d '[:space:]' < "${SCRIPT_DIR}/paddleformers.commit")
ACTUAL_COMMIT=$(git -C "${PADDLEFORMERS_DIR}" rev-parse HEAD)
PATCH_FILE=${SCRIPT_DIR}/patches/paddleformers-gemma3-cinn.patch

if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "ERROR: PaddleFormers commit mismatch" >&2
  echo "expected=${EXPECTED_COMMIT}" >&2
  echo "actual=${ACTUAL_COMMIT}" >&2
  exit 2
fi

if git -C "${PADDLEFORMERS_DIR}" apply --check --reverse "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "PaddleFormers patch is already applied."
elif git -C "${PADDLEFORMERS_DIR}" apply --check "${PATCH_FILE}" >/dev/null 2>&1; then
  git -C "${PADDLEFORMERS_DIR}" apply "${PATCH_FILE}"
  echo "Applied Gemma3 CINN compatibility patch."
else
  echo "ERROR: PaddleFormers tree is neither clean-patchable nor already patched." >&2
  echo "Inspect: git -C ${PADDLEFORMERS_DIR} status --short" >&2
  exit 3
fi

resolve_path() {
  local value=$1
  if [[ "${value}" = /* ]]; then
    printf '%s\n' "${value}"
  else
    printf '%s\n' "${SCRIPT_DIR}/${value}"
  fi
}

if [[ -z "${CHECKPOINT_DIR:-}" ]]; then
  LOCATION_FILE=${SCRIPT_DIR}/checkpoint/LOCATION.local
  if [[ -f "${LOCATION_FILE}" ]]; then
    CHECKPOINT_DIR=$(sed -n 's/^path=//p' "${LOCATION_FILE}" | head -n 1)
  fi
fi
if [[ -z "${CHECKPOINT_DIR:-}" ]]; then
  echo "ERROR: set CHECKPOINT_DIR or add path=<absolute-path> to checkpoint/LOCATION.md" >&2
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

echo "mode=${MODE} task=${TASK} gpu=${GPU_ID}"
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
