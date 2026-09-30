#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
MODEL_NAME=$(basename -- "${SCRIPT_DIR}")

# A mode is fixed by run_model.sh/run_model_cinn.sh, or selected explicitly by
# MODE when train.sh/infer.sh is used as a formal single-task entry point.
MODE=${CINN_MODE:-${MODE:-dynamic}}
case "${MODE}" in
  dynamic|cinn) ;;
  *)
    echo "ERROR: MODE must be dynamic or cinn (got ${MODE})" >&2
    exit 2
    ;;
esac

TASK=${TASK:-train}
case "${TASK}" in
  train|infer|all) ;;
  *)
    echo "ERROR: TASK must be train, infer, or all (got ${TASK})" >&2
    exit 2
    ;;
esac
PYTHON_BIN=${PYTHON_BIN:-python}
PADDLEFORMERS_DIR=${PADDLEFORMERS_DIR:-${REPO_DIR}/../PaddleFormers}
PADDLEFORMERS_DIR=$(cd -- "${PADDLEFORMERS_DIR}" && pwd)
COMMIT_FILE=${SCRIPT_DIR}/paddleformers.commit
if [[ ! -f "${COMMIT_FILE}" ]]; then
  echo "ERROR: missing ${COMMIT_FILE}" >&2
  exit 2
fi
EXPECTED_COMMIT=$(tr -d '[:space:]' < "${COMMIT_FILE}")
if ! ACTUAL_COMMIT=$(git -C "${PADDLEFORMERS_DIR}" rev-parse HEAD 2>/dev/null); then
  echo "ERROR: ${PADDLEFORMERS_DIR} is not a Git checkout" >&2
  exit 2
fi

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

GPU_ID=${GPU_ID:-0}
DEVICE=${DEVICE:-gpu:0}
SEED=${SEED:-2026}
PROFILE=${PROFILE:-quick}
RESULTS_DIR=${RESULTS_DIR:-${SCRIPT_DIR}/results}
LOGS_DIR=${LOGS_DIR:-${SCRIPT_DIR}/logs}
RAW_RESULTS_DIR=${RAW_RESULTS_DIR:-${RESULTS_DIR}/raw}

run_task() {
  local task=$1
  local data_value steps warmup learning_rate output log

  if [[ "${task}" == "train" ]]; then
    data_value=${TRAIN_DATA_DIR:-${DATA_DIR:-${SCRIPT_DIR}/data/train}}
    steps=${TRAIN_STEPS:-${STEPS:-20}}
    warmup=${TRAIN_WARMUP_STEPS:-${WARMUP_STEPS:-2}}
    learning_rate=${TRAIN_LEARNING_RATE:-${LEARNING_RATE:-0}}
    if [[ "${TASK}" == "all" ]]; then
      output=${TRAIN_OUTPUT:-${RAW_RESULTS_DIR}/${MODE}-train.json}
      log=${TRAIN_LOG:-${LOGS_DIR}/${MODE}/train.log}
    else
      output=${TRAIN_OUTPUT:-${OUTPUT:-${RAW_RESULTS_DIR}/${MODE}-train.json}}
      log=${TRAIN_LOG:-${LOG:-${LOGS_DIR}/${MODE}/train.log}}
    fi
  else
    data_value=${INFER_DATA_DIR:-${DATA_DIR:-${SCRIPT_DIR}/data/eval}}
    steps=${INFER_STEPS:-${STEPS:-20}}
    warmup=${INFER_WARMUP_STEPS:-${WARMUP_STEPS:-2}}
    # run_model.py accepts a learning-rate argument for a common interface;
    # inference never creates an optimizer and always ignores this value.
    learning_rate=0
    if [[ "${TASK}" == "all" ]]; then
      output=${INFER_OUTPUT:-${RAW_RESULTS_DIR}/${MODE}-infer.json}
      log=${INFER_LOG:-${LOGS_DIR}/${MODE}/infer.log}
    else
      output=${INFER_OUTPUT:-${OUTPUT:-${RAW_RESULTS_DIR}/${MODE}-infer.json}}
      log=${INFER_LOG:-${LOG:-${LOGS_DIR}/${MODE}/infer.log}}
    fi
  fi

  data_value=$(resolve_path "${data_value}")
  output=$(resolve_path "${output}")
  log=$(resolve_path "${log}")
  mkdir -p "$(dirname -- "${output}")" "$(dirname -- "${log}")"

  COMMAND=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/run_model.py"
    --mode "${MODE}"
    --task "${task}"
    --checkpoint "${CHECKPOINT_DIR}"
    --data "${data_value}"
    --steps "${steps}"
    --warmup-steps "${warmup}"
    --learning-rate "${learning_rate}"
    --device "${DEVICE}"
    --seed "${SEED}"
    --output "${output}"
  )

  echo "model=${MODEL_NAME} mode=${MODE} task=${task} profile=${PROFILE} gpu=${GPU_ID}"
  echo "paddleformers=${PADDLEFORMERS_DIR}@${ACTUAL_COMMIT}"
  echo "checkpoint=${CHECKPOINT_DIR}"
  echo "data=${data_value}"
  echo "output=${output}"
  echo "log=${log}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH="${PADDLEFORMERS_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${COMMAND[@]}" 2>&1 | tee "${log}"

  if [[ "${MODE}" == "cinn" ]]; then
    COMPILE_COUNT=$(grep -c 'Compiling subgraph with CINN backend' "${log}" || true)
    if [[ "${COMPILE_COUNT}" -lt 1 ]]; then
      echo "ERROR: CINN compile marker was not found in ${log}" >&2
      exit 5
    fi
    echo "cinn_compile_count=${COMPILE_COUNT}"
  fi

  LAST_OUTPUT=${output}
  LAST_LOG=${log}
}

if [[ "${TASK}" == "all" ]]; then
  run_task train
  TRAIN_RESULT=${LAST_OUTPUT}
  run_task infer
  INFER_RESULT=${LAST_OUTPUT}

  AGGREGATE_OUTPUT=${AGGREGATE_OUTPUT:-${RESULTS_DIR}/${MODE}.json}
  AGGREGATE_OUTPUT=$(resolve_path "${AGGREGATE_OUTPUT}")
  mkdir -p "$(dirname -- "${AGGREGATE_OUTPUT}")"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/aggregate_results.py" \
    --mode "${MODE}" \
    --profile "${PROFILE}" \
    --train "${TRAIN_RESULT}" \
    --infer "${INFER_RESULT}" \
    --output "${AGGREGATE_OUTPUT}"
  echo "aggregate_result=${AGGREGATE_OUTPUT}"
else
  run_task "${TASK}"
fi
