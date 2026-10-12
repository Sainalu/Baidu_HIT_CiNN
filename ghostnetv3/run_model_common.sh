#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${PADDLECLAS_DIR:?Set PADDLECLAS_DIR}"
: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR}"
export PADDLECLAS_DIR
export DATA_DIR=${DATA_DIR:-$here/data}
export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}
MODE=${MODE:-dynamic}
TASK=${TASK:-all}
PYTHON_BIN=${PYTHON_BIN:-python}
case "$MODE" in dynamic|cinn) ;; *) exit 2 ;; esac
case "$TASK" in train|infer|all) ;; *) exit 2 ;; esac
[[ $(git -C "$PADDLECLAS_DIR" rev-parse HEAD) = $(tr -d '[:space:]' < "$here/paddleclas.commit") ]] || { echo 'Wrong PaddleClas commit'; exit 2; }
if [[ -f "$here/patches/import-syntax.patch" ]]; then
    cmp <(git -C "$PADDLECLAS_DIR" diff HEAD) "$here/patches/import-syntax.patch" || { echo 'Apply only the documented import syntax patch'; exit 2; }
else
    [[ -z $(git -C "$PADDLECLAS_DIR" status --porcelain --untracked-files=no) ]] || exit 2
fi
"$PYTHON_BIN" "$here/verify_assets.py" "$CHECKPOINT_DIR"
output_root=${RESULTS_DIR:-$here/results}
mkdir -p "$output_root"
run_task() {
    local task=$1
    local output="$output_root/$MODE-$task"
    [[ ! -e "$output" ]] || { echo "Refusing to overwrite $output"; exit 3; }
    local checkpoint="$CHECKPOINT_DIR/initial.pdparams"
    if [[ "$task" = infer ]]; then checkpoint="$CHECKPOINT_DIR/inference.pdparams"; fi
    "$PYTHON_BIN" "$here/run_model.py" --mode "$MODE" --task "$task" --output "$output" --checkpoint "$checkpoint" --epochs "${EPOCHS:-1}" 2>&1 | tee "$output.log"
    if [[ "$MODE" = cinn ]]; then grep -q 'Compiling subgraph with CINN backend' "$output.log" || exit 4; fi
}
if [[ "$TASK" = all ]]; then run_task train; run_task infer; else run_task "$TASK"; fi
