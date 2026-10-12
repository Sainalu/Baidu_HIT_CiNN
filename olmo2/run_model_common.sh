#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${PADDLEFORMERS_DIR:?Set PADDLEFORMERS_DIR}"
export PADDLEFORMERS_DIR
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$here/checkpoint}
PYTHON_BIN=${PYTHON_BIN:-python}
export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}
MODE=${MODE:-dynamic}
TASK=${TASK:-all}
case "$MODE" in dynamic|cinn) ;; *) exit 2 ;; esac
case "$TASK" in train|infer|all) ;; *) exit 2 ;; esac
base=$(tr -d '[:space:]' < "$here/paddleformers.commit")
# Accept the original snapshot or the public base plus exactly the recorded patch.
cmp <(git -C "$PADDLEFORMERS_DIR" diff --no-ext-diff --binary "$base") "$here/source_snapshot.patch"
"$PYTHON_BIN" "$here/verify_assets.py" "$CHECKPOINT_DIR"
output_root=${RESULTS_DIR:-$here/results}
mkdir -p "$output_root"
run_task() {
    local task=$1 output="$output_root/$MODE-$1"
    [[ ! -e "$output" ]] || { echo "Refusing to overwrite $output" >&2; exit 3; }
    local args=(--mode "$MODE" --output "$output" --static-config-get --extended --probe-gradients)
    if [[ "$task" = infer ]]; then
        args+=(--inference-only --initial-checkpoint "$CHECKPOINT_DIR/inference.pdparams")
    else
        args+=(--steps "${STEPS:-20}" --initial-checkpoint "$CHECKPOINT_DIR/initial.pdparams")
        if [[ -n ${RESUME:-} ]]; then args+=(--resume "$RESUME"); fi
    fi
    "$PYTHON_BIN" "$here/run_model.py" "${args[@]}" 2>&1 | tee "$output.log"
    if [[ "$MODE" = cinn ]]; then grep -q 'Compiling subgraph with CINN backend' "$output.log"; fi
}
if [[ "$TASK" = all ]]; then run_task train; run_task infer; else run_task "$TASK"; fi
