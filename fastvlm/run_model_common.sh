#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${PADDLEFORMERS_DIR:?Set PADDLEFORMERS_DIR}"
: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR (weights are not uploaded)}"
export PADDLEFORMERS_DIR
MODE=${MODE:-dynamic}
TASK=${TASK:-all}
case "$MODE" in dynamic|cinn) ;; *) exit 2 ;; esac
case "$TASK" in train|infer|all) ;; *) exit 2 ;; esac
expected=$(tr -d '[:space:]' < "$here/paddleformers.commit")
actual=$(git -C "$PADDLEFORMERS_DIR" rev-parse HEAD)
if [[ "$actual" != "$expected" ]]; then
    echo "PaddleFormers commit mismatch: expected $expected, got $actual" >&2
    exit 2
fi
if [[ -n $(git -C "$PADDLEFORMERS_DIR" status --porcelain --untracked-files=no) ]]; then
    echo 'Tracked PaddleFormers changes found; use a clean worktree.' >&2
    exit 2
fi
PYTHON_BIN=${PYTHON_BIN:-python}
export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}
"$PYTHON_BIN" "$here/verify_assets.py" "$CHECKPOINT_DIR"
output_root=${RESULTS_DIR:-$here/results}
mkdir -p "$output_root"
run_task() {
    local task=$1
    local output="$output_root/$MODE-$task"
    [[ ! -e "$output" ]] || { echo "Refusing to overwrite $output" >&2; exit 3; }
    local args=(--mode "$MODE" --native-shape --split --output "$output")
    if [[ "$task" = infer ]]; then
        args+=(--infer --checkpoint "$CHECKPOINT_DIR/inference.pdparams")
    else
        args+=(--checkpoint "$CHECKPOINT_DIR/initial.pdparams" --steps "${STEPS:-20}")
        if [[ -n ${RESUME:-} ]]; then args+=(--resume "$RESUME"); fi
    fi
    "$PYTHON_BIN" "$here/run_model.py" "${args[@]}" 2>&1 | tee "$output.log"
    if [[ "$MODE" = cinn ]]; then
        grep -q 'Compiling subgraph with CINN backend' "$output.log" || { echo 'Missing CINN compile marker' >&2; exit 4; }
    fi
}
if [[ "$TASK" = all ]]; then run_task train; run_task infer; else run_task "$TASK"; fi
