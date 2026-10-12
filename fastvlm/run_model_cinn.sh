#!/usr/bin/env bash
set -euo pipefail
export MODE=cinn
export TASK=${TASK:-all}
exec bash "$(dirname "$0")/run_model_common.sh"
