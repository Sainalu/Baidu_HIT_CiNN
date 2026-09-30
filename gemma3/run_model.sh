#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export CINN_MODE=dynamic
exec bash "${SCRIPT_DIR}/run_model_common.sh" "$@"
