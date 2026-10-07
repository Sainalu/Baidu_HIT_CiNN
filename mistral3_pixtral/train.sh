#!/usr/bin/env bash
set -euo pipefail
export TASK=train
exec bash "$(dirname "$0")/run_model_common.sh"
