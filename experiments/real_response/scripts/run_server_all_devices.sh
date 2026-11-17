#!/usr/bin/env bash
set -euo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$PROJECT/../../scripts/runtime.sh"
cd "$PROJECT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
exec 9>pipeline.lock
flock -n 9 || { echo 'Another pipeline already owns pipeline.lock'; exit 2; }
exec "$PYTHON" -u -m real_response.cli pipeline --profile server_main --runtime-plan auto --resume "$@"
