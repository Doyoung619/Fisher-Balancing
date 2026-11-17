#!/usr/bin/env bash
set -euo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$PROJECT/../../scripts/runtime.sh"
RUN=${1:-$(cat "$PROJECT/CURRENT_RUN")}
cd "$PROJECT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$PYTHON" -m src.plotting --run-dir "$RUN" "${@:2}"
