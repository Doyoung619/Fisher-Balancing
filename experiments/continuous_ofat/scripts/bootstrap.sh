#!/usr/bin/env bash
set -euo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$PROJECT/../../scripts/runtime.sh"
cd "$PROJECT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$PYTHON" -c 'import sys,numpy,scipy,cvxpy,matplotlib,yaml,pytest; print("Interpreter:",sys.executable); print("numpy",numpy.__version__,"scipy",scipy.__version__,"cvxpy",cvxpy.__version__)'
mkdir -p "$PROJECT/audits"
"$PYTHON" -m pip freeze > "$PROJECT/audits/environment.freeze.txt"
"$PYTHON" -m pytest -q "$PROJECT/tests"
printf '%s\n' 'Dependencies verified and tests passed. This script does not alter the server Python environment.'
