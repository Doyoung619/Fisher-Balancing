#!/usr/bin/env bash
set -euo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$PROJECT/../../scripts/runtime.sh"
RUN=$(realpath "${1:?Usage: status.sh RUN_DIRECTORY}")
cd "$PROJECT"
exec "$PYTHON" -m src.supervisor status --run-dir "$RUN"
