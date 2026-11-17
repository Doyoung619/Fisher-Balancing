#!/usr/bin/env bash
set -euo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$PROJECT/../../scripts/runtime.sh"
RUN=${1:-$(cat "$PROJECT/CURRENT_RUN")}
cd "$PROJECT"
exec "$PYTHON" -m src.control status --run-dir "$RUN"
