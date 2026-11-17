#!/usr/bin/env bash
set -euo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$PROJECT/../../scripts/runtime.sh"
RUN=${1:-$(cat "$PROJECT/CURRENT_RUN")}
cd "$PROJECT"
"$PYTHON" -m src.control resume --run-dir "$RUN"
exec "$PROJECT/scripts/launch.sh" "$RUN" "${2:-28}" "${3:-4}"
