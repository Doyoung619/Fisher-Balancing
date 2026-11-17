#!/usr/bin/env bash
set -euo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$PROJECT/../../scripts/runtime.sh"
RUN=$(realpath "${1:?Usage: resume.sh RUN_DIRECTORY [WORKERS]}")
cd "$PROJECT"
"$PYTHON" -c 'from pathlib import Path; import sys; from src.common import validate_run; from src.checkpoint import file_lock; run=Path(sys.argv[1]); validate_run(run); lock=file_lock(run/".supervisor.lock"); lock.__enter__(); (run/"STOP_REQUESTED").unlink(missing_ok=True); lock.__exit__(None,None,None)' "$RUN"
exec "$PROJECT/scripts/launch_detached.sh" "$@"
