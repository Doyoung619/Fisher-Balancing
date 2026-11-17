#!/usr/bin/env bash
set -euo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$PROJECT/../../scripts/runtime.sh"
RUN=$(realpath "${1:?Usage: launch_detached.sh RUN_DIRECTORY [WORKERS]}")
WORKERS=${2:-}
cd "$PROJECT"
test -f "$RUN/config.json"
test -f "$RUN/manifest.json"
test ! -e "$RUN/STOP_REQUESTED" || { printf '%s\n' 'Run was stopped. Use resume.sh to clear its graceful-stop request.' >&2; exit 1; }
"$PYTHON" -c 'from pathlib import Path; import sys; from src.common import validate_run; validate_run(Path(sys.argv[1]))' "$RUN"
mkdir -p "$RUN/logs"
SESSION="ofat-$(basename "$RUN" | tr -cs '[:alnum:]_-' '-')"
COMMAND=("$PYTHON" -m src.supervisor watch --run-dir "$RUN")
if [[ -n "$WORKERS" ]]; then COMMAND+=(--workers "$WORKERS"); fi
printf -v QUOTED_COMMAND '%q ' "${COMMAND[@]}"
printf -v QUOTED_PROJECT '%q' "$PROJECT"
printf -v QUOTED_LOG '%q' "$RUN/logs/supervisor.log"
printf -v QUOTED_SESSION '%q' "$SESSION"
# Keep stdin attached to tmux's PTY. Closing all three PTY descriptors makes
# tmux close the pane and send SIGHUP before the watchdog can initialize.
WRAPPED="cd $QUOTED_PROJECT && export OFAT_TMUX_SESSION=$QUOTED_SESSION OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 && exec $QUOTED_COMMAND >>$QUOTED_LOG 2>&1"
if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "=$SESSION" 2>/dev/null; then
    printf 'Existing session: %s\n' "$SESSION"
  else
    tmux new-session -d -s "$SESSION" /bin/bash --noprofile --norc -c "$WRAPPED"
    sleep 1
    tmux has-session -t "=$SESSION" || { printf '%s\n' 'Watchdog exited immediately; inspect logs/supervisor.log.' >&2; exit 1; }
    printf 'Launched tmux session: %s\n' "$SESSION"
  fi
else
  nohup setsid /bin/bash -c "$WRAPPED" </dev/null >>"$RUN/logs/launcher.log" 2>&1 &
  printf 'Launched setsid watchdog PID: %s\n' "$!"
fi
printf 'Run: %s\nActual supervisor command: ' "$RUN"
printf '%q ' "${COMMAND[@]}"
printf '\n'
