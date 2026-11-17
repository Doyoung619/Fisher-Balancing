#!/usr/bin/env bash
set -euo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$PROJECT/../../scripts/runtime.sh"
RUN=$(realpath "${1:?Usage: launch.sh RUN [CPU_WORKERS=28] [GPU_WORKERS=4]}")
CPU_WORKERS=${2:-28}
GPU_WORKERS=${3:-4}
test -f "$RUN/config.json"
test ! -f "$RUN/STOP_REQUESTED" || { echo 'Run stopped; use resume.sh.' >&2; exit 1; }
mkdir -p "$RUN/logs"
SESSION="sampled-$(basename "$RUN" | tr -cs '[:alnum:]_-' '-')"
COMMAND=("$PYTHON" -m src.control watch --run-dir "$RUN" --cpu-workers "$CPU_WORKERS" --gpu-workers "$GPU_WORKERS")
printf -v QUOTED_COMMAND '%q ' "${COMMAND[@]}"
printf -v QUOTED_PROJECT '%q' "$PROJECT"
printf -v QUOTED_LOG '%q' "$RUN/logs/supervisor.log"
WRAPPED="cd $QUOTED_PROJECT && export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 && exec $QUOTED_COMMAND >>$QUOTED_LOG 2>&1"
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  printf 'Already running: %s\n' "$SESSION"
else
  tmux new-session -d -s "$SESSION" /bin/bash --noprofile --norc -c "$WRAPPED"
  sleep 1
  tmux has-session -t "=$SESSION"
  printf 'Started: %s\n' "$SESSION"
fi
printf '%s\n' "$SESSION" > "$RUN/tmux_session.txt"
printf 'Run: %s\n' "$RUN"
