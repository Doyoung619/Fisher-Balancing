#!/usr/bin/env bash
# Source from an experiment wrapper after setting PROJECT.
REPO_ROOT=$(cd -- "$PROJECT/../.." && pwd)
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON=$(command -v python3)
  fi
fi
export PYTHON
