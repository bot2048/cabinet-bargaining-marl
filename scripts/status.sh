#!/usr/bin/env bash
# Progress and expected completion of background experiments.
#   bash scripts/status.sh   or   watch -n 30 bash scripts/status.sh
cd "$(dirname "$0")/.."
PY=$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )
exec $PY scripts/status.py
