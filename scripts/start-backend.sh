#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/python ]]; then
  printf 'Run bash scripts/setup-mac.sh first.\n'; exit 1
fi
exec .venv/bin/python -m uvicorn challenge.backend.main:app --host 127.0.0.1 --port 8000
