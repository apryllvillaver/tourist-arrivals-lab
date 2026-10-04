#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v python3.11 >/dev/null 2>&1; then
  printf 'Python 3.11 is missing. Install it with: brew install python@3.11\n'
  exit 1
fi
if ! command -v node >/dev/null 2>&1; then
  printf 'Node.js is missing. Install it with: brew install node@22\n'
  exit 1
fi
node -e 'const [a,b]=process.versions.node.split(".").map(Number); if(a<20 || (a===20 && b<9)){ console.error("Node.js 20.9 or newer is required. Install Node 22."); process.exit(1); }'
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
npm --prefix challenge/frontend ci
printf '\nSetup finished.\nStreamlit: bash scripts/start-streamlit.sh\nBackend:   bash scripts/start-backend.sh\nFrontend:  bash scripts/start-frontend.sh\n'
