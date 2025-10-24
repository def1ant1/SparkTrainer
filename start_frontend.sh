#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/frontend"

if [ ! -d node_modules ]; then
  echo "[!] Node dependencies not installed. Run: cd frontend && npm install" >&2
  exit 1
fi

echo "Starting React frontend on http://localhost:3000"
exec npm run dev
