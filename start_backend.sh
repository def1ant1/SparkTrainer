#!/bin/bash
set -e

# Determine project root and export base dir for backend
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DGX_TRAINER_BASE_DIR="$SCRIPT_DIR"

cd "$SCRIPT_DIR/backend"

# Activate venv if present
if [ -d "venv" ]; then
  source venv/bin/activate
fi

# Quick dependency check with a helpful hint
python - <<'PY' || {
  echo "[!] Missing Python dependencies. Activate venv and run: pip install -r backend/requirements.txt" >&2
  exit 1
}
try:
    import flask, flask_cors  # noqa: F401
except Exception as e:
    raise SystemExit(e)
PY

echo "Starting Flask backend on http://localhost:5000"
exec python app.py
