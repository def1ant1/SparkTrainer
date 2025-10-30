#!/bin/bash
set -e
cd backend

# Ensure virtual environment exists
if [ ! -d "venv" ]; then
  echo "Creating Python virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies if critical packages are missing
NEED_PG=0
if [ -n "${DATABASE_URL}" ] && echo "$DATABASE_URL" | grep -qi '^postgres'; then
  NEED_PG=1
fi

if [ "$NEED_PG" = "1" ]; then
  PKG_CHECK="import flask, psycopg2"
else
  PKG_CHECK="import flask"
fi

if ! python - <<PY
try:
    $PKG_CHECK
except Exception:
    raise SystemExit(1)
PY
then
  echo "Installing backend dependencies..."
  python -m pip install --upgrade pip wheel
  pip install -r requirements.txt
fi

# Ensure repo src/ is importable for backend helpers
export PYTHONPATH="${PYTHONPATH}:$(cd .. && pwd)/src"

echo "Starting Flask backend on http://localhost:5000"
python app.py
