#!/bin/bash
set -e
cd backend

# Ensure virtual environment exists
if [ ! -d "venv" ]; then
  echo "Creating Python virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies if Flask is missing (first run) or when explicitly requested
if ! python -c "import flask" >/dev/null 2>&1; then
  echo "Installing backend dependencies..."
  python -m pip install --upgrade pip wheel
  pip install -r requirements.txt
fi

# Ensure repo src/ is importable for backend helpers
export PYTHONPATH="${PYTHONPATH}:$(cd .. && pwd)/src"

echo "Starting Flask backend on http://localhost:5000"
python app.py
