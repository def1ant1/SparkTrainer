#!/bin/bash
cd backend
source venv/bin/activate
echo "Starting Flask backend on http://localhost:5000"
python app.py
