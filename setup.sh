#!/bin/bash

echo "=========================================="
echo "DGX AI Trainer - Quick Start Script"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    echo "Error: Please run this script from the dgx-ai-trainer directory"
    exit 1
fi

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Step 1: Setting up Python backend...${NC}"

# Create directories
mkdir -p jobs models logs

# Setup Python virtual environment
cd backend
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo -e "${GREEN}✓ Backend setup complete${NC}"
echo ""

# Setup frontend
echo -e "${BLUE}Step 2: Setting up React frontend...${NC}"
cd ../frontend

if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
else
    echo "Node modules already installed"
fi

echo -e "${GREEN}✓ Frontend setup complete${NC}"
echo ""

# Create startup scripts
cd ..

echo -e "${BLUE}Step 3: Creating startup scripts...${NC}"

# Backend start script
cat > start_backend.sh << 'EOF'
#!/bin/bash
cd backend
source venv/bin/activate
echo "Starting Flask backend on http://localhost:5000"
python app.py
EOF

chmod +x start_backend.sh

# Frontend start script
cat > start_frontend.sh << 'EOF'
#!/bin/bash
cd frontend
echo "Starting React frontend on http://localhost:3000"
npm run dev
EOF

chmod +x start_frontend.sh

# Combined start script
cat > start.sh << 'EOF'
#!/bin/bash

echo "Starting DGX AI Trainer..."
echo ""
echo "Backend will start on: http://localhost:5000"
echo "Frontend will start on: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Start backend in background
./start_backend.sh &
BACKEND_PID=$!

# Wait a bit for backend to start
sleep 3

# Start frontend in background
./start_frontend.sh &
FRONTEND_PID=$!

# Wait for user interrupt
trap "echo 'Stopping services...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT

# Keep script running
wait
EOF

chmod +x start.sh

echo -e "${GREEN}✓ Startup scripts created${NC}"
echo ""

echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "To start the application:"
echo "  ./start.sh              (Start both backend and frontend)"
echo "  ./start_backend.sh      (Start backend only)"
echo "  ./start_frontend.sh     (Start frontend only)"
echo ""
echo "Or manually:"
echo "  Terminal 1: cd backend && source venv/bin/activate && python app.py"
echo "  Terminal 2: cd frontend && npm run dev"
echo ""
echo "Then open http://localhost:3000 in your browser"
echo ""

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
    nvidia-smi --query-gpu=name --format=csv,noheader
else
    echo -e "${RED}! No NVIDIA GPU detected. Training will use CPU.${NC}"
fi

echo ""
echo "Ready to train AI models on your DGX Spark!"
