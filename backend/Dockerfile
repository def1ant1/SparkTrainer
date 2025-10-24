FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python and basic tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Ensure pip is recent
RUN python3 -m pip install --upgrade pip setuptools wheel

# Keep project structure alignment: backend lives under /app/backend
WORKDIR /app/backend

# Copy requirements and install deps
COPY requirements.txt ./

# Install CUDA-enabled PyTorch first (CUDA 12.1), then remaining deps
RUN python3 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.2.0 torchvision==0.17.0 && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# Copy backend application code
COPY app.py ./

# Create necessary project directories at /app
RUN mkdir -p /app/jobs /app/models /app/logs /app/training_scripts

ENV DGX_TRAINER_BASE_DIR=/app

EXPOSE 5000

CMD ["python3", "app.py"]
