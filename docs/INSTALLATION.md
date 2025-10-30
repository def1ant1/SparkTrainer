# SparkTrainer Installation Guide

Complete installation instructions for all platforms and deployment scenarios.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
  - [Docker Compose (Recommended)](#docker-compose-recommended)
  - [Local Development Setup](#local-development-setup)
  - [Cloud Deployment](#cloud-deployment)
- [Post-Installation Setup](#post-installation-setup)
- [Verification](#verification)
- [Common Issues](#common-issues)

---

## System Requirements

### Minimum Requirements

- **CPU**: 4 cores (Intel/AMD x86_64 or ARM64)
- **RAM**: 8GB (16GB recommended)
- **Storage**: 50GB free space
- **OS**: Linux (Ubuntu 20.04+, CentOS 8+), macOS 11+, or Windows 10/11 with WSL2

### Recommended Requirements

- **CPU**: 8+ cores
- **RAM**: 32GB+
- **Storage**: 200GB+ SSD
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for training)
- **OS**: Linux with native Docker support

### GPU Requirements (Optional but Recommended)

For GPU-accelerated training, you need:

- **NVIDIA GPU** with Compute Capability 7.0+ (RTX 2000 series or newer)
- **CUDA**: 11.8 or 12.x
- **NVIDIA Drivers**: 525.x or newer
- **GPU Memory**:
  - 8GB minimum (can train models up to 7B parameters with QLoRA)
  - 16GB recommended (comfortable for most use cases)
  - 24GB+ ideal (for large models and batch processing)
  - 48GB+ for production (80GB for cutting-edge models)

**Supported GPUs:**
- Consumer: RTX 3060, 3070, 3080, 3090, 4070, 4080, 4090
- Professional: RTX A4000, A5000, A6000
- Data Center: A10, A30, A100, H100, L40
- DGX Systems: DGX-1, DGX-2, DGX A100, DGX H100

### Software Prerequisites

**Required:**
- **Docker**: Version 20.10+ ([Install Guide](https://docs.docker.com/get-docker/))
- **Docker Compose**: Version 2.0+ (usually included with Docker)

**For Local Development:**
- **Python**: 3.8, 3.9, 3.10, or 3.11
- **Node.js**: 16.x or 18.x
- **npm**: 8.x or newer

**For GPU Support:**
- **NVIDIA Container Toolkit** ([Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

---

## Installation Methods

### Docker Compose (Recommended)

This is the fastest and easiest way to get SparkTrainer running. Perfect for both development and production.

#### Step 1: Install Docker

**Ubuntu/Debian:**
```bash
# Update package index
sudo apt-get update

# Install dependencies
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Set up repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add your user to docker group (to run without sudo)
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker compose version
```

**macOS:**
```bash
# Install Docker Desktop from:
# https://docs.docker.com/desktop/install/mac-install/

# Or use Homebrew:
brew install --cask docker

# Start Docker Desktop and verify
docker --version
```

**Windows:**
```bash
# 1. Enable WSL2:
wsl --install

# 2. Download and install Docker Desktop:
# https://docs.docker.com/desktop/install/windows-install/

# 3. Open PowerShell and verify:
docker --version
```

#### Step 2: Install NVIDIA Container Toolkit (For GPU)

Only needed if you have an NVIDIA GPU and want GPU acceleration.

**Ubuntu/Debian:**
```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.x   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
...
```

#### Step 3: Clone SparkTrainer

```bash
# Clone the repository
git clone https://github.com/def1ant1/SparkTrainer.git
cd SparkTrainer

# Optional: Switch to a specific version
git checkout v1.0.0  # Replace with desired version
```

#### Step 4: Configure Environment

```bash
# Create .env file from template
cp .env.example .env

# Edit configuration (optional)
nano .env  # or use your favorite editor

# Minimal .env for getting started:
cat > .env << EOF
DATABASE_URL=postgresql://sparktrainer:password@postgres:5432/sparktrainer
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0
MLFLOW_TRACKING_URI=http://mlflow:5001
SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
EOF
```

#### Step 5: Start SparkTrainer

```bash
# Start all services
docker compose up -d

# Watch logs to ensure everything starts correctly
docker compose logs -f

# Wait for services to be healthy (about 60 seconds)
# Press Ctrl+C when you see "Application startup complete"
```

#### Step 6: Initialize Database

```bash
# Initialize database schema and create sample data
docker compose exec backend python init_db.py --sample-data

# If you see "Database initialized successfully" - you're good!
```

#### Step 7: Access SparkTrainer

Open your browser to:
- **Main Interface**: http://localhost:3000
- **MLflow**: http://localhost:5001
- **Flower**: http://localhost:5555
- **API**: http://localhost:5000

**You're done! Skip to [Verification](#verification) section.**

---

### Local Development Setup

For developers who want to modify code or contribute to SparkTrainer.

#### Step 1: Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y \
    python3.10 python3.10-venv python3.10-dev \
    nodejs npm \
    postgresql-14 postgresql-contrib-14 \
    redis-server \
    git curl wget \
    build-essential \
    libpq-dev
```

**macOS:**
```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.10 node postgresql@14 redis git
```

**Windows (WSL2):**
```bash
# In WSL2 Ubuntu terminal:
sudo apt-get update
sudo apt-get install -y \
    python3.10 python3.10-venv python3.10-dev \
    nodejs npm \
    postgresql redis-server \
    build-essential
```

#### Step 2: Clone and Setup

```bash
# Clone repository
git clone https://github.com/def1ant1/SparkTrainer.git
cd SparkTrainer

# Create Python virtual environment
cd backend
python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Python dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Install SparkTrainer package in development mode
cd ..
pip install -e .
```

#### Step 3: Setup Database

**Ubuntu/Debian:**
```bash
# Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres psql << EOF
CREATE DATABASE sparktrainer;
CREATE USER sparktrainer WITH ENCRYPTED PASSWORD 'password';
GRANT ALL PRIVILEGES ON DATABASE sparktrainer TO sparktrainer;
ALTER DATABASE sparktrainer OWNER TO sparktrainer;
\q
EOF
```

**macOS:**
```bash
# Start PostgreSQL
brew services start postgresql@14

# Create database
createdb sparktrainer
```

#### Step 4: Setup Redis

**Ubuntu/Debian:**
```bash
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

**macOS:**
```bash
brew services start redis
```

#### Step 5: Setup MLflow

```bash
# In backend directory with venv activated
cd backend

# Start MLflow server (in a separate terminal)
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5001
```

#### Step 6: Configure Environment

```bash
# Create .env in backend directory
cat > backend/.env << EOF
DATABASE_URL=postgresql://sparktrainer:password@localhost:5432/sparktrainer
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
MLFLOW_TRACKING_URI=http://localhost:5001
FLASK_ENV=development
FLASK_DEBUG=1
SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
EOF
```

#### Step 7: Initialize Database

```bash
cd backend
source venv/bin/activate
python init_db.py --sample-data
```

#### Step 8: Start Services

You'll need **3 terminal windows**:

**Terminal 1 - Backend API:**
```bash
cd backend
source venv/bin/activate
python app.py
```

**Terminal 2 - Celery Worker:**
```bash
cd backend
source venv/bin/activate
celery -A celery_app.celery worker --loglevel=info --pool=solo
```

**Terminal 3 - Frontend:**
```bash
cd frontend
npm install
npm run dev
```

#### Step 9: Access Application

Open http://localhost:3000

---

### Cloud Deployment

Deploy SparkTrainer on cloud platforms for production use.

#### AWS EC2

**Step 1: Launch Instance**
```bash
# Recommended instance types:
# - g4dn.xlarge (1x NVIDIA T4, 16GB GPU)
# - p3.2xlarge (1x NVIDIA V100, 16GB GPU)
# - p4d.24xlarge (8x NVIDIA A100, 40GB GPU each)

# AMI: Deep Learning AMI (Ubuntu 20.04)
# Storage: 200GB+ EBS GP3

# Security Group: Allow ports 22, 80, 443, 3000, 5000, 5001
```

**Step 2: Connect and Install**
```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Clone and setup
git clone https://github.com/def1ant1/SparkTrainer.git
cd SparkTrainer

# Use Docker Compose method (see above)
docker compose up -d

# Initialize
docker compose exec backend python init_db.py --sample-data
```

#### Google Cloud Platform (GCP)

**Step 1: Create VM**
```bash
gcloud compute instances create sparktrainer \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --maintenance-policy=TERMINATE

# Install NVIDIA drivers if needed
gcloud compute ssh sparktrainer --zone=us-central1-a
sudo /opt/deeplearning/install-driver.sh
```

**Step 2: Deploy**
```bash
# Follow Docker Compose installation steps above
git clone https://github.com/def1ant1/SparkTrainer.git
cd SparkTrainer
docker compose up -d
```

#### Azure

```bash
# Create VM with GPU
az vm create \
    --resource-group myResourceGroup \
    --name sparktrainer \
    --image microsoft-dsvm:ubuntu-2004:2004-gen2:latest \
    --size Standard_NC6s_v3 \
    --admin-username azureuser \
    --generate-ssh-keys

# Connect and deploy
ssh azureuser@your-vm-ip
git clone https://github.com/def1ant1/SparkTrainer.git
cd SparkTrainer
docker compose up -d
```

#### Kubernetes (Advanced)

Coming soon! See [KUBERNETES.md](KUBERNETES.md)

---

## Post-Installation Setup

### Create Your First User (Optional)

If authentication is enabled:

```bash
docker compose exec backend python << EOF
from backend.database import SessionLocal
from backend.models import User
from werkzeug.security import generate_password_hash

db = SessionLocal()
user = User(
    username="admin",
    email="admin@example.com",
    password_hash=generate_password_hash("changeme"),
    is_admin=True
)
db.add(user)
db.commit()
print("User created! Username: admin, Password: changeme")
EOF
```

### Configure GPU Settings

Edit `.env`:
```bash
# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1  # Use first two GPUs

# Or use all GPUs (default)
CUDA_VISIBLE_DEVICES=

# Disable GPU (CPU only)
CUDA_VISIBLE_DEVICES=""
```

### Setup Storage Backend

**For S3:**
```bash
# Add to .env
STORAGE_BACKEND=s3
S3_BUCKET=my-sparktrainer-bucket
S3_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

**For GCS:**
```bash
# Add to .env
STORAGE_BACKEND=gcs
GCS_BUCKET=my-sparktrainer-bucket
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

---

## Verification

### Check Services

```bash
# Check all containers are running
docker compose ps

# Should show all services as "Up"
# NAME                 STATUS
# sparktrainer_postgres_1    Up (healthy)
# sparktrainer_redis_1       Up
# sparktrainer_mlflow_1      Up
# sparktrainer_backend_1     Up
# sparktrainer_frontend_1    Up
# sparktrainer_worker_1      Up
```

### Test API

```bash
# Health check
curl http://localhost:5000/api/health

# Expected response:
# {"status": "healthy", "version": "1.0.0"}

# System info
curl http://localhost:5000/api/system/info

# Should show GPU information if available
```

### Test Web Interface

1. Open http://localhost:3000
2. You should see the SparkTrainer dashboard
3. Check "System Info" widget shows correct GPU count
4. Try creating a project

### Test Training

```bash
# Quick smoke test
docker compose exec backend python << EOF
from spark_trainer.recipes.lora_recipes import create_lora_recipe

config = {
    "base_model": "distilbert-base-uncased",
    "dataset_name": "sample_dataset",
    "num_epochs": 1,
    "batch_size": 2,
    "output_dir": "./test_output"
}

recipe = create_lora_recipe(config)
recipe.run()
print("✅ Training test passed!")
EOF
```

---

## Common Issues

### Docker Containers Won't Start

```bash
# Check logs
docker compose logs backend
docker compose logs postgres

# Common fixes:
docker compose down
docker volume prune  # WARNING: Deletes all data!
docker compose up -d
```

### Port Already in Use

```bash
# Find what's using the port
sudo lsof -i :5000  # Backend
sudo lsof -i :3000  # Frontend

# Kill the process
sudo kill -9 <PID>

# Or change ports in docker-compose.yml
```

### GPU Not Detected

```bash
# Verify NVIDIA driver
nvidia-smi

# Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# If fails, reinstall NVIDIA Container Toolkit:
sudo apt-get install --reinstall nvidia-container-toolkit
sudo systemctl restart docker
```

### Database Connection Failed

```bash
# Check PostgreSQL is running
docker compose ps postgres

# Restart if needed
docker compose restart postgres

# Reset database (WARNING: Deletes all data!)
docker compose down -v
docker compose up -d postgres
docker compose exec backend python init_db.py --sample-data
```

### Out of Disk Space

```bash
# Check disk usage
df -h

# Clean Docker
docker system prune -a --volumes  # WARNING: Deletes everything!

# Or selectively clean:
docker image prune  # Remove unused images
docker volume prune  # Remove unused volumes
```

---

## Next Steps

Now that SparkTrainer is installed:

1. **[Quick Start Guide](../QUICKSTART.md)**: Train your first model
2. **[Tutorial](TUTORIAL.md)**: Complete walkthrough
3. **[Configuration Guide](CONFIGURATION.md)**: Customize settings
4. **[User Guide](USER_GUIDE.md)**: Learn all features

Need help? Visit [GitHub Discussions](https://github.com/def1ant1/SparkTrainer/discussions)

---

## Updating SparkTrainer

```bash
# Pull latest changes
cd SparkTrainer
git pull

# Rebuild containers
docker compose down
docker compose build
docker compose up -d

# Run database migrations (if any)
docker compose exec backend alembic upgrade head
```

---

## Uninstallation

```bash
# Stop all services
docker compose down

# Remove all data (WARNING: Irreversible!)
docker compose down -v

# Remove directory
cd ..
rm -rf SparkTrainer
```
