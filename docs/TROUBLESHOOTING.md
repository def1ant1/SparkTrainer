# SparkTrainer Troubleshooting Guide

Solutions to common problems and errors when using SparkTrainer.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Training Errors](#training-errors)
- [GPU Problems](#gpu-problems)
- [Database Issues](#database-issues)
- [Performance Issues](#performance-issues)
- [API & Frontend Issues](#api--frontend-issues)
- [Dataset Problems](#dataset-problems)
- [Model Issues](#model-issues)
- [Getting Help](#getting-help)

---

## Installation Issues

### Docker Compose Fails to Start

**Problem:** `docker compose up -d` fails or containers don't start.

**Solutions:**

```bash
# 1. Check Docker is running
docker ps
# If error: "Cannot connect to the Docker daemon"
sudo systemctl start docker

# 2. Check for port conflicts
sudo lsof -i :5000  # Backend
sudo lsof -i :3000  # Frontend
sudo lsof -i :5432  # PostgreSQL
# Kill conflicting processes or change ports in docker-compose.yml

# 3. View logs to see what's wrong
docker compose logs backend
docker compose logs postgres

# 4. Clean start
docker compose down
docker compose up -d

# 5. If still failing, full reset (WARNING: Deletes data!)
docker compose down -v
docker system prune -a
docker compose up -d
```

### Permission Denied Errors

**Problem:** "Permission denied" when running Docker commands.

**Solution:**

```bash
# Add your user to docker group
sudo usermod -aG docker $USER

# Log out and back in, or run:
newgrp docker

# Test it works
docker ps
```

### Cannot Install NVIDIA Container Toolkit

**Problem:** GPU not accessible in Docker containers.

**Solutions:**

```bash
# 1. Verify NVIDIA driver is installed
nvidia-smi

# 2. If no driver, install it
sudo apt-get update
sudo apt-get install -y nvidia-driver-525  # Or latest version

# 3. Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 4. Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 5. Test
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Out of Disk Space

**Problem:** Installation fails with "no space left on device".

**Solutions:**

```bash
# 1. Check disk usage
df -h

# 2. Clean Docker images and volumes
docker system df  # See what's using space
docker system prune -a  # Remove unused everything
docker volume prune  # Remove unused volumes

# 3. Clean SparkTrainer artifacts
rm -rf models/*  # Delete old models
rm -rf datasets/*  # Delete old datasets
rm -rf logs/*  # Delete logs
rm -rf mlruns/*  # Delete MLflow artifacts

# 4. Move Docker to larger drive (if needed)
# See: https://docs.docker.com/config/daemon/#daemon-data-directory
```

---

## Training Errors

### Out of Memory (OOM) Errors

**Problem:** Training fails with "CUDA out of memory" or "Killed" message.

**Solutions:**

```bash
# SOLUTION 1: Reduce batch size
# In training config, change:
batch_size: 2  # Was 8 or 16

# SOLUTION 2: Enable QLoRA (4-bit quantization)
use_qlora: true  # In LoRA recipe
# Saves 75% GPU memory!

# SOLUTION 3: Enable gradient checkpointing
gradient_checkpointing: true
# Trades compute for memory

# SOLUTION 4: Reduce sequence length
max_length: 512  # Was 2048

# SOLUTION 5: Use gradient accumulation
batch_size: 1
gradient_accumulation_steps: 16
# Effective batch size = 1 * 16 = 16, but uses 1/16th memory

# SOLUTION 6: Use a smaller model
# Instead of: meta-llama/Llama-2-7b-hf
# Try: distilbert-base-uncased (much smaller)

# SOLUTION 7: Use DeepSpeed with CPU offloading
# Add to config:
deepspeed_config: configs/deepspeed_zero3.json
# Offloads optimizer and parameters to CPU RAM
```

**Check available memory:**

```bash
# GPU memory
nvidia-smi

# System memory
free -h

# Monitor in real-time
watch -n 1 nvidia-smi
```

### Training Hangs or Freezes

**Problem:** Training starts but makes no progress.

**Solutions:**

```bash
# 1. Check if worker is actually running
docker compose ps worker
# Or for local:
ps aux | grep celery

# 2. Check worker logs
docker compose logs worker
# Look for errors or stack traces

# 3. Restart worker
docker compose restart worker

# 4. Check GPU is not stuck
nvidia-smi
# If GPU utilization is 0%, training isn't running
# If GPU utilization is 100% but no progress, might be deadlocked

# 5. Kill and restart training
# In UI: Go to Jobs → Click job → Cancel
# Or via API:
curl -X POST http://localhost:5000/api/jobs/<job_id>/cancel

# 6. Check dataset is valid
# Go to Datasets → Your dataset → Preview samples
# Verify data loads correctly
```

### Training Loss is NaN or Infinity

**Problem:** Training starts but loss becomes NaN.

**Solutions:**

```python
# SOLUTION 1: Reduce learning rate
learning_rate: 1e-5  # Was 1e-3 or 2e-4
# Too high LR causes gradients to explode

# SOLUTION 2: Enable gradient clipping
max_grad_norm: 1.0
# Prevents gradient explosion

# SOLUTION 3: Check dataset for bad values
# Remove samples with:
# - Empty text
# - Very long sequences
# - Special characters that break tokenization
# - Null/undefined values

# SOLUTION 4: Use mixed precision carefully
fp16: false  # Disable if causing issues
# Or try:
bf16: true   # Better numerical stability than fp16

# SOLUTION 5: Check for data normalization
# If using images, ensure they're normalized:
# Values should be in [0, 1] or standardized

# SOLUTION 6: Use a smaller batch size
batch_size: 1  # Helps with numerical stability
```

### Training is Very Slow

**Problem:** Training takes much longer than expected.

**Check what's slow:**

```bash
# 1. Is GPU being used?
nvidia-smi
# GPU-Util should be 70-100%
# If 0%, training is on CPU (100x slower!)

# 2. Check data loading
# In logs, look for:
# - "Loading dataset..." (should be fast)
# - High disk I/O (iostat -x 1)

# 3. Check batch size
# Larger batch = faster training (up to memory limit)

# 4. Profile the training
# Add to config:
profile: true
# Generates profiling report
```

**Solutions:**

```python
# SOLUTION 1: Increase batch size
batch_size: 16  # Was 4
# Bigger batches are more efficient

# SOLUTION 2: Use data prefetching
dataloader_num_workers: 4  # Was 0
# Loads data in parallel

# SOLUTION 3: Use mixed precision
fp16: true  # Or bf16: true
# 2-3x faster on modern GPUs

# SOLUTION 4: Reduce logging frequency
logging_steps: 100  # Was 10
save_steps: 500  # Was 100
# Less I/O overhead

# SOLUTION 5: Use faster dataset format
# Convert to:
# - HuggingFace Arrow format (fast random access)
# - WebDataset (streaming)
# Instead of: JSONL, CSV (slow)

# SOLUTION 6: Enable compilation (PyTorch 2.0+)
compile_model: true  # Uses torch.compile
# 20-50% speedup

# SOLUTION 7: Use DeepSpeed
deepspeed: true
# Optimized for large-scale training
```

---

## GPU Problems

### GPU Not Detected

**Problem:** SparkTrainer doesn't see your GPU.

**Diagnosis:**

```bash
# 1. Check GPU is visible to system
nvidia-smi
# Should show your GPU(s)

# 2. Check Docker can access GPU
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
# Should show GPU info

# 3. Check SparkTrainer sees GPU
curl http://localhost:5000/api/system/info
# Look for "gpu_count" > 0

# 4. Check CUDA_VISIBLE_DEVICES
docker compose exec backend printenv CUDA_VISIBLE_DEVICES
# Should be empty or "0,1,2,3" etc., NOT ""
```

**Solutions:**

```bash
# SOLUTION 1: Verify NVIDIA driver
nvidia-smi
# If error, reinstall driver:
sudo apt-get install -y nvidia-driver-525

# SOLUTION 2: Install NVIDIA Container Toolkit
# See Installation Issues section above

# SOLUTION 3: Check Docker Compose config
# In docker-compose.yml, ensure:
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]

# SOLUTION 4: Set CUDA_VISIBLE_DEVICES correctly
# In .env:
CUDA_VISIBLE_DEVICES=0,1  # NOT ""

# SOLUTION 5: Restart Docker
sudo systemctl restart docker
docker compose down
docker compose up -d
```

### Multiple GPU Issues

**Problem:** Training doesn't use all GPUs or crashes on multi-GPU setup.

**Solutions:**

```python
# SOLUTION 1: Enable distributed training
# In training config:
distributed: true
world_size: 4  # Number of GPUs

# SOLUTION 2: Use DataParallel or FSDP
training_mode: "fsdp"  # Or "ddp"

# SOLUTION 3: Set CUDA_VISIBLE_DEVICES correctly
# In .env:
CUDA_VISIBLE_DEVICES=0,1,2,3  # Use GPUs 0-3

# SOLUTION 4: Check GPUs are healthy
nvidia-smi
# Look for:
# - Matching GPU memory
# - No ECC errors
# - No processes stuck

# SOLUTION 5: Use single GPU if problems persist
CUDA_VISIBLE_DEVICES=0  # Use only GPU 0
```

### GPU Memory Leak

**Problem:** GPU memory usage increases over time, eventually causing OOM.

**Solutions:**

```python
# SOLUTION 1: Clear cache periodically
# Add to training loop:
import torch
torch.cuda.empty_cache()

# SOLUTION 2: Enable garbage collection
import gc
gc.collect()

# SOLUTION 3: Fix gradient accumulation
# Make sure to zero gradients:
optimizer.zero_grad()  # Before each backward pass

# SOLUTION 4: Delete intermediate tensors
del loss, outputs  # After backward pass

# SOLUTION 5: Use gradient checkpointing
gradient_checkpointing: true

# SOLUTION 6: Restart training from checkpoint
# Cancel job and resume from last checkpoint
```

---

## Database Issues

### Cannot Connect to Database

**Problem:** "Could not connect to database" error.

**Solutions:**

```bash
# 1. Check PostgreSQL is running
docker compose ps postgres
# Should show "Up (healthy)"

# 2. Check database logs
docker compose logs postgres

# 3. Test connection
docker compose exec postgres psql -U sparktrainer -d sparktrainer
# Should open psql prompt

# 4. Check DATABASE_URL in .env
cat .env | grep DATABASE_URL
# Should be: postgresql://sparktrainer:password@postgres:5432/sparktrainer
# For local setup: postgresql://sparktrainer:password@localhost:5432/sparktrainer

# 5. Restart PostgreSQL
docker compose restart postgres

# 6. Recreate database (WARNING: Deletes data!)
docker compose down postgres
docker volume rm sparktrainer_postgres_data
docker compose up -d postgres
docker compose exec backend python init_db.py --sample-data
```

### Database Migration Errors

**Problem:** "Database schema mismatch" or migration fails.

**Solutions:**

```bash
# 1. Check current migration status
docker compose exec backend alembic current

# 2. View pending migrations
docker compose exec backend alembic history

# 3. Run migrations
docker compose exec backend alembic upgrade head

# 4. If migration fails, check logs
docker compose logs backend | grep alembic

# 5. Reset database (WARNING: Deletes all data!)
docker compose exec backend alembic downgrade base
docker compose exec backend alembic upgrade head

# 6. Full database reset (last resort)
docker compose down -v
docker compose up -d
docker compose exec backend python init_db.py --sample-data
```

### Database is Slow

**Problem:** Queries take a long time or UI is sluggish.

**Solutions:**

```bash
# 1. Check database size
docker compose exec postgres psql -U sparktrainer -d sparktrainer -c "SELECT pg_size_pretty(pg_database_size('sparktrainer'));"

# 2. Analyze query performance
docker compose exec postgres psql -U sparktrainer -d sparktrainer -c "SELECT * FROM pg_stat_activity WHERE state = 'active';"

# 3. Vacuum database
docker compose exec postgres psql -U sparktrainer -d sparktrainer -c "VACUUM ANALYZE;"

# 4. Check for missing indexes
# In PostgreSQL:
# Look at slow queries and add indexes

# 5. Increase PostgreSQL resources
# Edit docker-compose.yml:
environment:
  - POSTGRES_SHARED_BUFFERS=256MB  # Increase
  - POSTGRES_EFFECTIVE_CACHE_SIZE=1GB

# 6. Archive old data
# Delete old jobs/experiments you don't need
```

---

## Performance Issues

### High CPU Usage

**Problem:** Backend or worker uses 100% CPU.

**Solutions:**

```bash
# 1. Check what's using CPU
docker compose exec backend top
# Or:
htop  # On host

# 2. Reduce Celery worker concurrency
# In docker-compose.yml:
command: celery -A celery_app.celery worker --concurrency=2
# Was: --concurrency=4

# 3. Limit data preprocessing workers
# In training config:
dataloader_num_workers: 2  # Was 8

# 4. Reduce logging frequency
logging_steps: 100  # Was 10

# 5. Check for infinite loops in logs
docker compose logs worker | grep -i "loop\|hang\|stuck"
```

### High Memory Usage

**Problem:** System runs out of RAM.

**Solutions:**

```bash
# 1. Check memory usage
free -h
docker stats  # Per-container

# 2. Limit container memory
# In docker-compose.yml:
deploy:
  resources:
    limits:
      memory: 8G  # Limit to 8GB

# 3. Reduce dataset cache
# In training config:
cache_dir: null  # Disable caching
# Or:
max_cache_size: 1000  # Cache only 1000 samples

# 4. Process datasets in streaming mode
streaming: true  # Don't load entire dataset into memory

# 5. Reduce batch size (affects training memory)
batch_size: 4  # Was 16

# 6. Close unused services
docker compose stop mlflow  # If not using
docker compose stop flower  # If not monitoring
```

### Slow Frontend

**Problem:** Web UI is slow or unresponsive.

**Solutions:**

```bash
# 1. Check network latency
curl -w "@-" -o /dev/null -s http://localhost:5000/api/health <<'EOF'
    time_namelookup:  %{time_namelookup}\n
       time_connect:  %{time_connect}\n
    time_appconnect:  %{time_appconnect}\n
      time_redirect:  %{time_redirect}\n
   time_starttransfer:  %{time_starttransfer}\n
                     ------\n
         time_total:  %{time_total}\n
EOF

# 2. Check browser console for errors
# Open DevTools (F12) → Console

# 3. Clear browser cache
# Ctrl+Shift+Del → Clear cache

# 4. Check API performance
# Look for slow endpoints:
docker compose logs backend | grep "took.*ms"

# 5. Reduce polling frequency
# In frontend config:
POLLING_INTERVAL=5000  # Was 1000 (5 seconds instead of 1)

# 6. Disable auto-refresh on large lists
# When viewing 1000+ jobs/models
```

---

## API & Frontend Issues

### Frontend Shows "Cannot connect to backend"

**Problem:** UI can't reach API server.

**Solutions:**

```bash
# 1. Check backend is running
docker compose ps backend
curl http://localhost:5000/api/health

# 2. Check frontend environment
# In frontend/.env:
VITE_API_URL=http://localhost:5000
# Should match where backend is running

# 3. Check CORS settings
# In backend .env:
CORS_ORIGINS=*  # Allow all (development)
# Or specific:
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# 4. Check firewall
sudo ufw status
# Ensure ports 5000 and 3000 are allowed

# 5. Check Docker network
docker network inspect sparktrainer_default
# Ensure services are on same network

# 6. Restart both services
docker compose restart backend frontend
```

### API Returns 500 Error

**Problem:** API requests fail with 500 Internal Server Error.

**Solutions:**

```bash
# 1. Check backend logs
docker compose logs backend | tail -50

# 2. Enable debug mode
# In .env:
FLASK_DEBUG=1
FLASK_ENV=development

# 3. Check database connection
docker compose exec backend python -c "from backend.database import SessionLocal; db = SessionLocal(); print('DB OK')"

# 4. Check Redis connection
docker compose exec backend python -c "import redis; r = redis.from_url('redis://redis:6379/0'); print(r.ping())"

# 5. Restart backend
docker compose restart backend

# 6. Check for Python errors
docker compose logs backend | grep -i "error\|exception\|traceback" | tail -20
```

### Authentication Not Working

**Problem:** Cannot log in or "Unauthorized" errors.

**Solutions:**

```bash
# 1. Check if auth is enabled
# In .env:
ENABLE_AUTH=true  # Or false

# 2. Verify JWT secret is set
cat .env | grep JWT_SECRET_KEY
# Should be a long random string

# 3. Check token expiration
# Tokens expire after 24 hours by default
# Log out and log back in

# 4. Clear browser cookies
# DevTools → Application → Cookies → Clear

# 5. Check user exists
docker compose exec backend python << EOF
from backend.database import SessionLocal
from backend.models import User
db = SessionLocal()
users = db.query(User).all()
for u in users:
    print(f"User: {u.username}, Admin: {u.is_admin}")
EOF

# 6. Reset password
docker compose exec backend python << EOF
from backend.database import SessionLocal
from backend.models import User
from werkzeug.security import generate_password_hash

db = SessionLocal()
user = db.query(User).filter_by(username="admin").first()
user.password_hash = generate_password_hash("newpassword")
db.commit()
print("Password reset!")
EOF
```

---

## Dataset Problems

### Dataset Upload Fails

**Problem:** Cannot upload files or dataset creation fails.

**Solutions:**

```bash
# 1. Check file size limits
# In backend/app.py, increase if needed:
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# 2. Check disk space
df -h

# 3. Check permissions
ls -la datasets/
# Should be writable by Docker user

# 4. Check logs
docker compose logs backend | grep -i "upload\|dataset"

# 5. Try smaller files first
# Upload 1-2 files to test

# 6. Use CLI instead of UI
# Copy files directly:
docker compose exec backend mkdir -p /app/datasets/my_dataset
docker cp mydata.jsonl sparktrainer_backend_1:/app/datasets/my_dataset/
```

### Video Processing Fails

**Problem:** Video wizard fails to process videos.

**Solutions:**

```bash
# 1. Check FFmpeg is installed
docker compose exec backend ffmpeg -version

# 2. Check video file is valid
docker compose exec backend ffprobe /path/to/video.mp4

# 3. Check logs for specific error
docker compose logs worker | grep -i "video\|ffmpeg"

# 4. Try a different video codec
# Convert to standard format:
ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4

# 5. Reduce FPS in wizard
# Try FPS=1 instead of 30

# 6. Check GPU is available for video processing
nvidia-smi

# 7. Manually process video
docker compose exec worker python << EOF
from spark_trainer.ingestion.video_wizard import process_video
process_video("/path/to/video.mp4", output_dir="/tmp/test", fps=1)
EOF
```

### Dataset Not Showing Up

**Problem:** Dataset exists but doesn't appear in UI.

**Solutions:**

```bash
# 1. Check database
docker compose exec backend python << EOF
from backend.database import SessionLocal
from backend.models import Dataset
db = SessionLocal()
datasets = db.query(Dataset).all()
for d in datasets:
    print(f"Dataset: {d.name}, Path: {d.path}")
EOF

# 2. Refresh browser
# Hard refresh: Ctrl+Shift+R

# 3. Check API directly
curl http://localhost:5000/api/datasets

# 4. Check file permissions
docker compose exec backend ls -la /app/datasets/

# 5. Re-scan datasets
docker compose exec backend python << EOF
from backend.app import scan_datasets
scan_datasets()
EOF

# 6. Restart backend
docker compose restart backend
```

---

## Model Issues

### Model Download Fails

**Problem:** Cannot download model from HuggingFace.

**Solutions:**

```bash
# 1. Check internet connection
docker compose exec backend curl -I https://huggingface.co

# 2. Check HuggingFace is accessible
docker compose exec backend python << EOF
from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-uncased")
print("Download successful!")
EOF

# 3. Set HuggingFace cache directory
# In .env:
HF_HOME=/app/cache/huggingface

# 4. Use HuggingFace token for private models
# In .env:
HUGGINGFACE_TOKEN=hf_your_token_here

# 5. Download model manually
docker compose exec backend python << EOF
from huggingface_hub import snapshot_download
snapshot_download("meta-llama/Llama-2-7b-hf", local_dir="/app/models/llama-2-7b")
EOF

# 6. Check disk space
df -h
```

### Model Export Fails

**Problem:** Cannot export model to HuggingFace Hub.

**Solutions:**

```bash
# 1. Verify HuggingFace token
docker compose exec backend python << EOF
from huggingface_hub import login
login(token="hf_your_token")
print("Token valid!")
EOF

# 2. Check token has write permissions
# Go to https://huggingface.co/settings/tokens
# Ensure "Write" permission is enabled

# 3. Check model files exist
ls -la models/job_*/

# 4. Manually export
docker compose exec backend python << EOF
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="/app/models/job_123",
    repo_id="username/model-name",
    repo_type="model",
    token="hf_your_token"
)
EOF

# 5. Check repository exists
# Create repository first at:
# https://huggingface.co/new
```

### Model Inference Fails

**Problem:** Cannot load or run inference with trained model.

**Solutions:**

```python
# 1. Check model files are complete
import os
model_path = "./models/job_123"
required_files = ["pytorch_model.bin", "config.json", "tokenizer_config.json"]
for f in required_files:
    path = os.path.join(model_path, f)
    print(f"{f}: {'✓' if os.path.exists(path) else '✗'}")

# 2. Load model with error handling
from transformers import AutoModel, AutoTokenizer
try:
    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error: {e}")

# 3. Check if LoRA adapters need merging
# If you trained with LoRA:
from peft import PeftModel, PeftConfig
config = PeftConfig.from_pretrained(model_path)
base_model = AutoModel.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(base_model, model_path)
model = model.merge_and_unload()  # Merge adapters

# 4. Check CUDA/device
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# 5. Test with simple input
inputs = tokenizer("Hello world", return_tensors="pt").to(device)
outputs = model(**inputs)
print("Inference successful!")
```

---

## Getting Help

### Collect Debug Information

Before asking for help, gather this information:

```bash
# 1. System information
uname -a
docker --version
nvidia-smi  # If using GPU

# 2. SparkTrainer version
cd SparkTrainer
git log -1 --oneline

# 3. Service status
docker compose ps

# 4. Recent logs
docker compose logs --tail=100 backend > backend.log
docker compose logs --tail=100 worker > worker.log
docker compose logs --tail=100 postgres > postgres.log

# 5. Configuration (redact secrets!)
cat .env | grep -v "SECRET\|PASSWORD\|TOKEN"

# 6. API health
curl http://localhost:5000/api/health
curl http://localhost:5000/api/system/info
```

### Where to Get Help

1. **Documentation**
   - [README](../README.md)
   - [Installation Guide](INSTALLATION.md)
   - [FAQ](#faq-below)

2. **GitHub**
   - [Search Issues](https://github.com/def1ant1/SparkTrainer/issues)
   - [Create New Issue](https://github.com/def1ant1/SparkTrainer/issues/new)
   - [GitHub Discussions](https://github.com/def1ant1/SparkTrainer/discussions)

3. **Community**
   - Discord: [Coming Soon]
   - Reddit: r/SparkTrainer [Coming Soon]

---

## FAQ

### Q: Can I use SparkTrainer without a GPU?

**A:** Yes! SparkTrainer works on CPU. Training will be slower (10-100x) but everything else works normally.

### Q: How much GPU memory do I need?

**A:** Depends on model size:
- **4GB**: Small models (BERT, DistilBERT) with batch size 2-4
- **8GB**: 7B parameter models with QLoRA
- **16GB**: 7B models with full fine-tuning, 13B with QLoRA
- **24GB+**: 13B+ models with full fine-tuning

### Q: Training is stuck at "pending" status

**A:** Check Celery worker is running:
```bash
docker compose ps worker  # Should be "Up"
docker compose logs worker  # Check for errors
docker compose restart worker  # Restart if needed
```

### Q: Can I pause and resume training?

**A:** Yes! Use the Pause button in the UI or:
```bash
curl -X POST http://localhost:5000/api/jobs/<id>/pause
curl -X POST http://localhost:5000/api/jobs/<id>/resume
```

### Q: Where are my trained models saved?

**A:** Models are saved in:
- Docker: `/app/models/job_<id>/`
- Local: `./models/job_<id>/`
- MLflow: Check MLflow UI at http://localhost:5001

### Q: Can I use multiple GPUs?

**A:** Yes! Set in `.env`:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3  # Use GPUs 0-3
```
And enable distributed training in your job config.

### Q: How do I update SparkTrainer?

**A:**
```bash
cd SparkTrainer
git pull
docker compose down
docker compose build
docker compose up -d
```

### Q: Can I use SparkTrainer in production?

**A:** Yes! See [Deployment Guide](DEPLOYMENT.md) for production setup with:
- Reverse proxy (Nginx)
- SSL certificates
- Authentication
- Monitoring
- Backup strategies

---

**Still need help?** [Open an issue](https://github.com/def1ant1/SparkTrainer/issues/new) with:
- Description of problem
- Steps to reproduce
- Error messages/logs
- System information
