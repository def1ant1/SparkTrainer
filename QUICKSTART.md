# SparkTrainer Quick Start Guide

Get SparkTrainer up and running in **5 minutes** and train your first AI model!

## Prerequisites Check

Before starting, verify you have:

### Required
- **Docker** installed ([Get Docker](https://docs.docker.com/get-docker/))
- **Docker Compose** installed (usually comes with Docker)

### Recommended
- **NVIDIA GPU** with CUDA 11.8+ drivers
- **16GB+ RAM**
- **50GB+ free disk space**

### Verify Your Setup

```bash
# Check Docker
docker --version
docker-compose --version

# Check NVIDIA GPU (if you have one)
nvidia-smi
```

Expected output for nvidia-smi:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.x   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
...
```

> **Don't have a GPU?** No problem! SparkTrainer works on CPU too (just slower for training).

---

## Step 1: Get SparkTrainer (30 seconds)

```bash
# Clone the repository
git clone https://github.com/def1ant1/SparkTrainer.git
cd SparkTrainer
```

---

## Step 2: Start SparkTrainer (2 minutes)

### Option A: One Command (Easiest!)

```bash
# Start everything with Docker Compose
docker-compose up -d

# Wait for services to start (about 60 seconds)
# You'll see: Creating sparktrainer_postgres_1 ... done
#            Creating sparktrainer_redis_1    ... done
#            Creating sparktrainer_mlflow_1   ... done
#            Creating sparktrainer_backend_1  ... done
#            Creating sparktrainer_frontend_1 ... done

# Initialize database with sample data
docker-compose exec backend python init_db.py --sample-data

# You're done! SparkTrainer is running!
```

### Option B: Step-by-Step (For Development)

If you want to run services separately or modify code:

```bash
# 1. Start infrastructure (PostgreSQL, Redis, MLflow)
docker-compose up -d postgres redis mlflow

# 2. Set up Python backend
cd backend
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Initialize database
python init_db.py --sample-data

# 4. Start backend (Terminal 1)
python app.py

# 5. Start worker for training jobs (Terminal 2)
celery -A celery_app.celery worker --loglevel=info --pool=solo

# 6. Start frontend (Terminal 3)
cd ../frontend
npm install
npm run dev
```

---

## Step 3: Access SparkTrainer (10 seconds)

Open your browser and go to:

### 🎨 Main Application
**http://localhost:3000**

You should see the SparkTrainer dashboard!

### Other Interfaces (Optional)
- 📊 **MLflow (Experiment Tracking)**: http://localhost:5001
- 🌸 **Flower (Task Monitor)**: http://localhost:5555
- 🔌 **Backend API**: http://localhost:5000/api/health

---

## Step 4: Train Your First Model (3 minutes)

Now let's train a simple model to verify everything works!

### In the Web Interface:

1. **Go to Dashboard**
   - You should see "Welcome to SparkTrainer" at the top
   - Notice the "Create Job" button? Click it!

2. **Training Wizard Opens - Step 1: Select Data**
   - **Project**: Select "Demo Project" (created with sample data)
   - **Dataset**: Select "Sample Dataset" (pre-loaded for you)
   - Click **Next**

3. **Step 2: Choose Model**
   - Select **"From HuggingFace"**
   - Model ID: Enter `distilbert-base-uncased` (a small, fast model)
   - Click **Next**

4. **Step 3: Pick Recipe**
   - Select **"LoRA"** (efficient fine-tuning)
   - Click **Next**

5. **Step 4: Configure**
   - **Epochs**: 1 (just to test!)
   - **Batch Size**: 4
   - **Learning Rate**: Use suggested value (2e-4)
   - **LoRA Rank**: 8
   - Click **Launch Training**

6. **Watch It Train!**
   - You'll be redirected to the Job Details page
   - Watch real-time logs streaming
   - See GPU/CPU utilization
   - Training should complete in 1-3 minutes

### Via Command Line (Alternative):

```python
# Save this as test_train.py
from spark_trainer.recipes.lora_recipes import create_lora_recipe

config = {
    "base_model": "distilbert-base-uncased",
    "dataset_name": "sample_dataset",
    "lora_r": 8,
    "lora_alpha": 16,
    "learning_rate": 2e-4,
    "num_epochs": 1,
    "batch_size": 4,
    "output_dir": "./test_model"
}

recipe = create_lora_recipe(config, use_qlora=False)
recipe.run()

print("🎉 Training complete! Model saved to ./test_model")
```

Run it:
```bash
python test_train.py
```

---

## Step 5: Explore SparkTrainer (2 minutes)

Now that you have a trained model, explore the interface:

### Dashboard
- View system metrics (GPU, memory, jobs)
- See active and completed training jobs
- Quick access to create new jobs

### Jobs
- List all training jobs
- Click any job to see:
  - Real-time logs
  - Training metrics (loss, accuracy)
  - GPU utilization
  - Checkpoints

### Datasets
- View all datasets
- Upload new data
- See dataset statistics and samples

### Models
- Browse trained models
- Download or export to HuggingFace
- View model cards and metadata

### Leaderboard
- Compare model performance
- See rankings across benchmarks
- Filter by model type

### Experiments
- Track multiple training runs
- Compare hyperparameters
- Visualize metrics over time

---

## Common First Steps

### Add Your Own Dataset

**For Images:**
```bash
# Create dataset directory
mkdir -p datasets/my_images/images

# Copy your images
cp /path/to/your/images/* datasets/my_images/images/

# Create manifest (optional)
echo '{"name": "my_images", "type": "image"}' > datasets/my_images/manifest.json
```

**For Text:**
```bash
# Create JSONL file
cat > datasets/my_text/data.jsonl << EOF
{"text": "First training example", "label": "positive"}
{"text": "Second training example", "label": "negative"}
EOF
```

Then refresh the Datasets page in the UI!

### Train on Your Data

1. Go to **Dashboard** → **Create Job**
2. Select your new dataset
3. Choose a model (try `bert-base-uncased` for text)
4. Pick LoRA recipe
5. Launch!

### Monitor Training

Real-time monitoring options:
```bash
# Option 1: Watch logs
tail -f logs/job_*.log

# Option 2: Use the Dashboard
# Just click on your job in the Jobs page!

# Option 3: Use MLflow
# Open http://localhost:5001
```

### Export Your Model

After training completes:

**Via UI:**
1. Go to **Models** page
2. Click your model
3. Click **Download** or **Export to HuggingFace**

**Via CLI:**
```bash
# Find your model in the models/ directory
ls models/

# Use it with transformers
python
>>> from transformers import AutoModel
>>> model = AutoModel.from_pretrained("./models/job_123")
```

---

## Troubleshooting

### Ports Already in Use

If you see "port already allocated":

```bash
# Check what's using the port
sudo lsof -i :5000  # Backend
sudo lsof -i :3000  # Frontend

# Kill the process or change ports in docker-compose.yml
```

### Docker Containers Won't Start

```bash
# Check container logs
docker-compose logs backend
docker-compose logs postgres
docker-compose logs redis

# Restart all services
docker-compose down
docker-compose up -d
```

### GPU Not Detected

```bash
# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# If that fails, install NVIDIA Container Toolkit:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Out of Memory (OOM) Errors

If training fails with OOM:

1. **Reduce batch size**: Try 2 or 1
2. **Use QLoRA**: Enable 4-bit quantization
3. **Enable gradient checkpointing**
4. **Use a smaller model**: Try `distilbert` instead of `bert`

### Training is Slow

**On CPU?** Training is 10-100x slower without GPU.

**Speed it up:**
- Use a smaller model for testing
- Reduce dataset size
- Lower the number of epochs
- Consider cloud GPU (AWS, GCP, Lambda Labs)

### Database Connection Failed

```bash
# Restart PostgreSQL
docker-compose restart postgres

# Check if it's running
docker-compose ps postgres

# View logs
docker-compose logs postgres
```

---

## Next Steps

Now that SparkTrainer is running, dive deeper:

### Tutorials
- 📖 **[Complete Tutorial](docs/TUTORIAL.md)**: Full walkthrough with real examples
- 🎯 **[LoRA Guide](docs/app/recipes/lora.md)**: Master efficient fine-tuning
- 👁️ **[Vision Models](docs/VISION_LANGUAGE.md)**: Train multimodal models
- 📊 **[Datasets Guide](docs/DATASETS.md)**: Advanced data preparation

### Documentation
- 📘 **[Installation Guide](docs/INSTALLATION.md)**: Detailed setup instructions
- 🛠️ **[Configuration](docs/CONFIGURATION.md)**: All settings explained
- 🔌 **[API Reference](docs/api.md)**: REST API documentation
- ❓ **[Troubleshooting](docs/TROUBLESHOOTING.md)**: Solve common issues

### Community
- 💬 **[GitHub Discussions](https://github.com/def1ant1/SparkTrainer/discussions)**: Ask questions
- 🐛 **[Report Issues](https://github.com/def1ant1/SparkTrainer/issues)**: Found a bug?
- 🤝 **[Contributing](CONTRIBUTING.md)**: Help improve SparkTrainer

---

## Quick Reference Commands

### Start/Stop
```bash
# Start everything
docker-compose up -d

# Stop everything
docker-compose down

# Restart a service
docker-compose restart backend

# View logs
docker-compose logs -f backend
```

### Database
```bash
# Reset database
docker-compose exec backend python init_db.py --reset --sample-data

# Backup database
docker-compose exec postgres pg_dump -U sparktrainer sparktrainer > backup.sql

# Restore database
cat backup.sql | docker-compose exec -T postgres psql -U sparktrainer sparktrainer
```

### Training
```bash
# List all jobs
curl http://localhost:5000/api/jobs

# Get job details
curl http://localhost:5000/api/jobs/123

# Cancel a job
curl -X POST http://localhost:5000/api/jobs/123/cancel
```

---

## Success! 🎉

You've successfully:
- ✅ Installed SparkTrainer
- ✅ Started all services
- ✅ Trained your first model
- ✅ Explored the interface

**You're ready to build amazing AI models!**

Need help? Check the [full documentation](README.md) or [ask the community](https://github.com/def1ant1/SparkTrainer/discussions).

Happy training! 🚀
