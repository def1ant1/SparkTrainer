# SparkTrainer 🚀

**The Complete MLOps Platform for Modern AI Development**

SparkTrainer is an enterprise-grade, open-source machine learning training platform that makes GPU-accelerated AI model training accessible to everyone. Whether you're a researcher experimenting with new architectures, a data scientist building production models, or an ML team managing complex workflows, SparkTrainer provides everything you need in one unified, easy-to-use platform.

## What is SparkTrainer?

SparkTrainer simplifies the entire ML lifecycle:
- **For Beginners**: Start training state-of-the-art models in minutes with intuitive wizards and pre-built recipes
- **For Researchers**: Track experiments, compare results, and iterate faster with comprehensive MLflow integration
- **For Teams**: Collaborate on datasets, share models, and manage GPU resources efficiently
- **For Production**: Deploy models with confidence using A/B testing, model registry, and safety evaluations

No complex setup, no scattered tools, no headaches. Just one platform that does it all.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![CI/CD](https://github.com/def1ant1/SparkTrainer/actions/workflows/ci.yml/badge.svg)](https://github.com/def1ant1/SparkTrainer/actions/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

## 🆕 Latest MLOps Enhancements

SparkTrainer now includes comprehensive production-ready features:

- **📦 Data Ingestion & Versioning**: JSONL manifests, lakeFS/DVC integration, quality gates (dedupe, PII redaction, toxicity filtering), auto-generated dataset cards
- **🔒 Safety & Evaluation**: Toxicity/bias/jailbreak probes, calibration curves, comprehensive safety reports
- **🎯 Model Registry**: Lifecycle management (Dev→Staging→Production), promotion gates with approval workflows, signed model bundles
- **⚡ Inference Adapters**: vLLM, TGI, Triton, TorchServe support with unified interface
- **🧪 A/B Testing**: Shadow deployment, canary releases, statistical significance testing
- **🖥️ GPU Scheduling**: Smart placement, MIG awareness, auto-resume from checkpoints
- **📚 Recipe Templates**: Standardized workflows for LLM/Vision/Audio/Video with efficiency toggles

**📖 [Full MLOps Documentation](./docs/MLOPS_ENHANCEMENTS.md)**

## ✨ Key Features

### 🎯 **Job & Tracking Foundation**
- **Distributed Task Queue**: Celery + Redis for async job processing
- **Database-Backed**: PostgreSQL with comprehensive schema (projects, experiments, datasets, jobs, artifacts)
- **Job State Machine**: Validated status transitions with full audit trail
- **MLflow Integration**: Centralized experiment tracking with artifact storage
- **Real-time Monitoring**: Flower UI for worker health and task monitoring

### 📹 **Data Ingestion (Video-First)**
- **Video Wizard**: 4-step guided workflow for dataset creation
- **Automated Processing**:
  - Frame extraction (configurable FPS, resolution)
  - Audio extraction & Whisper transcription
  - Multi-backend captioning (BLIP-2, InternVL, Qwen2-VL, Florence-2)
  - Scene detection with PySceneDetect
- **Integrity Checks**: Video validation, corruption detection, format verification
- **Manifest Generation**: JSONL format with complete metadata

### 🧠 **Trainer Unification + LoRA**
- **Recipe Interface**: Standardized workflow (prepare → build → train → eval → package)
- **LoRA/QLoRA Support**: Parameter-efficient fine-tuning with 4-bit quantization
- **Auto-target Detection**: Automatically identify attention layers for LoRA
- **FSDP Ready**: Fully Sharded Data Parallel for large models
- **Multiple Backends**: PyTorch, HuggingFace Transformers, DeepSpeed

### 📊 **UX Dashboards & Evaluations**
- **Project Organization**: Hierarchical structure (Projects → Datasets → Experiments)
- **Leaderboard**: Multi-benchmark rankings with top-3 highlighting
- **MMLU Evaluation**: 57 subjects across STEM, humanities, social sciences
- **COCO Evaluation**: Captioning (BLEU, CIDEr, SPICE), detection, segmentation
- **Live Metrics**: Real-time training progress (WebSocket-ready)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Frontend (React + Vite)                     │
│  Projects | Datasets | Experiments | Leaderboard        │
└────────────────────────┬────────────────────────────────┘
                         │ REST API
                         ▼
┌─────────────────────────────────────────────────────────┐
│                 Backend (Flask)                          │
│  API • Job Management • MLflow Integration              │
└─────┬──────────────────┬──────────────┬────────────────┘
      │                  │              │
      ▼                  ▼              ▼
  PostgreSQL          Redis          MLflow
  (Metadata)       (Queue/Cache)   (Experiments)
      │                  │              │
      └──────────┬───────┴──────────────┘
                 ▼
      ┌──────────────────────┐
      │   Celery Workers     │
      │  - Training (GPU)    │
      │  - Preprocessing     │
      │  - Evaluation        │
      └──────────────────────┘
```

## 🚀 Quick Start - Get Running in 10 Minutes!

### Prerequisites

Before you begin, make sure you have:
- **Docker & Docker Compose** ([Install Guide](https://docs.docker.com/get-docker/))
- **NVIDIA GPU** with CUDA 11.8+ drivers ([Check with `nvidia-smi`](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/))
- **16GB+ RAM** (32GB recommended for training larger models)
- **50GB+ free disk space** (more if working with large datasets)

> **Don't have a GPU?** You can still use SparkTrainer for smaller models and CPU-based training!

### Installation (Two Options)

#### Option 1: Docker Compose (Recommended - Fastest Setup!)

Perfect for getting started quickly or running in production.

```bash
# 1. Clone and enter directory
git clone https://github.com/def1ant1/SparkTrainer.git
cd SparkTrainer

# 2. Start everything with one command!
docker-compose up -d

# 3. Initialize the database with sample data
docker-compose exec backend python init_db.py --sample-data

# 4. You're done! Open your browser
```

**Access the application:**
- 🎨 **Web Interface**: http://localhost:3000
- 📊 **MLflow (Experiments)**: http://localhost:5001
- 🌸 **Flower (Task Queue)**: http://localhost:5555
- 🔌 **API Documentation**: http://localhost:5000/api/docs

#### Option 2: Local Development Setup

Better for development and customization.

```bash
# 1. Clone the repository
git clone https://github.com/def1ant1/SparkTrainer.git
cd SparkTrainer

# 2. Start infrastructure services (PostgreSQL, Redis, MLflow)
docker-compose up -d postgres redis mlflow

# 3. Set up Python environment
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 4. Initialize database
python init_db.py --sample-data

# 5. Start backend (in Terminal 1)
python app.py

# 6. Start Celery worker for training jobs (in Terminal 2)
celery -A celery_app.celery worker --loglevel=info --pool=solo

# 7. Start frontend (in Terminal 3)
cd ../frontend
npm install
npm run dev
```

**Access the application:**
- 🎨 **Frontend**: http://localhost:3000
- 📊 **MLflow**: http://localhost:5001
- 🌸 **Flower**: http://localhost:5555
- 🔌 **Backend API**: http://localhost:5000

### Your First Training Job (2 Minutes!)

Once SparkTrainer is running, let's train your first model:

1. **Open** http://localhost:3000 in your browser
2. **Click** "Dashboard" → "Create Job" or use the "Training Wizard" button
3. **Select** a pre-loaded sample dataset (comes with `--sample-data`)
4. **Choose** a base model (e.g., `bert-base-uncased` for NLP or `resnet18` for vision)
5. **Pick** a recipe template (try "LoRA" for efficient fine-tuning!)
6. **Click** "Launch Training" and watch the magic happen!

You'll see real-time:
- GPU utilization and memory usage
- Training loss decreasing
- Live logs streaming
- Progress bars and ETA

**That's it!** Your model will be saved automatically when training completes.

### Need Help?

- 📖 **Detailed Guide**: See [INSTALLATION.md](INSTALLATION.md) for step-by-step instructions
- 🎓 **Complete Tutorial**: Check [docs/TUTORIAL.md](docs/TUTORIAL.md) for a full walkthrough
- ❓ **Troubleshooting**: Visit [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- 💬 **Community**: [GitHub Discussions](https://github.com/def1ant1/SparkTrainer/discussions)

## 📖 Complete User Guide

### Understanding the SparkTrainer Workflow

SparkTrainer organizes your ML work into a logical hierarchy:
- **Projects**: Group related work (e.g., "Customer Sentiment Analysis")
- **Datasets**: Your training data with versioning
- **Experiments**: Training runs with tracked parameters
- **Models**: Trained model artifacts and checkpoints

### 1. Create Your First Project

Projects help organize your work. Think of them like folders.

**Via Web UI:**
1. Click **Projects** in the sidebar
2. Click **+ New Project**
3. Enter:
   - **Name**: e.g., "Image Classification"
   - **Description**: "Building a cat vs dog classifier"
4. Click **Create**

**Via API:**
```bash
curl -X POST http://localhost:5000/api/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "Image Classification", "description": "Cat vs dog classifier"}'
```

### 2. Prepare Your Dataset

SparkTrainer supports multiple data types: images, videos, text, audio, and multimodal.

#### Option A: Video Dataset (with automatic preprocessing)

Perfect for vision-language models. The wizard does all the heavy lifting!

1. Navigate to **Datasets** → **Dataset Wizard**
2. **Step 1 - Upload**: Drag & drop your video files or select a folder
3. **Step 2 - Configure**:
   - **FPS**: Frames to extract per second (1-30). Higher = more frames, larger dataset
   - **Resolution**: Target size (e.g., 224x224 for most models)
   - **Audio Transcription**: Enable to extract speech-to-text using Whisper
   - **Captioning**: Choose a backend:
     - **BLIP-2**: Fast, good quality (recommended for beginners)
     - **InternVL**: State-of-the-art, slower
     - **Qwen2-VL**: Great for detailed descriptions
     - **Florence-2**: Best for object detection
   - **Scene Detection**: Auto-split videos at scene changes (optional)
4. **Step 3 - Integrity Check**: SparkTrainer validates all files
5. **Step 4 - Process**: Click "Start Processing" and grab coffee!

The system creates a `manifest.jsonl` file with all metadata automatically.

#### Option B: Image Dataset

1. Go to **Datasets** → **+ New Dataset**
2. Choose **Image** as type
3. Upload your images (supports .jpg, .png, .webp)
4. Optionally add labels or captions
5. Click **Create**

#### Option C: Text Dataset

1. Upload a JSONL file with format:
```json
{"text": "Your training text here", "label": "category"}
{"text": "Another example", "label": "category2"}
```

2. Or use a HuggingFace dataset:
```python
from spark_trainer.data import load_dataset
dataset = load_dataset("huggingface_dataset_name")
```

#### Option D: Bring Your Own Data

Place files in `datasets/` folder:
```
datasets/
  my_dataset/
    images/
      img1.jpg
      img2.jpg
    manifest.jsonl  # Optional metadata
```

### 3. Train a Model

#### Easy Mode: Web UI Training Wizard

1. Go to **Dashboard** → **Create Job** (big button!)
2. **Training Wizard** opens with 4 steps:

**Step 1 - Select Data:**
- Choose your project
- Pick a dataset
- Preview samples to verify

**Step 2 - Choose Model:**
- **From HuggingFace**: Enter any model ID (e.g., `bert-base-uncased`, `meta-llama/Llama-2-7b-hf`)
- **From Local**: Use a previously trained model
- **From Scratch**: Design custom architecture

**Step 3 - Pick Recipe:**
- **LoRA/QLoRA**: Efficient fine-tuning (recommended for large models!)
- **Full Fine-tune**: Train all parameters
- **Vision-Language**: For image+text models
- **Audio-Video**: For multimodal models
- **Text-only**: For language models

**Step 4 - Configure:**
- **Batch Size**: 4-32 (smaller for GPUs with less memory)
- **Learning Rate**: Auto-suggested based on model
- **Epochs**: 3-10 (more = better fit, risk overfitting)
- **LoRA Rank**: 8-16 (for LoRA recipe)
- **Quantization**: 4-bit/8-bit to save memory

Click **Launch Training**!

#### Advanced Mode: Python API

For automation and custom workflows:

```python
from spark_trainer.recipes.lora_recipes import create_lora_recipe

# Configure your training
config = {
    "base_model": "meta-llama/Llama-2-7b-hf",
    "dataset_name": "my_dataset",
    "lora_r": 8,              # LoRA rank
    "lora_alpha": 16,          # LoRA alpha
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "output_dir": "./my_model"
}

# Use QLoRA for 4-bit quantization (saves 75% GPU memory!)
recipe = create_lora_recipe(config, use_qlora=True)

# Start training
recipe.run()
```

**Other recipe types:**
```python
# Vision-Language model
from spark_trainer.recipes.vision_recipes import create_vision_language_recipe
recipe = create_vision_language_recipe(config)

# Text generation
from spark_trainer.recipes.text_recipes import create_text_generation_recipe
recipe = create_text_generation_recipe(config)
```

### 4. Monitor Training in Real-Time

While your model trains, you can watch everything:

**Via Dashboard:**
- Go to **Jobs** page
- Click on your running job
- See live updates every few seconds:
  - 📈 Training loss (should decrease!)
  - 🎯 Validation accuracy
  - 🖥️ GPU utilization & memory
  - ⚡ Power consumption
  - ⏱️ ETA to completion
  - 📝 Live training logs

**Via MLflow:**
- Open http://localhost:5001
- Find your experiment
- See detailed metrics, charts, and comparisons

**Via CLI:**
```bash
# Watch job logs in real-time
tail -f logs/job_<id>.log
```

### 5. Evaluate Your Model

After training, evaluate performance on standard benchmarks:

#### Language Models: MMLU Benchmark

Tests your model on 57 subjects from STEM to humanities.

**Via UI:**
1. Go to **Models** page
2. Find your trained model
3. Click **Evaluate** → **MMLU**
4. Select subjects (or "All")
5. Choose few-shot setting (0, 1, 5)
6. Click **Run Evaluation**

**Via CLI:**
```bash
python -m spark_trainer.evaluation.mmlu_eval \
    --model-path ./my_model \
    --output-dir ./eval_results \
    --num-fewshot 5 \
    --subjects abstract_algebra anatomy astronomy biology
```

#### Vision Models: COCO Benchmark

Tests captioning, object detection, and segmentation.

**Via CLI:**
```bash
python -m spark_trainer.evaluation.coco_eval \
    --model-path ./my_vision_model \
    --task captioning \
    --max-samples 1000 \
    --output-dir ./eval_results
```

Results automatically appear on the **Leaderboard**!

### 6. View Results on the Leaderboard

Compare your models against each other:

1. Navigate to **Leaderboard**
2. Filter by benchmark type (MMLU, COCO, etc.)
3. See rankings with medals 🥇🥈🥉
4. Click any model to see detailed metrics
5. Export results to CSV for analysis

### 7. Export and Deploy Your Model

#### Export to HuggingFace Hub

Share your model with the world!

**Via UI:**
1. Go to **Models** → Your model
2. Click **Export to HuggingFace**
3. Enter:
   - HuggingFace username
   - Repository name
   - Authentication token
4. Click **Export**

Model will be public at `huggingface.co/{username}/{repo_name}`

**Via CLI:**
```bash
python -m spark_trainer.export \
    --model-path ./my_model \
    --hf-repo username/model-name \
    --token your_hf_token
```

#### Download for Local Use

**Via UI:**
1. Go to **Models** → Your model
2. Click **Download as ZIP**
3. Extract and use with:

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("./my_model")
tokenizer = AutoTokenizer.from_pretrained("./my_model")

# Use for inference
outputs = model.generate(...)
```

#### Deploy for Inference

SparkTrainer supports multiple serving backends:

```python
from spark_trainer.inference.serving_adapters import VLLMAdapter

# Deploy with vLLM (fast inference server)
adapter = VLLMAdapter(
    model_path="./my_model",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9
)

adapter.serve(port=8000)
```

Access at http://localhost:8000 with OpenAI-compatible API!

## 🛠️ Configuration Reference

### Environment Variables Explained

Create a `.env` file in the project root. Here's what each variable does:

```bash
# ======================
# Database Configuration
# ======================
DATABASE_URL=postgresql://sparktrainer:password@localhost:5432/sparktrainer
# Where: postgresql://[user]:[password]@[host]:[port]/[database]
# Used for: Storing experiments, jobs, models, datasets metadata

# ======================
# Redis Configuration
# ======================
REDIS_URL=redis://localhost:6379/0
# Used for: Caching and session storage

CELERY_BROKER_URL=redis://localhost:6379/0
# Used for: Task queue message broker (distributing jobs to workers)

CELERY_RESULT_BACKEND=redis://localhost:6379/0
# Used for: Storing task results and status

# ======================
# MLflow Configuration
# ======================
MLFLOW_TRACKING_URI=http://localhost:5001
# Where MLflow server runs for experiment tracking

MLFLOW_ARTIFACT_ROOT=./mlruns
# Local storage for model artifacts and logs

# ======================
# Training Configuration
# ======================
DGX_TRAINER_BASE_DIR=/app
# Base directory for training jobs

CUDA_VISIBLE_DEVICES=0,1,2,3
# Which GPUs to use (0=first GPU, 0,1=first two GPUs, etc.)
# Leave empty to use all GPUs, set to "" to use CPU only

# ======================
# Flask Backend Configuration
# ======================
FLASK_ENV=development
# Options: development, production
# Use production for deployment!

FLASK_DEBUG=1
# Enable debug mode (0=off, 1=on)
# IMPORTANT: Set to 0 in production!

SECRET_KEY=your-secret-key-here
# Generate secure key with: python -c 'import secrets; print(secrets.token_hex(32))'
# Used for: Session encryption and security

API_HOST=0.0.0.0
# Listen on all interfaces (use 127.0.0.1 for localhost only)

API_PORT=5000
# Backend API port

# ======================
# Storage Configuration
# ======================
STORAGE_BACKEND=local
# Options: local, s3, gcs
# Where to store datasets and model files

LOCAL_STORAGE_PATH=./storage
# Path for local storage (when STORAGE_BACKEND=local)

# If using S3:
S3_BUCKET=your-bucket-name
S3_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret

# If using GCS:
GCS_BUCKET=your-bucket-name
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

# ======================
# Optional: Authentication
# ======================
ENABLE_AUTH=false
# Set to true to require login

JWT_SECRET_KEY=your-jwt-secret
# Separate secret for JWT tokens

JWT_ALGORITHM=HS256
# Encryption algorithm for JWT

# ======================
# Optional: Advanced Settings
# ======================
MAX_WORKERS=4
# Maximum concurrent training jobs

LOG_LEVEL=INFO
# Options: DEBUG, INFO, WARNING, ERROR

ENABLE_TELEMETRY=false
# Send anonymous usage stats (helps development!)
```

### Training Configuration Files

SparkTrainer uses YAML files for training configuration. See `configs/` directory:

**Example: LoRA Fine-tuning** (`configs/lora_example.yaml`)
```yaml
model:
  base_model: "meta-llama/Llama-2-7b-hf"
  use_qlora: true  # 4-bit quantization

lora:
  r: 8              # Rank (higher = more parameters)
  alpha: 16         # Scaling factor
  target_modules:   # Which layers to apply LoRA
    - q_proj
    - v_proj
  dropout: 0.05

training:
  num_epochs: 3
  batch_size: 4
  learning_rate: 2e-4
  gradient_accumulation_steps: 4
  warmup_steps: 100

dataset:
  name: "my_dataset"
  max_length: 2048

output:
  output_dir: "./output"
  save_steps: 500
  logging_steps: 10
```

**Example: Vision-Language** (`configs/train_vl_example.yaml`)
```yaml
model:
  architecture: "blip2"
  vision_model: "Salesforce/blip2-opt-2.7b"

training:
  num_epochs: 5
  batch_size: 8
  learning_rate: 1e-5

dataset:
  type: "image_text"
  image_size: 224
```

**Example: DeepSpeed for Large Models** (`configs/deepspeed_zero3.json`)
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    }
  },
  "fp16": {
    "enabled": true
  },
  "gradient_accumulation_steps": 4
}
```

For more examples, see [EXAMPLES.md](EXAMPLES.md)

### GPU Configuration Tips

**Single GPU:**
```bash
CUDA_VISIBLE_DEVICES=0  # Use first GPU
```

**Multi-GPU (Data Parallel):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3  # Use 4 GPUs
```

**CPU Only (for testing):**
```bash
CUDA_VISIBLE_DEVICES=""  # No GPU
```

**Memory Management:**
- Use QLoRA for models >7B parameters
- Reduce batch size if getting OOM errors
- Enable gradient checkpointing in training config
- Use DeepSpeed ZeRO-3 for models >13B parameters

## 📚 Comprehensive Documentation

### Getting Started
- 📘 **[INSTALLATION.md](docs/INSTALLATION.md)**: Detailed installation guide with screenshots
- 🚀 **[QUICKSTART.md](QUICKSTART.md)**: Get running in 5 minutes
- 🎓 **[TUTORIAL.md](docs/TUTORIAL.md)**: Complete walkthrough with examples
- ❓ **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)**: Common issues and solutions

### User Guides
- 👤 **[USER_GUIDE.md](docs/USER_GUIDE.md)**: Complete user manual
- 📊 **[DATASETS.md](docs/DATASETS.md)**: Working with datasets
- 🧠 **[TRAINING.md](docs/TRAINING.md)**: Training models effectively
- 📈 **[EVALUATION.md](docs/EVALUATION.md)**: Evaluating and benchmarking
- 🚢 **[DEPLOYMENT.md](docs/DEPLOYMENT.md)**: Deploying models

### Recipes & Techniques
- 🎯 **[LoRA Guide](docs/app/recipes/lora.md)**: Efficient fine-tuning with LoRA
- 👁️ **[Vision-Language Models](docs/VISION_LANGUAGE.md)**: Training multimodal models
- 🎵 **[Audio-Video Models](docs/AUDIO_VIDEO.md)**: Working with audio and video

### Reference Documentation
- 🔌 **[API Reference](docs/api.md)**: Complete REST API documentation
- ⚙️ **[Configuration Reference](docs/CONFIGURATION.md)**: All settings explained
- 🏗️ **[Architecture Guide](docs/ARCHITECTURE_BUILDER_GUIDE.md)**: System architecture
- 🔒 **[Safety & Quality Gates](docs/GATING_MECHANISMS.md)**: Safety features

### Advanced Topics
- 🌐 **[Distributed Training](docs/distributed_training.md)**: Multi-GPU training
- 📊 **[MLOps Features](docs/MLOPS_ENHANCEMENTS.md)**: Production MLOps capabilities
- 🎯 **[Model Registry](docs/MODEL_REGISTRY.md)**: Managing model lifecycle
- 🧪 **[A/B Testing](docs/AB_TESTING.md)**: Model testing in production

### Development
- 👨‍💻 **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)**: Developer setup and architecture
- 🤝 **[CONTRIBUTING.md](CONTRIBUTING.md)**: How to contribute
- 📜 **[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)**: Community guidelines
- ⚡ **[FEATURES.md](FEATURES.md)**: Comprehensive feature list

## 🧪 Testing

```bash
# Run all tests with coverage (requires 80% coverage)
make test

# Quick test without coverage
make test-quick

# Run specific test
pytest tests/test_lora_recipes.py

# Run linting
make lint

# Format code
make format
```

See the [Makefile](Makefile) for all available development commands.

## 🔧 Troubleshooting

This section covers common issues and their solutions. If you don't find your issue here, check our [GitHub Discussions](https://github.com/def1ant1/SparkTrainer/discussions) or open an issue.

### GPU Memory Errors

**Problem**: `CUDA out of memory` or `RuntimeError: CUDA error: out of memory`

**Solutions**:

1. **Reduce Batch Size**:
   ```json
   {
     "hyperparameters": {
       "batch_size": 2,  // Try 1, 2, or 4
       "gradient_accumulation_steps": 8  // Increase to maintain effective batch size
     }
   }
   ```

2. **Enable Gradient Checkpointing**:
   ```json
   {
     "hyperparameters": {
       "gradient_checkpointing": true
     }
   }
   ```

3. **Use Quantization** (QLoRA):
   ```json
   {
     "recipe": "lora_qlora",  // Uses 4-bit quantization
     "hyperparameters": {
       "load_in_4bit": true
     }
   }
   ```

4. **Reduce Sequence Length**:
   ```json
   {
     "hyperparameters": {
       "max_length": 1024  // Instead of 2048 or 4096
     }
   }
   ```

5. **Monitor GPU Memory**:
   ```bash
   # Check GPU memory usage
   nvidia-smi

   # Watch GPU memory in real-time
   watch -n 1 nvidia-smi
   ```

6. **Clear GPU Cache** (in training script):
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Whisper Transcription Failures

**Problem**: Whisper fails to transcribe audio or produces garbled output

**Solutions**:

1. **Check Audio Format**:
   - Whisper works best with 16kHz sampling rate
   - Ensure audio is properly extracted from video
   ```bash
   # Convert audio to correct format
   ffmpeg -i input.mp4 -ar 16000 -ac 1 output.wav
   ```

2. **Use Appropriate Whisper Model**:
   - `tiny`: Fast but less accurate (39M params)
   - `base`: Good balance (74M params) - **Recommended for most use cases**
   - `small`: Better accuracy (244M params)
   - `medium`: High accuracy (769M params)
   - `large`: Best accuracy but slow (1550M params)

3. **Adjust Whisper Configuration**:
   ```json
   {
     "video_config": {
       "transcription": {
         "model": "base",  // Try "small" or "medium" for better quality
         "language": "en",  // Specify language for better results
         "fp16": false  // Disable FP16 if encountering NaN issues
       }
     }
   }
   ```

4. **Handle Long Audio**:
   - Whisper works best on chunks < 30 seconds
   - SparkTrainer automatically chunks audio, but you can adjust:
   ```json
   {
     "video_config": {
       "transcription": {
         "chunk_length_s": 30
       }
     }
   }
   ```

5. **Check Logs**:
   ```bash
   docker-compose logs celery-worker | grep -i whisper
   ```

### Dataset Loading Issues

**Problem**: Dataset fails to load or process

**Solutions**:

1. **Check Dataset Format**:
   - SparkTrainer expects JSONL manifest format
   - Each line should be valid JSON
   ```bash
   # Validate JSONL format
   cat dataset.jsonl | jq . > /dev/null
   ```

2. **Verify File Paths**:
   ```bash
   # Ensure paths in manifest are accessible
   ls -lh datasets/my-dataset/
   ```

3. **Check Disk Space**:
   ```bash
   df -h
   ```

4. **Inspect Dataset Samples**:
   ```bash
   # View first few samples
   head -n 5 datasets/my-dataset/manifest.jsonl | jq .
   ```

5. **Dataset Size Limits**:
   - For very large datasets (>1M samples), consider:
     - Splitting into smaller batches
     - Using streaming datasets
     - Increasing worker timeout

### Connection Refused / Cannot Connect to Backend

**Problem**: Frontend cannot connect to backend API

**Solutions**:

1. **Check Services are Running**:
   ```bash
   docker-compose ps

   # All services should show "Up"
   ```

2. **Check Logs**:
   ```bash
   # Backend logs
   docker-compose logs backend

   # Database logs
   docker-compose logs postgres
   ```

3. **Verify Port Mappings**:
   - Frontend: `http://localhost:3000`
   - Backend: `http://localhost:5000`
   - MLflow: `http://localhost:5001`
   - Flower: `http://localhost:5555`

4. **Restart Services**:
   ```bash
   docker-compose down
   docker-compose up -d
   ```

5. **Check Network**:
   ```bash
   # Test backend connectivity
   curl http://localhost:5000/api/health

   # Should return: {"status": "healthy"}
   ```

### Training Job Stuck in "Queued" Status

**Problem**: Jobs remain in "queued" state and never start

**Solutions**:

1. **Check Celery Workers**:
   ```bash
   # View worker status
   docker-compose logs celery-worker

   # Check Flower dashboard
   open http://localhost:5555
   ```

2. **Verify Redis Connection**:
   ```bash
   docker-compose logs redis
   ```

3. **Restart Workers**:
   ```bash
   docker-compose restart celery-worker
   ```

4. **Check Queue Depth**:
   - Visit Flower UI: `http://localhost:5555`
   - Look for backed-up queues

5. **Resource Availability**:
   - Ensure GPUs are available
   - Check if other jobs are consuming all resources
   ```bash
   nvidia-smi
   ```

### MLflow Tracking Not Working

**Problem**: Experiments not appearing in MLflow or metrics not logging

**Solutions**:

1. **Check MLflow Service**:
   ```bash
   docker-compose logs mlflow

   # Access MLflow UI
   open http://localhost:5001
   ```

2. **Verify Environment Variables**:
   ```bash
   # In backend container
   docker-compose exec backend env | grep MLFLOW

   # Should show:
   # MLFLOW_TRACKING_URI=http://mlflow:5000
   ```

3. **Check Storage**:
   ```bash
   # Verify MLflow artifacts directory
   ls -lh mlflow_artifacts/
   ```

4. **Manually Test MLflow**:
   ```python
   import mlflow

   mlflow.set_tracking_uri("http://localhost:5001")

   with mlflow.start_run():
       mlflow.log_param("test", "value")
   ```

### Database Migration Errors

**Problem**: Alembic migration fails or database schema is incorrect

**Solutions**:

1. **Check Current Migration**:
   ```bash
   docker-compose exec backend alembic current
   ```

2. **Run Migrations**:
   ```bash
   docker-compose exec backend alembic upgrade head
   ```

3. **Reset Database** (⚠️ destroys all data):
   ```bash
   docker-compose down -v
   docker-compose up -d
   ```

4. **Check Migration History**:
   ```bash
   docker-compose exec backend alembic history
   ```

### Container Startup Failures

**Problem**: Docker containers fail to start or crash immediately

**Solutions**:

1. **Check Logs**:
   ```bash
   docker-compose logs --tail=100
   ```

2. **Verify Docker Resources**:
   - Ensure Docker has sufficient memory (8GB+ recommended)
   - Check Docker Desktop settings

3. **Clean Docker Environment**:
   ```bash
   # Remove all stopped containers
   docker-compose down

   # Remove volumes (⚠️ destroys data)
   docker-compose down -v

   # Rebuild images
   docker-compose build --no-cache
   docker-compose up -d
   ```

4. **Check Port Conflicts**:
   ```bash
   # Check if ports are already in use
   lsof -i :5000  # Backend
   lsof -i :3000  # Frontend
   lsof -i :5001  # MLflow
   ```

### Model Download Failures

**Problem**: Cannot download models from Hugging Face Hub

**Solutions**:

1. **Check Internet Connection**:
   ```bash
   ping huggingface.co
   ```

2. **Authenticate with Hugging Face**:
   ```bash
   # Set HF token in environment
   export HF_TOKEN="your_token_here"

   # Or in docker-compose.yml
   environment:
     - HF_TOKEN=${HF_TOKEN}
   ```

3. **Use Model Cache**:
   - Models are cached in `~/.cache/huggingface/`
   - Mount this directory to avoid re-downloading:
   ```yaml
   volumes:
     - ~/.cache/huggingface:/root/.cache/huggingface
   ```

4. **Check Model Availability**:
   - Verify model exists on HuggingFace Hub
   - Check if model requires authentication
   - Ensure model license allows usage

### Performance Issues / Slow Training

**Problem**: Training is slower than expected

**Solutions**:

1. **Enable Mixed Precision**:
   ```json
   {
     "hyperparameters": {
       "fp16": true  // Or "bf16": true for newer GPUs
     }
   }
   ```

2. **Optimize Data Loading**:
   ```json
   {
     "hyperparameters": {
       "dataloader_num_workers": 4,
       "dataloader_pin_memory": true
     }
   }
   ```

3. **Use Distributed Training**:
   ```json
   {
     "resources": {
       "gpu_count": 4,
       "distributed_backend": "nccl"
     }
   }
   ```

4. **Profile GPU Utilization**:
   ```bash
   # Monitor GPU usage
   nvidia-smi dmon -i 0

   # Should show high GPU utilization (>80%)
   ```

5. **Check Bottlenecks**:
   - CPU: Increase dataloader workers
   - GPU: Increase batch size
   - I/O: Use faster storage (NVMe SSD)
   - Network: Use local dataset cache

### Cannot Access Web UI

**Problem**: Web interface not loading or showing blank page

**Solutions**:

1. **Check Frontend Logs**:
   ```bash
   docker-compose logs frontend
   ```

2. **Verify Backend Connection**:
   - Open browser DevTools (F12)
   - Check Console for errors
   - Check Network tab for failed API calls

3. **Clear Browser Cache**:
   - Hard refresh: `Ctrl+Shift+R` (Windows/Linux) or `Cmd+Shift+R` (Mac)
   - Clear browser cache and cookies

4. **Check CORS Settings**:
   ```bash
   # In backend logs, look for CORS errors
   docker-compose logs backend | grep -i cors
   ```

### Getting Help

If you're still experiencing issues:

1. **Search Existing Issues**: [GitHub Issues](https://github.com/def1ant1/SparkTrainer/issues)
2. **Ask the Community**: [GitHub Discussions](https://github.com/def1ant1/SparkTrainer/discussions)
3. **Open a Bug Report**: Use the [Bug Report Template](https://github.com/def1ant1/SparkTrainer/issues/new?template=bug_report.md)
4. **Check Documentation**:
   - [Quick Start Guide](QUICKSTART.md)
   - [Developer Guide](DEVELOPER_GUIDE.md)
   - [API Documentation](http://localhost:5000/api/docs)

When reporting issues, please include:
- SparkTrainer version
- Docker version
- GPU model and driver version
- Relevant logs
- Steps to reproduce

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

**Ways to contribute**:
- Add new recipes (use the Recipe Submission issue template)
- Implement evaluation benchmarks
- Improve documentation
- Report bugs (use the Bug Report template)
- Request features (use the Feature Request template)

**Getting Started**:
1. Read the [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)
2. Check out [good first issues](https://github.com/def1ant1/SparkTrainer/labels/good%20first%20issue)
3. Set up pre-commit hooks: `make setup-pre-commit`
4. Follow our [Code of Conduct](CODE_OF_CONDUCT.md)

## 📊 Project Statistics

- **Backend**: Flask + SQLAlchemy + Celery
- **Frontend**: 22+ React components
- **Trainers**: 3 main classes + recipe system
- **Recipes**: 10+ pre-built recipes
- **Evaluations**: MMLU (57 subjects), COCO (3 tasks)
- **Storage**: PostgreSQL + MLflow + Redis
- **Deployment**: Docker Compose with GPU support

## 🎯 Roadmap

- [ ] WebSocket live metrics streaming
- [ ] User authentication & teams
- [ ] Hyperparameter optimization (Optuna)
- [ ] Model deployment (vLLM, TGI)
- [ ] More evaluation benchmarks
- [ ] Kubernetes deployment
- [ ] Model quantization pipelines
- [ ] Dataset versioning (DVC/LakeFS)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- HuggingFace Transformers team
- MLflow contributors
- Celery & Redis communities
- PyTorch team
- OpenAI Whisper
- BLIP-2, InternVL, Qwen2-VL teams

## 🔗 Links

- **GitHub**: https://github.com/def1ant1/SparkTrainer
- **Issues**: https://github.com/def1ant1/SparkTrainer/issues
- **Discussions**: https://github.com/def1ant1/SparkTrainer/discussions

---

**Built with ❤️ for the ML community**

For questions or support, open an issue or start a discussion on GitHub.
