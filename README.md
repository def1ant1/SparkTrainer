# SparkTrainer 🚀

**Production-ready multimodal AI training platform with comprehensive MLOps capabilities**

SparkTrainer is an enterprise-grade machine learning training platform that combines powerful distributed training, automated data ingestion, experiment tracking, and production deployment tools into a unified system.

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

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with CUDA 11.8+ (for training)
- 16GB+ RAM recommended
- 50GB+ disk space

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/def1ant1/SparkTrainer.git
cd SparkTrainer
```

2. **Start infrastructure services**
```bash
docker-compose up -d postgres redis mlflow
```

3. **Initialize database**
```bash
cd backend
python init_db.py --sample-data
```

4. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

5. **Start backend & workers**
```bash
# Terminal 1: Backend
python backend/app.py

# Terminal 2: Celery Worker
celery -A backend.celery_app.celery worker --loglevel=info --concurrency=2
```

6. **Start frontend (development)**
```bash
cd frontend
npm install
npm run dev
```

### Access UIs

- **Frontend**: http://localhost:3000
- **MLflow**: http://localhost:5001
- **Flower** (Celery monitoring): http://localhost:5555
- **Backend API**: http://localhost:5000/api

## 📖 Usage

### 1. Create a Project

Navigate to **Projects** → **New Project**
- Enter project name and description
- Projects group related datasets and experiments

### 2. Ingest Dataset

Go to **Dataset Wizard**:

**Step 1**: Upload videos (drag-drop or folder select)

**Step 2**: Configure processing
- FPS: 1-30 frames/second
- Resolution: e.g., 224x224
- Audio: Enable transcription (Whisper)
- Captioning: Choose backend (BLIP-2, InternVL, etc.)
- Scene detection: Optional

**Step 3**: Integrity check
- Validates video files
- Detects corruption
- Shows duration, size, format

**Step 4**: Start processing
- Extracts frames, audio
- Generates captions, transcripts
- Creates manifest.jsonl

### 3. Train a Model

#### Option A: Using LoRA Recipe

```python
from spark_trainer.recipes.lora_recipes import create_lora_recipe

config = {
    "base_model": "meta-llama/Llama-2-7b-hf",
    "dataset_name": "timdettmers/openassistant-guanaco",
    "lora_r": 8,
    "lora_alpha": 16,
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "batch_size": 4,
    "output_dir": "./lora_output"
}

recipe = create_lora_recipe(config, use_qlora=True)  # 4-bit QLoRA
recipe.run()
```

#### Option B: Via UI

1. Navigate to **Experiments**
2. Click **New Experiment**
3. Select:
   - Project
   - Dataset
   - Model type
   - Recipe (LoRA, Vision-Language, etc.)
4. Configure hyperparameters
5. Start training

Training job will:
- Queue on Celery
- Execute on GPU worker
- Log to MLflow
- Stream metrics real-time
- Save artifacts on completion

### 4. Evaluate Models

#### MMLU (Language Models)

```bash
python -m spark_trainer.evaluation.mmlu_eval \
    --model-path ./lora_output \
    --output-dir ./eval_results \
    --num-fewshot 5 \
    --subjects abstract_algebra astronomy biology
```

#### COCO (Vision Models)

```bash
python -m spark_trainer.evaluation.coco_eval \
    --model-path Salesforce/blip2-opt-2.7b \
    --task captioning \
    --max-samples 1000 \
    --output-dir ./eval_results
```

Evaluation results automatically:
- Save to database
- Update leaderboard
- Log to MLflow

### 5. View Leaderboard

Navigate to **Leaderboard** to see:
- Top models across benchmarks
- Rankings with medals (🥇🥈🥉)
- Filtering by benchmark type
- Export to CSV

## 🛠️ Configuration

### Environment Variables

Create `.env` file:

```bash
# Database
DATABASE_URL=postgresql://sparktrainer:password@localhost:5432/sparktrainer

# Redis
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5001

# Training
DGX_TRAINER_BASE_DIR=/app
CUDA_VISIBLE_DEVICES=0,1,2,3

# Flask
FLASK_ENV=development
SECRET_KEY=your-secret-key
```

### Training Configuration

See `configs/` directory for examples:
- `train_vl_example.yaml`: Vision-language training
- `train_diffusion_example.yaml`: Diffusion models
- `deepspeed_zero2.json`: DeepSpeed ZeRO-2 config
- `deepspeed_zero3.json`: DeepSpeed ZeRO-3 config

## 📚 Documentation

- **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)**: Developer setup and architecture guide
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: How to contribute to SparkTrainer
- **[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)**: Community guidelines
- **[FEATURES.md](FEATURES.md)**: Comprehensive feature documentation
- **[API Documentation](docs/)**: Complete API reference (Sphinx)
- **[Recipe Guide](docs/recipes.md)**: Creating custom recipes
- **[Deployment Guide](docs/deployment.md)**: Production deployment

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
