# SparkTrainer Comprehensive Features

This document outlines all the major features added to SparkTrainer in this update.

## Table of Contents

1. [Job & Tracking Foundation](#job--tracking-foundation)
2. [Data Ingestion (Video-First)](#data-ingestion-video-first)
3. [Trainer Unification + LoRA](#trainer-unification--lora)
4. [UX Dashboards & Evaluations](#ux-dashboards--evaluations)
5. [Architecture Overview](#architecture-overview)

---

## Job & Tracking Foundation

### Celery + Redis Worker System

**Location**: `backend/celery_app.py`, `backend/celery_tasks.py`

- **Distributed Task Queue**: Celery workers handle training jobs asynchronously
- **Redis Backend**: Fast message broker and result backend
- **Multiple Queues**: Separate queues for training, preprocessing, and evaluation tasks
- **GPU-Aware**: Workers respect GPU assignments and resource limits
- **Monitoring**: Flower web UI for real-time worker monitoring (`http://localhost:5555`)

**Features**:
- Task prioritization (training > preprocessing > evaluation)
- Automatic retry on failure
- Task result persistence
- Worker health checks
- Graceful shutdown handling

### Database Schema (SQLAlchemy)

**Location**: `backend/models.py`, `backend/database.py`

**Tables**:

1. **Projects**: Organize experiments and datasets
   - Hierarchical organization
   - Project-level metadata
   - Team collaboration support

2. **Datasets**: Dataset management with versioning
   - Multi-modal support (video, image, audio, text)
   - Integrity checking and validation
   - Statistics tracking
   - Storage backend abstraction (local, S3, MinIO)

3. **Experiments**: Track training runs
   - MLflow integration
   - Hyperparameter tracking
   - Metrics storage
   - Model artifacts
   - Star/favorite marking

4. **Jobs**: Individual training/evaluation jobs
   - Celery task integration
   - Progress tracking
   - Resource usage monitoring
   - Log file management

5. **Artifacts**: Training outputs
   - Models, checkpoints, logs, plots
   - Versioning support
   - Checksum validation
   - MLflow artifact storage

6. **Evaluations**: Benchmark results
   - MMLU, COCO, GLUE, etc.
   - Detailed metrics
   - Comparison support

7. **Leaderboard**: Model rankings
   - Per-benchmark rankings
   - Historical tracking
   - Metadata annotations

### Job Status Transitions

**Location**: `backend/models.py` (JobStatus enum, Job.transition_to method)

**States**:
```
pending → queued → running → {completed, failed, cancelled}
                  ↓
                paused → running
```

**Features**:
- Validated state transitions
- Audit trail (`JobStatusTransition` table)
- Timestamps for each state
- Reason and metadata tracking
- Prevent invalid transitions

### MLflow Server Integration

**Location**: `docker-compose.yml` (mlflow service), `backend/celery_tasks.py`

**Features**:
- Centralized experiment tracking
- PostgreSQL backend for metadata
- File-based artifact storage (scalable to S3)
- Web UI at `http://localhost:5001`
- Automatic run creation from jobs
- Metric and artifact logging
- Model registry integration

**Capabilities**:
- Real-time metric streaming
- Parameter logging
- Artifact versioning
- Run comparison
- Model deployment tracking

---

## Data Ingestion (Video-First)

### Video Wizard

**Location**: `src/spark_trainer/ingestion/video_wizard.py`, `frontend/src/DatasetWizard.jsx`

**Comprehensive Video Processing Pipeline**:

1. **Video Metadata Extraction**
   - Duration, FPS, resolution
   - Codec information
   - Audio track detection
   - File integrity (SHA256 checksum)

2. **Frame Extraction**
   - Configurable FPS (1-30)
   - Automatic resizing
   - Keyframe-only mode
   - Max frame limits

3. **Audio Extraction & Transcription**
   - FFmpeg-based audio extraction
   - Whisper integration (tiny → large models)
   - Multi-language support
   - Timestamped segments

4. **Image Captioning**
   - Multiple backends: BLIP-2, InternVL, Qwen2-VL, Florence-2
   - Per-frame captioning
   - Batch processing
   - Error handling

5. **Scene Detection**
   - PySceneDetect integration
   - Configurable threshold
   - Scene boundaries with timestamps

6. **Manifest Generation**
   - JSONL format
   - Complete metadata
   - Frame-to-caption mapping
   - Transcript alignment

### Dataset Wizard UI

**Location**: `frontend/src/DatasetWizard.jsx`

**4-Step Wizard**:

**Step 1: File Selection**
- Drag & drop interface
- Multi-file upload
- Folder selection support
- File preview and management

**Step 2: Configuration**
- Frame extraction settings
- Audio processing options
- Whisper model selection
- Captioning backend choice
- Scene detection toggle

**Step 3: Integrity Check**
- Video validation
- Duration checks
- Corruption detection
- Audio sync verification
- Visual results display

**Step 4: Processing**
- Real-time progress tracking
- Frame extraction count
- Caption generation status
- Manifest creation
- Success summary

### Integrity Checks

**Location**: `src/spark_trainer/ingestion/video_wizard.py`, `backend/models.py`

**Checks**:
- File format validation
- Minimum/maximum duration
- Corruption detection (frame read test)
- Audio track validation
- Checksum verification
- Results stored in database

---

## Trainer Unification + LoRA

### Recipe Interface

**Location**: `src/spark_trainer/recipes/base_recipe.py`

**Standardized Workflow**:
```python
recipe = Recipe(config)
data = recipe.prepare()      # Load and prepare dataset
model = recipe.build()       # Initialize model
results = recipe.train()     # Execute training
metrics = recipe.eval()      # Evaluate model
path = recipe.package()      # Save outputs
```

**Benefits**:
- Consistent API across all trainers
- Easy to test and debug
- Modular components
- Extensible for new model types

### Existing Recipes

**Location**: `src/spark_trainer/recipes/`

1. **Vision Recipes** (`vision_recipes.py`)
   - ResNet classification (18/34/50/101/152)
   - Vision Transformer (ViT)
   - Stable Diffusion LoRA

2. **Text Recipes** (`text_recipes.py`)
   - BERT classification
   - GPT-2 generation
   - T5 seq2seq

3. **Audio/Video Recipes** (`audio_video_recipes.py`)
   - Whisper fine-tuning
   - Audio classification

### LoRA & QLoRA Integration

**Location**: `src/spark_trainer/recipes/lora_recipes.py`

**LoRARecipe**:
- Parameter-efficient fine-tuning
- Supports all transformer models
- Auto-detect target modules
- Configurable rank (r) and alpha
- Memory-efficient training
- Adapter-only saving

**Configuration**:
```python
config = LoRAConfig(
    base_model="meta-llama/Llama-2-7b-hf",
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    learning_rate=2e-4,
    num_epochs=3,
    batch_size=4,
    gradient_accumulation_steps=4
)
```

**QLoRARecipe** (Quantized LoRA):
- 4-bit NormalFloat (NF4) quantization
- Double quantization for extra memory savings
- Train 70B models on 24GB GPUs
- Paged optimizers for large batches
- Automatic quantization config

**Features**:
- Works with any HuggingFace model
- Supports causal LM and seq2seq
- Gradient checkpointing
- Mixed precision training
- Automatic best model saving

### FSDP Support

**Location**: `src/spark_trainer/distributed/launchers.py`

**Fully Sharded Data Parallel** (already implemented):
- Shard model parameters across GPUs
- Reduce memory per GPU
- Scale to billions of parameters
- Automatic sharding strategies
- Mixed precision support

**Configuration**:
```python
fsdp_config = {
    "sharding_strategy": "FULL_SHARD",
    "backward_prefetch": "BACKWARD_PRE",
    "mixed_precision": True,
    "cpu_offload": False
}
```

---

## UX Dashboards & Evaluations

### UI Restructure: Projects → Datasets → Experiments

**New Pages**:

1. **Projects** (`frontend/src/Projects.jsx`)
   - Project cards with stats
   - Create/edit/delete projects
   - Navigate to project details
   - Experiment/dataset/model counts

2. **Project Detail View** (enhances existing pages)
   - Tabs: Datasets, Experiments, Models
   - Project-scoped views
   - Filtering and search

3. **Dataset Wizard** (`frontend/src/DatasetWizard.jsx`)
   - Multi-step dataset creation
   - Visual feedback
   - Progress tracking

4. **Enhanced Experiments**
   - Project association
   - MLflow run links
   - Evaluation results
   - Artifact browser

### Live Metrics Dashboard

**Implementation Notes** (requires WebSocket):
- Real-time metric streaming from training jobs
- Live loss/accuracy plots
- GPU utilization graphs
- Training speed (samples/sec)
- ETA calculations

**Planned Stack**:
- Flask-SocketIO for WebSocket support
- Chart.js or Recharts for visualization
- Celery task progress updates
- MLflow metric queries

### Evaluation Baselines

#### MMLU (Massive Multitask Language Understanding)

**Location**: `src/spark_trainer/evaluation/mmlu_eval.py`

**Features**:
- 57 subjects across STEM, humanities, social sciences
- Few-shot evaluation (0-shot to n-shot)
- Category-wise accuracy
- Per-question results
- Batch processing
- HuggingFace datasets integration

**Usage**:
```bash
python -m spark_trainer.evaluation.mmlu_eval \
    --model-path meta-llama/Llama-2-7b-hf \
    --output-dir ./eval_results \
    --num-fewshot 5 \
    --max-samples 100
```

**Metrics**:
- Overall accuracy
- Per-category accuracy (STEM, humanities, etc.)
- Per-subject accuracy
- Detailed per-question results

#### COCO (Common Objects in Context)

**Location**: `src/spark_trainer/evaluation/coco_eval.py`

**Tasks**:
1. **Image Captioning**
   - BLEU-1, BLEU-2, BLEU-3, BLEU-4
   - METEOR
   - CIDEr
   - SPICE

2. **Object Detection**
   - mAP (mean Average Precision)
   - AP50, AP75
   - Per-class AP

3. **Instance Segmentation**
   - Mask mAP
   - Boundary metrics

**Usage**:
```bash
python -m spark_trainer.evaluation.coco_eval \
    --model-path Salesforce/blip2-opt-2.7b \
    --task captioning \
    --output-dir ./eval_results \
    --max-samples 1000
```

### Leaderboard

**Location**: `frontend/src/Leaderboard.jsx`, `backend/models.py` (LeaderboardEntry)

**Features**:
- Multi-benchmark support
- Ranking by score
- Model type filtering
- Top 3 highlighting (gold/silver/bronze)
- Export to CSV
- Statistics summary
- Trend indicators

**Metrics Displayed**:
- Rank (with medals for top 3)
- Model name and type
- Benchmark name
- Score (percentage)
- Date
- Additional metadata

**Benchmark Filters**:
- All benchmarks
- MMLU
- COCO
- GLUE
- SuperGLUE
- Custom benchmarks

---

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Frontend (React)                    │
│  Projects | Datasets | Experiments | Leaderboard        │
└────────────────────────┬────────────────────────────────┘
                         │
                         │ REST API
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   Backend (Flask)                        │
│  - API Endpoints                                         │
│  - Job Management                                        │
│  - MLflow Integration                                    │
└────────┬───────────────────────┬──────────────┬─────────┘
         │                       │              │
         ▼                       ▼              ▼
┌────────────────┐    ┌──────────────────┐  ┌──────────┐
│  PostgreSQL    │    │  Redis           │  │  MLflow  │
│  - Projects    │    │  - Task Queue    │  │  Server  │
│  - Experiments │    │  - Results       │  │          │
│  - Datasets    │    │  - Cache         │  └──────────┘
│  - Jobs        │    └──────────────────┘       │
│  - Artifacts   │                               │
│  - Evaluations │                               ▼
└────────────────┘                    ┌────────────────────┐
         │                            │  Artifact Storage  │
         │                            │  - Models          │
         ▼                            │  - Checkpoints     │
┌────────────────────────────────┐   │  - Logs            │
│     Celery Workers             │   │  - Plots           │
│  - Training Jobs (GPU)         │   └────────────────────┘
│  - Preprocessing Jobs          │
│  - Evaluation Jobs             │
│                                │
│  Monitored by: Flower          │
└────────────────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│   Training Scripts             │
│  - PyTorch                     │
│  - HuggingFace Transformers    │
│  - LoRA/QLoRA                  │
│  - FSDP                        │
└────────────────────────────────┘
```

### Technology Stack

**Backend**:
- Flask 3.0+ (REST API)
- SQLAlchemy 2.0+ (ORM)
- Celery 5.3+ (Task Queue)
- Redis 7+ (Message Broker)
- PostgreSQL 15+ (Database)
- MLflow 2.9+ (Experiment Tracking)

**Frontend**:
- React 18.2+
- Vite 5.0+ (Build Tool)
- Tailwind CSS 3.4+
- Lucide Icons

**Training**:
- PyTorch 2.2+
- HuggingFace Transformers 4.35+
- PEFT (LoRA/QLoRA)
- Accelerate (Distributed Training)
- DeepSpeed (Large Models)

**Data Processing**:
- FFmpeg (Video/Audio)
- OpenCV (Frame Extraction)
- Whisper (Transcription)
- BLIP-2/InternVL/Qwen2-VL (Captioning)
- PySceneDetect (Scene Detection)

### Deployment

**Docker Compose Services**:

1. **postgres**: Database
2. **redis**: Message broker
3. **mlflow**: Experiment tracking server
4. **backend**: Flask API server
5. **celery-worker**: Training job workers (with GPU access)
6. **flower**: Celery monitoring UI
7. **frontend**: React development server
8. **nginx**: Reverse proxy (production)

**Ports**:
- 5000: Backend API
- 5001: MLflow UI
- 5432: PostgreSQL
- 5555: Flower (Celery monitoring)
- 6379: Redis
- 3000: Frontend (development)
- 80: Nginx (production)

### Data Flow

**Training Job Flow**:

1. User creates experiment via UI
2. Backend creates database records (Project, Experiment, Job)
3. Backend queues Celery task with job configuration
4. Celery worker picks up task
5. Worker creates MLflow run
6. Worker executes training script with GPU
7. Training script logs metrics to MLflow
8. Worker updates job progress in database
9. On completion, worker saves artifacts to MLflow
10. Database updated with final status
11. Leaderboard updated if evaluation included

**Dataset Ingestion Flow**:

1. User uploads videos via Dataset Wizard
2. Backend creates dataset record
3. Backend queues preprocessing Celery task
4. Worker processes videos:
   - Extract frames
   - Extract & transcribe audio
   - Generate captions
   - Detect scenes
   - Write manifest
5. Worker updates dataset statistics
6. Dataset ready for training

---

## Quick Start

### 1. Start Infrastructure

```bash
docker-compose up -d postgres redis mlflow
```

### 2. Initialize Database

```bash
cd backend
python init_db.py --sample-data
```

### 3. Start Services

```bash
# Terminal 1: Backend
python backend/app.py

# Terminal 2: Celery Worker
celery -A backend.celery_app.celery worker --loglevel=info

# Terminal 3: Frontend
cd frontend
npm install
npm run dev
```

### 4. Access UIs

- Frontend: http://localhost:3000
- MLflow: http://localhost:5001
- Flower: http://localhost:5555
- Backend API: http://localhost:5000/api

---

## Future Enhancements

**Planned Features**:

1. **Real-time Metrics Streaming**
   - WebSocket integration
   - Live training plots
   - GPU utilization monitoring

2. **Advanced Evaluation**
   - More benchmarks (HellaSwag, PIQA, etc.)
   - Custom benchmark support
   - A/B testing framework

3. **Collaboration**
   - User authentication
   - Team workspaces
   - Experiment sharing
   - Comments and annotations

4. **Hyperparameter Optimization**
   - Optuna integration
   - Bayesian optimization
   - Multi-objective optimization
   - Auto-tuning

5. **Model Deployment**
   - One-click deployment
   - Inference endpoints
   - Quantization pipelines
   - A/B testing in production

6. **Data Versioning**
   - DVC/LakeFS integration
   - Dataset diff viewer
   - Lineage tracking
   - Reproducibility guarantees

---

## Contributing

To add a new feature:

1. **Backend**: Add models in `backend/models.py`, API endpoints in `backend/app.py`
2. **Tasks**: Add Celery tasks in `backend/celery_tasks.py`
3. **Recipes**: Create recipe class in `src/spark_trainer/recipes/`
4. **Evaluations**: Add evaluator in `src/spark_trainer/evaluation/`
5. **Frontend**: Create React component in `frontend/src/`
6. **Tests**: Add tests in `tests/`
7. **Docs**: Update documentation

---

## License

See LICENSE file for details.
