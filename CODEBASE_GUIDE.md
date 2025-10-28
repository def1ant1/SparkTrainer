# SparkTrainer Codebase Structure and Guide

## Overview

SparkTrainer is a comprehensive ML training platform with:
- **Backend**: Flask-based REST API (Python)
- **Frontend**: React + Vite with Tailwind CSS
- **Database**: SQLAlchemy ORM (supports multiple backends)
- **Core Package**: spark_trainer (Python ML training library)
- **Container**: Docker support for both frontend and backend

---

## 1. Overall Directory Structure

```
SparkTrainer/
├── backend/                   # Flask API server
│   ├── app.py               # Main Flask application (247KB - large file!)
│   ├── models.py            # SQLAlchemy database models
│   ├── auth.py              # Authentication and authorization
│   ├── celery_app.py        # Celery task queue setup
│   ├── celery_tasks.py      # Background task definitions
│   ├── database.py          # Database connection and session management
│   ├── health.py            # Health check endpoints
│   ├── init_db.py           # Database initialization
│   ├── openapi_spec.py      # OpenAPI/Swagger specifications
│   ├── swagger_config.py    # Swagger configuration
│   ├── migrations/          # Alembic database migrations
│   └── utils/
│       └── video.py         # Video processing utilities (frame extraction, metadata, etc.)
│
├── frontend/                 # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx          # Main app (API definitions + routing)
│   │   ├── main.jsx         # Entry point
│   │   ├── index.css        # Global styles and CSS variables (dark/light theme)
│   │   └── components/
│   │       ├── ArchitectureBuilder.jsx    # Neural network architecture designer
│   │       ├── Builder.jsx               # [same as ArchitectureBuilder]
│   │       ├── Models.jsx                # Model browsing, comparison, management
│   │       ├── Datasets.jsx              # Dataset management, ingestion, versioning
│   │       ├── Labeling.jsx              # Data annotation tool (text, NER, image bbox)
│   │       ├── Pipelines.jsx             # DAG-based pipeline editor
│   │       ├── Profile.jsx               # User settings, API tokens, environment config
│   │       ├── Experiments.jsx           # Training experiment tracking
│   │       ├── JobWizard.jsx             # Interactive job creation wizard
│   │       ├── HPOViewer.jsx             # Hyperparameter optimization viewer
│   │       ├── CommandPalette.jsx        # Keyboard command palette
│   │       ├── QuickStart.jsx            # Quick start guide
│   │       └── ui/                       # Reusable UI components
│   │           ├── Button.jsx
│   │           ├── Modal.jsx
│   │           ├── Input.jsx
│   │           ├── Card.jsx
│   │           ├── Tabs.jsx
│   │           ├── Toast.jsx
│   │           ├── Progress.jsx
│   │           └── ...
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── Dockerfile
│
├── src/spark_trainer/        # Core Python ML library
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py               # Command-line interface
│   ├── config.py            # Configuration management
│   ├── logger.py            # Logging utilities
│   ├── data.py              # Data loading and preprocessing
│   ├── dataset_utils.py     # Dataset splitting, validation, consistency checking
│   ├── preprocess.py        # Preprocessing pipelines
│   ├── augmentation.py      # Data augmentation
│   ├── captioning.py        # Image/video captioning utilities
│   ├── object_detection.py  # Object detection utilities
│   ├── scene_detection.py   # Video scene detection
│   ├── smart_sampling.py    # Intelligent sampling strategies
│   ├── profiles.py          # Hardware/environment profiles
│   ├── provenance.py        # Data provenance tracking
│   ├── trainer_registry.py  # Trainer registration and lookup
│   │
│   ├── models/
│   │   └── templates.yaml   # Pre-configured model templates (Whisper, BLIP-2, Qwen2-VL, etc.)
│   │
│   ├── pipelines/
│   │   └── multimodal_pipeline.py  # Multimodal training pipeline
│   │
│   ├── trainers/            # Training implementations
│   │   ├── __init__.py
│   │   ├── multimodal_apotheon.py
│   │   ├── vision_language.py
│   │   └── diffusion.py
│   │
│   ├── distributed/         # Distributed training support
│   ├── evaluation/          # Evaluation metrics and utilities
│   ├── inference/           # Model inference engines
│   ├── infrastructure/      # Infrastructure utilities (GPU, resource management)
│   ├── ingestion/          # Data ingestion pipelines
│   ├── recipes/            # Training recipes and workflows
│   ├── storage/            # Storage backends
│   ├── versioning/         # Model/dataset versioning
│   └── utils/              # General utilities
│       ├── hashing.py
│       ├── gpu_validation.py
│       ├── ffmpeg_utils.py
│       └── manifest.py
│
├── configs/                # Configuration examples
│   ├── deepspeed_zero2.json
│   ├── deepspeed_zero3.json
│   ├── train_diffusion_example.yaml
│   ├── train_vl_example.yaml
│   ├── preprocess_example.yaml
│   └── augmentations.yaml
│
├── models/                 # Trained models storage (directory per model)
│   └── {model_id}/
│       ├── config.json
│       ├── metadata.json
│       ├── model.pth
│       └── env.json
│
├── datasets/              # Datasets storage [created at runtime]
│   └── {dataset_name}/
│       └── {version}/
│           ├── manifest.jsonl  # Line-delimited JSON metadata
│           ├── metadata.json   # Dataset version metadata
│           └── data/          # Actual data files
│
├── jobs/                  # Job definitions and results
│   ├── jobs.json          # Job index
│   ├── pipelines.json     # Saved pipelines
│   ├── schedules.json     # Scheduled jobs
│   ├── users.json         # User data
│   ├── teams.json         # Team data
│   ├── billing.json       # Billing info
│   ├── examples/          # Example job configurations
│   └── experiments/       # Training experiments
│
├── logs/                  # Execution logs
│   └── {job_id}.log
│
├── training_scripts/      # Training script templates
│   ├── context_extension_hf.py
│   └── deepspeed/
│       ├── zero2.json
│       └── zero3.json
│
├── docs/                  # Documentation
│   ├── ARCHITECTURE_BUILDER_GUIDE.md
│   └── MODEL_CREATION_GUIDE.md
│
├── examples/              # Example data and configurations
│   └── eval_results/
│
├── sdk/                   # SDKs for client access
│   └── typescript/        # TypeScript SDK
│
├── docker-compose.yml     # Docker composition for backend + frontend
├── Dockerfile             # Backend Dockerfile
├── nginx.conf             # NGINX reverse proxy config
├── setup.py               # Python package setup
├── requirements.txt       # Python dependencies
└── README.md

```

---

## 2. Frontend Components Details

### Key Pages and Their Files

#### **ArchitectureBuilder.jsx** (45KB)
**Location**: `/home/user/SparkTrainer/frontend/src/components/ArchitectureBuilder.jsx`

**Purpose**: Interactive neural network architecture designer

**Features**:
- 80+ layer types (embeddings, attention, conv, pooling, normalization, etc.)
- 13+ architecture templates (GPT-2, LLaMA, ViT, ResNet-50, BERT, etc.)
- Drag-and-drop canvas
- Layer parameter editing
- Connection management (edges)
- Export as JSON/YAML
- Architecture validation

**Key Elements**:
- `LAYERS` object with layer definitions
- `TEMPLATES` object with pre-built architectures
- Canvas rendering with drag-and-drop
- Parameter editor for selected layers

---

#### **Models.jsx** (42KB)
**Location**: `/home/user/SparkTrainer/frontend/src/components/Models.jsx`

**Purpose**: Model browsing, filtering, comparison, and management

**Features**:
- List/Grid view switching
- Search by name, framework, architecture, size, license, tags
- Sorting by date, size, accuracy
- Bulk operations (delete, export, compare)
- Model templates view
- Model detail view with metrics, card, adapters
- Model comparison dashboard

**Key API Methods**:
- `api.getModelsRaw()` - List models with filters
- `api.deleteModel()` - Delete a model
- `api.bulkDeleteModels()` - Bulk delete
- `api.updateModelMetadata()` - Update model info
- `api.updateModelCard()` - Update model card

---

#### **Datasets.jsx** (36KB)
**Location**: `/home/user/SparkTrainer/frontend/src/components/Datasets.jsx`

**Purpose**: Dataset management, versioning, ingestion, and annotation

**Features**:
- Dataset listing and search
- Version management
- Upload/sync datasets
- Streaming ingestion for large files
- Dataset templates
- Quality checking and metadata editing
- Sample preview
- Video dataset indexing

**Key API Methods**:
- `api.getDatasets()` - List all datasets
- `api.getDataset(name)` - Get dataset details
- `api.uploadDataset(name, file, version)` - Upload dataset
- `api.getDatasetSamples(name, params)` - Get dataset samples
- `api.ingestStreamStart/Chunk/Finalize()` - Streaming ingestion
- `api.updateDatasetMetadata()` - Update dataset metadata

---

#### **Labeling.jsx** (13KB)
**Location**: `/home/user/SparkTrainer/frontend/src/components/Labeling.jsx`

**Purpose**: Data annotation tool for text, NER, and image bbox tasks

**Features**:
- Text classification labeling
- Named Entity Recognition (NER) annotation
- Image bounding box annotation
- Label list management
- Annotation review queue
- Pre-labeling (stub provider)
- YOLO and COCO format export

**Sub-components**:
- `TextPreview` - Display text file content
- `NERAnnotator` - NER annotation interface
- `ImageBBoxAnnotator` - Image bbox drawing interface

**Styling Issue**: The inputs and text fields may have visibility issues due to dark theme conflicts

---

#### **Profile.jsx** (35KB)
**Location**: `/home/user/SparkTrainer/frontend/src/components/Profile.jsx`

**Purpose**: User settings, API tokens, environment configuration

**Features**:
- Dashboard tab with system stats
- Settings tab for personal info and preferences
- Environment info display
- Hugging Face model/dataset browser
- API key management (HF, OpenAI, W&B)
- Framework and theme preferences

**Key API Methods**:
- `fetch('/api/user/settings')` - Get user settings
- `fetch('/api/user/dashboard')` - Get dashboard data
- `fetch('/api/config/persistent')` - Get persistent config
- `fetch('/api/system/environment')` - Get environment info

**TODO**: Complete implementation of all tabs and features

---

#### **Pipelines.jsx** (19KB)
**Location**: `/home/user/SparkTrainer/frontend/src/components/Pipelines.jsx`

**Purpose**: DAG-based pipeline editor and executor

**Features**:
- Visual pipeline editor with drag-and-drop nodes
- Edge drawing between nodes
- Node parameter editing
- Pipeline templates (HPO sweep, cross-validation, ensemble)
- Pipeline execution with dependency resolution
- Save/load pipelines
- Pipeline history and execution tracking

**Node Types**: finetune, eval, train, split (extensible)

---

### **Styling Issues in Labeling and Other Components**

**Problem**: White text fields on light backgrounds (especially in dark theme)

**Root Cause**: 
- In `index.css`, inputs don't have explicit background color styling
- Tailwind classes like `border rounded px-3 py-2` don't include `bg-surface` or explicit background
- Dark theme colors don't auto-apply to input elements

**Solution Needed**:
- Add `bg-surface` class to all input/textarea/select elements
- Add explicit text color classes like `text-text` for inputs
- Ensure inputs inherit theme colors properly

**Files to Fix**:
- `Labeling.jsx` - Line 68, 128, 231
- `Datasets.jsx` - Lines with input elements
- Other components with form inputs

---

## 3. Dataset Handling Code

### Location: `/home/user/SparkTrainer/src/spark_trainer/dataset_utils.py`

**Key Classes**:
1. **`DatasetSplitter`** - Stratified train/val/test splitting
   - `split_manifest()` - Split manifest file
   - `_stratified_split()` - Stratification logic
   - Handles CSV/JSON/JSONL formats

2. **`ConsistencyChecker`** - Dataset validation
   - `check_manifest()` - Check for missing files, empty captions, JSON errors
   - `check_caption_quality()` - Validate caption length and quality

### Location: `/home/user/SparkTrainer/backend/utils/video.py`

**Key Functions**:

1. **`check_ffmpeg_installed()`** - Verify ffmpeg/ffprobe available

2. **`get_video_metadata(video_path)`** - Extract video info via ffprobe
   - Returns: duration, fps, width, height, codec, bitrate, nb_frames
   - **ISSUE**: Calls `json.loads()` on ffprobe output - can fail on malformed JSON

3. **`extract_frames()`** - Extract frames using ffmpeg

4. **`extract_audio()`** - Extract audio track

5. **`transcribe_audio_whisper()`** - Transcribe using Whisper model

6. **`scan_video_directory()`** - Find all video files in directory
   - Supports recursive scanning
   - Auto-detects labels from directory structure

7. **`build_video_manifest()`** - Create JSONL manifest
   - Calls `get_video_metadata()` for each video
   - Writes line-delimited JSON

8. **`get_video_stats()`** - Parse manifest and compute stats
   - **BUG**: Line 507 does `json.loads(line)` without error handling
   - If manifest contains empty lines or malformed JSON, this crashes

### Location: `/home/user/SparkTrainer/backend/app.py`

**Video Dataset Indexing Endpoint** (Line 3046):

```python
@app.route('/api/datasets/index/video', methods=['POST'])
def dataset_index_video():
```

**Flow**:
1. Validates dataset name and source path
2. Creates background job
3. Spawns thread to:
   - Scan for videos
   - Extract metadata
   - Build manifest
   - Compute statistics
   - Save metadata JSON

**Issue**: If manifest.jsonl gets malformed lines, `get_video_stats()` crashes

---

## 4. Pipeline Definitions

### Location: `/home/user/SparkTrainer/frontend/src/components/Pipelines.jsx`

**Stored as JSON in**: `/home/user/SparkTrainer/jobs/pipelines.json`

**Pipeline Structure**:
```json
{
  "id": "uuid",
  "name": "Pipeline Name",
  "nodes": [
    {
      "id": "node-id",
      "label": "Node Name",
      "type": "finetune|train|eval|split",
      "x": 100,
      "y": 100,
      "job": { /* job configuration */ }
    }
  ],
  "edges": [
    { "from": "node-id", "to": "node-id" }
  ],
  "executions": [
    {
      "id": "execution-id",
      "started": "ISO timestamp",
      "node_jobs": { "node-id": "created-job-id" },
      "status": "running|completed|failed"
    }
  ]
}
```

**Backend**: `/home/user/SparkTrainer/backend/app.py` (Line 3350+)
- `@app.route('/api/pipelines', methods=['GET', 'POST'])` - List/create
- `@app.route('/api/pipelines/<pid>', methods=['GET', 'PUT', 'DELETE'])` - CRUD
- `@app.route('/api/pipelines/<pid>/run', methods=['POST'])` - Execute pipeline

---

## 5. Model Definitions

### Location: `/home/user/SparkTrainer/src/spark_trainer/models/templates.yaml`

**Contains 50+ pre-configured templates**:

Categories:
- **Speech**: Whisper, multilingual ASR
- **Vision-Language**: BLIP-2, Qwen2-VL, GPT-4V-style, LLaVA
- **Vision**: SegFormer, DINO, DiffusionDet
- **Diffusion**: Stable Diffusion, SDXL, ControlNet, Kandinsky
- **Language**: BERT, RoBERTa, Llama, Phi
- **Multimodal**: CLIP variants
- **Specialized**: Medical imaging, Sign Language, Document Understanding

**Template Structure**:
```yaml
template-key:
  name: "Display Name"
  description: "..."
  category: "category"
  tags: ["tag1", "tag2"]
  model:
    base_model: "model_id"
    architecture: "arch_name"
    input_modalities: ["text", "image", "audio"]
    output_type: "text|image|audio"
  training:
    framework: "huggingface|pytorch|tensorflow"
    precision: "fp32|fp16|bf16"
    batch_size: 8
    learning_rate: 1e-5
    num_epochs: 10
    optimizer: "adamw"
    scheduler: "cosine|linear|constant"
  data:
    format: "dataset_format"
    preprocessing: ["step1", "step2"]
  metrics: ["metric1", "metric2"]
  resources:
    min_gpu_memory: "16GB"
```

### Storage: `/home/user/SparkTrainer/models/{model_id}/`

**Files**:
- `config.json` - Model configuration
- `model.pth` - Model weights
- `metadata.json` - Metadata (creation date, framework, etc.)
- `env.json` - Training environment info

### Backend: `/home/user/SparkTrainer/backend/app.py`

**Model Routes**:
- `GET /api/models` - List all models
- `GET /api/models/<id>` - Get model details
- `PUT /api/models/<id>/metadata` - Update metadata
- `PUT /api/models/<id>/card` - Update model card
- `POST /api/models/bulk_delete` - Bulk delete
- `GET /api/models/export` - Export models
- `GET /api/models/templates` - Get templates

**Missing**: Save model endpoint (needs to be added)

---

## 6. Database/Storage Structure

### Database Models: `/home/user/SparkTrainer/backend/models.py`

**Key Tables**:

1. **`projects`** - Project grouping
   - Fields: id, name, description, created_at, updated_at, metadata

2. **`datasets`** - Dataset versioning
   - Fields: id, project_id, name, version, modality, size_bytes, num_samples
   - manifest_path, storage_path, checksum, integrity_*
   - statistics, tags, metadata, timestamps

3. **`experiments`** - Training runs
   - Fields: id, project_id, dataset_id, name, status
   - model_type, recipe_name, mlflow_*, metrics, config, hyperparameters
   - model_path, checkpoint_path, tags, metadata

4. **`artifacts`** - Training outputs (models, checkpoints, logs)
   - Fields: id, experiment_id, type, path, size_bytes, metadata

5. **`job_status`** - Job execution tracking
   - States: PENDING → QUEUED → RUNNING → COMPLETED/FAILED/CANCELLED

### File Storage Paths

**Models**: `{MODELS_DIR}/{model_id}/`
**Datasets**: `{DATASETS_DIR}/{dataset_name}/{version}/`
**Jobs**: `{JOBS_DIR}/jobs.json`
**Logs**: `{LOGS_DIR}/{job_id}.log`

---

## 7. API Routes and Backend Services

### File: `/home/user/SparkTrainer/backend/app.py` (247KB file)

**Core Endpoints**:

#### System & Health
- `GET /api/system/info` - System information
- `GET /api/system/health` - Health check
- `GET /api/system/metrics/history` - Metrics history
- `GET /api/frameworks` - Available frameworks

#### Jobs
- `GET /api/jobs` - List jobs
- `POST /api/jobs` - Create job
- `GET /api/jobs/<id>` - Get job detail
- `POST /api/jobs/<id>/cancel` - Cancel job
- `GET /api/jobs/<id>/logs` - Stream logs
- `GET /api/jobs/<id>/checkpoints` - List checkpoints
- `POST /api/jobs/<id>/checkpoint/save` - Save checkpoint

#### Models (Line 448+)
- `GET /api/models` - List models
- `GET /api/models/<id>` - Get model detail
- `GET /api/models/<id>/metadata` - Get metadata
- `PUT /api/models/<id>/metadata` - Update metadata
- `PUT /api/models/<id>/card` - Update model card
- `GET /api/models/<id>/adapters` - List adapters
- `POST /api/models/<id>/adapters/merge` - Merge LoRA adapter
- `GET /api/models/<id>/similar` - Find similar models
- `POST /api/models/bulk_delete` - Bulk delete
- `GET /api/models/export` - Export models
- `GET /api/models/templates` - Get model templates

#### Datasets (Line 2500+)
- `GET /api/datasets` - List datasets
- `POST /api/datasets` - Create dataset
- `GET /api/datasets/<name>` - Get dataset detail
- `DELETE /api/datasets/<name>` - Delete dataset
- `POST /api/datasets/upload` - Upload dataset
- `GET /api/datasets/<name>/samples` - Get samples
- `POST /api/datasets/<name>/process` - Process (extract frames, transcribe)
- `POST /api/datasets/index/video` - **[VIDEO INDEXING - BUG HERE]** Index video directory
- `GET /api/datasets/index/<job_id>` - Get indexing job status

#### Annotations & Labeling
- `GET /api/datasets/<name>/annotations` - Get annotations
- `POST /api/datasets/<name>/annotations/save` - Save annotations
- `GET /api/datasets/<name>/annotations/export/yolo` - Export YOLO format
- `GET /api/datasets/<name>/annotations/export/coco` - Export COCO format
- `GET /api/datasets/<name>/annotations/queue` - Annotation queue
- `POST /api/datasets/<name>/annotations/prelabel` - Pre-label with ML

#### Pipelines (Line 3400+)
- `GET /api/pipelines` - List pipelines
- `POST /api/pipelines` - Create pipeline
- `GET /api/pipelines/<id>` - Get pipeline
- `PUT /api/pipelines/<id>` - Update pipeline
- `DELETE /api/pipelines/<id>` - Delete pipeline
- `POST /api/pipelines/<id>/run` - Execute pipeline

#### HPO (Hyperparameter Optimization)
- `GET /api/hpo/studies` - List studies
- `POST /api/hpo/studies/save` - Save study
- `GET /api/hpo/studies/<id>` - Get study detail

#### Misc
- `GET /api/gpu/partitions` - GPU partition info
- `POST /api/gpu/partition/apply` - Apply GPU config

---

## Key Issues Identified

### 1. **Video Dataset Indexing JSON Parse Error** 
**File**: `/home/user/SparkTrainer/backend/utils/video.py` Line 507

**Problem**:
```python
def get_video_stats(manifest_path: str):
    with open(manifest_path, 'r') as f:
        for line in f:
            record = json.loads(line)  # ← CRASHES if line is empty or invalid JSON
```

**Solution**: Add error handling
```python
def get_video_stats(manifest_path: str):
    with open(manifest_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats['errors'] += 1
                continue
```

### 2. **Labeling Page White Text Fields**
**File**: `/home/user/SparkTrainer/frontend/src/components/Labeling.jsx` and many others

**Problem**: Input fields have white text on white/light backgrounds in dark mode

**Solution**: Add `bg-surface text-text` classes to all form inputs
```jsx
// Before:
<input className="border rounded px-3 py-2" />

// After:
<input className="border rounded px-3 py-2 bg-surface text-text" />
```

### 3. **Save Model Feature Missing**
**File**: `/home/user/SparkTrainer/backend/app.py` - Need new endpoint around line 690+

**Implementation Needed**:
```python
@app.route('/api/models/save', methods=['POST'])
def save_model():
    """Save a model from training job or architecture builder"""
    data = request.json or {}
    # Create model directory
    # Save weights, config, metadata
    # Return model ID
```

And in frontend Models.jsx - need save button in ArchitectureBuilder that calls this.

### 4. **Profile Page Incomplete**
**File**: `/home/user/SparkTrainer/frontend/src/components/Profile.jsx`

**Missing**:
- Environment info tab implementation
- Hugging Face model/dataset browser integration
- Settings persistence
- API key management UI

### 5. **Example Datasets/Pipelines/Models Missing**
**Directory**: `/home/user/SparkTrainer/examples/`

**Need to create**:
- `examples/datasets/` - Sample datasets (CSV, JSON, images)
- `examples/pipelines/` - Example pipeline configs
- `examples/models/` - Example model configs
- Quickstart documentation with runnable examples

---

## Summary Table

| Component | Language | Location | Status | Lines | Key Purpose |
|-----------|----------|----------|--------|-------|-------------|
| Backend API | Python | backend/app.py | Working | 247KB | Flask REST API server |
| Frontend | React/JS | frontend/src/ | 95% | ~500KB | Web UI |
| ArchitectureBuilder | React/JS | components/ArchitectureBuilder.jsx | Working | 45KB | NN designer |
| Models | React/JS | components/Models.jsx | Working | 42KB | Model mgmt |
| Datasets | React/JS | components/Datasets.jsx | Working | 36KB | Dataset mgmt |
| Labeling | React/JS | components/Labeling.jsx | **Has bugs** | 13KB | Annotation tool |
| Profile | React/JS | components/Profile.jsx | Incomplete | 35KB | User settings |
| Pipelines | React/JS | components/Pipelines.jsx | Working | 19KB | Pipeline editor |
| Core ML Lib | Python | src/spark_trainer/ | Working | ~12KB | Training library |
| Dataset Utils | Python | src/spark_trainer/dataset_utils.py | **Has bugs** | 11KB | Dataset tools |
| Video Utils | Python | backend/utils/video.py | **Has bugs** | 19KB | Video processing |
| DB Models | Python | backend/models.py | Working | 15KB | SQLAlchemy models |

---

## Next Steps

1. **Fix JSON Parse Error** in `video.py` line 507
2. **Fix styling** in Labeling and other input fields
3. **Add save model endpoint** in backend
4. **Complete Profile page** implementation
5. **Create example datasets, pipelines, models** in `/examples/`
6. **Add integration tests** for video indexing pipeline

