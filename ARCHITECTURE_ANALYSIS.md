# SparkTrainer Codebase Architecture - Comprehensive Overview

## Executive Summary

SparkTrainer is a production-ready multimodal AI training platform built with a modern full-stack architecture:
- **Frontend**: React 18 + Vite + TailwindCSS (SPA with client-side routing)
- **Backend**: Flask REST API with SQLAlchemy ORM + PostgreSQL
- **Task Queue**: Celery + Redis for async job processing
- **Monitoring**: MLflow for experiment tracking + Flower for Celery monitoring
- **Infrastructure**: Docker Compose with GPU support

---

## 1. PROJECT STRUCTURE

### Root Directory Layout
```
SparkTrainer/
├── frontend/                    # React SPA
│   ├── src/
│   │   ├── App.jsx             # Main app with routing and navigation
│   │   ├── components/         # React components
│   │   └── index.css           # Global styles
│   ├── package.json            # Frontend dependencies
│   ├── vite.config.js          # Vite configuration
│   └── tailwind.config.js      # TailwindCSS configuration
├── backend/                     # Flask API server
│   ├── app.py                  # Main Flask application (6700+ lines)
│   ├── models.py               # SQLAlchemy ORM models
│   ├── auth.py                 # JWT authentication and RBAC
│   ├── database.py             # Database connection management
│   ├── celery_app.py           # Celery configuration
│   ├── celery_tasks.py         # Async task definitions
│   ├── health.py               # Health check endpoints
│   └── requirements.txt        # Backend dependencies
├── docker-compose.yml          # Service orchestration
├── src/spark_trainer/          # Python ML training package
├── configs/                    # Training configuration templates
├── jobs/                       # Job and experiment data
├── models/                     # Trained model storage
├── logs/                       # Training logs
└── docs/                       # Documentation
```

---

## 2. FRONTEND ARCHITECTURE

### Technology Stack
- **Framework**: React 18.2.0 with Hooks
- **Build Tool**: Vite 5.0.12 (fast bundling)
- **Styling**: TailwindCSS 3.4.1 + PostCSS
- **Icons**: Lucide React 0.263.1
- **Runtime**: Node.js/npm

### Frontend Directory Structure
```
frontend/src/
├── App.jsx                          # Main app (2586 lines)
│   ├── Dashboard3Col               # Multi-column dashboard
│   ├── Sidebar                     # Navigation sidebar
│   ├── HeaderBar                   # Top header with breadcrumbs
│   ├── SecondaryNav                # Sub-navigation
│   ├── ThemeToggle                 # Dark/light mode
│   └── CommandPalette              # Global search (Cmd+K)
├── components/
│   ├── Models.jsx                  # Model management (list, detail, compare)
│   ├── Datasets.jsx                # Dataset ingestion and management
│   ├── Experiments.jsx             # Experiment tracking
│   ├── JobWizard.jsx               # Advanced job creation wizard
│   ├── ArchitectureBuilder.jsx     # Neural network builder
│   ├── HPOViewer.jsx               # Hyperparameter optimization viewer
│   ├── Pipelines.jsx               # Training pipeline management
│   ├── Labeling.jsx                # Data labeling interface
│   ├── Profile.jsx                 # User profile and settings
│   ├── QuickStart.jsx              # Onboarding wizard
│   └── ui/                         # Reusable UI components
│       ├── Button.jsx
│       ├── Input.jsx
│       ├── Modal.jsx
│       ├── Card.jsx
│       ├── Toast.jsx
│       ├── Accordion.jsx
│       ├── Tabs.jsx
│       ├── Progress.jsx
│       ├── PageTransition.jsx      # Page transition animations
│       └── classNames.js           # Utility for class concatenation
└── main.jsx                        # Entry point
```

### Navigation Structure

The app uses client-side routing via `currentPage` state with the following pages:

**Main Pages**:
- `dashboard` - System overview with metrics
- `jobs` - Training job management
- `experiments` - Experiment tracking (MLflow integration)
- `pipelines` - Pipeline creation and execution
- `models` - Model browser, detail, and comparison views
- `datasets` - Dataset ingestion and versioning
- `labeling` - Data annotation interface
- `builder` - Neural architecture builder
- `wizard` - Quick start wizard for job creation

**Admin Pages**:
- `admin` - GPU partitioning configuration
- `hardware_gpus` - GPU hardware details
- `hardware_storage` - Storage usage and management
- `hardware_network` - Network monitoring

**Settings Pages**:
- `profile` - User profile, API keys, HuggingFace integration
- `settings_profile` - User settings
- `settings_teams` - Team management
- `settings_billing` - Billing and usage

**Specialized Pages**:
- `create` - Simple job creation form
- `hpo` - Hyperparameter optimization results viewer

### Navigation Implementation

```javascript
// Location: App.jsx lines 1288+
const [currentPage, setCurrentPage] = useState('dashboard');

// Sidebar Navigation (lines 2443+)
const items = [
  { key: 'dashboard', label: 'Dashboard', icon: <LayoutDashboard /> },
  { key: 'jobs', label: 'Jobs', icon: <Activity /> },
  { key: 'experiments', label: 'Experiments', icon: <List /> },
  { key: 'pipelines', label: 'Pipelines', icon: <List /> },
  { key: 'models', label: 'Models', icon: <BarChart3 /> },
  { key: 'datasets', label: 'Datasets', icon: <Database /> },
  { key: 'labeling', label: 'Labeling', icon: <List /> },
  { key: 'profile', label: 'Profile', icon: <User /> },
  { key: 'builder', label: 'Builder', icon: <Monitor /> },
  { key: 'wizard', label: 'Quick Start', icon: <List /> },
  { key: 'admin', label: 'Admin', icon: <Settings /> },
];
```

### Frontend API Service

The frontend uses a centralized `api` object (lines 22-107) with methods for:

**Job Management**:
- `createJob()` - Submit training job
- `getJobs()`, `getJob(id)` - List and get jobs
- `cancelJob(id)` - Cancel running job
- `validateJob(payload)` - Validate job configuration

**Model Management**:
- `getModels()` - List models
- `getModel(id)` - Get model details
- `updateModelMetadata(id, payload)` - Update metadata
- `updateModelCard(id, payload)` - Update model card
- `bulkDeleteModels(ids)` - Batch delete
- `saveModel(payload)` - Save model
- `listModelAdapters(id)` - List LoRA adapters
- `mergeModelAdapter(id, name)` - Merge adapters

**Dataset Management**:
- `getDatasets()`, `getDataset(name)` - List and get datasets
- `uploadDataset(name, file, version)` - Upload files
- `updateDatasetMetadata(name, payload)` - Update metadata
- `getDatasetSamples(name, params)` - Preview data
- Stream ingestion APIs for large files
- `getAnnotations()`, `saveAnnotations()` - Manage annotations
- Export formats: YOLO, COCO, etc.

**Experiments & HPO**:
- `listHpoStudies()` - List hyperparameter studies
- `saveHpoStudy(study)` - Save HPO results

**System**:
- `getSystemInfo()` - System metrics
- `getMetricsHistory()` - Historical metrics
- `getPartitions()` - GPU partitioning
- `getFrameworks()` - Available frameworks

### State Management

React useState hooks are used for local component state:
- Dashboard metrics updates every 10 seconds
- Jobs list updates every 5 seconds
- System info caches, refreshed on demand
- Model view state (single detail or comparison mode)
- Theme preference stored in localStorage

### UI Framework

**Component Library**:
- Custom TailwindCSS component wrappers (Button, Input, Modal, etc.)
- Lucide React icons throughout
- Toast notifications via custom hook
- Page transitions with CSS animations
- Dark/light mode toggle with system preference fallback
- Responsive grid layouts (mobile-first)

---

## 3. BACKEND ARCHITECTURE

### Technology Stack
- **Framework**: Flask 2.3.x
- **Database ORM**: SQLAlchemy
- **Database**: PostgreSQL 15
- **Task Queue**: Celery with Redis broker
- **Authentication**: JWT + Password hashing
- **API Format**: JSON REST

### Backend Directory Structure
```
backend/
├── app.py                          # Main Flask app (6700+ lines)
├── models.py                       # SQLAlchemy ORM models
├── auth.py                         # JWT and RBAC
├── database.py                     # DB connection management
├── celery_app.py                   # Celery configuration
├── celery_tasks.py                 # Async task definitions
├── health.py                       # Health checks
├── init_db.py                      # Database initialization
├── migrations/                     # Alembic migrations
│   └── versions/                   # Migration history
├── utils/                          # Utility modules
├── swagger_config.py               # OpenAPI/Swagger config
├── openapi_spec.py                 # API documentation generator
└── requirements.txt                # Python dependencies
```

### Core API Endpoints

**System & Health** (20+ endpoints):
```
GET    /api/health                  # Health check
GET    /api/system/info             # System information
GET    /api/system/io               # I/O metrics
GET    /api/system/stream           # WebSocket event stream
GET    /api/system/metrics/history  # Historical metrics
GET    /api/settings                # Global settings
GET    /api/environment             # Environment info
```

**Jobs** (15+ endpoints):
```
GET    /api/jobs                    # List jobs
POST   /api/jobs                    # Create job
GET    /api/jobs/<job_id>           # Get job details
POST   /api/jobs/validate           # Validate job config
POST   /api/jobs/<job_id>/cancel    # Cancel job
POST   /api/jobs/<job_id>/pause     # Pause job
POST   /api/jobs/<job_id>/resume    # Resume job
POST   /api/jobs/<job_id>/restart   # Restart job
GET    /api/jobs/<job_id>/metrics   # Get metrics
GET    /api/jobs/<job_id>/logs      # Get logs
GET    /api/jobs/<job_id>/logs/stream  # Stream logs
GET    /api/jobs/<job_id>/checkpoints  # List checkpoints
POST   /api/jobs/<job_id>/checkpoint/save  # Save checkpoint
```

**Models** (20+ endpoints):
```
GET    /api/models                  # List models
GET    /api/models/<model_id>       # Get model details
PUT    /api/models/<model_id>/metadata  # Update metadata
PUT    /api/models/<model_id>/card  # Update model card
POST   /api/models/save             # Save trained model
POST   /api/models/bulk_delete      # Batch delete
GET    /api/models/export           # Export models
GET    /api/models/<model_id>/evals # Get evaluations
GET    /api/models/<model_id>/similar  # Find similar models
GET    /api/models/<model_id>/adapters  # List LoRA adapters
POST   /api/models/<model_id>/adapters/merge  # Merge adapters
GET    /api/models/<model_id>/card.html  # Render model card
```

**Datasets** (25+ endpoints):
```
GET    /api/datasets                # List datasets
POST   /api/datasets/upload         # Upload dataset
GET    /api/datasets/<name>         # Get dataset info
DELETE /api/datasets/<name>         # Delete dataset
PUT    /api/datasets/<name>/metadata  # Update metadata
GET    /api/datasets/<name>/samples # Get data samples
GET    /api/datasets/<name>/annotations  # List annotations
POST   /api/datasets/<name>/annotations/save  # Save annotations
POST   /api/datasets/<name>/annotations/queue  # Queue annotations
POST   /api/datasets/<name>/annotations/prelabel  # Auto-label
GET    /api/datasets/<name>/annotations/export/yolo  # Export YOLO
GET    /api/datasets/<name>/annotations/export/coco  # Export COCO
POST   /api/datasets/ingest/stream_start  # Start streaming
POST   /api/datasets/ingest/stream_chunk  # Send chunk
POST   /api/datasets/ingest/stream_finalize  # Finalize
POST   /api/datasets/<name>/version/create  # Version dataset
POST   /api/datasets/<name>/version/diff    # Compare versions
POST   /api/datasets/<name>/version/rollback  # Rollback version
POST   /api/datasets/<name>/quality/apply  # Apply quality gates
```

**Experiments & HPO** (8+ endpoints):
```
GET    /api/experiments             # List experiments
POST   /api/experiments             # Create experiment
GET    /api/experiments/<exp_id>    # Get experiment
PUT    /api/experiments/<exp_id>    # Update experiment
DELETE /api/experiments/<exp_id>    # Delete experiment
POST   /api/experiments/<exp_id>/star  # Star experiment
GET    /api/experiments/<exp_id>/metrics  # Get metrics
GET    /api/hpo/studies             # List HPO studies
POST   /api/hpo/studies/save        # Save HPO study
GET    /api/hpo/<job_id>            # Get HPO results
```

**Pipelines** (10+ endpoints):
```
GET    /api/pipelines               # List pipelines
POST   /api/pipelines               # Create pipeline
GET    /api/pipelines/<pid>         # Get pipeline
PUT    /api/pipelines/<pid>         # Update pipeline
DELETE /api/pipelines/<pid>         # Delete pipeline
POST   /api/pipelines/<pid>/run     # Execute pipeline
POST   /api/pipelines/<pid>/execute  # Execute with params
GET    /api/pipelines/<pid>/status  # Get execution status
POST   /api/pipelines/<pid>/stages/<stage_id>/retry  # Retry stage
GET    /api/pipelines/templates     # List templates
```

**GPU & Hardware** (10+ endpoints):
```
GET    /api/gpu/partitions          # List GPU partitions
GET    /api/gpu/partition           # Get current partition
GET    /api/gpu/partition/config    # Get partition config
POST   /api/gpu/partition/apply     # Apply partition config
GET    /api/gpu/partition/presets   # Get preset configs
POST   /api/gpu/partition/recommend  # Recommend config
GET    /api/storage/usage           # Storage usage stats
GET    /api/storage/trends          # Storage usage trends
POST   /api/storage/checkpoints/cleanup  # Cleanup checkpoints
```

**Authentication & Users** (10+ endpoints):
```
POST   /api/auth/register           # User registration
POST   /api/auth/login              # User login
POST   /api/auth/logout             # User logout
GET    /api/user/profile            # Get user profile
PUT    /api/user/profile            # Update profile
GET    /api/user/settings           # Get settings
PUT    /api/user/settings           # Update settings
POST   /api/user/avatar             # Upload avatar
GET    /api/user/tokens             # List API tokens
POST   /api/user/tokens             # Create API token
DELETE /api/user/tokens/<token_id>  # Delete token
GET    /api/user/dashboard          # Dashboard data
GET    /api/user/preferences        # Get preferences
PUT    /api/user/preferences        # Update preferences
```

**Teams & Billing** (12+ endpoints):
```
GET    /api/teams                   # List teams
POST   /api/teams                   # Create team
GET    /api/teams/<team_id>         # Get team
PUT    /api/teams/<team_id>         # Update team
DELETE /api/teams/<team_id>         # Delete team
POST   /api/teams/<team_id>/members  # Add member
DELETE /api/teams/<team_id>/members/<user_id>  # Remove member
PUT    /api/teams/<team_id>/members/<user_id>/role  # Change role
GET    /api/teams/<team_id>/quota   # Get quota
PUT    /api/teams/<team_id>/quota   # Update quota
GET    /api/billing/usage           # Usage statistics
GET    /api/billing/breakdown       # Detailed breakdown
GET    /api/billing/invoices        # List invoices
GET    /api/billing/invoices/<id>   # Get invoice
POST   /api/billing/alerts          # Set alerts
```

**Integration** (6+ endpoints):
```
POST   /api/huggingface/download-model    # Download from HF
POST   /api/huggingface/download-dataset  # Download dataset
POST   /api/curation/ollama/run           # Ollama integration
POST   /api/curation/openai/run           # OpenAI integration
GET    /api/trainers/registry             # Trainer registry
POST   /api/trainers/registry/reload      # Reload trainers
```

### Database Schema (SQLAlchemy Models)

Located in `backend/models.py`:

**Core Tables**:
```python
Project
├── id: String(36) [PK]
├── name: String(255) [UNIQUE, INDEXED]
├── description: Text
├── metadata: JSON
└── timestamps: created_at, updated_at

Dataset
├── id: String(36) [PK]
├── project_id: String(36) [FK → Project]
├── name: String(255)
├── version: String(50)
├── modality: String(50)  # text, image, video, audio, multimodal
├── storage_path: String(500)
├── size_bytes: Integer
├── num_samples: Integer
├── checksum: String(64) [SHA256]
├── integrity_checked: Boolean
├── integrity_passed: Boolean
├── integrity_report: JSON
├── statistics: JSON
├── tags: JSON
├── metadata: JSON
└── timestamps

Experiment
├── id: String(36) [PK]
├── project_id: String(36) [FK → Project]
├── dataset_id: String(36) [FK → Dataset, nullable]
├── name: String(255)
├── description: Text
├── mlflow_run_id: String(64) [UNIQUE, INDEXED]
├── mlflow_experiment_id: String(64)
├── status: Enum(JobStatus)  # PENDING, QUEUED, RUNNING, PAUSED, COMPLETED, FAILED, CANCELLED
├── model_type: String(100)
├── recipe_name: String(100)
├── current_epoch: Integer
├── total_epochs: Integer
├── metrics: JSON
├── config: JSON
├── hyperparameters: JSON
├── model_path: String(500)
├── checkpoint_path: String(500)
├── tags: JSON
├── starred: Boolean
└── timestamps + started_at, completed_at

Job (Celery task)
├── id: String(36) [PK]
├── experiment_id: String(36) [FK → Experiment, INDEXED]
├── celery_task_id: String(64) [UNIQUE, INDEXED]
├── job_type: String(50)  # training, evaluation, preprocessing
├── command: Text
├── status: Enum(JobStatus) [INDEXED]
├── priority: Integer
├── progress: Float [0.0-1.0]
├── gpu_ids: JSON
├── gpu_memory_used: Float [GB]
├── cpu_percent: Float
├── memory_used: Float [GB]
├── log_file: String(500)
├── error_message: Text
├── config: JSON
└── timestamps + queued_at, started_at, completed_at

JobStatusTransition (audit trail)
├── id: Integer [PK, autoincrement]
├── job_id: String(36) [FK → Job, INDEXED]
├── from_status: Enum(JobStatus)
├── to_status: Enum(JobStatus)
├── reason: Text
├── metadata: JSON
└── timestamp [INDEXED]

Artifact
├── id: String(36) [PK]
├── experiment_id: String(36) [FK → Experiment, INDEXED]
├── artifact_type: String(50)  # model, checkpoint, log, plot, metric [INDEXED]
├── name: String(255)
├── file_path: String(500)
├── storage_backend: String(50)  # local, s3, minio
├── size_bytes: Integer
├── checksum: String(64) [SHA256]
├── mlflow_artifact_path: String(500)
├── version: String(50)
├── is_latest: Boolean [INDEXED]
├── tags: JSON
└── timestamps

Evaluation
├── id: String(36) [PK]
├── experiment_id: String(36) [FK → Experiment, INDEXED]
├── benchmark_name: String(100) [INDEXED]  # mmlu, coco, glue
├── benchmark_version: String(50)
├── subset: String(100)
├── score: Float [INDEXED]
├── metrics: JSON
├── eval_config: JSON
└── timestamps

LeaderboardEntry
├── id: String(36) [PK]
├── experiment_id: String(36) [FK → Experiment]
├── benchmark_name: String(100) [INDEXED]
├── score: Float [INDEXED]
├── rank: Integer [INDEXED]
├── model_name: String(255)
├── model_type: String(100)
└── timestamps
```

**Status State Machine** (lines 21-41):
```python
VALID_TRANSITIONS = {
    JobStatus.PENDING: [QUEUED, CANCELLED],
    JobStatus.QUEUED: [RUNNING, CANCELLED],
    JobStatus.RUNNING: [PAUSED, COMPLETED, FAILED, CANCELLED],
    JobStatus.PAUSED: [RUNNING, CANCELLED],
    JobStatus.COMPLETED: [],      # Terminal
    JobStatus.FAILED: [],          # Terminal
    JobStatus.CANCELLED: [],       # Terminal
}
```

### Authentication & Authorization

**JWT Implementation** (`backend/auth.py`):

```python
# Token Management
TokenManager.generate_access_token(user_id, username, role, projects)
  → exp: 60 minutes
  → payload: {user_id, username, role, projects, type: "access"}

TokenManager.generate_refresh_token(user_id)
  → exp: 30 days
  → payload: {user_id, type: "refresh"}

TokenManager.verify_token(token) → payload dict
TokenManager.refresh_access_token(refresh_token, user_data) → new access token

# Role-Based Access Control
Role.ADMIN       # Full access
Role.MAINTAINER  # Create/update/read operations
Role.VIEWER      # Read-only access

# Permission Matrix Examples:
"jobs:create" → [ADMIN, MAINTAINER]
"jobs:read" → [ADMIN, MAINTAINER, VIEWER]
"jobs:cancel" → [ADMIN, MAINTAINER]
"jobs:delete" → [ADMIN]
"experiments:*" → Role-based
"datasets:*" → Role-based
"users:create" → [ADMIN]
"system:metrics" → [ADMIN, MAINTAINER]
```

**Decorators**:
```python
@require_auth                                    # Any valid token
@PermissionManager.require_permission('jobs:create')  # Specific permission
@PermissionManager.require_role(Role.ADMIN)     # Minimum role
```

**User Storage** (`USERS_PATH = /jobs/users.json`):
```json
{
  "user_id": {
    "id": "user_id",
    "username": "user",
    "password_hash": "pbkdf2:sha256:...",
    "email": "user@example.com",
    "role": "admin|maintainer|viewer",
    "name": "Full Name",
    "avatar": "base64_data_url",
    "settings": {
      "name": "...",
      "email": "...",
      "hf_token": "...",
      "openai_api_key": "...",
      "wandb_api_key": "..."
    },
    "api_tokens": [
      {"id": "uuid", "token": "token_value", "created": "2024-..."}
    ]
  }
}
```

### Credential Storage Patterns

**Frontend (Profile.jsx)**:
- API keys stored in component state: `hf_token`, `openai_api_key`, `wandb_api_key`
- Sent to backend via `/api/user/settings` PUT endpoint
- Displayed as password inputs (`type="password"`) for security
- HuggingFace token tested against `https://huggingface.co/api/whoami`

**Backend (app.py)**:
- Settings persisted to `USERS_PATH` JSON file (lines 4665-4683)
- No encryption at rest (development pattern, needs hardening for production)
- Credentials included in user settings object: `users[user_id]['settings']`
- Used in downstream tasks for API authentication

**Security Recommendations**:
- Credentials should be encrypted at rest using environment-based key
- Should use vault/secrets manager in production (AWS Secrets Manager, HashiCorp Vault)
- Consider environment variables for sensitive data
- Add credential rotation policies
- Implement audit logging for credential access

---

## 4. ASYNC TASK PROCESSING (CELERY)

### Celery Configuration (`backend/celery_app.py`):

```python
# Broker & Result Backend
CELERY_BROKER_URL = redis://localhost:6379/0
CELERY_RESULT_BACKEND = redis://localhost:6379/0

# Task Serialization
task_serializer = "json"
result_serializer = "json"

# Execution Model
task_acks_late = True               # Ack after completion
task_reject_on_worker_lost = True   # Requeue if worker crashes
task_track_started = True           # Track task start
worker_prefetch_multiplier = 1      # Only fetch 1 task at a time
worker_max_tasks_per_child = 10     # Restart worker every 10 tasks

# Task Time Limits
task_soft_time_limit = 7200  # 2 hours soft limit
task_time_limit = 7500       # 2.08 hours hard limit

# Queue Configuration
Queue("default", priority=default)
Queue("training", priority=10)      # Highest priority
Queue("preprocessing", priority=5)
Queue("evaluation", priority=3)

# Task Routes
"celery_tasks.train_model" → queue: "training"
"celery_tasks.preprocess_data" → queue: "preprocessing"
"celery_tasks.evaluate_model" → queue: "evaluation"
```

### Task Definitions (`backend/celery_tasks.py`):

Tasks include:
- `train_model(job_id, ...)` - GPU-intensive training
- `preprocess_data(dataset_id, ...)` - Data preparation
- `evaluate_model(job_id, ...)` - Model evaluation
- Metric collection, checkpoint saving, etc.

### Monitoring

**Flower UI** (Celery monitoring dashboard):
- URL: `http://localhost:5555`
- Real-time worker status
- Task history and statistics
- Task execution graphs

---

## 5. INFRASTRUCTURE & DEPLOYMENT

### Docker Compose Services

**docker-compose.yml** defines 8 services:

```yaml
postgres:15-alpine
  - Port: 5432
  - Database: sparktrainer
  - User: sparktrainer
  - Password: sparktrainer_dev_pass
  - Volume: postgres_data:/var/lib/postgresql/data

redis:7-alpine
  - Port: 6379
  - Role: Message broker + cache
  - Volume: redis_data:/data

mlflow:v2.9.2
  - Port: 5001 (exposed as 5000 internal)
  - Role: Experiment tracking
  - Backend: PostgreSQL
  - Artifact storage: /mlflow/artifacts

backend (Flask API)
  - Port: 5000
  - Language: Python 3.10+
  - GPU: All available (gpus: all)
  - Volumes: jobs/, models/, logs/, training_scripts/
  - Depends on: postgres, redis

celery-worker
  - Role: Async task processing
  - GPU: All available
  - Command: celery worker --concurrency=2
  - Volumes: shared with backend

flower (Celery monitoring)
  - Port: 5555
  - Role: Celery UI and monitoring

frontend (React SPA)
  - Port: 3000
  - Language: Node.js
  - Build tool: Vite
  - API proxy: /api → backend:5000

nginx
  - Port: 80
  - Role: Reverse proxy + load balancer
  - Config: ./nginx.conf
```

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://sparktrainer:sparktrainer_dev_pass@postgres:5432/sparktrainer

# Redis
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000

# Base Directory
DGX_TRAINER_BASE_DIR=/app

# GPU Configuration
CUDA_VISIBLE_DEVICES=0,1,2,3

# Flask
FLASK_ENV=production
SECRET_KEY=your-secret-key

# Database Logging
SQL_DEBUG=false
```

---

## 6. KEY ARCHITECTURAL PATTERNS

### State Management
- **Frontend**: React hooks (useState, useEffect) + localStorage for theme
- **Backend**: Flask request context + JSON file storage (development)
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Cache**: Redis for session storage and Celery results

### API Communication Pattern
```
Frontend (React)
    ↓ (HTTP REST + JSON)
Backend API (Flask)
    ↓ (Queue tasks)
Celery Workers (GPU processing)
    ↓ (Monitor via)
MLflow (Experiment tracking)
    ↓ (Store in)
PostgreSQL + Redis
```

### Error Handling
- Flask global error handlers (404, 500, generic)
- Try-catch blocks in task execution
- Job status transitions with validation
- Audit trail via JobStatusTransition records

### Caching Strategy
- System info: 10-second refresh interval
- Jobs list: 5-second refresh interval
- GPU metrics: Historical deque with 1-hour window
- User preferences: localStorage

### Data Integrity
- Database schema with foreign keys and indexes
- Dataset integrity checking (checksums, validation)
- Job status state machine enforcement
- Audit trails for critical operations

---

## 7. CURRENT IMPLEMENTATION GAPS & PRODUCTION READINESS

### Implemented Features
- ✅ Multi-page React SPA with navigation
- ✅ Comprehensive REST API (100+ endpoints)
- ✅ SQLAlchemy ORM with migrations
- ✅ JWT authentication + RBAC
- ✅ Celery task queue with priority routing
- ✅ MLflow experiment tracking
- ✅ GPU partitioning and resource management
- ✅ Dataset versioning and quality gates
- ✅ Model artifact management
- ✅ Real-time metrics streaming
- ✅ Docker Compose orchestration

### Areas Needing Production Hardening
- ⚠️ Credentials stored in plain JSON (needs encryption)
- ⚠️ No input validation/sanitization layer
- ⚠️ Limited rate limiting/DDoS protection
- ⚠️ Session management in-memory (needs Redis/database)
- ⚠️ Logging not centralized (needs ELK/Splunk integration)
- ⚠️ No request logging/audit middleware
- ⚠️ Missing database connection pooling optimization
- ⚠️ No API versioning strategy

---

## 8. COMPONENT FILE LOCATIONS & SIZES

| File | Lines | Purpose |
|------|-------|---------|
| `frontend/src/App.jsx` | 2,586 | Main app, navigation, pages |
| `backend/app.py` | 6,700+ | All API routes |
| `backend/models.py` | 432 | Database schema |
| `backend/auth.py` | 282 | JWT + RBAC |
| `frontend/src/components/JobWizard.jsx` | 67,332 | Advanced job creation |
| `frontend/src/components/Models.jsx` | 42,301 | Model management |
| `frontend/src/components/Datasets.jsx` | 36,468 | Dataset ingestion |
| `frontend/src/components/ArchitectureBuilder.jsx` | 46,856 | Neural arch builder |
| `frontend/src/components/Profile.jsx` | 35,407 | User settings + credentials |

---

## 9. API RESPONSE FORMATS

### Standard Response Format
```json
// Success
{
  "status": "ok",
  "data": {...}
}

// Error
{
  "error": "error message",
  "status": 400|401|403|500,
  "message": "detailed message"
}
```

### Common Data Objects

**Job**:
```json
{
  "id": "uuid",
  "experiment_id": "uuid",
  "status": "running|completed|failed",
  "progress": 0.75,
  "gpu_ids": [0, 1],
  "created": "2024-01-01T00:00:00",
  "started_at": "2024-01-01T00:05:00",
  "completed_at": null,
  "metrics": {...}
}
```

**Model**:
```json
{
  "id": "uuid",
  "name": "bert-base",
  "architecture": "transformer",
  "parameters": 110000000,
  "size_bytes": 440000000,
  "created": "2024-01-01T00:00:00",
  "tags": ["language-model", "bert"],
  "model_card": "... markdown content ..."
}
```

---

## CONCLUSION

SparkTrainer is a well-structured, feature-rich ML platform with:
- Modern React frontend with comprehensive navigation
- Extensive Flask REST API (100+ endpoints)
- Robust database schema with audit trails
- Async task processing via Celery
- Integrated MLflow experiment tracking
- Production Docker Compose setup

Key architectural strengths are modularity, comprehensive API coverage, and clear separation of concerns. Main production improvement areas are credential security, input validation, and centralized logging.

