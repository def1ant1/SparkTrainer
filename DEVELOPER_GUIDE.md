# Developer Guide

This guide provides comprehensive information for developers who want to contribute to or extend SparkTrainer.

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Project Structure](#project-structure)
- [Backend Architecture](#backend-architecture)
- [Frontend Architecture](#frontend-architecture)
- [Database Schema](#database-schema)
- [API Design Patterns](#api-design-patterns)
- [Testing Strategy](#testing-strategy)
- [Code Style Guidelines](#code-style-guidelines)
- [Adding New Features](#adding-new-features)
- [Debugging Tips](#debugging-tips)
- [Performance Considerations](#performance-considerations)

## Development Environment Setup

### Prerequisites

- **Python 3.8+** (recommended: 3.12)
- **Node.js 20+** for frontend development
- **Docker & Docker Compose** for services
- **PostgreSQL 16** (via Docker or local)
- **Redis 7** (via Docker or local)
- **NVIDIA GPU** with CUDA 11.8+ (optional, for training)
- **Git** for version control

### Backend Setup

1. **Clone and navigate to the repository**
   ```bash
   git clone https://github.com/def1ant1/SparkTrainer.git
   cd SparkTrainer
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install backend dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r backend/requirements.txt
   pip install -e .  # Install spark_trainer in editable mode
   ```

4. **Install development dependencies**
   ```bash
   pip install black isort flake8 mypy pytest pytest-cov pre-commit
   ```

5. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

6. **Configure environment variables**

   Create a `.env` file in the root directory:
   ```bash
   # Database
   DATABASE_URL=postgresql://sparktrainer:sparktrainer@localhost:5432/sparktrainer

   # Redis
   REDIS_URL=redis://localhost:6379/0
   CELERY_BROKER_URL=redis://localhost:6379/0
   CELERY_RESULT_BACKEND=redis://localhost:6379/0

   # MLflow
   MLFLOW_TRACKING_URI=http://localhost:5001
   MLFLOW_ARTIFACT_ROOT=./mlruns

   # Flask
   FLASK_ENV=development
   FLASK_DEBUG=1
   SECRET_KEY=dev-secret-key-change-in-production

   # Training
   DGX_TRAINER_BASE_DIR=/app
   CUDA_VISIBLE_DEVICES=0  # Adjust based on available GPUs

   # Storage
   STORAGE_BACKEND=local
   LOCAL_STORAGE_PATH=./storage
   ```

7. **Start infrastructure services**
   ```bash
   docker-compose up -d postgres redis mlflow
   ```

8. **Initialize the database**
   ```bash
   cd backend
   alembic upgrade head
   python init_db.py --sample-data
   ```

9. **Run the backend server**
   ```bash
   python backend/app.py
   ```

   The backend will be available at http://localhost:5000

10. **Start Celery worker** (in a new terminal)
    ```bash
    celery -A backend.celery_app.celery worker --loglevel=info --concurrency=2
    ```

11. **Start Flower monitoring** (optional)
    ```bash
    celery -A backend.celery_app.celery flower --port=5555
    ```

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Configure environment variables**

   Create `frontend/.env`:
   ```bash
   VITE_API_URL=http://localhost:5000
   VITE_WS_URL=ws://localhost:5000
   ```

4. **Start development server**
   ```bash
   npm run dev
   ```

   The frontend will be available at http://localhost:3000

### Verifying the Setup

1. Check backend health:
   ```bash
   curl http://localhost:5000/healthz
   curl http://localhost:5000/readyz
   ```

2. Access the UIs:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000/api
   - MLflow: http://localhost:5001
   - Flower: http://localhost:5555

3. Run the test suite:
   ```bash
   pytest tests/ -v
   ```

## Project Structure

```
SparkTrainer/
├── backend/                    # Flask backend application
│   ├── app.py                 # Main Flask application
│   ├── celery_app.py          # Celery configuration
│   ├── celery_tasks.py        # Async task definitions
│   ├── database.py            # SQLAlchemy models and session
│   ├── auth.py                # Authentication and authorization
│   ├── experiment_api.py      # Experiment management endpoints
│   ├── huggingface_exporter.py # HF Hub integration
│   ├── migrations/            # Alembic database migrations
│   │   └── versions/          # Migration scripts
│   ├── swagger_config.py      # OpenAPI/Swagger configuration
│   └── requirements.txt       # Backend Python dependencies
├── frontend/                   # React frontend application
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── pages/             # Page-level components
│   │   ├── hooks/             # Custom React hooks
│   │   ├── utils/             # Utility functions
│   │   ├── types/             # TypeScript type definitions
│   │   └── App.tsx            # Root component
│   ├── package.json           # Frontend dependencies
│   └── vite.config.ts         # Vite build configuration
├── src/spark_trainer/         # Core training library
│   ├── recipes/               # Training recipe system
│   │   ├── recipe_interface.py
│   │   ├── lora_recipes.py
│   │   ├── vision_recipes.py
│   │   ├── audio_video_recipes.py
│   │   └── text_recipes.py
│   ├── trainers/              # Training implementations
│   │   ├── vision_language.py
│   │   ├── diffusion.py
│   │   └── multimodal_apotheon.py
│   ├── evaluation/            # Evaluation benchmarks
│   │   ├── mmlu_eval.py
│   │   ├── coco_eval.py
│   │   └── safety_probes.py
│   ├── ingestion/             # Data ingestion pipeline
│   │   ├── universal_ingestor.py
│   │   ├── video_wizard.py
│   │   ├── quality_gates.py
│   │   └── dataset_cards.py
│   ├── inference/             # Inference and serving
│   │   ├── serving.py
│   │   ├── model_registry.py
│   │   └── ab_testing.py
│   ├── distributed/           # Distributed training
│   │   ├── launchers.py
│   │   └── gpu_scheduler.py
│   └── storage/               # Storage backends
│       └── backends.py
├── tests/                     # Test suite
│   ├── test_spark_trainer.py
│   ├── test_cli.py
│   ├── test_manifest.py
│   └── test_hashing.py
├── examples/                  # Example scripts
│   ├── training_pipeline.py
│   ├── model_merging_pipeline.py
│   └── run_complete_example.py
├── sdk/                       # Client SDKs
│   ├── python/               # Python SDK
│   └── typescript/           # TypeScript SDK
├── docs/                      # Documentation
├── .github/
│   └── workflows/            # GitHub Actions CI/CD
├── docker-compose.yml        # Docker services configuration
├── requirements.txt          # Core Python dependencies
└── setup.py                  # Package setup file
```

## Backend Architecture

### Flask Application Structure

The backend follows a modular Flask application pattern:

**Core Components:**

1. **app.py** - Main Flask application with route definitions
2. **database.py** - SQLAlchemy ORM models and database session management
3. **celery_app.py** - Celery task queue configuration
4. **celery_tasks.py** - Async task definitions for training, preprocessing, and evaluation

**Key Design Patterns:**

- **Blueprint pattern** for modular route organization
- **Factory pattern** for database session creation
- **Repository pattern** for data access (via SQLAlchemy)
- **Task queue pattern** for async job processing (Celery)

### Database Models

The database schema includes:

- **Projects** - Top-level organizational unit
- **Datasets** - Data collections linked to projects
- **Experiments** - Training runs with hyperparameters
- **Jobs** - Async task tracking (queued, running, completed, failed)
- **Artifacts** - Model checkpoints, logs, evaluation results
- **Users** - Authentication and authorization
- **JobAuditLog** - State transition tracking

### Celery Task System

Tasks are organized by type:

- **Training tasks** - GPU-intensive model training
- **Preprocessing tasks** - Data ingestion, video processing
- **Evaluation tasks** - Running benchmarks (MMLU, COCO)
- **Export tasks** - HuggingFace Hub uploads

**Task Configuration:**
```python
# backend/celery_tasks.py
@celery.task(bind=True, max_retries=3)
def train_model(self, experiment_id, config):
    """
    Execute model training job.

    Args:
        experiment_id: Database ID of the experiment
        config: Training configuration dictionary

    Returns:
        dict: Training results with metrics and artifact paths
    """
    try:
        # Update job status
        job = update_job_status(self.request.id, "running")

        # Run training
        result = execute_training(experiment_id, config)

        # Update completion
        update_job_status(self.request.id, "completed", result=result)
        return result
    except Exception as e:
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=2 ** self.request.retries)
```

### API Design

RESTful API endpoints follow these conventions:

- **GET** `/api/projects` - List all projects
- **POST** `/api/projects` - Create a new project
- **GET** `/api/projects/{id}` - Get project details
- **PUT** `/api/projects/{id}` - Update project
- **DELETE** `/api/projects/{id}` - Delete project

**Response Format:**
```json
{
  "status": "success",
  "data": { ... },
  "message": "Operation completed successfully"
}
```

**Error Format:**
```json
{
  "status": "error",
  "message": "Detailed error description",
  "code": "ERROR_CODE"
}
```

## Frontend Architecture

### React Component Organization

**Component Categories:**

1. **Pages** - Top-level route components
   - `ProjectsPage.tsx`
   - `DatasetsPage.tsx`
   - `ExperimentsPage.tsx`
   - `LeaderboardPage.tsx`

2. **Features** - Complex feature components
   - `VideoWizard.tsx` - Multi-step data ingestion
   - `ExperimentForm.tsx` - Training configuration
   - `ModelExporter.tsx` - HuggingFace export

3. **Common** - Reusable UI components
   - `DataTable.tsx`
   - `StatusBadge.tsx`
   - `ProgressBar.tsx`
   - `MetricsChart.tsx`

### State Management

- **React Context** for global state (auth, theme)
- **React Query** for server state and caching
- **Local state** (useState) for component-level state

### API Integration

```typescript
// frontend/src/utils/api.ts
import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL,
  timeout: 30000,
});

export const fetchProjects = async () => {
  const response = await api.get('/api/projects');
  return response.data;
};
```

## Database Schema

### Core Tables

```sql
-- Projects
CREATE TABLE projects (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Datasets
CREATE TABLE datasets (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    dataset_type VARCHAR(50),
    manifest_path TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Experiments
CREATE TABLE experiments (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    dataset_id INTEGER REFERENCES datasets(id) ON DELETE SET NULL,
    name VARCHAR(255) NOT NULL,
    model_type VARCHAR(100),
    recipe_name VARCHAR(100),
    hyperparameters JSONB,
    status VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Jobs
CREATE TABLE jobs (
    id VARCHAR(255) PRIMARY KEY,  -- Celery task ID
    experiment_id INTEGER REFERENCES experiments(id) ON DELETE CASCADE,
    job_type VARCHAR(50),
    status VARCHAR(50),
    progress FLOAT,
    result JSONB,
    error TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### Migrations

Use Alembic for database migrations:

```bash
# Create a new migration
cd backend
alembic revision -m "Add new feature"

# Apply migrations
alembic upgrade head

# Rollback last migration
alembic downgrade -1
```

## API Design Patterns

### Request Validation

Use Pydantic for request validation:

```python
from pydantic import BaseModel, Field

class CreateExperimentRequest(BaseModel):
    """Request model for creating a new experiment."""
    name: str = Field(..., min_length=1, max_length=255)
    project_id: int = Field(..., gt=0)
    dataset_id: int = Field(..., gt=0)
    model_type: str = Field(..., regex="^(lora|vision|audio|video)$")
    hyperparameters: dict = Field(default_factory=dict)

@app.route('/api/experiments', methods=['POST'])
def create_experiment():
    try:
        request_data = CreateExperimentRequest(**request.json)
        # Process validated data
        experiment = create_experiment_in_db(request_data)
        return jsonify({"status": "success", "data": experiment}), 201
    except ValidationError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
```

### Error Handling

```python
from functools import wraps

def handle_errors(f):
    """Decorator for consistent error handling."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValidationError as e:
            return jsonify({"status": "error", "message": str(e)}), 400
        except NotFoundException as e:
            return jsonify({"status": "error", "message": str(e)}), 404
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return jsonify({"status": "error", "message": "Internal server error"}), 500
    return decorated_function
```

## Testing Strategy

### Unit Tests

Test individual functions and classes:

```python
# tests/test_manifest.py
import pytest
from spark_trainer.utils.manifest import ManifestWriter

def test_manifest_writer_creates_file():
    """Test that ManifestWriter creates a JSONL file."""
    writer = ManifestWriter("/tmp/test_manifest.jsonl")
    writer.add_entry({"id": 1, "text": "Sample"})
    writer.close()

    assert os.path.exists("/tmp/test_manifest.jsonl")
```

### Integration Tests

Test API endpoints with database:

```python
# tests/test_api_integration.py
import pytest
from backend.app import create_app
from backend.database import db, init_db

@pytest.fixture
def client():
    """Create test client with test database."""
    app = create_app(testing=True)
    with app.test_client() as client:
        with app.app_context():
            init_db()
        yield client

def test_create_project(client):
    """Test project creation endpoint."""
    response = client.post('/api/projects', json={
        "name": "Test Project",
        "description": "A test project"
    })
    assert response.status_code == 201
    assert response.json['status'] == 'success'
```

### Coverage Requirements

Aim for >80% code coverage:

```bash
pytest --cov=src --cov=backend --cov-report=html --cov-report=term
```

## Code Style Guidelines

### Python

Follow PEP 8 with these additional guidelines:

- **Line length**: 127 characters (configured in flake8)
- **Import ordering**: stdlib, third-party, local (enforced by isort)
- **Formatting**: Use Black for consistent code formatting
- **Type hints**: Use type hints for function signatures

```python
from typing import List, Optional, Dict, Any

def process_dataset(
    dataset_path: str,
    output_dir: str,
    batch_size: int = 32,
    config: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Process a dataset and generate outputs.

    Args:
        dataset_path: Path to the input dataset
        output_dir: Directory for output files
        batch_size: Number of samples per batch (default: 32)
        config: Optional configuration dictionary

    Returns:
        List of output file paths

    Raises:
        ValueError: If dataset_path doesn't exist
        IOError: If output_dir is not writable
    """
    # Implementation
    pass
```

### TypeScript/React

```typescript
// Use functional components with TypeScript
interface ExperimentCardProps {
  experimentId: number;
  name: string;
  status: 'running' | 'completed' | 'failed';
  onSelect?: (id: number) => void;
}

export const ExperimentCard: React.FC<ExperimentCardProps> = ({
  experimentId,
  name,
  status,
  onSelect,
}) => {
  // Component implementation
};
```

### Documentation

Use docstrings for all public functions, classes, and modules:

```python
"""
Module for data ingestion and preprocessing.

This module provides utilities for ingesting various data formats,
including video, audio, images, and text. It handles validation,
transformation, and manifest generation.

Example:
    >>> from spark_trainer.ingestion import UniversalIngestor
    >>> ingestor = UniversalIngestor(output_dir='./data')
    >>> ingestor.ingest_videos(video_paths=['video1.mp4'])
"""
```

## Adding New Features

### Adding a New Training Recipe

1. **Create recipe file**: `src/spark_trainer/recipes/my_recipe.py`

2. **Implement recipe interface**:
```python
from spark_trainer.recipes.recipe_interface import RecipeInterface

class MyCustomRecipe(RecipeInterface):
    """Custom recipe for specialized training."""

    def prepare(self):
        """Load and prepare the dataset."""
        pass

    def build(self):
        """Build the model and optimizer."""
        pass

    def train(self):
        """Execute the training loop."""
        pass

    def evaluate(self):
        """Run evaluation benchmarks."""
        pass

    def package(self):
        """Package the model for deployment."""
        pass
```

3. **Register the recipe**:
```python
# src/spark_trainer/trainer_registry.py
from spark_trainer.recipes.my_recipe import MyCustomRecipe

RECIPE_REGISTRY = {
    'my_custom_recipe': MyCustomRecipe,
    # ... other recipes
}
```

4. **Add tests**: `tests/test_my_recipe.py`

5. **Document the recipe**: Add to `docs/recipes.md`

### Adding a New API Endpoint

1. **Define the route** in `backend/app.py`:
```python
@app.route('/api/my-endpoint', methods=['POST'])
@handle_errors
def my_endpoint():
    """
    My new endpoint description.

    Request Body:
        {
            "param1": "value",
            "param2": 123
        }

    Returns:
        JSON response with results
    """
    data = request.json
    # Process request
    result = process_data(data)
    return jsonify({"status": "success", "data": result}), 200
```

2. **Add database operations** if needed in `backend/database.py`

3. **Create Celery task** if async processing is required

4. **Add frontend integration** in `frontend/src/utils/api.ts`

5. **Write tests** for the endpoint

## Debugging Tips

### Backend Debugging

1. **Enable Flask debug mode**:
   ```bash
   export FLASK_DEBUG=1
   python backend/app.py
   ```

2. **Use logging**:
   ```python
   import logging
   logger = logging.getLogger(__name__)
   logger.debug("Debug message")
   logger.info("Info message")
   logger.error("Error message")
   ```

3. **Interactive debugging with pdb**:
   ```python
   import pdb; pdb.set_trace()
   ```

### Celery Task Debugging

1. **Run Celery in foreground with debug logging**:
   ```bash
   celery -A backend.celery_app.celery worker --loglevel=debug
   ```

2. **Monitor task execution with Flower**:
   ```bash
   celery -A backend.celery_app.celery flower
   # Open http://localhost:5555
   ```

### Database Debugging

1. **Check database connection**:
   ```bash
   psql -U sparktrainer -d sparktrainer -h localhost
   ```

2. **View query logs**: Enable in `backend/database.py`:
   ```python
   engine = create_engine(DATABASE_URL, echo=True)
   ```

## Performance Considerations

### Database Optimization

- Use indexes on frequently queried columns
- Implement pagination for large result sets
- Use connection pooling (configured in SQLAlchemy)
- Optimize N+1 queries with eager loading

```python
# Avoid N+1 queries
experiments = db.session.query(Experiment)\
    .options(joinedload(Experiment.dataset))\
    .all()
```

### Caching Strategy

- Use Redis for caching expensive computations
- Cache API responses with short TTL
- Implement ETags for conditional requests

### Async Processing

- Offload heavy tasks to Celery
- Use appropriate concurrency settings
- Implement task retries with exponential backoff

---

For more information, see:
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [README.md](README.md) - Project overview
- [FEATURES.md](FEATURES.md) - Feature documentation
