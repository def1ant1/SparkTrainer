# Training Job Setup Wizard - API Endpoints Documentation

This document describes all the API endpoints added for the training job setup wizard feature.

## Table of Contents

- [Base Models API](#base-models-api)
- [Datasets API](#datasets-api)
- [Activity Feed API](#activity-feed-api)
- [Experiment Preflight API](#experiment-preflight-api)
- [Jobs API](#jobs-api)

---

## Base Models API

### GET /api/base-models

List all base models with filtering and pagination.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `family` | string | No | Filter by model family (e.g., "llama", "mistral", "gpt") |
| `modality` | string | No | Filter by modality ("text", "image", "audio", "video", "multimodal") |
| `stage` | string | No | Filter by stage ("staging", "production", "archived") |
| `trainable` | boolean | No | Filter to show only trainable models |
| `servable` | boolean | No | Filter to show only servable models |
| `search` or `q` | string | No | Search query (searches name, description, family) |
| `project` | string | No | Filter by project ID |
| `limit` | integer | No | Number of results to return (default: all) |
| `offset` | integer | No | Pagination offset (default: 0) |

**Response:**

```json
{
  "models": [
    {
      "id": "uuid",
      "name": "Llama-2-7B",
      "family": "llama",
      "description": "Llama 2 7 billion parameter model",
      "params_b": 7.0,
      "dtype": "bf16",
      "context_length": 4096,
      "hidden_size": 4096,
      "num_layers": 32,
      "architecture": "transformer",
      "modality": "text",
      "trainable": true,
      "servable": true,
      "quantized": false,
      "is_gguf": false,
      "stage": "production",
      "status": "active",
      "storage_path": "/models/llama-2-7b",
      "size_bytes": 13900000000,
      "checksum": "sha256...",
      "hf_repo_id": "meta-llama/Llama-2-7b-hf",
      "hf_revision": "main",
      "tokenizer_path": "/models/llama-2-7b/tokenizer",
      "vocab_size": 32000,
      "tags": ["language-model", "llama"],
      "metadata": {},
      "model_card": "# Llama 2...",
      "created_at": "2025-10-30T12:00:00Z",
      "updated_at": "2025-10-30T12:00:00Z"
    }
  ],
  "total": 42,
  "limit": 50,
  "offset": 0
}
```

**Example Requests:**

```bash
# Get all trainable models
curl http://localhost:5000/api/base-models?trainable=true

# Search for llama models
curl http://localhost:5000/api/base-models?q=llama

# Get production-ready text models
curl http://localhost:5000/api/base-models?stage=production&modality=text
```

---

### GET /api/base-models/:id

Get detailed information about a specific base model.

**Response:**

```json
{
  "id": "uuid",
  "name": "Llama-2-7B",
  ... (same fields as list endpoint)
}
```

---

## Datasets API

### GET /api/datasets

List datasets with optional filtering and model compatibility checking.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `project` | string | No | Filter by project ID |
| `modality` | string | No | Filter by modality ("text", "image", "audio", "video", "multimodal") |
| `compatible_with_model` | string | No | Filter to datasets compatible with given model ID |
| `search` or `q` | string | No | Search query (searches name, description) |
| `limit` | integer | No | Number of results to return |
| `offset` | integer | No | Pagination offset (default: 0) |

**Response:**

```json
{
  "datasets": [
    {
      "id": "uuid",
      "project_id": "uuid",
      "name": "alpaca-cleaned",
      "version": "1.0",
      "description": "Cleaned Alpaca instruction dataset",
      "modality": "text",
      "size_bytes": 25000000,
      "num_samples": 52000,
      "manifest_path": "/datasets/alpaca/manifest.jsonl",
      "storage_path": "/datasets/alpaca",
      "checksum": "sha256...",
      "integrity_checked": true,
      "integrity_passed": true,
      "integrity_report": {},
      "statistics": {
        "splits": ["train", "test"],
        "num_splits": 2
      },
      "tags": ["instruction", "cleaned"],
      "metadata": {
        "source": "huggingface",
        "hf_repo_id": "yahma/alpaca-cleaned"
      },
      "created_at": "2025-10-30T12:00:00Z",
      "updated_at": "2025-10-30T12:00:00Z",
      "compatibility": {
        "compatible": true,
        "warnings": [],
        "errors": []
      }
    }
  ],
  "total": 15,
  "limit": 50,
  "offset": 0
}
```

**Example Requests:**

```bash
# Get datasets compatible with a specific model
curl http://localhost:5000/api/datasets?compatible_with_model=<model-uuid>

# Get all text datasets
curl http://localhost:5000/api/datasets?modality=text

# Search for instruction datasets
curl http://localhost:5000/api/datasets?q=instruction
```

**Compatibility Object:**

When `compatible_with_model` is provided, each dataset includes a `compatibility` object:

- `compatible` (boolean): Whether the dataset is compatible with the model
- `warnings` (array): List of compatibility warnings
- `errors` (array): List of compatibility errors that prevent use

---

## Activity Feed API

### GET /api/activity

Get activity feed with filtering and pagination.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `limit` | integer | No | Number of activities to return (default: 100) |
| `offset` | integer | No | Pagination offset (default: 0) |
| `event_type` | string | No | Filter by event type |
| `entity_type` | string | No | Filter by entity type ("job", "transfer", "experiment", "dataset", "model") |
| `user_id` | string | No | Filter by user ID |
| `project_id` | string | No | Filter by project ID |
| `unread_only` | boolean | No | Show only unread activities |

**Event Types:**

- `job_created` - New job created
- `job_started` - Job started running
- `job_completed` - Job completed successfully
- `job_failed` - Job failed
- `job_cancelled` - Job cancelled
- `transfer_started` - Transfer started
- `transfer_completed` - Transfer completed
- `transfer_failed` - Transfer failed
- `dataset_imported` - Dataset imported from HuggingFace
- `model_created` - New model created

**Response:**

```json
{
  "activities": [
    {
      "id": "uuid",
      "event_type": "dataset_imported",
      "entity_type": "dataset",
      "entity_id": "uuid",
      "title": "Dataset imported: alpaca-cleaned",
      "message": "Successfully downloaded dataset from HuggingFace: yahma/alpaca-cleaned",
      "status": "success",
      "user_id": "uuid",
      "project_id": "uuid",
      "metadata": {
        "dataset_id": "uuid",
        "dataset_name": "alpaca-cleaned",
        "hf_repo_id": "yahma/alpaca-cleaned",
        "size_bytes": 25000000,
        "num_samples": 52000
      },
      "read": false,
      "created_at": "2025-10-30T12:00:00Z"
    }
  ],
  "total": 42,
  "limit": 100,
  "offset": 0
}
```

**Example Requests:**

```bash
# Get all unread activities
curl http://localhost:5000/api/activity?unread_only=true

# Get job-related activities
curl http://localhost:5000/api/activity?entity_type=job

# Get recent activities for a project
curl http://localhost:5000/api/activity?project_id=<uuid>&limit=20
```

---

### POST /api/activity

Create a new activity event.

**Request Body:**

```json
{
  "event_type": "job_completed",
  "entity_type": "job",
  "entity_id": "uuid",
  "title": "Training completed: Llama-2-7B Fine-tune",
  "message": "Successfully completed training job in 2h 34m",
  "status": "success",
  "user_id": "uuid",
  "project_id": "uuid",
  "metadata": {
    "job_id": "uuid",
    "job_name": "Llama-2-7B Fine-tune",
    "duration_seconds": 9240
  }
}
```

**Response:**

```json
{
  "id": "uuid",
  "event_type": "job_completed",
  ... (full activity object)
}
```

---

### POST /api/activity/:id/mark-read

Mark a specific activity as read.

**Response:**

```json
{
  "id": "uuid",
  "read": true,
  ... (full activity object)
}
```

---

### POST /api/activity/mark-all-read

Mark all activities as read for a user/project.

**Request Body:**

```json
{
  "user_id": "uuid",  // Optional
  "project_id": "uuid"  // Optional
}
```

**Response:**

```json
{
  "marked_read": 15
}
```

---

## Experiment Preflight API

### POST /api/experiments/preflight

Run preflight checks for an experiment configuration. Returns compatibility warnings/errors and resource estimates.

**Request Body:**

```json
{
  "base_model_id": "uuid",
  "dataset_id": "uuid",
  "recipe_id": "uuid",
  "adapters": [
    {
      "adapter_id": "uuid"
    }
  ],
  "train": {
    "max_steps": 1000,
    "global_batch_size": 8,
    "grad_accum": 4,
    "learning_rate": 2e-5
  },
  "resources": {
    "gpus": 1,
    "gpu_type": "a100",
    "strategy": "ddp",
    "mixed_precision": "bf16"
  }
}
```

**Response:**

```json
{
  "ok": true,
  "warnings": [
    "Dataset is small (5000 samples). Consider using a larger dataset for better results."
  ],
  "errors": [],
  "estimated_vram_mb": 24576,
  "vram_breakdown": {
    "model_params_gb": 14.0,
    "activations_gb": 4.0,
    "gradients_gb": 3.5,
    "optimizer_gb": 2.5,
    "total_with_buffer_gb": 24.0
  },
  "time_per_step_ms": 125,
  "throughput": {
    "samples_per_second": 64,
    "tokens_per_second": 32768
  }
}
```

**Error Response (Incompatible Configuration):**

```json
{
  "ok": false,
  "warnings": [],
  "errors": [
    "ERROR: Recipe 'Full Fine-Tune' does not support quantized models. Model 'Llama-2-7B-GPTQ' is int4.",
    "ERROR: Dataset failed integrity check. Please review and fix dataset issues before training."
  ],
  "estimated_vram_mb": 0,
  "vram_breakdown": {},
  "time_per_step_ms": 0,
  "throughput": {}
}
```

**Example Request:**

```bash
curl -X POST http://localhost:5000/api/experiments/preflight \
  -H "Content-Type: application/json" \
  -d '{
    "base_model_id": "uuid",
    "dataset_id": "uuid",
    "recipe_id": "uuid",
    "train": {
      "max_steps": 1000,
      "global_batch_size": 8,
      "grad_accum": 4
    },
    "resources": {
      "gpus": 1,
      "gpu_type": "a100"
    }
  }'
```

---

## Jobs API

### POST /api/jobs

Create and optionally start a new training job.

**Request Body:**

```json
{
  "name": "Llama-2-7B Fine-tune on Alpaca",
  "base_model_id": "uuid",
  "dataset_id": "uuid",
  "recipe_id": "uuid",
  "adapters": [],
  "train": {
    "max_steps": 1000,
    "global_batch_size": 8,
    "grad_accum": 4,
    "learning_rate": 2e-5,
    "warmup_steps": 100,
    "seed": 42,
    "checkpoint_interval": 100
  },
  "strategy": {
    "type": "ddp",
    "mixed_precision": "bf16"
  },
  "resources": {
    "gpus": 1,
    "gpu_type": "a100",
    "partition_id": null
  },
  "eval": {
    "suites": ["mmlu", "truthful_qa"],
    "interval": 100
  },
  "export": ["safetensors", "onnx"],
  "metadata": {
    "origin": "template",
    "template_id": "uuid"
  }
}
```

**Response:**

```json
{
  "id": "uuid",
  "experiment_id": "uuid",
  "name": "Llama-2-7B Fine-tune on Alpaca",
  "status": "PENDING",
  "created_at": "2025-10-30T12:00:00Z",
  "celery_task_id": "celery-task-id",
  ... (full job object)
}
```

**Example Request:**

```bash
curl -X POST http://localhost:5000/api/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Training Job",
    "base_model_id": "uuid",
    "dataset_id": "uuid",
    "recipe_id": "uuid",
    "train": {
      "max_steps": 1000,
      "global_batch_size": 8
    },
    "resources": {
      "gpus": 1
    },
    "metadata": {
      "origin": "template",
      "template_id": "uuid"
    }
  }'
```

---

## Status Codes

All endpoints use standard HTTP status codes:

- `200 OK` - Request succeeded
- `201 Created` - Resource created successfully
- `400 Bad Request` - Invalid request parameters
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

## Error Response Format

All error responses follow this format:

```json
{
  "error": "Error message describing what went wrong"
}
```

---

## Browser Notifications

The Activity Feed component supports browser notifications for:

- Job completions
- Job failures
- Transfer completions
- Dataset imports

Users must grant notification permission when prompted. Notifications include:

- **Title**: Activity title
- **Body**: Activity message
- **Click action**: Navigate to relevant entity (job detail, datasets page, etc.)
- **Badge**: Unread count indicator

---

## Data Flow

### Creating a Training Job from Template

1. User selects template → `GET /api/recipes/:id`
2. User selects base model → `GET /api/base-models?trainable=true`
3. System loads compatible datasets → `GET /api/datasets?compatible_with_model=:model_id`
4. User configures hyperparameters and resources
5. System runs preflight checks → `POST /api/experiments/preflight`
6. If preflight OK, user creates job → `POST /api/jobs`
7. System emits activity event → `POST /api/activity`
8. Frontend shows browser notification if enabled

### HuggingFace Dataset Import Flow

1. User initiates HF dataset download → Transfer created
2. Transfer task downloads dataset → `transfer_tasks.download_hf_dataset`
3. On completion, dataset imported to registry → `Dataset` created
4. Activity event emitted → `POST /api/activity`
5. Browser notification shown
6. Dataset appears in → `GET /api/datasets`
7. Dataset becomes available in job creation wizard

---

## Migration

To apply the Activity model migration:

```bash
cd backend
alembic upgrade head
```

This creates the `activities` table with all necessary indexes.

---

## Frontend Components

### BaseModelSelector

Reusable component for selecting base models with:
- Search and filter
- Chip badges (params, dtype, family, stage, trainable/servable)
- Empty state with helpful actions
- Pagination support

**Usage:**

```jsx
import BaseModelSelector from './components/BaseModelSelector';

<BaseModelSelector
  selectedModelId={modelId}
  onSelectModel={(model) => setModelId(model.id)}
  filterTrainable={true}
  projectId={projectId}
/>
```

### TrainingJobWizard

Multi-step wizard for creating training jobs:
- Base model selection
- Dataset selection (filtered by compatibility)
- Training style selection
- Hyperparameter configuration
- Resource configuration
- Pre-flight checks
- Review and create

**Usage:**

```jsx
import TrainingJobWizard from './components/TrainingJobWizard';

<TrainingJobWizard
  templateId={templateId}
  onClose={() => setShowWizard(false)}
  onSuccess={(job) => console.log('Job created:', job)}
/>
```

### ActivityFeed

Real-time activity feed with notifications:
- Bell icon with unread badge
- Dropdown with activities list
- Browser notification support
- Auto-refresh every 30 seconds
- Mark as read functionality

**Usage:**

```jsx
import ActivityFeed from './components/ActivityFeed';

<ActivityFeed
  userId={userId}
  projectId={projectId}
  showUnreadOnly={false}
/>
```

---

## Notes

- All timestamps are in ISO 8601 format (UTC)
- All IDs are UUIDs (string, 36 characters)
- JSON fields use PostgreSQL JSON type
- Pagination is offset-based
- All list endpoints support filtering and search
- Activity feed auto-refreshes every 30 seconds
- Browser notifications require user permission
- Preflight checks are required before job creation
- Template origin is tracked in job metadata
