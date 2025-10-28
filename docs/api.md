# SparkTrainer API Documentation

## Overview

The SparkTrainer API provides a comprehensive REST interface for managing machine learning training jobs, experiments, datasets, and models. This document covers all available endpoints, authentication, and usage examples.

## Base URL

```
Development: http://localhost:5001
Production: https://your-domain.com
```

## Interactive Documentation

SparkTrainer provides interactive API documentation through Swagger UI:

- **Swagger UI**: http://localhost:5001/api/docs
- **OpenAPI 3.0 Spec (JSON)**: http://localhost:5001/openapi.json
- **OpenAPI 3.0 Spec (YAML)**: http://localhost:5001/openapi.yaml

## Authentication

### JWT Authentication

SparkTrainer uses JWT (JSON Web Tokens) for authentication. Most API endpoints require a valid JWT token.

#### Login

```bash
curl -X POST http://localhost:5001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "your-password"}'
```

Response:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

#### Using Tokens

Include the access token in the `Authorization` header:

```bash
curl -H "Authorization: Bearer <access_token>" \
  http://localhost:5001/api/jobs
```

#### Refresh Token

When your access token expires, use the refresh token to get a new one:

```bash
curl -X POST http://localhost:5001/api/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "<refresh_token>"}'
```

## Rate Limiting

API requests are rate-limited per user:

- **Standard users**: 100 requests/minute
- **Premium users**: 1000 requests/minute

Rate limit headers are included in responses:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: Time when limit resets (Unix timestamp)

## Core Endpoints

### Health Checks

#### GET /healthz

Basic health check endpoint.

```bash
curl http://localhost:5001/healthz
```

Response:
```json
{
  "status": "healthy"
}
```

#### GET /readyz

Readiness check - verifies all dependencies are available.

```bash
curl http://localhost:5001/readyz
```

Response:
```json
{
  "status": "ready",
  "database": true,
  "redis": true,
  "celery": true
}
```

### Jobs

#### List Jobs

```bash
GET /api/jobs?status=running&limit=10&offset=0
```

Parameters:
- `status` (optional): Filter by status (pending, running, completed, failed, cancelled)
- `limit` (optional): Number of results (default: 20)
- `offset` (optional): Pagination offset (default: 0)

#### Submit Job

```bash
POST /api/jobs
Content-Type: application/json

{
  "name": "llama-finetuning",
  "command": "python train.py --config config.yaml",
  "gpu_count": 4,
  "priority": 10,
  "environment": {
    "CUDA_VISIBLE_DEVICES": "0,1,2,3"
  }
}
```

Response:
```json
{
  "id": "job-123abc",
  "name": "llama-finetuning",
  "status": "pending",
  "gpu_count": 4,
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### Get Job Details

```bash
GET /api/jobs/{job_id}
```

#### Cancel Job

```bash
POST /api/jobs/{job_id}/cancel
```

#### Get Job Logs

```bash
GET /api/jobs/{job_id}/logs?stream=stdout
```

Parameters:
- `stream`: `stdout` or `stderr`

#### Stream Job Metrics (SSE)

```bash
GET /api/jobs/{job_id}/stream
```

Returns a Server-Sent Events stream with real-time metrics:

```
data: {"step": 100, "loss": 0.45, "lr": 0.0001, "gpu_util": 95}

data: {"step": 101, "loss": 0.44, "lr": 0.0001, "gpu_util": 96}
```

#### Stream Job Metrics (WebSocket)

```javascript
const ws = new WebSocket('ws://localhost:5001/ws/jobs/{job_id}/stream');

ws.onmessage = (event) => {
  const metrics = JSON.parse(event.data);
  console.log(metrics);
};
```

### Experiments

#### List Experiments

```bash
GET /api/experiments
```

#### Create Experiment

```bash
POST /api/experiments
Content-Type: application/json

{
  "name": "llama-7b-finetuning",
  "description": "Fine-tuning Llama-7B on custom dataset",
  "tags": {
    "model": "llama-7b",
    "task": "instruction-tuning"
  }
}
```

#### Get Experiment Details

```bash
GET /api/experiments/{experiment_id}
```

#### List Experiment Runs

```bash
GET /api/experiments/{experiment_id}/runs
```

### Datasets

#### List Datasets

```bash
GET /api/datasets
```

#### Upload Dataset

```bash
POST /api/datasets
Content-Type: multipart/form-data

FormData:
- name: "my-dataset"
- file: <file>
```

#### Get Dataset Details

```bash
GET /api/datasets/{dataset_id}
```

#### Get Dataset Versions

```bash
GET /api/datasets/{dataset_id}/versions
```

Returns version history with DVC hashes:

```json
{
  "versions": [
    {
      "version": "v1.0.0",
      "hash": "abc123def456",
      "created_at": "2024-01-15T10:00:00Z",
      "size": 1073741824
    }
  ]
}
```

### Models

#### List Models

```bash
GET /api/models
```

#### Get Model Details

```bash
GET /api/models/{model_id}
```

#### Tag Model

```bash
POST /api/models/{model_id}/tag
Content-Type: application/json

{
  "tag": "production"
}
```

### GPUs

#### List GPU Status

```bash
GET /api/gpus
```

Response:
```json
[
  {
    "id": 0,
    "name": "NVIDIA A100-SXM4-80GB",
    "memory_total": 85899345920,
    "memory_used": 42949672960,
    "utilization": 85.5,
    "temperature": 72.0,
    "power_usage": 350.0
  }
]
```

### System Metrics

#### Get System Metrics

```bash
GET /api/metrics
```

Response:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "cpu_percent": 45.2,
  "memory_percent": 62.1,
  "disk_usage": 78.5,
  "network_io": {
    "bytes_sent": 1073741824,
    "bytes_recv": 2147483648
  }
}
```

### Hyperparameter Optimization

#### List HPO Studies

```bash
GET /api/hpo/studies
```

#### Create HPO Study

```bash
POST /api/hpo/studies
Content-Type: application/json

{
  "name": "llama-lr-search",
  "objective": "minimize",
  "search_space": {
    "learning_rate": {
      "type": "float",
      "low": 1e-5,
      "high": 1e-3,
      "log": true
    },
    "batch_size": {
      "type": "categorical",
      "choices": [16, 32, 64, 128]
    }
  },
  "n_trials": 100,
  "parallelism": 4
}
```

#### Get Study Results

```bash
GET /api/hpo/studies/{study_id}
```

Response:
```json
{
  "id": "study-123",
  "name": "llama-lr-search",
  "status": "running",
  "n_trials": 45,
  "best_value": 0.342,
  "best_params": {
    "learning_rate": 0.0002,
    "batch_size": 64
  },
  "trials": [...]
}
```

### Model Deployments

#### List Deployments

```bash
GET /api/deployments
```

#### Create Deployment

```bash
POST /api/deployments
Content-Type: application/json

{
  "name": "llama-7b-prod",
  "model_id": "model-abc123",
  "backend": "vllm",
  "replicas": 2,
  "gpu_count": 1
}
```

Supported backends:
- `vllm`: For LLM inference (optimized)
- `tgi`: Text Generation Inference (HuggingFace)
- `triton`: NVIDIA Triton Inference Server (multi-modal)

#### Get Deployment Status

```bash
GET /api/deployments/{deployment_id}
```

Response:
```json
{
  "id": "deploy-123",
  "name": "llama-7b-prod",
  "model_id": "model-abc123",
  "backend": "vllm",
  "status": "running",
  "endpoint": "http://localhost:8000/v1/completions",
  "replicas": 2,
  "requests_per_second": 145.2
}
```

#### Stop Deployment

```bash
DELETE /api/deployments/{deployment_id}
```

### Evaluation & Benchmarks

#### List Evaluations

```bash
GET /api/evaluations
```

#### Run Evaluation

```bash
POST /api/evaluations
Content-Type: application/json

{
  "model_id": "model-abc123",
  "benchmark": "mmlu",
  "config": {
    "subjects": ["mathematics", "physics", "computer_science"],
    "num_shots": 5
  }
}
```

Available benchmarks:
- `mmlu`: Massive Multitask Language Understanding (57 subjects)
- `coco`: COCO image evaluation (detection, segmentation, captioning)
- `safety`: Safety probe evaluation

#### Get Evaluation Results

```bash
GET /api/evaluations/{evaluation_id}
```

Response:
```json
{
  "id": "eval-123",
  "model_id": "model-abc123",
  "benchmark": "mmlu",
  "status": "completed",
  "results": {
    "overall_accuracy": 0.652,
    "subjects": {
      "mathematics": 0.678,
      "physics": 0.643,
      "computer_science": 0.735
    }
  }
}
```

#### Get Leaderboard

```bash
GET /api/leaderboard?benchmark=mmlu&sort=accuracy&format=csv
```

Parameters:
- `benchmark`: Filter by benchmark
- `sort`: Sort field (accuracy, name, date)
- `format`: Response format (json, csv)

## WebSocket API

SparkTrainer supports WebSocket connections for real-time metric streaming.

### Connect to Job Stream

```javascript
const ws = new WebSocket('ws://localhost:5001/ws/jobs/{job_id}/stream');

ws.onopen = () => {
  console.log('Connected to job stream');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Metrics:', data);
  // data = { step: 100, loss: 0.45, lr: 0.0001, ... }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected from job stream');
};
```

### Fallback to SSE

If WebSocket is unavailable, use Server-Sent Events:

```javascript
const eventSource = new EventSource(`/api/jobs/${jobId}/stream`);

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Metrics:', data);
};
```

## Error Handling

All errors follow a consistent format:

```json
{
  "error": "Error type",
  "message": "Detailed error message",
  "status": 400
}
```

Common HTTP status codes:
- `200`: Success
- `201`: Created
- `400`: Bad request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not found
- `429`: Rate limit exceeded
- `500`: Internal server error
- `503`: Service unavailable

## SDK Clients

SparkTrainer provides auto-generated SDK clients for easy integration:

### Python SDK

```bash
pip install sparktrainer-sdk
```

```python
from sparktrainer import SparkTrainerClient

client = SparkTrainerClient(
    base_url="http://localhost:5001",
    api_key="your-api-key"
)

# Submit a job
job = client.jobs.create(
    name="my-training-job",
    command="python train.py",
    gpu_count=4
)

# Stream metrics
for metrics in client.jobs.stream_metrics(job.id):
    print(f"Step {metrics.step}: loss={metrics.loss}")
```

### TypeScript SDK

```bash
npm install @sparktrainer/sdk
```

```typescript
import { SparkTrainerClient } from '@sparktrainer/sdk';

const client = new SparkTrainerClient({
  baseUrl: 'http://localhost:5001',
  apiKey: 'your-api-key'
});

// Submit a job
const job = await client.jobs.create({
  name: 'my-training-job',
  command: 'python train.py',
  gpuCount: 4
});

// Stream metrics
client.jobs.streamMetrics(job.id, (metrics) => {
  console.log(`Step ${metrics.step}: loss=${metrics.loss}`);
});
```

## Examples

### Complete Training Workflow

```bash
# 1. Upload dataset
curl -X POST http://localhost:5001/api/datasets \
  -H "Authorization: Bearer $TOKEN" \
  -F "name=my-dataset" \
  -F "file=@dataset.jsonl"

# 2. Create experiment
curl -X POST http://localhost:5001/api/experiments \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "llama-finetune", "description": "Fine-tuning experiment"}'

# 3. Submit training job
curl -X POST http://localhost:5001/api/jobs \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "llama-training",
    "command": "python -m spark_trainer.recipes.text_lora --config config.yaml",
    "gpu_count": 4
  }'

# 4. Monitor progress (SSE)
curl -N http://localhost:5001/api/jobs/{job_id}/stream \
  -H "Authorization: Bearer $TOKEN"

# 5. Deploy model
curl -X POST http://localhost:5001/api/deployments \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "llama-prod",
    "model_id": "{model_id}",
    "backend": "vllm",
    "replicas": 2
  }'
```

## Best Practices

1. **Always use HTTPS in production** to protect API tokens
2. **Implement token rotation** - refresh tokens before they expire
3. **Handle rate limits** - implement exponential backoff
4. **Use WebSocket fallback** - SSE for browsers that don't support WS
5. **Paginate large result sets** - use limit/offset parameters
6. **Cache responses** when appropriate
7. **Monitor health endpoints** for system status

## Support

For issues or questions:
- GitHub Issues: https://github.com/def1ant1/SparkTrainer/issues
- Documentation: https://sparktrainer.readthedocs.io
- API Status: http://localhost:5001/healthz
