# SparkTrainer Python SDK

Official Python SDK for the SparkTrainer MLOps Platform.

## Installation

```bash
pip install sparktrainer-sdk
```

## Quick Start

```python
from sparktrainer import SparkTrainerClient

# Initialize client
client = SparkTrainerClient(
    base_url="http://localhost:5001",
    username="admin",
    password="your-password"
)

# Or use API key
client = SparkTrainerClient(
    base_url="http://localhost:5001",
    api_key="your-api-key"
)

# Submit a training job
job = client.jobs.create(
    name="llama-finetuning",
    command="python -m spark_trainer.recipes.text_lora --config config.yaml",
    gpu_count=4,
    priority=10
)

print(f"Job submitted: {job.id}")

# Stream real-time metrics
for metrics in client.jobs.stream_metrics(job.id):
    print(f"Step {metrics['step']}: loss={metrics['loss']:.4f}")

# Check job status
job = client.jobs.get(job.id)
print(f"Status: {job.status}")

# Get logs
logs = client.jobs.logs(job.id, stream="stdout")
print(logs)
```

## Features

- **Job Management**: Submit, monitor, and manage training jobs
- **Real-time Streaming**: Stream metrics via SSE
- **Experiment Tracking**: Create and manage experiments
- **Dataset Management**: Upload and version datasets
- **Model Registry**: Access trained models
- **GPU Monitoring**: Check GPU status and utilization
- **HPO**: Run hyperparameter optimization studies
- **Deployments**: Deploy models with vLLM, TGI, or Triton

## Examples

### Create an Experiment

```python
experiment = client.experiments.create(
    name="llama-7b-finetuning",
    description="Fine-tuning Llama-7B on custom dataset",
    tags={"model": "llama-7b", "task": "instruction-tuning"}
)
```

### Upload a Dataset

```python
dataset = client.datasets.create(
    name="my-dataset",
    file_path="/path/to/dataset.jsonl"
)
```

### List GPUs

```python
gpus = client.gpus.list()
for gpu in gpus:
    print(f"GPU {gpu.id}: {gpu.name} - {gpu.utilization}% utilization")
```

### Create HPO Study

```python
study = client.hpo.create_study(
    name="llama-lr-search",
    objective="minimize",
    search_space={
        "learning_rate": {
            "type": "float",
            "low": 1e-5,
            "high": 1e-3,
            "log": True
        },
        "batch_size": {
            "type": "categorical",
            "choices": [16, 32, 64, 128]
        }
    },
    n_trials=100,
    parallelism=4
)
```

### Deploy a Model

```python
deployment = client.deployments.create(
    name="llama-7b-prod",
    model_id="model-abc123",
    backend="vllm",
    replicas=2,
    gpu_count=1
)

print(f"Deployment endpoint: {deployment.endpoint}")
```

## API Reference

### SparkTrainerClient

Main client class for interacting with the SparkTrainer API.

#### Methods

- `jobs`: Access job operations
- `experiments`: Access experiment operations
- `datasets`: Access dataset operations
- `models`: Access model operations
- `gpus`: Access GPU monitoring
- `deployments`: Access deployment operations
- `hpo`: Access HPO operations

### Jobs API

- `jobs.list(status=None, limit=20, offset=0)`: List jobs
- `jobs.create(name, command, gpu_count=1, priority=0, environment=None)`: Create job
- `jobs.get(job_id)`: Get job details
- `jobs.cancel(job_id)`: Cancel job
- `jobs.delete(job_id)`: Delete job
- `jobs.logs(job_id, stream="stdout")`: Get job logs
- `jobs.stream_metrics(job_id)`: Stream real-time metrics

### Experiments API

- `experiments.list()`: List experiments
- `experiments.create(name, description=None, tags=None)`: Create experiment
- `experiments.get(experiment_id)`: Get experiment details

### Datasets API

- `datasets.list()`: List datasets
- `datasets.create(name, file_path)`: Upload dataset
- `datasets.get(dataset_id)`: Get dataset details

### Models API

- `models.list()`: List models
- `models.get(model_id)`: Get model details

### GPUs API

- `gpus.list()`: List GPU status

### Deployments API

- `deployments.list()`: List deployments
- `deployments.create(name, model_id, backend, replicas=1, gpu_count=1)`: Create deployment
- `deployments.get(deployment_id)`: Get deployment details
- `deployments.delete(deployment_id)`: Delete deployment

### HPO API

- `hpo.list_studies()`: List HPO studies
- `hpo.create_study(name, objective, search_space, n_trials=100, parallelism=1)`: Create study
- `hpo.get_study(study_id)`: Get study details

## Error Handling

```python
from sparktrainer import (
    SparkTrainerClient,
    AuthenticationError,
    NotFoundError,
    RateLimitError
)

try:
    job = client.jobs.get("invalid-id")
except NotFoundError:
    print("Job not found")
except AuthenticationError:
    print("Authentication failed")
except RateLimitError:
    print("Rate limit exceeded")
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run type checking
mypy sparktrainer_sdk

# Format code
black sparktrainer_sdk
```

## License

MIT License - see LICENSE file for details
