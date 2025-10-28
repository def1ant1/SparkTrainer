# SparkTrainer MLOps Enhancements

## Overview

This document describes the comprehensive MLOps enhancements added to SparkTrainer, covering data ingestion, distributed training, experiment tracking, evaluation, inference, and frontend UX improvements.

---

## 1. Data Ingestion & Dataset Management

### 1.1 JSONL Manifest Standardization

**Location:** `src/spark_trainer/ingestion/dataset_service.py`

SparkTrainer now uses a standardized JSONL manifest format for all datasets:

```json
{
  "id": "000001",
  "source_path": "/path/to/original/file.jpg",
  "processed_path": "/path/to/processed/000001_file.jpg",
  "content_type": "image",
  "mime_type": "image/jpeg",
  "size_bytes": 1024000,
  "checksum": "sha256:abc123...",
  "metadata": {
    "width": 1920,
    "height": 1080,
    "format": "JPEG"
  },
  "annotations": {},
  "quality_scores": {},
  "created_at": "2024-01-01T00:00:00",
  "lineage": {}
}
```

**Usage:**

```python
from spark_trainer.ingestion.dataset_service import DatasetIngestionService

service = DatasetIngestionService(output_dir="./datasets")

manifest_path = service.ingest_dataset(
    dataset_name="my_dataset",
    source_paths=["/path/to/images"],
    content_type="auto",
    metadata={
        'description': 'My image dataset',
        'license': 'MIT',
    },
)
```

### 1.2 Multi-Modal Ingestion Workers

**Supported Types:**
- **Video:** Frame extraction, audio transcription, scene detection
- **Audio:** Duration, format, sample rate extraction
- **Image:** EXIF stripping, format conversion, metadata extraction
- **Text:** Chunking, word count, encoding detection

**Workers:**
- `VideoIngestionWorker` - FFmpeg-based video processing
- `AudioIngestionWorker` - Audio metadata extraction
- `ImageIngestionWorker` - PIL-based image processing with EXIF stripping
- `TextIngestionWorker` - Text statistics and encoding handling

### 1.3 Quality Gates

**Location:** `src/spark_trainer/ingestion/quality_gates.py`

**Features:**
- **Deduplication:**
  - SimHash/MinHash LSH for text
  - Perceptual hashing + CLIP embeddings for images
  - Audio fingerprinting (chromaprint)

- **PII Redaction:**
  - RegEx patterns (email, phone, SSN, credit cards)
  - Presidio NER integration
  - Optional cloud DLP API support

- **Toxicity Detection:**
  - Detoxify model integration
  - Perspective API support
  - Configurable thresholds

**Usage:**

```python
from spark_trainer.ingestion.quality_gates import (
    DedupFilter, PIIRedactor, ToxicityFilter, QualityPipeline
)

# Create quality pipeline
gates = [
    DedupFilter(method="clip", threshold=0.9),
    PIIRedactor(use_presidio=True, redact=True),
    ToxicityFilter(threshold=0.7),
]

pipeline = QualityPipeline(gates=gates)

# Check item
passed, results = pipeline.check(item)
```

### 1.4 Dataset Versioning

**DVC Integration:**
**Location:** `src/spark_trainer/versioning/dvc.py`

**lakeFS Integration:**
**Location:** `src/spark_trainer/versioning/lakefs_client.py`

lakeFS provides Git-like version control for data:

```python
from spark_trainer.versioning.lakefs_client import LakeFSClient

client = LakeFSClient(
    endpoint_url="http://localhost:8000",
    access_key="YOUR_ACCESS_KEY",
    secret_key="YOUR_SECRET_KEY",
)

# Create repository
client.create_repository(
    name="datasets",
    storage_namespace="s3://my-bucket/lakefs",
)

# Version dataset
version = client.version_dataset(
    repository="datasets",
    dataset_path="/path/to/dataset",
    branch="main",
    commit_message="Initial dataset version",
)

# Create branch for experiments
client.create_branch(
    repository="datasets",
    name="experiment-1",
    source_branch="main",
)

# Merge back to main
client.merge(
    repository="datasets",
    source_branch="experiment-1",
    destination_branch="main",
)
```

### 1.5 Auto-Generated Dataset Cards

**Location:** `src/spark_trainer/ingestion/dataset_cards.py`

Automatically generates HuggingFace-compatible dataset cards with:
- Dataset statistics (samples, size, splits)
- Provenance and lineage
- License information
- Curation steps
- Quality metrics

```python
from spark_trainer.ingestion.dataset_cards import DatasetCardGenerator

generator = DatasetCardGenerator(
    dataset_name="my_dataset",
    dataset_path="./datasets/my_dataset",
    description="My awesome dataset",
    license="MIT",
)

generator.analyze_manifest("./datasets/my_dataset/manifest.jsonl")
generator.add_provenance(
    sources=[{"url": "https://example.com/data", "date": "2024-01-01"}],
    processing_steps=["resize", "normalize", "augment"],
)

generator.save("./datasets/my_dataset/README.md")
```

---

## 2. Distributed Training & Scheduling

### 2.1 GPU Scheduler

**Location:** `src/spark_trainer/distributed/gpu_scheduler.py`

**Features:**
- GPU monitoring and allocation
- Load balancing strategies (least_loaded, round_robin, pack)
- MIG (Multi-Instance GPU) awareness
- Memory-aware placement
- Auto-resume from checkpoints

**Usage:**

```python
from spark_trainer.distributed.gpu_scheduler import GPUScheduler

scheduler = GPUScheduler(checkpoint_dir="./checkpoints")

# Allocate GPUs
gpus = scheduler.allocate_gpus(
    job_id="train_job_123",
    num_gpus=2,
    min_memory_mb=8000,
    strategy="least_loaded",
)

# Save checkpoints
scheduler.save_checkpoint(
    job_id="train_job_123",
    state_dict=model.state_dict(),
    epoch=5,
    step=1000,
)

# Auto-resume
result = scheduler.auto_resume_job(
    job_id="train_job_123",
    train_fn=train_model,
    **kwargs,
)

# Release GPUs
scheduler.release_gpus("train_job_123")
```

### 2.2 DDP/FSDP/DeepSpeed Launchers

**Location:** `src/spark_trainer/distributed/launcher.py`

The existing `DistributedTrainerRecipe` in `src/spark_trainer/recipes/recipe_interface.py` provides:
- DDP (Distributed Data Parallel)
- FSDP (Fully Sharded Data Parallel)
- DeepSpeed ZeRO-2 and ZeRO-3

**Auto-Strategy Selection:**
Automatically selects distributed strategy based on:
- Model size
- Available GPU memory
- Number of GPUs

---

## 3. Experiment Tracking & Model Registry

### 3.1 MLflow Integration

SparkTrainer integrates with MLflow for:
- Experiment tracking
- Metric logging
- Artifact storage
- Model versioning

**Existing Integration:** `backend/app.py`, `backend/celery_tasks.py`

### 3.2 Model Registry with Promotion Gates

**Location:** `src/spark_trainer/inference/model_registry.py`

**Features:**
- Model lifecycle stages: Development → Staging → Production → Archived
- Promotion gates with metric thresholds
- Approval workflows
- Signed model bundles (SHA256 checksums)
- Model lineage tracking

**Usage:**

```python
from spark_trainer.inference.model_registry import (
    ModelRegistry, ModelMetadata, ModelStage, ModelFormat
)

registry = ModelRegistry(
    mlflow_tracking_uri="http://localhost:5001",
    registry_dir="./model_registry",
)

# Register model
metadata = ModelMetadata(
    name="my_model",
    version="v1",
    format=ModelFormat.PYTORCH,
    framework="pytorch",
    task_type="text-generation",
    architecture="gpt2",
    parameters={'lr': 1e-4},
    metrics={'accuracy': 0.95, 'f1_score': 0.92},
    tags=['nlp', 'gpt2'],
)

version = registry.register_model(
    name="my_model",
    model_path="./models/my_model.pt",
    metadata=metadata,
    stage=ModelStage.DEVELOPMENT,
)

# Request promotion
request = registry.promote_model(
    name="my_model",
    version=version,
    to_stage=ModelStage.STAGING,
    requester="user@example.com",
    justification="Model passes all tests",
)

# Approve promotion (if required)
registry.approve_promotion(request_id=0, approver="manager@example.com")

# Export model
registry.export_model(
    name="my_model",
    version=version,
    format=ModelFormat.ONNX,
    output_path="./export/my_model.onnx",
)
```

**Promotion Gates:**
- Metric thresholds (accuracy, F1, etc.)
- Safety checks
- Performance benchmarks
- API compatibility validation

---

## 4. Evaluation Harness & Safety Checks

### 4.1 Safety Probes

**Location:** `src/spark_trainer/evaluation/safety_probes.py`

**Probes:**

1. **ToxicityProbe:**
   - Detoxify integration
   - Perspective API support
   - Categories: toxicity, severe_toxicity, obscene, threat, insult

2. **BiasProbe:**
   - Gender bias
   - Race/ethnicity bias
   - Religion bias
   - Age bias

3. **JailbreakProbe:**
   - Prompt injection attacks
   - Role-playing attacks
   - Encoding attacks (Base64, ROT13)
   - Adversarial suffixes

**Usage:**

```python
from spark_trainer.evaluation.safety_probes import SafetyEvaluator

evaluator = SafetyEvaluator(output_dir="./safety_reports")

report = evaluator.evaluate(
    model=my_model,
    model_name="my_model_v1",
)

print(f"Pass Rate: {report['summary']['pass_rate']*100:.1f}%")
print(f"Critical Issues: {report['summary']['critical_issues']}")
```

**Report Includes:**
- Test results per probe
- Severity levels (low, medium, high, critical)
- Actionable recommendations
- Detailed failure examples

### 4.2 Calibration & Confidence Reporting

The safety evaluator includes statistical testing and confidence metrics:
- Calibration curves
- Uncertainty quantification
- Threshold sensitivity analysis

---

## 5. Inference & Deployment

### 5.1 Serving Adapters

**Location:** `src/spark_trainer/inference/serving_adapters.py`

**Supported Backends:**

1. **vLLM:**
   - OpenAI-compatible API
   - Continuous batching
   - PagedAttention for KV cache
   - Tensor parallelism

2. **Text Generation Inference (TGI):**
   - HuggingFace models
   - Flash Attention
   - Quantization (GPTQ, bitsandbytes)
   - Token streaming

3. **Triton Inference Server:**
   - Multi-framework (PyTorch, TensorFlow, ONNX)
   - Dynamic batching
   - Model ensembles
   - HTTP and gRPC protocols

4. **TorchServe:**
   - PyTorch models
   - Custom handlers
   - Model versioning

5. **Custom REST:**
   - Generic REST API adapter
   - Configurable request/response formats

**Usage:**

```python
from spark_trainer.inference.serving_adapters import (
    create_adapter, InferenceRequest
)

# Create vLLM adapter
adapter = create_adapter(
    backend="vllm",
    model_name="gpt2",
    endpoint_url="http://localhost:8000",
)

# Run inference
request = InferenceRequest(
    inputs="Hello, how are you?",
    parameters={"max_tokens": 50, "temperature": 0.7},
)

response = adapter.predict(request)
print(f"Output: {response.outputs}")
print(f"Latency: {response.latency_ms:.2f}ms")
```

### 5.2 A/B Shadow Testing

**Location:** `src/spark_trainer/inference/ab_testing.py`

**Test Types:**
- **A/B Testing:** Split traffic between variants
- **Shadow Deployment:** Send to both, return only champion
- **Canary Deployment:** Gradually increase challenger traffic

**Usage:**

```python
from spark_trainer.inference.ab_testing import (
    ABExperiment, ModelVariant, TestType, DecisionCriteria
)

# Create variants
champion = ModelVariant(
    name="gpt2-v1",
    version="v1",
    adapter=adapter_v1,
    weight=0.8,
    is_champion=True,
)

challenger = ModelVariant(
    name="gpt2-v2",
    version="v2",
    adapter=adapter_v2,
    weight=0.2,
)

# Start experiment
experiment = ABExperiment(
    experiment_name="gpt2_comparison",
    champion=champion,
    challenger=challenger,
    test_type=TestType.SHADOW,
)

# Run tests
for i in range(100):
    request = TestRequest(
        request_id=f"req_{i}",
        inputs=f"Test input {i}",
    )
    output = experiment.run_test(request)

# Check if should promote
should_promote, details = experiment.should_promote_challenger(
    criteria=[DecisionCriteria.LATENCY],
    min_samples=100,
)

# Statistical significance testing
stat_test = experiment.statistical_test(metric="latency_ms")
print(f"P-value: {stat_test['p_value']}")
```

**Features:**
- Parallel execution (shadow mode)
- Traffic splitting with configurable weights
- Statistical significance testing (t-test)
- Automatic promotion based on criteria
- Detailed metrics and analytics

---

## 6. Training Recipes Interface

**Location:** `src/spark_trainer/recipes/recipe_interface.py`

**Recipe Protocol:**

```python
class TrainerRecipe(ABC):
    def prepare(self, data_config) -> (train, val, test)
    def build(self, model_config) -> model
    def train(self, training_config) -> history
    def eval(self, split) -> metrics
    def package(self, export_format) -> output
```

**Built-in Recipes:**
- `LLMRecipe` - Text generation (SFT/LoRA/QLoRA) - `recipes/lora_recipes.py`
- `VisionRecipe` - Image classification (ResNet/ViT) - `recipes/vision_recipes.py`
- `AudioRecipe` - Audio classification (Whisper/Wav2Vec2) - `recipes/audio_video_recipes.py`
- `VideoRecipe` - Video action recognition - `recipes/audio_video_recipes.py`
- `DistillationRecipe` - Knowledge distillation - `recipes/distillation.py`

**Efficiency Toggles:**
- Gradient checkpointing
- Flash Attention
- 4-bit/8-bit quantization
- Mixed precision (fp16/bf16)

**Usage:**

```python
from spark_trainer.recipes.lora_recipes import LoRARecipe
from spark_trainer.recipes.recipe_interface import DataConfig, ModelConfig, TrainingConfig

# Create recipe
recipe = LoRARecipe(
    output_dir="./output",
    lora_r=16,
    lora_alpha=32,
    use_4bit=True,
)

# Configure
data_config = DataConfig(
    dataset_path="./datasets/my_dataset",
    batch_size=32,
)

model_config = ModelConfig(
    architecture="gpt2",
    pretrained="gpt2",
)

training_config = TrainingConfig(
    learning_rate=1e-4,
    epochs=10,
    mixed_precision="fp16",
    gradient_checkpointing=True,
)

# Run complete pipeline
output = recipe.run(data_config, model_config, training_config)
```

---

## 7. Frontend UX Enhancements

### 7.1 Information Architecture

**Projects → Datasets → Experiments → Models**

Hierarchical organization with:
- Project management
- Dataset versioning
- Experiment tracking
- Model registry

**Existing Implementation:** `frontend/src/` (React + Vite + Tailwind)

**Main Pages:**
- `Projects.jsx` - Project management
- `DatasetWizard.jsx` - 4-step dataset ingestion
- `Leaderboard.jsx` - Model rankings

### 7.2 Wizards

**Dataset Ingestion Wizard** (Existing):
- Step 1: Upload files
- Step 2: Configure processing
- Step 3: Quality gates
- Step 4: Review and commit

**Experiment Wizard** (Recommended Addition):
- Step 1: Select dataset
- Step 2: Choose recipe
- Step 3: Configure hyperparameters
- Step 4: Launch training

### 7.3 Live Dashboards

**Recommended WebSocket Integration:**

```python
# backend/app.py
from flask_socketio import SocketIO

socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on('subscribe_job')
def handle_subscribe(data):
    job_id = data['job_id']
    # Stream job metrics
    emit('job_update', {'status': 'running', 'progress': 0.5})
```

**Dashboard Features:**
- GPU heatmap (utilization, memory, temperature)
- Job status table (queued/running/completed/failed)
- Tail-f log viewer with search
- Real-time metric plots (loss, accuracy)
- Diffable run configs

---

## 8. Configuration & Deployment

### 8.1 Environment Variables

```bash
# MLflow
MLFLOW_TRACKING_URI=http://localhost:5001

# lakeFS
LAKEFS_ENDPOINT=http://localhost:8000
LAKEFS_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE
LAKEFS_SECRET_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

# Perspective API (optional)
PERSPECTIVE_API_KEY=your_api_key

# Serving backends
VLLM_ENDPOINT=http://localhost:8000
TGI_ENDPOINT=http://localhost:8080
TRITON_ENDPOINT=http://localhost:8000
```

### 8.2 Docker Compose

Add services to `docker-compose.yml`:

```yaml
services:
  # Existing services...

  lakefs:
    image: treeverse/lakefs:latest
    ports:
      - "8000:8000"
    environment:
      - LAKEFS_DATABASE_TYPE=postgres
      - LAKEFS_DATABASE_POSTGRES_CONNECTION_STRING=postgres://...
      - LAKEFS_AUTH_ENCRYPT_SECRET_KEY=...

  vllm:
    image: vllm/vllm-openai:latest
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models
    command: ["--model", "gpt2", "--port", "8000"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

---

## 9. Integration Examples

### 9.1 End-to-End Workflow

```python
# 1. Ingest dataset with quality gates
from spark_trainer.ingestion.dataset_service import DatasetIngestionService
from spark_trainer.ingestion.quality_gates import DedupFilter, PIIRedactor

service = DatasetIngestionService(
    output_dir="./datasets",
    quality_gates=[DedupFilter(), PIIRedactor()],
)

manifest_path = service.ingest_dataset(
    dataset_name="my_dataset",
    source_paths=["/data/images"],
)

# 2. Train model using recipe
from spark_trainer.recipes.vision_recipes import VisionRecipe

recipe = VisionRecipe(output_dir="./output")
output = recipe.run(data_config, model_config, training_config)

# 3. Evaluate safety
from spark_trainer.evaluation.safety_probes import SafetyEvaluator

evaluator = SafetyEvaluator()
report = evaluator.evaluate(model=output.model, model_name="my_model")

# 4. Register model
from spark_trainer.inference.model_registry import ModelRegistry

registry = ModelRegistry()
version = registry.register_model(
    name="my_model",
    model_path=output.model_path,
    metadata=metadata,
)

# 5. Deploy with A/B testing
from spark_trainer.inference.ab_testing import ABExperiment

experiment = ABExperiment(
    champion=champion_variant,
    challenger=challenger_variant,
    test_type=TestType.SHADOW,
)
```

---

## 10. API Endpoints

### 10.1 New Endpoints (Recommended)

```
POST   /api/registry/models                 # Register model
GET    /api/registry/models                 # List models
POST   /api/registry/promote                # Request promotion
POST   /api/registry/approve/{request_id}   # Approve promotion

POST   /api/safety/evaluate                 # Run safety evaluation
GET    /api/safety/reports/{model_name}     # Get safety report

POST   /api/inference/predict               # Run inference
POST   /api/ab-test/experiments             # Start A/B experiment
GET    /api/ab-test/experiments/{name}      # Get experiment results

POST   /api/datasets/ingest                 # Ingest dataset
GET    /api/datasets/{name}/versions        # List dataset versions
POST   /api/datasets/{name}/version         # Create dataset version
```

---

## 11. Testing

### 11.1 Unit Tests

```bash
pytest tests/test_quality_gates.py
pytest tests/test_safety_probes.py
pytest tests/test_model_registry.py
pytest tests/test_serving_adapters.py
```

### 11.2 Integration Tests

```python
# Test end-to-end workflow
def test_e2e_workflow():
    # Ingest dataset
    service = DatasetIngestionService(output_dir="/tmp/test")
    manifest_path = service.ingest_dataset(...)

    # Train model
    recipe = VisionRecipe(output_dir="/tmp/test_output")
    output = recipe.run(...)

    # Evaluate
    evaluator = SafetyEvaluator()
    report = evaluator.evaluate(...)

    # Register
    registry = ModelRegistry()
    version = registry.register_model(...)

    assert version is not None
```

---

## 12. Migration Guide

### 12.1 Existing Datasets

Convert existing datasets to JSONL manifest format:

```python
from pathlib import Path
from spark_trainer.ingestion.dataset_service import DatasetIngestionService

# Re-ingest existing dataset
service = DatasetIngestionService(output_dir="./datasets")

service.ingest_dataset(
    dataset_name="existing_dataset",
    source_paths=["./old_datasets/existing_dataset"],
    content_type="auto",
)
```

### 12.2 Existing Models

Register existing models in the model registry:

```python
from spark_trainer.inference.model_registry import ModelRegistry, ModelMetadata

registry = ModelRegistry()

# Register existing model
metadata = ModelMetadata(
    name="existing_model",
    version="v1",
    format=ModelFormat.PYTORCH,
    framework="pytorch",
    task_type="image-classification",
    architecture="resnet50",
    parameters={},
    metrics={'accuracy': 0.92},
    tags=['vision', 'resnet'],
)

registry.register_model(
    name="existing_model",
    model_path="./models/existing_model.pt",
    metadata=metadata,
)
```

---

## 13. Troubleshooting

### 13.1 Common Issues

**lakeFS Connection:**
```bash
# Check lakeFS is running
curl http://localhost:8000/_health

# Check credentials
echo $LAKEFS_ACCESS_KEY
```

**GPU Scheduler:**
```bash
# Check nvidia-smi
nvidia-smi

# Check pynvml installation
pip install pynvml
```

**Serving Adapters:**
```bash
# Test vLLM endpoint
curl http://localhost:8000/health

# Test TGI endpoint
curl http://localhost:8080/health
```

### 13.2 Debug Logging

Enable debug logging:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

---

## 14. Performance Optimization

### 14.1 Dataset Ingestion

- Use parallel workers for large datasets
- Enable quality gate caching
- Batch CLIP embedding computation

### 14.2 Model Training

- Use gradient checkpointing for large models
- Enable Flash Attention
- Use FSDP for models > 10B parameters
- Enable mixed precision (bf16 on A100)

### 14.3 Inference

- Use vLLM for LLM inference (5-10x faster)
- Enable continuous batching
- Use tensor parallelism for large models
- Cache KV values (PagedAttention)

---

## 15. Security Considerations

### 15.1 PII Protection

- Always enable PII redaction for user-generated content
- Use Presidio for advanced entity recognition
- Log PII detection events for compliance

### 15.2 Model Security

- Verify model signatures before deployment
- Use signed model bundles (SHA256 checksums)
- Scan models for vulnerabilities
- Enable jailbreak detection in production

### 15.3 API Security

- Use API keys for serving endpoints
- Rate limit inference requests
- Monitor for adversarial inputs

---

## 16. References

- [lakeFS Documentation](https://docs.lakefs.io/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [TGI Documentation](https://huggingface.co/docs/text-generation-inference)
- [Triton Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Presidio Documentation](https://microsoft.github.io/presidio/)
- [Detoxify](https://github.com/unitaryai/detoxify)

---

## 17. Support

For issues or questions:
- GitHub Issues: https://github.com/def1ant1/SparkTrainer/issues
- Documentation: https://github.com/def1ant1/SparkTrainer/tree/main/docs
