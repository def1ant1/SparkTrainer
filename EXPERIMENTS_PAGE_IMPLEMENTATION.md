# Experiments Page — "Pick a Base Model" & Full Experiment Spec

This document describes the implementation of the comprehensive experiment creation feature with base model selection, recipe templates, and full experiment specification.

## Overview

This feature introduces a complete experiment creation workflow that includes:
- Base model registry and selection
- Recipe templates for different training styles (LoRA, QLoRA, Full-FT, etc.)
- Adapter management and composition
- Smart defaults for hyperparameters
- Compatibility validation
- Pre-flight checks with VRAM and throughput estimates
- Comprehensive experiment specification

## Architecture

### Backend Components

#### 1. Database Models (`backend/models.py`)

**New Tables:**
- `base_models` - Registry of available base models with properties:
  - Model metadata (name, family, params, dtype, architecture)
  - Capabilities (trainable, servable, quantized, GGUF)
  - Stage (staging, production)
  - HuggingFace integration
  - Tokenizer information

- `recipes` - Training recipe templates:
  - Recipe type (lora, qlora, full_ft, vision_lora, etc.)
  - Modality (text, image, audio, multimodal)
  - Default configurations
  - Compatibility constraints

- `adapters` - Adapter/LoRA registry:
  - Base model association
  - Adapter properties (type, rank, alpha, dropout)
  - Training status and metrics
  - Composition support

**Enhanced Experiment Model:**
- Added `base_model_id`, `recipe_id` foreign keys
- New JSON fields for comprehensive config:
  - `adapters` - List of attached adapters
  - `train` - Training hyperparameters
  - `strategy` - Distributed training strategy
  - `resources` - GPU allocation and partitioning
  - `eval` - Evaluation configuration
  - `export` - Export formats
  - `preflight_summary` - Cached preflight results

#### 2. Compatibility Engine (`backend/compatibility_engine.py`)

Validates experiment configurations at creation time:
- **Modality Compatibility** - Ensures model/dataset modalities align
- **Recipe Requirements** - Validates trainable, quantized, architecture constraints
- **Dataset Schema** - Checks for required columns (text+labels, image+captions, etc.)
- **Adapter Compatibility** - Verifies adapter rank/alpha match model hidden size
- **Machine-Readable Hints** - Provides actionable error messages

**Key Functions:**
- `check_compatibility()` - Main validation function
- `suggest_recipe()` - Recommends optimal recipes based on model/dataset
- `get_dataset_schema()` - Infers schema from metadata

#### 3. Smart Defaults Calculator (`backend/smart_defaults.py`)

Calculates intelligent hyperparameter defaults:
- **Training Params** - Learning rate, batch size, grad accumulation based on:
  - Model size (params_b)
  - Dataset size (num_samples)
  - Recipe type (LoRA uses higher LR, full FT uses lower)
  - Target effective batch size (64-128)

- **Strategy Selection** - Chooses distributed strategy:
  - Single GPU: No strategy
  - <7B params: DDP
  - 7-70B params: FSDP
  - >70B params: DeepSpeed

- **Resource Estimation** - Calculates VRAM requirements:
  - Model memory (params × dtype bytes)
  - Optimizer states (Adam = 8 bytes/param)
  - Gradients (fp16 = 2 bytes/param)
  - Activations (batch-dependent)
  - Overhead buffers (10-20%)

- **Throughput Estimation** - Predicts training speed:
  - GPU FLOPs (H100, A100, V100, etc.)
  - FLOPs per token (6 × params × seq_length)
  - Efficiency factor (~50% real-world)
  - Time per step (ms)

#### 4. API Endpoints (`backend/experiment_api.py`)

**Base Models:**
- `GET /api/base-models` - List with filters (family, stage, trainable, search)
- `POST /api/base-models` - Create new base model
- `GET /api/base-models/<id>` - Get model details
- `PUT /api/base-models/<id>` - Update model
- `DELETE /api/base-models/<id>` - Delete model

**Recipes:**
- `GET /api/recipes` - List with filters (modality, type, active_only)
- `POST /api/recipes` - Create new recipe
- `GET /api/recipes/<id>` - Get recipe details
- `PUT /api/recipes/<id>` - Update recipe
- `DELETE /api/recipes/<id>` - Delete recipe

**Adapters:**
- `GET /api/adapters` - List with filters (base_model_id, type, status)
- `POST /api/adapters` - Create new adapter
- `GET /api/adapters/<id>` - Get adapter details
- `PUT /api/adapters/<id>` - Update adapter
- `DELETE /api/adapters/<id>` - Delete adapter

**Experiment Utilities:**
- `POST /api/experiments/preflight` - Run compatibility checks and estimate resources
  - Returns: `{ ok, warnings[], errors[], estimated_vram_mb, vram_breakdown, time_per_step_ms, throughput }`

- `POST /api/experiments/smart-defaults` - Calculate smart defaults
  - Returns: `{ train: {...}, strategy: {...}, resources: {...}, eval: {...}, export: [...] }`

#### 5. Database Migration (`backend/migrations/versions/003_add_base_models_recipes_adapters.py`)

Alembic migration that:
- Creates `base_models`, `recipes`, `adapters` tables
- Adds new columns to `experiments` table
- Creates indexes for performance (base_model_id, recipe_id, family, stage, modality)
- Includes downgrade path for rollback

#### 6. Sample Data Seeder (`backend/seed_experiment_data.py`)

Seeds the database with sample data:
- **6 Base Models:**
  - Llama-2-7B, Llama-2-13B, Llama-2-70B (text)
  - Mistral-7B-v0.1 (text)
  - Qwen-7B-Chat (text, staging)
  - ViT-B-16 (image)

- **5 Recipes:**
  - QLoRA (recommended for LLMs)
  - LoRA
  - Full Fine-Tune
  - Vision LoRA
  - Prompt Tuning

- **2 Sample Adapters:**
  - Llama-2-7B Code Adapter
  - Llama-2-7B Math Adapter

Usage: `python backend/seed_experiment_data.py`

### Frontend Components

#### 1. CreateExperiment Component (`frontend/src/components/CreateExperiment.jsx`)

Comprehensive experiment creation UI with sections:

**Section Flow:**
1. **Project Selection** - Dropdown to select project
2. **Base Model Picker** - Searchable list with filters:
   - Search by name/family
   - Filter by stage (production/staging)
   - Filter by trainable
   - Shows inline chips: Trainable, Servable, Quantized, GGUF, Stage

3. **Dataset Selection** - Filtered by compatibility:
   - Only shows datasets with compatible modalities
   - Displays verification status
   - Shows sample count and version

4. **Recipe Template** - Card-based selector:
   - Recommended recipes highlighted
   - Shows description, requirements, min GPU memory
   - Recipe type badges

5. **Adapters (Optional)** - Multi-select:
   - Only shows adapters for selected base model
   - Displays rank, alpha, type
   - Status filtering (ready adapters only)

6. **Hyperparameters** - Smart defaults with override:
   - Main params: max_steps, batch_size, learning_rate, etc.
   - "Use Smart Defaults" button
   - Advanced accordion: scheduler, weight decay, seed, etc.

7. **Resources & Strategy** - GPU configuration:
   - Number of GPUs
   - GPU type selector (H100, A100, V100, etc.)
   - Distribution strategy (auto, DDP, FSDP, DeepSpeed)
   - Mixed precision (fp16, bf16)

8. **Eval & Export** - Evaluation and export config:
   - Eval interval (steps)
   - Eval suites (perplexity, MMLU, accuracy, etc.)
   - Export formats (safetensors, ONNX, GGUF)

9. **Pre-flight Checks** - Real-time validation:
   - Status indicator (Ready/Issues)
   - VRAM breakdown (model, optimizer, gradients, activations)
   - Throughput estimate (tokens/sec, time per step)
   - Warnings (yellow) and errors (red)
   - Disables "Create" button if errors present

**Features:**
- Real-time preflight checks triggered on config changes
- Smart defaults auto-calculated when model/dataset/recipe selected
- Compatibility filtering (only shows compatible datasets/recipes)
- Inline error messages and warnings
- Responsive grid layout (2 columns on large screens)

**State Management:**
- All form state managed in React hooks
- Debounced preflight checks to avoid excessive API calls
- Loading states for async operations
- Error handling and display

## Data Flow

```
User Action → Form State Update → Smart Defaults (optional) → Preflight Check
     ↓                                                               ↓
Create Button ← Enable/Disable ← Validation ← Compatibility Engine
     ↓
POST /api/experiments → Database → Success/Error
```

## Usage

### 1. Setup Database

```bash
# Run migration
cd backend
alembic upgrade head

# Seed sample data
python seed_experiment_data.py
```

### 2. Backend

The experiment API is automatically registered as a Flask blueprint in `app.py`.

### 3. Frontend

Import and use the `CreateExperiment` component:

```jsx
import CreateExperiment from './components/CreateExperiment';

function ExperimentsPage({ api, currentProject }) {
  const [showCreateModal, setShowCreateModal] = useState(false);

  return (
    <div>
      <button onClick={() => setShowCreateModal(true)}>
        Create Experiment
      </button>

      {showCreateModal && (
        <CreateExperiment
          onClose={(result) => {
            setShowCreateModal(false);
            if (result) {
              // Refresh experiments list
            }
          }}
          api={api}
          currentProject={currentProject}
        />
      )}
    </div>
  );
}
```

## API Examples

### Get Base Models with Filters

```bash
curl "http://localhost:5000/api/base-models?stage=production&trainable=true&search=llama"
```

### Run Preflight Check

```bash
curl -X POST http://localhost:5000/api/experiments/preflight \
  -H "Content-Type: application/json" \
  -d '{
    "base_model_id": "uuid",
    "dataset_id": "uuid",
    "recipe_id": "uuid",
    "train": {
      "global_batch_size": 4,
      "grad_accum": 4
    },
    "resources": {
      "gpus": 2,
      "gpu_type": "A100"
    }
  }'
```

Response:
```json
{
  "ok": true,
  "warnings": [
    "WARNING: Dataset is very small (500 samples). Consider using a larger dataset."
  ],
  "errors": [],
  "estimated_vram_mb": 45056,
  "vram_breakdown": {
    "model_gb": 14.0,
    "optimizer_gb": 0.56,
    "gradients_gb": 0.14,
    "activations_gb": 28.0,
    "total_with_buffer_gb": 44.0
  },
  "time_per_step_ms": 235,
  "throughput": {
    "tokens_per_sec": 1245,
    "samples_per_sec": 0.61
  }
}
```

### Get Smart Defaults

```bash
curl -X POST http://localhost:5000/api/experiments/smart-defaults \
  -H "Content-Type: application/json" \
  -d '{
    "base_model_id": "uuid",
    "dataset_id": "uuid",
    "recipe_id": "uuid",
    "num_gpus": 2,
    "gpu_type": "A100"
  }'
```

Response:
```json
{
  "train": {
    "max_steps": 3000,
    "global_batch_size": 8,
    "grad_accum": 4,
    "learning_rate": 0.0002,
    "warmup_steps": 300,
    "weight_decay": 0.01,
    "lr_scheduler": "cosine",
    "seed": 42,
    "checkpoint_interval": 300,
    "max_grad_norm": 1.0
  },
  "strategy": {
    "type": "ddp",
    "mixed_precision": "bf16"
  },
  "resources": {
    "gpus": 2,
    "estimated_vram_per_gpu_gb": 22.0,
    "total_vram_gb": 44.0,
    "gpu_type": "A100"
  },
  "eval": {
    "suites": ["perplexity", "mmlu", "hellaswag"],
    "interval": 500,
    "save_best": true
  },
  "export": ["safetensors", "gguf"]
}
```

## Compatibility Rules

### Modality Compatibility Matrix

| Model Modality | Compatible Dataset Modalities |
|---------------|------------------------------|
| text          | text, multimodal             |
| image         | image, multimodal            |
| audio         | audio, multimodal            |
| video         | video, multimodal            |
| multimodal    | text, image, audio, video, multimodal |

### Recipe Requirements

| Recipe Type     | Trainable? | Quantized OK? | Min Hidden Size | Dataset Schema |
|----------------|-----------|---------------|-----------------|----------------|
| LoRA           | Required  | Yes           | 64              | text+labels, image+labels |
| QLoRA          | Required  | Yes           | 64              | text+labels, text+text |
| Full FT        | Required  | No            | -               | text+labels, image+labels |
| Vision LoRA    | Required  | No            | 64              | image+labels, image+captions |
| Prompt Tuning  | Required  | Yes           | -               | text+text, text+labels |

### Adapter Compatibility

- Adapter `base_model_id` must match experiment's base model
- Adapter `rank` should be < model `hidden_size`
- Adapter `status` should be "ready" for use

## Edge Cases Handled

1. **Quantized Base Model:**
   - Allow LoRA/QLoRA
   - Block full FT with clear error message

2. **Encrypted/Private Models:**
   - Require credentials in metadata
   - Prompt for credentials in-flow (future enhancement)

3. **Oversized Dataset:**
   - Suggest QLoRA + FSDP
   - Show expected step time and ETA

4. **No GPU Available:**
   - Offer "Queue anyway" vs "Switch partition"
   - Show estimated queue time (future enhancement)

5. **GGUF Models:**
   - Block training entirely
   - Show error: "GGUF models cannot be trained. Use original format."

6. **Failed Dataset Integrity:**
   - Block experiment creation
   - Show error with link to dataset validation

## Future Enhancements

- [ ] Fine-Tune tab in Jobs page (quick fine-tune flow)
- [ ] Launch from Models page → Actions → Fine-Tune (preselect model)
- [ ] Context window extension configuration
- [ ] Knowledge distillation options
- [ ] Hyperparameter optimization (Optuna integration)
- [ ] Cost estimation and budget management
- [ ] Partition selection UI (if GPU partitioning enabled)
- [ ] Resume from checkpoint selection
- [ ] Experiment templates (save configurations for reuse)
- [ ] Comparison view (compare multiple experiments)

## Testing

### Unit Tests (TODO)

```bash
# Backend
pytest backend/tests/test_compatibility_engine.py
pytest backend/tests/test_smart_defaults.py
pytest backend/tests/test_experiment_api.py

# Frontend (Jest)
npm test -- CreateExperiment.test.jsx
```

### Integration Test Flow

1. Seed database with sample data
2. Open CreateExperiment modal
3. Select project → base model → dataset → recipe
4. Verify smart defaults populated
5. Verify preflight check runs automatically
6. Adjust hyperparameters
7. Verify preflight updates
8. Create experiment
9. Verify experiment created in DB with all fields
10. Verify job can be started from experiment

## File Structure

```
backend/
├── models.py                          # Enhanced models
├── experiment_api.py                  # New API routes
├── compatibility_engine.py            # Compatibility validation
├── smart_defaults.py                  # Smart defaults calculator
├── seed_experiment_data.py            # Sample data seeder
└── migrations/versions/
    └── 003_add_base_models_recipes_adapters.py

frontend/src/components/
└── CreateExperiment.jsx               # Main UI component

EXPERIMENTS_PAGE_IMPLEMENTATION.md     # This file
```

## Migration Notes

Existing experiments will continue to work with `null` values for new fields (`base_model_id`, `recipe_id`, etc.). The UI gracefully handles both old and new experiment formats.

To populate existing experiments with new fields, run:
```python
# Script to migrate existing experiments (TODO)
python backend/migrate_existing_experiments.py
```

## Contributing

When adding new recipes or base models:

1. **Recipes:** Add to `seed_experiment_data.py` or create via API
   - Define `default_config` with sensible defaults
   - List `required_fields` and `optional_fields`
   - Set `min_gpu_memory_gb` based on testing
   - Document in `supported_architectures`

2. **Base Models:** Register via API or seed script
   - Populate all metadata fields (params_b, dtype, hidden_size, etc.)
   - Set correct `stage` (staging for new models, production for vetted)
   - Add HuggingFace repo ID if available
   - Configure tokenizer path for text models

3. **Compatibility Rules:** Update `CompatibilityEngine` if adding new:
   - Modalities
   - Recipe types
   - Dataset schemas
   - Validation constraints

## License

Same as SparkTrainer project.
