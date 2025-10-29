# Models

Manage, compare, and export your trained models.

## Model Registry

All trained models are automatically registered and versioned. Each model includes:

- **Config**: Model architecture and hyperparameters
- **Weights**: Trained parameters
- **Metadata**: Training metrics, dataset info, creation date
- **Artifacts**: Checkpoints, logs, plots

## Features

### Model Comparison

Select multiple models to compare:
- Training metrics side-by-side
- Configuration differences
- Performance benchmarks (MMLU, COCO, etc.)

### Export to HuggingFace

Push models directly to your HuggingFace account:
1. Click **Export to HF** on any model
2. Choose repository name
3. Select visibility (public/private)
4. Auto-generates Model Card with training details

### Model Merging

Combine multiple models into one:
- **Weighted Average**: Simple interpolation
- **SLERP**: Spherical interpolation (preserves norms)
- **Layerwise**: Merge specific layers only

See [Model Merge Wizard](#) for details.

## Model Card

Every model has an automatically generated Model Card including:
- Model architecture
- Training dataset
- Hyperparameters
- Evaluation results
- Intended use and limitations
- Citation information
