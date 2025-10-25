# Model Creation and Fine-Tuning Guide

This guide explains the improved model creation and fine-tuning capabilities in SparkTrainer.

## Overview

SparkTrainer now provides enhanced model creation capabilities with the following features:

1. **Automatic Model Registration**: Models created during training are automatically available in the models list
2. **Model Reusability**: Trained models can be selected as base models for fine-tuning
3. **Complete Weight Persistence**: Models are saved with full weights and optimizer states for continued training
4. **Pre-flight Validation**: Configuration errors are caught before training starts

## Training a Model from Scratch

### PyTorch

When training a PyTorch model from scratch, you can choose between two architectures:

#### Custom Fully-Connected Network

```json
{
  "type": "train",
  "framework": "pytorch",
  "config": {
    "architecture": "custom",
    "input_size": 784,
    "output_size": 10,
    "hidden_layers": [512, 256, 128],
    "activation": "relu",
    "dropout": 0.2,
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001
  }
}
```

#### Custom ResNet

```json
{
  "type": "train",
  "framework": "pytorch",
  "config": {
    "architecture": "resnet",
    "num_classes": 10,
    "num_blocks": [2, 2, 2, 2],
    "channels": [64, 128, 256, 512],
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001
  }
}
```

### HuggingFace

```json
{
  "type": "train",
  "framework": "huggingface",
  "config": {
    "model_name": "bert-base-uncased",
    "task_type": "classification",
    "num_classes": 2,
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 2e-5
  }
}
```

## Fine-Tuning Models

### Using Models from the Registry

You can now fine-tune any model from your models list by using `model_source: "model_id"`:

#### PyTorch Example

```json
{
  "type": "finetune",
  "framework": "pytorch",
  "config": {
    "model_source": "model_id",
    "model_id": "abc123-def456-...",
    "num_classes": 5,
    "epochs": 5,
    "batch_size": 32,
    "learning_rate": 0.0001,
    "freeze_layers": false
  }
}
```

#### HuggingFace Example

```json
{
  "type": "finetune",
  "framework": "huggingface",
  "config": {
    "model_source": "model_id",
    "model_id": "abc123-def456-...",
    "task_type": "classification",
    "num_classes": 2,
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 2e-5
  }
}
```

### Using Pre-trained Models

#### From TorchVision (PyTorch)

```json
{
  "type": "finetune",
  "framework": "pytorch",
  "config": {
    "model_source": "torchvision",
    "model_name": "resnet50",
    "num_classes": 10,
    "freeze_layers": true,
    "epochs": 5
  }
}
```

Supported torchvision models: `resnet18`, `resnet50`, `vgg16`, `densenet121`

#### From HuggingFace Hub

```json
{
  "type": "finetune",
  "framework": "huggingface",
  "config": {
    "model_source": "huggingface",
    "model_name": "bert-base-uncased",
    "task_type": "classification",
    "num_classes": 2
  }
}
```

## Model Storage and Metadata

When a model is trained, the following files are saved:

### PyTorch Models

- `model.pth`: Contains model weights (`model_state_dict`) and optimizer state (`optimizer_state_dict`)
- `best_model.pth`: Best checkpoint during fine-tuning (based on validation loss)
- `config.json`: Complete architecture configuration for model reconstruction
- `env.json`: Environment snapshot (Python version, packages, CUDA info)
- `metadata.json`: Additional metadata (tags, license, domain)

### HuggingFace Models

- `pytorch_model.bin` or `model.safetensors`: Model weights
- `config.json`: HuggingFace model configuration
- `tokenizer_config.json`, `vocab.txt`, etc.: Tokenizer files
- `metadata.json`: Training metadata and base model information
- `adapters/`: LoRA adapters (if using PEFT)

## Pre-flight Validation

Before creating a training job, you can validate your configuration:

```bash
POST /api/jobs/validate
```

Example:

```json
{
  "type": "finetune",
  "framework": "pytorch",
  "config": {
    "model_source": "model_id",
    "model_id": "invalid-id",
    "num_classes": 10
  }
}
```

Response:

```json
{
  "valid": false,
  "errors": [
    "Model with ID 'invalid-id' not found in models directory"
  ]
}
```

## Validation Checks

The system validates:

1. **Framework**: Must be `pytorch`, `tensorflow`, or `huggingface`
2. **Job Type**: Must be `train` or `finetune`
3. **Numeric Parameters**: `epochs`, `batch_size`, and `learning_rate` must be positive numbers
4. **Model Existence**: When using `model_source: "model_id"`, verifies the model exists
5. **Model Files**: Checks that required files (`model.pth`, `config.json`) are present
6. **Architecture Info**: Ensures models have complete architecture information for reconstruction
7. **Required Parameters**: Validates that required parameters are present based on architecture

## Complete Workflow Example

### 1. Train a Model from Scratch

```json
POST /api/jobs
{
  "name": "Initial Model Training",
  "type": "train",
  "framework": "pytorch",
  "config": {
    "architecture": "custom",
    "input_size": 784,
    "output_size": 10,
    "hidden_layers": [512, 256, 128],
    "epochs": 20,
    "batch_size": 64,
    "learning_rate": 0.001
  }
}
```

This creates a model with ID (job ID), e.g., `abc123-def456-ghi789`

### 2. List Models

```bash
GET /api/models
```

Your trained model will appear in the list.

### 3. Fine-tune the Model

```json
POST /api/jobs
{
  "name": "Fine-tuning on New Task",
  "type": "finetune",
  "framework": "pytorch",
  "config": {
    "model_source": "model_id",
    "model_id": "abc123-def456-ghi789",
    "num_classes": 5,
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.0001,
    "freeze_layers": false
  }
}
```

### 4. Continue Fine-tuning

You can fine-tune the fine-tuned model:

```json
POST /api/jobs
{
  "name": "Further Fine-tuning",
  "type": "finetune",
  "framework": "pytorch",
  "config": {
    "model_source": "model_id",
    "model_id": "xyz789-uvw456-rst123",
    "num_classes": 5,
    "epochs": 5,
    "learning_rate": 0.00005
  }
}
```

## Best Practices

1. **Use Meaningful Names**: Give your models descriptive names to identify them easily
2. **Validate Before Training**: Use `/api/jobs/validate` to catch errors early
3. **Save Checkpoints**: For long training runs, use checkpoint saving features
4. **Track Lineage**: Models saved with `base_model_id` maintain a record of their source
5. **Use Lower Learning Rates for Fine-tuning**: When fine-tuning, use 10-100x smaller learning rates than initial training
6. **Consider Freezing Layers**: For transfer learning, freeze early layers and only train the final layers

## Troubleshooting

### Model Not Found in List

- Check that the training job completed successfully
- Models are stored in `models/{job_id}/` directory
- Verify the job status shows "completed"

### Validation Errors

- Read the error messages carefully - they indicate specific issues
- Ensure all required parameters are present
- Check that model IDs are correct (use `/api/models` to list available models)

### Architecture Mismatch

- When fine-tuning, the new `num_classes` can differ from the original model
- The system will automatically replace the final layer to match the new number of classes
- For other architecture changes, train a new model from scratch

## Advanced Features

### LoRA Fine-tuning (HuggingFace)

```json
{
  "type": "finetune",
  "framework": "huggingface",
  "config": {
    "model_source": "model_id",
    "model_id": "abc123",
    "lora": {
      "enabled": true,
      "r": 8,
      "alpha": 16,
      "dropout": 0.05,
      "target_modules": ["q_proj", "v_proj"]
    }
  }
}
```

### Mixed Precision Training

```json
{
  "config": {
    "precision": "float16",
    "gradient_accumulation_steps": 4
  }
}
```

### Distributed Training

```json
{
  "config": {
    "distributed": {
      "enabled": true,
      "backend": "nccl"
    }
  }
}
```
