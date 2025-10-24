# Example Training Configurations

This document provides example configurations for common training scenarios on the DGX AI Trainer.

## Table of Contents
1. [Image Classification from Scratch](#image-classification-from-scratch)
2. [Fine-tuning ResNet](#fine-tuning-resnet)
3. [Text Classification with BERT](#text-classification-with-bert)
4. [Training Large Language Models](#training-large-language-models)
5. [Custom Architecture](#custom-architecture)

---
## Hyperparameter Optimization (Optuna)

Define a search space and budget; the backend routes HF finetune to an Optuna runner when `config.hpo.enabled` is true.

```json
{
  "name": "GPT-2 HPO",
  "type": "finetune",
  "framework": "huggingface",
  "config": {
    "model_name": "gpt2",
    "task_type": "generation",
    "epochs": 3,
    "batch_size": 8,
    "learning_rate": 3e-5,
    "hpo": {
      "enabled": true,
      "metric": "eval_loss",
      "direction": "minimize",
      "max_trials": 10,
      "timeout_seconds": 0,
      "workers": 1,
      "sampler": "tpe",
      "pruner": "median",
      "trial_epochs": 1,
      "space": [
        { "name": "learning_rate", "type": "float", "low": 1e-6, "high": 5e-4, "log": true },
        { "name": "batch_size", "type": "int", "low": 4, "high": 32, "step": 4 },
        { "name": "lora.r", "type": "int", "low": 4, "high": 64, "step": 4 }
      ]
    }
  }
}
```

Results are saved under `models/<job_id>/hpo_results.json` and `hpo_trials.json` and can be viewed via the HPO page in the UI.

---

## Image Classification from Scratch

Train a custom ResNet-style model for image classification.

```json
{
  "name": "Custom Image Classifier",
  "type": "train",
  "framework": "pytorch",
  "config": {
    "architecture": "resnet",
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "num_classes": 10,
    "num_blocks": [2, 2, 2, 2],
    "channels": [64, 128, 256, 512],
    "num_samples": 50000
  }
}
```

**Use Case**: Training an image classifier from scratch on a custom dataset (e.g., medical images, satellite imagery).

---

## Fine-tuning ResNet

Fine-tune a pre-trained ResNet18 model for a new classification task.

```json
{
  "name": "ResNet18 Fine-tune",
  "type": "finetune",
  "framework": "pytorch",
  "config": {
    "model_source": "torchvision",
    "model_name": "resnet18",
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 0.0001,
    "optimizer": "adam",
    "weight_decay": 0.0001,
    "freeze_layers": true,
    "use_scheduler": true,
    "num_classes": 5,
    "num_samples": 5000
  }
}
```

**Use Case**: Transfer learning for image classification with limited data. The frozen layers retain general image features while the final layer learns task-specific features.

---

## Text Classification with BERT

Fine-tune BERT for sentiment analysis or text classification.

```json
{
  "name": "BERT Sentiment Analysis",
  "type": "finetune",
  "framework": "huggingface",
  "config": {
    "model_name": "bert-base-uncased",
    "task_type": "classification",
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "max_length": 128,
    "num_classes": 2,
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "num_samples": 10000
  }
}
```

**Use Case**: Sentiment analysis, spam detection, or any binary/multi-class text classification task.

---

## Training Large Language Models

Fine-tune GPT-2 for text generation tasks.

```json
{
  "name": "GPT-2 Fine-tune",
  "type": "finetune",
  "framework": "huggingface",
  "config": {
    "model_name": "gpt2",
    "task_type": "generation",
    "epochs": 5,
    "batch_size": 8,
    "learning_rate": 5e-5,
    "max_length": 256,
    "weight_decay": 0.01,
    "warmup_steps": 1000,
    "num_samples": 20000
  }
}
```

**Use Case**: Chatbots, code generation, creative writing, or domain-specific text generation.

---

## Advanced Training Strategies (LoRA/QLoRA optional)

Multi-stage pipeline with curriculum learning, knowledge distillation, mixed precision, and distributed options for Hugging Face models.

```json
{
  "name": "HF Advanced Fine-tune",
  "type": "finetune",
  "framework": "huggingface",
  "config": {
    "model_name": "gpt2",
    "task_type": "generation",
    "max_length": 256,
    "compute_dtype": "bf16",

    "lora": {
      "enabled": true,
      "r": 8,
      "alpha": 16,
      "dropout": 0.05,
      "qlora": false
    },

    "stages": [
      {"name": "warmup", "epochs": 1, "learning_rate": 5e-5, "batch_size": 8,
       "curriculum": {"incremental": true, "mode": "length", "start_frac": 0.4, "end_frac": 0.8}},
      {"name": "main", "epochs": 2, "learning_rate": 3e-5, "batch_size": 8},
      {"name": "refinement", "epochs": 1, "learning_rate": 1e-5, "batch_size": 8}
    ],

    "distillation": {
      "enabled": true,
      "teacher_model": "gpt2-medium",
      "temperature": 2.0,
      "alpha_distill": 0.5,
      "alpha_ce": 0.5
    },

    "grad_clip": {"type": "norm", "max_grad_norm": 1.0},

    "distributed": {
      "deepspeed": null,
      "fsdp": null,
      "gradient_accumulation_steps": 2
    }
  }
}
```

Notes:
- Set `precision` to `fp16`/`bf16`/`fp8` or use `compute_dtype` (fp16/bf16) to control mixed precision.
- Use `distributed.deepspeed` with a path to a DeepSpeed JSON config, or `distributed.fsdp` with an FSDP config string.
- Multi-node/multi-GPU execution typically requires launching with `torchrun` outside the backend process.

---

## DeepSpeed Config Templates

You can use these ready-to-go configs with Hugging Face Trainer by pointing `config.distributed.deepspeed` to their path.

- ZeRO-2: `training_scripts/deepspeed/zero2.json`
- ZeRO-3: `training_scripts/deepspeed/zero3.json`

Example snippet:

```json
{
  "name": "HF + DeepSpeed",
  "type": "finetune",
  "framework": "huggingface",
  "config": {
    "model_name": "gpt2",
    "task_type": "generation",
    "stages": [{"name":"main","epochs":2,"learning_rate":3e-5,"batch_size":8}],
    "distributed": {
      "deepspeed": "training_scripts/deepspeed/zero2.json",
      "gradient_accumulation_steps": 2
    }
  }
}
```

Note: Ensure the environment has DeepSpeed installed and launch via `torchrun` for multi-GPU/multi-node setups.

Torchrun quickstart:

```
torchrun \
  --nproc_per_node=4 \
  --master_port=29500 \
  training_scripts/finetune_huggingface.py \
  --job-id myjob --config '{"model_name":"gpt2","task_type":"generation","distributed":{"deepspeed":"training_scripts/deepspeed/zero3.json"}}'
```

Adjust `--nproc_per_node` to the number of GPUs on your node. For multi-node, also set `--nnodes` and `--node_rank`.

---
## Ready-to-run Advanced Examples

- Advanced HF + DeepSpeed (ZeRO-2): `jobs/examples/hf_advanced_deepspeed.json`
- Advanced HF + DeepSpeed (ZeRO-3): `jobs/examples/hf_advanced_deepspeed_zero3.json`

Load one of these examples via the Jobs API or copy its `config` into the Wizard’s Review step.

---

## Custom Architecture

Train a custom fully-connected network for structured data.

```json
{
  "name": "Custom Neural Network",
  "type": "train",
  "framework": "pytorch",
  "config": {
    "architecture": "custom",
    "epochs": 100,
    "batch_size": 128,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "activation": "relu",
    "dropout": 0.3,
    "input_size": 784,
    "output_size": 10,
    "hidden_layers": [1024, 512, 256, 128],
    "num_samples": 60000,
    "loss": "cross_entropy"
  }
}
```

**Use Case**: Tabular data classification, regression tasks, or custom ML problems.

---

## Advanced Configurations

### Multi-GPU Training

For utilizing multiple GPUs on your DGX Spark:

```json
{
  "name": "Multi-GPU Training",
  "type": "train",
  "framework": "pytorch",
  "config": {
    "architecture": "resnet",
    "epochs": 100,
    "batch_size": 256,
    "learning_rate": 0.01,
    "distributed": true,
    "num_gpus": 4,
    "mixed_precision": true,
    "gradient_accumulation_steps": 4
  }
}
```

### Large-Scale Fine-tuning

For fine-tuning large models like LLaMA or GPT-style models:

```json
{
  "name": "Large Model Fine-tune",
  "type": "finetune",
  "framework": "huggingface",
  "config": {
    "model_name": "meta-llama/Llama-2-7b-hf",
    "task_type": "generation",
    "epochs": 3,
    "batch_size": 1,
    "learning_rate": 1e-5,
    "max_length": 512,
    "gradient_accumulation_steps": 32,
    "gradient_checkpointing": true,
    "mixed_precision": "fp16",
    "lora_enabled": true,
    "lora_r": 8,
    "lora_alpha": 16
  }
}
```

---

## Performance Optimization Tips

### For DGX Spark Systems

1. **Batch Size Optimization**
   - Start with smaller batch sizes and increase until GPU memory is ~90% utilized
   - Use gradient accumulation for effective larger batch sizes

2. **Learning Rate Scaling**
   - Scale learning rate linearly with batch size: `lr = base_lr * (batch_size / base_batch_size)`
   - Use warmup for large learning rates

3. **Mixed Precision Training**
   - Enable TF32 for A100 GPUs: adds speed without accuracy loss
   - Use FP16 for memory savings and speed improvements

4. **Data Loading**
   - Use multiple workers for data loading (4-8 workers per GPU)
   - Pin memory for faster host-to-device transfers
   - Prefetch data to overlap computation and I/O

### Example Optimized Configuration

```json
{
  "name": "Optimized Training",
  "type": "train",
  "framework": "pytorch",
  "config": {
    "architecture": "resnet",
    "epochs": 90,
    "batch_size": 512,
    "learning_rate": 0.1,
    "optimizer": "sgd",
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "num_classes": 1000,
    
    "mixed_precision": true,
    "gradient_accumulation_steps": 4,
    "num_workers": 8,
    "pin_memory": true,
    "prefetch_factor": 2,
    
    "lr_schedule": "cosine",
    "warmup_epochs": 5,
    "label_smoothing": 0.1
  }
}
```

---

## Troubleshooting

### Out of Memory Errors

If you encounter CUDA OOM errors:
1. Reduce batch size by half
2. Enable gradient accumulation
3. Enable gradient checkpointing
4. Use mixed precision training

### Slow Training

If training is slower than expected:
1. Increase batch size (if memory allows)
2. Optimize data loading (more workers, pin memory)
3. Profile your code to find bottlenecks
4. Ensure GPUs are being fully utilized (check with `nvidia-smi`)

---

## Best Practices

1. **Start Small**: Begin with a small model and dataset to verify your pipeline works
2. **Monitor Metrics**: Watch training/validation loss to detect overfitting early
3. **Save Checkpoints**: Regularly save model checkpoints during training
4. **Use Validation**: Always evaluate on held-out validation data
5. **Log Everything**: Keep detailed logs of experiments for reproducibility
6. **Experiment Tracking**: Use consistent naming for jobs and track hyperparameters

---

## Next Steps

After successful training:
1. Evaluate model on test set
2. Export model for deployment
3. Optimize for inference (quantization, pruning)
4. Monitor model performance in production
5. Iterate based on results
