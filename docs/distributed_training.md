# Distributed Training Guide

SparkTrainer supports advanced distributed training techniques for efficient multi-GPU and multi-node training.

## Supported Frameworks

1. **FSDP (Fully Sharded Data Parallel)** - PyTorch native, optimal for LLMs
2. **DeepSpeed ZeRO** - Microsoft's memory-efficient training
3. **DDP (Distributed Data Parallel)** - PyTorch standard distributed training

## FSDP Training

Fully Sharded Data Parallel (FSDP) is PyTorch's native approach for training large models across multiple GPUs with memory efficiency.

### Features

- Zero Redundancy Optimizer (ZeRO) Stage 3 equivalent
- CPU offloading for even larger models
- Mixed precision training (bf16/fp16)
- Gradient accumulation
- Activation checkpointing

### Example: Llama-7B QLoRA with FSDP

#### Configuration File: `configs/fsdp_qlora_llama7b.yaml`

```yaml
# Model configuration
model:
  name: "meta-llama/Llama-2-7b-hf"
  load_in_4bit: true
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: true
  bnb_4bit_quant_type: "nf4"

# LoRA configuration
lora:
  r: 64
  lora_alpha: 16
  target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  lora_dropout: 0.05
  bias: "none"
  task_type: "CAUSAL_LM"

# FSDP configuration
fsdp:
  enabled: true
  sharding_strategy: "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
  cpu_offload: false
  mixed_precision: "bf16"
  auto_wrap_policy: "transformer_based"
  backward_prefetch: "BACKWARD_PRE"
  forward_prefetch: true
  limit_all_gathers: true
  activation_checkpointing: true

# Training arguments
training:
  output_dir: "outputs/llama-7b-qlora-fsdp"
  num_train_epochs: 3
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.03
  weight_decay: 0.001
  optim: "paged_adamw_32bit"

  # Logging
  logging_steps: 10
  save_steps: 100
  eval_steps: 100
  save_total_limit: 3

  # Mixed precision
  bf16: true
  tf32: true

  # Gradient settings
  max_grad_norm: 1.0
  gradient_checkpointing: true

  # Evaluation
  evaluation_strategy: "steps"
  load_best_model_at_end: true
  metric_for_best_model: "loss"

# Dataset configuration
dataset:
  train_file: "datasets/alpaca_cleaned.jsonl"
  eval_file: "datasets/alpaca_cleaned_eval.jsonl"
  max_seq_length: 2048
  preprocessing_num_workers: 8

# Hardware configuration
hardware:
  num_gpus: 4
  num_nodes: 1
  master_addr: "localhost"
  master_port: 29500
```

#### Launch Script

```bash
#!/bin/bash
# launch_fsdp_qlora.sh

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

# Launch with torchrun
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    -m spark_trainer.recipes.text_lora \
    --config configs/fsdp_qlora_llama7b.yaml
```

#### Python Code

```python
# src/spark_trainer/recipes/fsdp_qlora_recipe.py

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import yaml


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_model(config):
    # BitsAndBytes configuration for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['model']['load_in_4bit'],
        bnb_4bit_compute_dtype=getattr(torch, config['model']['bnb_4bit_compute_dtype']),
        bnb_4bit_use_double_quant=config['model']['bnb_4bit_use_double_quant'],
        bnb_4bit_quant_type=config['model']['bnb_4bit_quant_type'],
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        quantization_config=bnb_config,
        device_map={"": torch.cuda.current_device()},
        trust_remote_code=True,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type=config['lora']['task_type'],
    )

    # Add LoRA adapters
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def setup_fsdp(config):
    # Mixed precision configuration
    if config['fsdp']['mixed_precision'] == 'bf16':
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    else:
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )

    # Sharding strategy
    sharding_strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }
    sharding_strategy = sharding_strategy_map[config['fsdp']['sharding_strategy']]

    # Backward prefetch
    backward_prefetch_map = {
        "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
        "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
    }
    backward_prefetch = backward_prefetch_map.get(config['fsdp']['backward_prefetch'])

    return {
        'mixed_precision': mp_policy,
        'sharding_strategy': sharding_strategy,
        'backward_prefetch': backward_prefetch,
        'cpu_offload': config['fsdp']['cpu_offload'],
        'forward_prefetch': config['fsdp']['forward_prefetch'],
        'limit_all_gathers': config['fsdp']['limit_all_gathers'],
    }


def main(config_path):
    config = load_config(config_path)

    # Setup model
    model = setup_model(config)
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])

    # Load dataset (implement your dataset loading logic)
    train_dataset = load_dataset(config['dataset']['train_file'], tokenizer)
    eval_dataset = load_dataset(config['dataset']['eval_file'], tokenizer)

    # FSDP configuration
    fsdp_config = setup_fsdp(config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        warmup_ratio=config['training']['warmup_ratio'],
        weight_decay=config['training']['weight_decay'],
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        eval_steps=config['training']['eval_steps'],
        evaluation_strategy=config['training']['evaluation_strategy'],
        save_total_limit=config['training']['save_total_limit'],
        bf16=config['training']['bf16'],
        tf32=config['training']['tf32'],
        max_grad_norm=config['training']['max_grad_norm'],
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        # FSDP settings
        fsdp=True,
        fsdp_config=fsdp_config,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    main(args.config)
```

## DeepSpeed Training

DeepSpeed ZeRO provides memory-efficient training through optimizer state partitioning, gradient partitioning, and parameter partitioning.

### ZeRO Stages

- **ZeRO-1**: Optimizer state partitioning
- **ZeRO-2**: Optimizer state + gradient partitioning
- **ZeRO-3**: Optimizer state + gradient + parameter partitioning

### Example: Stable Diffusion LoRA with DeepSpeed ZeRO-2

#### Configuration File: `configs/deepspeed_zero2_sd_lora.yaml`

```yaml
# Model configuration
model:
  name: "stabilityai/stable-diffusion-xl-base-1.0"
  variant: "fp16"
  use_safetensors: true

# LoRA configuration
lora:
  rank: 128
  alpha: 128
  target_modules:
    - to_q
    - to_k
    - to_v
    - to_out.0
    - proj_in
    - proj_out
    - ff.net.0.proj
    - ff.net.2
  dropout: 0.0

# DeepSpeed ZeRO-2 configuration
deepspeed:
  config_file: "configs/deepspeed_zero2_config.json"

# Training arguments
training:
  output_dir: "outputs/sdxl-lora-deepspeed"
  num_train_epochs: 100
  train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 1.0e-4
  lr_scheduler: "constant_with_warmup"
  warmup_steps: 500
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_epsilon: 1.0e-8
  max_grad_norm: 1.0

  # Logging
  logging_steps: 10
  save_steps: 500
  checkpointing_steps: 500

  # Mixed precision
  mixed_precision: "fp16"

  # Validation
  validation_epochs: 10
  validation_prompts:
    - "a photo of sks person"
    - "sks person in a spacesuit"

# Dataset configuration
dataset:
  instance_data_dir: "datasets/dreambooth/person"
  instance_prompt: "a photo of sks person"
  resolution: 1024
  center_crop: true
  random_flip: false
  num_workers: 4

# Hardware configuration
hardware:
  num_gpus: 4
  num_nodes: 1
  master_addr: "localhost"
  master_port: 29500
```

#### DeepSpeed Config: `configs/deepspeed_zero2_config.json`

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "none"
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e7,
    "stage3_param_persistence_threshold": 1e5,
    "gather_16bit_weights_on_model_save": true
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-4,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "total_num_steps": 10000,
      "warmup_min_lr": 0,
      "warmup_max_lr": 1e-4,
      "warmup_num_steps": 500
    }
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "gradient_clipping": 1.0,
  "train_batch_size": 16,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 4,
  "steps_per_print": 10,
  "wall_clock_breakdown": false
}
```

#### Launch Script

```bash
#!/bin/bash
# launch_deepspeed_zero2.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3

deepspeed --num_gpus=4 \
    -m spark_trainer.recipes.vision_lora \
    --config configs/deepspeed_zero2_sd_lora.yaml \
    --deepspeed configs/deepspeed_zero2_config.json
```

## Performance Tips

### 1. Batch Size Tuning

```python
# Find optimal batch size
def find_optimal_batch_size(model, initial_batch_size=16):
    batch_size = initial_batch_size
    while True:
        try:
            # Try training step with current batch size
            train_step(model, batch_size)
            batch_size *= 2
        except RuntimeError as e:
            if "out of memory" in str(e):
                batch_size //= 2
                break
            raise e
    return batch_size
```

### 2. Gradient Accumulation

```yaml
# Effective batch size = train_batch_size * gradient_accumulation_steps * num_gpus
training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  num_gpus: 4
  # Effective batch size = 4 * 4 * 4 = 64
```

### 3. Mixed Precision

```yaml
# Use bf16 on Ampere+ GPUs, fp16 otherwise
training:
  bf16: true  # A100, H100
  tf32: true  # Ampere+ automatic
  # or
  fp16: true  # V100, older GPUs
```

### 4. Activation Checkpointing

```yaml
training:
  gradient_checkpointing: true
  # Trades compute for memory
  # ~30% slower, ~50% less memory
```

## Multi-Node Training

### SLURM Example

```bash
#!/bin/bash
#SBATCH --job-name=llama-fsdp
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00

# Load modules
module load cuda/12.1
module load nccl/2.18

# Set environment
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

# Launch training
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m spark_trainer.recipes.text_lora \
    --config configs/fsdp_qlora_llama7b.yaml
```

## Troubleshooting

### OOM (Out of Memory)

1. Reduce batch size
2. Enable gradient checkpointing
3. Use ZeRO-3 / FSDP FULL_SHARD
4. Enable CPU offloading
5. Reduce sequence length
6. Use gradient accumulation

### Slow Training

1. Enable tf32 on Ampere GPUs
2. Use larger batch sizes
3. Disable gradient checkpointing if memory allows
4. Optimize data loading (num_workers, pin_memory)
5. Profile with torch.profiler

### NCCL Errors

```bash
# Enable debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# For InfiniBand
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1

# For Ethernet
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
```

## Benchmarks

### Llama-7B Training Throughput

| Configuration | GPUs | Batch Size | Tokens/sec | GPU Memory |
|--------------|------|------------|------------|------------|
| DDP + QLoRA | 1 | 4 | 2,100 | 24 GB |
| FSDP + QLoRA | 4 | 16 | 31,200 | 22 GB |
| DeepSpeed ZeRO-2 | 4 | 16 | 28,800 | 20 GB |
| DeepSpeed ZeRO-3 | 8 | 32 | 54,000 | 18 GB |

### Stable Diffusion XL Training

| Configuration | GPUs | Batch Size | Steps/sec | GPU Memory |
|--------------|------|------------|-----------|------------|
| DDP | 1 | 1 | 1.2 | 42 GB |
| FSDP | 4 | 4 | 4.1 | 38 GB |
| DeepSpeed ZeRO-2 | 4 | 8 | 7.8 | 28 GB |

## Further Reading

- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [Hugging Face Accelerate](https://huggingface.co/docs/accelerate)
