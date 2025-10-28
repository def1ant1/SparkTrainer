# SparkTrainer Examples

This directory contains comprehensive examples demonstrating advanced features of SparkTrainer.

## 📚 Overview

The examples showcase:

- **Multimodal Multistep Datasets**: Creating complex datasets with multiple modalities (vision, audio, text) and reasoning chains
- **Advanced Model Architectures**: Mixture of Experts (MoE) with 125k context windows
- **Efficient Training**: LoRA, gradient checkpointing, flash attention, and quantization
- **Model Merging**: Multiple strategies for combining models (LERP, SLERP, TIES, DARE)
- **Complete Pipelines**: End-to-end training and fine-tuning workflows

## 🚀 Quick Start

Run the complete example workflow:

```bash
python run_complete_example.py
```

This will:
1. Create a multimodal multistep dataset
2. Build a MoE model with 125k context window
3. Run the training pipeline
4. Demonstrate model merging techniques

## 📖 Individual Examples

### 1. Multimodal Multistep Dataset

**File**: `multimodal_multistep_dataset.py`

Creates a dataset with:
- Image, text, audio, and video modalities
- Multi-step reasoning chains
- Multiple task types (visual counting, audio analysis, multimodal fusion, temporal reasoning)

**Usage**:
```bash
python multimodal_multistep_dataset.py
```

**Output**:
- Dataset in `/home/user/SparkTrainer/datasets/multimodal_multistep_vqa/v1/`
- 100 samples with synthetic images
- JSONL manifest file
- README and metadata

### 2. MoE Model with 125k Context Window

**File**: `moe_model_125k_context.py`

Implements a comprehensive MoE model with:
- **Mixture of Experts**: 8 experts with top-2 routing
- **125k Context Window**: Via YaRN RoPE scaling
- **LoRA Adapters**: For efficient fine-tuning
- **Gradient Checkpointing**: For memory efficiency
- **Flash Attention**: For speed optimization
- **Quantization Support**: FP4, FP8, INT4, INT8

**Architecture Highlights**:
```python
MoEModel125k(
    vocab_size=50000,
    hidden_dim=2048,
    num_layers=24,
    num_heads=16,
    num_experts=8,
    num_experts_per_token=2,
    max_seq_len=131072,  # ~125k tokens
    use_lora=True,
    use_gradient_checkpointing=True
)
```

**Usage**:
```bash
python moe_model_125k_context.py
```

**Output**:
- Model saved to `/home/user/SparkTrainer/models/moe_125k_example/`
- Includes model weights (`model.pth`) and config (`config.json`)

### 3. Training Pipeline

**File**: `training_pipeline.py`

Demonstrates a complete training pipeline with:
- Data loading and preprocessing
- Mixed precision training (FP16/BF16)
- Gradient accumulation
- Learning rate scheduling (warmup + decay)
- Checkpointing and resumption
- Distributed training ready

**Key Features**:
- Automatic mixed precision (AMP)
- Gradient checkpointing
- LoRA adapter training
- Checkpoint saving/loading
- Training metrics logging

**Usage**:
```bash
# Make sure dataset exists first
python multimodal_multistep_dataset.py
# Then run training
python training_pipeline.py
```

**Configuration**:
Edit the config dict in `main()` to adjust:
- Learning rate
- Batch size
- Number of epochs
- Gradient checkpointing
- LoRA settings

### 4. Model Merging Pipeline

**File**: `model_merging_pipeline.py`

Demonstrates various model merging strategies:

#### Merging Strategies:

1. **Linear Interpolation (LERP)**
   - Simple weighted average
   - Best for: Similar models
   ```python
   merged = pipeline.linear_merge([model_a, model_b], weights=[0.5, 0.5])
   ```

2. **SLERP (Spherical Linear Interpolation)**
   - Better for distant models
   - Preserves magnitude
   ```python
   merged = pipeline.slerp_merge(model_a, model_b, t=0.5)
   ```

3. **Task Arithmetic**
   - Base + weighted task deltas
   - Best for: Multi-task merging
   ```python
   merged = pipeline.task_arithmetic_merge(base, [task_a, task_b])
   ```

4. **TIES (Trim, Elect, Sign & Merge)**
   - Resolves sign conflicts
   - Trims small changes
   ```python
   merged = pipeline.ties_merge(base, [model_a, model_b], k=0.2)
   ```

5. **DARE (Drop and REscale)**
   - Random dropout of updates
   - Prevents interference
   ```python
   merged = pipeline.dare_merge(base, [model_a, model_b], drop_rate=0.5)
   ```

6. **Model Soup**
   - Average multiple checkpoints
   - Best for: Same training run
   ```python
   merged = pipeline.model_soup_merge(checkpoints)
   ```

**Usage**:
```bash
python model_merging_pipeline.py
```

**Output**:
- Merged models in `/home/user/SparkTrainer/outputs/merged_models/`

## 🔧 Configuration

### Environment Variables

```bash
# Optional: Set custom paths
export SPARK_TRAINER_DATA_DIR=/path/to/data
export SPARK_TRAINER_MODEL_DIR=/path/to/models
```

### Model Configuration

Edit configurations directly in each script's `main()` function:

```python
config = {
    'batch_size': 8,
    'learning_rate': 1e-4,
    'num_epochs': 10,
    'use_lora': True,
    'lora_rank': 8,
    'gradient_checkpointing': True,
    'use_amp': True,  # Mixed precision
}
```

## 📊 Expected Outputs

After running all examples, you'll have:

```
SparkTrainer/
├── datasets/
│   └── multimodal_multistep_vqa/
│       └── v1/
│           ├── manifest.jsonl
│           ├── images/
│           ├── metadata.json
│           └── README.md
├── models/
│   └── moe_125k_example/
│       ├── model.pth
│       └── config.json
└── outputs/
    ├── multimodal_training/
    │   ├── checkpoints/
    │   └── logs/
    └── merged_models/
        ├── merged_linear.pt
        ├── merged_slerp.pt
        ├── merged_task_arithmetic.pt
        ├── merged_ties.pt
        └── merged_dare.pt
```

## 🎯 Advanced Usage

### Training with Custom Data

1. Create your dataset following the multimodal format
2. Modify `training_pipeline.py` to load your data
3. Adjust model config for your task
4. Run training

### Fine-tuning with LoRA

```python
# Enable LoRA training (only train LoRA params)
model.enable_lora_training()

# Train as normal
for epoch in range(num_epochs):
    train_epoch(model, dataloader)
```

### Quantization

```python
# Prepare model for quantization
model = model.prepare_for_quantization({
    'method': 'nf4',  # or 'int8', 'fp8', 'int4'
    'bits': 4
})
```

### Gradient Checkpointing

```python
# Enable in model config
model_config = {
    'use_gradient_checkpointing': True
}

# Or enable manually
model.gradient_checkpointing_enable()
```

## 📈 Performance Tips

1. **Memory Optimization**:
   - Use gradient checkpointing for large models
   - Enable mixed precision (AMP)
   - Use smaller batch sizes with gradient accumulation

2. **Speed Optimization**:
   - Flash Attention for long sequences
   - Mixed precision training
   - DataLoader num_workers tuning

3. **Quality Optimization**:
   - LoRA for parameter-efficient fine-tuning
   - Proper learning rate scheduling
   - Model merging for multi-task learning

## 🐛 Troubleshooting

### Out of Memory (OOM)

- Reduce batch size
- Enable gradient checkpointing
- Use quantization (INT8 or FP8)
- Reduce sequence length

### Slow Training

- Enable flash attention
- Use mixed precision (FP16/BF16)
- Increase DataLoader num_workers
- Check GPU utilization

### Poor Quality

- Increase LoRA rank
- Adjust learning rate
- Add more training data
- Try different merging strategies

## 📚 References

- **MoE**: [Mixture of Experts Explained](https://arxiv.org/abs/2101.03961)
- **LoRA**: [Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- **Flash Attention**: [Fast and Memory-Efficient](https://arxiv.org/abs/2205.14135)
- **YaRN**: [RoPE Scaling for Long Context](https://arxiv.org/abs/2309.00071)
- **TIES**: [Resolving Interference in Model Merging](https://arxiv.org/abs/2306.01708)
- **DARE**: [Drop and Rescale](https://arxiv.org/abs/2311.03099)

## 🤝 Contributing

Have improvements or new examples? Contributions are welcome!

## 📝 License

See main SparkTrainer LICENSE file.

---

For more information, see the main SparkTrainer documentation.
