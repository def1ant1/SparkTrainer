# LoRA Fine-tuning

Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning method that reduces memory usage and training time.

## How LoRA Works

Instead of updating all model parameters, LoRA injects trainable low-rank matrices into each layer:

```
W = W₀ + BA
```

Where:
- `W₀`: Frozen pre-trained weights
- `B`, `A`: Low-rank trainable matrices (rank `r`)

## When to Use LoRA

✅ **Good for**:
- Limited GPU memory
- Quick experimentation
- Task-specific adaptation
- Preserving base model capabilities

❌ **Not ideal for**:
- Completely new domains
- Major architecture changes
- Tasks requiring full model capacity

## Configuration

### Recommended Settings

| Parameter | Small Model (<7B) | Large Model (>7B) |
|-----------|------------------|------------------|
| `lora_r` | 8-16 | 32-64 |
| `lora_alpha` | 16-32 | 64-128 |
| `lora_dropout` | 0.05 | 0.1 |
| `target_modules` | `q_proj,v_proj` | `q_proj,k_proj,v_proj,o_proj` |

### Advanced Options

- **QLoRA**: 4-bit quantization + LoRA for even lower memory
- **LoRA+**: Adaptive learning rates for `A` and `B` matrices
- **DoRA**: Weight-decomposed LoRA for better convergence

## Example

```yaml
recipe: lora
model: meta-llama/Llama-2-7b-hf
dataset: my-dataset-v1

lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj

learning_rate: 2e-4
num_epochs: 3
batch_size: 4
gradient_accumulation_steps: 4
```

## Tips

1. **Start small**: Begin with `r=8` and increase if needed
2. **Target modules**: Include all attention projections for best results
3. **Learning rate**: 10x higher than full fine-tuning is okay
4. **Monitoring**: Watch for overfitting; LoRA can memorize quickly
