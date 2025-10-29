# GPU Management

Efficiently manage GPU resources for optimal training performance.

## GPU Monitoring

SparkTrainer continuously monitors all GPUs in real-time:

- **Utilization**: GPU compute usage (%)
- **Memory**: VRAM usage (used/total)
- **Temperature**: GPU core temperature (°C)
- **Power Draw**: Current power consumption (W)
- **Power Limit**: Maximum power cap (W)

## DGX Spark Calibration

For DGX Spark systems, GPU 0 is automatically calibrated to display the correct 240W power limit. You'll see a **DGX** badge on GPU 0 in the dashboard.

## GPU Partitioning

Divide GPUs into logical partitions for better resource allocation:

### Creating Partitions

1. Navigate to **Admin** → **GPU Partitions**
2. Click **+ New Partition**
3. Configure:
   - GPU index
   - VRAM allocation (MB)
   - Max concurrent jobs
   - Priority level

### Partition Scheduling

Jobs are assigned to partitions based on:
- Available VRAM
- Current utilization
- Priority queue
- Model size requirements

## Best Practices

### Memory Management
- Reserve 1-2GB for system overhead
- Use gradient checkpointing for large models
- Enable mixed precision training (fp16/bf16)

### Multi-GPU Training
- Use `deepspeed` for >2 GPUs
- Enable ZeRO optimization for memory efficiency
- Monitor load balancing across GPUs

### Power Optimization
- Set power limits based on cooling capacity
- Monitor sustained power draw > 85%
- Use persistent mode for consistent performance

```bash
# Enable persistent mode (recommended)
nvidia-smi -pm 1

# Set power limit (example: 250W)
nvidia-smi -pl 250
```

## Troubleshooting

### GPU Not Detected
```bash
# Verify driver installation
nvidia-smi

# Check CUDA version
nvcc --version
```

### Out of Memory Errors
- Reduce batch size
- Enable gradient accumulation
- Use smaller model variant
- Check for memory leaks in data pipeline

### Low Utilization
- Increase batch size
- Reduce preprocessing complexity
- Use faster data loaders
- Check for I/O bottlenecks
