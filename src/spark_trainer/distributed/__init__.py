"""
Distributed training infrastructure for SparkTrainer.

Supports:
- DDP (DistributedDataParallel)
- FSDP (Fully Sharded Data Parallel)
- DeepSpeed ZeRO (1/2/3)
- Auto-selection based on model size
"""

from .launchers import (
    DistributedConfig,
    DistributedLauncher,
    TorchrunLauncher,
    FSDPLauncher,
    DeepSpeedLauncher,
    auto_select_backend,
    get_launcher,
    launch_distributed_training,
    estimate_model_size,
    print_distributed_info,
)

__all__ = [
    'DistributedConfig',
    'DistributedLauncher',
    'TorchrunLauncher',
    'FSDPLauncher',
    'DeepSpeedLauncher',
    'auto_select_backend',
    'get_launcher',
    'launch_distributed_training',
    'estimate_model_size',
    'print_distributed_info',
]
