"""
Distributed training launchers for DDP, FSDP, and DeepSpeed.

Provides automatic launcher selection based on model size and GPU memory.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import torch

logger = logging.getLogger(__name__)


class DistributedConfig:
    """Configuration for distributed training."""

    def __init__(
        self,
        backend: str = "auto",  # auto, ddp, fsdp, deepspeed
        world_size: Optional[int] = None,
        num_gpus: Optional[int] = None,
        master_addr: str = "localhost",
        master_port: int = 29500,
        deepspeed_config: Optional[Dict] = None,
        fsdp_config: Optional[Dict] = None,
    ):
        self.backend = backend
        self.world_size = world_size or torch.cuda.device_count()
        self.num_gpus = num_gpus or self.world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.deepspeed_config = deepspeed_config or self._default_deepspeed_config()
        self.fsdp_config = fsdp_config or self._default_fsdp_config()

    def _default_deepspeed_config(self) -> Dict:
        """Default DeepSpeed ZeRO-2 configuration."""
        return {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            "bf16": {
                "enabled": False,
            },
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "none",
                },
                "offload_param": {
                    "device": "none",
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "overlap_comm": True,
                "contiguous_gradients": True,
            },
            "gradient_clipping": 1.0,
            "steps_per_print": 100,
            "wall_clock_breakdown": False,
        }

    def _default_fsdp_config(self) -> Dict:
        """Default FSDP configuration."""
        return {
            "sharding_strategy": "FULL_SHARD",  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
            "cpu_offload": False,
            "auto_wrap_policy": "transformer",
            "backward_prefetch": "BACKWARD_PRE",
            "forward_prefetch": True,
            "limit_all_gathers": True,
        }

    def to_env(self) -> Dict[str, str]:
        """Convert config to environment variables."""
        env = os.environ.copy()
        env.update({
            "MASTER_ADDR": self.master_addr,
            "MASTER_PORT": str(self.master_port),
            "WORLD_SIZE": str(self.world_size),
        })
        return env


class DistributedLauncher:
    """Base class for distributed training launchers."""

    def __init__(self, config: DistributedConfig):
        self.config = config

    def launch(
        self,
        training_script: str,
        script_args: List[str],
    ) -> subprocess.CompletedProcess:
        """Launch distributed training."""
        raise NotImplementedError


class TorchrunLauncher(DistributedLauncher):
    """
    Launcher for DDP using torchrun.

    Recommended for:
    - Small to medium models
    - Good GPU memory availability
    - Simple synchronous training
    """

    def launch(
        self,
        training_script: str,
        script_args: List[str],
    ) -> subprocess.CompletedProcess:
        """Launch with torchrun (replaces torch.distributed.launch)."""
        cmd = [
            "torchrun",
            f"--nproc_per_node={self.config.num_gpus}",
            f"--nnodes=1",
            f"--node_rank=0",
            f"--master_addr={self.config.master_addr}",
            f"--master_port={self.config.master_port}",
            training_script,
        ] + script_args

        logger.info(f"Launching DDP with torchrun: {' '.join(cmd)}")

        env = self.config.to_env()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(self.config.num_gpus))

        result = subprocess.run(
            cmd,
            env=env,
            capture_output=False,  # Stream to console
            text=True,
        )

        return result


class FSDPLauncher(DistributedLauncher):
    """
    Launcher for FSDP (Fully Sharded Data Parallel).

    Recommended for:
    - Large models that don't fit on single GPU
    - Better memory efficiency than DDP
    - PyTorch 1.12+ required
    """

    def launch(
        self,
        training_script: str,
        script_args: List[str],
    ) -> subprocess.CompletedProcess:
        """Launch with FSDP via torchrun."""
        # FSDP uses same launcher as DDP but with FSDP config
        cmd = [
            "torchrun",
            f"--nproc_per_node={self.config.num_gpus}",
            f"--nnodes=1",
            f"--node_rank=0",
            f"--master_addr={self.config.master_addr}",
            f"--master_port={self.config.master_port}",
            training_script,
            "--distributed_backend=fsdp",
        ] + script_args

        logger.info(f"Launching FSDP: {' '.join(cmd)}")

        env = self.config.to_env()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(self.config.num_gpus))

        # Pass FSDP config via env
        env["FSDP_CONFIG"] = json.dumps(self.config.fsdp_config)

        result = subprocess.run(
            cmd,
            env=env,
            capture_output=False,
            text=True,
        )

        return result


class DeepSpeedLauncher(DistributedLauncher):
    """
    Launcher for DeepSpeed ZeRO.

    Recommended for:
    - Very large models (billions of parameters)
    - Maximum memory efficiency (ZeRO-3)
    - CPU offloading support
    - Advanced optimization features
    """

    def __init__(
        self,
        config: DistributedConfig,
        zero_stage: int = 2,
    ):
        super().__init__(config)
        self.zero_stage = zero_stage

        # Update ZeRO stage in config
        self.config.deepspeed_config["zero_optimization"]["stage"] = zero_stage

    def launch(
        self,
        training_script: str,
        script_args: List[str],
        hostfile: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        """Launch with DeepSpeed."""
        # Save DeepSpeed config
        config_path = Path("deepspeed_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.deepspeed_config, f, indent=2)

        logger.info(f"DeepSpeed config saved to {config_path}")

        cmd = [
            "deepspeed",
            f"--num_gpus={self.config.num_gpus}",
        ]

        if hostfile:
            cmd.append(f"--hostfile={hostfile}")

        cmd.extend([
            training_script,
            f"--deepspeed={config_path}",
        ])
        cmd.extend(script_args)

        logger.info(f"Launching DeepSpeed ZeRO-{self.zero_stage}: {' '.join(cmd)}")

        env = self.config.to_env()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(self.config.num_gpus))

        result = subprocess.run(
            cmd,
            env=env,
            capture_output=False,
            text=True,
        )

        return result


def auto_select_backend(
    model_size_gb: float,
    available_gpu_memory_gb: float,
    num_gpus: int = 1,
) -> str:
    """
    Automatically select best distributed backend.

    Args:
        model_size_gb: Model size in GB
        available_gpu_memory_gb: Available GPU memory in GB
        num_gpus: Number of GPUs

    Returns:
        Recommended backend: ddp, fsdp, or deepspeed
    """
    # Simple heuristics
    if model_size_gb <= available_gpu_memory_gb * 0.5:
        # Model fits comfortably on single GPU
        return "ddp"

    elif model_size_gb <= available_gpu_memory_gb * num_gpus * 0.7:
        # Model fits with FSDP sharding
        return "fsdp"

    else:
        # Need DeepSpeed ZeRO-3 with possible CPU offload
        return "deepspeed"


def get_launcher(
    backend: str = "auto",
    config: Optional[DistributedConfig] = None,
    model_size_gb: Optional[float] = None,
) -> DistributedLauncher:
    """
    Get appropriate distributed launcher.

    Args:
        backend: Backend type (auto, ddp, fsdp, deepspeed)
        config: Distributed configuration
        model_size_gb: Model size for auto-selection

    Returns:
        DistributedLauncher instance
    """
    if config is None:
        config = DistributedConfig(backend=backend)

    # Auto-select if requested
    if backend == "auto":
        if model_size_gb is None:
            raise ValueError("model_size_gb required for auto backend selection")

        # Get GPU memory
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            num_gpus = torch.cuda.device_count()
        else:
            raise RuntimeError("CUDA not available")

        backend = auto_select_backend(model_size_gb, gpu_memory_gb, num_gpus)
        logger.info(f"Auto-selected backend: {backend}")

    # Create launcher
    if backend == "ddp":
        return TorchrunLauncher(config)
    elif backend == "fsdp":
        return FSDPLauncher(config)
    elif backend == "deepspeed":
        return DeepSpeedLauncher(config)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def launch_distributed_training(
    training_script: str,
    script_args: Optional[List[str]] = None,
    backend: str = "auto",
    num_gpus: Optional[int] = None,
    model_size_gb: Optional[float] = None,
    deepspeed_config: Optional[Dict] = None,
    fsdp_config: Optional[Dict] = None,
) -> subprocess.CompletedProcess:
    """
    Convenience function to launch distributed training.

    Args:
        training_script: Path to training script
        script_args: Arguments to pass to script
        backend: Distributed backend (auto, ddp, fsdp, deepspeed)
        num_gpus: Number of GPUs to use
        model_size_gb: Model size for auto-selection
        deepspeed_config: Custom DeepSpeed config
        fsdp_config: Custom FSDP config

    Returns:
        CompletedProcess instance
    """
    script_args = script_args or []

    config = DistributedConfig(
        backend=backend,
        num_gpus=num_gpus,
        deepspeed_config=deepspeed_config,
        fsdp_config=fsdp_config,
    )

    launcher = get_launcher(backend, config, model_size_gb)

    return launcher.launch(training_script, script_args)


def estimate_model_size(model: torch.nn.Module) -> float:
    """
    Estimate model size in GB.

    Args:
        model: PyTorch model

    Returns:
        Model size in GB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size_gb = (param_size + buffer_size) / 1e9

    return total_size_gb


def print_distributed_info():
    """Print distributed training environment info."""
    print("\n" + "=" * 60)
    print("DISTRIBUTED TRAINING INFO")
    print("=" * 60)

    # PyTorch version
    print(f"PyTorch version: {torch.__version__}")

    # CUDA info
    if torch.cuda.is_available():
        print(f"CUDA available: True")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}:")
            print(f"  Name: {props.name}")
            print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")
    else:
        print("CUDA available: False")

    # Distributed env
    print(f"\nDistributed environment:")
    print(f"  WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'not set')}")
    print(f"  RANK: {os.environ.get('RANK', 'not set')}")
    print(f"  LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'not set')}")
    print(f"  MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'not set')}")
    print(f"  MASTER_PORT: {os.environ.get('MASTER_PORT', 'not set')}")

    # NCCL info
    if torch.cuda.is_available() and torch.distributed.is_nccl_available():
        print(f"\nNCCL available: True")
        print(f"NCCL version: {torch.cuda.nccl.version()}")
    else:
        print(f"\nNCCL available: False")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Example usage
    print_distributed_info()

    # Example launch
    result = launch_distributed_training(
        training_script="train.py",
        script_args=["--epochs=10", "--batch_size=32"],
        backend="ddp",
        num_gpus=2,
    )

    print(f"Training completed with return code: {result.returncode}")
