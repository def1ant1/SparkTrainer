"""
GPU validation and CUDA information logging.
"""
import subprocess
from typing import Dict, List, Optional

from ..logger import get_logger

logger = get_logger()


def validate_cuda() -> bool:
    """
    Validate CUDA availability and log GPU information.

    Returns:
        True if CUDA is available
    """
    try:
        import torch
    except ImportError:
        logger.error("PyTorch not installed. Cannot check CUDA availability.")
        return False

    cuda_available = torch.cuda.is_available()

    if not cuda_available:
        logger.warning("CUDA is not available. Training will use CPU.")
        return False

    # Log CUDA information
    logger.info(f"CUDA is available: {torch.cuda.is_available()}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")

    # Log each GPU
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i}: {props.name}")
        logger.info(f"  Compute capability: {props.major}.{props.minor}")
        logger.info(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")
        logger.info(f"  Multi-processors: {props.multi_processor_count}")

    return True


def get_gpu_info() -> List[Dict[str, any]]:
    """
    Get detailed GPU information using nvidia-smi.

    Returns:
        List of dictionaries with GPU information
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            logger.warning("nvidia-smi failed")
            return []

        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 8:
                continue

            gpu_info = {
                "index": int(parts[0]),
                "name": parts[1],
                "driver_version": parts[2],
                "memory_total_mb": int(parts[3]),
                "memory_free_mb": int(parts[4]),
                "memory_used_mb": int(parts[5]),
                "utilization_percent": int(parts[6]) if parts[6] != "N/A" else 0,
                "temperature_c": int(parts[7]) if parts[7] != "N/A" else 0,
            }
            gpus.append(gpu_info)

        return gpus

    except FileNotFoundError:
        logger.warning("nvidia-smi not found")
        return []
    except Exception as e:
        logger.warning(f"Error getting GPU info: {e}")
        return []


def log_gpu_info():
    """Log detailed GPU information from nvidia-smi."""
    gpus = get_gpu_info()

    if not gpus:
        logger.warning("No GPU information available from nvidia-smi")
        return

    logger.info(f"Detected {len(gpus)} GPU(s) via nvidia-smi:")
    for gpu in gpus:
        logger.info(f"GPU {gpu['index']}: {gpu['name']}")
        logger.info(f"  Driver version: {gpu['driver_version']}")
        logger.info(f"  Memory: {gpu['memory_used_mb']}/{gpu['memory_total_mb']} MB used ({gpu['memory_free_mb']} MB free)")
        logger.info(f"  Utilization: {gpu['utilization_percent']}%")
        logger.info(f"  Temperature: {gpu['temperature_c']}°C")


def validate_gpu_requirements(min_memory_gb: Optional[float] = None, min_gpus: int = 1) -> bool:
    """
    Validate GPU requirements for training.

    Args:
        min_memory_gb: Minimum memory per GPU in GB
        min_gpus: Minimum number of GPUs required

    Returns:
        True if requirements are met
    """
    try:
        import torch
    except ImportError:
        logger.error("PyTorch not installed")
        return False

    if not torch.cuda.is_available():
        logger.error("CUDA not available")
        return False

    num_gpus = torch.cuda.device_count()
    if num_gpus < min_gpus:
        logger.error(f"Insufficient GPUs: {num_gpus} available, {min_gpus} required")
        return False

    if min_memory_gb:
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            if memory_gb < min_memory_gb:
                logger.error(f"GPU {i} has insufficient memory: {memory_gb:.2f} GB < {min_memory_gb} GB")
                return False

    logger.info(f"GPU requirements validated: {num_gpus} GPU(s) available")
    return True


def set_cuda_visible_devices(device_ids: List[int]):
    """
    Set CUDA_VISIBLE_DEVICES environment variable.

    Args:
        device_ids: List of GPU device IDs to make visible
    """
    import os

    device_str = ",".join(map(str, device_ids))
    os.environ["CUDA_VISIBLE_DEVICES"] = device_str
    logger.info(f"Set CUDA_VISIBLE_DEVICES={device_str}")
