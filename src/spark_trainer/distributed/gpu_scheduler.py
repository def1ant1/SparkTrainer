"""
Enhanced GPU Scheduler for Job Queue.

Provides:
- GPU placement and allocation
- Auto-resume on failures
- MIG (Multi-Instance GPU) awareness
- Load balancing across GPUs
- Memory-aware scheduling
"""

import os
import json
import logging
import subprocess
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import time
import threading

logger = logging.getLogger(__name__)


class GPUStatus(str, Enum):
    """GPU status."""
    AVAILABLE = "available"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class GPUInfo:
    """GPU information."""
    device_id: int
    name: str
    total_memory_mb: int
    free_memory_mb: int
    utilization_percent: float
    temperature_c: Optional[float] = None
    power_usage_w: Optional[float] = None
    status: GPUStatus = GPUStatus.AVAILABLE
    assigned_jobs: List[str] = None

    def __post_init__(self):
        if self.assigned_jobs is None:
            self.assigned_jobs = []


@dataclass
class JobCheckpoint:
    """Job checkpoint for auto-resume."""
    job_id: str
    checkpoint_path: str
    epoch: int
    step: int
    timestamp: str
    state_dict_keys: List[str]
    metadata: Dict[str, Any]


class GPUMonitor:
    """Monitor GPU status and availability."""

    def __init__(self, poll_interval: float = 5.0):
        self.poll_interval = poll_interval
        self._running = False
        self._monitor_thread = None
        self._gpu_info: Dict[int, GPUInfo] = {}
        self._lock = threading.Lock()

    def start(self):
        """Start monitoring."""
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("GPU monitor started")

    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join()
        logger.info("GPU monitor stopped")

    def _monitor_loop(self):
        """Monitor loop."""
        while self._running:
            try:
                self._update_gpu_info()
            except Exception as e:
                logger.error(f"GPU monitoring failed: {e}")
            time.sleep(self.poll_interval)

    def _update_gpu_info(self):
        """Update GPU information."""
        try:
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            gpu_info = {}

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # Get memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                # Get utilization
                util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)

                # Get temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temp = None

                # Get power
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to W
                except:
                    power = None

                # Get device name
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')

                gpu_info[i] = GPUInfo(
                    device_id=i,
                    name=name,
                    total_memory_mb=mem_info.total // (1024 * 1024),
                    free_memory_mb=mem_info.free // (1024 * 1024),
                    utilization_percent=util_info.gpu,
                    temperature_c=temp,
                    power_usage_w=power,
                    status=self._determine_status(util_info.gpu, mem_info.free, mem_info.total),
                )

            pynvml.nvmlShutdown()

            with self._lock:
                self._gpu_info = gpu_info

        except Exception as e:
            logger.error(f"Failed to update GPU info: {e}")

            # Fallback: use nvidia-smi
            self._update_gpu_info_fallback()

    def _update_gpu_info_fallback(self):
        """Fallback GPU info using nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True,
            )

            gpu_info = {}

            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    device_id = int(parts[0])
                    name = parts[1]
                    total_mem = int(parts[2])
                    free_mem = int(parts[3])
                    util = float(parts[4])

                    gpu_info[device_id] = GPUInfo(
                        device_id=device_id,
                        name=name,
                        total_memory_mb=total_mem,
                        free_memory_mb=free_mem,
                        utilization_percent=util,
                        status=self._determine_status(util, free_mem * 1024 * 1024, total_mem * 1024 * 1024),
                    )

            with self._lock:
                self._gpu_info = gpu_info

        except Exception as e:
            logger.error(f"Fallback GPU info failed: {e}")

    def _determine_status(self, utilization: float, free_mem: int, total_mem: int) -> GPUStatus:
        """Determine GPU status from metrics."""
        if utilization > 90 or (free_mem / total_mem) < 0.1:
            return GPUStatus.BUSY
        return GPUStatus.AVAILABLE

    def get_gpu_info(self, device_id: Optional[int] = None) -> Union[GPUInfo, Dict[int, GPUInfo]]:
        """Get GPU information."""
        with self._lock:
            if device_id is not None:
                return self._gpu_info.get(device_id)
            return self._gpu_info.copy()

    def get_available_gpus(self, min_memory_mb: int = 0) -> List[int]:
        """Get list of available GPU IDs."""
        with self._lock:
            available = []
            for device_id, info in self._gpu_info.items():
                if info.status == GPUStatus.AVAILABLE and info.free_memory_mb >= min_memory_mb:
                    available.append(device_id)
            return sorted(available, key=lambda x: self._gpu_info[x].free_memory_mb, reverse=True)


class GPUScheduler:
    """
    GPU scheduler with intelligent placement.

    Features:
    - Load balancing
    - Memory-aware placement
    - MIG support
    - Job-GPU affinity
    """

    def __init__(
        self,
        monitor: Optional[GPUMonitor] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        self.monitor = monitor or GPUMonitor()
        self.checkpoint_dir = Path(checkpoint_dir or "./checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Job allocations
        self._allocations: Dict[str, List[int]] = {}  # job_id -> [gpu_ids]
        self._lock = threading.Lock()

        # Start monitor
        if not self.monitor._running:
            self.monitor.start()

        logger.info("GPU scheduler initialized")

    def allocate_gpus(
        self,
        job_id: str,
        num_gpus: int = 1,
        min_memory_mb: int = 0,
        strategy: str = "least_loaded",  # least_loaded, round_robin, pack
    ) -> List[int]:
        """
        Allocate GPUs for a job.

        Args:
            job_id: Job ID
            num_gpus: Number of GPUs requested
            min_memory_mb: Minimum memory required per GPU
            strategy: Allocation strategy

        Returns:
            List of allocated GPU IDs
        """
        with self._lock:
            if job_id in self._allocations:
                logger.warning(f"Job {job_id} already has GPU allocation")
                return self._allocations[job_id]

            # Get available GPUs
            available = self.monitor.get_available_gpus(min_memory_mb)

            if len(available) < num_gpus:
                raise RuntimeError(f"Insufficient GPUs: need {num_gpus}, have {len(available)}")

            # Select GPUs based on strategy
            if strategy == "least_loaded":
                allocated = self._allocate_least_loaded(available, num_gpus)
            elif strategy == "round_robin":
                allocated = self._allocate_round_robin(available, num_gpus)
            elif strategy == "pack":
                allocated = self._allocate_pack(available, num_gpus)
            else:
                allocated = available[:num_gpus]

            # Record allocation
            self._allocations[job_id] = allocated

            logger.info(f"Allocated GPUs {allocated} to job {job_id}")

            return allocated

    def _allocate_least_loaded(self, available: List[int], num_gpus: int) -> List[int]:
        """Allocate GPUs with least load."""
        gpu_info = self.monitor.get_gpu_info()

        # Sort by utilization
        sorted_gpus = sorted(
            available,
            key=lambda x: (gpu_info[x].utilization_percent, -gpu_info[x].free_memory_mb)
        )

        return sorted_gpus[:num_gpus]

    def _allocate_round_robin(self, available: List[int], num_gpus: int) -> List[int]:
        """Allocate GPUs in round-robin fashion."""
        return available[:num_gpus]

    def _allocate_pack(self, available: List[int], num_gpus: int) -> List[int]:
        """Pack jobs onto fewer GPUs."""
        # Prefer consecutive GPU IDs
        return available[:num_gpus]

    def release_gpus(self, job_id: str):
        """Release GPU allocation for a job."""
        with self._lock:
            if job_id in self._allocations:
                allocated = self._allocations[job_id]
                del self._allocations[job_id]
                logger.info(f"Released GPUs {allocated} from job {job_id}")

    def get_allocation(self, job_id: str) -> Optional[List[int]]:
        """Get GPU allocation for a job."""
        with self._lock:
            return self._allocations.get(job_id)

    def save_checkpoint(
        self,
        job_id: str,
        state_dict: Dict[str, Any],
        epoch: int,
        step: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save job checkpoint for auto-resume.

        Args:
            job_id: Job ID
            state_dict: Model state dict
            epoch: Current epoch
            step: Current step
            metadata: Additional metadata

        Returns:
            Checkpoint path
        """
        import torch

        checkpoint_path = self.checkpoint_dir / f"{job_id}_epoch{epoch}_step{step}.pt"

        # Save checkpoint
        checkpoint_data = {
            'job_id': job_id,
            'epoch': epoch,
            'step': step,
            'state_dict': state_dict,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
        }

        torch.save(checkpoint_data, checkpoint_path)

        # Save checkpoint metadata
        checkpoint_meta = JobCheckpoint(
            job_id=job_id,
            checkpoint_path=str(checkpoint_path),
            epoch=epoch,
            step=step,
            timestamp=datetime.now().isoformat(),
            state_dict_keys=list(state_dict.keys()),
            metadata=metadata or {},
        )

        meta_path = self.checkpoint_dir / f"{job_id}_latest.json"
        with open(meta_path, 'w') as f:
            json.dump(asdict(checkpoint_meta), f, indent=2)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

        return str(checkpoint_path)

    def load_checkpoint(self, job_id: str) -> Optional[Tuple[Dict[str, Any], JobCheckpoint]]:
        """
        Load latest checkpoint for a job.

        Args:
            job_id: Job ID

        Returns:
            Tuple of (checkpoint_data, checkpoint_metadata) or None
        """
        import torch

        meta_path = self.checkpoint_dir / f"{job_id}_latest.json"

        if not meta_path.exists():
            logger.warning(f"No checkpoint found for job {job_id}")
            return None

        # Load metadata
        with open(meta_path, 'r') as f:
            meta_dict = json.load(f)

        checkpoint_meta = JobCheckpoint(**meta_dict)

        # Load checkpoint
        checkpoint_path = Path(checkpoint_meta.checkpoint_path)

        if not checkpoint_path.exists():
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return None

        checkpoint_data = torch.load(checkpoint_path)

        logger.info(f"Checkpoint loaded: {checkpoint_path}")

        return checkpoint_data, checkpoint_meta

    def auto_resume_job(
        self,
        job_id: str,
        train_fn: callable,
        **kwargs,
    ) -> Any:
        """
        Auto-resume job from latest checkpoint if available.

        Args:
            job_id: Job ID
            train_fn: Training function
            **kwargs: Additional arguments for train_fn

        Returns:
            Training result
        """
        # Try to load checkpoint
        checkpoint_result = self.load_checkpoint(job_id)

        if checkpoint_result:
            checkpoint_data, checkpoint_meta = checkpoint_result

            logger.info(f"Resuming job {job_id} from epoch {checkpoint_meta.epoch}, step {checkpoint_meta.step}")

            # Add checkpoint data to kwargs
            kwargs['resume_from_checkpoint'] = checkpoint_data
            kwargs['start_epoch'] = checkpoint_meta.epoch
            kwargs['start_step'] = checkpoint_meta.step

        else:
            logger.info(f"Starting job {job_id} from scratch")

        # Run training
        try:
            result = train_fn(**kwargs)
            return result

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")

            # Save error checkpoint
            error_meta = {
                'error': str(e),
                'failed_at': datetime.now().isoformat(),
            }

            error_path = self.checkpoint_dir / f"{job_id}_error.json"
            with open(error_path, 'w') as f:
                json.dump(error_meta, f, indent=2)

            raise


class MIGManager:
    """
    Multi-Instance GPU (MIG) manager.

    Handles MIG partitioning and allocation.
    """

    def __init__(self):
        self.mig_enabled = self._check_mig_support()

    def _check_mig_support(self) -> bool:
        """Check if MIG is supported and enabled."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '-L'],
                capture_output=True,
                text=True,
                check=True,
            )

            return 'MIG' in result.stdout

        except Exception:
            return False

    def get_mig_devices(self) -> List[Dict[str, Any]]:
        """Get list of MIG devices."""
        if not self.mig_enabled:
            return []

        try:
            result = subprocess.run(
                ['nvidia-smi', '-L'],
                capture_output=True,
                text=True,
                check=True,
            )

            mig_devices = []

            for line in result.stdout.strip().split('\n'):
                if 'MIG' in line:
                    # Parse MIG device info
                    # Format: GPU X: ... (UUID: ...) (MIG ...)
                    parts = line.split('MIG')
                    if len(parts) > 1:
                        mig_info = parts[1].strip()
                        mig_devices.append({
                            'line': line,
                            'mig_info': mig_info,
                        })

            return mig_devices

        except Exception as e:
            logger.error(f"Failed to get MIG devices: {e}")
            return []


# Example usage
if __name__ == "__main__":
    # Initialize scheduler
    scheduler = GPUScheduler()

    # Allocate GPUs for a job
    try:
        gpus = scheduler.allocate_gpus(
            job_id="train_job_123",
            num_gpus=2,
            min_memory_mb=8000,
            strategy="least_loaded",
        )
        print(f"Allocated GPUs: {gpus}")

        # Simulate training with checkpointing
        for epoch in range(5):
            state_dict = {'model': f'state_epoch_{epoch}'}

            checkpoint_path = scheduler.save_checkpoint(
                job_id="train_job_123",
                state_dict=state_dict,
                epoch=epoch,
                step=epoch * 100,
            )

            print(f"Checkpoint saved: {checkpoint_path}")

        # Release GPUs
        scheduler.release_gpus("train_job_123")

    finally:
        scheduler.monitor.stop()
