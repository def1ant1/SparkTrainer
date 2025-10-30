"""
Base classes and utilities for benchmarking.
"""

import time
import json
import platform
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np


class BenchmarkRunner(ABC):
    """
    Base class for all benchmark runners.

    Provides common functionality for timing, hardware detection,
    and result saving.
    """

    def __init__(self, name: str):
        """
        Initialize the benchmark runner.

        Args:
            name: Name of the benchmark
        """
        self.name = name
        self.results: Dict[str, Any] = {}

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """
        Run the benchmark.

        Returns:
            Dictionary containing benchmark results
        """
        pass

    def get_hardware_info(self) -> Dict[str, Any]:
        """
        Detect and return hardware information.

        Returns:
            Dictionary with hardware details
        """
        info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        }

        # Try to get GPU info
        try:
            import torch

            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
                info["gpu_count"] = torch.cuda.device_count()
                info["cuda_version"] = torch.version.cuda
            else:
                info["gpu"] = "None"
        except ImportError:
            info["gpu"] = "PyTorch not available"

        # Try to get RAM info
        try:
            import psutil

            info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
        except ImportError:
            info["ram_gb"] = "Unknown"

        return info

    def time_function(self, func, *args, **kwargs) -> tuple:
        """
        Time the execution of a function.

        Args:
            func: Function to time
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Tuple of (result, elapsed_time_seconds)
        """
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed

    def compute_statistics(self, times: List[float]) -> Dict[str, float]:
        """
        Compute statistics from a list of timing measurements.

        Args:
            times: List of time measurements in seconds

        Returns:
            Dictionary with mean, std, min, max, and percentiles
        """
        times_array = np.array(times)
        return {
            "mean": float(np.mean(times_array)),
            "std": float(np.std(times_array)),
            "min": float(np.min(times_array)),
            "max": float(np.max(times_array)),
            "median": float(np.median(times_array)),
            "p50": float(np.percentile(times_array, 50)),
            "p95": float(np.percentile(times_array, 95)),
            "p99": float(np.percentile(times_array, 99)),
        }

    def save_results(self, results: Dict[str, Any], output_path: Optional[str] = None):
        """
        Save benchmark results to a JSON file.

        Args:
            results: Results dictionary
            output_path: Path to save results (default: benchmarks/results/{name}_{timestamp}.json)
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"benchmarks/results/{self.name}_{timestamp}.json"

        # Ensure results directory exists
        import os

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Add metadata
        full_results = {
            "benchmark_name": self.name,
            "timestamp": datetime.now().isoformat(),
            "hardware": self.get_hardware_info(),
            "results": results,
        }

        with open(output_path, "w") as f:
            json.dump(full_results, f, indent=2)

        print(f"Results saved to: {output_path}")

    def print_results(self, results: Dict[str, Any]):
        """
        Print results in a formatted way.

        Args:
            results: Results dictionary
        """
        print(f"\n{'=' * 60}")
        print(f"Benchmark: {self.name}")
        print(f"{'=' * 60}")
        self._print_dict(results)
        print(f"{'=' * 60}\n")

    def _print_dict(self, d: Dict[str, Any], indent: int = 0):
        """
        Recursively print dictionary with indentation.

        Args:
            d: Dictionary to print
            indent: Current indentation level
        """
        for key, value in d.items():
            if isinstance(value, dict):
                print(" " * indent + f"{key}:")
                self._print_dict(value, indent + 2)
            elif isinstance(value, float):
                print(" " * indent + f"{key}: {value:.6f}")
            else:
                print(" " * indent + f"{key}: {value}")
