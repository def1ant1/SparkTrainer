#!/usr/bin/env python
"""
Benchmark for training throughput on different hardware configurations.

This script measures training throughput (samples/second) for various
model sizes, batch sizes, and GPU configurations.
"""

import argparse
from typing import Dict, Any, List
import time

from benchmarks.base import BenchmarkRunner


class TrainingThroughputBenchmark(BenchmarkRunner):
    """
    Benchmark training throughput across different configurations.

    Measures:
    - Samples per second
    - GPU utilization
    - Memory usage
    - Scaling efficiency
    """

    def __init__(
        self,
        model_name: str,
        batch_sizes: List[int],
        num_steps: int = 100,
    ):
        """
        Initialize training throughput benchmark.

        Args:
            model_name: Name of the model to benchmark
            batch_sizes: List of batch sizes to test
            num_steps: Number of training steps per configuration
        """
        super().__init__("training_throughput")
        self.model_name = model_name
        self.batch_sizes = batch_sizes
        self.num_steps = num_steps

    def run(self) -> Dict[str, Any]:
        """
        Run the training throughput benchmark.

        Returns:
            Dictionary with benchmark results
        """
        results = {"model": self.model_name, "batch_sizes": {}}

        for batch_size in self.batch_sizes:
            print(f"\nBenchmarking batch size: {batch_size}")
            results["batch_sizes"][str(batch_size)] = self.benchmark_batch_size(batch_size)

        self.print_results(results)
        return results

    def benchmark_batch_size(self, batch_size: int) -> Dict[str, Any]:
        """
        Benchmark a specific batch size.

        Args:
            batch_size: Batch size to benchmark

        Returns:
            Dictionary with metrics for this batch size
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Load model
            print(f"Loading model: {self.model_name}")
            model = AutoModelForCausalLM.from_pretrained(self.model_name)

            if torch.cuda.is_available():
                model = model.cuda()
                device = "cuda"
            else:
                device = "cpu"

            model.train()

            # Create dummy data
            seq_length = 128
            vocab_size = model.config.vocab_size
            dummy_input = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
            dummy_labels = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)

            # Warmup
            print("Warming up...")
            for _ in range(10):
                outputs = model(dummy_input, labels=dummy_labels)
                loss = outputs.loss
                loss.backward()

            # Benchmark
            print(f"Running {self.num_steps} steps...")
            torch.cuda.synchronize() if torch.cuda.is_available() else None

            start_time = time.perf_counter()

            for _ in range(self.num_steps):
                outputs = model(dummy_input, labels=dummy_labels)
                loss = outputs.loss
                loss.backward()

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.perf_counter() - start_time

            # Calculate metrics
            total_samples = batch_size * self.num_steps
            throughput = total_samples / elapsed
            avg_step_time = elapsed / self.num_steps

            results = {
                "throughput_samples_per_sec": throughput,
                "avg_step_time_sec": avg_step_time,
                "total_time_sec": elapsed,
                "device": device,
            }

            # Add GPU metrics if available
            if torch.cuda.is_available():
                results["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
                results["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)

            # Cleanup
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            return results

        except Exception as e:
            return {"error": str(e)}


def main():
    """Run the training throughput benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark training throughput")
    parser.add_argument(
        "--model",
        default="gpt2",
        help="Model name to benchmark (default: gpt2)",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8",
        help="Comma-separated list of batch sizes (default: 1,2,4,8)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of training steps per configuration (default: 100)",
    )
    parser.add_argument(
        "--save-results",
        default=None,
        help="Path to save results JSON",
    )

    args = parser.parse_args()

    # Parse batch sizes
    batch_sizes = [int(bs.strip()) for bs in args.batch_sizes.split(",")]

    # Run benchmark
    benchmark = TrainingThroughputBenchmark(
        model_name=args.model,
        batch_sizes=batch_sizes,
        num_steps=args.num_steps,
    )
    results = benchmark.run()

    # Save results
    if args.save_results:
        benchmark.save_results(results, args.save_results)
    else:
        benchmark.save_results(results)


if __name__ == "__main__":
    main()
