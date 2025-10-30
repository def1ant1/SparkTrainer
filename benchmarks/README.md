# Benchmarking Scripts

This directory contains benchmarking scripts to measure the performance of various SparkTrainer components.

## Available Benchmarks

- **video_processing_benchmark.py** - Benchmark video frame extraction and processing throughput
- **training_throughput_benchmark.py** - Measure training throughput on different hardware configurations
- **data_loading_benchmark.py** - Benchmark data loading and preprocessing pipeline
- **inference_benchmark.py** - Measure model inference latency and throughput

## Running Benchmarks

Each benchmark script can be run standalone:

```bash
python benchmarks/video_processing_benchmark.py --input /path/to/videos --output results.json
python benchmarks/training_throughput_benchmark.py --model gpt2 --batch-sizes 1,2,4,8
```

## Benchmark Results

Results are saved in JSON format with the following structure:

```json
{
  "benchmark_name": "video_processing",
  "timestamp": "2024-01-15T10:30:00Z",
  "hardware": {
    "gpu": "NVIDIA A100",
    "cpu": "AMD EPYC 7742",
    "ram": "512GB"
  },
  "metrics": {
    "throughput": 150.5,
    "latency_p50": 0.012,
    "latency_p95": 0.025,
    "latency_p99": 0.045
  }
}
```

## Adding New Benchmarks

To add a new benchmark:

1. Create a new Python script in this directory
2. Use the `BenchmarkRunner` base class
3. Implement the `run()` method
4. Add documentation to this README

Example:

```python
from benchmarks.base import BenchmarkRunner

class MyBenchmark(BenchmarkRunner):
    def run(self):
        # Benchmark implementation
        pass

if __name__ == "__main__":
    benchmark = MyBenchmark()
    results = benchmark.run()
    benchmark.save_results(results)
```
