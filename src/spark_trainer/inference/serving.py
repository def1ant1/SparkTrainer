"""
Inference serving adapters for vLLM, TGI (Text Generation Inference), and Triton.

Provides unified interface for deploying models to production inference servers.
"""

import os
import subprocess
import requests
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference serving."""
    model_path: str
    model_type: str  # text, vision, audio, multimodal
    batch_size: int = 1
    max_seq_length: int = 2048
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    quantization: Optional[str] = None  # awq, gptq, squeezellm
    dtype: str = "float16"
    port: int = 8000
    host: str = "0.0.0.0"


class InferenceServer:
    """Base class for inference servers."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.process = None
        self.is_running = False

    def start(self):
        """Start inference server."""
        raise NotImplementedError

    def stop(self):
        """Stop inference server."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.is_running = False
            logger.info("Server stopped")

    def health_check(self) -> bool:
        """Check if server is healthy."""
        raise NotImplementedError

    def predict(self, inputs: Any) -> Any:
        """Make prediction."""
        raise NotImplementedError


class vLLMServer(InferenceServer):
    """
    vLLM serving adapter.

    Best for:
    - LLM text generation
    - High throughput
    - Continuous batching
    - PagedAttention memory optimization
    """

    def start(self):
        """Start vLLM server."""
        logger.info(f"Starting vLLM server for {self.config.model_path}")

        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.config.model_path,
            "--host", self.config.host,
            "--port", str(self.config.port),
            "--tensor-parallel-size", str(self.config.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
            "--max-model-len", str(self.config.max_seq_length),
            "--dtype", self.config.dtype,
        ]

        if self.config.quantization:
            cmd.extend(["--quantization", self.config.quantization])

        logger.info(f"Command: {' '.join(cmd)}")

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for server to start
        logger.info("Waiting for server to start...")
        for i in range(60):
            time.sleep(1)
            if self.health_check():
                self.is_running = True
                logger.info(f"vLLM server started at http://{self.config.host}:{self.config.port}")
                return

        raise RuntimeError("vLLM server failed to start")

    def health_check(self) -> bool:
        """Check vLLM server health."""
        try:
            response = requests.get(
                f"http://{self.config.host}:{self.config.port}/health",
                timeout=1,
            )
            return response.status_code == 200
        except Exception:
            return False

    def predict(self, prompt: str, **kwargs) -> Dict:
        """Generate text with vLLM."""
        url = f"http://{self.config.host}:{self.config.port}/v1/completions"

        payload = {
            "model": self.config.model_path,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 100),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        return response.json()

    def predict_batch(self, prompts: List[str], **kwargs) -> List[Dict]:
        """Batch text generation."""
        results = []
        for prompt in prompts:
            result = self.predict(prompt, **kwargs)
            results.append(result)
        return results


class TGIServer(InferenceServer):
    """
    Text Generation Inference (TGI) server adapter.

    Best for:
    - HuggingFace models
    - Streaming generation
    - Token-level control
    """

    def start(self):
        """Start TGI server."""
        logger.info(f"Starting TGI server for {self.config.model_path}")

        cmd = [
            "text-generation-launcher",
            "--model-id", self.config.model_path,
            "--hostname", self.config.host,
            "--port", str(self.config.port),
            "--max-total-tokens", str(self.config.max_seq_length),
            "--max-batch-prefill-tokens", str(self.config.max_seq_length),
        ]

        if self.config.quantization:
            cmd.extend(["--quantize", self.config.quantization])

        logger.info(f"Command: {' '.join(cmd)}")

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for server to start
        logger.info("Waiting for server to start...")
        for i in range(60):
            time.sleep(1)
            if self.health_check():
                self.is_running = True
                logger.info(f"TGI server started at http://{self.config.host}:{self.config.port}")
                return

        raise RuntimeError("TGI server failed to start")

    def health_check(self) -> bool:
        """Check TGI server health."""
        try:
            response = requests.get(
                f"http://{self.config.host}:{self.config.port}/health",
                timeout=1,
            )
            return response.status_code == 200
        except Exception:
            return False

    def predict(self, prompt: str, **kwargs) -> Dict:
        """Generate text with TGI."""
        url = f"http://{self.config.host}:{self.config.port}/generate"

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": kwargs.get("max_tokens", 100),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "do_sample": True,
            }
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        return response.json()

    def predict_stream(self, prompt: str, **kwargs):
        """Stream text generation."""
        url = f"http://{self.config.host}:{self.config.port}/generate_stream"

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": kwargs.get("max_tokens", 100),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "do_sample": True,
            }
        }

        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data:'):
                    data = json.loads(line[5:])
                    yield data


class TritonServer(InferenceServer):
    """
    NVIDIA Triton Inference Server adapter.

    Best for:
    - Multi-framework models (PyTorch, TensorFlow, ONNX)
    - Vision and audio models
    - Low latency
    - Dynamic batching
    """

    def start(self):
        """Start Triton server."""
        logger.info(f"Starting Triton server for {self.config.model_path}")

        # Triton expects model repository structure:
        # model_repository/
        #   model_name/
        #     config.pbtxt
        #     1/
        #       model.pt (or model.onnx, etc.)

        model_repo = Path(self.config.model_path).parent

        cmd = [
            "tritonserver",
            "--model-repository", str(model_repo),
            "--http-port", str(self.config.port),
            "--grpc-port", str(self.config.port + 1),
            "--metrics-port", str(self.config.port + 2),
        ]

        logger.info(f"Command: {' '.join(cmd)}")

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for server to start
        logger.info("Waiting for server to start...")
        for i in range(60):
            time.sleep(1)
            if self.health_check():
                self.is_running = True
                logger.info(f"Triton server started at http://{self.config.host}:{self.config.port}")
                return

        raise RuntimeError("Triton server failed to start")

    def health_check(self) -> bool:
        """Check Triton server health."""
        try:
            response = requests.get(
                f"http://{self.config.host}:{self.config.port}/v2/health/ready",
                timeout=1,
            )
            return response.status_code == 200
        except Exception:
            return False

    def predict(self, inputs: Any, model_name: str, **kwargs) -> Dict:
        """Run inference with Triton."""
        url = f"http://{self.config.host}:{self.config.port}/v2/models/{model_name}/infer"

        # Format inputs for Triton
        # This is model-specific, provide a generic template
        payload = {
            "inputs": [
                {
                    "name": "input",
                    "shape": inputs.shape if hasattr(inputs, 'shape') else [1],
                    "datatype": "FP32",
                    "data": inputs.tolist() if hasattr(inputs, 'tolist') else inputs,
                }
            ]
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        return response.json()


def get_inference_server(
    server_type: str,
    config: InferenceConfig,
) -> InferenceServer:
    """
    Factory function to get inference server.

    Args:
        server_type: Type of server (vllm, tgi, triton)
        config: Inference configuration

    Returns:
        InferenceServer instance
    """
    if server_type == "vllm":
        return vLLMServer(config)
    elif server_type == "tgi":
        return TGIServer(config)
    elif server_type == "triton":
        return TritonServer(config)
    else:
        raise ValueError(f"Unknown server type: {server_type}")


class ABTesting:
    """
    A/B testing framework for model comparison.

    Routes traffic between champion and challenger models.
    """

    def __init__(
        self,
        champion_server: InferenceServer,
        challenger_server: InferenceServer,
        traffic_split: float = 0.5,  # % of traffic to challenger
    ):
        self.champion = champion_server
        self.challenger = challenger_server
        self.traffic_split = traffic_split

        # Metrics
        self.champion_requests = 0
        self.challenger_requests = 0
        self.champion_latencies = []
        self.challenger_latencies = []

    def predict(self, inputs: Any, **kwargs) -> Dict:
        """
        Route request to champion or challenger based on split.
        """
        import random

        if random.random() < self.traffic_split:
            # Send to challenger
            start_time = time.time()
            result = self.challenger.predict(inputs, **kwargs)
            latency = time.time() - start_time

            self.challenger_requests += 1
            self.challenger_latencies.append(latency)

            result['model'] = 'challenger'
            result['latency'] = latency
        else:
            # Send to champion
            start_time = time.time()
            result = self.champion.predict(inputs, **kwargs)
            latency = time.time() - start_time

            self.champion_requests += 1
            self.champion_latencies.append(latency)

            result['model'] = 'champion'
            result['latency'] = latency

        return result

    def get_metrics(self) -> Dict:
        """Get A/B testing metrics."""
        import numpy as np

        return {
            'champion': {
                'requests': self.champion_requests,
                'avg_latency': np.mean(self.champion_latencies) if self.champion_latencies else 0,
                'p95_latency': np.percentile(self.champion_latencies, 95) if self.champion_latencies else 0,
            },
            'challenger': {
                'requests': self.challenger_requests,
                'avg_latency': np.mean(self.challenger_latencies) if self.challenger_latencies else 0,
                'p95_latency': np.percentile(self.challenger_latencies, 95) if self.challenger_latencies else 0,
            },
        }


def benchmark_server(
    server: InferenceServer,
    test_inputs: List[Any],
    num_runs: int = 100,
) -> Dict:
    """
    Benchmark inference server.

    Args:
        server: Inference server to benchmark
        test_inputs: List of test inputs
        num_runs: Number of benchmark runs

    Returns:
        Benchmark metrics
    """
    import numpy as np

    latencies = []

    logger.info(f"Running benchmark with {num_runs} requests...")

    for i in range(num_runs):
        input_sample = test_inputs[i % len(test_inputs)]

        start_time = time.time()
        server.predict(input_sample)
        latency = time.time() - start_time

        latencies.append(latency)

        if (i + 1) % 10 == 0:
            logger.info(f"Completed {i + 1}/{num_runs} requests")

    # Calculate metrics
    latencies = np.array(latencies)

    metrics = {
        'num_requests': num_runs,
        'avg_latency': np.mean(latencies),
        'p50_latency': np.percentile(latencies, 50),
        'p95_latency': np.percentile(latencies, 95),
        'p99_latency': np.percentile(latencies, 99),
        'throughput_rps': num_runs / np.sum(latencies),
    }

    logger.info(f"Benchmark results: {metrics}")

    return metrics


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Inference serving CLI")
    parser.add_argument("--server", choices=["vllm", "tgi", "triton"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--action", choices=["start", "test"], default="start")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = InferenceConfig(
        model_path=args.model_path,
        model_type="text",
        port=args.port,
    )

    server = get_inference_server(args.server, config)

    if args.action == "start":
        server.start()

        # Keep server running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            server.stop()

    elif args.action == "test":
        # Run test prediction
        server.start()

        result = server.predict("Hello, how are you?")
        print(f"Result: {result}")

        server.stop()


if __name__ == "__main__":
    main()
