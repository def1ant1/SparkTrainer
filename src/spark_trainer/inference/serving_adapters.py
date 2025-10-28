"""
Inference Serving Adapters - Unified interface for multiple serving engines.

Supports:
- vLLM (LLM inference)
- Text Generation Inference (TGI)
- Triton Inference Server (multi-modal)
- TorchServe
- Custom REST/gRPC endpoints
"""

import os
import json
import logging
import requests
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import time

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Inference request."""
    inputs: Union[str, List[str], np.ndarray, Dict[str, Any]]
    parameters: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class InferenceResponse:
    """Inference response."""
    outputs: Union[str, List[str], np.ndarray, Dict[str, Any]]
    latency_ms: float
    token_count: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class ServingAdapter(ABC):
    """Base class for serving adapters."""

    def __init__(
        self,
        model_name: str,
        endpoint_url: Optional[str] = None,
    ):
        self.model_name = model_name
        self.endpoint_url = endpoint_url

    @abstractmethod
    def predict(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if server is healthy."""
        pass

    def batch_predict(
        self,
        requests: List[InferenceRequest],
    ) -> List[InferenceResponse]:
        """Run batch inference."""
        return [self.predict(req) for req in requests]


class VLLMAdapter(ServingAdapter):
    """
    vLLM adapter for fast LLM inference.

    Supports:
    - OpenAI-compatible API
    - Continuous batching
    - PagedAttention for KV cache
    - Tensor parallelism
    """

    def __init__(
        self,
        model_name: str,
        endpoint_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
    ):
        super().__init__(model_name, endpoint_url)
        self.api_key = api_key
        self.api_base = f"{endpoint_url}/v1"

    def predict(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference using vLLM."""
        start_time = time.time()

        # Prepare request
        payload = {
            "model": self.model_name,
            "prompt": request.inputs if isinstance(request.inputs, str) else request.inputs[0],
            "max_tokens": request.parameters.get("max_tokens", 100) if request.parameters else 100,
            "temperature": request.parameters.get("temperature", 0.7) if request.parameters else 0.7,
            "top_p": request.parameters.get("top_p", 0.9) if request.parameters else 0.9,
        }

        # Add optional parameters
        if request.parameters:
            if "stop" in request.parameters:
                payload["stop"] = request.parameters["stop"]
            if "presence_penalty" in request.parameters:
                payload["presence_penalty"] = request.parameters["presence_penalty"]
            if "frequency_penalty" in request.parameters:
                payload["frequency_penalty"] = request.parameters["frequency_penalty"]

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            # Use completions endpoint
            response = requests.post(
                f"{self.api_base}/completions",
                json=payload,
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()

            result = response.json()
            latency_ms = (time.time() - start_time) * 1000

            # Extract output
            output_text = result["choices"][0]["text"]
            token_count = result.get("usage", {}).get("total_tokens")

            return InferenceResponse(
                outputs=output_text,
                latency_ms=latency_ms,
                token_count=token_count,
                metadata={
                    "model": result.get("model"),
                    "finish_reason": result["choices"][0].get("finish_reason"),
                },
            )

        except Exception as e:
            logger.error(f"vLLM inference failed: {e}")
            raise

    def health_check(self) -> bool:
        """Check vLLM server health."""
        try:
            response = requests.get(f"{self.endpoint_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"vLLM health check failed: {e}")
            return False


class TGIAdapter(ServingAdapter):
    """
    Text Generation Inference (TGI) adapter.

    Supports:
    - HuggingFace models
    - Flash Attention
    - Quantization (GPTQ, bitsandbytes)
    - Token streaming
    """

    def __init__(
        self,
        model_name: str,
        endpoint_url: str = "http://localhost:8080",
    ):
        super().__init__(model_name, endpoint_url)

    def predict(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference using TGI."""
        start_time = time.time()

        # Prepare request
        payload = {
            "inputs": request.inputs if isinstance(request.inputs, str) else request.inputs[0],
            "parameters": {
                "max_new_tokens": request.parameters.get("max_tokens", 100) if request.parameters else 100,
                "temperature": request.parameters.get("temperature", 0.7) if request.parameters else 0.7,
                "top_p": request.parameters.get("top_p", 0.9) if request.parameters else 0.9,
                "do_sample": request.parameters.get("do_sample", True) if request.parameters else True,
            }
        }

        # Add optional parameters
        if request.parameters:
            if "repetition_penalty" in request.parameters:
                payload["parameters"]["repetition_penalty"] = request.parameters["repetition_penalty"]
            if "stop_sequences" in request.parameters:
                payload["parameters"]["stop_sequences"] = request.parameters["stop_sequences"]

        try:
            response = requests.post(
                f"{self.endpoint_url}/generate",
                json=payload,
                timeout=60,
            )
            response.raise_for_status()

            result = response.json()
            latency_ms = (time.time() - start_time) * 1000

            # Extract output
            output_text = result["generated_text"]
            token_count = result.get("details", {}).get("generated_tokens")

            return InferenceResponse(
                outputs=output_text,
                latency_ms=latency_ms,
                token_count=token_count,
                metadata={
                    "finish_reason": result.get("details", {}).get("finish_reason"),
                    "seed": result.get("details", {}).get("seed"),
                },
            )

        except Exception as e:
            logger.error(f"TGI inference failed: {e}")
            raise

    def health_check(self) -> bool:
        """Check TGI server health."""
        try:
            response = requests.get(f"{self.endpoint_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"TGI health check failed: {e}")
            return False

    def stream_predict(self, request: InferenceRequest):
        """Stream inference results."""
        payload = {
            "inputs": request.inputs if isinstance(request.inputs, str) else request.inputs[0],
            "parameters": {
                "max_new_tokens": request.parameters.get("max_tokens", 100) if request.parameters else 100,
                "temperature": request.parameters.get("temperature", 0.7) if request.parameters else 0.7,
            },
            "stream": True,
        }

        try:
            response = requests.post(
                f"{self.endpoint_url}/generate_stream",
                json=payload,
                stream=True,
                timeout=60,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8').replace('data: ', ''))
                    yield data.get('token', {}).get('text', '')

        except Exception as e:
            logger.error(f"TGI streaming failed: {e}")
            raise


class TritonAdapter(ServingAdapter):
    """
    Triton Inference Server adapter.

    Supports:
    - Multi-framework models (PyTorch, TensorFlow, ONNX)
    - Dynamic batching
    - Model ensembles
    - Multi-GPU inference
    """

    def __init__(
        self,
        model_name: str,
        endpoint_url: str = "http://localhost:8000",
        model_version: str = "1",
        protocol: str = "http",  # http or grpc
    ):
        super().__init__(model_name, endpoint_url)
        self.model_version = model_version
        self.protocol = protocol

        # Initialize client
        if protocol == "grpc":
            self._init_grpc_client()
        else:
            self._init_http_client()

    def _init_http_client(self):
        """Initialize HTTP client."""
        try:
            import tritonclient.http as httpclient
            self.client = httpclient.InferenceServerClient(url=self.endpoint_url.replace('http://', ''))
            logger.info(f"Triton HTTP client initialized: {self.endpoint_url}")
        except Exception as e:
            logger.error(f"Failed to initialize Triton HTTP client: {e}")
            self.client = None

    def _init_grpc_client(self):
        """Initialize gRPC client."""
        try:
            import tritonclient.grpc as grpcclient
            grpc_url = self.endpoint_url.replace('http://', '').replace('https://', '')
            self.client = grpcclient.InferenceServerClient(url=grpc_url)
            logger.info(f"Triton gRPC client initialized: {grpc_url}")
        except Exception as e:
            logger.error(f"Failed to initialize Triton gRPC client: {e}")
            self.client = None

    def predict(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference using Triton."""
        if not self.client:
            raise RuntimeError("Triton client not initialized")

        start_time = time.time()

        try:
            if self.protocol == "grpc":
                return self._grpc_predict(request, start_time)
            else:
                return self._http_predict(request, start_time)

        except Exception as e:
            logger.error(f"Triton inference failed: {e}")
            raise

    def _http_predict(self, request: InferenceRequest, start_time: float) -> InferenceResponse:
        """Run HTTP inference."""
        import tritonclient.http as httpclient

        # Prepare inputs
        inputs = []

        # Handle different input types
        if isinstance(request.inputs, dict):
            for input_name, input_data in request.inputs.items():
                if isinstance(input_data, np.ndarray):
                    infer_input = httpclient.InferInput(input_name, input_data.shape, "FP32")
                    infer_input.set_data_from_numpy(input_data)
                    inputs.append(infer_input)
        else:
            # Default text input
            input_data = np.array([request.inputs], dtype=object)
            infer_input = httpclient.InferInput("INPUT", input_data.shape, "BYTES")
            infer_input.set_data_from_numpy(input_data)
            inputs.append(infer_input)

        # Prepare outputs
        outputs = [httpclient.InferRequestedOutput("OUTPUT")]

        # Run inference
        result = self.client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=inputs,
            outputs=outputs,
        )

        latency_ms = (time.time() - start_time) * 1000

        # Extract output
        output_data = result.as_numpy("OUTPUT")

        return InferenceResponse(
            outputs=output_data,
            latency_ms=latency_ms,
            metadata={
                "model_name": self.model_name,
                "model_version": self.model_version,
            },
        )

    def _grpc_predict(self, request: InferenceRequest, start_time: float) -> InferenceResponse:
        """Run gRPC inference."""
        import tritonclient.grpc as grpcclient

        # Prepare inputs (similar to HTTP)
        inputs = []

        if isinstance(request.inputs, dict):
            for input_name, input_data in request.inputs.items():
                if isinstance(input_data, np.ndarray):
                    infer_input = grpcclient.InferInput(input_name, input_data.shape, "FP32")
                    infer_input.set_data_from_numpy(input_data)
                    inputs.append(infer_input)
        else:
            input_data = np.array([request.inputs], dtype=object)
            infer_input = grpcclient.InferInput("INPUT", input_data.shape, "BYTES")
            infer_input.set_data_from_numpy(input_data)
            inputs.append(infer_input)

        # Prepare outputs
        outputs = [grpcclient.InferRequestedOutput("OUTPUT")]

        # Run inference
        result = self.client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=inputs,
            outputs=outputs,
        )

        latency_ms = (time.time() - start_time) * 1000

        # Extract output
        output_data = result.as_numpy("OUTPUT")

        return InferenceResponse(
            outputs=output_data,
            latency_ms=latency_ms,
            metadata={
                "model_name": self.model_name,
                "model_version": self.model_version,
            },
        )

    def health_check(self) -> bool:
        """Check Triton server health."""
        if not self.client:
            return False

        try:
            return self.client.is_server_live()
        except Exception as e:
            logger.error(f"Triton health check failed: {e}")
            return False


class TorchServeAdapter(ServingAdapter):
    """
    TorchServe adapter for PyTorch models.

    Supports:
    - PyTorch models
    - Custom handlers
    - Model versioning
    - Metrics
    """

    def __init__(
        self,
        model_name: str,
        endpoint_url: str = "http://localhost:8080",
        model_version: str = "1.0",
    ):
        super().__init__(model_name, endpoint_url)
        self.model_version = model_version

    def predict(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference using TorchServe."""
        start_time = time.time()

        # Prepare request
        if isinstance(request.inputs, str):
            data = request.inputs.encode('utf-8')
        elif isinstance(request.inputs, np.ndarray):
            data = request.inputs.tobytes()
        else:
            data = json.dumps(request.inputs).encode('utf-8')

        try:
            response = requests.post(
                f"{self.endpoint_url}/predictions/{self.model_name}/{self.model_version}",
                data=data,
                headers={'Content-Type': 'application/json'},
                timeout=60,
            )
            response.raise_for_status()

            latency_ms = (time.time() - start_time) * 1000

            # Parse response
            result = response.json() if response.headers.get('Content-Type') == 'application/json' else response.text

            return InferenceResponse(
                outputs=result,
                latency_ms=latency_ms,
            )

        except Exception as e:
            logger.error(f"TorchServe inference failed: {e}")
            raise

    def health_check(self) -> bool:
        """Check TorchServe health."""
        try:
            response = requests.get(f"{self.endpoint_url}/ping", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"TorchServe health check failed: {e}")
            return False


class CustomRESTAdapter(ServingAdapter):
    """
    Generic REST API adapter.

    Supports custom REST endpoints with configurable request/response formats.
    """

    def __init__(
        self,
        model_name: str,
        endpoint_url: str,
        request_format: str = "json",  # json, form, raw
        response_format: str = "json",  # json, text
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(model_name, endpoint_url)
        self.request_format = request_format
        self.response_format = response_format
        self.headers = headers or {}

    def predict(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference using custom REST API."""
        start_time = time.time()

        # Prepare request
        if self.request_format == "json":
            data = {
                "inputs": request.inputs,
                "parameters": request.parameters or {},
            }
            kwargs = {"json": data}
        elif self.request_format == "form":
            data = {
                "inputs": str(request.inputs),
            }
            kwargs = {"data": data}
        else:
            kwargs = {"data": request.inputs}

        try:
            response = requests.post(
                self.endpoint_url,
                headers=self.headers,
                timeout=60,
                **kwargs,
            )
            response.raise_for_status()

            latency_ms = (time.time() - start_time) * 1000

            # Parse response
            if self.response_format == "json":
                outputs = response.json()
            else:
                outputs = response.text

            return InferenceResponse(
                outputs=outputs,
                latency_ms=latency_ms,
            )

        except Exception as e:
            logger.error(f"Custom REST inference failed: {e}")
            raise

    def health_check(self) -> bool:
        """Check endpoint health."""
        try:
            response = requests.get(self.endpoint_url.replace('/predict', '/health'), timeout=5)
            return response.status_code == 200
        except Exception:
            return True  # Assume healthy if no health endpoint


def create_adapter(
    backend: str,
    model_name: str,
    endpoint_url: str,
    **kwargs,
) -> ServingAdapter:
    """
    Factory function to create serving adapter.

    Args:
        backend: Serving backend (vllm, tgi, triton, torchserve, rest)
        model_name: Model name
        endpoint_url: Endpoint URL
        **kwargs: Additional backend-specific arguments

    Returns:
        Serving adapter instance
    """
    backend = backend.lower()

    if backend == "vllm":
        return VLLMAdapter(model_name, endpoint_url, **kwargs)
    elif backend == "tgi":
        return TGIAdapter(model_name, endpoint_url, **kwargs)
    elif backend == "triton":
        return TritonAdapter(model_name, endpoint_url, **kwargs)
    elif backend == "torchserve":
        return TorchServeAdapter(model_name, endpoint_url, **kwargs)
    elif backend == "rest":
        return CustomRESTAdapter(model_name, endpoint_url, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# Example usage
if __name__ == "__main__":
    # vLLM example
    vllm_adapter = create_adapter(
        backend="vllm",
        model_name="gpt2",
        endpoint_url="http://localhost:8000",
    )

    request = InferenceRequest(
        inputs="Hello, how are you?",
        parameters={"max_tokens": 50, "temperature": 0.7},
    )

    if vllm_adapter.health_check():
        response = vllm_adapter.predict(request)
        print(f"Output: {response.outputs}")
        print(f"Latency: {response.latency_ms:.2f}ms")
    else:
        print("vLLM server not available")
