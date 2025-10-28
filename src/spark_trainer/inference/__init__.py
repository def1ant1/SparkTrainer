"""
Inference serving for SparkTrainer.

Supports:
- vLLM for LLM serving
- TGI (Text Generation Inference)
- NVIDIA Triton for multi-framework models
- A/B testing framework
"""

from .serving import (
    InferenceConfig,
    InferenceServer,
    vLLMServer,
    TGIServer,
    TritonServer,
    get_inference_server,
    ABTesting,
    benchmark_server,
)

__all__ = [
    'InferenceConfig',
    'InferenceServer',
    'vLLMServer',
    'TGIServer',
    'TritonServer',
    'get_inference_server',
    'ABTesting',
    'benchmark_server',
]
