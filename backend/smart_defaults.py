"""
Smart defaults calculator for experiment hyperparameters.
Suggests optimal configurations based on model, dataset, and hardware.
"""
from typing import Dict, List, Optional, Any
import math


class SmartDefaults:
    """Calculator for intelligent default hyperparameters."""

    # VRAM estimates per parameter (in GB)
    VRAM_PER_PARAM = {
        'fp32': 4.0,      # 4 bytes per parameter
        'fp16': 2.0,      # 2 bytes per parameter
        'bf16': 2.0,      # 2 bytes per parameter
        'int8': 1.0,      # 1 byte per parameter
        'int4': 0.5,      # 0.5 bytes per parameter
    }

    # Training overhead multipliers
    TRAINING_OVERHEAD = {
        'full_ft': 4.0,      # Activations + gradients + optimizer states
        'lora': 1.5,         # Base model + LoRA params + gradients
        'qlora': 1.2,        # Quantized base + LoRA params
        'prompt_tuning': 1.1  # Minimal overhead
    }

    # GPU memory capacities (in GB)
    GPU_MEMORY = {
        'A100': 80,
        'A100-40GB': 40,
        'H100': 80,
        'V100': 32,
        'A10': 24,
        'T4': 16,
        'L4': 24,
        'L40': 48,
    }

    @staticmethod
    def calculate_defaults(
        base_model: Optional[Dict],
        dataset: Optional[Dict],
        recipe: Optional[Dict],
        num_gpus: int = 1,
        gpu_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate smart defaults for an experiment.

        Returns a dict with suggested hyperparameters and resource configs.
        """
        defaults = {
            'train': {},
            'strategy': {},
            'resources': {},
            'eval': {},
            'export': [],
        }

        if not base_model or not dataset or not recipe:
            return defaults

        # Get model and dataset properties
        params_b = base_model.get('params_b', 1.0)
        num_samples = dataset.get('num_samples', 1000)
        recipe_type = recipe.get('recipe_type', 'lora')
        dtype = base_model.get('dtype', 'bf16')

        # Calculate training defaults
        defaults['train'] = SmartDefaults._calculate_training_params(
            params_b, num_samples, recipe_type, dtype
        )

        # Calculate strategy defaults
        defaults['strategy'] = SmartDefaults._calculate_strategy(
            params_b, num_gpus, recipe_type, dtype
        )

        # Calculate resource requirements
        defaults['resources'] = SmartDefaults._calculate_resources(
            params_b, recipe_type, dtype, num_gpus, gpu_type
        )

        # Suggest eval configuration
        defaults['eval'] = SmartDefaults._calculate_eval_config(
            base_model, num_samples
        )

        # Suggest export formats
        defaults['export'] = SmartDefaults._suggest_export_formats(
            base_model, recipe_type
        )

        return defaults

    @staticmethod
    def _calculate_training_params(
        params_b: float,
        num_samples: int,
        recipe_type: str,
        dtype: str
    ) -> Dict[str, Any]:
        """Calculate training hyperparameters."""
        # Learning rate: smaller for larger models
        if params_b < 1:
            base_lr = 5e-4
        elif params_b < 7:
            base_lr = 3e-4
        elif params_b < 13:
            base_lr = 2e-4
        elif params_b < 70:
            base_lr = 1e-4
        else:
            base_lr = 5e-5

        # Adjust LR for recipe type
        if recipe_type == 'full_ft':
            base_lr = base_lr * 0.5  # Lower LR for full fine-tuning
        elif recipe_type in ['lora', 'qlora']:
            base_lr = base_lr * 2.0  # Higher LR for parameter-efficient methods

        # Batch size: based on model size
        if params_b < 1:
            batch_size = 32
        elif params_b < 7:
            batch_size = 16
        elif params_b < 13:
            batch_size = 8
        elif params_b < 70:
            batch_size = 4
        else:
            batch_size = 2

        # Calculate gradient accumulation for effective batch size of 64-128
        target_batch = 64 if num_samples < 10000 else 128
        grad_accum = max(1, target_batch // batch_size)

        # Calculate max steps (aim for ~3 epochs)
        steps_per_epoch = math.ceil(num_samples / (batch_size * grad_accum))
        max_steps = steps_per_epoch * 3

        # Warmup steps (10% of training)
        warmup_steps = max(100, int(max_steps * 0.1))

        # Checkpoint interval (every 10% of training)
        checkpoint_interval = max(100, int(max_steps * 0.1))

        return {
            'max_steps': max_steps,
            'global_batch_size': batch_size,
            'grad_accum': grad_accum,
            'learning_rate': base_lr,
            'warmup_steps': warmup_steps,
            'weight_decay': 0.01,
            'lr_scheduler': 'cosine',
            'seed': 42,
            'checkpoint_interval': checkpoint_interval,
            'max_grad_norm': 1.0,
        }

    @staticmethod
    def _calculate_strategy(
        params_b: float,
        num_gpus: int,
        recipe_type: str,
        dtype: str
    ) -> Dict[str, Any]:
        """Calculate distributed training strategy."""
        # Choose distributed strategy
        if num_gpus == 1:
            strategy_type = None
        elif params_b < 7:
            strategy_type = 'ddp'  # Simple DDP for smaller models
        elif params_b < 70:
            strategy_type = 'fsdp'  # FSDP for medium-large models
        else:
            strategy_type = 'deepspeed'  # DeepSpeed for very large models

        # Choose mixed precision
        if dtype in ['int8', 'int4']:
            mixed_precision = None  # Already quantized
        elif recipe_type == 'qlora':
            mixed_precision = 'bf16'  # QLoRA typically uses bf16
        else:
            mixed_precision = 'bf16'  # bf16 is generally safer than fp16

        strategy = {
            'type': strategy_type,
            'mixed_precision': mixed_precision,
        }

        # Add FSDP-specific config
        if strategy_type == 'fsdp':
            strategy['fsdp_config'] = {
                'sharding_strategy': 'FULL_SHARD',  # Maximum memory efficiency
                'cpu_offload': params_b > 30,  # Offload for very large models
            }

        # Add DeepSpeed-specific config
        if strategy_type == 'deepspeed':
            if params_b < 100:
                strategy['deepspeed_config'] = {'stage': 2}  # Stage 2 for <100B
            else:
                strategy['deepspeed_config'] = {'stage': 3}  # Stage 3 for >100B

        return strategy

    @staticmethod
    def _calculate_resources(
        params_b: float,
        recipe_type: str,
        dtype: str,
        num_gpus: int,
        gpu_type: Optional[str]
    ) -> Dict[str, Any]:
        """Calculate resource requirements."""
        # Estimate VRAM needed
        bytes_per_param = SmartDefaults.VRAM_PER_PARAM.get(dtype, 2.0)
        overhead = SmartDefaults.TRAINING_OVERHEAD.get(recipe_type, 1.5)

        # Base memory for model
        model_memory_gb = (params_b * bytes_per_param * overhead)

        # Add buffer for activations and overhead (20%)
        total_memory_gb = model_memory_gb * 1.2

        # Divide by number of GPUs for distributed training
        memory_per_gpu = total_memory_gb / num_gpus if num_gpus > 1 else total_memory_gb

        # Suggest GPU count if not specified
        if gpu_type:
            gpu_memory = SmartDefaults.GPU_MEMORY.get(gpu_type, 40)
            suggested_gpus = max(1, math.ceil(total_memory_gb / (gpu_memory * 0.8)))
        else:
            suggested_gpus = num_gpus

        return {
            'gpus': suggested_gpus,
            'estimated_vram_per_gpu_gb': round(memory_per_gpu, 2),
            'total_vram_gb': round(total_memory_gb, 2),
            'gpu_type': gpu_type,
        }

    @staticmethod
    def _calculate_eval_config(
        base_model: Dict,
        num_samples: int
    ) -> Dict[str, Any]:
        """Calculate evaluation configuration."""
        modality = base_model.get('modality')

        # Suggest eval suites based on modality
        if modality == 'text':
            suites = ['perplexity']
            if 'gpt' in base_model.get('family', '').lower() or 'llama' in base_model.get('family', '').lower():
                suites.append('mmlu')
                suites.append('hellaswag')
        elif modality == 'image':
            suites = ['accuracy', 'f1']
        elif modality == 'audio':
            suites = ['wer', 'cer']
        else:
            suites = ['accuracy']

        # Eval interval based on dataset size
        if num_samples < 1000:
            interval = 100  # Eval every 100 steps for small datasets
        elif num_samples < 10000:
            interval = 500
        else:
            interval = 1000

        return {
            'suites': suites,
            'interval': interval,
            'save_best': True,
        }

    @staticmethod
    def _suggest_export_formats(
        base_model: Dict,
        recipe_type: str
    ) -> List[str]:
        """Suggest export formats based on model and recipe."""
        formats = ['safetensors']  # Always include safetensors

        modality = base_model.get('modality')
        servable = base_model.get('servable', True)

        # Add ONNX for smaller models that will be served
        if servable and base_model.get('params_b', 0) < 7:
            formats.append('onnx')

        # Add GGUF for text models (for llama.cpp inference)
        if modality == 'text' and recipe_type != 'full_ft':
            formats.append('gguf')

        return formats

    @staticmethod
    def estimate_vram(
        params_b: float,
        dtype: str,
        recipe_type: str,
        batch_size: int = 1,
        grad_accum: int = 1,
        sequence_length: int = 2048
    ) -> Dict[str, float]:
        """
        Detailed VRAM estimation for preflight checks.

        Returns dict with breakdown of memory usage.
        """
        bytes_per_param = SmartDefaults.VRAM_PER_PARAM.get(dtype, 2.0)

        # Model weights
        model_memory = params_b * bytes_per_param

        # Optimizer states (Adam: 8 bytes per param for fp32, even if model is fp16)
        if recipe_type == 'full_ft':
            optimizer_memory = params_b * 8.0  # Adam state
        elif recipe_type in ['lora', 'qlora']:
            # Only trainable params need optimizer state
            trainable_ratio = 0.01  # Typically 1% for LoRA
            optimizer_memory = params_b * trainable_ratio * 8.0
        else:
            optimizer_memory = 0.0

        # Gradients
        if recipe_type == 'full_ft':
            gradient_memory = params_b * 2.0  # fp16 gradients
        elif recipe_type in ['lora', 'qlora']:
            trainable_ratio = 0.01
            gradient_memory = params_b * trainable_ratio * 2.0
        else:
            gradient_memory = 0.0

        # Activations (depends on batch size and sequence length)
        # Rough estimate: num_layers * hidden_size * sequence_length * batch_size * 2 bytes
        # Simplified: ~10% of model size per sample for transformers
        activation_memory = (model_memory * 0.1 * batch_size * grad_accum)

        # Total
        total_memory = model_memory + optimizer_memory + gradient_memory + activation_memory

        # Add 10% buffer for misc overhead
        total_with_buffer = total_memory * 1.1

        return {
            'model_gb': round(model_memory, 2),
            'optimizer_gb': round(optimizer_memory, 2),
            'gradients_gb': round(gradient_memory, 2),
            'activations_gb': round(activation_memory, 2),
            'total_gb': round(total_memory, 2),
            'total_with_buffer_gb': round(total_with_buffer, 2),
        }

    @staticmethod
    def estimate_throughput(
        params_b: float,
        gpu_type: str,
        batch_size: int = 1,
        sequence_length: int = 2048
    ) -> Dict[str, Any]:
        """
        Estimate training throughput (tokens/sec, time per step).

        Returns dict with throughput estimates.
        """
        # Rough FLOPs estimates for different GPUs (TFLOPS for bf16)
        gpu_tflops = {
            'H100': 1000,
            'A100': 312,
            'A100-40GB': 312,
            'V100': 125,
            'A10': 125,
            'L4': 120,
            'T4': 65,
            'L40': 180,
        }

        tflops = gpu_tflops.get(gpu_type, 200)  # Default to 200 TFLOPS

        # Estimate FLOPs per token (rough approximation)
        # Forward pass: 2 * params * sequence_length
        # Backward pass: 4 * params * sequence_length (2x forward)
        flops_per_token = 6 * params_b * 1e9 * sequence_length

        # Tokens per second
        tokens_per_sec = (tflops * 1e12) / flops_per_token

        # Adjust for batch size (with efficiency factor)
        efficiency = 0.5  # Real-world efficiency ~50%
        tokens_per_sec_batch = tokens_per_sec * batch_size * efficiency

        # Time per step (in seconds)
        time_per_step_sec = 1.0 / (tokens_per_sec_batch / (batch_size * sequence_length))

        return {
            'tokens_per_sec': round(tokens_per_sec_batch, 2),
            'time_per_step_ms': round(time_per_step_sec * 1000, 2),
            'samples_per_sec': round(tokens_per_sec_batch / sequence_length, 2),
        }
