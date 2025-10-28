"""
Example: Model Merging Pipeline

This example demonstrates various model merging strategies:
- Linear interpolation (LERP)
- SLERP (Spherical Linear Interpolation)
- Task arithmetic
- TIES merging
- DARE (Drop and REscale)
- Model soup (averaging multiple checkpoints)
- LoRA adapter merging
"""

import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Optional
import copy


class ModelMergingPipeline:
    """Pipeline for merging multiple models or adapters"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self, model_path):
        """Load a model checkpoint"""
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        # Extract state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        return state_dict

    def linear_merge(self, models: List[Dict], weights: Optional[List[float]] = None):
        """
        Linear interpolation merge (weighted average)

        Args:
            models: List of model state dicts
            weights: Optional weights for each model (must sum to 1.0)
        """
        print("\n" + "="*60)
        print("Linear Interpolation Merge")
        print("="*60)

        if weights is None:
            weights = [1.0 / len(models)] * len(models)

        assert len(weights) == len(models), "Number of weights must match number of models"
        assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0"

        # Initialize merged model with first model structure
        merged = copy.deepcopy(models[0])

        # Merge each parameter
        for key in merged.keys():
            # Stack all model parameters for this key
            params = torch.stack([model[key].float() for model in models])

            # Weighted average
            weight_tensor = torch.tensor(weights, device=params.device, dtype=params.dtype)
            merged[key] = torch.sum(params * weight_tensor.view(-1, *([1] * (params.dim() - 1))), dim=0)

        print(f"  Merged {len(models)} models with weights {weights}")
        return merged

    def slerp_merge(self, model_a: Dict, model_b: Dict, t: float = 0.5):
        """
        Spherical linear interpolation merge

        Better for merging models that are far apart in parameter space

        Args:
            model_a: First model state dict
            model_b: Second model state dict
            t: Interpolation factor (0.0 = model_a, 1.0 = model_b)
        """
        print("\n" + "="*60)
        print(f"SLERP Merge (t={t})")
        print("="*60)

        merged = {}

        for key in model_a.keys():
            param_a = model_a[key].float()
            param_b = model_b[key].float()

            # Flatten parameters
            flat_a = param_a.flatten()
            flat_b = param_b.flatten()

            # Compute angle between vectors
            dot = torch.dot(flat_a, flat_b)
            norm_a = torch.norm(flat_a)
            norm_b = torch.norm(flat_b)

            cos_omega = dot / (norm_a * norm_b + 1e-8)
            omega = torch.acos(torch.clamp(cos_omega, -1.0, 1.0))

            # SLERP formula
            if omega.abs() < 1e-6:
                # Vectors are nearly parallel, use linear interpolation
                result = (1 - t) * flat_a + t * flat_b
            else:
                sin_omega = torch.sin(omega)
                result = (torch.sin((1 - t) * omega) / sin_omega) * flat_a + \
                         (torch.sin(t * omega) / sin_omega) * flat_b

            merged[key] = result.reshape(param_a.shape)

        print(f"  Merged using SLERP with t={t}")
        return merged

    def task_arithmetic_merge(self, base_model: Dict, task_models: List[Dict], weights: Optional[List[float]] = None):
        """
        Task arithmetic merge: base + sum(weight * (task_model - base))

        Args:
            base_model: Base/pretrained model
            task_models: List of task-specific fine-tuned models
            weights: Weights for each task delta
        """
        print("\n" + "="*60)
        print("Task Arithmetic Merge")
        print("="*60)

        if weights is None:
            weights = [1.0] * len(task_models)

        merged = copy.deepcopy(base_model)

        for key in merged.keys():
            # Start with base
            result = base_model[key].float()

            # Add weighted task deltas
            for task_model, weight in zip(task_models, weights):
                delta = task_model[key].float() - base_model[key].float()
                result += weight * delta

            merged[key] = result

        print(f"  Merged {len(task_models)} task models with base model")
        return merged

    def ties_merge(self, base_model: Dict, task_models: List[Dict], k: float = 0.2):
        """
        TIES (Trim, Elect Sign & Merge) merging

        1. Trim: Remove small magnitude changes
        2. Elect: Resolve sign conflicts by majority vote
        3. Merge: Average remaining parameters

        Args:
            base_model: Base model
            task_models: Task-specific models
            k: Fraction of smallest magnitude values to trim (0.0 to 1.0)
        """
        print("\n" + "="*60)
        print(f"TIES Merge (k={k})")
        print("="*60)

        merged = copy.deepcopy(base_model)

        for key in merged.keys():
            # Compute deltas
            deltas = [model[key].float() - base_model[key].float() for model in task_models]

            # Step 1: Trim - remove small magnitude values
            stacked_deltas = torch.stack(deltas)
            abs_deltas = torch.abs(stacked_deltas)

            # Find threshold (k-th percentile)
            threshold = torch.quantile(abs_deltas.flatten(), k)

            # Create mask for values to keep
            mask = abs_deltas > threshold

            # Step 2: Elect - resolve sign conflicts
            signs = torch.sign(stacked_deltas)
            # Majority vote on sign
            elected_sign = torch.sign(torch.sum(signs * mask.float(), dim=0))

            # Step 3: Merge - average aligned parameters
            aligned_deltas = torch.where(
                mask & (signs == elected_sign.unsqueeze(0)),
                stacked_deltas,
                torch.zeros_like(stacked_deltas)
            )

            merged_delta = torch.sum(aligned_deltas, dim=0) / torch.clamp(
                torch.sum(mask.float(), dim=0), min=1.0
            )

            merged[key] = base_model[key].float() + merged_delta

        print(f"  TIES merged {len(task_models)} models (trimmed {k*100:.1f}% of changes)")
        return merged

    def dare_merge(self, base_model: Dict, task_models: List[Dict], drop_rate: float = 0.5):
        """
        DARE (Drop and REscale) merging

        Randomly drop parameter updates and rescale remaining ones

        Args:
            base_model: Base model
            task_models: Task-specific models
            drop_rate: Probability of dropping each parameter update
        """
        print("\n" + "="*60)
        print(f"DARE Merge (drop_rate={drop_rate})")
        print("="*60)

        merged = copy.deepcopy(base_model)
        rescale_factor = 1.0 / (1.0 - drop_rate)

        for key in merged.keys():
            # Compute average delta
            deltas = [model[key].float() - base_model[key].float() for model in task_models]
            avg_delta = torch.mean(torch.stack(deltas), dim=0)

            # Random dropout mask
            mask = torch.rand_like(avg_delta) > drop_rate

            # Apply mask and rescale
            masked_delta = avg_delta * mask.float() * rescale_factor

            merged[key] = base_model[key].float() + masked_delta

        print(f"  DARE merged {len(task_models)} models (drop_rate={drop_rate})")
        return merged

    def model_soup_merge(self, checkpoints: List[Dict], selection_strategy: str = 'greedy'):
        """
        Model soup: Average multiple checkpoints from same training run

        Args:
            checkpoints: List of checkpoints (typically from different epochs)
            selection_strategy: 'uniform' or 'greedy' (based on validation performance)
        """
        print("\n" + "="*60)
        print(f"Model Soup Merge (strategy={selection_strategy})")
        print("="*60)

        if selection_strategy == 'uniform':
            # Simple uniform averaging
            return self.linear_merge(checkpoints)
        else:
            # Greedy soup: iteratively add checkpoints that improve performance
            # (Requires validation set - simplified here)
            print("  Greedy selection requires validation set (using uniform average)")
            return self.linear_merge(checkpoints)

    def merge_lora_adapters(self, base_model: Dict, lora_adapters: List[Dict], weights: Optional[List[float]] = None):
        """
        Merge multiple LoRA adapters into base model

        Args:
            base_model: Base model state dict
            lora_adapters: List of LoRA adapter state dicts
            weights: Optional weights for each adapter
        """
        print("\n" + "="*60)
        print("LoRA Adapter Merge")
        print("="*60)

        if weights is None:
            weights = [1.0] * len(lora_adapters)

        merged = copy.deepcopy(base_model)

        # Identify LoRA keys (typically contain 'lora_A' or 'lora_B')
        lora_keys = set()
        for adapter in lora_adapters:
            lora_keys.update([k for k in adapter.keys() if 'lora' in k.lower()])

        print(f"  Found {len(lora_keys)} LoRA parameters")

        # Merge LoRA parameters
        for key in lora_keys:
            # Average LoRA parameters across adapters
            params = [adapter.get(key, torch.zeros_like(lora_adapters[0][key])) for adapter in lora_adapters]
            weight_tensor = torch.tensor(weights, device=params[0].device, dtype=params[0].dtype)

            stacked = torch.stack([p.float() for p in params])
            merged[key] = torch.sum(stacked * weight_tensor.view(-1, *([1] * (stacked.dim() - 1))), dim=0)

        print(f"  Merged {len(lora_adapters)} LoRA adapters")
        return merged

    def save_merged_model(self, merged_state_dict: Dict, output_path: str, metadata: Optional[Dict] = None):
        """Save merged model"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': merged_state_dict,
            'merge_metadata': metadata or {}
        }

        torch.save(checkpoint, output_path)
        print(f"\nMerged model saved to: {output_path}")


def main():
    """Example usage of model merging pipeline"""

    config = {
        'output_dir': '/home/user/SparkTrainer/outputs/merged_models'
    }

    pipeline = ModelMergingPipeline(config)

    # Example: Create dummy models for demonstration
    print("Creating dummy models for demonstration...")

    # Create a simple model structure
    dummy_model = {
        'layer1.weight': torch.randn(512, 512),
        'layer1.bias': torch.randn(512),
        'layer2.weight': torch.randn(256, 512),
        'layer2.bias': torch.randn(256),
    }

    # Create variations
    model_a = {k: v + torch.randn_like(v) * 0.1 for k, v in dummy_model.items()}
    model_b = {k: v + torch.randn_like(v) * 0.1 for k, v in dummy_model.items()}
    model_c = {k: v + torch.randn_like(v) * 0.1 for k, v in dummy_model.items()}

    # Demonstrate different merging strategies

    # 1. Linear merge
    merged_linear = pipeline.linear_merge([model_a, model_b, model_c], weights=[0.5, 0.3, 0.2])
    pipeline.save_merged_model(
        merged_linear,
        f"{config['output_dir']}/merged_linear.pt",
        {'strategy': 'linear', 'num_models': 3}
    )

    # 2. SLERP merge
    merged_slerp = pipeline.slerp_merge(model_a, model_b, t=0.5)
    pipeline.save_merged_model(
        merged_slerp,
        f"{config['output_dir']}/merged_slerp.pt",
        {'strategy': 'slerp', 't': 0.5}
    )

    # 3. Task arithmetic
    merged_task = pipeline.task_arithmetic_merge(dummy_model, [model_a, model_b], weights=[0.7, 0.3])
    pipeline.save_merged_model(
        merged_task,
        f"{config['output_dir']}/merged_task_arithmetic.pt",
        {'strategy': 'task_arithmetic'}
    )

    # 4. TIES merge
    merged_ties = pipeline.ties_merge(dummy_model, [model_a, model_b, model_c], k=0.2)
    pipeline.save_merged_model(
        merged_ties,
        f"{config['output_dir']}/merged_ties.pt",
        {'strategy': 'ties', 'k': 0.2}
    )

    # 5. DARE merge
    merged_dare = pipeline.dare_merge(dummy_model, [model_a, model_b, model_c], drop_rate=0.5)
    pipeline.save_merged_model(
        merged_dare,
        f"{config['output_dir']}/merged_dare.pt",
        {'strategy': 'dare', 'drop_rate': 0.5}
    )

    print("\n" + "="*60)
    print("Model Merging Complete!")
    print("="*60)
    print(f"All merged models saved to: {config['output_dir']}")


if __name__ == '__main__':
    main()
