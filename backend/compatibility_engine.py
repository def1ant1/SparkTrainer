"""
Compatibility engine for validating experiment configurations.
Checks model, dataset, recipe, and adapter compatibility.
"""
from typing import Dict, List, Optional, Any, Tuple
import re


class CompatibilityEngine:
    """Engine for validating experiment component compatibility."""

    # Modality compatibility matrix
    MODALITY_COMPAT = {
        'text': ['text', 'multimodal'],
        'image': ['image', 'multimodal'],
        'audio': ['audio', 'multimodal'],
        'video': ['video', 'multimodal'],
        'multimodal': ['text', 'image', 'audio', 'video', 'multimodal']
    }

    # Recipe requirements by type
    RECIPE_REQUIREMENTS = {
        'lora': {
            'min_hidden_size': 64,
            'requires_trainable': True,
            'supports_quantized': True,
            'dataset_schema': ['text+labels', 'image+labels', 'text+text']
        },
        'qlora': {
            'min_hidden_size': 64,
            'requires_trainable': True,
            'supports_quantized': True,
            'requires_quantized': False,  # Will quantize if needed
            'dataset_schema': ['text+labels', 'text+text']
        },
        'full_ft': {
            'requires_trainable': True,
            'supports_quantized': False,  # Cannot full FT quantized models
            'dataset_schema': ['text+labels', 'image+labels', 'text+text']
        },
        'prompt_tuning': {
            'requires_trainable': True,
            'supports_quantized': True,
            'dataset_schema': ['text+text', 'text+labels']
        },
        'vision_lora': {
            'min_hidden_size': 64,
            'requires_trainable': True,
            'supports_quantized': False,
            'dataset_schema': ['image+labels', 'image+captions']
        },
        'asr': {
            'requires_trainable': True,
            'supports_quantized': False,
            'dataset_schema': ['audio+text']
        },
        'diffusion': {
            'requires_trainable': True,
            'supports_quantized': False,
            'dataset_schema': ['image+captions', 'image']
        }
    }

    @staticmethod
    def check_compatibility(
        base_model: Optional[Dict],
        dataset: Optional[Dict],
        recipe: Optional[Dict],
        adapters: Optional[List[Dict]] = None
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Check compatibility between components.

        Returns:
            Tuple of (ok: bool, warnings: List[str], errors: List[str])
        """
        warnings = []
        errors = []

        # Check model-recipe compatibility
        if base_model and recipe:
            model_recipe_ok, model_recipe_msgs = CompatibilityEngine._check_model_recipe(
                base_model, recipe
            )
            for msg in model_recipe_msgs:
                if 'error' in msg.lower() or 'cannot' in msg.lower():
                    errors.append(msg)
                else:
                    warnings.append(msg)

        # Check model-dataset compatibility
        if base_model and dataset:
            model_dataset_ok, model_dataset_msgs = CompatibilityEngine._check_model_dataset(
                base_model, dataset
            )
            for msg in model_dataset_msgs:
                if 'error' in msg.lower() or 'incompatible' in msg.lower():
                    errors.append(msg)
                else:
                    warnings.append(msg)

        # Check recipe-dataset compatibility
        if recipe and dataset:
            recipe_dataset_ok, recipe_dataset_msgs = CompatibilityEngine._check_recipe_dataset(
                recipe, dataset
            )
            for msg in recipe_dataset_msgs:
                if 'error' in msg.lower() or 'requires' in msg.lower():
                    errors.append(msg)
                else:
                    warnings.append(msg)

        # Check adapter compatibility
        if base_model and adapters:
            adapter_ok, adapter_msgs = CompatibilityEngine._check_adapters(
                base_model, adapters
            )
            for msg in adapter_msgs:
                if 'error' in msg.lower() or 'mismatch' in msg.lower():
                    errors.append(msg)
                else:
                    warnings.append(msg)

        ok = len(errors) == 0
        return ok, warnings, errors

    @staticmethod
    def _check_model_recipe(base_model: Dict, recipe: Dict) -> Tuple[bool, List[str]]:
        """Check compatibility between base model and recipe."""
        messages = []

        recipe_type = recipe.get('recipe_type')
        requirements = CompatibilityEngine.RECIPE_REQUIREMENTS.get(recipe_type, {})

        # Check trainable requirement
        if requirements.get('requires_trainable') and not base_model.get('trainable'):
            messages.append(
                f"ERROR: Model '{base_model.get('name')}' is not trainable. "
                f"Recipe '{recipe.get('display_name')}' requires a trainable model."
            )

        # Check quantization compatibility
        is_quantized = base_model.get('quantized', False)
        if is_quantized and not requirements.get('supports_quantized', False):
            messages.append(
                f"ERROR: Recipe '{recipe.get('display_name')}' does not support quantized models. "
                f"Model '{base_model.get('name')}' is {base_model.get('dtype')}."
            )

        # Check hidden size for LoRA-based methods
        min_hidden = requirements.get('min_hidden_size')
        if min_hidden and base_model.get('hidden_size'):
            if base_model['hidden_size'] < min_hidden:
                messages.append(
                    f"WARNING: Model hidden size ({base_model['hidden_size']}) is small. "
                    f"LoRA typically works better with hidden_size >= {min_hidden}."
                )

        # Check modality compatibility
        model_modality = base_model.get('modality')
        recipe_modality = recipe.get('modality')
        if model_modality and recipe_modality:
            compat_modalities = CompatibilityEngine.MODALITY_COMPAT.get(recipe_modality, [])
            if model_modality not in compat_modalities:
                messages.append(
                    f"ERROR: Modality mismatch. Model is '{model_modality}', "
                    f"recipe expects '{recipe_modality}'."
                )

        # Check GGUF models
        if base_model.get('is_gguf'):
            messages.append(
                "ERROR: GGUF models cannot be trained. Please use the original model format."
            )

        ok = not any('ERROR' in msg for msg in messages)
        return ok, messages

    @staticmethod
    def _check_model_dataset(base_model: Dict, dataset: Dict) -> Tuple[bool, List[str]]:
        """Check compatibility between base model and dataset."""
        messages = []

        # Check modality compatibility
        model_modality = base_model.get('modality')
        dataset_modality = dataset.get('modality')

        if model_modality and dataset_modality:
            compat_modalities = CompatibilityEngine.MODALITY_COMPAT.get(model_modality, [])
            if dataset_modality not in compat_modalities:
                messages.append(
                    f"ERROR: Modality incompatible. Model expects '{model_modality}', "
                    f"dataset is '{dataset_modality}'."
                )

        # Check tokenizer compatibility for text models
        if model_modality == 'text' and not base_model.get('tokenizer_path'):
            messages.append(
                "WARNING: Model does not have a tokenizer configured. "
                "You may need to specify a tokenizer manually."
            )

        # Check dataset size
        num_samples = dataset.get('num_samples', 0)
        if num_samples < 100:
            messages.append(
                f"WARNING: Dataset is very small ({num_samples} samples). "
                "Consider using a larger dataset for better results."
            )
        elif num_samples > 1000000:
            messages.append(
                f"INFO: Large dataset ({num_samples:,} samples). "
                "Training may take significant time. Consider using QLoRA + FSDP."
            )

        ok = not any('ERROR' in msg for msg in messages)
        return ok, messages

    @staticmethod
    def _check_recipe_dataset(recipe: Dict, dataset: Dict) -> Tuple[bool, List[str]]:
        """Check compatibility between recipe and dataset."""
        messages = []

        recipe_type = recipe.get('recipe_type')
        requirements = CompatibilityEngine.RECIPE_REQUIREMENTS.get(recipe_type, {})

        # Check dataset schema requirements
        required_schemas = requirements.get('dataset_schema', [])
        dataset_schema = dataset.get('metadata', {}).get('schema')

        if required_schemas and dataset_schema:
            if dataset_schema not in required_schemas:
                messages.append(
                    f"WARNING: Dataset schema '{dataset_schema}' may not be optimal. "
                    f"Recipe '{recipe.get('display_name')}' works best with: {', '.join(required_schemas)}."
                )

        # Check if dataset has been validated
        if not dataset.get('integrity_checked'):
            messages.append(
                "WARNING: Dataset integrity has not been verified. "
                "Run validation before training to catch potential issues."
            )
        elif dataset.get('integrity_checked') and not dataset.get('integrity_passed'):
            messages.append(
                "ERROR: Dataset failed integrity check. "
                "Please review and fix dataset issues before training."
            )

        ok = not any('ERROR' in msg for msg in messages)
        return ok, messages

    @staticmethod
    def _check_adapters(base_model: Dict, adapters: List[Dict]) -> Tuple[bool, List[str]]:
        """Check adapter compatibility with base model."""
        messages = []

        hidden_size = base_model.get('hidden_size')
        model_id = base_model.get('id')

        for adapter in adapters:
            adapter_name = adapter.get('name', 'Unknown')

            # Check if adapter is for this base model
            adapter_model_id = adapter.get('base_model_id')
            if adapter_model_id and adapter_model_id != model_id:
                messages.append(
                    f"ERROR: Adapter '{adapter_name}' was trained for a different base model. "
                    f"Cannot attach to '{base_model.get('name')}'."
                )

            # Check LoRA rank compatibility
            rank = adapter.get('rank')
            if rank and hidden_size:
                if rank >= hidden_size:
                    messages.append(
                        f"WARNING: Adapter '{adapter_name}' has rank={rank} which is >= "
                        f"hidden_size={hidden_size}. This may reduce efficiency."
                    )

            # Check adapter status
            status = adapter.get('status')
            if status != 'ready':
                messages.append(
                    f"WARNING: Adapter '{adapter_name}' status is '{status}'. "
                    "It may not be ready for use."
                )

        ok = not any('ERROR' in msg for msg in messages)
        return ok, messages

    @staticmethod
    def get_dataset_schema(dataset: Dict) -> Optional[str]:
        """Infer dataset schema from metadata or statistics."""
        metadata = dataset.get('metadata', {})
        stats = dataset.get('statistics', {})

        # Check if schema is explicitly defined
        if 'schema' in metadata:
            return metadata['schema']

        # Try to infer from statistics
        has_text = stats.get('has_text_column', False)
        has_labels = stats.get('has_labels_column', False)
        has_images = stats.get('has_image_column', False)
        has_audio = stats.get('has_audio_column', False)
        has_captions = stats.get('has_captions_column', False)

        if has_text and has_labels:
            return 'text+labels'
        elif has_text and not has_labels:
            return 'text+text'
        elif has_images and has_captions:
            return 'image+captions'
        elif has_images and has_labels:
            return 'image+labels'
        elif has_audio and has_text:
            return 'audio+text'
        elif has_images:
            return 'image'

        return None

    @staticmethod
    def suggest_recipe(base_model: Dict, dataset: Dict) -> List[Dict[str, Any]]:
        """
        Suggest compatible recipes based on model and dataset.

        Returns list of recipe suggestions with compatibility scores.
        """
        suggestions = []

        model_modality = base_model.get('modality')
        dataset_modality = dataset.get('modality')
        is_quantized = base_model.get('quantized', False)
        is_trainable = base_model.get('trainable', True)
        num_samples = dataset.get('num_samples', 0)

        # Text modality recipes
        if model_modality == 'text' and dataset_modality in ['text', 'multimodal']:
            if is_trainable:
                # QLoRA recommended for large models or datasets
                if base_model.get('params_b', 0) > 7 or num_samples > 100000:
                    suggestions.append({
                        'recipe_type': 'qlora',
                        'display_name': 'QLoRA (Recommended)',
                        'reason': 'Memory-efficient for large models/datasets',
                        'score': 0.95
                    })

                # LoRA for medium-sized models
                if not is_quantized:
                    suggestions.append({
                        'recipe_type': 'lora',
                        'display_name': 'LoRA',
                        'reason': 'Fast training with good quality',
                        'score': 0.90
                    })

                # Full fine-tuning for small models with small datasets
                if base_model.get('params_b', 0) < 1 and num_samples < 10000:
                    suggestions.append({
                        'recipe_type': 'full_ft',
                        'display_name': 'Full Fine-Tune',
                        'reason': 'Best quality for small models',
                        'score': 0.85
                    })

        # Image modality recipes
        elif model_modality == 'image' and dataset_modality in ['image', 'multimodal']:
            if is_trainable and not is_quantized:
                suggestions.append({
                    'recipe_type': 'vision_lora',
                    'display_name': 'Vision LoRA',
                    'reason': 'Efficient fine-tuning for vision models',
                    'score': 0.90
                })

        # Audio modality recipes
        elif model_modality == 'audio' and dataset_modality in ['audio', 'multimodal']:
            if is_trainable:
                suggestions.append({
                    'recipe_type': 'asr',
                    'display_name': 'ASR Fine-Tune',
                    'reason': 'Optimized for speech recognition',
                    'score': 0.90
                })

        # Sort by score
        suggestions.sort(key=lambda x: x['score'], reverse=True)
        return suggestions
