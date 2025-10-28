"""
Training recipes for SparkTrainer.

Provides standardized training workflows for:
- Text models (BERT, GPT-2, Llama LoRA)
- Vision models (ResNet, EfficientNet, ViT, Stable Diffusion LoRA)
- Audio models (Wav2Vec2, Whisper)
- Video models (TimeSformer, ViViT)
"""

from .recipe_interface import (
    TrainerRecipe,
    DistributedTrainerRecipe,
    AdapterTrainerRecipe,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    EvalMetrics,
    RecipeOutput,
    register_recipe,
    get_recipe,
    list_recipes,
    RECIPE_REGISTRY,
)

# Import all recipes to register them
from .text_recipes import (
    BERTClassificationRecipe,
    GPT2SFTRecipe,
    LlamaLoRARecipe,
)

from .vision_recipes import (
    ResNetClassificationRecipe,
    ViTClassificationRecipe,
    StableDiffusionLoRARecipe,
)

from .audio_video_recipes import (
    Wav2Vec2ASRRecipe,
    WhisperASRRecipe,
    VideoClassificationRecipe,
)

__all__ = [
    # Base classes
    'TrainerRecipe',
    'DistributedTrainerRecipe',
    'AdapterTrainerRecipe',

    # Config classes
    'DataConfig',
    'ModelConfig',
    'TrainingConfig',
    'EvalMetrics',
    'RecipeOutput',

    # Registry functions
    'register_recipe',
    'get_recipe',
    'list_recipes',
    'RECIPE_REGISTRY',

    # Text recipes
    'BERTClassificationRecipe',
    'GPT2SFTRecipe',
    'LlamaLoRARecipe',

    # Vision recipes
    'ResNetClassificationRecipe',
    'ViTClassificationRecipe',
    'StableDiffusionLoRARecipe',

    # Audio/Video recipes
    'Wav2Vec2ASRRecipe',
    'WhisperASRRecipe',
    'VideoClassificationRecipe',
]
