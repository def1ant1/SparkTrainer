"""
Trainers module - organized by task family.
"""
from .vision_language import VisionLanguageTrainer
from .diffusion import DiffusionTrainer

__all__ = ["VisionLanguageTrainer", "DiffusionTrainer"]
