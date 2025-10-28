"""
Model evaluation module for various benchmarks.
"""
from .mmlu_eval import MMLUEvaluator
from .coco_eval import COCOEvaluator

__all__ = ["MMLUEvaluator", "COCOEvaluator"]
