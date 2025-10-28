"""
Trainer Recipe Interface - Standardized training workflow.

Defines a stable interface for training recipes:
prepare(data) → build(model_cfg) → train(hparams) → eval(metrics) → package()
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import torch
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data configuration."""
    dataset_path: str
    batch_size: int = 32
    num_workers: int = 4
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    preprocessing: Optional[Dict] = None


@dataclass
class ModelConfig:
    """Model configuration."""
    architecture: str
    pretrained: Optional[str] = None
    num_classes: Optional[int] = None
    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None
    dropout: float = 0.1
    custom_params: Optional[Dict] = None


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 10
    warmup_steps: int = 0
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    optimizer: str = "adamw"
    scheduler: str = "linear"
    mixed_precision: str = "fp16"  # fp16, bf16, fp32
    gradient_checkpointing: bool = False
    early_stopping_patience: int = 3
    save_strategy: str = "epoch"  # epoch, steps
    save_steps: Optional[int] = None
    eval_strategy: str = "epoch"  # epoch, steps
    eval_steps: Optional[int] = None
    logging_steps: int = 10


@dataclass
class EvalMetrics:
    """Evaluation metrics."""
    loss: float
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    perplexity: Optional[float] = None
    bleu: Optional[float] = None
    custom_metrics: Optional[Dict[str, float]] = None


@dataclass
class RecipeOutput:
    """Output from recipe execution."""
    model_path: str
    metrics: EvalMetrics
    config: Dict[str, Any]
    artifacts: List[str]
    metadata: Dict[str, Any]


class TrainerRecipe(ABC):
    """
    Base class for training recipes.

    All training recipes must implement this interface.
    """

    def __init__(
        self,
        output_dir: str,
        experiment_name: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.data_config: Optional[DataConfig] = None
        self.model_config: Optional[ModelConfig] = None
        self.training_config: Optional[TrainingConfig] = None

        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @abstractmethod
    def prepare(self, data_config: DataConfig) -> Tuple[Any, Any, Any]:
        """
        Prepare datasets for training.

        Args:
            data_config: Data configuration

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        pass

    @abstractmethod
    def build(self, model_config: ModelConfig) -> Any:
        """
        Build model from configuration.

        Args:
            model_config: Model configuration

        Returns:
            Model instance
        """
        pass

    @abstractmethod
    def train(self, training_config: TrainingConfig) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            training_config: Training hyperparameters

        Returns:
            Training history and metrics
        """
        pass

    @abstractmethod
    def eval(self, split: str = "test") -> EvalMetrics:
        """
        Evaluate the model.

        Args:
            split: Dataset split to evaluate on (test, val)

        Returns:
            Evaluation metrics
        """
        pass

    @abstractmethod
    def package(self, export_format: str = "pytorch") -> RecipeOutput:
        """
        Package the trained model for deployment.

        Args:
            export_format: Export format (pytorch, onnx, torchscript, hf, vllm)

        Returns:
            Recipe output with model path and artifacts
        """
        pass

    def run(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        export_format: str = "pytorch",
    ) -> RecipeOutput:
        """
        Run complete training pipeline.

        Args:
            data_config: Data configuration
            model_config: Model configuration
            training_config: Training configuration
            export_format: Export format for final model

        Returns:
            Recipe output
        """
        logger.info(f"Starting recipe: {self.__class__.__name__}")

        # Step 1: Prepare data
        logger.info("Step 1/5: Preparing data...")
        self.data_config = data_config
        self.train_dataset, self.val_dataset, self.test_dataset = self.prepare(data_config)

        # Step 2: Build model
        logger.info("Step 2/5: Building model...")
        self.model_config = model_config
        self.model = self.build(model_config)

        # Step 3: Train
        logger.info("Step 3/5: Training...")
        self.training_config = training_config
        train_history = self.train(training_config)

        # Step 4: Evaluate
        logger.info("Step 4/5: Evaluating...")
        metrics = self.eval(split="test")

        # Step 5: Package
        logger.info("Step 5/5: Packaging...")
        output = self.package(export_format)

        logger.info(f"Recipe completed: {output.model_path}")

        return output

    def save_config(self, path: Optional[str] = None):
        """Save recipe configuration."""
        path = Path(path) if path else self.output_dir / "config.json"

        config = {
            'experiment_name': self.experiment_name,
            'recipe_class': self.__class__.__name__,
            'data_config': asdict(self.data_config) if self.data_config else None,
            'model_config': asdict(self.model_config) if self.model_config else None,
            'training_config': asdict(self.training_config) if self.training_config else None,
        }

        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Config saved to {path}")

    @classmethod
    def from_config(cls, config_path: str, output_dir: str):
        """Load recipe from configuration file."""
        with open(config_path, 'r') as f:
            config = json.load(f)

        recipe = cls(output_dir=output_dir, experiment_name=config.get('experiment_name'))

        if config.get('data_config'):
            recipe.data_config = DataConfig(**config['data_config'])

        if config.get('model_config'):
            recipe.model_config = ModelConfig(**config['model_config'])

        if config.get('training_config'):
            recipe.training_config = TrainingConfig(**config['training_config'])

        return recipe


class DistributedTrainerRecipe(TrainerRecipe):
    """
    Base class for distributed training recipes.

    Adds support for DDP, FSDP, DeepSpeed.
    """

    def __init__(
        self,
        output_dir: str,
        experiment_name: Optional[str] = None,
        distributed_backend: str = "ddp",  # ddp, fsdp, deepspeed
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        super().__init__(output_dir, experiment_name)

        self.distributed_backend = distributed_backend
        self.world_size = world_size or int(os.environ.get("WORLD_SIZE", 1))
        self.rank = rank or int(os.environ.get("RANK", 0))

        self.is_distributed = self.world_size > 1
        self.is_main_process = self.rank == 0

        if self.is_distributed:
            self._init_distributed()

    def _init_distributed(self):
        """Initialize distributed training."""
        if self.distributed_backend == "ddp":
            torch.distributed.init_process_group(backend="nccl")
            torch.cuda.set_device(self.rank)
            logger.info(f"DDP initialized: rank {self.rank}/{self.world_size}")

        elif self.distributed_backend == "fsdp":
            torch.distributed.init_process_group(backend="nccl")
            torch.cuda.set_device(self.rank)
            logger.info(f"FSDP initialized: rank {self.rank}/{self.world_size}")

        elif self.distributed_backend == "deepspeed":
            # DeepSpeed initialization handled by deepspeed.initialize()
            logger.info(f"DeepSpeed mode: rank {self.rank}/{self.world_size}")

    def wrap_model_distributed(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap model for distributed training."""
        if not self.is_distributed:
            return model

        if self.distributed_backend == "ddp":
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = model.to(self.rank)
            model = DDP(model, device_ids=[self.rank])

        elif self.distributed_backend == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import ShardingStrategy

            model = FSDP(
                model,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                device_id=self.rank,
            )

        # DeepSpeed wrapping handled in deepspeed.initialize()

        return model


class AdapterTrainerRecipe(TrainerRecipe):
    """
    Base class for adapter-based fine-tuning (LoRA, QLoRA, etc.).
    """

    def __init__(
        self,
        output_dir: str,
        experiment_name: Optional[str] = None,
        adapter_type: str = "lora",  # lora, qlora, ia3, dora
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        use_4bit: bool = False,
        use_8bit: bool = False,
    ):
        super().__init__(output_dir, experiment_name)

        self.adapter_type = adapter_type
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit

    def apply_adapters(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply adapters to model."""
        try:
            from peft import (
                get_peft_model,
                LoraConfig,
                TaskType,
                prepare_model_for_kbit_training,
            )

            # Prepare for quantization if needed
            if self.use_4bit or self.use_8bit:
                model = prepare_model_for_kbit_training(model)

            # Configure adapter
            if self.adapter_type in ["lora", "qlora"]:
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,  # Override in subclass if needed
                    r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout,
                    target_modules=self.target_modules,
                    bias="none",
                )

                model = get_peft_model(model, peft_config)

            logger.info(f"{self.adapter_type.upper()} adapters applied")
            logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        except ImportError:
            logger.warning("PEFT library not available. Install with: pip install peft")

        return model


# Recipe registry
RECIPE_REGISTRY: Dict[str, type] = {}


def register_recipe(name: str):
    """Decorator to register a recipe."""
    def decorator(cls):
        RECIPE_REGISTRY[name] = cls
        return cls
    return decorator


def get_recipe(name: str) -> type:
    """Get recipe class by name."""
    if name not in RECIPE_REGISTRY:
        raise ValueError(f"Recipe '{name}' not found. Available: {list(RECIPE_REGISTRY.keys())}")
    return RECIPE_REGISTRY[name]


def list_recipes() -> List[str]:
    """List available recipes."""
    return list(RECIPE_REGISTRY.keys())
