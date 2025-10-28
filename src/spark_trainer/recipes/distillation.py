"""
Knowledge distillation support for model compression.

Provides:
- Teacher-student training flows
- Logit distillation
- Feature distillation
- Temperature scaling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path
import logging

from .recipe_interface import (
    TrainerRecipe,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    EvalMetrics,
    RecipeOutput,
)

logger = logging.getLogger(__name__)


class DistillationLoss(nn.Module):
    """
    Combined distillation loss.

    Loss = alpha * distillation_loss + (1 - alpha) * student_loss
    """

    def __init__(
        self,
        alpha: float = 0.5,
        temperature: float = 3.0,
        distillation_type: str = "soft",  # soft, hard
    ):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.distillation_type = distillation_type

        self.student_criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distillation loss.

        Args:
            student_logits: Student model outputs
            teacher_logits: Teacher model outputs
            labels: Ground truth labels

        Returns:
            Combined loss
        """
        # Student loss (cross-entropy with labels)
        student_loss = self.student_criterion(student_logits, labels)

        # Distillation loss
        if self.distillation_type == "soft":
            # Soft targets with temperature scaling
            student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
            teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)

            distillation_loss = F.kl_div(
                student_soft,
                teacher_soft,
                reduction="batchmean",
            ) * (self.temperature ** 2)

        elif self.distillation_type == "hard":
            # Hard targets (argmax of teacher)
            teacher_labels = teacher_logits.argmax(dim=-1)
            distillation_loss = self.student_criterion(student_logits, teacher_labels)

        else:
            raise ValueError(f"Unknown distillation type: {self.distillation_type}")

        # Combine losses
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss

        return total_loss


class FeatureDistillationLoss(nn.Module):
    """
    Feature-level distillation loss.

    Matches intermediate representations between teacher and student.
    """

    def __init__(
        self,
        projection_dim: Optional[int] = None,
        loss_type: str = "mse",  # mse, cosine
    ):
        super().__init__()
        self.projection_dim = projection_dim
        self.loss_type = loss_type

        # Projection layers (if needed to match dimensions)
        self.projections = nn.ModuleDict()

    def add_projection(self, name: str, input_dim: int, output_dim: int):
        """Add a projection layer to match feature dimensions."""
        self.projections[name] = nn.Linear(input_dim, output_dim)

    def forward(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute feature distillation loss.

        Args:
            student_features: Dict of student intermediate features
            teacher_features: Dict of teacher intermediate features

        Returns:
            Feature distillation loss
        """
        total_loss = 0.0
        num_layers = 0

        for name in student_features.keys():
            if name not in teacher_features:
                continue

            student_feat = student_features[name]
            teacher_feat = teacher_features[name]

            # Project if needed
            if name in self.projections:
                student_feat = self.projections[name](student_feat)

            # Compute loss
            if self.loss_type == "mse":
                loss = F.mse_loss(student_feat, teacher_feat)
            elif self.loss_type == "cosine":
                loss = 1 - F.cosine_similarity(
                    student_feat.view(student_feat.size(0), -1),
                    teacher_feat.view(teacher_feat.size(0), -1),
                    dim=-1
                ).mean()
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")

            total_loss += loss
            num_layers += 1

        if num_layers > 0:
            total_loss /= num_layers

        return total_loss


class DistillationTrainer:
    """
    Knowledge distillation trainer.

    Supports:
    - Logit distillation
    - Feature distillation
    - Combined distillation
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        device: str = "cuda",
        alpha: float = 0.5,
        temperature: float = 3.0,
        distillation_type: str = "soft",
        use_feature_distillation: bool = False,
        feature_loss_weight: float = 0.1,
    ):
        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        self.device = device

        # Freeze teacher
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # Distillation loss
        self.distillation_loss = DistillationLoss(
            alpha=alpha,
            temperature=temperature,
            distillation_type=distillation_type,
        )

        # Feature distillation (optional)
        self.use_feature_distillation = use_feature_distillation
        self.feature_loss_weight = feature_loss_weight

        if use_feature_distillation:
            self.feature_distillation_loss = FeatureDistillationLoss()

        # Hooks for feature extraction
        self.teacher_features = {}
        self.student_features = {}

    def register_feature_hooks(
        self,
        teacher_layers: List[str],
        student_layers: List[str],
    ):
        """
        Register hooks to extract intermediate features.

        Args:
            teacher_layers: List of teacher layer names
            student_layers: List of student layer names
        """
        def get_teacher_hook(name):
            def hook(module, input, output):
                self.teacher_features[name] = output
            return hook

        def get_student_hook(name):
            def hook(module, input, output):
                self.student_features[name] = output
            return hook

        # Register teacher hooks
        for layer_name in teacher_layers:
            layer = dict(self.teacher_model.named_modules())[layer_name]
            layer.register_forward_hook(get_teacher_hook(layer_name))

        # Register student hooks
        for layer_name in student_layers:
            layer = dict(self.student_model.named_modules())[layer_name]
            layer.register_forward_hook(get_student_hook(layer_name))

        logger.info(f"Registered feature hooks for {len(teacher_layers)} layers")

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Input batch
            optimizer: Optimizer

        Returns:
            Dict of losses
        """
        self.student_model.train()

        # Move batch to device
        inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(self.device)

        # Forward pass - teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits if hasattr(teacher_outputs, 'logits') else teacher_outputs

        # Forward pass - student
        student_outputs = self.student_model(**inputs)
        student_logits = student_outputs.logits if hasattr(student_outputs, 'logits') else student_outputs

        # Compute distillation loss
        loss = self.distillation_loss(student_logits, teacher_logits, labels)
        losses = {"distillation_loss": loss.item()}

        # Add feature distillation if enabled
        if self.use_feature_distillation and self.teacher_features and self.student_features:
            feature_loss = self.feature_distillation_loss(self.student_features, self.teacher_features)
            loss = loss + self.feature_loss_weight * feature_loss
            losses["feature_loss"] = feature_loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses["total_loss"] = loss.item()

        # Clear feature caches
        self.teacher_features.clear()
        self.student_features.clear()

        return losses

    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate student model.

        Args:
            dataloader: Evaluation dataloader

        Returns:
            Evaluation metrics
        """
        self.student_model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(self.device)

                outputs = self.student_model(**inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs

                loss = criterion(logits, labels)
                total_loss += loss.item()

                predictions = logits.argmax(dim=-1)
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
        }


class DistillationRecipe(TrainerRecipe):
    """
    Knowledge distillation recipe.

    Trains a small student model to mimic a large teacher model.
    """

    def __init__(
        self,
        output_dir: str,
        teacher_model_path: str,
        experiment_name: Optional[str] = None,
        alpha: float = 0.5,
        temperature: float = 3.0,
        use_feature_distillation: bool = False,
    ):
        super().__init__(output_dir, experiment_name)

        self.teacher_model_path = Path(teacher_model_path)
        self.alpha = alpha
        self.temperature = temperature
        self.use_feature_distillation = use_feature_distillation

        self.teacher_model = None
        self.distillation_trainer = None

    def load_teacher(self, model_config: Optional[ModelConfig] = None):
        """Load teacher model."""
        # Override in subclass to load specific model type
        raise NotImplementedError("Subclass must implement load_teacher()")

    def prepare(self, data_config: DataConfig) -> Tuple[Any, Any, Any]:
        """Prepare data for distillation."""
        # Override in subclass
        raise NotImplementedError("Subclass must implement prepare()")

    def build(self, model_config: ModelConfig) -> Any:
        """Build student model."""
        # Override in subclass
        raise NotImplementedError("Subclass must implement build()")

    def train(self, training_config: TrainingConfig) -> Dict[str, Any]:
        """Train student model with distillation."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize distillation trainer
        self.distillation_trainer = DistillationTrainer(
            teacher_model=self.teacher_model,
            student_model=self.model,
            device=device,
            alpha=self.alpha,
            temperature=self.temperature,
            use_feature_distillation=self.use_feature_distillation,
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )

        # DataLoader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=True,
            num_workers=self.data_config.num_workers,
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=False,
            num_workers=self.data_config.num_workers,
        ) if self.val_dataset else None

        # Training loop
        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

        for epoch in range(training_config.epochs):
            # Train
            epoch_losses = []

            for batch in train_loader:
                losses = self.distillation_trainer.train_step(batch, optimizer)
                epoch_losses.append(losses["total_loss"])

            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            history["train_loss"].append(avg_train_loss)

            # Validate
            if val_loader:
                val_metrics = self.distillation_trainer.evaluate(val_loader)
                history["val_loss"].append(val_metrics["loss"])
                history["val_accuracy"].append(val_metrics["accuracy"])

                logger.info(
                    f"Epoch {epoch + 1}/{training_config.epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch + 1}/{training_config.epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}"
                )

        return history

    def eval(self, split: str = "test") -> EvalMetrics:
        """Evaluate student model."""
        dataset = self.test_dataset if split == "test" else self.val_dataset

        loader = DataLoader(
            dataset,
            batch_size=self.data_config.batch_size,
            shuffle=False,
            num_workers=self.data_config.num_workers,
        )

        metrics = self.distillation_trainer.evaluate(loader)

        return EvalMetrics(
            loss=metrics["loss"],
            accuracy=metrics["accuracy"],
        )

    def package(self, export_format: str = "pytorch") -> RecipeOutput:
        """Package student model."""
        model_path = self.output_dir / "student_model"

        # Save student model
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(model_path)
        else:
            torch.save(self.model.state_dict(), model_path / "model.pth")

        metrics = self.eval(split="test")

        return RecipeOutput(
            model_path=str(model_path),
            metrics=metrics,
            config={
                "model_config": self.model_config.__dict__,
                "training_config": self.training_config.__dict__,
                "distillation": {
                    "alpha": self.alpha,
                    "temperature": self.temperature,
                    "teacher_model": str(self.teacher_model_path),
                },
            },
            artifacts=[str(self.output_dir / "config.json")],
            metadata={"distillation": True},
        )


def calculate_compression_ratio(teacher_model: nn.Module, student_model: nn.Module) -> Dict[str, float]:
    """
    Calculate compression ratio between teacher and student.

    Args:
        teacher_model: Teacher model
        student_model: Student model

    Returns:
        Dict with compression metrics
    """
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())

    compression_ratio = teacher_params / student_params
    param_reduction = (1 - student_params / teacher_params) * 100

    return {
        "teacher_params": teacher_params,
        "student_params": student_params,
        "compression_ratio": compression_ratio,
        "param_reduction_pct": param_reduction,
    }
