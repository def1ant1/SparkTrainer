"""
Vision-Language Model Trainer
Supports various VL models: BLIP, BLIP-2, LLaVA, InternVL, Qwen2-VL, etc.
"""
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    BlipForConditionalGeneration,
    BlipProcessor,
    get_scheduler,
)

from ..config import TrainConfig
from ..data import ImageCaptionDataset
from ..logger import get_logger

logger = get_logger()


class VisionLanguageTrainer:
    """
    Trainer for vision-language models.
    """

    def __init__(self, config: TrainConfig):
        """
        Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.optimizer = None
        self.scheduler = None
        self.accelerator = None

        logger.info("VisionLanguageTrainer initialized")
        logger.info(f"Model: {config.model_name}")
        logger.info(f"Device: {self.device}")

    def setup_model(self):
        """Load model and processor."""
        logger.info(f"Loading model: {self.config.model_name}")

        # Detect model type from name
        model_name = self.config.model_name.lower()

        if "blip-2" in model_name or "blip2" in model_name:
            self.processor = Blip2Processor.from_pretrained(self.config.model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
        elif "blip" in model_name:
            self.processor = BlipProcessor.from_pretrained(self.config.model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
        else:
            # Generic AutoModel
            try:
                self.processor = AutoProcessor.from_pretrained(self.config.model_name, trust_remote_code=True)
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=True,
                )
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise

        # Load from checkpoint if resuming
        if self.config.resume_from_checkpoint:
            logger.info(f"Loading checkpoint: {self.config.resume_from_checkpoint}")
            checkpoint = torch.load(self.config.resume_from_checkpoint)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        if not self.config.use_accelerate:
            self.model.to(self.device)

        logger.info(f"Model loaded successfully. Trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def setup_optimizer(self):
        """Set up optimizer and scheduler."""
        if self.config.optimizer == "adamw":
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
            )
        elif self.config.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

        # Resume optimizer if checkpoint exists
        if self.config.resume_from_checkpoint:
            checkpoint = torch.load(self.config.resume_from_checkpoint)
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(f"Optimizer: {self.config.optimizer}, LR: {self.config.learning_rate}")

    def setup_scheduler(self, num_training_steps: int):
        """Set up learning rate scheduler."""
        self.scheduler = get_scheduler(
            name=self.config.scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps,
        )

        logger.info(f"Scheduler: {self.config.scheduler}, warmup steps: {self.config.warmup_steps}")

    def setup_accelerate(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None):
        """Set up Accelerate for distributed training."""
        if not self.config.use_accelerate:
            return

        try:
            from accelerate import Accelerator
        except ImportError:
            logger.error("Accelerate not installed. Install with: pip install accelerate")
            raise

        logger.info("Setting up Accelerate for distributed training")

        # Initialize Accelerator
        kwargs = {"mixed_precision": self.config.mixed_precision}
        if self.config.use_deepspeed and self.config.deepspeed_config:
            kwargs["deepspeed_plugin"] = str(self.config.deepspeed_config)

        self.accelerator = Accelerator(**kwargs)

        # Prepare model, optimizer, dataloaders
        if eval_dataloader:
            self.model, self.optimizer, train_dataloader, eval_dataloader = self.accelerator.prepare(
                self.model, self.optimizer, train_dataloader, eval_dataloader
            )
        else:
            self.model, self.optimizer, train_dataloader = self.accelerator.prepare(
                self.model, self.optimizer, train_dataloader
            )

        self.device = self.accelerator.device
        logger.info(f"Accelerate setup complete. Device: {self.device}, Process: {self.accelerator.process_index}")

        return train_dataloader, eval_dataloader

    def train_step(self, batch):
        """
        Single training step.

        Args:
            batch: Batch from dataloader

        Returns:
            Loss value
        """
        self.model.train()

        images = batch["image"]
        captions = batch["caption"]

        # Process inputs
        inputs = self.processor(images=images, text=captions, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        outputs = self.model(**inputs)
        loss = outputs.loss

        # Backward pass
        if self.config.use_accelerate:
            self.accelerator.backward(loss)
        else:
            loss.backward()

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            if self.config.use_accelerate:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return loss.item()

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")

        # Setup model and optimizer
        self.setup_model()
        self.setup_optimizer()

        # Create dataset and dataloader
        import torchvision.transforms as T

        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_dataset = ImageCaptionDataset(
            manifest_path=self.config.manifest_path,
            transform=transform,
            shuffle=True,
            seed=self.config.seed,
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        eval_dataloader = None
        if self.config.val_manifest_path:
            val_dataset = ImageCaptionDataset(
                manifest_path=self.config.val_manifest_path,
                transform=transform,
                shuffle=False,
                seed=self.config.seed,
            )
            eval_dataloader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=True,
            )

        # Setup scheduler
        # Note: For IterableDataset, we need to estimate steps
        estimated_steps_per_epoch = 1000  # Adjust based on your data
        num_training_steps = estimated_steps_per_epoch * self.config.num_epochs
        self.setup_scheduler(num_training_steps)

        # Setup Accelerate if enabled
        if self.config.use_accelerate:
            train_dataloader, eval_dataloader = self.setup_accelerate(train_dataloader, eval_dataloader)

        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Training loop
        global_step = 0
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")

            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
            epoch_loss = 0.0
            num_batches = 0

            for batch in progress_bar:
                loss = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1
                global_step += 1

                # Logging
                if global_step % self.config.logging_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    logger.info(f"Step {global_step} | Loss: {avg_loss:.4f} | LR: {self.scheduler.get_last_lr()[0]:.2e}")
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    checkpoint_path = output_dir / f"checkpoint-{global_step}"
                    self.save_checkpoint(checkpoint_path, global_step)

            # Epoch summary
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            logger.info(f"Epoch {epoch + 1} complete | Avg Loss: {avg_epoch_loss:.4f}")

        # Save final model
        final_path = output_dir / "final_model"
        self.save_checkpoint(final_path, global_step)
        logger.info(f"Training complete! Model saved to {final_path}")

    def save_checkpoint(self, path: Path, global_step: int):
        """Save model checkpoint."""
        path.mkdir(parents=True, exist_ok=True)

        if self.config.use_accelerate:
            self.accelerator.save_state(str(path))
        else:
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "global_step": global_step,
                "config": self.config.dict(),
            }
            torch.save(checkpoint, path / "checkpoint.pt")

        logger.info(f"Checkpoint saved: {path}")
