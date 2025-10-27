"""
Diffusion/Video Generation Model Trainer
Supports: Stable Video Diffusion (SVD), Wan2.2-Animate, AnimateDiff, etc.
"""
from pathlib import Path
from typing import Optional

import torch
from diffusers import (
    AnimateDiffPipeline,
    DDPMScheduler,
    StableVideoDiffusionPipeline,
    UNet2DConditionModel,
    UNetSpatioTemporalConditionModel,
)
from diffusers.optimization import get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import TrainConfig
from ..data import VideoFrameDataset
from ..logger import get_logger

logger = get_logger()


class DiffusionTrainer:
    """
    Trainer for diffusion-based video generation models.
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
        self.noise_scheduler = None
        self.optimizer = None
        self.scheduler = None
        self.accelerator = None

        logger.info("DiffusionTrainer initialized")
        logger.info(f"Model: {config.model_name}")
        logger.info(f"Device: {self.device}")

    def setup_model(self):
        """Load model and noise scheduler."""
        logger.info(f"Loading diffusion model: {self.config.model_name}")

        model_name = self.config.model_name.lower()

        if "stable-video-diffusion" in model_name or "svd" in model_name:
            # Load SVD UNet
            try:
                self.model = UNetSpatioTemporalConditionModel.from_pretrained(
                    self.config.model_name,
                    subfolder="unet" if "/" in self.config.model_name else None,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                )
            except Exception:
                # Try loading full pipeline and extract UNet
                pipeline = StableVideoDiffusionPipeline.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                )
                self.model = pipeline.unet
                self.noise_scheduler = pipeline.scheduler

        elif "animatediff" in model_name or "animate" in model_name:
            # Load AnimateDiff or similar
            try:
                self.model = UNet2DConditionModel.from_pretrained(
                    self.config.model_name,
                    subfolder="unet" if "/" in self.config.model_name else None,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                )
            except Exception:
                pipeline = AnimateDiffPipeline.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                )
                self.model = pipeline.unet
                self.noise_scheduler = pipeline.scheduler

        else:
            # Generic UNet loading
            try:
                self.model = UNet2DConditionModel.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                )
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise

        # Setup noise scheduler if not already loaded
        if self.noise_scheduler is None:
            self.noise_scheduler = DDPMScheduler.from_pretrained(
                self.config.model_name,
                subfolder="scheduler" if "/" in self.config.model_name else None,
            )

        # Load from checkpoint if resuming
        if self.config.resume_from_checkpoint:
            logger.info(f"Loading checkpoint: {self.config.resume_from_checkpoint}")
            checkpoint = torch.load(self.config.resume_from_checkpoint)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        if not self.config.use_accelerate:
            self.model.to(self.device)

        logger.info(f"Model loaded. Trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

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
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

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
            return train_dataloader, eval_dataloader

        try:
            from accelerate import Accelerator
        except ImportError:
            logger.error("Accelerate not installed. Install with: pip install accelerate")
            raise

        logger.info("Setting up Accelerate for distributed training")

        kwargs = {"mixed_precision": self.config.mixed_precision}
        if self.config.use_deepspeed and self.config.deepspeed_config:
            kwargs["deepspeed_plugin"] = str(self.config.deepspeed_config)

        self.accelerator = Accelerator(**kwargs)

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
        Single training step using diffusion loss.

        Args:
            batch: Batch from dataloader

        Returns:
            Loss value
        """
        self.model.train()

        # Get video clips (B, T, C, H, W)
        clips = batch["clip"].to(self.device)
        batch_size = clips.shape[0]

        # Sample random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=self.device
        ).long()

        # Add noise to clips
        noise = torch.randn_like(clips)
        noisy_clips = self.noise_scheduler.add_noise(clips, noise, timesteps)

        # Predict noise
        noise_pred = self.model(
            noisy_clips,
            timesteps,
            return_dict=False,
        )[0]

        # Compute loss (MSE between predicted and actual noise)
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

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
        logger.info("Starting diffusion training...")

        # Setup model and optimizer
        self.setup_model()
        self.setup_optimizer()

        # Create dataset and dataloader
        import torchvision.transforms as T

        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        train_dataset = VideoFrameDataset(
            manifest_path=self.config.manifest_path,
            clip_length=self.config.clip_length,
            clip_stride=self.config.clip_stride,
            clip_sampling=self.config.clip_sampling,
            transform=transform,
            shuffle_videos=True,
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
            val_dataset = VideoFrameDataset(
                manifest_path=self.config.val_manifest_path,
                clip_length=self.config.clip_length,
                clip_stride=self.config.clip_stride,
                clip_sampling="center",
                transform=transform,
                shuffle_videos=False,
                seed=self.config.seed,
            )
            eval_dataloader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=True,
            )

        # Setup scheduler
        estimated_steps_per_epoch = 1000
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

                if global_step % self.config.logging_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    logger.info(f"Step {global_step} | Loss: {avg_loss:.4f} | LR: {self.scheduler.get_last_lr()[0]:.2e}")
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

                if global_step % self.config.save_steps == 0:
                    checkpoint_path = output_dir / f"checkpoint-{global_step}"
                    self.save_checkpoint(checkpoint_path, global_step)

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
