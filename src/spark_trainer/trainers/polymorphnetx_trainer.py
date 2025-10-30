"""
PolymorphNet-X Multimodal Trainer

A large-scale multimodal model trainer for Apotheon.ai's PolymorphNet-X architecture.
Supports text, image, video, and audio inputs with MoE (Mixture of Experts) architecture.

Features:
- 12B+ parameter model with 48 layers, 64 attention heads
- MoE with 64 experts and top-k=4 routing
- Multimodal adapters for image, audio, and video
- Gradient checkpointing and DeepSpeed ZeRO-3 support
- Curriculum learning for progressive multimodal training
- SentencePiece tokenization with special control tokens

@trainer_template: polymorphnetx-multimodal
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Import trainer_template decorator
try:
    from spark_trainer.trainer_registry import trainer_template
except ImportError:
    # Fallback if running standalone
    def trainer_template(**kwargs):
        def decorator(cls):
            cls._trainer_template_metadata = kwargs
            return cls
        return decorator

# Add polymorphnetx model directory to path
POLYMORPHNETX_PATH = Path(__file__).parent.parent.parent.parent / "models" / "polymorphnetx"
if POLYMORPHNETX_PATH.exists():
    sys.path.insert(0, str(POLYMORPHNETX_PATH))

try:
    # Import PolymorphNetX components
    from model.model import PolymorphNetX, build_causal_mask
    from model.multimodal_model import PolymorphNetXMultimodal
    from model.config import ModelConfig, CONF, TOKEN_IDS
    from model.adapters import ImageAdapter, AudioAdapter
    from train.metrics import compute_metrics
    from train.curriculum import CurriculumScheduler
    POLYMORPHNETX_AVAILABLE = True
except ImportError as e:
    logger.warning(f"PolymorphNetX model not available: {e}")
    POLYMORPHNETX_AVAILABLE = False

try:
    from accelerate import Accelerator
    from transformers import get_scheduler
except ImportError as e:
    logger.warning(f"Some dependencies missing for PolymorphNetX trainer: {e}")


# =============================================================================
# Dataset
# =============================================================================

class PolymorphNetXDataset(Dataset):
    """Dataset for PolymorphNet-X multimodal training"""

    def __init__(
        self,
        manifest_path: str,
        max_seq_len: int = 8192,
        modalities: List[str] = ['text', 'image', 'audio'],
        tokenizer_path: Optional[str] = None,
    ):
        self.manifest_path = manifest_path
        self.max_seq_len = max_seq_len
        self.modalities = modalities

        # Load tokenizer if available
        self.tokenizer = None
        if tokenizer_path and os.path.exists(tokenizer_path):
            try:
                import sentencepiece as spm
                self.tokenizer = spm.SentencePieceProcessor()
                self.tokenizer.Load(tokenizer_path)
                logger.info(f"Loaded SentencePiece tokenizer from {tokenizer_path}")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}")

        # Load manifest
        self.samples = []
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                for line in f:
                    if line.strip():
                        self.samples.append(json.loads(line))

        logger.info(f"Loaded {len(self.samples)} samples from {manifest_path}")

    def __len__(self):
        return len(self.samples)

    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text using SentencePiece or fallback to byte encoding"""
        if self.tokenizer:
            ids = [TOKEN_IDS['BOS']] + self.tokenizer.EncodeAsIds(text)[:self.max_seq_len - 2] + [TOKEN_IDS['EOS']]
        else:
            # Fallback to simple byte encoding
            ids = [TOKEN_IDS['BOS']] + [min(b, CONF.vocab_size - 1) for b in text.encode('utf-8')[:self.max_seq_len - 2]] + [TOKEN_IDS['EOS']]

        # Pad or truncate
        if len(ids) < self.max_seq_len:
            ids = ids + [TOKEN_IDS['PAD']] * (self.max_seq_len - len(ids))
        else:
            ids = ids[:self.max_seq_len]

        return torch.tensor(ids, dtype=torch.long)

    def _load_image(self, image_path: str) -> Optional[torch.Tensor]:
        """Load and preprocess image"""
        try:
            from PIL import Image
            import torchvision.transforms as transforms

            img = Image.open(image_path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            return transform(img)
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None

    def _load_audio(self, audio_path: str) -> Optional[torch.Tensor]:
        """Load and preprocess audio"""
        try:
            import librosa
            audio_array, sr = librosa.load(audio_path, sr=16000, duration=30)
            # Pad or truncate to fixed length
            target_length = 16000 * 30  # 30 seconds
            if len(audio_array) < target_length:
                audio_array = np.pad(audio_array, (0, target_length - len(audio_array)))
            else:
                audio_array = audio_array[:target_length]
            return torch.tensor(audio_array, dtype=torch.float32)
        except Exception as e:
            logger.warning(f"Failed to load audio {audio_path}: {e}")
            return None

    def __getitem__(self, idx):
        sample = self.samples[idx]
        batch = {}

        # Load text (required)
        text = sample.get('text', '') or sample.get('meta', {}).get('caption', '') or sample.get('meta', {}).get('transcript', '')
        if text:
            batch['input_ids'] = self._tokenize(text)

        # Load image if available
        if 'image' in self.modalities:
            if 'frames_dir' in sample:
                # Load first frame from video
                import glob
                frames = glob.glob(os.path.join(sample['frames_dir'], '*.jpg'))
                if frames:
                    image = self._load_image(frames[0])
                    if image is not None:
                        batch['images'] = image
            elif 'image_path' in sample:
                image = self._load_image(sample['image_path'])
                if image is not None:
                    batch['images'] = image

        # Load audio if available
        if 'audio' in self.modalities and 'audio' in sample:
            audio = self._load_audio(sample['audio'])
            if audio is not None:
                batch['audio'] = audio

        # Load video frames if available
        if 'video' in self.modalities and 'frames_dir' in sample:
            import glob
            frames = sorted(glob.glob(os.path.join(sample['frames_dir'], '*.jpg')))[:8]
            if frames:
                video_frames = []
                for frame_path in frames:
                    frame = self._load_image(frame_path)
                    if frame is not None:
                        video_frames.append(frame)
                if video_frames:
                    batch['video'] = torch.stack(video_frames)

        return batch


# =============================================================================
# Trainer Configuration
# =============================================================================

@dataclass
class PolymorphNetXTrainingConfig:
    """Configuration for PolymorphNet-X training"""
    manifest_path: str
    val_manifest_path: Optional[str] = None
    output_dir: str = "./polymorphnetx_output"

    # Model config
    d_model: int = 12288
    n_layers: int = 48
    n_heads: int = 64
    vocab_size: int = 131072
    max_seq_len: int = 8192
    n_experts: int = 64
    top_k: int = 4
    expert_ffn_dim: int = 65536

    # Training config
    num_epochs: int = 20
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    total_steps: int = 100000
    max_grad_norm: float = 1.0

    # Advanced features
    mixed_precision: str = "fp16"
    gradient_checkpointing: bool = True
    use_ema: bool = True
    ema_decay: float = 0.999

    # Logging and checkpointing
    logging_steps: int = 10
    save_steps: int = 1000
    eval_steps: int = 1000

    # Curriculum learning
    curriculum_learning: bool = True
    curriculum_stages: List[Tuple[int, int, List[str]]] = field(default_factory=lambda: [
        (0, 20000, ['text']),
        (20000, 50000, ['text', 'image']),
        (50000, 100000, ['text', 'image', 'audio', 'video'])
    ])

    # Tokenizer
    tokenizer_path: Optional[str] = None


# =============================================================================
# Trainer
# =============================================================================

@trainer_template(
    name="polymorphnetx-multimodal",
    description="PolymorphNet-X Multimodal Trainer for large-scale multimodal models",
    model_types=["multimodal", "polymorphnetx"],
    tags=["multimodal", "vision", "language", "audio", "video", "moe", "apotheon"],
    author="Apotheon.ai",
    version="1.0.0",
    requirements=["torch>=2.0.0", "transformers>=4.30.0", "accelerate>=0.20.0", "sentencepiece"]
)
class PolymorphNetXTrainer:
    """Trainer for PolymorphNet-X Multimodal Model"""

    def __init__(self, model: nn.Module, config: PolymorphNetXTrainingConfig):
        self.model = model
        self.config = config

        # Initialize Accelerator
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps
        )

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Get current modalities based on curriculum
        current_modalities = self._get_current_modalities(0)

        # Setup dataset and dataloader
        self.train_dataset = PolymorphNetXDataset(
            manifest_path=config.manifest_path,
            max_seq_len=config.max_seq_len,
            modalities=current_modalities,
            tokenizer_path=config.tokenizer_path,
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=self._collate_fn
        )

        # Learning rate scheduler
        self.lr_scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.total_steps
        )

        # Prepare for training
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        # EMA model
        if config.use_ema:
            self.ema_model = self._create_ema_model()
        else:
            self.ema_model = None

        # Global step counter
        self.global_step = 0

        logger.info(f"Initialized PolymorphNetX trainer with {len(self.train_dataset)} samples")

    def _get_current_modalities(self, step: int) -> List[str]:
        """Get current modalities based on curriculum stage"""
        if not self.config.curriculum_learning:
            return ['text', 'image', 'audio', 'video']

        for start, end, modalities in self.config.curriculum_stages:
            if start <= step < end:
                return modalities

        # Default to all modalities
        return ['text', 'image', 'audio', 'video']

    def _collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch items"""
        collated = {}

        # Collate input_ids
        if 'input_ids' in batch[0]:
            collated['input_ids'] = torch.stack([item['input_ids'] for item in batch if 'input_ids' in item])

        # Collate images
        if any('images' in item for item in batch):
            images = [item.get('images') for item in batch]
            if any(img is not None for img in images):
                # Pad with zeros for missing images
                collated['images'] = torch.stack([img if img is not None else torch.zeros(3, 224, 224) for img in images])

        # Collate audio
        if any('audio' in item for item in batch):
            audio = [item.get('audio') for item in batch]
            if any(aud is not None for aud in audio):
                collated['audio'] = torch.stack([aud if aud is not None else torch.zeros(16000 * 30) for aud in audio])

        # Collate video
        if any('video' in item for item in batch):
            video = [item.get('video') for item in batch]
            if any(vid is not None for vid in video):
                collated['video'] = torch.stack([vid if vid is not None else torch.zeros(8, 3, 224, 224) for vid in video])

        return collated

    def _create_ema_model(self):
        """Create EMA model"""
        ema_model = type(self.model)(grad_checkpoint=self.config.gradient_checkpointing)
        ema_model.load_state_dict(self.model.state_dict())
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model

    def _update_ema(self):
        """Update EMA model"""
        if self.ema_model is None:
            return

        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.config.ema_decay).add_(model_param.data, alpha=1 - self.config.ema_decay)

    def train(self):
        """Main training loop"""
        self.model.train()

        logger.info(f"Starting training for {self.config.total_steps} steps")

        epoch = 0
        progress_bar = tqdm(total=self.config.total_steps, desc="Training")

        while self.global_step < self.config.total_steps:
            logger.info(f"Starting epoch {epoch + 1}")

            for step, batch in enumerate(self.train_dataloader):
                if self.global_step >= self.config.total_steps:
                    break

                with self.accelerator.accumulate(self.model):
                    # Forward pass
                    outputs = self.model(batch)

                    # Compute loss
                    if isinstance(outputs, dict) and 'loss' in outputs:
                        loss = outputs['loss']
                    else:
                        # Compute cross-entropy loss on output logits
                        logits = outputs
                        labels = batch['input_ids'][:, 1:]  # Shift labels
                        logits = logits[:, :-1, :]  # Shift logits
                        loss = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            labels.reshape(-1),
                            ignore_index=TOKEN_IDS['PAD']
                        )

                    # Add MoE auxiliary loss if available
                    if hasattr(outputs, '_moe_aux'):
                        loss = loss + outputs._moe_aux

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    # Update EMA
                    if self.config.use_ema:
                        self._update_ema()

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    progress_bar.set_postfix({
                        'loss': loss.item(),
                        'lr': self.lr_scheduler.get_last_lr()[0],
                        'step': self.global_step
                    })

                # Save checkpoint
                if self.global_step > 0 and self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-step-{self.global_step}")

                self.global_step += 1
                progress_bar.update(1)

            epoch += 1

        # Save final checkpoint
        self.save_checkpoint("checkpoint-final")
        logger.info("Training completed!")

    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint"""
        output_dir = os.path.join(self.config.output_dir, checkpoint_name)
        os.makedirs(output_dir, exist_ok=True)

        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # Save model state dict
        torch.save(unwrapped_model.state_dict(), os.path.join(output_dir, "model.pt"))

        # Save EMA model if available
        if self.ema_model is not None:
            torch.save(self.ema_model.state_dict(), os.path.join(output_dir, "ema_model.pt"))

        # Save config
        config_dict = {
            'd_model': self.config.d_model,
            'n_layers': self.config.n_layers,
            'n_heads': self.config.n_heads,
            'vocab_size': self.config.vocab_size,
            'max_seq_len': self.config.max_seq_len,
            'n_experts': self.config.n_experts,
            'top_k': self.config.top_k,
            'expert_ffn_dim': self.config.expert_ffn_dim,
        }
        with open(os.path.join(output_dir, "config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Save training state
        training_state = {
            'global_step': self.global_step,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.lr_scheduler.state_dict(),
        }
        torch.save(training_state, os.path.join(output_dir, "training_state.pt"))

        logger.info(f"Saved checkpoint to {output_dir}")


# =============================================================================
# Model Factory
# =============================================================================

def create_polymorphnetx_model(config: Dict[str, Any]) -> nn.Module:
    """
    Create PolymorphNetX model from config

    Args:
        config: Model configuration dictionary

    Returns:
        PolymorphNetX model
    """
    if not POLYMORPHNETX_AVAILABLE:
        raise ImportError("PolymorphNetX model is not available. Please check the installation.")

    # Extract model components config
    model_config = config.get('model', {})
    components = model_config.get('components', {})
    core_config = components.get('core_model', {})

    # Update CONF with custom values if provided
    if core_config:
        CONF.d_model = core_config.get('d_model', CONF.d_model)
        CONF.n_layers = core_config.get('n_layers', CONF.n_layers)
        CONF.n_heads = core_config.get('n_heads', CONF.n_heads)
        CONF.vocab_size = core_config.get('vocab_size', CONF.vocab_size)
        CONF.max_seq_len = core_config.get('max_seq_len', CONF.max_seq_len)
        CONF.dropout = core_config.get('dropout', CONF.dropout)

        moe_config = components.get('moe', {})
        CONF.n_experts = moe_config.get('n_experts', CONF.n_experts)
        CONF.top_k = moe_config.get('top_k', CONF.top_k)
        CONF.expert_ffn_dim = moe_config.get('expert_ffn_dim', CONF.expert_ffn_dim)

    # Create multimodal model
    gradient_checkpointing = config.get('training', {}).get('gradient_checkpointing', True)
    model = PolymorphNetXMultimodal()

    logger.info(f"Created PolymorphNetX model with {sum(p.numel() for p in model.parameters())/1e9:.2f}B parameters")

    return model


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for PolymorphNetX trainer"""
    import yaml

    # Load template config
    template_path = os.path.join(
        os.path.dirname(__file__),
        '../models/templates.yaml'
    )

    with open(template_path, 'r') as f:
        templates = yaml.safe_load(f)

    polymorphnetx_config = templates['templates']['polymorphnetx-multimodal']

    # Create model
    model = create_polymorphnetx_model(polymorphnetx_config)

    # Training config
    training_config = PolymorphNetXTrainingConfig(
        manifest_path="datasets/multimodal/manifest.jsonl",
        output_dir="./polymorphnetx_output",
        tokenizer_path="models/polymorphnetx/configs/spm.model",
        total_steps=100000,
        batch_size=2
    )

    # Initialize trainer
    trainer = PolymorphNetXTrainer(model, training_config)

    # Start training
    trainer.train()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
