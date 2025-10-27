"""
Apotheon MultiModal Trainer

A unified multimodal model for text, image, video, and audio understanding.
Combines CLIP-ViT (vision), Whisper (audio), and Llama3 (text) encoders
with a cross-attention fusion transformer for multi-task learning.

Objectives:
- Image captioning
- Audio transcription
- Contrastive alignment (CLIP-style)
- Video understanding

@trainer_template: apotheon-multimodal
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import logging

# HuggingFace imports
try:
    from transformers import (
        CLIPVisionModel,
        CLIPProcessor,
        WhisperModel,
        WhisperProcessor,
        AutoTokenizer,
        AutoModelForCausalLM,
        get_scheduler
    )
    from accelerate import Accelerator
except ImportError as e:
    print(f"Warning: Some dependencies missing for Apotheon trainer: {e}")

logger = logging.getLogger(__name__)


# =============================================================================
# Model Architecture Components
# =============================================================================

class CrossAttentionFusionLayer(nn.Module):
    """Cross-attention layer for fusing multimodal features"""

    def __init__(self, hidden_size: int, num_heads: int = 16, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        # Multi-head attention
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Layer norm and feedforward
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query_features, key_value_features, attention_mask=None):
        """
        Args:
            query_features: [batch, seq_len_q, hidden_size]
            key_value_features: [batch, seq_len_kv, hidden_size]
            attention_mask: [batch, seq_len_kv]
        """
        batch_size = query_features.size(0)

        # Multi-head attention
        Q = self.query(query_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key_value_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(key_value_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        output = self.out_proj(context)

        # Residual connection and layer norm
        query_features = self.layer_norm1(query_features + self.dropout(output))

        # Feedforward
        ff_output = self.feedforward(query_features)
        output = self.layer_norm2(query_features + ff_output)

        return output


class FusionTransformer(nn.Module):
    """Transformer for fusing multimodal representations"""

    def __init__(self, hidden_size: int = 2048, num_layers: int = 6, num_heads: int = 16, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionFusionLayer(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, query_features, key_value_features, attention_mask=None):
        """
        Fuse query features with key-value features through multiple cross-attention layers
        """
        output = query_features
        for layer in self.layers:
            output = layer(output, key_value_features, attention_mask)
        return output


class ApotheonMultiModalModel(nn.Module):
    """
    Unified multimodal model combining vision, audio, and text encoders
    with a fusion transformer for multi-task learning
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        # Extract component configs
        vision_config = config['model']['components']['vision_encoder']
        audio_config = config['model']['components']['audio_encoder']
        language_config = config['model']['components']['language_model']
        fusion_config = config['model']['components']['fusion']
        projection_configs = config['model']['components']['projections']

        # Vision Encoder (CLIP-ViT)
        logger.info(f"Loading vision encoder: {vision_config['model']}")
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_config['model'])
        self.vision_processor = CLIPProcessor.from_pretrained(vision_config['model'])
        if vision_config.get('freeze', False):
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        # Audio Encoder (Whisper)
        logger.info(f"Loading audio encoder: {audio_config['model']}")
        self.audio_encoder = WhisperModel.from_pretrained(audio_config['model'])
        self.audio_processor = WhisperProcessor.from_pretrained(audio_config['model'])
        if audio_config.get('freeze', False):
            for param in self.audio_encoder.parameters():
                param.requires_grad = False

        # Language Model (Llama3)
        logger.info(f"Loading language model: {language_config['model']}")
        self.language_model = AutoModelForCausalLM.from_pretrained(
            language_config['model'],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(language_config['model'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if language_config.get('freeze', False):
            for param in self.language_model.parameters():
                param.requires_grad = False

        # Projection layers
        self.vision_projection = nn.Linear(
            projection_configs['vision_projection']['input_dim'],
            projection_configs['vision_projection']['output_dim']
        )
        self.audio_projection = nn.Linear(
            projection_configs['audio_projection']['input_dim'],
            projection_configs['audio_projection']['output_dim']
        )
        self.text_projection = nn.Linear(
            projection_configs['text_projection']['input_dim'],
            projection_configs['text_projection']['output_dim']
        )

        # Fusion Transformer
        self.fusion_transformer = FusionTransformer(
            hidden_size=fusion_config['hidden_size'],
            num_layers=fusion_config['num_layers'],
            num_heads=fusion_config['num_heads'],
            dropout=fusion_config['dropout']
        )

        # Task-specific heads
        fusion_hidden_size = fusion_config['hidden_size']
        self.caption_head = nn.Linear(fusion_hidden_size, self.tokenizer.vocab_size)
        self.transcription_head = nn.Linear(fusion_hidden_size, self.audio_processor.tokenizer.vocab_size)

        # Contrastive learning temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * config['model']['objectives'][2]['temperature'])

    def encode_vision(self, pixel_values):
        """Encode images using CLIP-ViT"""
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        vision_features = vision_outputs.pooler_output  # [batch, hidden_size]
        vision_features = self.vision_projection(vision_features)  # [batch, fusion_hidden_size]
        return vision_features.unsqueeze(1)  # [batch, 1, fusion_hidden_size]

    def encode_audio(self, input_features):
        """Encode audio using Whisper"""
        audio_outputs = self.audio_encoder.encoder(input_features=input_features)
        audio_features = audio_outputs.last_hidden_state.mean(dim=1)  # [batch, hidden_size]
        audio_features = self.audio_projection(audio_features)  # [batch, fusion_hidden_size]
        return audio_features.unsqueeze(1)  # [batch, 1, fusion_hidden_size]

    def encode_text(self, input_ids, attention_mask):
        """Encode text using Llama3"""
        text_outputs = self.language_model.model(input_ids=input_ids, attention_mask=attention_mask)
        # Use last hidden state mean pooling
        text_features = text_outputs.last_hidden_state.mean(dim=1)  # [batch, hidden_size]
        text_features = self.text_projection(text_features)  # [batch, fusion_hidden_size]
        return text_features.unsqueeze(1)  # [batch, 1, fusion_hidden_size]

    def forward(self, batch):
        """
        Forward pass for multi-task training

        Args:
            batch: Dict containing:
                - pixel_values: Image tensors [batch, 3, 224, 224]
                - input_features: Audio features [batch, n_mels, seq_len]
                - input_ids: Text token IDs [batch, seq_len]
                - attention_mask: Text attention mask [batch, seq_len]
                - caption_labels: Caption token IDs [batch, seq_len] (optional)
                - transcription_labels: Transcription token IDs [batch, seq_len] (optional)
        """
        outputs = {}
        losses = {}

        # Encode modalities
        multimodal_features = []

        if 'pixel_values' in batch and batch['pixel_values'] is not None:
            vision_features = self.encode_vision(batch['pixel_values'])
            multimodal_features.append(vision_features)
            outputs['vision_features'] = vision_features

        if 'input_features' in batch and batch['input_features'] is not None:
            audio_features = self.encode_audio(batch['input_features'])
            multimodal_features.append(audio_features)
            outputs['audio_features'] = audio_features

        if 'input_ids' in batch and batch['input_ids'] is not None:
            text_features = self.encode_text(batch['input_ids'], batch['attention_mask'])
            multimodal_features.append(text_features)
            outputs['text_features'] = text_features

        # Concatenate all available modalities
        if len(multimodal_features) > 0:
            fused_features = torch.cat(multimodal_features, dim=1)  # [batch, num_modalities, hidden_size]
            outputs['fused_features'] = fused_features

        # Task 1: Image Captioning
        if 'caption_labels' in batch and 'vision_features' in outputs:
            caption_logits = self.caption_head(outputs['vision_features'])
            caption_loss = F.cross_entropy(
                caption_logits.view(-1, caption_logits.size(-1)),
                batch['caption_labels'].view(-1),
                ignore_index=-100
            )
            losses['captioning'] = caption_loss
            outputs['caption_logits'] = caption_logits

        # Task 2: Audio Transcription
        if 'transcription_labels' in batch and 'audio_features' in outputs:
            transcription_logits = self.transcription_head(outputs['audio_features'])
            transcription_loss = F.cross_entropy(
                transcription_logits.view(-1, transcription_logits.size(-1)),
                batch['transcription_labels'].view(-1),
                ignore_index=-100
            )
            losses['transcription'] = transcription_loss
            outputs['transcription_logits'] = transcription_logits

        # Task 3: Contrastive Alignment (CLIP-style)
        if 'vision_features' in outputs and 'text_features' in outputs:
            vision_embeds = F.normalize(outputs['vision_features'].squeeze(1), dim=-1)
            text_embeds = F.normalize(outputs['text_features'].squeeze(1), dim=-1)

            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * vision_embeds @ text_embeds.t()
            logits_per_text = logits_per_image.t()

            batch_size = vision_embeds.size(0)
            labels = torch.arange(batch_size, device=vision_embeds.device)

            contrastive_loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
            ) / 2

            losses['contrastive'] = contrastive_loss
            outputs['contrastive_logits'] = logits_per_image

        # Compute weighted loss
        total_loss = 0.0
        objective_weights = {obj['name']: obj['weight'] for obj in self.config['model']['objectives']}

        for task_name, loss_value in losses.items():
            weight = objective_weights.get(task_name.replace('_', '_'), 1.0)
            total_loss += weight * loss_value

        outputs['loss'] = total_loss
        outputs['losses'] = losses

        return outputs


# =============================================================================
# Dataset
# =============================================================================

class ApotheonMultiModalDataset(Dataset):
    """Dataset for Apotheon multimodal training"""

    def __init__(
        self,
        manifest_path: str,
        vision_processor,
        audio_processor,
        tokenizer,
        max_length: int = 512,
        modalities: List[str] = ['text', 'image', 'audio']
    ):
        self.manifest_path = manifest_path
        self.vision_processor = vision_processor
        self.audio_processor = audio_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.modalities = modalities

        # Load manifest
        self.samples = []
        with open(manifest_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line))

        logger.info(f"Loaded {len(self.samples)} samples from {manifest_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        batch = {}

        # Load and process image
        if 'image' in self.modalities and 'frames_dir' in sample:
            # Load first frame as representative image
            import glob
            from PIL import Image
            frames = glob.glob(os.path.join(sample['frames_dir'], '*.jpg'))
            if frames:
                image = Image.open(frames[0]).convert('RGB')
                pixel_values = self.vision_processor(images=image, return_tensors='pt')['pixel_values'][0]
                batch['pixel_values'] = pixel_values

                # Caption as label
                if 'meta' in sample and 'caption' in sample['meta']:
                    caption = sample['meta']['caption']
                    caption_encoding = self.tokenizer(
                        caption,
                        max_length=self.max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    batch['caption_labels'] = caption_encoding['input_ids'][0]

        # Load and process audio
        if 'audio' in self.modalities and 'audio' in sample:
            import librosa
            audio_path = sample['audio']
            if os.path.exists(audio_path):
                audio_array, sr = librosa.load(audio_path, sr=16000)
                input_features = self.audio_processor(
                    audio_array,
                    sampling_rate=16000,
                    return_tensors='pt'
                )['input_features'][0]
                batch['input_features'] = input_features

                # Transcription as label
                if 'meta' in sample and 'transcript' in sample['meta']:
                    transcript = sample['meta']['transcript']
                    transcript_encoding = self.audio_processor.tokenizer(
                        transcript,
                        max_length=self.max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    batch['transcription_labels'] = transcript_encoding['input_ids'][0]

        # Process text
        if 'text' in self.modalities:
            text = sample.get('meta', {}).get('caption', '') or sample.get('meta', {}).get('transcript', '')
            if text:
                text_encoding = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                batch['input_ids'] = text_encoding['input_ids'][0]
                batch['attention_mask'] = text_encoding['attention_mask'][0]

        return batch


# =============================================================================
# Trainer
# =============================================================================

@dataclass
class ApotheonTrainingConfig:
    """Configuration for Apotheon training"""
    manifest_path: str
    val_manifest_path: Optional[str] = None
    output_dir: str = "./apotheon_output"
    num_epochs: int = 10
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    mixed_precision: str = "bf16"
    curriculum_learning: bool = True


class ApotheonTrainer:
    """Trainer for Apotheon MultiModal Model"""

    def __init__(self, model: ApotheonMultiModalModel, config: ApotheonTrainingConfig):
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

        # Setup dataset and dataloader
        self.train_dataset = ApotheonMultiModalDataset(
            manifest_path=config.manifest_path,
            vision_processor=model.vision_processor,
            audio_processor=model.audio_processor,
            tokenizer=model.tokenizer
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        # Learning rate scheduler
        num_training_steps = len(self.train_dataloader) * config.num_epochs
        num_warmup_steps = int(num_training_steps * config.warmup_ratio)

        self.lr_scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        # Prepare for training
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        logger.info(f"Initialized Apotheon trainer with {len(self.train_dataset)} samples")

    def train(self):
        """Main training loop"""
        self.model.train()
        global_step = 0

        for epoch in range(self.config.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")

            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")

            for step, batch in enumerate(progress_bar):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(batch)
                    loss = outputs['loss']

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Logging
                if step % self.config.logging_steps == 0:
                    loss_dict = {k: v.item() for k, v in outputs['losses'].items()}
                    progress_bar.set_postfix({
                        'loss': loss.item(),
                        **loss_dict,
                        'lr': self.lr_scheduler.get_last_lr()[0]
                    })

                # Save checkpoint
                if global_step > 0 and global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{global_step}")

                global_step += 1

            # Save epoch checkpoint
            self.save_checkpoint(f"checkpoint-epoch-{epoch + 1}")

        logger.info("Training completed!")

    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint"""
        output_dir = os.path.join(self.config.output_dir, checkpoint_name)
        os.makedirs(output_dir, exist_ok=True)

        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # Save model
        torch.save(unwrapped_model.state_dict(), os.path.join(output_dir, "model.pt"))

        # Save config
        with open(os.path.join(output_dir, "config.json"), 'w') as f:
            json.dump(self.model.config, f, indent=2)

        logger.info(f"Saved checkpoint to {output_dir}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for Apotheon trainer"""
    import yaml

    # Load template config
    template_path = os.path.join(
        os.path.dirname(__file__),
        '../models/templates.yaml'
    )

    with open(template_path, 'r') as f:
        templates = yaml.safe_load(f)

    apotheon_config = templates['templates']['apotheon-multimodal']

    # Initialize model
    model = ApotheonMultiModalModel(apotheon_config)

    # Training config
    training_config = ApotheonTrainingConfig(
        manifest_path="datasets/multimodal/manifest.jsonl",
        output_dir="./apotheon_output",
        num_epochs=10,
        batch_size=4
    )

    # Initialize trainer
    trainer = ApotheonTrainer(model, training_config)

    # Start training
    trainer.train()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
