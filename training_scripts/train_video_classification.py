"""
Video Classification Training Script using VideoMAE / TimeSformer / ViViT

Supports:
- VideoMAE (Meta)
- TimeSformer
- ViViT
- Custom video transformers

Usage:
    python train_video_classification.py \\
        --dataset_name my_videos \\
        --model_name facebook/videomae-base \\
        --num_frames 16 \\
        --batch_size 4 \\
        --epochs 10
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

try:
    from transformers import (
        VideoMAEForVideoClassification,
        VideoMAEImageProcessor,
        TimesformerForVideoClassification,
        AutoImageProcessor,
        get_scheduler
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import decord
    decord.bridge.set_bridge('torch')
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


class VideoDataset(Dataset):
    """Dataset for video classification from manifest"""

    def __init__(self, manifest_path, processor, num_frames=16, split='train', train_ratio=0.8):
        self.manifest_path = manifest_path
        self.processor = processor
        self.num_frames = num_frames
        self.split = split

        # Load manifest
        self.videos = []
        self.labels = []
        label_set = set()

        with open(manifest_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                if 'metadata_error' in record:
                    continue

                self.videos.append(record)
                label = record.get('label', 'unknown')
                label_set.add(label)

        # Create label mapping
        self.label2id = {label: idx for idx, label in enumerate(sorted(label_set))}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

        # Split train/val
        np.random.seed(42)
        indices = np.random.permutation(len(self.videos))
        split_point = int(len(indices) * train_ratio)

        if split == 'train':
            self.indices = indices[:split_point]
        else:
            self.indices = indices[split_point:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        video_record = self.videos[self.indices[idx]]
        video_path = video_record['path']
        label = video_record.get('label', 'unknown')
        label_id = self.label2id[label]

        # Load video frames
        frames = self.load_video(video_path)

        # Process frames
        if self.processor is not None:
            inputs = self.processor(list(frames), return_tensors="pt")
            pixel_values = inputs['pixel_values'][0]  # Remove batch dim
        else:
            pixel_values = frames

        return {
            'pixel_values': pixel_values,
            'labels': torch.tensor(label_id, dtype=torch.long)
        }

    def load_video(self, video_path):
        """Load video and sample frames"""
        if not HAS_DECORD:
            raise ImportError("decord not installed. Run: pip install decord")

        try:
            vr = decord.VideoReader(video_path)
            total_frames = len(vr)

            # Sample frames uniformly
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            frames = vr.get_batch(indices).asnumpy()

            return frames

        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            # Return dummy frames on error
            return np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)


def train_epoch(model, dataloader, optimizer, scheduler, device, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(dataloader):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Mixed precision training
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(pixel_values, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

        scheduler.step()

        # Metrics
        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, "
                  f"Acc: {correct/max(1, total):.4f}")

    return {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / max(1, total)
    }


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(pixel_values, labels=labels)
        loss = outputs.loss

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / max(1, total)
    }


def main():
    parser = argparse.ArgumentParser(description='Train video classification model')

    # Data
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name')
    parser.add_argument('--dataset_version', type=str, default='', help='Dataset version')
    parser.add_argument('--manifest', type=str, default='manifest.jsonl', help='Manifest filename')

    # Model
    parser.add_argument('--model_name', type=str, default='facebook/videomae-base',
                        help='Model name or path (HuggingFace)')
    parser.add_argument('--num_frames', type=int, default=16, help='Number of frames per video')

    # Training
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=100, help='Warmup steps')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')

    # Output
    parser.add_argument('--job_id', type=str, default=None, help='Job ID for output')
    parser.add_argument('--name', type=str, default='video_classifier', help='Model name')

    args = parser.parse_args()

    # Check dependencies
    if not HAS_TRANSFORMERS:
        print("ERROR: transformers not installed. Run: pip install transformers")
        sys.exit(1)

    if not HAS_DECORD:
        print("ERROR: decord not installed. Run: pip install decord")
        sys.exit(1)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    if args.dataset_version:
        dataset_dir = os.path.join(DATASETS_DIR, args.dataset_name, args.dataset_version)
    else:
        # Find latest version
        dataset_base = os.path.join(DATASETS_DIR, args.dataset_name)
        versions = [d for d in os.listdir(dataset_base) if os.path.isdir(os.path.join(dataset_base, d))]
        versions.sort(reverse=True)
        dataset_dir = os.path.join(dataset_base, versions[0] if versions else '')

    manifest_path = os.path.join(dataset_dir, args.manifest)

    if not os.path.exists(manifest_path):
        print(f"ERROR: Manifest not found at {manifest_path}")
        print("Run video indexing first: POST /api/datasets/index")
        sys.exit(1)

    print(f"Loading dataset from: {manifest_path}")

    # Load processor
    print(f"Loading processor for {args.model_name}...")
    try:
        processor = AutoImageProcessor.from_pretrained(args.model_name)
    except Exception as e:
        print(f"Warning: Could not load processor: {e}")
        print("Using default VideoMAE processor...")
        processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")

    # Create datasets
    train_dataset = VideoDataset(
        manifest_path,
        processor,
        num_frames=args.num_frames,
        split='train'
    )

    val_dataset = VideoDataset(
        manifest_path,
        processor,
        num_frames=args.num_frames,
        split='val'
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.label2id)}")
    print(f"Classes: {train_dataset.id2label}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Load model
    print(f"Loading model: {args.model_name}...")
    num_labels = len(train_dataset.label2id)

    try:
        if 'videomae' in args.model_name.lower():
            model = VideoMAEForVideoClassification.from_pretrained(
                args.model_name,
                num_labels=num_labels,
                label2id=train_dataset.label2id,
                id2label=train_dataset.id2label,
                ignore_mismatched_sizes=True
            )
        elif 'timesformer' in args.model_name.lower():
            model = TimesformerForVideoClassification.from_pretrained(
                args.model_name,
                num_labels=num_labels,
                label2id=train_dataset.label2id,
                id2label=train_dataset.id2label,
                ignore_mismatched_sizes=True
            )
        else:
            print(f"Unknown model type: {args.model_name}")
            print("Trying VideoMAE...")
            model = VideoMAEForVideoClassification.from_pretrained(
                args.model_name,
                num_labels=num_labels,
                label2id=train_dataset.label2id,
                id2label=train_dataset.id2label,
                ignore_mismatched_sizes=True
            )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    model = model.to(device)

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    num_training_steps = args.epochs * len(train_loader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None

    # Training loop
    best_val_acc = 0.0
    job_id = args.job_id or datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = os.path.join(MODELS_DIR, job_id)
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(LOGS_DIR, f"{job_id}.log")

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Output directory: {output_dir}")

    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, scaler)
        print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")

        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")

        # Log
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_accuracy': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'timestamp': datetime.now().isoformat()
        }

        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            print(f"New best model! Val Acc: {best_val_acc:.4f}")

            # Save model
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

            # Save config
            config = {
                'name': args.name,
                'framework': 'huggingface',
                'base_model': args.model_name,
                'task_type': 'video_classification',
                'num_labels': num_labels,
                'label2id': train_dataset.label2id,
                'id2label': train_dataset.id2label,
                'num_frames': args.num_frames,
                'best_val_accuracy': best_val_acc,
                'best_val_loss': val_metrics['loss'],
                'created': datetime.now().isoformat(),
                'epochs_trained': epoch + 1,
                'total_epochs': args.epochs
            }

            with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
                json.dump(config, f, indent=2)

    print(f"\nTraining completed!")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {output_dir}")


if __name__ == '__main__':
    main()
