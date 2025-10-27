# SparkTrainer

Unified training framework for vision-language models, diffusion models, and video generation.

## Features

- **Vision-Language Models**: Train BLIP, BLIP-2, InternVL, Qwen2-VL, LLaVA
- **Diffusion Models**: Train Stable Video Diffusion, AnimateDiff, Wan2.2-Animate
- **Video Processing**: Extract frames, audio, generate captions
- **Multi-GPU Training**: Accelerate and DeepSpeed support
- **Structured Configuration**: Pydantic models with YAML config files
- **Deterministic Data Layout**: Hash-based organization for reproducibility
- **Manifest Schema v1**: JSONL format for scalable datasets

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (required for video processing)
./scripts/install_ffmpeg.sh

# Optional: Install DeepSpeed for multi-GPU training
pip install deepspeed

# Optional: Install Whisper for audio transcription
pip install openai-whisper
```

## Quick Start

### 1. Preprocess Videos

```bash
python -m src.spark_trainer preprocess \
  --video-dir /path/to/videos \
  --output-dir data/processed \
  --extract-frames \
  --frame-rate 1.0 \
  --generate-captions \
  --captioner-backend blip2
```

Or use a config file:

```bash
python -m src.spark_trainer preprocess --config configs/preprocess_example.yaml
```

This will:
- Extract frames from videos at 1 fps
- Generate captions using BLIP-2
- Create deterministic directory layout with hashing
- Generate `manifest_v1.jsonl` with metadata

### 2. Train a Model

```bash
python -m src.spark_trainer train \
  --config configs/train_vl_example.yaml
```

Or specify parameters directly:

```bash
python -m src.spark_trainer train \
  --model-name Salesforce/blip2-opt-2.7b \
  --model-type vision_language \
  --manifest-path data/processed/manifest_v1.jsonl \
  --batch-size 8 \
  --learning-rate 1e-5 \
  --num-epochs 10
```

### 3. Validate System

```bash
python -m src.spark_trainer validate --manifest data/processed/manifest_v1.jsonl
```

This checks:
- FFmpeg installation
- CUDA availability and GPU info
- Manifest integrity

## Architecture

### Directory Structure

```
src/spark_trainer/
├── __init__.py           # Package initialization
├── __main__.py           # CLI entry point
├── cli.py                # Command-line interface
├── config.py             # Pydantic configuration models
├── logger.py             # Central logging system
├── captioning.py         # Image/video captioning backends
├── data.py               # Dataset and DataLoader implementations
├── preprocess.py         # Video preprocessing pipeline
├── trainers/
│   ├── __init__.py
│   ├── vision_language.py  # VL model trainer
│   └── diffusion.py         # Diffusion model trainer
└── utils/
    ├── __init__.py
    ├── manifest.py          # Manifest schema v1
    ├── hashing.py           # Deterministic hashing
    ├── ffmpeg_utils.py      # FFmpeg validation and processing
    └── gpu_validation.py    # CUDA validation and logging
```

### Trainers by Task Family

- **Vision-Language (VL)**: BLIP, BLIP-2, InternVL, Qwen2-VL, LLaVA
- **Text-to-Image/Video (T2I/T2V)**: Stable Video Diffusion, AnimateDiff
- **ASR**: Not yet implemented
- **RFT**: Not yet implemented

## Configuration

SparkTrainer uses Pydantic models for structured configuration. You can load configs from YAML files using `--config`.

### Preprocessing Config

See `configs/preprocess_example.yaml` for a complete example.

Key options:
- `video_dir`: Input directory with videos
- `frame_rate`: Frames per second to extract
- `captioner_backend`: blip, blip2, internvl, qwen2-vl, florence2
- `deterministic_layout`: Use hash-based directory organization

### Training Config

See `configs/train_vl_example.yaml` and `configs/train_diffusion_example.yaml`.

Key options:
- `model_name`: HuggingFace model ID
- `model_type`: vision_language, diffusion, asr, rft
- `manifest_path`: Path to training manifest
- `use_accelerate`: Enable Accelerate for distributed training
- `use_deepspeed`: Enable DeepSpeed (requires deepspeed_config)
- `mixed_precision`: no, fp16, bf16

## Manifest Schema v1

JSONL format with one entry per video:

```json
{
  "id": "abc123...",
  "frames_dir": "/path/to/frames",
  "audio": "/path/to/audio.wav",
  "meta": {
    "caption": "A person walking on the beach",
    "transcript": "Audio transcription...",
    "duration": 10.5,
    "source_path": "/original/video.mp4"
  }
}
```

## Data Layout

Deterministic layout using SHA256 hashing:

```
data/processed/
├── ab/                        # First 2 chars of hash
│   └── abc123.../            # Full hash
│       ├── frames/
│       │   ├── frame_000001.jpg
│       │   ├── frame_000002.jpg
│       │   └── ...
│       └── audio.wav
└── manifest_v1.jsonl
```

## Captioner Backends

- **BLIP**: Salesforce/blip-image-captioning-base
- **BLIP-2**: Salesforce/blip2-opt-2.7b (recommended)
- **InternVL**: OpenGVLab/InternVL-Chat-V1-5
- **Qwen2-VL**: Qwen/Qwen2-VL-7B-Instruct
- **Florence-2**: microsoft/Florence-2-large

All captioners implement the `Captioner` interface with `predict(image_path)` method.

## Multi-GPU Training

### Accelerate

```yaml
train:
  use_accelerate: true
  mixed_precision: bf16
```

### DeepSpeed

```yaml
train:
  use_accelerate: true
  use_deepspeed: true
  deepspeed_config: configs/deepspeed_zero2.json
  mixed_precision: bf16
```

DeepSpeed configs:
- `configs/deepspeed_zero2.json`: ZeRO Stage 2
- `configs/deepspeed_zero3.json`: ZeRO Stage 3 with CPU offloading

## Logging

All logs are written to:
```
runs/<timestamp>/spark_trainer.log
```

Logs include:
- Training progress (loss, LR, step)
- GPU information (CUDA version, memory, utilization)
- FFmpeg validation
- Error messages and warnings

## Testing

Run tests:

```bash
pytest tests/
```

Test coverage includes:
- Hashing utilities
- Manifest read/write/validation
- CLI smoke tests

## Examples

### Preprocess with Whisper Transcription

```bash
python -m src.spark_trainer preprocess \
  --video-dir /data/videos \
  --output-dir data/processed \
  --extract-frames \
  --extract-audio \
  --transcribe \
  --whisper-model base \
  --generate-captions \
  --captioner-backend blip2
```

### Train Vision-Language Model on DGX

```bash
python -m src.spark_trainer train \
  --config configs/train_vl_example.yaml
```

With custom DeepSpeed config:

```yaml
train:
  use_accelerate: true
  use_deepspeed: true
  deepspeed_config: configs/deepspeed_zero3.json
  mixed_precision: bf16
  batch_size: 4
  gradient_accumulation_steps: 4
```

### Train Diffusion Model

```bash
python -m src.spark_trainer train \
  --config configs/train_diffusion_example.yaml
```

This uses:
- `VideoFrameDataset` with clip sampling
- Diffusion loss (noise prediction)
- UNet models from `diffusers`

## Requirements

- Python 3.8+
- PyTorch 2.0+
- FFmpeg (for video processing)
- CUDA (optional, for GPU training)

See `requirements.txt` for full dependencies.

## License

See repository root for license information.
