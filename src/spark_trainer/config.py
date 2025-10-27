"""
Pydantic configuration models for SparkTrainer.
Supports loading from YAML via --config config.yaml
"""
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, validator


class PreprocessConfig(BaseModel):
    """Configuration for video preprocessing and captioning."""

    # Input paths
    video_dir: Path = Field(..., description="Directory containing input videos")
    output_dir: Path = Field(default=Path("data/processed"), description="Output directory for processed data")

    # Video processing
    extract_frames: bool = Field(default=True, description="Extract frames from videos")
    frame_rate: Optional[float] = Field(default=1.0, description="Frame extraction rate (fps)")
    resolution: Optional[str] = Field(default="224x224", description="Target resolution (WxH)")
    max_frames: Optional[int] = Field(default=None, description="Maximum frames per video")

    # Audio processing
    extract_audio: bool = Field(default=True, description="Extract audio from videos")
    transcribe: bool = Field(default=False, description="Transcribe audio with Whisper")
    whisper_model: str = Field(default="base", description="Whisper model size")

    # Captioning
    generate_captions: bool = Field(default=True, description="Generate image/video captions")
    captioner_backend: Literal["blip", "blip2", "internvl", "qwen2-vl", "florence2"] = Field(
        default="blip2", description="Captioner backend to use"
    )
    captioner_model: Optional[str] = Field(default=None, description="Specific model ID for captioner")
    caption_batch_size: int = Field(default=8, description="Batch size for captioning")

    # Data layout
    deterministic_layout: bool = Field(default=True, description="Use hash-based deterministic directory layout")
    manifest_version: int = Field(default=1, description="Manifest schema version")

    # Processing options
    num_workers: int = Field(default=4, description="Number of parallel workers")
    recursive: bool = Field(default=True, description="Recursively process video_dir")

    @validator("resolution")
    def validate_resolution(cls, v):
        if v and "x" in v:
            parts = v.split("x")
            if len(parts) != 2:
                raise ValueError("Resolution must be in format WxH (e.g., 224x224)")
            try:
                int(parts[0]), int(parts[1])
            except ValueError:
                raise ValueError("Resolution dimensions must be integers")
        return v

    class Config:
        use_enum_values = True


class TrainConfig(BaseModel):
    """Configuration for model training."""

    # Model configuration
    model_name: str = Field(..., description="Model name or HuggingFace model ID")
    model_type: Literal["vision_language", "diffusion", "asr", "rft"] = Field(
        ..., description="Type of model to train"
    )
    pretrained: bool = Field(default=True, description="Use pretrained weights")

    # Data configuration
    manifest_path: Path = Field(..., description="Path to training manifest (JSONL)")
    val_manifest_path: Optional[Path] = Field(default=None, description="Path to validation manifest")
    batch_size: int = Field(default=8, description="Training batch size per device")
    num_workers: int = Field(default=4, description="DataLoader num_workers")

    # Training hyperparameters
    learning_rate: float = Field(default=1e-5, description="Learning rate")
    num_epochs: int = Field(default=10, description="Number of training epochs")
    warmup_steps: int = Field(default=500, description="Number of warmup steps")
    gradient_accumulation_steps: int = Field(default=1, description="Gradient accumulation steps")
    max_grad_norm: float = Field(default=1.0, description="Max gradient norm for clipping")

    # Optimization
    optimizer: Literal["adam", "adamw", "sgd"] = Field(default="adamw", description="Optimizer")
    weight_decay: float = Field(default=0.01, description="Weight decay")
    scheduler: Literal["linear", "cosine", "constant"] = Field(default="cosine", description="LR scheduler")

    # Distributed training
    use_accelerate: bool = Field(default=True, description="Use Accelerate for distributed training")
    use_deepspeed: bool = Field(default=False, description="Use DeepSpeed")
    deepspeed_config: Optional[Path] = Field(default=None, description="Path to DeepSpeed config JSON")
    mixed_precision: Literal["no", "fp16", "bf16"] = Field(default="bf16", description="Mixed precision training")

    # Logging and checkpointing
    output_dir: Path = Field(default=Path("runs/experiment"), description="Output directory for checkpoints")
    logging_steps: int = Field(default=10, description="Log every N steps")
    eval_steps: int = Field(default=500, description="Evaluate every N steps")
    save_steps: int = Field(default=1000, description="Save checkpoint every N steps")
    save_total_limit: int = Field(default=3, description="Maximum number of checkpoints to keep")

    # Sampling strategy (for video data)
    clip_sampling: Literal["random", "uniform", "center"] = Field(
        default="random", description="How to sample clips from videos"
    )
    clip_length: int = Field(default=16, description="Number of frames per clip")
    clip_stride: int = Field(default=1, description="Stride between frames in clip")

    # Resume training
    resume_from_checkpoint: Optional[Path] = Field(default=None, description="Path to checkpoint to resume from")

    # Seed
    seed: int = Field(default=42, description="Random seed for reproducibility")

    class Config:
        use_enum_values = True


class Config(BaseModel):
    """Root configuration model."""

    preprocess: Optional[PreprocessConfig] = Field(default=None, description="Preprocessing configuration")
    train: Optional[TrainConfig] = Field(default=None, description="Training configuration")

    # Global settings
    log_dir: Optional[Path] = Field(default=None, description="Global log directory override")
    gpu_validation: bool = Field(default=True, description="Validate GPU availability at startup")
    ffmpeg_validation: bool = Field(default=True, description="Validate FFmpeg installation at startup")

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            Config instance
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Path to output YAML file
        """
        with open(path, "w") as f:
            yaml.dump(self.dict(exclude_none=True), f, default_flow_style=False, sort_keys=False)


def load_config(path: Union[str, Path]) -> Config:
    """
    Load configuration from YAML file.

    Args:
        path: Path to YAML config file

    Returns:
        Config instance
    """
    return Config.from_yaml(path)
