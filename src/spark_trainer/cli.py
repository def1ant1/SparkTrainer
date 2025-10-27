"""
Command-line interface for SparkTrainer.
Routes to appropriate trainers and preprocessing tools.
"""
import argparse
import sys
from pathlib import Path

from .config import load_config
from .logger import setup_logger
from .utils.ffmpeg_utils import validate_ffmpeg_installation
from .utils.gpu_validation import log_gpu_info, validate_cuda


def preprocess_command(args):
    """Run preprocessing pipeline."""
    from .preprocess import run_preprocessing

    logger = setup_logger(run_name=args.run_name, log_dir=args.log_dir)
    logger.info("Starting preprocessing...")

    if args.config:
        config = load_config(args.config)
        preprocess_config = config.preprocess

        if preprocess_config is None:
            logger.error("No preprocessing configuration found in config file")
            sys.exit(1)

        # Validate FFmpeg if needed
        if config.ffmpeg_validation:
            if not validate_ffmpeg_installation():
                logger.error("FFmpeg validation failed")
                sys.exit(1)

    else:
        from .config import PreprocessConfig

        preprocess_config = PreprocessConfig(
            video_dir=Path(args.video_dir),
            output_dir=Path(args.output_dir) if args.output_dir else Path("data/processed"),
            extract_frames=args.extract_frames,
            frame_rate=args.frame_rate,
            extract_audio=args.extract_audio,
            transcribe=args.transcribe,
            generate_captions=args.generate_captions,
            captioner_backend=args.captioner_backend,
        )

    run_preprocessing(preprocess_config)
    logger.info("Preprocessing complete!")


def train_command(args):
    """Run training pipeline."""
    logger = setup_logger(run_name=args.run_name, log_dir=args.log_dir)
    logger.info("Starting training...")

    # Load config
    if args.config:
        config = load_config(args.config)
        train_config = config.train

        if train_config is None:
            logger.error("No training configuration found in config file")
            sys.exit(1)

        # Validate GPU and FFmpeg
        if config.gpu_validation:
            validate_cuda()
            log_gpu_info()

        if config.ffmpeg_validation:
            validate_ffmpeg_installation()

    else:
        from .config import TrainConfig

        if not args.model_name or not args.manifest_path:
            logger.error("--model-name and --manifest-path are required when not using --config")
            sys.exit(1)

        train_config = TrainConfig(
            model_name=args.model_name,
            model_type=args.model_type,
            manifest_path=Path(args.manifest_path),
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
        )

    # Route to appropriate trainer
    if train_config.model_type == "vision_language":
        from .trainers.vision_language import VisionLanguageTrainer

        trainer = VisionLanguageTrainer(train_config)
        trainer.train()

    elif train_config.model_type == "diffusion":
        from .trainers.diffusion import DiffusionTrainer

        trainer = DiffusionTrainer(train_config)
        trainer.train()

    elif train_config.model_type == "asr":
        logger.error("ASR trainer not yet implemented")
        sys.exit(1)

    elif train_config.model_type == "rft":
        logger.error("RFT trainer not yet implemented")
        sys.exit(1)

    else:
        logger.error(f"Unsupported model type: {train_config.model_type}")
        sys.exit(1)

    logger.info("Training complete!")


def validate_command(args):
    """Run validation checks."""
    logger = setup_logger()
    logger.info("Running validation checks...")

    # FFmpeg validation
    logger.info("Checking FFmpeg installation...")
    ffmpeg_ok = validate_ffmpeg_installation()

    # GPU validation
    logger.info("Checking CUDA availability...")
    cuda_ok = validate_cuda()
    if cuda_ok:
        log_gpu_info()

    # Manifest validation
    if args.manifest:
        from .utils.manifest import validate_manifest

        logger.info(f"Validating manifest: {args.manifest}")
        is_valid, errors = validate_manifest(args.manifest)

        if is_valid:
            logger.info("Manifest validation passed!")
        else:
            logger.error("Manifest validation failed:")
            for error in errors:
                logger.error(f"  - {error}")

    logger.info("Validation complete!")
    sys.exit(0 if (ffmpeg_ok and cuda_ok) else 1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SparkTrainer: Unified training framework for vision-language and diffusion models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess videos for training")
    preprocess_parser.add_argument("--config", type=str, help="Path to YAML config file")
    preprocess_parser.add_argument("--video-dir", type=str, help="Directory containing videos")
    preprocess_parser.add_argument("--output-dir", type=str, help="Output directory")
    preprocess_parser.add_argument("--extract-frames", action="store_true", default=True, help="Extract frames")
    preprocess_parser.add_argument("--frame-rate", type=float, default=1.0, help="Frame extraction rate (fps)")
    preprocess_parser.add_argument("--extract-audio", action="store_true", default=True, help="Extract audio")
    preprocess_parser.add_argument("--transcribe", action="store_true", help="Transcribe audio with Whisper")
    preprocess_parser.add_argument("--generate-captions", action="store_true", default=True, help="Generate captions")
    preprocess_parser.add_argument(
        "--captioner-backend",
        type=str,
        choices=["blip", "blip2", "internvl", "qwen2-vl", "florence2"],
        default="blip2",
        help="Captioner backend",
    )
    preprocess_parser.add_argument("--run-name", type=str, help="Run name for logging")
    preprocess_parser.add_argument("--log-dir", type=str, help="Log directory")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", type=str, help="Path to YAML config file")
    train_parser.add_argument("--model-name", type=str, help="Model name or HuggingFace model ID")
    train_parser.add_argument(
        "--model-type",
        type=str,
        choices=["vision_language", "diffusion", "asr", "rft"],
        default="vision_language",
        help="Model type",
    )
    train_parser.add_argument("--manifest-path", type=str, help="Path to training manifest")
    train_parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    train_parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    train_parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs")
    train_parser.add_argument("--run-name", type=str, help="Run name for logging")
    train_parser.add_argument("--log-dir", type=str, help="Log directory")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate system and data")
    validate_parser.add_argument("--manifest", type=str, help="Path to manifest to validate")

    args = parser.parse_args()

    if args.command == "preprocess":
        preprocess_command(args)
    elif args.command == "train":
        train_command(args)
    elif args.command == "validate":
        validate_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
