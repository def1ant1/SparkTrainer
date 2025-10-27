"""
Video preprocessing pipeline.
Extracts frames, audio, generates captions.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from tqdm import tqdm

from .captioning import create_captioner
from .config import PreprocessConfig
from .logger import get_logger
from .utils.ffmpeg_utils import extract_audio, extract_frames, validate_video
from .utils.hashing import create_deterministic_layout, get_video_id
from .utils.manifest import ManifestV1, save_manifest

logger = get_logger()


def find_videos(video_dir: Path, recursive: bool = True) -> List[Path]:
    """
    Find all video files in directory.

    Args:
        video_dir: Directory to search
        recursive: Whether to search recursively

    Returns:
        List of video file paths
    """
    extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"]

    if recursive:
        videos = []
        for ext in extensions:
            videos.extend(video_dir.rglob(f"*{ext}"))
    else:
        videos = []
        for ext in extensions:
            videos.extend(video_dir.glob(f"*{ext}"))

    return sorted(videos)


def process_video(video_path: Path, config: PreprocessConfig, captioner=None) -> ManifestV1:
    """
    Process a single video file.

    Args:
        video_path: Path to video file
        config: Preprocessing configuration
        captioner: Optional captioner instance

    Returns:
        ManifestV1 entry

    Raises:
        Exception: If processing fails
    """
    logger.info(f"Processing: {video_path.name}")

    # Validate video
    is_valid, error_msg = validate_video(video_path)
    if not is_valid:
        raise ValueError(f"Invalid video: {error_msg}")

    # Create deterministic layout
    if config.deterministic_layout:
        layout = create_deterministic_layout(
            video_path=video_path,
            output_base=config.output_dir,
            use_full_hash=False,
            hash_prefix_len=2,
        )
        video_id = layout["video_hash"]
        output_dir = layout["output_dir"]
        frames_dir = layout["frames_dir"]
        audio_path = layout["audio_path"]
    else:
        video_id = get_video_id(video_path, use_hash=False)
        output_dir = config.output_dir / video_id
        frames_dir = output_dir / "frames"
        audio_path = output_dir / "audio.wav"
        frames_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames
    if config.extract_frames:
        logger.info(f"  Extracting frames from {video_path.name}...")
        try:
            num_frames = extract_frames(
                video_path=video_path,
                output_dir=frames_dir,
                fps=config.frame_rate,
                resolution=config.resolution,
            )
            logger.info(f"  Extracted {num_frames} frames")
        except Exception as e:
            logger.error(f"  Failed to extract frames: {e}")
            raise

    # Extract audio
    audio_path_str = None
    if config.extract_audio:
        logger.info(f"  Extracting audio from {video_path.name}...")
        try:
            audio_extracted = extract_audio(
                video_path=video_path,
                output_path=audio_path,
            )
            if audio_extracted:
                audio_path_str = str(audio_path)
        except Exception as e:
            logger.warning(f"  Failed to extract audio: {e}")

    # Generate captions
    caption = None
    if config.generate_captions and captioner:
        logger.info(f"  Generating captions for {video_path.name}...")
        try:
            # Get sample frames for captioning
            frame_paths = sorted(frames_dir.glob("frame_*.jpg"))

            if frame_paths:
                # Caption first frame or a few frames
                if config.caption_batch_size > 1:
                    # Caption multiple frames
                    sample_frames = frame_paths[:: max(1, len(frame_paths) // config.caption_batch_size)]
                    sample_frames = sample_frames[: config.caption_batch_size]
                    captions = captioner.predict_batch(sample_frames)
                    # Combine captions
                    caption = " | ".join(captions)
                else:
                    # Caption single frame (middle frame)
                    middle_idx = len(frame_paths) // 2
                    caption = captioner.predict(frame_paths[middle_idx])

                logger.info(f"  Caption: {caption[:100]}...")
        except Exception as e:
            logger.warning(f"  Failed to generate captions: {e}")

    # Create metadata
    meta = {
        "source_path": str(video_path),
        "video_name": video_path.name,
    }

    if caption:
        meta["caption"] = caption

    # Transcribe audio
    if config.transcribe and audio_path_str:
        logger.info(f"  Transcribing audio from {video_path.name}...")
        try:
            import whisper

            model = whisper.load_model(config.whisper_model)
            result = model.transcribe(audio_path_str)
            meta["transcript"] = result["text"]
            logger.info(f"  Transcript: {result['text'][:100]}...")
        except Exception as e:
            logger.warning(f"  Failed to transcribe audio: {e}")

    # Create manifest entry
    entry = ManifestV1(
        id=video_id,
        frames_dir=str(frames_dir),
        audio=audio_path_str,
        meta=meta,
    )

    logger.info(f"  Completed: {video_path.name}")
    return entry


def run_preprocessing(config: PreprocessConfig):
    """
    Run the full preprocessing pipeline.

    Args:
        config: Preprocessing configuration
    """
    logger.info("Starting video preprocessing pipeline")
    logger.info(f"Video directory: {config.video_dir}")
    logger.info(f"Output directory: {config.output_dir}")

    # Find videos
    logger.info("Searching for videos...")
    videos = find_videos(config.video_dir, recursive=config.recursive)
    logger.info(f"Found {len(videos)} videos")

    if not videos:
        logger.warning("No videos found!")
        return

    # Initialize captioner if needed
    captioner = None
    if config.generate_captions:
        logger.info(f"Loading captioner: {config.captioner_backend}")
        captioner = create_captioner(
            backend=config.captioner_backend,
            model_name=config.captioner_model,
        )

    # Process videos in parallel
    manifest_entries = []
    failed_videos = []

    with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
        futures = {
            executor.submit(process_video, video, config, captioner): video for video in videos
        }

        with tqdm(total=len(videos), desc="Processing videos") as pbar:
            for future in as_completed(futures):
                video = futures[future]
                try:
                    entry = future.result()
                    manifest_entries.append(entry)
                except Exception as e:
                    logger.error(f"Failed to process {video.name}: {e}")
                    failed_videos.append(video)
                finally:
                    pbar.update(1)

    # Save manifest
    manifest_path = config.output_dir / "manifest_v1.jsonl"
    logger.info(f"Saving manifest to {manifest_path}")
    save_manifest(manifest_entries, manifest_path)

    # Summary
    logger.info("=" * 80)
    logger.info(f"Preprocessing complete!")
    logger.info(f"  Successfully processed: {len(manifest_entries)}/{len(videos)} videos")
    logger.info(f"  Failed: {len(failed_videos)} videos")
    logger.info(f"  Manifest saved: {manifest_path}")
    logger.info("=" * 80)

    if failed_videos:
        logger.warning("Failed videos:")
        for video in failed_videos:
            logger.warning(f"  - {video.name}")
