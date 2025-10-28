"""
Video ingestion wizard - comprehensive video processing with extraction,
transcription, captioning, and manifest generation.
"""
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import subprocess

import cv2
import av
import numpy as np
from tqdm import tqdm

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: Whisper not available. Audio transcription will be disabled.")


@dataclass
class VideoIngestionConfig:
    """Configuration for video ingestion pipeline."""
    # Input/Output
    input_path: str
    output_dir: str

    # Frame extraction
    fps: int = 1
    resolution: Tuple[int, int] = (224, 224)
    max_frames: Optional[int] = None
    keyframe_only: bool = False

    # Audio extraction
    extract_audio: bool = True
    audio_sample_rate: int = 16000

    # Transcription
    enable_transcription: bool = True
    whisper_model: str = "base"  # tiny, base, small, medium, large
    transcription_language: Optional[str] = None

    # Captioning
    enable_captioning: bool = True
    captioner_backend: str = "blip2"  # blip2, internvl, qwen2vl, florence2

    # Scene detection
    enable_scene_detection: bool = True
    scene_threshold: float = 27.0

    # Quality control
    min_video_duration: float = 1.0  # seconds
    max_video_duration: Optional[float] = None
    check_integrity: bool = True

    # Output
    generate_manifest: bool = True
    save_thumbnails: bool = True
    thumbnail_size: Tuple[int, int] = (160, 120)


@dataclass
class VideoMetadata:
    """Metadata extracted from video file."""
    file_path: str
    file_size: int
    duration: float
    fps: float
    width: int
    height: int
    num_frames: int
    codec: str
    has_audio: bool
    audio_codec: Optional[str] = None
    audio_sample_rate: Optional[int] = None
    checksum: Optional[str] = None


@dataclass
class IngestionResult:
    """Result of video ingestion."""
    video_id: str
    metadata: VideoMetadata
    frames_extracted: int
    audio_path: Optional[str] = None
    transcript: Optional[Dict[str, Any]] = None
    captions: Optional[List[Dict[str, str]]] = None
    scenes: Optional[List[Dict[str, Any]]] = None
    thumbnail_path: Optional[str] = None
    manifest_entry: Optional[Dict[str, Any]] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class VideoWizard:
    """
    Comprehensive video ingestion wizard.

    Handles:
    - Video validation and metadata extraction
    - Frame extraction with various strategies
    - Audio extraction and transcription (Whisper)
    - Image captioning
    - Scene detection
    - Manifest generation
    """

    def __init__(self, config: VideoIngestionConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize captioner
        self.captioner = None
        if config.enable_captioning:
            self._init_captioner()

        # Initialize Whisper
        self.whisper_model = None
        if config.enable_transcription and WHISPER_AVAILABLE:
            print(f"Loading Whisper model: {config.whisper_model}")
            self.whisper_model = whisper.load_model(config.whisper_model)

    def _init_captioner(self):
        """Initialize image captioning model."""
        from spark_trainer.captioning import get_captioner
        self.captioner = get_captioner(self.config.captioner_backend)

    def process_video(self, video_path: str) -> IngestionResult:
        """
        Process a single video file through the full ingestion pipeline.

        Args:
            video_path: Path to video file

        Returns:
            IngestionResult with all extracted data
        """
        video_path = Path(video_path)
        video_id = self._generate_video_id(video_path)

        print(f"\nProcessing video: {video_path.name}")
        print(f"Video ID: {video_id}")

        result = IngestionResult(
            video_id=video_id,
            metadata=None,
            frames_extracted=0,
            errors=[]
        )

        try:
            # 1. Extract metadata and validate
            print("Extracting metadata...")
            metadata = self._extract_metadata(video_path)
            result.metadata = metadata

            if not self._validate_video(metadata):
                result.errors.append("Video validation failed")
                return result

            # 2. Create output directories
            video_output_dir = self.output_dir / video_id
            frames_dir = video_output_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)

            # 3. Extract frames
            print(f"Extracting frames (fps={self.config.fps})...")
            frames = self._extract_frames(video_path, frames_dir)
            result.frames_extracted = len(frames)

            # 4. Generate thumbnail
            if self.config.save_thumbnails and frames:
                print("Generating thumbnail...")
                result.thumbnail_path = self._generate_thumbnail(
                    frames[0], video_output_dir
                )

            # 5. Extract and transcribe audio
            if self.config.extract_audio and metadata.has_audio:
                print("Extracting audio...")
                audio_path = self._extract_audio(video_path, video_output_dir)
                result.audio_path = str(audio_path)

                if self.config.enable_transcription and self.whisper_model:
                    print("Transcribing audio...")
                    transcript = self._transcribe_audio(audio_path)
                    result.transcript = transcript

            # 6. Caption frames
            if self.config.enable_captioning and self.captioner and frames:
                print(f"Captioning {len(frames)} frames...")
                captions = self._caption_frames(frames)
                result.captions = captions

            # 7. Detect scenes
            if self.config.enable_scene_detection:
                print("Detecting scenes...")
                scenes = self._detect_scenes(video_path)
                result.scenes = scenes

            # 8. Generate manifest entry
            if self.config.generate_manifest:
                manifest_entry = self._create_manifest_entry(result)
                result.manifest_entry = manifest_entry

            print(f"✓ Successfully processed {video_path.name}")
            print(f"  - Frames extracted: {result.frames_extracted}")
            if result.transcript:
                print(f"  - Transcript: {len(result.transcript.get('text', ''))} characters")
            if result.captions:
                print(f"  - Captions: {len(result.captions)}")

        except Exception as e:
            error_msg = f"Error processing video: {str(e)}"
            print(f"✗ {error_msg}")
            result.errors.append(error_msg)

        return result

    def process_batch(self, video_paths: List[str]) -> List[IngestionResult]:
        """
        Process multiple videos.

        Args:
            video_paths: List of video file paths

        Returns:
            List of IngestionResult
        """
        results = []

        for video_path in tqdm(video_paths, desc="Processing videos"):
            result = self.process_video(video_path)
            results.append(result)

        # Generate batch manifest
        if self.config.generate_manifest:
            self._write_manifest(results)

        return results

    def process_directory(self, directory: str, recursive: bool = True) -> List[IngestionResult]:
        """
        Process all videos in a directory.

        Args:
            directory: Directory path
            recursive: Whether to search recursively

        Returns:
            List of IngestionResult
        """
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
        directory = Path(directory)

        # Find all video files
        if recursive:
            video_files = [
                str(f) for f in directory.rglob("*")
                if f.suffix.lower() in video_extensions
            ]
        else:
            video_files = [
                str(f) for f in directory.glob("*")
                if f.suffix.lower() in video_extensions
            ]

        print(f"Found {len(video_files)} video files in {directory}")

        return self.process_batch(video_files)

    def _generate_video_id(self, video_path: Path) -> str:
        """Generate unique video ID from file path and content."""
        # Use file path and size for ID
        file_stat = video_path.stat()
        id_string = f"{video_path.name}_{file_stat.st_size}_{file_stat.st_mtime}"
        return hashlib.sha256(id_string.encode()).hexdigest()[:16]

    def _extract_metadata(self, video_path: Path) -> VideoMetadata:
        """Extract comprehensive video metadata."""
        container = av.open(str(video_path))
        video_stream = container.streams.video[0]
        audio_streams = container.streams.audio

        # Calculate checksum if integrity check is enabled
        checksum = None
        if self.config.check_integrity:
            checksum = self._calculate_checksum(video_path)

        metadata = VideoMetadata(
            file_path=str(video_path),
            file_size=video_path.stat().st_size,
            duration=float(video_stream.duration * video_stream.time_base),
            fps=float(video_stream.average_rate),
            width=video_stream.width,
            height=video_stream.height,
            num_frames=video_stream.frames,
            codec=video_stream.codec_context.name,
            has_audio=len(audio_streams) > 0,
            checksum=checksum
        )

        if audio_streams:
            audio_stream = audio_streams[0]
            metadata.audio_codec = audio_stream.codec_context.name
            metadata.audio_sample_rate = audio_stream.sample_rate

        container.close()
        return metadata

    def _validate_video(self, metadata: VideoMetadata) -> bool:
        """Validate video meets requirements."""
        if metadata.duration < self.config.min_video_duration:
            print(f"Video too short: {metadata.duration}s < {self.config.min_video_duration}s")
            return False

        if self.config.max_video_duration and metadata.duration > self.config.max_video_duration:
            print(f"Video too long: {metadata.duration}s > {self.config.max_video_duration}s")
            return False

        return True

    def _extract_frames(self, video_path: Path, output_dir: Path) -> List[str]:
        """Extract frames from video."""
        container = av.open(str(video_path))
        video_stream = container.streams.video[0]

        # Calculate frame sampling interval
        source_fps = float(video_stream.average_rate)
        frame_interval = int(source_fps / self.config.fps) if self.config.fps < source_fps else 1

        frames = []
        frame_count = 0

        for frame_idx, frame in enumerate(container.decode(video_stream)):
            if frame_idx % frame_interval != 0:
                continue

            if self.config.max_frames and frame_count >= self.config.max_frames:
                break

            # Convert to numpy array and resize
            img = frame.to_ndarray(format="rgb24")
            if self.config.resolution:
                img = cv2.resize(img, self.config.resolution)

            # Save frame
            frame_path = output_dir / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(frame_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            frames.append(str(frame_path))

            frame_count += 1

        container.close()
        return frames

    def _extract_audio(self, video_path: Path, output_dir: Path) -> Path:
        """Extract audio track from video."""
        audio_path = output_dir / "audio.wav"

        # Use FFmpeg for audio extraction
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM 16-bit
            "-ar", str(self.config.audio_sample_rate),
            "-ac", "1",  # Mono
            "-y",  # Overwrite
            str(audio_path)
        ]

        subprocess.run(cmd, capture_output=True, check=True)
        return audio_path

    def _transcribe_audio(self, audio_path: Path) -> Dict[str, Any]:
        """Transcribe audio using Whisper."""
        if not self.whisper_model:
            return None

        result = self.whisper_model.transcribe(
            str(audio_path),
            language=self.config.transcription_language
        )

        return {
            "text": result["text"],
            "language": result.get("language"),
            "segments": result.get("segments", [])
        }

    def _caption_frames(self, frame_paths: List[str]) -> List[Dict[str, str]]:
        """Generate captions for frames."""
        if not self.captioner:
            return []

        captions = []
        for frame_path in tqdm(frame_paths, desc="Captioning", leave=False):
            try:
                caption = self.captioner.caption_image(frame_path)
                captions.append({
                    "frame_path": frame_path,
                    "caption": caption
                })
            except Exception as e:
                print(f"Error captioning {frame_path}: {e}")
                captions.append({
                    "frame_path": frame_path,
                    "caption": "",
                    "error": str(e)
                })

        return captions

    def _detect_scenes(self, video_path: Path) -> List[Dict[str, Any]]:
        """Detect scene changes in video."""
        try:
            from scenedetect import detect, ContentDetector

            scenes = detect(
                str(video_path),
                ContentDetector(threshold=self.config.scene_threshold)
            )

            return [
                {
                    "start_time": scene[0].get_seconds(),
                    "end_time": scene[1].get_seconds(),
                    "start_frame": scene[0].get_frames(),
                    "end_frame": scene[1].get_frames()
                }
                for scene in scenes
            ]
        except Exception as e:
            print(f"Scene detection failed: {e}")
            return []

    def _generate_thumbnail(self, frame_path: str, output_dir: Path) -> str:
        """Generate thumbnail from first frame."""
        img = cv2.imread(frame_path)
        img = cv2.resize(img, self.config.thumbnail_size)

        thumbnail_path = output_dir / "thumbnail.jpg"
        cv2.imwrite(str(thumbnail_path), img)

        return str(thumbnail_path)

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _create_manifest_entry(self, result: IngestionResult) -> Dict[str, Any]:
        """Create manifest entry for ingestion result."""
        entry = {
            "video_id": result.video_id,
            "file_path": result.metadata.file_path,
            "metadata": asdict(result.metadata),
            "frames_extracted": result.frames_extracted,
            "timestamp": datetime.utcnow().isoformat()
        }

        if result.audio_path:
            entry["audio_path"] = result.audio_path

        if result.transcript:
            entry["transcript"] = result.transcript

        if result.captions:
            entry["captions"] = result.captions

        if result.scenes:
            entry["scenes"] = result.scenes

        if result.thumbnail_path:
            entry["thumbnail_path"] = result.thumbnail_path

        if result.errors:
            entry["errors"] = result.errors

        return entry

    def _write_manifest(self, results: List[IngestionResult]):
        """Write manifest file with all ingestion results."""
        manifest_path = self.output_dir / "manifest.jsonl"

        with open(manifest_path, "w") as f:
            for result in results:
                if result.manifest_entry:
                    f.write(json.dumps(result.manifest_entry) + "\n")

        print(f"\n✓ Manifest written to: {manifest_path}")
        print(f"  Total videos: {len(results)}")
        print(f"  Successful: {sum(1 for r in results if not r.errors)}")
        print(f"  Failed: {sum(1 for r in results if r.errors)}")


def create_wizard_from_dict(config_dict: Dict[str, Any]) -> VideoWizard:
    """Create VideoWizard from configuration dictionary."""
    config = VideoIngestionConfig(**config_dict)
    return VideoWizard(config)
