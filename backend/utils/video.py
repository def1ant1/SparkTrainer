"""
Video processing utilities for dataset ingestion and preprocessing.

Features:
- FFprobe integration for video metadata extraction
- FFmpeg for frame extraction and video preprocessing
- Optional Whisper integration for audio transcription
- Manifest generation for video datasets
- Background job support for large datasets
"""

import os
import subprocess
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import tempfile
import shutil


def check_ffmpeg_installed() -> bool:
    """Check if ffmpeg and ffprobe are installed"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """
    Extract video metadata using ffprobe.

    Returns:
        Dict with duration, fps, width, height, codec, bitrate, etc.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        # Find video stream
        video_stream = next(
            (s for s in data.get('streams', []) if s.get('codec_type') == 'video'),
            None
        )

        # Find audio stream
        audio_stream = next(
            (s for s in data.get('streams', []) if s.get('codec_type') == 'audio'),
            None
        )

        if not video_stream:
            raise ValueError(f"No video stream found in {video_path}")

        # Parse FPS
        fps_str = video_stream.get('r_frame_rate', '0/1')
        fps_num, fps_den = map(int, fps_str.split('/'))
        fps = fps_num / fps_den if fps_den > 0 else 0

        metadata = {
            'duration': float(data.get('format', {}).get('duration', 0)),
            'fps': fps,
            'width': int(video_stream.get('width', 0)),
            'height': int(video_stream.get('height', 0)),
            'codec': video_stream.get('codec_name', 'unknown'),
            'bitrate': int(data.get('format', {}).get('bit_rate', 0)),
            'size_bytes': int(data.get('format', {}).get('size', 0)),
            'nb_frames': int(video_stream.get('nb_frames', 0)) if 'nb_frames' in video_stream else None,
            'has_audio': audio_stream is not None,
            'audio_codec': audio_stream.get('codec_name') if audio_stream else None,
        }

        # Estimate frame count if not available
        if metadata['nb_frames'] is None and metadata['duration'] > 0:
            metadata['nb_frames'] = int(metadata['duration'] * metadata['fps'])

        return metadata

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFprobe error: {e.stderr}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse FFprobe output: {e}")


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    start_time: float = 0,
    duration: Optional[float] = None,
    quality: int = 2,
    format: str = 'jpg'
) -> List[str]:
    """
    Extract frames from video using ffmpeg.

    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract (None = extract all)
        max_frames: Maximum number of frames to extract
        start_time: Start time in seconds
        duration: Duration in seconds (None = until end)
        quality: JPEG quality (1-31, lower is better)
        format: Output format (jpg, png)

    Returns:
        List of extracted frame paths
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build ffmpeg command
    cmd = ['ffmpeg', '-i', video_path]

    # Add time range
    if start_time > 0:
        cmd.extend(['-ss', str(start_time)])
    if duration:
        cmd.extend(['-t', str(duration)])

    # Add FPS filter
    if fps:
        cmd.extend(['-vf', f'fps={fps}'])

    # Add frame limit
    if max_frames:
        cmd.extend(['-frames:v', str(max_frames)])

    # Quality settings
    if format == 'jpg':
        cmd.extend(['-q:v', str(quality)])

    # Output pattern
    output_pattern = os.path.join(output_dir, f'frame_%06d.{format}')
    cmd.append(output_pattern)

    # Run extraction
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Frame extraction failed: {e.stderr.decode()}")

    # Get list of extracted frames
    frames = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith('frame_') and f.endswith(f'.{format}')
    ])

    return frames


def extract_audio(video_path: str, output_path: str, format: str = 'wav') -> str:
    """
    Extract audio from video.

    Args:
        video_path: Path to video file
        output_path: Path to output audio file
        format: Audio format (wav, mp3, flac)

    Returns:
        Path to extracted audio file
    """
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vn',  # No video
        '-acodec', 'pcm_s16le' if format == 'wav' else 'copy',
        '-ar', '16000',  # 16kHz for Whisper
        '-ac', '1',  # Mono
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Audio extraction failed: {e.stderr.decode()}")


def transcribe_audio_whisper(
    audio_path: str,
    model: str = 'base',
    language: Optional[str] = None
) -> Dict[str, Any]:
    """
    Transcribe audio using Whisper.

    Args:
        audio_path: Path to audio file
        model: Whisper model size (tiny, base, small, medium, large)
        language: Language code (None = auto-detect)

    Returns:
        Dict with 'text', 'segments', 'language'
    """
    try:
        import whisper
    except ImportError:
        raise ImportError("Whisper not installed. Run: pip install openai-whisper")

    # Load model
    whisper_model = whisper.load_model(model)

    # Transcribe
    result = whisper_model.transcribe(
        audio_path,
        language=language,
        fp16=False  # CPU compatibility
    )

    return {
        'text': result['text'],
        'segments': result.get('segments', []),
        'language': result.get('language', 'unknown')
    }


def scan_video_directory(
    directory: str,
    extensions: List[str] = None,
    recursive: bool = True
) -> List[Dict[str, Any]]:
    """
    Scan directory for video files and extract basic info.

    Args:
        directory: Directory to scan
        extensions: List of video extensions (default: mp4, avi, mov, mkv, webm)
        recursive: Scan subdirectories

    Returns:
        List of dicts with video file information
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']

    extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in extensions]

    videos = []

    search_pattern = '**/*' if recursive else '*'
    for path in Path(directory).glob(search_pattern):
        if path.is_file() and path.suffix.lower() in extensions:
            rel_path = str(path.relative_to(directory))

            # Try to extract label from directory structure
            # Assumes structure like: /path/to/category/video.mp4
            parts = Path(rel_path).parts
            label = parts[-2] if len(parts) > 1 else 'unlabeled'

            videos.append({
                'path': str(path),
                'relative_path': rel_path,
                'filename': path.name,
                'label': label,
                'extension': path.suffix.lower(),
                'size_bytes': path.stat().st_size
            })

    return videos


def build_video_manifest(
    videos: List[Dict[str, Any]],
    output_path: str,
    extract_metadata: bool = True,
    progress_callback: Optional[callable] = None
) -> str:
    """
    Build a JSONL manifest file for video dataset.

    Args:
        videos: List of video info dicts from scan_video_directory
        output_path: Path to output manifest file (.jsonl)
        extract_metadata: Extract full metadata using ffprobe
        progress_callback: Optional callback(current, total) for progress

    Returns:
        Path to manifest file
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'w') as f:
        for idx, video in enumerate(videos):
            record = {
                'path': video['path'],
                'relative_path': video['relative_path'],
                'filename': video['filename'],
                'label': video['label'],
                'size_bytes': video['size_bytes']
            }

            if extract_metadata:
                try:
                    metadata = get_video_metadata(video['path'])
                    record.update(metadata)
                except Exception as e:
                    record['metadata_error'] = str(e)

            f.write(json.dumps(record) + '\n')

            if progress_callback:
                progress_callback(idx + 1, len(videos))

    return output_path


def sample_frames_uniform(
    video_path: str,
    num_frames: int,
    output_dir: str,
    format: str = 'jpg'
) -> List[str]:
    """
    Sample frames uniformly across video duration.

    Args:
        video_path: Path to video
        num_frames: Number of frames to sample
        output_dir: Output directory
        format: Image format

    Returns:
        List of frame paths
    """
    metadata = get_video_metadata(video_path)
    duration = metadata['duration']

    if duration <= 0:
        raise ValueError("Video has zero duration")

    # Calculate uniform time intervals
    interval = duration / (num_frames + 1)

    os.makedirs(output_dir, exist_ok=True)
    frames = []

    for i in range(num_frames):
        timestamp = (i + 1) * interval
        output_path = os.path.join(output_dir, f'frame_{i:06d}.{format}')

        cmd = [
            'ffmpeg',
            '-ss', str(timestamp),
            '-i', video_path,
            '-frames:v', '1',
            '-q:v', '2',
            output_path
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            frames.append(output_path)
        except subprocess.CalledProcessError:
            continue

    return frames


def process_video_for_training(
    video_path: str,
    output_dir: str,
    num_frames: int = 16,
    extract_audio: bool = False,
    transcribe: bool = False,
    whisper_model: str = 'base'
) -> Dict[str, Any]:
    """
    Full preprocessing pipeline for a single video.

    Args:
        video_path: Path to video
        output_dir: Output directory
        num_frames: Number of frames to extract
        extract_audio: Extract audio track
        transcribe: Transcribe audio with Whisper
        whisper_model: Whisper model size

    Returns:
        Dict with processing results
    """
    os.makedirs(output_dir, exist_ok=True)

    result = {
        'video_path': video_path,
        'metadata': {},
        'frames': [],
        'audio_path': None,
        'transcript': None
    }

    # Extract metadata
    try:
        result['metadata'] = get_video_metadata(video_path)
    except Exception as e:
        result['metadata_error'] = str(e)
        return result

    # Extract frames
    frames_dir = os.path.join(output_dir, 'frames')
    try:
        result['frames'] = sample_frames_uniform(
            video_path,
            num_frames,
            frames_dir
        )
    except Exception as e:
        result['frames_error'] = str(e)

    # Extract and transcribe audio
    if extract_audio and result['metadata'].get('has_audio'):
        audio_path = os.path.join(output_dir, 'audio.wav')
        try:
            result['audio_path'] = extract_audio(video_path, audio_path)

            if transcribe:
                try:
                    result['transcript'] = transcribe_audio_whisper(
                        audio_path,
                        model=whisper_model
                    )
                except Exception as e:
                    result['transcript_error'] = str(e)

        except Exception as e:
            result['audio_error'] = str(e)

    return result


# Utility functions for batch processing

def estimate_processing_time(
    video_paths: List[str],
    num_frames_per_video: int = 16,
    extract_audio: bool = False,
    transcribe: bool = False
) -> float:
    """
    Estimate total processing time in seconds.

    Returns rough estimate based on:
    - Metadata extraction: ~0.5s per video
    - Frame extraction: ~0.1s per frame
    - Audio extraction: ~2s per video
    - Transcription: ~duration/10 per video
    """
    total_time = 0

    for video_path in video_paths:
        # Metadata
        total_time += 0.5

        # Frames
        total_time += num_frames_per_video * 0.1

        if extract_audio:
            total_time += 2

            if transcribe:
                try:
                    metadata = get_video_metadata(video_path)
                    total_time += metadata.get('duration', 60) / 10
                except:
                    total_time += 6  # Assume 60s video

    return total_time


def get_video_stats(manifest_path: str) -> Dict[str, Any]:
    """
    Get statistics from a video manifest file.

    Returns:
        Dict with total videos, total duration, fps distribution, etc.
    """
    stats = {
        'total_videos': 0,
        'total_duration': 0,
        'total_frames': 0,
        'total_size_bytes': 0,
        'labels': {},
        'fps_distribution': {},
        'resolution_distribution': {},
        'errors': 0
    }

    with open(manifest_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            stats['total_videos'] += 1

            if 'metadata_error' in record:
                stats['errors'] += 1
                continue

            stats['total_duration'] += record.get('duration', 0)
            stats['total_frames'] += record.get('nb_frames', 0)
            stats['total_size_bytes'] += record.get('size_bytes', 0)

            # Count labels
            label = record.get('label', 'unknown')
            stats['labels'][label] = stats['labels'].get(label, 0) + 1

            # FPS distribution
            fps = int(record.get('fps', 0))
            stats['fps_distribution'][fps] = stats['fps_distribution'].get(fps, 0) + 1

            # Resolution
            res = f"{record.get('width', 0)}x{record.get('height', 0)}"
            stats['resolution_distribution'][res] = stats['resolution_distribution'].get(res, 0) + 1

    return stats
