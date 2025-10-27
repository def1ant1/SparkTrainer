"""
FFmpeg validation and video processing utilities.
"""
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..logger import get_logger

logger = get_logger()


def check_ffmpeg() -> tuple[bool, str]:
    """
    Check if ffmpeg is installed and accessible.

    Returns:
        Tuple of (is_installed, version_or_error_message)
    """
    if not shutil.which("ffmpeg"):
        return False, "ffmpeg not found in PATH"

    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            version_line = result.stdout.split("\n")[0]
            return True, version_line
        else:
            return False, "ffmpeg command failed"
    except Exception as e:
        return False, f"Error checking ffmpeg: {e}"


def check_ffprobe() -> tuple[bool, str]:
    """
    Check if ffprobe is installed and accessible.

    Returns:
        Tuple of (is_installed, version_or_error_message)
    """
    if not shutil.which("ffprobe"):
        return False, "ffprobe not found in PATH"

    try:
        result = subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            version_line = result.stdout.split("\n")[0]
            return True, version_line
        else:
            return False, "ffprobe command failed"
    except Exception as e:
        return False, f"Error checking ffprobe: {e}"


def validate_ffmpeg_installation() -> bool:
    """
    Validate FFmpeg installation and log results.

    Returns:
        True if both ffmpeg and ffprobe are available
    """
    ffmpeg_ok, ffmpeg_msg = check_ffmpeg()
    ffprobe_ok, ffprobe_msg = check_ffprobe()

    if ffmpeg_ok:
        logger.info(f"FFmpeg found: {ffmpeg_msg}")
    else:
        logger.error(f"FFmpeg check failed: {ffmpeg_msg}")

    if ffprobe_ok:
        logger.info(f"FFprobe found: {ffprobe_msg}")
    else:
        logger.error(f"FFprobe check failed: {ffprobe_msg}")

    if not (ffmpeg_ok and ffprobe_ok):
        logger.error("FFmpeg installation incomplete. Run scripts/install_ffmpeg.sh")
        return False

    return True


def get_video_metadata(video_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract video metadata using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video metadata

    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If ffprobe fails
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")

        metadata = json.loads(result.stdout)
        return metadata

    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse ffprobe output: {e}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffprobe command timed out")
    except Exception as e:
        raise RuntimeError(f"Error extracting video metadata: {e}")


def validate_video(video_path: Union[str, Path]) -> tuple[bool, Optional[str]]:
    """
    Validate if a file is a valid video.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (is_valid, error_message_if_invalid)
    """
    video_path = Path(video_path)

    if not video_path.exists():
        return False, f"File not found: {video_path}"

    if not video_path.is_file():
        return False, f"Not a file: {video_path}"

    try:
        metadata = get_video_metadata(video_path)

        # Check if video stream exists
        if "streams" not in metadata:
            return False, "No streams found in video"

        has_video = any(
            stream.get("codec_type") == "video"
            for stream in metadata["streams"]
        )

        if not has_video:
            return False, "No video stream found"

        return True, None

    except Exception as e:
        return False, str(e)


def extract_frames(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    fps: Optional[float] = None,
    resolution: Optional[str] = None,
    start_time: Optional[float] = None,
    duration: Optional[float] = None,
    quality: int = 2,
) -> int:
    """
    Extract frames from video using ffmpeg.

    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        fps: Frame rate for extraction (None = all frames)
        resolution: Target resolution as "WxH" (e.g., "224x224")
        start_time: Start time in seconds
        duration: Duration in seconds
        quality: JPEG quality (1-31, lower is better, 2 is recommended)

    Returns:
        Number of frames extracted

    Raises:
        RuntimeError: If ffmpeg fails
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = str(output_dir / "frame_%06d.jpg")

    cmd = ["ffmpeg", "-i", str(video_path)]

    if start_time is not None:
        cmd.extend(["-ss", str(start_time)])

    if duration is not None:
        cmd.extend(["-t", str(duration)])

    if fps is not None:
        cmd.extend(["-vf", f"fps={fps}"])

    if resolution is not None:
        scale_filter = f"scale={resolution}"
        if fps is not None:
            # Combine with fps filter
            cmd[-1] = f"{cmd[-1]},{scale_filter}"
        else:
            cmd.extend(["-vf", scale_filter])

    cmd.extend([
        "-q:v", str(quality),
        "-y",
        output_pattern,
    ])

    logger.debug(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes max
        )

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")

        # Count extracted frames
        frame_count = len(list(output_dir.glob("frame_*.jpg")))
        logger.info(f"Extracted {frame_count} frames from {video_path.name}")

        return frame_count

    except subprocess.TimeoutExpired:
        raise RuntimeError("Frame extraction timed out")
    except Exception as e:
        raise RuntimeError(f"Error extracting frames: {e}")


def extract_audio(
    video_path: Union[str, Path],
    output_path: Union[str, Path],
    sample_rate: int = 16000,
    mono: bool = True,
) -> bool:
    """
    Extract audio from video using ffmpeg.

    Args:
        video_path: Path to video file
        output_path: Path to output audio file (WAV)
        sample_rate: Audio sample rate
        mono: Convert to mono

    Returns:
        True if successful

    Raises:
        RuntimeError: If ffmpeg fails
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # PCM 16-bit
        "-ar", str(sample_rate),
    ]

    if mono:
        cmd.extend(["-ac", "1"])

    cmd.extend(["-y", str(output_path)])

    logger.debug(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes max
        )

        if result.returncode != 0:
            # Check if audio stream exists
            if "does not contain any stream" in result.stderr or "No audio" in result.stderr:
                logger.warning(f"No audio stream in {video_path.name}")
                return False
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")

        logger.info(f"Extracted audio from {video_path.name}")
        return True

    except subprocess.TimeoutExpired:
        raise RuntimeError("Audio extraction timed out")
    except Exception as e:
        raise RuntimeError(f"Error extracting audio: {e}")
