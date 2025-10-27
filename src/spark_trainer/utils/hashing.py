"""
Deterministic hashing and data layout for video processing.
"""
import hashlib
from pathlib import Path
from typing import Union


def hash_video(video_path: Union[str, Path], chunk_size: int = 8192) -> str:
    """
    Compute SHA256 hash of a video file.

    Args:
        video_path: Path to video file
        chunk_size: Chunk size for reading file (bytes)

    Returns:
        Hexadecimal hash string
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    sha256 = hashlib.sha256()
    with open(video_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha256.update(chunk)

    return sha256.hexdigest()


def hash_string(s: str) -> str:
    """
    Compute SHA256 hash of a string.

    Args:
        s: Input string

    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def create_deterministic_layout(
    video_path: Union[str, Path],
    output_base: Union[str, Path],
    use_full_hash: bool = False,
    hash_prefix_len: int = 8,
) -> dict:
    """
    Create deterministic directory layout for video processing outputs.

    The layout uses the video hash to organize files:
    - If use_full_hash=True: output_base/<full_hash>/
    - If use_full_hash=False: output_base/<hash_prefix>/<full_hash>/

    Args:
        video_path: Path to video file
        output_base: Base output directory
        use_full_hash: If True, use full hash as directory name
        hash_prefix_len: Length of hash prefix for hierarchical structure

    Returns:
        Dictionary with keys:
            - video_hash: Full hash of video
            - output_dir: Path to output directory for this video
            - frames_dir: Path to frames subdirectory
            - audio_path: Path to audio file
    """
    video_path = Path(video_path)
    output_base = Path(output_base)

    # Compute video hash
    video_hash = hash_video(video_path)

    # Create directory structure
    if use_full_hash:
        output_dir = output_base / video_hash
    else:
        hash_prefix = video_hash[:hash_prefix_len]
        output_dir = output_base / hash_prefix / video_hash

    frames_dir = output_dir / "frames"
    audio_path = output_dir / "audio.wav"

    # Create directories
    frames_dir.mkdir(parents=True, exist_ok=True)

    return {
        "video_hash": video_hash,
        "output_dir": output_dir,
        "frames_dir": frames_dir,
        "audio_path": audio_path,
    }


def get_video_id(video_path: Union[str, Path], use_hash: bool = True) -> str:
    """
    Get a unique identifier for a video.

    Args:
        video_path: Path to video file
        use_hash: If True, use hash as ID. If False, use filename stem.

    Returns:
        Unique video identifier
    """
    video_path = Path(video_path)

    if use_hash:
        return hash_video(video_path)
    else:
        return video_path.stem
