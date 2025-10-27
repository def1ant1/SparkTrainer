"""
Unit tests for hashing utilities.
"""
import tempfile
from pathlib import Path

import pytest

from src.spark_trainer.utils.hashing import (
    create_deterministic_layout,
    get_video_id,
    hash_string,
    hash_video,
)


def test_hash_string():
    """Test string hashing."""
    s1 = "hello world"
    s2 = "hello world"
    s3 = "different"

    h1 = hash_string(s1)
    h2 = hash_string(s2)
    h3 = hash_string(s3)

    assert h1 == h2, "Same strings should have same hash"
    assert h1 != h3, "Different strings should have different hashes"
    assert len(h1) == 64, "SHA256 hash should be 64 characters"


def test_hash_video():
    """Test video file hashing."""
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".mp4", delete=False) as f:
        f.write(b"fake video content")
        video_path = Path(f.name)

    try:
        h1 = hash_video(video_path)
        h2 = hash_video(video_path)

        assert h1 == h2, "Same file should have same hash"
        assert len(h1) == 64, "SHA256 hash should be 64 characters"
    finally:
        video_path.unlink()


def test_hash_video_not_found():
    """Test hash_video with non-existent file."""
    with pytest.raises(FileNotFoundError):
        hash_video("nonexistent.mp4")


def test_create_deterministic_layout():
    """Test deterministic layout creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create fake video
        video_path = Path(tmpdir) / "test.mp4"
        video_path.write_bytes(b"fake video")

        output_base = Path(tmpdir) / "output"

        layout = create_deterministic_layout(
            video_path=video_path,
            output_base=output_base,
            use_full_hash=False,
            hash_prefix_len=2,
        )

        assert "video_hash" in layout
        assert "output_dir" in layout
        assert "frames_dir" in layout
        assert "audio_path" in layout

        # Check that frames_dir was created
        assert layout["frames_dir"].exists()

        # Test with full hash
        layout2 = create_deterministic_layout(
            video_path=video_path,
            output_base=output_base,
            use_full_hash=True,
        )

        assert layout2["video_hash"] == layout["video_hash"]


def test_get_video_id():
    """Test video ID generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / "test_video.mp4"
        video_path.write_bytes(b"fake video")

        # Hash-based ID
        id_hash = get_video_id(video_path, use_hash=True)
        assert len(id_hash) == 64

        # Filename-based ID
        id_name = get_video_id(video_path, use_hash=False)
        assert id_name == "test_video"
