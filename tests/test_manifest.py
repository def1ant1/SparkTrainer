"""
Unit tests for manifest utilities.
"""
import tempfile
from pathlib import Path

import pytest

from src.spark_trainer.utils.manifest import (
    ManifestV1,
    iter_manifest,
    load_manifest,
    save_manifest,
    validate_manifest,
)


def test_manifest_v1_to_dict():
    """Test ManifestV1 to_dict method."""
    entry = ManifestV1(
        id="test123",
        frames_dir="/path/to/frames",
        audio="/path/to/audio.wav",
        meta={"key": "value"},
    )

    d = entry.to_dict()
    assert d["id"] == "test123"
    assert d["frames_dir"] == "/path/to/frames"
    assert d["audio"] == "/path/to/audio.wav"
    assert d["meta"] == {"key": "value"}


def test_manifest_v1_from_dict():
    """Test ManifestV1 from_dict method."""
    d = {
        "id": "test123",
        "frames_dir": "/path/to/frames",
        "audio": "/path/to/audio.wav",
        "meta": {"key": "value"},
    }

    entry = ManifestV1.from_dict(d)
    assert entry.id == "test123"
    assert entry.frames_dir == "/path/to/frames"
    assert entry.audio == "/path/to/audio.wav"
    assert entry.meta == {"key": "value"}


def test_save_and_load_manifest():
    """Test saving and loading manifest."""
    entries = [
        ManifestV1(id="video1", frames_dir="/frames1", audio="/audio1.wav", meta={"duration": 10}),
        ManifestV1(id="video2", frames_dir="/frames2", audio=None, meta={"duration": 20}),
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        manifest_path = Path(f.name)

    try:
        # Save
        save_manifest(entries, manifest_path)

        # Load
        loaded_entries = load_manifest(manifest_path)

        assert len(loaded_entries) == 2
        assert loaded_entries[0].id == "video1"
        assert loaded_entries[0].frames_dir == "/frames1"
        assert loaded_entries[1].id == "video2"
        assert loaded_entries[1].audio is None
    finally:
        manifest_path.unlink()


def test_iter_manifest():
    """Test iterating over manifest."""
    entries = [
        ManifestV1(id=f"video{i}", frames_dir=f"/frames{i}", audio=None, meta={}) for i in range(100)
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        manifest_path = Path(f.name)

    try:
        save_manifest(entries, manifest_path)

        # Iterate without loading all into memory
        count = 0
        for entry in iter_manifest(manifest_path):
            count += 1
            assert entry.id.startswith("video")

        assert count == 100
    finally:
        manifest_path.unlink()


def test_validate_manifest():
    """Test manifest validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create fake frames directories
        frames1 = tmpdir / "frames1"
        frames1.mkdir()
        frames2 = tmpdir / "frames2"
        frames2.mkdir()

        entries = [
            ManifestV1(id="video1", frames_dir=str(frames1), audio=None, meta={}),
            ManifestV1(id="video2", frames_dir=str(frames2), audio=None, meta={}),
            ManifestV1(id="video3", frames_dir="/nonexistent", audio=None, meta={}),
        ]

        manifest_path = tmpdir / "manifest.jsonl"
        save_manifest(entries, manifest_path)

        # Validate
        is_valid, errors = validate_manifest(manifest_path)

        assert not is_valid
        assert len(errors) > 0
        assert any("nonexistent" in error for error in errors)


def test_manifest_append():
    """Test appending to manifest."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        manifest_path = Path(f.name)

    try:
        # Save initial entries
        entries1 = [ManifestV1(id="video1", frames_dir="/frames1", audio=None, meta={})]
        save_manifest(entries1, manifest_path)

        # Append more entries
        entries2 = [ManifestV1(id="video2", frames_dir="/frames2", audio=None, meta={})]
        save_manifest(entries2, manifest_path, append=True)

        # Load all
        all_entries = load_manifest(manifest_path)
        assert len(all_entries) == 2
        assert all_entries[0].id == "video1"
        assert all_entries[1].id == "video2"
    finally:
        manifest_path.unlink()
