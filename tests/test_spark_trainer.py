"""
Automated test suite for SparkTrainer.

Tests FFmpeg utilities, Whisper integration, and trainer functionality.
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
import numpy as np


class TestFFmpegUtils:
    """Test FFmpeg utilities"""

    def test_ffmpeg_installed(self):
        """Test FFmpeg installation"""
        from spark_trainer.utils.ffmpeg_utils import validate_ffmpeg_installation

        result = validate_ffmpeg_installation()
        assert result, "FFmpeg is not installed or not in PATH"

    def test_validate_video(self):
        """Test video validation"""
        from spark_trainer.utils.ffmpeg_utils import validate_video

        # Test with non-existent file
        result = validate_video("nonexistent.mp4")
        assert not result

    def test_extract_frames(self):
        """Test frame extraction"""
        from spark_trainer.utils.ffmpeg_utils import extract_frames

        # Create test video (requires actual video file)
        # Skipped if no test video available
        test_video = Path("tests/data/test_video.mp4")
        if not test_video.exists():
            pytest.skip("Test video not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "frames"
            output_dir.mkdir()

            success = extract_frames(
                video_path=str(test_video),
                output_dir=str(output_dir),
                frame_rate=1.0
            )

            assert success
            frames = list(output_dir.glob("*.jpg"))
            assert len(frames) > 0


class TestHashing:
    """Test hashing utilities"""

    def test_deterministic_hash(self):
        """Test deterministic video hashing"""
        from spark_trainer.utils.hashing import get_video_hash

        # Test with same input
        hash1 = get_video_hash("test.mp4", deterministic=True)
        hash2 = get_video_hash("test.mp4", deterministic=True)

        assert hash1 == hash2
        assert len(hash1) > 0

    def test_directory_from_hash(self):
        """Test directory generation from hash"""
        from spark_trainer.utils.hashing import get_directory_from_hash

        hash_val = "abcdef1234567890"
        dir_path = get_directory_from_hash(hash_val, prefix_length=2)

        assert "ab" in dir_path
        assert hash_val in dir_path


class TestManifest:
    """Test manifest utilities"""

    def test_manifest_creation(self):
        """Test creating manifest entry"""
        from spark_trainer.utils.manifest import create_manifest_entry

        entry = create_manifest_entry(
            video_id="test123",
            frames_dir="/path/to/frames",
            audio_path="/path/to/audio.wav",
            captions=["A test caption"],
            duration=30.0
        )

        assert entry["id"] == "test123"
        assert entry["frames_dir"] == "/path/to/frames"
        assert entry["meta"]["duration"] == 30.0
        assert len(entry["meta"]["captions"]) == 1

    def test_manifest_validation(self):
        """Test manifest validation"""
        from spark_trainer.utils.manifest import validate_manifest_entry

        valid_entry = {
            "id": "test",
            "frames_dir": "/path",
            "meta": {"duration": 10.0}
        }

        assert validate_manifest_entry(valid_entry)

        invalid_entry = {"id": "test"}  # Missing required fields
        assert not validate_manifest_entry(invalid_entry)


class TestSceneDetection:
    """Test scene detection"""

    def test_scene_detector_creation(self):
        """Test creating scene detector"""
        from spark_trainer.scene_detection import SceneDetector

        detector = SceneDetector(threshold=30.0, min_scene_length=15)

        assert detector.threshold == 30.0
        assert detector.min_scene_length == 15

    def test_detect_scenes(self):
        """Test scene detection"""
        from spark_trainer.scene_detection import detect_scenes

        test_video = Path("tests/data/test_video.mp4")
        if not test_video.exists():
            pytest.skip("Test video not available")

        scenes = detect_scenes(str(test_video), threshold=30.0)

        assert isinstance(scenes, list)
        # May be empty if video is too short


class TestSmartSampling:
    """Test smart sampling"""

    def test_motion_aware_sampler(self):
        """Test motion-aware sampler creation"""
        from spark_trainer.smart_sampling import MotionAwareSampler

        sampler = MotionAwareSampler(
            base_fps=1.0,
            motion_threshold=0.3
        )

        assert sampler.base_fps == 1.0
        assert sampler.motion_threshold == 0.3

    def test_perceptual_hasher(self):
        """Test perceptual hashing"""
        from spark_trainer.smart_sampling import PerceptualHasher

        hasher = PerceptualHasher(hash_size=8)

        # Test hamming distance
        hash1 = "abcd1234"
        hash2 = "abcd1234"
        hash3 = "ffff1234"

        dist1 = hasher.hamming_distance(hash1, hash2)
        dist2 = hasher.hamming_distance(hash1, hash3)

        assert dist1 == 0  # Identical
        assert dist2 > 0   # Different


class TestObjectDetection:
    """Test YOLOv8 object detection"""

    def test_yolo_detector_creation(self):
        """Test creating YOLOv8 detector"""
        try:
            from spark_trainer.object_detection import YOLOv8Detector

            # Skip if ultralytics not installed
            detector = YOLOv8Detector(model_name="yolov8n.pt")
            assert detector.model is not None

        except ImportError:
            pytest.skip("ultralytics not installed")


class TestDatasetUtils:
    """Test dataset utilities"""

    def test_dataset_splitter(self):
        """Test dataset splitting"""
        from spark_trainer.dataset_utils import DatasetSplitter

        # Create test data
        samples = [
            {"id": f"sample_{i}", "label": i % 3}
            for i in range(100)
        ]

        splitter = DatasetSplitter(seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test manifest
            manifest_path = Path(tmpdir) / "test.jsonl"
            with open(manifest_path, 'w') as f:
                import json
                for sample in samples:
                    f.write(json.dumps(sample) + '\n')

            # Test split
            splits = splitter.split_manifest(
                manifest_path=manifest_path,
                splits=(0.8, 0.1, 0.1),
                stratify_key="label"
            )

            assert len(splits['train']) > 0
            assert len(splits['val']) > 0
            assert len(splits['test']) > 0
            assert len(splits['train']) + len(splits['val']) + len(splits['test']) == 100

    def test_consistency_checker(self):
        """Test consistency checker"""
        from spark_trainer.dataset_utils import ConsistencyChecker

        checker = ConsistencyChecker()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test manifest with issues
            manifest_path = Path(tmpdir) / "test.jsonl"
            with open(manifest_path, 'w') as f:
                import json
                # Valid sample
                f.write(json.dumps({
                    "id": "sample1",
                    "frames_dir": tmpdir,
                    "meta": {"captions": ["Valid caption"]}
                }) + '\n')

                # Invalid sample (empty caption)
                f.write(json.dumps({
                    "id": "sample2",
                    "frames_dir": tmpdir,
                    "meta": {"captions": [""]}
                }) + '\n')

            result = checker.check_manifest(manifest_path)

            assert result['stats']['total_samples'] == 2
            assert result['stats']['empty_captions'] > 0


class TestProvenance:
    """Test provenance tracking"""

    def test_provenance_tracker(self):
        """Test provenance tracker"""
        from spark_trainer.provenance import ProvenanceTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ProvenanceTracker(provenance_dir=tmpdir)

            # Create record
            record = tracker.create_record(
                dataset_id="test_dataset",
                source_paths=[],
                source_type="video",
                transformations=[{"type": "resize", "params": {"size": [224, 224]}}]
            )

            assert record.dataset_id == "test_dataset"
            assert len(record.transformations) == 1

            # Test retrieval
            retrieved = tracker.get_record("test_dataset")
            assert retrieved is not None
            assert retrieved.dataset_id == "test_dataset"

    def test_hash_computation(self):
        """Test hash computation"""
        from spark_trainer.provenance import ProvenanceTracker

        tracker = ProvenanceTracker()

        # Test content hash
        hash1 = tracker.compute_content_hash("test string")
        hash2 = tracker.compute_content_hash("test string")
        hash3 = tracker.compute_content_hash("different string")

        assert hash1 == hash2
        assert hash1 != hash3


class TestTrainerRegistry:
    """Test trainer autodiscovery"""

    def test_trainer_registry_creation(self):
        """Test creating trainer registry"""
        from spark_trainer.trainer_registry import TrainerRegistry

        registry = TrainerRegistry()

        assert registry.trainers_dir.exists()

    def test_trainer_discovery(self):
        """Test trainer discovery"""
        from spark_trainer.trainer_registry import get_trainer_registry

        registry = get_trainer_registry()
        trainers = registry.list_trainers()

        # Should discover at least the built-in trainers
        assert isinstance(trainers, list)


class TestAugmentation:
    """Test augmentation system"""

    def test_augmentation_manager(self):
        """Test augmentation manager"""
        from spark_trainer.augmentation import AugmentationManager

        manager = AugmentationManager()

        # Test pipeline retrieval
        pipelines = manager.list_pipelines()
        assert isinstance(pipelines, list)
        assert len(pipelines) > 0

    def test_get_augmentations(self):
        """Test getting augmentations"""
        from spark_trainer.augmentation import get_augmentation_manager

        manager = get_augmentation_manager()

        # Get image augmentations
        augs = manager.get_augmentations('image', ['basic'])

        assert isinstance(augs, list)


class TestProfiles:
    """Test profile management"""

    def test_profile_manager(self):
        """Test profile manager"""
        from spark_trainer.profiles import ProfileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProfileManager(base_dir=tmpdir)

            # Test settings
            settings = manager.get_settings()
            assert settings is not None

            # Test profile creation
            manager.create_profile(
                name="test_profile",
                config={"batch_size": 32},
                description="Test profile"
            )

            # Retrieve profile
            profile = manager.get_profile("test_profile")
            assert profile is not None
            assert profile["config"]["batch_size"] == 32

    def test_environment_info(self):
        """Test environment info"""
        from spark_trainer.profiles import get_profile_manager

        manager = get_profile_manager()
        env_info = manager.get_environment_info()

        assert 'system' in env_info
        assert 'pytorch' in env_info


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
