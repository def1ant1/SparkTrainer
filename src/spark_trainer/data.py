"""
Data loading and streaming for video datasets.
Supports IterableDataset with clip sampling strategies.
"""
import random
from pathlib import Path
from typing import Dict, Literal, Optional, Union

import torch
from PIL import Image
from torch.utils.data import IterableDataset

from .logger import get_logger
from .utils.manifest import ManifestV1, iter_manifest

logger = get_logger()


class VideoFrameDataset(IterableDataset):
    """
    Iterable dataset for streaming video frames from manifest.
    Supports various clip sampling strategies.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        clip_length: int = 16,
        clip_stride: int = 1,
        clip_sampling: Literal["random", "uniform", "center"] = "random",
        transform=None,
        load_audio: bool = False,
        max_clips_per_video: Optional[int] = None,
        shuffle_videos: bool = True,
        seed: int = 42,
    ):
        """
        Initialize video frame dataset.

        Args:
            manifest_path: Path to manifest JSONL file
            clip_length: Number of frames per clip
            clip_stride: Stride between frames in clip
            clip_sampling: Sampling strategy ('random', 'uniform', 'center')
            transform: Optional transform to apply to frames
            load_audio: Whether to load audio data
            max_clips_per_video: Maximum clips to sample per video (None = all)
            shuffle_videos: Whether to shuffle video order
            seed: Random seed for reproducibility
        """
        self.manifest_path = Path(manifest_path)
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.clip_sampling = clip_sampling
        self.transform = transform
        self.load_audio = load_audio
        self.max_clips_per_video = max_clips_per_video
        self.shuffle_videos = shuffle_videos
        self.seed = seed

        # Validate manifest exists
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        logger.info(f"VideoFrameDataset initialized with manifest: {self.manifest_path}")
        logger.info(f"  Clip length: {clip_length}, stride: {clip_stride}, sampling: {clip_sampling}")

    def _get_frame_paths(self, frames_dir: Path) -> list[Path]:
        """Get sorted list of frame paths from directory."""
        frame_paths = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
        return frame_paths

    def _sample_clips(self, frame_paths: list[Path]) -> list[list[Path]]:
        """
        Sample clips from frame paths based on sampling strategy.

        Returns:
            List of clips, where each clip is a list of frame paths
        """
        num_frames = len(frame_paths)
        frames_per_clip = self.clip_length * self.clip_stride
        clips = []

        if num_frames < frames_per_clip:
            # Not enough frames for even one clip
            logger.warning(f"Only {num_frames} frames available, need {frames_per_clip}")
            return []

        if self.clip_sampling == "center":
            # Sample from center of video
            start_idx = (num_frames - frames_per_clip) // 2
            clip_frames = frame_paths[start_idx : start_idx + frames_per_clip : self.clip_stride]
            clips.append(clip_frames)

        elif self.clip_sampling == "uniform":
            # Uniformly sample clips across video
            num_possible_clips = (num_frames - frames_per_clip) // self.clip_stride + 1

            if self.max_clips_per_video:
                num_clips = min(self.max_clips_per_video, num_possible_clips)
            else:
                num_clips = num_possible_clips

            # Uniform spacing
            if num_clips > 1:
                indices = torch.linspace(0, num_possible_clips - 1, num_clips).long().tolist()
            else:
                indices = [0]

            for idx in indices:
                start_idx = idx * self.clip_stride
                clip_frames = frame_paths[start_idx : start_idx + frames_per_clip : self.clip_stride]
                clips.append(clip_frames)

        elif self.clip_sampling == "random":
            # Randomly sample clips
            num_possible_clips = (num_frames - frames_per_clip) // self.clip_stride + 1

            if self.max_clips_per_video:
                num_clips = min(self.max_clips_per_video, num_possible_clips)
            else:
                num_clips = num_possible_clips

            # Random sampling with seed
            rng = random.Random(self.seed + hash(str(frame_paths[0])))
            indices = rng.sample(range(num_possible_clips), num_clips)

            for idx in indices:
                start_idx = idx * self.clip_stride
                clip_frames = frame_paths[start_idx : start_idx + frames_per_clip : self.clip_stride]
                clips.append(clip_frames)

        return clips

    def _load_clip(self, clip_frames: list[Path]) -> torch.Tensor:
        """
        Load frames and stack into tensor.

        Args:
            clip_frames: List of frame paths

        Returns:
            Tensor of shape (T, C, H, W)
        """
        frames = []
        for frame_path in clip_frames:
            image = Image.open(frame_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            frames.append(image)

        # Stack into tensor
        if isinstance(frames[0], torch.Tensor):
            clip_tensor = torch.stack(frames, dim=0)
        else:
            # If transform doesn't convert to tensor, do it manually
            import torchvision.transforms as T
            to_tensor = T.ToTensor()
            frames_tensors = [to_tensor(f) for f in frames]
            clip_tensor = torch.stack(frames_tensors, dim=0)

        return clip_tensor

    def __iter__(self):
        """
        Iterate over clips from all videos in manifest.

        Yields:
            Dictionary with keys:
                - video_id: Video identifier
                - clip: Tensor of frames (T, C, H, W)
                - audio: Optional audio data
                - meta: Metadata dict
        """
        # Load all manifest entries
        entries = list(iter_manifest(self.manifest_path))

        # Shuffle if requested
        if self.shuffle_videos:
            rng = random.Random(self.seed)
            rng.shuffle(entries)

        for entry in entries:
            frames_dir = Path(entry.frames_dir)

            if not frames_dir.exists():
                logger.warning(f"Frames directory not found: {frames_dir}")
                continue

            # Get frame paths
            frame_paths = self._get_frame_paths(frames_dir)

            if not frame_paths:
                logger.warning(f"No frames found in: {frames_dir}")
                continue

            # Sample clips
            clips = self._sample_clips(frame_paths)

            for clip_frames in clips:
                try:
                    clip_tensor = self._load_clip(clip_frames)

                    sample = {
                        "video_id": entry.id,
                        "clip": clip_tensor,
                        "meta": entry.meta or {},
                    }

                    # Load audio if requested
                    if self.load_audio and entry.audio:
                        audio_path = Path(entry.audio)
                        if audio_path.exists():
                            # TODO: Implement audio loading
                            sample["audio"] = str(audio_path)

                    yield sample

                except Exception as e:
                    logger.error(f"Error loading clip from {entry.id}: {e}")
                    continue


class ImageCaptionDataset(IterableDataset):
    """
    Iterable dataset for image-caption pairs from manifest.
    Useful for vision-language model training.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        transform=None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Initialize image-caption dataset.

        Args:
            manifest_path: Path to manifest JSONL file
            transform: Optional transform to apply to images
            shuffle: Whether to shuffle samples
            seed: Random seed
        """
        self.manifest_path = Path(manifest_path)
        self.transform = transform
        self.shuffle = shuffle
        self.seed = seed

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        logger.info(f"ImageCaptionDataset initialized with manifest: {self.manifest_path}")

    def __iter__(self):
        """
        Iterate over image-caption pairs.

        Yields:
            Dictionary with keys:
                - image: Transformed image tensor
                - caption: Caption string
                - video_id: Video identifier
                - meta: Metadata dict
        """
        entries = list(iter_manifest(self.manifest_path))

        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(entries)

        for entry in entries:
            frames_dir = Path(entry.frames_dir)

            if not frames_dir.exists():
                logger.warning(f"Frames directory not found: {frames_dir}")
                continue

            # Get all frames
            frame_paths = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))

            for frame_path in frame_paths:
                try:
                    image = Image.open(frame_path).convert("RGB")
                    if self.transform:
                        image = self.transform(image)

                    # Get caption from metadata
                    meta = entry.meta or {}
                    caption = meta.get("caption", "")

                    sample = {
                        "image": image,
                        "caption": caption,
                        "video_id": entry.id,
                        "frame_path": str(frame_path),
                        "meta": meta,
                    }

                    yield sample

                except Exception as e:
                    logger.error(f"Error loading frame {frame_path}: {e}")
                    continue


def create_video_dataloader(
    manifest_path: Union[str, Path],
    batch_size: int = 8,
    num_workers: int = 4,
    clip_length: int = 16,
    clip_stride: int = 1,
    clip_sampling: Literal["random", "uniform", "center"] = "random",
    transform=None,
    **kwargs,
):
    """
    Create a DataLoader for video frame clips.

    Args:
        manifest_path: Path to manifest file
        batch_size: Batch size
        num_workers: Number of worker processes
        clip_length: Frames per clip
        clip_stride: Stride between frames
        clip_sampling: Sampling strategy
        transform: Image transform
        **kwargs: Additional arguments for VideoFrameDataset

    Returns:
        torch.utils.data.DataLoader
    """
    from torch.utils.data import DataLoader

    dataset = VideoFrameDataset(
        manifest_path=manifest_path,
        clip_length=clip_length,
        clip_stride=clip_stride,
        clip_sampling=clip_sampling,
        transform=transform,
        **kwargs,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader
