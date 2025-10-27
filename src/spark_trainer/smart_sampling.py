"""
Smart sampling module for intelligent frame extraction.

Provides motion-aware FPS and perceptual hashing for deduplication.
"""

import os
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class MotionAwareSampler:
    """
    Motion-aware frame sampler that adapts sampling rate based on video content.

    High motion = more frames sampled
    Low motion = fewer frames sampled
    """

    def __init__(self,
                 base_fps: float = 1.0,
                 motion_threshold: float = 0.3,
                 adaptive_range: Tuple[float, float] = (0.5, 3.0)):
        """
        Initialize motion-aware sampler.

        Args:
            base_fps: Base sampling rate (frames per second)
            motion_threshold: Threshold for motion detection (0-1)
            adaptive_range: (min_fps, max_fps) for adaptive sampling
        """
        self.base_fps = base_fps
        self.motion_threshold = motion_threshold
        self.min_fps, self.max_fps = adaptive_range

    def calculate_frame_diff(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate difference between two frames.

        Args:
            frame1: First frame
            frame2: Second frame

        Returns:
            Normalized difference score (0-1)
        """
        try:
            import cv2

            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # Calculate absolute difference
            diff = cv2.absdiff(gray1, gray2)

            # Normalize to 0-1
            return np.mean(diff) / 255.0

        except Exception as e:
            logger.error(f"Frame diff calculation failed: {e}")
            return 0.0

    def sample_video(self,
                    video_path: Union[str, Path],
                    output_dir: Union[str, Path],
                    window_size: int = 30) -> List[Dict[str, Any]]:
        """
        Sample frames from video with motion-aware FPS.

        Args:
            video_path: Path to video
            output_dir: Output directory for frames
            window_size: Window size for motion analysis (frames)

        Returns:
            List of sampled frame info
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python required")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            cap = cv2.VideoCapture(str(video_path))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            sampled_frames = []
            video_name = Path(video_path).stem

            prev_frame = None
            frame_num = 0
            motion_scores = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Calculate motion if we have a previous frame
                if prev_frame is not None:
                    motion_score = self.calculate_frame_diff(prev_frame, frame)
                    motion_scores.append(motion_score)

                    # Keep sliding window of motion scores
                    if len(motion_scores) > window_size:
                        motion_scores.pop(0)

                    # Calculate average motion in window
                    avg_motion = np.mean(motion_scores)

                    # Adaptive FPS based on motion
                    if avg_motion > self.motion_threshold:
                        # High motion: sample more frequently
                        adaptive_fps = min(self.max_fps, self.base_fps * (1 + avg_motion))
                    else:
                        # Low motion: sample less frequently
                        adaptive_fps = max(self.min_fps, self.base_fps * avg_motion)

                    # Decide whether to save this frame
                    sample_interval = max(1, int(video_fps / adaptive_fps))
                    should_sample = (frame_num % sample_interval == 0)
                else:
                    should_sample = True  # Always sample first frame
                    avg_motion = 0.0
                    adaptive_fps = self.base_fps

                if should_sample:
                    timestamp = frame_num / video_fps if video_fps > 0 else frame_num
                    filename = f"{video_name}_frame{frame_num:06d}_t{timestamp:.2f}s.jpg"
                    output_path = output_dir / filename

                    cv2.imwrite(str(output_path), frame)

                    sampled_frames.append({
                        'frame_number': frame_num,
                        'timestamp': timestamp,
                        'motion_score': avg_motion,
                        'adaptive_fps': adaptive_fps,
                        'file_path': str(output_path)
                    })

                prev_frame = frame.copy()
                frame_num += 1

            cap.release()
            logger.info(f"Sampled {len(sampled_frames)} frames from {total_frames} total frames")
            return sampled_frames

        except Exception as e:
            logger.error(f"Motion-aware sampling failed: {e}")
            return []


class PerceptualHasher:
    """
    Perceptual hashing for image similarity and deduplication.

    Uses pHash (perceptual hash) to detect near-duplicate frames.
    """

    def __init__(self, hash_size: int = 8):
        """
        Initialize perceptual hasher.

        Args:
            hash_size: Size of hash (8x8 = 64-bit hash)
        """
        self.hash_size = hash_size

    def compute_phash(self, image_path: Union[str, Path]) -> str:
        """
        Compute perceptual hash of an image.

        Args:
            image_path: Path to image

        Returns:
            Hex string representation of hash
        """
        try:
            import cv2
            from scipy.fft import dct

            # Read image
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")

            # Resize to hash_size + 1 to allow for DCT
            img = cv2.resize(img, (self.hash_size + 1, self.hash_size + 1))

            # Compute DCT
            dct_img = dct(dct(img.T, norm='ortho').T, norm='ortho')

            # Keep only top-left corner (low frequencies)
            dct_reduced = dct_img[:self.hash_size, :self.hash_size]

            # Calculate median
            median = np.median(dct_reduced)

            # Create binary hash
            binary_hash = dct_reduced > median

            # Convert to hex string
            hash_array = binary_hash.flatten()
            hash_int = 0
            for bit in hash_array:
                hash_int = (hash_int << 1) | bit

            return format(hash_int, 'x')

        except Exception as e:
            logger.error(f"pHash computation failed for {image_path}: {e}")
            return ""

    def compute_phash_numpy(self, image: np.ndarray) -> str:
        """
        Compute perceptual hash from numpy array.

        Args:
            image: Image as numpy array

        Returns:
            Hex string representation of hash
        """
        try:
            import cv2
            from scipy.fft import dct

            # Convert to grayscale if needed
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Resize
            img = cv2.resize(image, (self.hash_size + 1, self.hash_size + 1))

            # Compute DCT
            dct_img = dct(dct(img.T, norm='ortho').T, norm='ortho')

            # Keep only top-left corner
            dct_reduced = dct_img[:self.hash_size, :self.hash_size]

            # Calculate median
            median = np.median(dct_reduced)

            # Create binary hash
            binary_hash = dct_reduced > median

            # Convert to hex
            hash_array = binary_hash.flatten()
            hash_int = 0
            for bit in hash_array:
                hash_int = (hash_int << 1) | bit

            return format(hash_int, 'x')

        except Exception as e:
            logger.error(f"pHash computation failed: {e}")
            return ""

    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """
        Calculate Hamming distance between two hashes.

        Args:
            hash1: First hash
            hash2: Second hash

        Returns:
            Hamming distance (number of differing bits)
        """
        if not hash1 or not hash2:
            return self.hash_size * self.hash_size

        try:
            # Convert hex to int
            int1 = int(hash1, 16)
            int2 = int(hash2, 16)

            # XOR and count bits
            xor = int1 ^ int2
            return bin(xor).count('1')

        except Exception as e:
            logger.error(f"Hamming distance calculation failed: {e}")
            return self.hash_size * self.hash_size

    def are_similar(self, hash1: str, hash2: str, threshold: int = 5) -> bool:
        """
        Check if two hashes are similar.

        Args:
            hash1: First hash
            hash2: Second hash
            threshold: Maximum Hamming distance for similarity

        Returns:
            True if similar, False otherwise
        """
        distance = self.hamming_distance(hash1, hash2)
        return distance <= threshold

    def deduplicate_images(self,
                          image_paths: List[Union[str, Path]],
                          similarity_threshold: int = 5) -> List[Union[str, Path]]:
        """
        Remove near-duplicate images from a list.

        Args:
            image_paths: List of image paths
            similarity_threshold: Hamming distance threshold

        Returns:
            List of unique image paths
        """
        if not image_paths:
            return []

        unique_images = []
        seen_hashes = []

        for image_path in image_paths:
            # Compute hash
            phash = self.compute_phash(image_path)

            if not phash:
                continue

            # Check if similar to any seen hash
            is_duplicate = False
            for seen_hash in seen_hashes:
                if self.are_similar(phash, seen_hash, similarity_threshold):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_images.append(image_path)
                seen_hashes.append(phash)

        logger.info(f"Deduplicated {len(image_paths)} images to {len(unique_images)} unique images")
        return unique_images


class SmartSampler:
    """
    Combined smart sampler with motion awareness and deduplication.
    """

    def __init__(self,
                 base_fps: float = 1.0,
                 motion_threshold: float = 0.3,
                 dedup_threshold: int = 5):
        """
        Initialize smart sampler.

        Args:
            base_fps: Base sampling rate
            motion_threshold: Motion detection threshold
            dedup_threshold: Deduplication threshold (Hamming distance)
        """
        self.motion_sampler = MotionAwareSampler(
            base_fps=base_fps,
            motion_threshold=motion_threshold
        )
        self.hasher = PerceptualHasher()
        self.dedup_threshold = dedup_threshold

    def sample_and_deduplicate(self,
                               video_path: Union[str, Path],
                               output_dir: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Sample frames with motion awareness and remove duplicates.

        Args:
            video_path: Path to video
            output_dir: Output directory

        Returns:
            List of unique sampled frames
        """
        # Step 1: Motion-aware sampling
        sampled_frames = self.motion_sampler.sample_video(video_path, output_dir)

        if not sampled_frames:
            return []

        # Step 2: Perceptual hashing and deduplication
        frame_paths = [f['file_path'] for f in sampled_frames]
        unique_paths = self.hasher.deduplicate_images(frame_paths, self.dedup_threshold)

        # Filter sampled_frames to only include unique ones
        unique_paths_set = set(unique_paths)
        unique_frames = [f for f in sampled_frames if f['file_path'] in unique_paths_set]

        # Remove duplicate files
        for frame in sampled_frames:
            if frame['file_path'] not in unique_paths_set:
                try:
                    os.remove(frame['file_path'])
                except:
                    pass

        logger.info(f"Smart sampling: {len(sampled_frames)} -> {len(unique_frames)} frames")
        return unique_frames


def smart_sample_video(video_path: Union[str, Path],
                      output_dir: Union[str, Path],
                      base_fps: float = 1.0,
                      motion_threshold: float = 0.3,
                      dedup_threshold: int = 5) -> List[Dict[str, Any]]:
    """
    Convenience function for smart video sampling.

    Args:
        video_path: Path to video
        output_dir: Output directory
        base_fps: Base sampling rate
        motion_threshold: Motion detection threshold
        dedup_threshold: Deduplication threshold

    Returns:
        List of sampled frames
    """
    sampler = SmartSampler(
        base_fps=base_fps,
        motion_threshold=motion_threshold,
        dedup_threshold=dedup_threshold
    )
    return sampler.sample_and_deduplicate(video_path, output_dir)
