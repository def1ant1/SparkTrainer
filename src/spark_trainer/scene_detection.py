"""
Scene detection module for SparkTrainer using PySceneDetect.

Provides functionality for detecting scene changes in videos
to enable intelligent sampling and segmentation.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class SceneDetector:
    """
    Scene detector using PySceneDetect for video segmentation.

    Supports multiple detection algorithms:
    - Content-based detection
    - Threshold-based detection
    - Adaptive detection
    """

    def __init__(self,
                 threshold: float = 30.0,
                 min_scene_length: int = 15,
                 method: str = "content"):
        """
        Initialize scene detector.

        Args:
            threshold: Detection threshold (algorithm-specific)
            min_scene_length: Minimum scene length in frames
            method: Detection method ('content', 'threshold', 'adaptive')
        """
        self.threshold = threshold
        self.min_scene_length = min_scene_length
        self.method = method

    def detect_scenes(self,
                     video_path: Union[str, Path],
                     start_time: float = 0.0,
                     end_time: Optional[float] = None,
                     show_progress: bool = False) -> List[Dict[str, Any]]:
        """
        Detect scenes in a video.

        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds (None = end of video)
            show_progress: Show progress bar

        Returns:
            List of scenes, each containing:
            - scene_number: int
            - start_frame: int
            - end_frame: int
            - start_time: float (seconds)
            - end_time: float (seconds)
            - duration: float (seconds)
        """
        try:
            from scenedetect import open_video, SceneManager
            from scenedetect.detectors import ContentDetector, ThresholdDetector, AdaptiveDetector
        except ImportError:
            raise ImportError(
                "scenedetect package not installed. Install with: pip install scenedetect[opencv]"
            )

        try:
            # Open video
            video = open_video(str(video_path))

            # Create scene manager
            scene_manager = SceneManager()

            # Add detector based on method
            if self.method == "content":
                detector = ContentDetector(threshold=self.threshold, min_scene_len=self.min_scene_length)
            elif self.method == "threshold":
                detector = ThresholdDetector(threshold=self.threshold, min_scene_len=self.min_scene_length)
            elif self.method == "adaptive":
                detector = AdaptiveDetector(min_scene_len=self.min_scene_length)
            else:
                raise ValueError(f"Unknown detection method: {self.method}")

            scene_manager.add_detector(detector)

            # Detect scenes
            scene_manager.detect_scenes(
                video=video,
                show_progress=show_progress
            )

            # Get scene list
            scene_list = scene_manager.get_scene_list()

            # Convert to our format
            scenes = []
            for i, (start_time_obj, end_time_obj) in enumerate(scene_list):
                scene = {
                    'scene_number': i + 1,
                    'start_frame': start_time_obj.get_frames(),
                    'end_frame': end_time_obj.get_frames(),
                    'start_time': start_time_obj.get_seconds(),
                    'end_time': end_time_obj.get_seconds(),
                    'duration': (end_time_obj - start_time_obj).get_seconds()
                }
                scenes.append(scene)

            logger.info(f"Detected {len(scenes)} scenes in {video_path}")
            return scenes

        except Exception as e:
            logger.error(f"Scene detection failed for {video_path}: {e}")
            return []

    def detect_with_stats(self,
                         video_path: Union[str, Path],
                         show_progress: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Detect scenes and return statistics.

        Args:
            video_path: Path to video
            show_progress: Show progress bar

        Returns:
            Tuple of (scenes list, statistics dict)
        """
        scenes = self.detect_scenes(video_path, show_progress=show_progress)

        if not scenes:
            return scenes, {}

        # Calculate statistics
        durations = [s['duration'] for s in scenes]
        stats = {
            'num_scenes': len(scenes),
            'total_duration': sum(durations),
            'avg_scene_duration': sum(durations) / len(durations),
            'min_scene_duration': min(durations),
            'max_scene_duration': max(durations),
            'median_scene_duration': sorted(durations)[len(durations) // 2]
        }

        return scenes, stats

    def extract_scene_frames(self,
                            video_path: Union[str, Path],
                            output_dir: Union[str, Path],
                            scenes: Optional[List[Dict[str, Any]]] = None,
                            frame_interval: int = 30,
                            include_timestamps: bool = True) -> List[Dict[str, Any]]:
        """
        Extract representative frames from each scene.

        Args:
            video_path: Path to video
            output_dir: Output directory for frames
            scenes: Pre-detected scenes (None = detect now)
            frame_interval: Extract one frame every N frames
            include_timestamps: Include timestamp in filename

        Returns:
            List of extracted frame info
        """
        if scenes is None:
            scenes = self.detect_scenes(video_path)

        if not scenes:
            logger.warning(f"No scenes detected in {video_path}")
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python required for frame extraction")

        extracted_frames = []
        video_name = Path(video_path).stem

        try:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)

            for scene in scenes:
                scene_num = scene['scene_number']
                start_frame = scene['start_frame']
                end_frame = scene['end_frame']

                # Extract frames from this scene
                for frame_num in range(start_frame, end_frame, frame_interval):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()

                    if not ret:
                        continue

                    # Generate filename
                    timestamp = frame_num / fps if fps > 0 else frame_num
                    if include_timestamps:
                        filename = f"{video_name}_scene{scene_num:03d}_frame{frame_num:06d}_t{timestamp:.2f}s.jpg"
                    else:
                        filename = f"{video_name}_scene{scene_num:03d}_frame{frame_num:06d}.jpg"

                    output_path = output_dir / filename

                    # Save frame
                    cv2.imwrite(str(output_path), frame)

                    extracted_frames.append({
                        'scene_number': scene_num,
                        'frame_number': frame_num,
                        'timestamp': timestamp,
                        'file_path': str(output_path)
                    })

            cap.release()
            logger.info(f"Extracted {len(extracted_frames)} frames from {len(scenes)} scenes")
            return extracted_frames

        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return []

    def save_scenes_to_csv(self,
                          scenes: List[Dict[str, Any]],
                          output_path: Union[str, Path]):
        """
        Save scene list to CSV file.

        Args:
            scenes: List of detected scenes
            output_path: Output CSV path
        """
        import csv

        with open(output_path, 'w', newline='') as f:
            if not scenes:
                return

            writer = csv.DictWriter(f, fieldnames=scenes[0].keys())
            writer.writeheader()
            writer.writerows(scenes)

        logger.info(f"Saved {len(scenes)} scenes to {output_path}")

    def split_video_by_scenes(self,
                             video_path: Union[str, Path],
                             output_dir: Union[str, Path],
                             scenes: Optional[List[Dict[str, Any]]] = None,
                             codec: str = "libx264",
                             preset: str = "fast") -> List[str]:
        """
        Split video into separate files for each scene using FFmpeg.

        Args:
            video_path: Path to input video
            output_dir: Output directory
            scenes: Pre-detected scenes (None = detect now)
            codec: Video codec
            preset: Encoding preset

        Returns:
            List of output file paths
        """
        if scenes is None:
            scenes = self.detect_scenes(video_path)

        if not scenes:
            logger.warning(f"No scenes detected in {video_path}")
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_name = Path(video_path).stem
        output_files = []

        try:
            import subprocess

            for scene in scenes:
                scene_num = scene['scene_number']
                start_time = scene['start_time']
                duration = scene['duration']

                output_file = output_dir / f"{video_name}_scene{scene_num:03d}.mp4"

                # FFmpeg command to extract scene
                cmd = [
                    'ffmpeg',
                    '-y',  # Overwrite output
                    '-i', str(video_path),
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-c:v', codec,
                    '-preset', preset,
                    '-c:a', 'copy',
                    str(output_file)
                ]

                subprocess.run(cmd, check=True, capture_output=True)
                output_files.append(str(output_file))

            logger.info(f"Split video into {len(output_files)} scene files")
            return output_files

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed: {e.stderr.decode()}")
            return []
        except FileNotFoundError:
            logger.error("FFmpeg not found. Please install FFmpeg.")
            return []


def detect_scenes(video_path: Union[str, Path],
                 threshold: float = 30.0,
                 min_scene_length: int = 15,
                 method: str = "content") -> List[Dict[str, Any]]:
    """
    Convenience function to detect scenes in a video.

    Args:
        video_path: Path to video
        threshold: Detection threshold
        min_scene_length: Minimum scene length in frames
        method: Detection method

    Returns:
        List of detected scenes
    """
    detector = SceneDetector(
        threshold=threshold,
        min_scene_length=min_scene_length,
        method=method
    )
    return detector.detect_scenes(video_path)


def extract_scene_thumbnails(video_path: Union[str, Path],
                             output_dir: Union[str, Path],
                             threshold: float = 30.0) -> List[str]:
    """
    Extract one thumbnail per scene.

    Args:
        video_path: Path to video
        output_dir: Output directory
        threshold: Detection threshold

    Returns:
        List of thumbnail paths
    """
    detector = SceneDetector(threshold=threshold)
    scenes = detector.detect_scenes(video_path)

    if not scenes:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))

        thumbnails = []
        video_name = Path(video_path).stem

        for scene in scenes:
            scene_num = scene['scene_number']
            # Extract frame from middle of scene
            mid_frame = (scene['start_frame'] + scene['end_frame']) // 2

            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, frame = cap.read()

            if not ret:
                continue

            output_path = output_dir / f"{video_name}_scene{scene_num:03d}_thumb.jpg"
            cv2.imwrite(str(output_path), frame)
            thumbnails.append(str(output_path))

        cap.release()
        return thumbnails

    except Exception as e:
        logger.error(f"Thumbnail extraction failed: {e}")
        return []
