#!/usr/bin/env python
"""
Benchmark for video processing throughput.

This script benchmarks the performance of video frame extraction,
scene detection, and other video processing operations.
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict, Any
import time

from benchmarks.base import BenchmarkRunner


class VideoProcessingBenchmark(BenchmarkRunner):
    """
    Benchmark video processing operations.

    Measures throughput and latency of:
    - Frame extraction
    - Scene detection
    - Video validation
    """

    def __init__(self, video_paths: List[str], output_dir: str):
        """
        Initialize video processing benchmark.

        Args:
            video_paths: List of paths to video files
            output_dir: Directory for output frames
        """
        super().__init__("video_processing")
        self.video_paths = video_paths
        self.output_dir = output_dir

    def run(self) -> Dict[str, Any]:
        """
        Run the video processing benchmark.

        Returns:
            Dictionary with benchmark results
        """
        results = {
            "num_videos": len(self.video_paths),
            "frame_extraction": self.benchmark_frame_extraction(),
            "scene_detection": self.benchmark_scene_detection(),
        }

        self.print_results(results)
        return results

    def benchmark_frame_extraction(self) -> Dict[str, Any]:
        """
        Benchmark frame extraction performance.

        Returns:
            Dictionary with throughput and latency metrics
        """
        print("Benchmarking frame extraction...")

        try:
            from spark_trainer.utils.video import extract_frames

            times = []
            total_frames = 0

            for video_path in self.video_paths:
                output_path = os.path.join(self.output_dir, Path(video_path).stem)
                os.makedirs(output_path, exist_ok=True)

                start = time.perf_counter()
                frames = extract_frames(video_path, output_path, fps=1)
                elapsed = time.perf_counter() - start

                times.append(elapsed)
                total_frames += len(frames)

            stats = self.compute_statistics(times)
            stats["total_frames"] = total_frames
            stats["avg_fps"] = total_frames / sum(times) if times else 0

            return stats
        except ImportError:
            return {"error": "Video processing utilities not available"}

    def benchmark_scene_detection(self) -> Dict[str, Any]:
        """
        Benchmark scene detection performance.

        Returns:
            Dictionary with throughput and latency metrics
        """
        print("Benchmarking scene detection...")

        try:
            from spark_trainer.scene_detection import detect_scenes

            times = []
            total_scenes = 0

            for video_path in self.video_paths:
                start = time.perf_counter()
                scenes = detect_scenes(video_path)
                elapsed = time.perf_counter() - start

                times.append(elapsed)
                total_scenes += len(scenes)

            stats = self.compute_statistics(times)
            stats["total_scenes"] = total_scenes
            stats["avg_scenes_per_video"] = total_scenes / len(self.video_paths)

            return stats
        except ImportError:
            return {"error": "Scene detection not available"}


def main():
    """Run the video processing benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark video processing operations")
    parser.add_argument("--input", required=True, help="Directory containing video files")
    parser.add_argument("--output", default="./benchmark_output", help="Output directory for frames")
    parser.add_argument("--max-videos", type=int, default=10, help="Maximum number of videos to process")
    parser.add_argument("--save-results", default=None, help="Path to save results JSON")

    args = parser.parse_args()

    # Find video files
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    video_paths = []

    for ext in video_extensions:
        video_paths.extend(Path(args.input).glob(f"*{ext}"))

    video_paths = [str(p) for p in video_paths[: args.max_videos]]

    if not video_paths:
        print(f"No video files found in {args.input}")
        return

    print(f"Found {len(video_paths)} video files")

    # Run benchmark
    benchmark = VideoProcessingBenchmark(video_paths, args.output)
    results = benchmark.run()

    # Save results
    if args.save_results:
        benchmark.save_results(results, args.save_results)
    else:
        benchmark.save_results(results)


if __name__ == "__main__":
    main()
