"""
Multimodal Dataset Creation Pipeline for Apotheon

Automated pipeline for creating multimodal datasets from videos:
1. Video ingestion
2. Frame extraction
3. Audio extraction
4. Image captioning (BLIP-2, Florence-2)
5. Audio transcription (Whisper)
6. Scene detection (PySceneDetect)
7. Manifest generation

This pipeline implements WORK ITEM 3 requirements.
"""

import os
import json
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm
import logging

# Optional imports (will be imported when needed)
try:
    from PIL import Image
    import numpy as np
except ImportError:
    Image = None
    np = None

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MultimodalPipelineConfig:
    """Configuration for multimodal dataset pipeline"""
    input_videos_dir: str
    output_dir: str = "datasets/multimodal"
    manifest_path: str = "datasets/multimodal/manifest.jsonl"

    # Frame extraction
    fps: int = 1
    resolution: Tuple[int, int] = (224, 224)
    frame_format: str = "jpg"

    # Audio extraction
    audio_sample_rate: int = 16000
    audio_format: str = "wav"

    # Captioning
    caption_backend: str = "blip2"  # blip2, florence2, blip
    caption_batch_size: int = 4

    # Transcription
    transcription_backend: str = "whisper"  # whisper
    transcription_model_size: str = "base"

    # Scene detection
    enable_scene_detection: bool = True
    scene_threshold: float = 27.0

    # Smart sampling (WORK ITEM 3)
    enable_smart_sampling: bool = True
    motion_threshold: float = 0.3
    use_perceptual_hashing: bool = True
    hash_size: int = 8

    # Metadata provenance (WORK ITEM 3)
    enable_provenance_tracking: bool = True

    # Quality control
    min_video_duration: float = 1.0
    max_video_duration: float = 600.0


# =============================================================================
# Utility Functions
# =============================================================================

def get_video_hash(video_path: str) -> str:
    """Generate SHA256 hash for video file"""
    sha256 = hashlib.sha256()
    with open(video_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


def get_video_info(video_path: str) -> Dict[str, Any]:
    """Extract video metadata using ffprobe"""
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
        info = json.loads(result.stdout)

        # Extract relevant info
        format_info = info.get('format', {})
        video_stream = next((s for s in info.get('streams', []) if s['codec_type'] == 'video'), {})

        return {
            'duration': float(format_info.get('duration', 0)),
            'size_bytes': int(format_info.get('size', 0)),
            'bitrate': int(format_info.get('bit_rate', 0)),
            'width': int(video_stream.get('width', 0)),
            'height': int(video_stream.get('height', 0)),
            'fps': eval(video_stream.get('r_frame_rate', '0/1')),
            'codec': video_stream.get('codec_name', 'unknown')
        }
    except Exception as e:
        logger.error(f"Failed to get video info for {video_path}: {e}")
        return {}


def perceptual_hash(image_path: str, hash_size: int = 8) -> str:
    """
    Compute perceptual hash (pHash) for image similarity detection
    WORK ITEM 3: Smart sampling with perceptual hashing
    """
    if Image is None or np is None:
        logger.warning("PIL and numpy required for perceptual hashing")
        return ""

    try:
        img = Image.open(image_path).convert('L')
        img = img.resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        pixels = np.array(img).flatten()

        # Compute DCT (discrete cosine transform) approximation
        avg = pixels.mean()
        diff = pixels > avg
        return ''.join(['1' if d else '0' for d in diff])
    except Exception as e:
        logger.error(f"Failed to compute perceptual hash for {image_path}: {e}")
        return ""


def compute_motion_score(frame1_path: str, frame2_path: str) -> float:
    """
    Compute motion score between two frames using pixel difference
    WORK ITEM 3: Motion-aware FPS sampling
    """
    if Image is None or np is None:
        return 0.0

    try:
        img1 = np.array(Image.open(frame1_path).convert('L'))
        img2 = np.array(Image.open(frame2_path).convert('L'))
        diff = np.abs(img1.astype(float) - img2.astype(float))
        motion_score = diff.mean() / 255.0
        return motion_score
    except Exception as e:
        logger.error(f"Failed to compute motion score: {e}")
        return 0.0


# =============================================================================
# Pipeline Steps
# =============================================================================

class MultimodalPipeline:
    """Main pipeline for multimodal dataset creation"""

    def __init__(self, config: MultimodalPipelineConfig):
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)

        # Initialize backends
        self.caption_model = None
        self.transcription_model = None

    def _get_caption_backend(self):
        """Lazy load caption model"""
        if self.caption_model is None:
            logger.info(f"Loading caption model: {self.config.caption_backend}")

            if self.config.caption_backend == 'blip2':
                from transformers import Blip2Processor, Blip2ForConditionalGeneration
                self.caption_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
                self.caption_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
            elif self.config.caption_backend == 'florence2':
                from transformers import AutoProcessor, AutoModelForCausalLM
                self.caption_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base")
                self.caption_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base")
            elif self.config.caption_backend == 'blip':
                from transformers import BlipProcessor, BlipForConditionalGeneration
                self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

            # Move to GPU if available
            import torch
            if torch.cuda.is_available():
                self.caption_model = self.caption_model.to('cuda')

        return self.caption_model, self.caption_processor

    def _get_transcription_backend(self):
        """Lazy load transcription model"""
        if self.transcription_model is None:
            logger.info(f"Loading transcription model: Whisper-{self.config.transcription_model_size}")

            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            model_name = f"openai/whisper-{self.config.transcription_model_size}"
            self.transcription_processor = WhisperProcessor.from_pretrained(model_name)
            self.transcription_model = WhisperForConditionalGeneration.from_pretrained(model_name)

            # Move to GPU if available
            import torch
            if torch.cuda.is_available():
                self.transcription_model = self.transcription_model.to('cuda')

        return self.transcription_model, self.transcription_processor

    def extract_frames(self, video_path: str, output_dir: str) -> List[str]:
        """
        Extract frames from video
        WORK ITEM 3: Smart sampling with motion-aware FPS
        """
        os.makedirs(output_dir, exist_ok=True)

        # Extract all frames first
        temp_output = os.path.join(output_dir, 'frame_%06d.jpg')
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'fps={self.config.fps},scale={self.config.resolution[0]}:{self.config.resolution[1]}',
            '-q:v', '2',
            temp_output,
            '-y'
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Frame extraction failed: {e.stderr.decode()}")
            return []

        # Get extracted frames
        frames = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.jpg')])

        # Smart sampling based on motion
        if self.config.enable_smart_sampling and len(frames) > 1:
            selected_frames = [frames[0]]  # Always keep first frame

            for i in range(1, len(frames)):
                motion_score = compute_motion_score(frames[i-1], frames[i])

                if motion_score > self.config.motion_threshold:
                    selected_frames.append(frames[i])
                else:
                    # Remove low-motion frame
                    if os.path.exists(frames[i]):
                        os.remove(frames[i])

            logger.info(f"Smart sampling: {len(frames)} → {len(selected_frames)} frames")
            return selected_frames

        return frames

    def extract_audio(self, video_path: str, output_path: str) -> bool:
        """Extract audio from video"""
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',
            '-ar', str(self.config.audio_sample_rate),
            '-ac', '1',  # Mono
            output_path,
            '-y'
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return os.path.exists(output_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"Audio extraction failed: {e.stderr.decode()}")
            return False

    def generate_captions(self, frame_paths: List[str]) -> List[str]:
        """
        Generate captions for frames using BLIP-2/Florence-2
        WORK ITEM 3: Auto-label pipeline with BLIP/Florence
        """
        model, processor = self._get_caption_backend()
        captions = []

        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for i in range(0, len(frame_paths), self.config.caption_batch_size):
            batch_paths = frame_paths[i:i + self.config.caption_batch_size]
            images = [Image.open(p).convert('RGB') for p in batch_paths]

            inputs = processor(images=images, return_tensors='pt').to(device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=50)

            batch_captions = processor.batch_decode(outputs, skip_special_tokens=True)
            captions.extend(batch_captions)

        return captions

    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio using Whisper
        WORK ITEM 3: Auto-label pipeline with Whisper
        """
        model, processor = self._get_transcription_backend()

        import torch
        import librosa

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load audio
        audio_array, sr = librosa.load(audio_path, sr=self.config.audio_sample_rate)

        # Process
        inputs = processor(audio_array, sampling_rate=self.config.audio_sample_rate, return_tensors='pt').to(device)

        with torch.no_grad():
            predicted_ids = model.generate(**inputs)

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription

    def detect_scenes(self, video_path: str) -> List[Dict[str, float]]:
        """
        Detect scene boundaries using PySceneDetect
        WORK ITEM 3: Scene detection using PySceneDetect
        """
        if not self.config.enable_scene_detection:
            return []

        try:
            from scenedetect import VideoManager, SceneManager
            from scenedetect.detectors import ContentDetector

            video_manager = VideoManager([video_path])
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=self.config.scene_threshold))

            video_manager.start()
            scene_manager.detect_scenes(frame_source=video_manager)
            scene_list = scene_manager.get_scene_list()
            video_manager.release()

            scenes = []
            for scene in scene_list:
                scenes.append({
                    'start_time': scene[0].get_seconds(),
                    'end_time': scene[1].get_seconds()
                })

            logger.info(f"Detected {len(scenes)} scenes in {video_path}")
            return scenes

        except ImportError:
            logger.warning("PySceneDetect not installed. Install with: pip install scenedetect[opencv]")
            return []
        except Exception as e:
            logger.error(f"Scene detection failed: {e}")
            return []

    def create_provenance_metadata(self, video_path: str, video_hash: str) -> Dict[str, Any]:
        """
        Create metadata provenance with SHA-based lineage tracking
        WORK ITEM 3: Metadata provenance with SHA-based lineage tracking
        """
        import time

        provenance = {
            'source_file': os.path.basename(video_path),
            'source_hash': video_hash,
            'processing_timestamp': time.time(),
            'pipeline_version': '1.0',
            'config': {
                'fps': self.config.fps,
                'resolution': self.config.resolution,
                'caption_backend': self.config.caption_backend,
                'transcription_backend': self.config.transcription_backend,
                'smart_sampling': self.config.enable_smart_sampling,
                'scene_detection': self.config.enable_scene_detection
            },
            'lineage': {
                'parent_hash': video_hash,
                'processing_steps': [
                    'frame_extraction',
                    'audio_extraction',
                    'captioning',
                    'transcription',
                    'scene_detection'
                ]
            }
        }

        return provenance

    def process_video(self, video_path: str) -> Optional[Dict[str, Any]]:
        """Process a single video through the pipeline"""
        logger.info(f"Processing video: {video_path}")

        # Get video info
        video_info = get_video_info(video_path)

        # Check duration constraints
        duration = video_info.get('duration', 0)
        if duration < self.config.min_video_duration or duration > self.config.max_video_duration:
            logger.warning(f"Skipping video {video_path}: duration {duration}s out of range")
            return None

        # Generate video hash
        video_hash = get_video_hash(video_path)

        # Create output directories
        video_output_dir = os.path.join(self.config.output_dir, video_hash[:2], video_hash)
        frames_dir = os.path.join(video_output_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)

        # Step 1: Extract frames
        logger.info("Extracting frames...")
        frame_paths = self.extract_frames(video_path, frames_dir)
        if not frame_paths:
            logger.error(f"No frames extracted from {video_path}")
            return None

        # Step 2: Extract audio
        logger.info("Extracting audio...")
        audio_path = os.path.join(video_output_dir, f'audio.{self.config.audio_format}')
        has_audio = self.extract_audio(video_path, audio_path)

        # Step 3: Generate captions
        logger.info("Generating captions...")
        captions = self.generate_captions(frame_paths)

        # Step 4: Transcribe audio
        transcript = ""
        if has_audio:
            logger.info("Transcribing audio...")
            transcript = self.transcribe_audio(audio_path)

        # Step 5: Detect scenes
        scenes = self.detect_scenes(video_path)

        # Step 6: Create provenance metadata
        provenance = self.create_provenance_metadata(video_path, video_hash)

        # Create manifest entry
        manifest_entry = {
            'id': video_hash,
            'frames_dir': frames_dir,
            'audio': audio_path if has_audio else None,
            'meta': {
                'source_path': video_path,
                'duration': duration,
                'num_frames': len(frame_paths),
                'caption': ' '.join(captions[:3]),  # First 3 captions
                'all_captions': captions,
                'transcript': transcript,
                'scenes': scenes,
                'video_info': video_info,
                'provenance': provenance if self.config.enable_provenance_tracking else None
            }
        }

        return manifest_entry

    def run(self):
        """Run the full pipeline"""
        logger.info("Starting multimodal dataset creation pipeline")

        # Find all videos
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        video_files = []

        for ext in video_extensions:
            video_files.extend(Path(self.config.input_videos_dir).glob(f'*{ext}'))
            video_files.extend(Path(self.config.input_videos_dir).glob(f'**/*{ext}'))

        video_files = list(set(video_files))  # Remove duplicates
        logger.info(f"Found {len(video_files)} videos to process")

        # Process videos
        manifest_entries = []

        for video_path in tqdm(video_files, desc="Processing videos"):
            try:
                entry = self.process_video(str(video_path))
                if entry:
                    manifest_entries.append(entry)
            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")
                continue

        # Write manifest
        os.makedirs(os.path.dirname(self.config.manifest_path), exist_ok=True)
        with open(self.config.manifest_path, 'w') as f:
            for entry in manifest_entries:
                f.write(json.dumps(entry) + '\n')

        logger.info(f"Pipeline complete! Created manifest with {len(manifest_entries)} entries")
        logger.info(f"Manifest saved to: {self.config.manifest_path}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Multimodal Dataset Creation Pipeline")
    parser.add_argument('--input', required=True, help='Input videos directory')
    parser.add_argument('--output', default='datasets/multimodal', help='Output directory')
    parser.add_argument('--fps', type=int, default=1, help='Frames per second to extract')
    parser.add_argument('--caption-backend', choices=['blip2', 'florence2', 'blip'], default='blip2')
    parser.add_argument('--no-smart-sampling', action='store_true', help='Disable smart sampling')
    parser.add_argument('--no-scene-detection', action='store_true', help='Disable scene detection')

    args = parser.parse_args()

    config = MultimodalPipelineConfig(
        input_videos_dir=args.input,
        output_dir=args.output,
        fps=args.fps,
        caption_backend=args.caption_backend,
        enable_smart_sampling=not args.no_smart_sampling,
        enable_scene_detection=not args.no_scene_detection
    )

    pipeline = MultimodalPipeline(config)
    pipeline.run()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
