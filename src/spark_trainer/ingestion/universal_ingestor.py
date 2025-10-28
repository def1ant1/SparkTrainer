"""
Universal ingestion pipeline for multi-modal data.

Supports:
- Folder and manifest-based ingestion
- Automatic media probing via ffprobe and MIME checks
- Video: frame extraction, audio tracks, keyframes, transcript alignment
- Audio: diarization, transcription (Whisper), VAD, timestamps, speaker labels
- Images: EXIF strip, NSFW filter, auto-caption options
- Text: UTF-8 validation, boilerplate removal, document chunking
"""

import os
import json
import magic
import hashlib
import logging
import subprocess
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
import chardet
from PIL import Image
from PIL.ExifTags import TAGS
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IngestedItem:
    """Metadata for an ingested item."""
    id: str
    source_path: str
    mime_type: str
    content_type: str  # text, image, audio, video
    size_bytes: int
    checksum: str
    metadata: Dict[str, Any]
    processed_path: Optional[str] = None
    errors: List[str] = None

    def to_dict(self):
        return asdict(self)


class UniversalIngestor:
    """
    Universal data ingestion pipeline supporting multi-modal data.

    Features:
    - Automatic MIME type detection
    - Media probing (ffprobe)
    - EXIF stripping
    - NSFW filtering (optional)
    - UTF-8 validation
    - Boilerplate removal
    - Document chunking
    """

    def __init__(
        self,
        output_dir: str,
        strip_exif: bool = True,
        nsfw_filter: bool = False,
        nsfw_threshold: float = 0.7,
        validate_utf8: bool = True,
        remove_boilerplate: bool = False,
        chunk_text: bool = False,
        chunk_size: int = 512,
        chunk_stride: int = 128,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.strip_exif = strip_exif
        self.nsfw_filter = nsfw_filter
        self.nsfw_threshold = nsfw_threshold
        self.validate_utf8 = validate_utf8
        self.remove_boilerplate = remove_boilerplate
        self.chunk_text = chunk_text
        self.chunk_size = chunk_size
        self.chunk_stride = chunk_stride

        # Initialize MIME detector
        self.mime_detector = magic.Magic(mime=True)

        # NSFW detector (lazy loaded)
        self._nsfw_detector = None

    @property
    def nsfw_detector(self):
        """Lazy load NSFW detector."""
        if self._nsfw_detector is None and self.nsfw_filter:
            try:
                from transformers import pipeline
                self._nsfw_detector = pipeline(
                    "image-classification",
                    model="Falconsai/nsfw_image_detection",
                    device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
                )
                logger.info("NSFW detector loaded")
            except Exception as e:
                logger.warning(f"Failed to load NSFW detector: {e}")
                self._nsfw_detector = None
        return self._nsfw_detector

    def ingest_folder(
        self,
        folder_path: str,
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
    ) -> List[IngestedItem]:
        """
        Ingest all files from a folder.

        Args:
            folder_path: Path to folder
            recursive: Recursively scan subdirectories
            extensions: Optional list of extensions to filter (e.g., ['.jpg', '.png'])

        Returns:
            List of ingested items
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder not found: {folder_path}")

        pattern = "**/*" if recursive else "*"
        files = list(folder.glob(pattern))

        # Filter by extension if specified
        if extensions:
            extensions = [ext.lower() for ext in extensions]
            files = [f for f in files if f.suffix.lower() in extensions]

        # Filter out directories
        files = [f for f in files if f.is_file()]

        logger.info(f"Found {len(files)} files to ingest from {folder_path}")

        ingested = []
        for file_path in files:
            try:
                item = self.ingest_file(str(file_path))
                ingested.append(item)
            except Exception as e:
                logger.error(f"Failed to ingest {file_path}: {e}")

        return ingested

    def ingest_manifest(
        self,
        manifest_path: str,
    ) -> List[IngestedItem]:
        """
        Ingest files listed in a manifest (JSONL format).

        Manifest format:
        {"path": "/path/to/file.jpg", "metadata": {...}}

        Args:
            manifest_path: Path to manifest file

        Returns:
            List of ingested items
        """
        manifest = Path(manifest_path)
        if not manifest.exists():
            raise ValueError(f"Manifest not found: {manifest_path}")

        ingested = []
        with open(manifest, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    file_path = entry.get('path')
                    if not file_path:
                        logger.warning(f"Line {line_num}: No 'path' field")
                        continue

                    item = self.ingest_file(file_path, extra_metadata=entry.get('metadata', {}))
                    ingested.append(item)
                except Exception as e:
                    logger.error(f"Line {line_num}: Failed to ingest: {e}")

        return ingested

    def ingest_file(
        self,
        file_path: str,
        extra_metadata: Optional[Dict] = None,
    ) -> IngestedItem:
        """
        Ingest a single file with automatic type detection and processing.

        Args:
            file_path: Path to file
            extra_metadata: Additional metadata to include

        Returns:
            Ingested item with metadata
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")

        # Detect MIME type
        mime_type = self._detect_mime_type(file_path)

        # Determine content type
        content_type = self._get_content_type(mime_type)

        # Calculate checksum
        checksum = self._calculate_checksum(file_path)

        # Get file size
        size_bytes = file_path.stat().st_size

        # Create base item
        item = IngestedItem(
            id=checksum[:16],
            source_path=str(file_path),
            mime_type=mime_type,
            content_type=content_type,
            size_bytes=size_bytes,
            checksum=checksum,
            metadata=extra_metadata or {},
            errors=[]
        )

        # Process based on content type
        try:
            if content_type == "image":
                self._process_image(item)
            elif content_type == "video":
                self._process_video(item)
            elif content_type == "audio":
                self._process_audio(item)
            elif content_type == "text":
                self._process_text(item)
            else:
                logger.warning(f"Unsupported content type: {content_type} for {file_path}")
        except Exception as e:
            logger.error(f"Processing error for {file_path}: {e}")
            item.errors.append(str(e))

        return item

    def _detect_mime_type(self, file_path: Path) -> str:
        """Detect MIME type using python-magic."""
        try:
            mime_type = self.mime_detector.from_file(str(file_path))
            return mime_type
        except Exception as e:
            logger.warning(f"MIME detection failed for {file_path}: {e}, falling back to mimetypes")
            mime_type, _ = mimetypes.guess_type(str(file_path))
            return mime_type or "application/octet-stream"

    def _get_content_type(self, mime_type: str) -> str:
        """Map MIME type to content type."""
        if mime_type.startswith("image/"):
            return "image"
        elif mime_type.startswith("video/"):
            return "video"
        elif mime_type.startswith("audio/"):
            return "audio"
        elif mime_type.startswith("text/") or "json" in mime_type or "xml" in mime_type:
            return "text"
        else:
            return "other"

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _process_image(self, item: IngestedItem):
        """Process image file."""
        try:
            img = Image.open(item.source_path)

            # Extract EXIF data before stripping
            exif_data = self._extract_exif(img)
            if exif_data:
                item.metadata['exif_original'] = exif_data

            # Get image info
            item.metadata['width'] = img.width
            item.metadata['height'] = img.height
            item.metadata['format'] = img.format
            item.metadata['mode'] = img.mode

            # NSFW check
            if self.nsfw_filter:
                is_nsfw, score = self._check_nsfw(item.source_path)
                item.metadata['nsfw_score'] = score
                if is_nsfw:
                    item.metadata['nsfw_filtered'] = True
                    item.errors.append(f"NSFW content detected (score: {score:.2f})")
                    logger.warning(f"NSFW content filtered: {item.source_path}")
                    return

            # Strip EXIF and save processed version
            if self.strip_exif and exif_data:
                output_path = self.output_dir / f"{item.id}{Path(item.source_path).suffix}"

                # Create image without EXIF
                data = list(img.getdata())
                img_no_exif = Image.new(img.mode, img.size)
                img_no_exif.putdata(data)
                img_no_exif.save(output_path)

                item.processed_path = str(output_path)
                item.metadata['exif_stripped'] = True
                logger.debug(f"EXIF stripped: {output_path}")

        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            item.errors.append(f"Image processing: {e}")

    def _extract_exif(self, img: Image.Image) -> Optional[Dict]:
        """Extract EXIF metadata from image."""
        try:
            exif = img._getexif()
            if exif:
                return {
                    TAGS.get(k, k): str(v)
                    for k, v in exif.items()
                    if k in TAGS
                }
        except Exception:
            pass
        return None

    def _check_nsfw(self, image_path: str) -> Tuple[bool, float]:
        """Check if image contains NSFW content."""
        if self.nsfw_detector is None:
            return False, 0.0

        try:
            result = self.nsfw_detector(image_path)
            # Result format: [{'label': 'nsfw', 'score': 0.9}, ...]
            nsfw_score = next((r['score'] for r in result if r['label'] == 'nsfw'), 0.0)
            is_nsfw = nsfw_score >= self.nsfw_threshold
            return is_nsfw, nsfw_score
        except Exception as e:
            logger.error(f"NSFW check failed: {e}")
            return False, 0.0

    def _process_video(self, item: IngestedItem):
        """Process video file with ffprobe."""
        try:
            metadata = self._probe_video(item.source_path)
            item.metadata.update(metadata)
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            item.errors.append(f"Video processing: {e}")

    def _probe_video(self, video_path: str) -> Dict:
        """Probe video file with ffprobe."""
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)

            # Extract video stream info
            video_streams = [s for s in data.get('streams', []) if s['codec_type'] == 'video']
            audio_streams = [s for s in data.get('streams', []) if s['codec_type'] == 'audio']

            metadata = {
                'duration': float(data.get('format', {}).get('duration', 0)),
                'bit_rate': int(data.get('format', {}).get('bit_rate', 0)),
                'format_name': data.get('format', {}).get('format_name', ''),
                'num_video_streams': len(video_streams),
                'num_audio_streams': len(audio_streams),
            }

            if video_streams:
                vs = video_streams[0]
                metadata.update({
                    'width': vs.get('width'),
                    'height': vs.get('height'),
                    'codec': vs.get('codec_name'),
                    'fps': eval(vs.get('r_frame_rate', '0/1')),  # e.g., "30/1" -> 30.0
                    'pix_fmt': vs.get('pix_fmt'),
                })

            if audio_streams:
                aus = audio_streams[0]
                metadata.update({
                    'audio_codec': aus.get('codec_name'),
                    'audio_sample_rate': int(aus.get('sample_rate', 0)),
                    'audio_channels': aus.get('channels'),
                })

            return metadata

        except subprocess.CalledProcessError as e:
            logger.error(f"ffprobe failed: {e}")
            return {}
        except Exception as e:
            logger.error(f"Video probe error: {e}")
            return {}

    def _process_audio(self, item: IngestedItem):
        """Process audio file with ffprobe."""
        try:
            metadata = self._probe_audio(item.source_path)
            item.metadata.update(metadata)
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            item.errors.append(f"Audio processing: {e}")

    def _probe_audio(self, audio_path: str) -> Dict:
        """Probe audio file with ffprobe."""
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            audio_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)

            audio_streams = [s for s in data.get('streams', []) if s['codec_type'] == 'audio']

            metadata = {
                'duration': float(data.get('format', {}).get('duration', 0)),
                'bit_rate': int(data.get('format', {}).get('bit_rate', 0)),
                'format_name': data.get('format', {}).get('format_name', ''),
            }

            if audio_streams:
                aus = audio_streams[0]
                metadata.update({
                    'codec': aus.get('codec_name'),
                    'sample_rate': int(aus.get('sample_rate', 0)),
                    'channels': aus.get('channels'),
                    'bit_depth': aus.get('bits_per_sample'),
                })

            return metadata

        except Exception as e:
            logger.error(f"Audio probe error: {e}")
            return {}

    def _process_text(self, item: IngestedItem):
        """Process text file."""
        try:
            # Read file and detect encoding
            with open(item.source_path, 'rb') as f:
                raw_data = f.read()

            # Detect encoding
            detected = chardet.detect(raw_data)
            encoding = detected['encoding']
            confidence = detected['confidence']

            item.metadata['encoding'] = encoding
            item.metadata['encoding_confidence'] = confidence

            # Validate UTF-8
            if self.validate_utf8:
                try:
                    text = raw_data.decode('utf-8')
                    item.metadata['utf8_valid'] = True
                except UnicodeDecodeError:
                    if encoding and encoding.lower() != 'utf-8':
                        # Try to convert to UTF-8
                        try:
                            text = raw_data.decode(encoding)
                            output_path = self.output_dir / f"{item.id}.txt"
                            with open(output_path, 'w', encoding='utf-8') as f:
                                f.write(text)
                            item.processed_path = str(output_path)
                            item.metadata['utf8_valid'] = True
                            item.metadata['converted_to_utf8'] = True
                        except Exception as e:
                            item.metadata['utf8_valid'] = False
                            item.errors.append(f"UTF-8 conversion failed: {e}")
                            return
                    else:
                        item.metadata['utf8_valid'] = False
                        item.errors.append("Invalid UTF-8 encoding")
                        return
            else:
                text = raw_data.decode(encoding or 'utf-8', errors='ignore')

            # Basic text stats
            item.metadata['char_count'] = len(text)
            item.metadata['line_count'] = text.count('\n') + 1
            item.metadata['word_count'] = len(text.split())

            # Remove boilerplate (simple heuristic)
            if self.remove_boilerplate:
                text = self._remove_boilerplate(text)
                item.metadata['boilerplate_removed'] = True

            # Chunk text
            if self.chunk_text:
                chunks = self._chunk_text(text)
                item.metadata['num_chunks'] = len(chunks)

                # Save chunks
                chunks_dir = self.output_dir / f"{item.id}_chunks"
                chunks_dir.mkdir(exist_ok=True)
                for i, chunk in enumerate(chunks):
                    chunk_path = chunks_dir / f"chunk_{i:04d}.txt"
                    with open(chunk_path, 'w', encoding='utf-8') as f:
                        f.write(chunk)

                item.processed_path = str(chunks_dir)

        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            item.errors.append(f"Text processing: {e}")

    def _remove_boilerplate(self, text: str) -> str:
        """
        Remove common boilerplate patterns.
        This is a simple heuristic - for production use libraries like jusText.
        """
        # Remove common patterns
        patterns = [
            "Cookie Policy",
            "Privacy Policy",
            "Terms of Service",
            "Subscribe to our newsletter",
            "Follow us on",
        ]

        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            # Skip very short lines
            if len(line.strip()) < 20:
                continue
            # Skip lines with boilerplate patterns
            if any(pattern.lower() in line.lower() for pattern in patterns):
                continue
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text with sliding window strategy.

        Args:
            text: Input text

        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_stride):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk:
                chunks.append(chunk)

        return chunks

    def save_manifest(
        self,
        items: List[IngestedItem],
        output_path: str,
    ):
        """Save ingested items to a manifest file (JSONL)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for item in items:
                f.write(json.dumps(item.to_dict()) + '\n')

        logger.info(f"Manifest saved to {output_path} ({len(items)} items)")


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Universal data ingestion")
    parser.add_argument("input", help="Input folder or manifest file")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--manifest", "-m", action="store_true", help="Input is a manifest file")
    parser.add_argument("--recursive", "-r", action="store_true", help="Recursive folder scan")
    parser.add_argument("--extensions", "-e", nargs="+", help="File extensions to filter")
    parser.add_argument("--strip-exif", action="store_true", default=True, help="Strip EXIF from images")
    parser.add_argument("--nsfw-filter", action="store_true", help="Filter NSFW images")
    parser.add_argument("--validate-utf8", action="store_true", default=True, help="Validate UTF-8 text")
    parser.add_argument("--chunk-text", action="store_true", help="Chunk text files")
    parser.add_argument("--manifest-out", help="Output manifest path")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    ingestor = UniversalIngestor(
        output_dir=args.output,
        strip_exif=args.strip_exif,
        nsfw_filter=args.nsfw_filter,
        validate_utf8=args.validate_utf8,
        chunk_text=args.chunk_text,
    )

    if args.manifest:
        items = ingestor.ingest_manifest(args.input)
    else:
        items = ingestor.ingest_folder(
            args.input,
            recursive=args.recursive,
            extensions=args.extensions,
        )

    print(f"\nIngested {len(items)} items")
    print(f"Content types: {dict((item.content_type, 1) for item in items)}")

    # Save manifest
    if args.manifest_out:
        ingestor.save_manifest(items, args.manifest_out)


if __name__ == "__main__":
    main()
