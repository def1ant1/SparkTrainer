"""
Dataset Ingestion Service with JSONL Manifest Standardization.

Provides:
- Standardized JSONL manifest format
- Multi-modal ingestion workers (video, audio, image, text)
- Metadata extraction (EXIF, duration, dimensions)
- Quality gates integration
- Dataset versioning integration
- Auto-generated dataset cards
"""

import os
import json
import logging
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import shutil

logger = logging.getLogger(__name__)


@dataclass
class ManifestEntry:
    """
    Standardized JSONL manifest entry.

    Schema:
    {
        "id": "unique_id",
        "source_path": "/path/to/original",
        "processed_path": "/path/to/processed",
        "content_type": "image|video|audio|text|multimodal",
        "mime_type": "image/jpeg",
        "size_bytes": 1024000,
        "checksum": "sha256:...",
        "metadata": {...},
        "annotations": {...},
        "quality_scores": {...},
        "created_at": "2024-01-01T00:00:00",
        "lineage": {...}
    }
    """
    id: str
    source_path: str
    processed_path: Optional[str]
    content_type: str
    mime_type: str
    size_bytes: int
    checksum: str
    metadata: Dict[str, Any]
    annotations: Optional[Dict[str, Any]] = None
    quality_scores: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    lineage: Optional[Dict[str, Any]] = None

    def to_jsonl(self) -> str:
        """Convert to JSONL line."""
        data = asdict(self)
        return json.dumps(data)

    @classmethod
    def from_jsonl(cls, line: str) -> 'ManifestEntry':
        """Create from JSONL line."""
        data = json.loads(line)
        return cls(**data)


class DatasetIngestionService:
    """
    Dataset ingestion service with standardized JSONL manifests.
    """

    def __init__(
        self,
        output_dir: str,
        quality_gates: Optional[List] = None,
        enable_versioning: bool = True,
        lakefs_client: Optional[Any] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.quality_gates = quality_gates or []
        self.enable_versioning = enable_versioning
        self.lakefs_client = lakefs_client

        # Ingestion workers
        self.workers = {
            'video': VideoIngestionWorker(),
            'audio': AudioIngestionWorker(),
            'image': ImageIngestionWorker(),
            'text': TextIngestionWorker(),
        }

        logger.info("Dataset ingestion service initialized")

    def ingest_dataset(
        self,
        dataset_name: str,
        source_paths: List[str],
        content_type: str = "auto",  # auto, video, audio, image, text
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Ingest dataset and generate JSONL manifest.

        Args:
            dataset_name: Dataset name
            source_paths: List of source file/directory paths
            content_type: Content type (auto-detect if not specified)
            metadata: Additional metadata

        Returns:
            Path to manifest file
        """
        logger.info(f"Starting ingestion: {dataset_name}")

        # Create dataset directory
        dataset_dir = self.output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        processed_dir = dataset_dir / "processed"
        processed_dir.mkdir(exist_ok=True)

        manifest_path = dataset_dir / "manifest.jsonl"

        # Collect all files
        all_files = []
        for source_path in source_paths:
            source_path = Path(source_path)

            if source_path.is_file():
                all_files.append(source_path)
            elif source_path.is_dir():
                all_files.extend(source_path.rglob("*"))

        # Filter to files only
        all_files = [f for f in all_files if f.is_file()]

        logger.info(f"Found {len(all_files)} files to ingest")

        # Ingest files
        manifest_entries = []

        for idx, file_path in enumerate(all_files):
            try:
                entry = self._ingest_file(
                    file_path=file_path,
                    processed_dir=processed_dir,
                    content_type=content_type,
                    index=idx,
                )

                # Apply quality gates
                if self.quality_gates:
                    quality_results = self._apply_quality_gates(entry)
                    entry.quality_scores = quality_results

                    # Filter out failed entries
                    if not all(r['passed'] for r in quality_results.values()):
                        logger.warning(f"Quality gate failed for {file_path.name}")
                        continue

                manifest_entries.append(entry)

            except Exception as e:
                logger.error(f"Failed to ingest {file_path}: {e}")

        # Write manifest
        with open(manifest_path, 'w') as f:
            for entry in manifest_entries:
                f.write(entry.to_jsonl() + '\n')

        logger.info(f"Manifest written: {manifest_path} ({len(manifest_entries)} entries)")

        # Generate dataset card
        self._generate_dataset_card(
            dataset_name=dataset_name,
            manifest_path=manifest_path,
            metadata=metadata,
        )

        # Version dataset
        if self.enable_versioning and self.lakefs_client:
            self._version_dataset(dataset_name, dataset_dir)

        return str(manifest_path)

    def _ingest_file(
        self,
        file_path: Path,
        processed_dir: Path,
        content_type: str,
        index: int,
    ) -> ManifestEntry:
        """Ingest a single file."""
        # Auto-detect content type
        if content_type == "auto":
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type:
                if mime_type.startswith('image/'):
                    content_type = 'image'
                elif mime_type.startswith('video/'):
                    content_type = 'video'
                elif mime_type.startswith('audio/'):
                    content_type = 'audio'
                elif mime_type.startswith('text/'):
                    content_type = 'text'
                else:
                    content_type = 'unknown'
            else:
                content_type = 'unknown'

        # Get appropriate worker
        worker = self.workers.get(content_type)

        if not worker:
            logger.warning(f"No worker for content type: {content_type}")
            # Generic file handling
            return self._ingest_generic_file(file_path, processed_dir, index)

        # Process with worker
        return worker.process(file_path, processed_dir, index)

    def _ingest_generic_file(
        self,
        file_path: Path,
        processed_dir: Path,
        index: int,
    ) -> ManifestEntry:
        """Ingest generic file."""
        # Copy to processed directory
        processed_path = processed_dir / f"{index:06d}_{file_path.name}"
        shutil.copy2(file_path, processed_path)

        # Get file stats
        file_size = file_path.stat().st_size

        # Compute checksum
        checksum = self._compute_checksum(file_path)

        # Get mime type
        mime_type, _ = mimetypes.guess_type(str(file_path))

        entry = ManifestEntry(
            id=f"{index:06d}",
            source_path=str(file_path),
            processed_path=str(processed_path),
            content_type='unknown',
            mime_type=mime_type or 'application/octet-stream',
            size_bytes=file_size,
            checksum=checksum,
            metadata={},
            created_at=datetime.now().isoformat(),
        )

        return entry

    def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum."""
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return f"sha256:{sha256_hash.hexdigest()}"

    def _apply_quality_gates(self, entry: ManifestEntry) -> Dict[str, Any]:
        """Apply quality gates to entry."""
        results = {}

        # Convert entry to dict for quality gates
        item = {
            'content_type': entry.content_type,
            'source_path': entry.source_path,
            'processed_path': entry.processed_path,
            'text': entry.metadata.get('text', ''),
            'metadata': entry.metadata,
        }

        for gate in self.quality_gates:
            result = gate.check(item)
            results[gate.__class__.__name__] = {
                'passed': result.passed,
                'score': result.score,
                'reason': result.reason,
            }

        return results

    def _generate_dataset_card(
        self,
        dataset_name: str,
        manifest_path: Path,
        metadata: Optional[Dict[str, Any]],
    ):
        """Generate dataset card."""
        from .dataset_cards import DatasetCardGenerator

        card_generator = DatasetCardGenerator(
            dataset_name=dataset_name,
            dataset_path=str(manifest_path.parent),
            description=metadata.get('description') if metadata else None,
            license=metadata.get('license') if metadata else None,
        )

        # Analyze manifest
        card_generator.analyze_manifest(str(manifest_path), split_name='train')

        # Add provenance
        if metadata:
            card_generator.add_provenance(
                sources=metadata.get('sources', []),
                processing_steps=metadata.get('processing_steps', []),
                tools={'spark_trainer': '1.0.0'},
            )

        # Generate card
        card_path = manifest_path.parent / "README.md"
        card_generator.save(str(card_path))

        logger.info(f"Dataset card generated: {card_path}")

    def _version_dataset(self, dataset_name: str, dataset_dir: Path):
        """Version dataset with lakeFS."""
        try:
            version = self.lakefs_client.version_dataset(
                repository="datasets",
                dataset_path=str(dataset_dir),
                branch="main",
                commit_message=f"Ingest dataset: {dataset_name}",
            )

            logger.info(f"Dataset versioned: {version.commit_id}")

        except Exception as e:
            logger.error(f"Dataset versioning failed: {e}")


class IngestionWorker:
    """Base class for ingestion workers."""

    def process(
        self,
        file_path: Path,
        processed_dir: Path,
        index: int,
    ) -> ManifestEntry:
        """Process file and return manifest entry."""
        raise NotImplementedError


class VideoIngestionWorker(IngestionWorker):
    """Video ingestion worker."""

    def process(self, file_path: Path, processed_dir: Path, index: int) -> ManifestEntry:
        """Process video file."""
        import subprocess

        # Get video metadata using ffprobe
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(file_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            probe_data = json.loads(result.stdout)

            # Extract metadata
            format_info = probe_data.get('format', {})
            video_stream = next((s for s in probe_data.get('streams', []) if s['codec_type'] == 'video'), None)

            metadata = {
                'duration': float(format_info.get('duration', 0)),
                'format_name': format_info.get('format_name', ''),
                'bit_rate': int(format_info.get('bit_rate', 0)),
            }

            if video_stream:
                metadata.update({
                    'width': video_stream.get('width'),
                    'height': video_stream.get('height'),
                    'fps': eval(video_stream.get('r_frame_rate', '0/1')),
                    'codec': video_stream.get('codec_name'),
                })

        except Exception as e:
            logger.error(f"FFprobe failed: {e}")
            metadata = {}

        # Copy to processed directory
        processed_path = processed_dir / f"{index:06d}_{file_path.name}"
        shutil.copy2(file_path, processed_path)

        # Get file size
        file_size = file_path.stat().st_size

        # Compute checksum
        checksum = self._compute_checksum(file_path)

        entry = ManifestEntry(
            id=f"{index:06d}",
            source_path=str(file_path),
            processed_path=str(processed_path),
            content_type='video',
            mime_type='video/mp4',
            size_bytes=file_size,
            checksum=checksum,
            metadata=metadata,
            created_at=datetime.now().isoformat(),
        )

        return entry

    def _compute_checksum(self, file_path: Path) -> str:
        """Compute checksum."""
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return f"sha256:{sha256_hash.hexdigest()}"


class AudioIngestionWorker(IngestionWorker):
    """Audio ingestion worker."""

    def process(self, file_path: Path, processed_dir: Path, index: int) -> ManifestEntry:
        """Process audio file."""
        # Similar to video worker, but extract audio-specific metadata
        processed_path = processed_dir / f"{index:06d}_{file_path.name}"
        shutil.copy2(file_path, processed_path)

        file_size = file_path.stat().st_size
        checksum = self._compute_checksum(file_path)

        entry = ManifestEntry(
            id=f"{index:06d}",
            source_path=str(file_path),
            processed_path=str(processed_path),
            content_type='audio',
            mime_type='audio/mpeg',
            size_bytes=file_size,
            checksum=checksum,
            metadata={},
            created_at=datetime.now().isoformat(),
        )

        return entry

    def _compute_checksum(self, file_path: Path) -> str:
        """Compute checksum."""
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return f"sha256:{sha256_hash.hexdigest()}"


class ImageIngestionWorker(IngestionWorker):
    """Image ingestion worker."""

    def process(self, file_path: Path, processed_dir: Path, index: int) -> ManifestEntry:
        """Process image file."""
        from PIL import Image

        # Load image
        img = Image.open(file_path)

        # Extract EXIF if available
        exif_data = {}
        try:
            exif = img._getexif()
            if exif:
                exif_data = {k: str(v) for k, v in exif.items()}
        except:
            pass

        # Get image metadata
        metadata = {
            'width': img.width,
            'height': img.height,
            'format': img.format,
            'mode': img.mode,
            'exif': exif_data,
        }

        # Strip EXIF and save to processed
        processed_path = processed_dir / f"{index:06d}_{file_path.stem}.jpg"

        # Convert to RGB if needed
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')

        # Save without EXIF
        img.save(processed_path, 'JPEG', quality=95)

        # Get file size
        file_size = processed_path.stat().st_size

        # Compute checksum
        checksum = self._compute_checksum(processed_path)

        entry = ManifestEntry(
            id=f"{index:06d}",
            source_path=str(file_path),
            processed_path=str(processed_path),
            content_type='image',
            mime_type='image/jpeg',
            size_bytes=file_size,
            checksum=checksum,
            metadata=metadata,
            created_at=datetime.now().isoformat(),
        )

        return entry

    def _compute_checksum(self, file_path: Path) -> str:
        """Compute checksum."""
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return f"sha256:{sha256_hash.hexdigest()}"


class TextIngestionWorker(IngestionWorker):
    """Text ingestion worker."""

    def process(self, file_path: Path, processed_dir: Path, index: int) -> ManifestEntry:
        """Process text file."""
        # Read text
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        # Get text statistics
        word_count = len(text.split())
        char_count = len(text)
        line_count = text.count('\n') + 1

        metadata = {
            'text': text[:1000],  # Store first 1000 chars
            'word_count': word_count,
            'char_count': char_count,
            'line_count': line_count,
        }

        # Copy to processed directory
        processed_path = processed_dir / f"{index:06d}_{file_path.name}"
        shutil.copy2(file_path, processed_path)

        # Get file size
        file_size = file_path.stat().st_size

        # Compute checksum
        checksum = self._compute_checksum(file_path)

        entry = ManifestEntry(
            id=f"{index:06d}",
            source_path=str(file_path),
            processed_path=str(processed_path),
            content_type='text',
            mime_type='text/plain',
            size_bytes=file_size,
            checksum=checksum,
            metadata=metadata,
            created_at=datetime.now().isoformat(),
        )

        return entry

    def _compute_checksum(self, file_path: Path) -> str:
        """Compute checksum."""
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return f"sha256:{sha256_hash.hexdigest()}"


# Example usage
if __name__ == "__main__":
    # Initialize service
    service = DatasetIngestionService(
        output_dir="/tmp/datasets",
        quality_gates=[],
    )

    # Ingest dataset
    manifest_path = service.ingest_dataset(
        dataset_name="test_dataset",
        source_paths=["/path/to/images"],
        content_type="auto",
        metadata={
            'description': 'Test dataset',
            'license': 'MIT',
        },
    )

    print(f"Manifest created: {manifest_path}")
