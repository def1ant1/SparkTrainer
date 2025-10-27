"""
Metadata provenance and lineage tracking for SparkTrainer.

Tracks the origin and transformations applied to datasets using SHA-based hashing.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProvenanceRecord:
    """Record of dataset provenance"""

    # Identity
    dataset_id: str
    dataset_hash: str  # SHA256 of dataset content
    version: str = "1.0"

    # Origin
    source_type: str = "unknown"  # video, image, audio, text, etc.
    source_paths: List[str] = field(default_factory=list)
    source_hashes: List[str] = field(default_factory=list)

    # Lineage
    parent_dataset_id: Optional[str] = None
    parent_dataset_hash: Optional[str] = None
    derived_from: List[str] = field(default_factory=list)  # List of parent IDs

    # Transformations
    transformations: List[Dict[str, Any]] = field(default_factory=list)
    augmentations: List[str] = field(default_factory=list)

    # Processing metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    created_by: str = "spark_trainer"
    processing_config: Dict[str, Any] = field(default_factory=dict)

    # Statistics
    num_samples: int = 0
    modalities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    # Validation
    validated: bool = False
    validation_timestamp: Optional[str] = None
    validation_results: Dict[str, Any] = field(default_factory=dict)


class ProvenanceTracker:
    """
    Tracker for dataset provenance and lineage.

    Maintains a graph of dataset relationships and transformations.
    """

    def __init__(self, provenance_dir: Optional[Path] = None):
        """
        Initialize provenance tracker.

        Args:
            provenance_dir: Directory to store provenance records
        """
        if provenance_dir is None:
            # Default to ~/.spark_trainer/provenance
            provenance_dir = Path.home() / ".spark_trainer" / "provenance"

        self.provenance_dir = Path(provenance_dir)
        self.provenance_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self._records: Dict[str, ProvenanceRecord] = {}

    def compute_file_hash(self, file_path: Union[str, Path]) -> str:
        """
        Compute SHA256 hash of a file.

        Args:
            file_path: Path to file

        Returns:
            SHA256 hex digest
        """
        sha256 = hashlib.sha256()

        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            logger.error(f"Failed to hash file {file_path}: {e}")
            return ""

    def compute_dataset_hash(self, manifest_path: Union[str, Path]) -> str:
        """
        Compute hash of a dataset manifest.

        Args:
            manifest_path: Path to manifest.jsonl

        Returns:
            SHA256 hex digest of manifest content
        """
        return self.compute_file_hash(manifest_path)

    def compute_content_hash(self, content: Union[str, bytes, Dict]) -> str:
        """
        Compute hash of arbitrary content.

        Args:
            content: Content to hash

        Returns:
            SHA256 hex digest
        """
        sha256 = hashlib.sha256()

        if isinstance(content, str):
            sha256.update(content.encode('utf-8'))
        elif isinstance(content, bytes):
            sha256.update(content)
        elif isinstance(content, dict):
            # Convert to stable JSON representation
            json_str = json.dumps(content, sort_keys=True)
            sha256.update(json_str.encode('utf-8'))
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

        return sha256.hexdigest()

    def create_record(self,
                     dataset_id: str,
                     source_paths: List[str],
                     source_type: str = "unknown",
                     parent_dataset_id: Optional[str] = None,
                     transformations: Optional[List[Dict[str, Any]]] = None,
                     processing_config: Optional[Dict[str, Any]] = None,
                     **kwargs) -> ProvenanceRecord:
        """
        Create a new provenance record.

        Args:
            dataset_id: Unique dataset identifier
            source_paths: List of source file paths
            source_type: Type of source (video, image, etc.)
            parent_dataset_id: Parent dataset ID if derived
            transformations: List of transformations applied
            processing_config: Processing configuration
            **kwargs: Additional fields

        Returns:
            ProvenanceRecord
        """
        # Compute source hashes
        source_hashes = []
        for path in source_paths:
            if Path(path).exists():
                hash_val = self.compute_file_hash(path)
                source_hashes.append(hash_val)

        # Compute dataset hash from processing config + source hashes
        hash_input = {
            'source_hashes': source_hashes,
            'transformations': transformations or [],
            'processing_config': processing_config or {}
        }
        dataset_hash = self.compute_content_hash(hash_input)

        # Get parent hash if parent exists
        parent_hash = None
        if parent_dataset_id and parent_dataset_id in self._records:
            parent_hash = self._records[parent_dataset_id].dataset_hash

        record = ProvenanceRecord(
            dataset_id=dataset_id,
            dataset_hash=dataset_hash,
            source_type=source_type,
            source_paths=source_paths,
            source_hashes=source_hashes,
            parent_dataset_id=parent_dataset_id,
            parent_dataset_hash=parent_hash,
            transformations=transformations or [],
            processing_config=processing_config or {},
            **kwargs
        )

        # Store in memory and disk
        self._records[dataset_id] = record
        self._save_record(record)

        logger.info(f"Created provenance record for dataset {dataset_id}")
        return record

    def add_transformation(self,
                          dataset_id: str,
                          transformation: Dict[str, Any]):
        """
        Add a transformation to a dataset's history.

        Args:
            dataset_id: Dataset ID
            transformation: Transformation description
        """
        if dataset_id not in self._records:
            logger.warning(f"Dataset {dataset_id} not found")
            return

        record = self._records[dataset_id]
        record.transformations.append({
            **transformation,
            'timestamp': datetime.now().isoformat()
        })

        # Update hash
        hash_input = {
            'source_hashes': record.source_hashes,
            'transformations': record.transformations,
            'processing_config': record.processing_config
        }
        record.dataset_hash = self.compute_content_hash(hash_input)

        self._save_record(record)

    def get_record(self, dataset_id: str) -> Optional[ProvenanceRecord]:
        """
        Get provenance record by dataset ID.

        Args:
            dataset_id: Dataset ID

        Returns:
            ProvenanceRecord or None
        """
        if dataset_id in self._records:
            return self._records[dataset_id]

        # Try to load from disk
        return self._load_record(dataset_id)

    def get_lineage(self, dataset_id: str) -> List[ProvenanceRecord]:
        """
        Get full lineage (all ancestors) of a dataset.

        Args:
            dataset_id: Dataset ID

        Returns:
            List of provenance records from root to current
        """
        lineage = []
        current_id = dataset_id

        while current_id:
            record = self.get_record(current_id)
            if not record:
                break

            lineage.append(record)
            current_id = record.parent_dataset_id

        # Reverse to get root -> current order
        return list(reversed(lineage))

    def get_descendants(self, dataset_id: str) -> List[ProvenanceRecord]:
        """
        Get all datasets derived from this one.

        Args:
            dataset_id: Dataset ID

        Returns:
            List of descendant records
        """
        descendants = []

        for record in self._records.values():
            if record.parent_dataset_id == dataset_id:
                descendants.append(record)

        return descendants

    def validate_lineage(self, dataset_id: str) -> bool:
        """
        Validate lineage integrity by checking hashes.

        Args:
            dataset_id: Dataset ID

        Returns:
            True if lineage is valid
        """
        lineage = self.get_lineage(dataset_id)

        for i, record in enumerate(lineage):
            # Verify source file hashes if files still exist
            for source_path, expected_hash in zip(record.source_paths, record.source_hashes):
                if Path(source_path).exists():
                    actual_hash = self.compute_file_hash(source_path)
                    if actual_hash != expected_hash:
                        logger.error(f"Hash mismatch for {source_path}")
                        return False

            # Verify parent hash matches
            if i > 0:
                parent_record = lineage[i - 1]
                if record.parent_dataset_hash != parent_record.dataset_hash:
                    logger.error(f"Parent hash mismatch for {record.dataset_id}")
                    return False

        return True

    def export_lineage_graph(self, output_path: Union[str, Path], format: str = "dot"):
        """
        Export lineage graph in DOT format for visualization.

        Args:
            output_path: Output file path
            format: Output format ('dot', 'json')
        """
        if format == "dot":
            self._export_dot(output_path)
        elif format == "json":
            self._export_json(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_dot(self, output_path: Union[str, Path]):
        """Export lineage as DOT graph"""
        lines = ["digraph Lineage {"]

        for record in self._records.values():
            label = f"{record.dataset_id}\\n{record.dataset_hash[:8]}"
            lines.append(f'  "{record.dataset_id}" [label="{label}"];')

            if record.parent_dataset_id:
                lines.append(f'  "{record.parent_dataset_id}" -> "{record.dataset_id}";')

        lines.append("}")

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        logger.info(f"Exported lineage graph to {output_path}")

    def _export_json(self, output_path: Union[str, Path]):
        """Export lineage as JSON"""
        data = {
            dataset_id: asdict(record)
            for dataset_id, record in self._records.items()
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported lineage data to {output_path}")

    def _save_record(self, record: ProvenanceRecord):
        """Save record to disk"""
        record_path = self.provenance_dir / f"{record.dataset_id}.json"

        try:
            with open(record_path, 'w') as f:
                json.dump(asdict(record), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save record: {e}")

    def _load_record(self, dataset_id: str) -> Optional[ProvenanceRecord]:
        """Load record from disk"""
        record_path = self.provenance_dir / f"{dataset_id}.json"

        if not record_path.exists():
            return None

        try:
            with open(record_path, 'r') as f:
                data = json.load(f)

            record = ProvenanceRecord(**data)
            self._records[dataset_id] = record
            return record
        except Exception as e:
            logger.error(f"Failed to load record: {e}")
            return None


# Global tracker instance
_provenance_tracker: Optional[ProvenanceTracker] = None


def get_provenance_tracker() -> ProvenanceTracker:
    """Get or create global provenance tracker"""
    global _provenance_tracker
    if _provenance_tracker is None:
        _provenance_tracker = ProvenanceTracker()
    return _provenance_tracker
