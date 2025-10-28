"""
Data ingestion modules for SparkTrainer.

Provides universal ingestion pipeline for multi-modal data:
- Text, image, audio, video support
- Automatic MIME detection
- Media probing and validation
- Quality filtering
"""

from .universal_ingestor import UniversalIngestor, IngestedItem
from .quality_gates import QualityGate, DedupFilter, ToxicityFilter, PIIRedactor
from .dataset_cards import DatasetCardGenerator

__all__ = [
    'UniversalIngestor',
    'IngestedItem',
    'QualityGate',
    'DedupFilter',
    'ToxicityFilter',
    'PIIRedactor',
    'DatasetCardGenerator',
]
