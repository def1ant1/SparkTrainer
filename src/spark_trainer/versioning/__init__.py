"""
Dataset versioning backends for SparkTrainer.

Supports:
- DVC (Data Version Control) with S3/MinIO
- LakeFS for Git-like data operations
"""

from .dvc_backend import DVCBackend, LakeFSBackend

__all__ = ['DVCBackend', 'LakeFSBackend']
