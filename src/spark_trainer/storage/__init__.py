"""
Storage backends for SparkTrainer.

Supports:
- Local filesystem
- S3
- MinIO
- Resumable multipart uploads
"""

from .backends import (
    StorageBackend,
    LocalStorageBackend,
    S3StorageBackend,
    get_storage_backend,
)

__all__ = [
    'StorageBackend',
    'LocalStorageBackend',
    'S3StorageBackend',
    'get_storage_backend',
]
