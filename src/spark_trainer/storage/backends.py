"""
Storage backends for SparkTrainer.

Supports:
- Local filesystem
- S3
- MinIO
- Resumable uploads (multipart)
"""

import os
import io
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, BinaryIO, Tuple
from abc import ABC, abstractmethod
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Base class for storage backends."""

    @abstractmethod
    def upload_file(
        self,
        local_path: str,
        remote_path: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Upload a file."""
        pass

    @abstractmethod
    def download_file(
        self,
        remote_path: str,
        local_path: str,
    ) -> str:
        """Download a file."""
        pass

    @abstractmethod
    def list_files(
        self,
        prefix: str = "",
        recursive: bool = True,
    ) -> List[str]:
        """List files with optional prefix."""
        pass

    @abstractmethod
    def delete_file(self, remote_path: str):
        """Delete a file."""
        pass

    @abstractmethod
    def file_exists(self, remote_path: str) -> bool:
        """Check if file exists."""
        pass

    @abstractmethod
    def get_file_size(self, remote_path: str) -> int:
        """Get file size in bytes."""
        pass


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend."""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, remote_path: str) -> Path:
        """Resolve remote path to local path."""
        return self.base_dir / remote_path

    def upload_file(
        self,
        local_path: str,
        remote_path: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Copy file to storage directory."""
        import shutil

        local_path = Path(local_path)
        dest_path = self._resolve_path(remote_path)

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(local_path, dest_path)

        logger.info(f"Uploaded {local_path} to {dest_path}")

        return str(dest_path)

    def download_file(
        self,
        remote_path: str,
        local_path: str,
    ) -> str:
        """Copy file from storage directory."""
        import shutil

        src_path = self._resolve_path(remote_path)
        local_path = Path(local_path)

        local_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src_path, local_path)

        logger.info(f"Downloaded {src_path} to {local_path}")

        return str(local_path)

    def list_files(
        self,
        prefix: str = "",
        recursive: bool = True,
    ) -> List[str]:
        """List files in storage directory."""
        search_path = self._resolve_path(prefix)

        if not search_path.exists():
            return []

        if recursive:
            files = [
                str(p.relative_to(self.base_dir))
                for p in search_path.rglob("*")
                if p.is_file()
            ]
        else:
            files = [
                str(p.relative_to(self.base_dir))
                for p in search_path.iterdir()
                if p.is_file()
            ]

        return files

    def delete_file(self, remote_path: str):
        """Delete file from storage."""
        path = self._resolve_path(remote_path)
        if path.exists():
            path.unlink()
            logger.info(f"Deleted {path}")

    def file_exists(self, remote_path: str) -> bool:
        """Check if file exists."""
        return self._resolve_path(remote_path).exists()

    def get_file_size(self, remote_path: str) -> int:
        """Get file size."""
        path = self._resolve_path(remote_path)
        if path.exists():
            return path.stat().st_size
        return 0


class S3StorageBackend(StorageBackend):
    """
    S3/MinIO storage backend.

    Supports:
    - AWS S3
    - MinIO (S3-compatible)
    - Multipart uploads for large files
    - Resumable uploads
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        endpoint_url: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        region: str = "us-east-1",
        multipart_threshold: int = 100 * 1024 * 1024,  # 100MB
        multipart_chunksize: int = 10 * 1024 * 1024,  # 10MB
    ):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")

        # S3 client config
        config = Config(
            signature_version='s3v4',
            retries={'max_attempts': 3, 'mode': 'standard'},
        )

        # Initialize S3 client
        self.s3 = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id or os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=region,
            config=config,
        )

        # Transfer config for multipart uploads
        self.transfer_config = boto3.s3.transfer.TransferConfig(
            multipart_threshold=multipart_threshold,
            multipart_chunksize=multipart_chunksize,
            max_concurrency=10,
            use_threads=True,
        )

        # Ensure bucket exists
        self._ensure_bucket()

        logger.info(f"S3 backend initialized: bucket={bucket}, prefix={prefix}")

    def _ensure_bucket(self):
        """Ensure bucket exists, create if it doesn't."""
        try:
            self.s3.head_bucket(Bucket=self.bucket)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                # Bucket doesn't exist, create it
                try:
                    self.s3.create_bucket(Bucket=self.bucket)
                    logger.info(f"Created bucket: {self.bucket}")
                except ClientError as create_error:
                    logger.error(f"Failed to create bucket: {create_error}")
            else:
                logger.error(f"Error checking bucket: {e}")

    def _full_key(self, remote_path: str) -> str:
        """Get full S3 key with prefix."""
        if self.prefix:
            return f"{self.prefix}/{remote_path}"
        return remote_path

    def upload_file(
        self,
        local_path: str,
        remote_path: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Upload file to S3 with multipart support."""
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        key = self._full_key(remote_path)

        # Calculate checksum
        md5_hash = self._calculate_md5(local_path)

        # Extra args
        extra_args = {
            'Metadata': metadata or {},
        }

        # Add content MD5 for integrity check
        extra_args['Metadata']['md5'] = md5_hash

        # Upload with progress callback
        file_size = local_path.stat().st_size

        def upload_progress(bytes_transferred):
            progress = (bytes_transferred / file_size) * 100
            logger.debug(f"Upload progress: {progress:.1f}%")

        logger.info(f"Uploading {local_path} ({file_size / 1e6:.1f} MB) to s3://{self.bucket}/{key}")

        self.s3.upload_file(
            str(local_path),
            self.bucket,
            key,
            ExtraArgs=extra_args,
            Config=self.transfer_config,
            Callback=upload_progress,
        )

        logger.info(f"Upload complete: s3://{self.bucket}/{key}")

        return f"s3://{self.bucket}/{key}"

    def download_file(
        self,
        remote_path: str,
        local_path: str,
    ) -> str:
        """Download file from S3."""
        key = self._full_key(remote_path)
        local_path = Path(local_path)

        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Get file size
        response = self.s3.head_object(Bucket=self.bucket, Key=key)
        file_size = response['ContentLength']

        def download_progress(bytes_transferred):
            progress = (bytes_transferred / file_size) * 100
            logger.debug(f"Download progress: {progress:.1f}%")

        logger.info(f"Downloading s3://{self.bucket}/{key} ({file_size / 1e6:.1f} MB) to {local_path}")

        self.s3.download_file(
            self.bucket,
            key,
            str(local_path),
            Config=self.transfer_config,
            Callback=download_progress,
        )

        # Verify checksum if available
        metadata = response.get('Metadata', {})
        if 'md5' in metadata:
            local_md5 = self._calculate_md5(local_path)
            if local_md5 != metadata['md5']:
                raise ValueError(f"Checksum mismatch: {local_md5} != {metadata['md5']}")
            logger.info("Checksum verified")

        logger.info(f"Download complete: {local_path}")

        return str(local_path)

    def list_files(
        self,
        prefix: str = "",
        recursive: bool = True,
    ) -> List[str]:
        """List files in S3 bucket."""
        full_prefix = self._full_key(prefix)

        paginator = self.s3.get_paginator('list_objects_v2')

        params = {
            'Bucket': self.bucket,
            'Prefix': full_prefix,
        }

        if not recursive:
            params['Delimiter'] = '/'

        files = []

        for page in paginator.paginate(**params):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    # Remove prefix
                    if self.prefix and key.startswith(self.prefix + '/'):
                        key = key[len(self.prefix) + 1:]
                    files.append(key)

        return files

    def delete_file(self, remote_path: str):
        """Delete file from S3."""
        key = self._full_key(remote_path)

        self.s3.delete_object(Bucket=self.bucket, Key=key)

        logger.info(f"Deleted s3://{self.bucket}/{key}")

    def file_exists(self, remote_path: str) -> bool:
        """Check if file exists in S3."""
        key = self._full_key(remote_path)

        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise

    def get_file_size(self, remote_path: str) -> int:
        """Get file size from S3."""
        key = self._full_key(remote_path)

        try:
            response = self.s3.head_object(Bucket=self.bucket, Key=key)
            return response['ContentLength']
        except ClientError:
            return 0

    def _calculate_md5(self, file_path: Path) -> str:
        """Calculate MD5 checksum of file."""
        md5 = hashlib.md5()

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)

        return md5.hexdigest()

    def create_multipart_upload(
        self,
        remote_path: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Create a multipart upload.

        Returns upload ID for resumable uploads.
        """
        key = self._full_key(remote_path)

        response = self.s3.create_multipart_upload(
            Bucket=self.bucket,
            Key=key,
            Metadata=metadata or {},
        )

        upload_id = response['UploadId']

        logger.info(f"Created multipart upload: {upload_id} for s3://{self.bucket}/{key}")

        return upload_id

    def upload_part(
        self,
        remote_path: str,
        upload_id: str,
        part_number: int,
        data: bytes,
    ) -> Dict:
        """Upload a single part of multipart upload."""
        key = self._full_key(remote_path)

        response = self.s3.upload_part(
            Bucket=self.bucket,
            Key=key,
            UploadId=upload_id,
            PartNumber=part_number,
            Body=data,
        )

        return {
            'PartNumber': part_number,
            'ETag': response['ETag'],
        }

    def complete_multipart_upload(
        self,
        remote_path: str,
        upload_id: str,
        parts: List[Dict],
    ):
        """Complete multipart upload."""
        key = self._full_key(remote_path)

        self.s3.complete_multipart_upload(
            Bucket=self.bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={'Parts': parts},
        )

        logger.info(f"Completed multipart upload: s3://{self.bucket}/{key}")

    def abort_multipart_upload(
        self,
        remote_path: str,
        upload_id: str,
    ):
        """Abort multipart upload."""
        key = self._full_key(remote_path)

        self.s3.abort_multipart_upload(
            Bucket=self.bucket,
            Key=key,
            UploadId=upload_id,
        )

        logger.info(f"Aborted multipart upload: {upload_id}")


def get_storage_backend(
    backend_type: str = "local",
    **kwargs
) -> StorageBackend:
    """
    Factory function to get storage backend.

    Args:
        backend_type: Type of backend (local, s3, minio)
        **kwargs: Backend-specific arguments

    Returns:
        StorageBackend instance
    """
    if backend_type == "local":
        base_dir = kwargs.get("base_dir", "./storage")
        return LocalStorageBackend(base_dir)

    elif backend_type in ["s3", "minio"]:
        return S3StorageBackend(**kwargs)

    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Storage backend CLI")
    parser.add_argument("action", choices=["upload", "download", "list", "delete"])
    parser.add_argument("--backend", default="local", choices=["local", "s3", "minio"])
    parser.add_argument("--local-path", help="Local file path")
    parser.add_argument("--remote-path", help="Remote file path")
    parser.add_argument("--bucket", help="S3 bucket name")
    parser.add_argument("--endpoint", help="S3 endpoint URL (for MinIO)")
    parser.add_argument("--prefix", default="", help="S3 prefix")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Create backend
    if args.backend == "local":
        backend = get_storage_backend("local")
    else:
        backend = get_storage_backend(
            args.backend,
            bucket=args.bucket,
            prefix=args.prefix,
            endpoint_url=args.endpoint,
        )

    # Execute action
    if args.action == "upload":
        result = backend.upload_file(args.local_path, args.remote_path)
        print(f"Uploaded to: {result}")

    elif args.action == "download":
        result = backend.download_file(args.remote_path, args.local_path)
        print(f"Downloaded to: {result}")

    elif args.action == "list":
        files = backend.list_files(args.remote_path or "")
        print(f"Files ({len(files)}):")
        for f in files:
            print(f"  {f}")

    elif args.action == "delete":
        backend.delete_file(args.remote_path)
        print(f"Deleted: {args.remote_path}")


if __name__ == "__main__":
    main()
