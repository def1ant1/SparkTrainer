"""
Celery tasks for HuggingFace model and dataset transfers.
Implements resumable downloads/uploads with bandwidth management.
"""
import os
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import requests
from celery import Task, current_task
from celery.exceptions import Ignore
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub import HfFolder, Repository
from huggingface_hub.utils import HfHubHTTPError

from celery_app import celery
from database import get_db
from models import Transfer, TransferStatus, TransferType
from bandwidth_manager import get_bandwidth_manager


class TransferTask(Task):
    """Base task class for transfer operations."""

    def __init__(self):
        super().__init__()
        self._hf_api = None
        self.bandwidth_manager = get_bandwidth_manager()

    @property
    def hf_api(self):
        """Lazy initialization of HuggingFace API client."""
        if self._hf_api is None:
            self._hf_api = HfApi()
        return self._hf_api

    def before_start(self, task_id, args, kwargs):
        """Called before task execution."""
        pass

    def on_success(self, retval, task_id, args, kwargs):
        """Called on successful task completion."""
        transfer_id = kwargs.get('transfer_id') or (args[0] if args else None)
        if transfer_id:
            self.bandwidth_manager.finish_transfer(transfer_id)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called on task failure."""
        transfer_id = kwargs.get('transfer_id') or (args[0] if args else None)
        if transfer_id:
            self.bandwidth_manager.finish_transfer(transfer_id)


class ThrottledDownloader:
    """Download with bandwidth throttling and progress tracking."""

    def __init__(self, transfer_id: str, bandwidth_manager, chunk_size: int = 8192):
        self.transfer_id = transfer_id
        self.bandwidth_manager = bandwidth_manager
        self.chunk_size = chunk_size
        self.total_downloaded = 0
        self.start_time = time.time()
        self.last_update_time = time.time()

    def download_file(self, url: str, destination: Path, resume_position: int = 0,
                      token: Optional[str] = None) -> int:
        """
        Download file with resume support and bandwidth throttling.

        Args:
            url: Download URL
            destination: Destination file path
            resume_position: Byte position to resume from
            token: HuggingFace API token

        Returns:
            Total bytes downloaded
        """
        headers = {}
        if token:
            headers['Authorization'] = f'Bearer {token}'

        # Resume from specific position
        if resume_position > 0:
            headers['Range'] = f'bytes={resume_position}-'

        # Open file in append mode if resuming
        mode = 'ab' if resume_position > 0 else 'wb'

        with requests.get(url, headers=headers, stream=True, timeout=30) as response:
            response.raise_for_status()

            # Get total size
            total_size = int(response.headers.get('content-length', 0))
            if resume_position > 0:
                total_size += resume_position

            with open(destination, mode) as f:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        # Throttle bandwidth
                        chunk_size = len(chunk)
                        if not self.bandwidth_manager.throttle(
                            self.transfer_id, chunk_size, timeout=5.0
                        ):
                            raise Exception("Bandwidth throttling timeout")

                        # Write chunk
                        f.write(chunk)
                        self.total_downloaded += chunk_size

                        # Update progress in database (every 0.5 seconds)
                        now = time.time()
                        if now - self.last_update_time >= 0.5:
                            self._update_progress(total_size)
                            self.last_update_time = now

        return self.total_downloaded

    def _update_progress(self, total_size: int):
        """Update transfer progress in database."""
        with get_db() as db:
            transfer = db.query(Transfer).filter(Transfer.id == self.transfer_id).first()
            if transfer:
                elapsed = time.time() - self.start_time
                current_rate = self.total_downloaded / elapsed if elapsed > 0 else 0
                transfer.update_progress(self.total_downloaded, current_rate)
                db.commit()


@celery.task(base=TransferTask, bind=True, name="transfer_tasks.download_hf_model")
def download_hf_model(self, transfer_id: str) -> Dict[str, Any]:
    """
    Download HuggingFace model with bandwidth management and resume support.

    Args:
        transfer_id: Transfer ID from database

    Returns:
        Result dictionary with status and metadata
    """
    with get_db() as db:
        transfer = db.query(Transfer).filter(Transfer.id == transfer_id).first()
        if not transfer:
            raise ValueError(f"Transfer {transfer_id} not found")

        try:
            # Check if we can start based on concurrency limits
            if not self.bandwidth_manager.can_start_transfer(transfer_id):
                # Re-queue for later
                raise Ignore()

            # Transition to downloading
            transfer.transition_to(TransferStatus.DOWNLOADING)
            transfer.celery_task_id = self.request.id
            db.commit()

            # Get transfer configuration
            model_id = transfer.source_url
            token = transfer.metadata.get('token')
            destination_path = Path(transfer.destination_path)
            destination_path.mkdir(parents=True, exist_ok=True)

            # Set bandwidth limit if specified
            if transfer.bandwidth_limit:
                self.bandwidth_manager.set_transfer_limit(
                    transfer_id, transfer.bandwidth_limit
                )

            # Get model info to calculate total size
            try:
                model_info = self.hf_api.model_info(model_id, token=token)
                siblings = model_info.siblings if hasattr(model_info, 'siblings') else []
                total_size = sum(
                    getattr(sibling, 'size', 0) for sibling in siblings
                )
                transfer.size_bytes = total_size
                db.commit()
            except Exception as e:
                print(f"Warning: Could not get model size: {e}")

            # Download model using snapshot_download (handles all files)
            download_path = snapshot_download(
                repo_id=model_id,
                cache_dir=str(destination_path),
                token=token,
                resume_download=True,  # Enable resume
                local_files_only=False,
            )

            # Mark as completed
            transfer.transition_to(TransferStatus.COMPLETED)
            transfer.progress = 100.0
            transfer.destination_path = download_path
            db.commit()

            return {
                "status": "completed",
                "transfer_id": transfer_id,
                "model_id": model_id,
                "download_path": download_path,
                "bytes_transferred": transfer.bytes_transferred
            }

        except Exception as e:
            # Handle errors
            error_message = str(e)
            transfer.error_message = error_message
            transfer.retries += 1

            if transfer.retries < transfer.max_retries:
                # Retry
                transfer.transition_to(TransferStatus.QUEUED, reason="Retry after error")
                db.commit()
                raise self.retry(exc=e, countdown=60 * (2 ** transfer.retries))
            else:
                # Max retries reached
                transfer.transition_to(TransferStatus.FAILED, reason=f"Max retries reached: {error_message}")
                db.commit()
                raise


@celery.task(base=TransferTask, bind=True, name="transfer_tasks.download_hf_dataset")
def download_hf_dataset(self, transfer_id: str) -> Dict[str, Any]:
    """
    Download HuggingFace dataset with bandwidth management and resume support.

    Args:
        transfer_id: Transfer ID from database

    Returns:
        Result dictionary with status and metadata
    """
    from datasets import load_dataset

    with get_db() as db:
        transfer = db.query(Transfer).filter(Transfer.id == transfer_id).first()
        if not transfer:
            raise ValueError(f"Transfer {transfer_id} not found")

        try:
            # Check concurrency limits
            if not self.bandwidth_manager.can_start_transfer(transfer_id):
                raise Ignore()

            # Transition to downloading
            transfer.transition_to(TransferStatus.DOWNLOADING)
            transfer.celery_task_id = self.request.id
            db.commit()

            # Get transfer configuration
            dataset_id = transfer.source_url
            token = transfer.metadata.get('token')
            destination_path = Path(transfer.destination_path)
            destination_path.mkdir(parents=True, exist_ok=True)

            # Set bandwidth limit if specified
            if transfer.bandwidth_limit:
                self.bandwidth_manager.set_transfer_limit(
                    transfer_id, transfer.bandwidth_limit
                )

            # Download dataset
            dataset = load_dataset(
                dataset_id,
                cache_dir=str(destination_path),
                token=token,
            )

            # Get dataset size
            dataset_size = 0
            for split in dataset.keys():
                split_data = dataset[split]
                dataset_size += split_data.dataset_size if hasattr(split_data, 'dataset_size') else 0

            # Mark as completed
            transfer.size_bytes = dataset_size
            transfer.bytes_transferred = dataset_size
            transfer.transition_to(TransferStatus.COMPLETED)
            transfer.progress = 100.0
            db.commit()

            return {
                "status": "completed",
                "transfer_id": transfer_id,
                "dataset_id": dataset_id,
                "download_path": str(destination_path),
                "size_bytes": dataset_size
            }

        except Exception as e:
            # Handle errors
            error_message = str(e)
            transfer.error_message = error_message
            transfer.retries += 1

            if transfer.retries < transfer.max_retries:
                # Retry
                transfer.transition_to(TransferStatus.QUEUED, reason="Retry after error")
                db.commit()
                raise self.retry(exc=e, countdown=60 * (2 ** transfer.retries))
            else:
                # Max retries reached
                transfer.transition_to(TransferStatus.FAILED, reason=f"Max retries reached: {error_message}")
                db.commit()
                raise


@celery.task(base=TransferTask, bind=True, name="transfer_tasks.upload_hf_model")
def upload_hf_model(self, transfer_id: str) -> Dict[str, Any]:
    """
    Upload model to HuggingFace Hub with bandwidth management.

    Args:
        transfer_id: Transfer ID from database

    Returns:
        Result dictionary with status and metadata
    """
    with get_db() as db:
        transfer = db.query(Transfer).filter(Transfer.id == transfer_id).first()
        if not transfer:
            raise ValueError(f"Transfer {transfer_id} not found")

        try:
            # Check concurrency limits
            if not self.bandwidth_manager.can_start_transfer(transfer_id):
                raise Ignore()

            # Transition to uploading
            transfer.transition_to(TransferStatus.UPLOADING)
            transfer.celery_task_id = self.request.id
            db.commit()

            # Get transfer configuration
            repo_id = transfer.source_url  # HF repo ID
            token = transfer.metadata.get('token')
            source_path = Path(transfer.destination_path)  # Local model path

            if not source_path.exists():
                raise ValueError(f"Source path does not exist: {source_path}")

            # Set bandwidth limit if specified
            if transfer.bandwidth_limit:
                self.bandwidth_manager.set_transfer_limit(
                    transfer_id, transfer.bandwidth_limit
                )

            # Calculate total size
            total_size = sum(f.stat().st_size for f in source_path.rglob('*') if f.is_file())
            transfer.size_bytes = total_size
            db.commit()

            # Upload to HuggingFace Hub
            api = HfApi()
            api.upload_folder(
                folder_path=str(source_path),
                repo_id=repo_id,
                token=token,
                repo_type="model",
            )

            # Mark as completed
            transfer.bytes_transferred = total_size
            transfer.transition_to(TransferStatus.COMPLETED)
            transfer.progress = 100.0
            db.commit()

            return {
                "status": "completed",
                "transfer_id": transfer_id,
                "repo_id": repo_id,
                "bytes_uploaded": total_size
            }

        except Exception as e:
            # Handle errors
            error_message = str(e)
            transfer.error_message = error_message
            transfer.retries += 1

            if transfer.retries < transfer.max_retries:
                # Retry
                transfer.transition_to(TransferStatus.QUEUED, reason="Retry after error")
                db.commit()
                raise self.retry(exc=e, countdown=60 * (2 ** transfer.retries))
            else:
                # Max retries reached
                transfer.transition_to(TransferStatus.FAILED, reason=f"Max retries reached: {error_message}")
                db.commit()
                raise


@celery.task(name="transfer_tasks.pause_transfer")
def pause_transfer(transfer_id: str) -> Dict[str, Any]:
    """
    Pause an active transfer.

    Args:
        transfer_id: Transfer ID to pause

    Returns:
        Result dictionary
    """
    with get_db() as db:
        transfer = db.query(Transfer).filter(Transfer.id == transfer_id).first()
        if not transfer:
            raise ValueError(f"Transfer {transfer_id} not found")

        if transfer.status in [TransferStatus.DOWNLOADING, TransferStatus.UPLOADING]:
            # Revoke Celery task
            if transfer.celery_task_id:
                celery.control.revoke(transfer.celery_task_id, terminate=True)

            # Update status
            transfer.transition_to(TransferStatus.PAUSED)
            db.commit()

            return {"status": "paused", "transfer_id": transfer_id}
        else:
            return {
                "status": "error",
                "message": f"Cannot pause transfer in status: {transfer.status}"
            }


@celery.task(name="transfer_tasks.resume_transfer")
def resume_transfer(transfer_id: str) -> Dict[str, Any]:
    """
    Resume a paused transfer.

    Args:
        transfer_id: Transfer ID to resume

    Returns:
        Result dictionary
    """
    with get_db() as db:
        transfer = db.query(Transfer).filter(Transfer.id == transfer_id).first()
        if not transfer:
            raise ValueError(f"Transfer {transfer_id} not found")

        if transfer.status == TransferStatus.PAUSED:
            # Re-queue the transfer task
            transfer.transition_to(TransferStatus.QUEUED)
            db.commit()

            # Dispatch appropriate task based on transfer type
            if transfer.transfer_type == TransferType.MODEL_DOWNLOAD:
                download_hf_model.apply_async(args=[transfer_id], queue='transfers')
            elif transfer.transfer_type == TransferType.DATASET_DOWNLOAD:
                download_hf_dataset.apply_async(args=[transfer_id], queue='transfers')
            elif transfer.transfer_type == TransferType.MODEL_UPLOAD:
                upload_hf_model.apply_async(args=[transfer_id], queue='transfers')

            return {"status": "resumed", "transfer_id": transfer_id}
        else:
            return {
                "status": "error",
                "message": f"Cannot resume transfer in status: {transfer.status}"
            }


@celery.task(name="transfer_tasks.cancel_transfer")
def cancel_transfer(transfer_id: str) -> Dict[str, Any]:
    """
    Cancel an active or paused transfer.

    Args:
        transfer_id: Transfer ID to cancel

    Returns:
        Result dictionary
    """
    with get_db() as db:
        transfer = db.query(Transfer).filter(Transfer.id == transfer_id).first()
        if not transfer:
            raise ValueError(f"Transfer {transfer_id} not found")

        # Revoke Celery task if active
        if transfer.celery_task_id:
            celery.control.revoke(transfer.celery_task_id, terminate=True)

        # Update status
        if transfer.status not in [TransferStatus.COMPLETED, TransferStatus.CANCELLED]:
            transfer.transition_to(TransferStatus.CANCELLED)
            db.commit()

        # Clean up bandwidth manager
        bandwidth_manager = get_bandwidth_manager()
        bandwidth_manager.finish_transfer(transfer_id)

        return {"status": "cancelled", "transfer_id": transfer_id}
