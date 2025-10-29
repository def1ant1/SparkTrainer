"""
SQLAlchemy database models for SparkTrainer.
Defines schemas for jobs, experiments, artifacts, and status transitions.
"""
from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional, Dict, Any
import json

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Text, JSON,
    ForeignKey, Enum, Boolean, Index, event
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.sql import func

Base = declarative_base()


class JobStatus(str, PyEnum):
    """Job status enumeration with valid state transitions."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TransferStatus(str, PyEnum):
    """Transfer status enumeration."""
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    UPLOADING = "uploading"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TransferType(str, PyEnum):
    """Transfer type enumeration."""
    MODEL_DOWNLOAD = "model_download"
    DATASET_DOWNLOAD = "dataset_download"
    MODEL_UPLOAD = "model_upload"
    DATASET_UPLOAD = "dataset_upload"


# Valid state transitions
VALID_TRANSITIONS = {
    JobStatus.PENDING: [JobStatus.QUEUED, JobStatus.CANCELLED],
    JobStatus.QUEUED: [JobStatus.RUNNING, JobStatus.CANCELLED],
    JobStatus.RUNNING: [JobStatus.PAUSED, JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED],
    JobStatus.PAUSED: [JobStatus.RUNNING, JobStatus.CANCELLED],
    JobStatus.COMPLETED: [],  # Terminal state
    JobStatus.FAILED: [],     # Terminal state
    JobStatus.CANCELLED: [],  # Terminal state
}

# Valid transfer state transitions
VALID_TRANSFER_TRANSITIONS = {
    TransferStatus.QUEUED: [TransferStatus.DOWNLOADING, TransferStatus.UPLOADING, TransferStatus.CANCELLED],
    TransferStatus.DOWNLOADING: [TransferStatus.PAUSED, TransferStatus.COMPLETED, TransferStatus.FAILED, TransferStatus.CANCELLED],
    TransferStatus.UPLOADING: [TransferStatus.PAUSED, TransferStatus.COMPLETED, TransferStatus.FAILED, TransferStatus.CANCELLED],
    TransferStatus.PAUSED: [TransferStatus.DOWNLOADING, TransferStatus.UPLOADING, TransferStatus.CANCELLED],
    TransferStatus.COMPLETED: [],  # Terminal state
    TransferStatus.FAILED: [TransferStatus.QUEUED],  # Allow retry
    TransferStatus.CANCELLED: [],  # Terminal state
}


class Project(Base):
    """Project grouping for experiments and datasets."""
    __tablename__ = "projects"

    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    metadata = Column(JSON, default={})

    # Relationships
    experiments = relationship("Experiment", back_populates="project", cascade="all, delete-orphan")
    datasets = relationship("Dataset", back_populates="project", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_project_name", "name"),
    )


class Dataset(Base):
    """Dataset management with versioning and integrity tracking."""
    __tablename__ = "datasets"

    id = Column(String(36), primary_key=True)
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False)
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    description = Column(Text, nullable=True)

    # Dataset properties
    modality = Column(String(50), nullable=False)  # text, image, video, audio, multimodal
    size_bytes = Column(Integer, default=0)
    num_samples = Column(Integer, default=0)
    manifest_path = Column(String(500), nullable=True)
    storage_path = Column(String(500), nullable=False)

    # Integrity tracking
    checksum = Column(String(64), nullable=True)  # SHA256
    integrity_checked = Column(Boolean, default=False)
    integrity_passed = Column(Boolean, default=False)
    integrity_report = Column(JSON, default={})

    # Metadata
    statistics = Column(JSON, default={})  # Dataset statistics
    tags = Column(JSON, default=[])
    metadata = Column(JSON, default={})

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    project = relationship("Project", back_populates="datasets")
    experiments = relationship("Experiment", back_populates="dataset")

    __table_args__ = (
        Index("idx_dataset_project", "project_id"),
        Index("idx_dataset_name_version", "name", "version"),
    )


class BaseModel(Base):
    """Base model registry for fine-tuning and training."""
    __tablename__ = "base_models"

    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)
    family = Column(String(100), nullable=False)  # llama, mistral, gpt, bert, vit, etc.
    description = Column(Text, nullable=True)

    # Model properties
    params_b = Column(Float, nullable=True)  # Parameters in billions
    dtype = Column(String(20), nullable=False)  # fp32, fp16, bf16, int8, int4
    context_length = Column(Integer, nullable=True)
    hidden_size = Column(Integer, nullable=True)
    num_layers = Column(Integer, nullable=True)

    # Model architecture
    architecture = Column(String(100), nullable=True)  # transformer, cnn, vit, etc.
    modality = Column(String(50), nullable=False)  # text, image, audio, multimodal

    # Capabilities
    trainable = Column(Boolean, default=True)
    servable = Column(Boolean, default=True)
    quantized = Column(Boolean, default=False)
    is_gguf = Column(Boolean, default=False)

    # Stage and status
    stage = Column(String(20), default="staging")  # staging, production, archived
    status = Column(String(20), default="active")  # active, deprecated, archived

    # Storage
    storage_path = Column(String(500), nullable=False)
    size_bytes = Column(Integer, default=0)
    checksum = Column(String(64), nullable=True)

    # HuggingFace integration
    hf_repo_id = Column(String(255), nullable=True)
    hf_revision = Column(String(64), nullable=True)

    # Tokenizer info
    tokenizer_path = Column(String(500), nullable=True)
    vocab_size = Column(Integer, nullable=True)

    # Metadata
    tags = Column(JSON, default=[])
    metadata = Column(JSON, default={})
    model_card = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    experiments = relationship("Experiment", back_populates="base_model")

    __table_args__ = (
        Index("idx_base_model_name", "name"),
        Index("idx_base_model_family", "family"),
        Index("idx_base_model_stage", "stage"),
        Index("idx_base_model_modality", "modality"),
    )


class Recipe(Base):
    """Training recipe templates (LoRA, QLoRA, Full-FT, etc.)."""
    __tablename__ = "recipes"

    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    display_name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Recipe properties
    recipe_type = Column(String(50), nullable=False)  # lora, qlora, full_ft, prompt_tuning, etc.
    modality = Column(String(50), nullable=False)  # text, image, audio, multimodal
    train_styles = Column(JSON, default=[])  # ["qlora","lora","full"]

    # Default configuration
    default_config = Column(JSON, default={})  # Default hyperparameters
    required_fields = Column(JSON, default=[])  # Required config fields
    optional_fields = Column(JSON, default=[])  # Optional config fields

    # Compatibility
    supported_architectures = Column(JSON, default=[])  # ["transformer", "cnn", etc.]
    min_gpu_memory_gb = Column(Float, nullable=True)
    supports_distributed = Column(Boolean, default=True)

    # Template
    template_path = Column(String(500), nullable=True)
    script_template = Column(Text, nullable=True)

    # Metadata
    tags = Column(JSON, default=[])
    metadata = Column(JSON, default={})
    is_active = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    experiments = relationship("Experiment", back_populates="recipe")

    __table_args__ = (
        Index("idx_recipe_type", "recipe_type"),
        Index("idx_recipe_modality", "modality"),
        Index("idx_recipe_active", "is_active"),
    )


class Adapter(Base):
    """Adapter/LoRA registry with composition support."""
    __tablename__ = "adapters"

    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)
    base_model_id = Column(String(36), ForeignKey("base_models.id"), nullable=False)
    description = Column(Text, nullable=True)

    # Adapter properties
    adapter_type = Column(String(50), nullable=False)  # lora, qlora, ia3, prompt_tuning, etc.
    rank = Column(Integer, nullable=True)  # LoRA rank (r)
    alpha = Column(Integer, nullable=True)  # LoRA alpha
    dropout = Column(Float, default=0.0)
    target_modules = Column(JSON, default=[])  # ["q_proj", "v_proj", etc.]

    # Status
    status = Column(String(20), default="training")  # training, ready, archived
    training_experiment_id = Column(String(36), nullable=True)

    # Storage
    storage_path = Column(String(500), nullable=False)
    size_bytes = Column(Integer, default=0)
    checksum = Column(String(64), nullable=True)

    # Performance metrics
    metrics = Column(JSON, default={})  # Final metrics from training

    # Metadata
    tags = Column(JSON, default=[])
    metadata = Column(JSON, default={})

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_adapter_base_model", "base_model_id"),
        Index("idx_adapter_type", "adapter_type"),
        Index("idx_adapter_status", "status"),
    )


class Experiment(Base):
    """Experiment tracking with MLflow integration and comprehensive configuration."""
    __tablename__ = "experiments"

    id = Column(String(36), primary_key=True)
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False)
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=True)
    base_model_id = Column(String(36), ForeignKey("base_models.id"), nullable=True)
    recipe_id = Column(String(36), ForeignKey("recipes.id"), nullable=True)

    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # MLflow integration
    mlflow_run_id = Column(String(64), nullable=True, unique=True)
    mlflow_experiment_id = Column(String(64), nullable=True)

    # Experiment properties
    model_type = Column(String(100), nullable=True)
    recipe_name = Column(String(100), nullable=True)

    # Status
    status = Column(Enum(JobStatus), default=JobStatus.PENDING)

    # Metrics (latest values for quick access)
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer, default=0)
    current_step = Column(Integer, default=0)
    total_steps = Column(Integer, default=0)

    # Performance metrics (stored as JSON for flexibility)
    metrics = Column(JSON, default={})

    # Configuration
    config = Column(JSON, default={})
    hyperparameters = Column(JSON, default={})

    # New: Comprehensive experiment specification
    adapters = Column(JSON, default=[])  # [{adapter_id, mode:"attach"|"compose"}]
    train = Column(JSON, default={})  # {max_steps, global_batch_size, grad_accum, lr, seed, checkpoint_interval}
    strategy = Column(JSON, default={})  # {type:"ddp"|"fsdp"|"deepspeed", mixed_precision:"bf16"|"fp16"}
    resources = Column(JSON, default={})  # {gpus:int, partition_id?:uuid}
    eval = Column(JSON, default={})  # {suites:[...], interval:int}
    export = Column(JSON, default=[])  # ["safetensors","onnx","gguf"]
    preflight_summary = Column(JSON, default={})  # {ok, estimated_vram_mb, time_per_step_ms, warnings[], errors[]}

    # Model output
    model_path = Column(String(500), nullable=True)
    checkpoint_path = Column(String(500), nullable=True)

    # Metadata
    tags = Column(JSON, default=[])
    metadata = Column(JSON, default={})
    starred = Column(Boolean, default=False)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    project = relationship("Project", back_populates="experiments")
    dataset = relationship("Dataset", back_populates="experiments")
    base_model = relationship("BaseModel", back_populates="experiments")
    recipe = relationship("Recipe", back_populates="experiments")
    jobs = relationship("Job", back_populates="experiment", cascade="all, delete-orphan")
    artifacts = relationship("Artifact", back_populates="experiment", cascade="all, delete-orphan")
    evaluations = relationship("Evaluation", back_populates="experiment", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_experiment_project", "project_id"),
        Index("idx_experiment_base_model", "base_model_id"),
        Index("idx_experiment_dataset", "dataset_id"),
        Index("idx_experiment_recipe", "recipe_id"),
        Index("idx_experiment_mlflow_run", "mlflow_run_id"),
        Index("idx_experiment_status", "status"),
    )


class Job(Base):
    """Training job with Celery task integration."""
    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True)
    experiment_id = Column(String(36), ForeignKey("experiments.id"), nullable=False)

    # Celery task tracking
    celery_task_id = Column(String(64), nullable=True, unique=True)

    # Job properties
    job_type = Column(String(50), default="training")  # training, evaluation, preprocessing
    command = Column(Text, nullable=True)

    # Status tracking
    status = Column(Enum(JobStatus), default=JobStatus.PENDING)
    priority = Column(Integer, default=0)

    # Progress
    progress = Column(Float, default=0.0)
    current_step = Column(Integer, default=0)
    total_steps = Column(Integer, default=0)

    # Resource usage
    gpu_ids = Column(JSON, default=[])
    gpu_memory_used = Column(Float, default=0.0)  # GB
    cpu_percent = Column(Float, default=0.0)
    memory_used = Column(Float, default=0.0)  # GB

    # Logs
    log_file = Column(String(500), nullable=True)
    error_message = Column(Text, nullable=True)

    # Configuration
    config = Column(JSON, default={})
    metadata = Column(JSON, default={})

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    queued_at = Column(DateTime, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    experiment = relationship("Experiment", back_populates="jobs")
    status_transitions = relationship(
        "JobStatusTransition",
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="JobStatusTransition.timestamp"
    )

    __table_args__ = (
        Index("idx_job_experiment", "experiment_id"),
        Index("idx_job_celery_task", "celery_task_id"),
        Index("idx_job_status", "status"),
        Index("idx_job_created", "created_at"),
    )

    def can_transition_to(self, new_status: JobStatus) -> bool:
        """Check if transition to new status is valid."""
        current = JobStatus(self.status)
        target = JobStatus(new_status)
        return target in VALID_TRANSITIONS.get(current, [])

    def transition_to(self, session: Session, new_status: JobStatus,
                     reason: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Transition job to new status with validation and audit trail."""
        if not self.can_transition_to(new_status):
            raise ValueError(
                f"Invalid status transition from {self.status} to {new_status}. "
                f"Valid transitions: {VALID_TRANSITIONS.get(JobStatus(self.status), [])}"
            )

        old_status = self.status
        self.status = new_status

        # Update timestamps
        now = datetime.utcnow()
        if new_status == JobStatus.QUEUED:
            self.queued_at = now
        elif new_status == JobStatus.RUNNING:
            self.started_at = now
        elif new_status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            self.completed_at = now

        # Create status transition record
        transition = JobStatusTransition(
            job_id=self.id,
            from_status=old_status,
            to_status=new_status,
            reason=reason,
            metadata=metadata or {}
        )
        session.add(transition)

        return transition


class JobStatusTransition(Base):
    """Audit trail for job status transitions."""
    __tablename__ = "job_status_transitions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(36), ForeignKey("jobs.id"), nullable=False)

    from_status = Column(Enum(JobStatus), nullable=False)
    to_status = Column(Enum(JobStatus), nullable=False)

    reason = Column(Text, nullable=True)
    metadata = Column(JSON, default={})

    timestamp = Column(DateTime, server_default=func.now())

    # Relationships
    job = relationship("Job", back_populates="status_transitions")

    __table_args__ = (
        Index("idx_transition_job", "job_id"),
        Index("idx_transition_timestamp", "timestamp"),
    )


class Artifact(Base):
    """Training artifacts (models, checkpoints, logs, plots)."""
    __tablename__ = "artifacts"

    id = Column(String(36), primary_key=True)
    experiment_id = Column(String(36), ForeignKey("experiments.id"), nullable=False)

    # Artifact properties
    artifact_type = Column(String(50), nullable=False)  # model, checkpoint, log, plot, metric
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Storage
    file_path = Column(String(500), nullable=False)
    storage_backend = Column(String(50), default="local")  # local, s3, minio
    size_bytes = Column(Integer, default=0)
    checksum = Column(String(64), nullable=True)  # SHA256

    # MLflow integration
    mlflow_artifact_path = Column(String(500), nullable=True)

    # Versioning
    version = Column(String(50), nullable=True)
    is_latest = Column(Boolean, default=True)

    # Metadata
    tags = Column(JSON, default=[])
    metadata = Column(JSON, default={})

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    experiment = relationship("Experiment", back_populates="artifacts")

    __table_args__ = (
        Index("idx_artifact_experiment", "experiment_id"),
        Index("idx_artifact_type", "artifact_type"),
        Index("idx_artifact_latest", "is_latest"),
    )


class Evaluation(Base):
    """Evaluation results for experiments (MMLU, COCO, etc.)."""
    __tablename__ = "evaluations"

    id = Column(String(36), primary_key=True)
    experiment_id = Column(String(36), ForeignKey("experiments.id"), nullable=False)

    # Evaluation properties
    benchmark_name = Column(String(100), nullable=False)  # mmlu, coco, glue, etc.
    benchmark_version = Column(String(50), nullable=True)
    subset = Column(String(100), nullable=True)  # e.g., "mmlu-abstract_algebra"

    # Results
    score = Column(Float, nullable=False)
    metrics = Column(JSON, default={})  # Detailed metrics

    # Configuration
    eval_config = Column(JSON, default={})
    num_samples = Column(Integer, nullable=True)

    # Metadata
    metadata = Column(JSON, default={})

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    experiment = relationship("Experiment", back_populates="evaluations")

    __table_args__ = (
        Index("idx_eval_experiment", "experiment_id"),
        Index("idx_eval_benchmark", "benchmark_name"),
        Index("idx_eval_score", "score"),
    )


class Transfer(Base):
    """HuggingFace transfer queue with bandwidth management and resumable transfers."""
    __tablename__ = "transfers"

    id = Column(String(36), primary_key=True)

    # Celery task tracking
    celery_task_id = Column(String(64), nullable=True, unique=True)

    # Transfer properties
    name = Column(String(500), nullable=False)  # Model/dataset name
    transfer_type = Column(Enum(TransferType), nullable=False)
    direction = Column(String(20), nullable=False)  # 'download' or 'upload'

    # Source and destination
    source_url = Column(String(1000), nullable=True)  # HF repo ID or URL
    destination_path = Column(String(1000), nullable=False)

    # Size and progress
    size_bytes = Column(Integer, default=0)  # Total size
    bytes_transferred = Column(Integer, default=0)  # Bytes transferred so far
    progress = Column(Float, default=0.0)  # Percentage (0.0-100.0)

    # Bandwidth metrics
    current_rate = Column(Float, default=0.0)  # Current transfer rate (bytes/sec)
    average_rate = Column(Float, default=0.0)  # Average transfer rate (bytes/sec)
    bandwidth_limit = Column(Integer, nullable=True)  # Max bytes/sec (null = unlimited)

    # Status tracking
    status = Column(Enum(TransferStatus), default=TransferStatus.QUEUED)
    priority = Column(Integer, default=0)  # Higher = more important

    # Resume support
    resume_token = Column(String(500), nullable=True)  # For resumable uploads
    last_byte_position = Column(Integer, default=0)  # For HTTP Range requests
    chunk_checksums = Column(JSON, default=[])  # Verify partial transfers

    # Retry logic
    retries = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)

    # Error handling
    error_message = Column(Text, nullable=True)
    error_details = Column(JSON, default={})

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    paused_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Estimated completion
    eta_seconds = Column(Integer, nullable=True)  # Estimated time remaining

    # Metadata
    metadata = Column(JSON, default={})  # Store HF token, config options, etc.

    __table_args__ = (
        Index("idx_transfer_status", "status"),
        Index("idx_transfer_priority", "priority"),
        Index("idx_transfer_type", "transfer_type"),
        Index("idx_transfer_created", "created_at"),
        Index("idx_transfer_celery_task", "celery_task_id"),
    )

    def can_transition_to(self, new_status: TransferStatus) -> bool:
        """Check if transition to new status is valid."""
        current = TransferStatus(self.status)
        target = TransferStatus(new_status)
        return target in VALID_TRANSFER_TRANSITIONS.get(current, [])

    def transition_to(self, new_status: TransferStatus, reason: Optional[str] = None):
        """Transition transfer to new status with validation."""
        if not self.can_transition_to(new_status):
            raise ValueError(
                f"Invalid status transition from {self.status} to {new_status}. "
                f"Valid transitions: {VALID_TRANSFER_TRANSITIONS.get(TransferStatus(self.status), [])}"
            )

        old_status = self.status
        self.status = new_status

        # Update timestamps
        now = datetime.utcnow()
        if new_status in [TransferStatus.DOWNLOADING, TransferStatus.UPLOADING]:
            if not self.started_at:
                self.started_at = now
        elif new_status == TransferStatus.PAUSED:
            self.paused_at = now
        elif new_status in [TransferStatus.COMPLETED, TransferStatus.FAILED, TransferStatus.CANCELLED]:
            self.completed_at = now

        return old_status

    def update_progress(self, bytes_transferred: int, current_rate: float = None):
        """Update transfer progress and calculate ETA."""
        self.bytes_transferred = bytes_transferred

        if self.size_bytes > 0:
            self.progress = (bytes_transferred / self.size_bytes) * 100.0

        if current_rate is not None:
            self.current_rate = current_rate

            # Calculate average rate
            if self.started_at:
                elapsed = (datetime.utcnow() - self.started_at).total_seconds()
                if elapsed > 0:
                    self.average_rate = bytes_transferred / elapsed

            # Calculate ETA
            remaining_bytes = self.size_bytes - bytes_transferred
            if self.average_rate > 0 and remaining_bytes > 0:
                self.eta_seconds = int(remaining_bytes / self.average_rate)
            else:
                self.eta_seconds = None


class LeaderboardEntry(Base):
    """Leaderboard rankings for experiments."""
    __tablename__ = "leaderboard"

    id = Column(String(36), primary_key=True)
    experiment_id = Column(String(36), ForeignKey("experiments.id"), nullable=False)

    # Ranking
    benchmark_name = Column(String(100), nullable=False)
    score = Column(Float, nullable=False)
    rank = Column(Integer, nullable=True)

    # Model info (denormalized for performance)
    model_name = Column(String(255), nullable=True)
    model_type = Column(String(100), nullable=True)

    # Metadata
    metadata = Column(JSON, default={})

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_leaderboard_benchmark", "benchmark_name"),
        Index("idx_leaderboard_score", "score"),
        Index("idx_leaderboard_rank", "rank"),
    )


# Event listeners for automatic timestamp updates
@event.listens_for(Project, 'before_update')
@event.listens_for(Dataset, 'before_update')
@event.listens_for(BaseModel, 'before_update')
@event.listens_for(Recipe, 'before_update')
@event.listens_for(Adapter, 'before_update')
@event.listens_for(Experiment, 'before_update')
@event.listens_for(Job, 'before_update')
@event.listens_for(Artifact, 'before_update')
@event.listens_for(Transfer, 'before_update')
def receive_before_update(mapper, connection, target):
    """Update updated_at timestamp before update."""
    target.updated_at = datetime.utcnow()


def validate_job_transition(mapper, connection, target):
    """Validate job status transitions."""
    if hasattr(target, '_sa_instance_state'):
        history = target._sa_instance_state.attrs.status.history
        if history.has_changes():
            old_status = history.deleted[0] if history.deleted else None
            new_status = target.status

            if old_status and old_status != new_status:
                if not JobStatus(new_status) in VALID_TRANSITIONS.get(JobStatus(old_status), []):
                    raise ValueError(
                        f"Invalid status transition from {old_status} to {new_status}"
                    )


# Register validation listener
event.listen(Job, 'before_update', validate_job_transition)
