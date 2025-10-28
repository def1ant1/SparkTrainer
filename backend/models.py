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


class Experiment(Base):
    """Experiment tracking with MLflow integration."""
    __tablename__ = "experiments"

    id = Column(String(36), primary_key=True)
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False)
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=True)

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
    jobs = relationship("Job", back_populates="experiment", cascade="all, delete-orphan")
    artifacts = relationship("Artifact", back_populates="experiment", cascade="all, delete-orphan")
    evaluations = relationship("Evaluation", back_populates="experiment", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_experiment_project", "project_id"),
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
@event.listens_for(Experiment, 'before_update')
@event.listens_for(Job, 'before_update')
@event.listens_for(Artifact, 'before_update')
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
