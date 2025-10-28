"""Initial database schema

Revision ID: 001
Revises:
Create Date: 2024-01-15 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create projects table
    op.create_table(
        'projects',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False, unique=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('metadata', postgresql.JSON, default={}),
    )
    op.create_index('idx_project_name', 'projects', ['name'])

    # Create datasets table
    op.create_table(
        'datasets',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('project_id', sa.String(36), sa.ForeignKey('projects.id'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('modality', sa.String(50), nullable=False),
        sa.Column('size_bytes', sa.Integer, default=0),
        sa.Column('num_samples', sa.Integer, default=0),
        sa.Column('manifest_path', sa.String(500), nullable=True),
        sa.Column('storage_path', sa.String(500), nullable=False),
        sa.Column('checksum', sa.String(64), nullable=True),
        sa.Column('integrity_checked', sa.Boolean, default=False),
        sa.Column('integrity_passed', sa.Boolean, default=False),
        sa.Column('integrity_report', postgresql.JSON, default={}),
        sa.Column('statistics', postgresql.JSON, default={}),
        sa.Column('tags', postgresql.JSON, default=[]),
        sa.Column('metadata', postgresql.JSON, default={}),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('idx_dataset_project', 'datasets', ['project_id'])
    op.create_index('idx_dataset_name_version', 'datasets', ['name', 'version'], unique=True)

    # Create experiments table
    op.create_table(
        'experiments',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('project_id', sa.String(36), sa.ForeignKey('projects.id'), nullable=False),
        sa.Column('dataset_id', sa.String(36), sa.ForeignKey('datasets.id'), nullable=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('framework', sa.String(50), nullable=False),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('config', postgresql.JSON, default={}),
        sa.Column('tags', postgresql.JSON, default=[]),
        sa.Column('metadata', postgresql.JSON, default={}),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('idx_experiment_project', 'experiments', ['project_id'])
    op.create_index('idx_experiment_dataset', 'experiments', ['dataset_id'])
    op.create_index('idx_experiment_status', 'experiments', ['status'])

    # Create jobs table
    op.create_table(
        'jobs',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('experiment_id', sa.String(36), sa.ForeignKey('experiments.id'), nullable=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('job_type', sa.String(50), nullable=False),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('priority', sa.Integer, default=0),
        sa.Column('num_gpus', sa.Integer, default=1),
        sa.Column('gpu_type', sa.String(100), nullable=True),
        sa.Column('gpu_memory_gb', sa.Integer, nullable=True),
        sa.Column('command', sa.Text, nullable=True),
        sa.Column('config', postgresql.JSON, default={}),
        sa.Column('resources', postgresql.JSON, default={}),
        sa.Column('environment', postgresql.JSON, default={}),
        sa.Column('artifacts', postgresql.JSON, default={}),
        sa.Column('metrics', postgresql.JSON, default={}),
        sa.Column('logs_path', sa.String(500), nullable=True),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('started_at', sa.DateTime, nullable=True),
        sa.Column('completed_at', sa.DateTime, nullable=True),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('idx_job_experiment', 'jobs', ['experiment_id'])
    op.create_index('idx_job_status', 'jobs', ['status'])
    op.create_index('idx_job_created', 'jobs', ['created_at'])

    # Create models table
    op.create_table(
        'models',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('experiment_id', sa.String(36), sa.ForeignKey('experiments.id'), nullable=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('framework', sa.String(50), nullable=False),
        sa.Column('model_type', sa.String(100), nullable=False),
        sa.Column('storage_path', sa.String(500), nullable=False),
        sa.Column('size_bytes', sa.Integer, default=0),
        sa.Column('checksum', sa.String(64), nullable=True),
        sa.Column('config', postgresql.JSON, default={}),
        sa.Column('metrics', postgresql.JSON, default={}),
        sa.Column('tags', postgresql.JSON, default=[]),
        sa.Column('metadata', postgresql.JSON, default={}),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('idx_model_experiment', 'models', ['experiment_id'])
    op.create_index('idx_model_name_version', 'models', ['name', 'version'], unique=True)

    # Create artifacts table
    op.create_table(
        'artifacts',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('job_id', sa.String(36), sa.ForeignKey('jobs.id'), nullable=True),
        sa.Column('experiment_id', sa.String(36), sa.ForeignKey('experiments.id'), nullable=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('artifact_type', sa.String(50), nullable=False),
        sa.Column('storage_path', sa.String(500), nullable=False),
        sa.Column('size_bytes', sa.Integer, default=0),
        sa.Column('checksum', sa.String(64), nullable=True),
        sa.Column('metadata', postgresql.JSON, default={}),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index('idx_artifact_job', 'artifacts', ['job_id'])
    op.create_index('idx_artifact_experiment', 'artifacts', ['experiment_id'])

    # Create job_transitions table (audit log)
    op.create_table(
        'job_transitions',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('job_id', sa.String(36), sa.ForeignKey('jobs.id'), nullable=False),
        sa.Column('from_status', sa.String(20), nullable=True),
        sa.Column('to_status', sa.String(20), nullable=False),
        sa.Column('reason', sa.Text, nullable=True),
        sa.Column('metadata', postgresql.JSON, default={}),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index('idx_transition_job', 'job_transitions', ['job_id'])
    op.create_index('idx_transition_created', 'job_transitions', ['created_at'])


def downgrade() -> None:
    op.drop_table('job_transitions')
    op.drop_table('artifacts')
    op.drop_table('models')
    op.drop_table('jobs')
    op.drop_table('experiments')
    op.drop_table('datasets')
    op.drop_table('projects')
