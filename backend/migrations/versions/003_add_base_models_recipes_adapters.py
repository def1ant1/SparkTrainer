"""Add base models, recipes, adapters and enhance experiments

Revision ID: 003
Revises: 002
Create Date: 2025-10-29 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '003'
down_revision: Union[str, None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create base_models table
    op.create_table(
        'base_models',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('family', sa.String(100), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('params_b', sa.Float, nullable=True),
        sa.Column('dtype', sa.String(20), nullable=False),
        sa.Column('context_length', sa.Integer, nullable=True),
        sa.Column('hidden_size', sa.Integer, nullable=True),
        sa.Column('num_layers', sa.Integer, nullable=True),
        sa.Column('architecture', sa.String(100), nullable=True),
        sa.Column('modality', sa.String(50), nullable=False),
        sa.Column('trainable', sa.Boolean, default=True),
        sa.Column('servable', sa.Boolean, default=True),
        sa.Column('quantized', sa.Boolean, default=False),
        sa.Column('is_gguf', sa.Boolean, default=False),
        sa.Column('stage', sa.String(20), default='staging'),
        sa.Column('status', sa.String(20), default='active'),
        sa.Column('storage_path', sa.String(500), nullable=False),
        sa.Column('size_bytes', sa.Integer, default=0),
        sa.Column('checksum', sa.String(64), nullable=True),
        sa.Column('hf_repo_id', sa.String(255), nullable=True),
        sa.Column('hf_revision', sa.String(64), nullable=True),
        sa.Column('tokenizer_path', sa.String(500), nullable=True),
        sa.Column('vocab_size', sa.Integer, nullable=True),
        sa.Column('tags', postgresql.JSON, default=[]),
        sa.Column('metadata', postgresql.JSON, default={}),
        sa.Column('model_card', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('idx_base_model_name', 'base_models', ['name'])
    op.create_index('idx_base_model_family', 'base_models', ['family'])
    op.create_index('idx_base_model_stage', 'base_models', ['stage'])
    op.create_index('idx_base_model_modality', 'base_models', ['modality'])

    # Create recipes table
    op.create_table(
        'recipes',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False, unique=True),
        sa.Column('display_name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('recipe_type', sa.String(50), nullable=False),
        sa.Column('modality', sa.String(50), nullable=False),
        sa.Column('train_styles', postgresql.JSON, default=[]),
        sa.Column('default_config', postgresql.JSON, default={}),
        sa.Column('required_fields', postgresql.JSON, default=[]),
        sa.Column('optional_fields', postgresql.JSON, default=[]),
        sa.Column('supported_architectures', postgresql.JSON, default=[]),
        sa.Column('min_gpu_memory_gb', sa.Float, nullable=True),
        sa.Column('supports_distributed', sa.Boolean, default=True),
        sa.Column('template_path', sa.String(500), nullable=True),
        sa.Column('script_template', sa.Text, nullable=True),
        sa.Column('tags', postgresql.JSON, default=[]),
        sa.Column('metadata', postgresql.JSON, default={}),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('idx_recipe_type', 'recipes', ['recipe_type'])
    op.create_index('idx_recipe_modality', 'recipes', ['modality'])
    op.create_index('idx_recipe_active', 'recipes', ['is_active'])

    # Create adapters table
    op.create_table(
        'adapters',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('base_model_id', sa.String(36), sa.ForeignKey('base_models.id'), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('adapter_type', sa.String(50), nullable=False),
        sa.Column('rank', sa.Integer, nullable=True),
        sa.Column('alpha', sa.Integer, nullable=True),
        sa.Column('dropout', sa.Float, default=0.0),
        sa.Column('target_modules', postgresql.JSON, default=[]),
        sa.Column('status', sa.String(20), default='training'),
        sa.Column('training_experiment_id', sa.String(36), nullable=True),
        sa.Column('storage_path', sa.String(500), nullable=False),
        sa.Column('size_bytes', sa.Integer, default=0),
        sa.Column('checksum', sa.String(64), nullable=True),
        sa.Column('metrics', postgresql.JSON, default={}),
        sa.Column('tags', postgresql.JSON, default=[]),
        sa.Column('metadata', postgresql.JSON, default={}),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('idx_adapter_base_model', 'adapters', ['base_model_id'])
    op.create_index('idx_adapter_type', 'adapters', ['adapter_type'])
    op.create_index('idx_adapter_status', 'adapters', ['status'])

    # Add new columns to experiments table
    op.add_column('experiments', sa.Column('base_model_id', sa.String(36), sa.ForeignKey('base_models.id'), nullable=True))
    op.add_column('experiments', sa.Column('recipe_id', sa.String(36), sa.ForeignKey('recipes.id'), nullable=True))
    op.add_column('experiments', sa.Column('adapters', postgresql.JSON, default=[]))
    op.add_column('experiments', sa.Column('train', postgresql.JSON, default={}))
    op.add_column('experiments', sa.Column('strategy', postgresql.JSON, default={}))
    op.add_column('experiments', sa.Column('resources', postgresql.JSON, default={}))
    op.add_column('experiments', sa.Column('eval', postgresql.JSON, default={}))
    op.add_column('experiments', sa.Column('export', postgresql.JSON, default=[]))
    op.add_column('experiments', sa.Column('preflight_summary', postgresql.JSON, default={}))

    # Create indexes for new foreign keys
    op.create_index('idx_experiment_base_model', 'experiments', ['base_model_id'])
    op.create_index('idx_experiment_recipe', 'experiments', ['recipe_id'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('idx_experiment_recipe', 'experiments')
    op.drop_index('idx_experiment_base_model', 'experiments')

    # Remove columns from experiments
    op.drop_column('experiments', 'preflight_summary')
    op.drop_column('experiments', 'export')
    op.drop_column('experiments', 'eval')
    op.drop_column('experiments', 'resources')
    op.drop_column('experiments', 'strategy')
    op.drop_column('experiments', 'train')
    op.drop_column('experiments', 'adapters')
    op.drop_column('experiments', 'recipe_id')
    op.drop_column('experiments', 'base_model_id')

    # Drop adapters table
    op.drop_index('idx_adapter_status', 'adapters')
    op.drop_index('idx_adapter_type', 'adapters')
    op.drop_index('idx_adapter_base_model', 'adapters')
    op.drop_table('adapters')

    # Drop recipes table
    op.drop_index('idx_recipe_active', 'recipes')
    op.drop_index('idx_recipe_modality', 'recipes')
    op.drop_index('idx_recipe_type', 'recipes')
    op.drop_table('recipes')

    # Drop base_models table
    op.drop_index('idx_base_model_modality', 'base_models')
    op.drop_index('idx_base_model_stage', 'base_models')
    op.drop_index('idx_base_model_family', 'base_models')
    op.drop_index('idx_base_model_name', 'base_models')
    op.drop_table('base_models')
