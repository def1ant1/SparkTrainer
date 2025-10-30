"""Add activity feed table

Revision ID: 004
Revises: 003
Create Date: 2025-10-30 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '004'
down_revision: Union[str, None] = '003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create activities table
    op.create_table(
        'activities',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('event_type', sa.String(50), nullable=False),
        sa.Column('entity_type', sa.String(50), nullable=False),
        sa.Column('entity_id', sa.String(36), nullable=True),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('message', sa.Text, nullable=True),
        sa.Column('status', sa.String(20), nullable=True),
        sa.Column('user_id', sa.String(36), nullable=True),
        sa.Column('project_id', sa.String(36), nullable=True),
        sa.Column('metadata', postgresql.JSON, server_default='{}'),
        sa.Column('read', sa.Boolean, server_default='false', nullable=False),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now(), nullable=False),
    )

    # Create indexes
    op.create_index('idx_activity_type', 'activities', ['event_type'])
    op.create_index('idx_activity_entity', 'activities', ['entity_type', 'entity_id'])
    op.create_index('idx_activity_user', 'activities', ['user_id'])
    op.create_index('idx_activity_project', 'activities', ['project_id'])
    op.create_index('idx_activity_created', 'activities', ['created_at'])
    op.create_index('idx_activity_read', 'activities', ['read'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('idx_activity_read', 'activities')
    op.drop_index('idx_activity_created', 'activities')
    op.drop_index('idx_activity_project', 'activities')
    op.drop_index('idx_activity_user', 'activities')
    op.drop_index('idx_activity_entity', 'activities')
    op.drop_index('idx_activity_type', 'activities')

    # Drop table
    op.drop_table('activities')
