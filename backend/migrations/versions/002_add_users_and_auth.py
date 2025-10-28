"""Add users and authentication tables

Revision ID: 002
Revises: 001
Create Date: 2024-01-15 11:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('username', sa.String(100), nullable=False, unique=True),
        sa.Column('email', sa.String(255), nullable=False, unique=True),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('full_name', sa.String(255), nullable=True),
        sa.Column('role', sa.String(50), nullable=False, default='viewer'),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('is_verified', sa.Boolean, default=False),
        sa.Column('last_login_at', sa.DateTime, nullable=True),
        sa.Column('metadata', postgresql.JSON, default={}),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('idx_user_username', 'users', ['username'])
    op.create_index('idx_user_email', 'users', ['email'])
    op.create_index('idx_user_role', 'users', ['role'])

    # Create teams table
    op.create_table(
        'teams',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False, unique=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('owner_id', sa.String(36), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('metadata', postgresql.JSON, default={}),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('idx_team_name', 'teams', ['name'])
    op.create_index('idx_team_owner', 'teams', ['owner_id'])

    # Create team_members table (many-to-many)
    op.create_table(
        'team_members',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('team_id', sa.String(36), sa.ForeignKey('teams.id'), nullable=False),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('role', sa.String(50), nullable=False, default='member'),
        sa.Column('joined_at', sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index('idx_team_member_team', 'team_members', ['team_id'])
    op.create_index('idx_team_member_user', 'team_members', ['user_id'])
    op.create_index('idx_team_member_unique', 'team_members', ['team_id', 'user_id'], unique=True)

    # Create api_tokens table
    op.create_table(
        'api_tokens',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('token_hash', sa.String(255), nullable=False, unique=True),
        sa.Column('scopes', postgresql.JSON, default=[]),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('expires_at', sa.DateTime, nullable=True),
        sa.Column('last_used_at', sa.DateTime, nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index('idx_token_user', 'api_tokens', ['user_id'])
    op.create_index('idx_token_hash', 'api_tokens', ['token_hash'])

    # Create refresh_tokens table
    op.create_table(
        'refresh_tokens',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('token_hash', sa.String(255), nullable=False, unique=True),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('expires_at', sa.DateTime, nullable=False),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index('idx_refresh_token_user', 'refresh_tokens', ['user_id'])
    op.create_index('idx_refresh_token_hash', 'refresh_tokens', ['token_hash'])

    # Add user_id and team_id to projects
    op.add_column('projects', sa.Column('owner_id', sa.String(36), sa.ForeignKey('users.id'), nullable=True))
    op.add_column('projects', sa.Column('team_id', sa.String(36), sa.ForeignKey('teams.id'), nullable=True))
    op.create_index('idx_project_owner', 'projects', ['owner_id'])
    op.create_index('idx_project_team', 'projects', ['team_id'])

    # Add user_id to jobs
    op.add_column('jobs', sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=True))
    op.create_index('idx_job_user', 'jobs', ['user_id'])


def downgrade() -> None:
    op.drop_index('idx_job_user', 'jobs')
    op.drop_column('jobs', 'user_id')

    op.drop_index('idx_project_team', 'projects')
    op.drop_index('idx_project_owner', 'projects')
    op.drop_column('projects', 'team_id')
    op.drop_column('projects', 'owner_id')

    op.drop_table('refresh_tokens')
    op.drop_table('api_tokens')
    op.drop_table('team_members')
    op.drop_table('teams')
    op.drop_table('users')
