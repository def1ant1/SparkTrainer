"""
Database connection and session management.
"""
import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool

from models import Base

# Database URL from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://sparktrainer:sparktrainer_dev_pass@localhost:5432/sparktrainer"
)

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before using
    pool_size=10,
    max_overflow=20,
    echo=os.getenv("SQL_DEBUG", "false").lower() == "true"
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database by creating all tables."""
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully")


def drop_db():
    """Drop all tables (use with caution)."""
    Base.metadata.drop_all(bind=engine)
    print("Database dropped successfully")


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.

    Usage:
        with get_db() as db:
            user = db.query(User).first()
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db_session() -> Session:
    """
    Get a database session for dependency injection.

    Usage:
        def my_function(db: Session = Depends(get_db_session)):
            ...
    """
    return SessionLocal()


# Event listener for connection pool
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """Set connection parameters if needed."""
    pass


# Optional: Add connection pooling metrics
class DatabaseMetrics:
    """Track database connection metrics."""

    @staticmethod
    def get_pool_status():
        """Get current connection pool status."""
        pool = engine.pool
        return {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "total": pool.size() + pool.overflow()
        }


# Export
__all__ = [
    "engine",
    "SessionLocal",
    "get_db",
    "get_db_session",
    "init_db",
    "drop_db",
    "DatabaseMetrics"
]
