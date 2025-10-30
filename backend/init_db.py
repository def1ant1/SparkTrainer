"""
Database initialization script.

Creates all tables and sets up initial data.
"""
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from database import init_db, engine, SessionLocal
from models import (
    Base, Project, Dataset, Experiment, Job, Artifact, Evaluation,
    LeaderboardEntry, JobStatus
)


def create_tables():
    """Create all database tables."""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✓ Tables created successfully")


def create_sample_data():
    """Create sample data for development/testing."""
    print("\nCreating sample data...")

    session = SessionLocal()

    try:
        # Check if data already exists
        if session.query(Project).first():
            print("Sample data already exists, skipping...")
            return

        # Create sample project
        project = Project(
            id="proj_sample_001",
            name="Sample ML Project",
            description="A sample project demonstrating SparkTrainer capabilities",
            meta={"created_by": "init_script"}
        )
        session.add(project)

        # Create sample dataset
        dataset = Dataset(
            id="dataset_sample_001",
            project_id=project.id,
            name="sample-video-dataset",
            version="1.0.0",
            modality="video",
            description="Sample video dataset",
            size_bytes=1024 * 1024 * 100,  # 100MB
            num_samples=10,
            storage_path="/app/datasets/sample",
            integrity_checked=True,
            integrity_passed=True,
            statistics={
                "total_duration": 120.0,
                "avg_fps": 30,
                "total_frames": 3600
            }
        )
        session.add(dataset)

        # Create sample experiment
        experiment = Experiment(
            id="exp_sample_001",
            project_id=project.id,
            dataset_id=dataset.id,
            name="BLIP-2 Fine-tuning",
            description="Fine-tune BLIP-2 on sample dataset",
            model_type="vision_language",
            recipe_name="vision_language_recipe",
            status=JobStatus.PENDING,
            total_epochs=3,
            config={
                "learning_rate": 2e-5,
                "batch_size": 8,
                "model": "Salesforce/blip2-opt-2.7b"
            },
            hyperparameters={
                "lr": 2e-5,
                "weight_decay": 0.01,
                "warmup_steps": 100
            }
        )
        session.add(experiment)

        session.commit()
        print("✓ Sample data created successfully")

        print("\nSample Project ID:", project.id)
        print("Sample Dataset ID:", dataset.id)
        print("Sample Experiment ID:", experiment.id)

    except Exception as e:
        session.rollback()
        print(f"✗ Error creating sample data: {e}")
        raise
    finally:
        session.close()


def verify_tables():
    """Verify all tables were created."""
    print("\nVerifying tables...")

    from sqlalchemy import inspect

    inspector = inspect(engine)
    tables = inspector.get_table_names()

    expected_tables = [
        'projects',
        'datasets',
        'experiments',
        'jobs',
        'job_status_transitions',
        'artifacts',
        'evaluations',
        'leaderboard'
    ]

    missing_tables = set(expected_tables) - set(tables)
    extra_tables = set(tables) - set(expected_tables)

    if missing_tables:
        print(f"✗ Missing tables: {missing_tables}")
        return False

    print(f"✓ All {len(expected_tables)} tables created successfully:")
    for table in expected_tables:
        print(f"  - {table}")

    if extra_tables:
        print(f"\nAdditional tables found: {extra_tables}")

    return True


def main():
    """Main initialization function."""
    print("=" * 60)
    print("SparkTrainer Database Initialization")
    print("=" * 60)

    # Check database connection
    try:
        engine.connect()
        print("✓ Database connection successful")
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        print("\nMake sure PostgreSQL is running and DATABASE_URL is set correctly.")
        print(f"Current DATABASE_URL: {os.getenv('DATABASE_URL', 'Not set')}")
        sys.exit(1)

    # Create tables
    try:
        create_tables()
    except Exception as e:
        print(f"✗ Error creating tables: {e}")
        sys.exit(1)

    # Verify tables
    if not verify_tables():
        print("\n✗ Table verification failed")
        sys.exit(1)

    # Create sample data (optional)
    if '--sample-data' in sys.argv:
        try:
            create_sample_data()
        except Exception as e:
            print(f"✗ Error creating sample data: {e}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("✓ Database initialization completed successfully")
    print("=" * 60)

    print("\nNext steps:")
    print("1. Start the backend server: python app.py")
    print("2. Start the Celery worker: celery -A celery_app.celery worker")
    print("3. Access MLflow UI: http://localhost:5001")
    print("4. Access Flower (Celery monitoring): http://localhost:5555")


if __name__ == "__main__":
    main()
