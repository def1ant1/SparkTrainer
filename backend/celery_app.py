"""
Celery application for distributed task processing.
"""
import os
from celery import Celery
from kombu import Queue, Exchange

# Celery configuration
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# Create Celery app
celery = Celery(
    "sparktrainer",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["celery_tasks"]  # Import tasks module
)

# Celery configuration
celery.conf.update(
    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_persistent=True,

    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task execution settings
    task_acks_late=True,  # Acknowledge tasks after completion
    task_reject_on_worker_lost=True,
    task_track_started=True,
    task_send_sent_event=True,

    # Worker settings
    worker_prefetch_multiplier=1,  # Only fetch one task at a time (important for GPU tasks)
    worker_max_tasks_per_child=10,  # Restart worker after 10 tasks to prevent memory leaks

    # Queue configuration
    task_default_queue="default",
    task_default_exchange="default",
    task_default_routing_key="default",

    # Task routes
    task_routes={
        "celery_tasks.train_model": {"queue": "training"},
        "celery_tasks.preprocess_data": {"queue": "preprocessing"},
        "celery_tasks.evaluate_model": {"queue": "evaluation"},
    },

    # Define queues
    task_queues=(
        Queue("default", Exchange("default"), routing_key="default"),
        Queue("training", Exchange("training"), routing_key="training", priority=10),
        Queue("preprocessing", Exchange("preprocessing"), routing_key="preprocessing", priority=5),
        Queue("evaluation", Exchange("evaluation"), routing_key="evaluation", priority=3),
    ),

    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,

    # Error handling
    task_soft_time_limit=7200,  # 2 hours soft limit
    task_time_limit=7500,  # 2.08 hours hard limit (slightly more than soft)
)


@celery.task(bind=True)
def debug_task(self):
    """Debug task to test Celery setup."""
    return f"Request: {self.request!r}"


if __name__ == "__main__":
    celery.start()
