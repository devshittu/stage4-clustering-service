"""
Celery Application Configuration for Stage 4 Clustering Service.

Configures Celery for distributed clustering task processing.
"""

import os
import logging
from celery import Celery

from src.config.config import get_config

logger = logging.getLogger(__name__)

# Load configuration
config = get_config()

# Get Celery broker and backend URLs from environment or config
CELERY_BROKER = os.getenv("CELERY_BROKER_URL") or config.get(
    "celery.broker_url", "redis://redis-broker:6379/6"
)
CELERY_BACKEND = os.getenv("CELERY_RESULT_BACKEND") or config.get(
    "celery.result_backend", "redis://redis-cache:6379/7"
)

logger.info(f"Initializing Celery app with broker: {CELERY_BROKER}")

# Create Celery app
celery_app = Celery(
    "stage4_clustering_tasks",
    broker=CELERY_BROKER,
    backend=CELERY_BACKEND,
    include=["src.api.celery_worker"],  # Auto-discover tasks
)

# Celery configuration
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Task tracking
    task_track_started=True,
    task_acks_late=True,  # Acknowledge after task completion
    task_reject_on_worker_lost=True,

    # Time limits
    task_time_limit=7200,  # 2 hours hard limit
    task_soft_time_limit=6600,  # 1 hour 50 minutes soft limit

    # Worker settings (optimized for CPU-intensive clustering)
    worker_prefetch_multiplier=1,  # One task at a time (prevents resource contention)
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks (prevent memory leaks)
    worker_disable_rate_limits=False,

    # Result backend
    result_expires=86400,  # Results expire after 24 hours
    result_persistent=True,  # Persist results to backend

    # Connection
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=10,

    # Task routing
    task_routes={
        "tasks.cluster_batch_task": {"queue": "clustering"},
        "tasks.cleanup_old_jobs": {"queue": "maintenance"},
        "tasks.aggregate_statistics": {"queue": "maintenance"},
    },

    # Default queue
    task_default_queue="clustering",
    task_default_exchange="clustering",
    task_default_routing_key="clustering",
)

logger.info("Celery app configured successfully")
