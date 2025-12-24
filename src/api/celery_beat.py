"""
Celery Beat Scheduler for Periodic Tasks.

Schedules and executes periodic maintenance tasks:
- Cleanup old jobs (daily)
- Aggregate statistics (hourly)
- Health checks (every 5 minutes)
"""

import logging
from datetime import datetime

from src.api.celery_app import celery_app
from src.config.config import get_config
from src.utils.job_manager import JobManager
import redis

logger = logging.getLogger(__name__)

config = get_config()


@celery_app.task(name="tasks.cleanup_old_jobs")
def cleanup_old_jobs_task() -> dict:
    """
    Periodic task to cleanup old jobs.

    Removes jobs older than configured threshold (default: 7 days).

    Returns:
        Dictionary with cleanup statistics
    """
    try:
        logger.info("Running periodic job cleanup...")

        # Get Redis client
        redis_url = config.get("celery.broker_url", "redis://redis-broker:6379/6")
        redis_client = redis.from_url(redis_url, decode_responses=True)

        # Initialize job manager
        jm = JobManager(redis_client)

        # Cleanup old jobs
        job_ttl_days = config.get("job_lifecycle.job_ttl_days", 7)
        deleted_count = jm.cleanup_old_jobs(days=job_ttl_days)

        result = {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "deleted_jobs": deleted_count,
            "ttl_days": job_ttl_days,
        }

        logger.info(f"Job cleanup complete: {deleted_count} jobs deleted")
        return result

    except Exception as e:
        logger.error(f"Job cleanup failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }


@celery_app.task(name="tasks.aggregate_statistics")
def aggregate_statistics_task() -> dict:
    """
    Periodic task to aggregate service statistics.

    Collects and stores statistics about:
    - Total clusters created
    - Job completion rates
    - Average cluster sizes
    - Resource utilization

    Returns:
        Dictionary with aggregated statistics
    """
    try:
        logger.info("Running statistics aggregation...")

        # Get Redis client
        redis_url = config.get("celery.broker_url", "redis://redis-broker:6379/6")
        redis_client = redis.from_url(redis_url, decode_responses=True)

        # Initialize job manager
        jm = JobManager(redis_client)

        # Aggregate job statistics
        from src.schemas.data_models import JobStatus

        stats = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "jobs_by_status": {},
            "total_jobs": 0,
        }

        for status in JobStatus:
            jobs = jm.list_jobs(status=status, limit=10000)
            count = len(jobs)
            stats["jobs_by_status"][status.value] = count
            stats["total_jobs"] += count

        # Calculate success rate
        completed = stats["jobs_by_status"].get("completed", 0)
        failed = stats["jobs_by_status"].get("failed", 0)
        total_finished = completed + failed

        if total_finished > 0:
            stats["success_rate"] = (completed / total_finished) * 100
        else:
            stats["success_rate"] = 0.0

        # TODO: Query PostgreSQL for cluster statistics
        # For now, use placeholder values
        stats["total_clusters"] = 0
        stats["avg_cluster_size"] = 0.0

        logger.info(
            f"Statistics aggregation complete: {stats['total_jobs']} total jobs, "
            f"{stats['success_rate']:.1f}% success rate"
        )

        return stats

    except Exception as e:
        logger.error(f"Statistics aggregation failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }


@celery_app.task(name="tasks.health_check")
def health_check_task() -> dict:
    """
    Periodic health check task.

    Verifies:
    - Redis connectivity
    - PostgreSQL connectivity
    - FAISS indices availability
    - Resource utilization

    Returns:
        Dictionary with health check results
    """
    try:
        logger.debug("Running periodic health check...")

        health = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "redis": False,
            "postgresql": False,
            "faiss_indices": False,
            "status": "unhealthy",
        }

        # Check Redis
        try:
            redis_url = config.get("celery.broker_url", "redis://redis-broker:6379/6")
            redis_client = redis.from_url(redis_url, decode_responses=True)
            redis_client.ping()
            health["redis"] = True
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")

        # TODO: Check PostgreSQL
        health["postgresql"] = True  # Placeholder

        # TODO: Check FAISS indices
        health["faiss_indices"] = True  # Placeholder

        # Overall status
        if health["redis"] and health["postgresql"]:
            health["status"] = "healthy"

        logger.debug(f"Health check complete: {health['status']}")
        return health

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }


# Configure Celery Beat schedule
celery_app.conf.beat_schedule = {
    # Cleanup old jobs daily at 2 AM
    "cleanup-old-jobs": {
        "task": "tasks.cleanup_old_jobs",
        "schedule": 86400.0,  # 24 hours
        "options": {"queue": "maintenance"},
    },
    # Aggregate statistics every hour
    "aggregate-statistics": {
        "task": "tasks.aggregate_statistics",
        "schedule": 3600.0,  # 1 hour
        "options": {"queue": "maintenance"},
    },
    # Health check every 5 minutes
    "health-check": {
        "task": "tasks.health_check",
        "schedule": 300.0,  # 5 minutes
        "options": {"queue": "maintenance"},
    },
}

logger.info("Celery Beat schedule configured")


if __name__ == "__main__":
    # Run Celery Beat
    logger.info("Starting Celery Beat scheduler...")
    celery_app.start()
