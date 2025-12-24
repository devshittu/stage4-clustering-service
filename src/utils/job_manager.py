"""
Job Lifecycle Manager for Stage 4 Clustering Service.

Manages batch job state, checkpoints, and progress tracking using Redis.
"""

import logging
import json
from typing import Optional, List, Dict, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import redis

from src.schemas.data_models import JobStatus

logger = logging.getLogger(__name__)


@dataclass
class JobMetadata:
    """Job metadata and state."""

    job_id: str
    status: JobStatus
    embedding_type: str
    algorithm: str
    total_items: int
    processed_items: int
    clusters_created: int
    outliers: int
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    paused_at: Optional[str] = None
    error_message: Optional[str] = None
    celery_task_id: Optional[str] = None
    algorithm_params: Optional[Dict[str, Any]] = None


class JobManager:
    """
    Manages job lifecycle using Redis for persistence.

    Responsibilities:
    - Job creation and state management
    - Progress tracking
    - Checkpoint save/load for pause/resume
    - Job cleanup
    """

    def __init__(self, redis_client: redis.Redis):
        """
        Initialize job manager.

        Args:
            redis_client: Redis client for persistence
        """
        self.redis = redis_client
        self.job_prefix = "job:"
        self.checkpoint_prefix = "checkpoint:"
        self.processed_docs_prefix = "processed:"

    def create_job(
        self,
        job_id: str,
        embedding_type: str,
        algorithm: str,
        total_items: int,
        algorithm_params: Optional[Dict[str, Any]] = None,
    ) -> JobMetadata:
        """
        Create a new job.

        Args:
            job_id: Unique job identifier
            embedding_type: Type of embeddings to cluster
            algorithm: Clustering algorithm
            total_items: Total items to process
            algorithm_params: Algorithm parameters

        Returns:
            JobMetadata instance
        """
        job = JobMetadata(
            job_id=job_id,
            status=JobStatus.QUEUED,
            embedding_type=embedding_type,
            algorithm=algorithm,
            total_items=total_items,
            processed_items=0,
            clusters_created=0,
            outliers=0,
            created_at=datetime.utcnow().isoformat() + "Z",
            algorithm_params=algorithm_params,
        )

        self._save_job(job)
        logger.info(f"Created job {job_id}: {embedding_type} clustering with {algorithm}")
        return job

    def get_job(self, job_id: str) -> Optional[JobMetadata]:
        """
        Retrieve job metadata.

        Args:
            job_id: Job identifier

        Returns:
            JobMetadata or None if not found
        """
        key = f"{self.job_prefix}{job_id}"
        data = self.redis.get(key)

        if not data:
            return None

        job_dict = json.loads(data)
        job_dict["status"] = JobStatus(job_dict["status"])
        return JobMetadata(**job_dict)

    def start_job(self, job_id: str, celery_task_id: str) -> bool:
        """
        Mark job as running.

        Args:
            job_id: Job identifier
            celery_task_id: Celery task ID

        Returns:
            True if successful
        """
        job = self.get_job(job_id)
        if not job:
            logger.error(f"Cannot start job {job_id}: not found")
            return False

        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow().isoformat() + "Z"
        job.celery_task_id = celery_task_id

        self._save_job(job)
        logger.info(f"Started job {job_id} (task: {celery_task_id})")
        return True

    def update_progress(
        self,
        job_id: str,
        processed_items: int,
        clusters_created: int = 0,
        outliers: int = 0,
    ) -> bool:
        """
        Update job progress.

        Args:
            job_id: Job identifier
            processed_items: Number of processed items
            clusters_created: Clusters created so far
            outliers: Outliers detected

        Returns:
            True if successful
        """
        job = self.get_job(job_id)
        if not job:
            logger.error(f"Cannot update job {job_id}: not found")
            return False

        job.processed_items = processed_items
        job.clusters_created = clusters_created
        job.outliers = outliers

        self._save_job(job)
        return True

    def complete_job(self, job_id: str, success: bool = True, error: Optional[str] = None) -> bool:
        """
        Mark job as completed or failed.

        Args:
            job_id: Job identifier
            success: Whether job completed successfully
            error: Error message if failed

        Returns:
            True if successful
        """
        job = self.get_job(job_id)
        if not job:
            logger.error(f"Cannot complete job {job_id}: not found")
            return False

        job.status = JobStatus.COMPLETED if success else JobStatus.FAILED
        job.completed_at = datetime.utcnow().isoformat() + "Z"
        job.error_message = error

        self._save_job(job)

        # Cleanup checkpoint if completed successfully
        if success:
            self.delete_checkpoint(job_id)

        logger.info(f"Completed job {job_id}: {'success' if success else 'failed'}")
        return True

    def pause_job(self, job_id: str) -> bool:
        """
        Pause a running job.

        Args:
            job_id: Job identifier

        Returns:
            True if successful
        """
        job = self.get_job(job_id)
        if not job:
            logger.error(f"Cannot pause job {job_id}: not found")
            return False

        if job.status != JobStatus.RUNNING:
            logger.warning(f"Job {job_id} is not running, cannot pause")
            return False

        job.status = JobStatus.PAUSED
        job.paused_at = datetime.utcnow().isoformat() + "Z"

        self._save_job(job)
        logger.info(f"Paused job {job_id}")
        return True

    def resume_job(self, job_id: str) -> bool:
        """
        Resume a paused job.

        Args:
            job_id: Job identifier

        Returns:
            True if successful
        """
        job = self.get_job(job_id)
        if not job:
            logger.error(f"Cannot resume job {job_id}: not found")
            return False

        if job.status != JobStatus.PAUSED:
            logger.warning(f"Job {job_id} is not paused, cannot resume")
            return False

        job.status = JobStatus.QUEUED
        job.paused_at = None

        self._save_job(job)
        logger.info(f"Resumed job {job_id}")
        return True

    def cancel_job(self, job_id: str, cleanup: bool = True) -> bool:
        """
        Cancel a job.

        Args:
            job_id: Job identifier
            cleanup: Whether to delete job data

        Returns:
            True if successful
        """
        job = self.get_job(job_id)
        if not job:
            logger.error(f"Cannot cancel job {job_id}: not found")
            return False

        job.status = JobStatus.CANCELED
        job.completed_at = datetime.utcnow().isoformat() + "Z"

        if cleanup:
            # Delete checkpoint and processed docs tracking
            self.delete_checkpoint(job_id)
            self._delete_processed_docs(job_id)

        self._save_job(job)
        logger.info(f"Canceled job {job_id} (cleanup={cleanup})")
        return True

    def save_checkpoint(self, job_id: str, checkpoint_data: Dict[str, Any]) -> bool:
        """
        Save job checkpoint for resume capability.

        Args:
            job_id: Job identifier
            checkpoint_data: Checkpoint data to save

        Returns:
            True if successful
        """
        key = f"{self.checkpoint_prefix}{job_id}"
        data = json.dumps(checkpoint_data)

        self.redis.set(key, data, ex=86400 * 7)  # 7 day TTL
        logger.debug(f"Saved checkpoint for job {job_id}")
        return True

    def load_checkpoint(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Load job checkpoint.

        Args:
            job_id: Job identifier

        Returns:
            Checkpoint data or None
        """
        key = f"{self.checkpoint_prefix}{job_id}"
        data = self.redis.get(key)

        if not data:
            return None

        return json.loads(data)

    def delete_checkpoint(self, job_id: str) -> bool:
        """
        Delete job checkpoint.

        Args:
            job_id: Job identifier

        Returns:
            True if deleted
        """
        key = f"{self.checkpoint_prefix}{job_id}"
        deleted = self.redis.delete(key)
        if deleted:
            logger.debug(f"Deleted checkpoint for job {job_id}")
        return bool(deleted)

    def mark_item_processed(self, job_id: str, item_id: str, success: bool = True) -> bool:
        """
        Mark an item as processed.

        Args:
            job_id: Job identifier
            item_id: Item identifier
            success: Whether processing was successful

        Returns:
            True if successful
        """
        key = f"{self.processed_docs_prefix}{job_id}"
        self.redis.sadd(key, item_id)
        self.redis.expire(key, 86400 * 7)  # 7 day TTL
        return True

    def get_processed_items(self, job_id: str) -> Set[str]:
        """
        Get set of processed item IDs.

        Args:
            job_id: Job identifier

        Returns:
            Set of processed item IDs
        """
        key = f"{self.processed_docs_prefix}{job_id}"
        items = self.redis.smembers(key)
        return set(items) if items else set()

    def _delete_processed_docs(self, job_id: str) -> bool:
        """Delete processed docs tracking."""
        key = f"{self.processed_docs_prefix}{job_id}"
        deleted = self.redis.delete(key)
        return bool(deleted)

    def list_jobs(self, status: Optional[JobStatus] = None, limit: int = 100) -> List[JobMetadata]:
        """
        List jobs with optional status filter.

        Args:
            status: Filter by status
            limit: Maximum jobs to return

        Returns:
            List of JobMetadata
        """
        pattern = f"{self.job_prefix}*"
        jobs = []

        for key in self.redis.scan_iter(pattern, count=limit):
            data = self.redis.get(key)
            if data:
                job_dict = json.loads(data)
                job_dict["status"] = JobStatus(job_dict["status"])
                job = JobMetadata(**job_dict)

                if status is None or job.status == status:
                    jobs.append(job)

                if len(jobs) >= limit:
                    break

        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs

    def cleanup_old_jobs(self, days: int = 7) -> int:
        """
        Delete jobs older than specified days.

        Args:
            days: Age threshold in days

        Returns:
            Number of jobs deleted
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        pattern = f"{self.job_prefix}*"
        deleted = 0

        for key in self.redis.scan_iter(pattern):
            data = self.redis.get(key)
            if data:
                job_dict = json.loads(data)
                created = datetime.fromisoformat(job_dict["created_at"].replace("Z", ""))

                if created < cutoff:
                    job_id = job_dict["job_id"]
                    self.redis.delete(key)
                    self.delete_checkpoint(job_id)
                    self._delete_processed_docs(job_id)
                    deleted += 1

        logger.info(f"Cleaned up {deleted} jobs older than {days} days")
        return deleted

    def _save_job(self, job: JobMetadata) -> None:
        """Save job metadata to Redis."""
        key = f"{self.job_prefix}{job.job_id}"
        job_dict = asdict(job)
        job_dict["status"] = job.status.value
        data = json.dumps(job_dict)

        self.redis.set(key, data, ex=86400 * 30)  # 30 day TTL
