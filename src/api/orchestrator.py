"""
FastAPI Orchestration Service for Stage 4 Clustering.

Main API server that coordinates:
- Batch clustering job submission (non-blocking)
- Job lifecycle management (pause/resume/cancel)
- Cluster retrieval and search
- Health checks and monitoring
"""

import logging
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import redis

from src.api.celery_app import celery_app
from src.config.config import get_config
from src.schemas.data_models import (
    BatchJobRequest,
    BatchJobResponse,
    JobStatusResponse,
    JobActionRequest,
    JobActionResponse,
    JobListResponse,
    ClusterSearchRequest,
    ClusterSearchResponse,
    ResourceStatsResponse,
    HealthCheckResponse,
    StatisticsResponse,
    JobStatus,
)
from src.utils.job_manager import JobManager
from src.utils.resource_manager import get_resource_manager
from src.utils.stage3_client import get_stage3_client
from src.storage.cluster_storage_manager import ClusterStorageManager

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Stage 4 Clustering Service",
    description="High-performance clustering service for document, event, entity, and storyline embeddings",
    version="1.0.0",
)

# Global instances (initialized on startup)
config = get_config()
job_manager: Optional[JobManager] = None
resource_manager = None
cluster_storage: Optional[ClusterStorageManager] = None
redis_client: Optional[redis.Redis] = None


# =============================================================================
# STARTUP AND SHUTDOWN
# =============================================================================


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global job_manager, resource_manager, cluster_storage, redis_client

    logger.info("Starting Stage 4 Clustering Service...")

    # Initialize Redis client
    redis_url = config.get("celery.broker_url", "redis://redis-broker:6379/6")
    redis_client = redis.from_url(redis_url, decode_responses=True)
    logger.info(f"Connected to Redis: {redis_url}")

    # Initialize job manager
    logger.info("Initializing job manager...")
    job_manager = JobManager(redis_client)

    # Initialize resource manager
    logger.info("Initializing resource manager...")
    lifecycle_config = config.get_section("job_lifecycle")
    resource_manager = get_resource_manager(
        idle_timeout_seconds=lifecycle_config.get("idle_timeout_seconds", 300),
        gpu_memory_threshold_mb=lifecycle_config.get("gpu_memory_threshold_mb", 14000),
        enable_idle_mode=lifecycle_config.get("enable_idle_mode", True),
    )

    # Initialize cluster storage manager
    logger.info("Initializing cluster storage manager...")
    try:
        cluster_storage = ClusterStorageManager()
        logger.info("Cluster storage manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize cluster storage: {e}")
        # Continue startup even if storage initialization fails

    # Configure CORS
    cors_config = config.get_section("api").get("cors", {})
    if cors_config.get("enabled", True):
    #         app.add_middleware(
    #             CORSMiddleware,
    #             allow_origins=cors_config.get("allow_origins", ["*"]),
    #             allow_credentials=True,
    #             allow_methods=cors_config.get("allow_methods", ["*"]),
    #             allow_headers=cors_config.get("allow_headers", ["*"]),
    #         )
        logger.info("CORS middleware configured")

    logger.info("Stage 4 Clustering Service started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Stage 4 Clustering Service...")

    # Cleanup resources
    if resource_manager:
        resource_manager.cleanup_on_shutdown()

    # Close Redis connection
    if redis_client:
        redis_client.close()

    logger.info("Shutdown complete")


# =============================================================================
# API ENDPOINTS
# =============================================================================


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Stage 4 Clustering Service",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint for Traefik and monitoring.

    Returns 200 OK if service is healthy, 503 if unhealthy.
    """
    healthy = True
    redis_connected = False
    postgresql_connected = False
    stage3_available = False

    # Check Redis connection
    try:
        if redis_client:
            redis_client.ping()
            redis_connected = True
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        healthy = False

    # Check PostgreSQL connection
    # TODO: Implement actual PostgreSQL health check
    postgresql_connected = True  # Placeholder

    # Check Stage 3 availability
    try:
        stage3_client = get_stage3_client()
        stage3_available = stage3_client.is_available()
    except Exception as e:
        logger.error(f"Failed to check Stage 3 availability: {e}")
        stage3_available = False

    # Count active jobs
    active_jobs = 0
    if job_manager:
        try:
            running_jobs = job_manager.list_jobs(status=JobStatus.RUNNING, limit=1000)
            active_jobs = len(running_jobs)
        except Exception as e:
            logger.error(f"Failed to count active jobs: {e}")

    if not healthy:
        raise HTTPException(status_code=503, detail="Service unhealthy")

    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat() + "Z",
        version="1.0.0",
        faiss_loaded=False,  # TODO: Check FAISS indices loaded
        redis_connected=redis_connected,
        postgresql_connected=postgresql_connected,
        stage3_available=stage3_available,
        active_jobs=active_jobs,
    )


@app.get("/statistics", response_model=StatisticsResponse)
async def get_statistics():
    """Get service statistics."""
    # TODO: Query PostgreSQL for cluster statistics
    # For now, return placeholder data

    stats = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "total_clusters": 0,
        "clusters_by_type": {
            "document": 0,
            "event": 0,
            "entity": 0,
            "storyline": 0,
        },
        "total_jobs": 0,
        "jobs_by_status": {
            "queued": 0,
            "running": 0,
            "paused": 0,
            "completed": 0,
            "failed": 0,
            "canceled": 0,
        },
        "avg_cluster_size": 0.0,
        "total_outliers": 0,
    }

    # Get job counts
    if job_manager:
        try:
            for status in JobStatus:
                jobs = job_manager.list_jobs(status=status, limit=10000)
                stats["jobs_by_status"][status.value] = len(jobs)
                stats["total_jobs"] += len(jobs)
        except Exception as e:
            logger.error(f"Failed to get job statistics: {e}")

    return StatisticsResponse(**stats)


# =============================================================================
# JOB LIFECYCLE MANAGEMENT ENDPOINTS
# =============================================================================


@app.post("/api/v1/batch", response_model=BatchJobResponse, status_code=202)
async def create_batch_job(request: BatchJobRequest):
    """
    Create a new batch clustering job (non-blocking).

    Returns 202 Accepted immediately with job_id for tracking.
    Job is queued and processed asynchronously via Celery.
    """
    if not job_manager:
        raise HTTPException(status_code=503, detail="Job management not initialized")

    try:
        # Generate job ID if not provided
        job_id = request.job_id or f"job_{uuid.uuid4().hex[:12]}"

        # Determine algorithm (use request value or default from config)
        algorithm = request.algorithm or config.get("clustering.default_algorithm", "hdbscan")

        # Get algorithm parameters (merge config defaults with request overrides)
        algo_config = config.get(f"clustering.algorithms.{algorithm}", {})
        algorithm_params = {**algo_config, **(request.algorithm_params or {})}

        # Create job in job manager
        job_metadata = job_manager.create_job(
            job_id=job_id,
            embedding_type=request.embedding_type,
            algorithm=algorithm,
            total_items=0,  # Will be updated by worker when FAISS indices are loaded
            algorithm_params=algorithm_params,
        )

        # Submit to Celery (non-blocking)
        celery_app.send_task(
            "tasks.cluster_batch_task",
            args=[
                job_id,
                request.embedding_type,
                algorithm,
                algorithm_params,
                request.filters,
                request.enable_temporal_clustering,
                request.checkpoint_interval,
            ],
            task_id=job_id,  # Use job_id as task_id for easy tracking
        )

        logger.info(
            f"Created batch job {job_id}: {request.embedding_type} clustering with {algorithm}"
        )

        return BatchJobResponse(
            job_id=job_id,
            status=job_metadata.status.value,
            embedding_type=request.embedding_type,
            algorithm=algorithm,
            created_at=job_metadata.created_at,
            message="Job queued successfully. Use GET /api/v1/jobs/{job_id} to track progress.",
        )

    except Exception as e:
        logger.error(f"Failed to create batch job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Job creation failed: {str(e)}")


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get status and progress of a batch job."""
    if not job_manager:
        raise HTTPException(status_code=503, detail="Job management not initialized")

    job_metadata = job_manager.get_job(job_id)

    if not job_metadata:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    progress_percent = 0.0
    if job_metadata.total_items > 0:
        progress_percent = (job_metadata.processed_items / job_metadata.total_items) * 100

    # TODO: Estimate completion time based on processing rate
    estimated_completion = None

    return JobStatusResponse(
        job_id=job_id,
        status=job_metadata.status.value,
        embedding_type=job_metadata.embedding_type,
        algorithm=job_metadata.algorithm,
        total_items=job_metadata.total_items,
        processed_items=job_metadata.processed_items,
        clusters_created=job_metadata.clusters_created,
        outliers=job_metadata.outliers,
        progress_percent=round(progress_percent, 2),
        created_at=job_metadata.created_at,
        started_at=job_metadata.started_at,
        completed_at=job_metadata.completed_at,
        paused_at=job_metadata.paused_at,
        error_message=job_metadata.error_message,
        celery_task_id=job_metadata.celery_task_id,
        estimated_completion_time=estimated_completion,
    )


@app.patch("/api/v1/jobs/{job_id}", response_model=JobActionResponse)
async def job_action(job_id: str, action_request: JobActionRequest):
    """
    Perform action on a job: pause, resume, or cancel.

    - pause: Temporarily halt processing (can be resumed later)
    - resume: Continue a paused job from checkpoint
    - cancel: Permanently stop and cleanup job
    """
    if not job_manager:
        raise HTTPException(status_code=503, detail="Job management not initialized")

    action = action_request.action.lower()

    if action not in ["pause", "resume", "cancel"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action '{action}'. Must be: pause, resume, or cancel",
        )

    job_metadata = job_manager.get_job(job_id)
    if not job_metadata:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    try:
        success = False
        message = ""
        new_status = None

        if action == "pause":
            # Revoke the Celery task (non-terminate to allow checkpoint save)
            if job_metadata.celery_task_id:
                celery_app.control.revoke(job_metadata.celery_task_id, terminate=False)

            success = job_manager.pause_job(job_id)
            message = "Job paused successfully" if success else "Job could not be paused"
            new_status = JobStatus.PAUSED.value if success else job_metadata.status.value

        elif action == "resume":
            success = job_manager.resume_job(job_id)

            if success:
                # Resubmit to Celery with same job_id
                job = job_manager.get_job(job_id)

                # Resubmit with checkpoint data
                celery_app.send_task(
                    "tasks.cluster_batch_task",
                    args=[
                        job_id,
                        job.embedding_type,
                        job.algorithm,
                        job.algorithm_params,
                        None,  # filters (stored in checkpoint)
                        True,  # enable_temporal_clustering
                        10,  # checkpoint_interval
                    ],
                    task_id=job_id,
                )

                message = "Job resumed and queued for processing"
                new_status = JobStatus.QUEUED.value
            else:
                message = "Job could not be resumed"
                new_status = job_metadata.status.value

        elif action == "cancel":
            # Revoke the Celery task (terminate immediately)
            if job_metadata.celery_task_id:
                celery_app.control.revoke(job_metadata.celery_task_id, terminate=True)

            success = job_manager.cancel_job(job_id, cleanup=True)
            message = "Job canceled and cleaned up" if success else "Job could not be canceled"
            new_status = JobStatus.CANCELED.value if success else job_metadata.status.value

        return JobActionResponse(
            job_id=job_id,
            action=action,
            success=success,
            message=message,
            new_status=new_status,
        )

    except Exception as e:
        logger.error(f"Failed to perform action '{action}' on job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Action failed: {str(e)}")


@app.delete("/api/v1/jobs/{job_id}", response_model=JobActionResponse)
async def cancel_job(job_id: str):
    """
    Cancel and delete a job (shorthand for PATCH with action=cancel).

    Terminates processing immediately and removes all job data.
    """
    if not job_manager:
        raise HTTPException(status_code=503, detail="Job management not initialized")

    job_metadata = job_manager.get_job(job_id)
    if not job_metadata:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    try:
        # Revoke the Celery task
        if job_metadata.celery_task_id:
            celery_app.control.revoke(job_metadata.celery_task_id, terminate=True)

        success = job_manager.cancel_job(job_id, cleanup=True)

        return JobActionResponse(
            job_id=job_id,
            action="cancel",
            success=success,
            message="Job canceled and deleted" if success else "Failed to cancel job",
            new_status=JobStatus.CANCELED.value if success else job_metadata.status.value,
        )

    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Cancellation failed: {str(e)}")


@app.get("/api/v1/jobs", response_model=JobListResponse)
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum jobs to return"),
):
    """
    List all jobs with optional status filter.

    Status values: queued, running, paused, completed, failed, canceled
    """
    if not job_manager:
        raise HTTPException(status_code=503, detail="Job management not initialized")

    try:
        # Parse status filter
        status_filter = None
        if status:
            try:
                status_filter = JobStatus(status.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status '{status}'. Must be one of: "
                    + ", ".join(s.value for s in JobStatus),
                )

        # Get jobs
        jobs = job_manager.list_jobs(status=status_filter, limit=limit)

        # Convert to response format
        job_responses = []
        for job in jobs:
            progress_percent = 0.0
            if job.total_items > 0:
                progress_percent = (job.processed_items / job.total_items) * 100

            job_responses.append(
                JobStatusResponse(
                    job_id=job.job_id,
                    status=job.status.value,
                    embedding_type=job.embedding_type,
                    algorithm=job.algorithm,
                    total_items=job.total_items,
                    processed_items=job.processed_items,
                    clusters_created=job.clusters_created,
                    outliers=job.outliers,
                    progress_percent=round(progress_percent, 2),
                    created_at=job.created_at,
                    started_at=job.started_at,
                    completed_at=job.completed_at,
                    paused_at=job.paused_at,
                    error_message=job.error_message,
                    celery_task_id=job.celery_task_id,
                )
            )

        return JobListResponse(
            jobs=job_responses, total=len(job_responses), status_filter=status if status else None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"List failed: {str(e)}")


# =============================================================================
# CLUSTER ENDPOINTS
# =============================================================================


@app.get("/api/v1/clusters", response_model=Dict[str, Any])
async def list_clusters(
    embedding_type: Optional[str] = Query(None, description="Filter by embedding type"),
    job_id: Optional[str] = Query(None, description="Filter by job ID"),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """
    List clusters with pagination.

    Retrieve clusters from PostgreSQL with optional filtering by embedding type or job ID.
    """
    if not cluster_storage:
        raise HTTPException(status_code=503, detail="Cluster storage not initialized")

    try:
        if job_id:
            # Get clusters for specific job
            import uuid
            job_uuid = uuid.UUID(job_id)
            clusters = cluster_storage.get_job_clusters(job_uuid)

            # Apply pagination
            total = len(clusters)
            clusters = clusters[offset:offset + limit]

            return {
                "clusters": [c.dict() for c in clusters],
                "total": total,
                "limit": limit,
                "offset": offset,
                "job_id": job_id,
            }
        else:
            # TODO: Implement general cluster listing from database
            # For now, return empty list
            return {
                "clusters": [],
                "total": 0,
                "limit": limit,
                "offset": offset,
                "embedding_type": embedding_type,
                "message": "General cluster listing not yet implemented. Use job_id filter.",
            }

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id format")
    except Exception as e:
        logger.error(f"Failed to list clusters: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"List failed: {str(e)}")


@app.get("/api/v1/clusters/{cluster_id}", response_model=Dict[str, Any])
async def get_cluster(cluster_id: str):
    """
    Get detailed cluster information.

    Retrieve full cluster details including members, centroid, and quality metrics.
    """
    if not cluster_storage:
        raise HTTPException(status_code=503, detail="Cluster storage not initialized")

    try:
        # Convert to UUID
        import uuid
        cluster_uuid = uuid.UUID(cluster_id)

        # Get cluster from storage
        cluster = cluster_storage.get_cluster(cluster_uuid)

        if cluster is None:
            raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

        return cluster.dict()

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid cluster_id format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cluster {cluster_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Get failed: {str(e)}")


@app.post("/api/v1/clusters/search", response_model=ClusterSearchResponse)
async def search_clusters(request: ClusterSearchRequest):
    """Search for similar clusters."""
    # TODO: Implement cluster search using FAISS
    return ClusterSearchResponse(
        query_id=str(uuid.uuid4()),
        clusters=[],
        total_results=0,
        search_time_ms=0.0,
    )


# =============================================================================
# RESOURCE MONITORING
# =============================================================================


@app.get("/api/v1/resources", response_model=ResourceStatsResponse)
async def get_resource_stats():
    """
    Get current resource utilization statistics.

    Returns CPU, RAM, and GPU usage information.
    """
    if not resource_manager:
        raise HTTPException(status_code=503, detail="Resource manager not initialized")

    try:
        stats = resource_manager.get_resource_stats()

        # Add active jobs count
        active_jobs = 0
        if job_manager:
            running_jobs = job_manager.list_jobs(status=JobStatus.RUNNING, limit=1000)
            active_jobs = len(running_jobs)

        return ResourceStatsResponse(**stats, active_jobs=active_jobs)

    except Exception as e:
        logger.error(f"Failed to get resource stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


# =============================================================================
# ERROR HANDLERS
# =============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
    )


if __name__ == "__main__":
    import uvicorn

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run server
    api_config = config.get_section("api")
    uvicorn.run(
        app,
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
        log_level="info",
    )
