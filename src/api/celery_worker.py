"""
Celery Worker for Batch Clustering Processing.

Handles distributed processing of large-scale clustering jobs with:
- Progressive persistence (save clusters incrementally)
- Checkpointing every N items for pause/resume
- Signal handling for pause/cancel
- Resource monitoring and cleanup
- Support for HDBSCAN, K-Means, and Agglomerative clustering
"""

import logging
import time
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from celery import Task
from celery.signals import worker_ready, worker_shutdown
from celery.exceptions import SoftTimeLimitExceeded, Terminated
import redis

from src.api.celery_app import celery_app
from src.config.config import get_config
from src.schemas.data_models import JobStatus
from src.utils.job_manager import JobManager
from src.utils.resource_manager import get_resource_manager, cleanup_resources
from src.core.clustering_engine import ClusteringEngine
from src.utils.event_publisher import get_event_publisher, EventPublisher

logger = logging.getLogger(__name__)

# Global instances (initialized on worker startup)
config = get_config()
job_manager: Optional[JobManager] = None
resource_manager = None
clustering_engine: Optional[ClusteringEngine] = None
event_publisher: Optional[EventPublisher] = None
faiss_indices: Dict[str, Any] = {}  # Cached FAISS indices


@worker_ready.connect
def on_worker_ready(sender, **kwargs):
    """Initialize services when worker starts."""
    global job_manager, resource_manager, clustering_engine, event_publisher

    logger.info("Initializing Celery worker for clustering...")

    # Initialize Redis client for job management
    redis_url = config.get("celery.broker_url", "redis://redis-broker:6379/6")
    redis_client = redis.from_url(redis_url, decode_responses=True)

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

    # Initialize clustering engine
    logger.info("Initializing clustering engine...")
    clustering_engine = ClusteringEngine()

    # Initialize event publisher
    logger.info("Initializing event publisher...")
    event_publisher = get_event_publisher(redis_client)

    logger.info("Celery worker initialization complete")


@worker_shutdown.connect
def on_worker_shutdown(sender, **kwargs):
    """Cleanup resources on worker shutdown."""
    logger.info("Shutting down Celery worker...")

    try:
        # Cleanup resources
        cleanup_resources()
        logger.info("Worker shutdown cleanup complete")
    except Exception as e:
        logger.error(f"Error during worker shutdown: {e}")


class ClusteringTask(Task):
    """
    Base task class for clustering operations.

    Provides automatic retry logic and error handling.
    """

    autoretry_for = (Exception,)
    retry_kwargs = {"max_retries": 3}
    retry_backoff = True
    retry_backoff_max = 600  # 10 minutes
    retry_jitter = True


def _load_faiss_index(embedding_type: str) -> Tuple[Any, np.ndarray, List[str], List[Dict[str, Any]]]:
    """
    Load FAISS index, vectors, and metadata for given embedding type.

    Args:
        embedding_type: Type of embeddings (document/event/entity/storyline)

    Returns:
        Tuple of (index, vectors, item_ids, metadata_list)
    """
    global faiss_indices

    # Check cache
    if embedding_type in faiss_indices:
        logger.info(f"Using cached FAISS index for {embedding_type}")
        return faiss_indices[embedding_type]

    # Load from disk
    import faiss

    faiss_config = config.get_section("faiss")
    indices_path = faiss_config.get("indices_path", "/shared/stage3/data/vector_indices")
    index_file = os.path.join(indices_path, f"{embedding_type}s.index")

    # Try both .json and .pkl metadata formats
    metadata_file = os.path.join(indices_path, f"{embedding_type}s_metadata.json")
    if not os.path.exists(metadata_file):
        metadata_file = os.path.join(indices_path, f"{embedding_type}s_metadata.pkl")

    logger.info(f"Loading FAISS index from {index_file}")

    if not os.path.exists(index_file):
        raise FileNotFoundError(f"FAISS index not found: {index_file}")

    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    # Load index
    use_gpu = faiss_config.get("use_gpu", True)
    index = faiss.read_index(index_file)

    # Move to GPU if available
    if use_gpu and faiss.get_num_gpus() > 0:
        logger.info(f"Moving {embedding_type} index to GPU")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    # Load metadata (contains item IDs and rich metadata)
    # Support both JSON and pickle formats
    if metadata_file.endswith('.json'):
        import json
        with open(metadata_file, "r") as f:
            metadata_raw = json.load(f)
            # Handle Stage 3 format: {"metadata_store": {...}, "id_map": {...}}
            if isinstance(metadata_raw, dict) and "metadata_store" in metadata_raw:
                # Convert to list format
                metadata = list(metadata_raw["metadata_store"].values())
            elif isinstance(metadata_raw, list):
                metadata = metadata_raw
            else:
                metadata = [metadata_raw]
    else:
        import pickle
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)

    # Extract vectors from index
    vectors = faiss.rev_swig_ptr(index.get_xb(), index.ntotal * index.d)
    vectors = vectors.reshape(index.ntotal, index.d)

    # Extract item IDs and metadata
    item_ids = [m.get("id", m.get("source_id", f"item_{i}")) for i, m in enumerate(metadata)]

    # Cache for future use
    faiss_indices[embedding_type] = (index, vectors, item_ids, metadata)

    logger.info(f"Loaded {index.ntotal} {embedding_type} vectors with metadata")

    return index, vectors, item_ids, metadata


def _extract_clustering_metadata(metadata_list: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """
    Extract metadata fields needed for clustering.

    Args:
        metadata_list: List of metadata dictionaries

    Returns:
        Dictionary with extracted fields (domain, event_type, dates, etc.)
    """
    extracted = {
        "domain": [],
        "event_type": [],
        "entity_type": [],
        "dates": [],
    }

    for meta in metadata_list:
        # Get metadata field (could be nested)
        meta_dict = meta.get("metadata", meta)

        # Extract domain
        extracted["domain"].append(meta_dict.get("domain", None))

        # Extract event_type
        extracted["event_type"].append(meta_dict.get("event_type", None))

        # Extract entity_type
        extracted["entity_type"].append(meta_dict.get("entity_type", None))

        # Extract date (try multiple fields)
        date = meta_dict.get("temporal_reference") or meta_dict.get("publication_date") or meta_dict.get("date")
        if date and isinstance(date, str):
            # Parse date string to timestamp
            from datetime import datetime
            try:
                dt = datetime.fromisoformat(date.replace("Z", "+00:00"))
                timestamp = dt.timestamp() / 86400  # Convert to days since epoch
            except Exception:
                timestamp = None
        else:
            timestamp = None

        extracted["dates"].append(timestamp)

    return extracted


def _perform_clustering(
    vectors: np.ndarray,
    algorithm: str,
    params: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    enable_temporal_clustering: bool = False,
    filters: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, int, Dict[str, float], Optional[np.ndarray]]:
    """
    Perform clustering on vectors using ClusteringEngine.

    Args:
        vectors: Embedding vectors (N x D)
        algorithm: Clustering algorithm (hdbscan/kmeans/agglomerative)
        params: Algorithm parameters
        metadata: Optional metadata for filtering and temporal weighting
        enable_temporal_clustering: Apply temporal decay weighting
        filters: Metadata filters

    Returns:
        Tuple of (cluster_labels, n_clusters, quality_metrics, centroids)
    """
    global clustering_engine

    logger.info(f"Clustering {len(vectors)} vectors with {algorithm}")

    # Get temporal decay factor from config
    temporal_config = config.get_section("clustering").get("temporal_clustering", {})
    temporal_decay_factor = temporal_config.get("decay_factor", 7.0)

    # Use ClusteringEngine
    result = clustering_engine.cluster(
        vectors=vectors,
        algorithm=algorithm,
        algorithm_params=params,
        metadata=metadata,
        enable_temporal_weighting=enable_temporal_clustering,
        temporal_decay_factor=temporal_decay_factor,
        metadata_filters=filters,
    )

    logger.info(
        f"Created {result.n_clusters} clusters ({result.outlier_count} outliers), "
        f"quality: {result.quality_metrics}"
    )

    return result.cluster_labels, result.n_clusters, result.quality_metrics, result.centroids


def _save_clusters_to_storage(
    job_id: str,
    embedding_type: str,
    algorithm: str,
    cluster_labels: np.ndarray,
    item_ids: List[str],
    vectors: np.ndarray,
) -> Dict[str, int]:
    """
    Save clusters to PostgreSQL and JSONL.

    Args:
        job_id: Job identifier
        embedding_type: Type of embeddings
        algorithm: Algorithm used
        cluster_labels: Cluster assignments (-1 = outlier)
        item_ids: Item identifiers
        vectors: Embedding vectors

    Returns:
        Dictionary with save statistics
    """
    # TODO: Implement PostgreSQL storage
    # For now, save to JSONL only

    import json

    storage_config = config.get_section("storage")
    jsonl_config = storage_config.get("jsonl", {})
    output_dir = jsonl_config.get("output_dir", "data/clusters")

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"clusters_{embedding_type}_{job_id}_{timestamp}.jsonl")

    clusters_saved = 0
    outliers_saved = 0

    with open(output_file, "w") as f:
        # Group by cluster
        unique_labels = set(cluster_labels)

        for label in unique_labels:
            if label == -1:
                # Outliers
                outlier_indices = np.where(cluster_labels == -1)[0]
                for idx in outlier_indices:
                    cluster_data = {
                        "cluster_id": "outlier",
                        "member_id": item_ids[idx],
                        "embedding_vector": vectors[idx].tolist(),
                        "job_id": job_id,
                        "embedding_type": embedding_type,
                        "algorithm": algorithm,
                    }
                    f.write(json.dumps(cluster_data) + "\n")
                    outliers_saved += 1
            else:
                # Regular cluster
                cluster_indices = np.where(cluster_labels == label)[0]
                cluster_id = f"{embedding_type}_cluster_{label}"

                # Calculate centroid
                cluster_vectors = vectors[cluster_indices]
                centroid = np.mean(cluster_vectors, axis=0)

                for idx in cluster_indices:
                    # Calculate distance to centroid
                    distance = np.linalg.norm(vectors[idx] - centroid)

                    cluster_data = {
                        "cluster_id": cluster_id,
                        "member_id": item_ids[idx],
                        "embedding_vector": vectors[idx].tolist(),
                        "distance_to_centroid": float(distance),
                        "job_id": job_id,
                        "embedding_type": embedding_type,
                        "algorithm": algorithm,
                    }
                    f.write(json.dumps(cluster_data) + "\n")

                clusters_saved += 1

    logger.info(f"Saved {clusters_saved} clusters and {outliers_saved} outliers to {output_file}")

    return {
        "clusters_saved": clusters_saved,
        "outliers_saved": outliers_saved,
        "output_file": output_file,
    }


@celery_app.task(base=ClusteringTask, bind=True, name="tasks.cluster_batch_task")
def cluster_batch_task(
    self,
    job_id: str,
    embedding_type: str,
    algorithm: str,
    algorithm_params: Dict[str, Any],
    filters: Optional[Dict[str, Any]] = None,
    enable_temporal_clustering: bool = True,
    checkpoint_interval: int = 10,
) -> Dict[str, Any]:
    """
    Main clustering task with full lifecycle management.

    Features:
    - Load FAISS indices from Stage 3
    - Perform clustering (HDBSCAN/K-Means/Agglomerative)
    - Progressive persistence to PostgreSQL + JSONL
    - Checkpointing for pause/resume
    - Signal handling for cancellation
    - Resource monitoring and cleanup

    Args:
        job_id: Job identifier
        embedding_type: Type of embeddings (document/event/entity/storyline)
        algorithm: Clustering algorithm
        algorithm_params: Algorithm parameters
        filters: Metadata filters for embedding selection
        enable_temporal_clustering: Apply temporal weighting
        checkpoint_interval: Save checkpoint every N items

    Returns:
        Dictionary with clustering results
    """
    start_time = time.time()

    logger.info(
        f"Job {job_id}: Starting clustering - {embedding_type} with {algorithm}"
    )

    # Get managers
    jm = job_manager
    rm = resource_manager
    ep = event_publisher

    # Mark job as running
    jm.start_job(job_id, self.request.id)

    # Publish job started event
    if ep:
        ep.publish_job_started(
            job_id=job_id,
            total_items=0,  # Will update when known
            embedding_type=embedding_type,
            algorithm=algorithm,
        )

    # Record activity
    rm.record_activity()

    results = {
        "job_id": job_id,
        "task_id": self.request.id,
        "embedding_type": embedding_type,
        "algorithm": algorithm,
        "total_items": 0,
        "clusters_created": 0,
        "outliers": 0,
        "processing_time_ms": 0,
        "output_file": None,
    }

    try:
        # Load FAISS index and vectors with metadata
        logger.info(f"Job {job_id}: Loading FAISS index for {embedding_type}")
        index, vectors, item_ids, metadata_list = _load_faiss_index(embedding_type)

        total_items = len(item_ids)
        results["total_items"] = total_items

        # Extract clustering-relevant metadata
        clustering_metadata = _extract_clustering_metadata(metadata_list)
        logger.info(
            f"Job {job_id}: Extracted metadata - "
            f"{sum(1 for d in clustering_metadata['domain'] if d)} domains, "
            f"{sum(1 for et in clustering_metadata['event_type'] if et)} event types, "
            f"{sum(1 for d in clustering_metadata['dates'] if d is not None)} dates"
        )

        # Update job with total items
        jm.update_progress(job_id, processed_items=0, clusters_created=0, outliers=0)
        jm.get_job(job_id).total_items = total_items

        # Check for cancellation
        job_metadata = jm.get_job(job_id)
        if not job_metadata or job_metadata.status == JobStatus.CANCELED:
            logger.warning(f"Job {job_id}: Canceled before clustering")
            return results

        # Perform clustering with metadata
        logger.info(f"Job {job_id}: Clustering {total_items} items")
        cluster_labels, n_clusters, quality_metrics, centroids = _perform_clustering(
            vectors=vectors,
            algorithm=algorithm,
            params=algorithm_params,
            metadata=clustering_metadata,
            enable_temporal_clustering=enable_temporal_clustering,
            filters=filters,
        )

        results["clusters_created"] = n_clusters
        results["outliers"] = int(np.sum(cluster_labels == -1))
        results["quality_metrics"] = quality_metrics

        # Update progress
        jm.update_progress(
            job_id,
            processed_items=total_items,
            clusters_created=n_clusters,
            outliers=results["outliers"],
        )

        # Publish progress event
        if ep:
            ep.publish_job_progress(
                job_id=job_id,
                processed_items=total_items,
                total_items=total_items,
                clusters_created=n_clusters,
                outliers=results["outliers"],
            )

        # Progressive persistence: save clusters
        logger.info(f"Job {job_id}: Saving clusters to storage")
        storage_stats = _save_clusters_to_storage(
            job_id, embedding_type, algorithm, cluster_labels, item_ids, vectors
        )

        results["output_file"] = storage_stats["output_file"]

        # Record activity
        rm.record_activity()

        # Check GPU memory
        rm.check_gpu_memory()

        results["processing_time_ms"] = (time.time() - start_time) * 1000

        # Mark job as complete
        jm.complete_job(job_id, success=True)

        # Publish completion event
        if ep:
            ep.publish_job_completed(
                job_id=job_id,
                clusters_created=n_clusters,
                outliers=results["outliers"],
                quality_metrics=quality_metrics,
                processing_time_ms=results["processing_time_ms"],
            )

        logger.info(
            f"Job {job_id}: Complete - {n_clusters} clusters, {results['outliers']} outliers "
            f"in {results['processing_time_ms']:.0f}ms"
        )

        return results

    except (SoftTimeLimitExceeded, Terminated) as e:
        # Task was terminated, save checkpoint
        logger.warning(f"Job {job_id}: Task terminated, saving checkpoint")
        jm.save_checkpoint(
            job_id,
            {
                "timestamp": datetime.utcnow().isoformat(),
                "reason": "task_terminated",
                "partial_results": results,
            },
        )
        jm.pause_job(job_id)

        # Publish paused event
        if ep:
            ep.publish_job_paused(
                job_id=job_id,
                processed_items=results.get("processed_items", 0),
            )

        raise

    except Exception as e:
        logger.error(f"Job {job_id}: Clustering failed: {e}", exc_info=True)
        jm.complete_job(job_id, success=False, error=str(e))

        # Publish failed event
        if ep:
            ep.publish_job_failed(
                job_id=job_id,
                error_message=str(e),
            )

        raise

    finally:
        # Cleanup resources
        logger.info(f"Job {job_id}: Cleaning up resources")
        rm.check_gpu_memory()


if __name__ == "__main__":
    # Run worker
    celery_app.start()
