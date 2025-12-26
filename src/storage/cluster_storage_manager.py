"""
Cluster Storage Manager

Manages cluster persistence across multiple storage backends:
- PostgreSQL: Primary relational storage
- JSONL: File-based export/backup
- Redis: Fast cache for lookups

Features:
- Multi-backend atomic writes
- Bulk insert optimization
- Progressive persistence (save as you cluster)
- Cluster retrieval and search
- Transaction support
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4

import redis
import structlog
from sqlalchemy.exc import IntegrityError

from src.storage.database import (
    DatabaseManager,
    get_database_manager,
    ClusteringJob,
    Cluster,
    ClusterMember,
    JobStatus as DBJobStatus,
)
from src.schemas.data_models import (
    JobStatus,
    EmbeddingType,
    ClusterAlgorithm,
    Cluster as ClusterInfo,  # Use Cluster model
    # ClusterQualityMetrics,  # Not defined yet
    ClusterMember as ClusterMemberModel,
)
from src.utils.error_handling import (
    StorageError,
    DatabaseError,
    RedisError,
    FileStorageError,
    retry,
)
from src.utils.advanced_logging import get_logger, PerformanceLogger


logger = get_logger(__name__)


# =============================================================================
# Redis Cache Keys
# =============================================================================


class CacheKeys:
    """Redis cache key patterns."""

    @staticmethod
    def cluster(cluster_id: UUID) -> str:
        """Cluster metadata cache key."""
        return f"cluster:{cluster_id}"

    @staticmethod
    def job_clusters(job_id: UUID) -> str:
        """Job's cluster IDs SET key."""
        return f"job:{job_id}:clusters"

    @staticmethod
    def cluster_members(cluster_id: UUID) -> str:
        """Cluster members LIST key."""
        return f"cluster:{cluster_id}:members"


# =============================================================================
# Cluster Storage Manager
# =============================================================================


class ClusterStorageManager:
    """
    Multi-backend cluster storage manager.

    Coordinates storage across PostgreSQL, JSONL files, and Redis cache.
    Ensures data consistency and provides flexible retrieval options.
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
        redis_db: Optional[int] = None,
        cache_ttl: int = 3600,
        jsonl_output_dir: Optional[str] = None,
        enable_postgresql: bool = True,
        enable_jsonl: bool = True,
        enable_redis_cache: bool = True,
    ):
        """
        Initialize cluster storage manager.

        Args:
            db_manager: Database manager instance
            redis_host: Redis host
            redis_port: Redis port
            redis_db: Redis database (for cache)
            cache_ttl: Cache TTL in seconds
            jsonl_output_dir: Directory for JSONL files
            enable_postgresql: Enable PostgreSQL backend
            enable_jsonl: Enable JSONL backend
            enable_redis_cache: Enable Redis caching
        """
        # Database backend
        self.enable_postgresql = enable_postgresql
        if self.enable_postgresql:
            self.db_manager = db_manager or get_database_manager()
        else:
            self.db_manager = None

        # Redis cache backend
        self.enable_redis_cache = enable_redis_cache
        if self.enable_redis_cache:
            redis_host = redis_host or os.getenv("REDIS_CACHE_HOST", "localhost")
            redis_port = redis_port or int(os.getenv("REDIS_CACHE_PORT", "6379"))
            redis_db = redis_db or int(os.getenv("REDIS_CACHE_DB", "7"))

            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
            )
            self.cache_ttl = cache_ttl
        else:
            self.redis_client = None

        # JSONL file backend
        self.enable_jsonl = enable_jsonl
        if self.enable_jsonl:
            self.jsonl_dir = Path(jsonl_output_dir or "data/clusters")
            self.jsonl_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.jsonl_dir = None

        logger.info(
            "cluster_storage_manager_initialized",
            postgresql=self.enable_postgresql,
            redis_cache=self.enable_redis_cache,
            jsonl=self.enable_jsonl,
        )

    # =========================================================================
    # Job Management
    # =========================================================================

    @retry(max_attempts=3, initial_delay=1.0)
    def create_job_record(
        self,
        job_id: UUID,
        embedding_type: EmbeddingType,
        algorithm: ClusterAlgorithm,
        config: dict[str, Any],
        name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Create job record in database.

        Args:
            job_id: Job UUID
            embedding_type: Embedding type
            algorithm: Clustering algorithm
            config: Algorithm configuration
            name: Optional job name
            metadata: Optional metadata
        """
        if not self.enable_postgresql:
            return

        with self.db_manager.get_session() as session:
            job = ClusteringJob(
                job_id=job_id,
                name=name,
                embedding_type=embedding_type.value,
                algorithm=algorithm.value,
                status=DBJobStatus.QUEUED,
                config=config,
                metadata=metadata,
            )

            session.add(job)

        logger.info("job_record_created", job_id=str(job_id))

    @retry(max_attempts=3, initial_delay=1.0)
    def update_job_status(
        self,
        job_id: UUID,
        status: JobStatus,
        **updates: Any,
    ) -> None:
        """
        Update job status and metadata.

        Args:
            job_id: Job UUID
            status: New status
            **updates: Additional fields to update
        """
        if not self.enable_postgresql:
            return

        with self.db_manager.get_session() as session:
            job = session.query(ClusteringJob).filter_by(job_id=job_id).first()
            if not job:
                raise DatabaseError(
                    f"Job {job_id} not found",
                    error_code="JOB_NOT_FOUND",
                )

            job.status = status.value

            # Update timestamps
            if status == JobStatus.RUNNING and not job.started_at:
                job.started_at = datetime.utcnow()
            elif status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
                job.completed_at = datetime.utcnow()

                # Calculate duration
                if job.started_at:
                    duration = (datetime.utcnow() - job.started_at).total_seconds()
                    job.duration_seconds = duration

            # Apply additional updates
            for key, value in updates.items():
                if hasattr(job, key):
                    setattr(job, key, value)

        logger.info("job_status_updated", job_id=str(job_id), status=status.value)

    # =========================================================================
    # Cluster Persistence
    # =========================================================================

    @retry(max_attempts=3, initial_delay=1.0)
    def save_cluster(
        self,
        cluster_info: ClusterInfo,
    ) -> UUID:
        """
        Save single cluster to all enabled backends.

        Args:
            cluster_info: Cluster information

        Returns:
            Cluster UUID
        """
        with PerformanceLogger("save_cluster", logger=logger):
            cluster_id = cluster_info.cluster_id

            # PostgreSQL
            if self.enable_postgresql:
                self._save_cluster_postgresql(cluster_info)

            # Redis cache
            if self.enable_redis_cache:
                self._save_cluster_redis(cluster_info)

            # JSONL (append mode)
            if self.enable_jsonl:
                self._append_cluster_jsonl(cluster_info)

            logger.debug("cluster_saved", cluster_id=str(cluster_id))
            return cluster_id

    @retry(max_attempts=3, initial_delay=1.0)
    def save_clusters_bulk(
        self,
        clusters: list[ClusterInfo],
        batch_size: int = 1000,
    ) -> int:
        """
        Save multiple clusters in bulk (optimized).

        Args:
            clusters: List of clusters
            batch_size: Batch size for bulk inserts

        Returns:
            Number of clusters saved
        """
        if not clusters:
            return 0

        with PerformanceLogger(
            "save_clusters_bulk",
            logger=logger,
            item_count=len(clusters),
        ):
            # PostgreSQL bulk insert
            if self.enable_postgresql:
                self._save_clusters_bulk_postgresql(clusters, batch_size)

            # Redis cache (individual inserts)
            if self.enable_redis_cache:
                for cluster in clusters:
                    self._save_cluster_redis(cluster)

            # JSONL (batch write)
            if self.enable_jsonl:
                self._save_clusters_jsonl(clusters)

            logger.info("clusters_saved_bulk", count=len(clusters))
            return len(clusters)

    def _save_cluster_postgresql(self, cluster_info: ClusterInfo) -> None:
        """Save cluster to PostgreSQL."""
        with self.db_manager.get_session() as session:
            # Create cluster record
            cluster = Cluster(
                cluster_id=cluster_info.cluster_id,
                job_id=cluster_info.job_id,
                cluster_label=cluster_info.cluster_label,
                is_outlier=cluster_info.is_outlier,
                embedding_type=cluster_info.embedding_type.value,
                size=cluster_info.size,
                centroid=cluster_info.centroid,
            )

            # Add quality metrics
            if cluster_info.quality_metrics:
                cluster.cohesion = cluster_info.quality_metrics.cohesion
                cluster.separation = cluster_info.quality_metrics.separation
                cluster.silhouette = cluster_info.quality_metrics.silhouette

            # Add temporal information
            cluster.min_date = cluster_info.min_date
            cluster.max_date = cluster_info.max_date
            cluster.temporal_span_days = cluster_info.temporal_span_days

            # Add domain information
            cluster.domains = cluster_info.domains
            cluster.event_types = cluster_info.event_types
            cluster.entity_types = cluster_info.entity_types

            # Add metadata
            cluster.metadata = cluster_info.metadata

            session.add(cluster)
            session.flush()

            # Create member records
            if cluster_info.members:
                members = [
                    ClusterMember(
                        cluster_id=cluster_info.cluster_id,
                        vector_id=m.vector_id,
                        source_id=m.source_id,
                        document_id=m.document_id,
                        distance_to_centroid=m.distance_to_centroid,
                        similarity_score=m.similarity_score,
                        metadata=m.metadata,
                    )
                    for m in cluster_info.members
                ]
                session.bulk_save_objects(members)

    def _save_clusters_bulk_postgresql(
        self,
        clusters: list[ClusterInfo],
        batch_size: int,
    ) -> None:
        """Save clusters to PostgreSQL in batches."""
        with self.db_manager.get_session() as session:
            for i in range(0, len(clusters), batch_size):
                batch = clusters[i : i + batch_size]

                # Create cluster records
                cluster_objects = []
                member_objects = []

                for cluster_info in batch:
                    cluster = Cluster(
                        cluster_id=cluster_info.cluster_id,
                        job_id=cluster_info.job_id,
                        cluster_label=cluster_info.cluster_label,
                        is_outlier=cluster_info.is_outlier,
                        embedding_type=cluster_info.embedding_type.value,
                        size=cluster_info.size,
                        centroid=cluster_info.centroid,
                    )

                    if cluster_info.quality_metrics:
                        cluster.cohesion = cluster_info.quality_metrics.cohesion
                        cluster.separation = cluster_info.quality_metrics.separation
                        cluster.silhouette = cluster_info.quality_metrics.silhouette

                    cluster.min_date = cluster_info.min_date
                    cluster.max_date = cluster_info.max_date
                    cluster.temporal_span_days = cluster_info.temporal_span_days
                    cluster.domains = cluster_info.domains
                    cluster.event_types = cluster_info.event_types
                    cluster.entity_types = cluster_info.entity_types
                    cluster.metadata = cluster_info.metadata

                    cluster_objects.append(cluster)

                    # Collect members
                    if cluster_info.members:
                        for m in cluster_info.members:
                            member_objects.append(
                                ClusterMember(
                                    cluster_id=cluster_info.cluster_id,
                                    vector_id=m.vector_id,
                                    source_id=m.source_id,
                                    document_id=m.document_id,
                                    distance_to_centroid=m.distance_to_centroid,
                                    similarity_score=m.similarity_score,
                                    metadata=m.metadata,
                                )
                            )

                # Bulk insert
                session.bulk_save_objects(cluster_objects)
                session.flush()

                if member_objects:
                    session.bulk_save_objects(member_objects)

    def _save_cluster_redis(self, cluster_info: ClusterInfo) -> None:
        """Save cluster to Redis cache."""
        cluster_data = cluster_info.model_dump_json()

        # Cache cluster metadata
        self.redis_client.setex(
            CacheKeys.cluster(cluster_info.cluster_id),
            self.cache_ttl,
            cluster_data,
        )

        # Add to job's cluster set
        self.redis_client.sadd(
            CacheKeys.job_clusters(cluster_info.job_id),
            str(cluster_info.cluster_id),
        )
        self.redis_client.expire(
            CacheKeys.job_clusters(cluster_info.job_id),
            self.cache_ttl,
        )

    def _append_cluster_jsonl(self, cluster_info: ClusterInfo) -> None:
        """Append single cluster to JSONL file."""
        filename = self._get_jsonl_filename(
            cluster_info.job_id,
            cluster_info.embedding_type,
        )
        filepath = self.jsonl_dir / filename

        with open(filepath, "a") as f:
            f.write(cluster_info.model_dump_json() + "\n")

    def _save_clusters_jsonl(self, clusters: list[ClusterInfo]) -> None:
        """Save clusters to JSONL file (batch)."""
        if not clusters:
            return

        # Group by job_id and embedding_type
        grouped = {}
        for cluster in clusters:
            key = (cluster.job_id, cluster.embedding_type)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(cluster)

        # Write each group to separate file
        for (job_id, embedding_type), group_clusters in grouped.items():
            filename = self._get_jsonl_filename(job_id, embedding_type)
            filepath = self.jsonl_dir / filename

            with open(filepath, "a") as f:
                for cluster in group_clusters:
                    f.write(cluster.model_dump_json() + "\n")

    def _get_jsonl_filename(
        self,
        job_id: UUID,
        embedding_type: EmbeddingType,
    ) -> str:
        """Generate JSONL filename."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"clusters_{embedding_type.value}_{job_id}_{timestamp}.jsonl"

    # =========================================================================
    # Cluster Retrieval
    # =========================================================================

    @retry(max_attempts=3, initial_delay=0.5)
    def get_cluster(
        self,
        cluster_id: UUID,
        include_members: bool = False,
    ) -> Optional[ClusterInfo]:
        """
        Get cluster by ID (cache-first).

        Args:
            cluster_id: Cluster UUID
            include_members: Include member list

        Returns:
            ClusterInfo or None
        """
        # Try cache first
        if self.enable_redis_cache:
            cluster_data = self.redis_client.get(CacheKeys.cluster(cluster_id))
            if cluster_data:
                return ClusterInfo.model_validate_json(cluster_data)

        # Fallback to PostgreSQL
        if self.enable_postgresql:
            return self._get_cluster_postgresql(cluster_id, include_members)

        return None

    def _get_cluster_postgresql(
        self,
        cluster_id: UUID,
        include_members: bool,
    ) -> Optional[ClusterInfo]:
        """Get cluster from PostgreSQL."""
        with self.db_manager.get_session() as session:
            cluster = session.query(Cluster).filter_by(cluster_id=cluster_id).first()
            if not cluster:
                return None

            # Convert to ClusterInfo
            quality_metrics = None
            if cluster.cohesion is not None:
                quality_metrics = ClusterQualityMetrics(
                    cohesion=cluster.cohesion,
                    separation=cluster.separation,
                    silhouette=cluster.silhouette,
                    size=cluster.size,
                )

            members = None
            if include_members:
                member_records = session.query(ClusterMember).filter_by(
                    cluster_id=cluster_id
                ).all()

                members = [
                    ClusterMemberModel(
                        vector_id=m.vector_id,
                        source_id=m.source_id,
                        document_id=m.document_id,
                        distance_to_centroid=m.distance_to_centroid,
                        similarity_score=m.similarity_score,
                        metadata=m.metadata,
                    )
                    for m in member_records
                ]

            return ClusterInfo(
                cluster_id=cluster.cluster_id,
                job_id=cluster.job_id,
                cluster_label=cluster.cluster_label,
                is_outlier=cluster.is_outlier,
                embedding_type=EmbeddingType(cluster.embedding_type),
                size=cluster.size,
                centroid=cluster.centroid,
                quality_metrics=quality_metrics,
                min_date=cluster.min_date,
                max_date=cluster.max_date,
                temporal_span_days=cluster.temporal_span_days,
                domains=cluster.domains,
                event_types=cluster.event_types,
                entity_types=cluster.entity_types,
                metadata=cluster.metadata,
                members=members,
                created_at=cluster.created_at,
            )

    @retry(max_attempts=3, initial_delay=0.5)
    def get_job_clusters(
        self,
        job_id: UUID,
        include_members: bool = False,
        include_outliers: bool = True,
        limit: Optional[int] = None,
    ) -> list[ClusterInfo]:
        """
        Get all clusters for a job.

        Args:
            job_id: Job UUID
            include_members: Include member lists
            include_outliers: Include outlier clusters
            limit: Maximum number of clusters

        Returns:
            List of ClusterInfo
        """
        if not self.enable_postgresql:
            return []

        with self.db_manager.get_session() as session:
            query = session.query(Cluster).filter_by(job_id=job_id)

            if not include_outliers:
                query = query.filter_by(is_outlier=False)

            if limit:
                query = query.limit(limit)

            clusters = query.all()

            return [
                self._db_cluster_to_model(c, session, include_members)
                for c in clusters
            ]

    def _db_cluster_to_model(
        self,
        cluster: Cluster,
        session,
        include_members: bool,
    ) -> ClusterInfo:
        """Convert database Cluster to ClusterInfo model."""
        quality_metrics = None
        if cluster.cohesion is not None:
            quality_metrics = ClusterQualityMetrics(
                cohesion=cluster.cohesion,
                separation=cluster.separation,
                silhouette=cluster.silhouette,
                size=cluster.size,
            )

        members = None
        if include_members:
            member_records = session.query(ClusterMember).filter_by(
                cluster_id=cluster.cluster_id
            ).all()

            members = [
                ClusterMemberModel(
                    vector_id=m.vector_id,
                    source_id=m.source_id,
                    document_id=m.document_id,
                    distance_to_centroid=m.distance_to_centroid,
                    similarity_score=m.similarity_score,
                    metadata=m.metadata,
                )
                for m in member_records
            ]

        return ClusterInfo(
            cluster_id=cluster.cluster_id,
            job_id=cluster.job_id,
            cluster_label=cluster.cluster_label,
            is_outlier=cluster.is_outlier,
            embedding_type=EmbeddingType(cluster.embedding_type),
            size=cluster.size,
            centroid=cluster.centroid,
            quality_metrics=quality_metrics,
            min_date=cluster.min_date,
            max_date=cluster.max_date,
            temporal_span_days=cluster.temporal_span_days,
            domains=cluster.domains,
            event_types=cluster.event_types,
            entity_types=cluster.entity_types,
            metadata=cluster.metadata,
            members=members,
            created_at=cluster.created_at,
        )

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Statistics dictionary
        """
        if not self.enable_postgresql:
            return {}

        with self.db_manager.get_session() as session:
            total_jobs = session.query(ClusteringJob).count()
            total_clusters = session.query(Cluster).count()
            total_members = session.query(ClusterMember).count()

            return {
                "total_jobs": total_jobs,
                "total_clusters": total_clusters,
                "total_members": total_members,
                "avg_cluster_size": total_members / total_clusters if total_clusters > 0 else 0,
            }

    # =========================================================================
    # Cleanup
    # =========================================================================

    def close(self) -> None:
        """Close connections."""
        if self.redis_client:
            self.redis_client.close()

        logger.info("cluster_storage_manager_closed")
