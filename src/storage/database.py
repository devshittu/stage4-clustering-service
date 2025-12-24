"""
Database Module

Provides PostgreSQL database connectivity and SQLAlchemy models for:
- Clustering jobs metadata
- Cluster definitions
- Cluster member associations
- Connection pooling and session management
"""

import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator, Optional

import structlog
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum as SQLEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker
from sqlalchemy.pool import QueuePool

from src.utils.error_handling import DatabaseError, retry


logger = structlog.get_logger(__name__)


# =============================================================================
# SQLAlchemy Base
# =============================================================================


Base = declarative_base()


# =============================================================================
# Enums
# =============================================================================


class JobStatus(str):
    """Job status enumeration."""

    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EmbeddingType(str):
    """Embedding type enumeration."""

    DOCUMENT = "document"
    EVENT = "event"
    ENTITY = "entity"
    STORYLINE = "storyline"


# =============================================================================
# Database Models
# =============================================================================


class ClusteringJob(Base):
    """
    Model for clustering job metadata.

    Tracks job lifecycle, configuration, and results.
    """

    __tablename__ = "clustering_jobs"

    # Primary Key
    job_id = Column(UUID(as_uuid=True), primary_key=True)

    # Job Identification
    name = Column(String(255), nullable=True)
    embedding_type = Column(String(50), nullable=False, index=True)
    algorithm = Column(String(50), nullable=False)

    # Job Status
    status = Column(String(50), nullable=False, default=JobStatus.QUEUED, index=True)
    progress = Column(Float, default=0.0)  # 0.0 to 1.0

    # Configuration
    config = Column(JSONB, nullable=False)  # Algorithm parameters
    metadata = Column(JSONB, nullable=True)  # Custom metadata

    # Metrics
    total_items = Column(Integer, nullable=True)
    processed_items = Column(Integer, default=0)
    num_clusters = Column(Integer, nullable=True)
    num_outliers = Column(Integer, nullable=True)

    # Quality Metrics
    silhouette_score = Column(Float, nullable=True)
    davies_bouldin_score = Column(Float, nullable=True)
    avg_cluster_size = Column(Float, nullable=True)

    # Error Tracking
    error_message = Column(Text, nullable=True)
    error_details = Column(JSONB, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Checkpoint Support
    checkpoint_path = Column(String(500), nullable=True)
    last_checkpoint_at = Column(DateTime, nullable=True)

    # Resource Management
    gpu_used = Column(Boolean, default=False)
    peak_memory_mb = Column(Float, nullable=True)
    duration_seconds = Column(Float, nullable=True)

    # Relationships
    clusters = relationship(
        "Cluster",
        back_populates="job",
        cascade="all, delete-orphan",
    )

    # Indexes
    __table_args__ = (
        Index("idx_job_status_created", "status", "created_at"),
        Index("idx_job_embedding_type", "embedding_type"),
        Index("idx_job_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<ClusteringJob(job_id={self.job_id}, status={self.status})>"

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "job_id": str(self.job_id),
            "name": self.name,
            "embedding_type": self.embedding_type,
            "algorithm": self.algorithm,
            "status": self.status,
            "progress": self.progress,
            "config": self.config,
            "metadata": self.metadata,
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "num_clusters": self.num_clusters,
            "num_outliers": self.num_outliers,
            "silhouette_score": self.silhouette_score,
            "davies_bouldin_score": self.davies_bouldin_score,
            "avg_cluster_size": self.avg_cluster_size,
            "error_message": self.error_message,
            "error_details": self.error_details,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "checkpoint_path": self.checkpoint_path,
            "last_checkpoint_at": self.last_checkpoint_at.isoformat() if self.last_checkpoint_at else None,
            "gpu_used": self.gpu_used,
            "peak_memory_mb": self.peak_memory_mb,
            "duration_seconds": self.duration_seconds,
        }


class Cluster(Base):
    """
    Model for cluster definitions.

    Represents a single cluster from a clustering job.
    """

    __tablename__ = "clustering_clusters"

    # Primary Key
    cluster_id = Column(UUID(as_uuid=True), primary_key=True)

    # Foreign Keys
    job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("clustering_jobs.job_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Cluster Identification
    cluster_label = Column(Integer, nullable=False)  # Cluster number (0, 1, 2, ...)
    is_outlier = Column(Boolean, default=False, index=True)

    # Cluster Metadata
    embedding_type = Column(String(50), nullable=False, index=True)
    size = Column(Integer, nullable=False)  # Number of members

    # Centroid
    centroid = Column(ARRAY(Float), nullable=True)  # Average embedding vector

    # Quality Metrics
    cohesion = Column(Float, nullable=True)  # Avg intra-cluster distance
    separation = Column(Float, nullable=True)  # Avg inter-cluster distance
    silhouette = Column(Float, nullable=True)  # Cluster silhouette score

    # Temporal Information
    min_date = Column(DateTime, nullable=True)
    max_date = Column(DateTime, nullable=True)
    temporal_span_days = Column(Float, nullable=True)

    # Domain Information
    domains = Column(ARRAY(String), nullable=True)
    event_types = Column(ARRAY(String), nullable=True)
    entity_types = Column(ARRAY(String), nullable=True)

    # Custom Metadata
    metadata = Column(JSONB, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    job = relationship("ClusteringJob", back_populates="clusters")
    members = relationship(
        "ClusterMember",
        back_populates="cluster",
        cascade="all, delete-orphan",
    )

    # Indexes
    __table_args__ = (
        Index("idx_cluster_job_label", "job_id", "cluster_label"),
        Index("idx_cluster_embedding_type", "embedding_type"),
        Index("idx_cluster_is_outlier", "is_outlier"),
        Index("idx_cluster_size", "size"),
    )

    def __repr__(self) -> str:
        return f"<Cluster(cluster_id={self.cluster_id}, label={self.cluster_label}, size={self.size})>"

    def to_dict(self, include_members: bool = False) -> dict[str, Any]:
        """
        Convert model to dictionary.

        Args:
            include_members: Include member list

        Returns:
            Dictionary representation
        """
        data = {
            "cluster_id": str(self.cluster_id),
            "job_id": str(self.job_id),
            "cluster_label": self.cluster_label,
            "is_outlier": self.is_outlier,
            "embedding_type": self.embedding_type,
            "size": self.size,
            "centroid": self.centroid,
            "cohesion": self.cohesion,
            "separation": self.separation,
            "silhouette": self.silhouette,
            "min_date": self.min_date.isoformat() if self.min_date else None,
            "max_date": self.max_date.isoformat() if self.max_date else None,
            "temporal_span_days": self.temporal_span_days,
            "domains": self.domains,
            "event_types": self.event_types,
            "entity_types": self.entity_types,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

        if include_members:
            data["members"] = [m.to_dict() for m in self.members]

        return data


class ClusterMember(Base):
    """
    Model for cluster membership.

    Associates vector IDs with clusters.
    """

    __tablename__ = "clustering_members"

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Foreign Keys
    cluster_id = Column(
        UUID(as_uuid=True),
        ForeignKey("clustering_clusters.cluster_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Member Identification
    vector_id = Column(String(255), nullable=False, index=True)  # From Stage 3
    source_id = Column(String(255), nullable=False, index=True)  # Original doc/event/entity ID
    document_id = Column(String(255), nullable=True, index=True)  # Parent document

    # Distance Metrics
    distance_to_centroid = Column(Float, nullable=True)
    similarity_score = Column(Float, nullable=True)  # Cosine similarity

    # Member Metadata
    metadata = Column(JSONB, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    cluster = relationship("Cluster", back_populates="members")

    # Indexes
    __table_args__ = (
        Index("idx_member_cluster", "cluster_id"),
        Index("idx_member_vector_id", "vector_id"),
        Index("idx_member_source_id", "source_id"),
        Index("idx_member_document_id", "document_id"),
        # Composite index for common queries
        Index("idx_member_cluster_source", "cluster_id", "source_id"),
    )

    def __repr__(self) -> str:
        return f"<ClusterMember(id={self.id}, vector_id={self.vector_id})>"

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "cluster_id": str(self.cluster_id),
            "vector_id": self.vector_id,
            "source_id": self.source_id,
            "document_id": self.document_id,
            "distance_to_centroid": self.distance_to_centroid,
            "similarity_score": self.similarity_score,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# =============================================================================
# Database Connection Manager
# =============================================================================


class DatabaseManager:
    """
    Manages PostgreSQL database connections and sessions.

    Provides connection pooling, session management, and transaction support.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
    ):
        """
        Initialize database manager.

        Args:
            host: PostgreSQL host
            port: PostgreSQL port
            database: Database name
            user: Database user
            password: Database password
            pool_size: Connection pool size
            max_overflow: Max overflow connections
            pool_timeout: Pool timeout in seconds
            pool_recycle: Pool recycle time in seconds
        """
        # Get configuration from environment if not provided
        self.host = host or os.getenv("POSTGRES_HOST", "localhost")
        self.port = port or int(os.getenv("POSTGRES_PORT", "5432"))
        self.database = database or os.getenv("POSTGRES_DB", "stage4_clustering")
        self.user = user or os.getenv("POSTGRES_USER", "stage4_user")
        self.password = password or os.getenv("POSTGRES_PASSWORD", "")

        # Build connection URL
        self.url = (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )

        # Create engine with connection pooling
        self.engine = create_engine(
            self.url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=pool_recycle,
            echo=False,  # Set to True for SQL logging
        )

        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

        logger.info(
            "database_manager_initialized",
            host=self.host,
            port=self.port,
            database=self.database,
            pool_size=pool_size,
        )

    @retry(max_attempts=3, initial_delay=1.0)
    def create_tables(self) -> None:
        """
        Create all database tables.

        Creates tables if they don't exist. Safe to call multiple times.
        """
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("database_tables_created")
        except Exception as e:
            logger.error("database_tables_creation_failed", error=str(e))
            raise DatabaseError(
                f"Failed to create database tables: {str(e)}",
                error_code="TABLE_CREATION_FAILED",
            ) from e

    @retry(max_attempts=3, initial_delay=1.0)
    def drop_tables(self) -> None:
        """
        Drop all database tables.

        WARNING: This will delete all data!
        """
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("database_tables_dropped")
        except Exception as e:
            logger.error("database_tables_drop_failed", error=str(e))
            raise DatabaseError(
                f"Failed to drop database tables: {str(e)}",
                error_code="TABLE_DROP_FAILED",
            ) from e

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions.

        Provides automatic session cleanup and rollback on errors.

        Example:
            with db_manager.get_session() as session:
                job = session.query(ClusteringJob).first()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("database_session_error", error=str(e))
            raise DatabaseError(
                f"Database operation failed: {str(e)}",
                error_code="SESSION_ERROR",
            ) from e
        finally:
            session.close()

    def test_connection(self) -> bool:
        """
        Test database connectivity.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
            logger.info("database_connection_test_passed")
            return True
        except Exception as e:
            logger.error("database_connection_test_failed", error=str(e))
            return False

    def close(self) -> None:
        """Close database connections."""
        self.engine.dispose()
        logger.info("database_manager_closed")


# =============================================================================
# Global Database Instance
# =============================================================================


# Singleton database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """
    Get global database manager instance.

    Returns:
        DatabaseManager instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def initialize_database() -> DatabaseManager:
    """
    Initialize database and create tables.

    Returns:
        DatabaseManager instance
    """
    db_manager = get_database_manager()
    db_manager.create_tables()
    return db_manager
