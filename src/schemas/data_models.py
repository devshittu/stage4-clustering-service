"""
data_models.py

Pydantic data models for Stage 4 Clustering Service.
Defines input/output schemas for clustering jobs, clusters, and API endpoints.

Schema Design:
- Input: Compatible with Stage 3 (Embedding Service) output
- Output: Compatible with Stage 5 (Graph Construction) input
- Internal: Rich structures for clustering state and results
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# ENUMS
# =============================================================================


class JobStatus(str, Enum):
    """Job lifecycle states."""

    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class ClusterAlgorithm(str, Enum):
    """Supported clustering algorithms."""

    HDBSCAN = "hdbscan"
    KMEANS = "kmeans"
    AGGLOMERATIVE = "agglomerative"


class EmbeddingType(str, Enum):
    """Embedding granularity levels."""

    DOCUMENT = "document"
    EVENT = "event"
    ENTITY = "entity"
    STORYLINE = "storyline"


# =============================================================================
# CLUSTER MODELS
# =============================================================================


class ClusterMember(BaseModel):
    """Individual member of a cluster."""

    member_id: str = Field(..., description="Unique member identifier (from Stage 3)")
    embedding_vector: Optional[List[float]] = Field(None, description="Embedding vector (768D)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Member metadata")
    distance_to_centroid: Optional[float] = Field(None, ge=0.0, description="Distance to cluster centroid")
    membership_confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Cluster membership confidence")


class Cluster(BaseModel):
    """Cluster representation."""

    cluster_id: str = Field(..., description="Unique cluster identifier")
    embedding_type: str = Field(..., description="Type of embeddings clustered")
    algorithm: str = Field(..., description="Algorithm used for clustering")
    members: List[ClusterMember] = Field(default_factory=list, description="Cluster members")
    centroid: Optional[List[float]] = Field(None, description="Cluster centroid vector")
    quality_score: float = Field(..., description="Cluster quality metric (0-1)")
    avg_intra_similarity: float = Field(..., description="Average intra-cluster similarity")
    size: int = Field(..., description="Number of members")
    created_at: str = Field(..., description="Cluster creation timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ClusteringResult(BaseModel):
    """Results of a clustering operation."""

    job_id: str
    embedding_type: str
    algorithm: str
    total_items: int
    clusters_created: int
    outliers: int
    avg_cluster_size: float
    quality_metrics: Dict[str, float]
    processing_time_ms: float
    created_at: str


# =============================================================================
# JOB MODELS
# =============================================================================


class BatchJobRequest(BaseModel):
    """Request to create a batch clustering job."""

    job_id: Optional[str] = Field(None, description="Optional job ID (auto-generated if not provided)")
    embedding_type: str = Field(..., description="Type of embeddings to cluster")
    algorithm: Optional[str] = Field(None, description="Algorithm to use (defaults to config)")
    algorithm_params: Optional[Dict[str, Any]] = Field(None, description="Override algorithm parameters")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters for embedding selection")
    enable_temporal_clustering: bool = Field(True, description="Apply temporal weighting")
    checkpoint_interval: int = Field(10, description="Save checkpoint every N items", ge=1)


class BatchJobResponse(BaseModel):
    """Response after creating a batch job."""

    job_id: str
    status: str
    embedding_type: str
    algorithm: str
    created_at: str
    message: str


class JobStatusResponse(BaseModel):
    """Job status and progress information."""

    job_id: str
    status: str
    embedding_type: str
    algorithm: str
    total_items: int
    processed_items: int
    clusters_created: int
    outliers: int
    progress_percent: float
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    paused_at: Optional[str] = None
    error_message: Optional[str] = None
    celery_task_id: Optional[str] = None
    estimated_completion_time: Optional[str] = None


class JobActionRequest(BaseModel):
    """Request to perform action on a job."""

    action: str = Field(..., description="Action: pause, resume, or cancel")


class JobActionResponse(BaseModel):
    """Response after job action."""

    job_id: str
    action: str
    success: bool
    message: str
    new_status: str


class JobListResponse(BaseModel):
    """List of jobs with optional filters."""

    jobs: List[JobStatusResponse]
    total: int
    status_filter: Optional[str] = None


# =============================================================================
# CLUSTER SEARCH/QUERY MODELS
# =============================================================================


class ClusterSearchRequest(BaseModel):
    """Request to search clusters."""

    embedding_type: str = Field(..., description="Type of embeddings")
    query_text: Optional[str] = Field(None, description="Text query to find similar clusters")
    query_vector: Optional[List[float]] = Field(None, description="Vector query")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    min_cluster_size: int = Field(3, ge=1, description="Minimum cluster size")
    top_k: int = Field(10, ge=1, le=100, description="Number of clusters to return")


class ClusterSearchResponse(BaseModel):
    """Response from cluster search."""

    query_id: str
    clusters: List[Cluster]
    total_results: int
    search_time_ms: float


# =============================================================================
# RESOURCE MONITORING
# =============================================================================


class ResourceStatsResponse(BaseModel):
    """Resource utilization statistics."""

    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    gpu_available: bool
    gpu_utilization_percent: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    active_jobs: int
    timestamp: str


# =============================================================================
# HEALTH & STATISTICS
# =============================================================================


class HealthCheckResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    version: str
    faiss_loaded: bool
    redis_connected: bool
    postgresql_connected: bool
    stage3_available: bool
    active_jobs: int


class StatisticsResponse(BaseModel):
    """Service statistics."""

    timestamp: str
    total_clusters: int
    clusters_by_type: Dict[str, int]
    total_jobs: int
    jobs_by_status: Dict[str, int]
    avg_cluster_size: float
    total_outliers: int


# =============================================================================
# ADDITIONAL CLUSTER MODELS FOR SPECIFIC EMBEDDING TYPES
# =============================================================================

class ClusterCentroid(BaseModel):
    """Cluster centroid representation."""
    centroid_vector: List[float] = Field(..., description="Centroid embedding vector")
    exemplar_id: Optional[str] = Field(None, description="ID of exemplar member (most representative)")
    mean_distance: Optional[float] = Field(None, ge=0.0, description="Mean distance to members")


class ClusterStatistics(BaseModel):
    """Statistical information about a cluster."""
    size: int = Field(..., ge=1, description="Number of members")
    avg_intra_cluster_distance: Optional[float] = Field(None, ge=0.0, description="Average intra-cluster distance")
    cohesion_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Cluster cohesion score")
    silhouette_score: Optional[float] = Field(None, ge=-1.0, le=1.0, description="Silhouette coefficient")
    density: Optional[float] = Field(None, ge=0.0, description="Cluster density")
    temporal_span_days: Optional[int] = Field(None, ge=0, description="Temporal span in days")


class DocumentCluster(BaseModel):
    """Document-level cluster."""
    cluster_id: str = Field(..., description="Unique cluster identifier")
    job_id: str = Field(..., description="Associated job ID")
    embedding_type: EmbeddingType = Field(default=EmbeddingType.DOCUMENT)
    algorithm: ClusterAlgorithm = Field(..., description="Algorithm used")
    members: List[ClusterMember] = Field(..., description="Cluster members")
    centroid: ClusterCentroid = Field(..., description="Cluster centroid")
    statistics: ClusterStatistics = Field(..., description="Cluster statistics")
    created_at: str = Field(..., description="Cluster creation time (ISO 8601)")
    label: Optional[str] = Field(None, description="Human-readable cluster label")
    is_outlier_cluster: bool = Field(default=False, description="Whether this is an outlier cluster")
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall quality score")

    # Document-specific fields
    primary_topics: Optional[List[str]] = Field(None, description="Primary topics in cluster")
    temporal_range: Optional[Tuple[str, str]] = Field(None, description="(start_date, end_date)")
    geographic_focus: Optional[List[str]] = Field(None, description="Geographic locations mentioned")


class EventCluster(BaseModel):
    """Event-level cluster."""
    cluster_id: str = Field(..., description="Unique cluster identifier")
    job_id: str = Field(..., description="Associated job ID")
    embedding_type: EmbeddingType = Field(default=EmbeddingType.EVENT)
    algorithm: ClusterAlgorithm = Field(..., description="Algorithm used")
    members: List[ClusterMember] = Field(..., description="Cluster members")
    centroid: ClusterCentroid = Field(..., description="Cluster centroid")
    statistics: ClusterStatistics = Field(..., description="Cluster statistics")
    created_at: str = Field(..., description="Cluster creation time (ISO 8601)")
    label: Optional[str] = Field(None, description="Human-readable cluster label")
    is_outlier_cluster: bool = Field(default=False, description="Whether this is an outlier cluster")
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall quality score")

    # Event-specific fields
    event_types: Optional[List[str]] = Field(None, description="Event types in cluster")
    primary_entities: Optional[List[str]] = Field(None, description="Key entities across events")
    temporal_pattern: Optional[str] = Field(None, description="Temporal pattern (e.g., 'weekly', 'sporadic')")
    causal_chain: Optional[List[str]] = Field(None, description="Event IDs in causal order")


class EntityCluster(BaseModel):
    """Entity-level cluster."""
    cluster_id: str = Field(..., description="Unique cluster identifier")
    job_id: str = Field(..., description="Associated job ID")
    embedding_type: EmbeddingType = Field(default=EmbeddingType.ENTITY)
    algorithm: ClusterAlgorithm = Field(..., description="Algorithm used")
    members: List[ClusterMember] = Field(..., description="Cluster members")
    centroid: ClusterCentroid = Field(..., description="Cluster centroid")
    statistics: ClusterStatistics = Field(..., description="Cluster statistics")
    created_at: str = Field(..., description="Cluster creation time (ISO 8601)")
    label: Optional[str] = Field(None, description="Human-readable cluster label")
    is_outlier_cluster: bool = Field(default=False, description="Whether this is an outlier cluster")
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall quality score")

    # Entity-specific fields
    entity_type: Optional[str] = Field(None, description="Primary entity type")
    canonical_form: Optional[str] = Field(None, description="Canonical entity representation")
    aliases: Optional[List[str]] = Field(None, description="Entity aliases/mentions")
    co_occurring_entities: Optional[List[str]] = Field(None, description="Frequently co-occurring entities")


class StorylineCluster(BaseModel):
    """Storyline-level cluster."""
    cluster_id: str = Field(..., description="Unique cluster identifier")
    job_id: str = Field(..., description="Associated job ID")
    embedding_type: EmbeddingType = Field(default=EmbeddingType.STORYLINE)
    algorithm: ClusterAlgorithm = Field(..., description="Algorithm used")
    members: List[ClusterMember] = Field(..., description="Cluster members")
    centroid: ClusterCentroid = Field(..., description="Cluster centroid")
    statistics: ClusterStatistics = Field(..., description="Cluster statistics")
    created_at: str = Field(..., description="Cluster creation time (ISO 8601)")
    label: Optional[str] = Field(None, description="Human-readable cluster label")
    is_outlier_cluster: bool = Field(default=False, description="Whether this is an outlier cluster")
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall quality score")

    # Storyline-specific fields
    storyline_summary: Optional[str] = Field(None, description="Storyline summary")
    key_events: Optional[List[str]] = Field(None, description="IDs of key events")
    narrative_arc: Optional[str] = Field(None, description="Narrative progression description")
    related_storyline_ids: Optional[List[str]] = Field(None, description="Related storyline cluster IDs")


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_job_id(job_id: str) -> bool:
    """
    Validate job ID format.

    Args:
        job_id: Job identifier

    Returns:
        True if valid, False otherwise
    """
    if not job_id or len(job_id) < 1:
        return False
    return True


def create_cluster_id(job_id: str, embedding_type: EmbeddingType, cluster_index: int) -> str:
    """
    Create unique cluster ID.

    Args:
        job_id: Job identifier
        embedding_type: Type of embedding
        cluster_index: Cluster index

    Returns:
        Unique cluster ID
    """
    return f"{job_id}_{embedding_type.value}_cluster_{cluster_index}"


def create_checkpoint_id(job_id: str, checkpoint_index: int) -> str:
    """
    Create unique checkpoint ID.

    Args:
        job_id: Job identifier
        checkpoint_index: Checkpoint index

    Returns:
        Unique checkpoint ID
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{job_id}_checkpoint_{checkpoint_index}_{timestamp}"
