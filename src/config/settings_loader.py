"""
settings_loader.py

Configuration management for Stage 4 Clustering Service.
Loads and validates settings from YAML configuration with environment variable substitution.

Features:
- YAML configuration loading with validation
- Environment variable substitution (${VAR_NAME} syntax)
- Singleton pattern for global settings access
- Type-safe configuration with Pydantic models
- Default values and validation
"""

import os
import re
import yaml
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Models (Pydantic)
# =============================================================================

class ServiceSettings(BaseModel):
    """General service settings."""
    name: str = Field(default="stage4-clustering", description="Service name")
    version: str = Field(default="1.0.0", description="Service version")
    environment: str = Field(default="production", description="Environment (development, staging, production)")


class HDBSCANSettings(BaseModel):
    """HDBSCAN clustering algorithm settings."""
    min_cluster_size: int = Field(default=5, ge=2, description="Minimum cluster size")
    min_samples: int = Field(default=3, ge=1, description="Minimum samples")
    cluster_selection_epsilon: float = Field(default=0.0, ge=0.0, description="Cluster selection epsilon")
    metric: str = Field(default="euclidean", description="Distance metric")
    cluster_selection_method: str = Field(default="eom", description="Cluster selection method (eom or leaf)")
    allow_single_cluster: bool = Field(default=False, description="Allow single cluster")


class KMeansSettings(BaseModel):
    """K-Means clustering algorithm settings."""
    n_clusters: int = Field(default=50, ge=2, description="Number of clusters")
    n_init: int = Field(default=10, ge=1, description="Number of initializations")
    max_iter: int = Field(default=300, ge=1, description="Maximum iterations")
    algorithm: str = Field(default="auto", description="Algorithm (auto, full, elkan)")
    random_state: int = Field(default=42, description="Random seed")


class AgglomerativeSettings(BaseModel):
    """Agglomerative clustering algorithm settings."""
    n_clusters: Optional[int] = Field(default=None, description="Number of clusters (null = use distance_threshold)")
    distance_threshold: float = Field(default=0.5, ge=0.0, description="Distance threshold")
    linkage: str = Field(default="ward", description="Linkage method (ward, complete, average, single)")
    affinity: str = Field(default="euclidean", description="Affinity metric")


class ClusteringAlgorithmsSettings(BaseModel):
    """Algorithm-specific settings."""
    hdbscan: HDBSCANSettings = Field(default_factory=HDBSCANSettings)
    kmeans: KMeansSettings = Field(default_factory=KMeansSettings)
    agglomerative: AgglomerativeSettings = Field(default_factory=AgglomerativeSettings)


class TemporalClusteringSettings(BaseModel):
    """Temporal clustering configuration."""
    enabled: bool = Field(default=True, description="Enable temporal clustering")
    decay_factor: int = Field(default=7, ge=1, description="Temporal decay factor in days")
    max_temporal_gap: int = Field(default=30, ge=1, description="Maximum temporal gap within cluster (days)")
    use_temporal_windows: bool = Field(default=True, description="Use sliding temporal windows")
    window_size: int = Field(default=14, ge=1, description="Sliding window size in days")


class MetadataFilteringSettings(BaseModel):
    """Metadata-aware clustering configuration."""
    enabled: bool = Field(default=True, description="Enable metadata filtering")
    filter_by_domain: bool = Field(default=True, description="Filter by domain")
    filter_by_event_type: bool = Field(default=True, description="Filter by event type")
    filter_by_entity_type: bool = Field(default=True, description="Filter by entity type")


class QualitySettings(BaseModel):
    """Clustering quality thresholds."""
    min_cluster_size: int = Field(default=3, ge=1, description="Minimum members per cluster")
    min_similarity: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum avg intra-cluster similarity")
    max_outlier_ratio: float = Field(default=0.1, ge=0.0, le=1.0, description="Max % of unclustered items")


class ClusteringLevelSettings(BaseModel):
    """Settings for a specific clustering level."""
    enabled: bool = Field(default=True, description="Enable this level")
    algorithm: str = Field(default="hdbscan", description="Algorithm to use")
    min_cluster_size: Optional[int] = Field(default=None, description="Override min_cluster_size")
    distance_threshold: Optional[float] = Field(default=None, description="Override distance_threshold")
    n_clusters: Optional[int] = Field(default=None, description="Override n_clusters")


class ClusteringLevelsSettings(BaseModel):
    """Multi-level clustering configuration."""
    document: ClusteringLevelSettings = Field(default_factory=lambda: ClusteringLevelSettings(algorithm="hdbscan", min_cluster_size=5))
    event: ClusteringLevelSettings = Field(default_factory=lambda: ClusteringLevelSettings(algorithm="hdbscan", min_cluster_size=5))
    entity: ClusteringLevelSettings = Field(default_factory=lambda: ClusteringLevelSettings(algorithm="agglomerative", distance_threshold=0.3))
    storyline: ClusteringLevelSettings = Field(default_factory=lambda: ClusteringLevelSettings(algorithm="kmeans", n_clusters=20))


class ClusteringSettings(BaseModel):
    """Main clustering configuration."""
    default_algorithm: str = Field(default="hdbscan", description="Default clustering algorithm")
    algorithms: ClusteringAlgorithmsSettings = Field(default_factory=ClusteringAlgorithmsSettings)
    temporal_clustering: TemporalClusteringSettings = Field(default_factory=TemporalClusteringSettings)
    metadata_filtering: MetadataFilteringSettings = Field(default_factory=MetadataFilteringSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)
    levels: ClusteringLevelsSettings = Field(default_factory=ClusteringLevelsSettings)


class FAISSSettings(BaseModel):
    """FAISS configuration."""
    use_gpu: bool = Field(default=True, description="Use GPU acceleration")
    indices_path: str = Field(default="/shared/stage3/data/vector_indices", description="Path to Stage 3 indices")
    load_on_startup: bool = Field(default=False, description="Load indices on startup or lazily")
    cache_indices: bool = Field(default=True, description="Keep indices in memory")
    gpu_memory_fraction: float = Field(default=0.8, ge=0.1, le=1.0, description="GPU memory fraction to use")


class PostgreSQLStorageSettings(BaseModel):
    """PostgreSQL storage configuration."""
    enabled: bool = Field(default=True, description="Enable PostgreSQL storage")
    table_prefix: str = Field(default="clustering_", description="Table name prefix")
    batch_size: int = Field(default=1000, ge=1, description="Bulk insert batch size")


class JSONLStorageSettings(BaseModel):
    """JSONL storage configuration."""
    enabled: bool = Field(default=True, description="Enable JSONL storage")
    output_dir: str = Field(default="data/clusters", description="Output directory")
    file_pattern: str = Field(default="clusters_{embedding_type}_{timestamp}.jsonl", description="File naming pattern")
    pretty_print: bool = Field(default=False, description="Pretty print JSON")


class RedisCacheSettings(BaseModel):
    """Redis cache configuration."""
    enabled: bool = Field(default=True, description="Enable Redis cache")
    ttl: int = Field(default=3600, ge=0, description="Cache TTL in seconds")
    prefix: str = Field(default="cluster:", description="Key prefix")


class StorageSettings(BaseModel):
    """Storage configuration."""
    enabled_backends: List[str] = Field(default_factory=lambda: ["postgresql", "jsonl"], description="Enabled storage backends")
    postgresql: PostgreSQLStorageSettings = Field(default_factory=PostgreSQLStorageSettings)
    jsonl: JSONLStorageSettings = Field(default_factory=JSONLStorageSettings)
    redis_cache: RedisCacheSettings = Field(default_factory=RedisCacheSettings)


class WorkerSettings(BaseModel):
    """Celery worker settings."""
    concurrency: int = Field(default=22, ge=1, description="Worker concurrency")
    prefetch_multiplier: int = Field(default=1, ge=1, description="Prefetch multiplier")
    max_tasks_per_child: int = Field(default=50, ge=1, description="Max tasks per worker process")


class JobSettings(BaseModel):
    """Batch job settings."""
    default_chunk_size: int = Field(default=1000, ge=1, description="Items per batch chunk")
    max_chunk_size: int = Field(default=5000, ge=1, description="Maximum chunk size")
    checkpoint_interval: int = Field(default=100, ge=1, description="Save checkpoint every N items")


class QueueSettings(BaseModel):
    """Queue management settings."""
    name: str = Field(default="clustering", description="Queue name")
    priority_levels: int = Field(default=3, ge=1, description="Number of priority levels")


class BatchSettings(BaseModel):
    """Batch processing configuration."""
    worker: WorkerSettings = Field(default_factory=WorkerSettings)
    job: JobSettings = Field(default_factory=JobSettings)
    queue: QueueSettings = Field(default_factory=QueueSettings)


class JobLifecycleSettings(BaseModel):
    """Job lifecycle management configuration."""
    enable_job_queue: bool = Field(default=True, description="Enable job queue")
    max_concurrent_jobs: int = Field(default=1, ge=1, description="Max concurrent jobs (GPU serialization)")
    job_ttl_days: int = Field(default=7, ge=1, description="Job TTL in days")
    enable_checkpoints: bool = Field(default=True, description="Enable checkpointing")
    checkpoint_interval: int = Field(default=10, ge=1, description="Checkpoint interval")
    checkpoint_ttl_days: int = Field(default=7, ge=1, description="Checkpoint TTL in days")
    enable_resource_management: bool = Field(default=True, description="Enable resource management")
    idle_timeout_seconds: int = Field(default=300, ge=1, description="Idle timeout in seconds")
    gpu_memory_threshold_mb: int = Field(default=14000, ge=1, description="GPU memory threshold in MB")
    enable_idle_mode: bool = Field(default=True, description="Enable idle mode")
    enable_progressive_persistence: bool = Field(default=True, description="Enable progressive persistence")
    persistence_batch_size: int = Field(default=10, ge=1, description="Save every N clusters")


class CORSSettings(BaseModel):
    """CORS configuration."""
    enabled: bool = Field(default=True, description="Enable CORS")
    allow_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000", "http://localhost:8080"])
    allow_methods: List[str] = Field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "PATCH"])
    allow_headers: List[str] = Field(default_factory=lambda: ["*"])


class RateLimitSettings(BaseModel):
    """Rate limiting configuration."""
    enabled: bool = Field(default=True, description="Enable rate limiting")
    requests_per_minute: int = Field(default=60, ge=1, description="Requests per minute")
    burst: int = Field(default=10, ge=1, description="Burst limit")


class PaginationSettings(BaseModel):
    """Pagination configuration."""
    default_limit: int = Field(default=50, ge=1, description="Default page size")
    max_limit: int = Field(default=1000, ge=1, description="Maximum page size")


class APISettings(BaseModel):
    """API configuration."""
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, ge=1, le=65535, description="API port")
    workers: int = Field(default=4, ge=1, description="Number of workers")
    reload: bool = Field(default=False, description="Enable auto-reload")
    cors: CORSSettings = Field(default_factory=CORSSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    pagination: PaginationSettings = Field(default_factory=PaginationSettings)


class ConsoleLoggingSettings(BaseModel):
    """Console logging configuration."""
    enabled: bool = Field(default=True, description="Enable console logging")
    level: str = Field(default="INFO", description="Console log level")


class FileLoggingSettings(BaseModel):
    """File logging configuration."""
    enabled: bool = Field(default=True, description="Enable file logging")
    path: str = Field(default="logs/clustering_service.log", description="Log file path")
    level: str = Field(default="INFO", description="File log level")
    max_size_mb: int = Field(default=100, ge=1, description="Max log file size in MB")
    backup_count: int = Field(default=5, ge=1, description="Number of backup files")
    rotation: str = Field(default="time", description="Rotation strategy (time or size)")


class LoggingSettings(BaseModel):
    """Logging configuration."""
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="json", description="Log format (json or text)")
    include_timestamp: bool = Field(default=True, description="Include timestamp")
    include_correlation_id: bool = Field(default=True, description="Include correlation ID")
    console: ConsoleLoggingSettings = Field(default_factory=ConsoleLoggingSettings)
    file: FileLoggingSettings = Field(default_factory=FileLoggingSettings)
    structured_fields: List[str] = Field(
        default_factory=lambda: ["stage", "service", "cluster_id", "job_id", "embedding_type", "algorithm"]
    )


class PrometheusSettings(BaseModel):
    """Prometheus metrics configuration."""
    enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    port: int = Field(default=9090, ge=1, le=65535, description="Metrics port")
    path: str = Field(default="/metrics", description="Metrics endpoint path")


class HealthCheckSettings(BaseModel):
    """Health check configuration."""
    enabled: bool = Field(default=True, description="Enable health checks")
    path: str = Field(default="/health", description="Health check endpoint")
    include_dependencies: bool = Field(default=True, description="Check dependencies (Stage 3, Redis, PostgreSQL)")


class PerformanceSettings(BaseModel):
    """Performance tracking configuration."""
    track_clustering_time: bool = Field(default=True, description="Track clustering time")
    track_memory_usage: bool = Field(default=True, description="Track memory usage")
    track_gpu_utilization: bool = Field(default=True, description="Track GPU utilization")


class MonitoringSettings(BaseModel):
    """Monitoring and observability configuration."""
    prometheus: PrometheusSettings = Field(default_factory=PrometheusSettings)
    health_check: HealthCheckSettings = Field(default_factory=HealthCheckSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)


class RetrySettings(BaseModel):
    """Retry policy configuration."""
    max_attempts: int = Field(default=3, ge=1, description="Maximum retry attempts")
    backoff_factor: int = Field(default=2, ge=1, description="Backoff multiplier")
    max_backoff: int = Field(default=60, ge=1, description="Maximum backoff in seconds")


class Stage3IntegrationSettings(BaseModel):
    """Stage 3 integration configuration."""
    api_url: str = Field(default="http://embeddings-orchestrator:8000", description="Stage 3 API URL")
    health_check_interval: int = Field(default=60, ge=1, description="Health check interval in seconds")
    health_check_timeout: int = Field(default=10, ge=1, description="Health check timeout in seconds")
    retry: RetrySettings = Field(default_factory=RetrySettings)


class Settings(BaseModel):
    """Root configuration model."""
    service: ServiceSettings = Field(default_factory=ServiceSettings)
    clustering: ClusteringSettings = Field(default_factory=ClusteringSettings)
    faiss: FAISSSettings = Field(default_factory=FAISSSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    batch: BatchSettings = Field(default_factory=BatchSettings)
    job_lifecycle: JobLifecycleSettings = Field(default_factory=JobLifecycleSettings)
    api: APISettings = Field(default_factory=APISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    stage3_integration: Stage3IntegrationSettings = Field(default_factory=Stage3IntegrationSettings)
    domains: List[str] = Field(default_factory=lambda: [
        "diplomatic_relations", "military_operations", "economic_activity",
        "political_events", "legal_judicial", "health_medical", "environmental",
        "technology_science", "cultural_social", "sports_entertainment",
        "infrastructure_development", "general_news"
    ])
    event_types: List[str] = Field(default_factory=lambda: [
        "contact_meet", "contact_phone", "conflict_attack", "conflict_demonstrate",
        "movement_transport", "transaction_transfer", "personnel_elect",
        "personnel_start_position", "personnel_end_position", "justice_arrest",
        "justice_charge", "justice_convict", "policy_change", "economic_indicator",
        "natural_disaster", "technology_release", "treaty_agreement", "election",
        "protest", "legislation"
    ])
    entity_types: List[str] = Field(default_factory=lambda: [
        "PER", "ORG", "LOC", "GPE", "DATE", "TIME", "MONEY", "MISC", "EVENT"
    ])


# =============================================================================
# Configuration Manager (Singleton)
# =============================================================================

class ConfigManager:
    """
    Singleton configuration manager that loads and caches settings.

    Features:
    - Loads YAML configuration from file
    - Substitutes environment variables using ${VAR_NAME} syntax
    - Validates configuration using Pydantic models
    - Provides global access to settings
    """

    _instance: Optional['ConfigManager'] = None
    _settings: Optional[Settings] = None

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def load_config(cls, config_path: Optional[str] = None) -> Settings:
        """
        Load configuration from YAML file with environment variable substitution.

        Args:
            config_path: Path to configuration file. If None, uses default path.

        Returns:
            Settings object with validated configuration

        Raises:
            FileNotFoundError: If configuration file not found
            ValueError: If configuration is invalid
        """
        if cls._settings is not None:
            return cls._settings

        # Determine config path
        if config_path is None:
            # Try multiple default locations
            possible_paths = [
                Path("/app/config/settings.yaml"),
                Path("config/settings.yaml"),
                Path("../config/settings.yaml"),
                Path(os.getenv("CONFIG_PATH", "config/settings.yaml"))
            ]

            config_path_obj = None
            for path in possible_paths:
                if path.exists():
                    config_path_obj = path
                    break

            if config_path_obj is None:
                raise FileNotFoundError(
                    f"Configuration file not found in any of: {[str(p) for p in possible_paths]}"
                )
        else:
            config_path_obj = Path(config_path)
            if not config_path_obj.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")

        logger.info(f"Loading configuration from: {config_path_obj}")

        # Load YAML file
        try:
            with open(config_path_obj, 'r') as f:
                raw_config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load YAML configuration: {e}")
            raise ValueError(f"Invalid YAML configuration: {e}")

        # Substitute environment variables
        config_dict = cls._substitute_env_vars(raw_config)

        # Validate and create Settings object
        try:
            cls._settings = Settings(**config_dict)
            logger.info("Configuration loaded and validated successfully")
            return cls._settings
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise ValueError(f"Invalid configuration: {e}")

    @classmethod
    def get_settings(cls) -> Settings:
        """
        Get cached settings. Loads from default path if not already loaded.

        Returns:
            Settings object
        """
        if cls._settings is None:
            cls.load_config()
        return cls._settings

    @classmethod
    def _substitute_env_vars(cls, config: Any) -> Any:
        """
        Recursively substitute environment variables in configuration.

        Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax.

        Args:
            config: Configuration dictionary or value

        Returns:
            Configuration with substituted values
        """
        if isinstance(config, dict):
            return {k: cls._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [cls._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Match ${VAR_NAME} or ${VAR_NAME:default}
            pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

            def replace_var(match):
                var_name = match.group(1)
                default_value = match.group(2) if match.group(2) is not None else ""
                return os.getenv(var_name, default_value)

            return re.sub(pattern, replace_var, config)
        else:
            return config

    @classmethod
    def reload_config(cls, config_path: Optional[str] = None) -> Settings:
        """
        Reload configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            Reloaded Settings object
        """
        cls._settings = None
        return cls.load_config(config_path)


# =============================================================================
# Convenience Functions
# =============================================================================

def get_settings() -> Settings:
    """
    Get application settings (convenience function).

    Returns:
        Settings object
    """
    return ConfigManager.get_settings()


def get_device() -> str:
    """
    Get device for clustering (cuda or cpu).

    Returns:
        Device string
    """
    settings = get_settings()
    if settings.faiss.use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            else:
                logger.warning("GPU requested but CUDA not available. Falling back to CPU.")
                return "cpu"
        except ImportError:
            logger.warning("PyTorch not available. Falling back to CPU.")
            return "cpu"
    return "cpu"


# =============================================================================
# Module Initialization
# =============================================================================

if __name__ == "__main__":
    # Test configuration loading
    import sys
    logging.basicConfig(level=logging.INFO)

    try:
        settings = ConfigManager.load_config()
        print("Configuration loaded successfully!")
        print(f"Service: {settings.service.name} v{settings.service.version}")
        print(f"Default algorithm: {settings.clustering.default_algorithm}")
        print(f"FAISS GPU: {settings.faiss.use_gpu}")
        print(f"Storage backends: {settings.storage.enabled_backends}")
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        sys.exit(1)
