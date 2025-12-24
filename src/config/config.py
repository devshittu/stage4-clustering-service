"""
Configuration loader for Stage 4 Clustering Service.

Loads settings from YAML files and environment variables.
"""

import os
import logging
from typing import Dict, Any, Optional
import yaml

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Load configuration from file.

        Args:
            config_path: Path to YAML config file (defaults to config/settings.yaml)
        """
        if config_path is None:
            # Default to config/settings.yaml relative to project root
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            config_path = os.path.join(project_root, "config", "settings.yaml")

        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
            return self._default_config()

        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config or {}
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "service": {"name": "stage4-clustering", "version": "1.0.0"},
            "clustering": {
                "default_algorithm": "hdbscan",
                "algorithms": {
                    "hdbscan": {
                        "min_cluster_size": 5,
                        "min_samples": 3,
                        "metric": "euclidean",
                    },
                    "kmeans": {"n_clusters": 50, "n_init": 10, "max_iter": 300},
                },
            },
            "faiss": {
                "use_gpu": True,
                "indices_path": "/shared/stage3/data/vector_indices",
                "load_on_startup": False,
            },
            "storage": {
                "enabled_backends": ["postgresql", "jsonl"],
                "postgresql": {"enabled": True, "batch_size": 1000},
                "jsonl": {"enabled": True, "output_dir": "data/clusters"},
            },
            "batch": {
                "worker": {"concurrency": 22, "prefetch_multiplier": 1},
                "job": {"default_chunk_size": 1000, "checkpoint_interval": 100},
            },
            "job_lifecycle": {
                "enable_job_queue": True,
                "max_concurrent_jobs": 1,
                "enable_checkpoints": True,
                "checkpoint_interval": 10,
                "enable_resource_management": True,
                "idle_timeout_seconds": 300,
                "gpu_memory_threshold_mb": 14000,
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "cors": {
                    "enabled": True,
                    "allow_origins": ["http://localhost:3000"],
                    "allow_methods": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                },
            },
            "logging": {"level": "INFO", "format": "json"},
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation, e.g., "clustering.default_algorithm")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.

        Args:
            section: Section name (e.g., "clustering", "faiss")

        Returns:
            Configuration section dictionary
        """
        return self.config.get(section, {})

    def update_from_env(self) -> None:
        """Update configuration from environment variables."""
        # Redis
        if os.getenv("REDIS_HOST"):
            self.config.setdefault("redis", {})
            self.config["redis"]["host"] = os.getenv("REDIS_HOST")
            self.config["redis"]["port"] = int(os.getenv("REDIS_PORT", "6379"))
            self.config["redis"]["db"] = int(os.getenv("REDIS_DB", "6"))

        # PostgreSQL
        if os.getenv("POSTGRES_HOST"):
            self.config.setdefault("postgresql", {})
            self.config["postgresql"]["host"] = os.getenv("POSTGRES_HOST")
            self.config["postgresql"]["port"] = int(os.getenv("POSTGRES_PORT", "5432"))
            self.config["postgresql"]["database"] = os.getenv(
                "POSTGRES_DB", "stage4_clustering"
            )
            self.config["postgresql"]["user"] = os.getenv("POSTGRES_USER", "stage4_user")
            self.config["postgresql"]["password"] = os.getenv("POSTGRES_PASSWORD", "")

        # Celery
        if os.getenv("CELERY_BROKER_URL"):
            self.config.setdefault("celery", {})
            self.config["celery"]["broker_url"] = os.getenv("CELERY_BROKER_URL")
            self.config["celery"]["result_backend"] = os.getenv(
                "CELERY_RESULT_BACKEND", os.getenv("CELERY_BROKER_URL")
            )

        # Logging
        if os.getenv("LOG_LEVEL"):
            self.config.setdefault("logging", {})
            self.config["logging"]["level"] = os.getenv("LOG_LEVEL", "INFO")

        logger.debug("Updated configuration from environment variables")


# Global config instance
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get or create singleton Config instance.

    Args:
        config_path: Path to config file

    Returns:
        Config singleton
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = Config(config_path)
        _config_instance.update_from_env()

    return _config_instance


def load_configuration() -> Dict[str, Any]:
    """
    Load full configuration dictionary.

    Returns:
        Configuration dictionary
    """
    config = get_config()
    return config.config
