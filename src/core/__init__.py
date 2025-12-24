"""
Core clustering module for Stage 4.

Exports:
- ClusteringEngine: Main orchestration class
- BaseClusteringAlgorithm: Base class for algorithms
- ClusteringResult: Result container
- ClusteringConfig: Configuration container
- Individual algorithm implementations
"""

from src.core.base_clustering import (
    BaseClusteringAlgorithm,
    ClusteringResult,
    ClusteringConfig,
)
from src.core.clustering_engine import ClusteringEngine
from src.core.hdbscan_algorithm import HDBSCANAlgorithm
from src.core.kmeans_algorithm import KMeansAlgorithm
from src.core.agglomerative_algorithm import AgglomerativeAlgorithm

__all__ = [
    "ClusteringEngine",
    "BaseClusteringAlgorithm",
    "ClusteringResult",
    "ClusteringConfig",
    "HDBSCANAlgorithm",
    "KMeansAlgorithm",
    "AgglomerativeAlgorithm",
]
