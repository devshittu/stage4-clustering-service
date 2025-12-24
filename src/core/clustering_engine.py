"""
Clustering Engine - Orchestrates clustering operations.

Main entry point for clustering functionality in Stage 4.
Manages algorithm selection, execution, and result handling.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np

from src.core.base_clustering import (
    BaseClusteringAlgorithm,
    ClusteringResult,
    ClusteringConfig,
)
from src.core.hdbscan_algorithm import HDBSCANAlgorithm
from src.core.kmeans_algorithm import KMeansAlgorithm
from src.core.agglomerative_algorithm import AgglomerativeAlgorithm

logger = logging.getLogger(__name__)


class ClusteringEngine:
    """
    Main clustering engine that orchestrates different algorithms.

    Provides a unified interface for all clustering operations regardless
    of the underlying algorithm.
    """

    # Registry of available algorithms
    ALGORITHMS = {
        "hdbscan": HDBSCANAlgorithm,
        "kmeans": KMeansAlgorithm,
        "agglomerative": AgglomerativeAlgorithm,
    }

    def __init__(self):
        """Initialize clustering engine."""
        logger.info("Initialized ClusteringEngine")

    def cluster(
        self,
        vectors: np.ndarray,
        algorithm: str,
        algorithm_params: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        enable_temporal_weighting: bool = False,
        temporal_decay_factor: float = 7.0,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> ClusteringResult:
        """
        Perform clustering using the specified algorithm.

        Args:
            vectors: Embedding vectors (N x D)
            algorithm: Algorithm name (hdbscan/kmeans/agglomerative)
            algorithm_params: Algorithm-specific parameters
            metadata: Optional metadata for each vector
            enable_temporal_weighting: Apply temporal decay weighting
            temporal_decay_factor: Decay factor in days for temporal weighting
            metadata_filters: Filters to apply before clustering

        Returns:
            ClusteringResult with labels and metrics

        Raises:
            ValueError: If algorithm is not supported
        """
        # Validate algorithm
        algorithm = algorithm.lower()
        if algorithm not in self.ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. "
                f"Supported: {list(self.ALGORITHMS.keys())}"
            )

        logger.info(f"Starting {algorithm} clustering on {len(vectors)} vectors")

        # Create configuration
        config = ClusteringConfig(
            algorithm_name=algorithm,
            params=algorithm_params,
            enable_temporal_weighting=enable_temporal_weighting,
            temporal_decay_factor=temporal_decay_factor,
            metadata_filters=metadata_filters,
        )

        # Get algorithm class and instantiate
        algorithm_class = self.ALGORITHMS[algorithm]
        clusterer = algorithm_class(config)

        # Perform clustering
        result = clusterer.cluster(vectors, metadata)

        logger.info(
            f"{algorithm} clustering complete: {result.n_clusters} clusters, "
            f"{result.outlier_count} outliers"
        )

        return result

    def get_recommended_algorithm(
        self,
        n_vectors: int,
        embedding_type: str,
    ) -> str:
        """
        Recommend clustering algorithm based on dataset characteristics.

        Args:
            n_vectors: Number of vectors to cluster
            embedding_type: Type of embeddings (document/event/entity/storyline)

        Returns:
            Recommended algorithm name
        """
        # Algorithm selection heuristics
        if n_vectors < 100:
            # Small dataset: Use Agglomerative for quality
            return "agglomerative"

        elif n_vectors < 10000:
            # Medium dataset: Use HDBSCAN for density-based clustering
            if embedding_type in ["event", "entity"]:
                # Events and entities benefit from outlier detection
                return "hdbscan"
            else:
                # Documents and storylines: K-Means for speed
                return "kmeans"

        else:
            # Large dataset: Use K-Means or MiniBatch K-Means
            return "kmeans"

    def estimate_optimal_k(
        self,
        vectors: np.ndarray,
        min_k: int = 2,
        max_k: int = 100,
    ) -> int:
        """
        Estimate optimal number of clusters using the elbow method.

        Args:
            vectors: Embedding vectors (N x D)
            min_k: Minimum number of clusters to try
            max_k: Maximum number of clusters to try

        Returns:
            Estimated optimal k
        """
        from sklearn.cluster import KMeans

        # Sample vectors if dataset is too large
        if len(vectors) > 10000:
            sample_indices = np.random.choice(len(vectors), 10000, replace=False)
            sample_vectors = vectors[sample_indices]
        else:
            sample_vectors = vectors

        # Adjust max_k to dataset size
        max_k = min(max_k, len(sample_vectors) - 1)

        # Try different k values
        inertias = []
        k_values = range(min_k, max_k + 1, max(1, (max_k - min_k) // 10))

        for k in k_values:
            kmeans = KMeans(n_clusters=k, n_init=3, max_iter=100, random_state=42)
            kmeans.fit(sample_vectors)
            inertias.append(kmeans.inertia_)

        # Find elbow using second derivative
        if len(inertias) > 2:
            # Calculate rate of change
            deltas = np.diff(inertias)
            second_deltas = np.diff(deltas)

            # Find point where second derivative is maximum (elbow)
            elbow_idx = np.argmax(second_deltas) + 1
            optimal_k = list(k_values)[elbow_idx]
        else:
            # Fallback to middle value
            optimal_k = list(k_values)[len(k_values) // 2]

        logger.info(f"Estimated optimal k={optimal_k} (tried k={min_k} to {max_k})")

        return optimal_k

    def validate_clustering_config(
        self,
        algorithm: str,
        params: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Validate clustering configuration.

        Args:
            algorithm: Algorithm name
            params: Algorithm parameters

        Returns:
            Dictionary of validation errors (empty if valid)
        """
        errors = {}

        if algorithm not in self.ALGORITHMS:
            errors["algorithm"] = f"Unsupported algorithm '{algorithm}'"
            return errors

        # Algorithm-specific validation
        if algorithm == "hdbscan":
            min_cluster_size = params.get("min_cluster_size", 5)
            min_samples = params.get("min_samples", 3)

            if min_cluster_size < 2:
                errors["min_cluster_size"] = "Must be >= 2"

            if min_samples < 1:
                errors["min_samples"] = "Must be >= 1"

        elif algorithm == "kmeans":
            n_clusters = params.get("n_clusters", 50)

            if n_clusters < 2:
                errors["n_clusters"] = "Must be >= 2"

        elif algorithm == "agglomerative":
            n_clusters = params.get("n_clusters")
            distance_threshold = params.get("distance_threshold")

            if n_clusters is None and distance_threshold is None:
                errors["config"] = (
                    "Must specify either n_clusters or distance_threshold"
                )

            if n_clusters is not None and n_clusters < 2:
                errors["n_clusters"] = "Must be >= 2 or None"

        return errors
