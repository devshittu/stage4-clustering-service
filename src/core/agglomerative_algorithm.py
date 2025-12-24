"""
Agglomerative Hierarchical Clustering Algorithm Implementation.

Agglomerative clustering is ideal for:
- Building hierarchical cluster trees (dendrograms)
- When cluster hierarchy is important
- Small to medium datasets
- Entity coreference resolution
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from src.core.base_clustering import (
    BaseClusteringAlgorithm,
    ClusteringResult,
    ClusteringConfig,
)

logger = logging.getLogger(__name__)


class AgglomerativeAlgorithm(BaseClusteringAlgorithm):
    """
    Agglomerative hierarchical clustering implementation.

    Best for: Entity coreference, hierarchical topic modeling
    Strengths: Builds hierarchy, flexible linkage criteria
    Weaknesses: Slow (O(n³)), high memory usage, not scalable
    """

    def __init__(self, config: ClusteringConfig):
        """
        Initialize Agglomerative algorithm.

        Args:
            config: Clustering configuration
        """
        super().__init__(config)

        # Extract Agglomerative-specific parameters
        self.n_clusters = config.params.get("n_clusters", None)
        self.distance_threshold = config.params.get("distance_threshold", None)
        self.linkage = config.params.get("linkage", "ward")
        self.affinity = config.params.get("affinity", "euclidean")
        self.compute_full_tree = config.params.get("compute_full_tree", "auto")

        # Validate parameters
        if self.n_clusters is None and self.distance_threshold is None:
            # Default to distance_threshold if neither is specified
            self.distance_threshold = 0.5
            logger.warning(
                "Neither n_clusters nor distance_threshold specified, "
                "defaulting to distance_threshold=0.5"
            )

        if self.n_clusters is not None and self.distance_threshold is not None:
            logger.warning(
                "Both n_clusters and distance_threshold specified. "
                "Setting n_clusters=None (distance_threshold takes precedence)"
            )
            self.n_clusters = None

        logger.info(
            f"Initialized Agglomerative: n_clusters={self.n_clusters}, "
            f"distance_threshold={self.distance_threshold}, linkage={self.linkage}"
        )

    def cluster(
        self,
        vectors: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ClusteringResult:
        """
        Perform Agglomerative clustering.

        Args:
            vectors: Embedding vectors (N x D)
            metadata: Optional metadata for filtering and temporal weighting

        Returns:
            ClusteringResult with labels and metrics
        """
        logger.info(f"Starting Agglomerative clustering on {len(vectors)} vectors")

        # Apply metadata filters if configured
        filtered_vectors, filter_indices = self._apply_metadata_filters(vectors, metadata)

        if len(filtered_vectors) == 0:
            logger.warning("All vectors filtered out, returning empty result")
            return ClusteringResult(
                cluster_labels=np.array([-1] * len(vectors)),
                n_clusters=0,
                outlier_count=len(vectors),
                quality_metrics={},
            )

        # Check dataset size (Agglomerative is O(n³), warn for large datasets)
        if len(filtered_vectors) > 10000:
            logger.warning(
                f"Agglomerative clustering on {len(filtered_vectors)} vectors "
                "may be slow and memory-intensive. Consider using K-Means or HDBSCAN."
            )

        # Apply temporal weighting if configured
        if self.config.enable_temporal_weighting and metadata and "dates" in metadata:
            dates = np.array(metadata["dates"])[filter_indices]
            filtered_vectors = self._apply_temporal_weighting(filtered_vectors, dates)

        # Adjust n_clusters if specified and dataset is too small
        actual_n_clusters = self.n_clusters
        if actual_n_clusters is not None:
            actual_n_clusters = min(actual_n_clusters, len(filtered_vectors))
            if actual_n_clusters < self.n_clusters:
                logger.warning(
                    f"Reducing n_clusters from {self.n_clusters} to {actual_n_clusters} "
                    f"due to small dataset size"
                )

        # Create Agglomerative clusterer
        clusterer = AgglomerativeClustering(
            n_clusters=actual_n_clusters,
            distance_threshold=self.distance_threshold,
            linkage=self.linkage,
            affinity=self.affinity,
            compute_full_tree=self.compute_full_tree,
        )

        # Fit and predict
        cluster_labels_filtered = clusterer.fit_predict(filtered_vectors)

        # Map back to original indices
        cluster_labels = np.full(len(vectors), -1, dtype=np.int32)
        cluster_labels[filter_indices] = cluster_labels_filtered

        # Count clusters
        n_clusters = len(set(cluster_labels_filtered))
        outlier_count = len(vectors) - len(filtered_vectors)

        logger.info(f"Agglomerative created {n_clusters} clusters")

        # Calculate centroids
        centroids = None
        if n_clusters > 0:
            unique_labels = set(cluster_labels_filtered)
            centroids = np.zeros((max(unique_labels) + 1, filtered_vectors.shape[1]))
            for label in unique_labels:
                cluster_mask = cluster_labels_filtered == label
                centroids[label] = np.mean(filtered_vectors[cluster_mask], axis=0)

        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            filtered_vectors, cluster_labels_filtered, centroids
        )

        # Add Agglomerative-specific metrics
        if hasattr(clusterer, "n_leaves_"):
            quality_metrics["n_leaves"] = int(clusterer.n_leaves_)

        if hasattr(clusterer, "n_connected_components_"):
            quality_metrics["n_connected_components"] = int(
                clusterer.n_connected_components_
            )

        # Calculate distances to cluster centers for confidence scores
        cluster_probabilities = None
        if centroids is not None:
            distances = np.zeros(len(vectors))
            for i in range(len(vectors)):
                if cluster_labels[i] != -1:
                    distances[i] = np.linalg.norm(
                        vectors[i] - centroids[cluster_labels[i]]
                    )

            # Convert distances to probabilities
            max_dist = np.max(distances[distances > 0])
            if max_dist > 0:
                cluster_probabilities = np.exp(-distances / max_dist)
                cluster_probabilities[cluster_labels == -1] = 0.0

        return ClusteringResult(
            cluster_labels=cluster_labels,
            n_clusters=n_clusters,
            outlier_count=outlier_count,
            quality_metrics=quality_metrics,
            centroids=centroids,
            cluster_probabilities=cluster_probabilities,
        )
