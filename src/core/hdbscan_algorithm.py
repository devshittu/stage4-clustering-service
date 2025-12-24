"""
HDBSCAN Clustering Algorithm Implementation.

Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN)
is ideal for:
- Finding clusters of varying densities
- Handling outliers/noise
- Not requiring the number of clusters as input
- Event clustering and entity disambiguation
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
import hdbscan

from src.core.base_clustering import (
    BaseClusteringAlgorithm,
    ClusteringResult,
    ClusteringConfig,
)

logger = logging.getLogger(__name__)


class HDBSCANAlgorithm(BaseClusteringAlgorithm):
    """
    HDBSCAN clustering implementation.

    Best for: Event clustering, entity disambiguation, storyline grouping
    Strengths: Handles variable density, automatic outlier detection
    Weaknesses: Slower than K-Means, doesn't scale to millions of vectors
    """

    def __init__(self, config: ClusteringConfig):
        """
        Initialize HDBSCAN algorithm.

        Args:
            config: Clustering configuration
        """
        super().__init__(config)

        # Extract HDBSCAN-specific parameters
        self.min_cluster_size = config.params.get("min_cluster_size", 5)
        self.min_samples = config.params.get("min_samples", 3)
        self.cluster_selection_epsilon = config.params.get("cluster_selection_epsilon", 0.0)
        self.metric = config.params.get("metric", "euclidean")
        self.cluster_selection_method = config.params.get("cluster_selection_method", "eom")
        self.allow_single_cluster = config.params.get("allow_single_cluster", False)
        self.alpha = config.params.get("alpha", 1.0)

        logger.info(
            f"Initialized HDBSCAN: min_cluster_size={self.min_cluster_size}, "
            f"min_samples={self.min_samples}, metric={self.metric}"
        )

    def cluster(
        self,
        vectors: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ClusteringResult:
        """
        Perform HDBSCAN clustering.

        Args:
            vectors: Embedding vectors (N x D)
            metadata: Optional metadata for filtering and temporal weighting

        Returns:
            ClusteringResult with labels and metrics
        """
        logger.info(f"Starting HDBSCAN clustering on {len(vectors)} vectors")

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

        # Apply temporal weighting if configured
        if self.config.enable_temporal_weighting and metadata and "dates" in metadata:
            dates = np.array(metadata["dates"])[filter_indices]
            filtered_vectors = self._apply_temporal_weighting(filtered_vectors, dates)

        # Create HDBSCAN clusterer
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
            allow_single_cluster=self.allow_single_cluster,
            alpha=self.alpha,
            core_dist_n_jobs=-1,  # Use all CPU cores
        )

        # Fit and predict
        cluster_labels_filtered = clusterer.fit_predict(filtered_vectors)

        # Map back to original indices
        cluster_labels = np.full(len(vectors), -1, dtype=np.int32)
        cluster_labels[filter_indices] = cluster_labels_filtered

        # Count clusters (excluding outliers labeled as -1)
        n_clusters = len(set(cluster_labels_filtered)) - (1 if -1 in cluster_labels_filtered else 0)
        outlier_count = np.sum(cluster_labels == -1)

        logger.info(f"HDBSCAN found {n_clusters} clusters with {outlier_count} outliers")

        # Calculate centroids
        centroids = None
        if n_clusters > 0:
            unique_labels = set(cluster_labels_filtered)
            if -1 in unique_labels:
                unique_labels.remove(-1)

            centroids = np.zeros((max(unique_labels) + 1, filtered_vectors.shape[1]))
            for label in unique_labels:
                cluster_mask = cluster_labels_filtered == label
                centroids[label] = np.mean(filtered_vectors[cluster_mask], axis=0)

        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            filtered_vectors, cluster_labels_filtered, centroids
        )

        # Add HDBSCAN-specific metrics
        if hasattr(clusterer, "probabilities_"):
            quality_metrics["avg_membership_probability"] = float(
                np.mean(clusterer.probabilities_[clusterer.probabilities_ > 0])
            )

        if hasattr(clusterer, "outlier_scores_"):
            quality_metrics["avg_outlier_score"] = float(
                np.mean(clusterer.outlier_scores_)
            )

        # Extract cluster probabilities
        cluster_probabilities = None
        if hasattr(clusterer, "probabilities_"):
            cluster_probabilities = np.zeros(len(vectors))
            cluster_probabilities[filter_indices] = clusterer.probabilities_

        return ClusteringResult(
            cluster_labels=cluster_labels,
            n_clusters=n_clusters,
            outlier_count=outlier_count,
            quality_metrics=quality_metrics,
            centroids=centroids,
            cluster_probabilities=cluster_probabilities,
        )
