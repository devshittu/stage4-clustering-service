"""
K-Means Clustering Algorithm Implementation.

K-Means is ideal for:
- Fast clustering of large datasets
- When number of clusters is known or can be estimated
- Spherical, evenly-sized clusters
- Document clustering and topic modeling
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from src.core.base_clustering import (
    BaseClusteringAlgorithm,
    ClusteringResult,
    ClusteringConfig,
)

logger = logging.getLogger(__name__)


class KMeansAlgorithm(BaseClusteringAlgorithm):
    """
    K-Means clustering implementation.

    Best for: Document clustering, fast pre-clustering for large datasets
    Strengths: Fast, scalable, simple
    Weaknesses: Requires k as input, assumes spherical clusters, sensitive to outliers
    """

    def __init__(self, config: ClusteringConfig):
        """
        Initialize K-Means algorithm.

        Args:
            config: Clustering configuration
        """
        super().__init__(config)

        # Extract K-Means-specific parameters
        self.n_clusters = config.params.get("n_clusters", 50)
        self.n_init = config.params.get("n_init", 10)
        self.max_iter = config.params.get("max_iter", 300)
        self.algorithm = config.params.get("algorithm", "lloyd")  # Changed from "auto" (deprecated in sklearn 1.3+)
        self.random_state = config.params.get("random_state", 42)
        self.use_minibatch = config.params.get("use_minibatch", False)
        self.batch_size = config.params.get("batch_size", 1024)
        self.tol = config.params.get("tol", 1e-4)

        logger.info(
            f"Initialized K-Means: n_clusters={self.n_clusters}, "
            f"n_init={self.n_init}, use_minibatch={self.use_minibatch}"
        )

    def cluster(
        self,
        vectors: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ClusteringResult:
        """
        Perform K-Means clustering.

        Args:
            vectors: Embedding vectors (N x D)
            metadata: Optional metadata for filtering and temporal weighting

        Returns:
            ClusteringResult with labels and metrics
        """
        logger.info(f"Starting K-Means clustering on {len(vectors)} vectors")

        # Validate input
        if len(vectors) == 0:
            raise ValueError("Cannot cluster empty vector array")

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

        # Adjust n_clusters if fewer vectors than requested clusters
        actual_n_clusters = min(self.n_clusters, len(filtered_vectors))

        if actual_n_clusters < self.n_clusters:
            logger.warning(
                f"Reducing n_clusters from {self.n_clusters} to {actual_n_clusters} "
                f"due to small dataset size"
            )

        # Create K-Means clusterer
        if self.use_minibatch and len(filtered_vectors) > self.batch_size * 10:
            # Use MiniBatchKMeans for large datasets
            clusterer = MiniBatchKMeans(
                n_clusters=actual_n_clusters,
                batch_size=self.batch_size,
                max_iter=self.max_iter,
                random_state=self.random_state,
                n_init=self.n_init,
                tol=self.tol,
                reassignment_ratio=0.01,
            )
            logger.info(f"Using MiniBatchKMeans with batch_size={self.batch_size}")
        else:
            # Use standard K-Means
            clusterer = KMeans(
                n_clusters=actual_n_clusters,
                n_init=self.n_init,
                max_iter=self.max_iter,
                algorithm=self.algorithm,
                random_state=self.random_state,
                tol=self.tol,
            )

        # Fit and predict
        cluster_labels_filtered = clusterer.fit_predict(filtered_vectors)

        # Map back to original indices
        cluster_labels = np.full(len(vectors), -1, dtype=np.int32)
        cluster_labels[filter_indices] = cluster_labels_filtered

        # K-Means doesn't have outliers, but we mark filtered-out items as outliers
        n_clusters = actual_n_clusters
        outlier_count = len(vectors) - len(filtered_vectors)

        logger.info(f"K-Means created {n_clusters} clusters")

        # Get centroids
        centroids = clusterer.cluster_centers_

        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            filtered_vectors, cluster_labels_filtered, centroids
        )

        # Add K-Means-specific metrics
        if hasattr(clusterer, "inertia_"):
            quality_metrics["inertia"] = float(clusterer.inertia_)

        if hasattr(clusterer, "n_iter_"):
            quality_metrics["iterations"] = int(clusterer.n_iter_)

        # Calculate distances to cluster centers for confidence scores
        cluster_probabilities = None
        if centroids is not None:
            # Calculate distance of each point to its assigned centroid
            distances = np.zeros(len(vectors))
            for i in range(len(vectors)):
                if cluster_labels[i] != -1:
                    distances[i] = np.linalg.norm(
                        vectors[i] - centroids[cluster_labels[i]]
                    )

            # Convert distances to probabilities (inverse exponential)
            # Higher distance = lower probability
            positive_distances = distances[distances > 0]
            if len(positive_distances) > 0:
                max_dist = np.max(positive_distances)
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
