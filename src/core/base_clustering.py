"""
Base Clustering Algorithm Interface.

Defines the contract for all clustering algorithms in Stage 4.
Supports pluggable algorithms with consistent API.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class ClusteringConfig:
    """Configuration for clustering algorithms."""

    algorithm_name: str
    params: Dict[str, Any]
    enable_temporal_weighting: bool = False
    temporal_decay_factor: float = 7.0  # days
    metadata_filters: Optional[Dict[str, Any]] = None


class ClusteringResult:
    """Results from clustering operation."""

    def __init__(
        self,
        cluster_labels: np.ndarray,
        n_clusters: int,
        outlier_count: int,
        quality_metrics: Dict[str, float],
        centroids: Optional[np.ndarray] = None,
        cluster_probabilities: Optional[np.ndarray] = None,
    ):
        self.cluster_labels = cluster_labels
        self.n_clusters = n_clusters
        self.outlier_count = outlier_count
        self.quality_metrics = quality_metrics
        self.centroids = centroids
        self.cluster_probabilities = cluster_probabilities

    @property
    def labels(self) -> np.ndarray:
        """Alias for cluster_labels for backward compatibility."""
        return self.cluster_labels

    @property
    def probabilities(self) -> Optional[np.ndarray]:
        """Alias for cluster_probabilities for backward compatibility."""
        return self.cluster_probabilities

    @property
    def cluster_centroids(self) -> Optional[Dict[int, np.ndarray]]:
        """Return centroids as dict mapping cluster_id -> centroid_vector."""
        if self.centroids is None:
            return None
        # Convert array of centroids to dict
        return {i: self.centroids[i] for i in range(len(self.centroids))}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_clusters": self.n_clusters,
            "outlier_count": self.outlier_count,
            "quality_metrics": self.quality_metrics,
            "total_items": len(self.cluster_labels),
        }


class BaseClusteringAlgorithm(ABC):
    """
    Abstract base class for clustering algorithms.

    All clustering algorithms (HDBSCAN, K-Means, Agglomerative) must inherit
    from this class and implement the cluster() method.
    """

    def __init__(self, config: ClusteringConfig):
        """
        Initialize clustering algorithm.

        Args:
            config: Clustering configuration
        """
        self.config = config
        self.name = config.algorithm_name

    @abstractmethod
    def cluster(
        self,
        vectors: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ClusteringResult:
        """
        Perform clustering on vectors.

        Args:
            vectors: Embedding vectors (N x D)
            metadata: Optional metadata for each vector

        Returns:
            ClusteringResult with labels and metrics
        """
        pass

    def _apply_metadata_filters(
        self,
        vectors: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter vectors based on metadata criteria.

        Args:
            vectors: Input vectors (N x D)
            metadata: Metadata dict with list of per-vector metadata

        Returns:
            Tuple of (filtered_vectors, filter_mask)
        """
        if not self.config.metadata_filters or not metadata:
            return vectors, np.arange(len(vectors))

        filters = self.config.metadata_filters
        mask = np.ones(len(vectors), dtype=bool)

        # Apply domain filter
        if "domain" in filters and "domain" in metadata:
            target_domain = filters["domain"]
            domains = metadata.get("domain", [])
            if isinstance(domains, list) and len(domains) == len(vectors):
                domain_mask = np.array([d == target_domain for d in domains])
                mask &= domain_mask

        # Apply event_type filter
        if "event_type" in filters and "event_type" in metadata:
            target_type = filters["event_type"]
            event_types = metadata.get("event_type", [])
            if isinstance(event_types, list) and len(event_types) == len(vectors):
                type_mask = np.array([et == target_type for et in event_types])
                mask &= type_mask

        # Apply temporal window filter
        if "temporal_window" in filters and "dates" in metadata:
            start_date = filters["temporal_window"].get("start")
            end_date = filters["temporal_window"].get("end")
            dates = metadata.get("dates", [])
            if start_date and end_date and dates:
                temporal_mask = np.array(
                    [start_date <= d <= end_date for d in dates]
                )
                mask &= temporal_mask

        indices = np.where(mask)[0]
        return vectors[mask], indices

    def _apply_temporal_weighting(
        self,
        vectors: np.ndarray,
        dates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply temporal decay weighting to vectors.

        Args:
            vectors: Input vectors (N x D)
            dates: Timestamps for each vector (Unix timestamps or datetime)

        Returns:
            Weighted vectors
        """
        if not self.config.enable_temporal_weighting or dates is None:
            return vectors

        # Check if dates array contains valid (non-None) values
        valid_dates_mask = np.array([d is not None for d in dates])
        if not valid_dates_mask.any():
            # All dates are None, skip temporal weighting
            return vectors

        # Calculate temporal weights using exponential decay
        # weight = exp(-|date - reference_date| / decay_factor)
        reference_date = np.max(dates[valid_dates_mask])  # Use most recent valid date as reference
        time_diffs = np.abs(dates - reference_date)  # Days difference

        # Exponential decay
        decay_factor = self.config.temporal_decay_factor
        weights = np.exp(-time_diffs / decay_factor)

        # Reshape weights for broadcasting (N x 1)
        weights = weights.reshape(-1, 1)

        # Apply weights to vectors
        weighted_vectors = vectors * weights

        return weighted_vectors

    def _calculate_quality_metrics(
        self,
        vectors: np.ndarray,
        labels: np.ndarray,
        centroids: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calculate clustering quality metrics.

        Args:
            vectors: Input vectors
            labels: Cluster labels
            centroids: Optional cluster centroids

        Returns:
            Dictionary of quality metrics
        """
        from sklearn.metrics import silhouette_score, davies_bouldin_score

        metrics = {}

        # Filter out outliers (-1 labels) for metrics calculation
        non_outlier_mask = labels != -1

        if np.sum(non_outlier_mask) > 1 and len(np.unique(labels[non_outlier_mask])) > 1:
            try:
                # Silhouette score (higher is better, range: -1 to 1)
                silhouette = silhouette_score(
                    vectors[non_outlier_mask],
                    labels[non_outlier_mask],
                )
                metrics["silhouette_score"] = float(silhouette)
            except Exception:
                metrics["silhouette_score"] = 0.0

            try:
                # Davies-Bouldin Index (lower is better)
                db_index = davies_bouldin_score(
                    vectors[non_outlier_mask],
                    labels[non_outlier_mask],
                )
                metrics["davies_bouldin_index"] = float(db_index)
            except Exception:
                metrics["davies_bouldin_index"] = 0.0

        # Calculate intra-cluster similarity
        if centroids is not None:
            intra_similarities = []
            for cluster_id in np.unique(labels):
                if cluster_id == -1:
                    continue
                cluster_vectors = vectors[labels == cluster_id]
                if len(cluster_vectors) > 0:
                    # Cosine similarity with centroid
                    centroid = centroids[cluster_id]
                    similarities = np.dot(cluster_vectors, centroid) / (
                        np.linalg.norm(cluster_vectors, axis=1)
                        * np.linalg.norm(centroid)
                    )
                    intra_similarities.append(np.mean(similarities))

            if intra_similarities:
                metrics["avg_intra_cluster_similarity"] = float(
                    np.mean(intra_similarities)
                )

        return metrics
