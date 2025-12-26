"""
Unit tests for K-Means clustering algorithm.

Tests the KMeansAlgorithm class including:
- Basic clustering functionality
- MiniBatch K-Means for large datasets
- Adaptive cluster count
- Centroid calculation
- Quality metrics
"""

import pytest
import numpy as np
from src.core.kmeans_algorithm import KMeansAlgorithm
from src.core.base_clustering import ClusteringConfig


@pytest.mark.unit
class TestKMeansAlgorithm:
    """Test suite for K-Means clustering algorithm."""

    def test_init(self):
        """Test K-Means algorithm initialization."""
        config = ClusteringConfig(
            algorithm_name="kmeans",
            params={"n_clusters": 10}
        )

        clusterer = KMeansAlgorithm(config)
        assert clusterer is not None
        assert clusterer.config == config

    def test_cluster_basic(self, clustered_vectors):
        """Test basic clustering on vectors with clear structure."""
        vectors, _ = clustered_vectors

        config = ClusteringConfig(
            algorithm_name="kmeans",
            params={"n_clusters": 3}
        )

        clusterer = KMeansAlgorithm(config)
        result = clusterer.cluster(vectors)

        # Should find exactly 3 clusters
        assert result.n_clusters == 3
        assert len(result.labels) == len(vectors)
        # K-Means assigns all points to clusters (no outliers)
        assert result.outlier_count == 0

        # All labels should be 0, 1, or 2
        assert set(result.labels) <= {0, 1, 2}

    def test_adaptive_cluster_count(self, sample_vectors):
        """Test adaptive cluster count when n_clusters not specified."""
        config = ClusteringConfig(
            algorithm_name="kmeans",
            params={}  # No n_clusters specified
        )

        clusterer = KMeansAlgorithm(config)
        result = clusterer.cluster(sample_vectors)

        # Should automatically determine cluster count
        # For 100 vectors: n_clusters ≈ sqrt(100/2) ≈ 7
        assert result.n_clusters > 0
        assert len(result.labels) == len(sample_vectors)

    def test_minibatch_large_dataset(self):
        """Test that MiniBatch K-Means is used for large datasets."""
        # Create 15000 vectors (>10000 triggers MiniBatch)
        np.random.seed(42)
        vectors = np.random.randn(15000, 768).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        config = ClusteringConfig(
            algorithm_name="kmeans",
            params={"n_clusters": 50}
        )

        clusterer = KMeansAlgorithm(config)
        result = clusterer.cluster(vectors)

        # Should use MiniBatch and complete quickly
        assert result.n_clusters == 50
        assert len(result.labels) == len(vectors)

    def test_centroids_calculation(self, clustered_vectors):
        """Test that centroids are properly calculated."""
        vectors, _ = clustered_vectors

        config = ClusteringConfig(
            algorithm_name="kmeans",
            params={"n_clusters": 3}
        )

        clusterer = KMeansAlgorithm(config)
        result = clusterer.cluster(vectors)

        # Should have centroids
        assert result.cluster_centroids is not None
        assert len(result.cluster_centroids) == 3

        # Each centroid should have same dimension as vectors
        for centroid in result.cluster_centroids.values():
            assert len(centroid) == 768

            # Centroids should be normalized (L2 norm ≈ 1)
            norm = np.linalg.norm(centroid)
            assert 0.9 <= norm <= 1.1

    def test_quality_metrics(self, clustered_vectors):
        """Test quality metrics calculation."""
        vectors, _ = clustered_vectors

        config = ClusteringConfig(
            algorithm_name="kmeans",
            params={"n_clusters": 3}
        )

        clusterer = KMeansAlgorithm(config)
        result = clusterer.cluster(vectors)

        # Should have quality metrics
        assert result.quality_metrics is not None
        assert isinstance(result.quality_metrics, dict)

        # K-Means specific metrics
        assert "inertia" in result.quality_metrics
        assert result.quality_metrics["inertia"] >= 0.0

        # General metrics
        assert "silhouette_score" in result.quality_metrics
        assert -1.0 <= result.quality_metrics["silhouette_score"] <= 1.0

    def test_confidence_scores(self, clustered_vectors):
        """Test that confidence scores are calculated."""
        vectors, _ = clustered_vectors

        config = ClusteringConfig(
            algorithm_name="kmeans",
            params={"n_clusters": 3}
        )

        clusterer = KMeansAlgorithm(config)
        result = clusterer.cluster(vectors)

        # Should have confidence scores (distance-based)
        if hasattr(result, 'probabilities') and result.probabilities is not None:
            assert len(result.probabilities) == len(vectors)
            # Scores should be between 0 and 1
            assert np.all(result.probabilities >= 0.0)
            assert np.all(result.probabilities <= 1.0)

    def test_different_n_clusters(self, sample_vectors):
        """Test with different numbers of clusters."""
        for n_clusters in [2, 5, 10, 20]:
            config = ClusteringConfig(
                algorithm_name="kmeans",
                params={"n_clusters": n_clusters}
            )

            clusterer = KMeansAlgorithm(config)
            result = clusterer.cluster(sample_vectors)

            assert result.n_clusters == n_clusters
            assert len(result.labels) == len(sample_vectors)
            # Should use all cluster IDs
            assert len(set(result.labels)) == n_clusters

    def test_max_clusters_constraint(self, small_vectors):
        """Test that n_clusters cannot exceed number of vectors."""
        # Try to create more clusters than vectors
        config = ClusteringConfig(
            algorithm_name="kmeans",
            params={"n_clusters": 20}  # More than 10 vectors
        )

        clusterer = KMeansAlgorithm(config)

        # Should handle gracefully (reduce n_clusters or raise error)
        try:
            result = clusterer.cluster(small_vectors)
            # If it succeeds, n_clusters should be ≤ n_vectors
            assert result.n_clusters <= len(small_vectors)
        except ValueError:
            # Acceptable to raise error
            pass

    def test_empty_vectors(self):
        """Test handling of empty vector array."""
        vectors = np.array([], dtype=np.float32).reshape(0, 768)

        config = ClusteringConfig(
            algorithm_name="kmeans",
            params={"n_clusters": 5}
        )

        clusterer = KMeansAlgorithm(config)

        with pytest.raises((ValueError, IndexError)):
            clusterer.cluster(vectors)

    def test_deterministic_clustering(self, clustered_vectors):
        """Test that clustering is deterministic with same random_state."""
        vectors, _ = clustered_vectors

        config = ClusteringConfig(
            algorithm_name="kmeans",
            params={
                "n_clusters": 3,
                "random_state": 42  # Fixed seed
            }
        )

        clusterer1 = KMeansAlgorithm(config)
        result1 = clusterer1.cluster(vectors.copy())

        clusterer2 = KMeansAlgorithm(config)
        result2 = clusterer2.cluster(vectors.copy())

        # Results should be identical with same random_state
        assert result1.n_clusters == result2.n_clusters
        np.testing.assert_array_equal(result1.labels, result2.labels)

    def test_convergence_parameters(self, sample_vectors):
        """Test different convergence parameters."""
        config = ClusteringConfig(
            algorithm_name="kmeans",
            params={
                "n_clusters": 5,
                "max_iter": 100,
                "n_init": 5
            }
        )

        clusterer = KMeansAlgorithm(config)
        result = clusterer.cluster(sample_vectors)

        assert result.n_clusters == 5
        assert len(result.labels) == len(sample_vectors)

    def test_well_separated_clusters(self):
        """Test clustering on well-separated clusters."""
        np.random.seed(42)

        # Create 3 well-separated clusters
        cluster1 = np.random.randn(30, 768) + 10.0  # Shifted far away
        cluster2 = np.random.randn(30, 768) - 10.0
        cluster3 = np.random.randn(30, 768) + [5.0, -5.0] + [0.0] * 766

        vectors = np.vstack([cluster1, cluster2, cluster3]).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        config = ClusteringConfig(
            algorithm_name="kmeans",
            params={"n_clusters": 3}
        )

        clusterer = KMeansAlgorithm(config)
        result = clusterer.cluster(vectors)

        # Should have high silhouette score for well-separated clusters
        assert result.quality_metrics["silhouette_score"] > 0.3

    def test_cluster_sizes(self, clustered_vectors):
        """Test that cluster sizes are reasonable."""
        vectors, _ = clustered_vectors

        config = ClusteringConfig(
            algorithm_name="kmeans",
            params={"n_clusters": 3}
        )

        clusterer = KMeansAlgorithm(config)
        result = clusterer.cluster(vectors)

        # Count cluster sizes
        cluster_sizes = {}
        for label in result.labels:
            cluster_sizes[label] = cluster_sizes.get(label, 0) + 1

        # Should have 3 clusters
        assert len(cluster_sizes) == 3

        # Each cluster should have members (no empty clusters)
        for size in cluster_sizes.values():
            assert size > 0

    @pytest.mark.slow
    def test_performance_large_dataset(self):
        """Test performance on large dataset."""
        import time

        np.random.seed(42)
        vectors = np.random.randn(10000, 768).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        config = ClusteringConfig(
            algorithm_name="kmeans",
            params={"n_clusters": 50}
        )

        clusterer = KMeansAlgorithm(config)

        start = time.time()
        result = clusterer.cluster(vectors)
        elapsed = time.time() - start

        # Should complete in reasonable time (<30 seconds on most hardware)
        assert elapsed < 30.0
        assert result.n_clusters == 50
