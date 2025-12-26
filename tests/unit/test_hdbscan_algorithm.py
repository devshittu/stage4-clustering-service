"""
Unit tests for HDBSCAN clustering algorithm.

Tests the HDBSCANAlgorithm class including:
- Basic clustering functionality
- Outlier detection
- Parameter handling
- Quality metrics calculation
- Error handling
"""

import pytest
import numpy as np
from src.core.hdbscan_algorithm import HDBSCANAlgorithm
from src.core.base_clustering import ClusteringConfig


@pytest.mark.unit
class TestHDBSCANAlgorithm:
    """Test suite for HDBSCAN clustering algorithm."""

    def test_init(self, clustering_config):
        """Test HDBSCAN algorithm initialization."""
        clusterer = HDBSCANAlgorithm(clustering_config)
        assert clusterer is not None
        assert clusterer.config == clustering_config

    def test_cluster_basic(self, clustered_vectors):
        """Test basic clustering on vectors with clear structure."""
        vectors, true_labels = clustered_vectors

        config = ClusteringConfig(
            algorithm_name="hdbscan",
            params={
                "min_cluster_size": 5,
                "min_samples": 3
            }
        )

        clusterer = HDBSCANAlgorithm(config)
        result = clusterer.cluster(vectors)

        # Should find at least 2 clusters (may find 3 if well-separated)
        assert result.n_clusters >= 2
        assert len(result.labels) == len(vectors)
        assert result.outlier_count >= 0

        # Check that most vectors are clustered (not outliers)
        clustered_ratio = 1.0 - (result.outlier_count / len(vectors))
        assert clustered_ratio > 0.7  # At least 70% clustered

    def test_cluster_with_metadata(self, clustered_vectors, sample_metadata):
        """Test clustering with metadata."""
        vectors, _ = clustered_vectors

        config = ClusteringConfig(
            algorithm_name="hdbscan",
            params={
                "min_cluster_size": 5,
                "min_samples": 3
            }
        )

        clusterer = HDBSCANAlgorithm(config)
        result = clusterer.cluster(vectors, sample_metadata[:len(vectors)])

        assert result.n_clusters >= 0
        assert len(result.labels) == len(vectors)

    def test_outlier_detection(self, sample_vectors):
        """Test that outliers are properly detected."""
        # Add a clear outlier
        outlier = np.array([[10.0] * 768], dtype=np.float32)
        vectors = np.vstack([sample_vectors, outlier])

        config = ClusteringConfig(
            algorithm_name="hdbscan",
            params={
                "min_cluster_size": 5,
                "min_samples": 3
            }
        )

        clusterer = HDBSCANAlgorithm(config)
        result = clusterer.cluster(vectors)

        # Outliers are labeled as -1
        assert -1 in result.labels
        assert result.outlier_count > 0

    def test_membership_probabilities(self, clustered_vectors):
        """Test that membership probabilities are calculated."""
        vectors, _ = clustered_vectors

        config = ClusteringConfig(
            algorithm_name="hdbscan",
            params={
                "min_cluster_size": 5,
                "min_samples": 3
            }
        )

        clusterer = HDBSCANAlgorithm(config)
        result = clusterer.cluster(vectors)

        # Should have probabilities for each vector
        if hasattr(result, 'probabilities') and result.probabilities is not None:
            assert len(result.probabilities) == len(vectors)
            # Probabilities should be between 0 and 1
            assert np.all(result.probabilities >= 0.0)
            assert np.all(result.probabilities <= 1.0)

    def test_quality_metrics(self, clustered_vectors):
        """Test quality metrics calculation."""
        vectors, _ = clustered_vectors

        config = ClusteringConfig(
            algorithm_name="hdbscan",
            params={
                "min_cluster_size": 5,
                "min_samples": 3
            }
        )

        clusterer = HDBSCANAlgorithm(config)
        result = clusterer.cluster(vectors)

        # Should have quality metrics
        assert result.quality_metrics is not None
        assert isinstance(result.quality_metrics, dict)

        # Check for expected metrics
        if result.n_clusters > 1:
            assert "silhouette_score" in result.quality_metrics
            # Silhouette score should be between -1 and 1
            assert -1.0 <= result.quality_metrics["silhouette_score"] <= 1.0

    def test_min_cluster_size_parameter(self, sample_vectors):
        """Test that min_cluster_size parameter is respected."""
        config_small = ClusteringConfig(
            algorithm_name="hdbscan",
            params={
                "min_cluster_size": 3,
                "min_samples": 2
            }
        )

        config_large = ClusteringConfig(
            algorithm_name="hdbscan",
            params={
                "min_cluster_size": 20,
                "min_samples": 5
            }
        )

        clusterer_small = HDBSCANAlgorithm(config_small)
        result_small = clusterer_small.cluster(sample_vectors)

        clusterer_large = HDBSCANAlgorithm(config_large)
        result_large = clusterer_large.cluster(sample_vectors)

        # Smaller min_cluster_size should generally find more clusters
        # (or at least not fewer)
        assert result_small.n_clusters >= result_large.n_clusters

    def test_empty_vectors(self):
        """Test handling of empty vector array."""
        vectors = np.array([], dtype=np.float32).reshape(0, 768)

        config = ClusteringConfig(
            algorithm_name="hdbscan",
            params={"min_cluster_size": 5}
        )

        clusterer = HDBSCANAlgorithm(config)

        with pytest.raises((ValueError, IndexError)):
            clusterer.cluster(vectors)

    def test_single_vector(self):
        """Test handling of single vector."""
        vectors = np.random.randn(1, 768).astype(np.float32)

        config = ClusteringConfig(
            algorithm_name="hdbscan",
            params={"min_cluster_size": 5}
        )

        clusterer = HDBSCANAlgorithm(config)

        # Should either handle gracefully or raise appropriate error
        try:
            result = clusterer.cluster(vectors)
            # If it succeeds, should have one label
            assert len(result.labels) == 1
        except ValueError:
            # Acceptable to raise ValueError for too few vectors
            pass

    def test_deterministic_clustering(self, clustered_vectors):
        """Test that clustering is deterministic (same input -> same output)."""
        vectors, _ = clustered_vectors

        config = ClusteringConfig(
            algorithm_name="hdbscan",
            params={
                "min_cluster_size": 5,
                "min_samples": 3
            }
        )

        clusterer1 = HDBSCANAlgorithm(config)
        result1 = clusterer1.cluster(vectors.copy())

        clusterer2 = HDBSCANAlgorithm(config)
        result2 = clusterer2.cluster(vectors.copy())

        # Results should be identical
        assert result1.n_clusters == result2.n_clusters
        np.testing.assert_array_equal(result1.labels, result2.labels)

    def test_different_dimensions(self):
        """Test error handling for vectors with wrong dimensions."""
        # Create vectors with wrong dimension
        vectors = np.random.randn(100, 512).astype(np.float32)

        config = ClusteringConfig(
            algorithm_name="hdbscan",
            params={"min_cluster_size": 5}
        )

        clusterer = HDBSCANAlgorithm(config)

        # Should work with any dimension (HDBSCAN is dimension-agnostic)
        result = clusterer.cluster(vectors)
        assert len(result.labels) == len(vectors)

    @pytest.mark.slow
    def test_large_dataset(self):
        """Test clustering on larger dataset."""
        # Create 1000 vectors
        np.random.seed(42)
        vectors = np.random.randn(1000, 768).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        config = ClusteringConfig(
            algorithm_name="hdbscan",
            params={
                "min_cluster_size": 10,
                "min_samples": 5
            }
        )

        clusterer = HDBSCANAlgorithm(config)
        result = clusterer.cluster(vectors)

        # Should complete without errors
        assert result.n_clusters >= 0
        assert len(result.labels) == len(vectors)

    def test_cluster_selection_methods(self, clustered_vectors):
        """Test different cluster selection methods."""
        vectors, _ = clustered_vectors

        for method in ["eom", "leaf"]:
            config = ClusteringConfig(
                algorithm_name="hdbscan",
                params={
                    "min_cluster_size": 5,
                    "min_samples": 3,
                    "cluster_selection_method": method
                }
            )

            clusterer = HDBSCANAlgorithm(config)
            result = clusterer.cluster(vectors)

            assert result.n_clusters >= 0
            assert len(result.labels) == len(vectors)
