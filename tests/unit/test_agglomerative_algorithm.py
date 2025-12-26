"""
Unit tests for Agglomerative clustering algorithm.

Tests the AgglomerativeAlgorithm class including:
- Hierarchical clustering functionality
- Different linkage methods
- Distance threshold clustering
- Dendrogram support (future)
"""

import pytest
import numpy as np
from src.core.agglomerative_algorithm import AgglomerativeAlgorithm
from src.core.base_clustering import ClusteringConfig


@pytest.mark.unit
class TestAgglomerativeAlgorithm:
    """Test suite for Agglomerative clustering algorithm."""

    def test_init(self):
        """Test Agglomerative algorithm initialization."""
        config = ClusteringConfig(
            algorithm_name="agglomerative",
            params={"n_clusters": 5}
        )

        clusterer = AgglomerativeAlgorithm(config)
        assert clusterer is not None
        assert clusterer.config == config

    def test_cluster_with_n_clusters(self, clustered_vectors):
        """Test clustering with specified n_clusters."""
        vectors, _ = clustered_vectors

        config = ClusteringConfig(
            algorithm_name="agglomerative",
            params={"n_clusters": 3}
        )

        clusterer = AgglomerativeAlgorithm(config)
        result = clusterer.cluster(vectors)

        # Should find exactly 3 clusters
        assert result.n_clusters == 3
        assert len(result.labels) == len(vectors)
        # Agglomerative assigns all points to clusters
        assert result.outlier_count == 0

    def test_cluster_with_distance_threshold(self, clustered_vectors):
        """Test clustering with distance threshold."""
        vectors, _ = clustered_vectors

        config = ClusteringConfig(
            algorithm_name="agglomerative",
            params={
                "n_clusters": None,
                "distance_threshold": 0.5
            }
        )

        clusterer = AgglomerativeAlgorithm(config)
        result = clusterer.cluster(vectors)

        # Should find some clusters (exact number depends on threshold)
        assert result.n_clusters > 0
        assert len(result.labels) == len(vectors)

    def test_different_linkage_methods(self, sample_vectors):
        """Test different linkage methods."""
        for linkage in ["ward", "complete", "average", "single"]:
            config = ClusteringConfig(
                algorithm_name="agglomerative",
                params={
                    "n_clusters": 5,
                    "linkage": linkage
                }
            )

            clusterer = AgglomerativeAlgorithm(config)
            result = clusterer.cluster(sample_vectors)

            assert result.n_clusters == 5
            assert len(result.labels) == len(sample_vectors)

    def test_quality_metrics(self, clustered_vectors):
        """Test quality metrics calculation."""
        vectors, _ = clustered_vectors

        config = ClusteringConfig(
            algorithm_name="agglomerative",
            params={"n_clusters": 3}
        )

        clusterer = AgglomerativeAlgorithm(config)
        result = clusterer.cluster(vectors)

        # Should have quality metrics
        assert result.quality_metrics is not None
        assert isinstance(result.quality_metrics, dict)

        # Silhouette score for multi-cluster result
        if result.n_clusters > 1:
            assert "silhouette_score" in result.quality_metrics
            assert -1.0 <= result.quality_metrics["silhouette_score"] <= 1.0

    def test_entity_coreference_use_case(self):
        """Test agglomerative for entity coreference (its primary use case)."""
        np.random.seed(42)

        # Simulate entity mentions with high similarity within groups
        # Group 1: "Biden", "President Biden", "Joe Biden"
        biden_base = np.random.randn(768)
        biden_mentions = [biden_base + np.random.randn(768) * 0.05 for _ in range(3)]

        # Group 2: "Trump", "Donald Trump", "Former President Trump"
        trump_base = np.random.randn(768) + 5.0  # Different from Biden
        trump_mentions = [trump_base + np.random.randn(768) * 0.05 for _ in range(3)]

        vectors = np.vstack(biden_mentions + trump_mentions).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        config = ClusteringConfig(
            algorithm_name="agglomerative",
            params={
                "distance_threshold": 0.3,
                "n_clusters": None,
                "linkage": "average"
            }
        )

        clusterer = AgglomerativeAlgorithm(config)
        result = clusterer.cluster(vectors)

        # Should ideally find 2 clusters (Biden group and Trump group)
        # May find more due to noise, but should be small number
        assert 2 <= result.n_clusters <= 4

    def test_warning_for_large_dataset(self, caplog):
        """Test that warning is logged for large datasets."""
        # Create a dataset larger than recommended (>5000)
        np.random.seed(42)
        vectors = np.random.randn(6000, 768).astype(np.float32)

        config = ClusteringConfig(
            algorithm_name="agglomerative",
            params={"n_clusters": 10}
        )

        clusterer = AgglomerativeAlgorithm(config)

        # Should log warning about performance
        with caplog.at_level("WARNING"):
            result = clusterer.cluster(vectors)

        # Check if warning was logged
        assert any("large dataset" in record.message.lower() for record in caplog.records)

    def test_small_dataset(self, small_vectors):
        """Test on small dataset."""
        config = ClusteringConfig(
            algorithm_name="agglomerative",
            params={"n_clusters": 3}
        )

        clusterer = AgglomerativeAlgorithm(config)
        result = clusterer.cluster(small_vectors)

        assert result.n_clusters == 3
        assert len(result.labels) == len(small_vectors)

    def test_empty_vectors(self):
        """Test handling of empty vector array."""
        vectors = np.array([], dtype=np.float32).reshape(0, 768)

        config = ClusteringConfig(
            algorithm_name="agglomerative",
            params={"n_clusters": 5}
        )

        clusterer = AgglomerativeAlgorithm(config)

        with pytest.raises((ValueError, IndexError)):
            clusterer.cluster(vectors)

    def test_single_cluster(self, sample_vectors):
        """Test creating single cluster."""
        config = ClusteringConfig(
            algorithm_name="agglomerative",
            params={"n_clusters": 1}
        )

        clusterer = AgglomerativeAlgorithm(config)
        result = clusterer.cluster(sample_vectors)

        assert result.n_clusters == 1
        # All vectors should have same label
        assert len(set(result.labels)) == 1

    def test_deterministic_clustering(self, clustered_vectors):
        """Test that clustering is deterministic."""
        vectors, _ = clustered_vectors

        config = ClusteringConfig(
            algorithm_name="agglomerative",
            params={"n_clusters": 3, "linkage": "ward"}
        )

        clusterer1 = AgglomerativeAlgorithm(config)
        result1 = clusterer1.cluster(vectors.copy())

        clusterer2 = AgglomerativeAlgorithm(config)
        result2 = clusterer2.cluster(vectors.copy())

        # Results should be identical (agglomerative is deterministic)
        assert result1.n_clusters == result2.n_clusters
        np.testing.assert_array_equal(result1.labels, result2.labels)

    def test_affinity_parameter(self, sample_vectors):
        """Test different affinity (distance) metrics."""
        for affinity in ["euclidean", "l1", "l2", "manhattan", "cosine"]:
            # Skip unsupported affinity/linkage combinations
            if affinity != "euclidean" and "ward" in str(affinity):
                continue

            config = ClusteringConfig(
                algorithm_name="agglomerative",
                params={
                    "n_clusters": 5,
                    "linkage": "average",  # Compatible with all affinities
                    "affinity": affinity
                }
            )

            clusterer = AgglomerativeAlgorithm(config)
            result = clusterer.cluster(sample_vectors)

            assert result.n_clusters == 5
            assert len(result.labels) == len(sample_vectors)

    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        # Both n_clusters and distance_threshold as None
        config = ClusteringConfig(
            algorithm_name="agglomerative",
            params={
                "n_clusters": None,
                "distance_threshold": None
            }
        )

        clusterer = AgglomerativeAlgorithm(config)

        # Should raise ValueError
        with pytest.raises(ValueError):
            clusterer.cluster(np.random.randn(100, 768).astype(np.float32))

    @pytest.mark.slow
    def test_performance_medium_dataset(self):
        """Test performance on medium dataset (agglomerative is O(n^2) or O(n^3))."""
        import time

        np.random.seed(42)
        # Use 1000 vectors (agglomerative is slow on large datasets)
        vectors = np.random.randn(1000, 768).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        config = ClusteringConfig(
            algorithm_name="agglomerative",
            params={"n_clusters": 10}
        )

        clusterer = AgglomerativeAlgorithm(config)

        start = time.time()
        result = clusterer.cluster(vectors)
        elapsed = time.time() - start

        # Should complete in reasonable time (<60 seconds)
        assert elapsed < 60.0
        assert result.n_clusters == 10
